"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

SiLU-gated RMSNorm on the vendored Cake kernels (SM100a / SM103a / SM107a).

:func:`rmsnorm_silu_gate` is the output norm of miles' GDN layers: per row of 128 of BF16 ``x`` and of the gate ``z``
(``[T, H, 128]`` or ``[T * H, 128]``, each with its own token and head strides) and a BF16 ``weight`` ``[128]``
shared by every token and head::

    rstd = 1 / sqrt(mean(x^2) + eps)                     # FP32
    y    = bf16(((x * rstd) * weight) * silu(z))         # FP32, one rounding

``det`` runs FlashInfer's deterministic package (``_vendor/cake_rmsnorm_silu_gate_det_train``): one forward and one
backward kernel. The backward adds each chunk's weight-gradient terms, FP32 runs folded into two-sum pairs, exactly
into fixed-point digits with integer atomics and rounds the sum once, so the weight gradient is the same bits for the
same rows on a device of the same SM count; from 2560 rows per SM on sm_100a and sm_103a and 1536 on sm_107a, packed
rows run a look-ahead program of the same bits. ``fast`` runs the non-deterministic package
(``_vendor/cake_rmsnorm_silu_gate_nondet_train``), which picks a program by the rows per SM and the layout
(``cake_jit.module_for_rows``): its ``y``, ``dx`` and ``dz`` repeat bit for bit and stay within one BF16 ulp of FP64,
and they are ``det``'s bits where its program is ``det``'s (the pipelined forwards' ``y``, the fused and the
repeatable backwards' ``dx`` and ``dz``). Its weight gradient is a fixed-order on-chip sum up to 256 rows (512 on
sm_100a and sm_107a) and ``det``'s exact sum past the fused band; in the fused band, up to 1000 rows per SM on sm_100a
and sm_103a and from 64 to 256 on sm_107a, its CTAs add FP32 two-sum runs to FP64 accumulators with atomics
(flashinfer-ai/flashinfer#9182). The order of those additions depends on scheduling, so there the weight gradient may
differ between runs, within ``det``'s bound ``|dw| * 2^-7 + sum|terms| * 2^-20``. ``fast`` serves every call ``det``
serves; while ``torch.use_deterministic_algorithms(True)`` is in effect, which its fused backward refuses, a call runs
``det`` and logs it once.

A call reads the configured variant (``miles_ext_kernels.cake.configure``) when its forward runs; its backward runs
the same package. :func:`prebuild` builds the libraries of both packages for a device, so no call compiles.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import ModuleType

import torch

from miles_ext_kernels.cake._runtime import config, loader

__all__ = ["check_supported", "prebuild", "release", "rmsnorm_silu_gate"]

OP = "gated_rmsnorm"
HEAD_DIM = 128
# the vendored package of each variant
PACKAGES = {"det": "cake_rmsnorm_silu_gate_det_train", "fast": "cake_rmsnorm_silu_gate_nondet_train"}

_BACKEND = "--linear-attention-output-norm-backend cake"


@functools.cache
def _package(variant: str) -> ModuleType:
    """The vendored package of ``variant``, its libraries built by the runtime loader."""
    return loader.load_package(PACKAGES[variant])


def _variant() -> str:
    """The variant a call runs: the configured one, or ``det`` for ``fast`` while torch's deterministic algorithms
    are enabled (the non-deterministic backward refuses them)."""
    variant = config.variant(OP)
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        config.warn_runs_det(OP, "torch.use_deterministic_algorithms(True)")
        return "det"
    return variant


def _arch(device) -> str:
    """The architecture of ``device`` (default: the current CUDA device), checked to be supported and to build with
    the toolkit."""
    if not torch.cuda.is_available():
        raise ValueError(f"{_BACKEND} needs a CUDA device of compute capability {tuple(loader.ARCH_BY_CAPABILITY)}")
    capability = tuple(torch.cuda.get_device_capability(device))
    arch = loader.ARCH_BY_CAPABILITY.get(capability)
    if arch is None:
        raise ValueError(
            f"{_BACKEND} runs on compute capability {tuple(loader.ARCH_BY_CAPABILITY)} (SM100a / SM103a / SM107a), not "
            f"{capability}; use --linear-attention-output-norm-backend fla"
        )
    loader.check_toolkit(arch, "the Cake gated RMSNorm kernels")
    return arch


def check_supported(
    head_dim: int, activation: str, weight_dtype: torch.dtype, *, has_bias: bool = False, device=None
) -> None:
    """Raise ``ValueError`` unless the Cake kernels run this gated output norm on ``device`` (default: the current
    CUDA device): a SiLU gate, rows of 128, a BF16 weight without bias, compute capability 10.0, 10.3 or 10.7;
    ``RuntimeError`` when the toolkit cannot build the device's kernels (they need CUDA 13.4 or newer)."""
    if activation != "silu":
        raise ValueError(
            f"{_BACKEND} computes RMSNorm(x) * w * SiLU(z), but this model gates its output norm with "
            f"{activation!r}; use --linear-attention-output-norm-backend fla"
        )
    if head_dim != HEAD_DIM:
        raise ValueError(f"{_BACKEND} normalizes rows of {HEAD_DIM}, got a value head dimension of {head_dim}")
    if has_bias:
        raise ValueError(f"{_BACKEND} has no bias")
    if weight_dtype != torch.bfloat16:
        raise ValueError(f"{_BACKEND} takes a BF16 weight, got {weight_dtype}")
    _arch(device)


def prebuild(*, device=None, max_workers: int | None = None) -> list[Path]:
    """Build (or find in the cache) every library of both variants' packages for the architecture of ``device``
    (default: the current CUDA device); returns their cache paths. Idempotent.

    Every route of both packages on that architecture is built: the forwards and backwards of every row-count band
    and layout (the tiny, short, pipelined, mid, long and extra-long forwards; the small-row, fused, repeatable and
    long backwards) and both weight-gradient dtypes, so no call of any shape compiles afterwards.
    ``NotImplementedError`` for an unsupported device, ``RuntimeError`` for a toolkit too old for it.
    """
    arch = loader.device_arch(device)
    paths: list[Path] = []
    for variant, package in PACKAGES.items():
        jit = _package(variant).cake_jit
        names = [jit.library_name(module, arch) for module in dict.fromkeys(jit.ROUTES[arch].values())]
        paths += loader.prebuild_libraries(package, names, max_workers)
    return paths


def release() -> None:
    """Nothing to drop: the packages keep no plans or graphs.

    Each package keeps the zero-filled workspaces of the streams its backward ran on (``det``: 18952 bytes per stream;
    ``fast``: one per stream and backward module, 29192 bytes on sm_100a and sm_103a and 36360 on sm_107a) and a
    reserve of zero-filled workspaces that CUDA graphs captured with it hold for good; they stay in place.
    """


def rmsnorm_silu_gate(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Autograd-enabled ``RMSNorm(x) * weight * SiLU(z)`` over the last dimension (128), in the configured variant.

    ``x`` and ``z``: BF16 of one ``[..., 128]`` shape. A gate that is a column slice of a wider projection is read in
    place when its last stride is 1, its storage offset and token and head strides are multiples of 16 elements and
    its storage is 32-byte aligned, else copied once (per call; the result is the same bits either way). ``weight``:
    BF16 ``[128]``, shared by every token and head of the call. Returns BF16 ``y`` of ``x``'s shape; the weight
    gradient comes in BF16. The autograd node saves only ``x``, ``z`` and ``weight``.

    The weight gradient sums over the rows of this call only. Under tensor parallelism each rank normalizes its own
    heads with the same replicated ``weight``, so the caller all-reduces the weight gradient over the tensor-parallel
    group (in miles, by passing the weight through ``copy_to_tensor_model_parallel_region``).
    """
    variant = _variant()
    package = _package(variant)
    if variant == "det":
        return package.cake_rmsnorm_silu_gate_det(x, z, weight, eps)
    return package.cake_rmsnorm_silu_gate_nondet(x, z, weight, eps)
