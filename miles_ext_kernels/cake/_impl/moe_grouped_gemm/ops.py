"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The ragged BF16 MoE grouped GEMM miles calls (SM100a / SM103a / SM107a), on two vendored FlashInfer packages whose
libraries the runtime's loader builds and loads (with a CUDA 13.4 or newer toolkit):
``_vendor/cake_moe_grouped_gemm_det_train`` (``det``) and ``_vendor/cake_moe_grouped_gemm_nondet_train``
(``fast``).

:func:`grouped_mm` is ``torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)`` for ``x [sum_m, K]``, ``w [E, N, K]``
and the int32 cumulative row ends ``offs [E]`` of the groups, differentiable in ``x`` and ``w``. The kernels read
``offs`` on the device (no host synchronisation); groups may be empty, one row long or of any size, and rows past
``offs[-1]`` are left alone. Every product is an FP32 accumulation of exact BF16 products rounded once to BF16 (the
cuBLAS / CUTLASS numerics, not their bits).

Variants. ``det`` reduces every output in a fixed order: the output and both gradients repeat bit for bit. ``fast``
gives the same output and input-gradient bits; its weight gradient, where it picks the 512-wide k tile (``K`` a multiple
of 512 and at least 4096 rows per group on average), adds the split tail of its last wave with unordered FP32 atomics
instead of a second, ordered reduction launch, so those tiles may differ in the last bit run to run. Every forward reads
the configured variant (``config.variant``); its backward runs the package of its forward. Under
``torch.use_deterministic_algorithms(True)`` a ``fast`` call runs ``det`` and logs it once (``config.warn_runs_det``):
a ``fast`` weight gradient with that unordered tail raises torch's non-determinism alert there.

32-bit addressing. The kernels address each output with 32-bit element offsets: ``sum_m * N`` (the output),
``sum_m * K`` (the input gradient) and ``E * N * K`` (the weight gradient) must stay at most ``2**31 - 1``.
:func:`check_grouped_dims` checks a weight ``[E, N, K]`` and the most rows a call may get where the model is built;
:func:`max_grouped_rows` is that bound. :func:`grouped_mm` checks every limit of the three products before its forward
launches anything, so a call the backward could not serve fails at its forward.

Workspaces. The packages keep small workspaces per device and CUDA stream (scheduler words, wave counters and the FP32
tail sums of the ``fast`` weight gradient) that every launch leaves at their rest state for the next one on its stream;
calls on two streams never share them, and a launch captured into a CUDA graph takes its own. They stay outside the
runtime's workspace pool, whose buffers do not keep their content. :func:`release` drops them.

Prebuild. :func:`prebuild` builds every program both variants may launch on the device's architecture (forward, input
gradient, both weight-gradient k tiles, every stage of each), and on SM100a / SM103a the probe of the memory domains the
input gradient compiles once per compute capability (``tvm_ffi.cpp.load_inline``, an nvcc build), so that no call
compiles afterwards.
"""

from __future__ import annotations

import functools
import logging
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader

__all__ = ["check_grouped_dims", "grouped_mm", "max_grouped_rows", "prebuild", "release"]

logger = logging.getLogger(__name__)

_OP = "moe_grouped_gemm"
# the vendored package of each variant
_PACKAGES = {"det": "cake_moe_grouped_gemm_det_train", "fast": "cake_moe_grouped_gemm_nondet_train"}
# miles_ext_kernels.cake._vendor
_VENDOR = f"{__name__.rsplit('._impl.', 1)[0]}._vendor"
_WHAT = "the Cake MoE grouped GEMM"
# The routes of one forward and backward with BF16 weights: (op, record_name keywords).
_ROUTES = (
    ("fwd", {}),
    ("dgrad", {}),
    ("wgrad", {"out_dtype": "bfloat16", "tile_k": 256}),
    ("wgrad", {"out_dtype": "bfloat16", "tile_k": 512}),
)
_DETERMINISTIC_ALGORITHMS = "calls under torch.use_deterministic_algorithms(True)"


def _module(variant: str, name: str) -> ModuleType:
    """The submodule ``name`` (``cake_autograd``, ``cake_backend``, ``cake_jit``) of ``variant``'s package."""
    return import_module(f"{_VENDOR}.{_PACKAGES[variant]}.{name}")


@functools.cache
def _int32_elems() -> int:
    """The largest element offset the programs of both packages address."""
    return min(int(_module(variant, "cake_jit").HOST_PLAN_CONSTANTS["int32_elems"]) for variant in _PACKAGES)


def max_grouped_rows(n: int, k: int) -> int:
    """The most rows (``sum_m``, padding rows included) a call with expert weights ``[E, N, K]`` serves: the output
    ``[sum_m, N]`` and the input gradient ``[sum_m, K]`` are addressed with 32-bit element offsets."""
    return _int32_elems() // max(int(n), int(k), 1)


def check_grouped_dims(n: int, k: int, *, num_groups: int | None = None, max_rows: int | None = None) -> None:
    """Raise ``ValueError`` unless the programs serve the expert weight ``[E, N, K]`` in all three products.

    The forward and the weight gradient need ``N`` a multiple of 256, the input and weight gradients ``K`` a multiple
    of 256. With ``num_groups`` (``E``), the weight gradient ``[E, N, K]`` must fit 32-bit element offsets; with
    ``max_rows``, the most rows a call may get (``sum_m``, padding rows included), so must the output and the input
    gradient (:func:`max_grouped_rows`). Call it where the model is built, with the bound of the configured run.
    """
    for name, dim in (("output features N", n), ("input features K", k)):
        if dim < 256 or dim % 256:
            raise ValueError(f"{_WHAT} needs expert {name} that are a multiple of 256, got {dim}")
    limit = _int32_elems()
    if num_groups is not None and num_groups * n * k > limit:
        raise ValueError(
            f"{_WHAT} addresses its weight gradient with 32-bit element offsets: {num_groups} experts of [{n}, {k}] "
            f"are {num_groups * n * k} elements, more than {limit}"
        )
    if max_rows is not None and max_rows > max_grouped_rows(n, k):
        raise ValueError(
            f"{_WHAT} addresses its output and input gradient with 32-bit element offsets: {max_rows} rows of "
            f"{max(n, k)} columns are more than {limit} elements; experts of [{n}, {k}] serve at most "
            f"{max_grouped_rows(n, k)} rows"
        )


def _device_index(device: Any) -> int:
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"{_WHAT} runs on CUDA devices, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


@functools.cache
def _arch(index: int) -> str:
    """The architecture of device ``index``, checked to be supported and its programs to build with the toolkit."""
    arch = loader.device_arch(index)
    loader.check_toolkit(arch, _WHAT)
    return arch


def grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """The ragged grouped GEMM ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``: ``[sum_m, N]`` BF16.

    ``x [sum_m, K]`` and ``w [E, N, K]`` are BF16 with unit stride on their last axis, ``offs`` the int32 cumulative
    row ends ``[E]`` on ``x``'s device (``torch._grouped_mm``'s ``offs``). The backward returns ``x.grad`` in BF16 and
    ``w.grad`` in ``w``'s dtype; an empty group gets an exactly zero weight gradient. The call runs the configured
    variant (``det`` unless configured); its backward runs the same package.
    """
    if x.dim() != 2 or w.dim() != 3 or x.shape[1] != w.shape[2]:
        raise ValueError(f"grouped_mm takes x [sum_m, K] and w [E, N, K], got {tuple(x.shape)} and {tuple(w.shape)}")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError(f"grouped_mm takes BF16 x and w, got {x.dtype} and {w.dtype}")
    if offs.dtype != torch.int32 or offs.dim() != 1 or offs.numel() != w.shape[0]:
        raise ValueError(
            f"offs must be the int32 cumulative row ends [E] of the {w.shape[0]} groups, got {offs.dtype} "
            f"{tuple(offs.shape)}"
        )
    if w.device != x.device or offs.device != x.device:
        raise ValueError(f"x, w and offs must be on one device, got {x.device}, {w.device} and {offs.device}")
    num_groups, n, k = (int(size) for size in w.shape)
    check_grouped_dims(n, k, num_groups=num_groups, max_rows=int(x.shape[0]))
    _arch(_device_index(x.device))
    variant = config.variant(_OP)
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        config.warn_runs_det(_OP, _DETERMINISTIC_ALGORITHMS)
        variant = "det"
    if variant == "det":
        return _module("det", "cake_autograd").cake_moe_grouped_gemm_det(x, w, offs)
    return _module("fast", "cake_autograd").cake_moe_grouped_gemm_nondet(x, w, offs)


def _library_names(variant: str, arch: str) -> list[str]:
    """Every library of ``variant``'s package the routes of one forward and backward bind on ``arch``."""
    jit = _module(variant, "cake_jit")
    names = []
    for op, keys in _ROUTES:
        if not jit.program_registered(op, arch, **keys):
            continue
        for stage in jit.ROUTES[jit.record_name(op, **keys)]["stages"]:
            if arch in jit.PROGRAMS[stage["program"]]["arches"]:
                names.append(jit.library_name(stage["program"], arch, stage["specializations"]))
    return names


def prebuild(*, device: Any = None, max_workers: int | None = None) -> list[Path]:
    """Build, ahead of the first call, every program :func:`grouped_mm` of either variant can launch on ``device``
    (default: the current device) with BF16 weights, whatever the group sizes: the forward, input-gradient and both
    weight-gradient routes, every stage built for its architecture (about a dozen libraries per variant), and the
    queue probe of the input gradient on SM100a / SM103a. No model parameter: one build serves every expert shape.
    Returns the cache paths of the libraries.

    ``NotImplementedError`` for a device other than SM100a / SM103a / SM107a; ``RuntimeError`` for a toolkit too old
    for its architecture.
    """
    index = _device_index(device)
    arch = _arch(index)
    paths = []
    for variant, package in _PACKAGES.items():
        paths += loader.prebuild_libraries(package, _library_names(variant, arch), max_workers)
    capability = torch.cuda.get_device_capability(index)
    for variant in _PACKAGES:
        backend = _module(variant, "cake_backend")
        if arch in backend.host_plan_constants()["domain_split_archs"]:
            # the probe the first probed launch on a stream would compile; the even split serves without it
            if backend._queue_probe_module(capability) is None:
                logger.warning(
                    "%s: the queue probe does not build on %s here; the input gradient splits its work queues evenly "
                    "(the same bits)",
                    _WHAT,
                    arch,
                )
    return paths


def release() -> None:
    """Drop the workspaces both packages keep per device and stream, with the queue splits probed on them."""
    for package in _PACKAGES.values():
        backend = sys.modules.get(f"{_VENDOR}.{package}.cake_backend")
        if backend is None:
            continue
        with backend._STREAM_WORKSPACES_LOCK:
            # a probed split is keyed by its workspace's address, which a later workspace may reuse
            backend._stream_workspaces.clear()
            backend._queue_splits.clear()
