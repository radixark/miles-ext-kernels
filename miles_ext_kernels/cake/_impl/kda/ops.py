"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Single-rank packed KDA chunked training on the vendored Cake packages ``cake_kda_det_train`` (``det``) and
``cake_kda_nondet_train`` (``fast``), SM100a / SM103a / SM107a; the op's :func:`prebuild` and :func:`release`.

:func:`chunk_kda`, :func:`chunk_kda_fwd` and :func:`chunk_kda_bwd` are FlashInfer's entry points (``cake_kda_det``,
``cake_kda_det_train_forward`` and ``cake_kda_det_train_backward`` of the deterministic package, ``cake_kda_nondet``,
``cake_kda_nondet_train_forward`` and ``cake_kda_nondet_train_backward`` of the other) without its API logging: they
default ``scale`` to ``128 ** -0.5`` and forward every keyword to the ``cake_backend`` of the package of the call,
which checks the operands, plans the launches from the host scalars alone and runs them on the current stream. The
entries of both packages take the operands and keywords passed here and return their outputs and gradients in the
same form.

The package of a call: ``config.variant("kda")`` when the forward runs (``det`` unless configured), in every gate
and L2 norm mode. The autograd node's backward is its forward's package; :func:`chunk_kda_bwd` takes the package of
the forward that wrote ``saved`` (the variant of the call for ``saved=None``, the replay).

* ``det``: every result is bit-identical across repeated calls on the same inputs (fixed-order reductions, no
  atomics), on the same GPU model and SM count, toolkit and build.
* ``fast``: in the activated-gate mode (``use_gate_in_kernel=False``) without the in-kernel L2 norm its backward adds
  the ``q * dq`` term of the gate gradient in the GLM intra program on the rows its registry lists, so ``dg`` may
  differ from ``det``'s by FP32 rounding. In the raw gate mode, and with the in-kernel L2 norm in either gate mode,
  every result is ``det``'s, bit for bit (FlashInfer's contract for these modes).

Neither variant reads ``torch.use_deterministic_algorithms``: no launch of either package reduces in an unordered
order, so neither raises PyTorch's non-determinism alert.

Operands, one packed batch of ``S`` tokens in ``num_docs`` documents with ``H`` heads (1 to 64): ``q, k, v`` BF16
``[1, S, H, 128]`` (the batch axis is optional; outputs take the rank of the inputs), ``g`` the gate ``[1, S, H,
128]``, ``beta`` ``[1, S, H]`` after the sigmoid, ``cu_seqlens`` int32 ``[num_docs + 1]`` on the device.

* ``beta`` may come in any floating dtype: it is rounded to BF16 here (miles hands flash-linear-attention the FP32
  sigmoid; at 12 heads on GB300 the cast adds about 1.6e-3 relative RMS to ``out``, against 4.1e-3 from the
  kernels), and its gradient takes the caller's dtype. ``v`` may be a strided view (a column slice of a fused
  projection): it is made contiguous here, as the kernels read dense rows.
* ``use_gate_in_kernel=True``: ``g`` is the raw BF16 forget gate (it may be a strided view), activated in the
  kernels as ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` with ``A_log`` FP32 ``[H]``, ``dt_bias`` FP32
  ``[H * 128]`` and ``-5 <= lower_bound < 0``; the gradients reach ``g``, ``A_log`` and ``dt_bias``. Otherwise ``g``
  is the activated FP32 gate.
* ``use_qk_l2norm_in_kernel=True``: ``q`` / ``k`` are the raw rows, L2-normalized per token and head in the kernels
  (eps ``1e-6``); they may be column slices of one packed projection (head stride 128, token stride a multiple of 8
  elements, 16-byte aligned). ``dq`` / ``dk`` are the raw rows' gradients. Otherwise ``q`` / ``k`` are the
  normalized rows, contiguous.

:func:`prebuild` compiles every library of the four packages for the device's architecture, so no call (either
variant, any head count, token or document count, gate or L2 norm mode, path, single-rank or context-parallel)
compiles inside a training step. Importing this module imports both single-rank packages and rebinds the toolkit
hook of the ``fast`` one (below); it imports no flashinfer.
"""

from __future__ import annotations

import logging
import operator
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader
from miles_ext_kernels.cake._vendor.cake_kda_det_train import cake_backend as det_backend
from miles_ext_kernels.cake._vendor.cake_kda_nondet_train import cake_backend as fast_backend

__all__ = [
    "DEFAULT_SCALE",
    "MAX_HEADS",
    "PACKAGES",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "libraries",
    "prebuild",
    "release",
]

logger = logging.getLogger(__name__)

OP = "kda"
# miles_ext_kernels.cake._vendor: the vendored packages are its subpackages
_VENDOR = det_backend.__name__.rsplit(".", 2)[0]
# The packages of the op: the single-rank ones and the context-parallel ones, det first.
PACKAGES = ("cake_kda_det_train", "cake_kda_det_train_kcp", "cake_kda_nondet_train", "cake_kda_nondet_train_kcp")
# The default scale of the kernels' contract: head_dim ** -0.5 with head dimension 128.
DEFAULT_SCALE = 128**-0.5
# The head counts the kernels are validated for (per call, so per tensor-parallel rank).
MAX_HEADS = 64


def _nvcc_version() -> str | None:
    """The full version (``"13.4.92"``) of the nvcc the runtime loader builds the libraries with.

    The ``fast`` package's checkpoint hand-off keys its row records on the toolkit that built them, which it asks
    FlashInfer's JIT for (``cake_backend._jit_nvcc_version``). The runtime loader builds them here, so this function
    replaces that hook when this module is imported, and flashinfer is never imported.
    """
    return fast_backend._full_version(loader.get_cuda_version().full)


fast_backend._jit_nvcc_version = _nvcc_version


def check_heads(num_heads: Any) -> int:
    """``num_heads`` as an int when the kernels serve it (1 to :data:`MAX_HEADS`), else ``ValueError``."""
    try:
        heads = None if isinstance(num_heads, bool) else operator.index(num_heads)
    except TypeError:
        heads = None
    if heads is None or not 1 <= heads <= MAX_HEADS:
        raise ValueError(
            f"the Cake KDA kernels serve 1 to {MAX_HEADS} heads per call (per tensor-parallel rank of a head-sharded "
            f"layer), got {num_heads!r}: shard the heads over more ranks with --tensor-model-parallel-size, or use "
            "--kda-backend fla"
        )
    return heads


def seam(v: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``v`` dense and ``beta`` BF16 and dense, as the kernels read them (differentiable: autograd casts the
    gradient of ``beta`` back to its dtype); other operands are left to the packages' checks."""
    if beta.is_floating_point() and beta.dtype != torch.bfloat16:
        beta = beta.to(torch.bfloat16)
    return v.contiguous(), beta.contiguous()


def _modes(scale, use_gate_in_kernel, A_log, dt_bias, lower_bound, use_qk_l2norm_in_kernel) -> dict[str, Any]:
    """The keywords of a call, the same for either package."""
    return dict(
        scale=DEFAULT_SCALE if scale is None else scale,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    """Packed KDA chunked training as an autograd node of the package of the call (module docstring): ``out`` BF16
    like ``v``, with gradients to ``q, k, v, g, beta`` and, with ``use_gate_in_kernel=True``, to ``A_log`` /
    ``dt_bias``. Compatible with ``torch.utils.checkpoint``."""
    check_heads(int(v.shape[-2]))
    v, beta = seam(v, beta)
    modes = _modes(scale, use_gate_in_kernel, A_log, dt_bias, lower_bound, use_qk_l2norm_in_kernel)
    if config.variant(OP) == "fast":
        return fast_backend.cake_kda_nondet(q, k, v, g, beta, cu_seqlens, **modes)
    return det_backend.cake_kda_det(q, k, v, g, beta, cu_seqlens, **modes)


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float | None = None,
    save_for_backward: bool = True,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, Any]:
    """The forward alone: ``(out, saved)``, ``saved`` the set :func:`chunk_kda_bwd` takes (``None`` without
    ``save_for_backward``). Pass the backward the same operands and modes."""
    check_heads(int(v.shape[-2]))
    v, beta = seam(v, beta)
    modes = _modes(scale, use_gate_in_kernel, A_log, dt_bias, lower_bound, use_qk_l2norm_in_kernel)
    if config.variant(OP) == "fast":
        return fast_backend.cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu_seqlens, save_for_backward=save_for_backward, **modes
        )
    return det_backend.cake_kda_det_train_forward(
        q, k, v, g, beta, cu_seqlens, save_for_backward=save_for_backward, **modes
    )


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dout: torch.Tensor,
    *,
    saved: Any = None,
    scale: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, ...]:
    """The backward alone, from the forward's operands and modes: ``(dq, dk, dv, dbeta, dg)``, and ``(dq, dk, dv,
    dbeta, dg, dA_log, ddt_bias)`` with ``use_gate_in_kernel=True``; ``dbeta`` in ``beta``'s dtype. It runs the
    package of the forward that wrote ``saved`` (a saved set is read by its own package only); ``saved=None`` replays
    the forward (checkpoint replay) in the package of the call, and under ``det`` gives the same bits as the saved-set
    backward."""
    check_heads(int(v.shape[-2]))
    beta_dtype = beta.dtype
    v, beta = seam(v, beta)
    modes = _modes(scale, use_gate_in_kernel, A_log, dt_bias, lower_bound, use_qk_l2norm_in_kernel)
    if saved is None:
        variant = config.variant(OP)
    else:
        variant = "fast" if isinstance(saved, fast_backend.CakeKDANondetChunkSaved) else "det"
    if variant == "fast":
        grads = fast_backend.cake_kda_nondet_train_backward(q, k, v, g, beta, cu_seqlens, dout, saved=saved, **modes)
    else:
        grads = det_backend.cake_kda_det_train_backward(q, k, v, g, beta, cu_seqlens, dout, saved=saved, **modes)
    grads = list(grads)
    grads[3] = grads[3].to(beta_dtype)
    return tuple(grads)


def _cake_build(package: str) -> Any:
    """The ``cake_build`` module of a vendored package of the op, the runtime loader installed on it."""
    loader.load_package(package)
    return import_module(f"{_VENDOR}.{package}.cake_build")


def libraries(arch: str) -> dict[str, list[str]]:
    """Every ``build.json`` library of the op's packages for ``arch``, per package (``build.test.json`` excluded)."""
    libraries = {}
    for package in PACKAGES:
        entries = _cake_build(package).libraries_by_name()
        libraries[package] = sorted(name for name, entry in entries.items() if entry["arch"] == arch)
    return libraries


def prebuild(*, device: Any = None, max_workers: int | None = None, num_heads: int) -> list[Path]:
    """Build every library a call of a ``num_heads``-head layer can launch on ``device`` (default: current).

    That is every library of the four packages for the device's architecture: the record a launch takes depends on
    the token and document counts of the call (through chunk, chain, slot and segment thresholds of the packages'
    registries), on its gate and L2 norm modes, its path (saving forward, forward without a saved set, saved-set and
    replay backwards, the checkpoint recompute), the variant and, for a context-parallel call, the rank's shard, so
    no smaller set holds for every call of the layer. ``num_heads`` is checked (1 to :data:`MAX_HEADS`,
    ``ValueError``) before anything is built; an unsupported device raises ``NotImplementedError``, a toolkit too old
    for it ``RuntimeError``. Returns the cache paths, built now or before; idempotent.
    """
    check_heads(num_heads)
    arch = loader.device_arch(device)
    loader.check_toolkit(arch, "the Cake KDA kernels")
    paths: list[Path] = []
    for package, names in libraries(arch).items():
        paths += loader.prebuild_libraries(_cake_build(package), names, max_workers)
    logger.info("miles_ext_kernels.cake.kda: %d %s libraries ready", len(paths), arch)
    return paths


def _clear(module: Any, pool: str, lock: str) -> None:
    with getattr(module, lock):
        getattr(module, pool).clear()


def release() -> None:
    """Drop the host state the packages keep between calls: the per-stream workspace records of the calls made
    without a workspace (their device memory is freed when each call returns) and, of a context-parallel package
    in use, its captured CUDA graphs with their memory pools."""
    for backend in (det_backend, fast_backend):
        _clear(backend, "_entry_pool", "_entry_lock")
    for package in ("cake_kda_det_train_kcp", "cake_kda_nondet_train_kcp"):
        kcp = sys.modules.get(f"{_VENDOR}.{package}.cake_kcp")
        if kcp is not None:
            kcp.graph_cache_clear()
