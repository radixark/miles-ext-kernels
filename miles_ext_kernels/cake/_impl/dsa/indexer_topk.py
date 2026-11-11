"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic DSA lightning-indexer top-k: FlashInfer's Cake package ``cake_dsa_indexer_det_train`` (built by the
runtime loader) on SM100a / SM103a / SM107a.

:func:`dsa_indexer_det_topk` scores every visible key of every query and selects the exact top-k in one call, without
a ``[T, T_kv]`` logits matrix::

    s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] * k[j, d])

with FP32 products, accumulation and score arithmetic in one fixed order, so a pair's score does not depend on how the
call is partitioned. Identical inputs give identical ids and score bits, call to call and process to process. With
``softmax_scale=1.0`` the scores are the head-summed ReLU logits of :func:`.ops.indexer_logits` (miles folds the
softmax scale into ``w``), so :func:`packed_key_ids` of the result feed :func:`.ops.sparse_attention` and the det
indexer backward (:func:`.ops.indexer_backward`); this call itself has no autograd graph.

* Geometry: ``q [T, 32, 128]`` BF16, ``k [T_kv, 128]`` BF16 (a row-strided view such as ``packed[:, :128]`` is read in
  place; the row stride must be a multiple of 8 elements), ``w [T, 32]`` FP32 (BF16 is widened, exactly), and
  ``1 <= top_k <= 4096``. :func:`indexer_det_topk_unserved_reason` names what another geometry misses; a call with one
  raises ``ValueError``.
* Segments: ``cu_seqlens_q`` / ``cu_seqlens_k`` ``[S + 1]`` int32 on the device, never read on the host. Key ``j``
  (segment-local) is visible to the query at segment-local position ``u`` iff ``j < L_k`` and
  ``j < floor((offset + u + 1) / ratio)``, with ``offset = q_causal_offsets[s]`` when given, else ``L_k - L_q`` for
  ``ratio == 1`` (the queries are the tail of their key prefix) and ``0`` for compressed keys.
* Selection: the exact top ``min(top_k, visible)`` keys of each row by score descending, then key id descending -- an
  equal-score tie keeps the **larger** key id (miles' canonical top-k keeps the smaller), and ``+0.0 == -0.0``.
* Output: ``indices [T, top_k]`` int32 segment-local key ids in ascending order and their ``scores [T, top_k]`` FP32;
  the tail is ``-1`` / ``-inf``.
* Workspace: :func:`dsa_indexer_det_topk_workspace_size` bytes for the call's geometry and the device's SM count
  (persistent candidate buffers, plus split staging when the dispatch splits the key ranges). Without ``workspace`` the
  call takes it from torch's caching allocator; a caller-owned uint8 buffer of at least that size may be reused across
  calls on one stream. No call synchronizes with the host.

The package picks one of its 65 (SM100a), 73 (SM103a) or 63 (SM107a) programs per call from ``T``, ``T_kv``, ``S``,
``ratio``, ``top_k`` and the SM count, so a new token count mid-run can reach a program not used before. Every library
of the architecture is therefore built ahead: by ``prebuild("dsa", ..., index_heads=32)``, else by the first call on the
architecture, unless ``--cake-dsa-lazy-build`` leaves each to its first launch.
"""

from __future__ import annotations

import functools
from importlib import import_module
from pathlib import Path

import torch

from miles_ext_kernels.cake._runtime import config, loader
from miles_ext_kernels.cake._runtime.loader import device_arch

__all__ = [
    "DEFAULT_TOP_K",
    "HEADS",
    "HEAD_DIM",
    "MAX_TOP_K",
    "dsa_indexer_det_topk",
    "dsa_indexer_det_topk_workspace_size",
    "indexer_det_topk_unserved_reason",
    "library_names",
    "packed_key_ids",
    "prebuild_libraries",
]

# The geometry the package serves (its cake_backend NUM_HEADS / HEAD_DIM / MAX_TOP_K / DEFAULT_TOP_K).
HEADS = 32
HEAD_DIM = 128
MAX_TOP_K = 4096
DEFAULT_TOP_K = 2048

PACKAGE = "cake_dsa_indexer_det_train"
_WHAT = "the Cake DSA indexer top-k kernels"
# architectures whose libraries are built (by the first call, or by prebuild)
_WARM: set[str] = set()


@functools.cache
def _package():
    """``(cake_build, cake_jit)`` of the vendored FlashInfer package, with the runtime loader installed."""
    name = loader.load_package(PACKAGE).__name__
    return import_module(f"{name}.cake_build"), import_module(f"{name}.cake_jit")


@functools.cache
def _backend():
    """The vendored package's ``cake_backend`` (validation, dispatch and launch; imports tvm_ffi)."""
    return import_module(f"{loader.load_package(PACKAGE).__name__}.cake_backend")


def indexer_det_topk_unserved_reason(heads: int, head_dim: int, top_k: int) -> str | None:
    """Why :func:`dsa_indexer_det_topk` cannot run a call of this geometry (``None`` when it can)."""
    if heads != HEADS or head_dim != HEAD_DIM:
        return f"{heads} index heads of dimension {head_dim} (only {HEADS} heads of dimension {HEAD_DIM})"
    if not 1 <= top_k <= MAX_TOP_K:
        return f"top_k {top_k} (only 1 to {MAX_TOP_K})"
    return None


def _arch(device) -> str:
    """The architecture of ``device``, checked to be served and its libraries to build with the toolkit."""
    arch = device_arch(device)
    loader.check_toolkit(arch, _WHAT)
    return arch


def library_names(arch: str) -> list[str]:
    """The ``build.json`` entries of every program the package registers for ``arch``."""
    _, cake_jit = _package()
    return sorted({cake_jit.library_name(program, arch) for program in cake_jit.PROGRAM_KEYS[arch].values()})


def prebuild_libraries(arch: str, max_workers: int | None = None) -> list[Path]:
    """Build (or find in the cache) every library of ``arch``; returns their paths. Idempotent."""
    paths = loader.prebuild_libraries(_package()[0], library_names(arch), max_workers)
    _WARM.add(arch)
    return paths


def _first_call(arch: str) -> None:
    if arch not in _WARM and not config.option("dsa", "cake_dsa_lazy_build"):
        prebuild_libraries(arch)


def dsa_indexer_det_topk_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = DEFAULT_TOP_K,
    ratio: int = 1,
    device=None,
) -> int:
    """Bytes of the workspace a call of this geometry needs on ``device`` (default: the current CUDA device).

    The bound follows the program the call dispatches, so it changes with ``T``, ``T_kv``, ``S``, ``ratio``, ``top_k``
    and the SM count: size a reused buffer for the largest bound over the geometries it serves.
    """
    _arch(device)
    return int(
        _backend().cake_dsa_indexer_det_train_topk_workspace_size(
            num_queries, num_keys, num_segments, top_k=top_k, ratio=ratio, device=device
        )
    )


def dsa_indexer_det_topk(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    top_k: int = DEFAULT_TOP_K,
    *,
    softmax_scale: float | None = None,
    q_causal_offsets: torch.Tensor | None = None,
    ratio: int = 1,
    max_seqlen_k: int | None = None,
    workspace: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused indexer scores and deterministic exact top-k of packed segments; returns ``(indices, scores)``.

    Args:
        index_q: BF16 ``[T, 32, 128]`` indexer queries after RoPE (made contiguous if it is not).
        index_k: BF16 ``[T_kv, 128]`` indexer keys; a row-strided view is read in place (row stride a multiple of 8).
        weights: ``[T, 32]`` head weights, FP32 or BF16 (widened); every scale except ``softmax_scale`` folded in.
        cu_seqlens_q, cu_seqlens_k: int32 ``[S + 1]`` query and key segment boundaries on the device.
        top_k: selected keys per query, ``1 <= top_k <= 4096``.
        softmax_scale: positive scale of the per-head dot products (default ``128 ** -0.5``; ``1.0`` when the scale
            is already in ``weights``, as in :func:`.ops.indexer_logits`).
        q_causal_offsets: optional int64 ``[S]`` per-segment query offsets (see the module docstring).
        ratio: key compression ratio (``>= 1``).
        max_seqlen_k: optional host bound on every key segment; selects the finalize program and the scan's segment
            order, never a result.
        workspace: optional caller-owned uint8 buffer of at least :func:`dsa_indexer_det_topk_workspace_size` bytes.
        indices, scores: optional caller-owned int32 / FP32 ``[T, top_k]`` outputs.

    Returns int32 segment-local key ids (ascending, ``-1`` padded; :func:`packed_key_ids` maps them to rows of
    ``index_k``) and their FP32 scores (``-inf`` padded). Ties keep the larger key id.
    """
    if index_q.dim() != 3:
        raise ValueError(f"index_q must be [T, {HEADS}, {HEAD_DIM}], got {tuple(index_q.shape)}")
    reason = indexer_det_topk_unserved_reason(int(index_q.shape[1]), int(index_q.shape[2]), int(top_k))
    if reason is not None:
        raise ValueError(f"{_WHAT} do not serve {reason}")
    arch = _arch(index_q.device)
    _first_call(arch)
    # the launches go to the inputs' device, whichever device is current
    with torch.cuda.device(index_q.device):
        return _backend().cake_dsa_indexer_det_train_topk(
            index_q.contiguous(),
            index_k,
            weights.to(torch.float32).contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            top_k=int(top_k),
            softmax_scale=softmax_scale,
            q_causal_offsets=q_causal_offsets,
            ratio=int(ratio),
            max_seqlen_k=max_seqlen_k,
            workspace_buffer=workspace,
            indices=indices,
            scores=scores,
        )


def packed_key_ids(indices: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor) -> torch.Tensor:
    """Segment-local key ids ``[T, top_k]`` -> rows of the packed key tensor: each id plus its query's segment start in
    ``cu_seqlens_k``; ``-1`` stays ``-1``. Ascending order is kept. No host synchronization."""
    rows = torch.arange(indices.shape[0], dtype=cu_seqlens_q.dtype, device=indices.device)
    # the segment of row t: the number of segment ends <= t (empty segments before it included)
    segment = torch.searchsorted(cu_seqlens_q[1:].contiguous(), rows, right=True)
    start = cu_seqlens_k.to(indices.dtype)[segment].unsqueeze(-1)
    return torch.where(indices >= 0, indices + start, indices)
