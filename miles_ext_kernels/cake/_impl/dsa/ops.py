"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

DSA (DeepSeek Sparse Attention) training operators: sparse attention and lightning indexer, forward and backward.

Public surface (all layouts of :mod:`.layouts`):

* :func:`sparse_attention` -- autograd sparse MQA/MLA attention over top-k indices with an optional FP32 attention
  sink, in the variant ``--cake-dsa-variant`` configures (read when the call runs):

  - ``det``: the DSA det version (``cake_dsa_sparse_det``, launched through :mod:`.kernels`). Every gradient is
    bit-reproducible run to run: no floating-point atomics, and the key-side gradients are reduced from per-slot FP32
    partials in an order fixed by the indices.
  - ``fast``: FlashInfer's Cake DSA training kernels (``cake_dsa_sparse_attention_nondet_train``) for 64 local query
    heads, ``d_v = 512`` with a 64-channel RoPE tail or none, and no attention sink. ``dq`` is bit-reproducible;
    ``dkv`` is accumulated with FP32 ``red.global`` atomics, so its summation order changes from run to run. A call
    ``fast`` does not serve (:func:`fast_unserved_reason`) runs ``det`` and says so once per reason; so does every call
    while ``torch.use_deterministic_algorithms(True)`` is in effect, which the FlashInfer backward refuses.

* :func:`indexer_logits`, :func:`select_topk`, :func:`indexer_topk_scores`, :func:`lightning_indexer` -- the
  deterministic indexer pipeline over a caller-supplied top-k callable (:func:`torch_topk` by default), with an
  explicit ``topk_indices`` override for rollout replay. The indexer serves 8, 16, 32 or 64 heads of dimension 128
  (:func:`indexer_unserved_reason`), forward and backward alike.
* :func:`sparse_attention_forward` / :func:`sparse_attention_backward` / :func:`indexer_backward` -- non-autograd
  ``det`` entry points for tests and benchmarks.
* :func:`check_device` and :func:`prebuild` -- the device check and the build of every library a layer may launch.

Indices: ``-1`` marks an empty slot, and so does any index outside the key rows the slot can reach (``[0, T_kv)`` of
the packed keys for ``thd``, ``[0, S_kv)`` of its own batch element for ``bshd``): it is never read. ``sm_scale`` has
no default: it is the model's softmax scale, which is not ``d_qk ** -0.5`` for the absorbed MLA of the models these
kernels serve.

Parallelism is not handled here: the kernels see local heads and the local-or-gathered key rows.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader
from miles_ext_kernels.cake._runtime.loader import device_arch

from . import kernels, layouts, native, reduction
from .layouts import RowMeta

OP = "dsa"

ATTENTION_TOPK_MULTIPLE = 64
LOG2E = 1.4426950408889634
# Reducer of the lightning-indexer backward: 4 = ``dsa_segmented_reduce_split4`` (fixed 4-way tree per key,
# deterministic FP32); 1 = the serial per-key reducer ``dsa_segmented_reduce``.
INDEXER_REDUCE_SPLIT = 4
# Reducer of the sparse-attention backward: the same fixed 4-way tree (bandwidth-bound; the serial reducer's
# per-key latency chain is the tail of every chunk).  Fixed-order FP32.
ATTENTION_REDUCE_SPLIT = 4

# The calls the fast variant serves: 64 local query heads, d_v 512 and d_qk 576 (the RoPE program) or 512 (the NoPE
# program), no attention sink.
FAST_HEADS = 64
FAST_D_V = 512
FAST_D_ROPE = 64

# Why a fast call runs det while torch's deterministic algorithms are enabled: the FlashInfer backward raises PyTorch's
# alert then (its dkv sums use atomics).
_DETERMINISTIC_ALGORITHMS = "calls under torch.use_deterministic_algorithms(True)"

TopkFn = Callable[[torch.Tensor, int], torch.Tensor]

# (program, architecture) whose fast libraries are built (by the first fast call, or by prebuild)
_FAST_WARM: set[tuple[str, str]] = set()


# --------------------------------------------------------------------------- #
# Device checks and builds
# --------------------------------------------------------------------------- #


def check_device(device=None) -> str:
    """The architecture tag of ``device`` (``sm_100a`` / ``sm_103a`` / ``sm_107a``); raises ``NotImplementedError``
    elsewhere, and ``RuntimeError`` when the toolkit cannot build the device's kernels (SM107a needs CUDA 13.4)."""
    if not torch.cuda.is_available():
        raise NotImplementedError("the Cake DSA kernels need a CUDA device (SM100a, SM103a or SM107a)")
    arch = device_arch(device)
    loader.check_toolkit(arch, "the DSA training kernels")
    return arch


def fast_unserved_reason(heads: int, d_qk: int, d_v: int, *, has_sink: bool, num_kv: int = 1) -> str | None:
    """Why the fast variant cannot run a call of this geometry (``None`` when it can); such a call runs ``det``."""
    if has_sink:
        return "calls with an attention sink"
    if heads != FAST_HEADS:
        return f"{heads} local query heads (only {FAST_HEADS})"
    if d_v != FAST_D_V or d_qk not in (FAST_D_V, FAST_D_V + FAST_D_ROPE):
        return f"d_qk {d_qk} with d_v {d_v} (only d_v {FAST_D_V} with d_qk {FAST_D_V + FAST_D_ROPE} or {FAST_D_V})"
    if num_kv == 0:
        return "an empty key set"
    return None


@functools.cache
def _fast_backend():
    """``(cake_build, cake_jit, cake_backend)`` of the vendored FlashInfer package
    ``cake_dsa_sparse_attention_nondet_train``."""
    package = loader.load_package("cake_dsa_sparse_attention_nondet_train").__name__
    from importlib import import_module

    return tuple(import_module(f"{package}.{name}") for name in ("cake_build", "cake_jit", "cake_backend"))


def _fast_program(d_qk: int) -> str:
    _, cake_jit, _ = _fast_backend()
    return cake_jit.PROGRAM if d_qk == FAST_D_V + FAST_D_ROPE else cake_jit.NOPE_PROGRAM


def _build_fast_program(program: str, arch: str, max_workers: int | None = None) -> list[Path]:
    """Build every library of the FlashInfer ``program`` for ``arch`` (cached ones are skipped)."""
    cake_build, cake_jit, _ = _fast_backend()
    names = [cake_jit.library_name(stage, arch, program) for stage in cake_jit.registered_stages(program)]
    paths = loader.prebuild_libraries(cake_build, names, max_workers)
    _FAST_WARM.add((program, arch))
    return paths


def _fast_first_call(program: str, arch: str) -> None:
    """On the first fast call of ``program`` on ``arch``, build all its libraries unless builds are lazy."""
    if (program, arch) not in _FAST_WARM and not config.option(OP, "cake_dsa_lazy_build"):
        _build_fast_program(program, arch)


def check_geometry(heads: int, d_qk: int, d_v: int) -> None:
    """``ValueError`` for a sparse-attention geometry the det kernels do not serve."""
    if heads < 1:
        raise ValueError(f"heads must be positive, got {heads}")
    if (d_v, d_qk - d_v) not in kernels.ATTENTION_SHAPES:
        raise ValueError(f"unsupported (d_v, d_tail) = {(d_v, d_qk - d_v)}; supported: {kernels.ATTENTION_SHAPES}")


def prebuild(
    *,
    device=None,
    max_workers: int | None = None,
    heads: int,
    d_qk: int,
    d_v: int = 512,
    attn_sink: bool = False,
    index_heads: int | None = None,
) -> list[Path]:
    """Build every library a DSA layer of this geometry may launch on ``device``; returns their cache paths.

    The model parameters: ``heads`` local query heads, ``d_qk`` / ``d_v`` (576 / 512 with a RoPE tail, 512 / 512
    without), ``attn_sink`` (the layer has one), ``index_heads`` (the indexer's heads, when the layer has an
    indexer; 32 also builds the det indexer top-k, the only head count it serves). Built: every det stage of the
    architecture (sparse attention, indexer logits and backward, reducers), the FlashInfer program the geometry
    runs under ``fast`` when fast serves it, and the det top-k libraries for ``index_heads == 32``; so no later call
    with these parameters compiles, whatever variant is configured and whatever the token count. With
    ``--cake-dsa-lazy-build`` it builds nothing and returns ``[]``. Idempotent.
    """
    from . import indexer_topk

    check_geometry(int(heads), int(d_qk), int(d_v))
    if index_heads is not None:
        kernels.check_indexer(int(index_heads), kernels.INDEXER_HEAD_DIM)
    arch = check_device(device)
    if config.option(OP, "cake_dsa_lazy_build"):
        return []
    paths = native.prebuild(arch, max_workers)
    kernels._WARM.add(arch)
    if fast_unserved_reason(int(heads), int(d_qk), int(d_v), has_sink=bool(attn_sink)) is None:
        paths += _build_fast_program(_fast_program(int(d_qk)), arch, max_workers)
    if index_heads == indexer_topk.HEADS:
        paths += indexer_topk.prebuild_libraries(arch, max_workers)
    return paths


def release() -> None:
    """Drop the cached FlashInfer bindings (with their workspaces) and the loaded det stages."""
    import sys

    backend = sys.modules.get(f"{loader.load_package('cake_dsa_sparse_attention_nondet_train').__name__}.cake_backend")
    if backend is not None:
        backend.BINDING_CACHE.clear()
    native.release()


# --------------------------------------------------------------------------- #
# Top-k selection (the caller owns the policy: a replay-manager wrapper, a range-aware top-k, ...)
# --------------------------------------------------------------------------- #


def torch_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """``torch.topk`` over the last axis as int32 indices; ``-inf`` picks and the short-axis pad become ``-1``."""
    count = min(topk, logits.shape[-1])
    scores, indices = torch.topk(logits, count, dim=-1)
    indices = indices.to(torch.int32).masked_fill(scores == float("-inf"), -1)
    if count < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - count), value=-1)
    return indices


def select_topk(logits: torch.Tensor, topk: int, topk_fn: TopkFn | None = None) -> torch.Tensor:
    """Select ``min(topk, num_keys)`` key indices per query row of ``logits[..., num_keys]`` with ``topk_fn``.

    Rows are flattened to ``[rows, num_keys]`` for ``topk_fn(logits, count)`` (:func:`torch_topk` by default) and
    the result is reshaped to ``[..., topk]``; short key axes are padded with ``-1``.
    """
    num_keys = logits.shape[-1]
    count = min(topk, num_keys)
    flat = logits.reshape(math.prod(logits.shape[:-1]), num_keys)
    indices = (topk_fn or torch_topk)(flat, count).to(torch.int32)
    if count < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - count), value=-1)
    return indices.reshape(*logits.shape[:-1], topk)


def extract_topk_scores(logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    """Gather ``logits`` at ``topk_indices`` along the last axis; ``-1`` slots and indices past the last key become
    ``-inf``."""
    valid = (topk_indices >= 0) & (topk_indices < logits.shape[-1])
    safe = torch.where(valid, topk_indices, 0).to(torch.int64)
    scores = torch.gather(logits, dim=-1, index=safe)
    return torch.where(valid, scores, torch.full_like(scores, float("-inf")))


# --------------------------------------------------------------------------- #
# Sparse attention: flat kernels
# --------------------------------------------------------------------------- #


def _sink_or_disabled(attn_sink: torch.Tensor | None, heads: int, device) -> torch.Tensor:
    if attn_sink is None:
        return torch.full((heads,), float("-inf"), dtype=torch.float32, device=device)
    if attn_sink.shape != (heads,):
        raise ValueError(f"attn_sink must be [heads={heads}], got {tuple(attn_sink.shape)}")
    return attn_sink.to(torch.float32).contiguous()


def _flat_attention_forward(q, kv, indices, sink, sm_scale, d_v, meta: RowMeta):
    """Run the flat forward; return ``(o_flat, lse_flat, o_layout)``.

    ``o_layout`` is the output in the model layout and is the tensor the wrapper hands back.  For batched
    layouts it is allocated here in layout shape and the kernel writes through a flat view of it, so the
    returned tensor is a fresh allocation rather than a view of an intermediate (callers such as the
    DeepSeek-V4 attention apply the inverse RoPE in place on the output, which autograd rejects for views
    produced inside a custom Function).
    """
    out = None
    o_layout = None
    if meta.layout != "thd":
        o_layout = torch.empty(meta.batch, meta.seq_len, q.shape[1], d_v, dtype=q.dtype, device=q.device)
        out = o_layout.view(q.shape[0], q.shape[1], d_v)
    o_flat, lse_flat = kernels.dsa_attention_forward(q, kv, indices, sink, float(sm_scale), d_v=d_v, out=out)
    return o_flat, lse_flat, (o_layout if o_layout is not None else o_flat)


def _budget(workspace_bytes: int | None) -> int:
    """The det backward's memory budget: the argument, else ``--cake-dsa-bwd-workspace-bytes``."""
    budget = config.option(OP, "cake_dsa_bwd_workspace_bytes") if workspace_bytes is None else workspace_bytes
    if int(budget) <= 0:
        raise ValueError(f"the det backward's workspace budget must be positive, got {budget}")
    return int(budget)


def _flat_attention_backward(q, kv, o, do, indices, lse, sink, sm_scale, d_v, workspace_bytes):
    num_rows, heads, d_qk = q.shape
    num_kv = kv.shape[0]
    topk = indices.shape[1]
    dsink = None if sink is None else torch.zeros(heads, dtype=torch.float32, device=q.device)
    if num_rows == 0:
        return torch.empty_like(q), torch.zeros(num_kv, d_qk, dtype=kv.dtype, device=kv.device), dsink
    do = do.contiguous()
    geo = kernels.attention_geometry(q, kv, indices, d_v)
    # delta = rowsum(O * dO) in FP32: one kernel, fixed order (bit-reproducible).
    delta = kernels.dsa_attention_delta_launch(o.contiguous(), do)
    dq = torch.empty_like(q)
    dkv32 = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    # FP32 partial rows per (query row, slot): one per head block, or one per row when the two head blocks of a row
    # run as a cluster and sum their contributions on chip (``geo.pair``), or a tcgen05 CTA holds the row's heads
    kv_partials = geo.kv_partials
    plan = reduction.plan_backward(num_rows, kv_partials * topk, d_qk, num_kv, _budget(workspace_bytes))
    rows_per_chunk = plan.rows_per_chunk
    # one stable sort of every (key, chunk) composite, before the partials exist, when it fits the budget
    segments = reduction.ChunkedSegments(indices, kv_partials, rows_per_chunk, num_kv) if plan.one_sort else None
    partial = torch.empty(rows_per_chunk * kv_partials * topk, d_qk, dtype=torch.float32, device=q.device)
    for chunk, (start, end) in enumerate(plan.chunks):
        rows = end - start
        part = partial[: rows * kv_partials * topk]
        kernels.dsa_attention_backward_rows(
            q[start:end],
            kv,
            do[start:end],
            indices[start:end],
            lse[start:end],
            delta[start:end],
            dq[start:end],
            part,
            float(sm_scale),
            geo,
        )
        if segments is not None:
            order, seg_start, seg_count = segments.segment(chunk)
        else:
            keys = indices[start:end].unsqueeze(1).expand(rows, kv_partials, topk).reshape(-1)
            order, seg_start, seg_count = reduction.segment_pairs(keys, num_kv)
        kernels.dsa_segmented_reduce_launch(part, order, seg_start, seg_count, dkv32, split=ATTENTION_REDUCE_SPLIT)
    dkv = dkv32.to(kv.dtype)
    if sink is not None:
        # d sink[h] = -sum_rows delta[row, h] * p_sink[row, h], p_sink = exp2(sink * log2 e - lse).
        dsink = -(delta * torch.exp2(sink.unsqueeze(0) * LOG2E - lse)).sum(dim=0)
    return dq, dkv, dsink


# --------------------------------------------------------------------------- #
# Sparse attention: public functional + autograd
# --------------------------------------------------------------------------- #


def _prepare_attention(q, kv, indices, layout):
    q_flat, kv_flat, idx_flat, meta = layouts.flatten_attention_inputs(q, kv, indices, layout)
    idx_flat = layouts.pad_topk(idx_flat, ATTENTION_TOPK_MULTIPLE)
    if q_flat.dtype != torch.bfloat16 or kv_flat.dtype != torch.bfloat16:
        raise TypeError("sparse attention expects BF16 q and kv")
    return q_flat, kv_flat, idx_flat, meta


def _check_scale(sm_scale: Any) -> float:
    if sm_scale is None:
        raise TypeError("sm_scale is required: pass the model's softmax scale")
    return float(sm_scale)


def sparse_attention_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    layout: str = "thd",
    d_v: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``det`` forward only: returns ``(output, lse)`` in the input layout (``lse`` is base-2, sink included)."""
    sm_scale = _check_scale(sm_scale)
    q_flat, kv_flat, idx_flat, meta = _prepare_attention(q, kv, indices, layout)
    sink = _sink_or_disabled(attn_sink, q_flat.shape[1], q_flat.device)
    _o_flat, lse_flat, o_layout = _flat_attention_forward(q_flat, kv_flat, idx_flat, sink, sm_scale, d_v, meta)
    return o_layout, meta.rows_to_layout(lse_flat)


def sparse_attention_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    *,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    layout: str = "thd",
    d_v: int = 512,
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``det`` backward only: returns ``(dq, dkv, d_attn_sink)`` in the input layout (``d_attn_sink`` is ``None``
    without a sink)."""
    sm_scale = _check_scale(sm_scale)
    q_flat, kv_flat, idx_flat, meta = _prepare_attention(q, kv, indices, layout)
    sink = None if attn_sink is None else _sink_or_disabled(attn_sink, q_flat.shape[1], q_flat.device)
    o_flat, _ = layouts.flatten_rows(o, layout, kind="q")
    do_flat, _ = layouts.flatten_rows(do, layout, kind="q")
    lse_flat, _ = layouts.flatten_rows(lse, layout, kind="q")
    dq, dkv, dsink = _flat_attention_backward(
        q_flat, kv_flat, o_flat, do_flat, idx_flat, lse_flat.float(), sink, sm_scale, d_v, workspace_bytes
    )
    dkv = meta.kv_rows_to_layout(dkv)
    if layout == "thd" and kv.dim() == 3:
        dkv = dkv.unsqueeze(1)
    return meta.rows_to_layout(dq), dkv, dsink


class SparseAttentionFunction(torch.autograd.Function):
    """Autograd wrapper around the det sparse attention kernels."""

    @staticmethod
    def forward(ctx, q, kv, attn_sink, indices, sm_scale, layout, d_v, workspace_bytes):
        q_flat, kv_flat, idx_flat, meta = _prepare_attention(q, kv, indices, layout)
        has_sink = attn_sink is not None
        sink = _sink_or_disabled(attn_sink, q_flat.shape[1], q_flat.device)
        o_flat, lse_flat, o_layout = _flat_attention_forward(q_flat, kv_flat, idx_flat, sink, sm_scale, d_v, meta)
        if o_layout is not o_flat:
            # o_flat is a view of the returned layout tensor, created here in no-grad mode. DeepSeek-V4
            # applies the inverse RoPE in place on the output before the projection, so save a copy that
            # the in-place update cannot reach (the pinned TileLang function saves o.clone() for the same reason).
            o_flat = o_flat.clone()
        ctx.save_for_backward(q_flat, kv_flat, idx_flat, o_flat, lse_flat, sink)
        ctx.meta = meta
        ctx.sm_scale = float(sm_scale)
        ctx.d_v = int(d_v)
        # the budget of the forward's call: a backward runs with what its forward saw
        ctx.workspace_bytes = _budget(workspace_bytes)
        ctx.has_sink = has_sink
        ctx.kv_had_group_dim = layout == "thd" and kv.dim() == 3
        return o_layout

    @staticmethod
    def backward(ctx, grad_output):
        q_flat, kv_flat, idx_flat, o_flat, lse_flat, sink = ctx.saved_tensors
        meta: RowMeta = ctx.meta
        do_flat, _ = layouts.flatten_rows(grad_output, meta.layout, kind="q")
        dq, dkv, dsink = _flat_attention_backward(
            q_flat,
            kv_flat,
            o_flat,
            do_flat,
            idx_flat,
            lse_flat,
            sink if ctx.has_sink else None,
            ctx.sm_scale,
            ctx.d_v,
            ctx.workspace_bytes,
        )
        dkv = meta.kv_rows_to_layout(dkv)
        if ctx.kv_had_group_dim:
            dkv = dkv.unsqueeze(1)
        return meta.rows_to_layout(dq), dkv, dsink, None, None, None, None, None


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` when its base is 16-byte aligned (the TMA descriptors of the fast kernels need it), else a copy."""
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def _fast_sparse_attention(q, kv, indices, sm_scale: float, layout: str, num_segments: int) -> torch.Tensor:
    """The fast variant on the flat contract: the FlashInfer RoPE program for ``d_qk = 576`` (the latent and the RoPE
    tail as strided views of the packed rows, no copy), the NoPE program for ``d_qk = 512``. ``num_segments`` is the
    packed segment count of the key rows: the backward plans its key-range passes from it (the results do not depend
    on it)."""
    q_flat, kv_flat, idx_flat, meta = layouts.flatten_attention_inputs(q, kv, indices, layout)
    q_flat, kv_flat = _aligned(q_flat), _aligned(kv_flat)
    _, cake_jit, backend = _fast_backend()
    program = _fast_program(q_flat.shape[-1])
    _fast_first_call(program, device_arch(q_flat.device))
    if program == cake_jit.PROGRAM:
        out = backend.cake_dsa_sparse_attention_nondet(
            q_flat[..., :FAST_D_V],
            q_flat[..., FAST_D_V:],
            kv_flat[:, :FAST_D_V],
            kv_flat[:, FAST_D_V:],
            idx_flat,
            softmax_scale=sm_scale,
            num_segments=num_segments,
        )
    else:
        out = backend.cake_dsa_sparse_attention_nondet_nope(
            q_flat, kv_flat, idx_flat, softmax_scale=sm_scale, num_segments=num_segments
        )
    return meta.rows_to_layout(out)


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    layout: str = "thd",
    d_v: int = 512,
    workspace_bytes: int | None = None,
    num_segments: int | None = None,
) -> torch.Tensor:
    """Sparse MQA/MLA attention over top-k key indices (autograd; gradients for ``q``, ``kv``, ``attn_sink``).

    Args:
        q: BF16 queries, ``[T, H, d_qk]`` (``thd``) or ``[B, S, H, d_qk]`` (``bshd``).
        kv: BF16 latent keys/values, ``[T_kv, d_qk]`` / ``[T_kv, 1, d_qk]`` (``thd``) or ``[B, S_kv, d_qk]``.
        indices: int32 key indices per query, ``-1`` for empty slots; ``[T, topk]`` / ``[T, 1, topk]`` (``thd``) or
            batch-local ``[B, S, topk]`` (``bshd``). An index outside the slot's key rows is an empty slot.
        sm_scale: the model's softmax scale (required).
        attn_sink: optional FP32 per-head sink logit ``[H]`` (DeepSeek-V4).
        d_v: number of leading latent channels that form the value (``512``).
        workspace_bytes: the ``det`` backward's memory budget (default ``--cake-dsa-bwd-workspace-bytes``).
        num_segments: packed sequences in the key rows (``thd``; default 1; ``bshd``: the batch). The ``fast``
            backward plans its key-range passes from it; the results do not depend on it.

    Returns the BF16 attention output ``[..., H, d_v]`` in the input layout. The variant is the configured one
    (``det`` unless configured); a call ``fast`` does not serve (:func:`fast_unserved_reason`) runs ``det``, and so
    does a ``fast`` call while ``torch.use_deterministic_algorithms(True)`` is in effect.
    """
    sm_scale = _check_scale(sm_scale)
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("sparse attention expects BF16 q and kv")
    segments = int(num_segments) if num_segments is not None else (1 if layout == "thd" else int(q.shape[0]))
    if segments < 1:
        raise ValueError(f"num_segments must be >= 1, got {segments}")
    if config.variant(OP) == "fast":
        num_kv = kv.shape[0] if layout == "thd" else kv.shape[0] * kv.shape[1]
        reason = fast_unserved_reason(
            int(q.shape[-2]), int(q.shape[-1]), int(d_v), has_sink=attn_sink is not None, num_kv=int(num_kv)
        )
        if reason is None and torch.are_deterministic_algorithms_enabled():
            reason = _DETERMINISTIC_ALGORITHMS
        if reason is None:
            return _fast_sparse_attention(q, kv, indices, sm_scale, layout, segments)
        config.warn_runs_det(OP, reason)
    return SparseAttentionFunction.apply(q, kv, attn_sink, indices, sm_scale, layout, d_v, workspace_bytes)


# --------------------------------------------------------------------------- #
# Lightning indexer
# --------------------------------------------------------------------------- #


def indexer_unserved_reason(heads: int, head_dim: int) -> str | None:
    """Why the det indexer (logits, selected scores and backward) cannot run ``heads`` index heads of dimension
    ``head_dim`` (``None`` when it can)."""
    return kernels.indexer_unserved_reason(int(heads), int(head_dim))


def indexer_logits(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    layout: str = "thd",
) -> torch.Tensor:
    """Head-summed ReLU indexer logits (no autograd; the indexer gradient flows through :func:`indexer_topk_scores`).

    ``index_q`` BF16 ``[T, H, D]`` / ``[B, S, H, D]`` / ``[S, B, H, D]``, ``index_k`` BF16
    ``[T_kv, D]`` / ``[B, S_kv, D]`` / ``[S_kv, B, D]``, ``weights`` FP32 ``[T, H]`` (or
    ``[T, H, 1]``) / ``[B, S, H]`` / ``[S, B, H]``.  ``cu_seqlen_ks``/``cu_seqlen_ke``
    are int32 per-query key bounds ``[S]`` shared across the batch.  Returns FP32
    logits ``[T, T_kv]`` (``thd``) or ``[B, S, S_kv]``; keys outside a query's bounds
    are ``-inf``.
    """
    q_flat, k_flat, w_flat, meta = layouts.flatten_indexer_inputs(index_q, index_k, weights, layout)
    kernels.check_indexer(int(q_flat.shape[1]), int(q_flat.shape[2]))
    batch, seq_len, seq_len_kv = meta.batch, meta.seq_len, meta.seq_len_kv
    if cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
        raise ValueError(
            f"cu_seqlen_ks/ke must be [{seq_len}], got {tuple(cu_seqlen_ks.shape)} / {tuple(cu_seqlen_ke.shape)}"
        )
    if q_flat.dtype != torch.bfloat16 or k_flat.dtype != torch.bfloat16:
        raise TypeError("indexer expects BF16 index_q and index_k")
    ks = cu_seqlen_ks.to(torch.int32).contiguous()
    ke = cu_seqlen_ke.to(torch.int32).contiguous()
    logits = kernels.dsa_indexer_logits(
        q_flat, k_flat, w_flat, ks, ke, batch=batch, seq_len=seq_len, seq_len_kv=seq_len_kv
    )
    return logits.squeeze(0) if layout == "thd" else logits


def _flat_indexer_backward(q, k, w, indices, grad_scores, workspace_bytes):
    num_rows, heads, dim = q.shape
    num_kv = k.shape[0]
    topk = indices.shape[1]
    dq = torch.empty_like(q)
    dw = torch.empty(num_rows, heads, dtype=torch.float32, device=q.device)
    dk32 = torch.zeros(num_kv, dim, dtype=torch.float32, device=q.device)
    if num_rows == 0:
        return dq, dw, dk32
    plan = reduction.plan_backward(num_rows, topk, dim, num_kv, _budget(workspace_bytes), one_sort=False)
    partial = torch.empty(plan.rows_per_chunk * topk, dim, dtype=torch.float32, device=q.device)
    for start, end in plan.chunks:
        part = partial[: (end - start) * topk]
        kernels.dsa_indexer_backward_rows(
            q[start:end],
            k,
            w[start:end],
            indices[start:end],
            grad_scores[start:end],
            dq[start:end],
            dw[start:end],
            part,
        )
        reduction.deterministic_scatter_add(part, indices[start:end].reshape(-1), dk32, split=INDEXER_REDUCE_SPLIT)
    return dq, dw, dk32


def indexer_backward(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    *,
    layout: str = "thd",
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward only: ``(d_index_q BF16, d_weights FP32, d_index_k FP32)`` in the input layout.

    ``topk_indices`` and ``grad_scores`` are ``[T, topk]`` (``thd``) or batch-major
    ``[B, S, topk]`` for both batched layouts; ``-1`` slots (and indices past the key rows)
    contribute nothing, whatever their ``grad_scores``.
    """
    q_flat, k_flat, w_flat, meta = layouts.flatten_indexer_inputs(index_q, index_k, weights, layout)
    kernels.check_indexer(int(q_flat.shape[1]), int(q_flat.shape[2]))
    idx_flat = layouts.pad_topk(
        layouts.flatten_topk_indices(topk_indices, meta), kernels.indexer_key_block(q_flat.shape[1])
    )
    grad_flat = layouts.flatten_scores(grad_scores, meta)
    if grad_flat.shape[1] != idx_flat.shape[1]:
        grad_flat = torch.nn.functional.pad(grad_flat, (0, idx_flat.shape[1] - grad_flat.shape[1]), value=0.0)
    # Empty slots carry no gradient (their upstream scores are -inf, so grad may be NaN / inf there):
    # the rows kernel loads an exact 0.0 for idx == -1, so no host-side masking launch is needed.
    dq, dw, dk = _flat_indexer_backward(q_flat, k_flat, w_flat, idx_flat, grad_flat.contiguous(), workspace_bytes)
    dk = meta.kv_rows_to_layout(dk)
    if layout == "thd" and index_k.dim() == 3:
        dk = dk.unsqueeze(1)
    dw = meta.rows_to_layout(dw)
    if layout == "thd" and weights.dim() == 3:
        dw = dw.unsqueeze(-1)
    return meta.rows_to_layout(dq), dw, dk


_BITS = (1, 2, 4, 8, 16, 32, 64, 128)


def _pack_bits(mask: torch.Tensor) -> torch.Tensor:
    """A bool ``[..., n]`` mask as uint8 ``[..., ceil(n / 8)]``, eight slots per byte."""
    padded = torch.nn.functional.pad(mask.to(torch.uint8), (0, -mask.shape[-1] % 8))
    bits = torch.tensor(_BITS, dtype=torch.uint8, device=mask.device)
    return (padded.view(*mask.shape[:-1], padded.shape[-1] // 8, 8) * bits).sum(dim=-1, dtype=torch.uint8)


def _unpack_bits(packed: torch.Tensor, n: int) -> torch.Tensor:
    """The bool ``[..., n]`` mask of :func:`_pack_bits`."""
    bits = torch.tensor(_BITS, dtype=torch.uint8, device=packed.device)
    return (packed.unsqueeze(-1) & bits).ne(0).reshape(*packed.shape[:-1], packed.shape[-1] * 8)[..., :n]


class IndexerScoresFunction(torch.autograd.Function):
    """Top-k indexer scores with the deterministic backward."""

    @staticmethod
    def forward(ctx, index_q, index_k, weights, topk_indices, logits, layout, workspace_bytes):
        scores = extract_topk_scores(logits, topk_indices)
        # A selected key outside its query's range has a -inf score (indexer_logits masks it): the score is constant,
        # so it has no gradient, whatever the upstream gradient there (a loss may give NaN at a -inf score). The
        # backward sees such a slot as empty; the mask is kept as bits (1/32 of the indices).
        ctx.save_for_backward(index_q, index_k, weights, topk_indices, _pack_bits(torch.isneginf(scores)))
        ctx.layout = layout
        ctx.workspace_bytes = _budget(workspace_bytes)
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        index_q, index_k, weights, topk_indices, empty = ctx.saved_tensors
        selected = topk_indices.to(torch.int32).masked_fill(_unpack_bits(empty, topk_indices.shape[-1]), -1)
        dq, dw, dk = indexer_backward(
            index_q,
            index_k,
            weights,
            selected,
            grad_scores,
            layout=ctx.layout,
            workspace_bytes=ctx.workspace_bytes,
        )
        return dq, dk, dw.to(weights.dtype), None, None, None, None


def indexer_topk_scores(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    logits: torch.Tensor,
    *,
    layout: str = "thd",
    workspace_bytes: int | None = None,
) -> torch.Tensor:
    """Gather the selected indexer scores from ``logits`` (autograd to ``index_q``, ``index_k``, ``weights``).

    ``logits`` is ``[T, T_kv]`` or ``[B, S, S_kv]`` as returned by :func:`indexer_logits`;
    ``topk_indices`` is ``[T, topk]`` or batch-local ``[B, S, topk]``.  Empty slots and selected keys
    outside their query's range yield ``-inf`` scores and receive no gradient.
    """
    heads = index_q.shape[-2]
    kernels.check_indexer(int(heads), int(index_q.shape[-1]))
    return IndexerScoresFunction.apply(index_q, index_k, weights, topk_indices, logits, layout, workspace_bytes)


def lightning_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    *,
    layout: str = "thd",
    topk_fn: TopkFn | None = None,
    topk_indices: torch.Tensor | None = None,
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indexer logits -> top-k selection -> differentiable selected scores.

    ``topk_fn`` is any ``fn(logits, topk)`` callable (a replay-manager wrapper, for
    instance; :func:`torch_topk` by default); ``topk_indices`` bypasses selection
    entirely (rollout replay).  Returns ``(scores, topk_indices)``: ``[T, topk]``
    (``thd``) or batch-major ``[B, S, topk]``.
    """
    logits = indexer_logits(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, layout=layout)
    if topk_indices is None:
        topk_indices = select_topk(logits, topk, topk_fn)
    scores = indexer_topk_scores(
        index_q,
        index_k,
        weights,
        topk_indices,
        logits,
        layout=layout,
        workspace_bytes=workspace_bytes,
    )
    return scores, topk_indices
