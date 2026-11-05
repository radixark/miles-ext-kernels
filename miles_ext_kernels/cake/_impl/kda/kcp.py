"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Context-parallel (KCP) packed KDA chunked training on the vendored Cake packages ``cake_kda_det_train_kcp``
(``det``) and ``cake_kda_train_kcp`` (``fast``), SM100a / SM103a / SM107a.

The packed batch of one step is split over the ranks of a ``torch.distributed`` process group, and the per-document
state recurrences cross the rank boundaries exactly::

    part = partition_kcp(cu_seqlens_global_cpu, world_size)   # the same object on every rank
    ctx = build_context_kcp(part, group, rank, device)         # this rank's tables and collectives
    out_local = chunk_kda_kcp(q_local, k_local, v_local, g_local, beta_local, ctx)   # autograd

:func:`partition_kcp` cuts the documents into per-rank fragments (``"contiguous"``, ``"zigzag"`` or
``"zigzag_per_document"`` layouts, or an explicit ``fragments`` table); :func:`build_context_kcp` gives a rank the
device tables of its fragments, the collectives of the state exchanges and the segment plan of its fragments (a
one-rank partition takes ``group=None``). A rank's operands are its shard: the tokens of its fragments in the
partition's order, with the shapes, dtypes and modes of the single-rank :func:`chunk_kda` (``beta`` rounded to BF16
and ``v`` made dense as there). Every rank of the group calls the same entry points in the same order: the
collectives run inside them.

A partition and a context hold no variant: a context serves the calls of both packages. Each call runs the package
of the call (``config.variant("kda")``; a ``fast`` call with the in-kernel q / k L2 norm, which the ``fast`` package
refuses, runs ``det`` and says so once); :func:`chunk_kda_bwd_kcp` runs the package that wrote ``saved``. ``det``
gives the same bits for the same inputs, partition and process group. ``fast``: in the activated-gate mode its ``dg``
may differ from ``det``'s by FP32 rounding; in the raw gate mode its parameter partial sums are FP64 sums in no fixed
order on SM100a / SM103a.

In the raw gate mode the gradients of ``A_log`` / ``dt_bias`` are this rank's partial sums (the gradients through its
own tokens): sum them over the group like the gradients of any replicated parameter, or with
:func:`reduce_gate_param_grads_kcp` for a sum with the same bits on every rank.

The libraries are built by ``miles_ext_kernels.cake.prebuild("kda", num_heads=...)``. Importing this module imports
torch but not the context-parallel packages: they load on the first call.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from . import ops

__all__ = [
    "build_context_kcp",
    "chunk_kda_bwd_kcp",
    "chunk_kda_fwd_kcp",
    "chunk_kda_kcp",
    "partition_kcp",
    "reduce_gate_param_grads_kcp",
]


def _det_kcp() -> Any:
    """``cake_kcp`` of the deterministic context-parallel package (its entry points and parameter reduce)."""
    return import_module(f"{ops._VENDOR}.cake_kda_det_train_kcp.cake_kcp")


def _det_plan() -> Any:
    """``cake_kcp_plan`` of the deterministic context-parallel package (the partition and the context)."""
    return import_module(f"{ops._VENDOR}.cake_kda_det_train_kcp.cake_kcp_plan")


def _fast_kcp() -> Any:
    """``cake_kcp`` of the non-deterministic context-parallel package."""
    return import_module(f"{ops._VENDOR}.cake_kda_train_kcp.cake_kcp")


def _gate(use_gate_in_kernel, A_log, dt_bias, lower_bound, use_qk_l2norm_in_kernel) -> dict[str, Any]:
    """The mode keywords of a call, the same for either package (the ``fast`` package refuses
    ``use_qk_l2norm_in_kernel`` other than ``False``: see ``ops._modes``)."""
    return dict(
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def partition_kcp(
    cu_seqlens_global_cpu: Any,
    world_size: int,
    layout: str = "contiguous",
    fragments: Any = None,
    *,
    cut_points: Any = None,
) -> Any:
    """The context-parallel partition of one packed batch over ``world_size`` ranks: per rank its fragments (document,
    token range). ``cu_seqlens_global_cpu`` is the whole batch's ``[num_docs + 1]`` document offsets on the host (a
    tensor or a sequence). ``layout``: ``"contiguous"`` (equal token shares in order), ``"zigzag"`` or
    ``"zigzag_per_document"``; ``fragments`` gives an explicit table instead, and ``cut_points`` (per document the
    ``2 * world_size + 1`` cut offsets) applies to ``"zigzag_per_document"``. The same object on every rank."""
    return _det_plan().partition_det_kcp(cu_seqlens_global_cpu, world_size, layout, fragments, cut_points=cut_points)


def build_context_kcp(
    part: Any,
    group: Any,
    rank: int,
    device: Any,
    conv_kernel_size: int = 4,
    *,
    sm_count: int | None = None,
    heads: Any = None,
    stream: Any = None,
) -> Any:
    """This rank's context of ``part``: the argument of every KCP entry point, reusable across the calls and layers of
    the same batch layout. ``group`` is the context-parallel process group (``None`` for a one-rank partition),
    ``rank`` this process's rank in it and ``device`` the CUDA device of its operands; ``conv_kernel_size - 1``
    history rows of the causal convolution cross the rank boundaries. ``heads`` (an int or a sequence) derives those
    head counts' plans now; ``stream`` takes the context's one asynchronous table copy (default: the current stream of
    ``device``), so a pack can be prepared while the previous one runs. No collective runs here."""
    return _det_plan().build_context_det_kcp(
        part, group, rank, device, conv_kernel_size, sm_count=sm_count, heads=heads, stream=stream
    )


def chunk_kda_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    **options: Any,
) -> torch.Tensor:
    """The context-parallel autograd node of this rank's fragments: ``out`` BF16 like ``v`` (this rank's tokens), with
    gradients to ``q, k, v, g, beta`` and, in the raw gate mode, to ``A_log`` / ``dt_bias`` (this rank's partial
    sums). Compatible with ``torch.utils.checkpoint``. ``options`` go to the package of the call (``save_mode``,
    ``boundary_mode``, ``workspace``, ``recompute_handle``, ``checkpoint_keep``, ``save_transfer``, ``exchange``,
    ``graph``); a workspace serves its own package only."""
    ops.check_heads(int(v.shape[-2]))
    v, beta = ops.seam(v, beta)
    gate = _gate(use_gate_in_kernel, A_log, dt_bias, lower_bound, use_qk_l2norm_in_kernel)
    if ops.call_variant(use_qk_l2norm_in_kernel) == "fast":
        return _fast_kcp().chunk_kda_kcp(q, k, v, g, beta, ctx, scale=scale, deterministic=False, **gate, **options)
    return _det_kcp().chunk_kda_det_kcp(q, k, v, g, beta, ctx, scale=scale, **gate, **options)


def chunk_kda_fwd_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: float | None = None,
    save_for_backward: bool = True,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    **options: Any,
) -> tuple[torch.Tensor, Any]:
    """The context-parallel forward alone: ``(out, saved)``, ``saved`` the set :func:`chunk_kda_bwd_kcp` takes
    (``None`` without ``save_for_backward``). Pass the backward the same operands and modes."""
    ops.check_heads(int(v.shape[-2]))
    v, beta = ops.seam(v, beta)
    gate = _gate(use_gate_in_kernel, A_log, dt_bias, lower_bound, use_qk_l2norm_in_kernel)
    common = dict(scale=scale, save_for_backward=save_for_backward, **gate)
    if ops.call_variant(use_qk_l2norm_in_kernel) == "fast":
        return _fast_kcp().chunk_kda_fwd_kcp(q, k, v, g, beta, ctx, deterministic=False, **common, **options)
    return _det_kcp().chunk_kda_det_fwd_kcp(q, k, v, g, beta, ctx, **common, **options)


def chunk_kda_bwd_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    dout: torch.Tensor,
    *,
    saved: Any = None,
    scale: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    **options: Any,
) -> tuple[torch.Tensor, ...]:
    """The context-parallel backward alone, from the forward's operands and modes: ``(dq, dk, dv, dbeta, dg)``, and
    ``(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)`` in the raw gate mode (``dA_log`` / ``ddt_bias`` this rank's partial
    sums); ``dbeta`` in ``beta``'s dtype. It runs the package that wrote ``saved`` (a saved set is read by its own
    package only; a ``fast`` set with ``use_qk_l2norm_in_kernel`` other than ``False``, a mode its forward did not
    run, is a ``ValueError`` of that package); ``saved=None`` replays the forward, its exchanges included, in the
    package of the call."""
    ops.check_heads(int(v.shape[-2]))
    beta_dtype = beta.dtype
    v, beta = ops.seam(v, beta)
    gate = _gate(use_gate_in_kernel, A_log, dt_bias, lower_bound, use_qk_l2norm_in_kernel)
    common = dict(saved=saved, scale=scale, **gate)
    if saved is None:
        variant = ops.call_variant(use_qk_l2norm_in_kernel)
    else:
        variant = "fast" if isinstance(saved, _fast_kcp().SavedKcp) else "det"
    if variant == "fast":
        grads = _fast_kcp().chunk_kda_bwd_kcp(q, k, v, g, beta, ctx, dout, deterministic=False, **common, **options)
    else:
        grads = _det_kcp().chunk_kda_det_bwd_kcp(q, k, v, g, beta, ctx, dout, **common, **options)
    grads = list(grads)
    grads[3] = grads[3].to(beta_dtype)
    return tuple(grads)


def reduce_gate_param_grads_kcp(
    ctx: Any, dA_log: torch.Tensor, ddt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The raw gate mode's full ``(dA_log, ddt_bias)`` from every rank's partial sums: one all-gather on the context's
    group, then a sum in FP32 in rank order, so every rank gets the same bits whatever the collective's algorithm (an
    all-reduce does not fix its summation order). For the partial sums of either variant. Every rank of the group
    calls it with the same shapes."""
    return _det_kcp().reduce_gate_param_grads_det_kcp(ctx, dA_log, ddt_bias)
