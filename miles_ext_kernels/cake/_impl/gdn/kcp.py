"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Context parallelism of the GDN kernels on the vendored Cake KCP packages.

KCP is the packages' context parallelism: every rank of the context-parallel group holds fragments of the documents of
one packed batch, and the ranks pass affine summaries of the per-document recurrent state (FP32 / TF32x3 class);
tokens never move between ranks. :func:`build_kcp_context` builds one rank's :class:`KcpContext` from the global
packed boundaries on the host and the process group. ``ops.chunk_gated_delta_rule(..., cp_context=<KcpContext>)`` and
``ops.kcp_chunk_gated_delta_rule`` run on it the KCP package of the configured variant:

* ``det``: ``cake_gdn_det_train_kcp``, bit for bit run to run for a fixed partition;
* ``fast``: ``cake_gdn_train_kcp``, within the det package's accuracy budgets, not its bits.

The packages share one partition contract, so both variants see the same local tokens; each builds its own segment
plan and device tables, on first use, once per context.

The partition (``layout``): ``"contiguous"`` gives rank ``r`` the ``r``-th of ``world_size`` near-equal token
intervals of the packed buffer (the first ``total % world_size`` one token longer), which is the layout FLA's context
parallelism uses when ``world_size`` divides the token count; ``"zigzag"`` and ``"zigzag_per_document"`` are the
packages' load-balanced layouts; ``fragments`` gives an explicit table. The caller lays the rank's tokens out in that
order; ``num_tokens`` and ``cu_seqlens`` / ``cu_seqlens_cpu`` (one local sequence per fragment) describe them.

A call that the KCP packages do not serve (see ``ops.chunk_gated_delta_rule``) runs ``cake_gdn_fla_cp_det``, the det
kernels of FLA's context parallelism, on the FLA context of :meth:`KcpContext.fla_context`, which exists for a
contiguous partition whose ``world_size`` divides the token count.
"""

from __future__ import annotations

import types
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import torch

__all__ = ["KCP_LAYOUTS", "KcpContext", "build_kcp_context", "release"]

KCP_LAYOUTS = ("contiguous", "zigzag", "zigzag_per_document")
# variant -> the name ``ops._package`` loads it by, and its partition / context builders
_PACKAGES = {
    "det": ("det_kcp", "partition_det_kcp", "build_context_det_kcp"),
    "fast": ("fast_kcp", "partition_kcp", "build_context_kcp"),
}


def _package(variant: str):
    from . import ops

    try:
        name = _PACKAGES[variant][0]
    except KeyError:
        raise ValueError(f"variant must be 'det' or 'fast', not {variant!r}") from None
    return ops._package(name)


def _host_boundaries(cu_seqlens) -> tuple[int, ...]:
    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.device.type != "cpu":
            raise ValueError("build_kcp_context: cu_seqlens_global must live on the host (no device read)")
        cu_seqlens = cu_seqlens.flatten().tolist()
    values = tuple(int(x) for x in cu_seqlens)
    if len(values) < 2 or values[0] != 0 or any(b < a for a, b in zip(values[:-1], values[1:], strict=True)):
        raise ValueError(f"build_kcp_context: cu_seqlens_global must be non-decreasing from 0, got {values[:8]}...")
    return values


class KcpContext:
    """One rank's view of a packed batch partitioned over a context-parallel group (:func:`build_kcp_context`).

    ``cu_seqlens_global`` (host ints), ``group`` (``None``: one rank), ``rank``, ``world_size``, ``device``,
    ``conv_kernel_size``, ``layout`` (``"explicit"`` with ``fragments``) describe the partition; ``num_tokens``,
    ``cu_seqlens_cpu`` (host ints) and ``cu_seqlens`` (int32 on ``device``) this rank's tokens, one local sequence per
    fragment. :meth:`partition` and :meth:`context` are the packages' own objects for a variant, built on first use;
    :meth:`fla_context` the FLA context of ``cake_gdn_fla_cp_det``. Share one context across the layers of a micro-batch:
    :func:`build_kcp_context` returns the same object for the same arguments.
    """

    def __init__(
        self,
        cu_seqlens_global: tuple[int, ...],
        group: Any,
        rank: int,
        world_size: int,
        device: torch.device,
        conv_kernel_size: int,
        layout: str,
        fragments: Any,
    ) -> None:
        self.cu_seqlens_global = cu_seqlens_global
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.conv_kernel_size = conv_kernel_size
        self.layout = layout
        self.fragments = fragments
        self._partitions: dict[str, Any] = {}
        self._contexts: dict[str, Any] = {}
        self._fla = None
        self.cu_seqlens_cpu: tuple[int, ...] = tuple(self.partition("det").local_cu_seqlens(rank))
        self.num_tokens = self.cu_seqlens_cpu[-1]
        self.cu_seqlens = torch.tensor(self.cu_seqlens_cpu, dtype=torch.int32).to(device, non_blocking=True)

    def partition(self, variant: str):
        """The partition object of ``variant``'s package (``PartitionDetKcp`` / ``PartitionKcp``)."""
        part = self._partitions.get(variant)
        if part is None:
            build = getattr(_package(variant), _PACKAGES[variant][1])
            if self.fragments is not None:
                part = build(self.cu_seqlens_global, self.world_size, fragments=self.fragments)
            else:
                part = build(self.cu_seqlens_global, self.world_size, self.layout)
            self._partitions[variant] = part
        return part

    def context(self, variant: str):
        """This rank's context of ``variant``'s package (``ContextDetKcp`` / ``ContextKcp``): its segment plan and
        device tables, built on first use. Raises the package's ``ValueError`` for a partition beyond its plan caps."""
        ctx = self._contexts.get(variant)
        if ctx is None:
            build = getattr(_package(variant), _PACKAGES[variant][2])
            ctx = build(self.partition(variant), self.group, self.rank, self.device, self.conv_kernel_size)
            self._contexts[variant] = ctx
        return ctx

    @property
    def fla_compatible(self) -> bool:
        """Whether this partition is FLA's: contiguous, with ``world_size`` dividing the token count."""
        return self.layout == "contiguous" and self.cu_seqlens_global[-1] % self.world_size == 0

    def fla_context(self):
        """The context of ``cake_gdn_fla_cp_det`` (``ops.cp_chunk_gated_delta_rule``) for the same tokens.

        One rank: a context without a process group. Several: ``fla.ops.cp.build_cp_context`` (needs
        flash-linear-attention). ``NotImplementedError`` unless :attr:`fla_compatible`.
        """
        if self._fla is not None:
            return self._fla
        if not self.fla_compatible:
            raise NotImplementedError(
                f"FLA's context parallelism needs a contiguous partition whose world_size divides the token count; this "
                f"context is {self.layout!r} with {self.cu_seqlens_global[-1]} tokens over {self.world_size} ranks"
            )
        if self.group is None:
            self._fla = types.SimpleNamespace(
                group=None, cu_seqlens=self.cu_seqlens, cu_seqlens_cpu=self.cu_seqlens_cpu
            )
        else:
            from fla.ops.cp import build_cp_context

            self._fla = build_cp_context(
                torch.tensor(self.cu_seqlens_global, dtype=torch.int32, device=self.device),
                self.group,
                conv1d_kernel_size=self.conv_kernel_size,
                cu_seqlens_cpu=torch.tensor(self.cu_seqlens_global, dtype=torch.int64),
            )
        return self._fla

    def __repr__(self) -> str:
        return (
            f"KcpContext(rank={self.rank}, world_size={self.world_size}, layout={self.layout!r}, "
            f"num_tokens={self.num_tokens}, documents={len(self.cu_seqlens_global) - 1}, device={self.device})"
        )


@lru_cache(maxsize=64)
def _cached(cu, group, rank, world_size, device, conv_kernel_size, layout, fragments) -> KcpContext:
    return KcpContext(cu, group, rank, world_size, device, conv_kernel_size, layout, fragments)


def _frozen(fragments) -> Any:
    if fragments is None:
        return None
    if isinstance(fragments, Sequence) and not isinstance(fragments, str):
        return tuple(_frozen(item) for item in fragments)
    return int(fragments)


def build_kcp_context(
    cu_seqlens_global,
    group: Any = None,
    *,
    device: Any = None,
    conv_kernel_size: int = 4,
    layout: str = "contiguous",
    fragments: Any = None,
) -> KcpContext:
    """This rank's :class:`KcpContext` of one packed batch: the context ``cp_context`` takes for KCP.

    ``cu_seqlens_global``: the packed documents' boundaries over the whole group, on the host (ints or a CPU tensor;
    Megatron's ``packed_seq_params.cu_seqlens_host`` before any context-parallel split). ``group``: the
    context-parallel ``torch.distributed`` process group (its rank and size are this rank's), ``None`` for one rank.
    ``device``: where the tables live (default: the current CUDA device). ``conv_kernel_size``: the short
    convolution's width, for the packages' halo plan. ``layout``: :data:`KCP_LAYOUTS`; ``fragments``: an explicit
    table per rank ``[[(doc, offset, length), ...], ...]`` in local order, instead of a layout.

    Host metadata only (no tensor read, no collective); the packages build their tables on first use. The same
    arguments give the same object (an LRU cache of 64), so every layer of a micro-batch shares one context.
    """
    if layout not in KCP_LAYOUTS:
        raise ValueError(f"build_kcp_context: layout must be one of {KCP_LAYOUTS}, not {layout!r}")
    if group is None:
        rank, world_size = 0, 1
    else:
        import torch.distributed as dist

        rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _cached(
        _host_boundaries(cu_seqlens_global),
        group,
        int(rank),
        int(world_size),
        device,
        int(conv_kernel_size),
        "explicit" if fragments is not None else layout,
        _frozen(fragments),
    )


def release() -> None:
    """Forget the cached contexts (:func:`build_kcp_context` builds a new one for the next layout)."""
    _cached.cache_clear()
