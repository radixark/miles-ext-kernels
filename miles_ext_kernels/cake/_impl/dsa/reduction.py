"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Deterministic scatter-add of per-slot FP32 partials into key rows, and the memory plan of the det backward.

Given ``partial[p, :]`` for pairs ``p`` and the key row ``keys[p]`` each pair belongs to (``-1``, or any value outside
``[0, num_out)``, for a slot that is a hole), :func:`deterministic_scatter_add` computes
``out[k, :] += sum_{p : keys[p] == k} partial[p, :]`` with a summation order that is a pure function of ``keys``:
pairs are stably sorted by key, so ties keep their pair order, and each key row is added either sequentially
(``split=1``) or as a fixed 4-way tree over its segment (``split=4``: four contiguous quarters summed in pair order,
then combined in quarter order). Repeated runs give bit-identical results; the atomics they replace do not. No host
synchronisation is needed: the segment bounds come from ``searchsorted`` on the sorted keys.

Memory (:func:`plan_backward`). The det backward runs query rows in chunks and reduces the chunks in order, which keeps
the accumulation deterministic. One budget (``--cake-dsa-bwd-workspace-bytes``) bounds its transient memory: the FP32
partials of a chunk and the index scratch of the sort that orders them (the sort keys, the sort order, the radix
sort's buffers and the segment bounds). The attention backward sorts the keys of every chunk at once when that fits
(:class:`ChunkedSegments`: one sort instead of one per chunk), else chunk by chunk; for the same chunking both give the
same segments, so the same bits. The chunking follows from the budget and the shape, so the det bits do too: identical
inputs and an identical budget repeat bit for bit.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

DEFAULT_WORKSPACE_BYTES = 2 << 30
# Index scratch of a stable sort of int32 keys, per pair, at its peak: the keys and their copy, the sorted keys, the
# int64 order and the iota it starts from, the radix sort's temporary keys and values, the int32 order kept for the
# reduction. An upper bound of what torch.sort and the composite key build allocate (36 bytes measured from the
# allocations it makes, rounded up).
SORT_BYTES_PER_PAIR = 48
# The same with int64 composite keys (more than 2**31 composites: key rows x chunks).
SORT_BYTES_PER_PAIR_INT64 = 64
# Index scratch kept per pair while the chunks run after one sort of every pair: the sorted composites and the int32
# order (int64 composites: 12).
KEPT_BYTES_PER_PAIR = 8
KEPT_BYTES_PER_PAIR_INT64 = 12
# Segment bounds of one chunk, per key row: the probe, the starts and the ends (int64 at most) and searchsorted's own.
BOUNDS_BYTES_PER_KEY = 32


class BackwardPlan(NamedTuple):
    """How the det backward runs: ``chunks`` (``(start, end)`` query rows, in order) and whether the attention
    backward sorts every pair at once (``one_sort``) or each chunk's pairs on their own."""

    chunks: list[tuple[int, int]]
    one_sort: bool

    @property
    def rows_per_chunk(self) -> int:
        return self.chunks[0][1] - self.chunks[0][0] if self.chunks else 0


def _chunks(num_rows: int, rows: int) -> list[tuple[int, int]]:
    rows = max(1, min(num_rows, rows))
    return [(start, min(num_rows, start + rows)) for start in range(0, num_rows, rows)]


def plan_backward(
    num_rows: int, pairs_per_row: int, dim: int, num_out: int, workspace_bytes: int, *, one_sort: bool = True
) -> BackwardPlan:
    """The chunking of a det backward of ``num_rows`` query rows within ``workspace_bytes``.

    Each row writes ``pairs_per_row`` FP32 partial rows of ``dim`` channels, reduced into ``num_out`` key rows.
    ``one_sort`` allows one sort of every pair (the attention backward); it is taken when its scratch fits the budget
    with at least one row of partials, else each chunk sorts its own pairs. A budget below one row still runs one row
    at a time.
    """
    if num_rows <= 0:
        return BackwardPlan([], False)
    partial_row = pairs_per_row * dim * 4
    bounds = num_out * BOUNDS_BYTES_PER_KEY
    if one_sort:
        pairs = num_rows * pairs_per_row
        wide = num_out * num_rows >= 2**31  # at most num_rows chunks: the composites may need int64
        sort_bytes = pairs * (SORT_BYTES_PER_PAIR_INT64 if wide else SORT_BYTES_PER_PAIR)
        kept = pairs * (KEPT_BYTES_PER_PAIR_INT64 if wide else KEPT_BYTES_PER_PAIR)
        rows = (workspace_bytes - kept - bounds) // max(partial_row, 1)
        if sort_bytes <= workspace_bytes and rows >= 1:
            return BackwardPlan(_chunks(num_rows, rows), True)
    rows = (workspace_bytes - bounds) // max(partial_row + pairs_per_row * SORT_BYTES_PER_PAIR, 1)
    return BackwardPlan(_chunks(num_rows, rows), False)


def segment_pairs(keys: torch.Tensor, num_out: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable-sort pair keys; return ``(order, seg_start, seg_count)`` indexed by key row.

    ``order[p]`` is the pair index at sorted position ``p``; key ``k`` owns sorted positions
    ``[seg_start[k], seg_start[k] + seg_count[k])``. Pairs whose key is outside ``[0, num_out)`` (``-1`` pads) sort
    first and belong to no key. Everything is computed on the device without a host synchronisation or atomics
    (``searchsorted`` on the sorted keys).
    """
    if keys.dim() != 1:
        raise ValueError("keys must be a flat [num_pairs] tensor")
    keys32 = keys.to(torch.int32)
    keys32 = torch.where((keys32 < 0) | (keys32 >= num_out), -1, keys32)
    # Narrow sort keys when every key (and the probe ``num_out``) fits int16: half the radix passes,
    # the same stable order (the cast is monotonic on [-1, num_out]).
    sort_keys = keys32.to(torch.int16) if num_out + 1 <= 32767 else keys32
    sorted_keys, order = torch.sort(sort_keys, stable=True)
    # One search over ``0 .. num_out`` gives every segment's start and (as the next start) its end;
    # the final probe ``num_out`` lands past the last key.
    probe = torch.arange(num_out + 1, device=keys.device, dtype=sort_keys.dtype)
    bounds = torch.searchsorted(sorted_keys, probe, out_int32=True)
    return order.to(torch.int32), bounds[:-1], (bounds[1:] - bounds[:-1]).contiguous()


def deterministic_scatter_add(partial: torch.Tensor, keys: torch.Tensor, out: torch.Tensor, *, split: int = 1) -> None:
    """``out[keys[p], :] += partial[p, :]`` in a fixed order (see module docstring).

    ``partial`` is ``[num_pairs, dim]`` FP32, ``keys`` ``[num_pairs]`` integer with ``-1`` (or any value outside
    ``out``'s rows) for pairs to skip, ``out`` ``[num_out, dim]`` FP32 (updated in place). ``split`` selects the
    per-key reduction order: ``1`` sequential, ``4`` the fixed 4-way tree (``dsa_segmented_reduce_split4``).
    """
    from . import kernels

    if partial.dim() != 2 or out.dim() != 2 or partial.shape[1] != out.shape[1]:
        raise ValueError("partial and out must be [*, dim] with the same dim")
    if partial.shape[0] != keys.numel():
        raise ValueError("keys must have one entry per partial row")
    if partial.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("deterministic scatter-add works on FP32 partials and outputs")
    if partial.shape[1] % 4 != 0:
        raise ValueError(f"segmented reduce needs dim divisible by 4, got {partial.shape[1]}")
    if keys.numel() == 0 or out.shape[0] == 0:
        return
    order, seg_start, seg_count = segment_pairs(keys, int(out.shape[0]))
    kernels.dsa_segmented_reduce_launch(partial.contiguous(), order, seg_start, seg_count, out, split=split)


class ChunkedSegments:
    """Per-chunk reducer segments from ONE global stable sort (the chunk loop of the attention backward).

    The attention backward processes query rows in chunks of ``rows_per_chunk`` and reduces each chunk's partials
    ``partial[(row_local * num_head_blocks + hb) * topk + slot, :]`` into ``out[key, :]``. Instead of sorting each
    chunk's keys (about 14 launches per chunk), the composite key ``key * num_chunks + chunk`` of every pair is stably
    sorted once. Within one ``(key, chunk)`` segment the pairs then appear in ascending global pair order -- exactly the
    sequence the per-chunk stable sort produces -- so the reduction is bit-identical to the per-chunk plan for the same
    chunking. Pairs whose key is outside ``[0, num_out)`` get the composite ``-1`` and sort before every segment.
    ``order_local`` is the pair index inside its chunk (``order % pairs_per_chunk``: every chunk but the last holds
    exactly ``pairs_per_chunk`` pairs), the row index the reducer uses into the chunk's partial buffer. The segment
    bounds of a chunk come from two ``searchsorted`` launches over the sorted composites (:meth:`segment`), so the
    kept scratch is the sorted composites and the order, not a table per chunk.
    """

    def __init__(self, indices: torch.Tensor, num_head_blocks: int, rows_per_chunk: int, num_out: int):
        num_rows, topk = indices.shape
        device = indices.device
        num_chunks = (num_rows + rows_per_chunk - 1) // rows_per_chunk
        comp_dtype = torch.int32 if num_out * num_chunks < 2**31 else torch.int64
        chunk_of_row = torch.arange(num_rows, device=device, dtype=comp_dtype) // rows_per_chunk
        comp = indices.to(comp_dtype) * num_chunks + chunk_of_row[:, None]
        comp = torch.where((indices < 0) | (indices >= num_out), -1, comp)
        comp = comp.unsqueeze(1).expand(num_rows, num_head_blocks, topk).reshape(-1)
        sorted_comp, order = torch.sort(comp, stable=True)
        del comp
        pairs_per_chunk = rows_per_chunk * num_head_blocks * topk
        self.order_local = (order % pairs_per_chunk).to(torch.int32)
        del order
        self.sorted_comp = sorted_comp
        self.key_probe = torch.arange(num_out, device=device, dtype=comp_dtype) * num_chunks
        self.num_chunks = num_chunks

    def segment(self, chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(order_local, seg_start, seg_count)`` of ``chunk`` (int32; two searchsorted launches)."""
        probe = self.key_probe + chunk
        starts = torch.searchsorted(self.sorted_comp, probe, out_int32=True)
        ends = torch.searchsorted(self.sorted_comp, probe, right=True, out_int32=True)
        return self.order_local, starts, ends - starts
