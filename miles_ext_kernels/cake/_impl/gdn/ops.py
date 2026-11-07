"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

``chunk_gated_delta_rule`` (forward + backward) for SM100a / SM103a / SM107a on five vendored packages.

:func:`chunk_gated_delta_rule` is a drop-in replacement for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` (FLA
v0.5.2) as the Qwen GDN wrappers call it: ``[B, T, H, K]`` inputs, packed varlen through ``cu_seqlens``, in-kernel q/k
L2 normalisation, optional FP32 initial/final state, and context parallelism through a ``cp_context``: a
:class:`~miles_ext_kernels.cake._impl.gdn.kcp.KcpContext` (:func:`~miles_ext_kernels.cake._impl.gdn.kcp.build_kcp_context`)
or an FLA ``FLACPContext``.  ``K = V = 128``, ``HV % H == 0`` (grouped value heads without materialised
``repeat_interleave``), BF16 q/k/v.  A call reads the variant (``config.variant("gdn")``) and
``--cake-gdn-recompute-state`` when it runs; the backward uses what its forward read.  It runs:

=============================================  ==========================  =============================
call                                           det                         fast
=============================================  ==========================  =============================
no ``cp_context``                              ``cake_gdn_det_train``      ``cake_gdn_nondet_train``
a ``KcpContext``                               ``cake_gdn_det_train_kcp``  ``cake_gdn_nondet_train_kcp``
an FLA context, or a ``KcpContext`` call the   ``cake_gdn_fla_cp_det``     ``cake_gdn_fla_cp_det``
KCP packages do not serve                                                  (logged once)
=============================================  ==========================  =============================

* ``cake_gdn_det_train`` (``cake_gdn_det``): both passes repeat bit for bit call to call (no atomic on a value, a fixed
  reduction order, a launch plan that is a pure function of the layout and the device).
* ``cake_gdn_nondet_train`` (``cake_gdn_nondet``): within the det package's accuracy budgets.
* ``cake_gdn_det_train_kcp`` (``cake_gdn_det_kcp``): bit for bit run to run for a fixed partition;
  ``cake_gdn_nondet_train_kcp`` (``cake_gdn_nondet_kcp``) within its accuracy budgets.  Their contract: FP32 g and beta
  (a BF16 beta is cast exactly, and autograd rounds the FP32 ``dbeta`` back), no initial or final state.
* ``cake_gdn_fla_cp_det``: the det kernels with FLA's context parallelism, generated from the flash-linear-attention
  v0.5.2 ``chunk_gated_delta_rule`` dataflow (chunk 64, K = V = 128, in-kernel q/k L2 normalisation, FLA's
  context-parallel state passing): register-MMA kernels with FP32 inter-chunk state carriers, BF16 hand-offs at the
  reference's rounding points and fixed-order reductions without atomics.  They mirror FLA's chunk-64 decomposition one
  to one (prep = L2 norm + gate cumsum, WY solve, state scan, output; backward: dv_local, adjoint state scan,
  dq/dk/dw/dg, WY pullback, finalize) and recompute the WY / state tensors exactly in the backward, as FLA does, so
  ``--cake-gdn-recompute-state`` gives the same gradient bits.  Context parallelism reuses FLA v0.5.2's ``fla.ops.cp``
  pre-processing exactly (``chunk_gated_delta_rule_fwd_h_pre_process``, ``compress_h0``, ``expand_h0``,
  ``chunk_gated_delta_rule_bwd_dhu_pre_process``); those helpers are imported only when a ``cp_context`` with a process
  group is supplied.

Scope of ``det``: the bits repeat for identical inputs (the whole micro-batch layout: the launch plan is a pure function
of every segment length and the device) on the same GPU model and SM count, toolkit and build, the same route (no
``cp_context``, a ``KcpContext``, an FLA context; the same partition; host or device boundaries) and the same
``--cake-gdn-recompute-state``.  Across those the plan may split the state scan differently (segmented and serial
scans), and the results agree within the accuracy budget only.

The context-parallel contract is FLA's: with any ``cp_context`` the call takes no ``initial_state`` and returns no final
state (rejected when the call is made), and ``cu_seqlens`` / ``cu_seqlens_cpu`` come from the context (the arguments are
ignored, as FLA does).

``--cake-gdn-recompute-state`` keeps the WY / state tensors (``w``, ``u``, ``h``, ``v_new``) off the autograd tape and
recomputes them in the backward, on every package: the single-rank packages take it as ``save_state``, the KCP packages
as ``save_mode`` (``"full"`` keeps the tapes, ``"light"`` replays them).

Host wiring of ``cake_gdn_fla_cp_det``: the chunk table is a pure function of the micro-batch's segment lengths, so it is
built once per distinct layout from the host copy of ``cu_seqlens`` (``cu_seqlens_cpu``; no device-to-host
synchronisation, no launch) and shared by every layer and by the activation-recompute forward; a device-only
``cu_seqlens`` builds it with the on-device table kernel.  The plan pins (``FORCED_*``) of this module apply to these
kernels only; the other packages plan with their own defaults.  Scratch buffers of the det and fast packages and of
these kernels go through the runtime's workspace pool (``workspace.take("gdn", role, ...)``): a fresh allocation per
call at the default bound 0; under a positive ``--cake-workspace-pool-bytes`` one buffer per role and stream, freed by
``release()``.  The 0-element state placeholders (no initial or final state) are one tensor per device, shape and
dtype, which holds no memory.  :func:`prebuild` builds every library of the five packages for a device's architecture.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader, registry, workspace

from . import native
from .kcp import KcpContext
from .native import MODULES, kernel, launcher

__all__ = [
    "ChunkGatedDeltaRuleFunction",
    "ChunkMeta",
    "chunk_gated_delta_rule",
    "chunk_meta_from_host",
    "cp_chunk_gated_delta_rule",
    "gdn_chunk_backward",
    "gdn_chunk_forward",
    "kcp_chunk_gated_delta_rule",
    "prebuild",
    "release",
    "scan_plans",
    "ScanPlan",
]

OP = "gdn"
logger = logging.getLogger("miles_ext_kernels")
CHUNK = 64
HEAD_DIM = 128
# Register-scan configurations (value rows per CTA, K split) in preference order; every warp owns
# 16 value rows and K / ksplit keys.  Must match the exported ``fwd_h_r*``/``dhu_r*`` stages.
REG_SCAN_CONFIGS = ((32, 2), (32, 1), (64, 1))
# Tests may pin a scan configuration of the generated kernels; production always derives it from shape and device.
FORCED_SCAN_CONFIG: tuple[int, int] | None = None
# Options of the KCP packages' calls (``CakeGDNDetKcpOptions`` / ``CakeGDNNondetKcpOptions`` fields, e.g. ``segments``
# or the torch emulation ``backend="fake"``) for tests and experiments; production passes none and runs the packages'
# defaults.
KCP_OPTIONS: dict[str, Any] | None = None


def _save_state() -> bool:
    """Keep w / u / h / v_new on the autograd tape (unless ``--cake-gdn-recompute-state``); read when a call runs."""
    return not config.option(OP, "cake_gdn_recompute_state")


# --------------------------------------------------------------------------- #
# chunk table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScanPlan:
    """Launch plan of one state scan: the register-scan configuration ``(value rows per CTA, K split)`` and the chunks
    per segment (``0``: one segment per sequence, the serial scan).  A pure function of the layout and the device
    (:func:`choose_scan_plan`); the segment table built from it carries the configuration, so the launch runs exactly
    the stages the plan was priced for on both table routes.
    """

    config: tuple[int, int]
    segment_len: int


@dataclass(frozen=True)
class SegmentTable:
    """Scan segments: every sequence's chunk range cut into near-equal pieces (all int32 CUDA tensors).

    Segment ``s`` covers chunks ``[seg_chunk_start[s], seg_chunk_start[s + 1])`` of sequence ``seg_seq[s]``;
    the segments of sequence ``i`` are ``[seq_seg_start[i], seq_seg_start[i + 1])`` in chunk order.
    ``num_segments`` is the launch extent: a table built on the device is padded to its upper bound with
    empty segments (``seg_seq == -1``) that the scan kernels skip.  ``segment_len == 0`` means one segment
    per sequence (the serial scan).  ``scan_config`` is the register-scan configuration ``(value rows per CTA,
    K split)`` the table was planned for (:class:`ScanPlan`).
    """

    seg_chunk_start: torch.Tensor
    seg_seq: torch.Tensor
    seq_seg_start: torch.Tensor
    num_segments: int
    segment_len: int
    scan_config: tuple[int, int]


@dataclass(frozen=True)
class ChunkMeta:
    """Chunk table for packed varlen sequences (all int32 CUDA tensors).

    ``num_chunks`` is the launch extent; when the table is built on the device from ``cu_seqlens``
    it is an upper bound (``T // 64 + num_seqs``) and the trailing entries are padding chunks with
    ``chunk_len == 0`` that every chunk kernel skips.  ``fwd_segments`` / ``bwd_segments`` are the
    segment tables of the forward state scan and of the backward state-gradient scan
    (:func:`scan_plans`; ``segment_len == 0`` keeps the serial scan).
    """

    chunk_start: torch.Tensor
    chunk_len: torch.Tensor
    seq_chunk_start: torch.Tensor
    num_chunks: int
    num_seqs: int
    seq_lens: tuple[int, ...] | None
    fwd_segments: SegmentTable
    bwd_segments: SegmentTable


def segment_bounds(num_chunks: int, segment_len: int) -> list[int]:
    """Segment boundaries of one sequence (chunk offsets), balanced to within one chunk.

    The device table builder (the ``meta`` stage) evaluates the same integer formula.
    """
    if segment_len <= 0 or num_chunks <= segment_len:
        return [0, num_chunks]
    count = -(-num_chunks // segment_len)
    return [(s * num_chunks) // count for s in range(count + 1)]


def _segment_lists(seq_chunk_start: list[int], segment_len: int) -> tuple[list[int], list[int], list[int]]:
    seg_chunk_start = [0]
    seg_seq: list[int] = []
    seq_seg_start = [0]
    for seq in range(len(seq_chunk_start) - 1):
        c0 = seq_chunk_start[seq]
        bounds = segment_bounds(seq_chunk_start[seq + 1] - c0, segment_len)
        seg_chunk_start.extend(c0 + b for b in bounds[1:])
        seg_seq.extend([seq] * (len(bounds) - 1))
        seq_seg_start.append(len(seg_seq))
    return seg_chunk_start, seg_seq, seq_seg_start


def build_chunk_meta(seq_lens, device, *, scan_plans: tuple[ScanPlan, ScanPlan]) -> ChunkMeta:
    """Exact chunk and segment tables from host sequence lengths (one pinned copy, no synchronisation).

    ``scan_plans`` are the layout's forward and backward scan plans (:func:`scan_plans`).
    """
    starts: list[int] = []
    lens: list[int] = []
    seq_chunk_start = [0]
    token = 0
    for seq_len in seq_lens:
        seq_len = int(seq_len)
        for chunk_start in range(0, seq_len, CHUNK):
            starts.append(token + chunk_start)
            lens.append(min(CHUNK, seq_len - chunk_start))
        seq_chunk_start.append(len(starts))
        token += seq_len
    parts: list[list[int]] = [starts, lens, seq_chunk_start]
    for plan in scan_plans:
        parts.extend(_segment_lists(seq_chunk_start, int(plan.segment_len)))

    # One host tensor, one asynchronous copy (pinned staging): a pageable ``torch.tensor(..., device="cuda")``
    # would synchronise the stream before each copy.
    table = torch.tensor([x for part in parts for x in part], dtype=torch.int32)
    device = torch.device(device)
    table = table.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else table.to(device)
    views = []
    offset = 0
    for part in parts:
        views.append(table[offset : offset + len(part)])
        offset += len(part)
    segments = [
        SegmentTable(
            seg_chunk_start=views[3 + 3 * i],
            seg_seq=views[4 + 3 * i],
            seq_seg_start=views[5 + 3 * i],
            num_segments=len(parts[4 + 3 * i]),
            segment_len=int(scan_plans[i].segment_len),
            scan_config=tuple(scan_plans[i].config),
        )
        for i in range(2)
    ]
    return ChunkMeta(
        chunk_start=views[0],
        chunk_len=views[1],
        seq_chunk_start=views[2],
        num_chunks=len(starts),
        num_seqs=len(seq_lens),
        seq_lens=tuple(int(x) for x in seq_lens),
        fwd_segments=segments[0],
        bwd_segments=segments[1],
    )


_HOST_META_BUILDS = 0  # chunk tables built from host boundaries (tests assert one per micro-batch layout)


def _seq_lens_from_host(cu_seqlens_host) -> tuple[int, ...]:
    """Segment lengths from a host copy of ``cu_seqlens``: a tuple / list of ints or a CPU integer tensor."""
    if isinstance(cu_seqlens_host, torch.Tensor):
        if cu_seqlens_host.is_cuda:
            raise ValueError("cu_seqlens_cpu must live on the host")
        values = cu_seqlens_host.flatten().tolist()
    else:
        values = list(cu_seqlens_host)
    values = [int(x) for x in values]
    if len(values) < 2:
        raise ValueError("cu_seqlens must have at least two entries")
    if values[0] != 0 or any(b < a for a, b in zip(values[:-1], values[1:], strict=True)):
        raise ValueError(f"cu_seqlens must be non-decreasing and start at 0, got {values[:8]}...")
    return tuple(b - a for a, b in zip(values[:-1], values[1:], strict=True))


@lru_cache(maxsize=256)
def _host_chunk_meta(seq_lens: tuple[int, ...], scan_plans: tuple[ScanPlan, ScanPlan], device_index: int) -> ChunkMeta:
    global _HOST_META_BUILDS
    _HOST_META_BUILDS += 1
    return build_chunk_meta(seq_lens, torch.device("cuda", device_index), scan_plans=scan_plans)


def _host_meta(seq_lens, num_v_heads: int, device_index: int, arch: str, total_tokens: int | None = None) -> ChunkMeta:
    # The scan plans (configuration + segment length) are part of the layout key; ``total_tokens`` is the padded
    # token extent the device table path sees (``T``) so both paths derive identical tables.
    seq_lens = tuple(int(x) for x in seq_lens)
    tokens = int(total_tokens) if total_tokens is not None else sum(seq_lens)
    plans = scan_plans(tokens, len(seq_lens), int(num_v_heads), int(device_index), arch, seq_lens)
    return _host_chunk_meta(seq_lens, plans, int(device_index))


def chunk_meta_from_host(
    cu_seqlens_host, device, num_v_heads: int, total_tokens: int | None = None, arch: str | None = None
) -> ChunkMeta:
    """Exact chunk (and, given ``num_v_heads``, scan segment) tables for a micro-batch layout, built once per
    distinct layout and device.

    The tables are a pure function of the scan plans, the head count and the device, so every layer of a
    micro-batch (and the activation-recompute forward) shares one build: a few tiny int32 tensors, no
    device-to-host synchronisation, no launch, exact launch extents.
    """
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if arch is None:
        arch = loader.device_arch(device)
    return _host_meta(_seq_lens_from_host(cu_seqlens_host), int(num_v_heads), int(index), arch, total_tokens)


def segments_upper_bound(num_chunks_max: int, num_seqs: int, segment_len: int) -> int:
    """Launch extent of a device-built segment table: ``sum(ceil(n_i / L)) <= num_chunks_max // L + num_seqs``."""
    return num_seqs + (num_chunks_max // int(segment_len) if int(segment_len) > 0 else 0)


def build_chunk_meta_from_cu_seqlens(
    cu_seqlens: torch.Tensor, total_tokens: int, arch: str, *, scan_plans: tuple[ScanPlan, ScanPlan]
) -> ChunkMeta:
    """Chunk and segment tables from a device ``cu_seqlens`` tensor without synchronising the host.

    The launch extents are upper bounds (``total_tokens // 64 + num_seqs`` chunks, :func:`segments_upper_bound`
    segments); the padding entries are skipped by every kernel.
    """
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1-D tensor with at least two entries")
    num_seqs = int(cu_seqlens.numel()) - 1
    num_chunks_max = total_tokens // CHUNK + num_seqs
    seg_max = [segments_upper_bound(num_chunks_max, num_seqs, plan.segment_len) for plan in scan_plans]
    cu32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    cu32 = cu32.contiguous()
    device = cu_seqlens.device
    sizes = [num_chunks_max, num_chunks_max, num_seqs + 1]
    for n in seg_max:
        sizes.extend((n + 1, n, num_seqs + 1))
    table = torch.empty(sum(sizes), dtype=torch.int32, device=device)
    views = []
    offset = 0
    for n in sizes:
        views.append(table[offset : offset + n])
        offset += n
    kernel("meta", arch).launch(
        grid=(1, 1, 1),
        cu_seqlens=cu32,
        chunk_start=views[0],
        chunk_len=views[1],
        seq_chunk_start=views[2],
        fwd_seg_chunk_start=views[3],
        fwd_seg_seq=views[4],
        fwd_seq_seg_start=views[5],
        bwd_seg_chunk_start=views[6],
        bwd_seg_seq=views[7],
        bwd_seq_seg_start=views[8],
        num_seqs=num_seqs,
        num_chunks_max=num_chunks_max,
        total_tokens=int(total_tokens),
        fwd_segment_len=int(scan_plans[0].segment_len),
        bwd_segment_len=int(scan_plans[1].segment_len),
        fwd_num_segments_max=seg_max[0],
        bwd_num_segments_max=seg_max[1],
    )
    segments = [
        SegmentTable(
            seg_chunk_start=views[3 + 3 * i],
            seg_seq=views[4 + 3 * i],
            seq_seg_start=views[5 + 3 * i],
            num_segments=seg_max[i],
            segment_len=int(scan_plans[i].segment_len),
            scan_config=tuple(scan_plans[i].config),
        )
        for i in range(2)
    ]
    return ChunkMeta(
        chunk_start=views[0],
        chunk_len=views[1],
        seq_chunk_start=views[2],
        num_chunks=num_chunks_max,
        num_seqs=num_seqs,
        seq_lens=None,
        fwd_segments=segments[0],
        bwd_segments=segments[1],
    )


# --------------------------------------------------------------------------- #
# scan configuration (a pure function of shape and device)
# --------------------------------------------------------------------------- #
@cache
def _device_props(device_index: int) -> tuple[int, int, int]:
    props = torch.cuda.get_device_properties(device_index)
    return int(props.multi_processor_count), int(props.shared_memory_per_block_optin), int(props.major)


def _scan_record(kind: str, vb: int, ks: int, arch: str) -> dict:
    return MODULES[f"{kind}_r{vb}k{ks}"][arch]


# --------------------------------------------------------------------------- #
# finalize kernel selection: a pure function of shape and device, mirroring the generator's finalize_stage
# --------------------------------------------------------------------------- #
FINALIZE_MLP_GROUP = (
    3  # value-head group the one-wave finalize is unrolled for (HV = 3 H at every TP degree of the 4 x 12 head layout)
)
FINALIZE_MLP_MIN_BLOCKS = (
    4  # its launch-bound residency: __launch_bounds__(128, 4) -> 128-register cap -> four CTAs per SM
)
FORCED_FINALIZE: str | None = None  # tests / experiments: "finalize" or "finalize_mlp"
FINALIZE_STAGES = ("finalize", "finalize_mlp")


def finalize_stage(num_chunks: int, num_heads: int, num_v_heads: int, sm_count: int) -> str:
    """Stage name of the finalize kernel for a layout on a device: ``finalize_mlp`` only where it cannot lose.

    The one-wave kernel needs the value-head group it is unrolled for (``FINALIZE_MLP_GROUP``) and is selected only
    while the (chunks, key heads) grid is co-resident in one wave at the residency its launch bound guarantees
    (``FINALIZE_MLP_MIN_BLOCKS`` CTAs per SM): up to that grid both kernels run every CTA concurrently with the same
    warps per SM, so the one-wave kernel keeps the thread-level parallelism of the shared kernel and adds the
    memory-level parallelism of a whole row pair in flight.  Beyond it the shared 40-register kernel's higher residency
    hides latency across more warps while the one-wave kernel would run several waves, so the shared kernel stays
    (the TP1 / TP2 / 16k-token rows).  :data:`FORCED_FINALIZE` pins a stage for experiments and tests; a pinned
    ``finalize_mlp`` outside its group contract raises instead of launching the kernel.
    """

    if num_heads <= 0 or num_v_heads % num_heads != 0:
        raise ValueError(f"num_v_heads={num_v_heads} must be a positive multiple of num_heads={num_heads}")
    forced = FORCED_FINALIZE
    group_ok = num_v_heads == FINALIZE_MLP_GROUP * num_heads
    if forced:
        if forced not in FINALIZE_STAGES:
            raise ValueError(f"FORCED_FINALIZE={forced!r}; expected one of {FINALIZE_STAGES}")
        if forced == "finalize_mlp" and not group_ok:
            raise ValueError(
                f"finalize_mlp is unrolled for a value-head group of {FINALIZE_MLP_GROUP}; "
                f"num_heads={num_heads}, num_v_heads={num_v_heads} is group {num_v_heads // num_heads}"
            )
        return forced
    if group_ok and int(num_chunks) * int(num_heads) <= FINALIZE_MLP_MIN_BLOCKS * int(sm_count):
        return "finalize_mlp"
    return "finalize"


def choose_register_scan_config(
    kind: str, num_seqs: int, num_v_heads: int, device_index: int, arch: str
) -> tuple[int, int]:
    """Register-scan (value rows per CTA, K split) whose grid is fully co-resident on the device.

    The preferred configuration (32 rows, K split in two -> four warps) has the shortest per-chunk
    chain and fills all four SM sub-partitions; wider blocks are used only when the preferred grid
    would need more than one wave.  :data:`FORCED_SCAN_CONFIG` ``= (vb, ksplit)`` pins a configuration for
    experiments and tests.  The choice depends only on shape and device, never on data; it is the baseline the
    segment planner (:func:`choose_scan_plan`) prices every alternative against.
    """
    if FORCED_SCAN_CONFIG is not None:
        return int(FORCED_SCAN_CONFIG[0]), int(FORCED_SCAN_CONFIG[1])
    return _choose_register_scan_config(kind, num_seqs, num_v_heads, device_index, arch)


@cache
def _choose_register_scan_config(kind: str, num_seqs: int, num_v_heads: int, device_index: int, arch: str):
    sm_count, smem_optin, _major = _device_props(device_index)
    for vb, ks in REG_SCAN_CONFIGS:
        ctas = num_seqs * num_v_heads * (HEAD_DIM // vb)
        threads = 32 * (vb // 16) * ks
        smem = int(_scan_record(kind, vb, ks, arch)["scan"]["select_smem_bytes"])
        resident = min(2048 // threads, 65536 // (256 * threads), smem_optin // smem, 4)
        if resident >= 1 and ctas <= sm_count * resident:
            return vb, ks
    return 64, 1


def _scan_stage(kind: str, config: tuple[int, int], device: torch.device, arch: str) -> tuple[str, int]:
    """Stage name and value block of the ``kind`` register scan in ``config`` (the planned configuration)."""
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    vb, ks = config
    _sm_count, smem_optin, major = _device_props(device_index)
    record = _scan_record(kind, vb, ks, arch)
    if major < 9 or int(record["scan"]["select_smem_bytes"]) > smem_optin:
        raise NotImplementedError(
            f"the register-resident {kind} scan does not fit device {device_index} (only SM100a / SM103a / SM107a)"
        )
    return f"{kind}_r{vb}k{ks}", vb


# --------------------------------------------------------------------------- #
# segment-parallel state scans (a pure function of the layout bound and the device)
# --------------------------------------------------------------------------- #
PRESCAN_CONFIGS = ((32, 2), (64, 1))  # the register-scan configurations with pre-scan stages (generator: same name)
MIN_SEGMENT_CHUNKS = 4  # 256 tokens: shorter segments pay more in pre-scan + stitch than they save
# Chunks one pre-scan CTA covers (its warm-up prologue walks one chunk per thread of the stage's block): the longest
# segment a split plan may cut.  A longer one would leave chunks unscanned, so the planner never proposes it; a layout
# that cannot be split within it (and the CTA limit) keeps the serial scan.
PRESCAN_MAX_CHUNKS = 128
# Cost model of the split, in chunk steps of the (32,2) register scan at one CTA per SM, fit on a GB300
# forced-L sweep (1x8192 and 1x16384 tokens at HV 12, L = 13/22/26/43/64/86, CUPTI per kernel; fwd_h 1.23 us and dhu
# 1.58 us per serial chunk step): a second co-resident CTA raises the per-step time to ~1.60x (192/240/288 CTAs on 160
# SMs measured 1.54/1.60/1.63x; the step is latency-bound but two CTAs share the SM's issue slots), a tail wave behind
# a full wave runs at the same factor (480/576 CTAs measured 3.40/3.21x), and a third round (960 CTAs, 6.77x) is
# outside the model (MAX_SCAN_ROUNDS).  At one CTA per SM (every dhu configuration: 140-157 KB of shared memory) there
# is no co-residency: the waves run back to back and ``_scan_makespan`` prices them exactly by slot release, so the
# wave count is quantized -- ceil(CTAs / SMs) full-length waves whatever the fill of the last one (192 serial CTAs on
# 148 / 152 SMs: two waves of 64 steps, the second 26-30 % full; 576 CTAs in 22-chunk segments: four waves of 22
# steps).  Such splits are priced up to MAX_SCAN_WAVES_SERIAL waves (forced-L arms measured at 3 and 4 waves); the
# co-resident bound stays at MAX_SCAN_ROUNDS rounds.  With the gate-adaptive warm-up the pre-scan runs only the
# trailing (forward) / leading (backward) chunks whose summed gate drops below the warm-up threshold: ~3.8 chunk steps
# per consumed segment = PRESCAN_WARMUP_CHUNKS x PRESCAN_STEP_FACTOR (1.16: one more operand tile) on the value blocks
# of the S-1 consumed segments (the identity blocks exit).  The stitch launch costs 3.4 us + 2.8 us per boundary (both
# directions, both lengths) = STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS x (max k - 1) fwd_h steps; a boundary without a
# map is a copy of ``B``.  The exact path (weak gating: full-segment pre-scan, identity blocks, 3xBF16 128^3 product
# + tile rebuild) is not priced.
# Mirror of the kernel generator's segment planner (same constants, same functions): both sides must agree so the
# host table and the device table split identically.
SCAN_CO_RESIDENT_FACTOR = 1.60
PRESCAN_STEP_FACTOR = 1.16
PRESCAN_WARMUP_CHUNKS = (
    3.3  # modelled warm-up length in chunk steps (strong gating: n_w = 1 plus the prologue and launch floor)
)
STITCH_VSPLIT = 4  # stitch CTAs per (sequence, head): V-column blocks of the [K][V] start state (grid z; mirrors the kernel constant)
STITCH_FIXED_CHUNKS = 2.75  # stitch launch floor in chunk steps
STITCH_COPY_CHUNKS = 2.3  # one copy-only stitch step (serial over the boundaries) in chunk steps
SPLIT_GAIN_THRESHOLD = 0.95  # split only when the modelled makespan beats the serial scan by at least 5 %
MAX_SCAN_ROUNDS = (
    2  # a co-resident split may launch at most two rounds of the resident set (the range the sweep calibrated)
)
MAX_SCAN_WAVES_SERIAL = 4  # a one-CTA-per-SM split may launch at most four sequential waves (measured forced-L arms)
# Per-chunk step time of each register-scan configuration relative to the preferred (32, 2) block in the same residency
# class: the (64, 1) block halves the grid (one CTA per 64 value rows) at a longer per-chunk chain (16
# instead of 8 key blocks per warp, no K-split reduction); (32, 1) keeps the (32, 2) grid and doubles the chain.  Scales
# the scan and pre-scan terms of a plan, not the stitch.  Initial values from measured per-configuration step times;
# re-fit per architecture from the pinned-configuration sweep.
SCAN_STEP_FACTOR = {(32, 2): 1.0, (32, 1): 1.18, (64, 1): 1.27}
FORCED_SEGMENT_LEN: int | None = None  # tests pin the segment length (0 = serial); production derives it
# Warm-up threshold of the pre-scans in log2: the trailing (forward) / leading (backward) chunks of a segment whose
# summed gate is below it bound the segment's map ``|M| <= 2^thresh`` and the dropped earlier contributions to ``B``
# to the same relative size; -10 nats is FlashQLA's warm-up rule (cp_fwd.py at 0.1.2+7c7dfe1).
# :data:`FORCED_WARMUP_LOG2` overrides it for experiments (``-inf``: the exact path on every segment).
WARMUP_THRESH_LOG2 = -10.0 / math.log(2.0)
FORCED_WARMUP_LOG2: float | None = None  # experiments pin the warm-up threshold (log2); None keeps the default


def warmup_thresh_log2() -> float:
    """The pre-scan warm-up threshold (log2), :data:`WARMUP_THRESH_LOG2` unless :data:`FORCED_WARMUP_LOG2` is set."""
    return float(FORCED_WARMUP_LOG2) if FORCED_WARMUP_LOG2 is not None else WARMUP_THRESH_LOG2


def _scan_residency(kind: str, vb: int, ks: int, device_index: int, arch: str) -> int:
    _sm_count, smem_optin, _major = _device_props(device_index)
    threads = 32 * (vb // 16) * ks
    smem = int(_scan_record(kind, vb, ks, arch)["scan"]["select_smem_bytes"])
    return max(1, min(2048 // threads, 65536 // (256 * threads), smem_optin // smem, 4))


def _wave_factor(ctas: int, sm_count: int, residency: int) -> float:
    """Makespan factor of ``ctas`` concurrently alive latency-bound scan CTAs relative to one CTA per SM.

    Co-resident CTAs (at most two per SM, when the scan's shared memory allows it) finish together at
    :data:`SCAN_CO_RESIDENT_FACTOR` x the single-CTA step time; CTAs beyond the resident set run as further waves.  A
    tail wave is priced at 1.0 only when it is the sole wave and fits one CTA per SM: behind a full wave it starts
    staggered as resident CTAs finish and shares SMs with them, so it runs at the full wave's factor (480 / 576 CTAs on
    160 SMs measured 3.40 / 3.21x against the 2.55 / 3.10x of a tail wave priced at 1.0).
    """
    co_resident = residency >= 2
    per_wave = sm_count * (2 if co_resident else 1)
    full_factor = SCAN_CO_RESIDENT_FACTOR if co_resident else 1.0
    full, tail = divmod(ctas, per_wave)
    cost = full * full_factor
    if tail:
        cost += full_factor if (full or tail > sm_count) else 1.0
    return cost


def _scan_makespan(
    chunks: Sequence[int], splits: Sequence[int], per_segment: int, sm_count: int, residency: int
) -> float:
    """Modelled makespan, in chunk steps of one CTA per SM, of a scan grid over ``chunks`` cut into ``splits`` segments.

    Sequence ``i`` runs ``splits[i] x per_segment`` CTAs for ``ceil(chunks[i] / splits[i])`` steps (its segments are
    equal-sized within one chunk).  The first ``sm_count x residency`` CTAs are resident from the start, the shortest
    first (the grid interleaves the sequences' segments, so every sequence is represented in the first wave and the
    short ones free their slots first); every further CTA starts when a resident CTA finishes and is priced at the
    longest duration.  Whenever more than one CTA per SM is launched the whole makespan runs at
    :data:`SCAN_CO_RESIDENT_FACTOR`: the dispatch order, not the durations, decides which SMs hold two CTAs, so a
    short sequence leaving early does not un-double the SMs that hold two long CTAs (the real 8481 + 223 micro-batch:
    192 CTAs with a 4-chunk pad measured 1.53x over the full 45 steps; its six-segment grid of 336 CTAs ran at
    1.6 x (4 + 23) steps, the 16 CTAs beyond the resident set starting when the pad's CTAs finished).  One sequence
    reduces to ``steps x _wave_factor(all CTAs)`` within :data:`MAX_SCAN_ROUNDS`.
    """
    runs = sorted((-(-n // k), k * per_segment) for n, k in zip(chunks, splits, strict=False))
    slots = sm_count * (2 if residency >= 2 else 1)
    longest = runs[-1][0]
    releases = []
    capacity, waiting = slots, 0
    for steps, ctas in runs:
        resident = min(ctas, capacity)
        capacity -= resident
        if resident:
            releases.append((steps, resident))
        waiting += ctas - resident
    makespan = longest
    while waiting > 0 and releases:
        next_releases = []
        for at, count in releases:
            if waiting <= 0:
                break
            started = min(count, waiting)
            waiting -= started
            makespan = max(makespan, at + longest)
            next_releases.append((at + longest, started))
        releases = next_releases
    factor = SCAN_CO_RESIDENT_FACTOR if (residency >= 2 and sum(c for _s, c in runs) > sm_count) else 1.0
    return factor * makespan


def bound_layout_chunks(chunks_bound: int, num_seqs: int) -> tuple[int, ...]:
    """Per-sequence chunk counts of the layout a padded chunk bound allows with the longest serial chain.

    One sequence holds every chunk but one per other sequence; this is all the device table builder (no host copy of
    ``cu_seqlens``) can price.
    """
    return (chunks_bound - (num_seqs - 1),) + (1,) * (num_seqs - 1)


def _serial_cost(chunks: Sequence[int], per_segment: int, sm_count: int, residency: int, step_factor: float) -> float:
    """Modelled makespan of the serial scan (one segment per sequence) in chunk steps of the (32, 2) scan."""
    return step_factor * _scan_makespan(chunks, (1,) * len(chunks), per_segment, sm_count, residency)


def split_cta_limit(sm_count: int, residency: int) -> int:
    """Largest grid (CTAs) a split may launch on ``sm_count`` SMs: :data:`MAX_SCAN_ROUNDS` rounds of the two-CTA
    resident set when the scan is co-resident, :data:`MAX_SCAN_WAVES_SERIAL` sequential waves at one CTA per SM.  Both
    bounds are four CTAs per SM; they are kept apart because the rounds and the waves were calibrated in separate
    sweeps."""
    return MAX_SCAN_ROUNDS * 2 * sm_count if residency >= 2 else MAX_SCAN_WAVES_SERIAL * sm_count


def _best_split(
    chunks: Sequence[int],
    per_segment: int,
    sm_count: int,
    residency: int,
    step_factor: float,
    max_segment_len: int = PRESCAN_MAX_CHUNKS,
) -> tuple[int, float]:
    """``(segment_len, modelled cost)`` of the cheapest admissible split of ``chunks`` (``(0, inf)`` when none is).

    A segment length ``L`` cuts sequence ``i`` into ``k_i = ceil(n_i / L)`` equal-sized segments (``k_i = 1``: unsplit)
    and costs ``f x makespan(k) + f x 3.3 x 1.16 x waves(sum (k_i - 1) x per_segment) + 2.75 + 2.3 x (max k_i - 1)``
    chunk steps for the configuration's step factor ``f`` -- the pass-2 scan over every sequence's segments
    (:func:`_scan_makespan`), the warm-up pre-scan of the consumed segments' value blocks, the stitch launch floor plus
    one serial copy-only step per boundary.  Candidates whose grid exceeds :func:`split_cta_limit` (two rounds of the
    resident set when co-resident, four sequential waves at one CTA per SM) lie outside the calibration and are skipped,
    and so are segments longer than ``max_segment_len`` chunks (the pre-scan capacity, :data:`PRESCAN_MAX_CHUNKS`).
    """
    longest = max(chunks)
    limit = split_cta_limit(sm_count, residency)
    best_len, best_cost = 0, math.inf
    for segments in range(2, longest // MIN_SEGMENT_CHUNKS + 1):
        seg_len = -(-longest // segments)
        if seg_len < MIN_SEGMENT_CHUNKS or seg_len >= longest or seg_len > max_segment_len:
            continue
        splits = tuple(max(1, -(-n // seg_len)) for n in chunks)
        if sum(splits) * per_segment > limit:
            continue
        scan = step_factor * _scan_makespan(chunks, splits, per_segment, sm_count, residency)
        pre = (
            PRESCAN_WARMUP_CHUNKS
            * PRESCAN_STEP_FACTOR
            * step_factor
            * _wave_factor(sum(k - 1 for k in splits) * per_segment, sm_count, residency)
        )
        cost = scan + pre + STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS * (max(splits) - 1)
        if cost < best_cost:
            best_len, best_cost = seg_len, cost
    return best_len, best_cost


def plan_segments(
    chunks_bound: int,
    num_seqs: int,
    per_segment: int,
    sm_count: int,
    residency: int,
    seq_chunks: tuple[int, ...] | None = None,
    step_factor: float = 1.0,
) -> tuple[int, float]:
    """``(segment_len, modelled makespan in chunk steps)`` of a layout in one configuration; ``0`` keeps the serial scan.

    Pure function of the layout and the device.  Without ``seq_chunks`` (the device route, which knows only the padded
    chunk bound ``T // 64 + num_seqs``) it prices :func:`bound_layout_chunks`; with ``seq_chunks`` (the host route:
    the exact per-sequence chunk counts) it prices the tables that will actually run.  The cheapest split
    (:func:`_best_split`) is taken when it beats ``0.95 x`` the serial scan's own makespan (:func:`_serial_cost`), both
    at the configuration's ``step_factor``.  Balanced packed layouts whose sequences are all shorter than ``L`` keep one
    segment per sequence (the pre-scan -- and, for the backward scan, the stitch -- then launch on the device route and
    exit immediately).
    """
    chunks = (
        tuple(int(c) for c in seq_chunks) if seq_chunks is not None else bound_layout_chunks(chunks_bound, num_seqs)
    )
    serial = _serial_cost(chunks, per_segment, sm_count, residency, step_factor)
    best_len, best_cost = _best_split(chunks, per_segment, sm_count, residency, step_factor)
    if best_len == 0 or best_cost >= SPLIT_GAIN_THRESHOLD * serial:
        return 0, serial
    return best_len, best_cost


def _scan_fits(kind: str, config: tuple[int, int], device_index: int, arch: str) -> str | None:
    """``None`` when the ``kind`` register scan in ``config`` is rendered for ``arch`` and fits the device, else why not."""
    vb, ks = config
    stage = f"{kind}_r{vb}k{ks}"
    _sm_count, smem_optin, major = _device_props(device_index)
    if stage not in MODULES or arch not in MODULES[stage]:
        return f"no {stage} stage is rendered for {arch}"
    if major < 9:
        return f"compute capability {major}.x has no register scan"
    smem = int(_scan_record(kind, vb, ks, arch)["scan"]["select_smem_bytes"])
    if smem > smem_optin:
        return f"the {kind} register scan {config} needs {smem} B of shared memory, above the device opt-in {smem_optin} B"
    return None


def _prescan_capacity(kind: str, config: tuple[int, int], arch: str) -> int:
    """Chunks one pre-scan CTA of ``kind`` in ``config`` covers on ``arch``: one per thread of its block, at most
    :data:`PRESCAN_MAX_CHUNKS`; ``0`` without a pre-scan stage."""
    records = MODULES.get(f"{kind}_pre_r{config[0]}k{config[1]}", {})
    if arch not in records:
        return 0
    return min(PRESCAN_MAX_CHUNKS, int(records[arch]["block"][0]))


def _prescan_fits(kind: str, config: tuple[int, int], device_index: int, arch: str) -> str | None:
    """``None`` when ``config`` can be split for ``kind`` on this device (pre-scan stage rendered, scan fits), else why not."""
    if config not in PRESCAN_CONFIGS:
        return f"its register-scan configuration {config} has no pre-scan stage (only {PRESCAN_CONFIGS} have one)"
    pre = f"{kind}_pre_r{config[0]}k{config[1]}"
    if pre not in MODULES or arch not in MODULES[pre]:
        return f"no {pre} stage is rendered for {arch}"
    return _scan_fits(kind, config, device_index, arch)


def choose_scan_plan(
    kind: str,
    total_tokens: int,
    num_seqs: int,
    num_v_heads: int,
    device_index: int,
    arch: str,
    seq_lens: tuple[int, ...] | None = None,
) -> ScanPlan:
    """Register-scan configuration and chunks per segment of the ``kind`` state scan of a layout.

    The baseline is :func:`choose_register_scan_config` run serially (one segment per sequence).  Every register
    configuration rendered for the device is priced serially at its :data:`SCAN_STEP_FACTOR`, and the configurations
    with pre-scan stages (:data:`PRESCAN_CONFIGS`) also at their cheapest split (:func:`_best_split`); the exact
    per-sequence chunk counts are priced when ``seq_lens`` is known (host route), the padded chunk bound
    ``T // 64 + num_seqs`` otherwise (device route).  The cheapest plan is taken when it beats ``0.95 x`` the baseline
    (one rule for a configuration switch and for a split); otherwise the baseline runs.
    :data:`FORCED_SCAN_CONFIG` pins the configuration (the segment length is then planned for it);
    :data:`FORCED_SEGMENT_LEN` pins the segment length of the baseline (or pinned) configuration, validated at plan
    time: a pin on a configuration without a pre-scan stage, or longer than the chunks one pre-scan CTA covers
    (:func:`_prescan_capacity`), is dropped with a :class:`RuntimeWarning` naming the layout, and the serial scan runs.
    No split plan cuts a segment longer than that capacity: a long sequence that cannot be split within it and the CTA
    limit (:func:`split_cta_limit`) keeps the serial scan.
    """
    forced_len = int(FORCED_SEGMENT_LEN) if FORCED_SEGMENT_LEN is not None else None
    baseline = choose_register_scan_config(kind, num_seqs, num_v_heads, device_index, arch)
    if forced_len is not None:
        unsplittable = _prescan_fits(kind, baseline, device_index, arch)
        capacity = _prescan_capacity(kind, baseline, arch)
        if unsplittable is None and forced_len > capacity:
            unsplittable = f"it exceeds the {capacity} chunks one pre-scan CTA covers"
        if forced_len > 0 and unsplittable is not None:
            warnings.warn(
                f"forced segment length {forced_len} ignored for the {kind} scan of this layout ({num_seqs} sequences, "
                f"{num_v_heads} value heads, {total_tokens} tokens): {unsplittable}; the serial scan runs",
                RuntimeWarning,
                stacklevel=2,
            )
            return ScanPlan(baseline, 0)
        return ScanPlan(baseline, forced_len)
    if _scan_fits(kind, baseline, device_index, arch) is not None:
        return ScanPlan(baseline, 0)  # the launch reports the unsupported device (_scan_stage)
    pinned = FORCED_SCAN_CONFIG is not None
    chunks_bound = total_tokens // CHUNK + num_seqs
    chunks = (
        tuple(-(-int(x) // CHUNK) for x in seq_lens)
        if seq_lens is not None
        else bound_layout_chunks(chunks_bound, num_seqs)
    )
    sm_count, _smem_optin, _major = _device_props(device_index)
    candidates = (
        [baseline]
        if pinned
        else [cand for cand in REG_SCAN_CONFIGS if _scan_fits(kind, cand, device_index, arch) is None]
    )
    baseline_cost = _serial_cost(
        chunks,
        num_v_heads * (HEAD_DIM // baseline[0]),
        sm_count,
        _scan_residency(kind, baseline[0], baseline[1], device_index, arch),
        SCAN_STEP_FACTOR[baseline],
    )
    best = (baseline, 0, baseline_cost)
    for cand in candidates:
        per_segment = num_v_heads * (HEAD_DIM // cand[0])
        residency = _scan_residency(kind, cand[0], cand[1], device_index, arch)
        options = [(0, _serial_cost(chunks, per_segment, sm_count, residency, SCAN_STEP_FACTOR[cand]))]
        if _prescan_fits(kind, cand, device_index, arch) is None:
            capacity = _prescan_capacity(kind, cand, arch)
            options.append(_best_split(chunks, per_segment, sm_count, residency, SCAN_STEP_FACTOR[cand], capacity))
        for seg_len, cost in options:
            if cost < best[2]:
                best = (cand, seg_len, cost)
    chosen, seg_len, cost = best
    if (chosen, seg_len) != (baseline, 0) and cost >= SPLIT_GAIN_THRESHOLD * baseline_cost:
        return ScanPlan(baseline, 0)
    return ScanPlan(chosen, seg_len)


def _plan_pins() -> tuple:
    """The test / experiment pins a plan depends on besides its arguments (part of the :func:`scan_plans` memo key)."""
    return (
        None if FORCED_SEGMENT_LEN is None else int(FORCED_SEGMENT_LEN),
        None if FORCED_SCAN_CONFIG is None else (int(FORCED_SCAN_CONFIG[0]), int(FORCED_SCAN_CONFIG[1])),
    )


@lru_cache(maxsize=4096)
def _scan_plans_memo(
    total_tokens: int,
    num_seqs: int,
    num_v_heads: int,
    device_index: int,
    arch: str,
    seq_lens: tuple[int, ...] | None,
    device_props: tuple[int, int, int],
    pins: tuple,
) -> tuple[ScanPlan, ScanPlan]:
    # ``device_props`` and ``pins`` are key material only: the plan reads the same values through the helpers
    return (
        choose_scan_plan("fwd_h", total_tokens, num_seqs, num_v_heads, device_index, arch, seq_lens),
        choose_scan_plan("dhu", total_tokens, num_seqs, num_v_heads, device_index, arch, seq_lens),
    )


def scan_plans(
    total_tokens: int,
    num_seqs: int,
    num_v_heads: int,
    device_index: int,
    arch: str,
    seq_lens: tuple[int, ...] | None = None,
) -> tuple[ScanPlan, ScanPlan]:
    """``(forward scan, backward scan)`` plans of a layout (see :func:`choose_scan_plan`).

    Memoised per layout, device (index, properties) and pin state, like the generator's ``scan_plans``: the host route
    (``chunk_meta_from_host``) plans on every call and the joint search costs ~60 us of host time per call; a
    repeated layout is a dictionary hit, a pin change always re-plans.
    """
    seq_key = None if seq_lens is None else tuple(int(x) for x in seq_lens)
    return _scan_plans_memo(
        int(total_tokens),
        int(num_seqs),
        int(num_v_heads),
        int(device_index),
        str(arch),
        seq_key,
        _device_props(int(device_index)),
        _plan_pins(),
    )


def segment_lens(
    total_tokens: int,
    num_seqs: int,
    num_v_heads: int,
    device_index: int,
    arch: str,
    seq_lens: tuple[int, ...] | None = None,
) -> tuple[int, int]:
    """The two plans' segment lengths (:func:`scan_plans`); the per-case record of the benchmark and DET harnesses."""
    return tuple(
        plan.segment_len for plan in scan_plans(total_tokens, num_seqs, num_v_heads, device_index, arch, seq_lens)
    )


# --------------------------------------------------------------------------- #
# host-side caches: chunk-table front cache, device architecture, 0-element state placeholders
# --------------------------------------------------------------------------- #
# Scratch buffers come from the workspace pool: ``workspace.take(OP, role, ...)``, valid until the next take of
# ``role`` on the same stream, never saved for backward, returned or kept.  Forward roles carry the prefix ``f_``,
# backward roles ``b_``, so a backward-time recompute of the forward stages shares no buffer with the backward's own
# scratch.
_DEVICE_ARCH: dict[int, str] = {}


def _arch_of(device_index: int) -> str:
    """``loader.device_arch`` of a CUDA device index, memoised per index (``torch.cuda.get_device_capability`` resolves
    the device in Python on every call otherwise)."""
    arch = _DEVICE_ARCH.get(device_index)
    if arch is None:
        arch = _DEVICE_ARCH[device_index] = loader.device_arch(torch.device("cuda", device_index))
    return arch


# (layout key, num_v_heads, device index, total_tokens, plan pins) -> ChunkMeta (see _chunk_meta)
_META_FRONT: dict = {}
_META_FRONT_LIMIT = 256
_META_FRONT_HITS = 0


# (device, shape, dtype) -> a 0-element tensor (see _no_state)
_NO_STATE: dict = {}
# the roles of the 0-element state placeholders, here and in the det and fast packages
_NO_STATE_ROLES = frozenset(("f_zero_state", "b_zero_state"))


def host_cache_clear():
    """Drop the host-side caches of this module: the chunk-table front cache, the device architectures, the 0-element
    state placeholders, the per-layout table memo ``_host_chunk_meta`` and the plan memo ``_scan_plans_memo``."""
    global _META_FRONT_HITS
    _META_FRONT.clear()
    _DEVICE_ARCH.clear()
    _NO_STATE.clear()
    _host_chunk_meta.cache_clear()
    _scan_plans_memo.cache_clear()
    _META_FRONT_HITS = 0


def _no_state(shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """The 0-element tensor of ``shape`` (one extent 0) that a launch without an initial or final state passes in place
    of the state (``use_initial_state=0`` / ``store_final_state=0``; the kernels never touch it), one per device, shape
    and dtype for the process.  It holds no memory, so calls on any stream share it without a race, and a call needs no
    allocation for it."""
    key = (device, shape, dtype)
    tensor = _NO_STATE.get(key)
    if tensor is None:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        if tensor.numel():
            raise ValueError(f"a state placeholder holds no element; got the shape {tuple(shape)}")
        _NO_STATE[key] = tensor
    return tensor


@cache
def _scan_stages(kind: str, config: tuple[int, int], device_index: int, arch: str) -> tuple[str, int, str | None, int]:
    """``(scan stage, value block, pre-scan stage or None, chunks the pre-scan prologue covers)`` of the ``kind``
    register scan in ``config`` on one device -- :func:`_scan_stage` plus the pre-scan lookups, resolved once per key.
    """
    stage, vb = _scan_stage(kind, config, torch.device("cuda", device_index), arch)
    pre = stage.replace(f"{kind}_r", f"{kind}_pre_r")
    if pre not in MODULES or arch not in MODULES[pre]:
        return stage, vb, None, 0
    return stage, vb, pre, int(MODULES[pre][arch]["block"][0])


@cache
def _split_scan_stage(stage: str, arch: str) -> str | None:
    """The split-plan instantiation ``<stage>_split`` of a forward scan stage (the forward stitch folded into its prologue;
    exported for every forward configuration), or ``None`` when this architecture's registry carries none."""
    split = f"{stage}_split"
    return split if split in MODULES and arch in MODULES[split] else None


# Positional launch orders = the exported plan order of every stage (registry ``arg_plan`` without the trailing grid
# entries; identical on sm_100a and sm_103a).  ``native.NativeKernel.launcher`` verifies the name set against the plan
# at first use and permutes if an export ever reorders a plan; the values at every call site follow these tuples one to
# one.
_PREP_ARGS = (
    "q",
    "k",
    "g",
    "beta",
    "beta32_out",
    "qn",
    "kn",
    "rstd_q",
    "rstd_k",
    "g_cs",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "normalize_qk",
)
_WY_ARGS = (
    "kn",
    "v",
    "g_cs",
    "beta",
    "A_out",
    "w_out",
    "u_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "recompute",
)
_FWD_H_ARGS = (
    "kn",
    "w",
    "u",
    "g_cs",
    "h0",
    "h_out",
    "v_new",
    "final_state",
    "seg_state",
    "chunk_start",
    "chunk_len",
    "seg_chunk_start",
    "seg_seq",
    "seq_chunk_start",
    "num_heads",
    "num_v_heads",
    "use_initial_state",
    "store_final_state",
)
# the split-plan instantiation of the forward scan reads the per-segment flags too (its prologue folds the forward
# stitch); the pre-scan adds the warm-up threshold on top
_FWD_H_SPLIT_ARGS = _FWD_H_ARGS + ("seg_flag",)
_FWD_H_PRE_ARGS = _FWD_H_SPLIT_ARGS + ("warmup_thresh_log2",)
_STITCH_ARGS = (
    "seg_state",
    "init_state",
    "state_out",
    "seq_seg_start",
    "seg_flag",
    "num_v_heads",
    "use_init",
    "reverse",
)
_FWD_O_ARGS = (
    "qn",
    "kn",
    "v_new",
    "h",
    "g_cs",
    "o_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "scale",
)
_DV_LOCAL_ARGS = (
    "qn",
    "kn",
    "do",
    "g_cs",
    "dv_local",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "scale",
)
_DHU_ARGS = (
    "qn",
    "kn",
    "w",
    "do",
    "dv_local",
    "g_cs",
    "dht",
    "dh_out",
    "dv2",
    "dh0",
    "seg_state",
    "chunk_start",
    "chunk_len",
    "seg_chunk_start",
    "seg_seq",
    "seq_chunk_start",
    "num_heads",
    "num_v_heads",
    "use_final_state_grad",
    "store_initial_state_grad",
    "scale",
)
_DHU_PRE_ARGS = _DHU_ARGS + ("seg_flag", "warmup_thresh_log2")
_DQKWG_ARGS = (
    "qn",
    "kn",
    "v_new",
    "do",
    "dv2",
    "h",
    "dh",
    "g_cs",
    "dq_out",
    "dk_out",
    "dw_out",
    "dg_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "scale",
)
_WY_BWD_ARGS = (
    "kn",
    "v",
    "A",
    "dw",
    "du",
    "g_cs",
    "beta",
    "dk2_out",
    "dv_out",
    "dbeta_out",
    "dg2_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
)
_FINALIZE_ARGS = (
    "qn",
    "kn",
    "rstd_q",
    "rstd_k",
    "dq_hv",
    "dk_hv",
    "dk2",
    "dg1",
    "dg2",
    "dq_out",
    "dk_out",
    "dg_out",
    "chunk_start",
    "chunk_len",
    "num_heads",
    "num_v_heads",
    "normalize_qk",
)
# stage -> call-site order, for the wiring test that pins every launcher to its exported plan
LAUNCH_ARGS = {
    "prep": _PREP_ARGS,
    "prep_bf16beta": _PREP_ARGS,
    "wy": _WY_ARGS,
    "fwd_o": _FWD_O_ARGS,
    "stitch": _STITCH_ARGS,
    "dv_local": _DV_LOCAL_ARGS,
    "dqkwg": _DQKWG_ARGS,
    "wy_bwd": _WY_BWD_ARGS,
    "wy_bwd_bf16": _WY_BWD_ARGS,
    "finalize": _FINALIZE_ARGS,
    "finalize_mlp": _FINALIZE_ARGS,  # same plan as finalize; the host picks the stage per layout / device (finalize_stage)
    **{f"fwd_h_r{vb}k{ks}": _FWD_H_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"fwd_h_r{vb}k{ks}_split": _FWD_H_SPLIT_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"fwd_h_pre_r{vb}k{ks}": _FWD_H_PRE_ARGS for vb, ks in PRESCAN_CONFIGS},
    **{f"dhu_r{vb}k{ks}": _DHU_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"dhu_pre_r{vb}k{ks}": _DHU_PRE_ARGS for vb, ks in PRESCAN_CONFIGS},
}


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def _launch_wy(
    arch,
    kn,
    v,
    g_cs,
    beta32,
    A,
    meta: ChunkMeta,
    num_heads: int,
    num_v_heads: int,
    total_tokens: int,
    *,
    recompute: bool,
):
    # ``w`` / ``u`` go on the autograd tape (unless recomputed) or into the returned forward dict: fresh, never pooled.
    w = torch.empty((total_tokens, num_v_heads, HEAD_DIM), dtype=kn.dtype, device=kn.device)
    u = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    launcher("wy", arch, _WY_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        kn,
        v,
        g_cs,
        beta32,
        A,
        w,
        u,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        1 if recompute else 0,
    )
    return w, u


def _launch_fwd_h(
    arch,
    kn,
    w,
    u,
    g_cs,
    initial_state,
    meta: ChunkMeta,
    num_heads: int,
    num_v_heads: int,
    *,
    store_final_state: bool,
):
    device = kn.device
    # ``h`` / ``v_new`` are saved for backward (or returned): fresh.  ``final_state`` is returned when requested; otherwise
    # it is the 0-element placeholder the kernels never touch (``store_final_state=0`` / ``use_initial_state=0``).
    h = torch.empty((meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM), dtype=kn.dtype, device=device)
    v_new = torch.empty(u.shape, dtype=u.dtype, device=device)
    if store_final_state:  # the kernel writes every element of ``final_state``; no fill kernel is needed
        final_state = torch.empty((meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=device)
    else:
        final_state = _no_state((0, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32, device)
    if initial_state is not None:
        h0 = initial_state.contiguous()
        if h0.dtype != torch.float32:
            h0 = h0.float()
        use_init = 1
    else:
        h0, use_init = final_state, 0
    segs = meta.fwd_segments
    stage, vb, pre, pre_threads = _scan_stages(
        "fwd_h", segs.scan_config, device.index, arch
    )  # the planned configuration
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        # Segment-parallel scan: every segment's affine map in parallel (value and identity column blocks of the
        # augmented state), then the split instantiation of the scan per segment, whose prologue derives the segment's
        # FP32 start state from the maps and flags (the forward stitch folded into pass 2: no stitch launch, no
        # start-state scratch).  ``seg_state`` / ``seg_flag`` (pre-scan -> scan) live only inside this call: scratch.
        # The serial plan launches the plain scan below.
        split_stage = _split_scan_stage(stage, arch)
        if pre is None or split_stage is None:
            raise NotImplementedError(
                f"the segmented fwd_h scan needs the stages {stage.replace('fwd_h_r', 'fwd_h_pre_r')} and {stage}_split for {arch}"
            )
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks per segment the {pre} warm-up prologue scans"
            )
        seg_state = workspace.take(
            OP, "f_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM), torch.float32, device
        )
        # per-segment flags written by the pre-scan (1: exact path, map block present) and read by the scan's prologue
        seg_flag = workspace.take(OP, "f_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device)
        launcher(pre, arch, _FWD_H_PRE_ARGS)(
            segs.num_segments,
            num_v_heads,
            2 * HEAD_DIM // vb,
            kn,
            w,
            u,
            g_cs,
            h0,
            h,
            v_new,
            final_state,
            seg_state,
            meta.chunk_start,
            meta.chunk_len,
            segs.seg_chunk_start,
            segs.seg_seq,
            meta.seq_chunk_start,
            num_heads,
            num_v_heads,
            use_init,
            0,
            seg_flag,
            warmup_thresh_log2(),
        )
        launcher(split_stage, arch, _FWD_H_SPLIT_ARGS)(
            segs.num_segments,
            num_v_heads,
            HEAD_DIM // vb,
            kn,
            w,
            u,
            g_cs,
            h0,
            h,
            v_new,
            final_state,
            seg_state,
            meta.chunk_start,
            meta.chunk_len,
            segs.seg_chunk_start,
            segs.seg_seq,
            meta.seq_chunk_start,
            num_heads,
            num_v_heads,
            use_init,
            1 if store_final_state else 0,
            seg_flag,
        )
        return h, v_new, final_state
    launcher(stage, arch, _FWD_H_ARGS)(
        segs.num_segments,
        num_v_heads,
        HEAD_DIM // vb,
        kn,
        w,
        u,
        g_cs,
        h0,
        h,
        v_new,
        final_state,
        final_state,
        meta.chunk_start,
        meta.chunk_len,
        segs.seg_chunk_start,
        segs.seg_seq,
        meta.seq_chunk_start,
        num_heads,
        num_v_heads,
        use_init,
        1 if store_final_state else 0,
    )
    return h, v_new, final_state


def gdn_chunk_forward_stage1(
    q, k, v, g, beta, *, meta: ChunkMeta, scale: float | None = None, normalize_qk: bool = True
) -> dict[str, Any]:
    """Prep (l2norm, gate cumsum) and WY (A, w, u); no state is touched yet.

    ``q`` / ``k`` / ``v`` may be ``[T, H, D]`` or the autograd wrapper's ``[B, T, H, D]``: the kernels address tokens
    through the chunk table and never see a tensor shape, so the batch dimension passes through unflattened (no
    ``reshape`` per call) and every per-token output is allocated in the input's own shape.
    """
    num_heads, key_dim = q.shape[-2], q.shape[-1]
    num_v_heads, value_dim = v.shape[-2], v.shape[-1]
    if key_dim != HEAD_DIM or value_dim != HEAD_DIM:
        raise ValueError("gdn_chunk_forward requires K = V = 128")
    if num_v_heads % num_heads != 0:
        raise ValueError("num_v_heads must be a multiple of num_heads")
    total_tokens = q.numel() // (num_heads * key_dim)
    if scale is None:
        scale = 1.0 / math.sqrt(key_dim)
    device = q.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = _arch_of(index)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g32 = g.contiguous()
    if g32.dtype != torch.float32:
        g32 = g32.float()
    beta = beta.contiguous()
    # BF16 beta is converted inside the prep kernel (no separate cast kernel); FP32 passes through.
    beta_bf16 = beta.dtype == torch.bfloat16
    if beta_bf16:
        beta32 = torch.empty((total_tokens, num_v_heads), dtype=torch.float32, device=device)
    else:
        beta32 = beta if beta.dtype == torch.float32 else beta.float()

    # every tensor below is saved for backward: fresh allocations (the pool holds scratch only)
    qn = torch.empty(q.shape, dtype=q.dtype, device=device)
    kn = torch.empty(k.shape, dtype=k.dtype, device=device)
    rstd_q = torch.empty((total_tokens, num_heads), dtype=torch.float32, device=device)
    rstd_k = torch.empty((total_tokens, num_heads), dtype=torch.float32, device=device)
    g_cs = torch.empty((total_tokens, num_v_heads), dtype=torch.float32, device=device)
    launcher("prep_bf16beta" if beta_bf16 else "prep", arch, _PREP_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        q,
        k,
        g32,
        beta,
        beta32,
        qn,
        kn,
        rstd_q,
        rstd_k,
        g_cs,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        1 if normalize_qk else 0,
    )
    A = torch.empty((total_tokens, num_v_heads, CHUNK), dtype=q.dtype, device=device)  # fully written by the WY kernel
    w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, total_tokens, recompute=False)
    return {
        "arch": arch,
        "normalize_qk": normalize_qk,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "total_tokens": total_tokens,
        "qn": qn,
        "kn": kn,
        "rstd_q": rstd_q,
        "rstd_k": rstd_k,
        "g_cs": g_cs,
        "A": A,
        "w": w,
        "u": u,
        "v": v,
        "beta32": beta32,
        "meta": meta,
        "scale": float(scale),
    }


def gdn_chunk_forward_stage2(
    stage1: dict[str, Any], *, initial_state=None, output_final_state: bool = False
) -> dict[str, Any]:
    """State scan and output; extends the stage-1 dict in place."""
    arch = stage1["arch"]
    meta: ChunkMeta = stage1["meta"]
    kn, qn, w, u, g_cs, v = (stage1[name] for name in ("kn", "qn", "w", "u", "g_cs", "v"))
    num_heads, num_v_heads = stage1["num_heads"], stage1["num_v_heads"]
    h, v_new, final_state = _launch_fwd_h(
        arch,
        kn,
        w,
        u,
        g_cs,
        initial_state,
        meta,
        num_heads,
        num_v_heads,
        store_final_state=output_final_state,
    )
    o = torch.empty(v.shape, dtype=v.dtype, device=v.device)  # returned: fresh, in v's own shape
    launcher("fwd_o", arch, _FWD_O_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        qn,
        kn,
        v_new,
        h,
        g_cs,
        o,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        float(stage1["scale"]),
    )
    stage1.update(
        {
            "output": o,
            "final_state": final_state if output_final_state else None,
            "initial_state": initial_state,
            "h": h,
            "v_new": v_new,
        }
    )
    return stage1


def gdn_chunk_forward(
    q,
    k,
    v,
    g,
    beta,
    *,
    meta: ChunkMeta,
    scale: float | None = None,
    initial_state=None,
    output_final_state: bool = False,
    normalize_qk: bool = True,
) -> dict[str, Any]:
    """Run the deterministic chunked forward; returns outputs and the tape ``gdn_chunk_backward`` needs."""
    stage1 = gdn_chunk_forward_stage1(q, k, v, g, beta, meta=meta, scale=scale, normalize_qk=normalize_qk)
    return gdn_chunk_forward_stage2(stage1, initial_state=initial_state, output_final_state=output_final_state)


# --------------------------------------------------------------------------- #
# backward
# --------------------------------------------------------------------------- #
def gdn_chunk_backward(
    forward: dict[str, Any], do, dht=None, *, initial_state=None, dht_from_dv_local=None, dbeta_dtype=None
) -> dict[str, Any]:
    """Run the deterministic chunked backward from the (possibly pruned) forward tape.

    ``forward`` must carry ``arch``, ``qn``, ``kn``, ``rstd_q``, ``rstd_k``, ``v``, ``g_cs``, ``beta32``,
    ``A``, ``meta``, ``scale`` and ``normalize_qk``.  ``w``/``u``/``h``/``v_new`` are recomputed exactly
    (same kernels, same order) when the tape does not carry them.  ``initial_state`` is the FP32 state
    the forward started from (``None`` for a zero state); its gradient is returned only when it is
    given.  ``dht_from_dv_local(w, dv_local)`` optionally supplies the final-state gradient after
    ``dv_local`` exists (context-parallel state passing).  ``g_shape`` / ``beta_shape`` in the tape give the
    shapes of the returned ``dg`` / ``dbeta`` (default ``[T, HV]``).

    Scratch of this call (``dv_local``, ``dh``, ``dv2``, the segmented scan's ``seg_state`` / ``seg_flag`` / ``dh_end``,
    ``dq_hv`` / ``dk_hv`` / ``dw`` / ``dg1``, ``dk2`` / ``dg2``) comes from the workspace pool, and the 0-element
    ``dh0`` placeholder from :func:`_no_state`; every returned gradient is a fresh allocation.
    """
    arch = forward["arch"]
    meta: ChunkMeta = forward["meta"]
    qn, kn = forward["qn"], forward["kn"]
    A = forward["A"]
    g_cs = forward["g_cs"]
    scale = float(forward["scale"])
    v = forward["v"]
    beta32 = forward["beta32"]
    num_heads = qn.shape[-2]
    num_v_heads = v.shape[-2]
    total_tokens = qn.numel() // (num_heads * qn.shape[-1])
    device = qn.device
    do = do.contiguous()
    if initial_state is None:
        initial_state = forward.get("initial_state")
    has_initial_state = initial_state is not None
    if forward.get("w") is None or forward.get("u") is None:
        w, u = _launch_wy(arch, kn, v, g_cs, beta32, A, meta, num_heads, num_v_heads, total_tokens, recompute=True)
    else:
        w, u = forward["w"], forward["u"]
    if forward.get("h") is None or forward.get("v_new") is None:
        h, v_new, _ = _launch_fwd_h(
            arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, store_final_state=False
        )
    else:
        h, v_new = forward["h"], forward["v_new"]

    token_hv = (total_tokens, num_v_heads, HEAD_DIM)
    dv_local = workspace.take(OP, "b_dv_local", token_hv, v.dtype, device)
    launcher("dv_local", arch, _DV_LOCAL_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        qn,
        kn,
        do,
        g_cs,
        dv_local,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        scale,
    )

    if dht_from_dv_local is not None:
        dht = dht_from_dv_local(w, dv_local)
    dh = workspace.take(OP, "b_dh", h.shape, h.dtype, device)
    dv2 = workspace.take(OP, "b_dv2", token_hv, v.dtype, device)
    if has_initial_state:  # written in full by the scan when requested (no fill kernel); returned: fresh
        dh0 = torch.empty((meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=device)
    else:
        dh0 = _no_state((0, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32, device)
    if dht is not None:
        dht32 = dht.contiguous()
        if dht32.dtype != torch.float32:
            dht32 = dht32.float()
        use_dht = 1
    else:
        dht32, use_dht = dh0, 0
    segs = meta.bwd_segments
    scan_stage, vb, pre, pre_threads = _scan_stages(
        "dhu", segs.scan_config, device.index, arch
    )  # the planned configuration
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        # Segment-parallel reverse scan: per-segment affine maps, reverse stitch of the segment end states,
        # then the ordinary reverse scan per segment from its FP32 end state.
        if pre is None:
            raise NotImplementedError(
                f"the segmented dhu scan needs the pre-scan stage {scan_stage.replace('dhu_r', 'dhu_pre_r')} for {arch}"
            )
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks per segment the {pre} warm-up prologue scans"
            )
        seg_state = workspace.take(
            OP, "b_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM), torch.float32, device
        )
        seg_flag = workspace.take(OP, "b_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device)
        launcher(pre, arch, _DHU_PRE_ARGS)(
            segs.num_segments,
            num_v_heads,
            2 * HEAD_DIM // vb,
            qn,
            kn,
            w,
            do,
            dv_local,
            g_cs,
            dht32,
            dh,
            dv2,
            dh0,
            seg_state,
            meta.chunk_start,
            meta.chunk_len,
            segs.seg_chunk_start,
            segs.seg_seq,
            meta.seq_chunk_start,
            num_heads,
            num_v_heads,
            use_dht,
            0,
            scale,
            seg_flag,
            warmup_thresh_log2(),
        )
        dh_end = workspace.take(
            OP, "b_dh_end", (segs.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32, device
        )
        launcher("stitch", arch, _STITCH_ARGS)(
            meta.num_seqs,
            num_v_heads,
            STITCH_VSPLIT,
            seg_state,
            dht32,
            dh_end,
            segs.seq_seg_start,
            seg_flag,
            num_v_heads,
            use_dht,
            1,
        )
        dht32, use_dht = dh_end, 1
    launcher(scan_stage, arch, _DHU_ARGS)(
        segs.num_segments,
        num_v_heads,
        HEAD_DIM // vb,
        qn,
        kn,
        w,
        do,
        dv_local,
        g_cs,
        dht32,
        dh,
        dv2,
        dh0,
        dh0,
        meta.chunk_start,
        meta.chunk_len,
        segs.seg_chunk_start,
        segs.seg_seq,
        meta.seq_chunk_start,
        num_heads,
        num_v_heads,
        use_dht,
        1 if has_initial_state else 0,
        scale,
    )

    dq_hv = workspace.take(OP, "b_dq_hv", token_hv, qn.dtype, device)
    dk_hv = workspace.take(OP, "b_dk_hv", token_hv, qn.dtype, device)
    dw = workspace.take(OP, "b_dw", w.shape, w.dtype, device)
    dg1 = workspace.take(OP, "b_dg1", (total_tokens, num_v_heads), torch.float32, device)
    launcher("dqkwg", arch, _DQKWG_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        qn,
        kn,
        v_new,
        do,
        dv2,
        h,
        dh,
        g_cs,
        dq_hv,
        dk_hv,
        dw,
        dg1,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        scale,
    )

    dk2 = workspace.take(OP, "b_dk2", token_hv, qn.dtype, device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=device)  # returned
    # ``dbeta`` is produced directly in the caller's dtype (BF16 or FP32); no cast kernel follows.  Returned: fresh.
    dbeta_bf16 = dbeta_dtype == torch.bfloat16
    dbeta = torch.empty(
        forward.get("beta_shape") or (total_tokens, num_v_heads),
        dtype=torch.bfloat16 if dbeta_bf16 else torch.float32,
        device=device,
    )
    dg2 = workspace.take(OP, "b_dg2", (total_tokens, num_v_heads), torch.float32, device)
    launcher("wy_bwd_bf16" if dbeta_bf16 else "wy_bwd", arch, _WY_BWD_ARGS)(
        meta.num_chunks,
        num_v_heads,
        1,
        kn,
        v,
        A,
        dw,
        dv2,
        g_cs,
        beta32,
        dk2,
        dv,
        dbeta,
        dg2,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
    )

    dq = torch.empty(qn.shape, dtype=qn.dtype, device=device)  # returned, in q's own shape
    dk = torch.empty(kn.shape, dtype=kn.dtype, device=device)
    dg = torch.empty(forward.get("g_shape") or (total_tokens, num_v_heads), dtype=torch.float32, device=device)
    sm_count, _smem_optin, _major = _device_props(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    launcher(finalize_stage(meta.num_chunks, num_heads, num_v_heads, sm_count), arch, _FINALIZE_ARGS)(
        meta.num_chunks,
        num_heads,
        1,
        qn,
        kn,
        forward["rstd_q"],
        forward["rstd_k"],
        dq_hv,
        dk_hv,
        dk2,
        dg1,
        dg2,
        dq,
        dk,
        dg,
        meta.chunk_start,
        meta.chunk_len,
        num_heads,
        num_v_heads,
        1 if forward.get("normalize_qk", True) else 0,
    )
    return {
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": dg,
        "dbeta": dbeta,
        "dinitial_state": dh0 if has_initial_state else None,
    }


# --------------------------------------------------------------------------- #
# autograd wrapper with FLA's signature
# --------------------------------------------------------------------------- #
_OFFSET_DTYPES = (torch.int32, torch.int64)
_INPUT_NAMES = ("q", "k", "v", "g", "beta", "initial_state", "cu_seqlens")
_OPTIONAL_INPUT = 5


def _check_inputs(q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu) -> None:
    """The extents of a ``cake_gdn_fla_cp_det`` call, from tensor metadata only (no device synchronisation): CUDA
    tensors on one device, q / k ``[B, T, H, 128]``, v ``[B, T, HV, 128]`` with ``HV % H == 0``, g and beta ``[B, T,
    HV]``, ``cu_seqlens`` a 1-D int32 / int64 tensor of at least two offsets with ``B = 1`` and a host copy of as many
    offsets, ``initial_state`` one ``[HV, 128, 128]`` state per sequence.  The values of a host copy are checked when its
    layout is first planned (:func:`_chunk_meta`: they must end at the tokens of q); those of a device-only
    ``cu_seqlens`` are not (that would synchronise).  The same checks as the single-rank packages."""
    index = -1
    for i, t in enumerate((q, k, v, g, beta, initial_state, cu_seqlens)):
        if t is None and i >= _OPTIONAL_INPUT:
            continue
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            raise ValueError(f"chunk_gated_delta_rule: {_INPUT_NAMES[i]} must be a CUDA tensor")
        if i == 0:
            index = t.get_device()
        elif t.get_device() != index:
            raise ValueError(f"chunk_gated_delta_rule: {_INPUT_NAMES[i]} is on {t.device}, q on {q.device}")
    q_shape, v_shape = q.shape, v.shape
    if len(q_shape) != 4 or q_shape[3] != HEAD_DIM:
        raise ValueError(f"chunk_gated_delta_rule: q must be [B, T, H, {HEAD_DIM}]; got {tuple(q_shape)}")
    if k.shape != q_shape:
        raise ValueError(f"chunk_gated_delta_rule: k must have q's shape {tuple(q_shape)}; got {tuple(k.shape)}")
    if len(v_shape) != 4 or v_shape[0] != q_shape[0] or v_shape[1] != q_shape[1] or v_shape[3] != HEAD_DIM:
        raise ValueError(
            f"chunk_gated_delta_rule: v must be [{q_shape[0]}, {q_shape[1]}, HV, {HEAD_DIM}]; got {tuple(v_shape)}"
        )
    num_heads, num_v_heads = q_shape[2], v_shape[2]
    if num_heads == 0 or num_v_heads == 0 or num_v_heads % num_heads != 0:
        raise ValueError(
            f"chunk_gated_delta_rule: the value heads (HV={num_v_heads}) must be a positive multiple of the q / k heads "
            f"(H={num_heads})"
        )
    gate_shape = v_shape[:3]
    if g.shape != gate_shape:
        raise ValueError(f"chunk_gated_delta_rule: g must be {tuple(gate_shape)}; got {tuple(g.shape)}")
    if beta.shape != gate_shape:
        raise ValueError(f"chunk_gated_delta_rule: beta must be {tuple(gate_shape)}; got {tuple(beta.shape)}")
    if cu_seqlens is None:
        num_seqs = q_shape[0]
    else:
        offsets = cu_seqlens.numel()
        if cu_seqlens.dim() != 1 or offsets < 2 or cu_seqlens.dtype not in _OFFSET_DTYPES:
            raise ValueError(
                "chunk_gated_delta_rule: cu_seqlens must be a 1-D int32 / int64 tensor of at least two offsets; got "
                f"{cu_seqlens.dtype} {tuple(cu_seqlens.shape)}"
            )
        if q_shape[0] != 1:
            raise ValueError("chunk_gated_delta_rule: cu_seqlens requires a leading batch dimension of 1")
        num_seqs = offsets - 1
        if cu_seqlens_cpu is not None:
            count = cu_seqlens_cpu.numel() if isinstance(cu_seqlens_cpu, torch.Tensor) else len(cu_seqlens_cpu)
            if count != offsets:
                raise ValueError(f"chunk_gated_delta_rule: cu_seqlens_cpu has {count} offsets, cu_seqlens {offsets}")
    if initial_state is not None:
        state_shape = (num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM)
        if initial_state.shape != state_shape:
            raise ValueError(
                f"chunk_gated_delta_rule: initial_state must be {state_shape} (one state per sequence); got "
                f"{tuple(initial_state.shape)}"
            )


def _chunk_meta(
    cu_seqlens, cu_seqlens_cpu, batch: int, seq_len: int, device, arch: str, num_v_heads: int
) -> ChunkMeta:
    """Chunk and segment tables without a device-to-host synchronisation.

    Equal-length batches and a caller-supplied host copy of ``cu_seqlens`` take the exact tables from the
    per-layout cache (one build per micro-batch layout, shared by every layer); a device-only ``cu_seqlens``
    builds them on the GPU (padded launch extents, one launch per call).  The per-call lookup itself is a single
    dictionary hit keyed by the layout (the ``cu_seqlens_cpu`` tuple, or ``("equal", B, T)``), the head count, the
    device, the token extent and the plan pins (:func:`_plan_pins`), so a repeated layout skips the boundary validation
    and the two ``lru_cache`` lookups of :func:`chunk_meta_from_host`.  A host copy must start at 0, not decrease and
    end at the ``B * T`` tokens of the call: tokens past its last offset would never be written.
    """
    global _META_FRONT_HITS
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    total_tokens = int(batch) * int(seq_len)
    if cu_seqlens is None:
        layout = ("equal", int(batch), int(seq_len))
    elif batch != 1:
        raise ValueError("cu_seqlens requires a leading batch dimension of 1")
    elif cu_seqlens_cpu is None:
        num_seqs = int(cu_seqlens.numel()) - 1
        return build_chunk_meta_from_cu_seqlens(
            cu_seqlens,
            total_tokens,
            arch,
            scan_plans=scan_plans(total_tokens, num_seqs, int(num_v_heads), int(index), arch),
        )
    else:
        # a CPU tensor takes the validated path below
        layout = cu_seqlens_cpu if type(cu_seqlens_cpu) is tuple else None
    key = None
    if layout is not None:
        key = (layout, int(num_v_heads), int(index), total_tokens, _plan_pins())
        meta = _META_FRONT.get(key)
        if meta is not None:
            _META_FRONT_HITS += 1
            return meta
    if cu_seqlens is None:
        meta = _host_meta((int(seq_len),) * int(batch), num_v_heads, int(index), arch, total_tokens)
    else:
        seq_lens = _seq_lens_from_host(cu_seqlens_cpu)
        if sum(seq_lens) != total_tokens:
            raise ValueError(f"cu_seqlens_cpu must end at the {total_tokens} tokens of q; it ends at {sum(seq_lens)}")
        meta = chunk_meta_from_host(cu_seqlens_cpu, device, num_v_heads, total_tokens, arch)
    if key is not None:
        if len(_META_FRONT) >= _META_FRONT_LIMIT:
            del _META_FRONT[next(iter(_META_FRONT))]
        _META_FRONT[key] = meta
    return meta


def _cp_modules():
    from fla.ops.cp.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        chunk_gated_delta_rule_fwd_h_pre_process,
        compress_h0,
        expand_h0,
    )

    return (
        chunk_gated_delta_rule_fwd_h_pre_process,
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        compress_h0,
        expand_h0,
    )


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """The ``cake_gdn_fla_cp_det`` kernels with FLA's autograd contract, including ``cp_context`` state passing
    (:func:`cp_chunk_gated_delta_rule`).  ``save_state`` keeps w / u / h / v_new on the tape (else the backward recomputes
    them, with the same bits).

    The incoming gradients are not materialised: an output whose gradient does not reach the backward arrives as
    ``None``.  An unused final state then adds no final-state gradient (the bits of the call without one), and a loss
    on the final state alone runs with a zero output gradient."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool,
        cp_context: Any,
        save_state: bool,
    ):
        batch, seq_len, num_heads, key_dim = q.shape
        num_v_heads, value_dim = v.shape[2], v.shape[3]
        index = q.device.index if q.device.index is not None else torch.cuda.current_device()
        arch = _arch_of(index)
        tokens = batch * seq_len
        # ``[B, T, H, D]`` passes through unflattened: the kernels address tokens through the chunk table and never see a
        # shape, and every per-token output / gradient is allocated in its input's own shape (no reshape per call).
        meta = _chunk_meta(cu_seqlens, cu_seqlens_cpu, batch, seq_len, q.device, arch, num_v_heads)

        # a context without a process group runs as one rank; the entry point rejected a state under any context
        use_cp = cp_context is not None and getattr(cp_context, "group", None) is not None

        # prep + WY first; CP derives this rank's incoming state from w/u
        fwd: dict[str, Any]
        if use_cp:
            fwd_h_pre, _bwd_pre, compress_h0, _expand_h0 = _cp_modules()
            fwd = gdn_chunk_forward_stage1(
                q, k, v, g, beta, meta=meta, scale=scale, normalize_qk=use_qk_l2norm_in_kernel
            )
            initial_state = fwd_h_pre(
                k=fwd["kn"].reshape(1, tokens, num_heads, key_dim),
                w=fwd["w"].reshape(1, tokens, num_v_heads, key_dim),
                u=fwd["u"].reshape(1, tokens, num_v_heads, value_dim),
                g=fwd["g_cs"].reshape(1, tokens, num_v_heads),
                cu_seqlens=cu_seqlens,
                initial_state=None,
                context=cp_context,
                state_v_first=False,
                chunk_size=CHUNK,
            )
            fwd = gdn_chunk_forward_stage2(fwd, initial_state=initial_state, output_final_state=output_final_state)
            saved_state = compress_h0(initial_state, context=cp_context)
        else:
            fwd = gdn_chunk_forward(
                q,
                k,
                v,
                g,
                beta,
                meta=meta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                normalize_qk=use_qk_l2norm_in_kernel,
            )
            saved_state = initial_state

        # The WY / state tensors stay on the tape by default: the backward would otherwise rerun the WY and state-scan
        # kernels (bit-identical either way, ~20 % of the backward's device time at 8k tokens).
        state_tensors = (fwd["w"], fwd["u"], fwd["h"], fwd["v_new"]) if save_state else ()
        ctx.save_for_backward(
            fwd["qn"],
            fwd["kn"],
            fwd["rstd_q"],
            fwd["rstd_k"],
            fwd["v"],
            fwd["g_cs"],
            fwd["beta32"],
            fwd["A"],
            saved_state,
            cu_seqlens,
            *state_tensors,
        )
        ctx.saved_state_tensors = bool(state_tensors)
        ctx.meta = meta
        ctx.arch = arch
        ctx.scale = float(scale)
        ctx.shapes = (batch, seq_len, num_heads, num_v_heads, key_dim, value_dim)
        ctx.grad_shapes = (g.shape, beta.shape)
        ctx.normalize_qk = bool(use_qk_l2norm_in_kernel)
        ctx.cp_context = cp_context if use_cp else None
        ctx.has_initial_state_input = initial_state is not None and not use_cp
        ctx.dtypes = (g.dtype, beta.dtype)
        ctx.set_materialize_grads(False)
        output = fwd["output"]  # allocated in v's shape [B, T, HV, V]
        if output.dtype != q.dtype:
            output = output.to(q.dtype)
        ctx.output_dtype = output.dtype
        final_state = fwd["final_state"] if output_final_state else None
        return output, final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor | None, dht: torch.Tensor | None):
        qn, kn, rstd_q, rstd_k, v_flat, g_cs, beta32, A, saved_state, cu_seqlens, *state_tensors = ctx.saved_tensors
        batch, seq_len, num_heads, num_v_heads, key_dim, value_dim = ctx.shapes
        g_shape, beta_shape = ctx.grad_shapes
        g_dtype, beta_dtype = ctx.dtypes
        if do is None:  # only the final state carries a gradient
            do = torch.zeros((batch, seq_len, num_v_heads, value_dim), dtype=ctx.output_dtype, device=qn.device)
        do = do.contiguous()  # [B, T, HV, V]; a no-op for the usual contiguous incoming gradient
        tape = {
            "arch": ctx.arch,
            "qn": qn,
            "kn": kn,
            "rstd_q": rstd_q,
            "rstd_k": rstd_k,
            "v": v_flat,
            "g_cs": g_cs,
            "beta32": beta32,
            "A": A,
            "meta": ctx.meta,
            "scale": ctx.scale,
            "normalize_qk": ctx.normalize_qk,
            "g_shape": g_shape,
            "beta_shape": beta_shape,
        }
        if ctx.saved_state_tensors:
            tape["w"], tape["u"], tape["h"], tape["v_new"] = state_tensors
        cp_context = ctx.cp_context
        initial_state = saved_state
        dht_hook = None
        if cp_context is not None:
            _fwd_pre, bwd_dhu_pre, _compress_h0, expand_h0 = _cp_modules()
            initial_state = expand_h0(saved_state, context=cp_context)
            tokens = batch * seq_len

            def dht_hook(w, dv_local):
                dht_local, _ = bwd_dhu_pre(
                    q=qn.reshape(1, tokens, num_heads, key_dim),
                    k=kn.reshape(1, tokens, num_heads, key_dim),
                    w=w.reshape(1, tokens, num_v_heads, key_dim),
                    do=do.reshape(1, tokens, num_v_heads, value_dim),
                    dv=dv_local.reshape(1, tokens, num_v_heads, value_dim),
                    g=g_cs.reshape(1, tokens, num_v_heads),
                    scale=ctx.scale,
                    cu_seqlens=cu_seqlens,
                    dht=None,
                    initial_state=initial_state,
                    context=cp_context,
                    state_v_first=False,
                    chunk_size=CHUNK,
                )
                return dht_local

        grads = gdn_chunk_backward(
            tape, do, dht, initial_state=initial_state, dht_from_dv_local=dht_hook, dbeta_dtype=beta_dtype
        )
        dq, dk, dv = grads["dq"], grads["dk"], grads["dv"]  # allocated in q / k / v's own shapes
        dg = grads["dg"]  # allocated in g's shape, FP32
        if dg.dtype != g_dtype:
            dg = dg.to(g_dtype)
        dbeta = grads["dbeta"]  # allocated in beta's shape and dtype
        if dbeta.dtype != beta_dtype:
            dbeta = dbeta.to(beta_dtype)
        dh0 = grads["dinitial_state"] if ctx.has_initial_state_input else None
        return dq, dk, dv, dg, dbeta, None, dh0, None, None, None, None, None, None


class _NoTokens(torch.autograd.Function):
    """A call with no token: an empty output, and the final state is the initial state (zero without one), as the
    recurrence gives it.  The gradient of the initial state is the final state's."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state, output_final_state: bool, num_seqs: int):
        ctx.inputs = tuple((t.shape, t.dtype, t.device) for t in (q, k, v, g, beta))
        ctx.state_dtype = None if initial_state is None else initial_state.dtype
        output = torch.empty(v.shape, dtype=q.dtype, device=v.device)
        if not output_final_state:
            return output, None
        if initial_state is not None:
            return output, initial_state.detach().float().clone()
        return output, torch.zeros((num_seqs, v.shape[-2], HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=v.device)

    @staticmethod
    def backward(ctx, do, dht):
        grads = [torch.zeros(shape, dtype=dtype, device=device) for shape, dtype, device in ctx.inputs]
        dh0 = None if ctx.state_dtype is None or dht is None else dht.to(ctx.state_dtype)
        return (*grads, dh0, None, None)


def _no_tokens(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens):
    """:class:`_NoTokens` of a call whose q holds no token (``B * T == 0``)."""
    num_seqs = q.shape[0] if cu_seqlens is None else int(cu_seqlens.numel()) - 1
    if initial_state is not None and tuple(initial_state.shape) != (num_seqs, v.shape[-2], HEAD_DIM, HEAD_DIM):
        raise ValueError(
            f"chunk_gated_delta_rule: initial_state must be {(num_seqs, v.shape[-2], HEAD_DIM, HEAD_DIM)} (one state "
            f"per sequence); got {tuple(initial_state.shape)}"
        )
    return _NoTokens.apply(q, k, v, g, beta, initial_state, bool(output_final_state), num_seqs)


# --------------------------------------------------------------------------- #
# the vendored Cake packages: det and fast, single-rank and KCP
# --------------------------------------------------------------------------- #
# name -> the package under ``_vendor/``
_PACKAGES = {
    "det": "cake_gdn_det_train",
    "fast": "cake_gdn_nondet_train",
    "det_kcp": "cake_gdn_det_train_kcp",
    "fast_kcp": "cake_gdn_nondet_train_kcp",
}


@cache
def _package(name: str):
    """The vendored package ``name`` with the runtime loader installed: ``det`` is ``cake_gdn_det_train``, ``fast``
    ``cake_gdn_nondet_train``, ``det_kcp`` ``cake_gdn_det_train_kcp`` and ``fast_kcp``
    ``cake_gdn_nondet_train_kcp``."""
    try:
        return loader.load_package(_PACKAGES[name])
    except KeyError:
        raise ValueError(f"package must be one of {sorted(_PACKAGES)}, not {name!r}") from None


def _share_scratch(name: str) -> None:
    """Point the scratch pool of the single-rank package ``name`` (``cake_chain._POOLED``, read when a call runs) at
    the runtime's workspace pool.

    The package keeps one buffer per role, shape and stream, up to four shapes per role, until the process ends; through
    the runtime pool its scratch is a fresh allocation per call at the default ``--cake-workspace-pool-bytes`` of 0,
    and under a positive bound one buffer per role and stream, freed by ``release()``.  Each role is taken once per
    forward or backward call on the caller's stream, never saved or returned, so one buffer per role and stream serves
    it.  The 0-element state placeholders (``f_zero_state`` / ``b_zero_state``) are this module's (:func:`_no_state`).
    The package's self-zeroing buffers (``take_zeroed``, kept zero by every launch that uses them) stay with the
    package: :func:`release` drops them.
    """
    chain = _package(name).cake_chain

    class SharedScratch(chain.Workspaces):
        def table(self, device_index: int) -> None:
            return None

        def take(self, table, role, shape, dtype, device):
            if role in _NO_STATE_ROLES:
                return _no_state(shape, dtype, device)
            return workspace.take(OP, role, shape, dtype, device)

    chain._POOLED = SharedScratch()


for _name in ("det", "fast"):
    _share_scratch(_name)
del _name


def _vendored_call(
    variant, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu, use_qk_l2norm
):
    """``variant``'s single-rank package with ``--cake-gdn-recompute-state`` and the package's own plan defaults.

    ``cu_seqlens_cpu`` selects the package's host-table route, as it selects the generated kernels' host table.  The
    packages check the extents of every input and make strided q / k / v contiguous themselves.
    """
    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    _arch_of(index)  # SM100a / SM103a / SM107a, else NotImplementedError
    package = _package(variant)
    kwargs = {
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": output_final_state,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_cpu": cu_seqlens_cpu,
        "use_qk_l2norm_in_kernel": use_qk_l2norm,
        "save_state": _save_state(),
    }
    if variant == "det":
        return package.cake_gdn_det(q, k, v, g, beta, **kwargs)
    return package.cake_gdn_nondet(q, k, v, g, beta, **kwargs)


# --------------------------------------------------------------------------- #
# the KCP packages (a KcpContext call)
# --------------------------------------------------------------------------- #
_KCP_LOGGED: set[str] = set()


def _log_fla_cp(variant: str, reason: str) -> None:
    """Log once per reason that a ``KcpContext`` call runs ``cake_gdn_fla_cp_det``, the det kernels of FLA's context
    parallelism, because ``variant``'s KCP package does not serve it."""
    if variant == "fast":
        config.warn_runs_det(OP, f"{reason} with a KcpContext")
        return
    if reason in _KCP_LOGGED:
        return
    _KCP_LOGGED.add(reason)
    logger.warning(
        "miles_ext_kernels.cake.gdn: the det KCP package does not serve %s; running the det kernels of FLA's context "
        "parallelism",
        reason,
    )


def _kcp_operand(x: torch.Tensor) -> torch.Tensor:
    """``x`` as the KCP packages read q / k / v: dense ``[heads, 128]`` rows at any token stride (a packed-row view
    passes through), else a contiguous copy (miles' per-key-head q / k views: one copy of q and of k per call)."""
    heads, dim = x.shape[-2], x.shape[-1]
    if x.stride(-1) == 1 and x.stride(-2) == dim and (x.shape[-3] <= 1 or x.stride(-3) >= heads * dim):
        return x
    return x.contiguous()


def _kcp_checks(q, v, kcp_context: KcpContext) -> None:
    """The errors a ``KcpContext`` call raises under every package."""
    if q.dim() == 4 and q.shape[0] != 1:
        raise ValueError("chunk_gated_delta_rule: context parallelism needs a leading batch dimension of 1")
    if q.shape[-3] != kcp_context.num_tokens:
        raise ValueError(
            f"chunk_gated_delta_rule: the operands carry {q.shape[-3]} tokens, rank {kcp_context.rank} of the "
            f"KcpContext holds {kcp_context.num_tokens}"
        )
    num_heads, num_v_heads = q.shape[-2], v.shape[-2]
    if num_heads < 1 or num_v_heads % num_heads:
        raise ValueError(f"chunk_gated_delta_rule: num_v_heads={num_v_heads} must be a multiple of {num_heads}")


def _kcp_rejection(variant: str, use_qk_l2norm: bool, kcp_context: KcpContext) -> str | None:
    """Why ``variant``'s KCP package does not serve a call, or ``None`` when it does."""
    package = _package(f"{variant}_kcp")
    if use_qk_l2norm and not package.cake_kcp.l2n_records_shipped():
        return "the in-kernel q / k L2 norm (no records in this package build)"
    try:
        kcp_context.context(variant)
    except ValueError as exc:  # a partition beyond the package's plan caps, refused before any launch
        return f"this partition ({exc})"
    return None


def _kcp_apply(variant: str, q, k, v, g, beta, scale: float, use_qk_l2norm: bool, kcp_context: KcpContext):
    """``variant``'s KCP package on this rank's tokens; returns the output (``[B, T, HV, V]`` like ``v``).

    ``g`` and ``beta`` go in as contiguous FP32 (``beta`` is cast exactly from BF16 when needed: one ``[T, HV]`` FP32
    copy forward, and autograd rounds the FP32 ``dbeta`` back to ``beta``'s dtype).  ``--cake-gdn-recompute-state``
    picks the package's ``save_mode`` (``"full"`` keeps the tapes, ``"light"`` replays them in the backward);
    :data:`KCP_OPTIONS` is passed as ``options``.
    """
    package = _package(f"{variant}_kcp")
    ctx = kcp_context.context(variant)
    q, k, v = _kcp_operand(q), _kcp_operand(k), _kcp_operand(v)
    g = g.contiguous()
    beta = (beta if beta.dtype == torch.float32 else beta.float()).contiguous()
    kwargs = {
        "scale": scale,
        "save_mode": "full" if _save_state() else "light",
        "use_qk_l2norm_in_kernel": bool(use_qk_l2norm),
        "options": None if KCP_OPTIONS is None else dict(KCP_OPTIONS),
    }
    if variant == "det":
        return package.cake_gdn_det_kcp(q, k, v, g, beta, ctx, **kwargs)
    return package.cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, **kwargs)


def kcp_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    kcp_context: KcpContext,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
) -> torch.Tensor:
    """The KCP package of the configured variant on this rank's tokens of ``kcp_context``; returns the output.

    ``det`` runs ``cake_gdn_det_train_kcp`` (bit for bit run to run for a fixed partition), ``fast``
    ``cake_gdn_nondet_train_kcp`` (within the det package's accuracy budgets).  The operands are this rank's tokens in
    the context's local order (``kcp_context.num_tokens``, local sequences ``kcp_context.cu_seqlens``): q / k ``[1,
    T_r, H, 128]``, v ``[1, T_r, HV, 128]`` BF16 with ``HV % H == 0``, g ``[1, T_r, HV]`` (FP32), beta ``[1, T_r, HV]``
    after the sigmoid (BF16 or FP32).  No initial or final state.  Unlike :func:`chunk_gated_delta_rule` it runs the
    KCP package only: a call the package does not serve raises.
    """
    if not isinstance(kcp_context, KcpContext):
        raise TypeError(f"kcp_context must be a KcpContext (build_kcp_context), not {type(kcp_context).__name__}")
    g, scale = _checked(q, k, v, g, scale, chunk_size, {})
    _kcp_checks(q, v, kcp_context)
    variant = config.variant(OP)
    reason = _kcp_rejection(variant, use_qk_l2norm_in_kernel, kcp_context)
    if reason is not None:
        raise NotImplementedError(f"kcp_chunk_gated_delta_rule: the {variant} KCP package does not serve {reason}")
    return _kcp_apply(variant, q, k, v, g, beta, scale, use_qk_l2norm_in_kernel, kcp_context)


def _kcp_call(variant: str, q, k, v, g, beta, scale, use_qk_l2norm, kcp_context: KcpContext):
    """A :func:`chunk_gated_delta_rule` call with a ``KcpContext`` (the context-parallel contract already checked):
    the KCP package of ``variant``, or ``cake_gdn_fla_cp_det`` on the same tokens for a call it does not serve
    (logged once per reason)."""
    _kcp_checks(q, v, kcp_context)
    reason = _kcp_rejection(variant, use_qk_l2norm, kcp_context)
    if reason is None:
        return _kcp_apply(variant, q, k, v, g, beta, scale, use_qk_l2norm, kcp_context), None
    if not kcp_context.fla_compatible:
        raise NotImplementedError(
            f"chunk_gated_delta_rule: the {variant} KCP package does not serve {reason}, and the det kernels of FLA's "
            "context parallelism need a contiguous partition whose world_size divides the token count "
            f"(this one is {kcp_context.layout!r})"
        )
    _log_fla_cp(variant, reason)
    fla = kcp_context.fla_context()
    cu_seqlens, cu_seqlens_cpu = _cp_contract(fla, None, False, None, None)
    return _cp_apply(
        q, k, v, g, beta, scale, None, False, cu_seqlens, cu_seqlens_cpu, use_qk_l2norm, fla, _save_state()
    )


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def _checked(q, k, v, g, scale, chunk_size, unsupported) -> tuple[torch.Tensor, float]:
    """The checks every package shares; returns ``g`` in FP32 and the scale (``1 / sqrt(K)`` by default)."""
    for name, value in unsupported.items():
        if value not in (None, False):
            raise NotImplementedError(f"chunk_gated_delta_rule: {name}={value!r} is not supported by the Cake kernels")
    if chunk_size != CHUNK:
        raise NotImplementedError(f"chunk_gated_delta_rule: chunk_size must be {CHUNK}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise NotImplementedError("chunk_gated_delta_rule: q/k/v must be bfloat16")
    if q.shape[-1] != HEAD_DIM or v.shape[-1] != HEAD_DIM:
        raise NotImplementedError("chunk_gated_delta_rule: head dims must be 128")
    if g.dtype != torch.float32:
        g = g.float()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return g, float(scale)


def _cp_contract(cp_context, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu):
    """FLA's context-parallel contract: with a ``cp_context`` no initial state and no final state, and the context's own
    ``cu_seqlens`` (and its host copy, when it has one) in place of the arguments.  Returns the boundaries to use."""
    if cp_context is None:
        return cu_seqlens, cu_seqlens_cpu
    if initial_state is not None:
        raise ValueError("chunk_gated_delta_rule: initial_state is not supported under context parallelism")
    if output_final_state:
        raise ValueError("chunk_gated_delta_rule: output_final_state is not supported under context parallelism")
    local = getattr(cp_context, "cu_seqlens", None)
    if local is None:
        raise ValueError("chunk_gated_delta_rule: context parallelism needs the cp_context's cu_seqlens")
    host = getattr(cp_context, "cu_seqlens_cpu", None)
    return local, (cu_seqlens_cpu if host is None else host)


def _cp_apply(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    cu_seqlens_cpu,
    use_qk_l2norm,
    cp_context,
    save_state,
):
    """``cake_gdn_fla_cp_det`` on checked inputs (the context-parallel contract applied)."""
    _check_inputs(q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
    if q.shape[0] * q.shape[1] == 0 and getattr(cp_context, "group", None) is None:
        return _no_tokens(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens)
    return ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm,
        cp_context,
        save_state,
    )


def cp_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | Sequence[int] | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cp_context: Any = None,
    chunk_size: int = 64,
    **unsupported: Any,
):
    """FLA v0.5.2 ``chunk_gated_delta_rule`` signature on ``cake_gdn_fla_cp_det``, the det kernels with FLA's context
    parallelism; :func:`chunk_gated_delta_rule` runs every call with an FLA ``cp_context`` here, and the ``KcpContext``
    calls the KCP packages do not serve.

    A ``cp_context`` with a process group passes the state between the ranks as FLA does; without one (a context whose
    ``group`` is ``None``) the kernels run as one rank.  Under any ``cp_context`` the contract is FLA's: no
    ``initial_state``, no final state, and the context's ``cu_seqlens`` / ``cu_seqlens_cpu`` in place of the arguments.
    A ``KcpContext`` runs on its FLA context (:meth:`~miles_ext_kernels.cake._impl.gdn.kcp.KcpContext.fla_context`).
    The inputs and options are those of :func:`chunk_gated_delta_rule`; the plan pins (``FORCED_*``) and
    ``--cake-gdn-recompute-state`` apply.
    """
    g, scale = _checked(q, k, v, g, scale, chunk_size, unsupported)
    if isinstance(cp_context, KcpContext):
        cp_context = cp_context.fla_context()
    cu_seqlens, cu_seqlens_cpu = _cp_contract(
        cp_context, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu
    )
    return _cp_apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
        cp_context,
        _save_state(),
    )


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | Sequence[int] | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cp_context: Any = None,
    chunk_size: int = 64,
    **unsupported: Any,
):
    """FLA v0.5.2 ``chunk_gated_delta_rule`` signature on the package of the configured variant.

    Supported: ``[B, T, H, K]`` q/k and ``[B, T, HV, V]`` v with ``K = V = 128`` and ``HV % H == 0``,
    BF16 q/k/v, packed varlen via ``cu_seqlens`` (``B = 1``), FP32 ``initial_state`` /
    ``output_final_state`` (V-last ``[N, HV, K, V]``), ``use_qk_l2norm_in_kernel``, and ``cp_context``
    state passing.  Gate fusion (``use_gate_in_kernel``), beta sigmoid fusion, ``state_v_first``,
    ``head_first``, and non-64 chunk sizes are rejected explicitly.  A call with no token returns an empty output (and
    the initial state as the final state).

    ``cu_seqlens_cpu`` (the same boundaries on the host: a tuple of ints such as Megatron's
    ``packed_seq_params.cu_seqlens_host``, or a CPU tensor) selects the exact per-layout chunk table that
    every layer of the micro-batch shares; without it the table is built on the device per call.

    Without a ``cp_context``, ``det`` runs ``cake_gdn_det_train`` and ``fast`` ``cake_gdn_nondet_train``.  Under any
    ``cp_context`` the contract is FLA's: ``initial_state`` and ``output_final_state`` are rejected, and the context's
    boundaries replace ``cu_seqlens`` / ``cu_seqlens_cpu``.  With a
    :class:`~miles_ext_kernels.cake._impl.gdn.kcp.KcpContext` the operands are this rank's tokens of it and the call runs
    the KCP package of the variant (:func:`kcp_chunk_gated_delta_rule`); a call the KCP package does not serve (the
    in-kernel L2 norm without its records in the package build, or a partition beyond the package's plan caps) runs
    ``cake_gdn_fla_cp_det`` on the context's FLA context and logs it once per reason, which needs a contiguous
    partition whose ``world_size`` divides the token count (otherwise the call raises).  A call with an FLA
    ``cp_context`` runs ``cake_gdn_fla_cp_det`` (:func:`cp_chunk_gated_delta_rule`) under either variant (``fast`` logs
    it once).
    """
    g, scale = _checked(q, k, v, g, scale, chunk_size, unsupported)
    variant = config.variant(OP)
    if isinstance(cp_context, KcpContext):
        _cp_contract(cp_context, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu)
        return _kcp_call(variant, q, k, v, g, beta, scale, use_qk_l2norm_in_kernel, cp_context)
    if cp_context is not None:
        if variant == "fast":
            config.warn_runs_det(OP, "an FLA cp_context")
        cu_seqlens, cu_seqlens_cpu = _cp_contract(
            cp_context, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu
        )
        return _cp_apply(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            cu_seqlens_cpu,
            use_qk_l2norm_in_kernel,
            cp_context,
            _save_state(),
        )
    if q.shape[0] * q.shape[1] == 0:
        return _no_tokens(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens)
    return _vendored_call(
        variant,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
    )


# --------------------------------------------------------------------------- #
# prebuild and release
# --------------------------------------------------------------------------- #
def prebuild(*, device: Any = None, max_workers: int | None = None) -> list[Path]:
    """Build (or find in the cache) every library of the five gdn packages for the architecture of ``device``.

    No model parameter: every call of the op on that architecture (either variant, with or without context
    parallelism, host or device boundaries, any layout and token count) launches libraries of this set, so after it
    returns no call compiles one of them.  One exception: the third-party Triton kernels that a caller-supplied FLA
    ``cp_context`` runs (FLA's own context-parallel pre-processing) are not prewarmed; they compile in FLA on their
    first call.  ``NotImplementedError`` for a device other than SM100a / SM103a / SM107a, ``RuntimeError`` when the
    toolkit is older than CUDA 13.4.  Returns the cache paths.
    """
    arch = loader.device_arch(device)
    paths: list[Path] = []
    for package in registry.op_spec(OP).packages:
        cake_build = loader.load_package(package).cake_build
        names = [name for name, entry in cake_build.libraries_by_name().items() if entry["arch"] == arch]
        paths += loader.prebuild_libraries(package, names, max_workers=max_workers)
    return paths


def release() -> None:
    """Drop what the gdn kernels keep between calls besides the workspace pool (which ``release`` empties itself): the
    chunk tables and plans of ``cake_gdn_fla_cp_det`` and its loaded launchers, the cached ``KcpContext`` objects, and
    the self-zeroing buffers of the single-rank packages.  The packages' own chunk tables and plans (a few int32
    tensors per layout) stay."""
    host_cache_clear()
    native.release()
    from . import kcp

    kcp.release()
    for name in ("det", "fast"):
        getattr(_package(name).cake_chain._POOLED, "zeroed", {}).clear()
