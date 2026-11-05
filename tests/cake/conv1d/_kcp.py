"""Shared helpers of the context-parallel short-conv tests: fragment tables, a one-rank context, the contexts of the
GDN / KDA context-parallel routes, and an FP64 reference of the whole pack.

The short conv takes the context of the layer's GDN or KDA context-parallel route. :func:`context_builder` returns
the partition and context builders of those vendored packages when this checkout carries them (``None`` otherwise),
and of ``"stand-in"``: :func:`build_stand_in_context` builds a rank's context with just the fields the short conv
reads, from the halo plan's rules, so the short conv's tests also run on a tree without those packages and check
the contexts of those packages against an independent one. :func:`one_rank_context` is the context of a single rank
holding every fragment (its halo exchange is the identity).
"""

from __future__ import annotations

import importlib
import importlib.util
import types

import torch

WIDTH = 4
HALO = WIDTH - 1
# The packages' contract: y / dx within one BF16 ulp of the FP64 reference (no element outside), the FP32
# weight-gradient partial within atol 5e-2 + rtol 1e-4 of the FP64 one.
Y_ATOL, Y_RTOL = 1e-3, 2**-7
DW_ATOL, DW_RTOL = 5e-2, 1e-4

# name -> (vendored package of the context-parallel route, partition builder, context builder); "stand-in" is
# :func:`partition_stand_in` / :func:`build_stand_in_context` of this module, always available
CONTEXT_BUILDERS = {
    "stand-in": (None, "partition_stand_in", "build_stand_in_context"),
    "gdn-det": ("miles_ext_kernels.cake._vendor.cake_gdn_det_train_kcp", "partition_det_kcp", "build_context_det_kcp"),
    "gdn-fast": ("miles_ext_kernels.cake._vendor.cake_gdn_train_kcp", "partition_kcp", "build_context_kcp"),
    "kda-det": ("miles_ext_kernels.cake._vendor.cake_kda_det_train_kcp", "partition_det_kcp", "build_context_det_kcp"),
    "kda-fast": ("miles_ext_kernels.cake._vendor.cake_kda_train_kcp", "partition_kcp", "build_context_kcp"),
}


VENDORED_BUILDERS = tuple(sorted(name for name, (package, _, _) in CONTEXT_BUILDERS.items() if package))


def context_builder(name: str):
    """``(partition, build_context)`` of the context-parallel route ``name`` (``CONTEXT_BUILDERS``), or ``None`` when
    this checkout does not vendor its package. ``partition(cu_seqlens, world_size, layout, fragments=None)``,
    ``build_context(partition, group, rank, device, conv_kernel_size=4)``."""
    package, partition, build = CONTEXT_BUILDERS[name]
    if package is None:
        return globals()[partition], globals()[build]
    if importlib.util.find_spec(package) is None:
        return None
    plan = importlib.import_module(f"{package}.cake_kcp_plan")
    return getattr(plan, partition), getattr(plan, build)


def fragments(lens, cuts):
    """``(S, cu, frags)``: the packed documents' prefix sums and the ``(doc, offset, length)`` fragments of the cut
    lengths ``cuts`` over the pack, split at document boundaries too (an empty cut is an empty fragment)."""
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    S = cu[-1]
    assert sum(cuts) == S, (cuts, S)
    frags = []
    pos = 0
    for n in cuts:
        end = pos + n
        if n == 0:
            d = max(i for i in range(len(lens)) if cu[i] <= pos) if pos < S else len(lens) - 1
            frags.append((d, pos - cu[d], 0))
        p = pos
        while p < end:
            d = max(i for i in range(len(lens)) if cu[i] <= p)
            e = min(end, cu[d + 1])
            frags.append((d, p - cu[d], e - p))
            p = e
        pos = end
    return S, tuple(cu), frags


def _near_equal(total: int, n: int) -> list[tuple[int, int]]:
    base, extra = divmod(total, n)
    bounds, pos = [], 0
    for i in range(n):
        size = base + (i < extra)
        bounds.append((pos, pos + size))
        pos += size
    return bounds


def partition_stand_in(cu_seqlens, world_size: int, layout: str = "contiguous", fragments=None):
    """A partition of the pack over ``world_size`` ranks with the fields the tests read (``world_size``,
    ``fragments[rank]`` of ``doc`` / ``off`` / ``length`` in document, then offset order): ``layout``
    ``"contiguous"`` (near-equal token intervals), ``"zigzag"`` (``2 W`` near-equal intervals, rank ``r`` holding
    ``r`` and ``2 W - 1 - r``), ``"zigzag_per_document"`` (the same per document), or the explicit per-rank
    ``(doc, off, length)`` tables ``fragments``."""
    cu = [int(c) for c in cu_seqlens]
    W = int(world_size)
    if fragments is not None:
        per_rank = [list(f) for f in fragments]
    else:
        intervals = []  # (start, end, rank) in pack coordinates
        if layout == "contiguous":
            intervals = [(a, b, r) for r, (a, b) in enumerate(_near_equal(cu[-1], W))]
        elif layout == "zigzag":
            pieces = _near_equal(cu[-1], 2 * W)
            intervals = [(*pieces[i], min(i, 2 * W - 1 - i)) for i in range(2 * W)]
        elif layout == "zigzag_per_document":
            for d in range(len(cu) - 1):
                pieces = _near_equal(cu[d + 1] - cu[d], 2 * W)
                intervals += [(cu[d] + a, cu[d] + b, min(i, 2 * W - 1 - i)) for i, (a, b) in enumerate(pieces)]
        else:
            raise ValueError(f"unknown layout {layout!r}")
        per_rank = [[] for _ in range(W)]
        for a, b, r in intervals:
            for d in range(len(cu) - 1):
                lo, hi = max(a, cu[d]), min(b, cu[d + 1])
                if hi > lo:
                    per_rank[r].append((d, lo - cu[d], hi - lo))
    table = tuple(
        tuple(types.SimpleNamespace(doc=d, off=off, length=length) for d, off, length in sorted(entries) if length > 0)
        for entries in per_rank
    )
    return types.SimpleNamespace(world_size=W, cu_seqlens=tuple(cu), fragments=table)


def build_stand_in_context(part, group, rank: int, device, conv_kernel_size: int = WIDTH):
    """The context of ``rank`` with the fields the short conv reads, from the halo plan's rules: the
    ``conv_kernel_size - 1`` tokens before a continuing fragment are history rows, sent by the ranks holding them
    (one send block per destination rank, one receive block per source rank, each in (consumer fragment, tap)
    order), and the backward sends the derivative of a fragment's first three tokens to every rank whose fragment
    ends right before them (in (consumer fragment, position) order)."""
    assert conv_kernel_size == WIDTH
    W = part.world_size
    owner = {}  # (doc, token) -> (rank, slot, position in the fragment)
    local_cu = []
    for r in range(W):
        offsets = [0]
        for slot, f in enumerate(part.fragments[r]):
            offsets.append(offsets[-1] + f.length)
            for t in range(f.off, f.off + f.length):
                owner[(f.doc, t)] = (r, slot, t - f.off)
        local_cu.append(offsets)
    # forward: (consumer rank, consumer slot, tap) <- (source rank, source local row)
    hist = []
    # backward: (consumer rank, consumer slot, tail position) <- (source rank, source slot, head position)
    head = []
    for r in range(W):
        for slot, f in enumerate(part.fragments[r]):
            if f.length == 0:  # an empty fragment has no outputs and no tail rows
                continue
            for k in range(HALO):
                token = f.off - HALO + k
                if f.off and token >= 0:
                    s, src_slot, position = owner[(f.doc, token)]
                    hist.append(((r, slot, k), (s, local_cu[s][src_slot] + position)))
            for m in range(HALO):
                source = owner.get((f.doc, f.off + f.length + m))
                if source is not None and source[2] < HALO:
                    head.append(((r, slot, m), source))
    hist.sort()
    head.sort()
    frags = part.fragments[rank]
    n_frag = len(frags)
    hist_row_off = [[-1] * HALO for _ in range(n_frag)]
    tail_row_off = [[-1] * HALO for _ in range(n_frag)]
    head_row_off = [[[-1] * HALO for _ in range(HALO)] for _ in range(n_frag)]
    hist_index = []
    fwd_send, fwd_recv, bwd_send, bwd_recv = ([0] * W for _ in range(4))
    received = 0
    for s in range(W):  # receive blocks in source-rank order
        for (r, slot, k), (src, _) in hist:
            if r == rank and src == s:
                hist_row_off[slot][k] = received
                received += 1
                fwd_recv[s] += 1
    for r in range(W):  # send blocks in destination-rank order
        for (dst, _, _), (src, row) in hist:
            if dst == r and src == rank:
                hist_index.append(row)
                fwd_send[r] += 1
    received = 0
    for s in range(W):
        for (r, slot, m), (src, _, _) in head:
            if r == rank and src == s:
                tail_row_off[slot][m] = received
                received += 1
                bwd_recv[s] += 1
    sent = 0
    for r in range(W):
        for (dst, _, _), (src, succ_slot, position) in head:
            if dst == r and src == rank:
                consumers = head_row_off[succ_slot][position]
                consumers[consumers.index(-1)] = sent
                sent += 1
                bwd_send[r] += 1
    i32 = torch.int32
    halo = types.SimpleNamespace(
        fwd_send_rows=tuple(fwd_send),
        fwd_recv_rows=tuple(fwd_recv),
        bwd_send_rows=tuple(bwd_send),
        bwd_recv_rows=tuple(bwd_recv),
        n_hist_rows=sum(fwd_recv),
        n_head_rows=sum(bwd_send),
        n_tail_rows=sum(bwd_recv),
    )
    return types.SimpleNamespace(
        rank=rank,
        world_size=W,
        group=group,
        conv_kernel_size=WIDTH,
        T_r=local_cu[rank][-1],
        n_frag=n_frag,
        cu_seqlens_local=torch.tensor(local_cu[rank], dtype=i32, device=device),
        cu_seqlens_local_cpu=tuple(local_cu[rank]),
        halo=halo,
        halo_tables={
            "hist_row_off": torch.tensor(hist_row_off, dtype=i32, device=device).reshape(n_frag, HALO),
            "tail_row_off": torch.tensor(tail_row_off, dtype=i32, device=device).reshape(n_frag, HALO),
            "head_row_off": torch.tensor(head_row_off, dtype=i32, device=device).reshape(n_frag, HALO, HALO),
            "hist_index": torch.tensor(hist_index, dtype=torch.int64, device=device),
        },
    )


def one_rank_context(frags, device):
    """The context of one rank holding every fragment of ``frags`` (``(doc, off, length)``, in document, then
    offset order; empty ones included): its halo exchange is the identity."""
    part = types.SimpleNamespace(
        world_size=1,
        fragments=(tuple(types.SimpleNamespace(doc=d, off=off, length=length) for d, off, length in frags),),
    )
    return build_stand_in_context(part, None, 0, device)


def split_pack(partition, lengths, world_size: int, layout: str):
    """``(part, cu)``: the pack of documents ``lengths`` cut over ``world_size`` ranks by ``partition`` (a builder of
    :data:`CONTEXT_BUILDERS`): a ``layout`` of the routes, or ``"ragged:<tokens of rank 0>,<tokens of rank 1>,..."``,
    a contiguous split with those token counts, cut at document boundaries too."""
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if not layout.startswith("ragged:"):
        return partition(cu, world_size, layout), cu
    tokens = [int(n) for n in layout.split(":")[1].split(",")]
    assert len(tokens) == world_size and sum(tokens) == cu[-1]
    nested = [[] for _ in range(world_size)]
    pos = 0
    for rank, n in enumerate(tokens):
        end = pos + n
        d = 0
        while d < len(cu) - 1 and cu[d + 1] <= pos:
            d += 1
        p = pos
        while p < end and d < len(cu) - 1:
            e = min(end, cu[d + 1])
            if e > p:
                nested[rank].append((d, p - cu[d], e - p))
            p = e
            d += 1
        pos = end
    return partition(cu, world_size, fragments=nested), cu


def local_tokens(part, rank: int, cu) -> list[int]:
    """The pack's token of every local row of ``rank`` (its fragments back to back)."""
    return [cu[f.doc] + f.off + i for f in part.fragments[rank] for i in range(f.length)]


def token_index(frags, cu, device):
    """The pack's token of every local row of ``frags`` (int64)."""
    rows = [cu[d] + off + i for d, off, length in frags for i in range(length)]
    return torch.tensor(rows, dtype=torch.int64, device=device)


def _shift(v, doc, d):
    """``out[t] = v[t - d]`` (``d > 0``) or ``v[t + |d|]`` (``d < 0``) within token ``t``'s document, else 0."""
    out = torch.zeros_like(v)
    n = abs(d)
    if d == 0:
        return v.clone()
    if n >= v.shape[0]:
        return out
    if d > 0:
        out[n:] = torch.where((doc[n:] == doc[:-n]).unsqueeze(1), v[:-n], 0.0)
    else:
        out[:-n] = torch.where((doc[:-n] == doc[n:]).unsqueeze(1), v[n:], 0.0)
    return out


def reference(x, w, cu, dy):
    """FP64 reference of the whole pack under the context-parallel route's semantics (SiLU' at the unrounded
    preactivation): ``(y, dx, dpre, taps_inputs)`` with BF16 ``y`` / ``dx`` (one rounding each), the FP64 preactivation
    derivative ``dpre`` and the four FP64 inputs each tap multiplies (``taps_inputs[k][t] = x[t - 3 + k]`` of
    ``t``'s document)."""
    S = x.shape[0]
    cu = torch.as_tensor(cu, device=x.device)
    doc = torch.bucketize(torch.arange(S, device=x.device), cu[1:].long(), right=True)
    xd, wd = x.double(), w.double()
    taps_inputs = [_shift(xd, doc, HALO - k) for k in range(WIDTH)]
    pre = sum(wd[:, k] * taps_inputs[k] for k in range(WIDTH))
    s = torch.sigmoid(pre)
    y = (pre * s).bfloat16()
    dpre = dy.double() * (s * (1.0 + pre * (1.0 - s)))
    dx = sum(wd[:, k] * _shift(dpre, doc, -(HALO - k)) for k in range(WIDTH)).bfloat16()
    return y, dx, dpre, taps_inputs


def dweight_partial(dpre, taps_inputs, index):
    """FP64 ``[C, 4]`` weight gradient over the tokens ``index`` of the pack."""
    return torch.stack([(dpre[index] * taps_inputs[k][index]).sum(0) for k in range(WIDTH)], dim=1)


def outside(actual, expected, *, atol, rtol) -> int:
    """Elements of ``actual`` outside ``atol + rtol * |expected|`` (non-finite ones count)."""
    a, b = actual.detach().double().reshape(-1), expected.detach().double().reshape(-1)
    bad = ~torch.isfinite(a) | ((a - b).abs() > atol + rtol * b.abs())
    return int(bad.sum())


def bits_equal(a, b) -> bool:
    """Same dtype, shape and bits (-0.0 and 0.0 differ)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))
