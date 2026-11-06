"""How ``causal_conv1d_kcp`` / ``causal_conv1d_kcp_multi`` hand a call to the vendored context-parallel routes, on
the CPU: the entry each variant runs and its arguments, the configured variant read at the call (``det`` under
``deterministic_mode``), ``det`` for ``fast`` under torch's deterministic algorithms and the FP32 taps of the route.
Also the contexts the GPU tests pass (the stand-in one and those of the GDN / KDA context-parallel routes this
checkout vendors): a simulated exchange of their halo tables delivers every history row and every head-row
derivative to the fragment that reads it. Needs torch and tvm_ffi, not CUDA."""

import logging
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.conv1d import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config  # noqa: E402

from ._kcp import (  # noqa: E402
    CONTEXT_BUILDERS,
    HALO,
    WIDTH,
    context_builder,
    fragments,
    local_tokens,
    one_rank_context,
    split_pack,
)


class _Route:
    """A stand-in for a package's ``kcp`` module: records each call and returns ``x * w[:, 3]`` (in FP32), so
    autograd reaches the taps."""

    def __init__(self):
        self.calls = []

    def _single(self, name):
        def entry(x, w, ctx, **kwargs):
            self.calls.append((name, x, w, ctx, kwargs))
            return (x.float() * w[:, 3]).to(x.dtype)

        return entry

    def _multi(self, name):
        def entry(xs, ws, ctx, **kwargs):
            self.calls.append((name, xs, ws, ctx, kwargs))
            return tuple((x.float() * w[:, 3]).to(x.dtype) for x, w in zip(xs, ws, strict=True))

        return entry

    def module(self):
        return types.SimpleNamespace(
            cake_causal_conv1d_det_kcp=self._single("det"),
            cake_causal_conv1d_nondet_kcp=self._single("fast"),
            cake_causal_conv1d_det_multi_kcp=self._multi("det_multi"),
            cake_causal_conv1d_nondet_kcp_multi=self._multi("fast_multi"),
        )


@pytest.fixture
def route(monkeypatch):
    recorder = _Route()
    module = recorder.module()
    monkeypatch.setattr(ops, "_kcp_module", lambda variant: module)
    cake.configure(cake_conv1d_variant="fast")
    return recorder


def _operands(C=16, T=5, taps=torch.float32):
    # small integers: every sum below is exact
    x = torch.randint(-4, 5, (T, C)).bfloat16()
    w = torch.randint(-4, 5, (C, 4)).to(taps)
    return x, w


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_each_variant_runs_its_packages_route(route, variant):
    x, w = _operands()
    ctx, handle = object(), types.SimpleNamespace(entries={})
    cake.configure(cake_conv1d_variant=variant)

    y = ops.causal_conv1d_kcp(x, w, ctx, recompute_handle=handle, cooperative=False)
    assert y.shape == x.shape
    ((name, x_seen, w_seen, ctx_seen, kwargs),) = route.calls
    assert name == variant and ctx_seen is ctx
    assert torch.equal(x_seen, x) and torch.equal(w_seen, w)
    assert kwargs == {"recompute_handle": handle, "cooperative": False}  # no variant keyword: the module is the mode


def test_the_configured_variant_is_the_default(route):
    x, w = _operands()
    for configured in ("det", "fast"):
        cake.configure(cake_conv1d_variant=configured)
        ops.causal_conv1d_kcp(x, w, object())
    cake.configure()
    ops.causal_conv1d_kcp_multi([x], [w], object())  # an unconfigured process runs det
    assert [call[0] for call in route.calls] == ["det", "fast", "det_multi"]
    assert route.calls[0][4] == {"recompute_handle": None, "cooperative": None}


def test_deterministic_mode_runs_det(route):
    x, w = _operands()
    cake.configure(deterministic_mode=True)

    ops.causal_conv1d_kcp(x, w, object())
    ops.causal_conv1d_kcp_multi([x], [w], object())
    assert [call[0] for call in route.calls] == ["det", "det_multi"]


def test_fast_runs_det_under_torch_deterministic_algorithms(route, monkeypatch, caplog):
    monkeypatch.setattr(config, "_warned", set())
    x, w = _operands()
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            ops.causal_conv1d_kcp(x, w, object())
            ops.causal_conv1d_kcp_multi([x], [w], object())
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    assert [call[0] for call in route.calls] == ["det", "det_multi"]
    assert all("deterministic" not in call[4] for call in route.calls)
    warnings = [r for r in caplog.records if "conv1d" in r.getMessage() and "running det" in r.getMessage()]
    assert len(warnings) == 1


def test_bf16_taps_run_as_fp32_and_get_a_bf16_gradient(route):
    x, w = _operands(taps=torch.bfloat16)
    w.requires_grad_()

    y = ops.causal_conv1d_kcp(x, w, object())
    w_seen = route.calls[0][2]
    assert w_seen.dtype == torch.float32 and torch.equal(w_seen, w.float())
    y.float().sum().backward()
    assert w.grad.dtype == torch.bfloat16
    expected = torch.zeros(w.shape)
    expected[:, 3] = x.float().sum(0)
    assert torch.equal(w.grad, expected.bfloat16())


def test_fp32_taps_pass_unchanged_and_inputs_become_contiguous(route):
    x, w = _operands(C=16, T=6)
    wide = torch.zeros(6, 24, dtype=torch.bfloat16)
    wide[:, :16] = x
    strided = wide[:, :16]
    assert not strided.is_contiguous()

    ops.causal_conv1d_kcp(strided, w, object())
    x_seen, w_seen = route.calls[0][1], route.calls[0][2]
    assert x_seen.is_contiguous() and torch.equal(x_seen, x)
    assert w_seen is w


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_the_multi_call_runs_its_packages_route(route, variant):
    x1, w1 = _operands(C=16)
    x2, w2 = _operands(C=8, taps=torch.bfloat16)
    ctx = object()
    cake.configure(cake_conv1d_variant=variant)

    ys = ops.causal_conv1d_kcp_multi((x1, x2), (w1, w2), ctx)
    assert [y.shape for y in ys] == [x1.shape, x2.shape]
    ((name, xs, ws, ctx_seen, kwargs),) = route.calls
    assert name == f"{variant}_multi" and ctx_seen is ctx
    assert [w.dtype for w in ws] == [torch.float32, torch.float32]
    assert kwargs == {"recompute_handle": None, "cooperative": None}


def test_the_multi_call_takes_sequences(route):
    x, w = _operands()
    with pytest.raises(TypeError, match="sequences of tensors"):
        ops.causal_conv1d_kcp_multi(x, [w], object())
    assert not route.calls


def test_the_one_rank_context_routes_every_halo_row():
    # documents 5000 and 1003 cut into fragments of 1, 2, 4997 and 1003 tokens: the third fragment reads its history
    # from the first two; its first token is a head row of both earlier fragments
    S, cu, frags = fragments((5000, 1003), (1, 2, 4997, 1003))
    ctx = one_rank_context(frags, "cpu")
    assert (ctx.T_r, ctx.n_frag, ctx.cu_seqlens_local_cpu) == (S, 4, (0, 1, 3, 5000, 6003))
    tables = ctx.halo_tables
    assert tables["hist_row_off"].tolist() == [[-1] * HALO, [-1, -1, 0], [1, 2, 3], [-1] * HALO]
    assert tables["hist_index"].tolist() == [0, 0, 1, 2]
    assert tables["tail_row_off"].tolist() == [[0, 1, 2], [3, 4, 5], [-1] * HALO, [-1] * HALO]
    assert tables["head_row_off"][1].tolist() == [[0, -1, -1], [1, -1, -1], [-1, -1, -1]]
    assert tables["head_row_off"][2].tolist() == [[2, 3, -1], [4, -1, -1], [5, -1, -1]]
    assert (ctx.halo.n_hist_rows, ctx.halo.n_head_rows, ctx.halo.n_tail_rows) == (4, 6, 6)


# (document lengths, world size, layout): the layouts of the routes and ragged tables with one-token fragments, a
# fragment reading its history from three predecessor ranks, and ranks without tokens
EXCHANGE_CASES = (
    ((2048, 5, 3000, 1), 2, "contiguous"),
    ((3000,), 2, "zigzag"),
    ((2048, 5, 3000, 1), 4, "zigzag_per_document"),
    ((5000, 1003), 2, "ragged:2,6001"),
    ((5000, 1003), 4, "ragged:1,2,4997,1003"),
    ((1200, 900, 900), 4, "ragged:1200,0,900,900"),
    ((6, 1, 2, 3, 4000), 8, "ragged:1,1,1,3,0,2,4000,4"),
)


def _blocks(rows, splits, world_size):
    assert len(splits) == world_size and sum(splits) == len(rows)
    out, pos = [], 0
    for n in splits:
        out.append(rows[pos : pos + n])
        pos += n
    return out


@pytest.mark.parametrize("lengths, world_size, layout", EXCHANGE_CASES)
@pytest.mark.parametrize("builder", sorted(CONTEXT_BUILDERS))
def test_every_context_routes_its_halo_rows(builder, lengths, world_size, layout):
    found = context_builder(builder)
    if found is None:
        pytest.skip(f"this checkout does not vendor the {builder} context-parallel package")
    partition, build_context = found
    part, cu = split_pack(partition, lengths, world_size, layout)
    ctxs = [build_context(part, None, rank, "cpu", conv_kernel_size=WIDTH) for rank in range(world_size)]
    tokens = [local_tokens(part, rank, cu) for rank in range(world_size)]
    doc_end = {d: cu[d + 1] - cu[d] for d in range(len(lengths))}

    # forward: every rank sends the tokens its hist_index names, one block per destination rank
    sent = []
    for rank, ctx in enumerate(ctxs):
        assert int(ctx.T_r) == len(tokens[rank]) and int(ctx.conv_kernel_size) == WIDTH
        rows = [tokens[rank][i] for i in ctx.halo_tables["hist_index"].tolist()]
        sent.append(_blocks(rows, list(ctx.halo.fwd_send_rows), world_size))
    for rank, ctx in enumerate(ctxs):
        assert [len(sent[source][rank]) for source in range(world_size)] == list(ctx.halo.fwd_recv_rows)
        received = [token for source in range(world_size) for token in sent[source][rank]]
        assert len(received) == int(ctx.halo.n_hist_rows)
        hist_row_off = ctx.halo_tables["hist_row_off"].tolist()
        for slot, f in enumerate(part.fragments[rank]):
            if not f.length:
                continue
            for k in range(HALO):
                token = f.off - HALO + k
                row = hist_row_off[slot][k]
                if f.off == 0 or token < 0:
                    assert row == -1, (rank, slot, k)
                else:
                    assert row >= 0 and received[row] == cu[f.doc] + token, (rank, slot, k)

    # backward: every head row carries the derivative of its fragment's token, one block per destination rank; each
    # tail row of a fragment receives the token right after it (when its document has one)
    sent = []
    for rank, ctx in enumerate(ctxs):
        rows = [None] * int(ctx.halo.n_head_rows)
        head_row_off = ctx.halo_tables["head_row_off"].tolist()
        for slot, f in enumerate(part.fragments[rank]):
            for position in range(HALO):
                for row in head_row_off[slot][position]:
                    if row >= 0:
                        assert rows[row] is None and position < f.length, (rank, slot, position)
                        rows[row] = cu[f.doc] + f.off + position
        assert None not in rows
        sent.append(_blocks(rows, list(ctx.halo.bwd_send_rows), world_size))
    for rank, ctx in enumerate(ctxs):
        assert [len(sent[source][rank]) for source in range(world_size)] == list(ctx.halo.bwd_recv_rows)
        received = [token for source in range(world_size) for token in sent[source][rank]]
        assert len(received) == int(ctx.halo.n_tail_rows)
        tail_row_off = ctx.halo_tables["tail_row_off"].tolist()
        for slot, f in enumerate(part.fragments[rank]):
            if not f.length:
                continue
            for m in range(HALO):
                token = f.off + f.length + m
                row = tail_row_off[slot][m]
                if token >= doc_end[f.doc]:
                    assert row == -1, (rank, slot, m)
                else:
                    assert row >= 0 and received[row] == cu[f.doc] + token, (rank, slot, m)
