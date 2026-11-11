"""Wiring of the vendored KCP packages into ``chunk_gated_delta_rule`` (CPU: needs torch and tvm_ffi, no GPU).

``build_kcp_context``: both variants see the same local tokens, a contiguous partition is FLA's, and one context
serves every layer of a batch layout. A ``KcpContext`` call reaches the KCP package of the configured variant with
contiguous FP32 g and beta (BF16 beta cast exactly), dense q / k / v rows, ``save_mode`` from
``--cake-gdn-recompute-state`` (``"full"``, else the package's light set: ``"light"``, or ``None`` for ``fast``, whose
default is ``"light_akk"`` with the in-kernel L2 norm) and the default scale. The context-parallel contract is FLA's:
an initial state or a final state is rejected when the call is made, and the context's boundaries replace
``cu_seqlens`` / ``cu_seqlens_cpu``. A call the package does not serve runs the det kernels of FLA's context
parallelism on the context's FLA context, logged once per reason; a call no kernel set serves raises. On the packages'
torch emulation (``backend="fake"``, which runs on the CPU) the call gives the bits of the package's own entry point
and its gradients reach every input.

The non-deterministic package recognises the deterministic package's packed q / k / v split by its class name and
the last two parts of its module (``cake_gdn_det_train_kcp.cake_kcp``), so under ``_vendor/`` too; that only decides
whether the backward writes one packed row, never a value (``test_a_det_split_feeding_fast_changes_no_gradient``), and
no entry point of this repository makes such views.
"""

import itertools
import logging
import math
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.gdn import ops  # noqa: E402
from miles_ext_kernels.cake._impl.gdn.kcp import KcpContext  # noqa: E402
from miles_ext_kernels.cake._runtime import config  # noqa: E402
from miles_ext_kernels.cake.gdn import build_kcp_context, kcp_chunk_gated_delta_rule  # noqa: E402

CPU = torch.device("cpu")
HEAD_DIM = 128
# four documents, 512 tokens: every split into 2 or 4 contiguous ranks cuts a document
LAYOUT = (130, 190, 140, 52)
HEADS = (2, 4)  # key heads, value heads
ENTRIES = {"det": "cake_gdn_det_kcp", "fast": "cake_gdn_nondet_kcp"}
# the save mode of a call under --cake-gdn-recompute-state: the package's light set (None: the fast package's default)
LIGHT = {"det": "light", "fast": None}


def _cu(seq_lens) -> tuple[int, ...]:
    return tuple(itertools.accumulate(seq_lens, initial=0))


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    # det, the tape kept; the conftest restores the default configuration afterwards
    cake.configure(cake_gdn_variant="det")
    monkeypatch.setattr(ops, "KCP_OPTIONS", None)
    monkeypatch.setattr(ops, "_KCP_LOGGED", set())
    monkeypatch.setattr(config, "_warned", set())


def _context(world_size=1, rank=0, layout="contiguous", seq_lens=LAYOUT) -> KcpContext:
    return KcpContext(_cu(seq_lens), None, rank, world_size, CPU, 4, layout, None)


def _inputs(num_tokens, heads=HEADS, *, seed=0, beta_dtype=torch.bfloat16) -> dict:
    num_heads, num_v_heads = heads
    gen = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn(1, num_tokens, num_heads, HEAD_DIM, generator=gen).bfloat16(),
        "k": torch.randn(1, num_tokens, num_heads, HEAD_DIM, generator=gen).bfloat16(),
        "v": torch.randn(1, num_tokens, num_v_heads, HEAD_DIM, generator=gen).bfloat16(),
        "g": torch.rand(1, num_tokens, num_v_heads, generator=gen) * -0.5,
        "beta": torch.rand(1, num_tokens, num_v_heads, generator=gen).to(beta_dtype),
        "do": torch.randn(1, num_tokens, num_v_heads, HEAD_DIM, generator=gen).bfloat16(),
    }


def _record(monkeypatch, variant) -> list:
    """Replace ``variant``'s KCP entry point with a recorder that returns zeros like ``v``."""
    package = ops._package(f"{variant}_kcp")
    calls = []

    def record(q, k, v, g, beta, ctx, **kwargs):
        calls.append({"q": q, "k": k, "v": v, "g": g, "beta": beta, "ctx": ctx, **kwargs})
        return torch.zeros_like(v)

    monkeypatch.setattr(package, ENTRIES[variant], record)
    return calls


def _call(inputs, ctx, **kwargs):
    return ops.chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        cu_seqlens=ctx.cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        cp_context=ctx,
        **kwargs,
    )


# --------------------------------------------------------------------------------------------------------------------
# the context


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_a_contiguous_partition_is_fla_s_and_both_variants_see_the_same_tokens(world_size):
    cu = _cu(LAYOUT)
    part = cu[-1] // world_size
    for rank in range(world_size):
        ctx = _context(world_size, rank)
        # FLA's get_cp_cu_seqlens: rank r holds tokens [r * part, (r + 1) * part), cut at the document boundaries
        start, end = rank * part, (rank + 1) * part
        fla = tuple(sorted({min(max(b, start), end) - start for b in cu}))

        assert ctx.cu_seqlens_cpu == fla
        assert ctx.num_tokens == part
        assert ctx.cu_seqlens.tolist() == list(fla) and ctx.cu_seqlens.dtype == torch.int32
        assert ctx.fla_compatible
        assert tuple(ctx.partition("fast").local_cu_seqlens(rank)) == ctx.cu_seqlens_cpu
        assert ctx.context("det").T_r == ctx.context("fast").T_r == part


def test_the_load_balanced_layouts_are_not_fla_s():
    for layout in ("zigzag", "zigzag_per_document"):
        ctx = _context(2, 1, layout)
        assert not ctx.fla_compatible
        with pytest.raises(NotImplementedError, match="contiguous partition"):
            ctx.fla_context()
    # nor a contiguous one that leaves a remainder
    assert not _context(2, 0, seq_lens=(131, 190)).fla_compatible


def test_one_context_per_batch_layout():
    cu = _cu(LAYOUT)
    first = build_kcp_context(cu, device="cpu")

    assert build_kcp_context(list(cu), device="cpu") is first
    assert build_kcp_context(torch.tensor(cu), device="cpu") is first
    assert build_kcp_context(cu, device="cpu", layout="zigzag") is not first
    assert (first.rank, first.world_size, first.group) == (0, 1, None)
    assert first.fla_context().group is None and first.fla_context().cu_seqlens_cpu == first.cu_seqlens_cpu
    explicit = build_kcp_context(cu, device="cpu", fragments=[[(d, 0, n) for d, n in enumerate(LAYOUT)]])
    assert explicit.layout == "explicit" and explicit.cu_seqlens_cpu == cu
    with pytest.raises(ValueError, match="host"):
        build_kcp_context(torch.tensor(cu, device="meta"), device="cpu")
    with pytest.raises(ValueError, match="layout"):
        build_kcp_context(cu, device="cpu", layout="ring")


# --------------------------------------------------------------------------------------------------------------------
# routing


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("save_state", [True, False])
def test_a_kcp_context_call_reaches_the_kcp_package_of_the_variant(monkeypatch, variant, save_state):
    calls = _record(monkeypatch, variant)
    other = _record(monkeypatch, "fast" if variant == "det" else "det")
    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=not save_state)
    ctx = _context()
    inputs = _inputs(ctx.num_tokens)

    out, final_state = _call(inputs, ctx)

    assert final_state is None and out.shape == inputs["v"].shape
    assert len(calls) == 1 and other == []
    call = calls[0]
    assert call["ctx"] is ctx.context(variant)
    assert call["save_mode"] == ("full" if save_state else LIGHT[variant])
    # the default scale of every kernel set, 1 / sqrt(K): the package's own default in FP32
    assert call["scale"] == 1.0 / math.sqrt(HEAD_DIM)
    package = ops._package(f"{variant}_kcp")
    package_scale = torch.tensor(package.cake_kcp.SCALE, dtype=torch.float32)
    assert torch.tensor(call["scale"], dtype=torch.float32) == package_scale
    assert call["use_qk_l2norm_in_kernel"] is True
    assert call["options"] is None
    assert "deterministic" not in call
    # g and beta in FP32, contiguous; BF16 beta cast exactly
    assert call["g"].dtype == call["beta"].dtype == torch.float32
    assert call["g"].is_contiguous() and call["beta"].is_contiguous()
    assert torch.equal(call["beta"], inputs["beta"].float())
    assert all(call[name] is inputs[name] for name in ("q", "k", "v"))


def test_kcp_options_reach_the_package(monkeypatch):
    calls = _record(monkeypatch, "det")
    monkeypatch.setattr(ops, "KCP_OPTIONS", {"segments": 2})
    ctx = _context()

    kcp_chunk_gated_delta_rule(**{n: t for n, t in _inputs(ctx.num_tokens).items() if n != "do"}, kcp_context=ctx)

    assert calls[0]["options"] == {"segments": 2}


def test_q_k_v_reach_the_package_with_dense_head_rows(monkeypatch):
    calls = _record(monkeypatch, "det")
    ctx = _context()
    num_heads, num_v_heads = HEADS
    tokens = ctx.num_tokens
    inputs = _inputs(tokens)
    # miles' split of the short convolution's output: per key head its q, its k and its value heads
    mixed = torch.randn(1, tokens, num_heads * 2 * HEAD_DIM + num_v_heads * HEAD_DIM).bfloat16()
    q, k, v = mixed.view(1, tokens, num_heads, -1).split(
        [HEAD_DIM, HEAD_DIM, num_v_heads // num_heads * HEAD_DIM], dim=-1
    )
    v = v.reshape(1, tokens, num_v_heads, HEAD_DIM)
    _call({**inputs, "q": q, "k": k, "v": v}, ctx)
    # the per-key-head views have no dense [heads, 128] rows: copied
    assert calls[-1]["q"].is_contiguous() and torch.equal(calls[-1]["q"], q)
    assert calls[-1]["k"].is_contiguous() and torch.equal(calls[-1]["k"], k)
    # head views of one packed [T, (2 H + HV) * 128] row read in place
    packed = torch.randn(1, tokens, (2 * num_heads + num_v_heads) * HEAD_DIM).bfloat16()
    views = packed.split([num_heads * HEAD_DIM, num_heads * HEAD_DIM, num_v_heads * HEAD_DIM], dim=-1)
    q, k, v = (x.view(1, tokens, -1, HEAD_DIM) for x in views)
    _call({**inputs, "q": q, "k": k, "v": v}, ctx)
    assert [calls[-1][n].data_ptr() for n in "qkv"] == [x.data_ptr() for x in (q, k, v)]


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_the_context_parallel_contract_is_fla_s(monkeypatch, variant):
    # FLA's chunk_gated_delta_rule under a cp_context: no initial state, no final state (both rejected when the call
    # is made, before any kernel), and the context's own boundaries in place of cu_seqlens / cu_seqlens_cpu
    calls = _record(monkeypatch, variant)
    cake.configure(cake_gdn_variant=variant)
    ctx = _context(2, 0)
    inputs = _inputs(ctx.num_tokens)

    with pytest.raises(ValueError, match="initial_state is not supported under context parallelism"):
        _call(inputs, ctx, initial_state=torch.zeros(len(ctx.cu_seqlens_cpu) - 1, HEADS[1], HEAD_DIM, HEAD_DIM))
    with pytest.raises(ValueError, match="output_final_state is not supported under context parallelism"):
        _call(inputs, ctx, output_final_state=True)
    assert calls == []
    # the global boundaries of the batch (host and device) are ignored, as FLA ignores them
    global_cu = _cu(LAYOUT)
    out, final_state = _call(inputs, ctx, cu_seqlens_cpu=global_cu)
    assert final_state is None and out.shape == inputs["v"].shape
    ops.chunk_gated_delta_rule(
        *(inputs[n] for n in ("q", "k", "v", "g", "beta")),
        cu_seqlens=torch.tensor(global_cu, dtype=torch.int32),
        cu_seqlens_cpu=global_cu,
        use_qk_l2norm_in_kernel=True,
        cp_context=ctx,
    )
    assert len(calls) == 2 and all(call["ctx"] is ctx.context(variant) for call in calls)


def test_what_no_kernel_set_serves_raises(monkeypatch):
    calls = _record(monkeypatch, "det")
    ctx = _context(2, 0)
    inputs = _inputs(ctx.num_tokens)

    with pytest.raises(ValueError, match="tokens"):
        _call(_inputs(ctx.num_tokens + 1), ctx)
    with pytest.raises(ValueError, match="batch"):
        _call({n: t.expand(2, *t.shape[1:]) for n, t in inputs.items()}, ctx)
    with pytest.raises(ValueError, match="multiple"):
        _call(_inputs(ctx.num_tokens, (3, 4)), ctx)
    with pytest.raises(TypeError, match="KcpContext"):
        kcp_chunk_gated_delta_rule(*(inputs[n] for n in ("q", "k", "v", "g", "beta")), types.SimpleNamespace())
    assert calls == []
    # a host copy of the context's own local layout is accepted
    _call(inputs, ctx, cu_seqlens_cpu=ctx.cu_seqlens_cpu)
    assert len(calls) == 1


L2N_REASON = "the in-kernel q / k L2 norm (no records in this package build)"


def _without_l2n_records(monkeypatch, variant):
    """``variant``'s KCP package as a build without the in-kernel L2 norm records."""
    monkeypatch.setattr(ops._package(f"{variant}_kcp").cake_kcp, "l2n_records_shipped", lambda: False)


@pytest.mark.parametrize("save_state", [True, False])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_call_the_package_does_not_serve_runs_the_fla_cp_det_kernels_and_says_so_once(
    monkeypatch, caplog, variant, save_state
):
    kcp_calls = _record(monkeypatch, variant)
    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=not save_state)
    fla_cp = []

    def cp_apply(q, k, v, g, beta, scale, h0, final, cu_seqlens, cu_seqlens_cpu, l2n, cp, save):
        fla_cp.append((h0, final, cu_seqlens, cu_seqlens_cpu, l2n, cp, save))
        return torch.zeros_like(v), None

    monkeypatch.setattr(ops, "_cp_apply", cp_apply)
    ctx = _context()
    inputs = _inputs(ctx.num_tokens)

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        # a package build without the in-kernel L2 norm records
        with monkeypatch.context() as patch:
            _without_l2n_records(patch, variant)
            for _ in range(2):
                _out, final_state = _call(inputs, ctx)
                assert final_state is None
        # a partition the package refuses at context build
        monkeypatch.setattr(KcpContext, "context", lambda self, v: (_ for _ in ()).throw(ValueError("over the caps")))
        _call(inputs, ctx)
        with pytest.raises(NotImplementedError, match="over the caps"):
            kcp_chunk_gated_delta_rule(*(inputs[n] for n in ("q", "k", "v", "g", "beta")), ctx)

    assert kcp_calls == []
    assert [entry[:2] for entry in fla_cp] == [(None, False)] * 3
    assert all(entry[2] is ctx.cu_seqlens and entry[3] == ctx.cu_seqlens_cpu for entry in fla_cp)
    assert all(entry[4] is True and entry[5] is ctx.fla_context() for entry in fla_cp)
    assert all(entry[6] is save_state for entry in fla_cp)
    messages = [r.getMessage() for r in caplog.records if r.name == "miles_ext_kernels"]
    reasons = (L2N_REASON, "this partition (over the caps)")
    if variant == "det":
        expected = [
            f"miles_ext_kernels.cake.gdn: the det KCP package does not serve {reason}; running the det kernels of "
            "FLA's context parallelism"
            for reason in reasons
        ]
    else:
        expected = [
            f"miles_ext_kernels.cake.gdn: the fast variant does not serve {reason} with a KcpContext; running det"
            for reason in reasons
        ]
    assert messages == expected


def test_without_an_fla_partition_a_call_the_package_does_not_serve_raises(monkeypatch):
    _record(monkeypatch, "det")
    _without_l2n_records(monkeypatch, "det")
    ctx = _context(2, 0, "zigzag_per_document")

    with pytest.raises(NotImplementedError, match="L2 norm.*contiguous partition"):
        _call(_inputs(ctx.num_tokens), ctx)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_fla_context_runs_the_fla_cp_det_kernels_with_the_context_s_boundaries(monkeypatch, caplog, variant):
    # an FLA cp_context (here without a process group): cake_gdn_fla_cp_det under either variant, the global
    # boundaries of the arguments replaced by the context's; fast logs once that it runs det
    seen = []
    monkeypatch.setattr(ops, "_cp_apply", lambda *args: seen.append(args) or (torch.zeros_like(args[2]), None))
    cake.configure(cake_gdn_variant=variant)
    local = _cu((100, 28))
    fla = types.SimpleNamespace(group=None, cu_seqlens=torch.tensor(local, dtype=torch.int32), cu_seqlens_cpu=local)
    inputs = _inputs(local[-1])
    global_cu = _cu(LAYOUT)

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        for _ in range(2):
            out, final_state = ops.chunk_gated_delta_rule(
                *(inputs[n] for n in ("q", "k", "v", "g", "beta")),
                cu_seqlens=torch.tensor(global_cu, dtype=torch.int32),
                cu_seqlens_cpu=global_cu,
                use_qk_l2norm_in_kernel=True,
                cp_context=fla,
            )
        with pytest.raises(ValueError, match="output_final_state is not supported under context parallelism"):
            ops.chunk_gated_delta_rule(
                *(inputs[n] for n in ("q", "k", "v", "g", "beta")), output_final_state=True, cp_context=fla
            )
        with pytest.raises(ValueError, match="initial_state is not supported under context parallelism"):
            ops.chunk_gated_delta_rule(
                *(inputs[n] for n in ("q", "k", "v", "g", "beta")),
                initial_state=torch.zeros(2, HEADS[1], HEAD_DIM, HEAD_DIM),
                cp_context=fla,
            )

    assert final_state is None and len(seen) == 2
    assert all(args[8] is fla.cu_seqlens and args[9] == local and args[11] is fla for args in seen)
    messages = [r.getMessage() for r in caplog.records if r.name == "miles_ext_kernels"]
    if variant == "fast":
        assert messages == [
            "miles_ext_kernels.cake.gdn: the fast variant does not serve an FLA cp_context; running det"
        ]
    else:
        assert messages == []


def test_an_fla_context_without_boundaries_is_rejected():
    inputs = _inputs(128)

    with pytest.raises(ValueError, match="the cp_context's cu_seqlens"):
        ops.chunk_gated_delta_rule(
            *(inputs[n] for n in ("q", "k", "v", "g", "beta")), cp_context=types.SimpleNamespace(group=None)
        )


def test_cp_chunk_gated_delta_rule_takes_a_kcp_context_as_its_fla_context(monkeypatch):
    seen = []
    monkeypatch.setattr(ops, "_cp_apply", lambda *args: seen.append(args) or (None, None))
    ctx = _context()
    inputs = _inputs(ctx.num_tokens)

    ops.cp_chunk_gated_delta_rule(*(inputs[n] for n in ("q", "k", "v", "g", "beta")), cp_context=ctx)

    assert len(seen[0]) == 13
    assert seen[0][8] is ctx.cu_seqlens and seen[0][9] == ctx.cu_seqlens_cpu and seen[0][11] is ctx.fla_context()
    assert seen[0][12] is True  # --cake-gdn-recompute-state unset: the tape is kept


# --------------------------------------------------------------------------------------------------------------------
# the packages' torch emulation (CPU)


def _leaves(inputs) -> dict:
    return {n: t.detach().clone().requires_grad_(True) for n, t in inputs.items() if n != "do"}


def _grads(leaves, out) -> dict:
    return {"o": out.detach(), **{f"d{n}": t.grad for n, t in leaves.items()}}


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("save_state", [True, False])
def test_on_the_emulation_a_call_gives_the_bits_of_the_package_entry_point(monkeypatch, variant, save_state):
    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=not save_state)
    monkeypatch.setattr(ops, "KCP_OPTIONS", {"backend": "fake"})
    ctx = _context()
    inputs = _inputs(ctx.num_tokens, seed=3)

    leaves = _leaves(inputs)
    out, _ = ops.chunk_gated_delta_rule(
        **{n: leaves[n] for n in ("q", "k", "v", "g", "beta")}, use_qk_l2norm_in_kernel=True, cp_context=ctx
    )
    out.backward(inputs["do"])
    got = _grads(leaves, out)

    direct = _leaves(inputs)
    entry = getattr(ops._package(f"{variant}_kcp"), ENTRIES[variant])
    beta32 = direct["beta"].float()
    ref_out = entry(
        *(direct[n] for n in ("q", "k", "v", "g")),
        beta32,
        ctx.context(variant),
        scale=HEAD_DIM**-0.5,
        save_mode="full" if save_state else LIGHT[variant],
        use_qk_l2norm_in_kernel=True,
        options={"backend": "fake"},
    )
    ref_out.backward(inputs["do"])
    want = _grads(direct, ref_out)

    assert got["dbeta"].dtype == torch.bfloat16
    for name in want:
        assert torch.isfinite(got[name].float()).all(), name
        assert torch.equal(got[name], want[name]), name
    assert all(float(got[name].float().abs().max()) > 0 for name in want), "a gradient did not reach its input"


def test_a_det_split_feeding_fast_changes_no_gradient(monkeypatch):
    """The fast package recognises the det package's split under ``_vendor/`` by the last two parts of its module, so
    both splits hand the packed gradient row on without a copy, and the gradient of the packed row is the same."""
    nd = ops._package("fast_kcp").cake_kcp
    det = ops._package("det_kcp").cake_kcp
    det_split = det.CakeGDNDetKcpPackedQKVSplitFunction
    assert det_split.__module__ == "miles_ext_kernels.cake._vendor.cake_gdn_det_train_kcp.cake_kcp"
    assert det_split.__module__ != nd._DET_PACKED_SPLIT[0] and nd._is_det_packed_split(det_split)
    cake.configure(cake_gdn_variant="fast")
    monkeypatch.setattr(ops, "KCP_OPTIONS", {"backend": "fake"})
    ctx = _context()
    num_heads, num_v_heads = HEADS
    inputs = _inputs(ctx.num_tokens, seed=5)
    row = torch.cat([inputs[n].flatten(-2) for n in ("q", "k", "v")], dim=-1)

    grads = {}
    views_of = {
        "fast": nd.cake_gdn_nondet_train_kcp_packed_qkv_views,
        "det": det.cake_gdn_det_train_kcp_packed_qkv_views,
    }
    for split, views in views_of.items():
        stats = (nd if split == "fast" else det).PACKED_SPLIT_STATS
        before = dict(stats)
        qkv = row.detach().clone().requires_grad_(True)
        q, k, v = views(qkv, H_qk=num_heads, HV=num_v_heads)
        out, _ = ops.chunk_gated_delta_rule(
            q, k, v, g=inputs["g"], beta=inputs["beta"], use_qk_l2norm_in_kernel=True, cp_context=ctx
        )
        out.backward(inputs["do"])
        grads[split] = qkv.grad
        moved = {key: stats[key] - before[key] for key in stats}
        assert moved == {"aliased": 1, "concatenated": 0}, split

    assert torch.equal(grads["det"], grads["fast"])
