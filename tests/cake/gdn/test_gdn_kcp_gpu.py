"""The GDN variants with a ``KcpContext``: the vendored KCP packages (``det``: ``cake_gdn_det_train_kcp``; ``fast``:
``cake_gdn_nondet_train_kcp``) at the shapes miles feeds them.

One GPU (a one-rank context): against the FP64 token recurrence, output and every input gradient within the accuracy
budgets of the package suite, on miles' strided q / k views and BF16 beta, with the tapes kept (``save_mode="full"``)
or replayed (``--cake-gdn-recompute-state``: ``"light"``), and at P = 2 segments per fragment with
``save_transfer=False`` (the case flashinfer-ai/flashinfer#8585 fixed); ``det`` bit for bit run to run, also under
PyTorch's deterministic algorithms, with the same bits kept or replayed, and BF16 beta with the bits of its exact FP32
cast (``fast`` within the budgets: it makes no promise about its bits); within the budgets of the single-rank
packages; an initial or a final state is rejected when the call is made, as FLA rejects it under context parallelism;
a call the package does not serve runs the det kernels of FLA's context parallelism with their bits and says so once.

Two and four GPUs (one process per GPU, NCCL; marked ``multi_gpu``): context parallelism 2 and 4 on contiguous (FLA's)
and per-document zigzag partitions whose cuts fall inside documents, against the FP64 recurrence of the whole batch and
the one-GPU call of the same variant within the budgets; ``det`` bit for bit run to run (also under deterministic
algorithms) and with the same bits kept or replayed; P = 2 with ``save_transfer=False``; a final state rejected on
every rank before any communication; and a call the package does not serve with the bits of the FLA context's own
call (needs flash-linear-attention). A case that raises on every rank before any communication fails its own tests
with the ranks' tracebacks, and the other cases of the shared multi-GPU run still report. A case that raises on some
ranks only is not isolated: the collectives of the ranks no longer match, and the process group's 5-minute timeout
bounds the resulting hang. The context-parallel cases skip when fewer GPUs are visible than they need, and only then.

Needs an SM100a / SM103a / SM107a GPU and the vendored packages; skips elsewhere.
"""

import contextlib
import itertools
import logging
import math
import traceback
import types
from datetime import timedelta
from functools import cache
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.gdn import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402
from miles_ext_kernels.cake.gdn import build_kcp_context  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the GDN kernels need SM100a / SM103a / SM107a", allow_module_level=True)

HEAD_DIM = 128
# (key heads, value heads) of one tensor-parallel rank of miles' GDN layers
HEADS = {"4x12": (4, 12), "16x32": (16, 32), "4x4": (4, 4)}
# short enough for the FP64 token recurrence
REF_LAYOUT = (63, 130, 257, 450)
# every document at least four chunks long: P = 2 segments per fragment exist
P2_LAYOUT = (257, 450, 320, 197)
# a TP4 micro-batch: three documents and the 512-token alignment pad as its own segment
MILES_LAYOUT = (3339, 2802, 1340, 263)
# 2048 tokens: CP 2 cuts document 2, CP 4 cuts documents 1, 2 and 3; every contiguous fragment is at least 188 tokens
CP_LAYOUT = (300, 400, 600, 748)
CP_HEADS = (4, 12)
NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")
# The package suite's accuracy budgets, per output tensor: the relative RMS error at most REL_RMS_MAX and the largest
# absolute error at most MAX_ABS_REL times the largest reference magnitude.
REL_RMS_MAX = 1e-2
MAX_ABS_REL = 2e-2
L2NORM_EPS = 1e-6  # the in-kernel q / k normalization: x / sqrt(sum x^2 + 1e-6)
# #8585's case: P > 1 segments per fragment and the full saved set without the received transfers
P2_OPTIONS = {"segments": 2, "save_transfer": False}
VARIANTS = pytest.mark.parametrize("variant", ["det", "fast"])
PAIRS = pytest.mark.parametrize("pair", list(HEADS))


@pytest.fixture(autouse=True)
def _kernel_settings(monkeypatch):
    # det, the tapes kept; the conftest restores the default configuration afterwards
    cake.configure(cake_gdn_variant="det")
    monkeypatch.setattr(ops, "KCP_OPTIONS", None)


def _cu(seq_lens) -> tuple[int, ...]:
    return tuple(itertools.accumulate(seq_lens, initial=0))


def _inputs(seq_lens, heads, *, seed: int, device="cuda") -> dict:
    """Packed ``[1, T, ...]`` operands drawn on the host: the short conv's output (``mixed``, rows group-major: per key
    head its q, its k, then its value heads), g in FP32 and beta in BF16, as miles' GDN layer computes them."""
    num_heads, num_v_heads = heads
    total = sum(seq_lens)
    gen = torch.Generator().manual_seed(seed)
    mixed = torch.randn(1, total, num_heads * 2 * HEAD_DIM + num_v_heads * HEAD_DIM, generator=gen)
    g = torch.rand(1, total, num_v_heads, generator=gen) * -0.5
    beta = torch.rand(1, total, num_v_heads, generator=gen)
    do = torch.randn(1, total, num_v_heads, HEAD_DIM, generator=gen)
    return {
        "heads": heads,
        "mixed": mixed.to(device=device, dtype=torch.bfloat16),
        "g": g.to(device),
        "beta": beta.to(device=device, dtype=torch.bfloat16),
        "do": do.to(device=device, dtype=torch.bfloat16),
    }


def _split(mixed, heads):
    """miles' ``LinearAttention.convolve`` split: q / k ``[1, T, H, 128]`` strided views of ``mixed``, v ``[1, T, HV,
    128]``."""
    num_heads, num_v_heads = heads
    batch, seq_len, _ = mixed.shape
    q, k, v = mixed.view(batch, seq_len, num_heads, -1).split(
        [HEAD_DIM, HEAD_DIM, num_v_heads // num_heads * HEAD_DIM], dim=-1
    )
    return q, k, v.reshape(batch, seq_len, num_v_heads, HEAD_DIM)


def _leaves(inputs, rows=None, *, beta_fp32=False) -> dict:
    """q / k / v (contiguous), g and beta as leaves, of every token or of the token indices ``rows``."""
    q, k, v = _split(inputs["mixed"], inputs["heads"])
    tensors = {"q": q, "k": k, "v": v, "g": inputs["g"], "beta": inputs["beta"]}
    if rows is not None:
        tensors = {n: t[:, rows] for n, t in tensors.items()}
    if beta_fp32:
        tensors["beta"] = tensors["beta"].float()
    return {
        n: t.detach().clone(memory_format=torch.contiguous_format).requires_grad_(True) for n, t in tensors.items()
    }


def _run(inputs, ctx, *, views=True, beta_fp32=False, output_final_state=False, kernel=None, cp_context=None) -> dict:
    """One forward and backward of every token on a one-rank ``KcpContext`` (``cp_context`` overrides it; ``False``:
    no ``cp_context``, the host layout as ``cu_seqlens_cpu``); the output and the gradients of q, k, v, g and beta.
    ``views``: q / k / v are miles' views of one ``mixed`` leaf."""
    heads = inputs["heads"]
    if views:
        mixed = inputs["mixed"].detach().clone().requires_grad_(True)
        q, k, v = _split(mixed, heads)
        g = inputs["g"].detach().clone().requires_grad_(True)
        beta = inputs["beta"].detach().clone()
        beta = (beta.float() if beta_fp32 else beta).requires_grad_(True)
    else:
        leaves = _leaves(inputs, beta_fp32=beta_fp32)
        q, k, v, g, beta = (leaves[n] for n in ("q", "k", "v", "g", "beta"))
    kernel = ops.chunk_gated_delta_rule if kernel is None else kernel
    if cp_context is False:
        context = {"cu_seqlens_cpu": ctx.cu_seqlens_cpu}
    else:
        context = {"cp_context": ctx if cp_context is None else cp_context}
    o, final_state = kernel(
        q,
        k,
        v,
        g=g,
        beta=beta,
        cu_seqlens=ctx.cu_seqlens,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        **context,
    )
    o.backward(inputs["do"])
    torch.cuda.synchronize()
    dq, dk, dv = _split(mixed.grad, heads) if views else (q.grad, k.grad, v.grad)
    out = {"o": o.detach(), "dq": dq, "dk": dk, "dv": dv, "dg": g.grad, "dbeta": beta.grad}
    if output_final_state:
        out["final_state"] = final_state.detach()
    return out


def _reference(inputs, seq_lens) -> dict:
    """The FP64 token recurrence of the gated delta rule and its autograd gradients, per sequence and value head ``hv``
    (key head ``hv // (HV / H)``) with the state ``S`` ``[K, V]`` from zero::

        S = S * exp(g_t);  u_t = beta_t (v_t - S^T k_t);  S = S + k_t u_t^T;  o_t = scale S^T q_t
    """
    dt = torch.float64
    num_heads, num_v_heads = inputs["heads"]
    q, k, v = (t.detach().to(dt).requires_grad_(True) for t in _split(inputs["mixed"], inputs["heads"]))
    g = inputs["g"].detach().to(dt).requires_grad_(True)
    beta = inputs["beta"].detach().to(dt).requires_grad_(True)
    qf, kf, vf, gf, bf = (t.flatten(0, 1) for t in (q, k, v, g, beta))
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + L2NORM_EPS)
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + L2NORM_EPS)
    group = num_v_heads // num_heads
    qf = qf.repeat_interleave(group, dim=1) * (1.0 / math.sqrt(HEAD_DIM))
    kf = kf.repeat_interleave(group, dim=1)
    outs = []
    for start, end in itertools.pairwise(_cu(seq_lens)):
        S = torch.zeros(num_v_heads, HEAD_DIM, HEAD_DIM, dtype=dt, device=qf.device)
        for t in range(start, end):
            S = S * torch.exp(gf[t])[:, None, None]
            u = bf[t][:, None] * (vf[t] - torch.einsum("hk,hkv->hv", kf[t], S))
            S = S + kf[t][:, :, None] * u[:, None, :]
            outs.append(torch.einsum("hk,hkv->hv", qf[t], S))
    o = torch.stack(outs).reshape(v.shape)
    (o * inputs["do"].to(dt)).sum().backward()
    return {"o": o.detach(), "dq": q.grad, "dk": k.grad, "dv": v.grad, "dg": g.grad, "dbeta": beta.grad}


@cache
def _reference_case(pair: str, layout: tuple[int, ...] = REF_LAYOUT, seed: int = 1) -> tuple[dict, dict]:
    inputs = _inputs(layout, HEADS[pair], seed=seed)
    return inputs, _reference(inputs, layout)


def _metrics(got: torch.Tensor, ref: torch.Tensor) -> dict:
    a, r = got.detach().double(), ref.detach().double()
    diff = (a - r).abs()
    rms_ref = float(r.square().mean().sqrt())
    return {
        "max_abs": float(diff.max()),
        "max_ref": float(r.abs().max()),
        "rel_rms": float(diff.square().mean().sqrt()) / rms_ref if rms_ref > 0 else 0.0,
        "finite": bool(torch.isfinite(a).all()),
    }


def _assert_within(got: dict, ref: dict, what: str, *, names=NAMES) -> None:
    for name in names:
        assert got[name].shape == ref[name].shape, (what, name, got[name].shape, ref[name].shape)
    report = {name: _metrics(got[name], ref[name]) for name in names}
    failed = [
        name
        for name, m in report.items()
        if not (m["finite"] and m["rel_rms"] <= REL_RMS_MAX and m["max_abs"] <= MAX_ABS_REL * m["max_ref"])
    ]
    assert not failed, f"{what}: outside the accuracy budgets: {failed}; {report}"


def _assert_same_bits(got: dict, want: dict, what: str) -> None:
    differing = [name for name in want if not torch.equal(got[name], want[name])]
    assert not differing, f"{what}: not the same bits: {differing}"


@contextlib.contextmanager
def _deterministic_algorithms():
    """PyTorch's deterministic algorithms, which miles' --deterministic-mode turns on."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _one_rank(seq_lens):
    return build_kcp_context(_cu(seq_lens), None, device=torch.device("cuda", torch.cuda.current_device()))


def _kcp_module(variant: str):
    return ops._package(f"{variant}_kcp").cake_kcp


# --------------------------------------------------------------------------------------------------------------------
# one GPU


@pytest.mark.parametrize("save_state", [True, False])
@PAIRS
@VARIANTS
def test_kcp_matches_the_fp64_recurrence(variant, pair, save_state):
    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=not save_state)
    inputs, ref = _reference_case(pair)

    got = _run(inputs, _one_rank(REF_LAYOUT))
    last = _kcp_module(variant).LAST_CALL["fwd"]
    assert (last["backend"], last["save_mode"]) == ("kernels", "full" if save_state else "light")
    _assert_within(got, ref, f"{variant} KCP vs the FP64 recurrence")


@VARIANTS
def test_kcp_at_two_segments_without_the_received_transfers(variant, monkeypatch):
    # flashinfer-ai/flashinfer#8585: the full saved set at P > 1 with save_transfer=False lost M_seg
    cake.configure(cake_gdn_variant=variant)
    monkeypatch.setattr(ops, "KCP_OPTIONS", P2_OPTIONS)
    inputs, ref = _reference_case("4x12", P2_LAYOUT, 2)

    got = _run(inputs, _one_rank(P2_LAYOUT))
    last = _kcp_module(variant).LAST_CALL["fwd"]
    assert (last["P"], last["save_transfer"], last["save_mode"]) == (2, False, "full")
    _assert_within(got, ref, f"{variant} KCP at P = 2 vs the FP64 recurrence")


@PAIRS
@VARIANTS
def test_det_repeats_bit_for_bit_and_fast_stays_within_the_budgets(variant, pair):
    # det: bit for bit run to run; fast makes no promise about its bits, only its accuracy
    same = _assert_same_bits if variant == "det" else _assert_within
    cake.configure(cake_gdn_variant=variant)
    inputs = _inputs(MILES_LAYOUT, HEADS[pair], seed=3)
    ctx = _one_rank(MILES_LAYOUT)

    first = _run(inputs, ctx)
    same(_run(inputs, ctx), first, "a second run")
    with _deterministic_algorithms():
        same(_run(inputs, ctx), first, "under deterministic algorithms")
    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=True)
    replayed = _run(inputs, ctx)
    same(_run(inputs, ctx), replayed, "a second run, tapes replayed")
    if variant == "det":
        # the package's light replay gives bitwise the gradients of the full saved set
        _assert_same_bits(replayed, first, "det: tapes replayed vs kept")
    else:
        _assert_within(replayed, first, "fast: tapes replayed vs kept")


@VARIANTS
def test_bf16_beta_is_cast_exactly(variant):
    # det: BF16 beta gives the bits of its exact FP32 cast, and miles' views those of contiguous inputs; fast: within
    # the budgets
    same = _assert_same_bits if variant == "det" else _assert_within
    cake.configure(cake_gdn_variant=variant)
    inputs = _inputs(MILES_LAYOUT, HEADS["4x12"], seed=4)
    ctx = _one_rank(MILES_LAYOUT)

    bf16 = _run(inputs, ctx)
    fp32 = _run(inputs, ctx, beta_fp32=True)
    assert bf16["dbeta"].dtype == torch.bfloat16 and fp32["dbeta"].dtype == torch.float32
    fp32["dbeta"] = fp32["dbeta"].to(torch.bfloat16)
    same(bf16, fp32, "BF16 beta vs its FP32 cast")
    same(_run(inputs, ctx, views=False), bf16, "contiguous inputs vs miles' views")


@VARIANTS
def test_kcp_matches_the_single_rank_package(variant):
    cake.configure(cake_gdn_variant=variant)
    inputs = _inputs(MILES_LAYOUT, HEADS["16x32"], seed=5)
    ctx = _one_rank(MILES_LAYOUT)

    kcp = _run(inputs, ctx)
    single = _run(inputs, ctx, cp_context=False)
    _assert_within(kcp, single, f"{variant} KCP vs the {variant} single-rank package")


@VARIANTS
def test_an_initial_or_a_final_state_is_rejected_when_the_call_is_made(variant, monkeypatch):
    # FLA's contract under context parallelism; rejected before any launch of either package
    cake.configure(cake_gdn_variant=variant)
    calls = []
    for name in ("det_kcp", "fast_kcp"):
        package = ops._package(name)
        entry = "cake_gdn_det_kcp" if name == "det_kcp" else "cake_gdn_nondet_kcp"
        monkeypatch.setattr(package, entry, lambda *args, _name=name, **kwargs: calls.append(_name))
    monkeypatch.setattr(ops, "_cp_apply", lambda *args: calls.append("fla_cp"))
    inputs = _inputs(REF_LAYOUT, HEADS["4x12"], seed=6)
    ctx = _one_rank(REF_LAYOUT)

    with pytest.raises(ValueError, match="output_final_state is not supported under context parallelism"):
        _run(inputs, ctx, output_final_state=True)
    h0 = torch.zeros(len(REF_LAYOUT), 12, HEAD_DIM, HEAD_DIM, device="cuda")
    q, k, v = _split(inputs["mixed"], inputs["heads"])
    with pytest.raises(ValueError, match="initial_state is not supported under context parallelism"):
        ops.chunk_gated_delta_rule(
            q, k, v, g=inputs["g"], beta=inputs["beta"], initial_state=h0, use_qk_l2norm_in_kernel=True, cp_context=ctx
        )
    assert calls == []


@VARIANTS
def test_a_call_the_package_does_not_serve_runs_the_fla_cp_det_kernels_and_says_so_once(variant, monkeypatch, caplog):
    cake.configure(cake_gdn_variant=variant)
    monkeypatch.setattr(ops, "_KCP_LOGGED", set())
    monkeypatch.setattr(config, "_warned", set())
    # a package build without the records of the in-kernel q / k L2 norm
    monkeypatch.setattr(_kcp_module(variant), "l2n_records_shipped", lambda: False)
    inputs = _inputs(REF_LAYOUT, HEADS["4x12"], seed=6)
    ctx = _one_rank(REF_LAYOUT)

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        routed = [_run(inputs, ctx) for _ in range(2)]
    # the det kernels with FLA's context parallelism as one rank, on the host table of the same layout
    one_rank = types.SimpleNamespace(group=None, cu_seqlens=ctx.cu_seqlens, cu_seqlens_cpu=_cu(REF_LAYOUT))
    direct = _run(inputs, ctx, kernel=ops.cp_chunk_gated_delta_rule, cp_context=one_rank)
    for run in routed:
        _assert_same_bits(run, direct, "the routed call vs the FLA-CP det kernels")
    reason = "the in-kernel q / k L2 norm (no records in this package build)"
    if variant == "det":
        expected = (
            f"miles_ext_kernels.cake.gdn: the det KCP package does not serve {reason}; running the det kernels of "
            "FLA's context parallelism"
        )
    else:
        expected = (
            f"miles_ext_kernels.cake.gdn: the fast variant does not serve {reason} with a KcpContext; running det"
        )
    assert [r.getMessage() for r in caplog.records if r.name == "miles_ext_kernels"] == [expected]


# --------------------------------------------------------------------------------------------------------------------
# two and four GPUs: one process per GPU

# name -> (variant, the tapes kept, layout, KCP_OPTIONS)
CP_CASES = {
    "det": ("det", True, "contiguous", None),
    "det_replayed": ("det", False, "contiguous", None),
    "fast": ("fast", True, "contiguous", None),
    "fast_replayed": ("fast", False, "contiguous", None),
    "det_zigzag": ("det", True, "zigzag_per_document", None),
    "fast_zigzag": ("fast", True, "zigzag_per_document", None),
    "det_p2": ("det", True, "contiguous", P2_OPTIONS),
    "fast_p2": ("fast", True, "contiguous", P2_OPTIONS),
}
CP_SEED = 11


def _rows(ctx, variant) -> list[int]:
    """The global token index of every local token of ``ctx``'s rank, in local order."""
    cu = ctx.cu_seqlens_global
    rows = []
    for fragment in ctx.partition(variant).fragments[ctx.rank]:
        start = cu[fragment.doc] + fragment.off
        rows.extend(range(start, start + fragment.length))
    return rows


def _rank_run(inputs, ctx, rows, *, output_final_state=False, cp_context=None, kernel=None) -> dict:
    """One forward and backward of this rank's tokens."""
    leaves = _leaves(inputs, rows)
    kernel = ops.chunk_gated_delta_rule if kernel is None else kernel
    o, final_state = kernel(
        *(leaves[n] for n in ("q", "k", "v")),
        g=leaves["g"],
        beta=leaves["beta"],
        cu_seqlens=ctx.cu_seqlens,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        cp_context=ctx if cp_context is None else cp_context,
    )
    o.backward(inputs["do"][:, rows])
    torch.cuda.synchronize()
    out = {"o": o.detach(), **{f"d{n}": t.grad for n, t in leaves.items()}}
    if output_final_state:
        out["final_state"] = final_state.detach()
    return {n: t.cpu() for n, t in out.items()}


def _same(a: dict, b: dict) -> list[str]:
    return [name for name in b if not torch.equal(a[name], b[name])]


def _guarded(results: dict, name: str, run) -> None:
    """``results[name] = run()``; when ``run`` raises, the traceback is kept as the case's ``error`` instead.  This
    isolates a case that raises on every rank before any communication: it fails its own tests and the other cases of
    the shared multi-GPU run still report.  A case that raises on some ranks only is not isolated (the collectives of
    the ranks no longer match); the process group's timeout (:func:`_cp_worker`) bounds the resulting hang."""
    try:
        results[name] = run()
    except Exception:  # reported by the case's tests (:func:`_case`)
        results[name] = {"error": traceback.format_exc()}


def _case_run(inputs, group, variant, save_state, layout, options) -> dict:
    """One context-parallel case on this rank: three runs (the third under deterministic algorithms)."""
    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=not save_state)
    ops.KCP_OPTIONS = options
    try:
        ctx = build_kcp_context(_cu(CP_LAYOUT), group, layout=layout)
        rows = _rows(ctx, variant)
        first = _rank_run(inputs, ctx, rows)
        second = _rank_run(inputs, ctx, rows)
        with _deterministic_algorithms():
            third = _rank_run(inputs, ctx, rows)
        last = _kcp_module(variant).LAST_CALL["fwd"]
    finally:
        ops.KCP_OPTIONS = None
    return {
        "rows": rows,
        "run": first,
        "differs_run_to_run": _same(second, first),
        "differs_deterministic_algorithms": _same(third, first),
        "P": last["P"],
        "backend": last["backend"],
    }


def _final_state_run(inputs, group) -> str | None:
    """A final state under context parallelism: the ``ValueError`` message of the call, or ``None`` when it ran."""
    cake.configure(cake_gdn_variant="det")
    ctx = build_kcp_context(_cu(CP_LAYOUT), group)
    rows = _rows(ctx, "det")
    try:
        _rank_run(inputs, ctx, rows, output_final_state=True)
    except ValueError as exc:
        return str(exc)
    return None


def _fla_cp_run(inputs, group) -> dict | None:
    """A call the package does not serve (here: a build without the in-kernel L2 norm records) runs the det kernels
    with FLA's context parallelism, against the call on an FLA context built as miles does; ``None`` without FLA."""
    try:
        from fla.ops.cp import build_cp_context
    except ImportError:
        return None
    cake.configure(cake_gdn_variant="fast")
    ctx = build_kcp_context(_cu(CP_LAYOUT), group)
    rows = _rows(ctx, "det")
    shipped = {variant: _kcp_module(variant).l2n_records_shipped for variant in ("det", "fast")}
    try:
        for variant in shipped:
            _kcp_module(variant).l2n_records_shipped = lambda: False
        routed = _rank_run(inputs, ctx, rows)
    finally:
        for variant, function in shipped.items():
            _kcp_module(variant).l2n_records_shipped = function
    fla = build_cp_context(torch.tensor(_cu(CP_LAYOUT), dtype=torch.int32, device="cuda"), group, 4)
    assert fla.cu_seqlens.tolist() == list(ctx.cu_seqlens_cpu)
    direct = _rank_run(inputs, ctx, rows, cp_context=fla, kernel=ops.cp_chunk_gated_delta_rule)
    return {"differs": _same(routed, direct), "names": sorted(routed)}


def _cp_worker(rank: int, world_size: int, store: str, out_dir: str) -> None:
    torch.cuda.set_device(rank)
    # a bounded timeout: unmatched collectives (a case that raised on some ranks only) fail the run instead of hanging
    dist.init_process_group(
        "nccl",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(minutes=5),
    )
    try:
        group = dist.group.WORLD
        inputs = _inputs(CP_LAYOUT, CP_HEADS, seed=CP_SEED)
        results = {}
        for case, spec in CP_CASES.items():
            _guarded(results, case, lambda spec=spec: _case_run(inputs, group, *spec))
        # a final state under context parallelism: rejected on every rank before any communication, as FLA does
        _guarded(results, "final_state", lambda: _final_state_run(inputs, group))
        _guarded(results, "fla_cp", lambda: _fla_cp_run(inputs, group))
        torch.save(results, Path(out_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=[2, 4], ids=lambda n: f"cp{n}")
def cp_results(request, tmp_path_factory):
    world_size = request.param
    visible = torch.cuda.device_count()
    if visible < world_size:
        pytest.skip(f"context parallelism {world_size} needs {world_size} GPUs; {visible} visible")
    out_dir = tmp_path_factory.mktemp(f"kcp-cp{world_size}")
    mp.spawn(_cp_worker, args=(world_size, str(out_dir / "store"), str(out_dir)), nprocs=world_size, join=True)
    per_rank = [torch.load(out_dir / f"rank{r}.pt", weights_only=False) for r in range(world_size)]
    return world_size, per_rank


def _case(per_rank, name: str) -> list:
    """Every rank's result of case ``name``; fails with the rank's traceback when the case raised there."""
    results = [result[name] for result in per_rank]
    for rank, result in enumerate(results):
        if isinstance(result, dict) and "error" in result:
            pytest.fail(f"{name} raised on rank {rank} of {len(results)}:\n{result['error']}")
    return results


def _assemble(results) -> dict:
    """The whole batch's output and gradients from every rank's local tokens (one case's per-rank results)."""
    total = sum(CP_LAYOUT)
    out = {}
    for name in NAMES:
        local = results[0]["run"][name]
        full = torch.empty((1, total, *local.shape[2:]), dtype=local.dtype)
        for result in results:
            full[:, result["rows"]] = result["run"][name]
        out[name] = full.cuda()
    return out


@cache
def _cp_reference() -> tuple[dict, dict]:
    inputs = _inputs(CP_LAYOUT, CP_HEADS, seed=CP_SEED)
    return inputs, _reference(inputs, CP_LAYOUT)


CASES = pytest.mark.parametrize("case", list(CP_CASES))
MULTI_GPU = pytest.mark.multi_gpu


@MULTI_GPU
@CASES
def test_cp_matches_the_fp64_recurrence_and_one_gpu(cp_results, case, monkeypatch):
    world_size, per_rank = cp_results
    results = _case(per_rank, case)
    variant, save_state, _layout, options = CP_CASES[case]
    assert sorted(r for result in results for r in result["rows"]) == list(range(sum(CP_LAYOUT)))
    assert {result["backend"] for result in results} == {"kernels"}
    if options is not None:
        assert {result["P"] for result in results} == {options["segments"]}
    got = _assemble(results)
    inputs, ref = _cp_reference()
    _assert_within(got, ref, f"CP {world_size} {case} vs the FP64 recurrence")

    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=not save_state)
    monkeypatch.setattr(ops, "KCP_OPTIONS", options)
    one_gpu = _run(inputs, _one_rank(CP_LAYOUT), views=False)
    _assert_within(got, one_gpu, f"CP {world_size} {case} vs one GPU")


@MULTI_GPU
@pytest.mark.parametrize("case", [case for case, spec in CP_CASES.items() if spec[0] == "det"])
def test_cp_det_repeats_bit_for_bit(cp_results, case):
    world_size, per_rank = cp_results
    for rank, result in enumerate(_case(per_rank, case)):
        assert result["differs_run_to_run"] == [], (world_size, rank, case)
        assert result["differs_deterministic_algorithms"] == [], (world_size, rank, case)


@MULTI_GPU
def test_cp_det_gives_the_same_bits_with_the_tapes_kept_or_replayed(cp_results):
    world_size, per_rank = cp_results
    for rank, (replayed, kept) in enumerate(zip(_case(per_rank, "det_replayed"), _case(per_rank, "det"))):
        assert _same(replayed["run"], kept["run"]) == [], (world_size, rank)


@MULTI_GPU
def test_cp_a_final_state_call_is_rejected_when_the_call_is_made(cp_results):
    world_size, per_rank = cp_results
    for rank, message in enumerate(_case(per_rank, "final_state")):
        assert message is not None, (world_size, rank, "a final state under context parallelism ran")
        assert "output_final_state is not supported under context parallelism" in message


@MULTI_GPU
def test_cp_a_call_the_package_does_not_serve_runs_the_fla_cp_det_kernels_with_their_bits(cp_results):
    world_size, per_rank = cp_results
    results = _case(per_rank, "fla_cp")
    if results[0] is None:
        pytest.skip("FLA's context parallelism with a process group needs flash-linear-attention (fla.ops.cp)")
    for rank, result in enumerate(results):
        assert result["names"] == sorted(NAMES)
        assert result["differs"] == [], (world_size, rank)
