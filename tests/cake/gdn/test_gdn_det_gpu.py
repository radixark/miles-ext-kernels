"""The det GDN variant without a ``cp_context`` (``--cake-gdn-variant det``: the vendored Cake deterministic package
``cake_gdn_det_train``) at the shapes miles feeds it.

Against the FP64 token recurrence, output, final state and every input gradient (the initial state's included) within
the accuracy budgets of the package's own suite; bit for bit run to run, also under PyTorch's deterministic algorithms;
miles' strided q / k views give the bits of contiguous inputs; the recomputed backward (``--cake-gdn-recompute-state``)
against the saved one (:data:`RECOMPUTE_SAME_BITS`); within the budgets of the det kernels with FLA's context
parallelism that a call with a ``cp_context`` runs; a call without a ``cp_context`` reaches the package with
``--cake-gdn-recompute-state`` and a call with one does not. The bits also repeat in a new process and on a side
stream alternating with the default one (with the workspace pool bounded, ``helpers.POOLED``, so each stream holds its
own buffers); under that bound the scratch buffers come from the pool, at the bits of the default unpooled run; the
extents of every input are checked before a launch; a call without a token returns an empty output and passes the state
through; one sequence of up to 128k tokens runs on the package and on the det kernels with FLA's context parallelism,
each bit for bit run to run.

The head pairs are TP-local (key heads, value heads) of miles' GDN layers, (4, 12) and (16, 32), and one value head per
key head, (4, 4), on ragged multi-document layouts whose lengths are not multiples of the 64-token chunk, on the
host-table route (``cu_seqlens_cpu``, as miles calls) and on the device route. Needs an SM100a / SM103a / SM107a GPU
and the vendored package; skips elsewhere.
"""

import contextlib
import itertools
import math
import os
import subprocess
import sys
import types
from functools import cache
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.gdn import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import loader, workspace  # noqa: E402

from ..helpers import POOLED  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the GDN kernels need SM100a / SM103a / SM107a", allow_module_level=True)

HEAD_DIM = 128
# (key heads, value heads) of one tensor-parallel rank
HEADS = {"4x12": (4, 12), "16x32": (16, 32), "4x4": (4, 4)}
# short enough for the FP64 token recurrence
REF_LAYOUT = (63, 130, 257, 450)
# a TP4 micro-batch: three documents and the 512-token alignment pad as its own segment
MILES_LAYOUT = (3339, 2802, 1340, 263)
NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")
STATE_NAMES = ("final_state", "dh0")
# The package suite's accuracy budgets, per output tensor: the relative RMS error at most REL_RMS_MAX and the largest
# absolute error at most MAX_ABS_REL times the largest reference magnitude.
REL_RMS_MAX = 1e-2
MAX_ABS_REL = 2e-2
L2NORM_EPS = 1e-6  # the in-kernel q / k normalization: x / sqrt(sum x^2 + 1e-6)
# Whether the recomputed backward (``save_state=False``) must give the saved backward's bits. The package promises its
# accuracy budgets here, not its bits: the recompute scans the state with the register scan, the forward with its fused
# scan. While False, the recomputed backward must stay within the accuracy budgets of the saved one, and each repeats
# its own bits.
RECOMPUTE_SAME_BITS = False


@pytest.fixture(autouse=True)
def _kernel_settings():
    # det without a cp_context, the tape kept; the conftest restores the default configuration afterwards
    cake.configure(cake_gdn_variant="det")


def _cu(seq_lens) -> tuple[int, ...]:
    return tuple(itertools.accumulate(seq_lens, initial=0))


def _inputs(seq_lens, heads, *, seed: int) -> dict:
    """Packed ``[1, T, ...]`` operands, drawn on the host: the short conv's output (``mixed``, rows group-major: per
    key head its q, its k, then its value heads), g in FP32 and beta in BF16, as miles' GDN layer computes them; and an
    FP32 initial state and final-state gradient per sequence for the runs that pass the state."""
    num_heads, num_v_heads = heads
    total = sum(seq_lens)
    gen = torch.Generator().manual_seed(seed)
    mixed = torch.randn(1, total, num_heads * 2 * HEAD_DIM + num_v_heads * HEAD_DIM, generator=gen)
    g = torch.rand(1, total, num_v_heads, generator=gen) * -0.5
    beta = torch.rand(1, total, num_v_heads, generator=gen)
    do = torch.randn(1, total, num_v_heads, HEAD_DIM, generator=gen)
    h0 = torch.randn(len(seq_lens), num_v_heads, HEAD_DIM, HEAD_DIM, generator=gen) * 0.1
    dht = torch.randn(len(seq_lens), num_v_heads, HEAD_DIM, HEAD_DIM, generator=gen) * 0.1
    return {
        "heads": heads,
        "mixed": mixed.to(device="cuda", dtype=torch.bfloat16),
        "g": g.cuda(),
        "beta": beta.to(device="cuda", dtype=torch.bfloat16),
        "do": do.to(device="cuda", dtype=torch.bfloat16),
        "h0": h0.cuda(),
        "dht": dht.cuda(),
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


def _run(inputs, seq_lens, *, route="host", views=False, state=False, kernel=None, cp_context=None) -> dict:
    """One forward and backward; the output and the gradients of q, k, v, g and beta.

    ``ops.chunk_gated_delta_rule`` runs (``det``, the autouse fixture), or ``kernel`` runs. ``route``: ``"host"``
    passes the host offsets as miles does (``cu_seqlens_cpu``), ``"device"`` the device tensor only. ``views``: q / k /
    v are miles' views of one ``mixed`` leaf instead of contiguous leaves. ``state``: an initial state goes in and the
    final state comes out, with a gradient, adding ``final_state`` and ``dh0``.
    """
    heads = inputs["heads"]
    g = inputs["g"].detach().clone().requires_grad_(True)
    beta = inputs["beta"].detach().clone().requires_grad_(True)
    if views:
        mixed = inputs["mixed"].detach().clone().requires_grad_(True)
        q, k, v = _split(mixed, heads)
        assert not q.is_contiguous() and not k.is_contiguous()
    else:
        q, k, v = (
            t.clone(memory_format=torch.contiguous_format).requires_grad_(True)
            for t in _split(inputs["mixed"].detach(), heads)
        )
    h0 = inputs["h0"].detach().clone().requires_grad_(True) if state else None
    kwargs = {"cu_seqlens": torch.tensor(_cu(seq_lens), dtype=torch.int32, device="cuda")}
    if route == "host":
        kwargs["cu_seqlens_cpu"] = _cu(seq_lens)
    if cp_context is not None:
        kwargs["cp_context"] = cp_context
    kernel = ops.chunk_gated_delta_rule if kernel is None else kernel
    o, final_state = kernel(
        q, k, v, g=g, beta=beta, initial_state=h0, output_final_state=state, use_qk_l2norm_in_kernel=True, **kwargs
    )
    if state:
        torch.autograd.backward([o, final_state], [inputs["do"], inputs["dht"]])
    else:
        o.backward(inputs["do"])
    torch.cuda.synchronize()
    dq, dk, dv = _split(mixed.grad, heads) if views else (q.grad, k.grad, v.grad)
    out = {"o": o.detach(), "dq": dq, "dk": dk, "dv": dv, "dg": g.grad, "dbeta": beta.grad}
    if state:
        out.update(final_state=final_state.detach(), dh0=h0.grad)
    return out


def _reference(inputs, seq_lens, *, state: bool) -> dict:
    """The FP64 token recurrence of the gated delta rule and its autograd gradients, per sequence and value head ``hv``
    (key head ``hv // (HV / H)``) with the state ``S`` ``[K, V]`` (zero, or the sequence's initial state)::

        S = S * exp(g_t);  u_t = beta_t (v_t - S^T k_t);  S = S + k_t u_t^T;  o_t = scale S^T q_t
    """
    dt = torch.float64
    num_heads, num_v_heads = inputs["heads"]
    q, k, v = (t.detach().to(dt).requires_grad_(True) for t in _split(inputs["mixed"], inputs["heads"]))
    g = inputs["g"].detach().to(dt).requires_grad_(True)
    beta = inputs["beta"].detach().to(dt).requires_grad_(True)
    h0 = inputs["h0"].detach().to(dt).requires_grad_(True)
    qf, kf, vf, gf, bf = (t.flatten(0, 1) for t in (q, k, v, g, beta))
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + L2NORM_EPS)
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + L2NORM_EPS)
    group = num_v_heads // num_heads
    qf = qf.repeat_interleave(group, dim=1) * (1.0 / math.sqrt(HEAD_DIM))
    kf = kf.repeat_interleave(group, dim=1)
    outs, finals = [], []
    for n, (start, end) in enumerate(itertools.pairwise(_cu(seq_lens))):
        S = h0[n] if state else torch.zeros(num_v_heads, HEAD_DIM, HEAD_DIM, dtype=dt, device=qf.device)
        for t in range(start, end):
            S = S * torch.exp(gf[t])[:, None, None]
            u = bf[t][:, None] * (vf[t] - torch.einsum("hk,hkv->hv", kf[t], S))
            S = S + kf[t][:, :, None] * u[:, None, :]
            outs.append(torch.einsum("hk,hkv->hv", qf[t], S))
        finals.append(S)
    o = torch.stack(outs).reshape(v.shape)
    loss = (o * inputs["do"].to(dt)).sum()
    if state:
        final_state = torch.stack(finals)
        loss = loss + (final_state * inputs["dht"].to(dt)).sum()
    loss.backward()
    ref = {"o": o.detach(), "dq": q.grad, "dk": k.grad, "dv": v.grad, "dg": g.grad, "dbeta": beta.grad}
    if state:
        ref.update(final_state=final_state.detach(), dh0=h0.grad)
    return ref


@cache
def _reference_case(pair: str, state: bool) -> tuple[dict, dict]:
    """The inputs of :data:`REF_LAYOUT` for one head pair and their FP64 reference (computed once per case)."""
    inputs = _inputs(REF_LAYOUT, HEADS[pair], seed=1)
    return inputs, _reference(inputs, REF_LAYOUT, state=state)


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
    """Every output ``names`` of ``got`` within the accuracy budgets of ``ref``."""
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


PAIRS = pytest.mark.parametrize("pair", list(HEADS))
ROUTES = pytest.mark.parametrize("route", ["host", "device"])


@pytest.mark.parametrize("state", [False, True])
@ROUTES
@PAIRS
def test_det_matches_the_fp64_recurrence(pair, route, state):
    inputs, ref = _reference_case(pair, state)

    det = _run(inputs, REF_LAYOUT, route=route, state=state)
    _assert_within(det, ref, "det vs the FP64 recurrence", names=NAMES + (STATE_NAMES if state else ()))


@ROUTES
@PAIRS
def test_det_repeats_bit_for_bit(pair, route):
    inputs = _inputs(MILES_LAYOUT, HEADS[pair], seed=3)

    first = _run(inputs, MILES_LAYOUT, route=route)
    _assert_same_bits(_run(inputs, MILES_LAYOUT, route=route), first, "a second run")
    # miles' --deterministic-mode: no launch of the package raises PyTorch's alert, and the bits stay
    with _deterministic_algorithms():
        _assert_same_bits(_run(inputs, MILES_LAYOUT, route=route), first, "under deterministic algorithms")
    # with the state passed in and out, too
    stateful = _run(inputs, MILES_LAYOUT, route=route, state=True)
    _assert_same_bits(_run(inputs, MILES_LAYOUT, route=route, state=True), stateful, "a second run with the state")


@PAIRS
def test_det_on_miles_views_gives_the_bits_of_contiguous_inputs(pair):
    inputs = _inputs(MILES_LAYOUT, HEADS[pair], seed=2)

    # the package makes strided q / k / v contiguous itself
    _assert_same_bits(
        _run(inputs, MILES_LAYOUT, views=True), _run(inputs, MILES_LAYOUT), "strided views vs contiguous inputs"
    )


@ROUTES
@PAIRS
def test_the_recomputed_backward(pair, route):
    # --cake-gdn-recompute-state: w / u / h / v_new recomputed in the backward, which repeats bit for bit, also under
    # deterministic algorithms; against the saved backward see RECOMPUTE_SAME_BITS
    inputs = _inputs(MILES_LAYOUT, HEADS[pair], seed=4)
    saved = _run(inputs, MILES_LAYOUT, route=route)

    cake.configure(cake_gdn_variant="det", cake_gdn_recompute_state=True)
    recomputed = _run(inputs, MILES_LAYOUT, route=route)
    _assert_same_bits(_run(inputs, MILES_LAYOUT, route=route), recomputed, "a second recomputed run")
    with _deterministic_algorithms():
        _assert_same_bits(_run(inputs, MILES_LAYOUT, route=route), recomputed, "recomputed, deterministic algorithms")
    if RECOMPUTE_SAME_BITS:
        _assert_same_bits(recomputed, saved, "the recomputed backward vs the saved one")
    else:
        _assert_within(recomputed, saved, "the recomputed backward vs the saved one")


@PAIRS
def test_det_matches_the_generated_cp_kernels(pair):
    # the generated kernels serve every call with a cp_context; here as one rank, on miles' views
    inputs = _inputs(MILES_LAYOUT, HEADS[pair], seed=5)

    det = _run(inputs, MILES_LAYOUT, views=True)
    cp = _run(inputs, MILES_LAYOUT, views=True, kernel=ops.cp_chunk_gated_delta_rule)
    _assert_within(det, cp, "det vs the generated kernels")


# One long sequence per case, (key heads, value heads) and tokens: the shapes at which the planner of the det kernels
# with FLA's context parallelism used to cut state-scan segments longer than the 128 chunks one pre-scan CTA covers
# (171 chunks forward at 48 value heads and 32k tokens, 171 backward at 12 and 64k, 228 forward at 16 and 128k).
LONG_CASES = {"48v_32k": ((16, 48), 32768), "12v_64k": ((4, 12), 65536), "16v_128k": ((4, 16), 131072)}


@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize("case", list(LONG_CASES))
def test_a_long_single_sequence_runs_on_the_det_package_and_the_fla_cp_det_kernels(case, route):
    heads, tokens = LONG_CASES[case]
    seq_lens = (tokens,)
    inputs = _inputs(seq_lens, heads, seed=13)

    det = _run(inputs, seq_lens, route=route)
    _assert_same_bits(_run(inputs, seq_lens, route=route), det, f"{case} {route}: the det package run to run")
    cp = _run(inputs, seq_lens, route=route, kernel=ops.cp_chunk_gated_delta_rule)
    _assert_same_bits(
        _run(inputs, seq_lens, route=route, kernel=ops.cp_chunk_gated_delta_rule),
        cp,
        f"{case} {route}: the det kernels with FLA's context parallelism run to run",
    )
    _assert_within(cp, det, f"{case} {route}: the det kernels with FLA's context parallelism vs the det package")


def _fla_context(seq_lens) -> types.SimpleNamespace:
    """An FLA context without a process group (one rank) over ``seq_lens``: its boundaries on device and host."""
    cu = _cu(seq_lens)
    return types.SimpleNamespace(
        group=None, cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="cuda"), cu_seqlens_cpu=cu
    )


def test_a_call_without_a_cp_context_reaches_the_package_and_a_cp_context_call_does_not(monkeypatch):
    package = ops._package("det")
    entry = package.cake_gdn_det
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["save_state"])
        return entry(*args, **kwargs)

    monkeypatch.setattr(package, "cake_gdn_det", counted)
    inputs = _inputs(REF_LAYOUT, HEADS["4x12"], seed=6)

    for save_state in (True, False):
        cake.configure(cake_gdn_variant="det", cake_gdn_recompute_state=not save_state)
        _run(inputs, REF_LAYOUT)
    assert calls == [True, False], "a det call without a cp_context must reach the package with save_state"
    # a CP context without a CP group: the det kernels with FLA's context parallelism run it as one rank
    _run(inputs, REF_LAYOUT, cp_context=_fla_context(REF_LAYOUT))
    assert calls == [True, False], "a cp_context call reached the package"


@pytest.mark.parametrize("cp", [False, True])
def test_det_repeats_on_a_side_stream_alternating_with_the_default_stream(cp):
    # the pooled scratch is kept per stream (a positive bound: the default, 0, pools nothing): alternating streams
    # never share a buffer, and the bits stay
    cake.configure(cake_gdn_variant="det", **POOLED)
    inputs = _inputs(MILES_LAYOUT, HEADS["16x32"], seed=8)
    kwargs = {"cp_context": _fla_context(MILES_LAYOUT)} if cp else {"state": True}
    first = _run(inputs, MILES_LAYOUT, **kwargs)
    side = torch.cuda.Stream()
    for _ in range(2):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            on_side = _run(inputs, MILES_LAYOUT, **kwargs)
        torch.cuda.current_stream().wait_stream(side)
        _assert_same_bits(on_side, first, "on a side stream")
        _assert_same_bits(_run(inputs, MILES_LAYOUT, **kwargs), first, "back on the default stream")


def test_the_scratch_comes_from_the_bounded_pool_and_the_pool_does_not_change_the_bits():
    cake.configure(cake_gdn_variant="det", **POOLED)
    index = torch.cuda.current_device()
    inputs = _inputs(MILES_LAYOUT, HEADS["4x12"], seed=9)
    cake.release("gdn")
    pooled = _run(inputs, MILES_LAYOUT)
    held = workspace.stats()[index]
    assert held["entries"] > 0 and 0 < held["bytes"] <= cake.option(None, "cake_workspace_pool_bytes")
    _assert_same_bits(_run(inputs, MILES_LAYOUT), pooled, "a second pooled run")
    again = workspace.stats()[index]
    # the second run of the layout reuses every pooled buffer: more hits, nothing allocated or evicted
    assert again["hits"] > held["hits"]
    kept = ("bytes", "entries", "evictions")
    assert [again[name] for name in kept] == [held[name] for name in kept]

    assert cake.release("gdn") == again["bytes"]
    assert workspace.stats()[index]["bytes"] == 0
    cake.configure(cake_gdn_variant="det", cake_workspace_pool_bytes=0)
    _assert_same_bits(_run(inputs, MILES_LAYOUT), pooled, "the pool off")
    assert workspace.stats()[index]["bytes"] == 0


_NEW_PROCESS = """
import sys
import torch
from miles_ext_kernels import cake
from cake.gdn import test_gdn_det_gpu as t

cake.configure(cake_gdn_variant="det")
inputs = t._inputs(t.MILES_LAYOUT, t.HEADS["4x12"], seed=10)
torch.save(
    {route: t._run(inputs, t.MILES_LAYOUT, route=route, state=True) for route in ("host", "device")}, sys.argv[1]
)
"""


def test_det_repeats_bit_for_bit_in_a_new_process(tmp_path):
    inputs = _inputs(MILES_LAYOUT, HEADS["4x12"], seed=10)
    here = {route: _run(inputs, MILES_LAYOUT, route=route, state=True) for route in ("host", "device")}
    tests = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(filter(None, [str(tests.parent), str(tests), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NEW_PROCESS, str(tmp_path / "bits.pt")],
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    there = torch.load(tmp_path / "bits.pt", map_location="cuda")
    for route in here:
        _assert_same_bits(there[route], here[route], f"a new process, {route} route")


@pytest.mark.parametrize("cp", [False, True])
def test_the_extents_are_checked_before_a_launch(cp):
    # the det package (no cp_context) and the det kernels with FLA's context parallelism refuse every input whose
    # extents do not match the call, from tensor metadata (no launch, no synchronisation); the call that follows is
    # well posed: q / k normalised in the kernel, as miles calls it (without the norm, the unit-variance q / k of these
    # inputs make the state grow without bound: FLA's chunked kernel also gives non-finite outputs from the second
    # chunk on)
    seq_lens = (100, 28)
    inputs = _inputs(seq_lens, HEADS["4x12"], seed=11)
    q, k, v = (t.contiguous() for t in _split(inputs["mixed"], HEADS["4x12"]))
    g, beta = inputs["g"], inputs["beta"]
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int32, device="cuda")
    base = {"cu_seqlens": cu, "cu_seqlens_cpu": _cu(seq_lens)}
    if cp:
        base = {"cp_context": _fla_context(seq_lens)}

    def call(q=q, k=k, v=v, g=g, beta=beta, **kwargs):
        return ops.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, **{**base, **kwargs})

    cases = [
        ("k must have q's shape", dict(k=k[:, :-1])),
        ("v must be", dict(v=v[:, :-1])),
        ("must be a positive multiple", dict(v=v[:, :, :-1])),
        ("g must be", dict(g=g[:, :, :-1])),
        ("beta must be", dict(beta=beta[:, :-1])),
        ("must be a CUDA tensor", dict(g=g.cpu())),
    ]
    if cp:
        # the context's own boundaries: they must cover the tokens of q
        cases += [("cu_seqlens_cpu must end at the 128 tokens", dict(cp_context=_fla_context((100, 27))))]
    else:
        cases += [
            ("cu_seqlens must be a 1-D int32 / int64 tensor", dict(cu_seqlens=cu.float())),
            ("cu_seqlens_cpu has 2 offsets", dict(cu_seqlens_cpu=(0, 128))),
            ("cu_seqlens_cpu must start at 0", dict(cu_seqlens_cpu=(0, 100, 127))),
            ("one state per sequence", dict(initial_state=torch.zeros(3, 12, HEAD_DIM, HEAD_DIM, device="cuda"))),
        ]
    for message, kwargs in cases:
        with pytest.raises(ValueError, match=message):
            call(**kwargs)
    o, _ = call()
    torch.cuda.synchronize()
    assert torch.isfinite(o.float()).all()


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_call_without_tokens(variant, cp):
    cake.configure(cake_gdn_variant=variant)
    heads = HEADS["4x12"]
    inputs = _inputs((5,), heads, seed=12)
    q, k, v = (t[:, :0].detach().clone().requires_grad_(True) for t in _split(inputs["mixed"], heads))
    g = inputs["g"][:, :0].detach().clone().requires_grad_(True)
    beta = inputs["beta"][:, :0].detach().clone().requires_grad_(True)
    if cp:
        kwargs = {"cp_context": _fla_context((0, 0))}
    else:
        h0 = inputs["h0"][:1].expand(2, -1, -1, -1).contiguous().requires_grad_(True)
        kwargs = {
            "cu_seqlens": torch.zeros(3, dtype=torch.int32, device="cuda"),
            "cu_seqlens_cpu": (0, 0, 0),
            "initial_state": h0,
            "output_final_state": True,
        }

    o, final_state = ops.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, **kwargs)

    assert o.shape == v.shape and o.dtype == torch.bfloat16 and o.is_cuda
    if cp:
        assert final_state is None
        o.sum().backward()
    else:
        assert torch.equal(final_state, h0.detach())
        dht = inputs["dht"][:1].expand(2, -1, -1, -1).contiguous()
        torch.autograd.backward([o, final_state], [torch.zeros_like(o), dht])
        assert torch.equal(h0.grad, dht)
    assert all(t.grad is not None and t.grad.shape == t.shape for t in (q, k, v, g, beta))
