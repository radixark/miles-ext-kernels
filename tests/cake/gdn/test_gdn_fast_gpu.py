"""The fast GDN variant (``--cake-gdn-variant fast``: the vendored Cake non-deterministic package) at the shapes miles
feeds it.

Against the FP64 token recurrence, the deterministic package that ``det`` runs without a ``cp_context`` and
flash-linear-attention, output, final state and every input gradient within the accuracy budgets of the package's own
suite, also on miles' strided q / k views, under PyTorch's deterministic algorithms and with the WY / state tensors
recomputed in the backward (``fast`` makes no promise about its bits); a call with an FLA ``cp_context`` runs the det
kernels with FLA's context parallelism, as ``det`` does, with their bits, and says so once (a ``KcpContext`` call:
``test_gdn_kcp_gpu.py``).

The head pairs are TP-local (key heads, value heads) of miles' GDN layers, (4, 12) and (16, 32), on ragged
multi-document layouts whose lengths are not multiples of the 64-token chunk. Needs an SM100a / SM103a / SM107a GPU,
fla and the vendored package; skips elsewhere.
"""

import contextlib
import itertools
import logging
import math
import types
from functools import cache

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.gdn import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the GDN kernels need SM100a / SM103a / SM107a", allow_module_level=True)

HEAD_DIM = 128
# (key heads, value heads) of one tensor-parallel rank
HEADS = {"4x12": (4, 12), "16x32": (16, 32)}
# short enough for the FP64 token recurrence
REF_LAYOUT = (63, 130, 257, 450)
# a TP4 micro-batch: three documents and the 512-token alignment pad as its own segment
MILES_LAYOUT = (3339, 2802, 1340, 263)
NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")
# The package suite's accuracy budgets, per output tensor: the relative RMS error at most REL_RMS_MAX and the largest
# absolute error at most MAX_ABS_REL times the largest reference magnitude. Two independent BF16 implementations
# (this package and fla) each carry their own error, so they are compared at twice the max-abs bound, as that suite
# compares its two table routes.
REL_RMS_MAX = 1e-2
MAX_ABS_REL = 2e-2
L2NORM_EPS = 1e-6  # the in-kernel q / k normalization: x / sqrt(sum x^2 + 1e-6)
RUNS_DET = "miles_ext_kernels.cake.gdn: the fast variant does not serve an FLA cp_context; running det"


@pytest.fixture(autouse=True)
def _kernel_settings():
    # fast, the tape kept; the tests switch the variant and the tape, and the conftest restores the default
    # configuration afterwards
    cake.configure(cake_gdn_variant="fast")


def _set_variant(variant: str) -> None:
    """Configure ``variant``, keeping ``--cake-gdn-recompute-state`` as configured."""
    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=cake.option("gdn", "cake_gdn_recompute_state"))


def _cu(seq_lens) -> tuple[int, ...]:
    return tuple(itertools.accumulate(seq_lens, initial=0))


def _inputs(seq_lens, heads, *, seed: int) -> dict:
    """Packed ``[1, T, ...]`` operands, drawn on the host: the short conv's output (``mixed``, rows group-major: per
    key head its q, its k, then its value heads), g in FP32 and beta in BF16, as miles' GDN layer computes them."""
    num_heads, num_v_heads = heads
    total = sum(seq_lens)
    gen = torch.Generator().manual_seed(seed)
    mixed = torch.randn(1, total, num_heads * 2 * HEAD_DIM + num_v_heads * HEAD_DIM, generator=gen)
    g = torch.rand(1, total, num_v_heads, generator=gen) * -0.5
    beta = torch.rand(1, total, num_v_heads, generator=gen)
    do = torch.randn(1, total, num_v_heads, HEAD_DIM, generator=gen)
    return {
        "heads": heads,
        "mixed": mixed.to(device="cuda", dtype=torch.bfloat16),
        "g": g.cuda(),
        "beta": beta.to(device="cuda", dtype=torch.bfloat16),
        "do": do.to(device="cuda", dtype=torch.bfloat16),
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


def _run(inputs, seq_lens, *, variant=None, route="host", views=False, kernel=None, cp_context=None) -> dict:
    """One forward and backward; the output and the gradients of q, k, v, g and beta.

    ``ops.chunk_gated_delta_rule`` runs with ``variant`` configured (``None``: as configured), or ``kernel`` runs.
    ``route``: ``"host"`` passes the host offsets as miles does (``cu_seqlens_cpu``), ``"device"`` the device tensor
    only. ``views``: q / k / v are miles' views of one ``mixed`` leaf instead of contiguous leaves.
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
    kwargs = {"cu_seqlens": torch.tensor(_cu(seq_lens), dtype=torch.int32, device="cuda")}
    if route == "host":
        kwargs["cu_seqlens_cpu"] = _cu(seq_lens)
    if cp_context is not None:
        kwargs["cp_context"] = cp_context
    if kernel is None:
        if variant is not None:
            _set_variant(variant)
        kernel = ops.chunk_gated_delta_rule
    o, _ = kernel(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True, **kwargs
    )
    o.backward(inputs["do"])
    torch.cuda.synchronize()
    dq, dk, dv = _split(mixed.grad, heads) if views else (q.grad, k.grad, v.grad)
    return {"o": o.detach(), "dq": dq, "dk": dk, "dv": dv, "dg": g.grad, "dbeta": beta.grad}


def _reference(inputs, seq_lens) -> dict:
    """The FP64 token recurrence of the gated delta rule and its autograd gradients, per sequence and value head ``hv``
    (key head ``hv // (HV / H)``) with the state ``S`` ``[K, V]``::

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
def _reference_case(pair: str) -> tuple[dict, dict]:
    """The inputs of :data:`REF_LAYOUT` for one head pair and their FP64 reference (computed once per pair)."""
    inputs = _inputs(REF_LAYOUT, HEADS[pair], seed=1)
    return inputs, _reference(inputs, REF_LAYOUT)


@cache
def _fla_kernel():
    return pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule


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


def _assert_within(got: dict, ref: dict, what: str, *, max_abs_rel: float = MAX_ABS_REL, names=NAMES) -> None:
    """Every output of ``got`` within the accuracy budgets of ``ref``."""
    for name in names:
        assert got[name].shape == ref[name].shape, (what, name, got[name].shape, ref[name].shape)
    report = {name: _metrics(got[name], ref[name]) for name in names}
    failed = [
        name
        for name, m in report.items()
        if not (m["finite"] and m["rel_rms"] <= REL_RMS_MAX and m["max_abs"] <= max_abs_rel * m["max_ref"])
    ]
    assert not failed, f"{what}: outside the accuracy budgets: {failed}; {report}"


def _assert_same_bits(got: dict, want: dict, what: str, *, names=NAMES) -> None:
    differing = [name for name in names if not torch.equal(got[name], want[name])]
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


@pytest.mark.parametrize("route", ["host", "device"])
@PAIRS
def test_fast_matches_the_fp64_recurrence_and_det(pair, route):
    inputs, ref = _reference_case(pair)

    fast = _run(inputs, REF_LAYOUT, variant="fast", route=route)
    _assert_within(fast, ref, "fast vs the FP64 recurrence")
    _assert_within(fast, _run(inputs, REF_LAYOUT, variant="det", route=route), "fast vs det")


def _forward(inputs, seq_lens, *, variant, route) -> dict:
    """The output and the final state of one forward (no backward) with ``variant`` configured."""
    q, k, v = (t.contiguous() for t in _split(inputs["mixed"], inputs["heads"]))
    kwargs = {"cu_seqlens": torch.tensor(_cu(seq_lens), dtype=torch.int32, device="cuda")}
    if route == "host":
        kwargs["cu_seqlens_cpu"] = _cu(seq_lens)
    _set_variant(variant)
    with torch.no_grad():
        o, final_state = ops.chunk_gated_delta_rule(
            q,
            k,
            v,
            g=inputs["g"],
            beta=inputs["beta"],
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            **kwargs,
        )
    torch.cuda.synchronize()
    return {"o": o, "final_state": final_state}


@pytest.mark.parametrize("route", ["host", "device"])
@PAIRS
def test_fast_output_and_final_state_match_det(pair, route):
    inputs = _inputs(MILES_LAYOUT, HEADS[pair], seed=6)

    fast = _forward(inputs, MILES_LAYOUT, variant="fast", route=route)
    det = _forward(inputs, MILES_LAYOUT, variant="det", route=route)
    _assert_within(fast, det, "fast vs det", names=("o", "final_state"))


@PAIRS
def test_fast_on_miles_views_matches_det_and_fla(pair):
    inputs = _inputs(MILES_LAYOUT, HEADS[pair], seed=2)

    fast = _run(inputs, MILES_LAYOUT, variant="fast", views=True)
    # the package makes strided q / k / v contiguous itself
    _assert_within(fast, _run(inputs, MILES_LAYOUT, variant="fast"), "strided views vs contiguous inputs")
    _assert_within(fast, _run(inputs, MILES_LAYOUT, variant="det"), "fast vs det")
    fla = _run(inputs, MILES_LAYOUT, kernel=_fla_kernel(), route="device")
    _assert_within(fast, fla, "fast vs fla", max_abs_rel=2 * MAX_ABS_REL)


@pytest.mark.parametrize("route", ["host", "device"])
@PAIRS
def test_fast_stays_within_the_budgets_run_to_run_and_with_the_recomputed_backward(pair, route):
    inputs = _inputs(MILES_LAYOUT, HEADS[pair], seed=3)

    first = _run(inputs, MILES_LAYOUT, variant="fast", route=route)
    _assert_within(_run(inputs, MILES_LAYOUT, route=route), first, "a second run")
    # no launch of the package raises PyTorch's alert under its deterministic algorithms
    with _deterministic_algorithms():
        _assert_within(_run(inputs, MILES_LAYOUT, route=route), first, "under deterministic algorithms")
    # --cake-gdn-recompute-state: w / u / h / v_new recomputed in the backward, within the budgets of the saved one
    cake.configure(cake_gdn_variant="fast", cake_gdn_recompute_state=True)
    recomputed = _run(inputs, MILES_LAYOUT, route=route)
    with _deterministic_algorithms():
        _assert_within(_run(inputs, MILES_LAYOUT, route=route), recomputed, "recomputed, deterministic algorithms")
    _assert_within(recomputed, first, "the recomputed backward vs the saved one")


def test_a_cp_context_call_runs_det_and_says_so_once(monkeypatch, caplog):
    monkeypatch.setattr(config, "_warned", set())
    calls = []
    for variant, entry_name in (("fast", "cake_gdn_nondet"), ("det", "cake_gdn_det")):
        package = ops._package(variant)

        def counted(*args, _entry=getattr(package, entry_name), _variant=variant, **kwargs):
            calls.append(_variant)
            return _entry(*args, **kwargs)

        monkeypatch.setattr(package, entry_name, counted)
    inputs = _inputs(REF_LAYOUT, HEADS["4x12"], seed=4)
    # an FLA context without a process group: the det kernels with FLA's context parallelism run it as one rank
    cu = _cu(REF_LAYOUT)
    no_group = types.SimpleNamespace(
        group=None, cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="cuda"), cu_seqlens_cpu=cu
    )

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        cake.configure(cake_gdn_variant="fast")
        fast = _run(inputs, REF_LAYOUT, route="device")
        assert calls == ["fast"]
        routed = [_run(inputs, REF_LAYOUT, route="device", cp_context=no_group) for _ in range(2)]
        cake.configure(cake_gdn_variant="det")
        det_cp = _run(inputs, REF_LAYOUT, route="device", cp_context=no_group)
        det = _run(inputs, REF_LAYOUT, route="device")
    assert calls == ["fast", "det"], "a cp_context call reached a vendored package"

    for run in routed:
        _assert_same_bits(run, det_cp, "a cp_context call under fast vs the same call under det")
    _assert_within(fast, det, "fast vs det")
    _assert_within(det_cp, det, "the FLA-CP det kernels vs the deterministic package")
    assert [record.getMessage() for record in caplog.records if record.name == "miles_ext_kernels"] == [RUNS_DET]
