"""The ``fast`` variant of the dense projection GEMMs on the device (configured with
``cake.configure(cake_dense_gemm_variant="fast")``), against ``det`` and FP64.

For every case the programs the non-deterministic package launches are the ones ``fast_projection_programs`` /
``fast_router_programs`` report for the device's architecture and SM count. The forward is bitwise ``det``'s, and so
is each gradient where ``fast`` runs ``det``'s plan of the deterministic program. Where it runs a program of its own (a
weight gradient, or a router input gradient on a short-token SIMT program), or the router weight gradient on the
deterministic program at another split count than ``det``'s, that gradient is within the deterministic package's
accuracy budget against FP64 (BF16 output: atol = rtol = 1e-2 and a relative RMS of 3e-3; the router's FP32 gradients:
relative Frobenius error 2e-6). The cases are miles' projections and routers and the single-head projections and
256-expert routers whose gradients the package's tables give a program of its own on some part, at token counts inside
those tables. The rows where ``fast``'s two-split weight gradient missed the BF16 budget with the package vendored
before (148-SM sm_103a parts, the op module's historical accuracy misses) now run the deterministic program, with
``det``'s bits. The single-head rows whose ``det`` weight gradient runs the deterministic package's weight-gradient
entry point (the transposed-store ``X.T @ G``) are held to the relative RMS budget against FP64, ``det`` and ``fast``
where it runs the deterministic program, also at a token count past 49152. Then ``fast`` running ``det`` under
``torch.use_deterministic_algorithms``. The prebuild of ``fast`` is checked in ``test_prebuild_gpu.py``, ``det``
repeating bit for bit in ``test_determinism_gpu.py``. Needs a CUDA device; the tests need SM100a / SM103a / SM107a and
skip elsewhere.
"""

import logging
import warnings

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.dense_gemm import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config  # noqa: E402
from miles_ext_kernels.cake.dense_gemm import (  # noqa: E402
    batched_projection,
    fast_projection_programs,
    fast_router_programs,
    router_gating,
)

from ..helpers import assert_rel_rms  # noqa: E402

if torch.cuda.get_device_capability() not in cake.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake dense projection GEMMs need SM100a / SM103a / SM107a", allow_module_level=True)

ARCH = cake.ARCH_BY_CAPABILITY[torch.cuda.get_device_capability()]
SM_COUNT = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
SEED = 20261105
BF16_ATOL = BF16_RTOL = 1e-2
# the relative RMS budget of a BF16 result (one rounding of an FP32 sum: about 1.1e-3 to 1.6e-3; twice the upper end)
BF16_REL_RMS = 3e-3
ROUTER_REL_FRO_MAX = 2e-6
# the relative Frobenius error of an FP32 result rounded once to BF16 (half a BF16 spacing is 2**-9 of the value)
BF16_ROUNDED_REL_FRO_MAX = 2.0**-8

# name -> (heads, d_in, d_out, weight layout, tokens)
PROJECTIONS = {
    "glm5-q-absorb-h16-t1001": (16, 192, 512, "in_out", 1001),
    "glm5-value-h16-t2049": (16, 512, 256, "out_in", 2049),
    "dsv4-o-group-g1-t1001": (1, 4096, 1024, "out_in", 1001),
    "dsv4-o-group-g8-t1001": (8, 4096, 1024, "out_in", 1001),
    **{
        f"dense-{d_in}-{d_out}-t{t}": (1, d_in, d_out, "out_in", t)
        for d_in, d_out in ((4096, 32), (5120, 64), (6144, 128), (8192, 128))
        for t in (1001, 16384)
    },
}
# name -> (experts, hidden, tokens)
ROUTERS = {
    "router-e384-h7168-t1001": (384, 7168, 1001),
    **{f"router-e256-h{h}-t{t}": (256, h, t) for h in (4096, 5120, 6144) for t in (128, 1001, 16384, 20480)},
}


def _fast_backend():
    return ops._vendored_fast()[0]


def _launched_own_programs() -> set[str]:
    """The templates of the non-deterministic package's own programs its convenience path planned since its caches
    were last cleared (an entry without a plan ran the deterministic program)."""
    fast = _fast_backend()
    return {
        entry.plan.template
        for cache in (fast._K1_ATOMIC_CACHE, fast._K2_ATOMIC_CACHE)
        for entry in cache.values()
        if entry is not None and entry.plan is not None
    }


def _clear_caches() -> None:
    # this package's caches and the deterministic package's
    _fast_backend().cake_dense_gemm_nondet_train_convenience_cache_clear()


# ---------------------------------------------------------------------------------------------------------------
# projections


def _projection_leaves(heads, d_in, d_out, layout, tokens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    shape = (heads, d_in, d_out) if layout == "in_out" else (heads, d_out, d_in)
    return randn(tokens, heads, d_in), randn(*shape, scale=d_in**-0.5), randn(tokens, heads, d_out)


def _projection_step(x, weight, grad, layout, variant):
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    if variant == "fp64":
        out = torch.einsum("thd,hdm->thm" if layout == "in_out" else "thm,hdm->thd", x, weight)
    else:
        cake.configure(cake_dense_gemm_variant=variant)
        out = batched_projection(x, weight, weight_layout=layout)
    out.backward(grad)
    return out.detach(), x.grad, weight.grad


def _assert_bf16_close(actual: torch.Tensor, expected_f64: torch.Tensor, what: str):
    assert actual.dtype == torch.bfloat16 and bool(torch.isfinite(actual).all()), what
    diff = (actual.double() - expected_f64).abs()
    outside = int((diff > BF16_ATOL + BF16_RTOL * expected_f64.abs()).sum())
    assert outside == 0, f"{what}: {outside} elements outside atol = rtol = 1e-2 (max abs {float(diff.max()):.3g})"


@pytest.mark.parametrize("name", list(PROJECTIONS))
def test_projection_fast_runs_the_planned_programs_with_det_bits_or_budget(name):
    heads, d_in, d_out, layout, tokens = PROJECTIONS[name]
    planned = fast_projection_programs(
        heads, d_in, d_out, weight_layout=layout, tokens=tokens, arch=ARCH, sm_count=SM_COUNT
    )
    x, weight, grad = _projection_leaves(heads, d_in, d_out, layout, tokens, seed=SEED + d_in + d_out + tokens)

    det = _projection_step(x, weight, grad, layout, "det")
    _clear_caches()
    fast = _projection_step(x, weight, grad, layout, "fast")
    torch.cuda.synchronize()

    assert planned["forward"] is None and planned["input_grad"] is None
    assert _launched_own_programs() == ({planned["weight_grad"]} - {None}), planned
    assert torch.equal(fast[0], det[0]), "output differs from det"
    assert torch.equal(fast[1], det[1]), "input gradient differs from det"
    if planned["weight_grad"] is None:
        assert torch.equal(fast[2], det[2]), "weight gradient differs from det on the deterministic program"
    else:
        want = _projection_step(x.double(), weight.double(), grad.double(), layout, "fp64")
        _assert_bf16_close(fast[2], want[2], f"weight gradient ({planned['weight_grad']})")
        assert_rel_rms({"weight gradient": fast[2]}, {"weight gradient": want[2]}, {"weight gradient": BF16_REL_RMS})


# The op module's "Historical accuracy misses" of fast on 148-SM sm_103a parts: name -> (heads, d_in, d_out, weight
# layout, tokens), at token counts where fast's two-split weight gradient missed the BF16 budget for one of 10 seeds
# with the package vendored before; the vendored package's tables run the deterministic program there now
FORMER_MISSES = {
    "dense-5120-48-t40961": (1, 5120, 48, "out_in", 40961),
    "dense-5120-48-t40962": (1, 5120, 48, "out_in", 40962),
    "dense-5120-64-t47105": (1, 5120, 64, "out_in", 47105),
}


@pytest.mark.parametrize("name", list(FORMER_MISSES))
def test_the_former_accuracy_misses_of_fast_run_det_bits(name):
    if (ARCH, SM_COUNT) != ("sm_103a", 148):
        pytest.skip("the former accuracy misses of fast are on 148-SM sm_103a parts")
    heads, d_in, d_out, layout, tokens = FORMER_MISSES[name]
    planned = fast_projection_programs(
        heads, d_in, d_out, weight_layout=layout, tokens=tokens, arch=ARCH, sm_count=SM_COUNT
    )
    assert set(planned.values()) == {None}, planned
    x, weight, grad = _projection_leaves(heads, d_in, d_out, layout, tokens, seed=SEED)
    det = _projection_step(x, weight, grad, layout, "det")
    _clear_caches()
    fast = _projection_step(x, weight, grad, layout, "fast")
    torch.cuda.synchronize()

    assert not _launched_own_programs()
    for got, want in zip(fast, det):
        assert torch.equal(got, want)


# The single-head out_in rows whose det weight gradient runs the transposed-store X.T @ G of the deterministic
# package's weight-gradient entry point, besides 4096 -> 32 (test_determinism_gpu.py): name -> (d_in, d_out, tokens),
# at the token counts of PROJECTIONS and at 65537; held to the relative RMS budget against FP64 (no elementwise check)
SINGLE_HEAD_WEIGHT_GRADIENTS = {
    f"dense-{d_in}-{d_out}-t{t}": (d_in, d_out, t)
    for d_in, d_out in ((5120, 48), (5120, 64), (6144, 128), (8192, 128))
    for t in (1001, 16384, 65537)
}


def _weight_gradient(x, weight, grad, variant):
    """The weight gradient of one ``out_in`` projection under ``variant`` (the input takes no gradient)."""
    weight = weight.detach().clone().requires_grad_(True)
    cake.configure(cake_dense_gemm_variant=variant)
    batched_projection(x, weight, weight_layout="out_in").backward(grad)
    return weight.grad


@pytest.mark.parametrize("name", list(SINGLE_HEAD_WEIGHT_GRADIENTS))
def test_single_head_weight_gradient_of_det_and_of_fast_on_det_is_within_budget(name, record_property):
    d_in, d_out, tokens = SINGLE_HEAD_WEIGHT_GRADIENTS[name]
    planned = fast_projection_programs(
        1, d_in, d_out, weight_layout="out_in", tokens=tokens, arch=ARCH, sm_count=SM_COUNT
    )
    x, weight, grad = _projection_leaves(1, d_in, d_out, "out_in", tokens, seed=SEED + d_in + d_out + tokens)

    got = {"det": _weight_gradient(x, weight, grad, "det")}
    if planned["weight_grad"] is None:
        _clear_caches()
        got["fast"] = _weight_gradient(x, weight, grad, "fast")
        torch.cuda.synchronize()
        assert not _launched_own_programs()
        assert torch.equal(got["fast"], got["det"]), "weight gradient differs from det on the deterministic program"
    want = (grad[:, 0].double().T @ x[:, 0].double()).unsqueeze(0)
    measured = assert_rel_rms(got, dict.fromkeys(got, want), dict.fromkeys(got, BF16_REL_RMS))
    record_property("rel_rms", measured)


# ---------------------------------------------------------------------------------------------------------------
# routers


def _router_leaves(experts, hidden, tokens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, 1, hidden, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(experts, hidden, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    grad = torch.randn(tokens, 1, experts, generator=gen, device="cuda") * 1e-3
    return x, weight, grad


def _router_step(x, weight, grad, variant):
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    cake.configure(cake_dense_gemm_variant=variant)
    logits = router_gating(x, weight)
    logits.backward(grad)
    return logits.detach(), x.grad, weight.grad


def _rel_fro(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.double() - expected).norm() / expected.norm())


@pytest.mark.parametrize("name", list(ROUTERS))
def test_router_fast_runs_the_planned_programs_with_det_bits_or_budget(name):
    experts, hidden, tokens = ROUTERS[name]
    planned = fast_router_programs(experts, hidden, tokens=tokens, arch=ARCH, sm_count=SM_COUNT)
    x, weight, grad = _router_leaves(experts, hidden, tokens, seed=SEED + experts + hidden + tokens)

    det = _router_step(x, weight, grad, "det")
    _clear_caches()
    fast = _router_step(x, weight, grad, "fast")
    torch.cuda.synchronize()

    assert planned["forward"] is None
    assert _launched_own_programs() == ({planned["input_grad"], planned["weight_grad"]} - {None}), planned
    assert torch.equal(fast[0], det[0]), "logits differ from det"
    g, x2 = grad.reshape(-1, experts).double(), x.reshape(-1, hidden).double()
    if planned["input_grad"] is None:
        assert torch.equal(fast[1], det[1]), "input gradient differs from det on the deterministic program"
    else:
        # the package's SIMT input gradient (other bits than det's BF16x3 emulation) within the router budget, and the
        # BF16 gradient router_gating returns within one rounding of it
        want_dx = g @ weight.double()
        dx32 = _fast_backend().cake_dense_gemm_nondet_train_router_dgrad(
            grad.reshape(-1, experts).float(), weight.float()
        )
        assert _rel_fro(dx32, want_dx) <= ROUTER_REL_FRO_MAX, planned["input_grad"]
        dx = fast[1].reshape(-1, hidden)
        assert dx.dtype == torch.bfloat16 and _rel_fro(dx, want_dx) <= BF16_ROUNDED_REL_FRO_MAX, planned["input_grad"]
    det_splits = ops._fast_router_wgrad_splits(experts, hidden, tokens=tokens, arch=ARCH, sm_count=SM_COUNT)
    if planned["weight_grad"] is None and det_splits is None:
        assert torch.equal(fast[2], det[2]), "weight gradient differs from det on the deterministic program"
        return
    program = planned["weight_grad"] or f"the deterministic program at {det_splits} splits"
    want = g.T @ x2
    # the package's FP32 weight gradient (the program router_gating runs) within the router budget, and the BF16
    # gradient router_gating returns within one rounding of it
    dw32 = _fast_backend().cake_dense_gemm_nondet_train_router_wgrad(
        grad.reshape(-1, experts).float(), x.reshape(-1, hidden).float()
    )
    assert _rel_fro(dw32, want) <= ROUTER_REL_FRO_MAX, program
    assert fast[2].dtype == torch.bfloat16 and _rel_fro(fast[2], want) <= BF16_ROUNDED_REL_FRO_MAX
    assert_rel_rms({"weight gradient": fast[2]}, {"weight gradient": want}, {"weight gradient": BF16_REL_RMS})


# ---------------------------------------------------------------------------------------------------------------
# fast under torch.use_deterministic_algorithms: it runs det


@pytest.mark.parametrize("warn_only", [False, True], ids=["alert", "warn_only"])
def test_fast_runs_det_under_torch_deterministic_algorithms(caplog, monkeypatch, warn_only):
    """While torch's deterministic algorithms are enabled (``warn_only=True`` included) a ``fast`` call runs ``det``:
    ``det``'s bits (``fast`` gives these rows other bits on some parts), one log line, no alert of the
    non-deterministic package."""
    monkeypatch.setattr(config, "_warned", set())
    router = _router_leaves(*ROUTERS["router-e256-h5120-t16384"][:2], 16384, SEED)
    heads, d_in, d_out, layout, tokens = PROJECTIONS["dense-4096-32-t16384"]
    projection = _projection_leaves(heads, d_in, d_out, layout, tokens, SEED)
    det = [*_router_step(*router, "det"), *_projection_step(*projection, layout, "det")]
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                fast = [*_router_step(*router, "fast"), *_projection_step(*projection, layout, "fast")]
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])

    for got, want in zip(fast, det):
        assert torch.equal(got, want)
    assert not [str(w.message) for w in seen if "cake_dense_gemm_nondet_train" in str(w.message)]
    assert [record.getMessage() for record in caplog.records if "dense_gemm" in record.getMessage()] == [
        "miles_ext_kernels.cake.dense_gemm: the fast variant does not serve calls under "
        "torch.use_deterministic_algorithms(True); running det"
    ]
