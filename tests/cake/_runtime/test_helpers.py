"""The shared accuracy and bitwise helpers of the op tests, on CPU tensors (needs torch)."""

import math

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake._runtime import loader

from ..helpers import assert_bitwise, assert_rel_rms, assert_repeatable, check_prebuild, rel_rms
from . import synthetic

torch = pytest.importorskip("torch")


def _pair(error: float, n: int = 4096):
    generator = torch.Generator().manual_seed(0)
    ref = torch.randn(n, generator=generator, dtype=torch.float64)
    noise = torch.randn(n, generator=generator, dtype=torch.float64)
    return (ref + error * noise / noise.square().mean().sqrt() * ref.square().mean().sqrt()).float(), ref


def test_rel_rms_is_relative_to_the_reference_and_falls_back_to_max_abs():
    got, ref = _pair(1e-3)

    assert math.isclose(rel_rms(got, ref), 1e-3, rel_tol=1e-3)
    assert rel_rms(torch.tensor([0.0, -2.0]), torch.zeros(2)) == 2.0
    with pytest.raises(AssertionError, match="shape"):
        rel_rms(torch.zeros(2), torch.zeros(3))


def test_a_tight_budget_passes_and_reports_the_measured_errors():
    out, ref = _pair(1e-3)

    measured = assert_rel_rms({"out": out}, {"out": ref}, {"out": 2e-3})
    assert math.isclose(measured["out"], 1e-3, rel_tol=1e-3)


def test_a_budget_that_lets_a_two_percent_error_through_is_too_loose():
    out, ref = _pair(1e-3)

    with pytest.raises(AssertionError, match="out: gate too loose: x1.02"):
        assert_rel_rms({"out": out}, {"out": ref}, {"out": 3e-2})
    assert_rel_rms({"out": out}, {"out": ref}, {"out": 3e-2}, loose={"out": "bf16 accumulation, about 1 %"})


def test_every_failure_is_reported_at_once():
    out, ref = _pair(1e-2)
    bad = out.clone()
    bad[0] = float("nan")

    with pytest.raises(AssertionError) as error:
        assert_rel_rms({"out": out, "dx": bad}, {"out": ref, "dx": ref}, {"out": 1e-3, "dx": 1e-3})
    assert "dx: not finite" in str(error.value) and "out: rel_rms 1.000e-02 > budget 1.000e-03" in str(error.value)


@pytest.mark.parametrize(
    ("budget", "loose", "error"),
    [
        ({"out": 1e-2}, {}, r"budget names \['out'\], the outputs are \['dx', 'out'\]"),
        ({"out": 1e-2, "dx": 1e-2}, {"dw": "x"}, r"loose names \['dw'\]"),
        ({"out": 1e-2, "dx": 1e-2}, {"dx": ""}, r"loose gives no reason for \['dx'\]"),
    ],
)
def test_the_budget_names_every_output(budget, loose, error):
    out, ref = _pair(1e-3)

    with pytest.raises(AssertionError, match=error):
        assert_rel_rms({"out": out, "dx": out}, {"out": ref, "dx": ref}, budget, loose=loose)


def test_bitwise_compares_bits_including_nan_payloads():
    a = torch.tensor([1.0, float("nan"), 0.0])
    b = a.clone()
    assert_bitwise(a, b)
    assert_bitwise({"x": a}, {"x": b})

    b.view(torch.int32)[1] += 1  # another NaN payload
    with pytest.raises(AssertionError, match=r"tensor: 1 of 3 elements differ, first at \(1,\)"):
        assert_bitwise(a, b)
    with pytest.raises(AssertionError, match="x: 1 of 3 elements differ"):
        assert_bitwise({"x": a}, {"x": b})
    with pytest.raises(AssertionError, match="1 of 1 elements differ"):
        assert_bitwise(torch.tensor([0.0]), torch.tensor([-0.0]))
    with pytest.raises(AssertionError, match="torch.float32 vs"):
        assert_bitwise(a, a.double())
    with pytest.raises(AssertionError, match="same tensors"):
        assert_bitwise({"x": a}, {"y": a})


def test_repeatable_runs_twice_and_compares_bits():
    runs = iter([torch.ones(2), torch.ones(2), torch.ones(2) * 2, torch.ones(2)])
    first = assert_repeatable(lambda: {"y": next(runs)})

    assert torch.equal(first["y"], torch.ones(2))
    with pytest.raises(AssertionError, match="y: 2 of 2 elements differ"):
        assert_repeatable(lambda: {"y": next(runs)})


def test_check_prebuild_prebuilds_once_then_runs_every_route_with_every_variant_without_building(synth):
    seen = []

    def route(variant):
        seen.append((variant, cake.variant("synth"), loader._NO_BUILD > 0))

    check_prebuild("synth", {"plain": route, "packed": route}, width=8)

    assert seen == [("det", "det", True)] * 2 + [("fast", "fast", True)] * 2
    assert synthetic.CALLS == [("prebuild", {"device": None, "max_workers": None, "width": 8})]


def test_check_prebuild_names_the_route_that_would_build(synth):
    def compiles(variant):
        raise RuntimeError("prog_sm_100a would compile inside a call")

    with pytest.raises(AssertionError, match="synth det, route long: prog_sm_100a would compile inside a call"):
        check_prebuild("synth", {"long": compiles})
    with pytest.raises(AssertionError, match="synth: no routes"):
        check_prebuild("synth", {})
