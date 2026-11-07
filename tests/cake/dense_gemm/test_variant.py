"""The variant a call of the dense projection GEMMs runs (CPU torch, no device: the K1 entry point is swapped for a
recorder and the architecture check for a constant). A call runs the configured variant, read when its forward runs,
and its backward runs the variant of its forward; while ``torch.use_deterministic_algorithms(True)`` is in effect
(``warn_only=True`` included) a ``fast`` call runs ``det`` and the runtime logs it once. ``test_fast_gpu.py`` checks
the bits of such a call on the device.
"""

import logging

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.dense_gemm import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config  # noqa: E402
from miles_ext_kernels.cake.dense_gemm import batched_projection, router_gating  # noqa: E402

RUNS_DET = (
    "miles_ext_kernels.cake.dense_gemm: the fast variant does not serve calls under "
    "torch.use_deterministic_algorithms(True); running det"
)


@pytest.fixture
def k1_calls(monkeypatch):
    """The variants of the K1 calls (the forwards of both entry points), in call order."""
    calls = []
    monkeypatch.setattr(ops, "_arch", lambda device=None: "sm_100a")
    monkeypatch.setattr(ops, "_k1_gemm", lambda variant: lambda a, b, out: calls.append(variant))
    monkeypatch.setattr(config, "_warned", set())
    return calls


def _calls() -> list[str]:
    """One call of each entry point; the variant each one's backward runs (its autograd node's)."""
    x = torch.zeros(4, 2, 16, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.zeros(2, 16, 8, dtype=torch.bfloat16, requires_grad=True)
    projected = batched_projection(x, weight, weight_layout="in_out")
    logits = router_gating(torch.zeros(4, 16, dtype=torch.bfloat16, requires_grad=True), weight[0].t())
    return [projected.grad_fn.variant, logits.grad_fn.variant]


def test_a_call_runs_the_configured_variant(k1_calls):
    assert _calls() == ["det", "det"]
    cake.configure(cake_dense_gemm_variant="fast")
    assert _calls() == ["fast", "fast"]

    assert k1_calls == ["det", "det", "fast", "fast"]


@pytest.mark.parametrize("warn_only", [False, True], ids=["alert", "warn_only"])
def test_fast_runs_det_under_torch_deterministic_algorithms_and_logs_it_once(k1_calls, caplog, warn_only):
    cake.configure(cake_dense_gemm_variant="fast")
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            backward = _calls() + _calls()
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])

    assert backward == ["det"] * 4
    assert k1_calls == ["det"] * 4
    assert [record.getMessage() for record in caplog.records] == [RUNS_DET]
    # the setting is read at every call
    assert _calls() == ["fast", "fast"]


def test_a_backward_under_torch_deterministic_algorithms_runs_det_for_a_fast_forward(monkeypatch, caplog):
    monkeypatch.setattr(config, "_warned", set())
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            assert ops._call_variant("fast") == "det"
            assert ops._call_variant("det") == "det"
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])

    assert ops._call_variant("fast") == "fast"
    assert [record.getMessage() for record in caplog.records] == [RUNS_DET]
