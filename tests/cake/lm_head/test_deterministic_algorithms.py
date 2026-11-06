"""The alert of a ``fast`` backward under ``torch.use_deterministic_algorithms(True)``, without a GPU (CPU torch): the
hook ``log_probs`` registers on the ``logp`` of a ``fast`` call, on a CPU autograd graph whose node below the hook
stands for the package's backward. The hook reads torch's flags when the backward runs: a ``RuntimeError`` before the
node below it runs, a ``UserWarning`` under ``warn_only=True`` (the backward then runs), nothing otherwise.
``test_lm_head_gpu.py`` checks both variants' calls on the device.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels.cake._impl.lm_head import ops  # noqa: E402

ALERT = r"does not have a deterministic implementation, but you set 'torch\.use_deterministic_algorithms"
DET = "--cake-lm-head-variant det is the deterministic variant"


@pytest.fixture
def deterministic_algorithms():
    """Set torch's deterministic algorithms for the test (``use(mode, warn_only=False)``); restored afterwards."""
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    yield lambda mode, warn_only=False: torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def _graph():
    """``x -> below -> logp`` with the hook on ``logp``; the list records each backward of ``below``."""
    ran = []

    class Below(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2

        @staticmethod
        def backward(ctx, grad):
            ran.append(True)
            return grad * 2

    x = torch.ones(4, requires_grad=True)
    logp = Below.apply(x)
    logp.register_hook(ops._alert_nondeterministic_backward)
    return x, logp, ran


def test_no_alert_outside_torch_deterministic_algorithms(deterministic_algorithms):
    deterministic_algorithms(False)
    x, logp, ran = _graph()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        logp.sum().backward()

    assert ran == [True] and torch.equal(x.grad, torch.full((4,), 2.0))


def test_the_backward_raises_the_alert_before_the_package_backward_runs(deterministic_algorithms):
    x, logp, ran = _graph()
    # enabled after the forward: the flag is read when the backward runs
    deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match=rf"{ALERT}\(True\)'; {DET}, or use the 'warn_only=True' option"):
        logp.sum().backward()

    assert ran == [] and x.grad is None


def test_warn_only_warns_and_runs_the_backward(deterministic_algorithms):
    x, logp, ran = _graph()
    deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match=rf"{ALERT}\(True, warn_only=True\)'; {DET}"):
        logp.sum().backward()

    assert ran == [True] and torch.equal(x.grad, torch.full((4,), 2.0))


def test_a_backward_after_torch_deterministic_algorithms_ends_does_not_alert(deterministic_algorithms):
    deterministic_algorithms(True)
    x, logp, ran = _graph()
    deterministic_algorithms(False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        logp.sum().backward()

    assert ran == [True]
