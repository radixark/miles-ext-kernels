"""The prebuild contract of ``mhc``: after ``prebuild("mhc", hidden=H, sinkhorn_iters=N)``, no call at ``H`` and ``N``
compiles, whatever the variant, the token count or the call's other switches.

``ROUTES`` covers every call-time switch at hidden 4096 and the Sinkhorn count 7 (not the count of record, 20: the
coefficient and token-owner passes then run programs of their own, which compiled inside the call when prebuild was
given the default count): a token-count sweep (0 to 65537, across the bands of the host rules), other epsilons,
frozen parameters or input, a forward without autograd, checkpointed connections (the statistics replay), BF16
parameters, a side stream, and the head (each variant's own package). The checks build into library caches of their
own, empty at the start of the module, so a library an earlier test built cannot hide one that prebuild leaves out.
"""

import functools
import sys

import pytest

torch = pytest.importorskip("torch")

from torch.utils.checkpoint import checkpoint  # noqa: E402

from miles_ext_kernels import cake  # noqa: E402

from ..helpers import check_prebuild, no_build, requires_arch  # noqa: E402

pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")

HIDDEN = 4096
SINKHORN_ITERS = 7
TOKENS = (0, 1, 257, 1025, 2049, 4097, 8193, 14928, 16263, 32769, 65537)


def _ops():
    from miles_ext_kernels.cake._impl.mhc import ops

    return ops


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _connection(
    variant: str,
    *,
    hidden: int,
    sinkhorn_iters: int,
    T: int = 257,
    params_dtype: torch.dtype = torch.float32,
    grad_X: bool = True,
    grad_params: bool = True,
    grad: bool = True,
    checkpointed: str | None = None,
    **epsilons,
) -> None:
    """One connection forward and backward (``grad=False``: forward only, without autograd)."""
    from . import _connection as connection

    ops = _ops()
    i = connection.make_inputs(T, hidden, _device())
    X = i["X"].requires_grad_(grad_X)
    fn, scale, base = (i[k].to(params_dtype).requires_grad_(grad_params) for k in ("fn", "scale", "base"))
    B = i["B"].requires_grad_(True)

    def prepare(X, fn, scale, base):
        return ops.mhc_prepare(X, fn, scale, base, sinkhorn_iters=sinkhorn_iters, **epsilons)

    def branch(U):
        return U * 0.5 + B  # stands for the layer's branch

    def layer(X, fn, scale, base):
        U, handle = prepare(X, fn, scale, base)
        return U, ops.mhc_finish(X, branch(U), handle)

    with torch.set_grad_enabled(grad):
        if checkpointed == "connection":  # the whole connection's forward replays for the backward
            U, Y = checkpoint(layer, X, fn, scale, base, use_reentrant=False)
        elif checkpointed == "finish":  # the finish replays the forward statistics
            U, handle = prepare(X, fn, scale, base)
            Y = checkpoint(ops.mhc_finish, X, branch(U), handle, use_reentrant=False)
        else:
            U, Y = layer(X, fn, scale, base)
    assert U.dtype == Y.dtype == torch.bfloat16 and tuple(Y.shape) == (T, 4, hidden)
    if grad:
        torch.autograd.backward([U, Y], [i["dU"], i["dY"]])
    torch.cuda.synchronize()


def _on_side_stream(variant: str, **model) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _connection(variant, T=2049, **model)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


def _head(variant: str, *, hidden: int, T: int = 257) -> None:
    X = torch.randn(T, 4, hidden, device=_device()).to(torch.bfloat16).requires_grad_(True)
    Y = _ops().mhc_head(X)
    expected = {"det": "CakeMHCHeadDetFunction", "fast": "CakeMHCHeadNondetFunction"}[variant]
    assert Y.grad_fn._forward_cls.__name__ == expected  # the variant's own head package
    Y.backward(torch.ones_like(Y))
    torch.cuda.synchronize()


def routes(hidden: int, sinkhorn_iters: int) -> dict:
    """Every call-time switch of an ``mhc`` call at the model shape ``hidden`` / ``sinkhorn_iters``."""
    connection = functools.partial(_connection, hidden=hidden, sinkhorn_iters=sinkhorn_iters)
    return {
        **{f"T={T}": functools.partial(connection, T=T) for T in TOKENS},
        "other epsilons": functools.partial(connection, rms_eps=1e-6, hc_eps=1e-5),
        "frozen parameters": functools.partial(connection, T=2049, grad_params=False),
        "frozen input": functools.partial(connection, T=2049, grad_X=False),
        "no autograd": functools.partial(connection, T=2049, grad=False),
        "checkpointed connection": functools.partial(connection, T=2049, checkpointed="connection"),
        "checkpointed finish": functools.partial(connection, T=2049, checkpointed="finish"),
        "bf16 parameters": functools.partial(connection, T=2049, params_dtype=torch.bfloat16),
        "side stream": functools.partial(_on_side_stream, hidden=hidden, sinkhorn_iters=sinkhorn_iters),
        "head": functools.partial(_head, hidden=hidden),
        "head T=0": functools.partial(_head, hidden=hidden, T=0),
    }


ROUTES = routes(HIDDEN, SINKHORN_ITERS)


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    """Two library caches, empty when the module starts: one for the count 7, one for the count of record."""
    return {count: tmp_path_factory.mktemp(f"tvm-ffi-sinkhorn{count}") for count in (SINKHORN_ITERS, 20)}


def _use(caches, monkeypatch, count: int) -> None:
    """Build into and load from the cache of ``count``; the loaded libraries and plans of earlier tests are dropped
    (the runtime's per-test cleanup drops them again after the test)."""
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(caches[count]))
    cake.release("mhc")


def test_prebuild_covers_every_route_at_the_models_sinkhorn_count(caches, monkeypatch):
    """Review: a call at 7 Sinkhorn iterations compiled two libraries inside the call."""
    _use(caches, monkeypatch, SINKHORN_ITERS)
    check_prebuild("mhc", ROUTES, hidden=HIDDEN, sinkhorn_iters=SINKHORN_ITERS)


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
def test_prebuild_covers_every_route_at_the_count_of_record(caches, monkeypatch, hidden):
    if hidden not in _ops().supported_hidden_sizes(_device()):
        pytest.skip(f"hidden {hidden} is not served on this device")
    _use(caches, monkeypatch, 20)
    check_prebuild("mhc", routes(hidden, 20), hidden=hidden, sinkhorn_iters=20)


def test_a_count_prebuild_was_not_given_would_compile_and_no_build_says_so(caches, monkeypatch):
    """What the review saw: a prebuild at the count of record leaves the other count's programs out."""
    _use(caches, monkeypatch, 20)
    cake.prebuild("mhc", hidden=HIDDEN, sinkhorn_iters=20)
    cake.configure(cake_mhc_variant="det")
    with no_build():
        _connection("det", hidden=HIDDEN, sinkhorn_iters=20)
        with pytest.raises(RuntimeError, match=r"would compile inside a call: prebuild\('mhc', \.\.\.\) first"):
            _connection("det", hidden=HIDDEN, sinkhorn_iters=SINKHORN_ITERS)


def test_the_variant_configured_after_a_call_runs_without_compiling(caches, monkeypatch):
    """configure(det), one call, configure(fast): the next call runs fast, built by the same prebuild."""
    _use(caches, monkeypatch, SINKHORN_ITERS)
    cake.prebuild("mhc", hidden=HIDDEN, sinkhorn_iters=SINKHORN_ITERS)
    ops = _ops()
    cake.configure(cake_mhc_variant="det")
    with no_build():
        _connection("det", hidden=HIDDEN, sinkhorn_iters=SINKHORN_ITERS)
        cake.configure(cake_mhc_variant="fast")
        X = torch.zeros(1, 4, HIDDEN, dtype=torch.bfloat16, device=_device())
        fn = torch.zeros(24, 4 * HIDDEN, device=_device())
        U, handle = ops.mhc_prepare(X, fn, torch.ones(3, device=_device()), torch.zeros(24, device=_device()))
        assert isinstance(handle, ops._fast._MHCTrainHandle)


def test_the_routes_import_no_flashinfer():
    assert not [name for name in sys.modules if name.split(".")[0] == "flashinfer"]
