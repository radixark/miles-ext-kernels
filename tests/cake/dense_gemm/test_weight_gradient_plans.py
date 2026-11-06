"""The weight gradient of a single-head ``out_in`` projection runs through the vendored packages' weight-gradient entry
points (CPU torch; the plans also need tvm_ffi, no device).

The deterministic package's ``cake_dense_gemm_det_train_projection_wgrad`` runs the GEMM in the orientation its
``wgrad_views`` gives the row (the transposed-store ``X.T @ G`` for a narrow ``D_out``); the non-deterministic
package's ``cake_dense_gemm_nondet_train_projection_wgrad`` plans its split-K program on ``X.T @ G`` and runs the
deterministic entry point's program on a row its planner does not take. The calls of ``batched_projection`` are
recorded on CPU tensors: only a single-head ``out_in`` weight gradient calls the weight-gradient entry point, on the
``[T, D]`` matrices; every other GEMM keeps its K1 call. On ``meta`` views, for every part and a sweep of token counts,
the deterministic plan of the views the backward hands the entry point is the package's own plan of its ``[T, D]``
matrices, and ``fast_projection_programs`` reports the program the non-deterministic entry point plans.
"""

import types

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.dense_gemm import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config  # noqa: E402
from miles_ext_kernels.cake.dense_gemm import batched_projection, fast_projection_programs  # noqa: E402

PARTS = (
    ("sm_100a", 148),
    ("sm_100a", 152),
    ("sm_103a", 148),
    ("sm_103a", 152),
    ("sm_103a", 160),
    ("sm_107a", 208),
    ("sm_107a", 212),
    ("sm_107a", 216),
)
TOKENS = (1, 7, 33, 64, 65, 128, 192, 1001, 1024, 2048, 4096, 4097, 8192, 16384, 40961, 47105, 49153, 65536, 65537)
# the L2 size enters the plan only through the TMA eviction hints; one size per part is enough to plan with
L2_BYTES = 132120576
# name -> (d_in, d_out): the single-head out_in rows of the fast tests and DeepSeek-V4's output-group projection at one
# local group
SINGLE_HEAD = {
    "dense-4096-32": (4096, 32),
    "dense-5120-48": (5120, 48),
    "dense-5120-64": (5120, 64),
    "dense-6144-128": (6144, 128),
    "dense-8192-128": (8192, 128),
    "dsv4-o-group-g1": (4096, 1024),
}
# the rows whose deterministic weight gradient runs as the transposed-store X.T @ G on every part (wgrad_views);
# 4096 -> 1024 keeps G.T @ X
SWAPPED = {"dense-4096-32", "dense-5120-48", "dense-5120-64", "dense-6144-128", "dense-8192-128"}
# name -> (heads, d_in, d_out, weight layout) of the recorded calls
CPU_CASES = {
    "one-head-out_in": (1, 64, 32, "out_in"),
    "two-heads-out_in": (2, 64, 32, "out_in"),
    "one-head-in_out": (1, 64, 32, "in_out"),
    "two-heads-in_out": (2, 64, 32, "in_out"),
}
SLOT = 16  # the input is the first d_in features of a (d_in + SLOT)-wide head


@pytest.fixture
def calls(monkeypatch):
    """The calls of the K1 and weight-gradient entry points as ``(entry, variant, *tensors)``, in call order."""
    recorded = []
    monkeypatch.setattr(ops, "_arch", lambda device=None: "sm_100a")
    monkeypatch.setattr(ops, "_k1_gemm", lambda variant: lambda *views: recorded.append(("gemm", variant, *views)))
    monkeypatch.setattr(ops, "_k1_wgrad", lambda variant: lambda *views: recorded.append(("wgrad", variant, *views)))
    # the backward makes the device's primary context current through its stream: nothing to do on CPU tensors
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: types.SimpleNamespace(query=lambda: True))
    monkeypatch.setattr(config, "_warned", set())
    return recorded


def _project(name: str):
    """One forward and backward of ``batched_projection`` on CPU: ``(act, grad)`` (``x`` is a slice of ``act``)."""
    heads, d_in, d_out, layout = CPU_CASES[name]
    tokens = 24
    act = torch.zeros(tokens, heads, d_in + SLOT, dtype=torch.bfloat16, requires_grad=True)
    shape = (heads, d_in, d_out) if layout == "in_out" else (heads, d_out, d_in)
    weight = torch.zeros(*shape, dtype=torch.bfloat16, requires_grad=True)
    out = batched_projection(act[..., :d_in], weight, weight_layout=layout)
    grad = torch.zeros(tokens, heads, d_out, dtype=torch.bfloat16)
    out.backward(grad)
    return act, grad


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("name", list(CPU_CASES))
def test_only_a_single_head_out_in_weight_gradient_calls_the_weight_gradient_entry_point(calls, variant, name):
    cake.configure(cake_dense_gemm_variant=variant)
    act, grad = _project(name)

    heads, d_in, d_out, layout = CPU_CASES[name]
    single = heads == 1 and layout == "out_in"
    assert [call[:2] for call in calls] == [("gemm", variant)] * 2 + [("wgrad" if single else "gemm", variant)]
    if single:
        _, _, g, x, dw = calls[-1]
        # the [T, D] matrices of the gradient and of the input slice, read in place, and the [D_out, D_in] gradient
        assert (g.shape, g.stride(), g.data_ptr()) == (grad.shape[::2], (d_out, 1), grad.data_ptr())
        assert (x.shape, x.stride(), x.data_ptr()) == ((act.shape[0], d_in), (d_in + SLOT, 1), act.data_ptr())
        assert (dw.shape, dw.stride()) == ((d_out, d_in), (d_in, 1))


def test_fast_runs_the_weight_gradient_entry_point_of_det_under_torch_deterministic_algorithms(calls):
    cake.configure(cake_dense_gemm_variant="fast")
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        _project("one-head-out_in")
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])

    assert [call[:2] for call in calls] == [("gemm", "det"), ("gemm", "det"), ("wgrad", "det")]


# ---------------------------------------------------------------------------------------------------------------
# the plans (meta views)


@pytest.fixture(scope="module")
def backends():
    """``(cake_backend of det, cake_backend of fast)``."""
    pytest.importorskip("tvm_ffi")
    return ops._vendored()[0], ops._vendored_fast()[0]


def _meta(*shape: int) -> torch.Tensor:
    return torch.empty(*shape, dtype=torch.bfloat16, device="meta")


def _wgrad_views(d_in: int, d_out: int, tokens: int):
    """``(G, X, dw)``: the views the backward hands the weight-gradient entry point (``meta``)."""
    x, g, weight = ops._projection_views(1, d_in, d_out, tokens, False)
    recorded = []

    def k1(*views):
        raise AssertionError("a single-head out_in weight gradient made a K1 call")

    ops._wgrad(k1, lambda *views: recorded.append(views), x, g, weight, False)
    (views,) = recorded
    return views


def _det_plan(det, g, x, dw, arch: str, sm_count: int):
    """The deterministic weight-gradient entry point's plan of ``(G, X, dw)``: its orientation and its planner."""
    a, b, transposed = det.wgrad_views(g, x, arch=arch, out_f32=dw.dtype == torch.float32)
    plan, *_ = det.cake_dense_gemm_det_train_projection_plan(
        a, b, dw, sm_count=sm_count, l2_bytes=L2_BYTES, transposed_out=transposed, arch=arch
    )
    return plan


def test_the_weight_gradient_entry_points_are_the_packages(backends):
    det, fast = backends

    assert ops._k1_wgrad("det") is det.cake_dense_gemm_det_train_projection_wgrad
    assert ops._k1_wgrad("fast") is fast.cake_dense_gemm_nondet_train_projection_wgrad


def test_det_plans_a_single_head_weight_gradient_as_the_package_plans_its_own_call(backends):
    det, _ = backends
    orientations = set()
    for arch, sm_count in PARTS:
        for name, (d_in, d_out) in SINGLE_HEAD.items():
            for tokens in TOKENS:
                g, x, dw = _wgrad_views(d_in, d_out, tokens)
                assert (g.shape, x.shape, dw.shape) == ((tokens, d_out), (tokens, d_in), (d_out, d_in))
                launched = _det_plan(det, g, x, dw, arch, sm_count)
                # the package's own call: contiguous [T, D] matrices and a [D_out, D_in] gradient
                own = _det_plan(det, _meta(tokens, d_out), _meta(tokens, d_in), _meta(d_out, d_in), arch, sm_count)

                assert launched == own, (arch, sm_count, name, tokens)
                assert launched.transposed_out == (name in SWAPPED), (arch, sm_count, name, tokens)
                orientations.add(launched.transposed_out)
    assert orientations == {False, True}


def test_fast_reports_the_program_of_the_nondet_weight_gradient_entry_point(backends):
    _, fast = backends
    own = set()
    for arch, sm_count in PARTS:
        for name, (d_in, d_out) in SINGLE_HEAD.items():
            for tokens in TOKENS:
                g, x, dw = _meta(tokens, d_out), _meta(tokens, d_in), _meta(d_out, d_in)
                plan, *_ = fast.plan_atomic_gemm(x.t(), g, dw, transposed_out=True, arch=arch, sm_count=sm_count)
                want = None if plan is None else plan.template
                reported = fast_projection_programs(
                    1, d_in, d_out, weight_layout="out_in", tokens=tokens, arch=arch, sm_count=sm_count
                )

                case = (arch, sm_count, name, tokens)
                assert reported == {"forward": None, "input_grad": None, "weight_grad": want}, case
                own.add(want)
    # some rows run the package's split-K program and some its deterministic sibling
    assert None in own and len(own) > 1, own
