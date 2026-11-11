"""gated_rmsnorm's prebuild contract (README, Prebuild): after one ``prebuild``, no call of either variant compiles.

``ROUTES`` runs one forward + backward per call-time path: the gate read in place and copied at every value-head
count (12, 16, 32, 48), a 2-D ``[N, 128]`` input, a token sweep (``_shapes.py``) whose row counts and layouts reach
every forward and BF16 backward module the non-deterministic package selects on sm_100a, sm_103a and sm_107a (its
tiny, short, pipelined, plain, mid, long and extra-long forwards and its short, exact tiny, fused, repeatable and long
repeatable backwards, as each architecture's bands select them; checked here against the device's SM count and in
``test_prebuild_routes.py`` against every shipped part's), a weight without a gradient, and ``fast`` under torch's
deterministic algorithms (which runs ``det``). Needs SM100a / SM103a / SM107a; skips elsewhere.
"""

import sys

import pytest

from ..helpers import check_prebuild, no_build, requires_arch
from ._shapes import CALL_ROLES, SHAPES, packed_rows, selected_modules, unreached_roles
from .inputs import make, reads_in_place, run

OP = "gated_rmsnorm"
pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")


def _shape_route(name: str):
    tokens, key_heads, value_heads, extra, in_place = SHAPES[name]

    def route(variant: str) -> None:
        x, z, w, dy = make(tokens, key_heads, value_heads, extra)
        assert reads_in_place(z) == in_place, f"{name}: the gate is not read {'in place' if in_place else 'copied'}"
        run(x, z, w, dy)

    return route


def _rows_2d(variant: str) -> None:
    x, z, w, dy = make(4099, 8, 16)
    run(x.flatten(0, 1), z.flatten(0, 1), w, dy.flatten(0, 1))


def _weight_without_gradient(variant: str) -> None:
    result = run(*make(1031, 16, 32), weight_grad=False)
    assert result["dw"] is None


def _torch_deterministic_algorithms(variant: str) -> None:
    import torch

    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        run(*make(1031, 16, 32))
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


ROUTES = {
    **{name: _shape_route(name) for name in SHAPES},
    "rows_2d": _rows_2d,
    "weight_without_gradient": _weight_without_gradient,
    "torch_deterministic_algorithms": _torch_deterministic_algorithms,
}


def test_one_prebuild_serves_every_route_of_every_variant():
    check_prebuild(OP, ROUTES)


def test_the_token_sweep_reaches_every_row_count_module():
    """The rows of the routes select every forward and BF16 backward module the non-deterministic package selects on
    this device (its SM count picks the module); the deterministic package has one module per role and layout."""
    import torch

    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops
    from miles_ext_kernels.cake._runtime import loader

    jit = ops._package("fast").cake_jit
    arch = loader.device_arch()
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    missing = unreached_roles(jit, arch, sm_count)

    assert len(selected_modules(jit, arch, sm_count)) > len(CALL_ROLES), f"{arch}: no row-count tier is routed"
    assert missing == [], f"{arch} with {sm_count} SMs: no route reaches {missing}"


def test_prebuild_returns_every_library_and_is_idempotent():
    from miles_ext_kernels import cake
    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops
    from miles_ext_kernels.cake._runtime import loader

    arch = loader.device_arch()
    paths = cake.prebuild(OP)
    modules = sum(len(set(ops._package(variant).cake_jit.ROUTES[arch].values())) for variant in ("det", "fast"))

    assert len(paths) == len(set(paths)) == modules and all(path.is_file() for path in paths)
    assert cake.prebuild(OP) == paths


def test_configure_after_a_call_switches_the_next_call_without_a_build(monkeypatch):
    from miles_ext_kernels import cake
    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops

    cake.prebuild(OP)
    calls = []
    for variant, name in (("det", "cake_rmsnorm_silu_gate_det"), ("fast", "cake_rmsnorm_silu_gate_nondet")):
        package = ops._package(variant)
        original = getattr(package, name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(package, name, record)
    x, z, w, dy = make(1031, 16, 32)
    cake.configure(cake_gated_rmsnorm_variant="det")
    run(x, z, w, dy)
    cake.configure(cake_gated_rmsnorm_variant="fast")
    with no_build():
        run(x, z, w, dy)

    assert calls == ["cake_rmsnorm_silu_gate_det", "cake_rmsnorm_silu_gate_nondet"]


def test_no_flashinfer_module_is_imported():
    from miles_ext_kernels import cake

    cake.prebuild(OP)
    for variant in ("det", "fast"):
        cake.configure(cake_gated_rmsnorm_variant=variant)
        run(*make(1031, 16, 32))

    assert [name for name in sys.modules if name == "flashinfer" or name.startswith("flashinfer.")] == []


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_the_routes_read_the_gate_and_the_rows_as_declared(monkeypatch, name):
    """Each route reads its gate in place or copied as ``SHAPES`` declares, and the non-deterministic package selects
    its forward and backward modules for the packed or the strided rows ``_shapes.packed_rows`` assumes."""
    from miles_ext_kernels import cake
    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops

    tokens, key_heads, value_heads, extra, in_place = SHAPES[name]
    x, z, w, dy = make(tokens, key_heads, value_heads, extra)
    package = ops._package("fast")
    module_for_rows = package.module_for_rows
    calls = []

    def record(arch, role, rows, sm_count, *, packed=True):
        calls.append((role, rows, packed))
        return module_for_rows(arch, role, rows, sm_count, packed=packed)

    monkeypatch.setattr(package, "module_for_rows", record)
    cake.configure(cake_gated_rmsnorm_variant="fast")
    run(x, z, w, dy)

    assert reads_in_place(z) == in_place
    rows = tokens * value_heads
    assert calls == [(role, rows, packed) for role, packed in zip(CALL_ROLES, packed_rows(tokens, in_place))]
