"""The prebuild contract of ``gated_act_mxfp8`` on the device (README, Prebuild).

``prebuild("gated_act_mxfp8")`` takes no model parameter: ``M`` and ``K`` are runtime arguments, and the call-time
switches are the direction (forward, backward), the outputs (rowwise, colwise, both) and the program the selection rule
picks for ``M * K``. ``ROUTES`` runs each output set forward and backward at ``ENTRY_SHAPES``, at which the rule picks
every program of each route (``test_registry.py``), so after one ``prebuild`` every library the op can load has been
exercised under ``no_build()``. The check builds into a private ``TVM_FFI_CACHE_DIR`` with a fresh ``cake_jit.load``
cache, so a library that an earlier test loaded, or built into the shared cache, cannot hide one ``prebuild`` misses.
Needs SM100a / SM103a / SM107a; skips elsewhere.
"""

import functools
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake SwiGLU + MXFP8 kernels need SM100a / SM103a / SM107a", allow_module_level=True)

from miles_ext_kernels.cake import gated_act_mxfp8  # noqa: E402
from miles_ext_kernels.cake._impl.gated_act_mxfp8 import ops  # noqa: E402

from ..helpers import check_prebuild, no_build  # noqa: E402
from . import _reference as reference  # noqa: E402
from ._shapes import ENTRY_SHAPES  # noqa: E402

DEVICE = torch.device("cuda", torch.cuda.current_device())
ARCH = loader.device_arch(DEVICE)


def _outputs(rowwise: bool, colwise: bool):
    """A route: forward and backward with these outputs at every entry shape (every program of both routes)."""

    def run(variant: str) -> None:
        assert cake.variant("gated_act_mxfp8") == variant
        for M, K in ENTRY_SHAPES:
            gated_input, grad_output = reference.make_inputs(M, K, seed=M + K, device=DEVICE)
            gated_act_mxfp8.silu_and_mul_mxfp8_quantize(gated_input, rowwise=rowwise, colwise=colwise)
            gated_act_mxfp8.silu_and_mul_mxfp8_quantize_backward(
                gated_input, grad_output, rowwise=rowwise, colwise=colwise
            )
        torch.cuda.synchronize(DEVICE)

    return run


ROUTES = {
    "rowwise": _outputs(True, False),
    "colwise": _outputs(False, True),
    "both": _outputs(True, True),
}


def test_every_route_runs_without_a_build_after_one_prebuild(monkeypatch, tmp_path):
    # The package keeps every loaded library in the cache of cake_jit.load, and a library is found in the shared
    # TVM_FFI_CACHE_DIR when any earlier test built it. A private directory and a fresh load cache make every program
    # the routes run load a library this prebuild built, so one it misses raises under no_build().
    jit = ops.backend.cake_jit
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(jit, "load", functools.cache(jit.load.__wrapped__))

    check_prebuild("gated_act_mxfp8", ROUTES)

    assert len(list(tmp_path.rglob("*.so"))) == len(ops.libraries(ARCH))
    programs = {name for kind in jit.KINDS for name in jit.route_modules(ARCH, kind)}
    assert jit.load.cache_info().currsize == len(programs)


def test_prebuild_returns_every_library_of_the_architecture_in_the_cache_and_is_idempotent():
    manifest = ops.backend.cake_build.libraries_by_name()
    paths = cake.prebuild("gated_act_mxfp8", device=DEVICE)

    assert set(ops.libraries(ARCH)) == {name for name, entry in manifest.items() if entry["arch"] == ARCH}
    assert len(paths) == len(ops.libraries(ARCH)) and all(Path(path).is_file() for path in paths)
    assert all(Path(path).is_relative_to(loader.cache_dir()) for path in paths)
    with no_build():
        assert cake.prebuild("gated_act_mxfp8", device=DEVICE) == paths
