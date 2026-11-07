"""The vendored mHC packages serve what ``miles_ext_kernels.cake.mhc`` relies on (torch-free, no device).

Each package's ``cake_jit.py`` is loaded by file path (its module level imports neither torch nor the package): every
library a route (or the head) can launch on sm_100a / sm_103a / sm_107a has its ``build.json`` entry, and the shapes,
constants, architectures and per-architecture hidden sizes (4096 only on sm_107a) of the three packages are the ones
``miles_ext_kernels.cake.mhc`` validates against.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake import mhc
from miles_ext_kernels.cake._runtime import loader

VENDOR = Path(cake.__file__).resolve().parent / "_vendor"
DET, FAST, HEAD = "cake_mhc_det_train", "cake_mhc_nondet_train", "cake_mhc_head_det_train"
FI_COMMIT = "ab63ff3afd8bbb74fe69ace64de706c4addb54c7"
FI_PATHS = {
    DET: "flashinfer/experimental/cake/cake_mhc_det_train",
    FAST: "flashinfer/experimental/cake/cake_mhc_nondet_train",
    HEAD: "flashinfer/experimental/cake/cake_mhc_head_det_train",
}
# the hidden sizes every route (and the head) serves on each architecture
HIDDEN_BY_ARCH = {"sm_100a": mhc._HIDDEN_SIZES, "sm_103a": mhc._HIDDEN_SIZES, "sm_107a": (4096,)}
ARCHES = tuple(HIDDEN_BY_ARCH)


def _cake_jit(package: str):
    path = VENDOR / package / "cake_jit.py"
    spec = importlib.util.spec_from_file_location(f"_mhc_routes_{package}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _libraries(package: str) -> set[str]:
    manifest = json.loads((VENDOR / package / "build.json").read_text())
    assert manifest["schema"] == "cake-build/1"
    return {entry["name"] for entry in manifest["libraries"]}


def test_the_three_packages_are_the_ops_block_of_the_manifest_at_one_flashinfer_commit():
    block = json.loads((VENDOR / "VENDORED.json").read_text())["ops"]["mhc"]

    assert set(block) == set(mhc.OP.packages) == set(FI_PATHS)
    for package, entry in block.items():
        assert entry["source"] == {
            "commit": FI_COMMIT,
            "kind": "flashinfer",
            "path": FI_PATHS[package],
            "repo": "https://github.com/flashinfer-ai/flashinfer",
        }
        assert "allow" not in entry and "local" not in entry  # no environment read, no flashinfer import


@pytest.mark.parametrize("package", [DET, FAST])
@pytest.mark.parametrize("arch", ARCHES)
def test_every_route_library_is_in_the_manifest(package, arch):
    jit = _cake_jit(package)
    libraries = _libraries(package)

    assert jit.supported_hidden_sizes(arch) == HIDDEN_BY_ARCH[arch]
    for hidden in HIDDEN_BY_ARCH[arch]:
        for kind in jit.KINDS:
            for record in jit.route_modules(arch, hidden, kind):
                assert jit.MODULES[record]["hidden"] == hidden
                assert jit.library_name(record, arch) in libraries, (package, arch, hidden, kind, record)


@pytest.mark.parametrize("package", [DET, FAST])
def test_the_connection_constants_are_the_ones_the_op_validates(package):
    jit = _cake_jit(package)

    assert jit.HIDDEN_SIZES == mhc._HIDDEN_SIZES
    assert jit.STREAMS == mhc._STREAMS
    assert jit.MAPPING_ROWS == 2 * mhc._STREAMS + mhc._STREAMS**2
    assert (jit.RMS_EPS, jit.HC_EPS, jit.SINKHORN_ITERS, jit.SINKHORN_ITERS_MAX) == (1e-5, 1e-6, 20, 64)
    assert jit.ARCH_BY_CAPABILITY == loader.ARCH_BY_CAPABILITY


@pytest.mark.parametrize("arch", ARCHES)
def test_the_head_libraries_are_in_the_manifest(arch):
    jit = _cake_jit(HEAD)
    libraries = _libraries(HEAD)

    assert jit.supported_hidden_sizes(arch) == HIDDEN_BY_ARCH[arch]
    assert jit.STREAMS == mhc._STREAMS
    for role in ("head_fwd", "head_bwd"):
        assert jit.library_name(jit.ROLES[role], arch) in libraries


@pytest.mark.parametrize("package", [DET, FAST])
@pytest.mark.parametrize("arch", ARCHES)
def test_another_sinkhorn_count_runs_run_time_count_programs(package, arch):
    """Why ``prebuild`` takes ``sinkhorn_iters``: a count other than 20 runs programs of their own at every hidden size
    (the coefficient pass and the token-owner pass), which a build for the count of record does not need."""
    jit = _cake_jit(package)

    for hidden in HIDDEN_BY_ARCH[arch]:
        run_time = {
            jit.MODULES[record]["role"]
            for kind in jit.KINDS
            for record in jit.route_modules(arch, hidden, kind)
            if (jit.MODULES[record]["select_rule"] or {}).get("kind") == "run_time_iters"
        }
        assert run_time == set(jit.RUN_TIME_ITERS_ROLES) == {"fwd_coef", "bwd_token"}, (hidden, run_time)
