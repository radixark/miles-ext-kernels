"""The vendored mHC packages serve what ``miles_ext_kernels.cake.mhc`` relies on (torch-free, no device).

Each package's ``cake_jit.py`` is loaded by file path (its module level imports neither torch nor the package): every
library a route (or a head) can launch on sm_100a / sm_103a / sm_107a has its ``build.json`` entry, and the shapes,
constants, architectures and per-architecture hidden sizes of the four packages are the ones
``miles_ext_kernels.cake.mhc`` validates against (on sm_107a the det package serves every hidden size, the other three
4096 only, so the op serves 4096 there). The nondet head is the det head's kernels under its own names.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake import mhc
from miles_ext_kernels.cake._runtime import loader

VENDOR = Path(cake.__file__).resolve().parent / "_vendor"
DET, FAST = "cake_mhc_det_train", "cake_mhc_nondet_train"
DET_HEAD, FAST_HEAD = "cake_mhc_head_det_train", "cake_mhc_head_nondet_train"
FI_COMMIT = "c5c39b0e87f027011c86881aa2935a83e1ce53e7"
FI_PATHS = {
    DET: "flashinfer/experimental/cake/cake_mhc_det_train",
    FAST: "flashinfer/experimental/cake/cake_mhc_nondet_train",
    DET_HEAD: "flashinfer/experimental/cake/cake_mhc_head_det_train",
    FAST_HEAD: "flashinfer/experimental/cake/cake_mhc_head_nondet_train",
}
# the hidden sizes the op serves on each architecture: the nondet package's and each head's (det serves them all)
HIDDEN_BY_ARCH = {"sm_100a": mhc._HIDDEN_SIZES, "sm_103a": mhc._HIDDEN_SIZES, "sm_107a": (4096,)}
ARCHES = tuple(HIDDEN_BY_ARCH)
SERVED = {
    DET: {arch: mhc._HIDDEN_SIZES for arch in ARCHES},
    FAST: HIDDEN_BY_ARCH,
    DET_HEAD: HIDDEN_BY_ARCH,
    FAST_HEAD: HIDDEN_BY_ARCH,
}


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


def test_the_packages_are_the_ops_block_of_the_manifest_at_one_flashinfer_commit():
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

    assert jit.supported_hidden_sizes(arch) == SERVED[package][arch]
    for hidden in SERVED[package][arch]:
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
@pytest.mark.parametrize("package", [DET_HEAD, FAST_HEAD])
def test_the_head_libraries_are_in_the_manifest(package, arch):
    """Each role's library of every served hidden size (built for that size), and the any-size library of a role the
    package routes small problems to (built for hidden 0)."""
    jit = _cake_jit(package)
    libraries = _libraries(package)

    assert jit.supported_hidden_sizes(arch) == SERVED[package][arch]
    assert jit.STREAMS == mhc._STREAMS
    for role in ("head_fwd", "head_bwd", "head_bwd_acc"):
        routes = jit.ROUTES[arch][role]
        assert tuple(routes) == SERVED[package][arch], (arch, role)
        for hidden, record in routes.items():
            assert jit.MODULES[record]["hidden"] == hidden and jit.MODULES[record]["role"] == role
            assert jit.library_name(record, arch) in libraries, (package, arch, role, hidden)
        small = jit.SMALL_ROUTES.get(arch, {}).get(role)
        if small is not None:
            assert jit.MODULES[small]["hidden"] == 0 and jit.MODULES[small]["role"] == role
            assert jit.library_name(small, arch) in libraries, (package, arch, role)


@pytest.mark.parametrize("package", [DET, FAST])
@pytest.mark.parametrize("arch", ARCHES)
def test_another_sinkhorn_count_runs_run_time_count_programs(package, arch):
    """Why ``prebuild`` takes ``sinkhorn_iters``: a count other than 20 runs programs of their own at every hidden size
    (the coefficient pass and the token-owner pass), which a build for the count of record does not need."""
    jit = _cake_jit(package)

    for hidden in SERVED[package][arch]:
        run_time = {
            jit.MODULES[record]["role"]
            for kind in jit.KINDS
            for record in jit.route_modules(arch, hidden, kind)
            if (jit.MODULES[record]["select_rule"] or {}).get("kind") == "run_time_iters"
        }
        assert run_time == set(jit.RUN_TIME_ITERS_ROLES) == {"fwd_coef", "bwd_token"}, (hidden, run_time)


def _stem(text: str) -> str:
    """The det head's library stem renamed to the nondet head's."""
    return text.replace("cake_mhc_det_train_head", "cake_mhc_train_head")


def test_the_nondet_head_is_the_det_heads_kernels_under_its_own_names():
    """FlashInfer exports the nondet head from the det head's kernel sources under the library stem
    ``cake_mhc_train_head``: every CUDA source, launch record and build flag is the det head's after that rename, so
    both variants of ``mhc_head`` compute the same bits."""
    det_csrc, fast_csrc = VENDOR / DET_HEAD / "csrc", VENDOR / FAST_HEAD / "csrc"
    det_sources = {_stem(path.name): path for path in det_csrc.glob("*.cu")}
    fast_sources = {path.name: path for path in fast_csrc.glob("*.cu")}
    assert set(fast_sources) == set(det_sources) and len(fast_sources) == 42
    for name, path in fast_sources.items():
        assert path.read_text() == _stem(det_sources[name].read_text()), name
    assert (fast_csrc / "cake_ffi_checks.h").read_bytes() == (det_csrc / "cake_ffi_checks.h").read_bytes()

    det_jit, fast_jit = _cake_jit(DET_HEAD), _cake_jit(FAST_HEAD)
    assert json.loads(_stem(json.dumps(det_jit.MODULES))) == json.loads(json.dumps(fast_jit.MODULES))
    for table in ("ROUTES", "SMALL_ROUTES"):
        det_table, fast_table = (json.dumps(getattr(jit, table)) for jit in (det_jit, fast_jit))
        assert json.loads(_stem(det_table)) == json.loads(fast_table), table
    assert det_jit.SMALL_ELEMENTS == fast_jit.SMALL_ELEMENTS
    assert det_jit.SMALL_ELEMENTS_BY_ROUTE == fast_jit.SMALL_ELEMENTS_BY_ROUTE

    det_build = json.loads((VENDOR / DET_HEAD / "build.json").read_text())
    fast_build = json.loads((VENDOR / FAST_HEAD / "build.json").read_text())
    assert det_build["common"] == fast_build["common"] and det_build["arch_flags"] == fast_build["arch_flags"]
    flags = ("arch", "entry", "gencode", "min_cuda", "nvcc_flags", "nvcc_flags_if")

    def build_flags(build: dict) -> dict:
        return {_stem(library["name"]): [library[key] for key in flags] for library in build["libraries"]}

    assert build_flags(det_build) == build_flags(fast_build) and len(build_flags(fast_build)) == 26
