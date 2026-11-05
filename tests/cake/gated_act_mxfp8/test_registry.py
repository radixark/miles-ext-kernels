"""The program registry of the vendored SwiGLU + MXFP8 kernels and the libraries ``prebuild`` builds, without a device.

``cake_jit`` (the routes, the selection rule and the library names) is loaded by path under a stand-in package, so the
package's ``__init__`` (which imports torch) does not run, with a stand-in ``tvm_ffi`` (which it needs only to launch).
Checked on every architecture: each of the six routes has one default program (the one without a threshold, run below
every threshold) and distinct thresholds; ``ENTRY_SHAPES`` select every program of each route, so the entry-point and
prebuild GPU tests run them all; the programs of the routes are exactly the architecture's ``build.json`` libraries; and
no library asks for a newer toolkit than ``loader.MIN_CUDA`` names for its architecture. With torch and tvm-ffi
installed (no GPU needed): the libraries ``prebuild`` builds are exactly those, an unsupported device raises
``NotImplementedError`` and a toolkit older than CUDA 13.4 refuses SM107a before anything compiles.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake._runtime import loader

from ._shapes import ENTRY_SHAPES

PACKAGE_DIR = (
    Path(__file__).resolve().parents[3] / "miles_ext_kernels" / "cake" / "_vendor" / "cake_gated_act_mxfp8_backend"
)
_STAND_IN = "_vendored_gated_act_mxfp8"


@pytest.fixture(scope="module")
def cake_jit():
    path = PACKAGE_DIR / "cake_jit.py"
    assert path.is_file(), f"the vendored package is missing: {PACKAGE_DIR}"
    package = types.ModuleType(_STAND_IN)
    package.__path__ = [str(PACKAGE_DIR)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, _STAND_IN, package)
        patch.setitem(sys.modules, "tvm_ffi", types.ModuleType("tvm_ffi"))
        spec = importlib.util.spec_from_file_location(f"{_STAND_IN}.cake_jit", path)
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    sys.modules.pop(f"{_STAND_IN}.cake_build", None)
    return module


def _version(text: str) -> tuple[int, int]:
    major, minor = text.split(".")[:2]
    return int(major), int(minor)


def _manifest_libraries(cake_jit, arch: str) -> set[str]:
    return {name for name, entry in cake_jit.cake_build.libraries_by_name().items() if entry["arch"] == arch}


def test_the_architectures_are_the_runtimes(cake_jit):
    assert cake_jit.ARCH_BY_CAPABILITY == loader.ARCH_BY_CAPABILITY
    assert set(cake_jit.ARCHES) == set(loader.ARCH_BY_CAPABILITY.values()) == set(loader.MIN_CUDA)


@pytest.mark.parametrize("arch", sorted(loader.MIN_CUDA))
def test_every_route_has_one_default_program_and_entry_shapes_select_every_program(cake_jit, arch):
    assert set(cake_jit.ROUTES[arch]) == set(cake_jit.KINDS)
    for kind in cake_jit.KINDS:
        names = cake_jit.route_modules(arch, kind)
        thresholds = {}
        defaults = []
        for name in names:
            rule = cake_jit.select_rule(cake_jit.MODULES[name])
            if rule is None or arch not in rule["elements"]:
                defaults.append(name)
            else:
                thresholds[name] = int(rule["elements"][arch])
        assert len(defaults) == 1, (arch, kind, defaults)
        assert len(set(thresholds.values())) == len(thresholds), (arch, kind, thresholds)
        assert cake_jit.selected_module(arch, kind, M=128, K=128) == defaults[0]
        if thresholds:
            largest = max(thresholds, key=thresholds.get)
            assert cake_jit.selected_module(arch, kind, M=2**17, K=2**13) == largest
        selected = {cake_jit.selected_module(arch, kind, M=M, K=K) for M, K in ENTRY_SHAPES}
        assert selected == set(names), (arch, kind, sorted(set(names) - selected))


@pytest.mark.parametrize("arch", sorted(loader.MIN_CUDA))
def test_the_routes_load_exactly_the_build_manifest_libraries(cake_jit, arch):
    libraries = cake_jit.cake_build.libraries_by_name()
    routed = {
        cake_jit.library_name(name, arch) for kind in cake_jit.KINDS for name in cake_jit.route_modules(arch, kind)
    }

    assert routed == _manifest_libraries(cake_jit, arch)
    for name in routed:
        entry = libraries[name]
        assert _version(entry["min_cuda"]) <= loader.MIN_CUDA[arch], (name, entry["min_cuda"])


# --- prebuild, with torch and tvm-ffi but no GPU ---


@pytest.fixture
def ops():
    pytest.importorskip("torch")
    pytest.importorskip("tvm_ffi")
    from miles_ext_kernels.cake._impl.gated_act_mxfp8 import ops

    return ops


@pytest.mark.parametrize("arch", sorted(loader.MIN_CUDA))
def test_prebuild_builds_exactly_the_architecture_libraries(cake_jit, ops, arch):
    assert set(ops.libraries(arch)) == _manifest_libraries(cake_jit, arch)


def test_prebuild_on_an_unsupported_device_raises_not_implemented(ops, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(NotImplementedError, match="compute capability 9.0"):
        cake.prebuild("gated_act_mxfp8")


def test_prebuild_refuses_sm107a_with_a_toolkit_older_than_cuda_13_4(ops, monkeypatch, tmp_path):
    """A CUDA 13.0 nvcc (here a stand-in that only reports its version) cannot build SM107a: ``prebuild`` raises
    before it compiles anything, and leaves the cache empty."""
    import torch

    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = --version ]; then echo "Cuda compilation tools, release 13.0, V13.0.88"; exit 0; fi\n'
        "echo 'unexpected nvcc call' >&2; exit 1\n"
    )
    nvcc.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(nvcc.parent.parent))
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    with pytest.raises(RuntimeError, match="13.4"):
        cake.prebuild("gated_act_mxfp8", max_workers=2)
    assert not (tmp_path / "cache").exists() or not any(os.scandir(tmp_path / "cache"))
