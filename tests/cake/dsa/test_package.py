"""The three vendored DSA packages as the op uses them (torch-free): every det stage on every architecture, the
architectures and toolkits each package builds for, and the CUDA 13.4 floor of SM107a whatever a package records."""

import json
import sys
from pathlib import Path

import pytest

import miles_ext_kernels.cake._vendor as vendor
from miles_ext_kernels.cake import dsa
from miles_ext_kernels.cake._runtime import loader

VENDOR = Path(vendor.__file__).resolve().parent
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
FLOOR = {"sm_100a": "12.8", "sm_103a": "12.9", "sm_107a": "13.4"}
# libraries per architecture of each package
LIBRARIES = {
    "cake_dsa_indexer_det": {"sm_100a": 52, "sm_103a": 52, "sm_107a": 53},
    "cake_dsa_sparse_det": {"sm_100a": 43, "sm_103a": 43, "sm_107a": 43},
    "cake_dsa_train": {"sm_100a": 14, "sm_103a": 14, "sm_107a": 14},
}

FAKE_NVCC = """#!@PYTHON@
import sys
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print("nvcc: NVIDIA (R) Cuda compiler driver")
    print("Cuda compilation tools, release @MAJOR@.@MINOR@, V@FULL@")
    sys.exit(0)
Path(__file__).with_name("compiled").touch()
sys.exit(1)
"""


def _build(package: str) -> dict:
    return json.loads((VENDOR / package / "build.json").read_text())


def _libraries(package: str, arch: str) -> list[dict]:
    return [entry for entry in _build(package)["libraries"] if entry["arch"] == arch]


def test_the_op_ships_the_three_packages_both_variants_and_its_options():
    assert dsa.OP.packages == ("cake_dsa_indexer_det", "cake_dsa_sparse_det", "cake_dsa_train")
    assert dsa.OP.variants == ("det", "fast")
    assert dsa.OP.selected_by == (("dsa_attention_backend", "cake"),)
    assert [flag.name for flag in dsa.OP.flags] == ["--cake-dsa-bwd-workspace-bytes", "--cake-dsa-lazy-build"]


@pytest.mark.parametrize("package", sorted(LIBRARIES))
def test_every_package_builds_for_every_architecture(package):
    counts = {arch: len(_libraries(package, arch)) for arch in ARCHES}

    assert counts == LIBRARIES[package]
    assert {entry["arch"] for entry in _build(package)["libraries"]} == set(ARCHES)


def test_the_det_version_has_one_library_per_stage_and_architecture():
    registry = json.loads((VENDOR / "cake_dsa_sparse_det" / "registry.json").read_text())
    build = _build("cake_dsa_sparse_det")
    by_name = {entry["name"]: entry for entry in build["libraries"]}

    assert len(registry) == 43
    for stage, record in sorted(registry.items()):
        assert record["arches"] == ["sm_100a", "sm_103a"], stage
        entries = {arch: by_name[f"{record['name']}_{arch}"] for arch in ARCHES}
        for arch, entry in entries.items():
            assert entry["arch"] == arch and entry["min_cuda"] == FLOOR[arch], (stage, arch)
        # SM107a compiles the SM103a sources (for compute_107a)
        assert entries["sm_107a"]["sources"] == entries["sm_103a"]["sources"], stage
    assert any("compute_107a" in flag for flag in build["arch_flags"]["sm_107a"])
    assert "-O3" in build["common"]["nvcc_flags"]


@pytest.fixture
def nvcc(tmp_path, monkeypatch):
    """``nvcc("13.0.88")``: a fake nvcc of that version as ``$CUDA_HOME/bin/nvcc`` that fails any compile."""
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path / "cache"))

    def make(full: str) -> Path:
        major, minor, _ = full.split(".")
        path = tmp_path / f"cuda-{full}" / "bin" / "nvcc"
        path.parent.mkdir(parents=True)
        script = FAKE_NVCC.replace("@PYTHON@", sys.executable).replace("@FULL@", full)
        path.write_text(script.replace("@MAJOR@", major).replace("@MINOR@", minor))
        path.chmod(0o755)
        monkeypatch.setenv("CUDA_HOME", str(path.parents[1]))
        return path

    return make


@pytest.mark.parametrize("package", sorted(LIBRARIES))
def test_sm107a_needs_cuda_13_4_whatever_the_package_records(package, nvcc):
    """nvcc 13.0 to 13.3 cannot compile compute_107a; the fast package's build.json records 13.0 for SM107a."""
    fake = nvcc("13.0.88")
    names = [entry["name"] for entry in _libraries(package, "sm_107a")]
    recorded = {entry["min_cuda"] for entry in _libraries(package, "sm_107a")}

    assert recorded == ({"13.0"} if package == "cake_dsa_train" else {"13.4"})
    for name in names[:2]:
        with pytest.raises(RuntimeError, match=r"needs CUDA 13\.4 or newer"):
            loader.prebuild_libraries(package, [name])
    assert not fake.with_name("compiled").exists()


@pytest.mark.parametrize("package", sorted(LIBRARIES))
def test_sm103a_needs_cuda_12_9(package, nvcc):
    fake = nvcc("12.8.93")
    name = _libraries(package, "sm_103a")[0]["name"]

    with pytest.raises(RuntimeError, match=r"needs CUDA 12\.9 or newer"):
        loader.prebuild_libraries(package, [name])
    assert not fake.with_name("compiled").exists()
