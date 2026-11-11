"""The kernel roles of the vendored MoK package against its build manifest, without a device (torch-free).

``sources.json`` names every kernel role with its architectures and library stem, ``build.json`` every library the
package can build. Checked: each role has one library per architecture it names (every architecture, the MXFP8
forward's multimacro programs SM107a only) and the manifest no other, so ``prebuild`` selects among exactly the
libraries a run can load; no library asks for a newer toolkit than the loader's ``MIN_CUDA`` names for its
architecture; and every use of ``cvt.rn.satfinite.e4m3x2.bf16x2`` (which the CUDA 13.0 ptxas rejects) in the MXFP8
sources sits under the CUDA 13.1 guard with the FP32 fallback, the vendored sources' own rule (the runtime builds with
CUDA 13.4 or newer, so it never takes the fallback).
"""

import json
from pathlib import Path

import pytest

from miles_ext_kernels.cake._runtime import loader

PACKAGE_DIR = Path(__file__).resolve().parents[3] / "miles_ext_kernels" / "cake" / "_vendor" / "cake_mok_bf16"
ARCHES = sorted(loader.MIN_CUDA)


@pytest.fixture(scope="module")
def roles():
    path = PACKAGE_DIR / "sources.json"
    assert path.is_file(), f"the vendored package is missing: {PACKAGE_DIR}"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def manifest():
    return json.loads((PACKAGE_DIR / "build.json").read_text())


def _version(text: str) -> tuple[int, int]:
    major, minor = text.split(".")[:2]
    return int(major), int(minor)


def test_every_role_is_built_for_its_architectures(roles):
    assert len(roles) == 52
    for role, record in roles.items():
        arches = ["sm_107a"] if role.endswith("_multimacro") else ARCHES
        assert sorted(record["arches"]) == arches, role
        assert record["library"] == f"cake_mok_{role}", role
    assert sorted(role for role in roles if role.endswith("_multimacro")) == [
        "forward_mxfp8_clamped_multimacro",
        "forward_mxfp8_multimacro",
    ]
    assert sorted(role for role in roles if role.endswith("_noctx")) == ["forward_clamped_noctx", "forward_noctx"]


@pytest.mark.parametrize("arch", ARCHES)
def test_the_roles_are_exactly_the_manifest_libraries(roles, manifest, arch):
    libraries = {entry["name"]: entry for entry in manifest["libraries"] if entry["arch"] == arch}

    assert {f"{record['library']}_{arch}" for record in roles.values() if arch in record["arches"]} == set(libraries)


@pytest.mark.parametrize("arch", ARCHES)
def test_no_library_needs_a_newer_toolkit_than_the_loader_names(manifest, arch):
    for entry in manifest["libraries"]:
        if entry["arch"] == arch:
            assert _version(entry["min_cuda"]) <= loader.MIN_CUDA[arch], (entry["name"], entry["min_cuda"])


CVT = 'asm("cvt.rn.satfinite.e4m3x2.bf16x2'
GUARD = (
    "#if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && "
    "(__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1))"
)
FALLBACK = 'asm("cvt.rn.satfinite.e4m3x2.f32'


def test_every_bf16x2_e4m3_conversion_has_the_cuda_13_0_fallback(manifest):
    sources = sorted({source for entry in manifest["libraries"] for source in entry["sources"]})
    converting = []
    for source in sources:
        lines = (PACKAGE_DIR / source).read_text().splitlines()
        uses = [index for index, line in enumerate(lines) if CVT in line]
        if not uses:
            continue
        converting.append(source)
        for index in uses:
            assert lines[index - 1].strip().startswith(GUARD), (source, index + 1)
            assert lines[index + 1].strip() == "#else", (source, index + 1)
            end = next(i for i in range(index + 2, len(lines)) if lines[i].strip() == "#endif")
            assert any(FALLBACK in line for line in lines[index + 2 : end]), (source, index + 1)
    assert converting and all("mxfp8" in source for source in converting), converting
