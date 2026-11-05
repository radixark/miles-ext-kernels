"""The five vendored packages of gdn and how the implementation names them (torch-free).

Every package builds every architecture the op serves; the det kernels with FLA's context parallelism
(``cake_gdn_fla_cp_det``) build one library per stage record and architecture from the sources their ``registry.json``
names, SM107a from the SM103a records; the implementation loads exactly the packages ``OP.packages`` declares.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from miles_ext_kernels.cake import gdn

REPO_ROOT = Path(__file__).resolve().parents[3]
CAKE = REPO_ROOT / "miles_ext_kernels" / "cake"
VENDOR = CAKE / "_vendor"
IMPL = CAKE / "_impl" / "gdn"
FLA_CP = VENDOR / "cake_gdn_fla_cp_det"
ARCHES = {"sm_100a", "sm_103a", "sm_107a"}


def _module_constant(path: Path, name: str):
    """A literal module-level constant of ``path``, read from the source (importing it would import torch)."""
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} assigns no {name}")


def _records() -> dict:
    """``(stage, arch) -> record`` of the FLA-CP registry, ``per_arch`` merged, SM107a compiling the SM103a record."""
    records = {}
    for stage, record in json.loads((FLA_CP / "registry.json").read_text()).items():
        per_arch = record.get("per_arch") or {}
        for arch in record["arches"]:
            records[stage, arch] = {**record, **per_arch.get(arch, {})}
    for stage in {stage for stage, _ in records}:
        records[stage, "sm_107a"] = records[stage, "sm_103a"]
    return records


def test_importing_the_op_and_registering_its_arguments_imports_neither_torch_nor_the_implementation():
    script = (
        "import argparse, sys\n"
        "import miles_ext_kernels\n"
        "from miles_ext_kernels.cake import gdn\n"
        "miles_ext_kernels.add_arguments(argparse.ArgumentParser())\n"
        "assert gdn.__all__ == ['OP', *gdn.OP.exports]\n"
        "loaded = sorted(n for n in sys.modules if n.split('.')[0] in ('torch', 'tvm_ffi')\n"
        "                or n.startswith(('miles_ext_kernels.cake._impl.', 'miles_ext_kernels.cake._vendor.')))\n"
        "assert loaded == [], loaded\n"
    )
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("package", gdn.OP.packages)
def test_every_package_builds_every_architecture_the_op_serves(package):
    build = json.loads((VENDOR / package / "build.json").read_text())

    assert {entry["arch"] for entry in build["libraries"]} == ARCHES
    assert (VENDOR / package / "__init__.py").is_file() and (VENDOR / package / "cake_build.py").is_file()


def test_the_implementation_loads_exactly_the_declared_packages():
    packages = _module_constant(IMPL / "ops.py", "_PACKAGES")
    native = _module_constant(IMPL / "native.py", "PACKAGE")

    assert sorted(packages) == ["det", "det_kcp", "fast", "fast_kcp"]
    assert sorted([*packages.values(), native]) == sorted(gdn.OP.packages)
    assert _module_constant(IMPL / "native.py", "EXPORT_ARCH") == {"sm_107a": "sm_103a"}


def test_every_stage_has_the_sm103a_export_sm107a_compiles():
    registry = json.loads((FLA_CP / "registry.json").read_text())

    assert registry
    assert sorted(stage for stage, record in registry.items() if "sm_103a" not in record["arches"]) == []
    assert sorted(stage for stage, record in registry.items() if "sm_107a" in record["arches"]) == []


def test_the_fla_cp_package_builds_one_library_per_stage_record_and_architecture():
    # the library names native.load asks the package's loader for: <record name>_<arch>, from the record's own sources
    records = _records()
    build = json.loads((FLA_CP / "build.json").read_text())
    libraries = {entry["name"]: entry for entry in build["libraries"]}

    wanted = {f"{record['name']}_{arch}": (arch, record) for (_, arch), record in records.items()}
    assert sorted(set(wanted) - set(libraries)) == []
    assert sorted(set(libraries) - set(wanted)) == []
    for name, (arch, record) in wanted.items():
        entry = libraries[name]
        assert entry["arch"] == arch
        assert entry["sources"] == [f"csrc/{source}" for source in record["sources"]], name
        assert entry["entry"] == "run"
        assert entry["min_cuda"] == {"sm_100a": "12.8", "sm_103a": "12.9", "sm_107a": "13.4"}[arch]
    sources = {source for entry in libraries.values() for source in entry["sources"]}
    assert sorted(source for source in sources if not (FLA_CP / source).is_file()) == []


def test_the_fla_cp_sources_build_with_the_flags_of_the_previous_build():
    # the torch-extension build compiled every stage with -O3 and the record's compile flags (none); the loader adds
    # -std=c++17, -fPIC and --expt-relaxed-constexpr, which the generated sources need
    build = json.loads((FLA_CP / "build.json").read_text())
    registry = json.loads((FLA_CP / "registry.json").read_text())

    assert {tuple(record.get("compile_flags", ())) for record in registry.values()} == {()}
    assert build["common"]["nvcc_flags"] == [
        "-std=c++17",
        "-O3",
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
    ]
    assert {tuple(entry["nvcc_flags"]) for entry in build["libraries"]} == {()}
    assert build["arch_flags"] == {arch: [f"-gencode=arch=compute_{arch[3:]},code={arch}"] for arch in ARCHES}
