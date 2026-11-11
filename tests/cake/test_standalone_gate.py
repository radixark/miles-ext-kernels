"""Every vendored package works as a copy of its own directory (``tools/check_standalone.py``).

CPU: the static pass over every package, the blocked import of every package from a copy (its children need torch
and tvm_ffi), and the gate's own checks on synthetic packages with a fake nvcc. GPU node with nvcc: every package
compiled for the device's architecture into a copy and loaded from it.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from miles_ext_kernels.cake._runtime import loader

from ._runtime import fake_nvcc
from .helpers import requires_arch

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "tools"
CAKE_BUILD = TOOLS / "templates" / "cake_build.py"

_spec = importlib.util.spec_from_file_location("check_standalone", TOOLS / "check_standalone.py")
check_standalone = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_standalone)

PACKAGES = check_standalone.discover(REPO_ROOT)
NAMES = [package.name for package in PACKAGES]


def _lines(findings: list[Any]) -> str:
    return "\n".join(finding.line() for finding in findings)


@pytest.fixture(scope="module")
def environment(tmp_path_factory):
    """Where this interpreter loads torch and tvm_ffi from: what the gate's children need."""
    pytest.importorskip("torch")
    pytest.importorskip("tvm_ffi")
    return check_standalone.probe(sys.executable, tmp_path_factory.mktemp("probe"))


def test_the_vendored_packages_keep_the_contract():
    """Only the allowlisted findings, and every allowlist entry still matches one."""
    findings = check_standalone.check_static(PACKAGES, PACKAGES)

    assert not findings, _lines(findings)


def test_every_allowlist_entry_names_a_vendored_module_and_why():
    for package, module, subject, reason in check_standalone.ALLOWLIST:
        assert package in NAMES and (REPO_ROOT / check_standalone.VENDOR / package / module).is_file()
        assert subject and reason


def test_the_build_floor_is_the_runtime_floor():
    assert check_standalone.MIN_CUDA == loader.MIN_CUDA
    assert set(check_standalone.ARCHES) == set(loader.ARCH_BY_CAPABILITY.values())


# imports every module named on its command line, with every flashinfer* package blocked
IN_PLACE = """\
import importlib
import importlib.abc
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0].startswith("flashinfer"):
            raise ModuleNotFoundError(f"{name} is blocked", name=name)
        return None


sys.meta_path.insert(0, Block())
failed = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {type(exc).__name__}: {exc}")
print("\\n".join(failed))
sys.exit(1 if failed else 0)
"""


def test_every_module_imports_in_place_without_flashinfer():
    """The tree as installed: every module under ``miles_ext_kernels.cake._vendor`` imports with flashinfer blocked."""
    pytest.importorskip("torch")
    pytest.importorskip("tvm_ffi")
    modules = [
        ".".join(["miles_ext_kernels.cake._vendor", package.name, *([stem] if stem else [])])
        for package in PACKAGES
        for stem in ["", *check_standalone.module_stems(package.directory)]
    ]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")

    result = subprocess.run(
        [sys.executable, "-c", IN_PLACE, *modules],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=check_standalone.CHILD_TIMEOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", NAMES)
def test_every_module_imports_from_a_standalone_copy(name, environment, tmp_path):
    packages = check_standalone.select(PACKAGES, [name])

    (report,), findings = check_standalone.check_imports(packages, PACKAGES, environment, tmp_path, REPO_ROOT)

    assert not findings, _lines(findings)
    assert report["failed"] == 0 and report["modules"] >= 2  # the package and its cake_build at least


# --- the gate on synthetic packages ---


def _manifest(name: str, arches=("sm_100a", "sm_103a"), min_cuda: str = "13.4", siblings=()) -> dict[str, Any]:
    """A ``cake-build/1`` manifest with one library ``prog_<arch>`` per architecture."""
    common: dict[str, Any] = {
        "include_dirs": ["csrc"],
        "link_flags": ["-shared", "-lcudart"],
        "nvcc_flags": ["-std=c++17", "-O3"],
        "nvcc_flags_if": [{"cuda": ">=13.4", "flags": ["-DSYNTHETIC_NEWER"]}],
    }
    if siblings:
        common["requires"] = {"packages": list(siblings)}
    return {
        "schema": "cake-build/1",
        "package": f"elsewhere.{name}",
        "arch_flags": {arch: [f"-gencode=arch=compute_{arch[3:]},code={arch}"] for arch in check_standalone.ARCHES},
        "common": common,
        "libraries": [
            {
                "arch": arch,
                "min_cuda": min_cuda,
                "name": f"prog_{arch}",
                "nvcc_flags": [f"-DPROG_{arch.upper()}"],
                "nvcc_flags_if": [],
                "path": f"aot/prog_{arch}.so",
                "sources": ["csrc/prog.cu"],
            }
            for arch in arches
        ],
    }


def _tree(root: Path, packages: dict[str, dict[str, str]], records: dict[str, dict[str, Any]] | None = None) -> Path:
    """A repository at ``root`` vendoring ``packages`` (name -> package-relative path -> text) under one op. Each
    package also gets FlashInfer's loader, an empty ``__init__.py``, ``csrc/prog.cu`` and :func:`_manifest`, unless
    it gives its own. ``records`` replaces the ``VENDORED.json`` record of a package."""
    vendor = root / check_standalone.VENDOR
    block = {}
    for name, files in packages.items():
        directory = vendor / name
        files = {"__init__.py": "", "csrc/prog.cu": "// kernel\n", "build.json": json.dumps(_manifest(name)), **files}
        for relative, text in files.items():
            (directory / relative).parent.mkdir(parents=True, exist_ok=True)
            (directory / relative).write_text(text)
        shutil.copyfile(CAKE_BUILD, directory / "cake_build.py")
        block[name] = (records or {}).get(name, {"source": {"kind": "flashinfer"}})
    (vendor / "VENDORED.json").write_text(json.dumps({"ops": {"synth": block}, "schema": "synthetic"}))
    return root


OUTSIDE = """\
import os
import tvm_ffi
from cuda.bindings import driver
import torch.distributed

import numpy

from ..cake_synth_b import helper
from ... import elsewhere

KNOB = os.environ.get("KNOB")


def jit():
    from flashinfer.jit import core

    return core
"""


def test_the_static_pass_reports_what_leaves_the_contract(tmp_path):
    repo = _tree(
        tmp_path / "repo",
        {
            "cake_synth_a": {"mod.py": OUTSIDE},
            "cake_synth_kcp": {
                "build.json": json.dumps(_manifest("cake_synth_kcp", siblings=["cake_synth_a"])),
                "sub/__init__.py": "from ...cake_synth_a import mod\nfrom . import inner\n",
                "sub/inner.py": "",
            },
            "cake_synth_lost": {"build.json": json.dumps(_manifest("cake_synth_lost", siblings=["cake_synth_gone"]))},
            "cake_synth_mok": {
                "sources.json": json.dumps({"fwd": {"library": "prog", "arches": ["sm_100a", "sm_107a"]}}),
            },
        },
    )
    packages = check_standalone.discover(repo)

    findings = check_standalone.check_static(packages, packages, allowlist=())

    assert {(finding.package, finding.module, finding.subject) for finding in findings} == {
        ("cake_synth_a", "mod.py", "import numpy"),
        ("cake_synth_a", "mod.py", "import ..cake_synth_b"),
        ("cake_synth_a", "mod.py", "import above the parent"),
        ("cake_synth_a", "mod.py", "environment"),
        ("cake_synth_a", "mod.py", "import flashinfer in jit"),
        ("cake_synth_lost", "build.json", "sibling cake_synth_gone"),
        ("cake_synth_mok", "sources.json", "registry"),
    }
    (registry,) = [finding for finding in findings if finding.subject == "registry"]
    assert registry.message.endswith(": prog_sm_107a")


def test_the_static_pass_reads_a_miles_pr_registry_per_architecture(tmp_path):
    """``per_arch`` renames a record for one architecture; ``arch_from`` compiles another architecture's records."""
    registry = {
        "fwd": {"name": "prog", "arches": ["sm_100a", "sm_103a"]},
        "bwd": {"name": "plain", "arches": ["sm_100a", "sm_103a"], "per_arch": {"sm_103a": {"name": "tuned"}}},
    }
    record = {
        "source": {"kind": "miles-pr"},
        "build": {"registry": "registry.json", "arch_from": {"sm_107a": "sm_103a"}},
    }
    files = {"registry.json": json.dumps(registry)}
    repo = _tree(tmp_path / "repo", {"cake_synth_pr": files}, {"cake_synth_pr": record})
    (package,) = check_standalone.discover(repo)

    findings = check_standalone.check_static([package], [package], allowlist=())

    assert check_standalone.registry_libraries(package) == {
        "prog_sm_100a",
        "prog_sm_103a",
        "prog_sm_107a",
        "plain_sm_100a",
        "tuned_sm_103a",
        "tuned_sm_107a",
    }
    (finding,) = findings
    assert finding.subject == "registry"
    assert finding.message.endswith(": plain_sm_100a, prog_sm_107a, tuned_sm_103a, tuned_sm_107a")


JIT_HOOK = """\
def _jit_toolchain():
    try:
        from flashinfer.jit import core
    except ImportError:
        return None
    return core
"""


@pytest.mark.parametrize(
    ("relative", "source", "subjects"),
    [
        ("cake_build.py", JIT_HOOK, []),
        ("cake_build.py", JIT_HOOK.replace("ImportError", "ValueError"), ["import flashinfer in _jit_toolchain"]),
        ("cake_build.py", "def _jit_toolchain():\n    import flashinfer\n", ["import flashinfer in _jit_toolchain"]),
        ("cake_build.py", JIT_HOOK.replace("_jit_toolchain", "_other"), ["import flashinfer in _other"]),
        ("helper.py", JIT_HOOK, ["import flashinfer in _jit_toolchain"]),
    ],
)
def test_the_one_flashinfer_import_is_the_guarded_jit_hook_of_the_loader(relative, source, subjects):
    findings = check_standalone.module_findings("cake_synth_a", relative, source.encode(), ())

    assert [finding.subject for finding in findings] == subjects


ALLOWED = """\
import os

KNOB = os.environ.get("KNOB")


def hook():
    from flashinfer.jit import core

    return core


class Toolchain:
    def other(self):
        import flashinfer

        return flashinfer
"""


def test_vendored_allow_scopes_a_flashinfer_import_to_its_function(tmp_path):
    """``VENDORED.json`` ``allow`` allows the environment reads of a module and the flashinfer imports of a function;
    an ``allow`` entry that matches nothing is stale."""
    allow = {"env_read": ["mod.py", "quiet.py"], "flashinfer_import": ["mod.py:gone", "mod.py:hook"]}
    record = {"source": {"kind": "flashinfer"}, "allow": allow}
    repo = _tree(tmp_path / "repo", {"cake_synth_a": {"mod.py": ALLOWED, "quiet.py": ""}}, {"cake_synth_a": record})
    packages = check_standalone.discover(repo)

    findings = check_standalone.check_static(packages, packages, allowlist=())

    assert {(finding.module, finding.subject) for finding in findings} == {
        ("mod.py", "import flashinfer in Toolchain.other"),
        ("mod.py", "import flashinfer in gone"),
        ("quiet.py", "environment"),
    }, _lines(findings)
    stale = [finding for finding in findings if "no longer occurs" in finding.message]
    assert len(stale) == 2 and all("VENDORED.json allow" in finding.message for finding in stale)


def test_the_allowlist_hides_its_findings_and_reports_its_stale_entries(tmp_path):
    repo = _tree(tmp_path / "repo", {"cake_synth_a": {"mod.py": "import os\n\nKNOB = os.getenv('KNOB')\n"}})
    packages = check_standalone.discover(repo)
    allowlist = [
        ("cake_synth_a", "mod.py", "environment", "a reason"),
        ("cake_synth_a", "mod.py", "import triton", "no longer imported"),
        ("cake_synth_gone", "mod.py", "environment", "no longer vendored"),
    ]

    findings = check_standalone.check_static(packages, packages, allowlist)

    assert [(finding.package, finding.subject) for finding in findings] == [
        ("cake_synth_a", "import triton"),
        ("cake_synth_gone", "environment"),
    ]
    assert all("no longer occurs: delete the entry" in finding.message for finding in findings)
    assert check_standalone.blocked_for("cake_synth_a", allowlist) == ["flashinfer", "miles_ext_kernels", "triton"]


# imports a module of the repository by its path
LEAK = """\
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("from_repo", "@HELPER@")
sys.modules["from_repo"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["from_repo"])
"""


def test_the_import_check_blocks_what_a_copy_must_not_need(tmp_path, environment):
    repo = _tree(
        tmp_path / "repo",
        {
            "cake_synth_a": {"mod.py": "VALUE = 1\n"},
            "cake_synth_kcp": {
                "build.json": json.dumps(_manifest("cake_synth_kcp", siblings=["cake_synth_a"])),
                "use.py": "from ..cake_synth_a.mod import VALUE\n",
            },
            "cake_synth_flashinfer": {"mod.py": "import flashinfer\n"},
            "cake_synth_miles": {"mod.py": "from miles_ext_kernels import cake\n"},
            "cake_synth_cubin": {"mod.py": "import flashinfer_cubin\n"},
            "cake_synth_undeclared": {"mod.py": "from ..cake_synth_a import mod\n"},
            "cake_synth_optional": {"mod.py": "import sys\n\nassert sys.modules['triton'] is None\n"},
            "cake_synth_leak": {"mod.py": LEAK.replace("@HELPER@", str(tmp_path / "repo" / "helper.py"))},
        },
    )
    (repo / "helper.py").write_text("")
    packages = check_standalone.discover(repo)
    allowlist = [("cake_synth_optional", "mod.py", "import triton", "optional")]
    work = tmp_path / "work"

    reports, findings = check_standalone.check_imports(packages, packages, environment, work, repo, allowlist)

    assert {(finding.package, finding.subject) for finding in findings} == {
        ("cake_synth_flashinfer", "import"),
        ("cake_synth_miles", "import"),
        ("cake_synth_cubin", "import"),
        ("cake_synth_undeclared", "import"),
        ("cake_synth_leak", "leak"),
    }, _lines(findings)
    by_package = {finding.package: finding.message for finding in findings}
    for package, blocked in (("cake_synth_flashinfer", "flashinfer"), ("cake_synth_miles", "miles_ext_kernels")):
        assert "ModuleNotFoundError" in by_package[package] and blocked in by_package[package]
    assert "flashinfer_cubin is blocked by the standalone gate" in by_package["cake_synth_cubin"]
    assert "beyond top-level package" in by_package["cake_synth_undeclared"]
    assert by_package["cake_synth_leak"] == "loaded from_repo"
    assert {report["package"]: report["failed"] for report in reports}["cake_synth_kcp"] == 0


def _build(repo: Path, work: Path, *args: str) -> tuple[int, dict[str, Any]]:
    report = work.parent / f"{work.name}-report.json"
    argv = ["build", "--repo-root", str(repo), "--work-dir", str(work), "--report-json", str(report), "--compile-only"]
    code = check_standalone.main([*argv, "--jobs", "1", *args])
    return code, json.loads(report.read_text())


BUILD_INFO = b"""\
{
  "arches": [
    "sm_100a",
    "sm_103a"
  ],
  "cuda": "13.4.92",
  "cudart_version": 13040,
  "schema": "cake-build-info/1"
}
"""


def test_build_compiles_every_entry_with_the_reference_command_into_a_copy(tmp_path, environment):
    repo = _tree(tmp_path / "repo", {"cake_synth_a": {}})
    nvcc = fake_nvcc.write(tmp_path / "cuda" / "bin" / "nvcc", "13.4.92")

    code, report = _build(repo, tmp_path / "work", "--nvcc", str(nvcc), "--arch", "sm_100a", "--arch", "sm_103a")

    assert code == 0, report
    (record,) = report["reports"]
    copy = Path(record["copy"])
    assert copy == (tmp_path / "work" / "build" / "cuda-13.4.92" / "cake_synth_a" / "vendored_cake_synth_a").resolve()
    manifest = json.loads((copy / "build.json").read_text())
    commands = [
        check_standalone.compile_command(str(nvcc), copy, manifest, entry, (13, 4), environment.includes)
        for entry in manifest["libraries"]
    ]
    assert fake_nvcc.calls(nvcc) == [command[1:] for command in commands]
    assert all("-DSYNTHETIC_NEWER" in command for command in commands)
    assert (copy / "aot" / "build_info.json").read_bytes() == BUILD_INFO
    assert check_standalone.render_build_info("13.4.92", 13040, ["sm_103a", "sm_100a", "sm_103a"]) == BUILD_INFO
    assert record["load"] is None
    assert not (repo / check_standalone.VENDOR / "cake_synth_a" / "aot").exists()


def test_build_compiles_a_sibling_once_and_places_its_copy_beside_the_package(tmp_path, environment):
    repo = _tree(
        tmp_path / "repo",
        {
            "cake_synth_a": {},
            "cake_synth_kcp": {"build.json": json.dumps(_manifest("cake_synth_kcp", siblings=["cake_synth_a"]))},
        },
    )
    nvcc = fake_nvcc.write(tmp_path / "cuda" / "bin" / "nvcc", "13.4.92")

    code, report = _build(repo, tmp_path / "work", "--nvcc", str(nvcc), "--arch=sm_100a", "--package=cake_synth_kcp")

    assert code == 0, report
    copies = {record["package"]: Path(record["copy"]) for record in report["reports"]}
    assert sorted(copies) == ["cake_synth_a", "cake_synth_kcp"]
    assert len(fake_nvcc.calls(nvcc)) == 2
    slot = copies["cake_synth_kcp"].parent / "cake_synth_a"
    assert copies["cake_synth_kcp"].parent.name == "vendored_cake_synth_kcp"
    assert (slot / "aot" / "prog_sm_100a.so").read_bytes() == b"synthetic library"
    info = copies["cake_synth_a"] / "aot" / "build_info.json"
    assert (slot / "aot" / "build_info.json").read_bytes() == info.read_bytes()


def test_build_takes_another_toolkit_for_an_architecture(tmp_path, environment):
    manifest = _manifest("cake_synth_a", ("sm_100a", "sm_107a"))
    repo = _tree(tmp_path / "repo", {"cake_synth_a": {"build.json": json.dumps(manifest)}})
    older = fake_nvcc.write(tmp_path / "cuda-13.4" / "bin" / "nvcc", "13.4.92")
    newer = fake_nvcc.write(tmp_path / "cuda-13.5" / "bin" / "nvcc", "13.5.20")

    code, report = _build(repo, tmp_path / "work", "--nvcc", str(older), "--nvcc-for", f"sm_107a={newer}")

    assert code == 0, report
    infos = {
        record["toolkit"]: json.loads((Path(record["copy"]) / "aot" / "build_info.json").read_text())
        for record in report["reports"]
    }
    assert infos["13.4.92"]["arches"] == ["sm_100a"] and infos["13.4.92"]["cudart_version"] == 13040
    assert infos["13.5.20"]["arches"] == ["sm_107a"] and infos["13.5.20"]["cudart_version"] == 13050
    assert [call[-1].rpartition("/")[2] for call in fake_nvcc.calls(older)] == ["prog_sm_100a.so"]
    assert [call[-1].rpartition("/")[2] for call in fake_nvcc.calls(newer)] == ["prog_sm_107a.so"]


def test_a_failed_compile_is_a_finding(tmp_path, environment):
    repo = _tree(tmp_path / "repo", {"cake_synth_a": {}})
    nvcc = fake_nvcc.write(tmp_path / "cuda" / "bin" / "nvcc", "13.4.92")
    nvcc.with_name("fail").touch()

    code, report = _build(repo, tmp_path / "work", "--nvcc", str(nvcc), "--arch", "sm_100a")

    assert code == 1
    (finding,) = report["findings"]
    assert finding["subject"] == "compile" and "synthetic failure" in finding["message"]
    assert not (Path(report["reports"][0]["copy"]) / "aot" / "build_info.json").exists()


@pytest.mark.parametrize(
    ("toolkit", "min_cuda", "args", "error"),
    [
        ("13.4.92", "13.5", [], "entries above their toolkit"),
        ("13.3.73", "13.0", [], "builds with CUDA 13.4 or newer"),
        ("13.4.92", "13.4", ["--arch", "sm_100a", "--nvcc-for", "sm_107a=nvcc"], "which this run does not build"),
        ("13.4.92", "13.4", ["inside"], "inside the repository"),
        ("13.4.92", "13", [], "malformed library entry"),
    ],
)
def test_build_refuses_rather_than_skips(tmp_path, capsys, toolkit, min_cuda, args, error):
    """Every refusal comes before the probe child: these need neither torch nor tvm_ffi."""
    repo = _tree(tmp_path / "repo", {"cake_synth_a": {"build.json": json.dumps(_manifest("a", min_cuda=min_cuda))}})
    nvcc = fake_nvcc.write(tmp_path / "cuda" / "bin" / "nvcc", toolkit)
    work = repo / "work" if args == ["inside"] else tmp_path / "work"

    code, report = _build(repo, work, "--nvcc", str(nvcc), *(args if args != ["inside"] else []))

    assert code == 2 and report["exit"] == 2
    assert error in report["error"] and error in capsys.readouterr().err
    assert fake_nvcc.calls(nvcc) == []


def test_build_needs_the_default_nvcc_only_for_an_architecture_without_nvcc_for(tmp_path, environment, monkeypatch):
    manifest = _manifest("cake_synth_a", ("sm_107a",))
    repo = _tree(tmp_path / "repo", {"cake_synth_a": {"build.json": json.dumps(manifest)}})
    newer = fake_nvcc.write(tmp_path / "cuda-13.5" / "bin" / "nvcc", "13.5.20")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "no-cuda"))
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))

    code, report = _build(repo, tmp_path / "work", "--arch", "sm_107a", "--nvcc-for", f"sm_107a={newer}")

    assert code == 0, report
    assert [record["toolkit"] for record in report["reports"]] == ["13.5.20"]


def test_find_nvcc_passes_over_a_cuda_home_without_nvcc(tmp_path, monkeypatch):
    nvcc = fake_nvcc.write(tmp_path / "bin" / "nvcc", "13.4.92")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "no-cuda"))
    monkeypatch.setenv("PATH", str(nvcc.parent))

    assert check_standalone.find_nvcc(None) == os.path.abspath(nvcc)
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    with pytest.raises(check_standalone.GateError, match="no nvcc"):
        check_standalone.find_nvcc(None)


def _nvcc() -> str | None:
    try:
        return check_standalone.find_nvcc(None)
    except check_standalone.GateError:
        return None


@requires_arch("sm_100a", "sm_103a", "sm_107a")
@pytest.mark.skipif(_nvcc() is None, reason="needs nvcc")
@pytest.mark.parametrize("name", NAMES)
def test_every_library_builds_into_a_copy_and_loads_from_it(name, tmp_path):
    """Every library of the device's architecture, compiled with plain nvcc from build.json alone and loaded through
    cake_build from a blocked copy."""
    report = tmp_path / "report.json"
    argv = ["build", "--package", name, "--arch", loader.device_arch(), "--work-dir", str(tmp_path / "work")]

    code = check_standalone.main([*argv, "--report-json", str(report)])

    document = json.loads(report.read_text())
    findings = "\n".join(f"{finding['package']}: {finding['message']}" for finding in document["findings"])
    assert code == 0, document["error"] or findings
    loads = [load for record in document["reports"] if record["load"] for load in record["load"]["loads"]]
    assert loads and all(load["ok"] for load in loads)
