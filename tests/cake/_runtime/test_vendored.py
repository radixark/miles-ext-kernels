"""The vendored tree matches ``VENDORED.json``, and tools/vendor_cake.py writes and checks it (torch-free; needs git).

The tool runs here on synthetic FlashInfer and miles repositories built in a temporary directory, into a scratch
checkout of this repository's layout.
"""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from miles_ext_kernels.cake._runtime import registry

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = REPO_ROOT / "tools" / "templates"

_spec = importlib.util.spec_from_file_location("vendor_cake", REPO_ROOT / "tools" / "vendor_cake.py")
vendor_cake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vendor_cake)

FAST = "flashinfer/experimental/cake/cake_synth_fast"
DET = "flashinfer/cake_synth_det_backend"
KCP = "flashinfer/experimental/cake_synth_fast_kcp"
PR_PATH = "miles/kernels/attention/synth/cake"
GIT = ["git", "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.com", "-c", "commit.gpgsign=false"]


def test_the_vendored_tree_matches_its_manifest():
    assert vendor_cake.check(REPO_ROOT) == []


def test_every_vendored_op_is_registered():
    manifest = json.loads((REPO_ROOT / vendor_cake.VENDOR / "VENDORED.json").read_text())

    assert set(manifest["ops"]) <= set(registry.OPS)


def test_the_templates_are_the_loader_and_the_host_surface():
    assert vendor_cake.git_blob_sha1((TEMPLATES / "cake_build.py").read_bytes()) == vendor_cake.CAKE_BUILD_BLOB
    shim = (TEMPLATES / "cake_host_shim.h").read_text()
    for name in ("TVMFFIEnvGetStream", "TVM_FFI_THROW", "TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_cuda_version"):
        assert name in shim
    includes = [line.split(None, 1)[1] for line in shim.splitlines() if line.startswith("#include")]
    assert "<tvm/ffi/function.h>" in includes
    assert not [path for path in includes if path.lstrip('<"').startswith(("ATen", "c10", "torch", "pybind11"))]


# --- synthetic upstream repositories ---


def _commit_tree(root: Path, files: dict[str, bytes]) -> str:
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        # let a shallow, tree-only fetch of any commit through file://, as GitHub serves it
        subprocess.run(["git", "-C", str(root), "config", "uploadpack.allowFilter", "true"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "uploadpack.allowAnySHA1InWant", "true"], check=True)
    subprocess.run([*GIT, "-C", str(root), "add", "-A", "-f", "."], check=True)
    subprocess.run([*GIT, "-C", str(root), "commit", "-q", "--allow-empty", "-m", "synthetic"], check=True)
    return _rev(root)


def _rev(root: Path, path: str = "") -> str:
    target = f"HEAD:{path}" if path else "HEAD"
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", target], check=True, capture_output=True, text=True
    ).stdout.strip()


def _fi_package(revision: str, extra: dict[str, bytes] | None = None) -> dict[str, bytes]:
    kernel = b"// kernel\n"
    libraries = [
        {
            "arch": arch,
            "closure_sha256": "0" * 64,
            "entry": "run",
            "min_cuda": "12.9",
            "name": f"prog_{arch}",
            "nvcc_flags": [],
            "nvcc_flags_if": [],
            "path": f"aot/prog_{arch}.so",
            "sha256": {"csrc/prog.cu": hashlib.sha256(kernel).hexdigest()},
            "sources": ["csrc/prog.cu"],
        }
        for arch in vendor_cake.REQUIRED_ARCHES
    ]
    build = {"schema": "cake-build/1", "generator": {"cake_revision": revision}, "arch_flags": {}, "common": {}}
    return {
        "__init__.py": b"from . import cake_build\n",
        "cake_build.py": (TEMPLATES / "cake_build.py").read_bytes(),
        "build.json": json.dumps({**build, "libraries": libraries}).encode(),
        "csrc/prog.cu": kernel,
        ".gitignore": b"aot/\n",
        "aot/prog_sm_100a.so": b"a prebuilt library is never vendored",
        **(extra or {}),
    }


@pytest.fixture
def flashinfer(tmp_path) -> Path:
    """A FlashInfer repository with Cake packages at its three positions; the KCP one imports ``..cake.<name>``."""
    root = tmp_path / "flashinfer"
    kcp = {"kcp.py": b"from ..cake.cake_synth_fast import cake_build\n"}
    files = {}
    for fi_path, revision, extra in ((FAST, "f" * 40, None), (DET, "d" * 40, None), (KCP, "k" * 40, kcp)):
        files.update({f"{fi_path}/{relative}": data for relative, data in _fi_package(revision, extra).items()})
    files["flashinfer/experimental/cake/cake_synth_fast_other.py"] = b"# not vendored\n"
    _commit_tree(root, files)
    return root


REGISTRY = {
    "fwd": {
        "name": "cake_synth_fwd_0a1b",
        "sources": ["cake_synth_fwd_0a1b_kernel.cu", "cake_synth_fwd_0a1b_binding.cu"],
        "ffi_entry": "run",
        "compile_flags": [],
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["grid", "grid_x"]],
    },
    "bwd": {
        "ffi_entry": "run",
        "compile_flags": ["-DSYNTH_BWD=1"],
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "dx"], ["grid", "grid_x"]],
        "per_arch": {
            "sm_100a": {"name": "cake_synth_bwd_100", "sources": ["cake_synth_bwd_100_kernel.cu"]},
            "sm_103a": {"name": "cake_synth_bwd_103", "sources": ["cake_synth_bwd_103_kernel.cu"]},
        },
    },
}


@pytest.fixture
def miles(tmp_path) -> Path:
    """A miles repository holding generated sources: a registry, kernels, launchers and a torch host surface."""
    root = tmp_path / "miles"
    sources = {source for record in REGISTRY.values() for source in record.get("sources", [])}
    sources |= {
        source
        for record in REGISTRY.values()
        for per in record.get("per_arch", {}).values()
        for source in per["sources"]
    }
    files = {f"{PR_PATH}/csrc/{source}": f"// {source}\n".encode() for source in sorted(sources)}
    files[f"{PR_PATH}/csrc/cake_host_shim.h"] = b"#include <torch/extension.h>\n"
    files[f"{PR_PATH}/registry.json"] = json.dumps(REGISTRY, indent=1).encode()
    files[f"{PR_PATH}/__init__.py"] = b"# host code, not vendored\n"
    _commit_tree(root, files)
    return root


@pytest.fixture
def ext(tmp_path) -> Path:
    """A scratch checkout of this repository's layout with nothing vendored."""
    root = tmp_path / "ext"
    vendor = root / vendor_cake.VENDOR
    vendor.mkdir(parents=True)
    (vendor / "__init__.py").write_bytes(b"")
    (vendor / "VENDORED.json").write_text(vendor_cake.canonical({"ops": {}, "schema": vendor_cake.SCHEMA}))
    return root


def _vendor_fi(flashinfer, ext, op="synth", paths=(FAST, DET), **kwargs):
    return vendor_cake.vendor_fi(flashinfer, "HEAD", op, list(paths), repo_root=ext, **kwargs)


def _vendor_pr(miles, ext, **kwargs):
    arguments = dict(
        pr=1234,
        op="synth",
        name="cake_synth_sparse_det",
        path=PR_PATH,
        include=["csrc", "registry.json"],
        exclude=["csrc/cake_host_shim.h"],
        arch_from={"sm_107a": "sm_103a"},
        repo_root=ext,
    )
    arguments.update(kwargs)
    return vendor_cake.vendor_miles_pr(miles, "HEAD", **arguments)


def _manifest(ext: Path) -> dict:
    return json.loads((ext / vendor_cake.VENDOR / "VENDORED.json").read_text())


def _write_manifest(ext: Path, manifest: dict) -> None:
    (ext / vendor_cake.VENDOR / "VENDORED.json").write_text(vendor_cake.canonical(manifest))


def _rewrite(ext: Path, package: str, relative: str, data: bytes) -> None:
    """Change a vendored file and record its new blob, so only the rule under test sees the change."""
    (ext / vendor_cake.VENDOR / package / relative).write_bytes(data)
    manifest = _manifest(ext)
    for block in manifest["ops"].values():
        if package in block:
            entry = block[package]
            entry["local" if relative in entry.get("local", {}) else "files"][relative] = vendor_cake.git_blob_sha1(
                data
            )
    _write_manifest(ext, manifest)


# --- FlashInfer packages ---


def test_fi_packages_are_vendored_byte_identically_one_level_deep(flashinfer, ext):
    entries = _vendor_fi(flashinfer, ext)
    vendor = ext / vendor_cake.VENDOR

    assert sorted(entries) == ["cake_synth_det_backend", "cake_synth_fast"]
    for fi_path, revision in ((FAST, "f" * 40), (DET, "d" * 40)):
        package = fi_path.rsplit("/", 1)[1]
        entry = _manifest(ext)["ops"]["synth"][package]
        assert entry["source"] == {
            "commit": _rev(flashinfer),
            "kind": "flashinfer",
            "path": fi_path,
            "repo": vendor_cake.FI_REPO,
        }
        assert entry["cake_revision"] == revision
        assert sorted(entry["files"]) == sorted(name for name in _fi_package(revision) if not name.startswith("aot/"))
        for relative, blob in entry["files"].items():
            assert blob == _rev(flashinfer, f"{fi_path}/{relative}")
            assert (vendor / package / relative).read_bytes() == (flashinfer / fi_path / relative).read_bytes()
    assert sorted(path.name for path in vendor.iterdir()) == [
        "VENDORED.json",
        "__init__.py",
        "cake_synth_det_backend",
        "cake_synth_fast",
    ]
    assert not list(vendor.rglob("aot"))
    assert vendor_cake.check(ext) == []


def test_fi_package_names_follow_flashinfer_positions():
    assert vendor_cake.fi_package_name(FAST) == "cake_synth_fast"
    assert vendor_cake.fi_package_name("flashinfer/experimental/cake_kda_det_train/") == "cake_kda_det_train"
    for bad in ("flashinfer/jit/cake_synth", "flashinfer/experimental/cake"):
        with pytest.raises(ValueError, match="expected a package directly under"):
            vendor_cake.fi_package_name(bad)


def test_a_vendored_module_imports_packages_of_its_own_op_only(flashinfer, ext):
    _vendor_fi(flashinfer, ext, paths=(FAST, KCP))  # kcp.py: from ..cake.cake_synth_fast import cake_build
    finding = "_vendor/cake_synth_fast_kcp/kcp.py: imports the vendored package cake_synth_fast, "

    assert vendor_cake.check(ext) == []
    assert not (ext / vendor_cake.VENDOR / "cake").exists()  # the runtime resolves ..cake.<name>; no file lists it
    vendor_cake.remove("synth", "cake_synth_fast", repo_root=ext)
    assert vendor_cake.check(ext) == [finding + "not vendored"]
    _vendor_fi(flashinfer, ext, op="other", paths=(FAST,))
    assert vendor_cake.check(ext) == [finding + "which belongs to other; an op imports its own packages only"]


@pytest.mark.parametrize(
    ("source", "names"),
    [
        ("from ..cake.cake_a import x\n", {"cake_a"}),
        ("from ..cake.cake_a.cake_jit import x\n", {"cake_a"}),
        ("from ..cake import cake_a, cake_b\n", {"cake_a", "cake_b"}),
        ("from ..cake_a.cake_jit import x\n", {"cake_a"}),
        ("from .. import cake_a as det\n", {"cake_a"}),
        ("def f():\n    from ..cake_a import x\n", {"cake_a"}),
        ("from . import cake_jit\nfrom .cake_jit import x\nimport torch\n", set()),
        ("from ...jit import x\n", {".."}),
    ],
)
def test_the_package_imports_of_a_module(source, names):
    assert vendor_cake.package_imports(source.encode(), depth=2) == names


def test_vendoring_again_rewrites_the_package_and_keeps_the_others_of_the_op(flashinfer, ext):
    _vendor_fi(flashinfer, ext)
    stale = ext / vendor_cake.VENDOR / "cake_synth_fast" / "stale.py"
    stale.write_bytes(b"")
    _vendor_fi(flashinfer, ext, paths=(FAST,))

    assert not stale.exists()
    assert sorted(_manifest(ext)["ops"]["synth"]) == ["cake_synth_det_backend", "cake_synth_fast"]
    assert vendor_cake.check(ext) == []


def test_a_package_belongs_to_one_op(flashinfer, ext):
    _vendor_fi(flashinfer, ext)

    with pytest.raises(ValueError, match="cake_synth_fast is vendored for synth"):
        _vendor_fi(flashinfer, ext, op="other", paths=(FAST,))
    assert sorted(_manifest(ext)["ops"]) == ["synth"]


def test_the_fi_packages_of_an_op_come_from_one_commit(flashinfer, ext):
    _vendor_fi(flashinfer, ext, paths=(FAST,))
    _commit_tree(flashinfer, {"README.md": b"a later commit\n"})
    _vendor_fi(flashinfer, ext, paths=(DET,))

    assert vendor_cake.check(ext) == ["VENDORED.json: the FlashInfer packages of synth come from 2 commits"]


@pytest.mark.parametrize(
    ("tamper", "finding"),
    [
        (lambda v: (v / "cake_synth_fast/csrc/prog.cu").write_bytes(b"// edited\n"), "csrc/prog.cu: content differs"),
        (lambda v: (v / "cake_synth_fast/extra.py").write_bytes(b""), "extra.py: not listed in VENDORED.json"),
        (lambda v: (v / "cake_synth_det_backend/csrc/prog.cu").unlink(), "csrc/prog.cu: missing"),
        (lambda v: (v / "cake_synth_fast/aot").mkdir(), "aot/ directory"),
        (lambda v: (v / "__init__.py").write_bytes(b"x = 1\n"), "_vendor/__init__.py: must be empty"),
        (lambda v: (v / "stray").mkdir(), "_vendor/stray: not listed in VENDORED.json"),
        (lambda v: (v / "cake").mkdir(), "_vendor/cake: not listed in VENDORED.json"),
        (lambda v: shutil.rmtree(v / "cake_synth_fast"), "_vendor/cake_synth_fast: missing"),
        (
            lambda v: (v / "VENDORED.json").write_text(json.dumps(json.loads((v / "VENDORED.json").read_text()))),
            "form",
        ),
        (lambda v: (v / "VENDORED.json").unlink(), "VENDORED.json: "),
    ],
)
def test_the_checker_reports_a_changed_tree(flashinfer, ext, tamper, finding):
    _vendor_fi(flashinfer, ext)
    tamper(ext / vendor_cake.VENDOR)

    findings = vendor_cake.check(ext)
    assert any(finding in problem for problem in findings), findings


def test_the_checker_reports_a_source_that_differs_from_build_json(flashinfer, ext):
    _vendor_fi(flashinfer, ext)
    _rewrite(ext, "cake_synth_fast", "csrc/prog.cu", b"// edited\n")

    assert vendor_cake.check(ext) == [
        f"_vendor/cake_synth_fast/csrc/prog.cu: sha256 differs from build.json (prog_{arch})"
        for arch in vendor_cake.REQUIRED_ARCHES
    ]


@pytest.mark.parametrize(
    ("edit", "finding"),
    [
        (lambda build: build.update(schema="cake-build/2"), "schema 'cake-build/2'"),
        (lambda build: build["libraries"].pop(), "no libraries for sm_107a"),
        (lambda build: build["generator"].update(cake_revision="e" * 40), "cake_revision"),
    ],
)
def test_the_checker_reports_a_manifest_it_cannot_serve(flashinfer, ext, edit, finding):
    _vendor_fi(flashinfer, ext)
    path = ext / vendor_cake.VENDOR / "cake_synth_det_backend" / "build.json"
    build = json.loads(path.read_text())
    edit(build)
    _rewrite(ext, "cake_synth_det_backend", "build.json", json.dumps(build).encode())

    findings = vendor_cake.check(ext)
    assert len(findings) == 1 and finding in findings[0], findings


def test_the_checker_pins_the_flashinfer_loader(flashinfer, ext):
    _vendor_fi(flashinfer, ext)
    _rewrite(ext, "cake_synth_fast", "cake_build.py", b"# another loader\n")

    assert vendor_cake.check(ext) == [
        f"_vendor/cake_synth_fast/cake_build.py: blob {vendor_cake.git_blob_sha1(b'# another loader' + bytes([10]))} "
        f"is not the loader {vendor_cake.CAKE_BUILD_BLOB}"
    ]


@pytest.mark.parametrize(
    ("source", "allow", "finding"),
    [
        ("import flashinfer.jit\n", {}, "imports flashinfer.jit"),
        ("from flashinfer.api_logging import flashinfer_api\n", {}, "imports flashinfer.api_logging"),
        ("import os\nflag = os.environ.get('CAKE_KNOB')\n", {}, "reads the environment (os.environ)"),
        ("import os\nflag = os.getenv('CAKE_KNOB')\n", {}, "reads the environment (os.getenv)"),
        ("from os import environ\n", {}, "reads the environment (from os import environ)"),
        ("import os\nflag = os.environ.get('CAKE_KNOB')\n", {"env_read": True}, None),
        ("def f():\n    import flashinfer.jit.cpp_ext\n", {"flashinfer_hooks": ["f"]}, None),
        ("class C:\n    def f(self):\n        import flashinfer\n", {"flashinfer_hooks": ["C.f"]}, None),
        ("def f():\n    pass\ndef g():\n    import flashinfer\n", {"flashinfer_hooks": ["f"]}, "imports flashinfer"),
        ("def f():\n    import flashinfer.jit.cpp_ext\n", {}, "imports flashinfer.jit.cpp_ext"),
        ("def f():\n    pass\nimport flashinfer.jit\n", {"flashinfer_hooks": ["f"]}, "imports flashinfer.jit"),
        ("from .. import cake_rmsnorm_det_train\nimport torch\nimport tvm_ffi\n", {}, None),
    ],
)
def test_the_checker_finds_flashinfer_imports_and_environment_reads(source, allow, finding):
    findings = vendor_cake.python_findings(source.encode(), "module.py", **allow)

    assert findings == ([] if finding is None else [f"module.py:{len(source.splitlines())}: {finding}"])


def test_a_flashinfer_hook_must_be_a_function_of_the_module():
    source = b"def f():\n    import flashinfer\n"

    assert vendor_cake.python_findings(source, "module.py", flashinfer_hooks=["f", "g"]) == [
        "module.py: allow names the function g, which the module does not define"
    ]


def test_allowed_modules_are_recorded_per_package(tmp_path, ext):
    flashinfer = tmp_path / "fi-env"
    extra = {"kcp.py": b"import os\n\ndef knob():\n    import flashinfer.jit\n    return os.environ.get('K')\n"}
    _commit_tree(flashinfer, {f"{DET}/{relative}": data for relative, data in _fi_package("d" * 40, extra).items()})

    _vendor_fi(flashinfer, ext, paths=(DET,))
    assert len(vendor_cake.check(ext)) == 2
    package = "cake_synth_det_backend"
    allow = {"allow_env_read": [f"{package}:kcp.py"], "allow_flashinfer_import": [f"{package}:kcp.py:knob"]}
    _vendor_fi(flashinfer, ext, paths=(DET,), **allow)
    assert _manifest(ext)["ops"]["synth"][package]["allow"] == {
        "env_read": ["kcp.py"],
        "flashinfer_import": ["kcp.py:knob"],
    }
    assert vendor_cake.check(ext) == []
    _vendor_fi(flashinfer, ext, paths=(DET,), allow_flashinfer_import=[f"{package}:kcp.py:other"])
    assert vendor_cake.check(ext) == [
        f"_vendor/{package}/kcp.py: allow names the function other, which the module does not define",
        f"_vendor/{package}/kcp.py:4: imports flashinfer.jit",
        f"_vendor/{package}/kcp.py:5: reads the environment (os.environ)",
    ]
    with pytest.raises(ValueError, match="--allow-env-read cake_synth_det_backend:nope.py: expected PKG:MODULE"):
        _vendor_fi(flashinfer, ext, paths=(DET,), allow_env_read=[f"{package}:nope.py"])
    with pytest.raises(ValueError, match=f"--allow-flashinfer-import {package}:kcp.py: expected PKG:MODULE:FUNCTION"):
        _vendor_fi(flashinfer, ext, paths=(DET,), allow_flashinfer_import=[f"{package}:kcp.py"])


# --- generated sources of a miles pull request ---


def test_miles_pr_sources_are_vendored_with_the_local_build_files(miles, ext):
    entry = _vendor_pr(miles, ext)
    package = ext / vendor_cake.VENDOR / "cake_synth_sparse_det"

    assert entry["source"] == {
        "commit": _rev(miles),
        "exclude": {"csrc/cake_host_shim.h": _rev(miles, f"{PR_PATH}/csrc/cake_host_shim.h")},
        "include": ["csrc", "registry.json"],
        "kind": "miles-pr",
        "path": PR_PATH,
        "pr": 1234,
        "repo": vendor_cake.MILES_REPO,
    }
    assert entry["cake_revision"] is None
    assert sorted(entry["files"]) == sorted(
        [
            "registry.json",
            *(
                f"csrc/{name}"
                for name in (
                    "cake_synth_fwd_0a1b_kernel.cu",
                    "cake_synth_fwd_0a1b_binding.cu",
                    "cake_synth_bwd_100_kernel.cu",
                    "cake_synth_bwd_103_kernel.cu",
                )
            ),
        ]
    )
    for relative, blob in entry["files"].items():
        assert blob == _rev(miles, f"{PR_PATH}/{relative}")
        assert (package / relative).read_bytes() == (miles / PR_PATH / relative).read_bytes()
    assert (package / "__init__.py").read_bytes() == b""
    assert (package / "cake_build.py").read_bytes() == (TEMPLATES / "cake_build.py").read_bytes()
    assert (package / "csrc" / "cake_host_shim.h").read_bytes() == (TEMPLATES / "cake_host_shim.h").read_bytes()
    assert entry["local"] == {
        relative: vendor_cake.git_blob_sha1((package / relative).read_bytes())
        for relative in ("__init__.py", "build.json", "cake_build.py", "csrc/cake_host_shim.h")
    }
    assert entry["build"] == {
        "arch_from": {"sm_107a": "sm_103a"},
        "link_flags": list(vendor_cake.DEFAULT_LINK_FLAGS),
        "nvcc_flags": list(vendor_cake.DEFAULT_NVCC_FLAGS),
        "registry": "registry.json",
    }
    assert vendor_cake.check(ext) == []


def test_the_generated_build_json_has_one_library_per_stage_and_architecture(miles, ext):
    _vendor_pr(miles, ext)
    package = ext / vendor_cake.VENDOR / "cake_synth_sparse_det"
    build = json.loads((package / "build.json").read_text())

    assert build["schema"] == "cake-build/1" and build["generator"] == {"cake_revision": None, "from": "registry.json"}
    assert build["package"] == "miles_ext_kernels.cake._vendor.cake_synth_sparse_det"
    assert build["common"] == {
        "include_dirs": ["csrc"],
        "link_flags": ["-shared", "-lcudart", "-lcuda"],
        "nvcc_flags": ["-std=c++17", "-O3", "--compiler-options=-fPIC", "--expt-relaxed-constexpr"],
        "nvcc_flags_if": [],
    }
    assert [(library["name"], library["arch"], library["min_cuda"]) for library in build["libraries"]] == [
        ("cake_synth_bwd_100_sm_100a", "sm_100a", "12.8"),
        ("cake_synth_bwd_103_sm_103a", "sm_103a", "12.9"),
        ("cake_synth_bwd_103_sm_107a", "sm_107a", "13.4"),  # the SM103a record, built for compute_107a
        ("cake_synth_fwd_0a1b_sm_100a", "sm_100a", "12.8"),
        ("cake_synth_fwd_0a1b_sm_103a", "sm_103a", "12.9"),
        ("cake_synth_fwd_0a1b_sm_107a", "sm_107a", "13.4"),
    ]
    bwd = build["libraries"][2]
    assert bwd["gencode"] == "-gencode=arch=compute_107a,code=sm_107a"
    assert bwd["sources"] == ["csrc/cake_synth_bwd_103_kernel.cu"] and bwd["nvcc_flags"] == ["-DSYNTH_BWD=1"]
    assert bwd["sha256"] == {
        "csrc/cake_synth_bwd_103_kernel.cu": hashlib.sha256(b"// cake_synth_bwd_103_kernel.cu\n").hexdigest()
    }
    assert bwd["entry"] == "run" and bwd["path"] == "aot/cake_synth_bwd_103_sm_107a.so"
    # the closure covers the host surface: another shim names other libraries
    files = {path.relative_to(package).as_posix(): path.read_bytes() for path in package.rglob("*") if path.is_file()}
    again = json.loads(
        vendor_cake.render_build_json(
            "cake_synth_sparse_det", files, _manifest(ext)["ops"]["synth"]["cake_synth_sparse_det"]["build"]
        )
    )
    assert again == build
    files["csrc/cake_host_shim.h"] += b"// changed\n"
    changed = json.loads(
        vendor_cake.render_build_json(
            "cake_synth_sparse_det", files, _manifest(ext)["ops"]["synth"]["cake_synth_sparse_det"]["build"]
        )
    )
    assert all(a["closure_sha256"] != b["closure_sha256"] for a, b in zip(build["libraries"], changed["libraries"]))


@pytest.mark.parametrize(
    ("relative", "data", "finding"),
    [
        ("build.json", b"{}\n", "build.json: not the file this tool generates"),
        ("csrc/cake_host_shim.h", b"// edited\n", "csrc/cake_host_shim.h: not the file this tool generates"),
        ("__init__.py", b"x = 1\n", "__init__.py: not the file this tool generates"),
    ],
)
def test_the_checker_regenerates_the_local_files(miles, ext, relative, data, finding):
    _vendor_pr(miles, ext)
    _rewrite(ext, "cake_synth_sparse_det", relative, data)

    findings = vendor_cake.check(ext)
    assert any(finding in problem for problem in findings), findings


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"exclude": []}, r"upstream \['csrc/cake_host_shim.h'\] would replace the generated local files"),
        ({"name": "cake_other_det"}, "a miles-pr package of synth is named cake_synth_<purpose>"),
        ({"include": ["csrc"], "exclude": ["csrc/cake_host_shim.h"]}, "the included files hold no registry.json"),
        ({"include": ["nope"]}, "--include nope: no such file or directory upstream"),
        ({"exclude": ["csrc/cake_host_shim.h", "README.md"]}, "--exclude README.md: not an included upstream file"),
        ({"arch_from": {"sm_90a": "sm_103a"}}, r"--arch-from names unknown architectures \['sm_90a'\]"),
    ],
)
def test_miles_pr_vendoring_rejects(miles, ext, changes, error):
    with pytest.raises(ValueError, match=error):
        _vendor_pr(miles, ext, **changes)
    assert _manifest(ext)["ops"] == {}


def test_one_manifest_holds_both_kinds_and_remove_deletes_a_package(flashinfer, miles, ext):
    _vendor_fi(flashinfer, ext)
    _vendor_pr(miles, ext)

    assert sorted(_manifest(ext)["ops"]["synth"]) == [
        "cake_synth_det_backend",
        "cake_synth_fast",
        "cake_synth_sparse_det",
    ]
    assert vendor_cake.check(ext) == []
    vendor_cake.remove("synth", "cake_synth_sparse_det", repo_root=ext)
    assert not (ext / vendor_cake.VENDOR / "cake_synth_sparse_det").exists()
    assert vendor_cake.check(ext) == []
    for name in ("cake_synth_det_backend", "cake_synth_fast"):
        vendor_cake.remove("synth", name, repo_root=ext)
    assert _manifest(ext) == {"ops": {}, "schema": vendor_cake.SCHEMA}
    with pytest.raises(ValueError, match="synth vendors no package cake_synth_fast"):
        vendor_cake.remove("synth", "cake_synth_fast", repo_root=ext)


# --- merging the manifest ---


def _doc(ops: dict) -> dict:
    return {"ops": ops, "schema": vendor_cake.SCHEMA}


def _pkg(commit: str) -> dict:
    """A package entry; only its identity matters to the merge."""
    return {"cake_revision": None, "files": {"a.py": "0" * 40}, "source": {"commit": commit * 40}}


def test_merge_takes_each_package_from_the_side_that_changed_it():
    base = _doc({"a": {"p": _pkg("1"), "q": _pkg("1")}, "b": {"r": _pkg("1")}})
    ours = _doc({"a": {"p": _pkg("2"), "q": _pkg("1")}, "b": {"r": _pkg("1")}, "c": {"s": _pkg("1")}})
    theirs = _doc({"a": {"p": _pkg("1"), "q": _pkg("3")}, "d": {"t": _pkg("1")}, "c": {"s": _pkg("1")}})

    assert vendor_cake.merge(base, ours, theirs) == (
        _doc({"a": {"p": _pkg("2"), "q": _pkg("3")}, "c": {"s": _pkg("1")}, "d": {"t": _pkg("1")}}),
        [],
    )


def test_merge_reports_two_different_changes_and_a_package_given_to_two_ops():
    base = _doc({"a": {"p": _pkg("1")}})
    ours = _doc({"a": {"p": _pkg("2")}, "b": {"x": _pkg("1")}})
    theirs = _doc({"a": {"p": _pkg("3")}, "c": {"x": _pkg("1")}})

    assert vendor_cake.merge(base, ours, theirs)[1] == [
        "ops.a.p: changed differently on both sides",
        "x: vendored for b and c",
    ]
    with pytest.raises(ValueError, match="theirs is not a miles-ext-kernels-vendored/1 manifest"):
        vendor_cake.merge(base, ours, {"ops": {}})


def test_the_merge_driver_merges_two_layers_that_each_add_an_op(tmp_path):
    """Two branches each append an op after the same one: a text merge conflicts, the driver does not."""
    repo = tmp_path / "repo"
    manifest = repo / vendor_cake.VENDOR / "VENDORED.json"
    manifest.parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / ".gitattributes", repo / ".gitattributes")
    git = [*GIT, "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "symbolic-ref", "HEAD", "refs/heads/main"], check=True)
    driver = f"{sys.executable} {REPO_ROOT / 'tools' / 'vendor_cake.py'} merge %O %A %B"
    subprocess.run([*git, "config", "merge.vendored.driver", driver], check=True)

    def commit(ops):
        manifest.write_text(vendor_cake.canonical(_doc(ops)))
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-q", "-m", "layer"], check=True)

    commit({"a": {"p": _pkg("1")}})
    subprocess.run([*git, "checkout", "-q", "-b", "other"], check=True)
    commit({"a": {"p": _pkg("1")}, "c": {"r": _pkg("3")}})
    subprocess.run([*git, "checkout", "-q", "main"], check=True)
    commit({"a": {"p": _pkg("1")}, "b": {"q": _pkg("2")}})

    result = subprocess.run([*git, "merge", "-q", "--no-edit", "other"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert manifest.read_text() == vendor_cake.canonical(
        _doc({"a": {"p": _pkg("1")}, "b": {"q": _pkg("2")}, "c": {"r": _pkg("3")}})
    )


def test_the_merge_command_writes_ours_or_reports_the_conflicts(tmp_path, capsys):
    base, ours, theirs = (tmp_path / name for name in ("base", "ours", "theirs"))
    base.write_text("")  # no common ancestor
    ours.write_text(vendor_cake.canonical(_doc({"a": {"p": _pkg("1")}})))
    theirs.write_text(json.dumps(_doc({"b": {"q": _pkg("2")}})))

    assert vendor_cake.main(["merge", str(base), str(ours), str(theirs)]) == 0
    assert ours.read_text() == vendor_cake.canonical(_doc({"a": {"p": _pkg("1")}, "b": {"q": _pkg("2")}}))
    theirs.write_text(json.dumps(_doc({"a": {"p": _pkg("3")}})))
    before = ours.read_text()
    assert vendor_cake.main(["merge", str(base), str(ours), str(theirs)]) == 1
    assert ours.read_text() == before
    assert "VENDORED.json merge conflict: ops.a.p: changed differently on both sides" in capsys.readouterr().err


# --- upstream drift ---


def _file_url(root: Path) -> str:
    return root.resolve().as_uri()


def test_upstream_matches_what_was_vendored(flashinfer, miles, ext, tmp_path):
    _vendor_fi(flashinfer, ext, repo=_file_url(flashinfer))
    _vendor_pr(miles, ext, repo=_file_url(miles))

    assert vendor_cake.check_upstream(ext, cache=tmp_path / "cache") == []
    assert vendor_cake.check_upstream(ext, cache=tmp_path / "cache") == []  # again, into the same cache


def test_upstream_drift_is_reported_for_both_kinds(flashinfer, miles, ext, tmp_path):
    _vendor_fi(flashinfer, ext, paths=(DET,), repo=_file_url(flashinfer))
    _vendor_pr(miles, ext, repo=_file_url(miles))
    manifest = _manifest(ext)
    det = manifest["ops"]["synth"]["cake_synth_det_backend"]
    del det["files"][".gitignore"]
    det["files"]["csrc/prog.cu"] = "0" * 40
    pr = manifest["ops"]["synth"]["cake_synth_sparse_det"]
    pr["source"]["exclude"]["csrc/cake_host_shim.h"] = "1" * 40
    _write_manifest(ext, manifest)

    assert vendor_cake.check_upstream(ext, cache=tmp_path / "cache") == [
        "_vendor/cake_synth_det_backend/.gitignore: upstream has it, not vendored",
        f"_vendor/cake_synth_det_backend/csrc/prog.cu: blob {'0' * 40} != upstream {_rev(flashinfer, DET + '/csrc/prog.cu')}",
        f"_vendor/cake_synth_sparse_det: excluded blobs {{'csrc/cake_host_shim.h': '{'1' * 40}'}} != upstream "
        f"{{'csrc/cake_host_shim.h': '{_rev(miles, PR_PATH + '/csrc/cake_host_shim.h')}'}}",
    ]


def test_an_unreachable_upstream_is_a_finding(flashinfer, ext, tmp_path):
    _vendor_fi(flashinfer, ext, paths=(DET,), repo=_file_url(tmp_path / "missing"))

    (finding,) = vendor_cake.check_upstream(ext, cache=tmp_path / "cache")
    assert finding.startswith("_vendor/cake_synth_det_backend: cannot fetch")


# --- command line ---


def test_the_command_line_vendors_checks_and_removes(flashinfer, miles, ext, capsys):
    root = ["--repo-root", str(ext)]
    fi = ["fi", "--checkout", str(flashinfer), "--commit", _rev(flashinfer), "--op", "synth", "--package", FAST]
    pr = [
        "miles-pr",
        "--checkout",
        str(miles),
        "--pr",
        "1234",
        "--commit",
        "HEAD",
        "--op",
        "synth",
        "--name",
        "cake_synth_sparse_det",
        "--path",
        PR_PATH,
        "--include",
        "csrc",
        "--include",
        "registry.json",
        "--exclude",
        "csrc/cake_host_shim.h",
        "--arch-from",
        "sm_107a=sm_103a",
        "--nvcc-flag=-O3",
        "--nvcc-flag=-std=c++17",
    ]

    assert vendor_cake.main([*root, *fi]) == 0
    assert vendor_cake.main([*root, *pr]) == 0
    assert _manifest(ext)["ops"]["synth"]["cake_synth_sparse_det"]["build"]["nvcc_flags"] == ["-O3", "-std=c++17"]
    assert vendor_cake.main([*root, "--check"]) == 0
    assert vendor_cake.main([*root, "--check", "synth"]) == 0
    (ext / vendor_cake.VENDOR / "cake_synth_fast" / "csrc" / "prog.cu").write_bytes(b"// edited\n")
    assert vendor_cake.main([*root, "--check", "synth"]) == 1
    assert vendor_cake.main([*root, "--check", "nope"]) == 1
    assert vendor_cake.main([*root, "remove", "--op", "synth", "--name", "cake_synth_fast"]) == 0
    assert vendor_cake.main([*root, *fi[:4], "HEAD~5", *fi[5:]]) == 2
    out = capsys.readouterr()
    assert "synth: cake_synth_fast <- flashinfer/experimental/cake/cake_synth_fast" in out.out
    assert "vendored (synth): ok, 2 packages" in out.out
    assert "VENDORED.json: no op 'nope'" in out.out
    assert "error: git rev-parse" in out.err
    for argv in ([], ["--upstream"], ["--check", "remove", "--op", "x", "--name", "y"]):
        with pytest.raises(SystemExit):
            vendor_cake.main([*root, *argv])


def test_the_command_line_checks_an_empty_tree(ext, capsys):
    assert vendor_cake.main(["--repo-root", str(ext), "--check"]) == 0
    assert capsys.readouterr().out == "vendored (every op): ok, 0 packages\n"
