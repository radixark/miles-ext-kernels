"""The loader against FlashInfer's own build rules (torch-free: a fake nvcc, a fake tvm_ffi and a fake torch).

``tools/templates/cake_build.py`` is FlashInfer's Cake package loader, byte-identical (flashinfer-ai/flashinfer
c5db8d3ccd3fa3842812280f7f208718dc0b8610, blob 00756137eb159e30a77907eda84fd245123d87ea, the loader of every vendored
package). The reference command is ``entry_flags`` / ``compile_command`` of FlashInfer's
``scripts/check_cake_build_manifests.py`` at the same commit, the plain-nvcc build of its vendored-package gate.
"""

import dataclasses
import fcntl
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import sys
import threading
import types
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

import miles_ext_kernels.cake._vendor
from miles_ext_kernels.cake._runtime import loader

from . import synthetic

CAKE_BUILD = Path(__file__).resolve().parents[3] / "tools" / "templates" / "cake_build.py"
CAKE_BUILD_BLOB = "00756137eb159e30a77907eda84fd245123d87ea"
INCLUDES = ("/synthetic/tvm_ffi/include", "/synthetic/tvm_ffi/dlpack/include")
CLOSURE = "c0105e" + "0" * 58
PACKAGE = "cake_synthetic"
SOURCES = {
    "csrc/prog_kernel.cu": b"// kernel\n",
    "csrc/prog_binding.cu": b"// binding\n",
    "csrc/cake_ffi_checks.h": b"// checks\n",
}
# every flag of the synthetic manifest that depends on the toolkit
CONDITIONAL = {"-static-global-template-stub=false", "-DSYNTHETIC_COMMON_13_4", "-Xptxas", "-O0", "-DSYNTHETIC_NEWER"}

FAKE_NVCC = """#!@PYTHON@
import json
import sys
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print("nvcc: NVIDIA (R) Cuda compiler driver")
    print("Cuda compilation tools, release @MAJOR@.@MINOR@, V@FULL@")
    print("Build cuda_@MAJOR@.@MINOR@.r@MAJOR@.@MINOR@/compiler.0_0")
    sys.exit(0)
with open(Path(__file__).with_name("calls.jsonl"), "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if Path(__file__).with_name("fail").exists():
    sys.stderr.write("error: synthetic failure\\n")
    sys.exit(2)
Path(sys.argv[sys.argv.index("-o") + 1]).write_bytes(b"synthetic library")
"""


# --- FlashInfer scripts/check_cake_build_manifests.py at c5db8d3c (reformatted to this repository's line length) ---
CUDA_CONSTRAINT_RE = re.compile(r"(==|>=)([0-9]+)\.([0-9]+)")


def cuda_constraint_matches(constraint: str, cuda: tuple[int, int]) -> bool:
    """Does toolkit ``cuda`` satisfy ``"==X.Y"`` / ``">=X.Y"``?

    The rule of ``cake_build._cuda_constraint_matches``, re-implemented so that
    nothing of the package is imported here.
    """
    match = CUDA_CONSTRAINT_RE.fullmatch(str(constraint))
    if match is None:
        raise ValueError(f"unsupported CUDA constraint {constraint!r}; expected '==X.Y' or '>=X.Y'")
    operator, major, minor = match.groups()
    version = (int(major), int(minor))
    return cuda == version if operator == "==" else cuda >= version


def conditional_flags(items: list[Any], cuda: tuple[int, int]) -> list[str]:
    """The flags of every ``nvcc_flags_if`` item whose ``cuda`` constraint matches."""
    flags: list[str] = []
    for item in items:
        if cuda_constraint_matches(item["cuda"], cuda):
            flags.extend(item["flags"])
    return flags


def entry_flags(manifest: dict[str, Any], entry: dict[str, Any], cuda: tuple[int, int]) -> list[str]:
    """The nvcc flags of ``entry`` from the manifest alone: ``common.nvcc_flags`` +
    the matching ``common.nvcc_flags_if`` + ``arch_flags[arch]`` + the entry's
    ``nvcc_flags`` + its matching ``nvcc_flags_if``."""
    common = manifest["common"]
    return [
        *common["nvcc_flags"],
        *conditional_flags(common["nvcc_flags_if"], cuda),
        *manifest["arch_flags"][entry["arch"]],
        *entry["nvcc_flags"],
        *conditional_flags(entry["nvcc_flags_if"], cuda),
    ]


def compile_command(
    nvcc: str,
    copy: Path,
    manifest: dict[str, Any],
    entry: dict[str, Any],
    cuda: tuple[int, int],
    includes: Iterable[str],
) -> list[str]:
    """The plain-nvcc command of ``entry``, run from ``copy``: the manifest flags,
    ``-I`` for ``common.include_dirs`` and ``includes`` (tvm-ffi, dlpack), the
    copy-relative sources, ``common.link_flags`` and ``-o <copy>/<entry.path>``."""
    common = manifest["common"]
    include_dirs = [str(copy / d) for d in common["include_dirs"]] + list(includes)
    return [
        nvcc,
        *entry_flags(manifest, entry, cuda),
        *(f"-I{d}" for d in include_dirs),
        *entry["sources"],
        *common["link_flags"],
        "-o",
        str(copy / entry["path"]),
    ]


# --- end of the FlashInfer copy ---


def _manifest(digests: dict[str, str]) -> dict[str, Any]:
    """A cake-build/1 manifest in the shape of FlashInfer's (cake_lm_head_loss: sm_103a '==13.0' -> -Xptxas -O0,
    sm_107a min_cuda 13.4)."""

    def entry(arch: str, min_cuda: str, nvcc_flags: list[str], nvcc_flags_if: list[dict[str, Any]]) -> dict[str, Any]:
        name = f"prog_{arch}"
        sources = ["csrc/prog_kernel.cu", "csrc/prog_binding.cu"]
        return {
            "arch": arch,
            "closure_sha256": CLOSURE,
            "entry": "run",
            "gencode": f"-gencode=arch=compute_{arch[3:]},code={arch}",
            "min_cuda": min_cuda,
            "name": name,
            "nvcc_flags": nvcc_flags,
            "nvcc_flags_if": nvcc_flags_if,
            "path": f"aot/{name}.so",
            "sha256": {source: digests[source] for source in sources},
            "sources": sources,
        }

    return {
        "schema": "cake-build/1",
        "package": f"synthetic.{PACKAGE}",
        "generator": {"cake_revision": "0" * 40, "cuda": "13.4.92", "rendered_at": "2026-11-03T00:00:00Z"},
        "arch_flags": {
            arch: [f"-gencode=arch=compute_{arch[3:]},code={arch}"] for arch in ("sm_100a", "sm_103a", "sm_107a")
        },
        "common": {
            "include_dirs": ["csrc"],
            "link_flags": ["-shared", "-lcudart", "-lcuda"],
            "nvcc_flags": [
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "--compiler-options=-fPIC",
                "--expt-relaxed-constexpr",
                "-Xfatbin=-compress-all",
                "--compress-mode=size",
            ],
            "nvcc_flags_if": [
                {"cuda": ">=12.8", "flags": ["-static-global-template-stub=false"]},
                {"cuda": "==13.4", "flags": ["-DSYNTHETIC_COMMON_13_4"]},
            ],
            "requires": {"cuda_toolkit": {"min": "12.9"}, "tvm_ffi": {"python": "apache-tvm-ffi>=0.1.11,<0.2"}},
        },
        "libraries": [
            entry("sm_100a", "12.9", ["--use_fast_math"], []),
            entry(
                "sm_103a",
                "12.9",
                ["-DCAKE_LOGITS_SM103=1"],
                [{"cuda": "==13.0", "flags": ["-Xptxas", "-O0"]}, {"cuda": ">=13.1", "flags": ["-DSYNTHETIC_NEWER"]}],
            ),
            entry("sm_107a", "13.4", [], []),
        ],
    }


class _Library:
    def __init__(self, cudart: int) -> None:
        self._cudart = cudart

    def build_cuda_version(self) -> int:
        return self._cudart


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache" / "miles_ext_kernels"


@pytest.fixture
def tvm_ffi(monkeypatch):
    """A stand-in for tvm_ffi: include finders, a version, and load_module returning ``cudart`` (default 13000)."""
    module = types.ModuleType("tvm_ffi")
    libinfo = types.ModuleType("tvm_ffi.libinfo")
    libinfo.find_include_path = lambda: INCLUDES[0]
    libinfo.find_dlpack_include_path = lambda: INCLUDES[1]
    module.libinfo = libinfo
    module.__version__ = "0.1.11"
    module.cudart = 13000
    module.loaded = []

    def load_module(path: str) -> _Library:
        module.loaded.append(path)
        return _Library(module.cudart)

    module.load_module = load_module
    monkeypatch.setitem(sys.modules, "tvm_ffi", module)
    monkeypatch.setitem(sys.modules, "tvm_ffi.libinfo", libinfo)
    return module


@pytest.fixture
def toolkit(tmp_path, monkeypatch):
    """``toolkit("13.0.88")``: a fake nvcc of that version as ``$CUDA_HOME/bin/nvcc``."""

    def make(full: str) -> Path:
        major, minor, _ = full.split(".")
        nvcc = tmp_path / f"cuda-{full}" / "bin" / "nvcc"
        nvcc.parent.mkdir(parents=True)
        script = FAKE_NVCC.replace("@PYTHON@", sys.executable).replace("@FULL@", full)
        nvcc.write_text(script.replace("@MAJOR@", major).replace("@MINOR@", minor))
        nvcc.chmod(0o755)
        monkeypatch.setenv("CUDA_HOME", str(nvcc.parents[1]))
        return nvcc

    return make


@pytest.fixture
def package(tmp_path):
    """FlashInfer's cake_build.py in a synthetic package, loaded as a fresh module."""
    root = tmp_path / "vendor" / PACKAGE
    (root / "csrc").mkdir(parents=True)
    for relative, data in SOURCES.items():
        (root / relative).write_bytes(data)
    digests = {relative: hashlib.sha256(data).hexdigest() for relative, data in SOURCES.items()}
    (root / "build.json").write_text(json.dumps(_manifest(digests), indent=2))
    shutil.copyfile(CAKE_BUILD, root / "cake_build.py")
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}_{uuid.uuid4().hex}", root / "cake_build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calls(nvcc: Path) -> list[list[str]]:
    log = nvcc.with_name("calls.jsonl")
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _entry(package, name: str) -> dict[str, Any]:
    return package.libraries_by_name()[name]


def _files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if "__pycache__" not in path.parts)


def test_the_fixture_is_flashinfer_cake_build():
    data = CAKE_BUILD.read_bytes()

    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == CAKE_BUILD_BLOB


def test_install_rebinds_the_toolchain_hook(package):
    assert loader.install(package) is package
    gen_jit_spec, get_cuda_version = package._jit_toolchain("prog_sm_100a")

    assert get_cuda_version is loader.get_cuda_version
    assert callable(gen_jit_spec)
    with pytest.raises(TypeError, match="no _jit_toolchain hook"):
        loader.install(types.ModuleType("not_a_loader"))


@pytest.mark.parametrize(
    ("full", "arch", "conditional"),
    [
        ("13.0.88", "sm_103a", ["-static-global-template-stub=false", "-Xptxas", "-O0"]),
        ("13.4.92", "sm_103a", ["-static-global-template-stub=false", "-DSYNTHETIC_COMMON_13_4", "-DSYNTHETIC_NEWER"]),
        ("13.0.88", "sm_100a", ["-static-global-template-stub=false"]),
        ("12.9.86", "sm_100a", ["-static-global-template-stub=false"]),
        ("13.4.92", "sm_107a", ["-static-global-template-stub=false", "-DSYNTHETIC_COMMON_13_4"]),
    ],
)
def test_the_command_is_flashinfer_compile_command(package, toolkit, tvm_ffi, cache, full, arch, conditional):
    nvcc = toolkit(full)
    loader.install(package)
    name = f"prog_{arch}"

    library = package.jit_spec(name).build()

    root = Path(package.__file__).parent
    entry = _entry(package, name)
    cuda = (int(full.split(".")[0]), int(full.split(".")[1]))
    reference = compile_command(str(nvcc), root, package.manifest(), entry, cuda, INCLUDES)
    # FlashInfer's gate compiles from the package copy with relative sources into aot/; the loader passes absolute
    # sources and compiles into a temporary file next to the cached library
    reference = [str(root / arg) if arg in entry["sources"] else arg for arg in reference]
    (call,) = _calls(nvcc)
    assert call[:-1] == reference[1:-1]
    assert Path(call[-1]).parent == library.parent == cache / PACKAGE
    assert [flag for flag in call if flag in CONDITIONAL] == conditional
    assert library.is_file()


def test_a_library_above_the_toolkit_raises_and_warmups_leave_it_out(package, toolkit, tvm_ffi):
    nvcc = toolkit("13.0.88")
    loader.install(package)

    with pytest.raises(RuntimeError, match="needs CUDA 13.4 or newer"):
        package.jit_spec("prog_sm_107a")
    assert package.jit_specs("sm_107a") == []
    assert [spec.name for spec in package.jit_specs("sm_103a")] == [f"prog_sm_103a_{CLOSURE[:20]}"]
    with pytest.raises(RuntimeError, match="needs CUDA 13.4 or newer"):
        loader.prebuild_libraries(package, ["prog_sm_107a"])
    assert _calls(nvcc) == []


def test_an_architecture_above_the_toolkit_raises_whatever_its_entry_records(package, toolkit, tvm_ffi):
    """nvcc 13.0 to 13.3 reject compute_107a: an SM107a entry recording a lower min_cuda still needs CUDA 13.4."""
    nvcc = toolkit("13.0.88")
    loader.install(package)
    _entry(package, "prog_sm_107a")["min_cuda"] = "13.0"

    error = r"compiling prog_sm_107a_\w+ for sm_107a needs CUDA 13\.4 or newer, but \S+ is CUDA 13\.0\.88"
    with pytest.raises(RuntimeError, match=error):
        package.jit_spec("prog_sm_107a")
    with pytest.raises(RuntimeError, match=r"for sm_107a needs CUDA 13\.4 or newer"):
        loader.prebuild_libraries(package, ["prog_sm_107a"])
    assert _calls(nvcc) == []


@pytest.mark.parametrize(
    ("full", "arch", "need"),
    [
        ("13.0.88", "sm_100a", None),
        ("13.0.88", "sm_103a", None),
        ("12.8.93", "sm_103a", "12.9"),
        ("13.0.88", "sm_107a", "13.4"),
        ("13.3.73", "sm_107a", "13.4"),
        ("13.4.92", "sm_107a", None),
    ],
)
def test_check_toolkit_names_the_cuda_version_the_architecture_needs(toolkit, full, arch, need):
    toolkit(full)
    if need is None:
        assert loader.check_toolkit(arch, "the kernels").full == full
    else:
        error = rf"compiling the kernels for {arch} needs CUDA {need} or newer, but \S+ is CUDA {full}"
        with pytest.raises(RuntimeError, match=error):
            loader.check_toolkit(arch, "the kernels")


def test_check_toolkit_reads_the_given_nvcc(toolkit):
    older = toolkit("13.0.88")
    toolkit("13.4.92")  # CUDA_HOME

    assert loader.check_toolkit("sm_107a", "the kernels").full == "13.4.92"
    with pytest.raises(RuntimeError, match=r"needs CUDA 13\.4 or newer"):
        loader.check_toolkit("sm_107a", "the kernels", str(older))


def test_every_architecture_has_a_toolkit_floor():
    assert set(loader.ARCH_BY_CAPABILITY.values()) == set(loader.MIN_CUDA)


def test_the_library_is_cached_outside_the_package_under_a_toolchain_key(
    package, toolkit, tvm_ffi, cache, monkeypatch
):
    toolkit("13.0.88")
    loader.install(package)
    root = Path(package.__file__).parent
    before = _files(root)

    path = package.jit_spec("prog_sm_100a").build()

    assert path.parent == cache / PACKAGE
    assert re.fullmatch(rf"prog_sm_100a_{CLOSURE[:20]}_[0-9a-f]{{16}}\.so", path.name)
    assert package.library_path("prog_sm_100a") == path
    assert _files(root) == before
    names = {path.name}
    toolkit("13.0.48")
    names.add(package.library_path("prog_sm_100a").name)
    tvm_ffi.__version__ = "0.1.12"
    names.add(package.library_path("prog_sm_100a").name)
    monkeypatch.setattr(platform, "machine", lambda: "synthetic-machine")
    names.add(package.library_path("prog_sm_100a").name)
    assert len(names) == 4


def test_a_cached_library_is_built_once(package, toolkit, tvm_ffi):
    nvcc = toolkit("13.0.88")
    loader.install(package)

    first = package.jit_spec("prog_sm_100a").build()
    second = package.jit_spec("prog_sm_100a").build()

    assert first == second
    assert len(_calls(nvcc)) == 1
    assert sorted(path.name for path in first.parent.iterdir()) == sorted([first.name, first.stem + ".lock"])


def test_load_library_builds_loads_and_checks_the_cudart_version(package, toolkit, tvm_ffi):
    toolkit("13.0.88")
    loader.install(package)

    library = package.load_library("prog_sm_100a")

    assert library.build_cuda_version() == 13000
    assert tvm_ffi.loaded == [str(package.library_path("prog_sm_100a"))]
    assert package.record_cuda_version("sm_100a") == (13, 0)
    tvm_ffi.cudart = 12090
    with pytest.raises(RuntimeError, match="compiled against CUDART 12090"):
        package.load_library("prog_sm_100a")


def test_a_failed_compile_raises_and_caches_nothing(package, toolkit, tvm_ffi):
    nvcc = toolkit("13.0.88")
    nvcc.with_name("fail").touch()
    loader.install(package)
    spec = package.jit_spec("prog_sm_100a")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        spec.build()
    assert [path.name for path in spec.get_library_path().parent.iterdir()] == [spec.get_library_path().stem + ".lock"]


@pytest.fixture
def held_lock(monkeypatch):
    """``held_lock(spec)``: another open file holding the build lock of ``spec``, with the wait shortened."""
    monkeypatch.setattr(loader, "_LOCK_POLL_SECONDS", 0.01)
    monkeypatch.setattr(loader, "_LOCK_LOG_SECONDS", 0.05)
    monkeypatch.setattr(loader, "_LOCK_TIMEOUT_SECONDS", 0.5)
    holders = []

    def hold(spec) -> Any:
        lock = spec.get_library_path().with_suffix(".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        holder = open(lock, "a")
        fcntl.flock(holder, fcntl.LOCK_EX)
        holders.append(holder)
        return holder

    yield hold
    for holder in holders:
        holder.close()


def test_a_held_build_lock_is_reported_and_given_up_on(package, toolkit, tvm_ffi, held_lock, caplog):
    nvcc = toolkit("13.0.88")
    loader.install(package)
    spec = package.jit_spec("prog_sm_100a")
    held_lock(spec)
    lock = spec.get_library_path().with_suffix(".lock")

    with caplog.at_level(logging.WARNING, logger=loader.__name__):
        with pytest.raises(RuntimeError, match=re.escape(f"for the build lock {lock}. Another process holds it")):
            spec.build()

    assert any(f"for the build lock {lock} of {spec.name}" in record.getMessage() for record in caplog.records)
    assert _calls(nvcc) == []
    assert not spec.get_library_path().exists()


def test_a_build_waits_for_a_released_lock(package, toolkit, tvm_ffi, held_lock, monkeypatch):
    monkeypatch.setattr(loader, "_LOCK_TIMEOUT_SECONDS", 60)
    nvcc = toolkit("13.0.88")
    loader.install(package)
    spec = package.jit_spec("prog_sm_100a")
    release = threading.Timer(0.1, held_lock(spec).close)
    release.start()

    path = spec.build()
    release.join()

    assert path.is_file()
    assert len(_calls(nvcc)) == 1


def test_prebuild_returns_every_library_and_compiles_only_the_missing_ones(package, toolkit, tvm_ffi):
    nvcc = toolkit("13.0.88")
    aot = Path(package.__file__).parent / "aot"
    aot.mkdir()
    (aot / "prog_sm_103a.so").write_bytes(b"prebuilt")
    names = ["prog_sm_100a", "prog_sm_103a", "prog_sm_100a"]

    first = loader.prebuild_libraries(package, names, max_workers=2)
    again = loader.prebuild_libraries(package, names)

    assert first == again == [package.jit_spec("prog_sm_100a").get_library_path(), aot / "prog_sm_103a.so"]
    assert len(_calls(nvcc)) == 1


def test_the_toolkit_is_cuda_home_else_path(toolkit, monkeypatch):
    on_path = toolkit("12.9.86")
    toolkit("13.0.88")
    monkeypatch.setenv("PATH", f"{on_path.parent}{os.pathsep}{os.environ.get('PATH', '')}")

    assert loader.get_cuda_version() == (13, 0, "13.0.88")
    monkeypatch.delenv("CUDA_HOME")
    assert loader.get_cuda_version() == (12, 9, "12.9.86")
    assert loader.get_cuda_version().cudart == 12090


def _forget_synthetic_package() -> None:
    """Drop ``_vendor.cake_synthetic`` and the alias package ``_vendor.cake``, with their submodules, and nothing else.

    A real package (``_vendor.cake_<name>``) shares the prefix ``_vendor.cake`` and stays imported: an op keeps
    references to its modules.
    """
    vendor = miles_ext_kernels.cake._vendor
    names = (f"{vendor.__name__}.{PACKAGE}", f"{vendor.__name__}.cake")
    submodules = tuple(f"{name}." for name in names)
    for name in [name for name in sys.modules if name in names or name.startswith(submodules)]:
        del sys.modules[name]
    for attribute in (PACKAGE, "cake"):
        vars(vendor).pop(attribute, None)


@pytest.fixture
def vendored(package, monkeypatch):
    """The synthetic package importable as ``miles_ext_kernels.cake._vendor.cake_synthetic``."""
    root = Path(package.__file__).parent
    (root / "__init__.py").write_bytes(b"")
    vendor = miles_ext_kernels.cake._vendor
    monkeypatch.setattr(vendor, "__path__", [*vendor.__path__, str(root.parent)])
    monkeypatch.setattr(loader, "_OP_OF_PACKAGE", {})
    yield PACKAGE
    _forget_synthetic_package()


def test_forgetting_the_synthetic_package_keeps_every_other_vendored_package(monkeypatch):
    vendor = miles_ext_kernels.cake._vendor.__name__
    forgotten = [f"{vendor}.{PACKAGE}", f"{vendor}.{PACKAGE}.cake_build", f"{vendor}.cake", f"{vendor}.cake.cake_x"]
    kept = [f"{vendor}.cake_x", f"{vendor}.cake_x.cake_build", f"{vendor}.{PACKAGE}_other"]
    for name in forgotten + kept:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    _forget_synthetic_package()

    assert [name for name in forgotten + kept if name in sys.modules] == kept


def test_load_package_imports_a_vendored_package_with_the_loader_installed(vendored, toolkit, tvm_ffi):
    toolkit("13.0.88")

    package = loader.load_package(vendored)

    assert package.__name__ == f"miles_ext_kernels.cake._vendor.{PACKAGE}"
    assert package.cake_build._jit_toolchain.__module__ == loader.__name__
    assert package.cake_build.load_library("prog_sm_100a").build_cuda_version() == 13000
    assert loader.prebuild_libraries(vendored, ["prog_sm_100a"]) == [package.cake_build.library_path("prog_sm_100a")]


def test_install_op_installs_the_loader_on_every_package_of_the_op(vendored, toolkit, tvm_ffi, monkeypatch):
    synthetic.install(monkeypatch, dataclasses.replace(synthetic.OP, packages=(vendored,)))
    loader.install_op("synth")
    loader.install_op("synth")  # idempotent

    cake_build = sys.modules[f"miles_ext_kernels.cake._vendor.{PACKAGE}.cake_build"]
    assert cake_build._jit_toolchain.__module__ == loader.__name__
    assert loader._OP_OF_PACKAGE == {PACKAGE: "synth"}


def test_the_cake_alias_is_the_vendored_package_itself(vendored, package):
    """FlashInfer's ``..cake.<name>`` imports reach the one copy of ``_vendor.<name>`` and of its submodules."""
    (Path(package.__file__).parent / "uses_alias.py").write_text(
        "from ..cake import cake_synthetic as by_name\n"
        "from ..cake.cake_synthetic import cake_build\n"
        "from ..cake.cake_synthetic.cake_build import load_library\n"
    )
    vendor = miles_ext_kernels.cake._vendor.__name__

    module = importlib.import_module(f"{vendor}.{PACKAGE}.uses_alias")
    real = sys.modules[f"{vendor}.{PACKAGE}"]
    assert module.by_name is real and sys.modules[f"{vendor}.cake.{PACKAGE}"] is real
    assert module.cake_build is sys.modules[f"{vendor}.{PACKAGE}.cake_build"]
    assert sys.modules[f"{vendor}.cake.{PACKAGE}.cake_build"] is module.cake_build
    assert module.load_library is module.cake_build.load_library
    assert sys.modules[f"{vendor}.cake"].__path__ == []
    with pytest.raises(ModuleNotFoundError, match=rf"No module named '{vendor}\.cake_nope'"):
        importlib.import_module(f"{vendor}.cake.cake_nope")


def test_no_build_raises_instead_of_compiling_and_names_the_op(vendored, toolkit, tvm_ffi, monkeypatch):
    nvcc = toolkit("13.0.88")
    synthetic.install(monkeypatch, dataclasses.replace(synthetic.OP, packages=(vendored,)))
    loader.install_op("synth")
    cake_build = sys.modules[f"miles_ext_kernels.cake._vendor.{PACKAGE}.cake_build"]
    error = r"prog_sm_100a_\w+ would compile inside a call: prebuild\('synth', \.\.\.\) first"

    with loader.no_build():
        with loader.no_build():
            with pytest.raises(RuntimeError, match=error):
                cake_build.load_library("prog_sm_100a")
        with pytest.raises(RuntimeError, match=error):
            loader.prebuild_libraries(vendored, ["prog_sm_100a"])
    assert _calls(nvcc) == []
    loader.prebuild_libraries(vendored, ["prog_sm_100a"])
    with loader.no_build():
        assert cake_build.load_library("prog_sm_100a").build_cuda_version() == 13000
    assert len(_calls(nvcc)) == 1


@pytest.mark.parametrize(("capability", "arch"), [((10, 0), "sm_100a"), ((10, 3), "sm_103a"), ((10, 7), "sm_107a")])
def test_device_arch_names_the_architecture_of_the_device(monkeypatch, capability, arch):
    asked = []

    def get_device_capability(device):
        asked.append(device)
        return capability

    cuda = types.SimpleNamespace(get_device_capability=get_device_capability)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))

    assert loader.device_arch() == arch
    assert loader.device_arch("cuda:1") == arch
    assert asked == [None, "cuda:1"]


def test_device_arch_rejects_other_devices(monkeypatch):
    cuda = types.SimpleNamespace(get_device_capability=lambda device: (9, 0))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))

    with pytest.raises(NotImplementedError, match="need SM100a, SM103a or SM107a, not compute capability 9.0"):
        loader.device_arch()


def test_a_generated_build_json_compiles_with_flashinfer_compile_command(tmp_path, toolkit, tvm_ffi, cache):
    """The build.json tools/vendor_cake.py generates for a miles pull request's sources builds like FlashInfer's."""
    spec = importlib.util.spec_from_file_location("vendor_cake", CAKE_BUILD.parents[1] / "vendor_cake.py")
    vendor_cake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor_cake)
    root = tmp_path / "vendor" / "cake_synth_generated"
    record = {"name": "gen_fwd", "sources": ["gen_kernel.cu", "gen_binding.cu"], "ffi_entry": "run"}
    registry = {"fwd": {**record, "compile_flags": ["-DGEN=1"], "arches": ["sm_100a", "sm_103a"]}}
    files = {
        "registry.json": json.dumps(registry).encode(),
        "csrc/gen_kernel.cu": b"// kernel\n",
        "csrc/gen_binding.cu": b"// binding\n",
        "csrc/cake_host_shim.h": (CAKE_BUILD.parent / "cake_host_shim.h").read_bytes(),
        "cake_build.py": CAKE_BUILD.read_bytes(),
    }
    build = {
        "arch_from": {"sm_107a": "sm_103a"},
        "link_flags": list(vendor_cake.DEFAULT_LINK_FLAGS),
        "nvcc_flags": list(vendor_cake.DEFAULT_NVCC_FLAGS),
        "registry": "registry.json",
    }
    files["build.json"] = vendor_cake.render_build_json(root.name, files, build)
    for relative, data in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(data)
    module_spec = importlib.util.spec_from_file_location(f"{root.name}_{uuid.uuid4().hex}", root / "cake_build.py")
    package = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(package)
    nvcc = toolkit("13.4.92")
    loader.install(package)

    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        package.jit_spec(f"gen_fwd_{arch}").build()

    calls = _calls(nvcc)
    assert len(calls) == 3
    for arch, call in zip(("sm_100a", "sm_103a", "sm_107a"), calls):
        entry = _entry(package, f"gen_fwd_{arch}")
        reference = compile_command(str(nvcc), root, package.manifest(), entry, (13, 4), INCLUDES)
        reference = [str(root / arg) if arg in entry["sources"] else arg for arg in reference]
        assert call[:-1] == reference[1:-1]
        assert f"-gencode=arch=compute_{arch[3:]},code={arch}" in call and "-DGEN=1" in call
