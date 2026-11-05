"""The contract every op module and its implementation keep, checked on every registered op and on ``synth``.

CPU checks only: ``tests/cake/<op>/test_prebuild_gpu.py`` checks the prebuild guarantee of each op on a GPU, over the
``ROUTES`` it declares.
"""

import ast
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from miles_ext_kernels.cake._runtime import config, registry
from miles_ext_kernels.cake._runtime.spec import VARIANTS

from . import synthetic

REPO_ROOT = Path(__file__).resolve().parents[3]
CAKE = REPO_ROOT / "miles_ext_kernels" / "cake"
VENDORED = CAKE / "_vendor" / "VENDORED.json"
VENDOR_MODULE = "miles_ext_kernels.cake._vendor"
TESTS = REPO_ROOT / "tests" / "cake"
# module-level state the runtime replaced: the variant, options, prebuild switches and workspaces live in _runtime
FORBIDDEN_GLOBALS = frozenset({"VARIANT", "SAVE_STATE", "PREBUILD", "WORKSPACES"})
# prebuild covers every variant, token count and call-time switch, so none of them is a model parameter
FORBIDDEN_PREBUILD_PARAMETERS = frozenset(
    {"variant", "token_counts", "num_tokens", "T", "temperature", "head", "kcp", "cp_context", "cu_seqlens"}
)
CONTRACT_OPS = ("synth", *registry.OPS)


@pytest.fixture(params=CONTRACT_OPS)
def op(request, monkeypatch):
    if request.param == synthetic.NAME:
        synthetic.install(monkeypatch)
    return request.param


def _module(op):
    return importlib.import_module(f"miles_ext_kernels.cake.{op}")


def _impl(op):
    return importlib.import_module(f"miles_ext_kernels.cake._impl.{op}")


def _sources(module_file: Path) -> list[Path]:
    """A package's ``__init__.py`` stands for every module of its directory; any other module for itself."""
    if module_file.name == "__init__.py":
        return sorted(module_file.parent.rglob("*.py"))
    return [module_file]


def source_findings(path: Path) -> list[str]:
    """Module-level state the runtime owns, environment reads, flashinfer and cpp_extension imports."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    findings = []
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        findings.extend(
            f"{path.name}:{node.lineno}: module-level {target.id}"
            for target in targets
            if isinstance(target, ast.Name) and target.id in FORBIDDEN_GLOBALS
        )
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Global):
            forbidden = [name for name in node.names if name in FORBIDDEN_GLOBALS]
            findings.extend(f"{path.name}:{node.lineno}: global {name}" for name in forbidden)
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            if node.module == "os":
                findings.extend(
                    f"{path.name}:{node.lineno}: reads the environment (from os import {alias.name})"
                    for alias in node.names
                    if alias.name in ("environ", "environb", "getenv", "getenvb")
                )
        findings.extend(
            f"{path.name}:{node.lineno}: imports {module}"
            for module in modules
            if module.split(".")[0] == "flashinfer" or module == "torch.utils.cpp_extension"
        )
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in ("environ", "environb", "getenv", "getenvb")
        ):
            findings.append(f"{path.name}:{node.lineno}: reads the environment (os.{node.attr})")
    return findings


def _assigns(path: Path, name: str) -> bool:
    """Does the module at ``path`` assign ``name`` at module level?"""
    for node in ast.parse(path.read_bytes(), filename=str(path)).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return True
    return False


def op_tests_findings(tests_dir: Path) -> list[str]:
    """``tests/cake/<op>/`` holds ``__init__.py``, at least one CPU test file and ``test_prebuild_gpu.py``, which
    declares the op's ``ROUTES``."""
    required = ("__init__.py", "test_prebuild_gpu.py")
    findings = [f"{tests_dir.name}/{name}: missing" for name in required if not (tests_dir / name).is_file()]
    if not any(not path.stem.endswith("_gpu") for path in tests_dir.glob("test_*.py")):
        findings.append(f"{tests_dir.name}: no CPU test file")
    prebuild = tests_dir / "test_prebuild_gpu.py"
    if prebuild.is_file() and not _assigns(prebuild, "ROUTES"):
        findings.append(f"{tests_dir.name}/test_prebuild_gpu.py: no module-level ROUTES")
    return findings


def hook_findings(op: str, manifest: dict) -> list[str]:
    """The flashinfer hooks of ``op`` that importing ``miles_ext_kernels.cake._impl.<op>`` left in place.

    A hook is a function a vendored module may import flashinfer in (``allow.flashinfer_import`` of the op's packages
    in ``VENDORED.json``: ``MODULE:FUNCTION``); the implementation rebinds it, so it is no longer the vendored code.
    """
    _impl(op)
    findings = []
    for package, entry in sorted(manifest["ops"].get(op, {}).items()):
        for hook in entry.get("allow", {}).get("flashinfer_import", []):
            relative, _, qualname = hook.partition(":")
            parts = list(PurePosixPath(relative).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            module = importlib.import_module(".".join([VENDOR_MODULE, package, *parts]))
            target = module
            for name in qualname.split("."):
                target = getattr(target, name)
            code = getattr(target, "__code__", None)
            if code is None or Path(code.co_filename).resolve() == Path(module.__file__).resolve():
                findings.append(f"{package}/{hook}: importing _impl.{op} did not rebind it")
    return findings


def test_the_registered_ops_are_the_op_modules():
    modules = sorted(path.stem for path in CAKE.glob("*.py") if not path.name.startswith("_"))

    assert sorted(registry.OPS) == modules
    assert len(set(registry.OPS)) == len(registry.OPS)


def test_the_op_module_declares_the_op(op):
    module = _module(op)

    assert module.OP.name == op
    assert registry.op_spec(op) is module.OP
    if op != synthetic.NAME:
        assert Path(module.__file__) == CAKE / f"{op}.py"


def test_variants_and_flags(op):
    spec = registry.op_spec(op)

    assert spec.variants and set(spec.variants) <= set(VARIANTS)
    assert spec.variants == tuple(variant for variant in VARIANTS if variant in spec.variants)
    assert set(spec.cli_choices) <= set(spec.variants)
    for flag in spec.flags:
        assert flag.name.startswith(f"--cake-{spec.dashed}-") and "help" in flag.kwargs and "default" in flag.kwargs


def test_no_two_ops_share_an_argument(synth):
    dests = [flag.dest for flag in config.RUNTIME_FLAGS]
    for spec in map(registry.op_spec, registry.OPS):
        dests += [spec.variant_dest, *(flag.dest for flag in spec.flags)]

    assert len(set(dests)) == len(dests)


def test_the_packages_are_the_op_block_of_the_manifest(op):
    manifest = json.loads(VENDORED.read_text())

    assert set(registry.op_spec(op).packages) == set(manifest["ops"].get(op, {}))


def test_the_exports_resolve_and_the_implementation_keeps_the_contract(op):
    module, spec = _module(op), registry.op_spec(op)

    assert module.__all__ == ["OP", *spec.exports]
    if op != synthetic.NAME:
        pytest.importorskip("torch")
    impl = _impl(op)
    assert hasattr(impl, "__path__"), f"{impl.__name__} is not a package"
    for name in spec.exports:
        assert getattr(module, name) is getattr(impl, name)
    parameters = inspect.signature(impl.prebuild).parameters
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()), parameters
    assert {"device", "max_workers"} <= set(parameters)
    assert not FORBIDDEN_PREBUILD_PARAMETERS & set(parameters)
    release = inspect.signature(impl.release).parameters.values()
    assert all(parameter.default is not inspect.Parameter.empty for parameter in release)


def test_the_sources_keep_no_runtime_state_and_read_no_environment(op):
    module = _module(op)
    if op == synthetic.NAME:
        paths = [Path(synthetic.__file__)]
    else:
        paths = [Path(module.__file__), *_sources(CAKE / "_impl" / op / "__init__.py")]

    assert [finding for path in paths for finding in source_findings(path)] == []


@pytest.mark.parametrize("op", registry.OPS)
def test_the_op_has_cpu_tests_and_a_prebuild_test(op):
    assert op_tests_findings(TESTS / op) == []


def _run(script: str) -> subprocess.CompletedProcess:
    """``script`` in a fresh interpreter that imports this checkout and the tests' packages."""
    paths = [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")]
    pythonpath = os.pathsep.join(filter(None, paths))
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )


def test_the_flashinfer_hooks_are_rebound_without_flashinfer(op):
    """In a process where flashinfer cannot be imported, importing the implementation rebinds every hook."""
    if op != synthetic.NAME:
        pytest.importorskip("torch")
    result = _run(
        "import json, sys\n"
        "sys.modules['flashinfer'] = None\n"
        "from cake._runtime import synthetic, test_op_contract as contract\n"
        f"synthetic.install() if {op!r} == 'synth' else None\n"
        f"print(json.dumps(contract.hook_findings({op!r}, json.loads(contract.VENDORED.read_text()))))\n"
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == []


def test_the_op_imports_and_configures_without_torch(op):
    """Spot check of one op in a process where torch, tvm_ffi and flashinfer cannot be imported."""
    script = (
        "import sys\n"
        "for name in ('torch', 'tvm_ffi', 'flashinfer'):\n"
        "    sys.modules[name] = None\n"
        "import argparse, importlib, miles_ext_kernels\n"
        "from cake._runtime import synthetic\n"
        f"synthetic.install() if {op!r} == 'synth' else importlib.import_module('miles_ext_kernels.cake.{op}')\n"
        "parser = argparse.ArgumentParser()\n"
        "miles_ext_kernels.add_arguments(parser)\n"
        "args = parser.parse_args([])\n"
        "args.deterministic_mode = True\n"
        "miles_ext_kernels.validate_args(args)\n"
        "miles_ext_kernels.configure(args)\n"
    )
    result = _run(script)

    assert result.returncode == 0, result.stderr


# --- the checkers themselves ---


def test_the_source_checker_finds_runtime_state_environment_reads_and_forbidden_imports(tmp_path):
    source = tmp_path / "ops.py"
    source.write_text(
        "import os\n"
        "from os import environ\n"
        "import flashinfer.jit\n"
        "from torch.utils import cpp_extension\n"
        "VARIANT = 'det'\n"
        "WORKSPACES: dict = {}\n"
        "def configure():\n"
        "    global PREBUILD\n"
        "    return os.getenv('X')\n"
    )

    assert sorted(source_findings(source)) == sorted(
        [
            "ops.py:2: reads the environment (from os import environ)",
            "ops.py:3: imports flashinfer.jit",
            "ops.py:4: imports torch.utils.cpp_extension",
            "ops.py:5: module-level VARIANT",
            "ops.py:6: module-level WORKSPACES",
            "ops.py:8: global PREBUILD",
            "ops.py:9: reads the environment (os.getenv)",
        ]
    )


def test_the_tests_checker_wants_cpu_tests_and_a_prebuild_test(tmp_path):
    tests_dir = tmp_path / "synth"
    tests_dir.mkdir()
    (tests_dir / "test_kernel_gpu.py").write_text("")

    assert op_tests_findings(tests_dir) == [
        "synth/__init__.py: missing",
        "synth/test_prebuild_gpu.py: missing",
        "synth: no CPU test file",
    ]
    for name in ("__init__.py", "test_prebuild_gpu.py", "test_arguments.py"):
        (tests_dir / name).write_text("")
    assert op_tests_findings(tests_dir) == ["synth/test_prebuild_gpu.py: no module-level ROUTES"]
    (tests_dir / "test_prebuild_gpu.py").write_text("def test_x():\n    ROUTES = {}\n")
    assert op_tests_findings(tests_dir) == ["synth/test_prebuild_gpu.py: no module-level ROUTES"]
    (tests_dir / "test_prebuild_gpu.py").write_text("ROUTES: dict = {'plain': print}\n")
    assert op_tests_findings(tests_dir) == []


def test_the_hook_checker_finds_a_hook_left_in_place(tmp_path, monkeypatch):
    package = tmp_path / "cake_synth_hooked"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "backend.py").write_text(
        "def nvcc():\n    import flashinfer.jit\n\n\nclass Plan:\n    def run(self):\n        import flashinfer\n"
    )
    vendor = importlib.import_module(VENDOR_MODULE)
    monkeypatch.setattr(vendor, "__path__", [*vendor.__path__, str(tmp_path)])
    synthetic.install(monkeypatch)
    hooks = ["backend.py:Plan.run", "backend.py:nvcc"]
    manifest = {"ops": {"synth": {"cake_synth_hooked": {"allow": {"flashinfer_import": hooks}}}}}
    try:
        assert hook_findings("synth", manifest) == [
            f"cake_synth_hooked/{hook}: importing _impl.synth did not rebind it" for hook in hooks
        ]
        backend = sys.modules[f"{VENDOR_MODULE}.cake_synth_hooked.backend"]
        monkeypatch.setattr(backend, "nvcc", lambda: "13.4.0")
        monkeypatch.setattr(backend.Plan, "run", lambda self: None)
        assert hook_findings("synth", manifest) == []
    finally:
        for name in [name for name in sys.modules if name.startswith(f"{VENDOR_MODULE}.cake_synth_hooked")]:
            del sys.modules[name]
        vars(vendor).pop("cake_synth_hooked", None)
