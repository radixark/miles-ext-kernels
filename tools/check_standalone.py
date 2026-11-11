#!/usr/bin/env python3
"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The standalone gate of ``miles_ext_kernels/cake/_vendor/``: every vendored package works as a copy of its own
directory, without ``flashinfer`` and without ``miles_ext_kernels``. The tool needs Python only. Its children need an
interpreter with torch and tvm_ffi (``--python``, default this one).

The contract of a vendored package (FlashInfer's Cake package contract, flashinfer-ai/flashinfer#6553 and #7047):

- it imports only itself, the sibling packages its ``build.json`` declares (``common.requires.packages``), torch,
  tvm_ffi, ``cuda.bindings`` and the standard library;
- its one flashinfer import is the JIT fallback of ``cake_build._jit_toolchain``, inside a ``try`` that catches the
  ``ImportError``;
- it reads no environment variable;
- its ``build.json`` (schema ``cake-build/1``) is the only record of what it compiles: a builder compiles every entry
  into ``aot/`` with plain nvcc and writes ``aot/build_info.json`` (schema ``cake-build-info/1``), and
  ``cake_build.load_library`` serves the prebuilt library before any JIT.

Commands::

    python tools/check_standalone.py static [--package NAME ...]
    python tools/check_standalone.py imports [--package NAME ...] [--work-dir DIR] [--python PY]
    python tools/check_standalone.py build --work-dir DIR [--package NAME ...] [--arch ARCH ...] [--jobs N]
        [--nvcc PATH] [--nvcc-for ARCH=PATH ...] [--python PY] [--compile-only]

Every command also takes ``--repo-root`` and ``--report-json PATH``. It exits 0 when it finds nothing, 1 with findings
and 2 when it cannot run.

``static`` reads the sources only (``ast``). It reports every import outside the contract, every relative import that
leaves the package for an undeclared sibling, every environment read (``os.environ``, ``os.getenv``), a declared
sibling that is not a vendored package of the same op, and a registry that names a library ``build.json`` does not
list. The registries it reads are ``sources.json`` (role -> ``{library, arches, arguments}``) and a miles pull
request's ``registry.json`` (stage -> record, the libraries ``tools/vendor_cake.py`` renders into ``build.json``). A
``cake_jit.py`` registry computes its names at import time; no step checks them against ``build.json`` (the build
loads the ``build.json`` entries only). ``build.json`` ``package`` may name any import path: the copy imports under
its own name.

``imports`` copies each package into ``<work>/imports/<name>/`` the way FlashInfer's vendored-package gate lays it
out: alone as the top-level package ``vendored_<name>``, or, with siblings, as ``vendored_<name>.<name>`` beside
``vendored_<name>.<sibling>`` in a namespace directory. A child ``python -I`` started in the work directory blocks
``flashinfer``, every ``flashinfer_*`` package and ``miles_ext_kernels`` (a meta-path finder and ``None`` in
``sys.modules``: an editable install's finder still runs under ``-I``). It drops every ``sys.path`` entry at or above
the repository or the copy, keeps the directories torch and tvm_ffi load from, and imports the package and every
module of it one by one. A module that fails, a blocked module that loaded anyway, or any module loaded from the
repository is a finding.

``build`` runs on a compute node with nvcc. For each toolkit (``--nvcc``, and ``--nvcc-for ARCH=PATH`` for an
architecture that needs another one) it copies each package into ``<work>/build/cuda-<version>/<name>/`` as above,
compiles every entry of the selected architectures with :func:`compile_command` (``--jobs`` at a time), and writes
``aot/build_info.json`` as :func:`render_build_info` renders it. A package's siblings are compiled once, as packages
of their own, and their copies (with ``aot/``) are placed beside it. Then a blocked child imports each copy and loads
every compiled library through ``cake_build.load_library``. The library must be the one in the copy's ``aot/``, its
``build_cuda_version()`` must be the toolkit's ``CUDART_VERSION``, and ``record_cuda_version(arch)`` must be the
toolkit. The load needs a GPU node (``libcuda.so.1``); ``--compile-only`` stops after the compile. The tool refuses
to run (exit 2) rather than skip, before it starts a child: a toolkit older than :data:`MIN_CUDA` for an architecture,
an entry whose ``min_cuda`` the toolkit does not reach, or a malformed ``build.json``. The default nvcc is needed only
for an architecture without ``--nvcc-for``.

``--work-dir`` must lie outside the repository, which must never hold an ``aot/`` directory
(``tools/vendor_cake.py --check``). The tool deletes and rewrites its own ``imports/`` and ``build/`` subdirectories
there. Without ``--work-dir``, ``imports`` uses a temporary directory and deletes it.

Two lists allow static findings, and an entry that matches no finding is itself a finding. The ``allow`` record of
a package in ``VENDORED.json`` (which ``tools/vendor_cake.py`` also enforces) allows the environment reads of its
``env_read`` modules and the flashinfer imports inside its ``flashinfer_import`` functions (``module:function``).
:data:`ALLOWLIST` holds the other findings the vendored tree has today, each with its reason. Both only shrink: the
change that fixes a module deletes its entry. An allowlisted optional import (``import triton``) is also blocked in
the children of that package, so the package must import without it.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = PurePosixPath("miles_ext_kernels/cake/_vendor")
MANIFEST = "VENDORED.json"
BUILD_SCHEMA = "cake-build/1"
BUILD_INFO_SCHEMA = "cake-build-info/1"
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
# The oldest toolkit the runtime builds each architecture with (miles_ext_kernels.cake._runtime.loader.MIN_CUDA).
MIN_CUDA = {"sm_100a": (13, 4), "sm_103a": (13, 4), "sm_107a": (13, 4)}
# What a vendored module may import besides its own package, its declared siblings and the standard library.
ALLOWED_IMPORTS = ("torch", "tvm_ffi", "cuda.bindings")
# The packages no child may import: a copy works without either installed. A child also blocks every flashinfer_*.
BLOCKED = ("flashinfer", "miles_ext_kernels")
LOADER = "cake_build.py"
JIT_HOOK = "_jit_toolchain"
CHILD_NAME = "check_standalone_child.py"
CHILD_TIMEOUT = 600  # seconds, per child
NVCC_STDERR_LINES = 40  # kept per failed compile
TAIL_LINES = 12  # kept per failed import or child

# The static findings of the vendored tree today that VENDORED.json ``allow`` does not cover: (package, module,
# subject, reason). Delete an entry with the code it names; never add one for new code.
ALLOWLIST = (
    (
        "cake_mok_bf16",
        "mxfp8.py",
        "import triton",
        "the optional Triton / Gluon MXFP8 quantizer; without Triton the PyTorch path writes the same bytes",
    ),
)

_NVCC_RELEASE = re.compile(r"release (\d+)\.(\d+)")
_NVCC_BUILD = re.compile(r"\bV(\d+\.\d+\.\d+)")
_ENV_NAMES = ("environ", "environb", "getenv", "getenvb")
_RESULT = "check-standalone-result: "


class GateError(Exception):
    """The gate cannot run (exit 2)."""


class Finding(NamedTuple):
    """One finding. ``module`` is package-relative (empty for the package as a whole); ``subject`` is what
    :data:`ALLOWLIST` matches."""

    package: str
    module: str
    subject: str
    message: str

    def line(self) -> str:
        where = f"{self.package}/{self.module}" if self.module else self.package
        return f"{where}: {self.message}"


class Package(NamedTuple):
    """A vendored package: ``_vendor/<name>/`` of ``ops.<op>`` in ``VENDORED.json``."""

    name: str
    op: str
    directory: Path
    siblings: tuple[str, ...]  # build.json common.requires.packages
    registry: str | None  # the data file naming the libraries the package loads
    arch_from: Mapping[str, str]  # miles-pr: target architecture -> the architecture whose records it compiles
    allow: Mapping[str, Sequence[str]]  # VENDORED.json allow: env_read modules, flashinfer_import module:function


class Environment(NamedTuple):
    """What the children need of ``python``: the directories torch and tvm_ffi load from, and tvm-ffi's headers."""

    python: str
    paths: tuple[str, ...]
    includes: tuple[str, ...]


class Toolkit(NamedTuple):
    """An nvcc and its toolkit: ``release X.Y`` and the full ``VX.Y.Z``, and the last line of ``--version``."""

    nvcc: str
    cuda: tuple[int, int]
    full: str
    line: str

    @property
    def cudart(self) -> int:
        """``CUDART_VERSION`` of the toolkit: ``1000 * major + 10 * minor``."""
        return 1000 * self.cuda[0] + 10 * self.cuda[1]


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


def render_build_info(cuda: str, cudart_version: int, arches: Iterable[str]) -> bytes:
    """``aot/build_info.json`` as the Cake exporter renders it: ``schema``, ``cuda`` (the full toolkit version),
    ``cudart_version`` and the sorted ``arches``; sorted keys, indent 2, ASCII and a trailing newline."""
    document = {
        "arches": sorted(set(arches)),
        "cuda": cuda,
        "cudart_version": cudart_version,
        "schema": BUILD_INFO_SCHEMA,
    }
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


# ---------------------------------------------------------------------------------------------------------------
# the vendored packages


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, str(exc)


def _declared_siblings(build: Any) -> tuple[str, ...]:
    """``common.requires.packages`` of a loaded ``build.json``; empty when it is missing or malformed."""
    try:
        names = build["common"]["requires"]["packages"]
    except (KeyError, TypeError):
        return ()
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return ()
    return tuple(names)


def discover(repo_root: Path = REPO_ROOT) -> list[Package]:
    """Every package ``VENDORED.json`` lists, by name."""
    vendor = repo_root / VENDOR
    manifest, error = _read_json(vendor / MANIFEST)
    if error is not None or not isinstance(manifest, dict) or not isinstance(manifest.get("ops"), dict):
        raise GateError(f"{vendor / MANIFEST}: not a vendored manifest ({error or 'no ops'})")
    packages = []
    for op, block in manifest["ops"].items():
        for name, record in block.items():
            directory = vendor / name
            kind = record["source"]["kind"]
            build, _ = _read_json(directory / "build.json")
            if (directory / "sources.json").is_file():
                registry = "sources.json"
            elif kind == "miles-pr":
                registry = record["build"]["registry"]
            else:
                registry = None
            arch_from = dict(record.get("build", {}).get("arch_from", {}))
            allow = dict(record.get("allow") or {})
            packages.append(Package(name, op, directory, _declared_siblings(build), registry, arch_from, allow))
    return sorted(packages, key=lambda package: package.name)


def select(packages: Sequence[Package], names: Iterable[str] | None) -> list[Package]:
    """The packages ``names`` names (all of them for None)."""
    if not names:
        return list(packages)
    by_name = {package.name: package for package in packages}
    if unknown := sorted(set(names) - set(by_name)):
        raise GateError(f"not vendored packages: {', '.join(unknown)}")
    return [by_name[name] for name in sorted(set(names))]


def registry_libraries(package: Package) -> set[str] | None:
    """The libraries the registry of ``package`` names; None when it has no data registry."""
    if package.registry is None:
        return None
    registry = json.loads((package.directory / package.registry).read_text(encoding="utf-8"))
    if package.registry == "sources.json":
        return {f"{record['library']}_{arch}" for record in registry.values() for arch in record["arches"]}
    names = set()
    for record in registry.values():
        arches = list(record["arches"])
        for arch in {*arches, *(target for target, source in package.arch_from.items() if source in arches)}:
            source_arch = arch if arch in arches else package.arch_from[arch]
            names.add(f"{record.get('per_arch', {}).get(source_arch, {}).get('name', record.get('name'))}_{arch}")
    return names


def sibling_findings(package: Package, by_name: Mapping[str, Package]) -> list[Finding]:
    """A declared sibling must be another vendored package of the same op."""
    found = []
    for sibling in package.siblings:
        other = by_name.get(sibling)
        if other is None or other.name == package.name or other.op != package.op:
            found.append(
                Finding(
                    package.name,
                    "build.json",
                    f"sibling {sibling}",
                    f"declares the sibling package {sibling}, which is not another vendored package of {package.op}",
                )
            )
    return found


# ---------------------------------------------------------------------------------------------------------------
# static


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(
        node is None or (isinstance(node, ast.Name) and node.id in ("ImportError", "ModuleNotFoundError", "Exception"))
        for node in types
    )


def _guarded_jit_imports(tree: ast.Module) -> set[int]:
    """The nodes inside a ``try`` that catches ``ImportError`` in the module-level function ``_jit_toolchain``."""
    guarded: set[int] = set()
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name != JIT_HOOK:
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Try) and any(_catches_import_error(handler) for handler in node.handlers):
                guarded.update(id(inner) for statement in node.body for inner in ast.walk(statement))
    return guarded


def _functions(tree: ast.Module) -> dict[str, ast.AST]:
    """The functions a module defines, by qualified name: its own (``f``) and its classes' methods (``C.f``), also
    under a module-level ``if`` / ``try`` / ``with`` (the names ``VENDORED.json`` ``flashinfer_import`` uses)."""
    found: dict[str, ast.AST] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[prefix + node.name] = node
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for block in (node.body, getattr(node, "orelse", []), getattr(node, "finalbody", [])):
                    visit(block, prefix)
                for handler in getattr(node, "handlers", []):
                    visit(handler.body, prefix)

    visit(tree.body, "")
    return found


def _allowed_absolute(module: str, names: Sequence[str]) -> bool:
    top = module.partition(".")[0]
    if top in sys.stdlib_module_names:
        return True
    if any(module == allowed or module.startswith(allowed + ".") for allowed in ALLOWED_IMPORTS):
        return True
    return module == "cuda" and bool(names) and all(name == "bindings" for name in names)


def module_findings(package: str, relative: str, source: bytes, siblings: Sequence[str]) -> list[Finding]:
    """The static findings of one module of ``package`` (``relative`` to its directory)."""
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as exc:
        return [Finding(package, relative, "syntax", f"does not parse: {exc}")]
    depth = len(PurePosixPath(relative).parts) - 1  # of the module's own package below the package directory
    guarded = _guarded_jit_imports(tree) if relative == LOADER else set()
    # a flashinfer import inside a function names it, so that VENDORED.json allow can name the function
    function_of = {id(inner): name for name, function in _functions(tree).items() for inner in ast.walk(function)}
    os_names = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(alias.asname or "os" for alias in node.names if alias.name.partition(".")[0] == "os")
    found = []

    def add(node: ast.AST, subject: str, message: str) -> None:
        found.append(Finding(package, relative, subject, f"{node.lineno}: {message}"))

    def add_import(node: ast.AST, module: str) -> None:
        top = module.partition(".")[0]
        if top != "flashinfer":
            add(node, f"import {top}", f"imports {module}")
        elif id(node) not in guarded:
            function = function_of.get(id(node))
            add(node, f"import flashinfer in {function}" if function else "import flashinfer", f"imports {module}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _allowed_absolute(alias.name, ()):
                    add_import(node, alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [alias.name for alias in node.names]
            if not _allowed_absolute(node.module, names):
                add_import(node, node.module)
            if node.module == "os":
                for name in names:
                    if name in _ENV_NAMES:
                        add(node, "environment", f"reads the environment (from os import {name})")
        elif isinstance(node, ast.ImportFrom) and node.level == depth + 2:
            # the parent package of a copy: its declared siblings only
            for name in [node.module.partition(".")[0]] if node.module else [alias.name for alias in node.names]:
                if name not in siblings:
                    message = f"imports ..{name}, a package build.json does not declare as a sibling"
                    add(node, f"import ..{name}", message)
        elif isinstance(node, ast.ImportFrom) and node.level > depth + 2:
            add(node, "import above the parent", f"imports {'.' * node.level}{node.module or ''}, above its parent")
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in os_names
            and node.attr in _ENV_NAMES
        ):
            add(node, "environment", f"reads the environment ({node.value.id}.{node.attr})")
    return found


def _python_files(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.relative_to(directory).parts
    )


def package_findings(package: Package, by_name: Mapping[str, Package]) -> list[Finding]:
    """The static findings of one package: ``build.json``, its siblings, its registry and every module."""
    found: list[Finding] = []
    build, error = _read_json(package.directory / "build.json")
    if error is not None:
        found.append(Finding(package.name, "build.json", "build.json", error))
    elif not isinstance(build, dict) or build.get("schema") != BUILD_SCHEMA:
        found.append(Finding(package.name, "build.json", "build.json", f"is not a {BUILD_SCHEMA} manifest"))
    else:
        requires = (build.get("common") or {}).get("requires") or {}
        if "packages" in requires and list(package.siblings) != requires["packages"]:
            found.append(
                Finding(
                    package.name,
                    "build.json",
                    "build.json",
                    f"common.requires.packages is {requires['packages']!r}, not a list of package names",
                )
            )
        found.extend(sibling_findings(package, by_name))
        listed = {library["name"] for library in build.get("libraries", [])}
        try:
            named = registry_libraries(package)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            found.append(Finding(package.name, str(package.registry), "registry", f"cannot be read: {exc!r}"))
            named = None
        if named is not None and (missing := sorted(named - listed)):
            found.append(
                Finding(
                    package.name,
                    str(package.registry),
                    "registry",
                    f"names {len(missing)} libraries build.json does not list: {', '.join(missing[:5])}",
                )
            )
    for relative in _python_files(package.directory):
        source = (package.directory / relative).read_bytes()
        found.extend(module_findings(package.name, relative, source, package.siblings))
    return found


def apply_allowlist(
    found: Sequence[Finding], allowlist: Iterable[tuple[str, str, str, str]], checked: Iterable[str]
) -> list[Finding]:
    """``found`` without the findings ``allowlist`` lists, plus a finding for every entry of a ``checked`` package that
    matches none."""
    allowlist, checked = list(allowlist), set(checked)
    listed = {(package, module, subject) for package, module, subject, _ in allowlist}
    present = {(finding.package, finding.module, finding.subject) for finding in found}
    kept = [finding for finding in found if (finding.package, finding.module, finding.subject) not in listed]
    stale = [
        Finding(package, module, subject, f"{subject!r} is allowed ({reason}) but no longer occurs: delete the entry")
        for package, module, subject, reason in allowlist
        if package in checked and (package, module, subject) not in present
    ]
    return kept + stale


def vendored_allowlist(packages: Iterable[Package]) -> list[tuple[str, str, str, str]]:
    """The allowlist entries of the ``VENDORED.json`` ``allow`` records of ``packages``: an ``env_read`` module allows
    its environment reads, a ``flashinfer_import`` ``module:function`` the flashinfer imports inside that function."""
    entries = []
    for package in packages:
        for module in package.allow.get("env_read", ()):
            entries.append((package.name, module, "environment", "VENDORED.json allow env_read"))
        for item in package.allow.get("flashinfer_import", ()):
            module, _, function = str(item).partition(":")
            subject = f"import flashinfer in {function}"
            entries.append((package.name, module, subject, "VENDORED.json allow flashinfer_import"))
    return entries


def check_static(
    packages: Sequence[Package],
    everything: Sequence[Package],
    allowlist: Iterable[tuple[str, str, str, str]] = ALLOWLIST,
) -> list[Finding]:
    """The static findings of ``packages`` (``everything`` holds their possible siblings), ``allowlist`` and the
    ``VENDORED.json`` ``allow`` records applied. An allowlist entry of a package that is not vendored is stale whenever
    the whole tree is checked."""
    allowlist = [*allowlist, *vendored_allowlist(everything)]
    by_name = {package.name: package for package in everything}
    found = [finding for package in packages for finding in package_findings(package, by_name)]
    checked = {package.name for package in packages}
    if checked == set(by_name):
        checked |= {entry[0] for entry in allowlist}
    return apply_allowlist(found, allowlist, checked)


def blocked_for(package: str, allowlist: Iterable[tuple[str, str, str, str]] = ALLOWLIST) -> list[str]:
    """What a child of ``package`` blocks: flashinfer, miles_ext_kernels and the optional imports the allowlist
    lists."""
    optional = set()
    for name, _, subject, _ in allowlist:
        verb, _, module = subject.partition(" ")
        if name == package and verb == "import" and module.isidentifier() and module not in BLOCKED:
            optional.add(module)
    return [*BLOCKED, *sorted(optional)]


# ---------------------------------------------------------------------------------------------------------------
# the children

# Written to ``<work>/check_standalone_child.py`` and run as ``<python> [-I -B] <child> <mode> <config.json>``:
# ``probe`` (without ``-I``) reports where torch and tvm_ffi load from; ``imports`` and ``load`` (with ``-I``) isolate
# the copy and import it, and ``load`` also loads its compiled libraries.
CHILD = '''\
# Written by tools/check_standalone.py: one child of the standalone gate.
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
import traceback

MODE = sys.argv[1]
with open(sys.argv[2], encoding="utf-8") as config:
    CONFIG = json.load(config)
RESULT = "@RESULT@"


def emit(record):
    print(RESULT + json.dumps(record), flush=True)


def probe():
    import tvm_ffi.libinfo

    paths = [path for path in sys.path if path.endswith(("site-packages", "dist-packages"))]
    for module in ("torch", "tvm_ffi"):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            raise SystemExit(f"{module} is not importable by {sys.executable}")
        paths.append(os.path.dirname(os.path.dirname(os.path.abspath(spec.origin))))
    includes = [tvm_ffi.libinfo.find_include_path(), tvm_ffi.libinfo.find_dlpack_include_path()]
    emit({"paths": list(dict.fromkeys(paths)), "includes": includes})


BLOCKED = tuple(CONFIG.get("blocked", ()))
REPO = os.path.realpath(CONFIG.get("repo", os.sep))
ROOT = os.path.realpath(CONFIG.get("root", os.sep))


def blocked(name):
    top = name.partition(".")[0]
    return top in BLOCKED or top.startswith("flashinfer_")


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if blocked(name):
            raise ModuleNotFoundError(f"{name} is blocked by the standalone gate", name=name)
        return None


def isolate():
    cwd = os.path.realpath(os.getcwd())

    def exposes(entry):
        # the working directory, or a directory at or above the repository or the copy
        entry = os.path.realpath(entry or os.curdir)
        return entry == cwd or any(os.path.commonpath([entry, d]) == entry for d in (REPO, ROOT))

    sys.path[:] = [entry for entry in sys.path if not exposes(entry)]
    for entry in reversed(CONFIG["paths"]):
        if entry not in sys.path and not exposes(entry):
            sys.path.insert(0, entry)
    sys.path.insert(0, ROOT)
    sys.meta_path.insert(0, Block())
    for name in BLOCKED:
        sys.modules[name] = None


def import_all():
    """Import the package, then every module of it one by one; the failures."""
    failed = []
    package = CONFIG["package"]
    for name in [package] + [f"{package}.{stem}" for stem in CONFIG["modules"]]:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except (Exception, SystemExit) as exc:
            lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            failed.append({"module": name, "error": f"{type(exc).__name__}: {exc}", "traceback": "".join(lines)})
            try:
                spec = importlib.util.find_spec(name)
            except Exception:
                spec = None
            if spec is not None and spec.submodule_search_locations is not None:
                # an unexecuted package instead, so its modules still import one by one
                sys.modules[name] = importlib.util.module_from_spec(spec)
    return failed


def leaked():
    """Blocked modules loaded anyway, and modules loaded from the repository (an environment inside it aside)."""
    environment = tuple(os.path.realpath(path) + os.sep for path in CONFIG["paths"])
    names = []
    for name, module in list(sys.modules.items()):
        if blocked(name):
            if module is not None:
                names.append(name)
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        origin = os.path.realpath(origin)
        if origin.startswith(REPO + os.sep) and not origin.startswith(environment):
            names.append(name)
    return sorted(names)


def origin():
    module = sys.modules.get(CONFIG["package"])
    return getattr(module, "__file__", None) or list(getattr(module, "__path__", []))


def load():
    import torch  # noqa: F401  (the libraries' CUDA runtime resolves through it)

    failed = import_all()
    cake_build = sys.modules.get(CONFIG["package"] + ".cake_build")
    loads, records = [], []
    for entry in CONFIG["entries"] if cake_build is not None else ():
        name = entry["name"]
        want = os.path.realpath(os.path.join(CONFIG["directory"], entry["path"]))
        try:
            module = cake_build.load_library(name)
            got = os.path.realpath(str(cake_build.library_path(name)))
            built = int(module.build_cuda_version())
        except Exception as exc:
            loads.append({"name": name, "ok": False, "error": repr(exc)})
            continue
        ok = got == want and built == CONFIG["cudart"]
        loads.append({"name": name, "ok": ok, "path": got, "build_cuda_version": built})
    for arch in CONFIG["arches"] if cake_build is not None else ():
        try:
            got = list(cake_build.record_cuda_version(arch))
        except Exception as exc:
            records.append({"arch": arch, "ok": False, "error": repr(exc)})
            continue
        records.append({"arch": arch, "ok": got == CONFIG["cuda"], "value": got})
    return {"failed": failed, "loads": loads, "records": records}


if MODE == "probe":
    probe()
else:
    isolate()
    result = load() if MODE == "load" else {"failed": import_all()}
    emit({**result, "leaked": leaked(), "origin": origin()})
'''


def write_child(work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    path = work / CHILD_NAME
    path.write_text(CHILD.replace("@RESULT@", _RESULT))
    return path


def _tail(text: str | bytes | None, lines: int = TAIL_LINES) -> str:
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return "\n    ".join((text or "").strip().splitlines()[-lines:])


def run_child(
    python: str, child: Path, mode: str, config: Mapping[str, Any], cwd: Path, *, cpu_only: bool
) -> tuple[dict[str, Any] | None, str]:
    """The result record of one child, or None and what went wrong. The config goes to ``<work>/config/`` (a
    command-line argument is limited to 128 KiB)."""
    isolation = [] if mode == "probe" else ["-I", "-B"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="") if cpu_only else dict(os.environ)
    path = child.parent / "config" / f"{mode}-{config.get('package', mode)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    try:
        proc = subprocess.run(
            [python, *isolation, str(child), mode, str(path)],
            cwd=cwd,
            env=environment,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=CHILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        return None, f"the {mode} child did not finish in {CHILD_TIMEOUT} s:\n    {_tail(exc.stderr)}"
    except OSError as exc:
        return None, f"cannot run {python}: {exc}"
    _, found, record = proc.stdout.rpartition(_RESULT)
    if proc.returncode or not found:
        return None, f"the {mode} child exited {proc.returncode} without a result:\n    {_tail(proc.stderr)}"
    return json.loads(record.partition("\n")[0]), ""


def probe(python: str, work: Path) -> Environment:
    """Where ``python`` loads torch and tvm_ffi from, and tvm-ffi's include directories."""
    record, problem = run_child(python, write_child(work), "probe", {}, work, cpu_only=True)
    if record is None:
        raise GateError(f"{python} cannot report its torch and tvm_ffi: {problem}")
    return Environment(python, tuple(record["paths"]), tuple(record["includes"]))


def layout(package: Package, root: Path) -> tuple[str, Path]:
    """The import name and directory of the copy of ``package`` under ``root``: ``vendored_<name>``, or with siblings
    ``vendored_<name>.<name>`` (``<root>/vendored_<name>/`` then holds the siblings too)."""
    neutral = f"vendored_{package.name}"
    if package.siblings:
        return f"{neutral}.{package.name}", root / neutral / package.name
    return neutral, root / neutral


def module_stems(directory: Path) -> list[str]:
    """Every module of ``directory`` but its ``__init__.py``: ``sub/inner.py`` is ``sub.inner`` and
    ``sub/__init__.py`` is ``sub``; sorted, so a subpackage precedes its modules."""
    stems = []
    for relative in _python_files(directory):
        parts = PurePosixPath(relative).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            stems.append(".".join(parts))
    return sorted(stems)


def _copy(source: Path, destination: Path, *, aot: bool = False) -> None:
    ignore = ("__pycache__",) if aot else ("aot", "__pycache__")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*ignore))


def _fresh(directory: Path) -> Path:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def child_findings(package: str, record: dict[str, Any], copy: Path) -> list[Finding]:
    """The findings of a child's result record: failed modules, leaked modules, a package loaded from elsewhere."""
    found = []
    for failure in record["failed"]:
        tail = _tail(failure["traceback"])
        found.append(Finding(package, "", "import", f"{failure['module']}: {failure['error']}\n    {tail}"))
    if record["leaked"]:
        found.append(Finding(package, "", "leak", f"loaded {', '.join(record['leaked'])}"))
    origins = record["origin"] if isinstance(record["origin"], list) else [record["origin"]]
    if not origins or not all(origin and Path(origin).resolve().is_relative_to(copy.resolve()) for origin in origins):
        found.append(Finding(package, "", "import", f"imported from {origins}, not from the copy {copy}"))
    return found


def check_imports(
    packages: Sequence[Package],
    everything: Sequence[Package],
    environment: Environment,
    work: Path,
    repo_root: Path = REPO_ROOT,
    allowlist: Iterable[tuple[str, str, str, str]] = ALLOWLIST,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    """Import every module of each package from a blocked copy; a report per package and the findings."""
    allowlist = list(allowlist)
    by_name = {package.name: package for package in everything}
    child = write_child(work)
    reports, found = [], []
    for package in packages:
        root = _fresh(work / "imports" / package.name)
        name, copy = layout(package, root)
        _copy(package.directory, copy)
        problems = sibling_findings(package, by_name)
        for sibling in package.siblings:
            if sibling in by_name:
                _copy(by_name[sibling].directory, copy.parent / sibling)
        stems = module_stems(copy)
        config = {
            "blocked": blocked_for(package.name, allowlist),
            "modules": stems,
            "package": name,
            "paths": list(environment.paths),
            "repo": str(repo_root),
            "root": str(root),
        }
        record, problem = run_child(environment.python, child, "imports", config, work, cpu_only=True)
        if record is None:
            problems.append(Finding(package.name, "", "import", problem))
        else:
            problems.extend(child_findings(package.name, record, copy))
        failed = len(record["failed"]) if record is not None else len(stems) + 1
        reports.append({"package": package.name, "copy": str(copy), "modules": len(stems) + 1, "failed": failed})
        found.extend(problems)
    return reports, found


# ---------------------------------------------------------------------------------------------------------------
# build


def find_nvcc(explicit: str | None) -> str:
    """``explicit``, else ``$CUDA_HOME/bin/nvcc`` when it exists, else ``nvcc`` on ``PATH``; a path is made
    absolute."""
    if not explicit:
        cuda_home = os.environ.get("CUDA_HOME")
        candidate = Path(cuda_home) / "bin" / "nvcc" if cuda_home else None
        explicit = str(candidate) if candidate is not None and candidate.is_file() else shutil.which("nvcc")
        if explicit is None:
            raise GateError("no nvcc: pass --nvcc, set CUDA_HOME or put nvcc on PATH")
    return os.path.abspath(explicit) if os.sep in explicit else explicit


def toolkit(nvcc: str) -> Toolkit:
    """The toolkit of ``nvcc --version``."""
    try:
        proc = subprocess.run(
            [nvcc, "--version"], capture_output=True, encoding="utf-8", errors="replace", check=False
        )
    except OSError as exc:
        raise GateError(f"cannot run {nvcc} --version: {exc}") from exc
    release = _NVCC_RELEASE.search(proc.stdout)
    build = _NVCC_BUILD.search(proc.stdout)
    if proc.returncode or release is None or build is None:
        raise GateError(f"{nvcc} --version (exit {proc.returncode}) reports no 'release X.Y, VX.Y.Z'")
    line = proc.stdout.strip().splitlines()[-1].strip()
    return Toolkit(nvcc, (int(release.group(1)), int(release.group(2))), build.group(1), line)


def _version(text: str) -> tuple[int, int]:
    major, minor = str(text).split(".")[:2]
    return (int(major), int(minor))


def run_compile(command: list[str], copy: Path) -> dict[str, Any]:
    """One nvcc run from ``copy``: ``rc``, ``seconds`` and, on failure, the first lines of ``stderr`` (``rc`` None and
    ``error`` when nvcc cannot start)."""
    start = time.monotonic()
    try:
        proc = subprocess.run(command, cwd=copy, capture_output=True, encoding="utf-8", errors="replace", check=False)
    except OSError as exc:
        return {"rc": None, "seconds": round(time.monotonic() - start, 1), "error": f"cannot run {command[0]}: {exc}"}
    record: dict[str, Any] = {"rc": proc.returncode, "seconds": round(time.monotonic() - start, 1)}
    if proc.returncode:
        record["stderr"] = proc.stderr.splitlines()[:NVCC_STDERR_LINES]
    return record


def compile_findings(package: str, copy: Path, records: Sequence[dict[str, Any]]) -> list[Finding]:
    found = []
    for record in records:
        name = record["name"]
        if record["rc"] is None:
            found.append(Finding(package, "", "compile", f"{name}: {record['error']}"))
        elif record["rc"]:
            tail = "\n    ".join(record.get("stderr") or ["(no stderr)"])
            found.append(Finding(package, "", "compile", f"{name}: nvcc exit {record['rc']}\n    {tail}"))
        elif not record["output"]:
            message = f"{name}: nvcc exit 0 but {copy / record['path']} is missing"
            found.append(Finding(package, "", "compile", message))
    return found


def load_findings(package: str, record: dict[str, Any], copy: Path) -> list[Finding]:
    found = child_findings(package, record, copy)
    for load in record["loads"]:
        if not load["ok"]:
            detail = load.get("error") or f"path {load['path']}, build_cuda_version {load['build_cuda_version']}"
            found.append(Finding(package, "", "load", f"{load['name']}: {detail}"))
    for item in record["records"]:
        if not item["ok"]:
            detail = item.get("error") or f"record_cuda_version {item['value']}"
            found.append(Finding(package, "", "load", f"{item['arch']}: {detail}"))
    return found


def _closure(packages: Sequence[Package], by_name: Mapping[str, Package]) -> list[Package]:
    """``packages`` and the vendored siblings they declare."""
    names = {package.name for package in packages}
    names |= {sibling for package in packages for sibling in package.siblings if sibling in by_name}
    return [by_name[name] for name in sorted(names)]


def plan_build(
    packages: Sequence[Package], arches: Sequence[str], toolkits: Mapping[str, Toolkit]
) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """``name -> (build.json, the entries of arches)`` of every package; a toolkit older than :data:`MIN_CUDA`, an
    entry's ``min_cuda`` or a malformed ``build.json`` raises :class:`GateError` before anything compiles."""
    for arch in arches:
        kit = toolkits[arch]
        if kit.cuda < MIN_CUDA[arch]:
            need = ".".join(map(str, MIN_CUDA[arch]))
            raise GateError(
                f"{arch} builds with CUDA {need} or newer, but {kit.nvcc} is CUDA {kit.full}: pass --nvcc-for"
            )
    plan, above = {}, []
    for package in packages:
        build, error = _read_json(package.directory / "build.json")
        if error is not None:
            raise GateError(f"{package.directory / 'build.json'}: {error}")
        try:
            entries = [entry for entry in build["libraries"] if entry["arch"] in arches]
            above.extend(
                f"{package.name}/{entry['name']} (min_cuda {entry['min_cuda']}, {toolkits[entry['arch']].nvcc} is "
                f"{toolkits[entry['arch']].full})"
                for entry in entries
                if _version(entry["min_cuda"]) > toolkits[entry["arch"]].cuda
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GateError(f"{package.directory / 'build.json'}: malformed library entry ({exc!r})") from exc
        plan[package.name] = (build, entries)
    if above:
        more = f" and {len(above) - 10} more" if len(above) > 10 else ""
        raise GateError(f"entries above their toolkit: {'; '.join(above[:10])}{more}: pass --nvcc-for")
    return plan


def check_build(
    packages: Sequence[Package],
    everything: Sequence[Package],
    environment: Environment,
    work: Path,
    arches: Sequence[str],
    toolkits: Mapping[str, Toolkit],
    *,
    jobs: int,
    load: bool = True,
    repo_root: Path = REPO_ROOT,
    allowlist: Iterable[tuple[str, str, str, str]] = ALLOWLIST,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    """Compile and load-check ``packages`` and their siblings for ``arches``, ``toolkits[arch]`` per architecture."""
    allowlist = list(allowlist)
    by_name = {package.name: package for package in everything}
    closure = _closure(packages, by_name)
    plan = plan_build(closure, arches, toolkits)
    child = write_child(work)
    reports: list[dict[str, Any]] = []
    found = [finding for package in closure for finding in sibling_findings(package, by_name)]
    groups: dict[Toolkit, list[str]] = {}
    for arch in arches:
        groups.setdefault(toolkits[arch], []).append(arch)
    directories: set[str] = set()
    for kit, group_arches in groups.items():
        label = f"cuda-{kit.full}" if f"cuda-{kit.full}" not in directories else f"cuda-{kit.full}-{len(directories)}"
        directories.add(label)
        group = _fresh(work / "build" / label)
        copies: dict[str, tuple[str, Path, Path]] = {}
        pending: dict[str, list[tuple[dict[str, Any], Future[dict[str, Any]] | None]]] = {}
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for package in closure:
                root = group / package.name
                name, copy = layout(package, root)
                _copy(package.directory, copy)
                (copy / "aot").mkdir()
                copies[package.name] = (name, copy, root)
                build, entries = plan[package.name]
                pending[package.name] = []
                for entry in entries:
                    if entry["arch"] not in group_arches:
                        continue
                    record: dict[str, Any] = {"name": entry["name"], "arch": entry["arch"], "path": entry["path"]}
                    try:
                        command = compile_command(kit.nvcc, copy, build, entry, kit.cuda, environment.includes)
                    except (KeyError, TypeError, ValueError) as exc:
                        record.update(rc=None, error=f"cannot form the nvcc command: {exc!r}")
                        pending[package.name].append((record, None))
                        continue
                    record["command"] = command
                    pending[package.name].append((record, pool.submit(run_compile, command, copy)))
            compiled: dict[str, list[dict[str, Any]]] = {}
            for package in closure:
                _, copy, _ = copies[package.name]
                records = []
                for record, future in pending[package.name]:
                    if future is not None:
                        record.update(future.result())
                    record["output"] = (copy / record["path"]).is_file()
                    records.append(record)
                found.extend(compile_findings(package.name, copy, records))
                compiled[package.name] = records
        for package in closure:
            _, copy, _ = copies[package.name]
            ok = [record for record in compiled[package.name] if record["rc"] == 0 and record["output"]]
            built = sorted({record["arch"] for record in ok})
            if not compiled[package.name]:
                found.append(Finding(package.name, "build.json", "build", f"has no library for {group_arches}"))
            if built:
                (copy / "aot" / "build_info.json").write_bytes(render_build_info(kit.full, kit.cudart, built))
        for package in closure:
            name, copy, root = copies[package.name]
            for sibling in package.siblings:
                if sibling in copies:
                    _copy(copies[sibling][1], copy.parent / sibling, aot=True)
            ok = [record for record in compiled[package.name] if record["rc"] == 0 and record["output"]]
            report: dict[str, Any] = {
                "package": package.name,
                "toolkit": kit.full,
                "nvcc": kit.nvcc,
                "nvcc_version": kit.line,
                "arches": group_arches,
                "copy": str(copy),
                "compiled": compiled[package.name],
                "load": None,
            }
            if load and ok:
                config = {
                    "arches": sorted({record["arch"] for record in ok}),
                    "blocked": blocked_for(package.name, allowlist),
                    "cuda": list(kit.cuda),
                    "cudart": kit.cudart,
                    "directory": str(copy),
                    "entries": [{"name": record["name"], "path": record["path"]} for record in ok],
                    "modules": module_stems(copy),
                    "package": name,
                    "paths": list(environment.paths),
                    "repo": str(repo_root),
                    "root": str(root),
                }
                record, problem = run_child(environment.python, child, "load", config, work, cpu_only=False)
                if record is None:
                    found.append(Finding(package.name, "", "load", problem))
                else:
                    found.extend(load_findings(package.name, record, copy))
                report["load"] = record
            reports.append(report)
    return reports, found


def summary(command: str, report: dict[str, Any]) -> str:
    """One line per package report."""
    if command == "imports":
        modules, failed = report["modules"], report["failed"]
        return f"{report['package']}: imported {modules - failed}/{modules} modules"
    records = report["compiled"]
    ok = sum(1 for record in records if record["rc"] == 0 and record["output"])
    line = f"{report['package']} [CUDA {report['toolkit']} {' '.join(report['arches'])}]: compiled {ok}/{len(records)}"
    if report["load"] is not None:
        loads = report["load"]["loads"]
        line += f", loaded {sum(1 for load in loads if load['ok'])}/{ok}"
    return line


# ---------------------------------------------------------------------------------------------------------------
# command line


def _outside(work: Path, repo_root: Path) -> Path:
    work, repo = work.resolve(), repo_root.resolve()
    if work == repo or work.is_relative_to(repo):
        raise GateError(f"the work directory {work} is inside the repository {repo}: use one outside it")
    work.mkdir(parents=True, exist_ok=True)
    return work


def _cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def _nvcc_for(items: Iterable[str]) -> dict[str, str]:
    toolkits = {}
    for item in items:
        arch, sep, path = item.partition("=")
        if not sep or arch not in ARCHES or not path:
            raise GateError(f"--nvcc-for {item}: expected ARCH=PATH with ARCH one of {', '.join(ARCHES)}")
        toolkits[arch] = path
    return toolkits


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="this repository's checkout")
    common.add_argument("--package", action="append", metavar="NAME", help="a vendored package (default: every one)")
    common.add_argument("--report-json", type=Path, metavar="PATH", help="write the reports and findings here")
    children = argparse.ArgumentParser(add_help=False)
    children.add_argument("--python", default=sys.executable, help="the interpreter of the children (torch, tvm_ffi)")
    parser = argparse.ArgumentParser(description="Check that every vendored Cake package works as a copy on its own.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("static", parents=[common], help="read the sources: imports, environment reads, registries")
    imports = commands.add_parser("imports", parents=[common, children], help="import every module from a copy")
    imports.add_argument("--work-dir", type=Path, help="outside the repository (default: a temporary directory)")
    build = commands.add_parser("build", parents=[common, children], help="compile into a copy and load from it")
    build.add_argument("--work-dir", type=Path, required=True, help="outside the repository, on scratch")
    build.add_argument("--arch", action="append", choices=ARCHES, help="an architecture (default: every one)")
    build.add_argument("--jobs", type=int, default=_cpus(), help="nvcc runs at a time (default: the usable CPUs)")
    build.add_argument("--nvcc", help="default: $CUDA_HOME/bin/nvcc, else nvcc on PATH")
    build.add_argument("--nvcc-for", action="append", default=[], metavar="ARCH=PATH", help="another nvcc for ARCH")
    build.add_argument("--compile-only", action="store_true", help="compile, but load nothing (no GPU needed)")
    args = parser.parse_args(argv)

    reports: list[dict[str, Any]] = []
    findings: list[Finding] = []
    temporary = None
    try:
        everything = discover(args.repo_root)
        packages = select(everything, args.package)
        if args.command == "static":
            findings = check_static(packages, everything)
        else:
            if args.work_dir is None:
                temporary = Path(tempfile.mkdtemp(prefix="check_standalone-"))
            work = _outside(args.work_dir or temporary, args.repo_root)
            if args.command == "imports":
                environment = probe(args.python, work)
                reports, findings = check_imports(packages, everything, environment, work, args.repo_root)
            else:
                arches = sorted(set(args.arch or ARCHES))
                nvcc_for = _nvcc_for(args.nvcc_for)
                if unused := sorted(set(nvcc_for) - set(arches)):
                    raise GateError(f"--nvcc-for names {', '.join(unused)}, which this run does not build")
                nvcc_for = {arch: find_nvcc(path) for arch, path in nvcc_for.items()}
                # the default nvcc only when an architecture has no --nvcc-for
                default = find_nvcc(args.nvcc) if set(arches) - set(nvcc_for) else None
                kits = {path: toolkit(path) for path in {*nvcc_for.values(), default} if path is not None}
                toolkits = {arch: kits[nvcc_for.get(arch, default)] for arch in arches}
                by_name = {package.name: package for package in everything}
                plan_build(_closure(packages, by_name), arches, toolkits)  # refuse before the probe starts a child
                environment = probe(args.python, work)
                reports, findings = check_build(
                    packages,
                    everything,
                    environment,
                    work,
                    arches,
                    toolkits,
                    jobs=args.jobs,
                    load=not args.compile_only,
                    repo_root=args.repo_root,
                )
    except GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        _write_report(args.report_json, args.command, reports, [], str(exc), 2)
        return 2
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    for report in reports:
        print(summary(args.command, report))
    for finding in findings:
        print(f"  {finding.line()}")
    scope = "every package" if not args.package else ", ".join(sorted(set(args.package)))
    print(f"standalone {args.command} ({scope}): {f'{len(findings)} findings' if findings else 'ok'}")
    code = 1 if findings else 0
    _write_report(args.report_json, args.command, reports, findings, None, code)
    return code


def _write_report(
    path: Path | None,
    command: str,
    reports: Sequence[dict[str, Any]],
    findings: Sequence[Finding],
    error: str | None,
    code: int,
) -> None:
    if path is None:
        return
    document = {
        "command": command,
        "error": error,
        "exit": code,
        "findings": [finding._asdict() for finding in findings],
        "reports": list(reports),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
