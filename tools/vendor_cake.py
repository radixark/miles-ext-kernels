#!/usr/bin/env python3
"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The one way sources enter ``miles_ext_kernels/cake/_vendor/``: vendor them byte for byte from a git commit, record them
in ``_vendor/VENDORED.json``, and check the tree against that record. Needs git and Python only (no torch).

Two kinds of source:

- ``fi``: Cake packages of FlashInfer (``flashinfer/<name>``, ``flashinfer/experimental/<name>`` or
  ``flashinfer/experimental/cake/<name>``), each to ``_vendor/<name>/``: every file of the directory except ``aot/``
  and ``__pycache__/``.
- ``miles-pr``: generated kernel sources of a miles pull request (a registry and its ``csrc/``), to
  ``_vendor/cake_<op>_<purpose>/``. Next to the upstream files the tool writes four local files, so the package builds
  through the same loader as a FlashInfer one: an empty ``__init__.py``, FlashInfer's loader ``cake_build.py``
  (``tools/templates/cake_build.py``), the TVM-FFI host surface ``csrc/cake_host_shim.h``
  (``tools/templates/cake_host_shim.h``, in place of the upstream torch-extension surface, which is excluded) and a
  ``build.json`` (schema ``cake-build/1``) generated from the registry.

Commands::

    python tools/vendor_cake.py fi --checkout PATH --commit SHA --op OP --package FI_PATH [--package FI_PATH ...]
        [--repo URL] [--allow-env-read PKG:MODULE ...] [--allow-flashinfer-import PKG:MODULE:FUNCTION ...]
    python tools/vendor_cake.py miles-pr --checkout PATH --pr N --commit SHA --op OP --name PKG --path DIR
        --include REL [--include REL ...] [--exclude REL ...] [--arch-from DST=SRC ...]
        [--nvcc-flag=FLAG ...] [--link-flag=FLAG ...] [--repo URL]
    python tools/vendor_cake.py remove --op OP --name PKG
    python tools/vendor_cake.py --check [OP ...] [--upstream] [--upstream-cache DIR]
    python tools/vendor_cake.py merge BASE OURS THEIRS

``fi`` and ``miles-pr`` read the commit with ``git ls-tree`` / ``git cat-file`` (never the working tree), rewrite each
named package whole, update its entry under ``ops.<op>`` (the op's other packages stay) and run ``--check OP``.
``remove`` deletes a package and its entry. ``VENDORED.json`` is written by this tool only, as
``json.dumps(manifest, indent=2, sort_keys=True)`` plus a newline::

    {"schema": "miles-ext-kernels-vendored/1",
     "ops": {<op>: {<package>: {
         "source": {"kind": "flashinfer" | "miles-pr", "repo": <URL>, "commit": <40-hex>, "path": <upstream dir>,
                    miles-pr only: "pr": <int>, "include": [<path under it>], "exclude": {<path>: <upstream blob>}},
         "cake_revision": <build.json generator.cake_revision> (miles-pr: null),
         "files": {<package-relative path>: <upstream git blob>},
         miles-pr only: "local": {<path>: <blob>}, "build": {"registry", "arch_from", "nvcc_flags", "link_flags"},
         optional: "allow": {"env_read": [<module>], "flashinfer_import": [<module>:<function>]}}}}}

``--check`` (offline) verifies the manifest's form, that every package name is unique and ``_vendor/`` holds nothing
else, the exact file set and every blob of each package, no ``aot/``, FlashInfer's loader in every package, each
``build.json`` (schema, Cake revision, the sha256 of every source, sm_100a / sm_103a / sm_107a libraries; the generated
ones byte for byte), one FlashInfer commit per op, that a vendored module imports other vendored packages of its own op
only (``..<name>``, or ``..cake.<name>``, FlashInfer's path of some packages, which the runtime resolves to
``_vendor.<name>``), and no environment read or flashinfer import in a vendored Python module unless ``allow`` lists
it. A flashinfer import is allowed only inside a listed function, a hook the op's implementation rebinds (its
contract test checks that it does). ``--upstream`` fetches every recorded commit (trees only) and checks that the
recorded files are exactly the upstream directory (``include`` minus ``exclude`` for a miles pull request) with the
same blobs, and the excluded blobs.

``merge`` is the git merge driver of ``VENDORED.json`` (``.gitattributes``): a three-way merge package by package.
A package changed on one side takes that side; changed on both sides, it must be the same. It writes the result to
OURS in this tool's form, or reports the conflicts and exits 1.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent / "templates"
VENDOR = PurePosixPath("miles_ext_kernels/cake/_vendor")
MANIFEST = "VENDORED.json"
SCHEMA = "miles-ext-kernels-vendored/1"
FI_REPO = "https://github.com/flashinfer-ai/flashinfer"
MILES_REPO = "https://github.com/radixark/miles"
# git blob of FlashInfer's cake_build.py, the loader every vendored package carries byte-identically
CAKE_BUILD_BLOB = "00756137eb159e30a77907eda84fd245123d87ea"
BUILD_SCHEMA = "cake-build/1"
REQUIRED_ARCHES = ("sm_100a", "sm_103a", "sm_107a")
MIN_CUDA = {"sm_100a": "12.8", "sm_103a": "12.9", "sm_107a": "13.4"}
DEFAULT_NVCC_FLAGS = ("-std=c++17", "-O3", "--compiler-options=-fPIC", "--expt-relaxed-constexpr")
DEFAULT_LINK_FLAGS = ("-shared", "-lcudart", "-lcuda")
FI_PARENTS = ("flashinfer", "flashinfer/experimental", "flashinfer/experimental/cake")
REGISTRY = "registry.json"
SHIM = "csrc/cake_host_shim.h"
LOCAL_FILES = ("__init__.py", "build.json", "cake_build.py", SHIM)
# the name FlashInfer's ``..cake.<name>`` imports reach from a vendored package; the runtime resolves it
ALIAS = "cake"
ALLOW_KEYS = ("env_read", "flashinfer_import")
EMPTY_BLOB = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"

_OP = re.compile(r"[a-z][a-z0-9_]*")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ENV_NAMES = ("environ", "environb", "getenv", "getenvb")


def git_blob_sha1(data: bytes) -> str:
    """The git object id of a blob holding ``data``."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def canonical(document: Any) -> str:
    """The one text form of ``VENDORED.json`` and of a generated ``build.json``."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _template(name: str) -> bytes:
    return (TEMPLATES / name).read_bytes()


# ---------------------------------------------------------------------------------------------------------------
# git


def _git(git_dir: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(git_dir), *args], capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed (exit {proc.returncode}): {proc.stderr.decode().strip()}")
    return proc.stdout


def _commit(git_dir: Path, commit: str) -> str:
    return _git(git_dir, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()


def _tree(git_dir: Path, commit: str, path: str) -> dict[str, tuple[str, str]]:
    """``path-relative file -> (mode, blob)`` under ``path`` at ``commit``, without ``aot/`` and ``__pycache__/``."""
    prefix = path.strip("/") + "/"
    files = {}
    for record in _git(git_dir, "ls-tree", "-r", "-z", "--full-tree", commit, "--", prefix[:-1]).split(b"\0"):
        if not record:
            continue
        meta, name = record.decode().split("\t", 1)
        if not name.startswith(prefix):
            continue
        mode, kind, blob = meta.split()
        relative = name[len(prefix) :]
        parts = PurePosixPath(relative).parts
        if parts[0] == "aot" or "__pycache__" in parts:
            continue
        if kind != "blob" or mode not in ("100644", "100755"):
            raise RuntimeError(f"{name}: cannot vendor a {kind} of mode {mode}")
        files[relative] = (mode, blob)
    if not files:
        raise RuntimeError(f"{path} has no files at {commit}")
    return files


class _BlobReader:
    """``git cat-file --batch``: one process for every blob of a run."""

    def __init__(self, git_dir: Path) -> None:
        self._proc = subprocess.Popen(
            ["git", "-C", str(git_dir), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

    def read(self, blob: str) -> bytes:
        self._proc.stdin.write(blob.encode() + b"\n")
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().split()
        if len(header) != 3 or header[0].decode() != blob or header[1] != b"blob":
            raise RuntimeError(f"git cat-file cannot read blob {blob}: {b' '.join(header).decode()}")
        data = self._proc.stdout.read(int(header[2]))
        self._proc.stdout.read(1)  # the newline after the content
        if git_blob_sha1(data) != blob:
            raise RuntimeError(f"blob {blob}: content does not hash to its id")
        return data

    def close(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()


# ---------------------------------------------------------------------------------------------------------------
# the manifest and the generated files


def fi_package_name(fi_path: str) -> str:
    """The package name of the FlashInfer package directory ``fi_path``."""
    path = PurePosixPath(fi_path.strip("/"))
    if str(path.parent) not in FI_PARENTS or path.name == "cake":
        raise ValueError(f"{fi_path}: expected a package directly under one of {', '.join(FI_PARENTS)}")
    return path.name


def select(
    tree: Mapping[str, tuple[str, str]], include: Iterable[str], exclude: Iterable[str]
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """The files of ``tree`` under ``include`` (files or directories) less ``exclude``, and the excluded blobs."""
    selected = {}
    for item in include:
        item = item.strip("/")
        matched = {path: meta for path, meta in tree.items() if path == item or path.startswith(item + "/")}
        if not matched:
            raise ValueError(f"--include {item}: no such file or directory upstream")
        selected.update(matched)
    excluded = {}
    for item in exclude:
        if item not in selected:
            raise ValueError(f"--exclude {item}: not an included upstream file")
        excluded[item] = selected.pop(item)[1]
    return dict(sorted(selected.items())), dict(sorted(excluded.items()))


def render_build_json(package: str, files: Mapping[str, bytes], build: Mapping[str, Any]) -> bytes:
    """The ``cake-build/1`` manifest of a miles-pr package from its registry; ``files`` holds every package file.

    One library per registry stage and architecture: the record's own architectures plus every ``arch_from`` target
    whose source architecture the record has (it compiles that architecture's record for the target). A record's
    ``per_arch`` entry overrides its fields for that architecture. The closure digest covers the flags, every header
    under ``csrc/`` and the sources, so a changed host surface or source names a new library.
    """
    registry = json.loads(files[build["registry"]])
    arch_from = dict(build["arch_from"])
    common = {
        "include_dirs": ["csrc"],
        "link_flags": list(build["link_flags"]),
        "nvcc_flags": list(build["nvcc_flags"]),
        "nvcc_flags_if": [],
    }
    headers = {
        path: _sha256(data)
        for path, data in files.items()
        if path.startswith("csrc/") and path.endswith((".h", ".cuh"))
    }
    libraries = []
    for stage in sorted(registry):
        record = registry[stage]
        arches = list(record["arches"])
        targets = sorted({*arches, *(target for target, source in arch_from.items() if source in arches)})
        for arch in targets:
            source_arch = arch if arch in arches else arch_from[arch]
            entry = {**record, **record.get("per_arch", {}).get(source_arch, {})}
            name = f"{entry['name']}_{arch}"
            sources = [f"csrc/{source}" for source in entry["sources"]]
            digests = {source: _sha256(files[source]) for source in sources}
            flags = [*common["nvcc_flags"], *entry["compile_flags"]]
            closure = json.dumps({"flags": flags, "headers": headers, "sources": digests}, sort_keys=True)
            libraries.append(
                {
                    "arch": arch,
                    "closure_sha256": _sha256(closure.encode()),
                    "entry": entry["ffi_entry"],
                    "gencode": f"-gencode=arch=compute_{arch[3:]},code={arch}",
                    "min_cuda": MIN_CUDA[arch],
                    "name": name,
                    "nvcc_flags": list(entry["compile_flags"]),
                    "nvcc_flags_if": [],
                    "path": f"aot/{name}.so",
                    "sha256": digests,
                    "sources": sources,
                }
            )
    names = [library["name"] for library in libraries]
    if len(set(names)) != len(names):
        raise ValueError(f"{package}: two registry stages name the same library")
    document = {
        "arch_flags": {arch: [f"-gencode=arch=compute_{arch[3:]},code={arch}"] for arch in REQUIRED_ARCHES},
        "common": common,
        "generator": {"cake_revision": None, "from": build["registry"]},
        "libraries": libraries,
        "package": f"{VENDOR.as_posix().replace('/', '.')}.{package}",
        "schema": BUILD_SCHEMA,
    }
    return canonical(document).encode()


def package_imports(source: bytes, depth: int) -> set[str]:
    """The vendored packages a module imports, and ``..`` for an import above ``_vendor``.

    That is the ``<name>`` of every ``from ..<name>[...] import``, ``from .. import <name>``,
    ``from ..cake.<name>[...] import`` and ``from ..cake import <name>`` reaching ``_vendor`` (``depth`` levels up).
    """
    names = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or node.level < depth:
            continue
        if node.level > depth:
            names.add("..")
            continue
        parts = node.module.split(".") if node.module else []
        if parts[:1] == [ALIAS]:
            parts = parts[1:]
        names.update(parts[:1] or [alias.name for alias in node.names])
    return names


def load_manifest(repo_root: Path) -> dict[str, Any]:
    return json.loads((repo_root / VENDOR / MANIFEST).read_text(encoding="utf-8"))


def _write(repo_root: Path, manifest: dict[str, Any]) -> None:
    (repo_root / VENDOR / MANIFEST).write_text(canonical(manifest), encoding="utf-8")


def _owner(manifest: Mapping[str, Any], package: str) -> str | None:
    return next((op for op, block in manifest["ops"].items() if package in block), None)


def _check_owner(manifest: Mapping[str, Any], op: str, package: str) -> None:
    if not _OP.fullmatch(op):
        raise ValueError(f"op {op!r} is not a module name")
    owner = _owner(manifest, package)
    if owner not in (None, op):
        raise ValueError(f"{package} is vendored for {owner}; a package belongs to one op")


def _place(repo_root: Path, manifest: dict[str, Any], op: str, package: str) -> Path:
    _check_owner(manifest, op, package)
    directory = repo_root / VENDOR / package
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def _write_files(directory: Path, files: Mapping[str, tuple[str, bytes]]) -> None:
    for relative, (mode, data) in files.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if mode == "100755":
            target.chmod(0o755)


def _allow(
    items: Iterable[str], packages: Mapping[str, Mapping[str, str]], what: str, *, function: bool = False
) -> dict[str, list[str]]:
    """``PKG:MODULE`` (``function``: ``PKG:MODULE:FUNCTION``) items -> package -> ``MODULE[:FUNCTION]``, each naming a
    vendored Python module of a package being vendored."""
    allowed: dict[str, list[str]] = {}
    form = "PKG:MODULE:FUNCTION" if function else "PKG:MODULE"
    for item in items:
        package, _, rest = item.partition(":")
        module, sep, name = rest.partition(":")
        if (
            package not in packages
            or module not in packages[package]
            or not module.endswith(".py")
            or bool(sep) != function
            or (function and not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", name))
        ):
            raise ValueError(f"--allow-{what} {item}: expected {form}, a Python module of a package vendored here")
        allowed.setdefault(package, []).append(rest)
    return {package: sorted(set(entries)) for package, entries in allowed.items()}


# ---------------------------------------------------------------------------------------------------------------
# commands


def vendor_fi(
    checkout: Path,
    commit: str,
    op: str,
    fi_paths: Iterable[str],
    *,
    repo_root: Path = REPO_ROOT,
    repo: str = FI_REPO,
    allow_env_read: Iterable[str] = (),
    allow_flashinfer_import: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Vendor the FlashInfer packages ``fi_paths`` of ``commit`` for ``op``; returns their manifest entries."""
    manifest = load_manifest(repo_root)
    full = _commit(checkout, commit)
    fi_paths = list(fi_paths)
    trees = {fi_package_name(path): (path.strip("/"), _tree(checkout, full, path)) for path in fi_paths}
    if len(trees) != len(fi_paths):
        raise ValueError(f"two of {fi_paths} name the same package")
    for package in trees:
        _check_owner(manifest, op, package)
    files_of = {
        package: {relative: meta[1] for relative, meta in tree.items()} for package, (_, tree) in trees.items()
    }
    allow = {
        "env_read": _allow(allow_env_read, files_of, "env-read"),
        "flashinfer_import": _allow(allow_flashinfer_import, files_of, "flashinfer-import", function=True),
    }
    reader = _BlobReader(checkout)
    entries = {}
    try:
        for package, (fi_path, tree) in trees.items():
            directory = _place(repo_root, manifest, op, package)
            data = {relative: (mode, reader.read(blob)) for relative, (mode, blob) in tree.items()}
            if "build.json" not in data:
                raise RuntimeError(f"{fi_path} has no build.json: not a Cake package")
            _write_files(directory, data)
            entry = {
                "cake_revision": json.loads(data["build.json"][1])["generator"]["cake_revision"],
                "files": dict(sorted(files_of[package].items())),
                "source": {"commit": full, "kind": "flashinfer", "path": fi_path, "repo": repo},
            }
            package_allow = {key: allow[key][package] for key in ALLOW_KEYS if package in allow[key]}
            if package_allow:
                entry["allow"] = package_allow
            manifest["ops"].setdefault(op, {})[package] = entries[package] = entry
    finally:
        reader.close()
    _write(repo_root, manifest)
    return entries


def vendor_miles_pr(
    checkout: Path,
    commit: str,
    *,
    pr: int,
    op: str,
    name: str,
    path: str,
    include: Iterable[str],
    exclude: Iterable[str] = (),
    arch_from: Mapping[str, str] | None = None,
    nvcc_flags: Iterable[str] = DEFAULT_NVCC_FLAGS,
    link_flags: Iterable[str] = DEFAULT_LINK_FLAGS,
    repo_root: Path = REPO_ROOT,
    repo: str = MILES_REPO,
) -> dict[str, Any]:
    """Vendor the generated sources of a miles pull request as the package ``name`` of ``op``; returns its entry."""
    if not re.fullmatch(rf"cake_{op}_[a-z0-9_]+", name):
        raise ValueError(f"a miles-pr package of {op} is named cake_{op}_<purpose>, not {name!r}")
    arch_from = dict(arch_from or {})
    include = list(include)
    if unknown := sorted({*arch_from, *arch_from.values()} - set(REQUIRED_ARCHES)):
        raise ValueError(f"--arch-from names unknown architectures {unknown}")
    manifest = load_manifest(repo_root)
    _check_owner(manifest, op, name)
    full = _commit(checkout, commit)
    selected, excluded = select(_tree(checkout, full, path), include, exclude)
    if clash := sorted(set(selected) & set(LOCAL_FILES)):
        raise ValueError(f"upstream {clash} would replace the generated local files: --exclude them")
    if REGISTRY not in selected:
        raise ValueError(f"{name}: the included files hold no {REGISTRY}")
    reader = _BlobReader(checkout)
    try:
        data = {relative: (mode, reader.read(blob)) for relative, (mode, blob) in selected.items()}
    finally:
        reader.close()
    build = {
        "arch_from": dict(sorted(arch_from.items())),
        "link_flags": list(link_flags),
        "nvcc_flags": list(nvcc_flags),
        "registry": REGISTRY,
    }
    local = _local_files(name, {relative: content for relative, (_, content) in data.items()}, build)
    directory = _place(repo_root, manifest, op, name)
    _write_files(directory, {**data, **{relative: ("100644", content) for relative, content in local.items()}})
    entry = {
        "build": build,
        "cake_revision": None,
        "files": {relative: blob for relative, (_, blob) in selected.items()},
        "local": {relative: git_blob_sha1(content) for relative, content in sorted(local.items())},
        "source": {
            "commit": full,
            "exclude": excluded,
            "include": sorted(item.strip("/") for item in include),
            "kind": "miles-pr",
            "path": path.strip("/"),
            "pr": int(pr),
            "repo": repo,
        },
    }
    manifest["ops"].setdefault(op, {})[name] = entry
    _write(repo_root, manifest)
    return entry


def _local_files(package: str, upstream: Mapping[str, bytes], build: Mapping[str, Any]) -> dict[str, bytes]:
    """The four files the tool writes into a miles-pr package, from its upstream files."""
    local = {"__init__.py": b"", "cake_build.py": _template("cake_build.py"), SHIM: _template("cake_host_shim.h")}
    local["build.json"] = render_build_json(package, {**upstream, **local}, build)
    return local


def remove(op: str, name: str, *, repo_root: Path = REPO_ROOT) -> None:
    """Delete the package ``name`` of ``op`` and its manifest entry."""
    manifest = load_manifest(repo_root)
    if name not in manifest["ops"].get(op, {}):
        raise ValueError(f"{op} vendors no package {name}")
    del manifest["ops"][op][name]
    if not manifest["ops"][op]:
        del manifest["ops"][op]
    shutil.rmtree(repo_root / VENDOR / name, ignore_errors=True)
    _write(repo_root, manifest)


def merge(base: Mapping[str, Any], ours: Mapping[str, Any], theirs: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """The three-way merge of three ``VENDORED.json`` documents, package by package, and its conflicts.

    For every ``(op, package)``: the same entry on both sides is kept; an entry one side left as it was in ``base``
    takes the other side (an addition, a change or a removal); two different changes are a conflict, and so is a
    package the merge would give to two ops.
    """
    for name, document in (("base", base), ("ours", ours), ("theirs", theirs)):
        if document.get("schema") != SCHEMA or not isinstance(document.get("ops"), dict):
            raise ValueError(f"{name} is not a {SCHEMA} manifest")
    keys: set[tuple[str, str]] = set()
    for document in (base, ours, theirs):
        keys.update((op, package) for op, block in document["ops"].items() for package in block)
    merged: dict[str, dict[str, Any]] = {}
    conflicts = []
    for op, package in sorted(keys):
        old, mine, other = (document["ops"].get(op, {}).get(package) for document in (base, ours, theirs))
        if mine == other or other == old:
            entry = mine
        elif mine == old:
            entry = other
        else:
            conflicts.append(f"ops.{op}.{package}: changed differently on both sides")
            continue
        if entry is not None:
            merged.setdefault(op, {})[package] = entry
    owners: dict[str, list[str]] = {}
    for op, block in merged.items():
        for package in block:
            owners.setdefault(package, []).append(op)
    conflicts += [
        f"{package}: vendored for {' and '.join(ops)}" for package, ops in sorted(owners.items()) if len(ops) > 1
    ]
    return {"ops": merged, "schema": SCHEMA}, conflicts


def merge_files(base: Path, ours: Path, theirs: Path) -> list[str]:
    """Merge the ``VENDORED.json`` files ``base``, ``ours`` and ``theirs`` into ``ours`` (git's ``%O %A %B``).

    An empty ``base`` (no common ancestor) counts as an empty manifest. ``ours`` is left as it was on a conflict.
    """
    documents = []
    for path in (base, ours, theirs):
        text = path.read_text(encoding="utf-8")
        documents.append(json.loads(text) if text.strip() else {"ops": {}, "schema": SCHEMA})
    merged, conflicts = merge(*documents)
    if not conflicts:
        ours.write_text(canonical(merged), encoding="utf-8")
    return conflicts


# ---------------------------------------------------------------------------------------------------------------
# checking


def _tree_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not path.is_dir() and "__pycache__" not in path.relative_to(root).parts
    }


def functions(tree: ast.Module) -> dict[str, ast.AST]:
    """The functions a module defines, by qualified name: its own (``f``) and its classes' methods (``C.f``), also
    under a module-level ``if`` / ``try`` / ``with``."""
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


def python_findings(
    source: bytes, where: str, *, env_read: bool = False, flashinfer_hooks: Iterable[str] = ()
) -> list[str]:
    """``flashinfer`` imports and environment reads of one vendored module.

    ``env_read`` allows environment reads. ``flashinfer_hooks`` names the functions (``f`` or ``C.f``) inside which a
    flashinfer import is allowed: FlashInfer hooks the op's implementation rebinds before any call reaches them.
    Anywhere else, at module level included, it is a finding; so is a hook the module does not define.
    """
    try:
        tree = ast.parse(source, filename=where)
    except SyntaxError as exc:
        return [f"{where}: does not parse: {exc}"]
    defined = functions(tree)
    findings = [
        f"{where}: allow names the function {name}, which the module does not define"
        for name in sorted(set(flashinfer_hooks) - set(defined))
    ]
    in_hook = {id(node) for name in flashinfer_hooks if name in defined for node in ast.walk(defined[name])}
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
            if node.module == "os" and not env_read:
                findings.extend(
                    f"{where}:{node.lineno}: reads the environment (from os import {alias.name})"
                    for alias in node.names
                    if alias.name in _ENV_NAMES
                )
        if id(node) not in in_hook:
            findings.extend(
                f"{where}:{node.lineno}: imports {module}"
                for module in modules
                if module.split(".")[0] == "flashinfer"
            )
        if (
            not env_read
            and isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in _ENV_NAMES
        ):
            findings.append(f"{where}:{node.lineno}: reads the environment (os.{node.attr})")
    return findings


def _check_build(package_dir: Path, entry: Mapping[str, Any], where: str) -> list[str]:
    try:
        build = json.loads((package_dir / "build.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{where}/build.json: {exc}"]
    if build.get("schema") != BUILD_SCHEMA:
        return [f"{where}/build.json: schema {build.get('schema')!r}, expected {BUILD_SCHEMA!r}"]
    findings = []
    revision = build.get("generator", {}).get("cake_revision")
    if revision != entry.get("cake_revision"):
        findings.append(
            f"{where}: VENDORED.json cake_revision {entry.get('cake_revision')!r} != build.json {revision!r}"
        )
    digests: dict[str, str | None] = {}
    arches = set()
    for library in build.get("libraries", []):
        arches.add(library["arch"])
        for source in library["sources"]:
            if source not in digests:
                path = package_dir / source
                digests[source] = _sha256(path.read_bytes()) if path.is_file() else None
            if digests[source] is None:
                findings.append(f"{where}/{source}: missing, a source of {library['name']}")
            elif digests[source] != library["sha256"].get(source):
                findings.append(f"{where}/{source}: sha256 differs from build.json ({library['name']})")
    if missing := [arch for arch in REQUIRED_ARCHES if arch not in arches]:
        findings.append(f"{where}/build.json: no libraries for {', '.join(missing)}")
    return findings


def _check_source(op: str, package: str, entry: Mapping[str, Any], where: str) -> list[str]:
    source = entry.get("source")
    if not isinstance(source, dict):
        return [f"{where}: no source"]
    kind = source.get("kind")
    findings = []
    if not isinstance(source.get("repo"), str) or not source["repo"]:
        findings.append(f"{where}: source has no repo")
    if not _COMMIT.fullmatch(str(source.get("commit"))):
        findings.append(f"{where}: source commit {source.get('commit')!r} is not a 40-hex commit")
    if kind == "flashinfer":
        try:
            if fi_package_name(str(source.get("path"))) != package:
                findings.append(f"{where}: source path {source.get('path')} names another package")
        except ValueError as exc:
            findings.append(f"{where}: {exc}")
        if {"local", "build"} & set(entry):
            findings.append(f"{where}: a FlashInfer package has no local files and no build inputs")
        if set(source) != {"commit", "kind", "path", "repo"}:
            findings.append(f"{where}: source keys {sorted(source)}")
    elif kind == "miles-pr":
        if not re.fullmatch(rf"cake_{op}_[a-z0-9_]+", package):
            findings.append(f"{where}: a miles-pr package of {op} is named cake_{op}_<purpose>")
        if set(source) != {"commit", "exclude", "include", "kind", "path", "pr", "repo"}:
            findings.append(f"{where}: source keys {sorted(source)}")
        elif not isinstance(source["pr"], int) or not source["include"] or not isinstance(source["exclude"], dict):
            findings.append(f"{where}: source pr / include / exclude malformed")
        if entry.get("cake_revision") is not None:
            findings.append(f"{where}: a miles-pr package has no Cake revision")
        if set(entry.get("build") or {}) != {"arch_from", "link_flags", "nvcc_flags", "registry"}:
            findings.append(f"{where}: build inputs {sorted(entry.get('build') or {})}")
        if set(entry.get("local") or {}) != set(LOCAL_FILES):
            findings.append(f"{where}: local files {sorted(entry.get('local') or {})}, expected {sorted(LOCAL_FILES)}")
    else:
        findings.append(f"{where}: source kind {kind!r}")
    return findings


def _check_package(
    vendor_root: Path, op: str, package: str, entry: Mapping[str, Any], manifest_ops: Mapping[str, Any]
) -> list[str]:
    where = f"_vendor/{package}"
    findings = _check_source(op, package, entry, where)
    if findings:
        return findings
    kind = entry["source"]["kind"]
    package_dir = vendor_root / package
    files: Mapping[str, str] = entry.get("files") or {}
    local: Mapping[str, str] = entry.get("local") or {}
    if overlap := sorted(set(files) & set(local)):
        findings.append(f"{where}: {overlap} are both upstream and local")
    expected = {**files, **local}
    actual = _tree_files(package_dir)
    findings.extend(f"{where}/{path}: missing" for path in sorted(set(expected) - actual))
    findings.extend(f"{where}/{path}: not listed in {MANIFEST}" for path in sorted(actual - set(expected)))
    for relative, blob in sorted(expected.items()):
        if PurePosixPath(relative).parts[0] == "aot":
            findings.append(f"{where}/{relative}: prebuilt aot/ content is not vendored")
        path = package_dir / relative
        if path.is_file() and git_blob_sha1(path.read_bytes()) != blob:
            findings.append(f"{where}/{relative}: content differs from blob {blob}")
    findings.extend(f"{where}/{path}: aot/ directory" for path in sorted(package_dir.rglob("aot")) if path.is_dir())
    if kind == "flashinfer" and files.get("cake_build.py") != CAKE_BUILD_BLOB:
        findings.append(
            f"{where}/cake_build.py: blob {files.get('cake_build.py')} is not the loader {CAKE_BUILD_BLOB}"
        )
    if kind == "miles-pr":
        upstream = {path: (package_dir / path).read_bytes() for path in files if (package_dir / path).is_file()}
        try:
            generated = _local_files(package, upstream, entry["build"])
        except (KeyError, ValueError) as exc:
            findings.append(f"{where}: cannot generate its local files from the vendored ones: {exc!r}")
            generated = {}
        findings.extend(
            f"{where}/{relative}: not the file this tool generates for the package"
            for relative, content in sorted(generated.items())
            if local.get(relative) != git_blob_sha1(content)
        )
    if "build.json" not in expected:
        findings.append(f"{where}: no build.json")
    elif (package_dir / "build.json").is_file():
        findings.extend(_check_build(package_dir, entry, where))
    allow = entry.get("allow") or {}
    hooks: dict[str, list[str]] = {}
    for item in allow.get("flashinfer_import", ()):
        module, sep, name = str(item).partition(":")
        hooks.setdefault(module if sep and name else "", []).append(name)
    if (
        set(allow) - set(ALLOW_KEYS)
        or any(module not in files for module in [*allow.get("env_read", ()), *hooks])
        or any(allow[key] != sorted(set(allow[key])) for key in allow if isinstance(allow[key], list))
    ):
        findings.append(f"{where}: allow {allow} names unknown keys or modules (flashinfer_import: MODULE:FUNCTION)")
    package_of = {name: owner for owner, block in manifest_ops.items() for name in block}
    for relative in sorted(files):
        path = package_dir / relative
        if not relative.endswith(".py") or not path.is_file():
            continue
        source = path.read_bytes()
        try:
            imported = package_imports(source, depth=len(PurePosixPath(relative).parts) + 1)
        except SyntaxError:
            imported = set()  # reported below
        for name in sorted(imported - {package}):
            if name == "..":
                findings.append(f"{where}/{relative}: a relative import reaches above _vendor")
            elif package_of.get(name) != op:
                owner = package_of.get(name)
                findings.append(
                    f"{where}/{relative}: imports the vendored package {name}, "
                    + (f"which belongs to {owner}; an op imports its own packages only" if owner else "not vendored")
                )
        # cake_build.py is pinned by its blob; its lazy flashinfer.jit import is the hook the runtime's loader replaces
        if relative != "cake_build.py":
            findings.extend(
                python_findings(
                    source,
                    f"{where}/{relative}",
                    env_read=relative in allow.get("env_read", ()),
                    flashinfer_hooks=hooks.get(relative, ()),
                )
            )
    return findings


def check(repo_root: Path = REPO_ROOT, ops: Iterable[str] | None = None) -> list[str]:
    """Findings of the vendored tree against ``VENDORED.json`` (of every op, or of ``ops``); empty when intact."""
    vendor_root = repo_root / VENDOR
    try:
        text = (vendor_root / MANIFEST).read_text(encoding="utf-8")
        manifest = json.loads(text)
    except (OSError, ValueError) as exc:
        return [f"{MANIFEST}: {exc}"]
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or not isinstance(manifest.get("ops"), dict):
        return [f"{MANIFEST}: expected {{'schema': {SCHEMA!r}, 'ops': {{...}}}}"]
    findings = []
    if set(manifest) != {"ops", "schema"}:
        findings.append(f"{MANIFEST}: unexpected keys {sorted(set(manifest) - {'ops', 'schema'})}")
    if text != canonical(manifest):
        findings.append(f"{MANIFEST}: not in the form this tool writes; rewrite it with the tool")
    owners: dict[str, str] = {}
    for op, block in sorted(manifest["ops"].items()):
        if not _OP.fullmatch(op) or not isinstance(block, dict) or not block:
            findings.append(f"{MANIFEST}: ops.{op} is not an op with packages")
            continue
        for package in block:
            if package in owners:
                findings.append(f"{MANIFEST}: {package} is listed under {owners[package]} and {op}")
            owners.setdefault(package, op)
    expected = {"__init__.py", MANIFEST, *owners}
    actual = {path.name for path in vendor_root.iterdir() if path.name != "__pycache__"}
    findings.extend(f"_vendor/{name}: missing" for name in sorted(expected - actual))
    findings.extend(f"_vendor/{name}: not listed in {MANIFEST}" for name in sorted(actual - expected))
    init = vendor_root / "__init__.py"
    if init.is_file() and init.stat().st_size:
        findings.append("_vendor/__init__.py: must be empty")
    for op in sorted(manifest["ops"]) if ops is None else list(ops):
        if op not in manifest["ops"]:
            findings.append(f"{MANIFEST}: no op {op!r}")
            continue
        block = manifest["ops"][op]
        if not isinstance(block, dict):
            continue
        for package, entry in sorted(block.items()):
            findings.extend(_check_package(vendor_root, op, package, entry, manifest["ops"]))
        commits = {
            entry["source"]["commit"]
            for entry in block.values()
            if isinstance(entry.get("source"), dict) and entry["source"].get("kind") == "flashinfer"
        }
        if len(commits) > 1:
            findings.append(f"{MANIFEST}: the FlashInfer packages of {op} come from {len(commits)} commits")
    return findings


def check_upstream(
    repo_root: Path = REPO_ROOT, ops: Iterable[str] | None = None, cache: Path | None = None
) -> list[str]:
    """Findings of the recorded files against their upstream commits (needs network; fetches trees, not blobs)."""
    manifest = load_manifest(repo_root)
    selected = sorted(manifest["ops"]) if ops is None else [op for op in ops if op in manifest["ops"]]
    entries = [(op, package, entry) for op in selected for package, entry in sorted(manifest["ops"][op].items())]
    if not entries:
        return []
    with tempfile.TemporaryDirectory(prefix="vendor-cake-upstream-") as scratch:
        cache = Path(cache or scratch)
        findings = []
        fetched: dict[tuple[str, str], Path | str] = {}
        for op, package, entry in entries:
            source = entry["source"]
            where = f"_vendor/{package}"
            key = (source["repo"], source["commit"])
            if key not in fetched:
                fetched[key] = _fetch(cache, *key)
            git_dir = fetched[key]
            if isinstance(git_dir, str):
                findings.append(f"{where}: {git_dir}")
                continue
            try:
                tree = _tree(git_dir, source["commit"], source["path"])
                if source["kind"] == "miles-pr":
                    upstream, excluded = select(tree, source["include"], source["exclude"])
                    if excluded != source["exclude"]:
                        findings.append(f"{where}: excluded blobs {source['exclude']} != upstream {excluded}")
                else:
                    upstream = tree
            except (RuntimeError, ValueError) as exc:
                findings.append(f"{where}: {exc}")
                continue
            recorded = entry["files"]
            findings.extend(
                f"{where}/{path}: upstream has it, not vendored" for path in sorted(set(upstream) - set(recorded))
            )
            findings.extend(f"{where}/{path}: not upstream" for path in sorted(set(recorded) - set(upstream)))
            findings.extend(
                f"{where}/{path}: blob {recorded[path]} != upstream {upstream[path][1]}"
                for path in sorted(set(upstream) & set(recorded))
                if recorded[path] != upstream[path][1]
            )
        return findings


def _fetch(cache: Path, repo: str, commit: str) -> Path | str:
    """A bare repository under ``cache`` holding ``commit`` of ``repo`` (trees only), or the error."""
    git_dir = cache / hashlib.sha1(repo.encode()).hexdigest()[:16]
    try:
        if not git_dir.is_dir():
            git_dir.mkdir(parents=True)
            _git(git_dir, "init", "-q", "--bare")
            _git(git_dir, "remote", "add", "origin", repo)
            # a promisor remote may be fetched without blobs
            _git(git_dir, "config", "remote.origin.promisor", "true")
            _git(git_dir, "config", "remote.origin.partialclonefilter", "blob:none")
        # also when the cache has it: a presence check would fetch a missing commit lazily, with its history
        _git(git_dir, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    except (OSError, RuntimeError) as exc:
        return f"cannot fetch {commit} from {repo}: {exc}"
    return git_dir


# ---------------------------------------------------------------------------------------------------------------
# command line


def _arch_from(items: Iterable[str]) -> dict[str, str]:
    pairs = {}
    for item in items:
        target, sep, source = item.partition("=")
        if not sep:
            raise ValueError(f"--arch-from {item}: expected DST=SRC")
        pairs[target] = source
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vendor sources into miles_ext_kernels/cake/_vendor/ and check them.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="this repository's checkout")
    parser.add_argument("--check", nargs="*", metavar="OP", help="check the vendored tree (every op, or OP ...)")
    parser.add_argument("--upstream", action="store_true", help="with --check: compare with the upstream commits")
    parser.add_argument("--upstream-cache", type=Path, help="a directory to keep the fetched upstream trees in")
    commands = parser.add_subparsers(dest="command")

    fi = commands.add_parser("fi", help="vendor FlashInfer Cake packages")
    fi.add_argument("--checkout", type=Path, required=True, help="a FlashInfer git checkout holding --commit")
    fi.add_argument("--commit", required=True)
    fi.add_argument("--op", required=True)
    fi.add_argument("--package", action="append", required=True, metavar="FI_PATH", help="e.g. flashinfer/<name>")
    fi.add_argument("--repo", default=FI_REPO, help=f"recorded as the source repository (default {FI_REPO})")
    fi.add_argument("--allow-env-read", action="append", default=[], metavar="PKG:MODULE")
    fi.add_argument("--allow-flashinfer-import", action="append", default=[], metavar="PKG:MODULE:FUNCTION")

    pr = commands.add_parser("miles-pr", help="vendor the generated sources of a miles pull request")
    pr.add_argument("--checkout", type=Path, required=True, help="a miles git checkout holding --commit")
    pr.add_argument("--pr", type=int, required=True)
    pr.add_argument("--commit", required=True)
    pr.add_argument("--op", required=True)
    pr.add_argument("--name", required=True, help="the package name, cake_<op>_<purpose>")
    pr.add_argument("--path", required=True, help="the directory of the sources in the miles repository")
    pr.add_argument(
        "--include", action="append", required=True, metavar="REL", help="a file or directory under --path"
    )
    pr.add_argument("--exclude", action="append", default=[], metavar="REL", help="an included file not to vendor")
    pr.add_argument("--arch-from", action="append", default=[], metavar="DST=SRC", help="build DST from SRC's records")
    pr.add_argument("--nvcc-flag", action="append", metavar="FLAG", help=f"default {' '.join(DEFAULT_NVCC_FLAGS)}")
    pr.add_argument("--link-flag", action="append", metavar="FLAG", help=f"default {' '.join(DEFAULT_LINK_FLAGS)}")
    pr.add_argument("--repo", default=MILES_REPO, help=f"recorded as the source repository (default {MILES_REPO})")

    rm = commands.add_parser("remove", help="delete a vendored package")
    rm.add_argument("--op", required=True)
    rm.add_argument("--name", required=True)

    mg = commands.add_parser("merge", help=f"the git merge driver of {MANIFEST}: merge BASE and THEIRS into OURS")
    mg.add_argument("base", type=Path)
    mg.add_argument("ours", type=Path)
    mg.add_argument("theirs", type=Path)

    args = parser.parse_args(argv)
    if (args.command is None) == (args.check is None):
        parser.error("give one command, or --check")
    if (args.upstream or args.upstream_cache) and args.check is None:
        parser.error("--upstream goes with --check")

    if args.command == "merge":
        try:
            conflicts = merge_files(args.base, args.ours, args.theirs)
        except (OSError, ValueError) as exc:
            print(f"error: {MANIFEST} merge: {exc}", file=sys.stderr)
            return 2
        for conflict in conflicts:
            print(f"{MANIFEST} merge conflict: {conflict}", file=sys.stderr)
        return 1 if conflicts else 0

    try:
        if args.command == "fi":
            entries = vendor_fi(
                args.checkout,
                args.commit,
                args.op,
                args.package,
                repo_root=args.repo_root,
                repo=args.repo,
                allow_env_read=args.allow_env_read,
                allow_flashinfer_import=args.allow_flashinfer_import,
            )
            for package, entry in entries.items():
                source = entry["source"]
                print(f"{args.op}: {package} <- {source['path']} @ {source['commit']} ({len(entry['files'])} files)")
        elif args.command == "miles-pr":
            entry = vendor_miles_pr(
                args.checkout,
                args.commit,
                pr=args.pr,
                op=args.op,
                name=args.name,
                path=args.path,
                include=args.include,
                exclude=args.exclude,
                arch_from=_arch_from(args.arch_from),
                nvcc_flags=args.nvcc_flag or DEFAULT_NVCC_FLAGS,
                link_flags=args.link_flag or DEFAULT_LINK_FLAGS,
                repo_root=args.repo_root,
                repo=args.repo,
            )
            source = entry["source"]
            print(
                f"{args.op}: {args.name} <- #{args.pr} {source['path']} @ {source['commit']} ({len(entry['files'])} files)"
            )
        elif args.command == "remove":
            remove(args.op, args.name, repo_root=args.repo_root)
            print(f"{args.op}: removed {args.name}")
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ops = [args.op] if args.command in ("fi", "miles-pr") else (args.check or None)
    findings = check(args.repo_root, ops)
    if args.upstream:
        findings += check_upstream(args.repo_root, ops, args.upstream_cache)
    scope = "every op" if ops is None else ", ".join(ops)
    if findings:
        for finding in findings:
            print(f"  {finding}")
        print(f"vendored ({scope}): {len(findings)} findings")
        return 1
    manifest = load_manifest(args.repo_root)
    packages = sum(len(block) for op, block in manifest["ops"].items() if ops is None or op in ops)
    print(f"vendored ({scope}): ok, {packages} packages{', upstream checked' if args.upstream else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
