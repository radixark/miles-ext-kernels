"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The loader of the vendored Cake packages (``miles_ext_kernels.cake._vendor.<package>``): plain nvcc and tvm-ffi, no
FlashInfer.

Every vendored package keeps FlashInfer's byte-identical ``cake_build.py`` and its ``build.json`` manifest (schema
``cake-build/1``). Without ``aot/<name>.so``, ``cake_build.load_library(name)`` returns
``jit_spec(name).build_and_load()``, and ``jit_spec``, ``jit_specs`` and ``record_cuda_version`` look up the module
attribute ``_jit_toolchain`` at call time. :func:`install` rebinds it to this module's
``(gen_jit_spec, get_cuda_version)``, so the package compiles its manifest entries with nvcc and loads them with
tvm-ffi.

The command is the one of FlashInfer's vendored-build gate (``compile_command`` of
``scripts/check_cake_build_manifests.py``): nvcc, ``common.nvcc_flags``, the ``common.nvcc_flags_if`` items matching
the toolkit, the flags ``cake_build.jit_spec`` passes (``arch_flags[arch]``, the entry's ``nvcc_flags`` and its
matching ``nvcc_flags_if``), ``-I`` for ``common.include_dirs`` and for the tvm-ffi and dlpack headers, the sources,
``common.link_flags`` and ``-o``. ``cake_build.jit_spec`` leaves out the ``common`` flags because FlashInfer's JIT
adds its own defaults; this loader adds them.

A library is cached as ``${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}/miles_ext_kernels/<package>/<name>_<key16>.so``
(``<name>`` is ``cake_build``'s ``<entry>_<closure20>``), never in the package directory, which may be read-only. The
key hashes the flags, the full nvcc version, the tvm-ffi version and the machine, so builds of different toolkits or
CPU architectures in one shared checkout do not collide. All ranks may start a build at once: one compiles under a
file lock into a temporary file and renames it into place, the others wait and load it.

A waiting rank logs every 30 s and fails after 30 minutes. On a filesystem whose locks are cluster-wide (Lustre), the
lock of a holder killed on another node can stay held for minutes after it died.

:data:`ARCH_BY_CAPABILITY` names the architecture of every compute capability the packages serve, and :data:`MIN_CUDA`
the oldest toolkit whose nvcc compiles it: SM107a needs CUDA 13.4 (nvcc 13.0 to 13.3 reject ``compute_107a``), so a
CUDA 13.0 toolkit builds the SM100a and SM103a libraries only. Every library build checks it, whatever ``min_cuda``
its manifest entry records, and raises naming the toolkit the architecture needs; :func:`check_toolkit` runs the same
check ahead of a build.

An op's ``_impl`` package calls :func:`install_op` once at import: it installs the loader on the ``cake_build`` of
every package the op declares (``OpSpec.packages``). :func:`load_package` does the same for one package and returns it,
and :func:`prebuild_libraries` compiles a package's libraries ahead of their first call. Inside :func:`no_build`, a
library that is not in the cache raises instead of compiling, which is how the tests check an op's ``prebuild``
contract. :func:`device_arch` names the architecture of a CUDA device.

FlashInfer keeps some Cake packages in ``flashinfer/experimental/cake/``, and its code imports them as
``..cake.<name>``. Every vendored package sits one level deep, at ``_vendor/<name>/``, so importing this module adds a
finder to ``sys.meta_path`` that resolves ``_vendor.cake`` to an empty package and ``_vendor.cake.<name>`` (and its
submodules) to the module ``_vendor.<name>`` itself: each package is loaded once, through whatever finder serves
``_vendor.<name>`` (a per-op pin included), and nothing in the vendored tree lists the names.

Importing this module does not import torch or tvm_ffi.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

__all__ = [
    "ARCH_BY_CAPABILITY",
    "MIN_CUDA",
    "CudaVersion",
    "JitSpec",
    "cache_dir",
    "check_toolkit",
    "device_arch",
    "find_nvcc",
    "get_cuda_version",
    "install",
    "install_op",
    "load_package",
    "no_build",
    "prebuild_libraries",
]

# The architecture of every compute capability the vendored packages serve.
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
# The oldest CUDA toolkit whose nvcc compiles each architecture.
MIN_CUDA = {"sm_100a": (12, 8), "sm_103a": (12, 9), "sm_107a": (13, 4)}

_CUDA_CONSTRAINT = re.compile(r"(==|>=)([0-9]+)\.([0-9]+)")
_NVCC_RELEASE = re.compile(r"release (\d+)\.(\d+)")
_NVCC_BUILD = re.compile(r"\bV(\d+\.\d+\.\d+)")
_STDERR_LINES = 40  # kept in the error of a failed compile
# The wait for another process's build lock (held for one nvcc call): tried every _LOCK_POLL_SECONDS, logged every
# _LOCK_LOG_SECONDS, given up after _LOCK_TIMEOUT_SECONDS.
_LOCK_POLL_SECONDS = 1.0
_LOCK_LOG_SECONDS = 30.0
_LOCK_TIMEOUT_SECONDS = 30 * 60

logger = logging.getLogger(__name__)

# miles_ext_kernels.cake._vendor: the vendored packages are its subpackages
_VENDOR = f"{__name__.rsplit('.', 2)[0]}._vendor"
# the name FlashInfer's ``..cake.<name>`` imports reach from a vendored package
_ALIAS = f"{_VENDOR}.cake"
# package -> the op whose install_op installed it, named by the error of a build inside no_build()
_OP_OF_PACKAGE: dict[str, str] = {}
# depth of the no_build() blocks in this process
_NO_BUILD = 0


class CudaVersion(NamedTuple):
    """The toolkit of ``nvcc --version``: ``release X.Y`` and the full ``VX.Y.Z``."""

    major: int
    minor: int
    full: str

    @property
    def cudart(self) -> int:
        """``CUDART_VERSION`` of the toolkit: ``1000 * major + 10 * minor``."""
        return 1000 * self.major + 10 * self.minor


def find_nvcc() -> str:
    """``$CUDA_HOME/bin/nvcc``, else ``nvcc`` on ``PATH``."""
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        return str(Path(cuda_home) / "bin" / "nvcc")
    found = shutil.which("nvcc")
    if found is None:
        raise RuntimeError("no nvcc to build the Cake kernels: set CUDA_HOME or put nvcc on PATH")
    return found


@functools.lru_cache(maxsize=None)
def _nvcc_version(nvcc: str, stamp: tuple[int, int]) -> CudaVersion:
    # ``stamp`` (mtime, size) re-runs nvcc when the file at that path changes
    try:
        proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"cannot run {nvcc} --version: {exc}") from exc
    release = _NVCC_RELEASE.search(proc.stdout)
    build = _NVCC_BUILD.search(proc.stdout)
    if proc.returncode or release is None or build is None:
        raise RuntimeError(
            f"{nvcc} --version (exit {proc.returncode}) reports no 'release X.Y, VX.Y.Z':\n{proc.stdout}{proc.stderr}"
        )
    return CudaVersion(int(release.group(1)), int(release.group(2)), build.group(1))


def _toolkit(nvcc: str | None = None) -> tuple[str, CudaVersion]:
    nvcc = nvcc or find_nvcc()
    try:
        stat = os.stat(nvcc)
    except OSError as exc:
        raise RuntimeError(f"cannot run {nvcc}: {exc}") from exc
    return nvcc, _nvcc_version(nvcc, (stat.st_mtime_ns, stat.st_size))


def get_cuda_version() -> CudaVersion:
    """The toolkit of :func:`find_nvcc` (``.major``, ``.minor`` as ``cake_build`` reads them, and ``.full``)."""
    return _toolkit()[1]


def _check_arch(arch: str, what: str, nvcc: str, cuda: CudaVersion) -> None:
    need = MIN_CUDA.get(arch)
    if need is not None and (cuda.major, cuda.minor) < need:
        raise RuntimeError(
            f"compiling {what} for {arch} needs CUDA {need[0]}.{need[1]} or newer, but {nvcc} is CUDA {cuda.full}: "
            f"set CUDA_HOME to a CUDA {need[0]}.{need[1]} or newer toolkit"
        )


def check_toolkit(arch: str, what: str, nvcc: str | None = None) -> CudaVersion:
    """The toolkit of ``nvcc`` (default: :func:`find_nvcc`), checked to compile ``arch`` (:data:`MIN_CUDA`).

    ``RuntimeError`` naming the CUDA version ``arch`` needs when the toolkit is older; ``what`` names the kernels in
    the message.
    """
    nvcc, cuda = _toolkit(nvcc)
    _check_arch(arch, what, nvcc, cuda)
    return cuda


def _conditional_flags(items: Iterable[dict[str, Any]], cuda: tuple[int, int]) -> list[str]:
    """The flags of every ``nvcc_flags_if`` item whose ``==X.Y`` / ``>=X.Y`` constraint the toolkit meets."""
    flags: list[str] = []
    for item in items:
        match = _CUDA_CONSTRAINT.fullmatch(str(item["cuda"]))
        if match is None:
            raise ValueError(f"unsupported CUDA constraint {item['cuda']!r}; expected '==X.Y' or '>=X.Y'")
        operator, major, minor = match.groups()
        version = (int(major), int(minor))
        if (cuda == version) if operator == "==" else (cuda >= version):
            flags.extend(item["flags"])
    return flags


def cache_dir() -> Path:
    """``$TVM_FFI_CACHE_DIR`` (tvm-ffi's own cache variable), else ``~/.cache/tvm-ffi``, + ``miles_ext_kernels``."""
    root = os.environ.get("TVM_FFI_CACHE_DIR", str(Path("~/.cache/tvm-ffi").expanduser()))
    return Path(root).expanduser() / "miles_ext_kernels"


def _lock_exclusive(lock: Any, lock_path: Path, name: str) -> None:
    """Take the exclusive flock of the open file ``lock``, waiting at most ``_LOCK_TIMEOUT_SECONDS``."""
    start = time.monotonic()
    logged = start
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            pass
        now = time.monotonic()
        if now - start >= _LOCK_TIMEOUT_SECONDS:
            raise RuntimeError(
                f"gave up building {name} after waiting {now - start:.0f} s for the build lock {lock_path}. "
                "Another process holds it: a build of the library, or a process killed on another node whose lock "
                "the filesystem has not released yet. Retry, or set TVM_FFI_CACHE_DIR to a node-local or per-job "
                "directory."
            )
        if now - logged >= _LOCK_LOG_SECONDS:
            logged = now
            logger.warning(
                "waiting %.0f s for the build lock %s of %s, held by another process", now - start, lock_path, name
            )
        time.sleep(_LOCK_POLL_SECONDS)


class JitSpec:
    """One manifest library of a vendored package: its nvcc build into the cache and its tvm-ffi load.

    The interface ``cake_build`` uses of FlashInfer's JIT spec: :meth:`build`, :meth:`get_library_path` and
    :meth:`build_and_load`.
    """

    def __init__(
        self,
        package: str,
        name: str,
        *,
        nvcc: str,
        cuda: CudaVersion,
        flags: list[str],
        include_dirs: list[str],
        sources: list[str],
        ldflags: list[str],
    ) -> None:
        import tvm_ffi.libinfo

        self.package = package
        self.name = name
        self.nvcc = nvcc
        self.cuda = cuda
        self.flags = flags
        self.include_dirs = [
            *include_dirs,
            tvm_ffi.libinfo.find_include_path(),
            tvm_ffi.libinfo.find_dlpack_include_path(),
        ]
        self.sources = sources
        self.ldflags = ldflags
        # Paths stay out of the key: the name carries the closure digest of the sources, and the tvm-ffi version
        # pins its headers, so checkouts in different directories share one entry.
        key = json.dumps(
            {
                "flags": [*flags, *ldflags],
                "nvcc": cuda.full,
                "tvm_ffi": tvm_ffi.__version__,
                "machine": platform.machine(),
            },
            sort_keys=True,
        )
        self.key = hashlib.sha256(key.encode()).hexdigest()
        self._path = cache_dir() / package / f"{name}_{self.key[:16]}.so"

    def command(self, output: Path) -> list[str]:
        """The nvcc command writing the library to ``output``."""
        return [
            self.nvcc,
            *self.flags,
            *(f"-I{directory}" for directory in self.include_dirs),
            *self.sources,
            *self.ldflags,
            "-o",
            str(output),
        ]

    def get_library_path(self) -> Path:
        """Where the library is cached (built or not)."""
        return self._path

    def build(self) -> Path:
        """Compile the library unless it is cached; return its path."""
        path = self._path
        if path.is_file():
            return path
        if _NO_BUILD:
            op = _OP_OF_PACKAGE.get(self.package, self.package)
            raise RuntimeError(f"{self.name} would compile inside a call: prebuild({op!r}, ...) first")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        with open(lock_path, "a") as lock:
            _lock_exclusive(lock, lock_path, self.name)  # released when the file closes
            if path.is_file():  # another rank built it while this one waited
                return path
            # unique across hosts: a lock that is local to each node still lets two nodes build at once
            temporary = path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.so")
            command = self.command(temporary)
            try:
                try:
                    proc = subprocess.run(command, capture_output=True, text=True, check=False)
                except OSError as exc:
                    raise RuntimeError(f"cannot run {self.nvcc} to build {self.name}: {exc}") from exc
                if proc.returncode or not temporary.is_file():
                    stderr = "\n".join(proc.stderr.splitlines()[:_STDERR_LINES])
                    raise RuntimeError(
                        f"nvcc failed to build {self.name} (exit {proc.returncode}):\n{' '.join(command)}\n{stderr}"
                    )
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def build_and_load(self) -> Any:
        """The loaded tvm-ffi module, checked to be compiled against the toolkit's CUDART."""
        path = self.build()
        import tvm_ffi

        module = tvm_ffi.load_module(str(path))
        built = int(module.build_cuda_version())
        if built != self.cuda.cudart:
            raise RuntimeError(
                f"{path} was compiled against CUDART {built}, but {self.nvcc} is CUDA {self.cuda.full} "
                f"(CUDART {self.cuda.cudart})"
            )
        return module


def install(cake_build_module: ModuleType) -> ModuleType:
    """Make the vendored ``cake_build`` module build its libraries through this loader; idempotent."""
    if not callable(getattr(cake_build_module, "_jit_toolchain", None)):
        raise TypeError(f"{cake_build_module.__name__} has no _jit_toolchain hook: not a cake-build/1 loader")
    package = Path(cake_build_module.__file__).parent.name

    def gen_jit_spec(
        name: str,
        sources: list[Path],
        *,
        arch: str,
        extra_cuda_cflags: list[str],
        extra_ldflags: list[str],
        extra_include_paths: list[Path],
        use_fast_math: bool = False,
    ) -> JitSpec:
        if use_fast_math:
            raise ValueError(f"{name}: use_fast_math must come from the manifest's flags")
        nvcc, cuda = _toolkit()
        _check_arch(arch, name, nvcc, cuda)
        common = cake_build_module.manifest()["common"]
        return JitSpec(
            package,
            name,
            nvcc=nvcc,
            cuda=cuda,
            flags=[
                *common["nvcc_flags"],
                *_conditional_flags(common["nvcc_flags_if"], (cuda.major, cuda.minor)),
                *extra_cuda_cflags,
            ],
            include_dirs=[str(path) for path in extra_include_paths],
            sources=[str(path) for path in sources],
            ldflags=list(extra_ldflags),
        )

    def _jit_toolchain(name: str) -> tuple[Any, Any]:
        # the entry's architecture, checked against MIN_CUDA when the library is built
        arch = cake_build_module.libraries_by_name()[name]["arch"]
        return functools.partial(gen_jit_spec, arch=arch), get_cuda_version

    cake_build_module._jit_toolchain = _jit_toolchain
    return cake_build_module


def load_package(name: str) -> ModuleType:
    """Import the vendored package ``name`` (``miles_ext_kernels.cake._vendor.<name>``) with this loader installed."""
    package = import_module(f"{_VENDOR}.{name}")
    install(import_module(f"{package.__name__}.cake_build"))
    return package


def install_op(op: str) -> None:
    """Install this loader on every package ``op`` declares (``OpSpec.packages``); idempotent.

    Called once by ``miles_ext_kernels.cake._impl.<op>`` at import, before any of its packages can build: a package
    that imports a sibling lazily (a context-parallel package and its single-rank sibling) finds it installed too.
    """
    from .registry import op_spec

    for name in op_spec(op).packages:
        _OP_OF_PACKAGE[name] = op
        load_package(name)


def prebuild_libraries(package: str | ModuleType, names: Iterable[str], max_workers: int | None = None) -> list[Path]:
    """Compile the libraries ``names`` of a vendored package ahead of their first call, ``max_workers`` at a time.

    ``package`` is a vendored package name or its ``cake_build`` module; this loader is installed on it. A library
    prebuilt under ``aot/`` is used as it is, a cached one is not compiled again; a library whose ``min_cuda`` or whose
    architecture's :data:`MIN_CUDA` exceeds the toolkit raises, as loading it would. Returns the path of every library
    of ``names``, in order.
    """
    if isinstance(package, str):
        cake_build_module = import_module(f"{load_package(package).__name__}.cake_build")
    else:
        cake_build_module = install(package)
    names = list(dict.fromkeys(names))
    paths = {name: cake_build_module.prebuilt_path(name) for name in names}
    pending = [name for name in names if not paths[name].is_file()]
    if pending:
        workers = max_workers or min(8, len(os.sched_getaffinity(0)))
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pending)))) as pool:
            built = pool.map(lambda name: cake_build_module.jit_spec(name).build(), pending)
            paths.update(zip(pending, built))
    return [paths[name] for name in names]


@contextmanager
def no_build() -> Iterator[None]:
    """Inside this block, a library that is not cached raises ``RuntimeError`` instead of compiling (nestable)."""
    global _NO_BUILD
    _NO_BUILD += 1
    try:
        yield
    finally:
        _NO_BUILD -= 1


class _VendorAlias(MetaPathFinder, Loader):
    """``_vendor.cake`` is an empty package; ``_vendor.cake.<name>[.<module>]`` is ``_vendor.<name>[.<module>]``.

    First on ``sys.meta_path``, so that a submodule reached through the alias is not found again under the alias name
    (it would load twice); every other name is left to the other finders.
    """

    def find_spec(self, fullname: str, path: Sequence[str] | None = None, target: Any = None) -> ModuleSpec | None:
        if fullname == _ALIAS:
            return ModuleSpec(fullname, self, is_package=True)
        if fullname.startswith(f"{_ALIAS}."):
            return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec: ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        if module.__name__ != _ALIAS:
            # the import system returns what sys.modules holds under the name once exec_module returns
            sys.modules[module.__name__] = import_module(f"{_VENDOR}.{module.__name__[len(_ALIAS) + 1 :]}")


if not any(isinstance(finder, _VendorAlias) for finder in sys.meta_path):
    sys.meta_path.insert(0, _VendorAlias())


def device_arch(device: Any = None) -> str:
    """``"sm_100a"``, ``"sm_103a"`` or ``"sm_107a"``: the architecture of the CUDA ``device`` (default: current).

    ``NotImplementedError`` for any other compute capability. Imports torch.
    """
    import torch

    capability = tuple(torch.cuda.get_device_capability(device))
    try:
        return ARCH_BY_CAPABILITY[capability]
    except KeyError:
        major, minor = capability
        raise NotImplementedError(
            f"the Cake kernels need SM100a, SM103a or SM107a, not compute capability {major}.{minor}"
        ) from None
