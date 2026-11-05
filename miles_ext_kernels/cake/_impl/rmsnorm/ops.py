"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Zero-centred RMSNorm on the vendored Cake RMSNorm kernels (SM100a / SM103a / SM107a).

:class:`ZeroCenteredRMSNorm` is the input layernorm of miles' GDN layers, ``y = bf16(x * rstd * (1 + w))`` over the
last dimension with ``rstd = 1 / sqrt(mean(x^2) + eps)`` in FP32. It holds ``w`` as its single ``weight`` parameter,
as Transformer Engine's ``zero_centered_gamma`` norm and HF's ``Qwen3NextRMSNorm`` do, so checkpoints and weight
conversion see the same parameter, and hands the kernels the FP32 effective weight ``1 + w`` (their ``_w32`` routes).
The kernels return the weight gradient in FP32; autograd rounds it into the parameter's dtype, as TE's dgamma. Like
TE's norm it accepts ``sequence_parallel``, which, when given, it sets on the weight (Megatron all-reduces the gradient
of such a parameter across the tensor-parallel ranks), builds on the meta device and has ``reset_parameters``.

Both variants go through FlashInfer's non-deterministic package (``_vendor/cake_rmsnorm_train_backend``):
``deterministic=True`` runs its deterministic sibling (``_vendor/cake_rmsnorm_det_train_backend``), whose backward is,
per token-count window, a main kernel and a fixed-order reduce (``dw`` bitwise reproducible for a given token count
and SM count) or, at the smallest token counts of every hidden size but 256, one kernel that sums each ``dw`` column
in a fixed row order (``dw`` bitwise reproducible for a given token count on any SM count); ``deterministic=False``
runs the same forward programs and, per token-count window, a single-kernel backward whose CTAs add their ``dw`` sums
with FP64 atomics (``dw`` not bitwise reproducible; ``dx`` reproducible, within one BF16 ulp of ``det``'s), a
column-task kernel at the smallest token counts, or ``det``'s kernel pair. A forward reads the configured variant
(``--cake-rmsnorm-variant`` through ``miles_ext_kernels.configure``; ``det`` when unconfigured) when it runs, and its
backward runs the same one. ``fast`` serves every hidden size ``det`` serves; it runs ``det``, logged once, while
``torch.use_deterministic_algorithms(True)`` is in effect, which the non-deterministic package refuses.

Only hidden sizes with FP32-weight routes in both packages run (:func:`supported_hidden_sizes`); the module raises at
construction for any other. The kernels pick a program per token-count window: :func:`prebuild` builds every program
of both variants for one hidden size, so that no call compiles; constructing the module builds nothing. SM107a needs a
CUDA 13.4 or newer toolkit to build its kernels.

On SM107a, hidden size 4096 also launches programs with more shared memory per block than the standard opt-in
(232448 B): ``det``'s backward above 4096 tokens and ``fast``'s from 4097 to 24064 tokens. They run in CUDA 13.4's
oversized shared-memory mode, which needs a CUDA runtime of 13.4 or newer in the process (an older one, such as the
CUDA 13.0 runtime wheel of a ``cu130`` torch, rejects the mode at the launch) and a driver whose oversized ceiling
holds them. Where the process cannot launch them, the module's construction on that device, :func:`check_supported`,
:func:`prebuild` and the first call of that width on that device raise ``RuntimeError`` naming the programs and the
runtime, before any launch.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader

__all__ = [
    "ZeroCenteredRMSNorm",
    "check_supported",
    "prebuild",
    "release",
    "supported_hidden_sizes",
    "zero_centered_rmsnorm",
]

_OP = "rmsnorm"
_SELECTOR = "--linear-attention-input-norm cake"
_KERNELS = "the Cake RMSNorm kernels"
# the packages of the deterministic and the non-deterministic kernels, in this order
_PACKAGES = ("cake_rmsnorm_det_train_backend", "cake_rmsnorm_train_backend")
# CUDA 13.4's oversized shared-memory mode: the binding of a program above the standard opt-in sets it on the kernel
# (cudaFuncAttributeSharedMemoryMode) before the launch. A runtime older than 13.4 rejects the attribute
# ("cudaFuncSetAttribute failed: invalid argument"); a driver without the mode reports no oversized ceiling.
_OVERSIZED_MODE = "cudaSharedMemoryModeAllowOversizedSharedMemory"
_DYNAMIC_SMEM = re.compile(r"\bdynamicSmemBytes = (\d+);")
_OVERSIZED_RUNTIME = 13040  # cudaRuntimeGetVersion of CUDA 13.4
_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
_CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK = 150


@functools.cache
def _packages() -> tuple[ModuleType, ModuleType]:
    """``(det, fast)``: the vendored packages, building their libraries through the runtime's loader. ``fast`` imports
    ``det`` as its sibling for ``deterministic=True``."""
    det, fast = map(loader.load_package, _PACKAGES)
    return det, fast


def supported_hidden_sizes(arch: str) -> tuple[int, ...]:
    """The hidden sizes whose FP32-weight forward and backward both packages ship for ``arch``."""
    det, fast = _packages()
    return tuple(
        hidden
        for hidden in det.cake_jit.HIDDEN_SIZES
        if all(
            "f32" in package.cake_jit.weight_dtypes(arch, hidden, kind, False)
            for package in (det, fast)
            for kind in ("fwd", "bwd")
        )
    )


def _check_hidden_size(hidden_size: int) -> None:
    """``ValueError`` unless some architecture serves ``hidden_size``."""
    sizes = {size for arch in loader.ARCH_BY_CAPABILITY.values() for size in supported_hidden_sizes(arch)}
    if hidden_size not in sizes:
        raise ValueError(
            f"{_SELECTOR} supports hidden sizes {sorted(sizes)}, not {hidden_size}; "
            "use --linear-attention-input-norm te"
        )


@functools.cache
def _oversized_programs(arch: str, hidden_size: int) -> tuple[tuple[str, int], ...]:
    """``(library, dynamic shared memory bytes)`` of every program of both packages' FP32-weight routes of
    ``hidden_size`` on ``arch`` that launches in the oversized shared-memory mode (its binding sets the mode), sorted.
    """
    found = set()
    for package in _packages():
        jit = package.cake_jit
        csrc = Path(package.__file__).parent / "csrc"
        for kind in ("fwd", "bwd"):
            for module in jit.route_modules(arch, hidden_size, kind, False, "f32"):
                binding = (csrc / f"{jit.MODULES[module]['library']}_binding.cu").read_text()
                if _OVERSIZED_MODE in binding:
                    found.add((jit.library_name(module, arch), int(_DYNAMIC_SMEM.search(binding)[1])))
    return tuple(sorted(found))


def _shared_library(soname: str) -> ctypes.CDLL | None:
    """The process's copy of ``soname``: the one already loaded (the copy a kernel library linking ``soname`` binds
    to), else the one the dynamic linker finds, else ``None``."""
    for mode in (os.RTLD_NOLOAD | os.RTLD_NOW, ctypes.DEFAULT_MODE):
        try:
            return ctypes.CDLL(soname, mode=mode)
        except OSError:
            continue
    return None


def _cuda_runtime_version(major: int) -> int | None:
    """``cudaRuntimeGetVersion`` of ``libcudart.so.<major>``, the runtime the kernel libraries link (``-lcudart``,
    resolved at load time), or ``None`` when it does not load or the call fails."""
    cudart = _shared_library(f"libcudart.so.{major}")
    version = ctypes.c_int(0)
    if cudart is None or cudart.cudaRuntimeGetVersion(ctypes.byref(version)) != 0:
        return None
    return version.value


def _device_attribute(index: int, attribute: int) -> int | None:
    """The driver's ``CUdevice_attribute`` ``attribute`` of CUDA device ``index``, or ``None`` when the driver does
    not report it (a driver older than 13.4 has no oversized shared-memory ceiling)."""
    cuda = _shared_library("libcuda.so.1")
    handle, value = ctypes.c_int(0), ctypes.c_int(0)
    if (
        cuda is None
        or cuda.cuInit(0) != 0
        or cuda.cuDeviceGet(ctypes.byref(handle), index) != 0
        or cuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, handle) != 0
    ):
        return None
    return value.value


def _runtime_name(version: int) -> str:
    return f"{version // 1000}.{version % 1000 // 10}"


def _check_shared_memory(arch: str, hidden_size: int, index: int, cuda_major: int) -> None:
    """``RuntimeError`` unless this process can launch every program of width ``hidden_size`` on CUDA device
    ``index`` of ``arch``: one in the oversized shared-memory mode needs a CUDA runtime (``libcudart.so.<cuda_major>``,
    the toolkit's major) of 13.4 or newer and a driver whose oversized ceiling holds its shared memory."""
    programs = _oversized_programs(arch, hidden_size)
    if not programs:
        return
    need = max(smem for _, smem in programs)
    runtime = _cuda_runtime_version(cuda_major)
    ceiling = _device_attribute(index, _CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK)
    problems = []
    if runtime is None:
        problems.append(f"no CUDA runtime libcudart.so.{cuda_major} loads")
    elif runtime < _OVERSIZED_RUNTIME:
        problems.append(f"the process's CUDA runtime (libcudart.so.{cuda_major}) is {_runtime_name(runtime)}")
    if ceiling is None:
        problems.append("the driver reports no oversized shared-memory ceiling")
    elif ceiling < need:
        problems.append(f"the driver's oversized shared-memory ceiling is {ceiling} B")
    if problems:
        optin = _device_attribute(index, _CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        raise RuntimeError(
            f"{_SELECTOR}: hidden size {hidden_size} on {arch} launches {', '.join(name for name, _ in programs)} "
            f"with {need} B of shared memory per block, above the {optin} B opt-in of CUDA device {index}; they need "
            f"CUDA 13.4's oversized shared-memory mode, which this process cannot use: {'; '.join(problems)}. "
            "Load a CUDA 13.4 or newer runtime (pip: nvidia-cuda-runtime>=13.4) or use "
            "--linear-attention-input-norm te"
        )


def _check_device(hidden_size: int, device: Any) -> str:
    """The architecture of the CUDA ``device``, checked to serve ``hidden_size`` and its kernels to build with the
    toolkit and launch in this process: ``NotImplementedError`` for another architecture, ``ValueError`` for a hidden
    size it does not serve, ``RuntimeError`` for a toolkit too old to build it (SM107a needs CUDA 13.4) or a width
    whose oversized shared-memory programs this process cannot launch (SM107a at 4096 needs a CUDA 13.4 runtime)."""
    arch = loader.device_arch(device)
    supported = supported_hidden_sizes(arch)
    if hidden_size not in supported:
        raise ValueError(
            f"{_SELECTOR} supports hidden sizes {list(supported)} on {arch}, not {hidden_size}; "
            "use --linear-attention-input-norm te"
        )
    cuda = loader.check_toolkit(arch, _KERNELS)
    index = torch.cuda.current_device() if device.index is None else device.index
    _check_shared_memory(arch, hidden_size, index, cuda.major)
    return arch


@functools.cache
def _check_call(index: int, hidden_size: int) -> None:
    """:func:`_check_device` for a call of width ``hidden_size`` on CUDA device ``index``, remembered once it
    passes."""
    _check_device(hidden_size, torch.device("cuda", index))


def _cuda_device(device: Any) -> torch.device:
    """``device`` as a CUDA device (default: the current one); ``NotImplementedError`` without CUDA."""
    if device is None:
        if not torch.cuda.is_available():
            raise NotImplementedError(f"{_KERNELS} need a CUDA device of SM100a, SM103a or SM107a; there is none")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device)


def check_supported(hidden_size: int, device: Any = None) -> None:
    """Raise unless the Cake kernels normalize ``hidden_size`` features on ``device`` (default: the current CUDA
    device).

    The hidden size is checked first (``ValueError``), so a process without CUDA gets that error too; then the device:
    ``NotImplementedError`` without CUDA or for an architecture other than SM100a / SM103a / SM107a, ``RuntimeError``
    when the toolkit cannot build its kernels (SM107a needs CUDA 13.4 or newer) or the process cannot launch the
    programs of that width in the oversized shared-memory mode (SM107a at 4096 needs a CUDA 13.4 or newer runtime).
    """
    hidden_size = int(hidden_size)
    _check_hidden_size(hidden_size)
    _check_device(hidden_size, _cuda_device(device))


def prebuild(*, device: Any = None, max_workers: int | None = None, hidden_size: int) -> list[Path]:
    """Build every library a call of width ``hidden_size`` may launch on ``device`` (default: the current CUDA device).

    Both variants, forward and backward, every token-count window of the FP32-weight routes: afterwards no call of
    that width compiles on the device's architecture, whatever variant is configured later. Returns the cache path of
    every library; idempotent. ``ValueError`` for a hidden size the kernels do not serve, ``NotImplementedError`` for
    an unsupported device, ``RuntimeError`` for a toolkit too old for it or a width whose oversized shared-memory
    programs this process cannot launch (:func:`check_supported`).
    """
    hidden_size = int(hidden_size)
    _check_hidden_size(hidden_size)
    arch = _check_device(hidden_size, _cuda_device(device))
    paths: list[Path] = []
    for package in _packages():
        jit = package.cake_jit
        names = [
            jit.library_name(module, arch)
            for kind in ("fwd", "bwd")
            for module in jit.route_modules(arch, hidden_size, kind, False, "f32")
        ]
        paths += loader.prebuild_libraries(package.cake_build, names, max_workers)
    return paths


def release() -> None:
    """Nothing to free: the packages keep no device memory between calls (a backward allocates its workspace on the
    call's stream), and their loaded libraries stay loaded for the CUDA graphs that may launch them."""


def _variant() -> str:
    """The variant a call runs: the configured one, ``det`` for ``fast`` under torch's deterministic algorithms."""
    variant = config.variant(_OP)
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        config.warn_runs_det(_OP, "calls under torch.use_deterministic_algorithms(True)")
        return "det"
    return variant


def zero_centered_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Autograd-enabled ``bf16(x * rstd * (1 + weight))`` over the last dimension of BF16 ``x`` ``[..., H]``.

    The leading dimensions of ``x`` collapse into rows (miles hands ``[T, 1, H]``); its rows may be strided
    (``stride(-2)`` a multiple of 8 elements, 32-byte aligned storage) and are copied once otherwise. ``weight``
    ``[H]`` is ``w`` of the ``(1 + w)`` scale, in any float dtype; its gradient comes in its dtype. The call runs the
    variant configured when it runs (``det`` when unconfigured), and its backward the same one. Returns BF16 ``y`` of
    ``x``'s shape. The first call of a width on a CUDA device runs :func:`check_supported` for them, so a width this
    process cannot launch raises its ``RuntimeError`` before any launch.
    """
    deterministic = _variant() == "det"
    hidden = x.shape[-1]
    if x.is_cuda:
        _check_call(x.device.index, hidden)
    _, fast = _packages()
    y = fast.cake_rmsnorm(x.reshape(-1, hidden), 1.0 + weight.float(), eps, deterministic=deterministic)
    return y.view(x.shape)


def _weight_device(device: Any) -> torch.device:
    """Where a new weight goes: ``device``; else where torch's factory functions allocate (a ``with torch.device(...)``
    block, ``torch.set_default_device``) unless that is the CPU; else the current CUDA device, if any; else the CPU."""
    if device is not None:
        return torch.device(device)
    default = torch.empty(0).device
    if default.type == "cpu" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return default


class ZeroCenteredRMSNorm(torch.nn.Module):
    """``y = bf16(x * rstd * (1 + weight))`` over the last dimension, ``weight`` initialized to zeros.

    Raises ``ValueError`` at construction for a hidden size the kernels do not serve and, when the weight is on a CUDA
    device, checks that device (:func:`check_supported`). Construction compiles nothing: call
    ``miles_ext_kernels.cake.prebuild("rmsnorm", hidden_size=...)`` before the first call. ``device`` defaults to the
    device torch's factory functions use (``with torch.device("meta")`` builds on the meta device), else the current
    CUDA device. ``sequence_parallel``, unless ``None``, is set on the weight at construction and by
    :meth:`reset_parameters`, as Transformer Engine's norms set it: under sequence parallelism each tensor-parallel
    rank sees other tokens, and Megatron all-reduces the gradient of such a parameter so that the ranks keep one
    weight. ``None`` (the default) leaves the attribute to the caller: :meth:`reset_parameters` keeps a mark the caller
    set on the weight. Each call runs the variant configured when it runs.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        device: Any = None,
        sequence_parallel: bool | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.sequence_parallel = None if sequence_parallel is None else bool(sequence_parallel)
        device = _weight_device(device)
        _check_hidden_size(self.hidden_size)
        if device.type == "cuda":
            _check_device(self.hidden_size, device)
        self.weight = torch.nn.Parameter(torch.empty(self.hidden_size, dtype=params_dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """``weight = 0`` (the scale ``1 + w`` is one) and, unless ``sequence_parallel`` is ``None``, its
        ``sequence_parallel`` attribute; also after ``to_empty()``, which replaces the parameter of a module built on
        the meta device (and with it any attribute set on the old one)."""
        torch.nn.init.zeros_(self.weight)
        if self.sequence_parallel is not None:
            self.weight.sequence_parallel = self.sequence_parallel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"expected [..., {self.hidden_size}] input, got {tuple(x.shape)}")
        return zero_centered_rmsnorm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}, sequence_parallel={self.sequence_parallel}"
