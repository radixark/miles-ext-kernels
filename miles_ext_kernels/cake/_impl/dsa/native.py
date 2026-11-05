"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The libraries of the DSA det version: the vendored package ``cake_dsa_sparse_det`` (generated sparse attention and
lightning-indexer kernels, forward and backward), built and loaded through the runtime loader.

The package's ``registry.json`` holds one record per stage: the argument plan, the launch block, the dynamic shared
memory, the kernel / launcher source pair under ``csrc/`` and the compile flags. The kernels are register-MMA and
tcgen05 code; one source set serves every architecture of the record (``arches``), and SM107a compiles the SM103a
records (the package's ``build.json`` lists them for ``compute_107a``, with a CUDA 13.4 or newer nvcc). Each stage of
each architecture is the ``build.json`` library ``<record name>_<arch>``; its launcher takes the bound tensors and
scalars and launches on the stream torch has made current on the tensors' device.
"""

from __future__ import annotations

import functools
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import loader

PACKAGE = "cake_dsa_sparse_det"
# argument kinds of a registry record's ``arg_plan``: tensors (``tma_buffer``: a tensor the launcher encodes a TMA
# descriptor for), scalars and the grid
ARG_KINDS = ("buffer", "tma_buffer", "parameter", "grid")


@functools.cache
def _cake_build() -> Any:
    """The package's ``cake_build`` with the runtime loader installed."""
    return import_module(f"{loader.load_package(PACKAGE).__name__}.cake_build")


@functools.cache
def records() -> dict[str, dict[str, Any]]:
    """``stage -> registry record`` of the package (``registry.json``)."""
    return json.loads((Path(_cake_build().__file__).parent / "registry.json").read_text())


def library_name(stage: str, arch: str) -> str:
    """The ``build.json`` library of ``stage`` on ``arch``."""
    return f"{records()[stage]['name']}_{arch}"


def library_names(arch: str) -> list[str]:
    """Every library of the package on ``arch``, one per stage."""
    known = _cake_build().libraries_by_name()
    names = [library_name(stage, arch) for stage in sorted(records())]
    if missing := [name for name in names if name not in known]:
        raise NotImplementedError(f"{PACKAGE} has no {arch} libraries {missing[:3]}")
    return names


def prebuild(arch: str, max_workers: int | None = None) -> list[Path]:
    """Build (or find in the cache) every library of the package on ``arch``; returns their paths."""
    return loader.prebuild_libraries(_cake_build(), library_names(arch), max_workers)


def _raw_stream(device_index: int) -> int:
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    return getter(device_index) if getter is not None else torch.cuda.current_stream(device_index).cuda_stream


@functools.cache
def _ffi_device(device_index: int) -> Any:
    import tvm_ffi

    return tvm_ffi.device(f"cuda:{device_index}")


class NativeKernel:
    """One stage on one architecture: named bindings are mapped onto the exported argument plan.

    ``launch(device_index, grid=(x, y, z), **bindings)`` binds tensors and scalars by their exported names, fills in
    the grid and launches on torch's current stream of ``device_index``. A ``tma_buffer`` argument is bound like a
    ``buffer``: the stage's launcher encodes its TMA descriptor from the tensor on the host.
    """

    def __init__(self, stage: str, arch: str) -> None:
        record = records()[stage]
        if any(kind not in ARG_KINDS for kind, _ in record["arg_plan"]):
            raise NotImplementedError(f"{stage}: argument kinds {sorted({kind for kind, _ in record['arg_plan']})}")
        self.stage = stage
        self.arch = arch
        self.record = record
        self._call = getattr(_cake_build().load_library(library_name(stage, arch)), record["ffi_entry"])
        self._arg_plan = tuple(tuple(item) for item in record["arg_plan"])

    def launch(self, device_index: int, *, grid, **bindings) -> None:
        import tvm_ffi

        grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
        args = []
        used = set()
        for kind, name in self._arg_plan:
            if kind == "grid":
                args.append(grid[("grid_x", "grid_y", "grid_z").index(name)])
                continue
            if name not in bindings:
                raise KeyError(f"{self.stage}: missing binding {name!r}")
            value = bindings[name]
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value = value.detach()
            args.append(value)
            used.add(name)
        if unexpected := set(bindings) - used:
            raise KeyError(f"{self.stage}: unexpected bindings {sorted(unexpected)!r}")
        with tvm_ffi.use_raw_stream(_ffi_device(device_index), _raw_stream(device_index)):
            self._call(*args)


@functools.cache
def kernel(stage: str, arch: str) -> NativeKernel:
    """The process-wide :class:`NativeKernel` of ``stage`` on ``arch`` (loads, and builds unless cached)."""
    return NativeKernel(stage, arch)


def release() -> None:
    """Forget the loaded stages (they load again on their next launch)."""
    kernel.cache_clear()
