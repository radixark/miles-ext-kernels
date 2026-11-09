"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Launchers of the generated det kernels with FLA's context parallelism (the vendored package ``cake_gdn_fla_cp_det``),
the ones ``ops.cp_chunk_gated_delta_rule`` runs.

The package's ``registry.json`` is written by the kernel generator: one record per stage with the argument plan, the
launch block, the dynamic shared memory, the device / launcher source pair under ``csrc/`` and the compile flags. The
generated device code is the same text for SM100a and SM103a (``__CUDA_ARCH__`` guards where they differ), so one
source pair serves both architectures (``arches``); a stage exported with a different schedule per architecture carries
its pairs under ``per_arch`` (``dqkwg``: eight warps on SM103a). SM107a has no export of its own: it compiles the SM103a
records (:data:`EXPORT_ARCH`, the package's ``build.json`` does the same), whose device code takes the same non-SM100a
paths there. Every library is ``<record name>_<arch>`` of the package's ``build.json``, built with nvcc and loaded with
tvm-ffi by the runtime loader (CUDA 13.4 or newer).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from miles_ext_kernels.cake._runtime import loader

__all__ = ["EXPORT_ARCH", "LAUNCH_ARG_KINDS", "MODULES", "PACKAGE", "NativeKernel", "kernel", "launcher", "libraries"]

# the vendored package of the generated kernels
PACKAGE = "cake_gdn_fla_cp_det"
# architecture -> the exported architecture whose records it compiles
EXPORT_ARCH = {"sm_107a": "sm_103a"}
# the argument kinds of an exported plan: tensors and scalars by name, then the grid
LAUNCH_ARG_KINDS = frozenset({"buffer", "parameter", "grid"})
_GRID_INDEX = {"grid_x": 0, "grid_y": 1, "grid_z": 2}


def _registry_path() -> Path:
    return Path(loader.load_package(PACKAGE).__file__).parent / "registry.json"


def _load_registry() -> dict[str, dict[str, dict[str, Any]]]:
    modules: dict[str, dict[str, dict[str, Any]]] = {}
    for stage, record in json.loads(_registry_path().read_text()).items():
        per_arch = record.pop("per_arch", None)
        for arch in record["arches"]:
            modules.setdefault(stage, {})[arch] = record if per_arch is None else {**record, **per_arch[arch]}
    for records in modules.values():
        for arch, exported in EXPORT_ARCH.items():
            if exported in records:
                records.setdefault(arch, records[exported])
    return modules


# MODULES[stage][arch] -> generated record (from registry.json); a stage exported once for every architecture shares one
# record object across them, and an EXPORT_ARCH architecture the record of its export.
MODULES = _load_registry()


def libraries(arch: str) -> list[str]:
    """The ``build.json`` library of every stage on ``arch`` (``<record name>_<arch>``)."""
    return [f"{records[arch]['name']}_{arch}" for records in MODULES.values() if arch in records]


@cache
def load(stage: str, arch: str) -> Any:
    """The loaded tvm-ffi module of ``stage`` on ``arch``, built by the runtime loader unless it is cached."""
    return loader.load_package(PACKAGE).cake_build.load_library(f"{MODULES[stage][arch]['name']}_{arch}")


class NativeKernel:
    """One generated stage; named bindings are mapped onto the exported argument plan.

    ``launch(grid=(x, y, z), **bindings)`` binds tensors and scalars by their exported names; the grid is filled in.
    """

    def __init__(self, stage: str, arch: str) -> None:
        self.stage = stage
        self.arch = arch
        record = MODULES[stage][arch]
        self.record = record
        self._call = getattr(load(stage, arch), record["ffi_entry"])
        self._arg_plan = tuple(tuple(item) for item in record["arg_plan"])
        # the launchers encode their TMA descriptors inside the binding, so a plan only carries buffers, parameters and
        # the grid
        unknown = sorted({kind for kind, _ in self._arg_plan} - LAUNCH_ARG_KINDS)
        if unknown:
            raise ValueError(f"{stage}: unsupported argument kinds {unknown!r} in the exported plan")
        self._binding_names = frozenset(name for kind, name in self._arg_plan if kind != "grid")

    def launch(self, *, grid, **bindings) -> Any:
        if len(bindings) != len(self._binding_names) or not self._binding_names.issuperset(bindings):
            missing = sorted(self._binding_names.difference(bindings))
            unexpected = sorted(set(bindings).difference(self._binding_names))
            raise KeyError(f"{self.stage}: missing bindings {missing!r}, unexpected bindings {unexpected!r}")
        if len(grid) < 3:
            grid = tuple(grid) + (1,) * (3 - len(grid))
        args = [int(grid[_GRID_INDEX[name]]) if kind == "grid" else bindings[name] for kind, name in self._arg_plan]
        return self._call(*args)

    def launcher(self, names) -> Any:
        """Positional launcher for one call site.

        ``names`` lists every buffer and parameter name of the exported plan once, in the order the call site passes
        its values; the order is resolved against the plan here, once. The returned ``run(grid_x, grid_y, grid_z,
        *values)`` forwards the values in plan order with the grid appended. A name set that does not match the plan
        raises here; a value count that does not match raises a ``TypeError`` naming the stage.
        """
        names = tuple(names)
        plan = tuple(name for kind, name in self._arg_plan if kind != "grid")
        grid_names = tuple(name for kind, name in self._arg_plan if kind == "grid")
        if sorted(names) != sorted(plan) or len(set(names)) != len(names):
            raise KeyError(
                f"{self.stage}: launcher names {sorted(set(names) - set(plan))!r} unknown / "
                f"{sorted(set(plan) - set(names))!r} missing"
            )
        if (
            grid_names != ("grid_x", "grid_y", "grid_z")
            or tuple(kind for kind, _ in self._arg_plan[-3:]) != ("grid",) * 3
        ):
            raise ValueError(
                f"{self.stage}: the exported plan does not end with grid_x, grid_y, grid_z: {self._arg_plan!r}"
            )
        call, stage, count = self._call, self.stage, len(plan)
        if names == plan:

            def run(grid_x, grid_y, grid_z, *values):
                if len(values) != count:
                    raise TypeError(f"{stage}: launcher expects {count} values, got {len(values)}")
                return call(*values, grid_x, grid_y, grid_z)

        else:
            order = tuple(names.index(name) for name in plan)  # call-site position of every plan slot

            def run(grid_x, grid_y, grid_z, *values):
                if len(values) != count:
                    raise TypeError(f"{stage}: launcher expects {count} values, got {len(values)}")
                return call(*[values[i] for i in order], grid_x, grid_y, grid_z)

        run.stage, run.arch, run.names, run.plan = self.stage, self.arch, names, plan
        return run


@cache
def kernel(stage: str, arch: str) -> NativeKernel:
    """The process-wide :class:`NativeKernel` of ``stage`` on ``arch``."""
    return NativeKernel(stage, arch)


@cache
def launcher(stage: str, arch: str, names: tuple[str, ...]) -> Any:
    """The process-wide positional launcher of ``stage`` on ``arch`` for the call-site order ``names``."""
    return kernel(stage, arch).launcher(names)


def release() -> None:
    """Drop the loaded libraries and their launchers (the next launch loads them from the cache again)."""
    launcher.cache_clear()
    kernel.cache_clear()
    load.cache_clear()
