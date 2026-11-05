"""Training kernels for miles, enabled by miles' ``--use-miles-ext-kernels``.

Kernels are grouped by source, one namespace each (:data:`SOURCES`); today that is ``miles_ext_kernels.cake``. miles
owns backend selection (``--linear-attention-backend cake``, ...) and imports an op at its dispatch seam
(``from miles_ext_kernels.cake.gdn import chunk_gated_delta_rule``). The four hooks below are the whole
process-level surface miles calls; each forwards to every source's ``_runtime``.

Importing this module imports neither torch, tvm_ffi nor any op.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = ["SOURCES", "add_arguments", "configure", "release", "validate_args"]

SOURCES = ("cake",)


def _runtimes() -> list[ModuleType]:
    return [import_module(f"{__name__}.{source}._runtime") for source in SOURCES]


def add_arguments(parser: ArgumentParser) -> None:
    """Register every source's arguments in one ``miles-ext-kernels`` group (miles calls it before parsing)."""
    group = parser.add_argument_group("miles-ext-kernels")
    for runtime in _runtimes():
        runtime.add_arguments(group)


def validate_args(args: Namespace) -> None:
    """Check miles' parsed flat namespace and resolve the values left to defaults (the driver process)."""
    for runtime in _runtimes():
        runtime.validate_args(args)


def configure(args: Namespace | Mapping[str, Any] | None = None, /, **values: Any) -> None:
    """Configure this process from the resolved values (miles passes ``args.miles_ext_kernels_args``)."""
    for runtime in _runtimes():
        runtime.configure(args, **values)


def release() -> int:
    """Free every source's workspaces and cached plans (miles calls it before it sleeps or offloads).

    Returns the bytes freed from the workspace pools; ``torch.cuda.empty_cache()`` is left to the caller.
    """
    return sum(runtime.release() for runtime in _runtimes())
