"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The declaration of one op: :class:`OpSpec` and its option flags (:class:`Flag`), torch-free.

``miles_ext_kernels/cake/<op>.py`` holds the op's ``OP = OpSpec(...)`` and nothing else but its lazy exports
(:func:`lazy_exports`); the implementation lives in ``miles_ext_kernels/cake/_impl/<op>/`` and is imported on first
use of an export. The runtime reads everything it needs (arguments, variants, vendored packages) from ``OP``, so
importing an op module stays as cheap as importing the runtime.
"""

from __future__ import annotations

import re
import sys
from argparse import Namespace
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from types import MappingProxyType
from typing import Any

__all__ = ["VARIANTS", "Flag", "OpSpec", "lazy_exports"]

# Every variant an op may ship: "det" repeats bit for bit for identical inputs, "fast" is faster.
VARIANTS = ("det", "fast")

_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class Flag:
    """One ``--cake-<op>-<what>`` option of an op: its name and the keyword arguments of ``add_argument``.

    The ``dest`` follows from the name (``--cake-gdn-recompute-state`` -> ``cake_gdn_recompute_state``); ``kwargs``
    must hold ``default`` and ``help``.
    """

    name: str
    kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"--cake-[a-z0-9]+(-[a-z0-9]+)*", self.name):
            raise ValueError(f"{self.name!r} is not a --cake-<op>-<what> flag")
        missing = sorted({"default", "help"} - set(self.kwargs))
        if missing:
            raise ValueError(f"{self.name}: add_argument keywords {missing} are missing")
        if "dest" in self.kwargs:
            raise ValueError(f"{self.name}: the dest follows from the name")
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))

    @cached_property
    def dest(self) -> str:  # read by config.option at every call of an op
        return self.name[2:].replace("-", "_")

    @property
    def default(self) -> Any:
        return self.kwargs["default"]


@dataclass(frozen=True)
class OpSpec:
    """What the runtime knows of one op.

    ``variants`` are the values ``configure`` accepts, a subset of :data:`VARIANTS` in its order (``det`` first).
    ``cli_variants`` are the values miles may choose with ``--cake-<op>-variant`` (``None``: all of ``variants``).
    The flag exists when the op has a miles selector (``selected_by``: the ``(dest, value)`` pairs of miles arguments
    that route a call site to this op) and more than one variant; it is hidden from the help when miles may choose
    only one, and still carries the variant ``validate_args`` resolves to every process. ``packages`` are the
    vendored packages the op loads, the keys of its block in ``_vendor/VENDORED.json``. ``exports`` are the public
    names ``miles_ext_kernels.cake.<op>`` re-exports from ``miles_ext_kernels.cake._impl.<op>``. ``flags`` are the
    op's options besides its variant. ``validate`` (torch-free) checks miles' parsed namespace when the op is
    selected.
    """

    name: str
    summary: str
    variants: tuple[str, ...]
    cli_variants: tuple[str, ...] | None = None
    selected_by: tuple[tuple[str, str], ...] = ()
    packages: tuple[str, ...] = ()
    exports: tuple[str, ...] = ()
    flags: tuple[Flag, ...] = ()
    validate: Callable[[Namespace], None] | None = None

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"op name {self.name!r} is not a module name")
        if not self.variants or self.variants != tuple(v for v in VARIANTS if v in self.variants):
            raise ValueError(
                f"{self.name}: variants {self.variants} must be a non-empty subset of {VARIANTS}, in order"
            )
        if self.cli_variants is not None and (
            not self.cli_variants or self.cli_variants != tuple(v for v in self.variants if v in self.cli_variants)
        ):
            raise ValueError(f"{self.name}: cli_variants {self.cli_variants} must be a non-empty subset of variants")
        prefix = f"--cake-{self.dashed}-"
        dests = [flag.dest for flag in self.flags]
        for flag in self.flags:
            if not flag.name.startswith(prefix) or flag.name == self.variant_flag:
                raise ValueError(f"{self.name}: option {flag.name} must start with {prefix} and not be its variant")
        for what, names in (("flag", dests), ("package", self.packages), ("export", self.exports)):
            if len(set(names)) != len(names):
                raise ValueError(f"{self.name}: a {what} is listed twice in {names}")
        if "OP" in self.exports:
            raise ValueError(f"{self.name}: OP is not an export")

    @property
    def dashed(self) -> str:
        return self.name.replace("_", "-")

    @property
    def variant_flag(self) -> str:
        return f"--cake-{self.dashed}-variant"

    @property
    def variant_dest(self) -> str:
        return f"cake_{self.name}_variant"

    @property
    def cli_choices(self) -> tuple[str, ...]:
        return self.variants if self.cli_variants is None else self.cli_variants

    @property
    def default_variant(self) -> str:
        """The variant of a process that never configured this op: ``det``, or the op's only variant."""
        return "det" if "det" in self.variants else self.variants[0]

    @property
    def has_variant_flag(self) -> bool:
        """Is ``--cake-<op>-variant`` registered? (It carries the resolved variant whenever the op has two.)"""
        return bool(self.selected_by) and len(self.variants) > 1

    @property
    def shows_variant_flag(self) -> bool:
        """Does the help list ``--cake-<op>-variant``? Only when miles may choose among more than one variant."""
        return self.has_variant_flag and len(self.cli_choices) > 1

    @property
    def selector(self) -> str:
        """The miles arguments that select this op, as a user types them."""
        return " or ".join(f"--{dest.replace('_', '-')} {value}" for dest, value in self.selected_by)


def lazy_exports(module_name: str, op: OpSpec) -> tuple[list[str], Callable[[str], Any], Callable[[], list[str]]]:
    """``(__all__, __getattr__, __dir__)`` of an op module: ``__all__ = ["OP", *op.exports]``.

    An export is imported from the op's ``_impl`` package (``<source>._impl.<op>``, next to the op module) on first
    access and cached on the module, so importing the op module imports neither torch nor the implementation.
    """
    names = ["OP", *op.exports]
    exports = frozenset(op.exports)
    impl = f"{module_name.rpartition('.')[0]}._impl.{op.name}"

    def __getattr__(name: str) -> Any:
        if name not in exports:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        value = getattr(import_module(impl), name)
        setattr(sys.modules[module_name], name, value)
        return value

    def __dir__() -> list[str]:
        return sorted({*vars(sys.modules[module_name]), *names})

    return names, __getattr__, __dir__
