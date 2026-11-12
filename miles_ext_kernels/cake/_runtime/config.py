"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The one argument surface of the Cake ops and the configuration of this process (torch-free).

miles calls four hooks, through ``miles_ext_kernels``:

- :func:`add_arguments` before it parses: for every op with a miles selector, ``--cake-<op>-variant`` (when the op
  has two variants; hidden from the help when miles may choose only one) and the op's own ``--cake-<op>-...``
  options; then the runtime options.
- :func:`validate_args` on its parsed flat namespace: rejects an option of an op that is not selected, resolves the
  variant of every selected op and writes it back, and runs the op's own checks.
- :func:`configure` in every process that calls a kernel, with the values miles keeps (``miles_ext_kernels_args``).
- ``release`` (in :mod:`.registry`) before it sleeps or offloads.

Variants. ``det`` repeats bit for bit for identical inputs (on the same GPU model and SM count, toolkit, build and
route); ``fast`` is faster. A process that never configured an op runs ``det`` (or the op's only variant).
``validate_args`` picks ``fast`` for a selected op whose flag is unset (if miles may choose it), so miles' command line
still defaults to ``fast``, and writes the resolved variant back so that it reaches every process; under
``--deterministic-mode`` it picks ``det``, rejects an explicit ``fast`` and rejects an op that has no ``det``.
``configure(..., deterministic_mode=True)`` holds the same rules for direct callers: it rejects an explicit ``fast``,
and :func:`variant` raises for an op without ``det``. The implementation reads :func:`variant` when a call runs (a
backward uses the variant its forward ran), so configuring after a module was built takes effect at its next call.

The configuration is one immutable mapping, replaced whole by every :func:`configure`.
"""

from __future__ import annotations

import argparse
import logging
from argparse import Namespace
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from . import registry
from .spec import Flag, OpSpec

__all__ = [
    "RUNTIME_FLAGS",
    "add_arguments",
    "configure",
    "deterministic_mode",
    "option",
    "validate_args",
    "variant",
    "warn_runs_det",
]

# Options of the runtime itself, read with option(None, dest).
RUNTIME_FLAGS = (
    Flag(
        "--cake-workspace-pool-bytes",
        dict(
            type=int,
            default=0,
            help=(
                "Upper bound of the bytes the Cake ops keep in their workspace pool per device. Default 0: nothing "
                "is kept and every scratch buffer is a new allocation; a positive bound reuses buffers across calls."
            ),
        ),
    ),
)

logger = logging.getLogger("miles_ext_kernels")

_state: Mapping[str, Any] = MappingProxyType({})
_warned: set[tuple[str, str]] = set()


def _specs() -> Iterator[OpSpec]:
    return (registry.op_spec(op) for op in registry.OPS)


def deterministic_mode(args: Any) -> bool:
    """Does miles run in deterministic mode?

    miles' parsed flat namespace carries Megatron's ``--deterministic-mode`` (and the FSDP backend's field of the
    same name) as ``args.deterministic_mode``; its typed configuration as ``args.backend.deterministic_mode``. The
    torchtitan backend has neither, so it never runs in deterministic mode.
    """
    value = getattr(args, "deterministic_mode", None)
    if value is None:
        value = getattr(getattr(args, "backend", None), "deterministic_mode", False)
    return bool(value)


def add_arguments(parser: Any) -> None:
    """Register every op's arguments and the runtime options on ``parser`` (a parser or an argument group)."""
    for spec in _specs():
        if not spec.selected_by:
            continue
        if spec.has_variant_flag:
            default = "fast" if "fast" in spec.cli_choices else "det"
            parser.add_argument(
                spec.variant_flag,
                choices=spec.cli_choices,
                default=None,
                help=(
                    (
                        f"Variant of the cake {spec.name} kernels ({spec.summary}) under {spec.selector}: 'det' "
                        f"repeats bit for bit for identical inputs, 'fast' is faster. Default: '{default}', or 'det' "
                        "under --deterministic-mode, which rejects 'fast'."
                    )
                    if spec.shows_variant_flag
                    else argparse.SUPPRESS  # one choice: the flag only carries the resolved variant
                ),
            )
        for flag in spec.flags:
            parser.add_argument(flag.name, **flag.kwargs)
    for flag in RUNTIME_FLAGS:
        parser.add_argument(flag.name, **flag.kwargs)


def _no_det(spec: OpSpec) -> ValueError:
    selector = f": {spec.selector} is not deterministic" if spec.selector else ": it cannot run in deterministic mode"
    return ValueError(f"the cake {spec.name} kernels have no det variant{selector}")


def _resolve(spec: OpSpec, requested: str | None, deterministic: bool, *, cli: bool) -> str | None:
    """The variant of ``spec`` to run; ``None`` leaves it unset. ``cli``: miles' default for an unset flag."""
    if requested is not None:
        choices = spec.cli_choices if cli else spec.variants
        if requested not in choices:
            raise ValueError(f"{spec.variant_flag} must be one of {choices}, not {requested!r}")
        if deterministic and requested == "fast":
            raise ValueError(
                f"{spec.variant_flag} fast is not the det variant that --deterministic-mode runs: "
                "use det or leave it unset"
            )
        return requested
    if deterministic:
        if "det" not in spec.variants:
            raise _no_det(spec)
        return "det"
    if cli:
        return "fast" if "fast" in spec.cli_choices else "det"
    return None


def validate_args(args: Namespace) -> None:
    """Check miles' parsed flat namespace and write the resolved variant of every selected op back to it."""
    deterministic = deterministic_mode(args)
    for spec in _specs():
        if not spec.selected_by:
            continue
        selected = any(getattr(args, dest, None) == value for dest, value in spec.selected_by)
        if not selected:
            flags = [(spec.variant_flag, spec.variant_dest, None)] if spec.has_variant_flag else []
            flags += [(flag.name, flag.dest, flag.default) for flag in spec.flags]
            for name, dest, default in flags:
                if getattr(args, dest, default) != default:
                    raise ValueError(f"{name} requires {spec.selector}")
            continue
        resolved = _resolve(spec, getattr(args, spec.variant_dest, None), deterministic, cli=True)
        if spec.has_variant_flag:
            setattr(args, spec.variant_dest, resolved)
        if spec.validate is not None:
            spec.validate(args)
    for flag in RUNTIME_FLAGS:
        _check_runtime(flag.dest, getattr(args, flag.dest, flag.default))


def _check_runtime(dest: str, value: Any) -> None:
    if dest == "cake_workspace_pool_bytes" and value < 0:
        raise ValueError(f"--cake-workspace-pool-bytes must be >= 0, not {value}")


def configure(args: Namespace | Mapping[str, Any] | None = None, /, **values: Any) -> None:
    """Replace this process's configuration with the ``cake_*`` values of ``args`` and ``values``.

    ``args`` is a namespace, a mapping (miles' ``miles_ext_kernels_args``) or ``None``; ``values`` override it. A
    ``None`` value counts as unset; keys that do not start with ``cake_`` are ignored, so a whole namespace may be
    passed. Raises ``ValueError`` for an unknown ``cake_*`` key, a variant the op does not ship, and ``fast`` when
    ``args`` or ``values`` say ``deterministic_mode``; in that mode :func:`variant` also raises for an unconfigured op
    without ``det``. Every call starts from the defaults, so it is idempotent.
    """
    if args is None:
        given: dict[str, Any] = {}
    elif isinstance(args, Mapping):
        given = dict(args)
    else:
        given = dict(vars(args))
    given.update(values)
    deterministic = deterministic_mode(Namespace(**given))
    known: dict[str, OpSpec | Flag] = {flag.dest: flag for flag in RUNTIME_FLAGS}
    for spec in _specs():
        known[spec.variant_dest] = spec
        known.update((flag.dest, flag) for flag in spec.flags)
    state: dict[str, Any] = {"deterministic_mode": deterministic}
    for key, value in given.items():
        if not key.startswith("cake_") or value is None:
            continue
        what = known.get(key)
        if what is None:
            raise ValueError(f"unknown cake option {key!r}")
        if isinstance(what, OpSpec):
            value = _resolve(what, value, deterministic, cli=False)
        else:
            _check_runtime(key, value)
        state[key] = value
    global _state
    _state = MappingProxyType(state)


def variant(op: str) -> str:
    """The variant ``op`` runs now: the configured one, else ``det`` (or the op's only variant).

    ``ValueError`` for an unconfigured op without ``det`` in a process configured with ``deterministic_mode``.
    """
    value = _state.get(f"cake_{op}_variant")
    if value is not None:
        return value
    spec = registry.op_spec(op)
    if _state.get("deterministic_mode") and "det" not in spec.variants:
        raise _no_det(spec)
    return spec.default_variant


def option(op: str | None, dest: str) -> Any:
    """The configured value of the option ``dest`` of ``op`` (``None``: a runtime option), else its default."""
    flags = RUNTIME_FLAGS if op is None else registry.op_spec(op).flags
    for flag in flags:
        if flag.dest == dest:
            return _state.get(dest, flag.default)
    raise KeyError(f"{'the runtime' if op is None else op} has no option {dest!r}")


def warn_runs_det(op: str, reason: str) -> None:
    """Log once per ``(op, reason)`` that a call the ``fast`` variant does not serve runs ``det``."""
    if (op, reason) in _warned:
        return
    _warned.add((op, reason))
    logger.warning("miles_ext_kernels.cake.%s: the fast variant does not serve %s; running det", op, reason)
