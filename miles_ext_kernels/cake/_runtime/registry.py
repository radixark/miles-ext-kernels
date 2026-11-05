"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The registered ops, and the two calls that reach every op's implementation: :func:`prebuild` and :func:`release`.

:data:`OPS` names every ``miles_ext_kernels/cake/<op>.py`` in layer order; the layer that adds an op appends one line.
Looking an op up (:func:`op_spec`) imports only its thin module, never its implementation.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Set
from importlib import import_module
from pathlib import Path
from typing import Any

from . import loader, workspace
from .spec import OpSpec

__all__ = ["OPS", "op_spec", "prebuild", "release"]

# Every op, one line each, in layer order.
OPS: tuple[str, ...] = ("gdn",)

# miles_ext_kernels.cake: op modules are its submodules, implementations those of its _impl package
_SOURCE = __name__.rsplit(".", 2)[0]
# What prebuild returned in this process: (op, (device index, architecture), toolkit, cache directory, model
# parameters, op options) -> library paths.
_PREBUILT: dict[tuple[Any, ...], tuple[Path, ...]] = {}


def op_spec(op: str) -> OpSpec:
    """The ``OP`` of ``miles_ext_kernels.cake.<op>``; ``KeyError`` for an op that is not registered."""
    if op not in OPS:
        raise KeyError(f"no cake op {op!r}; the ops are {OPS}")
    name = f"{_SOURCE}.{op}"
    # every call of an op reads its spec (variant, options): an imported module is looked up, not imported again
    spec = getattr(sys.modules.get(name), "OP", None)
    return import_module(name).OP if spec is None else spec


def prebuild(op: str, /, *, device: Any = None, max_workers: int | None = None, **model_params: Any) -> list[Path]:
    """Build every library ``op`` may launch for the model shape ``model_params`` on ``device`` (default: current).

    After it returns, no call of ``op`` with these model parameters compiles on that device: any variant (also one
    configured later), any token count, every route the op takes. ``model_params`` are model shapes only (head
    counts, hidden sizes, dtypes, ...), never a token count or a variant; each op documents its own. Unsupported
    parameters raise ``ValueError``, an unsupported device ``NotImplementedError``, a toolkit too old for the device
    ``RuntimeError``. Idempotent; returns the cache path of every library, built now or before.

    The paths are remembered per op, CUDA device (index and architecture), toolkit (nvcc and its CUDA version), cache
    directory, model parameters and options of the op until :func:`release`: a repeated call (miles makes one per
    layer it builds) returns them without looking at the libraries again.
    """
    op_spec(op)
    impl = import_module(f"{_SOURCE}._impl.{op}")
    # an iterator parameter is read once, for the key and the op alike
    model_params = {name: tuple(v) if isinstance(v, Iterator) else v for name, v in model_params.items()}
    key = _prebuilt_key(op, device, model_params)
    if key in _PREBUILT:
        return list(_PREBUILT[key])
    paths = tuple(impl.prebuild(device=device, max_workers=max_workers, **model_params))
    if key is not None:
        _PREBUILT[key] = paths
    return list(paths)


def _prebuilt_key(op: str, device: Any, model_params: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """The key :func:`prebuild` remembers its paths under, ``None`` when it cannot remember them.

    That is when the device cannot be told (no torch, no CUDA, not a CUDA device, an unsupported architecture) or the
    toolkit cannot (no nvcc, or it does not run): the op's own prebuild reports it. It is also ``None`` when a model
    parameter is neither hashable nor a collection of hashable values. Values are compared with their types: ``12``
    and ``12.0`` are different parameters.
    """
    from . import config  # config imports this module

    target = _device(device)
    if target is None:
        return None
    try:
        toolkit = loader._toolkit()  # (nvcc, its CUDA version); rereads the version when the nvcc file changes
    except RuntimeError:
        return None
    options = {flag.dest: config.option(op, flag.dest) for flag in op_spec(op).flags}
    try:
        key = (op, target, toolkit, str(loader.cache_dir()), _frozen(model_params), _frozen(options))
        hash(key)
    except TypeError:
        return None
    return key


def _device(device: Any) -> tuple[int, str] | None:
    """The index and architecture (``loader.ARCH_BY_CAPABILITY``) of the CUDA ``device``, ``None`` if it has none.

    ``device`` is an index, ``"cuda"``, ``"cuda:N"`` or a ``torch.device``; ``None`` and ``"cuda"`` mean the current
    device.
    """
    try:
        import torch

        if not isinstance(device, int):
            device = torch.device("cuda" if device is None else device)
            if device.type != "cuda":
                return None
            device = torch.cuda.current_device() if device.index is None else device.index
        arch = loader.ARCH_BY_CAPABILITY.get(tuple(torch.cuda.get_device_capability(device)))
    except (ImportError, AssertionError, RuntimeError, TypeError, ValueError):
        return None
    return None if arch is None else (device, arch)


def _frozen(value: Any) -> Any:
    """``value`` compared by content and type: mappings, sets, lists and tuples become hashable."""
    if isinstance(value, Mapping):
        return frozenset((name, _frozen(item)) for name, item in value.items())
    if isinstance(value, Set):
        return frozenset(_frozen(item) for item in value)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return type(value), value


def release(op: str | None = None) -> int:
    """Free the workspace pool (of every op, or of ``op``) and the caches of every imported implementation.

    Each imported ``miles_ext_kernels.cake._impl.<op>`` drops its plans, graphs and bindings in its ``release()``, and
    :func:`prebuild` forgets the paths it remembered for those ops. Returns the bytes freed from the pool;
    ``torch.cuda.empty_cache()`` is left to the caller. miles calls it through ``miles_ext_kernels.release()``
    before it sleeps or offloads.
    """
    ops = OPS if op is None else (op_spec(op).name,)
    for key in [key for key in _PREBUILT if op is None or key[0] == op]:
        del _PREBUILT[key]
    freed = workspace.release(op)
    for name in ops:
        impl = sys.modules.get(f"{_SOURCE}._impl.{name}")
        if impl is not None:
            impl.release()
    return freed
