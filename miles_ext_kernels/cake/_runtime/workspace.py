"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

One bounded pool of temporary device buffers for every op.

An op asks for scratch memory with :func:`take` under a key ``(op, role)``. The pool keeps at most
``--cake-workspace-pool-bytes`` per device. The default, ``0``, keeps nothing: every ``take`` is a plain
``torch.empty`` (measured training steps were no faster with a pool, which held memory between calls).

With a positive bound the pool keeps one buffer per ``(op, role)`` and CUDA stream, reused while it is large enough,
and evicts the least recently used buffers to stay under the bound; a request above the bound is not kept. A stream
that captures a CUDA graph bypasses the pool, so a graph never records a buffer that the pool frees later.

A buffer from :func:`take` is valid until the next :func:`take` of the same key on the same stream, or
:func:`release`: use it on that stream only, and never keep it in an autograd context or return it to a caller.

Importing this module does not import torch.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from . import config

__all__ = ["release", "stats", "take"]

# device -> (op, role, stream) -> uint8 buffer, least recently used first
_pools: dict[Any, OrderedDict[tuple[str, str, Any], Any]] = {}
# device -> hits, misses, evictions
_counters: dict[Any, dict[str, int]] = {}
_lock = threading.Lock()


def _device(device: Any) -> tuple[Any, Any]:
    """``(pool key, torch.device)``: the CUDA index (the current device for ``"cuda"``), else the device name."""
    import torch

    target = torch.device(device)
    if target.type != "cuda":
        return str(target), target
    index = torch.cuda.current_device() if target.index is None else target.index
    return index, torch.device("cuda", index)


def _stream(where: Any) -> tuple[Any, bool]:
    """``(stream handle, capturing)`` of the current stream of the CUDA device ``where``; ``(None, False)`` off CUDA."""
    if not isinstance(where, int):
        return None, False
    import torch

    with torch.cuda.device(where):
        if torch.cuda.is_current_stream_capturing():
            return None, True
        return torch.cuda.current_stream().cuda_stream, False


def _held(pool: OrderedDict) -> int:
    return sum(buffer.numel() for buffer in pool.values())


def _evict(pool: OrderedDict, counters: dict[str, int], room: int) -> None:
    """Drop the least recently used buffers until the pool holds at most ``room`` bytes."""
    held = _held(pool)
    while pool and held > room:
        _, buffer = pool.popitem(last=False)
        held -= buffer.numel()
        counters["evictions"] += 1


def take(op: str, role: str, shape: Sequence[int], dtype: Any, device: Any) -> Any:
    """A tensor of ``shape`` and ``dtype`` on ``device``, a view of the pooled buffer of ``(op, role)``.

    Its content is undefined. With the bound at 0 and nothing held, a plain ``torch.empty``.
    """
    import torch

    limit = config.option(None, "cake_workspace_pool_bytes")
    where, target = _device(device)
    with _lock:
        counters = _counters.setdefault(where, {"hits": 0, "misses": 0, "evictions": 0})
        pool = _pools.setdefault(where, OrderedDict())
        direct = not limit and not pool
        if direct:
            counters["misses"] += 1
    if direct:
        return torch.empty(shape, dtype=dtype, device=target)

    stream, capturing = _stream(where)
    if capturing:
        with _lock:
            counters["misses"] += 1
        return torch.empty(shape, dtype=dtype, device=target)

    nbytes = math.prod(shape) * dtype.itemsize
    key = (op, role, stream)
    with _lock:
        buffer = pool.get(key)
        if buffer is not None and nbytes <= buffer.numel() <= limit:
            counters["hits"] += 1
            pool.move_to_end(key)
            _evict(pool, counters, limit)
        else:
            counters["misses"] += 1
            if buffer is not None:  # too small, or above a lowered bound
                del pool[key]
            if not 0 < nbytes <= limit:
                _evict(pool, counters, limit)
                return torch.empty(shape, dtype=dtype, device=target)
            _evict(pool, counters, limit - nbytes)
            buffer = pool[key] = torch.empty(nbytes, dtype=torch.uint8, device=target)

    view = buffer[:nbytes]
    if dtype is not torch.uint8:
        view = view.view(dtype)
    return view.view(shape)


def release(op: str | None = None) -> int:
    """Drop the pooled buffers of every op, or of ``op``, on every device and stream; returns the bytes dropped."""
    freed = 0
    with _lock:
        for pool in _pools.values():
            for key in [key for key in pool if op is None or key[0] == op]:
                freed += pool.pop(key).numel()
    return freed


def stats() -> dict[Any, dict[str, int]]:
    """``{device: {"bytes", "entries", "hits", "misses", "evictions"}}``; ``device`` is the CUDA index."""
    with _lock:
        return {
            where: {"bytes": _held(_pools[where]), "entries": len(_pools[where]), **_counters[where]}
            for where in sorted(_counters, key=str)
        }
