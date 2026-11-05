"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

One bounded pool of temporary device buffers for every op.

An op asks for scratch memory with :func:`take` under a key ``(op, role)``. The bytes the pool keeps per device stay
under the runtime option ``--cake-workspace-pool-bytes``. Its default, ``0``, keeps nothing: every ``take`` is a plain
``torch.empty`` on the current stream (measured training steps were no faster with a pool, which held memory between
calls). With a positive bound the pool keeps one buffer per key, device and stream (the device's current CUDA stream
when ``take`` runs), reused when it is large enough and replaced when it is not, so calls on two streams never share a
buffer: every ``take`` evicts the least recently used other buffers until the pool fits (also after the bound was
lowered), and a request above the bound is allocated on its own and not kept. While the current stream captures a CUDA
graph, ``take`` allocates outside the pool (from the graph's private memory), so a graph never records a pooled buffer
that a later eviction or :func:`release` frees.

A buffer from :func:`take` is valid until the next :func:`take` of the same key on the same device and stream, or
until :func:`release`. Use it on that stream only (torch's caching allocator reuses freed memory in stream order),
never store it in an autograd context and never return it to a caller. The functions are thread-safe, but two threads
must not use one key on one stream at the same time: they would get the same buffer. :func:`release` drops the
buffers (miles calls it before it sleeps or offloads) and :func:`stats` reports what the pool holds.

``take`` runs on every call of an op, so its host work is small. With the bound at 0 and nothing held, it counts the
request and calls ``torch.empty``: the bound is read once per configuration and a device argument is resolved once
(``"cuda"``, the current device, is read at each call). With a positive bound it reads the current device, its current
stream and its capture state with one CUDA runtime query each, without switching devices when the device is the
current one, and each device keeps a running byte total, so a request that fits evicts nothing.

Importing this module does not import torch; :func:`release` and :func:`stats` never do.
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
# device -> the bytes of its buffers in _pools, kept as buffers come and go
_held: dict[Any, int] = {}
# device -> hits, misses, evictions
_counters: dict[Any, dict[str, int]] = {}
# guards _pools, _held and _counters, and every write of _direct
_lock = threading.Lock()

# the device argument of a take -> (device key, torch.device), or _CURRENT for a CUDA device without an index
_devices: dict[Any, Any] = {}
_CURRENT = object()
# CUDA index -> its torch.device, for _CURRENT
_cuda_devices: dict[int, Any] = {}
# (configuration, the bound it sets): config replaces its state whole, so the identity tells a new configuration
_bound: tuple[Any, int] = (None, 0)
# The configuration under which a take allocates directly: its bound is 0 and no device holds a buffer. Set under
# _lock when both hold; cleared when a buffer enters a pool (a take that read a positive bound before a configure).
_direct: Any = None
_torch: Any = None


def _import_torch() -> Any:
    global _torch
    import torch

    _torch = torch
    return torch


def _resolve(device: Any) -> tuple[Any, Any]:
    """``(device key, torch.device)`` of a device argument: the CUDA index (of the current device for ``"cuda"``), or
    the name of any other device (``"cpu"``, which only tests use).

    An argument is resolved once; ``"cuda"`` without an index stays the current device, read at every call.
    """
    resolved = _devices.get(device)
    if resolved is None:
        target = _torch.device(device)
        if target.type != "cuda":
            resolved = (str(target), target)
        elif target.index is None:
            resolved = _CURRENT
        else:
            resolved = (target.index, target)
        _devices[device] = resolved
    if resolved is _CURRENT:
        index = _torch._C._cuda_getDevice()
        target = _cuda_devices.get(index)
        if target is None:
            target = _cuda_devices[index] = _torch.device("cuda", index)
        return index, target
    return resolved


def _stream(where: Any) -> tuple[Any, bool]:
    """``(stream key, capturing)`` of the device key ``where``.

    For a CUDA index: the handle of the device's current stream (``torch.cuda.current_stream(where).cuda_stream``) and
    whether that stream captures a CUDA graph (the key is ``None`` then; a capture bypasses the pool), read with the
    runtime queries behind those functions, inside a device block only for a device that is not the current one. Any
    other device has one stream, ``None``, that never captures.
    """
    if type(where) is not int:
        return None, False
    cuda = _torch._C
    if cuda._cuda_getDevice() == where:
        if cuda._cuda_isCurrentStreamCapturing():
            return None, True
        return cuda._cuda_getCurrentRawStream(where), False
    with _torch.cuda.device(where):
        if cuda._cuda_isCurrentStreamCapturing():
            return None, True
        return cuda._cuda_getCurrentRawStream(where), False


def _device_state(where: Any) -> tuple[dict[str, int], OrderedDict]:
    """The counters and the pool of the device key ``where``, each created when missing, and its byte total,
    recomputed from the pool when missing (under _lock)."""
    counters = _counters.get(where)
    if counters is None:
        counters = _counters[where] = {"hits": 0, "misses": 0, "evictions": 0}
    pool = _pools.get(where)
    if pool is None:
        pool = _pools[where] = OrderedDict()
    if where not in _held:
        _held[where] = sum(buffer.numel() for buffer in pool.values())
    return counters, pool


def _evict(where: Any, counters: dict[str, int], room: int) -> None:
    """Drop the least recently used buffers of the pool of ``where`` until it holds at most ``room`` bytes."""
    pool = _pools[where]
    held = _held[where]
    while pool and held > room:
        _, evicted = pool.popitem(last=False)
        held -= evicted.numel()
        counters["evictions"] += 1
    _held[where] = held


def take(op: str, role: str, shape: Sequence[int], dtype: Any, device: Any) -> Any:
    """A tensor of ``shape`` and ``dtype`` on ``device``, a view of the pooled buffer of ``(op, role)``.

    Its content is undefined. Valid until the next ``take`` of the same key on the same device and stream, or
    :func:`release`. With the bound at 0 and nothing held, a plain ``torch.empty``.
    """
    if _direct is config._state:  # the bound is 0 and no device holds a buffer
        resolved = _devices.get(device)
        where, target = resolved if type(resolved) is tuple else _resolve(device)
        with _lock:
            counters = _counters.get(where)
            if counters is None:
                counters = _device_state(where)[0]
            counters["misses"] += 1
        return _torch.empty(shape, dtype=dtype, device=target)
    return _pooled(op, role, shape, dtype, device)


def _pooled(op: str, role: str, shape: Sequence[int], dtype: Any, device: Any) -> Any:
    """:func:`take` when the bound may be positive or a device may hold buffers."""
    global _bound, _direct
    torch = _torch or _import_torch()
    state = config._state
    read, limit = _bound
    if read is not state:
        limit = config.option(None, "cake_workspace_pool_bytes")
        _bound = (state, limit)
    resolved = _devices.get(device)
    where, target = resolved if type(resolved) is tuple else _resolve(device)
    if not limit:
        with _lock:
            counters = _device_state(where)[0]
            plain = not _held[where]
            if plain:  # nothing to keep and nothing to evict
                counters["misses"] += 1
                if not any(_held.values()):
                    _direct = state
        if plain:
            return torch.empty(shape, dtype=dtype, device=target)
    stream, capturing = _stream(where)
    if capturing:  # the pool is neither read nor changed
        with _lock:
            _device_state(where)[0]["misses"] += 1
        return torch.empty(shape, dtype=dtype, device=target)
    nbytes = math.prod(shape) * dtype.itemsize
    key = (op, role, stream)
    with _lock:
        counters, pool = _counters.get(where), _pools.get(where)
        if counters is None or pool is None or where not in _held:
            counters, pool = _device_state(where)
        buffer = pool.get(key)
        if buffer is not None:
            size = buffer.numel()
            if nbytes <= size <= limit:
                counters["hits"] += 1
                pool.move_to_end(key)  # the most recently used: evicted last, and never, as it fits the bound
            else:  # a buffer too small, or above a lowered bound, is dropped
                del pool[key]
                _held[where] -= size
                buffer = None
        if buffer is None:
            counters["misses"] += 1
            if not 0 < nbytes <= limit:
                if _held[where] > limit:
                    _evict(where, counters, limit)
                return torch.empty(shape, dtype=dtype, device=target)
            if _held[where] > limit - nbytes:
                _evict(where, counters, limit - nbytes)
            buffer = torch.empty(nbytes, dtype=torch.uint8, device=target)
            pool[key] = buffer  # the most recently used
            _held[where] += nbytes
            _direct = None
            size = nbytes
        elif _held[where] > limit:  # a hit after the bound was lowered
            _evict(where, counters, limit)
    view = buffer if nbytes == size else buffer[:nbytes]
    if dtype is not torch.uint8:
        view = view.view(dtype)
    return view.view(shape)  # a new tensor even for the whole uint8 buffer


def release(op: str | None = None) -> int:
    """Drop the pooled buffers of every op, or of ``op``, on every device and stream; returns the bytes dropped."""
    freed = 0
    with _lock:
        for where, pool in _pools.items():
            for key in [key for key in pool if op is None or key[0] == op]:
                freed += pool.pop(key).numel()
            _held[where] = sum(buffer.numel() for buffer in pool.values())
    return freed


def stats() -> dict[Any, dict[str, int]]:
    """``{device: {"bytes", "entries", "hits", "misses", "evictions"}}`` of the pool; ``device`` is the CUDA index."""
    with _lock:
        return {
            where: {
                "bytes": _held.get(where, 0),
                "entries": len(_pools.get(where, {})),
                **_counters.get(where, {"hits": 0, "misses": 0, "evictions": 0}),
            }
            for where in sorted({*_pools, *_counters}, key=str)
        }
