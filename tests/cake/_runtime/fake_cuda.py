"""A fake CUDA device layer, so the workspace pool's CUDA path runs on the CPU.

:class:`FakeCuda` stands in for what the pool reads of the CUDA runtime: ``torch.cuda.current_device``,
``torch.cuda.current_stream``, ``torch.cuda.is_current_stream_capturing`` and ``torch.cuda.device``; its ``empty``
allocates a CUDA request on the CPU and records the device it was asked for. :meth:`FakeCuda.patches` lists the
``(object, attribute, value)`` to set on torch. Importing this module does not import torch.
"""

from typing import Any


class FakeStream:
    """A stream: only its handle, ``cuda_stream``."""

    __slots__ = ("cuda_stream",)

    def __init__(self, handle: int) -> None:
        self.cuda_stream = handle


class _Switch:
    """``torch.cuda.device(index)``: the current device is ``index`` inside the block."""

    __slots__ = ("cuda", "index", "previous")

    def __init__(self, cuda: "FakeCuda", index: int) -> None:
        self.cuda, self.index = cuda, index

    def __enter__(self) -> None:
        self.previous, self.cuda.index = self.cuda.index, self.index

    def __exit__(self, *exc: Any) -> None:
        self.cuda.index = self.previous


class FakeCuda:
    """``devices`` CUDA devices; device ``i`` starts on stream ``1000 + i`` and nothing captures.

    Set ``index`` (the current device), ``streams[i]`` (device ``i``'s current stream) and ``capturing`` (the devices
    whose current stream captures a graph).
    """

    def __init__(self, torch: Any, devices: int = 2) -> None:
        self.torch = torch
        self.index = 0
        self.streams = {index: FakeStream(1000 + index) for index in range(devices)}
        self.capturing: set[int] = set()
        self.allocated: list[Any] = []  # the device argument of every CUDA empty
        self._empty = torch.empty

    def _index(self, device: Any) -> int:
        if isinstance(device, int):
            return device
        index = self.torch.device(device).index
        return self.index if index is None else index

    def current_device(self) -> int:
        return self.index

    def current_stream(self, device: Any = None) -> FakeStream:
        return self.streams[self.index if device is None else self._index(device)]

    def is_current_stream_capturing(self) -> bool:
        return self.index in self.capturing

    def device(self, device: Any) -> _Switch:
        return _Switch(self, self._index(device))

    def empty(self, *size: Any, device: Any = None, **kwargs: Any) -> Any:
        if device is not None and self.torch.device(device).type == "cuda":
            self.allocated.append(device)
            device = "cpu"
        return self._empty(*size, device=device, **kwargs)

    def patches(self) -> list[tuple[Any, str, Any]]:
        torch = self.torch
        return [
            (torch.cuda, "current_device", self.current_device),
            (torch.cuda, "current_stream", self.current_stream),
            (torch.cuda, "is_current_stream_capturing", self.is_current_stream_capturing),
            (torch.cuda, "device", self.device),
            (torch, "empty", self.empty),
        ]
