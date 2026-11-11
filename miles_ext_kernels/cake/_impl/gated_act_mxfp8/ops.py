"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Fused SwiGLU + MXFP8 quantization, forward and backward, on the vendored package ``cake_gated_act_mxfp8_det_train``
(SM100a / SM103a / SM107a).

``gated_input [M, 2K]`` holds the gate columns ``[0, K)`` and the up columns ``[K, 2K)``: BF16, contiguous, 32-byte
aligned, ``M`` and ``K`` positive multiples of 128 (an empty batch, ``M = 0``, raises ``ValueError`` too),
``M * 2K <= 2^31``. The forward quantizes the SwiGLU output ``[M, K]``; the backward quantizes the gate and up
gradients ``[M, 2K]`` (columns ``[0, K)`` the gate gradient, ``[K, 2K)`` the up gradient) of ``grad_output [M, K]``
(BF16, contiguous, aligned), with no BF16 intermediate in global memory. Each step is one FP32 operation, ``2^t`` the
hardware ``ex2.approx``::

    s     = 1 / (1 + 2^(-gate * log2(e)))
    act   = gate * s
    out   = bf16(act * up)                              # forward
    dgate = bf16((fma(act, 1 - s, s) * grad) * up)      # backward
    dup   = bf16(act * grad)

Each BF16 value is quantized per block of 32 elements of a row (``rowwise``) and / or of a column (``colwise``):
the E8M0 scale is RCEIL (the exponent of the block's absolute maximum, plus one when its mantissa is above 1.75, less
8, clamped at 0; 255 for an Inf or NaN maximum, whose codes are all NaN) and a code is the value times
``2^(127 - scale)`` rounded to E4M3, to nearest even. A call returns ``(row_output, col_output, row_scales,
col_scales)``: ``row_output`` E4M3 ``[M, N]`` row-major, ``col_output`` E4M3 ``[M, N]`` column-major (stride
``(1, M)``), the E8M0 scales in the swizzled 128 x 4 tile layout of the MXFP8 GEMM operands
(``row_scales [M, N / 32]``, ``col_scales`` flat ``[N * M / 32]``), ``N`` = ``K`` forward and ``2K`` backward; a
disabled orientation's pair is zero-size on the input device. At least one of ``rowwise`` / ``colwise``; an input the
kernels do not take (device, dtype, shape, strides, alignment) raises ``ValueError`` or ``TypeError``, another GPU
``RuntimeError``.

The entry points are the package's own (``silu_and_mul_mxfp8_quantize`` and its backward are its
``cake_gated_act_mxfp8_det_train_forward`` and ``_backward``): it is vendored byte-identical and builds its libraries
through :mod:`miles_ext_kernels.cake._runtime.loader` (nvcc and tvm-ffi, no FlashInfer). A call runs one program of its
route (the direction and the output set), chosen by ``M * K`` alone, and every program of a route writes the same bytes:
the outputs are a function of the inputs only, so they repeat bit for bit run to run (the op's one variant, ``det``).
``M`` and ``K`` are runtime arguments, so a new shape never builds a library once :func:`prebuild` has built the
architecture's. The host reads no tensor and never synchronizes the device; the kernels run on the current torch stream
and write fresh output tensors (no workspace), so the calls can be captured in a CUDA graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miles_ext_kernels.cake._runtime import loader
from miles_ext_kernels.cake._vendor import cake_gated_act_mxfp8_det_train as backend
from miles_ext_kernels.cake._vendor.cake_gated_act_mxfp8_det_train import (
    cake_gated_act_mxfp8_det_train_backward as silu_and_mul_mxfp8_quantize_backward,
    cake_gated_act_mxfp8_det_train_forward as silu_and_mul_mxfp8_quantize,
)

__all__ = ["libraries", "prebuild", "release", "silu_and_mul_mxfp8_quantize", "silu_and_mul_mxfp8_quantize_backward"]


def libraries(arch: str) -> list[str]:
    """The ``build.json`` libraries of every program a call on ``arch`` may run: the six routes (forward / backward x
    rowwise / colwise / both), each with every program its ``M * K`` selection rule picks."""
    jit = backend.cake_jit
    return sorted({jit.library_name(name, arch) for kind in jit.KINDS for name in jit.route_modules(arch, kind)})


def prebuild(*, device: Any = None, max_workers: int | None = None) -> list[Path]:
    """Build, or find in the cache, every library of the architecture of ``device`` (default: the current one).

    No model parameter: ``M`` and ``K`` are runtime arguments, so one set of libraries serves every shape and route.
    ``NotImplementedError`` on a device that is not SM100a / SM103a / SM107a, ``RuntimeError`` when the toolkit cannot
    build its architecture (it needs CUDA 13.4 or newer). Returns the cache path of every library.
    """
    arch = loader.device_arch(device)
    return loader.prebuild_libraries(backend.cake_build, libraries(arch), max_workers=max_workers)


def release() -> None:
    """Nothing to free: the op keeps no workspace, plan or graph, and the libraries it loaded stay loaded (a captured
    CUDA graph may still launch them)."""
