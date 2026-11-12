"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Fused SwiGLU + MXFP8 quantization, forward and backward, on SM100a / SM103a / SM107a.

The forward quantizes ``silu(gate) * up`` of a BF16 gate / up input ``[M, 2K]`` (``M`` and ``K`` positive multiples
of 128), the backward the gate and up gradients ``[M, 2K]`` of a BF16 ``grad_output [M, K]``: E4M3 values with RCEIL
E8M0 scales per 32 elements of a row and / or of a column, the scales in the swizzled 128 x 4 layout of the MXFP8 GEMM
operands. It is the activation of a SwiGLU MLP trained with MXFP8 GEMMs: the forward output is the down projection's
input, the backward output the gradient of the gate / up projection's output.

One variant, ``det``: a call runs one program chosen by ``M * K``, and every program of a route writes the same bytes,
so the outputs are a function of the inputs and repeat bit for bit. No miles argument selects the op; it has no
option. ``prebuild`` takes no model parameter: it builds every library of the device's architecture.
"""

from miles_ext_kernels.cake._runtime.spec import OpSpec, lazy_exports

OP = OpSpec(
    name="gated_act_mxfp8",
    summary="fused SwiGLU + MXFP8 quantization, forward + backward, BF16 [M, 2K] gate / up input",
    variants=("det",),
    packages=("cake_gated_act_mxfp8_det_train",),
    exports=("silu_and_mul_mxfp8_quantize", "silu_and_mul_mxfp8_quantize_backward"),
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
