"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

SiLU-gated RMSNorm training kernels, forward and backward: ``RMSNorm(x) * weight * SiLU(z)`` over rows of 128, the
output norm of miles' GDN linear-attention layers.

``det`` sums the weight gradient in a fixed order (bit for bit reproducible); ``fast`` adds its chunk sums with FP64
atomics (``y``, ``dx`` and ``dz`` are ``det``'s bits, the weight gradient within ``det``'s error bound). miles selects
the op with ``--linear-attention-output-norm-backend cake``. Importing this module imports neither torch nor the
implementation (``miles_ext_kernels.cake._impl.gated_rmsnorm``): the entry points load on first access.
"""

from miles_ext_kernels.cake._runtime.spec import OpSpec, lazy_exports

OP = OpSpec(
    name="gated_rmsnorm",
    summary="SiLU-gated RMSNorm, forward + backward, rows of 128, BF16",
    variants=("det", "fast"),
    selected_by=(("linear_attention_output_norm_backend", "cake"),),
    packages=("cake_rmsnorm_silu_gate_det_train", "cake_rmsnorm_silu_gate_nondet_train"),
    exports=("check_supported", "rmsnorm_silu_gate"),
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
