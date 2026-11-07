"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Zero-centred RMSNorm training kernels, forward and backward: ``y = bf16(x * rstd * (1 + w))`` over the last dimension
of BF16 ``x``, the input layernorm of miles' GDN linear-attention layers (``--linear-attention-input-norm cake``).

:class:`ZeroCenteredRMSNorm` holds the single ``weight`` parameter, ``w`` of the ``(1 + w)`` scale, as Transformer
Engine's ``zero_centered_gamma`` norm and HF's ``Qwen3NextRMSNorm`` hold it, and runs FlashInfer's Cake RMSNorm
packages on the FP32 effective weight ``1 + w``. ``det`` reduces the weight gradient in a fixed order; ``fast`` sums
it with FP64 atomics in the token-count windows that ship that backward. The implementation reads the variant at every
call; ``miles_ext_kernels.cake.prebuild("rmsnorm", hidden_size=H)`` builds every library either variant may launch
for ``H``, and building the module compiles nothing. On SM107a, ``H = 4096`` launches programs in CUDA 13.4's oversized
shared-memory mode, which needs a CUDA 13.4 or newer runtime in the process; without one the op raises
``RuntimeError`` for that width before any launch.
"""

from miles_ext_kernels.cake._runtime.spec import OpSpec, lazy_exports

OP = OpSpec(
    name="rmsnorm",
    summary="zero-centred RMSNorm x * rstd * (1 + w), forward + backward, BF16 activations",
    variants=("det", "fast"),
    selected_by=(("linear_attention_input_norm", "cake"),),
    packages=("cake_rmsnorm_det_train", "cake_rmsnorm_nondet_train"),
    exports=("ZeroCenteredRMSNorm", "check_supported", "supported_hidden_sizes", "zero_centered_rmsnorm"),
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
