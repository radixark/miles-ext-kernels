"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

MoK (Mixture of Kittens) expert-parallel MoE training layer for SM100a / SM103a / SM107a.

One MoE layer of routed experts in BF16 or MXFP8 and one BF16 shared expert, each a SwiGLU (or clamped SwiGLU) MLP,
with the token dispatch to the experts' ranks, the expert GEMMs, the combine and the backward (input, router-score and
all six weight gradients) fused into kernels that exchange rows over PyTorch symmetric memory with multicast, so every
expert-parallel rank sits in one NVLink domain. One variant, ``det``: the outputs are a function of the inputs only, so
they repeat bit for bit run to run.

No miles argument selects it. Importing this module imports neither torch nor the implementation: the entry points
load ``miles_ext_kernels.cake._impl.mok`` on first access.
"""

from miles_ext_kernels.cake._runtime.spec import OpSpec, lazy_exports

OP = OpSpec(
    name="mok",
    summary=(
        "expert-parallel MoE layer (dispatch, BF16 or MXFP8 routed and BF16 shared SwiGLU experts, combine), forward, "
        "context recompute and backward, one NVLink domain"
    ),
    variants=("det",),
    packages=("cake_mok_bf16",),
    exports=("create_mok_bf16_workspace", "prepare_mok_bf16", "quantize_mok_mxfp8_weights"),
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
