"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of ``miles_ext_kernels.cake.gated_act_mxfp8``: the entry points, ``prebuild`` and ``release`` of
:mod:`.ops`.
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("gated_act_mxfp8")

from .ops import prebuild, release, silu_and_mul_mxfp8_quantize, silu_and_mul_mxfp8_quantize_backward  # noqa: E402

__all__ = ["prebuild", "release", "silu_and_mul_mxfp8_quantize", "silu_and_mul_mxfp8_quantize_backward"]
