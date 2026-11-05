"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of :mod:`miles_ext_kernels.cake.dense_gemm` (imports torch).
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("dense_gemm")

from .ops import (  # noqa: E402
    OPERATIONS,
    WEIGHT_LAYOUTS,
    batched_projection,
    check_projection_dims,
    fast_projection_programs,
    fast_router_programs,
    prebuild,
    release,
    router_gating,
)

__all__ = [
    "OPERATIONS",
    "WEIGHT_LAYOUTS",
    "batched_projection",
    "check_projection_dims",
    "fast_projection_programs",
    "fast_router_programs",
    "prebuild",
    "release",
    "router_gating",
]
