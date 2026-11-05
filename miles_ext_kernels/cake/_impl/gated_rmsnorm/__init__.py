"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of ``miles_ext_kernels.cake.gated_rmsnorm`` (:mod:`.ops`).
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("gated_rmsnorm")  # the runtime loader on both packages' cake_build

from .ops import check_supported, prebuild, release, rmsnorm_silu_gate  # noqa: E402

__all__ = ["check_supported", "prebuild", "release", "rmsnorm_silu_gate"]
