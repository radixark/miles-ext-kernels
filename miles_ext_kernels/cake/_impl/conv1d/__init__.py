"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of :mod:`miles_ext_kernels.cake.conv1d`: the entry points, ``prebuild`` and ``release`` of
:mod:`.ops`, on the two vendored packages, which build their libraries through the runtime's loader (installed here,
before any call).
"""

from miles_ext_kernels.cake._runtime import loader

from .ops import causal_conv1d, causal_conv1d_kcp, causal_conv1d_kcp_multi, check_supported, prebuild, release

loader.install_op("conv1d")

__all__ = [
    "causal_conv1d",
    "causal_conv1d_kcp",
    "causal_conv1d_kcp_multi",
    "check_supported",
    "prebuild",
    "release",
]
