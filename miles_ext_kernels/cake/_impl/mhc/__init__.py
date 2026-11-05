"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of :mod:`miles_ext_kernels.cake.mhc` over its three vendored packages (:mod:`.ops`).
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("mhc")

from .ops import (  # noqa: E402
    HIDDEN_SIZES,
    STREAMS,
    is_supported,
    mhc_finish,
    mhc_head,
    mhc_prepare,
    prebuild,
    release,
    supported_hidden_sizes,
)

__all__ = [
    "HIDDEN_SIZES",
    "STREAMS",
    "is_supported",
    "mhc_finish",
    "mhc_head",
    "mhc_prepare",
    "prebuild",
    "release",
    "supported_hidden_sizes",
]
