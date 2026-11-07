"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of ``miles_ext_kernels.cake.rmsnorm``: its entry points, ``prebuild`` and ``release``.
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("rmsnorm")  # both packages build through the runtime's loader; each variant runs its own package

from .ops import (  # noqa: E402
    ZeroCenteredRMSNorm,
    check_supported,
    prebuild,
    release,
    supported_hidden_sizes,
    zero_centered_rmsnorm,
)

__all__ = [
    "ZeroCenteredRMSNorm",
    "check_supported",
    "prebuild",
    "release",
    "supported_hidden_sizes",
    "zero_centered_rmsnorm",
]
