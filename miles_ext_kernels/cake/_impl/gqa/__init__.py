"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of ``miles_ext_kernels.cake.gqa`` on its vendored package ``cake_gqa_train`` (:mod:`.ops`),
imported on first use of an entry point. Importing it imports torch.
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("gqa")  # the runtime loader on the op's package, before it can build

from miles_ext_kernels.cake.gqa import check_supported, is_supported  # noqa: E402

from .ops import prebuild, release, varlen_gqa_attention  # noqa: E402

__all__ = ["check_supported", "is_supported", "prebuild", "release", "varlen_gqa_attention"]
