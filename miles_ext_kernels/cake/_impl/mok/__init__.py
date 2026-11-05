"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of ``miles_ext_kernels.cake.mok``: the MoK adapter, its workspace and weight quantizer
(:mod:`.ops`), and the op's ``prebuild`` and ``release``.
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("mok")  # the vendored package builds its libraries through the runtime's loader

from .ops import (  # noqa: E402
    create_mok_bf16_workspace,
    prebuild,
    prepare_mok_bf16,
    quantize_mok_mxfp8_weights,
    release,
)

__all__ = [
    "create_mok_bf16_workspace",
    "prebuild",
    "prepare_mok_bf16",
    "quantize_mok_mxfp8_weights",
    "release",
]
