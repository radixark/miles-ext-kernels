"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The gdn implementation: the entry points of ``miles_ext_kernels.cake.gdn`` on its five vendored packages
(:mod:`.ops`), the KCP context (:mod:`.kcp`) and the launchers of the generated FLA-CP det kernels (:mod:`.native`).
Importing it imports torch.
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("gdn")  # the runtime loader on every package of the op, before any of them can build

from .kcp import KcpContext, build_kcp_context  # noqa: E402
from .ops import chunk_gated_delta_rule, kcp_chunk_gated_delta_rule, prebuild, release  # noqa: E402

__all__ = [
    "KcpContext",
    "build_kcp_context",
    "chunk_gated_delta_rule",
    "kcp_chunk_gated_delta_rule",
    "prebuild",
    "release",
]
