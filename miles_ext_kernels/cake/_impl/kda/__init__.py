"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of ``miles_ext_kernels.cake.kda``: the single-rank entry points, :func:`prebuild` and
:func:`release` (:mod:`.ops`) and the context-parallel entry points (:mod:`.kcp`).
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("kda")  # the loader on every package of OP.packages, before any of them can build

from .kcp import (  # noqa: E402
    build_context_kcp,
    chunk_kda_bwd_kcp,
    chunk_kda_fwd_kcp,
    chunk_kda_kcp,
    partition_kcp,
    reduce_gate_param_grads_kcp,
)
from .ops import DEFAULT_SCALE, MAX_HEADS, chunk_kda, chunk_kda_bwd, chunk_kda_fwd, prebuild, release  # noqa: E402

__all__ = [
    "DEFAULT_SCALE",
    "MAX_HEADS",
    "build_context_kcp",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_bwd_kcp",
    "chunk_kda_fwd",
    "chunk_kda_fwd_kcp",
    "chunk_kda_kcp",
    "partition_kcp",
    "prebuild",
    "reduce_gate_param_grads_kcp",
    "release",
]
