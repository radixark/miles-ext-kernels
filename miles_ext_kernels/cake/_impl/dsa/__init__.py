"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The implementation of ``miles_ext_kernels.cake.dsa``: the operators (:mod:`.ops`), the deterministic indexer top-k
(:mod:`.indexer_topk`), the det kernels' host side (:mod:`.kernels`, :mod:`.native`, :mod:`.reduction`) and the layout
adapters (:mod:`.layouts`).
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("dsa")

from .indexer_topk import (  # noqa: E402
    dsa_indexer_det_topk,
    dsa_indexer_det_topk_workspace_size,
    indexer_det_topk_unserved_reason,
    packed_key_ids,
)
from .ops import (  # noqa: E402
    check_device,
    extract_topk_scores,
    fast_unserved_reason,
    indexer_backward,
    indexer_logits,
    indexer_topk_scores,
    indexer_unserved_reason,
    lightning_indexer,
    prebuild,
    release,
    select_topk,
    sparse_attention,
    sparse_attention_backward,
    sparse_attention_forward,
    torch_topk,
)

__all__ = [
    "check_device",
    "dsa_indexer_det_topk",
    "dsa_indexer_det_topk_workspace_size",
    "extract_topk_scores",
    "fast_unserved_reason",
    "indexer_backward",
    "indexer_det_topk_unserved_reason",
    "indexer_logits",
    "indexer_topk_scores",
    "indexer_unserved_reason",
    "lightning_indexer",
    "packed_key_ids",
    "prebuild",
    "release",
    "select_topk",
    "sparse_attention",
    "sparse_attention_backward",
    "sparse_attention_forward",
    "torch_topk",
]
