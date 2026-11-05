"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

DSA (DeepSeek Sparse Attention) training kernels, forward and backward, on SM100a / SM103a / SM107a: the sparse MLA /
MQA attention over top-k key indices and the lightning indexer, for the GLM-5 / DeepSeek-V3.2 packed layout (``thd``,
absorbed MLA, ``d_qk`` 576 with a RoPE tail or 512 without, no sink) and the DeepSeek-V4 batched layout (``bshd``, MQA,
FP32 attention sink).

- ``det``, the DSA det version (``cake_dsa_sparse_det``): no floating-point atomics; the key-side gradients are reduced
  from FP32 partials in an order fixed by the indices (a 4-way tree per key), so identical inputs give identical bits.
  Every head count, with or without a sink. The indexer scores and their backward are always ``det``.
- ``fast`` (``cake_dsa_train``): 64 local query heads, ``d_v`` 512 with a 64-channel RoPE tail or none, no sink; ``dq``
  repeats bit for bit, ``dkv`` accumulates in FP32 with ``red.global`` atomics. Any other call runs ``det`` and logs it
  once per reason.
- :func:`dsa_indexer_det_topk` (``cake_dsa_indexer_det``): fused indexer scores and the exact top-k of packed segments
  for 32 index heads of dimension 128, bit for bit run to run.

miles selects the op with ``--dsa-attention-backend cake``. ``prebuild("dsa", heads=..., d_qk=..., d_v=...,
attn_sink=..., index_heads=...)`` builds every library a layer of that geometry may launch.
"""

from argparse import Namespace

from miles_ext_kernels.cake._runtime.spec import Flag, OpSpec, lazy_exports


def _validate(args: Namespace) -> None:
    budget = getattr(args, "cake_dsa_bwd_workspace_bytes", None)
    if budget is not None and budget <= 0:
        raise ValueError(f"--cake-dsa-bwd-workspace-bytes must be positive, got {budget}")


OP = OpSpec(
    name="dsa",
    summary="DeepSeek sparse attention and lightning indexer, forward + backward",
    variants=("det", "fast"),
    selected_by=(("dsa_attention_backend", "cake"),),
    packages=("cake_dsa_indexer_det", "cake_dsa_sparse_det", "cake_dsa_train"),
    exports=(
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
        "select_topk",
        "sparse_attention",
        "sparse_attention_backward",
        "sparse_attention_forward",
        "torch_topk",
    ),
    flags=(
        Flag(
            "--cake-dsa-bwd-workspace-bytes",
            dict(
                type=int,
                default=2 << 30,
                help=(
                    "Bound of the transient memory of the det backward (sparse attention and indexer): its FP32 "
                    "key-side partials and the index scratch of their fixed-order reduction. Query rows run in chunks "
                    "that fit; the chunking, and with it the det bits, follows from this value."
                ),
            ),
        ),
        Flag(
            "--cake-dsa-lazy-build",
            dict(
                action="store_true",
                default=False,
                help=(
                    "Build each DSA library on its first launch: prebuild('dsa', ...) and the first call build "
                    "nothing ahead. Gives up the prebuild guarantee: a library first used mid-run compiles inside a "
                    "training step."
                ),
            ),
        ),
    ),
    validate=_validate,
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
