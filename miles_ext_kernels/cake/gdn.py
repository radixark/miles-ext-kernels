"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Chunked gated delta rule (GDN) training kernels, forward and backward (K = V = 128), with packed sequences and context
parallelism, for SM100a / SM103a / SM107a.

:func:`chunk_gated_delta_rule` has the signature and autograd contract of FLA's ``chunk_gated_delta_rule`` (v0.5.2).
A call runs, by the configured variant and its ``cp_context``:

- no ``cp_context``: ``det`` the FlashInfer Cake det package, ``fast`` the non-deterministic one;
- a :class:`KcpContext` (:func:`build_kcp_context`, the packages' own context parallelism): the det or the
  non-deterministic KCP package (:func:`kcp_chunk_gated_delta_rule` calls it directly);
- an FLA ``cp_context``, or a ``KcpContext`` call the KCP packages do not serve: the det kernels with FLA's context
  parallelism under either variant (``fast`` logs it once).

Under any ``cp_context`` the contract is FLA's: no initial state, no final state, and the context's boundaries.

Determinism: ``det`` repeats bit for bit for identical inputs (the whole packed micro-batch: its launch plan is a pure
function of every sequence length and the device) on the same GPU model and SM count, toolkit and build, and on the
same route: without context parallelism, with a ``KcpContext`` or with an FLA context, the same partition, and the
same boundary source (``cu_seqlens_cpu`` on the host or ``cu_seqlens`` on the device). The plans of two such settings
may differ (another SM count or route can split the state scan differently); their results then agree within the
accuracy budget, not bit for bit. ``--cake-gdn-recompute-state`` is part of the setting.

miles selects the op with ``--linear-attention-backend cake``; ``--cake-gdn-recompute-state`` recomputes the WY and
state tensors in the backward instead of keeping them. Importing this module imports neither torch nor the kernels.
"""

from miles_ext_kernels.cake._runtime.spec import Flag, OpSpec, lazy_exports

OP = OpSpec(
    name="gdn",
    summary="chunked gated delta rule, forward + backward, K = V = 128, packed sequences, context parallelism",
    variants=("det", "fast"),
    selected_by=(("linear_attention_backend", "cake"),),
    packages=(
        "cake_gdn_det_train",
        "cake_gdn_det_train_kcp",
        "cake_gdn_fla_cp_det",
        "cake_gdn_nondet_train",
        "cake_gdn_nondet_train_kcp",
    ),
    exports=("KcpContext", "build_kcp_context", "chunk_gated_delta_rule", "kcp_chunk_gated_delta_rule"),
    flags=(
        Flag(
            "--cake-gdn-recompute-state",
            dict(
                action="store_true",
                default=False,
                help=(
                    "Recompute w/u/h/v_new in the backward instead of keeping them on the autograd tape (under KCP "
                    "context parallelism: the packages' light saved set instead of the full one): less activation "
                    "memory, more backward time. Without context parallelism, det and fast give gradients within the "
                    "BF16 accuracy budgets of the saved backward; with an FLA context they give the same gradient "
                    "bits, and with a KCP context det gives the same bits and fast stays within the budgets. Under "
                    "det each setting repeats bit for bit."
                ),
            ),
        ),
    ),
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
