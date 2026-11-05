"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Packed KDA (Kimi Delta Attention) chunked training kernels, forward and backward, for SM100a / SM103a / SM107a.

The chunked KDA recurrence (chunk 64, K = V = 128, 1 to 64 heads per call, per-document zero initial state) over one
packed batch of documents described by ``cu_seqlens``, from FlashInfer's Cake packages: the forget gate activated in
the kernels or by the caller (``use_gate_in_kernel``), the q / k L2 normalization in the kernels or by the caller
(``use_qk_l2norm_in_kernel``), and context-parallel entry points (``*_kcp``) that split one packed batch over the
ranks of a process group. Two variants, read when a call runs:

* ``det`` (``cake_kda_det_train``, ``cake_kda_det_train_kcp``): fixed-order reductions, no atomics; the same inputs
  give the same bits.
* ``fast`` (``cake_kda_train``, ``cake_kda_train_kcp``, FlashInfer's ``deterministic=False``): with the activated
  gate its backward runs other gate-gradient programs than ``det`` on some rows; with the raw gate its parameter
  gradients are reduced in FP64 in no fixed order on SM100a / SM103a. It has no in-kernel q / k L2 norm: such a call
  runs ``det`` and says so once.

miles selects the op with ``--kda-backend cake`` (under ``--use-miles-ext-kernels``), builds the libraries when a
layer is built (``miles_ext_kernels.cake.prebuild("kda", num_heads=...)``) and calls :func:`chunk_kda` in its
forward. Every miles call takes the in-kernel q / k L2 norm, which only ``det`` serves, so miles may choose ``det``
only: ``--cake-kda-variant`` is hidden and carries ``det`` to every process (a ``fast`` choice would change no
program miles runs). Direct callers choose either variant with ``miles_ext_kernels.configure(cake_kda_variant=...)``.

Importing this module imports neither torch nor the implementation; the entry points load on first access.
"""

from argparse import Namespace

from miles_ext_kernels.cake._runtime.spec import OpSpec, lazy_exports


def _validate(args: Namespace) -> None:
    """Under ``--kda-backend cake``: reject context parallelism and a layout other than ``thd``.

    The kernels run whole documents of one packed sequence on one rank. Their context-parallel entry points take a
    partition of the packed batch over the ranks, which miles' context-parallel KDA path does not build: it hands
    each rank a shard of every document with flash-linear-attention's context. ``bshd`` is not a packed sequence.
    """
    context_parallel_size = getattr(args, "context_parallel_size", None) or 1
    if context_parallel_size > 1:
        raise ValueError(
            f"--kda-backend cake does not support context parallelism yet (--context-parallel-size "
            f"{context_parallel_size}); use --kda-backend fla"
        )
    qkv_format = getattr(args, "qkv_format", "thd")
    if qkv_format != "thd":
        raise ValueError(f"--kda-backend cake needs --qkv-format thd (one packed sequence), got {qkv_format!r}")


OP = OpSpec(
    name="kda",
    summary="packed KDA chunked training, forward + backward, K = V = 128, 1 to 64 heads, context parallelism",
    variants=("det", "fast"),
    # miles' calls take the in-kernel q / k L2 norm, which fast does not serve: fast would run det's programs
    cli_variants=("det",),
    selected_by=(("kda_backend", "cake"),),
    packages=("cake_kda_det_train", "cake_kda_det_train_kcp", "cake_kda_train", "cake_kda_train_kcp"),
    exports=(
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
        "reduce_gate_param_grads_kcp",
    ),
    validate=_validate,
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
