"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Packed causal GQA attention training kernels, head dimension 256, forward and backward (SM100a / SM103a / SM107a).

``varlen_gqa_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, *, max_seqlen_k, softmax_scale=None)`` takes BF16
``q [T_q, H_q, 256]`` and ``k`` / ``v [T_k, H_kv, 256]`` with an even number of query heads per key/value head,
segments packed by int32 device ``cu_seqlens_q`` / ``cu_seqlens_k``, causal with FlashAttention's bottom-right
alignment per segment. ``max_seqlen_k`` is a hard contract: a host int at least every segment's key count. It sizes
the backward grid, so a smaller value leaves key blocks out of the backward and gives wrong dQ / dK / dV without an
error (the output is unaffected). ``--cake-gqa-check-seqlens`` checks it, with the segment boundaries, at every call.

One variant, ``fast``: no sum uses atomics, so the output, the LSE, dQ and dK / dV repeat bit for bit run to run for
one shape and one ``max_seqlen_k``, and ``torch.use_deterministic_algorithms(True)`` raises no alert. There is no
``det`` variant, so ``--deterministic-mode`` rejects the op.
:func:`check_supported` and :func:`is_supported` tell, without torch, whether the kernels run a head layout; miles'
argument validation applies them to the heads one tensor-parallel rank holds.

miles selects the kernels with ``--full-attention-backend cake``. Importing this module imports neither torch nor the
implementation.
"""

from argparse import Namespace

from miles_ext_kernels.cake._runtime.spec import Flag, OpSpec, lazy_exports

HEAD_DIM = 256


def check_supported(num_heads_q: int, num_heads_kv: int, head_dim: int = HEAD_DIM) -> None:
    """Raise ``ValueError`` naming what the kernels do not run, for the heads one call takes.

    The kernels take head dimension 256 and an even number of query heads per key/value head. Torch-free.
    """
    errors = []
    if head_dim != HEAD_DIM:
        errors.append(f"head dim {head_dim} (they take {HEAD_DIM})")
    if num_heads_q < 1 or num_heads_kv < 1 or num_heads_q % num_heads_kv or (num_heads_q // num_heads_kv) % 2:
        errors.append(
            f"{num_heads_q} query heads over {num_heads_kv} key/value heads (they need an even number of query heads "
            "per key/value head)"
        )
    if errors:
        raise ValueError("the cake gqa kernels do not run " + "; ".join(errors))


def is_supported(num_heads_q: int, num_heads_kv: int, head_dim: int = HEAD_DIM) -> bool:
    """Do the kernels run these heads (:func:`check_supported` without the reason)?"""
    try:
        check_supported(num_heads_q, num_heads_kv, head_dim)
    except ValueError:
        return False
    return True


def _validate(args: Namespace) -> None:
    """The heads of one tensor-parallel rank, from Megatron's arguments, must be heads the kernels run.

    A rank holds ``num_attention_heads / TP`` query heads and ``max(num_query_groups, TP) / TP`` key/value heads (the
    key/value heads are replicated when there are fewer than TP). Without ``--kv-channels`` the head dimension is
    Megatron's default, ``hidden_size / num_attention_heads``; with neither it is not checked here.
    """
    heads = getattr(args, "num_attention_heads", None)
    if heads is None:
        return
    tp = getattr(args, "tensor_model_parallel_size", None) or 1
    groups = getattr(args, "num_query_groups", None) if getattr(args, "group_query_attention", False) else None
    groups = groups or heads
    head_dim = getattr(args, "kv_channels", None)
    if head_dim is None and getattr(args, "hidden_size", None):
        head_dim = args.hidden_size // heads
    try:
        check_supported(heads // tp, max(groups, tp) // tp, HEAD_DIM if head_dim is None else head_dim)
    except ValueError as exc:
        raise ValueError(
            f"--full-attention-backend cake with --num-attention-heads {heads}, {groups} query groups and "
            f"--tensor-model-parallel-size {tp}: {exc}"
        ) from None


OP = OpSpec(
    name="gqa",
    summary="packed causal GQA attention, head dim 256, forward + backward",
    variants=("fast",),
    selected_by=(("full_attention_backend", "cake"),),
    packages=("cake_gqa_nondet_train",),
    exports=("check_supported", "is_supported", "varlen_gqa_attention"),
    flags=(
        Flag(
            "--cake-gqa-check-seqlens",
            dict(
                action="store_true",
                default=False,
                help=(
                    "Check at every call of the cake gqa kernels that cu_seqlens_q / cu_seqlens_k are well formed and "
                    "that max_seqlen_k is at least every segment's key count (a smaller one gives wrong gradients "
                    "without an error). It reads the boundaries on the host: one device synchronization per call."
                ),
            ),
        ),
    ),
    validate=_validate,
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
