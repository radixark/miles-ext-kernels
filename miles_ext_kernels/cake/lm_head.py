"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Chunked LM-head selected-token log-probabilities, forward and backward, on SM100a / SM103a / SM107a.

:func:`log_probs` takes the hidden rows ``X [T, H]``, the output weight ``W [V, H]`` (both BF16) and the target ids
``labels [T]`` and returns the FP32 ``logp [T]`` of ``softmax(X @ W^T / temperature)`` at the labels, differentiable
in ``X`` and ``W``. No ``[T, V]`` logits tensor exists beyond one token chunk (``--cake-lm-head-chunk-size`` rows): the
backward recomputes each chunk's logits. ``det`` repeats bit for bit; ``fast`` sums the K slices of the hidden-state
gradient in an unordered FP32 reduction and skips the GEMM work of rows whose incoming gradient is zero.

miles selects it with ``--log-probs-backend cake``. The kernels serve ``H`` a multiple of 512 and ``V`` a multiple of
256 (their dX / dW and logits tiles): a vocabulary that is not, such as 151,936, is rejected at startup, since padding
it would change the softmax.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterator

from miles_ext_kernels.cake._runtime.spec import Flag, OpSpec, lazy_exports

BACKEND = "--log-probs-backend cake"
CHUNK_SIZE_FLAG = "--cake-lm-head-chunk-size"
DEFAULT_CHUNK_SIZE = 4096
# the row kernels index a chunk's row by blockIdx.y
MAX_CHUNK_SIZE = 65535
# the losses whose scores are the log-probabilities of the target tokens alone
LOSS_TYPES = ("policy_loss", "sft_loss")
# the GEMM work items: V in 256-column logits tiles, H in 512-column dX / dW tiles
VOCAB_MULTIPLE = 256
HIDDEN_MULTIPLE = 512
# ``--target-modules`` names of the LoRA targets that adapt the output layer: the group and the HF / Megatron leaves
_UNEMBED_TARGETS = frozenset({"unembed", "lm_head", "head", "output_layer"})


def _validate(args: Namespace) -> None:
    """Reject at startup what the kernels do not serve (``args``: miles' flat namespace, the op selected).

    The checks read the miles / Megatron arguments where ``args`` has them and skip the ones it lacks. miles' own
    rules for ``--log-probs-backend cake`` (no true-on-policy mode, no sampling-support replay) are miles'.
    """
    chunk_size = getattr(args, "cake_lm_head_chunk_size", DEFAULT_CHUNK_SIZE)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError(f"{CHUNK_SIZE_FLAG} must be an integer in [1, {MAX_CHUNK_SIZE}], got {chunk_size!r}")
    problem = next(_unsupported(args), None)
    if problem is not None:
        raise ValueError(f"{BACKEND} {problem}")


def _unsupported(args: Namespace) -> Iterator[str]:
    """What ``args`` sets that the kernels do not serve, one message each (skipping the attributes ``args`` lacks)."""
    train_backend = getattr(args, "train_backend", None)
    if train_backend is not None and train_backend != "megatron":
        yield f"needs --train-backend megatron (the output layer hook of Megatron's GPTModel), got {train_backend}"
    tensor_parallel = getattr(args, "tensor_model_parallel_size", None)
    if tensor_parallel is not None and tensor_parallel != 1:
        yield (
            "needs --tensor-model-parallel-size 1: the kernels take the whole output weight on one GPU, "
            f"got {tensor_parallel}"
        )
    loss_type = getattr(args, "loss_type", None)
    if loss_type is not None and loss_type not in LOSS_TYPES:
        yield f"supports --loss-type {' and '.join(LOSS_TYPES)}, got {loss_type}"
    entropy_coef = getattr(args, "entropy_coef", None)
    if entropy_coef is not None and entropy_coef != 0:
        yield f"computes no entropy: it needs --entropy-coef 0, got {entropy_coef}"
    for flag in ("observe_training_entropy", "use_rollout_entropy"):
        if getattr(args, flag, False):
            yield f"computes no entropy: drop --{flag.replace('_', '-')}"
    if hasattr(args, "vocab_size"):
        vocab_size = args.vocab_size
        if vocab_size is None:
            yield "needs --vocab-size (the HF config's vocab_size): the softmax runs over the true vocabulary"
        elif vocab_size % VOCAB_MULTIPLE:
            yield (
                f"needs a --vocab-size that is a multiple of {VOCAB_MULTIPLE} (the kernels' logits tile; padding the "
                f"vocabulary would change the softmax), got {vocab_size}"
            )
    hidden_size = getattr(args, "hidden_size", None)
    if hidden_size is not None and hidden_size % HIDDEN_MULTIPLE:
        yield f"needs a --hidden-size that is a multiple of {HIDDEN_MULTIPLE}, got {hidden_size}"
    if getattr(args, "bf16", None) is False:
        yield "needs --bf16: the kernels take BF16 hidden states and output weights"
    if getattr(args, "multi_lora_n_adapters", 0):
        yield "does not support multi-LoRA (--multi-lora-n-adapters)"
    if getattr(args, "lora_rank", 0) and _targets_unembed(getattr(args, "target_modules", None)):
        yield "does not support a LoRA adapter on the output layer (the unembed / lm_head --target-modules)"
    if getattr(args, "use_mup", False):
        yield "does not apply MuP's output logit scaling (--use-mup)"


def _targets_unembed(target_modules) -> bool:
    """Does ``--target-modules`` (a comma-separated string or a list) name the output layer?"""
    if not target_modules:
        return False
    names = target_modules.split(",") if isinstance(target_modules, str) else target_modules
    return any(str(name).strip().rpartition(".")[2] in _UNEMBED_TARGETS for name in names)


OP = OpSpec(
    name="lm_head",
    summary="chunked LM-head selected-token log-probabilities, forward + backward, BF16 hidden rows and weight",
    variants=("det", "fast"),
    selected_by=(("log_probs_backend", "cake"),),
    packages=("cake_lm_head_det_train", "cake_lm_head_nondet_train"),
    exports=("log_probs",),
    flags=(
        Flag(
            CHUNK_SIZE_FLAG,
            dict(
                type=int,
                default=DEFAULT_CHUNK_SIZE,
                help=(
                    "Token rows per chunk of the cake lm_head kernels, at most 65535: the largest vocabulary-sized "
                    "buffer holds this many rows (BF16 logits of [chunk, V]); det's bits depend on it. "
                    f"Default: {DEFAULT_CHUNK_SIZE}."
                ),
            ),
        ),
    ),
    validate=_validate,
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
