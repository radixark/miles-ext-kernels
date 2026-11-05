"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Packed causal depthwise conv1d + SiLU (kernel size 4) training kernels for SM100a / SM103a / SM107a: the short
convolutions of miles' GDN and KDA linear-attention layers, forward and backward, with a context-parallel route.

Two variants, FlashInfer's two Cake packages: ``det`` sums the weight gradient in a fixed order, so it repeats bit for
bit; ``fast`` sums it in one backward launch with unordered reductions (``y`` and ``dx`` are ``det``'s bits, the
weight gradient stays within ``det``'s error budget). miles selects the op with ``--linear-attention-conv-backend
cake``. Entry points: :func:`causal_conv1d` (one rank's packed documents), :func:`causal_conv1d_kcp` and
:func:`causal_conv1d_kcp_multi` (a rank's fragments under context parallelism, with the context the layer's GDN or
KDA context-parallel route built) and :func:`check_supported` (the checks of a layer). ``prebuild("conv1d",
weight_dtype=...)`` builds every library of both variants, both SiLU' policies and the context-parallel route.
"""

from argparse import Namespace

from miles_ext_kernels.cake._runtime.spec import Flag, OpSpec, lazy_exports


def _validate(args: Namespace) -> None:
    """miles does not route context parallelism to the Cake short conv yet."""
    size = getattr(args, "context_parallel_size", None) or 1
    if size > 1:
        raise ValueError(
            "--linear-attention-conv-backend cake does not support context parallelism yet (--context-parallel-size "
            f"{size}): the Cake short conv runs context parallelism through "
            "miles_ext_kernels.cake.conv1d.causal_conv1d_kcp with the context of the layer's GDN / KDA "
            "context-parallel route, which miles does not call yet; use --linear-attention-conv-backend fla"
        )


OP = OpSpec(
    name="conv1d",
    summary="packed causal depthwise conv1d + SiLU, kernel size 4, forward + backward, context parallelism",
    variants=("det", "fast"),
    selected_by=(("linear_attention_conv_backend", "cake"),),
    packages=("cake_causal_conv1d_det_train_backend", "cake_causal_conv1d_train_backend"),
    exports=("causal_conv1d", "causal_conv1d_kcp", "causal_conv1d_kcp_multi", "check_supported"),
    flags=(
        Flag(
            "--cake-conv1d-silu-grad-preactivation",
            dict(
                choices=("bf16", "fp32"),
                default="bf16",
                help=(
                    "Where the short-conv backward evaluates the SiLU derivative: 'bf16' at the BF16-rounded "
                    "preactivation (flash-linear-attention's rounding point), 'fp32' at the unrounded FP32 "
                    "preactivation with an FP64 weight-gradient sum (a slower program). Default: 'bf16'. Only with "
                    "--linear-attention-conv-backend cake."
                ),
            ),
        ),
    ),
    validate=_validate,
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
