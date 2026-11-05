"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Ragged BF16 MoE grouped GEMM for SM100a / SM103a / SM107a: the expert projections of a routed MoE layer,
``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``, with the input and weight gradients, as
``torch._grouped_mm`` computes them.

Two variants. ``det`` reduces every output in a fixed order: the output and both gradients repeat bit for bit.
``fast`` runs the same forward and input-gradient programs (the same bits) and a weight gradient that, where it picks
the 512-wide k tile, adds the split tail of its last wave with unordered FP32 reductions.

miles selects the op with ``--moe-grouped-gemm-backend cake`` under ``--train-backend torchtitan``, and the variant
with ``--cake-moe-grouped-gemm-variant``. The kernels address their outputs with 32-bit element offsets:
:func:`check_grouped_dims` checks an expert weight and the most rows a call may get where the model is built, and
:func:`max_grouped_rows` names that bound. Importing this module does not import torch: the entry points load on
first access.
"""

from argparse import Namespace

from miles_ext_kernels.cake._runtime.spec import OpSpec, lazy_exports


def _validate(args: Namespace) -> None:
    backend = getattr(args, "train_backend", None)
    if backend != "torchtitan":
        raise ValueError(
            "--moe-grouped-gemm-backend cake replaces the expert GEMMs of miles' torchtitan models: it requires "
            f"--train-backend torchtitan (got {backend!r})"
        )


OP = OpSpec(
    name="moe_grouped_gemm",
    summary="ragged BF16 MoE grouped GEMM of the routed experts, forward + input and weight gradients",
    variants=("det", "fast"),
    selected_by=(("moe_grouped_gemm_backend", "cake"),),
    packages=("cake_moe_grouped_det_gemm", "cake_moe_grouped_gemm"),
    exports=("check_grouped_dims", "grouped_mm", "max_grouped_rows"),
    validate=_validate,
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
