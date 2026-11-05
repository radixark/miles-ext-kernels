"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Dense projection GEMMs for SM100a / SM103a / SM107a, forward and backward:

- ``batched_projection(x, weight, *, weight_layout)``: the per-head projections of the MLA / DSA attention layers (the
  q-absorb and value einsums, DeepSeek-V4's output-group projection) as one batched BF16 GEMM over strided views.
- ``router_gating(x, weight)``: the MoE router gate GEMM in FP32, as Megatron's ``TopKRouter.gating`` computes it with
  ``--moe-router-dtype fp32``.

``det`` runs the deterministic package and repeats bit for bit for identical inputs; ``fast`` runs the
non-deterministic package, whose weight gradients add split partials in the order they finish where its measured
tables select a program of its own, and the deterministic programs, with their bits, everywhere else
(``fast_projection_programs`` / ``fast_router_programs`` report its choice). miles selects these GEMMs with
``--batched-projection-gemm-backend cake`` and ``--moe-router-gemm-backend cake`` and runs ``det``. Neither variant is
batch-invariant: the plan of a GEMM depends on the token count, so a token's results can change in their last bits
with the number of tokens of the call. ``prebuild(projections=..., router=...)`` builds every program either variant
may run, at any token count.
"""

from argparse import Namespace

from miles_ext_kernels.cake._runtime.spec import OpSpec, lazy_exports

# the miles arguments that select these GEMMs
_SELECTORS = (("batched_projection_gemm_backend", "cake"), ("moe_router_gemm_backend", "cake"))


def _validate(args: Namespace) -> None:
    """The GEMMs replace GEMMs of miles' Megatron model code: a selected backend needs ``--train-backend megatron``."""
    for dest, value in _SELECTORS:
        if getattr(args, dest, None) != value:
            continue
        backend = getattr(args, "train_backend", None)
        if backend != "megatron":
            raise ValueError(
                f"--{dest.replace('_', '-')} {value} replaces a GEMM of miles' Megatron model code: it requires "
                f"--train-backend megatron (got {backend!r})"
            )


OP = OpSpec(
    name="dense_gemm",
    summary="batched per-head projection GEMM and FP32 MoE router gate GEMM, forward + backward",
    variants=("det", "fast"),
    cli_variants=("det",),
    selected_by=_SELECTORS,
    packages=("dense_projection_det_gemm", "dense_projection_gemm"),
    exports=(
        "OPERATIONS",
        "WEIGHT_LAYOUTS",
        "batched_projection",
        "check_projection_dims",
        "fast_projection_programs",
        "fast_router_programs",
        "router_gating",
    ),
    validate=_validate,
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
