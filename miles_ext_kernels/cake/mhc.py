"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Four-stream mHC (manifold-constrained hyper-connection) training kernels, forward and backward, and the model head's
stream mean, for SM100a / SM103a (hidden 4096, 5120, 7168) and SM107a (hidden 4096).

One connection of a layer is ``U, handle = mhc_prepare(X, fn, scale, base)`` (the RMS-normalised mapping GEMM, the
pre / post / Sinkhorn coefficients and the stream collapse ``U``), the layer's branch on ``U``, then
``Y = mhc_finish(X, branch_output, handle)`` (the residual mixing). The backward of the pair returns one single-cast
``dX`` and FP32 ``dfn``, ``dscale``, ``dbase`` (cast back to the parameters' dtype when they are BF16).
``mhc_head(X)`` is the model head's stream mean, bit for bit torch's BF16 ``X.mean(dim=1)`` and its gradient.

Variants: ``det`` sums the parameter gradients in a fixed order; ``fast`` adds the ``dfn`` partials with FP32
atomics (only the low-order bits of ``dfn`` vary from run to run; the forward and ``dX`` are the ``det`` bits). The
head has one program for both. miles selects these kernels with ``--mhc-backend cake``.

``miles_ext_kernels.cake.prebuild("mhc", hidden=..., sinkhorn_iters=...)`` builds both variants and the head for the
model's hidden size and Sinkhorn iteration count; the epsilons are launch arguments and select no library.

Importing this module imports neither torch nor the kernels: the entry points load on first access.
"""

from argparse import Namespace

from miles_ext_kernels.cake._runtime.spec import OpSpec, lazy_exports

# The connection shapes the kernels are built for: four residual streams of one of these hidden sizes.
_STREAMS = 4
_HIDDEN_SIZES = (4096, 5120, 7168)


def _validate(args: Namespace) -> None:
    hidden = getattr(args, "hidden_size", None)
    if hidden is not None and hidden not in _HIDDEN_SIZES:
        raise ValueError(f"--mhc-backend cake runs the hidden sizes {_HIDDEN_SIZES}, not --hidden-size {hidden}")


OP = OpSpec(
    name="mhc",
    summary="four-stream manifold-constrained hyper-connections, forward + backward, and the stream-mean head",
    variants=("det", "fast"),
    selected_by=(("mhc_backend", "cake"),),
    packages=(
        "cake_mhc_det_train_backend",
        "cake_mhc_det_train_head_backend",
        "cake_mhc_train_backend",
    ),
    exports=(
        "HIDDEN_SIZES",
        "STREAMS",
        "is_supported",
        "mhc_finish",
        "mhc_head",
        "mhc_prepare",
        "supported_hidden_sizes",
    ),
    validate=_validate,
)

__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)
