"""The accuracy budgets of the KDA GPU tests against the exact FP64 recurrence of the same inputs: twice the largest
error measured over :data:`SHAPES` and the architectures measured.

Measured with FlashInfer 7a13d7ba's packages for the ``det`` call miles makes (raw gate, in-kernel q / k L2 norm, q / k
/ v column slices of one packed projection) on the inputs of :data:`SHAPES` (``_reference.make_raw_inputs`` and
``_reference_l2n.raw_qk``): one B200 (SM100a) and one GB300 (SM103a), CUDA 13.0. SM107a (R200) is not measured yet: its
runs take these budgets.

* The token tensors (``out``, ``dq``, ``dk``, ``dv``, ``dbeta``, ``dg``): relative RMS error
  (``tests/cake/helpers.py:assert_rel_rms``), :data:`MEASURED` the largest of each. ``dg``, the raw gate's gradient, is
  about 1 %: exempt from the 1.02x detection check (:data:`LOOSE`), as the helpers prescribe for such a tensor.
* The gate parameters' gradients (``dA_log``, ``ddt_bias``) are sums over every token of a head. Their relative error
  depends on how far the terms cancel, which the shape sets (head count, document lengths): on one B200 and one GB300,
  0.8 % to 29 % for ``dA_log``, 4.1 % to 6.3 % for ``ddt_bias``. They are gated by :func:`accumulation_error` instead,
  the error over the root sum of squares of the exact terms (``_reference.gate_param_terms``), which does not depend on
  that cancellation: :data:`PARAMS_MEASURED` the largest of each. ``dA_log`` has one value per head, so at one head it
  is still one draw: its largest value is at one head. fla's errors on the same inputs are of the same size
  (``test_kda_gpu.py:test_fla_parity_with_fp32_beta``). The fast and context-parallel tests gate these two by twice
  ``det``'s relative RMS error on the same inputs: the same comparison in either measure, as the factor between them is
  that of the inputs.
"""

from __future__ import annotations

from collections.abc import Mapping

# (document lengths, heads): ragged packs of one and several documents whose lengths are not multiples of 64, at the
# per-rank head counts of 96-head KDA layers at tensor parallelism 2 / 4 / 8 (48 / 24 / 12), 64 heads, and the
# segmented route's 1 and 4 heads
SHAPES = [
    ((1000,), 1),
    ((63, 1, 65, 300), 1),
    ((1000, 37, 300), 4),
    ((777,), 12),
    ((300, 129, 500), 12),
    ((1100, 257), 24),
    ((513, 70, 900), 48),
    ((2000,), 64),
    ((1000, 37, 300, 63), 64),
]
TOKEN_TENSORS = ("out", "dq", "dk", "dv", "dbeta", "dg")
PARAM_TENSORS = ("dA_log", "ddt_bias")

# the largest relative RMS error over the 9 shapes and the architectures measured, det, the miles call
MEASURED = {
    "out": 4.223e-3,  # s1000-h1, GB300
    "dq": 4.074e-3,  # s300_129_500-h12, B200
    "dk": 3.764e-3,  # s1000_37_300-h4, B200
    "dv": 3.892e-3,  # s1000-h1, GB300
    "dbeta": 4.054e-3,  # s1000-h1, B200
    "dg": 1.004e-2,  # s63_1_65_300-h1, B200 (the raw gate's gradient)
}
# the largest accumulation_error over the 9 shapes and the architectures measured, det, the miles call
PARAMS_MEASURED = {
    "dA_log": 3.306e-2,  # s63_1_65_300-h1, B200
    "ddt_bias": 6.341e-2,  # s63_1_65_300-h1, B200
}
LOOSE = {
    "dg": "its own error is about 1 % (1.004e-2 at most)",
    "dA_log": "a sum over every token of a head: its own relative error is 1 % or more (PARAMS_MEASURED)",
    "ddt_bias": "a sum over every token of a head: its own relative error is 4 % or more (PARAMS_MEASURED)",
}


def shape_id(shape) -> str:
    """``s<lengths>-h<heads>`` of a :data:`SHAPES` entry."""
    lengths, heads = shape
    return "s" + "_".join(str(n) for n in lengths) + f"-h{heads}"


def budget() -> dict[str, float]:
    """The relative RMS budget of every token tensor of a ``det`` call."""
    return {name: 2 * MEASURED[name] for name in TOKEN_TENSORS}


def loose(names) -> dict[str, str]:
    """The :data:`LOOSE` entries of ``names``."""
    return {name: LOOSE[name] for name in names if name in LOOSE}


def accumulation_error(got, ref, terms) -> float:
    """``||got - ref|| / sqrt(sum terms^2)`` in FP64: the error of a sum over the root sum of squares of its exact
    terms (``terms`` ``[S, H, 128]``, ``_reference.gate_param_terms``), whatever the shape of the sum."""
    import torch

    error = (got.detach().to("cpu", torch.float64) - ref.detach().to("cpu", torch.float64)).norm()
    return (error / terms.detach().to("cpu", torch.float64).square().sum().sqrt()).item()


def assert_param_grads(got: Mapping, ref: Mapping, terms: Mapping) -> dict[str, float]:
    """``dA_log`` and ``ddt_bias`` of ``got`` finite and within twice :data:`PARAMS_MEASURED` in
    :func:`accumulation_error`; returns the measured errors, and reports every failure at once."""
    import torch

    measured, failures = {}, []
    for name in PARAM_TENSORS:
        if not bool(torch.isfinite(got[name]).all()):
            failures.append(f"{name}: not finite")
            continue
        error = measured[name] = accumulation_error(got[name], ref[name], terms[name])
        if error > 2 * PARAMS_MEASURED[name]:
            failures.append(f"{name}: accumulation error {error:.3e} > budget {2 * PARAMS_MEASURED[name]:.3e}")
    if failures:
        raise AssertionError("\n".join([*failures, f"measured: {measured}"]))
    return measured
