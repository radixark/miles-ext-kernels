"""The backward forms of a ``fast`` call (``_forms``) without a GPU: the GPU tests compare ``fast``'s ``dW`` bits only
on the calls these rules say keep ``det``'s, and their ``T`` values are chosen to run each form. The expected forms
are the package README's rules at the default chunk size (2 to 4 chunks vocabulary-major, except 2 chunks whose second
holds at least half a chunk; at most 65535 rows; the short-tail fold of fewer than three chunks of ``T``)."""

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels.cake._impl.lm_head import ops  # noqa: E402

from . import _forms  # noqa: E402

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
C = 4096


@pytest.mark.parametrize(
    ("T", "rows", "chunk", "folds", "vocab_major"),
    [
        (1, 1, C, False, False),  # one row
        (4095, 3890, C, False, False),  # one chunk
        (4097, 3892, C, False, False),  # 5 % ignored: one chunk of valid rows
        (4097, 4097, C, True, True),  # every row valid: a one-row tail, folded forward, vocabulary-major backward
        (4600, 4600, C, True, True),  # a short tail
        (6800, 6460, C, False, False),  # 2 chunks, the second holds at least half a chunk
        (8192, 8192, C, False, False),  # 2 full chunks
        (16231, 15419, C, False, True),  # 4 chunks (5 % ignored)
        (16231, 16231, C, False, True),  # 4 chunks, every row valid
        (16384, 10977, C, False, True),  # 3 chunks (a third ignored)
        (16385, 16385, C, False, False),  # 5 chunks
        (20481, 19457, C, False, False),  # 5 chunks (5 % ignored)
        (4097, 3892, 1024, False, True),  # 4 chunks of 1024
        (5000, 4750, 1024, False, False),  # 5 chunks of 1024
        (70001, 66500, 65535, False, False),  # 2 chunks with a short tail, but more than 65535 rows
    ],
)
def test_the_fast_backward_form_follows_the_package_rules(T, rows, chunk, folds, vocab_major):
    assert _forms.fast_folds(T, rows, chunk) is folds
    assert _forms.fast_vocab_major(rows, chunk) is vocab_major
    assert _forms.fast_keeps_det_dw(T, rows, chunk) is not (folds or vocab_major)


@pytest.mark.parametrize("arch", ARCHES)
def test_every_architecture_registers_the_vocabulary_major_forms(arch):
    """``_forms.fast_vocab_major`` assumes the record has the forms the package requires of it (without them the
    package runs the chunk loop), with and without a temperature."""
    jit = ops._jit("fast")
    record = jit.select_module(arch)
    plain = set(jit.registered_stages(record))
    tau = set(jit.registered_stages_with_temperature(record))
    common = {
        "gemm_dx",
        "gemm_dx_red",
        "gemm_dx_red_dcnt",
        "gemm_dw_cast_bf16",
        "gemm_dw_cast_bf16_dcnt",
    }
    assert common | {"gemm_logits_dz"} <= plain, sorted(common | {"gemm_logits_dz"} - plain)
    assert common | {"gemm_logits_dz_tau"} <= tau, sorted(common | {"gemm_logits_dz_tau"} - tau)
