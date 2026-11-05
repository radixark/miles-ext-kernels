"""The backward windows FlashInfer fixed, in the vendored packages' registry (needs torch and tvm_ffi, no GPU).

The token counts of ``test_dw_accuracy_gpu.py`` fall where FlashInfer changed the programs of the FP32-weight
backward:

- SM103a, H = 2048, T 58497 to 61952 (flashinfer-ai/flashinfer#8810): ``fast`` runs the programs ``det`` runs at
  these token counts, in place of a pair whose dw FlashInfer measured above the element budget at T 58560; at T 58496
  it keeps that pair.
- SM100a (H = 2048 from T 337 to 4096, H = 4096 from T 1793 to 3584, H = 5120 from T 65 to 6144) and SM107a (H = 2048
  from T 1025 to 16384, H = 5120 from T 4097 to 19968) (#8827): ``fast`` runs one kernel, in place of a pair whose dw
  error FlashInfer measured above that of the eager FP32 reduction; SM100a, H = 5120, above T 56256: one kernel in
  place of a pair (up to T 65536) and another kernel (above it), whose dw FlashInfer measured above the element
  budget.
- Above a cut, another single kernel than at the cut: on SM100a above T 65536 at H = 2048 and 4096 (#8827) and
  above T 131072 at H = 2048, 4096 and 5120 (#8848), on SM107a above T 65536 at H = 4096 (#8827), on SM103a above
  T 262144 at H = 4096 and T 196608 at H = 5120 (#8849: FlashInfer measured the kernel that ran there before above
  the element budget). #8848 gives SM100a a kernel with blocked FP32 dw sums from T 65537 (H = 2048 and 4096) or
  T 56257 (H = 5120) to T 131072; above T 131072 the compensated kernel of #8827 stays.
- SM107a, from T 65 to 127 at H = 2048 and to 256 at H = 4096 (#8827): ``fast`` runs another single kernel than the
  one it ran there before, which it still runs above T 50176 (H = 2048) and from T 24065 to 65536 (H = 4096).
- Every architecture, every width but 256 (#8813 for SM103a and SM107a): ``det`` sums each dw column in a fixed row
  order at its smallest token counts.
"""

import pytest

from ._windows import backward_form, backward_programs

pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")


def test_sm103a_fast_runs_the_det_programs_from_58497_to_61952():
    for T in (58497, 58560, 60224, 61952):
        assert backward_programs("fast", "sm_103a", 2048, T) == backward_programs("det", "sm_103a", 2048, T), T
        assert backward_form("fast", "sm_103a", 2048, T) == "pair", T
    assert backward_programs("fast", "sm_103a", 2048, 58496) != backward_programs("det", "sm_103a", 2048, 58496)


@pytest.mark.parametrize(
    ("arch", "hidden", "tokens"),
    [
        ("sm_100a", 2048, (337, 1024, 1025, 2048, 4096)),
        ("sm_100a", 4096, (1793, 2048, 3584)),
        ("sm_100a", 5120, (65, 1792, 1793, 6144)),
        ("sm_100a", 5120, (56257, 65536, 65537, 196609)),
        ("sm_107a", 2048, (1025, 2048, 4096, 8192, 16384)),
        ("sm_107a", 5120, (4097, 8192, 19968)),
    ],
)
def test_fast_runs_one_kernel_where_a_pair_fell_short(arch, hidden, tokens):
    for T in tokens:
        assert backward_form("fast", arch, hidden, T) == "single", T


# top: a token count of the window above the cut (its upper edge; 4 * cut above the last cut)
@pytest.mark.parametrize(
    ("arch", "hidden", "cut", "top"),
    [
        ("sm_100a", 2048, 65536, 131072),
        ("sm_100a", 2048, 131072, 524288),
        ("sm_100a", 4096, 65536, 131072),
        ("sm_100a", 4096, 131072, 524288),
        ("sm_100a", 5120, 131072, 524288),
        ("sm_107a", 4096, 65536, 262144),
        ("sm_103a", 4096, 262144, 1048576),
        ("sm_103a", 5120, 196608, 786432),
    ],
)
def test_fast_runs_another_single_kernel_above_a_cut(arch, hidden, cut, top):
    assert backward_form("fast", arch, hidden, cut) == backward_form("fast", arch, hidden, cut + 1) == "single"
    assert backward_programs("fast", arch, hidden, cut) != backward_programs("fast", arch, hidden, cut + 1)
    assert backward_programs("fast", arch, hidden, cut + 1) == backward_programs("fast", arch, hidden, top)


@pytest.mark.parametrize(("hidden", "top", "before"), [(2048, 127, 50177), (4096, 256, 24065)])
def test_sm107a_fast_runs_its_own_kernel_from_65(hidden, top, before):
    """``before``: a token count at which ``fast`` runs the kernel it ran from T 65 to ``top`` before #8827."""
    programs = backward_programs("fast", "sm_107a", hidden, 65)
    assert backward_programs("fast", "sm_107a", hidden, top) == programs
    assert backward_form("fast", "sm_107a", hidden, top) == "single"
    assert programs != backward_programs("fast", "sm_107a", hidden, before)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("hidden", [512, 1536, 2048, 4096, 5120, 6144])
def test_det_sums_each_dw_column_in_a_fixed_order_at_the_smallest_token_counts(arch, hidden):
    for T in (1, 2):
        assert backward_form("det", arch, hidden, T) == "column", T
