"""The shapes the MoE grouped GEMM serves, checked where the model is built and at the forward call, before any launch
(CPU; needs torch). The kernels address each output with 32-bit element offsets: ``sum_m * N``, ``sum_m * K`` and
``E * N * K`` stay at most ``2**31 - 1``."""

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels.cake import moe_grouped_gemm  # noqa: E402

INT32 = 2**31 - 1


def test_the_row_bound_is_the_32_bit_span_of_the_wider_output():
    # K = 7168, N = 4096: the input gradient [sum_m, 7168] is the wider output; 299,593 rows fit, 300,000 do not
    assert moe_grouped_gemm.max_grouped_rows(4096, 7168) == 299_593 == INT32 // 7168
    assert moe_grouped_gemm.max_grouped_rows(7168, 4096) == 299_593
    assert moe_grouped_gemm.max_grouped_rows(2048, 768) == INT32 // 2048


def test_the_dims_check_takes_the_most_rows_of_the_configured_run():
    moe_grouped_gemm.check_grouped_dims(4096, 7168)
    moe_grouped_gemm.check_grouped_dims(4096, 7168, max_rows=299_593)
    with pytest.raises(ValueError, match="300000 rows of 7168 columns .* serve at most 299593 rows"):
        moe_grouped_gemm.check_grouped_dims(4096, 7168, max_rows=300_000)


def test_the_dims_check_takes_the_expert_count_of_the_weight_gradient():
    # 73 experts of [4096, 7168] are 2,143,289,344 elements, 74 are 2,172,649,472
    moe_grouped_gemm.check_grouped_dims(4096, 7168, num_groups=73)
    with pytest.raises(ValueError, match="74 experts of \\[4096, 7168\\] are 2172649472 elements"):
        moe_grouped_gemm.check_grouped_dims(4096, 7168, num_groups=74)


@pytest.mark.parametrize("n, k", [(384, 1024), (1024, 960), (128, 1024), (1024, 0)])
def test_the_dims_check_wants_multiples_of_256(n, k):
    with pytest.raises(ValueError, match="multiple of 256"):
        moe_grouped_gemm.check_grouped_dims(n, k)


def _meta(rows: int, groups: int, n: int, k: int):
    x = torch.empty(rows, k, dtype=torch.bfloat16, device="meta")
    w = torch.empty(groups, n, k, dtype=torch.bfloat16, device="meta")
    return x, w, torch.empty(groups, dtype=torch.int32, device="meta")


def test_a_call_past_the_row_bound_fails_at_the_forward_before_it_reaches_the_device():
    """The backward's input gradient would overflow: the forward call refuses it, before it looks at the device."""
    with pytest.raises(ValueError, match="serve at most 299593 rows"):
        moe_grouped_gemm.grouped_mm(*_meta(300_000, 2, 4096, 7168))
    # at the bound the shape checks pass and the call goes on to the device check (meta tensors are no CUDA device)
    with pytest.raises(ValueError, match="runs on CUDA devices"):
        moe_grouped_gemm.grouped_mm(*_meta(299_593, 2, 4096, 7168))


def test_a_call_past_the_weight_gradient_bound_fails_at_the_forward():
    with pytest.raises(ValueError, match="74 experts"):
        moe_grouped_gemm.grouped_mm(*_meta(1024, 74, 4096, 7168))
