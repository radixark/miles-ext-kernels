"""The ``"boundary"`` inputs of ``_reference.py`` on the CPU (torch, no GPU): they put the quantized values on the E4M3
and E8M0 boundaries ``BOUNDARY_BLOCKS`` names, along rows and along columns, and the reference quantizes those blocks to
the RCEIL scale bytes of their maxima. The GPU tests compare every program's bytes with the reference on these inputs.
"""

import pytest

torch = pytest.importorskip("torch")

from . import _reference as reference  # noqa: E402

M, K = 128, 128
CPU = torch.device("cpu")
# The E8M0 scale byte of each ``BOUNDARY_BLOCKS`` block: the exponent of its BF16 maximum plus 127, less 8, plus one
# when the maximum's mantissa is above 1.75, clamped at 0.
SCALES = (128, 127, 128, 127, 127, 0, 247)


def targets(index: int) -> torch.Tensor:
    """The BF16-rounded values of block ``index``, 32 of them."""
    return torch.tensor(reference.BOUNDARY_BLOCKS[index], dtype=torch.float32).repeat(8).to(torch.bfloat16).float()


def boundary_inputs():
    return reference.make_inputs(M, K, values="boundary", seed=M + K, device=CPU)


def test_boundary_is_one_of_the_values():
    assert "boundary" in reference.VALUES
    assert len(reference.BOUNDARY_BLOCKS) == len(SCALES)


def test_boundary_blocks_put_the_products_on_the_targets():
    gated_input, grad_output = boundary_inputs()
    gate, up = gated_input.float().split(K, dim=1)
    grad = grad_output.float()
    for index in range(len(reference.BOUNDARY_BLOCKS)):
        for where, rows, cols in (("row", index, slice(0, 32)), ("column", slice(32, 64), index)):
            what = f"block {index} along a {where}"
            assert torch.equal(gate[rows, cols], torch.full((32,), 32.0)), what
            assert torch.equal(grad[rows, cols], torch.full((32,), 32.0)), what
            assert torch.equal(32.0 * up[rows, cols], targets(index)), what


@pytest.mark.parametrize("backward", (False, True), ids=("forward", "backward"))
def test_boundary_blocks_quantize_to_the_rceil_scales(backward):
    gated_input, grad_output = boundary_inputs()
    if backward:
        want = reference.backward(gated_input, grad_output, rowwise=True, colwise=True)
    else:
        want = reference.forward(gated_input, rowwise=True, colwise=True)
    N = 2 * K if backward else K
    row_index = reference.scale_index(M, N, CPU)
    col_index = reference.scale_index(N, M, CPU)
    for index, scale in enumerate(SCALES):
        # block 0 of row ``index``; block 1 (rows 32-63) of column ``index``, the gate gradient's in the backward
        assert int(want["row_scales"][row_index[index, 0]]) == scale, f"row block {index}"
        assert int(want["col_scales"][col_index[index, 1]]) == scale, f"column block {index}"
    # 448 at scale 127 is the E4M3 maximum, 0x7E; -446 rounds to -448
    assert want["row_output"][1, 0:2].tolist() == [0x7E, 0xFE]
