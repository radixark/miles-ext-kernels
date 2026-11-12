"""The non-deterministic package's band facts that ``ops.py`` documents (CPU; torch and tvm_ffi, no device).

- The gate read in place (strided rows) or copied (packed rows) can select forward programs that associate a row's
  sum of squares differently, so ``fast``'s ``y`` need not be the same bits either way (``det``'s is).
- On sm_107a the repeatable backward (``det``'s exact weight-gradient sum) also runs before the fused band: from 513
  rows up to 64 rows per SM.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels.cake._impl.gated_rmsnorm import ops  # noqa: E402

from .test_prebuild_routes import PARTS  # noqa: E402

# a forward role's association of a row's sum of squares (the package's documentation): the tiny and the short
# forward 8-term chains of 16 lanes; the sm_107a pipelined forward of packed rows four 4-term chains of 8 lanes; the
# others 16-term chains of 8 lanes
SUM_ORDER = {"fwd_tiny": "8x16", "fwd_short": "8x16", "fwd_pipe_packed": "4x4x8"}


def _forward_order(jit, arch: str, rows: int, sm_count: int, packed: bool) -> str:
    module = jit.module_for_rows(arch, "fwd", rows, sm_count, packed=packed)
    role = next(role for role, name in jit.ROUTES[arch].items() if name == module)
    return SUM_ORDER.get(role, "16x8")


@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_the_gate_layout_can_change_the_forward_sum_order(arch, sm_count):
    jit = ops._package("fast").cake_jit
    differ = [
        rows
        for rows in range(sm_count, 1000 * sm_count, sm_count)
        if _forward_order(jit, arch, rows, sm_count, True) != _forward_order(jit, arch, rows, sm_count, False)
    ]

    assert differ, f"{arch} with {sm_count} SMs: packed and strided rows always share a forward sum order"


@pytest.mark.parametrize(("arch", "sm_count"), [part for part in PARTS if part[0] == "sm_107a"])
def test_sm107a_runs_the_repeatable_backward_before_the_fused_band(arch, sm_count):
    jit = ops._package("fast").cake_jit
    routes = jit.ROUTES[arch]
    cut = jit.SM107A_BF16_REP_LAST * sm_count

    def role(rows: int) -> str:
        module = jit.module_for_rows(arch, "bwd_bf16", rows, sm_count, packed=False)
        return next(name for name, routed in routes.items() if routed == module)

    assert role(513) == role(cut) == "bwd_bf16_rep"
    assert role(512) == "bwd_bf16_t512"
    assert role(cut + 1) == "bwd_bf16_fused"
