"""Both deterministic GDN kernel sets against flash-linear-attention's ``chunk_gated_delta_rule``: output and every
input gradient within the BF16 tolerance, and bit-identical repeated passes.

``det`` is what a ``det`` call without a ``cp_context`` runs (the vendored Cake deterministic package); ``cp`` is what
a call with a ``cp_context`` runs (``ops.cp_chunk_gated_delta_rule``, the det kernels with FLA's context parallelism,
generated from FLA's dataflow), here as one rank.  Packed varlen (``cu_seqlens``) and an equal-length batch of two sequences, with one and two value heads
per key head, as miles' head-sharded GDN layer feeds the kernel.  Needs a CUDA device; the tests need SM100a / SM103a /
SM107a and fla, and skip elsewhere.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.gdn import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

NUM_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128
SEQLENS = [128, 256, 128]
# the deterministic entry points: a det call without a cp_context, and the generated kernels every cp_context call runs
DET_KERNELS = {"det": "chunk_gated_delta_rule", "cp": "cp_chunk_gated_delta_rule"}


def _fla_kernel():
    if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
        pytest.skip("the deterministic GDN kernels need SM100a / SM103a / SM107a")
    return pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule


def _grouped_inputs(v_per_k: int, use_cu: bool):
    """q/k with NUM_HEADS key heads, v / g / beta with ``v_per_k`` value heads per key head; packed varlen
    (``cu_seqlens``) or an equal-length batch of two sequences."""
    torch.manual_seed(0)
    total = sum(SEQLENS)
    num_v_heads = NUM_HEADS * v_per_k

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    query = randn(1, total, NUM_HEADS, HEAD_K_DIM)
    key = randn(1, total, NUM_HEADS, HEAD_K_DIM)
    value = randn(1, total, num_v_heads, HEAD_V_DIM)
    g = -torch.nn.functional.softplus(randn(1, total, num_v_heads, dtype=torch.float32))  # fp32 log-decay <= 0
    beta = randn(1, total, num_v_heads).sigmoid()  # sigmoid(b) in the activation dtype
    cu_seqlens = torch.tensor([0, *SEQLENS], device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    if not use_cu:
        query, key, value, g, beta = (t.reshape(2, total // 2, *t.shape[2:]) for t in (query, key, value, g, beta))
        cu_seqlens = None
    return (query, key, value, g, beta), cu_seqlens


def _autograd_step(kernel, inputs, cu_seqlens):
    query, key, value, g, beta = (t.detach().clone().requires_grad_(True) for t in inputs)
    out, _ = kernel(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    torch.manual_seed(1)
    out.backward(torch.randn_like(out))
    return {"out": out.detach(), "dq": query.grad, "dk": key.grad, "dv": value.grad, "dg": g.grad, "dbeta": beta.grad}


@pytest.mark.parametrize("v_per_k", [1, 2])
@pytest.mark.parametrize("use_cu", [True, False])
@pytest.mark.parametrize("kernel", list(DET_KERNELS))
def test_fla_equivalence_and_determinism(kernel, use_cu, v_per_k):
    fla_kernel = _fla_kernel()
    cake.configure(cake_gdn_variant="det")
    cake_kernel = getattr(ops, DET_KERNELS[kernel])
    inputs, cu_seqlens = _grouped_inputs(v_per_k, use_cu)

    ref = _autograd_step(fla_kernel, inputs, cu_seqlens)
    first = _autograd_step(cake_kernel, inputs, cu_seqlens)
    second = _autograd_step(cake_kernel, inputs, cu_seqlens)
    for name in ref:
        assert torch.equal(first[name], second[name]), f"cake {kernel} {name} is not bit-deterministic"
        actual, expected = first[name].float(), ref[name].float()
        scale = max(1.0, expected.abs().max().item()) if name == "dg" else 1.0
        torch.testing.assert_close(
            actual, expected, atol=1e-2 * scale, rtol=1e-2, msg=lambda m, name=name: f"{kernel} {name}: {m}"
        )
