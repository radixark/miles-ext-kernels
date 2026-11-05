"""The dense projection GEMMs on two devices of one part: ``det`` gives the same bits on both (its plans are a function
of the views' geometry and the device's architecture and SM count), each device keeps its own plans and workspaces in
one process, and ``fast`` keeps ``det``'s forward and input-gradient bits on both, with its weight gradient within the
accuracy budget against FP64. The variant is configured with ``cake.configure(cake_dense_gemm_variant=...)``. Marked
``multi_gpu``: a one-GPU run deselects the module with ``-m "not multi_gpu"``, and collecting it touches no device.
Needs two CUDA devices of the same SM100a / SM103a / SM107a part; skips otherwise.
"""

import functools

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake.dense_gemm import (  # noqa: E402
    batched_projection,
    fast_projection_programs,
    fast_router_programs,
    router_gating,
)

DEVICES = ("cuda:0", "cuda:1")
TWO_DEVICES = torch.cuda.device_count() >= 2
# (compute capability, SM count) of each device, read only where there are two
_PARTS = (
    {(torch.cuda.get_device_capability(d), torch.cuda.get_device_properties(d).multi_processor_count) for d in DEVICES}
    if TWO_DEVICES
    else set()
)
pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.skipif(not TWO_DEVICES, reason="two CUDA devices required"),
    pytest.mark.skipif(
        len(_PARTS) != 1 or next(iter(_PARTS))[0] not in cake.ARCH_BY_CAPABILITY,
        reason="two devices of one SM100a / SM103a / SM107a part required",
    ),
]
SEED = 20261105
BF16_ATOL = BF16_RTOL = 1e-2
# the relative Frobenius error of an FP32 result rounded once to BF16 (half a BF16 spacing is 2**-9 of the value)
BF16_ROUNDED_REL_FRO_MAX = 2.0**-8


def _leaves(shapes, seed, dtypes):
    """CPU tensors, copied to each device so both see the same values."""
    gen = torch.Generator().manual_seed(seed)
    return [(torch.randn(*shape, generator=gen) * scale).to(dtype) for (shape, scale), dtype in zip(shapes, dtypes)]


BF16 = (torch.bfloat16, torch.bfloat16, torch.bfloat16)
ROUTER = (torch.bfloat16, torch.bfloat16, torch.float32)
NAMES = ("glm5-q-absorb-h16-t1001", "dense-4096-32-t16384", "router-e256-h5120-t16384")


@functools.cache
def cases():
    """name -> (kind, weight layout, leaves, fast programs of the part), built at the first test."""
    ((capability, sm_count),) = _PARTS
    arch = cake.ARCH_BY_CAPABILITY[capability]
    table = {
        "glm5-q-absorb-h16-t1001": (
            "projection",
            "in_out",
            _leaves((((1001, 16, 192), 1.0), ((16, 192, 512), 192**-0.5), ((1001, 16, 512), 1.0)), SEED, BF16),
            fast_projection_programs(16, 192, 512, weight_layout="in_out", tokens=1001, arch=arch, sm_count=sm_count),
        ),
        "dense-4096-32-t16384": (
            "projection",
            "out_in",
            _leaves((((16384, 1, 4096), 1.0), ((1, 32, 4096), 4096**-0.5), ((16384, 1, 32), 1.0)), SEED + 1, BF16),
            fast_projection_programs(1, 4096, 32, weight_layout="out_in", tokens=16384, arch=arch, sm_count=sm_count),
        ),
        "router-e256-h5120-t16384": (
            "router",
            None,
            _leaves((((16384, 1, 5120), 1.0), ((256, 5120), 0.02), ((16384, 1, 256), 1e-3)), SEED + 2, ROUTER),
            fast_router_programs(256, 5120, tokens=16384, arch=arch, sm_count=sm_count),
        ),
    }
    assert tuple(table) == NAMES
    return table


def _step(name, device, variant):
    kind, layout, leaves, _ = cases()[name]
    x, weight, grad = (t.to(device, torch.float64 if variant == "fp64" else t.dtype) for t in leaves)
    x.requires_grad_(True)
    weight.requires_grad_(True)
    if variant == "fp64":
        if kind == "router":
            out = x @ weight.T
        else:
            out = torch.einsum("thd,hdm->thm" if layout == "in_out" else "thm,hdm->thd", x, weight)
    else:
        cake.configure(cake_dense_gemm_variant=variant)
        if kind == "router":
            out = router_gating(x, weight)
        else:
            out = batched_projection(x, weight, weight_layout=layout)
    out.backward(grad)
    torch.cuda.synchronize(device)
    return [t.detach().cpu() for t in (out, x.grad, weight.grad)]


@pytest.mark.parametrize("name", NAMES)
def test_det_gives_the_same_bits_on_both_devices(name):
    # interleaved, so each device's cached plans and workspaces are used again after the other device ran
    first = [_step(name, device, "det") for device in DEVICES]
    again = [_step(name, device, "det") for device in DEVICES]

    for results in (first[1], again[0], again[1]):
        for got, want in zip(results, first[0]):
            assert torch.equal(got, want)


@pytest.mark.parametrize("name", NAMES)
def test_fast_keeps_det_forward_and_input_gradient_bits_on_both_devices(name):
    kind, _, _, programs = cases()[name]
    want = _step(name, DEVICES[0], "fp64")[2]
    for device in DEVICES:
        det = _step(name, device, "det")
        fast = _step(name, device, "fast")

        assert torch.equal(fast[0], det[0]) and torch.equal(fast[1], det[1]), device
        if programs["weight_grad"] is None:
            assert torch.equal(fast[2], det[2]), device
        elif kind == "router":
            rel = float((fast[2].double() - want).norm() / want.norm())
            assert rel <= BF16_ROUNDED_REL_FRO_MAX, (device, programs["weight_grad"], rel)
        else:
            diff = (fast[2].double() - want).abs()
            outside = int((diff > BF16_ATOL + BF16_RTOL * want.abs()).sum())
            assert outside == 0, (device, programs["weight_grad"], outside)
