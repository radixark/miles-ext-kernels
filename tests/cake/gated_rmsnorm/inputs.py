"""Inputs of the gated_rmsnorm GPU tests in miles' GDN layout, one forward + backward, and the FP64 reference.

The gate ``z`` is a column slice of the fused input projection ``[q | k | v | z | b | a]`` of ``Hk`` key and ``Hv``
value heads: ``2 * Hk * 128 + 2 * Hv * 128 + 2 * Hv`` columns, padded to a multiple of 8 (plus ``extra`` columns).
That width is the gate's token stride, so the kernels read ``z`` in place when the width is a multiple of 16 elements
and copy it otherwise (:func:`reads_in_place`). Every function imports torch when it runs.
"""

from __future__ import annotations

D = 128


def projection_width(key_heads: int, value_heads: int, extra: int = 0) -> int:
    """Columns of the fused input projection: q, k, v, z, b and a, padded to a multiple of 8, plus ``extra``."""
    width = 2 * key_heads * D + 2 * value_heads * D + 2 * value_heads
    return -(-width // 8) * 8 + extra


def make(tokens: int, key_heads: int, value_heads: int, extra: int = 0, seed: int = 0):
    """``(x, z, w, dy)`` on the current CUDA device: ``x`` and ``dy`` contiguous ``[T, Hv, 128]``, ``z`` the gate
    columns of a projection of :func:`projection_width` columns; BF16 of N(0, 1), ``w`` of 1 + N(0, 0.25)."""
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=device).manual_seed(((seed * 1_000_003 + tokens) * 131 + value_heads) * 17 + extra)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=g).bfloat16()

    x = randn(tokens, value_heads, D)
    projection = randn(tokens, projection_width(key_heads, value_heads, extra))
    start = (2 * key_heads + value_heads) * D
    z = projection[:, start : start + value_heads * D].unflatten(-1, (value_heads, D))
    dy = randn(tokens, value_heads, D)
    w = (1.0 + 0.5 * torch.randn(D, device=device, generator=g)).bfloat16()
    return x, z, w, dy


def reads_in_place(z) -> bool:
    """Do the kernels of both variants read the gate ``z`` in place (rather than a copy)?"""
    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops

    in_place = []
    for variant in ("det", "fast"):
        view, _, _ = ops._package(variant)._kernel_view(z, "z", shape=z.shape, device=z.device)
        in_place.append(view.data_ptr() == z.data_ptr())
    assert in_place[0] == in_place[1], "the two packages read the gate differently"
    return in_place[0]


def run(x, z, w, dy, *, use_checkpoint: bool = False, weight_grad: bool = True) -> dict:
    """``{"y", "dx", "dz", "dw"}`` of one autograd forward + backward of the configured variant (``dw`` is ``None``
    when ``weight_grad`` is false). Checks the documented result types: ``y`` BF16 of ``x``'s shape, every gradient
    BF16, the weight gradient of shape ``[128]``."""
    import torch
    from torch.utils.checkpoint import checkpoint

    from miles_ext_kernels.cake.gated_rmsnorm import rmsnorm_silu_gate

    xi, zi = (t.detach().requires_grad_() for t in (x, z))
    wi = w.detach().requires_grad_(weight_grad)
    if use_checkpoint:
        y = checkpoint(rmsnorm_silu_gate, xi, zi, wi, 1e-6, use_reentrant=False)
    else:
        y = rmsnorm_silu_gate(xi, zi, wi, 1e-6)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape, (y.dtype, y.shape)
    y.backward(dy)
    assert xi.grad.dtype == zi.grad.dtype == torch.bfloat16, (xi.grad.dtype, zi.grad.dtype)
    if weight_grad:
        assert wi.grad.dtype == torch.bfloat16 and wi.grad.shape == (D,), (wi.grad.dtype, wi.grad.shape)
    return {"y": y.detach(), "dx": xi.grad, "dz": zi.grad, "dw": wi.grad}


def fp64_reference(dy, x, z, w, eps: float = 1e-6) -> dict:
    """FP64 forward and closed-form backward from the BF16 inputs: ``y``, ``dx``, ``dz``, ``dw`` and ``mag``, the sum
    of ``|terms|`` of each weight-gradient column."""
    import torch

    xd, zd, wd, gd = x.double(), z.double(), w.double(), dy.double()
    rstd = 1.0 / torch.sqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
    n = xd * rstd
    s = torch.sigmoid(zd)
    silu = zd * s
    a = gd * silu
    dn = a * wd
    terms = (a * n).reshape(-1, D)
    return {
        "y": n * wd * silu,
        "dx": rstd * dn - xd * rstd**3 * (dn * xd).mean(dim=-1, keepdim=True),
        "dz": gd * n * wd * s * (1.0 + zd * (1.0 - s)),
        "dw": terms.sum(dim=0),
        "mag": terms.abs().sum(dim=0),
    }
