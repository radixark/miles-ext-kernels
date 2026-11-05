"""
Copyright (c) 2026 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Independent references of the MoK layer, ported from FlashInfer at c5db8d3ccd3fa3842812280f7f208718dc0b8610:
``examples/mok_bf16_toy.py`` (the autograd reference of the nine outputs and the BF16 gate) and
``tests/experimental/_mok_reference.py`` (the MXFP8 weight recipe), with ``examples/mok_training_features.py``'s
MXFP8 bounds. Nothing here imports the package.

The reference runs every expert as a SwiGLU MLP with BF16 GEMMs (or FP32 ones for the MXFP8 bound): ``silu`` and the
product in FP32 on the GEMM outputs, rounded once to the input dtype, and the clamped form
``silu(min(gate, L)) * clamp(up, -L, L)`` whose clamp gradients vanish outside the clamp. With
``clamp_decision=torch.bfloat16`` the FP32 reference decides where they vanish on its GEMM outputs rounded to BF16, as
a BF16 data path decides it, and keeps its FP32 values. Each rank differentiates the score-weighted outputs of its own
experts over the global tokens; the partial outputs, input gradients and score gradients are summed over the ranks, and
the shared expert runs on the rank's own tokens.

The MXFP8 recipe: 32 consecutive elements along K share a UE8M0 byte ``round_up(max(amax * 0.002232142857, 1e-12))``
saturated to 254, the data is E4M3 ``x * 2**(127 - byte)`` (round to nearest even, saturating), and 128 x 128 scale
tiles store byte ``(row % 32) * 16 + (row // 32) * 4 + k_block`` as ``[R / 128, K / 128, 32, 16]``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

RESULT_NAMES = (
    "y",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)
# FlashInfer's acceptance of the BF16 path: every element within atol + rtol * |reference| of the BF16 reference
BF16_ATOL = BF16_RTOL = 1e-2
# FlashInfer's MXFP8 bounds: global relative L2 error against the FP32 reference of the routed-dependent outputs, of
# the BF16 shared-expert weight gradients, and the least distance of the MXFP8 output from the BF16 run's
MXFP8_REL_L2 = 0.15
SHARED_REL_L2 = 0.02
MXFP8_MIN_DISTANCE = 1e-3


def _decided(value, dtype):
    """The values the clamp's derivative is decided on: ``value``, or ``value`` rounded to ``dtype``."""
    return value if dtype is None else value.to(dtype)


class _ClampMax(torch.autograd.Function):
    """clamp(max=L) whose derivative is 1 where the input (rounded to ``decision``, when given) is <= L."""

    @staticmethod
    def forward(ctx, value, limit, decision=None):
        ctx.save_for_backward(value)
        ctx.limit, ctx.decision = limit, decision
        return value.clamp(max=limit)

    @staticmethod
    def backward(ctx, grad):
        (value,) = ctx.saved_tensors
        return grad * (_decided(value, ctx.decision) <= ctx.limit).to(grad.dtype), None, None


class _ClampBoth(torch.autograd.Function):
    """clamp(-L, L) whose derivative is 1 where -L <= input <= L (the input rounded to ``decision``, when given)."""

    @staticmethod
    def forward(ctx, value, limit, decision=None):
        ctx.save_for_backward(value)
        ctx.limit, ctx.decision = limit, decision
        return value.clamp(-limit, limit)

    @staticmethod
    def backward(ctx, grad):
        (value,) = ctx.saved_tensors
        value = _decided(value, ctx.decision)
        inside = (value >= -ctx.limit) & (value <= ctx.limit)
        return grad * inside.to(grad.dtype), None, None


def expert(x, gate, up, down, swiglu_limit=None, clamp_decision=None):
    a, b = (x @ gate.T).float(), (x @ up.T).float()
    if swiglu_limit is not None:
        a = _ClampMax.apply(a, swiglu_limit, clamp_decision)
        b = _ClampBoth.apply(b, swiglu_limit, clamp_decision)
    hidden = (torch.nn.functional.silu(a) * b).to(x.dtype)
    return hidden @ down.T


def reference(data, weights, *, fp32=False, source_counts, swiglu_limit=None, clamp_decision=None):
    """The nine outputs of this rank: ``y``, ``dx``, ``dscores``, the routed and the shared weight gradients.

    ``data`` holds the global ``x``, ``d_output``, ``expert_ids`` and ``scores`` (rank ``r`` owns the rows
    ``sum(source_counts[:r])`` onwards); ``weights`` this rank's shared gate / up / down and routed gate / up / down.
    ``clamp_decision``: the dtype the clamped SwiGLU's derivative is decided in (module docstring), default the GEMMs'.
    """
    rank, ep = dist.get_rank(), dist.get_world_size()
    device = weights[0].device
    dtype = torch.float32 if fp32 else torch.bfloat16
    x = data["x"].to(device=device, dtype=dtype)
    dy = data["d_output"].to(device=device, dtype=dtype)
    ids = data["expert_ids"].to(device)
    scores = data["scores"].to(device)
    t, h = x.shape
    if len(source_counts) != ep or sum(source_counts) != t:
        raise ValueError("source counts must cover the global input exactly")
    start = sum(source_counts[:rank])
    own_rows = slice(start, start + source_counts[rank])
    local_experts = weights[3].shape[0]
    y_sum = torch.zeros((t, h), dtype=torch.float32, device=device)
    dx_sum = torch.zeros_like(y_sum)
    dscores = torch.zeros_like(scores)
    dw = [torch.zeros_like(w, dtype=dtype) for w in weights[3:]]
    for e in range(local_experts):
        rows, slots = (ids == rank * local_experts + e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        xe = x[rows].detach().requires_grad_()
        we = [w[e].detach().to(dtype).requires_grad_() for w in weights[3:]]
        out = expert(xe, *we, swiglu_limit=swiglu_limit, clamp_decision=clamp_decision)
        # the scores are already scaled: differentiate them as they are
        se = scores[rows, slots].detach().requires_grad_()
        weighted = out.float() * se[:, None]
        grads = torch.autograd.grad(weighted, (xe, *we, se), dy[rows].float())
        y_sum.index_add_(0, rows, weighted.detach())
        dx_sum.index_add_(0, rows, grads[0].float())
        dscores[rows, slots] = grads[-1]
        for dest, grad in zip(dw, grads[1:4], strict=True):
            dest[e].copy_(grad)
    for tensor in (y_sum, dx_sum, dscores):
        dist.all_reduce(tensor)
    xs = x[own_rows].detach().requires_grad_()
    ws = [w.detach().to(dtype).requires_grad_() for w in weights[:3]]
    ys = expert(xs, *ws, swiglu_limit=swiglu_limit, clamp_decision=clamp_decision)
    shared_grads = torch.autograd.grad(ys, (xs, *ws), dy[own_rows])
    y = (y_sum[own_rows] + ys.detach().float()).to(dtype)
    dx = (dx_sum[own_rows] + shared_grads[0].float()).to(dtype)
    return (y, dx, dscores[own_rows], *dw, *shared_grads[1:])


def gate_failures(actual, expected, names=RESULT_NAMES) -> dict[str, str]:
    """This rank's outputs outside the BF16 gate (non-finite, or beyond ``atol + rtol * |reference|``)."""
    failures = {}
    for name, a, b in zip(names, actual, expected, strict=True):
        if a.shape != b.shape:
            failures[name] = f"shape {tuple(a.shape)} != {tuple(b.shape)}"
            continue
        af, bf = a.float(), b.float()
        allowed = BF16_ATOL + BF16_RTOL * bf.abs()
        bad = ~torch.isfinite(af) | ~torch.isfinite(bf) | ((af - bf).abs() > allowed)
        if bad.any():
            ratio = torch.nan_to_num((af - bf).abs() / allowed, nan=float("inf")).max().item()
            failures[name] = f"{int(bad.sum())} of {bad.numel()} beyond the gate (worst error / allowed {ratio:.3f})"
    return failures


def rel_l2(actual, expected) -> float:
    """Relative L2 error over every rank of the group."""
    stats = torch.zeros(2, dtype=torch.float64, device=actual.device)
    stats[0] = (actual.double() - expected.double()).square().sum()
    stats[1] = expected.double().square().sum()
    dist.all_reduce(stats)
    return (stats[0] / stats[1]).sqrt().item() if stats[1] > 0 else 0.0


def _e8m0_round_up(scale):
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return (exponent + (mantissa != 0).to(torch.int32)).clamp(max=254).to(torch.uint8)


def _quantize_rows(x):
    *lead, rows, cols = x.shape
    blocks = x.float().reshape(*lead, rows, cols // 32, 32)
    amax = blocks.abs().amax(-1)
    scale = torch.clamp(amax * torch.tensor(0.002232142857, dtype=torch.float32, device=x.device), min=1e-12)
    byte = _e8m0_round_up(scale)
    inverse = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    data = (blocks * inverse[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return data.reshape(*lead, rows, cols), byte


def _scale_tiles(byte):
    *lead, rows, kb = byte.shape
    tiles = byte.reshape(-1, rows // 128, 4, 32, kb // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return tiles.reshape(-1, kb // 4, 32, 16).contiguous()


def mxfp8_quantize(x):
    """``(fp8, scales, fp8_t, scales_t)`` of a BF16 ``[..., R, K]`` tensor."""
    data, byte = _quantize_rows(x)
    data_t, byte_t = _quantize_rows(x.transpose(-1, -2).contiguous())
    return data, _scale_tiles(byte), data_t, _scale_tiles(byte_t)


def routed_mxfp8_weights(gate, up, down):
    """The ``(forward_weights, backward_weights)`` the package's quantizer must write, byte for byte."""
    g, u, d = (mxfp8_quantize(w) for w in (gate, up, down))
    forward = ((g[0], g[1]), (u[0], u[1]), (d[0], d[1]))
    backward = (g, u, (d[2], d[3]))
    return forward, backward
