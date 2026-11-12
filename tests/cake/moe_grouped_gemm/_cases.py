"""Shapes, inputs, the FP32 reference and the accuracy budget of the MoE grouped GEMM's GPU tests (imports torch).

The shapes are the routed experts of miles' torchtitan MoE models on one expert-parallel rank, over imbalanced groups
with empty and one-row ones.
"""

import hashlib

import torch

from miles_ext_kernels.cake import moe_grouped_gemm

OP = "moe_grouped_gemm"
VARIANT = "cake_moe_grouped_gemm_variant"
SEED = 20260930
# rel RMS against the FP32 reference, per tensor. Each output is an FP32 accumulation of exact BF16 products rounded
# once to BF16, a relative RMS error of about 1.7e-3 (measured on sm_103a for y at K=7168, N=4096); the budget is twice
# that. Gated with detect_scale=1.02: y, dx or dw off by 2 % fails.
BUDGET = {"y": 3.4e-3, "dx": 3.4e-3, "dw": 3.4e-3}
# Group sizes, cycled over the experts: an empty group, a one-row group, sizes that are not tile multiples.
RAGGED = (300, 0, 1, 700, 4099, 33, 511, 257, 2048)
# (hidden size H, expert intermediate size I, local experts E): the routed experts of an MoE layer on one
# expert-parallel rank. Each gives two projections: w13 (gate and up, [E, 2I, H]) and w2 (down, [E, H, I]).
LAYERS = {
    "h2048-i768-e16": (2048, 768, 16),
    "h2048-i1536-e8": (2048, 1536, 8),
    "h4096-i2048-e9": (4096, 2048, 9),
}
CASES = [(layer, proj) for layer in LAYERS for proj in ("w13", "w2")]
# More than 4096 rows per group on average and K a multiple of 512: the weight gradient takes the 512-wide k tile,
# whose last wave splits its tail tiles (one of the vendored package's shapes whose tail splits on every SM count).
K512_SIZES = (16400, 0, 16666)
K512_SHAPE = (4864, 1024)  # (N, K)


def sizes_of(groups: int) -> list[int]:
    return [RAGGED[e % len(RAGGED)] for e in range(groups)]


def shape(layer: str, proj: str) -> tuple[list[int], int, int]:
    """``(group sizes, N, K)`` of one projection of a layer."""
    hidden, inter, groups = LAYERS[layer]
    n, k = (2 * inter, hidden) if proj == "w13" else (hidden, inter)
    return sizes_of(groups), n, k


def offs(sizes, device="cuda") -> torch.Tensor:
    return torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)


def inputs(sizes, n: int, k: int, seed: int, *, rows: int | None = None, device="cuda"):
    """``(x, w, g, offs)``; ``rows`` (default ``sum(sizes)``) may exceed the groups: the rows past ``offs[-1]``."""
    rows = sum(sizes) if rows is None else rows
    gen = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(rows, k, generator=gen, device=device) * k**-0.5).to(torch.bfloat16)
    w = (torch.randn(len(sizes), n, k, generator=gen, device=device) * k**-0.5).to(torch.bfloat16)
    g = torch.randn(rows, n, generator=gen, device=device).to(torch.bfloat16)
    return x, w, g, offs(sizes, device)


def bounds(sizes):
    ends = torch.tensor(sizes).cumsum(0).tolist()
    return list(zip([0, *ends[:-1]], ends))


def reference(x, w, g, sizes) -> dict:
    """Per-group FP32 products of the BF16 operands (TF32 off) over the rows of the groups."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        rows = sum(sizes)
        y = torch.zeros(rows, w.shape[1], device=x.device)
        dx = torch.zeros(rows, w.shape[2], device=x.device)
        dw = torch.zeros(w.shape, device=x.device)
        for e, (start, end) in enumerate(bounds(sizes)):
            if end > start:
                rows_x, rows_g, w_e = x[start:end].float(), g[start:end].float(), w[e].float()
                y[start:end] = rows_x @ w_e.T
                dx[start:end] = rows_g @ w_e
                dw[e] = rows_g.T @ rows_x
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return {"y": y, "dx": dx, "dw": dw}


def step(x, w, g, offs) -> dict:
    """One forward and backward of ``grouped_mm`` with the configured variant: ``{"y", "dx", "dw"}``."""
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    y = moe_grouped_gemm.grouped_mm(x, w, offs)
    y.backward(g)
    torch.cuda.synchronize()
    return {"y": y.detach(), "dx": x.grad, "dw": w.grad}


def valid(result: dict, sizes) -> dict:
    """The rows of the groups (the rows past ``offs[-1]`` cut off) and the whole weight gradient."""
    rows = sum(sizes)
    return {"y": result["y"][:rows], "dx": result["dx"][:rows], "dw": result["dw"]}


def assert_empty_groups_zero(dw, sizes) -> None:
    for e, size in enumerate(sizes):
        if size == 0:
            assert not dw[e].any(), f"the weight gradient of empty group {e} must be exactly zero"


def digest(result: dict) -> dict:
    """The sha256 of the bits of every tensor."""
    return {
        name: hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()
        for name, tensor in sorted(result.items())
    }
