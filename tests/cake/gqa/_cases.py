"""Inputs, the FP64 reference and runs of the gqa op, shared by its GPU tests and their child processes.

Inputs are drawn on the CPU from a seeded generator and copied to the device, so two processes get the same bits.
Imports torch.
"""

from __future__ import annotations

import hashlib
import math

import torch

from miles_ext_kernels.cake import gqa
from miles_ext_kernels.cake._impl.gqa import ops
from miles_ext_kernels.cake._vendor.cake_gqa_nondet_train import cake_backend

# the miles micro-batch, re-exported for the GPU tests
from ._chunks import MILES_SEGMENTS as MILES_SEGMENTS

D = 256
SCALE = D**-0.5  # the default, as miles runs it
# the tensors whose bits do not depend on the run: all of them (no sum uses atomics)
REPEATABLE = ("out", "lse", "dq", "dk", "dv")
# the tensors whose bits do not depend on max_seqlen_k either (it sets the workspace chunks, and with them the dQ
# rounding)
BOUND_INDEPENDENT = ("out", "lse", "dk", "dv")
# dQ of two valid max_seqlen_k of one input: one BF16 rounding step (DQ_RTOL) plus an absolute floor for the FP32
# reordering error where partials cancel, DQ_ATOL_FRACTION of the tensor's largest magnitude
DQ_RTOL = 2**-7
DQ_ATOL_FRACTION = 2**-16



def cu(lens) -> list[int]:
    bounds = [0]
    for n in lens:
        bounds.append(bounds[-1] + int(n))
    return bounds


def inputs(q_lens, k_lens, h_q, h_kv, device, seed=0) -> dict:
    """``q``, ``k``, ``v``, ``dout`` (BF16), the device boundaries ``cu_q`` / ``cu_k`` and their host copies."""
    g = torch.Generator().manual_seed(seed)
    t_q, t_k = sum(q_lens), sum(k_lens)

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)

    return {
        "q": randn(t_q, h_q, D),
        "k": randn(t_k, h_kv, D),
        "v": randn(t_k, h_kv, D),
        "dout": randn(t_q, h_q, D),
        "cu_q": torch.tensor(cu(q_lens), dtype=torch.int32, device=device),
        "cu_k": torch.tensor(cu(k_lens), dtype=torch.int32, device=device),
        "cq": cu(q_lens),
        "ck": cu(k_lens),
    }


def longest_keys(x) -> int:
    ck = x["ck"]
    return max((ck[i + 1] - ck[i] for i in range(len(ck) - 1)), default=0)


def run(x, *, max_seqlen_k: int | None = None, softmax_scale: float | None = SCALE) -> dict:
    """Forward and backward through autograd, plus the forward's LSE: ``{"out", "lse", "dq", "dk", "dv"}``.

    ``max_seqlen_k`` defaults to the longest key segment, as miles passes it.
    """
    bound = longest_keys(x) if max_seqlen_k is None else max_seqlen_k
    q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
    out = gqa.varlen_gqa_attention(q, k, v, x["cu_q"], x["cu_k"], max_seqlen_k=bound, softmax_scale=softmax_scale)
    out.backward(x["dout"])
    _, lse = cake_backend.cake_gqa_nondet_train_forward(
        x["q"],
        x["k"],
        x["v"],
        x["cu_q"],
        x["cu_k"],
        softmax_scale=SCALE if softmax_scale is None else softmax_scale,
        workspace=ops.PoolWorkspace(x["q"].device),
    )
    return {"out": out.detach(), "lse": lse, "dq": q.grad, "dk": k.grad, "dv": v.grad}


def repeatable(result: dict) -> dict:
    return {name: result[name] for name in REPEATABLE}


def bound_independent(result: dict) -> dict:
    return {name: result[name] for name in BOUND_INDEPENDENT}


def digests(result: dict) -> dict[str, str]:
    """sha256 of the bytes of every repeatable tensor."""
    return {
        name: hashlib.sha256(result[name].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name in REPEATABLE
    }


def assert_dq_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    """Two dQ of one input under two valid ``max_seqlen_k`` (see DQ_RTOL / DQ_ATOL_FRACTION)."""
    floor = DQ_ATOL_FRACTION * float(ref.float().abs().max()) if ref.numel() else 0.0
    torch.testing.assert_close(got, ref, rtol=DQ_RTOL, atol=floor)


def reference(x) -> dict:
    """FP64 out, lse and gradients of the same BF16 inputs (bottom-right causal per segment)."""
    q = x["q"].double().requires_grad_()
    k = x["k"].double().requires_grad_()
    v = x["v"].double().requires_grad_()
    group = q.shape[1] // k.shape[1]
    outs, lses = [], []
    for b in range(len(x["cq"]) - 1):
        q0, q1, k0, k1 = x["cq"][b], x["cq"][b + 1], x["ck"][b], x["ck"][b + 1]
        ql, kl = q1 - q0, k1 - k0
        ks = k[k0:k1].repeat_interleave(group, dim=1)
        vs = v[k0:k1].repeat_interleave(group, dim=1)
        s = torch.einsum("qhd,khd->hqk", q[q0:q1], ks) * SCALE
        seen = torch.arange(kl, device=q.device)[None, :] <= torch.arange(ql, device=q.device)[:, None] + (kl - ql)
        has_key = seen.any(dim=-1)
        # rows without a key take finite dummy scores (no NaN reaches the gradients) and a zero P
        s = s.masked_fill(~seen, -math.inf).masked_fill(~has_key[None, :, None], 0.0)
        lse = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - lse[..., None]) * seen
        outs.append(torch.einsum("hqk,khd->qhd", p, vs))
        lses.append(lse.detach().masked_fill(~has_key[None, :], -math.inf))
    out = torch.cat(outs)
    out.backward(x["dout"].double())
    return {"out": out.detach(), "lse": torch.cat(lses, dim=1), "dq": q.grad, "dk": k.grad, "dv": v.grad}


def no_key_rows(x) -> torch.Tensor:
    """Mask ``[T_q]`` of the query rows that see no key (``i + k_len - q_len < 0``)."""
    rows = []
    for b in range(len(x["cq"]) - 1):
        ql, kl = x["cq"][b + 1] - x["cq"][b], x["ck"][b + 1] - x["ck"][b]
        rows += [i + kl - ql < 0 for i in range(ql)]
    return torch.tensor(rows, dtype=torch.bool, device=x["q"].device)
