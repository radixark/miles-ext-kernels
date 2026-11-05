"""
Copyright (c) 2026 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The exact FP64 reference of packed KDA chunked training with the in-kernel q / k L2 norm
(``use_qk_l2norm_in_kernel=True``) in the raw gate mode, ported from FlashInfer
``tests/test_helpers/cake_kda_det_train_reference_l2n.py`` (``kda_exact_raw_l2n_with_grads``) and the raw q / k draw
of ``tests/experimental/test_cake_kda_det_train_l2n.py`` at c5db8d3ccd3fa3842812280f7f208718dc0b8610.

The raw rows are normalised in FP64 (``x / sqrt(sum x^2 + 1e-6)``) inside the autograd graph of the exact
per-document recurrence, so ``dq`` / ``dk`` are the exact gradients of the raw rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ._reference import LOWER_BOUND, RAW_GRAD_NAMES, D, kda_exact_forward, raw_gate

EPS = 1e-6


def raw_qk(lengths: Sequence[int], H: int, device, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (not normalised) BF16 ``q`` / ``k`` ``[S, H, 128]`` of a few magnitudes per token, as projections give."""
    S = int(sum(lengths))
    gen = torch.Generator().manual_seed(7817 + seed + 31 * S + H)
    mag = torch.exp(torch.randn(S, H, 1, generator=gen) * 1.5)
    q = (torch.randn(S, H, D, generator=gen) * mag).to(torch.bfloat16)
    k = (torch.randn(S, H, D, generator=gen) * mag.flip(0)).to(torch.bfloat16)
    return q.to(device), k.to(device)


def _normalize64(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + EPS)


def kda_exact_raw_l2n_with_grads(
    q, k, v, raw, beta, A_log, dt_bias, dout, cu_seqlens, scale, lower_bound: float = LOWER_BOUND, dtype=torch.float64
):
    """The exact output and the seven gradients (``RAW_GRAD_NAMES`` order: ``dq, dk, dv, dg`` (the raw gate's),
    ``dbeta, dA_log, ddt_bias``) in ``dtype``, from raw ``q`` / ``k`` ``[S, H, D]`` normalised inside the graph,
    ``v``, the raw gate, ``beta`` ``[S, H]`` (any dtype: taken as exact ``dtype`` values), ``A_log`` ``[H]`` and
    ``dt_bias``; ``ddt_bias`` has ``dt_bias``'s shape."""
    leaves = [x.detach().to(dtype).requires_grad_(True) for x in (q, k, v, raw, beta, A_log, dt_bias)]
    q64, k64, v64, raw64, beta64, a64, b64 = leaves
    g64 = raw_gate(raw64, a64, b64, lower_bound, dtype)
    o = kda_exact_forward(_normalize64(q64), _normalize64(k64), v64, g64, beta64, cu_seqlens, scale, dtype=dtype)
    grads = torch.autograd.grad(o, leaves, dout.to(dtype))
    return o.detach(), dict(zip(RAW_GRAD_NAMES, (x.detach() for x in grads), strict=True))
