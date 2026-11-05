"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Packed causal GQA attention, head dimension 256, forward and backward on the vendored ``cake_gqa_train`` package
(FlashInfer's ``flashinfer.gqa_train`` backend), SM100a / SM103a / SM107a.

Semantics. ``q [T_q, H_q, 256]``, ``k`` / ``v [T_k, H_kv, 256]`` (BF16, ``H_q / H_kv`` even), segments packed by
int32 device ``cu_seqlens_q`` / ``cu_seqlens_k`` (``cu[0] = 0``, non-decreasing, ``cu_q[B] = T_q``,
``cu_k[B] = T_k``). Query row ``i`` of a segment with ``q_len`` rows and ``k_len`` keys sees the keys
``j <= i + k_len - q_len`` (FlashAttention's bottom-right alignment); a row that sees no key gets a zero output and a
zero gradient.

``max_seqlen_k``. The backward main kernel runs one CTA pair per 128 keys of ``max_seqlen_k``, per key/value head and
segment, and a pair past its segment's last key block retires at once. A value below the longest segment's key count
leaves key blocks out: their dK / dV are never written and their dQ terms are missing, without an error. A larger value
is correct but launches pairs that only retire, which slows the backward down the more segments a call packs (``T_k``
is always correct, and the slowest). The kernels never read the boundaries on the host, so only the caller keeps this
contract; the option ``cake_gqa_check_seqlens`` (``--cake-gqa-check-seqlens``) checks it at every call, at the price
of one device synchronization.

Numerics. FP32 accumulation throughout, P and dS rounded to BF16 only as MMA operands. The output, the LSE and the BF16
dK / dV repeat bit for bit run to run, whatever valid ``max_seqlen_k`` is given; dQ is summed in FP32 with atomic adds,
one partial per 128-key block in a scheduling order, and rounded to BF16 once, so it is not reproducible bit for bit.

Memory. Each backward takes its scratch (FP32 ``[H_q / 2, rows_pad, 256]`` and two ``[H_q / 2, rows_pad]``,
``rows_pad = round_up(2 T_q + 128 B, 128)``) from the runtime's workspace pool on the current stream, and each forward
zeroes a fresh ``uint32`` tile counter; autograd saves ``(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)`` and scalars
only. Every launch is a plain launch on the current stream, and nothing synchronizes the device.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader, workspace
from miles_ext_kernels.cake._vendor.cake_gqa_train import cake_backend, cake_jit

__all__ = [
    "HEAD_DIM",
    "PACKAGE",
    "STAGES",
    "PoolWorkspace",
    "check_seqlens",
    "libraries",
    "prebuild",
    "release",
    "varlen_gqa_attention",
]

OP = "gqa"
PACKAGE = "cake_gqa_train"
HEAD_DIM = cake_backend.HEAD_DIM
# The stages one forward and backward of varlen_gqa_attention launch (BF16 dK / dV), in launch order.
STAGES = ("fwd", "bwd_pre", "bwd_bf16", "bwd_post")


def libraries(arch: str) -> list[str]:
    """The ``build.json`` libraries of :data:`STAGES` on ``arch``."""
    return [cake_backend.library_name(cake_backend.stage_record(stage, arch), arch) for stage in STAGES]


def prebuild(*, device: Any = None, max_workers: int | None = None) -> list[Path]:
    """Build the libraries of :data:`STAGES` for the architecture of ``device`` (default: the current one).

    No model parameter: every head layout, length and option runs these four libraries.
    """
    return loader.prebuild_libraries(PACKAGE, libraries(loader.device_arch(device)), max_workers=max_workers)


def release() -> None:
    """Nothing to drop here: the backward scratch lives in the runtime's workspace pool, the tile counter per call."""


class PoolWorkspace:
    """What the vendored launches ask their ``workspace`` for, holding nothing between calls.

    :meth:`tile_counter` is the forward's zeroed ``uint32`` ticket counter, new at every call (the kernel leaves it
    zero). :meth:`scratch` is the backward's ``(dq_accum, lse_log2, dpsum)``: views of one buffer of the workspace
    pool, key ``("gqa", "bwd_scratch")``, on the current stream.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def tile_counter(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.uint32, device=self.device)

    def scratch(self, total_q: int, num_heads_q: int, num_segments: int) -> tuple[torch.Tensor, ...]:
        nbytes = cake_jit.scratch_bytes(total_q, num_heads_q, num_segments)
        buffer = workspace.take(OP, "bwd_scratch", (nbytes,), torch.uint8, self.device)
        return tuple(
            buffer[offset : offset + 4 * math.prod(shape)].view(torch.float32).view(shape)
            for offset, shape in cake_jit.scratch_layout(total_q, num_heads_q, num_segments).values()
        )


def check_seqlens(
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, total_q: int, total_k: int, max_seqlen_k: int
) -> None:
    """``ValueError`` unless the call keeps the contract the kernels cannot check.

    Each boundary vector starts at 0, never decreases and ends at its token count, and no segment has more than
    ``max_seqlen_k`` keys. Reads both vectors on the host: one device synchronization.
    """
    split = cu_seqlens_q.numel()
    bounds = torch.cat((cu_seqlens_q.reshape(-1), cu_seqlens_k.reshape(-1))).tolist()
    for name, cu, total in (("cu_seqlens_q", bounds[:split], total_q), ("cu_seqlens_k", bounds[split:], total_k)):
        if not cu or cu[0] != 0 or cu[-1] != total or any(b < a for a, b in zip(cu, cu[1:])):
            shown = f"{cu[:8]}{' ...' if len(cu) > 8 else ''}"
            raise ValueError(f"{name} must start at 0, never decrease and end at {total}; got {shown}")
    keys = [b - a for a, b in zip(bounds[split:], bounds[split + 1 :])]
    longest = max(keys, default=0)
    if longest > max_seqlen_k:
        raise ValueError(
            f"max_seqlen_k is {max_seqlen_k}, but segment {keys.index(longest)} has {longest} keys: the backward "
            "would leave key blocks out and give wrong gradients"
        )


class _VarlenGQAAttention(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        out, lse = cake_backend.gqa_attention_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale=softmax_scale, workspace=PoolWorkspace(q.device)
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.max_seqlen_k = max_seqlen_k
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = cake_backend.gqa_attention_bwd(
            q,
            k,
            v,
            out,
            lse,
            dout.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale=ctx.softmax_scale,
            max_seqlen_k=ctx.max_seqlen_k,
            workspace=PoolWorkspace(q.device),
        )
        return dq, dk, dv, None, None, None, None


def varlen_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Packed causal GQA attention with autograd: BF16 ``out [T_q, H_q, 256]``; ``q``, ``k`` and ``v`` get BF16
    gradients.

    ``cu_seqlens_q`` / ``cu_seqlens_k``: int32 ``[B + 1]`` on the device, ``cu[0] = 0``, non-decreasing,
    ``cu_q[B] = T_q``, ``cu_k[B] = T_k``. ``max_seqlen_k`` (required): a host int at least every segment's key
    count, as FlashAttention's; a smaller value gives wrong gradients without an error (see the module documentation).
    ``softmax_scale``: default ``256 ** -0.5``.

    Raises ``ValueError`` for operands the kernels do not take (dtype, head dim, an odd query group, strides, the
    boundary vectors) and in a process configured for deterministic mode (there is no ``det`` variant),
    ``TypeError`` when ``max_seqlen_k`` is not an int, ``RuntimeError`` on another GPU. With the option
    ``cake_gqa_check_seqlens`` it also checks the boundaries and ``max_seqlen_k`` on the host (:func:`check_seqlens`).
    """
    config.variant(OP)  # "fast", the only variant; raises in a process configured for deterministic mode
    if isinstance(max_seqlen_k, bool) or not isinstance(max_seqlen_k, int):
        raise TypeError(
            f"max_seqlen_k must be a host int, at least every segment's key count; got {type(max_seqlen_k).__name__}"
        )
    if max_seqlen_k < 0:
        raise ValueError(f"max_seqlen_k must be >= 0, got {max_seqlen_k}")
    if config.option(OP, "cake_gqa_check_seqlens"):
        check_seqlens(cu_seqlens_q, cu_seqlens_k, q.shape[0], k.shape[0], max_seqlen_k)
    scale = HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    return _VarlenGQAAttention.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, scale, max_seqlen_k)
