"""One autograd step of ``miles_ext_kernels.cake.kda.chunk_kda`` on seeded inputs, in this process or, run as
``python -m cake.kda._steps VARIANT MODE L2N OUT`` (``tests/`` on ``PYTHONPATH``), in a fresh one that saves it to
``OUT``.

``MODE`` is the gate mode (``g``: the activated FP32 gate; ``raw``: the raw BF16 gate the kernels activate) and
``L2N`` (``0`` / ``1``) the in-kernel q / k L2 norm on raw q / k, as miles calls it (``raw`` and ``1``, with an FP32
``beta``).
"""

from __future__ import annotations

import sys

import torch

from miles_ext_kernels import cake
from miles_ext_kernels.cake import kda

from ._reference import LOWER_BOUND, make_raw_inputs, raw_gate
from ._reference_l2n import raw_qk

LENGTHS = (1000, 37, 300, 63)
HEADS = 12
LEAVES = ("q", "k", "v", "g", "beta")


def inputs(mode: str, l2n: bool, lengths=LENGTHS, heads: int = HEADS) -> dict:
    """The operands of a step on the device: raw q / k under the in-kernel L2 norm, the activated FP32 gate in the
    ``g`` mode, and miles' FP32 ``beta`` in its call (the BF16 values of the seeded inputs)."""
    t = make_raw_inputs(lengths, heads, "cuda")
    if l2n:
        t["q"], t["k"] = raw_qk(lengths, heads, "cuda")
        t["beta"] = t["beta"].float()
    if mode == "g":
        t["g"] = raw_gate(t["g"], t["A_log"], t["dt_bias"]).contiguous()
    return t


def autograd_step(variant: str, mode: str, l2n: bool, lengths=LENGTHS, heads: int = HEADS) -> dict:
    """``out`` and every gradient of one step under ``variant`` (configured here), on the host."""
    t = inputs(mode, l2n, lengths, heads)
    leaves = {n: t[n].clone().requires_grad_(True) for n in LEAVES}
    params = {n: t[n].clone().requires_grad_(True) for n in ("A_log", "dt_bias")} if mode == "raw" else {}
    modes = {"use_gate_in_kernel": True, "lower_bound": LOWER_BOUND, **params} if mode == "raw" else {}
    cake.configure(cake_kda_variant=variant)
    try:
        out = kda.chunk_kda(
            *(leaves[n].unsqueeze(0) for n in LEAVES), t["cu_seqlens"], use_qk_l2norm_in_kernel=l2n, **modes
        ).squeeze(0)
        out.backward(t["dout"])
    finally:
        cake.configure()
    result = {"out": out.detach(), **{"d" + n: x.grad for n, x in {**leaves, **params}.items()}}
    return {name: x.detach().cpu() for name, x in result.items()}


if __name__ == "__main__":
    variant, mode, l2n, path = sys.argv[1:5]
    torch.save(autograd_step(variant, mode, l2n == "1"), path)
