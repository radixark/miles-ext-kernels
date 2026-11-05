"""One rank of the context-parallel KDA GPU suite (``test_kda_kcp_gpu.py``), started by ``torch.multiprocessing``.

Every rank draws the whole packed batch from the same seeded generator, takes its shard (the tokens of its fragments in
the partition's order), runs each case's autograd step through ``miles_ext_kernels.cake.kda`` (twice for a case that
runs ``det``) and saves its outputs and gradients with their global token indices; the test process checks them
against the exact FP64 recurrence of the whole batch.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from miles_ext_kernels import cake
from miles_ext_kernels.cake import kda

from ._reference import LOWER_BOUND, make_raw_inputs, raw_gate
from ._reference_l2n import raw_qk

LENGTHS = (1000, 37, 300, 63, 700)
HEADS = 12
# name -> (variant, gate mode, in-kernel q / k L2 norm); "fast-l2n" is miles' call under fast, which runs det
CASES = {
    "det-miles": ("det", "raw", True),
    "det-g": ("det", "g", False),
    "fast-g": ("fast", "g", False),
    "fast-raw": ("fast", "raw", False),
    "fast-l2n": ("fast", "raw", True),
}
NAMES = ("q", "k", "v", "g", "beta")


def runs_det(case: str) -> bool:
    """Does ``case`` run the ``det`` package (so its results repeat bit for bit)?"""
    variant, _, l2n = CASES[case]
    return variant == "det" or l2n


def inputs(mode: str, l2n: bool, device="cpu") -> dict:
    """The whole batch of a case, on ``device``: raw q / k and miles' FP32 ``beta`` under the in-kernel L2 norm, the
    activated FP32 gate in the ``g`` mode."""
    tensors = make_raw_inputs(LENGTHS, HEADS, "cpu")
    if l2n:
        tensors["q"], tensors["k"] = raw_qk(LENGTHS, HEADS, "cpu")
        tensors["beta"] = tensors["beta"].float()
    if mode == "g":
        tensors["g"] = raw_gate(tensors["g"], tensors["A_log"], tensors["dt_bias"]).contiguous()
    return {name: t.to(device) for name, t in tensors.items()}


def shard_index(part, rank: int) -> torch.Tensor:
    """The global token indices of ``rank``'s shard, in the partition's order."""
    cu = part.cu_seqlens
    pieces = [torch.arange(cu[f.doc] + f.off, cu[f.doc] + f.off + f.length) for f in part.fragments[rank]]
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.long)


def _step(tensors: dict, idx: torch.Tensor, ctx, mode: str, l2n: bool, device) -> dict:
    leaves = {n: tensors[n][idx].to(device).requires_grad_(True) for n in NAMES}
    params = {n: tensors[n].to(device).requires_grad_(True) for n in ("A_log", "dt_bias")} if mode == "raw" else {}
    modes = {"use_gate_in_kernel": True, "lower_bound": LOWER_BOUND, **params} if mode == "raw" else {}
    operands = (leaves[n].unsqueeze(0) for n in NAMES)
    out = kda.chunk_kda_kcp(*operands, ctx, use_qk_l2norm_in_kernel=l2n, **modes).squeeze(0)
    out.backward(tensors["dout"][idx].to(device))
    result = {"out": out.detach()}
    result.update({"d" + n: leaf.grad for n, leaf in leaves.items()})
    if mode == "raw":
        result["dA_log"], result["ddt_bias"] = kda.reduce_gate_param_grads_kcp(
            ctx, params["A_log"].grad, params["dt_bias"].grad
        )
    torch.cuda.synchronize(device)
    return {name: t.detach().cpu() for name, t in result.items()}


def differing(a: dict, b: dict) -> list[str]:
    """The names whose tensors differ in shape, dtype or any bit."""
    differ = []
    for name in a:
        x, y = a[name].contiguous(), b[name].contiguous()
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x.view(torch.uint8), y.view(torch.uint8)):
            differ.append(name)
    return differ


def run(rank: int, world: int, port: int, layout: str, outdir: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, timeout=timedelta(minutes=20)
    )
    try:
        cu = [0]
        for n in LENGTHS:
            cu.append(cu[-1] + n)
        part = kda.partition_kcp(cu, world, layout)
        ctx = kda.build_context_kcp(part, dist.group.WORLD, rank, device, heads=HEADS)
        idx = shard_index(part, rank)
        saved = {"idx": idx, "cases": {}}
        for case, (variant, mode, l2n) in CASES.items():
            tensors = inputs(mode, l2n)
            cake.configure(cake_kda_variant=variant)
            first = _step(tensors, idx, ctx, mode, l2n, device)
            record = {"result": first}
            if runs_det(case):
                record["repeat_differs"] = differing(first, _step(tensors, idx, ctx, mode, l2n, device))
            if case == "fast-l2n":
                cake.configure(cake_kda_variant="det")
                record["det_differs"] = differing(first, _step(tensors, idx, ctx, mode, l2n, device))
            saved["cases"][case] = record
        torch.save(saved, os.path.join(outdir, f"rank{rank}.pt"))
        dist.barrier()
    finally:
        cake.configure()
        dist.destroy_process_group()
