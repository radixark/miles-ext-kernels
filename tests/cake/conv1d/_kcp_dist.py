"""One rank of the multi-GPU context-parallel short-conv test (``test_conv1d_kcp_dist.py``), started by
``torch.multiprocessing.spawn``: NCCL over ``world_size`` local GPUs, the contexts of one GDN / KDA context-parallel
route, and the checks of every row written to ``<result_dir>/rank<r>.json`` for the parent to assert."""

from __future__ import annotations

import json
import os
import socket
import types
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from miles_ext_kernels import cake
from miles_ext_kernels.cake._impl.conv1d import ops

from ._kcp import (
    DW_ATOL,
    DW_RTOL,
    WIDTH,
    Y_ATOL,
    Y_RTOL,
    bits_equal,
    context_builder,
    dweight_partial,
    local_tokens,
    outside,
    reference,
    split_pack,
)

SEED = 7815


def rows(world_size: int):
    """``(name, document lengths, layout, channels)``: the GDN / KDA routes' layouts and ragged tables whose
    fragments read history rows from one or two predecessor ranks (``ragged:<tokens per rank>``)."""
    ragged = {2: "ragged:2,6001", 4: "ragged:1,2,4997,1003"}[world_size]
    out = [
        ("contiguous", (2048, 5, 3000, 1), "contiguous", 384),
        ("zigzag", (3000,), "zigzag", 256),
        ("zigzag-per-document", (2048, 5, 3000, 1), "zigzag_per_document", 2560),
        ("ragged", (5000, 1003), ragged, 512),
    ]
    if world_size == 4:
        out.append(("empty-rank", (1200, 900, 900), "ragged:1200,0,900,900", 256))
    return out


def _run(x_local, w, dy_local, ctx, variant, **kwargs):
    cake.configure(cake_conv1d_variant=variant)
    xl = x_local.detach().clone().requires_grad_()
    wl = w.detach().clone().requires_grad_()
    y = ops.causal_conv1d_kcp(xl, wl, ctx, **kwargs)
    y.backward(dy_local)
    return y.detach(), xl.grad, wl.grad


def _same(a, b) -> bool:
    return all(bits_equal(u, v) for u, v in zip(a, b, strict=True))


def _row(partition, build_context, row, rank: int, world_size: int, device, flags: dict, stats: dict) -> None:
    name, lengths, layout, C = row
    part, cu = split_pack(partition, lengths, world_size, layout)
    ctx = build_context(part, dist.group.WORLD, rank, device, conv_kernel_size=WIDTH)
    g = torch.Generator(device="cpu").manual_seed(SEED + C)
    S = cu[-1]
    x = torch.randn(S, C, generator=g).bfloat16().to(device)
    w = (0.5 * torch.randn(C, WIDTH, generator=g)).to(device)
    dy = torch.randn(S, C, generator=g).bfloat16().to(device)
    index = torch.tensor(local_tokens(part, rank, cu), dtype=torch.int64, device=device)
    x_local, dy_local = x[index].contiguous(), dy[index].contiguous()

    def check(key, ok):
        flags[f"{name}/{key}"] = bool(ok)

    det = _run(x_local, w, dy_local, ctx, "det")
    with torch.no_grad():
        cu_device = torch.tensor(cu, dtype=torch.int32, device=device)
        y_single = ops.causal_conv1d(x, w, cu_device, silu_grad_preactivation="fp32")  # det, as configured
    check("det_y_is_the_single_gpu_forward", bits_equal(det[0], y_single[index]))
    y_ref, dx_ref, dpre, taps_inputs = reference(x, w, cu, dy)
    dw_ref = dweight_partial(dpre, taps_inputs, index)

    def within_reference(label, result):
        counts = (
            outside(result[0], y_ref[index], atol=Y_ATOL, rtol=Y_RTOL),
            outside(result[1], dx_ref[index], atol=Y_ATOL, rtol=Y_RTOL),
            outside(result[2], dw_ref, atol=DW_ATOL, rtol=DW_RTOL),
        )
        stats[f"{name}/{label}_outside_y_dx_dw"] = counts
        check(f"{label}_within_reference", counts == (0, 0, 0))

    within_reference("det", det)
    check("det_repeats_bitwise", _same(_run(x_local, w, dy_local, ctx, "det"), det))

    # det's default backward is the ordinary launch; the cooperative one keeps y and dx
    coop = _run(x_local, w, dy_local, ctx, "det", cooperative=True)
    check("det_cooperative_y_dx_are_det", bits_equal(coop[0], det[0]) and bits_equal(coop[1], det[1]))
    within_reference("det_cooperative", coop)
    check("det_cooperative_repeats_bitwise", _same(_run(x_local, w, dy_local, ctx, "det", cooperative=True), coop))

    fast = _run(x_local, w, dy_local, ctx, "fast")
    check("fast_y_dx_are_det", bits_equal(fast[0], det[0]) and bits_equal(fast[1], det[1]))
    within_reference("fast", fast)

    # two tensors of one layer with one exchange per direction: bitwise the single calls (det, as is the
    # checkpoint below)
    cake.configure(cake_conv1d_variant="det")
    half = C // 2
    xs = [x_local, x_local[:, :half].contiguous()]
    ws = [w, w[:half].contiguous()]
    dys = [dy_local, dy_local[:, :half].contiguous()]
    leaves_x = [t.detach().clone().requires_grad_() for t in xs]
    leaves_w = [t.detach().clone().requires_grad_() for t in ws]
    ys = ops.causal_conv1d_kcp_multi(leaves_x, leaves_w, ctx)
    torch.autograd.backward(list(ys), dys)
    singles = [det, _run(xs[1], ws[1], dys[1], ctx, "det")]
    check(
        "det_multi_is_the_single_calls",
        all(
            _same((y.detach(), lx.grad, lw.grad), single)
            for y, lx, lw, single in zip(ys, leaves_x, leaves_w, singles, strict=True)
        ),
    )

    # torch's checkpoint recompute with a handle reuses the forward's history rows: bitwise the direct pass
    xl = x_local.detach().clone().requires_grad_()
    wl = w.detach().clone().requires_grad_()
    handle = types.SimpleNamespace(entries={})
    y = checkpoint(
        lambda a, b, h: ops.causal_conv1d_kcp(a, b, ctx, recompute_handle=h),
        xl,
        wl,
        handle,
        use_reentrant=False,
    )
    y.backward(dy_local)
    check("det_checkpoint_with_handle_is_direct", _same((y.detach(), xl.grad, wl.grad), det))
    stats[f"{name}/T_r"] = int(x_local.shape[0])
    stats[f"{name}/n_hist_rows"] = int(ctx.halo.n_hist_rows)


def run_rank(rank: int, world_size: int, builder: str, port: int, result_dir: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    try:
        partition, build_context = context_builder(builder)
        flags: dict[str, bool] = {}
        stats: dict[str, object] = {}
        for row in rows(world_size):
            _row(partition, build_context, row, rank, world_size, device, flags, stats)
            torch.cuda.synchronize(device)
        dist.barrier()
        result = {
            "rank": rank,
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(device),
            "flags": flags,
            "stats": stats,
        }
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps(result, indent=1))
    finally:
        dist.destroy_process_group()
