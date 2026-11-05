"""The context-parallel short conv (``causal_conv1d_kcp``) on one GPU: one rank holding every fragment of a cut pack.

The rank's halo exchange is the identity, so the history rows, the backward's head rows and their fold into the
predecessor fragments' ``dx`` all run through the route's host flow and kernels. Rows: history rows from two source
fragments of one and two tokens, an empty fragment, a head token with three consumers, documents of 1-3 tokens, a
one-token fragment and miles' channel counts. Per row: ``det``'s ``y`` bitwise the single-GPU ``det`` forward of the
whole pack; ``y`` / ``dx`` within one BF16 ulp and the weight gradient within the route's budget of the FP64
reference; ``det`` bitwise run to run (both launch forms of its backward), in one process and in a new one; ``fast``
with ``y`` / ``dx`` bitwise ``det``'s and the weight gradient within budget. Also: BF16 taps, the multi-tensor call,
the checkpoint recompute handle, ``fast`` under torch's deterministic algorithms, calls on a side stream interleaved
with calls on the default stream, and the contexts of the GDN / KDA context-parallel routes (when vendored). Needs
SM100a / SM103a / SM107a; skips elsewhere.
"""

import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
pytest.importorskip("tvm_ffi")

from torch.utils.checkpoint import checkpoint  # noqa: E402

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.conv1d import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402

from ._kcp import (  # noqa: E402
    DW_ATOL,
    DW_RTOL,
    VENDORED_BUILDERS,
    WIDTH,
    Y_ATOL,
    Y_RTOL,
    bits_equal,
    context_builder,
    dweight_partial,
    fragments,
    one_rank_context,
    outside,
    reference,
    token_index,
)

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake short-conv kernels need SM100a / SM103a / SM107a", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEVICE = torch.device("cuda", torch.cuda.current_device())
SEED = 6919
# (name, document lengths, cut lengths over the pack, channels)
ROWS = (
    ("two-sources", (5000, 1003), (1, 2, 4997, 1003), 512),
    ("empty-fragment", (5000, 1003), (1, 2, 0, 6000), 512),
    ("three-consumers", (1000, 0, 3, 3094), (1, 1, 1, 4094), 2048),
    ("short-docs", (1, 2, 3, 57), (3, 30, 30), 384),
    ("one-token-fragment", (65,), (64, 1), 256),
    ("miles-gdn", (1, 2, 3, 0, 9000, 3, 23762), (8000, 8000, 8000, 8771), 2560),
    ("wide", (5000, 7000), (1, 2, 4997, 7000), 8192),
)
ROW_IDS = [row[0] for row in ROWS]


def _problem(lens, cuts, C, seed=SEED):
    """The whole pack (``x``, ``w``, ``dy``, ``cu``), its fragments and the rank's context and local view."""
    S, cu, frags = fragments(lens, cuts)
    g = torch.Generator(device=DEVICE).manual_seed(seed + C)
    x = torch.randn(S, C, device=DEVICE, generator=g).bfloat16()
    w = (0.5 * torch.randn(C, WIDTH, device=DEVICE, generator=g)).contiguous()
    dy = torch.randn(S, C, device=DEVICE, generator=g).bfloat16()
    index = token_index(frags, cu, DEVICE)
    return types.SimpleNamespace(
        S=S,
        C=C,
        cu=cu,
        frags=frags,
        x=x,
        w=w,
        dy=dy,
        index=index,
        ctx=one_rank_context(frags, DEVICE),
        x_local=x[index].contiguous(),
        dy_local=dy[index].contiguous(),
    )


def _run(p, variant, *, ctx=None, w=None, **kwargs):
    """``(y, dx, dweight)`` of one autograd forward + backward of the rank's fragments, ``variant`` configured."""
    cake.configure(cake_conv1d_variant=variant)
    xl = p.x_local.detach().clone().requires_grad_()
    wl = (p.w if w is None else w).detach().clone().requires_grad_()
    y = ops.causal_conv1d_kcp(xl, wl, p.ctx if ctx is None else ctx, **kwargs)
    y.backward(p.dy_local)
    assert y.dtype == torch.bfloat16 and y.shape == xl.shape
    assert xl.grad.dtype == torch.bfloat16 and wl.grad.dtype == wl.dtype
    return y.detach(), xl.grad, wl.grad


def _same(a, b):
    return all(bits_equal(u, v) for u, v in zip(a, b, strict=True))


def _check_reference(p, result, label):
    y_ref, dx_ref, dpre, taps_inputs = reference(p.x, p.w, p.cu, p.dy)
    dw_ref = dweight_partial(dpre, taps_inputs, p.index)
    y, dx, dw = result
    assert outside(y, y_ref[p.index], atol=Y_ATOL, rtol=Y_RTOL) == 0, f"{label} y"
    assert outside(dx, dx_ref[p.index], atol=Y_ATOL, rtol=Y_RTOL) == 0, f"{label} dx"
    assert outside(dw, dw_ref, atol=DW_ATOL, rtol=DW_RTOL) == 0, f"{label} dweight"


@pytest.mark.parametrize("name, lens, cuts, C", ROWS, ids=ROW_IDS)
def test_det_matches_the_reference_and_repeats_bitwise(name, lens, cuts, C):
    p = _problem(lens, cuts, C)
    det = _run(p, "det")
    with torch.no_grad():
        cu = torch.tensor(p.cu, dtype=torch.int32, device=DEVICE)
        y_single = ops.causal_conv1d(p.x, p.w, cu, silu_grad_preactivation="fp32")  # det, as configured
    assert bits_equal(det[0], y_single[p.index]), "y differs from the single-GPU det forward"
    _check_reference(p, det, f"det {name}")
    for _ in range(2):
        assert _same(_run(p, "det"), det), "det differs between runs"


@pytest.mark.parametrize("name, lens, cuts, C", ROWS, ids=ROW_IDS)
def test_det_split_backward_matches_the_reference_and_repeats_bitwise(name, lens, cuts, C):
    p = _problem(lens, cuts, C)
    cooperative = _run(p, "det")
    split = _run(p, "det", cooperative=False)
    assert bits_equal(split[0], cooperative[0]) and bits_equal(split[1], cooperative[1])
    _check_reference(p, split, f"det split {name}")
    assert _same(_run(p, "det", cooperative=False), split), "the split form differs between runs"


@pytest.mark.parametrize("name, lens, cuts, C", ROWS, ids=ROW_IDS)
def test_fast_keeps_det_y_and_dx_and_the_weight_gradient_budget(name, lens, cuts, C):
    p = _problem(lens, cuts, C)
    det = _run(p, "det")
    fast = _run(p, "fast")
    assert bits_equal(fast[0], det[0]), "fast y differs from det"
    assert bits_equal(fast[1], det[1]), "fast dx differs from det"
    _check_reference(p, fast, f"fast {name}")
    assert outside(fast[2], det[2], atol=DW_ATOL, rtol=DW_RTOL) == 0, "fast dweight outside det's budget"


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_bf16_taps_run_as_the_widened_fp32_taps(variant):
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512)
    taps = p.w.bfloat16()
    bf16 = _run(p, variant, w=taps)
    fp32 = _run(p, variant, w=taps.float())
    assert bits_equal(bf16[0], fp32[0]) and bits_equal(bf16[1], fp32[1])
    assert bf16[2].dtype == torch.bfloat16
    if variant == "det":
        assert bits_equal(bf16[2], fp32[2].bfloat16())
    else:
        assert outside(bf16[2], fp32[2], atol=DW_ATOL * (1 + 2**-8), rtol=2**-7) == 0


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_the_multi_call_is_the_single_calls(variant):
    p = _problem((1000, 0, 3, 3094), (1, 1, 1, 4094), 2048)
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    xs = [p.x_local, torch.randn(p.S, 1024, device=DEVICE, generator=g).bfloat16(), p.x_local[:, :512].contiguous()]
    ws = [p.w, 0.5 * torch.randn(1024, WIDTH, device=DEVICE, generator=g), p.w[:512].contiguous()]
    dys = [torch.randn_like(x) for x in xs]
    cake.configure(cake_conv1d_variant=variant)

    def run(multi):
        leaves_x = [x.detach().clone().requires_grad_() for x in xs]
        leaves_w = [w.detach().clone().requires_grad_() for w in ws]
        if multi:
            ys = ops.causal_conv1d_kcp_multi(leaves_x, leaves_w, p.ctx)
        else:
            ys = [ops.causal_conv1d_kcp(x, w, p.ctx) for x, w in zip(leaves_x, leaves_w)]
        torch.autograd.backward(list(ys), dys)
        return [(y.detach(), x.grad, w.grad) for y, x, w in zip(ys, leaves_x, leaves_w)]

    for one, single in zip(run(True), run(False), strict=True):
        assert bits_equal(one[0], single[0]) and bits_equal(one[1], single[1])
        if variant == "det":
            assert bits_equal(one[2], single[2])
        else:
            assert outside(one[2], single[2], atol=DW_ATOL, rtol=DW_RTOL) == 0


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_checkpoint_recompute_with_a_handle_is_bitwise_the_plain_one(variant):
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512)
    plain = _run(p, variant)  # configures variant for the checkpointed calls too

    def layer(x, w, handle):
        return ops.causal_conv1d_kcp(x, w, p.ctx, recompute_handle=handle)

    xl = p.x_local.detach().clone().requires_grad_()
    wl = p.w.detach().clone().requires_grad_()
    handle = types.SimpleNamespace(entries={})
    y = checkpoint(layer, xl, wl, handle, use_reentrant=False)
    assert "conv" in handle.entries
    y.backward(p.dy_local)
    assert bits_equal(y.detach(), plain[0]) and bits_equal(xl.grad, plain[1])
    if variant == "det":
        assert bits_equal(wl.grad, plain[2])
    else:
        assert outside(wl.grad, plain[2], atol=DW_ATOL, rtol=DW_RTOL) == 0


def test_fast_runs_det_under_torch_deterministic_algorithms(monkeypatch, caplog):
    monkeypatch.setattr(config, "_warned", set())
    p = _problem((1000, 0, 3, 3094), (1, 1, 1, 4094), 2048)
    det = _run(p, "det")
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            runs_det = _run(p, "fast")
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    assert _same(runs_det, det)
    assert any("conv1d" in r.getMessage() and "running det" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("builder", VENDORED_BUILDERS)
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_the_gdn_and_kda_contexts_serve_the_short_conv(builder, variant):
    found = context_builder(builder)
    if found is None:
        pytest.skip(f"this checkout does not vendor the {builder} context-parallel package")
    partition, build_context = found
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512)
    part = partition(list(p.cu), 1, fragments=[[f for f in p.frags if f[2]]])
    ctx = build_context(part, None, 0, DEVICE, conv_kernel_size=WIDTH)
    assert int(ctx.T_r) == p.S and int(ctx.n_frag) == p.ctx.n_frag
    own = _run(p, variant, ctx=ctx)
    stand_in = _run(p, variant)
    assert bits_equal(own[0], stand_in[0]) and bits_equal(own[1], stand_in[1])
    if variant == "det":
        assert bits_equal(own[2], stand_in[2])
    _check_reference(p, own, f"{builder} context")


def test_a_context_of_another_token_count_is_rejected():
    p = _problem((65,), (64, 1), 256)
    other = one_rank_context(fragments((70,), (64, 6))[2], DEVICE)
    with pytest.raises(ValueError, match="T_r"):
        ops.causal_conv1d_kcp(p.x_local, p.w, other)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_side_stream_calls_give_the_default_stream_bits(variant):
    """Calls alternating between the default stream and a side stream, both in flight at once: ``det`` gives the
    same bits on both (in both launch forms), ``fast`` the same ``y`` and ``dx`` and a weight gradient within
    budget."""
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    forms = [{}, {"cooperative": False}] if variant == "det" else [{}]
    for form in forms:
        results = []
        for on_side in (False, True, False, True):
            if on_side:
                with torch.cuda.stream(side):
                    results.append(_run(p, variant, **form))
            else:
                results.append(_run(p, variant, **form))
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for result in results[1:]:
            assert bits_equal(result[0], results[0][0]) and bits_equal(result[1], results[0][1]), form
            if variant == "det":
                assert bits_equal(result[2], results[0][2]), form
            else:
                assert outside(result[2], results[0][2], atol=DW_ATOL, rtol=DW_RTOL) == 0


# (row name, launch form) of the det runs a new process repeats
XPROC_RUNS = (("two-sources", None), ("two-sources", False), ("miles-gdn", None))
_XPROC = """
import sys
import torch
sys.path.insert(0, {tests!r})
from cake.conv1d import test_conv1d_kcp_gpu as t
rows = {{row[0]: row for row in t.ROWS}}
out = {{}}
for key, (name, cooperative) in enumerate(t.XPROC_RUNS):
    _, lens, cuts, C = rows[name]
    out[key] = [v.cpu() for v in t._run(t._problem(lens, cuts, C), "det", cooperative=cooperative)]
torch.save(out, {path!r})
"""


def test_det_repeats_bitwise_in_a_new_process(tmp_path):
    path = tmp_path / "det_kcp.pt"
    script = _XPROC.format(tests=str(REPO_ROOT / "tests"), path=str(path))
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": pythonpath}, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]
    child = torch.load(path)
    rows = {row[0]: row for row in ROWS}
    for key, (name, cooperative) in enumerate(XPROC_RUNS):
        _, lens, cuts, C = rows[name]
        here = _run(_problem(lens, cuts, C), "det", cooperative=cooperative)
        for a, b in zip(here, child[key], strict=True):
            assert bits_equal(a.cpu(), b), (name, cooperative)
