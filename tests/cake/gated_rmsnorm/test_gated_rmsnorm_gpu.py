"""The SiLU-gated RMSNorm kernels (``miles_ext_kernels.cake.gated_rmsnorm``) on the GPU, against an FP64 reference.

miles' GDN output norm (``inputs.py``): ``x`` the recurrence output ``[T, Hv, 128]``, the gate ``z`` a column slice of
the fused input projection. Every value-head count (12, 16, 32, 48) runs with a gate the kernels read in place and one
they copy (the projection width, which is the gate's token stride, a multiple of 16 elements or not), including the
16 key / 32 value-head layout of width 12352 and its unaligned twin of width 12360, at 1 to ~32k tokens: ``y``, ``dx``
and ``dz`` within one BF16 ulp of FP64 and the weight gradient within ``det``'s bound, each gate sensitive to a 2 %
error; ``det`` bitwise run to run in one process and in a new process; the in-place and the copied gate the same bits;
``fast`` within the same bounds, its ``y``, ``dx`` and ``dz`` bitwise run to run and ``det``'s bits where its program
is ``det``'s (the pipelined forwards' ``y``, the fused and the repeatable backwards' ``dx`` and ``dz``). Also calls
alternating between the default and a side stream, non-reentrant checkpointing, gradient accumulation, a head split as
tensor parallelism makes it (the caller sums the weight gradient) and the toolkit check. Needs SM100a /
SM103a / SM107a; skips elsewhere.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ..helpers import assert_bitwise, assert_repeatable, requires_arch
from ._shapes import packed_rows
from .inputs import fp64_reference, make, reads_in_place, run

torch = pytest.importorskip("torch")
pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")

REPO_ROOT = Path(__file__).resolve().parents[3]
EPS = 1e-6
BF16_RTOL = 2.0**-7  # one BF16 ulp relative to the value
DW_MAG_TOL = 2.0**-20  # the FP32 accumulation term of the weight-gradient bound, relative to sum |terms|
DETECT = 1.02  # every gate must reject an output 2 % off
# (key heads, value heads, extra projection columns, gate read in place): the projection width 2 * Hk * 128 +
# 2 * Hv * 128 + 2 * Hv (+ extra) is the gate's token stride, read in place when it is a multiple of 16 elements
LAYOUTS = (
    (6, 12, 0, False),
    (6, 12, 8, True),
    (8, 16, 0, True),
    (8, 16, 8, False),
    (16, 32, 0, True),  # width 12352
    (16, 32, 8, False),  # width 12360
    (24, 48, 0, True),
    (24, 48, 8, False),
)
TOKENS = (1, 7, 1031, 32771)
# the roles of the non-deterministic package whose programs are det's (its documentation): the pipelined forwards
# give det's y where det's gate does not flush (z above about -87.3; the inputs' z are N(0, 1)), and the fused and the
# repeatable backwards det's dx and dz
DET_Y_ROLES = ("fwd_pipe", "fwd_pipe_packed")
DET_GRAD_ROLES = ("bwd_bf16_fused", "bwd_bf16_rep", "bwd_bf16_rep_long")


def configure(variant: str) -> None:
    from miles_ext_kernels import cake

    cake.configure(cake_gated_rmsnorm_variant=variant)


def bf16_misses(got, want) -> int:
    """Elements of ``got`` beyond one BF16 ulp of ``want`` plus a floor of 1/64 ulp of its largest magnitude
    (cancellation)."""
    want = want.double()
    floor = want.abs().max().item() * BF16_RTOL / 64
    return int(((got.double() - want).abs() > want.abs() * BF16_RTOL + floor).sum())


def dw_misses(dw, want, mag, rtol_of=None) -> int:
    """Columns with ``|dw - want| > |rtol_of| * 2^-7 + sum|terms| * 2^-20`` (``rtol_of`` defaults to ``want``)."""
    rtol_of = want if rtol_of is None else rtol_of
    return int(((dw.double() - want).abs() > rtol_of.abs() * BF16_RTOL + mag * DW_MAG_TOL).sum())


def assert_accurate(got: dict, ref: dict, *, what: str) -> None:
    """``y``, ``dx`` and ``dz`` within one BF16 ulp, ``dw`` within ``det``'s bound; each gate fails a 2 % error."""
    for name in ("y", "dx", "dz"):
        assert bf16_misses(got[name], ref[name]) == 0, f"{what} {name}: beyond one BF16 ulp of FP64"
        assert bf16_misses(got[name].double() * DETECT, ref[name]) > 0, f"{what} {name}: gate too loose"
    assert dw_misses(got["dw"], ref["dw"], ref["mag"]) == 0, f"{what} dw: beyond det's bound"
    assert dw_misses(got["dw"].double() * DETECT, ref["dw"], ref["mag"]) > 0, f"{what} dw: gate too loose"


def pick(result: dict, names=("y", "dx", "dz")) -> dict:
    return {name: result[name] for name in names}


def det_bits(tokens: int, value_heads: int, in_place: bool) -> tuple[str, ...]:
    """The outputs of a ``fast`` call that are ``det``'s bits on this device: ``y`` where the non-deterministic
    package's forward runs a pipelined program, ``dx`` and ``dz`` where its backward runs the fused or the repeatable
    one (``cake_jit.module_for_rows`` on the device's SM count)."""
    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops
    from miles_ext_kernels.cake._runtime import loader

    jit = ops._package("fast").cake_jit
    arch = loader.device_arch()
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    rows = tokens * value_heads
    fwd_packed, bwd_packed = packed_rows(tokens, in_place)
    routes = jit.ROUTES[arch]
    names = ()
    forward = jit.module_for_rows(arch, "fwd", rows, sm_count, packed=fwd_packed)
    if forward in {routes.get(r) for r in DET_Y_ROLES}:
        names += ("y",)
    backward = jit.module_for_rows(arch, "bwd_bf16", rows, sm_count, packed=bwd_packed)
    if backward in {routes.get(r) for r in DET_GRAD_ROLES}:
        names += ("dx", "dz")
    return names


def _layout_id(layout) -> str:
    key_heads, value_heads, extra, in_place = layout
    return f"hk{key_heads}-hv{value_heads}-{'in_place' if in_place else 'copied'}"


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=_layout_id)
def test_det_and_fast_match_fp64_with_the_gate_in_place_or_copied(tokens, layout):
    key_heads, value_heads, extra, in_place = layout
    x, z, w, dy = make(tokens, key_heads, value_heads, extra)
    # with one token the token stride is never read, so the gate is read in place whatever the width
    assert reads_in_place(z) == (in_place or tokens == 1)
    ref = fp64_reference(dy, x, z, w, EPS)

    configure("det")
    det = assert_repeatable(lambda: run(x, z, w, dy), times=3)
    assert_accurate(det, ref, what="det")
    assert_bitwise(det, run(x, z.contiguous(), w, dy))  # a contiguous gate gives the same bits
    configure("fast")
    fast = run(x, z, w, dy)
    assert_accurate(fast, ref, what="fast")
    for _ in range(2):
        again = run(x, z, w, dy)
        assert_bitwise(pick(again), pick(fast))
        assert dw_misses(again["dw"], ref["dw"], ref["mag"]) == 0, "fast dw: beyond det's bound"
    same = det_bits(tokens, value_heads, in_place)
    assert_bitwise(pick(fast, same), pick(det, same))


_CHILD = """
import sys

import torch

import miles_ext_kernels
from cake.gated_rmsnorm import inputs

miles_ext_kernels.configure({"cake_gated_rmsnorm_variant": "det"})
result = inputs.run(*inputs.make(*map(int, sys.argv[1:5])))
torch.save({name: tensor.cpu() for name, tensor in result.items()}, sys.argv[5])
"""


@pytest.mark.parametrize("layout", [(32771, 16, 32, 0), (1031, 6, 12, 0)], ids=["in_place", "copied"])
def test_det_repeats_bit_for_bit_in_a_new_process(tmp_path, layout):
    configure("det")
    here = run(*make(*layout))
    out = tmp_path / "det.pt"
    paths = [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")]
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, *map(str, layout), str(out)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert_bitwise({name: tensor.cpu() for name, tensor in here.items()}, torch.load(out))


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_calls_alternating_between_the_default_and_a_side_stream_agree(variant):
    """Every kernel runs on the current stream (the non-deterministic package keeps a workspace per stream)."""
    x, z, w, dy = make(32771, 16, 32)
    configure(variant)
    side = torch.cuda.Stream()
    results = []
    for stream in (None, side, None, side):
        if stream is None:
            results.append(run(x, z, w, dy))
            continue
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results.append(run(x, z, w, dy))
        torch.cuda.current_stream().wait_stream(stream)

    names = ("y", "dx", "dz", "dw") if variant == "det" else ("y", "dx", "dz")
    for result in results[1:]:
        assert_bitwise(pick(results[0], names), pick(result, names))
    if variant == "fast":
        ref = fp64_reference(dy, x, z, w, EPS)
        assert [dw_misses(result["dw"], ref["dw"], ref["mag"]) for result in results] == [0] * len(results)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_non_reentrant_checkpoint_matches_the_plain_call(variant):
    x, z, w, dy = make(1031, 16, 32)
    configure(variant)
    plain = run(x, z, w, dy)
    checkpointed = run(x, z, w, dy, use_checkpoint=True)

    names = ("y", "dx", "dz", "dw") if variant == "det" else ("y", "dx", "dz")
    assert_bitwise(pick(plain, names), pick(checkpointed, names))
    if variant == "fast":
        ref = fp64_reference(dy, x, z, w, EPS)
        assert dw_misses(checkpointed["dw"], ref["dw"], ref["mag"]) == 0


def test_gradients_accumulate_across_micro_batches():
    from miles_ext_kernels.cake.gated_rmsnorm import rmsnorm_silu_gate

    x, z, w, dy1 = make(1031, 8, 16, seed=1)
    dy2 = make(1031, 8, 16, seed=2)[3]
    configure("det")
    xi, zi, wi = (t.detach().requires_grad_() for t in (x, z, w))
    for dy in (dy1, dy2):
        rmsnorm_silu_gate(xi, zi, wi, EPS).backward(dy)
    first, second = run(x, z, w, dy1), run(x, z, w, dy2)

    assert torch.equal(wi.grad, first["dw"] + second["dw"])
    assert torch.equal(xi.grad, first["dx"] + second["dx"])
    assert torch.equal(zi.grad, first["dz"] + second["dz"])


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_head_split_gives_each_rank_the_gradient_of_its_own_heads(variant):
    """Tensor parallelism gives each rank half the heads and the same weight: each rank's ``y``, ``dx`` and ``dz``
    are the whole call's bits, its weight gradient covers its own heads only, and the sum over the ranks (the
    caller's all-reduce) is the layer's."""
    x, z, w, dy = make(1031, 16, 32)
    configure(variant)
    whole = run(x, z, w, dy)
    ref = fp64_reference(dy, x, z, w, EPS)
    halves = []
    for heads in (slice(0, 16), slice(16, 32)):
        half = run(x[:, heads], z[:, heads], w, dy[:, heads])
        half_ref = fp64_reference(dy[:, heads], x[:, heads], z[:, heads], w, EPS)
        assert_bitwise(pick(half), {name: whole[name][:, heads] for name in ("y", "dx", "dz")})
        assert dw_misses(half["dw"], half_ref["dw"], half_ref["mag"]) == 0
        assert dw_misses(half["dw"], ref["dw"], ref["mag"]) > 0, "one rank's weight gradient is not the layer's"
        halves.append((half["dw"].double(), half_ref["dw"]))
    summed = halves[0][0] + halves[1][0]
    assert dw_misses(summed, ref["dw"], ref["mag"], rtol_of=halves[0][1].abs() + halves[1][1].abs()) == 0


def test_the_device_needs_its_architectures_toolkit(monkeypatch):
    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops
    from miles_ext_kernels.cake._runtime import loader

    ops.check_supported(128, "silu", torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(loader, "MIN_CUDA", {**loader.MIN_CUDA, "sm_107a": (99, 0)})
    with pytest.raises(RuntimeError, match=r"gated RMSNorm kernels for sm_107a needs CUDA 99\.0 or newer"):
        ops.check_supported(128, "silu", torch.bfloat16)
