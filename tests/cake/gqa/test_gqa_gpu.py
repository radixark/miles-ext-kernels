"""Packed causal GQA attention (``miles_ext_kernels.cake.gqa``) on SM100a / SM103a / SM107a.

Against an FP64 reference of the same BF16 inputs, on FlashInfer's suite layouts (unequal lengths, tails that are no
multiple of a tile, keys fewer than queries, empty segments, long keys) and on the tensor-parallel-local head pairs
miles runs (8/2, 6/1, 16/2, 8/1) over a packed micro-batch whose last segment is its alignment pad: the output, the LSE
and every gradient within FlashInfer's declared budget and twice the measured error, rows that see no key exactly zero.
The output, the LSE and dK / dV repeat bit for bit run to run (in one process and in a new one), on the default and a
side stream, for any valid ``max_seqlen_k``; dQ is within its FP32 reordering. Also: the opt-in check of
``max_seqlen_k`` and the boundaries, document isolation, checkpointing, no device synchronization in steady state, the
pooled backward scratch and ``release``, the rejected operands, the backward's alert under torch's deterministic
algorithms, and two GPUs giving the same bits. Needs a B200 / GB200, B300 / GB300 or R200 and skips elsewhere.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from torch.utils.checkpoint import checkpoint  # noqa: E402

import miles_ext_kernels  # noqa: E402
from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import gqa  # noqa: E402
from miles_ext_kernels.cake._runtime import workspace  # noqa: E402
from miles_ext_kernels.cake._vendor.cake_gqa_nondet_train import cake_backend, cake_jit  # noqa: E402

from ..helpers import POOLED, assert_bitwise, assert_rel_rms  # noqa: E402
from . import _cases as c  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in cake.ARCH_BY_CAPABILITY:
    pytest.skip(
        f"the kernels run on {sorted(cake.ARCH_BY_CAPABILITY)}, not {torch.cuda.get_device_capability()}",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
MIB = 2**20

# FlashInfer's accuracy budget for this package: per layout and tensor, the relative L2 error against the FP64
# reference of the same BF16 inputs is at most 1.10 times FlashAttention CuTe's (its dedicated head-dim-256 backward)
# on B300 against the same reference.
FA4_REL_L2 = {"out": 2.3e-3, "lse": 1.5e-7, "dq": 2.6e-3, "dk": 2.6e-3, "dv": 2.4e-3}
# The largest error per tensor over LAYOUTS, measured with this package on a B200 (sm_100a, CUDA 13.0) from
# standard-normal BF16 inputs drawn on the device (other bits than the CPU draws here): out 2.34e-3 and dV 2.35e-3 on
# long_keys, the LSE 8.04e-8 on keys_fewer_than_queries, dQ 2.62e-3 and dK 2.58e-3 on empty_segments. The test prints
# the errors on the device it runs on (REL_RMS).
MEASURED_REL_L2 = {"out": 2.34e-3, "lse": 8.04e-8, "dq": 2.62e-3, "dk": 2.58e-3, "dv": 2.35e-3}
# FlashInfer's budget, capped at twice the measured error: out, dQ, dK and dV keep FlashInfer's (1.08 to 1.12 times the
# measured error); the LSE, whose errors (7.0e-8 to 8.0e-8 over the layouts) are at FP32 rounding (2^-24 = 6.0e-8),
# gets twice its measured error, 1.61e-7, below FlashInfer's 1.65e-7.
BUDGET = {name: min(1.10 * FA4_REL_L2[name], 2 * MEASURED_REL_L2[name]) for name in FA4_REL_L2}
# Checkpointing: memory kept after a non-reentrant checkpointed forward, less the output storage.
CKPT_RETAINED_LIMIT_MIB = 2.0

# (name, q_lens, k_lens, H_q, H_kv)
FI_LAYOUTS = (
    ("tails", (77, 300, 1, 129), (77, 700, 1000, 129), 24, 4),
    ("keys_fewer_than_queries", (200, 150, 64), (77, 0, 63), 24, 4),
    ("empty_segments", (0, 100, 0, 37, 0), (0, 300, 50, 37, 0), 24, 4),
    ("zigzag_cp2", (250, 250, 129, 130), (250, 1000, 129, 517), 24, 4),
    ("long_keys", (256, 255), (4099, 2049), 24, 4),
    ("group_2", (300, 77), (513, 77), 8, 4),
)
MILES_HEADS = ((8, 2), (6, 1), (16, 2), (8, 1))
MILES_LAYOUTS = tuple(
    (f"miles_{h_q}_{h_kv}", c.MILES_SEGMENTS, c.MILES_SEGMENTS, h_q, h_kv) for h_q, h_kv in MILES_HEADS
)
LAYOUTS = FI_LAYOUTS + MILES_LAYOUTS
# the reviewer's layout of the max_seqlen_k contract: 2048-key segments
EQUAL_SEGMENTS = (2048,) * 4


def _ids(layouts):
    return [layout[0] for layout in layouts]


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = torch.device("cuda", torch.cuda.current_device())
    # build every library of the forward and backward before the first checked call
    assert len(cake.prebuild("gqa", device=dev)) == 4
    return dev


@pytest.mark.parametrize("layout", LAYOUTS, ids=_ids(LAYOUTS))
def test_matches_the_fp64_reference(layout, device):
    name, q_lens, k_lens, h_q, h_kv = layout
    x = c.inputs(q_lens, k_lens, h_q, h_kv, device)
    ref = c.reference(x)
    # miles hands over the longest segment; FlashInfer's layouts run the T_k bound, always correct
    got = c.run(x, max_seqlen_k=None if name.startswith("miles") else sum(k_lens))
    torch.cuda.synchronize()
    finite = torch.isfinite(ref["lse"])
    errors = assert_rel_rms(
        {**got, "lse": got["lse"][finite]}, {**ref, "lse": ref["lse"][finite]}, BUDGET, detect_scale=1.02
    )
    print(f"REL_RMS {name} " + " ".join(f"{t}={e:.3e}" for t, e in errors.items()))
    # rows that see no key: out 0, lse -inf, dq 0, exactly
    no_key = c.no_key_rows(x)
    assert torch.equal(got["out"][no_key], torch.zeros_like(got["out"][no_key]))
    assert bool((got["lse"][:, no_key] == -float("inf")).all())
    assert bool((got["lse"][:, ~no_key] > -float("inf")).all())
    assert torch.equal(got["dq"][no_key], torch.zeros_like(got["dq"][no_key]))
    # keys no query sees (a segment without queries) get exactly zero gradients
    unseen = (ref["dk"].abs().amax(dim=(1, 2)) == 0) & (ref["dv"].abs().amax(dim=(1, 2)) == 0)
    assert torch.equal(got["dk"][unseen], torch.zeros_like(got["dk"][unseen]))
    assert torch.equal(got["dv"][unseen], torch.zeros_like(got["dv"][unseen]))


@pytest.mark.parametrize("layout", MILES_LAYOUTS, ids=_ids(MILES_LAYOUTS))
def test_out_lse_dk_dv_repeat_bit_for_bit_in_one_process(layout, device):
    _, q_lens, k_lens, h_q, h_kv = layout
    x = c.inputs(q_lens, k_lens, h_q, h_kv, device, seed=5)
    first, second = c.run(x), c.run(x)
    torch.cuda.synchronize()

    assert_bitwise(c.repeatable(first), c.repeatable(second))
    c.assert_dq_close(second["dq"], first["dq"])


_NEW_PROCESS = """
import json, sys
import torch
from cake.gqa import _cases as c
h_q, h_kv, seed = (int(arg) for arg in sys.argv[1:])
x = c.inputs(c.MILES_SEGMENTS, c.MILES_SEGMENTS, h_q, h_kv, torch.device("cuda"), seed=seed)
result = c.run(x)
torch.cuda.synchronize()
print(json.dumps(c.digests(result)))
"""


@pytest.mark.parametrize("heads", [(16, 2), (8, 1)], ids=["16_2", "8_1"])
def test_out_lse_dk_dv_are_the_bits_of_a_new_process(heads, device):
    h_q, h_kv = heads
    x = c.inputs(c.MILES_SEGMENTS, c.MILES_SEGMENTS, h_q, h_kv, device, seed=11)
    here = c.digests(c.run(x))
    paths = [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")]
    result = subprocess.run(
        [sys.executable, "-c", _NEW_PROCESS, str(h_q), str(h_kv), "11"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))},
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert result.returncode == 0, result.stderr[-5000:]
    assert json.loads(result.stdout.splitlines()[-1]) == here


@pytest.mark.parametrize("segments", [c.MILES_SEGMENTS, EQUAL_SEGMENTS], ids=["miles", "equal_2048"])
def test_any_valid_max_seqlen_k_gives_the_same_out_lse_dk_dv(segments, device):
    """``max_seqlen_k`` sizes the backward grid only: the exact bound, a larger one and ``T_k`` agree bit for bit."""
    x = c.inputs(segments, segments, 16, 2, device, seed=3)
    longest = c.longest_keys(x)
    exact = c.run(x, max_seqlen_k=longest)
    for bound in (longest + 1, 2 * longest, sum(segments)):
        other = c.run(x, max_seqlen_k=bound)
        torch.cuda.synchronize()
        assert_bitwise(c.repeatable(exact), c.repeatable(other))
        c.assert_dq_close(other["dq"], exact["dq"])


@pytest.mark.parametrize("bound", [1024, 1920, 2047])
def test_the_check_rejects_a_max_seqlen_k_below_the_longest_segment(bound, device):
    x = c.inputs(EQUAL_SEGMENTS, EQUAL_SEGMENTS, 16, 2, device)
    cake.configure(cake_gqa_check_seqlens=True)

    with pytest.raises(ValueError, match=f"max_seqlen_k is {bound}, but segment 0 has 2048 keys"):
        c.run(x, max_seqlen_k=bound)


def test_the_check_passes_a_valid_call_and_changes_no_bit(device):
    x = c.inputs(c.MILES_SEGMENTS, c.MILES_SEGMENTS, 8, 2, device, seed=7)
    plain = c.run(x)
    cake.configure(cake_gqa_check_seqlens=True)
    checked = c.run(x)
    torch.cuda.synchronize()

    assert_bitwise(c.repeatable(plain), c.repeatable(checked))


def test_documents_are_isolated(device):
    q_lens, k_lens = (130, 77, 300, 41), (500, 77, 301, 1000)
    x = c.inputs(q_lens, k_lens, 8, 2, device)
    base = c.run(x)
    again = c.run(x)
    y = dict(x)
    b = 2
    q0, q1, k0, k1 = x["cq"][b], x["cq"][b + 1], x["ck"][b], x["ck"][b + 1]
    for key, lo, hi in (("q", q0, q1), ("dout", q0, q1), ("k", k0, k1), ("v", k0, k1)):
        y[key] = x[key].clone()
        y[key][lo:hi] = -y[key][lo:hi] * 1.5
    pert = c.run(y)
    torch.cuda.synchronize()
    q_other = torch.ones(sum(q_lens), dtype=torch.bool, device=device)
    q_other[q0:q1] = False
    k_other = torch.ones(sum(k_lens), dtype=torch.bool, device=device)
    k_other[k0:k1] = False
    assert not torch.equal(pert["out"][~q_other], base["out"][~q_other])
    assert torch.equal(pert["out"][q_other], base["out"][q_other])
    assert torch.equal(pert["lse"][:, q_other], base["lse"][:, q_other])
    assert torch.equal(pert["dk"][k_other], base["dk"][k_other])
    assert torch.equal(pert["dv"][k_other], base["dv"][k_other])
    # dQ is reduced with atomic adds: the other segments agree to the reduction-order rounding of two plain runs
    c.assert_dq_close(again["dq"], base["dq"])
    c.assert_dq_close(pert["dq"][q_other], base["dq"][q_other])


@pytest.mark.parametrize("reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_checkpointing_matches_eager(reentrant, device):
    x = c.inputs((1000, 3000, 128), (1000, 3000, 128), 8, 2, device)

    def fn(q, k, v):
        return gqa.varlen_gqa_attention(q, k, v, x["cu_q"], x["cu_k"], max_seqlen_k=3000, softmax_scale=c.SCALE)

    def step(use_checkpoint):
        q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
        out = checkpoint(fn, q, k, v, use_reentrant=reentrant) if use_checkpoint else fn(q, k, v)
        out.backward(x["dout"])
        torch.cuda.synchronize()
        return out.detach(), q.grad, k.grad, v.grad

    if not reentrant:
        q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        out = checkpoint(fn, q, k, v, use_reentrant=False)
        torch.cuda.synchronize()
        kept = torch.cuda.memory_allocated() - before - out.untyped_storage().nbytes()
        assert kept / MIB <= CKPT_RETAINED_LIMIT_MIB, f"a checkpointed forward kept {kept / MIB:.2f} MiB"
        del out
    eager = step(False)
    ckpt = step(True)
    assert torch.equal(ckpt[0], eager[0])
    c.assert_dq_close(ckpt[1], eager[1])
    assert torch.equal(ckpt[2], eager[2]) and torch.equal(ckpt[3], eager[3])


def test_steady_state_does_not_synchronize_and_the_check_does(device):
    xs = [c.inputs(q, q, 8, 2, device, seed=i) for i, q in enumerate(((700, 77, 3), (300,)))]

    def step(x):
        q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
        out = gqa.varlen_gqa_attention(q, k, v, x["cu_q"], x["cu_k"], max_seqlen_k=700)
        out.backward(x["dout"])

    for x in xs:
        step(x)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for x in xs + xs:
            step(x)
        cake.configure(cake_gqa_check_seqlens=True)
        with pytest.raises(RuntimeError, match="synchroniz"):
            step(xs[0])
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


def test_rejects_unsupported_operands(device):
    x = c.inputs((10,), (10,), 24, 4, device)
    q, k, v, cq, ck = x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"]
    with pytest.raises(ValueError, match="even multiple"):
        gqa.varlen_gqa_attention(q[:, :12], k[:, :4], v[:, :4], cq, ck, max_seqlen_k=10)  # group 3
    with pytest.raises(ValueError, match="bfloat16"):
        gqa.varlen_gqa_attention(q[..., :128], k[..., :128], v[..., :128], cq, ck, max_seqlen_k=10)  # head dim 128
    with pytest.raises(ValueError, match="bfloat16"):
        gqa.varlen_gqa_attention(q.float(), k.float(), v.float(), cq, ck, max_seqlen_k=10)
    with pytest.raises(ValueError, match="int32"):
        gqa.varlen_gqa_attention(q, k, v, cq.long(), ck.long(), max_seqlen_k=10)
    assert not gqa.is_supported(12, 4) and gqa.is_supported(24, 4)


def test_the_backward_raises_the_deterministic_algorithms_alert(device):
    """No det variant serves the op: under ``torch.use_deterministic_algorithms(True)`` the backward raises PyTorch's
    alert (dQ is summed with atomics); with ``warn_only=True`` it warns and runs, giving the bits of a plain call."""
    x = c.inputs(c.MILES_SEGMENTS, c.MILES_SEGMENTS, 16, 2, device, seed=11)
    plain = c.run(x)
    torch.cuda.synchronize()
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="does not have a deterministic implementation"):
            c.run(x)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match="does not have a deterministic implementation"):
            warned = c.run(x)
        torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])

    assert_bitwise(c.repeatable(plain), c.repeatable(warned))
    c.assert_dq_close(warned["dq"], plain["dq"])


def test_default_and_side_stream_calls_agree_bit_for_bit(device, pooled):
    """The pool keeps one backward scratch per stream: alternating streams changes no bit of out / lse / dK / dV."""
    x = c.inputs(c.MILES_SEGMENTS, c.MILES_SEGMENTS, 16, 2, device, seed=9)
    default, side = torch.cuda.current_stream(device), torch.cuda.Stream(device)
    results = []
    for stream in (default, side, default, side):
        stream.wait_stream(default)
        with torch.cuda.stream(stream):
            results.append(c.run(x))
        torch.cuda.synchronize()

    for other in results[1:]:
        assert_bitwise(c.repeatable(results[0]), c.repeatable(other))
        c.assert_dq_close(other["dq"], results[0]["dq"])
    assert workspace.stats()[device.index]["entries"] == 2  # one ("gqa", "bwd_scratch") buffer per stream


def test_the_backward_scratch_comes_from_the_pool_and_release_frees_it(device, pooled):
    x = c.inputs(c.MILES_SEGMENTS, c.MILES_SEGMENTS, 16, 2, device)
    c.run(x)
    torch.cuda.synchronize()
    need = cake_jit.scratch_bytes(sum(c.MILES_SEGMENTS), 16, len(c.MILES_SEGMENTS))

    assert workspace.stats()[device.index]["bytes"] >= need
    # the vendored per-stream workspaces, kept for the life of the process, stay unused
    assert cake_backend._DEFAULT_WORKSPACES == {}
    assert miles_ext_kernels.release() >= need
    assert workspace.stats()[device.index]["bytes"] == 0


def test_a_pool_bound_of_zero_keeps_nothing_and_changes_no_bit(device):
    x = c.inputs(c.MILES_SEGMENTS, c.MILES_SEGMENTS, 8, 2, device, seed=4)
    cake.configure(**POOLED)
    pooled = c.run(x)
    cake.release()
    cake.configure(cake_workspace_pool_bytes=0)
    unpooled = c.run(x)
    torch.cuda.synchronize()

    assert workspace.stats()[device.index]["bytes"] == 0
    assert_bitwise(c.repeatable(pooled), c.repeatable(unpooled))


@pytest.mark.multi_gpu
def test_two_gpus_give_the_same_bits(device, pooled):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    first, second = torch.device("cuda", 0), torch.device("cuda", 1)
    models = {(p.name, p.multi_processor_count) for p in map(torch.cuda.get_device_properties, (first, second))}
    if len(models) != 1:
        pytest.skip(f"the two GPUs differ in model or SM count: {sorted(models)}")
    cake.prebuild("gqa", device=second)
    results = []
    # cuda:1 runs while cuda:0 is the current device: the launches follow the operands' device
    for dev in (first, second):
        x = c.inputs(c.MILES_SEGMENTS, c.MILES_SEGMENTS, 16, 2, dev, seed=13)
        results.append({name: tensor.cpu() for name, tensor in c.repeatable(c.run(x)).items()})
        torch.cuda.synchronize(dev)

    assert_bitwise(results[0], results[1])
    assert workspace.stats()[1]["entries"] == 1
