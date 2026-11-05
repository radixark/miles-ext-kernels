"""The prebuild guarantee of the KDA op: after ``cake.prebuild("kda", num_heads=12)`` no call compiles, whatever the
variant (configured before or after), the gate and L2 norm modes, the operand layouts, the token and document counts,
the path (autograd node, explicit forward and backward, saved set, replay, checkpoint) or the context-parallel entry
points. Each route of :data:`ROUTES` runs once per variant under ``no_build`` (``helpers.check_prebuild``).

Needs an SM100a / SM103a / SM107a GPU, torch and tvm-ffi. A cold cache builds every library of the architecture
(188 / 186 / 189 on SM100a / SM103a / SM107a): minutes, once per toolkit and machine.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import kda  # noqa: E402

from ..helpers import check_prebuild, no_build, requires_arch  # noqa: E402
from ._reference import LOWER_BOUND, cu_seqlens_of, make_raw_inputs, raw_gate  # noqa: E402
from ._reference_l2n import raw_qk  # noqa: E402

pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")

HEADS = 12
LIBRARIES = {"sm_100a": 188, "sm_103a": 186, "sm_107a": 189}
LEAVES = ("q", "k", "v", "g", "beta")


def _inputs(lengths, heads: int, mode: str, l2n: bool) -> dict:
    """The operands of a call: raw q / k and an FP32 ``beta`` (miles') under the in-kernel L2 norm, the activated FP32
    gate in the ``g`` mode."""
    t = make_raw_inputs(lengths, heads, "cuda")
    if l2n:
        t["q"], t["k"] = raw_qk(lengths, heads, "cuda")
        t["beta"] = t["beta"].float()
    if mode == "g":
        t["g"] = raw_gate(t["g"], t["A_log"], t["dt_bias"]).contiguous()
    return t


def _leaves(t: dict, mode: str, packed: bool) -> tuple[list, dict, dict]:
    """The operands as leaves (q / k / v column slices of one packed leaf with ``packed``), the raw gate's parameters
    and the call's gate keywords."""
    if packed:
        S, H, D = t["v"].shape
        qkv = torch.cat([t[n].reshape(S, H * D) for n in ("q", "k", "v")], 1).requires_grad_(True)
        qkv_views = [qkv[:, i * H * D : (i + 1) * H * D].view(S, H, D) for i in range(3)]
        operands = [*qkv_views, *(t[n].clone().requires_grad_(True) for n in ("g", "beta"))]
    else:
        operands = [t[n].clone().requires_grad_(True) for n in LEAVES]
    params = {n: t[n].clone().requires_grad_(True) for n in ("A_log", "dt_bias")} if mode == "raw" else {}
    modes = {"use_gate_in_kernel": True, "lower_bound": LOWER_BOUND, **params} if mode == "raw" else {}
    return operands, params, modes


def _autograd(lengths, mode: str = "raw", l2n: bool = True, *, heads: int = HEADS, packed=False, checkpoint=None):
    """One autograd step of the single-rank node (miles' call by default)."""
    t = _inputs(lengths, heads, mode, l2n)
    operands, params, modes = _leaves(t, mode, packed)

    def step(*tensors):
        batched = (x.unsqueeze(0) for x in tensors)
        return kda.chunk_kda(*batched, t["cu_seqlens"], use_qk_l2norm_in_kernel=l2n, **modes).squeeze(0)

    if checkpoint is None:
        out = step(*operands)
    else:
        out = torch.utils.checkpoint.checkpoint(step, *operands, use_reentrant=checkpoint == "reentrant")
    out.backward(t["dout"])


def _explicit(lengths, mode: str = "raw", l2n: bool = True, *, heads: int = HEADS, save: bool = True):
    """The explicit forward and backward: from the saved set, or (``save=False``) the forward without one and the
    replay backward."""
    t = _inputs(lengths, heads, mode, l2n)
    operands = [t[n].unsqueeze(0) for n in LEAVES]
    _, _, modes = _leaves(t, mode, False)
    modes = {**modes, "use_qk_l2norm_in_kernel": l2n}
    out, saved = kda.chunk_kda_fwd(*operands, t["cu_seqlens"], save_for_backward=save, **modes)
    assert (saved is None) != save
    kda.chunk_kda_bwd(*operands, t["cu_seqlens"], t["dout"].unsqueeze(0), saved=saved, **modes)


def _modes(run, *args, **kwargs):
    """``run`` in each of :data:`MODES`: miles' call (which ``fast`` runs as ``det``), and the two calls ``fast`` runs
    itself."""
    for mode, l2n in MODES:
        run(*args, mode, l2n, **kwargs)


def _segmented_states(variant: str):
    """One document at one head long enough that the forward saves the segmented route's segment states."""
    from miles_ext_kernels.cake._impl.kda import ops

    backend = ops.det_backend
    index = torch.cuda.current_device()
    segments = backend.plan(1, 1, 1, backend.cake_jit.sm_count(index), cake.device_arch(index)).NSEG
    lengths = (backend.SEG_SAVE_MIN_TOKENS_PER_SEG * segments + 37,)
    _modes(_autograd, lengths, heads=1)
    _modes(_explicit, lengths, heads=1)


def _token_sweep(variant: str):
    """Token and document counts across the packages' chunk, chain, slot and segment thresholds, empty documents
    included."""
    for lengths in [(1,), (0, 63, 0), (64, 65), (2047, 1), (5000, 0, 37), (8192,), (40000,), (300, 129, 500)]:
        _modes(_autograd, lengths)


def _kcp_one_rank(mode: str, l2n: bool, explicit: bool):
    """The context-parallel entry points on a one-rank partition (``group=None``): the shard is the whole batch."""
    lengths = (1000, 37, 300, 63, 700)
    t = _inputs(lengths, HEADS, mode, l2n)
    part = kda.partition_kcp(cu_seqlens_of(lengths), 1)
    ctx = kda.build_context_kcp(part, None, 0, torch.device("cuda", torch.cuda.current_device()), heads=HEADS)
    operands, params, modes = _leaves(t, mode, False)
    batched = [x.unsqueeze(0) for x in operands]
    if explicit:
        modes = {**modes, "use_qk_l2norm_in_kernel": l2n}
        out, saved = kda.chunk_kda_fwd_kcp(*batched, ctx, **modes)
        for set_ in (saved, None):  # the saved set, then the replay
            grads = kda.chunk_kda_bwd_kcp(*batched, ctx, t["dout"].unsqueeze(0), saved=set_, **modes)
        if mode == "raw":
            kda.reduce_gate_param_grads_kcp(ctx, grads[5], grads[6])
        return
    out = kda.chunk_kda_kcp(*batched, ctx, use_qk_l2norm_in_kernel=l2n, **modes)
    out.squeeze(0).backward(t["dout"])
    if mode == "raw":
        kda.reduce_gate_param_grads_kcp(ctx, params["A_log"].grad, params["dt_bias"].grad)


# (gate mode, in-kernel q / k L2 norm): miles' call, then the activated and the raw gate without the L2 norm
MODES = (("raw", True), ("g", False), ("raw", False))
# name -> route(variant): every call-time switch of the op; the single-rank entry takes device cu_seqlens, the
# context-parallel partition host ones
ROUTES = {
    "miles-packed-fp32-beta": lambda variant: _autograd((1000, 37, 300, 63), packed=True),
    "modes": lambda variant: _modes(_autograd, (1000, 37, 300, 63)),
    "g-l2n": lambda variant: _autograd((1000, 37, 300, 63), "g", True),
    "explicit-saved": lambda variant: _modes(_explicit, (300, 129, 500)),
    "explicit-replay": lambda variant: _modes(_explicit, (300, 129, 500), save=False),
    "checkpoint-reentrant": lambda variant: _modes(_autograd, (1100, 257), checkpoint="reentrant"),
    "checkpoint-non-reentrant": lambda variant: _modes(_autograd, (1100, 257), checkpoint="non_reentrant"),
    "checkpoint-packed": lambda variant: _autograd((1100, 257), packed=True, checkpoint="reentrant"),
    "segmented-h4": lambda variant: _modes(_autograd, (1000, 37, 300), heads=4),
    "segmented-states-h1": _segmented_states,
    "heads-1-64": lambda variant: [_modes(_autograd, (777, 65), heads=heads) for heads in (1, 2, 16, 24, 32, 48, 64)],
    "documents-over-256": lambda variant: _modes(_autograd, (7,) * 300 + (0,) * 5 + (129,)),
    "tokens": _token_sweep,
    "kcp": lambda variant: [_kcp_one_rank(mode, l2n, False) for mode, l2n in MODES],
    "kcp-explicit": lambda variant: [_kcp_one_rank(mode, l2n, True) for mode, l2n in MODES],
}


def test_prebuild_builds_every_library_of_the_architecture_for_any_head_count():
    workers = min(32, len(os.sched_getaffinity(0)))
    paths = cake.prebuild("kda", num_heads=HEADS, max_workers=workers)

    assert len(paths) == len(set(paths)) == LIBRARIES[cake.device_arch()]
    assert all(path.is_file() for path in paths)
    with no_build():
        for heads in (1, 64):
            assert cake.prebuild("kda", num_heads=heads) == paths


def test_no_route_builds_after_prebuild():
    check_prebuild("kda", ROUTES, num_heads=HEADS)

    assert "flashinfer" not in sys.modules
