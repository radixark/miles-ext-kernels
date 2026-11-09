"""The ``det`` KDA kernels as miles calls them (raw gate, in-kernel q / k L2 norm, q / k / v column slices of one
packed projection, FP32 ``beta``, ``[1, S, H, 128]`` operands) against the exact FP64 recurrence, bit-identical
repeats (in one process, on another stream, in a fresh process), the backward routes, and flash-linear-attention.

Shapes: ``_budgets.SHAPES``. The module first runs ``cake.prebuild("kda", num_heads=12)``: every library of the
architecture (cached per toolkit and machine). Needs an SM100a / SM103a / SM107a GPU, torch and tvm-ffi; the fla
parity rows also need flash-linear-attention, and skip without it. ``-rP`` prints the fla parity tables.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import kda  # noqa: E402

from ..helpers import assert_bitwise, assert_rel_rms, assert_repeatable, rel_rms, requires_arch  # noqa: E402
from . import _budgets, _steps  # noqa: E402
from ._reference import LOWER_BOUND, RAW_GRAD_NAMES, D, gate_param_terms, make_raw_inputs  # noqa: E402
from ._reference_l2n import kda_exact_raw_l2n_with_grads, raw_qk  # noqa: E402

pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")

TESTS = Path(__file__).resolve().parents[2]
SHAPES = _budgets.SHAPES
VIEW_SHAPES = [((1000, 37, 300), 4), ((1100, 257), 24), ((1000, 37, 300, 63), 64)]
NAMES = ("out",) + RAW_GRAD_NAMES  # out, dq, dk, dv, dg (the raw gate's), dbeta, dA_log, ddt_bias
BWD_NAMES = ("dq", "dk", "dv", "dbeta", "dg", "dA_log", "ddt_bias")  # the order chunk_kda_bwd returns them in
# The miles call's keywords besides the operands.
MILES_MODES = {"use_gate_in_kernel": True, "use_qk_l2norm_in_kernel": True}


@pytest.fixture(scope="module", autouse=True)
def prebuilt():
    """Every library of the device's architecture, built once for the module."""
    cake.prebuild("kda", num_heads=12, max_workers=min(32, len(os.sched_getaffinity(0))))


def _det_backend():
    from miles_ext_kernels.cake._impl.kda import ops

    return ops.det_backend


def _device_kind() -> tuple[str, int]:
    backend = _det_backend()
    index = torch.cuda.current_device()
    return cake.device_arch(index), backend.cake_jit.sm_count(index)


def _inputs(lengths, heads: int) -> dict:
    """The raw-gate operands with raw (not normalised) q / k and the inputs' BF16 ``beta``."""
    inputs = make_raw_inputs(lengths, heads, "cuda")
    inputs["q"], inputs["k"] = raw_qk(lengths, heads, "cuda")
    return inputs


def _cake(q, k, v, g, beta, A_log, dt_bias, cu_seqlens):
    """miles' call: ``[1, S, H, 128]`` operands, the raw gate with lower bound -5, the in-kernel q / k L2 norm."""
    out = kda.chunk_kda(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu_seqlens,
        **MILES_MODES,
    )
    assert out.shape == (1, *v.shape) and out.dtype == torch.bfloat16
    return out.squeeze(0)


def _run(inputs: dict, *, packed: bool = False, checkpoint: str | None = None, beta=None, kernel=_cake) -> dict:
    """``out`` and the gradients (``NAMES``) of one autograd step of ``kernel``.

    ``packed``: q / k / v are column slices of one ``[S, 3 * H * 128]`` leaf (strided views, as miles hands them over).
    ``checkpoint``: under ``torch.utils.checkpoint`` (``"reentrant"`` as Megatron's recompute, or ``"non_reentrant"``).
    ``beta`` replaces the inputs' BF16 beta.
    """
    S, H, _ = inputs["v"].shape
    width = H * D
    cu_seqlens = inputs["cu_seqlens"]
    if packed:
        qkv = torch.cat([inputs[n].reshape(S, width) for n in ("q", "k", "v")], 1).requires_grad_(True)
        leaves = [qkv]

        def split(qkv):
            return tuple(qkv[:, i * width : (i + 1) * width].view(S, H, D) for i in range(3))

    else:
        leaves = [inputs[n].clone().requires_grad_(True) for n in ("q", "k", "v")]

        def split(q, k, v):
            return q, k, v

    beta = inputs["beta"] if beta is None else beta
    params = [x.clone().requires_grad_(True) for x in (inputs["g"], beta, inputs["A_log"], inputs["dt_bias"])]

    def step(*tensors):
        return kernel(*split(*tensors[: len(leaves)]), *tensors[len(leaves) :], cu_seqlens)

    if checkpoint is None:
        out = step(*leaves, *params)
    else:
        out = torch.utils.checkpoint.checkpoint(step, *leaves, *params, use_reentrant=checkpoint == "reentrant")
    out.backward(inputs["dout"])
    if packed:
        dq, dk, dv = (qkv.grad[:, i * width : (i + 1) * width].reshape(S, H, D) for i in range(3))
    else:
        dq, dk, dv = (leaf.grad for leaf in leaves)
    g, beta, A_log, dt_bias = (param.grad for param in params)
    result = {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": g, "dbeta": beta, "dA_log": A_log, "ddt_bias": dt_bias}
    return {name: tensor.detach() for name, tensor in result.items()}


def _reference(inputs: dict, beta=None) -> dict:
    out, grads = kda_exact_raw_l2n_with_grads(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"] if beta is None else beta,
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["dout"],
        inputs["cu_seqlens"],
        kda.DEFAULT_SCALE,
        LOWER_BOUND,
    )
    return {"out": out, **grads}


def test_head_counts_outside_1_to_64_and_batches_are_rejected():
    x = torch.zeros(1, 3, 65, D, dtype=torch.bfloat16, device="cuda")
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="serve 1 to 64 heads per call"):
        kda.chunk_kda(x, x, x, x, x[..., 0], cu_seqlens)
    inputs = _inputs((5, 5), 2)
    two = [inputs[n].reshape(2, 5, *inputs[n].shape[1:]) for n in ("q", "k", "v", "g", "beta")]
    with pytest.raises(ValueError, match="batch axis must be 1"):
        kda.chunk_kda(
            *two,
            cu_seqlens=inputs["cu_seqlens"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            lower_bound=LOWER_BOUND,
            **MILES_MODES,
        )


def _split(result: dict) -> tuple[dict, dict]:
    """The token tensors and the gate parameters' gradients of a result."""
    tokens = {name: result[name] for name in _budgets.TOKEN_TENSORS}
    return tokens, {name: result[name] for name in _budgets.PARAM_TENSORS}


@pytest.mark.parametrize("shape", SHAPES, ids=_budgets.shape_id)
def test_matches_the_exact_fp64_reference(shape):
    """Every output and gradient within its budget (``_budgets``: twice the largest error measured): the token tensors
    in relative RMS error, each but the loose ``dg`` failing it when scaled by 1.02; ``dA_log`` / ``ddt_bias`` in
    error over the size of their terms (``_budgets.accumulation_error``)."""
    lengths, heads = shape
    inputs = _inputs(lengths, heads)
    got = _run(inputs, packed=True)
    ref = _reference(inputs)
    terms = gate_param_terms(inputs["g"], inputs["dt_bias"], ref["dg"])
    tokens, _ = _split(got)
    measured = assert_rel_rms(tokens, ref, _budgets.budget(), loose=_budgets.loose(tokens))
    measured.update(_budgets.assert_param_grads(got, ref, terms))
    print(f"\n{_budgets.shape_id(shape)} {cake.device_arch()}: {measured}")


@pytest.mark.parametrize("shape", SHAPES, ids=_budgets.shape_id)
def test_repeated_steps_are_bitwise_identical(shape):
    lengths, heads = shape
    inputs = _inputs(lengths, heads)
    assert_repeatable(lambda: _run(inputs, packed=True), times=3)


@pytest.mark.parametrize("shape", VIEW_SHAPES, ids=_budgets.shape_id)
def test_packed_views_and_an_fp32_beta_give_the_bits_of_contiguous_bf16_operands(shape):
    """miles' operands (q / k / v column slices, the FP32 ``beta`` of the sigmoid) give the bits of contiguous copies
    and of a BF16 ``beta`` of the same values; ``dbeta`` is the same gradient in the caller's dtype."""
    lengths, heads = shape
    inputs = _inputs(lengths, heads)
    contiguous = _run(inputs, packed=False)
    assert_bitwise(_run(inputs, packed=True), contiguous)
    miles = _run(inputs, packed=True, beta=inputs["beta"].float())
    assert miles["dbeta"].dtype == torch.float32
    assert_bitwise(miles, {**contiguous, "dbeta": contiguous["dbeta"].float()})


def _states_shape() -> tuple[tuple[int, ...], int]:
    """One document at one head long enough that the forward saves the segmented route's segment states."""
    backend = _det_backend()
    arch, sm_count = _device_kind()
    segments = backend.plan(1, 1, 1, sm_count, arch).NSEG
    return (backend.SEG_SAVE_MIN_TOKENS_PER_SEG * segments + 37,), 1


@pytest.mark.parametrize("case", ["serial-h12", "segmented-h4", "segmented-states-h1"])
def test_autograd_saved_set_replay_and_checkpoint_give_the_same_bits(case):
    backend = _det_backend()
    arch, sm_count = _device_kind()
    lengths, heads = {
        "serial-h12": ((300, 129, 500), 12),
        "segmented-h4": ((1000, 37, 300), 4),
        "segmented-states-h1": _states_shape(),
    }[case]
    plan = backend.plan(sum(lengths), len(lengths), heads, sm_count, arch, gate="raw", l2n="fused")
    assert (plan.recur, plan.light_path) == {
        "serial-h12": ("serial", "light"),
        "segmented-h4": ("seg", "light"),
        "segmented-states-h1": ("seg", "light_states"),
    }[case]
    inputs = _inputs(lengths, heads)
    reference = _run(inputs)
    for checkpoint in ("reentrant", "non_reentrant"):
        assert_bitwise(_run(inputs, checkpoint=checkpoint), reference)

    operands = [inputs[n].unsqueeze(0) for n in ("q", "k", "v", "g", "beta")]
    modes = dict(A_log=inputs["A_log"], dt_bias=inputs["dt_bias"], lower_bound=LOWER_BOUND, **MILES_MODES)
    out, saved = kda.chunk_kda_fwd(*operands, inputs["cu_seqlens"], **modes)
    assert isinstance(saved, backend.CakeKDADetChunkSaved)
    assert (saved.hseg is not None) == (plan.light_path == "light_states")
    assert_bitwise(out.squeeze(0), reference["out"])
    dout = inputs["dout"].unsqueeze(0)
    for set_ in (saved, None):  # the saved set, then the replay
        grads = kda.chunk_kda_bwd(*operands, inputs["cu_seqlens"], dout, saved=set_, **modes)
        got = {name: grad.squeeze(0) if grad.dim() > 2 else grad for name, grad in zip(BWD_NAMES, grads)}
        assert_bitwise(got, {name: reference[name] for name in BWD_NAMES})


def test_the_default_stream_and_a_side_stream_give_the_same_bits():
    """The default entry keeps its workspaces per stream: a call on a side stream between two on the default stream
    gives their bits, and the default stream's second call reuses its workspace."""
    inputs = _inputs((300, 129, 500), 12)
    first = _run(inputs, packed=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        on_side = _run(inputs, packed=True)
    side.synchronize()
    again = _run(inputs, packed=True)

    assert_bitwise(on_side, first)
    assert_bitwise(again, first)


@pytest.mark.parametrize("case", [("raw", True), ("g", False), ("raw", False)], ids=["miles", "g", "raw"])
def test_a_fresh_process_gives_the_bits_of_this_one(tmp_path, case):
    """``det`` repeats bit for bit across processes: two steps here, one in a new interpreter (same cache)."""
    mode, l2n = case
    here = assert_repeatable(lambda: _steps.autograd_step("det", mode, l2n), times=2)
    path = tmp_path / "step.pt"
    pythonpath = os.pathsep.join(filter(None, [str(TESTS.parent), str(TESTS), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-m", "cake.kda._steps", "det", mode, "1" if l2n else "0", str(path)],
        cwd=TESTS.parent,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert_bitwise(torch.load(path), here)


@pytest.mark.parametrize("shape", SHAPES, ids=_budgets.shape_id)
def test_fla_parity_with_fp32_beta(monkeypatch, shape):
    """miles' call on both backends with the FP32 ``sigmoid(beta_logits)`` miles gives them, against the exact FP64
    recurrence of that FP32 beta: every Cake tensor within twice fla's error (measured: at most 1.28 times).
    ``-rP`` prints the table: relative RMS errors, for ``dA_log`` / ``ddt_bias`` also the error over the size of their
    terms (``_budgets.accumulation_error``: the same ratio to fla), and the cost of the BF16 rounding of ``beta`` alone
    (the FP64 recurrence of the BF16 beta against it)."""
    monkeypatch.setenv("FLA_TILELANG", "0")  # what miles sets on Blackwell
    fla_chunk_kda = pytest.importorskip("fla.ops.kda").chunk_kda
    lengths, heads = shape
    inputs = _inputs(lengths, heads)
    generator = torch.Generator().manual_seed(4194)
    logits = torch.randn(sum(lengths), heads, generator=generator).to(torch.bfloat16).float().cuda()
    beta32 = logits.sigmoid()

    def fla(q, k, v, g, beta, A_log, dt_bias, cu_seqlens):
        out, _ = fla_chunk_kda(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
        return out.squeeze(0)

    ours = _run(inputs, packed=True, beta=beta32)
    theirs = _run(inputs, beta=beta32, kernel=fla)  # fla on contiguous copies
    exact32 = _reference(inputs, beta=beta32)
    exact16 = _reference(inputs, beta=beta32.to(torch.bfloat16))
    fla_errors = {name: rel_rms(theirs[name], exact32[name]) for name in NAMES}
    rows = [
        f"{name:>9} {rel_rms(ours[name], exact32[name]):12.3e} {fla_errors[name]:12.3e} "
        f"{rel_rms(ours[name], theirs[name]):12.3e} {rel_rms(exact16[name], exact32[name]):12.3e}"
        for name in NAMES
    ]
    terms = gate_param_terms(inputs["g"], inputs["dt_bias"], exact32["dg"])
    rows += [
        f"{name + '/acc':>9} {_budgets.accumulation_error(ours[name], exact32[name], terms[name]):12.3e} "
        f"{_budgets.accumulation_error(theirs[name], exact32[name], terms[name]):12.3e}"
        for name in _budgets.PARAM_TENSORS
    ]
    print(f"\n{_budgets.shape_id(shape)} {cake.device_arch()}: error against the FP64 recurrence of the FP32 beta")
    print(f"{'tensor':>9} {'cake':>12} {'fla':>12} {'cake-fla':>12} {'bf16 beta':>12}")
    print("\n".join(rows))
    assert_rel_rms(
        ours,
        exact32,
        {name: 2 * error for name, error in fla_errors.items()},
        loose=_budgets.loose(NAMES),
    )
