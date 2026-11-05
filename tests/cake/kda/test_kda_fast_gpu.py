"""The ``fast`` KDA variant (the vendored non-deterministic package) on one GPU: accuracy against the exact FP64
recurrence, the forward / backward split and the checkpointed node, the backward of a saved set of either package,
and the ``det`` run of a call with the in-kernel q / k L2 norm, which the package does not serve.

``fast`` gives no bit guarantee (in the raw gate mode it sums the parameter gradients in FP64 in no fixed order on
SM100a / SM103a), so its results are gated by accuracy alone: every tensor within twice ``det``'s relative RMS error
on the same inputs and call, against the same FP64 reference. Two gate modes, as the package takes them: ``g``, the
activated FP32 gate, and ``raw``, the raw BF16 gate the kernels activate (``use_gate_in_kernel=True``); q / k
L2-normalised by the caller. Needs an SM100a / SM103a / SM107a GPU, torch and tvm-ffi; the module first runs
``cake.prebuild("kda", num_heads=12)``.
"""

import contextlib
import logging
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import kda  # noqa: E402
from miles_ext_kernels.cake._runtime import config  # noqa: E402

from ..helpers import assert_bitwise, assert_rel_rms, rel_rms, requires_arch  # noqa: E402
from . import _budgets  # noqa: E402
from ._reference import (  # noqa: E402
    GRAD_NAMES,
    LOWER_BOUND,
    RAW_GRAD_NAMES,
    kda_exact_raw_with_grads,
    kda_exact_with_grads,
    make_raw_inputs,
    raw_gate,
)
from ._reference_l2n import raw_qk  # noqa: E402

pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")

# serial and segmented routes, few and many heads
ROUTE_SHAPES = [((1000, 37, 300), 4), ((300, 129, 500), 12), ((1000, 37, 300, 63), 64)]
MODES = ("g", "raw")


@pytest.fixture(scope="module", autouse=True)
def prebuilt():
    """Every library of the device's architecture, built once for the module."""
    cake.prebuild("kda", num_heads=12, max_workers=min(32, len(os.sched_getaffinity(0))))


@pytest.fixture(autouse=True)
def fast_variant(monkeypatch):
    """Every test runs ``fast`` unless it runs ``det`` in :func:`_variant`; the det-run log starts empty."""
    cake.configure(cake_kda_variant="fast")
    monkeypatch.setattr(config, "_warned", set())


@contextlib.contextmanager
def _variant(variant: str):
    cake.configure(cake_kda_variant=variant)
    try:
        yield
    finally:
        cake.configure(cake_kda_variant="fast")


def _inputs(lengths, heads: int, mode: str) -> dict:
    """The operands of ``mode``: ``g`` replaced by the activated FP32 gate in the ``g`` mode."""
    inputs = make_raw_inputs(lengths, heads, "cuda")
    if mode == "g":
        inputs["g"] = raw_gate(inputs["g"], inputs["A_log"], inputs["dt_bias"]).contiguous()
    return inputs


def _names(mode: str) -> tuple[str, ...]:
    return ("out",) + (RAW_GRAD_NAMES if mode == "raw" else GRAD_NAMES)


def _bwd_names(mode: str) -> tuple[str, ...]:
    """The order ``chunk_kda_bwd`` returns the gradients in."""
    return ("dq", "dk", "dv", "dbeta", "dg") + (("dA_log", "ddt_bias") if mode == "raw" else ())


def _modes(inputs: dict, mode: str, params: dict | None = None) -> dict:
    """The keywords of a call in ``mode`` (``params``: the raw gate's parameters to use, default the inputs')."""
    if mode == "g":
        return {}
    params = params or {"A_log": inputs["A_log"], "dt_bias": inputs["dt_bias"]}
    return {"use_gate_in_kernel": True, "lower_bound": LOWER_BOUND, **params}


def _run(inputs: dict, mode: str, *, checkpoint: str | None = None, l2n: bool = False) -> dict:
    """``out`` and the gradients of one autograd step of ``kda.chunk_kda`` on ``[1, S, H, 128]`` operands."""
    leaves = {n: inputs[n].clone().requires_grad_(True) for n in ("q", "k", "v", "g", "beta")}
    params = {n: inputs[n].clone().requires_grad_(True) for n in ("A_log", "dt_bias")} if mode == "raw" else {}

    def step(q, k, v, g, beta, *gate):
        modes = _modes(inputs, mode, dict(zip(params, gate)))
        operands = (x.unsqueeze(0) for x in (q, k, v, g, beta))
        return kda.chunk_kda(*operands, inputs["cu_seqlens"], use_qk_l2norm_in_kernel=l2n, **modes).squeeze(0)

    tensors = [*leaves.values(), *params.values()]
    if checkpoint is None:
        out = step(*tensors)
    else:
        out = torch.utils.checkpoint.checkpoint(step, *tensors, use_reentrant=checkpoint == "reentrant")
    out.backward(inputs["dout"])
    grads = {"d" + name: leaf.grad for name, leaf in {**leaves, **params}.items()}
    return {"out": out.detach(), **grads}


def _reference(inputs: dict, mode: str) -> dict:
    operands = [inputs[n] for n in ("q", "k", "v", "g", "beta")]
    if mode == "g":
        out, grads = kda_exact_with_grads(*operands, inputs["dout"], inputs["cu_seqlens"], kda.DEFAULT_SCALE)
    else:
        out, grads = kda_exact_raw_with_grads(
            *operands,
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["dout"],
            inputs["cu_seqlens"],
            kda.DEFAULT_SCALE,
            LOWER_BOUND,
        )
    return {"out": out, **grads}


def _det_budget(inputs: dict, mode: str, want: dict) -> dict[str, float]:
    """Twice ``det``'s relative RMS error of the same step, per tensor."""
    with _variant("det"):
        det = _run(inputs, mode)
    return {name: 2 * rel_rms(det[name], want[name]) for name in _names(mode)}


def _gate(got: dict, want: dict, budget: dict) -> dict[str, float]:
    return assert_rel_rms(got, want, {name: budget[name] for name in got}, loose=_budgets.loose(got))


def _grads(mode: str, grads) -> dict:
    """The gradients ``chunk_kda_bwd`` returns, by name, without the batch axis."""
    return {name: grad.squeeze(0) if grad.dim() > 2 else grad for name, grad in zip(_bwd_names(mode), grads)}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", _budgets.SHAPES, ids=_budgets.shape_id)
def test_fast_matches_the_exact_fp64_reference(shape, mode):
    lengths, heads = shape
    inputs = _inputs(lengths, heads, mode)
    want = _reference(inputs, mode)
    budget = _det_budget(inputs, mode, want)
    measured = _gate(_run(inputs, mode), want, budget)
    print(f"\n{_budgets.shape_id(shape)} {mode} {cake.device_arch()}: fast {measured}, det x2 {budget}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ROUTE_SHAPES, ids=_budgets.shape_id)
def test_fast_forward_backward_split_and_checkpoint_match_the_reference(shape, mode):
    lengths, heads = shape
    inputs = _inputs(lengths, heads, mode)
    want = _reference(inputs, mode)
    budget = _det_budget(inputs, mode, want)
    for checkpoint in ("reentrant", "non_reentrant"):
        _gate(_run(inputs, mode, checkpoint=checkpoint), want, budget)

    operands = [inputs[n].unsqueeze(0) for n in ("q", "k", "v", "g", "beta")]
    modes = _modes(inputs, mode)
    out, saved = kda.chunk_kda_fwd(*operands, inputs["cu_seqlens"], **modes)
    assert type(saved).__name__ == "KdaChunkSaved"
    _gate({"out": out.squeeze(0)}, want, budget)
    dout = inputs["dout"].unsqueeze(0)
    for set_ in (saved, None):  # the saved set, then the replay
        grads = kda.chunk_kda_bwd(*operands, inputs["cu_seqlens"], dout, saved=set_, **modes)
        _gate(_grads(mode, grads), want, budget)


@pytest.mark.parametrize("mode", MODES)
def test_a_saved_set_is_read_by_the_package_that_wrote_it(mode):
    """A ``det`` forward's saved set under ``fast``: ``det``'s backward, bit for bit; a ``fast`` set under ``det``: the
    ``fast`` backward, within its budget."""
    inputs = _inputs((300, 129, 500), 12, mode)
    operands = [inputs[n].unsqueeze(0) for n in ("q", "k", "v", "g", "beta")]
    modes = _modes(inputs, mode)
    dout = inputs["dout"].unsqueeze(0)
    with _variant("det"):
        _, det_saved = kda.chunk_kda_fwd(*operands, inputs["cu_seqlens"], **modes)
        det_grads = kda.chunk_kda_bwd(*operands, inputs["cu_seqlens"], dout, saved=det_saved, **modes)
    _, fast_saved = kda.chunk_kda_fwd(*operands, inputs["cu_seqlens"], **modes)

    grads = kda.chunk_kda_bwd(*operands, inputs["cu_seqlens"], dout, saved=det_saved, **modes)
    assert_bitwise(_grads(mode, grads), _grads(mode, det_grads))
    want = _reference(inputs, mode)
    budget = _det_budget(inputs, mode, want)
    with _variant("det"):
        grads = kda.chunk_kda_bwd(*operands, inputs["cu_seqlens"], dout, saved=fast_saved, **modes)
    _gate(_grads(mode, grads), want, budget)


@pytest.mark.parametrize("shape", [((300, 129, 500), 12), ((1000, 37, 300, 63), 64)], ids=_budgets.shape_id)
def test_fast_runs_det_for_the_in_kernel_l2_norm_and_says_so_once(caplog, shape):
    """miles' call (raw gate, in-kernel q / k L2 norm): ``fast`` runs ``det``, bit for bit, and logs it once."""
    lengths, heads = shape
    inputs = _inputs(lengths, heads, "raw")
    inputs["q"], inputs["k"] = raw_qk(lengths, heads, "cuda")
    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        fast = [_run(inputs, "raw", l2n=True) for _ in range(2)]
    with _variant("det"):
        det = _run(inputs, "raw", l2n=True)

    for got in fast:
        assert_bitwise(got, det)
    messages = [record.getMessage() for record in caplog.records if record.name == "miles_ext_kernels"]
    assert len(messages) == 1 and "in-kernel q / k L2 norm" in messages[0], messages
