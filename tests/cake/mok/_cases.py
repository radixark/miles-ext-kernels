"""MoK training iterations and their checks, shared by the single-GPU tests, their fresh-process run, the prebuild
routes and the multi-GPU worker (needs CUDA and an initialized process group).

A problem follows FlashInfer's ``examples/mok_training_features.py``: hidden 512, intermediate 256, mini-batch 256 and
macro-batch 768 rows (so backward replays the forward ring), unequal source counts with an empty rank, random distinct
expert ids (or, with ``routing="empty"``, ids among the first top-k experts only, so the others are empty) and positive
scores; layouts ``(local experts, top-k)``: (4, 2) at EP 1, (64, 8) at EP 4 (256 experts), (36, 8) at EP 8 (288). An
iteration is ``build_schedule``, ``forward``, optionally ``recompute_forward_context`` in place of the saved context,
and ``backward``; its nine outputs are ``y``, ``dx``, ``dscores``, the routed and the shared weight gradients.

Checks of a case:

- accuracy against the FP32 reference, per tensor (``assert_rel_rms``): within twice the relative RMS error of the BF16
  reference, an independent BF16 implementation of the same layer, measured in the same run; a 2 % scale of the
  tensor must fail that budget. MXFP8 routed experts quantize the routed path, so the six tensors that depend on it
  keep FlashInfer's relative L2 bound instead (several % of error, so a 2 % scale is within it); the shared-expert
  weight gradients stay on the BF16 budget. An all-zero reference (an empty rank) requires zeros.
- clamped SwiGLU: the clamp's derivative jumps at the limit, and BF16 rounding of the gate and up GEMM outputs moves
  elements across it, so the tensors that derivative reaches (``CLAMP_DERIVATIVE``: ``dx``, the routed and the shared
  gate and up weight gradients) are several % off the FP32 reference in any BF16 implementation (the BF16 reference
  too), and a 2 % scale stays within twice that. They keep that budget without the 2 % check, and must also be within
  twice the BF16 reference's error of the FP32 reference whose clamp decisions are made on its GEMM outputs rounded to
  BF16 (``clamp_decision``), where a 2 % scale must fail. (MXFP8: the routed-dependent ones keep FlashInfer's bound.)
- FlashInfer's own acceptance: the BF16 cases elementwise within ``1e-2 + 1e-2 * |reference|`` of the BF16 reference;
  the MXFP8 cases within its relative L2 bounds of FP32 over the group and away from the BF16 run.
- determinism: two eager runs and two CUDA graph replays equal bit for bit; a checkpointed iteration equal to the
  saved-context one; another fixed communication-SM split, and a fixed split that differs between the ranks (no
  imbalance switch), equal bit for bit; an empty rank's shared gradients zero. The workspace keeps FlashInfer's
  imbalance-aware split (``"auto"``; ``precision`` selects its row): on GPUs with a row, routing that sends every row
  to one rank switches the split per rank, from the latest load observation that has completed, so every run above
  may launch with other splits than the run it is compared with.

``digest`` is the SHA-256 of the nine outputs' bytes, for run-to-run comparisons across processes.
"""

from __future__ import annotations

import dataclasses
import hashlib

import torch
import torch.distributed as dist

from miles_ext_kernels.cake import mok

from ..helpers import assert_rel_rms, rel_rms
from . import _reference as reference

LAYOUTS = {1: (4, 2), 4: (64, 8), 8: (36, 8)}
HIDDEN, INTERMEDIATE = 512, 256
LIMIT = 0.125  # clamps a large fraction of gate and up at this input scale
MXFP8_LIMIT = 0.375  # FlashInfer's: MXFP8 error flips fewer clamp masks against the FP32 reference
COMM = (8, 8)  # the fixed (forward, backward) communication-SM split of the problems
OTHER_COMM = (12, 16)
# fixed (forward, backward) splits that differ between the ranks: rank r launches with RANK_COMM[r % 4]
RANK_COMM = ((8, 16), (12, 8), (16, 12), (20, 20))
CASES = {
    "bf16": {},
    "bf16-clamped": {"swiglu_limit": LIMIT},
    "bf16-recompute": {"recompute": True},
    "bf16-fp32-wgrad": {"fp32_wgrad": True},
    "bf16-clamped-recompute-fp32-wgrad": {"swiglu_limit": LIMIT, "recompute": True, "fp32_wgrad": True},
    "mxfp8": {"mxfp8": True},
    "mxfp8-fp32-wgrad": {"mxfp8": True, "fp32_wgrad": True},
    "mxfp8-clamped-recompute": {"mxfp8": True, "swiglu_limit": MXFP8_LIMIT, "recompute": True},
    "mxfp8-clamped-fp32-wgrad": {"mxfp8": True, "swiglu_limit": MXFP8_LIMIT, "fp32_wgrad": True},
}
BF16_CASES = tuple(name for name in CASES if not CASES[name].get("mxfp8"))
MXFP8_CASES = tuple(name for name in CASES if CASES[name].get("mxfp8"))
# the outputs MXFP8 routed experts quantize: y, dx, dscores and the routed weight gradients
ROUTED_DEPENDENT = reference.RESULT_NAMES[:6]
# the outputs the clamped SwiGLU's derivative reaches: dx and the gate and up weight gradients (not y, dscores, down)
CLAMP_DERIVATIVE = ("d_x", "d_w_routed_gate", "d_w_routed_up", "d_w_shared_gate", "d_w_shared_up")


def source_counts(ep: int) -> list[int]:
    return [1031] if ep == 1 else ([513, 0, 1031, 256] * ep)[:ep]


def model(ep: int = 1) -> dict:
    """The ``prebuild('mok', ...)`` parameters of the problem at EP ``ep``: its layout, top-k and widths."""
    local_experts, topk = LAYOUTS[ep]
    return dict(ep_size=ep, local_experts=local_experts, topk=topk, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE)


def digest(outputs) -> str:
    hasher = hashlib.sha256()
    for tensor in outputs:
        hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        hasher.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return hasher.hexdigest()


class Problem:
    """One rank's view of a global problem: inputs, weights, the adapters and the workspace.

    ``precision`` selects the row of the workspace's imbalance split (FlashInfer's ``create_mok_bf16_workspace``; the
    workspace serves both routed precisions), ``wgrads`` the weight-gradient precisions of the adapters it creates
    (``True``: FP32 accumulators), and ``comm_split_log`` prints each change of the communication-SM split to stderr.
    """

    def __init__(
        self,
        group,
        device,
        *,
        routing: str = "random",
        precision: str = "bf16",
        wgrads: tuple[bool, ...] = (False, True),
        comm_split_log: bool = False,
    ):
        self.group, self.device, self.routing = group, device, routing
        self.rank, self.ep = dist.get_rank(group), dist.get_world_size(group)
        self.local_experts, self.topk = LAYOUTS[self.ep]
        experts = self.ep * self.local_experts
        self.counts = source_counts(self.ep)
        options = dict(device=device, dtype=torch.bfloat16)
        torch.manual_seed(32471)
        self.shared = [
            torch.randn(INTERMEDIATE, HIDDEN, **options) / HIDDEN**0.5,
            torch.randn(INTERMEDIATE, HIDDEN, **options) / HIDDEN**0.5,
            torch.randn(HIDDEN, INTERMEDIATE, **options) / INTERMEDIATE**0.5,
        ]
        routed = [
            torch.randn(experts, INTERMEDIATE, HIDDEN, **options) / HIDDEN**0.5,
            torch.randn(experts, INTERMEDIATE, HIDDEN, **options) / HIDDEN**0.5,
            torch.randn(experts, HIDDEN, INTERMEDIATE, **options) / INTERMEDIATE**0.5,
        ]
        own = slice(self.rank * self.local_experts, (self.rank + 1) * self.local_experts)
        self.routed = [w[own].contiguous() for w in routed]
        self.weights = (*self.shared, *self.routed)
        torch.manual_seed(8291)
        total = sum(self.counts)
        if routing == "empty":
            row = torch.arange(total, device=device)
            ids = torch.stack([(row + slot) % self.topk for slot in range(self.topk)], -1)
        else:
            ids = torch.rand(total, experts, device=device).argsort(-1)[:, : self.topk].contiguous()
        scores = torch.rand(total, self.topk, device=device) + 0.1
        self.data = dict(
            x=torch.randn(total, HIDDEN, **options) * 0.125,
            d_output=torch.randn(total, HIDDEN, **options) * 0.125,
            expert_ids=ids,
            scores=scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5),
        )
        start = sum(self.counts[: self.rank])
        section = slice(start, start + self.counts[self.rank])
        self.x, self.ids, self.scores, self.dy = (
            self.data[k][section].clone() for k in ("x", "expert_ids", "scores", "d_output")
        )
        self.adapters = {
            fp32: mok.prepare_mok_bf16(
                ep_size=self.ep, local_experts=self.local_experts, topk=self.topk, fp32_wgrad=fp32
            )
            for fp32 in wgrads
        }
        self.config, self.workspace = mok.create_mok_bf16_workspace(
            group=group,
            device=device,
            num_local_tokens=self.counts[self.rank],
            hidden_size=HIDDEN,
            topk=self.topk,
            fwd_num_comm_sms=COMM[0],
            bwd_num_comm_sms=COMM[1],
            minibatch_size=256,
            macrobatch_size=768,
            schedule_capacity_multiplier=1.0,
            precision=precision,
        )
        if comm_split_log:
            self.config = dataclasses.replace(self.config, comm_split_log=True)
        assert self.workspace.initial_source_counts == tuple(self.counts)
        self._mxfp8 = None

    def mxfp8_weights(self):
        if self._mxfp8 is None:
            self._mxfp8 = mok.quantize_mok_mxfp8_weights(*self.routed)
        return self._mxfp8

    def iteration(self, name: str, *, recompute: bool | None = None, config=None) -> Iteration:
        case = CASES[name]
        return Iteration(
            self,
            config or self.config,
            swiglu_limit=case.get("swiglu_limit"),
            mxfp8=case.get("mxfp8", False),
            fp32_wgrad=case.get("fp32_wgrad", False),
            recompute=case.get("recompute", False) if recompute is None else recompute,
        )

    def other_split(self):
        """The problem's config with another fixed communication-SM split."""
        return dataclasses.replace(self.config, fwd_num_comm_sms=OTHER_COMM[0], bwd_num_comm_sms=OTHER_COMM[1])

    def rank_split(self):
        """The problem's config with this rank's fixed communication-SM split (``RANK_COMM``, different on each of
        four consecutive ranks) and no imbalance switch."""
        fwd, bwd = RANK_COMM[self.rank % len(RANK_COMM)]
        return dataclasses.replace(self.config, fwd_num_comm_sms=fwd, bwd_num_comm_sms=bwd, imbalance_comm_sms=None)

    def tokens(self, count: int, *, workspace=None, config=None) -> tuple:
        """Schedule, forward and backward of ``count`` fresh source tokens of this rank (BF16 routed experts, plain
        SwiGLU) through the problem's workspace (or ``workspace`` and ``config``); returns the nine outputs. One
        rank only: the other ranks would have to call it collectively."""
        workspace = self.workspace if workspace is None else workspace
        config = self.config if config is None else config
        generator = torch.Generator(device=self.device).manual_seed(1000 + count)
        options = dict(device=self.device, generator=generator)
        x = torch.randn(count, HIDDEN, dtype=torch.bfloat16, **options) * 0.125
        dy = torch.randn(count, HIDDEN, dtype=torch.bfloat16, **options) * 0.125
        ids = torch.rand(count, self.ep * self.local_experts, **options).argsort(-1)[:, : self.topk].contiguous()
        scores = torch.rand(count, self.topk, **options) + 0.1
        functional = self.adapters[False]
        schedule = functional.build_schedule(workspace, config, ids, num_local_experts=self.local_experts)
        y, context = functional.forward(config, workspace, schedule, x, scores, *self.weights)
        gradients = functional.backward(config, workspace, schedule, context, dy, x, scores, *self.weights)
        torch.cuda.synchronize()
        return (y, *gradients)


class Iteration:
    """One schedule + forward (+ recompute) + backward, eager or CUDA-graph captured (FlashInfer's
    ``TrainingIteration``)."""

    def __init__(self, problem: Problem, config, *, swiglu_limit, mxfp8, fp32_wgrad, recompute):
        self.p, self.config = problem, config
        self.limit, self.recompute = swiglu_limit, recompute
        self.functional = problem.adapters[fp32_wgrad]
        self.functional.prepare(swiglu_limit, mxfp8)
        self.functional.prepare_recompute(swiglu_limit, mxfp8)
        weights = problem.weights
        if mxfp8:
            forward, backward = problem.mxfp8_weights()
            self.forward_weights, self.backward_weights = (*weights[:3], *forward), (*weights[:3], *backward)
        else:
            self.forward_weights = self.backward_weights = weights
        self.accumulators = [torch.zeros(w.shape, device=problem.device) for w in weights] if fp32_wgrad else None
        self.graph = None
        self.outputs = None

    def _execute(self):
        p, w = self.p, self.forward_weights
        self.schedule = self.functional.build_schedule(
            p.workspace, self.config, p.ids, num_local_experts=p.local_experts
        )
        y, context = self.functional.forward(
            self.config, p.workspace, self.schedule, p.x, p.scores, *w, swiglu_limit=self.limit
        )
        if self.recompute:
            context = None
            context = self.functional.recompute_forward_context(
                self.config, p.workspace, self.schedule, p.x, w[0], w[1], w[3], w[4], swiglu_limit=self.limit
            )
        extra = {}
        if self.accumulators is not None:
            for accumulator in self.accumulators:
                accumulator.zero_()
            extra["weight_grad_accumulators"] = self.accumulators
        gradients = self.functional.backward(
            self.config,
            p.workspace,
            self.schedule,
            context,
            p.dy,
            p.x,
            p.scores,
            *self.backward_weights,
            swiglu_limit=self.limit,
            **extra,
        )
        self.outputs = (y, *gradients)
        return self.outputs

    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._execute()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier(group=self.p.group)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self._execute()

    def run(self):
        if self.graph is None:
            outputs = self._execute()
        else:
            self.graph.replay()
            outputs = self.outputs
        torch.cuda.synchronize()
        result = tuple(value.detach().clone() for value in outputs)
        torch.cuda.synchronize()
        return result


def _equal(a, b) -> list[str]:
    """Names of the outputs that differ bit for bit."""
    return [n for n, x, y in zip(reference.RESULT_NAMES, a, b, strict=True) if not torch.equal(x, y)]


def accuracy(outputs, bf16_reference, fp32_reference, *, mxfp8: bool, clamp_decided_reference=None) -> dict:
    """The per-tensor gates against the FP32 reference (module docstring): ``rel_rms`` (the measured errors, when every
    tensor passes), ``rel_rms_bf16_reference`` (the BF16 reference's errors) and ``rel_rms_failures``; for a clamped
    case (``clamp_decided_reference``: the FP32 reference with BF16 clamp decisions) also ``rel_rms_clamp_decided`` and
    ``rel_rms_bf16_reference_clamp_decided``, the same against that reference, its failures prefixed ``clamp-decided``.
    """
    names = reference.RESULT_NAMES
    got = dict(zip(names, outputs, strict=True))
    ref = dict(zip(names, fp32_reference, strict=True))
    bf16_error = {n: rel_rms(b, f) for n, b, f in zip(names, bf16_reference, fp32_reference, strict=True)}
    budget = {n: 2 * bf16_error[n] for n in names}
    zero = "the reference is zero: the output must be zero"
    loose = {n: zero for n in names if not torch.count_nonzero(ref[n])}
    if mxfp8:
        for n in ROUTED_DEPENDENT:
            budget[n] = reference.MXFP8_REL_L2
            loose.setdefault(n, "MXFP8 routed experts: several % of error, FlashInfer's relative L2 bound")
    record = dict(rel_rms={}, rel_rms_bf16_reference=bf16_error, rel_rms_failures=[])
    if clamp_decided_reference is not None:
        clamped = [n for n in CLAMP_DERIVATIVE if n not in loose]
        decided = {n: v for n, v in zip(names, clamp_decided_reference, strict=True) if n in clamped}
        bf16 = dict(zip(names, bf16_reference, strict=True))
        decided_error = {n: rel_rms(bf16[n], decided[n]) for n in clamped}
        record["rel_rms_bf16_reference_clamp_decided"] = decided_error
        for n in clamped:
            loose[n] = (
                "clamped SwiGLU: BF16 rounding moves gate and up across the limit, where the derivative jumps "
                "(several % of error in any BF16 implementation); the 2 % check runs against BF16 clamp decisions"
            )
        try:
            record["rel_rms_clamp_decided"] = assert_rel_rms(
                {n: got[n] for n in clamped},
                decided,
                {n: 2 * decided_error[n] for n in clamped},
                loose={n: zero for n in clamped if not torch.count_nonzero(decided[n])},
            )
        except AssertionError as error:
            record["rel_rms_failures"] += [f"clamp-decided: {line}" for line in str(error).splitlines()]
    try:
        record["rel_rms"] = assert_rel_rms(got, ref, budget, loose=loose)
    except AssertionError as error:
        record["rel_rms_failures"] += str(error).splitlines()
    return record


def run_case(problem: Problem, name: str, *, checks: bool = True, bf16_outputs=None) -> tuple[dict, tuple]:
    """Run case ``name`` on every rank; returns this rank's record (``digest`` and, with ``checks``, every check's
    result: lists of differing outputs, gate failures, measured errors) and the eager outputs."""
    case = CASES[name]
    p = problem
    eager = p.iteration(name)
    outputs = eager.run()
    record = dict(case=name, rank=p.rank, ep=p.ep, counts=p.counts, digest=digest(outputs))
    expected_dtype = torch.float32 if case.get("fp32_wgrad") else torch.bfloat16
    record["dtypes_ok"] = [v.dtype for v in outputs] == [torch.bfloat16, torch.bfloat16, torch.float32] + [
        expected_dtype
    ] * 6
    if not checks:
        return record, outputs
    record["eager_repeat_differs"] = _equal(outputs, eager.run())
    dist.barrier(group=p.group)
    eager.capture()
    record["graph_replay_differs"] = sorted({n for _ in range(2) for n in _equal(outputs, eager.run())})
    del eager
    torch.cuda.synchronize()
    dist.barrier(group=p.group)
    if case.get("recompute"):
        saved = p.iteration(name, recompute=False).run()
        record["recompute_differs"] = _equal(saved, outputs)
    record["other_split_differs"] = _equal(outputs, p.iteration(name, config=p.other_split()).run())
    dist.barrier(group=p.group)
    record["rank_split_differs"] = _equal(outputs, p.iteration(name, config=p.rank_split()).run())
    dist.barrier(group=p.group)
    limit = case.get("swiglu_limit")
    expected = reference.reference(p.data, p.weights, source_counts=p.counts, swiglu_limit=limit)
    oracle = reference.reference(p.data, p.weights, fp32=True, source_counts=p.counts, swiglu_limit=limit)
    decided = None
    if limit is not None:
        decided = reference.reference(
            p.data, p.weights, fp32=True, source_counts=p.counts, swiglu_limit=limit, clamp_decision=torch.bfloat16
        )
    record.update(accuracy(outputs, expected, oracle, mxfp8=bool(case.get("mxfp8")), clamp_decided_reference=decided))
    if not case.get("mxfp8"):
        record["gate_failures"] = reference.gate_failures(outputs, expected)
    else:
        record["rel_l2_vs_fp32"] = {
            n: reference.rel_l2(a, b) for n, a, b in zip(reference.RESULT_NAMES, outputs, oracle, strict=True)
        }
        record["shared_gate_failures"] = reference.gate_failures(outputs[6:], expected[6:], reference.RESULT_NAMES[6:])
        if limit is None and bf16_outputs is not None:
            record["rel_l2_vs_bf16_run"] = reference.rel_l2(outputs[0], bf16_outputs[0])
    if p.counts[p.rank] == 0:
        record["empty_rank_shared_nonzero"] = [
            n for n, v in zip(reference.RESULT_NAMES[6:], outputs[6:], strict=True) if torch.count_nonzero(v).item()
        ]
    dist.barrier(group=p.group)
    return record, outputs


def failures(record: dict) -> list[str]:
    """Every failed check of a ``run_case`` record (empty when the case passes)."""
    bad = []
    if not record["dtypes_ok"]:
        bad.append("output dtypes")
    for key in (
        "eager_repeat_differs",
        "graph_replay_differs",
        "recompute_differs",
        "other_split_differs",
        "rank_split_differs",
    ):
        if record.get(key):
            bad.append(f"{key}: {record[key]}")
    for key in ("gate_failures", "shared_gate_failures"):
        if record.get(key):
            bad.append(f"{key}: {record[key]}")
    bad += [f"rel_rms: {line}" for line in record.get("rel_rms_failures", ())]
    if record.get("empty_rank_shared_nonzero"):
        bad.append(f"empty rank shared gradients nonzero: {record['empty_rank_shared_nonzero']}")
    rel = record.get("rel_l2_vs_fp32")
    if rel is not None:
        for name in reference.RESULT_NAMES:
            bound = reference.MXFP8_REL_L2 if name in ROUTED_DEPENDENT else reference.SHARED_REL_L2
            if not rel[name] < bound:
                bad.append(f"{name} relative L2 {rel[name]:.4g} vs FP32 >= {bound}")
    distance = record.get("rel_l2_vs_bf16_run")
    if distance is not None and not distance > reference.MXFP8_MIN_DISTANCE:
        bad.append(f"MXFP8 output within {distance:.3g} of the BF16 run: the routed GEMMs did not run in MXFP8")
    return bad
