"""The context-parallel KDA entry points (``kda.chunk_kda_kcp`` and its partition, context and parameter reduce) on 2 and
4 GPUs of one node, NCCL. Marked ``multi_gpu``: a one-GPU run deselects the module with ``-m "not multi_gpu"``.

For each process-group size and partition layout one set of worker processes (``_kcp_worker.py``) runs every case:
``det`` in miles' call (raw gate, in-kernel q / k L2 norm, FP32 ``beta``) and in the activated-gate mode, ``fast`` in
the activated and the raw gate modes, and miles' call under ``fast``, which runs ``det``. The ragged documents cross
the rank boundaries. Checked here: the shards put together match the exact FP64 recurrence of the whole batch within
twice the relative RMS error of the single-rank ``det`` call of the same batch and mode (per tensor), a ``det`` rank's
repeated step is bit-identical, the raw gate's parameter gradients reduced with ``reduce_gate_param_grads_kcp`` are
the same bits on every rank, and ``fast`` with the in-kernel L2 norm gives ``det``'s bits. Needs 4 SM100a / SM103a /
SM107a GPUs of one node (the 2-rank groups run on 2), torch, tvm-ffi and NCCL; the module first runs
``cake.prebuild("kda", num_heads=12)`` in the test process, so the ranks only load the libraries.
"""

import os
import socket
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

import torch.multiprocessing as mp  # noqa: E402

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import kda  # noqa: E402

from ..helpers import assert_rel_rms, rel_rms, requires_arch  # noqa: E402
from . import _budgets, _kcp_worker  # noqa: E402
from ._reference import (  # noqa: E402
    GRAD_NAMES,
    LOWER_BOUND,
    RAW_GRAD_NAMES,
    cu_seqlens_of,
    kda_exact_raw_with_grads,
    kda_exact_with_grads,
)
from ._reference_l2n import kda_exact_raw_l2n_with_grads  # noqa: E402

pytestmark = [pytest.mark.multi_gpu, requires_arch("sm_100a", "sm_103a", "sm_107a")]

GROUPS = [(2, "contiguous"), (2, "zigzag"), (4, "contiguous"), (4, "zigzag")]
TIMEOUT_SECONDS = 1800  # one group runs every case two or three times on prebuilt libraries
_RUNS: dict = {}
_REFERENCES: dict = {}


@pytest.fixture(scope="module", autouse=True)
def prebuilt():
    """Every library of the device's architecture, built once in the test process; NCCL present."""
    assert torch.distributed.is_available() and torch.distributed.is_nccl_available(), "NCCL required"
    cake.prebuild("kda", num_heads=12, max_workers=min(32, len(os.sched_getaffinity(0))))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _group_results(world: int, layout: str, tmp_path_factory) -> list[dict]:
    """Every rank's saved results of one group, run once per module."""
    visible = torch.cuda.device_count()
    if visible < world:
        pytest.skip(f"the {world}-rank groups need {world} GPUs of one node, {visible} visible")
    key = (world, layout)
    if key not in _RUNS:
        outdir = tmp_path_factory.mktemp(f"kcp-{world}-{layout}")
        context = mp.spawn(_kcp_worker.run, args=(world, _free_port(), layout, str(outdir)), nprocs=world, join=False)
        deadline = time.monotonic() + TIMEOUT_SECONDS
        try:
            while not context.join(timeout=30):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"the {world}-rank {layout} group did not finish in {TIMEOUT_SECONDS} s")
            _RUNS[key] = [torch.load(outdir / f"rank{rank}.pt") for rank in range(world)]
        except Exception as exc:  # every case of the group reports the group's failure, without running it again
            _RUNS[key] = exc
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
    if isinstance(_RUNS[key], Exception):
        raise _RUNS[key]
    return _RUNS[key]


def _names(mode: str) -> tuple[str, ...]:
    return ("out",) + (RAW_GRAD_NAMES if mode == "raw" else GRAD_NAMES)


def _single_rank_det(t: dict, cu_seqlens, mode: str, l2n: bool) -> dict:
    """``out`` and the gradients of the single-rank ``det`` call of the whole batch, on the first GPU."""
    leaves = {n: t[n].clone().requires_grad_(True) for n in _kcp_worker.NAMES}
    params = {n: t[n].clone().requires_grad_(True) for n in ("A_log", "dt_bias")} if mode == "raw" else {}
    modes = {"use_gate_in_kernel": True, "lower_bound": LOWER_BOUND, **params} if mode == "raw" else {}
    cake.configure(cake_kda_variant="det")
    try:
        operands = (leaves[n].unsqueeze(0) for n in _kcp_worker.NAMES)
        out = kda.chunk_kda(*operands, cu_seqlens, use_qk_l2norm_in_kernel=l2n, **modes).squeeze(0)
        out.backward(t["dout"])
    finally:
        cake.configure()
    return {"out": out.detach(), **{"d" + n: x.grad for n, x in {**leaves, **params}.items()}}


def _reference(case: str) -> tuple[dict, dict]:
    """The exact FP64 output and gradients of the whole batch of ``case``, and the budget of each: twice the error
    of the single-rank ``det`` call of the same batch and mode."""
    if case not in _REFERENCES:
        _, mode, l2n = _kcp_worker.CASES[case]
        t = _kcp_worker.inputs(mode, l2n, "cuda:0")
        cu_seqlens = torch.tensor(cu_seqlens_of(_kcp_worker.LENGTHS), dtype=torch.int32, device="cuda:0")
        operands = [t[n] for n in _kcp_worker.NAMES]
        if mode == "g":
            out, grads = kda_exact_with_grads(*operands, t["dout"], cu_seqlens, kda.DEFAULT_SCALE)
        else:
            exact = kda_exact_raw_l2n_with_grads if l2n else kda_exact_raw_with_grads
            out, grads = exact(
                *operands, t["A_log"], t["dt_bias"], t["dout"], cu_seqlens, kda.DEFAULT_SCALE, LOWER_BOUND
            )
        want = {"out": out, **grads}
        det = _single_rank_det(t, cu_seqlens, mode, l2n)
        _REFERENCES[case] = want, {name: 2 * rel_rms(det[name], want[name]) for name in _names(mode)}
    return _REFERENCES[case]


def _assembled(ranks: list[dict], case: str) -> dict:
    """The token-row results of every rank put back in global token order; the parameter gradients of rank 0."""
    total = sum(_kcp_worker.LENGTHS)
    whole = {}
    for saved in ranks:
        idx, result = saved["idx"], saved["cases"][case]["result"]
        for name, tensor in result.items():
            if name in ("dA_log", "ddt_bias"):
                whole.setdefault(name, tensor)
                continue
            if name not in whole:
                whole[name] = torch.zeros((total, *tensor.shape[1:]), dtype=tensor.dtype)
            whole[name][idx] = tensor
    covered = torch.cat([saved["idx"] for saved in ranks]).sort().values
    assert torch.equal(covered, torch.arange(total)), "the shards do not cover the batch exactly once"
    return whole


@pytest.mark.parametrize("case", list(_kcp_worker.CASES))
@pytest.mark.parametrize(("world", "layout"), GROUPS, ids=[f"cp{w}-{layout}" for w, layout in GROUPS])
def test_kcp_matches_the_exact_fp64_reference_and_repeats_bit_for_bit(tmp_path_factory, world, layout, case):
    ranks = _group_results(world, layout, tmp_path_factory)
    _, mode, _ = _kcp_worker.CASES[case]
    records = [saved["cases"][case] for saved in ranks]

    if _kcp_worker.runs_det(case):
        assert [record["repeat_differs"] for record in records] == [[]] * world
    if case == "fast-l2n":
        assert [record["det_differs"] for record in records] == [[]] * world
    if mode == "raw":
        # one sum in rank order: the same bits on every rank
        params = [{n: record["result"][n] for n in ("dA_log", "ddt_bias")} for record in records]
        assert [_kcp_worker.differing(rank_params, params[0]) for rank_params in params] == [[]] * world
    want, budget = _reference(case)
    got = {name: tensor.cuda() for name, tensor in _assembled(ranks, case).items()}
    measured = assert_rel_rms(got, want, budget, loose=_budgets.loose(got))
    print(f"\ncp{world}-{layout} {case}: {measured}, single-rank det x2 {budget}")
