"""The Cake MoE grouped GEMM (``cake.moe_grouped_gemm.grouped_mm``) on the expert projections of miles' torchtitan MoE
models: the forward, input and weight gradients against a per-group FP32 reference (relative RMS gates), ``det``
repeating bit for bit (in one process and in a new one), ``fast`` giving ``det``'s forward and input-gradient bits and
a weight gradient whose split tail it reduces in a fixed order within one launch (bit for bit run to run), the variant
read at each call, ``det`` under torch's deterministic algorithms, the per-stream workspaces, the 32-bit row limit
checked at the forward and the results at that limit, empty input and padding rows. Needs SM100a / SM103a / SM107a;
skips elsewhere.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import moe_grouped_gemm  # noqa: E402
from miles_ext_kernels.cake._impl.moe_grouped_gemm import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402

from ..helpers import assert_bitwise, assert_rel_rms, assert_repeatable  # noqa: E402
from . import _cases  # noqa: E402
from ._cases import BUDGET, CASES, K512_SHAPE, K512_SIZES, OP, SEED, VARIANT  # noqa: E402

try:
    ARCH = loader.device_arch()
except NotImplementedError:
    pytest.skip("the Cake MoE grouped GEMM needs SM100a / SM103a / SM107a", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
DET_NODE, FAST_NODE = "CakeMoEGroupedGemmDetFunctionBackward", "CakeMoEGroupedGemmNondetFunctionBackward"


@pytest.fixture(scope="module", autouse=True)
def _programs():
    """Build both variants' programs up front, in parallel, instead of one by one on first use."""
    return cake.prebuild(OP)


def _configure(variant: str) -> None:
    cake.configure(**{VARIANT: variant})


def _run(variant, x, w, g, offs) -> dict:
    _configure(variant)
    return _cases.step(x, w, g, offs)


@pytest.mark.parametrize("layer, proj", CASES)
def test_det_matches_the_reference_and_repeats_bit_for_bit(layer, proj):
    sizes, n, k = _cases.shape(layer, proj)
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + n + k)

    det = assert_repeatable(lambda: _run("det", x, w, g, offs))

    assert_rel_rms(det, _cases.reference(x, w, g, sizes), BUDGET)
    _cases.assert_empty_groups_zero(det["dw"], sizes)


@pytest.mark.parametrize("layer, proj", CASES)
def test_fast_gives_the_det_bits_where_its_reductions_are_ordered(layer, proj):
    """Below the 512-wide k tile's rows per group the weight gradient keeps the 256-wide k tile and its ordered tail
    reduction: fast is bitwise det in all three products."""
    sizes, n, k = _cases.shape(layer, proj)
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + n + k)

    assert_bitwise(_run("fast", x, w, g, offs), _run("det", x, w, g, offs))


def test_the_fast_weight_gradient_reduces_its_split_tail_in_order_within_the_budget():
    """The 512-wide k tile's split tail: every part lands in its own slot and the last unit sums the slots in order,
    so fast repeats bit for bit; its tail tiles may differ from det's within the budget."""
    sizes, (n, k) = list(K512_SIZES), K512_SHAPE
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + 512)
    plan = ops._module("fast", "cake_backend").cake_moe_grouped_gemm_nondet_train_prepare_wgrad(g, x, offs).plan
    assert plan["tile_k"] == 512 and plan["tail_splits"] > 1, plan
    assert plan["ordered_tail"] and not plan["unordered_tail"], plan

    det, fast = _run("det", x, w, g, offs), _run("fast", x, w, g, offs)
    reference = _cases.reference(x, w, g, sizes)

    assert_bitwise({"y": fast["y"], "dx": fast["dx"]}, {"y": det["y"], "dx": det["dx"]})
    assert_rel_rms(det, reference, BUDGET)
    assert_rel_rms(fast, reference, BUDGET)
    _cases.assert_empty_groups_zero(fast["dw"], sizes)
    assert_bitwise(_run("det", x, w, g, offs), det)
    assert_bitwise(_run("fast", x, w, g, offs), fast)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_new_group_boundaries_on_the_same_shapes_are_read_on_the_device(variant):
    sizes, n, k = _cases.shape("h2048-i768-e16", "w2")
    x, w, g, _ = _cases.inputs(sizes, n, k, seed=SEED)
    for moved in (sizes, sizes[::-1], [sum(sizes)] + [0] * (len(sizes) - 1)):
        got = _run(variant, x, w, g, _cases.offs(moved))
        assert_rel_rms(got, _cases.reference(x, w, g, moved), BUDGET)
        _cases.assert_empty_groups_zero(got["dw"], moved)


def test_the_variant_is_read_at_each_call_and_a_backward_runs_its_forwards():
    sizes, n, k = _cases.shape("h2048-i1536-e8", "w2")
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + 2)
    det = _run("det", x, w, g, offs)

    assert config.variant(OP) == "det"
    cake.configure()  # unconfigured: det
    xs, ws = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    y_det = moe_grouped_gemm.grouped_mm(xs, ws, offs)
    _configure("fast")
    y_fast = moe_grouped_gemm.grouped_mm(x.clone().requires_grad_(True), w.clone().requires_grad_(True), offs)
    y_det.backward(g)  # configured fast now: the backward still runs det, its forward's package

    assert type(y_det.grad_fn).__name__ == DET_NODE
    assert type(y_fast.grad_fn).__name__ == FAST_NODE
    assert_bitwise({"y": y_det.detach(), "dx": xs.grad, "dw": ws.grad}, det)


def test_torch_deterministic_algorithms_run_det_and_say_so_once(monkeypatch, caplog):
    sizes, (n, k) = list(K512_SIZES), K512_SHAPE
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + 1)
    det = _run("det", x, w, g, offs)
    monkeypatch.setattr(config, "_warned", set())
    _configure("fast")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            first, second = _cases.step(x, w, g, offs), _cases.step(x, w, g, offs)
            node = type(moe_grouped_gemm.grouped_mm(x, w.requires_grad_(True), offs).grad_fn).__name__
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    assert node == DET_NODE
    assert_bitwise(first, det)
    assert_bitwise(second, det)
    runs_det = [r for r in caplog.records if "fast variant does not serve" in r.getMessage()]
    assert len(runs_det) == 1, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_calls_alternating_between_the_default_and_a_side_stream_give_the_same_bits(variant):
    """The packages keep their workspaces per stream: calls on two streams, in turn, give the bits of the first."""
    sizes, n, k = _cases.shape("h2048-i768-e16", "w2")  # a probed data gradient on sm_100a / sm_103a
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + 3)
    backend = ops._module(variant, "cake_backend")
    if variant == "fast":  # the 256-wide k tile: the fast weight gradient reduces in order too
        assert backend.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(g, x, offs).plan["tile_k"] == 256
    _configure(variant)
    side = torch.cuda.Stream()
    results = []
    for turn in range(4):
        stream = side if turn % 2 else torch.cuda.current_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results.append(_cases.step(x, w, g, offs))
    torch.cuda.synchronize()

    for result in results[1:]:
        assert_bitwise(result, results[0])
    default_sched = backend.stream_workspace("sched", x.device)
    side_sched = backend.stream_workspace("sched", x.device, side)
    assert default_sched is not None and side_sched is not None
    assert default_sched.data_ptr() != side_sched.data_ptr()


# short groups (around the 64-row tile edges) with a large group every eighth expert
SHORT_GROUPS = (0, 1, 7, 31, 63, 64, 65, 127, 129, 255, 257)


@pytest.mark.parametrize("proj", ["w13", "w2"])
def test_short_groups_between_large_ones_repeat_bit_for_bit(proj):
    """Many short groups (empty, one row, around the tile edges) interleaved with large ones, eight passes per
    variant: det repeats its bits, fast gives det's bits (the 256-wide k tile), both within the budget."""
    sizes = [4099 if e % 8 == 7 else SHORT_GROUPS[e % len(SHORT_GROUPS)] for e in range(64)]
    _, n, k = _cases.shape("h2048-i768-e16", proj)
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + 8)
    plan = ops._module("fast", "cake_backend").cake_moe_grouped_gemm_nondet_train_prepare_wgrad(g, x, offs).plan
    assert plan["tile_k"] == 256
    det = assert_repeatable(lambda: _run("det", x, w, g, offs), times=8)
    for _ in range(8):
        assert_bitwise(_run("fast", x, w, g, offs), det)

    assert_rel_rms(det, _cases.reference(x, w, g, sizes), BUDGET)
    _cases.assert_empty_groups_zero(det["dw"], sizes)


def test_release_drops_the_stream_workspaces_and_later_calls_give_the_same_bits():
    sizes, n, k = _cases.shape("h4096-i2048-e9", "w13")
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + 4)
    det = _run("det", x, w, g, offs)
    backend = ops._module("det", "cake_backend")
    assert backend.stream_workspace("sched", x.device) is not None

    cake.release(OP)

    assert backend.stream_workspace("sched", x.device) is None
    assert not backend._queue_splits
    assert_bitwise(_run("det", x, w, g, offs), det)


def test_the_32_bit_row_limit_fails_at_the_forward_before_any_launch():
    """K = 7168, N = 4096: the input gradient [sum_m, K] addresses at most 299,593 rows. 300,000 rows fail at the
    forward call (the backward used to fail after the forward ran). At 299,593 rows the last rows of the output and of
    the input gradient (the highest 32-bit offsets) and the weight gradient match their references: the output
    gradient is one value per row, so the input gradient rows are those values times the column sums of the weight
    and every row of the weight gradient is the value-weighted sum of the input rows."""
    n, k, tail, chunk = 4096, 7168, 2048, 32768
    limit = moe_grouped_gemm.max_grouped_rows(n, k)
    assert limit == 299_593
    _configure("det")
    gen = torch.Generator(device="cuda").manual_seed(SEED + 9)
    w = (torch.randn(1, n, k, generator=gen, dtype=torch.bfloat16, device="cuda") * k**-0.5).requires_grad_(True)
    x = torch.zeros(limit + 1, k, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    offs = torch.tensor([limit + 1], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="at most 299593 rows"):
        moe_grouped_gemm.grouped_mm(x, w, offs)
    del x

    x = (torch.randn(limit, k, generator=gen, dtype=torch.bfloat16, device="cuda") * k**-0.5).requires_grad_(True)
    per_row = torch.randn(limit, generator=gen, dtype=torch.bfloat16, device="cuda")
    y = moe_grouped_gemm.grouped_mm(x, w, offs - 1)
    y.backward(per_row[:, None].expand(limit, n).contiguous())
    torch.cuda.synchronize()

    rows, s, xs, w0 = slice(limit - tail, limit), per_row.double(), x.detach(), w.detach()[0].double()
    dw_row = sum(s[i : i + chunk] @ xs[i : i + chunk].double() for i in range(0, limit, chunk))  # in FP64
    got = {"y": y.detach()[rows], "dx": x.grad[rows], "dw": w.grad[0]}
    reference = {"y": xs[rows].double() @ w0.T, "dx": s[rows, None] * w0.sum(0), "dw": dw_row.expand(n, k)}
    assert_rel_rms(got, reference, BUDGET)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_empty_input_gives_an_exact_zero_weight_gradient(variant):
    sizes, n, k = [0] * 8, 1024, 2048
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + 5)

    got = _run(variant, x, w, g, offs)

    assert got["y"].shape == (0, n) and got["dx"].shape == (0, k)
    assert got["dw"].shape == w.shape and not got["dw"].any()


@pytest.mark.parametrize("tail", ["zero", "random"])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_rows_past_the_last_group_end_leave_the_valid_results_alone(variant, tail):
    """torchtitan pads the routed rows past ``offs[-1]``: they change none of the valid rows and no weight gradient."""
    sizes, n, k = _cases.shape("h2048-i768-e16", "w13")
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + 6, rows=sum(sizes) + 300)
    if tail == "zero":
        x[sum(sizes) :] = 0
        g[sum(sizes) :] = 0
    unpadded = _run(variant, x[: sum(sizes)], w, g[: sum(sizes)], offs)

    padded = _cases.valid(_run(variant, x, w, g, offs), sizes)

    reference = _cases.reference(x, w, g, sizes)  # the rows of the groups only
    assert_rel_rms(padded, reference, BUDGET)
    assert_rel_rms(unpadded, reference, BUDGET)
    _cases.assert_empty_groups_zero(padded["dw"], sizes)


def test_calls_the_programs_do_not_serve_are_refused():
    x = torch.empty(64, 1024, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(4, 512, 1024, dtype=torch.bfloat16, device="cuda")
    offs = _cases.offs([16] * 4)
    with pytest.raises(ValueError, match="multiple of 256"):
        moe_grouped_gemm.grouped_mm(x, torch.empty(4, 384, 1024, dtype=torch.bfloat16, device="cuda"), offs)
    with pytest.raises(ValueError, match="multiple of 256"):
        moe_grouped_gemm.grouped_mm(x[:, :960], torch.empty(4, 512, 960, dtype=torch.bfloat16, device="cuda"), offs)
    with pytest.raises(ValueError, match="int32 cumulative row ends"):
        moe_grouped_gemm.grouped_mm(x, w, offs.long())
    with pytest.raises(ValueError, match="int32 cumulative row ends"):
        moe_grouped_gemm.grouped_mm(x, w, _cases.offs([16] * 5))
    with pytest.raises(ValueError, match="BF16"):
        moe_grouped_gemm.grouped_mm(x.float(), w, offs)
    with pytest.raises(ValueError, match=r"x \[sum_m, K\] and w \[E, N, K\]"):
        moe_grouped_gemm.grouped_mm(x, w[..., :512], offs)
    with pytest.raises(ValueError, match="one device"):
        moe_grouped_gemm.grouped_mm(x, w, offs.cpu())
    with pytest.raises(ValueError, match="runs on CUDA devices"):
        moe_grouped_gemm.grouped_mm(x.cpu(), w.cpu(), offs.cpu())


def test_the_device_needs_its_architectures_toolkit(monkeypatch):
    ops._arch.cache_clear()  # the architecture of a device index is cached once checked
    try:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
        with pytest.raises(NotImplementedError, match="SM100a, SM103a or SM107a"):
            cake.prebuild(OP)
        ops._arch.cache_clear()
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
        monkeypatch.setattr(loader, "MIN_CUDA", {**loader.MIN_CUDA, "sm_107a": (99, 0)})
        with pytest.raises(RuntimeError, match=r"MoE grouped GEMM for sm_107a needs CUDA 99\.0 or newer"):
            cake.prebuild(OP)
    finally:
        ops._arch.cache_clear()


_DIGESTS_IN_A_NEW_PROCESS = """
import json
from miles_ext_kernels import cake
from cake.moe_grouped_gemm import _cases

cake.configure(cake_moe_grouped_gemm_variant="det")
out = {}
for layer, proj in _cases.CASES:
    sizes, n, k = _cases.shape(layer, proj)
    out[f"{layer}-{proj}"] = _cases.digest(_cases.step(*_cases.inputs(sizes, n, k, seed=_cases.SEED + n + k)))
print(json.dumps(out))
"""


def test_det_repeats_bit_for_bit_in_a_new_process():
    expected = {}
    for layer, proj in CASES:
        sizes, n, k = _cases.shape(layer, proj)
        expected[f"{layer}-{proj}"] = _cases.digest(_run("det", *_cases.inputs(sizes, n, k, seed=SEED + n + k)))
    paths = [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")]
    result = subprocess.run(
        [sys.executable, "-c", _DIGESTS_IN_A_NEW_PROCESS],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1]) == expected


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_call_for_a_device_other_than_the_current_one_gives_its_bits(variant):
    """A launch for a device other than the current one takes workspaces of its own and gives the bits of the same
    call with that device current."""
    if loader.device_arch(1) != ARCH:
        pytest.skip("needs two GPUs of one architecture")
    cake.prebuild(OP, device="cuda:1")
    sizes, n, k = _cases.shape("h2048-i1536-e8", "w2")
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED + 7, device="cuda:1")
    with torch.cuda.device(1):
        current = _run(variant, x, w, g, offs)
    with torch.cuda.device(0):
        other = _run(variant, x, w, g, offs)
        torch.cuda.synchronize(1)

    assert_bitwise(other, current)
    assert_rel_rms(other, _cases.reference(x, w, g, sizes), BUDGET)


def test_the_calls_import_no_flashinfer():
    sizes, n, k = _cases.shape("h2048-i768-e16", "w13")
    for variant in ("det", "fast"):
        _run(variant, *_cases.inputs(sizes, n, k, seed=SEED))

    assert [name for name in sys.modules if name.split(".")[0] == "flashinfer"] == []
