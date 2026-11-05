"""The Cake mHC kernels (``miles_ext_kernels.cake.mhc``) on SM100a / SM103a / SM107a.

Ported from FlashInfer's backend suites of the three vendored packages at every hidden size the device serves (4096,
5120, 7168; 4096 only on SM107a, where the others are refused) and token counts 1, 257, 2049 and 16263, through the
autograd entry points miles calls:

* ``det``: every output against an FP64 reference of the function under FlashInfer's bars -- BF16 ``U``, ``Y``,
  ``dB``, ``dX``: ``|out - ref| <= 1e-2 + 1e-2 |ref|`` on every element; FP32 ``dfn``, ``dscale``, ``dbase``: rel-L2
  <= 1e-5, and ``dscale`` / ``dbase`` per element <= 1e-4 relative (floor: the vector's RMS) -- plus a relative RMS
  gate on every tensor that a 2 % error fails (``assert_rel_rms``), and a second run bitwise identical;
* ``fast``: ``U``, ``Y``, ``dB`` and ``dX`` bitwise the ``det`` outputs, ``dscale`` / ``dbase`` too outside the
  ``det`` group-row band (hidden 4096 above 14927 tokens, where ``det`` sums them in another fixed order: there under
  the FP32 bars), ``dfn`` under the FP32 bars and within FlashInfer's summation-order bound of the ``det`` ``dfn``;
* the variant of a call is the configured one, read when the call runs; deterministic mode without a variant is
  ``det``; ``fast`` under torch's deterministic algorithms raises torch's alert in its backward;
* BF16 parameters (FSDP2 ``param_dtype=torch.bfloat16``) run the FP32 parameters' bits;
* a non-default Sinkhorn count and non-default epsilons against FP64;
* ``det`` repeats bit for bit in a new process, on a side stream and on a second GPU;
* with the pool on (``POOLED``; the default bound, 0, keeps nothing), every launch takes its workspace from the
  runtime's pool, one buffer per role and stream: the bits of a pooled call repeat with the pool off (bound 0) or so
  small that the forward and backward evict each other's buffer, when a buffer sized for a larger call serves a
  smaller one, and after ``release`` frees the pool;
* the head: bit for bit torch's BF16 ``X.mean(dim=1)`` and its gradient.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.mhc import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import loader, workspace  # noqa: E402

from ..helpers import POOLED, assert_bitwise, assert_rel_rms, no_build, requires_arch  # noqa: E402
from . import _connection as connection  # noqa: E402
from ._connection import BF16, BF16_OUTPUTS, F32, F64, OUTPUTS, PARAMETER_GRADS, STREAMS  # noqa: E402

pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")
DEVICE = torch.device("cuda", torch.cuda.current_device())
ARCH = loader.ARCH_BY_CAPABILITY.get(tuple(torch.cuda.get_device_capability(DEVICE)))
# the hidden sizes the device serves: FlashInfer exports the packages for SM107a at hidden 4096 only
SERVED = (4096,) if ARCH == "sm_107a" else ops.HIDDEN_SIZES
REPO_ROOT = Path(__file__).resolve().parents[3]

BF16_ATOL = BF16_RTOL = 1e-2
FP32_REL_L2_MAX = 1e-5
FP32_ELEM_REL_MAX = 1e-4
FP32_UNIT = 2.0**-24
DFN_ABS_SLACK = 1.02
# Relative RMS budget of the BF16 tensors: each is one BF16 rounding of an FP32 result, about 1.6e-3 relative RMS
# (u = 2^-8); a 2 % error is caught. To be replaced by twice the measured value once recorded per architecture.
BF16_REL_RMS = 4e-3
CASES = [(hidden, T) for hidden in SERVED for T in (1, 257, 2049, 16263)]


@pytest.fixture(autouse=True, scope="module")
def prebuilt():
    """Build the libraries of both variants and the head first: no test pays for a build mid-call."""
    for hidden in SERVED:
        cake.prebuild("mhc", device=DEVICE, hidden=hidden, sinkhorn_iters=ops.SINKHORN_ITERS)
    cake.prebuild("mhc", device=DEVICE, hidden=4096, sinkhorn_iters=7)
    yield


def _run(i: dict, variant: str, *, pooled: bool = False, **kwargs) -> dict:
    """One connection with ``variant`` configured, and the pool on (``POOLED``) when ``pooled``."""
    cake.configure(cake_mhc_variant=variant, **(POOLED if pooled else {}))
    return connection.run(i, **kwargs)


def _assert_bf16(out: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    assert out.dtype == BF16 and tuple(out.shape) == tuple(ref.shape), (label, out.dtype, tuple(out.shape))
    err = (out.to(F64) - ref).abs()
    bad = err > BF16_ATOL + BF16_RTOL * ref.abs()
    message = f"{label}: {int(bad.sum())} of {bad.numel()} elements outside atol = rtol = 1e-2"
    assert torch.isfinite(out).all() and not bad.any(), f"{message} (max abs err {float(err.max()):.3e})"


def _assert_fp32(out: torch.Tensor, ref: torch.Tensor, label: str, *, per_element: bool) -> None:
    assert out.dtype == F32 and tuple(out.shape) == tuple(ref.shape), (label, out.dtype, tuple(out.shape))
    assert torch.isfinite(out).all(), f"{label}: non-finite values"
    err = (out.to(F64) - ref).abs()
    rel_l2 = float(err.norm() / ref.norm())
    assert rel_l2 <= FP32_REL_L2_MAX, f"{label}: rel-L2 {rel_l2:.3e} > {FP32_REL_L2_MAX:.0e}"
    if per_element:
        rms = ref.square().mean().sqrt()
        worst = float((err / torch.maximum(ref.abs(), rms)).max())
        assert worst <= FP32_ELEM_REL_MAX, f"{label}: per-element relative error {worst:.3e} > {FP32_ELEM_REL_MAX:.0e}"


def _assert_reference(out: dict, ref: dict, label: str, names=OUTPUTS) -> dict:
    """FlashInfer's bars and the relative RMS gate on ``names``; returns the measured relative RMS errors."""
    for k in names:
        if k in BF16_OUTPUTS:
            _assert_bf16(out[k], ref[k], f"{label} {k}")
        else:
            _assert_fp32(out[k], ref[k], f"{label} {k}", per_element=k != "dfn")
    budget = {k: BF16_REL_RMS if k in BF16_OUTPUTS else FP32_REL_L2_MAX for k in names}
    return assert_rel_rms({k: out[k] for k in names}, ref, budget)


def _det_group_rows(T: int, hidden: int) -> bool:
    """Whether ``det`` sums ``dscale`` / ``dbase`` over 8-token group rows at ``T`` (``fast``: its per-CTA form)."""
    jit = ops._det.cake_jit
    return jit.bwd_group_rows(T, jit.sm_count(DEVICE.index), ARCH, hidden)


def _dfn_factor(T: int, hidden: int) -> float:
    """``F`` of the ``dfn`` summation-order bound at ``T`` (FlashInfer's ``_dfn_factor``): ``2 P`` where the column
    slab sums the partials of the ``P`` chunks ``det`` sums (at most ``P`` FP32 roundings on each side), ``2 max(P_lpt,
    P) + 10`` where it runs the two-size ``lpt`` partition (the partials are another partition's)."""
    fast, det = ops._fast.cake_jit, ops._det.cake_jit
    sm_count = fast.sm_count(DEVICE.index)
    parts = fast.host_plan(T, sm_count, ARCH, hidden)["parts"]
    assert parts == det.host_plan(T, sm_count, ARCH, hidden)["parts"], (T, hidden)
    assert 1 <= parts <= 32, (T, hidden, parts)
    if not fast.bwd_slab_atomic_program(T, sm_count, ARCH, hidden).startswith("lpt"):
        return 2.0 * parts
    lpt_parts = fast.slab_plan(T, sm_count, ARCH, hidden)["lpt_parts"]
    assert 1 <= lpt_parts <= 32, (T, hidden, lpt_parts)
    return 2.0 * max(lpt_parts, parts) + 10.0


def _assert_fast_against_det(fast: dict, det: dict, ref: dict, T: int, hidden: int, label: str) -> None:
    for k in BF16_OUTPUTS:
        assert_bitwise(fast[k], det[k])
    for k in ("dscale", "dbase"):
        if _det_group_rows(T, hidden):
            _assert_fp32(fast[k], ref[k], f"{label} {k}", per_element=True)
        else:
            assert_bitwise(fast[k], det[k])
    _assert_fp32(fast["dfn"], ref["dfn"], f"{label} dfn", per_element=False)
    bound = _dfn_factor(T, hidden) * FP32_UNIT * DFN_ABS_SLACK * ref["dfn_abs"]
    excess = (fast["dfn"].to(F64) - det["dfn"].to(F64)).abs() - bound
    assert float(excess.max()) <= 0, f"{label} dfn: beyond the summation-order bound of det's dfn"


@pytest.mark.parametrize(("hidden", "T"), CASES, ids=[f"H{hidden}-T{T}" for hidden, T in CASES])
def test_det_and_fast_match_the_fp64_reference_and_each_other(hidden: int, T: int) -> None:
    i = connection.make_inputs(T, hidden, DEVICE)
    det = [_run(i, "det") for _ in range(2)]
    fast = [_run(i, "fast") for _ in range(2)]
    ref = connection.reference(i)
    label = f"H={hidden} T={T}"

    assert det[0]["handle"] is ops._det._MHCTrainHandle
    assert fast[0]["handle"] is ops._fast._MHCTrainHandle
    print(label, "det rel_rms", _assert_reference(det[0], ref, f"det {label}"))
    assert_bitwise(connection.tensors(det[1]), connection.tensors(det[0]))  # det: run to run, one process
    for n, run in enumerate(fast):
        _assert_fast_against_det(run, det[0], ref, T, hidden, f"fast run {n} {label}")
    for k in (*BF16_OUTPUTS, "dscale", "dbase"):
        assert_bitwise(fast[1][k], fast[0][k])


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_another_sinkhorn_count_and_other_epsilons_match_the_fp64_reference(variant: str) -> None:
    """The run-time-count programs (review: built by prebuild for the model's count) and the epsilon arguments."""
    constants = {"sinkhorn_iters": 7, "rms_eps": 1e-6, "hc_eps": 1e-5}
    i = connection.make_inputs(2049, 4096, DEVICE)
    ref = connection.reference(i, **constants)
    runs = [_run(i, variant, **constants) for _ in range(2)]

    names = OUTPUTS if variant == "det" else (*BF16_OUTPUTS, "dscale", "dbase")
    print(variant, "rel_rms", _assert_reference(runs[0], ref, f"{variant} iters=7", names))
    if variant == "fast":
        _assert_fp32(runs[0]["dfn"], ref["dfn"], "fast iters=7 dfn", per_element=False)
    for k in names:
        assert_bitwise(runs[1][k], runs[0][k])


# --- the variant of a call ---


def test_the_variant_is_read_when_each_call_runs() -> None:
    """configure(det), one call, configure(fast): the next call runs fast, and nothing compiles."""
    i = connection.make_inputs(257, 4096, DEVICE)
    with no_build():
        assert _run(i, "det")["handle"] is ops._det._MHCTrainHandle
        assert _run(i, "fast")["handle"] is ops._fast._MHCTrainHandle
        cake.configure()
        assert connection.run(i)["handle"] is ops._det._MHCTrainHandle


def test_deterministic_mode_without_a_variant_runs_det_bits() -> None:
    """Review: configure(Namespace(deterministic_mode=True)) without a variant ran fast."""
    i = connection.make_inputs(2049, 4096, DEVICE)
    det = _run(i, "det")

    cake.configure(argparse.Namespace(deterministic_mode=True))
    out = connection.run(i)
    assert out["handle"] is ops._det._MHCTrainHandle
    assert_bitwise(connection.tensors(out), connection.tensors(det))


def test_fast_under_torch_deterministic_algorithms_raises_torch_alert_and_det_runs() -> None:
    i = connection.make_inputs(257, 4096, DEVICE)
    det = _run(i, "det")
    torch.use_deterministic_algorithms(True)
    try:
        assert_bitwise(connection.tensors(_run(i, "det")), connection.tensors(det))
        with pytest.raises(RuntimeError, match="does not have a deterministic implementation"):
            _run(i, "fast")
    finally:
        torch.use_deterministic_algorithms(False)


# --- BF16 parameters (review: FSDP2 param_dtype=bf16 raised ValueError) ---


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_bf16_parameters_run_the_bits_of_their_fp32_values(variant: str) -> None:
    i = connection.make_inputs(2049, 4096, DEVICE)
    for k in ("fn", "scale", "base"):  # parameters representable in BF16, so both runs see the same values
        i[k] = i[k].to(BF16).to(F32)
    fp32 = _run(i, variant)
    bf16 = _run(i, variant, params_dtype=BF16)

    for k in BF16_OUTPUTS:
        assert_bitwise(bf16[k], fp32[k])
    for k in PARAMETER_GRADS:
        assert bf16[k].dtype == BF16, (k, bf16[k].dtype)
        if variant == "det" or k != "dfn":  # fast's dfn varies in its low-order bits from run to run
            assert_bitwise(bf16[k], fp32[k].to(BF16))


# --- determinism scope: process, stream, device; release ---


_CHILD = """
import sys, torch
from miles_ext_kernels import cake
from cake.mhc import _connection as connection
T, hidden, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
device = torch.device("cuda", torch.cuda.current_device())
cake.configure(cake_mhc_variant="det")
torch.save(connection.tensors(connection.run(connection.make_inputs(T, hidden, device))), path)
"""


@pytest.mark.parametrize(("hidden", "T"), [(4096, 2049), (SERVED[-1], 16263)])
def test_det_repeats_bit_for_bit_in_a_new_process(tmp_path, hidden: int, T: int) -> None:
    det = connection.tensors(_run(connection.make_inputs(T, hidden, DEVICE), "det"))
    path = tmp_path / "det.pt"
    pythonpath = os.pathsep.join(
        filter(None, [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(T), str(hidden), str(path)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert_bitwise(torch.load(path, map_location=DEVICE), det)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_calls_alternating_between_the_default_and_a_side_stream_repeat_the_bits(variant: str) -> None:
    i = connection.make_inputs(2049, 4096, DEVICE)
    first = _run(i, variant, pooled=True)
    side = torch.cuda.Stream(DEVICE)
    names = OUTPUTS if variant == "det" else (*BF16_OUTPUTS, "dscale", "dbase")
    for _ in range(2):
        side.wait_stream(torch.cuda.current_stream(DEVICE))
        with torch.cuda.stream(side):
            on_side = _run(i, variant, pooled=True)
        torch.cuda.current_stream(DEVICE).wait_stream(side)
        on_default = _run(i, variant, pooled=True)
        for out in (on_side, on_default):
            assert_bitwise({k: out[k] for k in names}, {k: first[k] for k in names})
    assert workspace.stats()[DEVICE.index]["entries"] == 4  # the forward and backward roles on each stream


def test_the_pool_holds_the_workspaces_and_release_frees_them_and_the_next_call_repeats_the_bits() -> None:
    i = connection.make_inputs(16263, 4096, DEVICE)
    det = _run(i, "det", pooled=True)
    _run(i, "fast", pooled=True)
    torch.cuda.synchronize(DEVICE)
    for package in (ops._det, ops._fast):
        assert package._DEFAULT_WORKSPACES == {} and package._DEFAULT_VIEWS == {}
    held = workspace.stats()[DEVICE.index]
    assert held["entries"] == 4 and held["bytes"] > 0  # det-fwd, det-bwd, fast-fwd, fast-bwd
    before = torch.cuda.memory_allocated(DEVICE)

    assert cake.release("mhc") == held["bytes"]
    assert workspace.stats()[DEVICE.index]["bytes"] == 0
    assert before - torch.cuda.memory_allocated(DEVICE) >= held["bytes"]
    with no_build():
        assert_bitwise(connection.tensors(_run(i, "det", pooled=True)), connection.tensors(det))


def _bucket_bytes(variant: str, T: int, hidden: int) -> tuple[int, int]:
    """The forward and backward workspace bytes of a ``variant`` call of ``T`` tokens (sized for its power-of-two
    bucket; the ``fast`` backward holds no ``dfn`` partials, so its buffer is the smaller one)."""
    package = ops._det if variant == "det" else ops._fast
    bucket = 1 << (T - 1).bit_length()
    return tuple(package._workspace_bytes(kind, bucket, DEVICE, hidden) for kind in ("fwd", "bwd"))


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("bound", ["off", "one buffer"])
def test_the_bits_repeat_under_any_pool_bound(variant: str, bound: str) -> None:
    """The bits of a pooled call (``POOLED``) repeat under bound 0: every launch allocates a buffer of its own
    (undefined content, tickets zeroed); and under one buffer: the forward's and the backward's buffers evict each
    other on every call."""
    i = connection.make_inputs(2049, 4096, DEVICE)
    first = _run(i, variant, pooled=True)
    cake.release("mhc")
    names = OUTPUTS if variant == "det" else (*BF16_OUTPUTS, "dscale", "dbase")
    limit = 0 if bound == "off" else max(_bucket_bytes(variant, 2049, 4096))
    evictions = workspace.stats()[DEVICE.index]["evictions"]
    for _ in range(2):
        cake.configure(cake_mhc_variant=variant, cake_workspace_pool_bytes=limit)
        out = connection.run(i)
        assert_bitwise({k: out[k] for k in names}, {k: first[k] for k in names})
    stats = workspace.stats()[DEVICE.index]
    assert stats["bytes"] <= limit
    if bound == "one buffer":
        assert stats["evictions"] > evictions


def test_a_buffer_of_a_larger_call_serves_a_smaller_one_and_the_bits_repeat() -> None:
    """The work tickets are the first 256 bytes of every layout, zeroed by every launch."""
    small = connection.make_inputs(257, 4096, DEVICE)
    first = connection.tensors(_run(small, "det", pooled=True))
    _run(connection.make_inputs(16263, SERVED[-1], DEVICE), "det", pooled=True)
    held = workspace.stats()[DEVICE.index]

    again = connection.tensors(_run(small, "det", pooled=True))
    stats = workspace.stats()[DEVICE.index]
    assert_bitwise(again, first)
    assert (stats["bytes"], stats["entries"]) == (held["bytes"], held["entries"])  # served by the larger buffers
    assert stats["hits"] >= held["hits"] + 2


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_interleaved_connections_repeat_the_bits_of_isolated_ones(variant: str) -> None:
    """Prepare A, prepare B, finish A, finish B, then B's backward before A's: B's larger token count replaces the
    pooled buffers A's forward used, and each backward takes its workspace when it runs."""
    inputs = [connection.make_inputs(T, 4096, DEVICE, connection.SEED + T) for T in (257, 2049)]
    alone = [_run(i, variant) for i in inputs]
    names = OUTPUTS if variant == "det" else (*BF16_OUTPUTS, "dscale", "dbase")

    cake.configure(cake_mhc_variant=variant, **POOLED)
    leaves, prepared = [], []
    for i in inputs:
        X = i["X"].detach().clone().requires_grad_(True)
        fn, scale, base = (i[k].detach().clone().requires_grad_(True) for k in ("fn", "scale", "base"))
        B = i["B"].detach().clone().requires_grad_(True)
        leaves.append({"X": X, "fn": fn, "scale": scale, "base": base, "B": B})
        prepared.append(ops.mhc_prepare(X, fn, scale, base))
    Ys = [ops.mhc_finish(leaf["X"], leaf["B"], handle) for leaf, (_, handle) in zip(leaves, prepared)]
    for n in (1, 0):
        torch.autograd.backward([prepared[n][0], Ys[n]], [inputs[n]["dU"], inputs[n]["dY"]])

    for n, leaf in enumerate(leaves):
        out = {"U": prepared[n][0].detach(), "Y": Ys[n].detach(), "dB": leaf["B"].grad, "dX": leaf["X"].grad}
        out.update(dfn=leaf["fn"].grad, dscale=leaf["scale"].grad, dbase=leaf["base"].grad)
        assert_bitwise({k: out[k] for k in names}, {k: alone[n][k] for k in names})


def _board(device: torch.device) -> tuple:
    """The GPU model and SM count of ``device``: det repeats its bits across devices of one board."""
    properties = torch.cuda.get_device_properties(device)
    return tuple(torch.cuda.get_device_capability(device)), properties.multi_processor_count


@pytest.mark.multi_gpu
def test_det_on_a_second_gpu_repeats_the_bits_and_prebuild_takes_the_device() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    other = torch.device("cuda", (DEVICE.index + 1) % torch.cuda.device_count())
    if _board(other) != _board(DEVICE):
        pytest.skip("the second GPU is another model")
    paths = cake.prebuild("mhc", device=other.index, hidden=4096, sinkhorn_iters=ops.SINKHORN_ITERS)
    assert paths and all(path.is_file() for path in paths)

    here = connection.tensors(_run(connection.make_inputs(2049, 4096, DEVICE), "det"))
    with torch.cuda.device(other), no_build():
        there = connection.tensors(_run(connection.make_inputs(2049, 4096, other), "det"))
    assert_bitwise({k: v.cpu() for k, v in there.items()}, {k: v.cpu() for k, v in here.items()})


# --- the device, the model shape and the toolkit ---


def test_the_device_serves_its_hidden_sizes_and_refuses_the_others() -> None:
    assert ops.supported_hidden_sizes(DEVICE) == SERVED
    for hidden in ops.HIDDEN_SIZES:
        assert ops.is_supported(DEVICE, hidden) is (hidden in SERVED)
        if hidden not in SERVED:
            with pytest.raises(ValueError, match=rf"hidden sizes \(4096,\) on {ARCH}, not {hidden}"):
                cake.prebuild("mhc", device=DEVICE, hidden=hidden, sinkhorn_iters=ops.SINKHORN_ITERS)


def test_prebuild_needs_a_served_device_and_the_architectures_toolkit(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(NotImplementedError, match="compute capability 9.0"):
        cake.prebuild("mhc", device=DEVICE, hidden=4096, sinkhorn_iters=20)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(loader, "MIN_CUDA", {**loader.MIN_CUDA, "sm_107a": (99, 0)})
    with pytest.raises(RuntimeError, match=r"mHC kernels for sm_107a needs CUDA 99\.0 or newer"):
        cake.prebuild("mhc", device=DEVICE, hidden=4096, sinkhorn_iters=20)


def test_prebuild_is_idempotent_and_returns_cached_libraries() -> None:
    first = cake.prebuild("mhc", device=DEVICE, hidden=4096, sinkhorn_iters=7)
    assert first == cake.prebuild("mhc", device=DEVICE.index, hidden=4096, sinkhorn_iters=7)
    assert all(path.is_file() and path.is_relative_to(loader.cache_dir()) for path in first)
    assert len(first) == len(set(first)) == sum(map(len, ops.libraries(ARCH, 4096, 7).values()))


def test_finish_refuses_a_foreign_handle() -> None:
    i = connection.make_inputs(1, 4096, DEVICE)
    with pytest.raises(TypeError, match="handle must be the handle mhc_prepare returned"):
        ops.mhc_finish(i["X"], i["B"], object())


@pytest.mark.parametrize("hidden", SERVED)
@pytest.mark.parametrize("T", [0, 1, 257, 16263])
def test_the_head_is_torch_mean_bit_for_bit(hidden: int, T: int) -> None:
    g = torch.Generator(device=DEVICE).manual_seed(connection.SEED)
    X = torch.randn(T, STREAMS, hidden, generator=g, device=DEVICE).to(BF16)
    dY = torch.randn(T, hidden, generator=g, device=DEVICE).to(BF16)
    Xk, Xr = X.clone().requires_grad_(True), X.clone().requires_grad_(True)

    Y, Yr = ops.mhc_head(Xk), Xr.mean(dim=1)
    assert_bitwise(Y, Yr)
    Y.backward(dY)
    Yr.backward(dY)
    assert_bitwise(Xk.grad, Xr.grad)


def test_nothing_imports_flashinfer() -> None:
    assert json.dumps(sorted(name for name in sys.modules if name.split(".")[0] == "flashinfer")) == "[]"
