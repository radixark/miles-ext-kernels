"""The Cake short-conv kernels (``miles_ext_kernels.cake.conv1d``) against a document-local FP64 reference.

Both variants at FlashInfer's edge cases, at miles' TP-local channel counts (GDN and KDA layers of 2560 to 24576
channels per rank) on ~32k packed tokens with documents shorter than the kernel, with FP32 and BF16 taps, and at
FlashInfer's S x C > 2^31 case. ``y`` and ``dx`` within one BF16 ulp of the reference; ``dweight`` within the
deterministic package's budgets; ``det`` bitwise run to run, in one process and in a new one; ``fast`` with ``y`` and
``dx`` bitwise ``det``'s and ``dweight`` within budget of ``det``; ``fast`` runs ``det`` under torch's deterministic
algorithms. Also: an unconfigured process runs ``det``; ``cu_seqlens=None`` neither synchronizes the host nor waits
for queued work; calls on a side stream, interleaved with calls on the default stream, give the same bits. Needs
SM100a / SM103a / SM107a; skips elsewhere.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.conv1d import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402

from ..helpers import assert_bitwise  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake short-conv kernels need SM100a / SM103a / SM107a", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[3]

WIDTH = 4
F32, BF16 = torch.float32, torch.bfloat16
DEVICE = torch.device("cuda", torch.cuda.current_device())
# FlashInfer's contract: y / dx within one BF16 ulp of the reference; dweight within atol 5e-2 + rtol 1e-4 of the FP64
# sum under the bf16 SiLU' policy and atol 1e-3 + rtol 2e-6 under the fp32 policy (a BF16 dweight widened by its own
# rounding).
Y_ATOL, Y_RTOL = 1e-3, 2**-7
DW_ATOL, DW_RTOL = 5e-2, 1e-4
DW_FP32G_ATOL, DW_FP32G_RTOL = 1e-3, 2e-6
BLOCK = 8192  # the reference's token block
COMPARE_CHUNK = 2**26
PERF_FRACTIONS = (0.37, 0.29, 0.21)
SEED = 2033
REPEATS = 2
NAMES = ("y", "dx", "dweight")

# FlashInfer's edge cases (S, C, documents): documents of 1-3 tokens and empty ones, sub-tile boundaries, the perf
# packing, multi-tile channel counts and the fallback channel splits.
CASES = (
    (5, 136, (2, 0, 3)),
    (63, 384, (1, 2, 3, 57)),
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (4097, 2560, (1000, 0, 3, 3094)),
    (1000, 1600, "perf"),
)
# miles' per-rank conv channels of its GDN and KDA layers.
MILES_CHANNELS = (2560, 3072, 4096, 8192, 10240, 24576)
# ~32k packed tokens: documents shorter than the kernel (1, 2, 3 tokens), an empty one and two long ones.
MILES_DOCS = (1, 2, 3, 0, 9000, 3, 23762)
LARGE_CASE = (262401, 8192, "perf")  # S x C > 2^31
LARGE_CASE_FREE_BYTES = 40 * 2**30


def doc_lengths(S, docs):
    if docs is None:
        return (S,)
    if docs == "perf":
        lens = []
        for fraction in PERF_FRACTIONS:
            n = int(fraction * S)
            lens.append(n - 1 if n % 64 == 0 else n)
        return (*lens, S - sum(lens))
    assert sum(docs) == S
    return tuple(docs)


def _inputs(S, C, docs, *, taps, key=0):
    """``(x, w, cu, dy)``: BF16 ``x`` / ``dy`` of N(0, 1), taps of N(0, 0.25) in ``taps``."""
    generator = torch.Generator(device=DEVICE).manual_seed((SEED * 1_000_003 + S * 7919 + C * 31 + key) % (2**31 - 1))
    x = torch.randn(1, S, C, device=DEVICE, generator=generator).bfloat16()
    w = (0.5 * torch.randn(C, WIDTH, device=DEVICE, generator=generator)).to(taps)
    dy = torch.randn(1, S, C, device=DEVICE, generator=generator).bfloat16()
    lens = doc_lengths(S, docs)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE)
    return x, w, cu, dy


def _shift(v, ids, d):
    """``out[t] = v[t - d]`` (``d > 0``) or ``v[t + |d|]`` (``d < 0``) within token ``t``'s document, else 0."""
    out = torch.zeros_like(v)
    if d == 0:
        return v.clone()
    n = abs(d)
    if n >= v.shape[0]:
        return out
    if d > 0:
        out[n:] = torch.where((ids[n:] == ids[:-n]).unsqueeze(1), v[:-n], 0.0)
    else:
        out[:-n] = torch.where((ids[:-n] == ids[n:]).unsqueeze(1), v[n:], 0.0)
    return out


def _preactivation(x, ids, w):
    pre = torch.zeros_like(x)
    for k in range(WIDTH):
        pre = pre + w[:, k].to(x.dtype) * _shift(x, ids, WIDTH - 1 - k)
    return pre


def reference(x, w, cu, dy, policy="bf16"):
    """``(y, dx, dweight)``: BF16 ``y`` / ``dx`` of the FP32 (fp32 policy: FP64) math, FP64 ``dweight``.

    The bf16 policy evaluates SiLU' at the BF16-rounded FP32 preactivation, as the kernels do (FlashInfer's
    reference); the fp32 policy evaluates everything in FP64 and rounds once."""
    x, dy = x[0], dy[0]
    S, C = x.shape
    ids = torch.bucketize(torch.arange(S, device=x.device), cu[1:].long(), right=True)
    y = torch.empty_like(x)
    dx = torch.empty_like(x)
    dw = torch.zeros(C, WIDTH, dtype=torch.float64, device=x.device)
    for a in range(0, S, BLOCK):
        b = min(a + BLOCK, S)
        lo, hi = max(a - (WIDTH - 1), 0), min(b + (WIDTH - 1), S)
        win = ids[lo:hi]
        rows = slice(a - lo, b - lo)
        pre32 = _preactivation(x[lo:hi].float(), win, w)
        y[a:b] = (pre32 * torch.sigmoid(pre32)).bfloat16()[rows]
        if policy == "fp32":
            xf = x[lo:hi].double()
            p = _preactivation(xf, win, w)
        else:
            xf = x[lo:hi].float()
            p = pre32.bfloat16().float()
        s = torch.sigmoid(p)
        dpre = dy[lo:hi].to(xf.dtype) * (s * (1.0 + p * (1.0 - s)))
        dxw = torch.zeros_like(dpre)
        for k in range(WIDTH):
            dxw = dxw + w[:, k].to(xf.dtype) * _shift(dpre, win, -(WIDTH - 1 - k))
        dx[a:b] = dxw[rows].bfloat16()
        for k in range(WIDTH):
            dw[:, k] += (dpre[rows].double() * _shift(xf, win, WIDTH - 1 - k)[rows].double()).sum(dim=0)
    return y.unsqueeze(0), dx.unsqueeze(0), dw


def _grad_tolerance(policy, dtype):
    atol, rtol = (DW_ATOL, DW_RTOL) if policy == "bf16" else (DW_FP32G_ATOL, DW_FP32G_RTOL)
    if dtype == BF16:
        return atol * (1 + 2**-8), 2**-7
    return atol, rtol


def assert_within(actual, expected, *, atol, rtol, label):
    """No element outside ``atol + rtol * |expected|``."""
    assert tuple(actual.shape) == tuple(expected.shape), label
    flat_actual, flat_expected = actual.reshape(-1), expected.reshape(-1)
    count, first = 0, None
    for start in range(0, flat_actual.numel(), COMPARE_CHUNK):
        a = flat_actual[start : start + COMPARE_CHUNK].double()
        b = flat_expected[start : start + COMPARE_CHUNK].double()
        assert torch.isfinite(a).all(), f"{label} has non-finite values"
        outside = (a - b).abs() > atol + rtol * b.abs()
        chunk = int(outside.sum())
        if chunk and first is None:
            index = int(outside.nonzero()[0])
            first = (start + index, a[index].item(), b[index].item())
        count += chunk
    assert (
        not count
    ), f"{label}: {count} of {flat_actual.numel()} elements outside atol={atol} rtol={rtol}; first {first}"


def run(variant, x, w, cu, dy, policy="bf16"):
    """``(y, dx, dweight)`` of one autograd forward + backward of ``variant`` (configured for it; ``None``: the
    variant configured now) and ``policy`` (``None``: the configured one)."""
    if variant is not None:
        cake.configure(cake_conv1d_variant=variant)
    xi = x.detach().requires_grad_()  # new leaves sharing the inputs' storage
    wi = w.detach().requires_grad_()
    y = ops.causal_conv1d(xi, wi, cu, silu_grad_preactivation=policy)
    y.backward(dy)
    assert y.dtype == BF16 and y.shape == x.shape
    assert xi.grad.dtype == BF16 and wi.grad.dtype == w.dtype and wi.grad.shape == w.shape
    return y.detach(), xi.grad, wi.grad


def check(x, w, cu, dy, *, policy="bf16", label):
    """``det`` against the reference and bitwise run to run; ``fast`` with ``y`` / ``dx`` bitwise ``det``'s and
    ``dweight`` within budget of the reference and of ``det``."""
    y_ref, dx_ref, dw_ref = reference(x, w, cu, dy, policy)
    atol, rtol = _grad_tolerance(policy, w.dtype)
    det = run("det", x, w, cu, dy, policy)
    assert_within(det[0], y_ref, atol=Y_ATOL, rtol=Y_RTOL, label=f"det y {label}")
    assert_within(det[1], dx_ref, atol=Y_ATOL, rtol=Y_RTOL, label=f"det dx {label}")
    assert_within(det[2].float(), dw_ref.float(), atol=atol, rtol=rtol, label=f"det dweight {label}")
    for _ in range(REPEATS):
        again = run("det", x, w, cu, dy, policy)
        for name, a, b in zip(("y", "dx", "dweight"), det, again, strict=True):
            assert torch.equal(a, b), f"det {name} differs between runs ({label})"
    fast = run("fast", x, w, cu, dy, policy)
    assert torch.equal(fast[0], det[0]), f"fast y differs from det ({label})"
    assert torch.equal(fast[1], det[1]), f"fast dx differs from det ({label})"
    assert_within(fast[2].float(), dw_ref.float(), atol=atol, rtol=rtol, label=f"fast dweight {label}")
    assert_within(fast[2].float(), det[2].float(), atol=atol, rtol=rtol, label=f"fast dweight vs det {label}")


def _case_id(case):
    S, C, docs = case
    return f"S{S}-C{C}-{docs if isinstance(docs, str) else len(doc_lengths(S, docs))}docs"


@pytest.mark.parametrize("taps", [F32, BF16], ids=["fp32-taps", "bf16-taps"])
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_edge_cases_match_the_reference(case, taps):
    S, C, docs = case
    check(*_inputs(S, C, docs, taps=taps), label=f"{case} {taps}")


@pytest.mark.parametrize("taps", [F32, BF16], ids=["fp32-taps", "bf16-taps"])
@pytest.mark.parametrize("channels", MILES_CHANNELS)
def test_miles_shapes_match_the_reference(channels, taps):
    S = sum(MILES_DOCS)
    check(*_inputs(S, channels, MILES_DOCS, taps=taps), label=f"S={S} C={channels} {taps}")


@pytest.mark.parametrize("taps", [F32, BF16], ids=["fp32-taps", "bf16-taps"])
def test_the_fp32_silu_policy_matches_the_reference(taps):
    check(*_inputs(4097, 1280, (1000, 0, 3, 3094), taps=taps), policy="fp32", label=f"fp32 policy {taps}")


def test_more_than_2_to_the_31_elements():
    S, C, docs = LARGE_CASE
    free, _ = torch.cuda.mem_get_info(DEVICE)
    if free < LARGE_CASE_FREE_BYTES:
        pytest.skip(f"needs {LARGE_CASE_FREE_BYTES >> 30} GiB free, {free >> 30} GiB available")
    check(*_inputs(S, C, docs, taps=F32), label=str(LARGE_CASE))


def test_a_strided_input_matches_the_contiguous_one():
    # miles' GDN q/k/v is a column slice of the fused projection
    S, C = 1000, 1536
    x, w, cu, dy = _inputs(S, C, "perf", taps=BF16)
    wide = torch.randn(1, S, C + 520, device=DEVICE).bfloat16()
    wide[..., :C] = x
    strided = wide[..., :C]
    assert not strided.is_contiguous()
    for variant in ("det", "fast"):
        expected = run(variant, x, w, cu, dy)[0]
        assert torch.equal(ops.causal_conv1d(strided, w, cu), expected)


def test_no_cu_seqlens_is_one_document():
    x, w, cu, dy = _inputs(4097, 1024, None, taps=F32)
    for variant in ("det", "fast"):
        cake.configure(cake_conv1d_variant=variant)
        assert torch.equal(ops.causal_conv1d(x, w), ops.causal_conv1d(x, w, cu))


def test_a_batch_of_several_sequences_is_rejected():
    x = torch.zeros(2, 64, 256, dtype=BF16, device=DEVICE)
    w = torch.zeros(256, WIDTH, device=DEVICE)
    with pytest.raises(ValueError, match="one packed batch"):
        ops.causal_conv1d(x, w)


def test_fast_runs_det_under_torch_deterministic_algorithms(monkeypatch, caplog):
    monkeypatch.setattr(config, "_warned", set())
    x, w, cu, dy = _inputs(4097, 2560, (1000, 0, 3, 3094), taps=BF16)
    det = run("det", x, w, cu, dy)
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            runs_det = [run("fast", x, w, cu, dy) for _ in range(2)]
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    for result in runs_det:
        for name, a, b in zip(("y", "dx", "dweight"), det, result, strict=True):
            assert torch.equal(a, b), f"fast running det: {name} differs from det"
    warnings = [r for r in caplog.records if "conv1d" in r.getMessage() and "running det" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]


@pytest.fixture
def entries(monkeypatch):
    """The package entry each call runs, with its SiLU' policy, recorded (the calls still run the kernels)."""
    calls = []
    for variant, name in (("det", "cake_causal_conv1d_det"), ("fast", "cake_causal_conv1d")):
        package = ops._package(variant)
        original = getattr(package, name)

        def record(*args, _variant=variant, _original=original, **kwargs):
            calls.append((_variant, kwargs.get("silu_grad_preactivation")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(package, name, record)
    return calls


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_selects_the_variant_of_a_call(entries, variant):
    cake.configure(cake_conv1d_variant=variant, cake_conv1d_silu_grad_preactivation="fp32")
    x, w, cu, _ = _inputs(1000, 512, "perf", taps=F32)

    ops.causal_conv1d(x, w, cu)
    assert entries == [(variant, "fp32")]


@pytest.mark.parametrize("args", [None, argparse.Namespace()], ids=["never-configured", "empty-namespace"])
def test_an_unconfigured_process_runs_det(entries, args):
    """The default of a process that configured no variant is det, whose weight gradient repeats bit for bit."""
    x, w, cu, dy = _inputs(4097, 2560, (1000, 0, 3, 3094), taps=BF16)
    if args is not None:
        cake.configure(args)
    default = run(None, x, w, cu, dy, None)
    det = run("det", x, w, cu, dy)
    assert entries == [("det", "bf16"), ("det", "bf16")]
    assert_bitwise(dict(zip(NAMES, default)), dict(zip(NAMES, det)))


def test_miles_convolutions_are_supported():
    for channels in MILES_CHANNELS:
        ops.check_supported(channels, 4, "silu")


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_no_cu_seqlens_does_not_synchronize(variant):
    """One document without ``cu_seqlens``: the forward and the backward make no synchronizing call (the prefix
    sums used to be a copy from pageable host memory)."""
    x, w, _, dy = _inputs(4097, 2560, None, taps=F32)
    run(variant, x, w, None, dy)  # loads the libraries
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run(variant, x, w, None, dy)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def _cycles_per_ms() -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(10**7)
    end.record()
    end.synchronize()
    return 10**7 / start.elapsed_time(end)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_no_cu_seqlens_returns_before_queued_work_finishes(variant):
    """Behind 200 ms of queued GPU work the forward returns at once (it waited for the queue before)."""
    x, w, _, _ = _inputs(4097, 2560, None, taps=F32)
    cake.configure(cake_conv1d_variant=variant)
    ops.causal_conv1d(x, w)
    torch.cuda.synchronize()
    torch.cuda._sleep(int(200 * _cycles_per_ms()))
    start = time.perf_counter()
    ops.causal_conv1d(x, w)
    host_ms = 1e3 * (time.perf_counter() - start)
    torch.cuda.synchronize()
    assert host_ms < 100, f"the call took {host_ms:.1f} ms of host time behind 200 ms of queued work"


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_side_stream_calls_give_the_default_stream_bits(variant):
    """Calls alternating between the default stream and a side stream, both in flight at once: ``det`` gives the
    same bits on both, ``fast`` the same ``y`` and ``dx`` and a weight gradient within budget."""
    x, w, cu, dy = _inputs(4097, 2560, (1000, 0, 3, 3094), taps=BF16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    results = []
    for on_side in (False, True, False, True):
        if on_side:
            with torch.cuda.stream(side):
                results.append(run(variant, x, w, cu, dy))
        else:
            results.append(run(variant, x, w, cu, dy))
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    atol, rtol = _grad_tolerance("bf16", BF16)
    first = dict(zip(NAMES, results[0]))
    for result in results[1:]:
        result = dict(zip(NAMES, result))
        if variant == "det":
            assert_bitwise(result, first)
        else:
            assert_bitwise({k: result[k] for k in ("y", "dx")}, {k: first[k] for k in ("y", "dx")})
            assert_within(result["dweight"].float(), first["dweight"].float(), atol=atol, rtol=rtol, label="dweight")


# (S, C, documents, taps, SiLU' policy) of the det runs a new process repeats
XPROC_CASES = (
    (4097, 2560, (1000, 0, 3, 3094), F32, "bf16"),
    (4097, 1280, (1000, 0, 3, 3094), BF16, "fp32"),
    (sum(MILES_DOCS), 8192, MILES_DOCS, BF16, "bf16"),
)
_XPROC = """
import sys
import torch
sys.path.insert(0, {tests!r})
from cake.conv1d import test_conv1d_gpu as t
out = {{}}
for key, (S, C, docs, taps, policy) in enumerate(t.XPROC_CASES):
    out[key] = [v.cpu() for v in t.run("det", *t._inputs(S, C, docs, taps=taps), policy)]
torch.save(out, {path!r})
"""


def test_det_repeats_bitwise_in_a_new_process(tmp_path):
    path = tmp_path / "det.pt"
    script = _XPROC.format(tests=str(REPO_ROOT / "tests"), path=str(path))
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": pythonpath}, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]
    child = torch.load(path)
    for key, (S, C, docs, taps, policy) in enumerate(XPROC_CASES):
        here = run("det", *_inputs(S, C, docs, taps=taps), policy)
        assert_bitwise(dict(zip(NAMES, (v.cpu() for v in here))), dict(zip(NAMES, child[key])))
