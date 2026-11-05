"""The Cake zero-centred input RMSNorm (``miles_ext_kernels.cake.rmsnorm``) against an FP64 reference, on a GPU.

miles' GDN input layernorm: ``[T, 1, H]`` BF16 hidden states at H = 2048 / 4096 / 5120 (Qwen3.5 / 3.6 widths the
kernels serve) and small, irregular and 16k-32k token counts, ``weight`` the w of ``(1 + w)`` in BF16 or FP32. ``y``
and ``dx`` at FlashInfer's tolerance; the FP32 weight gradient within its budget (rel-L2 <= 1e-4, elementwise
atol = rtol = 1e-4), a BF16 one widened by its rounding; ``det`` bitwise run to run, twice within a process and once
in a fresh one, also at both sides of the upper edge of each of its backward windows at the smallest token counts
whose kernel sums each ``dw`` column in a fixed row order, and on a side stream; ``fast`` with ``y`` bitwise ``det``'s,
``dx`` reproducible and within one BF16 ulp of ``det``'s; ``fast`` runs ``det`` under torch's deterministic
algorithms; a backward runs its forward's variant. A width whose oversized shared-memory programs this process cannot
launch (SM107a at 4096 without a CUDA 13.4 runtime) is refused before any launch: its tests assert that refusal, and
the refusal holds exactly where an unchecked launch fails. Needs SM100a / SM103a / SM107a; skips elsewhere.
"""

import hashlib
import importlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake import rmsnorm
from miles_ext_kernels.cake._runtime import config, loader

from ..helpers import assert_bitwise, assert_repeatable, no_build, requires_arch
from ._refusal import refusal, runs_or_is_refused
from ._windows import backward_form

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")

EPS = 1e-6
REPO_ROOT = Path(__file__).resolve().parents[3]
HIDDEN_SIZES = (2048, 4096, 5120)
TOKENS = (1, 7, 1000, 16385, 32769)
# FlashInfer's budgets: y / dx within atol = rtol = 1e-2 of the reference scaled by its RMS; the FP32 dw within
# rel-L2 1e-4 and elementwise atol = rtol = 1e-4 of the FP64 sum.
ATOL = RTOL = 1e-2
DW_REL_L2 = 1e-4
DW_ATOL = DW_RTOL = 1e-4


@pytest.fixture(scope="module", autouse=True)
def _prebuilt():
    """Every library of the widths below that this process can launch, built once: the tests then run with no build
    allowed where they say so (a refused width builds nothing)."""
    for hidden in HIDDEN_SIZES:
        if refusal(hidden) is None:
            cake.prebuild("rmsnorm", hidden_size=hidden)


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def make(T, H, *, seed=0):
    """``(x, w, dy)``: BF16 ``[T, 1, H]`` hidden states and gradient of N(0, 1), BF16 ``w`` of N(0, 0.01)."""
    g = torch.Generator(device=_device()).manual_seed(seed * 1_000_003 + T * 131 + H)
    x = torch.randn(T, 1, H, device=_device(), generator=g).bfloat16()
    dy = torch.randn(T, 1, H, device=_device(), generator=g).bfloat16()
    w = (0.1 * torch.randn(H, device=_device(), generator=g)).bfloat16()
    return x, w, dy


def reference(x, w, dy):
    """FP64 ``(y, dx, dw)`` of ``x * rstd * (1 + w)`` at the kernels' FP32 effective weight ``1 + w``."""
    xd, gd = x.double(), dy.double()
    weight = (1.0 + w.float()).double()
    rstd = torch.rsqrt(xd.square().mean(-1, keepdim=True) + EPS)
    y = xd * rstd * weight
    a = gd * weight
    dx = rstd * a - xd * rstd**3 * (a * xd).mean(-1, keepdim=True)
    dw = (gd * xd * rstd).reshape(-1, x.shape[-1]).sum(0)
    return y, dx, dw


def assert_close_at_rms(actual, expected, label):
    assert actual.dtype == torch.bfloat16, label
    a, b = actual.double(), expected.double()
    assert torch.isfinite(a).all(), f"{label} has non-finite values"
    scale = b.square().mean().sqrt().item() or 1.0
    torch.testing.assert_close(a / scale, b / scale, atol=ATOL, rtol=RTOL, msg=label)


def assert_dw(dw, dw_ref, label):
    """FP32: FlashInfer's budget; BF16: the FP32 budget widened by the BF16 rounding."""
    diff = (dw.double() - dw_ref).abs()
    if dw.dtype == torch.float32:
        rel_l2 = ((dw.double() - dw_ref).norm() / dw_ref.norm()).item()
        assert rel_l2 <= DW_REL_L2, f"{label}: dw rel-L2 {rel_l2:.3e}"
        bound = DW_ATOL + DW_RTOL * dw_ref.abs()
    else:
        bound = 2**-8 * dw_ref.abs() + (1 + 2**-8) * (DW_ATOL + DW_RTOL * dw_ref.abs())
    bad = diff > bound
    assert not bad.any(), f"{label}: {int(bad.sum())} dw elements outside the budget"


def run(variant, x, w, dy, *, params_dtype=torch.bfloat16, use_checkpoint=False):
    """``{"y", "dx", "dw"}`` of one forward + backward of a :class:`ZeroCenteredRMSNorm` holding ``w``, with
    ``variant`` configured (``None``: the configuration as it is)."""
    from torch.utils.checkpoint import checkpoint

    if variant is not None:
        cake.configure(cake_rmsnorm_variant=variant)
    norm = rmsnorm.ZeroCenteredRMSNorm(x.shape[-1], EPS, params_dtype=params_dtype)
    with torch.no_grad():
        norm.weight.copy_(w)
    xi = x.detach().requires_grad_()
    y = checkpoint(norm, xi, use_reentrant=False) if use_checkpoint else norm(xi)
    y.backward(dy)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    assert xi.grad.dtype == torch.bfloat16 and norm.weight.grad.dtype == params_dtype
    return {"y": y.detach(), "dx": xi.grad, "dw": norm.weight.grad}


@pytest.mark.parametrize("params_dtype", [torch.bfloat16, torch.float32], ids=["bf16-weight", "fp32-weight"])
@pytest.mark.parametrize("T", TOKENS)
@pytest.mark.parametrize("H", HIDDEN_SIZES)
def test_det_and_fast_match_fp64(H, T, params_dtype):
    x, w, dy = make(T, H)
    w = w.to(params_dtype)
    y_ref, dx_ref, dw_ref = reference(x, w, dy)
    with runs_or_is_refused(H):
        det = assert_repeatable(lambda: run("det", x, w, dy, params_dtype=params_dtype), times=3)
        assert_close_at_rms(det["y"], y_ref, "det y")
        assert_close_at_rms(det["dx"], dx_ref, "det dx")
        assert_dw(det["dw"], dw_ref, "det dw")
        fast = run("fast", x, w, dy, params_dtype=params_dtype)
        assert_bitwise(fast["y"], det["y"])
        ulp = 2**-7 * torch.maximum(det["dx"].double().abs(), det["dx"].double().square().mean().sqrt())
        assert ((fast["dx"].double() - det["dx"].double()).abs() <= ulp).all(), "fast dx beyond one BF16 ulp of det"
        assert_dw(fast["dw"], dw_ref, "fast dw")
        for _ in range(2):
            assert_bitwise(run("fast", x, w, dy, params_dtype=params_dtype)["dx"], fast["dx"])


def smallest_token_counts(H):
    """1, 2 and both sides of the upper edge of the first window of ``det``'s FP32-weight backward of ``H`` here and of
    each next window while the one above it still sums each ``dw`` column in a fixed row order (the backward of the
    smallest token counts of every width but 256; SM103a runs two such kernels at H = 512, 1536 and 2048)."""
    arch = cake.device_arch()
    det = loader.load_package("cake_rmsnorm_det_train_backend")
    edges = []
    for _, edge, _ in det.cake_jit.program_windows(arch, H, "bwd", False, "f32"):
        if edge is None:
            break
        edges.append(edge)
        if backward_form("det", arch, H, edge + 1) != "column":
            break
    return sorted({1, 2} | {T for edge in edges for T in (edge, edge + 1)})


def det_digests(cases):
    """``{"H T": [sha256 of y, dx, dw]}`` of one ``det`` forward + backward with an FP32 weight per ``(H, T)``."""
    digests = {}
    for H, T in cases:
        x, w, dy = make(T, H)
        outputs = run("det", x, w.float(), dy, params_dtype=torch.float32)
        digests[f"{H} {T}"] = [
            hashlib.sha256(outputs[name].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for name in ("y", "dx", "dw")
        ]
    return digests


@pytest.mark.parametrize("H", HIDDEN_SIZES)
def test_det_repeats_bit_for_bit_at_the_smallest_token_counts(H):
    for T in smallest_token_counts(H):
        x, w, dy = make(T, H)
        y_ref, dx_ref, dw_ref = reference(x, w, dy)
        with runs_or_is_refused(H):
            first = assert_repeatable(lambda: run("det", x, w.float(), dy, params_dtype=torch.float32), times=3)
            assert_close_at_rms(first["y"], y_ref, f"det y, T={T}")
            assert_close_at_rms(first["dx"], dx_ref, f"det dx, T={T}")
            assert_dw(first["dw"], dw_ref, f"det dw, T={T}")


_FRESH_PROCESS = """
import json
import sys

from cake.rmsnorm import test_rmsnorm_gpu as module

print(json.dumps(module.det_digests(json.loads(sys.argv[1]))))
"""


def test_det_repeats_bit_for_bit_in_a_fresh_process():
    """Every width this process can launch (a refused width is refused in the fresh process too)."""
    cases = [(H, T) for H in HIDDEN_SIZES if refusal(H) is None for T in (*smallest_token_counts(H), 1000, 16385)]
    expected = det_digests(cases)
    paths = [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")]
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS, json.dumps(cases)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1]) == expected


@pytest.mark.parametrize("H", (4096, 5120))
def test_det_on_a_side_stream_repeats_the_default_stream_bit_for_bit(H):
    x, w, dy = make(16385, H)
    side = torch.cuda.Stream()
    with runs_or_is_refused(H):
        on_default = run("det", x, w, dy)
        results = []
        for stream in (side, torch.cuda.current_stream(), side):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                results.append(run("det", x, w, dy))
            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for result in results:
            assert_bitwise(result, on_default)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_strided_rows_match_contiguous_rows(variant):
    T, H = 1000, 2048
    x, w, dy = make(T, H)
    wide = torch.zeros(T, 1, H + 64, dtype=torch.bfloat16, device=_device())
    wide[..., :H] = x
    strided = wide[..., :H]
    assert not strided.is_contiguous()
    plain, from_strided = run(variant, x, w, dy), run(variant, strided, w, dy)
    assert_bitwise({k: plain[k] for k in ("y", "dx")}, {k: from_strided[k] for k in ("y", "dx")})


@pytest.mark.parametrize("H", (4096, 5120))
def test_non_reentrant_checkpoint_matches_the_plain_call(H):
    x, w, dy = make(16385, H)
    with runs_or_is_refused(H):
        assert_bitwise(run("det", x, w, dy, use_checkpoint=True), run("det", x, w, dy))


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_no_tokens(variant):
    x, w, dy = make(0, 2048)
    out = run(variant, x, w, dy)
    assert out["y"].shape == (0, 1, 2048) and out["dx"].shape == (0, 1, 2048)
    assert not out["dw"].any()


def test_the_module_has_the_te_parameter_surface_on_the_current_device():
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS)
    assert list(norm.state_dict()) == ["weight"]
    assert norm.weight.dtype == torch.bfloat16 and norm.weight.device == _device()
    assert not norm.weight.any()
    x = make(7, 2048)[0]
    assert_bitwise(norm(x), rmsnorm.zero_centered_rmsnorm(x, torch.zeros(2048, device=_device()), EPS))


def test_a_module_built_on_the_meta_device_runs_after_to_empty_and_reset_parameters():
    x, w, dy = make(1000, 2048)
    with torch.device("meta"):
        norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS)
    norm.to_empty(device=_device())
    norm.reset_parameters()
    with torch.no_grad():
        norm.weight.copy_(w)
    cake.configure(cake_rmsnorm_variant="det")
    xi = x.detach().requires_grad_()
    y = norm(xi)
    y.backward(dy)
    assert_bitwise({"y": y.detach(), "dx": xi.grad, "dw": norm.weight.grad}, run("det", x, w, dy))


def test_construction_on_a_gpu_compiles_nothing(monkeypatch, tmp_path):
    """Reviewer item: a module built with det configured compiled at its construction and, after configure(fast), its
    first call compiled fast's programs (16 s of nvcc). Here the cache is empty and nothing may build."""
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path))
    with no_build():
        for variant in ("det", "fast"):
            cake.configure(cake_rmsnorm_variant=variant)
            rmsnorm.ZeroCenteredRMSNorm(2048, EPS)
    assert not any(tmp_path.rglob("*.so"))


def test_the_device_needs_its_architectures_toolkit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(NotImplementedError, match=r"SM100a, SM103a or SM107a"):
        rmsnorm.check_supported(2048)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(loader, "MIN_CUDA", {**loader.MIN_CUDA, "sm_107a": (99, 0)})
    with pytest.raises(RuntimeError, match=r"RMSNorm kernels for sm_107a needs CUDA 99\.0 or newer"):
        rmsnorm.check_supported(2048)
    with pytest.raises(RuntimeError, match=r"needs CUDA 99\.0 or newer"):
        rmsnorm.ZeroCenteredRMSNorm(2048, EPS)


def test_fast_runs_det_under_torch_deterministic_algorithms(monkeypatch, caplog):
    monkeypatch.setattr(config, "_warned", set())
    x, w, dy = make(16385, 2048)
    det = run("det", x, w, dy)
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            runs_det = [run("fast", x, w, dy) for _ in range(2)]
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    for result in runs_det:
        assert_bitwise(result, det)
    warnings = [r for r in caplog.records if "rmsnorm" in r.getMessage() and "running det" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("forward, backward", [("fast", "det"), ("det", "fast")])
def test_a_backward_runs_the_variant_of_its_forward(monkeypatch, forward, backward):
    fast = loader.load_package("cake_rmsnorm_train_backend")
    original = fast.cake_rmsnorm_train_backward
    flags = []

    def record(*args, **kwargs):
        flags.append(kwargs["deterministic"])
        return original(*args, **kwargs)

    monkeypatch.setattr(fast, "cake_rmsnorm_train_backward", record)
    x, w, dy = make(1000, 2048)
    cake.configure(cake_rmsnorm_variant=forward)
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS)
    xi = x.detach().requires_grad_()
    y = norm(xi)
    cake.configure(cake_rmsnorm_variant=backward)
    y.backward(dy)
    assert flags == [forward == "det"]


_UNCHECKED_LAUNCH = """
import json

from miles_ext_kernels.cake._impl.rmsnorm import ops
from cake.rmsnorm import test_rmsnorm_gpu as module

ops._check_shared_memory = lambda *args: None  # launch as if the op did not check
try:
    for variant in ("det", "fast"):
        module.run(variant, *module.make(16385, 4096))
    module.torch.cuda.synchronize()
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
else:
    print(json.dumps({"error": None}))
"""


def test_the_refusal_is_exactly_what_fails_to_launch():
    """The op refuses hidden size 4096 here exactly where its programs fail to launch. A fresh process with the check
    disabled runs det and fast at 4096 over 16385 tokens (on SM107a, det's backward reduce and fast's copy of it launch
    in the oversized shared-memory mode): where the op refuses the width, the binding's CUDA error ends the call
    (R200 with a CUDA 13.0 runtime: "cudaFuncSetAttribute failed: invalid argument"); elsewhere both run."""
    refused = refusal(4096)
    paths = [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")]
    result = subprocess.run(
        [sys.executable, "-c", _UNCHECKED_LAUNCH],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    error = json.loads(result.stdout.splitlines()[-1])["error"]
    if refused is None:
        assert error is None
        return
    arch = cake.device_arch()
    ops = importlib.import_module("miles_ext_kernels.cake._impl.rmsnorm.ops")
    kernels = [f"kernel_{name.removesuffix('_' + arch)}" for name, _ in ops._oversized_programs(arch, 4096)]
    assert error is not None and re.search(r"cuda(FuncSetAttribute|LaunchKernelExC) failed", error), error
    assert any(kernel in error for kernel in kernels), error


def test_a_width_the_process_cannot_launch_is_refused_before_any_launch(monkeypatch):
    """With the process's CUDA runtime taken as 13.0 (the runtime wheel of a cu130 torch), SM107a refuses hidden size
    4096 at check_supported, prebuild, the module's construction and the first call, before any launch, and a call of
    another width then runs (the failed launch had left a CUDA error that broke the next one). The other widths, and
    every width on SM100a / SM103a, launch no oversized program and run."""
    ops = importlib.import_module("miles_ext_kernels.cake._impl.rmsnorm.ops")
    monkeypatch.setattr(ops, "_cuda_runtime_version", lambda major: 13000)
    ops._check_call.cache_clear()
    cake.release("rmsnorm")  # forgets the paths an earlier prebuild returned, so prebuild asks the op again
    arch = cake.device_arch()
    refused = r"hidden size 4096 on sm_107a launches .* the process's CUDA runtime \(libcudart\.so\.\d+\) is 13\.0"
    try:
        for H in HIDDEN_SIZES:
            if (arch, H) != ("sm_107a", 4096):
                rmsnorm.check_supported(H)
                continue
            x, w, _ = make(16385, H)
            with pytest.raises(RuntimeError, match=refused):
                rmsnorm.check_supported(H)
            with pytest.raises(RuntimeError, match=refused):
                cake.prebuild("rmsnorm", hidden_size=H)
            with pytest.raises(RuntimeError, match=refused):
                rmsnorm.ZeroCenteredRMSNorm(H, EPS)
            with pytest.raises(RuntimeError, match=refused):
                rmsnorm.zero_centered_rmsnorm(x, w, EPS)
        x, w, dy = make(1000, 2048)
        with no_build():
            assert_bitwise(run("det", x, w, dy), run("det", x, w, dy))
    finally:
        ops._check_call.cache_clear()
