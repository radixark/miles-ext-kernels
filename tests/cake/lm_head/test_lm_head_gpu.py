"""The cake lm_head log-probabilities (``det`` and ``fast``) against a PyTorch reference, at the hidden / vocabulary
sizes miles trains with ``--log-probs-backend cake`` (TP 1, so the whole vocabulary per GPU). Needs SM100a / SM103a /
SM107a; skips elsewhere.

Accuracy gate (FlashInfer's own, from its chunked LM-head tests): against a high-precision oracle, each error of a
variant is at most 1.05x the error of the unchunked PyTorch path at the same rounding boundaries (BF16 ``z = X @ W^T``,
FP32 ``log_softmax(z / tau)`` gathered at the labels, autograd ``dX`` / ``dW`` in BF16) plus a small floor. The oracle
evaluates the same BF16 input values in FP32 with TF32 off and no intermediate rounding: its own error (about 1e-6
relative) is three orders below the BF16 errors gated (about 1e-3), and FP64 GEMMs are too slow for this matrix. The
gate is checked to fail an output 2 % off.

Bitwise properties: ``det`` repeats bit for bit, in the same process and in a new one; ``fast`` gives ``det``'s
``logp`` and, without zero-gradient rows, ``det``'s ``dW``; a no-grad forward gives the training forward's ``logp``; a
frozen input leaves the other gradient's bits unchanged (``det``); a caller-given valid-row count gives the counted
call's bits; a call on a side stream or on another GPU gives the default stream's bits.

Every kernel library is built before the tests (``cake.prebuild("lm_head")``); set ``TVM_FFI_CACHE_DIR`` to keep them.
"""

import contextlib
import functools
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import lm_head  # noqa: E402
from miles_ext_kernels.cake._impl.lm_head import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in cake.ARCH_BY_CAPABILITY:
    pytest.skip("the cake lm_head kernels need SM100a / SM103a / SM107a", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
CUDA = torch.device("cuda")
IGNORE_INDEX = -100
SEED = 20261104
VARIANTS = ("det", "fast")
# (H, V) of the models the cake log-prob path serves (H a multiple of 512, V of 256)
SHAPES = [(4096, 129280), (5120, 248320), (4096, 154880), (6144, 154880), (7168, 129280)]
# one row, one chunk short of full, two chunks with a one-row tail, four chunks with a ragged tail
ROWS = [1, 4095, 4097, 16231]
TEMPERATURES = [None, 0.7]
# (H, V, T) of the CUDA 13.0 fault reports on the sm_100a dW / dX GEMMs, about a third of the labels ignored
ISSUE_SHAPES = [
    (6144, 154880, 16384),
    (4096, 154880, 16384),
    (6144, 154880, 16231),
    (4096, 154880, 16231),
    (6144, 154880, 4096),
    (5120, 248320, 4096),
    (5120, 248320, 16231),
    (5120, 248320, 16384),
]
GATE_MARGIN = 1.05
GATE_FLOOR = dict(loss_rel=1e-6, logp_max_abs=1e-5, dX_rel_l2=1e-3, dW_rel_l2=1e-3)
# an output this far off must fail the gate (README, Development)
DETECT_SCALE = 1.02
_ORACLE_ROWS = 1024  # rows per oracle step: one [rows, V] FP32 logits block
_NORM_ROWS = 8192  # rows per step of an error norm
# the device sleep queued ahead of a call that must not wait for it
SLEEP_SECONDS = 0.3


@pytest.fixture(scope="module", autouse=True)
def built():
    """Every library of both variants (with the temperature forms) built up front, in parallel."""
    workers = min(16, len(os.sched_getaffinity(0)))
    libraries = cake.prebuild("lm_head", max_workers=workers)
    assert libraries and all(path.is_file() for path in libraries)
    built = {path: path.stat().st_mtime_ns for path in libraries}
    # all cached now: a second call returns the same libraries and rebuilds none of them
    assert {path: path.stat().st_mtime_ns for path in cake.prebuild("lm_head")} == built


@pytest.fixture(autouse=True)
def _fresh_bindings():
    """Each test starts without remembered bindings (their workspaces) from earlier shapes."""
    yield
    cake.release("lm_head")
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
_WEIGHTS: dict[tuple[int, int], torch.Tensor] = {}


def _weight(H: int, V: int) -> torch.Tensor:
    """``W [V, H]`` BF16 ~ N(0, 1 / H), one per shape for the module."""
    if (H, V) not in _WEIGHTS:
        gen = torch.Generator(device=CUDA)
        gen.manual_seed(SEED + 7 * H + V)
        W = torch.empty((V, H), dtype=torch.bfloat16, device=CUDA)
        for r0 in range(0, V, 32768):
            r1 = min(V, r0 + 32768)
            W[r0:r1] = (torch.randn((r1 - r0, H), generator=gen, device=CUDA) * H**-0.5).to(torch.bfloat16)
        _WEIGHTS[(H, V)] = W
    return _WEIGHTS[(H, V)]


@dataclass
class Problem:
    X: torch.Tensor  # BF16 [T, H]
    W: torch.Tensor  # BF16 [V, H]
    labels: torch.Tensor  # int64 [T], -100 ignored
    dlogp: torch.Tensor  # FP32 [T] incoming gradient, 0 on ignored rows
    temperature: float | None

    @property
    def valid(self) -> torch.Tensor:
        return self.labels >= 0

    def tau_rows(self) -> torch.Tensor | None:
        if self.temperature is None:
            return None
        return torch.full((self.X.shape[0],), self.temperature, dtype=torch.float32, device=CUDA)


def _problem(H, V, T, *, temperature=None, ignore_frac=0.05, zero_dlogp_frac=0.0, seed=SEED) -> Problem:
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(seed + T)
    X = (torch.randn((T, H), generator=gen, device=CUDA) * 0.5).to(torch.bfloat16)
    labels = torch.randint(0, V, (T,), generator=gen, device=CUDA)
    ignored = round(ignore_frac * T)
    if ignored:
        labels[torch.randperm(T, generator=gen, device=CUDA)[:ignored]] = IGNORE_INDEX
    dlogp = torch.randn((T,), generator=gen, device=CUDA) * 0.05
    if zero_dlogp_frac:
        dlogp[torch.rand((T,), generator=gen, device=CUDA) < zero_dlogp_frac] = 0.0
    dlogp = torch.where(labels >= 0, dlogp, torch.zeros_like(dlogp))
    return Problem(X, _weight(H, V), labels, dlogp.contiguous(), temperature)


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _fp32_gemms():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@contextlib.contextmanager
def _unsplit_gemms():
    """Each BF16 GEMM of the PyTorch path as one FP32 accumulation over K (cuBLASLt without split-K), the same form
    on every GPU model."""
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_bf16_reduced_precision_reduction_split_k"):
        yield
        return
    library = torch.backends.cuda.preferred_blas_library()
    reduction = (matmul.allow_bf16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction_split_k)
    torch.backends.cuda.preferred_blas_library("cublaslt")
    matmul.allow_bf16_reduced_precision_reduction = (False, False)
    try:
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = reduction
        torch.backends.cuda.preferred_blas_library(library)


@torch.no_grad()
def _oracle(p: Problem) -> dict:
    """``logp`` / ``dX`` / ``dW`` and the downstream loss ``sum(logp * dlogp)`` from the BF16 input values in FP32,
    without intermediate rounding, ``_ORACLE_ROWS`` rows at a time."""
    T, H = p.X.shape
    W32 = p.W.float()
    tau = p.tau_rows()
    logp = torch.zeros((T,), dtype=torch.float32, device=CUDA)
    dX = torch.empty((T, H), dtype=torch.float32, device=CUDA)
    dW = torch.zeros_like(W32)
    with _fp32_gemms():
        for r0 in range(0, T, _ORACLE_ROWS):
            r1 = min(T, r0 + _ORACLE_ROWS)
            x = p.X[r0:r1].float()
            z = x @ W32.t()
            if tau is not None:
                z /= tau[r0:r1, None]
            lse = torch.logsumexp(z, dim=-1)
            valid = p.valid[r0:r1]
            index = torch.where(valid, p.labels[r0:r1], torch.zeros_like(p.labels[r0:r1]))
            logp[r0:r1] = torch.where(valid, z.gather(1, index[:, None]).squeeze(1) - lse, torch.zeros_like(lse))
            d = p.dlogp[r0:r1] if tau is None else p.dlogp[r0:r1] / tau[r0:r1]
            d = torch.where(valid, d, torch.zeros_like(d))
            dz = z.sub_(lse[:, None]).exp_().neg_()  # -softmax, in place of the logits
            dz.scatter_add_(1, index[:, None], torch.ones((r1 - r0, 1), dtype=dz.dtype, device=CUDA))
            dz.mul_(d[:, None])  # ignored rows: d = 0
            dX[r0:r1] = dz @ W32
            dW.addmm_(dz.t(), x)
            del z, dz
    return dict(logp=logp, dX=dX, dW=dW, loss=_loss(logp, p))


def _baseline(p: Problem) -> dict:
    """The unchunked PyTorch path at the kernels' rounding boundaries."""
    X = p.X.detach().requires_grad_(True)
    W = p.W.detach().requires_grad_(True)
    tau = p.tau_rows()
    with torch.enable_grad(), _unsplit_gemms():
        z = (X @ W.t()).float()
        if tau is not None:
            z = z / tau[:, None]
        lse = torch.logsumexp(z, dim=-1)
        index = torch.where(p.valid, p.labels, torch.zeros_like(p.labels))
        logp = torch.where(p.valid, z.gather(1, index[:, None]).squeeze(1) - lse, torch.zeros_like(lse))
        del z
        logp.backward(p.dlogp)
    return dict(logp=logp.detach(), dX=X.grad, dW=W.grad, loss=_loss(logp.detach(), p))


def _loss(logp: torch.Tensor, p: Problem) -> torch.Tensor:
    return (logp.double() * p.dlogp.double())[p.valid].sum()


def _configure(variant: str, chunk_size: int | None = None) -> None:
    values = {"cake_lm_head_variant": variant}
    if chunk_size is not None:
        values["cake_lm_head_chunk_size"] = chunk_size
    cake.configure(**values)


def _run(
    p: Problem, variant, *, train_x=True, train_w=True, grad=True, chunk_size=None, labels=None, **kwargs
) -> dict:
    """One forward (and, with ``grad`` and a trainable input, backward with ``p.dlogp``) through
    ``lm_head.log_probs`` with ``variant`` configured (``None``: as configured)."""
    if variant is not None:
        _configure(variant, chunk_size)
    kwargs.setdefault("temperature", p.temperature)
    X = p.X.detach().requires_grad_(train_x)
    W = p.W.detach().requires_grad_(train_w)
    with torch.set_grad_enabled(grad):
        logp = lm_head.log_probs(X, W, p.labels if labels is None else labels, **kwargs)
    assert logp.dtype == torch.float32 and tuple(logp.shape) == (p.X.shape[0],)
    if grad and (train_x or train_w):
        logp.backward(p.dlogp)
    torch.cuda.synchronize()
    return dict(
        logp=logp.detach(), dX=X.grad, dW=W.grad, loss=_loss(logp.detach(), p), requires_grad=logp.requires_grad
    )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _rel_l2(a: torch.Tensor, ref: torch.Tensor) -> float:
    """``||a - ref|| / ||ref||`` in FP64 sums, ``_NORM_ROWS`` rows at a time."""
    num = den = 0.0
    for r0 in range(0, a.shape[0], _NORM_ROWS):
        x, y = a[r0 : r0 + _NORM_ROWS].float(), ref[r0 : r0 + _NORM_ROWS].float()
        num += float((x - y).double().square().sum())
        den += float(y.double().square().sum())
    return num**0.5 / max(den**0.5, 1e-30)


def _finite(t) -> bool:
    return t is None or bool(torch.isfinite(t).all())


def _errors(result: dict, oracle: dict, p: Problem) -> dict:
    valid = p.valid
    errors = dict(loss_rel=abs(float(result["loss"]) - float(oracle["loss"])) / max(abs(float(oracle["loss"])), 1e-30))
    diff = (result["logp"].double() - oracle["logp"].double()).abs()
    errors["logp_max_abs"] = float(diff[valid].max()) if bool(valid.any()) else 0.0
    errors["logp_ignored_zero"] = bool((result["logp"][~valid] == 0).all())
    for name in ("dX", "dW"):
        if result[name] is not None:
            errors[f"{name}_rel_l2"] = _rel_l2(result[name], oracle[name])
    if result["dX"] is not None:
        errors["dX_ignored_zero"] = bool((result["dX"][~valid] == 0).all())
    errors["finite"] = all(_finite(result[name]) for name in ("logp", "dX", "dW"))
    return errors


def _gate_failures(errors: dict, baseline: dict) -> list[str]:
    return [
        f"{key}: {errors[key]:.3e} vs the unchunked PyTorch path's {baseline[key]:.3e}"
        for key, floor in GATE_FLOOR.items()
        if key in errors and errors[key] > GATE_MARGIN * baseline[key] + floor
    ]


def _assert_gate(result: dict, oracle: dict, baseline: dict, p: Problem, what: str) -> dict:
    errors = _errors(result, oracle, p)
    assert errors["finite"], f"{what}: a non-finite output"
    failures = _gate_failures(errors, baseline)
    assert not failures, f"{what}: " + "; ".join(failures)
    assert errors["logp_ignored_zero"], f"{what}: logp of an ignored row is not 0"
    if "dX_ignored_zero" in errors:
        assert errors["dX_ignored_zero"], f"{what}: dX of an ignored row is not 0"
    return errors


def _assert_gate_detects(result: dict, oracle: dict, baseline: dict, p: Problem, what: str) -> None:
    """Each gated tensor scaled by DETECT_SCALE fails the gate: the gate catches a 2 % error."""
    for name in ("logp", "dX", "dW"):
        if result[name] is None:
            continue
        scaled = dict(result, **{name: result[name].float() * DETECT_SCALE})
        if name == "logp":
            scaled["loss"] = _loss(scaled["logp"], p)
        key = "logp_max_abs" if name == "logp" else f"{name}_rel_l2"
        errors = _errors(scaled, oracle, p)
        assert errors[key] > GATE_MARGIN * baseline[key] + GATE_FLOOR[key], f"{what}: the gate passes {name} x1.02"


def _references(p: Problem) -> tuple[dict, dict]:
    """The oracle and the PyTorch path's errors against it (the PyTorch outputs are dropped)."""
    oracle = _oracle(p)
    baseline = _errors(_baseline(p), oracle, p)
    torch.cuda.empty_cache()
    return oracle, baseline


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _same_bits(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _assert_same_bits(a: dict, b: dict, keys, what: str) -> None:
    for key in keys:
        assert _same_bits(a[key], b[key]), f"{what}: {key} bits differ"


def _digests(result: dict) -> dict:
    """sha256 of the bytes of ``logp``, ``dX`` and ``dW``."""
    return {
        name: hashlib.sha256(_bits(result[name]).contiguous().cpu().numpy().tobytes()).hexdigest()
        for name in ("logp", "dX", "dW")
    }


# --------------------------------------------------------------------------- #
# accuracy and bitwise properties
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("temperature", TEMPERATURES, ids=lambda t: f"tau{t}")
@pytest.mark.parametrize("T", ROWS, ids=lambda t: f"T{t}")
@pytest.mark.parametrize(("H", "V"), SHAPES)
def test_log_probs_match_the_reference(H, V, T, temperature):
    """Both variants pass the gate; ``det`` repeats bit for bit; ``fast`` has ``det``'s ``logp`` and ``dW`` bits; the
    no-grad forward has the training forward's ``logp`` bits."""
    p = _problem(H, V, T, temperature=temperature)
    oracle, baseline = _references(p)

    det = _run(p, "det")
    _assert_gate(det, oracle, baseline, p, "det")
    _assert_gate_detects(det, oracle, baseline, p, "det")
    _assert_same_bits(_run(p, "det"), det, ("logp", "dX", "dW"), "det repeat")
    _assert_same_bits(_run(p, "det", train_x=False, train_w=False, grad=False), det, ("logp",), "det no-grad")

    fast = _run(p, "fast")
    _assert_gate(fast, oracle, baseline, p, "fast")
    _assert_same_bits(fast, det, ("logp", "dW"), "fast vs det")
    _assert_same_bits(_run(p, "fast", train_x=False, train_w=False, grad=False), det, ("logp",), "fast no-grad")


@pytest.mark.parametrize(("H", "V", "T"), ISSUE_SHAPES)
def test_log_probs_at_the_cuda_13_0_fault_shapes(H, V, T):
    """The shapes of the CUDA 13.0 dW / dX GEMM fault reports, a third of the rows ignored (the compacted path):
    three ``det`` runs with the same bits within the gate, and ``fast`` within the gate."""
    p = _problem(H, V, T, ignore_frac=0.33)
    oracle, baseline = _references(p)

    det = _run(p, "det")
    _assert_gate(det, oracle, baseline, p, "det")
    for repeat in range(2):
        _assert_same_bits(_run(p, "det"), det, ("logp", "dX", "dW"), f"det repeat {repeat}")
    _assert_gate(_run(p, "fast"), oracle, baseline, p, "fast")


_NEW_PROCESS = """
import hashlib, json, sys
import torch
from miles_ext_kernels import cake
from miles_ext_kernels.cake import lm_head
from miles_ext_kernels.cake._runtime import loader

H, V, T, seed, temperature = json.loads(sys.argv[1])
cake.configure(cake_lm_head_variant="det")
gen = torch.Generator(device="cuda")
gen.manual_seed(seed + 7 * H + V)
W = torch.empty((V, H), dtype=torch.bfloat16, device="cuda")
for r0 in range(0, V, 32768):
    r1 = min(V, r0 + 32768)
    W[r0:r1] = (torch.randn((r1 - r0, H), generator=gen, device="cuda") * H**-0.5).to(torch.bfloat16)
gen.manual_seed(seed + T)
X = (torch.randn((T, H), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
labels = torch.randint(0, V, (T,), generator=gen, device="cuda")
ignored = round(0.05 * T)
labels[torch.randperm(T, generator=gen, device="cuda")[:ignored]] = -100
dlogp = torch.randn((T,), generator=gen, device="cuda") * 0.05
dlogp = torch.where(labels >= 0, dlogp, torch.zeros_like(dlogp)).contiguous()
X.requires_grad_(True)
W.requires_grad_(True)
with loader.no_build():
    logp = lm_head.log_probs(X, W, labels, temperature=temperature)
    logp.backward(dlogp)
torch.cuda.synchronize()
bits = {torch.bfloat16: torch.int16, torch.float32: torch.int32}
print(json.dumps({
    name: hashlib.sha256(t.view(bits[t.dtype]).contiguous().cpu().numpy().tobytes()).hexdigest()
    for name, t in (("logp", logp.detach()), ("dX", X.grad), ("dW", W.grad))
}))
"""


def _subprocess(script: str, *args: str) -> subprocess.CompletedProcess:
    """``script`` in a fresh interpreter on this checkout (the same cache of built libraries)."""
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.mark.parametrize("temperature", TEMPERATURES, ids=lambda t: f"tau{t}")
@pytest.mark.parametrize(("H", "V", "T"), [(4096, 129280, 4097), (5120, 248320, 16231)])
def test_det_repeats_bit_for_bit_in_a_new_process(H, V, T, temperature):
    """``det`` in this process twice and in a fresh one (same inputs from the same seeds): the same ``logp``, ``dX``
    and ``dW`` bits; the fresh process compiles nothing (``no_build``)."""
    p = _problem(H, V, T, temperature=temperature)
    here = _digests(_run(p, "det"))
    assert _digests(_run(p, "det")) == here

    result = _subprocess(_NEW_PROCESS, json.dumps([H, V, T, SEED, temperature]))
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1]) == here


@pytest.mark.parametrize("temperature", TEMPERATURES, ids=lambda t: f"tau{t}")
@pytest.mark.parametrize("variant", VARIANTS)
def test_an_empty_call_returns_no_rows_and_a_zero_weight_gradient(variant, temperature):
    """A rank that scores no row stays in the autograd graph: empty ``logp`` and ``dX``, a zero ``dW``."""
    H, V = SHAPES[0]
    p = _problem(H, V, 0, temperature=temperature)

    for count in (None, 0):
        out = _run(p, variant, num_valid_rows=count)
        assert out["requires_grad"]
        assert out["dX"] is None or (tuple(out["dX"].shape) == (0, H) and out["dX"].dtype == torch.bfloat16)
        # a weight gradient every rank forms, so the data-parallel gradient reduction sees this rank's (zero) share
        assert tuple(out["dW"].shape) == (V, H) and out["dW"].dtype == torch.bfloat16
        assert not bool((out["dW"] != 0).any())


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_frozen_input_gets_no_gradient_and_leaves_the_other_one(variant):
    """``X`` or ``W`` frozen: its gradient is not formed, the other one passes the gate (and, for ``det``, keeps the
    bits of the call with both trainable); with both frozen the output needs no gradient."""
    H, V = SHAPES[0]
    p = _problem(H, V, 4097, temperature=0.7)
    oracle, baseline = _references(p)
    both = _run(p, variant)
    x_only = _run(p, variant, train_w=False)
    w_only = _run(p, variant, train_x=False)
    neither = _run(p, variant, train_x=False, train_w=False)

    assert x_only["dW"] is None and w_only["dX"] is None and neither["dX"] is None and neither["dW"] is None
    assert not neither["requires_grad"]
    for name, result in (("X only", x_only), ("W only", w_only), ("neither", neither)):
        _assert_gate(result, oracle, baseline, p, f"{variant} {name}")
        _assert_same_bits(result, both, ("logp",), f"{variant} {name}")
    _assert_same_bits(w_only, both, ("dW",), f"{variant} W only")
    if variant == "det":
        _assert_same_bits(x_only, both, ("dX",), "det X only")


@pytest.mark.parametrize("ignore_frac", [0.0, 0.05], ids=["all-valid", "some-ignored"])
@pytest.mark.parametrize("T", [4097, 16231], ids=lambda t: f"T{t}")
def test_a_given_valid_row_count_gives_the_counted_bits(T, ignore_frac):
    """miles knows the count (every scored row has a target): the call with ``num_valid_rows`` gives the bits of
    the call that counts on the device (``fast``: ``logp`` and ``dW``)."""
    H, V = SHAPES[2]
    p = _problem(H, V, T, temperature=0.7, ignore_frac=ignore_frac)
    count = int(p.valid.sum())

    for variant, keys in (("det", ("logp", "dX", "dW")), ("fast", ("logp", "dW"))):
        counted = _run(p, variant)
        _assert_same_bits(_run(p, variant, num_valid_rows=count), counted, keys, f"{variant} num_valid_rows")


@pytest.mark.parametrize("variant", VARIANTS)
def test_zero_gradient_rows_get_a_zero_hidden_gradient(variant):
    """PPO-clipped and masked tokens reach the kernels as rows with a zero incoming gradient (``fast`` skips their
    GEMM work from the architecture's floor of valid rows on): within the gate, with exact zeros in their ``dX``
    rows."""
    H, V = SHAPES[0]
    p = _problem(H, V, 16231, temperature=0.7, zero_dlogp_frac=0.3)
    oracle, baseline = _references(p)

    out = _run(p, variant)
    _assert_gate(out, oracle, baseline, p, variant)
    zero = p.valid & (p.dlogp == 0)
    assert int(zero.sum()) > 0
    assert not bool((out["dX"][zero] != 0).any())


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_unit_temperature_is_no_temperature(variant):
    H, V = SHAPES[0]
    p = _problem(H, V, 4097)
    plain = _run(p, variant)

    keys = ("logp", "dX", "dW") if variant == "det" else ("logp", "dW")
    for unit in (1.0, 1):
        _assert_same_bits(_run(p, variant, temperature=unit), plain, keys, f"{variant} temperature {unit}")


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_configured_variant_and_chunk_size_are_read_at_call_time(variant):
    """A call reads the configuration when it runs: configuring after a first call takes effect at the next one."""
    H, V = SHAPES[0]
    p = _problem(H, V, 4097)
    other = "fast" if variant == "det" else "det"
    keys = ("logp", "dX", "dW") if variant == "det" else ("logp", "dW")
    reference = _run(p, variant, chunk_size=1024)

    _configure(other)
    _run(p, None)
    _configure(variant, 1024)
    _assert_same_bits(_run(p, None), reference, keys, f"{variant} configured after a call of {other}")


# --------------------------------------------------------------------------- #
# streams and devices
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", VARIANTS)
def test_a_call_on_a_side_stream_gives_the_default_stream_bits(variant):
    """Calls alternate between the default stream and a side stream (the inputs made on the default stream): each one
    gives the default stream's bits (``fast``: ``logp`` and ``dW``)."""
    H, V = SHAPES[0]
    p = _problem(H, V, 16231, temperature=0.7)
    keys = ("logp", "dX", "dW") if variant == "det" else ("logp", "dW")
    reference = _run(p, variant)
    side = torch.cuda.Stream()

    for repeat in range(2):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            on_side = _run(p, variant)
        torch.cuda.current_stream().wait_stream(side)
        _assert_same_bits(on_side, reference, keys, f"{variant} side stream {repeat}")
        _assert_same_bits(_run(p, variant), reference, keys, f"{variant} default stream {repeat}")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("variant", VARIANTS)
def test_a_call_on_another_gpu_gives_the_same_bits(variant):
    """The inputs on ``cuda:1`` while ``cuda:0`` is current: the call runs there and gives ``cuda:0``'s bits (GPUs of
    one model; ``fast``: ``logp`` and ``dW``)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
        pytest.skip("needs two GPUs of one model")
    cake.prebuild("lm_head", device=1)
    H, V = SHAPES[0]
    p = _problem(H, V, 4097, temperature=0.7)
    keys = ("logp", "dX", "dW") if variant == "det" else ("logp", "dW")
    reference = _run(p, variant)
    other = torch.device("cuda", 1)
    q = Problem(p.X.to(other), p.W.to(other), p.labels.to(other), p.dlogp.to(other), p.temperature)

    with torch.cuda.device(0):
        _configure(variant)
        X = q.X.detach().requires_grad_(True)
        W = q.W.detach().requires_grad_(True)
        logp = lm_head.log_probs(X, W, q.labels, temperature=q.temperature)
        logp.backward(q.dlogp)
        torch.cuda.synchronize(other)
    assert logp.device == other and X.grad.device == other and W.grad.device == other
    there = dict(logp=logp.detach().to(CUDA), dX=X.grad.to(CUDA), dW=W.grad.to(CUDA))
    _assert_same_bits(there, reference, keys, f"{variant} on cuda:1")


# --------------------------------------------------------------------------- #
# review items: host synchronization, labels, the weight gradient, geometry
# --------------------------------------------------------------------------- #
@functools.cache
def _sleep_cycles_per_second() -> float:
    cycles = 50_000_000
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    end.synchronize()
    return cycles / (start.elapsed_time(end) / 1e3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_call_given_the_valid_row_count_does_not_wait_for_the_device(variant, caplog):
    """Behind a queued device sleep: the forward given ``num_valid_rows`` returns before the sleep ends, and the one
    that counts its rows waits for it; on SM103a with CUDA 13.0 programs both wait (the packages' host-count path),
    which the first call logs once."""
    H, V = SHAPES[0]
    p = _problem(H, V, 4097, temperature=0.7)
    count = int(p.valid.sum())
    arch = cake.device_arch()
    waits = ops.host_count_path(arch)
    _run(p, variant, num_valid_rows=count)  # bindings and allocator blocks in place
    _run(p, variant)
    X = p.X.detach().requires_grad_(True)
    W = p.W.detach().requires_grad_(True)
    ops._check_toolkit.cache_clear()

    elapsed = {}
    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        for given in (count, None, count):
            torch.cuda.synchronize()
            torch.cuda._sleep(int(_sleep_cycles_per_second() * SLEEP_SECONDS))
            start = time.perf_counter()
            logp = lm_head.log_probs(X, W, p.labels, temperature=p.temperature, num_valid_rows=given)
            elapsed.setdefault(given, []).append(time.perf_counter() - start)
            logp.backward(p.dlogp)
            torch.cuda.synchronize()
    logged = [record.getMessage() for record in caplog.records if "host-count path" in record.getMessage()]

    assert min(elapsed[None]) > SLEEP_SECONDS / 2, elapsed  # the documented readback of the count
    if waits:
        assert min(elapsed[count]) > SLEEP_SECONDS / 2, elapsed
        assert len(logged) == 1 and arch in logged[0], logged
    else:
        assert max(elapsed[count]) < SLEEP_SECONDS / 2, elapsed
        assert logged == []


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_negative_label_is_an_ignored_row(variant):
    """``-1``, ``-7`` and ``-2**40`` mark an ignored row exactly as ``-100`` does: ``logp`` 0, ``dX`` 0, the same bits
    (``fast``: ``logp`` and ``dW``)."""
    H, V = SHAPES[0]
    p = _problem(H, V, 4097, temperature=0.7, ignore_frac=0.1)
    keys = ("logp", "dX", "dW") if variant == "det" else ("logp", "dW")
    reference = _run(p, variant)
    ignored = ~p.valid

    for value in (-1, -7, -(2**40)):
        labels = torch.where(ignored, torch.full_like(p.labels, value), p.labels)
        out = _run(p, variant, labels=labels)
        _assert_same_bits(out, reference, keys, f"{variant} label {value}")
        assert not bool((out["logp"][ignored] != 0).any()) and not bool((out["dX"][ignored] != 0).any())


_OUT_OF_RANGE = """
import json, sys
import torch
from miles_ext_kernels import cake
from miles_ext_kernels.cake import lm_head

variant, label = sys.argv[1], int(sys.argv[2])
H, V, T = 4096, 129280, 64
cake.configure(cake_lm_head_variant=variant)
torch.manual_seed(0)
X = (torch.randn((T, H), device="cuda") * 0.5).to(torch.bfloat16)
W = (torch.randn((V, H), device="cuda") * H**-0.5).to(torch.bfloat16)
labels = torch.randint(0, V, (T,), device="cuda")
labels[5] = label
logp = lm_head.log_probs(X, W, labels)
torch.cuda.synchronize()
print("NO_ASSERTION", json.dumps(float(logp[5])))
"""


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_label_at_or_above_v_stops_the_stream(variant):
    """A label ``V``, ``V + 7`` or ``2**20`` stops the stream with a device-side assertion (a fresh process: the
    assertion ends its CUDA context) instead of returning a plausible ``logp``; ``V - 1`` runs."""
    V = 129280
    control = _subprocess(_OUT_OF_RANGE, variant, str(V - 1))
    assert control.returncode == 0 and "NO_ASSERTION" in control.stdout, control.stderr[-4000:]

    for label in (V, V + 7, 2**20):
        result = _subprocess(_OUT_OF_RANGE, variant, str(label))
        assert result.returncode != 0 and "NO_ASSERTION" not in result.stdout, (label, result.stdout)
        assert "device-side assert" in result.stderr, (label, result.stderr[-4000:])


def test_the_weight_gradient_is_one_bf16_cast_of_the_fp32_accumulation():
    """``dW`` reaches ``W`` as BF16: the round-to-nearest cast of the FP32 ``[V, H]`` accumulator the backward forms
    (the package's accumulator, uncast), bit for bit (``det``; ``fast`` has ``det``'s ``dW`` bits)."""
    H, V = SHAPES[0]
    p = _problem(H, V, 16231, temperature=0.7)
    out = _run(p, "det")
    assert out["dW"].dtype == torch.bfloat16

    entry = ops._entry("det")
    chunk = cake.option("lm_head", "cake_lm_head_chunk_size")
    forward = entry.forward_logprob_temperature(p.X, p.W, p.labels, chunk_size=chunk, temperature=p.temperature)
    _, accumulator = entry.backward_logprob_temperature(
        p.X,
        p.W,
        p.labels,
        forward.lse,
        p.dlogp,
        chunk_size=chunk,
        need_dx=False,
        need_dw=True,
        fuse_dw_cast=False,
        row_index=forward.row_index,
        dw_rows=V,
        temperature_rows=forward.temperature_rows,
    )
    torch.cuda.synchronize()
    assert accumulator.dtype == torch.float32 and tuple(accumulator.shape) == (V, H)
    assert _same_bits(accumulator.to(torch.bfloat16), out["dW"])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    ("H", "V", "message"),
    [(4352, 1024, "512"), (4096, 1000, "multiple of 256"), (4096, 151936, "multiple of 256")],
)
def test_a_geometry_the_kernels_do_not_serve_is_a_clear_error(variant, H, V, message):
    """``H`` off the 512-column dX / dW tiles or ``V`` off the 256-column logits tile (151,936 included) is a
    ``ValueError`` before any launch."""
    X = torch.zeros((8, H), dtype=torch.bfloat16, device=CUDA)
    W = torch.zeros((V, H), dtype=torch.bfloat16, device=CUDA)
    labels = torch.zeros((8,), dtype=torch.int64, device=CUDA)
    _configure(variant)
    with pytest.raises(ValueError, match=message):
        lm_head.log_probs(X, W, labels)


def test_an_unsupported_device_or_toolkit_is_a_clear_error(monkeypatch):
    H, V = SHAPES[0]
    p = _problem(H, V, 8)
    with pytest.raises(ValueError, match="run on a CUDA device"):
        lm_head.log_probs(p.X.cpu(), p.W[:256].cpu(), p.labels.cpu())

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(NotImplementedError, match=r"need SM100a, SM103a or SM107a, not compute capability 9\.0"):
        lm_head.log_probs(p.X, p.W, p.labels)
    with pytest.raises(NotImplementedError, match="compute capability 9.0"):
        cake.prebuild("lm_head")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(loader, "MIN_CUDA", {**loader.MIN_CUDA, "sm_107a": (99, 0)})
    ops._check_toolkit.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=r"lm_head kernels for sm_107a needs CUDA 99\.0 or newer"):
            lm_head.log_probs(p.X, p.W, p.labels)
        with pytest.raises(RuntimeError, match=r"lm_head kernels for sm_107a needs CUDA 99\.0 or newer"):
            cake.prebuild("lm_head")
    finally:
        ops._check_toolkit.cache_clear()
