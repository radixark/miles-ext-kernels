"""The weight gradient of the Cake input RMSNorm in the token-count windows FlashInfer fixed, on a GPU.

miles runs this op at H = 2048, 4096 and 5120 (``test_rmsnorm_gpu.py``) on the FP32 effective weight ``1 + w``,
``fast`` by default. FlashInfer changed the programs of ``fast``'s FP32-weight backward in these windows:

- H = 2048: on SM103a from T 58497 to 61952 (flashinfer-ai/flashinfer#8810), on SM100a from T 337 to 4096, from
  T 45057 to 58880 and above T 65536 and on SM107a from T 65 to 127 and from T 1025 to 16384 (#8827), on SM100a
  from T 65537 to 131072 again (#8848);
- H = 4096: on SM100a from T 1793 to 3584 and above T 65536, on SM107a from T 65 to 256 and above T 65536 (#8827),
  on SM103a above T 262144 (#8849), on SM100a from T 65537 to 131072 again (#8848);
- H = 5120: on SM100a from T 65 to 6144 and above T 56256, on SM107a from T 4097 to 19968 (#8827), on SM103a above
  T 196608 (#8849), on SM100a from T 56257 to 131072 again (#8848).

#8813 gave ``det``'s smallest token counts on SM103a and SM107a a new column-task backward. The H = 2048 token counts
below run on every architecture (where one falls in another window, the same checks hold), ``det`` up to T 65536 and
``fast``; at H = 4096 and 5120 (and H = 2048 above T 131072), ``fast`` runs one token count per window above, on the
architectures of that window: the top of the window or the first count above its lower edge. Every case runs an FP32
and a BF16 weight parameter, two inputs each, against FlashInfer's gates (``tests/norm/test_cake_rmsnorm_train.py`` and
``test_cake_rmsnorm_det_train.py``):

- ``rstd`` within relative L2 1e-5 of the FP64 one; ``y`` and ``dx`` within FlashInfer's tolerance of FP64;
- dw element by element within ``atol + rtol * |ref|`` (1e-4 each) of the FP64 sum ``ref = sum(dy * x * rstd)`` at
  the forward's FP32 ``rstd``, and within relative L2 1e-4 of it; a BF16 parameter's gradient is the FP32 one rounded
  to BF16, so its bounds are widened by that rounding;
- where ``fast``'s backward runs one kernel, the FP32 dw's relative L2 error at most that of the eager FP32 reduction
  ``(dy * x * rstd).sum(0)`` of the same inputs, up to one FP32 ulp;
- ``det`` bit for bit twice;
- over the FP32-parameter cases of the run (the last test), the median ratio of the dw error to the eager FP32
  reduction's at most 1: over every ``det`` case and for each backward form of ``fast``.

These are budget checks: they hold the installed build to FlashInfer's budget and do not tell the builds before these
fixes apart (FlashInfer measured the errors it fixed over many more inputs and token counts). ``test_fixed_windows.py``
checks the routes. Memory: a case's BF16 ``x``, ``dy``, ``y`` and ``dx`` take ``8 * T * H`` bytes (8 GiB at H = 4096,
T = 262145, the largest); the eager FP32 reduction of an FP32-parameter case adds up to ``12 * T * H`` bytes while it
runs; the FP64 references go over 8192-row chunks. Needs SM100a / SM103a / SM107a; skips elsewhere.
"""

import math
import statistics

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake import rmsnorm
from miles_ext_kernels.cake._runtime import loader

from ..helpers import assert_bitwise, assert_repeatable, requires_arch
from ._refusal import refusal, runs_or_is_refused
from ._windows import backward_form

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
EPS = 1e-6
# H = 2048, every architecture: inside the fixed windows (module docstring) on the architecture FlashInfer fixed them
# for: 337 to 4096 (SM100a), 1025 to 16384 (SM107a), 58497 to 61952 (SM103a), 65537 to 131072 (SM100a, fast only:
# 98304); 128 is a det column-task count everywhere.
TOKENS_2048 = (128, 337, 1025, 4096, 16384, 58497, 58560, 61952, 65536)
FAST_ONLY_TOKENS_2048 = (98304,)
# H = 4096 / 5120 (and H = 2048 above T 131072), fast: one token count per window FlashInfer changed, with the
# architectures of that window
WINDOW_CASES = (
    (2048, 131073, ("sm_100a",)),
    (4096, 256, ("sm_107a",)),
    (4096, 3584, ("sm_100a",)),
    (4096, 65537, ("sm_100a", "sm_107a")),
    (4096, 131073, ("sm_100a",)),
    (4096, 262145, ("sm_103a",)),
    (5120, 1792, ("sm_100a",)),
    (5120, 6144, ("sm_100a",)),
    (5120, 19968, ("sm_107a",)),
    (5120, 56257, ("sm_100a",)),
    (5120, 131073, ("sm_100a",)),
    (5120, 196609, ("sm_103a",)),
)
CASES = (
    [("det", 2048, T, ARCHES) for T in TOKENS_2048]
    + [("fast", 2048, T, ARCHES) for T in TOKENS_2048 + FAST_ONLY_TOKENS_2048]
    + [("fast", H, T, arches) for H, T, arches in WINDOW_CASES]
)
SEEDS = (0, 1)
# FlashInfer's budgets: y / dx within atol = rtol = 1e-2 of the reference scaled by its RMS; rstd within relative L2
# 1e-5; dw within relative L2 1e-4 and every element within atol + rtol * |ref| (1e-4 each) of the FP64 sum; a
# single-kernel dw not worse than the eager FP32 reduction by more than one FP32 ulp (relative).
ATOL = RTOL = 1e-2
RSTD_REL_L2 = 1e-5
DW_REL_L2 = 1e-4
DW_ATOL = DW_RTOL = 1e-4
DW_EAGER_FLOOR = 2.0**-23
CHUNK = 8192
# (variant, backward form, label, dw relative L2, eager FP32 dw relative L2) of every FP32-parameter case this
# process checked, for the aggregate test at the end of the module
_DW_ERRORS = []


@pytest.fixture(scope="module", autouse=True)
def _prebuilt():
    """Every library of the widths this device runs below, built once, before the first case (a width this process
    cannot launch builds nothing)."""
    arch = cake.device_arch()
    for hidden in sorted({H for _variant, H, _T, arches in CASES if arch in arches}):
        if refusal(hidden) is None:
            cake.prebuild("rmsnorm", hidden_size=hidden)


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def make(H, T, seed, params_dtype):
    """BF16 ``x`` and ``dy`` ``[T, 1, H]`` of N(0, 1) and ``w`` of N(0, 0.01) in ``params_dtype``."""
    g = torch.Generator(device=_device()).manual_seed(8066 * 1_000_003 + seed * 131_071 + T + (H - 2048) * 1_009)
    x = torch.randn(T, 1, H, device=_device(), generator=g).bfloat16()
    dy = torch.randn(T, 1, H, device=_device(), generator=g).bfloat16()
    w = (0.1 * torch.randn(H, device=_device(), generator=g)).to(params_dtype)
    return x, w, dy


def run(variant, x, w, dy):
    """``{"y", "dx", "dw"}`` of one forward + backward of a :class:`ZeroCenteredRMSNorm` holding ``w``."""
    cake.configure(cake_rmsnorm_variant=variant)
    norm = rmsnorm.ZeroCenteredRMSNorm(x.shape[-1], EPS, params_dtype=w.dtype)
    with torch.no_grad():
        norm.weight.copy_(w)
    xi = x.detach().requires_grad_()
    y = norm(xi)
    y.backward(dy)
    assert norm.weight.grad.dtype == w.dtype
    return {"y": y.detach(), "dx": xi.grad, "dw": norm.weight.grad}


def forward_rstd(variant, x, w):
    """``(y, rstd)`` of the packages' forward on the effective weight the op passes (the program the op's forward
    runs): the FP32 ``rstd`` its backward reads."""
    fast = loader.load_package("cake_rmsnorm_train_backend")
    with torch.no_grad():
        y, rstd = fast.cake_rmsnorm_train_forward(
            x.reshape(-1, x.shape[-1]), 1.0 + w.float(), EPS, deterministic=variant == "det"
        )
    return y, rstd


def chunks(rows):
    return [(start, min(start + CHUNK, rows)) for start in range(0, rows, CHUNK)]


def rel_l2(actual, ref):
    den = ref.norm().item()
    return (actual.double() - ref).norm().item() / den if den > 0 else (actual.double() - ref).norm().item()


def rstd_rel_l2(x, rstd):
    """Relative L2 error of the forward's FP32 ``rstd`` against the FP64 ``1 / sqrt(mean(x^2) + eps)``."""
    xs = x.reshape(-1, x.shape[-1])
    num = den = 0.0
    for start, stop in chunks(xs.shape[0]):
        ref = torch.rsqrt(xs[start:stop].double().square().mean(-1) + EPS)
        num += (rstd[start:stop].double() - ref).square().sum().item()
        den += ref.square().sum().item()
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)


def assert_rows_close(out, x, w, dy, rstd, label):
    """``y`` and ``dx`` within FlashInfer's tolerance of FP64 at the FP32 ``rstd``, both sides scaled by the
    reference's RMS (over row chunks: a first pass for the scales, a second for the comparison)."""
    H = x.shape[-1]
    xs, gs = x.reshape(-1, H), dy.reshape(-1, H)
    actual = {"y": out["y"].reshape(-1, H), "dx": out["dx"].reshape(-1, H)}
    weight = (1.0 + w.float()).double()

    def reference(start, stop):
        xd, gd = xs[start:stop].double(), gs[start:stop].double()
        r = rstd[start:stop].double().unsqueeze(-1)
        a = gd * weight
        return {"y": xd * r * weight, "dx": r * a - xd * r.pow(3) * (a * xd).mean(-1, keepdim=True)}

    squares = {"y": 0.0, "dx": 0.0}
    for start, stop in chunks(xs.shape[0]):
        for name, ref in reference(start, stop).items():
            squares[name] += ref.square().sum().item()
    scales = {name: math.sqrt(total / xs.numel()) or 1.0 for name, total in squares.items()}
    for start, stop in chunks(xs.shape[0]):
        for name, ref in reference(start, stop).items():
            a = actual[name][start:stop]
            assert a.dtype == torch.bfloat16, f"{label}: {name}"
            a = a.double()
            assert torch.isfinite(a).all(), f"{label}: {name} has non-finite values"
            torch.testing.assert_close(
                a / scales[name], ref / scales[name], atol=ATOL, rtol=RTOL, msg=f"{label}: {name} rows {start}:{stop}"
            )


def dw_reference(x, dy, rstd):
    """The FP64 sum ``(dy * x * rstd).sum(0)`` at the FP32 ``rstd``, over row chunks."""
    H = x.shape[-1]
    xs, gs = x.reshape(-1, H), dy.reshape(-1, H)
    dw = torch.zeros(H, dtype=torch.float64, device=x.device)
    for start, stop in chunks(xs.shape[0]):
        dw += (gs[start:stop].double() * xs[start:stop].double() * rstd[start:stop].double().unsqueeze(-1)).sum(0)
    return dw


def eager_dw(x, dy, rstd):
    """FlashInfer's eager FP32 reduction ``(dy * x * rstd).sum(0)`` of the same inputs."""
    H = x.shape[-1]
    return (dy.reshape(-1, H).float() * x.reshape(-1, H).float() * rstd.float().unsqueeze(-1)).sum(0)


def assert_dw(dw, ref, eager, label, *, eager_gate):
    """The element bound and relative L2 (both widened by the BF16 rounding for a BF16 gradient) and, with
    ``eager_gate``, the eager FP32 reduction's relative L2 error as a ceiling; returns the dw relative L2 error and,
    given ``eager``, the eager one."""
    assert torch.isfinite(dw).all(), f"{label}: non-finite dw"
    error = rel_l2(dw, ref)
    bound = DW_ATOL + DW_RTOL * ref.abs()
    if dw.dtype == torch.bfloat16:
        bound = 2**-8 * ref.abs() + (1 + 2**-8) * bound
        rel_bound = DW_REL_L2 + 2**-8
    else:
        assert dw.dtype == torch.float32, label
        rel_bound = DW_REL_L2
    ratio = (dw.double() - ref).abs() / bound
    worst = int(ratio.argmax())
    assert float(ratio.max()) <= 1.0, (
        f"{label}: {int((ratio > 1).sum())} dw elements outside atol + rtol * |ref| (1e-4 each); worst column "
        f"{worst}: {float(dw[worst])!r} vs FP64 {float(ref[worst])!r}, {float(ratio.max()):.3f} x the bound"
    )
    assert error <= rel_bound, f"{label}: dw relative L2 {error:.3e} > {rel_bound:.3e}"
    eager_error = None if eager is None else rel_l2(eager, ref)
    if eager_gate:
        assert error <= max(
            eager_error, DW_EAGER_FLOOR
        ), f"{label}: single-kernel dw relative L2 {error:.3e} > eager FP32 {eager_error:.3e}"
    return error, eager_error


def check(variant, H, T, seed, params_dtype, arch, form):
    label = f"{arch} {variant} H{H} T{T} {params_dtype} seed {seed} ({form} backward)"
    x, w, dy = make(H, T, seed, params_dtype)
    if variant == "det":
        out = assert_repeatable(lambda: run(variant, x, w, dy), times=2)
    else:
        out = run(variant, x, w, dy)
    y_fwd, rstd = forward_rstd(variant, x, w)
    assert_bitwise(out["y"].reshape(-1, H), y_fwd)
    del y_fwd
    error = rstd_rel_l2(x, rstd)
    assert error <= RSTD_REL_L2, f"{label}: rstd relative L2 {error:.3e} > {RSTD_REL_L2:.0e}"
    assert_rows_close(out, x, w, dy, rstd, label)
    fp32 = params_dtype == torch.float32
    # FlashInfer's gate for the one-kernel backward of deterministic=False, on the FP32 dw it returns
    eager_gate = fp32 and variant == "fast" and form in ("single", "column")
    eager = eager_dw(x, dy, rstd) if fp32 else None
    error, eager_error = assert_dw(out["dw"], dw_reference(x, dy, rstd), eager, label, eager_gate=eager_gate)
    if fp32:
        _DW_ERRORS.append((variant, form, label, error, eager_error))


@pytest.mark.parametrize("params_dtype", [torch.float32, torch.bfloat16], ids=["fp32-weight", "bf16-weight"])
@pytest.mark.parametrize(("variant", "H", "T", "arches"), CASES, ids=[f"{v}-H{H}-T{T}" for v, H, T, _ in CASES])
def test_dw_meets_the_budget_in_the_fixed_windows(variant, H, T, arches, params_dtype):
    arch = cake.device_arch()
    if arch not in arches:
        pytest.skip(f"H {H} T {T} stands for a window FlashInfer changed on {', '.join(arches)}")
    form = backward_form(variant, arch, H, T)
    with runs_or_is_refused(H):
        for seed in SEEDS:
            check(variant, H, T, seed, params_dtype, arch, form)


def test_dw_not_worse_than_eager_fp32_over_the_cases():
    """FlashInfer's aggregate dw gate over the FP32-parameter cases this process checked (a BF16 parameter's gradient
    is the FP32 one rounded to BF16): the median ratio of the dw relative L2 error to the eager FP32 reduction's is at
    most 1 over every ``det`` case (its deterministic suite's gate) and over each backward form of ``fast``, one
    kernel or a pair (its non-deterministic suite's gate). Runs after the cases above; skipped, as FlashInfer's is,
    when fewer than 8 of them ran in this process (``-k`` selection)."""
    if len(_DW_ERRORS) < 8:
        pytest.skip("needs the dw cases of this module in the same process")
    groups = {}
    for variant, form, label, error, eager in _DW_ERRORS:
        key = "det" if variant == "det" else f"fast {'pair' if form == 'pair' else 'one kernel'}"
        groups.setdefault(key, []).append((label, error, eager))
    for key, cases in sorted(groups.items()):
        ratios = [error / eager if eager > 0 else (0.0 if error == 0 else math.inf) for _, error, eager in cases]
        median = statistics.median(ratios)
        worst = max(cases, key=lambda case: case[1])
        assert median <= 1.0, (
            f"{key}: median dw error ratio to the eager FP32 reduction {median:.3f} > 1 over {len(cases)} cases "
            f"(worst dw relative L2 {worst[1]:.3e} at {worst[0]})"
        )
