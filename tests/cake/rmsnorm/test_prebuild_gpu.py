"""The prebuild contract of rmsnorm (README, Prebuild) on a GPU.

After one ``prebuild("rmsnorm", hidden_size=H)`` no call of width ``H`` compiles: every route of :data:`ROUTES`, the
call-time paths of the op, runs forward and backward with every variant under ``no_build()``. Also: a variant
configured after a module was built and called runs at its next call without a build, and no flashinfer module is
ever imported. A width whose oversized shared-memory programs this process cannot launch (SM107a at 4096 without a
CUDA 13.4 runtime) is refused by ``prebuild`` before it builds anything. Needs SM100a / SM103a / SM107a; skips
elsewhere.
"""

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake import rmsnorm
from miles_ext_kernels.cake._runtime import loader

from ..helpers import check_prebuild, no_build, requires_arch
from ._refusal import runs_or_is_refused

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")

EPS = 1e-6
REPO_ROOT = Path(__file__).resolve().parents[3]
# every width the kernels serve, on every architecture (test_module.py checks the list)
HIDDEN_SIZES = (256, 512, 1536, 2048, 4096, 5120, 6144)


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _inputs(T, H, *, lead=(1,)):
    g = torch.Generator(device=_device()).manual_seed(T * 131 + H)
    x = torch.randn(T, *lead, H, device=_device(), generator=g).bfloat16()
    dy = torch.randn(T, *lead, H, device=_device(), generator=g).bfloat16()
    w = (0.1 * torch.randn(H, device=_device(), generator=g)).bfloat16()
    return x, w, dy


def _fwd_bwd(x, w, dy, *, params_dtype=torch.bfloat16, use_checkpoint=False):
    from torch.utils.checkpoint import checkpoint

    norm = rmsnorm.ZeroCenteredRMSNorm(x.shape[-1], EPS, params_dtype=params_dtype)
    with torch.no_grad():
        norm.weight.copy_(w)
    xi = x.detach().requires_grad_()
    y = checkpoint(norm, xi, use_reentrant=False) if use_checkpoint else norm(xi)
    y.backward(dy)
    torch.cuda.synchronize()
    assert torch.isfinite(y).all() and torch.isfinite(xi.grad).all() and torch.isfinite(norm.weight.grad).all()


def window_token_counts(hidden):
    """1 and the first token count of every token-count window of both packages' FP32-weight forward and backward of
    ``hidden`` on this device: one call per program a call of this width may launch."""
    arch = cake.device_arch()
    counts = {1}
    for name in cake.op_spec("rmsnorm").packages:
        jit = loader.load_package(name).cake_jit
        for kind in ("fwd", "bwd"):
            counts |= {above + 1 for above, _, _ in jit.program_windows(arch, hidden, kind, False, "f32")}
    return sorted(counts)


def token_count_windows(variant, hidden):
    for T in window_token_counts(hidden):
        _fwd_bwd(*_inputs(T, hidden))


def two_dimensional_input(variant, hidden):
    _fwd_bwd(*_inputs(1000, hidden, lead=()))


def strided_rows(variant, hidden):
    x, w, dy = _inputs(1000, hidden)
    wide = torch.zeros(1000, 1, hidden + 64, dtype=torch.bfloat16, device=_device())
    wide[..., :hidden] = x
    _fwd_bwd(wide[..., :hidden], w, dy)


def fp32_weight(variant, hidden):
    x, w, dy = _inputs(1000, hidden)
    _fwd_bwd(x, w.float(), dy, params_dtype=torch.float32)


def no_tokens(variant, hidden):
    _fwd_bwd(*_inputs(0, hidden))


def non_reentrant_checkpoint(variant, hidden):
    _fwd_bwd(*_inputs(1000, hidden), use_checkpoint=True)


def torch_deterministic_algorithms(variant, hidden):
    """fast runs det here (logged once); det is det."""
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        _fwd_bwd(*_inputs(1000, hidden))
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


# name -> run(variant, hidden): every call-time path of the op (the hidden size is the one model parameter)
ROUTES = {
    "token-count windows": token_count_windows,
    "[T, H] input": two_dimensional_input,
    "strided rows": strided_rows,
    "FP32 weight": fp32_weight,
    "no tokens": no_tokens,
    "non-reentrant checkpoint": non_reentrant_checkpoint,
    "torch deterministic algorithms": torch_deterministic_algorithms,
}


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_after_one_prebuild_no_route_builds(hidden):
    routes = {name: functools.partial(run, hidden=hidden) for name, run in ROUTES.items()}

    with runs_or_is_refused(hidden):
        check_prebuild("rmsnorm", routes, hidden_size=hidden)


def test_prebuild_returns_every_library_and_is_idempotent():
    first = cake.prebuild("rmsnorm", hidden_size=2048)
    with no_build():
        again = cake.prebuild("rmsnorm", hidden_size=2048)

    assert first and first == again and all(path.is_file() for path in first)
    assert len({path.parent.name for path in first}) == 2  # the libraries of both packages


def test_a_variant_configured_after_the_module_was_built_runs_at_its_next_call_without_a_build(monkeypatch):
    """Reviewer item: configure(det), construct, call, configure(fast): the next call compiled fast's programs."""
    det = loader.load_package("cake_rmsnorm_det_train")
    fast = loader.load_package("cake_rmsnorm_nondet_train")
    ran = []

    def recorder(variant, original):
        def record(*args, **kwargs):
            ran.append(variant)
            return original(*args, **kwargs)

        return record

    monkeypatch.setattr(det, "cake_rmsnorm_det", recorder("det", det.cake_rmsnorm_det))
    monkeypatch.setattr(fast, "cake_rmsnorm_nondet", recorder("fast", fast.cake_rmsnorm_nondet))
    cake.prebuild("rmsnorm", hidden_size=2048)
    x, w, dy = _inputs(4096, 2048)
    with no_build():
        cake.configure(cake_rmsnorm_variant="det")
        norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS)
        norm(x.detach().requires_grad_()).backward(dy)
        cake.configure(cake_rmsnorm_variant="fast")
        norm(x.detach().requires_grad_()).backward(dy)
    assert ran == ["det", "fast"]


_FRESH_PROCESS = """
import json
import sys

from miles_ext_kernels import cake
from cake.rmsnorm import test_prebuild_gpu as module

cake.prebuild("rmsnorm", hidden_size=2048)
for variant in ("det", "fast"):
    cake.configure(cake_rmsnorm_variant=variant)
    module._fwd_bwd(*module._inputs(1000, 2048))
print(json.dumps(sorted(name for name in sys.modules if name.split(".")[0] == "flashinfer")))
"""


def test_no_flashinfer_module_is_imported():
    """In a fresh process: import the implementation, run both variants, and find no flashinfer module loaded."""
    paths = [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")]
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1]) == []
