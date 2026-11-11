"""The prebuild contract of conv1d on a GPU: after one ``prebuild("conv1d", weight_dtype=...)`` no call compiles.

:data:`ROUTES` maps each call-time path of the op to one forward + backward of it: packed documents over every channel
threshold of the packages' program selection (``_shapes.py``; ``test_conv1d_ops.py`` checks on the CPU that these
shapes reach every program), one document without ``cu_seqlens``, the ``fp32`` SiLU' policy, a forward without
autograd, and the context-parallel route (both launch forms of ``det``'s backward, the multi-tensor call, the
checkpoint recompute handle and a pack past the token floor of the single-rank forward program its forward runs). The
check runs once per taps dtype (FP32, BF16), each in a new process with an empty library cache of its own, so that a
library only the other dtype's ``prebuild`` builds cannot hide a gap: one ``prebuild``, then every route of each
variant under ``no_build()``; that process also checks that nothing imported flashinfer. In this process: a variant
configured after a call takes effect at the next call, without a build. Needs SM100a / SM103a / SM107a.
"""

import functools
import itertools
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
pytest.importorskip("tvm_ffi")

from torch.utils.checkpoint import checkpoint  # noqa: E402

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.conv1d import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

from ..helpers import no_build  # noqa: E402
from ._kcp import WIDTH, fragments, one_rank_context  # noqa: E402
from ._shapes import KCP_CHANNELS, KCP_CUTS, KCP_DOCUMENTS, KCP_LONG_POINT, SINGLE_RANK_POINTS  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake short-conv kernels need SM100a / SM103a / SM107a", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEVICE = torch.device("cuda", torch.cuda.current_device())
F32, BF16 = torch.float32, torch.bfloat16


def _taps(C, taps, generator):
    return (0.5 * torch.randn(C, WIDTH, device=DEVICE, generator=generator)).to(taps).requires_grad_()


def _single(lengths, C, taps, *, cu_seqlens=True, grad=True):
    """One ``causal_conv1d`` of documents of ``lengths`` tokens and ``C`` channels, forward and backward."""
    S = sum(lengths)
    g = torch.Generator(device=DEVICE).manual_seed(S * 31 + C)
    x = torch.randn(1, S, C, device=DEVICE, generator=g).bfloat16().requires_grad_()
    w = _taps(C, taps, g)
    cu = None
    if cu_seqlens:
        cu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=DEVICE)
    if not grad:
        with torch.no_grad():
            ops.causal_conv1d(x, w, cu)
        return
    y = ops.causal_conv1d(x, w, cu)
    y.backward(torch.ones_like(y))


def _kcp_inputs(C, taps, documents=KCP_DOCUMENTS, cuts=KCP_CUTS):
    S, _, frags = fragments(documents, cuts)
    g = torch.Generator(device=DEVICE).manual_seed(S * 31 + C)
    x = torch.randn(S, C, device=DEVICE, generator=g).bfloat16().requires_grad_()
    return x, _taps(C, taps, g), one_rank_context(frags, DEVICE)


def packed_documents(variant, taps=F32):
    for C, lengths in SINGLE_RANK_POINTS:
        _single(lengths, C, taps)


def no_cu_seqlens(variant, taps=F32):
    for C in (136, 512, 2560):
        _single((4097,), C, taps, cu_seqlens=False)


def silu_fp32(variant, taps=F32):
    cake.configure(cake_conv1d_variant=variant, cake_conv1d_silu_grad_preactivation="fp32")
    for C, lengths in SINGLE_RANK_POINTS:
        _single(lengths, C, taps)


def forward_only(variant, taps=F32):
    for C, lengths in SINGLE_RANK_POINTS:
        _single(lengths, C, taps, grad=False)


def kcp(variant, taps=F32, **kwargs):
    for C in KCP_CHANNELS:
        x, w, ctx = _kcp_inputs(C, taps)
        y = ops.causal_conv1d_kcp(x, w, ctx, **kwargs)
        y.backward(torch.ones_like(y))


def kcp_cooperative_backward(variant, taps=F32):
    """``det``'s cooperative backward (its default is the ordinary launch); ``fast`` takes only the ordinary one."""
    kcp(variant, taps, cooperative=variant == "det")


def kcp_long_pack(variant, taps=F32):
    C, lengths = KCP_LONG_POINT
    x, w, ctx = _kcp_inputs(C, taps, lengths, lengths)
    y = ops.causal_conv1d_kcp(x, w, ctx)
    y.backward(torch.ones_like(y))


def kcp_multi(variant, taps=F32):
    inputs = [_kcp_inputs(C, taps) for C in KCP_CHANNELS]
    ys = ops.causal_conv1d_kcp_multi([x for x, _, _ in inputs], [w for _, w, _ in inputs], inputs[0][2])
    torch.autograd.backward(list(ys), [torch.ones_like(y) for y in ys])


def kcp_recompute_handle(variant, taps=F32):
    for C in KCP_CHANNELS:
        x, w, ctx = _kcp_inputs(C, taps)
        handle = types.SimpleNamespace(entries={})
        layer = functools.partial(ops.causal_conv1d_kcp, kcp_context=ctx, recompute_handle=handle)
        y = checkpoint(layer, x, w, use_reentrant=False)
        y.backward(torch.ones_like(y))


# every call-time path of the op: name -> run(variant, taps=...), one forward + backward
ROUTES = {
    "packed-documents": packed_documents,
    "no-cu_seqlens": no_cu_seqlens,
    "silu-fp32": silu_fp32,
    "forward-only": forward_only,
    "kcp": kcp,
    "kcp-cooperative-backward": kcp_cooperative_backward,
    "kcp-long-pack": kcp_long_pack,
    "kcp-multi": kcp_multi,
    "kcp-recompute-handle": kcp_recompute_handle,
}

_FRESH = """
import functools
import importlib
import sys

sys.path.insert(0, {tests!r})
import torch

from miles_ext_kernels import cake
from cake.conv1d import test_prebuild_gpu as t
from cake.helpers import check_prebuild

for op in cake.OPS:
    importlib.import_module(f"miles_ext_kernels.cake._impl.{{op}}")
taps = getattr(torch, {taps!r})
routes = {{name: functools.partial(run, taps=taps) for name, run in t.ROUTES.items()}}
check_prebuild("conv1d", routes, weight_dtype=taps)
assert not [name for name in sys.modules if name.split(".")[0] == "flashinfer"], "flashinfer was imported"
print("FRESH_OK")
"""


@pytest.mark.parametrize("taps", ["float32", "bfloat16"], ids=["fp32-taps", "bf16-taps"])
def test_no_call_builds_after_prebuild(tmp_path, taps):
    """In a new process with an empty library cache of its own: one ``prebuild`` for taps of ``taps``, then every
    route of every variant under ``no_build()``."""
    script = _FRESH.format(tests=str(REPO_ROOT / "tests"), taps=taps)
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    env = {**os.environ, "PYTHONPATH": pythonpath, "TVM_FFI_CACHE_DIR": str(tmp_path / "cache")}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and "FRESH_OK" in result.stdout, result.stdout[-2000:] + result.stderr[-6000:]


@pytest.fixture
def entries(monkeypatch):
    """The package entry each call runs, recorded (the calls still run the kernels)."""
    calls = []
    for variant in ops.PACKAGES:
        package, kcp_module = ops._package(variant), ops._kcp_module(variant)
        names = (
            [(package, "cake_causal_conv1d_det"), (kcp_module, "cake_causal_conv1d_det_kcp")]
            if variant == "det"
            else [(package, "cake_causal_conv1d_nondet"), (kcp_module, "cake_causal_conv1d_nondet_kcp")]
        )
        for module, name in names:
            original = getattr(module, name)

            def record(*args, _call=(variant, name), _original=original, **kwargs):
                calls.append(_call)
                return _original(*args, **kwargs)

            monkeypatch.setattr(module, name, record)
    return calls


def test_a_variant_configured_after_a_call_runs_at_the_next_call_without_a_build(entries):
    cake.prebuild("conv1d", weight_dtype=F32)
    cake.configure(cake_conv1d_variant="det")
    packed_documents("det")
    kcp("det")
    cake.configure(cake_conv1d_variant="fast")
    with no_build():
        _single((1000, 0, 3, 3094), 2560, F32)
        kcp("fast")
    det_calls = [("det", "cake_causal_conv1d_det")] * len(SINGLE_RANK_POINTS)
    det_calls += [("det", "cake_causal_conv1d_det_kcp")] * len(KCP_CHANNELS)
    fast_calls = [("fast", "cake_causal_conv1d_nondet")]
    fast_calls += [("fast", "cake_causal_conv1d_nondet_kcp")] * len(KCP_CHANNELS)
    assert entries == det_calls + fast_calls
