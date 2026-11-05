"""The det kernels with FLA's context parallelism (``cake_gdn_fla_cp_det``), built by the runtime loader with nvcc and
loaded with tvm-ffi, as every call with an FLA ``cp_context`` runs them (here one rank: a context without a process
group; two and four ranks: ``test_gdn_kcp_gpu.py``).

Every stage of the device's architecture loads from the loader's library, compiled by the toolkit the loader uses;
the call repeats bit for bit in one process (also under PyTorch's deterministic algorithms), in a new process and on a
side stream alternating with the default one (with the workspace pool bounded, ``helpers.POOLED``); the context's
boundaries replace the call's ``cu_seqlens`` / ``cu_seqlens_cpu`` (FLA's contract), with the bits of the plain call on
those boundaries; an unused final state passes no gradient (the bits of the call without one) and a loss on the final
state alone backpropagates (the bits of an explicit zero output gradient); output and gradients within the BF16
tolerance of flash-linear-attention. Needs an SM100a / SM103a / SM107a GPU; the FLA comparison needs fla.
"""

import contextlib
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

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.gdn import native, ops  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

from ..helpers import POOLED  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the GDN kernels need SM100a / SM103a / SM107a", allow_module_level=True)

HEAD_DIM = 128
HEADS = (4, 12)  # key heads, value heads of one tensor-parallel rank
# a TP4 micro-batch: three documents and the 512-token alignment pad as its own segment
LAYOUT = (3339, 2802, 1340, 263)
NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")


@pytest.fixture(autouse=True)
def _kernel_settings():
    cake.configure(cake_gdn_variant="det")


def _cu(seq_lens) -> tuple[int, ...]:
    return tuple(itertools.accumulate(seq_lens, initial=0))


def _context(seq_lens) -> types.SimpleNamespace:
    """An FLA context without a process group: this rank's boundaries on the device and on the host."""
    cu = _cu(seq_lens)
    return types.SimpleNamespace(
        group=None, cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="cuda"), cu_seqlens_cpu=cu
    )


def _inputs(seq_lens, *, seed: int) -> dict:
    num_heads, num_v_heads = HEADS
    total = sum(seq_lens)
    gen = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn(1, total, num_heads, HEAD_DIM, generator=gen).to("cuda", torch.bfloat16),
        "k": torch.randn(1, total, num_heads, HEAD_DIM, generator=gen).to("cuda", torch.bfloat16),
        "v": torch.randn(1, total, num_v_heads, HEAD_DIM, generator=gen).to("cuda", torch.bfloat16),
        "g": (torch.rand(1, total, num_v_heads, generator=gen) * -0.5).cuda(),
        "beta": torch.rand(1, total, num_v_heads, generator=gen).to("cuda", torch.bfloat16),
        "do": torch.randn(1, total, num_v_heads, HEAD_DIM, generator=gen).to("cuda", torch.bfloat16),
    }


def _run(inputs, kernel=None, **kwargs) -> dict:
    """One forward and backward; the output and the gradients of q, k, v, g and beta."""
    leaves = {n: inputs[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g", "beta")}
    kernel = ops.chunk_gated_delta_rule if kernel is None else kernel
    o, final_state = kernel(**leaves, use_qk_l2norm_in_kernel=True, **kwargs)
    assert final_state is None
    o.backward(inputs["do"])
    torch.cuda.synchronize()
    return {"o": o.detach(), **{f"d{n}": t.grad for n, t in leaves.items()}}


def _assert_same_bits(got: dict, want: dict, what: str) -> None:
    differing = [name for name in want if not torch.equal(got[name], want[name])]
    assert not differing, f"{what}: not the same bits: {differing}"


@contextlib.contextmanager
def _deterministic_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_every_stage_loads_from_the_loader_s_library():
    arch = loader.device_arch()
    cake_build = loader.load_package(native.PACKAGE).cake_build
    names = native.libraries(arch)
    assert names and set(names) <= set(cake_build.libraries_by_name())
    cudart = loader.get_cuda_version().cudart
    for stage, records in native.MODULES.items():
        if arch not in records:
            continue
        module = native.load(stage, arch)
        assert int(module.build_cuda_version()) == cudart, stage
        assert callable(getattr(module, records[arch]["ffi_entry"])), stage


@pytest.mark.parametrize("route", ["host", "device"])
def test_the_call_repeats_bit_for_bit(route):
    inputs = _inputs(LAYOUT, seed=1)
    context = _context(LAYOUT)
    if route == "device":
        context.cu_seqlens_cpu = None

    first = _run(inputs, cp_context=context)
    _assert_same_bits(_run(inputs, cp_context=context), first, "a second run")
    with _deterministic_algorithms():
        _assert_same_bits(_run(inputs, cp_context=context), first, "under deterministic algorithms")
    cake.configure(cake_gdn_variant="det", cake_gdn_recompute_state=True)
    # the kernels recompute w / u / h / v_new exactly: the same bits with the tape off
    _assert_same_bits(_run(inputs, cp_context=context), first, "the recomputed backward")


def test_a_side_stream_alternating_with_the_default_stream_gives_the_same_bits():
    # with a positive bound the pool keeps the scratch per stream (the default, 0, pools nothing)
    cake.configure(cake_gdn_variant="det", **POOLED)
    inputs = _inputs(LAYOUT, seed=2)
    context = _context(LAYOUT)
    first = _run(inputs, cp_context=context)
    side = torch.cuda.Stream()
    for _ in range(2):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            on_side = _run(inputs, cp_context=context)
        torch.cuda.current_stream().wait_stream(side)
        _assert_same_bits(on_side, first, "on a side stream")
        _assert_same_bits(_run(inputs, cp_context=context), first, "back on the default stream")


_NEW_PROCESS = """
import sys
import torch
from miles_ext_kernels import cake
from cake.gdn import test_gdn_fla_cp_gpu as t

cake.configure(cake_gdn_variant="det")
torch.save(t._run(t._inputs(t.LAYOUT, seed=3), cp_context=t._context(t.LAYOUT)), sys.argv[1])
"""


def test_the_call_repeats_bit_for_bit_in_a_new_process(tmp_path):
    here = _run(_inputs(LAYOUT, seed=3), cp_context=_context(LAYOUT))
    tests = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(filter(None, [str(tests.parent), str(tests), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NEW_PROCESS, str(tmp_path / "bits.pt")],
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    _assert_same_bits(torch.load(tmp_path / "bits.pt", map_location="cuda"), here, "a new process")


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_the_context_s_boundaries_replace_the_call_s(variant):
    # FLA's contract: under a cp_context the call's cu_seqlens / cu_seqlens_cpu are not read; both variants run the
    # det kernels, with the bits of the call without a context on the same boundaries
    cake.configure(cake_gdn_variant=variant)
    inputs = _inputs(LAYOUT, seed=4)
    whole = _cu((sum(LAYOUT),))

    got = _run(
        inputs,
        cp_context=_context(LAYOUT),
        cu_seqlens=torch.tensor(whole, dtype=torch.int32, device="cuda"),
        cu_seqlens_cpu=whole,
    )
    want = _run(
        inputs,
        kernel=ops.cp_chunk_gated_delta_rule,
        cu_seqlens=torch.tensor(_cu(LAYOUT), dtype=torch.int32, device="cuda"),
        cu_seqlens_cpu=_cu(LAYOUT),
    )
    _assert_same_bits(got, want, "the context's boundaries vs the same boundaries as arguments")


def _with_final_state(inputs, grads: dict) -> dict:
    """One call of the kernels as one rank without a context (:func:`ops.cp_chunk_gated_delta_rule`), returning the
    final state, and one backward of ``grads``: the incoming gradient of each output (``"o"``, ``"final_state"``) that
    takes part. The output and the gradients of q, k, v, g and beta."""
    leaves = {n: inputs[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "g", "beta")}
    cu = _cu(LAYOUT)
    o, final_state = ops.cp_chunk_gated_delta_rule(
        **leaves,
        output_final_state=True,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        cu_seqlens_cpu=cu,
        use_qk_l2norm_in_kernel=True,
    )
    outputs = {"o": o, "final_state": final_state}
    torch.autograd.backward([outputs[n] for n in grads], list(grads.values()))
    torch.cuda.synchronize()
    return {"o": o.detach(), **{f"d{n}": t.grad for n, t in leaves.items()}}


def test_an_unused_final_state_passes_no_gradient_and_a_final_state_alone_backpropagates():
    # the backward takes None for an output whose gradient does not reach it (no zeros are materialised)
    inputs = _inputs(LAYOUT, seed=6)
    cu = _cu(LAYOUT)
    without = _run(
        inputs,
        kernel=ops.cp_chunk_gated_delta_rule,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        cu_seqlens_cpu=cu,
    )
    _assert_same_bits(_with_final_state(inputs, {"o": inputs["do"]}), without, "an unused final state")

    gen = torch.Generator().manual_seed(7)
    dht = (torch.randn(len(LAYOUT), HEADS[1], HEAD_DIM, HEAD_DIM, generator=gen) * 0.1).cuda()
    alone = _with_final_state(inputs, {"final_state": dht})
    explicit = _with_final_state(inputs, {"o": torch.zeros_like(inputs["do"]), "final_state": dht})
    _assert_same_bits(alone, explicit, "a loss on the final state alone vs an explicit zero output gradient")
    assert all(torch.isfinite(alone[n].float()).all() for n in NAMES)


def test_the_call_matches_fla():
    fla = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule
    inputs = _inputs(LAYOUT, seed=5)
    cu = torch.tensor(_cu(LAYOUT), dtype=torch.int32, device="cuda")

    got = _run(inputs, cp_context=_context(LAYOUT))
    ref = _run(inputs, kernel=fla, cu_seqlens=cu)
    # two BF16 implementations of one dataflow: relative RMS error at most 1e-2 and the largest absolute error at most
    # 4e-2 of the largest reference magnitude, per tensor
    report = {}
    for name in NAMES:
        a, r = got[name].double(), ref[name].double()
        diff = (a - r).abs()
        rms = float(r.square().mean().sqrt())
        rel = float(diff.square().mean().sqrt()) / rms if rms else 0.0
        report[name] = (rel, float(diff.max()), float(r.abs().max()))
    failed = [n for n, (rel, worst, top) in report.items() if not (rel <= 1e-2 and worst <= 4e-2 * top)]
    assert not failed and all(torch.isfinite(got[n].float()).all() for n in NAMES), (failed, report)
