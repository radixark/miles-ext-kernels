"""The fused SwiGLU + MXFP8 quantization kernels (``miles_ext_kernels.cake.gated_act_mxfp8``) on the device.

Every output byte (the E4M3 codes of the rowwise and colwise values and the E8M0 scale bytes in the swizzled 128 x 4
layout) is compared exactly, with no tolerance, against the plain-torch reference of ``_reference.py``: every program
of every route (forward / backward x rowwise / colwise / both) of the device's architecture, run directly, on unit
normals, block-scaled normals, special values, blocks on the E4M3 and E8M0 boundaries (along rows and along columns)
and every BF16 bit pattern as a gate; and the entry points at sizes at which the selection rule picks each program. An
exact comparison is the accuracy gate of this op: any error, a 2 % one included, changes bytes. Independently of that
emulation, the dequantized outputs lie within the MXFP8 rounding bound of the FP64 SwiGLU and of its gradients.

``det`` (the op's one variant) repeats bit for bit: three calls in one process and one in a fresh process. Calls on a
side stream match the default stream, the host never synchronizes, CUDA graph replays match eager calls, a new shape
loads no library, no FlashInfer module is imported; disabled orientations are zero-size and invalid arguments (an
empty batch included) raise. With two GPUs (``multi_gpu``), a call on the device that is not current matches the
reference. Needs SM100a / SM103a / SM107a; skips elsewhere.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake SwiGLU + MXFP8 kernels need SM100a / SM103a / SM107a", allow_module_level=True)

from miles_ext_kernels.cake import gated_act_mxfp8  # noqa: E402
from miles_ext_kernels.cake._impl.gated_act_mxfp8 import ops  # noqa: E402

from . import _reference as reference  # noqa: E402
from ._shapes import ENTRY_SHAPES, KINDS, PROGRAM_SHAPES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEVICE = torch.device("cuda", torch.cuda.current_device())
ARCH = loader.device_arch(DEVICE)
PACKAGE = ops.backend
# (M, K, values) of the run-to-run test: three programs of each route on every architecture, and the boundary blocks
REPEAT_CASES = ((256, 256, "special"), (256, 256, "boundary"), (2048, 1024, "scaled"), (8192, 8192, "normal"))
# |dequantized - FP64| <= |FP64| * (E4M3 half ulp + BF16 rounding and FP32 slack) + the block's E4M3 subnormal half ulp
E4M3_REL = 2.0**-4
BF16_FP32_REL = 2.0**-7
E4M3_SUBNORMAL_HALF_ULP_LOG2 = -10


@pytest.fixture(autouse=True, scope="module")
def built():
    cake.prebuild("gated_act_mxfp8", device=DEVICE)


def call(gated_input, grad_output, *, backward, rowwise, colwise):
    if backward:
        return gated_act_mxfp8.silu_and_mul_mxfp8_quantize_backward(
            gated_input, grad_output, rowwise=rowwise, colwise=colwise
        )
    return gated_act_mxfp8.silu_and_mul_mxfp8_quantize(gated_input, rowwise=rowwise, colwise=colwise)


def expected(gated_input, grad_output, *, backward, rowwise, colwise):
    if backward:
        return reference.backward(gated_input, grad_output, rowwise=rowwise, colwise=colwise)
    return reference.forward(gated_input, rowwise=rowwise, colwise=colwise)


def run_program(name, gated_input, grad_output, *, backward, rowwise, colwise):
    """Run the vendored program ``name`` directly, not through the selection rule."""
    M, K = PACKAGE.cake_gated_act_mxfp8_det_train_check_inputs(
        gated_input, grad_output if backward else None, rowwise=rowwise, colwise=colwise
    )
    outputs = PACKAGE.cake_gated_act_mxfp8_det_train_empty_outputs(
        gated_input, backward=backward, rowwise=rowwise, colwise=colwise
    )
    row_output, col_output, row_scales, col_scales = outputs
    values = {
        "gated_input": gated_input,
        "row_output": row_output.view(torch.uint8),
        "col_output": (col_output.t() if colwise else col_output).view(torch.uint8),
        "row_scales": row_scales,
        "col_scales": col_scales,
        "M": M,
        "K": K,
    }
    if backward:
        values["grad_output"] = grad_output
    PACKAGE.cake_jit.run_module(name, values, device_index=gated_input.device.index)
    return PACKAGE.cake_gated_act_mxfp8_det_train_logical_outputs(outputs)


def assert_bytes(outputs, want, *, rowwise, colwise, what):
    got = reference.output_bytes(outputs, rowwise=rowwise, colwise=colwise)
    assert set(got) == set(want), what
    for key, value in want.items():
        mismatched = int((got[key] != value).sum())
        assert mismatched == 0, f"{what}: {key} differs in {mismatched} bytes"


def as_bytes(outputs) -> dict:
    """The four outputs as uint8 tensors, for a bitwise comparison."""
    names = ("row_output", "col_output", "row_scales", "col_scales")
    return {name: tensor.view(torch.uint8) for name, tensor in zip(names, outputs, strict=True)}


def assert_same_bits(a, b, what):
    for (name, x), y in zip(as_bytes(a).items(), as_bytes(b).values(), strict=True):
        assert x.shape == y.shape and x.stride() == y.stride(), f"{what}: {name} layout differs"
        assert torch.equal(x, y), f"{what}: {name} differs"


def digest(outputs) -> str:
    """SHA-256 of the bytes of the four outputs, each made row-major."""
    hasher = hashlib.sha256()
    for tensor in outputs:
        hasher.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return hasher.hexdigest()


@pytest.mark.parametrize("values", reference.VALUES)
@pytest.mark.parametrize("kind", tuple(KINDS))
def test_every_program_matches_the_reference(kind, values):
    backward, rowwise, colwise = KINDS[kind]
    names = PACKAGE.cake_jit.route_modules(ARCH, kind)
    for M, K in ((256, 256),) if values == "bits" else PROGRAM_SHAPES:
        gated_input, grad_output = reference.make_inputs(M, K, values=values, seed=M + K, device=DEVICE)
        want = expected(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
        for name in names:
            outputs = run_program(name, gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
            assert_bytes(outputs, want, rowwise=rowwise, colwise=colwise, what=f"{kind} {name} M={M} K={K} {values}")


@pytest.mark.parametrize("shape", ENTRY_SHAPES, ids=[f"{M}x{K}" for M, K in ENTRY_SHAPES])
@pytest.mark.parametrize("kind", tuple(KINDS))
def test_entry_points_match_the_reference(kind, shape):
    backward, rowwise, colwise = KINDS[kind]
    M, K = shape
    gated_input, grad_output = reference.make_inputs(M, K, values="normal", seed=7, device=DEVICE)
    outputs = call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
    want = expected(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
    assert_bytes(outputs, want, rowwise=rowwise, colwise=colwise, what=f"{kind} M={M} K={K}")


@pytest.mark.parametrize("values", ("normal", "scaled"))
@pytest.mark.parametrize("kind", tuple(KINDS))
def test_dequantized_outputs_are_within_the_mxfp8_bound_of_fp64(kind, values):
    backward, rowwise, colwise = KINDS[kind]
    for M, K in ((256, 384), (2048, 2048)):
        gated_input, grad_output = reference.make_inputs(M, K, values=values, seed=11, device=DEVICE)
        row_output, col_output, row_scales, col_scales = call(
            gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise
        )
        exact = reference.fp64_values(gated_input, grad_output if backward else None)
        checks = []
        if rowwise:
            checks.append(("row", row_output, row_scales, exact))
        if colwise:  # the column-major values are stored as their [N, M] transpose, scaled per 32 rows
            checks.append(("col", col_output.t(), col_scales, exact.t()))
        for orientation, codes, scales, want in checks:
            got, exponent = reference.dequantize(codes, scales)
            bound = want.abs() * (E4M3_REL + BF16_FP32_REL) + torch.exp2(
                (exponent - 127 + E4M3_SUBNORMAL_HALF_ULP_LOG2).double()
            )
            err = (got - want).abs()
            bad = err > bound
            assert not bad.any(), (
                f"{kind} {orientation} M={M} K={K} {values}: {int(bad.sum())} of {bad.numel()} beyond the MXFP8 "
                f"bound; worst error / bound {(err / bound).max().item():.3f}"
            )


_FRESH_PROCESS = """
import hashlib, json, sys
import torch
from miles_ext_kernels.cake.gated_act_mxfp8 import silu_and_mul_mxfp8_quantize, silu_and_mul_mxfp8_quantize_backward

device = torch.device("cuda", int(sys.argv[2]))
torch.cuda.set_device(device)
out = {}
for gated_input, grad_output, routes in torch.load(sys.argv[1]):
    gated_input, grad_output = gated_input.to(device), grad_output.to(device)
    for key, (backward, rowwise, colwise) in routes.items():
        if backward:
            outputs = silu_and_mul_mxfp8_quantize_backward(gated_input, grad_output, rowwise=rowwise, colwise=colwise)
        else:
            outputs = silu_and_mul_mxfp8_quantize(gated_input, rowwise=rowwise, colwise=colwise)
        hasher = hashlib.sha256()
        for tensor in outputs:
            hasher.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        out[key] = hasher.hexdigest()
print(json.dumps(out))
"""


def test_det_repeats_bit_for_bit_in_process_and_in_a_fresh_process(tmp_path):
    cases = []  # (gated_input, grad_output, {key: route}): each input is saved once and shared by the six routes
    digests = {}
    for M, K, values in REPEAT_CASES:
        gated_input, grad_output = reference.make_inputs(M, K, values=values, seed=M * K, device=DEVICE)
        routes = {}
        for kind, (backward, rowwise, colwise) in KINDS.items():
            key = f"{kind} M={M} K={K} {values}"
            first = call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
            for _ in range(2):
                again = call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
                assert_same_bits(first, again, f"{key}, between calls")
            digests[key] = digest(first)
            routes[key] = (backward, rowwise, colwise)
        cases.append((gated_input.cpu(), grad_output.cpu(), routes))
    path = tmp_path / "cases.pt"
    torch.save(cases, path)
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS, str(path), str(DEVICE.index)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    fresh = json.loads(result.stdout.strip().splitlines()[-1])
    assert fresh == digests


def test_calls_on_a_side_stream_match_the_default_stream():
    """The kernels run on the current torch stream: alternating a side stream and the default stream gives the same
    bits on every route."""
    gated_input, grad_output = reference.make_inputs(2048, 2048, values="scaled", seed=13, device=DEVICE)
    side = torch.cuda.Stream(device=DEVICE)
    for kind, (backward, rowwise, colwise) in KINDS.items():
        want = call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
        for round_ in range(4):
            stream = side if round_ % 2 == 0 else torch.cuda.current_stream(DEVICE)
            stream.wait_stream(torch.cuda.current_stream(DEVICE))
            with torch.cuda.stream(stream):
                got = call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
            torch.cuda.current_stream(DEVICE).wait_stream(stream)
            assert_same_bits(want, got, f"{kind} round {round_}")


def test_calls_never_synchronize_the_host():
    gated_input, grad_output = reference.make_inputs(1024, 2048, device=DEVICE)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for backward, rowwise, colwise in KINDS.values():
            call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


def test_output_layouts_and_disabled_outputs():
    M, K = 256, 384
    gated_input, grad_output = reference.make_inputs(M, K, device=DEVICE)
    for backward in (False, True):
        N = 2 * K if backward else K
        for rowwise, colwise in ((True, False), (False, True), (True, True)):
            outputs = call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
            row_output, col_output, row_scales, col_scales = outputs
            assert row_output.dtype == col_output.dtype == torch.float8_e4m3fn
            assert row_scales.dtype == col_scales.dtype == torch.float8_e8m0fnu
            assert all(tensor.device == DEVICE for tensor in outputs)
            if rowwise:
                assert row_output.shape == (M, N) and row_output.stride() == (N, 1)
                assert row_scales.shape == (M, N // 32)
            else:
                assert row_output.numel() == 0 and row_scales.numel() == 0
            if colwise:
                assert col_output.shape == (M, N) and col_output.stride() == (1, M)
                assert col_scales.shape == (N * (M // 32),)
            else:
                assert col_output.numel() == 0 and col_scales.numel() == 0


def test_invalid_arguments_raise():
    M, K = 256, 256
    gated_input, grad_output = reference.make_inputs(M, K, device=DEVICE)
    forward = gated_act_mxfp8.silu_and_mul_mxfp8_quantize
    backward = gated_act_mxfp8.silu_and_mul_mxfp8_quantize_backward
    with pytest.raises(ValueError, match="CUDA tensor"):
        forward(gated_input.cpu())
    with pytest.raises(TypeError, match="bfloat16"):
        forward(gated_input.float())
    with pytest.raises(ValueError, match="contiguous"):
        forward(gated_input.t())
    with pytest.raises(ValueError, match="multiples of 128"):
        forward(gated_input[:200])
    with pytest.raises(ValueError, match="multiples of 128"):
        forward(gated_input[:, : 2 * 192].contiguous())
    with pytest.raises(ValueError, match="multiples of 128"):  # an empty batch
        forward(gated_input[:0])
    with pytest.raises(ValueError, match="multiples of 128"):
        backward(gated_input[:0], grad_output[:0])
    shifted = torch.empty(M * 2 * K + 8, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError, match="32-byte aligned"):
        forward(shifted[8:].view(M, 2 * K))
    with pytest.raises(ValueError, match="rowwise or colwise"):
        forward(gated_input, rowwise=False, colwise=False)
    with pytest.raises(ValueError, match="grad_output must be contiguous"):
        backward(gated_input, grad_output[:128])
    with pytest.raises(ValueError, match="grad_output must be contiguous"):
        backward(gated_input, grad_output.t())
    with pytest.raises(TypeError, match="bfloat16"):
        backward(gated_input, grad_output.float())
    with pytest.raises(ValueError, match="CUDA tensor"):
        backward(gated_input, grad_output.cpu())


def test_cuda_graph_replays_match_eager():
    M, K = 512, 1024
    gated_input, grad_output = reference.make_inputs(M, K, values="scaled", seed=3, device=DEVICE)
    routes = list(KINDS.values())
    eager = [call(gated_input, grad_output, backward=b, rowwise=r, colwise=c) for b, r, c in routes]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for b, r, c in routes:
            call(gated_input, grad_output, backward=b, rowwise=r, colwise=c)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [call(gated_input, grad_output, backward=b, rowwise=r, colwise=c) for b, r, c in routes]
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for want, got in zip(eager, captured, strict=True):
            assert_same_bits(want, got, "graph replay")


def test_a_new_shape_loads_no_library():
    jit = PACKAGE.cake_jit
    for kind in jit.KINDS:
        for name in jit.route_modules(ARCH, kind):
            jit.load(name, ARCH)
    loaded = jit.load.cache_info().misses
    for M, K in ((128, 128), (256, 384), (4096, 2048), (1024, 7168), (8192, 8192)):
        gated_input, grad_output = reference.make_inputs(M, K, device=DEVICE)
        for backward, rowwise, colwise in KINDS.values():
            call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
    torch.cuda.synchronize()
    assert jit.load.cache_info().misses == loaded, "a new problem size loaded another library"


def test_the_libraries_build_through_the_runtime_loader_without_flashinfer():
    gated_input, grad_output = reference.make_inputs(256, 256, device=DEVICE)
    for backward, rowwise, colwise in KINDS.values():
        call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
    torch.cuda.synchronize()

    assert PACKAGE.cake_build._jit_toolchain.__module__ == loader.__name__
    assert sorted(name for name in sys.modules if name == "flashinfer" or name.startswith("flashinfer.")) == []


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_a_call_on_a_device_that_is_not_current_matches_the_reference():
    other = torch.device("cuda", (DEVICE.index + 1) % torch.cuda.device_count())
    if tuple(torch.cuda.get_device_capability(other)) not in loader.ARCH_BY_CAPABILITY:
        pytest.skip(f"{other} is not SM100a / SM103a / SM107a")
    cake.prebuild("gated_act_mxfp8", device=other)
    with torch.cuda.device(DEVICE):
        for kind, (backward, rowwise, colwise) in KINDS.items():
            gated_input, grad_output = reference.make_inputs(1024, 2048, values="scaled", seed=5, device=other)
            outputs = call(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
            assert all(tensor.device == other for tensor in outputs)
            want = expected(gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise)
            assert_bytes(outputs, want, rowwise=rowwise, colwise=colwise, what=f"{kind} on {other}")
            here = call(
                gated_input.to(DEVICE), grad_output.to(DEVICE), backward=backward, rowwise=rowwise, colwise=colwise
            )
            assert digest(outputs) == digest(here), f"{kind}: {other} and {DEVICE} differ"
