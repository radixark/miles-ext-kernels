"""The MoK expert-parallel MoE training layer (``miles_ext_kernels.cake.mok``) on one GPU: an expert-parallel group of
one rank, the scheduler layout (1, 4) with top-2.

Every case of ``_cases.CASES`` (BF16 and MXFP8 routed experts; plain and clamped SwiGLU; context recompute; BF16 and
FP32 weight-gradient accumulation) runs a full iteration (schedule, forward, backward) over several macro-batches, and
with experts that receive no token. Checked (``_cases``): every output and gradient against the FP32 reference within
twice the BF16 reference's error (MXFP8: FlashInfer's bound on the quantized path; clamped SwiGLU: the tensors its
derivative reaches also against the FP32 reference with BF16 clamp decisions), FlashInfer's own gates, and the
nine outputs repeating bit for bit across two eager calls, CUDA graph replays, two other fixed communication-SM splits
(one with the imbalance switch off) and a fresh process; the checkpointed iteration equals the saved-context one.
Also: a call with no token; calls alternating between the default and a side stream; the weight quantizer's bytes on
the device, whatever the layout of the weights; the libraries ``prebuild`` returns and a new token count loading none;
unsupported layouts raising; no FlashInfer module imported. Needs SM100a / SM103a / SM107a; skips elsewhere.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

import torch.distributed as dist  # noqa: E402

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import mok  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in cake.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake MoK kernels need SM100a / SM103a / SM107a", allow_module_level=True)

from miles_ext_kernels.cake._impl.mok import ops  # noqa: E402

from ..helpers import assert_bitwise  # noqa: E402
from . import _cases  # noqa: E402
from . import _reference as reference  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEVICE = torch.device("cuda", torch.cuda.current_device())
ARCH = cake.device_arch(DEVICE)


@pytest.fixture(scope="module")
def group():
    """A process group of this one rank (NCCL: the workspace gathers its shapes on the device)."""
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            pytest.skip("the single-GPU tests run outside a multi-rank process group")
        yield dist.group.WORLD
        return
    torch.cuda.set_device(DEVICE)
    dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1, device_id=DEVICE)
    try:
        yield dist.group.WORLD
    finally:
        torch.cuda.synchronize()
        dist.destroy_process_group()


@pytest.fixture(scope="module", autouse=True)
def exact_matmul():
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    yield
    (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    ) = saved


@pytest.fixture(scope="module")
def built():
    return cake.prebuild("mok", device=DEVICE, **_cases.model())


@pytest.fixture(scope="module", params=["random", "empty"])
def problem(request, group, built):
    return _cases.Problem(group, DEVICE, routing=request.param)


@pytest.fixture(scope="module")
def results(problem):
    """Case name -> (record, eager outputs), filled on first use."""
    return {}


def run(problem, results, name):
    if name not in results:
        bf16 = results["bf16"][1] if name.startswith("mxfp8") and "bf16" in results else None
        results[name] = _cases.run_case(problem, name, bf16_outputs=bf16)
    return results[name]


@pytest.mark.parametrize("name", _cases.BF16_CASES)
def test_bf16_cases(problem, results, name):
    record, _ = run(problem, results, name)
    assert _cases.failures(record) == [], record


@pytest.mark.parametrize("name", _cases.MXFP8_CASES)
def test_mxfp8_cases(problem, results, name):
    run(problem, results, "bf16")
    record, _ = run(problem, results, name)
    assert _cases.failures(record) == [], record
    if name == "mxfp8":
        assert "rel_l2_vs_bf16_run" in record


def test_outputs_repeat_bit_for_bit_in_a_fresh_process(problem, results, tmp_path):
    names = tuple(_cases.CASES)
    digests = {name: run(problem, results, name)[0]["digest"] for name in names}
    out = tmp_path / "fresh.json"
    env = {k: v for k, v in os.environ.items() if k not in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR")}
    command = [sys.executable, str(HERE / "_worker.py"), "--out", str(out), "--device", str(DEVICE.index)]
    result = subprocess.run(
        [*command, "--routing", problem.routing, "--no-checks", "--cases", *names],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    fresh = {record["case"]: record["digest"] for record in json.loads(out.read_text())["records"]}
    assert fresh == digests


def test_a_call_with_no_token(problem, results):
    """T = 0 on the rank (a workspace sized for more): empty outputs, zero weight gradients; the workspace then still
    serves a full call, bit for bit."""
    _, expected = run(problem, results, "bf16")
    outputs = problem.tokens(0)

    assert [tuple(v.shape) for v in outputs[:3]] == [(0, _cases.HIDDEN), (0, _cases.HIDDEN), (0, problem.topk)]
    weights = (*problem.weights[3:], *problem.weights[:3])  # the gradients' order: routed, then shared
    for name, value, weight in zip(reference.RESULT_NAMES[3:], outputs[3:], weights, strict=True):
        assert value.shape == weight.shape and torch.count_nonzero(value).item() == 0, name
    _, again = _cases.run_case(problem, "bf16", checks=False)
    assert_bitwise(dict(zip(reference.RESULT_NAMES, expected)), dict(zip(reference.RESULT_NAMES, again)))


def test_calls_alternating_with_a_side_stream_repeat_the_bits(problem, results):
    """The kernels run on the caller's current stream: the same iteration on the default stream and on a side
    stream, alternating, gives the same bits (the eager result of the case)."""
    _, expected = run(problem, results, "bf16")
    iteration = problem.iteration("bf16")
    side = torch.cuda.Stream(DEVICE)
    for stream in (side, torch.cuda.current_stream(DEVICE), side, torch.cuda.current_stream(DEVICE)):
        stream.wait_stream(torch.cuda.current_stream(DEVICE))
        with torch.cuda.stream(stream):
            got = iteration.run()
        torch.cuda.synchronize(DEVICE)
        assert_bitwise(dict(zip(reference.RESULT_NAMES, expected)), dict(zip(reference.RESULT_NAMES, got)))


def test_the_weight_quantizer_writes_the_recipe_bytes_on_the_device():
    torch.manual_seed(17)
    gate = torch.randn(3, 256, 512, device=DEVICE, dtype=torch.bfloat16)
    gate[0, 5, 64:96] = 0
    gate[1, 7, :32] *= 1e4
    up = torch.randn_like(gate)
    down = torch.randn(3, 512, 256, device=DEVICE, dtype=torch.bfloat16)
    want = reference.routed_mxfp8_weights(gate, up, down)
    # contiguous, a transposed view and a 2-byte offset copy: one kernel, the same bytes
    flat = torch.empty(down.numel() + 1, device=DEVICE, dtype=torch.bfloat16)
    offset = flat[1:].view(down.shape)
    offset.copy_(down)
    for weights in ((gate, up, down), (gate, up.transpose(1, 2).contiguous().transpose(1, 2), offset)):
        got = mok.quantize_mok_mxfp8_weights(*weights)
        for got_side, want_side in zip(got, want, strict=True):
            for got_weight, want_weight in zip(got_side, want_side, strict=True):
                for a, b in zip(got_weight, want_weight, strict=True):
                    assert a.device == DEVICE and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_prebuild_builds_the_model_libraries_and_new_token_counts_load_none(group, problem, built):
    model = _cases.model()
    names = ops.libraries(ARCH, ep_size=model["ep_size"], local_experts=model["local_experts"], topk=model["topk"])
    assert len(names) == 25
    # <library>_<closure digest>_<cache key>.so in the runtime's cache, one per library of the model
    assert sorted(Path(path).name.rsplit("_", 2)[0] for path in built) == sorted(names)
    assert all(Path(path).is_file() and Path(path).is_relative_to(loader.cache_dir()) for path in built)
    jit = sys.modules["miles_ext_kernels.cake._vendor.cake_mok_bf16.jit"]
    loaded = jit._load.cache_info().misses
    with loader.no_build():
        for tokens in (1, 256, 777):
            config, workspace = mok.create_mok_bf16_workspace(
                group=group, device=DEVICE, num_local_tokens=tokens, hidden_size=_cases.HIDDEN, topk=problem.topk
            )
            problem.tokens(tokens, workspace=workspace, config=config)
    assert jit._load.cache_info().misses == loaded, "a new token count loaded another library"


def test_unsupported_layouts_raise(group):
    with pytest.raises(ValueError, match="layout"):
        mok.prepare_mok_bf16(ep_size=1, local_experts=8, topk=2)
    with pytest.raises(ValueError, match="layout"):
        mok.prepare_mok_bf16(ep_size=1, local_experts=4, topk=4)


def test_the_libraries_build_through_the_loader_without_flashinfer(problem, results):
    run(problem, results, "bf16")
    assert loader.load_package("cake_mok_bf16").cake_build._jit_toolchain.__module__ == loader.__name__
    assert sorted(name for name in sys.modules if name == "flashinfer" or name.startswith("flashinfer.")) == []
