"""The module surface of the Cake input RMSNorm on CPU and meta tensors (needs torch and tvm_ffi, no GPU).

Construction (meta device, ``reset_parameters``, ``sequence_parallel``, hidden-size refusal, no build), the variant a
call reads, ``fast`` running ``det`` under torch's deterministic algorithms and the prebuild arguments (the refusal
without a CUDA device in a child process that sees none); the kernels themselves are replaced by a recorder, so no call
launches anything. Also the programs that need CUDA 13.4's oversized shared-memory mode and the refusal of their width
in a process that cannot launch them, with the runtime and driver facts replaced.
"""

import importlib
import inspect
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

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

REPO_ROOT = Path(__file__).resolve().parents[3]
EPS = 1e-6
SUPPORTED = (256, 512, 1536, 2048, 4096, 5120, 6144)
UNSUPPORTED = (1024, 2560, 3072)


def _packages():
    """``(det, fast)``: the vendored packages of the two variants."""
    return loader.load_package("cake_rmsnorm_det_train"), loader.load_package("cake_rmsnorm_nondet_train")


@pytest.fixture
def calls(monkeypatch):
    """The variant of every kernel call (``"det"``: the det package's autograd entry, ``"fast"``: the nondet
    package's); each call returns its input rows instead of launching. The stand-ins replace existing entries only and
    take no ``deterministic`` keyword, as the packages' entries."""
    recorded = []

    def entry(variant):
        def run(x, w, eps, residual=None):
            assert w.dtype == torch.float32 and eps == EPS
            recorded.append(variant)
            return x.clone()

        return run

    det, fast = _packages()
    monkeypatch.setattr(det, "cake_rmsnorm_det", entry("det"))
    monkeypatch.setattr(fast, "cake_rmsnorm_nondet", entry("fast"))
    return recorded


@pytest.fixture
def no_build(monkeypatch):
    """Any library build or prebuild raises."""

    def refuse(*args, **kwargs):
        raise AssertionError("a library was built")

    monkeypatch.setattr(loader, "prebuild_libraries", refuse)
    monkeypatch.setattr(loader.JitSpec, "build", refuse)


def _sequence_parallel_params(module):
    """The parameters whose gradient Megatron all-reduces across tensor-parallel ranks under sequence parallelism."""
    return [param for param in module.parameters() if getattr(param, "sequence_parallel", False)]


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_every_architecture_serves_the_same_hidden_sizes(arch):
    assert rmsnorm.supported_hidden_sizes(arch) == SUPPORTED


def _ops():
    return importlib.import_module("miles_ext_kernels.cake._impl.rmsnorm.ops")


# The programs that launch in CUDA 13.4's oversized shared-memory mode: on SM107a, det's backward reduce above 4096
# tokens and its copy in fast's backward from 4097 to 65536 tokens, both at hidden size 4096.
OVERSIZED = {
    ("sm_107a", 4096): (
        ("cake_rmsnorm_det_train_10e24516314c244771f4_sm_107a", 271360),
        ("cake_rmsnorm_train_10e24516314c244771f4_sm_107a", 271360),
    ),
}


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_only_sm107a_at_4096_launches_programs_in_the_oversized_shared_memory_mode(arch):
    for hidden in SUPPORTED:
        assert _ops()._oversized_programs(arch, hidden) == OVERSIZED.get((arch, hidden), ()), hidden


@pytest.fixture
def device_facts(monkeypatch):
    """The CUDA runtime version and the driver's shared-memory attributes the op reads, as set in the returned dict
    (an R200 with a CUDA 13.4 runtime unless a test changes them); no device is touched."""
    ops = _ops()
    facts = {"runtime": 13040, 97: 232448, 150: 334848}
    monkeypatch.setattr(ops, "_cuda_runtime_version", lambda major: facts["runtime"] if major == 13 else None)
    monkeypatch.setattr(ops, "_device_attribute", lambda index, attribute: facts[attribute])
    return facts


def test_a_cuda_13_4_runtime_and_driver_launch_the_oversized_programs(device_facts):
    for hidden in SUPPORTED:
        _ops()._check_shared_memory("sm_107a", hidden, 0, 13)


@pytest.mark.parametrize(
    "facts, problem",
    [
        ({"runtime": 13000}, r"the process's CUDA runtime \(libcudart\.so\.13\) is 13\.0"),
        ({"runtime": None}, r"no CUDA runtime libcudart\.so\.13 loads"),
        ({150: None}, r"the driver reports no oversized shared-memory ceiling"),
        ({150: 262144}, r"the driver's oversized shared-memory ceiling is 262144 B"),
    ],
    ids=["runtime-13.0", "no-runtime", "no-oversized-ceiling", "small-ceiling"],
)
def test_a_process_that_cannot_launch_them_is_refused_at_4096_on_sm107a(device_facts, facts, problem):
    """R200 with torch's CUDA 13.0 runtime wheel: the oversized programs failed at their launch with
    "cudaFuncSetAttribute failed: invalid argument". The op refuses the width first, naming the programs, the reason
    and the remedy; every other width and architecture runs."""
    device_facts.update(facts)
    ops = _ops()
    with pytest.raises(RuntimeError) as refused:
        ops._check_shared_memory("sm_107a", 4096, 1, 13)
    message = str(refused.value)
    assert message.startswith("--linear-attention-input-norm cake: hidden size 4096 on sm_107a launches ")
    for library, smem in OVERSIZED[("sm_107a", 4096)]:
        assert library in message and f"with {smem} B of shared memory per block" in message
    assert "above the 232448 B opt-in of CUDA device 1" in message
    assert re.search(problem, message), message
    assert message.endswith("(pip: nvidia-cuda-runtime>=13.4) or use --linear-attention-input-norm te")
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for hidden in SUPPORTED:
            if (arch, hidden) != ("sm_107a", 4096):
                ops._check_shared_memory(arch, hidden, 1, 13)


@pytest.mark.parametrize("hidden", UNSUPPORTED)
def test_unsupported_hidden_sizes_are_refused_at_construction(hidden):
    with pytest.raises(ValueError, match=f"not {hidden}; use --linear-attention-input-norm te"):
        rmsnorm.ZeroCenteredRMSNorm(hidden, EPS, device="meta")


@pytest.mark.parametrize("hidden", UNSUPPORTED)
def test_check_supported_refuses_the_hidden_size_before_the_device(hidden):
    with pytest.raises(ValueError, match=f"not {hidden}"):
        rmsnorm.check_supported(hidden)


def test_the_module_has_the_te_parameter_surface(no_build):
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS, device="cpu")

    assert list(norm.state_dict()) == ["weight"]
    assert norm.weight.dtype == torch.bfloat16 and norm.weight.shape == (2048,) and not norm.weight.any()
    # as TE's norms: no sequence_parallel argument, no attribute (the caller may set it)
    assert not hasattr(norm.weight, "sequence_parallel") and _sequence_parallel_params(norm) == []
    assert "sequence_parallel=None" in repr(norm)


def test_a_wrong_width_is_refused(no_build, calls):
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS, device="cpu")

    with pytest.raises(ValueError, match=r"expected \[\.\.\., 2048\] input"):
        norm(torch.zeros(3, 1, 4096, dtype=torch.bfloat16))
    assert calls == []


def test_meta_device_construction(no_build):
    """Reviewer item: device="meta" raised "Expected a cuda device", and under a meta device block the weight landed
    on cuda:0."""
    explicit = rmsnorm.ZeroCenteredRMSNorm(2048, EPS, device="meta", params_dtype=torch.float32)
    with torch.device("meta"):
        default = rmsnorm.ZeroCenteredRMSNorm(4096, EPS)

    assert explicit.weight.device.type == "meta" and explicit.weight.dtype == torch.float32
    assert default.weight.device.type == "meta" and default.weight.shape == (4096,)


@pytest.mark.parametrize("sequence_parallel", [None, False, True])
def test_reset_parameters_initializes_the_weight_after_to_empty(no_build, sequence_parallel):
    """Reviewer item: no reset_parameters(), so after to_empty() the weight stayed uninitialized. to_empty() replaces
    the parameter of a module built on the meta device; reset_parameters() marks the new one from the argument, if
    any."""
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS, device="meta", sequence_parallel=sequence_parallel)
    norm.to_empty(device="cpu")
    with torch.no_grad():
        norm.weight.fill_(float("nan"))

    norm.reset_parameters()
    assert norm.weight.device.type == "cpu" and not norm.weight.any()
    assert getattr(norm.weight, "sequence_parallel", None) is sequence_parallel


def test_reset_parameters_keeps_the_callers_sequence_parallel_mark(no_build):
    """A caller that marks the weight after construction (miles marks its layers' norm parameters with
    config.sequence_parallel) keeps the mark through to_empty() and reset_parameters(): with no sequence_parallel
    argument the module leaves the attribute alone, as TE's norms do, so Megatron still all-reduces the gradient."""
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS, device="cpu")
    norm.weight.sequence_parallel = True

    norm.to_empty(device="cpu")
    norm.reset_parameters()
    assert norm.weight.sequence_parallel is True and _sequence_parallel_params(norm) == [norm.weight]
    assert not norm.weight.any()


def test_sequence_parallel_marks_the_weight_for_the_tensor_parallel_all_reduce(no_build):
    """Reviewer item: the weight lacked ``sequence_parallel``, which TE's ``RMSNorm(sequence_parallel=True)`` sets, so
    Megatron's gradient all-reduce selected no parameter and the ranks' weights diverged under TP + SP."""
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS, device="cpu", sequence_parallel=True)

    assert norm.weight.sequence_parallel is True
    assert _sequence_parallel_params(norm) == [norm.weight]
    norm.to_empty(device="cpu")
    norm.reset_parameters()
    assert _sequence_parallel_params(norm) == [norm.weight]


def test_construction_compiles_nothing(no_build):
    """Reviewer item: building the module compiled the configured variant's programs; it builds nothing now."""
    for device in ("cpu", "meta"):
        for variant in ("det", "fast"):
            cake.configure(cake_rmsnorm_variant=variant)
            rmsnorm.ZeroCenteredRMSNorm(5120, EPS, device=device)


def test_an_unconfigured_call_runs_det(no_build, calls):
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS, device="cpu")
    x = torch.randn(7, 1, 2048).bfloat16()

    y = norm(x)
    assert calls == ["det"]
    assert y.shape == x.shape and torch.equal(y, x)


def test_the_variant_is_read_when_a_call_runs_not_when_the_module_is_built(no_build, calls):
    """Reviewer item: configure(det), construct, configure(fast) still ran the variant read at construction."""
    cake.configure(cake_rmsnorm_variant="det")
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS, device="cpu")
    x = torch.randn(5, 2048).bfloat16()

    norm(x)
    cake.configure(cake_rmsnorm_variant="fast")
    norm(x)
    cake.configure(cake_rmsnorm_variant="det")
    rmsnorm.zero_centered_rmsnorm(x, norm.weight, EPS)
    assert calls == ["det", "fast", "det"]


def test_fast_runs_det_under_torch_deterministic_algorithms_and_logs_it_once(no_build, calls, monkeypatch, caplog):
    monkeypatch.setattr(config, "_warned", set())
    cake.configure(cake_rmsnorm_variant="fast")
    norm = rmsnorm.ZeroCenteredRMSNorm(2048, EPS, device="cpu")
    x = torch.randn(3, 2048).bfloat16()
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            norm(x)
            norm(x)
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    norm(x)

    assert calls == ["det", "det", "fast"]
    messages = [record.getMessage() for record in caplog.records if "cake.rmsnorm" in record.getMessage()]
    assert messages == [
        "miles_ext_kernels.cake.rmsnorm: the fast variant does not serve calls under "
        "torch.use_deterministic_algorithms(True); running det"
    ]


def test_prebuild_takes_the_hidden_size_and_no_call_time_switch():
    impl = importlib.import_module("miles_ext_kernels.cake._impl.rmsnorm")
    parameters = inspect.signature(impl.prebuild).parameters

    assert list(parameters) == ["device", "max_workers", "hidden_size"]
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values())
    assert parameters["hidden_size"].default is inspect.Parameter.empty


@pytest.mark.parametrize("hidden", UNSUPPORTED)
def test_prebuild_refuses_an_unsupported_hidden_size(no_build, hidden):
    with pytest.raises(ValueError, match=f"not {hidden}"):
        cake.prebuild("rmsnorm", hidden_size=hidden)


# A process that sees no CUDA device (CUDA_VISIBLE_DEVICES empty), run as a child so that the check also runs on a
# node with GPUs; any build would raise.
_NO_CUDA = """
import sys

import torch

from miles_ext_kernels import cake
from miles_ext_kernels.cake._runtime import loader


def refuse(*args, **kwargs):
    raise AssertionError("a library was built")


loader.prebuild_libraries = refuse
loader.JitSpec.build = refuse
assert not torch.cuda.is_available()
try:
    cake.prebuild("rmsnorm", hidden_size=2048)
except NotImplementedError as exc:
    print(exc)
else:
    sys.exit("prebuild returned without a CUDA device")
"""


def test_prebuild_without_cuda_names_the_device():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_CUDA],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "need a CUDA device" in result.stdout, result.stdout


def test_release_frees_nothing_and_the_runtime_release_reaches_it():
    import miles_ext_kernels

    assert miles_ext_kernels.release() == 0
    assert cake.release("rmsnorm") == 0
