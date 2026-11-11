"""Which vendored package each KDA entry point runs, what the seam hands it, and the op's prebuild and release (CPU:
torch and tvm-ffi, no GPU).

The packages are replaced by recorders, except in the tests of the entry signatures, of the ``fast`` package's toolkit
hook, of the plans and of the imports, which load the vendored packages themselves (host code only, no launch).
Skipped without torch (the torch-free CPU suite); with torch, a missing tvm-ffi is an error.
"""

import inspect
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import tvm_ffi  # noqa: E402, F401

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.kda import kcp, ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = {"use_gate_in_kernel": True, "A_log": "A_log", "dt_bias": "dt_bias", "lower_bound": -5.0}
# every library of the four packages, per architecture (README, Prebuild)
LIBRARIES = {"sm_100a": 216, "sm_103a": 218, "sm_107a": 216}
# the entry points each variant's call runs: single rank (node, forward, backward), then context-parallel
DET_ENTRIES = ["cake_kda_det", "cake_kda_det_train_forward", "cake_kda_det_train_backward"]
DET_KCP_ENTRIES = ["cake_kda_det_kcp", "cake_kda_det_train_kcp_forward", "cake_kda_det_train_kcp_backward"]
FAST_ENTRIES = ["cake_kda_nondet", "cake_kda_nondet_train_forward", "cake_kda_nondet_train_backward"]
FAST_KCP_ENTRIES = ["cake_kda_nondet_kcp", "cake_kda_nondet_train_kcp_forward", "cake_kda_nondet_train_kcp_backward"]


class _FastSaved:
    """Stands for the saved set of a ``fast`` forward (``CakeKDANondetChunkSaved`` / ``CakeKDANondetKcpSaved`` of the
    recorders)."""


class _Recorder:
    """A stand-in package: every other attribute is an entry point that records its name, operands and keywords; a
    backward returns five BF16 gradients, any other entry its name."""

    def __init__(self, **attributes):
        self.calls = []
        vars(self).update(attributes)

    def __getattr__(self, entry):
        def call(*args, **kwargs):
            self.calls.append((entry, args, kwargs))
            if entry.endswith("_backward"):
                return tuple(torch.zeros(1, 5, 4, dtype=torch.bfloat16) for _ in range(5))
            return entry

        return call

    def entries(self):
        return [entry for entry, _, _ in self.calls]


@pytest.fixture
def packages(monkeypatch):
    """Recorders for the four packages and a fresh once-per-process det-run log; the conftest restores the
    configuration after."""
    det, det_kcp, det_plan = _Recorder(), _Recorder(), _Recorder()
    fast, fast_kcp = _Recorder(CakeKDANondetChunkSaved=_FastSaved), _Recorder(CakeKDANondetKcpSaved=_FastSaved)
    monkeypatch.setattr(ops, "det_backend", det)
    monkeypatch.setattr(ops, "fast_backend", fast)
    monkeypatch.setattr(kcp, "_det_kcp", lambda: det_kcp)
    monkeypatch.setattr(kcp, "_det_plan", lambda: det_plan)
    monkeypatch.setattr(kcp, "_fast_kcp", lambda: fast_kcp)
    monkeypatch.setattr(config, "_warned", set())
    return {"det": det, "fast": fast, "det_kcp": det_kcp, "det_plan": det_plan, "fast_kcp": fast_kcp}


def _operands(heads: int = 4):
    x = torch.zeros(1, 5, heads, 128, dtype=torch.bfloat16)
    return x, x, x, x, x[..., 0]


def _single_rank_calls(l2n: bool, saved=None):
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
    ops.chunk_kda(*_operands(), cu_seqlens, use_qk_l2norm_in_kernel=l2n, **RAW)
    ops.chunk_kda_fwd(*_operands(), cu_seqlens, use_qk_l2norm_in_kernel=l2n, **RAW)
    ops.chunk_kda_bwd(*_operands(), cu_seqlens, "dout", saved=saved, use_qk_l2norm_in_kernel=l2n, **RAW)


def _kcp_calls(l2n: bool, saved=None):
    kcp.chunk_kda_kcp(*_operands(), "ctx", use_qk_l2norm_in_kernel=l2n, save_mode="light", **RAW)
    kcp.chunk_kda_fwd_kcp(*_operands(), "ctx", use_qk_l2norm_in_kernel=l2n, **RAW)
    kcp.chunk_kda_bwd_kcp(*_operands(), "ctx", "dout", saved=saved, use_qk_l2norm_in_kernel=l2n, **RAW)


@pytest.mark.parametrize("l2n", [False, True])
def test_det_runs_every_call_on_the_deterministic_packages(packages, l2n):
    _single_rank_calls(l2n)
    _kcp_calls(l2n)

    assert packages["det"].entries() == DET_ENTRIES
    assert packages["det_kcp"].entries() == DET_KCP_ENTRIES
    for _, _, kwargs in packages["det"].calls + packages["det_kcp"].calls:
        assert kwargs["use_qk_l2norm_in_kernel"] is l2n and kwargs["lower_bound"] == -5.0
        assert "deterministic" not in kwargs
    assert packages["det"].calls[0][2]["scale"] == ops.DEFAULT_SCALE
    assert packages["det_kcp"].calls[0][2]["save_mode"] == "light"  # the KCP options are forwarded
    assert packages["fast"].calls == packages["fast_kcp"].calls == []


@pytest.mark.parametrize("l2n", [False, True])
def test_fast_runs_every_call_on_the_non_deterministic_packages(packages, caplog, l2n):
    """In every gate and L2 norm mode: the in-kernel q / k L2 norm (miles' call) included, a ``fast`` call runs the
    non-deterministic packages, with no ``det`` run and no log line, however often it is made."""
    cake.configure(cake_kda_variant="fast")
    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        for _ in range(2):
            _single_rank_calls(l2n)
            _kcp_calls(l2n)

    assert packages["fast"].entries() == FAST_ENTRIES * 2
    assert packages["fast_kcp"].entries() == FAST_KCP_ENTRIES * 2
    for _, _, kwargs in packages["fast"].calls + packages["fast_kcp"].calls:
        assert kwargs["use_qk_l2norm_in_kernel"] is l2n and "deterministic" not in kwargs
        assert kwargs["use_gate_in_kernel"] is True and kwargs["A_log"] == "A_log"
    assert packages["fast"].calls[0][2]["scale"] == ops.DEFAULT_SCALE
    assert packages["fast_kcp"].calls[0][2]["save_mode"] == "light"
    assert packages["det"].calls == packages["det_kcp"].calls == []
    assert caplog.records == [] and config._warned == set()


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_backward_runs_the_package_that_wrote_its_saved_set(packages, variant):
    """A saved set is read by its own package only, so the configured variant does not choose the backward of one,
    in either L2 norm mode."""
    cake.configure(cake_kda_variant=variant)
    for l2n in (False, True):
        _single_rank_calls(l2n, saved=_FastSaved())
        _kcp_calls(l2n, saved=_FastSaved())
        _single_rank_calls(l2n, saved=object())  # a set of the det package
        _kcp_calls(l2n, saved=object())

    calls = [call for recorder in packages.values() for call in recorder.calls]
    bwd = [(entry, kwargs["saved"]) for entry, _, kwargs in calls if entry.endswith("_backward")]
    # per package, one backward per L2 norm mode (the recorders list their own calls in order)
    det_bwd = [("cake_kda_det_train_backward", object)] * 2 + [("cake_kda_det_train_kcp_backward", object)] * 2
    fast_bwd = [("cake_kda_nondet_train_backward", _FastSaved)] * 2
    fast_bwd += [("cake_kda_nondet_train_kcp_backward", _FastSaved)] * 2
    assert [(entry, type(saved)) for entry, saved in bwd if entry.startswith("cake_kda_det")] == det_bwd
    assert [(entry, type(saved)) for entry, saved in bwd if not entry.startswith("cake_kda_det")] == fast_bwd


# (det entry, fast entry) of the single-rank and the context-parallel packages
ENTRY_PAIRS = list(zip(DET_ENTRIES, FAST_ENTRIES))
KCP_ENTRY_PAIRS = list(zip(DET_KCP_ENTRIES, FAST_KCP_ENTRIES))


def test_the_fast_entries_take_the_arguments_of_the_det_entries():
    """What routing every mode by the configured variant rests on, read from the vendored packages: each ``fast`` entry
    takes the positional parameters of its ``det`` counterpart in the same order and every keyword of it (the in-kernel
    L2 norm's included) with the same default; a parameter only it takes has a default. The saved-set classes the
    backwards are routed by exist."""
    pairs = [(getattr(ops.det_backend, det), getattr(ops.fast_backend, fast)) for det, fast in ENTRY_PAIRS]
    pairs += [(getattr(kcp._det_kcp(), det), getattr(kcp._fast_kcp(), fast)) for det, fast in KCP_ENTRY_PAIRS]
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    for det, fast in pairs:
        det_parameters = inspect.signature(det).parameters
        fast_parameters = inspect.signature(fast).parameters
        assert "use_qk_l2norm_in_kernel" in det_parameters, det.__name__
        det_positional = [n for n, p in det_parameters.items() if p.kind in positional]
        assert [n for n, p in fast_parameters.items() if p.kind in positional] == det_positional, fast.__name__
        for name, parameter in det_parameters.items():
            assert fast_parameters[name].kind == parameter.kind, (fast.__name__, name)
            assert fast_parameters[name].default == parameter.default, (fast.__name__, name)
        for name, parameter in fast_parameters.items():
            if name not in det_parameters:
                assert parameter.default is not inspect.Parameter.empty, (fast.__name__, name)
    assert isinstance(ops.fast_backend.CakeKDANondetChunkSaved, type)
    assert isinstance(kcp._fast_kcp().CakeKDANondetKcpSaved, type)


def test_the_seam_hands_the_packages_bf16_beta_and_a_dense_v(packages):
    """miles hands flash-linear-attention an FP32 ``beta`` and ``v`` as a column slice of one packed projection; the
    packages take a BF16 ``beta`` and dense ``v`` rows, and ``dbeta`` returns in the caller's dtype."""
    packed = torch.randn(1, 5, 4, 3 * 128, dtype=torch.bfloat16)
    q, k, v = packed[..., :128], packed[..., 128:256], packed[..., 256:]
    g = torch.zeros(1, 5, 4, 128)
    beta = torch.rand(1, 5, 4, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
    ops.chunk_kda(q, k, v, g, beta, cu_seqlens, use_qk_l2norm_in_kernel=True, **RAW)
    grads = ops.chunk_kda_bwd(q, k, v, g, beta, cu_seqlens, "dout", use_qk_l2norm_in_kernel=True, **RAW)
    kcp_grads = kcp.chunk_kda_bwd_kcp(q, k, v, g, beta, "ctx", "dout", use_qk_l2norm_in_kernel=True, **RAW)

    for _, args, _ in packages["det"].calls + packages["det_kcp"].calls:
        got_q, got_k, got_v, got_g, got_beta = args[:5]
        assert got_q is q and got_k is k and got_g is g  # the packages check these
        assert got_v.is_contiguous() and torch.equal(got_v, v)
        assert got_beta.dtype == torch.bfloat16 and torch.equal(got_beta, beta.to(torch.bfloat16))
    assert grads[3].dtype == kcp_grads[3].dtype == torch.float32
    assert [grad.dtype for grad in grads[:3] + grads[4:]] == [torch.bfloat16] * 4


def test_the_seam_passes_bf16_dense_operands_through_unchanged(packages):
    q, k, v, g, beta = _operands()
    beta = beta.contiguous()
    ops.chunk_kda(q, k, v, g, beta, torch.tensor([0, 5], dtype=torch.int32))

    [(_, args, _)] = packages["det"].calls
    assert args[2] is v and args[4] is beta


def test_the_seam_is_differentiable(packages, monkeypatch):
    """The cast and the copy are autograd nodes: the gradients reach the FP32 ``beta`` and the packed projection."""

    def forward(q, k, v, g, beta, cu_seqlens, **kwargs):
        return v.float() * beta.float().unsqueeze(-1)

    monkeypatch.setattr(packages["det"], "cake_kda_det", forward, raising=False)
    packed = torch.randn(1, 5, 4, 3 * 128, dtype=torch.bfloat16, requires_grad=True)
    beta = torch.rand(1, 5, 4, dtype=torch.float32, requires_grad=True)
    q, k, v = packed[..., :128], packed[..., 128:256], packed[..., 256:]
    out = ops.chunk_kda(q, k, v, torch.zeros(1, 5, 4, 128), beta, torch.tensor([0, 5], dtype=torch.int32))
    out.sum().backward()

    assert beta.grad.dtype == torch.float32
    torch.testing.assert_close(beta.grad, v.detach().float().sum(-1), rtol=1e-2, atol=1e-2)  # rounded to BF16 once
    assert torch.equal(packed.grad[..., :256], torch.zeros(1, 5, 4, 256, dtype=torch.bfloat16))
    assert torch.equal(packed.grad[..., 256:], beta.detach().to(torch.bfloat16).unsqueeze(-1).expand(1, 5, 4, 128))


@pytest.mark.parametrize("heads", [0, 65, 96])
def test_the_head_count_is_checked_before_any_package_runs(packages, heads):
    cake.configure(cake_kda_variant="fast")
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
    for call in (
        lambda: ops.chunk_kda(*_operands(heads), cu_seqlens),
        lambda: ops.chunk_kda_fwd(*_operands(heads), cu_seqlens),
        lambda: ops.chunk_kda_bwd(*_operands(heads), cu_seqlens, "dout"),
        lambda: kcp.chunk_kda_kcp(*_operands(heads), "ctx"),
        lambda: kcp.chunk_kda_fwd_kcp(*_operands(heads), "ctx"),
        lambda: kcp.chunk_kda_bwd_kcp(*_operands(heads), "ctx", "dout"),
    ):
        with pytest.raises(ValueError, match="serve 1 to 64 heads per call"):
            call()
    assert all(recorder.calls == [] for recorder in packages.values())


def test_the_partition_context_and_parameter_reduction_are_the_deterministic_packages_for_either_variant(packages):
    for variant in ("det", "fast"):
        cake.configure(cake_kda_variant=variant)
        kcp.partition_kcp([0, 5, 9], 2, "zigzag")
        kcp.build_context_kcp("part", "group", 1, "cuda:1", heads=4)
        kcp.reduce_gate_param_grads_kcp("ctx", "dA_log", "ddt_bias")

    plan_entries = ["cake_kda_det_train_kcp_partition", "cake_kda_det_train_kcp_build_context"]
    assert packages["det_plan"].entries() == plan_entries * 2
    assert packages["det_plan"].calls[0][2] == {"cut_points": None}
    assert packages["det_plan"].calls[1][2] == {"sm_count": None, "heads": 4, "stream": None}
    assert packages["det_kcp"].entries() == ["cake_kda_det_train_kcp_reduce_gate_param_grads"] * 2
    assert packages["fast_kcp"].calls == []


def test_the_fast_package_reads_its_toolkit_from_the_runtime_loader(monkeypatch):
    """The checkpoint hand-off's row records key on the toolkit that builds the libraries: the runtime loader's nvcc,
    not FlashInfer's JIT (which this repository does not have)."""
    monkeypatch.setattr(loader, "get_cuda_version", lambda: loader.CudaVersion(13, 4, "13.4.92"))
    backend = ops.fast_backend

    assert backend._jit_nvcc_version is ops._nvcc_version
    assert backend._jit_nvcc_version() == "13.4.92"
    plan = backend.plan(4096, 3, 16, 148, "sm_100a")
    records = sorted({name for name in plan.records["fwd"].values() if name is not None})
    assert records
    # no aot/ library is vendored: the one build of the records is the loader's, so its toolkit is the hook's
    assert backend._one_build_toolkit(records, "sm_100a") == "13.4.92"
    assert "flashinfer" not in sys.modules


# the problems the plan search below covers: (tokens, documents, heads), on each architecture's SM count
PLAN_SHAPES = [(s, n, h) for s in (1024, 8192, 65536) for n in (1, 4, 64) for h in (1, 4, 12, 16, 32, 64)]
SM_COUNTS = {"sm_100a": 148, "sm_103a": 148, "sm_107a": 216}


def _selected(backend, gate: str, arch: str, l2n: str = "off") -> set[str]:
    """Every record a plan of the :data:`PLAN_SHAPES` in the gate mode ``gate`` and the L2 norm mode ``l2n`` selects on
    ``arch``, over every path."""
    selected = set()
    for tokens, docs, heads in PLAN_SHAPES:
        plan = backend.plan(tokens, docs, heads, SM_COUNTS[arch], arch, gate=gate, l2n=l2n)
        selected |= {name for records in plan.records.values() for name in records.values() if name is not None}
    return selected


def _programs(backend, gate: str, arch: str, l2n: str) -> list[tuple]:
    """Per plan of the :data:`PLAN_SHAPES`, path and stage: the program (the library name without the package's prefix:
    the same name in both packages for a copy of one program) and its launch grids."""
    jit = backend.cake_jit
    prefix = next(iter(jit.MODULES.values()))["library"].rsplit("_", 1)[0] + "_"
    programs = []
    for tokens, docs, heads in PLAN_SHAPES:
        plan = backend.plan(tokens, docs, heads, SM_COUNTS[arch], arch, gate=gate, l2n=l2n)
        for path, records in plan.records.items():
            for stage, record in records.items():
                if record is not None:
                    program = jit.library_name(record, arch).removeprefix(prefix)
                    programs.append((tokens, docs, heads, path, stage, program, plan.grids[record]))
    return sorted(programs)


@pytest.mark.parametrize("arch", sorted(SM_COUNTS))
def test_fast_plans_other_programs_than_det_with_the_activated_gate_alone(arch):
    """``fast`` is not ``det`` under another name: with the activated gate and without the in-kernel L2 norm its
    backward runs the ``S1`` programs (the ``q * dq`` term of ``dg`` in GLM intra) on some rows of every architecture,
    which ``det`` does not ship. With the raw gate, and with the in-kernel L2 norm in either gate mode, every path of
    every plan launches ``det``'s programs (copies in the ``fast`` package) on ``det``'s grids."""
    fast_jit = ops.fast_backend.cake_jit
    det_jit = ops.det_backend.cake_jit
    fast_g = _selected(ops.fast_backend, "g", arch)
    det_g = _selected(ops.det_backend, "g", arch)

    assert fast_g & set(fast_jit.S1_RECORDS)
    assert not det_g & set(fast_jit.S1_RECORDS)
    assert not set(det_jit.MODULES) & set(fast_jit.S1_RECORDS)
    for gate, l2n in (("raw", "off"), ("g", "fused"), ("raw", "fused")):
        assert _programs(ops.fast_backend, gate, arch, l2n) == _programs(ops.det_backend, gate, arch, l2n), (gate, l2n)


# --- prebuild and release ---


@pytest.fixture
def builds(monkeypatch):
    """The loader's device, toolkit and build calls replaced by recorders: no device, nvcc or compile."""
    calls = []
    monkeypatch.setattr(loader, "device_arch", lambda device=None: calls.append(("arch", device)) or "sm_100a")
    monkeypatch.setattr(loader, "check_toolkit", lambda arch, what, nvcc=None: calls.append(("toolkit", arch)))

    def prebuild_libraries(package, names, max_workers=None):
        calls.append(("build", package.__name__.split(".")[-2], tuple(names), max_workers))
        return [Path(name) for name in names]

    monkeypatch.setattr(loader, "prebuild_libraries", prebuild_libraries)
    return calls


@pytest.mark.parametrize("arch", sorted(LIBRARIES))
def test_the_op_builds_every_library_of_its_packages_for_the_architecture(arch):
    libraries = ops.libraries(arch)

    assert list(libraries) == list(ops.PACKAGES)
    assert sum(map(len, libraries.values())) == LIBRARIES[arch]
    assert all(names == sorted(set(names)) for names in libraries.values())


@pytest.mark.parametrize("num_heads", [1, 12, 64])
def test_prebuild_builds_the_same_libraries_for_every_head_count(builds, num_heads):
    paths = cake.prebuild("kda", num_heads=num_heads, max_workers=3)

    expected = ops.libraries("sm_100a")
    assert builds[:2] == [("arch", None), ("toolkit", "sm_100a")]
    assert builds[2:] == [("build", package, tuple(names), 3) for package, names in expected.items()]
    assert paths == [Path(name) for names in expected.values() for name in names]
    assert len(paths) == LIBRARIES["sm_100a"]


@pytest.mark.parametrize("num_heads", [0, 65, 96, -1, True, 12.0, "12", None])
def test_prebuild_checks_the_head_count_before_anything(builds, num_heads):
    with pytest.raises(ValueError, match="serve 1 to 64 heads per call"):
        cake.prebuild("kda", num_heads=num_heads)
    assert builds == []


def test_prebuild_takes_the_head_count_alone(builds):
    with pytest.raises(TypeError):
        cake.prebuild("kda")
    with pytest.raises(TypeError):
        cake.prebuild("kda", num_heads=12, head_dim=128)
    with pytest.raises(TypeError):
        cake.prebuild("kda", num_heads=12, token_counts=[4096])
    assert builds == []


def test_release_drops_the_entry_pools_and_the_graphs_of_a_loaded_context_parallel_package(monkeypatch):
    cleared = []
    stubs = {}
    for package in ("cake_kda_det_train_kcp", "cake_kda_nondet_train_kcp"):
        stub = stubs[package] = type(sys)(f"{ops._VENDOR}.{package}.cake_kcp")
        stub.graph_cache_clear = lambda package=package: cleared.append(package) or 0
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    # the workspaces of the calls made without one: the det package's pool, the fast package's clear function
    det_kcp = stubs["cake_kda_det_train_kcp"]
    det_kcp._default_pool, det_kcp._default_lock = {("stand-in",): []}, threading.Lock()
    stubs["cake_kda_nondet_train_kcp"].default_workspace_cache_clear = lambda: cleared.append("fast defaults") or 0
    for backend in (ops.det_backend, ops.fast_backend):
        monkeypatch.setitem(backend._entry_pool, ("stand-in",), [])

    cake.release("kda")

    assert len(ops.det_backend._entry_pool) == len(ops.fast_backend._entry_pool) == 0
    assert det_kcp._default_pool == {}
    assert cleared == ["cake_kda_det_train_kcp", "cake_kda_nondet_train_kcp", "fast defaults"]


_IMPORTS = """
import importlib
import sys

sys.modules["flashinfer"] = None  # any import of flashinfer raises ImportError

from miles_ext_kernels.cake import kda

VENDOR = "miles_ext_kernels.cake._vendor."
PACKAGES = {"cake_kda_det_train", "cake_kda_det_train_kcp", "cake_kda_nondet_train", "cake_kda_nondet_train_kcp"}
assert not [n for n in sys.modules if n.startswith(VENDOR)]  # the op module loads no package
assert kda.chunk_kda  # the first entry point loads the implementation, which installs the loader on every package
loaded = {n[len(VENDOR):].split(".")[0] for n in sys.modules if n.startswith(VENDOR)}
assert loaded == PACKAGES, loaded
# the context-parallel entry modules load on the first context-parallel call
assert not [n for n in sys.modules if n.startswith(VENDOR) and n.endswith(".cake_kcp")]

from miles_ext_kernels.cake._impl.kda import kcp

det_kcp, det_plan, fast_kcp = kcp._det_kcp(), kcp._det_plan(), kcp._fast_kcp()
assert det_kcp.cake_kda_det_kcp and det_plan.cake_kda_det_train_kcp_partition and fast_kcp.cake_kda_nondet_kcp
# each context-parallel package imports its single-rank sibling by name from its own parent: the vendored sibling
for package, sibling in (
    ("cake_kda_det_train_kcp", "cake_kda_det_train"),
    ("cake_kda_nondet_train_kcp", "cake_kda_nondet_train"),
):
    single_rank = importlib.import_module(VENDOR + package + "._single_rank")
    assert single_rank.NAME == VENDOR + sibling, single_rank.NAME
    assert single_rank.module("cake_backend") is sys.modules[VENDOR + sibling + ".cake_backend"]
assert not [n for n in sys.modules if n.startswith(VENDOR + "cake.")]
for package in PACKAGES:
    assert VENDOR + package + ".cake_build" in sys.modules, package
assert "flashinfer" not in [n.split(".")[0] for n, m in sys.modules.items() if m is not None]
"""


def test_the_packages_load_without_flashinfer_and_only_when_first_used():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _IMPORTS],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
