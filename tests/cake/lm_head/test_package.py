"""The lm_head implementation without a GPU (CPU torch): what ``prebuild`` builds, the host-count rule and its log
(which the runtime's CUDA 13.4 floor leaves unreached), the package entry each variant calls, ``release`` and the
errors raised before any device work."""

import json
import logging
import sys
import types
from importlib import import_module
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.lm_head import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402

ARCHES = ("sm_100a", "sm_103a", "sm_107a")


def _unlaunched_stages(variant: str) -> set[str]:
    """The stages besides the candidate-support ones (``support_*``) that the package's programs register and
    ``log_probs`` never launches, as the package's backend names them: ``fast``'s accumulating dX slab sum (the
    candidate-support path's dense dX) and the loss entry's one-pass cross-entropy programs; none for ``det``."""
    if variant == "det":
        return set()
    backend = import_module(".cake_backend", ops._package(variant).__name__)
    one_pass = {stage for forms in backend.ONE_PASS_FWD_STAGES.values() for stage in forms}
    return {backend.DX_SLAB_SUM_ACC_STAGE, backend.ONE_PASS_COMBINE_STAGE, backend.ONE_PASS_DW_STAGE, *one_pass}


def _clear_toolkit_caches():
    ops._check_toolkit.cache_clear()
    for variant in ops.PACKAGES:
        ops._jit(variant).toolchain_runs_hidden_count.cache_clear()


@pytest.fixture
def toolkit(monkeypatch):
    """Pretend the local toolkit is CUDA ``major.minor`` (``toolkit(13, 0)``), with no nvcc call."""

    def use(major: int, minor: int) -> None:
        version = loader.CudaVersion(major, minor, f"{major}.{minor}.0")
        monkeypatch.setattr(loader, "get_cuda_version", lambda: version)
        monkeypatch.setattr(loader, "check_toolkit", lambda arch, what, nvcc=None: version)
        _clear_toolkit_caches()

    yield use
    _clear_toolkit_caches()


@pytest.mark.parametrize("arch", ARCHES)
def test_prebuild_builds_every_log_prob_stage_with_and_without_a_temperature(monkeypatch, toolkit, arch):
    toolkit(13, 4)
    requested: dict[str, list[str]] = {}
    monkeypatch.setattr(loader, "device_arch", lambda device=None: arch)
    monkeypatch.setattr(
        loader,
        "prebuild_libraries",
        lambda package, names, max_workers=None: requested.setdefault(package, list(names)),
    )

    cake.prebuild("lm_head", max_workers=3)

    assert set(requested) == set(ops.PACKAGES.values())
    for variant, package in ops.PACKAGES.items():
        jit = ops._jit(variant)
        record = jit.select_module(arch)
        plain = {jit.library_name(record, stage) for stage in jit.registered_stages(record)}
        tau_stages = set(jit.registered_stages_with_temperature(record)) - set(jit.registered_stages(record))
        tau = {jit.library_name(record, stage) for stage in tau_stages}
        # the temperature forms of the logits GEMMs and of row_finalize: a temperature call compiles nothing more
        assert {stage.endswith("_tau") for stage in tau_stages} == {True}, tau_stages
        assert "row_finalize_tau" in tau_stages
        assert tau - plain, f"{package} {arch}: no temperature library"
        assert set(requested[package]) == plain | tau
        if variant == "fast":
            # the ordered dX: the slab sums after each K-sliced dX GEMM are log-probability stages
            backend = import_module(".cake_backend", ops._package(variant).__name__)
            assert set(backend.DX_SLAB_SUM_STAGES) <= set(jit.registered_stages(record)), f"{package} {arch}"
        # the architecture's other libraries are the candidate-support stages (support_*) and the stages of
        # _unlaunched_stages, which log_probs never launches
        stages = set(jit.MODULES[record]["stages"])
        unlaunched = stages - set(jit.registered_stages_with_temperature(record))
        support = {stage for stage in unlaunched if stage.startswith("support_")}
        assert support, f"{package} {arch}: no candidate-support stage"
        assert unlaunched - support == _unlaunched_stages(variant), f"{package} {arch}: {sorted(unlaunched)}"
        others = {jit.library_name(record, stage) for stage in unlaunched}
        manifest = json.loads(Path(jit.__file__).with_name("build.json").read_text())
        libraries = {library["name"] for library in manifest["libraries"] if library["arch"] == arch}
        assert not others & set(requested[package]), f"{package} {arch}: {sorted(others & set(requested[package]))}"
        assert libraries == set(requested[package]) | others


@pytest.mark.parametrize(
    ("version", "arch", "host_count"),
    [
        ((13, 0), "sm_103a", True),
        ((13, 0), "sm_100a", False),
        ((12, 9), "sm_103a", False),
        ((13, 1), "sm_103a", False),
        ((13, 4), "sm_107a", False),
    ],
)
def test_only_sm103a_programs_built_by_cuda_13_0_take_the_host_count_path(toolkit, version, arch, host_count):
    toolkit(*version)

    assert ops.host_count_path(arch) is host_count
    # the packages share the rule
    assert {not ops._jit(variant).toolchain_runs_hidden_count(arch) for variant in ops.PACKAGES} == {host_count}


@pytest.mark.parametrize(("version", "logged"), [((13, 0), 1), ((13, 1), 0)])
def test_the_host_count_path_is_logged_once_per_architecture(toolkit, caplog, version, logged):
    toolkit(*version)

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        for _ in range(3):
            ops._check_toolkit("sm_103a")
    messages = [record.getMessage() for record in caplog.records if record.name.startswith("miles_ext_kernels")]

    assert len(messages) == logged
    if logged:
        assert "sm_103a programs built by CUDA 13.0" in messages[0] and "also with num_valid_rows" in messages[0]


@pytest.mark.parametrize("arch", ARCHES)
def test_the_runtime_refuses_a_cuda_13_0_toolkit_before_the_host_count_path(monkeypatch, caplog, arch):
    """CUDA 13.4 is the runtime's floor on every architecture, so no call reaches the packages' CUDA 13.0 path."""
    cuda_13_0 = loader.CudaVersion(13, 0, "13.0.88")
    monkeypatch.setattr(loader, "_toolkit", lambda nvcc=None: ("/usr/local/cuda/bin/nvcc", cuda_13_0))
    _clear_toolkit_caches()

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        with pytest.raises(RuntimeError, match=rf"for {arch} needs CUDA 13\.4 or newer"):
            ops._check_toolkit(arch)
    assert not [record for record in caplog.records if record.name.startswith("miles_ext_kernels")]
    _clear_toolkit_caches()


def test_the_package_modules_take_the_toolkit_from_the_loader():
    """The program registries read the toolkit through this loader (``ops`` imports them with it installed)."""
    for variant, package in ops.PACKAGES.items():
        jit = ops._jit(variant)

        assert jit.__name__ == f"{ops._VENDOR}.{package}.cake_jit"
        assert jit.cake_build._jit_toolchain.__module__ == loader.__name__


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_log_probs_calls_the_mode_word_entry_of_its_variants_package(monkeypatch, variant):
    """Without a device: ``log_probs`` calls the log-probability entry of its variant's package by its mode-word name,
    without a ``deterministic`` keyword. The stand-ins are the package's own public functions (``cake_backend_opt_in``)
    replaced by recorders, so a name the package does not define, or a keyword its entries do not take, fails."""
    module = ops._entry(variant)
    expected = f"cake_lm_head_{'det' if variant == 'det' else 'nondet'}_train_logprob_temperature"
    assert callable(getattr(module, expected, None)), expected
    calls = []

    def stand_in(name):
        def entry(X, W, labels, *, chunk_size, num_valid_rows, temperature):
            calls.append((name, chunk_size, num_valid_rows, temperature))
            return torch.zeros(labels.shape, dtype=torch.float32)

        return entry

    public = [name for name in dir(module) if not name.startswith("_") and callable(getattr(module, name))]
    monkeypatch.setattr(ops, "_entry", lambda v: types.SimpleNamespace(**{name: stand_in(name) for name in public}))
    monkeypatch.setattr(ops, "_arch", lambda device: "sm_100a")
    cake.configure(cake_lm_head_variant=variant)
    X = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
    W = torch.zeros((256, 512), dtype=torch.bfloat16)
    labels = torch.zeros((3,), dtype=torch.int64)

    logp = ops.log_probs(X, W, labels, temperature=0.5, num_valid_rows=3)
    assert calls == [(expected, config.option("lm_head", "cake_lm_head_chunk_size"), 3, 0.5)]
    assert logp.shape == (3,)


def test_a_call_off_cuda_is_rejected_before_any_device_work():
    X = torch.zeros((8, 512), dtype=torch.bfloat16)
    W = torch.zeros((256, 512), dtype=torch.bfloat16)
    labels = torch.zeros((8,), dtype=torch.int64)

    with pytest.raises(ValueError, match="run on a CUDA device, X is on cpu"):
        ops.log_probs(X, W, labels)


def test_release_clears_the_binding_caches_of_the_imported_package_modules(monkeypatch):
    cleared = []

    class Cache:
        def __init__(self, name):
            self.name = name

        def clear(self):
            cleared.append(self.name)

    for package in ops.PACKAGES.values():
        for module in ops._BINDING_MODULES:
            name = f"{ops._VENDOR}.{package}.{module}"
            monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(BINDING_CACHE=Cache(name)))

    assert cake.release("lm_head") == 0
    assert sorted(cleared) == sorted(
        f"{ops._VENDOR}.{package}.{module}" for package in ops.PACKAGES.values() for module in ops._BINDING_MODULES
    )


def test_release_imports_no_package_module(monkeypatch):
    names = [
        f"{ops._VENDOR}.{package}.{module}" for package in ops.PACKAGES.values() for module in ops._BINDING_MODULES
    ]
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    ops.release()
    assert [name for name in names if name in sys.modules] == []
