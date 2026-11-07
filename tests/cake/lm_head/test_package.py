"""The lm_head implementation without a GPU (CPU torch): what ``prebuild`` builds, the host-count rule and its log,
the package entry each variant calls, ``release`` and the errors raised before any device work."""

import json
import logging
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.lm_head import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402

ARCHES = ("sm_100a", "sm_103a", "sm_107a")


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
        # the architecture's other libraries are the candidate-support stages (support_*), which log_probs never
        # launches
        stages = jit.MODULES[record]["stages"]
        support = {jit.library_name(record, stage) for stage in stages if stage.startswith("support_")}
        manifest = json.loads(Path(jit.__file__).with_name("build.json").read_text())
        libraries = {library["name"] for library in manifest["libraries"] if library["arch"] == arch}
        assert support and not support & set(requested[package]), f"{package} {arch}: {sorted(support)}"
        assert libraries == set(requested[package]) | support


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
