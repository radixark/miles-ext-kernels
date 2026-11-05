"""prebuild and release reach the op implementations (torch-free), shown on the synthetic op ``synth``."""

import dataclasses
import sys
import types

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake._runtime import loader, registry, workspace

from . import synthetic


def test_op_spec_returns_the_op_declaration(synth):
    assert cake.op_spec("synth") is synthetic.OP
    with pytest.raises(KeyError, match="no cake op 'nope'"):
        cake.op_spec("nope")


def test_op_spec_reads_the_op_module_in_sys_modules_at_every_call(synth, monkeypatch):
    replaced = synthetic.install(monkeypatch, dataclasses.replace(synthetic.OP, variants=("fast",)))

    assert cake.op_spec("synth") is replaced.OP and cake.variant("synth") == "fast"
    monkeypatch.delattr(replaced, "OP")  # a module that has no OP yet (still being imported) is imported
    monkeypatch.setattr(registry, "import_module", lambda name: types.SimpleNamespace(OP=("imported", name)))
    assert cake.op_spec("synth") == ("imported", synthetic.MODULE)
    monkeypatch.delitem(sys.modules, synthetic.MODULE)  # and so is one that is not imported
    assert cake.op_spec("synth") == ("imported", synthetic.MODULE)


def test_the_registered_ops_are_unique():
    assert len(set(registry.OPS)) == len(registry.OPS)


def test_prebuild_passes_the_device_the_workers_and_the_model_parameters(synth):
    assert cake.prebuild("synth", width=8) == []
    cake.prebuild("synth", device="cuda:1", max_workers=4)

    assert synthetic.CALLS == [
        ("prebuild", {"device": None, "max_workers": None, "width": 8}),
        ("prebuild", {"device": "cuda:1", "max_workers": 4, "width": None}),
    ]


def test_prebuild_raises_for_unsupported_parameters_and_unknown_ops(synth):
    with pytest.raises(ValueError, match="synth serves width <= 64"):
        cake.prebuild("synth", width=65)
    with pytest.raises(TypeError):
        cake.prebuild("synth", num_heads=8)
    with pytest.raises(KeyError, match="no cake op 'nope'"):
        cake.prebuild("nope")


TOOLKIT = ("/usr/local/cuda/bin/nvcc", loader.CudaVersion(13, 0, "13.0.88"))


@pytest.fixture
def counted(synth, monkeypatch, tmp_path):
    """The model parameters of each prebuild that reached ``synth``.

    Devices 0 (the current one) and 1 are sm_100a, device 2 sm_103a; the toolkit is ``TOOLKIT``.
    """
    calls = []

    def prebuild(*, device=None, max_workers=None, **model_params):
        calls.append(model_params)
        if model_params.get("hidden", 0) > 64:
            raise ValueError("synth serves hidden <= 64")
        return [tmp_path / f"lib{len(calls)}.so"]

    monkeypatch.setattr(sys.modules[synthetic.IMPL], "prebuild", prebuild)
    monkeypatch.setattr(registry, "_device", lambda device: (device or 0, "sm_103a" if device == 2 else "sm_100a"))
    monkeypatch.setattr(loader, "_toolkit", lambda nvcc=None: TOOLKIT)
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path / "cache"))
    return calls


def test_prebuild_remembers_the_libraries_of_one_configuration(counted, tmp_path):
    first = cake.prebuild("synth", hidden=8, layouts=["nt", "tn"], experts=iter([(8, 4)]))
    first.append(tmp_path / "the caller's own list")
    again = cake.prebuild("synth", device=0, max_workers=4, layouts=("nt", "tn"), experts=[(8, 4)], hidden=8)

    assert again == [tmp_path / "lib1.so"]
    assert counted == [{"hidden": 8, "layouts": ["nt", "tn"], "experts": ((8, 4),)}]


def test_prebuild_runs_again_for_other_parameters_options_architectures_and_cache_directories(
    counted, monkeypatch, tmp_path
):
    cake.prebuild("synth", hidden=8)
    cake.prebuild("synth", hidden=16)
    cake.prebuild("synth", hidden=8.0)
    cake.prebuild("synth", hidden=True)
    cake.prebuild("synth", device=2, hidden=8)
    cake.configure(cake_synth_scale=2.0)
    cake.prebuild("synth", hidden=8)
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path / "other"))
    cake.prebuild("synth", hidden=8)
    cake.prebuild("synth", hidden=8)

    assert [params["hidden"] for params in counted] == [8, 16, 8.0, True, 8, 8, 8]
    assert [type(params["hidden"]) for params in counted[1:4]] == [int, float, bool]


def test_prebuild_runs_again_on_another_device_of_the_same_architecture(counted):
    cake.prebuild("synth", hidden=8)
    cake.prebuild("synth", device=1, hidden=8)  # the op warms what it keeps per device on device 1 too
    cake.prebuild("synth", device=1, hidden=8)
    cake.prebuild("synth", device=0, hidden=8)

    assert len(counted) == 2


def test_prebuild_runs_again_with_another_toolkit(counted, monkeypatch):
    old = ("/opt/cuda-12.8/bin/nvcc", loader.CudaVersion(12, 8, "12.8.93"))
    replaced = (TOOLKIT[0], loader.CudaVersion(12, 9, "12.9.86"))  # another toolkit installed at the same path
    cake.prebuild("synth", hidden=8)
    for toolkit in (old, replaced, TOOLKIT, old):
        monkeypatch.setattr(loader, "_toolkit", lambda nvcc=None: toolkit)
        cake.prebuild("synth", hidden=8)

    assert len(counted) == 3


def test_prebuild_remembers_nothing_without_a_toolkit_and_the_op_reports_it(counted, monkeypatch):
    def no_toolkit(nvcc=None):
        raise RuntimeError("no nvcc to build the Cake kernels: set CUDA_HOME or put nvcc on PATH")

    monkeypatch.setattr(loader, "_toolkit", no_toolkit)
    cake.prebuild("synth", hidden=8)
    cake.prebuild("synth", hidden=8)
    assert len(counted) == 2

    # an op that compiles names the missing toolkit itself
    monkeypatch.setattr(sys.modules[synthetic.IMPL], "prebuild", lambda **_: loader.check_toolkit("sm_100a", "synth"))
    with pytest.raises(RuntimeError, match="no nvcc to build"):
        cake.prebuild("synth", hidden=8)


def test_prebuild_remembers_no_failure_and_nothing_without_the_device(counted, monkeypatch):
    for _ in range(2):
        with pytest.raises(ValueError, match="hidden <= 64"):
            cake.prebuild("synth", hidden=65)
    monkeypatch.setattr(registry, "_device", lambda device: None)
    cake.prebuild("synth", hidden=8)
    cake.prebuild("synth", hidden=8)

    assert [params["hidden"] for params in counted] == [65, 65, 8, 8]


def test_the_device_of_the_key_never_raises():
    assert registry._device("cpu") is None  # no torch, no CUDA or not a CUDA device: the op reports it


def test_the_device_of_the_key_is_the_cuda_index_and_its_architecture(monkeypatch):
    torch = pytest.importorskip("torch")
    capabilities = {0: (10, 0), 1: (10, 0), 2: (10, 3), 3: (9, 0)}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capabilities[device])

    for device in (None, "cuda", 1, "cuda:1", torch.device("cuda", 1), torch.device("cuda")):
        assert registry._device(device) == (1, "sm_100a"), device
    assert registry._device(0) == (0, "sm_100a")
    assert registry._device("cuda:2") == (2, "sm_103a")
    assert registry._device(3) is None  # the op reports the unsupported device


def test_release_forgets_what_prebuild_remembered(counted):
    cake.prebuild("synth", hidden=8)
    cake.release("synth")
    cake.prebuild("synth", hidden=8)
    miles_ext_kernels.release()
    cake.prebuild("synth", hidden=8)
    cake.prebuild("synth", hidden=8)

    assert len(counted) == 3


def test_release_empties_the_pool_and_every_imported_implementation(synth, monkeypatch):
    monkeypatch.setattr(workspace, "release", lambda op=None: 1000 if op is None else 10)

    assert miles_ext_kernels.release() == 1000
    assert cake.release("synth") == 10
    assert synthetic.CALLS == [("release", {}), ("release", {})]


def test_release_skips_an_implementation_that_was_never_imported(synth, monkeypatch):
    monkeypatch.delitem(sys.modules, synthetic.IMPL)

    assert cake.release() == 0
    assert synthetic.CALLS == []
    with pytest.raises(KeyError, match="no cake op 'nope'"):
        cake.release("nope")


def test_an_export_is_imported_from_the_implementation_on_first_use(synth):
    assert synth.__all__ == ["OP", "scale"]
    assert "scale" not in vars(synth)
    assert synth.scale is synthetic.scale
    assert "scale" in vars(synth) and "scale" in dir(synth)
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        _ = synth.nope
