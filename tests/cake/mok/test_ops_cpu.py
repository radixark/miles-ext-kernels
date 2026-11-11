"""The MoK implementation without a GPU (needs torch; the runtime-API test also tvm_ffi).

Checked: the libraries ``prebuild`` selects for each model (every library of the build manifest with nothing set, the 16
(17 on SM107a) or 13 of one model, the counts the GPU tests assert for their model, every value of an unset parameter);
its parameter checks, which raise before any device is touched; the vendored package imports with this PyTorch and
tvm-ffi and has what it uses of them (symmetric memory, ``Work.block_current_stream``, ``tvm_ffi.Shape`` and the
raw-stream context) and builds through the runtime's loader; the workspace check refuses a multi-rank group without a
multicast address; the weight quantizer's PyTorch path (CPU tensors) writes the bytes of the independent MXFP8 recipe,
whatever the layout of the weights; ``release`` drops the package's caches.
"""

import functools
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels.cake._impl.mok import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

from . import _reference as reference  # noqa: E402

ARCHES = sorted(loader.MIN_CUDA)
VENDORED = "miles_ext_kernels.cake._vendor.cake_mok_bf16"
ONE_MODEL = dict(ep_size=8, local_experts=36, topk=8, clamped_swiglu=False, fp32_wgrad=False)


def _manifest(arch):
    cake_build = loader.load_package("cake_mok_bf16").cake_build
    return {name for name, entry in cake_build.libraries_by_name().items() if entry["arch"] == arch}


@pytest.mark.parametrize("arch", ARCHES)
def test_with_nothing_set_the_libraries_are_the_build_manifest(arch):
    every = ops.libraries(arch)
    bf16 = ops.libraries(arch, mxfp8=False)

    # SM107a also builds the MXFP8 forward's multimacro programs
    multimacro = 2 if arch == "sm_107a" else 0
    assert len(every) == len(set(every)) == 50 + multimacro
    assert set(every) == _manifest(arch)
    assert set(every) - set(bf16) == {name for name in every if "_mxfp8" in name}
    assert len(set(every) - set(bf16)) == 8 + multimacro


@pytest.mark.parametrize("arch", ARCHES)
def test_one_model_builds_its_layout_topk_form_and_weight_gradients(arch):
    stems = (
        "zero expert_layout gather barrier count_8_36 rows_8_36 pad_8 epilogue_forward_8 epilogue_backward_8 "
        "forward forward_noctx recompute backward forward_mxfp8 recompute_mxfp8 backward_mxfp8"
    ).split()
    if arch == "sm_107a":
        stems.append("forward_mxfp8_multimacro")

    assert sorted(ops.libraries(arch, **ONE_MODEL)) == sorted(f"cake_mok_{stem}_{arch}" for stem in stems)
    assert len(ops.libraries(arch, **ONE_MODEL, mxfp8=False)) == 13


@pytest.mark.parametrize("arch", ARCHES)
def test_the_gpu_tests_model_selects_27_libraries_29_on_sm107a(arch):
    """The counts the GPU prebuild tests assert for ``_cases.model()``: on SM107a each SwiGLU form's MXFP8 forward
    adds its multimacro library."""
    from . import _cases

    model = _cases.model()
    layout = dict(ep_size=model["ep_size"], local_experts=model["local_experts"], topk=model["topk"])
    one_form = dict(layout, clamped_swiglu=False, fp32_wgrad=False)
    multimacro = 1 if arch == "sm_107a" else 0

    assert len(ops.libraries(arch, **layout)) == 27 + 2 * multimacro
    assert len(ops.libraries(arch, **one_form)) == 16 + multimacro
    assert len(ops.libraries(arch, **one_form, mxfp8=False)) == 13


def test_an_unset_parameter_builds_every_value():
    both = set(ops.roles(**{**ONE_MODEL, "clamped_swiglu": None, "fp32_wgrad": None}))

    assert both - set(ops.roles(**ONE_MODEL)) == {
        "forward_clamped",
        "forward_clamped_noctx",
        "recompute_clamped",
        "backward_clamped",
        "backward_fp32",
        "backward_clamped_fp32",
        "forward_mxfp8_clamped",
        "forward_mxfp8_clamped_multimacro",
        "recompute_mxfp8_clamped",
        "backward_mxfp8_clamped",
        "backward_mxfp8_fp32",
        "backward_mxfp8_clamped_fp32",
    }
    layouts = set(ops.roles(topk=8)) - set(ops.roles(ep_size=8, local_experts=36, topk=8))
    assert {"count_1_4", "rows_64_4", "pad_64", "count_32_9"} <= layouts
    assert {"epilogue_forward_2", "epilogue_backward_2"} <= set(ops.roles()) - set(ops.roles(topk=8))


@pytest.mark.parametrize(
    "params, message",
    [
        (dict(ep_size=8), "together"),
        (dict(ep_size=1, local_experts=8), "ep_size, local_experts"),
        (dict(ep_size=True, local_experts=4), "must be ints"),
        (dict(topk=4), "topk must be one of"),
        (dict(topk=True), "topk must be one of"),
        (dict(clamped_swiglu=1), "clamped_swiglu must be one of"),
        (dict(fp32_wgrad="yes"), "fp32_wgrad must be one of"),
        (dict(mxfp8=None), "mxfp8 must be a bool"),
        (dict(intermediate_size=256), "needs hidden_size"),
        (dict(hidden_size=512), "needs intermediate_size"),
        (dict(hidden_size=768, intermediate_size=256), "hidden_size must be a positive multiple of 512"),
        (dict(hidden_size=512, intermediate_size=384), "intermediate_size must be a positive multiple of 256"),
        (dict(mxfp8=False, hidden_size=-256), "hidden_size must be a positive multiple of 256"),
    ],
)
def test_prebuild_rejects_parameters_the_kernels_do_not_serve(params, message):
    with pytest.raises(ValueError, match=message):
        ops.prebuild(**params)


def test_the_runtime_has_what_the_package_uses():
    tvm_ffi = pytest.importorskip("tvm_ffi")
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    from miles_ext_kernels.cake._vendor.cake_mok_bf16 import _kernels, backend, workspace

    for name in ("empty", "rendezvous", "enable_symm_mem_for_group"):
        assert callable(getattr(symm_mem, name, None)), f"torch.distributed._symmetric_memory has no {name}"
    assert callable(getattr(dist.Work, "block_current_stream", None)), "torch.distributed.Work.block_current_stream"
    for name in ("Shape", "device", "use_raw_stream"):
        assert hasattr(tvm_ffi, name), f"tvm_ffi has no {name}"
    divmod7 = _kernels._fast_divmod(7)
    assert isinstance(divmod7, tvm_ffi.Shape) and len(divmod7) == 3 and divmod7[0] == 7
    assert tuple(_kernels._fast_divmod(1)) == (1, 0, 0)
    assert backend.MoKFunctional and workspace.MoKConfig


def test_the_package_builds_through_the_runtime_loader():
    assert loader.load_package("cake_mok_bf16").cake_build._jit_toolchain.__module__ == loader.__name__


def test_a_multi_rank_workspace_needs_a_multicast_address(monkeypatch):
    def fake_api(ep_size, multicast):
        storage = types.SimpleNamespace(
            ep_size=ep_size,
            all_gather_top_experts_buffer_multicast_ptr=multicast,
            barrier_buffer_multicast_ptr=multicast,
        )
        create = lambda **kwargs: ("config", types.SimpleNamespace(storage=storage, kwargs=kwargs))  # noqa: E731
        return types.SimpleNamespace(create_mok_bf16_workspace=create)

    arguments = dict(group="group", device="cuda:0", num_local_tokens=512, hidden_size=512, topk=8)
    monkeypatch.setattr(ops, "_api", lambda: fake_api(4, 0))
    assert issubclass(ops.NoMulticastError, RuntimeError)
    with pytest.raises(ops.NoMulticastError, match="symmetric-memory multicast across the 4 expert-parallel ranks"):
        ops.create_mok_bf16_workspace(**arguments)
    monkeypatch.setattr(ops, "_api", lambda: fake_api(4, 1 << 40))
    config, workspace = ops.create_mok_bf16_workspace(**arguments, minibatch_size=256)
    assert config == "config" and workspace.kwargs == {**arguments, "minibatch_size": 256}
    monkeypatch.setattr(ops, "_api", lambda: fake_api(1, 0))
    ops.create_mok_bf16_workspace(**arguments)


def _weights():
    torch.manual_seed(17)
    gate = torch.randn(3, 256, 512, dtype=torch.bfloat16)
    gate[0, 5, 64:96] = 0  # an all-zero block takes the minimum scale
    gate[1, 7, :32] *= 1e4  # large blocks saturate to the finite range
    return gate, torch.randn_like(gate), torch.randn(3, 512, 256, dtype=torch.bfloat16)


def _assert_same_bytes(got, want):
    for got_side, want_side in zip(got, want, strict=True):
        for got_weight, want_weight in zip(got_side, want_side, strict=True):
            assert len(got_weight) == len(want_weight)
            for a, b in zip(got_weight, want_weight, strict=True):
                assert a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_the_weight_quantizer_writes_the_recipe_bytes_on_cpu():
    gate, up, down = _weights()

    _assert_same_bytes(ops.quantize_mok_mxfp8_weights(gate, up, down), reference.routed_mxfp8_weights(gate, up, down))


def test_weights_of_another_layout_are_copied_and_quantized_alike():
    gate, up, down = _weights()
    flat = torch.empty(down.numel() + 1, dtype=down.dtype)
    offset = flat[1:].view(down.shape)  # 2 bytes past an aligned allocation
    offset.copy_(down)
    strided = up.transpose(1, 2).contiguous().transpose(1, 2)

    assert ops._dense(gate) is gate
    assert ops._dense(strided) is not strided and ops._dense(strided).is_contiguous()
    assert ops._dense(offset) is not offset and ops._dense(offset).data_ptr() % 16 == 0
    _assert_same_bytes(
        ops.quantize_mok_mxfp8_weights(gate, strided, offset), reference.routed_mxfp8_weights(gate, up, down)
    )


def test_release_drops_the_loaded_modules_and_the_device_constants(monkeypatch):
    from miles_ext_kernels.cake._vendor.cake_mok_bf16 import jit, mxfp8

    monkeypatch.setattr(jit, "_load", functools.cache(lambda role, device: object()))
    jit._load("forward", 0)
    mxfp8._scale_factor(torch.device("cpu"))
    assert jit._load.cache_info().currsize == 1 and mxfp8._scale_factor.cache_info().currsize >= 1

    ops.release()

    assert jit._load.cache_info().currsize == 0 and mxfp8._scale_factor.cache_info().currsize == 0
    assert f"{VENDORED}.jit" in sys.modules
