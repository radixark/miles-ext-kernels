"""How ``causal_conv1d`` hands a call to the vendored packages, on the CPU with stand-in packages: the variant and the
SiLU' policy read at the call (``det`` in a process that never configured the op; only ``configure`` selects the
variant, so ``deterministic_mode`` holds for every call), the one-document prefix sums of ``cu_seqlens=None`` built
on the device, ``det`` for ``fast`` under torch's deterministic algorithms; the checks of
``check_supported`` (a biased convolution is rejected); ``prebuild``'s arguments and the libraries it builds against
the packages' program selection; ``release``. Needs torch and tvm_ffi, not CUDA."""

import inspect
import logging
import types
from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.conv1d import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402

from ._shapes import KCP_CHANNELS, KCP_DOCUMENTS, KCP_LONG_POINT, SINGLE_RANK_POINTS  # noqa: E402

ARCHES = tuple(loader.ARCH_BY_CAPABILITY.values())
TAPS = (torch.float32, torch.bfloat16)


class _Packages:
    """Stand-ins for the two vendored packages: each entry records its call and returns ``x * w[:, 3]``."""

    def __init__(self):
        self.calls = []

    def _entry(self, variant):
        def entry(x, w, cu, bias, **kwargs):
            self.calls.append(types.SimpleNamespace(variant=variant, x=x, w=w, cu=cu, bias=bias, kwargs=kwargs))
            return (x.float() * w[:, 3].float()).to(x.dtype)

        return entry

    def package(self, variant):
        if variant == "det":
            return types.SimpleNamespace(cake_causal_conv1d_det=self._entry("det"))
        return types.SimpleNamespace(cake_causal_conv1d_nondet=self._entry("fast"))


@pytest.fixture
def packages(monkeypatch):
    stand_ins = _Packages()
    monkeypatch.setattr(ops, "_package", stand_ins.package)
    return stand_ins


def _operands(S=6, C=16, taps=torch.float32):
    x = torch.randint(-4, 5, (1, S, C)).bfloat16()
    w = torch.randint(-4, 5, (C, 4)).to(taps)
    return x, w


def test_an_unconfigured_process_runs_det_with_the_bf16_policy(packages):
    x, w = _operands()
    cu = torch.tensor([0, 2, 6], dtype=torch.int32)

    ops.causal_conv1d(x, w, cu)
    (call,) = packages.calls
    assert call.variant == "det" and call.bias is None
    assert call.kwargs == {"silu_grad_preactivation": "bf16"}
    assert torch.equal(call.cu, cu) and call.cu.dtype == torch.int32


def test_each_call_reads_the_configured_variant_and_policy(packages):
    x, w = _operands()
    cake.configure(cake_conv1d_variant="fast", cake_conv1d_silu_grad_preactivation="fp32")
    ops.causal_conv1d(x, w)
    cake.configure(cake_conv1d_variant="det")
    ops.causal_conv1d(x, w)
    cake.configure()
    ops.causal_conv1d(x, w)

    assert [(call.variant, call.kwargs) for call in packages.calls] == [
        ("fast", {"silu_grad_preactivation": "fp32"}),
        ("det", {"silu_grad_preactivation": "bf16"}),
        ("det", {"silu_grad_preactivation": "bf16"}),
    ]


def test_the_silu_argument_overrides_the_configured_policy_for_one_call(packages):
    x, w = _operands()
    cake.configure(cake_conv1d_variant="fast")

    ops.causal_conv1d(x, w, silu_grad_preactivation="fp32")
    ops.causal_conv1d(x, w)
    assert [(call.variant, call.kwargs.get("silu_grad_preactivation")) for call in packages.calls] == [
        ("fast", "fp32"),
        ("fast", "bf16"),
    ]
    with pytest.raises(ValueError, match="silu_grad_preactivation must be one of"):
        ops.causal_conv1d(x, w, silu_grad_preactivation="fp16")
    cake.configure(cake_conv1d_silu_grad_preactivation="fp16")  # configure does not check option values
    with pytest.raises(ValueError, match="silu_grad_preactivation must be one of"):
        ops.causal_conv1d(x, w)


def test_no_call_selects_a_variant_so_deterministic_mode_runs_det(packages):
    """Only ``configure`` selects the variant (the entry points took a ``variant`` argument before, which ran
    ``fast`` in a process configured with ``deterministic_mode``)."""
    for entry in (ops.causal_conv1d, ops.causal_conv1d_kcp, ops.causal_conv1d_kcp_multi):
        assert "variant" not in inspect.signature(entry).parameters, entry.__name__
    x, w = _operands()
    cake.configure(deterministic_mode=True)

    ops.causal_conv1d(x, w)
    with pytest.raises(TypeError, match="variant"):
        ops.causal_conv1d(x, w, variant="fast")
    assert [call.variant for call in packages.calls] == ["det"]


@pytest.mark.parametrize("shape", [(1, 37, 16), (37, 16)], ids=["batch", "flat"])
def test_no_cu_seqlens_is_one_document_built_without_a_host_copy(packages, monkeypatch, shape):
    """``cu_seqlens=None`` used to copy a host list to the device (a synchronizing copy from pageable memory)."""
    x = torch.zeros(shape, dtype=torch.bfloat16)
    w = torch.zeros(16, 4)

    def no_host_tensor(*args, **kwargs):
        raise AssertionError("causal_conv1d built cu_seqlens from host data")

    monkeypatch.setattr(torch, "tensor", no_host_tensor)
    ops.causal_conv1d(x, w)
    (call,) = packages.calls
    assert call.cu.dtype == torch.int32 and call.cu.device == x.device
    assert call.cu.tolist() == [0, 37]


def test_strided_inputs_are_made_contiguous_and_cu_seqlens_int32(packages):
    wide = torch.randint(-4, 5, (1, 6, 24)).bfloat16()
    x = wide[..., :16]
    w = torch.randint(-4, 5, (4, 16)).float().t()
    assert not x.is_contiguous() and not w.is_contiguous()

    ops.causal_conv1d(x, w, torch.tensor([0, 6], dtype=torch.int64))
    (call,) = packages.calls
    assert call.x.is_contiguous() and torch.equal(call.x, x)
    assert call.w.is_contiguous() and torch.equal(call.w, w)
    assert call.cu.dtype == torch.int32 and call.cu.tolist() == [0, 6]


def test_a_batch_of_several_sequences_is_rejected(packages):
    with pytest.raises(ValueError, match="one packed batch"):
        ops.causal_conv1d(torch.zeros(2, 4, 16, dtype=torch.bfloat16), torch.zeros(16, 4))
    assert not packages.calls


def test_fast_runs_det_under_torch_deterministic_algorithms(packages, monkeypatch, caplog):
    monkeypatch.setattr(config, "_warned", set())
    x, w = _operands()
    cake.configure(cake_conv1d_variant="fast")
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            ops.causal_conv1d(x, w)
            ops.causal_conv1d(x, w)
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    ops.causal_conv1d(x, w)
    assert [call.variant for call in packages.calls] == ["det", "det", "fast"]
    warnings = [r for r in caplog.records if "conv1d" in r.getMessage() and "running det" in r.getMessage()]
    assert len(warnings) == 1


# --- check_supported ---


@pytest.mark.parametrize(
    "channels, kernel_size, activation, has_bias, match",
    [
        (4096, 3, "silu", False, "kernel size 4"),
        (4096, 4, "sigmoid", False, "applies SiLU"),
        (4096, 4, "silu", True, "without a bias"),
        (4100, 4, "silu", False, "multiple of 8"),
    ],
    ids=["kernel-size", "activation", "bias", "channels"],
)
def test_unsupported_convolutions_are_rejected(channels, kernel_size, activation, has_bias, match):
    with pytest.raises(ValueError, match=match):
        ops.check_supported(channels, kernel_size, activation, has_bias=has_bias)


@pytest.fixture
def device(monkeypatch):
    """A stand-in CUDA device: ``device.capability`` and ``device.cuda`` (the toolkit's ``(major, minor)``)."""
    state = types.SimpleNamespace(capability=(10, 0), cuda=(13, 4))

    def toolkit(nvcc=None):
        return "nvcc", loader.CudaVersion(*state.cuda, "{}.{}.0".format(*state.cuda))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: state.capability)
    monkeypatch.setattr(loader, "_toolkit", toolkit)
    return state


def test_check_supported_needs_a_supported_device_and_its_toolkit(device, monkeypatch):
    for capability in loader.ARCH_BY_CAPABILITY:
        device.capability = capability
        ops.check_supported(4096, 4, "silu")
    device.capability = (9, 0)
    with pytest.raises(ValueError, match=r"SM100a / SM103a / SM107a"):
        ops.check_supported(4096, 4, "silu")
    device.capability, device.cuda = (10, 7), (13, 0)
    with pytest.raises(RuntimeError, match=r"short-conv kernels for sm_107a needs CUDA 13\.4 or newer"):
        ops.check_supported(4096, 4, "silu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="needs a CUDA device"):
        ops.check_supported(4096, 4, "silu")


# --- prebuild ---


def test_prebuild_takes_fp32_or_bf16_taps_on_a_supported_device(device, monkeypatch):
    built = []
    monkeypatch.setattr(loader, "prebuild_libraries", lambda package, names, max_workers=None: built.append(names))
    with pytest.raises(ValueError, match="FP32 or BF16 taps"):
        cake.prebuild("conv1d", weight_dtype=torch.float16)
    with pytest.raises(TypeError):
        cake.prebuild("conv1d")
    device.capability = (9, 0)
    with pytest.raises(NotImplementedError):
        cake.prebuild("conv1d", weight_dtype=torch.float32)
    device.capability, device.cuda = (10, 7), (13, 0)
    with pytest.raises(RuntimeError, match="needs CUDA 13.4"):
        cake.prebuild("conv1d", weight_dtype=torch.float32)
    assert not built


def _built(monkeypatch, device, arch, taps):
    """``{package name: library names}`` that ``prebuild`` builds for ``taps`` on a device of ``arch``."""
    device.capability = next(cap for cap, name in loader.ARCH_BY_CAPABILITY.items() if name == arch)
    built = {}

    def prebuild_libraries(cake_build, names, max_workers=None):
        built[cake_build.__name__.rsplit(".", 2)[-2]] = list(names)
        return []

    monkeypatch.setattr(loader, "prebuild_libraries", prebuild_libraries)
    assert cake.prebuild("conv1d", weight_dtype=taps, max_workers=3) == []
    return built


def _with_reduce(registry, name, arch, single_rank):
    names = [registry.library_name(name, arch)]
    reduce = registry.MODULES[name].get("reduce")
    if reduce:
        names.append(single_rank.library_name(reduce, arch))
    return names


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("taps", TAPS, ids=["fp32-taps", "bf16-taps"])
def test_prebuild_builds_every_program_a_call_selects(device, monkeypatch, arch, taps):
    """Every program a call of either variant, either SiLU' policy, any channel and token count, single rank or
    context-parallel, selects on ``arch`` (with its reduce program) is among the libraries ``prebuild`` builds; the
    context-parallel forward's main program is a program of the single-rank forward route of FP32 taps, also for BF16
    taps (the route widens them)."""
    built = _built(monkeypatch, device, arch, taps)
    assert sorted(built) == sorted(ops.PACKAGES.values())
    bf16 = taps == torch.bfloat16
    for variant, name in ops.PACKAGES.items():
        package = ops._package(variant)
        jit, kcp_jit = package.cake_jit, import_module(".kcp_jit", package.__name__)
        kcp = import_module(".kcp", package.__name__)
        assert all(library.endswith(f"_{arch}") for library in built[name])
        missing = set()
        for fp32_grad in (False, True):
            routes = (
                jit.forward_route(bias=False, bf16_taps=bf16),
                jit.backward_route(bias=False, fp32_grad=fp32_grad, bf16_taps_or_dweight=bf16),
            )
            for route in routes:
                for C in (8, 136, 256, 264, 512, 768, 1024, 1032, 2048, 2560, 8192, 24576):
                    for S in (5, 4097, 262144, 262401):
                        for module in jit.selected_modules(arch, route, C=C, S=S):
                            missing |= set(_with_reduce(jit, module, arch, jit)) - set(built[name])
        for kind in kcp_jit.ROUTES[arch]:
            for C in (8, 512, 1024, 2560, 8192):
                for module in kcp_jit.selected_modules(arch, kind, C=C, S=4097):
                    missing |= set(_with_reduce(kcp_jit, module, arch, jit)) - set(built[name])
        if arch in kcp.FWD_MAIN_ARCHS | kcp.MULTI_FWD_MAIN_ARCHS:
            assert kcp.FWD_MAIN_ROUTE == jit.forward_route(bias=False, bf16_taps=False)
            for C in (8, 256, 512, 1024, 2048, 2560, 8192):
                for S in (5, 4097, 65536, 262401):
                    for module in jit.selected_modules(arch, kcp.FWD_MAIN_ROUTE, C=C, S=S):
                        missing |= set(_with_reduce(jit, module, arch, jit)) - set(built[name])
        assert not missing, (variant, sorted(missing))


@pytest.mark.parametrize("arch", ARCHES)
def test_the_prebuild_routes_reach_every_program(arch):
    """The shapes of ``test_prebuild_gpu.py`` select every program of every route the op runs on ``arch``, so its
    check under ``no_build()`` covers every library ``prebuild`` builds."""
    for variant in ops.PACKAGES:
        package = ops._package(variant)
        jit, kcp_jit = package.cake_jit, import_module(".kcp_jit", package.__name__)
        kinds = set()
        for bf16 in (False, True):
            kinds.add(jit.forward_route(bias=False, bf16_taps=bf16))
            for fp32_grad in (False, True):
                kinds.add(jit.backward_route(bias=False, fp32_grad=fp32_grad, bf16_taps_or_dweight=bf16))
        for kind in sorted(kinds):
            reached = {
                module
                for C, lengths in SINGLE_RANK_POINTS
                for module in jit.selected_modules(arch, kind, C=C, S=sum(lengths))
            }
            assert reached == set(jit.route_modules(arch, kind)), (variant, kind)
        for kind in kcp_jit.ROUTES[arch]:
            reached = {module for C in KCP_CHANNELS for module in kcp_jit.selected_modules(arch, kind, C=C, S=4097)}
            assert reached == set(kcp_jit.route_modules(arch, kind)), (variant, kind)
        # the context-parallel forward's main program, which the BF16-taps process of ``test_prebuild_gpu.py`` reaches
        # through the context-parallel calls only
        kcp = import_module(".kcp", package.__name__)
        if arch in kcp.FWD_MAIN_ARCHS | kcp.MULTI_FWD_MAIN_ARCHS:
            points = [(C, sum(KCP_DOCUMENTS)) for C in KCP_CHANNELS]
            points.append((KCP_LONG_POINT[0], sum(KCP_LONG_POINT[1])))
            reached = {
                module for C, S in points for module in jit.selected_modules(arch, kcp.FWD_MAIN_ROUTE, C=C, S=S)
            }
            assert reached == set(jit.route_modules(arch, kcp.FWD_MAIN_ROUTE)), (variant, "kcp main forward")


# --- release ---


def test_release_drops_the_packages_host_plans(monkeypatch):
    cleared = []
    fast = ops._package("fast")
    monkeypatch.setattr(fast, "cake_causal_conv1d_nondet_train_plan_cache_clear", lambda: cleared.append("fast plans"))
    clears = {
        "det": (
            "cake_causal_conv1d_det_train_kcp_host_cache_clear",
            "cake_causal_conv1d_det_train_kcp_graph_cache_clear",
        ),
        "fast": (
            "cake_causal_conv1d_nondet_train_kcp_host_cache_clear",
            "cake_causal_conv1d_nondet_train_kcp_graph_cache_clear",
        ),
    }
    for variant in ops.PACKAGES:
        kcp = ops._kcp_module(variant)
        host, graphs = clears[variant]
        monkeypatch.setattr(kcp, host, lambda variant=variant: cleared.append(f"{variant} kcp host"))
        monkeypatch.setattr(kcp, graphs, lambda variant=variant: cleared.append(f"{variant} kcp graphs"))

    assert cake.release("conv1d") == 0
    assert cleared == ["fast plans", "det kcp host", "det kcp graphs", "fast kcp host", "fast kcp graphs"]
