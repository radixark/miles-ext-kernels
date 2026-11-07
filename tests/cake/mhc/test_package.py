"""The mHC implementation without a GPU (CPU torch): what ``prebuild`` builds, the parameter dtypes the kernels
receive, the variant each call runs, the workspaces each launch takes from the runtime's pool and what ``release``
drops. Stand-ins replace the kernels where a call would launch them.
"""

import argparse
import ast
import inspect

import pytest

torch = pytest.importorskip("torch")

import miles_ext_kernels  # noqa: E402
from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import mhc  # noqa: E402
from miles_ext_kernels.cake._impl.mhc import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, workspace  # noqa: E402

DET, FAST, HEAD = "cake_mhc_det_train", "cake_mhc_nondet_train", "cake_mhc_head_det_train"
HIDDEN_BY_ARCH = {"sm_100a": (4096, 5120, 7168), "sm_103a": (4096, 5120, 7168), "sm_107a": (4096,)}
CASES = [(arch, hidden) for arch, sizes in HIDDEN_BY_ARCH.items() for hidden in sizes]


def test_the_exports_are_the_implementation_and_its_constants():
    assert mhc.HIDDEN_SIZES == mhc._HIDDEN_SIZES == ops.HIDDEN_SIZES == (4096, 5120, 7168)
    assert mhc.STREAMS == mhc._STREAMS == 4
    for name in mhc.OP.exports:
        assert getattr(mhc, name) is getattr(ops, name)
    assert (ops.RMS_EPS, ops.HC_EPS, ops.SINKHORN_ITERS, ops.SINKHORN_ITERS_MAX) == (1e-5, 1e-6, 20, 64)


# --- prebuild: the model's hidden size and Sinkhorn count (review: another count compiled inside the call) ---


def test_prebuild_takes_the_hidden_size_and_the_sinkhorn_count_and_requires_both():
    parameters = inspect.signature(ops.prebuild).parameters

    assert list(parameters) == ["device", "max_workers", "hidden", "sinkhorn_iters"]
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values())
    assert parameters["hidden"].default is inspect.Parameter.empty
    assert parameters["sinkhorn_iters"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="sinkhorn_iters"):
        cake.prebuild("mhc", hidden=4096)


def _run_time_libraries(package: str, arch: str, hidden: int) -> set[str]:
    jit = getattr(ops, {DET: "_det", FAST: "_fast"}[package]).cake_jit
    return {
        jit.library_name(record, arch)
        for kind in jit.KINDS
        for record in jit.route_modules(arch, hidden, kind)
        if (jit.MODULES[record]["select_rule"] or {}).get("kind") == "run_time_iters"
    }


@pytest.mark.parametrize(("arch", "hidden"), CASES)
def test_another_sinkhorn_count_adds_the_run_time_count_programs(arch, hidden):
    at_record, at_seven = ops.libraries(arch, hidden, 20), ops.libraries(arch, hidden, 7)

    assert set(at_record) == set(at_seven) == {DET, FAST, HEAD}
    assert at_seven[HEAD] == at_record[HEAD] and len(at_record[HEAD]) == 2
    for package in (DET, FAST):
        added = set(at_seven[package]) - set(at_record[package])
        assert set(at_record[package]) < set(at_seven[package])
        assert added == _run_time_libraries(package, arch, hidden) and len(added) == 2, sorted(added)
        assert ops.libraries(arch, hidden, 1) == ops.libraries(arch, hidden, 64) == at_seven


# SM counts besides the boards of record (B200 148, GB300 152, R200 208 / 216): other boards of each architecture
SM_COUNTS = {"sm_100a": {148}, "sm_103a": {148, 152, 160}, "sm_107a": {208, 212, 216}}


def _token_counts(jit) -> list[int]:
    counts = {1, 2, 3, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 1023, 1024, 1025, 2047, 2048, 2049}
    counts |= {4095, 4096, 4097, 8191, 8192, 8193, 14927, 14928, 16263, 16384, 16385, 32768, 32769, 65536, 65537}
    counts |= {131072, 262144, 1 << 20}
    counts |= {T for T in jit.RULES_T_VALUES if T >= 1}  # the host rules hold for T >= 1; T = 0 launches nothing
    return sorted(counts)


@pytest.mark.parametrize("package", [DET, FAST])
@pytest.mark.parametrize(("arch", "hidden"), CASES)
@pytest.mark.parametrize("sinkhorn_iters", [20, 7])
def test_prebuild_covers_the_programs_of_every_token_count(package, arch, hidden, sinkhorn_iters):
    """Every program the host rules select for a call, at any token count, is a library prebuild builds."""
    jit = getattr(ops, {DET: "_det", FAST: "_fast"}[package]).cake_jit
    built = set(ops.libraries(arch, hidden, sinkhorn_iters)[package])
    sm_counts = SM_COUNTS[arch] | {sm_count for board, sm_count in jit.RULES_BOARDS if board == arch}

    missing = set()
    for sm_count in sorted(sm_counts):
        for T in _token_counts(jit):
            for kind in jit.KINDS:
                for record in jit.selected_modules(
                    arch, hidden, kind, rows=T, sm_count=sm_count, sinkhorn_iters=sinkhorn_iters
                ):
                    if jit.library_name(record, arch) not in built:
                        missing.add((sm_count, T, kind, record))
    assert not missing, sorted(missing)[:10]


@pytest.mark.parametrize(
    ("arch", "hidden", "sinkhorn_iters", "error"),
    [
        ("sm_107a", 5120, 20, r"hidden sizes \(4096,\) on sm_107a, not 5120"),
        ("sm_100a", 2048, 20, r"hidden sizes \(4096, 5120, 7168\) on sm_100a, not 2048"),
        ("sm_100a", 4096, 0, "sinkhorn_iters from 1 to 64, not 0"),
        ("sm_100a", 4096, 65, "sinkhorn_iters from 1 to 64, not 65"),
        ("sm_100a", 4096, True, "sinkhorn_iters must be an int"),
        ("sm_100a", 4096, 7.0, "sinkhorn_iters must be an int"),
        ("sm_100a", "4096", 20, "hidden must be an int"),
    ],
)
def test_a_model_shape_the_kernels_do_not_serve_is_a_value_error(arch, hidden, sinkhorn_iters, error):
    with pytest.raises(ValueError, match=error):
        ops.libraries(arch, hidden, sinkhorn_iters)


def test_prebuild_refuses_a_device_that_is_not_cuda():
    with pytest.raises(NotImplementedError, match="CUDA devices, not cpu"):
        ops.prebuild(device="cpu", hidden=4096, sinkhorn_iters=20)


def test_no_cuda_device_supports_nothing():
    assert ops.is_supported("cpu") is False
    assert ops.supported_hidden_sizes("cpu") == ()


# --- BF16 parameters (review: FSDP2's param_dtype=bf16 hands them to the layer and they raised) ---


def test_a_bf16_parameter_is_cast_to_fp32_and_its_gradient_returns_in_bf16():
    generator = torch.Generator().manual_seed(7)
    fn = (torch.randn(24, 64, generator=generator) * 1e-2).to(torch.bfloat16).requires_grad_(True)
    weight = torch.randn(24, 64, generator=generator, dtype=torch.float64)

    as_fp32 = ops._fp32(fn)
    assert as_fp32.dtype == torch.float32 and torch.equal(as_fp32.double(), fn.detach().double())  # exact
    (as_fp32.double() * weight).sum().backward()
    assert fn.grad.dtype == torch.bfloat16
    assert torch.equal(fn.grad, weight.float().to(torch.bfloat16))  # the FP32 gradient, rounded once

    fp32 = torch.zeros(3)
    assert ops._fp32(fp32) is fp32 and ops._fp32(None) is None
    half = torch.zeros(3, dtype=torch.float16)
    assert ops._fp32(half) is half  # the kernels' own validation rejects it


def _record(monkeypatch, package, entry):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return entry, None

    monkeypatch.setattr(package, entry, fake)
    return calls


def _inputs(dtype):
    X = torch.zeros(3, 4, 4096, dtype=torch.bfloat16)
    return X, torch.ones(24, 4 * 4096, dtype=dtype), torch.ones(3, dtype=dtype), torch.ones(24, dtype=dtype)


def test_prepare_hands_the_kernels_fp32_parameters(monkeypatch):
    calls = _record(monkeypatch, ops._det, "cake_mhc_det_train_prepare")

    X, fn, scale, base = _inputs(torch.bfloat16)
    ops.mhc_prepare(X, fn, scale, base)
    [(args, _)] = calls
    assert args[0] is X and [tensor.dtype for tensor in args[1:]] == [torch.float32] * 3
    assert all(torch.equal(got, want.float()) for got, want in zip(args[1:], (fn, scale, base)))

    X, fn, scale, base = _inputs(torch.float32)
    ops.mhc_prepare(X, fn, scale, base)
    assert all(got is want for got, want in zip(calls[1][0], (X, fn, scale, base)))  # no copy


# --- the variant of a call (review: deterministic mode without a variant ran fast) ---


def test_each_call_runs_the_variant_configured_when_it_runs(monkeypatch):
    det = _record(monkeypatch, ops._det, "cake_mhc_det_train_prepare")
    fast = _record(monkeypatch, ops._fast, "cake_mhc_nondet_train_prepare")
    X, fn, scale, base = _inputs(torch.float32)

    def runs(**constants):
        return ops.mhc_prepare(X, fn, scale, base, **constants)[0]

    assert runs() == "cake_mhc_det_train_prepare"  # never configured: det
    cake.configure(argparse.Namespace(deterministic_mode=True))
    assert runs() == "cake_mhc_det_train_prepare"
    miles_ext_kernels.configure({"cake_mhc_variant": "fast"})
    assert runs(rms_eps=1e-6, hc_eps=1e-5, sinkhorn_iters=7) == "cake_mhc_nondet_train_prepare"
    cake.configure()
    assert runs() == "cake_mhc_det_train_prepare"

    assert fast[0][1] == {"rms_eps": 1e-6, "hc_eps": 1e-5, "sinkhorn_iters": 7}
    assert det[0][1] == {"rms_eps": ops.RMS_EPS, "hc_eps": ops.HC_EPS, "sinkhorn_iters": ops.SINKHORN_ITERS}


def test_the_variant_comes_from_the_configuration_only(monkeypatch):
    """torch's deterministic algorithms no longer turn fast into det (the fast backward raises torch's alert)."""
    _record(monkeypatch, ops._fast, "cake_mhc_nondet_train_prepare")
    cake.configure(cake_mhc_variant="fast")
    torch.use_deterministic_algorithms(True)
    try:
        assert ops.mhc_prepare(*_inputs(torch.float32))[0] == "cake_mhc_nondet_train_prepare"
    finally:
        torch.use_deterministic_algorithms(False)


def test_finish_runs_the_package_of_the_handle_and_refuses_another(monkeypatch):
    finished = []
    monkeypatch.setattr(ops._det, "cake_mhc_det_train_finish", lambda X, B, handle: finished.append(("det", handle)))
    monkeypatch.setattr(
        ops._fast, "cake_mhc_nondet_train_finish", lambda X, B, handle: finished.append(("fast", handle))
    )
    det_handle = ops._det._MHCTrainHandle(ops._det._MHCTrainState(), ())
    fast_handle = ops._fast._MHCTrainHandle(ops._fast._MHCTrainState(), ())

    cake.configure(cake_mhc_variant="fast")  # the handle decides, not the configuration
    ops.mhc_finish(None, None, det_handle)
    ops.mhc_finish(None, None, fast_handle)
    assert finished == [("det", det_handle), ("fast", fast_handle)]
    with pytest.raises(TypeError, match="handle must be the handle mhc_prepare returned, not object"):
        ops.mhc_finish(None, None, object())


def test_the_head_runs_the_head_packages_entry(monkeypatch):
    """``mhc_head`` calls the head package's mode-word autograd entry (a stand-in here: no device work)."""
    monkeypatch.setattr(ops._head, "cake_mhc_head_det", lambda X: ("head", X))
    X = torch.zeros(3, 4, 4096, dtype=torch.bfloat16)

    name, given = ops.mhc_head(X)
    assert name == "head" and given is X


# --- workspaces: the runtime's pool (review: one bounded pool with a release hook) ---


def _functions_using(tree: ast.Module, name: str, *, stored: bool = False) -> set[str]:
    """The functions of ``tree`` whose body reads ``name`` (``stored``: assigns an item of it)."""
    found = set()
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        for node in ast.walk(function):
            if stored:
                hit = (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == name
                )
            else:
                hit = isinstance(node, ast.Name) and node.id == name
            if hit:
                found.add(function.name)
    return found


def test_both_packages_take_every_launch_workspace_through_the_pooled_resolver():
    """``_workspace_for`` of each package is the op's pooled resolver, and nothing else in the package creates or
    records a default workspace: ``_default_workspace`` is reached only from ``_workspace_for``, and ``_DEFAULT_VIEWS``
    is written only there (``_fast_workspace`` reads it, then calls ``_workspace_for``). A FlashInfer update that adds
    another path fails here."""
    for package in (ops._det, ops._fast):
        resolver = package._workspace_for
        assert resolver.__module__ == ops.__name__ and resolver.__qualname__.startswith("_pooled_workspaces.")
        tree = ast.parse(inspect.getsource(package))
        assert _functions_using(tree, "_default_workspace") == {"_workspace_for"}, package.__name__
        assert _functions_using(tree, "_DEFAULT_VIEWS") == {"_workspace_for", "_fast_workspace"}, package.__name__
        assert _functions_using(tree, "_DEFAULT_VIEWS", stored=True) == {"_workspace_for"}, package.__name__
        assert "_fast_workspace" in _functions_using(tree, "_workspace_for"), package.__name__
        module_level = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        aliases = [
            node
            for statement in module_level
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and node.id == "_workspace_for"
        ]
        assert not aliases, f"{package.__name__} keeps a reference to its own _workspace_for"


class _Package:
    """What the pooled resolver reads of a vendored package: 512 bytes per token of the bucket, the 256-byte ticket
    region first."""

    def __init__(self):
        self.caller = []

    @staticmethod
    def _workspace_bytes(kind, rows, device, hidden):
        return 512 * rows

    @staticmethod
    def _workspace_views(region, kind, *, rows, device, hidden):
        return {"ticket": region[:256], "data": region[256 : 512 * rows]}

    def _caller_views(self, caller, kind, rows, hidden, device):
        self.caller.append((caller, kind, rows, hidden, device))
        return {"caller": caller}


CPU = torch.device("cpu")


def test_a_pooled_workspace_has_zero_tickets_every_time_the_pool_hands_it_out(pooled):
    package = _Package()
    resolve = ops._pooled_workspaces(package, "det")

    # the pool's buffer of ("mhc", "det-fwd") for the bucket of 3 tokens (4), filled with non-zero bytes
    workspace.take("mhc", "det-fwd", (2048,), torch.uint8, CPU).fill_(255)
    hits = workspace.stats()["cpu"]["hits"]
    views = resolve(None, "fwd", 3, 4096, CPU)
    assert workspace.stats()["cpu"]["hits"] == hits + 1  # the same buffer
    assert (workspace.stats()["cpu"]["bytes"], workspace.stats()["cpu"]["entries"]) == (2048, 1)
    assert not views["ticket"].any() and bool((views["data"] == 255).all())  # only the tickets get a value

    views["ticket"].fill_(1)  # whatever the tickets hold, the next launch on the same buffer zeroes them
    again = resolve(None, "fwd", 4, 4096, CPU)
    assert workspace.stats()["cpu"]["hits"] == hits + 2 and again["ticket"].data_ptr() == views["ticket"].data_ptr()
    assert not again["ticket"].any() and bool((again["data"] == 255).all())
    assert not resolve(None, "fwd", 5, 4096, CPU)["ticket"].any()  # bucket 8: a larger buffer, zeroed
    assert workspace.stats()["cpu"]["bytes"] == 4096
    assert not resolve(None, "bwd", 5, 4096, CPU)["ticket"].any()  # the role of the backward: its own buffer
    assert workspace.stats()["cpu"]["entries"] == 2

    assert cake.release("mhc") == 8192
    assert not resolve(None, "fwd", 5, 4096, CPU)["ticket"].any()  # a new buffer after release


def test_a_workspace_outside_the_pool_has_its_tickets_zeroed_every_call():
    resolve = ops._pooled_workspaces(_Package(), "fast")
    cake.configure(cake_workspace_pool_bytes=0)  # 0 keeps nothing: every request is a buffer of its own
    for rows in (3, 3, 5):
        assert not resolve(None, "bwd", rows, 4096, CPU)["ticket"].any()
    assert workspace.stats()["cpu"]["bytes"] == 0


def test_at_the_default_bound_of_zero_take_keeps_nothing_and_release_frees_nothing():
    """Never configured, the bound is 0: every launch takes a buffer of its own (tickets zeroed) and the pool keeps
    none of them, so ``release`` frees nothing."""
    assert config.option(None, "cake_workspace_pool_bytes") == 0
    before = workspace.stats().get("cpu", {"hits": 0, "misses": 0})
    resolve = ops._pooled_workspaces(_Package(), "det")
    for kind, rows in (("fwd", 3), ("fwd", 3), ("bwd", 5)):
        assert not resolve(None, kind, rows, 4096, CPU)["ticket"].any()
    after = workspace.stats()["cpu"]
    assert (after["bytes"], after["entries"]) == (0, 0)
    assert (after["hits"], after["misses"]) == (before["hits"], before["misses"] + 3)
    assert cake.release("mhc") == 0 and miles_ext_kernels.release() == 0


def test_a_caller_workspace_goes_to_the_package():
    package = _Package()
    caller = torch.zeros(4096, dtype=torch.uint8)
    assert ops._pooled_workspaces(package, "det")(caller, "bwd", 3, 4096, CPU)["caller"] is caller
    assert package.caller == [(caller, "bwd", 3, 4096, CPU)]
    assert workspace.stats().get("cpu", {"bytes": 0})["bytes"] == 0


# --- release ---


def test_release_frees_the_pooled_workspaces_and_drops_the_plans_and_loaded_libraries(pooled):
    caches = ("_EDGES", "_PLANS", "_FAST_ROUTES")
    for package in (ops._det, ops._fast):
        for name in caches:
            getattr(package, name)[("stand-in", name)] = torch.zeros(1)
    ops._built[("stand-in",)] = []
    ops._pooled_workspaces(_Package(), "det")(None, "bwd", 2, 4096, CPU)

    assert miles_ext_kernels.release() == 1024  # the pool's mHC buffer (bucket 2)
    for package in (ops._det, ops._fast):
        assert {name: dict(getattr(package, name)) for name in caches} == {name: {} for name in caches}
        assert package._DEFAULT_WORKSPACES == {} and package._DEFAULT_VIEWS == {}
    for package in (ops._det, ops._fast, ops._head):
        assert package.cake_jit.load.cache_info().currsize == 0
    assert ops._built == {}
