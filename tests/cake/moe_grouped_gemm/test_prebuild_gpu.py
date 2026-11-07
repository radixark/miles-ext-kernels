"""The prebuild contract of the MoE grouped GEMM (README, Prebuild): after one ``prebuild("moe_grouped_gemm")`` no call
compiles, whatever the variant and the group sizes. ``ROUTES`` covers every call-time switch: group sizes that pick each
stage twin of the forward and both gradients (empty experts, uneven routing, a sweep of row counts, the weight
gradient's 512-wide k tile with its split tail), no rows at all, and a side stream (its own workspaces and, on SM100a /
SM103a, its own probe of the memory domains, compiled by ``prebuild`` and not by a call). Needs SM100a / SM103a /
SM107a; skips elsewhere.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.moe_grouped_gemm import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

from ..helpers import check_prebuild, no_build  # noqa: E402
from . import _cases  # noqa: E402
from ._cases import K512_SHAPE, K512_SIZES, OP, SEED, VARIANT  # noqa: E402

try:
    ARCH = loader.device_arch()
except NotImplementedError:
    pytest.skip("the Cake MoE grouped GEMM needs SM100a / SM103a / SM107a", allow_module_level=True)


def _call(sizes, n, k, seed=SEED, rows=None):
    """One forward and backward of the configured variant."""
    return _cases.step(*_cases.inputs(sizes, n, k, seed, rows=rows))


def _uneven(rows: int, groups: int, seed: int) -> list[int]:
    """``rows`` routed unevenly over ``groups`` experts (some empty)."""
    gen = torch.Generator().manual_seed(seed)
    weights = torch.rand(groups, generator=gen) ** 4
    picks = torch.multinomial(weights, rows, replacement=True, generator=gen) if rows else torch.zeros(0, dtype=int)
    return torch.bincount(picks, minlength=groups).tolist()


def _side_stream(sizes, n, k):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _call(sizes, n, k)
    torch.cuda.current_stream().wait_stream(side)


# name -> run(variant): one forward and backward (the variant is configured by check_prebuild)
ROUTES = {
    # every layer shape, both projections: empty and one-row experts, sizes that are not tile multiples
    **{
        f"ragged-{layer}-{proj}": (lambda variant, layer=layer, proj=proj: _call(*_cases.shape(layer, proj)))
        for layer, proj in _cases.CASES
    },
    # all rows on one expert, the others empty
    "one-expert": lambda variant: _call([6000] + [0] * 7, 1536, 2048),
    # no rows at all
    "no-rows": lambda variant: _call([0] * 8, 1536, 2048),
    # uneven routing over a sweep of row counts (rows per expert below and above every stage threshold)
    **{
        f"uneven-{rows}-rows": (lambda variant, rows=rows: _call(_uneven(rows, 8, rows), 4096, 2048))
        for rows in (1, 255, 3000, 33000, 140000)
    },
    # padding rows past the last group end
    "padded": lambda variant: _call(_cases.sizes_of(8), 3072, 2048, rows=sum(_cases.sizes_of(8)) + 300),
    # the weight gradient's 512-wide k tile, whose split tail fast adds unordered
    "k512-split-tail": lambda variant: _call(list(K512_SIZES), *K512_SHAPE),
    # many rows per expert at K >= 4096: the evict-first / 3-3 data-gradient twins and the wave-synchronised wgrad
    "wide-k-many-rows": lambda variant: _call([16400, 16400], 4096, 4096),
    # many rows per expert at K <= 2048: the 256-wide tail of the 512-wide k tile
    "narrow-k-many-rows": lambda variant: _call([9000, 9000], 4096, 2048),
    # a side stream: its own kept workspaces and probe
    "side-stream": lambda variant: _side_stream(*_cases.shape("h2048-i768-e16", "w2")),
}


def _probe_modules():
    return [ops._module(variant, "cake_backend")._queue_probe_module for variant in ("det", "fast")]


def test_after_prebuild_no_route_compiles(monkeypatch):
    """Every route of both variants runs with no nvcc build (``no_build``) and no tvm-ffi inline build: ``prebuild``
    compiled the probe of the memory domains as well, which a call would otherwise compile on first use."""
    for probe in _probe_modules():
        probe.cache_clear()
    cake.prebuild(OP)
    import tvm_ffi.cpp

    inline_builds = []
    real_load_inline = tvm_ffi.cpp.load_inline

    def load_inline(*args, **kwargs):
        inline_builds.append(kwargs.get("name"))
        return real_load_inline(*args, **kwargs)

    monkeypatch.setattr(tvm_ffi.cpp, "load_inline", load_inline)
    cake.release(OP)  # fresh stream workspaces: the first probed launch on each starts its probe

    check_prebuild(OP, ROUTES)

    assert inline_builds == []
    if ARCH in ("sm_100a", "sm_103a"):  # the probed data gradient ran on both variants
        probed = {variant: len(ops._module(variant, "cake_backend")._queue_splits) for variant in ("det", "fast")}
        assert all(probed.values()), probed


def test_a_variant_configured_after_a_call_runs_at_the_next_call_without_building():
    cake.prebuild(OP)
    sizes, (n, k) = list(K512_SIZES), K512_SHAPE
    x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED)
    from miles_ext_kernels.cake import moe_grouped_gemm

    with no_build():
        cake.configure(**{VARIANT: "det"})
        y_det = moe_grouped_gemm.grouped_mm(x, w.clone().requires_grad_(True), offs)
        cake.configure(**{VARIANT: "fast"})
        y_fast = moe_grouped_gemm.grouped_mm(x, w.clone().requires_grad_(True), offs)
        y_fast.backward(g)
        y_det.backward(g)
        torch.cuda.synchronize()

    assert type(y_det.grad_fn).__name__ == "CakeMoEGroupedGemmDetFunctionBackward"
    assert type(y_fast.grad_fn).__name__ == "CakeMoEGroupedGemmNondetFunctionBackward"


def test_prebuild_builds_every_program_the_planner_binds():
    """Every stage the vendored planners bind for the routes' products is among the prebuilt libraries."""
    prebuilt = {path.stem.rsplit("_", 2)[0] for path in cake.prebuild(OP)}  # <library>_<closure20>_<key16>.so
    assert prebuilt
    shapes = [_cases.shape(layer, proj) for layer, proj in _cases.CASES]
    shapes += [(list(K512_SIZES), *K512_SHAPE), ([16400, 16400], 4096, 4096), ([9000, 9000], 4096, 2048)]
    shapes += [(_uneven(rows, 8, rows), 4096, 2048) for rows in (1, 255, 3000, 33000, 140000)]
    launched = set()
    for variant, mode in (("det", "det"), ("fast", "nondet")):
        backend, jit = ops._module(variant, "cake_backend"), ops._module(variant, "cake_jit")
        name = f"cake_moe_grouped_gemm_{mode}_train_prepare_"
        prepare = {op: getattr(backend, name + op) for op in ("forward", "dgrad", "wgrad")}
        for sizes, n, k in shapes:
            x, w, g, offs = _cases.inputs(sizes, n, k, seed=SEED)
            for launch in (prepare["forward"](x, w, offs), prepare["dgrad"](g, w, offs), prepare["wgrad"](g, x, offs)):
                for stage in launch.stages:
                    binding = jit.stage_binding(launch.record_name, stage, ARCH)
                    launched.add(jit.library_name(binding["program"], ARCH, binding["specializations"]))

    assert launched and launched <= prebuilt, sorted(launched - prebuilt)


def test_prebuild_is_idempotent():
    assert cake.prebuild(OP) == cake.prebuild(OP)
