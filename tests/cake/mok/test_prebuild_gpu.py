"""The prebuild contract of ``mok`` on one GPU: after one ``prebuild('mok', ...)`` for a model, no route of the model
compiles a library or a Triton kernel, or loads a library the prebuild did not return.

The model of ``ROUTES``: the layout (1, 4), top-2, hidden 512, intermediate 256, with both SwiGLU forms and both
weight-gradient precisions, as they are left unset (27 libraries). ``ROUTES`` covers every call-time switch of the op:
BF16 and MXFP8 routed weights (each case of ``_cases.CASES``: plain and clamped SwiGLU through ``swiglu_limit``,
context recompute, the context-free forward, BF16 and FP32 weight gradients), token counts (0, 1, 777) through the
model's workspace, a new workspace (256 tokens), another communication-SM split, and the weight quantizer on weights of
another layout. Each route runs a forward and a backward and builds its adapter inside the check, so creating one is
checked too. The same model with ``clamped_swiglu=False`` and ``fp32_wgrad=False`` (16 libraries; 13 with
``mxfp8=False``) runs the routes of that form only, with adapters of BF16 weight gradients only.

The checks do not depend on what this process ran before: ``release('mok')`` drops the loaded kernel modules, so every
load of a route goes through the vendored ``cake_build.load_library``, which records the library; the in-process
caches of the vendored Triton kernels are cleared, so that ``prebuild`` compiles them into a private, empty
``TRITON_CACHE_DIR``. ``check_prebuild`` then runs the routes under ``no_build()`` (nvcc); the libraries they load must
be the ones the prebuild returned, every one of them, and the private Triton cache gains no entry. Needs SM100a /
SM103a / SM107a.
"""

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

from ..helpers import check_prebuild  # noqa: E402
from . import _cases  # noqa: E402

try:
    import triton
    import triton.language as tl
    from triton.runtime.jit import JITFunction
except ImportError:  # the weight quantizer then runs PyTorch operations, and there is nothing to compile
    triton = None

if triton is not None:

    @triton.jit
    def _probe(out):
        """A kernel only this test compiles: it shows that the private ``TRITON_CACHE_DIR`` is in effect."""
        tl.store(out, 1.0)


DEVICE = torch.device("cuda", torch.cuda.current_device())
ARCH = cake.device_arch(DEVICE)
# on SM107a, the one architecture it is built for, each MXFP8 forward loads its multimacro library beside it
MULTIMACRO = 1 if ARCH == "sm_107a" else 0
MODEL = _cases.model()
# the problem of the routes, created by the first route (inside the check), with the adapters of STATE["wgrads"]
_STATE: dict = {}


def _problem() -> _cases.Problem:
    if "problem" not in _STATE:
        _STATE["problem"] = _cases.Problem(dist.group.WORLD, DEVICE, wgrads=_STATE["wgrads"])
    return _STATE["problem"]


def _case(name):
    def run(variant):
        assert cake.variant("mok") == variant
        _problem().iteration(name).run()

    return run


def _tokens(count):
    def run(variant):
        _problem().tokens(count)

    return run


def _new_workspace(variant):
    problem = _problem()
    config, workspace = mok.create_mok_bf16_workspace(
        group=dist.group.WORLD, device=DEVICE, num_local_tokens=256, hidden_size=_cases.HIDDEN, topk=problem.topk
    )
    problem.tokens(256, workspace=workspace, config=config)


def _other_split(variant):
    _problem().iteration("bf16", config=_problem().other_split()).run()


def _quantizer_layouts(variant):
    """A transposed view and a copy at a 2-byte offset: the quantizer copies them and runs the prebuilt kernel."""
    gate, up, down = _problem().routed
    flat = torch.empty(down.numel() + 1, device=DEVICE, dtype=down.dtype)
    offset = flat[1:].view(down.shape)
    offset.copy_(down)
    mok.quantize_mok_mxfp8_weights(gate.transpose(1, 2).contiguous().transpose(1, 2), up, offset)


ROUTES = {
    **{name: _case(name) for name in _cases.CASES},
    "tokens-0": _tokens(0),
    "tokens-1": _tokens(1),
    "tokens-777": _tokens(777),
    "new-workspace": _new_workspace,
    "other-comm-split": _other_split,
    "weight-quantizer-layouts": _quantizer_layouts,
}


def _routes_of_one_form(mxfp8: bool) -> dict:
    """The routes of the model with plain SwiGLU and BF16 weight gradients: its cases (MXFP8 ones only with ``mxfp8``)
    and the other routes (the weight quantizer only with ``mxfp8``)."""
    cases = [
        name
        for name, case in _cases.CASES.items()
        if "swiglu_limit" not in case and not case.get("fp32_wgrad") and (mxfp8 or not case.get("mxfp8"))
    ]
    others = [name for name in ROUTES if name not in _cases.CASES and (mxfp8 or name != "weight-quantizer-layouts")]
    return {name: ROUTES[name] for name in (*cases, *others)}


@pytest.fixture(scope="module")
def group():
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


def _entries(directory: Path) -> list[str]:
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file())


def _library(path) -> str:
    """The library of a cached path, ``<library>_<closure digest>_<cache key>.so``."""
    return Path(path).name.rsplit("_", 2)[0]


def _forget_triton_kernels() -> None:
    """Clear the in-process caches of the vendored Triton kernels and of ``_probe``: a kernel an earlier test of this
    process compiled would be launched from memory, without a look at ``TRITON_CACHE_DIR``."""
    kernels = [value for value in vars(ops._vendored("mxfp8")).values() if isinstance(value, JITFunction)]
    assert kernels, "the vendored weight quantizer has no Triton kernel"
    for kernel in (*kernels, _probe):
        kernel.device_caches.clear()


def _check(tmp_path, monkeypatch, model: dict, routes: dict, *, libraries: int, wgrads: tuple[bool, ...]) -> None:
    """``check_prebuild('mok', routes, **model)`` after a prebuild of ``libraries`` libraries, whatever this process
    ran before (module docstring); the routes' problem has the adapters of ``wgrads``."""
    triton_cache = tmp_path / "triton"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton_cache))
    cake.release("mok")
    if triton is not None:
        _forget_triton_kernels()
        _probe[(1,)](torch.zeros(1, device=DEVICE))
        torch.cuda.synchronize(DEVICE)
        assert _entries(triton_cache), "TRITON_CACHE_DIR is not in effect"
    probed = _entries(triton_cache)
    paths = cake.prebuild("mok", device=DEVICE, **model)
    prebuilt = {_library(path) for path in paths}
    assert len(paths) == len(prebuilt) == libraries, sorted(prebuilt)
    before = _entries(triton_cache)
    if triton is not None and model.get("mxfp8", True):
        assert len(before) > len(probed), "prebuild compiled no Triton kernel"
    else:
        assert before == probed, "prebuild compiled a Triton kernel"

    cake_build = loader.load_package("cake_mok_bf16").cake_build
    load_library = cake_build.load_library
    loaded = set()

    def recording_load_library(name):
        loaded.add(name)
        return load_library(name)

    monkeypatch.setattr(cake_build, "load_library", recording_load_library)
    _STATE["wgrads"] = wgrads
    try:
        check_prebuild("mok", routes, **model)
    finally:
        _STATE.clear()

    assert _entries(triton_cache) == before, "a call compiled a Triton kernel"
    assert not loaded - prebuilt, f"calls loaded libraries the prebuild did not return: {sorted(loaded - prebuilt)}"
    assert not prebuilt - loaded, f"no route loaded these prebuilt libraries: {sorted(prebuilt - loaded)}"


def test_after_prebuild_no_route_compiles(group, tmp_path, monkeypatch):
    _check(tmp_path, monkeypatch, MODEL, ROUTES, libraries=27 + 2 * MULTIMACRO, wgrads=(False, True))


@pytest.mark.parametrize("mxfp8,libraries", [(True, 16 + MULTIMACRO), (False, 13)], ids=["mxfp8", "bf16"])
def test_after_a_prebuild_of_one_form_its_routes_compile_nothing(group, tmp_path, monkeypatch, mxfp8, libraries):
    """``clamped_swiglu=False`` and ``fp32_wgrad=False`` set: the narrower prebuild still serves the routes of the
    model, adapters included."""
    model = dict(MODEL, clamped_swiglu=False, fp32_wgrad=False, mxfp8=mxfp8)
    _check(tmp_path, monkeypatch, model, _routes_of_one_form(mxfp8), libraries=libraries, wgrads=(False,))
