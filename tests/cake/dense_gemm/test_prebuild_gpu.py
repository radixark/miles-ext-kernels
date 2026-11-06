"""The prebuild contract of the dense projection GEMMs on the device: one ``prebuild`` of the model shapes, then no
call compiles, at any token count and in either variant.

``prebuild`` takes the model shapes only (no token count): the plan of a call changes with the token count, so it
builds every program the two packages register for the device's architecture for the GEMM kinds it is given. Each
route of :data:`ROUTES` runs the forward and the backward over token counts from 1 to 20480, among them 239 and seeded
random counts, under ``no_build`` in ``det`` and in ``fast`` (``check_prebuild`` configures each). The routes cover
every switch a call takes at call time: the variant, the weight layout, a strided input slice and a weight split of
a parameter (read in place), a weight that is not 16-byte aligned (copied), one gradient only, a side stream, the
router's ``[T, 1, H]`` and ``[T, H]`` inputs, and ``fast`` under ``torch.use_deterministic_algorithms`` (it runs
``det``). The cold prebuild compiles every registered program of the architecture (about 100 libraries). A fresh
process that imports the implementation and runs both variants loads no flashinfer module. Needs a CUDA device; the
tests need SM100a / SM103a / SM107a and skip elsewhere.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.dense_gemm import ops  # noqa: E402
from miles_ext_kernels.cake.dense_gemm import (  # noqa: E402
    batched_projection,
    fast_projection_programs,
    fast_router_programs,
    router_gating,
)

from ..helpers import check_prebuild, no_build  # noqa: E402

if torch.cuda.get_device_capability() not in cake.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake dense projection GEMMs need SM100a / SM103a / SM107a", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCH = cake.ARCH_BY_CAPABILITY[torch.cuda.get_device_capability()]
SM_COUNT = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
SEED = 20261105
# the tile, split and window boundaries the plans change at, 239 (a count no earlier prebuild named) and seeded
# random counts
TOKENS = sorted(
    {1, 33, 64, 128, 239, 1001, 2049, 4097, 10240, 16384, 20480, *random.Random(SEED).sample(range(2, 20481), 4)}
)

# name -> (heads, d_in, d_out, weight layout): miles' projections and the single-head rows fast has programs for
PROJECTIONS = {
    "glm5-q-absorb-h16": (16, 192, 512, "in_out"),
    "glm5-value-h16": (16, 512, 256, "out_in"),
    "glm53-q-absorb-h64": (64, 256, 512, "in_out"),
    "dsv4-o-group-g8": (8, 4096, 1024, "out_in"),
    "dsv4-o-group-g1": (1, 4096, 1024, "out_in"),
    "dense-4096-32": (1, 4096, 32, "out_in"),
}
# name -> (experts, hidden)
ROUTERS = {f"router-e{e}-h{h}": (e, h) for e, h in ((256, 4096), (256, 5120), (256, 6144), (384, 7168))}
MODEL = dict(projections=sorted(PROJECTIONS.values()), router=sorted(ROUTERS.values()))


def _projection_step(shape, tokens, *, slot=None, split=None, offset=0, grads=("x", "weight")):
    """One forward and backward of ``batched_projection``. ``slot``: ``x`` is the first ``d_in`` features of a
    ``slot``-wide head; ``split``: ``(head_rows, row0)``, the weight is ``rows`` rows from ``row0`` of each head of a
    ``[H * head_rows, cols]`` parameter; ``offset``: the parameter starts ``offset`` elements into its buffer;
    ``grads``: the leaves that require a gradient."""
    heads, d_in, d_out, layout = shape
    rows, cols = (d_in, d_out) if layout == "in_out" else (d_out, d_in)
    head_rows, row0 = split or (rows, 0)
    gen = torch.Generator(device="cuda").manual_seed(SEED + tokens)
    x_full = torch.randn(tokens, heads, slot or d_in, generator=gen, device="cuda").to(torch.bfloat16)
    buffer = (torch.randn(offset + heads * head_rows * cols, generator=gen, device="cuda") * d_in**-0.5).to(
        torch.bfloat16
    )
    x_full.requires_grad_("x" in grads)
    buffer.requires_grad_("weight" in grads)
    x = x_full[..., :d_in]
    weight = buffer[offset:].view(heads, head_rows, cols)[:, row0 : row0 + rows]
    out = batched_projection(x, weight, weight_layout=layout)
    out.backward(torch.randn(out.shape, generator=gen, device="cuda").to(torch.bfloat16))


def _router_step(experts, hidden, tokens, *, flat=False):
    gen = torch.Generator(device="cuda").manual_seed(SEED + tokens)
    shape = (tokens, hidden) if flat else (tokens, 1, hidden)
    x = torch.randn(*shape, generator=gen, device="cuda").to(torch.bfloat16).requires_grad_(True)
    weight = (torch.randn(experts, hidden, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    weight.requires_grad_(True)
    logits = router_gating(x, weight)
    logits.backward(torch.randn(logits.shape, generator=gen, device="cuda") * 1e-3)


def _over_tokens(step, *args, **kwargs):
    def run(variant):
        for tokens in TOKENS:
            step(*args, tokens, **kwargs)
        torch.cuda.synchronize()

    return run


def _on_a_side_stream(run):
    def on_stream(variant):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            run(variant)
        stream.synchronize()

    return on_stream


def _under_deterministic_algorithms(run):
    def deterministic(variant):
        torch.use_deterministic_algorithms(True)
        try:
            run(variant)
        finally:
            torch.use_deterministic_algorithms(False)

    return deterministic


ROUTES = {
    **{name: _over_tokens(_projection_step, shape) for name, shape in PROJECTIONS.items()},
    # x is a slice of the 256-wide q head slot; the weights are splits of linear_kv_up_proj.weight [H * 448, 512]
    "glm5-q-absorb-h16-slice-and-split": _over_tokens(
        _projection_step, PROJECTIONS["glm5-q-absorb-h16"], slot=256, split=(448, 0)
    ),
    "glm5-value-h16-split": _over_tokens(_projection_step, PROJECTIONS["glm5-value-h16"], split=(448, 192)),
    "glm5-value-h16-unaligned-weight": _over_tokens(_projection_step, PROJECTIONS["glm5-value-h16"], offset=1),
    "glm5-q-absorb-h16-input-gradient-only": _over_tokens(
        _projection_step, PROJECTIONS["glm5-q-absorb-h16"], grads=("x",)
    ),
    "dense-4096-32-weight-gradient-only": _over_tokens(
        _projection_step, PROJECTIONS["dense-4096-32"], grads=("weight",)
    ),
    "dsv4-o-group-g8-side-stream": _on_a_side_stream(_over_tokens(_projection_step, PROJECTIONS["dsv4-o-group-g8"])),
    **{name: _over_tokens(_router_step, *shape) for name, shape in ROUTERS.items()},
    "router-e384-h7168-flat": _over_tokens(_router_step, *ROUTERS["router-e384-h7168"], flat=True),
    "router-e256-h5120-side-stream": _on_a_side_stream(_over_tokens(_router_step, *ROUTERS["router-e256-h5120"])),
    "router-e256-h5120-deterministic-algorithms": _under_deterministic_algorithms(
        _over_tokens(_router_step, *ROUTERS["router-e256-h5120"])
    ),
    "dense-4096-32-deterministic-algorithms": _under_deterministic_algorithms(
        _over_tokens(_projection_step, PROJECTIONS["dense-4096-32"])
    ),
}


def test_one_prebuild_of_the_model_shapes_serves_every_route_and_token_count():
    check_prebuild("dense_gemm", ROUTES, **MODEL)


_FRESH_PROCESS = """
import json
import sys

from miles_ext_kernels import cake
from cake.dense_gemm import test_prebuild_gpu as module

cake.prebuild("dense_gemm", **module.MODEL)
for variant in ("det", "fast"):
    cake.configure(cake_dense_gemm_variant=variant)
    module._projection_step(module.PROJECTIONS["dense-4096-32"], 4097)
    module._router_step(*module.ROUTERS["router-e256-h5120"], 16384)
print(json.dumps(sorted(name for name in sys.modules if name.split(".")[0] == "flashinfer")))
"""


def test_a_fresh_process_running_both_variants_imports_no_flashinfer_module():
    paths = [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")]
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1]) == []


def test_the_routes_reach_the_fast_programs_of_this_device():
    """``fast`` runs a program of its own for some route on this part, or the part has none for these shapes."""
    part = dict(arch=ARCH, sm_count=SM_COUNT)
    planned = [
        fast_projection_programs(heads, d_in, d_out, weight_layout=layout, tokens=tokens, **part)
        for heads, d_in, d_out, layout in PROJECTIONS.values()
        for tokens in TOKENS
    ]
    planned += [
        fast_router_programs(experts, hidden, tokens=tokens, **part)
        for experts, hidden in ROUTERS.values()
        for tokens in TOKENS
    ]
    own = {programs["weight_grad"] for programs in planned} - {None}
    if not own:
        pytest.skip(f"fast runs no program of its own for these shapes on {ARCH} with {SM_COUNT} SMs")
    prebuilt = set(ops.programs_to_prebuild(ARCH, ["projections", "router"])["cake_dense_gemm_nondet_train"])
    _, fast_jit = ops._vendored_fast()

    assert {fast_jit.library_name(fast_jit.select_module(ARCH, program), ARCH) for program in own} <= prebuilt


def test_a_second_prebuild_compiles_nothing_and_returns_the_same_libraries():
    first = cake.prebuild("dense_gemm", **MODEL)
    with no_build():
        second = cake.prebuild("dense_gemm", **MODEL)

    assert second == first and all(path.is_file() for path in first)
    expected = ops.programs_to_prebuild(ARCH, ["projections", "router"])
    assert len(first) == sum(len(names) for names in expected.values())
