"""Every program a call of the dense projection GEMMs may launch, at any token count, is one ``prebuild`` builds (CPU;
torch and tvm_ffi, no device).

``prebuild`` builds every program the two packages register for the architecture for the GEMM kinds it is given,
because the plan of a call changes with the token count. Here the packages' own planners run on ``meta`` views of
miles' projections and routers, as the entry points call them, for every part (architecture and SM count) and a sweep
of token counts: every count up to 300, the neighbours of the tile, split and window boundaries, the token counts of
the exact-token rules, 239 and seeded random counts up to 70000; in both variants (``fast`` plans only weight
gradients of its own). Each planned program must be in the set ``prebuild`` builds for that GEMM kind. The GPU check
of the same guarantee is ``test_prebuild_gpu.py``.
"""

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels.cake._impl.dense_gemm import ops  # noqa: E402

DET, FAST = "cake_dense_gemm_det_train", "cake_dense_gemm_nondet_train"
PARTS = (
    ("sm_100a", 148),
    ("sm_100a", 152),
    ("sm_103a", 148),
    ("sm_103a", 152),
    ("sm_103a", 160),
    ("sm_107a", 208),
    ("sm_107a", 212),
    ("sm_107a", 216),
)
# the L2 size enters the plan only through the TMA eviction hints; one size per part is enough to plan with
L2_BYTES = 132120576
_EDGES = (32, 64, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
TOKENS = sorted(
    {
        *range(1, 301),
        *(edge + delta for edge in _EDGES for delta in (-1, 0, 1)),
        239,
        1001,
        2049,
        10240,
        16172,
        16231,
        20480,
        49152,
        *random.Random(20261105).sample(range(301, 70001), 24),
    }
)
# name -> (heads, d_in, d_out, weight layout): miles' projections and two single-head rows fast plans on some parts
PROJECTIONS = {
    **{f"glm5-q-absorb-h{h}": (h, 192, 512, "in_out") for h in (8, 16, 64)},
    **{f"glm5-value-h{h}": (h, 512, 256, "out_in") for h in (8, 16, 64)},
    **{f"glm53-q-absorb-h{h}": (h, 256, 512, "in_out") for h in (8, 16, 64)},
    **{f"dsv4-o-group-g{g}": (g, 4096, 1024, "out_in") for g in (1, 2, 8)},
    "dense-4096-32": (1, 4096, 32, "out_in"),
    "dense-6144-128": (1, 6144, 128, "out_in"),
}
# name -> (experts, hidden)
ROUTERS = {f"router-e{e}-h{h}": (e, h) for e, h in ((256, 4096), (256, 5120), (256, 6144), (384, 7168), (128, 2048))}


class _Planner:
    """The programs the entry points launch, as ``(package, module)``, planned on ``meta`` views for one part."""

    def __init__(self, arch: str, sm_count: int):
        self.arch, self.sm_count = arch, sm_count
        self.det_backend, self.det_jit = ops._vendored()
        _, self.fast_jit = ops._vendored_fast()

    def k1(self, a, b, out, variant: str) -> tuple[str, str]:
        """The K1 call ``gemm(a, b, out)`` of ``variant`` (the convenience path of either package)."""
        if variant == "fast":
            template = ops._fast_k1_template(a, b, out, arch=self.arch, sm_count=self.sm_count)
            if template is not None:
                return FAST, self.fast_jit.select_module(self.arch, template)
        backend = self.det_backend
        # a batched BF16 forward of few rows over K-major operands runs the swapped program, when one fits its size
        if out.dtype == torch.bfloat16 and a.dim() == 3 and a.stride(2) == 1 and b.stride(1) == 1:
            batches, rows, depth = a.shape
            cols = b.shape[2]
            if backend.swap_small_m(batches, rows, cols, False) and depth % backend.BLOCK_K == 0:
                template = backend.swap_ab_template(batches, rows, cols, self.sm_count, self.arch)
                if template is not None:
                    return DET, self.det_jit.select_module(self.arch, template)
        planned, *_ = backend.cake_dense_gemm_det_train_projection_plan(
            a, b, out, sm_count=self.sm_count, l2_bytes=L2_BYTES, arch=self.arch
        )
        return DET, self.det_jit.select_module(self.arch, planned.template)

    def k2(self, a, b, out, variant: str) -> tuple[str, str]:
        """The K2 call ``out = a @ b`` of ``variant``."""
        if variant == "fast":
            template = ops._fast_k2_template(a, b, out, arch=self.arch, sm_count=self.sm_count)
            if template is not None:
                return FAST, self.fast_jit.select_module(self.arch, template)
        planned, *_ = self.det_backend.cake_dense_gemm_det_train_router_plan(a, b, out)
        return DET, self.det_jit.select_module(self.arch, planned.template)

    def library(self, package: str, module: str) -> str:
        return (self.det_jit if package == DET else self.fast_jit).library_name(module, self.arch)


def _projection_launches(planner: _Planner, shape, tokens: int, variant: str) -> set[tuple[str, str]]:
    heads, d_in, d_out, layout = shape
    in_out = layout == "in_out"
    launched = set()

    def gemm(a, b, out):
        launched.add(planner.k1(a, b, out, variant))

    x, g, weight = ops._projection_views(heads, d_in, d_out, tokens, in_out)
    ops._forward(gemm, x, weight, g, in_out)
    ops._dgrad(gemm, g, weight, x, in_out)
    ops._wgrad(gemm, x, g, weight, in_out)
    return launched


def _router_launches(planner: _Planner, shape, tokens: int, variant: str) -> set[tuple[str, str]]:
    forward, dgrad, wgrad = ops._router_views(*shape, tokens)
    launched = {planner.k1(*forward, variant), planner.k2(*dgrad, variant), planner.k2(*wgrad, variant)}
    split = planner.det_backend.fused_split_module(planner.arch)
    if split is not None:
        launched.add((DET, split))
    return launched


@pytest.mark.parametrize("arch, sm_count", PARTS, ids=[f"{arch}-{sms}" for arch, sms in PARTS])
def test_every_planned_program_at_any_token_count_is_prebuilt(arch, sm_count):
    planner = _Planner(arch, sm_count)
    kinds = (
        ("projections", PROJECTIONS, _projection_launches),
        ("router", ROUTERS, _router_launches),
    )
    missing, reached = [], set()
    for kind, shapes, launches in kinds:
        programs = ops.programs_to_prebuild(arch, [kind])
        prebuilt = {(package, name) for package, names in programs.items() for name in names}
        for name, shape in shapes.items():
            for tokens in TOKENS:
                for variant in ("det", "fast"):
                    for package, module in launches(planner, shape, tokens, variant):
                        library = (package, planner.library(package, module))
                        reached.add(library)
                        if library not in prebuilt:
                            missing.append((name, tokens, variant, *library))

    assert not missing, missing[:20]
    # the sweep reaches programs of both packages, so the check is not vacuous
    assert {package for package, _ in reached} == {DET, FAST}, sorted(reached)
    assert len(reached) >= 8, sorted(reached)
