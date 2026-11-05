"""Which programs the ``fast`` variant of the dense projection GEMMs runs, per shape, token count, architecture and SM
count (``fast_projection_programs`` / ``fast_router_programs``: the vendored non-deterministic package's own planner on
``meta`` views; needs torch and tvm_ffi, no device).

The table below is the record: for each part (B200 148 SMs, GB200 152; B300 148, GB300 152 and 160; R200 208, 212
and 216) it lists the cases whose weight gradient runs a program of the non-deterministic package, by token count.
Every other (case, token count, part), and every forward and input gradient, runs the deterministic program, with the
bits of ``det``. The cases are the projections and routers of miles' models (MLA q-absorb and value at 8,
16 and 64 local heads, DeepSeek-V4's output groups, routers of 128 to 512 experts at hidden 2048 to 7168) and
single-head dense projections the package's tables name. Of miles' calls only the router weight gradient of 256
experts at hidden 4096, 5120 or 6144 runs a program of its own, on 152-SM sm_100a / sm_103a parts and 208- to 216-SM
sm_107a parts, inside the token windows below; on 148-SM parts and 160-SM sm_103a parts every miles call runs the
deterministic program.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels.cake._impl.dense_gemm import ops  # noqa: E402
from miles_ext_kernels.cake.dense_gemm import OPERATIONS, fast_projection_programs, fast_router_programs  # noqa: E402

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
TOKENS = (64, 128, 192, 1001, 4096, 10240, 16384, 20480, 32768, 49152, 65536)

# name -> (heads, d_in, d_out, weight layout)
PROJECTIONS = {
    **{f"glm5-q-absorb-h{h}": (h, 192, 512, "in_out") for h in (8, 16, 64)},
    **{f"glm5-value-h{h}": (h, 512, 256, "out_in") for h in (8, 16, 64)},
    **{f"glm53-q-absorb-h{h}": (h, 256, 512, "in_out") for h in (8, 16, 64)},
    **{f"dsv4-o-group-g{g}": (g, 4096, 1024, "out_in") for g in (1, 2, 8)},
    **{f"dense-{d_in}-{d_out}": (1, d_in, d_out, "out_in") for d_in, d_out in ((4096, 32), (5120, 64), (6144, 128))},
    "dense-8192-128": (1, 8192, 128, "out_in"),
}
# name -> (experts, hidden)
ROUTERS = {f"router-e{e}-h{h}": (e, h) for e in (128, 256, 288, 384, 512) for h in (2048, 4096, 5120, 6144, 7168)}

K1_N64 = "dense_proj_gemm_atomic_nn_n64_t"
K1_N128 = "dense_proj_gemm_atomic_nn_n128_t"
K2_SPANS = "router_fp32_gemm_atomic_nn_t"
K2_HEAD_TAIL = "router_fp32_gemm_atomic_nn_t_ht_c3_nreg"
K2_HEAD_TAIL_TS = "router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg"
FROM_128 = TOKENS[1:]  # two 64-token K blocks per split: a two-split program needs 65 tokens or more


def _at(template: str, tokens) -> dict[int, str]:
    return {t: template for t in tokens}


def _without(tokens, *dropped) -> tuple[int, ...]:
    return tuple(t for t in tokens if t not in dropped)


# (arch, SM count) -> case -> {token count: template of the weight gradient's program}
EXPECTED_WEIGHT_GRAD: dict[tuple[str, int], dict[str, dict[int, str]]] = {
    ("sm_100a", 148): {
        "dense-4096-32": _at(K1_N64, FROM_128),
        "dense-5120-64": _at(K1_N64, FROM_128),
    },
    ("sm_100a", 152): {
        "dense-4096-32": _at(K1_N64, FROM_128),
        "dense-5120-64": _at(K1_N64, _without(FROM_128, 1001)),
        "router-e256-h4096": _at(K2_HEAD_TAIL, (10240, 16384)),
        "router-e256-h5120": {**_at(K2_SPANS, (64, 128)), **_at(K2_HEAD_TAIL, (10240, 16384, 20480, 32768))},
    },
    ("sm_103a", 148): {
        "dense-4096-32": _at(K1_N64, FROM_128),
        "dense-5120-64": _at(K1_N64, FROM_128),
        "dense-6144-128": _at(K1_N128, FROM_128),
    },
    ("sm_103a", 152): {
        "dense-4096-32": _at(K1_N64, _without(FROM_128, 65536)),
        "dense-5120-64": _at(K1_N64, (128, 192, 4096, 10240, 16384, 20480)),
        "dense-6144-128": _at(K1_N128, _without(FROM_128, 1001, 4096)),
        "router-e256-h4096": _at(K2_HEAD_TAIL_TS, (20480, 32768, 49152)),
        "router-e256-h5120": {**_at(K2_SPANS, (64, 128, 192)), **_at(K2_HEAD_TAIL_TS, (16384, 20480, 32768))},
        "router-e256-h6144": _at(K2_HEAD_TAIL_TS, (16384, 20480)),
    },
    ("sm_103a", 160): {
        "dense-4096-32": _at(K1_N64, _without(FROM_128, 65536)),
        "dense-5120-64": _at(K1_N64, (128, 192, 4096, 10240, 16384, 20480)),
        "dense-6144-128": _at(K1_N128, _without(FROM_128, 1001, 4096)),
    },
    ("sm_107a", 208): {
        "router-e256-h5120": _at(K2_HEAD_TAIL, (16384,)),
    },
    ("sm_107a", 212): {
        "dense-8192-128": _at(K1_N128, (16384,)),
        "router-e256-h5120": _at(K2_HEAD_TAIL_TS, (16384,)),
        "router-e256-h6144": _at(K2_HEAD_TAIL_TS, (16384,)),
    },
    ("sm_107a", 216): {
        "dense-4096-32": _at(K1_N64, _without(FROM_128, 128, 192)),
        "dense-8192-128": _at(K1_N128, FROM_128),
        "router-e256-h5120": _at(K2_HEAD_TAIL, (16384,)),
    },
}
# the miles call cases: every one but the single-head dense projections
MILES_CASES = [name for name in [*PROJECTIONS, *ROUTERS] if not name.startswith("dense-")]


def _programs(name: str, tokens: int, arch: str, sm_count: int) -> dict[str, str | None]:
    if name in PROJECTIONS:
        heads, d_in, d_out, layout = PROJECTIONS[name]
        return fast_projection_programs(
            heads, d_in, d_out, weight_layout=layout, tokens=tokens, arch=arch, sm_count=sm_count
        )
    experts, hidden = ROUTERS[name]
    return fast_router_programs(experts, hidden, tokens=tokens, arch=arch, sm_count=sm_count)


@pytest.fixture(scope="module")
def table() -> dict[tuple[str, int], dict[str, dict[str, dict[int, str]]]]:
    """(arch, SM count) -> case -> operation -> {token count: template} of every program ``fast`` runs of its own."""
    found = {}
    for arch, sm_count in PARTS:
        part = found[(arch, sm_count)] = {}
        for name in [*PROJECTIONS, *ROUTERS]:
            for tokens in TOKENS:
                for op, template in _programs(name, tokens, arch, sm_count).items():
                    if template is not None:
                        part.setdefault(name, {}).setdefault(op, {})[tokens] = template
    return found


def test_fast_binds_the_vendored_deterministic_package():
    det_backend, _ = ops._vendored()
    fast_backend, fast_jit = ops._vendored_fast()

    assert fast_backend.__name__ == "miles_ext_kernels.cake._vendor.dense_projection_gemm.cake_backend"
    assert fast_backend.deterministic_backend() is det_backend
    assert fast_jit.deterministic_module("cake_jit") is ops._vendored()[1]


def test_only_weight_gradients_run_programs_of_their_own(table):
    other = {
        part: {name: sorted(ops_) for name, ops_ in cases.items() if set(ops_) - {"weight_grad"}}
        for part, cases in table.items()
    }

    assert {part: cases for part, cases in other.items() if cases} == {}


def test_the_programs_fast_runs_are_the_recorded_ones(table):
    found = {part: {name: ops_["weight_grad"] for name, ops_ in cases.items()} for part, cases in table.items()}

    assert found == EXPECTED_WEIGHT_GRAD, json.dumps(
        {f"{arch} {sms}": cases for (arch, sms), cases in found.items()}, indent=1, sort_keys=True
    )


def test_miles_calls_run_the_deterministic_program_on_148_and_160_sm_parts(table):
    for part in (("sm_100a", 148), ("sm_103a", 148), ("sm_103a", 160)):
        assert not set(table[part]) & set(MILES_CASES), part


# The rows of the op module's "Known accuracy misses" (148-SM sm_103a parts): d_out -> (the token counts whose miss
# was measured, the token range a later FlashInfer change runs the deterministic program on)
KNOWN_MISSES = {48: ((40961, 40962), (40961, 49151)), 64: ((47105,), (47105, 49152))}


@pytest.mark.parametrize("d_out", sorted(KNOWN_MISSES))
def test_the_documented_accuracy_misses_are_rows_fast_runs_its_two_split_program_on(d_out):
    measured, (first, last) = KNOWN_MISSES[d_out]
    doc = " ".join(ops.__doc__.split())
    assert f"``5120 -> {d_out}`` BF16 at {first} to {last} tokens" in doc
    for tokens in (65, *measured, (first + last) // 2, last):
        on_148 = fast_projection_programs(
            1, 5120, d_out, weight_layout="out_in", tokens=tokens, arch="sm_103a", sm_count=148
        )
        assert on_148["weight_grad"] == K1_N64, tokens
    # the package's tables already run the deterministic program at these token counts on 152-SM parts
    on_152 = fast_projection_programs(
        1, 5120, d_out, weight_layout="out_in", tokens=last, arch="sm_103a", sm_count=152
    )
    assert on_152["weight_grad"] is None


def test_fast_runs_a_program_of_its_own_on_every_part(table):
    # the variant changes which program runs for some supported input on every part
    assert all(table[part] for part in PARTS)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_every_recorded_program_is_built_by_the_package(arch):
    _, fast_jit = ops._vendored_fast()
    libraries = {entry["name"] for entry in fast_jit.cake_build.manifest()["libraries"]}
    templates = {
        template
        for (part_arch, _), cases in EXPECTED_WEIGHT_GRAD.items()
        if part_arch == arch
        for tokens in cases.values()
        for template in tokens.values()
    }

    assert templates
    for template in templates:
        module = fast_jit.select_module(arch, template)
        assert not fast_jit.is_det_program(module), template
        assert fast_jit.library_name(module, arch) in libraries, (arch, template)


def test_the_planner_inputs_are_checked():
    with pytest.raises(ValueError, match="weight_layout"):
        fast_projection_programs(1, 4096, 32, weight_layout="io", tokens=1001, arch="sm_100a", sm_count=148)
    with pytest.raises(ValueError, match="multiple of 8"):
        fast_projection_programs(1, 4096, 36, weight_layout="out_in", tokens=1001, arch="sm_100a", sm_count=148)
    with pytest.raises(ValueError, match="arch must be one of"):
        fast_router_programs(256, 4096, tokens=1001, arch="sm_90a", sm_count=132)


def test_the_operations_are_the_reported_keys():
    assert tuple(fast_router_programs(256, 5120, tokens=64, arch="sm_100a", sm_count=152)) == OPERATIONS
    assert (
        tuple(fast_projection_programs(1, 4096, 32, weight_layout="out_in", tokens=1001, arch="sm_100a", sm_count=148))
        == OPERATIONS
    )
