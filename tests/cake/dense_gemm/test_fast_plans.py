"""Which programs the ``fast`` variant of the dense projection GEMMs runs, per shape, token count, architecture and SM
count (``fast_projection_programs`` / ``fast_router_programs``: the vendored non-deterministic package's own planner on
``meta`` views; needs torch and tvm_ffi, no device).

The tables below record which operations run programs from the non-deterministic package on B200, GB200, B300, GB300
and R200 parts. The current upstream tables select short-token SIMT router weight and input gradients on some parts,
split-K, two-chain and stream-K weight gradients of some single-head projections, and the deterministic sibling
elsewhere; on 208-SM R200 parts every case runs the deterministic sibling. The cases cover miles' projections and
routers and the single-head dense projections named in the package's routing tables.
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
K1_N256_C2 = "dense_proj_gemm_atomic_nn_n256_t_c2"
K1_SK_N256 = "dense_proj_gemm_sk_nn_n256_t"
SIMT_N64 = "router_fp32_gemm_simt_n64m64s2"
SIMT_N128 = "router_fp32_gemm_simt_n128m64s2"
DGRAD_M32_S4 = "router_fp32_dgrad_simt_m32n64s4"
DGRAD_M32_S8 = "router_fp32_dgrad_simt_m32n64s8"
DGRAD_M64_S4 = "router_fp32_dgrad_simt_m64n64s4"
FROM_128 = TOKENS[1:]  # two 64-token K blocks per split: a two-split program needs 65 tokens or more


def _at(template: str, tokens) -> dict[int, str]:
    return {t: template for t in tokens}


def _without(tokens, *dropped) -> tuple[int, ...]:
    return tuple(t for t in tokens if t not in dropped)


# (arch, SM count) -> case -> {token count: template of the weight gradient's program}
EXPECTED_WEIGHT_GRAD: dict[tuple[str, int], dict[str, dict[int, str]]] = {
    ("sm_100a", 148): {
        "dsv4-o-group-g1": _at(K1_N256_C2, (1001, 4096, 10240, 16384)),
        "dense-4096-32": _at(K1_N64, _without(FROM_128, 65536)),
        "dense-5120-64": _at(K1_N64, _without(FROM_128, 1001)),
        "router-e256-h4096": _at(SIMT_N128, (64, 128, 192)),
        "router-e256-h5120": _at(SIMT_N64, (64, 128, 192)),
        "router-e256-h6144": _at(SIMT_N64, (64, 128, 192)),
    },
    ("sm_100a", 152): {
        "dense-4096-32": _at(K1_N64, _without(FROM_128, 65536)),
        "dense-5120-64": _at(K1_N64, _without(FROM_128, 1001)),
        "router-e256-h4096": _at(SIMT_N128, (64, 128, 192)),
        "router-e256-h5120": _at(SIMT_N64, (64, 128, 192)),
        "router-e256-h6144": _at(SIMT_N64, (64, 128, 192)),
    },
    ("sm_103a", 148): {
        "dense-4096-32": _at(K1_N64, _without(FROM_128, 65536)),
        "dense-5120-64": _at(K1_N64, _without(FROM_128, 1001, 49152)),
        "dense-6144-128": _at(K1_N128, _without(FROM_128, 1001, 4096)),
        "router-e256-h4096": _at(SIMT_N128, (64, 128, 192)),
        "router-e256-h5120": _at(SIMT_N64, (64, 128, 192)),
        "router-e256-h6144": _at(SIMT_N64, (64, 128, 192)),
    },
    ("sm_103a", 152): {
        "dense-4096-32": _at(K1_N64, _without(FROM_128, 65536)),
        "dense-5120-64": _at(K1_N64, (128, 192, 4096, 10240, 16384, 20480, 32768)),
        "dense-6144-128": _at(K1_N128, _without(FROM_128, 1001, 4096)),
        "router-e256-h4096": _at(SIMT_N64, (64, 128, 192)),
        "router-e256-h5120": _at(SIMT_N64, (64, 128, 192)),
        "router-e256-h6144": _at(SIMT_N64, (64, 128, 192)),
    },
    ("sm_103a", 160): {
        "dense-4096-32": _at(K1_N64, _without(FROM_128, 65536)),
        "dense-5120-64": _at(K1_N64, (128, 192, 4096, 10240, 16384, 20480, 32768)),
        "dense-6144-128": _at(K1_N128, _without(FROM_128, 1001, 4096)),
    },
    ("sm_107a", 208): {},
    ("sm_107a", 212): {
        "dsv4-o-group-g1": _at(K1_SK_N256, (16384,)),
        "dense-8192-128": _at(K1_N128, (4096, 10240, 16384, 20480, 32768)),
        "router-e256-h4096": _at(SIMT_N128, (64, 128, 192)),
        "router-e256-h5120": _at(SIMT_N128, (64, 128, 192)),
        "router-e256-h6144": _at(SIMT_N128, (64, 128, 192)),
    },
    ("sm_107a", 216): {
        "dense-8192-128": _at(K1_N128, _without(FROM_128, 65536)),
    },
}
EXPECTED_INPUT_GRAD = {
    ("sm_100a", 148): {
        "router-e256-h4096": _at(DGRAD_M32_S8, (64,)),
        "router-e256-h5120": _at(DGRAD_M32_S4, (64, 128)),
    },
    ("sm_100a", 152): {
        "router-e256-h4096": _at(DGRAD_M32_S4, (64, 128)),
        "router-e256-h5120": _at(DGRAD_M32_S4, (64, 128)),
        "router-e256-h6144": _at(DGRAD_M32_S4, (64,)),
    },
    ("sm_103a", 148): {"router-e256-h4096": _at(DGRAD_M32_S8, (64,))},
    ("sm_103a", 152): {
        "router-e256-h4096": _at(DGRAD_M32_S4, (64, 128)),
        "router-e256-h5120": _at(DGRAD_M32_S4, (64, 128)),
        "router-e256-h6144": _at(DGRAD_M32_S4, (64,)),
    },
    ("sm_107a", 212): {
        "router-e256-h4096": {64: DGRAD_M32_S8, 128: DGRAD_M64_S4},
        "router-e256-h5120": {64: DGRAD_M32_S8, 128: DGRAD_M64_S4},
        "router-e256-h6144": _at(DGRAD_M32_S8, (64,)),
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

    assert fast_backend.__name__ == "miles_ext_kernels.cake._vendor.cake_dense_gemm_nondet_train.cake_backend"
    assert fast_backend.deterministic_backend() is det_backend
    assert fast_jit.deterministic_module("cake_jit") is ops._vendored()[1]


def test_router_input_gradients_and_no_forwards_run_programs_of_their_own(table):
    found = {
        part: {name: ops_["input_grad"] for name, ops_ in cases.items() if "input_grad" in ops_}
        for part, cases in table.items()
    }
    assert {part: cases for part, cases in found.items() if cases} == EXPECTED_INPUT_GRAD
    assert not any("forward" in ops_ for cases in table.values() for ops_ in cases.values())


def test_the_programs_fast_runs_are_the_recorded_ones(table):
    found = {part: {name: ops_["weight_grad"] for name, ops_ in cases.items()} for part, cases in table.items()}

    assert set(found) == set(EXPECTED_WEIGHT_GRAD) == set(PARTS)
    # the cases whose programs differ from the record: (arch, SM count, case) -> {token count: template} found
    differing = {
        f"{arch} {sms} {name}": found[(arch, sms)].get(name)
        for (arch, sms), expected in EXPECTED_WEIGHT_GRAD.items()
        for name in {*found[(arch, sms)], *expected}
        if found[(arch, sms)].get(name) != expected.get(name)
    }
    assert not differing, json.dumps(differing, indent=1, sort_keys=True)


def test_148_sm_short_token_router_and_group_projection_routes_and_160_sm_det_routes(table):
    routers = {"router-e256-h4096", "router-e256-h5120", "router-e256-h6144"}
    # the two-chain weight gradient of DeepSeek-V4's single-group output projection, 961 to 16384 tokens on B200
    assert set(table[("sm_100a", 148)]) & set(MILES_CASES) == routers | {"dsv4-o-group-g1"}
    assert set(table[("sm_103a", 148)]) & set(MILES_CASES) == routers
    for part in (("sm_100a", 148), ("sm_103a", 148)):
        for name in routers:
            assert set(table[part][name]["weight_grad"]) == {64, 128, 192}
    assert not set(table[("sm_103a", 160)]) & set(MILES_CASES)


# Historical accuracy misses on 148-SM sm_103a parts: measured tokens and the range now routed to det.
KNOWN_MISSES = {48: ((40961, 40962), (40961, 49151)), 64: ((47105,), (47105, 49152))}


@pytest.mark.parametrize("d_out", sorted(KNOWN_MISSES))
def test_the_documented_accuracy_misses_now_route_to_det(d_out):
    measured, (first, last) = KNOWN_MISSES[d_out]
    doc = " ".join(ops.__doc__.split())
    assert f"``5120 -> {d_out}`` BF16 at {first} to {last} tokens" in doc
    for tokens in (65, first - 1):
        on_148 = fast_projection_programs(
            1, 5120, d_out, weight_layout="out_in", tokens=tokens, arch="sm_103a", sm_count=148
        )
        assert on_148["weight_grad"] == K1_N64, tokens
    for tokens in (*measured, first, (first + last) // 2, last):
        on_148 = fast_projection_programs(
            1, 5120, d_out, weight_layout="out_in", tokens=tokens, arch="sm_103a", sm_count=148
        )
        assert on_148["weight_grad"] is None, tokens
    # the package's tables already run the deterministic program at these token counts on 152-SM parts
    on_152 = fast_projection_programs(
        1, 5120, d_out, weight_layout="out_in", tokens=last, arch="sm_103a", sm_count=152
    )
    assert on_152["weight_grad"] is None


def test_the_stream_k_table_holds_every_row_of_the_deterministic_weight_gradient_table():
    # fast_projection_programs reports None as det's plan and bits: the package's entry point reads its stream-K
    # table before K1_WGRAD_DET_RULES_BY_SM, so a row of that table runs the stream-K program while that table holds it
    backend, jit = ops._vendored_fast()
    det_rules = backend.K1_WGRAD_DET_RULES_BY_SM
    assert det_rules
    for (arch, sm_count), rows in det_rules.items():
        stream_k = backend.K1_WGRAD_SK_RULES_BY_SM.get((arch, sm_count), {})
        for d_in, d_out, tokens, dtype in rows:
            template = stream_k.get((d_in, d_out, tokens, dtype))
            assert template in jit.KERNELS[arch], (arch, sm_count, d_in, d_out, tokens, dtype)
            if dtype == "bf16":
                planned = fast_projection_programs(
                    1, d_in, d_out, weight_layout="out_in", tokens=tokens, arch=arch, sm_count=sm_count
                )
                assert planned["weight_grad"] == template, (arch, sm_count, d_in, d_out, tokens)


def test_fast_runs_a_program_of_its_own_on_every_part_but_208_sm_r200(table):
    # the variant changes which program runs for some supported input on every part but 208-SM sm_107a parts, where
    # the package's tables keep every case on the deterministic program (no router window, no BF16 split-K row)
    assert [part for part in PARTS if not table[part]] == [("sm_107a", 208)]


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_every_recorded_program_is_built_by_the_package(arch):
    _, fast_jit = ops._vendored_fast()
    libraries = {entry["name"] for entry in fast_jit.cake_build.manifest()["libraries"]}
    templates = {
        template
        for expected in (EXPECTED_WEIGHT_GRAD, EXPECTED_INPUT_GRAD)
        for (part_arch, _), cases in expected.items()
        if part_arch == arch
        for tokens in cases.values()
        for template in tokens.values()
    }

    assert templates
    for template in templates:
        module = fast_jit.select_module(arch, template)
        assert module in fast_jit.MODULES, template
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
