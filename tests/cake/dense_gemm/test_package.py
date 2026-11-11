"""The vendored packages of the dense projection GEMMs and what ``prebuild`` builds from them (CPU; torch, no device).

``prebuild`` does not plan per token count: it builds every program the two packages register for the architecture
for the GEMM kinds it is given. These tests check that every registered program has a kind, that the two kinds split
the registry, that the shapes are checked before any device is touched, and that the docs state the batch-variance of
the results.
"""

import ast
import json
import re
from pathlib import Path

import pytest

import miles_ext_kernels.cake.dense_gemm as dense_gemm
from miles_ext_kernels import cake

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR = REPO_ROOT / "miles_ext_kernels" / "cake" / "_vendor"
VENDORED = VENDOR / "VENDORED.json"
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
FI_COMMIT = "c5c39b0e87f027011c86881aa2935a83e1ce53e7"
# The Cake revisions the two packages record at FI_COMMIT. The deterministic package records the same revision as at
# the FlashInfer commit vendored before (f5ab85cc) although its code changed, so the two records differ; the next test
# checks instead that the non-deterministic package finds every name it reads from the deterministic one.
CAKE_REVISIONS = {
    "cake_dense_gemm_det_train": "72d0af54b2ddd96cd9f931508e70999338e18cce",
    "cake_dense_gemm_nondet_train": "749760171088226c39074b121579802137265034",
}


def _ops():
    pytest.importorskip("torch")
    from miles_ext_kernels.cake._impl.dense_gemm import ops

    return ops


def test_both_packages_come_from_one_flashinfer_commit_and_the_recorded_cake_revisions():
    block = json.loads(VENDORED.read_text())["ops"]["dense_gemm"]

    assert set(block) == {"cake_dense_gemm_det_train", "cake_dense_gemm_nondet_train"}
    assert {entry["source"]["commit"] for entry in block.values()} == {FI_COMMIT}
    assert {package: entry["cake_revision"] for package, entry in block.items()} == CAKE_REVISIONS
    assert (
        block["cake_dense_gemm_nondet_train"]["source"]["path"]
        == "flashinfer/experimental/cake/cake_dense_gemm_nondet_train"
    )
    assert all("allow" not in entry for entry in block.values())


def _top_level_names(source: str) -> set[str]:
    """The names a module's top level defines, assigns or imports."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names |= {n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    return names


def test_the_non_deterministic_package_finds_every_name_it_reads_from_the_deterministic_one():
    nondet = VENDOR / "cake_dense_gemm_nondet_train"
    build = json.loads((nondet / "build.json").read_text())
    assert "cake_dense_gemm_det_train" in build["common"]["requires"]["packages"]
    source = (nondet / "cake_backend.py").read_text()
    # the deterministic backend module is bound as ``det`` (``det = deterministic_backend()``, ``det: ModuleType``)
    read = set(re.findall(r"\bdet\.([A-Za-z_]\w*)", source))
    read |= set(re.findall(r"deterministic_backend\(\)\.([A-Za-z_]\w*)", source))
    assert {"_template_knobs", "_TEMPLATE_PREFIX"} <= read
    defined = _top_level_names((VENDOR / "cake_dense_gemm_det_train" / "cake_backend.py").read_text())
    assert not sorted(read - defined)


@pytest.mark.parametrize("arch", ARCHES)
def test_every_registered_program_has_a_kind_and_the_kinds_split_the_registry(arch):
    ops = _ops()
    for package, cake_jit in ops._registries():
        templates = set(cake_jit.KERNELS[arch])
        kinds = {template: ops.program_kind(template) for template in templates}  # raises for an unknown template
        assert set(kinds.values()) <= {"projections", "router"}, package
        projections = ops.programs_to_prebuild(arch, ["projections"])[package]
        router = ops.programs_to_prebuild(arch, ["router"])[package]
        both = ops.programs_to_prebuild(arch, ["projections", "router"])[package]

        assert not set(projections) & set(router), package
        assert sorted(projections + router) == both, package
        assert len(both) == len(templates), package
        libraries = {entry["name"] for entry in cake_jit.cake_build.manifest()["libraries"]}
        assert set(both) <= libraries, sorted(set(both) - libraries)


@pytest.mark.parametrize("arch", ARCHES)
def test_the_router_kind_holds_the_k2_programs_the_split_and_the_fp32_forwards(arch):
    ops = _ops()
    det_jit = dict(ops._registries())["cake_dense_gemm_det_train"]
    kinds = {template: ops.program_kind(template) for template in det_jit.KERNELS[arch]}

    k2 = ("router_fp32_det_gemm_kk", "router_fp32_det_gemm_kn", "router_fp32_det_gemm_nn_t")
    assert [kinds[template] for template in k2] == ["router"] * 3
    assert kinds["split_fp32_det_bf16x3"] == "router"
    assert any(kind == "router" and "_f32" in template for template, kind in kinds.items())
    assert {kind for template, kind in kinds.items() if template.startswith("swap_ab_")} == {"projections"}
    bf16_k1 = [template for template in kinds if template.startswith("dense_") and "f32" not in template.split("_")]
    assert bf16_k1 and {kinds[template] for template in bf16_k1} == {"projections"}


def test_an_unknown_template_has_no_kind():
    with pytest.raises(ValueError, match="no GEMM kind"):
        _ops().program_kind("dense_proj_v2_gemm_kk_n256")


@pytest.mark.parametrize(
    "params, match",
    [
        (dict(projections=[(16, 196, 512, "in_out")]), "multiple of 8"),
        (dict(projections=[(16, 192, 512, "io")]), "weight_layout"),
        (dict(projections=[(0, 192, 512, "in_out")]), "at least one head"),
        (dict(projections=[(16, 192, 512)]), "tuples of 4 values"),
        (dict(router=[(260, 4096)]), "multiples of 8"),
        (dict(router=[(256, 4096, 8)]), "tuples of 2 values"),
    ],
)
def test_prebuild_checks_the_shapes_before_the_device(params, match):
    _ops()
    with pytest.raises(ValueError, match=match):
        cake.prebuild("dense_gemm", device="cpu", **params)


def test_prebuild_takes_shapes_only():
    _ops()
    assert cake.prebuild("dense_gemm") == []  # no GEMM, nothing to build; no device touched
    with pytest.raises(TypeError):
        cake.prebuild("dense_gemm", token_counts=(1001,))
    with pytest.raises(ValueError, match="run on CUDA devices"):
        cake.prebuild("dense_gemm", device="cpu", router=[(256, 4096)])


def _text(doc: str) -> str:
    """``doc`` with its line breaks and indentation as single spaces."""
    return " ".join(doc.split())


def test_the_docs_state_that_the_results_are_not_batch_invariant():
    assert "Neither variant is batch-invariant" in _text(dense_gemm.__doc__)
    ops = _ops()
    assert "Neither variant is batch-invariant" in _text(ops.__doc__)
    assert "the top-k of the router logits" in _text(ops.__doc__)  # routing can depend on the packing of the tokens
    for entry in (ops.batched_projection, ops.router_gating):
        assert re.search(r"depends on the (token count|number of tokens)", _text(entry.__doc__)), entry.__name__
