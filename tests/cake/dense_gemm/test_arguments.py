"""The arguments of the dense projection GEMMs (torch-free): two miles selectors, one hidden variant flag that carries
``det``, and the Megatron train backend check."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import dense_gemm

REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTORS = ["--batched-projection-gemm-backend", "--moe-router-gemm-backend"]
VARIANT = "--cake-dense-gemm-variant"


def _parser() -> argparse.ArgumentParser:
    """miles' backend flags and the deterministic mode, then the ext arguments, as miles registers them."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batched-projection-gemm-backend", default="torch")
    parser.add_argument("--moe-router-gemm-backend", default="megatron")
    parser.add_argument("--train-backend", default="megatron")
    parser.add_argument("--deterministic-mode", action="store_true")
    miles_ext_kernels.add_arguments(parser)
    return parser


def _validated(argv: list[str]) -> argparse.Namespace:
    args = _parser().parse_args(argv)
    miles_ext_kernels.validate_args(args)
    return args


def test_the_op_declares_two_selectors_two_variants_and_miles_runs_det():
    spec = cake.op_spec("dense_gemm")

    assert spec is dense_gemm.OP
    assert spec.selected_by == (("batched_projection_gemm_backend", "cake"), ("moe_router_gemm_backend", "cake"))
    assert spec.variants == ("det", "fast") and spec.cli_choices == ("det",)
    assert spec.packages == ("cake_dense_gemm_det_train", "cake_dense_gemm_nondet_train")
    assert spec.flags == ()


def test_the_variant_flag_is_hidden_and_offers_det_only():
    parser = _parser()
    action = parser._option_string_actions[VARIANT]

    assert action.help == argparse.SUPPRESS and tuple(action.choices) == ("det",)
    assert VARIANT not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args([VARIANT, "fast"])


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("deterministic", [False, True])
def test_a_selected_op_resolves_det_and_writes_it_back(selector, deterministic):
    args = _validated([selector, "cake", *(["--deterministic-mode"] if deterministic else [])])

    assert args.cake_dense_gemm_variant == "det"
    cake.configure(vars(args))
    assert cake.variant("dense_gemm") == "det"


def test_the_flag_of_an_op_that_is_not_selected_is_rejected():
    assert _validated([]).cake_dense_gemm_variant is None
    with pytest.raises(ValueError, match="--cake-dense-gemm-variant requires --batched-projection-gemm-backend cake"):
        _validated([VARIANT, "det"])


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("train_backend", ["fsdp", "torchtitan"])
def test_a_cake_backend_outside_megatron_is_rejected(selector, train_backend):
    with pytest.raises(ValueError, match=f"{selector} cake replaces a GEMM of miles' Megatron model code"):
        _validated([selector, "cake", "--train-backend", train_backend])


def test_a_namespace_without_a_train_backend_is_rejected():
    args = argparse.Namespace(moe_router_gemm_backend="cake", cake_dense_gemm_variant=None)

    with pytest.raises(ValueError, match=r"requires --train-backend megatron \(got None\)"):
        miles_ext_kernels.validate_args(args)


def test_configure_reaches_fast_for_direct_callers_but_not_in_deterministic_mode():
    assert cake.variant("dense_gemm") == "det"  # never configured
    cake.configure(cake_dense_gemm_variant="fast")
    assert cake.variant("dense_gemm") == "fast"
    with pytest.raises(ValueError, match="not the det variant that --deterministic-mode runs"):
        cake.configure(cake_dense_gemm_variant="fast", deterministic_mode=True)
    with pytest.raises(ValueError, match="must be one of"):
        cake.configure(cake_dense_gemm_variant="nd")


_WITHOUT_TORCH = """
import argparse
import sys

for blocked in ("torch", "tvm_ffi", "flashinfer"):
    sys.modules[blocked] = None  # importing it raises ImportError

import miles_ext_kernels
from miles_ext_kernels.cake import dense_gemm

assert "router_gating" in dir(dense_gemm) and "batched_projection" in dir(dense_gemm)
parser = argparse.ArgumentParser()
parser.add_argument("--moe-router-gemm-backend", default="cake")
miles_ext_kernels.add_arguments(parser)
args = parser.parse_args([])
args.train_backend = "megatron"
miles_ext_kernels.validate_args(args)
miles_ext_kernels.configure(args)
assert args.cake_dense_gemm_variant == "det"
assert not [name for name in sys.modules if name.startswith("miles_ext_kernels.cake._impl.")]
print("ok")
"""


def test_the_op_module_and_its_arguments_need_no_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_TORCH],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
