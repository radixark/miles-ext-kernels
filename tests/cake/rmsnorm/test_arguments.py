"""The arguments of the Cake input RMSNorm: registration, variant resolution, rejection and configure (torch-free)."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import rmsnorm

REPO_ROOT = Path(__file__).resolve().parents[3]
VARIANT = "--cake-rmsnorm-variant"


def _parse(argv: list[str], **miles_args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(argv)
    for name, value in miles_args.items():  # the miles arguments validate_args reads from the same namespace
        setattr(args, name, value)
    return args


def test_the_op_is_selected_by_the_cake_input_norm_and_declares_both_packages():
    assert rmsnorm.OP.name == "rmsnorm" and "rmsnorm" in cake.OPS
    assert rmsnorm.OP.selected_by == (("linear_attention_input_norm", "cake"),)
    assert rmsnorm.OP.variants == ("det", "fast") and rmsnorm.OP.cli_choices == ("det", "fast")
    assert rmsnorm.OP.packages == ("cake_rmsnorm_det_train", "cake_rmsnorm_nondet_train")
    assert rmsnorm.OP.flags == ()


def test_the_variant_flag_is_registered_and_shown():
    parser = argparse.ArgumentParser(add_help=False)
    miles_ext_kernels.add_arguments(parser)
    (action,) = [action for action in parser._actions if VARIANT in action.option_strings]

    assert action.choices == ("det", "fast") and action.default is None
    assert action.help != argparse.SUPPRESS and "--linear-attention-input-norm cake" in action.help


@pytest.mark.parametrize("input_norm", [None, "te", "hf"])
def test_the_defaults_validate_and_leave_the_variant_unset(input_norm):
    args = _parse([], **({} if input_norm is None else {"linear_attention_input_norm": input_norm}))

    miles_ext_kernels.validate_args(args)
    assert args.cake_rmsnorm_variant is None


@pytest.mark.parametrize(
    "deterministic_mode, expected", [(False, "fast"), (True, "det")], ids=["default", "deterministic-mode"]
)
def test_the_cake_input_norm_resolves_the_variant(deterministic_mode, expected):
    args = _parse([], linear_attention_input_norm="cake", deterministic_mode=deterministic_mode)

    miles_ext_kernels.validate_args(args)
    assert args.cake_rmsnorm_variant == expected


def test_the_typed_backend_configuration_carries_deterministic_mode_too():
    args = _parse([], linear_attention_input_norm="cake", backend=argparse.Namespace(deterministic_mode=True))

    miles_ext_kernels.validate_args(args)
    assert args.cake_rmsnorm_variant == "det"


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_explicit_variant_is_kept(variant):
    args = _parse([VARIANT, variant], linear_attention_input_norm="cake")

    miles_ext_kernels.validate_args(args)
    assert args.cake_rmsnorm_variant == variant


def test_explicit_fast_under_deterministic_mode_is_rejected():
    args = _parse([VARIANT, "fast"], linear_attention_input_norm="cake", deterministic_mode=True)

    with pytest.raises(ValueError, match="--cake-rmsnorm-variant fast is not the det variant"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("input_norm", [None, "te", "hf"])
def test_the_variant_without_the_cake_input_norm_is_rejected(input_norm):
    args = _parse([VARIANT, "det"])
    if input_norm is not None:
        args.linear_attention_input_norm = input_norm

    with pytest.raises(ValueError, match="--cake-rmsnorm-variant requires --linear-attention-input-norm cake"):
        miles_ext_kernels.validate_args(args)


def test_an_unconfigured_process_runs_det():
    """Reviewer item: the unconfigured default was fast; a process that never configured the op runs det."""
    cake.configure()

    assert cake.variant("rmsnorm") == "det"


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_applies_the_resolved_variant(variant):
    args = _parse([VARIANT, variant], linear_attention_input_norm="cake")
    miles_ext_kernels.validate_args(args)

    for _ in range(2):  # miles calls it once per layer
        miles_ext_kernels.configure({"cake_rmsnorm_variant": args.cake_rmsnorm_variant})
        assert cake.variant("rmsnorm") == variant


def test_configure_under_deterministic_mode_runs_det_and_rejects_fast():
    cake.configure(deterministic_mode=True)
    assert cake.variant("rmsnorm") == "det"

    with pytest.raises(ValueError, match="not the det variant"):
        cake.configure(cake_rmsnorm_variant="fast", deterministic_mode=True)


def test_configure_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="--cake-rmsnorm-variant must be one of"):
        cake.configure(cake_rmsnorm_variant="exact")


_NO_TORCH = """
import argparse
import sys

for name in ("torch", "tvm_ffi", "flashinfer"):
    sys.modules[name] = None  # importing it raises ImportError

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import rmsnorm

assert rmsnorm.OP.name == "rmsnorm"
parser = argparse.ArgumentParser()
miles_ext_kernels.add_arguments(parser)
args = parser.parse_args([])
args.linear_attention_input_norm = "cake"
miles_ext_kernels.validate_args(args)
assert args.cake_rmsnorm_variant == "fast", args
miles_ext_kernels.configure(vars(args))
assert cake.variant("rmsnorm") == "fast"
assert not [name for name in sys.modules if name.startswith("miles_ext_kernels.cake._impl.")]
"""


def test_the_op_and_its_arguments_import_without_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
