"""The MoE grouped GEMM's arguments through miles' hooks (torch-free): its variant flag, resolved as miles' deterministic
mode and the torchtitan backend require, and the variant a process runs once configured."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import moe_grouped_gemm

REPO_ROOT = Path(__file__).resolve().parents[3]
OP = "moe_grouped_gemm"
DEST = "cake_moe_grouped_gemm_variant"


def _parse(argv: list[str], **values) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(argv)
    for name, value in values.items():
        setattr(args, name, value)
    return args


def test_the_op_declares_its_variants_selector_and_packages():
    spec = moe_grouped_gemm.OP

    assert cake.op_spec(OP) is spec and OP in cake.OPS
    assert spec.variants == ("det", "fast") and spec.cli_choices == ("det", "fast")
    assert spec.selected_by == (("moe_grouped_gemm_backend", "cake"),)
    assert spec.packages == ("cake_moe_grouped_gemm_det_train", "cake_moe_grouped_gemm_nondet_train")
    assert moe_grouped_gemm.__all__ == ["OP", "check_grouped_dims", "grouped_mm", "max_grouped_rows"]
    assert spec.flags == () and spec.variant_flag == "--cake-moe-grouped-gemm-variant"


def test_the_variant_flag_is_registered_unset():
    args = _parse([])

    assert getattr(args, DEST) is None
    with pytest.raises(SystemExit):
        _parse(["--cake-moe-grouped-gemm-variant", "fastest"])


def test_without_the_cake_backend_the_variant_stays_unset():
    args = _parse([], moe_grouped_gemm_backend="torch", train_backend="megatron")

    miles_ext_kernels.validate_args(args)

    assert getattr(args, DEST) is None


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_variant_without_the_cake_backend_is_rejected(variant):
    args = _parse(["--cake-moe-grouped-gemm-variant", variant], moe_grouped_gemm_backend="torch")

    with pytest.raises(ValueError, match="requires --moe-grouped-gemm-backend cake"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize(
    "argv, deterministic_mode, expected",
    [
        ([], None, "fast"),
        ([], False, "fast"),
        ([], True, "det"),
        (["--cake-moe-grouped-gemm-variant", "det"], None, "det"),
        (["--cake-moe-grouped-gemm-variant", "det"], True, "det"),
        (["--cake-moe-grouped-gemm-variant", "fast"], None, "fast"),
    ],
)
def test_the_cake_backend_resolves_the_variant_and_writes_it_back(argv, deterministic_mode, expected):
    args = _parse(argv, moe_grouped_gemm_backend="cake", train_backend="torchtitan")
    if deterministic_mode is not None:
        args.deterministic_mode = deterministic_mode

    miles_ext_kernels.validate_args(args)

    assert getattr(args, DEST) == expected


def test_an_explicit_fast_variant_under_deterministic_mode_is_rejected():
    args = _parse(
        ["--cake-moe-grouped-gemm-variant", "fast"],
        moe_grouped_gemm_backend="cake",
        train_backend="torchtitan",
        deterministic_mode=True,
    )

    with pytest.raises(ValueError, match="not the det variant"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("train_backend", ["megatron", "fsdp", None])
def test_the_cake_backend_outside_torchtitan_is_rejected(train_backend):
    args = _parse([], moe_grouped_gemm_backend="cake")
    if train_backend is not None:
        args.train_backend = train_backend

    with pytest.raises(ValueError, match="requires --train-backend torchtitan"):
        miles_ext_kernels.validate_args(args)


def test_an_unconfigured_process_runs_det():
    assert cake.variant(OP) == "det"


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_sets_the_variant_the_next_call_runs(variant):
    args = _parse(
        ["--cake-moe-grouped-gemm-variant", variant], moe_grouped_gemm_backend="cake", train_backend="torchtitan"
    )
    miles_ext_kernels.validate_args(args)
    kept = {key: value for key, value in vars(args).items() if key.startswith("cake_")}  # miles_ext_kernels_args

    miles_ext_kernels.configure(kept)
    miles_ext_kernels.configure(kept)  # idempotent

    assert cake.variant(OP) == variant
    miles_ext_kernels.configure()
    assert cake.variant(OP) == "det"


def test_configure_refuses_fast_in_deterministic_mode_and_unknown_variants():
    with pytest.raises(ValueError, match="not the det variant"):
        cake.configure(**{DEST: "fast"}, deterministic_mode=True)
    with pytest.raises(ValueError, match="must be one of"):
        cake.configure(**{DEST: "fastest"})
    cake.configure(deterministic_mode=True)
    assert cake.variant(OP) == "det"


_WITHOUT_TORCH = """
import argparse
import sys

for name in ("torch", "tvm_ffi", "flashinfer"):
    sys.modules[name] = None  # importing it raises ImportError

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import moe_grouped_gemm

assert {"grouped_mm", "check_grouped_dims", "max_grouped_rows"} <= set(dir(moe_grouped_gemm))
parser = argparse.ArgumentParser()
miles_ext_kernels.add_arguments(parser)
args = parser.parse_args(["--cake-moe-grouped-gemm-variant", "det"])
args.moe_grouped_gemm_backend, args.train_backend = "cake", "torchtitan"
miles_ext_kernels.validate_args(args)
miles_ext_kernels.configure(args)
assert cake.variant("moe_grouped_gemm") == "det"
assert not [name for name in sys.modules if name.startswith("miles_ext_kernels.cake._impl.")]
print("ok")
"""


def test_the_op_and_its_arguments_import_without_torch():
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
