"""The conv1d op's arguments and configuration through the runtime (torch-free): registration, variant resolution,
the SiLU' option, the rejection rules and the variant a process runs before and after ``configure``."""

import argparse
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import conv1d

REPO_ROOT = Path(__file__).resolve().parents[3]
VARIANT = "--cake-conv1d-variant"
SILU_GRAD = "--cake-conv1d-silu-grad-preactivation"
SELECTOR = "--linear-attention-conv-backend cake"


def _parse(argv: list[str], **miles_args) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(argv)
    for name, value in miles_args.items():  # the miles arguments validate_args reads from the same namespace
        setattr(args, name, value)
    return args


def _miles_flow(argv: list[str], **miles_args) -> dict:
    """Parse, validate on the flat namespace, move the registered values into one field, configure a process."""
    args = _parse(argv, **miles_args)
    miles_ext_kernels.validate_args(args)
    registered = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(registered)
    values = {action.dest: getattr(args, action.dest) for action in registered._actions if action.dest != "help"}
    miles_ext_kernels.configure(values)
    return values


def test_the_op_declares_its_selector_packages_and_entry_points():
    spec = cake.op_spec("conv1d")

    assert spec is conv1d.OP
    assert spec.variants == ("det", "fast") and spec.cli_choices == ("det", "fast")
    assert spec.selected_by == (("linear_attention_conv_backend", "cake"),)
    assert spec.packages == ("cake_causal_conv1d_det_train_backend", "cake_causal_conv1d_nondet_train")
    assert conv1d.__all__ == ["OP", "causal_conv1d", "causal_conv1d_kcp", "causal_conv1d_kcp_multi", "check_supported"]
    assert [flag.name for flag in spec.flags] == [SILU_GRAD]


def test_the_flags_are_registered_and_unset_by_default():
    args = _parse([])

    assert args.cake_conv1d_variant is None
    assert args.cake_conv1d_silu_grad_preactivation == "bf16"
    miles_ext_kernels.validate_args(args)
    assert args.cake_conv1d_variant is None


@pytest.mark.parametrize(
    "deterministic_mode, expected", [(False, "fast"), (True, "det")], ids=["default", "deterministic-mode"]
)
def test_the_cake_backend_resolves_the_variant_and_the_silu_policy(deterministic_mode, expected):
    args = _parse([], linear_attention_conv_backend="cake", deterministic_mode=deterministic_mode)

    miles_ext_kernels.validate_args(args)
    assert args.cake_conv1d_variant == expected
    assert args.cake_conv1d_silu_grad_preactivation == "bf16"


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("policy", ["bf16", "fp32"])
def test_explicit_settings_are_kept(variant, policy):
    args = _parse([VARIANT, variant, SILU_GRAD, policy], linear_attention_conv_backend="cake")

    miles_ext_kernels.validate_args(args)
    assert (args.cake_conv1d_variant, args.cake_conv1d_silu_grad_preactivation) == (variant, policy)


def test_the_typed_config_carries_deterministic_mode_too():
    args = _parse([], linear_attention_conv_backend="cake")
    args.backend = types.SimpleNamespace(deterministic_mode=True)

    miles_ext_kernels.validate_args(args)
    assert args.cake_conv1d_variant == "det"


def test_explicit_fast_under_deterministic_mode_is_rejected():
    args = _parse([VARIANT, "fast"], linear_attention_conv_backend="cake", deterministic_mode=True)

    with pytest.raises(ValueError, match="--cake-conv1d-variant fast is not bit-reproducible"):
        miles_ext_kernels.validate_args(args)


def test_an_unknown_silu_policy_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        _parse([SILU_GRAD, "fp16"])


@pytest.mark.parametrize("backend", [None, "fla"])
@pytest.mark.parametrize(
    "argv, flag", [([VARIANT, "det"], VARIANT), ([SILU_GRAD, "fp32"], SILU_GRAD)], ids=["variant", "silu-grad"]
)
def test_an_argument_without_the_cake_backend_is_rejected(backend, argv, flag):
    args = _parse(argv, linear_attention_conv_backend=backend)

    with pytest.raises(ValueError, match=f"{flag} requires {SELECTOR}"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("backend", [None, "fla"])
def test_the_default_silu_policy_without_the_cake_backend_is_accepted(backend):
    """The option defaults to bf16 (it had no default before), so the parser cannot tell an explicit bf16 from no
    flag: only a value other than the default requires the cake backend."""
    args = _parse([SILU_GRAD, "bf16"], linear_attention_conv_backend=backend)

    miles_ext_kernels.validate_args(args)
    assert args.cake_conv1d_silu_grad_preactivation == "bf16"


def test_context_parallelism_is_rejected():
    args = _parse([], linear_attention_conv_backend="cake", context_parallel_size=2)

    with pytest.raises(ValueError, match="does not support context parallelism"):
        miles_ext_kernels.validate_args(args)
    # without the cake backend, context parallelism is not this op's concern
    miles_ext_kernels.validate_args(_parse([], linear_attention_conv_backend="fla", context_parallel_size=2))
    miles_ext_kernels.validate_args(_parse([], linear_attention_conv_backend="cake", context_parallel_size=1))


# --- the variant a process runs ---


def test_a_process_that_never_configured_runs_det():
    """The unconfigured default is det, as for every op (it was fast before)."""
    assert cake.variant("conv1d") == "det"
    assert cake.option("conv1d", "cake_conv1d_silu_grad_preactivation") == "bf16"


@pytest.mark.parametrize("args", [None, argparse.Namespace(), {}], ids=["none", "namespace", "mapping"])
def test_configure_without_the_op_s_values_runs_det(args):
    cake.configure(cake_conv1d_variant="fast")
    assert cake.variant("conv1d") == "fast"

    cake.configure(args)
    assert cake.variant("conv1d") == "det"
    assert cake.option("conv1d", "cake_conv1d_silu_grad_preactivation") == "bf16"


@pytest.mark.parametrize(
    "argv, miles_args, variant",
    [
        ([], {}, "fast"),
        ([], {"deterministic_mode": True}, "det"),
        ([VARIANT, "det"], {}, "det"),
    ],
    ids=["cli-default", "deterministic-mode", "explicit-det"],
)
def test_miles_command_line_reaches_every_process(argv, miles_args, variant):
    """miles' command line defaults to fast: validate_args writes the resolved variant back, configure applies it."""
    values = _miles_flow([*argv, SILU_GRAD, "fp32"], linear_attention_conv_backend="cake", **miles_args)

    assert values["cake_conv1d_variant"] == variant
    assert cake.variant("conv1d") == variant
    assert cake.option("conv1d", "cake_conv1d_silu_grad_preactivation") == "fp32"


def test_configure_rejects_an_unknown_variant_and_fast_in_deterministic_mode():
    with pytest.raises(ValueError, match="--cake-conv1d-variant must be one of"):
        cake.configure(cake_conv1d_variant="exact")
    with pytest.raises(ValueError, match="not bit-reproducible"):
        cake.configure(cake_conv1d_variant="fast", deterministic_mode=True)
    cake.configure(deterministic_mode=True)
    assert cake.variant("conv1d") == "det"


_NO_TORCH = """
import argparse
import sys

for name in ("torch", "tvm_ffi", "flashinfer"):
    sys.modules[name] = None  # any import of them raises ImportError

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import conv1d

parser = argparse.ArgumentParser()
miles_ext_kernels.add_arguments(parser)
args = parser.parse_args([])
args.linear_attention_conv_backend = "cake"
miles_ext_kernels.validate_args(args)
assert args.cake_conv1d_variant == "fast", args
miles_ext_kernels.configure(args)
assert cake.variant("conv1d") == "fast"
assert conv1d.OP.name == "conv1d" and "causal_conv1d" in dir(conv1d)
"""


def test_the_op_module_and_its_arguments_import_without_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
