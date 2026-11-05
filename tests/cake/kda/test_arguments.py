"""The arguments of the KDA op (torch-free): ``--cake-kda-variant`` is hidden and carries ``det`` (miles' calls take
the in-kernel q / k L2 norm, which the ``fast`` package does not serve), the settings ``--kda-backend cake`` rejects
at startup, the variants a direct caller configures, and an import without torch."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import kda

REPO_ROOT = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    """miles' parser as far as the op reads it: its selector and settings, and the hooks' arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kda-backend", default="fla")
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--qkv-format", default="thd")
    parser.add_argument("--deterministic-mode", action="store_true")
    miles_ext_kernels.add_arguments(parser)
    return parser


def _validated(*argv: str) -> argparse.Namespace:
    args = _parser().parse_args(list(argv))
    miles_ext_kernels.validate_args(args)
    return args


def test_the_op_ships_both_variants_and_lets_miles_choose_det_only():
    spec = cake.op_spec("kda")

    assert spec is kda.OP
    assert spec.variants == ("det", "fast") and spec.cli_choices == ("det",)
    assert spec.has_variant_flag and not spec.shows_variant_flag
    assert spec.selected_by == (("kda_backend", "cake"),) and spec.flags == ()


def test_the_variant_flag_is_registered_hidden_with_det_its_only_choice():
    parser = _parser()
    (action,) = [action for action in parser._actions if "--cake-kda-variant" in action.option_strings]

    assert action.help == argparse.SUPPRESS
    assert tuple(action.choices) == ("det",) and action.default is None
    assert "--cake-kda-variant" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--kda-backend", "cake", "--cake-kda-variant", "fast"])


@pytest.mark.parametrize("argv", [(), ("--deterministic-mode",), ("--cake-kda-variant", "det")])
def test_validate_args_writes_det_back_for_every_process(argv):
    args = _validated("--kda-backend", "cake", *argv)

    assert args.cake_kda_variant == "det"
    cake.configure(vars(args))
    assert cake.variant("kda") == "det"


def test_the_flag_needs_the_cake_backend():
    with pytest.raises(ValueError, match="--cake-kda-variant requires --kda-backend cake"):
        _validated("--cake-kda-variant", "det")
    assert _validated().cake_kda_variant is None  # not selected: left unset


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("--kda-backend", "cake"),
        ("--kda-backend", "fla", "--context-parallel-size", "2"),
        ("--kda-backend", "fla", "--qkv-format", "bshd"),
    ],
)
def test_supported_settings_validate(argv):
    _validated(*argv)


@pytest.mark.parametrize("context_parallel_size", [2, 8])
def test_cake_rejects_context_parallelism(context_parallel_size):
    pattern = f"context parallelism yet \\(--context-parallel-size {context_parallel_size}\\)"
    with pytest.raises(ValueError, match=pattern):
        _validated("--kda-backend", "cake", "--context-parallel-size", str(context_parallel_size))


def test_cake_rejects_the_bshd_layout():
    with pytest.raises(ValueError, match="--kda-backend cake needs --qkv-format thd"):
        _validated("--kda-backend", "cake", "--qkv-format", "bshd")


def test_the_op_validation_reads_a_namespace_without_mileses_other_settings():
    kda.OP.validate(argparse.Namespace(kda_backend="cake"))  # defaults: no context parallelism, thd
    kda.OP.validate(argparse.Namespace(kda_backend="cake", context_parallel_size=None))


def test_direct_callers_configure_either_variant():
    assert cake.variant("kda") == "det"  # never configured
    cake.configure(cake_kda_variant="fast")
    assert cake.variant("kda") == "fast"
    cake.configure({"cake_kda_variant": "det"})
    assert cake.variant("kda") == "det"
    cake.configure()
    assert cake.variant("kda") == "det"


def test_deterministic_mode_rejects_an_explicit_fast():
    with pytest.raises(ValueError, match="--cake-kda-variant fast is not bit-reproducible"):
        cake.configure(cake_kda_variant="fast", deterministic_mode=True)
    with pytest.raises(ValueError, match="must be one of"):
        cake.configure(cake_kda_variant="nd")


_NO_TORCH = """
import argparse
import sys

for name in ("torch", "tvm_ffi", "flashinfer"):
    sys.modules[name] = None  # any import of them raises ImportError

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import kda

parser = argparse.ArgumentParser()
parser.add_argument("--kda-backend", default="fla")
miles_ext_kernels.add_arguments(parser)
args = parser.parse_args(["--kda-backend", "cake"])
args.deterministic_mode = True
miles_ext_kernels.validate_args(args)
miles_ext_kernels.configure(vars(args))
assert cake.variant("kda") == "det" and kda.OP.name == "kda"
prefixes = ("miles_ext_kernels.cake._impl.", "miles_ext_kernels.cake._vendor.")
loaded = sorted(name for name in sys.modules if name.startswith(prefixes))
assert loaded == [], loaded
"""


def test_the_op_imports_validates_and_configures_without_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
