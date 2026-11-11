"""The declaration of ``gated_act_mxfp8``, its (absent) arguments and its variant, torch-free.

One variant, ``det``: the outputs are a function of the inputs only. No miles argument selects the op, so it
registers no flag, and the runtime runs ``det`` whatever the mode.
"""

import argparse
import ast
from pathlib import Path

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import gated_act_mxfp8

IMPL = Path(gated_act_mxfp8.__file__).resolve().parent / "_impl" / "gated_act_mxfp8"
ENTRY_POINTS = ("silu_and_mul_mxfp8_quantize", "silu_and_mul_mxfp8_quantize_backward")


def test_the_op_is_declared_with_one_det_variant_and_no_miles_selector():
    op = gated_act_mxfp8.OP

    assert "gated_act_mxfp8" in cake.OPS
    assert cake.op_spec("gated_act_mxfp8") is op
    assert op.variants == ("det",)
    assert op.cli_choices == ("det",)
    assert op.selected_by == () and op.flags == () and op.validate is None
    assert not op.has_variant_flag
    assert op.packages == ("cake_gated_act_mxfp8_det_train",)
    assert op.exports == ENTRY_POINTS
    assert gated_act_mxfp8.__all__ == ["OP", *ENTRY_POINTS]


def test_no_argument_is_registered():
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)

    assert [action.dest for action in parser._actions if "gated_act" in action.dest] == []
    assert not any("gated-act" in option for action in parser._actions for option in action.option_strings)


@pytest.mark.parametrize("deterministic_mode", [False, True], ids=["default", "deterministic-mode"])
def test_every_mode_runs_det(deterministic_mode):
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args([])
    args.deterministic_mode = deterministic_mode
    before = dict(vars(args))

    miles_ext_kernels.validate_args(args)
    miles_ext_kernels.configure(args)

    assert vars(args) == before  # nothing resolved or written back for an op miles does not select
    assert cake.variant("gated_act_mxfp8") == "det"


def test_configure_takes_det_and_rejects_fast():
    cake.configure(cake_gated_act_mxfp8_variant="det", deterministic_mode=True)
    assert cake.variant("gated_act_mxfp8") == "det"

    with pytest.raises(ValueError, match="must be one of"):
        cake.configure(cake_gated_act_mxfp8_variant="fast")
    with pytest.raises(KeyError, match="no option"):
        cake.option("gated_act_mxfp8", "cake_gated_act_mxfp8_chunk_size")


def test_every_entry_point_is_defined_by_the_implementation():
    """The lazy exports resolve to names ``_impl/gated_act_mxfp8`` binds (checked here without importing torch)."""
    bound = set()
    for path in (IMPL / "__init__.py", IMPL / "ops.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.ImportFrom):
                bound.update(alias.asname or alias.name for alias in node.names)

    assert {*ENTRY_POINTS, "prebuild", "release"} <= bound


def test_an_unknown_attribute_raises_attribute_error():
    # prebuild is cake.prebuild("gated_act_mxfp8"), not an export of the op module
    with pytest.raises(AttributeError, match="has no attribute 'prebuild'"):
        gated_act_mxfp8.prebuild  # noqa: B018
    assert set(ENTRY_POINTS) <= set(dir(gated_act_mxfp8))
