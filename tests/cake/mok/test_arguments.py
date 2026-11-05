"""The MoK op module: its declaration, arguments and lazy entry points (torch-free).

MoK ships one variant, ``det`` (its outputs repeat bit for bit), and no miles argument selects it: it registers no
argument, deterministic mode allows it, and ``configure`` accepts ``det`` only.
"""

import argparse

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import mok


def test_the_op_is_registered_with_one_det_variant_and_no_selector():
    assert "mok" in cake.OPS
    assert cake.op_spec("mok") is mok.OP
    assert mok.OP.variants == ("det",)
    assert mok.OP.selected_by == () and mok.OP.flags == ()
    assert mok.OP.packages == ("cake_mok_bf16",)


def test_no_argument_is_registered():
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)

    assert not [action.dest for action in parser._actions if action.dest.startswith("cake_mok")]


@pytest.mark.parametrize("deterministic_mode", [False, True], ids=["default", "deterministic-mode"])
def test_every_mode_runs_det(deterministic_mode):
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args([])
    args.deterministic_mode = deterministic_mode

    miles_ext_kernels.validate_args(args)
    assert not hasattr(args, "cake_mok_variant")
    miles_ext_kernels.configure(args)
    assert cake.variant("mok") == "det"


def test_configure_accepts_det_only():
    cake.configure(cake_mok_variant="det", deterministic_mode=True)
    assert cake.variant("mok") == "det"
    with pytest.raises(ValueError, match="must be one of"):
        cake.configure(cake_mok_variant="fast")


def test_the_entry_points_are_lazy():
    assert mok.__all__ == ["OP", "create_mok_bf16_workspace", "prepare_mok_bf16", "quantize_mok_mxfp8_weights"]
    assert set(mok.__all__) <= set(dir(mok))
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        mok.nope  # noqa: B018
