"""The mHC arguments and variant rules, through the hooks miles calls (torch-free)."""

import argparse
import types

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import mhc

FLAG = "--cake-mhc-variant"


def _parse(argv: list[str], **attributes) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(argv)
    for name, value in attributes.items():
        setattr(args, name, value)
    return args


def test_the_op_declares_both_variants_the_selector_and_the_packages():
    assert mhc.OP.name == "mhc" and cake.op_spec("mhc") is mhc.OP
    assert mhc.OP.variants == ("det", "fast") and mhc.OP.cli_choices == ("det", "fast")
    assert mhc.OP.selected_by == (("mhc_backend", "cake"),)
    assert mhc.OP.variant_flag == FLAG and mhc.OP.flags == ()
    assert set(mhc.OP.packages) == {
        "cake_mhc_det_train",
        "cake_mhc_head_det_train",
        "cake_mhc_head_nondet_train",
        "cake_mhc_nondet_train",
    }
    assert mhc.__all__ == ["OP", *mhc.OP.exports]
    assert {"mhc_prepare", "mhc_finish", "mhc_head", "is_supported", "HIDDEN_SIZES"} <= set(dir(mhc))
    with pytest.raises(AttributeError):
        mhc.no_such_name


def test_the_flag_is_registered_through_the_top_level_hook_and_the_defaults_validate():
    args = _parse([])

    assert args.cake_mhc_variant is None
    miles_ext_kernels.validate_args(args)
    assert args.cake_mhc_variant is None  # without --mhc-backend cake the variant stays unset


@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_unset_resolves_to_fast_or_to_det_under_deterministic_mode(deterministic, variant):
    args = _parse([], mhc_backend="cake", deterministic_mode=deterministic)

    miles_ext_kernels.validate_args(args)
    assert args.cake_mhc_variant == variant


def test_deterministic_mode_of_the_typed_configuration_resolves_to_det():
    args = _parse([], mhc_backend="cake", backend=types.SimpleNamespace(deterministic_mode=True))

    miles_ext_kernels.validate_args(args)
    assert args.cake_mhc_variant == "det"


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_explicit_variant_is_kept_outside_deterministic_mode(variant):
    args = _parse([FLAG, variant], mhc_backend="cake", deterministic_mode=False)

    miles_ext_kernels.validate_args(args)
    assert args.cake_mhc_variant == variant


def test_explicit_fast_is_rejected_under_deterministic_mode_and_det_is_kept():
    # fast adds the dfn partials with FP32 atomics: deterministic mode cannot run it
    with pytest.raises(ValueError, match="--cake-mhc-variant fast is not bit-reproducible under --deterministic-mode"):
        miles_ext_kernels.validate_args(_parse([FLAG, "fast"], mhc_backend="cake", deterministic_mode=True))

    args = _parse([FLAG, "det"], mhc_backend="cake", deterministic_mode=True)
    miles_ext_kernels.validate_args(args)
    assert args.cake_mhc_variant == "det"


@pytest.mark.parametrize("backend", [None, "megatron"])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_the_variant_without_the_cake_backend_is_rejected(backend, variant):
    args = _parse([FLAG, variant])
    if backend is not None:
        args.mhc_backend = backend

    with pytest.raises(ValueError, match="--cake-mhc-variant requires --mhc-backend cake"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("hidden", [None, 4096, 5120, 7168])
def test_the_kernels_hidden_sizes_validate(hidden):
    miles_ext_kernels.validate_args(_parse([], mhc_backend="cake", hidden_size=hidden))


def test_another_hidden_size_is_rejected_only_under_the_cake_backend():
    with pytest.raises(ValueError, match=r"--mhc-backend cake runs the hidden sizes \(4096, 5120, 7168\)"):
        miles_ext_kernels.validate_args(_parse([], mhc_backend="cake", hidden_size=2048))

    miles_ext_kernels.validate_args(_parse([], mhc_backend="megatron", hidden_size=2048))


# --- the variant a process runs (review: deterministic mode without a variant ran fast) ---


def test_an_unconfigured_process_runs_det():
    assert cake.variant("mhc") == "det"
    cake.configure()
    assert cake.variant("mhc") == "det"


@pytest.mark.parametrize(
    "args",
    [
        argparse.Namespace(deterministic_mode=True),
        argparse.Namespace(deterministic_mode=True, cake_mhc_variant=None),
        argparse.Namespace(backend=types.SimpleNamespace(deterministic_mode=True)),
        {"deterministic_mode": True},
    ],
)
def test_configure_in_deterministic_mode_without_a_variant_runs_det(args):
    cake.configure(cake_mhc_variant="fast")
    assert cake.variant("mhc") == "fast"

    cake.configure(args)  # replaces the whole configuration
    assert cake.variant("mhc") == "det"


def test_configure_in_deterministic_mode_rejects_an_explicit_fast():
    with pytest.raises(ValueError, match="--cake-mhc-variant fast is not bit-reproducible"):
        cake.configure(argparse.Namespace(deterministic_mode=True, cake_mhc_variant="fast"))
    with pytest.raises(ValueError, match="--cake-mhc-variant fast is not bit-reproducible"):
        cake.configure({"deterministic_mode": True}, cake_mhc_variant="fast")


def test_configure_sets_the_variant_from_miles_values_and_rejects_another():
    for variant in ("det", "fast", "det"):  # miles may configure once per process or per layer
        miles_ext_kernels.configure({"cake_mhc_variant": variant, "cake_workspace_pool_bytes": 1 << 30})
        assert cake.variant("mhc") == variant
    miles_ext_kernels.configure({})
    assert cake.variant("mhc") == "det"
    with pytest.raises(ValueError, match="--cake-mhc-variant must be one of"):
        cake.configure(cake_mhc_variant="fastest")


@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_the_resolved_variant_reaches_a_process_through_miles_ext_kernels_args(deterministic, variant):
    """miles validates the flat namespace, moves the registered values into one field and configures with it."""
    args = _parse([], mhc_backend="cake", deterministic_mode=deterministic)
    miles_ext_kernels.validate_args(args)
    registered = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(registered)
    values = {action.dest: getattr(args, action.dest) for action in registered._actions if action.dest != "help"}

    miles_ext_kernels.configure(values)
    assert cake.variant("mhc") == variant
