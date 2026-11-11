"""The gdn arguments on the runtime's argument surface, as miles registers, validates and configures them (torch-free).

``--cake-gdn-variant`` and ``--cake-gdn-recompute-state`` exist only with ``--linear-attention-backend cake``. An unset
variant resolves to ``fast`` (miles' command line default), to ``det`` under ``--deterministic-mode``, which rejects an
explicit ``fast``; a process that never configured gdn runs ``det``. The kernels read both values when a call runs.
"""

import argparse
import types

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import gdn
from miles_ext_kernels.cake._runtime import config

RECOMPUTE_REJECTION = "--cake-gdn-recompute-state requires --linear-attention-backend cake"
VARIANT_REJECTION = "--cake-gdn-variant requires --linear-attention-backend cake"
DETERMINISTIC_REJECTION = "--cake-gdn-variant fast is not the det variant that --deterministic-mode runs"


def _parse(argv: list[str], **attributes) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(argv)
    for name, value in attributes.items():  # miles' own arguments of the flat namespace it validates
        setattr(args, name, value)
    return args


def test_the_op_is_registered_with_its_flags_and_the_defaults_validate():
    assert "gdn" in cake.OPS and cake.op_spec("gdn") is gdn.OP
    args = _parse([])

    assert args.cake_gdn_recompute_state is False
    assert args.cake_gdn_variant is None
    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_variant is None  # no cake backend: nothing to resolve


@pytest.mark.parametrize("backend", [None, "fla", "flashqla"])
def test_recompute_state_without_the_cake_backend_is_rejected(backend):
    args = _parse(["--cake-gdn-recompute-state"])
    if backend is not None:
        args.linear_attention_backend = backend

    with pytest.raises(ValueError, match=RECOMPUTE_REJECTION):
        miles_ext_kernels.validate_args(args)


def test_recompute_state_with_the_cake_backend_is_accepted():
    args = _parse(["--cake-gdn-recompute-state"], linear_attention_backend="cake")

    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_recompute_state is True


@pytest.mark.parametrize("variant", ["det", "fast"])
@pytest.mark.parametrize("backend", [None, "fla", "flashqla"])
def test_a_variant_without_the_cake_backend_is_rejected(backend, variant):
    args = _parse(["--cake-gdn-variant", variant])
    if backend is not None:
        args.linear_attention_backend = backend

    with pytest.raises(ValueError, match=VARIANT_REJECTION):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_unset_resolves_to_fast_or_to_det_under_deterministic_mode(deterministic, variant):
    args = _parse([], linear_attention_backend="cake", deterministic_mode=deterministic)

    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_variant == variant


def test_deterministic_mode_of_the_typed_configuration_resolves_to_det():
    args = _parse([], linear_attention_backend="cake", backend=types.SimpleNamespace(deterministic_mode=True))

    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_variant == "det"


@pytest.mark.parametrize("deterministic", [False, True])
def test_an_explicit_det_is_kept(deterministic):
    args = _parse(["--cake-gdn-variant", "det"], linear_attention_backend="cake", deterministic_mode=deterministic)

    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_variant == "det"


def test_an_explicit_fast_is_kept_and_rejected_under_deterministic_mode():
    # fast does not repeat its bits run to run: deterministic mode runs det only
    args = _parse(["--cake-gdn-variant", "fast"], linear_attention_backend="cake")
    miles_ext_kernels.validate_args(args)
    assert args.cake_gdn_variant == "fast"

    args = _parse(["--cake-gdn-variant", "fast"], linear_attention_backend="cake", deterministic_mode=True)
    with pytest.raises(ValueError, match=DETERMINISTIC_REJECTION):
        miles_ext_kernels.validate_args(args)
    with pytest.raises(ValueError, match=DETERMINISTIC_REJECTION):
        miles_ext_kernels.configure(cake_gdn_variant="fast", deterministic_mode=True)


def test_an_unknown_variant_is_rejected_by_the_parser_and_by_configure():
    with pytest.raises(SystemExit):
        _parse(["--cake-gdn-variant", "fastest"])
    with pytest.raises(ValueError, match="--cake-gdn-variant must be one of"):
        miles_ext_kernels.configure(argparse.Namespace(cake_gdn_variant="fastest"))


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_configure_sets_the_variant_and_the_recompute_switch_the_kernels_read(recompute, variant):
    argv = ["--cake-gdn-variant", variant] + (["--cake-gdn-recompute-state"] if recompute else [])
    args = _parse(argv, linear_attention_backend="cake")
    miles_ext_kernels.validate_args(args)

    for _ in range(2):  # miles calls it once per layer
        miles_ext_kernels.configure(args)
        assert config.variant("gdn") == variant
        assert config.option("gdn", "cake_gdn_recompute_state") is recompute


def test_without_the_arguments_the_kernels_keep_the_tape_and_run_det():
    # a process that never configured gdn, or a namespace without the arguments (never registered, never validated)
    assert config.variant("gdn") == "det"
    assert config.option("gdn", "cake_gdn_recompute_state") is False

    miles_ext_kernels.configure(argparse.Namespace())
    assert config.variant("gdn") == "det"
    miles_ext_kernels.configure(argparse.Namespace(cake_gdn_variant=None))
    assert config.variant("gdn") == "det"


@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_the_resolved_variant_reaches_configure_as_miles_hands_it_over(deterministic, variant):
    # miles validates the flat namespace, then moves the registered values into miles_ext_kernels_args and configures
    # every process that builds a cake GDN layer with them
    args = _parse([], linear_attention_backend="cake", deterministic_mode=deterministic)
    miles_ext_kernels.validate_args(args)
    registered = argparse.ArgumentParser(add_help=False)
    miles_ext_kernels.add_arguments(registered)
    ext_args = {action.dest: vars(args).pop(action.dest) for action in registered._actions}

    miles_ext_kernels.configure(ext_args)
    assert config.variant("gdn") == variant
    assert config.option("gdn", "cake_gdn_recompute_state") is False


def test_the_recompute_help_states_the_accuracy_and_the_bits():
    (flag,) = gdn.OP.flags
    text = " ".join(flag.kwargs["help"].split())

    assert flag.name == "--cake-gdn-recompute-state" and flag.default is False
    assert "within the BF16 accuracy budgets" in text
    assert "Under det each setting repeats bit for bit." in text
