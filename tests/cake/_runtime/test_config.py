"""The one argument surface and the process configuration (torch-free), on the synthetic op ``synth``."""

import argparse
import dataclasses
import logging
from types import SimpleNamespace

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake._runtime import config, registry
from miles_ext_kernels.cake._runtime.spec import Flag

from . import synthetic

VARIANT = "--cake-synth-variant"


@pytest.fixture(autouse=True)
def _no_other_ops(monkeypatch):
    """These tests see ``synth`` alone, whatever ops the tree registers."""
    monkeypatch.setattr(registry, "OPS", ())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--synth-backend", default="torch")
    parser.add_argument("--deterministic-mode", action="store_true")
    miles_ext_kernels.add_arguments(parser)
    return parser


def _validated(argv: list[str]) -> argparse.Namespace:
    args = _parser().parse_args(argv)
    miles_ext_kernels.validate_args(args)
    return args


def _install(monkeypatch, **changes):
    return synthetic.install(monkeypatch, dataclasses.replace(synthetic.OP, **changes))


def test_the_top_level_registers_one_group_with_every_option(synth):
    parser = _parser()
    (group,) = [group for group in parser._action_groups if group.title == "miles-ext-kernels"]

    assert [action.option_strings[0] for action in group._group_actions] == [
        VARIANT,
        "--cake-synth-scale",
        "--cake-workspace-pool-bytes",
    ]
    assert parser.parse_args([]).cake_synth_variant is None
    assert parser.parse_args([VARIANT, "det"]).cake_synth_variant == "det"
    with pytest.raises(SystemExit):
        parser.parse_args([VARIANT, "fastest"])


def test_without_ops_only_the_runtime_options_are_registered():
    parser = argparse.ArgumentParser(add_help=False)
    miles_ext_kernels.add_arguments(parser)

    assert [action.dest for action in parser._actions] == ["cake_workspace_pool_bytes"]
    miles_ext_kernels.validate_args(parser.parse_args([]))


def test_an_op_without_a_selector_or_a_second_variant_registers_no_variant_flag(monkeypatch):
    _install(monkeypatch, selected_by=())
    parser = _parser()
    assert VARIANT not in parser._option_string_actions and "--cake-synth-scale" not in parser._option_string_actions

    _install(monkeypatch, variants=("fast",))
    assert VARIANT not in _parser()._option_string_actions


@pytest.mark.parametrize("choice", ["det", "fast"])
def test_a_single_cli_choice_hides_the_flag_but_keeps_its_value(monkeypatch, choice):
    _install(monkeypatch, cli_variants=(choice,))
    parser = _parser()

    assert parser._option_string_actions[VARIANT].help == argparse.SUPPRESS
    assert VARIANT not in parser.format_help()
    assert parser.parse_args([]).cake_synth_variant is None
    with pytest.raises(SystemExit):
        parser.parse_args([VARIANT, "fast" if choice == "det" else "det"])


def test_the_help_names_the_selector_and_the_default(synth):
    help_text = _parser()._option_string_actions[VARIANT].help

    assert "--synth-backend cake" in help_text
    assert "Default: 'fast', or 'det' under --deterministic-mode, which rejects 'fast'" in help_text


@pytest.mark.parametrize(
    ("argv", "variant"),
    [
        (["--synth-backend", "cake"], "fast"),
        (["--synth-backend", "cake", "--deterministic-mode"], "det"),
        (["--synth-backend", "cake", VARIANT, "det"], "det"),
        (["--synth-backend", "cake", VARIANT, "fast"], "fast"),
        (["--synth-backend", "cake", VARIANT, "det", "--deterministic-mode"], "det"),
    ],
)
def test_validate_args_resolves_the_variant_of_a_selected_op_and_writes_it_back(synth, argv, variant):
    assert _validated(argv).cake_synth_variant == variant


def _miles_flow(argv: list[str]) -> dict:
    """Parse, validate on the flat namespace, move the registered values into one field, configure a process."""
    args = _validated(argv)
    registered = argparse.ArgumentParser(add_help=False)
    miles_ext_kernels.add_arguments(registered)
    miles_ext_kernels_args = {action.dest: vars(args).pop(action.dest) for action in registered._actions}
    miles_ext_kernels.configure(miles_ext_kernels_args)
    return miles_ext_kernels_args


def test_the_cli_default_is_det_when_miles_may_not_choose_fast(monkeypatch):
    _install(monkeypatch, cli_variants=("det",))

    assert _miles_flow(["--synth-backend", "cake"])["cake_synth_variant"] == "det"
    assert cake.variant("synth") == "det"


@pytest.mark.parametrize(("deterministic", "variant"), [([], "fast"), (["--deterministic-mode"], "det")])
def test_a_single_cli_choice_reaches_every_process(monkeypatch, deterministic, variant):
    """miles may choose fast alone: the resolved variant still travels to the processes that configure."""
    _install(monkeypatch, cli_variants=("fast",))

    assert _miles_flow(["--synth-backend", "cake", *deterministic])["cake_synth_variant"] == variant
    assert cake.variant("synth") == variant


def test_deterministic_mode_rejects_an_explicit_fast(synth):
    with pytest.raises(ValueError, match="--cake-synth-variant fast is not bit-reproducible under --deterministic"):
        _validated(["--synth-backend", "cake", VARIANT, "fast", "--deterministic-mode"])


def test_deterministic_mode_rejects_an_op_without_det(monkeypatch):
    _install(monkeypatch, variants=("fast",))

    assert _validated(["--synth-backend", "cake"]).synth_backend == "cake"
    with pytest.raises(ValueError, match="have no det variant: --synth-backend cake is not deterministic"):
        _validated(["--synth-backend", "cake", "--deterministic-mode"])
    _validated(["--deterministic-mode"])  # not selected


@pytest.mark.parametrize("argv", [[VARIANT, "det"], ["--cake-synth-scale", "2"]])
def test_an_option_of_an_unselected_op_is_rejected(synth, argv):
    with pytest.raises(ValueError, match=f"{argv[0]} requires --synth-backend cake"):
        _validated(argv)


def test_an_unselected_op_keeps_its_defaults(synth):
    args = _validated(["--cake-synth-scale", "1.0"])

    assert args.cake_synth_variant is None and args.cake_synth_scale == 1.0


def test_the_op_validates_its_options_only_when_selected(synth):
    with pytest.raises(ValueError, match="--cake-synth-scale must be positive"):
        _validated(["--synth-backend", "cake", "--cake-synth-scale", "-1"])


def test_a_negative_pool_bound_is_rejected():
    with pytest.raises(ValueError, match="--cake-workspace-pool-bytes must be >= 0"):
        _validated(["--cake-workspace-pool-bytes", "-1"])
    with pytest.raises(ValueError, match="--cake-workspace-pool-bytes must be >= 0"):
        cake.configure(cake_workspace_pool_bytes=-1)


def test_the_miles_flow_keeps_the_command_line_default_fast(synth):
    assert _miles_flow(["--synth-backend", "cake", "--cake-synth-scale", "3"]) == {
        "cake_synth_variant": "fast",
        "cake_synth_scale": 3.0,
        "cake_workspace_pool_bytes": 0,
    }
    assert cake.variant("synth") == "fast"
    assert cake.option("synth", "cake_synth_scale") == 3.0


def test_an_unconfigured_process_runs_det_and_the_defaults(synth):
    assert cake.variant("synth") == "det"
    assert cake.option("synth", "cake_synth_scale") == 1.0
    assert cake.option(None, "cake_workspace_pool_bytes") == 0


def test_an_op_without_det_defaults_to_its_only_variant(monkeypatch):
    _install(monkeypatch, variants=("fast",))

    assert cake.variant("synth") == "fast"


@pytest.mark.parametrize(
    "args",
    [
        argparse.Namespace(cake_synth_variant="fast", cake_synth_scale=2.0, unrelated=1),
        {"cake_synth_variant": "fast", "cake_synth_scale": 2.0},
        SimpleNamespace(cake_synth_variant="fast", cake_synth_scale=2.0),
    ],
)
def test_configure_takes_a_namespace_or_a_mapping(synth, args):
    cake.configure(args)

    assert cake.variant("synth") == "fast"
    assert cake.option("synth", "cake_synth_scale") == 2.0


def test_every_configure_replaces_the_whole_configuration(synth):
    cake.configure(cake_synth_variant="fast", cake_synth_scale=2.0)
    cake.configure({"cake_synth_scale": 5.0}, cake_synth_scale=4.0)

    assert cake.variant("synth") == "det"
    assert cake.option("synth", "cake_synth_scale") == 4.0
    cake.configure()
    assert cake.option("synth", "cake_synth_scale") == 1.0


def test_none_is_unset(synth):
    cake.configure(cake_synth_variant=None, cake_synth_scale=None)

    assert cake.variant("synth") == "det" and cake.option("synth", "cake_synth_scale") == 1.0


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"cake_synth_variant": "fastest"}, r"--cake-synth-variant must be one of \('det', 'fast'\)"),
        ({"cake_synth_typo": 1}, "unknown cake option 'cake_synth_typo'"),
        ({"cake_synth_variant": "fast", "deterministic_mode": True}, "not bit-reproducible"),
        ({"cake_synth_variant": "fast", "backend": SimpleNamespace(deterministic_mode=True)}, "not bit-reproducible"),
    ],
)
def test_configure_rejects(synth, values, error):
    cake.configure(cake_synth_variant="fast")

    with pytest.raises(ValueError, match=error):
        cake.configure(**values)
    assert cake.variant("synth") == "fast"  # a rejected configuration changes nothing


def test_configure_accepts_a_variant_miles_may_not_choose(monkeypatch):
    _install(monkeypatch, cli_variants=("det",))
    cake.configure(cake_synth_variant="fast")

    assert cake.variant("synth") == "fast"


def test_deterministic_mode_without_a_variant_runs_det(synth):
    cake.configure(argparse.Namespace(deterministic_mode=True))

    assert cake.variant("synth") == "det"


def test_deterministic_mode_refuses_to_run_an_op_without_det(monkeypatch):
    _install(monkeypatch, variants=("fast",))
    cake.configure(deterministic_mode=True)

    with pytest.raises(ValueError, match="the cake synth kernels have no det variant"):
        cake.variant("synth")
    cake.configure()
    assert cake.variant("synth") == "fast"


def test_a_flag_keeps_its_dest_and_its_equality_once_the_dest_was_read():
    flag = Flag("--cake-synth-scale", dict(type=float, default=1.0, help="h"))
    first = flag.dest

    assert first == "cake_synth_scale" and flag.dest is first
    assert flag == Flag("--cake-synth-scale", dict(type=float, default=1.0, help="h"))
    assert dataclasses.replace(flag) == flag and dataclasses.replace(flag).dest == "cake_synth_scale"
    assert "dest" not in {field.name for field in dataclasses.fields(flag)} and "dest" not in repr(flag)
    with pytest.raises(dataclasses.FrozenInstanceError):
        flag.name = "--cake-synth-other"


def test_unknown_ops_and_options_raise(synth):
    with pytest.raises(KeyError, match="no cake op 'nope'"):
        cake.variant("nope")
    with pytest.raises(KeyError, match="synth has no option 'cake_workspace_pool_bytes'"):
        cake.option("synth", "cake_workspace_pool_bytes")
    with pytest.raises(KeyError, match="the runtime has no option 'cake_synth_scale'"):
        cake.option(None, "cake_synth_scale")


def test_a_refused_variant_is_refused_at_every_call(monkeypatch):
    _install(monkeypatch, variants=("fast",))
    cake.configure(deterministic_mode=True)

    for _ in range(2):
        with pytest.raises(ValueError, match="the cake synth kernels have no det variant"):
            cake.variant("synth")


@pytest.mark.parametrize(
    ("args", "deterministic"),
    [
        (SimpleNamespace(), False),
        (SimpleNamespace(deterministic_mode=False), False),
        (SimpleNamespace(deterministic_mode=True), True),
        (SimpleNamespace(backend=SimpleNamespace(deterministic_mode=True)), True),  # miles' typed configuration
        (SimpleNamespace(backend=SimpleNamespace()), False),
    ],
)
def test_deterministic_mode_is_read_from_the_namespace_miles_validates(args, deterministic):
    assert cake.deterministic_mode(args) is deterministic


def test_the_variant_is_read_when_the_call_runs(synth):
    torch = pytest.importorskip("torch")
    x = torch.arange(4, dtype=torch.float32)

    cake.configure(cake_synth_variant="det")
    assert synth.scale(x)[0] == "det"
    cake.configure(cake_synth_variant="fast", cake_synth_scale=2.0)
    variant, y = synth.scale(x)
    assert variant == "fast" and torch.equal(y, 2 * x)


def test_a_det_run_is_logged_once_per_op_and_reason(caplog, monkeypatch):
    monkeypatch.setattr(config, "_warned", set())

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        for _ in range(3):
            cake.warn_runs_det("synth", "context parallelism")
        cake.warn_runs_det("synth", "packed sequences")
        cake.warn_runs_det("other", "context parallelism")

    assert [record.getMessage() for record in caplog.records] == [
        "miles_ext_kernels.cake.synth: the fast variant does not serve context parallelism; running det",
        "miles_ext_kernels.cake.synth: the fast variant does not serve packed sequences; running det",
        "miles_ext_kernels.cake.other: the fast variant does not serve context parallelism; running det",
    ]
