"""The DSA arguments, registered, resolved and validated through the hooks miles calls, and the configuration the
kernels read when a call runs (torch-free)."""

import argparse
import logging

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake

GIB = 1 << 30


def _parse(argv: list[str], **miles_args) -> argparse.Namespace:
    """The ext arguments parsed as miles parses them, plus the miles fields ``miles_args`` (backend, mode)."""
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(argv)
    for name, value in miles_args.items():
        setattr(args, name, value)
    return args


def test_the_op_registers_its_variant_and_options_with_their_defaults():
    args = _parse([])

    assert (args.cake_dsa_variant, args.cake_dsa_bwd_workspace_bytes, args.cake_dsa_lazy_build) == (
        None,
        2 * GIB,
        False,
    )
    miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("backend", [None, "tilelang"])
def test_the_defaults_validate_without_the_cake_backend_and_leave_the_variant_unset(backend):
    args = _parse([]) if backend is None else _parse([], dsa_attention_backend=backend)

    miles_ext_kernels.validate_args(args)
    assert args.cake_dsa_variant is None


@pytest.mark.parametrize(
    ("deterministic", "expected"), [(False, "fast"), (True, "det")], ids=["default-fast", "deterministic-det"]
)
def test_an_unset_variant_resolves_by_deterministic_mode_and_is_written_back(deterministic, expected):
    args = _parse([], dsa_attention_backend="cake", deterministic_mode=deterministic)

    miles_ext_kernels.validate_args(args)
    assert args.cake_dsa_variant == expected


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_explicit_variant_is_kept_outside_deterministic_mode(variant):
    args = _parse(["--cake-dsa-variant", variant], dsa_attention_backend="cake", deterministic_mode=False)

    miles_ext_kernels.validate_args(args)
    assert args.cake_dsa_variant == variant


def test_explicit_fast_is_rejected_under_deterministic_mode():
    # the fast backward sums dK/dV with FP32 atomics in a run-dependent order
    args = _parse(["--cake-dsa-variant", "fast"], dsa_attention_backend="cake", deterministic_mode=True)

    with pytest.raises(ValueError, match="--cake-dsa-variant fast is not bit-reproducible"):
        miles_ext_kernels.validate_args(args)


def test_an_unknown_variant_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        _parse(["--cake-dsa-variant", "nd"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--cake-dsa-variant", "det"],
        ["--cake-dsa-variant", "fast"],
        ["--cake-dsa-bwd-workspace-bytes", str(GIB)],
        ["--cake-dsa-lazy-build"],
    ],
    ids=["variant-det", "variant-fast", "workspace", "lazy-build"],
)
@pytest.mark.parametrize("backend", [None, "tilelang"])
def test_every_dsa_argument_requires_the_cake_backend(argv, backend):
    args = _parse(argv) if backend is None else _parse(argv, dsa_attention_backend=backend)

    with pytest.raises(ValueError, match=f"{argv[0]} requires --dsa-attention-backend cake"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("workspace", ["0", "-1"])
def test_a_non_positive_workspace_is_rejected(workspace):
    args = _parse(["--cake-dsa-bwd-workspace-bytes", workspace], dsa_attention_backend="cake")

    with pytest.raises(ValueError, match="must be positive"):
        miles_ext_kernels.validate_args(args)


def test_configure_takes_the_values_miles_carries_and_is_idempotent():
    args = _parse(
        ["--cake-dsa-bwd-workspace-bytes", str(GIB), "--cake-dsa-lazy-build"],
        dsa_attention_backend="cake",
        deterministic_mode=False,
    )
    miles_ext_kernels.validate_args(args)
    carried = {key: value for key, value in vars(args).items() if key.startswith("cake_")}

    for _ in range(2):  # miles calls it in every process that builds a layer
        miles_ext_kernels.configure(carried)
        assert cake.variant("dsa") == "fast"
        assert cake.option("dsa", "cake_dsa_bwd_workspace_bytes") == GIB
        assert cake.option("dsa", "cake_dsa_lazy_build") is True


def test_an_unconfigured_process_runs_det_with_the_default_options():
    cake.configure()

    assert cake.variant("dsa") == "det"
    assert cake.option("dsa", "cake_dsa_bwd_workspace_bytes") == 2 * GIB
    assert cake.option("dsa", "cake_dsa_lazy_build") is False


def test_configure_rejects_an_unknown_variant_and_fast_in_deterministic_mode():
    with pytest.raises(ValueError, match="--cake-dsa-variant must be one of"):
        cake.configure(cake_dsa_variant="nd")
    with pytest.raises(ValueError, match="not bit-reproducible"):
        cake.configure(cake_dsa_variant="fast", deterministic_mode=True)
    cake.configure(deterministic_mode=True)
    assert cake.variant("dsa") == "det"


def test_a_call_fast_does_not_serve_is_logged_once_per_reason(caplog):
    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        for _ in range(3):
            cake.warn_runs_det("dsa", "a test reason the dsa tests own")
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "miles_ext_kernels.cake.dsa: the fast variant does not serve a test reason the dsa tests own; running det"
    ]
