"""The gated_rmsnorm op on the CPU: its declaration, its argument, the variant a call runs, and the checks of
``check_supported`` that run before any device query.

The argument tests are torch-free; the call tests swap the vendored packages for recorders, so they need CPU torch
only.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import gated_rmsnorm

REPO_ROOT = Path(__file__).resolve().parents[3]
VARIANT = "--cake-gated-rmsnorm-variant"
DET = "cake_rmsnorm_silu_gate_det_train"
FAST = "cake_rmsnorm_silu_gate_nondet_train"


def _validated(argv: list[str], **miles_args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(argv)
    for name, value in miles_args.items():  # the miles arguments validate_args reads from the same namespace
        setattr(args, name, value)
    miles_ext_kernels.validate_args(args)
    return args


def test_the_op_declaration():
    op = gated_rmsnorm.OP

    assert op.name == "gated_rmsnorm" and op.variants == ("det", "fast") and op.cli_choices == ("det", "fast")
    assert op.selected_by == (("linear_attention_output_norm_backend", "cake"),)
    assert op.packages == (DET, FAST)
    assert gated_rmsnorm.__all__ == ["OP", "check_supported", "rmsnorm_silu_gate"]
    assert op.has_variant_flag and op.shows_variant_flag and op.variant_flag == VARIANT and op.flags == ()


def test_the_variant_flag_is_listed_in_the_help():
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)

    assert VARIANT in parser.format_help()


def test_an_unconfigured_process_runs_det():
    """The default of a process that never configured the op is det (it used to be fast)."""
    assert cake.variant("gated_rmsnorm") == "det"
    cake.configure()
    assert cake.variant("gated_rmsnorm") == "det"
    cake.configure({"cake_gated_rmsnorm_variant": None})
    assert cake.variant("gated_rmsnorm") == "det"


def test_without_the_cake_backend_the_variant_stays_unset():
    args = _validated([], linear_attention_output_norm_backend="fla")

    assert args.cake_gated_rmsnorm_variant is None
    cake.configure(args)
    assert cake.variant("gated_rmsnorm") == "det"


@pytest.mark.parametrize(
    "deterministic_mode, expected", [(False, "fast"), (True, "det")], ids=["default", "deterministic-mode"]
)
def test_the_cake_backend_resolves_the_variant_for_every_process(deterministic_mode, expected):
    args = _validated([], linear_attention_output_norm_backend="cake", deterministic_mode=deterministic_mode)

    assert args.cake_gated_rmsnorm_variant == expected
    cake.configure({"cake_gated_rmsnorm_variant": args.cake_gated_rmsnorm_variant})  # miles_ext_kernels_args
    assert cake.variant("gated_rmsnorm") == expected


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_an_explicit_variant_is_kept(variant):
    args = _validated([VARIANT, variant], linear_attention_output_norm_backend="cake")

    assert args.cake_gated_rmsnorm_variant == variant


def test_explicit_fast_is_rejected_in_deterministic_mode():
    with pytest.raises(ValueError, match=f"{VARIANT} fast is not bit-reproducible under --deterministic-mode"):
        _validated([VARIANT, "fast"], linear_attention_output_norm_backend="cake", deterministic_mode=True)
    with pytest.raises(ValueError, match=f"{VARIANT} fast is not bit-reproducible"):
        cake.configure(cake_gated_rmsnorm_variant="fast", deterministic_mode=True)
    cake.configure(deterministic_mode=True)
    assert cake.variant("gated_rmsnorm") == "det"


@pytest.mark.parametrize("backend", [None, "fla"])
def test_the_variant_without_the_cake_backend_is_rejected(backend):
    miles_args = {} if backend is None else {"linear_attention_output_norm_backend": backend}

    with pytest.raises(ValueError, match=f"{VARIANT} requires --linear-attention-output-norm-backend cake"):
        _validated([VARIANT, "det"], **miles_args)


def test_configure_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match=f"{VARIANT} must be one of"):
        cake.configure(cake_gated_rmsnorm_variant="exact")


_NO_TORCH = """
import argparse
import sys

for blocked in ("torch", "tvm_ffi", "flashinfer"):
    sys.modules[blocked] = None  # importing it raises ImportError

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import gated_rmsnorm

assert gated_rmsnorm.OP.name == "gated_rmsnorm"
parser = argparse.ArgumentParser()
miles_ext_kernels.add_arguments(parser)
args = parser.parse_args([])
args.linear_attention_output_norm_backend = "cake"
miles_ext_kernels.validate_args(args)
assert args.cake_gated_rmsnorm_variant == "fast", args
miles_ext_kernels.configure({"cake_gated_rmsnorm_variant": args.cake_gated_rmsnorm_variant})
assert cake.variant("gated_rmsnorm") == "fast"
assert "miles_ext_kernels.cake._impl.gated_rmsnorm" not in sys.modules
"""


def test_the_op_imports_and_configures_without_torch():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


# --- the call: which package runs (CPU torch, the vendored packages replaced by recorders) ---


@pytest.fixture
def ops(monkeypatch):
    """The implementation with both packages replaced: ``calls`` records the chosen package."""
    pytest.importorskip("torch")
    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops

    calls = []

    def det(x, z, weight, eps):
        calls.append((DET, None))
        return x

    def fast(x, z, weight, eps):
        calls.append((FAST, None))
        return x

    packages = {
        "det": SimpleNamespace(cake_rmsnorm_silu_gate_det=det),
        "fast": SimpleNamespace(cake_rmsnorm_silu_gate_nondet=fast),
    }
    monkeypatch.setattr(ops, "_package", packages.__getitem__)
    monkeypatch.setattr(ops.config, "_warned", set())
    monkeypatch.setattr(ops, "calls", calls, raising=False)
    return ops


def _call(ops):
    import torch

    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    return ops.rmsnorm_silu_gate(x, x, torch.ones(128, dtype=torch.bfloat16), 1e-6)


def test_an_unconfigured_call_runs_the_det_package(ops):
    _call(ops)

    assert ops.calls == [(DET, None)]


def test_the_configured_variant_is_read_at_every_call(ops):
    for variant in ("fast", "det", "fast"):
        cake.configure(cake_gated_rmsnorm_variant=variant)
        _call(ops)

    assert ops.calls == [(FAST, None), (DET, None), (FAST, None)]


def test_fast_runs_det_under_torch_deterministic_algorithms(ops, caplog):
    import torch

    cake.configure(cake_gated_rmsnorm_variant="fast")
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            _call(ops)
            _call(ops)
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    _call(ops)

    assert ops.calls == [(DET, None), (DET, None), (FAST, None)]
    warnings = [record.getMessage() for record in caplog.records if "gated_rmsnorm" in record.getMessage()]
    assert warnings == [
        "miles_ext_kernels.cake.gated_rmsnorm: the fast variant does not serve "
        "torch.use_deterministic_algorithms(True); running det"
    ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(head_dim=128, activation="sigmoid"), "gates its output norm with 'sigmoid'"),
        (dict(head_dim=64, activation="silu"), "rows of 128"),
        (dict(head_dim=128, activation="silu", has_bias=True), "no bias"),
        (dict(head_dim=128, activation="silu", weight_dtype="float32"), "BF16 weight"),
    ],
    ids=["sigmoid-gate", "head-dim-64", "bias", "fp32-weight"],
)
def test_unsupported_norms_are_rejected_before_the_device_is_queried(monkeypatch, kwargs, match):
    torch = pytest.importorskip("torch")
    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops

    def no_device(*args, **kwargs):
        raise AssertionError("queried the device")

    monkeypatch.setattr(torch.cuda, "is_available", no_device)
    kwargs = dict(kwargs)
    weight_dtype = getattr(torch, kwargs.pop("weight_dtype", "bfloat16"))
    with pytest.raises(ValueError, match=match):
        ops.check_supported(weight_dtype=weight_dtype, **kwargs)


def test_a_device_of_another_architecture_is_rejected(monkeypatch):
    torch = pytest.importorskip("torch")
    from miles_ext_kernels.cake._impl.gated_rmsnorm import ops

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(ValueError, match=r"SM100a / SM103a / SM107a\), not \(9, 0\); use --linear-attention"):
        ops.check_supported(128, "silu", torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="needs a CUDA device"):
        ops.check_supported(128, "silu", torch.bfloat16)
