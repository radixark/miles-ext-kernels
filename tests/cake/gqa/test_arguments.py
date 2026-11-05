"""The gqa op's arguments, variant and head-layout predicate, through the hooks miles calls (torch-free)."""

import argparse
from types import SimpleNamespace

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import gqa

NO_DET = "the cake gqa kernels have no det variant"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    return parser.parse_args(argv)


def _miles(argv: tuple[str, ...] = (), **values) -> argparse.Namespace:
    """miles' parsed flat namespace selecting the op: a Qwen3.5-style full-attention layer (16 query heads, 2
    key/value groups, head dim 256) on one tensor-parallel rank, unless ``values`` say otherwise."""
    args = _parse(list(argv))
    vars(args).update(
        full_attention_backend="cake",
        deterministic_mode=False,
        num_attention_heads=16,
        num_query_groups=2,
        group_query_attention=True,
        kv_channels=256,
        hidden_size=2048,
        tensor_model_parallel_size=1,
    )
    vars(args).update(values)
    return args


def test_the_op_is_registered_with_its_one_variant_and_no_variant_flag():
    spec = cake.op_spec("gqa")

    assert "gqa" in cake.OPS and spec is gqa.OP
    assert spec.variants == ("fast",) and spec.packages == ("cake_gqa_train",)
    assert spec.selected_by == (("full_attention_backend", "cake"),)
    assert not spec.has_variant_flag
    assert "cake_gqa_variant" not in vars(_parse([]))
    with pytest.raises(SystemExit):
        _parse(["--cake-gqa-variant", "fast"])


def test_the_check_option_is_off_by_default_and_needs_the_backend():
    assert _parse([]).cake_gqa_check_seqlens is False
    args = _parse(["--cake-gqa-check-seqlens"])
    assert args.cake_gqa_check_seqlens is True

    with pytest.raises(ValueError, match="--cake-gqa-check-seqlens requires --full-attention-backend cake"):
        miles_ext_kernels.validate_args(args)
    miles_ext_kernels.validate_args(_miles(["--cake-gqa-check-seqlens"]))


@pytest.mark.parametrize("backend", [None, "megatron", "cake"])
def test_every_backend_validates_outside_deterministic_mode(backend):
    args = _miles(full_attention_backend=backend)

    miles_ext_kernels.validate_args(args)

    assert "cake_gqa_variant" not in vars(args)


def test_deterministic_mode_rejects_the_op():
    with pytest.raises(ValueError, match=f"{NO_DET}: --full-attention-backend cake is not deterministic"):
        miles_ext_kernels.validate_args(_miles(deterministic_mode=True))


def test_the_typed_configuration_s_deterministic_mode_rejects_it_too():
    # the configuration miles builds after parsing carries the flag as args.backend.deterministic_mode
    args = _miles(backend=SimpleNamespace(deterministic_mode=True))
    del args.deterministic_mode

    with pytest.raises(ValueError, match=NO_DET):
        miles_ext_kernels.validate_args(args)


def test_another_attention_backend_validates_in_deterministic_mode():
    miles_ext_kernels.validate_args(_miles(full_attention_backend="megatron", deterministic_mode=True))


@pytest.mark.parametrize(
    "heads, kv_heads, head_dim",
    [(2, 1, 256), (4, 2, 256), (8, 2, 256), (6, 1, 256), (16, 2, 256), (24, 4, 256), (32, 2, 256), (8, 1, 256)],
)
def test_the_predicate_accepts_an_even_number_of_query_heads_per_key_head(heads, kv_heads, head_dim):
    gqa.check_supported(heads, kv_heads, head_dim)

    assert gqa.is_supported(heads, kv_heads, head_dim)


@pytest.mark.parametrize(
    "heads, kv_heads, head_dim, reason",
    [
        (3, 1, 256, "3 query heads over 1 key/value heads"),
        (1, 1, 256, "1 query heads over 1 key/value heads"),
        (24, 8, 256, "24 query heads over 8 key/value heads"),
        (8, 3, 256, "8 query heads over 3 key/value heads"),
        (0, 1, 256, "0 query heads over 1 key/value heads"),
        (8, 0, 256, "8 query heads over 0 key/value heads"),
        (8, 2, 128, "head dim 128"),
    ],
)
def test_the_predicate_names_what_the_kernels_do_not_run(heads, kv_heads, head_dim, reason):
    with pytest.raises(ValueError, match=f"the cake gqa kernels do not run .*{reason}"):
        gqa.check_supported(heads, kv_heads, head_dim)

    assert not gqa.is_supported(heads, kv_heads, head_dim)


@pytest.mark.parametrize(
    "values, local",
    [
        # 24 query heads over 4 groups at TP 8: 3 query heads over 1 replicated key/value head per rank
        (dict(num_attention_heads=24, num_query_groups=4, tensor_model_parallel_size=8), "3 query heads over 1"),
        (dict(num_attention_heads=8, num_query_groups=8, tensor_model_parallel_size=8), "1 query heads over 1"),
        (dict(num_attention_heads=16, num_query_groups=4, tensor_model_parallel_size=8), None),
        # without --group-query-attention every query head has its own key/value head
        (dict(group_query_attention=False), "16 query heads over 16"),
        (dict(kv_channels=128), "head dim 128"),
        # no --kv-channels: Megatron's default, hidden_size / num_attention_heads
        (dict(kv_channels=None, hidden_size=4096, num_attention_heads=16), None),
        (dict(kv_channels=None, hidden_size=2048, num_attention_heads=16), "head dim 128"),
        (dict(num_attention_heads=32, num_query_groups=4, tensor_model_parallel_size=2), None),
        (dict(num_attention_heads=16, num_query_groups=2, tensor_model_parallel_size=4), None),
        (dict(num_attention_heads=32, num_query_groups=2, tensor_model_parallel_size=8), None),
    ],
)
def test_validate_args_checks_the_heads_of_one_tensor_parallel_rank(values, local):
    args = _miles(**values)
    if local is None:
        miles_ext_kernels.validate_args(args)
        return
    with pytest.raises(ValueError, match=f"--full-attention-backend cake with --num-attention-heads .*{local}"):
        miles_ext_kernels.validate_args(args)


def test_validate_args_leaves_heads_it_cannot_see_to_the_layer():
    miles_ext_kernels.validate_args(_miles(num_attention_heads=None, kv_channels=None, hidden_size=None))


def test_an_unconfigured_process_runs_fast():
    assert cake.variant("gqa") == "fast"
    assert cake.option("gqa", "cake_gqa_check_seqlens") is False


def test_configure_holds_the_deterministic_rules_and_the_option():
    cake.configure(deterministic_mode=True)
    with pytest.raises(ValueError, match=NO_DET):
        cake.variant("gqa")
    with pytest.raises(ValueError, match="fast is not bit-reproducible under --deterministic-mode"):
        cake.configure(cake_gqa_variant="fast", deterministic_mode=True)
    with pytest.raises(ValueError, match="must be one of"):
        cake.configure(cake_gqa_variant="det")

    cake.configure(cake_gqa_variant="fast", cake_gqa_check_seqlens=True)
    assert cake.variant("gqa") == "fast"
    assert cake.option("gqa", "cake_gqa_check_seqlens") is True
    cake.configure()
    assert cake.option("gqa", "cake_gqa_check_seqlens") is False


def test_the_values_miles_keeps_configure_a_worker():
    args = _miles(["--cake-gqa-check-seqlens"])
    miles_ext_kernels.validate_args(args)

    miles_ext_kernels.configure({key: value for key, value in vars(args).items() if key.startswith("cake_")})

    assert cake.variant("gqa") == "fast"
    assert cake.option("gqa", "cake_gqa_check_seqlens") is True
