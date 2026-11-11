"""The lm_head arguments through miles' hooks: registration, variant resolution, the startup rejections and the
process configuration (torch-free)."""

import argparse

import pytest

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake import lm_head

VARIANT = "--cake-lm-head-variant"
CHUNK_SIZE = "--cake-lm-head-chunk-size"

# miles' flat namespace of a run the kernels serve (TP 1)
SERVED = dict(
    log_probs_backend="cake",
    train_backend="megatron",
    tensor_model_parallel_size=1,
    loss_type="policy_loss",
    entropy_coef=0.0,
    observe_training_entropy=False,
    use_rollout_entropy=False,
    vocab_size=129280,
    hidden_size=4096,
    bf16=True,
    multi_lora_n_adapters=0,
    lora_rank=0,
    target_modules=None,
    deterministic_mode=False,
)


def _parse(argv=(), **attributes) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    args = parser.parse_args(list(argv))
    for name, value in attributes.items():
        setattr(args, name, value)
    return args


def _served(argv=(), **overrides) -> argparse.Namespace:
    return _parse(argv, **{**SERVED, **overrides})


def test_the_op_declares_both_packages_both_variants_and_one_entry_point():
    spec = cake.op_spec("lm_head")

    assert spec is lm_head.OP
    assert spec.variants == ("det", "fast") and spec.cli_choices == ("det", "fast")
    assert spec.packages == ("cake_lm_head_det_train", "cake_lm_head_nondet_train")
    assert spec.exports == ("log_probs",)
    assert spec.selected_by == (("log_probs_backend", "cake"),)
    assert spec.has_variant_flag and spec.shows_variant_flag
    assert [flag.name for flag in spec.flags] == [CHUNK_SIZE]


def test_the_flags_are_registered_in_the_miles_ext_kernels_group_and_the_defaults_validate():
    parser = argparse.ArgumentParser()
    miles_ext_kernels.add_arguments(parser)
    (group,) = [group for group in parser._action_groups if group.title == "miles-ext-kernels"]
    names = [action.option_strings[0] for action in group._group_actions]
    args = parser.parse_args([])

    assert VARIANT in names and CHUNK_SIZE in names
    assert args.cake_lm_head_variant is None and args.cake_lm_head_chunk_size == lm_head.DEFAULT_CHUNK_SIZE
    miles_ext_kernels.validate_args(args)
    assert args.cake_lm_head_variant is None


def test_the_flags_parse_their_values():
    args = _parse([VARIANT, "det", CHUNK_SIZE, "2048"])

    assert args.cake_lm_head_variant == "det" and args.cake_lm_head_chunk_size == 2048
    with pytest.raises(SystemExit):
        _parse([VARIANT, "fastest"])


@pytest.mark.parametrize("backend", [None, "torch", "fused"])
@pytest.mark.parametrize(
    ("argv", "flag"),
    [([VARIANT, "det"], VARIANT), ([VARIANT, "fast"], VARIANT), ([CHUNK_SIZE, "2048"], CHUNK_SIZE)],
)
def test_an_lm_head_argument_without_the_cake_backend_is_rejected(backend, argv, flag):
    args = _parse(argv)
    if backend is not None:
        args.log_probs_backend = backend

    with pytest.raises(ValueError, match=f"{flag} requires --log-probs-backend cake"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("backend", [None, "torch", "fused"])
def test_without_the_cake_backend_nothing_is_resolved_or_rejected(backend):
    # a run on another backend keeps every Megatron / miles setting the kernels would reject
    args = _served(log_probs_backend=backend, tensor_model_parallel_size=8, bf16=False, vocab_size=151936)

    miles_ext_kernels.validate_args(args)
    assert args.cake_lm_head_variant is None


@pytest.mark.parametrize(("deterministic", "variant"), [(False, "fast"), (True, "det")])
def test_an_unset_variant_resolves_to_fast_or_to_det_under_deterministic_mode(deterministic, variant):
    args = _served(deterministic_mode=deterministic)

    miles_ext_kernels.validate_args(args)
    assert args.cake_lm_head_variant == variant
    assert args.cake_lm_head_chunk_size == lm_head.DEFAULT_CHUNK_SIZE


@pytest.mark.parametrize("deterministic", [False, True])
def test_an_explicit_det_variant_is_kept(deterministic):
    args = _served([VARIANT, "det"], deterministic_mode=deterministic)

    miles_ext_kernels.validate_args(args)
    assert args.cake_lm_head_variant == "det"


def test_an_explicit_fast_variant_is_kept_outside_deterministic_mode_and_rejected_in_it():
    args = _served([VARIANT, "fast"])
    miles_ext_kernels.validate_args(args)
    assert args.cake_lm_head_variant == "fast"

    # fast sums the dX K slices in a run-dependent order
    args = _served([VARIANT, "fast"], deterministic_mode=True)
    with pytest.raises(ValueError, match=f"{VARIANT} fast is not the det variant that --deterministic-mode runs"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize("chunk_size", [1, 1024, 4097, 65535])
def test_an_explicit_chunk_size_is_kept(chunk_size):
    args = _served([CHUNK_SIZE, str(chunk_size)])

    miles_ext_kernels.validate_args(args)
    assert args.cake_lm_head_chunk_size == chunk_size


@pytest.mark.parametrize("chunk_size", [0, -4096, 65536])
def test_a_chunk_size_out_of_range_is_rejected(chunk_size):
    args = _served([CHUNK_SIZE, str(chunk_size)])

    with pytest.raises(ValueError, match=r"--cake-lm-head-chunk-size must be an integer in \[1, 65535\]"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(train_backend="fsdp"), "needs --train-backend megatron"),
        (dict(train_backend="torchtitan"), "needs --train-backend megatron"),
        (dict(tensor_model_parallel_size=2), "needs --tensor-model-parallel-size 1"),
        (dict(loss_type="custom_loss"), "supports --loss-type policy_loss and sft_loss, got custom_loss"),
        (dict(loss_type="score_centering"), "supports --loss-type policy_loss and sft_loss"),
        (dict(entropy_coef=0.01), "computes no entropy: it needs --entropy-coef 0"),
        (dict(observe_training_entropy=True), "computes no entropy: drop --observe-training-entropy"),
        (dict(use_rollout_entropy=True), "computes no entropy: drop --use-rollout-entropy"),
        (dict(vocab_size=None), "needs --vocab-size"),
        # a vocabulary that is not a multiple of the 256-column logits tile is not served (padding changes the softmax)
        (
            dict(vocab_size=151936),
            r"needs a --vocab-size that is a multiple of 256 \(the kernels' logits tile; padding the vocabulary would "
            r"change the softmax\), got 151936",
        ),
        (dict(vocab_size=129280 + 128), "needs a --vocab-size that is a multiple of 256"),
        (dict(hidden_size=2304), "needs a --hidden-size that is a multiple of 512, got 2304"),
        (dict(bf16=False), "needs --bf16"),
        (dict(multi_lora_n_adapters=4), "does not support multi-LoRA"),
        (dict(lora_rank=8, target_modules="attn,unembed"), "does not support a LoRA adapter on the output layer"),
        (dict(lora_rank=8, target_modules="q_proj, lm_head"), "does not support a LoRA adapter on the output layer"),
        (
            dict(lora_rank=8, target_modules=["language_model.lm_head"]),
            "does not support a LoRA adapter on the output layer",
        ),
        (dict(lora_rank=8, target_modules=["output_layer"]), "does not support a LoRA adapter on the output layer"),
        (dict(use_mup=True), "does not apply MuP's output logit scaling"),
    ],
)
def test_what_the_kernels_do_not_serve_is_rejected_at_startup(overrides, message):
    args = _served(**overrides)

    with pytest.raises(ValueError, match=f"^--log-probs-backend cake {message}"):
        miles_ext_kernels.validate_args(args)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(loss_type="sft_loss"),
        dict(vocab_size=154880, hidden_size=6144),
        dict(vocab_size=248320, hidden_size=5120),
        dict(lora_rank=8, target_modules=None),
        dict(lora_rank=8, target_modules="attn,mlp"),
        dict(lora_rank=8, target_modules=["q_proj", "v_proj", "head_dim_proj"]),
        dict(lora_rank=0, target_modules="unembed"),
        dict(use_mup=False),
    ],
)
def test_what_the_kernels_serve_is_accepted(overrides):
    miles_ext_kernels.validate_args(_served(**overrides))


def test_the_rules_skip_the_settings_a_namespace_lacks():
    # e.g. a caller validating only the kernel arguments: nothing to reject but what it names
    args = _parse([], log_probs_backend="cake")

    miles_ext_kernels.validate_args(args)
    assert args.cake_lm_head_variant == "fast"


def test_an_unconfigured_process_runs_det_with_the_default_chunk_size():
    assert cake.variant("lm_head") == "det"
    assert cake.option("lm_head", "cake_lm_head_chunk_size") == lm_head.DEFAULT_CHUNK_SIZE


@pytest.mark.parametrize(
    ("argv", "deterministic", "variant", "chunk_size"),
    [
        ([], False, "fast", 4096),
        ([], True, "det", 4096),
        ([VARIANT, "det", CHUNK_SIZE, "2048"], False, "det", 2048),
        ([CHUNK_SIZE, "1024"], False, "fast", 1024),
    ],
)
def test_configure_applies_the_resolved_arguments(argv, deterministic, variant, chunk_size):
    args = _served(argv, deterministic_mode=deterministic)
    miles_ext_kernels.validate_args(args)
    # miles hands every process the resolved values of the kernel arguments (miles_ext_kernels_args)
    values = {name: getattr(args, name) for name in ("cake_lm_head_variant", "cake_lm_head_chunk_size")}

    for _ in range(2):
        miles_ext_kernels.configure(values)
        assert cake.variant("lm_head") == variant
        assert cake.option("lm_head", "cake_lm_head_chunk_size") == chunk_size


def test_configure_rejects_fast_under_deterministic_mode_and_an_unknown_variant():
    with pytest.raises(ValueError, match=f"{VARIANT} fast is not the det variant"):
        cake.configure(deterministic_mode=True, cake_lm_head_variant="fast")
    with pytest.raises(ValueError, match=f"{VARIANT} must be one of"):
        cake.configure(cake_lm_head_variant="fastest")


def test_configure_starts_from_the_defaults_every_time():
    cake.configure(cake_lm_head_variant="fast", cake_lm_head_chunk_size=1024)
    cake.configure()

    assert cake.variant("lm_head") == "det"
    assert cake.option("lm_head", "cake_lm_head_chunk_size") == lm_head.DEFAULT_CHUNK_SIZE
