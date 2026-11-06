"""The deterministic indexer top-k (FlashInfer's Cake ``cake_dsa_indexer_det_train``) on SM100a / SM103a / SM107a.

* accuracy -- packed segments at the recorded training token counts (16231 and 16172 queries in 8 segments, top-k 2048
  and 4096), keys longer than the queries, compressed keys (ratio 4), explicit and negative causal offsets, empty
  segments and rows without keys, top-k 1 to 4096, against an FP64 reference: each row holds ``min(top_k, visible)``
  unique visible keys in ascending order, then ``-1`` / ``-inf``; each score is within the package's FP32 rounding
  bound ``gamma_388 * A`` of the reference; and no unselected key beats a selected one by more than the two bounds.
* exact ties -- integer-valued inputs, whose scores are exact in FP32: the ids and score bits are the reference
  selection's, an equal-score tie keeps the larger key id, and a zero score is ``+0.0``.
* repeatability -- bit-identical ids and scores across calls, in a fresh process, with a caller-owned workspace, with a
  row-strided key view, and on a second GPU (two-device test).
* the det indexer logits -- with ``softmax_scale=1.0`` the scores at the packed ids match ``ops.indexer_logits``, the
  selection is its top-k up to rounding, and the packed ids drive the det indexer backward.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (10, 7)}:
    pytest.skip("the Cake DSA indexer top-k kernels target SM100a / SM103a / SM107a", allow_module_level=True)

from miles_ext_kernels.cake._impl.dsa import indexer_topk, ops  # noqa: E402
from miles_ext_kernels.cake._runtime import loader  # noqa: E402

from ._indexer_topk_reference import (  # noqa: E402
    GAMMA,
    REPEAT_CASES,
    check_selection,
    expected_selection,
    make_inputs,
    run_case,
    segment_lengths,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_SCRIPT = Path(__file__).resolve().parent / "_indexer_topk_reference.py"
DEVICE = "cuda"


def _call(x, top_k, **kwargs):
    return indexer_topk.dsa_indexer_det_topk(x["q"], x["k"], x["w"], x["cu_q"], x["cu_k"], top_k, **kwargs)


def _bits(indices, scores):
    return indices.cpu(), scores.cpu().view(torch.int32)


def _assert_same_bits(a, b):
    assert torch.equal(_bits(*a)[0], _bits(*b)[0]), "ids differ"
    assert torch.equal(_bits(*a)[1], _bits(*b)[1]), "score bits differ"


# --------------------------------------------------------------------------- #
# Accuracy against the FP64 reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("seg_q", "seg_k", "top_k", "seed"),
    [
        (segment_lengths(16231, 8, 1), segment_lengths(16231, 8, 1), 2048, 21),
        (segment_lengths(16172, 8, 2), segment_lengths(16172, 8, 2), 4096, 22),
        ([4096], [4096], 2048, 23),
        ([1000, 2500, 600], [1000, 2500, 600], 512, 24),
        ([200, 300], [200, 2500], 128, 25),
        ([700, 900], [700, 900], 1, 26),
        ([5000], [5000], 4096, 27),
    ],
    ids=["recorded_16231", "recorded_16172_k4096", "one_4096", "three_segments", "prefix_keys", "k1", "k4096"],
)
def test_matches_the_fp64_reference_within_the_rounding_bound(seg_q, seg_k, top_k, seed):
    x = make_inputs(seg_q, seg_k, seed, device=DEVICE)

    indices, scores = _call(x, top_k)

    check_selection(x, indices, scores, top_k, scale=128**-0.5)


def test_compressed_keys_see_floor_of_position_over_ratio():
    x = make_inputs([2048, 1024], [512, 256], 31, device=DEVICE)

    indices, scores = _call(x, 512, ratio=4)

    check_selection(x, indices, scores, 512, scale=128**-0.5, ratio=4)


@pytest.mark.parametrize("offsets", [[300, 1100], [-150, 0], [0, -5000]])
def test_explicit_causal_offsets_set_each_rows_visible_prefix(offsets):
    x = make_inputs([1024, 512], [2048, 1536], 32, device=DEVICE)
    offsets = torch.tensor(offsets, dtype=torch.int64, device=DEVICE)

    indices, scores = _call(x, 1024, q_causal_offsets=offsets)

    check_selection(x, indices, scores, 1024, scale=128**-0.5, offsets=offsets)


def test_empty_segments_and_rows_without_keys_are_padding():
    x = make_inputs([100, 0, 200, 50], [0, 50, 400, 0], 33, device=DEVICE)

    indices, scores = _call(x, 256)

    check_selection(x, indices, scores, 256, scale=128**-0.5)
    assert torch.all(indices[:100] == -1) and torch.all(indices[300:] == -1)


def test_a_call_without_queries_or_without_keys():
    x = make_inputs([0], [64], 34, device=DEVICE)
    indices, scores = _call(x, 64)
    assert tuple(indices.shape) == tuple(scores.shape) == (0, 64)

    x = make_inputs([64], [0], 35, device=DEVICE)
    indices, scores = _call(x, 64)
    assert torch.all(indices == -1) and torch.all(scores == float("-inf"))


# --------------------------------------------------------------------------- #
# Exact ties
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("seg", "top_k"), [([3000], 2048), ([700, 1300], 256), ([4608], 4096)])
def test_exact_scores_and_ties_to_the_larger_key_id(seg, top_k):
    x = make_inputs(seg, seg, 41, device=DEVICE, integer=True)

    indices, scores = _call(x, top_k, softmax_scale=1.0)

    check_selection(x, indices, scores, top_k, scale=1.0)
    want_indices, want_scores, cutoff_ties = expected_selection(x, top_k, scale=1.0)
    assert torch.equal(indices.long(), want_indices), "the selection is not score descending, then key id descending"
    assert torch.equal(scores.view(torch.int32), want_scores.view(torch.int32)), "score bits differ"
    assert cutoff_ties > 0, "no row has a tie at its cutoff"


# --------------------------------------------------------------------------- #
# Repeatability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", sorted(REPEAT_CASES))
def test_repeats_bit_for_bit_across_calls_and_in_a_fresh_process(case):
    first = run_case(case)
    assert run_case(case) == first

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run(
        [sys.executable, str(REFERENCE_SCRIPT), case], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {case: first}


def test_a_caller_owned_workspace_gives_the_same_bits_and_a_short_one_is_rejected():
    seg_a, seg_b = segment_lengths(16231, 8, 1), [4096]
    xa = make_inputs(seg_a, seg_a, 51, device=DEVICE)
    xb = make_inputs(seg_b, seg_b, 52, device=DEVICE)
    sizes = [
        indexer_topk.dsa_indexer_det_topk_workspace_size(sum(s), sum(s), len(s), top_k=2048) for s in (seg_a, seg_b)
    ]
    workspace = torch.empty(max(sizes), dtype=torch.uint8, device=DEVICE)

    want_a, want_b = _call(xa, 2048), _call(xb, 2048)
    for _ in range(2):  # one buffer shared by two geometries, twice
        _assert_same_bits(_call(xa, 2048, workspace=workspace), want_a)
        _assert_same_bits(_call(xb, 2048, workspace=workspace), want_b)
    with pytest.raises(ValueError, match="workspace_buffer needs"):
        _call(xa, 2048, workspace=workspace[:1024])


def test_caller_owned_outputs_and_a_row_strided_key_view_give_the_same_bits():
    seg = [3000, 2000]
    x = make_inputs(seg, seg, 53, device=DEVICE)
    want = _call(x, 2048)
    packed = torch.randn(x["k"].shape[0], 704, device=DEVICE).to(torch.bfloat16)
    packed[:, :128] = x["k"]
    strided = {**x, "k": packed[:, :128]}
    indices = torch.empty(5000, 2048, dtype=torch.int32, device=DEVICE)
    scores = torch.empty(5000, 2048, dtype=torch.float32, device=DEVICE)

    got = _call(strided, 2048, indices=indices, scores=scores)

    assert got[0] is indices and got[1] is scores
    _assert_same_bits(got, want)


@pytest.mark.multi_gpu
def test_two_devices_give_the_same_bits():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
        pytest.skip("the two GPUs differ in compute capability")
    torch.cuda.set_device(0)
    for case in ("recorded_16231", "prefix_keys"):
        # the call on cuda:1 runs while cuda:0 is the current device
        assert run_case(case, device="cuda:1") == run_case(case, device="cuda:0")


# --------------------------------------------------------------------------- #
# The det indexer logits and backward
# --------------------------------------------------------------------------- #


def test_scale_one_matches_the_det_indexer_logits_and_feeds_the_det_backward():
    seg = [1024, 1536]
    x = make_inputs(seg, seg, 61, device=DEVICE)
    starts = torch.cat([torch.full((n,), sum(seg[:i]), dtype=torch.int32) for i, n in enumerate(seg)]).to(DEVICE)
    ends = torch.arange(1, sum(seg) + 1, dtype=torch.int32, device=DEVICE)
    logits = ops.indexer_logits(x["q"], x["k"], x["w"], starts, ends, layout="thd")

    indices, scores = _call(x, 512, softmax_scale=1.0)
    packed = indexer_topk.packed_key_ids(indices, x["cu_q"], x["cu_k"])

    check_selection(x, indices, scores, 512, scale=1.0)
    valid = packed >= 0
    per_row = valid.sum(1)
    assert torch.all(packed[valid] >= starts.repeat_interleave(per_row))
    assert torch.all(packed[valid] < ends.repeat_interleave(per_row))
    gathered = ops.extract_topk_scores(logits, packed)
    keys = x["k"].double().abs()[packed.clamp(min=0).long()]  # [T, 512, 128]
    mass = (x["w"].double().abs()[:, :, None] * torch.einsum("thd,tsd->ths", x["q"].double().abs(), keys)).sum(1)
    assert torch.all((scores.double() - gathered.double()).abs()[valid] <= (2 * GAMMA * mass)[valid])

    q_, k_, w_ = (t.clone().requires_grad_(True) for t in (x["q"], x["k"], x["w"]))
    selected = ops.indexer_topk_scores(q_, k_, w_, packed, logits, layout="thd")
    grad = torch.randn_like(scores).masked_fill(~valid, 0.0)
    selected.backward(grad)
    assert torch.equal(selected.detach(), gathered)
    for t in (q_, k_, w_):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_prebuild_builds_every_library_of_the_device_architecture():
    arch = loader.device_arch()
    paths = indexer_topk.prebuild_libraries(arch)
    cake_build, _ = indexer_topk._package()

    with loader.no_build():
        built = [cake_build.jit_spec(name).build() for name in indexer_topk.library_names(arch)]

    assert len(built) == (53 if arch == "sm_107a" else 52) and all(Path(path).is_file() for path in built)
    assert [Path(path) for path in paths] == [Path(path) for path in built]
