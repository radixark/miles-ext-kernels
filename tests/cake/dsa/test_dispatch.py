"""Host-side behaviour of the DSA ops without a GPU (needs torch, not CUDA): which det stage a shape selects, how each
det stage binds its arguments, which calls the fast variant serves and how the others run det, the required softmax
scale, the segment count the fast backward receives, the index ranges, the memory plan of the det backward, empty
calls, the indexer head counts and the gradient mask of the selected scores."""

import contextlib
import inspect
import json
import logging
import re
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import dsa  # noqa: E402
from miles_ext_kernels.cake._impl.dsa import kernels, layouts, native, ops, reduction  # noqa: E402
from miles_ext_kernels.cake._runtime import config  # noqa: E402

GIB = 1 << 30
TC_STAGES = {f"dsa_attention_fwd_tc_h{h}_d512t{t}" for h in (8, 16, 32, 64) for t in (64, 0)}
TC_BWD_STAGES = {f"dsa_attention_bwd_tc_h{h}_d512t{t}" for h in (8, 16, 32, 64) for t in (64, 0)}
INDEXER_TC_STAGES = {f"dsa_indexer_logits_tc_h{h}" for h in (8, 16, 32, 64)}
INDEXER_TC_BWD_STAGES = {f"dsa_indexer_bwd_tc_h{h}" for h in (32, 64)}
SCALE = 0.0722  # a model's softmax scale, not d_qk ** -0.5
DET_STAGES = sorted(
    json.loads((Path(native.__file__).parents[2] / "_vendor" / native.PACKAGE / "registry.json").read_text())
)


# --- the det stages a shape selects ---


def test_the_indexer_backward_selects_the_tcgen05_stage_for_32_and_64_heads(monkeypatch):
    monkeypatch.setattr(kernels, "INDEXER_BWD_TC", True)
    assert kernels.INDEXER_TC_BWD_HEADS == (32, 64)
    assert kernels.INDEXER_TC_BWD_MIN_TOPK == {32: 1024, 64: 0}
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert kernels.indexer_tc_bwd_stage(32, 2048, arch) == "dsa_indexer_bwd_tc_h32"
        assert kernels.indexer_tc_bwd_stage(32, 1024, arch) == "dsa_indexer_bwd_tc_h32"
        assert kernels.indexer_tc_bwd_stage(64, 512, arch) == "dsa_indexer_bwd_tc_h64"
        assert kernels.indexer_tc_bwd_stage(64, 128, arch) == "dsa_indexer_bwd_tc_h64"
        for heads in (32, 64):
            assert kernels.indexer_tc_bwd_stage(heads, 2048, arch) in INDEXER_TC_BWD_STAGES
        assert kernels.indexer_tc_bwd_stage(32, 512, arch) is None
        assert kernels.indexer_tc_bwd_stage(32, 992, arch) is None
        for heads in (1, 8, 16, 17, 24, 33, 48, 65, 128):
            assert kernels.indexer_tc_bwd_stage(heads, 2048, arch) is None
    assert kernels.indexer_tc_bwd_stage(32, 2048, "sm_90a") is None
    monkeypatch.setattr(kernels, "INDEXER_BWD_TC", False)
    assert kernels.indexer_tc_bwd_stage(32, 2048, "sm_100a") is None


def test_the_attention_backward_selects_the_tcgen05_stages_for_8_16_32_and_64_heads(monkeypatch):
    monkeypatch.setattr(kernels, "ATTENTION_BWD_TC", True)
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for heads in (8, 16, 32, 64):
            for tail in (64, 0):
                stage = kernels.attention_tc_bwd_stage(heads, tail, arch)
                assert stage == f"dsa_attention_bwd_tc_h{heads}_d512t{tail}" and stage in TC_BWD_STAGES
        for heads in (1, 4, 12, 24, 40, 48, 65, 128):
            assert kernels.attention_tc_bwd_stage(heads, 0, arch) is None
    assert kernels.attention_tc_bwd_stage(64, 0, "sm_90a") is None
    monkeypatch.setattr(kernels, "ATTENTION_BWD_TC", False)
    assert kernels.attention_tc_bwd_stage(64, 0, "sm_100a") is None
    assert kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, False, True).kv_partials == 1
    assert kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, True).kv_partials == 1
    assert kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, False).kv_partials == 2
    assert kernels.AttentionGeometry(16, 512, 64, 16, 1, 2048, False, True).d_qk == 576


def test_the_attention_forward_and_indexer_logits_select_the_tcgen05_stages(monkeypatch):
    monkeypatch.setattr(kernels, "ATTENTION_TC", True)
    monkeypatch.setattr(kernels, "INDEXER_TC", True)
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for heads in (8, 16, 32, 64):
            for tail in (64, 0):
                assert kernels.attention_tc_stage(heads, tail, arch) in TC_STAGES
            assert kernels.indexer_tc_stage(heads, arch) == f"dsa_indexer_logits_tc_h{heads}"
            assert kernels.indexer_tc_stage(heads, arch) in INDEXER_TC_STAGES
            assert kernels.INDEXER_TC_QUERY_COLUMNS % heads == 0
        for heads in (1, 4, 17, 24, 48, 65, 128):
            assert kernels.attention_tc_stage(heads, 0, arch) is None
            assert kernels.indexer_tc_stage(heads, arch) is None
    monkeypatch.setattr(kernels, "ATTENTION_TC", False)
    monkeypatch.setattr(kernels, "INDEXER_TC", False)
    assert kernels.attention_tc_stage(64, 0, "sm_100a") is None
    assert kernels.indexer_tc_stage(32, "sm_100a") is None
    assert kernels.attention_block_h(64) == 32 and kernels.attention_head_blocks(64) == 2


# --- how each det stage binds its arguments ---


@pytest.fixture
def launches(monkeypatch):
    """Every det stage's library replaced by a recorder of its exported entry's arguments (no build, no device)."""
    calls = []

    class Library:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, entry):
            return lambda *args: calls.append((self.name, entry, args))

    real = native._cake_build()
    stub = types.SimpleNamespace(__file__=real.__file__, load_library=Library)
    monkeypatch.setattr(native, "_cake_build", lambda: stub)
    monkeypatch.setattr(native, "_ffi_device", lambda index: f"cuda:{index}")
    monkeypatch.setattr(native, "_raw_stream", lambda index: 0)
    tvm_ffi = types.SimpleNamespace(use_raw_stream=lambda device, stream: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "tvm_ffi", tvm_ffi)
    return calls


def test_the_registry_holds_tma_buffer_arguments():
    plans = {stage: {kind for kind, _ in record["arg_plan"]} for stage, record in native.records().items()}

    assert sorted(plans) == DET_STAGES
    assert set().union(*plans.values()) == set(native.ARG_KINDS)
    assert {stage for stage, kinds in plans.items() if "tma_buffer" in kinds} == (
        TC_STAGES | TC_BWD_STAGES | INDEXER_TC_STAGES | INDEXER_TC_BWD_STAGES
    )


@pytest.mark.parametrize("stage", DET_STAGES)
def test_every_det_stage_binds_its_named_arguments_in_the_exported_order(launches, stage):
    """A tma_buffer is a named tensor binding like a buffer: the stage's launcher encodes its TMA descriptor."""
    plan = native.records()[stage]["arg_plan"]
    bindings = {name: f"<{kind} {name}>" for kind, name in plan if kind != "grid"}
    grid = {"grid_x": 3, "grid_y": 5, "grid_z": 7}

    native.NativeKernel(stage, "sm_103a").launch(0, grid=(3, 5, 7), **bindings)

    [(library, entry, args)] = launches
    assert library == native.library_name(stage, "sm_103a") and entry == native.records()[stage]["ffi_entry"]
    assert list(args) == [grid[name] if kind == "grid" else bindings[name] for kind, name in plan]


def test_a_stage_rejects_a_missing_or_an_unexpected_binding(launches):
    stage = "dsa_indexer_logits_tc_h32"
    plan = native.records()[stage]["arg_plan"]
    bindings = {name: 0 for kind, name in plan if kind != "grid"}
    kernel = native.NativeKernel(stage, "sm_100a")

    with pytest.raises(KeyError, match="missing binding 'index_q'"):
        kernel.launch(0, grid=(1,), **{name: value for name, value in bindings.items() if name != "index_q"})
    with pytest.raises(KeyError, match="unexpected bindings"):
        kernel.launch(0, grid=(1,), extra=1, **bindings)
    assert launches == []


def test_an_argument_kind_the_launcher_does_not_bind_is_rejected(launches, monkeypatch):
    record = {**native.records()["dsa_indexer_clean"], "arg_plan": [["workspace", "scratch"], ["grid", "grid_x"]]}
    monkeypatch.setattr(native, "records", lambda: {"dsa_indexer_clean": record})

    with pytest.raises(NotImplementedError, match="argument kinds"):
        native.NativeKernel("dsa_indexer_clean", "sm_100a")


# --- the calls fast serves ---


@pytest.mark.parametrize("d_qk", [576, 512], ids=["rope", "nope"])
def test_fast_serves_64_heads_without_a_sink_at_d_v_512(d_qk):
    assert ops.fast_unserved_reason(64, d_qk, 512, has_sink=False, num_kv=4096) is None


@pytest.mark.parametrize(
    ("heads", "d_qk", "d_v", "has_sink", "num_kv", "reason"),
    [
        (64, 512, 512, True, 64, "calls with an attention sink"),
        (16, 576, 512, False, 64, "16 local query heads (only 64)"),
        (24, 576, 512, False, 64, "24 local query heads (only 64)"),
        (8, 512, 512, True, 64, "calls with an attention sink"),
        (128, 576, 512, False, 64, "128 local query heads (only 64)"),
        (64, 576, 448, False, 64, "d_qk 576 with d_v 448"),
        (64, 640, 512, False, 64, "d_qk 640 with d_v 512"),
        (64, 576, 512, False, 0, "an empty key set"),
    ],
)
def test_fast_names_why_it_does_not_serve_a_call(heads, d_qk, d_v, has_sink, num_kv, reason):
    assert ops.fast_unserved_reason(heads, d_qk, d_v, has_sink=has_sink, num_kv=num_kv).startswith(reason)


@pytest.fixture
def routes(monkeypatch):
    """Record the route ``sparse_attention`` takes and what it hands it, without kernels (CPU tensors)."""
    taken = []

    def fast(q, kv, indices, sm_scale, layout, num_segments):
        taken.append(("fast", layout, sm_scale, num_segments))
        return "fast"

    class Det:
        @staticmethod
        def apply(q, kv, attn_sink, indices, sm_scale, layout, d_v, workspace_bytes):
            taken.append(("det", layout, sm_scale, workspace_bytes))
            return "det"

    monkeypatch.setattr(ops, "_fast_sparse_attention", fast)
    monkeypatch.setattr(ops, "SparseAttentionFunction", Det)
    monkeypatch.setattr(config, "_warned", set())
    return taken


def _thd(heads, d_qk=576, rows=4):
    q = torch.zeros(rows, heads, d_qk, dtype=torch.bfloat16)
    kv = torch.zeros(rows, 1, d_qk, dtype=torch.bfloat16)
    return q, kv, torch.zeros(rows, 1, 64, dtype=torch.int32)


def test_the_configured_variant_is_read_when_the_call_runs(routes):
    q, kv, idx = _thd(64)

    assert dsa.sparse_attention(q, kv, idx, sm_scale=SCALE) == "det"  # unconfigured: det
    cake.configure(cake_dsa_variant="fast")
    assert dsa.sparse_attention(q, kv, idx, sm_scale=SCALE) == "fast"
    cake.configure(cake_dsa_variant="det")
    assert dsa.sparse_attention(q, kv, idx, sm_scale=SCALE) == "det"
    cake.configure(deterministic_mode=True)
    assert dsa.sparse_attention(q, kv, idx, sm_scale=SCALE) == "det"
    assert [route for route, *_ in routes] == ["det", "fast", "det", "det"]
    assert {scale for _, _, scale, _ in routes} == {SCALE}


def test_the_call_takes_no_variant_argument():
    assert "variant" not in inspect.signature(ops.sparse_attention).parameters


@pytest.mark.parametrize("function", ["sparse_attention", "sparse_attention_forward", "sparse_attention_backward"])
def test_sm_scale_is_a_required_keyword(function):
    parameter = inspect.signature(getattr(ops, function)).parameters["sm_scale"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_a_call_without_sm_scale_raises_on_every_route(routes):
    q, kv, idx = _thd(64)
    for variant in ("det", "fast"):
        cake.configure(cake_dsa_variant=variant)
        with pytest.raises(TypeError, match="sm_scale"):
            dsa.sparse_attention(q, kv, idx)
        with pytest.raises(TypeError, match="sm_scale is required"):
            dsa.sparse_attention(q, kv, idx, sm_scale=None)
    with pytest.raises(TypeError, match="sm_scale is required"):
        dsa.sparse_attention_forward(q, kv, idx, sm_scale=None)
    assert routes == []


def test_fast_receives_the_segment_count(routes):
    cake.configure(cake_dsa_variant="fast")
    q, kv, idx = _thd(64)
    batched = (
        torch.zeros(3, 4, 64, 512, dtype=torch.bfloat16),
        torch.zeros(3, 5, 512, dtype=torch.bfloat16),
        torch.zeros(3, 4, 64, dtype=torch.int32),
    )

    dsa.sparse_attention(q, kv, idx, sm_scale=SCALE)
    dsa.sparse_attention(q, kv, idx, sm_scale=SCALE, num_segments=8)
    dsa.sparse_attention(*batched, sm_scale=SCALE, layout="bshd")
    dsa.sparse_attention(*batched, sm_scale=SCALE, layout="bshd", num_segments=2)

    assert routes == [
        ("fast", "thd", SCALE, 1),
        ("fast", "thd", SCALE, 8),
        ("fast", "bshd", SCALE, 3),
        ("fast", "bshd", SCALE, 2),
    ]


@pytest.mark.parametrize("segments", [0, -1])
def test_a_segment_count_below_one_is_rejected(routes, segments):
    with pytest.raises(ValueError, match="num_segments must be >= 1"):
        dsa.sparse_attention(*_thd(64), sm_scale=SCALE, num_segments=segments)
    assert routes == []


def test_det_receives_the_call_budget(routes):
    dsa.sparse_attention(*_thd(64), sm_scale=SCALE)
    dsa.sparse_attention(*_thd(64), sm_scale=SCALE, workspace_bytes=GIB)

    assert [budget for *_, budget in routes] == [None, GIB]


def test_a_call_fast_does_not_serve_runs_det_and_warns_once_per_reason(routes, caplog):
    cake.configure(cake_dsa_variant="fast")
    sink = torch.zeros(64)
    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        for _ in range(3):
            assert dsa.sparse_attention(*_thd(64), sm_scale=SCALE, attn_sink=sink) == "det"
            assert dsa.sparse_attention(*_thd(24), sm_scale=SCALE) == "det"
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "miles_ext_kernels.cake.dsa: the fast variant does not serve calls with an attention sink; running det",
        "miles_ext_kernels.cake.dsa: the fast variant does not serve 24 local query heads (only 64); running det",
    ]
    assert [route for route, *_ in routes] == ["det"] * 6


def test_non_bf16_inputs_are_rejected_on_every_route(routes):
    q, kv, idx = _thd(64)
    for variant in ("det", "fast"):
        cake.configure(cake_dsa_variant=variant)
        with pytest.raises(TypeError, match="BF16"):
            dsa.sparse_attention(q.float(), kv, idx, sm_scale=SCALE)
    assert routes == []


# --- index ranges ---


def test_batch_local_indices_outside_their_keys_become_empty_slots():
    # batch 0 index 5 (S_kv + 1) would otherwise reach batch 1's key 1
    local = torch.tensor([[[0, 3, 5, -1]], [[4, 0, -2, 1]]], dtype=torch.int32)

    flat = layouts.globalize_indices(local, 4)

    assert flat.tolist() == [[0, 3, -1, -1], [-1, 4, -1, 5]]


def test_batch_one_indices_stay_as_given():
    local = torch.tensor([[[0, 9, -1]]], dtype=torch.int64)

    flat = layouts.globalize_indices(local, 4)

    assert flat.dtype == torch.int32 and flat.tolist() == [[0, 9, -1]]


def test_indexer_indices_past_the_keys_become_empty_slots():
    meta = layouts.RowMeta("thd", 1, 2, 6)
    indices = torch.tensor([[0, 5, 6, -1], [7, 2, -3, 1]], dtype=torch.int64)

    assert layouts.flatten_topk_indices(indices, meta).tolist() == [[0, 5, -1, -1], [-1, 2, -1, 1]]


def test_selected_scores_past_the_last_key_are_minus_infinity():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    scores = ops.extract_topk_scores(logits, torch.tensor([[2, -1, 3, 0]]))

    assert scores.tolist() == [[3.0, float("-inf"), float("-inf"), 1.0]]


# --- the memory plan of the det backward ---


def _peak(plan, num_rows, pairs_per_row, dim, num_out):
    """The transient bytes the plan's backward holds at its peak (the accounting of reduction.py)."""
    rows = plan.rows_per_chunk
    partials = rows * pairs_per_row * dim * 4
    bounds = num_out * reduction.BOUNDS_BYTES_PER_KEY
    if plan.one_sort:
        pairs = num_rows * pairs_per_row
        wide = num_out * num_rows >= 2**31
        kept = pairs * (reduction.KEPT_BYTES_PER_PAIR_INT64 if wide else reduction.KEPT_BYTES_PER_PAIR)
        sort = pairs * (reduction.SORT_BYTES_PER_PAIR_INT64 if wide else reduction.SORT_BYTES_PER_PAIR)
        return max(sort, kept + partials + bounds)
    return partials + rows * pairs_per_row * reduction.SORT_BYTES_PER_PAIR + bounds


@pytest.mark.parametrize(
    ("num_rows", "pairs_per_row", "dim", "num_out", "budget"),
    [
        (8192, 2048, 576, 8192, 256 << 20),  # the reviewed shape: 64 heads (one partial per slot), topk 2048
        (8192, 2048, 576, 8192, 2 * GIB),
        (8192, 4096, 576, 8192, 2 * GIB),  # two head blocks
        (65536, 2048, 576, 65536, 2 * GIB),
        (4096, 2048, 128, 4096, 64 << 20),  # the indexer backward
        (100, 64, 576, 100, 8 << 20),
    ],
)
def test_the_det_backward_keeps_its_partials_and_sort_scratch_within_the_budget(
    num_rows, pairs_per_row, dim, num_out, budget
):
    for one_sort in (True, False):
        plan = reduction.plan_backward(num_rows, pairs_per_row, dim, num_out, budget, one_sort=one_sort)

        assert plan.one_sort <= one_sort
        assert plan.chunks[0][0] == 0 and plan.chunks[-1][1] == num_rows
        assert all(a[1] == b[0] for a, b in zip(plan.chunks, plan.chunks[1:]))
        assert all(end - start <= plan.rows_per_chunk for start, end in plan.chunks)
        assert _peak(plan, num_rows, pairs_per_row, dim, num_out) <= budget


def test_the_reviewed_shape_sorts_chunk_by_chunk_at_256_mib_and_once_at_2_gib():
    small = reduction.plan_backward(8192, 2048, 576, 8192, 256 << 20)
    large = reduction.plan_backward(8192, 2048, 576, 8192, 2 * GIB)

    assert not small.one_sort and small.rows_per_chunk == 55
    assert large.one_sort and large.rows_per_chunk == 426


def test_a_budget_below_one_row_still_runs_one_row_at_a_time_and_no_rows_plan_nothing():
    plan = reduction.plan_backward(10, 2048, 576, 10, 1)

    assert plan.rows_per_chunk == 1 and len(plan.chunks) == 10 and not plan.one_sort
    assert reduction.plan_backward(0, 2048, 576, 10, GIB) == reduction.BackwardPlan([], False)


@pytest.mark.parametrize("budget", [0, -1])
def test_a_non_positive_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="must be positive"):
        ops._budget(budget)


def test_the_budget_defaults_to_the_configured_option():
    assert ops._budget(None) == 2 * GIB
    cake.configure(cake_dsa_bwd_workspace_bytes=GIB)
    assert ops._budget(None) == GIB
    assert ops._budget(123) == 123


def _segments(order, start, count, num_out):
    return [order[int(start[k]) : int(start[k]) + int(count[k])].tolist() for k in range(num_out)]


def test_segment_pairs_drops_keys_outside_the_output_rows():
    keys = torch.tensor([3, -1, 7, 1, 3, 4, 1], dtype=torch.int32)

    order, start, count = reduction.segment_pairs(keys, 4)

    assert order.dtype == start.dtype == count.dtype == torch.int32
    assert _segments(order, start, count, 4) == [[], [3, 6], [], [0, 4]]
    assert int(count.sum()) == 4


@pytest.mark.parametrize("rows_per_chunk", [1, 2, 3, 5])
def test_one_sort_of_every_chunk_gives_the_segments_of_one_sort_per_chunk(rows_per_chunk):
    generator = torch.Generator().manual_seed(0)
    num_rows, blocks, topk, num_out = 5, 2, 6, 7
    indices = torch.randint(-2, num_out + 2, (num_rows, topk), generator=generator, dtype=torch.int32)

    segments = reduction.ChunkedSegments(indices, blocks, rows_per_chunk, num_out)

    for chunk, start in enumerate(range(0, num_rows, rows_per_chunk)):
        rows = min(rows_per_chunk, num_rows - start)
        keys = indices[start : start + rows].unsqueeze(1).expand(rows, blocks, topk).reshape(-1)
        expected = _segments(*reduction.segment_pairs(keys, num_out), num_out)
        assert _segments(*segments.segment(chunk), num_out) == expected


# --- empty calls ---


@pytest.mark.parametrize("sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("layout", ["thd", "bshd"])
def test_an_attention_backward_without_query_rows_returns_empty_and_zero_gradients(layout, sink):
    heads, d_qk, num_kv = 64, 576, 5
    if layout == "thd":
        q = torch.zeros(0, heads, d_qk, dtype=torch.bfloat16)
        kv = torch.ones(num_kv, d_qk, dtype=torch.bfloat16)
        o, idx = torch.zeros(0, heads, 512, dtype=torch.bfloat16), torch.zeros(0, 64, dtype=torch.int32)
        lse = torch.zeros(0, heads)
    else:
        q = torch.zeros(2, 0, heads, d_qk, dtype=torch.bfloat16)
        kv = torch.ones(2, num_kv, d_qk, dtype=torch.bfloat16)
        o, idx = torch.zeros(2, 0, heads, 512, dtype=torch.bfloat16), torch.zeros(2, 0, 64, dtype=torch.int32)
        lse = torch.zeros(2, 0, heads)
    attn_sink = torch.zeros(heads) if sink else None

    dq, dkv, dsink = dsa.sparse_attention_backward(
        q, kv, o, o, idx, lse, sm_scale=SCALE, attn_sink=attn_sink, layout=layout
    )

    assert dq.shape == q.shape and dkv.shape == kv.shape and dkv.dtype == kv.dtype
    assert not dkv.any()
    if sink:
        assert dsink.tolist() == [0.0] * heads
    else:
        assert dsink is None


@pytest.mark.parametrize("layout", ["thd", "bshd", "sbhd"])
def test_an_indexer_backward_without_query_rows_returns_empty_and_zero_gradients(layout):
    heads, dim, num_kv = 32, 128, 7
    if layout == "thd":
        q, k, w = torch.zeros(0, heads, dim), torch.ones(num_kv, dim), torch.zeros(0, heads)
        idx, grad = torch.zeros(0, 64, dtype=torch.int32), torch.zeros(0, 64)
    elif layout == "bshd":
        q, k, w = torch.zeros(2, 0, heads, dim), torch.ones(2, num_kv, dim), torch.zeros(2, 0, heads)
        idx, grad = torch.zeros(2, 0, 64, dtype=torch.int32), torch.zeros(2, 0, 64)
    else:
        q, k, w = torch.zeros(0, 2, heads, dim), torch.ones(num_kv, 2, dim), torch.zeros(0, 2, heads)
        idx, grad = torch.zeros(2, 0, 64, dtype=torch.int32), torch.zeros(2, 0, 64)

    dq, dw, dk = dsa.indexer_backward(q.bfloat16(), k.bfloat16(), w, idx, grad, layout=layout)

    assert dq.shape == q.shape and dw.shape == w.shape and dk.shape == k.shape
    assert dk.dtype == torch.float32 and not dk.any()


@pytest.mark.parametrize("num_kv", [0, 7], ids=["no-keys", "keys"])
@pytest.mark.parametrize(
    ("layout", "batch", "seq_len"),
    [("thd", 1, 0), ("bshd", 2, 0), ("bshd", 0, 3), ("sbhd", 2, 0), ("sbhd", 0, 3)],
    ids=["thd-T0", "bshd-S0", "bshd-B0", "sbhd-S0", "sbhd-B0"],
)
def test_a_lightning_indexer_call_without_query_rows_runs_forward_and_backward(layout, batch, seq_len, num_kv):
    """No query rows launch nothing (thd T = 0, a batched S = 0 or B = 0, with or without keys): the scores and
    indices are empty and every gradient is zero."""
    heads, dim = 32, 128
    if layout == "thd":
        q_shape, k_shape, w_shape, rows = (seq_len, heads, dim), (num_kv, dim), (seq_len, heads), (seq_len,)
    elif layout == "bshd":
        q_shape, k_shape = (batch, seq_len, heads, dim), (batch, num_kv, dim)
        w_shape, rows = (batch, seq_len, heads), (batch, seq_len)
    else:
        q_shape, k_shape = (seq_len, batch, heads, dim), (num_kv, batch, dim)
        w_shape, rows = (seq_len, batch, heads), (batch, seq_len)
    index_q = torch.zeros(q_shape, dtype=torch.bfloat16, requires_grad=True)
    index_k = torch.ones(k_shape, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.zeros(w_shape, requires_grad=True)
    bounds = torch.zeros(seq_len, dtype=torch.int32)

    scores, indices = dsa.lightning_indexer(index_q, index_k, weights, bounds, bounds, 128, layout=layout)
    scores.sum().backward()

    assert scores.shape == indices.shape == (*rows, 128) and indices.dtype == torch.int32
    assert index_q.grad.shape == index_q.shape and weights.grad.shape == weights.shape
    assert index_k.grad.shape == index_k.shape and not index_k.grad.any()


@pytest.mark.parametrize("shape", [(0, 0), (2, 0, 0), (0, 5), (3, 0)])
def test_select_topk_pads_rows_without_keys_with_empty_slots(shape):
    indices = dsa.select_topk(torch.zeros(shape), 4)

    assert indices.shape == (*shape[:-1], 4) and indices.dtype == torch.int32
    assert (indices == -1).all()


# --- indexer head counts ---


@pytest.mark.parametrize(("heads", "head_dim"), [(8, 128), (16, 128), (32, 128), (64, 128)])
def test_the_indexer_serves_8_16_32_and_64_heads_of_128(heads, head_dim):
    assert dsa.indexer_unserved_reason(heads, head_dim) is None


@pytest.mark.parametrize(("heads", "head_dim"), [(24, 128), (12, 128), (128, 128), (32, 64)])
def test_every_indexer_call_rejects_an_unserved_head_count_before_the_device(heads, head_dim):
    rows, num_kv, topk = 3, 5, 4
    q = torch.zeros(rows, heads, head_dim, dtype=torch.bfloat16)
    k = torch.zeros(num_kv, head_dim, dtype=torch.bfloat16)
    w = torch.zeros(rows, heads)
    idx = torch.zeros(rows, topk, dtype=torch.int32)
    cu = torch.zeros(rows, dtype=torch.int32)
    reason = dsa.indexer_unserved_reason(heads, head_dim)
    match = re.escape(f"do not serve {reason}")

    assert reason in (f"{heads} index heads (only 8, 16, 32, 64)", f"index head dimension {head_dim} (only 128)")
    with pytest.raises(ValueError, match=match):
        dsa.indexer_logits(q, k, w, cu, cu)
    with pytest.raises(ValueError, match=match):
        dsa.indexer_backward(q, k, w, idx, torch.zeros(rows, topk))
    with pytest.raises(ValueError, match=match):
        dsa.indexer_topk_scores(q, k, w, idx, torch.zeros(rows, num_kv))
    with pytest.raises(ValueError, match=match):
        dsa.lightning_indexer(q, k, w, cu, cu, topk)


def test_prebuild_rejects_unserved_model_parameters_before_the_device():
    with pytest.raises(ValueError, match="do not serve 24 index heads"):
        cake.prebuild("dsa", heads=64, d_qk=576, index_heads=24)
    with pytest.raises(ValueError, match="unsupported"):
        cake.prebuild("dsa", heads=64, d_qk=640)
    with pytest.raises(ValueError, match="heads must be positive"):
        cake.prebuild("dsa", heads=0, d_qk=576)


# --- the gradient mask of the selected scores ---


@pytest.mark.parametrize("n", [1, 7, 8, 9, 64, 2049])
def test_the_slot_mask_packs_eight_slots_per_byte_and_unpacks_exactly(n):
    mask = torch.rand(3, 2, n, generator=torch.Generator().manual_seed(n)) < 0.3

    packed = ops._pack_bits(mask)

    assert packed.dtype == torch.uint8 and packed.shape == (3, 2, (n + 7) // 8)
    assert torch.equal(ops._unpack_bits(packed, n), mask)


@pytest.mark.parametrize("shape", [(0, 9), (2, 0, 9), (0, 0), (3, 0)])
def test_the_slot_mask_of_a_call_without_rows_or_slots_round_trips(shape):
    mask = torch.zeros(shape, dtype=torch.bool)

    packed = ops._pack_bits(mask)

    assert packed.dtype == torch.uint8 and packed.shape == (*shape[:-1], (shape[-1] + 7) // 8)
    assert torch.equal(ops._unpack_bits(packed, shape[-1]), mask)


def test_slots_with_a_minus_infinity_score_reach_the_backward_as_empty_slots(monkeypatch):
    """A selected key outside its query's window (a -inf score) gets no gradient, even a NaN one."""
    captured = {}

    def backward(index_q, index_k, weights, topk_indices, grad_scores, *, layout, workspace_bytes):
        captured.update(indices=topk_indices.clone(), grad=grad_scores.clone(), budget=workspace_bytes)
        return torch.zeros_like(index_q), torch.zeros(weights.shape), torch.zeros(index_k.shape)

    monkeypatch.setattr(ops, "indexer_backward", backward)
    inf = float("-inf")
    q = torch.zeros(2, 32, 128, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(6, 128, dtype=torch.bfloat16, requires_grad=True)
    w = torch.zeros(2, 32, requires_grad=True)
    # row 0: key 4 is outside its window; row 1: key 9 is past the keys, slot 2 is empty
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, inf, inf], [1.0, 2.0, 3.0, inf, inf, inf]])
    topk = torch.tensor([[4, 0, 2], [1, 9, -1]], dtype=torch.int32)

    scores = dsa.indexer_topk_scores(q, k, w, topk, logits, workspace_bytes=GIB)
    scores.backward(torch.full_like(scores, float("nan")))

    assert scores.tolist() == [[inf, 1.0, 3.0], [2.0, inf, inf]]
    assert captured["indices"].tolist() == [[-1, 0, 2], [1, -1, -1]]
    assert captured["budget"] == GIB
