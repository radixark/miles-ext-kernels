"""The launch plan of the vendored GQA kernels, without a device (torch-free).

``cake_jit`` (the registry, the grids, the workspace chunks and the scratch layout; it imports only the standard
library) is loaded by path, so neither torch nor the vendored package's loader is imported; ``MIN_CUDA`` comes from the
torch-free runtime. Checked: every stage ``ops.STAGES`` launches has exactly one library per target architecture, which
the oldest toolkit the runtime accepts for that architecture builds (``MIN_CUDA``: CUDA 13.4 on every architecture);
the forward grid covers every tile; the backward's workspace chunks and their grids cover every key head and the key
blocks of ``max_seqlen_k`` and no more (the ``max_seqlen_k`` contract), or of ``T_k`` without it, within the workspace
budget, and the budgets the GPU tests force split a miles micro-batch into the chunks they mean to reach; the
dS^T workspace of a long sequence at the default budget is the one the gqa module documents; the backward scratch is
aligned and sized.
"""

import ast
import importlib.util
import json
import random
from pathlib import Path

import pytest

from miles_ext_kernels.cake import MIN_CUDA

from . import _chunks

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "miles_ext_kernels" / "cake" / "_vendor" / "cake_gqa_nondet_train"
OPS = REPO_ROOT / "miles_ext_kernels" / "cake" / "_impl" / "gqa" / "ops.py"
# ops.STAGES, the forward and the BF16 dK / dV backward, which is the path the entry point runs
STAGES = ("fwd", "bwd_pre", "bwd_plan", "bwd_main_bf16", "bwd_dq")
# target architecture -> the oldest toolkit the runtime builds it with (README, Install); no library may ask for a
# newer one, which that toolkit would refuse
TOOLKIT = dict(MIN_CUDA)
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
D = 256


@pytest.fixture(scope="module")
def cake_jit():
    path = PACKAGE_DIR / "cake_jit.py"
    assert path.is_file(), f"the vendored package is missing: {PACKAGE_DIR}"
    spec = importlib.util.spec_from_file_location("vendored_gqa_cake_jit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest():
    return json.loads((PACKAGE_DIR / "build.json").read_text())


def test_ops_launches_the_stages_checked_here():
    # ops imports torch: read its STAGES from the source
    tree = ast.parse(OPS.read_text())
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["STAGES"]
    ]
    assert values == [STAGES]


def test_every_stage_has_one_library_per_target_arch_the_toolkit_builds(cake_jit, manifest):
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    assert manifest["schema"] == "cake-build/1"
    assert sorted(manifest["arch_flags"]) == sorted(TOOLKIT) == sorted(ARCHES)
    for arch in ARCHES:
        for stage in STAGES:
            serving = [
                name for name, rec in cake_jit.MODULES.items() if rec["stage"] == stage and arch in rec["arches"]
            ]
            assert serving == [cake_jit.stage_record(stage, arch)], (stage, arch, serving)
            entry = entries[f"{cake_jit.MODULES[serving[0]]['library']}_{arch}"]
            assert entry["arch"] == arch
            assert tuple(int(part) for part in entry["min_cuda"].split(".")) <= TOOLKIT[arch], entry["name"]
            launch = cake_jit.MODULES[serving[0]]["launch"]
            assert launch["cooperative"] is False and launch["use_pdl"] is False, stage


def test_grids_cover_every_tile(cake_jit):
    rng = random.Random(7073)
    for _ in range(500):
        n = rng.randint(1, 40)
        q_lens = [rng.choice((0, 1, rng.randint(1, 600), rng.randint(1, 9000))) for _ in range(n)]
        k_lens = [rng.choice((0, q, q + rng.randint(0, 20000))) for q in q_lens]
        h_q, h_kv = rng.choice(((24, 4), (8, 2), (6, 1), (16, 2), (8, 1)))
        dims = dict(
            total_q=sum(q_lens),
            total_k=sum(k_lens),
            num_heads_q=h_q,
            num_heads_kv=h_kv,
            num_segments=n,
            sm_count=148,
        )
        fwd_tiles = sum(-(-q // cake_jit.FWD_TILE_ROWS) for q in q_lens)
        bound = dims["total_q"] // cake_jit.FWD_TILE_ROWS + n
        assert bound >= fwd_tiles
        # the forward is persistent: any pair count >= 1 covers every tile, capped at one pair per two SMs
        assert cake_jit.grid("fwd", **dims) == (2 * max(1, min(148 // 2, h_q * bound)), 1, 1)
        # the backward: per workspace chunk, a CTA pair per (128-key block of the chunk, key head, segment); the chunks
        # cover every key head and the key blocks of the longest segment (of T_k without a bound), rounded up to even
        longest = max(-(-k // cake_jit.BWD_TILE_KEYS) for k in k_lens)
        plan_dims = {name: value for name, value in dims.items() if name != "sm_count"}
        for max_k, blocks in ((max(k_lens), longest), (None, -(-dims["total_k"] // cake_jit.BWD_TILE_KEYS))):
            blocks = max(1, blocks)
            blocks += blocks & 1
            # the vendored default, which ops.PoolWorkspace keeps, and a budget that splits most layouts here
            for budget in (cake_jit.DEFAULT_WORKSPACE_BUDGET, 64 << 20):
                plan = cake_jit.chunk_plan(**plan_dims, max_seqlen_k=max_k, budget_bytes=budget)
                assert plan["key_blocks"] == blocks, (max_k, budget, plan)
                ranges = plan["ranges"]
                assert [n0 for n0, _ in ranges] == [0] + [n1 for _, n1 in ranges[:-1]] and ranges[-1][1] == blocks
                assert all(n0 % 2 == 0 and n1 > n0 for n0, n1 in ranges), ranges
                h_chunk = plan["h_chunk"]
                assert 1 <= h_chunk <= h_kv
                # the dS^T workspace stays within the budget unless one key head and two key blocks exceed it
                assert plan["ws_tiles"] * cake_jit.DS_TILE_BYTES <= budget or (h_chunk, plan["n_chunk"]) == (1, 2)
                # the grids depend on the chunk's extent only
                heads = {min(h_chunk, h_kv - h0) for h0 in range(0, h_kv, h_chunk)}
                for width, hc in {(n1 - n0, hc) for n0, n1 in ranges for hc in heads}:
                    chunk = dict(chunk_blocks=width, chunk_heads=hc)
                    assert cake_jit.grid("bwd_main_bf16", **dims, **chunk) == (2 * width, hc, n)
                    # the dQ GEMM is persistent over the Q-tile pairs of the chunk's head pairs
                    items = cake_jit.pair_bound(dims["total_q"], n) * hc * (h_q // h_kv // 2)
                    assert cake_jit.grid("bwd_dq", **dims, **chunk) == (2 * max(1, min(148 // 2, items)), 1, 1)
        assert cake_jit.pair_bound(dims["total_q"], n) >= sum(-(-q // (2 * cake_jit.BWD_TILE_ROWS)) for q in q_lens)
        rows = cake_jit.rows_pad(dims["total_q"], n)
        assert rows % 128 == 0 and rows >= sum(-(-2 * q // 128) * 128 for q in q_lens)
        assert cake_jit.grid("bwd_pre", **dims) == (rows // 32, h_q // 2, 1)
        assert cake_jit.grid("bwd_plan", **dims) == (1, 1, 1)


@pytest.mark.parametrize("longest", [128, 129, 1024, 1792, 1793, 1920, 2048])
def test_the_backward_grid_covers_the_key_blocks_of_max_seqlen_k_only(cake_jit, longest):
    """Why ``max_seqlen_k`` is a contract: the grids have the key blocks of the value given (rounded up to an even
    count), so a segment with more keys than that has key blocks no CTA pair visits (their dK / dV are never
    written)."""
    dims = dict(total_q=8 * 2048, total_k=8 * 2048, num_heads_q=16, num_heads_kv=2, num_segments=8)
    budget = cake_jit.DEFAULT_WORKSPACE_BUDGET
    plan = cake_jit.chunk_plan(**dims, max_seqlen_k=longest, budget_bytes=budget)
    blocks = sum(
        cake_jit.grid("bwd_main_bf16", **dims, sm_count=148, chunk_blocks=n1 - n0, chunk_heads=plan["h_chunk"])[0] // 2
        for n0, n1 in plan["ranges"]
    )

    assert blocks == plan["key_blocks"] == 2 * -(-longest // (2 * cake_jit.BWD_TILE_KEYS))
    assert (blocks * cake_jit.BWD_TILE_KEYS >= 2048) == (longest > 1792)
    # without a bound the grids cover T_k keys: correct for any segment, with pairs that only retire
    unbounded = cake_jit.chunk_plan(**dims, max_seqlen_k=None, budget_bytes=budget)
    assert unbounded["key_blocks"] == 8 * 2048 // cake_jit.BWD_TILE_KEYS


@pytest.mark.parametrize("heads", [(16, 2), (8, 2), (24, 4)], ids=["16_2", "8_2", "24_4"])
def test_the_gpu_test_budgets_chunk_the_backward(cake_jit, heads):
    """The default budget runs the miles micro-batch in one chunk; the GPU test's ``"kv_heads"`` budget runs one
    key/value head per chunk with every key block, its ``"key_blocks"`` budget one key/value head and ranges of
    ``KEY_BLOCKS_PER_CHUNK`` key blocks whose dQ partials carry over between chunks."""
    h_q, h_kv = heads
    segments = _chunks.MILES_SEGMENTS
    dims = dict(
        total_q=sum(segments),
        total_k=sum(segments),
        num_heads_q=h_q,
        num_heads_kv=h_kv,
        num_segments=len(segments),
        max_seqlen_k=max(segments),
    )
    budgets = _chunks.budgets(cake_jit, **dims)
    blocks = cake_jit.key_blocks(dims["total_k"], dims["max_seqlen_k"])
    width = _chunks.KEY_BLOCKS_PER_CHUNK
    assert blocks > 2 * width

    default = cake_jit.chunk_plan(**dims, budget_bytes=cake_jit.DEFAULT_WORKSPACE_BUDGET)
    assert (default["h_chunk"], default["ranges"]) == (h_kv, [(0, blocks)])
    by_head = cake_jit.chunk_plan(**dims, budget_bytes=budgets["kv_heads"])
    assert (by_head["h_chunk"], by_head["ranges"]) == (1, [(0, blocks)])
    assert by_head["ws_tiles"] * cake_jit.DS_TILE_BYTES <= budgets["kv_heads"]
    by_range = cake_jit.chunk_plan(**dims, budget_bytes=budgets["key_blocks"])
    assert by_range["h_chunk"] == 1
    assert by_range["ranges"] == [(n0, min(n0 + width, blocks)) for n0 in range(0, blocks, width)]
    assert by_range["ws_tiles"] * cake_jit.DS_TILE_BYTES <= budgets["key_blocks"]


def test_a_long_sequence_takes_the_documented_ds_workspace(cake_jit):
    """The gqa module's figure: one 32768-token sequence with 8 query and 2 key/value heads (all of dS^T 16 GiB, one
    tile more than the default budget holds) runs one key/value head per chunk, an 8 GiB chunk per backward."""
    dims = dict(total_q=32768, total_k=32768, num_heads_q=8, num_heads_kv=2, num_segments=1, max_seqlen_k=32768)
    plan = cake_jit.chunk_plan(**dims, budget_bytes=cake_jit.DEFAULT_WORKSPACE_BUDGET)

    assert (plan["key_blocks"], plan["h_chunk"], plan["ranges"]) == (256, 1, [(0, 256)])
    assert plan["ws_tiles"] * cake_jit.DS_TILE_BYTES == (8 << 30) + cake_jit.DS_TILE_BYTES


def test_scratch_layout_is_aligned_and_sized(cake_jit):
    for total_q, heads, segs in ((1, 24, 1), (30193, 24, 12), (77, 8, 3), (7744, 6, 4), (3584, 16, 4)):
        rows = cake_jit.rows_pad(total_q, segs)
        layout = cake_jit.scratch_layout(total_q, heads, segs)
        assert layout["dq_accum"][1] == (heads // 2, rows, D)
        assert layout["lse_log2"][1] == layout["dpsum"][1] == (heads // 2, rows)
        for offset, _shape in layout.values():
            assert offset % cake_jit.SCRATCH_ALIGN == 0
        floats = (heads // 2) * rows * (D + 2)
        assert 4 * floats <= cake_jit.scratch_bytes(total_q, heads, segs) <= 4 * floats + 3 * cake_jit.SCRATCH_ALIGN
