"""The launch plan of the vendored GQA kernels, without a device (torch-free).

``cake_jit`` (the registry, the grids and the scratch layout; it imports only ``typing``) is loaded by path, so neither
torch nor the vendored package's loader is imported; ``MIN_CUDA`` comes from the torch-free runtime. Checked: every
stage ``ops.STAGES`` launches has exactly one library per target architecture, which the oldest toolkit the runtime
accepts for that architecture builds (``MIN_CUDA``: CUDA 12.8 for SM100a, 12.9 for SM103a, 13.4 for SM107a); the
forward grid covers every tile; the backward grid covers the key blocks of ``max_seqlen_k`` and no more (the
``max_seqlen_k`` contract), or of ``T_k`` without it; the backward scratch is aligned and sized.
"""

import ast
import importlib.util
import json
import random
from pathlib import Path

import pytest

from miles_ext_kernels.cake import MIN_CUDA

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "miles_ext_kernels" / "cake" / "_vendor" / "cake_gqa_nondet_train"
OPS = REPO_ROOT / "miles_ext_kernels" / "cake" / "_impl" / "gqa" / "ops.py"
# ops.STAGES, the forward and the BF16 dK / dV backward, which is the path the entry point runs
STAGES = ("fwd", "bwd_pre", "bwd_bf16", "bwd_post")
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
        # the backward: a CTA pair per (128-key block of the longest segment, key head, segment)
        longest = max(-(-k // cake_jit.BWD_TILE_KEYS) for k in k_lens)
        gx, gy, gz = cake_jit.grid("bwd_bf16", **dims)
        assert gx % 2 == 0 and gx // 2 >= longest and (gy, gz) == (h_kv, n)
        assert cake_jit.grid("bwd_bf16", **dims, max_seqlen_k=max(k_lens)) == (2 * max(1, longest), h_kv, n)
        rows = cake_jit.rows_pad(dims["total_q"], n)
        assert rows % 128 == 0 and rows >= sum(-(-2 * q // 128) * 128 for q in q_lens)
        assert cake_jit.grid("bwd_pre", **dims) == (rows // 32, h_q // 2, 1)
        assert cake_jit.grid("bwd_post", **dims) == (rows // 64, h_q // 2, 1)


@pytest.mark.parametrize("longest", [128, 129, 1024, 1920, 2048])
def test_the_backward_grid_covers_the_key_blocks_of_max_seqlen_k_only(cake_jit, longest):
    """Why ``max_seqlen_k`` is a contract: the grid has the key blocks of the value given, so a segment with more keys
    than that has key blocks no CTA pair visits (their dK / dV are never written)."""
    dims = dict(total_q=8 * 2048, total_k=8 * 2048, num_heads_q=16, num_heads_kv=2, num_segments=8, sm_count=148)
    blocks = cake_jit.grid("bwd_bf16", **dims, max_seqlen_k=longest)[0] // 2

    assert blocks == -(-longest // cake_jit.BWD_TILE_KEYS)
    assert (blocks * cake_jit.BWD_TILE_KEYS >= 2048) == (longest > 1920)
    # without a bound the grid covers T_k keys: correct for any segment, with pairs that only retire
    assert cake_jit.grid("bwd_bf16", **dims)[0] // 2 == 8 * 2048 // cake_jit.BWD_TILE_KEYS


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
