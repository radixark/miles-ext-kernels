"""The miles micro-batch of the gqa tests and the dS^T workspace budgets that make the vendored backward run it in
chunks, shared by the torch-free plan test and the GPU tests.

Torch-free: :func:`budgets` takes the vendored ``cake_jit`` module (imported, or loaded by path).
"""

from __future__ import annotations

from types import ModuleType

# A miles micro-batch: three sequences, then the pad that aligns the packed tokens to a multiple of 128 as its own
# segment; self-attention, so the query and key boundaries are one.
MILES_SEGMENTS = (1000, 517, 2049, 18)
# the key blocks of one chunk under the "key_blocks" budget (even, as the vendored chunks are)
KEY_BLOCKS_PER_CHUNK = 4


def budgets(
    cake_jit: ModuleType,
    *,
    total_q: int,
    total_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    num_segments: int,
    max_seqlen_k: int | None,
) -> dict[str, int]:
    """dS^T budgets in bytes for one backward call without host boundaries.

    ``"kv_heads"``: one key/value head per chunk with every key block (needs two or more key/value heads).
    ``"key_blocks"``: one key/value head per chunk and key-block ranges of :data:`KEY_BLOCKS_PER_CHUNK` (needs more
    key blocks than that). ``cake_jit.chunk_plan`` fits ``budget // DS_TILE_BYTES - 1`` tiles (one is the zero tile).
    """
    pairs = num_heads_q // num_heads_kv // 2
    per_block = pairs * cake_jit.tiles_per_key_block(total_q, num_segments)
    per_head = cake_jit.key_blocks(total_k, max_seqlen_k) * per_block
    return {
        "kv_heads": (per_head + 1) * cake_jit.DS_TILE_BYTES,
        "key_blocks": (KEY_BLOCKS_PER_CHUNK * per_block + 1) * cake_jit.DS_TILE_BYTES,
    }
