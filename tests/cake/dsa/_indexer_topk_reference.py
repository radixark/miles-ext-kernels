"""The FP64 reference of the deterministic indexer top-k GPU tests (torch only; runs on any device).

* :func:`make_inputs` builds the inputs from a CPU generator, so every process builds the same bits.
* :func:`check_selection` checks a call's ``(indices, scores)`` against FP64 scores: the row structure, every score
  within the package's FP32 rounding bound ``gamma_388 * A``, and the exact top-k up to that bound.
* :func:`expected_selection` is the exact selection of integer-valued inputs (exact FP32 scores): score descending,
  then key id descending, ids ascending, ``+0.0`` for a zero score.
* :func:`digest` hashes a call's outputs. Run as a script, this module prints the digest of each named
  :data:`REPEAT_CASES` call on the current device, for the fresh-process repeatability test.
"""

import hashlib
import json
import math
import sys

import torch

HEADS = 32
HEAD_DIM = 128
U = 2.0**-24
N_ROUNDINGS = 2 * HEAD_DIM + 4 * HEADS + 4  # the package's FP32 bound: n = 2D + 4H + 4
GAMMA = N_ROUNDINGS * U / (1 - N_ROUNDINGS * U)
CHUNK = 128  # query rows per reference block


def segment_lengths(total: int, segments: int, seed: int) -> list[int]:
    """``segments`` positive lengths summing to ``total`` (random cut points)."""
    gen = torch.Generator().manual_seed(seed)
    cuts = torch.randperm(total - 1, generator=gen)[: segments - 1].add(1).sort().values.tolist()
    bounds = [0, *cuts, total]
    return [b - a for a, b in zip(bounds, bounds[1:])]


def make_inputs(seg_q, seg_k, seed: int, *, device="cuda", integer: bool = False) -> dict:
    """``q [T, 32, 128]`` / ``k [T_kv, 128]`` BF16, ``w [T, 32]`` FP32 and the int32 segment boundaries.

    ``integer``: small integer queries, keys drawn from eight distinct integer rows and power-of-two weights, so every
    score is exact in FP32 and each row has at most eight distinct scores (ties at every cutoff).
    """
    gen = torch.Generator().manual_seed(seed)
    num_q, num_k = sum(seg_q), sum(seg_k)
    if integer:
        q = torch.randint(-2, 3, (num_q, HEADS, HEAD_DIM), generator=gen).to(torch.bfloat16)
        rows = torch.randint(-2, 3, (8, HEAD_DIM), generator=gen).to(torch.bfloat16)
        k = rows[torch.randint(0, 8, (num_k,), generator=gen)]
        levels = torch.tensor([-1.0, -0.5, 0.5, 1.0, 2.0])
        w = levels[torch.randint(0, len(levels), (num_q, HEADS), generator=gen)]
    else:
        q = torch.randn(num_q, HEADS, HEAD_DIM, generator=gen).to(torch.bfloat16)
        k = torch.randn(num_k, HEAD_DIM, generator=gen).to(torch.bfloat16)
        w = torch.randn(num_q, HEADS, generator=gen) * HEADS**-0.5
    cu_q = torch.tensor([0, *torch.tensor(seg_q, dtype=torch.int64).cumsum(0).tolist()], dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(seg_k, dtype=torch.int64).cumsum(0).tolist()], dtype=torch.int32)
    return {"q": q.to(device), "k": k.to(device), "w": w.to(device), "cu_q": cu_q.to(device), "cu_k": cu_k.to(device)}


def _segments(cu: torch.Tensor) -> list[tuple[int, int]]:
    bounds = cu.tolist()
    return list(zip(bounds, bounds[1:]))


def reference_blocks(x: dict, *, scale: float, ratio: int = 1, offsets=None):
    """Yield ``(row0, rows, ref, bound, visible)`` per block of query rows: the FP64 scores ``[rows, L_k]`` against the
    keys of the block's segment (``-inf`` where invisible), the bound ``gamma_n * A`` and each row's visible key
    count (``offsets``: the per-segment causal offsets, default ``L_k - L_q`` for ``ratio`` 1 and 0 otherwise)."""
    q, k, w = x["q"], x["k"], x["w"]
    offsets = None if offsets is None else offsets.tolist()
    for s, ((q0, q1), (k0, k1)) in enumerate(zip(_segments(x["cu_q"]), _segments(x["cu_k"]))):
        lq, lk = q1 - q0, k1 - k0
        if offsets is not None:
            offset = offsets[s]
        else:
            offset = lk - lq if ratio == 1 else 0
        keys = k[k0:k1].double()
        for r0 in range(q0, q1, CHUNK):
            r1 = min(r0 + CHUNK, q1)
            u = torch.arange(r0 - q0, r1 - q0, device=q.device, dtype=torch.int64)
            visible = torch.div(u + offset + 1, ratio, rounding_mode="floor").clamp(0, lk)
            qb = q[r0:r1].double()
            wb = w[r0:r1].double()
            ref = (wb[:, :, None] * torch.relu(scale * torch.einsum("thd,kd->thk", qb, keys))).sum(1)
            mass = scale * (wb.abs()[:, :, None] * torch.einsum("thd,kd->thk", qb.abs(), keys.abs())).sum(1)
            inside = torch.arange(lk, device=q.device)[None, :] < visible[:, None]
            yield r0, r1 - r0, ref.masked_fill(~inside, float("-inf")), GAMMA * mass, visible


def check_selection(x: dict, indices, scores, top_k: int, *, scale: float, ratio: int = 1, offsets=None) -> None:
    """Assert the structure, the score bound and the exact top-k (up to the bound) of every row."""
    assert indices.dtype == torch.int32 and scores.dtype == torch.float32
    assert tuple(indices.shape) == tuple(scores.shape) == (x["q"].shape[0], top_k)
    slots = torch.arange(top_k, device=indices.device)
    rows_seen = 0
    for r0, rows, ref, bound, visible in reference_blocks(x, scale=scale, ratio=ratio, offsets=offsets):
        idx, sc = indices[r0 : r0 + rows].long(), scores[r0 : r0 + rows]
        count = visible.clamp(max=top_k)
        valid = idx >= 0
        assert torch.equal(valid, slots[None, :] < count[:, None]), "the selected slots are not min(top_k, visible)"
        assert torch.all(sc[~valid] == float("-inf")) and torch.all(idx[~valid] == -1)
        assert torch.all(idx < visible[:, None]), "a selected key is not visible"
        both = valid[:, 1:] & valid[:, :-1]
        assert torch.all(idx[:, 1:][both] > idx[:, :-1][both]), "ids are not unique and ascending"
        rows_seen += rows
        width = ref.shape[1]
        if width == 0:
            continue
        safe = idx.clamp(min=0)
        excess = ((sc.double() - ref.gather(1, safe)).abs() - bound.gather(1, safe))[valid]
        assert torch.all(excess <= 0), f"score error {float(excess.max()):.3e} above the bound"
        # exact top-k up to rounding: no unselected visible key beats a selected one by more than the two bounds
        selected = torch.zeros(rows, width + 1, dtype=torch.bool, device=ref.device)
        selected = selected.scatter_(1, torch.where(valid, idx, width), True)[:, :width]
        lowest = torch.where(selected, ref + bound, torch.full_like(ref, math.inf)).amin(1)
        rest = ~selected & torch.isfinite(ref)
        highest = torch.where(rest, ref - bound, torch.full_like(ref, -math.inf)).amax(1)
        assert torch.all(lowest >= highest), "an unselected key beats a selected one beyond rounding"
    assert rows_seen == x["q"].shape[0]


def expected_selection(x: dict, top_k: int, *, scale: float = 1.0):
    """The exact ``(indices, scores, cutoff_ties)`` of integer-valued inputs: the top ``min(top_k, visible)`` keys by
    score descending, then key id descending, in ascending id order; ``cutoff_ties`` counts the rows whose last
    selected score is shared by an unselected visible key (rows the tie-break decides)."""
    num_q = x["q"].shape[0]
    indices = torch.full((num_q, top_k), -1, dtype=torch.int64, device=x["q"].device)
    scores = torch.full((num_q, top_k), float("-inf"), dtype=torch.float32, device=x["q"].device)
    cutoff_ties = 0
    for r0, rows, ref, _, visible in reference_blocks(x, scale=scale):
        exact = ref.float() + 0.0  # exact FP32 values; -0.0 becomes +0.0
        lk = exact.shape[1]
        width = min(top_k, lk)
        if width == 0:
            continue
        by_id = torch.arange(lk - 1, -1, -1, device=exact.device).expand(rows, -1)
        # a stable sort by score of the ids in descending order: equal scores keep the larger id first
        ranked = by_id.gather(1, exact.gather(1, by_id).sort(dim=1, descending=True, stable=True).indices)
        count = visible.clamp(max=top_k)
        keep = torch.arange(width, device=exact.device)[None, :] < count[:, None]
        want = torch.where(keep, ranked[:, :width], lk).sort(dim=1).values
        want = torch.where(want < lk, want, -1)
        indices[r0 : r0 + rows, :width] = want
        scores[r0 : r0 + rows, :width] = torch.where(
            want >= 0, exact.gather(1, want.clamp(min=0)), torch.full_like(want, float("-inf"), dtype=torch.float32)
        )
        kth = exact.gather(1, ranked[:, width - 1 : width])
        tied = (exact == kth).sum(1) > (exact.gather(1, ranked[:, :width]) == kth).sum(1)
        cutoff_ties += int((tied & (visible > top_k)).sum())
    return indices, scores, cutoff_ties


def digest(indices: torch.Tensor, scores: torch.Tensor) -> str:
    """SHA-256 of the ids' and the scores' bytes."""
    sha = hashlib.sha256()
    for tensor in (indices, scores):
        sha.update(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


# name -> (query segment lengths, key segment lengths, top_k, ratio, seed)
REPEAT_CASES = {
    "recorded_16231": (segment_lengths(16231, 8, 1), segment_lengths(16231, 8, 1), 2048, 1, 11),
    "recorded_16172_k4096": (segment_lengths(16172, 8, 2), segment_lengths(16172, 8, 2), 4096, 1, 12),
    "prefix_keys": ([200, 300], [200, 2500], 128, 1, 13),
    "compressed_ratio4": ([2048, 1024], [512, 256], 512, 4, 14),
}


def run_case(name: str, device="cuda") -> str:
    """The digest of one :data:`REPEAT_CASES` call on ``device``."""
    from miles_ext_kernels.cake.dsa import dsa_indexer_det_topk

    seg_q, seg_k, top_k, ratio, seed = REPEAT_CASES[name]
    x = make_inputs(seg_q, seg_k, seed, device=device)
    indices, scores = dsa_indexer_det_topk(x["q"], x["k"], x["w"], x["cu_q"], x["cu_k"], top_k, ratio=ratio)
    return digest(indices, scores)


if __name__ == "__main__":
    print(json.dumps({name: run_case(name) for name in sys.argv[1:]}))
