"""The prebuild guarantee of the DSA ops (README, Prebuild): after one ``prebuild("dsa", **model_params)``, no call with
those model parameters compiles, whichever variant runs.

``ROUTES`` holds one forward + backward per call-time path: attention head counts from 4 to 128 (24 included; the
tcgen05 and the register-MMA stages), the RoPE and NoPE geometries, with and without an attention sink, ``thd`` and
``bshd``, no query rows (T = 0), the indexer logits and backward at each indexer head count (both backward stages of 32
heads), ``thd`` and ``sbhd``, and the det indexer top-k. ``GROUPS`` names the model parameters of each route; each
group is checked with one prebuild.
"""

from __future__ import annotations

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake import dsa

from ..helpers import check_prebuild, requires_arch, requires_cuda

SCALE = 192**-0.5
NUM_KV = 256


def _generator(seed: int):
    import torch

    return torch.Generator(device="cuda").manual_seed(seed)


def _attention(layout: str, heads: int, d_qk: int, sink: bool, rows: int, topk: int = 128):
    """A sparse-attention call: ``rows`` query rows (per batch element for ``bshd``, batch 2), forward + backward."""

    def run(variant: str) -> None:
        import torch

        gen = _generator(heads * 7 + rows + d_qk)
        if layout == "thd":
            q = torch.randn(rows, heads, d_qk, device="cuda", dtype=torch.bfloat16, generator=gen)
            kv = torch.randn(NUM_KV, d_qk, device="cuda", dtype=torch.bfloat16, generator=gen)
            indices = torch.randint(-1, NUM_KV, (rows, topk), device="cuda", generator=gen, dtype=torch.int32)
        else:
            q = torch.randn(2, rows, heads, d_qk, device="cuda", dtype=torch.bfloat16, generator=gen)
            kv = torch.randn(2, NUM_KV // 2, d_qk, device="cuda", dtype=torch.bfloat16, generator=gen)
            indices = torch.randint(-1, NUM_KV // 2, (2, rows, topk), device="cuda", generator=gen, dtype=torch.int32)
        attn_sink = torch.randn(heads, device="cuda", generator=gen).requires_grad_(True) if sink else None
        q.requires_grad_(True)
        kv.requires_grad_(True)
        out = dsa.sparse_attention(q, kv, indices, sm_scale=SCALE, attn_sink=attn_sink, layout=layout)
        out.backward(torch.randn(out.shape, device="cuda", generator=gen).to(out.dtype))
        torch.cuda.synchronize()

    return run


def _indexer(layout: str, heads: int, rows: int, topk: int):
    """Indexer logits, top-k selection and the selected scores' backward (``sbhd``: batch 2)."""

    def run(variant: str) -> None:
        import torch

        gen = _generator(heads * 11 + rows + topk)
        positions = torch.arange(rows, device="cuda", dtype=torch.int32)
        ks, ke = torch.zeros_like(positions), positions + 1
        if layout == "thd":
            q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
            k = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
            w = torch.randn(rows, heads, device="cuda", generator=gen)
        else:
            q = torch.randn(rows, 2, heads, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
            k = torch.randn(rows, 2, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
            w = torch.randn(rows, 2, heads, device="cuda", generator=gen)
        for tensor in (q, k, w):
            tensor.requires_grad_(True)
        scores, _ = dsa.lightning_indexer(q, k, w, ks, ke, topk, layout=layout)
        scores.masked_fill(torch.isinf(scores), 0.0).sum().backward()
        torch.cuda.synchronize()

    return run


def _det_topk(rows: int, topk: int):
    """The det indexer top-k over two packed segments, feeding the selected scores' backward."""

    def run(variant: str) -> None:
        import torch

        gen = _generator(rows + topk)
        q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
        k = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
        w = torch.randn(rows, 32, device="cuda", generator=gen)
        cu = torch.tensor([0, rows // 2, rows], device="cuda", dtype=torch.int32)
        indices, _ = dsa.dsa_indexer_det_topk(q, k, w, cu, cu, topk, softmax_scale=1.0)
        packed = dsa.packed_key_ids(indices, cu, cu)
        segment = (torch.arange(rows, device="cuda") >= rows // 2).int()
        ks = (segment * (rows // 2)).int()
        ke = torch.arange(1, rows + 1, device="cuda", dtype=torch.int32)
        logits = dsa.indexer_logits(q, k, w, ks, ke)
        for tensor in (q, k, w):
            tensor.requires_grad_(True)
        selected = dsa.indexer_topk_scores(q, k, w, packed, logits)
        selected.masked_fill(torch.isinf(selected), 0.0).sum().backward()
        torch.cuda.synchronize()

    return run


ROUTES = {
    # 64 heads: the fast variant's RoPE and NoPE programs, and det
    "thd-h64-rope": _attention("thd", 64, 576, False, 256),
    "thd-h64-rope-T0": _attention("thd", 64, 576, False, 0),
    "bshd-h64-rope": _attention("bshd", 64, 576, False, 128),
    "thd-h64-nope": _attention("thd", 64, 512, False, 256),
    "thd-h64-nope-T0": _attention("thd", 64, 512, False, 0),
    "bshd-h64-nope": _attention("bshd", 64, 512, False, 128),
    # with an attention sink (det under either variant)
    "bshd-h64-nope-sink": _attention("bshd", 64, 512, True, 128),
    "bshd-h64-nope-sink-T0": _attention("bshd", 64, 512, True, 0),
    "bshd-h16-nope-sink": _attention("bshd", 16, 512, True, 128),
    "bshd-h16-nope-sink-T0": _attention("bshd", 16, 512, True, 0),
    # every other head count (det under either variant)
    **{f"thd-h{heads}-rope": _attention("thd", heads, 576, False, 256) for heads in (4, 8, 12, 16, 24, 32, 48, 128)},
    **{f"thd-h{heads}-rope-T0": _attention("thd", heads, 576, False, 0) for heads in (4, 8, 12, 16, 24, 32, 48, 128)},
    # the indexer
    "indexer-thd-h8": _indexer("thd", 8, 256, 128),
    "indexer-thd-h16": _indexer("thd", 16, 256, 128),
    "indexer-thd-h32-topk2048": _indexer("thd", 32, 2048, 2048),
    "indexer-thd-h32-topk512": _indexer("thd", 32, 1024, 512),
    "indexer-thd-h32-T0": _indexer("thd", 32, 0, 128),
    "indexer-sbhd-h64": _indexer("sbhd", 64, 256, 128),
    "det-topk-h32": _det_topk(4096, 2048),
}

# group -> (model parameters, routes)
GROUPS = {
    "h64-rope-index32": (
        dict(heads=64, d_qk=576, index_heads=32),
        [
            "thd-h64-rope",
            "thd-h64-rope-T0",
            "bshd-h64-rope",
            "indexer-thd-h32-topk2048",
            "indexer-thd-h32-topk512",
            "indexer-thd-h32-T0",
            "det-topk-h32",
        ],
    ),
    "h64-nope": (dict(heads=64, d_qk=512), ["thd-h64-nope", "thd-h64-nope-T0", "bshd-h64-nope"]),
    "h64-nope-sink-index64": (
        dict(heads=64, d_qk=512, attn_sink=True, index_heads=64),
        ["bshd-h64-nope-sink", "bshd-h64-nope-sink-T0", "indexer-sbhd-h64"],
    ),
    "h16-nope-sink-index16": (
        dict(heads=16, d_qk=512, attn_sink=True, index_heads=16),
        ["bshd-h16-nope-sink", "bshd-h16-nope-sink-T0", "indexer-thd-h16"],
    ),
    "h8-rope-index8": (dict(heads=8, d_qk=576, index_heads=8), ["thd-h8-rope", "thd-h8-rope-T0", "indexer-thd-h8"]),
    **{
        f"h{heads}-rope": (dict(heads=heads, d_qk=576), [f"thd-h{heads}-rope", f"thd-h{heads}-rope-T0"])
        for heads in (4, 12, 16, 24, 32, 48, 128)
    },
}


def test_every_route_is_checked_in_a_group():
    grouped = [name for _, names in GROUPS.values() for name in names]

    assert sorted(grouped) == sorted(ROUTES)


@requires_cuda
@requires_arch("sm_100a", "sm_103a", "sm_107a")
@pytest.mark.parametrize("group", sorted(GROUPS))
def test_nothing_compiles_after_prebuild(group):
    params, names = GROUPS[group]

    check_prebuild("dsa", {name: ROUTES[name] for name in names}, **params)


@requires_cuda
@requires_arch("sm_100a", "sm_103a", "sm_107a")
def test_prebuild_returns_the_libraries_and_lazy_builds_return_none():
    paths = cake.prebuild("dsa", heads=64, d_qk=576, index_heads=32)

    assert paths and all(path.is_file() for path in paths)
    assert cake.prebuild("dsa", heads=64, d_qk=576, index_heads=32) == paths
    cake.configure(cake_dsa_lazy_build=True)
    assert cake.prebuild("dsa", heads=64, d_qk=576, index_heads=32) == []
