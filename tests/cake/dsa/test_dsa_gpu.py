"""The DSA ops on SM100a / SM103a / SM107a at the TP-local shapes miles runs.

* det sparse attention -- the packed ``thd`` MLA (``d_qk = 512 + 64``, no sink) at 4 / 8 / 12 / 16 / 24 / 32 / 48 /
  64 / 128 heads (tcgen05 and register-MMA stages), the batched ``bshd`` MQA (``d = 512``, FP32 sink) at 16 / 64 heads,
  and window + compressed rows (128 + 512 / 128 + 1024 slots with ``-1`` holes): output and gradients against an FP32
  reference, bit-identical across passes, in a fresh process, on a side stream and on a second device.
* det lightning indexer -- logits and selected scores against an FP32 reference, bit-identical across passes; selected
  keys outside their window get no gradient.
* fast sparse attention -- the 64-head RoPE and NoPE geometries against the same reference and against det: ``out``
  and ``dq`` bit-identical across passes, ``dkv`` within the spread of its FP32 atomic accumulation, and independent of
  the segment count; the calls fast does not serve, and every call under torch's deterministic algorithms, run det.
* the reviewed limits -- batch-local indices outside their batch element, packed indices past the keys, the det
  backward's memory budget, calls without query rows.

The two-device tests are marked ``multi_gpu``.
"""

import json
import logging
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (10, 7)}:
    pytest.skip("the DSA kernels target SM100a / SM103a / SM107a", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import dsa  # noqa: E402
from miles_ext_kernels.cake._impl.dsa import ops, reduction  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader  # noqa: E402

from ..helpers import assert_bitwise  # noqa: E402
from . import _det_repeat  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEVICE = "cuda"
TOL = 1e-2  # max |error| / max |reference|: BF16 operands and outputs, FP32 accumulation
MLA_SCALE = _det_repeat.MLA_SCALE  # the absorbed MLA's softmax scale, not 576 ** -0.5
NOPE_SCALE = 1.0 / 16.0
MQA_SCALE = 512**-0.5
MIB = 1 << 20


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(config, "_warned", set())


def _rel_err(actual, expected):
    expected = expected.float()
    actual = actual.float()
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite), "finite/-inf pattern differs"
    scale = expected[finite].abs().max().clamp_min(1e-6)
    return ((actual[finite] - expected[finite]).abs().max() / scale).item()


def _generator(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _causal_indices(rows, topk, num_keys, gen):
    """Row ``r`` selects ``min(topk, limit_r)`` distinct keys below ``limit_r = (r + 1) * num_keys // rows`` in random
    slot order; ``-1`` pads the rest."""
    limits = ((torch.arange(rows, device=DEVICE) + 1) * num_keys // rows).clamp(min=1, max=num_keys)
    keys = torch.arange(num_keys, device=DEVICE)
    score = torch.rand(rows, num_keys, device=DEVICE, generator=gen).masked_fill(keys >= limits[:, None], -1.0)
    count = min(topk, num_keys)
    picked = torch.topk(score, count, dim=-1)
    idx = torch.where(picked.values >= 0, picked.indices, -1).int()
    if count < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - count), value=-1)
    return idx


def _window_compressed_indices(seq, window, compressed_topk, ratio, gen):
    """Window + compressed rows: ``window`` sliding-window keys (``-1`` before the sequence start) then ``compressed_topk``
    picks among the ``(r + 1) // ratio`` visible compressed keys (stored after the ``seq`` window keys), with ``-1``
    holes scattered over the compressed slots."""
    rows = torch.arange(seq, device=DEVICE)
    window_idx = rows[:, None] - (window - 1) + torch.arange(window, device=DEVICE)[None, :]
    window_idx = torch.where(window_idx >= 0, window_idx, -1)
    n_comp = seq // ratio
    visible = (rows + 1) // ratio
    score = torch.rand(seq, n_comp, device=DEVICE, generator=gen)
    score = score.masked_fill(torch.arange(n_comp, device=DEVICE)[None, :] >= visible[:, None], -1.0)
    picked = torch.topk(score, min(compressed_topk, n_comp), dim=-1)
    comp_idx = torch.where(picked.values >= 0, picked.indices + seq, -1)
    if comp_idx.shape[1] < compressed_topk:
        comp_idx = torch.nn.functional.pad(comp_idx, (0, compressed_topk - comp_idx.shape[1]), value=-1)
    comp_idx = comp_idx[:, torch.randperm(compressed_topk, device=DEVICE, generator=gen)]
    return torch.cat([window_idx, comp_idx], dim=-1).int(), seq + n_comp


# --------------------------------------------------------------------------- #
# FP32 reference on the flat contract (-inf sink disables, empty rows -> 0)
# --------------------------------------------------------------------------- #


def ref_sparse_attention(q, kv, indices, do, sm_scale, sink=None, d_v=512, rows_per_chunk=32):
    """FP32 forward and backward of sparse attention, ``q [T, H, D]``, ``kv [S, D]``, global ``indices [T, K]``.

    Returns ``(out, dq, dkv, dsink)``; computed in row chunks with the closed-form gradients
    (``dS = P (dP - rowsum(dO O))``), so long rows fit in memory.
    """
    q32, kv32, do32 = q.float(), kv.float(), do.float()
    num_rows, heads, dim = q.shape
    num_kv = kv.shape[0]
    out = torch.empty(num_rows, heads, d_v, device=q.device)
    dq = torch.empty(num_rows, heads, dim, device=q.device)
    dkv = torch.zeros(num_kv, dim, device=q.device)
    dsink = None if sink is None else torch.zeros(heads, device=q.device)
    sink32 = None if sink is None else sink.float().view(1, heads, 1)
    for start in range(0, num_rows, rows_per_chunk):
        rows = slice(start, min(num_rows, start + rows_per_chunk))
        idx = indices[rows].long()
        valid = (idx >= 0) & (idx < num_kv)
        safe = idx.clamp(0, num_kv - 1)
        keys = kv32[safe]  # [r, K, D]
        scores = torch.einsum("rhd,rkd->rhk", q32[rows], keys) * sm_scale
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        row_max = scores.amax(dim=-1, keepdim=True)
        if sink32 is not None:
            row_max = torch.maximum(row_max, sink32)
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        expo = torch.exp(scores - row_max)
        denom = expo.sum(dim=-1, keepdim=True)
        if sink32 is not None:
            denom = denom + torch.exp(sink32 - row_max)
        denom = denom.clamp_min(torch.finfo(torch.float32).tiny)
        probs = expo / denom
        o = torch.einsum("rhk,rkd->rhd", probs, keys[..., :d_v])
        out[rows] = o
        dp = torch.einsum("rhd,rkd->rhk", do32[rows], keys[..., :d_v])
        delta = (o * do32[rows]).sum(dim=-1, keepdim=True)
        ds = probs * (dp - delta)
        dq[rows] = torch.einsum("rhk,rkd->rhd", ds, keys) * sm_scale
        dkeys = torch.einsum("rhk,rhd->rkd", ds, q32[rows]) * sm_scale
        dkeys[..., :d_v] += torch.einsum("rhk,rhd->rkd", probs, do32[rows])
        dkeys = dkeys * valid[..., None]
        dkv.index_add_(0, safe.reshape(-1), dkeys.reshape(-1, dim))
        if dsink is not None:
            p_sink = torch.exp(sink32 - row_max) / denom
            dsink -= (p_sink * delta).sum(dim=(0, 2))
    return out, dq, dkv, dsink


def ref_indexer_logits(index_q, index_k, weights, ks, ke):
    sims = torch.relu(torch.einsum("thd,kd->thk", index_q.float(), index_k.float()))
    logits = torch.einsum("thk,th->tk", sims, weights.float())
    keys = torch.arange(index_k.shape[0], device=index_k.device)
    inside = (keys[None, :] >= ks[:, None]) & (keys[None, :] < ke[:, None])
    return logits.masked_fill(~inside, float("-inf"))


def ref_indexer_scores(index_q, index_k, weights, topk_indices):
    valid = topk_indices >= 0
    gathered = index_k.float()[topk_indices.clamp(min=0).long()]
    sims = torch.relu(torch.einsum("thd,tkd->thk", index_q.float(), gathered))
    return torch.einsum("thk,th->tk", sims, weights.float()).masked_fill(~valid, float("-inf"))


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def _run(q, kv, indices, do, *, sink=None, variant="det", **kwargs):
    """One autograd pass of ``variant``: ``(out, dq, dkv, dsink)``."""
    cake.configure(cake_dsa_variant=variant)
    q_ = q.clone().requires_grad_(True)
    kv_ = kv.clone().requires_grad_(True)
    s_ = None if sink is None else sink.clone().requires_grad_(True)
    out = dsa.sparse_attention(q_, kv_, indices, attn_sink=s_, **kwargs)
    out.backward(do)
    return out.detach(), q_.grad, kv_.grad, None if s_ is None else s_.grad


def _named(result, names=("out", "dq", "dkv", "dsink")):
    return {name: tensor for name, tensor in zip(names, result) if tensor is not None}


def _assert_matches(got, ref, tol=TOL, names=("out", "dq", "dkv", "dsink")):
    for name, x, y in zip(names, got, ref):
        if y is not None:
            err = _rel_err(x, y)
            assert err < tol, f"{name}: rel err {err:.3e}"


def _thd_inputs(rows, heads, topk, seed, d_qk=576, d_v=512):
    gen = _generator(seed)
    q = torch.randn(rows, heads, d_qk, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(rows, 1, d_qk, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    indices = _causal_indices(rows, topk, rows, gen).unsqueeze(1)  # [T, 1, topk] as the model passes it
    do = torch.randn(rows, heads, d_v, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    return q, kv, indices, do


def _flat_reference(q, kv, indices, do, sm_scale, sink=None):
    kv_flat = kv.reshape(-1, kv.shape[-1])
    return ref_sparse_attention(q, kv_flat, indices.reshape(q.shape[0], -1), do, sm_scale, sink)


def _bshd_inputs(batch, seq, heads, topk, seed, d_qk=512):
    gen = _generator(seed)
    q = torch.randn(batch, seq, heads, d_qk, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(batch, seq, d_qk, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    indices = torch.stack([_causal_indices(seq, topk, seq, gen) for _ in range(batch)])
    sink = torch.randn(heads, device=DEVICE, generator=gen)
    do = torch.randn(batch, seq, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    return q, kv, indices, sink, do


def _globalized(indices, seq_kv):
    batch = indices.shape[0]
    offsets = (torch.arange(batch, device=DEVICE) * seq_kv).view(batch, 1, 1)
    return torch.where(indices >= 0, indices + offsets, indices).reshape(-1, indices.shape[-1])


# --------------------------------------------------------------------------- #
# det sparse attention
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("heads", "rows", "topk", "d_qk"),
    [
        (4, 512, 256, 576),
        (8, 2048, 2048, 576),
        (12, 512, 256, 576),
        (16, 512, 256, 576),
        (24, 512, 256, 576),
        (24, 512, 256, 512),
        (32, 2048, 2048, 576),
        (48, 512, 256, 576),
        (64, 512, 256, 576),
        (64, 512, 256, 512),
        (128, 512, 256, 576),
    ],
    ids=[
        "h4",
        "h8-topk2048",
        "h12",
        "h16",
        "h24-rope",
        "h24-nope",
        "h32-topk2048",
        "h48",
        "h64-rope",
        "h64-nope",
        "h128",
    ],
)
def test_det_thd_attention_matches_the_reference_and_is_bit_identical(heads, rows, topk, d_qk):
    q, kv, indices, do = _thd_inputs(rows, heads, topk, seed=1, d_qk=d_qk)
    sm_scale = MLA_SCALE if d_qk == 576 else NOPE_SCALE

    first = _run(q, kv, indices, do, sm_scale=sm_scale, layout="thd")
    assert_bitwise(_named(first), _named(_run(q, kv, indices, do, sm_scale=sm_scale, layout="thd")))

    out, dq, dkv, _ = first
    _assert_matches((out, dq, dkv.squeeze(1)), _flat_reference(q, kv, indices, do, sm_scale)[:3])


@pytest.mark.parametrize("heads", [16, 64], ids=["h16", "h64"])
def test_det_bshd_attention_with_sink_matches_the_reference_and_is_bit_identical(heads):
    batch, seq, topk = 2, 256, 192  # padded to 256 slots inside
    q, kv, indices, sink, do = _bshd_inputs(batch, seq, heads, topk, seed=4)

    first = _run(q, kv, indices, do, sink=sink, sm_scale=MQA_SCALE, layout="bshd")
    assert_bitwise(_named(first), _named(_run(q, kv, indices, do, sink=sink, sm_scale=MQA_SCALE, layout="bshd")))

    out, dq, dkv, dsink = first
    ref = ref_sparse_attention(
        q.reshape(batch * seq, heads, 512),
        kv.reshape(batch * seq, 512),
        _globalized(indices, seq),
        do.reshape(batch * seq, heads, 512),
        MQA_SCALE,
        sink,
    )
    _assert_matches(
        (out.reshape(batch * seq, heads, 512), dq.reshape(batch * seq, heads, 512), dkv.reshape(-1, 512), dsink), ref
    )


@pytest.mark.parametrize(
    ("heads", "batch", "seq", "compressed_topk"),
    [(8, 2, 2048, 512), (16, 1, 2048, 512), (64, 1, 4096, 1024)],
    ids=["h8-128+512", "h16-128+512", "h64-128+1024"],
)
def test_det_window_and_compressed_rows_with_holes(heads, batch, seq, compressed_topk):
    gen = _generator(7)
    per_batch = [_window_compressed_indices(seq, 128, compressed_topk, 4, gen) for _ in range(batch)]
    indices = torch.stack([idx for idx, _ in per_batch])
    seq_kv = per_batch[0][1]
    assert (indices[:, :, 128:] == -1).any() and indices.shape[-1] == 128 + compressed_topk
    q = torch.randn(batch, seq, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(batch, seq_kv, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    sink = torch.randn(heads, device=DEVICE, generator=gen)
    do = torch.randn(batch, seq, heads, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)

    first = _run(q, kv, indices, do, sink=sink, sm_scale=MQA_SCALE, layout="bshd")
    assert_bitwise(_named(first), _named(_run(q, kv, indices, do, sink=sink, sm_scale=MQA_SCALE, layout="bshd")))

    out, dq, dkv, dsink = first
    rows = batch * seq
    ref = ref_sparse_attention(
        q.reshape(rows, heads, 512),
        kv.reshape(-1, 512),
        _globalized(indices, seq_kv),
        do.reshape(rows, heads, 512),
        MQA_SCALE,
        sink,
    )
    _assert_matches((out.reshape(rows, heads, 512), dq.reshape(rows, heads, 512), dkv.reshape(-1, 512), dsink), ref)


def test_det_bshd_output_allows_the_in_place_inverse_rope_before_backward():
    """A model may apply the inverse RoPE in place on the attention output before the output projection."""
    q, kv, indices, sink, do = _bshd_inputs(2, 128, 16, 64, seed=5)
    q_ = q.clone().requires_grad_(True)
    out = dsa.sparse_attention(q_, kv, indices, sm_scale=MQA_SCALE, attn_sink=sink, layout="bshd")
    assert out.shape == q.shape and out._base is None, "the output must be a fresh layout-shaped tensor, not a view"
    out[..., -64:].mul_(-1.0)  # in place on a slice, as apply_rotary_emb(o[..., -rd:], ..., inverse=True) does
    out.backward(do)
    q_ref = q.clone().requires_grad_(True)
    out_ref = dsa.sparse_attention(q_ref, kv, indices, sm_scale=MQA_SCALE, attn_sink=sink, layout="bshd")
    scale = torch.ones_like(out_ref)
    scale[..., -64:] = -1.0
    (out_ref * scale).backward(do)
    assert torch.equal(q_.grad, q_ref.grad)


# --------------------------------------------------------------------------- #
# det lightning indexer
# --------------------------------------------------------------------------- #


def _indexer_inputs(rows, heads, seed):
    gen = _generator(seed)
    index_q = torch.randn(rows, heads, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    index_k = torch.randn(rows, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    weights = torch.randn(rows, heads, device=DEVICE, generator=gen)
    ks = torch.zeros(rows, device=DEVICE, dtype=torch.int32)
    ke = (torch.arange(rows, device=DEVICE) + 1).int()
    return index_q, index_k, weights, ks, ke


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
def test_det_lightning_indexer_matches_the_reference_and_is_bit_identical(heads):
    rows, topk = 512, 128
    index_q, index_k, weights, ks, ke = _indexer_inputs(rows, heads, seed=2)

    logits = dsa.indexer_logits(index_q, index_k, weights, ks, ke, layout="thd")
    assert torch.equal(logits, dsa.indexer_logits(index_q, index_k, weights, ks, ke, layout="thd"))
    assert _rel_err(logits, ref_indexer_logits(index_q, index_k, weights, ks, ke)) < 1e-3

    def run():
        q_ = index_q.clone().requires_grad_(True)
        k_ = index_k.clone().requires_grad_(True)
        w_ = weights.clone().requires_grad_(True)
        scores, topk_indices = dsa.lightning_indexer(q_, k_, w_, ks, ke, topk, layout="thd")
        grad = torch.randn(scores.shape, device=DEVICE, generator=_generator(3)).masked_fill(
            ~torch.isfinite(scores), 0.0
        )
        scores.masked_fill(~torch.isfinite(scores), 0.0).backward(grad)
        return scores.detach(), topk_indices, q_.grad, k_.grad, w_.grad, grad

    scores, topk_indices, dq, dk, dw, grad = run()
    again = run()
    names = ("scores", "indices", "dq", "dk", "dw")
    assert_bitwise(dict(zip(names, (scores, topk_indices, dq, dk, dw))), dict(zip(names, again)))

    q32 = index_q.clone().float().requires_grad_(True)
    k32 = index_k.clone().float().requires_grad_(True)
    w32 = weights.clone().requires_grad_(True)
    ref = ref_indexer_scores(q32, k32, w32, topk_indices)
    assert _rel_err(scores, ref) < 1e-3
    ref.masked_fill(~torch.isfinite(ref), 0.0).backward(grad)
    assert _rel_err(dq, q32.grad) < 2e-2  # ReLU mask flips at |q.k| ~ bf16 eps
    assert _rel_err(dk, k32.grad) < 1e-2
    assert _rel_err(dw, w32.grad) < 1e-3


@pytest.mark.parametrize("seq_len_kv", [512, 509], ids=["aligned", "unaligned"])
def test_det_indexer_logits_on_short_key_rows(seq_len_kv):
    index_q, index_k, weights, ks, ke = _indexer_inputs(512, 32, seed=6)
    index_k = index_k[:seq_len_kv].contiguous()
    ke = ke.clamp(max=seq_len_kv)

    logits = dsa.indexer_logits(index_q, index_k, weights, ks, ke, layout="thd")
    assert logits.shape == (512, seq_len_kv)
    assert _rel_err(logits, ref_indexer_logits(index_q, index_k, weights, ks, ke)) < 1e-3


def test_det_sbhd_batched_indexer_logits_match_the_reference():
    gen = _generator(5)
    seq, batch, heads, ratio = 256, 2, 64, 4
    seq_kv = seq // ratio
    q = torch.randn(seq, batch, heads, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(seq_kv, batch, 128, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    weights = torch.randn(seq, batch, heads, device=DEVICE, generator=gen)
    positions = torch.arange(seq, device=DEVICE, dtype=torch.int32)
    ks = torch.zeros(seq, device=DEVICE, dtype=torch.int32)
    ke = ((positions + 1) // ratio).to(torch.int32)

    logits = dsa.indexer_logits(q, k, weights, ks, ke, layout="sbhd")
    assert logits.shape == (batch, seq, seq_kv)
    assert torch.equal(logits, dsa.indexer_logits(q, k, weights, ks, ke, layout="sbhd"))
    for b in range(batch):
        assert _rel_err(logits[b], ref_indexer_logits(q[:, b], k[:, b], weights[:, b], ks, ke)) < 1e-3


def test_selected_keys_outside_their_window_get_no_gradient_even_a_nan_one():
    """Two packed segments: replayed indices pick keys of the other segment, whose scores are -inf; a loss that gives
    NaN there leaves the gradients finite and equal to those of the same selection with the slots empty."""
    rows, heads, topk = 256, 32, 64
    index_q, index_k, weights, _, _ = _indexer_inputs(rows, heads, seed=8)
    half = rows // 2
    positions = torch.arange(rows, device=DEVICE, dtype=torch.int32)
    ks = torch.where(positions < half, 0, half).int()
    ke = positions + 1
    gen = _generator(9)
    own = torch.stack([torch.randint(int(s), int(e), (topk,), device=DEVICE, generator=gen) for s, e in zip(ks, ke)])
    other = torch.where(positions[:, None] < half, half + own % half, own % half)  # the other segment's keys
    leaky = torch.where(torch.rand(rows, topk, device=DEVICE, generator=gen) < 0.25, other, own).int()
    outside = (leaky < ks[:, None]) | (leaky >= ke[:, None])
    assert outside.any()

    def run(topk_indices, nan):
        q_ = index_q.clone().requires_grad_(True)
        k_ = index_k.clone().requires_grad_(True)
        w_ = weights.clone().requires_grad_(True)
        scores, _ = dsa.lightning_indexer(q_, k_, w_, ks, ke, topk, topk_indices=topk_indices)
        grad = torch.ones_like(scores).masked_fill(torch.isinf(scores), float("nan") if nan else 0.0)
        scores.backward(grad)
        return {"scores": scores.detach(), "dq": q_.grad, "dk": k_.grad, "dw": w_.grad}

    got = run(leaky, nan=True)
    assert torch.equal(torch.isinf(got["scores"]), outside)
    assert all(torch.isfinite(got[name].float()).all() for name in ("dq", "dk", "dw"))
    assert_bitwise(got, run(leaky.masked_fill(outside, -1), nan=False))


def test_det_indexer_backward_treats_indices_past_the_keys_as_empty_slots():
    rows, heads, topk = 256, 32, 64
    index_q, index_k, weights, _, _ = _indexer_inputs(rows, heads, seed=10)
    gen = _generator(11)
    indices = torch.randint(0, rows, (rows, topk), device=DEVICE, generator=gen, dtype=torch.int32)
    past = torch.rand(rows, topk, device=DEVICE, generator=gen) < 0.2
    grad = torch.randn(rows, topk, device=DEVICE, generator=gen)

    got = dsa.indexer_backward(index_q, index_k, weights, indices.masked_fill(past, rows + 3), grad)
    want = dsa.indexer_backward(index_q, index_k, weights, indices.masked_fill(past, -1), grad)

    assert_bitwise(dict(zip(("dq", "dw", "dk"), got)), dict(zip(("dq", "dw", "dk"), want)))


# --------------------------------------------------------------------------- #
# fast sparse attention and the calls it does not serve
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("d_qk", "sm_scale", "rows", "topk"),
    [(576, MLA_SCALE, 2048, 2048), (512, NOPE_SCALE, 2048, 2048), (576, MLA_SCALE, 512, 200)],
    ids=["rope-topk2048", "nope-topk2048", "rope-short-topk"],
)
def test_fast_64_heads_matches_the_reference_and_det(d_qk, sm_scale, rows, topk):
    q, kv, indices, do = _thd_inputs(rows, 64, topk, seed=11, d_qk=d_qk)

    first = _run(q, kv, indices, do, sm_scale=sm_scale, layout="thd", variant="fast")
    second = _run(q, kv, indices, do, sm_scale=sm_scale, layout="thd", variant="fast")
    # out and dq are written once per row; dkv sums FP32 atomics in a run-dependent order
    assert_bitwise(_named(first[:2]), _named(second[:2]))
    assert _rel_err(second[2], first[2]) < TOL

    out, dq, dkv, _ = first
    assert dkv.shape == kv.shape and dkv.dtype == torch.bfloat16
    _assert_matches((out, dq, dkv.squeeze(1)), _flat_reference(q, kv, indices, do, sm_scale)[:3])
    det = _run(q, kv, indices, do, sm_scale=sm_scale, layout="thd", variant="det")
    _assert_matches((out, dq, dkv), det[:3], tol=2 * TOL)


def test_fast_runs_batched_rows():
    batch, seq, topk = 2, 512, 256
    q, kv, indices, _, do = _bshd_inputs(batch, seq, 64, topk, seed=12, d_qk=576)

    out, dq, dkv, _ = _run(q, kv, indices, do, sm_scale=MLA_SCALE, layout="bshd", variant="fast")
    assert out.shape == (batch, seq, 64, 512) and dq.shape == q.shape and dkv.shape == kv.shape
    rows = batch * seq
    ref = ref_sparse_attention(
        q.reshape(rows, 64, 576), kv.reshape(-1, 576), _globalized(indices, seq), do.reshape(rows, 64, 512), MLA_SCALE
    )
    _assert_matches((out.reshape(rows, 64, 512), dq.reshape(rows, 64, 576), dkv.reshape(-1, 576)), ref[:3])


def test_fast_results_do_not_depend_on_the_segment_count():
    """Eight packed segments: num_segments=8 lets the backward plan its key passes per segment, with the same bits."""
    segments, seg_q, seg_kv, topk = 8, 128, 2048, 2048
    gen = _generator(16)
    rows, num_kv = segments * seg_q, segments * seg_kv
    q = torch.randn(rows, 64, 576, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    kv = torch.randn(num_kv, 576, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    start = (torch.arange(rows, device=DEVICE) // seg_q * seg_kv)[:, None]
    order = torch.argsort(torch.rand(rows, seg_kv, device=DEVICE, generator=gen), dim=-1)
    indices = (start + order[:, :topk]).int()
    do = torch.randn(rows, 64, 512, device=DEVICE, dtype=torch.bfloat16, generator=gen)

    one = _run(q, kv, indices, do, sm_scale=MLA_SCALE, layout="thd", variant="fast", num_segments=1)
    eight = _run(q, kv, indices, do, sm_scale=MLA_SCALE, layout="thd", variant="fast", num_segments=segments)

    assert_bitwise(_named(one[:2]), _named(eight[:2]))
    assert _rel_err(eight[2], one[2]) < TOL


def test_calls_fast_does_not_serve_run_det_bit_identically_and_warn_once(caplog):
    q, kv, indices, sink, do = _bshd_inputs(1, 256, 64, 192, seed=13)
    h24 = _thd_inputs(256, 24, 128, seed=14)

    with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
        sink_fast = _run(q, kv, indices, do, sink=sink, sm_scale=MQA_SCALE, layout="bshd", variant="fast")
        heads_fast = _run(*h24, sm_scale=MLA_SCALE, layout="thd", variant="fast")
        _run(*h24, sm_scale=MLA_SCALE, layout="thd", variant="fast")
    assert_bitwise(_named(sink_fast), _named(_run(q, kv, indices, do, sink=sink, sm_scale=MQA_SCALE, layout="bshd")))
    assert_bitwise(_named(heads_fast), _named(_run(*h24, sm_scale=MLA_SCALE, layout="thd")))
    assert [record.getMessage() for record in caplog.records] == [
        "miles_ext_kernels.cake.dsa: the fast variant does not serve calls with an attention sink; running det",
        "miles_ext_kernels.cake.dsa: the fast variant does not serve 24 local query heads (only 64); running det",
    ]


def test_fast_runs_det_bit_identically_under_torch_deterministic_algorithms(caplog):
    """The fast backward raises PyTorch's alert while torch's deterministic algorithms are enabled, so a fast call runs
    det then: det's bits, one log line, no alert."""
    args = _thd_inputs(512, 64, 200, seed=17)
    det = _run(*args, sm_scale=MLA_SCALE, layout="thd", variant="det")
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger="miles_ext_kernels"):
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                fast = [_run(*args, sm_scale=MLA_SCALE, layout="thd", variant="fast") for _ in range(2)]
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])

    for result in fast:
        assert_bitwise(_named(result), _named(det))
    # the fast backward's alert names its entry (the autograd backward or a prepared runner's)
    alerts = [str(w.message) for w in seen if "deterministic implementation" in str(w.message)]
    assert not [message for message in alerts if "cake_dsa_sparse_attention_nondet" in message or "Runner" in message]
    assert [record.getMessage() for record in caplog.records] == [
        "miles_ext_kernels.cake.dsa: the fast variant does not serve calls under "
        "torch.use_deterministic_algorithms(True); running det"
    ]


# --------------------------------------------------------------------------- #
# The reviewed limits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("variant", "heads", "sink"), [("det", 16, True), ("det", 64, False), ("fast", 64, False)], ids=str
)
def test_batch_local_indices_outside_their_batch_element_are_empty_slots(variant, heads, sink):
    """Index S_kv + 5 of batch element 0 would read element 1's key 5; it is an empty slot, like -1."""
    batch, seq, topk = 2, 256, 128
    q, kv, indices, attn_sink, do = _bshd_inputs(batch, seq, heads, topk, seed=17)
    attn_sink = attn_sink if sink else None
    reach = torch.zeros_like(indices, dtype=torch.bool)
    reach[0, :, ::7] = True
    reach[1, :, 3::11] = True
    leaky = torch.where(reach, indices.clamp(min=0) + seq + (indices < 0).int() * 5, indices)
    assert (leaky[0][reach[0]] >= seq).all() and (leaky[1][reach[1]] >= seq).all()

    got = _run(q, kv, leaky, do, sink=attn_sink, sm_scale=MQA_SCALE, layout="bshd", variant=variant)
    want = _run(
        q, kv, indices.masked_fill(reach, -1), do, sink=attn_sink, sm_scale=MQA_SCALE, layout="bshd", variant=variant
    )

    if variant == "det":
        assert_bitwise(_named(got), _named(want))
    else:  # the same flat call; dkv sums FP32 atomics in a run-dependent order
        assert_bitwise(_named(got[:2]), _named(want[:2]))
        assert _rel_err(got[2], want[2]) < TOL


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_packed_indices_past_the_keys_are_empty_slots(variant):
    """A ``thd`` index at or past T_kv reads no key: the call gives the bits of the same call with ``-1`` there."""
    rows, heads, topk = 512, 64, 256
    q, kv, indices, do = _thd_inputs(rows, heads, topk, seed=19)
    gen = _generator(20)
    past = torch.rand(indices.shape, device=DEVICE, generator=gen) < 0.2
    beyond = rows + torch.randint(0, 1 << 20, indices.shape, device=DEVICE, generator=gen, dtype=torch.int32)

    got = _run(q, kv, torch.where(past, beyond, indices), do, sm_scale=MLA_SCALE, layout="thd", variant=variant)
    want = _run(q, kv, indices.masked_fill(past, -1), do, sm_scale=MLA_SCALE, layout="thd", variant=variant)

    if variant == "det":
        assert_bitwise(_named(got), _named(want))
    else:  # dkv sums FP32 atomics in a run-dependent order
        assert_bitwise(_named(got[:2]), _named(want[:2]))
        assert _rel_err(got[2], want[2]) < TOL
    ref = _flat_reference(q, kv, indices.masked_fill(past, -1), do, MLA_SCALE)
    _assert_matches((got[0], got[1], got[2].squeeze(1)), ref[:3])


def _transient(run, outputs_bytes):
    """Peak bytes ``run()`` allocates beyond what it returns (``outputs_bytes(result)``)."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base - outputs_bytes(result), result


# the per-call tensors outside the plan: the delta row sums, the FP32 key-side accumulator, the sink and its gradient
_SLACK = 4 * MIB


@pytest.mark.parametrize("budget_mib", [64, 256, 2048])
def test_det_attention_backward_stays_within_its_workspace_budget(budget_mib):
    rows, heads, topk = 2048, 64, 2048
    q, kv, indices, do = _thd_inputs(rows, heads, topk, seed=18)
    kv, indices = kv.squeeze(1), indices.squeeze(1)
    o, lse = dsa.sparse_attention_forward(q, kv, indices, sm_scale=MLA_SCALE)
    budget = budget_mib * MIB
    plan = reduction.plan_backward(rows, topk, 576, rows, budget)
    accumulator = rows * 576 * 4 + rows * heads * 4  # dkv32 and delta, freed when the call returns

    def run():
        return dsa.sparse_attention_backward(q, kv, o, do, indices, lse, sm_scale=MLA_SCALE, workspace_bytes=budget)

    transient, (dq, dkv, _) = _transient(run, lambda result: result[0].nbytes + result[1].nbytes)

    assert plan.one_sort == (budget_mib >= 256)  # 64 MiB: the sort of every pair does not fit
    assert transient <= budget + accumulator + _SLACK, (transient, budget, plan.rows_per_chunk)
    ref = ref_sparse_attention(q, kv, indices, do, MLA_SCALE)
    _assert_matches((dq, dkv), ref[1:3])


def test_det_indexer_backward_stays_within_its_workspace_budget():
    rows, heads, topk = 4096, 32, 2048
    index_q, index_k, weights, _, _ = _indexer_inputs(rows, heads, seed=19)
    gen = _generator(20)
    indices = torch.randint(-1, rows, (rows, topk), device=DEVICE, generator=gen, dtype=torch.int32)
    grad = torch.randn(rows, topk, device=DEVICE, generator=gen)
    budget = 64 * MIB
    masked = indices.nbytes  # the copy of the indices with the slots past the keys emptied

    def run():
        return dsa.indexer_backward(index_q, index_k, weights, indices, grad, workspace_bytes=budget)

    transient, _ = _transient(run, lambda result: sum(t.nbytes for t in result))

    assert transient <= budget + masked + _SLACK, (transient, budget)


def test_one_sort_and_one_sort_per_chunk_give_the_same_bits(monkeypatch):
    """The attention backward's single sort of every chunk and the per-chunk sorts give the same segments."""
    rows, heads, topk = 1024, 64, 2048
    q, kv, indices, do = _thd_inputs(rows, heads, topk, seed=21)
    budget = 512 * MIB
    assert reduction.plan_backward(rows, topk, 576, rows, budget).one_sort
    once = _run(q, kv, indices, do, sm_scale=MLA_SCALE, layout="thd", workspace_bytes=budget)
    plan = reduction.plan_backward

    def per_chunk(*args, **kwargs):
        chunks, _ = plan(*args, **kwargs)
        return reduction.BackwardPlan(chunks, False)

    monkeypatch.setattr(reduction, "plan_backward", per_chunk)
    assert_bitwise(
        _named(once), _named(_run(q, kv, indices, do, sm_scale=MLA_SCALE, layout="thd", workspace_bytes=budget))
    )


@pytest.mark.parametrize("sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("layout", ["thd", "bshd"])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_call_without_query_rows_returns_empty_outputs_and_zero_gradients(variant, layout, sink):
    heads, num_kv = 64, 96
    if layout == "thd":
        q = torch.zeros(0, heads, 576, device=DEVICE, dtype=torch.bfloat16)
        kv = torch.randn(num_kv, 576, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.zeros(0, 64, device=DEVICE, dtype=torch.int32)
        do = torch.zeros(0, heads, 512, device=DEVICE, dtype=torch.bfloat16)
    else:
        q = torch.zeros(2, 0, heads, 576, device=DEVICE, dtype=torch.bfloat16)
        kv = torch.randn(2, num_kv, 576, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.zeros(2, 0, 64, device=DEVICE, dtype=torch.int32)
        do = torch.zeros(2, 0, heads, 512, device=DEVICE, dtype=torch.bfloat16)
    attn_sink = torch.zeros(heads, device=DEVICE) if sink else None

    out, dq, dkv, dsink = _run(q, kv, indices, do, sink=attn_sink, sm_scale=MLA_SCALE, layout=layout, variant=variant)

    assert out.shape == (*q.shape[:-1], 512) and dq.shape == q.shape and dkv.shape == kv.shape
    assert not dkv.any()
    assert (dsink is None) != sink and (dsink is None or not dsink.any())


def test_an_indexer_call_without_query_rows_returns_empty_scores_and_zero_gradients():
    index_q = torch.zeros(0, 32, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    index_k = torch.randn(64, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.zeros(0, 32, device=DEVICE, requires_grad=True)
    bounds = torch.zeros(0, device=DEVICE, dtype=torch.int32)

    scores, indices = dsa.lightning_indexer(index_q, index_k, weights, bounds, bounds, 128)
    scores.sum().backward()

    assert scores.shape == indices.shape == (0, 128)
    assert index_q.grad.shape == index_q.shape and weights.grad.shape == weights.shape
    assert index_k.grad.shape == index_k.shape and not index_k.grad.any()


# --------------------------------------------------------------------------- #
# Repeatability: streams, processes, devices
# --------------------------------------------------------------------------- #


def test_default_and_side_stream_calls_give_the_same_bits():
    q, kv, indices, do = _thd_inputs(512, 64, 512, seed=22)
    side = torch.cuda.Stream()
    results = {}
    for variant in ("det", "fast"):
        runs = []
        for on_side in (False, True, False, True):
            if on_side:
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    result = _run(q, kv, indices, do, sm_scale=MLA_SCALE, layout="thd", variant=variant)
                torch.cuda.current_stream().wait_stream(side)
            else:
                result = _run(q, kv, indices, do, sm_scale=MLA_SCALE, layout="thd", variant=variant)
            runs.append(result if variant == "det" else result[:2])
        for result in runs[1:]:
            assert_bitwise(_named(runs[0]), _named(result))
        results[variant] = runs[0]
    _assert_matches(results["fast"], results["det"][:2])


@pytest.mark.parametrize("case", sorted(_det_repeat.CASES))
def test_det_repeats_bit_for_bit_in_one_process_and_in_a_fresh_one(case):
    first = _det_repeat.run_case(case)
    assert _det_repeat.run_case(case) == first

    pythonpath = os.pathsep.join(
        filter(None, [str(REPO_ROOT), str(REPO_ROOT / "tests"), os.environ.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, "-m", "cake.dsa._det_repeat", case],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {case: first}


@pytest.mark.multi_gpu
def test_a_second_device_gives_the_same_bits():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
        pytest.skip("the two GPUs differ in compute capability")
    torch.cuda.set_device(0)
    for case in sorted(_det_repeat.CASES):
        # the call on cuda:1 runs while cuda:0 is the current device
        assert _det_repeat.run_case(case, device="cuda:1") == _det_repeat.run_case(case, device="cuda:0"), case


@pytest.mark.multi_gpu
def test_fast_on_a_second_device_gives_the_same_out_and_dq():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
        pytest.skip("the two GPUs differ in compute capability")
    torch.cuda.set_device(0)
    inputs = _thd_inputs(512, 64, 512, seed=23)
    on_0 = _run(*inputs, sm_scale=MLA_SCALE, layout="thd", variant="fast")
    on_1 = _run(*(t.to("cuda:1") for t in inputs), sm_scale=MLA_SCALE, layout="thd", variant="fast")

    assert on_1[0].device == torch.device("cuda:1")
    assert_bitwise(_named(on_0[:2]), _named(tuple(t.to("cuda:0") for t in on_1[:2])))


# --------------------------------------------------------------------------- #
# The device check
# --------------------------------------------------------------------------- #


def test_check_device_names_the_architecture():
    assert dsa.check_device() == loader.ARCH_BY_CAPABILITY[torch.cuda.get_device_capability()]
    assert math.isclose(ops.LOG2E, math.log2(math.e))


def test_check_device_needs_the_architectures_toolkit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(NotImplementedError, match="SM100a, SM103a or SM107a"):
        dsa.check_device()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(loader, "MIN_CUDA", {**loader.MIN_CUDA, "sm_107a": (99, 0)})
    with pytest.raises(RuntimeError, match=r"DSA training kernels for sm_107a needs CUDA 99\.0 or newer"):
        dsa.check_device()
