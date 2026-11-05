"""Fixed det DSA calls whose outputs and gradients the GPU tests compare bit for bit: in one process, in a fresh
process (``python _det_repeat.py CASE [DEVICE]`` prints ``{CASE: {tensor: sha256}}``) and on a second device.

Each case makes its inputs from a fixed seed on the device, runs the forward and the backward of the det variant and
returns the SHA-256 of every output's bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys

import torch

from miles_ext_kernels.cake import dsa

MLA_SCALE = 192**-0.5  # the absorbed MLA's softmax scale: that of the (128 + 64)-channel head, not 576 ** -0.5


def _digest(tensor: torch.Tensor) -> str:
    flat = tensor.detach().contiguous().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _causal_indices(rows: int, topk: int, num_keys: int, gen: torch.Generator, device) -> torch.Tensor:
    limits = ((torch.arange(rows, device=device) + 1) * num_keys // rows).clamp(min=1, max=num_keys)
    keys = torch.arange(num_keys, device=device)
    score = torch.rand(rows, num_keys, device=device, generator=gen).masked_fill(keys >= limits[:, None], -1.0)
    count = min(topk, num_keys)
    picked = torch.topk(score, count, dim=-1)
    idx = torch.where(picked.values >= 0, picked.indices, -1).int()
    return torch.nn.functional.pad(idx, (0, topk - count), value=-1) if count < topk else idx


def _attention(device, *, layout, heads, d_qk, rows, topk, sink, seed, batch=1):
    gen = torch.Generator(device=device).manual_seed(seed)
    if layout == "thd":
        q = torch.randn(rows, heads, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
        kv = torch.randn(rows, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
        indices = _causal_indices(rows, topk, rows, gen, device)
        do = torch.randn(rows, heads, 512, device=device, dtype=torch.bfloat16, generator=gen)
    else:
        q = torch.randn(batch, rows, heads, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
        kv = torch.randn(batch, rows, d_qk, device=device, dtype=torch.bfloat16, generator=gen)
        indices = torch.stack([_causal_indices(rows, topk, rows, gen, device) for _ in range(batch)])
        do = torch.randn(batch, rows, heads, 512, device=device, dtype=torch.bfloat16, generator=gen)
    attn_sink = torch.randn(heads, device=device, generator=gen).requires_grad_(True) if sink else None
    q.requires_grad_(True)
    kv.requires_grad_(True)
    scale = MLA_SCALE if d_qk == 576 else 512**-0.5
    out = dsa.sparse_attention(q, kv, indices, sm_scale=scale, attn_sink=attn_sink, layout=layout)
    out.backward(do)
    result = {"out": out, "dq": q.grad, "dkv": kv.grad}
    if sink:
        result["dsink"] = attn_sink.grad
    return result


def _indexer(device, *, heads, rows, topk, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    index_q = torch.randn(rows, heads, 128, device=device, dtype=torch.bfloat16, generator=gen).requires_grad_(True)
    index_k = torch.randn(rows, 128, device=device, dtype=torch.bfloat16, generator=gen).requires_grad_(True)
    weights = torch.randn(rows, heads, device=device, generator=gen).requires_grad_(True)
    ks = torch.zeros(rows, device=device, dtype=torch.int32)
    ke = (torch.arange(rows, device=device) + 1).int()
    scores, indices = dsa.lightning_indexer(index_q, index_k, weights, ks, ke, topk)
    grad = torch.randn(scores.shape, device=device, generator=gen).masked_fill(torch.isinf(scores), 0.0)
    scores.masked_fill(torch.isinf(scores), 0.0).backward(grad)
    return {"scores": scores, "indices": indices, "dq": index_q.grad, "dk": index_k.grad, "dw": weights.grad}


CASES = {
    "rope_h64": lambda device: _attention(
        device, layout="thd", heads=64, d_qk=576, rows=1024, topk=2048, sink=False, seed=21
    ),
    "rope_h24": lambda device: _attention(
        device, layout="thd", heads=24, d_qk=576, rows=512, topk=256, sink=False, seed=22
    ),
    "nope_sink_h16": lambda device: _attention(
        device, layout="bshd", heads=16, d_qk=512, rows=256, topk=192, sink=True, seed=23, batch=2
    ),
    "indexer_h32": lambda device: _indexer(device, heads=32, rows=1024, topk=256, seed=24),
}


def run_case(case: str, device: str = "cuda") -> dict[str, str]:
    """``{tensor: sha256}`` of every output and gradient of ``case`` on ``device`` (the current device stays)."""
    result = CASES[case](device)
    torch.cuda.synchronize(device)
    return {name: _digest(tensor) for name, tensor in result.items()}


if __name__ == "__main__":
    name = sys.argv[1]
    print(json.dumps({name: run_case(name, *sys.argv[2:])}, sort_keys=True))
