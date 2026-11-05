"""Tensor + sequence parallelism on two GPUs: the norm's weight stays one weight across the ranks.

Under sequence parallelism each tensor-parallel rank normalizes other tokens, so each computes another partial weight
gradient. Megatron (``finalize_model_grads``) all-reduces the gradient of every parameter that carries
``sequence_parallel = True`` across the tensor-parallel group; Transformer Engine's norms mark their weight so when
built with ``sequence_parallel=True``. Two ranks run that rule over ``ZeroCenteredRMSNorm(sequence_parallel=True)`` for
a few SGD steps: the rule selects the weight, the ranks' weights stay bit-identical, and they follow the single-GPU run
over the whole sequence. Needs two SM100a / SM103a / SM107a GPUs and NCCL.
"""

import json
import socket

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake import rmsnorm

from ..helpers import rel_rms, requires_arch

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

WORLD = 2
HIDDEN = 2048
TOKENS_PER_RANK = 1000
STEPS = 3
LR = 1e-2
EPS = 1e-6

pytestmark = [
    pytest.mark.multi_gpu,
    requires_arch("sm_100a", "sm_103a", "sm_107a"),
    pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason=f"needs {WORLD} GPUs"),
]


def _shard(step, rank, device):
    """``(x, dy)`` of ``rank``'s tokens at ``step``: the rank's slice of the step's whole sequence."""
    g = torch.Generator(device=device).manual_seed(7919 * step + 17)
    x = torch.randn(WORLD * TOKENS_PER_RANK, 1, HIDDEN, device=device, generator=g).bfloat16()
    dy = torch.randn(WORLD * TOKENS_PER_RANK, 1, HIDDEN, device=device, generator=g).bfloat16()
    rows = slice(rank * TOKENS_PER_RANK, (rank + 1) * TOKENS_PER_RANK)
    return x[rows], dy[rows]


def _norm(device):
    norm = rmsnorm.ZeroCenteredRMSNorm(HIDDEN, EPS, device=device, sequence_parallel=True)
    g = torch.Generator(device=device).manual_seed(1234)
    with torch.no_grad():
        norm.weight.copy_(0.1 * torch.randn(HIDDEN, device=device, generator=g))
    return norm


def _worker(rank, port, path):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    try:
        cake.configure(cake_rmsnorm_variant="det")
        cake.prebuild("rmsnorm", hidden_size=HIDDEN)
        norm = _norm(device)
        # Megatron's rule: the parameters whose gradient it all-reduces across the tensor-parallel ranks under SP
        selected = [name for name, param in norm.named_parameters() if getattr(param, "sequence_parallel", False)]
        optimizer = torch.optim.SGD(norm.parameters(), lr=LR)
        for step in range(STEPS):
            x, dy = _shard(step, rank, device)
            optimizer.zero_grad(set_to_none=True)
            norm(x).backward(dy)
            for param in norm.parameters():
                if getattr(param, "sequence_parallel", False):
                    dist.all_reduce(param.grad)
            optimizer.step()
        weights = [torch.empty_like(norm.weight) for _ in range(WORLD)]
        dist.all_gather(weights, norm.weight.detach())
        if rank == 0:
            same = all(torch.equal(weights[0].view(torch.int16), w.view(torch.int16)) for w in weights)
            result = {"selected": selected, "same": same, "weight": weights[0].float().cpu().tolist()}
            with open(path, "w") as file:
                json.dump(result, file)
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_the_ranks_keep_one_weight_under_sequence_parallelism(tmp_path):
    path = tmp_path / "result.json"
    torch.multiprocessing.spawn(_worker, args=(_free_port(), str(path)), nprocs=WORLD, join=True)
    result = json.loads(path.read_text())

    assert result["selected"] == ["weight"]
    assert result["same"], "the ranks' weights differ"
    # the same steps on one GPU over each step's whole sequence: the all-reduced gradient is the whole one
    cake.configure(cake_rmsnorm_variant="det")
    device = torch.device("cuda", torch.cuda.current_device())
    norm = _norm(device)
    optimizer = torch.optim.SGD(norm.parameters(), lr=LR)
    for step in range(STEPS):
        shards = [_shard(step, rank, device) for rank in range(WORLD)]
        x = torch.cat([x for x, _ in shards])
        optimizer.zero_grad(set_to_none=True)
        norm(x).backward(torch.cat([dy for _, dy in shards]))
        optimizer.step()
    initial = _norm(device).weight.detach().double().cpu()
    ranks = torch.tensor(result["weight"], dtype=torch.float64) - initial
    single = norm.weight.detach().double().cpu() - initial
    # The updates differ by the BF16 rounding of the partial gradients and of their sum (well under 1 %); a gradient
    # that was not all-reduced (half of it) or reduced twice would be 50 % or 100 % off.
    assert rel_rms(ranks, single) <= 0.05
