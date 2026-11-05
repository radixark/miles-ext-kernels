"""The context-parallel short conv (``causal_conv1d_kcp``) on 2 and 4 GPUs against the single-GPU reference.

One process per rank (``torch.multiprocessing.spawn``, NCCL), with the contexts of each GDN / KDA context-parallel
route this checkout vendors (a route it does not vendor skips). Layouts: contiguous, zigzag, per-document zigzag,
ragged tables whose fragments read history rows from one or two predecessor ranks, and a rank without tokens
(4 GPUs). Per row and rank: ``det``'s ``y`` bitwise the single-GPU ``det`` forward of the whole pack on the rank's
tokens; ``y`` / ``dx`` within one BF16 ulp and the rank's weight-gradient partial within the route's budget of the
FP64 reference; ``det`` bitwise run to run in both launch forms of its backward; ``fast`` with ``y`` / ``dx`` bitwise
``det``'s and its partial within budget; the multi-tensor call bitwise the single calls; a checkpoint recompute
with a handle bitwise the direct pass. Needs ``world_size`` SM100a / SM103a / SM107a GPUs (marked ``multi_gpu``).
"""

import json
import socket

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
pytest.importorskip("tvm_ffi")

import torch.multiprocessing as mp  # noqa: E402

from miles_ext_kernels.cake._runtime import loader  # noqa: E402

from ._kcp import CONTEXT_BUILDERS, context_builder  # noqa: E402
from ._kcp_dist import rows, run_rank  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake short-conv kernels need SM100a / SM103a / SM107a", allow_module_level=True)

pytestmark = pytest.mark.multi_gpu


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("builder", sorted(CONTEXT_BUILDERS))
def test_context_parallel_ranks_match_the_single_gpu_reference(builder, world_size, tmp_path):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs, {torch.cuda.device_count()} visible")
    if context_builder(builder) is None:
        pytest.skip(f"this checkout does not vendor the {builder} context-parallel package")

    mp.spawn(run_rank, args=(world_size, builder, _free_port(), str(tmp_path)), nprocs=world_size, join=True)

    expected = 10 * len(rows(world_size))  # ten checks per row
    for rank in range(world_size):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        flags = result["flags"]
        assert len(flags) == expected, sorted(flags)
        failed = sorted(key for key, ok in flags.items() if not ok)
        assert not failed, (rank, failed, result["stats"])
