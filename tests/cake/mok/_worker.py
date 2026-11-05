"""Run MoK cases in a fresh process, one rank per process: under torchrun (an expert-parallel group of every launched
rank, env:// rendezvous) or alone (a group of one rank on ``--device``). Writes this rank's records as JSON to
``--out`` (``--out.rank<r>`` under torchrun) and exits non-zero when a case fails a check. Each change of a rank's
communication-SM split is printed to stderr (``[cake_mok_bf16 rank <r>] communication split -> (<fwd>, <bwd>) ...``).
When PyTorch symmetric memory gives the group no multicast address (the op's ``NoMulticastError``: the platform has no
NVLink multicast for these GPUs), the rank runs no case and records the error as ``no_multicast``.

``test_mok_gpu.py`` runs it alone to repeat its cases in a fresh process; ``test_mok_distributed_gpu.py`` launches it
on 4 and 8 GPUs, after it prebuilt their libraries. Not collected by pytest.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[1]
REPO_ROOT = HERE.parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", type=int, default=0, help="the CUDA device of a single-rank run")
    parser.add_argument("--routing", choices=("random", "empty"), default="random")
    parser.add_argument(
        "--precision", choices=("bf16", "mxfp8"), default="bf16", help="the row of the workspace's imbalance split"
    )
    parser.add_argument("--cases", nargs="*", help="default: every case")
    parser.add_argument("--no-checks", action="store_true", help="only the digests of the eager outputs")
    args = parser.parse_args()
    # the test package ``cake.mok`` (tests/cake/mok) and this checkout
    sys.path[:0] = [str(REPO_ROOT), str(TESTS)]

    import torch
    import torch.distributed as dist
    from cake.mok import _cases as cases

    if "RANK" in os.environ:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(seconds=900))
        out = Path(f"{args.out}.rank{dist.get_rank()}")
    else:
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1, device_id=device)
        out = Path(args.out)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    from miles_ext_kernels import cake
    from miles_ext_kernels.cake._impl.mok import ops
    from miles_ext_kernels.cake._runtime import loader

    names = args.cases or list(cases.CASES)
    result = dict(
        rank=dist.get_rank(),
        ep=dist.get_world_size(),
        arch=cake.device_arch(device),
        device=torch.cuda.get_device_name(device),
        nvcc=loader.get_cuda_version().full,
        records=[],
    )
    try:
        problem = cases.Problem(
            dist.group.WORLD, device, routing=args.routing, precision=args.precision, comm_split_log=True
        )
    except ops.NoMulticastError as error:
        result["no_multicast"] = str(error)
        out.write_text(json.dumps(result, indent=1))
        dist.destroy_process_group()
        return 0
    result.update(
        routing=args.routing,
        precision=args.precision,
        imbalance_comm_sms=problem.config.imbalance_comm_sms,
        imbalance_threshold=problem.config.imbalance_threshold,
    )
    bf16 = None
    for name in names:
        record, outputs = cases.run_case(problem, name, checks=not args.no_checks, bf16_outputs=bf16)
        if name == "bf16":
            bf16 = outputs
        record["failures"] = cases.failures(record)
        result["records"].append(record)
        print(json.dumps(dict(case=name, rank=record["rank"], digest=record["digest"], failures=record["failures"])))
    out.write_text(json.dumps(result, indent=1))
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    return 1 if any(record["failures"] for record in result["records"]) else 0


if __name__ == "__main__":
    sys.exit(main())
