"""The MoK expert-parallel MoE training layer on 4 and 8 GPUs of one node (every rank in one NVLink domain).

The libraries of each layout are prebuilt here first (``prebuild('mok', ...)``), then ``_worker.py`` runs under
torchrun: EP 4 with 256 routed experts (64 per rank) and EP 8 with 288 (36 per rank), top-8, unequal source counts with
empty ranks (513, 0, 1031, 256, ...). On every rank, every case of ``_cases.CASES`` passes its checks (the accuracy
gates against the FP32 reference and FlashInfer's own, bit-for-bit eager repeats, graph replays, recompute, another
communication-SM split and a split that differs between the ranks, zero shared gradients on an empty rank). A second
launch, in fresh processes, repeats every output of every rank bit for bit.

Each layout runs with random routing and with routing that sends every row to the experts of rank 0 (a load of EP
times the mean). On a GPU with an imbalance split row for the workspace's precision (B200: BF16 and MXFP8; SM107a
with 212 SMs: MXFP8), that routing switches each rank's communication-SM split from its latest completed load
observation, a timing-dependent choice, and every rank must log the switch; the MXFP8 row runs only where the GPU has
one. Needs the GPUs (SM100a / SM103a / SM107a); skips with fewer. The kernels exchange rows over NVLink multicast:
where PyTorch symmetric memory gives the group no multicast address (every rank's workspace raises the op's
``NoMulticastError``, a property of the platform), the launch skips with that error.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.mok import ops  # noqa: E402

from . import _cases  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
# test id -> (EP, routing, the workspace precision that selects the imbalance split row)
LAUNCHES = {
    f"ep{ep}-{name}": (ep, routing, precision)
    for ep in (4, 8)
    for name, routing, precision in (
        ("random", "random", "bf16"),
        ("empty", "empty", "bf16"),
        ("empty-mxfp8-split", "empty", "mxfp8"),
    )
}


def launch(ep: int, out: Path, *extra: str) -> tuple[list[dict], str]:
    """Every rank's result and the launch's stderr."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={ep}",
        str(HERE / "_worker.py"),
        "--out",
        str(out),
        *extra,
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, f"{' '.join(command)}\n{result.stdout[-4000:]}\n{result.stderr[-6000:]}"
    return [json.loads(Path(f"{out}.rank{rank}").read_text()) for rank in range(ep)], result.stderr


@pytest.mark.multi_gpu
@pytest.mark.parametrize("ep,routing,precision", list(LAUNCHES.values()), ids=list(LAUNCHES))
def test_expert_parallel_training_on_one_node(ep, routing, precision, tmp_path):
    if torch.cuda.device_count() < ep:
        pytest.skip(f"needs {ep} GPUs")
    capabilities = {tuple(torch.cuda.get_device_capability(index)) for index in range(ep)}
    if not capabilities <= set(cake.ARCH_BY_CAPABILITY):
        pytest.skip(f"the Cake MoK kernels need SM100a / SM103a / SM107a, not {sorted(capabilities)}")
    if len(capabilities) != 1:
        pytest.skip(f"one GPU model per launch, not {sorted(capabilities)}")
    api = ops._vendored("api")
    hot = api.resolve_imbalance_comm_sms(precision, 0)
    if precision != "bf16" and hot is None:
        pytest.skip(f"no {precision} imbalance split on this GPU: the run with the bf16 workspace covers this routing")
    switches = routing == "empty" and hot is not None and ep >= api.resolve_imbalance_threshold(precision, 0)
    cake.prebuild("mok", device=0, **_cases.model(ep))
    options = ("--routing", routing, "--precision", precision)

    checked, log = launch(ep, tmp_path / "checked", *options)
    no_multicast = [result.get("no_multicast") for result in checked]
    if any(no_multicast):
        assert all(no_multicast), no_multicast
        pytest.skip(f"no NVLink multicast for the {ep} GPUs: {no_multicast[0]}")
    names = [record["case"] for record in checked[0]["records"]]
    assert names, checked[0]
    for rank, result in enumerate(checked):
        assert result["rank"] == rank and result["ep"] == ep
        assert [record["case"] for record in result["records"]] == names
        for record in result["records"]:
            assert record["failures"] == [], (rank, record)
    assert sorted(names) == sorted(_cases.CASES)
    if switches:
        for rank in range(ep):
            assert f"[cake_mok_bf16 rank {rank}] communication split -> {tuple(hot)}" in log, (rank, log[-6000:])

    fresh, _ = launch(ep, tmp_path / "fresh", *options, "--no-checks", "--cases", *names)
    for rank, (a, b) in enumerate(zip(checked, fresh, strict=True)):
        assert {r["case"]: r["digest"] for r in a["records"]} == {r["case"]: r["digest"] for r in b["records"]}, rank
