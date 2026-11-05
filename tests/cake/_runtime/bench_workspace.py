"""Micro-benchmark (not a test) of the host time of ``workspace.take`` and of the lookups an op makes at every call.

It runs on the CPU and needs torch. ``cuda`` rows run the pool's CUDA path on the fake device layer of
:mod:`fake_cuda` (buffers on the CPU): they time the pool's own Python, not the CUDA runtime calls a real device adds
(switching devices, reading the stream and the capture state) nor a CUDA allocation, so they understate what a GPU
process pays. Each take row cycles over 13 roles, the takes of one GDN backward; the ``empty`` rows are the floor, one
``torch.empty`` of the same shapes (on the fake layer for ``cuda``). Run it from a checkout whose ``miles_ext_kernels``
is importable (``pip install -e .`` or ``PYTHONPATH=.``); another checkout's package may be measured with this script,
as long as it has the same entry points::

    python tests/cake/_runtime/bench_workspace.py [--calls 100000] [--repeat 7] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from importlib import import_module
from pathlib import Path

import torch

import miles_ext_kernels
from miles_ext_kernels import cake
from miles_ext_kernels.cake._runtime import workspace

# tests/, where the synthetic op and the fake device layer are the package cake._runtime
TESTS = Path(__file__).resolve().parents[2]
ROLES = [(f"b_role{index}", (4, 12, 64 + index)) for index in range(13)]


def _per_call(run, calls: int, repeat: int) -> dict[str, float]:
    """Median and minimum nanoseconds per call of ``run(calls)`` over ``repeat`` timed runs, after one warm-up."""
    run(max(1, calls // 10))
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        run(calls)
        samples.append((time.perf_counter_ns() - start) / calls)
    return {"median_ns": round(statistics.median(samples), 1), "min_ns": round(min(samples), 1)}


def _takes(device):
    take, dtype = workspace.take, torch.float32

    def run(calls: int) -> None:
        for _ in range(calls // len(ROLES)):
            for role, shape in ROLES:
                take("bench", role, shape, dtype, device)

    return run


def _empties(device):
    empty, dtype = torch.empty, torch.float32

    def run(calls: int) -> None:
        for _ in range(calls // len(ROLES)):
            for _role, shape in ROLES:
                empty(shape, dtype=dtype, device=device)

    return run


def _repeated(fn, *args):
    def run(calls: int) -> None:
        for _ in range(calls):
            fn(*args)

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=13 * 8000, help="calls per timed run (default %(default)s)")
    parser.add_argument("--repeat", type=int, default=7, help="timed runs per row (default %(default)s)")
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    parser.add_argument("--label", default="", help="a name for the measured tree, kept in the output")
    args = parser.parse_args()

    sys.path.insert(0, str(TESTS))
    synthetic = import_module("cake._runtime.synthetic")
    fake = import_module("cake._runtime.fake_cuda").FakeCuda(torch)
    torch.set_num_threads(1)
    synthetic.install()  # registers the synthetic op "synth" for the lookups
    rows = []

    def row(name: str, run, **what) -> None:
        rows.append({"row": name, **what, **_per_call(run, args.calls, args.repeat)})
        print(f"{name:<34} {json.dumps(what):<44} {rows[-1]['median_ns']:>10.1f} ns  (min {rows[-1]['min_ns']:.1f})")

    missing = object()  # torch._C._cuda_* of a CPU build of torch
    saved = [(owner, name, getattr(owner, name, missing)) for owner, name, _ in fake.patches()]

    def faked(rows_of) -> None:
        for owner, name, value in fake.patches():
            setattr(owner, name, value)
        try:
            rows_of()
        finally:
            for owner, name, value in saved:
                if value is missing:
                    delattr(owner, name)
                else:
                    setattr(owner, name, value)

    for bound in (0, 1 << 30):
        cake.configure(cake_workspace_pool_bytes=bound)
        cake.release()
        row("take", _takes(torch.device("cpu")), device="cpu", bound=bound)

        def cuda_rows(bound=bound) -> None:
            row("take", _takes(torch.device("cuda", 0)), device="cuda:0 (fake, current)", bound=bound)
            row("take", _takes("cuda"), device="cuda (fake, current device)", bound=bound)
            row("take", _takes(torch.device("cuda", 1)), device="cuda:1 (fake, not current)", bound=bound)

        faked(cuda_rows)
        cake.release()
    row("empty", _empties(torch.device("cpu")), device="cpu")
    faked(lambda: row("empty", _empties(torch.device("cuda", 0)), device="cuda:0 (fake)"))

    cake.configure()
    row("variant (unconfigured)", _repeated(cake.variant, "synth"))
    row("option (op)", _repeated(cake.option, "synth", "cake_synth_scale"))
    row("option (runtime)", _repeated(cake.option, None, "cake_workspace_pool_bytes"))
    cake.configure(cake_synth_variant="fast")
    row("variant (configured)", _repeated(cake.variant, "synth"))
    cake.configure()

    report = {
        "label": args.label,
        "package": str(Path(miles_ext_kernels.__file__).resolve().parent),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "calls": args.calls,
        "repeat": args.repeat,
        "rows": rows,
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
