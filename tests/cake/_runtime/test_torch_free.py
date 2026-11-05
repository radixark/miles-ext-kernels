"""miles registers, validates and configures every op in a process without torch, tvm_ffi or flashinfer."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS = REPO_ROOT / "tests"

_WITHOUT_TORCH = """
import argparse
import importlib
import sys
from pathlib import Path

for blocked in ("torch", "tvm_ffi", "flashinfer"):
    sys.modules[blocked] = None  # importing it raises ImportError

import miles_ext_kernels

assert Path(miles_ext_kernels.__file__).resolve().parents[1] == Path(sys.argv[1]).resolve(), miles_ext_kernels.__file__
loaded = sorted(name for name in sys.modules if name.startswith("miles_ext_kernels."))
assert loaded == [], f"importing miles_ext_kernels imported {loaded}"

from cake._runtime import synthetic  # the tests' synthetic op, next to the registered ones
from miles_ext_kernels import cake
from miles_ext_kernels.cake._runtime import registry

real = registry.OPS
for op in real:
    importlib.import_module(f"miles_ext_kernels.cake.{op}")
synthetic.install()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--synth-backend")
miles_ext_kernels.add_arguments(parser)
# miles hands validate_args its whole parsed namespace, deterministic mode included
for argv in ([], ["--synth-backend", "cake"]):
    for deterministic in (False, True):
        args = parser.parse_args(argv)
        args.deterministic_mode = deterministic
        miles_ext_kernels.validate_args(args)
        miles_ext_kernels.configure(args)
miles_ext_kernels.configure()
assert miles_ext_kernels.release() == 0
for blocked in ("torch", "tvm_ffi", "flashinfer"):
    assert sys.modules[blocked] is None, blocked
print(len(real), len(registry.OPS))
"""


def test_every_op_registers_validates_and_configures_without_torch_tvm_ffi_or_flashinfer():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), str(TESTS), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_TORCH, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    real, total = map(int, result.stdout.split())
    assert total == real + 1  # the synthetic op ran through every hook too
