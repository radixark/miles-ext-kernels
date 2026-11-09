"""Mode-word CAKE vendor packages import without FlashInfer installed (#8928)."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[2] / "miles_ext_kernels" / "cake" / "_vendor"
MODE_WORD_PACKAGE = re.compile(r"cake_.+_(?:det|nondet)_train(?:_kcp)?$")


def test_mode_word_vendor_modules_import_with_flashinfer_blocked():
    # the vendored modules import torch and tvm_ffi at module level
    pytest.importorskip("torch")
    pytest.importorskip("tvm_ffi")
    manifest = json.loads((VENDOR / "VENDORED.json").read_text())
    packages = sorted(
        name
        for op in manifest["ops"].values()
        for name, record in op.items()
        if MODE_WORD_PACKAGE.fullmatch(name)
        and record["source"]["kind"] == "flashinfer"
        and "/experimental/cake/" in record["source"]["path"]
    )
    assert packages, "No mode-word CAKE packages found in VENDORED.json"

    child = r"""
import importlib
import pathlib
import sys

sys.modules["flashinfer"] = None
root = pathlib.Path(sys.argv[1])
for name in sys.argv[2:]:
    directory = root / name
    assert directory.is_dir(), directory
    for path in sorted(directory.rglob("*.py")):
        parts = path.relative_to(directory).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module = "miles_ext_kernels.cake._vendor." + name
        if parts:
            module += "." + ".".join(parts)
        importlib.import_module(module)
assert sys.modules["flashinfer"] is None
assert not any(name.startswith("flashinfer.") for name in sys.modules)
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", child, str(VENDOR), *packages],
        cwd=VENDOR.parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
