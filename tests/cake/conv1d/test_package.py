"""The conv1d op's vendored packages: their manifest entries (one FlashInfer commit, the environment read their
context-parallel modules may do) and how the runtime loads them (the loader installed on both, FlashInfer's
``..cake.<name>`` path resolving to the same module, nothing importing flashinfer). The loading checks need torch and
tvm_ffi, not CUDA."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED = REPO_ROOT / "miles_ext_kernels" / "cake" / "_vendor" / "VENDORED.json"
FI_COMMIT = "c5c39b0e87f027011c86881aa2935a83e1ce53e7"
FI_PATHS = {
    "cake_causal_conv1d_det_train": "flashinfer/experimental/cake/cake_causal_conv1d_det_train",
    "cake_causal_conv1d_nondet_train": "flashinfer/experimental/cake/cake_causal_conv1d_nondet_train",
}
CAKE_BUILD_BLOB = "00756137eb159e30a77907eda84fd245123d87ea"


def test_the_manifest_records_both_packages_at_one_flashinfer_commit():
    block = json.loads(VENDORED.read_text())["ops"]["conv1d"]

    assert sorted(block) == sorted(FI_PATHS)
    for package, entry in block.items():
        assert entry["source"] == {
            "commit": FI_COMMIT,
            "kind": "flashinfer",
            "path": FI_PATHS[package],
            "repo": "https://github.com/flashinfer-ai/flashinfer",
        }
        assert entry["allow"] == {"env_read": ["kcp.py"]}
        assert entry["files"]["cake_build.py"] == CAKE_BUILD_BLOB
        assert {"__init__.py", "build.json", "cake_jit.py", "kcp.py", "kcp_jit.py"} <= set(entry["files"])


def test_the_implementation_installs_the_loader_on_both_packages():
    pytest.importorskip("torch")
    pytest.importorskip("tvm_ffi")
    from miles_ext_kernels.cake._impl.conv1d import ops
    from miles_ext_kernels.cake._runtime import loader

    vendor = "miles_ext_kernels.cake._vendor"
    for variant, name in ops.PACKAGES.items():
        package = ops._package(variant)
        assert package.__name__ == f"{vendor}.{name}"
        assert package.cake_build._jit_toolchain.__module__ == loader.__name__
        # the context-parallel module builds through the loader too
        assert ops._kcp_module(variant).__name__ == f"{vendor}.{name}.kcp"
        assert sys.modules[f"{vendor}.{name}.cake_build"]._jit_toolchain.__module__ == loader.__name__
        # FlashInfer keeps both packages in flashinfer/experimental/cake/: its ``..cake.<name>`` path is the package
        assert importlib.import_module(f"{vendor}.cake.{name}") is package


_NO_FLASHINFER = """
import sys

sys.modules["flashinfer"] = None  # any import of flashinfer raises ImportError
from miles_ext_kernels.cake._impl.conv1d import ops

for variant in ops.PACKAGES:
    ops._kcp_module(variant)
print(sorted(name for name in sys.modules if name.startswith("miles_ext_kernels.cake._vendor.")))
"""


def test_the_packages_import_without_flashinfer():
    pytest.importorskip("torch")
    pytest.importorskip("tvm_ffi")
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _NO_FLASHINFER],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for name in FI_PATHS:
        assert f"miles_ext_kernels.cake._vendor.{name}.kcp" in result.stdout
