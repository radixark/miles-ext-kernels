"""The widths this process cannot launch, for the GPU tests of the Cake input RMSNorm.

On SM107a, hidden size 4096 launches programs in CUDA 13.4's oversized shared-memory mode; a process whose CUDA
runtime is older than 13.4 cannot launch them, and the op refuses that width with a ``RuntimeError`` before any launch.
A GPU test at such a width asserts that refusal instead of running (:func:`runs_or_is_refused`);
``test_rmsnorm_gpu.py::test_the_refusal_is_exactly_what_fails_to_launch`` holds the refusal to an unchecked launch.
"""

import contextlib
import re

import pytest

from miles_ext_kernels.cake import rmsnorm


def refusal(hidden):
    """``None`` where this process can launch every program of width ``hidden`` on the current device; else the
    ``RuntimeError`` the op raises for it before any launch. Any other error propagates."""
    try:
        rmsnorm.check_supported(hidden)
    except RuntimeError as exc:
        if "oversized shared-memory mode" not in str(exc):
            raise
        return exc
    return None


@contextlib.contextmanager
def runs_or_is_refused(hidden):
    """The body of a test at width ``hidden``: runs as it is where this process launches the width; elsewhere it must
    raise the op's refusal of the width (at its first construction, prebuild or call)."""
    refused = refusal(hidden)
    if refused is None:
        yield
        return
    with pytest.raises(RuntimeError, match=re.escape(str(refused))):
        yield
