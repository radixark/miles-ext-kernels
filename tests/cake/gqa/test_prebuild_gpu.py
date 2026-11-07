"""The prebuild contract of the gqa op: after ``prebuild("gqa")``, no call compiles (README, Prebuild).

``ROUTES`` names every call-time path of the op, each run forward and backward: the exact ``max_seqlen_k`` and the
``T_k`` bound, the head groups miles runs, keys fewer than queries and empty segments, long keys, no query and no
key, a softmax scale, the option ``cake_gqa_check_seqlens``, a side stream and the workspace pool turned on (the
default bound, 0, keeps nothing). The op takes no model parameter. In a fresh process, the libraries a call loads are
the four ``prebuild`` built, and flashinfer is never imported. Needs a B200 / GB200, B300 / GB300 or R200 and skips
elsewhere.
"""

import os
import subprocess
import sys
from functools import partial
from pathlib import Path

from miles_ext_kernels import cake
from miles_ext_kernels.cake import gqa

from ..helpers import POOLED, check_prebuild, requires_arch

pytestmark = requires_arch("sm_100a", "sm_103a", "sm_107a")

REPO_ROOT = Path(__file__).resolve().parents[3]
MILES = (1000, 517, 2049, 18)


def _fwd_bwd(q_lens, k_lens, h_q, h_kv, *, max_seqlen_k=None, softmax_scale=None, stream=None):
    import torch

    from . import _cases as c

    device = torch.device("cuda", torch.cuda.current_device())
    x = c.inputs(q_lens, k_lens, h_q, h_kv, device)
    bound = c.longest_keys(x) if max_seqlen_k is None else max_seqlen_k
    stream = stream or torch.cuda.current_stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
        out = gqa.varlen_gqa_attention(q, k, v, x["cu_q"], x["cu_k"], max_seqlen_k=bound, softmax_scale=softmax_scale)
        out.backward(x["dout"])
    torch.cuda.synchronize(device)


def _route(variant, *args, options=None, side_stream=False, **kwargs):
    if options:
        cake.configure(cake_gqa_variant=variant, **options)
    if side_stream:
        import torch

        kwargs["stream"] = torch.cuda.Stream()
    _fwd_bwd(*args, **kwargs)


ROUTES = {
    "miles_exact_max_seqlen_k": partial(_route, q_lens=MILES, k_lens=MILES, h_q=8, h_kv=2),
    "max_seqlen_k_total_keys": partial(_route, q_lens=MILES, k_lens=MILES, h_q=8, h_kv=2, max_seqlen_k=sum(MILES)),
    "group_2": partial(_route, q_lens=(300, 77), k_lens=(513, 77), h_q=8, h_kv=4),
    "group_6": partial(_route, q_lens=MILES, k_lens=MILES, h_q=6, h_kv=1),
    "group_8": partial(_route, q_lens=MILES, k_lens=MILES, h_q=16, h_kv=2),
    "keys_fewer_than_queries": partial(_route, q_lens=(200, 150, 64), k_lens=(77, 0, 63), h_q=24, h_kv=4),
    "empty_segments": partial(_route, q_lens=(0, 100, 0, 37, 0), k_lens=(0, 300, 50, 37, 0), h_q=24, h_kv=4),
    "long_keys": partial(_route, q_lens=(256, 255), k_lens=(4099, 2049), h_q=24, h_kv=4),
    "single_segment": partial(_route, q_lens=(4096,), k_lens=(4096,), h_q=16, h_kv=2),
    "no_queries": partial(_route, q_lens=(0, 0), k_lens=(64, 32), h_q=8, h_kv=2),
    "no_keys": partial(_route, q_lens=(64, 32), k_lens=(0, 0), h_q=8, h_kv=2),
    "softmax_scale": partial(_route, q_lens=MILES, k_lens=MILES, h_q=8, h_kv=2, softmax_scale=0.05),
    "check_seqlens": partial(
        _route, q_lens=MILES, k_lens=MILES, h_q=8, h_kv=2, options={"cake_gqa_check_seqlens": True}
    ),
    "side_stream": partial(_route, q_lens=MILES, k_lens=MILES, h_q=8, h_kv=2, side_stream=True),
    "pooled": partial(_route, q_lens=MILES, k_lens=MILES, h_q=8, h_kv=2, options=dict(POOLED)),
}


def test_one_prebuild_serves_every_route():
    check_prebuild("gqa", ROUTES)


def test_prebuild_returns_the_four_libraries_once_built():
    from miles_ext_kernels.cake._impl.gqa import ops

    paths = cake.prebuild("gqa")
    again = cake.prebuild("gqa")

    assert again == paths and len(paths) == len(ops.STAGES) and all(path.is_file() for path in paths)
    assert [path.parent.name for path in paths] == [ops.PACKAGE] * len(ops.STAGES)


_FRESH_PROCESS = """
import sys

import torch

from miles_ext_kernels import cake
from miles_ext_kernels.cake import gqa
from miles_ext_kernels.cake._impl.gqa import ops
from miles_ext_kernels.cake._runtime import loader
from miles_ext_kernels.cake._vendor.cake_gqa_nondet_train import cake_build

loaded = []
load_library = cake_build.load_library


def recording_load_library(name):
    loaded.append(name)
    return load_library(name)


cake_build.load_library = recording_load_library
paths = cake.prebuild("gqa")
g = torch.Generator().manual_seed(0)
cu = torch.tensor([0, 300, 811], dtype=torch.int32, device="cuda")
q, k, v = (torch.randn(811, h, 256, generator=g).bfloat16().cuda().requires_grad_() for h in (8, 2, 2))
with loader.no_build():
    out = gqa.varlen_gqa_attention(q, k, v, cu, cu, max_seqlen_k=511)
    out.backward(torch.randn_like(out))
torch.cuda.synchronize()
expected = ops.libraries(cake.device_arch())
assert sorted(loaded) == sorted(expected), (loaded, expected)
assert sorted(path.name.rsplit("_", 1)[0] for path in paths) == sorted(
    f"{name}_{cake_build.libraries_by_name()[name]['closure_sha256'][:20]}" for name in expected
), paths
flashinfer = sorted(n for n in sys.modules if n == "flashinfer" or n.startswith("flashinfer."))
assert flashinfer == [], flashinfer
print("loaded", len(loaded))
"""


def test_a_fresh_process_loads_only_the_prebuilt_libraries_and_never_flashinfer():
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert result.returncode == 0, result.stderr[-5000:]
    assert result.stdout.split() == ["loaded", "4"]
