"""``det`` of the dense projection GEMMs repeats bit for bit, and what it promises across token counts.

Run to run: the same inputs give the same bits twice in one process, on the default stream and on a side stream in
turn, and once more in a fresh process. Across token counts: the results are not batch-invariant (the plan depends on
the token count), so the tokens shared by calls of 1 to 16384 tokens are only held to the accuracy budget against
FP64 and, for the router, to the same top-k experts on the rows without a near tie; the number of elements whose bits
differ from the largest call is recorded, not asserted. Needs a CUDA device; the tests need SM100a / SM103a / SM107a
and skip elsewhere.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402

from ..helpers import assert_bitwise, assert_rel_rms, assert_repeatable  # noqa: E402

if torch.cuda.get_device_capability() not in cake.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake dense projection GEMMs need SM100a / SM103a / SM107a", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[3]

# The cases, their inputs and the digest of their results, as source: this process and a fresh one run the same code.
_CASES = """
import hashlib

import torch

from miles_ext_kernels.cake.dense_gemm import batched_projection, router_gating

# name -> (kind, shape, tokens): a projection's shape is (heads, d_in, d_out, weight layout), a router's
# (experts, hidden)
CASES = {
    "glm5-q-absorb-h16-t1001": ("projection", (16, 192, 512, "in_out"), 1001),
    "glm5-value-h16-t239": ("projection", (16, 512, 256, "out_in"), 239),
    "dsv4-o-group-g8-t2049": ("projection", (8, 4096, 1024, "out_in"), 2049),
    "dense-4096-32-t16384": ("projection", (1, 4096, 32, "out_in"), 16384),
    "router-e256-h5120-t16384": ("router", (256, 5120), 16384),
    "router-e384-h7168-t1001": ("router", (384, 7168), 1001),
}


def leaves(kind, shape, tokens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "router":
        experts, hidden = shape
        x = torch.randn(tokens, 1, hidden, generator=gen, device="cuda").to(torch.bfloat16)
        weight = (torch.randn(experts, hidden, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
        return x, weight, torch.randn(tokens, 1, experts, generator=gen, device="cuda") * 1e-3
    heads, d_in, d_out, layout = shape
    w_shape = (heads, d_in, d_out) if layout == "in_out" else (heads, d_out, d_in)
    x = torch.randn(tokens, heads, d_in, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(*w_shape, generator=gen, device="cuda") * d_in**-0.5).to(torch.bfloat16)
    return x, weight, torch.randn(tokens, heads, d_out, generator=gen, device="cuda").to(torch.bfloat16)


def step(kind, shape, x, weight, grad):
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    out = router_gating(x, weight) if kind == "router" else batched_projection(x, weight, weight_layout=shape[3])
    out.backward(grad)
    return {"output": out.detach(), "input gradient": x.grad, "weight gradient": weight.grad}


def run_all():
    # every case's output and gradients from fresh inputs, on the current stream, copied to the host
    results = {}
    for index, (name, (kind, shape, tokens)) in enumerate(CASES.items()):
        for what, tensor in step(kind, shape, *leaves(kind, shape, tokens, 20261105 + index)).items():
            results[f"{name} {what}"] = tensor.cpu()
    return results


def digest(results):
    sha = hashlib.sha256()
    for name in sorted(results):
        sha.update(name.encode())
        sha.update(results[name].contiguous().view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()
"""


@pytest.fixture(scope="module")
def cases():
    namespace = {}
    exec(_CASES, namespace)
    return namespace


def test_det_repeats_bit_for_bit_in_one_process_and_in_a_fresh_one(cases):
    cake.configure(cake_dense_gemm_variant="det")
    results = assert_repeatable(cases["run_all"])  # twice in this process

    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    fresh = subprocess.run(
        [sys.executable, "-c", _CASES + "\nprint(digest(run_all()))\n"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )

    assert fresh.returncode == 0, fresh.stderr[-4000:]
    assert fresh.stdout.split()[-1] == cases["digest"](results)


def test_det_gives_the_same_bits_on_the_default_stream_and_a_side_stream_in_turn(cases):
    cake.configure(cake_dense_gemm_variant="det")
    side = torch.cuda.Stream()
    results = []
    for on_side in (False, True, False, True):
        if on_side:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                results.append(cases["run_all"]())
            side.synchronize()
        else:
            results.append(cases["run_all"]())

    for other in results[1:]:
        assert_bitwise(results[0], other)


# Token counts of calls over the first tokens of one batch: the tile and split boundaries of the plans lie between
POOL_TOKENS = (1, 33, 257, 2049, 16384)
# Relative RMS budgets against FP64 (see test_batched_projection_gpu.py and test_router_gpu.py): one BF16 rounding of
# an FP32 sum is about 1.1e-3 to 1.6e-3 relative RMS, the FP32 logits are within the package's 5e-5.
BUDGETS = {
    "projection": {"output": 3e-3, "input gradient": 3e-3, "weight gradient": 3e-3},
    "router": {"output": 5e-5, "input gradient": 3e-3, "weight gradient": 3e-3},
}
TOP_K = 8
# Rows whose k-th and (k+1)-th FP64 logits are closer than this (relative to the largest logit) may order either way.
TOP_K_TIE_MARGIN = 1e-4
POOL_CASES = {
    "glm5-q-absorb-h16": ("projection", (16, 192, 512, "in_out")),
    "dsv4-o-group-g8": ("projection", (8, 4096, 1024, "out_in")),
    "dense-4096-32": ("projection", (1, 4096, 32, "out_in")),
    "router-e256-h5120": ("router", (256, 5120)),
    "router-e384-h7168": ("router", (384, 7168)),
}


def _reference(kind, shape, x, weight, grad):
    x, weight, grad = x.double(), weight.double(), grad.double()
    if kind == "router":
        x2, g = x.reshape(-1, x.shape[-1]), grad.reshape(-1, grad.shape[-1])
        out = (x2 @ weight.T).view(grad.shape)
        return {"output": out, "input gradient": (g @ weight).view(x.shape), "weight gradient": g.T @ x2}
    equation = "thd,hdm->thm" if shape[3] == "in_out" else "thm,hdm->thd"
    x.requires_grad_(True)
    weight.requires_grad_(True)
    out = torch.einsum(equation, x, weight)
    out.backward(grad)
    return {"output": out.detach(), "input gradient": x.grad, "weight gradient": weight.grad}


def _differing(got, want) -> int:
    """The number of elements whose bits differ."""
    integer = {2: torch.int16, 4: torch.int32}[got.element_size()]
    return int((got.contiguous().view(integer) != want.contiguous().view(integer)).sum())


@pytest.mark.parametrize("name", list(POOL_CASES))
def test_det_is_not_batch_invariant_but_every_token_count_is_within_budget(cases, name, record_property):
    cake.configure(cake_dense_gemm_variant="det")
    kind, shape = POOL_CASES[name]
    x, weight, grad = cases["leaves"](kind, shape, max(POOL_TOKENS), 20261105)
    per_count, measured = {}, {}
    for tokens in POOL_TOKENS:
        leaves = (x[:tokens], weight, grad[:tokens])
        results = per_count[tokens] = assert_repeatable(lambda leaves=leaves: cases["step"](kind, shape, *leaves))
        reference = _reference(kind, shape, *leaves)
        measured[tokens] = assert_rel_rms(results, reference, BUDGETS[kind])
        if kind == "router":
            logits, expected = results["output"].reshape(tokens, -1).double(), reference["output"].reshape(tokens, -1)
            ranked = torch.topk(expected, TOP_K + 1, dim=1).values
            clear = (ranked[:, TOP_K - 1] - ranked[:, TOP_K]) > TOP_K_TIE_MARGIN * float(expected.abs().max())
            top = torch.topk(logits, TOP_K, dim=1).indices.sort(dim=1).values
            want = torch.topk(expected, TOP_K, dim=1).indices.sort(dim=1).values
            assert torch.equal(top[clear], want[clear]), tokens

    # recorded, not asserted: the elements of the shared tokens whose bits differ from the call of all the tokens
    largest = per_count[max(POOL_TOKENS)]
    differing = {
        f"T={tokens} {what}": _differing(results[what], largest[what][:tokens])
        for tokens, results in per_count.items()
        for what in ("output", "input gradient")
    }
    record_property("bits_differing_from_the_largest_call", differing)
    record_property("rel_rms_per_token_count", measured)
    print(name, differing)
