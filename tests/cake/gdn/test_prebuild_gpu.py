"""The prebuild contract of gdn (README, Prebuild): after one ``prebuild("gdn")`` no call compiles (the FLA context
here has no process group, so FLA's own Triton pre-processing, which ``prebuild`` does not prewarm, never runs).

``ROUTES`` holds one call per call-time path, forward and backward: without context parallelism, with a ``KcpContext``
and with an FLA context; host and device boundaries; one and several sequences, and a sequence longer than 128 chunks;
an initial and a final state; the recomputed backward; the in-kernel L2 norm off with an FP32 beta; an equal-length
batch; a call without a token; a side stream. ``check_prebuild`` runs every route under each variant. gdn has no model
parameter: ``prebuild`` builds every library of the op's five packages for the device's architecture. Needs an SM100a /
SM103a / SM107a GPU and nvcc; skips elsewhere.
"""

import itertools
import sys
import types

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._runtime import loader, workspace  # noqa: E402

from ..helpers import POOLED, check_prebuild, no_build  # noqa: E402

if tuple(torch.cuda.get_device_capability()) not in loader.ARCH_BY_CAPABILITY:
    pytest.skip("the GDN kernels need SM100a / SM103a / SM107a", allow_module_level=True)

HEAD_DIM = 128
HEADS = (4, 12)  # key heads, value heads
LAYOUT = (300, 517, 64, 7)  # several sequences, lengths not multiples of the 64-token chunk
LONG = (9000,)  # one sequence of 141 chunks: longer than one pre-scan CTA covers


def _cu(seq_lens) -> tuple[int, ...]:
    return tuple(itertools.accumulate(seq_lens, initial=0))


def _call(
    variant,
    seq_lens=LAYOUT,
    *,
    route="host",
    cp=None,
    state=False,
    recompute=False,
    l2norm=True,
    beta_dtype=torch.bfloat16,
    batch=None,
    **options,
):
    """One forward and backward of ``chunk_gated_delta_rule`` under ``variant``.

    ``route``: ``"host"`` passes ``cu_seqlens_cpu``, ``"device"`` the device boundaries only. ``cp``: ``"kcp"`` a
    one-rank ``KcpContext``, ``"fla"`` an FLA context without a process group. ``batch``: an equal-length batch of
    that many sequences without boundaries. ``options``: more configuration (e.g. ``POOLED``).
    """
    from miles_ext_kernels.cake.gdn import build_kcp_context, chunk_gated_delta_rule

    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=recompute, **options)
    num_heads, num_v_heads = HEADS
    gen = torch.Generator().manual_seed(0)
    rows, tokens = (batch, seq_lens[0]) if batch else (1, sum(seq_lens))

    def leaf(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(rows, tokens, *shape, generator=gen) * scale).to("cuda", dtype).requires_grad_(True)

    q, k = leaf(num_heads, HEAD_DIM), leaf(num_heads, HEAD_DIM)
    v = leaf(num_v_heads, HEAD_DIM)
    g = (-torch.rand(rows, tokens, num_v_heads, generator=gen) * 0.5).cuda().requires_grad_(True)
    beta = torch.rand(rows, tokens, num_v_heads, generator=gen).to("cuda", beta_dtype).requires_grad_(True)
    kwargs = {"use_qk_l2norm_in_kernel": l2norm}
    cu = _cu(seq_lens)
    if not batch:
        kwargs["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device="cuda")
        if route == "host":
            kwargs["cu_seqlens_cpu"] = cu
    if cp == "kcp":
        kwargs["cp_context"] = build_kcp_context(cu, None, device=torch.device("cuda", torch.cuda.current_device()))
    elif cp == "fla":
        kwargs["cp_context"] = types.SimpleNamespace(
            group=None, cu_seqlens=kwargs["cu_seqlens"], cu_seqlens_cpu=cu if route == "host" else None
        )
    leaves = [q, k, v, g, beta]
    if state:
        h0 = torch.randn(len(seq_lens), num_v_heads, HEAD_DIM, HEAD_DIM, generator=gen) * 0.1
        h0 = h0.cuda().requires_grad_(True)
        kwargs.update(initial_state=h0, output_final_state=True)
        leaves.append(h0)
    o, final_state = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **kwargs)
    loss = o.float().square().sum() + (final_state.square().sum() if state else 0.0)
    loss.backward()
    torch.cuda.synchronize()
    assert all(t.grad is not None and t.grad.shape == t.shape for t in leaves)


def _on_a_side_stream(variant):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _call(variant)
        _call(variant, cp="fla")
    torch.cuda.current_stream().wait_stream(side)
    _call(variant)


ROUTES = {
    "plain_host": lambda variant: _call(variant),
    "plain_device": lambda variant: _call(variant, route="device"),
    "plain_one_sequence": lambda variant: _call(variant, (777,)),
    "plain_long_host": lambda variant: _call(variant, LONG),
    "plain_long_device": lambda variant: _call(variant, LONG, route="device"),
    "plain_state": lambda variant: _call(variant, state=True),
    "plain_state_device": lambda variant: _call(variant, route="device", state=True),
    "plain_recompute": lambda variant: _call(variant, recompute=True),
    "plain_no_l2norm_fp32_beta": lambda variant: _call(variant, l2norm=False, beta_dtype=torch.float32),
    "plain_batch": lambda variant: _call(variant, (256,), batch=2),
    "plain_no_token": lambda variant: _call(variant, (0, 0), state=True),
    "kcp": lambda variant: _call(variant, cp="kcp"),
    "kcp_long": lambda variant: _call(variant, LONG, cp="kcp"),
    "kcp_recompute": lambda variant: _call(variant, cp="kcp", recompute=True),
    "kcp_no_l2norm_fp32_beta": lambda variant: _call(variant, cp="kcp", l2norm=False, beta_dtype=torch.float32),
    "fla_cp_host": lambda variant: _call(variant, cp="fla"),
    "fla_cp_device": lambda variant: _call(variant, cp="fla", route="device"),
    "fla_cp_long_host": lambda variant: _call(variant, LONG, cp="fla"),
    "fla_cp_long_device": lambda variant: _call(variant, LONG, cp="fla", route="device"),
    "fla_cp_recompute": lambda variant: _call(variant, cp="fla", recompute=True),
    "fla_cp_no_l2norm_fp32_beta": lambda variant: _call(variant, cp="fla", l2norm=False, beta_dtype=torch.float32),
    "fla_cp_no_token": lambda variant: _call(variant, (0, 0), cp="fla"),
    "side_stream": _on_a_side_stream,
}


def test_after_one_prebuild_no_route_compiles():
    check_prebuild("gdn", ROUTES)

    assert not [name for name in sys.modules if name == "flashinfer" or name.startswith("flashinfer.")]


def test_a_call_runs_the_variant_configured_when_it_is_made(monkeypatch):
    # configure det and call, then configure fast and call: the second call runs the fast package, and neither compiles
    from miles_ext_kernels.cake._impl.gdn import ops

    cake.prebuild("gdn")
    ran = []
    for variant, entry in (("det", "chunk_gated_delta_rule_det"), ("fast", "chunk_gated_delta_rule")):
        package = ops._package(variant)

        def counted(*args, _variant=variant, _entry=getattr(package, entry), **kwargs):
            ran.append(_variant)
            return _entry(*args, **kwargs)

        monkeypatch.setattr(package, entry, counted)
    with no_build():
        _call("det")
        _call("fast")
    assert ran == ["det", "fast"]


def test_prebuild_returns_every_library_of_the_architecture_and_is_idempotent():
    arch = loader.device_arch()
    expected = 0
    for package in cake.op_spec("gdn").packages:
        libraries = loader.load_package(package).cake_build.libraries_by_name()
        expected += sum(1 for entry in libraries.values() if entry["arch"] == arch)

    paths = cake.prebuild("gdn")
    assert len(paths) == len(set(paths)) == expected
    assert all(path.is_file() for path in paths)
    assert cake.prebuild("gdn") == paths


def test_a_det_call_keeps_its_scratch_in_the_bounded_pool():
    cake.prebuild("gdn")
    cake.release()
    _call("det", LONG, **POOLED)

    held = workspace.stats()[torch.cuda.current_device()]
    assert held["entries"] > 0 and 0 < held["bytes"] <= cake.option(None, "cake_workspace_pool_bytes")
    assert cake.release("gdn") == held["bytes"]
