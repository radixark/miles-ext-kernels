"""Host-side behaviour of the gdn entry points that needs no GPU (CPU: needs torch and tvm_ffi).

The state-scan planner of the det kernels with FLA's context parallelism never cuts a segment longer than one pre-scan
CTA covers (one chunk per thread, at most 128 chunks), so a long single sequence keeps the serial scan instead of
launching a split the pre-scan cannot serve. A call without a token returns an empty output under either variant, its
final state is the initial state (zeros without one), and its gradients are zeros and the final state's gradient. Host
boundaries that end before the call's tokens are rejected before any CUDA call. The 0-element state placeholder of a
launch without an initial or final state is one tensor per device, shape and dtype, not an allocation per call; every
other scratch take of the det and fast packages goes through the workspace pool. ``release`` drops the fast
package's self-zeroing buffers and the recorded backward plans that hold them.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.gdn import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import workspace  # noqa: E402

HEAD_DIM = 128
SM_COUNTS = (148, 152, 160)  # B200, the GB300 nodes the tests run on, a full GB300 die
ARCHES = ("sm_100a", "sm_103a", "sm_107a")


# --------------------------------------------------------------------------------------------------------------------
# the planner's segment cap


def test_one_pre_scan_cta_covers_at_most_128_chunks():
    assert ops.PRESCAN_MAX_CHUNKS == 128
    for kind in ("fwd_h", "dhu"):
        for config in ops.PRESCAN_CONFIGS:
            for arch in ARCHES:
                assert 0 < ops._prescan_capacity(kind, config, arch) <= ops.PRESCAN_MAX_CHUNKS, (kind, config, arch)
    # a configuration without a pre-scan stage has no capacity
    assert ops._prescan_capacity("fwd_h", (32, 1), "sm_100a") == 0


def test_without_the_cap_a_long_single_sequence_would_cut_segments_the_pre_scan_cannot_serve():
    # 131072 tokens (2048 chunks) in one sequence, 32 value heads at (32, 2): four segments fill the CTA limit, so the
    # cheapest split cuts 512-chunk segments; within the cap no split fits the limit and the serial scan runs
    chunks, per_segment = (2048,), 32 * (HEAD_DIM // 32)
    uncapped, _ = ops._best_split(chunks, per_segment, 148, 2, 1.0, max_segment_len=math.inf)
    assert uncapped > ops.PRESCAN_MAX_CHUNKS

    assert ops._best_split(chunks, per_segment, 148, 2, 1.0) == (0, math.inf)
    assert ops.plan_segments(2049, 1, per_segment, 148, 2, chunks)[0] == 0


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("num_v_heads", [4, 12, 16, 32, 48])
@pytest.mark.parametrize("tokens", [16384, 32768, 65536, 131072, 262144])
def test_no_plan_cuts_a_segment_longer_than_the_cap(tokens, num_v_heads, sm_count):
    num_chunks = -(-tokens // ops.CHUNK)
    layouts = [(num_chunks,), (num_chunks, 3), (num_chunks // 2, num_chunks // 2, 1)]
    for chunks in layouts:
        for config in ops.PRESCAN_CONFIGS:
            per_segment = num_v_heads * (HEAD_DIM // config[0])
            for residency in (1, 2):
                factor = ops.SCAN_STEP_FACTOR[config]
                seg_len, _ = ops._best_split(chunks, per_segment, sm_count, residency, factor)
                assert seg_len <= ops.PRESCAN_MAX_CHUNKS, (chunks, config, residency)
                seg_len, _ = ops.plan_segments(
                    sum(chunks) + len(chunks), len(chunks), per_segment, sm_count, residency, chunks, factor
                )
                assert seg_len <= ops.PRESCAN_MAX_CHUNKS, (chunks, config, residency)
                # the device route prices the chunk bound
                bound = sum(chunks) + len(chunks)
                seg_len, _ = ops.plan_segments(bound, len(chunks), per_segment, sm_count, residency)
                assert seg_len <= ops.PRESCAN_MAX_CHUNKS, (chunks, config, residency)


def test_a_lower_capacity_bounds_the_split():
    chunks, per_segment = (300,), 4 * (HEAD_DIM // 32)
    seg_len, _ = ops._best_split(chunks, per_segment, 148, 2, 1.0)
    assert 0 < seg_len <= ops.PRESCAN_MAX_CHUNKS

    for cap in (64, 16):
        capped, _ = ops._best_split(chunks, per_segment, 148, 2, 1.0, max_segment_len=cap)
        assert 0 < capped <= cap


# --------------------------------------------------------------------------------------------------------------------
# host boundaries


@pytest.mark.parametrize("host", ["tuple", "tensor"])
def test_host_boundaries_must_end_at_the_tokens_before_any_cuda_call(host, monkeypatch):
    # tokens past the last offset would never be written; the check runs on the host copy alone, before the tables are
    # built on the device: any CUDA initialisation here fails the test (it also runs where CUDA is absent or hidden)
    def no_cuda(*args, **kwargs):
        raise AssertionError("the boundary check made a CUDA call")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    cu_seqlens_cpu = (0, 100, 127) if host == "tuple" else torch.tensor([0, 100, 127])

    with pytest.raises(ValueError, match="must end at the 128 tokens of q; it ends at 127"):
        ops._chunk_meta(torch.tensor([0, 100, 127]), cu_seqlens_cpu, 1, 128, torch.device("cuda", 0), "sm_100a", 4)


# --------------------------------------------------------------------------------------------------------------------
# a call without a token


def _empty(heads=(2, 4), num_seqs=2):
    num_heads, num_v_heads = heads
    leaves = {
        "q": torch.zeros(1, 0, num_heads, HEAD_DIM, dtype=torch.bfloat16),
        "k": torch.zeros(1, 0, num_heads, HEAD_DIM, dtype=torch.bfloat16),
        "v": torch.zeros(1, 0, num_v_heads, HEAD_DIM, dtype=torch.bfloat16),
        "g": torch.zeros(1, 0, num_v_heads),
        "beta": torch.zeros(1, 0, num_v_heads, dtype=torch.bfloat16),
    }
    return {name: tensor.requires_grad_(True) for name, tensor in leaves.items()}


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_call_without_tokens_passes_the_state_through(variant, state, recompute):
    cake.configure(cake_gdn_variant=variant, cake_gdn_recompute_state=recompute)
    leaves = _empty()
    num_v_heads = leaves["v"].shape[2]
    gen = torch.Generator().manual_seed(0)
    h0 = torch.randn(2, num_v_heads, HEAD_DIM, HEAD_DIM, generator=gen).requires_grad_(True) if state else None
    dht = torch.randn(2, num_v_heads, HEAD_DIM, HEAD_DIM, generator=gen)

    o, final_state = ops.chunk_gated_delta_rule(
        **leaves,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 0, 0], dtype=torch.int32),
        cu_seqlens_cpu=(0, 0, 0),
        use_qk_l2norm_in_kernel=True,
    )

    assert o.shape == leaves["v"].shape and o.dtype == torch.bfloat16
    assert final_state.dtype == torch.float32 and final_state.shape == dht.shape
    want = h0.detach() if state else torch.zeros_like(dht)
    assert torch.equal(final_state.detach(), want)
    torch.autograd.backward([o, final_state], [torch.zeros_like(o), dht])
    for name, leaf in leaves.items():
        assert leaf.grad is not None and leaf.grad.shape == leaf.shape and leaf.grad.dtype == leaf.dtype, name
    if state:
        assert torch.equal(h0.grad, dht)


@pytest.mark.parametrize("variant", ["det", "fast"])
def test_a_call_without_tokens_and_without_a_final_state(variant):
    cake.configure(cake_gdn_variant=variant)
    leaves = _empty()

    o, final_state = ops.chunk_gated_delta_rule(**leaves, cu_seqlens=torch.tensor([0, 0], dtype=torch.int32))

    assert final_state is None and o.shape == leaves["v"].shape
    o.sum().backward()
    assert all(leaf.grad is not None for leaf in leaves.values())


def test_a_call_without_tokens_checks_the_initial_state():
    leaves = _empty()

    with pytest.raises(ValueError, match="one state per sequence"):
        ops.chunk_gated_delta_rule(
            **leaves,
            initial_state=torch.zeros(3, 4, HEAD_DIM, HEAD_DIM),
            output_final_state=True,
            cu_seqlens=torch.tensor([0, 0, 0], dtype=torch.int32),
        )


# --------------------------------------------------------------------------------------------------------------------
# scratch


def test_the_state_placeholder_is_one_tensor_per_device_shape_and_dtype():
    # a launch without an initial or final state passes a 0-element state the kernels never touch: it holds no memory,
    # so every call on any stream can pass the same tensor instead of allocating one
    cpu = torch.device("cpu")
    shape = (0, 12, HEAD_DIM, HEAD_DIM)
    first = ops._no_state(shape, torch.float32, cpu)
    assert first.shape == shape and first.dtype == torch.float32 and first.numel() == 0
    assert ops._no_state(shape, torch.float32, cpu) is first
    assert ops._no_state((0, 4, HEAD_DIM, HEAD_DIM), torch.float32, cpu) is not first
    with pytest.raises(ValueError, match="holds no element"):
        ops._no_state((1, 12, HEAD_DIM, HEAD_DIM), torch.float32, cpu)
    cake.release("gdn")  # release() drops the placeholders with the other host caches
    assert ops._no_state(shape, torch.float32, cpu) is not first


@pytest.mark.parametrize("package", ["det", "fast"])
def test_the_packages_take_their_scratch_from_the_pool_and_share_the_state_placeholders(package, monkeypatch):
    cpu = torch.device("cpu")
    scratch = ops._package(package).cake_chain._POOLED
    zero = (0, 12, HEAD_DIM, HEAD_DIM)
    placeholder = ops._no_state(zero, torch.float32, cpu)
    made = []
    empty = torch.empty

    def counted(*args, **kwargs):
        made.append(args)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", counted)
    for _ in range(2):
        for role in ("f_zero_state", "b_zero_state"):
            assert scratch.take(None, role, zero, torch.float32, cpu) is placeholder
    assert made == [], "a state placeholder was allocated"
    # at the default bound, 0, any other role is a fresh buffer of the pool's: a take that misses, nothing kept
    misses = workspace.stats().get("cpu", {"misses": 0})["misses"]
    shape = (5, 12, HEAD_DIM, HEAD_DIM)
    dh = scratch.take(None, "b_dh", shape, torch.bfloat16, cpu)
    again = scratch.take(None, "b_dh", shape, torch.bfloat16, cpu)
    assert dh.shape == again.shape == shape and dh.dtype == torch.bfloat16
    assert dh.data_ptr() != again.data_ptr() and len(made) == 2
    stats = workspace.stats()["cpu"]
    assert stats["misses"] == misses + 2 and (stats["bytes"], stats["entries"]) == (0, 0)


def test_release_drops_the_self_zeroing_buffers_and_the_backward_plans_of_the_fast_package(monkeypatch):
    # the fast package keeps self-zeroing buffers per (device, stream) and recorded backward plans that hold them; the
    # det package keeps neither, so a release that skipped one of them would keep device memory alive
    fast = ops._package("fast").cake_chain
    det = ops._package("det").cake_chain
    assert not hasattr(det._POOLED, "zeroed") and not hasattr(det, "_BWD_PLANS")
    monkeypatch.setitem(fast._POOLED.zeroed, ("stand-in", 0), {"role": torch.zeros(1)})
    monkeypatch.setitem(fast._BWD_PLANS, ("stand-in",), object())

    cake.release("gdn")

    assert fast._POOLED.zeroed == {} and fast._BWD_PLANS == {}
