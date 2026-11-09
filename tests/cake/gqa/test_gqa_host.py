"""The host side of the gqa entry point, on the CPU: what it checks before any launch, the opt-in check of the
boundaries and ``max_seqlen_k``, the pooled backward scratch and the prebuild errors.

Needs torch and tvm_ffi (the implementation imports them); no GPU. CPU tensors reach the kernels' own operand check,
which refuses them, so a call that gets that far shows that the checks before it passed.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import gqa  # noqa: E402
from miles_ext_kernels.cake._impl.gqa import ops  # noqa: E402
from miles_ext_kernels.cake._runtime import config, loader, workspace  # noqa: E402
from miles_ext_kernels.cake._vendor.cake_gqa_nondet_train import cake_jit  # noqa: E402

NO_DET = "the cake gqa kernels have no det variant"
REACHED_THE_KERNELS = "must be on q's CUDA device"
# a micro-batch of three segments: 1000, 517 and 2049 keys
LENS = (1000, 517, 2049)
LONGEST = max(LENS)


def _cu(lens) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)


def _operands(lens=LENS, h_q=8, h_kv=2):
    total = sum(lens)
    q = torch.zeros(total, h_q, 256, dtype=torch.bfloat16)
    k = torch.zeros(total, h_kv, 256, dtype=torch.bfloat16)
    return q, k, k.clone(), _cu(lens), _cu(lens)


def test_a_process_configured_for_deterministic_mode_rejects_the_call_when_it_runs():
    cake.configure(deterministic_mode=True)
    with pytest.raises(ValueError, match=NO_DET):
        gqa.varlen_gqa_attention(*_operands(), max_seqlen_k=LONGEST)

    cake.configure()  # read at the call: the next call goes on to the kernels' operand check
    with pytest.raises(ValueError, match=REACHED_THE_KERNELS):
        gqa.varlen_gqa_attention(*_operands(), max_seqlen_k=LONGEST)


def test_max_seqlen_k_is_a_required_host_int():
    with pytest.raises(TypeError, match="max_seqlen_k"):
        gqa.varlen_gqa_attention(*_operands())
    for value in (None, 2049.0, True, torch.tensor(LONGEST)):
        with pytest.raises(TypeError, match="max_seqlen_k must be a host int"):
            gqa.varlen_gqa_attention(*_operands(), max_seqlen_k=value)
    with pytest.raises(ValueError, match="max_seqlen_k must be >= 0"):
        gqa.varlen_gqa_attention(*_operands(), max_seqlen_k=-1)


def test_the_check_accepts_the_contract():
    q, k, _, cu_q, cu_k = _operands()

    for bound in (LONGEST, LONGEST + 1, sum(LENS)):
        ops.check_seqlens(cu_q, cu_k, q.shape[0], k.shape[0], bound)
    ops.check_seqlens(_cu(()), _cu(()), 0, 0, 0)  # no segment
    ops.check_seqlens(_cu((0, 5, 0)), _cu((7, 0, 3)), 5, 10, 7)  # empty segments, keys without queries


@pytest.mark.parametrize("bound", [0, 1024, 1920, LONGEST - 1])
def test_the_check_rejects_a_max_seqlen_k_below_the_longest_segment(bound):
    q, k, _, cu_q, cu_k = _operands()

    with pytest.raises(ValueError, match=f"max_seqlen_k is {bound}, but segment 2 has 2049 keys: the backward"):
        ops.check_seqlens(cu_q, cu_k, q.shape[0], k.shape[0], bound)


@pytest.mark.parametrize(
    "which, bounds, total, message",
    [
        ("q", [1, 1000, 1517, 3566], 3566, "cu_seqlens_q must start at 0"),
        ("k", [0, 1000, 900, 3566], 3566, "cu_seqlens_k must start at 0, never decrease"),
        ("k", [0, 1000, 1517, 3565], 3566, r"cu_seqlens_k .* end at 3566; got \[0, 1000, 1517, 3565\]"),
        ("q", [], 0, r"cu_seqlens_q .*; got \[\]"),
    ],
)
def test_the_check_rejects_malformed_boundaries(which, bounds, total, message):
    good = _cu(LENS)
    cu = torch.tensor(bounds, dtype=torch.int32)
    cu_q, cu_k = (cu, good) if which == "q" else (good, cu)

    with pytest.raises(ValueError, match=message):
        ops.check_seqlens(cu_q, cu_k, total if which == "q" else sum(LENS), total if which == "k" else sum(LENS), 4096)


def test_the_option_checks_before_any_launch_and_is_off_by_default():
    too_small = LONGEST - 1
    with pytest.raises(ValueError, match=REACHED_THE_KERNELS):
        gqa.varlen_gqa_attention(*_operands(), max_seqlen_k=too_small)

    cake.configure(cake_gqa_check_seqlens=True)
    with pytest.raises(ValueError, match="but segment 2 has 2049 keys"):
        gqa.varlen_gqa_attention(*_operands(), max_seqlen_k=too_small)
    with pytest.raises(ValueError, match=REACHED_THE_KERNELS):
        gqa.varlen_gqa_attention(*_operands(), max_seqlen_k=LONGEST)


def test_the_backward_scratch_is_one_pooled_buffer_in_the_vendored_layout(pooled):
    total_q, heads, segments = 3566, 8, 3
    hits = workspace.stats().get("cpu", {}).get("hits", 0)
    views = ops.PoolWorkspace(torch.device("cpu")).scratch(total_q, heads, segments)
    layout = cake_jit.scratch_layout(total_q, heads, segments)
    nbytes = cake_jit.scratch_bytes(total_q, heads, segments)

    assert [tuple(view.shape) for view in views] == [shape for _, shape in layout.values()]
    assert all(view.dtype == torch.float32 and view.is_contiguous() for view in views)
    base = views[0].data_ptr()
    assert [view.data_ptr() - base for view in views] == [offset for offset, _ in layout.values()]
    assert workspace.stats()["cpu"]["bytes"] == nbytes and workspace.stats()["cpu"]["entries"] == 1
    # the same shapes again reuse the buffer; release() frees it
    again = ops.PoolWorkspace(torch.device("cpu")).scratch(total_q, heads, segments)
    assert again[0].data_ptr() == base and workspace.stats()["cpu"]["hits"] == hits + 1
    assert cake.release() == nbytes and workspace.stats()["cpu"]["bytes"] == 0


def test_at_the_default_bound_of_zero_the_backward_scratch_is_not_kept_and_release_frees_nothing():
    """Never configured, the bound is 0: every backward takes a scratch buffer of its own, in the vendored layout, and
    the pool keeps none of them, so ``release`` frees nothing."""
    assert config.option(None, "cake_workspace_pool_bytes") == 0
    total_q, heads, segments = 3566, 8, 3
    before = workspace.stats().get("cpu", {"hits": 0, "misses": 0})
    layout = cake_jit.scratch_layout(total_q, heads, segments)
    for _ in range(2):
        views = ops.PoolWorkspace(torch.device("cpu")).scratch(total_q, heads, segments)
        assert [tuple(view.shape) for view in views] == [shape for _, shape in layout.values()]
    after = workspace.stats()["cpu"]
    assert (after["bytes"], after["entries"]) == (0, 0)
    assert (after["hits"], after["misses"]) == (before["hits"], before["misses"] + 2)
    assert cake.release() == 0


def test_prebuild_takes_no_model_parameter_and_rejects_another_gpu(monkeypatch):
    with pytest.raises(TypeError):
        cake.prebuild("gqa", num_heads=16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(NotImplementedError, match="SM100a, SM103a or SM107a, not compute capability 9.0"):
        cake.prebuild("gqa")


def test_prebuild_refuses_a_toolkit_too_old_for_sm107a(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    cuda_13_0 = loader.CudaVersion(13, 0, "13.0.88")
    monkeypatch.setattr(loader, "_toolkit", lambda nvcc=None: ("/usr/local/cuda/bin/nvcc", cuda_13_0))

    with pytest.raises(RuntimeError, match=r"13\.4"):
        cake.prebuild("gqa")
