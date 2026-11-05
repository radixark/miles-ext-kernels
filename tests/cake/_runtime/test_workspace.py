"""The bounded workspace pool, on CPU tensors (needs torch); its CUDA path also on a fake CUDA device layer, and
streams and graph capture on a CUDA device."""

import random
import threading

import pytest

from miles_ext_kernels import cake
from miles_ext_kernels.cake._runtime import config, workspace

from ..helpers import POOLED, requires_cuda
from .fake_cuda import FakeCuda, FakeStream

torch = pytest.importorskip("torch")

KIB = 1024


@pytest.fixture(autouse=True)
def fresh_pool(monkeypatch):
    for name in ("_pools", "_held", "_counters", "_devices", "_cuda_devices"):
        monkeypatch.setattr(workspace, name, {})
    monkeypatch.setattr(workspace, "_bound", (None, 0))
    monkeypatch.setattr(workspace, "_direct", None)


@pytest.fixture
def place(monkeypatch):
    """The CPU device with a stream and a capture state the test sets (``place.stream``, ``place.capturing``)."""
    state = type("Place", (), {"stream": None, "capturing": False})()
    monkeypatch.setattr(workspace, "_stream", lambda where: (state.stream, state.capturing))
    return state


@pytest.fixture
def cuda(monkeypatch):
    """Two fake CUDA devices whose buffers live on the CPU (:mod:`.fake_cuda`)."""
    fake = FakeCuda(torch)
    for owner, name, value in fake.patches():
        monkeypatch.setattr(owner, name, value, raising=False)  # torch._C._cuda_* is absent from a CPU build
    return fake


def _take(op, role, numel, dtype=torch.float32, device="cpu"):
    return workspace.take(op, role, (numel,), dtype, device)


def _check_total():
    """The running byte total of every device equals the bytes of its buffers."""
    for where, pool in workspace._pools.items():
        assert workspace._held[where] == sum(buffer.numel() for buffer in pool.values()), where
        assert workspace.stats()[where]["bytes"] == workspace._held[where]


def test_the_default_bound_keeps_nothing_and_reads_no_stream(monkeypatch):
    monkeypatch.setattr(workspace, "_stream", lambda where: pytest.fail("a take without a pool read the stream"))
    first = _take("a", "x", 256)
    second = _take("a", "x", 256)

    assert cake.option(None, "cake_workspace_pool_bytes") == 0
    assert first.shape == (256,) and first.dtype == torch.float32
    assert second.data_ptr() != first.data_ptr()
    assert workspace.stats() == {"cpu": {"bytes": 0, "entries": 0, "hits": 0, "misses": 2, "evictions": 0}}


def test_the_bound_is_read_once_per_configuration(monkeypatch):
    reads = []
    option = config.option

    def counted(op, dest):
        reads.append((op, dest))
        return option(op, dest)

    monkeypatch.setattr(config, "option", counted)
    for _ in range(5):
        _take("a", "x", 64)
    assert reads == [(None, "cake_workspace_pool_bytes")]
    cake.configure(cake_workspace_pool_bytes=KIB)
    for _ in range(5):
        _take("a", "x", 64)
    assert len(reads) == 2 and workspace.stats()["cpu"]["hits"] == 4


def test_the_default_bound_allocates_directly_after_the_first_take(monkeypatch):
    _take("a", "x", 64)
    assert workspace._direct is config._state
    monkeypatch.setattr(workspace, "_pooled", lambda *args: pytest.fail("a take left the direct path"))
    allocated = []
    empty = torch.empty

    def recorded(shape, *, dtype, device):
        allocated.append((shape, dtype, device))
        return empty(shape, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "empty", recorded)
    shape = torch.Size([2, 3])
    view = workspace.take("a", "y", shape, torch.bfloat16, "cpu")

    assert allocated == [(shape, torch.bfloat16, torch.device("cpu"))]
    assert view.shape == shape and view.dtype == torch.bfloat16
    assert workspace.stats() == {"cpu": {"bytes": 0, "entries": 0, "hits": 0, "misses": 2, "evictions": 0}}


def test_a_new_configuration_leaves_the_direct_path_until_its_bound_is_read():
    _take("a", "x", 64)
    cake.configure(cake_workspace_pool_bytes=KIB)
    _take("a", "x", 64)  # the new bound applies at once: the request is pooled

    assert workspace._direct is None and workspace.stats()["cpu"]["entries"] == 1
    cake.configure()
    _take("b", "x", 64)  # back at 0: this take evicts what the pool kept, the next one is direct

    assert workspace._direct is None and workspace.stats()["cpu"]["bytes"] == 0
    _take("a", "x", 64)

    assert workspace._direct is config._state
    assert workspace.stats()["cpu"] == {"bytes": 0, "entries": 0, "hits": 0, "misses": 4, "evictions": 1}


def test_a_buffer_held_on_any_device_keeps_takes_off_the_direct_path(monkeypatch, cuda):
    unbounded = config._state
    _take("a", "x", 64)  # direct under the default configuration
    cake.configure(**POOLED)
    workspace.take("a", "x", (16,), torch.float32, "cuda:1")  # kept on device 1
    monkeypatch.setattr(config, "_state", unbounded)  # a configuration of bound 0 again, e.g. one a test restores
    _take("a", "x", 64)  # device "cpu" holds nothing, device 1 does: no direct path

    assert workspace._direct is None and workspace.stats()[1]["entries"] == 1
    workspace.take("b", "x", (16,), torch.float32, "cuda:1")  # evicts device 1's buffer
    _take("a", "x", 64)

    assert workspace.stats()[1] == {"bytes": 0, "entries": 0, "hits": 0, "misses": 2, "evictions": 1}
    assert workspace._direct is unbounded


def test_a_failed_allocation_leaves_the_pool_consistent(monkeypatch):
    cake.configure(cake_workspace_pool_bytes=4 * KIB)
    _take("a", "x", 256)
    _take("b", "x", 256)
    _take("c", "x", 128)
    empty = torch.empty

    class OutOfMemory(RuntimeError):
        pass

    def out_of_memory(*args, dtype, **kwargs):
        if dtype is torch.uint8:  # a new pooled buffer
            raise OutOfMemory("fake: out of memory")
        return empty(*args, dtype=dtype, **kwargs)

    monkeypatch.setattr(torch, "empty", out_of_memory)
    with pytest.raises(OutOfMemory):
        _take("a", "x", 750)  # drops a's 1 KiB buffer as too small, evicts b to make room for 3000 bytes, then fails

    assert list(workspace._pools["cpu"]) == [("c", "x", None)]
    assert workspace.stats()["cpu"] == {"bytes": 512, "entries": 1, "hits": 0, "misses": 4, "evictions": 1}
    _check_total()
    monkeypatch.setattr(torch, "empty", empty)
    _take("a", "x", 750)
    _check_total()


def test_a_missing_byte_total_is_recomputed_from_the_pool(pooled):
    _take("a", "x", 256)
    _take("b", "x", 64)
    del workspace._held["cpu"]
    _take("a", "x", 128)  # a hit

    assert workspace._held["cpu"] == 1280
    _check_total()
    del workspace._held["cpu"]
    assert workspace.release("a") == KIB and workspace._held["cpu"] == 256
    _check_total()


def test_take_returns_a_view_of_the_requested_shape_and_dtype(pooled):
    view = workspace.take("a", "x", (3, 4), torch.bfloat16, torch.device("cpu"))

    assert view.shape == (3, 4) and view.dtype == torch.bfloat16 and view.device.type == "cpu"
    assert workspace.stats() == {"cpu": {"bytes": 24, "entries": 1, "hits": 0, "misses": 1, "evictions": 0}}


def test_every_take_returns_a_new_tensor_on_the_pooled_buffer(pooled):
    first = workspace.take("a", "x", (64,), torch.uint8, "cpu")  # the whole buffer, as uint8
    first.unsqueeze_(0)  # changing a returned tensor's shape leaves the pool's buffer as it was
    again = workspace.take("a", "x", (64,), torch.uint8, "cpu")
    (buffer,) = workspace._pools["cpu"].values()

    assert again is not buffer and again is not first and again.shape == (64,) and buffer.shape == (64,)
    assert again.data_ptr() == first.data_ptr() == buffer.data_ptr()
    typed = workspace.take("a", "x", (2, 8), torch.float32, "cpu")
    assert typed.shape == (2, 8) and typed.dtype == torch.float32 and typed.data_ptr() == buffer.data_ptr()


def test_a_key_reuses_its_buffer_while_it_is_large_enough(pooled):
    first = _take("a", "x", 256)
    smaller = _take("a", "x", 64, torch.int8)
    larger = _take("a", "x", 512)

    assert smaller.data_ptr() == first.data_ptr()
    assert workspace.stats()["cpu"] == {"bytes": 2 * KIB, "entries": 1, "hits": 1, "misses": 2, "evictions": 0}
    assert larger.untyped_storage().nbytes() == 2 * KIB
    _check_total()


def test_keys_are_separate(pooled):
    a = _take("a", "x", 256)
    b = _take("a", "y", 256)
    c = _take("b", "x", 256)

    assert len({a.data_ptr(), b.data_ptr(), c.data_ptr()}) == 3
    assert workspace.stats()["cpu"]["entries"] == 3


def test_the_least_recently_used_keys_are_evicted_to_stay_under_the_bound():
    cake.configure(cake_workspace_pool_bytes=3 * KIB)
    _take("a", "x", 256)
    _take("b", "x", 256)
    _take("a", "x", 256)  # hit: b is now the oldest
    _take("c", "x", 384)  # 1.5 KiB more: evicts b

    pool = workspace._pools["cpu"]
    assert list(pool) == [("a", "x", None), ("c", "x", None)]
    assert workspace.stats()["cpu"] == {"bytes": 2560, "entries": 2, "hits": 1, "misses": 3, "evictions": 1}
    _check_total()


def test_a_request_above_the_bound_is_not_kept():
    cake.configure(cake_workspace_pool_bytes=KIB)
    _take("a", "x", 128)
    big = _take("b", "x", 1024)

    assert big.numel() == 1024
    assert list(workspace._pools["cpu"]) == [("a", "x", None)]
    _check_total()


def test_a_zero_bound_keeps_nothing(pooled):
    _take("a", "x", 128)
    cake.configure(cake_workspace_pool_bytes=0)
    _take("a", "x", 128)
    _take("a", "x", 128)

    assert workspace.stats()["cpu"] == {"bytes": 0, "entries": 0, "hits": 0, "misses": 3, "evictions": 0}


def test_a_zero_bound_evicts_what_a_larger_one_kept_unless_the_stream_captures(pooled, place):
    _take("a", "x", 128)
    _take("b", "x", 128)
    cake.configure(cake_workspace_pool_bytes=0)
    place.capturing = True
    _take("c", "x", 128)  # a capture neither reads nor changes the pool

    assert workspace.stats()["cpu"] == {"bytes": 2 * 512, "entries": 2, "hits": 0, "misses": 3, "evictions": 0}
    place.capturing = False
    _take("c", "x", 128)  # evicts everything the larger bound kept

    assert workspace.stats()["cpu"] == {"bytes": 0, "entries": 0, "hits": 0, "misses": 4, "evictions": 2}
    _check_total()


def test_release_drops_the_buffers_of_one_op_or_all_and_returns_the_bytes(pooled):
    _take("a", "x", 256)
    _take("a", "y", 128)
    _take("b", "x", 64)

    assert workspace.release("a") == 1536
    assert list(workspace._pools["cpu"]) == [("b", "x", None)]
    _check_total()
    assert workspace.release() == 256
    assert workspace.stats()["cpu"]["bytes"] == 0
    _check_total()


def test_an_empty_pool_releases_nothing():
    assert workspace.release() == 0
    assert workspace.stats() == {}


def test_a_lowered_bound_applies_to_the_buffers_already_held(pooled):
    _take("a", "x", 256)
    _take("b", "x", 256)
    _take("c", "x", 512)
    cake.configure(cake_workspace_pool_bytes=1536)
    hit = _take("a", "x", 256)  # a hit, which evicts until the pool fits

    assert list(workspace._pools["cpu"]) == [("a", "x", None)]
    assert workspace.stats()["cpu"] == {"bytes": KIB, "entries": 1, "hits": 1, "misses": 3, "evictions": 2}
    cake.configure(cake_workspace_pool_bytes=512)
    smaller = _take("a", "x", 64)  # the held 1 KiB buffer is above the bound: replaced, not reused

    assert smaller.data_ptr() != hit.data_ptr()
    assert workspace.stats()["cpu"] == {"bytes": 256, "entries": 1, "hits": 1, "misses": 4, "evictions": 2}
    _check_total()


def test_the_running_total_follows_every_hit_miss_eviction_drop_and_release():
    rng = random.Random(7)
    bounds = [0, 700, 3 * KIB, 16 * KIB]
    cake.configure(cake_workspace_pool_bytes=16 * KIB)
    for step in range(600):
        if step % 97 == 0:
            cake.configure(cake_workspace_pool_bytes=rng.choice(bounds))
        if step % 89 == 0:
            workspace.release(rng.choice([None, "a", "b"]))
        _take(rng.choice("abc"), rng.choice("xyz"), rng.choice([0, 1, 16, 100, 256, 1000, 5000]))
        _check_total()
        limit = cake.option(None, "cake_workspace_pool_bytes")
        assert workspace._held["cpu"] <= limit
    stats = workspace.stats()["cpu"]
    assert stats["hits"] + stats["misses"] == 600 and stats["hits"] > 0 and stats["evictions"] > 0


def test_each_stream_has_its_own_buffer(pooled, place):
    first = _take("a", "x", 256)
    place.stream = 7
    other = _take("a", "x", 256)
    place.stream = None
    again = _take("a", "x", 256)

    assert other.data_ptr() != first.data_ptr() and again.data_ptr() == first.data_ptr()
    assert list(workspace._pools["cpu"]) == [("a", "x", 7), ("a", "x", None)]
    assert workspace.release("a") == 2 * KIB


def test_a_capturing_stream_bypasses_the_pool(pooled, place):
    kept = _take("a", "x", 256)
    place.capturing = True
    captured = _take("a", "x", 256)
    cake.configure(cake_workspace_pool_bytes=0)
    _take("b", "x", 256)  # nothing is evicted while capturing either

    assert captured.data_ptr() != kept.data_ptr()
    assert list(workspace._pools["cpu"]) == [("a", "x", None)]
    assert workspace.stats()["cpu"] == {"bytes": KIB, "entries": 1, "hits": 0, "misses": 3, "evictions": 0}


@pytest.mark.parametrize("bound", [0, 8 * KIB])
def test_concurrent_takes_and_stats_keep_the_pool_consistent(bound):
    cake.configure(cake_workspace_pool_bytes=bound)
    errors = []

    def work(index):
        try:
            for step in range(300):
                _take(f"op{index}", f"r{step % 5}", 64 + step % 7)
                workspace.stats()
        except Exception as exc:  # reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    stats = workspace.stats()["cpu"]
    assert stats["hits"] + stats["misses"] == 4 * 300 and stats["bytes"] <= bound
    _check_total()


def test_cuda_follows_the_current_device_and_an_index_is_resolved_once(cuda):
    cake.configure(**POOLED)
    workspace.take("a", "x", (16,), torch.float32, "cuda")
    cuda.index = 1
    workspace.take("a", "x", (16,), torch.float32, "cuda")  # "cuda" is the new current device, not a remembered one
    workspace.take("a", "x", (16,), torch.float32, torch.device("cuda", 1))

    assert {where: (s["entries"], s["hits"]) for where, s in workspace.stats().items()} == {0: (1, 0), 1: (1, 1)}
    assert list(workspace._pools[0]) == [("a", "x", 1000)] and list(workspace._pools[1]) == [("a", "x", 1001)]
    assert workspace._devices["cuda"] is workspace._CURRENT
    assert workspace._devices[torch.device("cuda", 1)] == (1, torch.device("cuda", 1))


def test_cuda_reads_the_stream_of_the_current_device_without_switching(cuda):
    cake.configure(**POOLED)
    first = workspace.take("a", "x", (16,), torch.float32, "cuda:0")
    cuda.streams[0] = FakeStream(2000)  # another stream on the same device: another buffer
    other = workspace.take("a", "x", (16,), torch.float32, "cuda:0")

    assert cuda.switches == 0 and other.data_ptr() != first.data_ptr()
    assert list(workspace._pools[0]) == [("a", "x", 1000), ("a", "x", 2000)]
    workspace.take("a", "x", (16,), torch.float32, "cuda:1")  # not the current device: read inside its device block

    assert cuda.switches == 1 and cuda.index == 0 and list(workspace._pools[1]) == [("a", "x", 1001)]


def test_cuda_capture_of_the_device_bypasses_its_pool(cuda):
    cake.configure(**POOLED)
    kept = workspace.take("a", "x", (16,), torch.float32, "cuda:1")
    cuda.capturing.add(1)
    captured = workspace.take("a", "x", (16,), torch.float32, "cuda:1")  # read inside device 1's block
    cuda.index = 1
    again = workspace.take("a", "x", (16,), torch.float32, "cuda")

    assert captured.data_ptr() != kept.data_ptr() and again.data_ptr() != kept.data_ptr()
    assert workspace.stats()[1] == {"bytes": 64, "entries": 1, "hits": 0, "misses": 3, "evictions": 0}


def test_cuda_with_the_default_bound_allocates_on_the_current_device(cuda):
    cuda.index = 1
    view = workspace.take("a", "x", (4, 4), torch.float32, "cuda")

    assert view.shape == (4, 4) and cuda.switches == 0
    assert workspace.stats() == {1: {"bytes": 0, "entries": 0, "hits": 0, "misses": 1, "evictions": 0}}


def test_cuda_at_the_default_bound_follows_the_current_device_on_the_direct_path(cuda):
    workspace.take("a", "x", (4,), torch.float32, "cuda")
    cuda.index = 1
    assert workspace._direct is config._state
    workspace.take("a", "x", (4,), torch.float32, "cuda")
    workspace.take("a", "x", (4,), torch.float32, torch.device("cuda", 0))

    assert cuda.allocated == [torch.device("cuda", 0), torch.device("cuda", 1), torch.device("cuda", 0)]
    assert {where: s["misses"] for where, s in workspace.stats().items()} == {0: 2, 1: 1}
    assert cuda.switches == 0


def test_cuda_reads_the_runtime_queries_and_builds_no_stream_object(cuda, monkeypatch):
    for name in ("current_device", "current_stream", "is_current_stream_capturing"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, name=name: pytest.fail(f"take called torch.cuda.{name}"))
    cake.configure(**POOLED)
    first = workspace.take("a", "x", (16,), torch.float32, "cuda")
    again = workspace.take("a", "x", (16,), torch.float32, "cuda:0")
    workspace.take("a", "x", (16,), torch.float32, "cuda:1")  # not the current device: inside its device block
    cuda.capturing.add(0)
    captured = workspace.take("a", "x", (16,), torch.float32, "cuda:0")

    assert again.data_ptr() == first.data_ptr() != captured.data_ptr()
    assert list(workspace._pools[0]) == [("a", "x", 1000)] and list(workspace._pools[1]) == [("a", "x", 1001)]
    assert cuda.switches == 1


def test_a_cuda_build_of_torch_has_the_runtime_queries_the_pool_reads():
    if torch.version.cuda is None and getattr(torch.version, "hip", None) is None:
        pytest.skip("a CPU build of torch")
    for name in ("_cuda_getDevice", "_cuda_getCurrentRawStream", "_cuda_isCurrentStreamCapturing"):
        assert callable(getattr(torch._C, name, None)), name


@requires_cuda
def test_cuda_streams_get_separate_buffers_and_a_capture_keeps_out_of_the_pool():
    cake.configure(**POOLED)
    device = torch.device("cuda", torch.cuda.current_device())
    default = workspace.take("a", "x", (256,), torch.float32, device)
    assert workspace._stream(device.index) == (torch.cuda.current_stream(device).cuda_stream, False)
    side = torch.cuda.Stream(device)
    with torch.cuda.stream(side):
        assert workspace._stream(device.index) == (side.cuda_stream, False)
        other = workspace.take("a", "x", (256,), torch.float32, device)
    assert other.data_ptr() != default.data_ptr()
    assert workspace.take("a", "x", (256,), torch.float32, device).data_ptr() == default.data_ptr()

    graph = torch.cuda.CUDAGraph()
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        workspace.take("a", "x", (256,), torch.float32, device)  # warm the side stream's buffer before capturing
        with torch.cuda.graph(graph, stream=side):
            assert workspace._stream(device.index) == (None, True)
            captured = workspace.take("a", "x", (256,), torch.float32, device)
            captured.fill_(1.0)
    torch.cuda.synchronize(device)
    pooled = {buffer.data_ptr() for buffer in workspace._pools[device.index].values()}
    assert captured.data_ptr() not in pooled
