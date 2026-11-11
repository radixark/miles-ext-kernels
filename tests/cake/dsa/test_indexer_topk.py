"""The deterministic indexer top-k wrapper without a GPU: the geometry it serves, the vendored package's contract and
library list, the first-call build, the arguments it hands the package, and the segment-local to packed key ids
(needs torch, not CUDA)."""

import contextlib
import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import miles_ext_kernels.cake._vendor as vendor  # noqa: E402
from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.dsa import indexer_topk  # noqa: E402

VENDOR = Path(vendor.__file__).resolve().parent / "cake_dsa_indexer_det_train"
ARCHES = ("sm_100a", "sm_103a", "sm_107a")


@pytest.mark.parametrize(
    ("heads", "head_dim", "top_k", "reason"),
    [
        (32, 128, 1, None),
        (32, 128, 2048, None),
        (32, 128, 4096, None),
        (64, 128, 2048, "64 index heads of dimension 128"),
        (16, 128, 512, "16 index heads of dimension 128"),
        (32, 64, 2048, "32 index heads of dimension 64"),
        (32, 128, 0, "top_k 0"),
        (32, 128, 4097, "top_k 4097"),
    ],
)
def test_the_served_geometry_is_32_heads_of_128_and_top_k_up_to_4096(heads, head_dim, top_k, reason):
    got = indexer_topk.indexer_det_topk_unserved_reason(heads, head_dim, top_k)
    if reason is None:
        assert got is None
    else:
        assert got is not None and got.startswith(reason)


def test_the_geometry_constants_are_the_vendored_backends():
    pytest.importorskip("tvm_ffi")
    backend = indexer_topk._backend()

    assert backend.__name__ == "miles_ext_kernels.cake._vendor.cake_dsa_indexer_det_train.cake_backend"
    assert (backend.NUM_HEADS, backend.HEAD_DIM, backend.MAX_TOP_K, backend.DEFAULT_TOP_K) == (
        indexer_topk.HEADS,
        indexer_topk.HEAD_DIM,
        indexer_topk.MAX_TOP_K,
        indexer_topk.DEFAULT_TOP_K,
    )
    assert set(backend.SUPPORTED_COMPUTE_CAPABILITIES.values()) == set(ARCHES)


@pytest.mark.parametrize("arch", ARCHES)
def test_prebuild_names_every_library_the_package_builds_for_the_architecture(arch):
    build = json.loads((VENDOR / "build.json").read_text())
    entries = {entry["name"] for entry in build["libraries"] if entry["arch"] == arch}

    names = indexer_topk.library_names(arch)

    assert names == sorted(entries)
    assert len(names) == {"sm_100a": 65, "sm_103a": 73, "sm_107a": 63}[arch]


@pytest.mark.parametrize(
    ("q_shape", "top_k", "match"),
    [
        ((8, 64, 128), 16, "64 index heads"),
        ((8, 32, 64), 16, "dimension 64"),
        ((8, 32, 128), 4097, "top_k 4097"),
        ((8, 32, 128), 0, "top_k 0"),
        ((8, 4096), 16, r"index_q must be \[T, 32, 128\]"),
    ],
)
def test_a_call_of_an_unserved_geometry_raises_before_it_reaches_the_device(q_shape, top_k, match):
    q = torch.zeros(q_shape, dtype=torch.bfloat16)
    k = torch.zeros(16, 128, dtype=torch.bfloat16)
    w = torch.zeros(8, 32)
    cu = torch.tensor([0, 8], dtype=torch.int32)

    with pytest.raises(ValueError, match=match):
        indexer_topk.dsa_indexer_det_topk(q, k, w, cu, cu, top_k)


def test_packed_key_ids_add_the_query_segments_key_start_and_keep_the_padding():
    # segments: 2 queries / 10 keys, 0 queries / 3 keys, 3 queries / 7 keys
    cu_q = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 10, 13, 20], dtype=torch.int32)
    local = torch.tensor([[0, 9, -1], [3, -1, -1], [0, 1, 6], [2, -1, -1], [-1, -1, -1]], dtype=torch.int32)

    packed = indexer_topk.packed_key_ids(local, cu_q, cu_k)

    assert packed.dtype == torch.int32
    assert packed.tolist() == [[0, 9, -1], [3, -1, -1], [13, 14, 19], [15, -1, -1], [-1, -1, -1]]
    empty = indexer_topk.packed_key_ids(torch.empty(0, 4, dtype=torch.int32), cu_q[:1], cu_k[:1])
    assert tuple(empty.shape) == (0, 4)


@pytest.fixture
def stub_package(monkeypatch):
    """The wrapper on a fake SM100a device: records the libraries it builds and the arguments it hands the package."""
    built, calls = [], []
    backend = types.SimpleNamespace(
        cake_dsa_indexer_det_train_topk=lambda *args, **kwargs: calls.append((args, kwargs)) or ("indices", "scores")
    )
    monkeypatch.setattr(indexer_topk, "_WARM", set())
    monkeypatch.setattr(indexer_topk, "_arch", lambda device: "sm_100a")
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(indexer_topk, "_package", lambda: ("cake_build", "cake_jit"))
    monkeypatch.setattr(indexer_topk, "_backend", lambda: backend)
    monkeypatch.setattr(indexer_topk, "library_names", lambda arch: [f"lib_a_{arch}", f"lib_b_{arch}"])
    monkeypatch.setattr(
        indexer_topk.loader,
        "prebuild_libraries",
        lambda module, names, max_workers=None: built.append((module, [*names])) or [],
    )
    return built, calls


def _inputs():
    q = torch.zeros(4, 128, 32, dtype=torch.bfloat16).transpose(1, 2)  # [4, 32, 128], not contiguous
    k = torch.zeros(6, 128, dtype=torch.bfloat16)
    w = torch.ones(4, 32, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    return q, k, w, cu


def test_the_first_call_on_an_architecture_builds_all_its_libraries(stub_package):
    built, calls = stub_package
    q, k, w, cu = _inputs()

    assert indexer_topk.dsa_indexer_det_topk(q, k, w, cu, cu, 2) == ("indices", "scores")
    indexer_topk.dsa_indexer_det_topk(q, k, w, cu, cu, 2)

    assert built == [("cake_build", ["lib_a_sm_100a", "lib_b_sm_100a"])]
    assert len(calls) == 2


def test_lazy_builds_leave_each_library_to_its_first_launch(stub_package):
    built, calls = stub_package
    cake.configure(cake_dsa_lazy_build=True)
    q, k, w, cu = _inputs()

    indexer_topk.dsa_indexer_det_topk(q, k, w, cu, cu, 2)

    assert built == [] and len(calls) == 1


def test_the_call_hands_the_package_contiguous_queries_fp32_weights_and_the_workspace(stub_package):
    _, calls = stub_package
    q, k, w, cu = _inputs()
    offsets = torch.zeros(1, dtype=torch.int64)
    workspace = torch.empty(64, dtype=torch.uint8)

    indexer_topk.dsa_indexer_det_topk(
        q, k, w, cu, cu, 3, softmax_scale=1.0, q_causal_offsets=offsets, ratio=2, max_seqlen_k=6, workspace=workspace
    )

    (args, kwargs) = calls[0]
    q_arg, k_arg, w_arg, cu_q, cu_k = args
    assert q_arg.is_contiguous() and torch.equal(q_arg, q)
    assert k_arg is k
    assert w_arg.dtype == torch.float32 and w_arg.is_contiguous() and torch.equal(w_arg, w.float())
    assert cu_q is cu and cu_k is cu
    assert kwargs.pop("q_causal_offsets") is offsets and kwargs.pop("workspace_buffer") is workspace
    assert kwargs == {"top_k": 3, "softmax_scale": 1.0, "ratio": 2, "max_seqlen_k": 6, "indices": None, "scores": None}
