"""The prebuild contract of lm_head on a GPU (README, Prebuild): after one ``cake.prebuild("lm_head")`` no call
compiles, whatever the variant, the temperature (none, a scalar, a per-row tensor, 1.0), the token count (none, one
row, one chunk, several chunks with a ragged tail, a short tail ``fast``'s forward folds, one chunk whose logits
``det`` keeps for the backward), ``fast``'s two backward forms (vocabulary-major for 2 to 4 chunks, the chunk loop
otherwise, each with zero-gradient rows), the chunk size, a given valid-row count, ignored rows, frozen inputs or a
forward without gradients, at two geometries. ``ROUTES`` lists these call-time paths; each runs one forward and backward
under ``no_build()`` for each variant."""

import sys

import pytest

torch = pytest.importorskip("torch")

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake import lm_head  # noqa: E402
from miles_ext_kernels.cake._impl.lm_head import ops  # noqa: E402

from ..helpers import check_prebuild, no_build, requires_arch, requires_cuda  # noqa: E402

pytestmark = [requires_cuda, requires_arch("sm_100a", "sm_103a", "sm_107a")]

# (H, V): the smallest and the widest-hidden geometry of the ones miles trains (one program serves both)
GEOMETRIES = ((4096, 129280), (6144, 154880))
SEED = 20261105
_WEIGHTS: dict = {}


def _weight(H: int, V: int):
    if (H, V) not in _WEIGHTS:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(SEED + H + V)
        _WEIGHTS[(H, V)] = (torch.randn((V, H), generator=gen, device="cuda") * H**-0.5).to(torch.bfloat16)
    return _WEIGHTS[(H, V)]


def _fwd_bwd(
    H: int,
    V: int,
    T: int,
    *,
    temperature=None,
    ignore: float = 0.05,
    zero: float = 0.0,
    given_count: bool = False,
    train_x: bool = True,
    train_w: bool = True,
    grad: bool = True,
    chunk_size: int | None = None,
) -> None:
    """One ``log_probs`` forward and, when an input trains, its backward. ``temperature="rows"``: a per-row tensor;
    ``chunk_size``: configured with the variant the call runs."""
    if chunk_size is not None:
        cake.configure(cake_lm_head_variant=cake.variant("lm_head"), cake_lm_head_chunk_size=chunk_size)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED + T)
    X = (torch.randn((T, H), generator=gen, device="cuda") * 0.5).to(torch.bfloat16).requires_grad_(train_x)
    W = _weight(H, V).detach().requires_grad_(train_w)
    labels = torch.randint(0, V, (T,), generator=gen, device="cuda")
    labels[torch.rand((T,), generator=gen, device="cuda") < ignore] = -100
    dlogp = torch.randn((T,), generator=gen, device="cuda") * 0.05
    dlogp[torch.rand((T,), generator=gen, device="cuda") < zero] = 0.0
    dlogp = torch.where(labels >= 0, dlogp, torch.zeros_like(dlogp))
    if isinstance(temperature, str):
        temperature = 0.5 + torch.rand((T,), generator=gen, device="cuda")
    count = int((labels >= 0).sum()) if given_count else None
    with torch.set_grad_enabled(grad):
        logp = lm_head.log_probs(X, W, labels, temperature=temperature, num_valid_rows=count)
    if grad and (train_x or train_w):
        logp.backward(dlogp)
    torch.cuda.synchronize()


# the call-time paths of one geometry: name -> (H, V) -> one forward and backward
CASES = {
    "two chunks, a one-row tail, no temperature": lambda H, V: _fwd_bwd(H, V, 4097),
    "scalar temperature": lambda H, V: _fwd_bwd(H, V, 4097, temperature=0.7),
    "per-row temperature": lambda H, V: _fwd_bwd(H, V, 4097, temperature="rows"),
    "unit temperature": lambda H, V: _fwd_bwd(H, V, 4097, temperature=1.0),
    "one row": lambda H, V: _fwd_bwd(H, V, 1, temperature=0.7, ignore=0.0),
    "one chunk": lambda H, V: _fwd_bwd(H, V, 4095, temperature=0.7),
    "one chunk, every row valid, no temperature (det keeps the logits)": lambda H, V: _fwd_bwd(H, V, 4095, ignore=0.0),
    "two chunks, all valid, a one-row tail (fast: folded, vocab-major)": lambda H, V: _fwd_bwd(H, V, 4097, ignore=0.0),
    "W frozen, a one-row tail (fast: folded chunk loop)": lambda H, V: _fwd_bwd(H, V, 4097, ignore=0.0, train_w=False),
    "four chunks, zero-gradient rows (fast: vocab-major)": lambda H, V: _fwd_bwd(H, V, 16231, zero=0.3),
    "four chunks, zero-grad rows, tau (vocab-major)": lambda H, V: _fwd_bwd(H, V, 16231, temperature=0.7, zero=0.3),
    "five chunks, zero-gradient rows (fast: chunk loop)": lambda H, V: _fwd_bwd(H, V, 20481, zero=0.3),
    "five chunks, zero-grad rows, tau (chunk loop)": lambda H, V: _fwd_bwd(H, V, 20481, temperature=0.7, zero=0.3),
    "two chunks, a long tail, zero-gradient rows (fast: chunk loop)": lambda H, V: _fwd_bwd(H, V, 6800, zero=0.3),
    "a third of the rows ignored": lambda H, V: _fwd_bwd(H, V, 16384, ignore=0.33),
    "given valid-row count": lambda H, V: _fwd_bwd(H, V, 16231, temperature=0.7, given_count=True),
    "given count, every row valid": lambda H, V: _fwd_bwd(H, V, 4097, ignore=0.0, given_count=True),
    "every row ignored": lambda H, V: _fwd_bwd(H, V, 4097, temperature=0.7, ignore=1.0),
    "no rows": lambda H, V: _fwd_bwd(H, V, 0, temperature=0.7),
    "no gradients": lambda H, V: _fwd_bwd(H, V, 4097, temperature=0.7, grad=False),
    "X trains, W frozen": lambda H, V: _fwd_bwd(H, V, 4097, temperature=0.7, train_w=False),
    "W trains, X frozen": lambda H, V: _fwd_bwd(H, V, 4097, temperature=0.7, train_x=False),
    "chunk size 1024": lambda H, V: _fwd_bwd(H, V, 4097, temperature=0.7, chunk_size=1024),
    "chunk size 1024, five chunks (fast: chunk loop)": lambda H, V: _fwd_bwd(H, V, 5000, zero=0.3, chunk_size=1024),
    "chunk size 65535, two chunks": lambda H, V: _fwd_bwd(H, V, 70001, chunk_size=65535),
}


def _route(case, H: int, V: int):
    return lambda variant: case(H, V)


ROUTES = {f"H{H} V{V}: {name}": _route(case, H, V) for H, V in GEOMETRIES for name, case in CASES.items()}


def test_after_one_prebuild_no_route_of_either_variant_compiles():
    check_prebuild("lm_head", ROUTES)


def test_prebuild_returns_every_log_prob_stage_with_and_without_a_temperature_and_is_idempotent():
    arch = cake.device_arch()
    paths = cake.prebuild("lm_head", device=torch.cuda.current_device())
    expected = 0
    for variant in ops.PACKAGES:
        jit = ops._jit(variant)
        record = jit.select_module(arch)
        plain = {jit.library_name(record, stage) for stage in jit.registered_stages(record)}
        every = {jit.library_name(record, stage) for stage in jit.registered_stages_with_temperature(record)}
        assert every > plain, f"{variant}: no temperature library"
        expected += len(every)

    assert len(paths) == len(set(paths)) == expected
    assert all(path.is_file() for path in paths)
    built = {path: path.stat().st_mtime_ns for path in paths}
    assert {path: path.stat().st_mtime_ns for path in cake.prebuild("lm_head")} == built


# the entry of each variant's package that log_probs calls
ENTRIES = {
    "det": "cake_lm_head_det_train_logprob_temperature",
    "fast": "cake_lm_head_nondet_train_logprob_temperature",
}


def _spy(calls: list, variant: str, original):
    def entry(*args, **kwargs):
        calls.append(variant)
        return original(*args, **kwargs)

    return entry


def test_the_variant_configured_after_a_call_runs_at_the_next_call_without_compiling(monkeypatch):
    calls: list[str] = []
    for variant, name in ENTRIES.items():
        entry = ops._entry(variant)
        monkeypatch.setattr(entry, name, _spy(calls, variant, getattr(entry, name)))
    H, V = GEOMETRIES[0]
    cake.prebuild("lm_head")

    cake.configure(cake_lm_head_variant="det")
    _fwd_bwd(H, V, 4097, temperature=0.7)
    cake.configure(cake_lm_head_variant="fast")
    with no_build():
        _fwd_bwd(H, V, 4097, temperature=0.7)

    assert calls == ["det", "fast"]


def test_the_calls_import_no_flashinfer():
    cake.prebuild("lm_head")
    for name, run in list(ROUTES.items())[:3]:
        run("det")

    assert [name for name in sys.modules if name.split(".")[0] == "flashinfer"] == []
