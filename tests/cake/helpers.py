"""Shared helpers of the op tests: accuracy gates, bitwise comparison, the prebuild contract, device markers.

Accuracy: compare every output and every gradient with :func:`assert_rel_rms` against an FP64 (or FP32) reference,
with a budget per tensor of at most twice the measured error (note the value, architecture and shape next to it).
Determinism: :func:`assert_repeatable` runs a ``det`` call twice in one process and compares the bits. Prebuild:
:func:`check_prebuild` runs every route of an op after one ``prebuild``, with no build allowed. Workspace pool: a test
of pooled buffers configures :data:`POOLED` (the default bound, 0, keeps nothing).

The functions import torch when they run. Importing this module imports torch if it is installed, to evaluate
:data:`requires_cuda`; without torch it still imports, and the CUDA tests skip.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import pytest

from miles_ext_kernels.cake._runtime import loader

__all__ = [
    "POOLED",
    "assert_bitwise",
    "assert_rel_rms",
    "assert_repeatable",
    "check_prebuild",
    "no_build",
    "rel_rms",
    "requires_arch",
    "requires_cuda",
]

no_build = loader.no_build

# The configuration of a test of the workspace pool: a bound of 1 GiB. configure() replaces the whole configuration, so
# pass it with every configure call of such a test, e.g. ``cake.configure(cake_gdn_recompute_state=True, **POOLED)``;
# the fixture ``pooled`` (tests/conftest.py) configures it alone.
POOLED: Mapping[str, Any] = MappingProxyType({"cake_workspace_pool_bytes": 1 << 30})


def rel_rms(got: Any, ref: Any) -> float:
    """``sqrt(mean((got - ref)^2)) / sqrt(mean(ref^2))`` in float64; ``max|got|`` when ``ref`` is all zero."""
    import torch

    if tuple(got.shape) != tuple(ref.shape):
        raise AssertionError(f"shape {tuple(got.shape)} != reference shape {tuple(ref.shape)}")
    got = got.detach().to(device="cpu", dtype=torch.float64)
    ref = ref.detach().to(device="cpu", dtype=torch.float64)
    if got.numel() == 0:
        return 0.0
    norm = ref.square().mean().sqrt().item()
    if norm == 0.0:
        return got.abs().max().item()
    return (got - ref).square().mean().sqrt().item() / norm


def assert_rel_rms(
    got: Mapping[str, Any],
    ref: Mapping[str, Any],
    budget: Mapping[str, float],
    *,
    detect_scale: float | None = 1.02,
    loose: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Gate every tensor of ``got`` by its relative RMS error against ``ref``; returns the measured errors.

    ``budget`` must name exactly the tensors of ``got`` (every output and every gradient). Each tensor must be finite
    and within its budget. ``detect_scale``: the same tensor scaled by this factor must fail its budget, or the gate is
    too loose to catch a 2 % error; ``loose`` names the tensors exempt from that check, with the reason (a tensor
    whose own error is about 1 % or more). All failures are reported at once, with the measured values.
    """
    import torch

    loose = dict(loose or {})
    if set(budget) != set(got):
        raise AssertionError(f"budget names {sorted(budget)}, the outputs are {sorted(got)}")
    if missing := sorted(set(got) - set(ref)):
        raise AssertionError(f"no reference for {missing}")
    if unknown := sorted(set(loose) - set(got)):
        raise AssertionError(f"loose names {unknown}, which are not outputs")
    if empty := sorted(name for name, reason in loose.items() if not reason):
        raise AssertionError(f"loose gives no reason for {empty}")
    measured: dict[str, float] = {}
    failures: list[str] = []
    for name in sorted(got):
        tensor = got[name]
        if not bool(torch.isfinite(tensor).all()):
            failures.append(f"{name}: not finite")
            continue
        error = measured[name] = rel_rms(tensor, ref[name])
        if error > budget[name]:
            failures.append(f"{name}: rel_rms {error:.3e} > budget {budget[name]:.3e}")
        elif detect_scale is not None and name not in loose:
            scaled = rel_rms(tensor.detach().to(torch.float64) * detect_scale, ref[name])
            if scaled <= budget[name]:
                failures.append(
                    f"{name}: gate too loose: x{detect_scale} gives rel_rms {scaled:.3e} <= budget {budget[name]:.3e}"
                )
    if failures:
        raise AssertionError("\n".join([*failures, f"measured: {measured}"]))
    return measured


def _bits(tensor: Any) -> Any:
    """The bytes of every element of ``tensor`` as rows of a uint8 matrix (NaN payloads included)."""
    import torch

    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    return flat.view(torch.uint8).reshape(flat.numel(), tensor.element_size())


def _index(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    index = []
    for size in reversed(shape):
        flat, position = divmod(flat, size)
        index.append(position)
    return tuple(reversed(index))


def assert_bitwise(a: Any, b: Any) -> None:
    """Same shape, dtype and bit pattern: one tensor, or mappings of tensors with the same keys.

    Reports the first differing index of every tensor that differs.
    """
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not (isinstance(a, Mapping) and isinstance(b, Mapping)) or set(a) != set(b):
            raise AssertionError("assert_bitwise compares two tensors or two mappings of the same tensors")
        pairs = [(name, a[name], b[name]) for name in sorted(a)]
    else:
        pairs = [("tensor", a, b)]
    failures = []
    for name, x, y in pairs:
        if tuple(x.shape) != tuple(y.shape) or x.dtype != y.dtype:
            failures.append(f"{name}: {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}")
            continue
        differs = (_bits(x) != _bits(y)).any(dim=1).nonzero()
        if differs.numel():
            first = _index(int(differs[0, 0]), tuple(x.shape))
            failures.append(f"{name}: {differs.shape[0]} of {x.numel()} elements differ, first at {first}")
    if failures:
        raise AssertionError("\n".join(failures))


def assert_repeatable(run: Callable[[], Mapping[str, Any]], times: int = 2) -> Mapping[str, Any]:
    """Call ``run`` (which makes its own fresh inputs) ``times`` times; every result equals the first, bit for bit."""
    first = run()
    for _ in range(times - 1):
        assert_bitwise(first, run())
    return first


def check_prebuild(op: str, routes: Mapping[str, Callable[[str], Any]], **model_params: Any) -> None:
    """The prebuild contract of ``op``: one ``prebuild(op, **model_params)``, then nothing compiles.

    ``routes`` (the ``ROUTES`` of ``tests/cake/<op>/test_prebuild_gpu.py``) maps a name to a call that runs one route
    of the op, forward and backward. Each route runs once per variant of the op under :func:`no_build`, with that
    variant configured; it receives the variant, so a route that needs options configures them with it.
    """
    from miles_ext_kernels import cake

    if not routes:
        raise AssertionError(f"{op}: no routes to check")
    spec = cake.op_spec(op)
    cake.prebuild(op, **model_params)
    for variant in spec.variants:
        for name, run in routes.items():
            cake.configure(**{spec.variant_dest: variant})
            try:
                with no_build():
                    run(variant)
            except Exception as exc:
                raise AssertionError(f"{op} {variant}, route {name}: {exc}") from exc


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _device_arch() -> str | None:
    if not _cuda_available():
        return None
    try:
        return loader.device_arch()
    except NotImplementedError:
        return None


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA device")


def requires_arch(*arches: str) -> pytest.MarkDecorator:
    """Skip unless the current CUDA device is one of ``arches`` (``"sm_100a"``, ``"sm_103a"``, ``"sm_107a"``)."""
    arch = _device_arch()
    return pytest.mark.skipif(arch not in arches, reason=f"needs one of {arches}, the device is {arch}")
