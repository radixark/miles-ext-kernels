"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The entry points of the Cake four-stream mHC training kernels (SM100a / SM103a / SM107a).

The function of one connection, per token ``t`` (FP32 arithmetic, BF16 only at the activation boundaries)::

    u = float32(X[t]).reshape(4 H);  r = rsqrt(mean(u * u) + rms_eps);  z = (u r) @ fn^T    # [24]
    pre  = sigmoid(z[0:4] scale[0] + base[0:4]) + hc_eps
    post = 2 sigmoid(z[4:8] scale[1] + base[4:8])
    C    = Sinkhorn(z[8:24] scale[2] + base[8:24])     # 4 x 4: row softmax + hc_eps, then sinkhorn_iters
                                                       # column / row normalisations, each sum + hc_eps
    U[t]    = BF16(sum_i pre_i X[t, i])                                  # mhc_prepare
    Y[t, j] = BF16(post_j B[t] + sum_i C_ij X[t, i])                     # mhc_finish, B = branch(U)

The coefficients stay FP32 (Megatron's hyper-connection module casts them to the activation dtype first). Tensor
contract: ``X`` BF16 ``[T, 4, H]``, ``fn`` ``[24, 4 H]``, ``scale`` ``[3]``, ``base`` ``[24]``, ``B`` BF16 ``[T, H]``,
``H`` in :data:`HIDDEN_SIZES`, all contiguous with 32-byte aligned storage: the kernels read every operand in place,
and an operand they cannot read is a ``ValueError``, never a silent copy. The parameters ``fn``, ``scale`` and
``base`` are FP32, or BF16 (what a mixed-precision wrapper such as FSDP2 with ``param_dtype=torch.bfloat16`` hands the
layer): a BF16 parameter is cast to FP32 at the call, which is exact, and autograd returns its gradient in BF16 (the
FP32 gradient rounded once). ``T`` is a run-time scalar (``T == 0`` launches nothing).

The kernels are four vendored FlashInfer packages:

* ``cake_mhc_det_train`` (``det``): forward, and a backward whose ``dfn`` / ``dscale`` / ``dbase`` are fixed-order
  reductions, bitwise reproducible for a token count on a device.
* ``cake_mhc_nondet_train`` (``fast``): copies of the ``det`` forward programs (the same bits) and a backward that
  adds the ``dfn`` partial of every (column slab, token chunk) unit with FP32 atomics, so only the summation order of
  ``dfn`` varies from run to run (each element within ``P <= 32`` FP32 roundings of the fixed-order sum of its
  per-chunk partials). ``dX`` is the ``det`` bits; ``dscale`` / ``dbase`` are the ``det`` package's per-CTA
  reduction, whose low-order bits differ from ``det`` at hidden 4096 above 14927 tokens (``det`` sums them over
  8-token group rows there).
* ``cake_mhc_head_det_train`` (``det``) and ``cake_mhc_head_nondet_train`` (``fast``): the model head's stream mean,
  bit for bit torch's BF16 ``X.mean(dim=1)``, and its gradient. The head is elementwise per token (no cross-token
  reduction, no atomics); the ``fast`` package runs the ``det`` package's kernels under its own library names, so
  both give the same bits.

Variant. :func:`mhc_prepare` and :func:`mhc_head` read ``config.variant("mhc")`` when they run: ``det`` unless
``configure`` set ``fast`` (``validate_args`` resolves miles' default to ``fast`` and deterministic mode to ``det``).
The handle :func:`mhc_prepare` returns belongs to that variant's package, so :func:`mhc_finish` and the backward of
the pair run the variant of the prepare; the backward of the head runs the package of its forward. The variant comes
from the configuration only: under ``torch.use_deterministic_algorithms(True)`` the backward of a ``fast`` connection
raises PyTorch's alert for an operation without a deterministic implementation, and the head of either variant raises
nothing (it has no order-varying step).

Workspaces. Each launch of the forward, the statistics replay and the backward takes its workspace from the runtime's
workspace pool when it runs (roles ``"det-fwd"``, ``"det-bwd"``, ``"fast-fwd"``, ``"fast-bwd"``, sized for the call's
power-of-two token-count bucket), never from the forward's autograd context; the packages' per-device default
workspaces stay empty. The kernels' work tickets (the first 256 bytes of a workspace) must be zero before a launch,
and the content of a workspace is undefined, so every launch zeroes its tickets first (256 bytes, on the stream of the
call), whether the buffer comes from the pool or is allocated for the call alone (bound ``0``, a request above the
bound, a CUDA graph capture). One op must not run on two streams at once; calls that alternate between streams use
one buffer per stream. ``miles_ext_kernels.release()`` frees the pool's buffers, then :func:`release` drops the
launch plans and the loaded libraries.

:func:`prebuild` builds every library both variants and their heads may launch at one hidden size and Sinkhorn
iteration count. The epsilons are launch arguments of every program: they select no library.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader, workspace
from miles_ext_kernels.cake._vendor import cake_mhc_det_train as _det
from miles_ext_kernels.cake._vendor import cake_mhc_head_det_train as _det_head
from miles_ext_kernels.cake._vendor import cake_mhc_head_nondet_train as _fast_head
from miles_ext_kernels.cake._vendor import cake_mhc_nondet_train as _fast

__all__ = [
    "HC_EPS",
    "HIDDEN_SIZES",
    "RMS_EPS",
    "SINKHORN_ITERS",
    "SINKHORN_ITERS_MAX",
    "STREAMS",
    "is_supported",
    "libraries",
    "mhc_finish",
    "mhc_head",
    "mhc_prepare",
    "prebuild",
    "release",
    "supported_hidden_sizes",
]

_OP = "mhc"
# The connection shapes: four residual streams of one of these hidden sizes (4096 only on SM107a).
HIDDEN_SIZES: tuple[int, ...] = tuple(_det.cake_jit.HIDDEN_SIZES)
STREAMS: int = _det.cake_jit.STREAMS
# FlashInfer's defaults of the epsilons and the Sinkhorn count; every count from 1 to SINKHORN_ITERS_MAX runs (a count
# other than SINKHORN_ITERS on the run-time-count programs of the coefficient and token-owner passes).
RMS_EPS: float = _det.cake_jit.RMS_EPS
HC_EPS: float = _det.cake_jit.HC_EPS
SINKHORN_ITERS: int = _det.cake_jit.SINKHORN_ITERS
SINKHORN_ITERS_MAX: int = _det.cake_jit.SINKHORN_ITERS_MAX

# (cache directory, architecture, hidden, sinkhorn_iters) -> the library paths a prebuild returned
_built: dict[tuple[str, str, int, int], list[Path]] = {}


def _device(device: Any) -> torch.device:
    """A CUDA device (``None``: the current one; an int: that index)."""
    if device is None or isinstance(device, int):
        index = torch.cuda.current_device() if device is None else device
        return torch.device("cuda", index)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _fp32(parameter: Any) -> Any:
    """A BF16 parameter as FP32 (exact; autograd hands the gradient back in BF16), anything else unchanged."""
    if isinstance(parameter, torch.Tensor) and parameter.dtype == torch.bfloat16:
        return parameter.float()
    return parameter


def _pooled_workspaces(package: ModuleType, variant: str) -> Callable[..., dict[str, torch.Tensor]]:
    """What ``package`` calls for the region views of a launch's workspace (its ``_workspace_for``): a buffer of the
    runtime's workspace pool, taken when the launch runs, instead of the package's per-device default workspaces."""
    views_of, nbytes_of = package._workspace_views, package._workspace_bytes

    def workspace_for(
        caller: torch.Tensor | None, kind: str, rows: int, hidden: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        if caller is not None:  # a caller-owned workspace (this op never passes one)
            return package._caller_views(caller, kind, rows, hidden, device)
        bucket = 1 << (max(int(rows), 1) - 1).bit_length()  # the packages' token-count buckets
        region = workspace.take(
            _OP, f"{variant}-{kind}", (nbytes_of(kind, bucket, device, hidden),), torch.uint8, device
        )
        views = views_of(region, kind, rows=rows, device=device, hidden=hidden)
        if "ticket" in views:  # no region at T == 0
            views["ticket"].zero_()
        return views

    return workspace_for


# Every launch of either package gets its workspace from ``_workspace_for`` (directly or through ``_fast_workspace``,
# which reads only the default views ``_workspace_for`` would have filled).
_det._workspace_for = _pooled_workspaces(_det, "det")
_fast._workspace_for = _pooled_workspaces(_fast, "fast")


def is_supported(device: Any, hidden: int | None = None) -> bool:
    """Whether ``device`` (and ``hidden``, when given) runs every entry point of both variants and their heads."""
    device = _device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if tuple(torch.cuda.get_device_capability(device)) not in loader.ARCH_BY_CAPABILITY:
        return False
    if hidden is not None and int(hidden) not in HIDDEN_SIZES:
        return False
    return (
        _det.is_cake_mhc_det_train_supported(device, hidden)
        and _fast.is_cake_mhc_nondet_train_supported(device, hidden)
        and _det_head.is_cake_mhc_head_det_train_supported(device, hidden)
        and _fast_head.is_cake_mhc_head_nondet_train_supported(device, hidden)
    )


def supported_hidden_sizes(device: Any) -> tuple[int, ...]:
    """The hidden sizes :func:`is_supported` accepts on ``device``: all of :data:`HIDDEN_SIZES` on SM100a / SM103a,
    4096 on SM107a, none on another device."""
    return tuple(hidden for hidden in HIDDEN_SIZES if is_supported(device, hidden))


def mhc_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = RMS_EPS,
    hc_eps: float = HC_EPS,
    sinkhorn_iters: int = SINKHORN_ITERS,
) -> tuple[torch.Tensor, Any]:
    """Autograd-enabled mapping, coefficients and collapse of one connection: ``(U, handle)``.

    ``U`` is BF16 ``[T, H]``; hand ``handle`` to one :func:`mhc_finish` with the same ``X``. ``X``, ``fn``,
    ``scale`` and ``base`` may require gradients (``rms_eps`` / ``hc_eps`` / ``sinkhorn_iters`` as in the module
    docstring; a ``sinkhorn_iters`` other than :data:`SINKHORN_ITERS` runs the run-time-count programs, which
    ``prebuild`` builds for the model's count). The configured variant of ``mhc`` picks the backward.
    """
    fn, scale, base = _fp32(fn), _fp32(scale), _fp32(base)
    if config.variant(_OP) == "fast":
        return _fast.cake_mhc_nondet_train_prepare(
            X, fn, scale, base, rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters
        )
    return _det.cake_mhc_det_train_prepare(
        X, fn, scale, base, rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters
    )


def mhc_finish(X: torch.Tensor, B: torch.Tensor, handle: Any) -> torch.Tensor:
    """Autograd-enabled residual mixing: BF16 ``Y`` ``[T, 4, H]`` from ``X``, the branch output ``B`` ``[T, H]`` and
    the ``handle`` of :func:`mhc_prepare`, in the package (the variant) that made the handle.

    The backward returns ``dB`` and no gradient for ``X``: the residual-mixing contribution is part of the single-cast
    ``dX`` of the prepare node. A reentrant checkpoint region holds both calls or starts after the prepare.
    """
    if isinstance(handle, _fast._MHCTrainHandle):
        return _fast.cake_mhc_nondet_train_finish(X, B, handle)
    if isinstance(handle, _det._MHCTrainHandle):
        return _det.cake_mhc_det_train_finish(X, B, handle)
    raise TypeError(f"handle must be the handle mhc_prepare returned, not {type(handle).__name__}")


def mhc_head(X: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled head: BF16 ``Y = X.mean(dim=1)`` ``[T, H]`` of the four streams ``X`` ``[T, 4, H]``, and its
    gradient, bit for bit torch's, in the head package of the configured variant (its backward runs the same
    package)."""
    if config.variant(_OP) == "fast":
        return _fast_head.cake_mhc_head_nondet(X)
    return _det_head.cake_mhc_head_det(X)


def _run_time(jit: ModuleType, record: str) -> bool:
    return (jit.MODULES[record]["select_rule"] or {}).get("kind") == "run_time_iters"


def _route_libraries(jit: ModuleType, arch: str, hidden: int, sinkhorn_iters: int) -> tuple[str, ...]:
    """The ``build.json`` libraries every route of ``hidden`` on ``arch`` may launch at ``sinkhorn_iters``.

    At the count of record the run-time-count programs never run; at another count they stand in for the compiled
    programs of their role (the coefficient pass and the token-owner pass), and are built with every other program of
    the routes.
    """
    names = [
        jit.library_name(record, arch)
        for kind in jit.KINDS
        for record in jit.route_modules(arch, hidden, kind)
        if sinkhorn_iters != jit.SINKHORN_ITERS or not _run_time(jit, record)
    ]
    return tuple(dict.fromkeys(names))


def _head_libraries(jit: ModuleType, arch: str) -> tuple[str, ...]:
    """The ``build.json`` libraries of a head package's forward and backward on ``arch`` (one per role, any shape)."""
    return tuple(jit.library_name(jit.ROLES[role], arch) for role in ("head_fwd", "head_bwd"))


def libraries(arch: str, hidden: int, sinkhorn_iters: int) -> dict[str, tuple[str, ...]]:
    """Package -> the libraries any call of either variant, and of its head, may launch on ``arch`` at the model
    shape ``hidden`` / ``sinkhorn_iters``, whatever the token count. ``ValueError`` for a shape the packages do not
    serve on ``arch``.

    A head runs its forward and backward programs: :func:`mhc_head` never launches the package's backward that adds
    into an existing gradient (autograd adds the head's gradient to one that is already there)."""
    if isinstance(hidden, bool) or not isinstance(hidden, int):
        raise ValueError(f"hidden must be an int, not {hidden!r}")
    if isinstance(sinkhorn_iters, bool) or not isinstance(sinkhorn_iters, int):
        raise ValueError(f"sinkhorn_iters must be an int, not {sinkhorn_iters!r}")
    if not 1 <= sinkhorn_iters <= SINKHORN_ITERS_MAX:
        raise ValueError(
            f"the Cake mHC kernels run sinkhorn_iters from 1 to {SINKHORN_ITERS_MAX}, not {sinkhorn_iters}"
        )
    served = set(_det.cake_jit.supported_hidden_sizes(arch))
    served &= set(_fast.cake_jit.supported_hidden_sizes(arch))
    served &= set(_det_head.cake_jit.supported_hidden_sizes(arch))
    served &= set(_fast_head.cake_jit.supported_hidden_sizes(arch))
    if hidden not in served:
        raise ValueError(f"the Cake mHC kernels run the hidden sizes {tuple(sorted(served))} on {arch}, not {hidden}")
    return {
        "cake_mhc_det_train": _route_libraries(_det.cake_jit, arch, hidden, sinkhorn_iters),
        "cake_mhc_nondet_train": _route_libraries(_fast.cake_jit, arch, hidden, sinkhorn_iters),
        "cake_mhc_head_det_train": _head_libraries(_det_head.cake_jit, arch),
        "cake_mhc_head_nondet_train": _head_libraries(_fast_head.cake_jit, arch),
    }


def prebuild(*, device: Any = None, max_workers: int | None = None, hidden: int, sinkhorn_iters: int) -> list[Path]:
    """Build every library :func:`mhc_prepare` / :func:`mhc_finish` and :func:`mhc_head` (either variant) may launch
    at ``hidden`` and ``sinkhorn_iters`` on ``device`` (default: the current CUDA device), ``max_workers`` nvcc
    processes at a time.

    Both model parameters are required: a call at another Sinkhorn count launches programs a prebuild at the default
    count does not build. The epsilons select no library. Returns the cache path of every library, built now or
    before; a repeated call in a process builds nothing, and ranks starting at once build each library once (the
    loader's file lock). ``ValueError`` for a hidden size or count the device's architecture does not serve
    (:func:`libraries`), ``NotImplementedError`` for a device other than SM100a / SM103a / SM107a, ``RuntimeError``
    when the toolkit cannot build its libraries (SM107a needs CUDA 13.4 or newer).
    """
    device = _device(device)
    if device.type != "cuda":
        raise NotImplementedError(f"the Cake mHC kernels run on CUDA devices, not {device}")
    arch = loader.device_arch(device)
    names = libraries(arch, hidden, sinkhorn_iters)
    loader.check_toolkit(arch, "the Cake mHC kernels")
    key = (str(loader.cache_dir()), arch, hidden, sinkhorn_iters)
    paths = _built.get(key)
    if paths is None:
        paths = []
        for package, package_libraries in names.items():
            paths += loader.prebuild_libraries(package, package_libraries, max_workers)
        _built[key] = paths
    return list(paths)


def release() -> None:
    """Drop what the packages keep between calls: the edge stand-ins, the launch plans and the loaded libraries (a
    later call reloads them from the cache, without building). ``miles_ext_kernels.release()`` calls it after it has
    dropped the pool's buffers.

    Call it when no mHC kernel runs (before miles sleeps or offloads); a connection whose backward is still pending
    takes its workspace from the pool when it runs.
    """
    for package in (_det, _fast):
        for cache in (package._EDGES, package._PLANS, package._FAST_ROUTES):
            cache.clear()
    for package in (_det, _fast, _det_head, _fast_head):
        package.cake_jit.load.cache_clear()
    _built.clear()
