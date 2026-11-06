"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Packed causal depthwise conv1d + SiLU (kernel size 4) on the vendored Cake kernels (SM100a / SM103a / SM107a).

:func:`causal_conv1d` is the short convolution of miles' GDN and KDA layers: BF16 ``x`` ``[1, S, C]`` (or
``[S, C]``) holds the packed documents ``cu_seqlens`` delimits, ``weight`` ``[C, 4]`` the FP32 or BF16 taps
(``weight[:, 3]`` multiplies the current token), no bias::

    y[t, c] = bf16(silu(sum_k weight[c, k] * x[t - 3 + k, c]))      # x of another document is 0

``det`` runs FlashInfer's deterministic package (``_vendor/cake_causal_conv1d_det_train_backend``): its backward
writes per-CTA partial rows of the weight gradient and a second launch sums them in a fixed order, so the gradients
are bitwise reproducible for a given ``(S, C, device)``. ``fast`` runs the non-deterministic package
(``_vendor/cake_causal_conv1d_nondet_train``): the same forward programs (``y`` is ``det``'s bits) and a one-launch
backward whose ``dx`` is ``det``'s bits and whose weight gradient is summed across CTAs with unordered FP32 reductions
(FP64 under ``silu_grad_preactivation="fp32"``), within ``det``'s error budget. Both packages serve every route miles
uses (FP32 or BF16 taps, either SiLU' policy). The one call ``fast`` does not serve is a call while
``torch.use_deterministic_algorithms(True)`` is in effect (its backward refuses it): that call runs ``det`` and the
runtime logs it once.

Each call reads its variant (``miles_ext_kernels.cake.variant("conv1d")``: ``det`` until configured otherwise) and the
SiLU' policy (the option ``cake_conv1d_silu_grad_preactivation``, default ``bf16``) when it runs. Only ``configure``
selects the variant, so a process configured with ``deterministic_mode`` runs ``det`` on every call; the
``silu_grad_preactivation`` argument overrides the policy for one call. The backward runs the package and policy of
its forward. ``cu_seqlens=None`` (one document) builds its prefix sums on the device, with no host synchronization.

Context parallelism: :func:`causal_conv1d_kcp` convolves the fragments of the packed documents that one rank of a
context-parallel group holds, through the context-parallel (KCP) route both packages carry (``kcp.py``). The short
conv has no partition of its own: it takes the context object the GDN or KDA context-parallel route of the same layer
built for the rank (``conv_kernel_size=4``) and reads the rank's fragments, token count, halo plan and process group
from it. The three document tokens before every continuing fragment come from the ranks that hold them, one
``all_to_all_single`` per direction. With FP32 taps the forward's ``y`` is bitwise the single-GPU ``det`` ``y`` of the
whole pack on the rank's tokens. The backward evaluates SiLU' at the unrounded FP32 preactivation (the ``fp32``
policy's derivative; the ``silu_grad_preactivation`` setting does not apply), so ``dx`` is within one BF16 ulp of the
FP64 reference, and the weight gradient is the rank's FP32 partial over its own tokens: the caller's
context-parallel gradient reduction sums the ranks' partials. ``det`` sums the partial in a fixed order, so it repeats
bit for bit for a given context and device; ``fast`` keeps ``y`` and ``dx`` bitwise ``det``'s and sums the partial
with unordered reductions, within ``det``'s budget. The route takes FP32 taps only, so BF16 taps are widened (an exact
cast) and their gradient comes back rounded to BF16. :func:`causal_conv1d_kcp_multi` runs the convolutions of several
tensors of one layer (q, k and v) with one exchange per direction, bitwise the single calls.

The packages compile their libraries on first use through the runtime's loader; ``prebuild(weight_dtype=...)`` builds
every library a call may launch (both variants, both SiLU' policies, every channel count and token count, the
context-parallel route) before the first step. SM107a needs a CUDA 13.4 or newer toolkit.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader

__all__ = [
    "causal_conv1d",
    "causal_conv1d_kcp",
    "causal_conv1d_kcp_multi",
    "check_supported",
    "prebuild",
    "release",
]

OP = "conv1d"
# the vendored package of each variant
PACKAGES = {"det": "cake_causal_conv1d_det_train_backend", "fast": "cake_causal_conv1d_nondet_train"}
KERNEL_SIZE = 4
CHANNEL_MULTIPLE = 8
SILU_GRAD_PREACTIVATIONS = ("bf16", "fp32")
TAP_DTYPES = (torch.float32, torch.bfloat16)

_BACKEND = "--linear-attention-conv-backend cake"
_SILU_GRAD = "cake_conv1d_silu_grad_preactivation"
_KCP_MODULES: dict[str, ModuleType] = {}  # variant -> the kcp module its context-parallel calls run
# the host-cache and graph-cache clears of each variant's kcp module
_KCP_CLEARS = {
    "det": ("host_cache_clear", "graph_cache_clear"),
    "fast": (
        "cake_causal_conv1d_nondet_train_kcp_host_cache_clear",
        "cake_causal_conv1d_nondet_train_kcp_graph_cache_clear",
    ),
}


@functools.cache
def _package(variant: str) -> ModuleType:
    """The vendored package of ``variant``, its libraries built through the runtime's loader."""
    return loader.load_package(PACKAGES[variant])


def _kcp_module(variant: str) -> ModuleType:
    """The context-parallel host module (``kcp``) of ``variant``'s vendored package, imported by the variant's first
    context-parallel call."""
    module = _KCP_MODULES.get(variant)
    if module is None:
        module = _KCP_MODULES[variant] = import_module(f"{loader.load_package(PACKAGES[variant]).__name__}.kcp")
    return module


def _variant() -> str:
    """The configured variant; ``det`` for ``fast`` while torch's deterministic algorithms are enabled, which the
    non-deterministic backward refuses."""
    variant = config.variant(OP)
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        config.warn_runs_det(OP, "torch.use_deterministic_algorithms(True)")
        return "det"
    return variant


def _policy(silu_grad_preactivation: str | None) -> str:
    policy = config.option(OP, _SILU_GRAD) if silu_grad_preactivation is None else silu_grad_preactivation
    if policy not in SILU_GRAD_PREACTIVATIONS:
        raise ValueError(f"silu_grad_preactivation must be one of {SILU_GRAD_PREACTIVATIONS}, not {policy!r}")
    return policy


def _capability(device) -> tuple[int, int]:
    """The compute capability of ``device`` (default: the current CUDA device), checked to be supported and its
    kernels to build with the toolkit."""
    supported = tuple(loader.ARCH_BY_CAPABILITY)
    if not torch.cuda.is_available():
        raise ValueError(f"{_BACKEND} needs a CUDA device of compute capability {supported}")
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability not in loader.ARCH_BY_CAPABILITY:
        raise ValueError(
            f"{_BACKEND} runs on compute capability {supported} (SM100a / SM103a / SM107a), not {capability}; use "
            "--linear-attention-conv-backend fla"
        )
    loader.check_toolkit(loader.ARCH_BY_CAPABILITY[capability], "the Cake short-conv kernels")
    return capability


def check_supported(channels: int, kernel_size: int, activation: str, device=None, *, has_bias: bool = False) -> None:
    """Raise ``ValueError`` unless the Cake kernels run this short convolution on ``device`` (default: the current
    CUDA device): kernel size 4, SiLU, no bias, a channel count that is a multiple of 8, compute capability 10.0, 10.3
    or 10.7; ``RuntimeError`` when the toolkit cannot build the device's kernels (SM107a needs CUDA 13.4 or newer).

    ``has_bias``: whether the convolution adds a per-channel bias. The entry points take no bias, so a layer with one
    must not run them: it would lose its bias silently.
    """
    if kernel_size != KERNEL_SIZE:
        raise ValueError(f"{_BACKEND} needs kernel size {KERNEL_SIZE}, got {kernel_size}")
    if activation != "silu":
        raise ValueError(f"{_BACKEND} applies SiLU, got activation {activation!r}")
    if has_bias:
        raise ValueError(f"{_BACKEND} convolves without a bias; use --linear-attention-conv-backend fla")
    if channels % CHANNEL_MULTIPLE:
        raise ValueError(f"{_BACKEND} needs a channel count that is a multiple of {CHANNEL_MULTIPLE}, got {channels}")
    _capability(device)


def _libraries(package: ModuleType, arch: str, bf16_taps: bool) -> list[str]:
    """Every library of ``package`` a call may launch on ``arch``: the forward route of the taps' dtype, the backward
    routes of both SiLU' policies (each program with its fixed-order reduce program, if any), and every program of
    the context-parallel route (forward, head-row and main backward programs, both launch forms of the backward)."""
    jit = package.cake_jit
    names = []
    routes = [jit.forward_route(bias=False, bf16_taps=bf16_taps)]
    for fp32_grad in (False, True):
        routes.append(jit.backward_route(bias=False, fp32_grad=fp32_grad, bf16_taps_or_dweight=bf16_taps))
    for route in routes:
        for module in jit.route_modules(arch, route):
            names.append(jit.library_name(module, arch))
            reduce = jit.MODULES[module].get("reduce")  # the det backward's fixed-order reduce program
            if reduce:
                names.append(jit.library_name(reduce, arch))
    kcp_jit = import_module(".kcp_jit", package.__name__)
    for kind in kcp_jit.ROUTES[arch]:
        for module in kcp_jit.route_modules(arch, kind):
            names.append(kcp_jit.library_name(module, arch))
            reduce = kcp_jit.MODULES[module].get("reduce")  # a program of the single-rank registry
            if reduce:
                names.append(jit.library_name(reduce, arch))
    return names


def prebuild(*, device=None, max_workers: int | None = None, weight_dtype: torch.dtype) -> list[Path]:
    """Build every library a call may launch for taps of ``weight_dtype`` on ``device`` (default: the current CUDA
    device), whatever its variant, SiLU' policy, channel count, token count or route (with or without
    ``cu_seqlens``, single rank or context-parallel); returns their cache paths. Idempotent.

    ``ValueError`` for taps that are neither FP32 nor BF16, ``NotImplementedError`` for a device that is not SM100a,
    SM103a or SM107a, ``RuntimeError`` for a toolkit too old for the device.
    """
    if weight_dtype not in TAP_DTYPES:
        raise ValueError(f"{_BACKEND} takes FP32 or BF16 taps, got {weight_dtype}")
    if not torch.cuda.is_available():
        raise NotImplementedError(f"{_BACKEND} needs a CUDA device")
    arch = loader.device_arch(device)
    loader.check_toolkit(arch, "the Cake short-conv kernels")
    paths: list[Path] = []
    for variant in PACKAGES:
        package = _package(variant)
        names = _libraries(package, arch, weight_dtype == torch.bfloat16)
        paths += loader.prebuild_libraries(package.cake_build, names, max_workers)
    return paths


def release() -> None:
    """Drop the packages' host plans (single-rank rows, context-parallel host tables and graphs); the next call of
    each plans again. The libraries stay loaded."""
    _package("fast").plan_cache_clear()  # the det package keeps no plans
    for variant in PACKAGES:
        kcp = _KCP_MODULES.get(variant)
        if kcp is not None:
            for clear in _KCP_CLEARS[variant]:
                getattr(kcp, clear)()


def _one_document(S: int, device: torch.device) -> torch.Tensor:
    """The int32 prefix sums ``[0, S]`` of one document, built on ``device``: no host copy, no synchronization."""
    return torch.arange(2, dtype=torch.int32, device=device).mul_(S)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    silu_grad_preactivation: str | None = None,
) -> torch.Tensor:
    """Autograd-enabled ``silu(causal depthwise conv1d)`` of the packed documents of ``x``.

    ``x``: BF16 ``[1, S, C]`` or ``[S, C]`` (copied once when not contiguous); ``weight``: FP32 or BF16 ``[C, 4]``;
    ``cu_seqlens``: ``[num_docs + 1]`` prefix sums of the document lengths on ``x``'s device (``None``: one
    document, built on the device). Runs the configured variant; ``silu_grad_preactivation`` (``bf16`` / ``fp32``)
    defaults to the configured policy, read now. Returns BF16 ``y`` of ``x``'s shape; the weight gradient comes in
    ``weight``'s dtype.
    """
    if x.dim() == 3 and x.shape[0] != 1:
        raise ValueError(
            f"{_BACKEND} takes one packed batch [1, S, C] whose documents cu_seqlens delimits, got shape "
            f"{tuple(x.shape)}"
        )
    variant = _variant()
    policy = _policy(silu_grad_preactivation)
    x = x.contiguous()
    if cu_seqlens is None:
        cu_seqlens = _one_document(x.shape[-2], x.device)
    else:
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
    weight = weight.contiguous()
    package = _package(variant)
    if variant == "det":
        return package.cake_causal_conv1d_det(x, weight, cu_seqlens, None, silu_grad_preactivation=policy)
    return package.cake_causal_conv1d_nondet(x, weight, cu_seqlens, None, silu_grad_preactivation=policy)


def _kcp_taps(weight: torch.Tensor) -> torch.Tensor:
    """FP32 taps for the context-parallel route: BF16 taps widened (exact; autograd rounds their gradient back)."""
    if weight.dtype == torch.bfloat16:
        weight = weight.float()
    return weight.contiguous()


def causal_conv1d_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    kcp_context: Any,
    *,
    recompute_handle: Any = None,
    cooperative: bool | None = None,
) -> torch.Tensor:
    """Autograd-enabled ``silu(causal depthwise conv1d)`` of this rank's fragments under context parallelism.

    ``x_local``: BF16 ``[T_r, C]`` (or ``[1, T_r, C]``; copied once when not contiguous), the rank's fragments back to
    back in the order of the context's local ``cu_seqlens``. ``weight``: FP32 or BF16 ``[C, 4]`` taps, the same on
    every rank. ``kcp_context``: the rank's context of the layer's GDN or KDA context-parallel route, built with
    ``conv_kernel_size=4`` (this function reads its ``T_r``, ``n_frag``, ``cu_seqlens_local`` /
    ``cu_seqlens_local_cpu``, ``halo``, ``halo_tables``, ``group``, ``world_size`` and calls its ``wait_ready()`` when
    present). Every rank of the context's group calls this function for the same layers in the same order: the halo
    exchange is a collective.

    Runs the configured variant; ``fast`` runs ``det`` while ``torch.use_deterministic_algorithms(True)`` is in
    effect. ``recompute_handle``: ``None``, or an object with an ``entries`` dict (for example the recompute handle
    of the layer's GDN / KDA context-parallel call) that makes
    ``torch.utils.checkpoint``'s recompute of this call skip the forward exchange: the original forward stores its
    history rows under the key ``"conv"``, the recompute reuses them, bitwise. One handle per call and micro-step,
    created outside the checkpointed region; every rank passes one or none. ``cooperative``: the launch form of
    ``det``'s main backward program, ``None`` (the package default, one cooperative launch) or ``True``, or ``False``
    (an ordinary launch one SM short of co-resident and a fixed-order reduce program, which starts beside a kernel of
    another stream; its weight-gradient bits differ from the cooperative form's). ``fast``'s backward is one ordinary
    launch: it takes ``None`` or ``False``.

    Returns BF16 ``y`` of ``x_local``'s shape. The backward returns BF16 ``dx`` and the rank's FP32 weight-gradient
    partial (in ``weight``'s dtype).
    """
    variant = _variant()
    kcp = _kcp_module(variant)
    x_local = x_local.contiguous()
    weight = _kcp_taps(weight)
    if variant == "det":
        return kcp.cake_causal_conv1d_det_kcp(
            x_local, weight, kcp_context, recompute_handle=recompute_handle, cooperative=cooperative
        )
    return kcp.cake_causal_conv1d_nondet_kcp(
        x_local, weight, kcp_context, recompute_handle=recompute_handle, cooperative=cooperative
    )


def causal_conv1d_kcp_multi(
    xs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    kcp_context: Any,
    *,
    recompute_handle: Any = None,
    cooperative: bool | None = None,
) -> tuple[torch.Tensor, ...]:
    """The context-parallel convolutions of several tensors of one layer (for example its q, k and v) with one halo
    exchange per direction instead of one per tensor.

    ``xs``: ``n >= 1`` BF16 ``[T_r, C_i]`` tensors (the channel counts may differ), ``weights``: their ``[C_i, 4]``
    taps; the other arguments as in :func:`causal_conv1d_kcp`, one ``recompute_handle`` for the whole call. Every
    rank passes the same tensor count and channel counts. The outputs and ``dx`` are bitwise
    ``causal_conv1d_kcp(x, w, kcp_context)`` per tensor; so is every weight gradient under ``det``.
    """
    if isinstance(xs, torch.Tensor) or isinstance(weights, torch.Tensor):
        raise TypeError("xs and weights must be sequences of tensors (one per convolution), not a tensor")
    variant = _variant()
    kcp = _kcp_module(variant)
    xs = [x.contiguous() for x in xs]
    weights = [_kcp_taps(w) for w in weights]
    if variant == "det":
        return kcp.cake_causal_conv1d_det_multi_kcp(
            xs, weights, kcp_context, recompute_handle=recompute_handle, cooperative=cooperative
        )
    return kcp.cake_causal_conv1d_nondet_kcp_multi(
        xs, weights, kcp_context, recompute_handle=recompute_handle, cooperative=cooperative
    )
