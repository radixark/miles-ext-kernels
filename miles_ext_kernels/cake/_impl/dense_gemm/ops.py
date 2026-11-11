"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The dense projection GEMMs miles calls (SM100a / SM103a / SM107a), on the vendored FlashInfer packages
``cake_dense_gemm_det_train`` (``det``) and ``cake_dense_gemm_nondet_train`` (``fast``), whose libraries the runtime's
loader builds with nvcc and loads with tvm-ffi (with a CUDA 13.4 or newer toolkit).

K1 (``cake_dense_gemm_det_train``) computes ``out[l, m, n] = sum_k A[l, m, k] B[l, k, n]`` over strided BF16 views with
FP32 accumulation and one rounding at the store. Its launch plan, split-K and stream-K fixup order are a pure function
of the views' geometry and the device, so every call is bit-reproducible. It reads a view in place when the view starts
at a 16-byte aligned address, has unit stride on one matrix axis and its other strides in whole 16-byte units, and it
needs the output columns ``N`` to be a multiple of 8. :func:`batched_projection` hands it the token-major ``[T, H, D]``
activations and the per-head weight views as they are, and copies a view only when it breaks that contract (a weight
whose storage is not 16-byte aligned in its parameter buffer, a gradient that is not contiguous). The weight gradient
of a single-head ``out_in`` projection goes through the package's weight-gradient entry point on the ``[T, D]``
matrices, which runs the GEMM in the orientation the package gives the row (the transposed-store ``X.T @ G`` for a
narrow ``D_out`` such as ``4096 -> 32``).

K2 (``cake_dense_gemm_det_train_router``) is an FP32 GEMM through the exact three-way BF16 split of both operands:
the products of the splits accumulate with chunked FP32 promotion and the split-K partials are summed in a fixed order,
so it is bit-reproducible with FP32-class error (a relative Frobenius error of about 2e-7 against FP64).

Variants. A call runs the variant configured when its forward runs (``config.variant("dense_gemm")``); its backward
runs the variant of its forward. ``det`` runs the deterministic package. ``fast`` runs the non-deterministic package,
which has the deterministic package's operations and dtypes and binds it as its sibling
(``use_deterministic_backend``). Where its measured tables select the row for the device's architecture and SM count
and the token count, it runs a program of its own: a split-K, multi-chain, cluster or stream-K program for a K1 weight
gradient, a short-token SIMT or a K-order program for the router weight gradient, a short-token SIMT program for the
router input gradient, and an input-gradient program with ``det``'s bits for the router forward (a 2-D K1 GEMM of the
input-gradient layout). A split-K program adds its partials in the order they finish, so a gradient whose elements get
more than two partials is not bit-reproducible run to run; the other programs repeat their bits (those of the
gradients other bits than ``det``'s). On some rows it runs a program of the deterministic package on another plan than
``det``'s (a router weight gradient at another split count, a single-head weight gradient on another program): other
bits than ``det``'s, the same on every run. Every other forward and gradient has the bits of ``det``.
:func:`fast_projection_programs` and :func:`fast_router_programs` say which programs ``fast`` runs for a shape, a token
count and a device. While ``torch.use_deterministic_algorithms(True)`` is in effect (``warn_only=True`` included) a
``fast`` call runs ``det`` and the runtime logs it once: the non-deterministic package raises PyTorch's alert at its
order-varying launches then (a warning with ``warn_only=True``).

Historical accuracy misses of ``fast``. On 148-SM sm_103a parts (B300), the two-split K1 weight gradient of
the single-head ``5120 -> 48`` and ``5120 -> 64`` BF16 projections missed the
deterministic package's BF16 budget (atol = rtol = 1e-2, no element outside) for one of 10 seeds at 40961 and 40962
tokens (``5120 -> 48``) and at 47105 tokens (``5120 -> 64``), where the deterministic program met it on every seed.
These are the measurements of flashinfer-ai/flashinfer#8743. The current FlashInfer planner runs the deterministic
program for ``5120 -> 48`` BF16 at 40961 to 49151 tokens and for ``5120 -> 64`` BF16 at 47105 to 49152 tokens on these
parts. miles calls neither shape and runs ``det``.

Batch invariance. Neither variant is batch-invariant. The plan of a GEMM (its tile, its split-K or stream-K partition,
the swapped program of a few-token batched forward) is a function of the token count, so the same token's output,
logits and gradients can differ in their last bits between calls with different token counts, within the accuracy
budget. ``det`` repeats bit for bit for identical inputs of the same shape on the same device; it makes no promise
for a token packed with other tokens. A routing decision (the top-k of the router logits) can therefore depend on how
the tokens of a step are packed when two of its logits are that close.

Prebuild. :func:`prebuild` compiles every program the two packages register for the device's architecture for the
GEMM kinds it is given (``projections``: the BF16-output K1 programs, the swapped few-token forwards and the
non-deterministic K1 weight gradients; ``router``: the FP32-output K1 programs, the K2 programs of both packages and
the BF16x3 split). The plan of a call picks one of these programs per token count, so after :func:`prebuild` no call of
those GEMMs compiles on that device, at any token count and in either variant. The shapes are checked, not planned.

Memory. The vendored entry points plan once per (device, stream, view geometry) and keep a bounded cache of planned
launches; the cached launches of a stream share a growable workspace of that device and stream (calls on two streams
never share one), dropped by :func:`release`. Do not capture these calls in a CUDA graph: a later growth replaces, and
an eviction frees, the workspace a graph would replay.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader

__all__ = [
    "OPERATIONS",
    "WEIGHT_LAYOUTS",
    "batched_projection",
    "check_projection_dims",
    "fast_projection_programs",
    "fast_router_programs",
    "prebuild",
    "release",
    "router_gating",
]

# The orientation of the per-head weights of batched_projection: [H, D_in, D_out] or [H, D_out, D_in].
WEIGHT_LAYOUTS = ("in_out", "out_in")
# The three GEMMs of a projection or of the router, in the order fast_projection_programs / fast_router_programs report
OPERATIONS = ("forward", "input_grad", "weight_grad")

_OP = "dense_gemm"
_DET = "cake_dense_gemm_det_train"
_FAST = "cake_dense_gemm_nondet_train"
_VENDOR = "miles_ext_kernels.cake._vendor"


@functools.cache
def _vendored() -> tuple[ModuleType, ModuleType]:
    """``(cake_backend, cake_jit)`` of the deterministic package."""
    from miles_ext_kernels.cake._vendor.cake_dense_gemm_det_train import cake_backend, cake_jit

    return cake_backend, cake_jit


@functools.cache
def _vendored_fast() -> tuple[ModuleType, ModuleType]:
    """``(cake_backend, cake_jit)`` of the non-deterministic package, its deterministic sibling bound to the vendored
    deterministic package (it resolves its deterministic programs, plans and rules there)."""
    det_backend, _ = _vendored()
    from miles_ext_kernels.cake._vendor.cake_dense_gemm_nondet_train import cake_backend, cake_jit

    cake_backend.use_deterministic_backend(det_backend)
    return cake_backend, cake_jit


def _call_variant(variant: str) -> str:
    """``variant``, or ``det`` for ``fast`` while torch's deterministic algorithms are enabled (``warn_only=True``
    included: the non-deterministic package raises PyTorch's alert at its order-varying launches then)."""
    if variant == "fast" and torch.are_deterministic_algorithms_enabled():
        config.warn_runs_det(_OP, "calls under torch.use_deterministic_algorithms(True)")
        return "det"
    return variant


def _k1_gemm(variant: str) -> Callable:
    """The K1 entry point ``gemm(A, B, out)`` of ``variant``."""
    if variant == "det":
        return _vendored()[0].cake_dense_gemm_det_train_projection
    return _vendored_fast()[0].cake_dense_gemm_nondet_train_projection


def _k1_wgrad(variant: str) -> Callable:
    """The K1 weight-gradient entry point ``wgrad(G, X, out)`` of ``variant``: ``out[N, K] = G[T, N].T @ X[T, K]``.
    The deterministic package plans the GEMM in the orientation its measured rules or its small-``N`` rule give the
    row (``wgrad_views``: for a narrow ``N`` the transposed-store ``X.T @ G``); the non-deterministic package runs its
    stream-K program on a row of its stream-K table, else plans its split-K program on ``X.T @ G`` with the
    transposed store, and runs a program of the deterministic package on a row its planner does not take."""
    if variant == "det":
        return _vendored()[0].cake_dense_gemm_det_train_projection_wgrad
    return _vendored_fast()[0].cake_dense_gemm_nondet_train_projection_wgrad


def _device_index(device: Any) -> int:
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"the Cake dense projection GEMMs run on CUDA devices, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


@functools.cache
def _arch_of(index: int) -> str:
    """The architecture of device ``index``, checked to be supported and its programs to build with the toolkit."""
    arch = loader.device_arch(index)
    loader.check_toolkit(arch, "the Cake dense projection GEMMs")
    return arch


def _arch(device: Any = None) -> str:
    return _arch_of(_device_index(device))


def check_projection_dims(d_in: int, d_out: int) -> None:
    """Raise ``ValueError`` unless the K1 programs serve a per-head projection ``d_in -> d_out``: each is the output
    width of one of its GEMMs (the forward, the input gradient), so both must be positive multiples of 8."""
    for name, dim in (("input", d_in), ("output", d_out)):
        if dim < 8 or dim % 8:
            raise ValueError(
                f"the Cake batched projection needs per-head {name} features that are a positive multiple of 8, "
                f"got {dim}"
            )


def _check_router_dims(experts: int, hidden: int) -> None:
    if experts < 8 or hidden < 8 or experts % 8 or hidden % 8:
        raise ValueError(
            f"router_gating needs the expert count and the hidden size to be positive multiples of 8, got {experts} "
            f"and {hidden}"
        )


def _k1_view(t: torch.Tensor) -> torch.Tensor:
    """``t`` when the K1 programs read it in place (unit stride on its last axis, its other strides in whole 16-byte
    units, a 16-byte aligned start), else a contiguous copy."""
    if t.stride(-1) == 1 and all(stride % 8 == 0 for stride in t.stride()[:-1]) and t.data_ptr() % 16 == 0:
        return t
    return t.clone(memory_format=torch.contiguous_format)


def _k2_view(t: torch.Tensor) -> torch.Tensor:
    """The FP32 matrix ``t`` when the K2 programs read it in place (unit inner stride, 16-byte rows and start), else a
    contiguous copy."""
    if t.stride(1) == 1 and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0:
        return t
    return t.clone(memory_format=torch.contiguous_format)


# The three GEMMs of a projection, as K1 calls ``gemm(A, B, out)`` over [H, ...] views of the token-major tensors
# ``x [T, H, D_in]``, ``g`` / ``out [T, H, D_out]`` and of ``weight`` ([H, D_in, D_out] in_out, [H, D_out, D_in]
# out_in). The weight gradient is stored in the weight's own orientation. That of an in_out or multi-head projection
# is a K1 call in that orientation; that of a single-head out_in projection is the package's weight-gradient call
# ``wgrad(G [T, D_out], X [T, D_in], dw [D_out, D_in])`` (``_k1_wgrad``), which runs the transposed-store ``X.T @ G``
# where the package gives the row that orientation (a narrow ``D_out``, such as 4096 -> 32).
def _forward(gemm: Callable, x, weight, out, in_out: bool) -> None:
    gemm(x.transpose(0, 1), weight if in_out else weight.transpose(1, 2), out.transpose(0, 1))


def _dgrad(gemm: Callable, g, weight, dx, in_out: bool) -> None:
    gemm(g.transpose(0, 1), weight.transpose(1, 2) if in_out else weight, dx.transpose(0, 1))


def _wgrad(gemm: Callable, wgrad: Callable, x, g, dw, in_out: bool) -> None:
    if in_out:
        gemm(x.permute(1, 2, 0), g.transpose(0, 1), dw)
    elif dw.shape[0] == 1:
        wgrad(g[:, 0], x[:, 0], dw[0])
    else:
        gemm(g.permute(1, 2, 0), x.transpose(0, 1), dw)


class _BatchedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, in_out: bool, variant: str) -> torch.Tensor:
        x, weight = _k1_view(x), _k1_view(weight)
        d_out = weight.shape[2] if in_out else weight.shape[1]
        out = torch.empty(x.shape[0], x.shape[1], d_out, dtype=torch.bfloat16, device=x.device)
        _forward(_k1_gemm(variant), x, weight, out, in_out)
        ctx.save_for_backward(x, weight)
        ctx.in_out = in_out
        ctx.variant = variant
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x, weight = ctx.saved_tensors
        # The autograd engine may run this on a thread that has not used the CUDA runtime yet; the programs encode
        # their tensor maps through the driver API, which needs the device's primary context current on this thread.
        torch.cuda.current_stream(x.device).query()
        variant = _call_variant(ctx.variant)
        gemm = _k1_gemm(variant)
        g = _k1_view(grad)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
            _dgrad(gemm, g, weight, dx, ctx.in_out)
        if ctx.needs_input_grad[1]:
            dw = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
            _wgrad(gemm, _k1_wgrad(variant), x, g, dw, ctx.in_out)
        return dx, dw, None, None


def _in_out(weight_layout: str) -> bool:
    if weight_layout not in WEIGHT_LAYOUTS:
        raise ValueError(f"weight_layout must be one of {WEIGHT_LAYOUTS}, got {weight_layout!r}")
    return weight_layout == "in_out"


def batched_projection(x: torch.Tensor, weight: torch.Tensor, *, weight_layout: str) -> torch.Tensor:
    """The per-head projection ``x [T, H, D_in] -> [T, H, D_out]`` in BF16, differentiable in ``x`` and ``weight``.

    ``weight_layout`` is the orientation of the per-head weights: ``"in_out"`` is ``weight [H, D_in, D_out]``, the
    product ``torch.einsum("thd,hdm->thm", x, weight)`` (the MLA q-absorb projection); ``"out_in"`` is
    ``weight [H, D_out, D_in]``, ``torch.einsum("thm,hdm->thd", x, weight)`` (the value projection, DeepSeek-V4's
    output-group projection). Both may be views: ``x`` a slice of a wider head slot, ``weight`` a split of a
    parameter. The result is a new contiguous ``[T, H, D_out]`` tensor (the kernel writes its ``[H, T, D_out]`` view
    in place) and the gradient of ``weight`` keeps the weight's orientation.

    As cuBLAS computes the einsum, every element is an FP32 sum of exact BF16 products rounded once to BF16. Under
    ``det`` the sums run in a fixed order for the call's shape: the output and both gradients are bit-identical call to
    call (and not bitwise equal to cuBLAS). Under ``fast`` the output and the input gradient have ``det``'s bits; the
    weight gradient runs the non-deterministic package's program where :func:`fast_projection_programs` names one
    (single-head rows such as ``4096 -> 32``; a split-K program whose partials add in the order they finish, or a
    program that repeats its bits) and a program of the deterministic package elsewhere. The plan, and with it the
    last bits of a token's results, depends on the token count ``T``.
    """
    in_out = _in_out(weight_layout)
    if x.dim() != 3 or weight.dim() != 3:
        raise ValueError(
            f"batched_projection takes x [T, H, D_in] and a 3-D weight, got {tuple(x.shape)} and {tuple(weight.shape)}"
        )
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(f"batched_projection takes BF16 x and weight, got {x.dtype} and {weight.dtype}")
    heads, d_in = weight.shape[0], weight.shape[1 if in_out else 2]
    d_out = weight.shape[2 if in_out else 1]
    if (x.shape[1], x.shape[2]) != (heads, d_in):
        raise ValueError(
            f"x [T, H, D_in] = {tuple(x.shape)} does not match the {weight_layout} weight {tuple(weight.shape)} "
            f"(H = {heads}, D_in = {d_in})"
        )
    check_projection_dims(d_in, d_out)
    if weight.device != x.device:
        raise ValueError(f"x and weight must be on one device, got {x.device} and {weight.device}")
    _arch(x.device)
    return _BatchedProjection.apply(x, weight, in_out, _call_variant(config.variant(_OP)))


def _router_weight_operand(weight: torch.Tensor) -> torch.Tensor:
    """The router weight ``[E, H]`` as the K1 ``B`` operand of the logits ``x @ weight.T``: ``weight.T`` stored
    ``[H, E]`` with unit stride on ``E`` (N-major). K1 serves an FP32 output with a K-major ``A`` only for an N-major
    ``B`` (its ``kn`` programs; it registers no ``kk`` program with FP32 output), so the forward copies the weight
    (``E * H`` BF16 elements, small next to the activations) into that layout."""
    return weight.t().contiguous()


def _router_backward(variant: str) -> tuple[Callable, Callable]:
    """``(dgrad(g, w), wgrad(g, x))``: the K2 input and weight gradients of ``variant``."""
    if variant == "det":
        backend = _vendored()[0]
        return backend.cake_dense_gemm_det_train_router_dgrad, backend.cake_dense_gemm_det_train_router_wgrad
    backend = _vendored_fast()[0]
    return (
        backend.cake_dense_gemm_nondet_train_router_dgrad,
        backend.cake_dense_gemm_nondet_train_router_wgrad,
    )


class _RouterGating(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, variant: str) -> torch.Tensor:
        x2 = _k1_view(x.reshape(-1, x.shape[-1]))
        logits = torch.empty(x2.shape[0], weight.shape[0], dtype=torch.float32, device=x.device)
        _k1_gemm(variant)(x2, _router_weight_operand(weight), logits)
        ctx.save_for_backward(x2, weight)
        ctx.x_shape = x.shape
        ctx.variant = variant
        return logits.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x2, weight = ctx.saved_tensors
        torch.cuda.current_stream(x2.device).query()  # see _BatchedProjection.backward
        dgrad, wgrad = _router_backward(_call_variant(ctx.variant))
        g = _k2_view(grad.reshape(-1, grad.shape[-1]).float())
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = dgrad(g, weight.float()).to(x2.dtype).view(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            dw = wgrad(g, x2.float()).to(weight.dtype)
        return dx, dw, None


def router_gating(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 router logits ``x @ weight.T`` for BF16 ``x [..., H]`` and ``weight [E, H]``: ``[..., E]`` FP32.

    The computation of Megatron's ``RouterGatingLinearFunction`` with ``--moe-router-dtype fp32`` (no bias): the
    forward multiplies the BF16 operands with FP32 accumulation and output (K1), the backward computes
    ``dx = g @ weight.float()`` and ``dweight = g.T @ x.float()`` in FP32 (K2) and rounds them to the dtypes of ``x``
    and ``weight``. ``E`` and ``H`` must be multiples of 8. Under ``det`` every call is bit-reproducible call to call.
    Under ``fast`` the logits have ``det``'s bits; a gradient runs a program of the non-deterministic package where
    :func:`fast_router_programs` names one (256 experts at hidden 4096, 5120 or 6144 in measured token windows): a
    short-token SIMT program of the weight or input gradient, which stores every element once, or a K-order weight
    gradient, which adds a tile's partials in K order (other bits than ``det``'s, the same on every run). Elsewhere it
    runs the deterministic program, at ``det``'s split count except in the token windows where the package's tables
    give the weight gradient another (other bits, the same on every run). The plan, and with it the last bits of a
    token's logits, depends on the number of tokens of the call.
    """
    if weight.dim() != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"router_gating takes x [..., H] and weight [E, H], got {tuple(x.shape)} and {tuple(weight.shape)}"
        )
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(f"router_gating takes BF16 x and weight, got {x.dtype} and {weight.dtype}")
    _check_router_dims(*weight.shape)
    if weight.device != x.device:
        raise ValueError(f"x and weight must be on one device, got {x.device} and {weight.device}")
    _arch(x.device)
    return _RouterGating.apply(x, weight, _call_variant(config.variant(_OP)))


# ---------------------------------------------------------------------------------------------------------------
# What fast runs


def _check_part(arch: str, sm_count: int) -> None:
    if arch not in loader.ARCH_BY_CAPABILITY.values():
        raise ValueError(f"arch must be one of {sorted(loader.ARCH_BY_CAPABILITY.values())}, got {arch!r}")
    if sm_count < 2:
        raise ValueError(f"sm_count must be a device's SM count, got {sm_count}")


def _fast_table_template(a, b, out, *, transposed_out: bool, arch: str, sm_count: int) -> str | None:
    """The template the non-deterministic package's program tables name for the 2-D K1 call ``gemm(a, b, out)`` of
    BF16 operands on a device of ``arch`` with ``sm_count`` SMs, as its generic entry points read them before any
    planner (``_k1_table_call``): the forward (both operands K-major) in ``K1_FORWARD_PROGRAM_RULES_BY_SM``, the input
    gradient (``a`` K-major, ``b`` MN-major) in ``K1_DGRAD_PROGRAM_RULES_BY_SM`` then ``K1_DGRAD_NVJET_RULES_BY_SM``,
    the transposed-store weight gradient ``X.T @ G`` in ``K1_WGRAD_SK_RULES_BY_SM``. ``None``: no row, a view those
    programs do not take (a batched view, an operand or ``out`` without a unit stride on its last axis), or a
    template the package does not register for ``arch``."""
    backend, jit = _vendored_fast()
    if a.dim() != 2 or b.dim() != 2 or out.dim() != 2 or out.stride(1) != 1:
        return None
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        return None
    dtype = "f32" if out.dtype == torch.float32 else "bf16"
    a_k, b_k = a.stride(1) == 1, b.stride(0) == 1
    if a_k and b_k and not transposed_out and out.dtype == torch.bfloat16:
        # X = a [T, K], W = b.T [N, K]: (K, N, T)
        key = (a.shape[1], b.shape[1], a.shape[0])
        tables = ("K1_FORWARD_PROGRAM_RULES_BY_SM",)
    elif a_k and not b_k and b.stride(1) == 1 and not transposed_out:
        # G = a [T, N], W = b [N, K]: (K, N, T, out dtype)
        key = (b.shape[1], b.shape[0], a.shape[0], dtype)
        tables = ("K1_DGRAD_PROGRAM_RULES_BY_SM", "K1_DGRAD_NVJET_RULES_BY_SM")
    elif not a_k and a.stride(0) == 1 and b.stride(1) == 1 and transposed_out:
        # X = a.T [T, d_in], G = b [T, d_out]: (d_in, d_out, T, out dtype)
        key = (a.shape[0], b.shape[1], b.shape[0], dtype)
        tables = ("K1_WGRAD_SK_RULES_BY_SM",)
    else:
        return None
    for table in tables:
        template = getattr(backend, table).get((arch, sm_count), {}).get(key)
        if template is not None and template in jit.KERNELS.get(arch, {}):
            return template
    return None


def _fast_k1_template(a, b, out, *, transposed_out: bool = False, arch: str, sm_count: int) -> str | None:
    """The template of the non-deterministic package's program that ``fast`` launches for the K1 call
    ``gemm(a, b, out)`` (``transposed_out``: the transposed store, ``out`` is ``[N, M]``) on a device of ``arch`` with
    ``sm_count`` SMs, or ``None``: the call runs a program of the deterministic package. These are the package's own
    program tables (:func:`_fast_table_template`), then its planner (``plan_atomic_gemm``), which its launches call
    with the device's architecture and SM count; the planner plans only the weight gradients (both operands MN-major)
    of the rows its tables select."""
    template = _fast_table_template(a, b, out, transposed_out=transposed_out, arch=arch, sm_count=sm_count)
    if template is not None:
        return template
    backend = _vendored_fast()[0]
    plan, *_ = backend.plan_atomic_gemm(a, b, out, transposed_out=transposed_out, arch=arch, sm_count=sm_count)
    return None if plan is None else plan.template


def _fast_k2_template(a, b, out, *, arch: str, sm_count: int) -> str | None:
    """As :func:`_fast_k1_template` for the K2 call ``out = a @ b`` (FP32 views; ``plan_atomic_router``): only the
    router weight gradient and input gradient of the rows and token windows the package's tables select get a
    program (the input gradient only in its short-token SIMT windows)."""
    plan, _ = _vendored_fast()[0].plan_atomic_router(a, b, out, arch=arch, sm_count=sm_count)
    return None if plan is None else plan.template


def _fast_router_wgrad_splits(
    num_experts: int, hidden_size: int, *, tokens: int, arch: str, sm_count: int
) -> int | None:
    """The split count at which ``fast`` runs the deterministic program of the router weight gradient where the
    non-deterministic package's tables give it another count than ``det``'s (``router_det_splits``: other bits than
    ``det``'s, the same on every run), else ``None``."""
    splits = _vendored_fast()[0].router_det_splits(
        "nn_t", arch=arch, M=hidden_size, N=num_experts, K=tokens, sm_count=sm_count
    )
    det = _vendored()[0].router_splits("nn_t", hidden_size, num_experts, tokens, arch, sm_count)
    return None if splits in (None, det) else splits


def _meta(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device="meta")


def _projection_views(num_heads: int, d_in: int, d_out: int, tokens: int, in_out: bool):
    """``(x, g, weight)``: ``meta`` tensors of a projection's activation, output (gradient) and weight."""
    weight = _meta(num_heads, d_in, d_out) if in_out else _meta(num_heads, d_out, d_in)
    return _meta(tokens, num_heads, d_in), _meta(tokens, num_heads, d_out), weight


def _router_views(num_experts: int, hidden_size: int, tokens: int):
    """The views the router's three GEMMs launch: ``((A, B, out) of the K1 forward, of the K2 input gradient, of the
    K2 weight gradient)`` as ``meta`` tensors (the weight gradient as the package's
    ``cake_dense_gemm_nondet_train_router_wgrad`` launches it)."""
    fp32 = torch.float32
    forward = (_meta(tokens, hidden_size), _meta(hidden_size, num_experts), _meta(tokens, num_experts, dtype=fp32))
    g, x = _meta(tokens, num_experts, dtype=fp32), _meta(tokens, hidden_size, dtype=fp32)
    dgrad = (g, _meta(num_experts, hidden_size, dtype=fp32), _meta(tokens, hidden_size, dtype=fp32))
    wgrad = (g.t(), x, _meta(num_experts, hidden_size, dtype=fp32))
    return forward, dgrad, wgrad


def fast_projection_programs(
    num_heads: int, d_in: int, d_out: int, *, weight_layout: str, tokens: int, arch: str, sm_count: int
) -> dict[str, str | None]:
    """Which program :func:`batched_projection` runs under ``fast`` for ``num_heads`` heads of ``d_in -> d_out`` at
    ``tokens`` tokens on a device of ``arch`` (``sm_100a`` / ``sm_103a`` / ``sm_107a``) with ``sm_count`` SMs:
    ``{"forward": ..., "input_grad": ..., "weight_grad": ...}``, each the template of the non-deterministic package's
    program, or ``None`` where it runs a program of the deterministic package (``det``'s plan and bits, except on a row
    of the package's ``K1_WGRAD_DET_RULES_BY_SM``, which runs another program of that package: other bits, the same on
    every run). The weight gradient of a single-head ``out_in`` projection is planned as the package's weight-gradient
    entry point plans it (its stream-K table, then the transposed-store ``X.T @ G``). Needs torch, no device: the
    package's tables and planner are a function of the views' geometry, the architecture, the SM count and the token
    count."""
    in_out = _in_out(weight_layout)
    check_projection_dims(d_in, d_out)
    _check_part(arch, sm_count)
    templates = []

    def plan(a, b, out, transposed_out=False):
        templates.append(_fast_k1_template(a, b, out, transposed_out=transposed_out, arch=arch, sm_count=sm_count))

    def plan_wgrad(g2, x2, out):
        # the package's weight-gradient entry point plans its program on the transposed-store X.T @ G
        plan(x2.t(), g2, out, transposed_out=True)

    x, g, weight = _projection_views(num_heads, d_in, d_out, tokens, in_out)
    _forward(plan, x, weight, g, in_out)
    _dgrad(plan, g, weight, x, in_out)
    _wgrad(plan, plan_wgrad, x, g, weight, in_out)
    return dict(zip(OPERATIONS, templates))


def fast_router_programs(
    num_experts: int, hidden_size: int, *, tokens: int, arch: str, sm_count: int
) -> dict[str, str | None]:
    """Which program :func:`router_gating` runs under ``fast`` for ``num_experts`` experts of ``hidden_size`` at
    ``tokens`` tokens on a device of ``arch`` with ``sm_count`` SMs, as :func:`fast_projection_programs` reports it. A
    ``None`` weight gradient runs the deterministic program, at ``det``'s split count except in the token windows where
    the package's tables give it another (other bits, the same on every run)."""
    _check_part(arch, sm_count)
    forward, dgrad, wgrad = _router_views(num_experts, hidden_size, tokens)
    return {
        "forward": _fast_k1_template(*forward, arch=arch, sm_count=sm_count),
        "input_grad": _fast_k2_template(*dgrad, arch=arch, sm_count=sm_count),
        "weight_grad": _fast_k2_template(*wgrad, arch=arch, sm_count=sm_count),
    }


# ---------------------------------------------------------------------------------------------------------------
# prebuild and release


def program_kind(template: str) -> str:
    """The GEMM kind a registered program of either package serves.

    ``"router"``: the K2 programs (``router_fp32_*``), the BF16x3 split (``split_fp32_*``) and the K1 programs with an
    FP32 output (``_f32``; the router's forward is the only FP32-output K1 GEMM of these entry points). The K1
    programs are the ``dense_proj_*`` ones (the non-deterministic package's split-K, cluster, one-wave forward,
    input-gradient, stream-K and nvjet programs) and the deterministic package's three-chain weight gradients
    (``chain_det_gemm_*``).
    ``"projections"``: the other K1 programs (BF16 output) and the swapped few-token forwards (``swap_ab_*``).
    ``ValueError`` for a template of no known kind.
    """
    if template.startswith(("router_fp32_", "router_partials_", "split_fp32_")):
        return "router"
    k1 = (
        "dense_proj_det_gemm_",
        "dense_proj_gemm_atomic_",
        "dense_proj_gemm_dsm_",
        "dense_proj_gemm_wave_",
        "dense_proj_gemm_dgrad_",
        "dense_proj_gemm_sk_",
        "dense_proj_gemm_nvjet_",
        "chain_det_gemm_",
    )
    if template.startswith(k1):
        return "router" if "f32" in template.split("_") else "projections"
    if template.startswith("swap_ab_"):
        return "projections"
    raise ValueError(f"no GEMM kind for the dense projection GEMM program template {template!r}")


def _registries() -> tuple[tuple[str, ModuleType], ...]:
    """``(package, cake_jit)`` of both packages: their registered programs per architecture (``KERNELS``)."""
    from miles_ext_kernels.cake._vendor.cake_dense_gemm_det_train import cake_jit as det_jit
    from miles_ext_kernels.cake._vendor.cake_dense_gemm_nondet_train import cake_jit as fast_jit

    return (_DET, det_jit), (_FAST, fast_jit)


def programs_to_prebuild(arch: str, kinds: Iterable[str]) -> dict[str, list[str]]:
    """``{package: [library, ...]}``: the ``build.json`` entries of every program of ``kinds`` (see
    :func:`program_kind`) that the two packages register for ``arch``."""
    kinds = set(kinds)
    return {
        package: sorted(
            cake_jit.library_name(module, arch)
            for template, module in cake_jit.KERNELS.get(arch, {}).items()
            if program_kind(template) in kinds
        )
        for package, cake_jit in _registries()
    }


def _checked(name: str, items: Iterable[Any], size: int, check: Callable[..., None]) -> list[tuple]:
    checked = []
    for item in items:
        item = tuple(item)
        if len(item) != size:
            raise ValueError(f"{name} takes tuples of {size} values, got {item!r}")
        check(*item)
        checked.append(item)
    return checked


def _check_projection(num_heads: int, d_in: int, d_out: int, weight_layout: str) -> None:
    if num_heads < 1:
        raise ValueError(f"a projection needs at least one head, got {num_heads}")
    check_projection_dims(d_in, d_out)
    _in_out(weight_layout)


def prebuild(
    *,
    device: Any = None,
    max_workers: int | None = None,
    projections: Iterable[tuple[int, int, int, str]] = (),
    router: Iterable[tuple[int, int]] = (),
) -> list[Path]:
    """Compile every program :func:`batched_projection` of ``projections`` and :func:`router_gating` of ``router`` may
    run on ``device`` (default: the current device), in either variant and at any token count.

    ``projections`` lists ``(num_heads, d_in, d_out, weight_layout)`` and ``router`` lists ``(num_experts,
    hidden_size)``. The plan of a call is a function of the token count, so the programs are not planned per shape:
    every program of the GEMM kinds given that the two packages register for the device's architecture is built (or
    found in the cache), ``max_workers`` at a time. The shapes are checked (``ValueError``) so a shape the GEMMs refuse
    fails here. Returns the cache paths; nothing when both lists are empty.
    """
    kinds = [
        kind
        for kind, items in (
            ("projections", _checked("projections", projections, 4, _check_projection)),
            ("router", _checked("router", router, 2, _check_router_dims)),
        )
        if items
    ]
    if not kinds:
        return []
    arch = _arch(device)
    paths: list[Path] = []
    for package, names in programs_to_prebuild(arch, kinds).items():
        paths += loader.prebuild_libraries(package, names, max_workers)
    return paths


def release() -> None:
    """Drop the planned launches and workspaces of both vendored packages."""
    for package in (_FAST, _DET):
        backend = sys.modules.get(f"{_VENDOR}.{package}.cake_backend")
        if backend is not None:
            getattr(backend, f"{package}_convenience_cache_clear")()
