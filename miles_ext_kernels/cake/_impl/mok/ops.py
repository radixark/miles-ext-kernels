"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

MoK (Mixture of Kittens) expert-parallel MoE training on the vendored Cake package ``cake_mok_bf16``.

One MoE layer of ``E = ep_size * local_experts`` routed experts and one shared expert, each a SwiGLU MLP: for a BF16
token ``x`` routed to the experts ``ids[t, :topk]`` with positive FP32 scores ``scores[t, :topk]``::

    expert(x) = down @ bf16(silu(gate @ x) * (up @ x))       # clamped: silu(min(g, L)) * clamp(u, -L, L)
    y[t]      = shared(x[t]) + sum_k scores[t, k] * expert_{ids[t, k]}(x[t])

:func:`prepare_mok_bf16` returns the adapter (``build_schedule``, ``forward``, ``recompute_forward_context``,
``backward``) and :func:`create_mok_bf16_workspace` its collectively created ``(config, workspace)``. Every rank of
the expert-parallel group holds its ``local_experts`` routed experts and its own source tokens (counts may differ,
including zero); the fused kernels dispatch the routed rows to their owners, run the expert GEMMs and SwiGLU, combine
the outputs and, in backward, return ``dx``, the score gradients (``dot(d_hidden, hidden) / score``), the routed gate /
up / down weight gradients and the shared ones (this rank's tokens only), in BF16, or added into six caller-owned FP32
accumulators (``fp32_wgrad``). Routed experts run in BF16, or in MXFP8 with weights prequantized by
:func:`quantize_mok_mxfp8_weights` after every weight update; the shared expert stays BF16. Scheduler layouts
``(ep_size, local_experts)``: (1, 4) and (4, 4), 256 routed experts at EP 4, 8, 16, 32 or 64, 288 at EP 8 or 32;
top-k 2 or 8; hidden and intermediate widths multiples of 256 (MXFP8: hidden a multiple of 512).

These are the semantics of FlashInfer's ``flashinfer.mok``: its package is vendored byte-identical (its README holds
the full contract) and builds its libraries through the runtime's loader (nvcc and tvm-ffi, no FlashInfer). The op has
one variant, ``det``: the forward and backward are a function of the inputs (the routed weight gradients are summed
over the macro-batches in a fixed order; the kernels' atomics are integer flags and counters), so the outputs repeat
bit for bit run to run, whatever communication-SM split each rank launches with: a fixed one, or the one the
workspace's imbalance-aware split picks per rank from the latest load observation that has completed.

Runtime needs: an initialized ``torch.distributed`` NCCL group over the expert-parallel ranks, its CUDA device current;
PyTorch symmetric memory with multicast (``torch.distributed._symmetric_memory``), so every rank of the group sits in
one peer-accessible NVLink domain (one NVL72 domain may span hosts); tvm-ffi. Triton (the ``mok`` extra) runs the
weight quantizer; without it PyTorch operations write the same bytes. Every architecture needs a CUDA 13.4 or newer
toolkit. The MXFP8 kernels convert BF16 pairs to E4M3 with ``cvt.rn.satfinite.e4m3x2.bf16x2``; their sources keep the
FP32 form of the same conversion for an nvcc older than CUDA 13.1, which the runtime's CUDA 13.4 floor never uses.

:func:`prebuild` builds the libraries of a model before its first call: every scheduler layout, top-k, SwiGLU form,
weight-gradient precision and routed precision the model parameters leave open, the BF16 context-free forward, the
context recompute and every token count; with MXFP8 it also compiles the weight quantizer's Triton kernel for the routed
weight shape. Prebuild before
CUDA graph capture: an adapter loads its libraries when it is created (and in ``prepare`` / ``prepare_recompute``).
"""

from __future__ import annotations

import functools
import json
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import loader

__all__ = [
    "NoMulticastError",
    "create_mok_bf16_workspace",
    "libraries",
    "prebuild",
    "prepare_mok_bf16",
    "quantize_mok_mxfp8_weights",
    "release",
    "roles",
]

PACKAGE = "cake_mok_bf16"
WHAT = "the Cake MoK kernels"
# the MXFP8 routed kernels need a hidden width that is a multiple of this; every other width a multiple of 256
_MXFP8_HIDDEN_MULTIPLE = 512
_WIDTH_MULTIPLE = 256


class NoMulticastError(RuntimeError):
    """A workspace of more than one rank got no multicast address from PyTorch symmetric memory: the platform gives
    the group's GPUs no NVLink multicast (they are not in one NVLink domain that supports it), which every
    multi-rank kernel launch needs."""


def _vendored(module: str) -> ModuleType:
    """A module of the vendored package, which builds its libraries through the runtime's loader."""
    return import_module(f"{loader.load_package(PACKAGE).__name__}.{module}")


def _api() -> ModuleType:
    return _vendored("api")


@functools.cache
def _sources() -> dict[str, dict[str, Any]]:
    """``sources.json`` of the vendored package: kernel role -> ``{"library", "arches", "arguments"}``."""
    path = Path(loader.load_package(PACKAGE).__file__).parent / "sources.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _one_of(name: str, value: Any, every: tuple) -> tuple:
    """``every`` when ``value`` is ``None``, else ``(value,)``; ``ValueError`` for a value not in ``every``."""
    if value is None:
        return every
    if type(value) is not type(every[0]) or value not in every:
        raise ValueError(f"{name} must be one of {every} or None, not {value!r}")
    return (value,)


def roles(
    *,
    ep_size: int | None = None,
    local_experts: int | None = None,
    topk: int | None = None,
    mxfp8: bool = True,
    clamped_swiglu: bool | None = None,
    fp32_wgrad: bool | None = None,
) -> list[str]:
    """The kernel roles (``sources.json`` keys) a model may launch: its scheduler layout ``(ep_size, local_experts)``,
    ``topk``, SwiGLU form and weight-gradient precision, each of them every value when ``None``; the MXFP8 routed
    kernels unless ``mxfp8`` is false (an adapter loads the BF16 ones when it is created, so they are always in). Every
    form includes the context recompute; the BF16 forward its context-free program (``forward[_clamped]_noctx``,
    which ``forward(..., save_context=False)`` runs), and the MXFP8 forward its multimacro program
    (``forward_mxfp8[_clamped]_multimacro``), which the package loads beside it on SM107a, the one architecture it is
    built for. ``ValueError`` for a value the kernels do not serve."""
    kernels = _vendored("_kernels")
    if (ep_size is None) != (local_experts is None):
        raise ValueError("pass ep_size and local_experts together (one scheduler layout), or neither")
    if ep_size is None:
        layouts = kernels.SCHEDULER_LAYOUTS
    elif type(ep_size) is not int or type(local_experts) is not int:
        raise ValueError(f"ep_size and local_experts must be ints, not {ep_size!r} and {local_experts!r}")
    else:
        layouts = _one_of("(ep_size, local_experts)", (ep_size, local_experts), kernels.SCHEDULER_LAYOUTS)
    topks = _one_of("topk", topk, kernels.EPILOGUE_TOPKS)
    forms = _one_of("clamped_swiglu", clamped_swiglu, (False, True))
    wgrads = _one_of("fp32_wgrad", fp32_wgrad, (False, True))
    if type(mxfp8) is not bool:
        raise ValueError(f"mxfp8 must be a bool, not {mxfp8!r}")
    names = ["zero", "expert_layout", "gather", "barrier"]
    for ep, local in layouts:
        names += [f"count_{ep}_{local}", f"rows_{ep}_{local}", f"pad_{ep}"]
    names += [f"epilogue_{direction}_{k}" for k in topks for direction in ("forward", "backward")]
    for precision in ("", "_mxfp8") if mxfp8 else ("",):
        for clamped in forms:
            forward = kernels._role(f"forward{precision}", clamped)
            names += [forward, kernels._role(f"recompute{precision}", clamped)]
            names.append(forward + ("_multimacro" if precision else "_noctx"))
            names += [kernels._role(f"backward{precision}", clamped, wgrad) for wgrad in wgrads]
    return list(dict.fromkeys(names))


def libraries(
    arch: str,
    *,
    ep_size: int | None = None,
    local_experts: int | None = None,
    topk: int | None = None,
    mxfp8: bool = True,
    clamped_swiglu: bool | None = None,
    fp32_wgrad: bool | None = None,
) -> list[str]:
    """The ``build.json`` libraries of :func:`roles` on ``arch`` (``cake_mok_<role>_<arch>``); with every parameter
    unset, every library of the architecture."""
    sources = _sources()
    selected = roles(
        ep_size=ep_size,
        local_experts=local_experts,
        topk=topk,
        mxfp8=mxfp8,
        clamped_swiglu=clamped_swiglu,
        fp32_wgrad=fp32_wgrad,
    )
    return [f"{sources[role]['library']}_{arch}" for role in selected if arch in sources[role]["arches"]]


def _check_widths(hidden_size: int | None, intermediate_size: int | None, mxfp8: bool) -> None:
    widths = (
        ("hidden_size", hidden_size, _MXFP8_HIDDEN_MULTIPLE if mxfp8 else _WIDTH_MULTIPLE),
        ("intermediate_size", intermediate_size, _WIDTH_MULTIPLE),
    )
    for name, value, multiple in widths:
        if value is None:
            if mxfp8:
                raise ValueError(
                    f"prebuild('mok') needs {name} for MXFP8 routed experts: the weight quantizer's Triton kernel is "
                    "compiled for the routed weight shape (pass mxfp8=False for BF16 routed experts only)"
                )
        elif type(value) is not int or value <= 0 or value % multiple:
            raise ValueError(f"{name} must be a positive multiple of {multiple}, not {value!r}")


def prebuild(
    *,
    device: Any = None,
    max_workers: int | None = None,
    ep_size: int | None = None,
    local_experts: int | None = None,
    topk: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    mxfp8: bool = True,
    clamped_swiglu: bool | None = None,
    fp32_wgrad: bool | None = None,
) -> list[Path]:
    """Build every library a model may launch on the architecture of ``device`` (default: the current one), and with
    ``mxfp8`` compile the weight quantizer's Triton kernel on ``device``; returns the libraries' cache paths.

    The model: its scheduler layout ``(ep_size, local_experts)``, ``topk``, SwiGLU form (``clamped_swiglu``) and
    weight-gradient precision (``fp32_wgrad``), each of them every value the kernels serve when unset; ``mxfp8``
    (default true) whether its routed experts may run in MXFP8, which needs ``hidden_size`` and ``intermediate_size``.
    Covered whatever the token counts: the BF16 and MXFP8 forward, the BF16 context-free forward, the context
    recompute, the backward, the scheduler and the epilogues. With the layout, ``topk``, ``clamped_swiglu`` and
    ``fp32_wgrad`` unset that is all 50 libraries of the architecture (52 on SM107a); with the four set, 16 (17 on
    SM107a; 13 without MXFP8). ``ValueError`` for parameters the kernels do not serve, ``NotImplementedError`` for
    another GPU, ``RuntimeError`` for a toolkit older than CUDA 13.4.
    """
    selection = dict(
        ep_size=ep_size,
        local_experts=local_experts,
        topk=topk,
        mxfp8=mxfp8,
        clamped_swiglu=clamped_swiglu,
        fp32_wgrad=fp32_wgrad,
    )
    roles(**selection)
    _check_widths(hidden_size, intermediate_size, mxfp8)
    arch = loader.device_arch(device)
    loader.check_toolkit(arch, WHAT)
    paths = loader.prebuild_libraries(PACKAGE, libraries(arch, **selection), max_workers)
    if mxfp8 and _vendored("mxfp8").triton is not None:
        # Triton specializes the kernel on the widths, not on the expert count: one expert of each shape suffices
        with torch.cuda.device(device):
            options = dict(dtype=torch.bfloat16, device="cuda")
            gate = torch.zeros((1, intermediate_size, hidden_size), **options)
            quantize_mok_mxfp8_weights(gate, gate, torch.zeros((1, hidden_size, intermediate_size), **options))
    return paths


def release() -> None:
    """Drop what the vendored package caches for the process: its loaded kernel modules and per-device constants.

    Adapters, workspaces and captured graphs belong to the caller and keep what they hold; a new adapter loads its
    libraries again from the cache, without a build.
    """
    package = loader.load_package(PACKAGE).__name__
    jit = sys.modules.get(f"{package}.jit")
    if jit is not None:
        jit._load.cache_clear()
    mxfp8 = sys.modules.get(f"{package}.mxfp8")
    if mxfp8 is not None:
        mxfp8._scale_factor.cache_clear()


def prepare_mok_bf16(*, ep_size=16, local_experts=16, topk=8, clamped_swiglu=False, fp32_wgrad=False):
    """The MoK adapter of an expert-parallel group of ``ep_size`` ranks with ``local_experts`` routed experts each and
    ``topk`` experts per token, on the current CUDA device.

    Loads the BF16 forward and backward kernels of the SwiGLU form ``clamped_swiglu`` selects now (``swiglu_limit``
    stays a per-call value) and the others on first use: ``prebuild`` the model, or call the adapter's
    ``prepare(swiglu_limit, mxfp8)`` and ``prepare_recompute(swiglu_limit, mxfp8)`` for every other form, before CUDA
    graph capture; ``prepare(swiglu_limit, save_context=False)`` also loads the context-free forward that
    ``forward(..., save_context=False)`` runs for an activation-checkpointed step (BF16 routed experts: ``y`` bit for
    bit the context forward's, no context; ``recompute_forward_context`` rebuilds it before ``backward``). With
    ``fp32_wgrad``, ``backward`` adds every weight gradient into the six FP32 ``weight_grad_accumulators`` it is passed
    and returns them. A layout or top-k the kernels do not serve raises ``ValueError``.
    """
    return _api().prepare_mok_bf16(
        ep_size=ep_size,
        local_experts=local_experts,
        topk=topk,
        clamped_swiglu=clamped_swiglu,
        fp32_wgrad=fp32_wgrad,
    )


def create_mok_bf16_workspace(*, group, device, num_local_tokens, hidden_size, topk, **options):
    """Collectively create the ``(config, workspace)`` of the expert-parallel ``group``: symmetric-memory buffers
    sized for every rank's ``num_local_tokens`` (which may differ, including zero), on ``device``.

    ``options`` are the keyword arguments of FlashInfer's ``create_mok_bf16_workspace``, with its defaults:
    ``source_capacity``, ``fwd_num_comm_sms`` / ``bwd_num_comm_sms``, ``minibatch_size``, ``macrobatch_size``,
    ``schedule_capacity_multiplier``, ``precision``, ``imbalance_comm_sms``, ``imbalance_threshold``. Call it on every
    rank, outside CUDA graph capture. :class:`NoMulticastError` (a ``RuntimeError``) when the group has more than one
    rank and PyTorch symmetric memory gives it no multicast address (its ranks are not in one NVLink domain that
    supports multicast).
    """
    config, workspace = _api().create_mok_bf16_workspace(
        group=group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        **options,
    )
    storage = workspace.storage
    if storage.ep_size > 1 and not (
        storage.all_gather_top_experts_buffer_multicast_ptr and storage.barrier_buffer_multicast_ptr
    ):
        raise NoMulticastError(
            f"{WHAT} need symmetric-memory multicast across the {storage.ep_size} expert-parallel ranks, but PyTorch "
            "symmetric memory gave the group no multicast address: every rank must sit in one peer-accessible NVLink "
            "domain with multicast support"
        )
    return config, workspace


def _dense(weight: Any) -> Any:
    """``weight`` when it is contiguous at a 16-byte aligned address (every PyTorch allocation is), else a contiguous
    copy, so the quantizer runs the kernel :func:`prebuild` compiled; the bytes it writes do not depend on the
    layout."""
    if isinstance(weight, torch.Tensor) and not (weight.is_contiguous() and weight.data_ptr() % 16 == 0):
        return weight.clone(memory_format=torch.contiguous_format)
    return weight


def quantize_mok_mxfp8_weights(gate, up, down):
    """``(forward_weights, backward_weights)`` of the BF16 routed weights ``gate`` / ``up`` ``[E_local, I, H]`` and
    ``down`` ``[E_local, H, I]`` for MXFP8 routed experts: E4M3 data with UE8M0 scales per 32 elements along the
    reduction dimension, in the tensor-core scale tiles, in both operand orientations. Pass ``forward_weights`` as the
    routed weights of ``forward`` (its first two to ``recompute_forward_context``) and ``backward_weights`` to
    ``backward``; requantize after every weight update. CUDA tensors are quantized without a host synchronization (one
    Triton kernel per weight, compiled by :func:`prebuild` for the model's widths); a weight that is not contiguous, or
    not 16-byte aligned, is copied first."""
    return _api().quantize_mok_mxfp8_weights(_dense(gate), _dense(up), _dense(down))
