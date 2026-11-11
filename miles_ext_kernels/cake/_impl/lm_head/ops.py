"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Selected-token log-probabilities of an LM head on the vendored Cake chunked LM-head packages (SM100a / SM103a /
SM107a, built with a CUDA 13.4 or newer toolkit).

``logp_t = z[t, y_t] / tau_t - logsumexp_v(z[t, v] / tau_t)`` with ``z = bf16(X @ W^T)``, chunk by chunk: the forward
keeps the FP32 log-sum-exp of each row, the backward recomputes the logits and forms ``dX`` and ``dW`` in FP32 with
one BF16 cast each. No vocabulary-sized buffer spans more than one chunk (``fast``'s folded chunk: fewer than 1.5
chunks of rows). Each call
reads its variant (``config.variant("lm_head")``) and chunk size when it runs; its backward runs the package of its
forward:

* ``det``: ``cake_lm_head_det_train``. Fixed chunk order, no atomics: ``logp``, ``dX`` and ``dW`` repeat bit for bit
  for a fixed chunk size.
* ``fast``: ``cake_lm_head_nondet_train``. ``det``'s ``logp`` bits. A call without ``num_valid_rows`` of fewer than
  three chunks of ``T`` rows whose valid rows make two chunks, the second shorter than half a chunk, runs its forward
  as one chunk (the short-tail fold: fewer than 1.5 chunks of rows). The backward takes one of two forms (the
  package's rules):

  - vocabulary-major: a call without ``num_valid_rows`` that trains ``W``, of at most 65535 valid rows that make 2 to
    4 chunks (except 2 chunks whose second chunk holds at least half a chunk). Per vocabulary block, in block order,
    it recomputes the block's ``dz`` over every valid row, one weight-gradient GEMM writes the block's BF16 ``dW``
    rows and the dX GEMM adds the block's product into the FP32 ``dX`` accumulator (one K slice). No FP32 ``[V, H]``
    accumulator; the ``dz`` block holds about one chunk's logits.
  - the chunk loop, for every other call: each chunk's ``dX`` and ``dW`` add into FP32 accumulators. The dX GEMM of a
    K-sliced chunk stores each slice into an FP32 slab that a second kernel sums in slice order. A folded call's
    chunk loop runs its one chunk.

  Bits, as the package documents them: its slab-summed ``dX`` and the vocabulary-major ``dX`` and ``dW`` repeat bit
  for bit run to run; it makes no run-to-run promise for the chunk loop's ``dW``. ``dW`` is ``det``'s bits when no
  valid row has a zero incoming gradient, except on a folded call or a vocabulary-major call (other summation
  groupings within the same accuracy budgets). ``dX`` can differ from ``det``'s. This module promises no bits for
  ``fast``. A backward of a call with an ignored row and at least the architecture's floor of valid rows (1946 on
  SM100a and SM107a, 3891 on SM103a) skips the GEMM work of the rows whose incoming gradient is zero (PPO-clipped or
  masked tokens), in both forms.

Host synchronization. A call without ``num_valid_rows`` reads its count of valid rows back to the host once, in the
forward (the backward reuses the forward's row index). With ``num_valid_rows`` the forward plans on the given count
and checks it on the device, without waiting for the device; a ``fast`` call then keeps its chunks of the chunk size
(no short-tail fold). The packages also keep a host-count path for SM103a programs built by CUDA 13.0 (at ``ptxas
-O0``, their mitigation of a ptxas 13.0 fault), where each forward waits for the device and checks a given count on
the host; the runtime builds with CUDA 13.4 or newer, so no call takes it (:func:`host_count_path` and its log stay
for the packages' rule).

Labels. A negative label (``-100`` by convention; every value below 0 alike) marks an ignored row: ``logp`` 0, no
gradient. A label at or above ``V`` is an error: each call queues the device-side assertion ``max(labels) < V`` ahead
of the kernels (no host synchronization), so such a label stops the CUDA stream with a device-side assertion error,
which ends the CUDA context, instead of returning a plausible ``logp``.

Deterministic algorithms. A ``fast`` call's backward reads ``torch.are_deterministic_algorithms_enabled()`` when it
runs: with a trainable input it then raises PyTorch's alert (``RuntimeError``) before the package launches anything,
or, under ``warn_only=True``, warns and runs ``fast``. The forward runs as always and ``det`` never alerts. That is the
rule of the packages before the mode-word naming; ``cake_lm_head_nondet_train`` itself alerts only before a K-sliced
dX reduction, ordered though it is (under ``warn_only=True`` it then warns a second time), so this module raises the
alert for every shape.

Weight gradient. ``dW`` is BF16, one cast of its FP32 accumulation. A chunk-loop backward of more than one chunk
keeps an FP32 ``[V, H]`` accumulator (``4 * V * H`` bytes) next to one chunk's BF16 logits; ``fast``'s
vocabulary-major backward keeps none (its FP32 ``[T_v, H]`` ``dX`` accumulator instead), nor does a folded call.
``det`` picks its form per call of more than one chunk (the package's automatic memory form): when the chunk loop's
predicted step peak exceeds the bytes of a head that keeps the BF16 logits of the valid rows
(``2 * T_v * V + 2 * V * H + 2 * T * H``, less a 2 % margin), the backward runs the fastest exact form that fits:
an FP32 accumulator split in two vocabulary halves and cast in place (``"fused_ip"``), the accumulator cast once
after ``dX`` is written (``"unfused"``), or an accumulator over part of the vocabulary (or none) with the other
``dW`` rows recomputed in vocabulary blocks (``"partial_ip"``, ``"partial"``, ``"recompute"``); when none fits, the
chunk loop. Every run gives the same bits; ``dX`` and ``dW``
keep the chunk loop's bits unless a valid row has a zero incoming gradient (the budget forms do not skip those
rows). A ``det`` call without a temperature or ``num_valid_rows``, of fewer than three chunks of ``T`` rows, whose
valid rows fit one chunk and whose backward runs no zero-gradient row skip, keeps their BF16 logits
(``2 * T_v * V`` bytes) from its forward to its backward instead of recomputing them (the same bits). There is no
FP32 ``dW``: the autograd engine casts a gradient to the dtype of its input, BF16 for ``W``.
"""

from __future__ import annotations

import functools
import logging
import sys
import warnings
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from miles_ext_kernels.cake._runtime import config, loader

__all__ = ["log_probs", "prebuild", "release"]

OP = "lm_head"
# the vendored package of each variant
PACKAGES = {"det": "cake_lm_head_det_train", "fast": "cake_lm_head_nondet_train"}
_VENDOR = "miles_ext_kernels.cake._vendor"
# the modules of a package whose binding caches keep device workspaces (release() clears them)
_BINDING_MODULES = ("cake_backend", "cake_backend_opt_in")
# what the alert of a fast backward under torch.use_deterministic_algorithms(True) names
_FAST_BACKWARD = "the backward of the cake lm_head fast variant"
_DET_VARIANT = "--cake-lm-head-variant det is the deterministic variant"

logger = logging.getLogger(__name__)


def _package(variant: str) -> ModuleType:
    """The vendored package of ``variant``, with the loader installed on the ``cake_build`` its modules import."""
    return loader.load_package(PACKAGES[variant])


@functools.cache
def _jit(variant: str) -> ModuleType:
    """The program registry (``cake_jit``) of the package of ``variant``."""
    return import_module(".cake_jit", _package(variant).__name__)


@functools.cache
def _entry(variant: str) -> ModuleType:
    """The log-probability entries (``cake_backend_opt_in``) of the package of ``variant``."""
    return import_module(".cake_backend_opt_in", _package(variant).__name__)


@functools.cache
def _check_toolkit(arch: str) -> None:
    """Check once per architecture that the toolkit builds ``arch``; say once whether its calls wait for the device."""
    loader.check_toolkit(arch, "the cake lm_head kernels")
    if host_count_path(arch):
        logger.warning(
            "miles_ext_kernels.cake.lm_head: the %s programs built by CUDA 13.0 run at ptxas -O0 on the host-count "
            "path, so every forward waits for the device once, also with num_valid_rows (the packages' mitigation of "
            "a ptxas 13.0 fault)",
            arch,
        )


def host_count_path(arch: str) -> bool:
    """Do the calls on ``arch`` read their valid-row count back to the host even when given ``num_valid_rows``?

    True for SM103a programs built by CUDA 13.0 (the packages' own rule; the same for both variants), which the
    runtime's CUDA 13.4 floor does not build.
    """
    return not _jit("det").toolchain_runs_hidden_count(arch)


def _arch(device: Any) -> str:
    arch = loader.device_arch(device)
    _check_toolkit(arch)
    return arch


def _assert_labels_below(labels: torch.Tensor, vocab: int) -> None:
    """Queue ``max(labels) < vocab`` as a device-side assertion: no host synchronization, ahead of every kernel that
    reads the labels."""
    if not (labels.is_cuda and labels.numel()):
        return  # the package rejects labels off the device; an empty call reads none
    condition = labels.max() < vocab
    message = f"cake lm_head: a label is >= V = {vocab} (a negative label marks an ignored row)"
    try:
        torch._assert_async(condition, message)
    except TypeError:  # a PyTorch without the message argument
        torch._assert_async(condition)


def _alert_nondeterministic_backward(grad: torch.Tensor) -> None:
    """PyTorch's alert for the backward of a ``fast`` call, a hook on its ``logp`` (so it runs before the package's
    backward launches anything): a ``RuntimeError`` while torch's deterministic algorithms are enabled, a
    ``UserWarning`` under ``warn_only=True`` (the backward then runs ``fast``), nothing otherwise. Torch's flags are
    read when the backward runs."""
    if not torch.are_deterministic_algorithms_enabled():
        return
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{_FAST_BACKWARD} does not have a deterministic implementation, but you set "
            f"'torch.use_deterministic_algorithms(True, warn_only=True)'; {_DET_VARIANT}",
            stacklevel=2,
        )
        return
    raise RuntimeError(
        f"{_FAST_BACKWARD} does not have a deterministic implementation, but you set "
        f"'torch.use_deterministic_algorithms(True)'; {_DET_VARIANT}, or use the 'warn_only=True' option if that is "
        "acceptable for your application"
    )


def log_probs(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float | torch.Tensor | None = None,
    num_valid_rows: int | None = None,
) -> torch.Tensor:
    """Differentiable FP32 ``logp [T]`` of ``labels`` under ``softmax(X @ W^T / temperature)``.

    Args:
        X: BF16 ``[T, H]`` hidden rows on a CUDA device, row-major with a 16-byte row pitch (otherwise copied); ``H`` a
            multiple of 512.
        W: BF16 ``[V, H]`` output weight over the vocabulary the softmax runs over, contiguous; ``V`` a multiple of
            256. ``dW`` reaches it as BF16 (module docstring, Weight gradient).
        labels: int64 ``[T]`` target ids: below ``V``, or negative (``-100``) for an ignored row (``logp`` 0, no
            gradient). A label ``>= V`` ends the CUDA context with a device-side assertion (module docstring, Labels).
        temperature: ``None`` (or a number whose FP32 value is 1.0) for none; a positive finite number; or an FP32
            ``[T]`` tensor on ``X``'s device, positive and finite on every row, ignored rows included (checked by a
            device-side assertion). Not differentiable.
        num_valid_rows: the host count of ``labels >= 0`` when the caller knows it (every row valid: ``T``): the
            forward plans on it instead of reading the count back (module docstring, Host synchronization). It is
            checked on the device: a wrong count stops the stream with a device-side assertion. ``det``: the same
            bits as without it. ``fast``: the same ``logp``, and the same ``dW`` unless a valid row has a zero
            incoming gradient or the call without the count folds or runs the vocabulary-major backward (the call
            with the count runs the chunk loop without the fold); ``dX`` may round differently (module docstring).

    The variant and the chunk size are the configured ones (``--cake-lm-head-variant``,
    ``--cake-lm-head-chunk-size``), read now; ``det``'s bits depend on the chunk size. Raises ``ValueError`` for
    shapes, dtypes or values the kernels do not serve and for ``X`` off CUDA, ``NotImplementedError`` off SM100a /
    SM103a / SM107a and ``RuntimeError`` when the toolkit cannot build the device's kernels. A call of ``T == 0`` rows
    returns an empty ``logp`` (zero ``dW``). The backward of a ``fast`` call raises PyTorch's alert under
    ``torch.use_deterministic_algorithms(True)`` (module docstring, Deterministic algorithms).
    """
    if X.device.type != "cuda":
        raise ValueError(f"the cake lm_head kernels run on a CUDA device, X is on {X.device}")
    _arch(X.device)
    variant = config.variant(OP)
    chunk_size = config.option(OP, "cake_lm_head_chunk_size")
    entry = _entry(variant)
    if W.ndim == 2:
        _assert_labels_below(labels, int(W.shape[0]))
    if variant == "det":
        return entry.cake_lm_head_det_train_logprob_temperature(
            X, W, labels, chunk_size=chunk_size, num_valid_rows=num_valid_rows, temperature=temperature
        )
    logp = entry.cake_lm_head_nondet_train_logprob_temperature(
        X, W, labels, chunk_size=chunk_size, num_valid_rows=num_valid_rows, temperature=temperature
    )
    if logp.requires_grad:
        logp.register_hook(_alert_nondeterministic_backward)
    return logp


def prebuild(*, device: Any = None, max_workers: int | None = None) -> list[Path]:
    """Build every library :func:`log_probs` may launch on ``device``'s architecture (default: the current device):
    every log-probability stage of both variants' programs, with and without a temperature, ``max_workers`` compiles
    at a time. The programs' other stages serve other entries of the packages: the candidate-support stages
    (``support_*``, with ``fast``'s accumulating dX slab sum) and ``fast``'s one-pass cross-entropy of the loss entry.
    No call of :func:`log_probs` launches them, so they are not built.

    lm_head takes no model parameters: one program per architecture serves every admissible ``H`` and ``V`` (launch
    scalars), every token count and chunk size. Returns the cache paths, built now or before.
    """
    arch = _arch(device)
    paths: list[Path] = []
    for variant, package in PACKAGES.items():
        jit = _jit(variant)
        record = jit.select_module(arch)
        names = [jit.library_name(record, stage) for stage in jit.registered_stages_with_temperature(record)]
        paths += loader.prebuild_libraries(package, names, max_workers)
    return paths


def release() -> None:
    """Drop the bindings both packages' entries remember, with the device workspaces they own (a package that was
    never imported is left alone)."""
    for package in PACKAGES.values():
        for name in _BINDING_MODULES:
            module = sys.modules.get(f"{_VENDOR}.{package}.{name}")
            if module is not None:
                module.BINDING_CACHE.clear()
