"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The Cake training kernels: one module per op (``miles_ext_kernels.cake.<op>``, listed in ``OPS``), one runtime
(``_runtime``), one tree of vendored packages (``_vendor``) and the op implementations (``_impl``).

Importing this package or an op module imports neither torch nor tvm_ffi; an op's entry points import its
implementation on first use. The names below are the runtime's public API.
"""

from ._runtime import (
    ARCH_BY_CAPABILITY,
    MIN_CUDA,
    OPS,
    Flag,
    OpSpec,
    add_arguments,
    cache_dir,
    configure,
    deterministic_mode,
    device_arch,
    op_spec,
    option,
    prebuild,
    release,
    validate_args,
    variant,
    warn_runs_det,
)

__all__ = [
    "ARCH_BY_CAPABILITY",
    "MIN_CUDA",
    "OPS",
    "Flag",
    "OpSpec",
    "add_arguments",
    "cache_dir",
    "configure",
    "deterministic_mode",
    "device_arch",
    "op_spec",
    "option",
    "prebuild",
    "release",
    "validate_args",
    "variant",
    "warn_runs_det",
]
