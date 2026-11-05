"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The runtime every Cake op shares (torch-free to import):

- :mod:`.spec`: an op's declaration, ``OpSpec`` and ``Flag``.
- :mod:`.registry`: the registered ops, ``prebuild`` and ``release``.
- :mod:`.config`: the argument surface, the configuration of the process and the variant each op runs.
- :mod:`.loader`: the nvcc + tvm-ffi loader of the vendored packages.
- :mod:`.workspace`: the bounded workspace pool.

``miles_ext_kernels.cake`` re-exports the names below; an op's ``_impl`` package imports the modules themselves.
"""

from .config import add_arguments, configure, deterministic_mode, option, validate_args, variant, warn_runs_det
from .loader import ARCH_BY_CAPABILITY, MIN_CUDA, cache_dir, device_arch
from .registry import OPS, op_spec, prebuild, release
from .spec import Flag, OpSpec

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
