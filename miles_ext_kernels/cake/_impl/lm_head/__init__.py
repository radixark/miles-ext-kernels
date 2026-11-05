"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

The cake lm_head op on the vendored chunked LM-head packages (``det``: ``cake_lm_head_det_loss``, ``fast``:
``cake_lm_head_loss``): :func:`log_probs`, :func:`prebuild` and :func:`release` (:mod:`.ops`).
"""

from miles_ext_kernels.cake._runtime import loader

loader.install_op("lm_head")

from .ops import log_probs, prebuild, release  # noqa: E402

__all__ = ["log_probs", "prebuild", "release"]
