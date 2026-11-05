"""A synthetic op, ``synth``, so the runtime tests run while no op is registered.

It is laid out as an op is: :data:`OP` is what ``miles_ext_kernels/cake/synth.py`` would hold, and :func:`scale`,
:func:`prebuild` and :func:`release` what ``miles_ext_kernels/cake/_impl/synth/`` would define. :func:`install`
registers both modules under those names and appends ``synth`` to ``registry.OPS``, so every check written for the ops
also runs on it. Importing this module does not import torch.
"""

import sys
import types
from argparse import Namespace

from miles_ext_kernels.cake._runtime import config, registry, workspace
from miles_ext_kernels.cake._runtime.spec import Flag, OpSpec, lazy_exports

NAME = "synth"
MODULE = f"miles_ext_kernels.cake.{NAME}"
IMPL = f"miles_ext_kernels.cake._impl.{NAME}"
MAX_WIDTH = 64


def _validate(args: Namespace) -> None:
    if args.cake_synth_scale <= 0:
        raise ValueError(f"--cake-synth-scale must be positive, not {args.cake_synth_scale}")


OP = OpSpec(
    name=NAME,
    summary="synthetic: y = scale * x",
    variants=("det", "fast"),
    selected_by=(("synth_backend", "cake"),),
    exports=("scale",),
    flags=(Flag("--cake-synth-scale", dict(type=float, default=1.0, help="Factor of the synthetic op.")),),
    validate=_validate,
)

# (call, keyword arguments) of every prebuild / release the implementation received since install()
CALLS: list[tuple[str, dict]] = []


def scale(x):
    """``(variant, scale * x)``, the variant and the option read at this call; the product goes through the pool."""
    import torch

    out = workspace.take(NAME, "product", x.shape, x.dtype, x.device)
    torch.mul(x, config.option(NAME, "cake_synth_scale"), out=out)
    return config.variant(NAME), out.clone()


def prebuild(*, device=None, max_workers=None, width=None):
    if width is not None and width > MAX_WIDTH:
        raise ValueError(f"synth serves width <= {MAX_WIDTH}, not {width}")
    CALLS.append(("prebuild", {"device": device, "max_workers": max_workers, "width": width}))
    return []


def release():
    CALLS.append(("release", {}))


def install(monkeypatch=None, op: OpSpec = OP) -> types.ModuleType:
    """Register ``synth`` (``op``: :data:`OP` or a variation of it) as an op; ``monkeypatch`` undoes it after a test.

    Returns the op module. Both modules point their ``__file__`` at this file.
    """
    module = types.ModuleType(MODULE, (sys.modules[__name__].__doc__ or "").splitlines()[0])
    module.__file__ = __file__
    module.OP = op
    module.__all__, module.__getattr__, module.__dir__ = lazy_exports(MODULE, op)
    impl = types.ModuleType(IMPL)
    impl.__file__ = __file__
    impl.__path__ = []
    impl.scale, impl.prebuild, impl.release = scale, prebuild, release
    CALLS.clear()
    ops = (*[name for name in registry.OPS if name != NAME], NAME)
    if monkeypatch is None:
        registry.OPS = ops
        sys.modules[MODULE], sys.modules[IMPL] = module, impl
    else:
        monkeypatch.setattr(registry, "OPS", ops)
        monkeypatch.setitem(sys.modules, MODULE, module)
        monkeypatch.setitem(sys.modules, IMPL, impl)
    return module
