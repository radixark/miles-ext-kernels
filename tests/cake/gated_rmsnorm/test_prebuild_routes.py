"""The row-count modules of the non-deterministic package on the shipped parts (CPU; torch and tvm_ffi, no device).

The package picks the module of a forward and of a BF16 backward from the call's rows, the device's SM count and
whether the rows are packed (``cake_jit.module_for_rows``), so the modules a call can run depend on the part. For every
part (architecture and SM count): every band names a routed module (``cake_jit.routes_every_call``); each module a
call selects, at any row count and in either layout, is a library ``prebuild`` builds, so no call compiles after it;
and the routes of ``test_prebuild_gpu.py`` (``_shapes.SHAPES``) select every one of those modules, so that test's run
under ``no_build()`` loads each of them. The GPU test checks the routes on the device's own SM count and the layout
each route's rows are read in.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels.cake._impl.gated_rmsnorm import ops  # noqa: E402

from ._shapes import CALL_ROLES, selected_modules, unreached_roles  # noqa: E402

# (architecture, SM count) of the shipped parts
PARTS = (
    ("sm_100a", 148),
    ("sm_100a", 152),
    ("sm_103a", 148),
    ("sm_103a", 152),
    ("sm_103a", 160),
    ("sm_107a", 208),
    ("sm_107a", 212),
    ("sm_107a", 216),
    ("sm_107a", 224),
)


@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_every_module_a_call_selects_is_prebuilt(monkeypatch, arch, sm_count):
    built = {}
    monkeypatch.setattr(ops.loader, "device_arch", lambda device=None: arch)
    monkeypatch.setattr(
        ops.loader,
        "prebuild_libraries",
        lambda package, names, max_workers=None: built.setdefault(package, set()).update(names) or [],
    )
    ops.prebuild()
    jit = ops._package("fast").cake_jit

    selected = {jit.library_name(module, arch) for module in selected_modules(jit, arch, sm_count)}

    assert selected <= built[ops.PACKAGES["fast"]], sorted(selected - built[ops.PACKAGES["fast"]])


@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_the_token_sweep_reaches_every_row_count_module(arch, sm_count):
    jit = ops._package("fast").cake_jit
    missing = unreached_roles(jit, arch, sm_count)

    assert jit.routes_every_call(arch), f"{arch}: a band names a module the package does not route"
    assert len(selected_modules(jit, arch, sm_count)) > len(CALL_ROLES), f"{arch}: no row-count tier is routed"
    assert missing == [], f"{arch} with {sm_count} SMs: no route reaches {missing}"
