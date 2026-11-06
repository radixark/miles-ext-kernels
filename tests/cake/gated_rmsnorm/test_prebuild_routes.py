"""The row-count modules of the non-deterministic package on the shipped parts (CPU; torch and tvm_ffi, no device).

The package picks the module of a forward and of a BF16 backward from the call's rows and the device's SM count
(``cake_jit.module_for_rows``), so the modules a call can run depend on the part. For every part (architecture and SM
count): each module a call selects, at any row count, is a library ``prebuild`` builds, so no call compiles after
it; and the token sweep of ``test_prebuild_gpu.py`` (``_shapes.SHAPES``) selects every forward and BF16 backward
module the package routes, so that test's run under ``no_build()`` loads each of them. The GPU test checks the sweep
on the device's own SM count.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

from miles_ext_kernels.cake._impl.gated_rmsnorm import ops  # noqa: E402

from ._shapes import CALL_ROLES, routed_roles, unreached_roles  # noqa: E402

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
# rows per SM past the highest floor of a row-count tier (the SM100a / SM103a long backward's 12000)
MAX_ROWS_PER_SM = 20000


def _row_counts(sm_count: int) -> list[int]:
    """Every row count up to 300 (the short backward's fixed bound) and every multiple of the SM count up to
    ``MAX_ROWS_PER_SM`` rows per SM with its neighbours (the per-SM bounds and floors)."""
    per_sm = {k * sm_count + d for k in range(1, MAX_ROWS_PER_SM + 1) for d in (-1, 0, 1)}
    return sorted({*range(1, 301), *per_sm})


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

    selected = {
        jit.library_name(jit.module_for_rows(arch, role, rows, sm_count), arch)
        for rows in _row_counts(sm_count)
        for role in CALL_ROLES
    }

    assert selected <= built[ops.PACKAGES["fast"]], sorted(selected - built[ops.PACKAGES["fast"]])


@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_the_token_sweep_reaches_every_row_count_module(arch, sm_count):
    jit = ops._package("fast").cake_jit
    missing = unreached_roles(jit, arch, sm_count)

    assert set(CALL_ROLES) < set(routed_roles(jit, arch)), f"{arch}: no row-count tier is routed"
    assert missing == [], f"{arch} with {sm_count} SMs: no route reaches {missing}"
