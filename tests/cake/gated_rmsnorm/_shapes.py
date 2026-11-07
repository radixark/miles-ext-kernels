"""The shapes the prebuild contract test runs (``test_prebuild_gpu.py``), torch-free so that a CPU test checks them
against the non-deterministic package's module selection on every shipped part (``test_prebuild_routes.py``).

The package picks the module of a forward and of a BF16 backward from the call's rows (tokens x value heads) and the
device's SM count (``cake_jit.module_for_rows``): the short module up to its row bound, else the first of the
extra-long, long and mid modules the architecture routes whose rows-per-SM floor the call reaches, else the regular
one. The rows of miles' value-head counts (12, 16, 32, 48) at 1031 tokens and of a token sweep at 32 value heads
reach: short calls (1 and 7 tokens), regular ones (64 tokens and the 1031-token shapes), mid ones (8192 tokens; on
SM107a the long backward, from 1024 rows per SM), the SM103a long forward (13000 tokens, 416000 rows: it runs from
2400 rows per SM up to the extra-long forward's 3000), the extra-long forward (32771 tokens) and the SM100a / SM103a
long backward (65536 tokens, 2097152 rows: from 12000 rows per SM).
"""

# route -> (tokens, key heads, value heads, extra projection columns, gate read in place)
SHAPES = {
    "h12_in_place": (1031, 6, 12, 8, True),
    "h12_copied": (1031, 6, 12, 0, False),
    "h16_in_place": (1031, 8, 16, 0, True),
    "h16_copied": (1031, 8, 16, 8, False),
    "h32_in_place": (1031, 16, 32, 0, True),
    "h32_copied": (1031, 16, 32, 8, False),
    "h48_in_place": (1031, 24, 48, 0, True),
    "h48_copied": (1031, 24, 48, 8, False),
    "tokens_1": (1, 16, 32, 0, True),
    "tokens_7": (7, 16, 32, 0, True),
    "tokens_64": (64, 16, 32, 0, True),
    "tokens_8192": (8192, 16, 32, 0, True),
    "tokens_13000": (13000, 16, 32, 0, True),
    "tokens_32771": (32771, 16, 32, 0, True),
    "tokens_65536": (65536, 16, 32, 0, True),
}

# the roles a call of the op runs: the forward and the backward of a BF16 weight gradient (the op's weight is BF16)
CALL_ROLES = ("fwd", "bwd_bf16")


def routed_roles(jit, arch: str) -> list[str]:
    """The roles of ``CALL_ROLES`` and their row-count tiers (``<role>_short``, ``_mid``, ...) that the package's
    ``cake_jit`` routes on ``arch``."""
    return [
        role for role in jit.ROUTES[arch] if any(role == base or role.startswith(f"{base}_") for base in CALL_ROLES)
    ]


def unreached_roles(jit, arch: str, sm_count: int) -> list[str]:
    """The routed roles (:func:`routed_roles`) whose module no row count of ``SHAPES`` selects on ``sm_count`` SMs."""
    rows = {tokens * value_heads for tokens, _, value_heads, _, _ in SHAPES.values()}
    reached = {jit.module_for_rows(arch, role, count, sm_count) for count in rows for role in CALL_ROLES}
    return sorted(role for role in routed_roles(jit, arch) if jit.ROUTES[arch][role] not in reached)
