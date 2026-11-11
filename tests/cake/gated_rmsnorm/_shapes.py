"""The shapes the prebuild contract test runs (``test_prebuild_gpu.py``), torch-free so that a CPU test checks them
against the non-deterministic package's module selection on every shipped part (``test_prebuild_routes.py``).

The package picks the module of a forward and of a BF16 backward from the call's rows (tokens x value heads), the
device's SM count and whether the kernels read the rows packed, row ``n`` at element ``128 n``
(``cake_jit.module_for_rows``): the forward by bands of rows per SM, one table for packed rows (tiny, pipelined, mid,
long and extra-long forwards) and one for other layouts (short, mid, long, extra-long and plain forwards); the backward
by the short and the exact tiny backward up to a count of rows, then by bands of rows per SM of the fused and the
repeatable backward, the repeatable backward's long program in its place on packed rows from a rows-per-SM cut. ``x``
and ``dy`` are contiguous, so a route's rows are packed when its gate is copied (:func:`packed_rows`). The rows of
miles' value-head counts (12, 16, 32, 48) at 1031 tokens, the gate read in place and copied, and of a token sweep at
32 value heads reach every module a call selects on every shipped part: the tiny forward and the short backward at 1
and 7 tokens, the exact tiny backward at 12, the extra-long forward of other layouts at 2560 (sm_100a, 400 to 700 rows
per SM), the plain forward and the repeatable backward from 8192 tokens, and the pipelined forwards and the long
repeatable backward at 13000 tokens with the gate copied.
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
    "tokens_12": (12, 16, 32, 0, True),
    "tokens_64": (64, 16, 32, 0, True),
    "tokens_2560": (2560, 16, 32, 0, True),
    "tokens_8192": (8192, 16, 32, 0, True),
    "tokens_13000": (13000, 16, 32, 0, True),
    "tokens_13000_copied": (13000, 16, 32, 8, False),
    "tokens_32771": (32771, 16, 32, 0, True),
    "tokens_65536": (65536, 16, 32, 0, True),
}

# the roles a call of the op runs: the forward and the backward of a BF16 weight gradient (the op's weight is BF16)
CALL_ROLES = ("fwd", "bwd_bf16")
# rows per SM past the highest cut of a band (the sm_103a packed forward's 8192)
MAX_ROWS_PER_SM = 20000


def packed_rows(tokens: int, in_place: bool) -> tuple[bool, bool]:
    """Whether the forward and the backward of a route read packed rows (``module_for_rows``' ``packed``): ``x`` and
    ``dy`` are contiguous and a copied gate is too, while a gate read in place has the projection's token stride. A
    single token's stride is passed as 0, which the forward's test takes as packed and the backward's does not."""
    return not in_place or tokens == 1, not in_place and tokens > 1


def row_counts(sm_count: int) -> list[int]:
    """Every row count up to 1024 (past the small-row backwards' bounds of 256 and 512 rows) and every multiple of the
    SM count up to ``MAX_ROWS_PER_SM`` rows per SM with its neighbours (the rows-per-SM cuts)."""
    per_sm = {k * sm_count + d for k in range(1, MAX_ROWS_PER_SM + 1) for d in (-1, 0, 1)}
    return sorted({*range(1, 1025), *per_sm})


def selected_modules(jit, arch: str, sm_count: int) -> set[str]:
    """The modules the package's ``cake_jit`` selects for a call role on ``arch`` with ``sm_count`` SMs at any row
    count (:func:`row_counts`) and either layout."""
    return {
        jit.module_for_rows(arch, role, rows, sm_count, packed=packed)
        for rows in row_counts(sm_count)
        for role in CALL_ROLES
        for packed in (True, False)
    }


def unreached_roles(jit, arch: str, sm_count: int) -> list[str]:
    """The roles of the modules a call selects (:func:`selected_modules`) that no route of ``SHAPES`` selects on
    ``sm_count`` SMs."""
    reached = set()
    for tokens, _, value_heads, _, in_place in SHAPES.values():
        for role, packed in zip(CALL_ROLES, packed_rows(tokens, in_place)):
            reached.add(jit.module_for_rows(arch, role, tokens * value_heads, sm_count, packed=packed))
    missing = selected_modules(jit, arch, sm_count) - reached
    return sorted(role for role, module in jit.ROUTES[arch].items() if module in missing)
