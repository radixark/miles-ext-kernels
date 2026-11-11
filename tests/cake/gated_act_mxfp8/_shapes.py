"""The routes and problem sizes the gated_act_mxfp8 tests share (torch-free)."""

# route -> (backward, rowwise, colwise); the route names of the vendored package's ``cake_jit.KINDS``
KINDS = {
    "fwd_row": (False, True, False),
    "fwd_col": (False, False, True),
    "fwd_both": (False, True, True),
    "bwd_row": (True, True, False),
    "bwd_col": (True, False, True),
    "bwd_both": (True, True, True),
}
# Entry-point shapes (M, K): M * K = 2^14, 2^16, 2^17, 2^19, 2^21, 3 * 2^20, 2^22, 3 * 2^21, 2^23, 2^24, 2^25, 7 * 2^23,
# 2^26, 2^28 and 2^29, at which the selection rule picks every program of each route on each architecture
# (``test_registry.py`` checks it).
ENTRY_SHAPES = (
    (128, 128),
    (256, 256),
    (256, 512),
    (512, 1024),
    (2048, 1024),
    (1536, 2048),
    (2048, 2048),
    (3072, 2048),
    (4096, 2048),
    (4096, 4096),
    (8192, 4096),
    (7168, 8192),
    (8192, 8192),
    (32768, 8192),
    (65536, 8192),
)
# (M, K) every program runs at directly: one tile row of scale tiles; odd counts of 128-column scale tiles.
PROGRAM_SHAPES = ((128, 128), (256, 384), (384, 256))
