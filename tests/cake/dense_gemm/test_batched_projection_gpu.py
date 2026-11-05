"""The batched per-head projection (``dense_gemm.batched_projection``) on the Cake K1 programs: the output and the
input and weight gradients against an FP64 einsum, and bit-identical reruns.

The cases are miles' tensor-parallel-local shapes: the MLA q-absorb and value projections of GLM-5 / GLM-5.2 and
GLM-5.3-Flash at 8, 16 and 64 local heads, with the weights as splits of ``linear_kv_up_proj.weight`` and the q-absorb
input as a slice of the wider q head, and DeepSeek-V4's output-group projection at 1, 2 and 8 local groups; odd token
counts throughout. Then the vendored package's own MLA, strided-view and output-layout cases. Needs a CUDA device; the
tests need SM100a / SM103a / SM107a and skip elsewhere. The process never configures a variant, so the calls run det.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from miles_ext_kernels import cake  # noqa: E402
from miles_ext_kernels.cake._impl.dense_gemm import ops  # noqa: E402
from miles_ext_kernels.cake.dense_gemm import batched_projection, check_projection_dims  # noqa: E402

from ..helpers import assert_rel_rms  # noqa: E402

if torch.cuda.get_device_capability() not in cake.ARCH_BY_CAPABILITY:
    pytest.skip("the Cake dense projection GEMMs need SM100a / SM103a / SM107a", allow_module_level=True)

SEED = 20260930
BF16_ATOL = BF16_RTOL = 1e-2
F32_REL_FRO_MAX = 5e-5
F32_MAX_ABS_OVER_REF_ABSMAX = 1e-3
# Relative RMS budget of a BF16 result: the store rounds the exact FP32 sum once, half a BF16 spacing per element,
# which is a relative RMS of about 1.1e-3 to 1.6e-3 for these Gaussian operands; the budget is twice the upper end,
# and a 2 % error fails it (assert_rel_rms checks that).
BF16_REL_RMS = 3e-3
NAMES = ("output", "input gradient", "weight gradient")
TOKENS = (33, 1001, 2049)
HEADS = (8, 16, 64)
KV_LORA_RANK = 512

# The MLA projections. The weight is a split of linear_kv_up_proj.weight [H * head_rows, 512] unflattened per head:
# ``rows`` rows from ``row0`` of each head (rows = d_in for "in_out", d_out for "out_in"). The input is the first d_in
# columns of a ``slot``-wide head.
MLA_ROWS = {
    # GLM-5 / GLM-5.2: q head 192 + 64 rope, qk head 192, v head 256 -> w_kc [H, 192, 512], w_vc [H, 256, 512]
    "glm5-q-absorb": dict(d_in=192, d_out=KV_LORA_RANK, layout="in_out", slot=256, head_rows=448, row0=0),
    "glm5-value": dict(d_in=KV_LORA_RANK, d_out=256, layout="out_in", slot=KV_LORA_RANK, head_rows=448, row0=192),
    # GLM-5.3-Flash (NoPE MLA): q head 256 -> w_kc [H, 256, 512]
    "glm53-q-absorb": dict(d_in=256, d_out=KV_LORA_RANK, layout="in_out", slot=256, head_rows=512, row0=0),
}
# DeepSeek-V4's output-group projection: o [T, g, 4096] by linear_o_group_proj [g * 1024, 4096] viewed [g, 1024, 4096]
DSV4_IN, DSV4_OUT = 4096, 1024
DSV4_GROUPS = (1, 2, 8)

CASES = [(row, heads, tokens) for row in MLA_ROWS for heads in HEADS for tokens in TOKENS]
CASES += [("dsv4-o-group", groups, tokens) for groups in DSV4_GROUPS for tokens in TOKENS]


def _spec(row: str) -> dict:
    if row == "dsv4-o-group":
        return dict(d_in=DSV4_IN, d_out=DSV4_OUT, layout="out_in", slot=DSV4_IN, head_rows=DSV4_OUT, row0=0)
    return MLA_ROWS[row]


@pytest.fixture(scope="module", autouse=True)
def _programs():
    """Compile the programs of every case up front, in parallel, instead of one by one on first use."""
    shapes = {(heads, _spec(row)["d_in"], _spec(row)["d_out"], _spec(row)["layout"]) for row, heads, _ in CASES}
    cake.prebuild("dense_gemm", projections=sorted(shapes))


def _leaves(spec: dict, heads: int, tokens: int, seed: int):
    """The model's tensors of one projection: the activation buffer, the parameter and the output gradient."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    act = randn(tokens, heads, spec["slot"])
    columns = spec["d_out"] if spec["layout"] == "in_out" else spec["d_in"]
    param = randn(heads * spec["head_rows"], columns, scale=spec["d_in"] ** -0.5)
    grad = randn(tokens, heads, spec["d_out"])
    return act, param, grad


def _views(spec: dict, act: torch.Tensor, param: torch.Tensor):
    """``(x, weight)`` as the model hands them to the projection: views of the activation and of the parameter."""
    rows = spec["d_in"] if spec["layout"] == "in_out" else spec["d_out"]
    weight = param.unflatten(0, (-1, spec["head_rows"]))[:, spec["row0"] : spec["row0"] + rows]
    return act[..., : spec["d_in"]], weight


def _einsum(x, weight, layout):
    return torch.einsum("thd,hdm->thm" if layout == "in_out" else "thm,hdm->thd", x, weight)


def _cake(x, weight, layout):
    return batched_projection(x, weight, weight_layout=layout)


def _step(project, spec, act, param, grad):
    """One forward and backward through the views: ``(out, act.grad, param.grad)``."""
    act = act.detach().clone().requires_grad_(True)
    param = param.detach().clone().requires_grad_(True)
    x, weight = _views(spec, act, param)
    out = project(x, weight, spec["layout"])
    out.backward(grad)
    return out.detach(), act.grad, param.grad


def _assert_bf16_close(actual: torch.Tensor, expected_f64: torch.Tensor, what: str):
    assert actual.dtype == torch.bfloat16 and bool(torch.isfinite(actual).all()), what
    diff = (actual.double() - expected_f64).abs()
    outside = int((diff > BF16_ATOL + BF16_RTOL * expected_f64.abs()).sum())
    assert outside == 0, f"{what}: {outside} elements outside atol = rtol = 1e-2 (max abs {float(diff.max()):.3g})"


@pytest.mark.parametrize("row, heads, tokens", CASES, ids=[f"{r}-h{h}-t{t}" for r, h, t in CASES])
def test_projection_matches_fp64_and_reruns_bit_identically(row, heads, tokens, record_property):
    spec = _spec(row)
    act, param, grad = _leaves(spec, heads, tokens, seed=SEED + heads + tokens)

    want = _step(_einsum, spec, act.double(), param.double(), grad.double())
    first = _step(_cake, spec, act, param, grad)
    second = _step(_cake, spec, act, param, grad)

    assert first[0].shape == (tokens, heads, spec["d_out"]) and first[0].is_contiguous()
    for name, got, again, expected in zip(NAMES, first, second, want):
        assert torch.equal(got, again), f"{name} differs between two identical calls"
        _assert_bf16_close(got, expected, name)
    measured = assert_rel_rms(dict(zip(NAMES, first)), dict(zip(NAMES, want)), dict.fromkeys(NAMES, BF16_REL_RMS))
    record_property("rel_rms", measured)
    # the gradient leaves the rest of the head slot and the other projection's weight rows exactly zero
    assert not first[1][..., spec["d_in"] :].any()
    rows = spec["d_in"] if spec["layout"] == "in_out" else spec["d_out"]
    per_head = first[2].unflatten(0, (-1, spec["head_rows"]))
    assert not per_head[:, : spec["row0"]].any() and not per_head[:, spec["row0"] + rows :].any()


@pytest.mark.parametrize("layout", ["in_out", "out_in"])
def test_views_the_kernel_cannot_read_in_place_are_copied(layout, record_property):
    """A weight view starting 2 bytes past a 16-byte boundary (a parameter at an odd offset of its buffer) and an
    input whose head stride is not a multiple of 16 bytes give the reference result, bit-identically twice."""
    heads, tokens, d_in, d_out = 16, 1001, 192, KV_LORA_RANK
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    act = torch.randn(tokens, heads, d_in + 4, generator=gen, device="cuda").to(torch.bfloat16)
    rows, cols = (d_in, d_out) if layout == "in_out" else (d_out, d_in)
    buffer = (torch.randn(1 + heads * rows * cols, generator=gen, device="cuda") * d_in**-0.5).to(torch.bfloat16)
    grad = torch.randn(tokens, heads, d_out, generator=gen, device="cuda").to(torch.bfloat16)

    def step(project, act, buffer, grad):
        act = act.detach().clone().requires_grad_(True)
        buffer = buffer.detach().clone().requires_grad_(True)
        x, weight = act[..., :d_in], buffer[1:].view(heads, rows, cols)
        assert x.stride(1) % 8 and weight.data_ptr() % 16
        out = project(x, weight, layout)
        out.backward(grad)
        return out.detach(), act.grad, buffer.grad

    want = step(_einsum, act.double(), buffer.double(), grad.double())
    first, second = step(_cake, act, buffer, grad), step(_cake, act, buffer, grad)
    for name, got, again, expected in zip(NAMES, first, second, want):
        assert torch.equal(got, again), name
        _assert_bf16_close(got, expected, name)
    measured = assert_rel_rms(dict(zip(NAMES, first)), dict(zip(NAMES, want)), dict.fromkeys(NAMES, BF16_REL_RMS))
    record_property("rel_rms", measured)


def test_shapes_outside_the_kernel_contract_are_refused():
    with pytest.raises(ValueError, match="multiple of 8"):
        check_projection_dims(196, 512)
    x = torch.empty(4, 8, 192, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="does not match"):
        batched_projection(x, torch.empty(8, 256, 512, dtype=torch.bfloat16, device="cuda"), weight_layout="in_out")
    with pytest.raises(ValueError, match="weight_layout"):
        batched_projection(x, torch.empty(8, 192, 512, dtype=torch.bfloat16, device="cuda"), weight_layout="io")
    with pytest.raises(ValueError, match="BF16"):
        batched_projection(x.float(), torch.empty(8, 192, 512, device="cuda"), weight_layout="in_out")


# ---------------------------------------------------------------------------
# Cases of the vendored package's own tests (tests/experimental/test_cake_dense_projection_det_gemm.py at the
# vendored commit), on its prepared launches.
# ---------------------------------------------------------------------------

FI_MLA_ROWS = {
    "qabs": dict(H=64, D_in=192, D_out=512, act_stride_head=256, weight_layout="in_out"),
    "vproj": dict(H=64, D_in=512, D_out=256, act_stride_head=512, weight_layout="out_in"),
}
L2_BYTES = 132120576


def _fi_mla_views(row, op, T, dtype=torch.bfloat16, device="cuda"):
    spec = FI_MLA_ROWS[row]
    H, Din, Dout, slot = spec["H"], spec["D_in"], spec["D_out"], spec["act_stride_head"]
    act = torch.empty(T, H, slot, dtype=torch.bfloat16, device=device)[..., :Din]
    if spec["weight_layout"] == "in_out":
        weight = torch.empty(H, Din, Dout, dtype=torch.bfloat16, device=device)
        w_in_out = weight
    else:
        weight = torch.empty(H, Dout, Din, dtype=torch.bfloat16, device=device)
        w_in_out = weight.transpose(1, 2)
    d_out = torch.empty(T, H, Dout, dtype=torch.bfloat16, device=device)
    if op == "fwd":
        out = torch.empty(T, H, Dout, dtype=dtype, device=device).permute(1, 0, 2)
        return dict(A=act.permute(1, 0, 2), B=w_in_out, out=out, act=act, weight=weight, d_out=d_out)
    if op == "dgrad":
        buf = torch.empty(T, H, slot, dtype=dtype, device=device)
        out = buf[..., :Din].permute(1, 0, 2)
        return dict(
            A=d_out.permute(1, 0, 2), B=w_in_out.transpose(1, 2), out=out, buf=buf, act=act, weight=weight, d_out=d_out
        )
    if spec["weight_layout"] == "in_out":
        A, B = act.permute(1, 2, 0), d_out.permute(1, 0, 2)
    else:
        A, B = d_out.permute(1, 2, 0), act.permute(1, 0, 2)
    return dict(A=A, B=B, out=torch.empty_like(weight, dtype=dtype), act=act, weight=weight, d_out=d_out)


def _check(actual, expected_f64, *, untouched=None):
    diff = (actual.double() - expected_f64).abs()
    assert bool(torch.isfinite(actual).all())
    if actual.dtype == torch.bfloat16:
        violations = int((diff > BF16_ATOL + BF16_RTOL * expected_f64.abs()).sum())
        assert violations == 0, f"{violations} bf16 elements outside atol=rtol=1e-2 (max abs {float(diff.max()):.4g})"
    else:
        tiny = torch.finfo(torch.float64).tiny
        assert float(diff.norm() / max(float(expected_f64.norm()), tiny)) <= F32_REL_FRO_MAX
        assert float(diff.max()) / max(float(expected_f64.abs().max()), tiny) <= F32_MAX_ABS_OVER_REF_ABSMAX
    if untouched is not None:
        assert bool(torch.isnan(untouched.float()).all())


@pytest.mark.parametrize("row", list(FI_MLA_ROWS))
@pytest.mark.parametrize("op", ["fwd", "dgrad", "wgrad"])
def test_mla_batched_rows_match_fp64_reference(row, op):
    backend, _ = ops._vendored()
    v = _fi_mla_views(row, op, 257)
    gen = torch.Generator(device="cuda").manual_seed(SEED + 1)
    for name, scale in (("act", 1.0), ("weight", FI_MLA_ROWS[row]["D_in"] ** -0.5), ("d_out", 1.0)):
        if name in v:
            v[name].copy_((torch.randn(v[name].shape, generator=gen, device="cuda") * scale).to(torch.bfloat16))
    untouched = None
    if "buf" in v:
        v["buf"].fill_(float("nan"))  # the rope part of the slot must keep its sentinel
        untouched = v["buf"][..., FI_MLA_ROWS[row]["D_in"] :]
    prepared = backend.prepare_dense_projection_det_gemm(v["A"], v["B"], v["out"])
    out = prepared.launch()
    torch.cuda.synchronize()
    assert out is v["out"] and prepared.plan.L == 64
    _check(v["out"], torch.matmul(v["A"].double(), v["B"].double()), untouched=untouched)


def test_strided_views_and_storage_offsets_match_fp64_reference():
    backend, _ = ops._vendored()
    gen = torch.Generator(device="cuda").manual_seed(SEED + 2)
    T, K, N = 301, 704, 264
    # A: a column slice inside a wider buffer starting at a 16-byte aligned offset; B: a row slice of a taller
    # weight; out: a slice inside a larger buffer with a padded row stride
    A = torch.randn(T, K + 64, generator=gen, device="cuda").to(torch.bfloat16)[:, 8 : 8 + K]
    W = (torch.randn(N + 16, K, generator=gen, device="cuda") * K**-0.5).to(torch.bfloat16)[16:]
    out_buf = torch.full((T + 3, N + 24), float("nan"), device="cuda", dtype=torch.bfloat16)
    out = out_buf[3:, 8 : 8 + N]
    backend.prepare_dense_projection_det_gemm(A, W.t(), out).launch()
    torch.cuda.synchronize()
    _check(out, torch.matmul(A.double(), W.double().t()))
    assert bool(torch.isnan(out_buf[:3].float()).all()) and bool(torch.isnan(out_buf[3:, :8].float()).all())
    assert bool(torch.isnan(out_buf[3:, 8 + N :].float()).all())
    # the swapped small-N weight gradient X.T @ G (both operands MN-major) with the transposed fp32 store into a
    # [N, K] output inside a wider buffer (padded row stride)
    Nf = 96
    Gs = torch.randn(T, Nf, generator=gen, device="cuda").to(torch.bfloat16)
    out_t_buf = torch.full((Nf, K + 8), float("nan"), device="cuda", dtype=torch.float32)
    out_t = out_t_buf[:, :K]
    prepared = backend.prepare_projection_det_wgrad(Gs, A, out_t)
    assert prepared.plan.transposed_out
    prepared.launch()
    torch.cuda.synchronize()
    _check(out_t, torch.matmul(Gs.double().t(), A.double()))
    assert bool(torch.isnan(out_t_buf[:, K:]).all())


def test_batched_rows_satisfy_the_kernel_output_layout_predicate():
    """The strides the host hands the kernel for the batched head-interleaved MLA rows ([T, H, D] written as
    [H, T, D], D a slice of a wider slot) satisfy the bindings' output-layout predicate: with rows / cols the stored
    matrix's extents (N / M under the transposed store), ldo >= cols, out_l >= cols and either batch-major
    (out_l >= rows * ldo) or interleaved within a row (ldo >= L * out_l). Planned on ``meta`` views."""
    backend, _ = ops._vendored()
    checked = 0
    for row in FI_MLA_ROWS:
        for op in ("fwd", "dgrad", "wgrad"):
            for T in (1001, 16231):
                v = _fi_mla_views(row, op, T, device="meta")
                plan, _, _, out3 = backend.plan_dense_projection_det_gemm(
                    v["A"], v["B"], v["out"], sm_count=152, l2_bytes=L2_BYTES, arch="sm_100a"
                )
                ldo, out_l = backend._out_strides(out3)
                rows, cols = (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
                assert ldo >= cols and out_l >= cols, (row, op, T, ldo, out_l)
                assert out_l >= rows * ldo or ldo >= plan.L * out_l, (row, op, T, ldo, out_l, rows, plan.L)
                assert FI_MLA_ROWS[row]["H"] == plan.L
                checked += 1
    assert checked == 12
    # the interleaved store of the qabs input gradient (D = 192 of a 256-wide slot): the second form, with equality
    v = _fi_mla_views("qabs", "dgrad", 33, device="meta")
    plan, _, _, out3 = backend.plan_dense_projection_det_gemm(
        v["A"], v["B"], v["out"], sm_count=152, l2_bytes=L2_BYTES
    )
    ldo, out_l = backend._out_strides(out3)
    assert (ldo, out_l, plan.L) == (64 * 256, 256, 64) and out_l < plan.M * ldo and ldo == plan.L * out_l
