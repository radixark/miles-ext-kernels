"""One mHC connection for the GPU tests: FlashInfer's unit-regime inputs, a forward and backward through the entry
points miles calls, and the FP64 reference of the function (autograd through an FP64 forward).

Inputs: ``X``, ``B``, ``dY``, ``dU`` ~ N(0, 1) BF16, ``fn`` ~ 1e-2 N(0, 1), ``scale`` ~ 1 + 0.1 N, ``base`` ~ 0.1 N,
drawn in FlashInfer's order from the device generator seeded :data:`SEED`. Imported by the tests (and by the
subprocess of the cross-process determinism test); imports torch.
"""

from __future__ import annotations

import torch

from miles_ext_kernels.cake._impl.mhc import ops

BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
STREAMS, MAPPING_ROWS = 4, 24
SEED = 2032
BLOCK_TOKENS = 8192  # rows of the FP64 reference per block
BF16_OUTPUTS = ("U", "Y", "dB", "dX")
PARAMETER_GRADS = ("dfn", "dscale", "dbase")
OUTPUTS = (*BF16_OUTPUTS, *PARAMETER_GRADS)


def make_inputs(T: int, hidden: int, device: torch.device, seed: int = SEED) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=device, dtype=F32).to(dtype)

    X = randn(T, STREAMS, hidden, dtype=BF16)
    fn = randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2
    scale = 1.0 + 0.1 * randn(3)
    base = 0.1 * randn(MAPPING_ROWS)
    B = randn(T, hidden, dtype=BF16)
    dY = randn(T, STREAMS, hidden, dtype=BF16)
    dU = randn(T, hidden, dtype=BF16)
    return {"X": X, "fn": fn.contiguous(), "scale": scale, "base": base, "B": B, "dY": dY, "dU": dU}


def run(
    i: dict[str, torch.Tensor],
    *,
    params_dtype: torch.dtype = F32,
    grad_X: bool = True,
    grad_params: bool = True,
    **constants,
) -> dict:
    """One connection forward and backward with the configured variant; ``B`` stands for the branch output.

    ``constants`` are ``rms_eps`` / ``hc_eps`` / ``sinkhorn_iters`` of :func:`ops.mhc_prepare`. Returns the outputs,
    the gradients of the inputs that require one, and the type of the prepare handle (which names the variant).
    """
    X = i["X"].detach().clone().requires_grad_(grad_X)
    fn, scale, base = (i[k].detach().to(params_dtype).requires_grad_(grad_params) for k in ("fn", "scale", "base"))
    B = i["B"].detach().clone().requires_grad_(True)
    U, handle = ops.mhc_prepare(X, fn, scale, base, **constants)
    Y = ops.mhc_finish(X, B, handle)
    torch.autograd.backward([U, Y], [i["dU"], i["dY"]])
    out = {"U": U.detach(), "Y": Y.detach(), "dB": B.grad, "handle": type(handle)}
    if grad_X:
        out["dX"] = X.grad
    if grad_params:
        out.update(dfn=fn.grad, dscale=scale.grad, dbase=base.grad)
    return out


def tensors(out: dict) -> dict[str, torch.Tensor]:
    """The tensors of a :func:`run` result (its handle type left out)."""
    return {name: value for name, value in out.items() if name != "handle"}


def _sinkhorn(logits: torch.Tensor, iters: int, hc_eps: float) -> torch.Tensor:
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def _reference_block(X, B, dY, dU, fn, scale, base, iters, rms_eps, hc_eps) -> dict:
    """FP64 forward of one row block, its backward by autograd, and ``sum_t |dz'_t| |u_t|`` for the order bound."""
    T = X.shape[0]
    with torch.enable_grad():
        Xd, fnd, sd, bd, Bd = (t.detach().to(F64).requires_grad_(True) for t in (X, fn, scale, base, B))
        u = Xd.reshape(T, -1)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + rms_eps)
        z = (u * r) @ fnd.T
        pre = torch.sigmoid(z[:, 0:4] * sd[0] + bd[0:4]) + hc_eps
        post = 2 * torch.sigmoid(z[:, 4:8] * sd[1] + bd[4:8])
        C = _sinkhorn((z[:, 8:] * sd[2] + bd[8:]).reshape(T, STREAMS, STREAMS), iters, hc_eps)
        U = (pre.unsqueeze(-1) * Xd).sum(dim=1)
        Y = post.unsqueeze(-1) * Bd.unsqueeze(1) + torch.einsum("tij,tid->tjd", C, Xd)
        dX, dfn, dscale, dbase, dB, dz = torch.autograd.grad(
            [U, Y], [Xd, fnd, sd, bd, Bd, z], grad_outputs=[dU.to(F64), dY.to(F64)]
        )
    with torch.no_grad():
        dfn_abs = (dz * r).abs().T @ u.abs()
    return {
        "U": U.detach(),
        "Y": Y.detach(),
        "dB": dB,
        "dX": dX,
        "dfn": dfn,
        "dscale": dscale,
        "dbase": dbase,
        "dfn_abs": dfn_abs,
    }


def reference(
    i: dict[str, torch.Tensor],
    *,
    sinkhorn_iters: int = ops.SINKHORN_ITERS,
    rms_eps: float = ops.RMS_EPS,
    hc_eps: float = ops.HC_EPS,
) -> dict[str, torch.Tensor]:
    """The FP64 outputs and gradients of the connection on ``i`` (``dfn_abs``: the scale of ``dfn``'s rounding)."""
    T = i["X"].shape[0]
    blocks = [
        _reference_block(
            *(i[k][lo : lo + BLOCK_TOKENS] for k in ("X", "B", "dY", "dU")),
            i["fn"],
            i["scale"],
            i["base"],
            sinkhorn_iters,
            rms_eps,
            hc_eps,
        )
        for lo in range(0, T, BLOCK_TOKENS)
    ]
    out = {k: torch.cat([b[k] for b in blocks]) for k in BF16_OUTPUTS}
    for k in (*PARAMETER_GRADS, "dfn_abs"):
        out[k] = sum(b[k] for b in blocks[1:]) + blocks[0][k]
    return out
