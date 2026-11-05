"""The backward programs of the vendored Cake RMSNorm packages for the FP32 effective weight, read from their registry
(no GPU): which programs a call of ``T`` tokens launches and in which form."""

from miles_ext_kernels.cake._runtime import loader

PACKAGES = {"det": "cake_rmsnorm_det_train_backend", "fast": "cake_rmsnorm_train_backend"}


def _records(variant, arch, hidden, T):
    jit = loader.load_package(PACKAGES[variant]).cake_jit
    return [jit.MODULES[name] for name in jit.selected_modules(arch, hidden, "bwd", False, rows=T, w_dtype="f32")]


def backward_programs(variant, arch, hidden, T):
    """The programs (library names without the package prefix) ``variant``'s backward of ``T`` tokens launches on
    ``arch``, in launch order."""
    return tuple(record["library"].rsplit("_", 1)[-1] for record in _records(variant, arch, hidden, T))


def backward_form(variant, arch, hidden, T):
    """``"single"``: one kernel that adds its dw sums with FP64 atomics or sums each dw column in a fixed row order
    (``"column"`` for the latter); ``"pair"``: a main kernel and its fixed-order dw reduce."""
    records = _records(variant, arch, hidden, T)
    owners = [record["workspace_rule"]["kind"] for record in records if record["workspace_rule"]]
    if len(records) == 1 and owners == ["claim_sum_f64"]:
        return "column" if records[0]["role"].startswith("bwd_col") else "single"
    assert len(records) == 2 and owners == ["chunk_partials"], [record["library"] for record in records]
    return "pair"
