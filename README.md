# miles-ext-kernels

Training kernels for [miles](https://github.com/radixark/miles), consumed as the `3rdparty/miles-ext-kernels`
submodule and enabled by miles' `--use-miles-ext-kernels`. Kernels are grouped by source; today that is
`miles_ext_kernels.cake`.

## Install

```bash
pip install -e .[test]
```

| extra | for |
|---|---|
| `test` | the tests (`pytest`) |
| `gdn` | `gdn` with an FLA `cp_context` (`flash-linear-attention>=0.5.2`) |

Kernels are compiled with nvcc on first use or by `prebuild` (sm_100a: CUDA 12.8+, sm_103a: 12.9+, sm_107a: 13.4+)
and cached under `${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}`.

## Using it from miles

miles owns `--use-miles-ext-kernels`, backend selection and the dispatch seam
(`from miles_ext_kernels.cake.<op> import ...`). This repo owns every `--cake-*` argument through four hooks:
`add_arguments`, `validate_args`, `configure` and `release`.

## Ops

| op | entry points | miles selector | flags | variants | GPUs | prebuild params | tests |
|---|---|---|---|---|---|---|---|
| `gdn` | `chunk_gated_delta_rule`, `kcp_chunk_gated_delta_rule`, `build_kcp_context`, `KcpContext` | `--linear-attention-backend cake` | `--cake-gdn-variant`, `--cake-gdn-recompute-state` | `det`, `fast` | `sm_100a`, `sm_103a`, `sm_107a` | none | `tests/cake/gdn/` |
| `kda` | `chunk_kda`, `chunk_kda_fwd`, `chunk_kda_bwd`, `partition_kcp`, `build_context_kcp`, `chunk_kda_kcp`, `chunk_kda_fwd_kcp`, `chunk_kda_bwd_kcp`, `reduce_gate_param_grads_kcp` | `--kda-backend cake` | `--cake-kda-variant` hidden, always `det` | `det`, `fast` (miles runs `det`: its calls take the in-kernel q / k L2 norm, which only `det` serves) | `sm_100a`, `sm_103a`, `sm_107a` | `num_heads` | `tests/cake/kda/` |
| `conv1d` | `causal_conv1d`, `causal_conv1d_kcp`, `causal_conv1d_kcp_multi`, `check_supported` | `--linear-attention-conv-backend cake` | `--cake-conv1d-variant`, `--cake-conv1d-silu-grad-preactivation` | `det`, `fast` | `sm_100a`, `sm_103a`, `sm_107a` | `weight_dtype` | `tests/cake/conv1d/` |
| `gated_rmsnorm` | `rmsnorm_silu_gate`, `check_supported` | `--linear-attention-output-norm-backend cake` | `--cake-gated-rmsnorm-variant` | `det`, `fast` | `sm_100a`, `sm_103a`, `sm_107a` | none | `tests/cake/gated_rmsnorm/` |

`det` repeats bit for bit; `fast` is faster and is miles' default; `--deterministic-mode` forces `det`.

## Prebuild

| op | model_params |
|---|---|
| `gdn` | none: every library of its packages for the device's architecture (FLA's own Triton pre-processing, which a `cp_context` with a process group runs, compiles in FLA on its first call) |
| `kda` | `num_heads` (1 to 64 heads per call, checked); builds every library of its four packages for the architecture (188 / 186 / 189 on `sm_100a` / `sm_103a` / `sm_107a`), as the library a call takes depends on its token and document counts, modes and route. From an empty cache, `prebuild("kda", num_heads=12, max_workers=32)` took 75 s on one B200 node and 112 s on one GB300 node (48 CPUs, nvcc 13.0); the cache keeps them |
| `conv1d` | `weight_dtype`: `torch.float32` or `torch.bfloat16`, the taps' dtype (builds both variants, both SiLU' policies and the context-parallel route) |
| `gated_rmsnorm` | none: both variants, every row-count route |

## Layout

```
miles_ext_kernels/cake/
  _runtime/    # variants, arguments, loader, prebuild, workspace
  _impl/<op>/  # op implementations
  _vendor/     # written only by tools/vendor_cake.py, recorded in VENDORED.json
  <op>.py      # one thin module per op
tests/cake/<op>/
```

Adding an op: vendor its packages with `tools/vendor_cake.py`, add `cake/<op>.py` and `cake/_impl/<op>/`, append it
to `OPS` in `_runtime/registry.py`, add its rows to the tables above and its tests under `tests/cake/<op>/`.

## Development

```bash
pre-commit run --all-files
pytest tests
python tools/vendor_cake.py --check
```
