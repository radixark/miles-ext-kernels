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
| `dense_gemm` | the dense GEMM packages call the CUDA driver through these bindings for the L2 size (where torch does not report it) and for the zero fill of the FP32 router gradients under `fast` (`cuda-bindings>=12.8`) |

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
| `rmsnorm` | `ZeroCenteredRMSNorm`, `zero_centered_rmsnorm`, `check_supported`, `supported_hidden_sizes` | `--linear-attention-input-norm cake` | `--cake-rmsnorm-variant` | `det`, `fast` | `sm_100a`, `sm_103a`, `sm_107a` | `hidden_size` | `tests/cake/rmsnorm/` |
| `dsa` | `sparse_attention`, `sparse_attention_forward`, `sparse_attention_backward`, `lightning_indexer`, `indexer_logits`, `indexer_backward`, `indexer_topk_scores`, `extract_topk_scores`, `select_topk`, `torch_topk`, `packed_key_ids`, `dsa_indexer_det_topk`, `dsa_indexer_det_topk_workspace_size`, `check_device`, `fast_unserved_reason`, `indexer_unserved_reason`, `indexer_det_topk_unserved_reason` | `--dsa-attention-backend cake` | `--cake-dsa-variant`, `--cake-dsa-bwd-workspace-bytes`, `--cake-dsa-lazy-build` | `det`, `fast` (64 heads, no sink; other calls run `det`) | `sm_100a`, `sm_103a`, `sm_107a` | `heads`, `d_qk`, `d_v`, `attn_sink`, `index_heads` | `tests/cake/dsa/` |
| `mhc` | `mhc_prepare`, `mhc_finish`, `mhc_head`, `is_supported`, `supported_hidden_sizes` | `--mhc-backend cake` | `--cake-mhc-variant` | `det`, `fast` | `sm_100a`, `sm_103a`, `sm_107a` | `hidden`, `sinkhorn_iters` | `tests/cake/mhc/` |
| `gqa` | `varlen_gqa_attention`, `check_supported`, `is_supported` | `--full-attention-backend cake` | `--cake-gqa-check-seqlens` (checks at every call that the boundaries are well formed and that `max_seqlen_k` is at least every segment's key count, as the kernels require: a smaller one gives wrong gradients without an error; one device synchronization) | `fast` only (dQ is summed with FP32 atomic adds); `--deterministic-mode` rejects the op | `sm_100a`, `sm_103a`, `sm_107a` | none | `tests/cake/gqa/` |
| `dense_gemm` | `batched_projection`, `router_gating`, `check_projection_dims`, `fast_projection_programs`, `fast_router_programs` | `--batched-projection-gemm-backend cake`, `--moe-router-gemm-backend cake` (Megatron only) | `--cake-dense-gemm-variant` hidden, always `det` | `det`, `fast` (miles runs `det`; `fast` changes only weight-gradient programs, on the rows and token counts its tables select for the device; the op module lists the rows where they are known to miss `det`'s BF16 accuracy budget); not batch-invariant: a token's results can change in their last bits with the token count of the call | `sm_100a`, `sm_103a`, `sm_107a` | `projections`, `router` | `tests/cake/dense_gemm/` |

`det` repeats bit for bit; `fast` is faster and is miles' default; `--deterministic-mode` forces `det`.

## Prebuild

| op | model_params |
|---|---|
| `gdn` | none: every library of its packages for the device's architecture (FLA's own Triton pre-processing, which a `cp_context` with a process group runs, compiles in FLA on its first call) |
| `kda` | `num_heads` (1 to 64 heads per call, checked); builds every library of its four packages for the architecture (188 / 186 / 189 on `sm_100a` / `sm_103a` / `sm_107a`), as the library a call takes depends on its token and document counts, modes and route. From an empty cache, `prebuild("kda", num_heads=12, max_workers=32)` took 75 s on one B200 node and 112 s on one GB300 node (48 CPUs, nvcc 13.0); the cache keeps them |
| `conv1d` | `weight_dtype`: `torch.float32` or `torch.bfloat16`, the taps' dtype (builds both variants, both SiLU' policies and the context-parallel route) |
| `gated_rmsnorm` | none: both variants, every row-count route |
| `rmsnorm` | `hidden_size`: 256, 512, 1536, 2048, 4096, 5120 or 6144 (every architecture); both variants, every token-count window. On `sm_107a`, 4096 launches programs above the 232448 B shared-memory opt-in in CUDA 13.4's oversized mode: where the process's CUDA runtime is older than 13.4 (for example the 13.0 runtime wheel of a `cu130` torch) or the driver's oversized ceiling does not hold them, `prebuild`, `check_supported`, the module's construction and the first call raise `RuntimeError` before any launch |
| `dsa` | `heads`, `d_qk`, `d_v=512`, `attn_sink=False`, `index_heads=None` (32 also builds the det indexer top-k); `--cake-dsa-lazy-build` builds nothing and gives up the guarantee |
| `mhc` | `hidden` (4096, 5120, 7168 on `sm_100a` and `sm_103a`; 4096 on `sm_107a`) and `sinkhorn_iters` (1 to 64), both required: both variants and the head; the epsilons are launch arguments and select no library |
| `gqa` | none: the four libraries of the forward and the BF16 dK / dV backward serve every head layout and length |
| `dense_gemm` | `projections`: `(num_heads, d_in, d_out, weight_layout)` tuples; `router`: `(num_experts, hidden_size)` tuples. The plan of a GEMM depends on the token count, so it builds every program the two packages register for the architecture for the GEMM kinds given (the projections, the router), not one plan per shape |

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
