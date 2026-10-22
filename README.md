# miles-ext-kernels

Training kernels for [miles](https://github.com/radixark/miles), consumed as the `3rdparty/miles-ext-kernels`
submodule and enabled by miles' `--use-miles-ext-kernels`.

## Contract with miles

| Owned by miles | Owned by this repo |
|---|---|
| `--use-miles-ext-kernels` | kernel sources, JIT loader, host/ops glue |
| backend selection (`--linear-attention-backend cake`, ...) and the `cake` choice | every per-kernel argument (no `MILES_*` env knobs) |
| the dispatch seam: `if backend == "cake": from miles_ext_kernels.<kernel> import ...` | kernel tests (CPU and GPU) |
| `CAKE_BACKEND_ARGS` in `miles/utils/arguments.py`: `cake` without `--use-miles-ext-kernels` is rejected | |

Under `--use-miles-ext-kernels`, miles calls `miles_ext_kernels.add_arguments(parser)` before parsing and
`miles_ext_kernels.validate_args(args)` from `miles_validate_args`. Without the flag nothing here is imported.

## Layout (target)

```
miles_ext_kernels/
  __init__.py      # _KERNEL_PACKAGES: each exposes add_arguments(group) and validate_args(args)
  cake_native/     # shared JIT loader + host shim
  gdn/             # chunk_gated_delta_rule (Qwen GDN)
  kda/             # chunked KDA (Kimi-K3)
  dsa/             # DSA indexer, sparse attention, top-k (GLM-5, DeepSeek-V4)
tests/
```

## Adding a kernel

1. PR here: the package under `miles_ext_kernels/<kernel>/`, its arguments in `add_arguments` / `validate_args`
   (registered in `_KERNEL_PACKAGES`), and its tests.
2. PR in miles: bump the submodule, add `cake` to the backend's choices, add the dispatch seam, and add the
   backend argument to `CAKE_BACKEND_ARGS`. GPU CI runs there.

## Development

```bash
pip install -e .
pre-commit run --all-files
pytest tests
```
