# miles-ext-kernels

Training kernels for [miles](https://github.com/radixark/miles), consumed as the `3rdparty/miles-ext-kernels`
submodule and enabled by miles' `--use-miles-ext-kernels`.

## Contract with miles

| Owned by miles | Owned by this repo |
|---|---|
| `--use-miles-ext-kernels` | kernel sources, the loader, host/ops glue |
| backend selection (`--linear-attention-backend cake`, ...) and the `cake` choice | every `--cake-*` argument (no `MILES_*` env knobs) |
| the dispatch seam: `if backend == "cake": from miles_ext_kernels.cake.<op> import ...` | kernel tests (CPU and GPU) |
| `CAKE_BACKEND_ARGS` in `miles/utils/arguments.py`: `cake` without `--use-miles-ext-kernels` is rejected | |

Under `--use-miles-ext-kernels`, miles calls `miles_ext_kernels.add_arguments(parser)` before parsing,
`validate_args(args)` from `miles_validate_args`, `configure(args.miles_ext_kernels_args)` in every process that runs
a kernel, and `release()` before it sleeps or offloads. Without the flag nothing here is imported.

## Layout

```
miles_ext_kernels/cake/
  _runtime/    # variants, arguments, loader, prebuild, workspace
  _impl/<op>/  # op implementations
  _vendor/     # written only by tools/vendor_cake.py, recorded in VENDORED.json
  <op>.py      # one thin module per op: its OP and entry points
tests/cake/<op>/
```

## Adding a kernel

1. PR here: vendor its packages with `tools/vendor_cake.py`, add `cake/<op>.py` and `cake/_impl/<op>/`, append the op
   to `OPS` in `cake/_runtime/registry.py`, and add its tests under `tests/cake/<op>/`.
2. PR in miles: bump the submodule, add `cake` to the backend's choices, add the dispatch seam, and add the backend
   argument to `CAKE_BACKEND_ARGS`. GPU CI runs there.

## Development

```bash
pip install -e .[test]
pre-commit run --all-files
pytest tests
python tools/vendor_cake.py --check
```
