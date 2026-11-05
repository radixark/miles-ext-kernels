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

Kernels are compiled with nvcc on first use or by `prebuild` (sm_100a: CUDA 12.8+, sm_103a: 12.9+, sm_107a: 13.4+)
and cached under `${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}`.

## Using it from miles

miles owns `--use-miles-ext-kernels`, backend selection and the dispatch seam
(`from miles_ext_kernels.cake.<op> import ...`). This repo owns every `--cake-*` argument through four hooks:
`add_arguments`, `validate_args`, `configure` and `release`.

## Ops

| op | entry points | miles selector | flags | variants | GPUs | prebuild params | tests |
|---|---|---|---|---|---|---|---|

`det` repeats bit for bit; `fast` is faster and is miles' default; `--deterministic-mode` forces `det`.

## Prebuild

| op | model_params |
|---|---|

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
