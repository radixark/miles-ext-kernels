# miles-ext-kernels

Training kernels for [miles](https://github.com/radixark/miles), consumed as the `3rdparty/miles-ext-kernels`
submodule and enabled by miles' `--use-miles-ext-kernels`.

Kernels are grouped by source, one namespace each with the same shape: a runtime (`_runtime/`), one tree of vendored
sources (`_vendor/`) and one thin module per op. Today there is one source, `miles_ext_kernels.cake`, the Cake training
kernels.

## Install

```bash
pip install -e .            # torch, apache-tvm-ffi>=0.1.11,<0.2
pip install -e .[test]      # + pytest
```

| extra | for |
|---|---|
| `test` | the tests (`pytest`) |

The kernels are compiled with nvcc on the machine that runs them (`$CUDA_HOME/bin/nvcc`, else `nvcc` on `PATH`) and
loaded with tvm-ffi; FlashInfer itself is not needed. Each architecture needs a minimum toolkit:

| architecture | GPUs | nvcc |
|---|---|---|
| `sm_100a` | B200, GB200 | 12.8 |
| `sm_103a` | B300, GB300 | 12.9 |
| `sm_107a` | R200 | 13.4 |

Libraries are cached under `${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}/miles_ext_kernels/`; one rank builds each library
under a file lock while the others wait. For multi-node jobs on a cluster-wide filesystem (Lustre), use a node-local
or per-job `TVM_FFI_CACHE_DIR`.

## Using it from miles

| Owned by miles | Owned by this repo |
|---|---|
| `--use-miles-ext-kernels` | kernel sources, the loader, host code |
| backend selection (`--linear-attention-backend cake`, ...) and the `cake` choice | every `--cake-*` argument (no environment knobs) |
| the dispatch seam: `from miles_ext_kernels.cake.<op> import ...` | kernel tests (CPU here, GPU in miles CI) |
| `CAKE_BACKEND_ARGS`: `cake` without `--use-miles-ext-kernels` is rejected | |

miles calls four hooks of `miles_ext_kernels`; importing it imports neither torch, tvm-ffi nor any op.

| hook | when | what |
|---|---|---|
| `add_arguments(parser)` | before parsing | registers every `--cake-*` argument in one `miles-ext-kernels` group |
| `validate_args(args)` | on the parsed flat namespace | rejects options of unselected ops, resolves and writes back each selected op's variant |
| `configure(args.miles_ext_kernels_args)` | in every process that calls a kernel | sets this process's variants and options |
| `release()` | before miles sleeps or offloads | frees the workspace pool and cached plans |

An op module is a thin declaration: `miles_ext_kernels.cake.<op>.OP` lists the op's variants, arguments and vendored
packages, and its entry points import the implementation on first use.

## Ops

| op | entry points | miles selector | flags | variants | GPUs | prebuild params | tests |
|---|---|---|---|---|---|---|---|

An op's vendored packages, their sources and commits are its block in `miles_ext_kernels/cake/_vendor/VENDORED.json`.

## Variants and determinism

`det` repeats bit for bit for identical inputs (same GPU model and SM count, toolkit, build and route); `fast` is
faster. A process that never configured an op runs `det`; miles' command line defaults to `fast`, and
`--deterministic-mode` forces `det` and rejects an explicit `fast`. The full rules are in
`miles_ext_kernels/cake/_runtime/config.py`.

## Prebuild

`miles_ext_kernels.cake.prebuild(op, **model_params)` builds every library `op` may launch for these model shapes,
so no later call compiles, whatever the variant and token count (contract: `registry.prebuild`; each op checks it in
`tests/cake/<op>/test_prebuild_gpu.py`).

| op | model_params |
|---|---|

## Memory

`--cake-workspace-pool-bytes` bounds the per-device workspace pool; the default `0` keeps nothing between calls.
`miles_ext_kernels.release()` frees the pool and every op's cached plans (details: `_runtime/workspace.py`).

## Layout

```
miles_ext_kernels/
  __init__.py              # SOURCES, add_arguments, validate_args, configure, release
  cake/
    __init__.py            # the runtime's public API
    _runtime/              # spec (OpSpec, Flag), registry (OPS, prebuild, release), config, loader, workspace
    _impl/<op>/            # an op's implementation
    _vendor/               # vendored packages, one level deep, and VENDORED.json
    <op>.py                # one thin module per op: OP and its lazy entry points
tests/cake/_runtime/       # the runtime, the vendoring tool and the op contract
tests/cake/<op>/           # an op's tests
tools/vendor_cake.py       # the one vendoring entry point
tools/templates/           # FlashInfer's package loader and the TVM-FFI host surface
```

## Vendored sources

Everything under `miles_ext_kernels/cake/_vendor/` is written by `tools/vendor_cake.py` byte for byte from an upstream
commit (FlashInfer, or generated sources of a miles pull request) and recorded in `_vendor/VENDORED.json`; never edit
it in place. Usage is in the docstring of `tools/vendor_cake.py`.

## Adding an op

1. Vendor its packages: `tools/vendor_cake.py fi ...` / `miles-pr ...` for the op.
2. `miles_ext_kernels/cake/<op>.py`: `OP = OpSpec(...)` (variants, miles selector, packages, entry points, options)
   and `__all__, __getattr__, __dir__ = lazy_exports(__name__, OP)`.
3. `miles_ext_kernels/cake/_impl/<op>/`: `loader.install_op("<op>")`, the entry points, `prebuild(*, device=None,
   max_workers=None, **model_params)` and `release()`; read `config.variant` / `config.option` at call time and take
   temporary buffers from `workspace.take`.
4. Append `"<op>",` to `OPS` in `miles_ext_kernels/cake/_runtime/registry.py`.
5. Add the op's rows to the Ops and Prebuild tables above.
6. `tests/cake/<op>/`: `__init__.py`, CPU tests and `test_prebuild_gpu.py` (`ROUTES` and `check_prebuild`).

`tests/cake/_runtime/test_op_contract.py` checks every registered op against this list.

## Development

```bash
pip install -e .[test]
pre-commit run --all-files
pytest tests                                   # CPU here; GPU tests skip without CUDA and run in miles CI
python tools/vendor_cake.py --check
git config merge.vendored.driver "python tools/vendor_cake.py merge %O %A %B"
```

The last line enables the `VENDORED.json` merge driver named in `.gitattributes`.
