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
| `gdn` | `gdn` with an FLA `cp_context` (`flash-linear-attention>=0.5.2`) |

The kernels are compiled with nvcc on the machine that runs them (`$CUDA_HOME/bin/nvcc`, else `nvcc` on `PATH`) and
loaded with tvm-ffi; FlashInfer itself is not needed. Each architecture needs a minimum toolkit:

| architecture | GPUs | nvcc |
|---|---|---|
| `sm_100a` | B200, GB200 | 12.8 |
| `sm_103a` | B300, GB300 | 12.9 |
| `sm_107a` | R200 | 13.4 |

Libraries are cached under `${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}/miles_ext_kernels/<package>/`, never in the
package directory, keyed by the flags, the nvcc version, the tvm-ffi version and the CPU architecture. One rank builds
each library under a file lock; the others wait, log every 30 s and fail after 30 minutes. On a filesystem whose locks
are cluster-wide, such as Lustre, a rank killed on another node can leave its lock held for minutes: for multi-node
jobs prefer a node-local or per-job `TVM_FFI_CACHE_DIR`.

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
| `add_arguments(parser)` | before parsing | one `miles-ext-kernels` group: `--cake-<op>-variant` and the `--cake-<op>-...` options of every op miles can select, and `--cake-workspace-pool-bytes`; the variant flag of an op that ships two variants but lets miles choose one is hidden, and only carries the resolved variant |
| `validate_args(args)` | on the parsed flat namespace | rejects an option of an op that is not selected, resolves each selected op's variant and writes it back (so it reaches every process), runs the op's own checks |
| `configure(args.miles_ext_kernels_args)` | in every process that calls a kernel | replaces the process configuration (variants and options) |
| `release()` | before miles sleeps or offloads | frees the workspace pool and the ops' cached plans; returns the bytes freed |

An op module is a thin declaration: `miles_ext_kernels.cake.<op>.OP` lists the op's variants, arguments and vendored
packages, and its entry points import the implementation on first use.

## Ops

| op | entry points | miles selector | flags | variants | GPUs | prebuild params | tests |
|---|---|---|---|---|---|---|---|
| `gdn` | `chunk_gated_delta_rule`, `kcp_chunk_gated_delta_rule`, `build_kcp_context`, `KcpContext` | `--linear-attention-backend cake` | `--cake-gdn-variant`, `--cake-gdn-recompute-state` | `det`, `fast` | `sm_100a`, `sm_103a`, `sm_107a` | none | `tests/cake/gdn/` |
| `kda` | `chunk_kda`, `chunk_kda_fwd`, `chunk_kda_bwd`, `partition_kcp`, `build_context_kcp`, `chunk_kda_kcp`, `chunk_kda_fwd_kcp`, `chunk_kda_bwd_kcp`, `reduce_gate_param_grads_kcp` | `--kda-backend cake` | `--cake-kda-variant` hidden, always `det` | `det`, `fast` (miles runs `det`: its calls take the in-kernel q / k L2 norm, which only `det` serves) | `sm_100a`, `sm_103a`, `sm_107a` | `num_heads` | `tests/cake/kda/` |
| `conv1d` | `causal_conv1d`, `causal_conv1d_kcp`, `causal_conv1d_kcp_multi`, `check_supported` | `--linear-attention-conv-backend cake` | `--cake-conv1d-variant`, `--cake-conv1d-silu-grad-preactivation` | `det`, `fast` | `sm_100a`, `sm_103a`, `sm_107a` | `weight_dtype` | `tests/cake/conv1d/` |
| `gated_rmsnorm` | `rmsnorm_silu_gate`, `check_supported` | `--linear-attention-output-norm-backend cake` | `--cake-gated-rmsnorm-variant` | `det`, `fast` | `sm_100a`, `sm_103a`, `sm_107a` | none | `tests/cake/gated_rmsnorm/` |
| `rmsnorm` | `ZeroCenteredRMSNorm`, `zero_centered_rmsnorm`, `check_supported`, `supported_hidden_sizes` | `--linear-attention-input-norm cake` | `--cake-rmsnorm-variant` | `det`, `fast` | `sm_100a`, `sm_103a`, `sm_107a` | `hidden_size` | `tests/cake/rmsnorm/` |

An op's vendored packages, their sources and commits are its block in `miles_ext_kernels/cake/_vendor/VENDORED.json`.

## Variants and determinism

`det` repeats bit for bit for identical inputs; `fast` is faster. An op ships both, or one of them.

| case | variant |
|---|---|
| the process never called `configure`, or gave no value for the op | `det` (an op without `det`: its only variant) |
| `validate_args`, op selected, flag unset | `fast` if miles may choose it, else `det` (miles' command line defaults to `fast`) |
| `validate_args`, `--deterministic-mode`, flag unset | `det`; an op without `det` is rejected |
| `--deterministic-mode` (or `configure(..., deterministic_mode=True)`) with an explicit `fast` | rejected |
| `configure(..., deterministic_mode=True)`, no value for an op without `det` | `variant()` raises at the call |
| a value the op does not ship, or an unknown `cake_*` option, in `configure` | rejected |
| at a call | the op reads its variant when the call runs; a backward runs the variant of its forward |

Scope of `det`: det repeats bit for bit for identical inputs on the same GPU model and SM count, the same toolkit,
the same build of this package and the same route (host or device tables, with or without context parallelism, the
same partition). Bits may differ across versions, GPU models, SM counts and routes.

A call that `fast` does not serve runs `det` and logs it once per op and reason.

## Prebuild

`miles_ext_kernels.cake.prebuild(op, *, device=None, max_workers=None, **model_params)` builds, or finds in the cache,
every library any variant of `op` may launch for these model parameters on the device's architecture: every stage,
token-count window, route and option. After it returns, no call of `op` with these model parameters compiles on that
device, whatever variant is configured later and whatever the token count. `model_params` are model shapes only (head
counts, hidden sizes, dtypes, iteration counts), never token counts or variants. It returns the cache paths, is
idempotent, and raises `ValueError` for unsupported parameters, `NotImplementedError` for an unsupported device and
`RuntimeError` for a toolkit too old for the device. A Triton or Gluon kernel counts as a library: `prebuild` compiles
it too. The only way around the guarantee is an op's explicit lazy-build option (`--cake-dsa-lazy-build`), listed
below, which gives it up. Third-party kernels are outside it: a caller that passes FLA's own context-parallel context
(`cp_context`) also runs FLA's Triton preprocessing, which FLA compiles on its first call and `prebuild` does not warm.
A repeated call with the same op, CUDA device (index and architecture), toolkit (nvcc and its CUDA version), cache
directory, model parameters and op options returns the paths it remembered, without looking at the libraries again,
until `release()`.

Each op checks the guarantee in `tests/cake/<op>/test_prebuild_gpu.py`. The file declares `ROUTES`, one entry per
call-time path of the op: every route and switch a call can take besides the model parameters (with and without
context parallelism, host and device sequence tables, each optional input such as a temperature, each iteration count
a call may pass, a sweep of token counts). `tests/cake/helpers.py:check_prebuild` calls `prebuild` once, then runs
every route, forward and backward, with every variant under `no_build()`, which raises when a library would compile;
an op with Triton or Gluon kernels also checks that the calls add nothing to a private `TRITON_CACHE_DIR`.

| op | model_params |
|---|---|
| `gdn` | none: every library of its packages for the device's architecture (FLA's own Triton pre-processing, which a `cp_context` with a process group runs, compiles in FLA on its first call) |
| `kda` | `num_heads` (1 to 64 heads per call, checked); builds every library of its four packages for the architecture (188 / 186 / 189 on `sm_100a` / `sm_103a` / `sm_107a`), as the library a call takes depends on its token and document counts, modes and route. From an empty cache, `prebuild("kda", num_heads=12, max_workers=32)` took 75 s on one B200 node and 112 s on one GB300 node (48 CPUs, nvcc 13.0); the cache keeps them |
| `conv1d` | `weight_dtype`: `torch.float32` or `torch.bfloat16`, the taps' dtype (builds both variants, both SiLU' policies and the context-parallel route) |
| `gated_rmsnorm` | none: both variants, every row-count route |
| `rmsnorm` | `hidden_size`: 256, 512, 1536, 2048, 4096, 5120 or 6144 (every architecture); both variants, every token-count window. On `sm_107a`, 4096 launches programs above the 232448 B shared-memory opt-in in CUDA 13.4's oversized mode: where the process's CUDA runtime is older than 13.4 (for example the 13.0 runtime wheel of a `cu130` torch) or the driver's oversized ceiling does not hold them, `prebuild`, `check_supported`, the module's construction and the first call raise `RuntimeError` before any launch |

## Memory

Temporary buffers an op asks for come from one workspace pool per process and device, which keeps at most
`--cake-workspace-pool-bytes` per device. The default is `0`: the pool keeps nothing and every request is a plain
allocation on the current stream (in measured GDN training steps on B200 and GB300, a 1 GiB pool made no step faster
and kept up to 0.75 GiB per device between calls). A positive bound reuses buffers across calls: one buffer per
`(op, role)` and CUDA stream, each request evicts the least recently used buffers until the pool fits the bound, and a
request above the bound is allocated on its own. Calls on two streams never share a buffer. While the current stream
captures a CUDA graph, requests bypass the pool (the graph keeps its own memory), so a graph never refers to a pooled
buffer that is freed later. A pooled buffer is valid until the next request of the same `(op, role)` on the same
stream, or `release()`; it is used on that stream only and never kept in an autograd context or returned. The pool is
thread-safe, but two threads must not run the same op on the same stream at once. `miles_ext_kernels.release()` drops
the pool and every op's cached plans, graphs and bindings; it leaves `torch.cuda.empty_cache()` to the caller.
`miles_ext_kernels.cake._runtime.workspace.stats()` reports bytes, entries, hits, misses and evictions per device.

Some vendored packages keep buffers of their own outside the pool, which the bound and `stats()` do not count: the
self-zeroing tail buffers of the `gdn` fast package (at most about 256 MiB per stream) and the per-stream workspaces of
the `dense_gemm` and `moe_grouped_gemm` packages, which grow to the largest call. `release()` frees them. The `conv1d`
and `gated_rmsnorm` packages keep small zero-filled per-stream counters and reserves (hundreds of KiB per stream, a
few MiB per device) for the life of the process.

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

The modules an op owns are `miles_ext_kernels.cake.<op>`, `miles_ext_kernels.cake._impl.<op>` with its submodules,
and `miles_ext_kernels.cake._vendor.<package>` with its submodules for every package of the op's block in
`VENDORED.json`; `miles_ext_kernels.cake._vendor.cake.<package>`, the name FlashInfer's own imports use for some
packages, is the same module (the runtime resolves it; no file lists it). A vendored module imports packages of its
own op only. Everything else (`miles_ext_kernels`, `miles_ext_kernels.cake`, `_runtime`, `_impl`, `_vendor`,
`_vendor.cake`) is shared by every op.

## Vendored sources

Everything under `miles_ext_kernels/cake/_vendor/` is written by `tools/vendor_cake.py` and never edited in place.
`_vendor/VENDORED.json` records, per op and package, the source (`flashinfer` or `miles-pr`, repository, commit, path;
for a miles pull request also its number and the included and excluded paths), the Cake revision, the git blob of every
upstream file, and for generated miles sources the local files the tool writes and the inputs of the `build.json` it
generates. Each package records its own commit, so ops can take different upstream commits.

```bash
# FlashInfer Cake packages
python tools/vendor_cake.py fi --checkout ../flashinfer --commit <sha> --op <op> \
    --package flashinfer/experimental/<name> [--package ...] [--allow-env-read PKG:MODULE] \
    [--allow-flashinfer-import PKG:MODULE:FUNCTION]
# generated sources of a miles pull request, built through the same loader
python tools/vendor_cake.py miles-pr --checkout ../miles --pr <N> --commit <sha> --op <op> \
    --name cake_<op>_<purpose> --path <dir> --include csrc --include registry.json \
    --exclude csrc/cake_host_shim.h --arch-from sm_107a=sm_103a
python tools/vendor_cake.py remove --op <op> --name <package>
python tools/vendor_cake.py --check [OP ...]       # offline: the tree, every blob, every build.json
python tools/vendor_cake.py --check --upstream     # + drift against the recorded upstream commits
```

A miles pull request's generated launchers use a small host surface; the tool replaces the upstream torch-extension
one with `tools/templates/cake_host_shim.h` (TVM-FFI) and writes FlashInfer's loader and a `cake-build/1` `build.json`
generated from the registry, so these sources build with nvcc and load with tvm-ffi like every other package.

Vendored modules neither read the environment nor import flashinfer. `--allow-env-read` lists a module that does;
`--allow-flashinfer-import` lists a function that imports flashinfer inside its body, a FlashInfer hook the op's
implementation must replace when it is imported (`test_op_contract.py` checks it, with flashinfer unimportable).

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

The last line enables the merge driver `.gitattributes` names for `VENDORED.json`: a merge or rebase across layers
merges the manifest package by package, and stops only when both sides changed the same package differently.

Accuracy tests compare every output and every gradient with `tests/cake/helpers.py:assert_rel_rms` against an FP64 (or
FP32) reference, with a budget per tensor of at most twice the measured error (noted next to it with the architecture
and shape); the gate also fails when a 2 % error would pass. `det` tests run the same call twice in one process and
compare the bits (`assert_repeatable`). Tests of pooled buffers configure `tests/cake/helpers.py:POOLED` (a 1 GiB
bound; the fixture `pooled` applies it) and pass it to every further `configure` call, which replaces the whole
configuration. Tests that need several GPUs are marked `multi_gpu`.
