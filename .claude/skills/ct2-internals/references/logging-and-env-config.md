# Logging & environment configuration

The operational debugging card: spdlog wiring, what each `CT2_VERBOSE` level prints, and
the complete (grepped, verified) environment-variable surface of the engine.

**Sources (all citations from real lines):**

- `src/logging.cc`, `include/ctranslate2/logging.h`, `python/cpp/logging.cc`
- `src/env.cc` / `src/env.h` (`read_{string,bool,int}_from_env`)
- env-var read sites: `src/cpu/backend.cc`, `src/cpu/cpu_isa.cc`, `src/types.cc`,
  `src/utils.cc`, `src/cuda/allocator.cc`, `src/cuda/utils.cc`, `src/cuda/cublas_stub.cc`

## 1. spdlog wiring

`init_logger()` (`src/logging.cc:64-72`) creates a `stderr_logger_mt("ctranslate2")` once
(`std::once_flag`) with pattern `[date] [ctranslate2] [thread t] [level] msg` and sets the
level from `CT2_VERBOSE`. It's invoked lazily from `log_system_config`
(`src/utils.cc:29`, which runs on model load) and from `set_log_level`/`get_log_level`
(`logging.cc:74-82`) — also bound to Python as `ctranslate2.set_log_level(LogLevel)`
(`python/cpp/logging.cc:19-20`).

`LogLevel` (`include/ctranslate2/logging.h:5-13`) is numerically aligned with
`CT2_VERBOSE`: `Off=-3, Critical=-2, Error=-1, Warning=0, Info=1, Debug=2, Trace=3`.
`get_default_level` reads the int and rejects values outside [-3,3]
(`logging.cc:54-62`); **default is 0 = Warning**.

What each level adds (call-site survey):

- `0` Warning — compute-type fallback warnings (`src/models/model.cc:883-889`), tensor-parallel
  caveats. The quiet default.
- `1` Info — the load banner: model path/binary version/spec revision/selected compute type
  (`model.cc:872-879`) plus `log_system_config`'s CPU/ISA/GEMM-backend/GPU capability dump
  (`src/utils.cc:28-71`). **First thing to turn on** when a backend picks the wrong path.
- `2` Debug — per-batch tracing: "Running batch translation on N examples" / "Finished"
  (`src/translator.cc:176-190`, same in `generator.cc`).
- `3` Trace — reserved; effectively nothing beyond Debug in the engine today.

## 2. The env helpers

`src/env.cc:9-23`: `read_string_from_env` (getenv with default),
`read_bool_from_env` (via `string_to_bool` — "1"/"true"/"TRUE", `src/utils.cc:24-26`),
`read_int_from_env`. All env reads below go through these (one raw `getenv` exception noted).

## 3. The real env-var table

Grepped `getenv|read_.*_from_env` across `src/` — this is the complete list:

| Variable                            | Effect                                                                                                           | Read at                      |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `CT2_VERBOSE`                       | Log level int −3…3 (see §1); default 0=Warning                                                                   | `src/logging.cc:55`          |
| `CT2_FORCE_CPU_ISA`                 | Force ISA: `AVX512`/`AVX2`/`AVX`/`NEON`/`GENERIC`; **AVX512 is env-only, never auto-selected** (`cpu_isa.cc:65`) | `src/cpu/cpu_isa.cc:45`      |
| `CT2_USE_MKL`                       | Force MKL on/off (default: genuine-Intel CPU only); throws if set truthy in a non-MKL build                      | `src/cpu/backend.cc:11`      |
| `CT2_PACKED_GEMM`                   | Enable MKL packed-GEMM weight prepacking (default **true**; only consulted when the GEMM backend is MKL)         | `src/cpu/backend.cc:104`     |
| `OMP_NUM_THREADS`                   | Fallback intra-op thread count when `intra_threads`=0 (then default `min(4, hw)`)                                | `src/utils.cc:87`            |
| `CT2_CUDA_ALLOW_FP16`               | Permit fp16 compute on pre-Tensor-Core GPUs                                                                      | `src/types.cc:114`           |
| `CT2_CUDA_ALLOW_BF16`               | Permit bf16 compute on pre-Ampere GPUs                                                                           | `src/types.cc:98`            |
| `CT2_CUDA_TRUE_FP16_GEMM`           | fp16 GEMM accumulates in fp16 (default true); set 0 for fp32 accumulation                                        | `src/cuda/utils.cc:260`      |
| `CT2_CUDA_ALLOCATOR`                | `cub_caching` vs `cuda_malloc_async` (default: malloc_async when supported)                                      | `src/cuda/allocator.cc:146`  |
| `CT2_CUDA_CACHING_ALLOCATOR_CONFIG` | `bin_growth,min_bin,max_bin,max_cached_bytes` for the cub allocator (defaults 4,3,12,200MB); raw `getenv`        | `src/cuda/allocator.cc:42`   |
| `CUDA_PATH`                         | Search prefix when dlopen-ing cublas (`CUDA_DYNAMIC_LOADING` builds)                                             | `src/cuda/cublas_stub.cc:56` |

Where the prompt-era folklore differs from reality: there is **no**
`CT2_USE_EXPERIMENTAL_PACKED_GEMM` (it's plain `CT2_PACKED_GEMM`, default-on), no
`CT2_CPU_BACKEND`, and no `CT2_TRANSLATORS_CORE_OFFSET` — core pinning is the
`ReplicaPoolConfig.cpu_core_offset` **API field**, not an env var (see
parallelism-and-thread-config.md §5).

## 4. Metal: zero env vars

`grep CT2_ src/metal/` finds only MSL threadgroup-size constants (`CT2_SOFTMAX_TG`,
`CT2_NORM_TG`, `CT2_QUANT_TG`, `CT2_GEMM_S8_BM/BN` in `kernels/kernels_msl.h`) — **no
environment variables**. `CT2_NO_MPS_ACT`, the per-op CPU-reference bisection switch used
during the Gemma2 NaN hunt, **no longer exists**; it was a temporary debugging gate and
was removed with the fix. Don't cite it as a live knob.

### Relevance to the Metal backend

- `CT2_VERBOSE=1` is the fastest check that a Metal run resolved the compute type you
  think it did ("Selected compute type", `model.cc:878-879`) — e.g. confirming int8 stayed
  int8 instead of falling back.
- On this machine's `OPENMP_RUNTIME=NONE` builds, `OMP_NUM_THREADS` still works — it feeds
  `cpu::set_num_threads` for the BS pool, sizing every CPU-reference op that runs on
  Metal-resident data.
- `CT2_FORCE_CPU_ISA=GENERIC` vs `NEON` is a useful A/B when bisecting whether a Metal
  parity failure is actually a NEON CPU-kernel issue on the reference side.
- The Metal backend deliberately has no env switches; debugging toggles (like the late
  `CT2_NO_MPS_ACT`) are added temporarily and removed — re-grep before assuming one exists.
