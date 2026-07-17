---
title: "CPU ISA dispatch & vectorized kernels"
summary: >-
  How one CT2 binary carries several ISA variants of the CPU inner loops and
  selects at runtime: the arch-gated CpuIsa enum (AVX/AVX2/AVX512 on x86, NEON
  on arm64, plus GENERIC), get_cpu_isa()/init_isa() with the CT2_FORCE_CPU_ISA
  override and the rule that AVX512 is never auto-selected, and cpu_features-
  based detection. It explains the build mechanic where ENABLE_CPU_DISPATCH
  copies kernels.cc into per-ISA object files with -mavx2/-mavx512f/-DUSE_NEON
  flags, each compiling one TARGET_ISA explicit-specialization so symbols
  don't collide, and the CPU_ISA_DISPATCH constexpr-binding switch. Also
  covers the Vec<T,ISA> SIMD abstraction (width 4/8/16) and the separate
  get_gemm_backend priority ladder (MKL/DNNL/Accelerate/OpenBLAS/Ruy). For
  Metal: Apple Silicon builds get NEON+GENERIC, and CPU-reference ops on
  Metal-resident data run these NEON kernels.
---

How one binary carries several ISA variants of the CPU inner loops (AVX/AVX2/AVX512/NEON) and picks one at runtime.

**Sources (all citations from real lines):**

- `src/cpu/cpu_isa.h` / `cpu_isa.cc` (the enum, selection, dispatch macro)
- `src/cpu/cpu_info.h` / `cpu_info.cc` (feature detection)
- `src/cpu/kernels.h` / `kernels.cc` (the per-ISA kernel templates)
- `src/cpu/vec.h`, `vec_avx.h`, `vec_avx512.h`, `vec_neon.h` (the `Vec<T, ISA>` abstraction)
- `CMakeLists.txt` (the multi-compile mechanics)
- `src/cpu/backend.cc` (GEMM backend selection — related but separate)

---

## 1. The `CpuIsa` enum and runtime selection

`cpu_isa.h:8-17` — the enum is **arch-gated at compile time**: x86 builds (`CT2_X86_BUILD`) know `AVX/AVX2/AVX512`, arm64 builds (`CT2_ARM64_BUILD`) know `NEON`; both have `GENERIC`. The arch define comes from CMake (`CMakeLists.txt:276-281`: arm64 → `-DCT2_ARM64_BUILD`, x86_64 → `-DCT2_X86_BUILD`).

`get_cpu_isa()` (`cpu_isa.cc:80-83`) caches `init_isa()` in a static. `init_isa` (`cpu_isa.cc:44-78`):

1. **`CT2_FORCE_CPU_ISA` env var** wins if set — accepted values `AVX512`/`AVX2`/`AVX` (x86), `NEON` (arm64), `GENERIC`; it throws if the CPU lacks the feature or the binary wasn't built with dispatch (`try_isa`, `cpu_isa.cc:11-24`).
2. Otherwise, auto-detect under `CT2_WITH_CPU_DISPATCH`: x86 picks AVX2 → AVX → GENERIC; **AVX512 is never auto-selected** ("can only be enabled with the environment variable", `cpu_isa.cc:66-70`). arm64 picks NEON.
3. Without dispatch compiled in: `GENERIC` (but see §3 — the single compiled variant is then the default case anyway).

Feature detection (`cpu_info.cc`): x86 via the vendored `cpu_features` submodule (`GetX86Info()`, line 12; `cpu_supports_avx512` requires f+cd+vl+dq+bw together, lines 34-40, matching the compile flags); arm64 is trivial — `cpu_supports_neon()` returns `true` (lines 54-56), NEON being baseline on AArch64.

The selection is logged at startup: `log_system_config()` prints "Selected ISA: …" (`src/utils.cc:46`).

## 2. The build mechanics: one source, N object files

`ENABLE_CPU_DISPATCH` (default ON, `CMakeLists.txt:20`) defines `CT2_WITH_CPU_DISPATCH` (`CMakeLists.txt:297`) and invokes the macro `ct2_compile_kernels_for_isa(isa flag)` (`CMakeLists.txt:265-274`), which **copies `src/cpu/kernels.cc` into the build dir as `kernels_${isa}.cc`** and sets per-file `COMPILE_FLAGS`:

- x86: `avx` → `-mavx`, `avx2` → `-mavx2 -mfma`, `avx512` → `-mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq` (MSVC: `/arch:*`) (`CMakeLists.txt:299-307`).
- arm64: a single extra copy, `neon` → `-DUSE_NEON` (`CMakeLists.txt:309`).

The original `src/cpu/kernels.cc` is _also_ in `SOURCES` (`CMakeLists.txt:123`) with no special flags — that translation unit becomes the GENERIC variant. So the library links 4 copies of kernels.cc on x86, 2 on arm64.

## 3. How one kernel template gets stamped per ISA

Inside `kernels.cc`, the compile flags select `TARGET_ISA` and the matching vec header (`kernels.cc:5-20`): `__AVX512F__` → `CpuIsa::AVX512` + `vec_avx512.h`; `__AVX2__`/`__AVX__` similarly; NEON when `(__ARM_NEON && !CT2_WITH_CPU_DISPATCH) || USE_NEON` — i.e. on arm64 _with_ dispatch, the unflagged copy deliberately compiles GENERIC and only the `-DUSE_NEON` copy compiles NEON. Every kernel in the file is then defined as an explicit specialization for `TARGET_ISA` only (e.g. `template<> void dequantize_gemm_output<TARGET_ISA>(...)`, `kernels.cc:724-725`), so the N object files don't collide — each contributes one ISA's set of symbols for the templates declared in `kernels.h` (`template <CpuIsa ISA> void exp/gelu/softmax/layer_norm/rms_norm/quantize_s8/...`, `kernels.h:10-121`).

The runtime selector is `CPU_ISA_DISPATCH` (`cpu_isa.h:42-83`) — the same constexpr-binding switch trick as `DEVICE_DISPATCH`: each `CPU_ISA_CASE` binds `constexpr cpu::CpuIsa ISA = ...` (`cpu_isa.h:27-32`). With `CT2_WITH_CPU_DISPATCH` it switches over all built ISAs; without it, it collapses to a single `default:` case whose ISA is whatever the whole library was compiled for (`cpu_isa.h:58-83`). Callers are the CPU primitives (`CPU_ISA_DISPATCH((cpu::gelu<ISA>(...)))`, `src/cpu/primitives.cc:307`) and CPU op kernels (`src/ops/softmax_cpu.cc:15`).

## 4. `Vec<T, ISA>` — the SIMD abstraction

`vec.h:13-18` defines the generic `Vec<T, ISA>` with `width = 1` (scalar fallback for any type/ISA pair without a specialization). Each vec header specializes `Vec<float, TARGET_ISA>`:

- `vec_neon.h:17-22` — `float32x4_t`, width 4 (partial loads go through an aligned stack temp, `vec_neon.h:32-40`).
- `vec_avx.h:47` — width 8; `vec_avx512.h:16` — width 16.

Kernels are written once against this interface: `vectorized_unary_transform` strides by `Vec<T, ISA>::width` and handles the tail with count-limited `load`/`store` (`kernels.cc:47-61`). Transcendentals come from the vendored `*_mathfun.h` headers (`neon_mathfun.h` etc., included by the vec headers), and hot loops opt into local fast-math via `CT2_FFAST_MATH_BEGIN/END` (`kernels.cc:30-39`).

## 5. CPU GEMM backends (separate axis from ISA)

GEMM never goes through kernels.cc — `cpu::get_gemm_backend(compute_type)` (`src/cpu/backend.cc:50-90`) picks an external library by compile-time availability **in priority order**: MKL (fp32/int16/int8, only if `mayiuse_mkl()`) → DNNL (fp32/int8) → Accelerate (fp32 only) → OpenBLAS (fp32 only) → Ruy (int8 + fp32) → NONE. `mayiuse_mkl()` requires a genuine-Intel CPU unless overridden by the `CT2_USE_MKL` env var (`backend.cc:10-31`). Consequences: on an Accelerate build, int8 GEMM needs Ruy (`WITH_RUY=ON`) or it has no backend (`has_gemm_backend`, `backend.cc:92-94` — this feeds compute-type resolution, see `compute-type-resolution.md`); MKL packed weights are gated by `CT2_PACKED_GEMM` (`backend.cc:101-106`); MKL/DNNL prefer the u8s8s32 form (`prefer_u8s8s32_gemm`, `backend.cc:96-99`). The chosen backends are logged at startup (`src/utils.cc:48-57`).

### Relevance to the Metal backend

- On Apple Silicon the build is `CT2_ARM64_BUILD`: the dispatch set is just NEON + GENERIC, and `get_cpu_isa()` lands on NEON. Every CPU-reference op that runs on Metal-resident data (via `METAL_DEVICE_CASE` binding `D = CPU`) executes these NEON kernels.
- `CT2_FORCE_CPU_ISA=GENERIC` is a useful bisection lever when a numerical discrepancy might be in the NEON path rather than a Metal kernel.
- GEMM on the CPU side of a Metal build comes from Accelerate (fp32) per §5; Metal's own GEMM (MPS / int8 MSL kernels) is targeted routing, not a `GemmBackend` — see `apple-silicon` skill.
- The Metal MSL kernels in `src/metal/kernels/` are an entirely separate kernel layer; nothing in kernels.cc is reused there.
