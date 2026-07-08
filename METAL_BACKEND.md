# Apple Metal Backend — Progress & Roadmap

Status as of 2026-07-07. This document tracks the in-progress Apple Metal GPU backend
(`Device::METAL`, built with `-DWITH_METAL=ON`) for CTranslate2.

## TL;DR

A new GPU backend for Apple Silicon. **A full encoder-decoder transformer runs
end-to-end on Metal in both float32 and float16, with output matching the CPU.** The
entire per-token forward pass — GEMM (MPS), softmax, RMSNorm, LayerNorm, rotary/RoPE,
gather, bias+activation, standalone activations, and elementwise mul/add — executes as
real GPU kernels, in both precisions, covering GPT-2-style and Llama/Mistral-style
(SwiGLU) architectures — as do concat/split, int8 quantize/dequantize/GEMM, and the
sampling ops (TopK/TopPMask/GumbelMax/Multinomial, seeded-reproducible). The remaining
ops (conv, general-axis norms, …) run correctly via a CPU-reference path over unified
memory, behind a full regression net (the existing op/layer/storage suites all run on
`Device::METAL`). Perf state: since the fused decode-attention kernel (M16, 2026-07-07),
**Metal beats the Accelerate CPU baseline in every measured regime** — prefill (fp16
2.6×, int8 at fp16 parity) and now autoregressive decode too (Qwen bs=1 edges CPU,
bs=8 ~2.8× over CPU).

## Why this is tractable

Two facts shape the whole design:

1. **Apple Silicon has unified memory.** A Metal buffer allocated with
   `MTLResourceStorageModeShared` exposes a CPU-addressable `contents` pointer. That
   pointer satisfies CTranslate2's existing pointer-based `Allocator`/`StorageView`
   contract unchanged, host↔device copies become plain `memcpy`, and CPU code can operate
   directly on Metal-resident data.

2. **Metal is not source-compatible with CUDA.** Unlike AMD ROCm (which rides the CUDA
   `.cu` files via HIP), Metal needs its own Objective-C++ (`.mm`) host code and Metal
   Shading Language (`.metal`/MSL) kernels, plus Metal Performance Shaders (MPS) for GEMM.
   So it is a genuinely new backend, not a re-skin.

## Build

Requires macOS on Apple Silicon (arm64). Initialize submodules first
(`git submodule update --init --recursive`).

```bash
mkdir build && cd build
cmake .. -DWITH_METAL=ON -DWITH_MKL=OFF -DWITH_ACCELERATE=ON \
         -DOPENMP_RUNTIME=NONE -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
./tests/ctranslate2_test ../tests/data --gtest_filter='METAL/*:MetalTest.*'
```

Non-obvious flags (defaults fail on Apple Silicon):

- `-DWITH_MKL=OFF -DWITH_ACCELERATE=ON` — Intel MKL is x86-only; Accelerate is the
  Apple-Silicon CPU BLAS (and the CPU-reference backend for Metal).
- `-DOPENMP_RUNTIME=NONE` — the default `INTEL` runtime expects libiomp5, which is absent.

`WITH_METAL` is mutually exclusive with `WITH_CUDA`/`WITH_HIP` and requires arm64.

## Design

### Memory: a shared-buffer allocator with an offset-aware side table

`src/metal/allocator.mm` specializes `get_allocator<Device::METAL>()`. Each allocation is
an `MTLBuffer` (shared storage); the allocator returns `[buffer contents]` and records the
allocation in an address-ordered `std::map<uintptr_t, {buffer, size}>`. `buffer_and_offset(ptr)`
does a range lookup so that pointers offset into an allocation (StorageView sub-views,
strided-batch GEMM matrices) resolve to the owning buffer plus a byte offset. Manual
retain/release (not ARC).

### Dispatch: a CPU-reference binding, graduated by targeted routing

`src/device_dispatch.h` binds the `Device::METAL` dispatch case to
`constexpr Device D = Device::CPU`. Because Metal memory is CPU-addressable, the existing
CPU kernels run correctly on Metal-resident data — so the whole engine works on
`Device::METAL` immediately, with **no** per-op `compute<Device::METAL,T>` instantiations
and **no** `primitives<Device::METAL>` specialization.

Individual hot paths are then "graduated" to real GPU kernels by **targeted routing**: an
op checks the real device (`a.device() == Device::METAL`, which is correct even though the
dispatch bound `D=CPU`) and calls a `metal::` entry point before falling through to the
CPU path. Example (`src/ops/matmul.cc`):

```cpp
#ifdef CT2_WITH_METAL
if constexpr (std::is_same<T, float>::value) {
  if (a.device() == Device::METAL) { metal::gemm(...); return; }
}
#endif
```

**Important invariant:** allocation and device-index resolution must NOT follow the
CPU binding — `get_allocator(Device)` (`src/allocator.cc`) and `get_device_index`/
`set_device_index` (`src/devices.cc`) early-return for `Device::METAL` to the real Metal
allocator/index. If they fell through to the CPU-bound dispatch, Metal StorageViews would
get plain CPU memory and GPU kernels could not find their buffers.

**fp16 gotcha for un-graduated ops.** An op with no `metal::` kernel falls through to the
CPU-reference path — but that reference is **float32-only**, and `DEVICE_AND_FLOAT_DISPATCH`
(`src/dispatch.h`) has no fp16 case off CUDA, so an fp16 input on Metal throws
`"<Op> only supports float types"`. Until such an op is graduated to a GPU kernel, give it a
Metal fp16 branch that **upcasts to fp32, runs the CPU-reference path over unified memory, and
downcasts back** (see `Conv1D` in `src/ops/conv1d.cc` and `ApplyTimestampRules` in
`src/models/whisper.cc` — both unblocked fp16 Whisper this way). Whisper/Wav2Vec2 conv-stem
models hit this first because the encoder leads with `Conv1D`.

### Kernels & device context

`src/metal/device.mm` holds a process-wide singleton: `MTLDevice`, one
`MTLCommandQueue`, and an `MTLLibrary` compiled at runtime via `newLibraryWithSource`
from MSL embedded as a C++ raw string in `src/metal/kernels/kernels_msl.h`. (Runtime
compilation avoids `.metallib` path-resolution issues with the bare shared library; an
offline `.metallib` is a future optimization.) Compute pipeline states are cached per
kernel name. Each `metal::` op commits its own command buffer asynchronously (no per-op
wait); `metal::flush()` waits on the last-committed buffer before any CPU access to Metal
memory. Reusing one command buffer across many ops (committing once per step) was tried and
reverted — it removed CPU/GPU overlap and regressed GEMM-heavy regimes; see
`METAL_BENCHMARKS.md`.

**Autorelease pool (load-bearing — a long run will be OOM-killed without it).** Command
buffers (`[queue commandBuffer]`), compute encoders, and `MPSMatrixDescriptor`s are
autoreleased (+0). CTranslate2 drives ops from C++ worker threads that have **no run loop**,
so without an explicit pool these objects never drain and accumulate as **wired** memory for
the entire run — process heap RSS stays flat while the GPU/wired footprint balloons until the
OS SIGKILLs the process (this is what made a 730s Whisper transcribe die at ~155s). Each op is
a flat `new_command_buffer() → encode → commit_command_buffer()` pair, so `new_command_buffer()`
pushes a thread-local `NSAutoreleasePool` and `commit_command_buffer()` drains it, bracketing
exactly that op's autoreleased temporaries (the committed buffer survives the drain because
`g_last_committed` retains it). **Any new code path that creates Metal objects outside a
new→commit pair must wrap itself in `@autoreleasepool`.**

## File map

| File                                 | Role                                                                                        |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| `src/metal/utils.h`                  | Pure-C++ surface (`has_gpu`, `get_gpu_count`, `synchronize`) — safe to include from `.cc`   |
| `src/metal/device.h`                 | Objective-C++ internals (device/queue/pipeline accessors, `buffer_and_offset`) — `.mm` only |
| `src/metal/device.mm`                | Device/queue/library singleton, pipeline cache, lifecycle                                   |
| `src/metal/allocator.mm`             | Shared-buffer allocator + address-ordered side table                                        |
| `src/metal/primitives.h`             | Pure-C++ declarations of `metal::` compute entry points                                     |
| `src/metal/primitives.mm`            | `metal::add`, `metal::softmax` (kernel dispatch)                                            |
| `src/metal/gemm.mm`                  | `metal::gemm` / `gemm_batch_strided` via MPSMatrixMultiplication                            |
| `src/metal/kernels/kernels_msl.h`    | MSL kernel source (add, softmax) as a raw string                                            |
| `src/device_dispatch.h`              | `METAL_DEVICE_CASE` (CPU-reference binding)                                                 |
| `src/allocator.cc`, `src/devices.cc` | METAL early-returns for allocator/device-index/synchronize                                  |
| `tests/metal_test.cc`                | Metal-specific tests incl. end-to-end translation                                           |

CMake wiring lives in the `if(WITH_METAL)` block in `CMakeLists.txt` (Objective-C++ via
`enable_language(OBJCXX)`, links `Metal`/`MetalPerformanceShaders`/`Foundation`).

## Milestones

### ✅ M1 — Tracer bullet

Enum + string/lifecycle wiring, shared-buffer allocator, device/queue/library singleton,
one hand-written `add` kernel. Proves a `StorageView` lives on the GPU and survives a
kernel round-trip. Verified by `tests/metal_test.cc`.

### ✅ M2 — Dispatch on-ramp (CPU reference)

The `METAL→CPU` dispatch binding. The entire existing op / layer / storage-view test
suite is parameterized over `Device::METAL` and passes (executing the CPU reference on
unified memory). This is the regression harness that guards every later GPU kernel.

### ✅ M3 — GEMM on the GPU (MPS)

`metal::gemm` + `gemm_batch_strided` (float32) via `MPSMatrixMultiplication`, row-major
(no cuBLAS-style operand swap). Routed from `MatMul` and `Gemm` ops. Required the
offset-aware allocator for strided-batch and sub-view operands. All layer tests — whose
attention and FFN matmuls now run on MPS — pass.

### ✅ M4 — End-to-end model + first reduction kernel

- **End-to-end:** the default transliteration transformer translates on `Device::METAL`
  with output identical to CPU (`MetalTest.EndToEndTranslation`). Full encoder-decoder,
  attention, FFN, layer norm, and beam search on the GPU-backed device.
- **GPU softmax:** `ct2_softmax_float` (threadgroup tree reduction), matching CPU
  semantics including `lengths` masking and log-softmax. Routed from the `SoftMax` op.

### ✅ M5 — fp16 foundation

fp16 for the ops already on the GPU. `mayiuse_float16(Device::METAL)` → true so the
`FLOAT16` compute type resolves on Metal (`AUTO` left CPU-like, so fp16 is explicit
opt-in). float16 overloads of `metal::gemm`/`gemm_batch_strided` (`MPSDataTypeFloat16`)
and a `ct2_softmax_half` kernel. Op routing moved to the `operator()` level (the generic
dispatch throws on fp16 in a non-CUDA build), covering fp32 + fp16 in one path. Verified
by `Float16GemmMatchesFloat32` / `Float16SoftMaxMatchesFloat32` (parity vs fp32, tol
2e-2). At this milestone a full fp16 model was not yet possible (more ops needed half
kernels); M6–M9 closed that, and M10 verified it.

### ✅ M6–M9 — the rest of the forward pass

Real GPU kernels (float + half) for **RMSNorm**, **LayerNorm** (last-axis affine),
**Rotary/RoPE**, **Gather** (type-agnostic byte copy), **fused BiasAdd + activation**,
**standalone activations** (ReLU/GELU×3/Swish/Sigmoid/Tanh), and **elementwise Mul** —
each routed via the targeted-routing pattern and parity-checked against the CPU reference
(the existing op-suite tests flip to GPU execution). GELU uses a `ct2_erf` approximation
since Metal has no `erf`. The kernel library compiles lazily so a kernel bug can't break
allocation/MPS.

> **War story / gotcha:** the GELU-**tanh** variant once emitted `NaN` on large activations
> (Gemma2 deep layers) because Metal's `tanh(x)` overflows to `NaN` for big `x`, where CPU
> `std::tanh` saturates to ±1. Fixed by clamping the `tanh` argument (`ct2_tanh_safe` in
> `kernels_msl.h`). Full debugging writeup — including the CPU-reference bisection method and
> the "mid-pipeline CPU reads of MPS output are unreliable" caveat — in
> [`METAL_GEMMA2_NAN_POSTMORTEM.md`](METAL_GEMMA2_NAN_POSTMORTEM.md).

### ✅ M10 — fp16 inference end-to-end

`MetalTest.EndToEndTranslationFloat16` loads the model with `ComputeType::FLOAT16` and
translates entirely in fp16 on Metal, output matching CPU fp32. Running it surfaced one
fp16 gap — `TopK` (the non-CUDA float dispatch rejects fp16) — fixed by calling the
already-instantiated `compute<Device::CPU, float16_t, int32_t>` directly (TopK is
comparison-based and runs on the CPU reference; it is not a hot op).

### ✅ M11 — int8 Phase 1: quantization plumbing + GEMM shim (2026-06-11)

The int8 path runs end-to-end on Metal (per `INT8_METAL_PLAN.md` Phase 1).
`mayiuse_int8(METAL)` is true and `get_supported_compute_types("metal")` reports
`int8` / `int8_float32` / `int8_float16`. New native kernels: `ct2_quantize_s8_*`
(per-row amax tree reduce; `precise::divide` for the scale, `rint` for round-to-even —
bit-exact against the CPU reference in the op suite), `ct2_dequantize_s8_*`, and
`ct2_dequant_gemm_out_*` (int32 → `/(a_scale·b_scale)` + bias + every ActivationType),
all fp32+fp16. The **int8 GEMM is a Phase-1 shim**: int8 operands widen to fp32 and ride
the cached MPS float GEMM, product cast back to int32 (integer-exact below 2^24 —
verified at k=2048 against a host int32 reference; fp16 was NOT usable here, real
accumulators overflow it). `ComputeType::AUTO` is pinned to FLOAT32 on Metal so auto
users don't land on the shim. **No resident-memory win yet** — weights widen per call;
that is Phase 2 (native int8×int8→int32 MSL GEMM, under these same tests).

Measured on Qwen2.5-0.5B-int8 (M4 Max, 2026-06-11, single run): coherent greedy output,
**92/100 teacher-forced next-token agreement vs the fp16 reference** (5 prompts × 20
steps), 3-prompt × 24-token generation 6.4s int8 vs 2.6s fp16 — the shim's extra cast
passes make Phase-1 int8 ~2.5× slower than fp16, as expected.

This milestone also fixed a **pre-existing command-buffer race**: `commit_command_buffer`
committed outside `g_commit_mutex`, so two worker threads (grouped Conv1D's
`parallel_for`) could commit in one order and record `g_last_committed` in the other,
leaving `flush()` waiting on a buffer that wasn't last in the queue. Surfaced as an
intermittent `METAL/...Conv1DGroup` failure once the new kernels shifted library-compile
timing; the commit now happens inside the lock (10/10 clean full-suite runs).

### Verification snapshot

- Full METAL suite: **84 passed, 2 skipped, 0 failed** (skips = Conv1D dilation and
  grouped-quantized, which the CPU reference does not implement).
- Full suite (all devices): **271/275** — the lone failure is a pre-existing
  `CPU/...Conv1DGroupNoBiasQuantized` "No INT8 GEMM backend for CPU" artifact of the
  MKL-less build, unrelated to Metal.
- CPU-only build: no new warnings introduced.

### ✅ M12 — int8 Phase 2: native int8×int8→int32 GEMM, weights int8-resident (2026-06-11)

The Phase-1 fp32 GEMM shim is **replaced by native MSL kernels** (`ct2_gemm_s8` /
`ct2_gemv_s8` behind one `metal::gemm_s8` entry point), so quantized weights stay
**int8-resident on the GPU** — no per-call widening — and accumulation is **bit-exact
int32 at any depth** (the suite now includes an all-saturated k=2048 case whose
accumulator, 3.3e7, exceeds fp32's 2^24 integer-exact range: impossible for the shim).
MPS has no integer GEMM and `simdgroup_matrix` has no int8 element type (MSL spec 2.4 —
half/bfloat/float only), so hand-tiled is the only native path:

- **m ≤ 8 (decode), Dense layout:** SIMD-group GEMV — one SIMD-group per output element,
  lane-strided `char4` k-loop, `simd_sum` fold. Memory-bound regime, int8 moves half the
  bytes: **beats MPS fp16** (per-token lm_head GEMM 0.49 vs 0.84 ms — 1.7×).
- **m > 8 (prefill):** threadgroup-tiled 64×64 C-tile, 32-deep k chunks, 4×4 int4
  micro-tile. ALU-bound at ~2.4 T-MAC/s (~3.7 ceiling: int MAC = 2 ALU ops, no integer
  matrix units) — structurally ~3–5× slower than MPS fp16 at the kernel level.
  **Superseded on macOS 26+ by M14's MPP path** (ties fp16); the tiled kernel remains
  the fallback (older OSes, non-NT layouts, integral `alpha != 1`).

Routing (`src/ops/gemm.cc` INT8 branch) additionally guards an **integral alpha** —
a float alpha cannot be applied exactly to an int32 accumulator. The shim casts
(`s8_to_float` / `float_to_s32` and their kernels) are removed.

Measured on Qwen2.5-0.5B-int8 (M4 Max, 2026-06-11, 3 runs each, warm; full tables in
`METAL_BENCHMARKS.md`): **92/100 teacher-forced agreement** vs fp16 (5 prompts × 20
steps — identical to Phase 1, as a bit-exact GEMM swap should be), e2e 3-prompt × 24-token
**1.22–1.27 s vs fp16 0.99 s** (the shim's ~2.5× penalty is now ~1.26×), decode bs1
28.8–30.2 vs 25.2–25.7 ms/token, and **peak RSS 1453 vs 2494 MB (−42%)** — the
resident-memory win that was the point of the phase. New tests: saturated-accumulator
exactness, all-four-transpose-layouts vs a host triple loop, deep-k oracle now covering
both kernels (m=3 GEMV / m=16 tiled), `DISABLED_BenchmarkGemmInt8`.

### ✅ M13 — downstream validation harness, int8 green in all four consumers (2026-06-11)

`scripts/validate-downstream.sh` + `tests/downstream/` (config, drivers, goldens,
results): builds the lib from this worktree → `cmake --install` to a pinned prefix →
rebuilds the wheel (`CTRANSLATE2_ROOT`) → force-reinstalls into each consumer venv →
runs each consumer's canonical job → diffs against an **fp16-on-Metal golden** from the
same build, with quant-error tolerances. This is the loose end-to-end oracle from
`INT8_METAL_PLAN.md`; the C++ op suite remains the bit-tight one.

Measured 2026-06-11 (M4 Max, single full run; details in
`tests/downstream/results/RESULTS.md`) — **4/4 pass**:

| consumer                 | metric                   | int8 value | tolerance |
| ------------------------ | ------------------------ | ---------- | --------- |
| whisperX (small, 30s)    | WER vs fp16              | 0.000      | ≤ 0.10    |
| faster-whisper (small)   | WER vs fp16              | 0.071      | ≤ 0.10    |
| Qwen2.5-0.5B-int8        | teacher-forced agreement | 0.900      | ≥ 0.90    |
| NLLB-600M-int8 (eng→fra) | char similarity vs fp16  | 1.000      | ≥ 0.90    |

The harness immediately caught a real bug: with int8 now advertised on Metal, model
loading quantized **conv** weights too, and Whisper crashed (`Conversion from int8 to
float32 is not yet implemented`) — Metal has no quantized convolution (Conv1D runs via
the CPU reference). Fix: `src/models/model.cc` keeps conv weights in `float_dtype` on
`Device::METAL`, the same guard CUDA/DNNL already use. Post-fix suite:
`*METAL*` 73 passed / 2 skipped / 0 failed, `*Metal*` 22/22.

### ✅ M14 — int8 prefill at fp16 speed: Metal-4 MPP `matmul2d` (2026-06-11)

Closes the one honest weakness M12 left: large-m int8 prefill. The m>8 / Dense-layout /
`alpha == 1` case of `metal::gemm_s8` now routes to a Metal Performance Primitives
`mpp::tensor_ops::matmul2d` kernel (`ct2_mpp_gemm_s8_nt` in
`src/metal/kernels/kernels_mpp_msl.h`) — int8_t×int8_t→int32_t is a base-Metal-4
supported combination (the SDK header's type table; the macOS-26 successor to the
"MPS is float-only / `simdgroup_matrix` has no int8" wall the hand-tiled kernel was
built against). Measured **int32-bit-exact** vs the host triple loop (k=2048, full int8
range, edge shapes), so this is a pure speed swap — the deep-accumulator oracle holds
unchanged, and Qwen int8 output tokens are byte-identical with the path on or off.

Numbers (M4 Max, macOS 26.4.1, 2026-06-11, best-of-3; full tables and tuning notes in
`METAL_BENCHMARKS.md`): 2048³ **7.19 → 1.51 ms** (ties MPS fp16's 1.49); Qwen Dense
prefill shapes 1.14 → 0.33 and 1.70 → 0.37 ms (fp16: 0.34 / 0.37); Qwen2.5-0.5B e2e
prefill (batch 8 × 128) **555 → 350 ms** (fp16 300). Decode GEMV is untouched and still
beats fp16.

Engineering shape (the part a future reader needs):

- **Separate MSL 4.0 library.** MPP needs `languageVersion 4.0` (macOS 26+), so
  `kernels_mpp_msl.h` compiles as a second `newLibraryWithSource` library in `device.mm`
  behind `@available(macOS 26, *)`; any compile/pipeline failure is cached and
  `get_pipeline_mpp` returns nil — callers fall back to the classic kernels, so older
  OSes and GPUs are unaffected. The main library stays at the default language version.
- **In-shader `tensor_inline` views** wrap the existing raw buffer args (pointer +
  extents + strides, dim 0 fastest) — no host-side MTLTensor objects, no new binding
  model, same `setBuffer` encoding as every other kernel.
- **Tuning that mattered:** 16×64 tile on **2** cooperating SIMD-groups (Apple's 4-SG
  header example is 2–5× slower on every shape measured) and the interior/edge
  `slice<Extents...>` static-extent split. Element types must be exactly
  `int8_t`/`int32_t` (non-const) or MPP's dispatch static_asserts.
- `CT2_NO_MPP_GEMM=1` forces the tiled fallback (bisection lever, like `CT2_NO_MPS_ACT`).
- Test coverage: the deep-k oracle now pins all three routes — m=3 GEMV, m=16 MPP,
  m=16/alpha=2 tiled (alpha≠1 is MPP-ineligible, keeping fallback coverage).

### ✅ M15 — sampling ops on Metal: TopK / TopPMask / GumbelMax / Multinomial, seeded-reproducible (2026-07-07)

Sampled LLM decode no longer round-trips to the CPU: the four sampling ops route to
Metal kernels at `operator()` (house pattern, no `DEVICE_CASE`), with
`CT2_NO_METAL_SAMPLING=1` forcing the CPU reference as the bisection lever.

- **TopK** (fp32/fp16) — row-wise, deterministic order; **bit-parity** with the CPU
  kernel incl. large-vocab shapes (over `topk_max_k()` falls back to CPU).
- **TopPMask** (fp32/fp16) — mirrors the CPU kernel's sequential accumulation order
  exactly; bit-parity (depth-capped by `topp_mask_max_depth()`).
- **GumbelMax** (fp32/fp16) — counter-based RNG noise kernel, per-launch seed drawn from
  the CT2 host generator; distributional parity (`GumbelNoiseStatistics`). Also fixed a
  real hole: GumbelMax previously threw on Metal fp16.
- **Multinomial** (fp32/fp16, `sample_size == 1`) — inverse-CDF kernel; the uniforms are
  drawn on the host from the CT2 generator and ride in the command buffer as inline
  bytes (batch capped by `multinomial_max_batch_size()`).

Closing the milestone required a fix in **shared core**, not Metal: `set_random_seed`
(`src/random.cc`) only stored the seed in an atomic, while `get_random_generator()`'s
`thread_local` mt19937 was seeded once, lazily, at first use — so re-seeding a live
process (or seeding from the main thread while sampling runs on worker threads) never
took effect, on any backend. Now `set_random_seed` bumps a seed **epoch** and each
thread's generator reseeds on its next use when it sees a new epoch. Upstream intent
check: `set_random_seed` was added (#648) explicitly for reproducible sampling — its own
test asserts same-seed → same output — so the seed-once behavior was a lazy-init
accident, not a contract. (CUDA's device-side curand states are still seeded once at
first use per thread; untouched here.)

Verification (M4 Max, macOS 26.4.1, 2026-07-07): `MetalTest` 28/28 incl. the previously
disabled `MultinomialSeededReproducible` (passes on both the Metal kernel and, with
`CT2_NO_METAL_SAMPLING=1`, the CPU-reference path — proving the fix is in the RNG core);
`*Metal*:*METAL*` filter 101 passed / 2 skipped (known CPU-reference Conv1D) / 0 failed;
full suite 288/292 with the only failure the pre-existing
`CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized` (present in the 2026-06-11 baseline).

### ✅ M16 — fused decode attention: Metal wins decode (2026-07-07)

Closes the last regime where the CPU won. The decode profile put 26.3% of a step in
`dot_product_attention` (MatMul q·K^T → SoftMax → MatMul ·V) — understated, because
batched MatMul (`gemm.mm`) encodes **one MPS GEMM per batch index**: decode attention at
bs=8 issued ~5,400 MPS encodes per step. The `ct2_sdpa_*` kernel (`kernels_msl.h`)
collapses each layer's three ops into **one launch**: one threadgroup per score row,
4 SIMD-groups striding the key axis with online-softmax partials merged in threadgroup
memory, float accumulation in both precisions, and no `[rows, T]` score tensor
materialized. Routed at the top of `dot_product_attention` (`attention.cc`) for the
decode regime (`q_len ≤ 8`, `d_head ≤ 256`; guards exclude relative-position/bias terms,
ALiBi, and attention-weights output). The lengths mask follows the SoftMax per-row
contract, so causal short-prefill masks and cross-attention memory masks (Whisper
decode) are honored. `CT2_NO_METAL_SDPA=1` forces the unfused reference.

Qwen2.5-0.5B decode (M4 Max, 2026-07-07, two runs, same binary A/B via the env var):
bs=1 fp32 33.9 → **75.9–79.0 tok/s** (CPU 72.6–75.8), bs=1 fp16 35.8 → **74.6–82.2**;
bs=8 fp32 62.0 → **403–418** (CPU 175–177), bs=8 fp16 62.7 → **486–493 (~7.9×)**. The
bs=8 blowout is the per-matrix encode loop dying: fused cost is one launch per layer
regardless of batch × heads. Full tables in `METAL_BENCHMARKS.md`. Verification: direct
kernel-vs-reference parity tests (`SdpaFusedParityWithReference`,
`SdpaFusedMaskedAndBeamParity` — d_head 64/80/256, T=3 and T=1500, beam-shaped q,
lengths incl. a fully masked row), `DecodeParityLLM` greedy-token equality on real Qwen,
`*Metal*:*METAL*` 103 passed / 2 skipped with the path on and off, full suite 288/292
(only the pre-existing CPU Conv1D baseline failure).

## What runs where today

| Operation                                                                  | Metal execution                                                                                                                             |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| GEMM / MatMul (float32 and float16)                                        | **GPU** — MPSMatrixMultiplication                                                                                                           |
| SoftMax / LogSoftMax (float32 and float16)                                 | **GPU** — custom kernel                                                                                                                     |
| RMSNorm (float32 and float16)                                              | **GPU** — custom kernel                                                                                                                     |
| LayerNorm, last axis + affine (float32 and float16)                        | **GPU** — custom kernel                                                                                                                     |
| Rotary / RoPE (float32 and float16)                                        | **GPU** — custom kernel                                                                                                                     |
| Gather (all dtypes)                                                        | **GPU** — custom kernel                                                                                                                     |
| BiasAdd + activation, last axis (float32 and float16)                      | **GPU** — fused custom kernel (ReLU/GELU/GELUTanh/GELUSigmoid/Swish/Tanh/Sigmoid)                                                           |
| Standalone activations: ReLU/GELU/Swish/Sigmoid/Tanh (float32 and float16) | **GPU** — custom kernel                                                                                                                     |
| Elementwise Mul (float32 and float16)                                      | **GPU** — custom kernel                                                                                                                     |
| Elementwise add (float32 and float16)                                      | **GPU** — custom kernel (the residual connections; fp16 path added after profiling a real LLM)                                              |
| Concat / Split / Slide (all dtypes)                                        | **GPU** — strided-copy kernel                                                                                                               |
| Quantize int8 (fp32/fp16 in; signed path)                                  | **GPU** — custom kernel (u8-shift variant falls through to CPU reference)                                                                   |
| Dequantize int8, simple + GEMM-output forms (fp32/fp16 out)                | **GPU** — custom kernels (GEMM-output form: the Dense `!trans_a && trans_b` layout, all activations)                                        |
| GEMM int8×int8→int32                                                       | **GPU (native)** — exact int32 accumulation; SIMD-group GEMV at m ≤ 8, Metal-4 MPP `matmul2d` at m > 8 (macOS 26+), hand-tiled MSL fallback |
| TopK / TopPMask (float32 and float16)                                      | **GPU** — custom kernels, bit-parity with CPU (size caps fall back to CPU reference)                                                        |
| GumbelMax / Multinomial sampling (float32 and float16)                     | **GPU** — host-seeded kernels (`set_random_seed`-reproducible); Multinomial at sample_size 1                                                |
| Decode attention: q·K^T → softmax → ·V (float32 and float16)               | **GPU** — fused single-launch SDPA kernel at q_len ≤ 8 (greedy/beam decode, short prefill); larger q_len uses MPS GEMM + softmax kernel     |
| Everything else (general-axis LayerNorm/BiasAdd, conv, int Mul, …)         | CPU reference over unified memory (correct, float32 only)                                                                                   |
| fp16 for ungraduated ops                                                   | Not yet — CPU reference is float32-only, so a full fp16 model needs those ops graduated to half kernels first                               |
| bf16 compute                                                               | Not yet                                                                                                                                     |

## What's left

### Near term — graduate more ops to GPU kernels

Each follows the established pattern: write an MSL kernel, add a `metal::` entry point,
add `if (device == Device::METAL)` routing in the op, verify parity against the CPU
reference via the existing suite.

- Remaining elementwise variants (sub/min/max/scalar-add) used in decoding
- Next decode-fusion candidates (post-M16 profile order): the projection GEMM epilogues,
  KV-cache `Concat` append, RMSNorm, Rotary

### fp16 — foundation done, full-model fp16 remaining

Where Apple Silicon actually gets fast. The foundation shipped in M5: `mayiuse_float16`
is true for Metal, and GEMM + softmax run in fp16. **A full fp16 model is still blocked**
because the CPU-reference binding is float32-only — every op a model touches needs a half
kernel. So full fp16 ≈ the "graduate more ops" list above, done with `half` kernels.
Remaining fp16 work:

- fp16 `half` kernels for the rest of the decoder path (norms, gather, rotary, bias/act).
- Optionally a `Device::METAL`-aware `DEVICE_AND_FLOAT_DISPATCH` in `src/dispatch.h` (it
  hardcodes `Device::CUDA` for fp16/bf16) if any fp16 op is routed through the generic
  dispatch rather than at `operator()` level.
- `get_preferred_size_multiple` `Device::METAL` branch in `src/types.cc` (padding hint;
  currently returns 1).
- Consider enabling fp16 in the `AUTO` compute-type path once the full path supports it.

### Performance work (after correctness)

- ~~Batch op encoding into fewer command buffers / reduce per-op synchronize.~~ — tried and
  reverted: per-thread command-buffer reuse (one commit per step) measured neutral-to-negative
  on a real LLM because it destroys CPU/GPU overlap; per-op async commit is already
  near-optimal. See `METAL_BENCHMARKS.md`.
- Offline `.metallib` compilation (faster startup than `newLibraryWithSource`), located
  at runtime via `dladdr()` with a source-compile fallback.
- Avoid the first-GEMM MPS pipeline warmup cost on the hot path.

### Deferred / out of scope for now

- Flash-attention (Metal equivalent of the CUDA flash-attn path)
- `Conv1D` (CUDA path uses cuDNN; needed for Whisper / Wav2Vec2 encoders)
- AWQ int4 GEMM
- bf16 (only on newer Apple GPUs)
- NCCL / tensor parallelism (multi-GPU)

## Gotchas for contributors

- **Don't add `Device::METAL` as a real `DEVICE_CASE`** in `device_dispatch.h` — it would
  force `primitives<Device::METAL>`/`compute<Device::METAL,T>` to be instantiated at ~50
  sites and break the link. Graduate ops via targeted routing instead.
- **Allocator/device-index early-returns are load-bearing** (see the invariant above).
- **All referenced Metal buffers must be bound** before dispatch — e.g. `metal::softmax`
  binds the input buffer as a dummy at the `lengths` index when there is no mask.
- **MPS is row-major** like `StorageView`; do not copy the cuBLAS column-major a/b swap.
- The `.mm` files use **manual retain/release (not ARC)**.
- **Metal has no `erf`** (nor `precise::erf`, any language version) — exact GELU uses the
  `ct2_erf` Abramowitz-Stegun approximation in `kernels_msl.h`. Check before assuming a
  libm function exists in MSL.
- **The kernel library compiles lazily** (`ensure_library()` on first pipeline use), so a
  kernel that fails to compile only breaks ops that use it — not allocation or MPS GEMM.
  A bad kernel still surfaces as a clear runtime error from the first op that needs it.
