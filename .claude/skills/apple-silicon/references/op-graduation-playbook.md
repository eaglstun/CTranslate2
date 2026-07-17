---
title: "Op-graduation playbook (CT2 → Metal GPU kernel)"
summary: >-
  The CT2-specific procedure for moving an op from the Device::METAL-to-CPU-
  reference binding onto a real Metal GPU kernel: the route-don't-switch model
  (an if device==METAL branch calling a metal:: entry point before generic
  dispatch, never a real DEVICE_CASE), and the four steps — write the
  templated MSL kernel in kernels_msl.h, add fp32/fp16 metal:: entry points
  resolving StorageView pointers via buffer_and_offset, route at operator()
  level, verify parity. It details the two fp16 paths (real half kernel vs
  direct-instantiation bypass, the latter requiring metal::synchronize() first
  to avoid reading stale GPU data). It catalogs MSL landmines (no erf, lazy
  library compilation, use 1/sqrt not rsqrt, bind every buffer, host scalars
  by value) and states sampling/Conv1D aren't worth a kernel.
---

How to correctly move a CTranslate2 op from the CPU-reference binding onto a real Metal
GPU kernel — the repo procedure and its sharp edges. Unlike the other references here,
this is not an Apple-API topic: it is the CT2-specific "how to make this change" companion
to `compute-kernels-and-dispatch.md` (MSL/dispatch), `mps-matrix-multiplication.md` (GEMM),
and `storage-and-synchronization.md` (the flush model). Read those for the API; read this
for the pattern.

## Mental model: route, don't switch

`Device::METAL` is bound to the CPU implementation in `src/device_dispatch.h`
(`METAL_DEVICE_CASE` → `constexpr Device D = Device::CPU`). So by default every op already
"works" on Metal data via the CPU kernel over unified memory — correct but not GPU-fast.
You graduate ONE op at a time by adding a **targeted route**: an `if (device == METAL)`
branch that calls a `metal::` entry point and returns, placed BEFORE the generic dispatch.
You never add `Device::METAL` as a real `DEVICE_CASE` (that breaks the link — see
`METAL_BACKEND.md`).

## Steps to graduate an op

1. **Write the MSL kernel** in `src/metal/kernels/kernels_msl.h` (inline raw string). MSL
   supports C++ templates: write a templated `device` function body + concrete `[[kernel]]`
   wrappers per dtype (e.g. `ct2_softmax_float` / `ct2_softmax_half` both calling a shared
   `template <typename T>` impl). This is confirmed working.

2. **Add a `metal::` entry point** — declare in `src/metal/primitives.h`, implement in
   `src/metal/primitives.mm`. Provide fp32 and fp16 overloads. In the encoder, bind buffers
   at indices matching the MSL arg order (from 0). Resolve each StorageView pointer to its
   `(MTLBuffer, offset)` via `buffer_and_offset(ptr)` — this handles sub-views and
   strided-batch operands (the allocator side table is an address-ordered `std::map`).

3. **Route the op** at the `operator()` level in `src/ops/<op>.cc`:

   ```cpp
   #ifdef CT2_WITH_METAL
   #  include "metal/primitives.h"   // or metal/utils.h for synchronize() only
   #endif
   ...
   #ifdef CT2_WITH_METAL
     if (x.device() == Device::METAL && /* supported dtype/shape */) {
       metal::my_op(...);   // a.device() is the REAL device even though the binding makes D=CPU
       return;
     }
   #endif
     // ...generic dispatch follows (unchanged)
   ```

   Route at `operator()` level (not inside `compute`) so it sits before the dispatch that
   throws on fp16 in a non-CUDA build. Partial coverage is fine: route only the common case
   (e.g. LayerNorm routes axis==rank-1 && gamma && beta; general-axis falls through to CPU).

4. **Verify parity** (see below) and check the full suite is still at baseline.

## fp16: two paths

The CPU-reference binding CANNOT do fp16 (CPU has no half compute) — so an fp16 model needs
a real half kernel for every op it touches. Two ways to satisfy a given op:

- **Real half kernel** (preferred for hot/compute ops): add the `_half` MSL wrapper + fp16
  `metal::` overload, route fp16 alongside fp32. Compute in `float`, cast back, so half has
  the same rounding as the fp32 path.

- **Direct-instantiation bypass** (for cold, comparison/copy/RNG-based ops — sampling, etc.):
  call the already-instantiated fp16 CPU `compute` directly, bypassing the throwing dispatch.
  Two requirements:
  1. Instantiate the fp16 CPU path — add `DECLARE_IMPL(float16_t)` in the op's `_cpu.cc`
     (its kernel body must compile for `half`: comparisons, `std::discrete_distribution`
     from half, etc. all work).
  2. **Call `metal::synchronize()` FIRST.** This is the easy-to-miss correctness bug: the
     direct `compute<Device::CPU,...>` call SKIPS the `metal::flush()` that
     `METAL_DEVICE_CASE` performs before every normal CPU-reference op. Without the flush,
     the CPU reads GPU-produced data (e.g. softmax output, prior-op logits) over unified
     memory before the async command buffer has completed → stale/garbage. Pattern:

     ```cpp
     #ifdef CT2_WITH_METAL
       if (device == Device::METAL && dtype == DataType::FLOAT16) {
         metal::synchronize();
         compute<Device::CPU, float16_t>(input, output);
         return;
       }
     #endif
     ```

     (Confirmed: `topk.cc`, `topp_mask.cc`, `multinomial.cc` use exactly this. TopK
     originally omitted the `synchronize()` and survived only on a beam-search timing race —
     fixed 2026-06-09.)

Also flip `mayiuse_float16(Device::METAL)` → true in `src/types.cc` so FLOAT16 resolves on
Metal (AUTO stays CPU-like, so fp16 is explicit opt-in). Output StorageView must be
pre-typed FLOAT16 for the op to write half — caller's responsibility.

## MSL landmines (each cost real debugging)

- **No `erf` in MSL** — any language version, no `precise::erf` either. Exact GELU needs a
  hand-rolled Abramowitz-Stegun 7.1.26 approximation (`ct2_erf`, ~1.5e-7 max error).
- **Library compilation is LAZY** (`ensure_library()` on first `pipeline()` call), NOT eager
  in the MetalContext constructor. Keep it that way: a single kernel that fails to compile
  must not take down device setup / allocation / MPS GEMM (it did, once, when erf broke the
  eager compile and bricked the whole backend including the allocator).
- **Use `1.0f/sqrt(x)`, not `rsqrt(x)`**, in norms — `rsqrt` diverges from the CPU reference
  and breaks bit-for-bit parity.
- **Row-reduction kernels use a fixed 256-thread threadgroup** (tree reduction for max/sum).
  Keep threadgroup sizes a multiple of `threadExecutionWidth` (32 on Apple GPUs).
- **Every referenced buffer must be bound.** If an optional operand (`lengths`, a scalar) is
  absent, bind a dummy buffer at its index anyway, or the dispatch fails.
- **Scalar operands live on the HOST.** For e.g. `MulScalar`, the scalar `b` is CPU memory —
  `buffer_and_offset(b)` will fail. Read it on the host (unified) and pass it to the kernel
  BY VALUE as a uniform; bind the unused operand as the dummy buffer.

## Verification strategy

- **Parity comes free from the existing op suite.** `tests/ops_test.cc` / `layers_test.cc`
  are parameterized over `Device::METAL`, so once an op is routed, its existing fp32 tests
  run on the GPU and compare against the CPU reference automatically.
- **fp16 has no op-suite coverage** — add an explicit parity test in `tests/metal_test.cc`
  (`Float16<Op>MatchesFloat32`, tolerance ~2e-2), following the existing ones. For RNG ops
  (Multinomial) exact parity is meaningless — assert runs-without-throw + valid output
  instead. fp16 mask/sentinel values must be representable (a large-negative mask saturates
  to fp16 -inf; use 0 in a TopPMask parity test).
- **Full-suite baseline:** all pass except the pre-existing `Conv1DGroupNoBiasQuantized`
  CPU int8 failure (MKL-less build artifact, not Metal). Confirm the count went UP by your
  new tests and the only failure is that known one.

## What is NOT worth a GPU kernel

Sampling (TopK/TopPMask/Multinomial) runs once per token on a vocab-sized vector — not a
bottleneck, selection/sort/RNG-heavy. Leave it on the CPU reference; only close the fp16
dispatch gap via the bypass above. Conv1D and concat/split are also fine on the reference
for the tiny model (concat was graduated to GPU for KV-cache residency but did NOT move the
e2e needle). See the perf graveyard in `SKILL.md` / `METAL_BENCHMARKS.md` before optimizing.
