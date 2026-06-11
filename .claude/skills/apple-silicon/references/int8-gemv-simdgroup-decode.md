# int8 SIMD-group GEMV for decode (`ct2_gemv_s8`) — the small-m fast path

Source: `src/metal/kernels/kernels_msl.h` (kernel `ct2_gemv_s8`, lines ~831–869),
`src/metal/primitives.mm` (routing + dispatch inside `metal::gemm_s8`, ~599–636),
commit `68239d13`. Measurements: `METAL_BENCHMARKS.md` int8 tables, M4 Max (40-core
GPU), Release, 2026-06-11, 30 iters/cell.

## The decode regime: GEMM tiles starve at m = batch

Autoregressive decode runs **every Dense at m = batch** (one token row per sequence) —
see `dispatch-overlap-and-perf-model.md` for why decode is the hostile regime. At
m = 1 the 64×64 tiled kernel (`int8-gemm-kernel-design.md`) wastes **63/64 of its
A-tile**: a whole 16×16 threadgroup stages a 64-row tile of which one row exists.
The fix is a different work assignment, not a smaller tile.

## Routing — the exact condition (`primitives.mm` ~603)

`metal::gemm_s8` takes the GEMV path only when ALL of these hold; anything else falls
to the general tiled kernel (correct, just slower):

```objc
if (!transpose_a && transpose_b && m <= 8
    && k % 4 == 0 && lda % 4 == 0 && ldb % 4 == 0
    && a_buffer.offset % 4 == 0 && b_buffer.offset % 4 == 0) {
```

- **Dense layout only** (`!trans_a && trans_b`): both the A row and the B row are then
  k-contiguous, which is what makes the lane-strided load pattern work.
- **m ≤ 8**: covers decode at realistic batch sizes; beyond that the tiled kernel's
  amortization wins back.
- **4-byte alignment of k, both leading dims, and both `MTLBuffer` offsets**: the
  kernel reinterprets the operand pointers as `device const char4*`, so every operand
  row must start 4-aligned and k must be a multiple of 4.

Upstream of this, `src/ops/gemm.cc` (~113) has already guaranteed beta == 0, integral
alpha, and no shift compensation — the same exactness contract as the tiled kernel.

## Kernel design: one SIMD-group per output element

Lanes stride the k axis in `char4` steps; one `simd_sum` folds the int32 partials —
no threadgroup memory, no barriers (see `simd-group-functions.md` for the primitives):

```c
const uint i = tg.y;                      // output row (a token)
const uint j = tg.x * 8u + simd_group;    // output col; 8 SIMD-groups per threadgroup
if (j >= n)
  return;  // uniform per SIMD-group, so the simd_sum below stays in uniform control flow

device const char4* a4 = (device const char4*)(a + (ulong)i * lda);
device const char4* b4 = (device const char4*)(b + (ulong)j * ldb);
int acc = 0;
for (uint v = simd_lane; v < kvec; v += simd_size) {   // kvec = k / 4
  const int4 av = int4(a4[v]);
  const int4 bv = int4(b4[v]);
  acc += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
}
acc = simd_sum(acc);
if (simd_lane == 0u)
  c[(ulong)i * ldc + j] = alpha * acc;
```

Host dispatch (`primitives.mm` ~626): threadgroups of `threadExecutionWidth * 8`
threads = **8 SIMD-groups per threadgroup, each producing one output element**; grid
is `(⌈n/8⌉, m)` via `dispatchThreadgroups`. The kernel's `j = tg.x * 8 + simd_group`
mapping assumes exactly this shape — change one side, change both. Note the early
`return` on `j >= n` is **uniform across the SIMD-group** (j depends only on
`tg`/`simd_group`), so the later `simd_sum` never executes in divergent control flow.

## Why it's bandwidth-bound — and therefore why int8 wins here

At m = 1 the dominant traffic is the weight matrix B: every output element reads a
full k-length row exactly once, so the kernel moves ~n·k weight bytes per call —
**int8 moves half the bytes of fp16**. The arithmetic is trivial by comparison
(2·n·k integer ops over n·k bytes ≈ 2 ops/byte — far below the ALU:bandwidth ratio),
so time ≈ bytes / bandwidth and halving the bytes halves the time. Worked example,
the Qwen2.5 lm_head at m=1, n=151936, k=896: B is ~136 MB int8 vs ~272 MB fp16; at
0.49 ms the int8 kernel sustains ~280 GB/s — the right order for an M4 Max
(~546 GB/s peak), confirming the memory-bound read.

This is the inversion of the tiled-kernel story: large-m int8 loses to MPS fp16
because it is ALU-bound with no integer matrix units; small-m int8 **beats** every
float path because the regime is memory-bound and int8 is the smallest operand.
**This is where the quantization speed win actually lives** (the RSS win is
everywhere).

## Measured (M4 Max, 2026-06-11)

| m, n, k (kernel)      | Metal int8     | Metal fp16 (MPS) | Metal fp32 (MPS) |
| --------------------- | -------------- | ---------------- | ---------------- |
| 1, 4864, 896 (GEMV)   | **0.157 ms**   | 0.172 ms         | 0.176 ms         |
| 1, 151936, 896 (GEMV) | **0.49 ms** 🔥 | 0.84 ms          | 1.35 ms          |

1.7× over fp16 on the per-token lm_head GEMM. E2e decode on Qwen2.5-0.5B-int8 is
28.8–30.2 ms/token vs fp16's 25.2–25.7 — int8 still trails ~15% e2e because each
quantized Dense adds a quantize + dequant-epilogue launch on an API-floor-bound loop
(`dispatch-overlap-and-perf-model.md`); the GEMV itself is faster than the fp16 GEMM
it replaces.

Exactness coverage: `Int8GemmDeepAccumulatorMatchesHostReference`
(tests/metal_test.cc:472) runs **m = 3 specifically to route here** (m = 16 covers the
tiled kernel) at k = 2048 against a host int32 triple loop.

### Relevance to the CT2 Metal backend

- Wired: routed inside `metal::gemm_s8` (`src/metal/primitives.mm`) — callers and the
  `ops::Gemm` guard never know which kernel ran. Hit by every Dense during int8
  decode, including the giant lm_head projection each step.
- Regressions to watch: the 8-SIMD-groups-per-threadgroup host/kernel coupling; the
  alignment preconditions (an unaligned weight slice silently falls back to the slow
  tiled path — a perf cliff, not a correctness bug); keep the m=3 deep-k oracle green.
- Future work: batch sizes > 8 between the regimes (a 2-row-per-SIMD-group variant);
  any int8 attention/KV work would want this same lane-strided `char4` + `simd_sum`
  pattern.
