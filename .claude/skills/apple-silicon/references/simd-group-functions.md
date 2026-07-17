---
title: "SIMD-group (and quad-group) functions"
summary: >-
  SIMD-group and quad-group functions (<metal_simdgroup>, MSL spec
  §6.10.2-6.10.3) that let 32-lane lockstep threads share data without
  threadgroup memory or barriers: reductions
  (simd_sum/product/min/max/and/or/xor and prefix scans), shuffle/broadcast
  permutes (simd_shuffle_down, simd_shuffle_xor butterfly), and vote/ballot
  (simd_all/any/ballot/is_first), plus the position attributes
  threads_per_simdgroup, thread_index_in_simdgroup, and
  simdgroup_index_in_threadgroup, and the canonical two-level threadgroup
  reduction. The load-bearing CT2 fact: rewriting the row-reduction kernels
  ct2_softmax, ct2_rms_norm, ct2_layer_norm from 256-thread tree reductions to
  a simd_sum-based version was tried, measured, and REVERTED (2026-06-09)
  because these kernels are memory-bound, so the SIMD version only added
  barriers and a cross-group bounce; don't re-chase it without fusing the
  reduction into an adjacent op.
semantic_id: "yhwU5onqom3Tw-39S30pHYVgo2-TyqulCk-mOZSn5N4uHemHX6erXw85eF17iwUahCEs291zjR5-9a6TxjT_pA"
---

Source (Apple): Metal Shading Language Specification, §6.10.2–6.10.3 (v4.1, 2026-06-04).
PDF: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
(The MSL standard-library functions are only in the spec PDF — there is no DocC HTML/JSON
page for them, so this was extracted from the PDF, not the usual DocC-JSON endpoint.)

SIMD-group functions let threads in one SIMD-group share data **without threadgroup memory
or barriers**. Header `<metal_simdgroup>`. Available on all current Apple Silicon (macOS
Metal 2 / 2.1+, iOS Metal 2.2 / 2.3+, visionOS always). A SIMD-group is the set of lanes the
GPU runs in lockstep; its width is `[[threads_per_simdgroup]]` (32 on Apple GPUs — the same
value as `MTLComputePipelineState.threadExecutionWidth`).

`T` = scalar or vector of any integer or floating-point type **EXCEPT** `bool`, `bfloat`,
`long`, `ulong`, `void`, `size_t`, `ptrdiff_t`. Bitwise ops use `Ti` (integer only).
"Active" threads only: an inactive thread (flow control, or insufficient work to fill the
group) does not contribute.

## Kernel thread-position attributes (put these in the kernel signature)

```metal
uint simd_size     [[threads_per_simdgroup]]        // SIMD width (32 on Apple GPUs)
uint simd_lane_id  [[thread_index_in_simdgroup]]    // 0..simd_size-1, this thread's lane
uint simd_group_id [[simdgroup_index_in_threadgroup]] // which SIMD-group within the threadgroup
```

## Reduction functions (Table 6.15) — return the reduced value to ALL active lanes

```metal
T  simd_sum(T data)        T  simd_product(T data)
T  simd_min(T data)        T  simd_max(T data)
Ti simd_and(Ti data)       Ti simd_or(Ti data)        Ti simd_xor(Ti data)
```

Prefix (scan) variants — per-thread partial over lower-indexed active lanes:

```metal
T simd_prefix_inclusive_sum(T data)     T simd_prefix_exclusive_sum(T data)     // first lane -> T(0)
T simd_prefix_inclusive_product(T data) T simd_prefix_exclusive_product(T data) // first lane -> T(1)
```

## Permute / shuffle functions (Table 6.14)

```metal
T  simd_broadcast(T data, ushort broadcast_lane_id)  // value from a specific lane
T  simd_broadcast_first(T data)                      // value from the lowest active lane
T  simd_shuffle(T data, ushort simd_lane_id)         // value from an arbitrary lane
T  simd_shuffle_up(T data, ushort delta)             // from lane (self-delta), no wrap, lower lanes unchanged
T  simd_shuffle_down(T data, ushort delta)           // from lane (self+delta), no wrap, upper lanes unchanged
T  simd_shuffle_rotate_up(T data, ushort delta)      // ...with wrap-around
T  simd_shuffle_rotate_down(T data, ushort delta)    // ...with wrap-around
Ti simd_shuffle_xor(Ti value, ushort mask)           // from lane (self ^ mask) — the butterfly used in tree reductions
// Metal 2.4+: simd_shuffle_and_fill_up/down(T data, T filling, ushort delta[, ushort modulo])
```

`delta`/`mask`/`lane_id` constraints: must be the same across the group for the shuffle/
rotate forms; `simd_shuffle`'s lane may differ per thread but must be a valid lane.

## Vote / ballot functions

```metal
bool      simd_all(bool expr)                 // true iff all active threads' expr is true
bool      simd_any(bool expr)                 // true iff any active thread's expr is true
simd_vote simd_ballot(bool expr)              // bitmask of expr over active lanes; inactive -> 0
simd_vote simd_active_threads_mask()          // == simd_ballot(true)
bool      simd_is_first()                     // true only on the lowest-indexed active lane
bool      simd_is_helper_thread()             // fragment-only
```

`simd_vote` wraps a `vote_t` bitmask; `.all()` / `.any()` test it, and it casts to `vote_t`.

## Quad-group functions (§6.10.3) — same idea at width 4

Header `<metal_quadgroup>`. `quad_sum/product/min/max/and/or/xor`, `quad_prefix_*`,
`quad_broadcast`, `quad_shuffle[_up/_down/_xor]`, `quad_all/any/ballot`, `quad_is_first`,
attributes `[[thread_index_in_quadgroup]]` / `[[quadgroup_index_in_threadgroup]]`. Useful
for small fixed-width reductions; less relevant to CT2's row reductions (use the full SIMD
width).

## Canonical threadgroup reduction (verbatim from the spec, §6.10.2.1)

The efficient pattern: reduce within each SIMD-group with `simd_shuffle_down` (or a single
`simd_sum`), write one partial per SIMD-group to threadgroup memory, barrier, then reduce
the partials.

```metal
kernel void reduce(const device int *input  [[buffer(0)]],
                   device atomic_int *output [[buffer(1)]],
                   threadgroup int   *ldata  [[threadgroup(0)]],
                   uint gid          [[thread_position_in_grid]],
                   uint lid          [[thread_position_in_threadgroup]],
                   uint lsize        [[threads_per_threadgroup]],
                   uint simd_size    [[threads_per_simdgroup]],
                   uint simd_lane_id [[thread_index_in_simdgroup]],
                   uint simd_group_id[[simdgroup_index_in_threadgroup]])
{
    int val = input[gid] + input[gid + lsize];
    for (uint s = lsize/simd_size; s > simd_size; s /= simd_size) {
        for (uint offset = simd_size/2; offset > 0; offset /= 2)
            val += simd_shuffle_down(val, offset);          // per-SIMD partial reduce
        if (simd_lane_id == 0) ldata[simd_group_id] = val;  // one partial per SIMD-group
        threadgroup_barrier(mem_flags::mem_threadgroup);
        val = (lid < s) ? ldata[lid] : 0;
    }
    for (uint offset = simd_size/2; offset > 0; offset /= 2)
        val += simd_shuffle_down(val, offset);              // final reduce
    if (lid == 0)
        atomic_fetch_add_explicit(output, val, memory_order_relaxed);
}
```

(The inner loop is equivalent to `val = simd_sum(val)` — `simd_sum` exists precisely so you
don't hand-roll the shuffle butterfly.)

---

### Relevance to the CT2 Metal backend

- CT2's row-reduction kernels — `ct2_softmax`, `ct2_rms_norm`, `ct2_layer_norm` in
  `kernels_msl.h` — do **full 256-thread tree reductions through threadgroup scratch
  arrays** (one threadgroup per row, fixed 256 threads). The textbook SIMD-group rewrite
  (one `simd_sum`/`simd_max` per 32-lane group → fold 8 partials → broadcast) **was tried,
  measured, and REVERTED (2026-06-09): it lost in every cell** (softmax 16384×512 fp32
  0.71→1.01ms, norms +8–15%). These kernels are **memory-bound** — loading the row
  dominates — so the existing tree reduction is already good and the SIMD version just adds
  barriers + a cross-group threadgroup bounce. The A/B harness `MetalTest.
DISABLED_BenchmarkReduction` is kept for any future attempt. DON'T re-chase this without a
  fundamentally different idea (e.g. fusing the reduction into an adjacent op to cut the
  memory traffic, not just changing how the reduction itself is computed).
- LayerNorm reduces sum **and** sum-of-squares together — do two `simd_sum`s (or a `float2`),
  keeping the existing single-pass structure.
- **Compute in `float`.** `T` excludes `bfloat` (and the fp16 kernels already widen to float
  for accumulation), so `simd_sum<float>` is the call; cast back to `half` on store. This
  also means if bf16 ever lands, its reductions must widen to float — `simd_sum` won't take
  `bfloat` directly.
- Keep threadgroup sizes a multiple of `threadExecutionWidth` (32). CT2's fixed-256 choice
  already is (8 SIMD-groups); `[[threads_per_simdgroup]]` is that 32 at runtime — read it,
  don't hardcode, in case of future GPUs.
- **Measure before committing** (per the project's perf culture): the tree reduction may
  already be memory-bound at these row sizes, in which case the SIMD-group rewrite is
  cleaner but not faster. Benchmark a graduated kernel against the existing one before
  declaring a win — see the perf graveyard in `SKILL.md` for why.
