# Threadgroup & SIMD-group synchronization (barriers)

Source (Apple): Metal Shading Language Specification, §6.10.1 + §4.4.1 (v4.1, 2026-06-04).
PDF: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
(The MSL standard-library functions are only in the spec PDF — there is no DocC HTML/JSON
page for them, so this was extracted from the PDF, not the usual DocC-JSON endpoint.)

Header `<metal_compute>`. A barrier is **both an execution barrier and (optionally) a memory
fence** — two separate jobs:

- **Execution ordering**: no thread continues past the barrier until all threads in the
  threadgroup (`threadgroup_barrier`) or SIMD-group (`simdgroup_barrier`) have reached it.
- **Memory ordering**: the `mem_flags` argument additionally fences reads/writes so that
  memory operations before the barrier are visible to the other threads after it. `mem_none`
  gives you the execution barrier with NO fence.

## The functions (Table 6.12)

```metal
void threadgroup_barrier(mem_flags flags)
void simdgroup_barrier(mem_flags flags)
// Metal 4.1+ variants add ordering/scope params (defaults shown):
void threadgroup_barrier(mem_flags, memory_order = memory_order_seq_cst,
                         thread_scope = thread_scope_threadgroup)
void simdgroup_barrier(mem_flags, memory_order = memory_order_seq_cst,
                       thread_scope = thread_scope_simdgroup)
```

`simdgroup_barrier`: macOS Metal 2+, iOS Metal 1.2+. Usable in kernel, fragment, mesh,
object functions and `[[visible]]` functions they call.

## mem_flags (Table 6.13) — what each variant orders

| Flag                         | Orders memory operations to…                                      |
| ---------------------------- | ----------------------------------------------------------------- |
| `mem_none`                   | nothing — execution barrier only, no memory fence                 |
| `mem_device`                 | **device** memory (buffers) for threads in the group              |
| `mem_threadgroup`            | **threadgroup** memory for threads in the group                   |
| `mem_texture`                | textures with `read_write` access (macOS Metal 1.2+/iOS Metal 2+) |
| `mem_threadgroup_imageblock` | threadgroup imageblock memory (tile shading — N/A here)           |
| `mem_object_data`            | `object_data` memory (mesh pipelines — N/A here)                  |

Flags are a bit field — combine when a phase wrote both spaces. The Metal 4.1 `scope`
argument widens _who observes_ the fenced accesses (threadgroup / SIMD-group / device);
the pre-4.1 forms the backend uses imply the function's natural scope.

## The divergence rule (the UB trap)

Verbatim semantics from §6.10.1:

- If a barrier is inside a **conditional** and any thread executes it, **all** threads in the
  threadgroup (or SIMD-group) must enter the conditional and execute it.
- If a barrier is inside a **loop**, then for **each iteration**, if any thread executes the
  barrier, all threads must execute it before any continue past it.

A barrier in divergent control flow is undefined behavior — typically a GPU hang. One spec
nuance (Apple silicon, v4.1): _"a thread that has ended no longer participates or blocks
remaining threads at a barrier"_ — i.e. a thread that **returned** (ended) does not deadlock
the others on Apple GPUs. Do not design around this: a guard that makes some threads skip a
barrier _and keep executing_ is still UB, and portable code should keep barriers uniform.

## When a barrier is NOT needed

SIMD-group functions (`simd_sum`, `simd_shuffle_*`, `simd_broadcast`… — see
`simd-group-functions.md`) let threads in one SIMD-group share data _"without using
threadgroup memory or requiring any synchronization operations, such as a barrier"_
(§6.10.2). Within a SIMD-group, `simdgroup_barrier` is only needed to order accesses that
go **through memory** (threadgroup or device); data exchanged via the SIMD-group functions
themselves needs nothing. §4.4.1: a SIMD-group is a collection of threads executing
concurrently; the SIMD-group mapping is invariant for the kernel's execution, and all
SIMD-groups in a threadgroup are the same size except possibly the last.

---

### Relevance to the CT2 Metal backend

All in `src/metal/kernels/kernels_msl.h`:

- **The tiled int8 GEMM (`ct2_gemm_s8`) is the textbook double-barrier loop**: stage the
  `As`/`Bs` char tiles → `threadgroup_barrier(mem_flags::mem_threadgroup)` → int32 MAC
  over the tile → second barrier before the next k-chunk overwrites the tiles. Both barriers
  are in a `for (k0 …)` loop whose trip count is uniform (depends only on `k`), satisfying
  the per-iteration all-threads rule. Removing the _second_ barrier is the classic bug:
  fast threads reload the tile while slow ones still read it.
- **The 256-thread tree reductions** (`ct2_softmax_*`, `ct2_rms_norm_impl`,
  `ct2_layer_norm_impl`, `ct2_quantize_s8_impl`) barrier after the strided load into
  `scratch`, then once per halving step (`for s = 128…1`). The `if (tid < s)` is fine —
  the barrier is _outside_ the conditional; only the scratch write diverges.
- **Early-exit guards must be uniform per group.** `ct2_softmax_*`'s `if (size == 0u)
return;` is safe because `size` depends only on `row` = the threadgroup id — all 256
  threads return together, before any barrier. `ct2_gemv_s8`'s `if (j >= n) return;` is
  uniform per SIMD-group (the kernel's own comment says so) so the later `simd_sum` stays
  in uniform control flow. The bug class to review for: a guard on `tid`/`gid` that lets
  _some_ threads of a group skip a barrier or a `simd_sum`.
- The reductions use **no `simdgroup_barrier` at all** — within-SIMD-group sharing isn't
  exploited (the SIMD-group rewrite was tried and reverted as a perf loss; see
  `simd-group-functions.md`). `ct2_gemv_s8` needs no barrier either: `simd_sum` does the
  cross-lane fold barrier-free.
