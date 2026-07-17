---
title: "Atomic functions, memory order, and fences"
summary: >-
  Metal Shading Language atomic functions, memory order, and fences (MSL spec
  2.6 and 6.16): which types support atomic<T> (int/uint/bool/ulong/float,
  with no atomic_char/short/half/bfloat and float restricted to device-memory
  add/sub), the atomic_store/load/exchange/compare_exchange_weak/fetch_KEY
  function surface, and why memory_order_relaxed is effectively the only
  usable order until the Metal 4.1 floor. It explains atomic_thread_fence,
  thread_scope, and that nothing barriers across threadgroups within a
  dispatch except relaxed device atomics or splitting into two dispatches. The
  CT2 relevance is that kernels_msl.h is atomics-free by design (each
  threadgroup owns its output tile), so adding an atomic is a design change; a
  split-k int8 GEMM could safely use atomic_fetch_add on int32 partials while
  a float split-k would not be bit-exact.
semantic_id: "yhm85o7uo33Sg6z9w_1pvYUgN2vTi6MhDlf2GZSfweQqDXkHXKcvXi94SB1_p48Kgi0Eks_7hZ5utW2jBvxroA"
---

Source (Apple): Metal Shading Language Specification, §2.6 (atomic types) and §6.16
(functions, memory order, thread scope, fences) (v4.1, 2026-06-04).
PDF: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
(The MSL standard-library functions are only in the spec PDF — there is no DocC HTML/JSON
page for them, so this was extracted from the PDF, not the usual DocC-JSON endpoint.)

Header `<metal_atomic>` — "a subset of the C++17 atomics and synchronization operations."
Atomic functions operate **only** on Metal atomic types (§2.6).

## Atomic-capable types (§2.6) — and what is NOT

`atomic<T>` with `T` ∈ `int`/`int32_t`, `uint`/`uint32_t`, `bool` (Metal 2.4+), `ulong`
(Metal 2.4+), `float` (Metal 3+). Aliases: `atomic_int`, `atomic_uint`, `atomic_bool`,
`atomic_ulong`, `atomic_float`.

- **No `atomic_char`/`short`/`half`/`bfloat`** — int8 or fp16 data cannot be updated
  atomically; widen partials to int32/float first.
- **`atomic_float` is restricted**: load/store/exchange/compare-exchange work, but among the
  fetch-modify ops only **add and sub**, and only in **device** memory ("Metal 3 supports the
  atomic_float for device memory only"; Metal 4.1 adds threadgroup float add/sub). No float
  fetch_max/min.
- **`atomic_ulong` (64-bit) is its own narrow API** (§6.16.4.6, Metal 2.4+, Apple silicon,
  check the Feature Set Tables): only `atomic_max_explicit`/`atomic_min_explicit`, device
  memory, returning `void`.

## Memory order (§6.16.1) — relaxed is effectively the whole story

```metal
enum memory_order { memory_order_relaxed, memory_order_acquire, memory_order_release,
                    memory_order_acq_rel, memory_order_seq_cst };
```

Spec: "For atomic operations other than `atomic_thread_fence`, **`memory_order_relaxed` is
the only enumeration value**" — atomicity and a consistent modification order, no
synchronization, no ordering of surrounding accesses. The stronger orders exist only for:
`atomic_thread_fence` with `seq_cst` (Metal 3.2+), and — **Metal 4.1+ only** —
acquire/release/acq_rel on atomic ops, fences, `threadgroup_barrier`, `simdgroup_barrier`.
Until the deployment floor is Metal 4.1, write kernels as if relaxed is all there is and get
ordering from barriers/fences, not from the atomics themselves.

## The functions (§6.16.4) — `A` atomic type, `C` its plain type

All take a final `memory_order order` (= relaxed); each has `threadgroup A*` and `device A*`
overloads (plus `volatile` variants; Metal 4.1 adds trailing `mem_flags` overloads).

```metal
void atomic_store_explicit(A* obj, C desired, memory_order)
C    atomic_load_explicit(const A* obj, memory_order)
C    atomic_exchange_explicit(A* obj, C desired, memory_order)
bool atomic_compare_exchange_weak_explicit(A* obj, thread C* expected, C desired,
                                           memory_order success, memory_order failure)
     // on failure, *expected is overwritten with the actual value; weak = may spuriously fail
C    atomic_fetch_KEY_explicit(A* obj, C operand, memory_order)   // returns the OLD value
     // KEY ∈ add, sub, and, or, xor, max, min   (Table 6.27)
     // atomic_int / atomic_uint: all keys; atomic_float: add, sub only (device memory)
```

Signed overflow in fetch-add/sub is defined: two's complement, silent wrap-around.

## Fences and barriers (§6.16.2–6.16.3, §6.10.1)

```metal
void atomic_thread_fence(mem_flags flags, memory_order order,
                         thread_scope scope = thread_scope_device)  // Metal 3.2+, Apple silicon
enum thread_scope { thread_scope_thread, thread_scope_simdgroup,
                    thread_scope_threadgroup, thread_scope_device };
```

`mem_flags` picks the address space(s) fenced (`mem_threadgroup`, `mem_device`, …); `order`
picks acquire/release/seq_cst semantics (relaxed = no effect); `scope` picks which threads the
ordering reaches. Distinct from `threadgroup_barrier(mem_flags)`: the barrier is an
**execution** rendezvous + memory fence for one threadgroup only. **Nothing barriers across
threadgroups within a dispatch** — relaxed device atomics (plus, if ordering is needed, a
device-scope fence) are the only cross-threadgroup mechanism; otherwise split the work into
two dispatches and let the command buffer order them.

The spec's own reduce example (quoted in full in `simd-group-functions.md`) shows the
canonical pattern: per-threadgroup reduction via `simd_shuffle_down` + `threadgroup_barrier`,
then one `atomic_fetch_add_explicit(output, val, memory_order_relaxed)` per threadgroup to
combine across the grid.

---

### Relevance to the CT2 Metal backend

- **`src/metal/kernels/kernels_msl.h` is atomics-free by design** (grep confirms: zero
  `atomic` uses). Every kernel gives each threadgroup exclusive ownership of its output —
  `ct2_gemm_s8` owns a 64×64 C tile, `ct2_gemv_s8` a single C element per SIMD-group, the
  quantize/norm/softmax kernels a whole row — so no cross-threadgroup writes ever collide.
  Intra-threadgroup reductions use `threadgroup_barrier` + scratch, not atomics. **Adding an
  atomic is therefore a design change, not a tweak**: it introduces cross-threadgroup
  coupling the current correctness argument doesn't cover, and (for float) order-dependent
  rounding that breaks CPU-parity expectations (`math-functions-and-numeric-parity.md`).
- **If split-k GEMM is ever attempted** (k-axis split across threadgroups to fill the GPU at
  small m×n), the int8 path has one real advantage: partials are **int32**, and integer
  addition is associative/commutative — `atomic_fetch_add_explicit` on `atomic_int` keeps the
  result bit-exact regardless of arrival order, preserving the "bit-exact by construction"
  contract. A float split-k (fp16/fp32 GEMM) would not be deterministic. The alternative that
  stays atomics-free: write partials to a scratch buffer and reduce in a second dispatch.
- Relaxed-only ordering is sufficient for such accumulation (atomicity + modification order;
  no inter-thread data dependencies) — but a "flag + payload" handshake between threadgroups
  is **not** expressible pre-Metal-4.1 without fences; don't design one in.
- The `c[...] = alpha * acc` stores in `ct2_gemv_s8` are plain stores precisely because
  `j >= n` SIMD-groups returned early and every (i, j) is written exactly once — keep that
  invariant when touching the host-side routing in `gemm_s8` (`src/metal/primitives.mm`).
