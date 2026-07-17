---
title: "MSL address spaces (device / constant / threadgroup / thread)"
summary: >-
  Explains MSL's four address spaces from spec sections 4-4.8: device
  (runtime-sized readable/writable buffer memory for all tensor data),
  constant (read-only, required for program-scope variables initialized with
  core constant expressions and used for uniform broadcast data), thread (per-
  thread, invisible to others), and threadgroup (shared within one
  threadgroup, tile-backed and faster than device). It stresses that address
  spaces are disjoint with no casts, every pointer declaration and every
  kernel pointer argument must name one explicitly, and device memory has only
  threadgroup coherence by default. For the CT2 backend it maps the
  kernels_msl.h signature pattern: arrays are device, host scalars arrive as
  constant T& via setBytes, tuning constants like CT2_GEMM_S8_BM are program-
  scope constant, GEMM tiles are threadgroup, and char4 element
  reinterpretation within a space is legal.
---

Source (Apple): Metal Shading Language Specification, §4–4.4.1, §4.8 (v4.1, 2026-06-04).
PDF: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
(The MSL language rules are only in the spec PDF — there is no DocC HTML/JSON page for
them, so this was extracted from the PDF, not the usual DocC-JSON endpoint.)

Address spaces are **disjoint regions with access restrictions**, and they are part of the
pointer's type:

- Every pointer/reference declaration needs an explicit address-space attribute — a missing
  one is a **compilation error**.
- Every pointer/reference **argument to a kernel function** must be `device`,
  `constant`, `threadgroup`, or `threadgroup_imageblock`.
- There are no address-space casts: the spaces are disjoint, so a `threadgroup char*`
  cannot become a `device char*` etc. (Reinterpreting _element type_ within one space —
  e.g. `(const threadgroup char4*)&tile[i]` — is ordinary pointer casting and fine if
  alignment holds.)
- A variable at **program scope** must be in the `constant` address space.

## device (§4.1)

Buffer memory, **readable and writable**. Declared as pointer/reference to scalar, vector,
or user-defined struct; the Metal API call that allocates the `MTLBuffer` determines the
actual size (the kernel just sees a pointer). This is the space for anything sized at
runtime — all tensor data.

## constant (§4.2)

Buffer memory, **read-only** (writing is a compile-time error). Two distinct uses:

1. **Program-scope variables** must live here and must be **initialized at declaration
   with a core constant expression** (C++17 §5.20 — compile-time evaluable). Declaring one
   uninitialized is a compile-time error. Values persist across all function invocations
   in the program.
2. **Kernel arguments**: pointers/references to `constant` are allowed as function
   arguments. The spec (this version) imposes no compile-time-known-size rule on such
   buffers; the stated constraint is a GPU-dependent **minimum offset alignment** ("Minimum
   constant buffer offset alignment", Metal Feature Set Tables).

Use `constant` for data that is uniform across the dispatch and read by all threads
(scalars, small parameter blocks) — the hardware can broadcast/cache it; `device` reads are
optimized for per-thread divergent access.

## thread (§4.3)

Per-thread memory. Every variable declared inside a kernel/graphics function lives here and
is invisible to other threads. `thread float* p = &x;` is the explicit spelling.

## threadgroup (§4.4)

Memory shared by the threads of one threadgroup, "faster on most devices than sharing data
in device memory". Used to (a) declare a variable inside a kernel (`threadgroup float
tile[10];`) or (b) take a kernel parameter `threadgroup float* p [[threadgroup(0)]]` whose
size the host sets with `setThreadgroupMemoryLength:atIndex:`. Lifetime = the threadgroup's
execution of the kernel. §4.4.1 (SIMD-groups within a threadgroup) is summarized in
`threadgroup-and-simdgroup-synchronization.md`.

## The others (not relevant to this backend)

- `threadgroup_imageblock` (§4.5) — tile-shading imageblocks; no render passes here.
- `ray_data` (§4.6) — ray-tracing intersection payloads; not used.
- `object_data` (§4.7) — mesh-shader object→mesh payloads; not used.

## Memory coherency (§4.8)

`thread` memory has thread coherence, `threadgroup` memory threadgroup coherence, and
`device` memory **threadgroup coherence by default** — writes to a device buffer are only
guaranteed visible across _different_ threadgroups if marked `coherent(device)` (Metal
3.2+) and properly synchronized. The backend's kernels never communicate across
threadgroups within a dispatch, so the default suffices.

---

### Relevance to the CT2 Metal backend

All in `src/metal/kernels/kernels_msl.h`, dispatched from `src/metal/primitives.mm` and
`src/metal/gemm.mm`:

- **Arrays are `device`, host scalars are `constant T&`** — the file-wide signature
  pattern: `ct2_softmax_float(device const float* input …, constant uint& depth …)`,
  `ct2_gemm_s8(device const char* a …, constant uint& m …, constant int& alpha …)`. The
  `constant uint&` args pair with host-side `[encoder setBytes:&depth_u
length:sizeof(depth_u) atIndex:3]` (`primitives.mm`) — setBytes data lands in the
  constant address space, no `MTLBuffer` needed. See
  `compute-kernels-and-dispatch.md` for the encoder side.
- **Program-scope tuning constants** use rule (1): `constant uint CT2_SOFTMAX_TG = 256;`,
  `CT2_NORM_TG`, `CT2_QUANT_TG`, `CT2_GEMM_S8_BM/BN/BK` — compile-time initialized at
  program scope, as required.
- **Threadgroup tiles**: the int8 GEMM stages operands as `threadgroup char
As[CT2_GEMM_S8_BK][CT2_GEMM_S8_BM]` / `Bs[…]`, and the reductions declare
  `threadgroup float scratch[256]` inside the kernel (variant (a) — sized in-source, not
  host-bound, so no `[[threadgroup(n)]]` argument or `setThreadgroupMemoryLength` is
  involved).
- **In-space element reinterpretation** is load-bearing in the int8 kernels:
  `(const threadgroup char4*)(&As[kk][tid.y * 4u])` and `(device const char4*)(a +
(ulong)i * lda)` — legal because the address space is unchanged and the host routes only
  alignment-safe shapes to `ct2_gemv_s8` (the kernel's comment says so). What you can NOT
  do is cast between spaces.
- Helper functions show the parameter rule: `ct2_rms_norm_impl(device const T* input, …,
threadgroup float* scratch)` — non-kernel functions can take any address space, but
  every pointer still names one explicitly.
