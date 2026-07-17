---
title: "The Gemm Op & Its Dtype Dispatch"
summary: >-
  Details ops::Gemm and its dtype dispatch: the construction-time
  alpha/beta/transpose interface, the operator() switch on a.dtype() routing
  INT8 to compute<D, int8_t, int32_t> (int8xint8 into an int32 accumulator),
  INT16 to CPU only, and floats to same-in-same-out, followed by unconditional
  apply_bias_and_activation. It explains that compute<D,In,Out> collapses
  leading dims into m and calls primitives<D>::gemm per device (CPU switching
  on the runtime gemm_s8_backend, CUDA via cublasGemmEx, Metal via
  operator()-level early routing). The load-bearing contract is that integer
  alpha=1/beta=0 is convention not assertion — enforced by callers like
  quantized Dense while Dequantize owns all scaling — and that CUDA truncates,
  Ruy emulates, and the Metal route guards on beta==0 with integral alpha and
  no shift compensation. Also explains a_shift_compensation: MKL's s8-to-u8
  shift-by-128 trick and its per-column compute_u8_compensation, which signed-
  int8 backends (cuBLAS, Metal, CUDA) skip entirely.
---

How `ops::Gemm` resolves dtype → implementation, what the integer-GEMM alpha/beta
contract really is, and why the MKL u8-shift compensation exists.

**Sources (read these, all citations below are from real lines):**

- `include/ctranslate2/ops/gemm.h`
- `src/ops/gemm.cc`
- `src/cpu/primitives.cc` (the int8 CPU GEMM, `:903-1075`), `src/cpu/backend.cc`
- `src/cuda/primitives.cu` (the cuBLAS int8 wrapper, `:573-597`)

---

## 1. Interface

```cpp
// include/ctranslate2/ops/gemm.h:17-30
Gemm(float alpha = 1, float beta = 1,
     bool trans_a = false, bool trans_b = false,
     bool a_is_packed = false, bool b_is_packed = false,
     const ActivationType* activation_type = nullptr);

void operator()(const StorageView& a,
                const StorageView& b,
                StorageView& c,
                const StorageView* a_shift_compensation = nullptr,
                const StorageView* bias = nullptr,
                const StorageView* residual = nullptr) const;
```

Alpha/beta/transposes are **construction-time** state; the kernel is the private
`compute<Device D, typename In, typename Out>` (`ops/gemm.h:56-60`) — note the separate
input/output types, which is what lets int8 inputs produce an int32 output.

## 2. `operator()`: a dtype switch, then bias+activation

`Gemm::operator()` (`src/ops/gemm.cc:98-151`) switches on `a.dtype()`:

- **INT8** → `DEVICE_DISPATCH(a.device(), (compute<D, int8_t, int32_t>(a, b, c, a_shift_compensation)))`
  (`ops/gemm.cc:119`). This is the quantized path: int8×int8 → **int32 accumulator**.
- **INT16** → CPU only, `compute<Device::CPU, int16_t, int32_t>` (`ops/gemm.cc:122-125`).
- **FLOAT32/FLOAT16/BFLOAT16** → `DEVICE_AND_FLOAT_DISPATCH("Gemm", ..., (compute<D, T, T>(...)))`
  (`ops/gemm.cc:141-142`) — same type in and out.

After the switch, unconditionally: `apply_bias_and_activation(c, bias, _activation_type, residual)`
(`ops/gemm.cc:150`; the helper is defined at `ops/gemm.cc:16-30`, declared `ops/gemm.h:9-13` — it
computes `activation(x + bias + residual)` via `BiasAdd`/`Add`). For the **quantized**
path the caller deliberately passes no activation and no bias to Gemm — they belong to
Dequantize's epilogue, because an int32 accumulator can't take a float bias
(see `dense-layer-and-quantized-linear.md` and `quantization-scheme-and-ops.md`).

## 3. `compute<D, In, Out>` → `primitives<D>::gemm`

`Gemm::compute` (`ops/gemm.cc:153-181`) derives `m/n/k` and leading dimensions from the
transpose flags, **collapses leading dims** into `m` (`const dim_t m = a.size() / k;`,
`ops/gemm.cc:160`), resizes `c`, and calls `primitives<D>::gemm(...)` (`ops/gemm.cc:172-180`)
with alpha, beta, and the optional `a_shift_compensation` pointer. Per device:

- **CPU**: `primitives<Device::CPU>::gemm` for int8 (`src/cpu/primitives.cc:903`)
  switches on the runtime-selected backend (`switch (gemm_s8_backend)`,
  `primitives.cc:917`; chosen once at `primitives.cc:544`): MKL `cblas_gemm_s8u8s32`
  (`:963-971`), DNNL (`:989-1004`), Ruy (`:1013-1067`), else
  `throw "No INT8 GEMM backend for CPU"` (`:1071`).
- **CUDA**: `primitives<Device::CUDA>::gemm` for int8 wraps `cublasGemmEx` with
  `CUDA_R_8I` operands, `CUDA_R_32I` output and `CUBLAS_COMPUTE_32I`
  (`src/cuda/primitives.cu:573-597`).
- **Metal** is _not_ a `primitives<>` specialization — it's the operator()-level
  early-routing pattern (`ops/gemm.cc:108-118` for int8, `:131-140` for fp32/fp16), returning
  before the generic dispatch. See `dispatch-and-op-implementation.md` §3-4 and the
  `apple-silicon` skill.

## 4. The integer alpha/beta contract — convention, not assertion

Precisely what the code enforces:

- **Gemm itself asserts nothing** about alpha/beta for int8. There is no check in
  `gemm.cc`'s INT8 case (`ops/gemm.cc:106-120`) or in `compute`.
- **The contract is established by the caller**: quantized `Dense` constructs its Gemm
  with `/*alpha=*/1, /*beta=*/0` (`src/layers/common.cc:291-292`), and `Dequantize` owns
  _all_ scaling — it divides the int32 output by `a_scale * b_scale`
  (`src/ops/dequantize_gpu.cu:40-48`). Scales are floats, so folding them into the GEMM
  would force float math inside the accumulation; keeping the GEMM at alpha=1/beta=0
  keeps it exact pure-int32.
- What each backend actually does with other values (why straying is unsafe):
  - CUDA **truncates** to integers: `int32_t alpha_i = alpha; int32_t beta_i = beta;`
    (`src/cuda/primitives.cu:582-583`) — a fractional alpha is silently floored.
  - Ruy runs the multiply at alpha=1/beta=0 and **emulates** other values with extra
    full passes over `c` (plus a saved copy of `c` for beta) — `src/cpu/primitives.cc:1039-1060`.
  - MKL/DNNL accept float alpha/beta natively in `cblas_gemm_s8u8s32` (`:963-971`).
  - The Metal route **guards and falls through**: it is taken only when
    `_beta == 0 && _alpha == static_cast<float>(static_cast<int32_t>(_alpha))`
    (`ops/gemm.cc:113-114`); anything else drops to the generic dispatch, since "a float
    alpha cannot be applied exactly to an int32 accumulator" (`ops/gemm.cc:76-77`).

  So alpha=1/beta=0 is the only configuration that is exact and identically supported
  everywhere — and it's the only one the engine's own layers use.

## 5. `a_shift_compensation`: the MKL s8→u8 story

MKL has no signed×signed int8 GEMM; it implements `s8s8s32` via `cblas_gemm_s8u8s32`,
which "expects a to be unsigned and b to be signed" (`src/cpu/primitives.cc:921-924`).
So the activation matrix A is **shifted by +128** into uint8 (`shift_to_u8`,
`primitives.cc:864-866` — or already shifted at quantization time via Quantize's
`shift_to_uint8` option), which adds an error of `128 * Σ_k B[k, j]` per output column.
The fix is a per-column **compensation row** added to C:
`compute_u8_compensation` computes `-128 * alpha * Σ_k B[k, j]` (`primitives.cc:870-898`;
exposed as `Gemm::compensate_u8_input`, `ops/gemm.cc:244-260`) and MKL applies it via its
`CblasRowOffset` mechanism (`:963-971`).

Two flows inside the MKL case (`primitives.cc:929-945`): if `a_shift_compensation` is
passed, A is assumed already shifted (the weight's compensation was precomputed at
model load as `{scope}/weight_compensation` — see Dense); if null, the impl shifts A and
computes the compensation on the fly. The whole mechanism is keyed on
`cpu::prefer_u8s8s32_gemm()` = backend is MKL or DNNL (`src/cpu/backend.cc:96-99`).

**Signed-int8 backends skip all of it**: the CUDA wrapper's compensation parameter is
unnamed and ignored (`const int32_t*`, `src/cuda/primitives.cu:594` — cuBLAS is natively
s8s8s32), and the Metal route requires `!a_shift_compensation` (`ops/gemm.cc:113`).

---

### Relevance to the Metal backend

- The int8 hook is `metal_gemm_int8` (`src/ops/gemm.cc:78-95`) → `metal::gemm_s8`
  (declared `src/metal/primitives.h:129`, implemented in `src/metal/primitives.mm` with
  MSL kernels in `src/metal/kernels/kernels_msl.h`) — a native int8×int8→int32 tiled
  GEMM/GEMV, so quantized weights stay int8-resident on the GPU.
- The guard at `ops/gemm.cc:113-114` (no compensation, beta==0, integral alpha) is the
  alpha/beta contract of §4 made explicit: unexpected combinations fall through to the
  CPU reference instead of silently dropping terms.
- int32 accumulation made parity testable bit-exactly at any depth (deep-GEMM int32
  exactness tests in `tests/metal_test.cc`).
- Apple GPUs have no int8 matrix units, so this is an ALU-bound kernel — perf model and
  kernel details in the `apple-silicon` skill, measurements in `METAL_BENCHMARKS.md`.
