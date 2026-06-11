# int8 quantize/dequantize kernels — the three supporting kernels around the GEMM

Source: `src/metal/kernels/kernels_msl.h` (kernels `ct2_quantize_s8_{float,half}`
~612–672, `ct2_dequantize_s8_{float,half}` ~674–699, `ct2_dequant_gemm_out_{float,half}`
~701–742, `ct2_apply_activation` ~518–530), `src/metal/primitives.mm` (dispatch impls
~442–580), routing in `src/ops/quantize.cc` (~44–60) and `src/ops/dequantize.cc`
(~38–52, ~76–101). Commit `eaa0f187` (Phase 1). Tests: `tests/metal_test.cc` (named
below), verified green 2026-06-11 (M4 Max).

These three kernels are what make the native int8 GEMM (`int8-gemm-kernel-design.md`)
usable: activations are quantized per row on the fly, and the GEMM's int32 output is
rescaled + biased + activated in one epilogue pass. All exist in fp32 and fp16
variants (the `_half` twins matter: without them, fp16 tensors in an `int8_float16`
model would hit the float32-only CPU reference dispatch and throw in a non-CUDA build).

## 1. `ct2_quantize_s8_*` — per-row symmetric quantization

CT2's int8 scheme is **symmetric per-row dynamic quantization, no zero-point**:
`scale = 127 / amax(row)`, `y = round_or_truncate(x * scale)`.

- **Geometry: one threadgroup per row**, fixed `CT2_QUANT_TG = 256` threads (host
  mirror `kQuantizeThreadgroup` — must match, the tree reduction assumes it), via
  `dispatchThreadgroups(batch_size × 1 × 1, 256 × 1 × 1)`.
- **amax**: each thread strides the row accumulating a local `fabs` max, then the same
  fixed power-of-two tree reduction as the norm kernels (threadgroup scratch +
  barriers). Zero rows get `scale = 1.0` (guard against 127/0;
  `QuantizeINT8ZeroRow` in ops_test.cc covers it on METAL).
- **`precise::divide(127.0f, amax)`** for the scale: the library compiles under
  default fast math (see `math-functions-and-numeric-parity.md`), whose division is
  not correctly rounded; the CPU reference divides with IEEE semantics and the op
  tests compare scales near-exactly, so the precise variant is load-bearing.
- **Rounding**: `rint(v)` — round-half-to-even, matching the CPU's
  `nearbyintf`/`vrndnq_f32` — when `round_before_cast` is set (the modern path);
  the legacy `round_before_cast=false` path truncates toward zero like a C cast.
  This is why `QuantizeINT8` is **bit-exact** against the CPU reference on METAL.

Scalars (`depth`, `round_before_cast`) go in via `setBytes`; the row scales come back
as a `float` buffer (one per row) consumed later as `a_scale`.

Routing (`src/ops/quantize.cc`): Metal device + signed path + fp32/fp16 input →
GPU kernel. The `shift_to_uint8` variant is a CPU-GEMM-backend (u8s8s32) concern and
deliberately falls through to the CPU reference over unified memory.

## 2. `ct2_dequantize_s8_*` — the simple form

`y = (float)x * (1 / scale[row])` — **reciprocal-then-multiply, spelled exactly like
the CPU kernel** (which precomputes `r_scale = 1/scale` per row), so fp32 output is
arithmetic-identical. One thread per element, `dispatchThreads(batch*depth)`; the row
is recovered as `gid / depth`. Used for int8 **embeddings** (the fp16 output variant
exists for `int8_float16` models). Routed in `src/ops/dequantize.cc` first overload.

## 3. `ct2_dequant_gemm_out_*` — the quantized-Dense epilogue

The int32 GEMM accumulator → float output, fused with bias and activation:

```c
float v = (float)c[gid] * precise::divide(1.0f, a_scale[row]);  // dynamic per-row act scale
v = precise::divide(v, b_scale[col]);                           // static per-out-channel weight scale
if (has_bias != 0u)
  v += (float)bias[col];
y[gid] = (T)ct2_apply_activation(v, act);
```

Operation order and the reciprocal/divide split deliberately **mirror the CPU
kernel**. One thread per element. The fused activations are the shared
`ct2_apply_activation` switch — **all seven `ActivationType`s plus identity**
(`act = -1` when none): `ReLU`(0), `GELUTanh`(1), `Swish`(2), `GELU`(3, via the
hand-rolled `ct2_erf` — MSL has none), `GELUSigmoid`(4), `Tanh`(5, via the
NaN-clamped `ct2_tanh_safe`), `Sigmoid`(6). The int passed from the host is
`static_cast<int>(*_activation_type)`, so the enum order in
`include/ctranslate2/ops/activation.h` is the contract.

Routing (`src/ops/dequantize.cc` second overload) requires the Dense layout —
`!transpose_a && transpose_b`, **non-scalar** per-row `a_scale` (size = batch) and
per-output-channel `b_scale` (size = depth); other scale layouts fall through to the
CPU reference.

**Encoder gotcha** (`dequantize_gemm_output_s8_impl`, primitives.mm ~518): every bound
buffer index must exist (see `op-graduation-playbook.md`), so when there is no bias,
index 3 is bound to the **`c` buffer as a never-read dummy** with `has_bias = 0`.

## fp16 vs fp32 handling — one template, two entry points

All three kernels are templates over the storage type `T` with **all arithmetic in
float**: fp16 values are upcast on load and downcast on the final store. Scales are
always `float` buffers regardless of `T`. So fp32 outputs are bit/near-exact vs the
CPU and fp16 outputs are tested at half tolerance (2e-2).

## Test coverage (`tests/metal_test.cc`)

- `Int8QuantizeFloat16MatchesFloat32` (:395) — fp16 input quantizes to **identical
  int8 codes and scales** as the fp32 CPU reference (fp16 has no CPU reference at all).
- `Int8DequantizeMatchesCPU` (:373) — simple form, fp32 exact + fp16 within 2e-2.
- `Int8DequantizeGemmOutputMatchesCPU` (:417) — the epilogue through **every
  activation variant × {bias, no-bias} × {fp32, fp16}**; tight tolerance (1e-4), not
  bit-exact — GELU rides `ct2_erf` and the transcendentals are fast-math.
- Plus from the parameterized op suite running on METAL: `QuantizeINT8`,
  `QuantizeINT8ZeroRow` (bit-exact), `GemmInt8`.

### Relevance to the CT2 Metal backend

- Wired: `ops::Quantize` / `ops::Dequantize` (both overloads) route at `operator()`
  level to `metal::quantize_s8` / `dequantize_s8` / `dequantize_gemm_output_s8`
  (`src/metal/primitives.{h,mm}`). Every quantized Dense on Metal runs
  quantize → `gemm_s8` → dequant-epilogue; these launches are also the structural
  per-Dense overhead that keeps int8 decode ~15% behind fp16 e2e
  (`dispatch-overlap-and-perf-model.md`).
- Regressions to watch: `precise::divide` and `rint` are the parity load-bearers — a
  fast-math "simplification" breaks bit-exactness against the CPU reference; the
  256-thread host/kernel coupling; the dummy-bias binding; the
  `ActivationType`-enum-order ↔ `ct2_apply_activation`-switch coupling.
- Future work: fusing the quantize or epilogue into the GEMM kernels (fewer launches
  per Dense — the honest attack on the remaining decode gap) starts by absorbing
  these kernels, so their contracts above are the spec.
