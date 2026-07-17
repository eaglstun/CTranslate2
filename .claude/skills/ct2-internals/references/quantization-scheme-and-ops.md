---
title: "Quantization Scheme & the Quantize/Dequantize Ops"
summary: >-
  CT2's int8 quantization scheme — symmetric, per-row, scale=127/amax with no
  zero-point — and the Quantize and Dequantize ops that implement it on any
  device. The same convention is defined in the CPU kernel quantize_s8_row,
  the CUDA quantize_kernel, and the Python converter _quantize; activations
  are quantized dynamically each forward pass while weight scales are static
  in {scope}/weight_scale. It details Quantize's three ctor options
  (int16_scale_type governing int16 only, shift_to_uint8 for CPU u8s8s32
  backends and CUDA/Metal-skipped, round_before_cast gated on binary version
  5) and Dequantize's two overloads: simple int8-to-float and the GEMM-output
  rescale that divides int32 by a_scale*b_scale then fuses bias and
  activation. int32 accumulation is the contract, keeping GEMM pure-integer
  and exact; AWQ ({scope}/weight_zero) is a separate asymmetric scheme. Metal
  routes all three kernels with the fused epilogue and also accepts fp16
  endpoints.
---

CT2's int8 scheme (symmetric, per-row, no zero-point) and the two ops that implement it
at runtime, on any device.

**Sources (read these, all citations below are from real lines):**

- `include/ctranslate2/ops/quantize.h` / `include/ctranslate2/ops/dequantize.h`
- `src/ops/quantize.cc` / `src/ops/quantize_cpu.cc` / `src/ops/quantize_gpu.cu`
- `src/ops/dequantize.cc` / `src/ops/dequantize_cpu.cc` / `src/ops/dequantize_gpu.cu`
- `src/cpu/kernels.cc` (`quantize_s8_row`), `python/ctranslate2/specs/model_spec.py` (`_quantize`)

---

## 1. The scheme: symmetric per-row, scale = 127/amax, no zero-point

Both weights and activations use the same convention: for each **row** (last-dim vector),
`scale = 127 / amax(row)`, `q = round(x * scale)`, recovered as `x ≈ q / scale`. The scale
**multiplies** on the way in and **divides** on the way out; there is no zero-point.
Defined identically in three places:

- CPU kernel `quantize_s8_row`: `const auto scale = (amax != 0.f ? int8_max / amax : 1.f)` (`src/cpu/kernels.cc:585-586`).
- CUDA `quantize_kernel`: `scale = max != 0.f ? 127.f / max : 1.f; scales[blockIdx.x] = scale` (`src/ops/quantize_gpu.cu:75-76`).
- Python converter `_quantize`: `scale = 127.0 / amax` per output row, `amax[amax == 0] = 127.0` (`python/ctranslate2/specs/model_spec.py:234-236`).

**Activations are quantized dynamically** — the `Quantize` op runs every forward pass and
emits one scale per row of the flattened input: `scale.resize({batch_size})` where
`batch_size = input.size() / input.dim(-1)` (`src/ops/quantize.cc:40-42`).

**Weight scales are static** — computed offline by the converter (`model_spec.py:199-239`)
or at load time by `Model::ensure_dtype`, which reuses the same `Quantize` op ("Use the
same quantization logic as in model_spec.py", `src/models/model.cc:303-328`). They are
stored as the model variable `{scope}/weight_scale` and picked up by layers:
`model.get_variable_if_exists(scope + "/weight_scale")` (`src/layers/common.cc:277`
for Dense; `:56` Embeddings; `:531` Conv1D).

> **Scope note — `_qzero`/AWQ is a DIFFERENT scheme.** A `{scope}/weight_zero` variable
> (`src/layers/common.cc:278`) routes Dense to the asymmetric AWQ path
> (`ops::DequantizeAwq` / `GemmAwq` / `GemvAwq`, `src/layers/common.cc:406-442`), which has
> zero-points and its own packed format. Nothing below applies to it.

## 2. The Quantize op

```cpp
// include/ctranslate2/ops/quantize.h:18-20
Quantize(const ScaleType int16_scale_type = ScaleType::GLOBAL,
         const bool shift_to_uint8 = false,
         const bool round_before_cast = true);
```

The three ctor options:

- **`int16_scale_type`** — governs **int16 only** (int8 is always per-row, no choice).
  `GLOBAL` uses the fixed constant `global_int16_scale = 1000` (`src/ops/quantize.cc:12`);
  `PER_LAYER` computes `(1 << 10) / amax` over the whole tensor — 10 bits of input so the
  product is 20 bits, leaving 12 for accumulation (`src/ops/quantize_cpu.cc:54-58`). The
  enum has a `PER_ROW` member (`quantize.h:10-14`) but the ctor rejects it for int16
  (`src/ops/quantize.cc:21-22`) — it exists only as the implicit int8 behavior.
- **`shift_to_uint8`** — emit `uint8` by subtracting `int8_min` (i.e. adding 128) after
  scaling (`src/cpu/kernels.cc:593-602`). This exists for CPU GEMM backends that only
  implement `u8s8s32` (MKL/DNNL — see the a_shift/compensation story in
  `gemm-op-and-dtype-dispatch.md`). It is CPU-only: the CUDA impl throws
  `"Shift to uin8_t is not defined on CUDA"` (`src/ops/quantize_gpu.cu:91-92`).
- **`round_before_cast`** — apply `nearbyint`/round before the int cast instead of relying
  on the cast's truncation (`src/ops/quantize_gpu.cu:81-84`; CPU picks the round functor at
  `src/ops/quantize_cpu.cc:63-66`). Models declare it via
  `round_before_cast_in_quantization()` = `_binary_version >= 5`
  (`include/ctranslate2/models/model.h:87-89`) — old models were saved without rounding,
  and reproducing their quantization bit-exactly matters more than the accuracy nit.

`Quantize::operator()` (`src/ops/quantize.cc:25-72`) switches on the **output** dtype:
INT16 is CPU-only (`:33-35`); INT8 sizes the per-row scale (`:40-42`) then runs
`DEVICE_AND_FLOAT_DISPATCH("Quantize", ..., (quantize<D, T, int8_t>(...)))` (`:62-63`).
CPU instantiates `<Device::CPU, float, int8_t>` via `CPU_ISA_DISPATCH` →
`cpu::quantize_s8<ISA>` (`src/ops/quantize_cpu.cc:11-25`); CUDA instantiates
float/float16/bfloat16 inputs (`src/ops/quantize_gpu.cu:113-115`).

## 3. The Dequantize op — two overloads

`Dequantize` takes the activation as a ctor option (`dequantize.h:11`), because the second
overload fuses it. The two overloads (`include/ctranslate2/ops/dequantize.h:13-24`):

**Simple int8→float**: `operator()(input, scale, output)` — per-row `y = x / scale`
(CPU `dequantize_kernel` multiplies by `1.f / scale`, `src/ops/dequantize_cpu.cc:12-21`).
Validates one scale per row: `scale.size() != batch_size` throws (`src/ops/dequantize.cc:34-36`).
Used e.g. by quantized `Embeddings` after gathering int8 rows (`src/layers/common.cc:71-84`).

**GEMM-output rescale** — the int8 pipeline's epilogue:

```cpp
// include/ctranslate2/ops/dequantize.h:17-24
// Rescales the int32 GEMM output to float32, given the input scales.
void operator()(const StorageView& c,        // int32 GEMM accumulator
                const StorageView& a_scale,  // per-row activation scales
                const StorageView& b_scale,  // per-output-channel weight scales
                const bool transpose_a,
                const bool transpose_b,
                StorageView& y,
                const StorageView* bias = nullptr) const;
```

Semantics (as the CUDA kernel comments it, `src/ops/dequantize_gpu.cu:40-48`):
`y[i][j] = c[i][j] / (a_scales[trans_a ? j : i] * b_scales[trans_b ? j : i])`, then
`+ bias[j]` (`:50-51`), then the activation epilogue (`:52`; the activation switch covers
ReLU/GELU/GELUTanh/GELUSigmoid/Sigmoid/Swish/Tanh, `src/ops/dequantize_gpu.cu:70-119`).
The CPU side has a vectorized fused kernel for the common `!trans_a && trans_b` layout
(`src/ops/dequantize_cpu.cc:75-83`) and a generic fallback that divides then calls
`apply_bias_and_activation` (`:85-99`).

## 4. int32 accumulation is the contract

Integer GEMM always accumulates in **int32** — `Gemm::compute<D, int8_t, int32_t>`
(`src/ops/gemm.cc:119`) — and `dequantize_gemm_output` always reads int32
(`c.data<int32_t>()`, `src/ops/dequantize_cpu.cc:60`, `ops/dequantize_gpu.cu:134`). All
scaling/bias/activation lives in Dequantize, so the GEMM itself runs pure-integer and
exact (see the alpha/beta contract in `gemm-op-and-dtype-dispatch.md`). The full pipeline
that `Dense` orchestrates (`dense-layer-and-quantized-linear.md`):

```text
fp activation ──Quantize──▶ int8 + a_scale ─┐
                                            ├─▶ int8 GEMM ──▶ int32 ──Dequantize(a_scale, b_scale,
weight (int8, static b_scale from           │                          +bias, +activation)──▶ fp out
  {scope}/weight_scale) ────────────────────┘
```

---

### Relevance to the Metal backend

- Both ops route to GPU kernels by the operator()-level targeted-routing pattern:
  `src/ops/quantize.cc:44-60` → `metal::quantize_s8`, `src/ops/dequantize.cc:38-52` and
  `:76-101` → `metal::dequantize_s8` / `metal::dequantize_gemm_output_s8` (declared in
  `src/metal/primitives.h:99-121`, MSL bodies in `src/metal/kernels/kernels_msl.h`).
- The Metal routing also accepts **fp16** endpoints (int8_float16 models), which the
  generic `DEVICE_AND_FLOAT_DISPATCH` would reject in a non-CUDA build.
- `shift_to_uint8` is skipped on Metal exactly like CUDA — it's a CPU-GEMM-backend
  (u8s8s32) concern; the Metal guard is `!_shift_to_uint8` (`src/ops/quantize.cc:50`).
- The int8-Metal project (branch `fable/int8-metal`) implemented all three kernels GPU-side
  with the dequantize epilogue fused (scales+bias+activation in one launch) and changed
  nothing in this scheme — evidence the per-row/no-zero-point contract is backend-portable.
  Kernel specifics: `apple-silicon` skill.
