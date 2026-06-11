# The Conv1D Op

The one convolution in the engine: interface, the three backend strategies
(im2col+GEMM, DNNL direct, cuDNN), its dtype story — including the quantization
exclusion and the single backend that DOES run int8 conv — and who uses it.

**Sources (read these, all citations below are from real lines):**

- `include/ctranslate2/ops/conv1d.h`, `src/ops/conv1d.cc`
- `src/ops/conv1d_cpu.cc` (two impls: DNNL vs im2col+GEMM), `src/ops/conv1d_gpu.cu`,
  `src/ops/conv1d_cudnn_gpu.cu`, `CMakeLists.txt:161-162,239,628-629`
- `src/layers/common.cc:521-552` (the layer wrapper), `src/models/model.cc:204-227` (load guard)

---

## 1. Interface

```cpp
// include/ctranslate2/ops/conv1d.h:11-26
Conv1D(dim_t stride = 1, dim_t padding = 0, dim_t dilation = 1, dim_t groups = 1,
       const ActivationType* activation_type = nullptr);
void operator()(input, weight, bias, output, const StorageView* qscale = nullptr);
void operator()(input, weight, output, const StorageView* qscale = nullptr);  // no bias
```

Shapes: input `[batch, in_channels, time]`, weight `[out_channels, in_channels/groups,
kernel]`, output `[batch, out_channels, out_time]` with `out_time = (time + 2*padding -
(dilation*(kernel-1)+1)) / stride + 1` (`src/ops/conv1d.cc:42-49`). The activation is a
ctor option fused into the epilogue, like Gemm's. The private `operator()` resizes the
output, then `DEVICE_AND_FLOAT_DISPATCH` into `compute<D, T>` (`conv1d.cc:74-75`).

## 2. CPU: im2col+GEMM by default, DNNL direct when built with it

`src/ops/conv1d_cpu.cc` is one file with two complete implementations switched by
`#ifdef CT2_WITH_DNNL` (`conv1d_cpu.cc:3,119`):

- **Default (no DNNL): im2col + GEMM.** `im2col_transposed` (`conv1d_cpu.cc:215-243`)
  unrolls windows into `[batch, groups, out_time, in_ch_per_group*kernel]` —
  **transposed** on purpose: the GEMM runs `weight × patches^T` (`Gemm(1.0, 0.0, false,
true)`, `:175`), and since `Quantize` is per-row, the patch rows line up with the
  columns being contracted, making input quantization legal (the design comment,
  `:150-159`). Per `(batch×group)` slice in a `cpu::parallel_for` (`:181-212`): with a
  `qscale`, dynamically quantize the patches, run the **int8 GEMM**, and
  `Dequantize(weight_group_scale, input_scale)` (`:198-207`); without, a plain float GEMM
  (`:209`). Bias+activation via `apply_bias_and_activation(..., axis=-2)` (`:138`).
  Limitation: `_dilation != 1` throws (`:133-134`).
- **DNNL build: direct convolution** (`dnnl::algorithm::convolution_direct`,
  `conv1d_cpu.cc:52-77`) with memory-format reorders and dilation support — but `qscale`
  throws `"Quantization is not supported in this Conv1D implementation"` (`:15-16`).

## 3. CUDA: cuDNN when available, else the same im2col+GEMM

CMake swaps the source file: `WITH_CUDNN` replaces `conv1d_gpu.cu` with
`conv1d_cudnn_gpu.cu` (`CMakeLists.txt:628-629`).

- **cuDNN path** (`src/ops/conv1d_cudnn_gpu.cu`): NCHW descriptors with height 1,
  IMPLICIT(\_PRECOMP)\_GEMM algo (`:59-62`), tensor-op math for fp16 (`:54-56`), workspace
  from the CUDA caching allocator. `qscale` throws (`:14-15`).
- **Plain CUDA path** (`src/ops/conv1d_gpu.cu`): an `im2col_transposed_kernel` (`:12-53`)
  then per-group `primitives<Device::CUDA>::gemm_batch_strided` (`:104-112`) — float
  only; the `qscale` parameter is unnamed/ignored (`:62`) and instantiations are
  float/fp16/bf16 (`:126-128`).

## 4. Dtype: the quantization exclusion, and the one int8-conv backend

The convert-time `_quantize` DOES quantize conv weights (it reshapes the 3D kernel to 2D
for per-out-channel scales, `python/ctranslate2/specs/model_spec.py:230-242`), and
`Conv1DSpec` carries `weight_scale` (`specs/common_spec.py:55-58`). What keeps conv
float at runtime is the **load-time guard** in `Model::set_compute_type`
(`src/models/model.cc:204-227`): for variables whose name contains `conv`, the target
dtype is forced to the _float_ dtype on CUDA, **Metal**, or any DNNL build, because those
backends' conv impls reject/ignore `qscale` (§2-3). Full pipeline context:
`weight-loading-and-conversion.md` (this guard is what fixed the int8-Whisper load crash,
commit 351b1990).

**Answer to "does any backend run int8 conv?": yes, exactly one** — the CPU im2col+GEMM
path in a build _without_ DNNL (`conv1d_cpu.cc:198-207`): int8 weights with per-group
sliced scales, dynamically quantized inputs, int32 accumulation, fused dequantize. The
test `Conv1DGroupNoBiasQuantized` pins this matrix down: skipped on DNNL builds and on
any non-CPU device (`tests/ops_test.cc:1320-1326`).

## 5. Users — audio front-ends only

The `layers::Conv1D` wrapper (`src/layers/common.cc:521-552`) binds `{scope}/weight`,
optional bias and `weight_scale`, and forwards `_qscale` to the op (`:546-551`); note
`output_type()` returns the **weight's** dtype (`:534-536`). Users:

- **Whisper encoder stem**: `_conv1`/`_conv2` — stride 1 then 2, padding 1, GELU fused
  (`src/layers/whisper.cc:9-12`; members `include/ctranslate2/layers/whisper.h:46-47`).
- **wav2vec2**: feature-extraction conv stack (`include/ctranslate2/layers/wav2vec2.h:29,50`).
- **wav2vec2-BERT**: pointwise/depthwise convs in the conformer blocks
  (`include/ctranslate2/layers/wav2vec2bert.h:43-48,80-83`).

No text transformer touches Conv1D — it is an audio-front-end op.

---

### Relevance to the Metal backend

- Metal has **no conv kernel**; `Device::METAL` falls into the CPU-reference dispatch
  case. The fp32-only CPU reference would reject fp16 inputs, so `conv1d.cc:51-72`
  carries a Metal-specific fp16→fp32 upcast island (run CPU reference, downcast back) —
  this is what made fp16 Whisper work end-to-end (see the Whisper-on-Metal memory).
- The load guard (§4) includes `Device::METAL` precisely because this CPU-reference path
  has `qscale == nullptr` for the float stems — int8 conv weights would otherwise load
  int8 and crash/silently mismatch.
- Graduation options (MPSCNNConvolution vs MPSGraph vs hand-rolled im2col reusing the
  Metal GEMM) are weighed in the `apple-silicon` skill's `mps-convolution-options.md`.
- The im2col+GEMM strategy (§2) is the natural Metal port: it reduces conv to the GEMM
  the backend already owns — including, in principle, the int8 path.
