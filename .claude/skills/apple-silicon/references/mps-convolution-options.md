---
title: "GPU options for Conv1D — MPSCNNConvolution vs MPSGraph conv2d vs custom MSL"
summary: >-
  Scoping card for graduating CT2's Conv1D to the GPU, comparing three
  options: MPSCNNConvolution (MPSImage feature maps with an
  MPSCNNConvolutionDataSource callback and OHWI weight layout — poor fit
  because CT2 activations are row-major MTLBuffers), MPSGraph convolution2D
  (buffer-native rank-4 tensors with declared dataLayout/weightsLayout, the
  most natural MPS fit but drags in the whole-graph framework), and a custom
  MSL kernel (the op-graduation-playbook path, most control, removes the
  fp16-fp32 round-trip). Current state: Conv1D runs on the CPU reference on
  Metal with fp16 upcast to fp32, and conv weights stay float via the
  src/models/model.cc is_conv guard. The consumer is the Whisper/Wav2Vec2 two-
  Conv1D encoder stem (per-encode, not per-token); the decision rule is
  prototype the custom MSL kernel first.
semantic_id: "yhwQ5guuv22S4e11SskLvAcoN2uTiq-nCpcnmoSl4t6qj3gG9--tXU-6epFj768TTCos29Ty2Y5OvI7_jrZupA"
---

Sources: https://developer.apple.com/documentation/metalperformanceshaders/mpscnnconvolution,
https://developer.apple.com/documentation/metalperformanceshaders/mpscnnconvolutiondatasource,
https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphconvolution2dopdescriptor
(fetched via DocC JSON, 2026-06-11; weight-layout prose from the SDK header
`MPSNeuralNetwork.framework/Headers/MPSCNNConvolution.h` where DocC is stripped).

Scoping card for graduating CT2's `Conv1D` to the GPU. Current state (verified
2026-06-11): `Conv1D` runs on the **CPU reference** on Metal; fp16 inputs upcast to
fp32, run on CPU, downcast back (`src/ops/conv1d.cc`, Metal fp16 branch ~lines 51–69);
conv weights are **kept in `float_dtype`** on `Device::METAL` even in int8 models — the
`src/models/model.cc` is_conv guard, same as CUDA/DNNL, added after the downstream
harness caught Whisper crashing on int8 conv weights (METAL_BACKEND.md). The consumer is
the Whisper/Wav2Vec2 conv stem: Whisper's encoder leads with exactly two `Conv1D` layers
(`_conv1` stride 1, `_conv2` stride 2 — `src/layers/whisper.cc`), so this is per-encode
cost, not per-token.

## Option A — MPSCNNConvolution (macOS 10.13+)

- Classic MPS CNN kernel over **MPSImage** feature maps, not MTLBuffers. Configured by
  `MPSCNNConvolutionDescriptor` (`kernelWidth/Height`, `inputFeatureChannels`,
  `outputFeatureChannels`, `strideInPixelsX/Y`, `dilationRateX/Y`, `groups`, fusable
  neuron/activation).
- Weights come from an **`MPSCNNConvolutionDataSource`** protocol object (`-load`,
  `-weights`, `-biasTerms`, `-dataType`, `-purge`) — a callback contract, not a buffer
  handoff. Layout (header): `weight[outputChannels][kernelHeight][kernelWidth][inputChannels/groups]`,
  float32, "converted to half float (fp16) internally for best performance"; biases are
  float32, one per output channel.
- The 1-D fit is awkward twice over: CT2 activations live in row-major MTLBuffers, so
  every call needs buffer→MPSImage→buffer packaging (texture-backed, channel-quartet
  layout), and a conv1d must pose as a height-1 image. Verdict: wrong shape for this
  backend; listed for completeness.

## Option B — MPSGraph convolution2D (macOS 11+)

- `convolution2D(_:weights:descriptor:name:)`: source and weights are **rank-4 tensors**;
  `MPSGraphConvolution2DOpDescriptor` carries `strideInX/Y`, `dilationRateInX/Y`,
  `groups`, explicit or styled padding, and — the key part — **`dataLayout` and
  `weightsLayout` properties**, so NCHW-style tensors are declared, not physically
  repacked.
- Buffer-native: feed via `MPSGraphTensorData(MTLBuffer:shape:dataType:)` over the
  existing Metal allocations; conv1d maps cleanly to conv2d with height 1
  (input `[N, C_in, 1, L]`, weight `[C_out, C_in, 1, K]` in a declared layout).
- Cost: it drags in the whole-graph framework — per-shape compiled
  `MPSGraphExecutable` caching and `encode(to:)` integration, with the build/compile
  overhead and overlap caveats in `mpsgraph-for-inference.md`. For two convs per encode
  (not per decode step), a cached executable amortizes well; this is the most natural
  MPS-provided fit.

## Option C — custom MSL kernel

- The op-graduation playbook path (`op-graduation-playbook.md`): MSL kernel in
  `kernels_msl.h`, `metal::` entry point, `Device::METAL` routing in
  `Conv1D::operator()` before generic dispatch, parity via the existing suite (note the
  two currently-skipped Metal conv tests: dilation and one group variant —
  METAL_BACKEND.md).
- Whisper's shapes are tame: kernel 3, padding 1, strides 1 and 2, channels
  mel→d_model — an im2col-free direct kernel (one threadgroup per output row, fp32
  accumulate, fused bias+GELU like `ct2_dequant_gemm_out`) is small and removes the
  fp16↔fp32 round-trip outright. Cost: it's hand-written code competing with cuDNN-class
  kernels — but at 2 calls per 30s audio window, simplicity beats peak throughput.

## Honest comparison

|                   | data layout                 | weights                         | fp16         | effort                      | fit                |
| ----------------- | --------------------------- | ------------------------------- | ------------ | --------------------------- | ------------------ |
| MPSCNNConvolution | MPSImage (repack both ways) | DataSource callback, OHWI float | internal     | medium                      | poor               |
| MPSGraph conv2D   | MTLBuffer, declared layouts | graph constant/placeholder      | native dtype | medium (framework adoption) | good               |
| Custom MSL        | CT2's buffers as-is         | CT2's float conv weights as-is  | native       | medium (one kernel)         | good, most control |

### Relevance to the CT2 Metal backend

- `Conv1D` is the highest-value op not yet on the GPU: METAL_BACKEND.md lists it under
  "Deferred" ("CUDA path uses cuDNN; needed for Whisper / Wav2Vec2 encoders"), and the
  Whisper bringup made it the one remaining CPU stage at the head of every fp16 encode
  (`src/ops/conv1d.cc` upcast path). Sampling aside, everything around it already runs
  on Metal.
- Whatever option wins, the quantization contract stays: conv weights remain float on
  Metal (`src/models/model.cc` guard) — none of the three options changes that; a
  quantized conv would be a separate, currently-unsupported project.
- Decision rule: prototype C (custom MSL) first — it respects the backend's per-op
  command-buffer model (`storage-and-synchronization.md`) with zero new dependencies;
  hold B in reserve if dilation/groups coverage grows past what a simple kernel wants to
  own. Then benchmark Whisper encode end-to-end before/after per
  `benchmarking-and-profiling.md`, and write the numbers into METAL_BENCHMARKS.md.
