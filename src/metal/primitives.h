#pragma once

#include <cstddef>
#include <cstdint>

#include "ctranslate2/types.h"

// Pure C++ entry points for Metal compute kernels. Safe to include from .cc files and
// tests. During the initial bring-up these are called directly (not through the generic
// DEVICE_DISPATCH path), so the signatures are concrete rather than templated.

namespace ctranslate2 {
  namespace metal {

    // Elementwise add over `size` elements: c[i] = a[i] + (b_is_scalar ? scalar : b[i]).
    // The a/c pointers must be data() pointers of StorageViews on Device::METAL; when
    // b_is_scalar, b is ignored (the scalar is read on the host and passed by value).
    void add(const float* a, const float* b, float* c, dim_t size, bool b_is_scalar, float scalar);
    void add(const float16_t* a, const float16_t* b, float16_t* c, dim_t size,
             bool b_is_scalar, float scalar);

    // Row-wise softmax over the last dimension (depth) for batch_size rows, matching
    // primitives/ops SoftMax semantics: optional per-row `lengths` masking (nullptr =
    // full depth), and `log` for log-softmax. Synchronous. float32 only.
    void softmax(bool log,
                 const float* input,
                 const int32_t* lengths,
                 float* output,
                 dim_t batch_size,
                 dim_t depth);
    void softmax(bool log,
                 const float16_t* input,
                 const int32_t* lengths,
                 float16_t* output,
                 dim_t batch_size,
                 dim_t depth);

    // Row-wise RMS norm over the last dimension (depth) for batch_size rows. gamma is
    // required; use_residual selects the (1 + gamma) variant. float32 and float16.
    void rms_norm(const float* input, const float* gamma, float* output,
                  dim_t batch_size, dim_t depth, float epsilon, bool use_residual);
    void rms_norm(const float16_t* input, const float16_t* gamma, float16_t* output,
                  dim_t batch_size, dim_t depth, float epsilon, bool use_residual);

    // Fused residual-add + RMSNorm: sum_out = a + b (the new residual stream), normed_out =
    // rms_norm(sum_out). One launch + one fewer device read pass than Add then rms_norm.
    void add_rms_norm(const float* a, const float* b, const float* gamma,
                      float* sum_out, float* normed_out,
                      dim_t batch_size, dim_t depth, float epsilon, bool use_residual);
    void add_rms_norm(const float16_t* a, const float16_t* b, const float16_t* gamma,
                      float16_t* sum_out, float16_t* normed_out,
                      dim_t batch_size, dim_t depth, float epsilon, bool use_residual);

    // Row-wise layer norm with affine (gamma, beta both required) over the last
    // dimension (depth) for batch_size rows. float32 and float16.
    void layer_norm(const float* input, const float* gamma, const float* beta,
                    float* output, dim_t batch_size, dim_t depth, float epsilon);
    void layer_norm(const float16_t* input, const float16_t* gamma, const float16_t* beta,
                    float16_t* output, dim_t batch_size, dim_t depth, float epsilon);

    // Fused residual-add + LayerNorm: sum_out = a + b, normed_out = layer_norm(sum_out).
    void add_layer_norm(const float* a, const float* b, const float* gamma, const float* beta,
                        float* sum_out, float* normed_out,
                        dim_t batch_size, dim_t depth, float epsilon);
    void add_layer_norm(const float16_t* a, const float16_t* b, const float16_t* gamma,
                        const float16_t* beta, float16_t* sum_out, float16_t* normed_out,
                        dim_t batch_size, dim_t depth, float epsilon);

    // Rotary position embedding over a [batch_size * max_time, depth] tensor; sin/cos are
    // [max_time, ndims]. Elements >= ndims are copied through. float32 and float16.
    void rotary(const float* input, const float* sin, const float* cos, float* output,
                dim_t batch_size, dim_t max_time, dim_t ndims, dim_t depth, bool interleave);
    void rotary(const float16_t* input, const float16_t* sin, const float16_t* cos,
                float16_t* output, dim_t batch_size, dim_t max_time, dim_t ndims, dim_t depth,
                bool interleave);

    // Fused bias-add + optional activation over a [.., depth] tensor of `size` elements:
    // output[i] = activation(value[i] + bias[i % depth] + (residual ? residual[i] : 0)).
    // `activation` matches the ActivationType enum; pass a negative value for none.
    // float32 and float16.
    void bias_add(const float* value, const float* bias, const float* residual,
                  float* output, dim_t size, dim_t depth, int activation);
    void bias_add(const float16_t* value, const float16_t* bias, const float16_t* residual,
                  float16_t* output, dim_t size, dim_t depth, int activation);

    // Standalone unary activation over `size` elements: output[i] = activation(input[i]).
    // `activation` matches the ActivationType enum. float32 and float16.
    void activation(const float* input, float* output, dim_t size, int act);
    void activation(const float16_t* input, float16_t* output, dim_t size, int act);

    // Elementwise multiply over `size` elements: c[i] = a[i] * (b_is_scalar ? scalar : b[i]).
    // For the scalar case the value is passed by host value (the scalar operand may live on
    // a different device) and `b` may be null. float32 and float16.
    void mul(const float* a, const float* b, float* c, dim_t size, bool b_is_scalar, float scalar);
    void mul(const float16_t* a, const float16_t* b, float16_t* c, dim_t size,
             bool b_is_scalar, float scalar);

    // Per-row symmetric int8 quantization over the last dimension, matching the CPU
    // quantize_s8 kernel: scale[row] = amax != 0 ? 127/amax : 1, output = round-to-even
    // (round_before_cast) or truncate-toward-zero of input * scale[row]. The u8-shift
    // variant is CPU-only and is not provided here. Scales are always fp32.
    void quantize_s8(const float* input, int8_t* output, float* scales,
                     dim_t batch_size, dim_t depth, bool round_before_cast);
    void quantize_s8(const float16_t* input, int8_t* output, float* scales,
                     dim_t batch_size, dim_t depth, bool round_before_cast);

    // Simple int8 dequantization: output = input * (1 / scales[row]), per-row scales.
    void dequantize_s8(const int8_t* input, const float* scales, float* output,
                       dim_t batch_size, dim_t depth);
    void dequantize_s8(const int8_t* input, const float* scales, float16_t* output,
                       dim_t batch_size, dim_t depth);

    // Quantized-GEMM output dequantization (the Dense epilogue, !trans_a && trans_b):
    // y[i][j] = act((c[i][j] / a_scale[i]) / b_scale[j] + bias[j]) over a [batch, depth]
    // int32 accumulator. `activation` matches the ActivationType enum; pass a negative
    // value for none; bias may be null.
    void dequantize_gemm_output_s8(const int32_t* c, const float* a_scale, const float* b_scale,
                                   const float* bias, float* y,
                                   dim_t batch_size, dim_t depth, int activation);
    void dequantize_gemm_output_s8(const int32_t* c, const float* a_scale, const float* b_scale,
                                   const float16_t* bias, float16_t* y,
                                   dim_t batch_size, dim_t depth, int activation);

    // Native int8 GEMM: C(int32) = alpha * op(A)(int8) * op(B)(int8), beta == 0. Row-major
    // with leading dimensions lda/ldb/ldc; op() transposes when the corresponding flag is
    // set. Accumulation is int32 throughout (bit-exact, no float detour); alpha is integral
    // because a float alpha cannot be applied exactly to an int32 accumulator — non-integral
    // alphas must not be routed here. Hand-tiled MSL kernel (MPS has no integer GEMM).
    void gemm_s8(bool transpose_a, bool transpose_b,
                 dim_t m, dim_t n, dim_t k,
                 int32_t alpha,
                 const int8_t* a, dim_t lda,
                 const int8_t* b, dim_t ldb,
                 int32_t* c, dim_t ldc);

    // Type-agnostic strided copy underlying Concat/Split/Slide: for i in [0, iter_size) and
    // d in [0, copy_size_bytes), dst[i*dst_step_bytes + d] = src[i*src_step_bytes + d]. All
    // sizes/strides are in BYTES. Pointers must be Metal-allocated.
    void strided_copy(const void* src, void* dst,
                      dim_t copy_size_bytes, dim_t src_step_bytes, dim_t dst_step_bytes,
                      dim_t iter_size);

    // Type-agnostic gather: output[i] = data[batch_of(i)*batch_stride + indices[i]*copy_size],
    // where copy_size and batch_stride are in BYTES. Pointers must be Metal-allocated.
    void gather(const void* data, const int32_t* indices, void* output,
                dim_t copy_size_bytes, dim_t batch_stride_bytes,
                dim_t num_indices, dim_t num_indices_per_batch);

    // Row-wise top-k over the last dimension: values/indices are [batch_size, k] outputs.
    // Deterministic (value descending, index ascending) order; values are bit-copies of
    // the selected input elements. Requires k <= depth and k <= topk_max_k().
    dim_t topk_max_k();
    void topk(const float* x, float* values, int32_t* indices,
              dim_t batch_size, dim_t depth, dim_t k);
    void topk(const float16_t* x, float16_t* values, int32_t* indices,
              dim_t batch_size, dim_t depth, dim_t k);

    // Row-wise top-p (nucleus) masking, mirroring the CPU TopPMask kernel exactly
    // (including the sequential float accumulation order over the descending-sorted
    // probabilities). probs must hold softmax(x). Requires depth <= topp_mask_max_depth().
    dim_t topp_mask_max_depth();
    void topp_mask(const float* x, const float* probs, float* y,
                   dim_t batch_size, dim_t depth, float p, float mask_value);
    void topp_mask(const float16_t* x, const float16_t* probs, float16_t* y,
                   dim_t batch_size, dim_t depth, float p, float mask_value);

    // One multinomial draw per row via inverse-CDF search: output is [batch_size] int32.
    // uniforms is a HOST array of batch_size values in (0, 1] (drawn by the caller from
    // the CT2 random generator); batch_size is limited to multinomial_max_batch_size()
    // because the uniforms ride in the command buffer as inline bytes.
    dim_t multinomial_max_batch_size();
    void multinomial(const float* probs, const float* uniforms, int32_t* output,
                     dim_t batch_size, dim_t depth);
    void multinomial(const float16_t* probs, const float* uniforms, int32_t* output,
                     dim_t batch_size, dim_t depth);

    // Elementwise y = x + (-log(u)), u ~ U(0, 1), the GumbelMax noise step. seed selects
    // the per-launch counter-based RNG stream (drawn by the caller from the CT2 random
    // generator so set_random_seed reproducibility holds).
    void add_gumbel_noise(const float* x, float* y, dim_t size, uint64_t seed);
    void add_gumbel_noise(const float16_t* x, float16_t* y, dim_t size, uint64_t seed);

    // Single-precision GEMM on the GPU via Metal Performance Shaders, matching the
    // semantics of primitives<D>::gemm: C = alpha * op(A) * op(B) + beta * C, where A,
    // B, C are row-major with leading dimensions lda/ldb/ldc and op() transposes when
    // the corresponding flag is set. Synchronous.
    void gemm(bool transpose_a, bool transpose_b,
              dim_t m, dim_t n, dim_t k,
              float alpha,
              const float* a, dim_t lda,
              const float* b, dim_t ldb,
              float beta,
              float* c, dim_t ldc);

    // Batched strided variant: batch_size independent m*k / k*n / m*n matrices laid out
    // contiguously with the given element strides, matching gemm_batch_strided.
    void gemm_batch_strided(bool transpose_a, bool transpose_b,
                            dim_t m, dim_t n, dim_t k,
                            float alpha,
                            const float* a, dim_t lda, dim_t stridea,
                            const float* b, dim_t ldb, dim_t strideb,
                            float beta,
                            float* c, dim_t ldc, dim_t stridec,
                            dim_t batch_size);

    // Half-precision (float16) overloads, computed via MPSDataTypeFloat16.
    void gemm(bool transpose_a, bool transpose_b,
              dim_t m, dim_t n, dim_t k,
              float alpha,
              const float16_t* a, dim_t lda,
              const float16_t* b, dim_t ldb,
              float beta,
              float16_t* c, dim_t ldc);

    void gemm_batch_strided(bool transpose_a, bool transpose_b,
                            dim_t m, dim_t n, dim_t k,
                            float alpha,
                            const float16_t* a, dim_t lda, dim_t stridea,
                            const float16_t* b, dim_t ldb, dim_t strideb,
                            float beta,
                            float16_t* c, dim_t ldc, dim_t stridec,
                            dim_t batch_size);

  }
}
