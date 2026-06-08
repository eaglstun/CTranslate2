#pragma once

#include <cstddef>
#include <cstdint>

#include "ctranslate2/types.h"

// Pure C++ entry points for Metal compute kernels. Safe to include from .cc files and
// tests. During the initial bring-up these are called directly (not through the generic
// DEVICE_DISPATCH path), so the signatures are concrete rather than templated.

namespace ctranslate2 {
  namespace metal {

    // Element-wise c[i] = a[i] + b[i] over `size` floats. The pointers must be data()
    // pointers of StorageViews allocated on Device::METAL (i.e. tracked by the Metal
    // allocator). The call is synchronous: it returns once the GPU work has completed.
    void add(const float* a, const float* b, float* c, size_t size);

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

    // Row-wise layer norm with affine (gamma, beta both required) over the last
    // dimension (depth) for batch_size rows. float32 and float16.
    void layer_norm(const float* input, const float* gamma, const float* beta,
                    float* output, dim_t batch_size, dim_t depth, float epsilon);
    void layer_norm(const float16_t* input, const float16_t* gamma, const float16_t* beta,
                    float16_t* output, dim_t batch_size, dim_t depth, float epsilon);

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

    // Type-agnostic gather: output[i] = data[batch_of(i)*batch_stride + indices[i]*copy_size],
    // where copy_size and batch_stride are in BYTES. Pointers must be Metal-allocated.
    void gather(const void* data, const int32_t* indices, void* output,
                dim_t copy_size_bytes, dim_t batch_stride_bytes,
                dim_t num_indices, dim_t num_indices_per_batch);

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
