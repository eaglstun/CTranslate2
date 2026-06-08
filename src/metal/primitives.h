#pragma once

#include <cstddef>

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

  }
}
