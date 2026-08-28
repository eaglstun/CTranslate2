#include "metal/primitives.h"
#include "metal/device.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdlib>
#include <map>
#include <tuple>

// Single- and half-precision GEMM on the GPU via Metal Performance Shaders.
//
// CTranslate2 StorageViews are row-major, and MPSMatrix is also row-major, so the
// operands map directly: a stored A of shape (rows, cols) with leading dimension lda
// becomes an MPSMatrix with rowBytes = lda * element_size. This is unlike the cuBLAS
// path, which swaps A and B to compensate for cuBLAS's column-major convention — do NOT
// replicate that swap here.

namespace ctranslate2 {
  namespace metal {

    namespace {

      MPSMatrixDescriptor* descriptor(NSUInteger rows, NSUInteger columns, dim_t ld,
                                      size_t element_size, MPSDataType data_type) {
        return [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                     columns:columns
                                                    rowBytes:static_cast<NSUInteger>(ld) * element_size
                                                    dataType:data_type];
      }

      // MPSMatrixMultiplication is configured with dims/transpose/alpha/beta at init and
      // takes the operand matrices only at encode time, so one object is reused for every
      // GEMM of the same shape — and a decoder repeats a handful of shapes every layer and
      // step. Caching them removes a per-op allocation (a measurable chunk of encode cost).
      // Thread-local: an MPS kernel object must not be encoded from two threads at once.
      using GemmKey = std::tuple<dim_t, dim_t, dim_t, bool, bool, float, float, int>;

      MPSMatrixMultiplication* cached_gemm(dim_t m, dim_t n, dim_t k,
                                           bool transpose_a, bool transpose_b,
                                           float alpha, float beta) {
        static thread_local std::map<GemmKey, MPSMatrixMultiplication*> cache;
        const GemmKey key{m, n, k, transpose_a, transpose_b, alpha, beta, 0};
        auto it = cache.find(key);
        if (it != cache.end())
          return it->second;
        MPSMatrixMultiplication* mm =
          [[MPSMatrixMultiplication alloc] initWithDevice:get_metal_device()
                                            transposeLeft:transpose_a
                                           transposeRight:transpose_b
                                               resultRows:m
                                            resultColumns:n
                                          interiorColumns:k
                                                    alpha:alpha
                                                     beta:beta];
        cache.emplace(key, mm);
        return mm;
      }

      using GemvKey = std::tuple<dim_t, dim_t, bool, float, float>;

      MPSMatrixVectorMultiplication* cached_gemv(dim_t rows, dim_t columns,
                                                 bool transpose,
                                                 float alpha, float beta) {
        static thread_local std::map<GemvKey, MPSMatrixVectorMultiplication*> cache;
        const GemvKey key{rows, columns, transpose, alpha, beta};
        auto it = cache.find(key);
        if (it != cache.end())
          return it->second;
        MPSMatrixVectorMultiplication* mv =
          [[MPSMatrixVectorMultiplication alloc] initWithDevice:get_metal_device()
                                                     transpose:transpose
                                                          rows:rows
                                                       columns:columns
                                                         alpha:alpha
                                                          beta:beta];
        cache.emplace(key, mv);
        return mv;
      }

      // Encodes one C = alpha * op(A) * op(B) + beta * C into the command buffer.
      void encode_gemm(id<MTLCommandBuffer> command_buffer,
                       bool transpose_a, bool transpose_b,
                       dim_t m, dim_t n, dim_t k,
                       float alpha, float beta,
                       id<MTLBuffer> a, NSUInteger a_offset, dim_t lda,
                       id<MTLBuffer> b, NSUInteger b_offset, dim_t ldb,
                       id<MTLBuffer> c, NSUInteger c_offset, dim_t ldc,
                       size_t element_size, MPSDataType data_type) {
        // Dimensions of the stored (pre-transpose) matrices.
        const NSUInteger a_rows = transpose_a ? k : m;
        const NSUInteger a_cols = transpose_a ? m : k;
        const NSUInteger b_rows = transpose_b ? n : k;
        const NSUInteger b_cols = transpose_b ? k : n;

        MPSMatrix* a_matrix = [[MPSMatrix alloc] initWithBuffer:a offset:a_offset
                                 descriptor:descriptor(a_rows, a_cols, lda, element_size, data_type)];
        MPSMatrix* b_matrix = [[MPSMatrix alloc] initWithBuffer:b offset:b_offset
                                 descriptor:descriptor(b_rows, b_cols, ldb, element_size, data_type)];
        MPSMatrix* c_matrix = [[MPSMatrix alloc] initWithBuffer:c offset:c_offset
                                 descriptor:descriptor(m, n, ldc, element_size, data_type)];

        MPSMatrixMultiplication* mm = cached_gemm(m, n, k, transpose_a, transpose_b, alpha, beta);
        [mm encodeToCommandBuffer:command_buffer
                       leftMatrix:a_matrix
                      rightMatrix:b_matrix
                     resultMatrix:c_matrix];

        [a_matrix release];
        [b_matrix release];
        [c_matrix release];
        // mm is owned by the thread-local cache; do not release.
      }

      // Encodes y = alpha * op(matrix) * x + beta * y. rows/columns describe op(matrix),
      // so the stored matrix dimensions are reversed when transpose is true.
      void encode_gemv(id<MTLCommandBuffer> command_buffer,
                       bool transpose,
                       dim_t rows, dim_t columns,
                       float alpha, float beta,
                       id<MTLBuffer> matrix, NSUInteger matrix_offset, dim_t ldm,
                       id<MTLBuffer> x, NSUInteger x_offset,
                       id<MTLBuffer> y, NSUInteger y_offset,
                       size_t element_size, MPSDataType data_type) {
        const NSUInteger stored_rows = transpose ? columns : rows;
        const NSUInteger stored_columns = transpose ? rows : columns;
        MPSMatrix* matrix_view =
          [[MPSMatrix alloc] initWithBuffer:matrix offset:matrix_offset
                                descriptor:descriptor(stored_rows, stored_columns, ldm,
                                                      element_size, data_type)];
        MPSVectorDescriptor* x_descriptor =
          [MPSVectorDescriptor vectorDescriptorWithLength:columns dataType:data_type];
        MPSVectorDescriptor* y_descriptor =
          [MPSVectorDescriptor vectorDescriptorWithLength:rows dataType:data_type];
        MPSVector* x_vector = [[MPSVector alloc] initWithBuffer:x
                                                        offset:x_offset
                                                    descriptor:x_descriptor];
        MPSVector* y_vector = [[MPSVector alloc] initWithBuffer:y
                                                        offset:y_offset
                                                    descriptor:y_descriptor];

        MPSMatrixVectorMultiplication* mv = cached_gemv(rows, columns, transpose,
                                                        alpha, beta);
        [mv encodeToCommandBuffer:command_buffer
                      inputMatrix:matrix_view
                      inputVector:x_vector
                     resultVector:y_vector];

        [matrix_view release];
        [x_vector release];
        [y_vector release];
        // mv is owned by the thread-local cache; descriptors are autoreleased inside the
        // per-op pool opened by new_command_buffer().
      }

      template <typename T>
      void gemv_impl(bool transpose,
                     dim_t rows, dim_t columns,
                     float alpha,
                     const T* matrix, dim_t ldm,
                     const T* x,
                     float beta,
                     T* y,
                     size_t element_size, MPSDataType data_type) {
        const BufferRange matrix_buffer = buffer_and_offset(matrix);
        const BufferRange x_buffer = buffer_and_offset(x);
        const BufferRange y_buffer = buffer_and_offset(y);

        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        encode_gemv(command_buffer, transpose, rows, columns, alpha, beta,
                    matrix_buffer.buffer, matrix_buffer.offset, ldm,
                    x_buffer.buffer, x_buffer.offset,
                    y_buffer.buffer, y_buffer.offset,
                    element_size, data_type);
        commit_command_buffer(command_buffer);
      }

      template <typename T>
      void gemm_impl(bool transpose_a, bool transpose_b,
                     dim_t m, dim_t n, dim_t k,
                     float alpha,
                     const T* a, dim_t lda,
                     const T* b, dim_t ldb,
                     float beta,
                     T* c, dim_t ldc,
                     size_t element_size, MPSDataType data_type) {
        const BufferRange a_buffer = buffer_and_offset(a);
        const BufferRange b_buffer = buffer_and_offset(b);
        const BufferRange c_buffer = buffer_and_offset(c);

        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        // Experimental Dense decode route. For A[1,k] * op(B)[k,n], transpose the
        // equation to C^T = op(B)^T * A^T and feed stored B as the GEMV matrix. Dense's
        // common trans_b=true layout stores B as [n,k], so GEMV needs no transpose.
        static const bool use_mps_gemv = std::getenv("CT2_MPS_GEMV") != nullptr;
        if (use_mps_gemv && m == 1 && !transpose_a && lda == k && ldc == n) {
          encode_gemv(command_buffer,
                      /*transpose=*/!transpose_b,
                      /*rows=*/n,
                      /*columns=*/k,
                      alpha, beta,
                      b_buffer.buffer, b_buffer.offset, ldb,
                      a_buffer.buffer, a_buffer.offset,
                      c_buffer.buffer, c_buffer.offset,
                      element_size, data_type);
        } else {
          encode_gemm(command_buffer, transpose_a, transpose_b, m, n, k, alpha, beta,
                      a_buffer.buffer, a_buffer.offset, lda,
                      b_buffer.buffer, b_buffer.offset, ldb,
                      c_buffer.buffer, c_buffer.offset, ldc,
                      element_size, data_type);
        }
        commit_command_buffer(command_buffer);
      }

      template <typename T>
      void gemm_batch_strided_impl(bool transpose_a, bool transpose_b,
                                   dim_t m, dim_t n, dim_t k,
                                   float alpha,
                                   const T* a, dim_t lda, dim_t stridea,
                                   const T* b, dim_t ldb, dim_t strideb,
                                   float beta,
                                   T* c, dim_t ldc, dim_t stridec,
                                   dim_t batch_size,
                                   size_t element_size, MPSDataType data_type) {
        const BufferRange a_buffer = buffer_and_offset(a);
        const BufferRange b_buffer = buffer_and_offset(b);
        const BufferRange c_buffer = buffer_and_offset(c);

        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        for (dim_t i = 0; i < batch_size; ++i) {
          encode_gemm(command_buffer, transpose_a, transpose_b, m, n, k, alpha, beta,
                      a_buffer.buffer, a_buffer.offset + i * stridea * element_size, lda,
                      b_buffer.buffer, b_buffer.offset + i * strideb * element_size, ldb,
                      c_buffer.buffer, c_buffer.offset + i * stridec * element_size, ldc,
                      element_size, data_type);
        }
        commit_command_buffer(command_buffer);
      }

    }

    void gemm(bool transpose_a, bool transpose_b,
              dim_t m, dim_t n, dim_t k,
              float alpha,
              const float* a, dim_t lda,
              const float* b, dim_t ldb,
              float beta,
              float* c, dim_t ldc) {
      gemm_impl<float>(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       sizeof(float), MPSDataTypeFloat32);
    }

    void gemv(bool transpose,
              dim_t rows, dim_t columns,
              float alpha,
              const float* matrix, dim_t ldm,
              const float* x,
              float beta,
              float* y) {
      gemv_impl<float>(transpose, rows, columns, alpha, matrix, ldm, x, beta, y,
                       sizeof(float), MPSDataTypeFloat32);
    }

    void gemm(bool transpose_a, bool transpose_b,
              dim_t m, dim_t n, dim_t k,
              float alpha,
              const float16_t* a, dim_t lda,
              const float16_t* b, dim_t ldb,
              float beta,
              float16_t* c, dim_t ldc) {
      gemm_impl<float16_t>(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                           sizeof(float16_t), MPSDataTypeFloat16);
    }

    void gemv(bool transpose,
              dim_t rows, dim_t columns,
              float alpha,
              const float16_t* matrix, dim_t ldm,
              const float16_t* x,
              float beta,
              float16_t* y) {
      gemv_impl<float16_t>(transpose, rows, columns, alpha, matrix, ldm, x, beta, y,
                           sizeof(float16_t), MPSDataTypeFloat16);
    }

    void gemm_batch_strided(bool transpose_a, bool transpose_b,
                            dim_t m, dim_t n, dim_t k,
                            float alpha,
                            const float* a, dim_t lda, dim_t stridea,
                            const float* b, dim_t ldb, dim_t strideb,
                            float beta,
                            float* c, dim_t ldc, dim_t stridec,
                            dim_t batch_size) {
      gemm_batch_strided_impl<float>(transpose_a, transpose_b, m, n, k, alpha,
                                     a, lda, stridea, b, ldb, strideb, beta, c, ldc, stridec,
                                     batch_size, sizeof(float), MPSDataTypeFloat32);
    }

    void gemm_batch_strided(bool transpose_a, bool transpose_b,
                            dim_t m, dim_t n, dim_t k,
                            float alpha,
                            const float16_t* a, dim_t lda, dim_t stridea,
                            const float16_t* b, dim_t ldb, dim_t strideb,
                            float beta,
                            float16_t* c, dim_t ldc, dim_t stridec,
                            dim_t batch_size) {
      gemm_batch_strided_impl<float16_t>(transpose_a, transpose_b, m, n, k, alpha,
                                         a, lda, stridea, b, ldb, strideb, beta, c, ldc, stridec,
                                         batch_size, sizeof(float16_t), MPSDataTypeFloat16);
    }

  }
}
