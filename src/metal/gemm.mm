#include "metal/primitives.h"
#include "metal/device.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Single-precision GEMM on the GPU via Metal Performance Shaders.
//
// CTranslate2 StorageViews are row-major, and MPSMatrix is also row-major, so the
// operands map directly: a stored A of shape (rows, cols) with leading dimension lda
// becomes an MPSMatrix with rowBytes = lda * sizeof(float). This is unlike the cuBLAS
// path, which swaps A and B to compensate for cuBLAS's column-major convention — do NOT
// replicate that swap here.

namespace ctranslate2 {
  namespace metal {

    namespace {

      MPSMatrixDescriptor* descriptor(NSUInteger rows, NSUInteger columns, dim_t ld) {
        return [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                     columns:columns
                                                    rowBytes:static_cast<NSUInteger>(ld) * sizeof(float)
                                                    dataType:MPSDataTypeFloat32];
      }

      // Encodes one C = alpha * op(A) * op(B) + beta * C into the command buffer.
      void encode_gemm(id<MTLCommandBuffer> command_buffer,
                       bool transpose_a, bool transpose_b,
                       dim_t m, dim_t n, dim_t k,
                       float alpha, float beta,
                       id<MTLBuffer> a, NSUInteger a_offset, dim_t lda,
                       id<MTLBuffer> b, NSUInteger b_offset, dim_t ldb,
                       id<MTLBuffer> c, NSUInteger c_offset, dim_t ldc) {
        // Dimensions of the stored (pre-transpose) matrices.
        const NSUInteger a_rows = transpose_a ? k : m;
        const NSUInteger a_cols = transpose_a ? m : k;
        const NSUInteger b_rows = transpose_b ? n : k;
        const NSUInteger b_cols = transpose_b ? k : n;

        MPSMatrix* a_matrix = [[MPSMatrix alloc] initWithBuffer:a offset:a_offset
                                                     descriptor:descriptor(a_rows, a_cols, lda)];
        MPSMatrix* b_matrix = [[MPSMatrix alloc] initWithBuffer:b offset:b_offset
                                                     descriptor:descriptor(b_rows, b_cols, ldb)];
        MPSMatrix* c_matrix = [[MPSMatrix alloc] initWithBuffer:c offset:c_offset
                                                     descriptor:descriptor(m, n, ldc)];

        MPSMatrixMultiplication* mm =
          [[MPSMatrixMultiplication alloc] initWithDevice:get_metal_device()
                                            transposeLeft:transpose_a
                                           transposeRight:transpose_b
                                               resultRows:m
                                            resultColumns:n
                                          interiorColumns:k
                                                    alpha:alpha
                                                     beta:beta];
        [mm encodeToCommandBuffer:command_buffer
                       leftMatrix:a_matrix
                      rightMatrix:b_matrix
                     resultMatrix:c_matrix];

        [a_matrix release];
        [b_matrix release];
        [c_matrix release];
        [mm release];
      }

    }

    void gemm(bool transpose_a, bool transpose_b,
              dim_t m, dim_t n, dim_t k,
              float alpha,
              const float* a, dim_t lda,
              const float* b, dim_t ldb,
              float beta,
              float* c, dim_t ldc) {
      const BufferRange a_buffer = buffer_and_offset(a);
      const BufferRange b_buffer = buffer_and_offset(b);
      const BufferRange c_buffer = buffer_and_offset(c);

      id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
      encode_gemm(command_buffer, transpose_a, transpose_b, m, n, k, alpha, beta,
                  a_buffer.buffer, a_buffer.offset, lda,
                  b_buffer.buffer, b_buffer.offset, ldb,
                  c_buffer.buffer, c_buffer.offset, ldc);
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }

    void gemm_batch_strided(bool transpose_a, bool transpose_b,
                            dim_t m, dim_t n, dim_t k,
                            float alpha,
                            const float* a, dim_t lda, dim_t stridea,
                            const float* b, dim_t ldb, dim_t strideb,
                            float beta,
                            float* c, dim_t ldc, dim_t stridec,
                            dim_t batch_size) {
      const BufferRange a_buffer = buffer_and_offset(a);
      const BufferRange b_buffer = buffer_and_offset(b);
      const BufferRange c_buffer = buffer_and_offset(c);

      id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
      for (dim_t i = 0; i < batch_size; ++i) {
        encode_gemm(command_buffer, transpose_a, transpose_b, m, n, k, alpha, beta,
                    a_buffer.buffer, a_buffer.offset + i * stridea * sizeof(float), lda,
                    b_buffer.buffer, b_buffer.offset + i * strideb * sizeof(float), ldb,
                    c_buffer.buffer, c_buffer.offset + i * stridec * sizeof(float), ldc);
      }
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }

  }
}
