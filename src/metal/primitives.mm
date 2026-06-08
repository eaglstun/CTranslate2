#include "metal/primitives.h"
#include "metal/device.h"

#include <cstdint>

namespace ctranslate2 {
  namespace metal {

    void add(const float* a, const float* b, float* c, size_t size) {
      if (size == 0)
        return;

      const BufferRange a_buffer = buffer_and_offset(a);
      const BufferRange b_buffer = buffer_and_offset(b);
      const BufferRange c_buffer = buffer_and_offset(c);

      id<MTLComputePipelineState> pso = get_pipeline("ct2_add_float");

      id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:a_buffer.buffer offset:a_buffer.offset atIndex:0];
      [encoder setBuffer:b_buffer.buffer offset:b_buffer.offset atIndex:1];
      [encoder setBuffer:c_buffer.buffer offset:c_buffer.offset atIndex:2];

      const uint32_t n = static_cast<uint32_t>(size);
      [encoder setBytes:&n length:sizeof(n) atIndex:3];

      NSUInteger threads_per_group = pso.maxTotalThreadsPerThreadgroup;
      if (threads_per_group > size)
        threads_per_group = size;

      // Apple Silicon GPUs support non-uniform threadgroup sizes, so dispatchThreads
      // handles a grid that is not a multiple of the threadgroup size.
      const MTLSize grid = MTLSizeMake(size, 1, 1);
      const MTLSize group = MTLSizeMake(threads_per_group, 1, 1);
      [encoder dispatchThreads:grid threadsPerThreadgroup:group];
      [encoder endEncoding];

      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }

    // Must match CT2_SOFTMAX_TG in kernels_msl.h.
    static constexpr NSUInteger kSoftmaxThreadgroup = 256;

    namespace {
      void softmax_impl(const char* pipeline_name,
                        bool log,
                        const void* input,
                        const int32_t* lengths,
                        void* output,
                        dim_t batch_size,
                        dim_t depth) {
        if (batch_size == 0 || depth == 0)
          return;

        const BufferRange in_buffer = buffer_and_offset(input);
        const BufferRange out_buffer = buffer_and_offset(output);
        const uint32_t has_lengths = lengths ? 1u : 0u;
        // Index 2 must always be bound; reuse the input buffer as a never-read dummy when
        // there is no lengths array.
        const BufferRange len_buffer = lengths ? buffer_and_offset(lengths) : in_buffer;

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);

        id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:in_buffer.buffer offset:in_buffer.offset atIndex:0];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:1];
        [encoder setBuffer:len_buffer.buffer offset:len_buffer.offset atIndex:2];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        const uint32_t is_log = log ? 1u : 0u;
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:3];
        [encoder setBytes:&has_lengths length:sizeof(has_lengths) atIndex:4];
        [encoder setBytes:&is_log length:sizeof(is_log) atIndex:5];

        // One threadgroup per row, a fixed power-of-two number of threads per group (the
        // kernel's tree reduction assumes the threadgroup size is CT2_SOFTMAX_TG).
        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
        const MTLSize group = MTLSizeMake(kSoftmaxThreadgroup, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
      }
    }

    void softmax(bool log,
                 const float* input,
                 const int32_t* lengths,
                 float* output,
                 dim_t batch_size,
                 dim_t depth) {
      softmax_impl("ct2_softmax_float", log, input, lengths, output, batch_size, depth);
    }

    void softmax(bool log,
                 const float16_t* input,
                 const int32_t* lengths,
                 float16_t* output,
                 dim_t batch_size,
                 dim_t depth) {
      softmax_impl("ct2_softmax_half", log, input, lengths, output, batch_size, depth);
    }

    // Must match CT2_NORM_TG in kernels_msl.h.
    static constexpr NSUInteger kNormThreadgroup = 256;

    namespace {
      void rms_norm_impl(const char* pipeline_name,
                         const void* input, const void* gamma, void* output,
                         dim_t batch_size, dim_t depth, float epsilon, bool use_residual) {
        if (batch_size == 0 || depth == 0)
          return;

        const BufferRange in_buffer = buffer_and_offset(input);
        const BufferRange gamma_buffer = buffer_and_offset(gamma);
        const BufferRange out_buffer = buffer_and_offset(output);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:in_buffer.buffer offset:in_buffer.offset atIndex:0];
        [encoder setBuffer:gamma_buffer.buffer offset:gamma_buffer.offset atIndex:1];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:2];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        const uint32_t residual_u = use_residual ? 1u : 0u;
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder setBytes:&residual_u length:sizeof(residual_u) atIndex:5];

        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
        const MTLSize group = MTLSizeMake(kNormThreadgroup, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
      }

      void layer_norm_impl(const char* pipeline_name,
                           const void* input, const void* gamma, const void* beta, void* output,
                           dim_t batch_size, dim_t depth, float epsilon) {
        if (batch_size == 0 || depth == 0)
          return;

        const BufferRange in_buffer = buffer_and_offset(input);
        const BufferRange gamma_buffer = buffer_and_offset(gamma);
        const BufferRange beta_buffer = buffer_and_offset(beta);
        const BufferRange out_buffer = buffer_and_offset(output);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:in_buffer.buffer offset:in_buffer.offset atIndex:0];
        [encoder setBuffer:gamma_buffer.buffer offset:gamma_buffer.offset atIndex:1];
        [encoder setBuffer:beta_buffer.buffer offset:beta_buffer.offset atIndex:2];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:3];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:4];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:5];

        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
        const MTLSize group = MTLSizeMake(kNormThreadgroup, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
      }
    }

    void rms_norm(const float* input, const float* gamma, float* output,
                  dim_t batch_size, dim_t depth, float epsilon, bool use_residual) {
      rms_norm_impl("ct2_rms_norm_float", input, gamma, output, batch_size, depth, epsilon, use_residual);
    }

    void rms_norm(const float16_t* input, const float16_t* gamma, float16_t* output,
                  dim_t batch_size, dim_t depth, float epsilon, bool use_residual) {
      rms_norm_impl("ct2_rms_norm_half", input, gamma, output, batch_size, depth, epsilon, use_residual);
    }

    void layer_norm(const float* input, const float* gamma, const float* beta,
                    float* output, dim_t batch_size, dim_t depth, float epsilon) {
      layer_norm_impl("ct2_layer_norm_float", input, gamma, beta, output, batch_size, depth, epsilon);
    }

    void layer_norm(const float16_t* input, const float16_t* gamma, const float16_t* beta,
                    float16_t* output, dim_t batch_size, dim_t depth, float epsilon) {
      layer_norm_impl("ct2_layer_norm_half", input, gamma, beta, output, batch_size, depth, epsilon);
    }

  }
}
