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

    namespace {
      void rotary_impl(const char* pipeline_name,
                       const void* input, const void* sin, const void* cos, void* output,
                       dim_t batch_size, dim_t max_time, dim_t ndims, dim_t depth,
                       bool interleave) {
        const dim_t total = batch_size * max_time * depth;
        if (total == 0)
          return;

        const BufferRange in_buffer = buffer_and_offset(input);
        const BufferRange sin_buffer = buffer_and_offset(sin);
        const BufferRange cos_buffer = buffer_and_offset(cos);
        const BufferRange out_buffer = buffer_and_offset(output);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:in_buffer.buffer offset:in_buffer.offset atIndex:0];
        [encoder setBuffer:sin_buffer.buffer offset:sin_buffer.offset atIndex:1];
        [encoder setBuffer:cos_buffer.buffer offset:cos_buffer.offset atIndex:2];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:3];
        const uint32_t max_time_u = static_cast<uint32_t>(max_time);
        const uint32_t ndims_u = static_cast<uint32_t>(ndims);
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        const uint32_t interleave_u = interleave ? 1u : 0u;
        [encoder setBytes:&max_time_u length:sizeof(max_time_u) atIndex:4];
        [encoder setBytes:&ndims_u length:sizeof(ndims_u) atIndex:5];
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:6];
        [encoder setBytes:&interleave_u length:sizeof(interleave_u) atIndex:7];

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > (NSUInteger)total) tg = total;
        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
      }
    }

    void rotary(const float* input, const float* sin, const float* cos, float* output,
                dim_t batch_size, dim_t max_time, dim_t ndims, dim_t depth, bool interleave) {
      rotary_impl("ct2_rotary_float", input, sin, cos, output,
                  batch_size, max_time, ndims, depth, interleave);
    }

    void rotary(const float16_t* input, const float16_t* sin, const float16_t* cos,
                float16_t* output, dim_t batch_size, dim_t max_time, dim_t ndims, dim_t depth,
                bool interleave) {
      rotary_impl("ct2_rotary_half", input, sin, cos, output,
                  batch_size, max_time, ndims, depth, interleave);
    }

    namespace {
      void bias_add_impl(const char* pipeline_name,
                         const void* value, const void* bias, const void* residual,
                         void* output, dim_t size, dim_t depth, int activation) {
        if (size == 0)
          return;

        const BufferRange value_buffer = buffer_and_offset(value);
        const BufferRange bias_buffer = buffer_and_offset(bias);
        const BufferRange out_buffer = buffer_and_offset(output);
        const uint32_t has_residual = residual ? 1u : 0u;
        // Index 2 must always be bound; reuse value as a never-read dummy when no residual.
        const BufferRange res_buffer = residual ? buffer_and_offset(residual) : value_buffer;

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:value_buffer.buffer offset:value_buffer.offset atIndex:0];
        [encoder setBuffer:bias_buffer.buffer offset:bias_buffer.offset atIndex:1];
        [encoder setBuffer:res_buffer.buffer offset:res_buffer.offset atIndex:2];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:3];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        const int32_t act = activation;
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:4];
        [encoder setBytes:&has_residual length:sizeof(has_residual) atIndex:5];
        [encoder setBytes:&act length:sizeof(act) atIndex:6];

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > (NSUInteger)size) tg = size;
        [encoder dispatchThreads:MTLSizeMake(size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
      }
    }

    void bias_add(const float* value, const float* bias, const float* residual,
                  float* output, dim_t size, dim_t depth, int activation) {
      bias_add_impl("ct2_bias_add_float", value, bias, residual, output, size, depth, activation);
    }

    void bias_add(const float16_t* value, const float16_t* bias, const float16_t* residual,
                  float16_t* output, dim_t size, dim_t depth, int activation) {
      bias_add_impl("ct2_bias_add_half", value, bias, residual, output, size, depth, activation);
    }

    void gather(const void* data, const int32_t* indices, void* output,
                dim_t copy_size_bytes, dim_t batch_stride_bytes,
                dim_t num_indices, dim_t num_indices_per_batch) {
      if (num_indices == 0 || copy_size_bytes == 0)
        return;

      const BufferRange data_buffer = buffer_and_offset(data);
      const BufferRange idx_buffer = buffer_and_offset(indices);
      const BufferRange out_buffer = buffer_and_offset(output);

      id<MTLComputePipelineState> pso = get_pipeline("ct2_gather_bytes");
      id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:data_buffer.buffer offset:data_buffer.offset atIndex:0];
      [encoder setBuffer:idx_buffer.buffer offset:idx_buffer.offset atIndex:1];
      [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:2];
      const uint32_t copy_u = static_cast<uint32_t>(copy_size_bytes);
      const uint32_t stride_u = static_cast<uint32_t>(batch_stride_bytes);
      const uint32_t per_batch_u = static_cast<uint32_t>(num_indices_per_batch);
      [encoder setBytes:&copy_u length:sizeof(copy_u) atIndex:3];
      [encoder setBytes:&stride_u length:sizeof(stride_u) atIndex:4];
      [encoder setBytes:&per_batch_u length:sizeof(per_batch_u) atIndex:5];

      NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
      if (tg > (NSUInteger)num_indices) tg = num_indices;
      [encoder dispatchThreads:MTLSizeMake(num_indices, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }

  }
}
