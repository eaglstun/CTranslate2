#include "metal/primitives.h"
#include "metal/device.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace ctranslate2 {
  namespace metal {

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

        id<MTLCommandBuffer> command_buffer = new_command_buffer();
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

        commit_command_buffer(command_buffer);
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

    // Must match CT2_SDPA_SG (SIMD-groups of 32 lanes) in kernels_msl.h.
    static constexpr NSUInteger kSdpaThreadgroup = 4 * 32;

    namespace {
      void sdpa_impl(const char* pipeline_name,
                     const void* queries,
                     const void* keys,
                     const void* values,
                     const int32_t* lengths,
                     void* output,
                     dim_t num_rows,
                     dim_t rows_per_bh,
                     dim_t num_keys,
                     dim_t kv_stride,
                     dim_t depth,
                     float scale) {
        if (num_rows == 0 || depth == 0)
          return;

        const BufferRange q_buffer = buffer_and_offset(queries);
        const BufferRange k_buffer = buffer_and_offset(keys);
        const BufferRange v_buffer = buffer_and_offset(values);
        const BufferRange out_buffer = buffer_and_offset(output);
        const uint32_t has_lengths = lengths ? 1u : 0u;
        // The lengths index must always be bound; reuse the queries buffer as a never-read
        // dummy when there is no lengths array (same convention as softmax).
        const BufferRange len_buffer = lengths ? buffer_and_offset(lengths) : q_buffer;

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);

        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:q_buffer.buffer offset:q_buffer.offset atIndex:0];
        [encoder setBuffer:k_buffer.buffer offset:k_buffer.offset atIndex:1];
        [encoder setBuffer:v_buffer.buffer offset:v_buffer.offset atIndex:2];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:3];
        [encoder setBuffer:len_buffer.buffer offset:len_buffer.offset atIndex:4];
        const uint32_t num_keys_u = static_cast<uint32_t>(num_keys);
        const uint32_t kv_stride_u = static_cast<uint32_t>(kv_stride);
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        const uint32_t rows_per_bh_u = static_cast<uint32_t>(rows_per_bh);
        [encoder setBytes:&num_keys_u length:sizeof(num_keys_u) atIndex:5];
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:6];
        [encoder setBytes:&rows_per_bh_u length:sizeof(rows_per_bh_u) atIndex:7];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:8];
        [encoder setBytes:&has_lengths length:sizeof(has_lengths) atIndex:9];
        [encoder setBytes:&kv_stride_u length:sizeof(kv_stride_u) atIndex:10];

        // One threadgroup per score row; the kernel assumes exactly CT2_SDPA_SG SIMD-groups.
        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(num_rows), 1, 1);
        const MTLSize group = MTLSizeMake(kSdpaThreadgroup, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];

        commit_command_buffer(command_buffer);
      }
    }

    void sdpa(const float* queries, const float* keys, const float* values,
              const int32_t* lengths, float* output,
              dim_t num_rows, dim_t rows_per_bh, dim_t num_keys, dim_t kv_stride,
              dim_t depth, float scale) {
      sdpa_impl("ct2_sdpa_float", queries, keys, values, lengths, output,
                num_rows, rows_per_bh, num_keys, kv_stride, depth, scale);
    }

    void sdpa(const float16_t* queries, const float16_t* keys, const float16_t* values,
              const int32_t* lengths, float16_t* output,
              dim_t num_rows, dim_t rows_per_bh, dim_t num_keys, dim_t kv_stride,
              dim_t depth, float scale) {
      sdpa_impl("ct2_sdpa_half", queries, keys, values, lengths, output,
                num_rows, rows_per_bh, num_keys, kv_stride, depth, scale);
    }

    namespace {
      void kv_cache_dispatch(const char* pipeline_name,
                             const void* old_keys, const void* old_values,
                             const void* keys, const void* values,
                             void* cached_keys, void* cached_values,
                             dim_t batch_heads, dim_t cache_length, dim_t append_length,
                             dim_t old_capacity, dim_t new_capacity, dim_t row_size_bytes,
                             bool grow) {
        if (batch_heads == 0 || row_size_bytes == 0 || (!grow && append_length == 0)
            || (grow && cache_length + append_length == 0))
          return;

        const BufferRange old_k = grow ? buffer_and_offset(old_keys)
                                       : buffer_and_offset(cached_keys);
        const BufferRange old_v = grow ? buffer_and_offset(old_values)
                                       : buffer_and_offset(cached_values);
        const BufferRange new_k = append_length ? buffer_and_offset(keys) : old_k;
        const BufferRange new_v = append_length ? buffer_and_offset(values) : old_v;
        const BufferRange dst_k = buffer_and_offset(cached_keys);
        const BufferRange dst_v = buffer_and_offset(cached_values);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:old_k.buffer offset:old_k.offset atIndex:0];
        [encoder setBuffer:old_v.buffer offset:old_v.offset atIndex:1];
        [encoder setBuffer:new_k.buffer offset:new_k.offset atIndex:2];
        [encoder setBuffer:new_v.buffer offset:new_v.offset atIndex:3];
        [encoder setBuffer:dst_k.buffer offset:dst_k.offset atIndex:4];
        [encoder setBuffer:dst_v.buffer offset:dst_v.offset atIndex:5];
        const uint32_t cache_length_u = static_cast<uint32_t>(cache_length);
        const uint32_t append_length_u = static_cast<uint32_t>(append_length);
        const uint32_t old_capacity_u = static_cast<uint32_t>(old_capacity);
        const uint32_t new_capacity_u = static_cast<uint32_t>(new_capacity);
        const uint32_t row_size_u = static_cast<uint32_t>(row_size_bytes);
        [encoder setBytes:&cache_length_u length:sizeof(cache_length_u) atIndex:6];
        [encoder setBytes:&append_length_u length:sizeof(append_length_u) atIndex:7];
        [encoder setBytes:&old_capacity_u length:sizeof(old_capacity_u) atIndex:8];
        [encoder setBytes:&new_capacity_u length:sizeof(new_capacity_u) atIndex:9];
        [encoder setBytes:&row_size_u length:sizeof(row_size_u) atIndex:10];

        const dim_t rows = grow ? cache_length + append_length : append_length;
        const NSUInteger bytes = static_cast<NSUInteger>(batch_heads * rows * row_size_bytes);
        NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, bytes);
        [encoder dispatchThreads:MTLSizeMake(bytes, 2, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }
    }

    void kv_cache_append(const void* keys, const void* values,
                         void* cached_keys, void* cached_values,
                         dim_t batch_heads, dim_t cache_length, dim_t append_length,
                         dim_t capacity, dim_t row_size_bytes) {
      kv_cache_dispatch("ct2_kv_cache_append_bytes", nullptr, nullptr, keys, values,
                        cached_keys, cached_values, batch_heads, cache_length, append_length,
                        capacity, capacity, row_size_bytes, false);
    }

    void kv_cache_grow(const void* old_keys, const void* old_values,
                       const void* keys, const void* values,
                       void* cached_keys, void* cached_values,
                       dim_t batch_heads, dim_t cache_length, dim_t append_length,
                       dim_t old_capacity, dim_t new_capacity, dim_t row_size_bytes) {
      kv_cache_dispatch("ct2_kv_cache_grow_bytes", old_keys, old_values, keys, values,
                        cached_keys, cached_values, batch_heads, cache_length, append_length,
                        old_capacity, new_capacity, row_size_bytes, true);
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
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
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
        commit_command_buffer(command_buffer);
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
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
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
        commit_command_buffer(command_buffer);
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

    namespace {
      void add_rms_norm_impl(const char* pipeline_name,
                             const void* a, const void* b, const void* gamma,
                             void* sum_out, void* normed_out,
                             dim_t batch_size, dim_t depth, float epsilon, bool use_residual) {
        if (batch_size == 0 || depth == 0)
          return;

        const BufferRange a_buffer = buffer_and_offset(a);
        const BufferRange b_buffer = buffer_and_offset(b);
        const BufferRange gamma_buffer = buffer_and_offset(gamma);
        const BufferRange sum_buffer = buffer_and_offset(sum_out);
        const BufferRange normed_buffer = buffer_and_offset(normed_out);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:a_buffer.buffer offset:a_buffer.offset atIndex:0];
        [encoder setBuffer:b_buffer.buffer offset:b_buffer.offset atIndex:1];
        [encoder setBuffer:gamma_buffer.buffer offset:gamma_buffer.offset atIndex:2];
        [encoder setBuffer:sum_buffer.buffer offset:sum_buffer.offset atIndex:3];
        [encoder setBuffer:normed_buffer.buffer offset:normed_buffer.offset atIndex:4];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        const uint32_t residual_u = use_residual ? 1u : 0u;
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:5];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:6];
        [encoder setBytes:&residual_u length:sizeof(residual_u) atIndex:7];

        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
        const MTLSize group = MTLSizeMake(kNormThreadgroup, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }
    }

    void add_rms_norm(const float* a, const float* b, const float* gamma,
                      float* sum_out, float* normed_out,
                      dim_t batch_size, dim_t depth, float epsilon, bool use_residual) {
      add_rms_norm_impl("ct2_add_rms_norm_float", a, b, gamma, sum_out, normed_out,
                        batch_size, depth, epsilon, use_residual);
    }

    void add_rms_norm(const float16_t* a, const float16_t* b, const float16_t* gamma,
                      float16_t* sum_out, float16_t* normed_out,
                      dim_t batch_size, dim_t depth, float epsilon, bool use_residual) {
      add_rms_norm_impl("ct2_add_rms_norm_half", a, b, gamma, sum_out, normed_out,
                        batch_size, depth, epsilon, use_residual);
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
      void add_layer_norm_impl(const char* pipeline_name,
                               const void* a, const void* b, const void* gamma, const void* beta,
                               void* sum_out, void* normed_out,
                               dim_t batch_size, dim_t depth, float epsilon) {
        if (batch_size == 0 || depth == 0)
          return;

        const BufferRange a_buffer = buffer_and_offset(a);
        const BufferRange b_buffer = buffer_and_offset(b);
        const BufferRange gamma_buffer = buffer_and_offset(gamma);
        const BufferRange beta_buffer = buffer_and_offset(beta);
        const BufferRange sum_buffer = buffer_and_offset(sum_out);
        const BufferRange normed_buffer = buffer_and_offset(normed_out);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:a_buffer.buffer offset:a_buffer.offset atIndex:0];
        [encoder setBuffer:b_buffer.buffer offset:b_buffer.offset atIndex:1];
        [encoder setBuffer:gamma_buffer.buffer offset:gamma_buffer.offset atIndex:2];
        [encoder setBuffer:beta_buffer.buffer offset:beta_buffer.offset atIndex:3];
        [encoder setBuffer:sum_buffer.buffer offset:sum_buffer.offset atIndex:4];
        [encoder setBuffer:normed_buffer.buffer offset:normed_buffer.offset atIndex:5];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:6];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:7];

        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
        const MTLSize group = MTLSizeMake(kNormThreadgroup, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }
    }

    void add_layer_norm(const float* a, const float* b, const float* gamma, const float* beta,
                        float* sum_out, float* normed_out,
                        dim_t batch_size, dim_t depth, float epsilon) {
      add_layer_norm_impl("ct2_add_layer_norm_float", a, b, gamma, beta, sum_out, normed_out,
                          batch_size, depth, epsilon);
    }

    void add_layer_norm(const float16_t* a, const float16_t* b, const float16_t* gamma,
                        const float16_t* beta, float16_t* sum_out, float16_t* normed_out,
                        dim_t batch_size, dim_t depth, float epsilon) {
      add_layer_norm_impl("ct2_add_layer_norm_half", a, b, gamma, beta, sum_out, normed_out,
                          batch_size, depth, epsilon);
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
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
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
        commit_command_buffer(command_buffer);
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
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
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
        commit_command_buffer(command_buffer);
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

    namespace {
      void activation_impl(const char* pipeline_name, const void* input, void* output,
                           dim_t size, int act) {
        if (size == 0)
          return;
        const BufferRange in_buffer = buffer_and_offset(input);
        const BufferRange out_buffer = buffer_and_offset(output);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:in_buffer.buffer offset:in_buffer.offset atIndex:0];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:1];
        const int32_t act32 = act;
        [encoder setBytes:&act32 length:sizeof(act32) atIndex:2];

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > (NSUInteger)size) tg = size;
        [encoder dispatchThreads:MTLSizeMake(size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }

      // Elementwise binary op with an optional host scalar second operand. Operation-
      // agnostic: the named pipeline (ct2_mul_*, ct2_add_*) selects add vs. multiply.
      void binary_elementwise_impl(const char* pipeline_name, const void* a, const void* b,
                                   void* c, dim_t size, bool b_is_scalar, float scalar) {
        if (size == 0)
          return;
        const BufferRange a_buffer = buffer_and_offset(a);
        const BufferRange c_buffer = buffer_and_offset(c);
        // For the scalar case b may be null / off-device; bind a as an unused dummy.
        const BufferRange b_buffer = b_is_scalar ? a_buffer : buffer_and_offset(b);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:a_buffer.buffer offset:a_buffer.offset atIndex:0];
        [encoder setBuffer:b_buffer.buffer offset:b_buffer.offset atIndex:1];
        [encoder setBuffer:c_buffer.buffer offset:c_buffer.offset atIndex:2];
        const uint32_t scalar_flag = b_is_scalar ? 1u : 0u;
        [encoder setBytes:&scalar_flag length:sizeof(scalar_flag) atIndex:3];
        [encoder setBytes:&scalar length:sizeof(scalar) atIndex:4];

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > (NSUInteger)size) tg = size;
        [encoder dispatchThreads:MTLSizeMake(size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }
    }

    void activation(const float* input, float* output, dim_t size, int act) {
      activation_impl("ct2_activation_float", input, output, size, act);
    }

    void activation(const float16_t* input, float16_t* output, dim_t size, int act) {
      activation_impl("ct2_activation_half", input, output, size, act);
    }

    void mul(const float* a, const float* b, float* c, dim_t size, bool b_is_scalar, float scalar) {
      binary_elementwise_impl("ct2_mul_float", a, b, c, size, b_is_scalar, scalar);
    }

    void mul(const float16_t* a, const float16_t* b, float16_t* c, dim_t size,
             bool b_is_scalar, float scalar) {
      binary_elementwise_impl("ct2_mul_half", a, b, c, size, b_is_scalar, scalar);
    }

    void add(const float* a, const float* b, float* c, dim_t size, bool b_is_scalar, float scalar) {
      binary_elementwise_impl("ct2_add_float", a, b, c, size, b_is_scalar, scalar);
    }

    void add(const float16_t* a, const float16_t* b, float16_t* c, dim_t size,
             bool b_is_scalar, float scalar) {
      binary_elementwise_impl("ct2_add_half", a, b, c, size, b_is_scalar, scalar);
    }

    // Must match CT2_QUANT_TG in kernels_msl.h.
    static constexpr NSUInteger kQuantizeThreadgroup = 256;

    namespace {
      void quantize_s8_impl(const char* pipeline_name,
                            const void* input, int8_t* output, float* scales,
                            dim_t batch_size, dim_t depth, bool round_before_cast) {
        if (batch_size == 0 || depth == 0)
          return;

        const BufferRange in_buffer = buffer_and_offset(input);
        const BufferRange out_buffer = buffer_and_offset(output);
        const BufferRange scales_buffer = buffer_and_offset(scales);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:in_buffer.buffer offset:in_buffer.offset atIndex:0];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:1];
        [encoder setBuffer:scales_buffer.buffer offset:scales_buffer.offset atIndex:2];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        const uint32_t round_u = round_before_cast ? 1u : 0u;
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:3];
        [encoder setBytes:&round_u length:sizeof(round_u) atIndex:4];

        // One threadgroup per row; the kernel's tree reduction assumes CT2_QUANT_TG threads.
        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
        const MTLSize group = MTLSizeMake(kQuantizeThreadgroup, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }

      void dequantize_s8_impl(const char* pipeline_name,
                              const int8_t* input, const float* scales, void* output,
                              dim_t batch_size, dim_t depth) {
        const dim_t total = batch_size * depth;
        if (total == 0)
          return;

        const BufferRange in_buffer = buffer_and_offset(input);
        const BufferRange scales_buffer = buffer_and_offset(scales);
        const BufferRange out_buffer = buffer_and_offset(output);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:in_buffer.buffer offset:in_buffer.offset atIndex:0];
        [encoder setBuffer:scales_buffer.buffer offset:scales_buffer.offset atIndex:1];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:2];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:3];

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > (NSUInteger)total) tg = total;
        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }

      void dequantize_gemm_output_s8_impl(const char* pipeline_name,
                                          const int32_t* c, const float* a_scale,
                                          const float* b_scale, const void* bias, void* y,
                                          dim_t batch_size, dim_t depth, int activation) {
        const dim_t total = batch_size * depth;
        if (total == 0)
          return;

        const BufferRange c_buffer = buffer_and_offset(c);
        const BufferRange a_scale_buffer = buffer_and_offset(a_scale);
        const BufferRange b_scale_buffer = buffer_and_offset(b_scale);
        const BufferRange y_buffer = buffer_and_offset(y);
        const uint32_t has_bias = bias ? 1u : 0u;
        // Index 3 must always be bound; reuse c as a never-read dummy when there is no bias.
        const BufferRange bias_buffer = bias ? buffer_and_offset(bias) : c_buffer;

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:c_buffer.buffer offset:c_buffer.offset atIndex:0];
        [encoder setBuffer:a_scale_buffer.buffer offset:a_scale_buffer.offset atIndex:1];
        [encoder setBuffer:b_scale_buffer.buffer offset:b_scale_buffer.offset atIndex:2];
        [encoder setBuffer:bias_buffer.buffer offset:bias_buffer.offset atIndex:3];
        [encoder setBuffer:y_buffer.buffer offset:y_buffer.offset atIndex:4];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        const int32_t act = activation;
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:5];
        [encoder setBytes:&has_bias length:sizeof(has_bias) atIndex:6];
        [encoder setBytes:&act length:sizeof(act) atIndex:7];

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > (NSUInteger)total) tg = total;
        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }

    }

    void quantize_s8(const float* input, int8_t* output, float* scales,
                     dim_t batch_size, dim_t depth, bool round_before_cast) {
      quantize_s8_impl("ct2_quantize_s8_float", input, output, scales,
                       batch_size, depth, round_before_cast);
    }

    void quantize_s8(const float16_t* input, int8_t* output, float* scales,
                     dim_t batch_size, dim_t depth, bool round_before_cast) {
      quantize_s8_impl("ct2_quantize_s8_half", input, output, scales,
                       batch_size, depth, round_before_cast);
    }

    void dequantize_s8(const int8_t* input, const float* scales, float* output,
                       dim_t batch_size, dim_t depth) {
      dequantize_s8_impl("ct2_dequantize_s8_float", input, scales, output, batch_size, depth);
    }

    void dequantize_s8(const int8_t* input, const float* scales, float16_t* output,
                       dim_t batch_size, dim_t depth) {
      dequantize_s8_impl("ct2_dequantize_s8_half", input, scales, output, batch_size, depth);
    }

    void dequantize_gemm_output_s8(const int32_t* c, const float* a_scale, const float* b_scale,
                                   const float* bias, float* y,
                                   dim_t batch_size, dim_t depth, int activation) {
      dequantize_gemm_output_s8_impl("ct2_dequant_gemm_out_float", c, a_scale, b_scale,
                                     bias, y, batch_size, depth, activation);
    }

    void dequantize_gemm_output_s8(const int32_t* c, const float* a_scale, const float* b_scale,
                                   const float16_t* bias, float16_t* y,
                                   dim_t batch_size, dim_t depth, int activation) {
      dequantize_gemm_output_s8_impl("ct2_dequant_gemm_out_half", c, a_scale, b_scale,
                                     bias, y, batch_size, depth, activation);
    }

    // Tile sizes must match CT2_GEMM_S8_BM/BN in kernels_msl.h (16x16 threads per group).
    static constexpr NSUInteger kGemmS8TileM = 64;
    static constexpr NSUInteger kGemmS8TileN = 64;

    void gemm_s8(bool transpose_a, bool transpose_b,
                 dim_t m, dim_t n, dim_t k,
                 int32_t alpha,
                 const int8_t* a, dim_t lda,
                 const int8_t* b, dim_t ldb,
                 int32_t* c, dim_t ldc) {
      if (m == 0 || n == 0)
        return;

      const BufferRange a_buffer = buffer_and_offset(a);
      const BufferRange b_buffer = buffer_and_offset(b);
      const BufferRange c_buffer = buffer_and_offset(c);

      // Small-m fast path (decode runs every Dense at m = batch): one SIMD-group per
      // output element instead of a mostly-empty 64x64 tile. Dense layout only, and the
      // kernel reads char4, so k and both operand offsets/strides must be 4-byte aligned;
      // anything else takes the MPP or tiled path below (correct, just slower at m <= 8).
      if (!transpose_a && transpose_b && m <= 8
          && k % 4 == 0 && lda % 4 == 0 && ldb % 4 == 0
          && a_buffer.offset % 4 == 0 && b_buffer.offset % 4 == 0) {
        id<MTLComputePipelineState> pso = get_pipeline("ct2_gemv_s8");
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:a_buffer.buffer offset:a_buffer.offset atIndex:0];
        [encoder setBuffer:b_buffer.buffer offset:b_buffer.offset atIndex:1];
        [encoder setBuffer:c_buffer.buffer offset:c_buffer.offset atIndex:2];
        const uint32_t n_u = static_cast<uint32_t>(n);
        const uint32_t k_u = static_cast<uint32_t>(k);
        const uint32_t lda_u = static_cast<uint32_t>(lda);
        const uint32_t ldb_u = static_cast<uint32_t>(ldb);
        const uint32_t ldc_u = static_cast<uint32_t>(ldc);
        const int32_t alpha_i = alpha;
        [encoder setBytes:&n_u length:sizeof(n_u) atIndex:3];
        [encoder setBytes:&k_u length:sizeof(k_u) atIndex:4];
        [encoder setBytes:&lda_u length:sizeof(lda_u) atIndex:5];
        [encoder setBytes:&ldb_u length:sizeof(ldb_u) atIndex:6];
        [encoder setBytes:&ldc_u length:sizeof(ldc_u) atIndex:7];
        [encoder setBytes:&alpha_i length:sizeof(alpha_i) atIndex:8];

        // 8 SIMD-groups per threadgroup, each producing one output element; the kernel's
        // j = tg.x * 8 + simd_group mapping assumes exactly this shape.
        const NSUInteger simd_width = pso.threadExecutionWidth;
        const MTLSize grid = MTLSizeMake((static_cast<NSUInteger>(n) + 7) / 8,
                                         static_cast<NSUInteger>(m), 1);
        const MTLSize group = MTLSizeMake(simd_width * 8, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
        return;
      }

      // Prefill / large-m fast path: Metal-4 MPP matmul2d (int8 x int8 -> int32),
      // measured int32-exact and ~4.8x faster than the tiled kernel at 2048^3 on M4 Max
      // (matches MPS fp16 throughput; see METAL_BENCHMARKS.md). Dense layout and
      // alpha == 1 only; anything else (other transposes, integral alpha != 1, k == 0)
      // takes the general tiled kernel below. get_pipeline_mpp returns nil on
      // pre-macOS-26 OSes or unsupported GPUs. CT2_NO_MPP_GEMM forces the tiled kernel
      // (bisection lever, same spirit as CT2_NO_MPS_ACT).
      static const bool mpp_disabled = std::getenv("CT2_NO_MPP_GEMM") != nullptr;
      if (!mpp_disabled && !transpose_a && transpose_b && alpha == 1 && k > 0) {
        if (id<MTLComputePipelineState> pso = get_pipeline_mpp("ct2_mpp_gemm_s8_nt")) {
          id<MTLCommandBuffer> command_buffer = new_command_buffer();
          id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
          [encoder setComputePipelineState:pso];
          [encoder setBuffer:a_buffer.buffer offset:a_buffer.offset atIndex:0];
          [encoder setBuffer:b_buffer.buffer offset:b_buffer.offset atIndex:1];
          [encoder setBuffer:c_buffer.buffer offset:c_buffer.offset atIndex:2];
          const uint32_t m_u = static_cast<uint32_t>(m);
          const uint32_t n_u = static_cast<uint32_t>(n);
          const uint32_t k_u = static_cast<uint32_t>(k);
          const uint32_t lda_u = static_cast<uint32_t>(lda);
          const uint32_t ldb_u = static_cast<uint32_t>(ldb);
          const uint32_t ldc_u = static_cast<uint32_t>(ldc);
          [encoder setBytes:&m_u length:sizeof(m_u) atIndex:3];
          [encoder setBytes:&n_u length:sizeof(n_u) atIndex:4];
          [encoder setBytes:&k_u length:sizeof(k_u) atIndex:5];
          [encoder setBytes:&lda_u length:sizeof(lda_u) atIndex:6];
          [encoder setBytes:&ldb_u length:sizeof(ldb_u) atIndex:7];
          [encoder setBytes:&ldc_u length:sizeof(ldc_u) atIndex:8];

          // Tile/scope shape must match CT2_MPP_GEMM_S8_{TM,TN,SG} in kernels_mpp_msl.h.
          const MTLSize grid = MTLSizeMake((static_cast<NSUInteger>(n) + 63) / 64,
                                           (static_cast<NSUInteger>(m) + 15) / 16,
                                           1);
          const MTLSize group = MTLSizeMake(pso.threadExecutionWidth * 2, 1, 1);
          [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
          [encoder endEncoding];
          commit_command_buffer(command_buffer);
          return;
        }
      }

      id<MTLComputePipelineState> pso = get_pipeline("ct2_gemm_s8");
      id<MTLCommandBuffer> command_buffer = new_command_buffer();
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:a_buffer.buffer offset:a_buffer.offset atIndex:0];
      [encoder setBuffer:b_buffer.buffer offset:b_buffer.offset atIndex:1];
      [encoder setBuffer:c_buffer.buffer offset:c_buffer.offset atIndex:2];
      const uint32_t m_u = static_cast<uint32_t>(m);
      const uint32_t n_u = static_cast<uint32_t>(n);
      const uint32_t k_u = static_cast<uint32_t>(k);
      const uint32_t lda_u = static_cast<uint32_t>(lda);
      const uint32_t ldb_u = static_cast<uint32_t>(ldb);
      const uint32_t ldc_u = static_cast<uint32_t>(ldc);
      const uint32_t trans_a_u = transpose_a ? 1u : 0u;
      const uint32_t trans_b_u = transpose_b ? 1u : 0u;
      const int32_t alpha_i = alpha;
      [encoder setBytes:&m_u length:sizeof(m_u) atIndex:3];
      [encoder setBytes:&n_u length:sizeof(n_u) atIndex:4];
      [encoder setBytes:&k_u length:sizeof(k_u) atIndex:5];
      [encoder setBytes:&lda_u length:sizeof(lda_u) atIndex:6];
      [encoder setBytes:&ldb_u length:sizeof(ldb_u) atIndex:7];
      [encoder setBytes:&ldc_u length:sizeof(ldc_u) atIndex:8];
      [encoder setBytes:&trans_a_u length:sizeof(trans_a_u) atIndex:9];
      [encoder setBytes:&trans_b_u length:sizeof(trans_b_u) atIndex:10];
      [encoder setBytes:&alpha_i length:sizeof(alpha_i) atIndex:11];

      // One 16x16 threadgroup per 64x64 output tile; the kernel zero-pads edge loads, so
      // the rounded-up grid is safe (only the C store is guarded).
      const MTLSize grid = MTLSizeMake((static_cast<NSUInteger>(n) + kGemmS8TileN - 1) / kGemmS8TileN,
                                       (static_cast<NSUInteger>(m) + kGemmS8TileM - 1) / kGemmS8TileM,
                                       1);
      const MTLSize group = MTLSizeMake(16, 16, 1);
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
      [encoder endEncoding];
      commit_command_buffer(command_buffer);
    }

    void strided_copy(const void* src, void* dst,
                      dim_t copy_size_bytes, dim_t src_step_bytes, dim_t dst_step_bytes,
                      dim_t iter_size) {
      if (iter_size == 0 || copy_size_bytes == 0)
        return;

      const BufferRange src_buffer = buffer_and_offset(src);
      const BufferRange dst_buffer = buffer_and_offset(dst);
      const dim_t total = iter_size * copy_size_bytes;

      id<MTLComputePipelineState> pso = get_pipeline("ct2_strided_copy_bytes");
      id<MTLCommandBuffer> command_buffer = new_command_buffer();
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:src_buffer.buffer offset:src_buffer.offset atIndex:0];
      [encoder setBuffer:dst_buffer.buffer offset:dst_buffer.offset atIndex:1];
      const uint32_t copy_u = static_cast<uint32_t>(copy_size_bytes);
      const uint32_t src_step_u = static_cast<uint32_t>(src_step_bytes);
      const uint32_t dst_step_u = static_cast<uint32_t>(dst_step_bytes);
      [encoder setBytes:&copy_u length:sizeof(copy_u) atIndex:2];
      [encoder setBytes:&src_step_u length:sizeof(src_step_u) atIndex:3];
      [encoder setBytes:&dst_step_u length:sizeof(dst_step_u) atIndex:4];

      NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
      if (tg > (NSUInteger)total) tg = total;
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [encoder endEncoding];
      commit_command_buffer(command_buffer);
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
      id<MTLCommandBuffer> command_buffer = new_command_buffer();
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

      // 2D grid: x = 16-byte chunks within a row (CT2_GATHER_CHUNK in the kernel),
      // y = gathered rows. Rows can be megabytes (KV-cache reorder), so the row copy
      // must be parallel too.
      const NSUInteger chunks = (static_cast<NSUInteger>(copy_size_bytes) + 15) / 16;
      NSUInteger tg_x = pso.maxTotalThreadsPerThreadgroup;
      if (tg_x > chunks) tg_x = chunks;
      [encoder dispatchThreads:MTLSizeMake(chunks, num_indices, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
      [encoder endEncoding];
      commit_command_buffer(command_buffer);
    }

    void transpose(const void* x, void* y,
                   const dim_t* out_dims, const dim_t* in_strides, dim_t item_size) {
      const dim_t total = out_dims[0] * out_dims[1] * out_dims[2] * out_dims[3];
      if (total == 0)
        return;

      const char* pipeline_name = nullptr;
      switch (item_size) {
      case 1: pipeline_name = "ct2_transpose_b1"; break;
      case 2: pipeline_name = "ct2_transpose_b2"; break;
      case 4: pipeline_name = "ct2_transpose_b4"; break;
      default:
        throw std::invalid_argument("metal::transpose: unsupported item size "
                                    + std::to_string(item_size));
      }

      const BufferRange x_buffer = buffer_and_offset(x);
      const BufferRange y_buffer = buffer_and_offset(y);

      id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
      id<MTLCommandBuffer> command_buffer = new_command_buffer();
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:x_buffer.buffer offset:x_buffer.offset atIndex:0];
      [encoder setBuffer:y_buffer.buffer offset:y_buffer.offset atIndex:1];
      uint32_t dims_u[4];
      uint32_t strides_u[4];
      for (int i = 0; i < 4; ++i) {
        dims_u[i] = static_cast<uint32_t>(out_dims[i]);
        strides_u[i] = static_cast<uint32_t>(in_strides[i]);
      }
      [encoder setBytes:dims_u length:sizeof(dims_u) atIndex:2];
      [encoder setBytes:strides_u length:sizeof(strides_u) atIndex:3];

      NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
      if (tg > (NSUInteger)total) tg = total;
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [encoder endEncoding];
      commit_command_buffer(command_buffer);
    }

    // Must match CT2_TOPK_MAX_TG in kernels_msl.h; the kernel reads the actual
    // threadgroup size at runtime, so the host clamps to the pipeline's limit (rounded
    // down to a power of two for the tree reduction).
    static constexpr NSUInteger kTopKMaxThreadgroup = 1024;

    dim_t topk_max_k() {
      // Each selected element costs a full pass over the row; cap so a degenerate k
      // cannot turn the kernel into an O(depth^2) scan. Decode-time k (beam size,
      // sampling_topk) is far below this.
      return 256;
    }

    namespace {
      void topk_impl(const char* pipeline_name,
                     const void* x, void* values, int32_t* indices,
                     dim_t batch_size, dim_t depth, dim_t k) {
        if (batch_size == 0 || depth == 0 || k == 0)
          return;

        const BufferRange x_buffer = buffer_and_offset(x);
        const BufferRange v_buffer = buffer_and_offset(values);
        const BufferRange i_buffer = buffer_and_offset(indices);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:x_buffer.buffer offset:x_buffer.offset atIndex:0];
        [encoder setBuffer:v_buffer.buffer offset:v_buffer.offset atIndex:1];
        [encoder setBuffer:i_buffer.buffer offset:i_buffer.offset atIndex:2];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        const uint32_t k_u = static_cast<uint32_t>(k);
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:3];
        [encoder setBytes:&k_u length:sizeof(k_u) atIndex:4];

        NSUInteger tg = std::min(kTopKMaxThreadgroup, pso.maxTotalThreadsPerThreadgroup);
        while (tg & (tg - 1))  // round down to a power of two
          tg &= tg - 1;
        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
        const MTLSize group = MTLSizeMake(tg, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }
    }

    void topk(const float* x, float* values, int32_t* indices,
              dim_t batch_size, dim_t depth, dim_t k) {
      topk_impl("ct2_topk_float", x, values, indices, batch_size, depth, k);
    }

    void topk(const float16_t* x, float16_t* values, int32_t* indices,
              dim_t batch_size, dim_t depth, dim_t k) {
      topk_impl("ct2_topk_half", x, values, indices, batch_size, depth, k);
    }

    // Must match CT2_TOPP_TG / CT2_TOPP_MAX_DEPTH in kernels_msl.h.
    static constexpr NSUInteger kTopPThreadgroup = 256;
    static constexpr dim_t kTopPMaxDepth = 4096;

    dim_t topp_mask_max_depth() {
      return kTopPMaxDepth;
    }

    namespace {
      void topp_mask_impl(const char* pipeline_name,
                          const void* x, const void* probs, void* y,
                          dim_t batch_size, dim_t depth, float p, float mask_value) {
        if (batch_size == 0 || depth == 0)
          return;

        const BufferRange x_buffer = buffer_and_offset(x);
        const BufferRange probs_buffer = buffer_and_offset(probs);
        const BufferRange y_buffer = buffer_and_offset(y);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:x_buffer.buffer offset:x_buffer.offset atIndex:0];
        [encoder setBuffer:probs_buffer.buffer offset:probs_buffer.offset atIndex:1];
        [encoder setBuffer:y_buffer.buffer offset:y_buffer.offset atIndex:2];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:3];
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        [encoder setBytes:&mask_value length:sizeof(mask_value) atIndex:5];

        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
        const MTLSize group = MTLSizeMake(kTopPThreadgroup, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }
    }

    void topp_mask(const float* x, const float* probs, float* y,
                   dim_t batch_size, dim_t depth, float p, float mask_value) {
      topp_mask_impl("ct2_topp_mask_float", x, probs, y, batch_size, depth, p, mask_value);
    }

    void topp_mask(const float16_t* x, const float16_t* probs, float16_t* y,
                   dim_t batch_size, dim_t depth, float p, float mask_value) {
      topp_mask_impl("ct2_topp_mask_half", x, probs, y, batch_size, depth, p, mask_value);
    }

    // Must match CT2_MULTINOMIAL_TG in kernels_msl.h.
    static constexpr NSUInteger kMultinomialThreadgroup = 256;

    dim_t multinomial_max_batch_size() {
      // The host-drawn uniforms are passed via setBytes (4 KB limit).
      return 512;
    }

    namespace {
      void multinomial_impl(const char* pipeline_name,
                            const void* probs, const float* uniforms, int32_t* output,
                            dim_t batch_size, dim_t depth) {
        if (batch_size == 0 || depth == 0)
          return;

        const BufferRange probs_buffer = buffer_and_offset(probs);
        const BufferRange out_buffer = buffer_and_offset(output);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:probs_buffer.buffer offset:probs_buffer.offset atIndex:0];
        [encoder setBytes:uniforms length:batch_size * sizeof(float) atIndex:1];
        [encoder setBuffer:out_buffer.buffer offset:out_buffer.offset atIndex:2];
        const uint32_t depth_u = static_cast<uint32_t>(depth);
        [encoder setBytes:&depth_u length:sizeof(depth_u) atIndex:3];

        const MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1);
        const MTLSize group = MTLSizeMake(kMultinomialThreadgroup, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }
    }

    void multinomial(const float* probs, const float* uniforms, int32_t* output,
                     dim_t batch_size, dim_t depth) {
      multinomial_impl("ct2_multinomial_float", probs, uniforms, output, batch_size, depth);
    }

    void multinomial(const float16_t* probs, const float* uniforms, int32_t* output,
                     dim_t batch_size, dim_t depth) {
      multinomial_impl("ct2_multinomial_half", probs, uniforms, output, batch_size, depth);
    }

    namespace {
      void add_gumbel_noise_impl(const char* pipeline_name,
                                 const void* x, void* y, dim_t size, uint64_t seed) {
        if (size == 0)
          return;

        const BufferRange x_buffer = buffer_and_offset(x);
        const BufferRange y_buffer = buffer_and_offset(y);

        id<MTLComputePipelineState> pso = get_pipeline(pipeline_name);
        id<MTLCommandBuffer> command_buffer = new_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:x_buffer.buffer offset:x_buffer.offset atIndex:0];
        [encoder setBuffer:y_buffer.buffer offset:y_buffer.offset atIndex:1];
        [encoder setBytes:&seed length:sizeof(seed) atIndex:2];

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > (NSUInteger)size) tg = size;
        [encoder dispatchThreads:MTLSizeMake(size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        commit_command_buffer(command_buffer);
      }
    }

    void add_gumbel_noise(const float* x, float* y, dim_t size, uint64_t seed) {
      add_gumbel_noise_impl("ct2_gumbel_noise_float", x, y, size, seed);
    }

    void add_gumbel_noise(const float16_t* x, float16_t* y, dim_t size, uint64_t seed) {
      add_gumbel_noise_impl("ct2_gumbel_noise_half", x, y, size, seed);
    }

  }
}
