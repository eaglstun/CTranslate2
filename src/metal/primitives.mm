#include "metal/primitives.h"
#include "metal/device.h"

#include <cstdint>

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

      void cast_impl(const char* pipeline_name, const void* x, void* y, dim_t size) {
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

        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > (NSUInteger)size) tg = size;
        [encoder dispatchThreads:MTLSizeMake(size, 1, 1)
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

    void s8_to_float(const int8_t* x, float* y, dim_t size) {
      cast_impl("ct2_s8_to_float", x, y, size);
    }

    void float_to_s32(const float* x, int32_t* y, dim_t size) {
      cast_impl("ct2_float_to_s32", x, y, size);
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

      NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
      if (tg > (NSUInteger)num_indices) tg = num_indices;
      [encoder dispatchThreads:MTLSizeMake(num_indices, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [encoder endEncoding];
      commit_command_buffer(command_buffer);
    }

  }
}
