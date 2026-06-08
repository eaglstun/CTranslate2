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

  }
}
