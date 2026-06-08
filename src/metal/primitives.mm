#include "metal/primitives.h"
#include "metal/device.h"

#include <cstdint>

namespace ctranslate2 {
  namespace metal {

    void add(const float* a, const float* b, float* c, size_t size) {
      if (size == 0)
        return;

      // The allocator returns one buffer per allocation and hands back its base
      // `contents` pointer, so each tracked pointer maps to offset 0 of its buffer.
      id<MTLBuffer> a_buffer = buffer_for(a);
      id<MTLBuffer> b_buffer = buffer_for(b);
      id<MTLBuffer> c_buffer = buffer_for(c);

      id<MTLComputePipelineState> pso = get_pipeline("ct2_add_float");

      id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:a_buffer offset:0 atIndex:0];
      [encoder setBuffer:b_buffer offset:0 atIndex:1];
      [encoder setBuffer:c_buffer offset:0 atIndex:2];

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
