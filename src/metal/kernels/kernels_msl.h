#pragma once

// Metal Shading Language source for the backend kernels, embedded as a C++ raw string
// and compiled at runtime via newLibraryWithSource. Keeping the MSL inline avoids any
// runtime .metallib path resolution during bring-up (CTranslate2 ships a bare shared
// library, not a framework bundle, so NSBundle resource lookup is unreliable). A later
// milestone can move these kernels into a precompiled .metallib for faster startup.

namespace ctranslate2 {
  namespace metal {

    inline const char* get_kernels_source() {
      return R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void ct2_add_float(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c       [[buffer(2)]],
                          constant uint& n      [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid < n)
    c[gid] = a[gid] + b[gid];
}
)MSL";
    }

  }
}
