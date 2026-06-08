#pragma once

#include <cstddef>

// Pure C++ entry points for Metal compute kernels. Safe to include from .cc files and
// tests. During the initial bring-up these are called directly (not through the generic
// DEVICE_DISPATCH path), so the signatures are concrete rather than templated.

namespace ctranslate2 {
  namespace metal {

    // Element-wise c[i] = a[i] + b[i] over `size` floats. The pointers must be data()
    // pointers of StorageViews allocated on Device::METAL (i.e. tracked by the Metal
    // allocator). The call is synchronous: it returns once the GPU work has completed.
    void add(const float* a, const float* b, float* c, size_t size);

  }
}
