#include "ctranslate2/allocator.h"
#include "metal/device.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

// Objective-C++ WITHOUT ARC (manual retain/release). newBufferWithLength: returns a +1
// owned object which we store raw in the side table and [release] in free().

namespace ctranslate2 {
  namespace metal {

    // Allocator backed by shared-storage MTLBuffers. On Apple Silicon's unified memory
    // the buffer's `contents` pointer is directly CPU-addressable, so it satisfies the
    // pointer-based Allocator contract and StorageView can treat it like a host pointer
    // (data<T>()). A side table maps that pointer back to its MTLBuffer for kernel
    // dispatch and release.
    class MetalAllocator : public Allocator {
    public:
      void* allocate(size_t size, int /*device_index*/) override {
        if (size == 0)
          size = 1;  // MTLBuffer of length 0 is invalid; keep a unique tracked pointer.

        id<MTLDevice> device = get_metal_device();
        id<MTLBuffer> buffer = [device newBufferWithLength:size
                                                   options:MTLResourceStorageModeShared];
        if (!buffer)
          throw std::runtime_error("Metal: failed to allocate buffer of size "
                                   + std::to_string(size));

        void* ptr = [buffer contents];
        std::lock_guard<std::mutex> lock(_mutex);
        _buffers.emplace(ptr, buffer);
        return ptr;
      }

      void free(void* ptr, int /*device_index*/) override {
        if (!ptr)
          return;
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _buffers.find(ptr);
        if (it == _buffers.end())
          return;
        [it->second release];
        _buffers.erase(it);
      }

      id<MTLBuffer> buffer_for(const void* ptr) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _buffers.find(const_cast<void*>(ptr));
        if (it == _buffers.end())
          throw std::runtime_error("Metal: pointer is not a tracked Metal buffer");
        return it->second;
      }

    private:
      std::mutex _mutex;
      std::unordered_map<void*, id<MTLBuffer>> _buffers;
    };

    static MetalAllocator& get_metal_allocator() {
      static MetalAllocator allocator;
      return allocator;
    }

    id<MTLBuffer> buffer_for(const void* ptr) {
      return get_metal_allocator().buffer_for(ptr);
    }

  }

  template<>
  Allocator& get_allocator<Device::METAL>() {
    return metal::get_metal_allocator();
  }

}
