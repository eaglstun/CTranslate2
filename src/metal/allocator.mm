#include "ctranslate2/allocator.h"
#include "metal/device.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

// Objective-C++ WITHOUT ARC (manual retain/release). newBufferWithLength: returns a +1
// owned object which we store raw in the side table and [release] in free().

namespace ctranslate2 {
  namespace metal {

    // Allocator backed by shared-storage MTLBuffers. On Apple Silicon's unified memory
    // the buffer's `contents` pointer is directly CPU-addressable, so it satisfies the
    // pointer-based Allocator contract and StorageView can treat it like a host pointer
    // (data<T>()). An address-ordered side table maps the contents pointer back to its
    // MTLBuffer for kernel dispatch, and supports range lookups so that pointers offset
    // into an allocation (StorageView sub-views, strided-batch matrices) resolve to the
    // owning buffer plus a byte offset.
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
        _buffers.emplace(reinterpret_cast<uintptr_t>(ptr), Block{buffer, size});
        return ptr;
      }

      void free(void* ptr, int /*device_index*/) override {
        if (!ptr)
          return;
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _buffers.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == _buffers.end())
          return;
        [it->second.buffer release];
        _buffers.erase(it);
      }

      BufferRange buffer_and_offset(const void* ptr) {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        std::lock_guard<std::mutex> lock(_mutex);
        // Find the allocation with the greatest base address <= addr.
        auto it = _buffers.upper_bound(addr);
        if (it == _buffers.begin())
          throw std::runtime_error("Metal: pointer is not a tracked Metal buffer");
        --it;
        const uintptr_t base = it->first;
        if (addr >= base + it->second.size)
          throw std::runtime_error("Metal: pointer is not inside a tracked Metal buffer");
        return BufferRange{it->second.buffer, static_cast<NSUInteger>(addr - base)};
      }

    private:
      struct Block {
        id<MTLBuffer> buffer;
        size_t size;
      };
      std::mutex _mutex;
      std::map<uintptr_t, Block> _buffers;
    };

    static MetalAllocator& get_metal_allocator() {
      static MetalAllocator allocator;
      return allocator;
    }

    BufferRange buffer_and_offset(const void* ptr) {
      return get_metal_allocator().buffer_and_offset(ptr);
    }

  }

  template<>
  Allocator& get_allocator<Device::METAL>() {
    return metal::get_metal_allocator();
  }

}
