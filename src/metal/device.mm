#include "metal/utils.h"
#include "metal/device.h"
#include "metal/kernels/kernels_msl.h"

#import <Foundation/Foundation.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

// NOTE: This translation unit is compiled as Objective-C++ WITHOUT ARC (manual
// retain/release). Objects created via Create/New rules are owned (+1); singleton-held
// objects live for the process lifetime and are intentionally never released.

namespace ctranslate2 {
  namespace metal {

    namespace {

      class MetalContext {
      public:
        static MetalContext& instance() {
          static MetalContext ctx;
          return ctx;
        }

        id<MTLDevice> device() const { return _device; }
        id<MTLCommandQueue> queue() const { return _queue; }

        id<MTLComputePipelineState> pipeline(const std::string& name) {
          std::lock_guard<std::mutex> lock(_mutex);
          auto it = _pipelines.find(name);
          if (it != _pipelines.end())
            return it->second;

          ensure_library();
          id<MTLFunction> function =
            [_library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
          if (!function)
            throw std::runtime_error("Metal: kernel function not found: " + name);

          NSError* error = nil;
          id<MTLComputePipelineState> pso =
            [_device newComputePipelineStateWithFunction:function error:&error];
          [function release];
          if (!pso) {
            std::string msg = "Metal: failed to create pipeline state for " + name;
            if (error)
              msg += std::string(": ") + [[error localizedDescription] UTF8String];
            throw std::runtime_error(msg);
          }

          _pipelines.emplace(name, pso);
          return pso;
        }

      private:
        MetalContext() {
          _device = MTLCreateSystemDefaultDevice();
          if (!_device)
            throw std::runtime_error("Metal: no default device available");

          _queue = [_device newCommandQueue];
          if (!_queue)
            throw std::runtime_error("Metal: failed to create command queue");
        }

        // The kernel library is compiled lazily on first pipeline use so that device
        // setup, allocation, and MPS-based ops (which don't need it) stay usable even if
        // a kernel fails to compile. Callers hold _mutex.
        void ensure_library() {
          if (_library)
            return;
          NSError* error = nil;
          NSString* source = [NSString stringWithUTF8String:get_kernels_source()];
          MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
          _library = [_device newLibraryWithSource:source options:options error:&error];
          [options release];
          if (!_library) {
            std::string msg = "Metal: failed to compile kernel library";
            if (error)
              msg += std::string(": ") + [[error localizedDescription] UTF8String];
            throw std::runtime_error(msg);
          }
        }

        id<MTLDevice> _device = nil;
        id<MTLCommandQueue> _queue = nil;
        id<MTLLibrary> _library = nil;
        std::mutex _mutex;
        std::unordered_map<std::string, id<MTLComputePipelineState>> _pipelines;
      };

    }

    bool has_gpu() {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      const bool available = (device != nil);
      [device release];
      return available;
    }

    int get_gpu_count() {
      return has_gpu() ? 1 : 0;
    }

    void synchronize() {
      // Ops currently commit a command buffer and wait on it synchronously, so there is
      // no outstanding asynchronous work to flush. Kept for parity with the CUDA backend.
    }

    id<MTLDevice> get_metal_device() {
      return MetalContext::instance().device();
    }

    id<MTLCommandQueue> get_command_queue() {
      return MetalContext::instance().queue();
    }

    id<MTLComputePipelineState> get_pipeline(const char* function_name) {
      return MetalContext::instance().pipeline(function_name);
    }

  }
}
