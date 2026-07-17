#include "metal/utils.h"
#include "metal/device.h"
#include "metal/kernels/kernels_msl.h"
#include "metal/kernels/kernels_mpp_msl.h"

#import <Foundation/Foundation.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

        // Pipeline from the Metal-4 MPP library (kernels_mpp_msl.h). Returns nil — never
        // throws — when MSL 4.0 is unavailable (pre-macOS-26 OS, or the GPU/toolchain
        // rejects the source); callers fall back to the classic kernels. Failure is
        // remembered so the compile is attempted at most once per process.
        id<MTLComputePipelineState> pipeline_mpp(const std::string& name) {
          std::lock_guard<std::mutex> lock(_mutex);
          auto it = _mpp_pipelines.find(name);
          if (it != _mpp_pipelines.end())
            return it->second;
          if (_mpp_unavailable)
            return nil;

          if (!_mpp_library) {
            bool os_ok = false;
            if (@available(macOS 26.0, *))
              os_ok = true;
            if (os_ok) {
              MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
              // MTLLanguageVersion4_0 spelled numerically so this file still compiles
              // against pre-macOS-26 SDKs; the @available check gates it at runtime.
              options.languageVersion = (MTLLanguageVersion)((4 << 16) + 0);
              NSError* error = nil;
              NSString* source = [NSString stringWithUTF8String:get_mpp_kernels_source()];
              _mpp_library = [_device newLibraryWithSource:source options:options error:&error];
              [options release];
            }
            if (!_mpp_library) {
              _mpp_unavailable = true;
              return nil;
            }
          }

          id<MTLFunction> function =
            [_mpp_library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
          if (!function) {
            _mpp_unavailable = true;
            return nil;
          }
          NSError* error = nil;
          id<MTLComputePipelineState> pso =
            [_device newComputePipelineStateWithFunction:function error:&error];
          [function release];
          if (!pso) {
            _mpp_unavailable = true;
            return nil;
          }
          _mpp_pipelines.emplace(name, pso);
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
        id<MTLLibrary> _mpp_library = nil;
        bool _mpp_unavailable = false;
        std::mutex _mutex;
        std::unordered_map<std::string, id<MTLComputePipelineState>> _pipelines;
        std::unordered_map<std::string, id<MTLComputePipelineState>> _mpp_pipelines;
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

    // Asynchronous submission. Each Metal op encodes its own command buffer and commits it
    // WITHOUT waiting — so consecutive GPU ops pipeline instead of forcing a CPU<->GPU
    // round-trip per op. The single command queue executes committed buffers in FIFO order,
    // so the most-recently-committed buffer completing implies all prior work is done.
    // flush() waits on that buffer; it is called before any CPU access to Metal memory
    // (CPU-reference ops via METAL_DEVICE_CASE, synchronize, host copies). A GLOBAL (not
    // thread-local) last-committed handle is required because ops may be issued from worker
    // threads (e.g. Conv1D's parallel_for) while a different thread performs the read.
    static std::mutex g_commit_mutex;
    static id<MTLCommandBuffer> g_last_committed = nil;

    // Dispatch statistics, dumped at exit when CT2_METAL_STATS=1. Counter increments are
    // a few ns against a ~10us command-buffer encode, so they stay unconditionally on.
    // A "stalled" flush found the last-committed buffer not yet completed — the CPU
    // actually blocked in waitUntilCompleted; the stall time is summed separately.
    static std::atomic<uint64_t> g_stat_command_buffers{0};
    static std::atomic<uint64_t> g_stat_flushes{0};
    static std::atomic<uint64_t> g_stat_flushes_stalled{0};
    static std::atomic<uint64_t> g_stat_flush_wait_us{0};

    static struct MetalStatsAtExit {
      MetalStatsAtExit() {
        const char* env = std::getenv("CT2_METAL_STATS");
        if (!env || !*env || env[0] == '0')
          return;
        std::atexit([] {
          std::fprintf(stderr,
                       "[metal-stats] command_buffers=%llu flushes=%llu stalled_flushes=%llu "
                       "flush_wait_ms=%.1f\n",
                       (unsigned long long)g_stat_command_buffers.load(),
                       (unsigned long long)g_stat_flushes.load(),
                       (unsigned long long)g_stat_flushes_stalled.load(),
                       g_stat_flush_wait_us.load() / 1000.0);
        });
      }
    } g_metal_stats_at_exit;

    // Per-op autorelease pool. Command buffers ([queue commandBuffer]), compute encoders, and
    // MPSMatrixDescriptors are all autoreleased (+0). CTranslate2 issues ops from C++ worker
    // threads that have no run loop, so without an explicit pool these objects are never
    // drained — over a long op chain (e.g. Whisper's 730s transcribe) they accumulate as wired
    // memory until the OS kills the process, while heap RSS stays flat. Each op is a flat
    // new_command_buffer() -> encode -> commit_command_buffer() pair, so we push a thread-local
    // pool in new_command_buffer() and drain it in commit_command_buffer(), bracketing exactly
    // that op's autoreleased temporaries. The committed buffer is retained by g_last_committed,
    // so it outlives the drain.
    static thread_local NSAutoreleasePool* g_op_pool = nil;

    id<MTLCommandBuffer> new_command_buffer() {
      g_stat_command_buffers.fetch_add(1, std::memory_order_relaxed);
      g_op_pool = [[NSAutoreleasePool alloc] init];
      return [get_command_queue() commandBuffer];
    }

    void commit_command_buffer(id<MTLCommandBuffer> command_buffer) {
      {
        // The commit MUST happen inside the lock: flush()'s "waiting on the last-committed
        // buffer implies all prior work is done" only holds if the queue's commit order
        // matches the order of g_last_committed updates. With the commit outside the lock,
        // two worker threads (e.g. Conv1D's grouped parallel_for) can commit in one order
        // and record in the other, leaving flush() waiting on a buffer that is not last in
        // the queue — the CPU then reads rows whose GPU write has not completed.
        std::lock_guard<std::mutex> lock(g_commit_mutex);
        [command_buffer commit];
        [command_buffer retain];
        [g_last_committed release];
        g_last_committed = command_buffer;
      }
      // Drain this op's autoreleased temporaries (encoder, descriptors, the command buffer's
      // own +0). The retain above keeps the committed buffer alive past the drain.
      if (g_op_pool) {
        [g_op_pool release];
        g_op_pool = nil;
      }
    }

    void flush() {
      id<MTLCommandBuffer> to_wait = nil;
      {
        std::lock_guard<std::mutex> lock(g_commit_mutex);
        to_wait = [g_last_committed retain];
      }
      g_stat_flushes.fetch_add(1, std::memory_order_relaxed);
      if (to_wait) {
        if (to_wait.status != MTLCommandBufferStatusCompleted) {
          g_stat_flushes_stalled.fetch_add(1, std::memory_order_relaxed);
          const auto start = std::chrono::steady_clock::now();
          [to_wait waitUntilCompleted];
          const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start);
          g_stat_flush_wait_us.fetch_add(elapsed.count(), std::memory_order_relaxed);
        } else {
          [to_wait waitUntilCompleted];
        }
        [to_wait release];
      }
    }

    void synchronize() {
      flush();
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

    id<MTLComputePipelineState> get_pipeline_mpp(const char* function_name) {
      return MetalContext::instance().pipeline_mpp(function_name);
    }

  }
}
