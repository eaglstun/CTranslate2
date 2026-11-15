#include "metal/utils.h"
#include "metal/device.h"
#include "metal/kernels/kernels_msl.h"
#include "metal/kernels/kernels_mpp_msl.h"

#import <Foundation/Foundation.h>

#include <algorithm>
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

    // Submission tuning, read once from the environment (see the "Submission" block below).
    static const unsigned long kDefaultBatchOps = 8;
    static const unsigned long kDefaultMaxInflight = 0;
    static const unsigned long kDefaultBatchTargetUs = 100;

    static unsigned long env_ulong(const char* name, unsigned long default_value) {
      const char* env = std::getenv(name);
      if (!env || !*env)
        return default_value;
      char* end = nullptr;
      const unsigned long value = std::strtoul(env, &end, 10);
      return (end == env || *end != '\0') ? default_value : value;
    }

    static bool env_flag(const char* name) {
      const char* env = std::getenv(name);
      return env && *env && env[0] != '0';
    }

    static bool stats_enabled() {
      static const bool enabled = env_flag("CT2_METAL_STATS");
      return enabled;
    }

    // Ops per command buffer. 1 (or CT2_NO_METAL_CB_BATCH=1) is the per-op commit path.
    static unsigned long batch_ops() {
      static const unsigned long value = env_flag("CT2_NO_METAL_CB_BATCH")
        ? 1ul
        : std::max(1ul, env_ulong("CT2_METAL_CB_OPS", kDefaultBatchOps));
      return value;
    }

    static bool batching_enabled() {
      return batch_ops() > 1;
    }

    // Maximum number of uncompleted command buffers (the open one included) before the
    // CPU blocks waiting for the GPU. 0 keeps the queue's own default (64).
    static unsigned long max_inflight() {
      static const unsigned long value = env_flag("CT2_NO_METAL_CB_BATCH")
        ? 0ul
        : env_ulong("CT2_METAL_CB_MAX_INFLIGHT", kDefaultMaxInflight);
      return value;
    }

    // Estimated GPU time at which a batch is committed even if it is not full, in
    // microseconds; 0 sizes batches by op count only.
    static uint64_t batch_target_ns() {
      static const uint64_t value = batching_enabled()
        ? static_cast<uint64_t>(env_ulong("CT2_METAL_CB_TARGET_US", kDefaultBatchTargetUs)) * 1000
        : 0;
      return value;
    }

    // CT2_METAL_CB_EAGER=0 holds every batch until it is full, even with the GPU idle.
    static bool eager_when_idle() {
      static const bool value = env_ulong("CT2_METAL_CB_EAGER", 1) != 0;
      return value;
    }

    // Diagnostic: CT2_METAL_CB_SHARE_ENCODER=0 gives each op its own compute encoder
    // inside the shared command buffer.
    static bool share_encoder() {
      static const bool value = env_ulong("CT2_METAL_CB_SHARE_ENCODER", 1) != 0;
      return value;
    }

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

          const unsigned long inflight = max_inflight();
          _queue = inflight > 0
            ? [_device newCommandQueueWithMaxCommandBufferCount:inflight]
            : [_device newCommandQueue];
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

    // ---------------------------------------------------------------------------------
    // Submission: bounded command-buffer batching
    //
    // Every Metal op is a CommandScope (device.h). By default consecutive ops are appended
    // to one process-wide open command buffer, which is committed WITHOUT waiting as soon
    // as it is due (batch_is_due_locked(): full, enough estimated GPU work, or GPU idle);
    // flush() commits whatever is open and waits. Compared to one command buffer per op
    // this removes most of the fixed per-buffer cost (CPU: create + commit; GPU: a few
    // microseconds between buffers) for small ops, while the bounds keep the CPU/GPU
    // overlap: the GPU is never left idle with work pending, and large ops still get a
    // buffer each. Committing once per decode step / forward pass instead was tried and
    // reverted in 2026-06: it left the GPU idle until the end. When and how a batch is
    // committed changes timing only; the invariants below hold for any policy.
    //
    // Invariants everything else relies on (all enforced under g_commit_mutex):
    //  1. Issue order == encode order == commit order == GPU execution order. A scope
    //     holds the mutex from construction to commit(), so ops from different threads
    //     never interleave inside an op, land in the open buffer in the order the scopes
    //     were entered, and buffers are committed to the single FIFO queue in the order
    //     they were filled. There is exactly one open buffer for the whole process — not
    //     one per thread — so an op issued on a worker thread (Conv1D's parallel_for
    //     chunks, a second replica) can never be left behind in a batch that the reading
    //     thread does not know about, and can never run before an earlier op of another
    //     thread whose output it reads.
    //  2. g_last_committed is the most recently committed buffer, so "it completed"
    //     implies every earlier commit completed too.
    //  3. flush() = commit the open buffer (if any) + wait for g_last_committed. After it
    //     returns, every op whose scope was committed before the call has executed. It is
    //     the single coherence point before host access to Metal memory (CPU-reference
    //     ops via METAL_DEVICE_CASE, synchronize, host copies, sampling readback).
    //
    // CT2_NO_METAL_CB_BATCH=1 restores one command buffer per op (the scope then only
    // takes the mutex for the commit itself, exactly as before batching).
    // ---------------------------------------------------------------------------------
    static std::mutex g_commit_mutex;
    static id<MTLCommandBuffer> g_last_committed = nil;        // Retained.
    static id<MTLCommandBuffer> g_open_buffer = nil;           // Retained, not committed.
    static id<MTLComputeCommandEncoder> g_open_encoder = nil;  // Retained, encoding g_open_buffer.
    static unsigned long g_open_ops = 0;

    // Dispatch statistics, dumped at exit when CT2_METAL_STATS=1. Counter increments are
    // a few ns against a ~10us command-buffer encode, so they stay unconditionally on.
    // A "stalled" flush found the last-committed buffer not yet completed — the CPU
    // actually blocked in waitUntilCompleted; the stall time is summed separately.
    // The timed figures (acquire_wait, gpu_busy) need a clock read or a completion
    // handler per command buffer and are only collected when stats are enabled.
    static std::atomic<uint64_t> g_stat_command_buffers{0};
    static std::atomic<uint64_t> g_stat_ops{0};
    static std::atomic<uint64_t> g_stat_flushes{0};
    static std::atomic<uint64_t> g_stat_flushes_stalled{0};
    static std::atomic<uint64_t> g_stat_flushes_partial{0};
    static std::atomic<uint64_t> g_stat_flush_wait_us{0};
    static std::atomic<uint64_t> g_stat_acquire_wait_us{0};
    static std::atomic<uint64_t> g_stat_gpu_busy_ns{0};

    // Moving average of measured GPU nanoseconds per op (0 until the first completion).
    static std::atomic<uint64_t> g_gpu_ns_per_op{0};

    static struct MetalStatsAtExit {
      MetalStatsAtExit() {
        if (!stats_enabled())
          return;
        std::atexit([] {
          std::fprintf(stderr,
                       "[metal-stats] command_buffers=%llu ops=%llu flushes=%llu "
                       "stalled_flushes=%llu partial_batch_flushes=%llu flush_wait_ms=%.1f "
                       "acquire_wait_ms=%.1f gpu_busy_ms=%.1f batch_ops=%lu max_inflight=%lu "
                       "batch_target_us=%llu last_gpu_us_per_op=%.1f\n",
                       (unsigned long long)g_stat_command_buffers.load(),
                       (unsigned long long)g_stat_ops.load(),
                       (unsigned long long)g_stat_flushes.load(),
                       (unsigned long long)g_stat_flushes_stalled.load(),
                       (unsigned long long)g_stat_flushes_partial.load(),
                       g_stat_flush_wait_us.load() / 1000.0,
                       g_stat_acquire_wait_us.load() / 1000.0,
                       g_stat_gpu_busy_ns.load() / 1.0e6,
                       batch_ops(),
                       max_inflight(),
                       (unsigned long long)(batch_target_ns() / 1000),
                       g_gpu_ns_per_op.load() / 1000.0);
        });
      }
    } g_metal_stats_at_exit;

    // [queue commandBuffer] blocks while the queue already has its maximum number of
    // uncompleted buffers: that is the backpressure that bounds in-flight work. The result
    // is autoreleased; callers are inside a CommandScope's pool.
    static id<MTLCommandBuffer> acquire_command_buffer() {
      g_stat_command_buffers.fetch_add(1, std::memory_order_relaxed);
      if (!stats_enabled())
        return [get_command_queue() commandBuffer];
      const auto start = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> command_buffer = [get_command_queue() commandBuffer];
      const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start);
      g_stat_acquire_wait_us.fetch_add(elapsed.count(), std::memory_order_relaxed);
      return command_buffer;
    }

    // Commits a retained command buffer and makes it g_last_committed (which takes over
    // the caller's reference). Caller holds g_commit_mutex: the commit MUST happen inside
    // the lock, or two threads could commit in one order and record in the other, leaving
    // flush() waiting on a buffer that is not last in the queue.
    static void commit_locked(id<MTLCommandBuffer> command_buffer, unsigned long ops) {
      if (batch_target_ns() > 0 || stats_enabled()) {
        // Runs on a Metal-owned thread after the buffer completed: only touches atomics.
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
          const double seconds = completed.GPUEndTime - completed.GPUStartTime;
          if (seconds <= 0 || ops == 0)
            return;
          const uint64_t ns = static_cast<uint64_t>(seconds * 1.0e9);
          g_stat_gpu_busy_ns.fetch_add(ns, std::memory_order_relaxed);
          // Exponential moving average (weight 1/8) of GPU time per op. A lost update
          // between two completion threads only drops one sample.
          const uint64_t sample = ns / ops;
          const uint64_t previous = g_gpu_ns_per_op.load(std::memory_order_relaxed);
          const uint64_t next = previous == 0
            ? sample
            : previous - previous / 8 + sample / 8;
          g_gpu_ns_per_op.store(std::max<uint64_t>(next, 1), std::memory_order_relaxed);
        }];
      }
      [command_buffer commit];
      [g_last_committed release];
      g_last_committed = command_buffer;
    }

    static void end_open_encoder_locked() {
      if (g_open_encoder) {
        [g_open_encoder endEncoding];
        [g_open_encoder release];
        g_open_encoder = nil;
      }
    }

    // True when every committed buffer has completed. Caller holds g_commit_mutex.
    static bool gpu_idle_locked() {
      return !g_last_committed
        || g_last_committed.status >= MTLCommandBufferStatusCompleted;
    }

    // Decides whether the open batch, which just received an op, is committed now.
    // Caller holds g_commit_mutex.
    //  - Full: batch_ops() ops.
    //  - Enough GPU work: its estimated GPU time (ops x the moving average of measured
    //    GPU time per op) reached batch_target_ns(). Batching only pays for small ops,
    //    where the fixed per-buffer cost is comparable to the work. Large ops (prefill
    //    GEMMs and row kernels at hundreds of microseconds each) measured FASTER in
    //    their own buffers — the GPU overlaps consecutive command buffers but runs the
    //    passes of one buffer strictly one after the other — so in that regime the
    //    estimate exceeds the target after one op and this degenerates to per-op commit.
    //  - GPU idle: the last committed buffer already completed, so holding ops back
    //    would only lose CPU/GPU overlap. Batches build up exactly while the GPU is
    //    behind.
    static bool batch_is_due_locked() {
      ++g_open_ops;
      if (g_open_ops >= batch_ops())
        return true;
      const uint64_t target = batch_target_ns();
      if (target > 0
          && g_open_ops * g_gpu_ns_per_op.load(std::memory_order_relaxed) >= target)
        return true;
      return eager_when_idle() && gpu_idle_locked();
    }

    // Commits the open batch, if any. Caller holds g_commit_mutex.
    static bool commit_open_locked() {
      if (!g_open_buffer)
        return false;
      end_open_encoder_locked();
      commit_locked(g_open_buffer, g_open_ops);
      g_open_buffer = nil;
      g_open_ops = 0;
      return true;
    }

    // Per-op autorelease pool. Command buffers ([queue commandBuffer]), compute encoders, and
    // MPSMatrixDescriptors are all autoreleased (+0). CTranslate2 issues ops from C++ worker
    // threads that have no run loop, so without an explicit pool these objects are never
    // drained — over a long op chain (e.g. Whisper's 730s transcribe) they accumulate as wired
    // memory until the OS kills the process, while heap RSS stays flat. Each op is one
    // CommandScope, which pushes a pool in its constructor and drains it in commit() (or
    // its destructor), bracketing exactly that op's autoreleased temporaries on the thread
    // that issued it. Objects that outlive the op (the open buffer and its encoder, the
    // last-committed buffer) are explicitly retained, so they survive the drain — and can
    // be ended/committed from another thread later.
    CommandScope::CommandScope()
      : _pool([[NSAutoreleasePool alloc] init])
      , _own_buffer(nil)
      , _own_encoder(nil)
      , _locked(false)
    {
      if (!batching_enabled()) {
        _own_buffer = acquire_command_buffer();
        return;
      }
      g_commit_mutex.lock();
      _locked = true;
      if (!g_open_buffer)
        g_open_buffer = [acquire_command_buffer() retain];
    }

    id<MTLComputeCommandEncoder> CommandScope::compute_encoder() {
      if (!batching_enabled()) {
        if (!_own_encoder)
          _own_encoder = [_own_buffer computeCommandEncoder];
        return _own_encoder;
      }
      if (!g_open_encoder)
        g_open_encoder = [[g_open_buffer computeCommandEncoder] retain];
      return g_open_encoder;
    }

    id<MTLCommandBuffer> CommandScope::command_buffer() {
      if (!batching_enabled())
        return _own_buffer;
      end_open_encoder_locked();
      return g_open_buffer;
    }

    void CommandScope::commit() {
      g_stat_ops.fetch_add(1, std::memory_order_relaxed);
      if (!batching_enabled()) {
        if (_own_encoder) {
          [_own_encoder endEncoding];
          _own_encoder = nil;
        }
        {
          std::lock_guard<std::mutex> lock(g_commit_mutex);
          commit_locked([_own_buffer retain], 1);
        }
        _own_buffer = nil;
      } else {
        if (!share_encoder())
          end_open_encoder_locked();
        if (batch_is_due_locked())
          commit_open_locked();
        _locked = false;
        g_commit_mutex.unlock();
      }
      // Drain this op's autoreleased temporaries.
      [static_cast<NSAutoreleasePool*>(_pool) release];
      _pool = nullptr;
    }

    CommandScope::~CommandScope() {
      // Only reached with work pending when encoding threw before commit().
      if (_own_encoder)
        [_own_encoder endEncoding];  // An encoder must be ended before it is released.
      if (_locked)
        g_commit_mutex.unlock();
      if (_pool)
        [static_cast<NSAutoreleasePool*>(_pool) release];
    }

    void flush() {
      id<MTLCommandBuffer> to_wait = nil;
      {
        std::lock_guard<std::mutex> lock(g_commit_mutex);
        if (commit_open_locked())
          g_stat_flushes_partial.fetch_add(1, std::memory_order_relaxed);
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
