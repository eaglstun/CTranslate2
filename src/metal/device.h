#pragma once

// Objective-C++ internals of the Metal backend. This header imports Metal and may ONLY
// be included from .mm translation units. Ordinary C++ code should use metal/utils.h
// and metal/primitives.h instead.

#import <Metal/Metal.h>

namespace ctranslate2 {
  namespace metal {

    // The process-wide default Metal device.
    id<MTLDevice> get_metal_device();

    // The shared command queue used to submit work.
    id<MTLCommandQueue> get_command_queue();

    // One GPU op: everything a kernel entry point encodes between construction and
    // commit(). This is the only way Metal work is submitted.
    //
    // With command-buffer batching (the default) consecutive ops, from any thread, are
    // appended to ONE process-wide open command buffer, in the order their scopes are
    // entered; the scope holds the batch lock for its lifetime, so encoding is serialized
    // and the GPU executes ops in exactly the order they were issued. The open buffer is
    // committed (asynchronously) once it holds CT2_METAL_CB_OPS ops, and by metal::flush()
    // (see utils.h) before any host access. With CT2_NO_METAL_CB_BATCH=1 each scope gets
    // its own command buffer, committed at commit(), as before batching existed.
    //
    // The scope also brackets the op in an autorelease pool, so the autoreleased encoder,
    // command buffer and MPS descriptors are drained per op on the issuing thread. A scope
    // destroyed without commit() (an exception while encoding) releases the lock and the
    // pool; whatever it already encoded stays in the batch.
    //
    // Do not call metal::flush(), or open a second scope, while one is alive on the same
    // thread: the batch lock is not recursive.
    class CommandScope {
    public:
      CommandScope();
      ~CommandScope();
      CommandScope(const CommandScope&) = delete;
      CommandScope& operator=(const CommandScope&) = delete;

      // The compute encoder to encode this op's dispatches into. When batching, it is
      // shared with the previous ops of the open command buffer (dispatches of one encoder
      // execute serially, in order), so the op must set every piece of state its kernel
      // reads: pipeline, buffers and bytes are NOT reset between ops.
      id<MTLComputeCommandEncoder> compute_encoder();

      // The command buffer itself, for APIs that encode their own passes
      // (MPSMatrixMultiplication encodeToCommandBuffer:). Ends the shared compute encoder
      // first: a command buffer can only have one active encoder.
      id<MTLCommandBuffer> command_buffer();

      // Ends the op. Commits the command buffer when the batch is full (always, without
      // batching).
      void commit();

    private:
      void* _pool;                       // NSAutoreleasePool*
      id<MTLCommandBuffer> _own_buffer;  // Unbatched mode only.
      id<MTLComputeCommandEncoder> _own_encoder;
      bool _locked;
    };

    // Returns (and lazily creates + caches) the compute pipeline state for the kernel
    // function of the given name from the embedded shader library.
    id<MTLComputePipelineState> get_pipeline(const char* function_name);

    // Same, from the Metal-4 MPP library (kernels_mpp_msl.h, MSL 4.0). Returns nil —
    // never throws — when the OS, GPU, or toolchain cannot compile it; callers must
    // fall back to the classic kernels.
    id<MTLComputePipelineState> get_pipeline_mpp(const char* function_name);

    // A Metal buffer together with a byte offset into it.
    struct BufferRange {
      id<MTLBuffer> buffer;
      NSUInteger offset;
    };

    // Maps a pointer (possibly offset into a larger allocation, e.g. a StorageView
    // sub-view or a strided-batch matrix) back to its backing MTLBuffer and the byte
    // offset within it. Throws if the pointer is not inside any tracked allocation.
    BufferRange buffer_and_offset(const void* ptr);

  }
}
