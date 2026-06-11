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

    // Create a fresh command buffer for an op to encode into.
    id<MTLCommandBuffer> new_command_buffer();

    // Commit a command buffer asynchronously (no wait) and record it as the global
    // last-committed buffer that metal::flush() (see utils.h) waits on.
    void commit_command_buffer(id<MTLCommandBuffer> command_buffer);

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
