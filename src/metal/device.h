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

    // Returns (and lazily creates + caches) the compute pipeline state for the kernel
    // function of the given name from the embedded shader library.
    id<MTLComputePipelineState> get_pipeline(const char* function_name);

    // Maps a pointer returned by the Metal allocator back to its backing MTLBuffer.
    // Throws if the pointer was not produced by the Metal allocator.
    id<MTLBuffer> buffer_for(const void* ptr);

  }
}
