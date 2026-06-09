#pragma once

// Pure C++ surface of the Metal backend. This header contains no Objective-C and is
// safe to include from ordinary .cc translation units (e.g. devices.cc). The
// Objective-C++ internals live in metal/device.h, which must only be included by .mm
// files.

namespace ctranslate2 {
  namespace metal {

    // Returns true if a default Metal device is available on this system.
    bool has_gpu();

    // Number of usable Metal devices (0 or 1 in the current single-device model).
    int get_gpu_count();

    // Commits the current thread's open command buffer (if any) and blocks until it
    // completes. This is the single point that makes pending GPU writes visible to the
    // CPU; it is called before any CPU access to Metal-resident memory. synchronize() is
    // an alias kept for parity with the CUDA backend's naming.
    void flush();
    void synchronize();

  }
}
