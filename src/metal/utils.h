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

    // Blocks until all submitted Metal work has completed. Ops currently commit and
    // wait synchronously, so this is a no-op today, but callers should not rely on
    // that and should keep calling it where the CUDA backend synchronizes.
    void synchronize();

  }
}
