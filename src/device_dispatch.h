#pragma once

#include <stdexcept>

#include "ctranslate2/devices.h"

#define UNSUPPORTED_DEVICE_CASE(DEVICE)                       \
  case DEVICE: {                                              \
    throw std::runtime_error("unsupported device " #DEVICE);  \
    break;                                                    \
  }

#define DEVICE_CASE(DEVICE, STMT)               \
  case DEVICE: {                                \
    constexpr Device D = DEVICE;                \
    STMT;                                       \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__

// Metal bring-up uses the CPU implementation as a reference backend. On Apple Silicon's
// unified memory, a Metal buffer's contents pointer is CPU-addressable, so binding
// constexpr Device D = Device::CPU here runs the existing CPU primitives/op kernels
// directly on Metal-resident data — correct, just not yet GPU-accelerated. This avoids
// instantiating primitives<Device::METAL>/compute<Device::METAL,T> at all ~50 dispatch
// sites. Later milestones graduate individual hot paths (e.g. GEMM via MPS) to real
// Metal kernels via targeted routing. When CT2_WITH_METAL is not defined, Metal is an
// explicit unsupported case (also silences -Wswitch).
#ifdef CT2_WITH_METAL
#  define METAL_DEVICE_CASE(STMTS)              \
  case Device::METAL: {                         \
    constexpr Device D = Device::CPU;           \
    STMTS;                                       \
    break;                                       \
  }
#else
#  define METAL_DEVICE_CASE(STMTS) UNSUPPORTED_DEVICE_CASE(Device::METAL)
#endif

#ifndef CT2_WITH_CUDA
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    METAL_DEVICE_CASE(SINGLE_ARG(STMTS))                \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#else
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    METAL_DEVICE_CASE(SINGLE_ARG(STMTS))                \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#endif
