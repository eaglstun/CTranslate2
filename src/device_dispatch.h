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
// Metal is intentionally not a DEVICE_CASE here during bring-up: routing it through
// this macro would instantiate primitives<Device::METAL> at every dispatch site before
// that specialization exists. Until then it is an explicit unsupported case, which both
// silences -Wswitch and turns a misrouted Metal device into a clear runtime error
// instead of a silent fall-through. Metal-resident work reaches the backend through
// dedicated entry points (e.g. the get_allocator early-return and metal:: functions).
#ifndef CT2_WITH_CUDA
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    UNSUPPORTED_DEVICE_CASE(Device::METAL)              \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#else
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    UNSUPPORTED_DEVICE_CASE(Device::METAL)              \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#endif
