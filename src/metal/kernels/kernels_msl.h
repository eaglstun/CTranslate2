#pragma once

// Metal Shading Language source for the backend kernels, embedded as a C++ raw string
// and compiled at runtime via newLibraryWithSource. Keeping the MSL inline avoids any
// runtime .metallib path resolution during bring-up (CTranslate2 ships a bare shared
// library, not a framework bundle, so NSBundle resource lookup is unreliable). A later
// milestone can move these kernels into a precompiled .metallib for faster startup.

namespace ctranslate2 {
  namespace metal {

    inline const char* get_kernels_source() {
      return R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void ct2_add_float(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c       [[buffer(2)]],
                          constant uint& n      [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid < n)
    c[gid] = a[gid] + b[gid];
}

// One threadgroup per row, CT2_SOFTMAX_TG threads each. Matches the CPU softmax: per row
// size = has_lengths ? lengths[row] : depth; positions [size, depth) are set to 0; the
// reduction (max then sum of exp) runs over [0, size); is_log selects log-softmax.
constant uint CT2_SOFTMAX_TG = 256;

kernel void ct2_softmax_float(device const float* input  [[buffer(0)]],
                              device float* output       [[buffer(1)]],
                              device const int* lengths  [[buffer(2)]],
                              constant uint& depth        [[buffer(3)]],
                              constant uint& has_lengths  [[buffer(4)]],
                              constant uint& is_log       [[buffer(5)]],
                              uint row  [[threadgroup_position_in_grid]],
                              uint tid  [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[CT2_SOFTMAX_TG];

  uint size = depth;
  if (has_lengths != 0u) {
    const int len = lengths[row];
    size = len > 0 ? (uint)len : 0u;
  }

  device const float* x = input + (ulong)row * (ulong)depth;
  device float* y = output + (ulong)row * (ulong)depth;

  // Zero out the masked tail.
  for (uint j = size + tid; j < depth; j += CT2_SOFTMAX_TG)
    y[j] = 0.0f;
  if (size == 0u)
    return;

  // Reduce max over [0, size).
  float local_max = -INFINITY;
  for (uint j = tid; j < size; j += CT2_SOFTMAX_TG)
    local_max = max(local_max, x[j]);
  scratch[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = CT2_SOFTMAX_TG / 2u; s > 0u; s >>= 1) {
    if (tid < s)
      scratch[tid] = max(scratch[tid], scratch[tid + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float x_max = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce sum of exp(x - max) over [0, size).
  float local_sum = 0.0f;
  for (uint j = tid; j < size; j += CT2_SOFTMAX_TG)
    local_sum += exp(x[j] - x_max);
  scratch[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = CT2_SOFTMAX_TG / 2u; s > 0u; s >>= 1) {
    if (tid < s)
      scratch[tid] += scratch[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float exp_sum = scratch[0];

  if (is_log != 0u) {
    const float log_sum = log(exp_sum);
    for (uint j = tid; j < size; j += CT2_SOFTMAX_TG)
      y[j] = (x[j] - x_max) - log_sum;
  } else {
    const float inv_sum = 1.0f / exp_sum;
    for (uint j = tid; j < size; j += CT2_SOFTMAX_TG)
      y[j] = exp(x[j] - x_max) * inv_sum;
  }
}

// Half-precision softmax: same algorithm as ct2_softmax_float, but reads/writes half and
// accumulates the reductions in float for accuracy.
kernel void ct2_softmax_half(device const half* input  [[buffer(0)]],
                             device half* output       [[buffer(1)]],
                             device const int* lengths  [[buffer(2)]],
                             constant uint& depth        [[buffer(3)]],
                             constant uint& has_lengths  [[buffer(4)]],
                             constant uint& is_log       [[buffer(5)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint tid  [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[CT2_SOFTMAX_TG];

  uint size = depth;
  if (has_lengths != 0u) {
    const int len = lengths[row];
    size = len > 0 ? (uint)len : 0u;
  }

  device const half* x = input + (ulong)row * (ulong)depth;
  device half* y = output + (ulong)row * (ulong)depth;

  for (uint j = size + tid; j < depth; j += CT2_SOFTMAX_TG)
    y[j] = (half)0.0f;
  if (size == 0u)
    return;

  float local_max = -INFINITY;
  for (uint j = tid; j < size; j += CT2_SOFTMAX_TG)
    local_max = max(local_max, (float)x[j]);
  scratch[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = CT2_SOFTMAX_TG / 2u; s > 0u; s >>= 1) {
    if (tid < s)
      scratch[tid] = max(scratch[tid], scratch[tid + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float x_max = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float local_sum = 0.0f;
  for (uint j = tid; j < size; j += CT2_SOFTMAX_TG)
    local_sum += exp((float)x[j] - x_max);
  scratch[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = CT2_SOFTMAX_TG / 2u; s > 0u; s >>= 1) {
    if (tid < s)
      scratch[tid] += scratch[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float exp_sum = scratch[0];

  if (is_log != 0u) {
    const float log_sum = log(exp_sum);
    for (uint j = tid; j < size; j += CT2_SOFTMAX_TG)
      y[j] = (half)(((float)x[j] - x_max) - log_sum);
  } else {
    const float inv_sum = 1.0f / exp_sum;
    for (uint j = tid; j < size; j += CT2_SOFTMAX_TG)
      y[j] = (half)(exp((float)x[j] - x_max) * inv_sum);
  }
}
)MSL";
    }

  }
}
