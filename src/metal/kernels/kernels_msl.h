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

// ---- Normalizations (one threadgroup per row, fixed power-of-two reduction) ----
// Reductions accumulate in float; 1.0f/sqrt is used (not rsqrt) to match the CPU kernels.
constant uint CT2_NORM_TG = 256;

template <typename T>
inline void ct2_rms_norm_impl(device const T* input, device const T* gamma, device T* output,
                              uint depth, float epsilon, uint use_residual,
                              uint row, uint tid, threadgroup float* scratch) {
  device const T* x = input + (ulong)row * (ulong)depth;
  device T* y = output + (ulong)row * (ulong)depth;

  float local_ss = 0.0f;
  for (uint j = tid; j < depth; j += CT2_NORM_TG) {
    const float v = (float)x[j];
    local_ss += v * v;
  }
  scratch[tid] = local_ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = CT2_NORM_TG / 2u; s > 0u; s >>= 1) {
    if (tid < s)
      scratch[tid] += scratch[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms = 1.0f / sqrt(scratch[0] / (float)depth + epsilon);
  for (uint j = tid; j < depth; j += CT2_NORM_TG) {
    const float g = (float)gamma[j];
    y[j] = (T)((float)x[j] * inv_rms * (use_residual != 0u ? (1.0f + g) : g));
  }
}

kernel void ct2_rms_norm_float(device const float* input  [[buffer(0)]],
                               device const float* gamma  [[buffer(1)]],
                               device float* output        [[buffer(2)]],
                               constant uint& depth         [[buffer(3)]],
                               constant float& epsilon      [[buffer(4)]],
                               constant uint& use_residual  [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[CT2_NORM_TG];
  ct2_rms_norm_impl<float>(input, gamma, output, depth, epsilon, use_residual, row, tid, scratch);
}

kernel void ct2_rms_norm_half(device const half* input  [[buffer(0)]],
                              device const half* gamma  [[buffer(1)]],
                              device half* output        [[buffer(2)]],
                              constant uint& depth         [[buffer(3)]],
                              constant float& epsilon      [[buffer(4)]],
                              constant uint& use_residual  [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[CT2_NORM_TG];
  ct2_rms_norm_impl<half>(input, gamma, output, depth, epsilon, use_residual, row, tid, scratch);
}

template <typename T>
inline void ct2_layer_norm_impl(device const T* input, device const T* gamma,
                                device const T* beta, device T* output,
                                uint depth, float epsilon,
                                uint row, uint tid,
                                threadgroup float* s_sum, threadgroup float* s_sq) {
  device const T* x = input + (ulong)row * (ulong)depth;
  device T* y = output + (ulong)row * (ulong)depth;

  float local_sum = 0.0f;
  float local_sq = 0.0f;
  for (uint j = tid; j < depth; j += CT2_NORM_TG) {
    const float v = (float)x[j];
    local_sum += v;
    local_sq += v * v;
  }
  s_sum[tid] = local_sum;
  s_sq[tid] = local_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = CT2_NORM_TG / 2u; s > 0u; s >>= 1) {
    if (tid < s) {
      s_sum[tid] += s_sum[tid + s];
      s_sq[tid] += s_sq[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float mean = s_sum[0] / (float)depth;
  const float variance = max(s_sq[0] / (float)depth - mean * mean, 0.0f);
  const float rstd = 1.0f / sqrt(variance + epsilon);
  for (uint j = tid; j < depth; j += CT2_NORM_TG)
    y[j] = (T)(((float)x[j] - mean) * rstd * (float)gamma[j] + (float)beta[j]);
}

kernel void ct2_layer_norm_float(device const float* input  [[buffer(0)]],
                                 device const float* gamma  [[buffer(1)]],
                                 device const float* beta   [[buffer(2)]],
                                 device float* output        [[buffer(3)]],
                                 constant uint& depth         [[buffer(4)]],
                                 constant float& epsilon      [[buffer(5)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float s_sum[CT2_NORM_TG];
  threadgroup float s_sq[CT2_NORM_TG];
  ct2_layer_norm_impl<float>(input, gamma, beta, output, depth, epsilon, row, tid, s_sum, s_sq);
}

kernel void ct2_layer_norm_half(device const half* input  [[buffer(0)]],
                                device const half* gamma  [[buffer(1)]],
                                device const half* beta   [[buffer(2)]],
                                device half* output        [[buffer(3)]],
                                constant uint& depth         [[buffer(4)]],
                                constant float& epsilon      [[buffer(5)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float s_sum[CT2_NORM_TG];
  threadgroup float s_sq[CT2_NORM_TG];
  ct2_layer_norm_impl<half>(input, gamma, beta, output, depth, epsilon, row, tid, s_sum, s_sq);
}

// ---- Rotary (RoPE) ----  one thread per element of the [batch*max_time, depth] tensor.
// sin/cos are indexed by time only (t * ndims). Elements >= ndims are copied through.
template <typename T>
inline void ct2_rotary_impl(device const T* input, device const T* sin, device const T* cos,
                            device T* output, uint max_time, uint ndims, uint depth,
                            uint interleave, uint gid) {
  const uint row = gid / depth;
  const uint e = gid - row * depth;
  const uint t = row % max_time;

  device const T* x = input + (ulong)row * (ulong)depth;
  device T* y = output + (ulong)row * (ulong)depth;

  if (e >= ndims) {
    y[e] = x[e];
    return;
  }

  device const T* s = sin + (ulong)t * (ulong)ndims;
  device const T* c = cos + (ulong)t * (ulong)ndims;
  const uint middle = ndims / 2u;

  float pair;
  if (interleave != 0u)
    pair = (e % 2u == 0u) ? -(float)x[e + 1u] : (float)x[e - 1u];
  else
    pair = (e < middle) ? -(float)x[e + middle] : (float)x[e - middle];

  y[e] = (T)((float)x[e] * (float)c[e] + pair * (float)s[e]);
}

kernel void ct2_rotary_float(device const float* input [[buffer(0)]],
                             device const float* sin   [[buffer(1)]],
                             device const float* cos   [[buffer(2)]],
                             device float* output       [[buffer(3)]],
                             constant uint& max_time     [[buffer(4)]],
                             constant uint& ndims        [[buffer(5)]],
                             constant uint& depth        [[buffer(6)]],
                             constant uint& interleave   [[buffer(7)]],
                             uint gid [[thread_position_in_grid]]) {
  ct2_rotary_impl<float>(input, sin, cos, output, max_time, ndims, depth, interleave, gid);
}

kernel void ct2_rotary_half(device const half* input [[buffer(0)]],
                            device const half* sin   [[buffer(1)]],
                            device const half* cos   [[buffer(2)]],
                            device half* output       [[buffer(3)]],
                            constant uint& max_time     [[buffer(4)]],
                            constant uint& ndims        [[buffer(5)]],
                            constant uint& depth        [[buffer(6)]],
                            constant uint& interleave   [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
  ct2_rotary_impl<half>(input, sin, cos, output, max_time, ndims, depth, interleave, gid);
}

// ---- Fused bias-add + activation ----  out[i] = act(value[i] + bias[i % depth] + res[i]).
// `act` matches the ActivationType enum (0 ReLU, 1 GELUTanh, 2 Swish, 3 GELU(erf),
// 4 GELUSigmoid, 5 Tanh, 6 Sigmoid); any other value means no activation. Formulas mirror
// the CPU kernels exactly.
// Metal has no built-in erf, so approximate it (Abramowitz & Stegun 7.1.26, max abs
// error ~1.5e-7 — comfortably within the float32 test tolerance). Used for exact GELU.
inline float ct2_erf(float x) {
  const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
  const float a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
  const float s = (x < 0.0f) ? -1.0f : 1.0f;
  const float ax = fabs(x);
  const float t = 1.0f / (1.0f + p * ax);
  const float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
  return s * y;
}

inline float ct2_apply_activation(float v, int act) {
  switch (act) {
    case 0: return max(v, 0.0f);
    case 1: { const float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
              return 0.5f * v * (1.0f + tanh(u)); }
    case 2: return v / (1.0f + exp(-v));
    case 3: return 0.5f * v * (1.0f + ct2_erf(v * 0.7071067811865475f));
    case 4: return v / (1.0f + exp(-1.702f * v));
    case 5: return tanh(v);
    case 6: return 1.0f / (1.0f + exp(-v));
    default: return v;
  }
}

template <typename T>
inline void ct2_bias_add_impl(device const T* value, device const T* bias,
                              device const T* residual, device T* output,
                              uint depth, uint has_residual, int act, uint gid) {
  float v = (float)value[gid] + (float)bias[gid % depth];
  if (has_residual != 0u)
    v += (float)residual[gid];
  output[gid] = (T)ct2_apply_activation(v, act);
}

kernel void ct2_bias_add_float(device const float* value    [[buffer(0)]],
                               device const float* bias     [[buffer(1)]],
                               device const float* residual [[buffer(2)]],
                               device float* output         [[buffer(3)]],
                               constant uint& depth          [[buffer(4)]],
                               constant uint& has_residual   [[buffer(5)]],
                               constant int& act             [[buffer(6)]],
                               uint gid [[thread_position_in_grid]]) {
  ct2_bias_add_impl<float>(value, bias, residual, output, depth, has_residual, act, gid);
}

kernel void ct2_bias_add_half(device const half* value    [[buffer(0)]],
                              device const half* bias     [[buffer(1)]],
                              device const half* residual [[buffer(2)]],
                              device half* output         [[buffer(3)]],
                              constant uint& depth          [[buffer(4)]],
                              constant uint& has_residual   [[buffer(5)]],
                              constant int& act             [[buffer(6)]],
                              uint gid [[thread_position_in_grid]]) {
  ct2_bias_add_impl<half>(value, bias, residual, output, depth, has_residual, act, gid);
}

// ---- Gather ----  type-agnostic byte copy: one thread per gathered row.
// output[i] = data[batch_of(i) * batch_stride + indices[i] * copy_size], copy_size bytes.
kernel void ct2_gather_bytes(device const uchar* data    [[buffer(0)]],
                             device const int* indices    [[buffer(1)]],
                             device uchar* output         [[buffer(2)]],
                             constant uint& copy_size      [[buffer(3)]],
                             constant uint& batch_stride   [[buffer(4)]],
                             constant uint& num_per_batch  [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
  const uint batch = i / num_per_batch;
  const uint read = (uint)indices[i];
  device const uchar* src = data + (ulong)batch * (ulong)batch_stride + (ulong)read * (ulong)copy_size;
  device uchar* dst = output + (ulong)i * (ulong)copy_size;
  for (uint b = 0; b < copy_size; ++b)
    dst[b] = src[b];
}
)MSL";
    }

  }
}
