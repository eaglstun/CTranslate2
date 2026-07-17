#pragma once

// Metal Shading Language source for the backend kernels, embedded as a C++ raw string
// and compiled at runtime via newLibraryWithSource. Keeping the MSL inline avoids any
// runtime .metallib path resolution (CTranslate2 ships a bare shared library, not a
// framework bundle, so NSBundle resource lookup is unreliable).
//
// A precompiled .metallib was considered for faster startup and MEASURED to be not worth
// it: newLibraryWithSource costs ~123ms only on a truly cold system shader cache (clean
// install / driver update); macOS caches compiled shaders by source hash at the system
// level, so every subsequent process — even fresh ones — pays ~0.5ms. Precompiling would
// save ~123ms once per machine in exchange for build-time xcrun metal + the bundle
// path-resolution problem above. Bad trade; keep the inline source.

namespace ctranslate2 {
  namespace metal {

    inline const char* get_kernels_source() {
      return R"MSL(
#include <metal_stdlib>
using namespace metal;

// Elementwise add with an optional scalar second operand (b_is_scalar): when set, every
// element adds `scalar` instead of b[gid] (b may then be an unused dummy buffer). Compute
// in float and cast back so half has the same rounding as the fp32 path.
template <typename T>
inline void ct2_add_impl(device const T* a, device const T* b, device T* c,
                         uint b_is_scalar, float scalar, uint gid) {
  const float bv = (b_is_scalar != 0u) ? scalar : (float)b[gid];
  c[gid] = (T)((float)a[gid] + bv);
}

kernel void ct2_add_float(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c        [[buffer(2)]],
                          constant uint& b_is_scalar [[buffer(3)]],
                          constant float& scalar     [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
  ct2_add_impl<float>(a, b, c, b_is_scalar, scalar, gid);
}

kernel void ct2_add_half(device const half* a [[buffer(0)]],
                         device const half* b [[buffer(1)]],
                         device half* c        [[buffer(2)]],
                         constant uint& b_is_scalar [[buffer(3)]],
                         constant float& scalar     [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
  ct2_add_impl<half>(a, b, c, b_is_scalar, scalar, gid);
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

// ---- Fused decode-step attention (single-query SDPA) ----  collapses the three-op
// MatMul(q·K^T, scale) → SoftMax(lengths) → MatMul(·V) sequence of dot_product_attention
// into one launch, never materializing the [rows, num_keys] score tensor. One threadgroup
// per score row (batch*heads*q_len rows; q_len is 1 in greedy decode, the beam width in
// beam decode, or a short prompt length). lengths follows the SoftMax mask contract
// exactly: per-row int32 size, len == 0 zeroes the row, positions >= len contribute
// nothing. CT2_SDPA_SG SIMD-groups stride the key axis, each keeping an online-softmax
// partial (running max m, denominator l, weighted V accumulator); the partials merge
// through threadgroup memory at the end. All accumulation is float in both precisions.
// Lane j owns output dims j, j+32, ... — the 32 is the Apple SIMD-group width, fixed on
// every Apple GPU (this backend is Apple Silicon only).
constant uint CT2_SDPA_SG = 4;       // SIMD-groups (of 32 lanes) per row
constant uint CT2_SDPA_MAX_D = 256;  // max head depth; the host routes here only when
                                     // depth <= this (8 accumulator registers per lane)

template <typename T>
inline void ct2_sdpa_impl(device const T* q_buf,
                          device const T* k_buf,
                          device const T* v_buf,
                          device T* out,
                          device const int* lengths,
                          uint num_keys, uint depth, uint rows_per_bh,
                          float scale, uint has_lengths,
                          uint row, uint simd_lane, uint simd_group,
                          threadgroup float* tg_m,
                          threadgroup float* tg_l,
                          threadgroup float* tg_acc) {
  const uint bh = row / rows_per_bh;
  device const T* q_row = q_buf + (ulong)row * depth;
  device const T* k_base = k_buf + (ulong)bh * num_keys * depth;
  device const T* v_base = v_buf + (ulong)bh * num_keys * depth;
  device T* out_row = out + (ulong)row * depth;

  uint len = num_keys;
  if (has_lengths != 0u) {
    const int l = lengths[row];
    len = l > 0 ? (uint)l : 0u;
  }

  if (len == 0u) {  // SoftMax contract: fully masked row → exact zeros
    for (uint j = simd_group * 32u + simd_lane; j < depth; j += CT2_SDPA_SG * 32u)
      out_row[j] = T(0);
    return;
  }

  const uint dims_per_lane = (depth + 31u) / 32u;
  float m = -INFINITY;
  float l_sum = 0.0f;
  float acc[CT2_SDPA_MAX_D / 32u];
  for (uint i = 0u; i < dims_per_lane; ++i)
    acc[i] = 0.0f;

  // Keys t = simd_group, simd_group + CT2_SDPA_SG, ... — score then online update.
  // exp(-INFINITY - m_new) == 0 makes the first iteration's rescale a no-op.
  for (uint t = simd_group; t < len; t += CT2_SDPA_SG) {
    device const T* k_row = k_base + (ulong)t * depth;
    float partial = 0.0f;
    for (uint j = simd_lane; j < depth; j += 32u)
      partial += (float)q_row[j] * (float)k_row[j];
    const float score = simd_sum(partial) * scale;

    const float m_new = max(m, score);
    const float rescale = exp(m - m_new);
    const float p = exp(score - m_new);
    l_sum = l_sum * rescale + p;
    device const T* v_row = v_base + (ulong)t * depth;
    for (uint i = 0u; i < dims_per_lane; ++i) {
      const uint j = simd_lane + 32u * i;
      const float v_val = j < depth ? (float)v_row[j] : 0.0f;
      acc[i] = acc[i] * rescale + p * v_val;
    }
    m = m_new;
  }

  // Merge the per-SIMD-group partials. len > 0 guarantees group 0 saw key 0, so the
  // global max is finite; a group with no keys (len < CT2_SDPA_SG) contributes
  // exp(-INFINITY - m_all) == 0.
  if (simd_lane == 0u) {
    tg_m[simd_group] = m;
    tg_l[simd_group] = l_sum;
  }
  for (uint i = 0u; i < dims_per_lane; ++i)
    tg_acc[simd_group * CT2_SDPA_MAX_D + simd_lane + 32u * i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group == 0u) {
    float m_all = -INFINITY;
    for (uint g = 0u; g < CT2_SDPA_SG; ++g)
      m_all = max(m_all, tg_m[g]);
    float l_all = 0.0f;
    for (uint g = 0u; g < CT2_SDPA_SG; ++g)
      l_all += tg_l[g] * exp(tg_m[g] - m_all);
    const float inv_l = 1.0f / l_all;
    for (uint i = 0u; i < dims_per_lane; ++i) {
      const uint j = simd_lane + 32u * i;
      if (j < depth) {
        float o = 0.0f;
        for (uint g = 0u; g < CT2_SDPA_SG; ++g)
          o += tg_acc[g * CT2_SDPA_MAX_D + j] * exp(tg_m[g] - m_all);
        out_row[j] = T(o * inv_l);
      }
    }
  }
}

kernel void ct2_sdpa_float(device const float* q      [[buffer(0)]],
                           device const float* k      [[buffer(1)]],
                           device const float* v      [[buffer(2)]],
                           device float* out          [[buffer(3)]],
                           device const int* lengths  [[buffer(4)]],
                           constant uint& num_keys     [[buffer(5)]],
                           constant uint& depth        [[buffer(6)]],
                           constant uint& rows_per_bh  [[buffer(7)]],
                           constant float& scale       [[buffer(8)]],
                           constant uint& has_lengths  [[buffer(9)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint simd_lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float tg_m[CT2_SDPA_SG];
  threadgroup float tg_l[CT2_SDPA_SG];
  threadgroup float tg_acc[CT2_SDPA_SG * CT2_SDPA_MAX_D];
  ct2_sdpa_impl<float>(q, k, v, out, lengths, num_keys, depth, rows_per_bh, scale,
                       has_lengths, row, simd_lane, simd_group, tg_m, tg_l, tg_acc);
}

kernel void ct2_sdpa_half(device const half* q       [[buffer(0)]],
                          device const half* k       [[buffer(1)]],
                          device const half* v       [[buffer(2)]],
                          device half* out           [[buffer(3)]],
                          device const int* lengths  [[buffer(4)]],
                          constant uint& num_keys     [[buffer(5)]],
                          constant uint& depth        [[buffer(6)]],
                          constant uint& rows_per_bh  [[buffer(7)]],
                          constant float& scale       [[buffer(8)]],
                          constant uint& has_lengths  [[buffer(9)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint simd_lane [[thread_index_in_simdgroup]],
                          uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float tg_m[CT2_SDPA_SG];
  threadgroup float tg_l[CT2_SDPA_SG];
  threadgroup float tg_acc[CT2_SDPA_SG * CT2_SDPA_MAX_D];
  ct2_sdpa_impl<half>(q, k, v, out, lengths, num_keys, depth, rows_per_bh, scale,
                      has_lengths, row, simd_lane, simd_group, tg_m, tg_l, tg_acc);
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

// Fused residual-add + RMSNorm. Writes BOTH sum = a + b (the new residual-stream value,
// needed by the next residual add) and normed = rmsnorm(sum) * (use_residual ? 1+g : g).
// vs separate Add+RMSNorm this removes one kernel launch and one device read pass (the Add
// writes sum, then the norm re-reads it; here phase 2 re-reads the sum we just wrote — 3
// reads instead of 4). Same tree reduction as ct2_rms_norm (SIMD-group reduction lost; see
// the perf log). Buffers: a, b, gamma, sum_out, normed_out, then depth/epsilon/use_residual.
template <typename T>
inline void ct2_add_rms_norm_impl(device const T* a, device const T* b, device const T* gamma,
                                  device T* sum_out, device T* normed_out,
                                  uint depth, float epsilon, uint use_residual,
                                  uint row, uint tid, threadgroup float* scratch) {
  const ulong base = (ulong)row * (ulong)depth;
  device const T* a_r = a + base;
  device const T* b_r = b + base;
  device T* s_r = sum_out + base;
  device T* y_r = normed_out + base;

  float local_ss = 0.0f;
  for (uint j = tid; j < depth; j += CT2_NORM_TG) {
    const float v = (float)a_r[j] + (float)b_r[j];
    s_r[j] = (T)v;
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
    y_r[j] = (T)((float)s_r[j] * inv_rms * (use_residual != 0u ? (1.0f + g) : g));
  }
}

kernel void ct2_add_rms_norm_float(device const float* a [[buffer(0)]],
                                   device const float* b [[buffer(1)]],
                                   device const float* gamma [[buffer(2)]],
                                   device float* sum_out    [[buffer(3)]],
                                   device float* normed_out [[buffer(4)]],
                                   constant uint& depth        [[buffer(5)]],
                                   constant float& epsilon     [[buffer(6)]],
                                   constant uint& use_residual [[buffer(7)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[CT2_NORM_TG];
  ct2_add_rms_norm_impl<float>(a, b, gamma, sum_out, normed_out, depth, epsilon, use_residual,
                               row, tid, scratch);
}

kernel void ct2_add_rms_norm_half(device const half* a [[buffer(0)]],
                                  device const half* b [[buffer(1)]],
                                  device const half* gamma [[buffer(2)]],
                                  device half* sum_out    [[buffer(3)]],
                                  device half* normed_out [[buffer(4)]],
                                  constant uint& depth        [[buffer(5)]],
                                  constant float& epsilon     [[buffer(6)]],
                                  constant uint& use_residual [[buffer(7)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[CT2_NORM_TG];
  ct2_add_rms_norm_impl<half>(a, b, gamma, sum_out, normed_out, depth, epsilon, use_residual,
                              row, tid, scratch);
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

// Fused residual-add + LayerNorm (the LayerNorm analogue of ct2_add_rms_norm). Writes
// sum = a + b (new residual stream) and normed = (sum-mean)/sqrt(var+eps)*gamma + beta.
// Uses 1.0f/sqrt (NOT rsqrt) so the result double-rounds like the CPU's 1.0/std::sqrt under
// the library's fast-math default (see apple-silicon/math-functions-and-numeric-parity.md).
template <typename T>
inline void ct2_add_layer_norm_impl(device const T* a, device const T* b,
                                    device const T* gamma, device const T* beta,
                                    device T* sum_out, device T* normed_out,
                                    uint depth, float epsilon, uint row, uint tid,
                                    threadgroup float* s_sum, threadgroup float* s_sq) {
  const ulong base = (ulong)row * (ulong)depth;
  device const T* a_r = a + base;
  device const T* b_r = b + base;
  device T* s_r = sum_out + base;
  device T* y_r = normed_out + base;

  float local_sum = 0.0f;
  float local_sq = 0.0f;
  for (uint j = tid; j < depth; j += CT2_NORM_TG) {
    const float v = (float)a_r[j] + (float)b_r[j];
    s_r[j] = (T)v;
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
    y_r[j] = (T)(((float)s_r[j] - mean) * rstd * (float)gamma[j] + (float)beta[j]);
}

kernel void ct2_add_layer_norm_float(device const float* a [[buffer(0)]],
                                     device const float* b [[buffer(1)]],
                                     device const float* gamma [[buffer(2)]],
                                     device const float* beta  [[buffer(3)]],
                                     device float* sum_out    [[buffer(4)]],
                                     device float* normed_out [[buffer(5)]],
                                     constant uint& depth     [[buffer(6)]],
                                     constant float& epsilon  [[buffer(7)]],
                                     uint row [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float s_sum[CT2_NORM_TG];
  threadgroup float s_sq[CT2_NORM_TG];
  ct2_add_layer_norm_impl<float>(a, b, gamma, beta, sum_out, normed_out, depth, epsilon,
                                 row, tid, s_sum, s_sq);
}

kernel void ct2_add_layer_norm_half(device const half* a [[buffer(0)]],
                                    device const half* b [[buffer(1)]],
                                    device const half* gamma [[buffer(2)]],
                                    device const half* beta  [[buffer(3)]],
                                    device half* sum_out    [[buffer(4)]],
                                    device half* normed_out [[buffer(5)]],
                                    constant uint& depth     [[buffer(6)]],
                                    constant float& epsilon  [[buffer(7)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float s_sum[CT2_NORM_TG];
  threadgroup float s_sq[CT2_NORM_TG];
  ct2_add_layer_norm_impl<half>(a, b, gamma, beta, sum_out, normed_out, depth, epsilon,
                                row, tid, s_sum, s_sq);
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

// Metal's tanh(x) computes (exp(2x)-1)/(exp(2x)+1); for large |x| exp(2x) overflows to Inf
// and Inf/Inf = NaN, whereas tanh mathematically saturates to +-1. tanh(+-15) already equals
// +-1.0 in float32, so clamping the argument to [-15,15] is exact for the saturated region and
// avoids the overflow. This bites GELU-tanh on Gemma2, whose large deep-layer activations make
// the cubic argument huge (NaN -> <pad> collapse on the GPU; CPU std::tanh saturates correctly).
inline float ct2_tanh_safe(float x) {
  return tanh(clamp(x, -15.0f, 15.0f));
}

inline float ct2_apply_activation(float v, int act) {
  switch (act) {
    case 0: return max(v, 0.0f);
    case 1: { const float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
              return 0.5f * v * (1.0f + ct2_tanh_safe(u)); }
    case 2: return v / (1.0f + exp(-v));
    case 3: return 0.5f * v * (1.0f + ct2_erf(v * 0.7071067811865475f));
    case 4: return v / (1.0f + exp(-1.702f * v));
    case 5: return ct2_tanh_safe(v);
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

// ---- Standalone activation ----  out[i] = act(in[i]).  `act` matches ActivationType.
template <typename T>
inline void ct2_activation_impl(device const T* x, device T* y, int act, uint gid) {
  y[gid] = (T)ct2_apply_activation((float)x[gid], act);
}

kernel void ct2_activation_float(device const float* x [[buffer(0)]],
                                 device float* y        [[buffer(1)]],
                                 constant int& act       [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
  ct2_activation_impl<float>(x, y, act, gid);
}

kernel void ct2_activation_half(device const half* x [[buffer(0)]],
                                device half* y        [[buffer(1)]],
                                constant int& act      [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
  ct2_activation_impl<half>(x, y, act, gid);
}

// ---- Elementwise multiply ----  c[i] = a[i] * (b_is_scalar ? scalar : b[i]).
// When b_is_scalar, the scalar is passed by value (the scalar operand may live on the
// host / a different device), and the b buffer is an unused dummy.
template <typename T>
inline void ct2_mul_impl(device const T* a, device const T* b, device T* c,
                         uint b_is_scalar, float scalar, uint gid) {
  const float bv = (b_is_scalar != 0u) ? scalar : (float)b[gid];
  c[gid] = (T)((float)a[gid] * bv);
}

kernel void ct2_mul_float(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c        [[buffer(2)]],
                          constant uint& b_is_scalar [[buffer(3)]],
                          constant float& scalar     [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
  ct2_mul_impl<float>(a, b, c, b_is_scalar, scalar, gid);
}

kernel void ct2_mul_half(device const half* a [[buffer(0)]],
                         device const half* b [[buffer(1)]],
                         device half* c        [[buffer(2)]],
                         constant uint& b_is_scalar [[buffer(3)]],
                         constant float& scalar     [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
  ct2_mul_impl<half>(a, b, c, b_is_scalar, scalar, gid);
}

// ---- INT8 quantization ----
// CT2's int8 scheme is symmetric per-row dynamic quantization (no zero-point):
// scale = 127 / amax(row), y = round_or_truncate(x * scale). One threadgroup per row,
// same fixed power-of-two tree reduction as the norms. Divisions use precise::divide:
// the CPU reference divides with IEEE semantics, and the Quantize op tests compare the
// scales near-exactly — fast-math division is not correctly rounded.
constant uint CT2_QUANT_TG = 256;

template <typename T>
inline void ct2_quantize_s8_impl(device const T* input, device char* output,
                                 device float* scales,
                                 uint depth, uint round_before_cast,
                                 uint row, uint tid, threadgroup float* scratch) {
  device const T* x = input + (ulong)row * (ulong)depth;
  device char* y = output + (ulong)row * (ulong)depth;

  float local_amax = 0.0f;
  for (uint j = tid; j < depth; j += CT2_QUANT_TG)
    local_amax = max(local_amax, fabs((float)x[j]));
  scratch[tid] = local_amax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = CT2_QUANT_TG / 2u; s > 0u; s >>= 1) {
    if (tid < s)
      scratch[tid] = max(scratch[tid], scratch[tid + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float amax = scratch[0];
  const float scale = (amax != 0.0f) ? precise::divide(127.0f, amax) : 1.0f;
  if (tid == 0u)
    scales[row] = scale;

  // rint = round half to even, matching the CPU's nearbyintf / vrndnq_f32; the
  // legacy (round_before_cast=false) path truncates toward zero like a C cast.
  for (uint j = tid; j < depth; j += CT2_QUANT_TG) {
    const float v = (float)x[j] * scale;
    y[j] = (char)(round_before_cast != 0u ? rint(v) : v);
  }
}

kernel void ct2_quantize_s8_float(device const float* input [[buffer(0)]],
                                  device char* output        [[buffer(1)]],
                                  device float* scales       [[buffer(2)]],
                                  constant uint& depth        [[buffer(3)]],
                                  constant uint& round_before_cast [[buffer(4)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[CT2_QUANT_TG];
  ct2_quantize_s8_impl<float>(input, output, scales, depth, round_before_cast, row, tid, scratch);
}

kernel void ct2_quantize_s8_half(device const half* input [[buffer(0)]],
                                 device char* output       [[buffer(1)]],
                                 device float* scales      [[buffer(2)]],
                                 constant uint& depth       [[buffer(3)]],
                                 constant uint& round_before_cast [[buffer(4)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[CT2_QUANT_TG];
  ct2_quantize_s8_impl<half>(input, output, scales, depth, round_before_cast, row, tid, scratch);
}

// ---- INT8 dequantization (simple form) ----  y = (float)x * (1 / scale[row]).
// The reciprocal-then-multiply spelling matches the CPU kernel exactly (it precomputes
// r_scale = 1/scale per row); one thread per element.
template <typename T>
inline void ct2_dequantize_s8_impl(device const char* input, device const float* scales,
                                   device T* output, uint depth, uint gid) {
  const uint row = gid / depth;
  const float r_scale = precise::divide(1.0f, scales[row]);
  output[gid] = (T)((float)input[gid] * r_scale);
}

kernel void ct2_dequantize_s8_float(device const char* input   [[buffer(0)]],
                                    device const float* scales [[buffer(1)]],
                                    device float* output       [[buffer(2)]],
                                    constant uint& depth        [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  ct2_dequantize_s8_impl<float>(input, scales, output, depth, gid);
}

kernel void ct2_dequantize_s8_half(device const char* input   [[buffer(0)]],
                                   device const float* scales [[buffer(1)]],
                                   device half* output        [[buffer(2)]],
                                   constant uint& depth        [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
  ct2_dequantize_s8_impl<half>(input, scales, output, depth, gid);
}

// ---- INT8 GEMM-output dequantization ----  the Dense epilogue for quantized GEMM:
// y[i][j] = act(((float)c[i][j] * (1/a_scale[i])) / b_scale[j] + bias[j]).
// a_scale is the dynamic per-row activation scale, b_scale the static per-output-channel
// weight scale (the !trans_a && trans_b layout Dense always uses). Operation order and
// the reciprocal/divide split mirror the CPU kernel. One thread per element.
template <typename T>
inline void ct2_dequant_gemm_out_impl(device const int* c, device const float* a_scale,
                                      device const float* b_scale, device const T* bias,
                                      device T* y, uint depth, uint has_bias, int act,
                                      uint gid) {
  const uint row = gid / depth;
  const uint col = gid - row * depth;
  float v = (float)c[gid] * precise::divide(1.0f, a_scale[row]);
  v = precise::divide(v, b_scale[col]);
  if (has_bias != 0u)
    v += (float)bias[col];
  y[gid] = (T)ct2_apply_activation(v, act);
}

kernel void ct2_dequant_gemm_out_float(device const int* c        [[buffer(0)]],
                                       device const float* a_scale [[buffer(1)]],
                                       device const float* b_scale [[buffer(2)]],
                                       device const float* bias    [[buffer(3)]],
                                       device float* y             [[buffer(4)]],
                                       constant uint& depth         [[buffer(5)]],
                                       constant uint& has_bias      [[buffer(6)]],
                                       constant int& act            [[buffer(7)]],
                                       uint gid [[thread_position_in_grid]]) {
  ct2_dequant_gemm_out_impl<float>(c, a_scale, b_scale, bias, y, depth, has_bias, act, gid);
}

kernel void ct2_dequant_gemm_out_half(device const int* c        [[buffer(0)]],
                                      device const float* a_scale [[buffer(1)]],
                                      device const float* b_scale [[buffer(2)]],
                                      device const half* bias     [[buffer(3)]],
                                      device half* y              [[buffer(4)]],
                                      constant uint& depth         [[buffer(5)]],
                                      constant uint& has_bias      [[buffer(6)]],
                                      constant int& act            [[buffer(7)]],
                                      uint gid [[thread_position_in_grid]]) {
  ct2_dequant_gemm_out_impl<half>(c, a_scale, b_scale, bias, y, depth, has_bias, act, gid);
}

// ---- INT8 GEMM (native) ----  C(int32, m*n) = alpha * op(A) * op(B), beta == 0 (the
// only form CT2's quantized Dense and the int8 Gemm tests use; alpha is integral so the
// product stays exact). int8 stays int8 end-to-end: operands are staged through
// threadgroup memory as char and every multiply-accumulate runs in int32 — bit-exact by
// construction, no float detour. MPS has no integer GEMM and simdgroup_matrix has no
// int8 element type (MSL spec 2.4: half/bfloat/float only), so this is hand-tiled.
//
// Each 16x16 threadgroup computes a 64x64 tile of C, looping over k in 32-deep chunks;
// each thread accumulates a 4x4 register micro-tile. Both staging tiles are stored
// depth-major ([kk][i] / [kk][j]) so the inner loop reads both contiguously regardless
// of the transpose flags, which are resolved once at tile-load time. Out-of-range loads
// stage 0 (a no-op in the dot product), so only the C store needs an edge guard.
constant uint CT2_GEMM_S8_BM = 64;
constant uint CT2_GEMM_S8_BN = 64;
constant uint CT2_GEMM_S8_BK = 32;

kernel void ct2_gemm_s8(device const char* a [[buffer(0)]],
                        device const char* b [[buffer(1)]],
                        device int* c         [[buffer(2)]],
                        constant uint& m      [[buffer(3)]],
                        constant uint& n      [[buffer(4)]],
                        constant uint& k      [[buffer(5)]],
                        constant uint& lda    [[buffer(6)]],
                        constant uint& ldb    [[buffer(7)]],
                        constant uint& ldc    [[buffer(8)]],
                        constant uint& trans_a [[buffer(9)]],
                        constant uint& trans_b [[buffer(10)]],
                        constant int& alpha    [[buffer(11)]],
                        uint2 tg  [[threadgroup_position_in_grid]],
                        uint2 tid [[thread_position_in_threadgroup]]) {
  threadgroup char As[CT2_GEMM_S8_BK][CT2_GEMM_S8_BM];  // As[kk][i]: A rows, depth-major
  threadgroup char Bs[CT2_GEMM_S8_BK][CT2_GEMM_S8_BN];  // Bs[kk][j]: B cols, depth-major

  const uint row0 = tg.y * CT2_GEMM_S8_BM;
  const uint col0 = tg.x * CT2_GEMM_S8_BN;
  const uint lin = tid.y * 16u + tid.x;  // 0..255

  int4 acc[4] = {int4(0), int4(0), int4(0), int4(0)};  // acc[r][s]: micro-tile row r, col s

  for (uint k0 = 0; k0 < k; k0 += CT2_GEMM_S8_BK) {
    for (uint t = lin; t < CT2_GEMM_S8_BM * CT2_GEMM_S8_BK; t += 256u) {
      const uint i = t / CT2_GEMM_S8_BK;
      const uint kk = t - i * CT2_GEMM_S8_BK;
      const uint gi = row0 + i;
      const uint gk = k0 + kk;
      char v = 0;
      if (gi < m && gk < k)
        v = (trans_a != 0u) ? a[(ulong)gk * lda + gi] : a[(ulong)gi * lda + gk];
      As[kk][i] = v;
    }
    for (uint t = lin; t < CT2_GEMM_S8_BK * CT2_GEMM_S8_BN; t += 256u) {
      const uint kk = t / CT2_GEMM_S8_BN;
      const uint j = t - kk * CT2_GEMM_S8_BN;
      const uint gj = col0 + j;
      const uint gk = k0 + kk;
      char v = 0;
      if (gj < n && gk < k)
        v = (trans_b != 0u) ? b[(ulong)gj * ldb + gk] : b[(ulong)gk * ldb + gj];
      Bs[kk][j] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The depth-major tiles make both operand reads contiguous char4s (tid.{x,y}*4 is
    // 4-aligned and the tile rows are 64-byte aligned), 4 MACs per int4 op.
    for (uint kk = 0; kk < CT2_GEMM_S8_BK; ++kk) {
      const int4 av = int4(*(const threadgroup char4*)(&As[kk][tid.y * 4u]));
      const int4 bv = int4(*(const threadgroup char4*)(&Bs[kk][tid.x * 4u]));
      acc[0] += av.x * bv;
      acc[1] += av.y * bv;
      acc[2] += av.z * bv;
      acc[3] += av.w * bv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint r = 0; r < 4u; ++r) {
    const uint gi = row0 + tid.y * 4u + r;
    if (gi >= m)
      continue;
    for (uint s = 0; s < 4u; ++s) {
      const uint gj = col0 + tid.x * 4u + s;
      if (gj < n)
        c[(ulong)gi * ldc + gj] = alpha * acc[r][s];
    }
  }
}

// ---- INT8 GEMV ----  the small-m fast path (autoregressive decode runs every Dense at
// m = batch). The tiled kernel wastes 63/64 of its A-tile there; this one assigns one
// SIMD-group per output element instead: lanes stride the k axis in char4 steps and a
// single simd_sum folds the int32 partials. Dense layout only (!trans_a && trans_b, so
// both the A row and the B row are k-contiguous); the host routes here only when k and
// the operand alignments allow the char4 reinterpretation. This regime is memory-bound,
// and int8 moves half the bytes of fp16 — it beats the float GEMM rather than losing.
kernel void ct2_gemv_s8(device const char* a [[buffer(0)]],
                        device const char* b [[buffer(1)]],
                        device int* c         [[buffer(2)]],
                        constant uint& n      [[buffer(3)]],
                        constant uint& k      [[buffer(4)]],
                        constant uint& lda    [[buffer(5)]],
                        constant uint& ldb    [[buffer(6)]],
                        constant uint& ldc    [[buffer(7)]],
                        constant int& alpha   [[buffer(8)]],
                        uint2 tg  [[threadgroup_position_in_grid]],
                        uint simd_size [[threads_per_simdgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint i = tg.y;
  const uint j = tg.x * 8u + simd_group;  // 8 SIMD-groups per threadgroup (see host)
  if (j >= n)
    return;  // uniform per SIMD-group, so the simd_sum below stays in uniform control flow

  device const char4* a4 = (device const char4*)(a + (ulong)i * lda);
  device const char4* b4 = (device const char4*)(b + (ulong)j * ldb);
  const uint kvec = k / 4u;

  int acc = 0;
  for (uint v = simd_lane; v < kvec; v += simd_size) {
    const int4 av = int4(a4[v]);
    const int4 bv = int4(b4[v]);
    acc += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
  }
  acc = simd_sum(acc);
  if (simd_lane == 0u)
    c[(ulong)i * ldc + j] = alpha * acc;
}

// ---- Strided copy ----  type-agnostic byte copy underlying Concat/Split/Slide:
// dst[i*dst_step + d] = src[i*src_step + d] for i in [0,iter), d in [0,copy_size) (bytes).
kernel void ct2_strided_copy_bytes(device const uchar* src   [[buffer(0)]],
                                   device uchar* dst          [[buffer(1)]],
                                   constant uint& copy_size    [[buffer(2)]],
                                   constant uint& src_step     [[buffer(3)]],
                                   constant uint& dst_step     [[buffer(4)]],
                                   uint g [[thread_position_in_grid]]) {
  const uint i = g / copy_size;
  const uint d = g - i * copy_size;
  dst[(ulong)i * dst_step + d] = src[(ulong)i * src_step + d];
}

// ---- Gather ----  type-agnostic byte copy, parallel across rows AND within each row.
// output[i] = data[batch_of(i) * batch_stride + indices[i] * copy_size], copy_size bytes.
// The grid is 2D: g.y indexes the gathered row, g.x a CT2_GATHER_CHUNK-byte chunk of it.
// Beam-search KV-cache reorder gathers a handful of ~megabyte rows; a one-thread-per-row
// copy leaves the GPU nearly idle there (it was the dominant decode cost for Whisper).
#define CT2_GATHER_CHUNK 16u
kernel void ct2_gather_bytes(device const uchar* data    [[buffer(0)]],
                             device const int* indices    [[buffer(1)]],
                             device uchar* output         [[buffer(2)]],
                             constant uint& copy_size      [[buffer(3)]],
                             constant uint& batch_stride   [[buffer(4)]],
                             constant uint& num_per_batch  [[buffer(5)]],
                             uint2 g [[thread_position_in_grid]]) {
  const uint i = g.y;
  const uint offset = g.x * CT2_GATHER_CHUNK;
  if (offset >= copy_size)
    return;
  const uint batch = i / num_per_batch;
  const uint read = (uint)indices[i];
  device const uchar* src = data + (ulong)batch * (ulong)batch_stride
                            + (ulong)read * (ulong)copy_size + offset;
  device uchar* dst = output + (ulong)i * (ulong)copy_size + offset;
  const uint n = min(copy_size - offset, CT2_GATHER_CHUNK);
  for (uint b = 0; b < n; ++b)
    dst[b] = src[b];
}

// ---- Transpose ----  type-agnostic permute for ranks <= 4 (leading dims padded to 1).
// One thread per element: g is the output linear index, decomposed into output coords;
// the input offset applies the permuted input strides (element units). For permutations
// that keep the innermost axis (e.g. split/combine heads' {0,2,1,3}) consecutive threads
// read consecutive input elements, so loads stay coalesced.
template <typename T>
inline void ct2_transpose_impl(device const T* x, device T* y,
                               uint4 dims, uint4 strides, uint g) {
  uint r = g;
  const uint i3 = r % dims.w; r /= dims.w;
  const uint i2 = r % dims.z; r /= dims.z;
  const uint i1 = r % dims.y;
  const uint i0 = r / dims.y;
  y[g] = x[(ulong)i0 * strides.x + (ulong)i1 * strides.y
           + (ulong)i2 * strides.z + (ulong)i3 * strides.w];
}

kernel void ct2_transpose_b1(device const uchar* x [[buffer(0)]],
                             device uchar* y [[buffer(1)]],
                             constant uint4& dims [[buffer(2)]],
                             constant uint4& strides [[buffer(3)]],
                             uint g [[thread_position_in_grid]]) {
  ct2_transpose_impl<uchar>(x, y, dims, strides, g);
}

kernel void ct2_transpose_b2(device const ushort* x [[buffer(0)]],
                             device ushort* y [[buffer(1)]],
                             constant uint4& dims [[buffer(2)]],
                             constant uint4& strides [[buffer(3)]],
                             uint g [[thread_position_in_grid]]) {
  ct2_transpose_impl<ushort>(x, y, dims, strides, g);
}

kernel void ct2_transpose_b4(device const uint* x [[buffer(0)]],
                             device uint* y [[buffer(1)]],
                             constant uint4& dims [[buffer(2)]],
                             constant uint4& strides [[buffer(3)]],
                             uint g [[thread_position_in_grid]]) {
  ct2_transpose_impl<uint>(x, y, dims, strides, g);
}

// ---- Sampling ops ----

// "a ranks before b" in the shared (value desc, index asc) total order used by TopK and
// TopPMask. index -1 marks "no candidate" and ranks after everything.
inline bool ct2_rank_before(float va, int ia, float vb, int ib) {
  if (ib < 0) return ia >= 0;
  if (ia < 0) return false;
  return va > vb || (va == vb && ia < ib);
}

// TopK: one threadgroup per row, k extract-max passes over the row. Each pass scans the
// row for the best element ranking strictly after the previously selected one, then
// tree-reduces. Values are re-read from the input at the selected index so the output is
// a bit-copy (half stays half). Tie-break is deterministic (smaller index first); the CPU
// partial_sort leaves tie order unspecified, so parity on ties is not promised. The
// host dispatches a power-of-two threadgroup size <= CT2_TOPK_MAX_TG.
constant uint CT2_TOPK_MAX_TG = 1024;

template <typename T>
inline void ct2_topk_impl(device const T* x, device T* values, device int* indices,
                          uint depth, uint k, uint row, uint tid, uint tg,
                          threadgroup float* red_v, threadgroup int* red_i,
                          threadgroup float* bound_v, threadgroup int* bound_i) {
  device const T* x_row = x + (ulong)row * (ulong)depth;
  device T* v_row = values + (ulong)row * (ulong)k;
  device int* i_row = indices + (ulong)row * (ulong)k;

  if (tid == 0u) {
    *bound_v = INFINITY;
    *bound_i = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint it = 0; it < k; ++it) {
    const float bv = *bound_v;
    const int bi = *bound_i;

    float best_v = 0.0f;
    int best_i = -1;
    for (uint j = tid; j < depth; j += tg) {
      const float v = (float)x_row[j];
      // Candidate iff it ranks strictly after the current bound (bi < 0 is the initial
      // bound, before everything).
      const bool after_bound = bi < 0 || v < bv || (v == bv && (int)j > bi);
      if (after_bound && ct2_rank_before(v, (int)j, best_v, best_i)) {
        best_v = v;
        best_i = (int)j;
      }
    }
    red_v[tid] = best_v;
    red_i[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2u; s > 0u; s >>= 1) {
      if (tid < s && ct2_rank_before(red_v[tid + s], red_i[tid + s], red_v[tid], red_i[tid])) {
        red_v[tid] = red_v[tid + s];
        red_i[tid] = red_i[tid + s];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
      const int sel = red_i[0] >= 0 ? red_i[0] : 0;  // k <= depth guarantees a candidate
      v_row[it] = x_row[sel];
      i_row[it] = sel;
      *bound_v = red_v[0];
      *bound_i = red_i[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void ct2_topk_float(device const float* x   [[buffer(0)]],
                           device float* values     [[buffer(1)]],
                           device int* indices      [[buffer(2)]],
                           constant uint& depth      [[buffer(3)]],
                           constant uint& k          [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tg  [[threads_per_threadgroup]]) {
  threadgroup float red_v[CT2_TOPK_MAX_TG];
  threadgroup int red_i[CT2_TOPK_MAX_TG];
  threadgroup float bound_v;
  threadgroup int bound_i;
  ct2_topk_impl<float>(x, values, indices, depth, k, row, tid, tg,
                       red_v, red_i, &bound_v, &bound_i);
}

kernel void ct2_topk_half(device const half* x   [[buffer(0)]],
                          device half* values     [[buffer(1)]],
                          device int* indices     [[buffer(2)]],
                          constant uint& depth     [[buffer(3)]],
                          constant uint& k         [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg  [[threads_per_threadgroup]]) {
  threadgroup float red_v[CT2_TOPK_MAX_TG];
  threadgroup int red_i[CT2_TOPK_MAX_TG];
  threadgroup float bound_v;
  threadgroup int bound_i;
  ct2_topk_impl<half>(x, values, indices, depth, k, row, tid, tg,
                      red_v, red_i, &bound_v, &bound_i);
}

// Order-preserving float <-> uint mapping (u1 < u2 iff f1 < f2) for sort keys.
inline uint ct2_float_order_key(float f) {
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

inline float ct2_float_from_order_key(uint key) {
  const uint u = (key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key;
  return as_type<float>(u);
}

// TopPMask: one threadgroup per row, whole row sorted in threadgroup memory (depth must
// be <= CT2_TOPP_MAX_DEPTH; the op falls back to the CPU kernel above that). Mirrors the
// CPU reference exactly: sort by probability descending, then a single thread accumulates
// the exclusive prefix sum sequentially in float — same addition order and rounding as
// the CPU loop — and y[id] = prefix < p ? x[id] : mask. Tie order is (prob desc, index
// asc), deterministic; the CPU std::sort tie order is unspecified.
constant uint CT2_TOPP_TG = 256;
constant uint CT2_TOPP_MAX_DEPTH = 4096;

template <typename T>
inline void ct2_topp_mask_impl(device const T* x, device const T* probs, device T* y,
                               float p, float mask, uint depth, uint row, uint tid,
                               threadgroup uint* keys, threadgroup ushort* ids,
                               threadgroup uint* kept_count) {
  device const T* x_row = x + (ulong)row * (ulong)depth;
  device const T* probs_row = probs + (ulong)row * (ulong)depth;
  device T* y_row = y + (ulong)row * (ulong)depth;

  // Padded power-of-two size for the bitonic network.
  uint n = 1;
  while (n < depth)
    n <<= 1;

  for (uint i = tid; i < n; i += CT2_TOPP_TG) {
    if (i < depth) {
      keys[i] = ct2_float_order_key((float)probs_row[i]);
      ids[i] = (ushort)i;
    } else {
      keys[i] = 0u;          // ranks after every real probability (probs are >= 0)
      ids[i] = 0xFFFFu;      // and after real elements on a key tie
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Bitonic sort, descending by (key, then id ascending).
  for (uint size = 2; size <= n; size <<= 1) {
    for (uint stride = size >> 1; stride > 0; stride >>= 1) {
      for (uint i = tid; i < n; i += CT2_TOPP_TG) {
        const uint partner = i ^ stride;
        if (partner > i) {
          const bool descending = (i & size) == 0u;
          const uint k1 = keys[i], k2 = keys[partner];
          const ushort d1 = ids[i], d2 = ids[partner];
          const bool first_ranks_before = k1 > k2 || (k1 == k2 && d1 < d2);
          if (first_ranks_before != descending) {
            keys[i] = k2; keys[partner] = k1;
            ids[i] = d2; ids[partner] = d1;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Sequential exclusive prefix walk, same float accumulation order as the CPU loop.
  if (tid == 0u) {
    uint kept = depth;
    float total_p = 0.0f;
    for (uint r = 0; r < depth; ++r) {
      if (!(total_p < p)) {
        kept = r;
        break;
      }
      total_p += ct2_float_from_order_key(keys[r]);
    }
    *kept_count = kept;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint kept = *kept_count;
  for (uint r = tid; r < depth; r += CT2_TOPP_TG) {
    const uint id = (uint)ids[r];
    y_row[id] = r < kept ? x_row[id] : (T)mask;
  }
}

kernel void ct2_topp_mask_float(device const float* x      [[buffer(0)]],
                                device const float* probs   [[buffer(1)]],
                                device float* y             [[buffer(2)]],
                                constant uint& depth         [[buffer(3)]],
                                constant float& p            [[buffer(4)]],
                                constant float& mask         [[buffer(5)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]]) {
  threadgroup uint keys[CT2_TOPP_MAX_DEPTH];
  threadgroup ushort ids[CT2_TOPP_MAX_DEPTH];
  threadgroup uint kept_count;
  ct2_topp_mask_impl<float>(x, probs, y, p, mask, depth, row, tid, keys, ids, &kept_count);
}

kernel void ct2_topp_mask_half(device const half* x      [[buffer(0)]],
                               device const half* probs   [[buffer(1)]],
                               device half* y             [[buffer(2)]],
                               constant uint& depth        [[buffer(3)]],
                               constant float& p           [[buffer(4)]],
                               constant float& mask        [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]]) {
  threadgroup uint keys[CT2_TOPP_MAX_DEPTH];
  threadgroup ushort ids[CT2_TOPP_MAX_DEPTH];
  threadgroup uint kept_count;
  ct2_topp_mask_impl<half>(x, probs, y, p, mask, depth, row, tid, keys, ids, &kept_count);
}

// Multinomial (one sample per row): inverse-CDF sampling. The uniform draw u in (0, 1]
// comes from the host (the CT2 random generator, so set_random_seed reproducibility
// holds); the kernel finds the smallest index whose inclusive prefix sum reaches
// u * sum(probs). Thread t owns the contiguous chunk [t*chunk, (t+1)*chunk); chunk sums
// are combined by thread 0 so the prefix order is the row order; the min-reduce resolves
// any rounding ambiguity at chunk boundaries to the earliest crossing.
constant uint CT2_MULTINOMIAL_TG = 256;

template <typename T>
inline void ct2_multinomial_impl(device const T* probs, device const float* uniforms,
                                 device int* output, uint depth, uint row, uint tid,
                                 threadgroup float* chunk_sum, threadgroup float* chunk_prefix,
                                 threadgroup float* target, threadgroup int* red_i) {
  device const T* probs_row = probs + (ulong)row * (ulong)depth;

  const uint chunk = (depth + CT2_MULTINOMIAL_TG - 1u) / CT2_MULTINOMIAL_TG;
  const uint begin = tid * chunk;
  const uint end = min(begin + chunk, depth);

  float s = 0.0f;
  for (uint i = begin; i < end; ++i)
    s += (float)probs_row[i];
  chunk_sum[tid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    float running = 0.0f;
    for (uint t = 0; t < CT2_MULTINOMIAL_TG; ++t) {
      chunk_prefix[t] = running;
      running += chunk_sum[t];
    }
    *target = uniforms[row] * running;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float tgt = *target;
  int candidate = 0x7FFFFFFF;
  if (begin < end && chunk_prefix[tid] < tgt) {
    float run = chunk_prefix[tid];
    for (uint i = begin; i < end; ++i) {
      run += (float)probs_row[i];
      if (run >= tgt) {
        candidate = (int)i;
        break;
      }
    }
  }
  red_i[tid] = candidate;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint sred = CT2_MULTINOMIAL_TG / 2u; sred > 0u; sred >>= 1) {
    if (tid < sred)
      red_i[tid] = min(red_i[tid], red_i[tid + sred]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u)
    output[row] = red_i[0] != 0x7FFFFFFF ? red_i[0] : (int)depth - 1;
}

kernel void ct2_multinomial_float(device const float* probs    [[buffer(0)]],
                                  device const float* uniforms  [[buffer(1)]],
                                  device int* output            [[buffer(2)]],
                                  constant uint& depth           [[buffer(3)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float chunk_sum[CT2_MULTINOMIAL_TG];
  threadgroup float chunk_prefix[CT2_MULTINOMIAL_TG];
  threadgroup float target;
  threadgroup int red_i[CT2_MULTINOMIAL_TG];
  ct2_multinomial_impl<float>(probs, uniforms, output, depth, row, tid,
                              chunk_sum, chunk_prefix, &target, red_i);
}

kernel void ct2_multinomial_half(device const half* probs     [[buffer(0)]],
                                 device const float* uniforms  [[buffer(1)]],
                                 device int* output            [[buffer(2)]],
                                 constant uint& depth           [[buffer(3)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float chunk_sum[CT2_MULTINOMIAL_TG];
  threadgroup float chunk_prefix[CT2_MULTINOMIAL_TG];
  threadgroup float target;
  threadgroup int red_i[CT2_MULTINOMIAL_TG];
  ct2_multinomial_impl<half>(probs, uniforms, output, depth, row, tid,
                             chunk_sum, chunk_prefix, &target, red_i);
}

// Gumbel-max noise: y[i] = x[i] + (-log(u_i)), u_i ~ U(0, 1), matching the CPU and CUDA
// add_gumbel_noise semantics. The RNG is counter-based (splitmix64 of seed + index) with
// a per-launch seed drawn from the host CT2 generator: a different stream than the CPU
// std::mt19937 (bit-parity with the CPU draw order is not meaningful), but deterministic
// under set_random_seed. u is built from 24 random bits offset by 0.5, so u is in (0, 1)
// and -log(u) is finite (same finite-tail property as curand_uniform on CUDA).
inline float ct2_gumbel_noise(ulong seed, uint gid) {
  ulong z = seed + (ulong)gid * 0x9E3779B97F4A7C15UL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
  z = z ^ (z >> 31);
  const float u = ((float)(uint)(z >> 40) + 0.5f) * 0x1p-24f;
  return -log(u);
}

kernel void ct2_gumbel_noise_float(device const float* x [[buffer(0)]],
                                   device float* y        [[buffer(1)]],
                                   constant ulong& seed    [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
  y[gid] = x[gid] + ct2_gumbel_noise(seed, gid);
}

kernel void ct2_gumbel_noise_half(device const half* x [[buffer(0)]],
                                  device half* y        [[buffer(1)]],
                                  constant ulong& seed   [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
  y[gid] = (half)((float)x[gid] + ct2_gumbel_noise(seed, gid));
}
)MSL";
    }

  }
}
