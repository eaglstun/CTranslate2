#pragma once

// Metal 4 (MSL 4.0) kernel source, compiled at runtime into a SEPARATE library from
// get_kernels_source(): mpp::tensor_ops requires languageVersion 4.0 (macOS 26+), and
// the main library must keep compiling on older OSes. device.mm gates this library
// behind an @available check and falls back silently when compilation is unavailable.
//
// ct2_mpp_gemm_s8_nt: int8 x int8 -> int32 GEMM via Metal Performance Primitives
// matmul2d, for the quantized Dense layout (A row-major MxK, B row-major NxK i.e.
// transpose_right, C row-major MxN). The i8/i8 -> i32 combination is in the MPP
// supported-type table and was measured int32-EXACT against a host triple loop
// (k=2048, full int8 range), so it upholds the same exactness contract as
// ct2_gemm_s8 — see METAL_BACKEND.md.
//
// Tile choice (M4 Max, 2026-06-11 sweep over {16..128}x{32..256} x {1,2,4,8}
// SIMD-groups): 16x64 tiles on TWO cooperating SIMD-groups. Wider scopes (the 4
// SIMD-groups Apple's header example uses) were 2-5x slower on every shape measured;
// per-threadgroup tiles >=128 wide collapsed on deep-k shapes. 2048^3 throughput with
// this config matches MPS fp16 GEMM (~11.5 TFLOPS effective).

namespace ctranslate2 {
  namespace metal {

    inline const char* get_mpp_kernels_source() {
      return R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Dispatch in primitives.mm must match: grid = (ceil(n/TN), ceil(m/TM), 1),
// threadgroup = (threadExecutionWidth * SG, 1, 1).
constant constexpr int CT2_MPP_GEMM_S8_TM = 16;
constant constexpr int CT2_MPP_GEMM_S8_TN = 64;
constant constexpr int CT2_MPP_GEMM_S8_SG = 2;

kernel void ct2_mpp_gemm_s8_nt(device const char* a [[buffer(0)]],
                               device const char* b [[buffer(1)]],
                               device int* c [[buffer(2)]],
                               constant uint& m [[buffer(3)]],
                               constant uint& n [[buffer(4)]],
                               constant uint& k [[buffer(5)]],
                               constant uint& lda [[buffer(6)]],
                               constant uint& ldb [[buffer(7)]],
                               constant uint& ldc [[buffer(8)]],
                               uint2 tgid [[threadgroup_position_in_grid]]) {
  constexpr int TM = CT2_MPP_GEMM_S8_TM;
  constexpr int TN = CT2_MPP_GEMM_S8_TN;

  // mode::multiply overwrites C (no read of prior contents), matching the
  // alpha=1/beta=0 contract of the routing in primitives.mm.
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                            /*transpose_left=*/false,
                                            /*transpose_right=*/true,
                                            /*relaxed_precision=*/false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<CT2_MPP_GEMM_S8_SG>> op;

  // Inline tensors over the raw buffer args; dim 0 is the fastest-moving axis.
  // The MPP dispatch matches element types EXACTLY — int8_t/int32_t, non-const
  // (char or const int8_t hit a "Unsupported type" static_assert).
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>
      ta((device int8_t*)a, dextents<int32_t, 2>(int(k), int(m)),
         array<int32_t, 2>{1, int(lda)});
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>
      tb((device int8_t*)b, dextents<int32_t, 2>(int(k), int(n)),
         array<int32_t, 2>{1, int(ldb)});
  tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>
      tc(c, dextents<int32_t, 2>(int(n), int(m)), array<int32_t, 2>{1, int(ldc)});

  const int row0 = int(tgid.y) * TM;
  const int col0 = int(tgid.x) * TN;
  if (row0 + TM <= int(m) && col0 + TN <= int(n)) {
    // Interior tile: static extents let the op skip bounds checks. (The MPP header
    // comment calls this static_slice; the shipping stdlib spells it slice<Extents...>.)
    auto sa = ta.slice<dynamic_extent, TM>(0, row0);
    auto sb = tb.slice<dynamic_extent, TN>(0, col0);
    auto sc = tc.slice<TN, TM>(col0, row0);
    op.run(sa, sb, sc);
  } else {
    auto sa = ta.slice(0, row0);
    auto sb = tb.slice(0, col0);
    auto sc = tc.slice(col0, row0);
    op.run(sa, sb, sc);
  }
}
)MSL";
    }

  }
}
