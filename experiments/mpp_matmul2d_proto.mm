// Standalone prototype: mpp::tensor_ops::matmul2d (Metal 4 MPP) int8 x int8 -> int32.
// Answers, in order:
//   1. Does newLibraryWithSource compile MSL that includes
//      <MetalPerformancePrimitives/MetalPerformancePrimitives.h> at MTLLanguageVersion4_0?
//   2. Is the result int32-EXACT vs a host triple loop (k=2048, random int8)?
//   3. How fast is it on the BenchmarkGemmInt8 shapes vs the recorded ct2_gemm_s8 baseline?
//
// Layout under test = CT2's quantized Dense: A row-major MxK, B row-major NxK
// (transpose_right), C row-major MxN int32.
//
// Build:
//   clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
//     experiments/mpp_matmul2d_proto.mm -o /tmp/mpp_proto && /tmp/mpp_proto

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static const char* kSource = R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Tile sizes: one threadgroup of 4 SIMD-groups computes a TM x TN output tile.
constant constexpr int TM = 64;
constant constexpr int TN = 32;

kernel void mpp_gemm_s8_nt(device const char* a [[buffer(0)]],
                           device const char* b [[buffer(1)]],
                           device int* c [[buffer(2)]],
                           constant uint& m [[buffer(3)]],
                           constant uint& n [[buffer(4)]],
                           constant uint& k [[buffer(5)]],
                           constant uint& lda [[buffer(6)]],
                           constant uint& ldb [[buffer(7)]],
                           constant uint& ldc [[buffer(8)]],
                           uint2 tgid [[threadgroup_position_in_grid]])
{
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                            /*transpose_left=*/false,
                                            /*transpose_right=*/true,
                                            /*relaxed_precision=*/false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;

  // tensor dims: dim 0 = fastest-moving. A is MxK row-major -> (k, m) with row stride lda.
  // B is NxK row-major; transpose_right reads it as KxN -> (k, n) with row stride ldb.
  // C is MxN int32 row-major -> (n, m) with row stride ldc.
  // The MPP dispatch matches element types EXACTLY (int8_t/int32_t, non-const) — char or
  // const int8_t hits the "Unsupported type" static_assert.
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>
      tA((device int8_t*)a, dextents<int32_t, 2>(int(k), int(m)), array<int32_t, 2>{1, int(lda)});
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>
      tB((device int8_t*)b, dextents<int32_t, 2>(int(k), int(n)), array<int32_t, 2>{1, int(ldb)});
  tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>
      tC(c, dextents<int32_t, 2>(int(n), int(m)), array<int32_t, 2>{1, int(ldc)});

  auto sA = tA.slice(0, int(tgid.y) * TM);
  auto sB = tB.slice(0, int(tgid.x) * TN);
  auto sC = tC.slice(int(tgid.x) * TN, int(tgid.y) * TM);
  op.run(sA, sB, sC);
}
)MSL";

struct Shape { int m, n, k; };

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    printf("device: %s\n", dev.name.UTF8String);

    // --- 1. compile at language version 4.0 ---
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = (MTLLanguageVersion)((4 << 16) + 0);  // MTLLanguageVersion4_0
    opts.mathMode = MTLMathModeSafe;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                           options:opts
                                             error:&err];
    if (!lib) {
      printf("COMPILE FAILED: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    printf("compile at MSL 4.0 via newLibraryWithSource: OK\n");

    id<MTLFunction> fn = [lib newFunctionWithName:@"mpp_gemm_s8_nt"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      printf("PSO FAILED: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    printf("pipeline: OK (threadExecutionWidth=%lu, maxTotalThreads=%lu)\n",
           (unsigned long)pso.threadExecutionWidth,
           (unsigned long)pso.maxTotalThreadsPerThreadgroup);

    const int TM = 64, TN = 32;
    auto run_gemm = [&](id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
                        int m, int n, int k, bool wait) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:a offset:0 atIndex:0];
      [enc setBuffer:b offset:0 atIndex:1];
      [enc setBuffer:c offset:0 atIndex:2];
      uint32_t mu = m, nu = n, ku = k, lda = k, ldb = k, ldc = n;
      [enc setBytes:&mu length:4 atIndex:3];
      [enc setBytes:&nu length:4 atIndex:4];
      [enc setBytes:&ku length:4 atIndex:5];
      [enc setBytes:&lda length:4 atIndex:6];
      [enc setBytes:&ldb length:4 atIndex:7];
      [enc setBytes:&ldc length:4 atIndex:8];
      MTLSize grid = MTLSizeMake((n + TN - 1) / TN, (m + TM - 1) / TM, 1);
      MTLSize group = MTLSizeMake(pso.threadExecutionWidth * 4, 1, 1);
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
      [enc endEncoding];
      [cb commit];
      if (wait) [cb waitUntilCompleted];
    };

    // --- 2. exactness: asymmetric shape catches coordinate swaps; k=2048 catches
    //        accumulator narrowing; values span the full int8 range. Also a
    //        saturation-style case and edge (non-tile-multiple) sizes. ---
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d(-128, 127);
    for (Shape s : {Shape{96, 160, 2048}, Shape{67, 51, 333}, Shape{1, 33, 64},
                    Shape{200, 31, 100}}) {
      std::vector<int8_t> av(size_t(s.m) * s.k), bv(size_t(s.n) * s.k);
      for (auto& v : av) v = (int8_t)d(rng);
      for (auto& v : bv) v = (int8_t)d(rng);
      id<MTLBuffer> a = [dev newBufferWithBytes:av.data() length:av.size()
                                        options:MTLResourceStorageModeShared];
      id<MTLBuffer> b = [dev newBufferWithBytes:bv.data() length:bv.size()
                                        options:MTLResourceStorageModeShared];
      // Pre-fill C with garbage to learn whether mode::multiply overwrites or accumulates.
      std::vector<int32_t> garbage(size_t(s.m) * s.n, 0x7777);
      id<MTLBuffer> c = [dev newBufferWithBytes:garbage.data()
                                         length:garbage.size() * 4
                                        options:MTLResourceStorageModeShared];
      run_gemm(a, b, c, s.m, s.n, s.k, true);

      const int32_t* got = (const int32_t*)c.contents;
      long mismatches = 0;
      int32_t first_got = 0, first_want = 0;
      long first_at = -1;
      for (int i = 0; i < s.m; ++i)
        for (int j = 0; j < s.n; ++j) {
          int64_t acc = 0;
          for (int kk = 0; kk < s.k; ++kk)
            acc += int32_t(av[size_t(i) * s.k + kk]) * int32_t(bv[size_t(j) * s.k + kk]);
          int32_t want = (int32_t)acc;
          if (got[size_t(i) * s.n + j] != want) {
            if (first_at < 0) {
              first_at = size_t(i) * s.n + j;
              first_got = got[size_t(i) * s.n + j];
              first_want = want;
            }
            ++mismatches;
          }
        }
      printf("exactness m=%d n=%d k=%d: %s", s.m, s.n, s.k,
             mismatches == 0 ? "EXACT\n" : "");
      if (mismatches)
        printf("FAILED (%ld mismatches; first at %ld: got %d want %d)\n",
               mismatches, first_at, first_got, first_want);
    }

    // --- 3. throughput on the BenchmarkGemmInt8 shapes (flush per iter, like the
    //        in-tree benchmark; first run untimed warms up). ---
    printf("\nshape                    ms/iter    GFLOPS (2mnk)\n");
    for (Shape s : {Shape{256, 256, 256}, Shape{1024, 1024, 1024},
                    Shape{2048, 2048, 2048}, Shape{256, 4864, 896},
                    Shape{256, 896, 4864}}) {
      const int iters = (int64_t(s.m) * s.n * s.k > (int64_t)1 << 29) ? 8 : 30;
      std::vector<int8_t> av(size_t(s.m) * s.k, 3), bv(size_t(s.n) * s.k, -5);
      id<MTLBuffer> a = [dev newBufferWithBytes:av.data() length:av.size()
                                        options:MTLResourceStorageModeShared];
      id<MTLBuffer> b = [dev newBufferWithBytes:bv.data() length:bv.size()
                                        options:MTLResourceStorageModeShared];
      id<MTLBuffer> c = [dev newBufferWithLength:size_t(s.m) * s.n * 4
                                         options:MTLResourceStorageModeShared];
      run_gemm(a, b, c, s.m, s.n, s.k, true);  // warmup
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i)
        run_gemm(a, b, c, s.m, s.n, s.k, true);
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
      double gflops = 2.0 * s.m * s.n * double(s.k) / (ms * 1e6);
      printf("m=%-5d n=%-6d k=%-5d  %8.3f   %8.1f\n", s.m, s.n, s.k, ms, gflops);
    }
  }
  return 0;
}
