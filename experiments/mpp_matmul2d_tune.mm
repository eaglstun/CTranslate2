// Tile-size + edge-split sweep for mpp::tensor_ops::matmul2d int8x8->int32.
// v1 (mpp_matmul2d_proto.mm) established: compiles via newLibraryWithSource at MSL 4.0,
// int32-EXACT, but naive slice() at 64x32 is barely faster than ct2_gemm_s8.
// This sweep: tile sizes x simdgroup counts, with the header-recommended
// inside/edge static_slice split, on the BenchmarkGemmInt8 prefill shapes.
//
// Build:
//   clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
//     experiments/mpp_matmul2d_tune.mm -o /tmp/mpp_tune && /tmp/mpp_tune

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// %d slots: TM, TN, SG (simdgroups per threadgroup)
static const char* kTemplate = R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int TM = %d;
constant constexpr int TN = %d;
constant constexpr int SG = %d;

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
                                            false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>
      tA((device int8_t*)a, dextents<int32_t, 2>(int(k), int(m)), array<int32_t, 2>{1, int(lda)});
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>
      tB((device int8_t*)b, dextents<int32_t, 2>(int(k), int(n)), array<int32_t, 2>{1, int(ldb)});
  tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>
      tC(c, dextents<int32_t, 2>(int(n), int(m)), array<int32_t, 2>{1, int(ldc)});

  const int row0 = int(tgid.y) * TM;
  const int col0 = int(tgid.x) * TN;
  if (row0 + TM <= int(m) && col0 + TN <= int(n)) {
    // Inside tile: static extents let the op skip bounds checks. (The MPP header
    // comment calls this static_slice; the shipping stdlib spells it slice<Extents...>.)
    auto sA = tA.slice<dynamic_extent, TM>(0, row0);
    auto sB = tB.slice<dynamic_extent, TN>(0, col0);
    auto sC = tC.slice<TN, TM>(col0, row0);
    op.run(sA, sB, sC);
  } else {
    auto sA = tA.slice(0, row0);
    auto sB = tB.slice(0, col0);
    auto sC = tC.slice(col0, row0);
    op.run(sA, sB, sC);
  }
}
)MSL";

struct Shape { int m, n, k; };
struct Cfg { int tm, tn, sg; };

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    printf("device: %s\n\n", dev.name.UTF8String);

    std::vector<Cfg> cfgs = {
        {32, 128, 2}, {16, 128, 2}, {32, 64, 2}, {64, 64, 2}, {32, 192, 2},
        {32, 256, 2}, {48, 128, 2}, {16, 64, 2}, {32, 128, 1}, {16, 128, 1},
        {32, 64, 1}, {16, 64, 1},
    };
    std::vector<Shape> shapes = {
        {2048, 2048, 2048}, {256, 4864, 896}, {256, 896, 4864}, {1024, 1024, 1024},
    };

    // Exactness inputs once (asymmetric, full int8 range, deep k).
    const Shape es{96, 160, 2048};
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d(-128, 127);
    std::vector<int8_t> eav(size_t(es.m) * es.k), ebv(size_t(es.n) * es.k);
    for (auto& v : eav) v = (int8_t)d(rng);
    for (auto& v : ebv) v = (int8_t)d(rng);
    std::vector<int32_t> want(size_t(es.m) * es.n);
    for (int i = 0; i < es.m; ++i)
      for (int j = 0; j < es.n; ++j) {
        int64_t acc = 0;
        for (int kk = 0; kk < es.k; ++kk)
          acc += int32_t(eav[size_t(i) * es.k + kk]) * int32_t(ebv[size_t(j) * es.k + kk]);
        want[size_t(i) * es.n + j] = (int32_t)acc;
      }

    for (const Cfg& cfg : cfgs) {
      char src[8192];
      snprintf(src, sizeof(src), kTemplate, cfg.tm, cfg.tn, cfg.sg);
      MTLCompileOptions* opts = [MTLCompileOptions new];
      opts.languageVersion = (MTLLanguageVersion)((4 << 16) + 0);
      NSError* err = nil;
      id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                                             options:opts
                                               error:&err];
      if (!lib) {
        printf("TM=%-3d TN=%-3d SG=%d: compile FAILED (%s)\n", cfg.tm, cfg.tn, cfg.sg,
               err.localizedDescription.UTF8String ?: "?");
        continue;
      }
      id<MTLFunction> fn = [lib newFunctionWithName:@"mpp_gemm_s8_nt"];
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
      if (!pso) {
        printf("TM=%-3d TN=%-3d SG=%d: PSO FAILED\n", cfg.tm, cfg.tn, cfg.sg);
        continue;
      }
      if (pso.maxTotalThreadsPerThreadgroup < pso.threadExecutionWidth * cfg.sg) {
        printf("TM=%-3d TN=%-3d SG=%d: threadgroup too large, skip\n", cfg.tm, cfg.tn, cfg.sg);
        continue;
      }

      auto run_gemm = [&](id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
                          int m, int n, int k) {
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
        MTLSize grid = MTLSizeMake((n + cfg.tn - 1) / cfg.tn, (m + cfg.tm - 1) / cfg.tm, 1);
        MTLSize group = MTLSizeMake(pso.threadExecutionWidth * cfg.sg, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
      };

      // exactness gate per config
      {
        id<MTLBuffer> a = [dev newBufferWithBytes:eav.data() length:eav.size()
                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> b = [dev newBufferWithBytes:ebv.data() length:ebv.size()
                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> c = [dev newBufferWithLength:want.size() * 4
                                           options:MTLResourceStorageModeShared];
        run_gemm(a, b, c, es.m, es.n, es.k);
        if (memcmp(c.contents, want.data(), want.size() * 4) != 0) {
          printf("TM=%-3d TN=%-3d SG=%d: NOT EXACT — skipping\n", cfg.tm, cfg.tn, cfg.sg);
          continue;
        }
      }

      printf("TM=%-3d TN=%-3d SG=%d (exact):", cfg.tm, cfg.tn, cfg.sg);
      for (const Shape& s : shapes) {
        const int iters = (int64_t(s.m) * s.n * s.k > (int64_t)1 << 29) ? 10 : 30;
        std::vector<int8_t> av(size_t(s.m) * s.k, 3), bv(size_t(s.n) * s.k, -5);
        id<MTLBuffer> a = [dev newBufferWithBytes:av.data() length:av.size()
                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> b = [dev newBufferWithBytes:bv.data() length:bv.size()
                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> c = [dev newBufferWithLength:size_t(s.m) * s.n * 4
                                           options:MTLResourceStorageModeShared];
        run_gemm(a, b, c, s.m, s.n, s.k);  // warmup
        // Best-of-3: round-to-round spread on this machine can exceed 2x.
        double ms = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
          auto t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < iters; ++i) run_gemm(a, b, c, s.m, s.n, s.k);
          auto t1 = std::chrono::steady_clock::now();
          double r = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
          if (r < ms) ms = r;
        }
        double gflops = 2.0 * s.m * s.n * double(s.k) / (ms * 1e6);
        printf("  [%dx%dx%d: %.3fms %.0fGF]", s.m, s.n, s.k, ms, gflops);
      }
      printf("\n");
    }
  }
  return 0;
}
