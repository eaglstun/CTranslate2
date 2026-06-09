#include "test_utils.h"

#ifdef CT2_WITH_METAL

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include <cstdlib>

#include <ctranslate2/devices.h>
#include <ctranslate2/generator.h>
#include <ctranslate2/generation.h>
#include <ctranslate2/models/model.h>
#include <ctranslate2/ops/ops.h>
#include <ctranslate2/profiler.h>
#include <ctranslate2/storage_view.h>
#include <ctranslate2/translator.h>

#include "metal/primitives.h"
#include "metal/utils.h"

using namespace ctranslate2;

// These tests exercise the Milestone 1 Metal "tracer bullet": device discovery, the
// shared-buffer allocator (a METAL StorageView's data() pointer is CPU-addressable on
// unified memory), and a single hand-written compute kernel dispatched end-to-end.

class MetalTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!metal::has_gpu())
      GTEST_SKIP() << "No Metal device available";
  }
};

TEST_F(MetalTest, DeviceDiscovery) {
  EXPECT_GE(get_device_count(Device::METAL), 1);
  EXPECT_EQ(device_to_str(Device::METAL), "metal");
  EXPECT_EQ(str_to_device("metal"), Device::METAL);
}

TEST_F(MetalTest, AllocateAndRoundTrip) {
  // Allocate uninitialized memory on the Metal device, write to it through the
  // CPU-addressable shared-buffer pointer, and read it back.
  const std::vector<float> host = {1.5f, -2.0f, 3.25f, 42.0f};
  StorageView x({static_cast<dim_t>(host.size())}, DataType::FLOAT32, Device::METAL);
  ASSERT_EQ(x.device(), Device::METAL);

  std::memcpy(x.data<float>(), host.data(), host.size() * sizeof(float));

  std::vector<float> readback(host.size());
  std::memcpy(readback.data(), x.data<float>(), host.size() * sizeof(float));
  EXPECT_EQ(readback, host);
}

TEST_F(MetalTest, ElementwiseAdd) {
  const std::vector<float> a_host = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const std::vector<float> b_host = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  const dim_t n = static_cast<dim_t>(a_host.size());

  StorageView a({n}, DataType::FLOAT32, Device::METAL);
  StorageView b({n}, DataType::FLOAT32, Device::METAL);
  StorageView c({n}, DataType::FLOAT32, Device::METAL);

  std::memcpy(a.data<float>(), a_host.data(), a_host.size() * sizeof(float));
  std::memcpy(b.data<float>(), b_host.data(), b_host.size() * sizeof(float));

  metal::add(a.data<float>(), b.data<float>(), c.data<float>(), n, /*b_is_scalar=*/false, 0.f);
  metal::synchronize();  // GPU work is batched; flush before reading on the host.

  std::vector<float> result(a_host.size());
  std::memcpy(result.data(), c.data<float>(), a_host.size() * sizeof(float));

  for (size_t i = 0; i < a_host.size(); ++i)
    EXPECT_FLOAT_EQ(result[i], a_host[i] + b_host[i]) << "at index " << i;
}

// End-to-end: run the full encoder-decoder transformer (the default transliteration
// model) on the Metal device and require its output to match the CPU reference exactly.
// This exercises the whole stack on Device::METAL — embeddings, attention (with MPS
// matmuls), feed-forward, layer norm, and beam search.
TEST_F(MetalTest, EndToEndTranslation) {
  const std::vector<std::vector<std::string>> inputs = {
    {"آ", "ت", "ز", "م", "و", "ن"},
  };

  Translator cpu_translator(default_model_dir(), Device::CPU);
  Translator metal_translator(default_model_dir(), Device::METAL);

  const auto cpu_results = cpu_translator.translate_batch(inputs);
  const auto metal_results = metal_translator.translate_batch(inputs);

  ASSERT_EQ(metal_results.size(), cpu_results.size());
  for (size_t i = 0; i < cpu_results.size(); ++i) {
    EXPECT_FALSE(metal_results[i].output().empty());
    EXPECT_EQ(metal_results[i].output(), cpu_results[i].output())
        << "Metal translation diverged from CPU for input " << i;
  }
}

// Run the whole encoder-decoder in float16 on Metal: weights and activations are fp16,
// so this exercises the integration of every fp16 Metal kernel (GEMM, softmax, layer
// norm, bias+activation, gather, …). Requires the forward pass to be fp16-complete.
TEST_F(MetalTest, EndToEndTranslationFloat16) {
  const std::vector<std::vector<std::string>> inputs = {{"آ", "ت", "ز", "م", "و", "ن"}};

  Translator cpu_translator(default_model_dir(), Device::CPU);
  const auto cpu_results = cpu_translator.translate_batch(inputs);

  auto model = models::Model::load(default_model_dir(), Device::METAL, 0,
                                   ComputeType::FLOAT16);
  Translator metal_translator(model);
  const auto metal_results = metal_translator.translate_batch(inputs);

  ASSERT_EQ(metal_results.size(), cpu_results.size());
  EXPECT_FALSE(metal_results[0].output().empty());
  // fp16 may diverge slightly from fp32; for this small model the decoded tokens still
  // match. If a larger model diverges, relax this to a BLEU/overlap check.
  EXPECT_EQ(metal_results[0].output(), cpu_results[0].output());
}

// fp16 compute on Metal: GEMM (MPSDataTypeFloat16) and softmax (half kernel) should
// match the float32 GPU result within half-precision tolerance.
TEST_F(MetalTest, Float16GemmMatchesFloat32) {
  StorageView a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, Device::METAL);
  StorageView b({3, 2}, std::vector<float>{6, 5, 4, 3, 2, 1}, Device::METAL);

  StorageView c32(Device::METAL);
  ops::MatMul()(a, b, c32);

  StorageView c16(DataType::FLOAT16, Device::METAL);
  ops::MatMul()(a.to(DataType::FLOAT16), b.to(DataType::FLOAT16), c16);
  EXPECT_EQ(c16.dtype(), DataType::FLOAT16);

  expect_storage_eq(c16.to_float32(), c32, 2e-2);
}

TEST_F(MetalTest, Float16SoftMaxMatchesFloat32) {
  StorageView x({2, 4}, std::vector<float>{1, 2, 3, 4, 4, 3, 2, 1}, Device::METAL);

  StorageView y32(Device::METAL);
  ops::SoftMax()(x, y32);

  StorageView y16(DataType::FLOAT16, Device::METAL);
  ops::SoftMax()(x.to(DataType::FLOAT16), y16);
  EXPECT_EQ(y16.dtype(), DataType::FLOAT16);

  expect_storage_eq(y16.to_float32(), y32, 2e-2);
}

TEST_F(MetalTest, Float16RMSNormMatchesFloat32) {
  StorageView x({2, 4}, std::vector<float>{0.5, -1.0, 2.0, 0.25, 1.5, 0.75, -0.5, 1.0}, Device::METAL);
  StorageView gamma({4}, std::vector<float>{1.0, 0.5, 2.0, 1.5}, Device::METAL);

  StorageView y32(Device::METAL);
  ops::RMSNorm()(gamma, x, y32);

  StorageView y16(DataType::FLOAT16, Device::METAL);
  ops::RMSNorm()(gamma.to(DataType::FLOAT16), x.to(DataType::FLOAT16), y16);
  EXPECT_EQ(y16.dtype(), DataType::FLOAT16);

  expect_storage_eq(y16.to_float32(), y32, 2e-2);
}

TEST_F(MetalTest, Float16LayerNormMatchesFloat32) {
  StorageView x({2, 4}, std::vector<float>{0.5, -1.0, 2.0, 0.25, 1.5, 0.75, -0.5, 1.0}, Device::METAL);
  StorageView gamma({4}, std::vector<float>{1.0, 0.5, 2.0, 1.5}, Device::METAL);
  StorageView beta({4}, std::vector<float>{0.1, -0.1, 0.0, 0.2}, Device::METAL);

  StorageView y32(Device::METAL);
  ops::LayerNorm()(beta, gamma, x, y32);

  StorageView y16(DataType::FLOAT16, Device::METAL);
  ops::LayerNorm()(beta.to(DataType::FLOAT16), gamma.to(DataType::FLOAT16),
                   x.to(DataType::FLOAT16), y16);
  EXPECT_EQ(y16.dtype(), DataType::FLOAT16);

  expect_storage_eq(y16.to_float32(), y32, 2e-2);
}

TEST_F(MetalTest, Float16BiasAddGELUMatchesFloat32) {
  StorageView value({2, 3}, std::vector<float>{0.5, -1.0, 2.0, 1.5, 0.75, -0.5}, Device::METAL);
  StorageView bias({3}, std::vector<float>{0.1, -0.2, 0.3}, Device::METAL);
  const ops::ActivationType gelu = ops::ActivationType::GELU;
  ops::BiasAdd bias_add(&gelu);

  StorageView out32(Device::METAL);
  bias_add(value, bias, out32);

  StorageView out16(DataType::FLOAT16, Device::METAL);
  bias_add(value.to(DataType::FLOAT16), bias.to(DataType::FLOAT16), out16);
  EXPECT_EQ(out16.dtype(), DataType::FLOAT16);

  expect_storage_eq(out16.to_float32(), out32, 2e-2);
}

TEST_F(MetalTest, Float16TopPMaskMatchesFloat32) {
  // Well-separated logits so the nucleus (top-p) set is identical in fp16 and fp32.
  StorageView x({2, 5}, std::vector<float>{1.0, 3.0, 0.5, 2.0, 0.1,
                                           0.2, 0.4, 2.5, 1.0, 0.3}, Device::METAL);
  // mask of 0 is representable in fp16 (a large-negative mask would saturate to -inf).
  ops::TopPMask topp(/*p=*/0.8f, /*mask_value=*/0.0f);

  StorageView out32(Device::METAL);
  topp(x, out32);

  StorageView out16(DataType::FLOAT16, Device::METAL);
  topp(x.to(DataType::FLOAT16), out16);
  EXPECT_EQ(out16.dtype(), DataType::FLOAT16);

  expect_storage_eq(out16.to_float32(), out32, 2e-2);
}

TEST_F(MetalTest, Float16MultinomialRuns) {
  // Multinomial is RNG-based, so this verifies the fp16 path runs (does not throw on the
  // non-CUDA float dispatch) and produces valid in-range samples, rather than exact parity.
  StorageView probs({2, 4}, std::vector<float>{0.1, 0.2, 0.3, 0.4,
                                               0.4, 0.3, 0.2, 0.1}, Device::METAL);
  ops::Multinomial multinomial(/*sample_size=*/1);

  StorageView out16(DataType::INT32, Device::METAL);
  EXPECT_NO_THROW(multinomial(probs.to(DataType::FLOAT16), out16));
  EXPECT_EQ(out16.dtype(), DataType::INT32);

  const StorageView out_cpu = out16.to(Device::CPU);
  EXPECT_EQ(out_cpu.size(), 2);
  const auto* idx = out_cpu.data<int32_t>();
  for (dim_t i = 0; i < out_cpu.size(); ++i) {
    EXPECT_GE(idx[i], 0);
    EXPECT_LT(idx[i], 4);
  }
}

// Fused residual-add + RMSNorm must match the unfused Add then RMSNorm, for both the
// residual sum and the normed output (fp32 and fp16).
TEST_F(MetalTest, AddRMSNormMatchesUnfused) {
  const dim_t rows = 4, depth = 8;
  std::vector<float> av(rows * depth), bv(rows * depth), gv(depth);
  for (size_t i = 0; i < av.size(); ++i) { av[i] = 0.1f * (i % 5) - 0.2f; bv[i] = 0.05f * (i % 3); }
  for (size_t i = 0; i < gv.size(); ++i) gv[i] = 0.5f + 0.1f * i;
  const float eps = 1e-6f;

  for (DataType dt : {DataType::FLOAT32, DataType::FLOAT16}) {
    StorageView a = StorageView({rows, depth}, av, Device::METAL).to(dt);
    StorageView b = StorageView({rows, depth}, bv, Device::METAL).to(dt);
    StorageView g = StorageView({depth}, gv, Device::METAL).to(dt);

    StorageView sum_ref(dt, Device::METAL);
    ops::Add()(a, b, sum_ref);
    StorageView normed_ref(dt, Device::METAL);
    ops::RMSNorm(eps, /*use_residual=*/false)(g, sum_ref, normed_ref);

    StorageView sum(dt, Device::METAL); sum.resize_as(a);
    StorageView normed(dt, Device::METAL); normed.resize_as(a);
    const double tol = (dt == DataType::FLOAT16) ? 2e-2 : 1e-5;
    if (dt == DataType::FLOAT16)
      metal::add_rms_norm(a.data<float16_t>(), b.data<float16_t>(), g.data<float16_t>(),
                          sum.data<float16_t>(), normed.data<float16_t>(), rows, depth, eps, false);
    else
      metal::add_rms_norm(a.data<float>(), b.data<float>(), g.data<float>(),
                          sum.data<float>(), normed.data<float>(), rows, depth, eps, false);
    metal::synchronize();

    expect_storage_eq(sum.to_float32(), sum_ref.to_float32(), tol);
    expect_storage_eq(normed.to_float32(), normed_ref.to_float32(), tol);
  }
}

// Fused residual-add + LayerNorm must match the unfused Add then LayerNorm (sum + normed).
TEST_F(MetalTest, AddLayerNormMatchesUnfused) {
  const dim_t rows = 4, depth = 8;
  std::vector<float> av(rows * depth), bv(rows * depth), gv(depth), bt(depth);
  for (size_t i = 0; i < av.size(); ++i) { av[i] = 0.1f * (i % 5) - 0.2f; bv[i] = 0.05f * (i % 3); }
  for (size_t i = 0; i < gv.size(); ++i) { gv[i] = 0.5f + 0.1f * i; bt[i] = 0.01f * i - 0.03f; }
  const float eps = 1e-5f;

  for (DataType dt : {DataType::FLOAT32, DataType::FLOAT16}) {
    StorageView a = StorageView({rows, depth}, av, Device::METAL).to(dt);
    StorageView b = StorageView({rows, depth}, bv, Device::METAL).to(dt);
    StorageView g = StorageView({depth}, gv, Device::METAL).to(dt);
    StorageView beta = StorageView({depth}, bt, Device::METAL).to(dt);

    StorageView sum_ref(dt, Device::METAL);
    ops::Add()(a, b, sum_ref);
    StorageView normed_ref(dt, Device::METAL);
    ops::LayerNorm(-1, eps)(beta, g, sum_ref, normed_ref);

    StorageView sum(dt, Device::METAL); sum.resize_as(a);
    StorageView normed(dt, Device::METAL); normed.resize_as(a);
    const double tol = (dt == DataType::FLOAT16) ? 2e-2 : 1e-5;
    if (dt == DataType::FLOAT16)
      metal::add_layer_norm(a.data<float16_t>(), b.data<float16_t>(), g.data<float16_t>(),
                            beta.data<float16_t>(), sum.data<float16_t>(),
                            normed.data<float16_t>(), rows, depth, eps);
    else
      metal::add_layer_norm(a.data<float>(), b.data<float>(), g.data<float>(),
                            beta.data<float>(), sum.data<float>(),
                            normed.data<float>(), rows, depth, eps);
    metal::synchronize();

    expect_storage_eq(sum.to_float32(), sum_ref.to_float32(), tol);
    expect_storage_eq(normed.to_float32(), normed_ref.to_float32(), tol);
  }
}

TEST_F(MetalTest, RotaryMatchesCPU) {
  // input is [batch, max_time, depth]; with is_transposed=true, max_time = dim(-2).
  const std::vector<float> in_vec = {0.1f, 0.2f, 0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f};
  const std::vector<float> sin_vec = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
  const std::vector<float> cos_vec = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f};
  ops::Rotary rotary(/*ndims=*/4, /*interleave=*/false);

  // CPU reference.
  StorageView in_cpu({1, 2, 4}, in_vec);
  StorageView sin_cpu({2, 4}, sin_vec);
  StorageView cos_cpu({2, 4}, cos_vec);
  StorageView ref;
  rotary(in_cpu, sin_cpu, cos_cpu, ref, /*is_transposed=*/true);

  StorageView in_m({1, 2, 4}, in_vec, Device::METAL);
  StorageView sin_m({2, 4}, sin_vec, Device::METAL);
  StorageView cos_m({2, 4}, cos_vec, Device::METAL);

  // float32 on Metal must match CPU closely.
  StorageView y32(Device::METAL);
  rotary(in_m, sin_m, cos_m, y32, true);
  expect_storage_eq(y32.to_float32(), ref, 1e-5);

  // float16 on Metal within half tolerance.
  StorageView y16(DataType::FLOAT16, Device::METAL);
  rotary(in_m.to(DataType::FLOAT16), sin_m.to(DataType::FLOAT16), cos_m.to(DataType::FLOAT16),
         y16, true);
  expect_storage_eq(y16.to_float32(), ref, 2e-2);
}

// ---------------------------------------------------------------------------
// Benchmarks (disabled by default; run with:
//   ./ctranslate2_test <data> --gtest_also_run_disabled_tests --gtest_filter='*Benchmark*')
// ---------------------------------------------------------------------------

template <typename Fn>
static double time_ms(int iters, Fn&& fn) {
  fn();  // warmup (MPS pipeline compilation, allocator priming, …)
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i)
    fn();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

TEST_F(MetalTest, DISABLED_BenchmarkGemm) {
  std::cout << "\n=== Square GEMM (m=n=k), ms/iter and GFLOPS ===\n";
  for (dim_t n : {256, 512, 1024, 2048}) {
    const std::vector<float> av(n * n, 0.01f);
    const std::vector<float> bv(n * n, 0.02f);
    const int iters = n <= 512 ? 50 : (n <= 1024 ? 20 : 8);

    auto run = [&](const std::string& label, Device dev, DataType dt) {
      StorageView a = StorageView({n, n}, av, dev).to(dt);
      StorageView b = StorageView({n, n}, bv, dev).to(dt);
      StorageView c(dt, dev);
      const ops::MatMul mm;
      // Flush each iteration so we measure GPU execution, not just command encoding.
      const double ms = time_ms(iters, [&] { mm(a, b, c); synchronize_device(dev, 0); });
      const double gflops = 2.0 * double(n) * n * n / (ms * 1e6);
      std::cout << "  n=" << n << "  " << label << ":  " << ms << " ms,  "
                << gflops << " GFLOPS\n";
    };

    run("CPU   fp32", Device::CPU, DataType::FLOAT32);
    run("METAL fp32", Device::METAL, DataType::FLOAT32);
    run("METAL fp16", Device::METAL, DataType::FLOAT16);
    std::cout << "\n";
  }
}

// Isolate per-op ENCODE cost from GPU execution: time many GEMMs that are committed but
// not waited (one flush at the end) vs. flush-per-iter. The gap is the wait/round-trip;
// the batched number is the floor set by command-buffer + MPS-object creation + commit.
TEST_F(MetalTest, DISABLED_BenchmarkGemmEncode) {
  std::cout << "\n=== GEMM per-op cost: flush-per-iter vs batched-encode (ms/iter) ===\n";
  for (dim_t n : {256, 1024}) {
    const std::vector<float> av(n * n, 0.01f);
    const std::vector<float> bv(n * n, 0.02f);
    const int iters = 200;
    StorageView a = StorageView({n, n}, av, Device::METAL).to(DataType::FLOAT16);
    StorageView b = StorageView({n, n}, bv, Device::METAL).to(DataType::FLOAT16);
    StorageView c(DataType::FLOAT16, Device::METAL);
    const ops::MatMul mm;

    const double per_iter = time_ms(iters, [&] { mm(a, b, c); synchronize_device(Device::METAL, 0); });

    // Batched: encode all, flush once; report per-iter.
    mm(a, b, c); synchronize_device(Device::METAL, 0);  // warmup
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      mm(a, b, c);
    synchronize_device(Device::METAL, 0);
    const auto t1 = std::chrono::steady_clock::now();
    const double batched = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    std::cout << "  n=" << n << " fp16:  flush-per-iter " << per_iter
              << " ms,  batched-encode " << batched << " ms\n";
  }
}

// Row-reduction kernels (softmax / rms_norm / layer_norm). One threadgroup per row; the
// reduction is the cost being optimized. Representative LLM shapes: attention-score softmax
// (many rows, key-length depth) and norms (rows = batch*seq, depth = hidden). A/B harness
// for the SIMD-group-reduction rewrite — run before and after the kernel change.
TEST_F(MetalTest, DISABLED_BenchmarkReduction) {
  std::cout << "\n=== Row-reduction ops on METAL (ms/iter, flush per iter) ===\n";
  struct Shape { dim_t rows; dim_t depth; };
  for (Shape s : {Shape{16384, 512}, Shape{8192, 896}, Shape{4096, 2048}}) {
    const std::vector<float> xv(size_t(s.rows) * s.depth, 0.01f);
    const std::vector<float> gv(s.depth, 1.0f);
    const std::vector<float> bv(s.depth, 0.0f);
    const int iters = 30;

    auto bench = [&](const std::string& label, DataType dt) {
      StorageView x = StorageView({s.rows, s.depth}, xv, Device::METAL).to(dt);
      StorageView g = StorageView({s.depth}, gv, Device::METAL).to(dt);
      StorageView b = StorageView({s.depth}, bv, Device::METAL).to(dt);
      StorageView y(dt, Device::METAL);

      const ops::SoftMax softmax;
      const double sm = time_ms(iters, [&] { softmax(x, y); synchronize_device(Device::METAL, 0); });
      const ops::RMSNorm rms_norm;
      const double rn = time_ms(iters, [&] { rms_norm(g, x, y); synchronize_device(Device::METAL, 0); });
      const ops::LayerNorm layer_norm;
      const double ln = time_ms(iters, [&] { layer_norm(b, g, x, y); synchronize_device(Device::METAL, 0); });

      std::cout << "  rows=" << s.rows << " depth=" << s.depth << "  " << label
                << ":  SoftMax " << sm << "  RMSNorm " << rn << "  LayerNorm " << ln << "  ms\n";
    };
    bench("fp32", DataType::FLOAT32);
    bench("fp16", DataType::FLOAT16);
    std::cout << "\n";
  }
}

// Fusion A/B: residual-add + RMSNorm as two ops (ops::Add then ops::RMSNorm, two launches)
// vs the single fused metal::add_rms_norm. The fused kernel removes one launch and one
// device read pass; this measures whether that beats the two-op sequence on real shapes.
TEST_F(MetalTest, DISABLED_BenchmarkAddRMSNorm) {
  std::cout << "\n=== residual-Add + RMSNorm: two ops vs fused (ms/iter, speedup) ===\n";
  struct Shape { dim_t rows; dim_t depth; };
  for (Shape s : {Shape{16384, 512}, Shape{8192, 896}, Shape{4096, 2048}}) {
    const std::vector<float> xv(size_t(s.rows) * s.depth, 0.01f);
    const std::vector<float> gv(s.depth, 1.0f);
    const int iters = 30;

    auto bench = [&](const std::string& label, DataType dt) {
      StorageView a = StorageView({s.rows, s.depth}, xv, Device::METAL).to(dt);
      StorageView b = StorageView({s.rows, s.depth}, xv, Device::METAL).to(dt);
      StorageView g = StorageView({s.depth}, gv, Device::METAL).to(dt);
      StorageView sum(dt, Device::METAL); sum.resize_as(a);
      StorageView normed(dt, Device::METAL); normed.resize_as(a);

      const ops::Add add;
      const ops::RMSNorm rms_norm;
      const double two = time_ms(iters, [&] {
        add(a, b, sum);
        rms_norm(g, sum, normed);
        synchronize_device(Device::METAL, 0);
      });
      const double fused = time_ms(iters, [&] {
        if (dt == DataType::FLOAT16)
          metal::add_rms_norm(a.data<float16_t>(), b.data<float16_t>(), g.data<float16_t>(),
                              sum.data<float16_t>(), normed.data<float16_t>(), s.rows, s.depth, 1e-6f, false);
        else
          metal::add_rms_norm(a.data<float>(), b.data<float>(), g.data<float>(),
                              sum.data<float>(), normed.data<float>(), s.rows, s.depth, 1e-6f, false);
        synchronize_device(Device::METAL, 0);
      });
      std::cout << "  rows=" << s.rows << " depth=" << s.depth << "  " << label
                << ":  two-op " << two << "  fused " << fused << "  (" << (two / fused) << "x)\n";
    };
    bench("fp32", DataType::FLOAT32);
    bench("fp16", DataType::FLOAT16);
    std::cout << "\n";
  }
}

TEST_F(MetalTest, DISABLED_BenchmarkTranslation) {
  std::cout << "\n=== End-to-end translation, ms per batch of 32 ===\n";
  const std::vector<std::vector<std::string>> batch(32, {"آ", "ت", "ز", "م", "و", "ن"});

  auto run = [&](const std::string& label, Device dev, ComputeType ct) {
    auto model = models::Model::load(default_model_dir(), dev, 0, ct);
    Translator translator(model);
    const double ms = time_ms(10, [&] { translator.translate_batch(batch); });
    std::cout << "  " << label << ":  " << ms << " ms/batch\n";
  };

  run("CPU   fp32", Device::CPU, ComputeType::FLOAT32);
  run("METAL fp32", Device::METAL, ComputeType::FLOAT32);
  run("METAL fp16", Device::METAL, ComputeType::FLOAT16);
}

// Real decoder-only LLM end-to-end, vs the tiny transliteration model above. The tiny
// model is the worst case for any GPU backend (ops too small to amortize per-op API
// cost); a real LLM has large hidden size and GEMM-dominated layers, the favorable end of
// the GEMM table. Point CT2_LLM_MODEL at a converted decoder dir (e.g. Qwen2.5-0.5B).
TEST_F(MetalTest, DISABLED_BenchmarkLLM) {
  const char* model_env = std::getenv("CT2_LLM_MODEL");
  if (!model_env)
    GTEST_SKIP() << "Set CT2_LLM_MODEL to a converted decoder-only model directory";
  const std::string model_dir(model_env);

  // Synthetic prompt built from tokens known to exist in a byte-level BPE vocab. Content
  // is irrelevant to op shapes — only prompt length and the forced step count matter.
  const std::vector<std::string> content =
      {"ing", "ion", "ent", "ate", "ation", "ort", "ame", "ist", "ers", "ass", "int", "urn"};
  auto make_prompt = [&](size_t len) {
    std::vector<std::string> p;
    p.reserve(len);
    for (size_t i = 0; i < len; ++i)
      p.push_back(content[i % content.size()]);
    return p;
  };

  auto run = [&](const std::string& label, Device dev, ComputeType ct,
                 size_t batch_size, size_t prompt_len, size_t decode_steps) {
    GenerationOptions options;
    options.beam_size = 1;
    options.sampling_topk = 1;             // greedy / deterministic
    options.max_length = decode_steps;     // stop after exactly decode_steps new tokens
    options.min_length = decode_steps;     // forbid early EOS so every run does the same work
    options.include_prompt_in_result = false;

    auto model = models::Model::load(model_dir, dev, 0, ct);
    Generator generator(model);
    const std::vector<std::vector<std::string>> batch(batch_size, make_prompt(prompt_len));

    auto wait = [&] {
      auto futures = generator.generate_batch_async(batch, options);
      for (auto& f : futures)
        f.get();
    };

    wait();  // warmup (includes first-GEMM MPS pipeline build)
    const int iters = 3;
    const double ms = time_ms(iters, wait);
    const double toks = double(batch_size) * decode_steps;  // decode throughput
    std::cout << "  bs=" << batch_size << "  " << label << ":  " << ms << " ms,  "
              << (toks / (ms / 1000.0)) << " tok/s\n";
  };

  // Profiling mode (build with -DENABLE_PROFILING=ON): isolate WHY fp16 prefill is slower
  // than fp32 at bs=1 by dumping the per-op time breakdown for each. The profiler flushes
  // per scope, so absolute times are inflated, but the fp32-vs-fp16 comparison is apples to
  // apples. Set CT2_LLM_PROFILE=1.
  if (std::getenv("CT2_LLM_PROFILE")) {
    // Profile two regimes: prefill-bound (long prompt, 1 step → big GEMMs dominate) and
    // decode-bound (short prompt, many 1-token steps → tiny matrix-vector ops, op-count
    // bound). The op breakdown differs sharply; small-op fusions (e.g. add_rms_norm) matter
    // most in decode. The profiler flushes per scope, so absolute times are inflated.
    auto profile = [&](const std::string& label, Device dev, ComputeType ct,
                       size_t prompt_len, size_t steps) {
      GenerationOptions options;
      options.beam_size = 1;
      options.sampling_topk = 1;
      options.max_length = steps;
      options.min_length = steps;
      options.include_prompt_in_result = false;
      const std::vector<std::vector<std::string>> batch(1, make_prompt(prompt_len));
      auto model = models::Model::load(model_dir, dev, 0, ct);
      Generator generator(model);
      auto wait = [&] {
        auto futures = generator.generate_batch_async(batch, options);
        for (auto& f : futures)
          f.get();
      };
      wait();  // warmup
      init_profiling(dev, 1);
      for (int i = 0; i < 10; ++i)
        wait();
      std::cerr << "\n##### PROFILE " << label << " #####\n";
      dump_profiling(std::cerr);
    };
    profile("METAL fp16 PREFILL (prompt 512, 1 step)", Device::METAL, ComputeType::FLOAT16, 512, 1);
    profile("METAL fp16 DECODE (prompt 8, 64 steps)", Device::METAL, ComputeType::FLOAT16, 8, 64);
    return;
  }

  // Two regimes: a decode-bound run (short prompt, many tiny batch=N steps) and a
  // prefill-bound run (long prompt → one big seq×hidden GEMM, 1 decode step) that isolates
  // the compute-bound path where the GEMM table predicts Metal should win.
  struct Regime { const char* name; size_t prompt; size_t decode; };
  for (const Regime r : {Regime{"decode-bound", 32, 32}, Regime{"prefill-bound", 512, 1}}) {
    std::cout << "\n--- " << r.name << ": prompt=" << r.prompt
              << ", decode=" << r.decode << " ---\n";
    for (size_t bs : {size_t(1), size_t(8)}) {
      run("CPU   fp32", Device::CPU, ComputeType::FLOAT32, bs, r.prompt, r.decode);
      run("METAL fp32", Device::METAL, ComputeType::FLOAT32, bs, r.prompt, r.decode);
      run("METAL fp16", Device::METAL, ComputeType::FLOAT16, bs, r.prompt, r.decode);
      std::cout << "\n";
    }
  }
}

// CPU-vs-Metal DECODE-PARITY gate for a real decoder-only LLM. This exercises the
// autoregressive Generator path — rotary at nonzero positions, KV-cache append/read, the
// per-step decode loop — which the tiny transliteration EndToEnd test does NOT cover
// (that's encoder-decoder). Greedy + forced length = deterministic, so Metal fp32 MUST
// reproduce CPU fp32 token-for-token; a mismatch means a Metal op is wrong, not "fp16
// noise". This is the assertion the speed-only BenchmarkLLM lacks — it stops "fast" from
// silently meaning "fast garbage". Point CT2_LLM_MODEL at a converted decoder dir (e.g.
// Qwen2.5-0.5B). No decoder-only model ships in test data, so this is a manual gate via
// env var, not a CI gate; supply a model to run it.
TEST_F(MetalTest, DISABLED_DecodeParityLLM) {
  const char* model_env = std::getenv("CT2_LLM_MODEL");
  if (!model_env)
    GTEST_SKIP() << "Set CT2_LLM_MODEL to a converted decoder-only model directory";
  const std::string model_dir(model_env);

  // Prompt selection. CAVEAT (learned the hard way): a meaningless prompt can make BOTH
  // backends degenerate into the SAME loop, so CPU==Metal holds on garbage and the gate
  // FALSE-PASSES. Real bugs only surface on a prompt the model actually computes over —
  // for BOS-sensitive models (e.g. Gemma2) that means a leading <bos>. So pass a real,
  // model-appropriate prompt via CT2_LLM_PROMPT (space-separated token pieces, e.g.
  // "<bos> The") to make this a meaningful gate; the synthetic default below only proves
  // the loop runs deterministically.
  std::vector<std::string> prompt =
      {"ing", "ion", "ent", "ate", "ation", "ort", "ame", "ist"};
  if (const char* prompt_env = std::getenv("CT2_LLM_PROMPT")) {
    prompt.clear();
    std::istringstream iss(prompt_env);
    for (std::string tok; iss >> tok;)
      prompt.push_back(tok);
  }
  const size_t steps = 24;

  GenerationOptions options;
  options.beam_size = 1;
  options.sampling_topk = 1;            // greedy / deterministic
  options.max_length = steps;
  options.min_length = steps;           // forbid early EOS so the decode loop runs fully
  options.include_prompt_in_result = false;
  const std::vector<std::vector<std::string>> batch(1, prompt);

  auto gen = [&](Device dev, ComputeType ct) {
    auto model = models::Model::load(model_dir, dev, 0, ct);
    Generator generator(model);
    auto futures = generator.generate_batch_async(batch, options);
    return futures[0].get().sequences[0];
  };

  const auto cpu     = gen(Device::CPU,   ComputeType::FLOAT32);
  const auto metal32 = gen(Device::METAL, ComputeType::FLOAT32);
  const auto metal16 = gen(Device::METAL, ComputeType::FLOAT16);

  auto prefix_match = [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
    size_t n = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i, ++n)
      if (a[i] != b[i]) break;
    return n;
  };
  auto dump = [](const char* label, const std::vector<std::string>& s) {
    std::cerr << "  " << label << " (" << s.size() << "): ";
    for (auto& t : s) std::cerr << t << "|";
    std::cerr << "\n";
  };
  std::cerr << "\nDecode parity (" << model_dir << "), " << steps << " greedy steps:\n";
  dump("CPU   fp32", cpu);
  dump("METAL fp32", metal32);
  dump("METAL fp16", metal16);
  std::cerr << "  fp32 prefix match: " << prefix_match(cpu, metal32) << "/" << cpu.size()
            << "   fp16 prefix match: " << prefix_match(cpu, metal16) << "/" << cpu.size() << "\n";

  // fp32 Metal runs the SAME precision as CPU, so any divergence is a wrong Metal op,
  // not rounding. This is the real correctness gate.
  EXPECT_EQ(metal32, cpu) << "Metal fp32 decode diverged from CPU fp32 — a Metal op is incorrect.";
  // fp16 may flip the occasional argmax on a near-tie, but must NOT collapse (the
  // <pad>-forever failure mode); require it to track CPU for most of the sequence.
  EXPECT_GT(prefix_match(cpu, metal16), cpu.size() / 2)
      << "Metal fp16 decode collapsed early (not mere rounding) — suspect a real Metal bug.";
}

// Report-only diagnostic for the fused pre_post (Gemma2-style) add_norm path. Set
// CT2_GEMMA_MODEL to a converted Gemma2 dir. NOTE: this does NOT assert Metal==CPU because
// Gemma2 has a SEPARATE, PRE-EXISTING Metal correctness bug — Metal emits '<pad>' forever
// while CPU is coherent, and this reproduces with the add_norm fusion DISABLED, so it is not
// caused by the fusion (verified 2026-06-09: fused and unfused Metal output are byte-
// identical). The fusion itself is validated by AddRMSNorm/AddLayerNormMatchesUnfused. Once
// the Gemma2-Metal bug is fixed, restore an EXPECT_EQ(metal32, cpu) here as the e2e gate.
TEST_F(MetalTest, DISABLED_Gemma2PrePostParity) {
  const char* model_env = std::getenv("CT2_GEMMA_MODEL");
  if (!model_env)
    GTEST_SKIP() << "Set CT2_GEMMA_MODEL to a converted Gemma2 model directory";
  const std::string model_dir(model_env);

  GenerationOptions options;
  options.beam_size = 1;
  options.sampling_topk = 1;            // greedy / deterministic
  options.max_length = 24;
  options.min_length = 24;
  options.include_prompt_in_result = false;
  const std::vector<std::vector<std::string>> batch(1, {"<bos>", "The"});

  auto gen = [&](Device dev, ComputeType ct) {
    auto model = models::Model::load(model_dir, dev, 0, ct);
    Generator generator(model);
    auto futures = generator.generate_batch_async(batch, options);
    return futures[0].get().sequences[0];
  };

  const auto cpu = gen(Device::CPU, ComputeType::FLOAT32);
  const auto metal32 = gen(Device::METAL, ComputeType::FLOAT32);
  const auto metal16 = gen(Device::METAL, ComputeType::FLOAT16);

  auto prefix_match = [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
    size_t n = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
      if (a[i] != b[i]) break;
      ++n;
    }
    return n;
  };
  std::cerr << "\nGemma2 greedy (" << cpu.size() << " tokens):\n";
  std::cerr << "  CPU fp32   : "; for (auto& t : cpu) std::cerr << t << "|"; std::cerr << "\n";
  std::cerr << "  METAL fp32 : "; for (auto& t : metal32) std::cerr << t << "|"; std::cerr << "\n";
  std::cerr << "  METAL fp32 prefix match vs CPU: " << prefix_match(cpu, metal32) << "/" << cpu.size() << "\n";
  std::cerr << "  METAL fp16 prefix match vs CPU: " << prefix_match(cpu, metal16) << "/" << cpu.size() << "\n";
  if (metal32 != cpu)
    std::cerr << "  (Metal != CPU: pre-existing Gemma2-on-Metal bug, NOT the add_norm fusion)\n";
}

#endif  // CT2_WITH_METAL
