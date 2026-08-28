#include "test_utils.h"

#ifdef CT2_WITH_METAL

#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include <cstdlib>

#include <algorithm>

#include <ctranslate2/devices.h>
#include <ctranslate2/generator.h>
#include <ctranslate2/random.h>
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

TEST_F(MetalTest, MpsGemvMatchesHostReference) {
  const dim_t rows = 4;
  const dim_t columns = 3;
  const float alpha = 0.5f;
  const float beta = 0.25f;
  const std::vector<float> logical_matrix = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    2, 4, 8,
  };
  const std::vector<float> x_values = {2, -1, 0.5f};
  const std::vector<float> initial_y = {1, 2, 3, 4};
  std::vector<float> expected(rows);
  for (dim_t row = 0; row < rows; ++row) {
    float dot = 0;
    for (dim_t column = 0; column < columns; ++column)
      dot += logical_matrix[row * columns + column] * x_values[column];
    expected[row] = alpha * dot + beta * initial_y[row];
  }

  for (const bool transpose : {false, true}) {
    std::vector<float> stored_matrix(logical_matrix.size());
    if (transpose) {
      for (dim_t row = 0; row < rows; ++row)
        for (dim_t column = 0; column < columns; ++column)
          stored_matrix[column * rows + row] = logical_matrix[row * columns + column];
    } else {
      stored_matrix = logical_matrix;
    }

    for (const DataType dtype : {DataType::FLOAT32, DataType::FLOAT16}) {
      StorageView matrix = StorageView(transpose ? Shape{columns, rows} : Shape{rows, columns},
                                       stored_matrix, Device::METAL).to(dtype);
      StorageView x = StorageView({columns}, x_values, Device::METAL).to(dtype);
      StorageView y = StorageView({rows}, initial_y, Device::METAL).to(dtype);
      const dim_t ldm = transpose ? rows : columns;
      if (dtype == DataType::FLOAT16)
        metal::gemv(transpose, rows, columns, alpha,
                    matrix.data<float16_t>(), ldm, x.data<float16_t>(), beta,
                    y.data<float16_t>());
      else
        metal::gemv(transpose, rows, columns, alpha,
                    matrix.data<float>(), ldm, x.data<float>(), beta, y.data<float>());

      StorageView expected_view({rows}, expected);
      if (dtype == DataType::FLOAT16)
        expect_storage_eq(y.to_float32().to(Device::CPU), expected_view, 2e-2);
      else
        expect_storage_eq(y.to(Device::CPU), expected_view, 1e-5);
    }
  }
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

TEST_F(MetalTest, Float16Conv1DMatchesFloat32) {
  // Whisper's encoder leads with a Conv1D in fp16. Conv1D has no Metal kernel, so it runs the
  // float32-only CPU reference over unified memory; fp16 used to throw "only supports float
  // types". The fp16 path now upcasts to fp32 and downcasts back — verify it runs and matches.
  const StorageView input({1, 2, 6}, std::vector<float>{
      0.5f, -1.0f, 2.0f, 0.25f, 1.5f, 0.75f,
      -0.5f, 1.0f, 0.3f, -0.7f, 1.2f, 0.4f}, Device::METAL);
  const StorageView weight({3, 2, 3}, std::vector<float>{
      0.1f, 0.2f, 0.3f, -0.1f, 0.0f, 0.1f,
      0.2f, -0.2f, 0.1f, 0.3f, 0.1f, -0.1f,
      0.0f, 0.2f, -0.3f, 0.1f, 0.1f, 0.1f}, Device::METAL);
  const StorageView bias({3}, std::vector<float>{0.1f, -0.2f, 0.3f}, Device::METAL);

  const ops::Conv1D conv(/*stride=*/1, /*padding=*/1);

  StorageView y32(Device::METAL);
  conv(input, weight, bias, y32);

  StorageView y16(DataType::FLOAT16, Device::METAL);
  conv(input.to(DataType::FLOAT16), weight.to(DataType::FLOAT16), bias.to(DataType::FLOAT16), y16);
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

// ---- Sampling ops on the GPU ----
// TopK and TopPMask are deterministic comparison/sort ops: the bar is bit-parity with
// the CPU reference (modulo tie order, which the CPU partial_sort/std::sort leaves
// unspecified — the GPU kernels are deterministic index-ascending). Multinomial and
// GumbelMax are RNG ops: the GPU uses a different (host-seeded) stream than the CPU
// std::mt19937, so the right checks are seeded reproducibility and distribution match,
// not bit-parity.

namespace {

// Bitwise comparison of two same-length float vectors (EXPECT_FLOAT_EQ allows 4 ulps,
// which would weaken a bit-parity claim).
void expect_bits_eq(const std::vector<float>& got, const std::vector<float>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < got.size(); ++i) {
    uint32_t g, e;
    std::memcpy(&g, &got[i], 4);
    std::memcpy(&e, &expected[i], 4);
    EXPECT_EQ(g, e) << "bit mismatch at index " << i << ": " << got[i] << " vs " << expected[i];
  }
}

}  // namespace

// fp32 TopK at the Qwen2.5 vocabulary size: values AND indices must be bit-identical to
// the CPU reference (random fp32 logits are tie-free, so tie order cannot interfere).
TEST_F(MetalTest, TopKLargeVocabBitParityWithCPU) {
  const dim_t batch = 3;
  const dim_t depth = 151936;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-8.f, 8.f);
  std::vector<float> logits(batch * depth);
  for (auto& v : logits)
    v = dist(rng);

  for (const dim_t k : {dim_t(1), dim_t(50)}) {
    const ops::TopK op(k);

    StorageView x_gpu({batch, depth}, logits, Device::METAL);
    StorageView v_gpu(DataType::FLOAT32, Device::METAL);
    StorageView i_gpu(DataType::INT32, Device::METAL);
    op(x_gpu, v_gpu, i_gpu);

    StorageView x_cpu({batch, depth}, logits, Device::CPU);
    StorageView v_cpu(DataType::FLOAT32, Device::CPU);
    StorageView i_cpu(DataType::INT32, Device::CPU);
    op(x_cpu, v_cpu, i_cpu);

    expect_bits_eq(v_gpu.to(Device::CPU).to_vector<float>(), v_cpu.to_vector<float>());
    const auto ig = i_gpu.to(Device::CPU).to_vector<int32_t>();
    const auto ic = i_cpu.to_vector<int32_t>();
    ASSERT_EQ(ig.size(), ic.size());
    for (size_t i = 0; i < ig.size(); ++i)
      EXPECT_EQ(ig[i], ic[i]) << "index mismatch at " << i << " (k=" << k << ")";
  }
}

// fp16 TopK: at a 151936 vocabulary fp16 ties are unavoidable (fewer distinct finite
// halfs than classes), so indices are only checked for consistency (each index must
// point at its value); the VALUE sequence is uniquely determined even with ties and must
// be bit-identical to the fp32 CPU reference run on the fp16-rounded data (fp16 -> fp32
// conversion is exact and order-preserving).
TEST_F(MetalTest, Float16TopKLargeVocabValueParityWithCPU) {
  const dim_t batch = 2;
  const dim_t depth = 151936;
  const dim_t k = 50;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-8.f, 8.f);
  std::vector<float> logits(batch * depth);
  for (auto& v : logits)
    v = dist(rng);

  const ops::TopK topk_op(k);

  StorageView x16({batch, depth}, logits, Device::METAL);
  x16 = x16.to(DataType::FLOAT16);
  StorageView v16(DataType::FLOAT16, Device::METAL);
  StorageView i16(DataType::INT32, Device::METAL);
  topk_op(x16, v16, i16);

  StorageView x_cpu = x16.to_float32().to(Device::CPU);
  StorageView v_cpu(DataType::FLOAT32, Device::CPU);
  StorageView i_cpu(DataType::INT32, Device::CPU);
  topk_op(x_cpu, v_cpu, i_cpu);

  expect_bits_eq(v16.to_float32().to(Device::CPU).to_vector<float>(), v_cpu.to_vector<float>());

  // Index consistency: x16[row][idx[j]] must equal the returned value bit-for-bit.
  const auto values = v16.to_float32().to(Device::CPU).to_vector<float>();
  const auto indices = i16.to(Device::CPU).to_vector<int32_t>();
  const auto x_host = x16.to_float32().to(Device::CPU).to_vector<float>();
  for (dim_t b = 0; b < batch; ++b)
    for (dim_t j = 0; j < k; ++j) {
      const int32_t idx = indices[b * k + j];
      ASSERT_GE(idx, 0);
      ASSERT_LT(idx, depth);
      EXPECT_EQ(x_host[b * depth + idx], values[b * k + j]);
    }
}

// fp32 TopPMask at a non-power-of-two depth within the GPU kernel's in-threadgroup
// limit: output must be bit-identical to the CPU reference (the GPU kernel mirrors the
// CPU accumulation order exactly; outputs are bit-copies of the input or the mask).
TEST_F(MetalTest, TopPMaskBitParityWithCPU) {
  const dim_t batch = 4;
  const dim_t depth = 1777;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-4.f, 4.f);
  std::vector<float> logits(batch * depth);
  for (auto& v : logits)
    v = dist(rng);

  const ops::TopPMask op(0.9f);

  StorageView x_gpu({batch, depth}, logits, Device::METAL);
  StorageView y_gpu(DataType::FLOAT32, Device::METAL);
  op(x_gpu, y_gpu);

  StorageView x_cpu({batch, depth}, logits, Device::CPU);
  StorageView y_cpu(DataType::FLOAT32, Device::CPU);
  op(x_cpu, y_cpu);

  expect_bits_eq(y_gpu.to(Device::CPU).to_vector<float>(), y_cpu.to_vector<float>());
}

// Fused decode-step SDPA vs the unfused CPU reference (the exact MatMul → SoftMax →
// MatMul sequence it replaces in dot_product_attention). Cases cover the kernel's
// corners: head depth not a multiple of the SIMD width (80), fewer keys than SIMD-groups
// (T=3 < 4), a long key axis (online-softmax accumulation), beam-shaped q (2 rows per
// batch*head), and the SoftMax lengths contract including a fully masked len==0 row.
namespace {

void sdpa_reference_cpu(const StorageView& q, const StorageView& k, const StorageView& v,
                        const StorageView* lengths, float scale, StorageView& out) {
  StorageView scores(DataType::FLOAT32, Device::CPU);
  ops::MatMul(/*trans_a=*/false, /*trans_b=*/true, scale)(q, k, scores);
  StorageView attn(DataType::FLOAT32, Device::CPU);
  ops::SoftMax()(scores, lengths, attn);
  ops::MatMul()(attn, v, out);
}

void run_sdpa_case(dim_t batch, dim_t heads, dim_t q_len, dim_t num_keys, dim_t depth,
                   const std::vector<int32_t>* lengths, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.f, 2.f);
  std::vector<float> q_host(batch * heads * q_len * depth);
  std::vector<float> k_host(batch * heads * num_keys * depth);
  std::vector<float> v_host(batch * heads * num_keys * depth);
  for (auto& x : q_host) x = dist(rng);
  for (auto& x : k_host) x = dist(rng);
  for (auto& x : v_host) x = dist(rng);
  const float scale = 1.f / std::sqrt(static_cast<float>(depth));

  const StorageView q_cpu({batch, heads, q_len, depth}, q_host, Device::CPU);
  const StorageView k_cpu({batch, heads, num_keys, depth}, k_host, Device::CPU);
  const StorageView v_cpu({batch, heads, num_keys, depth}, v_host, Device::CPU);
  std::unique_ptr<StorageView> len_cpu;
  if (lengths)
    len_cpu = std::make_unique<StorageView>(
        Shape{static_cast<dim_t>(lengths->size())}, *lengths, Device::CPU);
  StorageView ref(DataType::FLOAT32, Device::CPU);
  sdpa_reference_cpu(q_cpu, k_cpu, v_cpu, len_cpu.get(), scale, ref);

  const dim_t num_rows = batch * heads * q_len;

  // float32
  {
    StorageView q_gpu = q_cpu.to(Device::METAL);
    StorageView k_gpu = k_cpu.to(Device::METAL);
    StorageView v_gpu = v_cpu.to(Device::METAL);
    std::unique_ptr<StorageView> len_gpu;
    if (len_cpu)
      len_gpu = std::make_unique<StorageView>(len_cpu->to(Device::METAL));
    StorageView out({batch, heads, q_len, depth}, DataType::FLOAT32, Device::METAL);
    metal::sdpa(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(),
                len_gpu ? len_gpu->data<int32_t>() : nullptr, out.data<float>(),
                num_rows, q_len, num_keys, depth, scale);
    metal::synchronize();
    expect_storage_eq(out.to(Device::CPU), ref, 1e-5);
  }

  // float16 against the fp32 reference (house fp16 tolerance)
  {
    StorageView q_gpu = q_cpu.to(Device::METAL).to(DataType::FLOAT16);
    StorageView k_gpu = k_cpu.to(Device::METAL).to(DataType::FLOAT16);
    StorageView v_gpu = v_cpu.to(Device::METAL).to(DataType::FLOAT16);
    std::unique_ptr<StorageView> len_gpu;
    if (len_cpu)
      len_gpu = std::make_unique<StorageView>(len_cpu->to(Device::METAL));
    StorageView out({batch, heads, q_len, depth}, DataType::FLOAT16, Device::METAL);
    metal::sdpa(q_gpu.data<float16_t>(), k_gpu.data<float16_t>(), v_gpu.data<float16_t>(),
                len_gpu ? len_gpu->data<int32_t>() : nullptr, out.data<float16_t>(),
                num_rows, q_len, num_keys, depth, scale);
    metal::synchronize();
    expect_storage_eq(out.to_float32().to(Device::CPU), ref, 2e-2);
  }
}

}  // namespace

TEST_F(MetalTest, SdpaFusedParityWithReference) {
  // Greedy decode shape: q_len 1, d_head 64, mid-size cache.
  run_sdpa_case(2, 3, 1, 333, 64, nullptr, 11);
  // Head depth not a multiple of 32, tiny cache (fewer keys than SIMD-groups).
  run_sdpa_case(1, 2, 1, 3, 80, nullptr, 22);
  // Long key axis: online-softmax accumulation over many strided chunks.
  run_sdpa_case(1, 4, 1, 1500, 64, nullptr, 33);
  // Max supported head depth.
  run_sdpa_case(1, 2, 1, 64, 256, nullptr, 44);
}

TEST_F(MetalTest, SdpaFusedMaskedAndBeamParity) {
  // Beam-shaped q (2 rows per batch*head) with per-row lengths following the SoftMax
  // contract, including a fully masked row (len 0 → exact zero output) and rows shorter
  // than the SIMD-group count.
  const std::vector<int32_t> lengths = {57, 31, 1, 0, 44, 57,     // batch 0, 3 heads x 2 rows
                                        2, 57, 19, 3, 57, 5};     // batch 1
  run_sdpa_case(2, 3, 2, 57, 64, &lengths, 55);
}

// Metal multinomial draws its uniforms from the CT2 host generator, so the same seed
// must reproduce the same GPU samples (set_random_seed reseeds live generators).
TEST_F(MetalTest, MultinomialSeededReproducible) {
  const dim_t batch = 2;
  const dim_t depth = 1000;
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  std::vector<float> weights(batch * depth);
  for (auto& w : weights)
    w = dist(rng);
  StorageView probs({batch, depth}, weights, Device::METAL);

  const auto draw_sequence = [&]() {
    std::vector<int32_t> draws;
    for (int i = 0; i < 50; ++i) {
      StorageView output(DataType::INT32, Device::METAL);
      ops::Multinomial(1)(probs, output);
      const auto host = output.to(Device::CPU).to_vector<int32_t>();
      draws.insert(draws.end(), host.begin(), host.end());
    }
    return draws;
  };

  set_random_seed(1234);
  const auto first = draw_sequence();
  set_random_seed(1234);
  const auto second = draw_sequence();
  EXPECT_EQ(first, second);
  for (const int32_t d : first) {
    EXPECT_GE(d, 0);
    EXPECT_LT(d, depth);
  }
}

// The GumbelMax noise is -log(u), u ~ U(0,1): Exp(1) distributed. Check the GPU stream's
// sample moments and support (all positive, mean and variance near 1). This is the
// deterministic-given-noise half of the verification: the argmax/TopK that consumes the
// noised scores is covered bit-exactly by the TopK parity tests above.
TEST_F(MetalTest, GumbelNoiseStatistics) {
  const dim_t n = 100000;
  StorageView x({n}, 0.f, Device::METAL);
  StorageView y({n}, DataType::FLOAT32, Device::METAL);
  metal::add_gumbel_noise(x.data<float>(), y.data<float>(), n, /*seed=*/0x1234ABCDu);
  metal::synchronize();

  const auto noise = y.to(Device::CPU).to_vector<float>();
  double sum = 0, sum_sq = 0;
  float min_v = noise[0], max_v = noise[0];
  for (const float v : noise) {
    sum += v;
    sum_sq += double(v) * v;
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
  }
  const double mean = sum / n;
  const double var = sum_sq / n - mean * mean;
  EXPECT_GT(min_v, 0.f);          // u < 1 strictly
  EXPECT_LT(max_v, 20.f);         // 24-bit u: -log(u) <= ~16.6
  EXPECT_NEAR(mean, 1.0, 0.02);   // Exp(1) mean, sd of the estimate ~0.003
  EXPECT_NEAR(var, 1.0, 0.05);    // Exp(1) variance
}

// GumbelMax implements the same stochastic map on both backends (argmax of scores plus
// Exp(1) noise), so the sampled index distributions must agree within sampling error
// even though the RNG streams differ. Also covers fp16 on Metal, which previously threw
// (the CPU reference is only instantiated for float).
TEST_F(MetalTest, GumbelMaxMatchesCPUDistribution) {
  const dim_t depth = 8;
  const std::vector<float> logits = {1.2f, -0.5f, 2.0f, 0.0f, 0.7f, -1.0f, 1.5f, 0.3f};
  constexpr int num_draws = 5000;
  set_random_seed(99);

  const auto histogram = [&](Device device, DataType dtype) {
    StorageView x = StorageView({1, depth}, logits, device).to(dtype);
    std::vector<float> freq(depth, 0.f);
    for (int i = 0; i < num_draws; ++i) {
      StorageView indices(DataType::INT32, device);
      ops::GumbelMax(1)(x, indices);
      const auto idx = indices.to(Device::CPU).to_vector<int32_t>();
      if (idx[0] < 0 || idx[0] >= depth) {
        ADD_FAILURE() << "sampled index out of range: " << idx[0];
        continue;
      }
      freq[idx[0]] += 1.f / num_draws;
    }
    return freq;
  };

  const auto cpu_freq = histogram(Device::CPU, DataType::FLOAT32);
  const auto gpu_freq = histogram(Device::METAL, DataType::FLOAT32);
  const auto gpu_freq16 = histogram(Device::METAL, DataType::FLOAT16);
  for (dim_t i = 0; i < depth; ++i) {
    EXPECT_NEAR(gpu_freq[i], cpu_freq[i], 0.05) << "fp32 frequency mismatch at class " << i;
    EXPECT_NEAR(gpu_freq16[i], cpu_freq[i], 0.05) << "fp16 frequency mismatch at class " << i;
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
// INT8: quantize / dequantize / GEMM all run as native Metal kernels; the GEMM is the
// hand-tiled int8x8->int32 kernel (Phase 2), bit-exact at any depth. Parity oracle is
// the CPU int8 reference.
// (QuantizeINT8 / QuantizeINT8ZeroRow / GemmInt8 in ops_test.cc already run on METAL;
// these cover what the op suite does not: the dequantize forms, fp16, and GEMM depth.)
// ---------------------------------------------------------------------------

TEST_F(MetalTest, Int8DequantizeMatchesCPU) {
  const std::vector<int8_t> q_vec = {-127, -38, 64, 25, 30, 127, -18, 0};
  const std::vector<float> scale_vec = {12.7f, 6.047619f};

  StorageView ref;
  ops::Dequantize()(StorageView({2, 4}, q_vec), StorageView({2}, scale_vec), ref);

  StorageView q_m({2, 4}, q_vec, Device::METAL);
  StorageView scale_m({2}, scale_vec, Device::METAL);

  // float32 output uses the same reciprocal-then-multiply arithmetic as the CPU kernel.
  StorageView y32(DataType::FLOAT32, Device::METAL);
  ops::Dequantize()(q_m, scale_m, y32);
  expect_storage_eq(y32.to(Device::CPU), ref);

  // float16 output (int8 embeddings in an int8_float16 model) within half tolerance.
  StorageView y16(DataType::FLOAT16, Device::METAL);
  ops::Dequantize()(q_m, scale_m, y16);
  EXPECT_EQ(y16.dtype(), DataType::FLOAT16);
  expect_storage_eq(y16.to_float32().to(Device::CPU), ref, 2e-2);
}

TEST_F(MetalTest, Int8QuantizeFloat16MatchesFloat32) {
  // fp16 inputs have no CPU reference (the CPU Quantize is float32-only); quantize the
  // same fp16-representable values in fp32 on the CPU and expect identical int8 codes
  // and scales — the kernel reduces and rescales in float either way.
  const std::vector<float> in_vec = {-10, -3, 5, 2, 5, 21, -3, 0};
  const ops::Quantize quantize_op(ops::Quantize::ScaleType::GLOBAL,
                                  /*shift_to_uint8=*/false,
                                  /*round_before_cast=*/true);

  StorageView ref_q(DataType::INT8);
  StorageView ref_scale;
  quantize_op(StorageView({2, 4}, in_vec), ref_q, ref_scale);

  StorageView in16 = StorageView({2, 4}, in_vec, Device::METAL).to(DataType::FLOAT16);
  StorageView q(DataType::INT8, Device::METAL);
  StorageView scale(DataType::FLOAT32, Device::METAL);
  quantize_op(in16, q, scale);

  expect_storage_eq(q.to(Device::CPU), ref_q);
  expect_storage_eq(scale.to(Device::CPU), ref_scale);
}

TEST_F(MetalTest, Int8DequantizeGemmOutputMatchesCPU) {
  // The Dense epilogue: int32 accumulator / (a_scale[row] * b_scale[col]) + bias,
  // through every activation variant. GELU rides the hand-rolled ct2_erf and the
  // transcendentals are fast-math, so parity is tight-tolerance, not bit-exact.
  const std::vector<int32_t> c_vec = {-1205, 2249, -1269, -4226,
                                      -3697, 1272, 2436, -1676,
                                      -5560, -1767, -668, 6};
  const std::vector<float> a_scale_vec = {12.7f, 6.05f, 25.4f};
  const std::vector<float> b_scale_vec = {110.f, 95.5f, 130.2f, 80.75f};
  const std::vector<float> bias_vec = {0.6f, -0.4f, 0.1f, -1.2f};

  const StorageView c_cpu({3, 4}, c_vec);
  const StorageView a_scale_cpu({3}, a_scale_vec);
  const StorageView b_scale_cpu({4}, b_scale_vec);
  const StorageView bias_cpu({4}, bias_vec);

  const StorageView c_m({3, 4}, c_vec, Device::METAL);
  const StorageView a_scale_m({3}, a_scale_vec, Device::METAL);
  const StorageView b_scale_m({4}, b_scale_vec, Device::METAL);
  const StorageView bias_m({4}, bias_vec, Device::METAL);

  static const ops::ActivationType kReLU = ops::ActivationType::ReLU;
  static const ops::ActivationType kGELUTanh = ops::ActivationType::GELUTanh;
  static const ops::ActivationType kSwish = ops::ActivationType::Swish;
  static const ops::ActivationType kGELU = ops::ActivationType::GELU;
  static const ops::ActivationType kGELUSigmoid = ops::ActivationType::GELUSigmoid;
  static const ops::ActivationType kTanh = ops::ActivationType::Tanh;
  static const ops::ActivationType kSigmoid = ops::ActivationType::Sigmoid;

  const std::vector<const ops::ActivationType*> activations = {
    nullptr, &kReLU, &kGELUTanh, &kSwish, &kGELU, &kGELUSigmoid, &kTanh, &kSigmoid};

  for (const ops::ActivationType* act : activations) {
    const ops::Dequantize dequantize_op(act);
    for (const bool with_bias : {true, false}) {
      StorageView ref;
      dequantize_op(c_cpu, a_scale_cpu, b_scale_cpu,
                    /*transpose_a=*/false, /*transpose_b=*/true,
                    ref, with_bias ? &bias_cpu : nullptr);

      StorageView y32(DataType::FLOAT32, Device::METAL);
      dequantize_op(c_m, a_scale_m, b_scale_m, false, true,
                    y32, with_bias ? &bias_m : nullptr);
      expect_storage_eq(y32.to(Device::CPU), ref, 1e-4);

      StorageView bias16 = bias_m.to(DataType::FLOAT16);
      StorageView y16(DataType::FLOAT16, Device::METAL);
      dequantize_op(c_m, a_scale_m, b_scale_m, false, true,
                    y16, with_bias ? &bias16 : nullptr);
      EXPECT_EQ(y16.dtype(), DataType::FLOAT16);
      expect_storage_eq(y16.to_float32().to(Device::CPU), ref, 2e-2);
    }
  }
}

TEST_F(MetalTest, Int8GemmDeepAccumulatorMatchesHostReference) {
  // Validate the int32 contract at a realistic LLM depth (k=2048, Dense's trans_b
  // layout) against a host int32 triple loop. The native int8 kernels accumulate in
  // int32 throughout, so this is bit-exact by construction — including the all-saturated
  // adversarial inputs the retired Phase-1 fp32 shim could not represent above 2^24.
  // Routing coverage: m = 3 / alpha = 1 takes the SIMD-group GEMV kernel; m = 16 /
  // alpha = 1 takes the Metal-4 MPP matmul2d path where available (pre-macOS-26 it
  // falls back to the tiled kernel — same contract either way); m = 16 / alpha = 2
  // pins the threadgroup-tiled kernel, which only MPP-ineligible calls reach now.
  // All three must hold the same exactness bar.
  struct Case { dim_t m; int32_t alpha; };
  for (const Case tc : {Case{3, 1}, Case{16, 1}, Case{16, 2}}) {
    const dim_t m = tc.m, n = 5, k = 2048;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-127, 127);

    std::vector<int8_t> a_vec(m * k), b_vec(n * k);
    for (auto& v : a_vec) v = static_cast<int8_t>(dist(rng));
    for (auto& v : b_vec) v = static_cast<int8_t>(dist(rng));

    std::vector<int32_t> expected_vec(m * n, 0);
    for (dim_t i = 0; i < m; ++i)
      for (dim_t j = 0; j < n; ++j) {
        int32_t acc = 0;
        for (dim_t kk = 0; kk < k; ++kk)
          acc += static_cast<int32_t>(a_vec[i * k + kk]) * static_cast<int32_t>(b_vec[j * k + kk]);
        expected_vec[i * n + j] = tc.alpha * acc;
      }

    StorageView a({m, k}, a_vec, Device::METAL);
    StorageView b({n, k}, b_vec, Device::METAL);
    StorageView c(DataType::INT32, Device::METAL);
    ops::Gemm(tc.alpha, /*beta=*/0, /*trans_a=*/false, /*trans_b=*/true)(a, b, c);

    expect_storage_eq(c.to(Device::CPU), StorageView({m, n}, expected_vec));
  }
}

TEST_F(MetalTest, Int8GemmSaturatedAccumulatorExact) {
  // All-saturated operands at k=2048 drive the accumulator to 2048 * 127 * 127 =
  // 33,032,192 — above fp32's 2^24 integer-exact range, so the retired Phase-1 fp32
  // shim could not have produced this value. Only a true int32 accumulation passes.
  const dim_t m = 2, n = 3, k = 2048;
  StorageView a({m, k}, std::vector<int8_t>(m * k, 127), Device::METAL);
  StorageView b({n, k}, std::vector<int8_t>(n * k, 127), Device::METAL);
  StorageView c(DataType::INT32, Device::METAL);
  ops::Gemm(/*alpha=*/1, /*beta=*/0, /*trans_a=*/false, /*trans_b=*/true)(a, b, c);

  const std::vector<int32_t> expected_vec(m * n, 2048 * 127 * 127);
  expect_storage_eq(c.to(Device::CPU), StorageView({m, n}, expected_vec));
}

TEST_F(MetalTest, Int8GemmAllTransposeCombinations) {
  // The kernel resolves both transpose flags at tile-load time; cover all four layouts
  // (the op suite only runs notrans/notrans and the Dense path only notrans/trans_b)
  // against a host int32 triple loop, with edge-size dims that exercise the tile guards.
  const dim_t m = 5, n = 7, k = 9;
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> dist(-127, 127);

  std::vector<int8_t> a_logical(m * k), b_logical(k * n);
  for (auto& v : a_logical) v = static_cast<int8_t>(dist(rng));
  for (auto& v : b_logical) v = static_cast<int8_t>(dist(rng));

  std::vector<int32_t> expected_vec(m * n, 0);
  for (dim_t i = 0; i < m; ++i)
    for (dim_t j = 0; j < n; ++j) {
      int32_t acc = 0;
      for (dim_t kk = 0; kk < k; ++kk)
        acc += static_cast<int32_t>(a_logical[i * k + kk])
             * static_cast<int32_t>(b_logical[kk * n + j]);
      expected_vec[i * n + j] = acc;
    }
  const StorageView expected({m, n}, expected_vec);

  for (const bool trans_a : {false, true}) {
    for (const bool trans_b : {false, true}) {
      std::vector<int8_t> a_vec(m * k), b_vec(k * n);
      for (dim_t i = 0; i < m; ++i)
        for (dim_t kk = 0; kk < k; ++kk)
          (trans_a ? a_vec[kk * m + i] : a_vec[i * k + kk]) = a_logical[i * k + kk];
      for (dim_t kk = 0; kk < k; ++kk)
        for (dim_t j = 0; j < n; ++j)
          (trans_b ? b_vec[j * k + kk] : b_vec[kk * n + j]) = b_logical[kk * n + j];

      StorageView a(trans_a ? Shape{k, m} : Shape{m, k}, a_vec, Device::METAL);
      StorageView b(trans_b ? Shape{n, k} : Shape{k, n}, b_vec, Device::METAL);
      StorageView c(DataType::INT32, Device::METAL);
      ops::Gemm(/*alpha=*/1, /*beta=*/0, trans_a, trans_b)(a, b, c);

      expect_storage_eq(c.to(Device::CPU), expected);
    }
  }
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

// Native int8 GEMM (ct2_gemm_s8) vs the MPS float GEMMs, square and Dense-shaped (the
// !trans_a && trans_b layout the quantized Dense always uses). GFLOPS counts the same
// 2*m*n*k madds for all dtypes so the columns compare directly.
TEST_F(MetalTest, DISABLED_BenchmarkGemmInt8) {
  std::cout << "\n=== int8 GEMM (Dense layout, trans_b), ms/iter and GFLOPS ===\n";
  struct GemmShape { dim_t m; dim_t n; dim_t k; };
  for (GemmShape s : {GemmShape{256, 256, 256}, GemmShape{1024, 1024, 1024},
                      GemmShape{2048, 2048, 2048},
                      // Qwen2.5-0.5B Dense shapes at prefill (m = batch*seq = 8*32):
                      GemmShape{256, 4864, 896}, GemmShape{256, 896, 4864},
                      // and at bs1 decode:
                      GemmShape{1, 4864, 896}, GemmShape{1, 151936, 896}}) {
    const int iters = (s.m * s.n * s.k > (dim_t)1 << 29) ? 8 : 30;
    const double flops = 2.0 * double(s.m) * s.n * s.k;

    {
      std::vector<int8_t> av(s.m * s.k, 3), bv(s.n * s.k, -5);
      StorageView a({s.m, s.k}, av, Device::METAL);
      StorageView b({s.n, s.k}, bv, Device::METAL);
      StorageView c(DataType::INT32, Device::METAL);
      const ops::Gemm gemm(1, 0, false, true);
      const double ms = time_ms(iters, [&] { gemm(a, b, c); synchronize_device(Device::METAL, 0); });
      std::cout << "  m=" << s.m << " n=" << s.n << " k=" << s.k
                << "  METAL int8:  " << ms << " ms,  " << flops / (ms * 1e6) << " GFLOPS\n";
    }
    for (DataType dt : {DataType::FLOAT16, DataType::FLOAT32}) {
      const std::vector<float> av(s.m * s.k, 0.01f), bv(s.n * s.k, 0.02f);
      StorageView a = StorageView({s.m, s.k}, av, Device::METAL).to(dt);
      StorageView b = StorageView({s.n, s.k}, bv, Device::METAL).to(dt);
      StorageView c(dt, Device::METAL);
      const ops::Gemm gemm(1, 0, false, true);
      const double ms = time_ms(iters, [&] { gemm(a, b, c); synchronize_device(Device::METAL, 0); });
      std::cout << "  m=" << s.m << " n=" << s.n << " k=" << s.k
                << "  METAL " << (dt == DataType::FLOAT16 ? "fp16" : "fp32")
                << ":  " << ms << " ms,  " << flops / (ms * 1e6) << " GFLOPS\n";
    }
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

// A/B the current degenerate 1xK MPSMatrixMultiplication against the dedicated
// MPSMatrixVectorMultiplication path on real Qwen2.5-0.5B decode shapes. Run with
// CT2_MPS_GEMV unset so the GEMM column remains the baseline implementation.
TEST_F(MetalTest, DISABLED_BenchmarkMpsGemv) {
  if (std::getenv("CT2_MPS_GEMV"))
    GTEST_SKIP() << "Unset CT2_MPS_GEMV for the direct GEMM-vs-GEMV A/B";

  std::cout << "\n=== MPS GEMM vs GEMV at m=1 (ms/iter) ===\n";
  struct Shape { dim_t n; dim_t k; int iters; };
  for (const Shape shape : {Shape{896, 896, 100},
                            Shape{2688, 896, 100},
                            Shape{4864, 896, 50},
                            Shape{896, 4864, 50},
                            Shape{151936, 896, 10}}) {
    for (const DataType dtype : {DataType::FLOAT16, DataType::FLOAT32}) {
      StorageView matrix({shape.n, shape.k}, dtype, Device::METAL);
      StorageView x({shape.k}, dtype, Device::METAL);
      StorageView y({shape.n}, dtype, Device::METAL);
      if (dtype == DataType::FLOAT16) {
        std::fill(matrix.data<float16_t>(), matrix.data<float16_t>() + matrix.size(),
                  static_cast<float16_t>(0.02f));
        std::fill(x.data<float16_t>(), x.data<float16_t>() + x.size(),
                  static_cast<float16_t>(0.01f));
      } else {
        std::fill(matrix.data<float>(), matrix.data<float>() + matrix.size(), 0.02f);
        std::fill(x.data<float>(), x.data<float>() + x.size(), 0.01f);
      }

      auto gemm = [&] {
        if (dtype == DataType::FLOAT16)
          metal::gemm(false, true, 1, shape.n, shape.k, 1.f,
                      x.data<float16_t>(), shape.k,
                      matrix.data<float16_t>(), shape.k,
                      0.f, y.data<float16_t>(), shape.n);
        else
          metal::gemm(false, true, 1, shape.n, shape.k, 1.f,
                      x.data<float>(), shape.k, matrix.data<float>(), shape.k,
                      0.f, y.data<float>(), shape.n);
      };
      auto gemv = [&] {
        if (dtype == DataType::FLOAT16)
          metal::gemv(false, shape.n, shape.k, 1.f,
                      matrix.data<float16_t>(), shape.k, x.data<float16_t>(),
                      0.f, y.data<float16_t>());
        else
          metal::gemv(false, shape.n, shape.k, 1.f,
                      matrix.data<float>(), shape.k, x.data<float>(),
                      0.f, y.data<float>());
      };

      auto flush_per_iter = [&](auto&& fn) {
        return time_ms(shape.iters, [&] {
          fn();
          synchronize_device(Device::METAL, 0);
        });
      };
      auto batched_encode = [&](auto&& fn) {
        fn();
        synchronize_device(Device::METAL, 0);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < shape.iters; ++i)
          fn();
        synchronize_device(Device::METAL, 0);
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / shape.iters;
      };

      auto best_of_four = [&](auto&& measure) {
        double best = std::numeric_limits<double>::max();
        for (int repeat = 0; repeat < 4; ++repeat)
          best = std::min(best, measure());
        return best;
      };

      const double gemm_flush = best_of_four([&] { return flush_per_iter(gemm); });
      const double gemv_flush = best_of_four([&] { return flush_per_iter(gemv); });
      const double gemm_batched = best_of_four([&] { return batched_encode(gemm); });
      const double gemv_batched = best_of_four([&] { return batched_encode(gemv); });
      std::cout << "  n=" << shape.n << " k=" << shape.k << " "
                << (dtype == DataType::FLOAT16 ? "fp16" : "fp32")
                << ": GEMM " << gemm_flush << " / " << gemm_batched
                << ", GEMV " << gemv_flush << " / " << gemv_batched
                << "  (flush / batched)\n";
    }
    std::cout << "\n";
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

// Focused gate for CT2_MPS_GEMV: batch-1 autoregressive decode keeps every Dense
// projection at m=1. A longer forced decode amplifies that path while one model load and
// warmup stay outside the timed region. Run this test in separate processes with the env
// switch unset/set; unlike BenchmarkLLM it deliberately excludes CPU and batch-8 controls.
TEST_F(MetalTest, DISABLED_BenchmarkLLMMpsGemv) {
  const char* model_env = std::getenv("CT2_LLM_MODEL");
  if (!model_env)
    GTEST_SKIP() << "Set CT2_LLM_MODEL to a converted decoder-only model directory";

  const std::vector<std::string> content =
      {"ing", "ion", "ent", "ate", "ation", "ort", "ame", "ist", "ers", "ass", "int", "urn"};
  std::vector<std::string> prompt;
  prompt.reserve(32);
  for (size_t i = 0; i < 32; ++i)
    prompt.push_back(content[i % content.size()]);

  GenerationOptions options;
  options.beam_size = 1;
  options.sampling_topk = 1;
  options.max_length = 128;
  options.min_length = 128;
  options.include_prompt_in_result = false;

  std::cout << "\n=== Qwen batch-1 decode, prompt=32, decode=128 ===\n";
  for (const ComputeType compute_type : {ComputeType::FLOAT32, ComputeType::FLOAT16}) {
    auto model = models::Model::load(model_env, Device::METAL, 0, compute_type);
    Generator generator(model);
    const std::vector<std::vector<std::string>> batch = {prompt};
    auto wait = [&] {
      auto futures = generator.generate_batch_async(batch, options);
      for (auto& future : futures)
        future.get();
    };

    wait();
    const double ms = time_ms(5, wait);
    std::cout << "  METAL " << (compute_type == ComputeType::FLOAT16 ? "fp16" : "fp32")
              << ": " << ms << " ms, " << (128.0 / (ms / 1000.0)) << " tok/s\n";
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
