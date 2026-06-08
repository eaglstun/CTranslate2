#include "test_utils.h"

#ifdef CT2_WITH_METAL

#include <cstring>
#include <vector>

#include <ctranslate2/devices.h>
#include <ctranslate2/ops/ops.h>
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

  metal::add(a.data<float>(), b.data<float>(), c.data<float>(), a_host.size());

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

#endif  // CT2_WITH_METAL
