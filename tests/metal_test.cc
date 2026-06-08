#include "test_utils.h"

#ifdef CT2_WITH_METAL

#include <cstring>
#include <vector>

#include <ctranslate2/devices.h>
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

#endif  // CT2_WITH_METAL
