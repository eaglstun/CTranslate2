// int8-on-Metal e2e gate + prefill timing for the MPP matmul2d integration.
// Greedy (deterministic) generation with ComputeType::INT8. Because the MPP kernel is
// bit-exact vs the tiled kernel, output tokens must be IDENTICAL with and without
// CT2_NO_MPP_GEMM=1 — diff the two runs externally. Also reports prefill-regime timing
// (long prompt, 1 step) so the doc gets an e2e number, not just kernel microbenchmarks.
//
// Usage: int8_e2e_check <model_dir> <device: cpu|metal> [steps=24] [compute=int8|float16]
//
// Build (from build/):
//   clang++ -std=c++17 -I../include -L. -lctranslate2 -Wl,-rpath,$PWD \
//     ../experiments/int8_e2e_check.cc -o /tmp/int8_e2e

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <ctranslate2/generator.h>
#include <ctranslate2/models/model.h>

using namespace ctranslate2;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <model_dir> <cpu|metal> [steps]\n";
    return 2;
  }
  const std::string model_dir = argv[1];
  const Device device = std::string(argv[2]) == "metal" ? Device::METAL : Device::CPU;
  const size_t steps = argc > 3 ? std::stoul(argv[3]) : 24;
  const ComputeType compute =
      (argc > 4 && std::string(argv[4]) == "float16") ? ComputeType::FLOAT16 : ComputeType::INT8;

  auto model = models::Model::load(model_dir, device, 0, compute);
  Generator generator(model);

  GenerationOptions options;
  options.beam_size = 1;
  options.sampling_topk = 1;
  options.max_length = steps;
  options.min_length = steps;
  options.include_prompt_in_result = false;

  // Decode-regime output (short prompt): the token-identity gate.
  {
    const std::vector<std::vector<std::string>> batch(
        1, {"The", " quick", " brown", " fox", " jumps", " over"});
    auto res = generator.generate_batch_async(batch, options)[0].get();
    std::cout << "tokens:";
    for (const auto& t : res.sequences[0])
      std::cout << t << "|";
    std::cout << "\n";
  }

  // Prefill-regime timing: batch 8 x 128-token prompt, 1 step, median of 5.
  {
    GenerationOptions popt = options;
    popt.max_length = 1;
    popt.min_length = 1;
    std::vector<std::string> prompt;
    for (int i = 0; i < 128; ++i)
      prompt.push_back(i % 2 ? " quick" : " fox");
    const std::vector<std::vector<std::string>> batch(8, prompt);
    generator.generate_batch_async(batch, popt)[0].get();  // warmup
    std::vector<double> ms;
    for (int rep = 0; rep < 5; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      auto futures = generator.generate_batch_async(batch, popt);
      for (auto& f : futures)
        f.get();
      auto t1 = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::cout << "prefill batch8x128 ms (min/med/max of 5): "
              << ms.front() << " / " << ms[ms.size() / 2] << " / " << ms.back() << "\n";
  }
  return 0;
}
