#include "ctranslate2/ops/multinomial.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include <cstdlib>
#  include <random>
#  include <vector>
#  include "ctranslate2/random.h"
#  include "metal/primitives.h"
#  include "metal/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

    Multinomial::Multinomial(dim_t sample_size)
      : _sample_size(sample_size) {
    }

    void Multinomial::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Multinomial");

      Shape output_shape = input.shape();
      output_shape.back() = _sample_size;
      output.resize(std::move(output_shape));

      dispatch(input, output);
    }

    void Multinomial::dispatch(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_METAL
      if (input.device() == Device::METAL) {
        // GPU inverse-CDF sampling keeps the draw on-device (no flush + CPU pass over the
        // distribution). The uniform draws come from the CT2 host generator, so
        // set_random_seed reproducibility holds; the stream differs from the CPU
        // std::discrete_distribution (bit-parity with the CPU draws is not meaningful for
        // an RNG op), but the sampled distribution is the same. Like the CUDA kernel,
        // only sample_size == 1 runs on the GPU. CT2_NO_METAL_SAMPLING forces the CPU
        // reference.
        static const bool metal_sampling_disabled = std::getenv("CT2_NO_METAL_SAMPLING") != nullptr;
        const dim_t depth = input.dim(-1);
        const dim_t batch_size = input.size() / depth;
        if (!metal_sampling_disabled && _sample_size == 1
            && batch_size <= metal::multinomial_max_batch_size()
            && (input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::FLOAT16)) {
          auto& generator = get_random_generator();
          std::uniform_real_distribution<float> distribution(0.f, 1.f);
          std::vector<float> uniforms(batch_size);
          for (auto& u : uniforms)
            u = 1.f - distribution(generator);  // in (0, 1]
          if (input.dtype() == DataType::FLOAT32)
            metal::multinomial(input.data<float>(), uniforms.data(), output.data<int32_t>(),
                               batch_size, depth);
          else
            metal::multinomial(input.data<float16_t>(), uniforms.data(), output.data<int32_t>(),
                               batch_size, depth);
          return;
        }
        // CPU-reference fallback for fp16: the input probabilities are produced
        // asynchronously on the GPU; flush before the CPU kernel reads them over unified
        // memory. The kernel works on fp16, which the generic float dispatch rejects on a
        // non-CUDA build — call the fp16 CPU path directly.
        if (input.dtype() == DataType::FLOAT16) {
          metal::synchronize();
          compute<Device::CPU, float16_t>(input, output);
          return;
        }
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("Multinomial", input.device(), input.dtype(),
                                (compute<D, T>(input, output)));
    }

  }
}
