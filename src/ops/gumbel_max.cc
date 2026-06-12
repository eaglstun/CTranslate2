#include "ctranslate2/ops/gumbel_max.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include <cstdlib>
#  include "ctranslate2/random.h"
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    GumbelMax::GumbelMax(dim_t num_samples)
      : _num_samples(num_samples)
      , _topk_op(num_samples)
    {
    }

    void GumbelMax::operator()(const StorageView& x,
                               StorageView& values,
                               StorageView& indices) const {
      PROFILE("GumbelMax");

      StorageView y(x.shape(), x.dtype(), x.device());

#ifdef CT2_WITH_METAL
      // GPU noise keeps the perturb step on-device (and is the only fp16 path on Metal:
      // the CPU reference is only instantiated for float). The per-launch seed comes from
      // the CT2 host generator, so set_random_seed reproducibility holds; the noise
      // stream differs from the CPU std::mt19937 (bit-parity is not meaningful for an
      // RNG op). The TopK that follows routes to the GPU on its own.
      // CT2_NO_METAL_SAMPLING forces the CPU reference.
      static const bool metal_sampling_disabled = std::getenv("CT2_NO_METAL_SAMPLING") != nullptr;
      if (x.device() == Device::METAL && !metal_sampling_disabled
          && (x.dtype() == DataType::FLOAT32 || x.dtype() == DataType::FLOAT16)) {
        auto& generator = get_random_generator();
        const uint64_t seed = (uint64_t(generator()) << 32) | uint64_t(generator());
        if (x.dtype() == DataType::FLOAT32)
          metal::add_gumbel_noise(x.data<float>(), y.data<float>(), x.size(), seed);
        else
          metal::add_gumbel_noise(x.data<float16_t>(), y.data<float16_t>(), x.size(), seed);
      } else
#endif
      DEVICE_AND_FLOAT_DISPATCH("GumbelMax", x.device(), x.dtype(), (add_gumbel_noise<D, T>(x, y)));

      _topk_op(y, values, indices);

      Shape output_shape = x.shape();
      output_shape.back() = _num_samples;
      values.reshape(output_shape);
      indices.reshape(output_shape);
    }

    void GumbelMax::operator()(const StorageView& x, StorageView& indices) const {
      StorageView values(x.dtype(), x.device());
      operator()(x, values, indices);
    }

  }
}
