#include "ctranslate2/ops/multinomial.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
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
      // The input probabilities are produced asynchronously on the Metal GPU; the CPU
      // reference kernel below reads them over unified memory, so flush queued GPU work
      // first. The kernel is distribution/RNG-based and works on fp16, which the generic
      // float dispatch rejects on a non-CUDA build — call the fp16 CPU path directly.
      if (input.device() == Device::METAL && input.dtype() == DataType::FLOAT16) {
        metal::synchronize();
        compute<Device::CPU, float16_t>(input, output);
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("Multinomial", input.device(), input.dtype(),
                                (compute<D, T>(input, output)));
    }

  }
}
