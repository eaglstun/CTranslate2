#include "ctranslate2/ops/rotary.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    Rotary::Rotary(const dim_t ndims, const bool interleave)
      : _ndims(ndims)
      , _interleave(interleave)
    {
    }

    void Rotary::operator()(const StorageView& input,
                            const StorageView& sin,
                            const StorageView& cos,
                            StorageView& output,
                            bool is_transposed) const {
      PROFILE("Rotary");

      output.resize_as(input);

#ifdef CT2_WITH_METAL
      if (input.device() == Device::METAL
          && (input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::FLOAT16)) {
        const dim_t max_time = is_transposed ? input.dim(-2) : input.dim(-3);
        const dim_t depth = input.dim(-1);
        const dim_t batch_size = input.size() / (max_time * depth);
        const dim_t ndims = _ndims == 0 ? depth : _ndims;
        if (input.dtype() == DataType::FLOAT32)
          metal::rotary(input.data<float>(), sin.data<float>(), cos.data<float>(),
                        output.data<float>(), batch_size, max_time, ndims, depth, _interleave);
        else
          metal::rotary(input.data<float16_t>(), sin.data<float16_t>(), cos.data<float16_t>(),
                        output.data<float16_t>(), batch_size, max_time, ndims, depth, _interleave);
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("Rotary", input.device(), input.dtype(),
                                (compute<D, T>(input, sin, cos, output, is_transposed)));
    }

  }
}
