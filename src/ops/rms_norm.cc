#include "ctranslate2/ops/rms_norm.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    RMSNorm::RMSNorm(const float epsilon, const bool use_residual)
      : _epsilon(epsilon)
      , _use_residual(use_residual)
    {
    }

    void RMSNorm::operator()(const StorageView& gamma,
                             const StorageView& input,
                             StorageView& output) const {
      PROFILE("RMSNorm");

      output.resize_as(input);

#ifdef CT2_WITH_METAL
      if (input.device() == Device::METAL
          && (input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::FLOAT16)) {
        const dim_t depth = input.dim(-1);
        const dim_t batch_size = input.size() / depth;
        if (input.dtype() == DataType::FLOAT32)
          metal::rms_norm(input.data<float>(), gamma.data<float>(), output.data<float>(),
                          batch_size, depth, _epsilon, _use_residual);
        else
          metal::rms_norm(input.data<float16_t>(), gamma.data<float16_t>(), output.data<float16_t>(),
                          batch_size, depth, _epsilon, _use_residual);
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("RMSNorm", input.device(), input.dtype(),
                                (compute<D, T>(gamma, input, output)));
    }

  }
}
