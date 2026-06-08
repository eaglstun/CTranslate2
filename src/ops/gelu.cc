#include "ctranslate2/ops/gelu.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    GELU::GELU(const Approximation approximation)
      : _approximation(approximation)
    {
    }

    void GELU::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("GELU");

      y.resize_as(x);

#ifdef CT2_WITH_METAL
      if (x.device() == Device::METAL
          && (x.dtype() == DataType::FLOAT32 || x.dtype() == DataType::FLOAT16)) {
        // Match the ActivationType enum values used by ct2_apply_activation.
        const int act = _approximation == Approximation::Tanh ? 1
                      : _approximation == Approximation::Sigmoid ? 4
                      : 3;  // None -> exact GELU (erf)
        if (x.dtype() == DataType::FLOAT32)
          metal::activation(x.data<float>(), y.data<float>(), x.size(), act);
        else
          metal::activation(x.data<float16_t>(), y.data<float16_t>(), x.size(), act);
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("GELU", x.device(), x.dtype(), (compute<D, T>(x, y)));
    }

  }
}
