#include "ctranslate2/ops/relu.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    void ReLU::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("ReLU");
      y.resize_as(x);
#ifdef CT2_WITH_METAL
      if (x.device() == Device::METAL
          && (x.dtype() == DataType::FLOAT32 || x.dtype() == DataType::FLOAT16)) {
        if (x.dtype() == DataType::FLOAT32)
          metal::activation(x.data<float>(), y.data<float>(), x.size(), 0);
        else
          metal::activation(x.data<float16_t>(), y.data<float16_t>(), x.size(), 0);
        return;
      }
#endif
      DEVICE_AND_FLOAT_DISPATCH("ReLU", x.device(), x.dtype(), (compute<D, T>(x, y)));
    }

  }
}
