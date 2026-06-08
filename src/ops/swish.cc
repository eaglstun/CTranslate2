#include "ctranslate2/ops/swish.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    void Swish::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Swish");
      y.resize_as(x);
#ifdef CT2_WITH_METAL
      if (x.device() == Device::METAL
          && (x.dtype() == DataType::FLOAT32 || x.dtype() == DataType::FLOAT16)) {
        if (x.dtype() == DataType::FLOAT32)
          metal::activation(x.data<float>(), y.data<float>(), x.size(), 2);
        else
          metal::activation(x.data<float16_t>(), y.data<float16_t>(), x.size(), 2);
        return;
      }
#endif
      DEVICE_AND_FLOAT_DISPATCH("Swish", x.device(), x.dtype(), (compute<D, T>(x, y)));
    }

  }
}
