#include "ctranslate2/ops/tanh.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    void Tanh::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Tanh");

      y.resize_as(x);

#ifdef CT2_WITH_METAL
      if (x.device() == Device::METAL
          && (x.dtype() == DataType::FLOAT32 || x.dtype() == DataType::FLOAT16)) {
        if (x.dtype() == DataType::FLOAT32)
          metal::activation(x.data<float>(), y.data<float>(), x.size(), 5);
        else
          metal::activation(x.data<float16_t>(), y.data<float16_t>(), x.size(), 5);
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("Tanh", x.device(), x.dtype(),
                                (primitives<D>::tanh(x.data<T>(), y.data<T>(), x.size())));
    }

  }
}
