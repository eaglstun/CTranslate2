#include "ctranslate2/ops/add.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    void Add::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Add");
#ifdef CT2_WITH_METAL
      if (a.device() == Device::METAL
          && (a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FLOAT16)) {
        c.resize_as(a);
        const bool b_is_scalar = b.is_scalar();
        if (a.dtype() == DataType::FLOAT32) {
          // The scalar operand may live on the host; read it by value.
          const float scalar = b_is_scalar ? b.data<float>()[0] : 0.f;
          metal::add(a.data<float>(), b_is_scalar ? nullptr : b.data<float>(),
                     c.data<float>(), c.size(), b_is_scalar, scalar);
        } else {
          const float scalar = b_is_scalar ? static_cast<float>(b.data<float16_t>()[0]) : 0.f;
          metal::add(a.data<float16_t>(), b_is_scalar ? nullptr : b.data<float16_t>(),
                     c.data<float16_t>(), c.size(), b_is_scalar, scalar);
        }
        return;
      }
#endif
      DEVICE_AND_TYPE_DISPATCH(a.device(), a.dtype(), (compute<D, T>(a, b, c)));
    }

  }
}
