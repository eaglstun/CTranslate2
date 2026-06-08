#include "ctranslate2/ops/bias_add.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    BiasAdd::BiasAdd(const ActivationType* activation_type, const dim_t axis)
      : _activation_type(activation_type)
      , _axis(axis)
    {
    }

    void BiasAdd::operator()(const StorageView& value,
                             const StorageView& bias,
                             StorageView& output,
                             const StorageView* residual) const {
      PROFILE("BiasAdd");
      output.resize_as(value);

#ifdef CT2_WITH_METAL
      // Fused GPU bias-add + activation for the common last-axis case. Other axes fall
      // back to the CPU reference.
      if (value.device() == Device::METAL
          && (value.dtype() == DataType::FLOAT32 || value.dtype() == DataType::FLOAT16)
          && (_axis == -1 || _axis == value.rank() - 1)) {
        const dim_t depth = bias.size();
        const int activation = _activation_type ? static_cast<int>(*_activation_type) : -1;
        if (value.dtype() == DataType::FLOAT32)
          metal::bias_add(value.data<float>(), bias.data<float>(),
                          residual ? residual->data<float>() : nullptr,
                          output.data<float>(), value.size(), depth, activation);
        else
          metal::bias_add(value.data<float16_t>(), bias.data<float16_t>(),
                          residual ? residual->data<float16_t>() : nullptr,
                          output.data<float16_t>(), value.size(), depth, activation);
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("BiasAdd", value.device(), value.dtype(),
                                (compute<D, T>(value, bias, output, residual)));
    }

  }
}
