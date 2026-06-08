#include "ctranslate2/ops/layer_norm.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    LayerNorm::LayerNorm(const dim_t axis, const float epsilon)
      : _axis(axis)
      , _epsilon(epsilon)
    {
    }

    void LayerNorm::operator()(const StorageView& beta,
                               const StorageView& gamma,
                               const StorageView& input,
                               StorageView& output) const {
      operator()(&beta, &gamma, input, output);
    }

    void LayerNorm::operator()(StorageView& input) const {
      operator()(input, input);
    }

    void LayerNorm::operator()(const StorageView& input, StorageView& output) const {
      operator()(nullptr, nullptr, input, output);
    }

    void LayerNorm::operator()(const StorageView* beta,
                               const StorageView* gamma,
                               const StorageView& input,
                               StorageView& output) const {
      PROFILE("LayerNorm");
      output.resize_as(input);

      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t axis_size = input.dim(axis);

      dim_t inner_size = 1;
      dim_t outer_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        outer_size *= input.dim(i);
      for (dim_t i = axis + 1; i < input.rank(); ++i)
        inner_size *= input.dim(i);

#ifdef CT2_WITH_METAL
      // GPU path for the common case: normalizing the last axis with affine parameters.
      // Other axes / missing affine fall back to the CPU reference.
      if (input.device() == Device::METAL
          && (input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::FLOAT16)
          && axis == input.rank() - 1 && beta && gamma) {
        if (input.dtype() == DataType::FLOAT32)
          metal::layer_norm(input.data<float>(), gamma->data<float>(), beta->data<float>(),
                            output.data<float>(), outer_size, axis_size, _epsilon);
        else
          metal::layer_norm(input.data<float16_t>(), gamma->data<float16_t>(),
                            beta->data<float16_t>(), output.data<float16_t>(),
                            outer_size, axis_size, _epsilon);
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("LayerNorm", input.device(), input.dtype(),
                                (compute<D, T>(beta,
                                               gamma,
                                               input,
                                               axis,
                                               outer_size,
                                               axis_size,
                                               inner_size,
                                               output)));
    }

  }
}
