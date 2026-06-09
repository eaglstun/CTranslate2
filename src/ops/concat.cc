#include "ctranslate2/ops/concat.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    Concat::Concat(int axis)
      : _axis(axis) {
    }

    void Concat::operator()(const std::vector<const StorageView*>& inputs,
                            StorageView& output) const {
      PROFILE("Concat");
      const dim_t rank = inputs.front()->rank();
      const dim_t axis = _axis < 0 ? rank + _axis : _axis;
      dim_t concat_dims = 0;
      for (const StorageView* x : inputs) {
        assert(x->rank() == rank);
        concat_dims += x->dim(axis);
      }

      Shape output_shape(inputs.front()->shape());
      output_shape[axis] = concat_dims;
      output.resize(std::move(output_shape));

#ifdef CT2_WITH_METAL
      if (output.device() == Device::METAL) {
        const dim_t item = output.item_size();
        const dim_t dst_step = output.dim(axis) * output.stride(axis);  // elements
        void* out_base = nullptr;
        TYPE_DISPATCH(output.dtype(), out_base = output.data<T>());
        dim_t out_offset = 0;  // bytes
        for (const StorageView* input : inputs) {
          dim_t copy_size = 1;
          for (dim_t i = axis; i < input->rank(); ++i)
            copy_size *= input->dim(i);
          if (copy_size == 0)
            continue;
          dim_t iter_size = 1;
          for (dim_t i = 0; i < axis; ++i)
            iter_size *= input->dim(i);
          const void* in_ptr = nullptr;
          TYPE_DISPATCH(input->dtype(), in_ptr = input->data<T>());
          metal::strided_copy(in_ptr, static_cast<char*>(out_base) + out_offset,
                              copy_size * item, copy_size * item, dst_step * item, iter_size);
          out_offset += copy_size * item;
        }
        return;
      }
#endif

      DEVICE_AND_TYPE_DISPATCH(output.device(), output.dtype(), (compute<D, T>(inputs, output)));
    }

  }
}
