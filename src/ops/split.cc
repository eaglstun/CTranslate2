#include "ctranslate2/ops/split.h"

#include <numeric>

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    Split::Split(dim_t axis, bool no_copy)
      : _axis(axis)
      , _total_size(0)
      , _no_copy(no_copy) {
      check_arguments();
    }

    Split::Split(dim_t axis, const std::vector<dim_t>& split, bool no_copy)
      : _axis(axis)
      , _split(split)
      , _total_size(std::accumulate(split.begin(), split.end(), dim_t(0)))
      , _no_copy(no_copy) {
      check_arguments();
    }

    void Split::operator()(const StorageView& input,
                           StorageView& output1,
                           StorageView& output2) const {
      std::vector<StorageView*> outputs{&output1, &output2};
      operator()(input, outputs);
    }

    void Split::operator()(const StorageView& input,
                           StorageView& output1,
                           StorageView& output2,
                           StorageView& output3) const {
      std::vector<StorageView*> outputs{&output1, &output2, &output3};
      operator()(input, outputs);
    }

    void Split::operator()(const StorageView& input, std::vector<StorageView*>& outputs) const {
      PROFILE("Split");
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t dim = input.dim(axis);

      if (!_split.empty()) {
        if (_split.size() != outputs.size())
          throw std::invalid_argument(std::to_string(outputs.size())
                                      + " outputs are passed but "
                                      + std::to_string(_split.size())
                                      + " split sizes were configured");
        if (dim != _total_size)
          throw std::invalid_argument("axis " + std::to_string(axis) + " has dimension "
                                      + std::to_string(dim) + " but expected "
                                      + std::to_string(_total_size));

      } else if (dim % outputs.size() != 0)
        throw std::invalid_argument("axis " + std::to_string(axis) + " is not divisble by "
                                    + std::to_string(outputs.size()));

      dim_t offset = 0;
      for (size_t j = 0; j < outputs.size(); ++j) {
        auto& x = *outputs[j];
        auto shape = input.shape();
        const dim_t split_size = _split.empty() ? dim / outputs.size() : _split[j];
        shape[axis] = split_size;
        if (_no_copy) {
          TYPE_DISPATCH(input.dtype(),
                        x.view(const_cast<T*>(input.data<T>() + offset), std::move(shape)));
          offset += input.stride(0) * split_size;
        } else {
          x.resize(std::move(shape));
        }
      }

      if (!_no_copy) {
#ifdef CT2_WITH_METAL
        if (input.device() == Device::METAL) {
          const dim_t item = input.item_size();
          const dim_t src_step = input.dim(axis) * input.stride(axis);  // elements
          const void* in_base = nullptr;
          TYPE_DISPATCH(input.dtype(), in_base = input.data<T>());
          dim_t in_offset = 0;  // bytes
          for (StorageView* output : outputs) {
            dim_t copy_size = 1;
            for (dim_t i = axis; i < output->rank(); ++i)
              copy_size *= output->dim(i);
            if (copy_size == 0)
              continue;
            dim_t iter_size = 1;
            for (dim_t i = 0; i < axis; ++i)
              iter_size *= output->dim(i);
            void* out_ptr = nullptr;
            TYPE_DISPATCH(output->dtype(), out_ptr = output->data<T>());
            metal::strided_copy(static_cast<const char*>(in_base) + in_offset, out_ptr,
                                copy_size * item, src_step * item, copy_size * item, iter_size);
            in_offset += copy_size * item;
          }
          return;
        }
#endif
        DEVICE_AND_TYPE_DISPATCH(input.device(), input.dtype(), (compute<D, T>(input, outputs)));
      }
    }

    void Split::check_arguments() const {
      if (_no_copy && _axis != 0)
        throw std::invalid_argument("no_copy is only defined when splitting across the first dimension");
    }

  }
}
