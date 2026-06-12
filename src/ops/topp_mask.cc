#include "ctranslate2/ops/topp_mask.h"

#include "ctranslate2/ops/softmax.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include <cstdlib>
#  include "metal/primitives.h"
#  include "metal/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

    TopPMask::TopPMask(const float p, const float mask_value)
      : _p(p)
      , _mask_value(mask_value)
    {
    }

    void TopPMask::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("TopPMask");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      StorageView probs(dtype, device);
      ops::SoftMax()(input, probs);

      output.resize_as(input);

#ifdef CT2_WITH_METAL
      if (device == Device::METAL) {
        // GPU TopPMask sorts the row in threadgroup memory, so it is limited to
        // metal::topp_mask_max_depth() classes (the CUDA kernel has the same shape of
        // limit); it mirrors the CPU reference exactly, including the sequential float
        // accumulation order, so the kept set is bit-identical (tie order is index-
        // ascending where the CPU std::sort leaves it unspecified). Larger inputs take
        // the CPU reference below. CT2_NO_METAL_SAMPLING forces the CPU reference.
        static const bool metal_sampling_disabled = std::getenv("CT2_NO_METAL_SAMPLING") != nullptr;
        const dim_t depth = input.dim(-1);
        const dim_t batch_size = input.size() / depth;
        if (!metal_sampling_disabled && depth <= metal::topp_mask_max_depth()) {
          if (dtype == DataType::FLOAT32) {
            metal::topp_mask(input.data<float>(), probs.data<float>(), output.data<float>(),
                             batch_size, depth, _p, _mask_value);
            return;
          }
          if (dtype == DataType::FLOAT16) {
            metal::topp_mask(input.data<float16_t>(), probs.data<float16_t>(),
                             output.data<float16_t>(), batch_size, depth, _p, _mask_value);
            return;
          }
        }
        // CPU-reference fallback for fp16: SoftMax above ran on the GPU asynchronously,
        // so flush before the CPU kernel reads probs/input over unified memory (the
        // coherence point METAL_DEVICE_CASE provides for normal CPU-ref ops). The kernel
        // works on fp16 directly, which the generic float dispatch rejects on a non-CUDA
        // build — call the fp16 CPU path directly.
        if (dtype == DataType::FLOAT16) {
          metal::synchronize();
          compute<Device::CPU, float16_t>(input, probs, output);
          return;
        }
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("TopPMask", device, dtype, (compute<D, T>(input, probs, output)));
    }

    dim_t TopPMask::max_num_classes(const Device device) {
      dim_t num_classes = 0;
      DEVICE_DISPATCH(device, num_classes = max_num_classes<D>());
      return num_classes;
    }

  }
}
