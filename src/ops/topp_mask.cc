#include "ctranslate2/ops/topp_mask.h"

#include "ctranslate2/ops/softmax.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
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
      // SoftMax above runs on the Metal GPU for fp16 and commits asynchronously; the CPU
      // reference kernel below reads probs/input over unified memory, so flush the queued
      // GPU work first (the coherence point METAL_DEVICE_CASE provides for normal CPU-ref
      // ops). The kernel is sort/compare-based and works on fp16, which the generic float
      // dispatch rejects on a non-CUDA build — call the fp16 CPU path directly.
      if (device == Device::METAL && dtype == DataType::FLOAT16) {
        metal::synchronize();
        compute<Device::CPU, float16_t>(input, probs, output);
        return;
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
