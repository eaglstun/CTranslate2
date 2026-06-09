#include "ctranslate2/ops/topk.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

    TopK::TopK(dim_t k, dim_t axis)
      : _k(k) {
      if (axis != -1)
        throw std::invalid_argument("unsupported topk axis " + std::to_string(axis));
    }

    void TopK::operator()(const StorageView& x, StorageView& values, StorageView& indices) const {
      PROFILE("TopK");
      const dim_t batch_size = x.size() / x.dim(-1);
      values.resize({batch_size, _k});
      indices.resize({batch_size, _k});

#ifdef CT2_WITH_METAL
      // The TopK kernel is comparison-based and works on fp16 directly (and runs on the
      // CPU reference over unified memory). The generic float dispatch rejects fp16 on a
      // non-CUDA build, so call the already-instantiated fp16 path directly.
      if (x.device() == Device::METAL && x.dtype() == DataType::FLOAT16) {
        // x is produced asynchronously on the GPU; flush before the CPU reference reads it
        // over unified memory (the coherence point METAL_DEVICE_CASE provides otherwise).
        metal::synchronize();
        compute<Device::CPU, float16_t, int32_t>(x, values, indices);
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("TopK", x.device(), x.dtype(),
                                (compute<D, T, int32_t>(x, values, indices)));
    }

  }
}
