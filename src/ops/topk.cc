#include "ctranslate2/ops/topk.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include <cstdlib>
#  include "metal/primitives.h"
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
      if (x.device() == Device::METAL) {
        // GPU TopK keeps the sampling step of the decode loop on-device (no flush + CPU
        // sort over the vocabulary). Selection is comparison-based: values are bit-copies
        // of the input, so it matches the CPU reference exactly (tie order is index-
        // ascending where the CPU partial_sort leaves it unspecified). k above the kernel
        // cap takes the CPU reference below. CT2_NO_METAL_SAMPLING forces the CPU
        // reference for all sampling ops (bisection lever, same spirit as CT2_NO_MPS_ACT).
        static const bool metal_sampling_disabled = std::getenv("CT2_NO_METAL_SAMPLING") != nullptr;
        const dim_t depth = x.dim(-1);
        if (!metal_sampling_disabled && _k <= depth && _k <= metal::topk_max_k()) {
          if (x.dtype() == DataType::FLOAT32) {
            metal::topk(x.data<float>(), values.data<float>(), indices.data<int32_t>(),
                        batch_size, depth, _k);
            return;
          }
          if (x.dtype() == DataType::FLOAT16) {
            metal::topk(x.data<float16_t>(), values.data<float16_t>(), indices.data<int32_t>(),
                        batch_size, depth, _k);
            return;
          }
        }
        // CPU-reference fallback for fp16: the kernel works on fp16 directly over unified
        // memory, but the generic float dispatch rejects fp16 on a non-CUDA build, so call
        // the already-instantiated fp16 path directly. Flush first so the CPU sees the
        // asynchronously produced input (the coherence point METAL_DEVICE_CASE provides
        // otherwise).
        if (x.dtype() == DataType::FLOAT16) {
          metal::synchronize();
          compute<Device::CPU, float16_t, int32_t>(x, values, indices);
          return;
        }
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("TopK", x.device(), x.dtype(),
                                (compute<D, T, int32_t>(x, values, indices)));
    }

  }
}
