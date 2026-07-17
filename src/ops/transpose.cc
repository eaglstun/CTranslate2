#include "ctranslate2/ops/transpose.h"

#include <limits>

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    Transpose::Transpose(const std::vector<dim_t>& perm)
      : _perm(perm) {
    }

    void Transpose::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Transpose");
      if (x.rank() <= 1) {
        y = x;
        return;
      }

      std::vector<dim_t> perm;
      bool identity = true;
      if (_perm.empty()) {
        perm.resize(x.rank());
        for (dim_t i = 0; i < x.rank(); ++i)
          perm[i] = x.rank() - i - 1;
        identity = false;
      } else {
        assert(_perm.size() == x.rank());
        perm = _perm;
        for (dim_t i = 0; i < x.rank(); ++i) {
          if (perm[i] != i) {
            identity = false;
            break;
          }
        }
      }

      if (identity) {
        y = x;
        return;
      }

#ifdef CT2_WITH_METAL
      // Native permute kernel: transpose is pure data movement, so dispatch on the element
      // width instead of the dtype. Falling through to the CPU reference would force a full
      // GPU queue drain per call — split/combine heads in beam-search decode hit this ~4x
      // per layer per token, which made the CPU-ref path the dominant serialization point.
      if (x.device() == Device::METAL
          && x.rank() <= 4
          && (x.item_size() == 1 || x.item_size() == 2 || x.item_size() == 4)
          && x.size() <= std::numeric_limits<uint32_t>::max()) {
        Shape out_shape(x.rank());
        for (dim_t i = 0; i < x.rank(); ++i)
          out_shape[i] = x.dim(perm[i]);
        y.resize(std::move(out_shape));

        dim_t in_strides[4];
        {
          dim_t stride = 1;
          for (dim_t k = x.rank() - 1; k >= 0; --k) {
            in_strides[k] = stride;
            stride *= x.dim(k);
          }
        }
        const dim_t pad = 4 - x.rank();
        dim_t out_dims[4] = {1, 1, 1, 1};
        dim_t strides_for_out[4] = {0, 0, 0, 0};
        for (dim_t i = 0; i < x.rank(); ++i) {
          out_dims[pad + i] = y.dim(i);
          strides_for_out[pad + i] = in_strides[perm[i]];
        }
        metal::transpose(x.buffer(), y.buffer(), out_dims, strides_for_out, x.item_size());
        return;
      }
#endif

      DEVICE_AND_TYPE_DISPATCH(x.device(), x.dtype(), (compute<D, T>(x, perm, y)));
    }

  }
}
