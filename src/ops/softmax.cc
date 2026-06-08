#include "ctranslate2/ops/softmax.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    LogSoftMax::LogSoftMax()
      : SoftMax(/*log=*/true) {
    }

    SoftMax::SoftMax(bool log)
      : _log(log) {
    }

    void SoftMax::operator()(StorageView& x) const {
      operator()(x, nullptr, x);
    }

    void SoftMax::operator()(const StorageView& x, StorageView& y) const {
      operator()(x, nullptr, y);
    }

    void SoftMax::operator()(const StorageView& x, const StorageView& lengths, StorageView& y) const {
      operator()(x, &lengths, y);
    }

    void SoftMax::operator()(const StorageView& x, const StorageView* lengths, StorageView& y) const {
      PROFILE(_log ? "LogSoftMax" : "SoftMax");
      y.resize_as(x);

      const dim_t depth = x.dim(-1);

      if (depth == 0)
        return;

      if (lengths) {
        const dim_t batch_size = x.size() / depth;
        if (lengths->size() != batch_size)
          throw std::invalid_argument("Length mask has size "
                                      + std::to_string(lengths->size())
                                      + " which is different than the current batch size "
                                      + std::to_string(batch_size));
      }

#ifdef CT2_WITH_METAL
      // Graduate float softmax to a real GPU kernel (a.device() is the real device even
      // though the M2 binding dispatches other work with D=CPU).
      if (x.device() == Device::METAL && x.dtype() == DataType::FLOAT32) {
        const dim_t batch_size = x.size() / depth;
        metal::softmax(_log,
                       x.data<float>(),
                       lengths ? lengths->data<int32_t>() : nullptr,
                       y.data<float>(),
                       batch_size,
                       depth);
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("SoftMax", x.device(), x.dtype(), (compute<D, T>(x, lengths, y)));
    }

  }
}
