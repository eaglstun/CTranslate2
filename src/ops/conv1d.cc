#include "ctranslate2/ops/conv1d.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Conv1D::Conv1D(dim_t stride,
                   dim_t padding,
                   dim_t dilation,
                   dim_t groups,
                   const ActivationType* activation_type)
      : _stride(stride)
      , _padding(padding)
      , _dilation(dilation)
      , _groups(groups)
      , _activation_type(activation_type)
    {
    }

    void Conv1D::operator()(const StorageView& input,
                            const StorageView& weight,
                            const StorageView& bias,
                            StorageView& output,
                            const StorageView* qscale) const {
      operator()(input, weight, &bias, output, qscale);
    }

    void Conv1D::operator()(const StorageView& input,
                            const StorageView& weight,
                            StorageView& output,
                            const StorageView* qscale) const {
      operator()(input, weight, nullptr, output, qscale);
    }

    void Conv1D::operator()(const StorageView& input,
                            const StorageView& weight,
                            const StorageView* bias,
                            StorageView& output,
                            const StorageView* qscale) const {
      PROFILE("Conv1D");
      const dim_t batch_size = input.dim(0);
      const dim_t input_length = input.dim(2);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);
      const dim_t output_length = (
        input_length + (2 * _padding) - (_dilation * (kernel_size - 1) + 1)) / _stride + 1;

      output.resize({batch_size, out_channels, output_length});

#ifdef CT2_WITH_METAL
      // Conv1D has no Metal kernel; on Metal it runs the CPU reference over unified memory
      // (METAL binds to the CPU dispatch case). That reference is float32-only, so an fp16
      // input — e.g. Whisper's encoder conv stem in fp16 — would hit the generic
      // "only supports float types" throw. Upcast to fp32, run the proven CPU-reference path,
      // and downcast the result back to fp16. (Quantized conv keeps its own path: qscale is
      // null here for the float16 Whisper/Wav2Vec2 conv stems.)
      if (input.device() == Device::METAL && input.dtype() == DataType::FLOAT16) {
        const StorageView input32 = input.to_float32();
        const StorageView weight32 = weight.to_float32();
        StorageView output32(DataType::FLOAT32, input.device());
        output32.resize({batch_size, out_channels, output_length});
        if (bias) {
          const StorageView bias32 = bias->to_float32();
          compute<Device::CPU, float>(input32, weight32, &bias32, output32, qscale);
        } else {
          compute<Device::CPU, float>(input32, weight32, nullptr, output32, qscale);
        }
        output = output32.to_float16();
        return;
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH("Conv1D", input.device(), input.dtype(),
                                (compute<D, T>(input, weight, bias, output, qscale)));
    }

  }
}
