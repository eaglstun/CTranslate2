#include "ctranslate2/ops/quantize.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    const float Quantize::global_int16_scale = 1000;

    Quantize::Quantize(const ScaleType int16_scale_type,
                       const bool shift_to_uint8,
                       const bool round_before_cast)
      : _int16_scale_type(int16_scale_type)
      , _shift_to_uint8(shift_to_uint8)
      , _round_before_cast(round_before_cast)
    {
      if (int16_scale_type != ScaleType::GLOBAL && int16_scale_type != ScaleType::PER_LAYER)
        throw std::invalid_argument("INT16 quantization only supports GLOBAL and PER_LAYER scales");
    }

    void Quantize::operator()(const StorageView& input,
                              StorageView& output,
                              StorageView& scale) const {
      PROFILE("Quantize");
      output.resize_as(input);

      switch (output.dtype()) {
      case DataType::INT16: {
        if (input.device() != Device::CPU)
          throw std::invalid_argument("INT16 quantization is only supported on CPU");
        quantize<Device::CPU, float, int16_t>(input, output, scale);
        break;
      }

      case DataType::INT8: {
        const dim_t depth = input.dim(-1);
        const dim_t batch_size = input.size() / depth;
        scale.resize({batch_size});

#ifdef CT2_WITH_METAL
        // Per-row signed int8 quantization on the GPU; also covers fp16 inputs, which the
        // generic float dispatch below would reject in a non-CUDA build. The u8-shift
        // variant is a CPU GEMM-backend (u8s8s32) concern and falls through to the CPU
        // reference over unified memory.
        if (input.device() == Device::METAL
            && !_shift_to_uint8
            && (input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::FLOAT16)) {
          if (input.dtype() == DataType::FLOAT32)
            metal::quantize_s8(input.data<float>(), output.data<int8_t>(), scale.data<float>(),
                               batch_size, depth, _round_before_cast);
          else
            metal::quantize_s8(input.data<float16_t>(), output.data<int8_t>(), scale.data<float>(),
                               batch_size, depth, _round_before_cast);
          break;
        }
#endif

        DEVICE_AND_FLOAT_DISPATCH("Quantize", input.device(), input.dtype(),
                                  (quantize<D, T, int8_t>(input, output, scale)));

        break;
      }

      default:
        throw std::invalid_argument("Quantize: invalid quantized type " + dtype_name(output.dtype())
                                    + ", expected int8 or int16");
      }
    }

  }
}
