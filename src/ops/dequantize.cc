#include "ctranslate2/ops/dequantize.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    Dequantize::Dequantize(const ActivationType* activation_type)
      : _activation_type(activation_type)
    {
    }

    void Dequantize::operator()(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      PROFILE("Dequantize");
      output.resize_as(input);

      switch (input.dtype()) {
      case DataType::INT16: {
        if (input.device() != Device::CPU)
          throw std::invalid_argument("INT16 dequantization is only supported on CPU");
        if (!scale.is_scalar())
          throw std::invalid_argument("INT16 quantization scale should be a scalar value");
        dequantize<Device::CPU, int16_t, float>(input, scale, output);
        break;
      }

      case DataType::INT8: {
        const dim_t batch_size = input.size() / input.dim(-1);
        if (scale.size() != batch_size)
          throw std::invalid_argument("INT8 dequantization expects per-batch scales");

#ifdef CT2_WITH_METAL
        // GPU dequantization; also covers fp16 outputs (int8 embeddings in an
        // int8_float16 model), which the generic dispatch would reject without CUDA.
        if (output.device() == Device::METAL
            && (output.dtype() == DataType::FLOAT32 || output.dtype() == DataType::FLOAT16)) {
          const dim_t depth = input.dim(-1);
          if (output.dtype() == DataType::FLOAT32)
            metal::dequantize_s8(input.data<int8_t>(), scale.data<float>(),
                                 output.data<float>(), batch_size, depth);
          else
            metal::dequantize_s8(input.data<int8_t>(), scale.data<float>(),
                                 output.data<float16_t>(), batch_size, depth);
          break;
        }
#endif

        DEVICE_AND_FLOAT_DISPATCH("Dequantize", output.device(), output.dtype(),
                                  (dequantize<D, int8_t, T>(input, scale, output)));

        break;
      }

      default:
        throw std::invalid_argument("Dequantize: invalid quantized type " + dtype_name(input.dtype())
                                    + ", expected int8 or int16");
      }
    }

    void Dequantize::operator()(const StorageView& c,
                                const StorageView& a_scale,
                                const StorageView& b_scale,
                                const bool transpose_a,
                                const bool transpose_b,
                                StorageView& y,
                                const StorageView* bias) const {
      PROFILE("DequantizeGemmOutput");
      y.resize_as(c);

#ifdef CT2_WITH_METAL
      // The Dense epilogue layout: per-row activation scales, per-output-channel weight
      // scales (!trans_a && trans_b). Other scale layouts fall through to the CPU
      // reference. Also covers fp16 outputs (int8_float16 models).
      if (y.device() == Device::METAL
          && (y.dtype() == DataType::FLOAT32 || y.dtype() == DataType::FLOAT16)
          && !transpose_a && transpose_b
          && !a_scale.is_scalar() && !b_scale.is_scalar()) {
        const dim_t depth = c.dim(-1);
        const dim_t batch_size = c.size() / depth;
        if (a_scale.size() == batch_size && b_scale.size() == depth) {
          const int activation = _activation_type ? static_cast<int>(*_activation_type) : -1;
          if (y.dtype() == DataType::FLOAT32)
            metal::dequantize_gemm_output_s8(c.data<int32_t>(),
                                             a_scale.data<float>(), b_scale.data<float>(),
                                             bias ? bias->data<float>() : nullptr,
                                             y.data<float>(), batch_size, depth, activation);
          else
            metal::dequantize_gemm_output_s8(c.data<int32_t>(),
                                             a_scale.data<float>(), b_scale.data<float>(),
                                             bias ? bias->data<float16_t>() : nullptr,
                                             y.data<float16_t>(), batch_size, depth, activation);
          return;
        }
      }
#endif

      DEVICE_AND_FLOAT_DISPATCH(
        "DequantizeGemmOutput", y.device(), y.dtype(),
        (dequantize_gemm_output<D, T>(c, a_scale, b_scale, transpose_a, transpose_b, bias, y)));
    }

  }
}
