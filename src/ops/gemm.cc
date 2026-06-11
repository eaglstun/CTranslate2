#include "ctranslate2/ops/gemm.h"

#include "ctranslate2/ops/bias_add.h"

#include "dispatch.h"

#ifdef CT2_WITH_METAL
#  include <type_traits>
#  include "metal/primitives.h"
#endif

namespace ctranslate2 {
  namespace ops {

    // activation_type(x + bias + residual)
    void apply_bias_and_activation(StorageView& x,
                                   const StorageView* bias,
                                   const ActivationType* activation_type,
                                   const StorageView* residual,
                                   const dim_t axis) {
      if (bias) {
        const BiasAdd bias_add_op(activation_type, axis);
        bias_add_op(x, *bias, x, residual);
      } else {
        if (residual)
          Add()(*residual, x, x);
        if (activation_type)
          get_activation_op(*activation_type)(x, x);
      }
    }


    Gemm::Gemm(float alpha,
               float beta,
               bool trans_a,
               bool trans_b,
               bool a_is_packed,
               bool b_is_packed,
               const ActivationType* activation_type)
      : _alpha(alpha)
      , _beta(beta)
      , _trans_a(trans_a)
      , _trans_b(trans_b)
      , _a_is_packed(a_is_packed)
      , _b_is_packed(b_is_packed)
      , _activation_type(activation_type)
    {
    }

#ifdef CT2_WITH_METAL
    // Float GEMM on the GPU via MPS, sharing the shape logic with compute(). Reached from
    // operator() before the generic dispatch so it also covers fp16. Bias/activation are
    // applied afterwards by the caller, exactly as for the CPU/CUDA paths.
    template <typename T>
    static void metal_gemm(float alpha, float beta, bool trans_a, bool trans_b,
                           const StorageView& a, const StorageView& b, StorageView& c) {
      const dim_t k = a.dim(trans_a ? -2 : -1);
      const dim_t n = b.dim(trans_b ? -2 : -1);
      const dim_t m = a.size() / k;  // Collapse leading dimensions.
      const dim_t lda = trans_a ? m : k;
      const dim_t ldb = trans_b ? k : n;
      const dim_t ldc = n;

      Shape output_shape(a.shape());
      output_shape[output_shape.size() - 2] = a.dim(trans_a ? -1 : -2);  // m
      output_shape[output_shape.size() - 1] = n;
      c.resize(std::move(output_shape));

      metal::gemm(trans_a, trans_b, m, n, k, alpha,
                  a.data<T>(), lda, b.data<T>(), ldb, beta, c.data<T>(), ldc);
    }

    // Native int8 GEMM (Phase 2): int8 operands feed the hand-tiled int8x8->int32 MSL
    // kernel directly, so quantized weights stay int8-resident on the GPU (no per-call
    // widening) and accumulation is bit-exact int32 at any depth. The caller guarantees
    // beta == 0 and an integral alpha (a float alpha cannot be applied exactly to an
    // int32 accumulator); anything else falls through to the generic dispatch.
    static void metal_gemm_int8(float alpha, bool trans_a, bool trans_b,
                                const StorageView& a, const StorageView& b, StorageView& c) {
      const dim_t k = a.dim(trans_a ? -2 : -1);
      const dim_t n = b.dim(trans_b ? -2 : -1);
      const dim_t m = a.size() / k;  // Collapse leading dimensions.
      const dim_t lda = trans_a ? m : k;
      const dim_t ldb = trans_b ? k : n;
      const dim_t ldc = n;

      Shape output_shape(a.shape());
      output_shape[output_shape.size() - 2] = a.dim(trans_a ? -1 : -2);  // m
      output_shape[output_shape.size() - 1] = n;
      c.resize(std::move(output_shape));

      metal::gemm_s8(trans_a, trans_b, m, n, k, static_cast<int32_t>(alpha),
                     a.data<int8_t>(), lda, b.data<int8_t>(), ldb,
                     c.data<int32_t>(), ldc);
    }
#endif

    void Gemm::operator()(const StorageView& a,
                          const StorageView& b,
                          StorageView& c,
                          const StorageView* a_shift_compensation,
                          const StorageView* bias,
                          const StorageView* residual) const {
      PROFILE("Gemm");

      switch (a.dtype()) {
      case DataType::INT8:
#ifdef CT2_WITH_METAL
        // The u8-shift compensation, beta != 0 and a non-integral alpha never occur on
        // Metal (the first is CPU-GEMM-backend-specific, the others unused by quantized
        // Dense and unrepresentable in an exact int32 accumulator); guard anyway so an
        // unexpected combination falls through rather than silently dropping terms.
        if (a.device() == Device::METAL && !a_shift_compensation && _beta == 0
            && _alpha == static_cast<float>(static_cast<int32_t>(_alpha))) {
          metal_gemm_int8(_alpha, _trans_a, _trans_b, a, b, c);
          break;
        }
#endif
        DEVICE_DISPATCH(a.device(), (compute<D, int8_t, int32_t>(a, b, c, a_shift_compensation)));
        break;

      case DataType::INT16:
        if (a.device() != Device::CPU)
          throw std::invalid_argument("INT16 GEMM is only supported on CPU");
        compute<Device::CPU, int16_t, int32_t>(a, b, c, a_shift_compensation);
        break;

      case DataType::FLOAT32:
      case DataType::FLOAT16:
      case DataType::BFLOAT16: {
#ifdef CT2_WITH_METAL
        if (a.device() == Device::METAL && a.dtype() == DataType::FLOAT32) {
          metal_gemm<float>(_alpha, _beta, _trans_a, _trans_b, a, b, c);
          break;
        }
        if (a.device() == Device::METAL && a.dtype() == DataType::FLOAT16) {
          metal_gemm<float16_t>(_alpha, _beta, _trans_a, _trans_b, a, b, c);
          break;
        }
#endif
        DEVICE_AND_FLOAT_DISPATCH("Gemm", a.device(), a.dtype(),
                                  (compute<D, T, T>(a, b, c, a_shift_compensation)));
        break;
      }

      default:
        throw std::invalid_argument("Gemm: unsupported input type " + dtype_name(a.dtype()));
      }

      apply_bias_and_activation(c, bias, _activation_type, residual);
    }

    template <Device D, typename In, typename Out>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       StorageView& c,
                       const StorageView* a_shift_compensation) const {
      const dim_t k = a.dim(_trans_a ? -2 : -1);
      const dim_t n = b.dim(_trans_b ? -2 : -1);
      const dim_t m = a.size() / k;  // Collapse leading dimensions.
      const dim_t lda = _trans_a ? m : k;
      const dim_t ldb = _trans_b ? k : n;
      const dim_t ldc = n;

      {
        Shape output_shape(a.shape());
        output_shape[output_shape.size() - 2] = a.dim(_trans_a ? -1 : -2); // m
        output_shape[output_shape.size() - 1] = n;
        c.resize(std::move(output_shape));
      }

      primitives<D>::gemm(_a_is_packed, _b_is_packed,
                          _trans_a, _trans_b,
                          m, n, k,
                          _alpha,
                          a.data<In>(), lda,
                          b.data<In>(), ldb,
                          _beta,
                          c.data<Out>(), ldc,
                          a_shift_compensation ? a_shift_compensation->data<Out>() : nullptr);
    }

    template <typename T>
    static void pack_b(const StorageView& b,
                       const bool transpose,
                       const dim_t k,
                       const dim_t n,
                       const float alpha,
                       StorageView& packed) {
      const T* src = b.data<T>();
      const dim_t pack_bytes = primitives<Device::CPU>::gemm_pack_b(src,
                                                                    transpose,
                                                                    k, n,
                                                                    alpha);

      if (pack_bytes == 0)  // Packed Gemm is not supported.
        throw std::runtime_error("Packed GEMM APIs are not supported by this GEMM backend");

      const dim_t pack_size = pack_bytes / sizeof (T);
      const dim_t b_size = b.size();

      // We want the packed storage to have the same shape as the original weight
      // so that operators can query its shape, but also have enough space to store
      // the packed data.
      packed.reserve(std::max(b_size, pack_size));
      packed.resize_as(b);

      primitives<Device::CPU>::gemm_pack_b(src,
                                           transpose,
                                           k, n,
                                           alpha,
                                           packed.data<T>());
    }

    StorageView Gemm::pack_b_input(const StorageView& b,
                                   const bool transpose,
                                   const dim_t k,
                                   const dim_t n,
                                   const float alpha) {
      if (b.device() != Device::CPU)
        throw std::invalid_argument("Packed GEMM APIs are only defined on CPU");

      DataType dtype = b.dtype();
      StorageView packed(dtype);

      switch (dtype) {
      case DataType::FLOAT32:
        pack_b<float>(b, transpose, k, n, alpha, packed);
        break;
      case DataType::INT16:
        pack_b<int16_t>(b, transpose, k, n, alpha, packed);
        break;
      case DataType::INT8:
        pack_b<int8_t>(b, transpose, k, n, alpha, packed);
        break;
      default:
        throw std::invalid_argument("Cannot pack GEMM input of type " + dtype_name(dtype));
        break;
      }

      return packed;
    }

    StorageView Gemm::compensate_u8_input(const StorageView& b,
                                          const bool transpose,
                                          const dim_t k,
                                          const dim_t n,
                                          const float alpha) {
      if (b.device() != Device::CPU && b.dtype() != DataType::INT8)
        throw std::invalid_argument("Unsigned input compensation is only defined for "
                                    "INT8 GEMM on CPU");

      StorageView compensation({n}, DataType::INT32);
      primitives<Device::CPU>::compute_u8_compensation(b.data<int8_t>(),
                                                       transpose,
                                                       k, n,
                                                       alpha,
                                                       compensation.data<int32_t>());
      return compensation;
    }

  }
}
