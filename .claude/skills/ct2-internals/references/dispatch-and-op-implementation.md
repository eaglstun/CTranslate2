# Dispatch & Op Implementation

How CTranslate2 ops are structured and how they dispatch on device + dtype at runtime.

**Sources (read these, all citations below are from real lines):**

- `src/dispatch.h`
- `src/device_dispatch.h`
- `src/type_dispatch.h`
- `include/ctranslate2/ops/softmax.h` (flag-free op interface)
- `src/ops/softmax.cc` (input checks + dispatch + Metal targeted routing)
- `src/ops/softmax_cpu.cc` (CPU impl)
- `src/ops/softmax_gpu.cu` (CUDA impl)

---

## 1. The 4-file op pattern

Each op is split so the header in `include/` carries **no compilation flags** — the library is meant to be embedded, so backend selection (`WITH_CUDA`, `WITH_METAL`, ISA flags) must not leak into public headers. Flags live only in `.cc`/`.cu` files.

```text
include/ctranslate2/ops/softmax.h   # interface — NO compile flags
src/ops/softmax.cc                  # input checks + dispatch on device & dtype
src/ops/softmax_cpu.cc              # CPU implementation
src/ops/softmax_gpu.cu              # CUDA implementation
```

**Header (`include/ctranslate2/ops/softmax.h`)** — pure interface. The compute kernel is a private templated method `compute<D, T>`; the header never names a device backend or guards anything with `#ifdef`:

```cpp
// softmax.h:8-23
class SoftMax : public UnaryOp {
public:
  SoftMax(bool log = false);
  using UnaryOp::operator();
  void operator()(StorageView& x) const;
  void operator()(const StorageView& x, StorageView& y) const override;
  void operator()(const StorageView& x, const StorageView& lengths, StorageView& y) const;
  void operator()(const StorageView& x, const StorageView* lengths, StorageView& y) const;
private:
  template <Device D, typename T>
  void compute(const StorageView& input, const StorageView* lengths, StorageView& output) const;
  bool _log;
};
```

**`.cc`** — does input validation + the `DEVICE_AND_FLOAT_DISPATCH` (or similar) macro that resolves `<D, T>` at runtime, plus any targeted GPU routing (see §4). This is the only file that may `#include "dispatch.h"` and `#ifdef CT2_WITH_METAL`.

**`_cpu.cc` / `_gpu.cu`** — each provides a template specialization of `compute<D, T>` and explicitly instantiates it for its device/types. CPU instantiates `Device::CPU` (`softmax_cpu.cc:23-29`, only `float`); CUDA instantiates `Device::CUDA` for `float`, `float16_t`, `bfloat16_t` (`softmax_gpu.cu:33-40`). The CPU path further dispatches over runtime ISA via `CPU_ISA_DISPATCH` (`softmax_cpu.cc:15`).

---

## 2. The dispatch macros

The runtime trick throughout: a `switch` on a runtime enum (`Device` / `DataType`), where each `case` introduces a `constexpr` binding (`constexpr Device D = ...` / `typedef TYPE T`) so the body can be a template instantiation chosen at compile time but selected at runtime.

### `DEVICE_CASE` / `DEVICE_DISPATCH` (`device_dispatch.h`)

```cpp
// device_dispatch.h:17-22
#define DEVICE_CASE(DEVICE, STMT)               \
  case DEVICE: {                                \
    constexpr Device D = DEVICE;                \
    STMT;                                       \
    break;                                      \
  }
```

```cpp
// device_dispatch.h:49-63 — CUDA-off vs CUDA-on form
#ifndef CT2_WITH_CUDA
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    METAL_DEVICE_CASE(SINGLE_ARG(STMTS))                \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#else
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    METAL_DEVICE_CASE(SINGLE_ARG(STMTS))                \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#endif
```

`UNSUPPORTED_DEVICE_CASE` throws at runtime (`device_dispatch.h:11-15`); `SINGLE_ARG(...)` is `__VA_ARGS__`, used to pass comma-containing statements through the macro intact (`device_dispatch.h:24`).

### `TYPE_CASE` / `TYPE_DISPATCH` (`type_dispatch.h`)

`DataType` enum ↔ C++ type mapping is set up by `MATCH_TYPE_AND_ENUM` (`type_dispatch.h:29-48`), covering `float`/`int8_t`/`int16_t`/`int32_t`/`float16_t`/`bfloat16_t`.

```cpp
// type_dispatch.h:52-68
#define TYPE_CASE(TYPE, STMTS)                  \
  case DataTypeToEnum<TYPE>::value: {           \
    typedef TYPE T;                             \
    STMTS;                                      \
    break;                                      \
  }

#define TYPE_DISPATCH(TYPE_ENUM, STMTS)             \
  switch (TYPE_ENUM) {                              \
    TYPE_CASE(float, SINGLE_ARG(STMTS))             \
    TYPE_CASE(int8_t, SINGLE_ARG(STMTS))            \
    TYPE_CASE(int16_t, SINGLE_ARG(STMTS))           \
    TYPE_CASE(int32_t, SINGLE_ARG(STMTS))           \
    TYPE_CASE(float16_t, SINGLE_ARG(STMTS))         \
    TYPE_CASE(bfloat16_t, SINGLE_ARG(STMTS))        \
  }
```

### Combined macros (`dispatch.h`)

```cpp
// dispatch.h:6-7 — device then type
#define DEVICE_AND_TYPE_DISPATCH(DEVICE, TYPE, STMTS)   \
  DEVICE_DISPATCH(DEVICE, TYPE_DISPATCH(TYPE, (STMTS)))
```

`DEVICE_AND_FLOAT_DISPATCH(NAME, DEVICE, TYPE, STMTS)` is the float-restricted variant used by softmax. It has two forms gated on `CT2_WITH_CUDA` (`dispatch.h:15-43`). Without CUDA, only `float` is allowed and anything else hits `NON_FLOAT_CASE` which throws `"<NAME> only supports float types"` (`dispatch.h:10-12`):

```cpp
// dispatch.h:17-21 (CUDA-off)
#  define DEVICE_AND_FLOAT_DISPATCH(NAME, DEVICE, TYPE, STMTS)          \
  switch (TYPE) {                                                       \
    TYPE_CASE(float, DEVICE_DISPATCH(DEVICE, (STMTS)))                  \
    NON_FLOAT_CASE(NAME)                                                \
  }
```

With CUDA, `float16_t` / `bfloat16_t` are additionally accepted but **forced** to `constexpr Device D = Device::CUDA`, throwing if the device isn't CUDA (`dispatch.h:25-41`) — half precision is GPU-only.

So `DEVICE_AND_FLOAT_DISPATCH` switches on dtype first, then (for `float`) nests a `DEVICE_DISPATCH` to bind `D`. The net effect: both `D` and `T` become `constexpr`/typedef'd names usable to instantiate `compute<D, T>`.

---

## 3. The Metal special case

Metal is **not** a real `DEVICE_CASE`. It binds `constexpr Device D = Device::CPU` so the existing CPU `primitives<>`/`compute<Device::CPU,T>` run directly on Metal-resident buffers. The rationale is documented in the source (`device_dispatch.h:26-33`): Apple Silicon unified memory means a Metal buffer's `contents` pointer is CPU-addressable, and this avoids having to instantiate `primitives<Device::METAL>` / `compute<Device::METAL,T>` at all ~50 dispatch sites (which would break the link).

```cpp
// device_dispatch.h:34-47
#ifdef CT2_WITH_METAL
// Before running the CPU reference on Metal-resident memory, flush any pending GPU work so
// the CPU sees up-to-date data (no-op when nothing is queued). This is the coherence point
// that makes command-buffer batching safe.
#  define METAL_DEVICE_CASE(STMTS)              \
  case Device::METAL: {                         \
    ctranslate2::metal::flush();                \
    constexpr Device D = Device::CPU;           \
    STMTS;                                       \
    break;                                       \
  }
#else
#  define METAL_DEVICE_CASE(STMTS) UNSUPPORTED_DEVICE_CASE(Device::METAL)
#endif
```

Two consequences:

- When `CT2_WITH_METAL` is undefined, `Device::METAL` is an explicit unsupported (throwing) case — this also silences `-Wswitch` (`device_dispatch.h:33,46`).
- When defined, the case first calls `ctranslate2::metal::flush()` (coherence point — flush pending GPU work so the CPU reference reads up-to-date data) then runs the CPU kernel with `D = Device::CPU`. `metal/utils.h` is included for this (`device_dispatch.h:7-9`).

A real GPU kernel is reached **only** by targeted routing earlier in the op (§4), not through this case.

---

## 4. How a real op uses these — softmax walkthrough

`SoftMax::operator()` (`softmax.cc:32-66`) is the dispatch entry point. Sequence:

1. **Profiling + shape setup** (`softmax.cc:33-39`): `PROFILE(...)`, `y.resize_as(x)`, early-return if `depth == 0`.
2. **Input checks** (`softmax.cc:41-48`): if a `lengths` mask is given, validate `lengths->size() == batch_size`, else throw `std::invalid_argument` with the mismatched sizes.
3. **Metal targeted GPU routing** (`softmax.cc:50-63`), guarded by `#ifdef CT2_WITH_METAL`. Note it checks `x.device()` — the **real** device — even though the generic Metal dispatch case would bind `D = CPU`:

```cpp
// softmax.cc:53-62
if (x.device() == Device::METAL
    && (x.dtype() == DataType::FLOAT32 || x.dtype() == DataType::FLOAT16)) {
  const dim_t batch_size = x.size() / depth;
  const int32_t* len = lengths ? lengths->data<int32_t>() : nullptr;
  if (x.dtype() == DataType::FLOAT32)
    metal::softmax(_log, x.data<float>(), len, y.data<float>(), batch_size, depth);
  else
    metal::softmax(_log, x.data<float16_t>(), len, y.data<float16_t>(), batch_size, depth);
  return;   // <-- returns BEFORE the generic dispatch
}
```

This calls a `metal::` entry point (declared via `metal/primitives.h`, included at `softmax.cc:5-7`) and **returns before** reaching the generic dispatch. fp16 is handled here on GPU even though the CUDA-style float dispatch would otherwise reject fp16 on non-CUDA devices.

4. **Generic dispatch** (`softmax.cc:65`) for CPU and CUDA:

```cpp
DEVICE_AND_FLOAT_DISPATCH("SoftMax", x.device(), x.dtype(), (compute<D, T>(x, lengths, y)));
```

This expands (via §2) to a dtype switch around a device switch, binding `D` and `T` and instantiating `SoftMax::compute<D, T>`. The matching specialization lives in `softmax_cpu.cc` (`compute<Device::CPU, T>`, then `CPU_ISA_DISPATCH` into `cpu::softmax<ISA>`) or `softmax_gpu.cu` (`compute<Device::CUDA, T>` → `softmax_kernel` on the CUDA stream).

---

### Relevance to the Metal backend

Adding a GPU kernel for an op is **targeted routing at the `operator()` level** — guard with `#ifdef CT2_WITH_METAL`, check `x.device() == Device::METAL` (the real device), call the `metal::` entry point, and `return` **before** the generic `DEVICE_*_DISPATCH` (exactly as `softmax.cc:50-63` does). You do **NOT** add a real `DEVICE_CASE` for `Device::METAL`; the `METAL_DEVICE_CASE` deliberately stays bound to `constexpr Device D = Device::CPU` (the CPU reference fallback for any op not yet graduated), which is what keeps `primitives<Device::METAL>` from being instantiated at ~50 sites. For the full graduation procedure (entry-point signature, `flush()` coherence, fp16 handling, parity testing), see the apple-silicon skill's `op-graduation-playbook.md`.
