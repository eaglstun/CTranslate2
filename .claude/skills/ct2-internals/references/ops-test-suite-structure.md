# The C++ Test Suite — Structure & Device Parameterization

How `tests/` is organized, how one test body runs on CPU/CUDA/METAL, and the tolerance
conventions — the oracle that any backend or quantization change lives and dies by.

**Sources (read these, all citations below are from real lines):**

- `tests/test_utils.h` / `tests/test_utils.cc` / `tests/test.cc`
- `tests/ops_test.cc` (the device-parameterized op suite + instantiations at the bottom)
- `tests/layers_test.cc`, `tests/model_test.cc`, `tests/translator_test.cc`
- `tests/CMakeLists.txt`

---

## 1. One binary, suite per layer

`ctranslate2_test` is a single Google Test binary built from all the suites
(`tests/CMakeLists.txt:7-19`): `ops_test.cc` (ops), `primitives_test.cc`,
`storage_view_test.cc`, `layers_test.cc`, `attention_test.cc`, `batching_test.cc`,
`decoding_test.cc`, `model_test.cc`, `translator_test.cc` (end-to-end), plus
`metal_test.cc` (always compiled into the binary; its body is `#ifdef CT2_WITH_METAL`).
Run it with the data directory as argv[1] — `main` stores it in `g_data_dir` and throws
without it (`tests/test.cc:7-13`):

```bash
./tests/ctranslate2_test ../tests/data
```

`metal_test.cc` is the Metal-specific suite (allocator, kernels, e2e decode parity,
`DISABLED_Benchmark*` micro-benchmarks, e.g. `metal_test.cc:578,607`) — covered by the
`apple-silicon` skill; everything below is the device-agnostic machinery.

## 2. Device parameterization — value-parameterized suites, not typed tests

Two fixture classes in `ops_test.cc` carry almost the whole op suite:

```cpp
// tests/ops_test.cc:121-125
class OpDeviceTest : public ::testing::TestWithParam<Device> {};
class OpDeviceFPTest : public ::testing::TestWithParam<FloatType> {};
```

`FloatType` is NOT a gtest typed-test — it's a plain struct bundling device + dtype +
**tolerance** (`tests/test_utils.h:73-77`: `Device device; DataType dtype; float error;`),
named in test output by dtype via `fp_test_name` (`test_utils.h:79-81`).

`INSTANTIATE_TEST_SUITE_P` at the **bottom of the file** decides what actually runs
(`ops_test.cc:1414-1432`):

- CPU: always — `OpDeviceTest` over `Device::CPU`; `OpDeviceFPTest` over
  `{CPU, FLOAT32, 1e-5}` (`:1414-1417`).
- CUDA: under `#ifdef CT2_WITH_CUDA` — fp32 `1e-5`, fp16 `1e-2`, bf16 `4e-2`
  (`:1418-1425`).
- METAL: under `#ifdef CT2_WITH_METAL` — fp32 `1e-5` only (`:1426-1432`; the comment at
  `:1427-1428` predates the fp16/int8 kernels — Metal fp16 parity is exercised in
  `metal_test.cc` instead).

`layers_test.cc` repeats the pattern for `LayerDeviceFPTest` (`layers_test.cc:261-275`);
`storage_view_test.cc:77-82` and `primitives_test.cc:54-56` parameterize on bare `Device`.

**There are no GUARD/skip macros for unavailable devices.** Gating is (a) compile-time
`#ifdef` around the instantiations, and (b) runtime `GTEST_SKIP()` where needed: the
`MetalTest` fixture skips when `!metal::has_gpu()` (`tests/metal_test.cc:32-38`), and
individual tests skip unimplemented combos — e.g. dilated conv on CPU/Metal
(`ops_test.cc:1259`), quantized conv on DNNL/CUDA (`ops_test.cc:1322-1326`). A CUDA build
run without a GPU is simply not a supported test configuration.

Tests that don't depend on device/dtype are plain `TEST(OpTest, ...)` (shape ops,
`ops_test.cc:6-37`); int8 Quantize round-trips run per-device as
`TEST_P(OpDeviceTest, QuantizeINT8)` (`ops_test.cc:915`).

## 3. `expect_storage_eq` and tolerances

The one comparison helper everything uses (`tests/test_utils.h:63-71`):

1. copies both StorageViews **to CPU** first (`got.to(Device::CPU)`) — so it works on any
   device, including Metal (unified memory makes this cheap);
2. asserts dtype and shape exactly;
3. `TYPE_DISPATCH`es into `expect_array_eq` element-wise.

Tolerance semantics (`test_utils.h:25-47`): `abs_diff == 0` means **exact** —
`EXPECT_EQ` for integer types, `EXPECT_FLOAT_EQ` for float (also used whenever the
expected value is non-finite); otherwise `EXPECT_NEAR(x, y, abs_diff)`. There is no
relative-epsilon mode — per-dtype slack comes entirely from the `FloatType.error`
injected by the instantiation (fp32 `1e-5`, fp16 `1e-2`, bf16 `4e-2`/`1e-2`), sometimes
widened per-test (`std::max(GetParam().error, float(3e-3))` for quantized conv,
`ops_test.cc:1327`). `ASSERT_RAISES(STMT, EXCEPT)` is the exception-checking macro
(`test_utils.h:14-23`).

## 4. Test data

`tests/data/models/` holds tiny real models used by `model_test.cc` and
`translator_test.cc`: `v1/` and `v2/` of `aren-transliteration` plus `-i16` and (v2)
`-i8` quantized variants. `default_model_dir()` returns
`<data_dir>/models/v2/aren-transliteration` (`tests/test_utils.cc:9-11`).
`ModelVariantTest` loads each variant × compute_type, checks effective weight dtypes, and
translates a fixed input (`translator_test.cc:86-97`) — this is the load-time
quantize/dequantize oracle (`weight-loading-and-conversion.md`). `tests/data/audio/`
(jfk/mr_quilter `.npy`) feeds Whisper-alignment tests; `tests/data/marian/` feeds the
Marian vocab converter test.

## 5. Recipe: an op test that covers all devices for free

1. Add `TEST_P(OpDeviceFPTest, MyOp)` in `ops_test.cc` (or `OpDeviceTest` if dtype-fixed).
2. Read `GetParam().device / .dtype / .error` (pattern at `ops_test.cc:128-131`).
3. Build inputs as fp32 CPU `StorageView`s with literal expected values, then move:
   `input.to(device).to(dtype)` (see Conv1D tests, e.g. `ops_test.cc:1246-1252`).
4. Compare with `expect_storage_eq(output.to_float32(), expected, error)`.
5. Done — the existing instantiations fan it out to CPU (+CUDA, +METAL when built).
   `GTEST_SKIP()` inside the body for combos a backend legitimately lacks.

Filter one case: `--gtest_filter='*MyOp*'`; one device: `METAL/OpDeviceFPTest.*`.

---

### Relevance to the Metal backend

- A graduated Metal kernel needs **zero new tests** for baseline parity: the METAL
  instantiations (`ops_test.cc:1426-1432`) already run every `TEST_P` over
  `Device::METAL`, and `expect_storage_eq`'s to-CPU copy reads unified memory directly.
- The generic suite is fp32-only on Metal; fp16 and int8 behavior (including the int8
  GEMM's bit-exact int32 checks and decode parity) live in `tests/metal_test.cc` — see
  the `apple-silicon` skill for that suite's layout.
- The int8 work's loop was exactly §5: the per-device `QuantizeINT8` tests plus
  `ModelVariantTest`'s i8 model caught scheme regressions before any model-level run.
- Run on this machine: build with `-DBUILD_TESTS=ON -DWITH_METAL=ON` (Apple Silicon CMake
  recipe in CLAUDE.md), then `./tests/ctranslate2_test ../tests/data`.
