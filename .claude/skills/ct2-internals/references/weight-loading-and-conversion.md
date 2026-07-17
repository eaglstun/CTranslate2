---
title: "Weight loading & conversion"
summary: >-
  The model-load weight pipeline in src/models/model.cc: Model::load reading
  the binary header, parsing config.json for quantization_type, the variable
  loop that consumes raw bytes into StorageView buffers (with tensor-parallel
  split/concat), then set_compute_type dtype conversion, device placement,
  alias registration, and process_linear_weights/initialize finalize. It
  details ensure_dtype's quantize/dequantize/cast paths keyed by the paired
  {name}_scale variable, is_quantizable (name ends in weight) versus
  is_convertible logic, and the load-time nature of the conversion. The
  centerpiece is the conv-weight guard: conv weights are forced to stay
  float_dtype on CUDA/Metal/DNNL because quantized convolution is unsupported
  — and the bug story where advertising int8 for Metal left an int8 Whisper's
  conv weights un-dequantized until Metal was added to the guard, fixed by one
  line, since Conv1D on Metal runs the float32 CPU reference over unified
  memory.
---

The model-load weight pipeline in `src/models/model.cc`: binary read → variable index →
per-variable dtype conversion (quantize / dequantize / cast) → device placement. The
compute-type decision feeding this is `compute-type-resolution.md`; the on-disk bytes are
`model-binary-format.md`.

**Sources (all citations from real lines):**

- `src/models/model.cc` (`Model::load`, `set_compute_type`, `ensure_dtype`)
- `include/ctranslate2/models/model.h` (the virtual hooks, `_variable_index`)
- `src/ops/conv1d.cc`, `src/storage_view.cc` (the conv-guard failure path)
- commit `351b1990` + `METAL_BACKEND.md` (downstream-validation section, the bug story)

---

## 1. The load pipeline (`Model::load`, `model.cc:560-781`)

Ordered steps, all on the **CPU** until the device move near the end:

1. Read header: binary version, spec name, spec revision (`model.cc:593-603`);
   `create_model(spec)` instantiates the right `Model` subclass (`model.cc:605`);
   `check_version` enforces backward-only compatibility (`model.cc:457-469`).
2. Parse `config.json` if present (`model.cc:613-617`); `quantization_type` in the config
   selects `QUANTIZATION_TYPE` (CT2 vs AWQ, `model.cc:635-636`).
3. **Variable loop** (`model.cc:638-742`): for each of `num_variables`, read name, rank,
   dims, dtype, byte count, then `consume` the raw bytes straight into a fresh
   `StorageView`'s buffer (`model.cc:657-658`). (Tensor-parallel builds split/concat the
   variable across ranks here, `model.cc:659-740`.) Each ends in
   `register_variable(name, variable)` (`model.cc:741`).
4. **Dtype conversion**: `set_compute_type(compute_type, device, device_index)`
   (`model.cc:744-758`) — for CT2-quantized models; AWQ models instead pin `FLOAT16` with
   `update_weight=false`.
5. **Device placement**: `set_device(device, device_index)` (`model.cc:760-761`).
6. Register **aliases** (binary version ≥ 3, `model.cc:763-774`) — including auto-aliasing
   `{alias}_scale` / `{alias}_zero` to the target's scale/zero.
7. Finalize: `process_linear_weights()` + the model's `initialize(model_reader)` hook
   under a `ScopedDeviceSetter` (`model.cc:776-780`).

## 2. The variable index and its hooks

Variables live in `_variable_index`, a name → `shared_ptr<StorageView>` map
(`model.h:193`). `register_variable` emplaces (`model.cc:272-274`);
`register_variable_alias` inserts a second name pointing at the **same** `shared_ptr`
(`model.cc:276-281`) — aliases cost no memory. These are `virtual` (`model.h:164-168`)
precisely so a model subclass can remap names when a newer spec revision renamed a weight.

## 3. Per-variable conversion (`set_compute_type`, `model.cc:176-233`)

`compute_type_to_data_type(effective)` yields `(weight_dtype, float_dtype)`
(`model.cc:190-193`), then one pass over the index:

- `is_quantizable(name)` — name ends with `"weight"` (`model.cc:287-289`) → convert to
  `weight_dtype` via `ensure_dtype` (with the conv exception, §5).
- else `is_convertible` (not a scalar, name lacks `"_scale"`, `model.cc:299-301`) and
  float → cast to `float_dtype` (`model.cc:227-230`). Scales and int attributes pass
  through untouched.

## 4. `ensure_dtype` — quantize / dequantize / cast (`model.cc:303-368`)

The scale variable is paired by naming convention: `{name}_scale` (`model.cc:306`). A
non-float stored weight **must** have one (int16 models without it get the legacy global
scale, `model.cc:312-316`; otherwise throw, `model.cc:317`). If the dtype already matches
the target, return immediately (`model.cc:322-323`). Otherwise four paths:

- **float → float**: plain `variable.to(target_dtype)` (`model.cc:332-334`).
- **int8/int16 → float**: `Dequantize` with the saved scale, then drop the now-useless
  scale variable (`model.cc:335-345`).
- **float → int8/int16**: `Quantize` (same logic as `model_spec.py`'s converter-side
  quantization; `round_before_cast` keyed to binary version ≥ 5, `model.cc:325-328`,
  `model.h:87-89`), then `register_variable` the new `{name}_scale` (`model.cc:347-355`).
- **int → int**: dequantize to float32, requantize, swap the scale (`model.cc:357-365`).

## 5. The conv-weight guard (the centerpiece gotcha)

Conv weights are 3D `(out, in, kernel)`. Inside the quantizable branch
(`model.cc:206-226`) they are temporarily reshaped to 2D for scale handling, and — the
guard — **forced to stay float on devices without quantized convolution**:

```cpp
// model.cc:212-221
// For CUDA, Metal, and DNNL backend, quantized convolution is not supported. Hence, convert to float_dtype.
if (device == Device::CUDA
  || device == Device::METAL
#ifdef CT2_WITH_DNNL
  || true
#endif
  ) {
  variable_weight_dtype = float_dtype;
}
```

So on those devices a conv weight's target is `float_dtype`, not `weight_dtype`: an
int8-saved conv weight takes `ensure_dtype`'s **dequantize** path back to float at load,
while every linear weight stays/becomes int8.

**The bug this branch fixed** (commit `351b1990`, found by the downstream harness — see
`METAL_BACKEND.md` "downstream validation"): once `mayiuse_int8` returned true for Metal,
int8 compute types resolved there, but the guard only listed CUDA/DNNL. An int8-converted
Whisper model's conv weights therefore matched the target dtype and `ensure_dtype`
early-returned (`model.cc:322-323`) — conv weights stayed **int8 with no dequantization**.
Conv1D on Metal runs via the CPU reference with an upcast that calls
`weight.to_float32()` (`conv1d.cc:59-61`), and `StorageView::to(DataType)` implements only
float↔float pairings, so Whisper died at first use with `Conversion from int8 to float32
is not yet implemented` (`storage_view.cc:120`). The fix is one line: add
`|| device == Device::METAL` to the guard.

**Why model.cc is the right layer**: load time is the only point where the saved
`{name}_scale` still exists alongside the weight, so the dequantize is a one-time, exact
inversion of the converter's quantization. Patching it downstream (in `Conv1D`) would mean
re-dequantizing a weight on every call for an op that has no quantized kernel anyway — and
the CUDA/DNNL precedent already lives here.

## 6. Device placement and finalize

- `set_device` → `move_variables` (`model.cc:100-121`): each non-scalar variable is
  replaced by `variable.to(device)` (`move_variables_to_device`, `model.cc:89-97` —
  scalars stay on CPU), then `synchronize_device` waits for asynchronous deallocations
  (`model.cc:120`). The same synchronize-on-teardown happens in `~Model`
  (`model.cc:159-164`).
- `process_linear_weights` (`model.cc:388-424`) is CPU-only (`model.cc:389-390`): it
  precomputes the u8s8s32 **compensation** term for int8 weights when the GEMM backend
  prefers unsigned input (`model.cc:411-414`, `cpu::prefer_u8s8s32_gemm`,
  `backend.cc:96-99`) and optionally **packs** 2D linear weights for MKL
  (`model.cc:416-422`).

### Relevance to the Metal backend

- The guard's Metal trigger is this branch's int8 work: advertising int8 in
  `mayiuse_int8` (`types.cc:154-160`) silently changed what model loading does to conv
  weights — capability flags and the load-time conversion table must move together.
- Conv1D on Metal is a CPU-reference op over unified memory (`METAL_BACKEND.md` "what runs
  where"), so "Metal lacks quantized conv" really means "the float32 CPU reference is the
  only conv there".
- The whole load pipeline (host-side `consume` into buffers, `ensure_dtype`, packing
  checks) needed **zero** Metal changes besides the guard: `variable.to(device)` lands the
  bytes in a real `MTLBuffer` via the Metal allocator, and unified memory keeps every
  CPU-side pointer valid. See the `apple-silicon` skill (`storage-and-synchronization.md`)
  for the buffer/storage-mode side.
