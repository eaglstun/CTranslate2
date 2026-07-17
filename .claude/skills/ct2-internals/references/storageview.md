---
title: "StorageView — the core data structure"
summary: >-
  The definitive reference for StorageView, CTranslate2's light row-major
  buffer wrapper that adds shape but no math semantics, with dtype and device
  resolved at runtime (plain _dtype/_device fields, not template params) so
  heterogeneous storages share a collection and the header stays flag-free.
  Details the members (typeless void* _data, the _allocated_size capacity vs
  _size logical-count split, Shape as vector<dim_t>) and semantics: reshape
  reinterprets shape without reallocating while resize/reserve is the single
  allocation point encoding the reuse policy (shrinking or re-fitting never
  reallocates, growing past capacity frees and reallocates uninitialized).
  Covers data access with ASSERT_DTYPE guards,
  copy_from/operator=/shallow_copy/view (non-owning views leave _allocator
  null so owns_data is false), to(Device)/to(DataType) conversions, lazy
  allocator binding, and the performance contract that clear() preserves
  capacity and caching allocators recycle buffers so allocation churn is a
  bug. Explains why runtime dtype/device forces the
  TYPE_DISPATCH/DEVICE_DISPATCH macros, and the Metal note that a Metal
  StorageView's buffer is a shared-storage MTLBuffer whose CPU-addressable
  contents pointer (unified memory) satisfies the same pointer-based contract,
  with allocation routed to the real Metal allocator via early-returns in
  allocator.cc/devices.cc.
semantic_id: "yx4w1whOom3TEX1NQ9VpvIMYp2vbzqthThVyE4z85s4GK_UXd-dtWoVwas11voYZhA0suu_5zY5-vYyrTvxvoQ"
---

Source:

- `include/ctranslate2/storage_view.h` (full interface)
- `src/storage_view.cc` (implementation)

## 1. What it is

`StorageView` is a light wrapper around an allocated, row-major buffer that adds shape information. It is **tensor-like storage without math semantics** — it holds data + shape and supports resize/reshape/copy/view, but does not implement arithmetic (math lives in `primitives`/`ops`). Crucially, **dtype and device are resolved at runtime**, not as template parameters, so heterogeneous storages can live in the same collection.

The header comment states the contract (`storage_view.h:46-52`):

```cpp
// This class is a light wrapper around an allocated buffer which adds shape information.
//
// 1. it can be resized, reshaped, copied, and assigned;
// 2. it can view an existing buffer to avoid memory copy;
// 3. the buffer can be of any type and uses dynamic type dispatch (to allow collections
//    of heterogeneous storages);
// 4. allocation is aligned by default to 64 bytes.
```

Class declaration (`storage_view.h:53-54`):

```cpp
class StorageView
{
```

Key members — runtime dtype/device, the allocator, the raw buffer pointer, capacity vs. logical size, and shape (`storage_view.h:241-249`):

```cpp
protected:
  DataType _dtype = DataType::FLOAT32;
  Device _device = Device::CPU;
  int _device_index = 0;
  Allocator* _allocator = nullptr;
  void* _data = nullptr;
  dim_t _allocated_size = 0;   // reserved capacity, in elements
  dim_t _size = 0;             // logical element count (product of _shape)
  Shape _shape;
```

Note `_data` is a typeless `void*`; the element type is only known via `_dtype` at runtime. `_allocated_size` (reserved capacity) is tracked separately from `_size` (the logical size) — this split is the basis of the allocation-reuse contract (section 3). `Shape` is just `std::vector<dim_t>` (`storage_view.h:29`).

Constructors set `_device_index` from the device via `get_device_index(device)` (e.g. `storage_view.cc:11-15`). Typed constructors derive `_dtype` from the template parameter through `DataTypeToEnum<T>::value` (e.g. `storage_view.cc:30-37`).

## 2. Key methods and semantics

### Shape / metadata

- `rank()` → `_shape.size()` (`storage_view.h:116-118`).
- `shape()` → const ref to `_shape` (`storage_view.h:120-122`).
- `dim(d)` → size of dimension `d`; negative `d` wraps from the end, bounds-checked by `GUARD_DIM` (`storage_view.h:124-129`).
- `stride(d)` → row-major stride via `compute_stride` (`storage_view.h:131-136`, `storage_view.h:38-44`).
- `size()` → `_size`; `is_scalar()` ⇔ `_size == 1 && _shape.empty()`; `empty()` ⇔ `_size == 0` (`storage_view.h:138-150`).
- `item_size()` → `sizeof(T)` for the runtime dtype, via `TYPE_DISPATCH` (`storage_view.cc:177-181`).

### reshape vs. resize

- `reshape(new_shape)` only **reinterprets** `_shape`; it never reallocates and never changes `_size`. It supports a single `-1` dimension inferred from the existing `_size`, and throws if the new shape's element count is incompatible with the current `_size` (`storage_view.cc:183-217`).
- `resize(new_shape)` changes the logical size: it computes the new element count, calls `reserve(new_size)`, then sets `_size` and `_shape` (`storage_view.cc:240-246`). Convenience forms: `resize(dim, new_size)`, `grow(dim, size)`, `shrink(dim, size)`, `resize_as(other)` (`storage_view.cc:248-269`). `expand_dims`/`squeeze` insert/remove a size-1 axis without touching data (`storage_view.cc:219-238`).

### The allocation rule (reserve / capacity)

`reserve(size)` is the single point of (de)allocation and encodes the reuse policy (`storage_view.cc:161-171`):

```cpp
StorageView& StorageView::reserve(dim_t size) {
  if (size <= _allocated_size)
    return *this;              // shrinking / fitting: NO realloc
  release();
  _allocator = &get_allocator(_device);
  _data = _allocator->allocate(size * item_size(), _device_index);
  if (_data == nullptr)
    THROW_RUNTIME_ERROR("failed to allocated memory");
  _allocated_size = size;
  return *this;
}
```

So: **resizing to a smaller (or equal) element count does not reallocate** — it keeps the existing buffer and only `_size`/`_shape` change. **Growing past the reserved capacity reallocates**, freeing the old buffer first; the new memory is uninitialized. The header documents the same policy on `reserve` (`storage_view.h:109-113`) and `reserved_memory()` returns `_allocated_size * item_size()` in bytes (`storage_view.cc:141-143`).

### data access

- `data<T>()` returns `static_cast<T*>(_data)` but first asserts the requested type matches `_dtype` via `ASSERT_DTYPE` (`storage_view.cc:316-326`, macro at `storage_view.h:12-18`). `buffer()` returns the raw `void*` with no type check (`storage_view.cc:308-314`).
- `index<T>({...})` computes a row-major offset from per-dim indices and bounds-checks against `_size` (`storage_view.cc:342-367`); `at<T>` / `as_scalar<T>` / `scalar_at<T>` build on it (`storage_view.h:190-224`).
- `to_vector<T>()` copies to host, routing through `to(Device::CPU)` first if the storage is off-device (`storage_view.cc:328-335`).

### copy / assignment

- `copy_from(other)` calls `resize_as(other)` then dispatches a typed element copy (`storage_view.cc:369-373`). The typed `copy_from(const T*, size, device, synchronous)` requires `size == _size`, handles cross-device CUDA copies under `CT2_WITH_CUDA`, and optionally synchronizes the stream (`storage_view.cc:408-429`).
- `operator=(const StorageView&)` releases only if device/index differ, adopts the other's device/index/dtype, then `copy_from`s (`storage_view.cc:271-281`). The move-assign and move-ctor just swap/steal members (`storage_view.cc:283-293`, `73-84`).
- `shallow_copy(other)` makes this a non-owning **view** of another's buffer (via `view`), then adopts its device info (`storage_view.cc:295-302`).
- `view(data, shape)` points `_data` at an external buffer without copying: it `release()`s any owned memory, sets `_data`, and `_allocated_size = _size = compute_size(shape)`. Because no allocation happened, `_allocator` stays null (`storage_view.cc:382-395`). `owns_data()` is simply `_allocator != nullptr` (`storage_view.cc:173-175`), so views report `false`.

### to(Device) / to(DataType) / move_to

- `to(Device)` constructs a fresh `StorageView` on the target device and `copy_from`s into it (`storage_view.cc:90-93`).
- `to(DataType)` returns `*this` unchanged if already that dtype; otherwise allocates a converted storage and runs `primitives<D>::convert` via `DEVICE_DISPATCH`. Only the float16/float32/bfloat16 pairings are implemented; anything else throws (`storage_view.cc:95-123`). `to_float16`/`to_float32` are thin wrappers.
- `move_to(device, dtype)` converts dtype first (if needed) then device, reassigning `*this` (`storage_view.cc:133-139`).

### allocator acquisition

The storage does not own an allocator until it allocates: `reserve` lazily binds `_allocator = &get_allocator(_device)` on first allocation (`storage_view.cc:165`). `release()` frees through that allocator at `_device_index` and nulls everything (`storage_view.cc:151-159`); the destructor calls `release()` (`storage_view.cc:86-88`). Caching allocators are what make buffer reuse cheap (section 3).

## 3. The performance contract

Avoiding allocation churn is the entire point of separating `_allocated_size` from `_size`. The reuse paths:

- **Shrinking or re-fitting is free.** `reserve(size)` early-returns when `size <= _allocated_size` (`storage_view.cc:162-163`), so a `StorageView` reused across loop iterations with a varying-but-bounded size allocates at most once (at its high-water mark).
- **Capacity is preserved across `clear()`.** `clear()` zeroes `_size` and `_shape` but leaves `_data`/`_allocated_size` intact — "memory is still reserved" (`storage_view.h:105`, `storage_view.cc:145-149`). Only `release()` frees.
- **Caching allocators reuse buffers.** When a (re)allocation is unavoidable, `get_allocator(_device)` hands back a caching allocator so the underlying device buffer is recycled rather than returned to the OS/driver.

Per `CLAUDE.md`: resizing smaller does not reallocate, caching allocators reuse buffers, and **allocation churn should be treated as a bug**. The performance-critical idiom is to keep a long-lived `StorageView` and `resize`/`copy_from` into it, never reconstructing it per call.

## 4. Why dtype/device are runtime-resolved

`_dtype` and `_device` are plain runtime fields (`storage_view.h:242-243`), not template parameters. This enables heterogeneous collections (point 3 of the class comment) and a flag-free public header, but it means every operation that needs the concrete C++ type or device-specific code path must **dispatch at runtime**. That is exactly why the dispatch macros exist and are used throughout this file:

- `TYPE_DISPATCH(_dtype, ...)` — picks the C++ type `T` from the runtime `DataType` (e.g. `item_size`, the typeless `view`, `copy_from`, printing) (`storage_view.cc:179`, `393`, `371`, `446`).
- `DEVICE_DISPATCH(_device, ...)` — binds `constexpr Device D` to call `primitives<D>::…` (e.g. `fill`, `convert`, `copy`) (`storage_view.cc:100`, `399`, `422`).
- `DEVICE_AND_TYPE_DISPATCH(_device, _dtype, ...)` — both at once (`zero`, `scalar_at`) (`storage_view.cc:404`, `378`).
- `ASSERT_DTYPE` guards `data<T>()` so a typed access against the wrong runtime dtype throws instead of silently reinterpreting bytes (`storage_view.h:12-18`, `storage_view.cc:318`).

See `dispatch-and-op-implementation.md` for how these macros expand and how ops layer dispatch on top of `StorageView`.

### Relevance to the Metal backend

On `Device::METAL`, a `StorageView`'s buffer is a shared-storage `MTLBuffer` whose `contents` pointer is CPU-addressable thanks to Apple Silicon's **unified memory**. That CPU-addressable pointer satisfies the same pointer-based `Allocator`/`StorageView` contract the CPU path expects, which is what lets the (CPU-bound) reference kernels operate directly on Metal-resident data. Allocation is routed to the **real Metal allocator** (not CPU memory) because `src/allocator.cc` and `src/devices.cc` early-return for `Device::METAL` — so `reserve`/`release` here hit `MTLBuffer`s, while compute dispatch is bound to the CPU case in `device_dispatch.h`. See the `apple-silicon` skill's `storage-and-synchronization.md` for the buffer storage modes and CPU↔GPU synchronization details.
