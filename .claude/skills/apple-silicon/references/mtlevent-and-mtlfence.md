---
title: "MTLFence, MTLEvent, MTLSharedEvent — the explicit sync primitives"
summary: >-
  Compares Metal's three explicit synchronization primitives by scope:
  MTLFence orders passes/encoders within one command queue
  (updateFence/waitForFence, ordering memory not scheduling), MTLEvent orders
  across command buffers/queues within one device via a monotonic 64-bit
  value, and MTLSharedEvent extends that to CPU-GPU and cross-device/process
  with a CPU-readable/settable signaledValue, notify blocks, blocking waits,
  and XPC handles. It stresses that for .tracked resources (the default)
  Metal's automatic hazard tracking orders queue-submitted work with no
  explicit sync. Crucially none of the three is used in the CT2 Metal backend
  today, and that is correct: one command queue with Shared tracked buffers
  and per-op async commit means FIFO order plus tracking handles GPU-GPU
  dependencies while g_last_committed/metal::flush() handles CPU-GPU. It names
  the triggers that would make them load-bearing: a second command queue,
  untracked MTLHeap resources, or finer-grained mid-stream CPU waits.
semantic_id: "yh384gqqo-3Tg319Qp0pnoegI2ubz6sHDtXymoq_wkxOCHkO5a8vHA3yXR17h6YYhi-Mlszz1cZe7YXnRrRu4Q"
---

Sources: https://developer.apple.com/documentation/metal/mtlfence,
https://developer.apple.com/documentation/metal/mtlevent,
https://developer.apple.com/documentation/metal/mtlsharedevent,
https://developer.apple.com/documentation/metal/resource-synchronization,
https://developer.apple.com/documentation/metal/synchronizing-events-between-a-gpu-and-the-cpu,
https://developer.apple.com/documentation/metal/mtlhazardtrackingmode
(fetched via DocC JSON, 2026-06-11).

Three primitives, three scopes:

| Primitive        | Scope                                                      | Create with                |
| ---------------- | ---------------------------------------------------------- | -------------------------- |
| `MTLFence`       | Between passes/encoders, within a command queue            | `device.makeFence()`       |
| `MTLEvent`       | Across command buffers/queues, within ONE device (GPU↔GPU) | `device.makeEvent()`       |
| `MTLSharedEvent` | Across devices, processes, and **CPU↔GPU**                 | `device.makeSharedEvent()` |

## When you need NONE of them: automatic hazard tracking

For resources whose `hazardTrackingMode` is `.tracked` (the default for individually
created buffers), work submitted through `MTLCommandEncoder`/`MTLCommandBuffer`/
`MTLCommandQueue` is ordered automatically: "Metal automatically delays write operations
until previous read operations finish, and prevents subsequent commands from running until
write operations complete." Explicit sync becomes load-bearing only for `.untracked`
resources (incl. heap suballocations, untracked by default — see `mtlheap.md`), for the
Metal 4 `MTL4CommandQueue` (never tracks), or for cross-queue/CPU ordering that tracking
can't see.

## MTLFence — ordering passes inside a queue

A fence "instructs the GPU to finish running specific stages of a pass before starting
stages from another pass." Producer updates, consumer waits:

- Compute encoder: `updateFence(_:)` after the producing commands, `waitForFence(_:)`
  before the consuming ones (blit/accel encoders have the same pair; render encoders take
  a stage argument).
- **Submission-order rule:** producing passes must be encoded/committed _before_ consuming
  passes — fences order memory, not scheduling.
- Apple-family GPUs update/wait per stage, so non-dependent stages of a waiting pass can
  still run.
- Apple's tip: for passes on _different_ queues, prefer an `MTLEvent`.

## MTLEvent — GPU↔GPU across command buffers

An event wraps a monotonically increasing unsigned 64-bit value starting at 0. On an
`MTLCommandBuffer`:

- `encodeSignalEvent(_:value:)` after the producing workload;
- `encodeWaitForEvent(_:value:)` before each consuming workload.

The device proceeds past a wait when the event's value ≥ the target. Rules from the page:
you can only signal with a value **greater** than the current one; one signal can unblock
many waiters; multiple producers cannot share one event (values only ratchet up — give each
producer its own event); waits that sit too long can time out (a timed-out wait in an
`MTLCommandBuffer` terminates it with a timeout error).

## MTLSharedEvent — CPU↔GPU (and cross-device/process)

Inherits `MTLEvent`, works anywhere a regular event does, plus:

- `signaledValue` — readable AND settable from the CPU (setting a larger value unblocks
  GPU waits and fires CPU notifications).
- `notify(_:atValue:block:)` — registers a block on an `MTLSharedEventListener` that runs
  when the value reaches the threshold (the CPU-side completion mechanism).
- `wait(untilSignaledValue:timeoutMS:)` — blocking CPU wait, returns `Bool` on timeout.
- `makeSharedEventHandle()` — portable handle to pass to another process over XPC.

Apple's tip: "Start with an MTLEvent instance until you need to synchronize work with a
task that runs on the CPU or another Metal device, because an MTLSharedEvent can add
overhead."

### Relevance to the CT2 Metal backend

- **None of the three is used today** (grep `MTLFence|MTLEvent|MTLSharedEvent` in
  `src/metal/` — zero hits), and that is correct, not an omission. The backend's model
  (see `storage-and-synchronization.md`) is: ONE command queue (`MetalContext()` in
  `src/metal/device.mm` creates a single `newCommandQueue`), all buffers Shared with
  default **tracked** hazard mode (`src/metal/allocator.mm`), per-op async commit. Queue
  FIFO order + automatic tracking orders all GPU→GPU dependencies; CPU→GPU ordering is the
  global `g_last_committed` + `metal::flush()` wait, which is a coarse
  completion-handler-style sync.
- **Concrete triggers that would make these load-bearing:**
  1. **A second `MTLCommandQueue`** — e.g. overlapping a weight-upload/blit queue with the
     compute queue. FIFO no longer orders across queues; `MTLEvent` signal/wait is the
     designed tool (Apple explicitly steers cross-queue sync to events).
  2. **Untracked resources** — adopting an `MTLHeap` (untracked by default) or flipping
     buffers to `hazardTrackingModeUntracked` for scheduler freedom. Every inter-pass RAW/WAR
     on those resources then needs `MTLFence` update/wait pairs in the encoders in
     `src/metal/primitives.mm` / `gemm.mm`.
  3. **Finer-grained CPU waits** — if `metal::flush()`'s wait-on-last-committed ever proves
     too coarse (CPU needs result of op K while ops K+1… are still queued), an
     `MTLSharedEvent` signaled mid-stream with `wait(untilSignaledValue:timeoutMS:)` is the
     API answer — at the documented overhead cost, so measure first.
