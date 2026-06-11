# Command-buffer errors & GPU hangs (failure diagnosis)

What a command buffer's lifecycle looks like, how the GPU reports runtime failures, and
— the repo-specific punchline — **this backend never checks any of it today**, so a
GPU-side fault currently surfaces as silent garbage output. This card is the
debugging-time bolt-on.

Sources (Apple DocC JSON, fetched 2026-06-11): `MTLCommandBuffer.status`, `.error`,
`addCompletedHandler(_:)`, `MTLCommandBufferStatus` (+ case pages),
`MTLCommandBufferError` (`mtlcommandbuffererror-swift.struct` + `/code`),
`MTLCommandBufferDescriptor.errorOptions`, `MTLCommandBufferErrorOption`,
`MTLCommandBufferEncoderInfo` — under
<https://developer.apple.com/tutorials/data/documentation/metal/...json>.

## The status lifecycle (`MTLCommandBufferStatus`)

`notEnqueued → enqueued → committed → scheduled → completed | error`

DocC: a buffer's state "can only change to a state below it in the table, and ends its
life cycle at either `completed` or `error`." The first two states are where encoding
happens; `committed` means the queue is resolving dependencies; `scheduled` means
resources are ready and it's waiting on the GPU; `error` is "the unsuccessful, final
state … the GPU stopped running the buffer's commands because of a runtime issue."

In this backend every buffer goes notEnqueued → committed in one breath
(`commit_command_buffer()`, no explicit `enqueue`), and only `flush()` ever observes a
final state — implicitly, via `waitUntilCompleted`, **which returns the same way for
`completed` and `error`.**

## `.error` and the real error codes

`MTLCommandBuffer.error` — "remains nil unless the GPU can't successfully run the
command buffer." The codes (`MTLCommandBufferError.Code`, DocC-verified list):

`none`, **`timeout`** (system interrupted/terminated the buffer before it finished — the
GPU-hang watchdog), **`pageFault`** ("the command buffer generated a page fault the GPU
can't service" — the classic kernel-OOB-pointer outcome), `notPermitted`,
`outOfMemory`, `invalidResource`, `memoryless`, `deviceRemoved`, `stackOverflow`,
`accessRevoked`, `internal`. Domain: `MTLCommandBufferErrorDomain`.

A hung/faulting kernel typically lands as `timeout` or `pageFault`, and macOS may also
log "GPU Restart"/discarded-work messages to the console; subsequent command buffers on
the queue can then error in cascade — so **the first error after a flush is the one
that matters**. (The cascade/console behavior is field knowledge, not DocC-verifiable;
the codes above are.)

## Extended fault info: `errorOptions` + encoder infos

By default "a GPU driver doesn't report additional error information." To get
per-encoder blame: create buffers from a `MTLCommandBufferDescriptor` with
`errorOptions = .encoderExecutionStatus`; on failure, `error.userInfo` then carries
`MTLCommandBufferEncoderInfo` objects ("additional information about a runtime
failure") with `label`, `debugSignposts`, and `errorState` per encoder. Costs some
driver bookkeeping — debugging-time only. Note the backend sets no labels on buffers or
encoders today (see `mtlbuffer-api.md` on `label` being the free win), so encoder infos
would come back anonymous until labels are added.

## What the repo does today (verified by grep, 2026-06-11)

`grep -n "status\|error\|addCompletedHandler" src/metal/*.mm`: the **only** waits are
`waitUntilCompleted` in `flush()` (`src/metal/device.mm`); `NSError**` checking exists
only for _pipeline/library creation_ in `MetalContext` — **no code reads
`commandBuffer.status` or `.error`, and no `addCompletedHandler` exists.** A pageFault
in `ct2_gemm_s8` would complete the flush normally and the CPU would read whatever is
in the output buffer.

The 5-line check worth adding to `flush()` when debugging (MRC, matches house style):

```objc
[to_wait waitUntilCompleted];
if (to_wait.status == MTLCommandBufferStatusError) {
  NSError* err = to_wait.error;
  throw std::runtime_error(std::string("Metal: command buffer failed: ")
      + (err ? [[err localizedDescription] UTF8String] : "(no error object)"));
}
```

Caveat: `g_last_committed` is only the _last_ buffer; an earlier buffer in the FIFO
could be the one that errored. For full coverage during a hunt, add an
`addCompletedHandler` in `commit_command_buffer()` instead — DocC explicitly calls the
completion handler "a good place to check the status property" — logging
`status == error` with the op's pipeline name. Handlers run on a driver CPU thread;
keep them to logging.

## Triage order for "output is garbage" on Metal

1. **Stale read** (most common here): a CPU read without `metal::flush()` — the value
   changes run-to-run or matches the _previous_ step. See
   `storage-and-synchronization.md`; also recall the project gotcha that mid-pipeline
   CPU reads of MPS-GEMM output are unreliable — read at layer boundaries.
2. **Error'd command buffer**: add the status check above. If it fires with
   `pageFault`/`invalidResource` → a binding or OOB bug; with `timeout` → a hang
   (infinite loop in a kernel, or a barrier not reached by all threads — see
   `threadgroup-and-simdgroup-synchronization.md`).
3. **Kernel OOB that doesn't fault** (scribbles inside a mapped buffer): the status
   check stays clean. Run with `MTL_SHADER_VALIDATION=1` and capture a `.gputrace` —
   `gpu-capture-and-shader-validation.md` is the recipe.
4. **Numeric, not memory** (NaN/overflow): none of the above will fire — use the
   CPU-ref bisection + layer-boundary tripwire approach that found the Gemma2 tanh NaN
   (`math-functions-and-numeric-parity.md`, `common-functions.md`).

### Relevance to the CT2 Metal backend

- The only completion point is `metal::flush()` in `src/metal/device.mm`
  (`g_last_committed` + `waitUntilCompleted`); `commit_command_buffer()` is where a
  debugging `addCompletedHandler` belongs. Neither checks `status`/`error` today — by
  design kept minimal, this card is the opt-in.
- Buffers are created bare via `[get_command_queue() commandBuffer]`
  (`new_command_buffer()`, `src/metal/device.mm`) — switching to
  `commandBufferWithDescriptor:` with `errorOptions` is a 3-line debugging change at
  that single choke point.
- Cross-refs: stale-read mechanics in `storage-and-synchronization.md`; OOB tooling in
  `gpu-capture-and-shader-validation.md`; why per-op buffers exist at all in
  `dispatch-overlap-and-perf-model.md`.
