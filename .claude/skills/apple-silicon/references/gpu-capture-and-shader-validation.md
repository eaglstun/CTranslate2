# GPU capture & shader validation (catching the misplaced pointer)

Sources (Apple Developer Documentation, fetched via DocC JSON, 2026-06-11):

- <https://developer.apple.com/documentation/metal/mtlcapturemanager> (+ `mtlcapturedescriptor`, `mtlcapturescope`, `mtlcapturedestination`, `startcapture(with:)`, `stopcapture()`, `shared()`)
- <https://developer.apple.com/documentation/Xcode/Capturing-a-Metal-workload-programmatically> and `Capturing-a-Metal-workload-in-Xcode`
- <https://developer.apple.com/documentation/Xcode/Validating-your-apps-Metal-shader-usage> and `Validating-your-apps-Metal-API-usage`

CLAUDE.md warns "a misplaced pointer … can cost hours to debug." These are the tools
that catch it in minutes instead.

## Programmatic GPU capture (`.gputrace`)

- **Enable first** (capture is off by default outside Xcode): `Info.plist`
  `MetalCaptureEnabled = YES`, or on macOS 14+ just the env var
  **`MTL_CAPTURE_ENABLED=1`**. "Tiny, but measurable" CPU cost.
- One manager per process: `MTLCaptureManager.shared()`. Configure an
  `MTLCaptureDescriptor`: `captureObject` = a device, command queue, or
  `MTLCaptureScope`; `destination` = `.developerTools` (default — opens in Xcode if
  attached) or `.gpuTraceDocument` + `outputURL` (give it a **`.gputrace`** extension);
  check `supportsDestination(.gpuTraceDocument)` first. Then
  `try startCapture(with: descriptor)` … `stopCapture()`.
- Scope rule (bit you trip over): the capture records **only command buffers created
  after the capture starts and committed before it stops**. A custom
  `MTLCaptureScope` (`makeCaptureScope(device:)`, `begin()`/`end()`) narrows recording
  to exactly the ops you bracket.
- In Xcode the same trace is taken with the Metal Capture button (scheme must have GPU
  Frame Capture enabled); `Debug > Debug Executable` lets you attach Xcode to a CLI
  binary after the fact. The Xcode GPU debugger then shows every dispatch with its
  pipeline, bound buffers, and **buffer contents** (Memory view) — unlabeled buffers
  show as anonymous, which is why `label` matters (`mtlbuffer-api.md`).

## Metal Shader Validation (`MTL_SHADER_VALIDATION=1`)

"Detects errors only discoverable during shader execution, such as accesses to
non-resident resources, **out-of-bounds memory accesses**, undefined behavior, and
attempts to access nil textures." Works on **any binary, no source or Xcode needed** —
env vars only. Key family (full list: `man MetalValidation`):

| Env var                                                                     | Effect                                                                                        |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `MTL_SHADER_VALIDATION=1`                                                   | enable all shader-validation tests                                                            |
| `MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`                                  | copy reports to stderr (default goes to OS log; `log stream -process <name>`)                 |
| `MTL_SHADER_VALIDATION_GLOBAL_MEMORY=1`                                     | check all `device`-memory accesses                                                            |
| `MTL_SHADER_VALIDATION_THREADGROUP_MEMORY=1`                                | check all `threadgroup`-memory accesses                                                       |
| `MTL_SHADER_VALIDATION_FAIL_MODE`                                           | `zerofill` (default: invalid reads → 0, invalid writes dropped) or `allow`                    |
| `MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1`                                    | stop the program on first error                                                               |
| `MTL_SHADER_VALIDATION_DEFAULT_STATE=none` + `…_ENABLE_PIPELINES="<label>"` | instrument only named pipelines (uses pipeline labels/UIDs; `…_DUMP_PIPELINES=1` prints UIDs) |

**Perf cost (doc's words):** "a corresponding impact on GPU performance, and shaders
might take longer to compile at runtime. This layer adds instrumentation code to all
your GPU functions." Never benchmark with it on. Also: incompatible with Metal binary
archives.

## API Validation (`MTL_DEBUG_LAYER=1`)

The CPU-side sibling: catches incorrect Metal **API** usage (resource creation, encoding
errors, unretained-resource lifetime bugs via
`MTL_DEBUG_LAYER_VALIDATE_UNRETAINED_RESOURCES`). `MTL_DEBUG_LAYER_ERROR_MODE` =
`assert` (default) / `ignore` / `nslog`. "Small, but measurable" CPU cost. In Xcode both
layers live under Edit Scheme → Run → Diagnostics.

## Recipe: one validated `ctranslate2_test` run

```bash
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
MTL_DEBUG_LAYER=1 \
./tests/ctranslate2_test ../tests/data --gtest_filter='MetalTest.*'
```

For a trace of one op: `MTL_CAPTURE_ENABLED=1`, add a temporary
`startCapture(with:)`/`stopCapture()` bracket (destination `.gpuTraceDocument`,
`outputURL` ending `.gputrace`) around the suspect op, run, open the trace in Xcode.
Honest constraint: there is no pure-env "capture everything from the CLI" — you either
bracket programmatically or attach Xcode (`Debug > Debug Executable`) and click capture.

---

### Relevance to the CT2 Metal backend

- **What it catches**: a wrong `setBuffer` index, a stale `buffer_and_offset` mapping,
  an out-of-range `dispatchThreadgroups` write, a GEMV reading past `k` — exactly the
  misplaced-pointer class. With `zerofill` the symptom is silent wrong numbers; with
  validation on you get the kernel name and access.
- **What it would NOT have caught — the Gemma2 NaN, honestly**: the
  `<pad>`-collapse bug was `tanh()` overflowing to NaN on huge fp32 activations — a
  _numeric_ bug, fixed by the clamp in `ct2_tanh_safe`
  (`src/metal/kernels/kernels_msl.h`, `tanh(clamp(x, -15.0f, 15.0f))`). Shader
  validation's NaN check (`MTL_SHADER_VALIDATION_NAN_INF`) applies to **render-pipeline
  vertex interpolants**, not compute kernels — so the hunt's actual tools (per-layer NaN
  tripwires + CPU-ref bisection, see the memory log) remain the right ones for numeric
  bugs. Validation is for memory-safety bugs; don't reach for it expecting NaN forensics.
- The backend's pipelines are keyed by kernel name (`device.mm` pipeline cache) but the
  PSOs carry **no labels**, and buffers are unlabeled (`allocator.mm`) — labeling both
  is what makes `…_ENABLE_PIPELINES` selective instrumentation and capture buffer
  inspection usable. Free win, same as in `mtlbuffer-api.md`.
- `MTLCommandBuffer.status`/`.error` (DocC: `error` stays nil unless the GPU "can't
  successfully run the command buffer") are currently never checked after
  `waitUntilCompleted` in `flush()` (`device.mm`) — a one-line `if (cb.error)` log there
  would surface GPU faults that today vanish into zero-filled output.
