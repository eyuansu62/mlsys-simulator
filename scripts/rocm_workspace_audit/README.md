# Pinned ROCm Workspace audit tools

These are optional public source tools for pinned Workspace acquisition and
independent storage audits, outside the ordinary simulator import path.
Version 1.13 adds Worker-level prefill/decode collection, rank placement and
native PP lease auditing. The legacy single-card prefill driver remains a
separate interface. Source and native observer changes require new acquisition
receipts; current sources are not byte-equivalent to the historical tools.
The package adds no pip dependency or compiled extension. Build shared objects
locally with the already installed matching Torch/HIP/rocprofiler SDK; no `.so`
or GPU executable is distributed.

The Worker raw collector supports **Qwen3-1.7B BF16, single (TP1×PP1),
TP2×PP1 or TP1×PP2, synchronous eager MRv1**, with one active sequence and
four greedy outputs for each 16/32-token prompt. It fixes model/batch-token
limits to 512, KV to 256MiB, block size to 16 and seed to 0; cache, chunking,
mixed execution, swap/offload, async execution, graphs and custom AllReduce are
disabled. The legacy driver instead captures single-card prefill with one
output. Neither interface is a general adapter for arbitrary models, layouts,
devices or online policies. Raw success does not establish complete attribution
for a supported mode.

The build binding is:

- vLLM revision `98dff2a81`, V1 model runner.
- Actual HIP Torch git `a8ec1f47c5b233927bf23f5cd8f68db3475af226`, native allocator,
  C++11 ABI enabled. A historical upstream Torch revision is not binary proof.
- **rocprofiler SDK 1.4.1** public headers and callback ABI. This is an SDK
  version, not a claim that the ROCm Runtime version is 1.4.1.

Record the actual wheel, Runtime, SDK headers/libraries, compiler, GPU identity,
source and built-artifact hashes for every run. A matching version string alone
does not authenticate the binary. Models must already exist locally; the Worker
launcher defaults to the read-only container path `/app/models/Qwen3-1.7B` and
can receive another existing directory through `--model`. It downloads nothing.

## Files and completion boundary

| File | Role |
| --- | --- |
| `native_probe.cpp` | Bounded nine-domain SDK API/memory/scratch observation, typed RCCL calls, PID output and explicit stop/finalization evidence |
| `torch_allocator_probe.cpp` | Installed Torch allocator observer with thread, epoch, action order and read-only committed-event cursor |
| `device_heap_controls.hip` | Separate device-heap and compiled-private-scratch positive controls |
| `workspace_stage_session.py` | Bounded fresh single/TP2/PP2 supervisor; raw warmup and formal iterations, without fitting or acceptance |
| `worker_workspace_capture.py` | Real GPUWorker execute/sample observation, actual communication prewarm and PP handoff/retention/teardown |
| `workspace_stage_domain.py` | Actual prepared-batch, source/Runtime/rank/layout and supported-shape checks |
| `workspace_transport_audit.py` | Full transport-generation replay, retained PP ownership, overlap and release completion |
| `assemble_stage_workspace.py` | Independent Worker/Torch/native/PP combination; canonical data only after every gate closes |
| `workspace_capture_session.py` | Legacy single-card prefill acquisition |
| `analyze_native_probe.py` | SDK setup, pairing, bounds, loss, stop/finalize and optional correlation-scope diagnostics |
| `workspace_torch_audit.py` | Requested-storage generation/lifetime and exact TP-shard/PP-layer owner checks |
| `native_backing_audit.py` | Successful backing, exact Torch matching, controlled LT/communicator provenance and separate fixed budgets |
| `assemble_workspace.py` | Legacy prefill merger, with strict observed-domain/source/process/output/budget checks |
| `workspace_domain.py` | Legacy declared/observed software, device and Runtime domain checks |
| `native_probe_controls_v2.py` | Optional executable SDK/RCCL positive control; importing it performs real HIP work |
| `_native_probe_fixture.py` | Integer/dictionary CPU fixture builders used by repository tests; no real trace |

Every individual probe, acquisition and component audit retains
`complete_workspace=false` or an equivalent partial/candidate status, and
`hardware_validated=false`. Torch `CLOSED_CANDIDATE`, native
`NATIVE_BACKING_CHECKS_PASSED`, or a zero-event SDK epoch cannot establish
complete attribution separately. Only the final merger, with **all** supplied
evidence and required external build/control bindings satisfied, can emit
`ATTRIBUTION_COMPLETE_SUPPORTED_DOMAIN` and a validated Workspace dataset.
Any missing ownership, lifetime, rank/device/source, return witness or fixed
budget proof keeps the result partial. No status grants a general hardware PASS,
serving-accuracy result or physical OOM threshold.

## Container layout and local builds

Mount a writable task directory at `/app/work`, keep the downloaded model mount
at `/app/models`, and make the checkout or sdist sources available below
`/app/work/mlsys-simulator`. The installed simulator wheel must match these
sources. The Worker launcher defaults to `/app/work/native_probe.so` and
`/app/work/torch_allocator_probe.so`; its probe-path options can select separately
built actual DSOs. The legacy driver expects those exact default paths.

Resolve `SDK_ROOT` from the installed SDK prefix containing `include/` and
`lib/`. The packaged layout used by this domain places it under
`/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel`; verify the actual
installation rather than copying a guessed private ABI. Build only inside the
existing Linux HIP environment:

```sh
SCRIPT_ROOT=/app/work/mlsys-simulator/scripts/rocm_workspace_audit
TORCH_ROOT="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent)')"
# Set SDK_ROOT to the verified installed SDK prefix before these commands.

g++ -std=c++17 -D__HIP_PLATFORM_AMD__ -shared -fPIC -O2 -Wall -Wextra -Werror -pthread \
  -isystem "$SDK_ROOT/include" "$SCRIPT_ROOT/native_probe.cpp" \
  -L"$SDK_ROOT/lib" -Wl,-rpath,"$SDK_ROOT/lib" -lrocprofiler-sdk -ldl \
  -o /app/work/native_probe.so

g++ -std=c++17 -shared -fPIC -O2 -Wall -Wextra -Werror \
  -D_GLIBCXX_USE_CXX11_ABI=1 -D__HIP_PLATFORM_AMD__ \
  -isystem "$TORCH_ROOT/include" -isystem "$SDK_ROOT/include" \
  "$SCRIPT_ROOT/torch_allocator_probe.cpp" -L"$TORCH_ROOT/lib" \
  -Wl,-rpath,"$TORCH_ROOT/lib" -lc10_hip -lc10 -ltorch_hip \
  -o /app/work/torch_allocator_probe.so
```

Preserve compiler commands, return codes and output, plus source/header/library
and DSO hashes. Vendor headers are system includes so their warnings do not
disable `-Werror` on the project's own code. The native thread-source observer
uses standard Linux `libdl` through `-ldl`; this adds no pip dependency.
The Torch hook has no detach operation: load once and retain its
CDLL until process exit. It performs no callback-side Torch/HIP call, lock or
Python I/O. Its owner token names an observed inference-thread epoch, not a
semantic allocation owner.

`mlsys_torch_probe_cursor()` reads a stable committed event prefix without
changing epochs, allocating or synchronizing HIP. `UINT64_MAX` is uncertainty,
not a valid sequence. Transport observations bind real Tensor storage and
generation to this cursor; they never infer an owner from a reused address alone.

For the separate positive controls, compile two programs so one experiment does
not register the other's control kernel. Choose the actual target architecture
(`gfx1201` for RX 9070 XT, `gfx1100` for RX 7900 XTX):

```sh
hipcc -std=c++17 -O2 -gline-tables-only -Wall -Wextra -save-temps \
  --offload-arch=gfx1201 -DMLSYS_CONTROL_KIND=1 \
  "$SCRIPT_ROOT/device_heap_controls.hip" -ldl -o /app/work/device_heap_control
hipcc -std=c++17 -O2 -gline-tables-only -Wall -Wextra -save-temps \
  --offload-arch=gfx1201 -DMLSYS_CONTROL_KIND=2 \
  "$SCRIPT_ROOT/device_heap_controls.hip" -ldl -o /app/work/private_scratch_control
```

Each program requires `--probe /app/work/native_probe.so --out UNIQUE.json
--device 0` and the same startup SDK environment as acquisition below. Keep its
normal exit, checksum/pointer witnesses and final SDK report. A private-scratch
resource-gate skip, absent callback binding, crash or timeout is inconclusive;
semantic kernel success is not SDK visibility. Device heap reuse and retained
queue-scratch capacity must not become a zero-allocation claim.

The optional Python control exercises HIP malloc/free and a known
`ncclGetVersion` query in its own SDK epoch. It selects the unique actual
`librccl.so` mapping, or an explicit absolute `MLSYS_RCCL_LIBRARY` path to the
installed library. It rejects conflicting/ambiguous mappings and verifies the
selected library is actually mapped after loading; it does not search or try
guessed binaries. If Torch has not mapped RCCL, provide that exact path:

```sh
# Set RCCL_LIBRARY_PATH to the verified installed librccl.so file.
CONTROL_ROOT="$(mktemp -d /app/work/native-controls.XXXXXX)"
LD_PRELOAD=/app/work/native_probe.so \
ROCP_TOOL_LIBRARIES=/app/work/native_probe.so \
MLSYS_NATIVE_PROBE_OUT="$CONTROL_ROOT/native-final.json" \
MLSYS_NATIVE_PROBE_RETAIN_ALL=1 MLSYS_NATIVE_PROBE_MAX_EVENTS=262144 \
MLSYS_CONTROL_OUTPUT_DIR="$CONTROL_ROOT" \
MLSYS_RCCL_LIBRARY="$RCCL_LIBRARY_PATH" HIP_VISIBLE_DEVICES="$GPU_INDEX" \
timeout --signal=TERM --kill-after=5s 60s \
python "$SCRIPT_ROOT/native_probe_controls_v2.py"
```

`MLSYS_CONTROL_OUTPUT_DIR` defaults to `/app/work` for compatibility. Use a new
unique directory as above; existing result/checkpoint files are rejected before
GPU initialization, and the result is created exclusively. The SDK probe is
still loaded from `/app/work/native_probe.so`. The result records the selected library path/SHA,
original query return code and numeric version code. API success alone does not
prove callback delivery: inspect the **final** SDK report with the analyzer and
require paired typed-success RCCL callbacks and all nine configured domains.
The intentional HIP-allocation epoch remains a conservative partial diagnostic;
do not discard it to manufacture an overall zero-mutator verdict. Query failure
is recorded as a failed control and exits unsuccessfully. This program always
retains `complete_workspace=false` and `hardware_validated=false`.

## Acquire Worker-level Prefill and Decode

Use `workspace_stage_session.py` for new single/TP2/PP2 acquisitions. It runs
one owned child process group, selects visibility before GPU imports, installs
the real Worker subclass and enforces a timeout/log bound. It does not fit,
retry, or grant a hardware verdict. Every output directory must be new.

```sh
python "$SCRIPT_ROOT/workspace_stage_session.py" \
  --mode single --experiment /app/work/single.actual.json \
  --cohort train --repeats 3 --native-max-events 2097152 \
  --out /app/work/workspace-single-train
```

Use `tp2` or `pp2` with the corresponding complete actual experiment. Default
visibility is physical 1 for `single` and `0,1` for the two-card modes; an explicit
`--visible-devices` is recorded and must agree with actual worker/device facts.
PP2 needs the native transport and a bound covering the actual 512-token persistent
input-copy targets. Effective PP endpoint weight identity must match the loaded
model; no cross-device tied-storage alias is invented.

Each 16/32-token prompt naturally generates four outputs: prefill at context 0,
then decode query 1 at contexts 16/17/18 or 32/33/34. Three repeats give 24 formal
iterations per rank, including 18 decode iterations. All raw warmup and formal
iterations retain explicit cohort and actual prepared-batch metadata. Epoch,
PID, stage and TP rank bind them; opaque request names do not identify cohorts.
The native clocks are `CLOCK_MONOTONIC_RAW`; prepared-stage/parent clocks are
`CLOCK_MONOTONIC`. Do not subtract or directly filter across those clock domains.

The launcher sets `LD_PRELOAD`, `ROCP_TOOL_LIBRARIES` and per-process
`MLSYS_NATIVE_PROBE_OUT=.../native-%p.json`. Only literal `%p` is expanded by the
native probe; other percent formats and path overflow are rejected. Parent and
worker SDK files remain separate and are joined by actual worker PIDs, never by
filename order. The SDK event budget defaults to 1,048,576 and can be explicitly
set up to 4,194,304; the command above shows a 2,097,152-event acquisition budget.
Record the chosen value before training and repeat. A larger bound does not
repair a previously saturated trace or authorize dropping observations.

The original Worker execute boundary covers receive allocation, model work and
send handoff. Final-stage sampling supplies its actual CPU-only output witness.
PP sender output can remain live after Worker return through
`_pending_isends`; full storage generations, pending frees and natural reap must
remain visible. Only integer descriptors are retained by the observer. It must
not consume lazy receive tensors, wait early, reap early, or keep extra Tensor
references to manufacture lifetime closure. A final explicit drain/empty-cache
step is a separate teardown epoch and is excluded from formal peaks.

A dedicated fresh TP1×PP2 retention stress ran two 511-token prompts with one
output each. Full transport-generation replay, including pending releases,
observed 8,372,224 sender bytes: the previous and current full output pairs
overlapped. At a 512-token bound and one model wave, the old 4,194,304-byte sender
reservation was too small; `max(inflight, 2)` reserves 8,388,608 bytes and covered
the observation with 16,384 bytes remaining. The receiver's dynamic transport
peak was 4,186,112 bytes, separate from 4,194,304 bytes of persistent input-copy
targets. These are fixed-cohort transport observations, not Workspace training
rows or proof of complete native attribution. They do not establish a universal
bound for other workloads or async execution.

Preserve the real outer execution return code, timing, source/binary hashes and
receipt. After normal exit:

```sh
python "$SCRIPT_ROOT/assemble_stage_workspace.py" \
  --session /app/work/workspace-single-train \
  --receipt /app/work/workspace-single-train-receipt.json \
  --out /app/work/workspace-single-train-assembled
```

Multi-card modes additionally require actual per-rank communicator bindings
through `--communication-bindings`. Initialization/source/library/handle/rank,
native allocation, and communication-thread evidence are independently checked;
an ownership label or a component's success string is insufficient. Fixed
communication backing and PP buffers have separate budgets. Unknown allocation,
unsupported mutation, missing rank or unclosed lifetime keeps attribution
partial. A normally returned assembler can have written a partial report:
inspect `attribution_complete` and issues. Only complete output gets canonical
`dataset.json` for `calibrate --workspace`.

Freeze that Profile before starting a fresh `--cohort repeat` acquisition with
the same supported identity. Audit the repeat independently and use
`validate --workspace-repeats --training-dataset fit/dataset.json --profile ...
--dataset ...`; see the [full workflow](../../docs/workspace-and-capacity.md#validate-a-frozen-workspace-profile-independently).
Exact shape/rank coverage and byte underprediction are evaluated without
refitting or hardware PASS. Memory snapshots and native callbacks instrument
execution: the exported stage durations are diagnostic, not new latency
calibration samples.

## Native stop, finalization and communication causality

The current native probe distinguishes actual HIP context operations from IPC;
the letters in `hipCtx...` are not an IPC ownership proof. RCCL operations retain
their real names, arguments, typed return statuses, communicator handles and
correlation chains. A typed successful collective does not by itself register
every allocation on its process or thread as communication storage.

The optional thread-source observer records actual `pthread_create` calls made
under observed communicator initialization, the parent init correlation, original
start routine/module and the actual child TID. It calls the original routine
with the original argument and preserves its return. Audits require matching
loaded-library/source and allocation lineage; thread names or timing proximity
cannot fill missing causality. Recorded thread origin is a separate proof from
successful native allocation and fixed-backing lifetime. Unsupported proxy
activity still prevents complete attribution.

`mlsys_probe_stop()` first stops the SDK context, then drains both callback
execution and already-entered API calls under a bounded deadline. Open-call
underflow, unfinished calls, new API entry after stop, late callbacks or a
nonzero stop/drain status remain explicit failures. A checkpoint is a prefix,
not finalization proof.

The Worker close path calls `mlsys_probe_finalize()` after successful stop and
all potentially HIP-backed metadata reads. This wrapper invokes the installed
SDK's `rocprofiler_client_finalize_t`; it does not call `tool_fini` itself. The
SDK callback is void, so no SDK return code is invented. Wrapper success needs
the actual tool-finalize callback exactly once and a final report write. The
finalization fields, worker RPC result and normal process exit must agree.
Legacy programs and separate controls must likewise retain their actual final
SDK file after normal exit; the checkpoint alone is insufficient.

## Legacy single-card prefill acquisition

Select the physical card explicitly, retain the actual mapping, and provide a
complete bound experiment matching the domain above. The acquisition output path
must not exist. Use an outer owned-process supervisor and bounded logs; this
driver does not supervise itself or automatically retry failures.

```sh
SESSION_ROOT="$(mktemp -d /app/work/workspace-session.XXXXXX)"
# Set GPU_INDEX to the chosen physical visible-device index.
ROCP_TOOL_LIBRARIES=/app/work/native_probe.so \
LD_PRELOAD=/app/work/native_probe.so \
MLSYS_NATIVE_PROBE_OUT="$SESSION_ROOT/native-final.json" \
MLSYS_NATIVE_PROBE_RETAIN_ALL=1 MLSYS_NATIVE_PROBE_MAX_EVENTS=1048576 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
timeout --signal=TERM --kill-after=5s 300s \
python "$SCRIPT_ROOT/workspace_capture_session.py" \
  --experiment /app/work/bound-workspace-experiment.json \
  --cohort train --iterations 3 --out "$SESSION_ROOT/capture" \
  > "$SESSION_ROOT/worker.log" 2>&1
WORKER_RC=$?

python - "$WORKER_RC" "$SCRIPT_ROOT/workspace_capture_session.py" "$SESSION_ROOT/receipt.json" <<'PY'
import hashlib, json, pathlib, sys
driver = pathlib.Path(sys.argv[2]).resolve()
receipt = {"returncode": int(sys.argv[1]),
           "driver_source_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
           "driver_source_reference": str(driver)}
with pathlib.Path(sys.argv[3]).open("x") as stream:
    json.dump(receipt, stream, indent=2)
PY
```

Keep the actual process return code before running anything else. A receipt JSON
for assembly contains `returncode`, `driver_source_sha256` and
`driver_source_reference`, the latter pointing to the actual executed driver
file. Compute its digest from that file; do not invent a successful receipt or
overwrite one from a failed execution. Also retain the external process-start,
exit and build/control evidence. The merger checks source bytes and consistency;
it cannot authenticate a manually fabricated receipt or prove process freshness
from a cohort label alone.

The current driver rejects unsupported declared TP/PP/replica sizes, token and
sequence limits, KV/block/graph/swap budgets and cache/chunk/mixed flags before
loading the model. It does not silently replace a mismatching KV budget. It
observes the actual vLLM version, Torch build, visible GPU name/capacity and
allocator backend, then the real MRv1 parallel configuration and synchronous
policy. Existing ROCm and loaded-RCCL probes supply actual software metadata.
The 512-token/one-sequence eager configuration is checked against the real
Runtime snapshot. Its TP/PP/async fields come from the Runner, not the declared
Experiment. A nonempty input Runtime-config SHA must already match; an empty
one may bind the observed digest.

New sessions record `observed_runtime_domain` and `observed_runtime_version`.
The merger checks those facts, their supported domain, exact Runtime-config SHA
and per-epoch metadata against the session index. Legacy sessions missing new
observations remain partial under this stricter public merger; do not fabricate
or backfill facts into historical archives. Their original evidence and frozen
Profiles remain preserved under the original acquisition/audit provenance.

The session exports six formal rows (three repetitions each at 16/32), compact
Torch snapshots/full history, explicit owner registries, allocator callbacks,
returned-output witnesses and actual Runtime/model binding. Warmup occurs
outside formal epochs. The native probe must be loaded before the first ROCm
call and retain startup callbacks. Check the final file written by the SDK's
later tool-finalize callback after normal process exit; the checkpoint alone is
insufficient. Explicit stop/drain must succeed before SDK teardown. Nonzero stop
status, late callbacks, event loss, unknown operations or a saturated buffer
cannot be repaired by assuming zero growth.

After normal exit:

```sh
python "$SCRIPT_ROOT/analyze_native_probe.py" "$SESSION_ROOT/native-final.json" \
  > "$SESSION_ROOT/native-diagnostic.json"
python "$SCRIPT_ROOT/assemble_workspace.py" \
  --session "$SESSION_ROOT/capture" --native "$SESSION_ROOT/native-final.json" \
  --receipt "$SESSION_ROOT/receipt.json" --out "$SESSION_ROOT/assembled"
```

The first command is a conservative diagnostic, not a substitute for native
backing replay. The merger runs both component audits, preserves their reports
and issues, checks CPU-only return witnesses, and admits fixed Runtime/KV bytes
against their separate budgets. Dynamic Workspace uses owned requested storage
capacity with real generation and release-completed ordering. Allocator reserved
arenas, rounding and peak-minus-baseline are not substituted for that capacity.
Unknown GPU backing stays unknown; matched physical aliases are deduplicated.

On complete output, use the existing public Workspace calibration command to fit
and freeze the Profile. Only then start a new Runtime/session with `--cohort
repeat --iterations 3` and audit it independently. Cohort labels do not make a
process independent, and the repeat must not tune the frozen Profile. These
scripts intentionally do not automate calibration, selection or acceptance.

## Source lineage and CPU verification

Version 1.13 changes the native probe, Torch bridge and rank/transport audits.
Their current bytes must not be described as identical to the earlier 1.12
artifacts. New builds and acquisitions record their actual source, header,
library and DSO hashes in receipts/manifests. Keep the exact executed sources
beside the raw evidence and preserve the commit that produced them. Hash values
from a historical report are not replacements for those actual current files.

The legacy driver/merger and the Worker driver/merger are distinct interfaces.
Legacy sessions may lack newer observed facts and remain partial under newer
gates. Never backfill a runtime version, thread origin, source hash or successful
receipt into old observations to make them pass. Earlier formatting/fixture
replay equivalence applied only to those historical revisions; it does not
establish semantic equality for the current expanded native/placement/lifetime
rules. Independent repeat data and frozen Profile bytes remain unchanged.

Run the pure-data audit and domain fixtures from the checkout or extracted sdist:

```sh
python -m unittest discover -s tests -p 'test_rocm_*.py' -v
```

The fixtures test unsupported/missing evidence as well as complete candidates.
They import the distributed sources, require no private work directory, SDK or
GPU, and do not establish measured hardware results. Reproduction also needs
actual native compilation, separate positive controls, complete source/rank
binding, fresh process receipts and independently frozen validation. See the
[Workspace workflow](../../docs/workspace-and-capacity.md) and
[live evidence record](../../docs/workspace-live-validation.md) for their
separate scopes. Neither raw exit 0, a positive control, nor a component status
changes the overall hardware-acceptance flag.

## PP2 HSA signal and KFD backing evidence

Version 1.14 adds an optional PP2 source-evidence path. A surviving signal is
not relabeled as destroyed. Its public ABI body must have affirmative CPU
preferred-owner/range evidence; nonzero hardware event IDs additionally need
the original KFD event, GTT allocation and mapping generations. See the
[evidence contract](../../docs/pp2-synchronization-evidence.md).

On the supported Linux x86-64 host, compile the Native and passive KFD sources
into the same preload DSO. Use the actual installed SDK paths and preserve the
exact input sources, compiler dependency file and build receipt:

```sh
g++ -std=c++17 -D__HIP_PLATFORM_AMD__ -shared -fPIC -O2 \
  -Wall -Wextra -Werror -pthread -MD -MF "$BUILD_ROOT/combined.d" \
  -isystem "$ROCM_SDK_PREFIX/include" \
  "$SCRIPT_ROOT/native_probe.cpp" "$SCRIPT_ROOT/kfd_event_probe.cpp" \
  -L"$ROCM_SDK_PREFIX/lib" -Wl,-rpath,"$ROCM_SDK_PREFIX/lib" \
  -lrocprofiler-sdk -ldl -o "$COMBINED_SO"
g++ -std=c++17 -shared -fPIC -O2 -Wall -Wextra -Werror \
  -isystem "$ROCM_SDK_PREFIX/include" \
  "$SCRIPT_ROOT/hsa_signal_backing_probe.cpp" -ldl -o "$PROVIDER_SO"
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -isystem "$ROCM_SDK_PREFIX/include" \
  "$SCRIPT_ROOT/hsa_signal_namespace_control.cpp" \
  -L"$ROCR_LIBRARY_DIR" -Wl,-rpath,"$ROCR_LIBRARY_DIR" \
  -l:libhsa-runtime64.so.1 -ldl -o "$CONTROL_BIN"
mkdir "$CONTROL_ROOT"
LD_PRELOAD="$COMBINED_SO" ROCP_TOOL_LIBRARIES="$COMBINED_SO" \
  MLSYS_NATIVE_PROBE_OUT="$CONTROL_ROOT/native-%p.json" \
  MLSYS_NATIVE_PROBE_RETAIN_ALL=1 \
  "$CONTROL_BIN" "$CONTROL_ROOT" "$PROVIDER_SO" "$LOADED_ROCR_SO" 32
```

The control holds a bounded set of its own ordinary signals, samples each twice
and destroys only those signals. Preserve `control.jsonl`, `kfd.json` and the
actual final Native file. A zero process return is not a substitute for replay.

The caller's cold-control manifest has kind
`cold_hsa_kfd_namespace_control_manifest` and schema version 1. It references the
original `native_report`, `kfd_report`, `control_jsonl` and `build_receipt` with
`path`, `sha256` and `size_bytes`; its actual `create_cid`, `event_query_cid` and
`kfd_create_event_call_id` must identify a causally nested creation and getter.
`target_epochs` contains the original control epoch. The `source_binding`
contains the original compiled ABI, kernel/module identity, device/render/PCI
facts and seven actual artifacts: `probe_source`, `native_probe_source`,
`probe_library`, `hsa_runtime`, `libc`, `kfd_header` and `drm_header`. Record the
build receipt there too. Derive locators from raw events; never copy example IDs
or fill in missing observations. Preserve raw sysfs bytes in environment
snapshots, measuring their actual read length instead of virtual `stat` size.

Add these arguments to a new PP2 Worker acquisition:

```sh
--host-signal-provider "$PROVIDER_SO" \
--host-signal-rocr "$LOADED_ROCR_SO" \
--host-signal-abi-header "$ROCM_SDK_PREFIX/include/hsa/amd_hsa_signal.h" \
--host-signal-namespace-control "$NAMESPACE_MANIFEST"
```

The first three paths must be provided together. This integration supports PP2
with the default `worker_shutdown` finalization. It retains original formal
execution and GPU fences; queries run at the existing synchronized CPU boundary
before natural shutdown. The final KFD prefix follows actual successful SDK
stop. Cold-control/source/environment facts are shared evidence; each Worker's
actual Native, Torch, PP and KFD captures still require independent acquisition
identities and full replay. Missing or unknown proof remains partial.
