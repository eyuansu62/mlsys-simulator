# Pinned ROCm Workspace audit tools

These are optional public source tools for a narrowly defined Workspace
acquisition and independent storage audits, outside the ordinary simulator
import path. The current acquisition driver and merger add stricter observed
domain gates; they are not semantically equivalent to the historical versions.
Native probe and Torch bridge bytes are retained, while the public HIP control
has later metadata changes. The lineage below distinguishes those sources.
The package adds no pip dependency or compiled extension. Build shared objects
locally with the already installed matching Torch/HIP/rocprofiler SDK; no `.so`
or GPU executable is distributed.

The supported acquisition is **Qwen3-1.7B BF16, TP1/PP1, MRv1, synchronous eager,
one Prefill request of 16 or 32 tokens with context 0 and one greedy output**.
The driver fixes max model/batched tokens to 512, max sequences to 1, KV to
256 MiB, seed to 0, and disables Prefix Cache, Chunked Prefill, mixed execution,
async scheduling and custom AllReduce. Use one explicitly bound visible GPU and
an experiment matching this domain. These sources are not a general adapter for
arbitrary models, layouts, devices or online policies.

The build binding is:

- vLLM revision `98dff2a81`, V1 model runner.
- Actual HIP Torch git `a8ec1f47c5b233927bf23f5cd8f68db3475af226`, native allocator,
  C++11 ABI enabled. A historical upstream Torch revision is not binary proof.
- **rocprofiler SDK 1.4.1** public headers and callback ABI. This is an SDK
  version, not a claim that the ROCm Runtime version is 1.4.1.

Record the actual wheel, Runtime, SDK headers/libraries, compiler, GPU identity,
source and built-artifact hashes for every run. A matching version string alone
does not authenticate the binary. Models must already exist at the read-only
container path `/app/models/Qwen3-1.7B`; the driver does not accept another
model path or download models.

## Files and completion boundary

| File | Role |
| --- | --- |
| `native_probe.cpp` | Bounded SDK API/normalized-memory/private-scratch observation; protocol v2 includes nine domains and RCCL binding |
| `torch_allocator_probe.cpp` | Installed Torch allocator observer with thread, epoch and real action ordering |
| `device_heap_controls.hip` | Separate device-heap and compiled-private-scratch positive controls |
| `workspace_capture_session.py` | One fresh acquisition, full Torch history, entry/exit owners and returned-output witnesses |
| `analyze_native_probe.py` | SDK setup, pairing, bounds, loss, stop/finalize and optional correlation-scope diagnostics |
| `workspace_torch_audit.py` | Requested-storage generation/lifetime and fixed-owner audit |
| `native_backing_audit.py` | Successful native backing, exact Torch segment matching, controlled LT constructor and scratch reserve audit |
| `assemble_workspace.py` | Final combination, observed-domain/source/process/output/budget checks and conditional canonical dataset |
| `workspace_domain.py` | Shared strict declared/observed software, device, parallelism, Runtime configuration and hash checks |
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
sources. The acquisition expects the two DSOs at exactly
`/app/work/native_probe.so` and `/app/work/torch_allocator_probe.so`.

Resolve `SDK_ROOT` from the installed SDK prefix containing `include/` and
`lib/`. The packaged layout used by this domain places it under
`/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel`; verify the actual
installation rather than copying a guessed private ABI. Build only inside the
existing Linux HIP environment:

```sh
SCRIPT_ROOT=/app/work/mlsys-simulator/scripts/rocm_workspace_audit
TORCH_ROOT="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent)')"
# Set SDK_ROOT to the verified installed SDK prefix before these commands.

g++ -std=c++17 -D__HIP_PLATFORM_AMD__ -shared -fPIC -O2 -Wall -Wextra -pthread \
  -I"$SDK_ROOT/include" "$SCRIPT_ROOT/native_probe.cpp" \
  -L"$SDK_ROOT/lib" -Wl,-rpath,"$SDK_ROOT/lib" -lrocprofiler-sdk \
  -o /app/work/native_probe.so

g++ -std=c++17 -shared -fPIC -O2 -Wall -Wextra -Werror \
  -D_GLIBCXX_USE_CXX11_ABI=1 -D__HIP_PLATFORM_AMD__ \
  -I"$TORCH_ROOT/include" -I"$SDK_ROOT/include" \
  "$SCRIPT_ROOT/torch_allocator_probe.cpp" -L"$TORCH_ROOT/lib" \
  -Wl,-rpath,"$TORCH_ROOT/lib" -lc10_hip -lc10 -ltorch_hip \
  -o /app/work/torch_allocator_probe.so
```

Preserve compiler commands, return codes and output, plus source/header/library
and DSO hashes. The Torch hook has no detach operation: load once and retain its
CDLL until process exit. It performs no callback-side Torch/HIP call, lock or
Python I/O. Its owner token names an observed inference-thread epoch, not a
semantic allocation owner.

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

## Acquire one session, then audit after normal exit

Select the physical card explicitly, retain the actual mapping, and provide a
complete bound experiment matching the domain above. The acquisition output path
must not exist. Use an outer owned-process supervisor and bounded logs; this
driver does not supervise itself or automatically retry failures.

```sh
SESSION_ROOT="$(mktemp -d /app/work/workspace-session.XXXXXX)"
# Set GPU_INDEX to the chosen physical visible-device index.
ROCP_TOOL_LIBRARIES=/app/work/native_probe.so \
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

The current public native source SHA256 values are:

```text
6d70afa159fc49688ce1aa8eaef40004efa2502c3a7b7af6143dcab558121fb6  native_probe.cpp
153248a4d62c2178eaaf905103b4ce23434b8eb3cef83014338f738dd9cf6f3b  torch_allocator_probe.cpp
808ae2ae21ca152e27f424cd0c13797bffb4dc1fab17d20fd7b3de6f3da29d8d  device_heap_controls.hip
```

`native_probe.cpp` and `torch_allocator_probe.cpp` retain the recorded original
bytes. Earlier live HIP controls used
`42f43cd528e0570083bfdfed7e724f9343e0ebdf8f2e3c49db7a3d36e1feb955`;
the public `808ae2...` source includes later metadata changes. Both public HIP
variants have now been compiled and exited normally on the live host, as
separate controls. Successful program exit does not replace the retained SDK
visibility/control checks described above.

The three component auditors (`analyze_native_probe.py`,
`native_backing_audit.py` and `workspace_torch_audit.py`) retain their audited
rules after formatting/type/local-name corrections. Their original and public
versions passed the same 44 CPU fixtures and produced equal JSON for 113
fixture audit calls. The optional `native_probe_controls_v2.py` adds a real
known RCCL query and explicit library binding; it is not AST-equivalent to
the historical HIP-only control.

**The current `workspace_capture_session.py` and `assemble_workspace.py` are
semantically stricter**, with shared `workspace_domain.py` checks and nine new
CPU domain regressions. Earlier AST/replay equality results applied to the
pre-gate revisions and must not be presented as equivalence of the current
merger on legacy inputs. The new merger intentionally refuses canonical output
when historical inputs lack the newly observed facts. A fresh acquisition and
independent frozen-Profile repeat exercise the current driver/merger; historical
receipts, frozen Profiles and observations are not rewritten to simulate that
new acquisition.

**Python source hashes intentionally differ from the historical hardware
sources.** Existing hardware archives and their executed-source receipts retain
the original hashes, including the original acquisition driver
`af69fa92116d5987b506710364b6d49705f9bb8414104e645c48e0abcdf1f2cc`.
CPU replay with the public merger must still reference those original executed
bytes; do not rewrite a historical receipt to name the public driver. A new
acquisition using the public driver records its actual current source SHA and
gets a new receipt. Neither formatting equivalence nor a CPU replay authorizes
changing a frozen Profile or held-out observations.

Run the 44 integer/dictionary audit fixtures plus nine domain guards from the checkout or extracted sdist:

```sh
python -m unittest discover -s tests -p 'test_rocm_*.py' -v
```

They import the packaged sources using `importlib`, require no `work/` directory,
Torch, SDK or GPU, and never establish measured hardware evidence. They cover
storage geometry/aliases, delayed frees and generations, full history/callback
agreement, native API status/ancestry, exact backing, loss/unknowns and strict
partial results. Native compilation and live control observations require the
matching Linux environment above and are separate from these CPU checks.
