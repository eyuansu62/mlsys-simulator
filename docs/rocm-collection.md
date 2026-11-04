# AMD/ROCm collection and hardware-evidence acceptance

Version 1.5 introduced collection adapters around real Runtime calls and explicit
acceptance of supplied evidence. `simulate` remains a CPU model. Collection
does not install a driver, ROCm, PyTorch, RCCL tests or a serving Runtime.

## Start with a probe

```bash
mlsys-sim collect --kind probe --out runs/rocm-probe
```

Use a new output directory. A probe can run on a machine without AMD hardware;
read its availability and missing-evidence diagnostics. Completing a probe is
not a successful GPU measurement. Live stage collection requires both an
available device and a nonempty PyTorch HIP version. A CUDA-compatible API name
or `torch.cuda.is_available()` alone does not establish that the device is AMD.
There is no CPU timing fallback.

The collector keeps hardware/software facts separate from declared modeling
parameters. Device names and capacities, ROCm/HIP and driver evidence, Runtime
identity, model revision, explicit dtypes and selected placement must support the
resulting Profile identity. Unknown fields remain unknown or fail the relevant
collection boundary; the collector does not invent plausible version strings.

PyTorch's HIP backend intentionally uses `torch.cuda` interfaces. The namespace
does not turn the capture into NVIDIA CUDA evidence. See the official
[HIP semantics](https://docs.pytorch.org/docs/2.15/notes/hip.html).

## Supported collection paths

`collect --kind` selects one evidence boundary:

| Kind | Evidence and restriction |
| --- | --- |
| `probe` | Availability, device and software-version diagnostics |
| `steps` | Synchronized whole-stage Runtime calls with their exact query/context/member-role shape |
| `rccl` | Standalone RCCL AllReduce service and correctness evidence |
| `runtime-trace` | Explicit server request-arrival, Token-ready and terminal events |
| `vllm` | Token-ID-aware OpenAI-compatible completion streams, preserving client-delivery timing |
| `reference` | Read-only audit/import of supported user-supplied measurement packages |

Each path uses the experiment and plan inputs it requires. The CLI records raw
inputs and provenance alongside normalized artifacts; existing run directories
and earlier Profiles are not overwritten. Fixture imports require
`--allow-test-fixture`, and derived evidence retains its fixture identity.

`steps`, `rccl` and `vllm` use `--experiment` and `--plan`; steps additionally
requires `--adapter module:factory`. `runtime-trace` uses `--experiment` and
`--raw`, while `reference` uses `--reference-dir`. Use `--inventory` to provide
an already captured inventory where supported, and `--rccl-library` or
`--rocm-root` to select the local evidence sources. Flags for a different kind
are rejected rather than silently ignored.

Collection directories contain `manifest.json`, `result.json` and logs, plus
the artifacts returned by that adapter: for example `inventory.json`,
`experiment.json`, `dataset.json`, `raw_records.json`, `observed.json`,
`audit.json`, `client_metrics.json` or request-service/client-request JSONL.
The manifest binds emitted artifacts to hashes. A failed measurement retains
available partial evidence and diagnostics.

## Stage timing and allocator evidence

The stage adapter executes the configured Runtime callback. The collector owns
the monotonic clock and synchronizes participating HIP devices before and after
the measured call. Warmup and preparation remain outside the sampling interval.
The Runtime must honor the declared complete stage boundary, including its
local collectives and output/sampling work. The experiment fixes the model,
revision, dtypes and TP/PP stage layout; the adapter's Runtime metadata declares
the software and service boundary. The callable must execute that experiment's
actual stage rather than a shape-compatible substitute.

On the configured AMD host, a stage plan can be passed with an explicit local
factory:

```bash
mlsys-sim collect --kind steps --experiment your-amd-experiment.json \
  --plan examples/v1_5/steps.plan.json --adapter your_runtime:build_adapter \
  --out runs/stage-collection
```

The plan is a shape template, not measured data. Set its source reference for
the actual run and replace its shapes with the Runtime's supported domain.
The factory is called as `factory(experiment)` and creates an adapter with
`metadata()`, `prepare(sample)` and
`run_step(sample)`. Metadata declares `runtime_name`, `runtime_version`,
`timing_boundary="step_includes_output_and_sampling_v1"`,
`includes_local_collectives=true` and `workspace_attribution` as `"none"` or
`"runtime"`. The adapter is responsible for group-level completion inside its
callable; participant-device synchronization brackets that call. An optional
`workspace_peak_bytes(sample)` supplies attributed bounds, and `close()` releases
Runtime resources when provided. A generic matrix multiplication is not a
replacement for the declared model stage.

Live TP collection also requires `rccl_runtime_version` and
`rccl_library_sha256` from the Runtime's actually loaded RCCL library; they must
match the selected-library inventory. `rccl_binding_reference` can retain the
Runtime's verification method. Selecting a library to query does not establish
that the Runtime loaded it. Missing or mismatching proof stops live TP
collection; fixtures can omit it and remain fixtures.

Live AllReduce capture selects `half`, `bfloat16`, `float` or `double` from the
declared activation dtype and verifies the datatype printed in each log row.
Measured imports must match that dtype as well. Fixture `auto` inputs may use
the explicitly labeled default float parser exercise; this does not establish
a hardware datatype binding.

A GPU Event measures a particular device/stream interval; it does not by itself
establish a multi-rank, host-visible whole-stage service time. Deep profiling
for kernel attribution belongs in a separate run from low-interference timing.

Raw allocator evidence retains baseline/current/peak **allocated** and
**reserved** bytes separately. Neither peak allocation nor an unqualified
peak-minus-baseline difference identifies Workspace: weights, KV growth,
outputs, lazy initialization and allocator caching can contribute. Without a
Runtime attribution contract, canonical `workspace_peak_bytes` stays null.
The simulator then retains its labeled analytical Workspace assumption. See
[PyTorch peak-memory semantics](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.memory.reset_peak_memory_stats.html).

## RCCL timing and version identity

RCCL collection is for AllReduce. It retains the selected benchmark executable,
arguments, raw output, exit status, rank count, message bytes, timing unit,
in-place/out-of-place mode and correctness checks. The model needs a
synchronous-group service boundary, so the collector selects the maximum-rank
report and a declared Ring configuration rather than silently treating a
mean-rank result as the slowest rank. Table units must be explicit; bandwidth
columns are not a substitute for measured latency. Incorrect or ambiguous rows
are rejected.

The plan template requires two visible devices and a matching one-replica
experiment; edit the actual benchmark path and run reference before use:

```bash
mlsys-sim collect --kind rccl --experiment your-amd-tp2-experiment.json \
  --plan examples/v1_5/rccl.plan.json --rccl-library /path/to/librccl.so \
  --out runs/rccl-collection
```

The supported runner is a single process/thread controlling all selected
visible devices. MPI launches, split groups and implicit device remapping are
rejected by this adapter. A preserved text import declares `raw_path`,
`source`, `algorithm="Ring"` and `rank_aggregation="max"`, and omits live-run
controls. These launch facts remain declared import metadata; a log alone does
not prove flags it never recorded.

`torch.cuda.nccl.version()` exposes a collective build version. It is not proof
of the dynamically loaded RCCL version. When querying an explicit library with
`ncclGetVersion`, retain the selected library path and hash with the returned
version. A missing version cannot be filled with the version from a container
tag. See [RCCL API](https://rocmdocs.amd.com/projects/rccl/en/develop/userguide/source/api/comms.html)
and the [PyTorch collective implementation](https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/nccl.cpp).

The current `rccl-tests` implementation prints table time in microseconds. The
adapter checks the supported format before converting to milliseconds; it
does not assume every archived tool version uses that unit. Its
`sendrecv_perf` runs simultaneous ring send/receive and does not support the
same in-place interpretation as AllReduce. It is rejected as a calibration
source for the simulator's isolated unidirectional PP P2P cost. See
[RCCL test source](https://github.com/ROCm/rccl-tests/blob/develop/src/common.cu)
and [SendRecv source](https://github.com/ROCm/rccl-tests/blob/develop/src/sendrecv.cu).

Communication fitting produces an explicit Infra patch. Whole-stage Profiles
already include local collectives, so the patch is not added again to those
times. The supplied reference packages contain no multi-GPU RCCL validation.

## Server events and client streaming

A canonical Runtime trace identifies request arrival, each semantic Token-ready
event and request termination. Its time origin, unit, output counts, request
identity and final state must agree. A client stream records a different
boundary: when the client receives data over HTTP/SSE.

The hand-calculated server-event example is runnable without GPU hardware:

```bash
mlsys-sim collect --kind runtime-trace \
  --experiment examples/v1_5/experiment.fixture.json \
  --raw examples/v1_5/runtime-trace.fixture.json --allow-test-fixture \
  --out runs/runtime-trace-fixture
```

Its envelope uses `kind="runtime_trace"`,
`timing_boundary="server_token_ready_v1"`, source, `observed_until_ms` and an
ordered event array. `REQUEST_ARRIVAL` carries the request ID and timestamp;
each `TOKEN_READY` carries `token_count=1` and a contiguous one-based
`output_token_index`; `REQUEST_FINISH` declares the terminal status. Actual
arrivals and completed output counts must match the experiment. Normalization
never silently rewrites that workload to force comparability.

The vLLM path requests delta token IDs and usage metadata. It counts IDs rather
than SSE chunks, checks the final usage count, and records the first chunk that
contains token IDs even when its text is empty. A chunk may contain several
tokens. Their internal server generation times are unavailable, so the adapter
does not distribute one receive timestamp into invented per-token timestamps,
ITL or whole-stage observations. See the official
[vLLM completion protocol](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/completion/protocol.py).

On the configured AMD host, the client template is:

```bash
mlsys-sim collect --kind vllm --experiment your-amd-client-experiment.json \
  --plan examples/v1_5/vllm.plan.json --out runs/vllm-client
```

Set the actual served model, endpoint and run reference. `endpoint` is an
HTTP(S) origin with no embedded credentials or path. The adapter reads the
server's `/version` and calls `/v1/completions`; optional `runtime_version`
asserts the expected actual version. Authentication comes from the environment
variable named by `api_key_env`, not a secret inside the plan. The experiment
must provide prompt token IDs, no embedded warmup, and draining without a finite
drain limit. Worker, response-size and request time bounds are explicit.
This path emits client requests/metrics and raw evidence with
`canonical_comparable=false`; it does not produce an automatic `observed.json`
for server-token Compare.

The user-supplied 9070 XT package records nonempty **content chunks**, not token
IDs. Its client timestamps cannot be relabeled as canonical `TOKEN_READY`
events. Online Compare still requires matching workload, identity, policy and
timing contracts. Arrival schedules, send lag, output counts, warmup/cache state
and rate windows must be accounted for before comparing model predictions.

## Validate the complete evidence chain

```bash
mlsys-sim validate-hardware --plan examples/v1_5/fixture-validation.plan.json \
  --out runs/acceptance-fixture --allow-test-fixture
```

This example uses hand-calculated data adapted from the earlier V1 fixtures.
It contains no AMD measurements. Its permissive sample thresholds demonstrate
the workflow rather than define production accuracy or tail-latency criteria.

A full plan declares one training/holdout pair per local replica, online
experiment/observation pairs and explicit thresholds. For example:

```json
{
  "schema_version": 1,
  "steps": [{
    "replica_index": 0,
    "training_dataset": "training.fixture.json",
    "holdout_dataset": "holdout.fixture.json",
    "profile_version": "acceptance-fixture-v1"
  }],
  "online": [{
    "experiment": "experiment.fixture.json",
    "observation": "online-observation.fixture.json",
    "group_id": "fixture-demo"
  }],
  "thresholds": {
    "min_holdout_records": 2,
    "min_holdout_coverage": 1,
    "max_step_mape_percent": 10,
    "min_online_requests": 1,
    "min_online_repeats": 1,
    "max_online_relative_error": {"ttft_p50_ms": 0.1, "tpot_p50_ms": 0.1},
    "require_sla_agreement": false
  }
}
```

Paths resolve against the plan file. The workflow fits each new Profile from
training data, validates genuinely disjoint holdouts, inserts those Profiles
into copied online experiments, runs the simulator and compares sourced
observations. Input snapshots and intermediate reports remain available in
the acceptance directory. Record/group/shape leakage is an input error.
Communication datasets can additionally fit a separately selected communication
Profile; fitting does not silently patch stage timing.

Online observations need complete request-ID pairing and counts that match the
observed rows. An aggregate-only summary cannot satisfy request-level evidence
requirements. Distinct source references on identical observation content do
not constitute repeated runs. Missing relative-error denominators, coverage,
samples or repeat evidence remain unavailable rather than becoming zero error.
Selecting a p99 acceptance metric additionally requires at least 100 paired
requests, even if `min_online_requests` was set lower.

| Verdict | Meaning |
| --- | --- |
| `PASS` | All required evidence is declared measured and the specified coverage, sample and error criteria pass for the reported scope |
| `FAIL` | Sufficient comparable evidence exceeds an acceptance threshold |
| `INSUFFICIENT_EVIDENCE` | Coverage, counts, matching observations or comparable metrics cannot support the requested decision |
| `TEST_FIXTURE_ONLY` | At least one required input is a fixture; numerical criteria are reported separately and hardware validation stays false |

`status="COMPLETED"` says that evaluation ran. Read the verdict,
`criteria_passed`, `hardware_validated` and validation scope independently.
An explicit `PASS` is limited to the supplied identities and observed domain;
it is not a statement about every AMD GPU or serving policy.
The current gate evaluates step latency and the selected online metrics. An
optional standalone communication fit is retained separately and does not
establish held-out communication accuracy. Workspace or capacity correctness
also needs its own attributed measurements and acceptance evidence.

The output retains `validation-plan.json`, per-replica
`step-replica-N/{training,holdout,profile,validation}.json`, online
`online-N/{experiment,observation,comparison}.json` and its `predicted/` run,
plus `acceptance.json`. An audit-only plan can reference an `evidence_audit`
file; its verdict is `INSUFFICIENT_EVIDENCE` rather than fabricated stage or
online coverage.

## Supplied reference packages and their limits

The importer audits package structure, hashes, provenance, fields and coverage.
Package README, archived prompts, shell programs and Runtime snapshots are data
sources. Importing does not execute them or adopt their instructions. Raw
archives, full prompts and generated responses remain external to this library.

```bash
mlsys-sim collect --kind reference --reference-dir /path/to/extracted-release \
  --out runs/reference-audit
```

The supported inputs are extracted directories: the AMD layout with
`00_environment/` and `02_measured/`, or the legacy layout with `manifest.json`,
`experiment.json` and `tables/`. The importer does not extract an archive or
open nested raw tarballs. Unsafe paths, symlinks, missing metadata, non-finite
values and declared size limits fail explicitly. Count/hash discrepancies are
retained as audit failures rather than repaired; `status="COMPLETED"` still
means that the audit ran, not that every integrity check passed.

`audit.json` contains counts, sources, hashes, model compatibility and validation
readiness. Sanitized `request_service.jsonl` retains request identifiers and
numeric service fields. `raw_records.json` describes source table schemas,
units and counts rather than copying prompt or output text. No reference import
creates a canonical stage dataset, semantic Token trace or online Compare
contract by guessing missing fields, and every import keeps
`hardware_validated=false`.

| Reference | Recorded coverage | Boundary and compatibility limits |
| --- | --- | --- |
| Legacy RTX 5080 package | 48 runs, 2,400 formal requests, 63 profiling windows; NVIDIA/CUDA, BF16, TP1 | Closed-loop completion refill; no request arrival or token-ID trace; client content-chunk timing and separate deep profiling cannot become AMD stage Profiles |
| RX 9070 XT release | 195 runs across 65 cells and three repetitions; 9,750 measurement plus 1,950 warmup requests | Client content-chunk timestamps and C1 request-level service views; 123 closed-loop runs and 72 open-loop runs require their own arrival/window interpretation |

The legacy manifest has one README hash mismatch; its declared data/profiling
hashes match. Treat integrity diagnostics at the affected-file scope rather
than silently replacing the manifest. The AMD release selects the 9070 XT via
device visibility; a host's other GPU is not the sampled device. Its recorded
vLLM runtime build takes precedence over a container tag, and HIP build/runtime
values are not interchangeable with a ROCm distribution version.
In this release the selected GPU is RX 9070 XT (`gfx1201`, 16,304MiB), while the
host also lists a 7900 XTX. The recorded vLLM build is
`0.29.1.dev0+g98dff2a81.d20261002`, rather than the `0.29.0` image tag. Its Torch
wheel string `2.13.0+rocm10.1.0` and HIP build `7.16.26385` are preserved as
different facts; neither is rewritten to make a desired version match.

Both packages use pinned Qwen3 revisions and BF16. Version 1.6 describes their
tied vocabulary weights, Q/K and final Norm, and independent residual/query
widths through the strict Qwen3 importer. The 4B model retains hidden width
2560 and query width 4096; no source dimension is changed to make it validate.
Architecture support is recorded independently from missing measurement
boundaries. A prefix-group label cannot replace prompt token ancestry.
No supplied run establishes RCCL/TP/PP,
preemption or swap prediction accuracy, and neither package supplies the
whole-stage shape and attributed Workspace records needed to fill those gaps.

For version-pinned internal vLLM collection, see
[Qwen3 and internal hooks](qwen3-and-vllm-hooks.md). It obtains runner shapes
and supported complete service boundaries directly; the supplied historical
client-content records are not retroactively upgraded to those new boundaries.

Native PP lazy receive and TP-sliced dual-tensor transport can provide isolated
stage records under the explicit 1.7 contract; see [native PP capture](native-pp-capture.md).
Its dependency-wait diagnostics are not independent P2P latency measurements,
and boundary costs remain analytical unless a separate native communication
Profile is explicitly selected. The historical packages do not
retroactively establish this multi-GPU coverage.

Version 1.8 adds `collect --kind pp-communication` for independent single-node
HIP directed P2P and receiver AllGather service. Data groups use NCCL/RCCL,
while CPU Gloo barriers control rounds outside the measured interval. Buffers
are preallocated out of place; actual stack/RCCL/device and correctness evidence
are required. Fitting produces a placement-specific native communication
Profile, not an automatic Infra patch or an end-to-end Runtime-boundary claim.
See [native PP communication calibration](native-pp-communication.md).
