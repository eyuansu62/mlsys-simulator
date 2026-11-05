# Workspace attribution and capacity validation

Dynamic Workspace attribution and capacity validation have been separate
workflows since version 1.9. Capacity validation uses a complete effective
experiment. The two
workflows share memory identities, but answer different questions: Workspace
records identify owned dynamic allocations; a capacity sweep records whether a
fresh Runtime can execute each candidate workload/configuration.

Version 1.13 adds Worker-level prefill/decode acquisition and an independent
exact-Workspace repeat evaluator. Raw acquisition support and complete
attribution are separate: every selected rank and storage lifetime still needs
the proofs below before calibration is allowed.

The dependency-free CPU core does not collect GPU timings by substitution.
Live collection needs the supported installed HIP stack and a user Runtime
adapter. CPU-injected evidence remains `test_fixture`, and successful software
checks do not establish AMD accuracy or `hardware_validated=true`.

## Attribute complete dynamic Workspace

A canonical Workspace dataset requires a complete Runtime-owned allocation
ledger with these attribution fields:

```json
{
  "method": "runtime_workspace_ledger_v1",
  "complete": true,
  "scope": "exclusive_dynamic_workspace",
  "excluded_categories": [
    "weights", "kv_pool", "graphs", "runtime", "pp_buffers", "communication_buffers"
  ],
  "includes_non_torch": true
}
```

This is a Runtime integration contract, not a label an adapter may attach to an
allocator peak. The user adapter must install and remove instrumentation for
all scratch owners within its declared domain, including allocations outside
Torch. Weight/KV/graph/Runtime, native PP backing/copies and communication buffers
are accounted for separately and must not be charged again as dynamic Workspace.

Allocation events identify physical underlying storage and generation. Views
and aliases of one storage count once; simultaneous distinct storages both
count. A release request does not make a storage reusable. Pending frees remain
live until the corresponding release-completed evidence arrives. Reused storage
identifiers need a new generation so that old and new leases are not merged.

`record_tensor` observes underlying storage capacity, not the caching allocator's
entire reserved arena. Allocator rounding, fragmentation and external allocations
still need separate headroom and capacity evidence; a storage bound alone does
not establish the physical OOM boundary.

Each rank needs explicit completeness evidence. A completed rank that observed
zero scratch can report `0`; a missing rank, uninstrumented owner or partial
trace reports unknown, not zero. Live leases at close, unmatched lifetime events,
a saturated event limit or incomplete rank coverage prevent canonical Workspace
calibration. Raw evidence and failure reasons remain available.

## Keep passive evidence partial

The pinned vLLM WorkspaceManager observer watches an existing manager without
initializing, resizing or locking it on its own. It observes the original size
request once and records actual buffers, all slots/lanes and aligned requested
bytes separately from allocated capacity. Its scope is
`managed_scratch_capacity` and remains partial raw evidence.

That manager does not prove coverage of activations, backend scratch, graph
pools, direct HIP allocations or communication. A zero manager total does not
prove zero complete Workspace. Its warmed persistent capacity must also not be
charged again when already included in Runtime/graph reserves. A manager's
locked-growth AssertionError is a contract failure, not allocator OOM.

Global allocated/reserved peaks and `peak - baseline` likewise do not prove
ownership. Torch allocator snapshots cannot see all non-Torch allocations;
point-in-time snapshots do not reconstruct an unobserved high-water mark, and
a wrapping history buffer may have lost events. These remain diagnostics until
a complete ownership protocol establishes the supported Workspace bound.

## Fit an independent per-rank Workspace Profile

Workspace data uses `kind="workspace_measurements"`; fitting produces
`kind="workspace_profile"`, method `exact_rank_max_v1`. It contains no duration.
Each row identifies phase, stage, ordered query/context arrays and one observed
peak for every TP rank. Mixed rows also preserve ordered `request_phases`.

Repeated observations of the same shape are reduced by the maximum separately
for each rank. The result is an observed bound for that exact shape and identity,
not an interpolation model or a universal upper bound for unseen execution.
A selected Workspace Profile requires exact lookup: there is no interpolation,
extrapolation, silent zero or analytical fallback for missing data.

```bash
mlsys-sim calibrate --workspace --dataset workspace-measurements.json \
  --profile-version workspace-v1 --out runs/workspace-fit
```

Test data requires `--allow-test-fixture` and keeps fixture source in its Profile.
Files have strict identity, source, version and SHA provenance. A complete
adapter can be collected through the Workspace path using the existing step
sample plan and the Runtime adapter factory:

```bash
mlsys-sim collect --kind workspace --experiment bound-experiment.json \
  --plan workspace.plan.json --adapter runtime_package:factory \
  --out runs/workspace-capture
```

This is a live-host template. The adapter supplies the complete scratch
instrumentation; the default passive manager observer does not supply it.
Stage samples retain actual phase/query/context shapes, warmups and iterations.
Warmup is not converted into measured shape evidence. The adapter must preserve
original Runtime behavior and close its instrumentation after collection.
`examples/v1_9/workspace.plan.live-template.json` shows the step sample plan with
an explicit `max_events_per_sample` bound; it requires a real bound experiment
and complete Runtime integration to execute.

The adapter exposes `metadata()`, `prepare(sample)`, `run_step(sample)` and
cleanup. Complete attribution additionally requires
`begin_workspace_capture(sample, recorder)` and
`end_workspace_capture(sample, recorder)`. Metadata uses
`workspace_attribution="runtime_ledger"`,
`complete_workspace_attribution=true`, `includes_non_torch=true` and the exact
excluded-category set above, together with the supported Runtime/timing binding.
Begin must account for any dynamic scratch already live at epoch entry and
instrument subsequent ownership events. After execution the collector completes
GPU work before end removes instrumentation and completes every rank. End runs
in cleanup even on failure; an incomplete/failed epoch cannot become a dataset.

`WorkspaceRecorder` exposes `allocate(rank, allocation_id, storage_id, size_bytes)`,
`record_tensor(rank, allocation_id, tensor)`, `request_free`,
`release_completed`, `complete_rank` and `seal`. The following CPU-only event
example illustrates alias accounting and an explicitly completed zero rank; it
does not measure a Runtime or prove that all scratch owners were instrumented:

```python
from mlsys_sim.workspace_capture import WorkspaceRecorder

recorder = WorkspaceRecorder(2, epoch_id="cpu-ledger-example")
recorder.allocate(0, "owner", "scratch-generation-1", 4096)
recorder.allocate(0, "view", "scratch-generation-1", 4096)
for allocation in ("owner", "view"):
    recorder.request_free(0, allocation)
    recorder.release_completed(0, allocation)
recorder.complete_rank(0)
recorder.complete_rank(1)  # Observed zero, with explicit rank coverage.
assert recorder.seal() == [
    {"tp_rank": 0, "peak_bytes": 4096},
    {"tp_rank": 1, "peak_bytes": 0},
]
```

The live Workspace CLI has no raw-ledger importer and no CPU timing fallback.
CPU injection is a library test surface and retains fixture source.

## Collect Worker-level Prefill and Decode

The optional `scripts/rocm_workspace_audit/workspace_stage_session.py` supervisor
uses a pinned `WorkspaceWorker` subclass, preserving the real MRv1 model runner.
Its layouts are `single` (TP1×PP1), `tp2` (TP2×PP1) and `pp2` (TP1×PP2); these
are separate identities and acquisitions. Two physical GPUs do not establish
TP2×PP2 or a larger replica topology.

The current raw acquisition domain is Qwen3-1.7B BF16, synchronous eager MRv1,
one active sequence, FIFO, max model/batched token counts of 512, block size 16
and a fixed 256MiB KV budget. Cache, chunking, mixed batches, CPU swap/offload, graphs and
async execution are excluded from this collector's domain. The experiment must
declare the actual ordered devices, effective model/PP endpoint identity and
matching software/configuration; unsupported or missing facts are rejected.
The model is an existing local directory, not a network download.

Each prompt of length 16 or 32 naturally generates four greedy outputs. The
prepared batch supplies the real phase, query, context and sampling boundary:
one prefill at context 0, followed by decode query 1 at contexts 16/17/18 or
32/33/34. With `--repeats 3`, each rank has 24 formal iterations, including 18
decode iterations. Warmup remains in the raw capture with explicit cohort
labels. It is not inferred from request names or removed from the originals.

Build the matching optional SDK/Torch probes and install the matching simulator
wheel first; see the [tool instructions](../scripts/rocm_workspace_audit/README.md).
Run each layout with its own complete actual experiment and new output paths:

```bash
SCRIPT_ROOT=/app/work/mlsys-simulator/scripts/rocm_workspace_audit
python "$SCRIPT_ROOT/workspace_stage_session.py" \
  --mode single --experiment /app/work/single.actual.json \
  --cohort train --repeats 3 --out /app/work/ws-single-train
```

Use `--mode tp2` or `--mode pp2` with the corresponding actual experiment for
the other layouts. PP2 requires `vllm_native` transport with a physical-token
bound covering the actual 512-token persistent input targets. Visibility is
selected before child GPU imports, and per-worker SDK output uses a PID
template; actual PID, stage, TP rank and logical-to-physical device facts must
match. The parent supervisor does not create a CUDA/HIP context for timing.

`--native-max-events` is an explicit SDK retention budget: default 1,048,576,
with a supported upper bound of 4,194,304 when the compiled probe supports it.
Choose it before acquisition. A larger bound does not repair a saturated or
truncated trace, and no slow or unsupported rows may be discarded to fit it.

The outer execution supervisor must preserve the actual return code and
receipt, source/DSO/header/library hashes, logs and normal process exit. After
exit, assemble the unmodified raw evidence:

```bash
python "$SCRIPT_ROOT/assemble_stage_workspace.py" \
  --session /app/work/ws-single-train \
  --receipt /app/work/ws-single-train-receipt.json \
  --out /app/work/ws-single-train-assembled
```

The receipt must come from that actual execution, not an expected success.
Multi-card communication needs real per-rank constructor, library, communicator
and backing evidence through `--communication-bindings`; a declaration of
ownership does not supply that proof. The assembler independently checks the
native events and recorded source/Runtime facts. Unknown communicator backing,
proxy-thread causality, incomplete finalization, open API calls or dropped
events keep the result partial. An assembler command that writes a report can
exit normally while attribution is partial: inspect `attribution_complete` and
the issues. Only a complete report emits canonical `dataset.json` for fitting.

The Worker boundary includes receive allocation, the real runner, and the
sender's original handoff. PP output tensors can survive into the next
iteration through the group's self-retention queue. Their storage generations,
pending releases and natural reap are recorded; the observer must not pre-read
lazy receives or wait/reap early to make them disappear. A separate final
teardown drain can prove eventual closure, but is excluded from formal peaks.
PP buffers and fixed communicator storage stay in their own budgets, rather
than becoming dynamic Workspace. Shape/rank-complete Torch evidence alone is
still insufficient without native backing and ownership closure.

These snapshots, synchronization points and native callbacks materially
instrument execution. Exported stage durations are diagnostic; do not use them
to fit a new stage-latency Profile. A complete Workspace dataset deliberately
contains no duration. Actual hardware results and their supported domains belong
in the [live validation record](workspace-live-validation.md), rather than being
inferred from raw process exit or CPU fixtures.

## Validate a frozen Workspace Profile independently

Fit only a complete training dataset and freeze its Profile bytes before
starting the independent acquisition:

```bash
mlsys-sim calibrate --workspace \
  --dataset /app/work/ws-single-train-assembled/dataset.json \
  --profile-version workspace-single-v1 --out /app/work/ws-single-fit
```

Then run a fresh Runtime with the same supported layout/configuration,
`--cohort repeat`, and a new output directory. Audit it independently before
evaluation; do not fit on the repeat or backfill missing facts into it.

```bash
mlsys-sim validate --workspace-repeats \
  --training-dataset /app/work/ws-single-fit/dataset.json \
  --profile /app/work/ws-single-fit/profile.json \
  --dataset /app/work/ws-single-repeat-assembled/dataset.json \
  --out /app/work/ws-single-repeat-validation
```

Use the fit's exact canonical training snapshot. The API
`validate_workspace_repeats(profile, training_dataset, repeat_dataset, *,
allow_test_fixture=False)` checks its SHA, recomputes the retained per-rank fit,
and verifies identity, attribution and fitting provenance. Explicit acquisition
notes must supply session, clock domain and sorted raw capture hashes; reused
session/capture/record/group evidence is rejected. Shared source and binary
hashes remain distinct from raw capture hashes and do not make independent
acquisitions into reused observations.

The report preserves each stage and TP rank. It gives record coverage, rank
coverage, retained-shape coverage, absolute byte error and byte underprediction.
Its underprediction count is per rank cell; the number of affected records is
reported separately.
Observed zero is valid; a missing rank is invalid data and a missing shape is
out of domain. Evaluation is exact-only: no interpolation, extrapolation or
automatic split is performed. It is independent repetition within retained
shapes, not unseen-shape generalization or automatic hardware acceptance.
Fixtures require explicit opt-in and remain fixtures; hardware status stays
false and there is no online SLA verdict.

The existing simulator can consume matching Workspace Profiles with TP/PP,
Prefix Cache, chunking, CPU swap and mixed batches, as checked by CPU combination
regressions. That software compatibility does not extend this measured
collector domain. Each selected Profile still needs its actual Runtime identity
and exact queried shapes/member roles; enabling another policy cannot authorize
reuse of unmeasured Workspace bounds.

## Select Workspace independently

Extend a complete experiment with a local Workspace Profile:

```json
{
  "calibration": {
    "workspace_profile_path": "workspace-fit/profile.json",
    "workspace_profile_version": "workspace-v1"
  }
}
```

For multiple independent replicas, use `replica_workspace_profile_paths` in
replica order, each file matching that local layout. Test Profiles additionally
require `calibration.allow_test_fixture=true`. The CLI snapshots and hashes the
selected files independently from stage and communication Profiles.

An explicitly selected Workspace Profile replaces the dynamic Workspace budget
for the queried shape/rank. Fixed weights, KV pool, graphs, Runtime, native PP
receive/backing/copy buffers, communication reserves and safety constraints stay
separate. Static feasibility does not reject a selected complete Profile merely
because an unused analytical Workspace-base estimate was large; actual selected
Workspace is checked at admission per rank.

Without a selected Workspace Profile, the existing path remains: use a supported
stage Profile Workspace observation when present, otherwise a labeled analytical
assumption. Selecting a Workspace file does not change stage Profile identity
or timing. TP/PP, native communication Profiles, caches, chunking, mixed batches
and swap retain their existing resource ownership and safe recovery boundaries.

## Run the Workspace fixture workflow

The canonical fixture explicitly contains two observations of each shape and
all TP-rank values, including a zero Prefill Workspace on stage 0/rank 0. It is
hand-calculated software evidence, not the output of live Runtime attribution.

```bash
mlsys-sim calibrate --workspace \
  --dataset examples/v1_9/workspace-measurements.fixture.json \
  --profile-version workspace-fixture-v1 --allow-test-fixture \
  --out runs/workspace-fixture-fit
mlsys-sim predict --experiment examples/v1_9/native-pp-workspace.fixture.json \
  --out runs/workspace-fixture-predict
mlsys-sim simulate --experiment examples/v1_9/native-pp-workspace.fixture.json \
  --out runs/workspace-fixture-serving
```

The experiment selects the shipped `workspace.profile.fixture.json`, generated
from that training fixture, and the separate v1.8 native communication fixture.
It uses synthetic stage durations and preserves test source for each Profile.
Prediction/serving therefore run without a live adapter or any prior output
path. Missing requested shapes still fail, even for fixtures.

## Sweep capacity with a fresh Runtime

Capacity collection runs a real user Runtime rather than substituting synthetic
allocation for model inference. Every axis value and repeat starts a new
Runtime/process with a known initial state. An OOM-mutated Runtime is never reused
for the next point. Original exceptions, phase and partial evidence are retained.

Supported axes vary one dimension of the effective experiment:

| Axis | Meaning and required evidence |
| --- | --- |
| `input_tokens` | Change input length; existing prompt IDs may be truncated, never invented to extend a template |
| `request_count` | Change the request cohort; this does not assert an actual GPU batch size or replace the configured scheduler |
| `kv_pool_bytes` | Change configured KV capacity and require Runtime proof that the requested physical pool was actually applied |

Each point binds the complete effective configuration SHA, session/cohort and
source. Predictions run the normal simulator over that entire configuration,
including arrivals, scheduling, memory/cache policies and failures; a shape-only
byte formula is not substituted for online execution. Missing Workspace or other
calibration coverage remains an explicit prediction gap.

A live capacity plan selects the axis, candidate values and bounded repeats:

```json
{
  "schema_version": 1,
  "dataset_id": "capacity-live-01",
  "axis": {"name": "input_tokens", "values": [256, 512, 1024]},
  "repeats": 2,
  "timeout_s": 120,
  "max_output_bytes": 8388608,
  "source_reference": "Fresh Runtime capacity sweep on the bound AMD host"
}
```

```bash
mlsys-sim collect --kind memory-capacity --experiment bound-experiment.json \
  --plan capacity.plan.json --adapter runtime_package:factory \
  --out runs/capacity-capture
mlsys-sim validate --memory-capacity --dataset runs/capacity-capture/dataset.json \
  --out runs/capacity-validation
```

The shipped `examples/v1_9/capacity.plan.live-template.json` is a live-host
template. Input/request-count axes require one base request; when prompt IDs
are present, its template must already contain enough actual IDs for every
candidate. The plan does not synthesize missing prompt identity.

The adapter factory receives the effective experiment and supplies Runtime
metadata, `prepare()`, `run()` and optional `close()`. Run distinguishes completed
execution from explicit admission refusal. Metadata must report isolation,
effective-configuration and applied-axis evidence. A KV-pool request without
corresponding actual per-device storage proof is insufficient. A raw import
plan uses `raw_path` for a complete `memory_capacity_measurements` dataset,
not launcher stdout, plus an explicit source. Imports revalidate axis, effective
configuration and source; they do not rerun or repair archived Runtime state.

## Classify OOM and report intervals

The supported physical-OOM boundary requires a typed
`torch.cuda.OutOfMemoryError` on the verified HIP path and a bound device
allocator `num_ooms` increase for its actual failure phase. A typed exception
without that evidence remains an unsupported/censored result, not a proven
axis threshold. Startup/model loading, preparation and execution phases stay
distinct; only scoped preparation/execution failures can close a workload-axis
bracket. An OOM before workload preparation stays diagnostic evidence rather
than a claim about input/request/KV capacity.

A message containing “out of memory,” logical KV admission refusal, timeout,
process death, peer-rank failure or lock AssertionError is not silently
reclassified as that device OOM. One rank's OOM with missing peers also does not
provide complete same-session multi-rank evidence by itself. Completed trials
must identify the entire declared request cohort; cleanup failure does not
prove successful isolated execution.

For the KV-pool axis, successful application binds actual unique pool tensor
storage on every device. If the pool allocation itself fails, there may be no
completed pool storage to inspect. The supported OOM proof instead binds the
failed `torch.empty` request shape/element width and exact target pool bytes to
the effective configuration, failure phase and device whose allocator counter
increased. A requested pool setting or arbitrary later OOM is insufficient.

Physical device OOM and a fixed logical KV-pool limit are different capacity
boundaries. Allocated/reserved/free totals, static category reserves, allocator
policy and any artificial process limit need their own context. A process memory
fraction does not become physical device capacity.

Validation retains repeated outcomes rather than selecting the most convenient
one. Success/OOM disagreement at one value is unstable. A larger success after a
smaller OOM is nonmonotonic. Timeouts and other failures do not close an OOM
bracket. Missing success or OOM endpoints are censored evidence.

Only adjacent observed success→typed-OOM points can form a bracket, with their
axis resolution. This is an interval within the tested configuration and source,
not an exact threshold or permission to binary-search a universal knee. Reports
preserve unstable/nonmonotonic flags, prediction coverage and incomplete evidence.
A valid fixture bracket or a successful command remains
`hardware_validated=false` without separate matching live hardware acceptance.

## Exercise capacity validation with CPU fixtures

The raw capacity fixture contains a complete dataset with explicit fixture
source, effective configurations and per-trial evidence. It exercises import,
classification and bracket reporting; its OOM labels do not represent a physical
allocation failure on this machine.

```bash
mlsys-sim collect --kind memory-capacity \
  --experiment examples/v1_9/capacity-base.fixture.json \
  --plan examples/v1_9/capacity.plan.fixture.json \
  --allow-test-fixture --out runs/capacity-fixture-import
mlsys-sim validate --memory-capacity \
  --dataset runs/capacity-fixture-import/dataset.json \
  --allow-test-fixture --out runs/capacity-fixture-validation
```

The import plan references `capacity-measurements.fixture.json`. It does not
launch a Runtime or require a live adapter. Validation recomputes predictions
for the stored effective experiments and preserves fixture status and any
prediction disagreement. In the shipped fixture both repeats complete at four
input tokens and carry typed-OOM evidence at eight. The reported bracket is
`[4, 8]` with resolution `4` and `exact_threshold=null`; it is a software contract
example, not an AMD capacity result. No Profile fitting or hardware acceptance
is implied.

## Source and interpretation boundaries

The pinned primary sources explain why complete ownership and typed failures
are necessary: [vLLM WorkspaceManager](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/worker/workspace.py),
[vLLM startup non-KV profiling](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/utils/mem_utils.py),
[PyTorch allocator/snapshot implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/cuda/memory.py)
and [allocator history visibility](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/docs/source/torch_cuda_memory.md).
This library does not turn those partial mechanisms into automatic complete
Runtime attribution or claim AMD measurements from CPU fixtures.

## Pinned live Qwen3 KV-pool adapter

Version 1.12 provides the optional entry point
`mlsys_sim.vllm_capacity_adapter:kv_pool_factory`. It imports neither Torch nor
vLLM on ordinary library import. The existing local model must be selected
explicitly with `MLSYS_VLLM_CAPACITY_MODEL_PATH`; it does not download weights.

The initial supported scope is the fixed `98dff2a81` synchronous MRv1 ABI,
unquantized full-attention Qwen3 FP16/BF16, one visible HIP device, TP1/PP1,
block16, one immediate token-ID request, FIFO/eager, no graphs, sleep mode,
CPU offload, prefix/chunk/mixed/swap or separate internal-capture activation.
The adapter validates the local config identity and actual loaded architecture;
its metadata does not claim a full weight-file content revision proof.

For the `kv_pool_bytes` axis it verifies actual native KV descriptors, block/page
alignment and unique Tensor storage. Arbitrary GiB values may round down inside
the Runtime and are rejected if they do not equal the applied pool. The tested
Qwen3-1.7B BF16 configuration has a 1,835,008-byte block quantum; derive and verify
that quantum for the selected model rather than applying it universally.

Only the original pinned allocator's final `torch.zeros` call, with the exact
failed target/device/configuration and typed allocator-counter increase, can
supply this adapter's KV OOM proof. The validator retains existing `torch.empty`
support as well. Weight load, startup profiling, subsequent warmup, manager
assertions, admission refusal and unknown failures do not acquire invented KV
allocation evidence. Each OOM-mutated Runtime remains owned by its worker until
process exit and is not reused.

The experiment's declared analytical category reserves and efficiencies remain
prediction inputs; they are not claimed as actual Runtime allocations. This
adapter rejects an externally supplied execution-config digest it cannot prove
it applied. Capacity observations, Workspace attribution and general prediction
acceptance remain separate workflows.

The [limited Workspace validation record](workspace-live-validation.md) documents
the first complete live requested-storage domain. Its completeness applies only
to the stated exact shapes/configuration; capacity still needs real OOM evidence.
