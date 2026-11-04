# Python API

Install distribution `mlsys-simulator` and import `mlsys_sim`. Version 1.6 exposes the
simulation, profile, calibration and comparison functions through this package,
with optional hardware collection and evidence-acceptance commands;
`mlsys_sim.__version__` is the software version. JSON schema and Profile version
labels are separate contracts.

## Simulate and predict

```python
from mlsys_sim import CalibrationSpec, Experiment, RequestSpec, simulate

experiment = Experiment(
    workload=(RequestSpec("r1", arrival_ms=0, input_tokens=16, output_tokens=4),),
    calibration=CalibrationSpec(backend="synthetic"),
)
run = simulate(experiment)
assert run.result["status"] == "COMPLETED"
```

`Experiment` and its nested specifications are frozen dataclasses. Use
`dataclasses.replace` to derive a configuration. Durations use milliseconds,
storage uses bytes, bandwidth uses decimal GB/s and compute uses TFLOP/s.
`simulate` does not execute GPU tensors or sleep in wall-clock time. The separate
live collection adapters execute configured Runtime work on a supported AMD
machine; they are not a performance backend for the CPU simulator.

| Export | Purpose |
| --- | --- |
| `Experiment`, `RequestSpec` | Experiment and replay requests |
| `ModelSpec`, `DeviceSpec`, `InfraSpec` | Dense model, ordered devices and topology |
| `StrategySpec`, `MemoryConfig` | Scheduling, PP microbatches/feedback, GPU/host budgets and preemption policies |
| `CalibrationSpec` | Analytical/synthetic parameters or Profile selection and identity |
| `EvaluationSpec`, `ExecutionSpec` | SLA, admission window, drain, warmup, initial state and budgets |
| `SimulationResult` | `result`, `requests` and `events` in memory |
| `SimulationError` | Error with a machine-readable `code` |

`load_experiment(path)` reads strict JSON/TOML or optional YAML. Six top-level
sections may reference files; references resolve relative to the experiment,
and model references inside a referenced strategy resolve relative to that file.
`experiment_from_dict(mapping)` validates an inline mapping. Direct dataclass
calls to simulation are also validated. Future output lengths remain confined
to the replay execution oracle.

`predict_experiment(experiment)` analyzes workload rows as one shape batch,
ignoring arrival times. It reports stage costs, TP/P2P communication, memory
feasibility and batch-limit flags. It supplies no online metrics and reports
`SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING`; use `simulate` for online PP as well
as single-group serving. Inspect feasibility separately from command execution
status.

`predict_experiment` does not perform prefix lookups, chunk scheduling or online
mixed batching. Its
shape remains the complete input batch even when those online policies are
enabled; compare online policies through `simulate`.

## Describe and import Qwen3 models

`ModelSpec` accepts `tie_word_embeddings`, `qk_norm` and `final_norm`, all
`False` by default. Its read-only `query_projection_size` is
`num_attention_heads × head_dim`; this width need not equal `hidden_size`.
Q/K Norm contributes two shared head-dimension weight vectors per layer and
rank. Final RMSNorm contributes one residual-width vector on the last stage.
A tied embedding/LM-head allocation is counted once when both roles reside on
the same physical rank. PP endpoints on different devices retain distinct
physical copies. Tying storage does not remove LM-head computation.

```python
from mlsys_sim.qwen3 import load_qwen3_model, model_from_qwen3_config

bundle = load_qwen3_model(
    "examples/v1_6/qwen3-4b.hf.json", name="Qwen3-4B",
    revision="1cfa9a7208912126459214e8b04321603b3df60c",
)
model_data = bundle["model"]
```

`model_from_qwen3_config(config, *, name, revision, kv_dtype=None)` returns a
`ModelSpec`. `load_qwen3_model(path, ...)` returns JSON-ready model data and
provenance, including source bytes and canonical architecture hashes. Neither
loads weights, downloads code or infers a model revision. Only dense, full
attention, unquantized Qwen3ForCausalLM with supported inference options is
accepted. Explicit storage dtype selects its byte width; a Runtime-specific KV
dtype can be supplied separately.

`normalized_qwen3_architecture(config)` and
`qwen3_architecture_fingerprint(config)` expose the canonical supported
architecture mapping and its SHA256. The mapping retains execution-relevant
geometry, Norm/RoPE/context configuration and excludes HF path/version metadata.
`ModelSpec.architecture_config_sha256` defaults to an empty string; nonempty
values must be 64 lowercase hex characters and participate in Profile identity.
Default-false model flags and an empty fingerprint are omitted from canonical
legacy identities, preserving older Profile contracts.

`CalibrationSpec.runtime_config_sha256` independently fingerprints the actual
Runtime execution snapshot. Its empty default preserves old identities. A
nonempty lowercase SHA256 participates in Profile identity, so changing a
backend, graph or other captured execution setting cannot silently reuse a
different measured configuration.

## Configure multiple replicas

```python
from mlsys_sim import (
    CalibrationSpec, DeviceSpec, Experiment, InfraSpec, RequestSpec,
    StrategySpec, simulate,
)

experiment = Experiment(
    workload=tuple(RequestSpec(f"r{i}", 0, 4, 2) for i in range(4)),
    infra=InfraSpec(devices=(DeviceSpec("replica0"), DeviceSpec("replica1"))),
    strategy=StrategySpec(replicas=2, replica_routing="round_robin", scheduler="fifo"),
    calibration=CalibrationSpec(backend="synthetic", prefill_ms=2, decode_ms=1),
)
run = simulate(experiment)
print(run.result["routing"], run.result["replicas"])
```

`replicas` defaults to `1`; `replica_routing` defaults to `"least_loaded"` and
also accepts `"round_robin"` and `"prefix_aware"`. Provide exactly R×PP×TP devices
in replica/stage/TP-rank order. Each replica owns its GPU KV pool, Prefix Cache,
host swap pool and DMA lane; `cpu_swap_bytes` is per replica. Clock, arrival
window and execution budgets remain global. Requests stay on their selected
replica through recovery. The router never uses future output targets.

`result.replicas` contains `{replica_index, device_ids, result}` records.
`result.routing` retains assignments and load observations; request rows and
local events carry `replica_index`. Global service metrics are recomputed on
one common window. They are not averages of local percentiles. See
[multi-replica routing](multi-replica.md) for policy tie-breaks and failure scope.

For `backend="profile"` with R>1, supply exactly R
`CalibrationSpec.replica_profile_paths` entries and leave `profile_path=None`.
Each file describes that replica alone and must match its local device
identity. CLI runs snapshot these as `profiles/replica-0.json`, etc., and retain
hashes/sources in `manifest.replica_profiles`. R=1 keeps `profile_path` and the
existing `profile.json` artifact. Multi-replica prediction returns
`replica_predictions`, analyzing the same isolated input shape on each layout;
it performs no online routing and supplies no online SLA.

`MemoryPlanner` and direct performance models operate on one replica.
The internal layout helper `mlsys_sim.layout.replica_experiment(experiment, i)`
constructs the corresponding local experiment for these resource/shape APIs.
`MemoryPlanner.peek_prefix_tokens(request_spec)` observes a local match without
registering the request, pinning KV, changing counters or touching LRU order.
Dynamic request registration remains an engine/coordinator responsibility.

## Configure Prefix Cache and Chunked Prefill

```python
from mlsys_sim import CalibrationSpec, Experiment, RequestSpec, StrategySpec, simulate

tokens = tuple(range(24))
experiment = Experiment(
    workload=(
        RequestSpec("first", 0, 24, 2, prompt_token_ids=tokens, cache_namespace="demo"),
        RequestSpec("repeat", 20, 24, 2, prompt_token_ids=tokens, cache_namespace="demo"),
    ),
    strategy=StrategySpec(
        prefix_cache=True, prefix_cache_max_blocks=8,
        chunked_prefill=True, prefill_chunk_tokens=8, max_batch_tokens=8,
    ),
    calibration=CalibrationSpec(backend="synthetic", prefill_ms=2, decode_ms=1),
)
run = simulate(experiment)
print(run.result["prefix_cache"], run.result["prefill"])
```

The four StrategySpec defaults are `prefix_cache=False`,
`prefix_cache_max_blocks=128`, `chunked_prefill=False` and
`prefill_chunk_tokens=512`. Both counts must be positive integers. Workload
`prompt_token_ids` defaults to `()` and `cache_namespace` to `""`. Supplied IDs
must match `input_tokens` in length and be integers in `[0, vocab_size)`;
omitted IDs bypass caching. Prefix reuse leaves at least the final prompt Token
to compute its output. Cache storage shares the fixed GPU KV pool.

Chunk queries are bounded by remaining prompt work, `prefill_chunk_tokens` and
the available Batch Token budget. Intermediate chunks retain context and emit
no output; continuous batching gives Decode priority when chunking is enabled.
FIFO still holds one request to completion. These policies also operate through
the PP pipeline and safe swap/recompute recovery. See
[Prefix Cache and chunking](prefix-cache-and-chunking.md) for ownership, metrics
and the conservative per-chunk timing contract.

At the resource layer, `MemoryPlanner.abandon_prefix_lookup(request_id)` rolls
back a provisional prefix lookup when a request has not started execution and
admission cannot proceed. Schedulers use this to release temporary cache pins
and permit a later lookup retry. It must not be used to release KV that has
already launched; those references require the normal safe completion/recovery
boundary. Cache snapshots report these rollbacks as `abandoned_lookups`.

## Configure mixed Prefill/Decode batches

```python
from mlsys_sim import CalibrationSpec, Experiment, RequestSpec, StrategySpec, simulate

experiment = Experiment(
    workload=(RequestSpec("decode-owner", 0, 4, 3),
              RequestSpec("prompt", 1, 8, 2)),
    strategy=StrategySpec(
        mixed_batching=True, chunked_prefill=True, prefill_chunk_tokens=4,
        max_batch_size=2, max_batch_tokens=5,
    ),
    calibration=CalibrationSpec(
        backend="synthetic", prefill_ms=2, decode_ms=1, mixed_ms=3,
    ),
)
run = simulate(experiment)
```

`mixed_batching` defaults to `False` and does not imply chunking. Continuous
batching selects Decode first, then packs eligible Prefill into the same stage
execution. Without chunking, the whole remaining prompt must fit the available
Token budget. FIFO preserves its single owner, and a one-sequence batch cannot
mix phases. For PP, set `pipeline_microbatch_size` to at least two to permit a
mixed microbatch. Recompute retains its own homogeneous recovery boundary.

`CalibrationSpec.mixed_ms` defaults to positive `10.0` milliseconds and controls
only synthetic mixed service. Analytical mixed service uses one joint macro
estimate; it does not sum standalone prefill and decode times. Profile mixed
queries supply keyword-only `request_phases`, for example:

```python
step = model.estimate_step(
    "mixed", [1, 4], [32, 8], stage_index=0,
    request_phases=["decode", "prefill"],
)
```

Both serving roles are required and each Decode query must equal one. A
one-Token Prefill is still Prefill; query length cannot replace role metadata.
`estimate_profile` and `ProfileModel.workspace_bytes` accept the same
`request_phases` keyword. `MemoryPlanner.check` and `reserve` use it when checking
a complete joint Workspace and committing all member KV targets atomically.
Only mixed calls accept this metadata. Mixed Profiles require exact ordered
roles, queries, contexts and stage; missing shapes raise
`OUT_OF_CALIBRATION_DOMAIN` even when homogeneous extrapolation is enabled.

Prefix lookup rollback, safe cache publication, PP feedback and CPU swap pinning
apply per member. Intermediate Prefill chunks never produce an output merely
because another member decoded in that batch. See [mixed batching](mixed-batching.md)
for the complete record and lifecycle contract.

## Configure online PP and swap

```python
from mlsys_sim import (
    CalibrationSpec, DeviceSpec, Experiment, InfraSpec, RequestSpec,
    StrategySpec, simulate,
)

experiment = Experiment(
    workload=tuple(RequestSpec(f"r{i}", 0, 8, 1) for i in range(3)),
    infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))),
    strategy=StrategySpec(
        pipeline_parallel_size=2,
        stage_layers=(2, 2),
        pipeline_microbatch_size=1,
        pipeline_max_inflight_microbatches=3,
        max_batch_tokens=64,
        pipeline_p2p=False,  # Explicit ideal-link teaching case.
    ),
    calibration=CalibrationSpec(backend="synthetic", prefill_ms=2),
)
run = simulate(experiment)
print(run.result["pipeline"])
```

Each stage owns a TP group and microbatch queue. The same request cannot launch
another iteration until its sampled Token feedback is available. Final-stage
Token readiness and the later safe release of feedback references are distinct
boundaries. All-stage KV and persistent activation storage are budgeted before
launch; see [PP and swap](pipeline-and-swap.md) for the formula and link policy.

Select CPU swap with `StrategySpec(preemption="swap", memory=MemoryConfig(
cpu_swap_bytes=...))`. D2H/H2D bandwidth and latency are MemoryConfig fields.
The host capacity must be positive. `swap_fallback="recompute"` is optional and
explicit; default `"none"` does not substitute recomputation when host space is
exhausted. GPU work and DMA do not overlap in the modeled policy.

Request rows add copy counts, bytes, timing and KV residency; `result.swap`
reports aggregate host/DMA resources, while `result.pipeline` describes stage
activity and in-flight progress. A cutoff keeps pending ownership instead of
freeing both copies early. Existing simulation, Profile and metric entry points
remain the public interface; no caller needs to manage internal engine queues.

## Load and use a Profile

```python
from mlsys_sim import estimate_profile, load_profile

# Requires the calibration command shown in the README.
profile = load_profile("runs/calibration/profile.json", allow_test_fixture=True)
step = estimate_profile(
    profile, "prefill", [16], [0], allow_test_fixture=True,
)
print(step.duration_ms, step.source)
```

`ProfileModel(experiment)` snapshots the configured profile, matches its full
identity and records hash/version/source/domain provenance. Its `estimate_step`
accepts phase, query-token array, context-length array and optional stage index,
plus keyword-only `request_phases` for mixed queries.
`create_performance_model(experiment)` selects `AnalyticalModel` or `ProfileModel`.
`profile_identity(experiment)` returns the canonical model/device/software/layout
identity needed by external collectors.

Profile timings cover the entire stage including local collectives and
output/sampling. They are not decomposed into fabricated macro costs or added
again to analytical communication. PP boundary P2P remains separate. Missing
workspace observations return unknown; Memory Planner may use its explicitly
labeled analytical workspace assumption. See [calibration](calibration.md).

## Calibrate and hold out configurations

```python
from mlsys_sim import calibrate_dataset, load_dataset, split_dataset, validate_holdout

dataset = load_dataset(
    "examples/v1/step_measurements.fixture.json", allow_test_fixture=True,
)
training, holdout = split_dataset(
    dataset, holdout_groups=["prefill-16"], allow_test_fixture=True,
)
profile = calibrate_dataset(
    training, profile_version="api-holdout-v1", allow_test_fixture=True,
)
report = validate_holdout(profile, holdout, allow_test_fixture=True)
print(report["summary"]["coverage"])
```

`calibrate_dataset` fits exact-shape mean durations and observed workspace
maxima, retaining record/group/shape IDs and a dataset hash. Its optional
`parent_profile` establishes matching identity and version lineage; parent
samples are not merged and parent data is not mutated. Reusing the parent
version is rejected.

`load_communication_dataset` and `calibrate_communication` provide the separate
primitive/rank-specific communication fitting path. Its generated Infra patches
are selected explicitly; they are not automatically added to whole-stage step
timings. See [communication calibration](calibration.md#calibrate-standalone-communication).

`split_dataset` keeps records connected by equal shape **or** group ID together.
It accepts explicit `holdout_groups` or a seeded `holdout_fraction` of independent
components. Both partitions must be nonempty. `validate_holdout` rejects overlap
with profile training records, groups or shapes, then reports coverage,
MAE/MAPE/bias, group summaries and out-of-domain reasons. It cannot infer online
SLA or candidate rankings from step records.

`load_dataset` and `load_profile` validate declared source provenance.
`write_json_exclusive(path, payload)` saves a new JSON artifact without replacing
an existing one. No fitting or writing API changes a test fixture into measured
evidence.

## Compare online runs and candidates

```python
from mlsys_sim import compare_runs

# Requires the profile-serving simulation shown in the README.
comparison = compare_runs(
    "runs/profile-serving",
    "examples/v1/online_observation.fixture.json",
    allow_test_fixture=True,
)
print(comparison["coverage"])
```

`comparison_contract(experiment)` describes the common workload, model/device/
software identity, scheduling/memory policy, SLA, warmup, window/drain and timing
semantics. `compare_runs` requires this contract in the prediction manifest and
an identical contract in a separately sourced observed-run JSON. A simulated
result is not an acceptable observation. `extract_metrics(result)` returns the
supported flat metric map.

`compare_candidates(pairs, metric="goodput_requests_per_s",
higher_is_better=True)` compares candidate rankings under common workload and
evaluation conditions. Each pair has `candidate_id`, `predicted` and `observed`
paths. Reports preserve ties, excluded scores and pairwise ordering agreement.
Use `higher_is_better=False` for latency metrics. Fixture comparisons additionally
require `allow_test_fixture=True`.

## Collect and accept hardware evidence

The version-pinned internal path adds `collect --kind vllm-hook`. It imports
complete Runner/Scheduler JSON exports, validates complete rank groups and
retains raw/quarantine diagnostics. The plan uses `schema_version=1` and a
`captures` array, with optional independent `runtime_trace` path or explicit
`bind_runtime_workload=true` conditional replay. See the
[internal hook contract](qwen3-and-vllm-hooks.md) before enabling it in a Runtime.

```python
from mlsys_sim.vllm_hooks import install_runner_hook, install_scheduler_hook
from mlsys_sim.vllm_capture import load_vllm_captures, normalize_vllm_capture

captures = load_vllm_captures(["rank0.json", "scheduler.json"])
bundle = normalize_vllm_capture(experiment, captures)
```

`install_runner_hook(runner, experiment, *, session_id, ...)` returns a
`RunnerCapture`; `install_scheduler_hook(scheduler, experiment, *, session_id,
...)` returns a `SchedulerCapture`. Both expose `export_capture()`, `close()`
and `flush()`. Worker RPC methods and the opt-in general plugin provide
installation paths without replacing the existing inference scheduler.
`register_vllm_hooks()` is dormant without `MLSYS_VLLM_CAPTURE_CONFIG`;
`uninstall_vllm_hooks_plugin()` restores patches in the current process.

`normalize_vllm_capture(experiment, captures, *, allow_test_fixture=False,
runtime_trace=None, bind_runtime_workload=False)` produces per-replica datasets
only from complete compatible synchronous groups, using maximum per-rank
elapsed service. It emits an online observation only from a complete matching
published lifecycle. Async/raw candidates, missing ranks and unattributed
Workspace remain explicit gaps; one canonical-ready channel does not establish
hardware accuracy or a complete online trace.
Actual PP transport must also match the isolated-stage contract. Since 1.7,
`strategy.pipeline_transport="vllm_native"` observes natural lazy receive,
excludes wait/receiver AllGather from stage service and preserves physical
hidden/residual tensor descriptors through calibration. Native Profiles are
exact-only and require matching complete TP/adjacent-stage payload evidence.
The separate boundary cost uses those bytes/padding and remains explicitly
analytical; raw receive waiting is not fitted as independent link latency.
Sequence parallelism and unproven boundaries remain raw. See
[native PP capture](native-pp-capture.md) for the full contract.

```bash
mlsys-sim collect --kind probe --out runs/rocm-probe
mlsys-sim validate-hardware --plan examples/v1_5/fixture-validation.plan.json \
  --out runs/acceptance-fixture --allow-test-fixture
```

The optional collection paths are `probe`, `steps`, `rccl`, `runtime-trace`,
`vllm`, `vllm-hook` and `reference`. Each keeps its own timing and source contract. Stage
collection needs real HIP availability and synchronized Runtime execution;
fixture backends retain fixture source. RCCL imports preserve explicit timing
units, rank counts and correctness results. Client stream delivery cannot be
reclassified as server Token-ready events or complete stage timings.

The adapter modules provide the corresponding library entry points:

```python
from mlsys_sim.hardware import probe_rocm
from mlsys_sim.rocm_steps import collect_steps
from mlsys_sim.rccl_adapter import collect_rccl, parse_rccl_output
from mlsys_sim.hardware_validation import run_hardware_validation
from mlsys_sim.release_adapter import import_reference_data
from mlsys_sim.runtime_adapter import (
    RuntimeTraceRecorder, collect_vllm, normalize_runtime_trace,
)

inventory = probe_rocm()  # Inspect status and issues; does not install software.
```

`collect_steps(experiment, plan, adapter, inventory, *, torch_module=None,
clock_ns=None, allow_test_fixture=False)` returns a collection bundle containing
the effective experiment, canonical dataset and raw allocator/timing records.
Supplying a test module or clock makes the records fixtures. The Runtime
contract and shape-plan template are described in [ROCm collection](rocm-collection.md).
`collect_rccl(experiment, plan, inventory=None, *, base_dir=None,
allow_test_fixture=False, runner=None)` captures supported AllReduce runs or
preserved logs. A custom runner is a fixture injection. `parse_rccl_output`
validates the supported complete text table and takes explicit rank/mode inputs.
`import_reference_data(path, *, allow_test_fixture=False)` audits a supported
extracted AMD or legacy reference directory and returns an audit plus sanitized
numeric request-service records. It does not execute archived code, extract
nested archives or infer missing canonical timing/model fields. Integrity
discrepancies remain reportable audit failures rather than silent repairs.
`RuntimeTraceRecorder(experiment, *, source, clock_ns=None)` exposes arrival,
token-ready and terminal hooks plus `snapshot()`. Its explicit
`replay_experiment()` selects recorded arrivals and output lengths for a
conditional replay; normalization itself does not alter the input contract.
`normalize_runtime_trace(experiment, raw, *, allow_test_fixture=False)` returns
a canonical observation only for matching semantic server events.
`collect_vllm(experiment, plan, inventory=None, *, base_dir=None,
allow_test_fixture=False, transport=None, clock_ns=None)` records client token-ID
delivery. Its output is not an automatic server-token observation; injected
transport or clock dependencies force fixture provenance.

`validate-hardware` consumes explicit training/holdout, online and threshold
inputs. It writes new Profiles and copied prediction/observation artifacts,
then evaluates coverage, sample/repeat counts and declared error criteria.
Fixture evidence produces `TEST_FIXTURE_ONLY` and keeps
`hardware_validated=False`, independently of whether its numerical criteria
pass. A real-evidence `PASS` applies only to the recorded validation scope.
Individual `calibrate`, `validate` and `compare` commands retain their existing
source and hardware-status behavior. See [collection contracts](rocm-collection.md)
for plans and evidence boundaries.

The library gate is `run_hardware_validation(plan, output_dir, *, base_dir=None,
allow_test_fixture=False)`. Relative input paths resolve against `base_dir`
(normally the plan directory). It preserves `validation-plan.json`, per-replica
fit/holdout artifacts, copied online experiments and observations, predicted
runs and `acceptance.json`. Its current acceptance scope is step latency and
the explicitly selected online metrics; an optional communication fit does not
automatically establish held-out communication or memory-capacity accuracy.

## Interpret simulation results

`simulate` writes no files. The CLI creates immutable run directories with
results, request/event rows and provenance.

- Run `COMPLETED` means the loop finished; inspect completion counts, domain
  coverage and `sla_satisfied` independently. Budget/no-progress termination is
  `UNRESOLVED`; configuration/mechanism errors raise `SimulationError`.
- Request outcomes include `COMPLETED`, `FAILED`, `TIMED_OUT`, `UNFINISHED` and
  `NOT_ADMITTED`. Out-of-window arrivals are not admitted or included in
  completion-rate denominators. Failed/unfinished admitted requests remain in
  those denominators.
- TTFT is first Token minus arrival. Completed requests with at least two
  outputs have TPOT equal to first-to-last Token time divided by `outputs-1`;
  single-output TPOT is `None`. ITL includes recovery gaps.
- Goodput includes completed requests meeting every declared SLA. An inapplicable
  TPOT cannot meet a declared TPOT SLA. Percentiles use linear interpolation at
  `(n-1) × q`; empty samples are `None`.
- `execution.warmup` runs independently and contributes neither measurement rows
  nor residual KV. Only `initial_state="empty_cache"` is supported.
- Arrival windows are `[0, measurement_window_ms)`. `drain=False` stops at the
  window; otherwise admitted work may finish, bounded by `max_drain_ms` when
  specified. Other execution budgets still apply.

Detailed logging is observational. Same-time handling orders completion/release
before timeout, arrival and scheduling. In-flight references remain pinned at
cutoff. Software support, Profile calibration and hardware validation are
separate; see the [validation protocol](validation-protocol.md).
