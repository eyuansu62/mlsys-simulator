# Python API

Install distribution `mlsys-simulator` and import `mlsys_sim`. Version 1.19 exposes the
simulation, profile, calibration and comparison functions through this package,
with optional hardware collection and evidence-acceptance commands;
`mlsys_sim.__version__` is the software version. JSON schema and Profile version
labels are separate contracts.

The optional `mlsys-sim-web` / `python -m mlsys_sim.web` entry serves the local
single-GPU analytical form and charts. It reuses the core engine and has no GPU,
framework or CDN dependency. See the [Web interface](web-interface.md).

## Independent finite closed-loop studies

`generate_request_stream(spec, replication=0)` freezes request IDs and categorical
input lengths without inventing arrival times. `predict_closed_loop(experiment,
spec)` releases replacement clients after simulated terminal feedback and ends
after exactly the declared request cohort plus drain. It does not change the
duration-limited `run_load_sweep` contract.

`collect_closed_loop(backend, spec, concurrency, replication=0)` drives an explicit
synchronous Runtime backend. Its timestamps are **Runtime API feedback**, not
server-token readiness. The optional `python -m mlsys_sim.vllm_sla_worker` entry
uses the pinned ROCm/vLLM Runtime and a separate Scheduler hook to obtain actual
server-token events. It imports GPU dependencies only when explicitly invoked.

`validate_closed_loop(predictions, observations, contract)` pairs all declared
cells by frozen workload policy and request stream, permits their endogenous
arrivals to differ, and preserves acquisition, prediction-error and SLA results.
It never refits or replays measured arrivals into the predictor. A joint SLA
failure at a deliberately overloaded point is distinct from a capture failure
or an inaccurate prediction.

`mlsys_sim.load_validation_report.write_closed_loop_study_report(study, out)`
writes sanitized HTML/Markdown, CSV, JSON and SVG from one study snapshot.
See [closed-loop validation](closed-loop-validation.md) for the protocol.

## Opt-in course workflows

`run_chapter1(plan)` and `write_chapter1_report(report, new_directory)` provide the
explicit Chapter 1 counting protocol and report. `generate_workload(spec,
replication=0)` returns canonical request dictionaries and seeded provenance.
`run_load_sweep(experiment, plan)` and `write_load_report` retain every rate/client
point and repetition. `simulate_dag(plan)` runs one frozen dependency graph;
`run_dag_candidates(plan)` compares placements without changing its nodes.
`write_dag_report` writes candidate or single-run artifacts and Profile replay
inputs. Plans can be supplied to `chapter1`, `generate-workload`, `sweep` and
`dag` CLI commands. These new plan schemas do not alter legacy `Experiment`
inputs. See [course workflows](course-workflows.md) for schemas, examples,
empirical boundary definitions and source limitations.

`evaluate_simulation_constraints(run, spec)` and
`evaluate_dag_constraints(result, spec, pool_specs=...)` assess joint per-rank
memory, latency and explicitly timed throughput without rerunning the engine.
`run_candidate_screening(plan)` and `write_screening_report` execute a fixed
workload over candidate configurations, retaining failed/unknown candidates.
`run_load_sweep` accepts an optional `constraints` block; old plans preserve
their existing output contract.

`import_course_calibration(raw_bytes, source=...)` reads the course's primitive
fit, `write_course_calibration_report` saves its evidence and guarded patch,
and `run_calibrated_chapter1(plan, calibration)` builds the separate calibrated
grid. `apply_course_calibration(experiment, calibration, allow_test_fixture=False)`
applies the patch explicitly to analytical experiments and rejects
Profile/synthetic backends. Applying imported fixture coefficients requires
the helper's explicit `allow_test_fixture=True`; this preserves the existing
`CalibrationSpec.allow_test_fixture` contract.
See [joint constraints and course calibration](course-constraints-and-calibration.md).

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

Storage byte widths (`weight_bytes`, `kv_bytes`, `activation_bytes`) accept any
positive finite value up to 8; quantized formats are declared by their
effective bytes per element plus `weight_dequant` (`none`, `fused`,
`materialized`) and `calibration.dequant_flops_per_weight`. Dtype labels must
agree with the bytes. No quantization accuracy is modeled; see
[quantized storage](quantization.md).

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

`validate_profile_repeats(profile, training_dataset, repeat_dataset, *,
allow_test_fixture=False)` evaluates a frozen Profile at retained **exact**
shapes from an independent acquisition. It checks the canonical training SHA,
recomputes the fit to verify its samples and provenance, and rejects reused
sessions, raw capture hashes, record IDs or group IDs. Each dataset needs
`notes.acquisition` with explicit `session_id`, `clock_domain` and
`capture_input_sha256`; the vLLM normalizer derives these from its actual capture
inputs. Identity and native PP payload descriptors must match. Missing exact
shapes count as rejected coverage; interpolation and extrapolation are disabled.
The report labels within-domain repeats and retains `hardware_validated=false`;
acquisition metadata alone does not prove a fresh Runtime. Ordinary
`validate_holdout` and hardware-acceptance shape-leakage checks are unchanged.

```bash
mlsys-sim validate --exact-repeats --profile runs/fit/profile.json \
  --training-dataset runs/fit/dataset.json --dataset independent-repeat.json \
  --out runs/repeat-validation
```

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
The separate boundary cost uses those bytes/padding and is analytical unless
a native communication Profile is explicitly selected; raw receive waiting is
not fitted as independent link latency.
Sequence parallelism and unproven boundaries remain raw. See
[native PP capture](native-pp-capture.md) for the full contract.

```bash
mlsys-sim collect --kind probe --out runs/rocm-probe
mlsys-sim validate-hardware --plan examples/v1_5/fixture-validation.plan.json \
  --out runs/acceptance-fixture --allow-test-fixture
```

The optional collection paths are `probe`, `steps`, `rccl`, `runtime-trace`,
`vllm`, `vllm-hook`, `pp-communication`, `workspace`, `memory-capacity` and
`reference`. Each keeps its own timing and source contract. Stage
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

## Independent native PP communication API and CLI

`collect --kind pp-communication` runs or imports the independent native
primitive protocol. `calibrate --native-pp-communication` fits its distinct
dataset contract, and `validate --native-pp-communication --profile ...` checks
an existing Profile against an independent dataset. The validation command
does not fit a new Profile or automatically split its input.

Calibration accepts `--native-pp-model {affine,size-lookup}` only together with
`--native-pp-communication`. The default remains `affine`, preserving existing
v1 datasets and Profiles. For example:

```bash
mlsys-sim calibrate --native-pp-communication --native-pp-model size-lookup \
  --dataset runs/native-communication/dataset.json \
  --profile-version native-lookup-v1 --out runs/native-lookup-fit
```

Collection defaults to `native_pp_primitives_v1`. A plan can explicitly select
`"protocol": "native_pp_primitives_v2"` to omit world `device_id` initialization
and use lazy world/pair-group creation. V2 records the fixed
`communicator_initialization="lazy_world_and_pair_groups_v1"` and a launch
contract with `world_init_device_id=false`; these describe the initialization
API, not an observed RCCL algorithm. The timer, real P2P warmup and correctness
checks stay the same. Selected inherited flags, including BLOCKING_WAIT aliases
and `NCCL_P2P_DISABLE`, are recorded alongside the full communication-environment
SHA256. The collector does not select BLOCKING_WAIT automatically. Keep v1 and
v2 captures, parent lineage and final holdout in separate matching domains;
raw v2 imports require the matching envelope, rank protocols and launch contract.

The optional `CalibrationSpec` fields are
`native_pp_communication_profile_path`,
`replica_native_pp_communication_profile_paths`,
`native_pp_communication_profile_version` and
`native_pp_communication_allow_extrapolation`. Fixture use still requires
`allow_test_fixture=True`. A selected Profile requires native transport with
`pipeline_p2p=True`; it affects only boundary service, not stage identity or
model-local collectives. See the [communication contract](native-pp-communication.md)
for byte conventions, placement/stack checks and fitting/holdout semantics.

Library entry points live in `mlsys_sim.pp_benchmark` and
`mlsys_sim.pp_communication`:

```python
from mlsys_sim.pp_benchmark import collect_pp_communication
from mlsys_sim.pp_communication import (
    load_pp_communication_dataset, load_pp_communication_profile,
    calibrate_pp_communication, validate_pp_communication_holdout,
    PPCommunicationModel,
)
```

Dataset/Profile loaders validate source and schema, returning dictionaries.
`calibrate_pp_communication(dataset, profile_version=..., method=...)` fits a new
immutable Profile. Its default `method="nonnegative_affine_ols_v1"` retains the
affine model; `method="piecewise_size_lookup_v1"` explicitly selects a lookup.
The lookup requires at least two sizes with five samples per size at each
primitive placement. Knots retain every sample and predict their arithmetic
mean, alongside counts, quantiles and population standard deviation. Slow
samples remain in both the Profile and error calculations.

Lookup interpolates linearly in bytes only between adjacent training knots
passing fixed guards: at most a 4× size gap, and no qualifying latency jump
of at least 0.05 ms with a ratio over 2× (an upward ratio is adjusted for byte
growth). Exact knots remain usable beside blocked intervals. Internal holes and
all extrapolation fail with `OUT_OF_COMMUNICATION_CALIBRATION_DOMAIN`, even if
`native_pp_communication_allow_extrapolation=True`; that option continues to
apply to affine Profiles. The [communication contract](native-pp-communication.md#choose-an-explicit-size-lookup-for-discontinuous-service)
defines the exact guard formulas. Method and acquisition protocol are separate
explicit choices.

`validate_pp_communication_holdout(profile, dataset)` evaluates the frozen
Profile against fresh observations without refitting. Protocol/initialization,
identity, stack/environment and record/group leakage checks remain required.
Reports retain sample errors, `by_message_size` mean errors,
`mean_target_metrics`, exact/unseen size summaries and small-message coverage.
Rejected rows retain a reason and null error; they reduce coverage rather than
becoming zero-error predictions. A size's mean error requires complete coverage
at that size. Freeze the model and guards before the final independent holdout;
new training coverage needs a new version and final holdout.

Fixture inputs require `allow_test_fixture=True`. The collector's
optional `runner` injection is a test interface and cannot promote CPU results
to measured evidence. `PPCommunicationModel` loads the selected local Profile
and exposes placement/size estimates with provenance. Lookup estimates also
identify exact/interpolated results and their contributing knots. These APIs do
not infer a hardware cause for latency transitions or establish serving accuracy.

## Workspace and memory-capacity workflows

`collect --kind workspace --adapter module:factory` uses a Runtime-owned scratch
ledger and step sample plan. `calibrate --workspace` produces an independent
exact per-rank Workspace Profile without duration. Optional `CalibrationSpec`
fields are `workspace_profile_path`, `replica_workspace_profile_paths` and
`workspace_profile_version`; fixtures still require `allow_test_fixture=True`.
Explicit selection replaces dynamic bytes and rejects missing shapes instead
of using a fallback. Stage timing and fixed allocations retain their identities.

The `mlsys_sim.workspace` module provides `load_workspace_dataset`,
`load_workspace_profile`, `validate_workspace_dataset`,
`validate_workspace_profile`, `calibrate_workspace`, `validate_workspace_repeats`
and `WorkspaceModel`. `validate_workspace_repeats` is also a package-level export.
Calibration reduces repeats by per-rank maximum. Each canonical record must
contain complete exclusive Runtime ownership, including non-Torch scratch; a
passive manager observer supplies partial raw evidence instead.

`validate_workspace_repeats(profile, training_dataset, repeat_dataset, *,
allow_test_fixture=False)` evaluates a frozen Profile using a separate
acquisition. It checks the exact training canonical SHA and recomputes its fit,
then requires distinct acquisition sessions, capture hashes, record IDs and
group IDs. Training acquisition facts are copied into the fitted Profile only
when explicitly present; legacy fitting without them retains its old output,
but cannot supply the facts required for this repeat evaluator.

The evaluator accepts only retained exact phase/stage/query/context/member-role
shapes. It preserves every TP-rank value, reports record and rank coverage,
retained-shape coverage, byte errors and underprediction, and treats an observed
zero as a valid value. Missing TP ranks fail dataset validation; missing shapes
remain out of domain. `summary.underprediction_count` counts underestimated
rank cells; `underpredicted_record_count` counts affected records. It does not
interpolate, extrapolate, fit on repeat rows,
prove process freshness from a session label, or assert hardware acceptance.
`hardware_validated=false`, `unseen_shape_generalization=false` and no online SLA
remain explicit. The corresponding CLI is:

```bash
mlsys-sim validate --workspace-repeats \
  --training-dataset runs/workspace-fit/dataset.json \
  --profile runs/workspace-fit/profile.json \
  --dataset workspace-repeat.json --out runs/workspace-repeat-validation
```

Use the fit's canonical `dataset.json`, rather than a differently normalized
input file. The output snapshots training, repeat and Profile inputs and their
hashes. Split and extrapolation options are not applicable.

`collect --kind memory-capacity` runs or imports a fresh-Runtime axis sweep,
and `validate --memory-capacity --dataset ...` assesses complete configuration
evidence and success-to-typed-OOM brackets. This validation needs no fitted
Profile argument. Request count denotes a cohort, not actual batch size;
non-OOM failures and prediction-domain gaps remain separate. See
[Workspace and capacity](workspace-and-capacity.md) for adapter, ledger, fixture
and interpretation contracts.

The optional pinned live adapter
`mlsys_sim.vllm_capacity_adapter:kv_pool_factory` verifies a real single-GPU
Qwen3 MRv1 KV-pool allocation and executes the declared request. Set
`MLSYS_VLLM_CAPACITY_MODEL_PATH` to an existing local checkpoint; it downloads
nothing and imports Torch/vLLM only when called. Its narrow eager TP1×PP1
configuration excludes cache, chunking, mixed batches, offload and swap. The
requested byte axis must match unique actual KV storage. A failed final native
`torch.zeros` allocation is eligible only with the exact applied-axis and typed
allocator OOM evidence; other failures remain censored or distinct. The existing
`torch.empty` evidence path remains supported. See the
[live capacity limits](capacity-live-validation.md).

Optional source tools in `scripts/rocm_workspace_audit` join installed Torch
allocator, native SDK backing and Runtime-owner evidence. The legacy single-card
prefill driver remains separate from the new `workspace_stage_session.py`
supervisor and `worker_workspace_capture.WorkspaceWorker`. The latter uses the
pinned Worker's real execute/sample boundaries for single/TP2/PP2 and records
per-rank prefill and decode iterations. `assemble_stage_workspace.py` independently
audits source/Runtime/rank, Torch, native and PP transport evidence before
emitting a canonical dataset. A raw process exit or component success is not
complete attribution.

The optional PP2 `--host-signal-provider`, `--host-signal-rocr` and
`--host-signal-abi-header` arguments enable public HSA backing observations at
the existing synchronized CPU boundary. Hardware events additionally require
`--host-signal-namespace-control` and independent passive KFD evidence. Source,
ABI, actual libraries, generations and raw lifecycle prefixes are checked before
admission; CPU/GTT observations do not replace GPU backing or budget checks.
See the [synchronization evidence contract](pp2-synchronization-evidence.md).

These sources are outside the normal package import path, require existing
matching native tools, and add no required pip dependency. Their current raw
domain is Qwen3-1.7B BF16, synchronous eager MRv1, seq1/512-token limits, fixed
256MiB KV and no cache/chunk/mixed/swap. Memory-capture stage durations are
diagnostic and must not become new latency calibration rows. See the
[Worker workflow](workspace-and-capacity.md#collect-worker-level-prefill-and-decode),
[live Workspace domain](workspace-live-validation.md) and tool README for
separate software and actual evidence limits.

`mlsys_sim.workspace_capture` provides `WorkspaceRecorder`, `collect_workspace`
and `PinnedVllmWorkspaceObserver`. The recorder receives explicit storage
leases/releases and rank completion; it does not infer ownership from a process
peak. Complete adapters expose `begin_workspace_capture(sample, recorder)` and
`end_workspace_capture(sample, recorder)` as well as the stage adapter methods.
The pinned manager observer stays partial `managed_scratch_capacity`, without
canonical Workspace data. Live Workspace collection has no raw-ledger importer
or CPU fallback; injected backends are a fixture-only library test interface.

### Explicit P2P threshold policy (1.11)

A PP collection plan may select `protocol="native_pp_primitives_v3"`. It requests
`NCCL_P2P_LL_THRESHOLD=0` and `NCCL_P2P_LL128_THRESHOLD=0` only in workers;
conflicting explicit caller values fail. Complete per-rank evidence binds the
`simple_only_thresholds_v1` policy, both actual string values, lazy initialization
and unchanged synchronized host completion timing. v1/v2 remain supported.
Profile parent/holdout validation rejects cross-domain reuse. vLLM Hook Runtime
fingerprints additionally bind explicitly set thresholds and SHM copy policies
without changing old unset fingerprints. See the
[policy and serving constraints](native-pp-communication.md#select-the-explicit-p2p-simple-only-policy).

Holdout `by_message_size[].observed_distribution_ms` describes every observation
at that placement/size with `count`, `mean`, `min`, `max`, `p10`, `p50`, `p90`,
`p99` and population `stddev`. It includes rejected and slow observations; it
does not replace the existing errors, coverage or mean prediction target.
