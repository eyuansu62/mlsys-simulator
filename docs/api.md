# Python API

Install distribution `mlsys-simulator` and import `mlsys_sim`. Version 1.1 exposes the
simulation, profile, calibration and comparison functions through this package;
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
The simulator does not execute GPU tensors or sleep in wall-clock time.

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
accepts phase, query-token array, context-length array and optional stage index.
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
