# Python API

The distribution name is `mlsys-simulator`; import the library as `mlsys_sim`.
The initial development API exposes the existing model and engine without a
second simulation implementation. API changes before version 1.0 will be
recorded in the changelog.

## Create and simulate an experiment

```python
from mlsys_sim import CalibrationSpec, Experiment, RequestSpec, simulate

experiment = Experiment(
    workload=(
        RequestSpec("short", arrival_ms=0, input_tokens=16, output_tokens=4),
        RequestSpec("long", arrival_ms=2, input_tokens=32, output_tokens=8),
    ),
    calibration=CalibrationSpec(backend="synthetic"),
)
run = simulate(experiment)

assert run.result["status"] == "COMPLETED"
print(run.result["counts"])
```

`Experiment` and its nested specifications are frozen dataclasses. Use
`dataclasses.replace` to derive a changed configuration. Durations use
milliseconds, storage uses bytes, memory/link bandwidth uses decimal GB/s and
compute uses TFLOP/s. The core does not access a GPU or sleep in wall-clock time.

| Export | Purpose |
| --- | --- |
| `Experiment` | Six-part experiment configuration and schema version |
| `RequestSpec` | Arrival, prompt/output lengths, priority and optional timeout |
| `ModelSpec`, `DeviceSpec`, `InfraSpec` | Dense model shape and device/link assumptions |
| `StrategySpec`, `MemoryConfig` | Scheduling, TP/PP layout and memory budgets |
| `CalibrationSpec` | Analytical coefficients or synthetic phase durations |
| `EvaluationSpec`, `ExecutionSpec` | SLA/window rules, deterministic seed and run budgets |
| `SimulationResult` | System result, request rows and event rows |
| `SimulationError` | Configuration/mechanism error with a machine-readable `code` |

All direct API experiments are validated. Output lengths from the replay input
are used only by the execution oracle to stop generation, not by scheduling or
victim selection.

## Load strict configuration

```python
from mlsys_sim import experiment_from_dict, load_experiment, simulate

experiment = load_experiment("examples/fifo.json")
run = simulate(experiment)

small = experiment_from_dict({
    "workload": [{
        "request_id": "r1", "arrival_ms": 0,
        "input_tokens": 16, "output_tokens": 2,
    }],
    "calibration": {"backend": "synthetic"},
})
```

The parser supports JSON and TOML, plus JSONL request lists and optional YAML
when PyYAML is installed. Six top-level sections can reference files relative to
the experiment file. A model reference inside a referenced strategy resolves
relative to that strategy file. Unknown fields, invalid types, non-finite
values and unsupported mechanisms fail explicitly.

## Predict a shape

```python
from mlsys_sim import load_experiment, predict_experiment

report = predict_experiment(load_experiment("examples/pp2_analytical.json"))
print(report["stages"])
print(report["prefill_memory_feasible"])
```

`predict_experiment` treats workload rows as one shape batch and ignores their
arrival times. It reports per-stage service costs, TP/P2P communication and
per-rank capacity. It does not provide queueing, online PP latency percentiles
or online goodput. Corresponding `online_metrics` entries are `None` with a
reason. Batch-limit and memory-feasibility fields must be checked separately
from the report's execution status.

`AnalyticalModel` and `MemoryPlanner` are available for experiments requiring
step estimates or explicit reservations. Their implementation contracts are
narrow: homogeneous phase batches, the supported Dense GQA shard plan and a
fixed KV pool. Callers managing reservations are responsible for safe release
boundaries; the `simulate` entry point manages that lifecycle itself.

## Interpret results

`simulate` returns `SimulationResult(result, requests, events)` without writing
files. The CLI writes those objects and provenance into a new run directory.

- `result["status"]` is `COMPLETED` when the simulation loop finishes, or
  `UNRESOLVED` when a budget/no-progress condition stops it. Inspect request
  counts and `sla_satisfied` independently.
- Request status is `COMPLETED`, `FAILED`, `TIMED_OUT` or `UNFINISHED` at return.
  Configuration and unsupported-mechanism errors raise `SimulationError`.
- `TTFT = first_token_ms - arrival_ms`. Completed requests with at least two
  outputs have `TPOT = (last_token_ms - first_token_ms)/(outputs - 1)`.
  Single-output TPOT is `None`.
- ITL includes waiting and recomputation gaps. A resumed request keeps its
  original first-token timestamp and does not emit historical outputs again.
- Completion-rate denominators retain failed and unfinished requests.
  Goodput counts completed requests satisfying all declared SLAs; a request
  with no applicable TPOT cannot satisfy a declared TPOT SLA.
- A declared measurement window selects requests by arrival in `[0, window)`
  and allows their completion during drain. Without one, rate denominators use
  elapsed simulated time from zero. Window/cohort fields state the rule used.
- Percentiles use linear interpolation at `(n - 1) × q`; empty samples return
  `None`. TTFT/TPOT percentiles use completed requests; ITL retains observed
  intervals from the selected cohort, including failed requests.

Detailed event logging is observational: disabling it does not alter scheduling
or metrics. Same-time processing orders completion/release before timeout,
arrival and new scheduling. Resources referenced by in-flight work remain
pinned until its safe completion boundary.

## Calibration boundary

`synthetic` fixes each phase's Batch duration for hand-checkable tests.
`analytical` uses explicit but uncalibrated compute, bandwidth and communication
assumptions. Neither backend is a measured hardware profile. Actual tensor
execution, allocator fragmentation, CPU swap and online PP remain outside this
version's scope.
