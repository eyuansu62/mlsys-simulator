# MLSys Simulator

A deterministic, CPU-only library for LLM serving experiments. It models request
arrivals, batching, KV memory pressure, preemption and recomputation, with
request timelines, per-rank memory accounting and service metrics.

[中文说明](README.zh-CN.md) · [Python API](docs/api.md) ·
[Calibration](docs/calibration.md) · [Validation protocol](docs/validation-protocol.md) ·
[Specification](docs/specification.md) · [Contributing](CONTRIBUTING.md)

**V1 software scope:** serving simulation, analytical models, versioned lookup
profiles, calibration, grouped holdout evaluation and run/candidate comparison.
No AMD measurements have been supplied or hardware accuracy established. The
repository is [private](https://github.com/eyuansu62/mlsys-simulator), the license
is pending selection, and no package has been published to PyPI.

## Install from a checkout

Requires Python 3.11 or newer. The core has no third-party runtime dependencies.

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

JSON and TOML use the standard library. YAML works when PyYAML is already
installed; it is not a required dependency.

## Run the serving examples

From the checkout:

```bash
mlsys-sim simulate --experiment examples/fifo.json --out runs/fifo
mlsys-sim simulate --experiment examples/memory_pressure.json --out runs/memory
mlsys-sim simulate --experiment examples/tp2.json --out runs/tp2
mlsys-sim predict --experiment examples/pp2_analytical.json --out runs/pp2
```

Use `python -m mlsys_sim` in place of `mlsys-sim` if preferred. The original
`simulate.py` and `predict.py` source entry points remain available. Every output
directory must be new; existing results are never overwritten.

## Exercise the calibration loop

The supplied V1 records are **hand-calculated test fixtures, not measurements**.
They require an explicit fixture opt-in, and retain that provenance in profiles
and reports. These commands demonstrate the software workflow:

```bash
mlsys-sim calibrate --dataset examples/v1/step_measurements.fixture.json \
  --profile-version fixture-v1 --out runs/calibration --allow-test-fixture
mlsys-sim simulate --experiment examples/v1/profile_serving.json \
  --out runs/profile-serving
mlsys-sim compare --predicted runs/profile-serving \
  --observed examples/v1/online_observation.fixture.json \
  --out runs/compare --allow-test-fixture
mlsys-sim validate --dataset examples/v1/step_measurements.fixture.json \
  --profile-version holdout-v1 --holdout-group prefill-16 \
  --out runs/holdout --allow-test-fixture
```

`profile_serving.json` references the first command's `profile.json` and opts into
fixture consumption. Step calibration fits repeated-shape means; validation
holds out complete configuration groups. Online comparison uses a separate
observed-run contract. See [calibration](docs/calibration.md) and
[validation](docs/validation-protocol.md) before supplying real records.

## Use the Python API

```python
from mlsys_sim import CalibrationSpec, Experiment, RequestSpec, simulate

experiment = Experiment(
    workload=(RequestSpec("r1", arrival_ms=0, input_tokens=16, output_tokens=4),),
    calibration=CalibrationSpec(backend="synthetic"),
)
run = simulate(experiment)
print(run.result["counts"])
```

Use `load_experiment(path)` or `experiment_from_dict(mapping)` for strict
configuration. `predict_experiment(experiment)` analyzes a shape without online
queueing. Profile and validation APIs are described in [docs/api.md](docs/api.md).

## Model and evidence boundaries

The online engine supports FIFO and simplified prefill-first continuous batching
on one GPU or one synchronous TP group. Dense MHA/GQA models use explicit
sharding and KV replication. Physical budgets and logical KV blocks are checked
before an all-rank reservation commits. Request preemption rebuilds KV while
preserving output history and accounting for recovery latency.

The `synthetic` backend uses fixed phase durations; `analytical` uses uncalibrated
compute/bandwidth assumptions. The `profile` backend checks model, device,
software, layout and timing identities, then uses exact shapes or complete
homogeneous interpolation cells. Out-of-domain requests fail unless exploratory
extrapolation is explicitly enabled. Whole-stage profile times already include
local collectives; those costs are not added twice.

Warmup executes on an isolated timeline and leaves measurement with empty KV.
An arrival window is half-open; out-of-window requests are not admitted. Drain
behavior and budgets are explicit. See the [specification](docs/specification.md).

PP is **analytical only**. Online PP, multiple replicas, CPU swap, prefix caching,
chunked prefill and quantized storage formats remain outside V1. Allocator
fragmentation, kernel cycles and live CUDA/HIP execution are not simulated.
The [capability matrix](docs/coverage-matrix.md) separates software support from
hardware evidence. A calibrated profile does not itself establish held-out
hardware accuracy.

## Outputs and development

Simulation output includes `manifest.json`, `result.json`, `requests.jsonl`,
optional `events.jsonl`, and `logs.txt`. Manifests retain effective configuration,
code/input hashes, comparison contracts and profile provenance. Calibration and
validation preserve dataset lineage and fit/holdout membership. A completed
command can still report failed requests, incomplete domain coverage or missed
SLAs; inspect the report fields.

```bash
python -m unittest discover -s tests -v
ruff check .
mypy --check-untyped-defs --ignore-missing-imports mlsys_sim
python -m build
twine check dist/*
```

CI configuration is included. Refer to the validation report and actual workflow
runs for executed checks; this README does not imply remote CI or hardware
validation passed. License selection and any PyPI publication remain separate
steps in the [release checklist](docs/releasing.md).
