# MLSys Simulator

A deterministic, CPU-only library for LLM serving experiments. It models request
arrivals, batching, KV memory pressure, preemption and recomputation, with
request timelines, per-rank memory accounting and service metrics.

[中文说明](README.zh-CN.md) · [Python API](docs/api.md) ·
[Calibration](docs/calibration.md) · [Validation protocol](docs/validation-protocol.md) ·
[Specification](docs/specification.md) · [PP and CPU swap](docs/pipeline-and-swap.md) ·
[Prefix Cache and Chunked Prefill](docs/prefix-cache-and-chunking.md) ·
[Contributing](CONTRIBUTING.md)

**Version 1.2 software scope:** serving simulation with Prefix Cache, Chunked
Prefill, online PP and optional CPU KV swap, analytical models, versioned lookup
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

## Run pipeline and CPU swap examples

```bash
mlsys-sim simulate --experiment examples/v1_1/pp_golden.json --out runs/pp-golden
mlsys-sim simulate --experiment examples/v1_1/cpu_swap.json --out runs/cpu-swap
mlsys-sim simulate --experiment examples/v1_1/pp_cpu_swap.json --out runs/pp-cpu-swap
```

PP uses independent stage/microbatch events and waits for sampled Token feedback
before the same request launches its next iteration. CPU swap keeps KV in a
bounded host pool, charges D2H/H2D transfer time and restores it without
recomputation. Both are CPU simulation mechanisms, not real device execution.
The golden PP fixture explicitly disables P2P for the ideal three-microbatch,
two-stage hand calculation. See [PP and CPU swap](docs/pipeline-and-swap.md) for
configuration, buffer budgets, lifecycle and conservative overlap assumptions.

## Run prefix caching and chunking examples

```bash
mlsys-sim simulate --experiment examples/v1_2/prefix_cache.json --out runs/prefix-cache
mlsys-sim simulate --experiment examples/v1_2/chunked_prefill.json --out runs/chunked-prefill
mlsys-sim simulate --experiment examples/v1_2/pp_prefix_chunk.json --out runs/pp-prefix-chunk
```

Prefix reuse requires explicit `prompt_token_ids`; equal prompt lengths never
imply a cache hit. Namespace and the full token ancestry identify immutable KV
blocks. Retained blocks share the fixed GPU pool and unreferenced blocks are
evicted in LRU order. Chunking retains prefix context, limits query work per
iteration and emits the first output only after the final prompt chunk. In
continuous batching, enabling chunking gives Decode priority at each scheduling
boundary; batches remain homogeneous by phase. See the
[configuration and resource contract](docs/prefix-cache-and-chunking.md).

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

The online engine supports FIFO and simplified continuous batching
on one GPU, one synchronous TP group, or a pipeline of TP groups. Dense MHA/GQA
models use explicit sharding and KV replication. PP reserves all-stage KV before
launch and budgets persistent activation buffers. Request preemption either
rebuilds KV or swaps it through a bounded CPU pool, preserving output history and
accounting for recovery latency. PP pressure drains in-flight waves before
reclaiming request KV; DMA stalls new GPU work under a conservative no-overlap
policy.
Continuous batching is prefill-first by default and decode-first with chunking.

The `synthetic` backend uses fixed phase durations; `analytical` uses uncalibrated
compute/bandwidth assumptions. The `profile` backend checks model, device,
software, layout and timing identities, then uses exact shapes or complete
homogeneous interpolation cells. Out-of-domain requests fail unless exploratory
extrapolation is explicitly enabled. Whole-stage profile times already include
local collectives; those costs are not added twice.

Warmup executes on an isolated timeline and leaves measurement with empty KV.
An arrival window is half-open; out-of-window requests are not admitted. Drain
behavior and budgets are explicit. See the [specification](docs/specification.md).

`simulate` models online PP stage queues, P2P transfers and feedback dependencies.
`predict` remains shape-only analysis and never supplies online SLA metrics.
Multiple replicas, NVMe swap, mixed-phase kernels and quantized storage
formats remain outside this version. Allocator fragmentation, kernel cycles and
live CUDA/HIP or host DMA execution are not simulated.
The [capability matrix](docs/coverage-matrix.md) separates software support from
hardware evidence. A calibrated profile does not itself establish held-out
hardware accuracy.

## Outputs and development

Simulation output includes `manifest.json`, `result.json`, `requests.jsonl`,
optional `events.jsonl`, and `logs.txt`. Pipeline runs add per-stage activity and
progress diagnostics; swap-enabled runs add host/DMA resources and per-request
copy counters, byte counts and timing. Prefix/chunk runs expose cache occupancy,
reuse and actual prompt work. Manifests retain effective configuration,
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
