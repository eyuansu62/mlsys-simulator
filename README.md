# MLSys Simulator

A deterministic, CPU-only simulator for learning and experimenting with LLM
serving systems. It models request arrivals, batching, KV memory pressure,
preemption and recomputation, and reports request timelines and service metrics.

[中文说明](README.zh-CN.md) · [Python API](docs/api.md) ·
[Capability matrix](docs/coverage-matrix.md) · [Contributing](CONTRIBUTING.md)

**Status:** initial development version. The license is pending selection; no
open-source license has been granted and this project has not been published to
PyPI. See the [release checklist](docs/releasing.md).

## Install from a checkout

Requires Python 3.11 or newer. The core library has no third-party runtime
dependencies.

```bash
python -m pip install .
```

For development, install the package in editable mode and add the development
tools:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

JSON and TOML work with the standard library. YAML is supported only when
PyYAML is already installed; it is not a required dependency.

## Run an experiment

From the checkout:

```bash
mlsys-sim simulate --experiment examples/fifo.json --out runs/fifo
mlsys-sim simulate --experiment examples/memory_pressure.json --out runs/memory
mlsys-sim simulate --experiment examples/tp2.json --out runs/tp2
mlsys-sim predict --experiment examples/pp2_analytical.json --out runs/pp2
```

The equivalent module command is `python -m mlsys_sim simulate ...`.
The original `simulate.py` and `predict.py` source-checkout entry points remain
available. Each output directory must be new; an existing directory returns
`OUTPUT_EXISTS` instead of overwriting results.

| Example | Demonstrates |
| --- | --- |
| `fifo.json` | A hand-checkable three-request timeline |
| `memory_pressure.json` | A small KV pool forcing request preemption and recomputation |
| `tp2.json` | Synchronous two-rank TP with explicit memory and communication costs |
| `pp2_analytical.json` | Uneven PP stages, capacity and isolated forward-time analysis |

## Use the Python library

```python
from mlsys_sim import CalibrationSpec, Experiment, RequestSpec, simulate

experiment = Experiment(
    workload=(
        RequestSpec(
            request_id="request-1",
            arrival_ms=0,
            input_tokens=16,
            output_tokens=4,
        ),
    ),
    calibration=CalibrationSpec(backend="synthetic"),
)

run = simulate(experiment)
print(run.result["counts"])
print(run.requests[0]["ttft_ms"])
```

Use `load_experiment(path)` for a file or `experiment_from_dict(mapping)` for
strict configuration parsing. `predict_experiment(experiment)` returns shape,
memory and communication analysis without simulating online queueing.

## Model scope

The online engine supports FIFO and a simplified prefill-first continuous
batching policy on one GPU or one synchronous TP group. Dense MHA/GQA models use
an explicit shard plan, with KV replication when required. Every step checks
physical memory and logical KV blocks before committing reservations across all
ranks. A preempted request retains its output history and rebuilds its KV before
continuing; waiting and recomputation remain part of its latency.

The fixed KV pool stays physically reserved when logical blocks are reclaimed.
Workspace is modeled by execution shape. The implementation does not simulate
allocator fragmentation, kernel cycles or live CUDA/HIP execution.

PP is **analytical only**. Online PP, multiple replicas, CPU swap, prefix caching,
chunked prefill and quantized storage formats are not implemented and are
explicitly rejected. See the [capability matrix](docs/coverage-matrix.md).

The `synthetic` backend uses fixed phase times for correctness tests.
The `analytical` backend provides uncalibrated estimates. Built-in hardware
parameters are teaching fixtures, and neither backend is validated against AMD
hardware. Performance results must be interpreted with those assumptions.

## Results

CLI runs produce `manifest.json`, `result.json`, `requests.jsonl`, optional
`events.jsonl`, and `logs.txt`. They include expanded configuration and code
hashes, provenance, request outcomes, TTFT/TPOT/ITL, throughput, goodput,
preemption, recomputation and per-rank memory accounting.

CLI exit codes are `0` for a completed run, `2` for an error and `3` for an
unresolved run. A completed run can still contain failed requests or miss its
SLA. The Python API returns an in-memory `SimulationResult`; file writing belongs
to the CLI. See [API and metric semantics](docs/api.md).

## Development

```bash
python -m unittest discover -s tests -v
ruff check .
mypy --check-untyped-defs --ignore-missing-imports mlsys_sim
python -m build
twine check dist/*
```

Tests cover formulas, deterministic timelines, memory conservation, atomic TP
reservations, timeout lifetimes, recomputation and bounded generated workloads.
CI configuration is included; remote CI results are available only after a
repository is created and the workflow runs.

The next implementation work is measured performance profiles and held-out
validation, followed by PP stage/microbatch events. The design basis is the
[simulator development plan](https://gist.github.com/eyuansu62/f91df8b7597801ca4fd91bb3bb8c98cc).
