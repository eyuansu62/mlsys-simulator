# MLSys Simulator

A deterministic CPU simulator for LLM serving experiments. It models request
arrivals, batching, KV memory pressure, preemption and recomputation, with
request timelines, per-rank memory accounting and service metrics. Optional
collection adapters instrument an installed Runtime separately from simulation.

[中文说明](README.zh-CN.md) · [Python API](docs/api.md) ·
[Calibration](docs/calibration.md) · [Validation protocol](docs/validation-protocol.md) ·
[Specification](docs/specification.md) · [PP and CPU swap](docs/pipeline-and-swap.md) ·
[Prefix Cache and Chunked Prefill](docs/prefix-cache-and-chunking.md) ·
[Mixed batching](docs/mixed-batching.md) · [Multi-replica routing](docs/multi-replica.md) ·
[ROCm collection](docs/rocm-collection.md) · [Qwen3 and vLLM hooks](docs/qwen3-and-vllm-hooks.md) ·
[Native PP capture and Profiles](docs/native-pp-capture.md) ·
[Native PP communication calibration](docs/native-pp-communication.md) ·
[Quantized storage](docs/quantization.md) ·
[Course workflows](docs/course-workflows.md) ·
[Joint constraints and course calibration](docs/course-constraints-and-calibration.md) ·
[Independent closed-loop validation](docs/closed-loop-validation.md) ·
[Workspace and capacity validation](docs/workspace-and-capacity.md) · [Contributing](CONTRIBUTING.md)

**Version 1.18 software scope:** Frozen finite-cohort closed-loop predictions,
completion-feedback collection, independent server-token validation and
sanitized course reports, alongside joint per-rank memory, latency and throughput
constraints, candidate screening, imported course eta/beta reports and explicit
calibrated Chapter 1 grids, alongside nominal reporting,
seeded open/closed-loop load generation and scans, causal DAG coordination over
heterogeneous model pools, public HSA/driver GTT synchronization-resource
evidence and Worker-level Prefill/Decode Workspace acquisition
for the pinned single/TP2/PP2 layouts, provenance-checked independent exact
Workspace repeats, independent exact per-rank Workspace Profiles,
Runtime-owned scratch attribution and fresh-Runtime capacity/OOM-bracket
validation; independent native PP P2P/AllGather microbenchmarks and
placement-specific communication Profiles, native vLLM PP
lazy-receive capture and exact stage Profiles, Qwen3 geometry/import and
version-checked internal capture, with optional AMD/ROCm collection adapters, explicit hardware
evidence acceptance, multi-replica routing and load balancing with
mixed Prefill/Decode batches, Prefix Cache, Chunked Prefill, online TP/PP and
optional CPU KV swap, declared-width quantized weight/KV storage
(FP8/INT8/INT4/FP4 with explicit dequant work, no accuracy modeling), analytical models, versioned lookup
profiles, calibration, grouped holdout evaluation and run/candidate comparison.
User-supplied AMD and NVIDIA reference packages inform the import and audit
contracts; raw packages and prompt/output text are not distributed with this
library. Their coverage does not establish general hardware accuracy. The
source is on [GitHub](https://github.com/eyuansu62/mlsys-simulator), the license
is pending selection, and no package has been published to PyPI.

For the course, start with the [six runnable workflows](docs/course-workflows.md#run-the-supplied-experiments).
Reports contain HTML/CSV/SVG and explicit prediction provenance. The 52-node
32B/7B candidate example is a frozen synthetic teaching trace; empirical load
boundaries and declared hardware rates do not establish real deployment accuracy.

The [RX9070XT/Qwen3-4B course study](docs/course-sla-9070xt-20261111/README.md)
retains 39 fresh-Runtime cohorts and 39,000 completed requests after a remote
prediction-hash commitment. Collection passed, but the imported primitive model
missed its preregistered accuracy targets: its predicted safe client cap was 16,
while the largest observed baseline grid point passing all three repeats was 4.
All failed SLA and held-out outcomes remain in the report; no refitting was done.

The 1.12 limited live validation added complete requested-storage Workspace
attribution at two single-card prefill shapes, fresh-process physical KV-pool
success/OOM endpoints on both AMD cards, TP2 unseen-shape stage validation and
native PP2 independent exact-shape repeats. See the [Workspace](docs/workspace-live-validation.md),
[capacity](docs/capacity-live-validation.md) and [multi-GPU stage](docs/multigpu-stage-validation.md)
records for their separate scopes. Actual cache/chunk/mixed paths and two
independent physical replicas also passed limited
[correctness probes](docs/online-policy-live-validation.md). Prefix Cache and
chunking additionally met narrow p50 targets in three independent conditional
repeats each; mixed accuracy retains an explicit coverage gap. These results do not establish general
memory accuracy, online SLA, CPU swap or four-card TP2×PP2 accuracy.

The optional 1.13 Worker-based collector records real prefill and decode
iterations on each rank, including native PP buffer lifetimes. A raw acquisition
that exits successfully still needs complete Torch, native, transport and
observed-domain audits before it can be calibrated. The current raw collection
domain is BF16, synchronous eager MRv1, one active sequence, fixed 256MiB KV,
without cache/chunk/mixed/swap; CPU combination tests establish software
compatibility with those policies separately. See the
[Workspace workflow](docs/workspace-and-capacity.md#collect-worker-level-prefill-and-decode)
for the exact domain and independent-repeat commands. Instrumented stage
durations from these memory captures are diagnostic, not new latency Profiles.
The scoped single-card train/frozen-repeat pair covers eight exact shapes,
including six Decode contexts, with 24 formal observations per cohort, complete
coverage and zero byte underprediction. Multicard component evidence is admitted
only after every attribution gate closes. TP2 also completed training and a
frozen independent repeat, with 24 records/48 rank observations per cohort,
full exact-shape coverage and zero byte error. PP2 subsequently completed the
same workflow in 1.14: 48 records per cohort and 16 stage-specific shapes,
full coverage and zero byte error. See the
[Worker Workspace validation record](docs/workspace-stage-validation.md) for
completed results and retained gaps.

Version 1.14 adds source-bound CPU signal-body and passive KFD event-page
observations for PP2. Surviving handles retain their actual logical lifetime;
CPU regions and driver GTT capacity are reported separately, while all GPU
backing and fixed-budget audits remain required. See the
[synchronization evidence contract](docs/pp2-synchronization-evidence.md).

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
Live stage collection requires an already installed ROCm-enabled PyTorch and a
Runtime adapter on the AMD machine. CPU simulation and fixture validation keep
the dependency-free core; they do not install ROCm, PyTorch or serving software.

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
boundary. Batches are homogeneous unless mixed batching is explicitly enabled. See the
[configuration and resource contract](docs/prefix-cache-and-chunking.md).

## Run mixed Prefill/Decode batches

```bash
mlsys-sim simulate --experiment examples/v1_3/mixed_batching.json --out runs/mixed-batching
mlsys-sim simulate --experiment examples/v1_3/pp_mixed_cache_swap.json --out runs/pp-mixed-cache-swap
```

Set `strategy.mixed_batching=true` to pack ready Decode queries first, then use
remaining Batch Token budget for Prefill in the same stage execution. Chunking
is optional; without it, only a complete remaining prompt that fits can join.
PP uses the same member-specific progress and feedback rules, and cache/swap
ownership is retained through safe boundaries. Synthetic mixed service uses the
separate `calibration.mixed_ms` value. Profile-backed mixed execution requires
exact query/context arrays and ordered `request_phases`; pure-phase times cannot
stand in for missing mixed measurements. See [mixed batching](docs/mixed-batching.md).

## Run multi-replica routing

```bash
mlsys-sim simulate --experiment examples/v1_4/round_robin.json --out runs/round-robin
mlsys-sim simulate --experiment examples/v1_4/prefix_aware.json --out runs/prefix-aware
mlsys-sim simulate --experiment examples/v1_4/replicas_pp_mixed_cache_swap.json --out runs/replicas-pp-mixed-cache-swap
```

`strategy.replicas` selects independent model replicas. Devices are ordered by
replica, stage and TP rank. `replica_routing` supports `round_robin`,
`least_loaded` and `prefix_aware`; routing observes current load and local
cached prefixes without reading future output lengths. Each request keeps its
selected replica through recovery. KV/cache/host budgets and DMA lanes are local
to each replica, while clock, evaluation window and execution budgets are global.
Profile-backed runs supply one `replica_profile_paths` entry per replica. See
[multi-replica routing](docs/multi-replica.md) for contracts and diagnostics.

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

## Collect and assess hardware evidence

```bash
mlsys-sim collect --kind probe --out runs/rocm-probe
mlsys-sim validate-hardware --plan examples/v1_5/fixture-validation.plan.json \
  --out runs/acceptance-fixture --allow-test-fixture
```

The probe reports availability and version evidence without a CPU timing
fallback. The acceptance example uses hand-calculated fixtures and must retain
`hardware_validated=false`. A live stage adapter measures a synchronized Runtime
call; RCCL collection checks AllReduce timing and correctness; online imports
keep server Token-ready traces distinct from client SSE delivery. See
[ROCm collection](docs/rocm-collection.md) for the supported plans, reference-data
limits and explicit `PASS`/`FAIL`/`INSUFFICIENT_EVIDENCE`/`TEST_FIXTURE_ONLY`
verdicts. A completed command is not itself a hardware acceptance result.

## Import and simulate Qwen3 geometry

```bash
mlsys-sim import-model --kind qwen3 --config examples/v1_6/qwen3-4b.hf.json \
  --name Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --out runs/qwen3-model
mlsys-sim predict --experiment examples/v1_6/qwen3-4b-predict.json --out runs/qwen3-predict
mlsys-sim simulate --experiment examples/v1_6/qwen3-r2-tp2-pp2.json --out runs/qwen3-combined
```

Residual hidden width and query projection width are independent. Qwen3-4B uses
2560 residual features and 32×128=4096 query features. Model accounting includes
Q/K Norm, final RMSNorm and tied vocabulary storage. Tied weights share storage
on one physical rank; separate PP endpoint devices retain their own copies.
The combined synthetic example exercises Qwen3 geometry with R2×TP2×PP2,
Prefix Cache, mixed batches and CPU swap. It does not load model weights or
measure GPU performance. See [Qwen3 and internal hooks](docs/qwen3-and-vllm-hooks.md)
for architecture fingerprints and capture evidence boundaries.

## Capture native vLLM PP boundaries

```bash
mlsys-sim simulate --experiment examples/v1_7/qwen3-native-r2-tp2-pp2.json \
  --out runs/qwen3-native-combined
```

`strategy.pipeline_transport="vllm_native"` models the two Qwen3 intermediate
tensors, conditional TP wire slicing and receiver AllGather. The pinned MRv1
hook observes the real lazy receive without consuming it early and removes its
wait/AllGather interval from stage service. Native stage Profiles retain exact
physical payload descriptors; boundary communication is charged once using
explicitly selected communication Profiles or labeled analytical assumptions. Full receiver buffers remain resident on each rank.
The example is a CPU simulation with synthetic stage durations, not a live GPU
measurement. See [native PP capture](docs/native-pp-capture.md) for import,
calibration, padding bounds and unsupported Runtime modes.

## Calibrate native PP communication independently

Native communication calibration uses standalone unidirectional P2P and
receiver AllGather measurements, with preallocated buffers and explicit
per-placement/dtype/software identity. The collector requires an existing HIP
PyTorch/RCCL installation; CPU simulation keeps no required GPU dependency.
A selected communication Profile replaces the native boundary estimate without
changing stage Profile identity or adding local model collectives again.
Missing primitive placement or byte coverage fails explicitly. With no selected
communication Profile, the existing analytical boundary path remains available.
```bash
mlsys-sim simulate --experiment examples/v1_8/native-pp-serving.fixture.json \
  --out runs/native-comm-fixture-serving
```

The example selects a hand-calculated communication Profile and synthetic stage
times. See [native communication calibration](docs/native-pp-communication.md)
for collection, fitting, independent holdout and fixture provenance. Opt into
`calibrate --native-pp-communication --native-pp-model size-lookup` for repeated
size knots with conservative jump/gap guards; affine remains the default.
Lookup validation reports sample error and per-size mean error separately.
For the tested RCCL small-message LL regime, opt-in benchmark v3 requests both
P2P LL thresholds as zero, preserving full completion timing and binding the new
policy to its Profile. Existing benchmark defaults remain unchanged.

## Attribute Workspace and validate capacity

Workspace calibration accepts a complete Runtime-owned allocation ledger,
with physical-storage alias deduplication and completed-release lifetimes.
The resulting exact per-rank Profile supplies dynamic Workspace bytes without
changing stage duration or charging fixed reserves again. A selected Profile
has no interpolation or analytical fallback. Passive vLLM WorkspaceManager
observation remains partial managed-scratch evidence, not automatic attribution
of all Runtime temporary memory.

A capacity sweep starts a fresh Runtime/process for each candidate and repeat,
retains typed device OOM separately from logical admission refusal, timeout and
other failures, and compares observations with full-experiment simulation.
Reports give observed success-to-OOM brackets and resolution; unstable,
nonmonotonic or censored evidence does not become a precise universal threshold.
```bash
mlsys-sim predict --experiment examples/v1_9/native-pp-workspace.fixture.json \
  --out runs/workspace-fixture-predict
```

This example combines hand-calculated Workspace/communication Profiles with
synthetic stage time. See [Workspace and capacity validation](docs/workspace-and-capacity.md)
for adapter contracts, fixture workflows and evidence limits.

After freezing a complete Workspace fit, evaluate a new acquisition with
`validate --workspace-repeats --training-dataset fit/dataset.json --profile ...
--dataset ...`. The report keeps each TP rank, shape coverage and
underprediction separate. It does not refit, interpolate, or grant a hardware
acceptance verdict.

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
within each replica on one GPU, one synchronous TP group, or a pipeline of TP
groups. Multiple replicas run on one global event timeline. Dense MHA/GQA
models use explicit sharding and KV replication. PP reserves all-stage KV before
launch and budgets persistent activation buffers. Request preemption either
rebuilds KV or swaps it through a bounded CPU pool, preserving output history and
accounting for recovery latency. PP pressure drains in-flight waves before
reclaiming request KV; DMA stalls new GPU work under a conservative no-overlap
policy.
Continuous batching is prefill-first by default and decode-first with chunking
or mixed batching. FIFO retains its single-request owner and does not mix work.

The `synthetic` backend uses fixed phase durations; `analytical` uses uncalibrated
compute/bandwidth assumptions. The `profile` backend checks model, device,
software, layout and timing identities, then uses exact shapes or complete
homogeneous interpolation cells. Mixed shapes and native-PP Profiles require exact samples;
they do not interpolate or extrapolate. Native PP samples also retain the
observed padding and per-tensor transport descriptors. Other out-of-domain requests fail unless exploratory
extrapolation is explicitly enabled. Whole-stage profile times already include
local collectives; those costs are not added twice.

Warmup executes on an isolated timeline and leaves measurement with empty KV.
An arrival window is half-open; out-of-window requests are not admitted. Drain
behavior and budgets are explicit. See the [specification](docs/specification.md).

`simulate` models online PP stage queues, P2P transfers and feedback dependencies.
`predict` remains shape-only analysis and never supplies online SLA metrics.
NVMe swap remains outside this version; quantized storage is modeled as
declared effective byte widths with explicit dequant work, never as a format
parser (see [quantized storage](docs/quantization.md)). Allocator fragmentation, kernel cycles and
live CUDA/HIP or host DMA execution are not simulated.
The [capability matrix](docs/coverage-matrix.md) separates software support from
hardware evidence. A calibrated profile does not itself establish held-out
hardware accuracy.

## Outputs and development

Simulation output includes `manifest.json`, `result.json`, `requests.jsonl`,
optional `events.jsonl`, and `logs.txt`. Pipeline runs add per-stage activity and
progress diagnostics; swap-enabled runs add host/DMA resources and per-request
copy counters, byte counts and timing. Prefix/chunk runs expose cache occupancy,
reuse and actual prompt work. Mixed runs record member phases and joint-batch
execution diagnostics. Multi-replica runs add per-request routing, independent
resource reports and global metrics recomputed on one window. Manifests retain effective configuration,
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
