# Independent native PP communication calibration

Version 1.8 adds independent primitive measurements for the native PP boundary.
They complement the isolated stage service described in
[native PP capture](native-pp-capture.md). A receiver's lazy dependency wait is
still not a network benchmark: it can include upstream production and scheduling
delay. The new collector runs directed P2P and receiver AllGather separately.

The standalone live path requires an existing single-node HIP PyTorch/RCCL
installation with one GPU per participant rank. The CPU core has no new required
dependency and does not import Torch for ordinary simulation. Injected runners
and raw test fixtures require explicit fixture permission, retain that source
through fitting/validation, and keep `hardware_validated=false`. No AMD host has
been used to establish accuracy for the shipped examples.

## Measure a defined primitive

A P2P record measures a source rank's nonblocking send and its destination rank's
matching nonblocking receive in one direction. It is not ping-pong, not an RTT,
and not an RTT divided by two. The byte count is the actual one-way message
payload per paired TP rank.

An AllGather record measures the ordered TP group on a receiving PP stage.
`message_bytes` denotes the full gathered output, so each participant's input
has `message_bytes / TP` bytes. Input and output buffers are preallocated and
out of place; input/output aliasing is not the measured primitive.

The collector binds each rank to its declared GPU, uses NCCL/RCCL data groups
and CPU Gloo barriers for control. After initialization and allocation, it warms
up the primitive and performs correctness checks outside measurement. Each
measured round follows the defined control boundary, records local elapsed
service through the real work completion and device synchronization, and keeps
the maximum participant duration. It does not subtract timestamps from distinct
ranks or pretend that a control barrier is free of perturbation.

The timing interval excludes process/group startup, allocation, warmup, CPU
control barriers and correctness work. Those exclusions define the measurement,
not the entire cost of calling vLLM's native transfer implementation. A live
collector verifies actual device/stack information, including the loaded RCCL
library and its content hash, rather than accepting a version label as proof.

## Keep physical placement and byte conventions

Native communication uses separate `native_pp_communication_measurements` and
`native_pp_communication_profile` contracts. The generic AllReduce/P2P
communication-fit workflow remains an explicit Infra-patch tool; it is not a
substitute for the native boundary contract.

Fits stay separate for every primitive, dtype and physical placement:

- directed P2P source/destination devices for each paired TP rank at a boundary;
- the ordered receiver TP device group for each receiving PP stage.

Record `stage_index` identifies the receiving stage. P2P `tp_rank` identifies
one paired source/destination TP position; AllGather uses `tp_rank=null` and
the complete ordered receiver group. Each record retains one positive local
duration per participant, their exact maximum and zero correctness errors.

A measurement for one pair or receiver stage does not fill a missing placement
elsewhere. Stack, RCCL and physical identity must agree with the selected
experiment. A profile cannot silently cross replica, rank order, dtype or
software boundaries just because message sizes match.

Each placement uses a direct nonnegative affine fit:

```text
T(message_bytes) = intercept_ms + slope_ms_per_byte × message_bytes
```

This is a fit to the measured primitive. It does not impose a ring formula on
AllGather or reinterpret a P2P pair as an AllReduce. Dataset/source hashes,
training records and groups, version lineage, byte coverage and fitting errors
remain part of the immutable Profile provenance.

## Collect, fit and validate

On a configured live AMD host, supply a fully bound native experiment and a
collection plan:

```bash
mlsys-sim collect --kind pp-communication --experiment bound-native-experiment.json \
  --plan native-communication.plan.json --out runs/native-communication
mlsys-sim calibrate --native-pp-communication \
  --dataset runs/native-communication/dataset.json \
  --profile-version native-communication-v1 --out runs/native-communication-fit
```

Use a new output directory for each command. A completed collection is not a
hardware acceptance verdict. For test fixtures, add `--allow-test-fixture` to
commands that consume them; resulting Profiles retain fixture provenance.

A live plan contains the bounded acquisition settings. The shipped
`examples/v1_8/collector.plan.live-template.json` is a template, to be used with
an experiment bound to the real host rather than the synthetic example:

```json
{
  "schema_version": 1,
  "dataset_id": "native-pp-live-01",
  "message_bytes": [512, 4096, 32768, 65536],
  "warmup_iterations": 3,
  "iterations": 10,
  "repeats": 3,
  "timeout_s": 120,
  "max_output_bytes": 8388608,
  "source_reference": "Standalone acquisition on the bound AMD host"
}
```

The same size list requests one-way wire bytes for P2P and full-output bytes for
AllGather. Choose coverage for both actual native byte ranges, including the
padding captured by stage Profiles. Sizes must cover whole activation elements
and AllGather partitions; the common plan aligns them to TP × element size.
All ranks create data groups in the same order. The internal supervisor launches
one worker per GPU in a controlled process session with file rendezvous;
launch it outside any existing distributed group. Timeout and complete-output
bounds cover the worker process tree and evidence files, not just console text.

A raw import plan additionally supplies `raw_path` relative to the plan and an
explicit `source` object. It imports exported per-rank evidence without running
the live worker. Complete rank/participant, launch, device, correctness and
maximum-local-duration checks still apply. Each rank export carries the same
`session_id` from one acquisition; rank records from different launches must
not be assembled together, even when their plans and devices match. The
collection retains the agreed session in `benchmark.launch.session_id` for
audit. Injecting a Python runner is a library-only test surface and always
requires fixture opt-in.

Validate an already fitted Profile against an independent held-out dataset:

```bash
mlsys-sim validate --native-pp-communication \
  --dataset independent-communication-holdout.json \
  --profile runs/native-communication-fit/profile.json \
  --out runs/native-communication-holdout
```

This path validates an existing Profile. It does not create a train/holdout split
or refit on the held-out rows. Source, identity, placement, domain and training
leakage checks apply before reported errors can be interpreted. Keep disjoint
runs/configuration groups and all repeated observations from one acquisition
group together. Stage holdout and communication holdout evaluate different
service boundaries; neither supplies missing online token/lifecycle evidence.

## Run the CPU fixture workflow

All files in `examples/v1_8` are hand-calculated/source-shaped fixtures or a
live-host plan template. The fixture stack strings and library digest are
explicit placeholders, not facts from an AMD execution. A complete import,
fit and independent holdout can run without Torch or GPUs:

```bash
mlsys-sim collect --kind pp-communication \
  --experiment examples/v1_8/native-pp-collect.fixture.json \
  --plan examples/v1_8/collector.plan.fixture.json \
  --allow-test-fixture --out runs/native-primitive-fixture
mlsys-sim calibrate --native-pp-communication \
  --dataset examples/v1_8/communication-train.fixture.json \
  --profile-version native-comm-fixture-v1 \
  --allow-test-fixture --out runs/native-comm-fixture-fit
mlsys-sim validate --native-pp-communication \
  --dataset examples/v1_8/communication-holdout.fixture.json \
  --profile runs/native-comm-fixture-fit/profile.json \
  --allow-test-fixture --out runs/native-comm-fixture-holdout
mlsys-sim predict --experiment examples/v1_8/native-pp-predict.fixture.json \
  --out runs/native-comm-fixture-predict
mlsys-sim simulate --experiment examples/v1_8/native-pp-serving.fixture.json \
  --out runs/native-comm-fixture-serving
```

The collect example exercises raw per-rank import and reduction. The fitting
example separately uses the canonical training fixture; the holdout has
disjoint record/group IDs and intermediate message sizes. Prediction and
serving select the shipped `communication.profile.fixture.json`, generated
from that training dataset and version, so they do not depend on earlier
command output paths. Their calibrated communication source remains a test
fixture. The serving example exercises one Qwen3 TP2×PP2 replica with Prefix
Cache, chunking, mixed batches and CPU swap, using synthetic stage durations.
It does not establish any hardware throughput or correctness claim.

## Select a communication Profile explicitly

A single-replica native experiment may select its fitted file separately from
its stage backend:

```json
{
  "strategy": {
    "pipeline_transport": "vllm_native",
    "pipeline_p2p": true
  },
  "calibration": {
    "native_pp_communication_profile_path": "native-communication-fit/profile.json",
    "native_pp_communication_profile_version": "native-communication-v1",
    "native_pp_communication_allow_extrapolation": false
  }
}
```

This fragment extends a complete valid experiment. For multiple independent
replicas, use `replica_native_pp_communication_profile_paths` in replica order
and provide each local replica's matching Profile. Selection is valid only for
native PP with boundary communication enabled. Test Profiles additionally need
`calibration.allow_test_fixture=true`.

Without a selected communication Profile, the analytical native P2P/AllGather
path is unchanged. With a selected Profile, a missing primitive placement or
out-of-domain message size fails explicitly; it does not silently fall back to
analytical estimates or zero. Extrapolation is an explicit separate policy,
defaults to false and must remain visible in prediction provenance.

The CLI snapshots and hashes selected communication Profiles, including each
local replica's file. These snapshots are distinct from stage Profile files.
Selecting or refitting a communication Profile does not change stage Profile
identity or add model-local TP collectives again.

## Apply each boundary once

For every physical tensor described by native PP capture:

1. Query P2P service for each paired source/destination TP rank using that
   tensor's actual wire bytes; take the maximum paired-rank estimate.
2. If that tensor is sliced, query one receiver-group AllGather using its full
   output bytes. An unsliced tensor does not incur reconstruction AllGather.
3. Sum those boundary components for `hidden_states` and `residual` once.

An observed stage Profile supplies actual padding and physical geometry.
Configured analytical/synthetic stages retain their declared geometry rules.
The communication lookup uses those physical bytes, not the residual/query
projection width interchangeably. Online PP admission checks the required
boundary estimates before committing KV reservations. Shape prediction uses
the same boundary logic, and cache/chunk/mixed/swap ownership remains governed
by the existing safe boundaries.

This boundary cost remains separate from stage service, which already includes
local model collectives and excludes observed incoming receive/postprocess
AllGather. There is no extra local TP AllReduce charge. Existing full-resident
buffer budgets also remain in force: wire slicing does not reduce receiving
backing, reconstructed output or persistent input-copy allocations.

## Interpretation limits

The benchmark measures preallocated independent primitive service. It excludes
vLLM's per-call AllGather output `torch.empty` allocation, metadata ingress,
upstream production/waiting, shared-fabric contention and arbitrary Runtime
stream dependencies. Its Profile therefore does not claim to measure an entire
native vLLM boundary or reproduce native communication/compute overlap.

Standard NCCL/RCCL placement and dtype evidence does not automatically establish
agreement with every symmetric-memory, custom collective or transport path.
Treat those Runtime differences as explicit coverage limits. A valid fit or
small held-out primitive error does not establish end-to-end serving accuracy,
cache/swap accuracy or online SLA agreement. Those claims still need matching
live Runtime traces and the separate hardware-evidence acceptance workflow.
