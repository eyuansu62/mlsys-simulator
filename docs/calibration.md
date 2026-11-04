# Calibration and Profile data

The calibration workflow consumes step timing records and versioned lookup
Profiles. Version 1.5 adds separate optional ROCm collection adapters; the
`calibrate` command still fits supplied records rather than running a GPU.
The included `.fixture.json` files are hand-calculated tests and never become
measured evidence. See [ROCm collection](rocm-collection.md) for live prerequisites
and raw-record boundaries.

## Evidence and identity

Datasets and Profiles declare `source.kind` as `measured` or `test_fixture`, with
a nonempty `source.reference`. Simulated source data is rejected. Fixtures require
an explicit `allow_test_fixture` API/config setting or `--allow-test-fixture` CLI
flag at each relevant boundary; reports retain the fixture source.

Measured records require explicit model revision, device names and Runtime name/
version, without `unspecified`, `unknown` or toy placeholders. Driver/ROCm/RCCL
versions must be explicit or say `not-applicable` for an inapplicable component.
These checks enforce a declared data contract, not independent proof that a
measurement was collected correctly.

Measured identities must specify `weight_dtype`, `kv_dtype` and `activation_dtype`
explicitly. `float16` and `bfloat16` are distinct despite sharing a two-byte
width; explicit dtypes must agree with their byte widths. `auto` is accepted
for analytical assumptions and fixtures, but cannot identify measured data.

`profile_identity(experiment)` produces the full required identity: model
specification and revision, ordered devices, topology/link assumptions,
TP/PP/stage layers and shard plan, Runtime/software versions, timing boundary,
KV block size and persistent execution assumptions. A mismatch fails with
`CALIBRATION_MISMATCH`. Profile labels are independent of software versions.

Step datasets and Profiles retain a single-replica identity. For a deployment
with multiple replicas, calibrate each local TP/PP layout separately and supply
one `replica_profile_paths` entry per replica. A cluster identity is valid for
online Compare, but is rejected as a step dataset/Profile identity because the
records have no replica coordinate. `profile_identity` adds
`parallelism.replicas` only for multi-replica cluster identities; the earlier
single-replica structure is unchanged.

Multi-replica CLI runs preserve raw Profiles in `profiles/replica-0.json`,
`profiles/replica-1.json`, etc. The manifest binds each origin/snapshot path,
hash, version and declared source. Each replica's timing and Workspace use the
same immutable snapshot. Fixture and measured sources remain distinct even
when a cluster contains different source kinds. See
[multi-replica routing](multi-replica.md) for configuration and prediction scope.

## Dataset schema

A JSON dataset has these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `dataset_id` | Nonempty dataset identifier |
| `identity` | Full identity described above |
| `source` | `{kind, reference}` with declared provenance |
| `records` | Nonempty observation list |
| `notes` | Optional JSON object |

Each observation contains `record_id`, `group_id`, `phase`, `stage_index`,
`query_tokens`, `context_lengths` and positive finite `duration_ms`.
`workspace_peak_bytes` is an optional nonnegative integer, or `null` when not
observed. Record IDs are unique. Phases are `prefill`, `decode`, `recompute` and `mixed`;
arrays preserve every sequence shape. Decode queries are one Token each.
Prefill/recompute context is the number of KV positions already present before
the query, including a cached prefix or earlier chunks. A complete uncached
prompt starts at zero; later chunks need their actual nonzero context.

Mixed observations additionally require `request_phases`, with one ordered
`prefill` or `decode` entry per query and both roles present. Every Decode query
is one Token; a one-Token Prefill retains its explicit role. Homogeneous records
omit `request_phases`. For example, a record in a valid fixture dataset can be:

```json
{
  "record_id": "mixed-1",
  "group_id": "mixed-shape-a",
  "phase": "mixed",
  "stage_index": 0,
  "request_phases": ["decode", "prefill"],
  "query_tokens": [1, 4],
  "context_lengths": [32, 8],
  "duration_ms": 2.5,
  "workspace_peak_bytes": 4096
}
```

The enclosing dataset still needs full identity and explicit source provenance.
This illustrative record is not a measurement.

Use a group ID for observations that must stay together during validation.
Repeated exact shapes are connected even when their group IDs differ, preventing
repeated runs of the same configuration from crossing the fit/holdout boundary.
The sample dataset under `examples/v1/` illustrates the complete JSON structure.

## Fit a new version

```bash
mlsys-sim calibrate --dataset examples/v1/step_measurements.fixture.json \
  --profile-version fixture-v1 --out runs/calibration --allow-test-fixture
```

The result is a lookup `profile.json` with mean duration for each exact shape,
maximum observed Workspace for repeats, sample counts, dataset hash and training
record/group/shape IDs. It also records repeated-observation spread. Total step
times do not identify six macro coefficients, so fitting does not invent an
Attention/Linear/Misc attribution.

`--parent-profile FILE` checks matching identity and records version lineage;
it does not merge old samples into the new dataset. A different version label
is required. Use a fresh output directory; original datasets, Profiles and run
artifacts are not overwritten.

To consume a Profile, set `calibration.backend="profile"`, `profile_path`, and
its `profile_version` in an experiment. Relative Profile paths resolve with the
configuration loader. Match Runtime/software identity explicitly; keep
`allow_test_fixture` false for ordinary measured work. The supplied
`profile_serving.json` deliberately opts into fixture use.

## Lookup and domain policy

Exact matches include phase, stage and the complete query/context arrays.
Ragged batches require an exact match; equal total Tokens or average context
length is not a substitute.

Mixed lookup also matches the complete ordered member-phase array. It is
exact-only: no interpolation or extrapolation, and no substitution from
homogeneous Prefill/Decode records. This remains true when
`allow_extrapolation=true`. Mixed samples retain `request_phases` during fitting,
and shape provenance/holdout leakage checks include those roles. The legacy
homogeneous shape key remains unchanged; mixed keys append a fifth role-array
element under schema version `1`.

Otherwise, a homogeneous Batch may interpolate on `(batch size, query Tokens
per sequence, context Tokens per sequence)` for a fixed phase and stage. Every
required corner of the surrounding grid cell must exist. There is no
nearest-neighbor or analytical timing fallback when a corner or shape is absent.

Out-of-domain queries fail with `OUT_OF_CALIBRATION_DOMAIN` by default. Explicit
`allow_extrapolation` permits boundary-cell extrapolation only where the needed
axis samples and corners exist. Nonpositive/non-finite predictions are rejected.
Step estimates and run provenance record when extrapolation was used; it remains
exploratory evidence.

Whole-stage timings already include local collective communication, output and
sampling. Do not add analytical local communication again. Inter-stage PP P2P,
native receiver AllGather and explicit feedback are separate. Native PP
captures exclude the naturally observed receive/postprocess interval while
retaining persistent input copy and computation. Its Profile samples preserve
incoming/outgoing physical payload descriptors and use exact-only lookup;
repeated logical shapes with different payload geometry cannot be averaged.
The raw dependency wait does not calibrate independent network service. See
[native PP capture](native-pp-capture.md). Online PP can use stage Profiles in its
[online PP engine](pipeline-and-swap.md); shape-only `predict` still has no online
queueing or SLA metrics.

Chunking preserves this whole-stage timing boundary and conservatively charges
output/sampling work for every chunk, although only the final prompt chunk
emits an output Token. Collect matching chunk/query/context records rather than
reusing an unchunked zero-context time. Existing exact/grid domain rules apply:
an unsupported context fails instead of silently using a zero-context record.
Analytical chunk estimates use the same conservative output-cost convention;
synthetic durations are fixed per phase batch/chunk. See
[Prefix Cache and chunking](prefix-cache-and-chunking.md).

Mixed batches use a single whole-stage timing/Workspace observation for the
joint shape, including local collectives. An analytical mixed estimate assumes
common weight reuse and one set of macro launches/collectives. Synthetic mixed
service uses independent `calibration.mixed_ms`, rather than adding or taking
the maximum of the pure-phase settings. These are declared model policies;
collect matching mixed records to calibrate a particular Runtime.

## Workspace and provenance

Exact workspace queries use the maximum observed value for that shape.
Interpolation uses the maximum of measured corner values, only when all corners
have workspace observations. This is a stated policy, not a guarantee about an
unmeasured interior peak. Extrapolated or unobserved workspace is unknown.

Without an explicitly selected independent Workspace Profile, when stage
workspace is unknown, Memory Planner uses its labeled analytical
`base + query_tokens × bytes_per_token` assumption. Missing workspace does not
mean zero usage or measured feasibility. Timing lookup still follows its strict
domain policy.

Simulation manifests snapshot Profile identity/hash/version/source and domain
policy. Profile-backed simulation remains `backend="simulated"`; a measured
Profile means calibration data was supplied, not that held-out hardware
validation passed. See [validation](validation-protocol.md) for the next step.

## Calibrate standalone communication

Communication observations use a separate JSON dataset with
`kind="communication_measurements"`, identity/source metadata and records with
`record_id`, `group_id`, `primitive`, `ranks`, `message_bytes` and `duration_ms`.
Supported primitives are `allreduce` and `p2p`. Preserve the algorithm, rank
count, topology and timing source used by the collector.

For a user-supplied dataset:

```bash
mlsys-sim calibrate --communication --dataset communication-measurements.json \
  --profile-version comm-v1 --out runs/communication
```

The fit separates primitive/rank groups and produces alpha-beta parameters plus
an `infra_patch` for each group, including link latency and bandwidth. Use
`calibrate_communication` and `load_communication_dataset` for the equivalent
Python workflow. Select a matching group's patch explicitly when constructing
an analytical experiment; a group-specific fit is not a universal network model.

This path does not automatically add fitted communication to a whole-stage step
Profile, which already includes local collectives. Patching an unrelated
topology or adding both costs would violate the timing contract. Fixture inputs
still require explicit opt-in and retain their source kind.

## Independent native PP communication Profiles

Version 1.8 adds a separate native-boundary measurement/Profile workflow for
directed P2P and receiver AllGather, with placement/dtype/actual-stack identity.
It fits direct nonnegative affine primitive service, validates an existing
Profile against independent holdout records and is selected explicitly per
local replica. It is distinct from both a whole-stage Profile and the generic
communication Infra-patch fit. Native lazy receive waiting does not become an
independent network measurement. Missing placement or byte coverage never
silently falls back when a communication Profile is selected. See
[native PP communication calibration](native-pp-communication.md) for commands,
fixture examples and interpretation limits.

## Independent exact per-rank Workspace Profiles

`calibrate --workspace` consumes the complete ownership contract described in
[Workspace and capacity validation](workspace-and-capacity.md). It fits no
duration: repeated observations reduce to a maximum for each exact shape/rank,
including observed zero. Mixed roles and all TP rank peaks remain explicit.
An explicitly selected Workspace Profile replaces only the dynamic Workspace
budget, has no interpolation/extrapolation or missing-shape fallback, and leaves
fixed reserves and stage timing identity unchanged. Without selection, the
legacy stage-observation/analytical policy above remains available.

The Runtime adapter owns complete scratch instrumentation, including non-Torch
allocations; passive vLLM manager capacity or allocator peak-minus-baseline is
partial evidence and cannot be relabeled canonical Workspace. Profile snapshots
retain attribution scope, source, version and SHA per local replica. Capacity
validation is a separate full-experiment workflow and does not fit Workspace
from an observed OOM point.
