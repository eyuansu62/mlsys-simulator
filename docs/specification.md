# Version 1.9 software specification

This document records the original serving-core contract. Later opt-in teaching
workflows are specified in [course workflows](course-workflows.md): the explicit
Chapter 1 counting protocol, seeded open/closed-loop load scans, and dependency
DAGs over independent heterogeneous model pools. Their separate plan schemas do
not add fields to the original request schema or change its fixed-arrival semantics.

The CPU simulator separates system events, performance/communication models and
calibration data. It has no required third-party runtime dependency and performs
no real GPU execution. This specification describes supported software behavior;
hardware prediction accuracy requires separate evidence.

## Configuration and units

Experiment schema version is `1`. Top-level fields are `schema_version` and six
sections, with strict unknown-field/type/range checks:

| Section | Contract |
| --- | --- |
| `workload` | Unique request IDs, nonnegative arrival ms, positive input/output lengths, priority, optional relative timeout and explicit prompt token IDs/cache namespace |
| `infra` | Ordered devices, memory bytes, effective compute/bandwidth assumptions and named topology/link parameters |
| `strategy` | Dense model/revision, replica count/routing, TP/PP/stage layers, microbatch/inflight/feedback/link policy, Batch/chunk limits, mixed batching, Prefix Cache and preemption/fallback |
| `calibration` | Analytical/synthetic parameters, local/per-replica stage, independent Workspace and native communication Profile paths/versions, full software identity and explicit fixture/extrapolation policy |
| `evaluation` | Optional TTFT/TPOT SLA, half-open arrival window, drain and optional drain limit |
| `execution` | Seed, time/event limits, detailed logging, isolated warmup requests and `empty_cache` initial state |

Times use ms, byte quantities use bytes, bandwidth uses decimal GB/s and compute
uses TFLOP/s. Float storage widths of 2/4/8 bytes are supported; packed/8-bit
formats are rejected rather than omitting their metadata. Model/device defaults
are toy assumptions. Measured data needs explicit non-placeholder identities.

## Model geometry and weight ownership

The residual feature width is `hidden_size`. Query and attention-output width
is `num_attention_heads × head_dim`, independently of that residual width.
Q/O matrix parameters per TP rank therefore use
`2 × hidden_size × query_projection_size / TP`; K/V matrices and KV storage
continue using local KV heads × head dimension, including GQA replication.
PP payload width remains `hidden_size`, independently of the query projection.
The default transport carries one full-hidden tensor per rank; explicit
`vllm_native` carries Qwen3 hidden/residual tensors with observed padding and
conditional TP wire slicing plus receiver AllGather.

Optional `qk_norm` adds two shared head-dimension Norm vectors per layer/rank;
`final_norm` adds a residual-width Norm vector on the last stage/rank. With
`tie_word_embeddings=true` and PP=1, embedding and LM-head roles share one
vocabulary-sharded allocation. PP>1 stores a copy at each physical endpoint.
Analytical LM-head work remains present even when its weights alias embeddings.
The old flags default false and preserve the previous model domain and costs.

Dense Qwen3 HF configuration import retains actual dimensions, revision, dtype,
tying and Norm choices. It rejects unsupported architecture/attention/storage
options. A normalized `architecture_config_sha256` binds execution-relevant
configuration to Profile identity while raw input hashes remain provenance.
See [Qwen3 and internal hooks](qwen3-and-vllm-hooks.md).

## Event and scheduling semantics

For PP=1, each replica has one GPU or synchronous TP group executing one Batch
at a time.
Homogeneous phase batches remain the default. FIFO holds one request until it finishes. Continuous batching selects
Prefill at safe boundaries before Decode by default, with a pressure-drain
policy for recovery. Enabling Chunked Prefill instead gives Decode priority;
prompt queries retain context and respect the configured per-chunk and Batch
Token budgets. `mixed_batching=true` independently selects ready Decode first,
then packs Prefill into the remaining sequence/Token budget of the same Batch.
Without chunking, a complete remaining prompt must fit before it can join.
During pressure draining, fresh prompts remain restricted. FIFO never mixes
requests, and recomputation retains a separate homogeneous recovery Batch.

PP>1 uses independent stage queues/resources and opt-in mixed microbatches;
different stages can execute different microbatches concurrently.
Each stage owns a TP group. Every prefill chunk is a separate iteration through
the stages and feedback boundary. A mixed microbatch records ordered
`request_phases`, queries and contexts through all stages; each member advances
according to its own phase. Multiple replicas route requests on a shared global
event timeline; each replica retains independent local scheduling/resources.

The stable heap orders completion/release, timeout, arrival and new scheduling.
Prefill builds input KV and emits the first output only when the final prompt
chunk completes. The latest sampled output is processed by the next Decode, so committed output count and computed KV position
are distinct. The future output target is available only to the execution oracle.

For R replicas, devices are ordered replica-major, then stage-major, then TP
rank. Routing policies are round-robin, least current unfinished load, or
longest local prefix among the minimum-load replicas. The router uses arrived
state only, with inflight work and used KV blocks as deterministic tie-breaks.
Requests stay on their chosen replica through recovery. All replicas share one
clock, arrival window and time/event budget; local execution remains independent.
Local resource stops make a replica unavailable while other replicas continue,
and the cluster retains an unresolved result rather than hiding local failures.

A PP request has at most one outstanding iteration. The final stage includes
output/sampling in its service cost, then the request waits
`pipeline_feedback_latency_us` before its next Decode can launch. KV stays pinned
through that feedback boundary even after a final Token or timeout. Timeout and
cutoff handling preserve in-flight references and partial stage progress without
committing a finished iteration prematurely.

## Memory and recovery

Each rank maintains separate physical and logical budgets. Physical use includes
weights, the entire fixed KV pool, persistent Runtime/Graph/communication use,
Workspace, PP activation storage and safety/external constraints. Logical use
counts unique retained cache blocks plus private request blocks, rounded up by
block size; shared prefix references do not duplicate their GPU storage. Used KV
is not added again to a physically reserved pool. GQA replication is part of the per-rank shard plan.

All ranks must pass a candidate reservation before any allocation commits.
PP conservatively reserves target KV on every stage before launch at stage 0.
Per-rank persistent activation storage is adjacent stage boundaries multiplied by
max inflight microbatches, max Batch Tokens, hidden size and activation bytes.
Native transport instead uses its effective physical-token bound and separates
outgoing full hidden/residual pairs, incoming reconstruction plus retained
sliced-receive backing per inflight wave, and one fixed full input-copy target
pair per receiving stage. An interior stage sums incoming and outgoing budgets.
Wire slicing never divides the resident budget by TP. A payload descriptor's
resident bytes represent one final pair, not total allocator peak. See the
[native buffer contract](native-pp-capture.md#charge-transport-once-and-reserve-full-buffers).
This is separate from collective communication buffers.
An explicitly selected independent Workspace Profile replaces dynamic bytes
per exact shape and rank, using complete ownership and repeated per-rank maxima.
Observed zero is valid; unknown or missing shapes do not silently fall back.
Static feasibility retains fixed reserves while shape admission checks the
selected dynamic bytes instead of an unused analytical-base floor. Without
selection, the old stage-Workspace/analytical path remains. Releasing request
blocks preserves the fixed pool allocation and cannot by itself solve a Workspace
OOM. See [Workspace and capacity](workspace-and-capacity.md).

Prefix Cache uses namespace and full token ancestry from explicit prompt token
IDs. Missing IDs bypass cache lookup. Only complete immutable prompt blocks
publish; the longest contiguous resident hit leaves at least one prompt Token
for output computation. Cache entries and private request blocks share the same
fixed pool. References pin active cache blocks; unreferenced entries are evicted
in LRU order to satisfy cache capacity or a new reservation before preempting a
live request. Failed admission rolls back provisional lookup references before
execution so they cannot block resident Decode progress; later admission can
retry lookup. Incremental SHA256 keys include parent ancestry, namespace and
block endpoint without retaining each complete token prefix. Cache
lookup/publish work has no separately measured CPU overhead.

KV pressure limits admission and shrinks Batch work. A safe request-level victim
can use the selected `none`, `recompute` or `swap` preemption policy. Recompute
rebuilds the known prefix without re-emitting outputs. Swap retains KV in a
bounded host pool and restores it without recomputing historical positions.
A partial prefill resumes its remaining input after recovery; chunked
recomputation rebuilds only its previously computed progress without emitting
outputs. Both retain request timing and output history. PP stops new launches
and drains in-flight stages, links and feedback before reclamation.

CPU swap reserves host capacity before D2H, releases GPU blocks only at D2H
completion, reserves all GPU destinations before H2D and keeps the host copy until
H2D completes. Both directions share one serial DMA lane; no new GPU work starts
during DMA. Copies are block-aligned and aggregated across stages/TP ranks.
Swap copies the full request, including shared blocks, then restores private
GPU blocks. Unreferenced GPU cache entries remain separately reclaimable;
there is no CPU Prefix Cache tier.
For multiple replicas, these GPU/cache/host capacities and DMA lanes are local
to each replica. Shared host bandwidth and cross-replica fabric contention are
not modeled; requests and KV are not migrated between replicas.
Timeouts and cutoffs retain pending source/destination references. Host-capacity
failure is explicit; only `swap_fallback="recompute"` allows recompute fallback.
Impossible shapes and bounded no-progress/preemption conditions end explicitly.
NVMe and arbitrary Kernel interruption remain unsupported.

## Performance backends

`synthetic` fixes phase Batch times for tests. `analytical` computes serialized
Linear/Attention/Misc, output and communication costs using declared effective
rates. Prefill/recompute preserve actual query and prefix-context lengths;
Decode preserves each context. Mixed stage execution combines the selected
queries in one macro estimate with ideal common weight reuse and one set of
launch/collective costs, rather than summing independent phase estimates.
Synthetic mixed batches use the independent positive `mixed_ms` parameter,
apportioned by stage layers. Per-chunk stage service conservatively includes
the declared output/sampling boundary even for an intermediate prompt chunk
that emits no Token. Synthetic durations apply per phase batch/chunk.
TP uses the supported Dense shard plan, local KV replication, Ring AllReduce and
logits AllGather, without communication overlap or shared-link contention.

`profile` snapshots a versioned whole-stage lookup and checks identity. Ragged
shapes require exact observations; homogeneous grid interpolation requires every
corner. Missing timing does not silently fall back to analytical estimates.
Mixed shapes require an exact ordered `request_phases`/query/context/stage
observation, even if exploratory extrapolation is enabled; there is no mixed
interpolation, homogeneous timing substitution or phase-sum fallback.
Workspace queries use that same complete shape. Explicit extrapolation for
supported homogeneous grids is flagged. Local collectives are already included in
stage times; PP boundary P2P and native receiver AllGather are separate.
Native Profiles preserve physical payload descriptors and use exact-only
lookup, without interpolation or extrapolation. See [calibration](calibration.md).

Step datasets and Profiles describe one replica. Profile-backed cluster runs
select one file per replica through `replica_profile_paths`; each must match
that local layout. Online comparison instead retains the full cluster identity
and routing strategy. CLI artifacts snapshot every local Profile, preserving
its raw bytes, hash, version and source for both timing and Workspace.

Pinned internal vLLM collection validates the supported MRv1 ABI and records
actual batch members at Runner boundaries. Complete synchronous TP groups use
maximum local elapsed duration; execute-only final stages, async consumption,
missing/failed channels and unsupported recovery are not silently converted to
whole-stage observations. Scheduler enqueue and published output/terminal
events form a separate lifecycle channel. Runtime execution fingerprints are
independent of architecture fingerprints, and both participate in identity when
present. Collection perturbation and hardware validation remain explicit. See
[Qwen3 and internal hooks](qwen3-and-vllm-hooks.md).

The pinned Runtime's native PP path can slice each outgoing tensor and run
receiver TP AllGather through lazy incoming postprocessing. With explicit
`vllm_native` transport, capture observes the original natural wait without early
consumption, excludes its wait/AllGather interval and retains subsequent input
copy in stage service. Physical hidden/residual descriptors, actual padding,
TP agreement and adjacent-stage edge agreement are required for calibration.
Raw receive waiting includes upstream dependencies and is not independent
network service. Sequence-parallel execution remains outside the boundary. See
[native PP capture](native-pp-capture.md).

Independent communication observations can fit alpha-beta parameters per
primitive/rank group and produce an explicit Infra patch. A selected patch must
match the intended topology and workload model; it is never automatically
summed with whole-stage Profile timings.

Since 1.8, native PP has a separate independent-primitive dataset/Profile path
for directed P2P and receiver AllGather. It fits direct nonnegative affine
service by physical placement/dtype/stack rather than imposing a ring formula
or producing a global Infra patch. Explicit local/per-replica Profile selection
replaces the native boundary estimate; missing placement/byte coverage fails
without analytical fallback. Each tensor incurs maximum paired-rank P2P plus
one receiver AllGather only when sliced. Stage Profile identity and model-local
collectives remain unchanged. Without selection the old analytical boundary
path remains. See [native communication calibration](native-pp-communication.md).

Online PP uses independent stage compute resources and a serialized activation
link for each adjacent boundary. TP boundary links are paired and independent,
with full per-rank payload by default. Native transport uses per-tensor wire
slices where applicable, followed by receiver AllGather. Both predict and the
online engine charge that boundary once; shared fabric contention is not modeled.
`pipeline_p2p=False` explicitly disables link cost for ideal teaching cases;
default `True` includes it. Explicit feedback remains separate from stage costs.

`predict` continues to report stage capacity, isolated service time and P2P only.
It performs no prefix lookup, chunk scheduling or online mixed batching. It has no arrival queue, so
online metrics are null with
`SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING`. Use `simulate` for the PP timeline.
With multiple replicas, prediction reports isolated `replica_predictions` and
annotated memory rows; it does not route or infer online aggregate throughput.
Detailed configuration, lifetimes and ideal bubbles are described in
[pipeline and swap](pipeline-and-swap.md).

## Warmup and evaluation

Warmup uses the same configured model/backend on a separate timeline. It must
finish successfully before measurement; its costs are reported separately and
its request KV and retained prefix entries are discarded. Measurement starts
with an empty cache; warmup does not seed measured prefix hits.
Multi-replica warmup also resets routing state and its round-robin cursor.

With a measurement window, arrivals are admitted in `[0, window)` only. Other
rows are `NOT_ADMITTED` and excluded from completion denominators. With
`drain=False`, evaluation stops at the window; with `drain=True`, admitted work
may finish subject to an optional maximum drain duration. Execution budgets
still apply. Without a window, rate denominators use elapsed simulated time.

TTFT is arrival-to-first-Token. Completed multi-Token TPOT averages first-to-last
output gaps; single-Token TPOT is null. ITL retains recovery gaps. Failed,
timed-out and unfinished admitted requests remain in completion denominators.
Goodput counts completed requests meeting all declared SLAs. Percentiles use
linear interpolation; empty/unavailable values remain null.
Cluster metrics are recomputed from every request on the common evaluation
window rather than averaging local percentiles or summing local rate reports.

## Data and result contracts

Simulation output retains effective inputs, code/input hashes, Profile snapshot
and provenance, comparison contract, request/event rows and diagnostics. PP adds
stage activity/bubble and progress reports; swap adds host/DMA state and
per-request copy counts, bytes and time. Mixed Batch events preserve member
roles, and results distinguish joint execution from homogeneous batches. Prefix/chunk reports add cache
occupancy/reuse and completed prompt work. Output
paths are exclusive. A run can execute successfully while requests fail or SLA
criteria are missed.

Step datasets, Profiles and observed runs use strict JSON with explicit source
kind/reference. Calibration produces a new version and records training members;
validation prevents configuration leakage. Online comparison requires equivalent
contracts and keeps missing values/ties explicit. Their formats and acceptance
rules are documented in [calibration](calibration.md) and
[validation](validation-protocol.md). They establish no hardware evidence that
has not actually been supplied.


Profile JSON remains schema version 1. Comparison contracts include the new
strategy and memory parameters; supported older contracts using the equivalent
defaults are normalized for comparison. Non-default PP/swap/cache/chunk/mixed policies
and supplied prompt identities must be represented, not silently discarded as
irrelevant metadata.

See [Prefix Cache and Chunked Prefill](prefix-cache-and-chunking.md) for field
defaults, cache lifecycle, result fields and examples.
See [mixed batching](mixed-batching.md) for member-phase and joint-Profile contracts.
See [multi-replica routing](multi-replica.md) for placement, resource scope,
dispatch policies and per-replica diagnostics.

## Independent Workspace and capacity evidence

Workspace calibration is independent of stage duration. Complete Runtime-owned
allocation/lease events deduplicate physical-storage aliases, preserve pending
frees until completion and explicitly complete each rank. The canonical scope
excludes fixed weight/KV/graph/Runtime/PP/communication storage and includes
non-Torch scratch. Partial manager observation and allocator deltas remain raw.
Selected Workspace Profiles are exact-only and retain local replica identity,
source/version/SHA and immutable CLI snapshots.

Capacity collection gives every axis candidate/repeat a fresh process/Runtime.
Input-token, request-cohort and KV-pool axes preserve effective configuration
evidence; existing prompt IDs may only be truncated and KV-pool application
requires actual Runtime storage proof. Prediction simulates the entire effective
experiment. Typed verified-device allocator OOM requires a scoped preparation/execution
phase and allocator `num_ooms` evidence, and is classified separately from
admission rejection, lock errors, timeout and missing peers. Validation reports
repeat instability, nonmonotonic/censored evidence and adjacent success-to-OOM
brackets with resolution, never an inferred exact universal knee. See the
[full contract](workspace-and-capacity.md).
