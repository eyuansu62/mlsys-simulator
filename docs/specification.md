# Version 1.2 software specification

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
| `strategy` | Dense model/revision, one replica, TP/PP/stage layers, microbatch/inflight/feedback/link policy, Batch/chunk limits, Prefix Cache and preemption/fallback |
| `calibration` | Analytical/synthetic parameters or Profile path/version, full software identity and explicit fixture/extrapolation policy |
| `evaluation` | Optional TTFT/TPOT SLA, half-open arrival window, drain and optional drain limit |
| `execution` | Seed, time/event limits, detailed logging, isolated warmup requests and `empty_cache` initial state |

Times use ms, byte quantities use bytes, bandwidth uses decimal GB/s and compute
uses TFLOP/s. Float storage widths of 2/4/8 bytes are supported; packed/8-bit
formats are rejected rather than omitting their metadata. Model/device defaults
are toy assumptions. Measured data needs explicit non-placeholder identities.

## Event and scheduling semantics

For PP=1, one GPU or synchronous TP group executes one homogeneous phase Batch
at a time. FIFO holds one request until it finishes. Continuous batching selects
Prefill at safe boundaries before Decode by default, with a pressure-drain
policy for recovery. Enabling Chunked Prefill instead gives Decode priority;
prompt queries retain context and respect the configured per-chunk and Batch
Token budgets. PP>1 uses independent stage queues/resources and homogeneous
microbatches; different stages can execute different microbatches concurrently.
Each stage owns a TP group. Every prefill chunk is a separate iteration through
the stages and feedback boundary. Mixed-phase kernels and multi-replica routing
remain unsupported.

The stable heap orders completion/release, timeout, arrival and new scheduling.
Prefill builds input KV and emits the first output only when the final prompt
chunk completes. The latest sampled output is processed by the next Decode, so committed output count and computed KV position
are distinct. The future output target is available only to the execution oracle.

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
Endpoints have one boundary, interior stages two and PP=1 none; activations are
replicated per TP rank. This is separate from collective communication buffers.
Workspace uses a supported Profile observation when available; otherwise its
source is explicitly analytical. Releasing request blocks preserves the fixed
pool's physical allocation and cannot by itself solve a Workspace OOM.

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
Timeouts and cutoffs retain pending source/destination references. Host-capacity
failure is explicit; only `swap_fallback="recompute"` allows recompute fallback.
Impossible shapes and bounded no-progress/preemption conditions end explicitly.
NVMe and arbitrary Kernel interruption remain unsupported.

## Performance backends

`synthetic` fixes phase Batch times for tests. `analytical` computes serialized
Linear/Attention/Misc, output and communication costs using declared effective
rates. Prefill/recompute preserve actual query and prefix-context lengths;
Decode preserves each context. Per-chunk stage service conservatively includes
the declared output/sampling boundary even for an intermediate prompt chunk
that emits no Token. Synthetic durations apply per phase batch/chunk.
TP uses the supported Dense shard plan, local KV replication, Ring AllReduce and
logits AllGather, without communication overlap or shared-link contention.

`profile` snapshots a versioned whole-stage lookup and checks identity. Ragged
shapes require exact observations; homogeneous grid interpolation requires every
corner. Missing timing does not silently fall back to analytical estimates.
Explicit extrapolation is flagged. Local collectives are already included in
stage times; PP boundary P2P is separate. See [calibration](calibration.md).

Independent communication observations can fit alpha-beta parameters per
primitive/rank group and produce an explicit Infra patch. A selected patch must
match the intended topology and workload model; it is never automatically
summed with whole-stage Profile timings.

Online PP uses independent stage compute resources and a serialized activation
link for each adjacent boundary. TP boundary links are paired and independent,
with replicated per-rank payload; shared fabric contention is not modeled.
`pipeline_p2p=False` explicitly disables link cost for ideal teaching cases;
default `True` includes it. Explicit feedback remains separate from stage costs.

`predict` continues to report stage capacity, isolated service time and P2P only.
It performs no prefix lookup or chunk scheduling. It has no arrival queue, so
online metrics are null with
`SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING`. Use `simulate` for the PP timeline.
Detailed configuration, lifetimes and ideal bubbles are described in
[pipeline and swap](pipeline-and-swap.md).

## Warmup and evaluation

Warmup uses the same configured model/backend on a separate timeline. It must
finish successfully before measurement; its costs are reported separately and
its request KV and retained prefix entries are discarded. Measurement starts
with an empty cache; warmup does not seed measured prefix hits.

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

## Data and result contracts

Simulation output retains effective inputs, code/input hashes, Profile snapshot
and provenance, comparison contract, request/event rows and diagnostics. PP adds
stage activity/bubble and progress reports; swap adds host/DMA state and
per-request copy counts, bytes and time. Prefix/chunk reports add cache
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
defaults are normalized for comparison. Non-default PP/swap/cache/chunk policies
and supplied prompt identities must be represented, not silently discarded as
irrelevant metadata.

See [Prefix Cache and Chunked Prefill](prefix-cache-and-chunking.md) for field
defaults, cache lifecycle, result fields and examples.
