# V1 software specification

The CPU simulator separates system events, performance/communication models and
calibration data. It has no required third-party runtime dependency and performs
no real GPU execution. This specification describes supported software behavior;
hardware prediction accuracy requires separate evidence.

## Configuration and units

Experiment schema version is `1`. Top-level fields are `schema_version` and six
sections, with strict unknown-field/type/range checks:

| Section | Contract |
| --- | --- |
| `workload` | Unique request IDs, nonnegative arrival ms, positive input/output lengths, priority and optional relative timeout |
| `infra` | Ordered devices, memory bytes, effective compute/bandwidth assumptions and named topology/link parameters |
| `strategy` | Dense model/revision, one replica, scheduler, TP/PP/stage layers, Batch limits, preemption and memory policy |
| `calibration` | Analytical/synthetic parameters or Profile path/version, full software identity and explicit fixture/extrapolation policy |
| `evaluation` | Optional TTFT/TPOT SLA, half-open arrival window, drain and optional drain limit |
| `execution` | Seed, time/event limits, detailed logging, isolated warmup requests and `empty_cache` initial state |

Times use ms, byte quantities use bytes, bandwidth uses decimal GB/s and compute
uses TFLOP/s. Float storage widths of 2/4/8 bytes are supported; packed/8-bit
formats are rejected rather than omitting their metadata. Model/device defaults
are toy assumptions. Measured data needs explicit non-placeholder identities.

## Event and scheduling semantics

One GPU or one synchronous TP group executes one homogeneous phase Batch at a
time. FIFO holds one request until it finishes. Continuous batching selects
Prefill work at safe boundaries before Decode, with a pressure-drain policy for
recovery. No Prefix Cache, chunked/mixed-phase prefill or multi-replica routing is
implied by this simplified policy.

The stable heap orders completion/release, timeout, arrival and new scheduling.
Prefill builds input KV and emits the first output. The latest sampled output is
processed by the next Decode, so committed output count and computed KV position
are distinct. The future output target is available only to the execution oracle.

Timeouts do not free in-flight references until their safe boundary. A cutoff
retains in-flight reservations, reports unfinished requests and accounts for the
executed fraction of a Step without committing an unfinished Token.

## Memory and recovery

Each rank maintains separate physical and logical budgets. Physical use includes
weights, the entire fixed KV pool, persistent Runtime/Graph/communication use,
Workspace and safety/external constraints. Logical use counts request blocks,
rounded up by block size. Used KV is not added again to a physically reserved
pool. GQA replication is part of the per-rank shard plan.

All ranks must pass a candidate reservation before any allocation commits.
Workspace uses a supported Profile observation when available; otherwise its
source is explicitly analytical. Releasing request blocks preserves the fixed
pool's physical allocation and cannot by itself solve a Workspace OOM.

KV pressure limits admission and shrinks Batch work. If no active Decode can
advance, a safe low-priority/recently arrived victim can release KV and wait for
recompute. Its known output history remains intact. Recovery rebuilds the prefix
without re-emitting old outputs; delay and compute cost enter the metrics.
Impossible single-request shapes and bounded no-progress/preemption conditions
end explicitly. CPU swap and arbitrary Kernel interruption are outside V1.

## Performance backends

`synthetic` fixes phase Batch times for tests. `analytical` computes serialized
Linear/Attention/Misc, output and communication costs using declared effective
rates. Prefill preserves each sequence length; Decode preserves each context.
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

PP analysis reports stage capacity, isolated Batch service time and P2P. It does
not simulate stage/microbatch queues, bubbles or autoregressive feedback, and
online PP metrics remain unavailable. `simulate` rejects PP greater than one.

## Warmup and evaluation

Warmup uses the same configured model/backend on a separate timeline. It must
finish successfully before measurement; its costs are reported separately and
its KV is discarded. Only an empty initial cache is supported.

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
and provenance, comparison contract, request/event rows and diagnostics. Output
paths are exclusive. A run can execute successfully while requests fail or SLA
criteria are missed.

Step datasets, Profiles and observed runs use strict JSON with explicit source
kind/reference. Calibration produces a new version and records training members;
validation prevents configuration leakage. Online comparison requires equivalent
contracts and keeps missing values/ties explicit. Their formats and acceptance
rules are documented in [calibration](calibration.md) and
[validation](validation-protocol.md). They establish no hardware evidence that
has not actually been supplied.
