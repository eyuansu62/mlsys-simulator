# Mixed Prefill/Decode batching

Version 1.3 can put Prefill and Decode members into one stage batch. The online
simulation retains each member's phase, context, KV target and output rules.
This is a CPU scheduling and service model, with no live GPU kernel execution.

## Run the examples

```bash
mlsys-sim simulate --experiment examples/v1_3/mixed_batching.json --out runs/mixed-batching
mlsys-sim simulate --experiment examples/v1_3/pp_mixed_cache_swap.json --out runs/pp-mixed-cache-swap
```

Use new output directories. The first example exposes joint batches with small
synthetic times. The second combines a PP pipeline, shared prompt identities,
chunking, mixed execution and a bounded GPU/CPU KV budget. They are software
fixtures, not GPU timing or throughput measurements.

| Example | Deterministic synthetic result |
| --- | --- |
| `mixed_batching.json` | 2 requests complete, 2 mixed batches, 6ms mixed compute, 9ms makespan |
| `pp_mixed_cache_swap.json` | 5 requests complete, 2 mixed batches, 12 reused prefix Tokens, 1 D2H and 1 H2D copy, 53.6584ms makespan, no recomputation |

The combined example deliberately uses one inflight microbatch and an explicit
zero-cost P2P link to make the resource interaction reproducible. Other PP link
and feedback settings retain their configured costs.

## Configuration and packing

| Field | Default | Contract |
| --- | --- | --- |
| `strategy.mixed_batching` | `false` | Permit Prefill and Decode in the same online batch |
| `strategy.max_batch_size` | `8` | Shared sequence limit across both member roles |
| `strategy.max_batch_tokens` | `4096` | Shared query Token budget, counting every Decode query |
| `strategy.chunked_prefill` | `false` | Independently permit bounded partial-prompt queries |
| `strategy.prefill_chunk_tokens` | `512` | Per-request Prefill chunk cap when chunking is enabled |
| `strategy.pipeline_microbatch_size` | `1` | PP sequence cap; at least two is needed for an actual mixed microbatch |
| `calibration.mixed_ms` | `10.0` | Positive fixed synthetic service time per mixed batch before stage apportionment |

In continuous batching, the mixed policy packs eligible Decode work first.
Each Decode member uses one query Token. Remaining sequence/Token capacity can
hold partial Prefill and then eligible fresh prompts. Fresh admission remains
restricted while memory pressure drains existing residents. With chunking,
each selected prompt query is bounded by remaining work, its chunk cap and the
remaining shared budget. Without chunking, the complete uncached suffix must
fit; enabling mixed batching alone does not truncate a prompt.

FIFO preserves its single owner and never mixes requests. A one-member batch
or a batch whose members all have the same phase uses ordinary homogeneous
service. Recompute remains a separate homogeneous recovery batch; rebuilding
KV does not become new sampled output.

## Joint execution and member completion

A mixed selection reserves all member KV targets atomically, after checking
the complete joint Workspace. One stage service estimate covers the whole
selection. Mixed events preserve ordered `request_phases`, `query_tokens` and
`context_lengths`; each array entry identifies the same request.

At completion, a Decode member commits its next output normally. A Prefill
member advances only its computed input progress and commits its first output
only after its final prompt chunk. The presence of Decode work does not make
an intermediate Prefill chunk emit a Token. Output targets remain confined to
the replay execution oracle and do not guide packing.

When a complete joint batch exceeds Workspace capacity, resource admission may
reduce the selection. It must not retain unlaunched prefix-cache references or
reserve a fragment of a rejected batch. Missing mixed timing data is a domain
error, rather than a reason to fabricate service from pure-phase timings.

## PP, Prefix Cache and CPU swap

Every member of a mixed PP microbatch traverses the same stage sequence. Each
stage receives one joint service estimate; boundary activation payload remains
`sum(query_tokens) × hidden_size × activation_bytes` per TP rank. TP/PP placement,
inflight limits, persistent activation budgets and all-stage atomic KV
reservation remain in force.

A request still has one outstanding iteration through compute, P2P and
feedback. Its next iteration waits for feedback even if another member became
ready earlier. Cache publication waits for the safe feedback boundary and uses
only that member's completed prompt blocks. Final or timed-out members retain
their references until the wave's normal release boundary. A cutoff retains
pending ownership and partial work.

Prefix identity still uses explicit token IDs, namespace and complete ancestry.
Failed admission rolls back provisional cache references so waiting prompts
cannot prevent a resident Decode from progressing. Cached and private blocks
share the same fixed KV pool; unreferenced cached entries remain evictable.

PP recovery drains in-flight waves before reclaiming a request. CPU swap still
pins source and destination through copy completion, copies complete
block-aligned request KV and restores private GPU blocks. Partial Prefill
resumes its input after recovery, while Decode retains its output history. DMA
is serialized and does not overlap stage execution in this model.
With multiple replicas, every local scheduler applies these same rules after
one global arrival route. Mixed microbatches never span replicas; resource
budgets and host/DMA lanes remain independent. Global metrics use the common
evaluation window. See [multi-replica routing](multi-replica.md).

## Timing and Profile contract

The analytical backend combines all queries in one macro estimate. It assumes
ideal common weight reuse across member roles and one set of macro launch and
collective costs. Attention still uses each member's own query and prefix
context. This is an uncalibrated model of joint execution, not a sum or maximum
of independently estimated Prefill and Decode times.

The synthetic backend uses independent `mixed_ms`, apportioned by stage layer
count. Pure batches retain `prefill_ms`, `decode_ms` or `recompute_ms`.
Output/sampling service is conservatively charged for every sequence, including
an intermediate prompt chunk that does not emit a Token.

A mixed observation uses `phase="mixed"` and an ordered `request_phases` array:

```json
{
  "record_id": "joint-stage-0",
  "group_id": "joint-shape-a",
  "phase": "mixed",
  "stage_index": 0,
  "request_phases": ["decode", "prefill"],
  "query_tokens": [1, 4],
  "context_lengths": [32, 8],
  "duration_ms": 2.5,
  "workspace_peak_bytes": 4096
}
```

This illustrative record belongs inside a dataset with full identity and source
metadata. It is not a measurement. Both serving roles must be present, one per
query; Decode queries equal one, while Prefill queries can also equal one and
must retain their explicit role. Homogeneous observations omit `request_phases`.

Mixed Profiles require an exact ordered role/query/context/stage match. They do
not interpolate or extrapolate, even when homogeneous exploratory extrapolation
is enabled. Pure-phase observations cannot substitute for a missing mixed
shape. Workspace uses the same joint shape. When a matching sample declares
`workspace_peak_bytes=null`, the existing labeled analytical Workspace
assumption applies; a missing mixed shape is still a domain error.

Legacy homogeneous shape keys remain four elements. Mixed keys append a fifth
element containing ordered roles, under the additive schema version `1`
contract. Calibration retains this metadata in repeat grouping and samples.
Holdout connected components and leakage checks include the same role-aware
shape. Whole-stage timings include local collectives; PP boundary P2P and
feedback remain separate.

## Results and interpretation

`result.batching` reports whether mixing is enabled, admitted and fully serviced
mixed batches, executed mixed compute time and the packing/recovery policy.
Its fields are `mixed_batching`, `mixed_batches_admitted`,
`mixed_batches_completed`, `mixed_compute_busy_ms`, `compute_busy_ms_semantics`,
`policy` and `recompute_policy`.
`mixed_batches_completed` commits after a single-group step or the final PP
stage completes; a wave can still retain feedback references after that service
boundary. `mixed_compute_busy_ms` includes executed fragments at a cutoff and
uses aggregate stage time for PP, consistent with `compute_busy_ms_semantics`.
It is not an attribution of joint service to either member phase.

Request prompt-work, cache, output and swap fields retain their existing
meanings. Detailed events show which members shared a service operation.
Disabling event logs does not change scheduling or counters. `predict` remains
separate Prefill/Decode shape analysis and does not execute online mixing.
Comparison contracts preserve the mixed policy rather than treating different
scheduling assumptions as equivalent.

The simulator has no supplied hardware validation of mixed-kernel efficiency,
cache metadata overhead, stream overlap or the resulting SLA/throughput
predictions. These remain distinct from deterministic software validation.
