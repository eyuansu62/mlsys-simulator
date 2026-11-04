# Online PP and CPU KV swap

Version 1.1 adds stage/microbatch pipeline events and a bounded host KV tier.
Both execute in virtual time on the CPU. They model dependencies, occupancy and
transfer costs; they do not allocate device tensors or perform real DMA.

## Run the examples

```bash
mlsys-sim simulate --experiment examples/v1_1/pp_golden.json --out runs/pp-golden
mlsys-sim simulate --experiment examples/v1_1/cpu_swap.json --out runs/cpu-swap
mlsys-sim simulate --experiment examples/v1_1/pp_cpu_swap.json --out runs/pp-cpu-swap
```

These examples use declared synthetic/analytical assumptions. Their purpose is
to expose timelines and resource lifetimes, not establish GPU or PCIe accuracy.
Use a new output directory for each experiment.

## Configure a pipeline

Set `strategy.pipeline_parallel_size` and provide devices in stage-major, then
TP-rank order. `stage_layers` gives each stage's layer count; its sum must match
the model. A stage owns one synchronous TP group, its own queue and compute
resource. Microbatches are homogeneous by default. Since 1.3,
`mixed_batching=true` permits Prefill and Decode in one microbatch when its
sequence and Token budgets allow; `pipeline_microbatch_size` must be at least
two for an actual mix. Different stages can process different microbatches
concurrently. See [mixed batching](mixed-batching.md).

| Strategy field | Default | Meaning |
| --- | --- | --- |
| `pipeline_microbatch_size` | `1` | Maximum sequences in one pipeline microbatch |
| `pipeline_max_inflight_microbatches` | `8` | Bound on outstanding microbatches across the pipeline |
| `pipeline_feedback_latency_us` | `0.0` | Sampled-Token feedback delay before the same request continues |
| `pipeline_p2p` | `true` | Model activation transport; `false` explicitly selects zero-link-cost teaching mode |
| `pipeline_transport` | `"full_hidden"` | Legacy one-tensor/full-rank transport or explicit `"vllm_native"` dual-tensor/sliced P2P and receiver AllGather |
| `pipeline_transport_max_tokens` | `0` | Native physical-token bound; zero resolves to `max_batch_tokens`, nonzero must cover it |

Batch Token limits still apply. The scheduler never uses a request's future
actual output length to choose its work. Each request has at most one outstanding
iteration, including downstream transfer and feedback dependencies. Since 1.2,
each prompt chunk follows that same dependency chain; intermediate chunks emit
no Token and resume input work after feedback. Cache blocks publish at the safe
feedback boundary, and reuse advances every stage's known prefix. See
[Prefix Cache and chunking](prefix-cache-and-chunking.md).

An iteration traverses all stages. Final-stage service already includes the
output/sampling work declared by its performance backend. The Token is ready at
that stage's completion; the request may enter its next Decode only after the
configured feedback delay. Referenced KV remains pinned until feedback is safe,
including for a request that has finished or timed out. A cutoff preserves
in-flight ownership and partial stage progress instead of committing a complete
iteration early.

## Communication and activation memory

Each adjacent-stage boundary has a serialized activation-transfer resource.
With the default `full_hidden` transport, TP ranks use paired independent links
with replicated per-rank activations. A rank's payload is `sum(query_tokens) × hidden_size × activation_bytes`;
neither dividing it by TP nor multiplying a single link's latency by TP matches
this model. Boundaries do not compete for a shared fabric. These link/topology
assumptions must match the experiment being interpreted.

Explicit `vllm_native` transport instead retains `hidden_states` and `residual`
physical shapes/dtypes, including padding. Each tensor uses a TP wire slice only
when its element count is divisible by TP; reconstructed slices require receiver
AllGather. Those costs are separate from isolated stage Profile time and are
charged once using the same rule in `predict` and the online engine. They use
an explicitly selected independent communication Profile or remain analytical;
upstream-dependent receive waiting is never used as the primitive measurement.
See [native PP sampling and calibration](native-pp-capture.md) and the separate
[independent communication Profile](native-pp-communication.md) workflow.

Before stage 0 launches an iteration, the planner conservatively reserves its
target KV length on every stage/rank. It also budgets persistent activation
storage per rank:

```text
adjacent_boundaries(stage)
× pipeline_max_inflight_microbatches
× max_batch_tokens
× hidden_size
× activation_bytes
```

An endpoint has one adjacent boundary, an interior stage has two, and PP=1 has
none. Activations are replicated per TP rank in the default model. Native transport instead uses the
effective `pipeline_transport_max_tokens` bound and distinguishes output,
receive/reconstruction backing and the persistent input-copy destination.
With `H = bound × hidden_size × activation_bytes` and `I` inflight waves,
each outgoing boundary reserves `I × 2H`. Each incoming boundary reserves
`I × (2H + B)` where `B` is the retained full backing of sliced tensors, plus
one fixed `2H` input-copy target per receiving stage. `B=2H` when both tensors
are sliced and zero when neither is sliced. Interior stages sum incoming and
outgoing reservations. Wire slicing never divides this budget by TP; see the
[native allocation contract](native-pp-capture.md#charge-transport-once-and-reserve-full-buffers).
This budget is separate from
`communication_buffer_bytes`; it stays reserved rather than tracking only the
currently live tensor bytes. Large inflight/Token caps can therefore make a
layout infeasible even with few active requests. Reduce the configured bounds
when a smaller working set is intended.

Whole-stage Profile times already include local model collectives. PP boundary
P2P, native receiver AllGather and explicit feedback are separate. Native stage
Profiles preserve the physical transport descriptor and are exact-only. Profile queries preserve phase, stage and
microbatch shape, with the same strict domain rules as other execution paths.
Mixed shapes additionally preserve the ordered `request_phases` array and
require exact mixed observations. One mixed service is estimated per stage;
P2P payload still uses all member queries. Prefill members only sample after
their final input chunk, while Decode members can commit their next output.

## Read an ideal timeline

For `p` identical stages taking `tau` per forward microbatch, `m` independent
microbatches, zero P2P cost and zero feedback cost, the ideal forward makespan is
`(m + p - 1) × tau`. Thus two 1ms stages and three microbatches give 4ms. The
golden example explicitly selects the ideal communication mode.

This formula is a hand-checkable special case. Autoregressive feedback,
different phases/shapes, limited inflight work, stage imbalance, links and
memory pressure change the timeline. `result.pipeline` reports stage activity,
idle/bubble diagnostics and in-flight progress; interpret these with the stated
measurement window. `predict` analyzes a shape only and supplies no online
queueing/SLA result, even though `simulate` now supports online PP.

`stage_busy_ms`, `stage_queue_wait_ms` and `stage_utilization` are per-stage
arrays. `idle_fraction` includes arrival idle time, pipeline fill/drain, feedback
and pressure stalls; it is not an isolated fill/drain-bubble measurement.
`inflight_microbatches` and request `stage_computed_tokens` preserve cutoff
progress. PP `compute_busy_ms` is aggregate stage time, so it can exceed elapsed
wall-clock-equivalent simulated time when stages run concurrently.

## Select recompute or CPU swap

`strategy.preemption` accepts `none`, `recompute` or `swap`. Recompute releases
safe GPU KV and later rebuilds the known prefix. Swap preserves computed KV in
a host pool and restores it without recomputation or repeated historical output.

| Memory/strategy field | Default | Meaning |
| --- | --- | --- |
| `memory.cpu_swap_bytes` | `0` | Global host KV capacity in bytes; must be positive for swap |
| `memory.swap_d2h_bandwidth_gbps` | `12.0` | Modeled decimal GB/s for device-to-host copy |
| `memory.swap_h2d_bandwidth_gbps` | `12.0` | Modeled decimal GB/s for host-to-device copy |
| `memory.swap_latency_us` | `10.0` | Fixed latency charged to each transfer |
| `swap_fallback` | `none` | Explicit `recompute` permits that fallback when host space cannot be reserved |

The host capacity covers block-aligned KV bytes summed across all pipeline
stages and TP ranks, including replicated KV. It is not a per-rank capacity.
Both directions share one serialized DMA lane. A copy duration is
`latency_us/1000 + bytes/(bandwidth_gbps × 1e6)` milliseconds, plus any lane queue
wait. Bandwidth values are modeling inputs, not measured PCIe capability.

## Preserve source and destination lifetimes

1. **Swap out:** reserve host capacity while GPU KV remains allocated. Only D2H
   completion makes the host copy usable and releases the GPU logical blocks.
2. **Swap in:** reserve the GPU destination atomically on all ranks before H2D.
   Retain the host copy through H2D completion, then free it and resume execution.
3. **Timeout/cancel:** active or queued transfer references stay pinned until
   their completion boundary. A host-only copy with no pending transfer can be
   released immediately. A cutoff reports pending ownership and elapsed copy
   work without pretending the copy completed.

PP pressure stops new launches and drains stage work, boundary transfers and
feedback before reclaiming a whole request. During DMA no new GPU work starts;
there is no modeled compute/copy overlap. This is conservative, not a claim
about a Runtime's stream-level concurrency.

Host exhaustion is explicit (`HOST_SWAP_CAPACITY_EXCEEDED`); recompute is not
silently substituted. GPU destination pressure can delay or reject restoration
according to the engine's bounded progress rules. Invalid, duplicate or premature
transfer completions are rejected without changing ownership.

## Inspect migration results

Request rows retain TTFT/TPOT/ITL semantics and add `swap_out_count`,
`swap_in_count`, `swap_out_bytes`, `swap_in_bytes`, `swap_ms` and `kv_residency`.
Counts and completed bytes commit only when the completion event is handled.
At an event-budget cutoff, a fully elapsed but uncommitted copy remains in the
pending diagnostics and retains its references.
Counts and byte totals represent completed physical copies, including a
cancelled request's copy if it reached completion. `swap_ms` includes actual
executed transfer time, both complete copies and partial work at cutoff; it does
not pretend a pending copy completed. Per-copy latency is part of that active
transfer time, while queue waiting remains in request latency.
Aggregate `result.swap` records host capacity/reservation peaks, transfer bytes/
time, pending operations and entries. A transfer or feedback gap remains part of
request latency; neither policy resets its original arrival or first-Token time.

Use `completed_transfers_by_direction` and `completed_bytes_by_direction` for
finished DMA work; pending bytes and `dma_busy_ms` also report executed in-flight
fragments. Pending capacity remains reserved even when only part of the payload
has been copied.

Logical GPU block release still does not shrink the fixed physical KV pool, so
swap cannot automatically solve an unrelated Workspace OOM. With Prefix Cache,
swap conservatively copies all block-aligned request KV, including shared
blocks; it detaches GPU references on safe D2H completion and restores private
GPU blocks. Retained GPU cache entries remain subject to idle-block LRU. Partial
prefill/recompute resumes its remaining input rather than entering Decode early.
With multiple replicas, these GPU/host capacities and DMA lanes belong to each
replica independently; the global coordinator retains shared clock and budgets.
See [multi-replica routing](multi-replica.md). NVMe and quantized storage remain
outside the supported scope. Actual transfer measurements and hardware holdout
validation remain separate work.
