# Multi-replica scheduling and load balancing

Version 1.4 routes arriving requests across independent model replicas on one
global event timeline. Each selected replica runs the existing single-group or
TP×PP scheduler, including Prefix Cache, Chunked Prefill, mixed Prefill/Decode
batches and CPU swap. All execution remains a CPU simulation.

## Run the examples

```bash
mlsys-sim simulate --experiment examples/v1_4/round_robin.json --out runs/round-robin
mlsys-sim simulate --experiment examples/v1_4/prefix_aware.json --out runs/prefix-aware
mlsys-sim simulate --experiment examples/v1_4/replicas_pp_mixed_cache_swap.json --out runs/replicas-pp-mixed-cache-swap
```

Use new output directories. The configurations use synthetic service times.
The first demonstrates independent FIFO work on two replicas. The second puts
a reusable prompt on replica 1, then routes its repeat after both replicas are
idle. The third duplicates the mixed/PP/cache/swap trace from version 1.3 across
two replicas. Its P2P cost is explicitly disabled and each replica allows one
inflight microbatch, making its ownership interactions easy to inspect.

| Example | Observed synthetic result |
| --- | --- |
| `round_robin.json` | Four requests route 0/1/0/1 and finish in 6ms; the same single-replica FIFO work takes 12ms |
| `prefix_aware.json` | Three requests route 0/1/1; the repeat reuses four Tokens on replica 1 and finishes at 7ms |
| `replicas_pp_mixed_cache_swap.json` | Ten requests finish at 53.6584ms, with four mixed batches, 24 reused prefix Tokens, two D2H and two H2D transfers, and zero recomputation |

## Placement and local capacity

Set `strategy.replicas` to a positive integer and provide exactly
`replicas × pipeline_parallel_size × tensor_parallel_size` devices. Placement
is replica-major, then stage-major, then TP rank. For two replicas with PP=2 and
TP=2, devices 0–3 belong to replica 0 and devices 4–7 to replica 1; within each
replica the first two devices form stage 0's TP group.

The model and serving strategy are shared configuration. Devices can have
different declared capacities and effective rates; each replica's model and
memory planner consume only its assigned devices. KV pools, Prefix Cache bounds,
PP activation budgets and host swap capacity apply independently to each
replica. In particular, `memory.cpu_swap_bytes` is a capacity **per replica**,
summed across that replica's stages/ranks. Two replicas configured with 64KiB
each have 128KiB of modeled total host capacity.

Each replica also owns its bidirectional DMA lane. Replicas can execute and
copy independently; shared host-memory bandwidth, network/fabric contention and
inter-replica transfer are not modeled. The no-overlap policy between DMA and
GPU work remains local to each replica.

## Routing policies

`strategy.replica_routing` defaults to `"least_loaded"`:

| Value | Selection rule |
| --- | --- |
| `round_robin` | Advance a deterministic cursor through available replicas |
| `least_loaded` | Minimize unfinished request count, then inflight work, used KV blocks and replica index |
| `prefix_aware` | Among replicas with the minimum unfinished count, prefer the longest local cached prefix; then use inflight work, used blocks and index |

An unfinished request includes waiting, active, recovering and swapped work
that has not reached a terminal state. Inflight work breaks ties even when a
request has timed out but still pins compute or feedback resources. Routing
uses the state visible at arrival; future output lengths belong only to the
execution oracle and do not guide load balancing.

Prefix-aware routing probes explicit token ancestry and namespace without
pinning blocks, changing LRU order or incrementing cache lookup counters.
Missing prompt identities offer no affinity. It never selects a larger
unfinished-request queue merely to gain a cache hit. Actual admission still
performs the normal local lookup and atomic reservation, so a routing probe is
not a guarantee that every observed block will remain available.

A request is routed once and stays on that replica through batching,
preemption, swap and output completion. Prefix entries are not shared between
replicas, and recovery does not migrate a request to another replica. The
single-replica configuration retains its existing serving behavior.

## Global event and failure semantics

All replicas share the experiment's virtual clock, time/event budgets and
measurement window. At a timestamp, completion/release events run before
timeouts, then stable arrival IDs are routed. Local scheduling follows those
events. Zero-duration P2P or feedback events retain their completion priority.
Simultaneous arrivals see earlier routing assignments at that same timestamp.

Out-of-window requests become `NOT_ADMITTED` and do not consume the routing
cursor. A global cutoff can leave a request unassigned; its request row has no
replica index and remains `UNFINISHED`. Inflight compute, feedback and copy
ownership is retained at a cutoff according to the existing local contracts.

A replica that stops on a local resource/no-progress condition becomes
unavailable for later routing. Other available replicas continue serving their
work. The final cluster report preserves the local failure and reports an
unresolved run; successful replicas do not hide unfinished requests. Invalid
configuration, Profile identity or timing-domain errors still fail explicitly.

Warmup uses a separate complete routing timeline. Measurement starts with new
local engines, empty caches and a reset round-robin cursor. Warmup costs and
routing diagnostics remain separate from measured metrics.

## Profiles and shape prediction

Step datasets and Profiles describe **one replica**. Calibrate each replica's
TP/PP stages with its own model/device/software identity. For a multi-replica
Profile-backed experiment use:

```json
{
  "calibration": {
    "backend": "profile",
    "replica_profile_paths": ["profiles/replica-0.json", "profiles/replica-1.json"],
    "profile_version": "deployment-v1"
  }
}
```

The array must contain one path per replica and `profile_path` must be absent
or null. A single-replica run keeps the existing `profile_path` field. Paths
resolve relative to their calibration section file. Each local Profile must
match that replica's ordered devices and timing identity; it is not silently
reused on a different device identity.

CLI runs preserve every raw Profile in `profiles/replica-0.json`,
`profiles/replica-1.json`, and so on. `manifest.replica_profiles` binds replica
index, origin path, snapshot path, SHA256, Profile ID/version and source.
Timing and Workspace consume these same snapshots. Fixture opt-in and source
labels remain explicit; multiple replicas do not turn fixtures into measured
evidence.

Online Compare uses the full cluster identity and routing strategy.
`profile_identity` adds `parallelism.replicas` for multi-replica identities;
single-replica identities retain their earlier structure. Older comparison
contracts with equivalent defaults remain comparable.

`predict_experiment` returns `replica_predictions` for isolated shape analysis
on each local layout, plus memory rows annotated with `replica_index`. It
analyzes the workload shape on each replica without routing requests or modeling
arrival queues. It supplies no online SLA metrics and does not infer cluster
throughput by summing isolated stage latencies.

## Results

Request rows and local events identify `replica_index`. A PP microbatch ID is
unique only within that replica, so pair it with the replica index when reading
a cluster trace. Unassigned request rows use null for their replica index.

`result.replicas` contains each replica's index, device IDs and local result.
`result.routing` records the selected policy, assignments, request counts,
observed peak unfinished counts and unavailable replicas. Assignment records
include arrival time, chosen replica, prior unfinished load and cached-prefix
probe result.
`cached_prefix_tokens` in an assignment is the score used by `prefix_aware`
routing. It is zero for policies that did not probe a prefix and is distinct
from actual reuse, reported by request `prefix_cached_tokens` and aggregate
`prefix_cache.hit_tokens` after local admission.
`unavailable_replicas` contains replicas stopped by a local resource failure;
`unresolved_replicas` also includes healthy replicas whose work was interrupted
by a global budget or window cutoff.

Global completion, latency and SLA metrics are recomputed from all request
states on the common evaluation window. Global percentiles are not averages of
per-replica percentiles. Compute busy time aggregates replica/stage resource
time and can exceed elapsed virtual time. Local memory and copy diagnostics
remain available to explain occupancy or pressure on an individual replica.

Aggregated cache `peak_cached_blocks` and swap `host_peak_reserved_bytes` use
the global high watermarks observed after events and scheduling, labeled
`peak_scope="observed_global_timeline"`. The separate
`sum_replica_peak_cached_blocks` and `sum_replica_host_peak_reserved_bytes`
fields sum local high watermarks and are upper bounds on simultaneous global
use. Current entries, pending copies and rank memory rows retain replica indices.

The simulator has no supplied hardware validation of replica routing overhead,
shared-fabric contention or the resulting load-balancing/SLA predictions.
