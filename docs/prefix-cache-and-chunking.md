# Prefix Cache and Chunked Prefill

Version 1.2 models shared prompt KV blocks and incremental prompt execution in
the online simulator. Both work on a single GPU, a synchronous TP group or a
TP×PP pipeline. All storage, references and execution times are simulated on the
CPU; no GPU tensors, tokenizer or live Runtime cache is involved.

## Run the examples

```bash
mlsys-sim simulate --experiment examples/v1_2/prefix_cache.json --out runs/prefix-cache
mlsys-sim simulate --experiment examples/v1_2/chunked_prefill.json --out runs/chunked-prefill
mlsys-sim simulate --experiment examples/v1_2/pp_prefix_chunk.json --out runs/pp-prefix-chunk
```

Use new output directories. These synthetic examples demonstrate the event and
resource contracts; they are not hardware measurements. The prefix example
repeats a prompt after its producer finishes. The chunking example has an input
longer than its Batch Token budget. The PP example combines both mechanisms.

## Configuration

| Field | Default | Contract |
| --- | --- | --- |
| `strategy.prefix_cache` | `false` | Enable prompt KV lookup, sharing and retention |
| `strategy.prefix_cache_max_blocks` | `128` | Positive bound on resident logical cache blocks across this replica |
| `strategy.chunked_prefill` | `false` | Split Prefill/recompute and use Decode priority in continuous batching |
| `strategy.prefill_chunk_tokens` | `512` | Positive maximum query Tokens per request per chunk |
| `workload[].prompt_token_ids` | `[]` | Explicit token IDs; omitted IDs bypass cache reuse |
| `workload[].cache_namespace` | `""` | Separate identities that must not share cached KV |

Supplied token IDs must be integers in `[0, model.vocab_size)` and their count
must equal `input_tokens`. Prompt length alone cannot identify a prefix.
Namespace is part of every key: callers should use separate namespaces for
tenants or contexts whose KV must not be shared. Model and layout identity are
fixed by the experiment; the cache is not shared across runs or replicas.

Both policies are opt-in. A disabled chunk policy still computes the entire
remaining prompt in one batch after any cache hit, so that residual query must
fit `max_batch_tokens`. `predict` remains an uncached, unchunked shape analysis
of all workload rows; use `simulate` to evaluate these online policies.

## Prefix identity and output readiness

A cache key includes the namespace, block endpoint and complete token ancestry
through that boundary. An incremental SHA256 digest combines the parent digest
with the current block's token IDs, preserving ancestry without retaining every
complete prefix. Key storage therefore grows with the block count rather than
quadratically with prompt length. Model identity is scoped to the experiment.
Equal block suffixes under different earlier tokens do not match.
Only fully computed prompt blocks publish; generated output positions and
incomplete trailing blocks do not become cache entries.

Lookup attaches the longest contiguous resident full-block prefix while leaving
at least the final prompt Token to execute. With a 16-Token block and a 32-Token
prompt, an exact prompt repeat can reuse 16 Tokens, then computes the remaining
16 to produce its first output. A longer prompt may reuse both full blocks.
This model stores KV rather than cached logits, so a complete KV hit cannot
produce an output at zero cost.

Initial lookup happens when a request is scheduled. A cache hit temporarily
attaches references before admission; if admission fails before execution, those
references are rolled back so a waiting request cannot pin cache blocks needed
by resident Decode work. The request may retry lookup on a later admission
attempt. Publishing may deduplicate a block already produced by another
request. Both attachment and publication happen at safe execution boundaries;
neither exposes partially computed KV.

## Memory, references and eviction

Cache entries occupy the existing fixed GPU KV pool. They do not create a
second pool or add used KV bytes to the physical pool allocation twice.
For each rank, logical use is:

```text
unique resident cached blocks
+ sum(request logical blocks - attached shared blocks)
```

One block may have several request references while consuming one block of
per-rank storage. Active references pin that entry. Completion, preemption or
safe migration releases request references and private storage; the cache entry
can remain resident without references. Retained cache blocks after all
requests finish are intentional occupancy, not leaked request KV.

Unreferenced cache entries are evicted in LRU order when the configured cache
bound or a GPU reservation needs room. A memory check is read-only; successful
reservation commits any required eviction atomically across all stages/ranks.
If pinned blocks and private allocations still prevent progress, normal
admission, request preemption and bounded failure rules apply. Eviction cannot
shrink the fixed physical pool or fix a separate Workspace budget failure.

## Chunk execution and scheduling

Each Prefill query is the minimum of remaining input Tokens, the configured
chunk cap and available Batch Token budget. Every chunk records its actual
prefix context. Intermediate chunks advance computed KV and leave the request
in Prefill; only the final prompt chunk commits the first output Token. The
latest sampled Token is processed by the next Decode, as in unchunked serving.

With continuous batching, enabling chunking gives ready Decode work priority
at scheduling boundaries, then continues or admits prompt work. Batches and PP
microbatches remain homogeneous by default. Since 1.3, the independent
`mixed_batching=true` policy packs Decode and Prefill into the same stage
execution while preserving member roles and shared budgets. See
[mixed batching](mixed-batching.md) for its exact Profile contract. FIFO continues to
hold its owner until completion, even when that owner's prompt is chunked.

Recomputation can also be chunked. It rebuilds previously computed KV progress
without re-emitting committed outputs, then resumes the unfinished prompt or
Decode as appropriate. A time/event cutoff does not count an unfinished chunk
as complete or emit its prospective output. Timeout retains in-flight storage
until the relevant execution boundary.

## PP and CPU swap

Each chunk traverses all PP stages, boundary P2P and feedback dependencies. A
request has one outstanding iteration at a time. Prefix hits initialize the
known KV progress of every stage. Publication waits for the safe feedback
boundary, so a newly finished stage cannot prematurely advertise cache blocks
to another request. KV references remain pinned through pending waves,
feedback and timeout handling.

Pressure recovery still drains the pipeline before reclaiming request KV. CPU
swap conservatively copies the entire block-aligned request KV, including
shared blocks, across every stage/rank. D2H completion detaches GPU references;
unpinned GPU cache blocks can remain resident or be evicted independently.
H2D restores private GPU blocks. It does not reconstruct sharing or offer a CPU
Prefix Cache tier. Partial prompt execution resumes its remaining input after
restoration; migration does not emit or duplicate output Tokens. DMA remains
serialized and does not overlap modeled GPU work.

## Timing and Profile domain

Analytical attention uses each chunk's query length and existing prefix context.
Workspace Profile lookups use those same context lengths. Exact ragged lookup,
complete homogeneous interpolation cells and explicit extrapolation rules are
unchanged. A zero-context prefill observation cannot substitute for a missing
nonzero-context shape merely because its query length matches.

The existing stage timing boundary includes output/sampling. Chunk estimates
conservatively charge that service boundary for every chunk, although only the
final prompt chunk emits a Token. Analytical output costs use the same policy;
synthetic phase durations are fixed per batch/chunk. Supply matching chunk
records when using Profiles, and interpret this convention when comparing a
Runtime that avoids intermediate output work. Cache lookup, hashing, metadata
and eviction have no separately measured CPU overhead in this version.

## Results and evidence

Request rows add `prefix_cached_tokens`, `prefill_chunks` and
`prefill_computed_tokens`. Counts commit on completed input work; pending chunks
remain visible in event/PP progress diagnostics. Recomputed work is reported
separately through the existing recompute fields.

`result.prefill` summarizes policy, completed chunks, computed prompt Tokens and
reused prefix Tokens. `result.prefix_cache` reports lookup/hit/miss/bypass
counts, hit Tokens, publication/deduplication/eviction counts and current/peak
occupancy. Its hit rate excludes requests that bypassed identity lookup.
`abandoned_lookups` records provisional lookups rolled back before admission.
Current entries expose prefix length, request references and reference count
rather than the prompt token contents. Cache keys retain digests rather than
complete token prefixes. Per-rank memory snapshots distinguish retained
cached blocks, pinned/shared blocks and private request blocks.

Warmup uses an isolated timeline. Its request allocations and retained prefix
entries are discarded before measurement; `initial_state="empty_cache"`
remains the supported initial-state contract. Comparison contracts include
policy settings and supplied prompt identities so runs with different caching
or chunking assumptions cannot silently compare as the same experiment.

The high-level mechanisms are also described by the primary
[vLLM Prefix Cache documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
and [chunked-prefill tuning documentation](https://docs.vllm.ai/en/latest/configuration/optimization/).
The ownership, scheduling, timing and no-overlap rules above
are this simulator's declared policies, not a claim of Runtime equivalence.
There is no supplied hardware validation of cache overhead, chunk latency or
the resulting throughput/SLA predictions.
