# Limited live policy and replica observations

The initial Runtime correctness probes and later independent conditional latency
repeats answer separate questions. The correctness probes keep
`performance_validated=false`, `sla_validated=false` and
`hardware_validated=false`. The narrow latency results below do not grant
general serving accuracy, throughput, tail latency, SLA or policy-ranking
acceptance; overall hardware status remains false.

## Single-card policies

Each predeclared probe started a fresh Qwen3-1.7B BF16 synchronous eager MRv1
on RX9070XT, with actual Runner and Scheduler exports entering the public
normalizer. Output lengths were checked against the four requested greedy
tokens. Policy settings were fixed before acquisition and remained unchanged
after observing the results.

| Policy | Actual observation |
| --- | --- |
| Prefix Cache | After a 64-token seed request, a new identical prompt had 48 computed context tokens and 16 remaining query tokens: three full 16-token KV blocks were reused. Both requests finished. |
| Chunked Prefill | A 64-token prompt with a 16-token iteration budget had three partial prefill steps with no sampled token, then one final prefill step and three decode steps. |
| Mixed batching | Two concurrent prompts of lengths8/64, max sequences2 and token budget16, produced three executions containing both a decode member and a partial-prefill member. Both requests finished with four outputs. |

These prove that the measured paths occurred. A short successful probe does
not show that the simulator reproduces their queueing or service distribution.
Partial prefill, final sampling, cache reuse and mixed member roles must remain
correctly distinguished throughout an accuracy evaluation.

## Independent conditional Prefix Cache and chunking repeats

Separate training acquisitions produced policy-specific stage Profiles. Both
were frozen at **2026-11-04 17:10:15 UTC**, before the six new Runtime sessions:

- Prefix Cache Profile SHA256:
  `dcde118d3359c9a9d980187b01a2da58ccd068b543ae8fd7adb002851f86c149`
- Chunked Prefill Profile SHA256:
  `3f1c682acaba3019d1d562e4b1bc26edb9f483639e7c37940fa2088886e84080`

Validation ran from **17:10:48 to 17:13:14 UTC**. Each policy used three fresh
Runtimes, without retrying, refitting or changing the frozen Profile bytes.
Each Prefix Cache repeat retains the seed and subsequent reused-prefix request
(six requests total); each chunking repeat retains its one request (three
total). All **45 formal stage rows** had exact Profile coverage, and all **9/9
requests** paired with the observed cohort.

The comparison uses each new cohort's actual arrivals. In the Prefix Cache
case, the second arrival follows completion of the real seed request, so the
arrival schedule is partly conditioned on observed execution. Seed work is
retained in both sides; it is not dropped to improve the cache-hit metrics.
This is a short fixed-cohort conditional replay, not an open-loop throughput
experiment.

| Policy / independent repeat | TTFT p50 relative error | TPOT p50 relative error | E2E p50 relative error |
| --- | ---: | ---: | ---: |
| Prefix Cache 1 | 1.1345% | 2.5474% | 0.9689% |
| Prefix Cache 2 | 8.9372% | 7.8536% | 8.3312% |
| Prefix Cache 3 | 0.8793% | 2.1476% | 0.8450% |
| Chunked Prefill 1 | 4.2252% | 8.9311% | 5.5640% |
| Chunked Prefill 2 | 2.6750% | 1.4121% | 2.3076% |
| Chunked Prefill 3 | 3.1030% | 6.6726% | 4.1259% |

All six repeats met the predeclared coverage≥95% and each listed p50 relative
error≤15% targets. The samples are too small to establish tail distributions,
SLA agreement, policy ordering or broad workload performance. These Profiles
and results remain bound to their actual policy/configuration and exact shapes.

### Mixed-policy coverage gap

The real mixed training cohort began with a two-member prefill batch with query
lengths `[8,8]`. Under its real microsecond-separated arrivals, the simulator
starts the first request immediately as a solo `[8]` prefill. The corresponding
solo shape was absent from that exact training Profile, so prediction correctly
reported an out-of-domain coverage gap. The real arrivals were not changed to
force a matching batch and the missing shape was not fitted from validation.
The earlier actual mixed execution remains a correctness observation; independent
mixed-policy accuracy has not passed.

## Two independent physical replicas

Two fresh TP1×PP1 instances ran simultaneously, one on RX7900XTX and one on
RX9070XT. A stdlib-only supervisor assigned the fixed four requests of lengths
8/16/8/16 by round robin to replicas0/1/0/1. Each worker had its own physical
visibility mask, process group, Runtime, session and IPC channel.

All four requests completed with four greedy output tokens. The real native
IDs returned by the pinned engine API were mapped to router IDs and checked
against the actual Runner and Scheduler request sets. Each instance's raw
capture was normalized separately; both were stage-ready and canonically
comparable within that instance. The supervisor did not relabel independent
sessions as one native distributed Runtime domain.

The source was frozen before execution, SHA256
`89fa609715e92f75a5c7637332d4f8c766bc24be6bba628863c2d7f4c06a9f74`.
The owned supervisor ran from 2026-11-04 16:53:09.978733 UTC to
16:53:37.035693 UTC and returned0. Router and IPC timestamps retain their own
meaning; they are not substituted for server token-ready events.

This establishes fixed routing, ownership and output completion only. It does
not validate least-loaded or prefix-aware choices, routing cost, cross-replica
contention or latency benefit. There are only two physical GPUs: TP2×PP2 and
R2×TP2×PP2 remain outside this host's available topology.

The pinned installed vLLM does not provide the native CPU KV-swap path needed
to verify the simulator's evict/restore semantics end to end. Recompute, host
model offload and a standalone DMA copy would not establish that contract.
CPU swap therefore remains a separate uncompleted real-Runtime validation.

All sources, raw captures, request-ID maps and execution receipts are retained
in the private evidence directory. No prompt text, credentials or connection
addresses are distributed with this report.
