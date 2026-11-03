# Version 1.3 capability matrix

This table describes the version 1.3 software scope. Available fixture demonstrations and
unit tests do not establish AMD hardware accuracy.

| Capability | Software status | Supported scope and evidence boundary |
| --- | --- | --- |
| Six inputs and strict configuration | Implemented | JSON/TOML, JSONL workload, relative references; optional installed PyYAML |
| C01 macro performance model | Implemented | Shape-aware Dense MHA/GQA costs; analytical parameters are uncalibrated |
| C02 memory planning | Implemented | Per-rank physical budget, fixed KV pool, logical blocks and labeled Workspace source |
| C03 serving DES | Implemented | FIFO, default prefill-first batching or opt-in decode-first chunked/mixed scheduling, Token events, timeout and bounded execution |
| KV preemption/recovery | Implemented | Safe request boundary, recompute or explicit CPU swap, output history preserved |
| Warmup and initial state | Implemented | Isolated successful warmup; measurement starts with empty KV only |
| Arrival window and drain | Implemented | Half-open admission, NOT_ADMITTED outcomes, optional bounded drain |
| TP | Implemented | TP 1/2/4 in each stage, GQA replication and atomic all-stage/rank reservation |
| C04 communication analysis | Implemented | Ring AllReduce, logits AllGather and PP P2P, no shared-link contention/overlap |
| C04 communication calibration | Implemented | Per-primitive/rank alpha-beta fit and explicit Infra patch; no automatic double addition to stage profiles |
| PP analysis | Implemented | Shape-only capacity/forward/P2P; online metrics remain null because analysis has no queue |
| Online PP | Implemented | Stage/microbatch queues, boundary P2P, one outstanding iteration/request and sampled-Token feedback |
| PP activation budgets | Implemented | Conservative persistent per-rank buffers and all-stage KV reservation before stage 0 |
| CPU KV swap | Implemented | Bounded global host tier, serial D2H/H2D, safe source/destination ownership, explicit fallback |
| PP pressure and DMA | Implemented | Drain in-flight waves before reclaim; no new GPU work during DMA, conservative no-overlap policy |
| Versioned step Profiles | Implemented | Strict identity/source, exact ragged/mixed lookup, complete homogeneous interpolation cells, explicit homogeneous exploratory extrapolation |
| Calibration and lineage | Implemented | Repeated-shape mean time/max Workspace, new versions, input and training provenance |
| Grouped holdout evaluation | Implemented | Shape/group connected components, leakage rejection, coverage and latency errors |
| Online run/candidate comparison | Implemented | Matching contracts, per-metric/request errors, SLA agreement, ties and candidate ordering |
| Measured Workspace consumption | Implemented | Supported observed values; missing/extrapolated values use labeled analytical assumptions |
| AMD measurement and accuracy | Not established | No AMD dataset supplied; actual collection and held-out hardware evidence remain required |
| Prefix Cache | Implemented | Explicit token ancestry/namespace, full shared KV blocks in the fixed pool, reference pinning and bounded idle-block LRU |
| Chunked Prefill/recompute | Implemented | Retained contexts, final-chunk-only output, decode priority; single-group and TP×PP recovery |
| Mixed Prefill/Decode batches | Implemented | Opt-in shared stage execution, decode-first packing, ordered member roles, joint Workspace and exact mixed Profiles; PP/cache/swap lifetimes preserved |
| Multiple replicas | Unsupported | No replica routing |
| NVMe swap and quantized formats | Unsupported, rejected | CPU swap only; 2/4/8-byte float storage assumptions only |

The shipped calibration data uses `source.kind="test_fixture"`. Fixture opt-in is explicit
and does not grant measured status. Profiles backed by measured records are
calibrated inputs, not an automatic hardware-validation verdict. Actual test and
CI results belong in the validation report and workflow run records.


PP and CPU swap have no supplied hardware validation. Their virtual links/DMA
use declared rates and explicit conservative policies; successful synthetic
examples do not establish actual multi-GPU throughput or PCIe copy accuracy.
See [pipeline and swap](pipeline-and-swap.md) for configuration and lifetimes.

Prefix/chunk behavior is a declared simulator policy, not a reproduction of a
particular Runtime implementation. Cached-prefix and chunked Profile queries
need matching context-domain evidence. See [Prefix Cache and chunking](prefix-cache-and-chunking.md).

Mixed execution uses an uncalibrated joint macro model, declared synthetic
service or exact mixed-stage records. It does not execute a GPU kernel or infer
mixed latency from homogeneous measurements. See [mixed batching](mixed-batching.md).
