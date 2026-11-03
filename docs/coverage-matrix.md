# V1 capability matrix

This table describes the V1 software scope. Available fixture demonstrations and
unit tests do not establish AMD hardware accuracy.

| Capability | Software status | Supported scope and evidence boundary |
| --- | --- | --- |
| Six inputs and strict configuration | Implemented | JSON/TOML, JSONL workload, relative references; optional installed PyYAML |
| C01 macro performance model | Implemented | Shape-aware Dense MHA/GQA costs; analytical parameters are uncalibrated |
| C02 memory planning | Implemented | Per-rank physical budget, fixed KV pool, logical blocks and labeled Workspace source |
| C03 serving DES | Implemented | FIFO, prefill-first batching, Token events, timeout and bounded execution |
| KV preemption/recovery | Implemented | Safe request boundary, block release and prefix recompute preserving output history |
| Warmup and initial state | Implemented | Isolated successful warmup; measurement starts with empty KV only |
| Arrival window and drain | Implemented | Half-open admission, NOT_ADMITTED outcomes, optional bounded drain |
| TP | Implemented | One synchronous group, TP 1/2/4, GQA replication and atomic reservation |
| C04 communication analysis | Implemented | Ring AllReduce, logits AllGather and PP P2P, no shared-link contention/overlap |
| C04 communication calibration | Implemented | Per-primitive/rank alpha-beta fit and explicit Infra patch; no automatic double addition to stage profiles |
| PP analysis | Implemented | Stage capacity, isolated forward cost and P2P; no online PP SLA |
| Versioned step Profiles | Implemented | Strict identity/source, exact ragged lookup, complete homogeneous interpolation cells, explicit exploratory extrapolation |
| Calibration and lineage | Implemented | Repeated-shape mean time/max Workspace, new versions, input and training provenance |
| Grouped holdout evaluation | Implemented | Shape/group connected components, leakage rejection, coverage and latency errors |
| Online run/candidate comparison | Implemented | Matching contracts, per-metric/request errors, SLA agreement, ties and candidate ordering |
| Measured Workspace consumption | Implemented | Supported observed values; missing/extrapolated values use labeled analytical assumptions |
| AMD measurement and accuracy | Not established | No AMD dataset supplied; actual collection and held-out hardware evidence remain required |
| Online PP | Outside V1, rejected | Future stage/microbatch/feedback event model |
| Multiple replicas, chunked prefill, Prefix Cache | Outside V1, rejected | No silent fallback |
| CPU/NVMe swap and quantized formats | Outside V1, rejected | Recompute recovery; 2/4/8-byte float storage assumptions only |

The shipped V1 data uses `source.kind="test_fixture"`. Fixture opt-in is explicit
and does not grant measured status. Profiles backed by measured records are
calibrated inputs, not an automatic hardware-validation verdict. Actual test and
CI results belong in the validation report and workflow run records.
