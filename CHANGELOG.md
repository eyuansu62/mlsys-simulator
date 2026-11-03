# Changelog

## 1.3.0 — 2026-11-04

Prefill and Decode can share a stage batch while retaining their separate
progress, output and KV ownership rules. This extends the existing CPU model;
hardware-validation and license-selection status remain unchanged.

### Added

- Opt-in `mixed_batching` with Decode-first packing, shared sequence/Token
  budgets and member-specific completion in single-group and TP×PP execution.
- Joint analytical macro service and independent synthetic `mixed_ms` timing.
- Exact mixed Profile records with ordered `request_phases`; calibration,
  Workspace and holdout provenance preserve the full member-aware shape.
- Mixed examples, event metadata and execution diagnostics with Prefix Cache,
  Chunked Prefill and safe CPU swap/recompute continuation.

### Compatibility and limits

- Existing homogeneous defaults and shape keys remain unchanged. FIFO retains
  one owner, and recompute uses a separate homogeneous recovery batch.
- Chunking remains independent; an unchunked remaining prompt joins only if it
  fits the available mixed-batch Token budget.
- Mixed Profiles require exact observations; there is no interpolation,
  extrapolation or homogeneous timing fallback for a mixed shape.
- PP still drains in-flight waves before migration and DMA does not overlap
  stage execution. There is no live GPU kernel or measured mixed-cache overhead.

## 1.2.0 — 2026-11-04

Prefix reuse and incremental prompt execution extend online serving across
single groups and TP×PP layouts. These are CPU resource and timing models;
hardware-validation and license-selection status remain unchanged.

### Added

- Opt-in Prefix Cache using explicit prompt token IDs, namespace and complete
  prefix ancestry, with block sharing, reference counts and bounded LRU eviction.
- Shared-cache/private-block accounting inside the existing fixed GPU KV pool;
  idle cache blocks are reclaimable before live-request preemption.
- Incremental SHA256 ancestry keys avoid storing a complete prompt prefix at
  every block boundary; failed admission rolls back provisional cache references.
- Chunked Prefill and chunked recomputation with retained context, no early
  output and decode-priority continuous batching using homogeneous phase batches.
- Prefix/chunk request and run metrics, single-group and PP examples, and
  continuation through CPU swap or recomputation without repeated output.

### Changed

- Prefill/recompute Profiles accept nonzero prefix context. Every chunk still
  needs a supported query/context shape; zero-context records do not substitute
  for missing cached-prefix or chunk observations.
- Comparison contracts include the new policies and prompt identities.
- Warmup caches are discarded before measurement. Shape-only `predict` remains
  an uncached, unchunked analysis of the workload rows.

### Limits

- Batches remain homogeneous by phase; no mixed prefill/decode kernel is modeled.
- Stage timings conservatively charge their output/sampling boundary per chunk;
  only the final prompt chunk emits a Token.
- Host swap copies complete request KV, including shared blocks, and restores
  private GPU blocks; this is not a CPU Prefix Cache tier.
- No actual GPU execution, measured cache overhead or hardware accuracy claim.

## 1.1.0 — 2026-11-03

Online pipeline-parallel serving and CPU KV swap are modeled in virtual time.
License selection and hardware-validation status are unchanged; consult the
validation report and CI runs for checks actually executed.

### Added

- Independent PP stage and microbatch scheduling, stage-local TP groups,
  serialized boundary links, autoregressive feedback and cutoff progress.
- Conservative all-stage KV reservation and persistent per-rank activation
  buffer budgets, including replicated TP boundary payloads.
- A bounded CPU KV pool and serialized D2H/H2D transfers with safe source/
  destination ownership through completion, timeout and cutoff.
- Explicit `preemption="swap"` and opt-in recompute fallback, host-capacity
  failures, migration metrics and PP/swap examples.
- Pipeline pressure draining before eviction; no new GPU work starts during
  modeled DMA. These overlap restrictions are deliberate modeling assumptions.

### Changed

- `simulate` supports online PP; `predict` still performs shape-only analysis
  and leaves online metrics unavailable.
- Shared request state and metric reporting preserve single-group interfaces.
- Comparison contracts include the new strategy/memory defaults while retaining
  compatibility with supported prior default contracts; Profile JSON remains
  schema version 1.

### Limits

- No real GPU/CPU DMA, measured transfer validation or hardware accuracy claim.
- NVMe, multiple replicas, Prefix Cache, chunked prefill and quantized formats
  remain outside the supported scope.

## 1.0.0

V1 software implementation. The repository remains private while license
selection is pending; this entry is not a PyPI publication or hardware-accuracy
claim. Consult validation records for checks actually executed.

### Added

- Versioned whole-stage timing Profiles with strict model/device/software/layout
  identity, exact ragged lookup, complete-grid interpolation and explicit
  exploratory extrapolation.
- Dataset calibration with repeated-shape means, Workspace maxima, training
  hashes/membership and immutable version lineage.
- Grouped holdout splitting and leakage checks, coverage, per-group latency
  errors and explicit out-of-domain reporting.
- Saved online run comparisons and candidate rankings, including contract
  matching, missing metrics, request pairing, SLA agreement and ties.
- Independent warmup, empty-cache validation, half-open arrival admission and
  explicit drain controls.
- Profile-aware Workspace accounting and standalone communication alpha-beta
  calibration with explicit Infra patches.
- CLI calibration, comparison and validation workflows, fixture demonstrations,
  software specification and validation documentation.

### Preserved

- CPU deterministic events, FIFO/prefill-first batching, TP memory/communication,
  fixed KV pools, request preemption and prefix recomputation.
- Installable Python API/CLI and dependency-free core.

### Evidence and limits

- Shipped measurement-shaped records are test fixtures, not GPU measurements.
- No AMD measurement or hardware prediction accuracy has been established.
- Online PP, CPU swap, multiple replicas, Prefix Cache, chunked prefill and
  quantized storage formats remain outside V1.
