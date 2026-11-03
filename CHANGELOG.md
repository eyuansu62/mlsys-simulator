# Changelog

## 1.1.0

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
