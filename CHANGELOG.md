# Changelog

## 1.14.0 — 2026-11-05

### Added

- Optional source-bound public HSA signal backing capture, including actual
  pointer-information extents, CPU preferred-owner queries, hardware event IDs,
  ordinary-signal controls and stable generation/cursor evidence.
- Passive Linux x86-64 KFD event-page observation and independent replay of
  original event creation, GTT allocation, GPU/render-fd acquisition and mapping
  generations. Unknown requests, lost records and unsupported bindings remain
  partial; queried available memory is never substituted for Workspace.
- A standalone ordinary-signal namespace control and hash-bound cold-control
  artifacts shared as source evidence, independently of train/repeat acquisition.

### Evidence boundaries

- PP2 completed formal training and a frozen independent Runtime repeat:
  48 records per cohort, 16 stage-specific shapes, full record/rank/shape
  coverage, zero byte error and no underprediction. All attribution and
  fixed-budget gates passed independently before validation.
- A live logical signal is not relabeled as destroyed. CPU ABI regions and
  driver GTT event pages are reported separately; they do not establish a
  physical zero-VRAM result, complete private CPU-object size or host-memory peak.
- Every original GPU backing, pool, communication, PP transport and scratch
  obligation remains subject to independent attribution and budget checks.
  Memory-instrumented timings are diagnostic and do not fit a latency Profile.

## 1.13.0 — 2026-11-05

Workspace acquisition now observes each real Worker iteration and separates
rank-local scratch from retained native PP and communicator storage.

### Added

- Optional source-pinned single/TP2/PP2 Worker acquisition for prefill lengths 16/32 and
  the following three decode contexts, retaining warmup, per-rank storage
  generations, real PP handoff/retention and separate teardown evidence.
- Rank-placement ownership checks for Qwen3 TP shards, PP layer ranges and
  distinct PP endpoint vocabulary copies; missing ownership remains partial.
- `validate_workspace_repeats` and `validate --workspace-repeats` for independent
  acquisitions at retained exact shapes and TP ranks, with frozen training
  SHA/recomputed-fit checks, capture separation, coverage and byte underprediction.

### Changed

- Native PP sender reservations cover `max(inflight, 2)` full output pairs for
  the supported synchronous MRv1 path. The existing inflight component is
  preserved and the extra one-wave retention slot is reported separately.
  Receiving buffers, legacy transport and latency calculations are unchanged.
- Native observation uses explicit bounded event budgets and preserves API
  context, finalization and communication-causality evidence. Unknown backing,
  unfinished calls, missing lineage or lost events cannot become complete data.

### Evidence boundaries

- Scoped single-card and TP2 train/frozen-repeat pairs completed full attribution
  and public Workspace validation: eight exact shapes and 24 records per cohort
  (48 rank observations for TP2), full coverage, zero byte error and no
  underprediction. PP2 remains partial and has no admitted Workspace fit.
- The raw Worker collector is restricted to BF16, synchronous eager MRv1,
  seq1/512-token limits and 256MiB KV, without cache/chunk/mixed/swap. Its success
  does not imply that all native communication attribution gates passed.
- Memory instrumentation changes observed durations; those timings are not
  used to calibrate a new latency Profile. Complete Workspace datasets contain
  no duration and remain exact-only when consumed by existing serving policies.
- CPU combination regressions preserve TP/PP, cache, swap, mixed and replica
  software contracts; they do not extend a measured Profile to unmeasured domains.
  Independent repeats do not establish unseen-shape generalization or hardware PASS.

## 1.12.0 — 2026-11-05

Real Runtime verification now distinguishes exact-domain repeated stage
measurements, complete requested-storage attribution and physical KV-pool OOM.

### Added

- Optional source-pinned single-GPU Qwen3 physical KV capacity adapter with
  actual storage/configuration binding and exact native allocation-failure proof.
  The capacity validator retains `torch.empty` and accepts real `torch.zeros`.
- Explicit `validate --exact-repeats --training-dataset ... --profile ...` for
  independent acquisition sessions at retained exact shapes, with frozen training
  SHA/recomputed-fit checks and strict session/capture/record/group separation.
  Unseen-shape holdout and hardware-acceptance leakage rules are unchanged.
- vLLM acquisition provenance in normalized datasets and fitted Profile notes.
- Optional ROCm Workspace audit sources, native Torch allocator/thread tracking,
  nine-domain HIP/HSA/RCCL/private-scratch evidence, explicit fixed-owner budgets
  and fail-closed per-generation replay. No mandatory simulator dependency or
  precompiled GPU binary is added.
- A narrowly scoped live Workspace Profile with a frozen independent repeat:
  Qwen3-1.7B BF16 RX9070XT TP1/PP1 single Prefill16/32, requested-storage capacity.
- Public Workspace acquisition/assembly gates bind observed software, device and
  Runner settings; missing legacy observations cannot produce complete output.
- Scoped live reports for confirmed physical KV/OOM endpoints, TP2 unseen-shape
  stage validation, PP2 independent exact repeats, conditional Prefix/Chunk
  online repeats, actual mixed execution and two independent physical replicas.

### Evidence boundaries

- Exact repeats evaluate retained shapes; they do not assert unseen-shape
  generalization, fresh Runtime isolation by themselves or hardware PASS.
- Individual allocation probes remain partial/candidates. Only the complete
  source/owner/native/lifetime/budget/return/process merger can emit a complete
  Workspace dataset. Allocator rounding/cache/driver headroom remains separate.
- Capacity adapters run fresh actual Runtimes and retain typed allocator evidence;
  startup, admission, warmup, lock and unknown failures are not fabricated KV OOM.

## 1.11.0 — 2026-11-04

Native PP collection can explicitly disable the P2P LL/LL128 threshold path,
which is independent of the collective `NCCL_PROTO` setting in the tested RCCL
build. This makes the mitigation and the resulting Profile domain auditable.

### Added

- Opt-in `native_pp_primitives_v3` with lazy initialization and positive-payload
  P2P Simple-only threshold requests. Only child workers receive both zero
  thresholds; conflicting caller settings fail before launch. The host timer,
  real device completion, warmup and untrimmed observations are unchanged.
- Strict per-rank threshold evidence and benchmark policy binding. Raw capture,
  parent calibration and holdout cannot silently cross protocol/policy domains.
- Per-size held-out distribution statistics retaining outliers and tails, in
  addition to existing individual and observed-mean errors and coverage.
- Explicit P2P threshold and SHM copy-policy settings in pinned vLLM Runtime
  fingerprints, preserving the legacy fingerprint when these settings are unset.

### Compatibility

- v1/v2, affine/mean lookup, TP/PP transport accounting and Profile consumer
  interfaces remain compatible. v3 is an explicit operational policy, not a
  claim that the old default LL domain has become stable.
- No event-time substitution, synchronization-time subtraction, trimming or
  stochastic prediction is used to reduce reported point errors.

## 1.10.0 — 2026-11-04

Small-message native PP calibration can retain nonmonotonic size regimes instead
of forcing one global affine curve across a measured latency jump.

### Added

- Opt-in `--native-pp-model size-lookup` with per-placement arithmetic-mean
  knots, all retained observations, recomputed dispersion/provenance and fixed
  training-only guards for sparse coverage and latency jumps. Exact knots remain
  usable; unsupported intervals and all outer extrapolation fail explicitly.
- Independent holdout reports separate per-sample error from per-size observed
  mean error, with full coverage/rejection counts, small-message results and
  exact-versus-unseen size summaries.
- Explicit `native_pp_primitives_v2` collection matching lazy world communicator
  initialization. Protocol and selected transport/wait flags remain auditable;
  parent calibration and holdout reject mixed measurement semantics.
- TP/PP two-tensor charging, pre-admission failure and per-replica stage /
  communication / Workspace Profile integration regressions with cache,
  chunking, swap and mixed batches.

### Compatibility

- Affine fitting and benchmark v1 remain the defaults. Existing affine Profile
  files and unconfigured analytical communication behavior retain their contract.
- Sampling keeps real completion synchronization and correctness checks. It does
  not subtract the small-message slow path or discard outliers.

## 1.9.1 — 2026-11-04

### Fixed

- Delay checkpoint-dependent embedding tying validation until the loaded local
  weights are observable. The pinned Runtime may temporarily untie redundant
  checkpoint heads and later re-share their storage without restoring the HF
  flag. Canonical capture now requires the prepared, effective storage proof;
  missing or contradictory proof remains ineligible.
- Qualify internal capture record IDs by session so independent runs with the
  same local iteration counter can undergo holdout validation. Reused records,
  groups and shapes retain the existing leakage checks.
- Bind explicitly configured P2P, shared-memory, IB/network and blocking-wait
  transport policies to the Runtime fingerprint without changing the unset
  legacy fingerprint.
- Recognize the strict unscaled default RoPE alias exposed by Transformers 5
  without accepting genuine scaling, conflicting theta values or extra fields.

## 1.9.0 — 2026-11-04

Dynamic Workspace can use a separately attributed per-rank Profile, while
capacity evidence distinguishes allocator OOM from admission and Runtime failure.

### Added

- Complete Runtime-owned scratch-ledger attribution with underlying-storage
  alias deduplication, pending-release lifetimes and explicit per-rank
  completeness; partial and saturated captures cannot form Workspace data.
- Independent exact-only Workspace datasets/Profiles, repeated-shape per-rank
  maximum fitting, observed-zero support and explicit local/per-replica
  selection with immutable CLI snapshots.
- Per-rank Workspace budget selection alongside stage and native communication
  Profiles, preserving static reserves and existing cache/swap/TP/PP lifetimes.
- Optional pinned vLLM WorkspaceManager observation, labeled partial managed
  scratch capacity rather than complete temporary Workspace.
- Fresh-Runtime/process capacity collection across input-token, request-cohort
  or KV-pool axes, typed OOM phase/allocator-counter proof, scoped KV allocation
  evidence and full-configuration binding.
- Capacity validation reporting repeat instability, nonmonotonic evidence,
  censored endpoints and adjacent observed success-to-OOM brackets.

### Evidence limits

- Full scratch instrumentation is a user Runtime adapter contract. Allocator
  peak minus baseline and managed scratch alone are not complete attribution.
- Capacity collection preserves the scheduler; request count is a cohort size,
  not a claim about actual GPU batch size. Logical KV rejection, timeout,
  lock AssertionError and peer failure are not silently relabeled device OOM.
- CPU fixtures test contracts and orchestration; no live AMD accuracy or
  hardware-validation verdict is established by their completion.

## 1.8.0 — 2026-11-04

Native PP boundary communication can use independently collected, explicitly
selected primitive Profiles alongside isolated stage Profiles.

### Added

- Standalone single-node HIP distributed collection for directed P2P pairs and
  receiver TP AllGather groups, using preallocated out-of-place buffers,
  synchronized local timing and correctness evidence.
- Separate native communication dataset/Profile contracts with physical
  placement, dtype, actual stack/RCCL binding, direct nonnegative affine fits,
  immutable provenance and explicit coverage/extrapolation policy.
- Independent held-out communication validation with source/identity and
  training-leakage checks.
- Explicit per-local-replica communication Profile selection, CLI snapshots
  and native transfer integration: maximum paired-rank P2P plus one receiver
  AllGather per sliced tensor, charged separately from stage service.
- Collection/calibration/validation commands, fixture examples and protocol
  documentation without a new required dependency.

### Compatibility and evidence

- Unselected native communication Profiles leave the existing analytical path
  unchanged. Stage Profile identities and local model-collective timing remain
  unchanged; selected communication Profiles do not add either cost again.
- Directed P2P is not a ping-pong round trip or RTT divided by two. AllGather
  message bytes denote full output bytes, with one input shard per TP rank.
- Measured primitive service excludes allocation, startup, warmup, CPU control
  barriers and correctness work. It does not measure vLLM metadata ingress,
  upstream wait, Runtime output allocation, contention or the complete native
  boundary. CPU fixtures remain fixtures and do not establish AMD accuracy.

## 1.7.0 — 2026-11-04

Pinned native PP capture separates dependency waiting from reusable stage
service and preserves the physical transfer geometry through calibration.

### Added

- Explicit native Qwen3 PP transport contract with hidden/residual tensor
  descriptors, actual padding, conditional per-tensor TP slicing and receiver
  AllGather; legacy full-hidden transport remains the default.
- Reversible observation of the Runtime's natural lazy receive, without early
  tensor access or a pre-receive device-wide synchronization. A natural-wait
  compute-stream boundary retains preparation work in stage service; raw wait
  evidence is retained while service excludes receive/postprocess AllGather.
- Exact native stage Profile lookup, strict repeated-shape payload agreement,
  complete TP and adjacent-stage descriptor checks, and quarantine for missing
  or unsupported evidence.
- Shared online/offline boundary accounting and full resident activation
  buffer budgets, including retained sliced-receive backing, persistent input
  copies and a declared physical-token bound.
- A native R2×TP2×PP2/cache/chunk/swap/mixed CPU example and sampling/calibration
  documentation.

### Evidence limits

- Native receive wait includes upstream dependency delay. It is not fitted as
  independent network latency; P2P and receiver AllGather remain explicitly
  analytical communication costs, separate from calibrated stage service.
- Source-contract/CPU verification does not establish live AMD/vLLM accuracy.
  Async output, SP, DBO, native DP/EP, broadcast PP and unsupported ABIs remain
  outside the canonical sampling domain.

## 1.6.0 — 2026-11-04

Qwen3 geometry keeps residual and query projection widths distinct while
preserving the existing serving and evidence contracts.

### Added

- Optional tied vocabulary storage, per-layer Q/K Norm and final RMSNorm,
  including physical TP/PP ownership and independent query-width costs.
- Strict dependency-free dense Qwen3 HF config import with model revision,
  explicit dtype and canonical architecture fingerprints.
- Qwen3 model/config examples, isolated shape prediction and a synthetic
  R2×TP2×PP2/cache/swap/mixed serving demonstration.
- Version-pinned MRv1 runner and Scheduler capture with reversible instance hooks, explicit
  worker RPC integration and a dormant, opt-in general-plugin entry point.
- Per-rank capture validation and complete synchronous TP-group aggregation,
  retaining quarantine reasons, Runtime execution fingerprints and raw evidence.
- Published Scheduler lifecycle normalization with explicit conditional workload
  binding; runner sampling candidates remain separate from served output.

### Compatibility and evidence

- Default model flags and empty fingerprints preserve legacy Profile identity;
  nondefault architecture choices cannot silently share a different Profile.
- Tied storage reduces one-rank allocation; separate PP endpoint copies and
  LM-head execution remain accounted for.
- Imported model geometry does not turn client SSE chunks, C1 request latency
  or allocator peaks into missing canonical hardware measurements.
- CPU fixture verification does not establish live vLLM or GPU accuracy.
- Async result consumption remains observable raw diagnostics; it does not
  establish an exclusive whole-stage service sample or a Scheduler lifecycle.


## 1.5.0 — 2026-11-04

Optional hardware adapters connect collected Runtime evidence to calibration
and explicit, scoped acceptance. The serving simulator remains a CPU model.

### Added

- AMD/ROCm availability and version probing, synchronized stage-call collection
  and raw allocated/reserved allocator evidence with explicit Workspace attribution.
- RCCL AllReduce collection/import with preserved units, rank/mode identity,
  correctness checks and raw provenance.
- Semantic Runtime trace and token-ID-aware vLLM client collection paths that
  retain their different server/client timing boundaries.
- Read-only reference-package auditing for the supplied NVIDIA legacy and AMD
  9070 XT release schemas, including integrity and model/timing compatibility.
- A hardware-evidence workflow with per-replica fitting, disjoint holdout,
  online comparison, sample/repeat/error criteria and explicit verdicts.
- Hand-calculated acceptance fixtures and collection/evidence documentation.

### Evidence limits

- A CPU fixture remains a fixture, including when its numerical criteria pass.
  A completed command is not automatically a hardware-validation verdict.
- Client content-chunk times and profiling attribution do not become complete
  stage times or semantic per-token events; raw allocator peaks do not imply Workspace.
- AllReduce support does not establish unidirectional P2P, TP/PP or swap accuracy.
  Supplied Qwen3 model gaps are reported rather than silently changing real shapes.
- Raw archives and prompt/output text are not distributed with the library;
  license selection and private-repository status remain unchanged.

## 1.4.0 — 2026-11-04

Arriving requests can share a global serving timeline across independent model
replicas, with current-state load balancing and local cache affinity.

### Added

- Round-robin, least-loaded and minimum-load prefix-aware routing, with sticky
  request ownership and no future-output information in routing decisions.
- Replica-major TP/PP placement, independent KV/cache/host/DMA budgets and
  per-replica mixed-batching/recovery execution under global time/event limits.
- Routing assignments, per-replica diagnostics, common-window global metrics
  and explicit isolation of local resource/no-progress stops.
- Per-replica Profile paths, strict local identity checks, immutable CLI
  snapshots and full-cluster online comparison identities.
- Multi-replica shape prediction and synthetic routing/cache/PP/swap examples.

### Limits

- Requests and cache entries do not migrate or share storage across replicas.
- Dispatcher latency is zero-cost and shared fabric/host bandwidth contention
  is not modeled; GPU/DMA assumptions remain conservative within each replica.
- Step Profiles remain single-replica observations. Software support does not
  establish hardware accuracy or change pending license/private-repository status.

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
