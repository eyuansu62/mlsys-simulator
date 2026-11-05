# Version 1.14 capability matrix

This table describes the version 1.14 software scope. Available fixture demonstrations
and unit tests do not establish AMD hardware accuracy. User-supplied reference
packages are external evidence with their own timing and model-coverage limits.

| Capability | Software status | Supported scope and evidence boundary |
| --- | --- | --- |
| Six inputs and strict configuration | Implemented | JSON/TOML, JSONL workload, relative references; optional installed PyYAML |
| Qwen3 geometry and HF import | Implemented | Independent residual/query widths, tied vocabulary storage, Q/K and final Norm, strict unquantized full-attention config and architecture fingerprints |
| C01 macro performance model | Implemented | Shape-aware Dense MHA/GQA costs; analytical parameters are uncalibrated |
| C02 memory planning | Implemented | Per-rank physical budget, fixed KV pool, logical blocks and labeled Workspace source |
| C03 serving DES | Implemented | FIFO, default prefill-first batching or opt-in decode-first chunked/mixed scheduling, Token events, timeout and bounded execution |
| KV preemption/recovery | Implemented | Safe request boundary, recompute or explicit CPU swap, output history preserved |
| Warmup and initial state | Implemented | Isolated successful warmup; measurement starts with empty KV only |
| Arrival window and drain | Implemented | Half-open admission, NOT_ADMITTED outcomes, optional bounded drain |
| TP | Implemented | TP 1/2/4 in each stage, GQA replication and atomic all-stage/rank reservation |
| C04 communication analysis | Implemented | Ring AllReduce, logits AllGather, legacy PP P2P or explicit native two-tensor/sliced P2P plus receiver AllGather; no shared-link contention/overlap |
| C04 communication calibration | Implemented | Per-primitive/rank alpha-beta fit and explicit Infra patch; no automatic double addition to stage profiles |
| Native PP primitive collection/calibration | Implemented adapter/model | Independent directed P2P and receiver AllGather; preallocated service, actual stack/placement identity, default affine or guarded size-lookup Profiles and explicit local-replica selection; opt-in v3 P2P Simple-only threshold policy, unchanged host completion timing |
| Native PP communication holdout | Implemented | Existing Profile versus independent dataset; leakage/identity/protocol/placement/domain rejection, per-sample and per-size mean errors, untrimmed observed distributions, full rejected coverage, without an automatic split |
| PP analysis | Implemented | Shape-only capacity/forward/P2P; online metrics remain null because analysis has no queue |
| Online PP | Implemented | Stage/microbatch queues, boundary P2P, one outstanding iteration/request and sampled-Token feedback |
| PP activation budgets | Implemented | Conservative persistent per-rank buffers and all-stage KV reservation; native transport budgets output pairs, receive/backing copies and persistent input targets to a physical-token bound |
| CPU KV swap | Implemented | Bounded host tier and serial D2H/H2D per replica, safe source/destination ownership, explicit fallback |
| PP pressure and DMA | Implemented | Drain in-flight waves before reclaim; no new GPU work during DMA, conservative no-overlap policy |
| Versioned step Profiles | Implemented | Strict identity/source; native PP preserves physical payloads and is exact-only; legacy exact ragged/mixed lookup and complete homogeneous interpolation cells |
| Calibration and lineage | Implemented | Repeated-shape mean time/max Workspace, new versions, input and training provenance |
| Grouped holdout evaluation | Implemented | Shape/group connected components, leakage rejection, coverage and latency errors |
| Independent exact-stage repeats | Implemented | Frozen training SHA/fit/provenance, distinct acquisition session/capture/record/group evidence and retained exact shapes only; separate from unseen-shape holdout and hardware acceptance |
| Online run/candidate comparison | Implemented | Matching contracts, per-metric/request errors, SLA agreement, ties and candidate ordering |
| Independent Workspace Profiles | Implemented | Complete Runtime ownership, exact per-rank repeated maxima including observed zero; selected missing shapes fail without fallback |
| Independent Workspace repeats | Implemented | Frozen training SHA and exact-rank-max replay, separate acquisition identities, record/rank/retained-shape coverage and byte underprediction; no automatic fitting on repeat data |
| Worker Workspace acquisition | Implemented optional adapter | Source-pinned single/TP2/PP2 Prefill/Decode capture, actual prepared shapes, Torch generations, native communication ancestry and PP retention; full attribution requires every component gate |
| HSA/KFD synchronization ownership | Implemented optional adapter | Actual public CPU signal-region queries, cold event-ID namespace control, passive GTT/event/render-mapping generation replay and natural-shutdown prefixes; live handles are not relabeled as destroyed, and unknown backing remains partial |
| Partial managed-scratch observation | Implemented observer | Existing pinned vLLM WorkspaceManager capacity stays raw/partial; it does not prove complete dynamic Workspace |
| Legacy stage Workspace consumption | Implemented | Without an explicit Workspace Profile, supported stage observations or labeled analytical assumptions |
| Capacity/OOM collection and validation | Implemented adapter/workflow | Fresh Runtime/process per point/repeat, typed OOM and applied-axis/configuration evidence; pinned native vLLM single-GPU KV factory; unstable/nonmonotonic/censored outcomes and observed brackets |
| ROCm inventory and stage collection | Implemented adapter | Optional live HIP Runtime timing with explicit device/software identity; CPU fixture injection is not a GPU measurement |
| Pinned vLLM internal runner hooks | Implemented adapter | Explicit MRv1 ABI/version, member-aware shapes and sampling boundaries; native lazy receive/receiver AG excluded from stage service; async, SP and unproven intervals remain raw |
| Pinned vLLM Scheduler lifecycle | Implemented adapter | Actual enqueue, published output and terminal events; conditional replay is explicit, with prompt identity required for Prefix Cache |
| Internal capture aggregation | Implemented | Complete TP groups use maximum local service; native physical descriptors also match adjacent stages; strict model/Runtime identity, per-replica datasets and quarantine |
| RCCL AllReduce collection/import | Implemented adapter | Explicit benchmark units/ranks/mode, maximum-rank timing and correctness checks; sendrecv is not accepted as unidirectional PP P2P |
| Runtime trace and vLLM client collection | Implemented adapters | Semantic server Token events and token-ID client streaming retain different boundaries; SSE chunk counts do not become token counts |
| Reference-package audit | Implemented | Source hashes, structure, timing/shape coverage and compatibility diagnostics; archived programs and instructions are not executed |
| Hardware-evidence acceptance | Implemented gate | Explicit training/holdout and online criteria, leakage rejection, paired requests, independent repeats and scoped verdicts; fixtures keep hardware status false |
| AMD prediction accuracy | Not established generally | Supplied 9070 XT client traces do not provide compatible whole-stage/Workspace or multi-GPU measurements; acceptance is limited to evidenced identities and domains |
| Prefix Cache | Implemented | Explicit token ancestry/namespace, full shared KV blocks in the fixed pool, reference pinning and bounded idle-block LRU |
| Chunked Prefill/recompute | Implemented | Retained contexts, final-chunk-only output, decode priority; single-group and TP×PP recovery |
| Mixed Prefill/Decode batches | Implemented | Opt-in shared stage execution, decode-first packing, ordered member roles, joint Workspace and exact mixed Profiles; PP/cache/swap lifetimes preserved |
| Multiple replicas and load balancing | Implemented | Global causal DES; round-robin, least-loaded and load-bounded prefix affinity; independent TP/PP/cache/swap/mixed execution |
| NVMe swap and quantized formats | Unsupported, rejected | CPU swap only; 2/4/8-byte float storage assumptions only |

The shipped calibration data uses `source.kind="test_fixture"`. Fixture opt-in is explicit
and does not grant measured status. Profiles backed by measured records are
calibrated inputs, not an automatic hardware-validation verdict. Actual test and
CI results belong in the validation report and workflow run records.


TP2 and PP2 have limited independent real stage observations, and native P2P
has its own independent primitive validation. CPU KV swap has no end-to-end
hardware validation. Virtual links/DMA use declared rates and explicit
conservative policies; successful synthetic examples do not establish actual
multi-GPU throughput or PCIe copy accuracy.
See [pipeline and swap](pipeline-and-swap.md) for configuration and lifetimes, and
[native PP capture](native-pp-capture.md) for source-pinned sampling. Natural
receive waiting is dependency evidence, not a calibrated independent network
latency. Native boundary communication uses an explicitly selected independent
communication Profile or the labeled analytical default; see
[native PP communication calibration](native-pp-communication.md).

Live collectors require the relevant tools already installed on an AMD host;
the core simulator keeps no required third-party dependency. Probe failure does
not trigger CPU timing. Raw allocated/reserved memory is not automatically
Workspace. Complete scratch attribution is an explicit user Runtime integration
contract; passive manager capture does not establish it. A PyTorch collective
build version is not a verified RCCL runtime
version. The user reference packages do not establish TP/PP, preemption, swap or
RCCL accuracy. See [ROCm collection](rocm-collection.md) for boundaries and the
explicit acceptance plan.

Prefix/chunk behavior is a declared simulator policy, not a reproduction of a
particular Runtime implementation. Cached-prefix and chunked Profile queries
need matching context-domain evidence. See [Prefix Cache and chunking](prefix-cache-and-chunking.md).

Mixed execution uses an uncalibrated joint macro model, declared synthetic
service or exact mixed-stage records. It does not execute a GPU kernel or infer
mixed latency from homogeneous measurements. See [mixed batching](mixed-batching.md).

Multi-replica routing keeps request ownership and resources local while sharing
clock, execution budgets and evaluation. Each replica needs its own strict
Profile identity; online Compare retains the cluster identity. Shared-fabric
contention and cross-replica migration are outside the model. See
[multi-replica routing](multi-replica.md).

Workspace and capacity fixtures establish software contracts only. Capacity
request-count sweeps preserve the scheduler and do not imply actual GPU batch
size. Logical KV refusal, timeout, peer death and lock errors are distinct from
typed allocator OOM. Observed brackets retain resolution and evidence gaps; see
[Workspace and capacity validation](workspace-and-capacity.md).

## Limited live evidence added in 1.12

These claims are separate, scoped results; none grants general hardware
acceptance or lets one Profile silently cover another Runtime/layout domain.

| Evidence | Observed scope | Remaining boundary |
| --- | --- | --- |
| Complete Workspace | RX9070XT, Qwen3-1.7B BF16, TP1×PP1 eager single prefill16/32; independent three repeats per shape, 100% exact coverage and no underprediction | Owned requested-storage capacity only; no decode, other shapes, multi-rank or concurrent policies |
| Physical KV pool endpoints | Each device has independently repeated successful and typed native allocation-OOM endpoints | Finite resolution, `exact_threshold=null`; retained RX7900XTX interior worker failures are censored; default analytical budgets falsely reject some successes |
| TP2 stage latency | 24 unseen-shape observations, 100% coverage, MAPE2.9520% | Short shapes, one active sequence, pinned eager Runtime; no online SLA |
| PP2 stage latency | 72 new exact-shape repeat observations, 100% coverage, MAPE6.1565% | Retained exact domain; no unseen-shape generalization or four-GPU TP2×PP2 |
| Prefix/chunk/mixed behavior | Actual prefix reuse, partial prompt chunks and simultaneous prefill/decode observed with complete outputs | Correctness probes are distinct from the conditional latency results below |
| Prefix Cache / chunking conditional latency | Frozen policy Profiles; three fresh Runtime repeats each, 45 exact-covered stage rows, 9/9 paired requests; all TTFT/TPOT/E2E p50 relative errors≤15% | Short fixed cohorts with actual conditional arrivals, including cache seeds; no open-loop throughput, tails, SLA or ranking claim |
| Mixed-policy accuracy | Real initial `[8,8]` batch differs from the simulator's immediate solo `[8]` under actual arrivals | Exact Profile lacks solo `[8]`; retained out-of-domain coverage gap, not an accuracy pass |
| Two physical replicas | Two independent TP1×PP1 instances; fixed round-robin0/1/0/1, four completed requests, native request-ID ownership and separate public normalization checked | Routing/output smoke only; no least-loaded/prefix-aware comparison, routing-cost, contention, performance or SLA claim |

See the [Workspace](workspace-live-validation.md),
[capacity](capacity-live-validation.md) and
[multi-GPU stage](multigpu-stage-validation.md) and
[policy/replica](online-policy-live-validation.md) reports. The tested vLLM revision
has no native CPU KV-swap path for an end-to-end eviction/restore comparison.
Only two physical GPUs are available; full TP2×PP2 and its receiving AllGather
path require a different hardware cohort.

## Scoped live Workspace evidence added in 1.13

Single-card Qwen3-1.7B BF16 eager MRv1 completed training and a frozen independent
Runtime repeat: 24 formal observations per cohort, eight exact shapes including
six Decode contexts, full record/rank/shape coverage, zero byte error and no
underprediction. TP2 independently completed the same train/frozen-repeat
workflow: 24 records and 48 rank observations per cohort, eight exact shapes,
full record/rank/shape coverage, zero byte error and no underprediction.
Earlier PP2 acquisitions retained partial per-rank component evidence and
could not produce a canonical Workspace fit. The PP process-exit
diagnostic preserved actual unsuccessful SDK stop codes and unresolved formal
signal generations.

A separate PP2 stress observation justified the synchronous sender budget's
two-output-pair minimum. It is not a Workspace training record. Cache, swap and
mixed execution remain software composition checks outside this measured domain.
See [Worker Workspace validation](workspace-stage-validation.md) for exact scope,
frozen Profile lineage and promotion gates. No result grants general hardware
acceptance.

Version 1.14's source-bound HSA/KFD path subsequently completed PP2 attribution,
calibration and a frozen independent Runtime repeat. Each cohort contains 48
records/rank observations and 16 stage-specific exact shapes; record/rank/shape
coverage is 1, byte errors and underpredictions are zero, and frozen Profile
bytes are unchanged. Earlier diagnostic data remain partial and unmodified.
This requested-storage result does not establish physical zero-VRAM overhead,
unseen-shape behavior, general capacity or unmeasured policy accuracy.
