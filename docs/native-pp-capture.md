# Native vLLM PP capture and stage Profiles

Version 1.7 supports the native lazy-receive PP path of the pinned vLLM MRv1
build `0.29.1.dev0+g98dff2a81.d20261002`. It records actual physical tensor
geometry and isolates stage service before calibration. A received dependency
wait is retained as raw evidence; it is not treated as independent link service.

The core remains dependency-free. Live capture needs the supported vLLM,
ROCm-enabled PyTorch and real devices already installed. CPU fixtures and the
example below do not establish live vLLM/AMD correctness or prediction accuracy.

## Configure the transport contract

The original `strategy.pipeline_transport="full_hidden"` default preserves the
single full-hidden-tensor-per-rank model. Select the native contract explicitly:

```json
{
  "strategy": {
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 2,
    "pipeline_transport": "vllm_native",
    "pipeline_transport_max_tokens": 8,
    "max_batch_tokens": 5
  }
}
```

This fragment extends a complete bound experiment. Native transport requires
PP > 1 and an explicit `model.activation_dtype` whose byte width agrees with
`activation_bytes`. A nonzero `pipeline_transport_max_tokens` must cover
`max_batch_tokens`; zero resolves to `max_batch_tokens`. Set the bound high
enough for actual Runtime padding. A captured physical payload above the bound
is not silently truncated or accepted. The bound is part of native Profile
identity and the persistent activation-memory budget.

For a live receiving stage, this bound must also cover the actual loaded
Runner's persistent input capacity (`max_num_tokens`), even when the captured
batch is small. Capture checks the full stored `hidden_states`/`residual`
buffers, dtype and capacity and retains `native_pp_persistent_binding`.
A small batch does not prove that a much larger persistent pool fits the bound.

The complete CPU example enables native communication in the existing
R2×TP2×PP2 Prefix Cache, Chunked Prefill, CPU swap and mixed-batch scenario:

```bash
mlsys-sim simulate --experiment examples/v1_7/qwen3-native-r2-tp2-pp2.json \
  --out runs/qwen3-native-combined
```

It uses synthetic stage times and declared device/link rates. It does not load
weights, perform device transfers or collect a measured Profile. Independent
simulator replicas remain distinct from unsupported native vLLM DP.

## Preserve the real PP payload

Dense Qwen3 sends `hidden_states` and `residual`. Each tensor descriptor retains
its actual full `[padded_tokens, hidden_size]` shape, dtype, element size, element
count and full bytes. `padded_tokens` is observed from the actual tensor, not
inferred from the logical query sum. The descriptor also retains:

- whether this tensor uses the pinned Runtime's TP slice;
- wire bytes per rank and whether receiver AllGather is required;
- full resident bytes per rank for one reconstructed hidden/residual pair.
  This descriptor is not the total allocator peak or the complete buffer budget.

With SP disabled and the pinned Worker's default per-key policy, a tensor is
sliced when TP > 1 and its element count is divisible by TP. The wire payload
then has `full_bytes / TP`; otherwise each paired rank sends the full tensor.
The receiver reconstructs a sliced tensor using TP AllGather. Both tensors are
checked separately. Unknown per-key overrides or unsupported transport behavior
cannot be inferred from TP size alone.

The native descriptor contract is `vllm_native_pp_v1`. Import checks its integer
arithmetic, dtype, padding bound and slice decision. Every TP rank must agree
on the physical descriptor, and each outgoing payload must match the adjacent
stage's incoming payload for that same iteration and rank. Missing edge/rank
proof, inconsistent geometry or an unsupported boundary cannot produce a
canonical native stage sample.

## Observe the natural receive boundary

The pinned Worker posts `irecv_tensor_dict` before calling Runner execution.
Its `AsyncIntermediateTensors` waits only when the Runtime naturally accesses
the tensors. A receive contains work-handle waits and postprocess callbacks,
including reconstruction AllGather. The hook observes that natural
`wait_for_comm()` invocation on the original object:

1. It does not read `.tensors`, call wait early or replace communication handles.
2. A receiving stage with outstanding lazy work does not issue a device-wide
   synchronization at Runner entry.
3. When the Runtime naturally invokes the receive wait, only the current
   compute stream is synchronized first. This completes this iteration's
   preparation GPU work inside stage service before the excluded interval
   starts. `pre_wait_compute_stream_synchronized=true` records that proof.
4. The original wait and postprocess then run. A device synchronization after
   their return establishes completion of the receive/AllGather interval.
5. Its positive, contained interval is excluded from the whole Runner wall
   span. Preparation work, persistent-buffer input copy, model execution and
   final-stage sampling stay inside stage service.

`pp_receive_timing` retains the natural interval, excluded nanoseconds and full
wall duration. The reusable `duration_ms` is the wall span less this excluded
interval. Import validates the arithmetic and requires positive remaining
service. Already materialized/external-completed inputs need an explicit
before-Runner completion scope; they do not fabricate a natural wait event.
For those external-completed inputs, an explicit readiness synchronization
precedes a reset stage start, recorded as `external_receive_synchronized=true`.
This keeps an outstanding external GPU tail out of the stage sample. Original
return values, exceptions and lazy wait idempotence remain the Runtime's
behavior. Closing capture restores observed instance methods.

The compute-stream boundary relies on the supported standard NCCL/RCCL
ProcessGroup path: posting receive work orders the communication stream after
existing compute-stream work; the original work wait establishes the reverse
consumer dependency. It does not assert safety for arbitrary custom transport
streams or early consumer waits outside the checked Runtime contract.

The stage Profile still includes local model collectives and output/sampling.
It excludes incoming PP receive/postprocess AllGather. The receiver's raw wait
may include upstream production or scheduling delay, so it is not a reusable
P2P latency sample. Metadata ingress and previous-send waits outside Runner
also do not become an invented stage or network cost.

## Import and calibrate

Use the existing opt-in plugin or worker RPC path described in
[Qwen3 and internal hooks](qwen3-and-vllm-hooks.md). The live bound experiment
must select `pipeline_transport="vllm_native"` and describe actual device,
model, software, padding bound, scheduler/cache and TP/PP identities. Select
MRv1 with `VLLM_USE_V2_MODEL_RUNNER=0` and synchronous scheduling. A synthetic
example's device defaults cannot stand in for inventory from a live host.

Collect complete per-stage/per-TP-rank exports in one finite diagnostic cohort.
An import plan lists the actual files, with paths relative to the plan:

```json
{
  "schema_version": 1,
  "captures": [
    "captures/stage0-tp0.json",
    "captures/stage0-tp1.json",
    "captures/stage1-tp0.json",
    "captures/stage1-tp1.json"
  ]
}
```

These paths are a live-host template, not files distributed with the library.
Then import and fit the per-replica dataset:

```bash
mlsys-sim collect --kind vllm-hook --experiment bound-native-experiment.json \
  --plan native-capture.plan.json --out runs/native-capture
mlsys-sim calibrate --dataset runs/native-capture/datasets/replica-0.json \
  --profile-version native-stage-v1 --out runs/native-fit
```

Test captures require `--allow-test-fixture` in both commands and keep fixture
provenance. Measured records require the normal strict hardware/software
identity. With multiple independent replicas, fit each dataset separately and
provide the matching `replica_profile_paths`. To use a single-replica result,
set `calibration.backend="profile"`, `profile_path` and `profile_version` in
the effective experiment exported by collection. Use new output directories.

Native dataset rows and Profile samples carry `pp_transport`: an incoming
descriptor for receiving stages and an outgoing descriptor for nonfinal stages.
Calibration preserves these descriptors. Repeated observations of one logical
shape must have identical physical transport geometry; different padding or
paths do not get averaged together. Native Profiles use exact shape lookup,
including ordered mixed roles. They do not interpolate or extrapolate a
physical payload for an unobserved shape. Missing shapes fail through the
existing calibration-domain error path.

Stage calibration/holdout evaluates isolated stage duration. It does not turn
raw dependency waits into network validation. Complete compatible runner
records can establish stage readiness; Scheduler arrival/published-token/
terminal evidence is still required for canonical online comparison and
Prefix Cache replay. Prompt-ID, namespace and explicit workload-binding rules
remain unchanged.

## Charge transport once and reserve full buffers

Both shape prediction and online PP simulation use the same boundary rule.
For every native tensor, the declared analytical model charges one P2P message
using the actual per-rank wire bytes, then receiver AllGather when required.
A ring AllGather uses full logical tensor bytes and TP rank count. Two tensors
therefore incur two message costs; slicing does not remove receiver work.
The observed Profile descriptor supplies padding for calibrated stage shapes.
Analytical/synthetic shapes without observed geometry use their declared
logical token count and label that payload source explicitly.

The boundary is charged separately from stage Profile service, exactly once.
Without an explicitly selected communication Profile its source remains
`analytical_uncalibrated`; no α/β is fitted from the lazy wait.
Since 1.8, independent communication microbench calibration provides separately
selected per-placement P2P/receiver-AllGather Profiles. This leaves the lazy
wait diagnostic unchanged: it is still not a network-service sample. See
[native PP communication calibration](native-pp-communication.md). `pipeline_p2p=false` retains its explicit ideal zero-cost mode.
There is no claim of shared-fabric contention or asynchronous communication/
compute-overlap prediction.

Resident activation memory is conservative and independent of wire slicing.
Let `H = effective_pipeline_transport_max_tokens × hidden_size × activation_bytes`
be one full hidden-sized tensor and `I = pipeline_max_inflight_microbatches`.
The budget distinguishes each direction rather than multiplying one pair by
the number of adjacent boundaries:

| Stage-side allocation | Per-rank reservation |
| --- | --- |
| Outgoing boundary | `max(I, 2) × 2H` for full hidden/residual outputs |
| Incoming boundary | `I × (2H + B)` for reconstructed results plus retained receive backing |
| Receiving-stage persistent input-copy target | One additional `2H`, independent of `I` |

`B` is the total full backing of the tensors that use slicing. The pinned
receiver allocates full backing before selecting a wire slice; its postprocess
closure retains that backing while AllGather creates reconstructed full tensors.
For both sliced tensors `B=2H`; with no slicing `B=0`. Thus a receiving TP1 stage
reserves `I × 2H + 2H`, while a receiving stage with both tensors sliced reserves
`I × 4H + 2H`. An interior stage additionally reserves its outgoing boundary.
The first stage has no incoming/persistent-input reservation, and the final
stage has no outgoing reservation.

The outgoing floor of two pairs accounts for the pinned synchronous MRv1
Worker's actual retention order. The previous send's tensors stay in the PP
group until the next `isend_tensor_dict` naturally reaps them, after the current
forward has produced its output. With `I=1`, previous and current pairs can
therefore coexist. For `I>=2`, the existing inflight reservation already covers
those two slots. Reports preserve `outgoing_inflight_bytes=I × 2H` and add
`outgoing_retention_slack_bytes=max(0, 2-I) × 2H`; the latter is budget slack,
not an extra network payload. This bound is confined to the supported synchronous
path and does not claim an async/DBO retention bound.

A separate fresh TP1×PP2 stress acquisition tested two 511-token prompts with
one output each, a 512-token transport bound and `I=1`. Full transport-storage
generation replay observed a sender peak of **8,372,224 bytes**, including
previous/current output pairs and pending releases. The old one-pair budget
of 4,194,304 bytes was insufficient; the new 8,388,608-byte reservation covered
that observation with 16,384 bytes remaining. This supports the source-derived
bound for the fixed synchronous path and stress cohort. It was not Workspace
training data and does not establish complete native-allocation attribution,
an arbitrary-workload upper bound, or async coverage.

The descriptor's `resident_bytes_per_rank=2H` names one final reconstructed
pair, not all these simultaneously live copies. The complete buffer budget
remains reserved alongside weights, the fixed KV pool and other declared
memory; it is never divided by TP because a wire tensor was sliced. KV/cache
pinning, atomic all-stage reservation, pressure draining, swap DMA and feedback
lifetimes keep the existing policies.

The optional Worker Workspace collector observes the original send, receive
and natural-reap points without waiting or reaping early to manufacture a
closed epoch. Pending PP leases keep their real storage generations and release
completion across iterations; a final teardown drain is separate from formal
Workspace peaks. Registered PP and communication storage are budgeted in their
own categories, not added again as dynamic scratch. See the
[Worker Workspace workflow](workspace-and-capacity.md#collect-worker-level-prefill-and-decode).

## Supported scope and evidence limits

The contract is dense unquantized Qwen3, fixed MRv1 source revision, synchronous
output, no SP, and the verified default native PP path using the standard
NCCL/RCCL ProcessGroup stream-ordering contract. Async output consumption,
sequence parallelism, DBO, native DP/EP/context parallelism, broadcast PP,
speculation, unsupported resume/recompute boundaries and unknown ABIs remain
raw or rejected under the existing guards. Simulation support for caches and
CPU swap does not assert that vLLM provides those same native policies.

CPU tests can verify object semantics, interval accounting, rank/edge matching,
Profile preservation and virtual resource lifetimes. They do not validate a
live ROCm transport, measure RCCL accuracy or establish hardware prediction
accuracy. `hardware_validated` remains false without the separate hardware
evidence acceptance workflow.

The pinned primary-source boundaries are the [Worker and lazy intermediate
object](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/worker/gpu_worker.py),
[MRv1 input-copy and execution path](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/worker/gpu_model_runner.py),
[per-tensor P2P/AllGather policy](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/distributed/parallel_state.py)
and [Qwen2/Qwen3 intermediate model
contract](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/models/qwen2.py).

The stream-ordering boundary also follows pinned PyTorch 2.13.0 primary
sources: [ProcessGroupNCCL stream ordering and work
wait](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)
and [nonblocking receive work
return](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/distributed/distributed_c10d.py).
