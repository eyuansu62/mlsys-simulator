# Scoped Worker Workspace validation

This record separates completed Workspace validation from diagnostic component
observations. Every artifact in this study retains `hardware_validated=false`.
The repository contains the capture and audit code; raw traces, build receipts,
failures and experiment artifacts remain in the private evidence archive.

The measured domain is Qwen3-1.7B BF16, vLLM
`0.29.1.dev0+g98dff2a81.d20261002`, Torch `2.13.0+rocm10.1.0`, eager MRv1,
FIFO with one sequence, 512 model/batch tokens and a 256 MiB KV pool. Prefix
cache, chunked prefill, mixed batches and CPU KV swap are disabled in this actual
domain. Their software composition remains a separate regression concern.
The local model identity binds its configuration and effective loaded geometry;
it does not claim a content revision for the weight files.

Only two physical cards were available: RX 7900 XTX (`gfx1100`) and RX 9070 XT
(`gfx1201`). The actual placements are single, TP2 and PP2. These observations
provide no four/eight-card, unseen-shape, general serving/SLA or physical OOM
acceptance result.

## Completed single-card validation

The training and repeat each used a fresh Runtime. Each retained all warmup and
24 formal records: prompt lengths 16/32, each generating four outputs naturally,
with three repetitions. The resulting eight shapes contain two prefills and
six decode contexts: 16/17/18 and 32/33/34.

The independent native backing and Torch storage-generation audits completed
before training. The Profile was frozen at `2026-11-05T03:00:09.823531+00:00`,
before the independent repeat acquisition, with SHA256
`383fd907f65f09b4b46f85a1ddba0505aa927e11627c12fbc6b472eaa8593daa`.
The public `validate --workspace-repeats` path reported record, rank and retained
shape coverage of 1, zero underpredictions and a maximum absolute error of zero
bytes across the 24 repeat records. The original Profile bytes stayed unchanged.
The observed decode requested-storage peak was 920,844 bytes.

The ordinary public `simulate` CLI also consumed this measured Profile with
nonempty 16/32-token, four-output requests. Its memory report used
`workspace_source=workspace_profile`, retained the exact frozen Profile SHA256
in its snapshot provenance, and reported a 1,703,940-byte Workspace peak across
the prefill/decode workload. This checks Profile consumption; the simulation's
analytical timing is not an additional serving-accuracy measurement.

The fitted values replace dynamic Workspace only. Weight padding, runtime,
communication, KV and native PP transport retain their own admitted budgets.
Allocator reserved capacity, fragmentation and driver headroom remain separate
capacity evidence.

## Completed TP2 validation

The TP2 training and repeat each used a separate Runtime and retained 24 formal
records, with 48 rank observations and the same eight exact Prefill/Decode
shapes. Full source/domain, Native, Torch-generation, synchronization and fixed
budget replay passed independently for each cohort before dataset admission.
Each rank's 20 communication backings, totaling 17,113,392 bytes, were separately
attributed and kept outside dynamic Workspace.

The public `calibrate --workspace` CLI fitted the training dataset. The Profile
was frozen at `2026-11-05T05:49:37.167722+00:00`, before the independent repeat,
with SHA256
`f2e6a8a3024586e086e1e9d8c1327b93a391d81c920c283e1c09ecbeac6c546c`.
The public `validate --workspace-repeats` CLI then reported record, rank and
retained-shape coverage of 1, zero underpredictions and maximum absolute error
of zero bytes. The original Profile bytes stayed unchanged. This establishes
only requested-storage Workspace at retained exact shapes in this TP2 domain.

The Native replay also required real teardown ancestry: releases on RCCL's
cleanup thread were admitted only for already registered objects, inside the
original shutdown window and a successful same-thread, contained NCCL cleanup
call. Thread names, generic shutdown labels and epoch tags alone were
insufficient; this rule did not assign new allocations or relax PP resource
checks.

## PP2 completed attribution and calibration

Version 1.14's source-bound HSA and passive KFD evidence completed the PP2
formal training attribution: 48 canonical records and rank observations across
16 stage-specific Prefill/Decode shapes. Native, synchronization, Torch, PP
transport, source/domain and fixed-budget audits all passed with no issues.
The public Workspace calibration CLI fitted this complete training acquisition.
The Profile was frozen at `2026-11-05T11:41:52.680458+00:00` with SHA256
`93850258993ff25bd3998c3d18c31ff256dd42779f888a78db9206935dee45fe`,
before a separate Runtime repeat was started. That repeat independently passed
complete attribution with 48 records and rank observations. The public
`validate --workspace-repeats` path reported record, rank and retained-shape
coverage of 1 across all 16 stage-specific shapes, zero underpredictions,
maximum underprediction of zero bytes and maximum absolute error of zero bytes.
The original frozen Profile bytes stayed unchanged. This establishes the
retained exact requested-storage domain only, without latency, unseen-shape,
physical-capacity or general hardware-acceptance claims.

The ordinary public `simulate` CLI also consumed the frozen PP2 Profile for two
nonempty 16/32-token requests, each generating four outputs. Both stages used
`workspace_source=workspace_profile`, with a 1,572,868-byte Workspace peak each,
and the original frozen Profile SHA stayed unchanged. This checks normal
Profile consumption; analytical timing is not a serving-accuracy measurement.

The new evidence does not pretend that surviving HSA handles were destroyed.
Actual public CPU regions, original KFD GTT allocations, event/mapping
generations, stream ownership and unrelated host cleanup ranges are replayed
independently. See the [synchronization evidence contract](pp2-synchronization-evidence.md).

## Retained earlier PP2 diagnostics

Actual TP2 and PP2 worker acquisitions retained rank/stage, prepared-batch,
return, Torch-generation and native callback evidence. Independent component
replay closed 80 earlier formal Torch epochs, including both PP transport
ledgers. Candidate decode peaks were 920,844 bytes for single/TP2 and PP2's final
stage, and 49,156 bytes for PP2's first stage. These earlier component numbers
alone do not constitute complete attribution. TP2 subsequently completed the
separate formal workflow above. These earlier PP2 captures remain partial under
their own recorded evidence; new observations are not backfilled into them.

A later PP2 diagnostic kept the native SDK window open until the original
`WorkerProc.shutdown` returned. Both owned workers exited normally, and the
actual CPU `std::thread` control plus RCCL vtable/typeinfo/dispatch module facts
established the two proxy allocations that earlier diagnostics had left
unknown. The raw acquisition and the independent audits are preserved.
Its independent Torch replay also closed all 16 formal candidate epochs, and
both PP transport-generation ledgers passed again.

PP2 attribution remains partial whenever any native API purpose, backing or
synchronization-resource lifetime is unproved. In particular, a successful
Worker shutdown does not prove an opaque HSA signal generation's ownership or
final lifetime. A generation without a matched later destroy is not paired with an earlier destroy
of the same address, supplied with an unsupported pointer-provider call, or
converted to zero bytes. No PP2 canonical dataset, fit or repeat result
is admitted from a partial audit.

A separate one-repeat PP2 process-exit diagnostic retained the original Runtime
shutdown and allowed the SDK's automatic tool-finalization callback to run.
Both workers exited with code zero and produced actual final native files.
Both SDK stop attempts nevertheless returned code 2,
`ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND`, with
`observation_window_closed=false`. Those return codes and the failed closure
remain recorded. Neither successful process exit nor a final file substitutes
for a proven observation window; this diagnostic stays ineligible for fitting.
Full callback replay still found no later destruction for the new formal signal
generations: zero of eight on the sender and zero of sixteen on the receiver.
Other destructor records did not match those generations and were not reused
as closure evidence.

This earlier PP result identifies missing ownership/lifetime evidence for opaque
synchronization resources. It does not establish a GPU-memory leak, an OOM
condition or a Workspace byte count. Passed Torch/PP transport and individual
native backing proofs remain separate from complete PP Workspace calibration.

## Reproduction and promotion gates

Use the [audit tools](../scripts/rocm_workspace_audit/README.md) with a new output
directory and preserve the actual parent-process exit receipt. Executed source,
probe DSO and loaded-library hashes must still match their immutable capture
paths. The assembler independently verifies actual ProcessGroup controls and
function/file hashes, raw `std::thread` control/header/runtime-library facts,
and natural shutdown receipt bytes, original method source and actual worker
exit outcomes. Producer success labels alone cannot promote evidence.

The assembler CLI returns zero when analysis completes, including a partial
analysis. Require `attribution_complete=true`, no issues and an emitted
`dataset.json` before invoking `calibrate --workspace`. A formal dataset must
contain all ranks and all predeclared shapes from three complete repetitions.
Freeze the fitted Profile bytes and timestamp before a separate Runtime repeat,
then invoke `validate --workspace-repeats --training-dataset ... --profile ...`.
The repeat must have full record/rank/retained-shape coverage and zero
underprediction; it must never be used to refit the frozen Profile.
