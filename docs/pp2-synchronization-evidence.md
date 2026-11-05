# PP2 synchronization-resource evidence

Workspace measures GPU-owned requested storage. A live logical HSA signal is
not evidence that its backing belongs to dynamic GPU Workspace, and an absent
allocation callback is not evidence of zero bytes. Version 1.14 adds affirmative
ownership observations while retaining every independent GPU-backing obligation.

## Signal body and logical lifetime

The optional provider runs at the existing synchronized CPU boundary after the
formal iterations and before original Worker shutdown. It calls public
`hsa_amd_pointer_info`, queries the returned owner's actual device type, and
obtains `hsa_amd_signal_get_event_id`. It does not call
`hsa_amd_signal_value_pointer`, change signal attributes or destroy Runtime signals.
Its ordinary control creates and destroys only its own signals.

The installed headers determine structure size, alignment, enum values and
returned-field extents. Failed queries, UNKNOWN pointer types, truncated
structures and unproved owner/range identities remain partial. Actual Native
callback pairs independently identify each query by its observed cursor, time,
thread and original arguments; producer status labels do not supply a proof.

The [LLVM HSA signal ABI](https://llvm.org/docs/AMDGPUUsage.html#hsa-signals)
describes handles as structure addresses and warns that the structure varies
with the Runtime. The target's installed header and ordinary-signal control
therefore remain necessary. The [public pointer-information API](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/develop/api-reference/api.html)
also distinguishes UNKNOWN addresses and defines the owner as a preferred-access
agent. A CPU preferred-owner observation does not establish physical zero VRAM.

`CPU_QUERY_REGION` records a positively queried CPU region. It does not invent
an allocation birth, allocation generation or final free. Signal generations
still come from the actual successful create callback and handle; earlier
destroys of a reused address cannot close a later generation. Two stable
samples, the original target-device completion fence and A/B/C generation
replay are required. GPU use, migration, rebind or unexplained release invalidates
the relevant proof. A surviving logical handle remains explicitly live.

Event-associated signals and signals created inside a stream completion fence
have distinct ownership proofs. The latter require the exact successful parent
`hipStreamSynchronize`, original stream/device context, same-thread containment
and a later source-bound Worker DeviceSync before querying. No synthetic HIP
event is supplied for a stream-fence-owned signal, and parent success alone
does not establish its backing or final destruction.

Host unregistration is not excluded by its API name or its parent label. The
original HIP registration extent and lower HSA lock/unlock pointers, extents,
aliases, successful returns and actual ancestry must establish the relevant
range. Only a page-rounded range disjoint from the entire queried CPU region
can be shown unrelated. The page granule comes from the sampled process;
missing pointers, overlapping pages and incomplete lock histories remain partial.

## Hardware event page

A nonzero hardware event ID needs a second proof. The passive Linux x86-64
observer records original libc/UAPI operations; it initiates no KFD allocation,
event, mapping, wait or reset. It preserves original call arguments, results and
errno, and exposes loss, recursion, forwarding and unfinished-mutation counters.
Unsupported inherited KFD resources or ambiguous prefixes cannot be promoted.

Independent replay joins the public getter to one active KFD event generation,
then checks the following original facts:

1. Successful `CREATE_EVENT`, including its input BO handle and returned event
   ID, slot index and page offset. Event ID and slot index are distinct fields.
2. A successful GTT-only allocation for that exact opaque handle and generation.
   VRAM, USERPTR, unknown memory types and missing birth records are rejected.
3. Original GPU/render-fd `ACQUIRE_VM`, actual device/PCI identity, matching
   allocation mmap offset, successful shared CPU mapping and complete slot bounds.
4. No relevant intervening event destruction, BO free, fd replacement or mapping
   invalidation. Close/dup replacement, close-range, unmap, fixed mapping and
   remap effects are retained; changed input-only FREE/DESTROY arguments reject.

A separate cold ordinary-signal process proves the public getter/KFD event-ID
namespace through an actual nested creation. Its raw Native/KFD tapes and
control JSONL are hashed source evidence, shared across acquisitions without
entering the train/repeat capture-reuse set. Worker event-pool reuse is legal
only when the original event generation is still active at the actual query.

The final KFD prefix is saved after successful Native SDK stop and before
official SDK finalization. The post-query prefix must be an unchanged prefix
of that final tape. Neither snapshot is described as a whole-process final
lifetime. All source files, compiled ABI, combined DSO, actual libc/ROCr modules,
headers, kernel/module identity and device observations are independently bound.

## Accounting and promotion

The CPU signal span, queried CPU region capacity and unique driver GTT pages
are reported separately. These are observed capacities, not a complete private
CPU-object size or host-memory peak. The observer's own CPU storage is not
dynamic GPU Workspace. All weights, KV, Runtime, communication, PP transport,
allocator backing, private scratch and unknown native GPU storage still pass
their original ownership and budget audits.

Raw acquisition success is insufficient. Require complete source, domain,
Native, synchronization, Torch-generation, PP-transport and fixed-budget replay
before emitting a canonical Workspace dataset. Train with all predeclared
stage/rank/shape repetitions, freeze Profile bytes, then acquire a separate
Runtime repeat and use `validate --workspace-repeats`. Never refit on that repeat.

The tools remain pinned optional adapters. Cache, chunking, swap and mixed-batch
software composition is checked separately; a measured Profile covers only its
actual Runtime identity and exact retained shapes. Memory-instrumented durations
do not calibrate new stage latency. See the [scoped live record](workspace-stage-validation.md)
for completed results and the [audit tools](../scripts/rocm_workspace_audit/README.md)
for acquisition interfaces.
