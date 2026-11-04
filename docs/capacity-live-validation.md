# Fresh-Runtime physical KV capacity observations

The authorized two-card AMD host was tested with the local Qwen3-1.7B BF16
checkpoint and pinned synchronous eager MRv1. Each trial uses one GPU, one
immediate 16-token request and one output token, block16, no cache/chunk/mixed,
graphs, CPU offload or swap. These are per-device physical KV-pool configuration
observations, not a claim about maximum concurrent requests or arbitrary models.

## What was proved

The adapter executes the real LLM. It verifies actual Runtime/model/configuration
and native KV descriptors; unique underlying Tensor storage must equal the
requested pool, with aliased layer views counted once. Each point and repeat
owns a new process/Runtime. OOM-mutated Runtimes are never reused.

The tested block quantum is **1,835,008 bytes**. Values are aligned to it and the
actual native page requirements. An arbitrary requested budget that the Runtime
silently rounds down does not pass the storage-equality proof.

A valid OOM endpoint requires an actual typed allocator exception, an increase
in that device's `num_ooms`, and a failure at the original pinned allocator's
final `torch.zeros` call for the exact whole KV backing. Its requested shape,
element size, device, axis and effective configuration are preserved. Other
startup/load/warmup/assertion/admission/worker failures remain distinct or
censored. The validator continues to accept existing `torch.empty` evidence.

## Observed intervals and independent confirmation

Exploration and local-grid results are retained separately from confirmation.
The endpoints were frozen before new acquisition sessions repeated both twice.

| Device | Confirmed successful KV bytes | Confirmed native KV allocator OOM bytes | Resolution |
| --- | ---: | ---: | ---: |
| RX9070XT / gfx1201 | 12,977,176,576 | 13,153,337,344 | 176,160,768 bytes (168MiB) |
| RX7900XTX / gfx1100 | 21,491,613,696 | 21,843,935,232 | 352,321,536 bytes (336MiB) |

Each confirmation has two complete real requests at the lower endpoint and two
eligible typed allocation failures at the upper endpoint, no censored endpoint
and no observed repeat instability. Both report `BRACKETED`, while
`exact_threshold=null` and `hardware_validated=false` remain unchanged.
The intervals describe observed endpoints and resolution; they do not establish
monotonicity or an exact threshold between the points.

The safe adapter check also completed at **939,524,096 bytes (896MiB)**, bound to
one actual unique storage. Its native descriptor reported 512 blocks and the
same block quantum, and the Runtime finished the entire declared request.

## Retained failure and prediction gaps

RX7900XTX's local grid also tested **21,726,494,720 bytes** twice. Both workers
exited abnormally without a valid structured typed-OOM capture. These points
remain `ERROR` / `CENSORED`, not allocator OOM. Raw logs reach actual model
execution, but the supervisor cannot prove a precise failure phase from an
absent worker capture. The broad confirmed interval includes this known unknown
region: it is not a guarantee that every smaller configuration is usable.

The default simulator budgets are conservative. For example, both successful
RX9070XT confirmation trials are analytically predicted `PHYSICAL_INFEASIBLE`;
the upper endpoints are also predicted infeasible. Those false-positive
rejections remain in `physical_confusion`. The physical observations therefore
do not grant general memory-prediction acceptance. Fixed category reserves,
fragmentation/headroom and full-workload calibration still require matching
evidence rather than tuning the held-out endpoints until agreement.

Only a config hash and loaded architecture/tying proof are claimed for the
local checkpoint identity; the adapter does not assert independent full weight
content revision verification. Input/source/Runtime hashes, all exploratory and
censored trials, endpoint-freeze and fresh-process receipts are retained in the
separate evidence directory. No model weights, credentials or raw machine logs
are distributed with this report.
