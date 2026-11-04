# Limited live Workspace attribution

The complete live Workspace domain is RX9070XT/gfx1201, Qwen3-1.7B BF16,
TP1×PP1, synchronous pinned MRv1, eager/no graphs, one uncached/unmixed prefill,
context zero and greedy one-token output, at **16 and 32 query tokens only**.
It does not extend to decode, other shapes, TP/PP ranks or scheduling policies.

The byte basis is **owned requested-storage capacity**. Caching-allocator
rounding, inactive reserved arenas, fragmentation and driver overhead remain
separate capacity/headroom evidence. This is not an allocator peak minus a
baseline, and it is not a claim about all physical device memory.

## Attribution evidence

Collection used the installed PyTorch2.13.0+rocm10.1.0 build, actual Torch commit
`a8ec1f47c5b233927bf23f5cd8f68db3475af226`, HIP7.16.26385 and
ROCprofiler SDK1.4.1. Earlier upstream cf30153 source analysis was not treated
as proof of this vendor binary. Actual headers, symbols and ABI were checked.

The pipeline joins independent evidence rather than trusting a completeness
label from any one component:

- The final public driver reads actual vLLM/Torch/device and hardware versions,
  Runner parallelism, scheduling and cache configuration. The merger checks
  those observations, their execution-config SHA and the epoch/session records.
  Unsupported declarations fail; legacy sessions missing these new observations
  remain partial under the current merger.
- Real parameter, shared KV storage and fixed Runtime tensor objects are checked
  against a narrow field/shape/dtype whitelist. Tied embedding aliases count once;
  empty views retain their real backing capacity and cannot establish new owners.
- The real warmed ATen LT workspace getter returns an existing 80MiB allocation;
  its pointer and capacity match the entry allocator block, with no allocation
  caused by the readback. Unneeded regular BLAS initialization is excluded.
- Successful HSA API returns are correlated with normalized allocation records.
  Native GPU backing is matched to Torch reserved segments or an explicitly
  executed LT-constructor scope. Unknown backing prevents canonical output.
- The whole stable ROCr queue scratch pool is charged to the fixed Runtime
  reserve. Its internal reuse is not added again as per-step Workspace. The
  observed fixed Runtime component bound is **210,783,373 bytes**, covered by
  the declared 1GiB Runtime budget.
- Full Torch history and a native allocator observer agree on ordered allocation,
  free-request and free-completion generations. Allocations belong to the
  designated execution thread. There are no surviving temporary generations,
  unknown pools, mapping events, trace loss or unclassified owners.
- Nine SDK callback domains retain the full startup trace, actual CPU/GPU agent
  inventory, typed statuses, correlation chains and successful explicit stop/drain.
  ModelRunnerOutput is CPU-only after the original synchronous sample returns.

Device-side malloc/read/write/free positive controls remain visible through
successful HSA allocation/free calls after warmup. A private-array kernel reports
1040 bytes of local storage and correct results; its first backing allocation is
visible, while later reuse has no allocation event. This is why the entire
registered scratch backing is budgeted, rather than inferring zero scratch from
zero callbacks. Positive controls and exploratory attempts are not fitting rows.

Exact read-only HSA queries and host-registration cancellation have separate
audits. Raw operations and conservative classifier flags are preserved. Host
unpin is accepted only outside formal epochs, with paired successful HSA return
and a real paired successful copying ancestor; no host storage owner is inferred.
VMM, IPC, unknown free, failed calls and unsupported formal mutations still fail.

## Frozen Profile and independent repeat

Three real observations per shape were accepted into the training dataset.
The public `calibrate --workspace` command produced `exact_rank_max_v1`.
The final public-tool Profile was frozen at **2026-11-04 17:20:44.295733 UTC**, before starting
the independent repeat Runtime. Its file SHA256 is
`cb302436e7ae8e07d2235ce533da2fcd7ef828ab6e2755d4b3161a1d8450f60e`.
The executed public driver SHA256 is
`4c7c4b7376c2593fee07a6eacfb6ec6230165ae19d512184482edd9bedf0bec1`.

| Exact shape | Training observed maximum | Independent repeat, three observations |
| --- | ---: | ---: |
| Prefill `[16]`, context `[0]` | 982,284 bytes | 982,284 bytes each |
| Prefill `[32]`, context `[0]` | 1,703,940 bytes | 1,703,940 bytes each |

Both processes passed the complete attribution audit and exited normally.
Their actual process IDs, sessions, record/group IDs and raw-input hashes are
distinct. The frozen Profile was unchanged, exact coverage was 100%, and no
repeat exceeded its selected Workspace budget. The Profile also entered the
normal prediction path successfully. There was no fitting on repeat rows.

These final public-source sessions ran from 17:20:01 to 17:21:27 UTC and
passed the new observed-domain gates. An earlier training/repeat pair and its
15:59 frozen Profile (`3147b2f39c25facc51d6f3102bb4cc489b6ea4deab3aeb75b96612638f39a0b6`)
are retained unchanged as historical evidence. They are not relabeled as new
acquisitions or patched with missing observations.

Attribution completeness, within-shape repeat agreement and physical OOM are
separate claims. The returned hardware status stays false; this record does not
establish broad workload capacity, online latency/SLA or a universal upper bound.
The [capacity workflow](workspace-and-capacity.md#sweep-capacity-with-a-fresh-runtime)
tests actual allocator OOM separately.

## Reproduction and limits

The optional sources in `scripts/rocm_workspace_audit` compile against an already
installed SDK and Torch ABI; they introduce no required simulator dependency.
Each individual probe/auditor always reports `complete_workspace=false`. Only
the final merger may generate a measured Workspace dataset after every gate,
actual driver/source binding, fixed-budget and return-value proof passes.
Any backing growth, unsupported shape/path, missing evidence or failed teardown
keeps that acquisition partial and prevents a canonical dataset.

Raw logs, unsuccessful setup attempts, full traces, acquisition receipts and
input/source hashes stay in the separate verification directory. Credentials,
connection addresses, model weights and raw traces are not distributed here.
