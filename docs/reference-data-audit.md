# Reference data audit — 2026-11-04

The owner supplied two external sampling archives during version 1.5 work.
They were read as data. No archived script, agent prompt or setup instruction
was executed. Full archives, prompts and generated response text are kept
outside the repository and distribution. This report contains only evidence
boundaries, aggregate counts and provenance hashes.

| Archive | SHA256 |
| --- | --- |
| `mlsys_data_2026_1020(1).zip` | `39ad8554ea14a782470472fd75240498df4ef6d327e0069316532a7223ed0562` |
| `mlsys_9070xt_v1_release_20261104_r2(1).tgz` | `3b1098c339a0314a44fe0f1f828dee084e76bf15f771cf65567f4b157f3ae787` |

## RX 9070 XT release

The release declares one AMD Radeon RX 9070 XT, gfx1201, 16304 MiB, TP=1 and
PP=1. Host Torch enumeration lists another GPU first; the launch selects the
9070 XT through `HIP_VISIBLE_DEVICES=1`. Visible process indices must not be
confused with host indices. The actual vLLM version is
`0.29.1.dev0+g98dff2a81.d20261002`; the image tag says `0.29.0` and is not a
substitute. The release records Torch `2.13.0+rocm10.1.0`, HIP `7.16.26385` and
BF16 model storage. These are preserved archive declarations, not a new probe
of the remote machine in this session.

The importer observed 195 runs, 65 cells, 11700 request rows: 1950 warmup and
9750 measurement rows. The measured rows include baseline, diagnostic and
open-loop protocols, which retain their distinct startup/polling conditions.
Declared checksums of the known data files read by the importer match. The
audit reports its checked-file scope and unread manifest entries; it does not
claim to have replayed or validated every nested raw archive.

The package is useful request-service and coverage evidence, with limits:

- Client TTFT starts at planned arrival and ends at the first nonempty content
  chunk; send lag and client overhead remain present.
- `token_timestamps` stores SSE content-chunk times. A chunk can contain several
  generated tokens, and token IDs are absent. These are not individual server
  token-ready events or a source of exact per-token ITL.
- The 1350 C1 service rows are request observations. They do not identify each
  executed batch's queries, contexts and whole-stage duration. They cannot be
  inserted into a stage Profile as measured Prefill/Decode costs.
- Closed-loop arrivals depend on earlier completions. Replaying their measured
  send times is a conditional replay, not a new closed-loop counterfactual.
  Open-loop runs have separate independent arrival schedules.
- Device VRAM peaks and derived memory remainders do not attribute Workspace,
  graph memory or temporary tensor lifetimes.
- Qwen3 models declare tied embeddings. Qwen3-4B additionally has hidden width
  2560 and query-head width 4096. Version 1.6 represents those real widths,
  vocabulary tying and Norm structure through the strict Qwen3 importer; model
  geometry support no longer requires changing the source dimensions. The
  files declare BF16 without quantization and with sliding-window attention
  disabled. Representable geometry does not add the missing stage/Workspace or
  semantic server-token timing evidence.
- The traces contain no observed preemptions or overload, and no RCCL, TP/PP,
  multi-replica or CPU-swap measurement evidence. Fifty measured requests per
  run provide only limited p99 evidence.

## Legacy package

The earlier package is NVIDIA RTX 5080/CUDA data, with 48 runs and 2400 measured
request rows. It is not relabeled as AMD or RCCL evidence. Its closed-loop
request metrics and separate deep-profiling windows retain their own timing
boundaries. It lacks actual request-arrival and server per-token timestamps.

The known data files checked by the importer match their manifest entries.
`README.md` has a hash mismatch; the audit retains expected and actual hashes
without repairing source bytes. This discrepancy is reported independently
from schema parsing and successful command execution.

## Acceptance outcome

Both reference imports complete and preserve sanitized numeric request-service
records. Both report `hardware_validated=false`. Their canonical stage Profile,
server token trace and online-comparison readiness remain false. Audit-only
`validate-hardware` returns `INSUFFICIENT_EVIDENCE`; it does not fill the gaps
with synthetic timings or convert C1 latency into stage service.

The shipped acceptance example is independently marked `test_fixture`. It
exercises fitting, disjoint holdout, simulation and Compare under numerical
criteria, returning `TEST_FIXTURE_ONLY` even when those criteria pass. No live
AMD collection was performed in this implementation environment.
