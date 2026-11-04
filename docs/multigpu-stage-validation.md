# Independent multi-GPU stage observations

The two actual devices were used as **TP2×PP1** and **TP1×PP2** separately.
This is not four-card TP2×PP2 evidence. The local Qwen3-1.7B BF16 checkpoint ran
under the pinned synchronous MRv1, eager/no graphs, max model/batched tokens512,
one active sequence, KV256MiB, no prefix/chunk/mixed, greedy four-token output.
Custom AllReduce was disabled; P2P used the explicit two-zero-threshold policy.

## Independent capture and freeze

Each training/validation acquisition owns a new process, actual session and
communicator cohort. Every selected prompt length first ran a real warmup;
the unmodified raw Runner and Scheduler exports retain those observations.
All raw data entered the public vLLM normalizer before a separate formal-stage
selector matched source session/replica/stage/sample tuples and actual
`CLOCK_MONOTONIC` intervals. It did not infer warmup from opaque IDs or remove
slow rows. Ambiguous boundary, missing rank or payload mismatch fails selection.

Training used lengths8/16/32, each repeated three times with four output tokens.
The formal datasets have 36 TP2 stage observations and 72 PP2 observations
(36 per stage). Twelve and 24 warmup observations respectively were excluded
with full source mapping and exclusion lineage, and remain in raw evidence.

Profiles were frozen at **2026-11-04 16:37:25 UTC**, before the independent
validation output directories existed. File SHA256 values:

- TP2: `1529d5266200dd11df88cf23215fdf103b44d2028a8eeca97c600f6ce88b9641`
- PP2: `e7bb8c0b0066c5f565b68025632b541c02bace0f8ca05210cab205c2474e0ceb`

The original whole-stage Profile contract includes local TP collectives once;
native PP receive waits/receiver AllGather are excluded from isolated stage
service and are not converted into primitive network samples.

## Results and distinct scopes

| Metric | TP2 unseen-shape holdout | PP2 independent exact-shape repeats |
| --- | ---: | ---: |
| Formal observations | 24 | 72 |
| Coverage | 100% | 100% |
| MAPE | **2.9520%** | **6.1565%** |
| MAE | 0.723308ms | 0.865454ms |
| Maximum absolute error | 2.048525ms | 5.792152ms |
| Extrapolated observations | 0 | 0 |

Both met the predeclared coverage≥95% and MAPE≤15% targets, without refitting
from validation data.

TP2 validation used new lengths12/24, repeated three times, with four outputs
each. Its prefill and decode shapes do not intersect training shapes. The
original grouped-holdout leakage checks and supported interpolation rules
were retained, and no extrapolation was enabled.

Native PP remains exact-only. PP2 validation therefore used new observations
at retained lengths8/16/32 in a fresh Runtime. The explicit
`validate --exact-repeats --training-dataset ... --profile ...` path checks the
frozen training canonical SHA, recomputes its fitting provenance, and rejects
reused session/capture/record/group evidence. Exact payload descriptors must
match. It does not claim unseen-shape generalization or relax ordinary holdout
or hardware-acceptance leakage rules.

Full normalized `observed.json` includes warmups and is not mislabeled a
formal-only online cohort. These results cover isolated stage latency at the
stated short shapes, software and placement. They do not establish online
TTFT/TPOT tail, throughput, SLA, concurrent-policy ordering or broad model
accuracy. Both evaluators retain non-asserted hardware status. The complete
single-card Prefill Workspace evidence is a different domain and is not
silently reused for multi-rank Workspace.

Raw captures, formal-selection lineages, fresh-process/freeze receipts and
source hashes are kept in the separate verification directory. Only this
sanitized report and software fixtures are distributed with the repository.
