# Validation protocol

`validate-hardware --plan FILE --out DIR` evaluates the explicit collection and
acceptance workflow described in [ROCm collection](rocm-collection.md). Use
`--require-pass` in automation to exit with code 3 unless its scoped measured
verdict is `PASS` and `hardware_validated=true`. Without that flag, successful
evaluation exits 0 even when its verdict is `FAIL`, `INSUFFICIENT_EVIDENCE` or
`TEST_FIXTURE_ONLY`; inspect the verdict separately from execution status.

V1 separates software correctness, calibration fit and prediction validation.
The supplied fixtures exercise the workflow. User-provided hardware packages
are audited as external evidence; raw packages are not included in this
repository and do not establish general hardware accuracy.

## Establish a comparable experiment

Before collecting data, fix the model/revision, device and topology, Runtime/
driver/library versions, precision, TP/PP layout and timing boundary. Use
`profile_identity(experiment)` for step datasets and
`comparison_contract(experiment)` for online observations. Preserve the raw
records and their collector reference.

A whole-stage sample includes its local collective and output/sampling costs;
queueing is separate. Online TTFT starts at the declared request arrival.
Warmup, initial state, admission window, drain and timeout rules must match.
Deep profiling for attribution should be a separate run from low-interference
performance timing, with overhead documented by the collector.

## Hold out complete configurations

```bash
mlsys-sim validate --dataset examples/v1/step_measurements.fixture.json \
  --profile-version holdout-v1 --holdout-group prefill-16 \
  --out runs/holdout --allow-test-fixture
```

This mode splits data, fits a new Profile on the training partition and evaluates
the holdout. Repeated shapes and shared group IDs form connected components;
components stay intact even when the same shape appears under another group.
`--holdout-group` may repeat. Alternatively use `--holdout-fraction 0.25 --seed
42`; the fraction applies to independent components, rounded up while retaining
at least one component in each partition.

The directory contains dataset, training, holdout and Profile snapshots plus
manifest/result/logs. To evaluate an existing Profile, use `--profile FILE` and
pass an already held-out dataset; split options are not accepted in that mode.
Any overlap of record IDs, group IDs or shape keys with training fails as
`VALIDATION_DATA_LEAKAGE`.

Read coverage, rejected records, MAE in ms, MAPE in percent, bias, group errors
and sample counts together. Missing grid corners or out-of-domain shapes remain
rejections, not zero error. `--allow-extrapolation` enables explicitly labeled
exploratory predictions; report them separately. A step dataset has no arrival
queue, so its holdout report leaves online SLA and candidate ranking unavailable.

## Compare an online run

```bash
mlsys-sim compare --predicted runs/profile-serving \
  --observed examples/v1/online_observation.fixture.json \
  --out runs/compare --allow-test-fixture
```

Run the preceding Profile-backed simulation from the README first. The observed
JSON uses envelope `kind="measured_run"`, a declared `source`, a complete
`comparison_contract`, a flat `metrics` object and optional request rows. The
fixture envelope tests this format; its `source.kind="test_fixture"` keeps it
from being mistaken for a collected measurement.

The prediction must be a saved simulated run with manifest and result. Hardware,
workload, policy, SLA, warmup, initial state, window/drain and timing contracts
must match exactly. Simulated results cannot serve as observed measurements.

Metrics are compared in their own units. Missing/null values report a reason
and reduce coverage. Relative error at an observed zero is unavailable; absolute
error is still reported. Request rows pair by request ID and preserve unpaired
counts. Reports also record SLA judgment agreement and limits on tail evidence.
Do not average ms, requests/s and fractions into one aggregate error.

## Compare candidate rankings

Supply `--pairs FILE`, where the JSON array contains one object per candidate:
`candidate_id`, `predicted` run path and `observed` JSON path. Paths resolve
relative to that pairs file. Every prediction/observation pair must match its own
contract, and candidates must share workload, evaluation, execution and timing
conditions.

Select a numeric metric with `--metric`; the default is
`goodput_requests_per_s`. Use `--lower-is-better` for latency. Reports retain tied
ranks, pairwise ordering agreement and candidates excluded by unavailable scores.
A single candidate has no pairwise agreement value. Step holdout accuracy alone
cannot establish online candidate ordering.

## Report hardware evidence separately

A hardware-validation report needs actual measurement provenance, held-out
configurations, request/sample counts, repeated-run variation and domain
coverage. Tail percentiles from a tiny sample are a correctness demonstration,
not reliable p99 evidence. Define acceptable error and SLA/ranking criteria for
the intended experiment before judging its results.

Individual calibration, holdout and comparison tools retain calibrated or
fixture provenance and do not automatically set `hardware_validated` to true.
Version 1.5 adds a separate `validate-hardware` acceptance gate with explicit
sample, repeat, coverage and error thresholds. Its scoped `PASS` requires
measured inputs and satisfied criteria. Any required fixture produces
`TEST_FIXTURE_ONLY`; missing evidence produces `INSUFFICIENT_EVIDENCE`.
Completing a workflow does not fill in missing evidence or license/publication
steps. See [ROCm collection and acceptance](rocm-collection.md).

## Workspace and capacity evidence

Independent Workspace validation starts with complete Runtime-owned attribution,
not allocator deltas or passive manager capacity. Exact per-rank Profiles use
repeated maxima with observed zero distinct from missing evidence. Their source
and coverage do not establish unseen-shape or hardware accuracy.

Capacity validation uses the full effective configuration and a fresh Runtime
for each candidate/repeat. The supported OOM boundary requires a typed
verified-device exception and a failure-phase-bound allocator `num_ooms`
increase. Pre-workload OOM does not close a workload-axis bracket. Logical KV
refusal, timeout, lock AssertionError, peer failure and incomplete cohorts retain
separate classifications. Repeated disagreement is unstable, and
success beyond a smaller OOM is nonmonotonic. Missing endpoints are censored.
Reports may state adjacent observed success-to-OOM intervals and axis resolution;
they do not manufacture a precise universal knee. Fixture source remains fixture.
See [Workspace and capacity validation](workspace-and-capacity.md).
