# Historical measured study inputs

`frozen-commitment/` preserves the exact input and receipt bytes used for the
2026-11-11 RX9070XT/Qwen3-4B course study. These files are historical records;
editing them invalidates the measured study's commitments. They are not a new
prediction freeze for a different host, model revision or Runtime.

The complete prediction payloads are locally archived rather than distributed
here. Their raw/canonical hashes were committed to the measurement host before
GPU collection, and exactly revealed from frozen code and inputs after it.
`predictions-summary.json` contains the compact pre-measurement metrics; the
full raw request/token captures remain in the private evidence archive.

For a new study, create a new output directory, probe the actual hardware and
checkpoint, bind a new experiment and generate all predictions before collecting
new observations. Never reuse this historical receipt as evidence that a new
prediction preceded a new measurement. JSON-normalize experiments through
`experiment_from_dict` / `load_experiment` before hashing or predicting.

See the [complete measured report](../../../docs/course-sla-9070xt-20261111/README.md)
and [protocol](../../../docs/closed-loop-validation.md).
