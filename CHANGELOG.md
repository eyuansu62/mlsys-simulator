# Changelog

## Unreleased

Initial development version, with package version `0.1.0`. No public release has
been made; license selection remains pending.

### Added

- CPU-only deterministic serving simulation with FIFO and prefill-first
  continuous batching.
- Strict configuration, Dense MHA/GQA macro performance models, TP communication
  and analytical PP stage reports.
- Per-rank physical budgets, fixed KV block pools, atomic reservation and
  request preemption with recomputation.
- Request timelines, TTFT/TPOT/ITL, goodput, memory diagnostics and reproducible
  run artifacts.
- Python library exports, an installable CLI, distribution configuration,
  contributor documentation and CI checks.

### Limits

- Timing profiles are synthetic or uncalibrated analytical estimates.
- Measured profiles, AMD validation, online PP, CPU swap, multiple replicas,
  prefix caching, chunked prefill and quantized storage formats remain future
  work.
