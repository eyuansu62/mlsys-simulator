"""Conditional costs for a declared CPU sibling exposure during a short C2 study."""

import math

from .schema import SimulationError, StepEstimate


def _invalid(message):
    raise SimulationError("INVALID_SIBLING_SCENARIO", message)


def validate_scenario(scenario, spec):
    if not isinstance(scenario, dict) or set(scenario) != {
        "kind",
        "base_model_sha256",
        "increment_ms",
        "windows",
    }:
        _invalid("A declared sibling exposure needs a base model, increments and per-repetition windows")
    if scenario["kind"] != "declared_c2_sibling_exposure_v1":
        _invalid("Unknown sibling exposure scenario")
    digest = scenario["base_model_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        _invalid("Expected a frozen runtime model SHA256")
    if (
        spec["concurrency_grid"] != [2]
        or set(spec["input_lengths"]) != {64, 128}
        or spec["output_tokens"] != 16
        or spec["think_ms"] != 0
        or spec["requests_per_run"] % 2
    ):
        _invalid("Sibling exposure requires full synchronous short C2 pairs with 16 output tokens")
    values = scenario["increment_ms"]
    if (
        not isinstance(values, dict)
        or set(values) != {"prefill", "decode"}
        or not all(type(x) in (float, int) and 0 <= x <= 1000 for x in values.values())
    ):
        _invalid("Each phase requires a finite nonnegative declared cost in milliseconds")
    reps = scenario["windows"]
    if not isinstance(reps, list) or len(reps) != spec["replications"]:
        _invalid("Declare the exposed step ranges for every repetition")
    limit = spec["requests_per_run"] // 2 * 16
    for rep in reps:
        end = 0
        if not isinstance(rep, list) or len(rep) > limit:
            _invalid("Invalid bounded window list")
        for pair in rep:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(x) is not int for x in pair)
                or not end <= pair[0] < pair[1] <= limit
            ):
                _invalid("Exposure ranges must be ordered nonoverlapping half-open step ranges")
            end = pair[1]


class SiblingExposure:
    def __init__(self, base, scenario, replication):
        if (
            not isinstance(base.provenance, dict)
            or base.provenance.get("model_sha256") != scenario["base_model_sha256"]
        ):
            _invalid("Runtime model does not match the declared exposure calibration")
        self.base = base
        self.windows = scenario["windows"][replication]
        self.increment = scenario["increment_ms"]
        self.at = 0
        self.steps: list = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def estimate_step(self, phase, query_tokens, context_lengths, stage_index=0, *, request_phases=None):
        if stage_index or len(query_tokens) != 2 or phase not in ("prefill", "decode"):
            _invalid("Actual step is outside the calibrated C2 study")
        nominal = self.base.estimate_step(
            phase, query_tokens, context_lengths, stage_index, request_phases=request_phases
        )
        index = len(self.steps)
        if self.at < len(self.windows) and index >= self.windows[self.at][1]:
            self.at += 1
        exposed = self.at < len(self.windows) and self.windows[self.at][0] <= index < self.windows[self.at][1]
        delta = self.increment[phase] if exposed else 0.0
        duration = nominal.duration_ms + delta
        if not math.isfinite(duration):
            _invalid("Nonfinite sibling exposure duration")
        self.steps.append([phase, list(query_tokens), list(context_lengths), duration])
        if not exposed:
            return nominal
        return StepEstimate(
            duration,
            {
                **nominal.breakdown,
                "declared_sibling_exposure": {
                    "step_index": index,
                    "increment_ms": delta,
                    "base_ms": nominal.duration_ms,
                },
            },
            "conditional_cpu_sibling_exposure",
            (*nominal.assumptions, "Exposure schedule supplied before simulation."),
        )
