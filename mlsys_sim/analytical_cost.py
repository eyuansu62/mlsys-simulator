"""Runtime-scoped empirical corrections over the existing nominal macros.

The stage intercept is unmodeled work inferred from independent observations,
not a measurement of CPU time. These entries never grant hardware acceptance.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math

from .schema import AnalyticalCostSpec, Experiment, SimulationError, StepEstimate


_SELECTORS = ("prefill_execution_mode", "decode_execution_mode",
              "prefill_attention_backend", "decode_attention_backend",
              "analytical_cost_timing_boundary")
_NONATTENTION = ("linear", "misc", "embedding_and_output")


def _invalid(message: str) -> None:
    raise SimulationError("INVALID_ANALYTICAL_COST", message)


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _digest(value: str) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _text(value: str) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= 512
            and not any(ord(char) < 32 for char in value))


def analytical_cost_identity(experiment: Experiment) -> str:
    """Hash model/device/runtime/layout plus the explicit stage-wall boundary.

    The cost boundary is separate from the historical whole-stage Profile ABI;
    using it here does not change that ABI or its existing identity hashes.
    """
    from .profiles import profile_identity

    identity = profile_identity(experiment)
    identity["runtime"]["timing_boundary"] = experiment.calibration.analytical_cost_timing_boundary
    # A wall-step intercept includes scheduling work. Query equality alone is
    # not permission to reuse it across unmeasured dispatcher/cache policies.
    identity["analytical_cost_policy"] = {name: getattr(experiment.strategy, name) for name in (
        "scheduler", "max_batch_size", "max_batch_tokens", "max_active_requests",
        "chunked_prefill", "prefill_chunk_tokens", "mixed_batching", "prefix_cache", "preemption")}
    return _sha(identity)


def validate_analytical_costs(experiment: Experiment) -> None:
    """Validate frozen parameter data without running or refitting a model."""
    calibration = experiment.calibration
    entries = calibration.analytical_costs
    for name in _SELECTORS:
        if not _text(getattr(calibration, name)):
            _invalid(f"calibration.{name} must be bounded nonempty text")
    if not entries:
        return
    if not isinstance(entries, tuple):
        _invalid("calibration.analytical_costs must be an immutable tuple")
    if len(entries) > 256:
        _invalid("analytical cost catalogs are limited to 256 entries")
    if calibration.backend != "analytical":
        raise SimulationError("UNSUPPORTED_MECHANISM", "Analytical costs cannot be added to Profile or synthetic timing")
    strategy = experiment.strategy
    if strategy.tensor_parallel_size != 1 or strategy.pipeline_parallel_size != 1 or strategy.replicas != 1:
        raise SimulationError("CALIBRATION_MISMATCH", "This analytical cost family is scoped to TP1/PP1/one replica")
    model = strategy.model
    if (model.weight_bytes != 2 or model.kv_bytes != 2 or model.activation_bytes != 2
            or model.weight_dequant != "none"
            or any(getattr(model, name) not in {"auto", "bfloat16"}
                   for name in ("weight_dtype", "kv_dtype", "activation_dtype"))):
        raise SimulationError("CALIBRATION_MISMATCH", "This analytical cost family requires its unquantized BF16 domain")
    if (calibration.runtime_name == "unspecified" or calibration.runtime_version == "unspecified"
            or not _digest(calibration.runtime_config_sha256)
            or calibration.analytical_cost_timing_boundary == "unspecified"):
        _invalid("Analytical costs require explicit Runtime name, version, configuration SHA and wall-time boundary")
    keys = set()
    for entry in entries:
        if not isinstance(entry, AnalyticalCostSpec) or entry.phase not in {"prefill", "decode"}:
            _invalid("Analytical cost entries require a prefill/decode phase")
        for name in ("execution_mode", "attention_backend", "calibration_id"):
            value = getattr(entry, name)
            if not _text(value) or value == "unspecified":
                _invalid(f"analytical_costs.{name} must name an explicit domain or calibration")
        if not _digest(entry.identity_sha256) or not _digest(entry.training_dataset_sha256):
            _invalid("Analytical cost identity and training dataset must have lowercase SHA256 hashes")
        if (not isinstance(entry.training_record_ids, tuple) or not entry.training_record_ids
                or len(set(entry.training_record_ids)) != len(entry.training_record_ids)
                or any(not _text(name) for name in entry.training_record_ids)):
            _invalid("Analytical cost training record IDs must be immutable, unique and nonempty")
        if entry.source_kind not in {"measured", "provided_trace", "test_fixture"}:
            _invalid("Analytical costs require a declared training source kind")
        if entry.source_kind == "test_fixture" and not calibration.allow_test_fixture:
            raise SimulationError("TEST_FIXTURE_NOT_ALLOWED", "Analytical test-fixture costs require explicit opt-in")
        for name in ("nonattention_scale", "attention_scale", "attention_fixed_us_per_layer", "runtime_fixed_us"):
            value = getattr(entry, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                _invalid(f"analytical_costs.{name} must be finite and nonnegative")
        for kind in ("batch_size", "query_tokens", "context_tokens"):
            lower, upper = getattr(entry, "min_" + kind), getattr(entry, "max_" + kind)
            floor = 0 if kind == "context_tokens" else 1
            if type(lower) is not int or type(upper) is not int or not floor <= lower <= upper:
                _invalid(f"analytical_costs {kind} bounds are invalid")
        key = entry.identity_sha256, entry.phase, entry.execution_mode, entry.attention_backend
        if key in keys:
            _invalid("Duplicate analytical cost domain")
        keys.add(key)


def _nominal_experiment(experiment: Experiment) -> Experiment:
    calibration = experiment.calibration
    fixture_needed = bool(calibration.workspace_profile_path or calibration.replica_workspace_profile_paths
                          or calibration.native_pp_communication_profile_path
                          or calibration.replica_native_pp_communication_profile_paths)
    calibration = replace(calibration, analytical_costs=(), compute_efficiency=1.0,
                          bandwidth_efficiency=1.0, launch_overhead_us=0.0, framework_overhead_us=0.0,
                          allow_test_fixture=calibration.allow_test_fixture if fixture_needed else False,
                          prefill_execution_mode="unspecified", decode_execution_mode="unspecified",
                          prefill_attention_backend="unspecified", decode_attention_backend="unspecified",
                          analytical_cost_timing_boundary="unspecified")
    return replace(experiment, calibration=calibration)


def _nominal_estimate(experiment: Experiment, phase: str, query_tokens, context_lengths,
                      stage_index: int) -> StepEstimate:
    from .performance import AnalyticalModel

    return AnalyticalModel(_nominal_experiment(experiment)).estimate_step(
        phase, query_tokens, context_lengths, stage_index)


def nominal_cost_features(experiment: Experiment, phase: str, query_tokens,
                          context_lengths, stage_index: int = 0) -> dict:
    """Return original nominal macro features, never a fit or a measured value."""
    if phase not in {"prefill", "decode"}:
        raise SimulationError("CALIBRATION_MISMATCH", "No calibrated feature family for mixed/recompute")
    estimate = _nominal_estimate(experiment, phase, query_tokens, context_lengths, stage_index)
    parts = estimate.breakdown
    return {"nonattention_ms": sum(parts[name]["duration_ms"] for name in _NONATTENTION),
            "attention_ms": parts["attention"]["duration_ms"],
            "layers": experiment.strategy.model.num_layers,
            "communication_ms": parts["communication"]["duration_ms"],
            "dequant_ms": parts.get("dequant", {}).get("duration_ms", 0.0)}


def analytical_cost_provenance(experiment: Experiment) -> dict | None:
    entries = experiment.calibration.analytical_costs
    if not entries:
        return None
    sources = sorted({entry.source_kind for entry in entries})
    return {"kind": "analytical_cost_v1", "profile_version": experiment.calibration.profile_version,
            "source": {"kind": sources[0] if len(sources) == 1 else "mixed_sources",
                       "reference": "Immutable embedded analytical-cost entries"},
            "hardware_validated": False, "source_kinds": sources,
            "timing_boundary": experiment.calibration.analytical_cost_timing_boundary,
            "calibration_ids": [entry.calibration_id for entry in entries],
            "training_dataset_sha256": sorted({entry.training_dataset_sha256 for entry in entries}),
            "interpretation": "Empirical runtime corrections, not direct CPU/launch measurements or hardware acceptance"}


def corrected_analytical_step(experiment: Experiment, phase: str, query_tokens,
                              context_lengths, stage_index: int = 0) -> StepEstimate:
    """Apply one exact-domain correction to nominal macros without double fees."""
    if phase not in {"prefill", "decode"}:
        raise SimulationError("CALIBRATION_MISMATCH", "No measured analytical-cost domain for mixed/recompute")
    calibration = experiment.calibration
    mode = getattr(calibration, phase + "_execution_mode")
    backend = getattr(calibration, phase + "_attention_backend")
    identity = analytical_cost_identity(experiment)
    selected = [entry for entry in calibration.analytical_costs
                if (entry.identity_sha256, entry.phase, entry.execution_mode, entry.attention_backend)
                == (identity, phase, mode, backend)]
    if len(selected) != 1:
        raise SimulationError("CALIBRATION_MISMATCH", "No unique analytical cost entry matches model/device/Runtime/phase/mode/backend")
    entry = selected[0]
    if (not entry.min_batch_size <= len(query_tokens) <= entry.max_batch_size
            or any(not entry.min_query_tokens <= value <= entry.max_query_tokens for value in query_tokens)
            or any(not entry.min_context_tokens <= value <= entry.max_context_tokens for value in context_lengths)):
        raise SimulationError("CALIBRATION_MISMATCH", "Analytical cost query is outside the retained shape/context domain")
    nominal = _nominal_estimate(experiment, phase, query_tokens, context_lengths, stage_index)
    parts = deepcopy(nominal.breakdown)
    for name in _NONATTENTION:
        part = parts[name]
        part["nominal_duration_ms"] = part["duration_ms"]
        part["empirical_scale"] = entry.nonattention_scale
        for field in ("duration_ms", "compute_ms", "memory_ms"):
            part[field] *= entry.nonattention_scale
    attention = parts["attention"]
    attention["nominal_duration_ms"] = attention["duration_ms"]
    attention["empirical_scale"] = entry.attention_scale
    fixed_attention = experiment.strategy.model.num_layers * entry.attention_fixed_us_per_layer / 1000
    for field in ("duration_ms", "compute_ms", "memory_ms"):
        attention[field] *= entry.attention_scale
    attention["duration_ms"] += fixed_attention
    attention["overhead_ms"] = fixed_attention
    attention["attention_fixed_us_per_layer"] = entry.attention_fixed_us_per_layer
    parts["runtime_fixed"] = {"duration_ms": entry.runtime_fixed_us / 1000,
                              "interpretation": "Empirical unmodeled stage intercept, not a direct CPU time measurement"}
    duration = sum(part["duration_ms"] for part in parts.values())
    if not math.isfinite(duration) or duration <= 0:
        raise SimulationError("INVALID_ANALYTICAL_COST", "Corrected duration must be finite and positive")
    parts["cost_calibration"] = {"family": "nominal_scaled_attention_fixed_v1",
        "calibration_id": entry.calibration_id, "identity_sha256": identity,
        "phase": phase, "execution_mode": mode, "attention_backend": backend,
        "timing_boundary": calibration.analytical_cost_timing_boundary,
        "source_kind": entry.source_kind, "training_dataset_sha256": entry.training_dataset_sha256,
        "training_record_count": len(entry.training_record_ids),
        "training_record_ids_sha256": _sha(entry.training_record_ids), "hardware_validated": False}
    return StepEstimate(duration, parts, "analytical_cost_calibrated", (
        "Runtime-scoped empirical correction over the original nominal analytical macros.",
        "Runtime fixed cost is an inferred unmodeled intercept, not a direct CPU or launch measurement.",
        "Execution mode names identify configured dispatch; actual graph/eager execution requires separate observation.",
        "Fixed coefficients and macro scales are neither semantic quality nor hardware acceptance.",
    ))
