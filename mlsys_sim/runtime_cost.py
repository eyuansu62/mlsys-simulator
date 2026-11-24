"""Explicit, bounded EngineCore-step costs; no GPU/CPU attribution is inferred.

Each serving phase and exact batch size has its own nonnegative affine model
over the existing joint AnalyticalModel. Raw source order and process-level
observations are retained as consistency evidence, not hardware attestation.
"""

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, NoReturn

from .batching import batch_members
from .schema import Experiment, InfraSpec, SimulationError, StepEstimate, StrategySpec


RUNTIME_COST_BOUNDARY = "engine_core_step_wall_v1"
RUNTIME_COST_KIND = "runtime_cost_model_v1"
RUNTIME_COST_REJECTION_KIND = "runtime_cost_rejected_v1"
RUNTIME_COST_ALGORITHM = "runtime_core_phase_exact_batch_affine_v1"
_DATASET_KIND = "runtime_cost_dataset_v1"
_POLICY_KIND = "runtime_cost_policy_v1"
_DOMAIN_KIND = "runtime_cost_domain_v1"
_OLD_BOUNDARY = "step_includes_output_and_sampling_v1"
_MAX_RECORDS = 100_000
_MAX_BYTES = 128 * 1024 * 1024
_REPLAY_REL_TOL = 1e-10
_REPLAY_ABS_TOL = 1e-12
_PHASES = {"prefill", "decode", "mixed"}
_RUNTIME_FIELDS = ("runtime_name", "runtime_version", "driver_version", "rocm_version",
                   "rccl_version", "runtime_config_sha256", "prefill_execution_mode", "decode_execution_mode",
                   "prefill_attention_backend", "decode_attention_backend")
_NOMINAL_FIELDS = ("compute_efficiency", "bandwidth_efficiency", "launch_overhead_us",
                   "framework_overhead_us", "dequant_flops_per_weight")
_BOUND_FIELDS = ("query_tokens", "context_tokens", "total_query_tokens", "total_context_tokens")
_LIMITATIONS = [
    "A bounded phase/exact-batch affine prediction is not proof that every shape inside its envelope was measured.",
    "The target is an equally weighted acquisition-session/exact-shape median, not a step P99 or a jitter model.",
    "The intercept is unmodeled EngineCore work, not measured CPU sampling or GPU execution time.",
    "EngineCore step wall includes schedule/execute/sample/publish; do not add runner or API wall time again.",
    "No batch interpolation, phase fallback, runtime/backend borrowing or hardware acceptance is granted.",
]


def _error(message: str, code: str = "INVALID_RUNTIME_COST") -> NoReturn:
    raise SimulationError(code, message)


def _canonical(value: Any) -> str:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(text.encode("utf-8")) > _MAX_BYTES:
            _error("Runtime-cost JSON exceeds its bounded size limit")
        return text
    except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError) as exc:
        raise SimulationError("INVALID_RUNTIME_COST", "Runtime costs require bounded finite JSON data") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _clone(value: Any) -> Any:
    return json.loads(_canonical(value))


def _object(value: Any, required: set[str], optional: set[str], name: str) -> dict:
    if (not isinstance(value, dict) or any(not isinstance(key, str) for key in value)
            or set(value) - required - optional or required - set(value)):
        _error(f"{name} has missing, unknown or invalid fields")
    return value


def _text(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > 4096
            or any(ord(char) < 32 for char in value)):
        _error(f"{name} must be bounded nonempty text")
    return value


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _error(f"{name} must be a bounded integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{name} must be finite numeric data")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_RUNTIME_COST", f"{name} must be finite") from exc
    if not math.isfinite(value) or value < minimum or (positive and value == 0):
        _error(f"{name} is outside its finite nonnegative domain")
    return value


def _digest(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        _error(f"{name} requires an explicit lowercase SHA256")
    return value


def _nominal_from_parts(strategy: dict, infra: dict, runtime: dict, nominal: dict) -> Experiment:
    from .config import experiment_from_dict

    # This is an internal analytical coordinate, not a relabeling of measured
    # EngineCore durations. The external domain always retains its real boundary.
    return experiment_from_dict({"strategy": strategy, "infra": infra,
        "calibration": {**runtime, **nominal, "backend": "analytical", "timing_boundary": _OLD_BOUNDARY}})


def _domain(value: Any) -> tuple[dict, Experiment]:
    required = {"schema_version", "kind", "timing_boundary", "strategy", "infra", "runtime",
                "nominal_calibration", "domain_sha256"}
    item = _object(value, required, set(), "runtime cost domain")
    if (type(item["schema_version"]) is not int or item["schema_version"] != 1
            or item["kind"] != _DOMAIN_KIND or item["timing_boundary"] != RUNTIME_COST_BOUNDARY):
        _error("Expected the explicit EngineCore-step runtime domain")
    strategy = _object(item["strategy"], {field.name for field in fields(StrategySpec)}, set(), "domain.strategy")
    infra = _object(item["infra"], {field.name for field in fields(InfraSpec)}, set(), "domain.infra")
    runtime = _object(item["runtime"], set(_RUNTIME_FIELDS), set(), "domain.runtime")
    nominal = _object(item["nominal_calibration"], set(_NOMINAL_FIELDS), set(), "domain.nominal_calibration")
    for name in _RUNTIME_FIELDS:
        _text(runtime[name], name)
    _digest(runtime["runtime_config_sha256"], "runtime_config_sha256")
    if runtime["runtime_name"] == "unspecified" or runtime["runtime_version"] == "unspecified":
        _error("Runtime cost domains require explicit runtime name and version")
    experiment = _nominal_from_parts(strategy, infra, runtime, nominal)
    if (experiment.strategy.tensor_parallel_size != 1 or experiment.strategy.pipeline_parallel_size != 1
            or experiment.strategy.replicas != 1):
        _error("This runtime-cost family requires TP1/PP1/one replica")
    normalized = {"schema_version": 1, "kind": _DOMAIN_KIND, "timing_boundary": RUNTIME_COST_BOUNDARY,
        "strategy": _clone(asdict(experiment.strategy)), "infra": _clone(asdict(experiment.infra)),
        "runtime": {name: getattr(experiment.calibration, name) for name in _RUNTIME_FIELDS},
        "nominal_calibration": {name: getattr(experiment.calibration, name) for name in _NOMINAL_FIELDS}}
    normalized["domain_sha256"] = _sha(normalized)
    if _canonical(item) != _canonical(normalized):
        _error("Runtime domain data, types or checksum differ from its complete normalized configuration")
    return normalized, experiment


def runtime_cost_domain(experiment: Experiment) -> dict:
    """Bind full strategy (including memory), GPU/model, Runtime and nominal basis."""
    cal = experiment.calibration
    if cal.backend != "runtime" or cal.timing_boundary != RUNTIME_COST_BOUNDARY:
        _error("Runtime costs require backend=runtime and the actual EngineCore-step wall boundary")
    if (cal.analytical_costs or cal.replica_profile_paths or cal.allow_extrapolation
            or cal.native_pp_communication_profile_path or cal.replica_native_pp_communication_profile_paths
            or cal.workspace_profile_path or cal.replica_workspace_profile_paths):
        _error("Runtime costs do not combine other empirical cost/Workspace/communication models or extrapolation")
    body = {"schema_version": 1, "kind": _DOMAIN_KIND, "timing_boundary": RUNTIME_COST_BOUNDARY,
        "strategy": _clone(asdict(experiment.strategy)), "infra": _clone(asdict(experiment.infra)),
        "runtime": {name: getattr(cal, name) for name in _RUNTIME_FIELDS},
        "nominal_calibration": {name: getattr(cal, name) for name in _NOMINAL_FIELDS}}
    # Normalize numeric configuration exactly as the ordinary parser does.
    nominal = _nominal_from_parts(body["strategy"], body["infra"], body["runtime"], body["nominal_calibration"])
    body["strategy"], body["infra"] = _clone(asdict(nominal.strategy)), _clone(asdict(nominal.infra))
    body["nominal_calibration"] = {name: getattr(nominal.calibration, name) for name in _NOMINAL_FIELDS}
    body["domain_sha256"] = _sha(body)
    return _domain(body)[0]


def _cell_key(phase: str, batch: int) -> str:
    return f"{phase}/batch-{batch}"


def _policy_cell(value: Any) -> dict:
    required = {"phase", "batch_size", "max_total_tokens", "max_sequence_tokens"} | {
        prefix + name for name in _BOUND_FIELDS for prefix in ("min_", "max_")}
    item = _object(value, required, set(), "runtime policy cell")
    phase = item["phase"]
    if not isinstance(phase, str) or phase not in _PHASES:
        _error("Runtime policies support prefill, decode and mixed")
    _int(item["batch_size"], "batch_size", 2 if phase == "mixed" else 1)
    for name in _BOUND_FIELDS:
        lower, upper = _int(item["min_" + name], "min_" + name), _int(item["max_" + name], "max_" + name)
        if lower > upper or ("query" in name and lower < 1):
            _error("Runtime policy bounds must be ordered and query budgets positive")
    if phase == "decode" and (item["min_query_tokens"] != 1 or item["max_query_tokens"] != 1):
        _error("Decode policies require exactly one query token per member")
    _int(item["max_total_tokens"], "max_total_tokens", 1)
    _int(item["max_sequence_tokens"], "max_sequence_tokens", 1)
    return deepcopy(item)


def freeze_runtime_cost_policy(cells: Sequence[dict], *, min_samples_per_shape_session: int = 8,
                              max_within_shape_ratio: float = 1.10,
                              max_cross_session_ratio: float = 1.10) -> dict:
    """Freeze a shape envelope and stability rules before new observations."""
    if not isinstance(cells, (list, tuple)) or not cells or len(cells) > 384:
        _error("A runtime policy requires a bounded nonempty list of phase/exact-batch cells")
    selected = [_policy_cell(row) for row in cells]
    keys = [_cell_key(row["phase"], row["batch_size"]) for row in selected]
    if len(set(keys)) != len(keys):
        _error("Runtime policy phase/exact-batch cells must be unique")
    minimum = _int(min_samples_per_shape_session, "min_samples_per_shape_session", 4)
    if minimum % 2:
        _error("Minimum per-shape samples must permit two complete equal halves")
    policy = {"schema_version": 1, "kind": _POLICY_KIND, "algorithm": RUNTIME_COST_ALGORITHM,
        "timing_boundary": RUNTIME_COST_BOUNDARY, "cells": sorted(selected, key=lambda row: _cell_key(row["phase"], row["batch_size"])),
        "min_samples_per_shape_session": minimum, "minimum_training_sessions": 2,
        "minimum_distinct_nominal_shapes": 2,
        "max_within_shape_ratio": _number(max_within_shape_ratio, "max_within_shape_ratio", minimum=1),
        "max_cross_session_ratio": _number(max_cross_session_ratio, "max_cross_session_ratio", minimum=1),
        "within_shape_method": "source-ordered equal halves; require even untrimmed sample count",
        "fit_statistic": "equal weight per acquisition-session/exact-shape median",
        "scope": "declared bounded envelope only; no batch interpolation or universal measured-accuracy claim"}
    policy["policy_sha256"] = _sha(policy)
    return policy


def _policy(value: Any) -> dict:
    if not isinstance(value, dict):
        _error("A separately frozen runtime policy is required")
    if _sha({key: item for key, item in value.items() if key != "policy_sha256"}) != value.get("policy_sha256"):
        _error("Runtime policy checksum changed")
    cells = value.get("cells")
    if not isinstance(cells, list):
        _error("Runtime policy cells must be an explicit array")
    expected = freeze_runtime_cost_policy(cells,
        min_samples_per_shape_session=_int(value.get("min_samples_per_shape_session"), "min_samples_per_shape_session", 4),
        max_within_shape_ratio=_number(value.get("max_within_shape_ratio"), "max_within_shape_ratio", minimum=1),
        max_cross_session_ratio=_number(value.get("max_cross_session_ratio"), "max_cross_session_ratio", minimum=1))
    if _canonical(expected) != _canonical(value):
        _error("Runtime policy schema, types, rules or checksum changed")
    return expected


def _shape(phase: Any, queries: Any, contexts: Any, stage: Any = 0, roles: Any = None) -> tuple[str, str]:
    from .profiles import shape_key

    if not isinstance(phase, str) or phase not in _PHASES or type(stage) is not int or stage != 0:
        _error("Runtime costs support prefill/decode/mixed on EngineCore stage0")
    if (not isinstance(queries, (list, tuple)) or not queries or not isinstance(contexts, (list, tuple))
            or len(queries) != len(contexts)):
        _error("Runtime query/context arrays must be nonempty and aligned")
    for value in queries:
        _int(value, "query_tokens[]", 1)
    for value in contexts:
        _int(value, "context_lengths[]")
    batch_members(phase, queries, roles, error_code="INVALID_RUNTIME_COST")
    return _cell_key(phase, len(queries)), shape_key(phase, queries, contexts, stage, request_phases=roles)


def _in_bounds(cell: dict, queries: Sequence[int], contexts: Sequence[int]) -> bool:
    features = {"query_tokens": queries, "context_tokens": contexts,
                "total_query_tokens": [sum(queries)], "total_context_tokens": [sum(contexts)]}
    return (sum(queries) + sum(contexts) <= cell["max_total_tokens"]
            and all(query + context <= cell["max_sequence_tokens"] for query, context in zip(queries, contexts))
            and all(cell["min_" + name] <= value <= cell["max_" + name]
                    for name, values in features.items() for value in values))


def _within_scheduler(strategy: dict, phase: str, queries: Sequence[int]) -> bool:
    active = strategy["max_active_requests"]
    return (len(queries) <= strategy["max_batch_size"] and sum(queries) <= strategy["max_batch_tokens"]
            and (active is None or len(queries) <= active)
            and (phase != "mixed" or strategy["mixed_batching"]))


def _source(value: Any, allow_test_fixture: bool) -> dict:
    if type(allow_test_fixture) is not bool:
        _error("allow_test_fixture must be boolean")
    item = _object(value, {"kind", "reference"}, set(), "runtime source")
    _text(item["reference"], "source.reference")
    if item["kind"] not in ("measured", "test_fixture"):
        _error("Runtime training requires measured or explicit test-fixture observations")
    if item["kind"] == "test_fixture" and not allow_test_fixture:
        _error("Test fixtures require explicit allow_test_fixture", "TEST_FIXTURE_NOT_ALLOWED")
    return deepcopy(item)


def _dataset(value: Any, policy: dict, allow_test_fixture: bool) -> tuple[dict, Experiment]:
    from .profiles import profile_identity, validate_identity

    _canonical(value)
    item = _object(value, {"schema_version", "kind", "dataset_id", "domain", "source", "records"}, {"notes"}, "runtime dataset")
    if type(item["schema_version"]) is not int or item["schema_version"] != 1 or item["kind"] != _DATASET_KIND:
        _error("Expected runtime_cost_dataset_v1")
    _text(item["dataset_id"], "dataset_id")
    domain, nominal = _domain(item["domain"])
    source = _source(item["source"], allow_test_fixture)
    if source["kind"] == "measured":
        validate_identity(profile_identity(nominal), measured=True)
    records = item["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= _MAX_RECORDS:
        _error("Runtime training needs a bounded nonempty raw record list")
    cells = {_cell_key(row["phase"], row["batch_size"]): row for row in policy["cells"]}
    names: set[str] = set()
    previous: dict[str, int] = {}
    normalized = []
    required = {"record_id", "session_id", "group_id", "source_order", "phase", "stage_index",
                "query_tokens", "context_lengths", "duration_ms"}
    for raw in records:
        row = _object(raw, required, {"request_phases", "workspace_peak_bytes"}, "runtime record")
        for name in ("record_id", "session_id", "group_id"):
            _text(row[name], name)
        if row["record_id"] in names:
            _error("Runtime training record IDs must be unique")
        names.add(row["record_id"])
        order = _int(row["source_order"], "source_order")
        if order <= previous.get(row["session_id"], -1):
            _error("Runtime records must retain each session's strict source chronology, not ID ordering")
        previous[row["session_id"]] = order
        key, _ = _shape(row["phase"], row["query_tokens"], row["context_lengths"], row["stage_index"], row.get("request_phases"))
        if row["phase"] != "mixed" and "request_phases" in row:
            _error("Only mixed runtime records carry request_phases")
        if key not in cells or not _in_bounds(cells[key], row["query_tokens"], row["context_lengths"]):
            _error("Training shape is outside the preregistered phase/batch/budget envelope")
        if not _within_scheduler(domain["strategy"], row["phase"], row["query_tokens"]):
            _error("Training shape conflicts with its actual scheduler batch/token/active/mixed policy")
        duration = _number(row["duration_ms"], "duration_ms", positive=True)
        workspace = row.get("workspace_peak_bytes")
        if workspace is not None:
            _int(workspace, "workspace_peak_bytes")
        normalized.append({**deepcopy(row), "duration_ms": duration, "workspace_peak_bytes": workspace})
    result = {**deepcopy(item), "domain": domain, "source": source, "records": normalized}
    if "notes" in result and not isinstance(result["notes"], dict):
        _error("Runtime dataset notes must be finite JSON object data")
    return _clone(result), nominal


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else float(statistics.mean(ordered[middle - 1:middle + 1]))


def _ratio(values: Sequence[float]) -> float | None:
    value = max(values) / min(values)
    return value if math.isfinite(value) else None


def _training(data: dict, policy: dict, nominal: Experiment) -> tuple[list[dict], dict, list[dict], dict[str, float]]:
    from .performance import AnalyticalModel

    model = AnalyticalModel(nominal)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    shapes: dict[str, float] = {}
    for row in data["records"]:
        _, shape = _shape(row["phase"], row["query_tokens"], row["context_lengths"], 0, row.get("request_phases"))
        grouped[row["session_id"], shape].append(row)
        if shape not in shapes:
            shapes[shape] = model.estimate_step(row["phase"], row["query_tokens"], row["context_lengths"],
                request_phases=row.get("request_phases")).duration_ms
    cells, per_session, reasons = [], [], []
    by_shape: dict[str, list[dict]] = defaultdict(list)
    for (session, shape), rows in sorted(grouped.items()):
        values = [row["duration_ms"] for row in rows]
        enough = len(values) >= policy["min_samples_per_shape_session"] and len(values) % 2 == 0
        middle = len(values) // 2
        halves = [_median(values[:middle]), _median(values[middle:])] if middle else []
        ratio = _ratio(halves) if halves else None
        item = {"session_id": session, "shape_key": shape, "phase": rows[0]["phase"],
            "batch_size": len(rows[0]["query_tokens"]), "record_ids": [row["record_id"] for row in rows],
            "sample_count": len(rows), "source_order": [row["source_order"] for row in rows],
            "source_ordered_half_medians_ms": halves, "within_shape_ratio": ratio,
            "median_ms": _median(values), "passed": enough and ratio is not None and ratio <= policy["max_within_shape_ratio"]}
        per_session.append(item)
        by_shape[shape].append(item)
        if not item["passed"]:
            reasons.append({"code": "UNSTABLE_OR_INSUFFICIENT_SOURCE_SERIES", "session_id": session, "shape_key": shape})
        cells.append({"session_id": session, "shape_key": shape, "phase": item["phase"], "batch_size": item["batch_size"],
            "record_ids": item["record_ids"], "sample_count": len(rows), "nominal_ms": shapes[shape], "observed_ms": item["median_ms"]})
    cross = []
    for shape, members in sorted(by_shape.items()):
        ratio = _ratio([row["median_ms"] for row in members])
        item = {"shape_key": shape, "session_ids": [row["session_id"] for row in members],
            "session_medians_ms": [row["median_ms"] for row in members], "cross_session_ratio": ratio,
            "passed": len(members) >= policy["minimum_training_sessions"] and ratio is not None and ratio <= policy["max_cross_session_ratio"]}
        cross.append(item)
        if not item["passed"]:
            reasons.append({"code": "UNSTABLE_OR_INSUFFICIENT_TRAINING_SESSIONS", "shape_key": shape})
    for rule in policy["cells"]:
        selected = [row for row in cells if (row["phase"], row["batch_size"]) == (rule["phase"], rule["batch_size"])]
        xs = [row["nominal_ms"] for row in selected]
        if not xs or max(xs) - min(xs) <= 1e-12 * max(xs):
            reasons.append({"code": "UNIDENTIFIABLE_NOMINAL_SHAPES", "phase": rule["phase"], "batch_size": rule["batch_size"]})
    return cells, {"per_session_shapes": per_session, "cross_session_shapes": cross,
                   "verdict": "FAIL" if reasons else "PASS"}, reasons, shapes


def _nnls(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, dict]:
    """Scale coordinates before solving the two-variable constrained objective."""
    sx, sy = max(xs), max(ys)
    u, v = [x / sx for x in xs], [y / sy for y in ys]
    mx, my = statistics.mean(u), statistics.mean(v)
    variance = sum((x - mx) ** 2 for x in u)
    if variance <= 0:
        _error("Distinct nominal training features are required")
    a = sum((x - mx) * (y - my) for x, y in zip(u, v)) / variance
    c = my - a * mx
    candidates = [(0.0, 0.0), (0.0, max(0.0, my)),
                  (max(0.0, sum(x * y for x, y in zip(u, v)) / sum(x * x for x in u)), 0.0)]
    if a >= 0 and c >= 0:
        candidates.append((a, c))
    def score(pair):
        return sum((pair[0] * x + pair[1] - y) ** 2 for x, y in zip(u, v))
    chosen = min(candidates, key=score)
    scale, intercept = chosen[0] * sy / sx, chosen[1] * sy
    _number(scale, "fitted scale")
    _number(intercept, "fitted intercept")
    return scale, intercept, {"solver": "scaled_two_variable_nonnegative_least_squares",
        "unconstrained_scale": a * sy / sx, "unconstrained_intercept_ms": c * sy,
        "normalized_sum_squared_error": score(chosen), "objective_y_scale_ms": sy,
        "constraint_active": chosen != (a, c), "scale": scale, "intercept_ms": intercept}


def _residuals(data: dict, parameters: list[dict], shapes: dict[str, float]) -> list[dict]:
    indexed = {_cell_key(row["phase"], row["batch_size"]): row for row in parameters}
    result = []
    for row in data["records"]:
        cell, shape = _shape(row["phase"], row["query_tokens"], row["context_lengths"], 0, row.get("request_phases"))
        fit = indexed[cell]
        predicted = fit["scale"] * shapes[shape] + fit["intercept_ms"]
        _number(predicted, "training prediction", positive=True)
        relative = abs(predicted - row["duration_ms"]) / row["duration_ms"]
        result.append({"record_id": row["record_id"], "session_id": row["session_id"], "shape_key": shape,
            "observed_ms": row["duration_ms"], "predicted_ms": predicted,
            "signed_error_ms": predicted - row["duration_ms"], "relative_error": relative if math.isfinite(relative) else None})
    return result


def fit_runtime_costs(dataset: dict, policy: dict, *, calibration_id: str,
                      allow_test_fixture: bool = False) -> dict:
    """Fit from training only; rejected process stability never emits parameters."""
    frozen = _policy(policy)
    data, nominal = _dataset(dataset, frozen, allow_test_fixture)
    _text(calibration_id, "calibration_id")
    cells, stationarity, reasons, shapes = _training(data, frozen, nominal)
    result = {"schema_version": 1, "algorithm": RUNTIME_COST_ALGORITHM, "calibration_id": calibration_id,
        "domain": data["domain"], "policy": frozen, "source": data["source"], "raw_dataset": data,
        "training_dataset_sha256": _sha(data), "training_record_ids": [row["record_id"] for row in data["records"]],
        "diagnostics": {"stationarity": stationarity, "fitting_cells": cells},
        "hardware_validated": False, "limitations": list(_LIMITATIONS)}
    if reasons:
        result.update(kind=RUNTIME_COST_REJECTION_KIND, status="REJECTED", rejections=reasons, fit_attempted=False)
        result["rejection_sha256"] = _sha(result)
        return result
    parameters = []
    for rule in frozen["cells"]:
        selected = [row for row in cells if (row["phase"], row["batch_size"]) == (rule["phase"], rule["batch_size"])]
        xs = [row["nominal_ms"] for row in selected]
        scale, intercept, diagnostics = _nnls(xs, [row["observed_ms"] for row in selected])
        parameters.append({"phase": rule["phase"], "batch_size": rule["batch_size"], "scale": scale,
            "intercept_ms": intercept, "training_nominal_min_ms": min(xs), "training_nominal_max_ms": max(xs),
            "fit_diagnostics": diagnostics, "equal_weight_cell_count": len(selected)})
    result.update(kind=RUNTIME_COST_KIND, parameters=parameters)
    result["diagnostics"]["training_residuals"] = _residuals(data, parameters, shapes)
    result["model_sha256"] = _sha(result)
    return result


def _derived_close(actual: Any, expected: Any) -> bool:
    """Only recomputed floating coordinates use numerical replay tolerance."""
    if expected is None:
        return actual is None
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or isinstance(expected, bool) or not isinstance(expected, (int, float))):
        return False
    try:
        return (math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=_REPLAY_REL_TOL, abs_tol=_REPLAY_ABS_TOL))
    except OverflowError:
        return False


def _derived_records_equal(actual: Any, expected: list[dict], floating_fields: set[str]) -> bool:
    """Keep record order, IDs, geometry and other metadata canonically exact."""
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for stored, replayed in zip(actual, expected):
        if not isinstance(stored, dict) or set(stored) != set(replayed):
            return False
        if (_canonical({key: value for key, value in stored.items() if key not in floating_fields})
                != _canonical({key: value for key, value in replayed.items() if key not in floating_fields})):
            return False
        if any(not _derived_close(stored[key], replayed[key]) for key in floating_fields):
            return False
    return True


def _fit_diagnostics_equal(stored: Any, replayed: dict, scale: float, intercept: float) -> bool:
    """Check frozen solver evidence without requiring a version-specific bool."""
    if not isinstance(stored, dict) or set(stored) != set(replayed):
        return False
    if (stored["solver"] != replayed["solver"] or type(stored["constraint_active"]) is not bool
            or stored["scale"] != scale or stored["intercept_ms"] != intercept):
        return False
    numeric = set(replayed) - {"solver", "constraint_active"}
    if any(not _derived_close(stored[key], replayed[key]) for key in numeric):
        return False
    if stored["normalized_sum_squared_error"] < 0 or stored["objective_y_scale_ms"] <= 0:
        return False
    unconstrained = stored["unconstrained_scale"], stored["unconstrained_intercept_ms"]
    differs = (scale, intercept) != unconstrained
    if stored["constraint_active"] == differs:
        return True
    # A near-zero constraint can change branches after a floating reduction.
    # Nonboundary labels must still agree with the artifact's own coefficients.
    return ((_derived_close(scale, 0.0) or _derived_close(intercept, 0.0))
            and _derived_close(scale, unconstrained[0]) and _derived_close(intercept, unconstrained[1]))


def validate_runtime_cost_model(artifact: dict, *, experiment: Experiment | None = None,
                                allow_test_fixture: bool = False) -> dict:
    """Validate immutable raw lineage and executable-domain semantics once."""
    _canonical(artifact)
    from .short_prefill import KIND, validate_short_prefill
    if isinstance(artifact, dict) and artifact.get("kind") == KIND:
        return validate_short_prefill(artifact, experiment, allow_test_fixture)
    required = {"schema_version", "kind", "algorithm", "calibration_id", "domain", "policy", "source",
                "raw_dataset", "training_dataset_sha256", "training_record_ids", "diagnostics",
                "parameters", "hardware_validated", "limitations", "model_sha256"}
    item = _object(artifact, required, set(), "runtime cost model")
    if (type(item["schema_version"]) is not int or item["schema_version"] != 1 or item["kind"] != RUNTIME_COST_KIND
            or item["algorithm"] != RUNTIME_COST_ALGORITHM or item["hardware_validated"] is not False
            or item["limitations"] != _LIMITATIONS):
        _error("Expected an executable bounded runtime-cost artifact")
    _text(item["calibration_id"], "calibration_id")
    if _sha({key: value for key, value in item.items() if key != "model_sha256"}) != item["model_sha256"]:
        _error("Runtime model checksum changed")
    policy = _policy(item["policy"])
    data, nominal = _dataset(item["raw_dataset"], policy, allow_test_fixture)
    if (_canonical(data) != _canonical(item["raw_dataset"]) or _canonical(item["source"]) != _canonical(data["source"])
            or _canonical(item["domain"]) != _canonical(data["domain"]) or item["training_dataset_sha256"] != _sha(data)
            or item["training_record_ids"] != [row["record_id"] for row in data["records"]]):
        _error("Runtime model raw data, domain, source or training membership changed")
    cells, stationarity, reasons, shapes = _training(data, policy, nominal)
    diagnostics = _object(item["diagnostics"], {"stationarity", "fitting_cells", "training_residuals"}, set(), "runtime diagnostics")
    if (reasons or _canonical(diagnostics["stationarity"]) != _canonical(stationarity)
            or not _derived_records_equal(diagnostics["fitting_cells"], cells, {"nominal_ms", "observed_ms"})):
        _error("Runtime stationarity or fitting observations differ from their untrimmed raw data")
    parameters = item["parameters"]
    if not isinstance(parameters, list) or len(parameters) != len(policy["cells"]):
        _error("Runtime parameters must cover every declared phase/exact batch once")
    indexed = {}
    for parameter in parameters:
        row = _object(parameter, {"phase", "batch_size", "scale", "intercept_ms", "training_nominal_min_ms",
            "training_nominal_max_ms", "fit_diagnostics", "equal_weight_cell_count"}, set(), "runtime parameter")
        if not isinstance(row["phase"], str) or row["phase"] not in _PHASES:
            _error("Runtime parameter phase is invalid")
        key = _cell_key(row["phase"], _int(row["batch_size"], "batch_size", 1))
        if key in indexed:
            _error("Duplicate runtime phase/exact-batch parameters")
        indexed[key] = row
        scale, intercept = _number(row["scale"], "scale"), _number(row["intercept_ms"], "intercept_ms")
        selected = [cell for cell in cells if _cell_key(cell["phase"], cell["batch_size"]) == key]
        if (not selected or _int(row["equal_weight_cell_count"], "equal_weight_cell_count", 1) != len(selected)
                or not _derived_close(_number(row["training_nominal_min_ms"], "training_nominal_min_ms", positive=True), min(cell["nominal_ms"] for cell in selected))
                or not _derived_close(_number(row["training_nominal_max_ms"], "training_nominal_max_ms", positive=True), max(cell["nominal_ms"] for cell in selected))):
            _error("Runtime parameter shape coverage differs from training")
        # Recompute the deterministic training-only objective as a consistency
        # check. Never replace the frozen coefficients or inspect validation data.
        expected_scale, expected_intercept, expected_diagnostics = _nnls(
            [cell["nominal_ms"] for cell in selected], [cell["observed_ms"] for cell in selected])
        if (not _derived_close(scale, expected_scale) or not _derived_close(intercept, expected_intercept)
                or not _fit_diagnostics_equal(row["fit_diagnostics"], expected_diagnostics, scale, intercept)):
            _error("Runtime parameter coefficients or solver diagnostics differ from their training-only fit")
    if set(indexed) != {_cell_key(row["phase"], row["batch_size"]) for row in policy["cells"]}:
        _error("Runtime parameters cannot substitute an untrained phase or batch")
    if not _derived_records_equal(diagnostics["training_residuals"], _residuals(data, parameters, shapes),
                                  {"predicted_ms", "signed_error_ms", "relative_error"}):
        _error("Runtime residuals differ from raw observations and frozen coefficients")
    if experiment is not None and _canonical(runtime_cost_domain(experiment)) != _canonical(item["domain"]):
        _error("Runtime model and requested GPU/model/Runtime/core/memory domain differ", "CALIBRATION_MISMATCH")
    return _clone(item)


def load_runtime_cost(path: str | Path, *, experiment: Experiment | None = None,
                       allow_test_fixture: bool = False) -> dict:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise SimulationError("INVALID_RUNTIME_COST", f"Cannot read runtime-cost artifact: {exc}") from exc
    return validate_runtime_cost_model(_parse_bytes(raw), experiment=experiment, allow_test_fixture=allow_test_fixture)


def _parse_bytes(raw: bytes) -> dict:
    if len(raw) > _MAX_BYTES:
        _error("Runtime model file exceeds its bounded size limit")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _error("Duplicate JSON keys in runtime-cost artifact")
            result[key] = value
        return result
    try:
        payload = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SimulationError("INVALID_RUNTIME_COST", "Cannot parse runtime-cost artifact JSON") from exc
    return payload


class RuntimeCostModel:
    """Read/validate once; every query checks the complete cheap runtime domain."""

    def __init__(self, experiment: Experiment):
        from .config import validate_experiment
        from .performance import AnalyticalModel

        validate_experiment(experiment)
        if not experiment.calibration.profile_path:
            _error("Runtime cost models require calibration.profile_path")
        path = Path(experiment.calibration.profile_path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise SimulationError("INVALID_RUNTIME_COST", f"Cannot read runtime-cost artifact: {exc}") from exc
        self._model = validate_runtime_cost_model(_parse_bytes(raw), experiment=experiment,
            allow_test_fixture=experiment.calibration.allow_test_fixture)
        self.experiment = experiment
        self._validated_experiment = experiment
        self._domain = self._model["domain"]
        self._nominal = AnalyticalModel(_domain(self._domain)[1])
        self._parameters = {_cell_key(row["phase"], row["batch_size"]): row for row in self._model["parameters"]}
        self._rules = {_cell_key(row["phase"], row["batch_size"]): row for row in self._model["policy"]["cells"]}
        self._observed_shapes: dict[str, list[dict]] = defaultdict(list)
        for row in self._model["raw_dataset"]["records"]:
            _, shape = _shape(row["phase"], row["query_tokens"], row["context_lengths"], 0, row.get("request_phases"))
            self._observed_shapes[shape].append(row)
        self._short_prefill = self._model.get("short_prefill", {}).get("table", {})
        requested = experiment.calibration.profile_version
        if requested != "uncalibrated-v1" and requested != self._model["calibration_id"]:
            _error("Configured runtime profile_version differs from its calibration ID", "CALIBRATION_MISMATCH")
        self.provenance = {"kind": "runtime_cost_v1", "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(), "profile_id": self._model["calibration_id"],
            "profile_version": self._model["calibration_id"], "source_kind": self._model["source"]["kind"],
            "source_reference": self._model["source"]["reference"], "timing_boundary": RUNTIME_COST_BOUNDARY,
            "domain_sha256": self._domain["domain_sha256"], "policy_sha256": self._model["policy"]["policy_sha256"],
            "model_sha256": self._model["model_sha256"], "hardware_validated": False,
            "domain_policy": "bounded_phase_exact_batch; no batch interpolation", "extrapolation_used": False}

    @property
    def profile(self) -> dict:
        return deepcopy(self._model)

    def _guard(self) -> None:
        _source(self._model["source"], self.experiment.calibration.allow_test_fixture)
        # Experiment and all nested configuration records are frozen
        # dataclasses. Reassignment is checked; repeated use needs no JSON work.
        if self.experiment is self._validated_experiment:
            return
        if runtime_cost_domain(self.experiment) != self._domain:
            _error("Runtime query domain changed after the immutable snapshot loaded", "CALIBRATION_MISMATCH")
        self._validated_experiment = self.experiment

    def _query(self, phase, queries, contexts, stage, roles) -> tuple[dict, str]:
        self._guard()
        cell, shape = _shape(phase, queries, contexts, stage, roles)
        if (cell not in self._rules or not _in_bounds(self._rules[cell], queries, contexts)
                or not _within_scheduler(self._domain["strategy"], phase, queries)):
            _error("Runtime phase/exact batch or q/context/total budget is outside its frozen envelope", "OUT_OF_CALIBRATION_DOMAIN")
        return self._parameters[cell], shape

    def estimate_step(self, phase: str, query_tokens: Sequence[int], context_lengths: Sequence[int],
                      stage_index: int = 0, *, request_phases: Sequence[str] | None = None) -> StepEstimate:
        parameter, shape = self._query(phase, query_tokens, context_lengths, stage_index, request_phases)
        entry = self._short_prefill.get(shape)
        if entry is not None:
            duration = entry["duration_ms"]
            return StepEstimate(duration, {"runtime_cost": {"duration_ms": duration, "phase": phase,
                "method": "ordered_shape_median", "batch_size": len(query_tokens),
                "timing_boundary": RUNTIME_COST_BOUNDARY, "shape_observed_in_training": True,
                "domain_sha256": self._domain["domain_sha256"], "model_sha256": self._model["model_sha256"],
                "hardware_validated": False}}, "runtime_cost_calibrated", (self._model["limitations"][-1],))
        nominal = self._nominal.estimate_step(phase, query_tokens, context_lengths, stage_index, request_phases=request_phases)
        duration = parameter["scale"] * nominal.duration_ms + parameter["intercept_ms"]
        _number(duration, "Runtime cost prediction", positive=True)
        return StepEstimate(duration, {"runtime_cost": {"duration_ms": duration, "nominal_duration_ms": nominal.duration_ms,
            "scale": parameter["scale"], "intercept_ms": parameter["intercept_ms"], "phase": phase,
            "batch_size": len(query_tokens), "timing_boundary": RUNTIME_COST_BOUNDARY,
            "domain_sha256": self._domain["domain_sha256"], "model_sha256": self._model["model_sha256"],
            "shape_observed_in_training": shape in self._observed_shapes,
            "within_training_nominal_span": parameter["training_nominal_min_ms"] <= nominal.duration_ms <= parameter["training_nominal_max_ms"],
            "hardware_validated": False}}, "runtime_cost_calibrated", tuple(_LIMITATIONS))

    def workspace_bytes(self, query_tokens: Sequence[int], phase: str, stage_index: int = 0,
                        context_lengths: Sequence[int] | None = None, *,
                        request_phases: Sequence[str] | None = None) -> int | None:
        self._guard()
        if context_lengths is None:
            cell, _ = _shape(phase, query_tokens, [0] * len(query_tokens), stage_index, request_phases)
            if phase == "mixed":
                _error("Mixed runtime Workspace requires actual per-member contexts")
            if phase == "decode":
                if cell not in self._rules:
                    _error("Runtime Workspace has no measured phase/exact batch", "OUT_OF_CALIBRATION_DOMAIN")
                return None
            context_lengths = [0] * len(query_tokens)
        _, shape = self._query(phase, query_tokens, context_lengths, stage_index, request_phases)
        rows = self._observed_shapes.get(shape)
        if not rows or any(row["workspace_peak_bytes"] is None for row in rows):
            return None
        return max(row["workspace_peak_bytes"] for row in rows)
