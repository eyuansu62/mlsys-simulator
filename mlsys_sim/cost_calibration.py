"""Explicit fitting and retrospective group holdout for analytical cost terms.

Profiler component observations and wall-step observations remain different
channels.  The fitted wall intercept is empirical unmodeled work, not a direct
measurement of CPU, launch or framework overhead.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import math
import statistics
from typing import Any, NoReturn

from .schema import AnalyticalCostSpec, SimulationError


DATASET_KIND = "analytical_cost_dataset_v1"
SPLIT_KIND = "analytical_cost_split_v1"
FIT_KIND = "analytical_cost_fit_v1"
ALGORITHM = "nonnegative_component_scale_attention_affine_wall_median_v1"
SOURCE_KINDS = {"provided_trace", "measured", "test_fixture"}


def _error(message: str) -> NoReturn:
    raise SimulationError("INVALID_COST_CALIBRATION", message)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise SimulationError("INVALID_COST_CALIBRATION", "Expected finite JSON cost data") from exc


def cost_data_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(f"{name} must be nonempty text")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _error(f"{name} must be a lowercase SHA256")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value < 0 or (positive and value == 0):
        _error(f"{name} must be a {'positive' if positive else 'nonnegative'} finite number")
    return float(value)


def _integers(value: Any, name: str, *, positive: bool) -> list[int]:
    if not isinstance(value, list) or not value:
        _error(f"{name} must be a nonempty integer list")
    if any(type(item) is not int or item < (1 if positive else 0) for item in value):
        _error(f"{name} has an invalid token count")
    return value


def _domain_key(row: dict) -> str:
    return cost_data_sha256({"phase": row["phase"], "domain": row["domain"]})


def _shape_key(row: dict) -> tuple[str, str]:
    return _domain_key(row), row["shape"]


def validate_cost_dataset(value: dict) -> dict:
    """Validate measurements and retain original metadata without interpreting it."""
    _canonical(value)
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1 or value.get("kind") != DATASET_KIND):
        _error("Expected analytical_cost_dataset_v1 schema_version 1")
    source = value.get("source")
    if not isinstance(source, dict) or source.get("kind") not in SOURCE_KINDS:
        _error("dataset.source.kind must identify provided_trace, measured or test_fixture")
    rows = value.get("records")
    if not isinstance(rows, list) or not rows:
        _error("dataset.records must not be empty")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            _error("Each cost record must be an object")
        name = _text(row.get("record_id"), "record_id")
        if name in seen:
            _error("Duplicate cost record_id")
        seen.add(name)
        for field in ("session_id", "group_id", "model_id", "shape"):
            _text(row.get(field), field)
        if row.get("phase") not in {"prefill", "decode"}:
            _error("Only separately observed prefill/decode phases can be fitted")
        domain = row.get("domain")
        if not isinstance(domain, dict):
            _error("Each record needs its explicit domain")
        _digest(domain.get("identity_sha256"), "domain.identity_sha256")
        for field in ("execution_mode", "attention_backend", "timing_boundary"):
            if _text(domain.get(field), "domain." + field) == "unspecified":
                _error("Fitted cost domain cannot be unspecified")
        queries = _integers(row.get("query_tokens"), "query_tokens", positive=True)
        contexts = _integers(row.get("context_lengths"), "context_lengths", positive=False)
        if len(queries) != len(contexts):
            _error("query_tokens and context_lengths must have equal length")
        if type(row.get("layers")) is not int or row["layers"] <= 0:
            _error("layers must be a positive integer")
        _number(row.get("nominal_nonattention_ms"), "nominal_nonattention_ms", positive=True)
        _number(row.get("nominal_attention_ms"), "nominal_attention_ms")
        if row.get("kind") == "gpu_components":
            _number(row.get("observed_nonattention_ms"), "observed_nonattention_ms")
            _number(row.get("observed_attention_ms"), "observed_attention_ms")
        elif row.get("kind") == "step_wall":
            _number(row.get("observed_ms"), "observed_ms", positive=True)
        else:
            _error("record.kind must be gpu_components or step_wall")
    return deepcopy(value)


def _membership(value: Any, available: set[str], name: str, *, nonempty: bool) -> list[str]:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        _error(f"{name} must explicitly list record IDs")
    if any(not isinstance(item, str) for item in value) or len(set(value)) != len(value):
        _error(f"{name} contains duplicate or invalid IDs")
    if not set(value) <= available:
        _error(f"{name} references unknown observations")
    return list(value)


def freeze_cost_split(dataset: dict, *, training_record_ids: list[str] | tuple[str, ...],
                      holdout_record_ids: list[str] | tuple[str, ...],
                      repeat_record_ids: list[str] | tuple[str, ...] = (),
                      selection_note: str = "Retrospective shape/session-disjoint selection; not a new blind acquisition",
                      max_relative_error: float = 0.3) -> dict:
    """Freeze explicit membership before fitting; never select by observed error."""
    data = validate_cost_dataset(dataset)
    threshold = _number(max_relative_error, "max_relative_error")
    by_id = {row["record_id"]: row for row in data["records"]}
    training = _membership(training_record_ids, set(by_id), "training_record_ids", nonempty=True)
    holdout = _membership(holdout_record_ids, set(by_id), "holdout_record_ids", nonempty=True)
    repeats = _membership(repeat_record_ids, set(by_id), "repeat_record_ids", nonempty=False)
    if set(training) & (set(holdout) | set(repeats)) or set(holdout) & set(repeats):
        _error("Training, holdout and repeat records must be disjoint")
    fit_rows = [by_id[name] for name in training]
    test_rows = [by_id[name] for name in holdout + repeats]
    if any(row["kind"] != "step_wall" for row in test_rows):
        _error("Validation uses independent step-wall observations, not component fit targets")
    if {row["session_id"] for row in fit_rows} & {row["session_id"] for row in test_rows}:
        _error("Training and validation share session_id; independent acquisition is required")
    if {row["group_id"] for row in fit_rows} & {by_id[name]["group_id"] for name in holdout}:
        _error("Shape holdout overlaps a training group_id")
    if {_shape_key(row) for row in fit_rows} & {_shape_key(by_id[name]) for name in holdout}:
        _error("Shape holdout overlaps a fitted shape; use repeat_record_ids for retained-shape repeats")
    if not {_shape_key(by_id[name]) for name in repeats} <= {_shape_key(row) for row in fit_rows}:
        _error("Repeat validation must retain shapes already present in training")
    domains = {_domain_key(row) for row in fit_rows}
    if any(_domain_key(row) not in domains for row in test_rows):
        _error("Holdout contains an identity/backend/mode domain not present in training")
    plan = {"schema_version": 1, "kind": SPLIT_KIND,
            "dataset_sha256": cost_data_sha256(data),
            "training_record_ids": training, "holdout_record_ids": holdout,
            "repeat_record_ids": repeats,
            "unused_record_ids": sorted(set(by_id) - set(training) - set(holdout) - set(repeats)),
            "training_dataset_sha256": cost_data_sha256(fit_rows),
            "selection_kind": "retrospective_group_holdout",
            "acceptance": {"metric": "per_session_shape_median_relative_error", "max_relative_error": threshold,
                           "require_full_coverage": True, "require_all_cells": True,
                           "independent_unit": "acquisition_session; individual steps are nested observations"},
            "selection_note": _text(selection_note, "selection_note")}
    plan["split_sha256"] = cost_data_sha256(plan)
    return plan


def _check_split(dataset: dict, plan: dict) -> tuple[dict, dict[str, dict]]:
    data = validate_cost_dataset(dataset)
    required = {"schema_version", "kind", "dataset_sha256", "training_record_ids", "holdout_record_ids",
                "repeat_record_ids", "unused_record_ids", "training_dataset_sha256", "selection_kind",
                "acceptance", "selection_note", "split_sha256"}
    if (not isinstance(plan, dict) or set(plan) != required or plan.get("kind") != SPLIT_KIND
            or type(plan.get("schema_version")) is not int or plan.get("schema_version") != 1
            or not isinstance(plan.get("acceptance"), dict)):
        _error("Expected a frozen cost split")
    expected = freeze_cost_split(data, training_record_ids=plan["training_record_ids"],
        holdout_record_ids=plan["holdout_record_ids"], repeat_record_ids=plan["repeat_record_ids"],
        selection_note=plan["selection_note"],
        max_relative_error=plan["acceptance"].get("max_relative_error"))
    if expected != plan:
        _error("Frozen cost split, source dataset or membership was modified")
    return data, {row["record_id"]: row for row in data["records"]}


def _attention_fit(rows: list[dict]) -> tuple[float, float, dict]:
    xs = [float(row["nominal_attention_ms"]) for row in rows]
    ys = [float(row["observed_attention_ms"]) for row in rows]
    if len(xs) < 2 or max(xs) - min(xs) <= 1e-12 * max(1.0, max(xs)):
        _error("Attention slope/intercept are not identifiable from distinct training features")
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    intercept = mean_y - slope * mean_x
    candidates = [(0.0, 0.0), (0.0, max(0.0, mean_y))]
    norm = sum(x * x for x in xs)
    if norm > 0:
        candidates.append((max(0.0, sum(x * y for x, y in zip(xs, ys)) / norm), 0.0))
    if slope >= 0 and intercept >= 0:
        candidates.append((slope, intercept))
    def score(pair):
        return sum((pair[0] * x + pair[1] - y) ** 2 for x, y in zip(xs, ys))
    chosen = min(candidates, key=score)
    return chosen[0], chosen[1], {"unconstrained_slope": slope,
        "unconstrained_intercept_ms": intercept, "sum_squared_error_ms2": score(chosen),
        "constraint_active": chosen != (slope, intercept)}


def _predicted(row: dict, cost: dict) -> float:
    value = (cost["nonattention_scale"] * row["nominal_nonattention_ms"]
            + cost["attention_scale"] * row["nominal_attention_ms"]
            + row["layers"] * cost["attention_fixed_us_per_layer"] / 1000
            + cost["runtime_fixed_us"] / 1000)
    return _number(value, "predicted cost")


def _errors(rows: list[dict], prediction, threshold: float) -> dict:
    records = []
    for row in rows:
        actual, predicted = row["observed_ms"], prediction(row)
        records.append({"record_id": row["record_id"], "session_id": row["session_id"],
            "shape": row["shape"], "phase": row["phase"], "model_id": row["model_id"],
            "domain_sha256": _domain_key(row), "observed_ms": actual, "predicted_ms": predicted,
            "signed_error_ms": predicted - actual, "relative_error": abs(predicted - actual) / actual,
            "passed": abs(predicted - actual) / actual <= threshold})
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["domain_sha256"], row["shape"], row["session_id"])].append(row)
    cells = []
    for (domain, shape, session), members in sorted(grouped.items()):
        actual = statistics.median(row["observed_ms"] for row in members)
        predicted = statistics.median(row["predicted_ms"] for row in members)
        cells.append({"domain_sha256": domain, "shape": shape, "session_id": session,
            "model_id": members[0]["model_id"], "phase": members[0]["phase"],
            "sample_count": len(members), "observed_ms": actual, "predicted_ms": predicted,
            "relative_error": abs(predicted - actual) / actual,
            "passed": abs(predicted - actual) / actual <= threshold})
    return {"records": records, "cells": cells, "record_count": len(records), "cell_count": len(cells),
            "record_mape": statistics.mean(row["relative_error"] for row in records) if records else None,
            "cell_mape": statistics.mean(row["relative_error"] for row in cells) if cells else None,
            "records_passed": sum(row["passed"] for row in records),
            "cells_passed": sum(row["passed"] for row in cells)}


def fit_analytical_costs(dataset: dict, split: dict, *, calibration_id: str) -> dict:
    """Fit only frozen training rows; retain constraints and residual diagnostics."""
    data, by_id = _check_split(dataset, split)
    _text(calibration_id, "calibration_id")
    training = [by_id[name] for name in split["training_record_ids"]]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in training:
        groups[_domain_key(row)].append(row)
    costs, diagnostics = [], []
    for key, rows in sorted(groups.items()):
        component = [row for row in rows if row["kind"] == "gpu_components"]
        walls = [row for row in rows if row["kind"] == "step_wall"]
        if len({row["shape"] for row in component}) < 2 or not walls:
            _error("Every cost domain needs at least two component shapes and training step-wall records")
        layers = {row["layers"] for row in rows}
        if len(layers) != 1:
            _error("A cost domain must identify one model layer count")
        denominator = sum(row["nominal_nonattention_ms"] ** 2 for row in component)
        if denominator <= 0:
            _error("Nonattention scale has no positive identifiable feature")
        scale = max(0.0, sum(row["nominal_nonattention_ms"] * row["observed_nonattention_ms"]
                             for row in component) / denominator)
        attention, fixed_ms, attention_diagnostic = _attention_fit(component)
        domain = rows[0]["domain"]
        cost = {"phase": rows[0]["phase"], "execution_mode": domain["execution_mode"],
            "attention_backend": domain["attention_backend"], "identity_sha256": domain["identity_sha256"],
            "nonattention_scale": scale, "attention_scale": attention,
            "attention_fixed_us_per_layer": fixed_ms * 1000 / next(iter(layers)), "runtime_fixed_us": 0.0,
            "min_batch_size": min(len(row["query_tokens"]) for row in rows),
            "max_batch_size": max(len(row["query_tokens"]) for row in rows),
            "min_query_tokens": min(min(row["query_tokens"]) for row in rows),
            "max_query_tokens": max(max(row["query_tokens"]) for row in rows),
            "min_context_tokens": min(min(row["context_lengths"]) for row in rows),
            "max_context_tokens": max(max(row["context_lengths"]) for row in rows),
            "calibration_id": calibration_id, "training_dataset_sha256": cost_data_sha256(rows),
            "training_record_ids": [row["record_id"] for row in rows], "source_kind": data["source"]["kind"]}
        residuals = [row["observed_ms"] - _predicted(row, cost) for row in walls]
        median = statistics.median(residuals)
        cost["runtime_fixed_us"] = max(0.0, median) * 1000
        _canonical(cost)
        costs.append(cost)
        diagnostics.append({"domain_sha256": key, "domain": deepcopy(domain),
            "model_id": rows[0]["model_id"], "phase": rows[0]["phase"],
            "training_record_ids": cost["training_record_ids"],
            "attention": attention_diagnostic, "unclamped_runtime_intercept_ms": median,
            "runtime_intercept_clamped": median < 0,
            "training_wall_residuals_ms": residuals,
            "component_nonattention_residuals_ms": [scale * row["nominal_nonattention_ms"] - row["observed_nonattention_ms"] for row in component],
            "training_wall_errors": _errors(walls, lambda row: _predicted(row, cost), split["acceptance"]["max_relative_error"])})
    boundaries = {row["domain"]["timing_boundary"] for row in training}
    if len(boundaries) != 1:
        _error("One fit artifact must use one explicitly defined target timing boundary")
    artifact = {"schema_version": 1, "kind": FIT_KIND, "algorithm": ALGORITHM,
        "calibration_id": calibration_id, "dataset_sha256": cost_data_sha256(data),
        "training_dataset_sha256": split["training_dataset_sha256"], "split": deepcopy(split),
        "cost_specs": costs, "diagnostics": diagnostics,
        "calibration_patch": {"backend": "analytical", "analytical_costs": costs,
                              "analytical_cost_timing_boundary": next(iter(boundaries))},
        "hardware_validated": False,
        "assumptions": ["Retrospective group holdout; aggregate observations were previously inspected.",
            "Component profiling and wall timing are separate acquisition channels.",
            "Runtime intercept is empirical unmodeled work, not directly measured CPU or launch time.",
            "Configured Graph/eager and backend identity do not prove an unobserved per-step dispatch mode.",
            "Approximate/reconstructed context evidence remains in the source dataset; no hidden alignment claim."]}
    artifact["fit_sha256"] = cost_data_sha256(artifact)
    return artifact


def _in_domain(row: dict, cost: dict) -> bool:
    return (cost["min_batch_size"] <= len(row["query_tokens"]) <= cost["max_batch_size"]
            and all(cost["min_query_tokens"] <= value <= cost["max_query_tokens"] for value in row["query_tokens"])
            and all(cost["min_context_tokens"] <= value <= cost["max_context_tokens"] for value in row["context_lengths"]))


def validate_analytical_costs(dataset: dict, artifact: dict, *, max_relative_error: float | None = None) -> dict:
    """Evaluate every frozen held-out row without changing any fitted parameter."""
    if not isinstance(artifact, dict) or artifact.get("kind") != FIT_KIND or artifact.get("schema_version") != 1:
        _error("Expected a frozen analytical cost fit")
    frozen = {key: value for key, value in artifact.items() if key != "fit_sha256"}
    if cost_data_sha256(frozen) != artifact.get("fit_sha256"):
        _error("Fitted artifact was changed after freezing")
    data, by_id = _check_split(dataset, artifact["split"])
    threshold = artifact["split"]["acceptance"]["max_relative_error"]
    if max_relative_error is not None and _number(max_relative_error, "max_relative_error") != threshold:
        _error("Validation threshold differs from the policy frozen before fitting")
    if artifact["dataset_sha256"] != cost_data_sha256(data):
        _error("Validation dataset differs from the frozen source")
    # Recreate no coefficients here: inspect immutable membership, domain and
    # support bounds. A checksum alone cannot establish their consistency.
    if (artifact.get("training_dataset_sha256") != artifact["split"]["training_dataset_sha256"]
            or len(artifact["diagnostics"]) != len(artifact["cost_specs"])
            or artifact.get("calibration_patch", {}).get("analytical_costs") != artifact["cost_specs"]):
        _error("Fitted artifact membership or calibration patch is inconsistent")
    training_groups: dict[str, list[dict]] = defaultdict(list)
    for name in artifact["split"]["training_record_ids"]:
        training_groups[_domain_key(by_id[name])].append(by_id[name])
    costs = {}
    for diagnostic, cost in zip(artifact["diagnostics"], artifact["cost_specs"]):
        if not isinstance(diagnostic, dict) or not isinstance(cost, dict):
            _error("Fitted cost domains must be objects")
        key = diagnostic.get("domain_sha256")
        if not isinstance(key, str) or key not in training_groups or key in costs:
            _error("Duplicate, unknown or missing fitted cost domain")
        selected = training_groups[key]
        expected_ids = [row["record_id"] for row in selected]
        if (cost.get("training_record_ids") != expected_ids
                or cost_data_sha256(selected) != cost.get("training_dataset_sha256")
                or diagnostic.get("training_record_ids") != expected_ids):
            _error("Cost parameter training lineage does not match frozen records")
        first, domain = selected[0], selected[0]["domain"]
        if (diagnostic.get("domain") != domain or diagnostic.get("phase") != first["phase"]
                or diagnostic.get("model_id") != first["model_id"]
                or cost.get("phase") != first["phase"]
                or any(cost.get(field) != domain[field]
                       for field in ("identity_sha256", "execution_mode", "attention_backend"))):
            _error("Fitted cost identity, phase, mode or backend differs from its training domain")
        if (set(cost) != {field.name for field in fields(AnalyticalCostSpec)}
                or cost.get("calibration_id") != artifact.get("calibration_id")
                or not isinstance(cost.get("calibration_id"), str)
                or not cost["calibration_id"].strip()):
            _error("Fitted cost fields or calibration identity are inconsistent")
        layers = {row["layers"] for row in selected}
        if len(layers) != 1 or any(row["layers"] not in layers for row in data["records"]
                                   if _domain_key(row) == key):
            _error("A fitted identity must retain one consistent model layer count")
        bounds = {
            "min_batch_size": min(len(row["query_tokens"]) for row in selected),
            "max_batch_size": max(len(row["query_tokens"]) for row in selected),
            "min_query_tokens": min(min(row["query_tokens"]) for row in selected),
            "max_query_tokens": max(max(row["query_tokens"]) for row in selected),
            "min_context_tokens": min(min(row["context_lengths"]) for row in selected),
            "max_context_tokens": max(max(row["context_lengths"]) for row in selected),
        }
        if any(type(cost.get(field)) is not int or cost[field] != expected
               for field, expected in bounds.items()):
            _error("Fitted cost shape/context bounds differ from the frozen training support")
        for name in ("nonattention_scale", "attention_scale", "attention_fixed_us_per_layer", "runtime_fixed_us"):
            _number(cost.get(name), name)
        if cost.get("source_kind") != data["source"]["kind"]:
            _error("Fitted cost source kind differs from its source dataset")
        costs[key] = cost
    if set(costs) != set(training_groups):
        _error("Duplicate or missing fitted cost domains")
    boundaries = {row["domain"]["timing_boundary"] for members in training_groups.values() for row in members}
    if (len(boundaries) != 1 or artifact.get("algorithm") != ALGORITHM
            or artifact["calibration_patch"].get("backend") != "analytical"
            or artifact["calibration_patch"].get("analytical_cost_timing_boundary") != next(iter(boundaries))):
        _error("Fitted cost family or timing boundary differs from its training domain")
    result = {"schema_version": 1, "kind": "analytical_cost_validation_v1",
        "fit_sha256": artifact["fit_sha256"], "dataset_sha256": artifact["dataset_sha256"],
        "split_sha256": artifact["split"]["split_sha256"], "max_relative_error": threshold,
        "acceptance": deepcopy(artifact["split"]["acceptance"]),
        "selection_kind": "retrospective_group_holdout", "hardware_validated": False,
        "unused_record_ids": artifact["split"]["unused_record_ids"]}
    all_pass = True
    for label, field in (("holdout", "holdout_record_ids"), ("repeats", "repeat_record_ids")):
        rows = [by_id[name] for name in artifact["split"][field]]
        rejected = [{"record_id": row["record_id"], "reason": "outside_declared_training_ranges"}
                    for row in rows if not _in_domain(row, costs[_domain_key(row)])]
        covered = [row for row in rows if _in_domain(row, costs[_domain_key(row)])]
        calibrated = _errors(covered, lambda row: _predicted(row, costs[_domain_key(row)]), threshold)
        nominal = _errors(covered, lambda row: row["nominal_nonattention_ms"] + row["nominal_attention_ms"], threshold)
        # The legacy default has launch terms not recoverable from aggregate N/A
        # features. Include it only when the importer supplied its actual estimate.
        default_rows = [row for row in covered if "default_ms" in row]
        default = (_errors(default_rows, lambda row: _number(row["default_ms"], "default_ms"), threshold)
                   if len(default_rows) == len(covered) else None)
        passed = not rejected and calibrated["cells_passed"] == calibrated["cell_count"]
        if label == "holdout":
            all_pass = passed
        result[label] = {"declared_record_count": len(rows), "covered_record_count": len(covered),
            "acquisition_session_count": len({row["session_id"] for row in rows}),
            "acquisition_sessions": sorted({row["session_id"] for row in rows}),
            "coverage": len(covered) / len(rows) if rows else None, "rejected": rejected,
            "calibrated": calibrated, "nominal": nominal, "legacy_default": default,
            "verdict": "NOT_REQUESTED" if not rows else "PASS" if passed else "FAIL"}
    result["verdict"] = "PASS" if all_pass else "FAIL"
    result["validation_partitions_share_sessions"] = bool(set(result["holdout"]["acquisition_sessions"])
                                                         & set(result["repeats"]["acquisition_sessions"]))
    result["note"] = "Criterion uses each held-out session/shape median; steps are nested within sessions. Repeat diagnostics can share holdout sessions and are not additional independent acquisitions. All raw errors/rejections are retained; no refitting was performed."
    return result
