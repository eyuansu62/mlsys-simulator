"""Opt-in whole-step affine costs with a predeclared process-stability gate.

This family needs no profiler component labels. Its shared macro scale and
intercept describe whole-step medians, not Attention or CPU attribution.
"""

from collections import defaultdict
from copy import deepcopy
import statistics

from . import cost_calibration as base


STEP_ALGORITHM = "whole_step_nonnegative_affine_session_medians_v1"
FIT_SPEC_KIND = "step_cost_fit_spec_v1"
REJECTION_KIND = "step_cost_fit_rejected_v1"
_AGGREGATION = "equal_weight_session_phase_shape_medians"
_STABILITY_METRIC = "same_domain_shape_max_session_median_over_min"


def freeze_step_fit_spec(split: dict, *, max_training_process_ratio: float = 1.25) -> dict:
    """Freeze the fit family and training gate separately from the old split ABI."""
    if (not isinstance(split, dict) or split.get("kind") != base.SPLIT_KIND
            or type(split.get("schema_version")) is not int or split.get("schema_version") != 1):
        base._error("Whole-step fit policy needs a frozen cost split")
    split_body = {key: value for key, value in split.items() if key != "split_sha256"}
    if base.cost_data_sha256(split_body) != split.get("split_sha256"):
        base._error("Cost split checksum differs before freezing the whole-step policy")
    threshold = base._number(max_training_process_ratio, "max_training_process_ratio", positive=True)
    if threshold < 1:
        base._error("Training process ratio threshold must be at least one")
    spec = {"schema_version": 1, "kind": FIT_SPEC_KIND, "algorithm": STEP_ALGORITHM,
        "split_sha256": base._digest(split.get("split_sha256"), "split_sha256"),
        "dataset_sha256": base._digest(split.get("dataset_sha256"), "dataset_sha256"),
        "aggregation": _AGGREGATION,
        "stability": {"metric": _STABILITY_METRIC, "max_training_process_ratio": threshold,
                      "minimum_sessions_per_domain": 2, "minimum_sessions_per_shape": 2,
                      "minimum_distinct_shapes": 2}}
    spec["fit_spec_sha256"] = base.cost_data_sha256(spec)
    return spec


def _check_fit_spec(split: dict, value: object) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("stability"), dict):
        base._error("A whole-step fit requires its separately frozen fit_spec")
    expected = freeze_step_fit_spec(split,
        max_training_process_ratio=value["stability"].get("max_training_process_ratio"))
    if expected != value:
        base._error("Whole-step fit policy, stability threshold or split binding was modified")
    return deepcopy(value)


def _training_groups(data: dict, split: dict) -> dict[str, list[dict]]:
    by_id = {row["record_id"]: row for row in data["records"]}
    groups: dict[str, list[dict]] = defaultdict(list)
    for name in split["training_record_ids"]:
        row = by_id[name]
        if row["kind"] != "step_wall":
            base._error("Whole-step fitting accepts selected step_wall observations only")
        groups[base._domain_key(row)].append(row)
    return groups


def _training_cells(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["session_id"], row["phase"], row["shape"])].append(row)
    cells = []
    for (session, phase, shape), members in sorted(grouped.items()):
        nominal = [base._number(row["nominal_nonattention_ms"] + row["nominal_attention_ms"],
                               "whole-step nominal cost", positive=True) for row in members]
        cells.append({"session_id": session, "phase": phase, "shape": shape,
            "sample_count": len(members), "record_ids": [row["record_id"] for row in members],
            "nominal_ms": statistics.median(nominal),
            "observed_ms": statistics.median(row["observed_ms"] for row in members)})
    return cells


def _stability(cells: list[dict], spec: dict) -> tuple[list[dict], list[str]]:
    policy = spec["stability"]
    reasons = []
    if len({cell["session_id"] for cell in cells}) < policy["minimum_sessions_per_domain"]:
        reasons.append("INSUFFICIENT_TRAINING_SESSIONS")
    by_shape: dict[str, list[dict]] = defaultdict(list)
    for cell in cells:
        by_shape[cell["shape"]].append(cell)
    if len(by_shape) < policy["minimum_distinct_shapes"]:
        reasons.append("INSUFFICIENT_DISTINCT_NOMINAL_SHAPES")
    diagnostics = []
    for shape, members in sorted(by_shape.items()):
        observed = [cell["observed_ms"] for cell in members]
        ratio = base._number(max(observed) / min(observed), "training process median ratio", positive=True)
        enough = len(members) >= policy["minimum_sessions_per_shape"]
        stable = enough and ratio <= policy["max_training_process_ratio"]
        if not enough:
            reasons.append("INSUFFICIENT_SHAPE_REPEATS")
        elif not stable:
            reasons.append("UNSTABLE_TRAINING_PROCESSES")
        diagnostics.append({"shape": shape, "session_count": len(members),
            "sessions": [cell["session_id"] for cell in members],
            "observed_session_medians_ms": observed,
            "nominal_session_medians_ms": [cell["nominal_ms"] for cell in members],
            "max_median_over_min": ratio, "passed": stable})
    features = [cell["nominal_ms"] for cell in cells]
    if max(features) - min(features) <= 1e-12 * max(1.0, max(features)):
        reasons.append("NON_IDENTIFIABLE_NOMINAL_FEATURES")
    return diagnostics, sorted(set(reasons))


def _domain_diagnostics(key: str, rows: list[dict], spec: dict) -> tuple[dict, list[str]]:
    layers = {row["layers"] for row in rows}
    if len(layers) != 1:
        base._error("A whole-step domain must identify one model layer count")
    cells = _training_cells(rows)
    stability, reasons = _stability(cells, spec)
    return {"domain_sha256": key, "domain": deepcopy(rows[0]["domain"]),
        "phase": rows[0]["phase"], "model_id": rows[0]["model_id"],
        "training_record_ids": [row["record_id"] for row in rows],
        "training_cells": cells, "stability": stability,
        "rejection_codes": reasons, "aggregation": _AGGREGATION,
        "interpretation": "Whole-step affine fit; component and CPU attribution are not identified"}, reasons


def fit_step_costs(dataset: dict, split: dict, *, calibration_id: str, fit_spec: dict) -> dict:
    """Fit whole-step medians only after all frozen training-domain gates pass.

    Statistical rejection is returned as a diagnostic object without executable
    costs. Invalid source data or changed policy/membership raises instead.
    """
    data, _ = base._check_split(dataset, split)
    base._text(calibration_id, "calibration_id")
    spec = _check_fit_spec(split, fit_spec)
    groups = _training_groups(data, split)
    diagnostics: list[dict] = []
    rejected: list[dict] = []
    for key, rows in sorted(groups.items()):
        diagnostic, reasons = _domain_diagnostics(key, rows, spec)
        diagnostics.append(diagnostic)
        rejected.extend({"domain_sha256": key, "code": reason} for reason in reasons)
    common = {"schema_version": 1, "algorithm": STEP_ALGORITHM,
        "calibration_id": calibration_id, "dataset_sha256": base.cost_data_sha256(data),
        "training_dataset_sha256": split["training_dataset_sha256"],
        "split": deepcopy(split), "step_fit_spec": spec, "fit_spec_sha256": spec["fit_spec_sha256"],
        "diagnostics": diagnostics, "hardware_validated": False}
    if rejected:
        result = {**common, "kind": REJECTION_KIND, "status": "REJECTED", "verdict": "FAIL",
                  "rejections": rejected, "fit_attempted": False,
                  "note": "No point model or calibration patch was produced; all training process medians are retained."}
        result["rejection_sha256"] = base.cost_data_sha256(result)
        return result
    costs = []
    for diagnostic in diagnostics:
        key = diagnostic["domain_sha256"]
        rows = groups[key]
        cells = diagnostic["training_cells"]
        # Reuse the established two-variable nonnegative affine solver. These
        # temporary keys are numeric inputs only, not component observations.
        inputs = [{"nominal_attention_ms": cell["nominal_ms"], "observed_attention_ms": cell["observed_ms"]}
                  for cell in cells]
        slope, intercept, numerical = base._attention_fit(inputs)
        domain = rows[0]["domain"]
        cost = {"phase": rows[0]["phase"], "execution_mode": domain["execution_mode"],
            "attention_backend": domain["attention_backend"], "identity_sha256": domain["identity_sha256"],
            "nonattention_scale": slope, "attention_scale": slope,
            "attention_fixed_us_per_layer": 0.0, "runtime_fixed_us": intercept * 1000,
            "min_batch_size": min(len(row["query_tokens"]) for row in rows),
            "max_batch_size": max(len(row["query_tokens"]) for row in rows),
            "min_query_tokens": min(min(row["query_tokens"]) for row in rows),
            "max_query_tokens": max(max(row["query_tokens"]) for row in rows),
            "min_context_tokens": min(min(row["context_lengths"]) for row in rows),
            "max_context_tokens": max(max(row["context_lengths"]) for row in rows),
            "calibration_id": calibration_id, "training_dataset_sha256": base.cost_data_sha256(rows),
            "training_record_ids": [row["record_id"] for row in rows], "source_kind": data["source"]["kind"]}
        costs.append(cost)
        diagnostic["whole_step_affine"] = {"scale": slope, "intercept_ms": intercept,
            "solver": "nonnegative_affine_least_squares", "numeric_diagnostics": numerical,
            "equal_weight_cell_count": len(cells)}
        diagnostic["training_wall_errors"] = base._errors(rows, lambda row: base._predicted(row, cost),
                                                         split["acceptance"]["max_relative_error"])
    boundaries = {row["domain"]["timing_boundary"] for rows in groups.values() for row in rows}
    if len(boundaries) != 1:
        base._error("One whole-step fit must use one target timing boundary")
    result = {**common, "kind": base.FIT_KIND, "cost_specs": costs,
        "calibration_patch": {"backend": "analytical", "analytical_costs": costs,
                              "analytical_cost_timing_boundary": next(iter(boundaries))},
        "assumptions": ["Whole-step affine fit from equally weighted acquisition-session/shape medians.",
            "The shared scale does not identify separate Attention, non-Attention or hardware efficiencies.",
            "The intercept is empirical unmodeled stage work, not measured CPU or launch time.",
            "The process stability threshold was frozen before fitting; validation targets were not used.",
            "Configured Graph/eager and backend domains remain exact; no unmeasured fallback is granted.",
            "The old split remains a retrospective data split; separate forward acquisition needs its own prediction freeze."]}
    result["fit_sha256"] = base.cost_data_sha256(result)
    return result


def validate_step_fit_metadata(dataset: dict, artifact: dict) -> None:
    """Bind this family's policy and training diagnostics without refitting it.

    Called alongside the existing common fit hash/domain/lineage checks.
    """
    if artifact.get("algorithm") != STEP_ALGORITHM or artifact.get("kind") != base.FIT_KIND:
        base._error("Expected a successful whole-step affine fit")
    data, _ = base._check_split(dataset, artifact["split"])
    spec = _check_fit_spec(artifact["split"], artifact.get("step_fit_spec"))
    if artifact.get("fit_spec_sha256") != spec["fit_spec_sha256"]:
        base._error("Whole-step fit policy hash differs from its frozen policy")
    groups = _training_groups(data, artifact["split"])
    diagnostics = {row["domain_sha256"]: row for row in artifact["diagnostics"]}
    if set(diagnostics) != set(groups) or len(diagnostics) != len(artifact["diagnostics"]):
        base._error("Whole-step diagnostics do not cover every trained domain exactly once")
    for key, rows in groups.items():
        expected, reasons = _domain_diagnostics(key, rows, spec)
        actual = diagnostics[key]
        if reasons or any(actual.get(field) != expected[field] for field in (
                "training_cells", "stability", "rejection_codes", "aggregation", "interpretation")):
            base._error("Whole-step training stability or median diagnostics differ from frozen observations")
    for cost in artifact["cost_specs"]:
        if (cost["nonattention_scale"] != cost["attention_scale"]
                or cost["attention_fixed_us_per_layer"] != 0):
            base._error("Whole-step affine costs must use one shared scale and zero Attention-only intercept")
        matching = [key for key, rows in groups.items()
                    if [row["record_id"] for row in rows] == cost.get("training_record_ids")]
        if len(matching) != 1:
            base._error("Whole-step costs must retain one unique training-domain membership")
        key = matching[0]
        diagnostic = diagnostics[key]
        affine = diagnostic.get("whole_step_affine")
        if (not isinstance(affine, dict)
                or base._number(affine.get("scale"), "whole-step diagnostic scale") != cost["nonattention_scale"]
                or base._number(affine.get("intercept_ms"), "whole-step diagnostic intercept") * 1000 != cost["runtime_fixed_us"]
                or affine.get("solver") != "nonnegative_affine_least_squares"
                or type(affine.get("equal_weight_cell_count")) is not int
                or affine["equal_weight_cell_count"] != len(diagnostic["training_cells"])):
            base._error("Whole-step coefficient diagnostics differ from frozen executable costs")
        expected_errors = base._errors(groups[key], lambda row: base._predicted(row, cost),
                                      artifact["split"]["acceptance"]["max_relative_error"])
        if diagnostic.get("training_wall_errors") != expected_errors:
            base._error("Whole-step training-error diagnostics differ from frozen executable costs")
