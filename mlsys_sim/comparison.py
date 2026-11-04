"""Compare simulated runs with explicitly sourced, comparable observations.

Comparison is a read-only operation. It never upgrades a simulated run into a
measurement or interprets a missing metric as a zero-error observation.
"""

from collections.abc import Sequence
from dataclasses import asdict, fields
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, NoReturn

from .layout import stage_layer_counts
from .schema import Experiment, SimulationError


METRIC_UNITS = {
    **{f"{name}_p{percentile}_ms": "ms"
       for name in ("ttft", "tpot", "itl", "end_to_end")
       for percentile in (50, 95, 99)},
    "throughput_requests_per_s": "requests/s",
    "throughput_output_tokens_per_s": "tokens/s",
    "goodput_requests_per_s": "requests/s",
    "goodput_output_tokens_per_s": "tokens/s",
    "completion_rate": "fraction",
    "sla_satisfied": "boolean",
    "request_count": "requests",
}
REQUEST_METRICS = ("ttft_ms", "tpot_ms", "end_to_end_ms")


def _error(message: str, code: str = "INVALID_COMPARISON_INPUT") -> NoReturn:
    raise SimulationError(code, message)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_COMPARISON_INPUT", "Expected finite JSON comparison data") from exc


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _error(f"{name} must be an object with string keys")
    _canonical(value)
    return value


def _keys(value: dict, required: set[str], optional: set[str], name: str) -> None:
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        _error(f"{name}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")


def _contract(value: Any, name: str, *, measured: bool = False) -> dict:
    from .config import experiment_from_dict
    from .profiles import profile_identity, validate_identity
    from .schema import EvaluationSpec, ExecutionSpec, StrategySpec

    contract = _object(value, name)
    _keys(contract, {"schema_version", "identity", "workload_sha256", "strategy", "evaluation",
                     "execution", "timing_boundary"}, set(), name)
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1:
        _error("Only comparison contract schema version 1 is supported")
    digest = contract["workload_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        _error("comparison_contract.workload_sha256 must be a SHA256 digest")
    for field in ("identity", "strategy", "evaluation", "execution"):
        _object(contract[field], f"{name}.{field}")
    try:
        identity = validate_identity(contract["identity"], measured=measured)
        # V1 contracts predate these additive policy fields. Their documented
        # defaults preserve legacy single-group semantics when omitted.
        added_policy = {"pipeline_microbatch_size", "pipeline_max_inflight_microbatches",
                        "pipeline_feedback_latency_us", "pipeline_p2p", "swap_fallback",
                        "prefix_cache", "prefix_cache_max_blocks", "chunked_prefill",
                        "prefill_chunk_tokens", "mixed_batching", "replica_routing"}
        _keys(contract["strategy"], {field.name for field in fields(StrategySpec)} - added_policy,
              added_policy, f"{name}.strategy")
        _keys(contract["evaluation"], {field.name for field in fields(EvaluationSpec)}, set(), f"{name}.evaluation")
        _keys(contract["execution"], {field.name for field in fields(ExecutionSpec)}
              - {"detailed_events", "max_events"}, set(), f"{name}.execution")
        runtime, topology = identity["runtime"], identity["topology"]
        reconstructed = experiment_from_dict({
            "strategy": contract["strategy"], "evaluation": contract["evaluation"],
            "execution": contract["execution"],
            "infra": {"devices": identity["devices"], "topology": topology["name"],
                      "link_bandwidth_gbps": topology["link_bandwidth_gbps"],
                      "link_latency_us": topology["link_latency_us"]},
            "calibration": {"runtime_name": runtime["name"], "runtime_version": runtime["version"],
                            "driver_version": runtime["driver_version"], "rocm_version": runtime["rocm_version"],
                            "rccl_version": runtime["rccl_version"], "timing_boundary": runtime["timing_boundary"],
                            "runtime_config_sha256": runtime.get("runtime_config_sha256", "")},
        })
        if profile_identity(reconstructed) != identity:
            _error("comparison_contract strategy and hardware identity disagree")
    except SimulationError as exc:
        raise SimulationError("INVALID_COMPARISON_INPUT", f"Invalid comparison contract: {exc}") from exc
    if (contract["timing_boundary"] != "step_includes_output_and_sampling_v1"
            or contract["timing_boundary"] != contract["identity"]["runtime"]["timing_boundary"]):
        _error("comparison_contract uses an unsupported or inconsistent timing boundary")
    normalized_strategy = asdict(reconstructed.strategy)
    normalized_strategy["stage_layers"] = stage_layer_counts(reconstructed)
    normalized_execution = asdict(reconstructed.execution)
    normalized_execution.pop("max_events")
    normalized_execution.pop("detailed_events")
    return json.loads(_canonical({
        "schema_version": 1, "identity": identity, "workload_sha256": digest,
        "strategy": normalized_strategy, "evaluation": asdict(reconstructed.evaluation),
        "execution": normalized_execution, "timing_boundary": contract["timing_boundary"],
    }))


def _duplicates(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            _error(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates)
    except (OSError, ValueError) as exc:
        raise SimulationError("INVALID_COMPARISON_INPUT", f"Cannot read {path}: {exc}") from exc
    return _object(value, str(path))


def comparison_contract(experiment: Experiment) -> dict:
    """Describe the hardware, workload and timing semantics a comparison needs.

    Profile fitting coefficients, output paths, event logging and the simulator
    CPU event budget are not part of physical experiment equivalence.
    """
    from .config import experiment_from_dict
    from .profiles import profile_identity

    experiment = experiment_from_dict(asdict(experiment))
    strategy = asdict(experiment.strategy)
    strategy["stage_layers"] = stage_layer_counts(experiment)
    execution = asdict(experiment.execution)
    execution.pop("detailed_events")
    execution.pop("max_events")
    # Preserve V1 workload fingerprints when additive prompt identity is absent.
    workload = asdict(experiment)["workload"]
    for row in workload:
        if not row["prompt_token_ids"]:
            row.pop("prompt_token_ids")
        if not row["cache_namespace"]:
            row.pop("cache_namespace")
    result = {
        "schema_version": 1,
        "identity": profile_identity(experiment),
        "workload_sha256": hashlib.sha256(_canonical(workload).encode()).hexdigest(),
        "strategy": strategy,
        "evaluation": asdict(experiment.evaluation),
        "execution": execution,
        "timing_boundary": experiment.calibration.timing_boundary,
    }
    # Convert tuple-valued dataclass fields to their stable JSON representation.
    return json.loads(_canonical(result))


def _metric(value: Any, name: str) -> float | bool | None:
    if value is None:
        return None
    if name == "sla_satisfied":
        if type(value) is not bool:
            _error("sla_satisfied must be boolean or null")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{name} must be a nonnegative finite number or null")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_COMPARISON_INPUT", f"{name} is not finite") from exc
    if not math.isfinite(number) or number < 0:
        _error(f"{name} must be a nonnegative finite number")
    if name == "completion_rate" and number > 1:
        _error("completion_rate must be in [0, 1]")
    if name == "request_count" and (not number.is_integer() or number > 2 ** 53 - 1):
        _error("request_count must be an exactly represented nonnegative integer")
    return number


def _nested(value: dict, *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_metrics(result: dict) -> dict[str, float | bool | None]:
    """Flatten simulator metrics without calculating unavailable observations."""
    _object(result, "result")
    values: dict[str, Any] = {
        f"{name}_p{percentile}_ms": _nested(result, "latency_ms", name, f"p{percentile}")
        for name in ("ttft", "tpot", "itl", "end_to_end")
        for percentile in (50, 95, 99)
    }
    for category in ("throughput", "goodput"):
        for suffix in ("requests_per_s", "output_tokens_per_s"):
            values[f"{category}_{suffix}"] = _nested(result, category, suffix)
    values.update(completion_rate=result.get("cohort_completion_rate", result.get("completion_rate")),
                  sla_satisfied=result.get("sla_satisfied"),
                  request_count=result.get("evaluation_count", _nested(result, "counts", "total")))
    return {name: _metric(value, name) for name, value in values.items()}


def _observation(data: dict, allow_test_fixture: bool) -> dict:
    if data.get("backend") == "simulated" or data.get("kind") != "measured_run":
        _error("Observations must be measured_run records, never simulator results",
               "INVALID_OBSERVATION_SOURCE")
    _keys(data, {"schema_version", "kind", "comparison_contract", "source", "metrics"},
          {"requests", "notes"}, "observed run")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        _error("Only measured_run schema version 1 is supported")
    source = _object(data["source"], "observed.source")
    _keys(source, {"kind", "reference"}, set(), "observed.source")
    if source["kind"] not in ("measured", "test_fixture"):
        _error("Observation source must be measured or an explicitly enabled test_fixture",
               "INVALID_OBSERVATION_SOURCE")
    if source["kind"] == "test_fixture" and not allow_test_fixture:
        _error("A test_fixture observation needs allow_test_fixture=True", "TEST_FIXTURE_NOT_ALLOWED")
    if not isinstance(source["reference"], str) or not source["reference"].strip():
        _error("Observation source needs a nonempty reference")
    data["comparison_contract"] = _contract(data["comparison_contract"], "observed.comparison_contract",
                                             measured=source["kind"] == "measured")
    metrics = _object(data["metrics"], "observed.metrics")
    if set(metrics) - METRIC_UNITS.keys():
        _error(f"Unknown observed metrics: {sorted(set(metrics) - METRIC_UNITS.keys())}")
    if not metrics:
        _error("Observations need at least one metric")
    for name, value in metrics.items():
        _metric(value, name)
    if "notes" in data:
        _object(data["notes"], "observed.notes")
    return data


def _difference(predicted: Any, observed: Any, name: str, unit: str) -> dict:
    predicted, observed = _metric(predicted, name), _metric(observed, name)
    result = {"predicted": predicted, "observed": observed, "unit": unit,
              "absolute_error": None, "relative_error": None, "agreement": None}
    if predicted is None or observed is None:
        missing = ("both_unavailable" if predicted is None and observed is None
                   else "prediction_unavailable" if predicted is None else "observation_unavailable")
        return {**result, "status": "unavailable", "reason": missing}
    if name == "sla_satisfied":
        return {**result, "status": "compared", "reason": None, "agreement": predicted == observed}
    absolute = abs(float(predicted) - float(observed))
    relative = absolute / float(observed) if observed != 0 else None
    if not math.isfinite(absolute) or (relative is not None and not math.isfinite(relative)):
        _error(f"{name}: comparison error overflow")
    return {**result, "status": "compared", "reason": None,
            "absolute_error": absolute, "relative_error": relative,
            "relative_error_reason": "zero_observation" if observed == 0 else None}


def _request_rows(rows: Any, *, observed: bool) -> dict[str, dict]:
    if not isinstance(rows, list):
        _error("Request rows must be an array")
    result: dict[str, dict] = {}
    for row in rows:
        row = _object(row, "request")
        if observed:
            _keys(row, {"request_id"}, set(REQUEST_METRICS), "observed request")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in result:
            _error("Request IDs must be nonempty and unique")
        for name in REQUEST_METRICS:
            _metric(row.get(name), name)
        result[request_id] = row
    return result


def _requests(prediction_directory: Path, observation: dict) -> dict:
    if "requests" not in observation:
        return {"available": False, "reason": "observed_request_rows_not_supplied"}
    observed = _request_rows(observation["requests"], observed=True)
    path = prediction_directory / "requests.jsonl"
    predicted: dict[str, dict] = {}
    if path.exists():
        try:
            rows = [json.loads(line, object_pairs_hook=_duplicates)
                    for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as exc:
            raise SimulationError("INVALID_COMPARISON_INPUT", f"Cannot read {path}: {exc}") from exc
        predicted = _request_rows(rows, observed=False)
    shared = sorted(predicted.keys() & observed.keys())
    return {
        "available": True,
        "predicted_count": len(predicted), "observed_count": len(observed), "paired_count": len(shared),
        "unpaired_predicted_ids": sorted(predicted.keys() - observed.keys()),
        "unpaired_observed_ids": sorted(observed.keys() - predicted.keys()),
        "complete_pairing": predicted.keys() == observed.keys(),
        "reason": None if path.exists() else "predicted_request_rows_not_supplied",
        "rows": [{"request_id": request_id,
                  "metrics": {name: _difference(predicted[request_id].get(name),
                                                 observed[request_id].get(name), name, "ms")
                              for name in REQUEST_METRICS}}
                 for request_id in shared],
    }


def compare_runs(
    predicted_path: str | Path,
    observed_path: str | Path,
    *,
    allow_test_fixture: bool = False,
) -> dict:
    """Compare a saved simulator run with a sourced observed-run JSON file."""
    if type(allow_test_fixture) is not bool:
        _error("allow_test_fixture must be boolean")
    prediction_path, observation_path = Path(predicted_path), Path(observed_path)
    prediction_directory = prediction_path if prediction_path.is_dir() else prediction_path.parent
    result_path = prediction_directory / "result.json" if prediction_path.is_dir() else prediction_path
    result, manifest = _load(result_path), _load(prediction_directory / "manifest.json")
    if result.get("backend") != "simulated" or manifest.get("backend") != "simulated":
        _error("Predicted runs must be simulator outputs", "INVALID_PREDICTION_SOURCE")
    if result.get("status") not in ("COMPLETED", "UNRESOLVED"):
        _error("Prediction did not produce a completed or bounded simulation")
    contract = _contract(manifest.get("comparison_contract"), "manifest.comparison_contract")
    observation = _observation(_load(observation_path), allow_test_fixture)
    if _canonical(contract) != _canonical(observation["comparison_contract"]):
        _error("Workload, hardware, policy or timing boundaries differ", "COMPARISON_MISMATCH")
    predicted = extract_metrics(result)
    observed = {name: _metric(observation["metrics"].get(name), name) for name in METRIC_UNITS}
    metrics = {name: _difference(predicted[name], observed[name], name, unit)
               for name, unit in METRIC_UNITS.items()}
    count = observed["request_count"]
    compared = sum(row["status"] == "compared" for row in metrics.values())
    return {
        "schema_version": 1, "status": "COMPLETED", "comparison_contract": contract,
        "prediction_status": result["status"],
        "source": dict(observation["source"]),
        "predicted_source": result.get("calibration_source", manifest.get("calibration_source")),
        "observation_sha256": hashlib.sha256(_canonical(observation).encode()).hexdigest(),
        "metrics": metrics,
        "coverage": {"compared_metrics": compared, "total_metrics": len(metrics),
                     "fraction": compared / len(metrics), "complete": compared == len(metrics)},
        "request_comparison": _requests(prediction_directory, observation),
        "sla_judgment_agreement": metrics["sla_satisfied"]["agreement"],
        "sample_count": count,
        "tail_latency_evidence": "sample_count_unknown" if count is None
                                 else "limited" if count < 100 else "reported_without_confidence_interval",
        "hardware_validated": False,
        "limitations": ["This comparison does not establish general hardware prediction accuracy.",
                        "Tail latency requires sample counts, repeated runs and uncertainty analysis.",
                        "Errors are reported in each metric's own units; no mixed-unit mean is computed."],
    }


def _ranking(values: dict[str, float], higher_is_better: bool) -> list[list[str]]:
    by_value: dict[float, list[str]] = {}
    for candidate, value in values.items():
        by_value.setdefault(value, []).append(candidate)
    return [sorted(by_value[value]) for value in sorted(by_value, reverse=higher_is_better)]


def _order(left: float, right: float) -> int:
    return (left > right) - (left < right)


def compare_candidates(
    pairs: Sequence[dict],
    *,
    metric: str = "goodput_requests_per_s",
    higher_is_better: bool = True,
    allow_test_fixture: bool = False,
) -> dict:
    """Compare candidate orderings, retaining ties and unavailable scores."""
    if not isinstance(pairs, (list, tuple)) or not pairs:
        _error("Candidate pairs must be a nonempty list")
    if not isinstance(metric, str) or metric not in METRIC_UNITS or metric in ("sla_satisfied", "request_count"):
        _error("Ranking needs a supported numeric performance metric")
    if type(higher_is_better) is not bool:
        _error("higher_is_better must be boolean")
    reports: dict[str, dict] = {}
    shared_contract: str | None = None
    for pair in pairs:
        if isinstance(pair, dict):
            pair = {key: str(value) if isinstance(value, Path) else value for key, value in pair.items()}
        pair = _object(pair, "candidate pair")
        _keys(pair, {"candidate_id", "predicted", "observed"}, set(), "candidate pair")
        candidate = pair["candidate_id"]
        if not isinstance(candidate, str) or not candidate or candidate in reports:
            _error("Candidate IDs must be nonempty and unique")
        if not isinstance(pair["predicted"], (str, Path)) or not isinstance(pair["observed"], (str, Path)):
            _error("Candidate prediction and observation must be paths")
        report = compare_runs(pair["predicted"], pair["observed"], allow_test_fixture=allow_test_fixture)
        contract = report["comparison_contract"]
        common = _canonical({key: contract.get(key) for key in
                             ("workload_sha256", "evaluation", "execution", "timing_boundary")})
        if shared_contract is None:
            shared_contract = common
        elif shared_contract != common:
            _error("Candidates need a common workload, evaluation and timing contract", "COMPARISON_MISMATCH")
        reports[candidate] = report
    predicted: dict[str, float] = {}
    observed: dict[str, float] = {}
    excluded = []
    for candidate, report in reports.items():
        row = report["metrics"][metric]
        if row["status"] != "compared":
            excluded.append({"candidate_id": candidate, "reason": row["reason"]})
        else:
            predicted[candidate], observed[candidate] = row["predicted"], row["observed"]
    comparisons: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(predicted), 2):
        predicted_order, observed_order = _order(predicted[left], predicted[right]), _order(observed[left], observed[right])
        comparisons.append({"left": left, "right": right,
                            "predicted_order": predicted_order, "observed_order": observed_order,
                            "agreement": predicted_order == observed_order})
    agreement = (sum(pair["agreement"] for pair in comparisons) / len(comparisons)
                 if comparisons else None)
    return {"schema_version": 1, "status": "COMPLETED", "metric": metric,
            "unit": METRIC_UNITS[metric], "higher_is_better": higher_is_better,
            "candidate_reports": reports, "excluded_candidates": excluded,
            "predicted_ranking": _ranking(predicted, higher_is_better),
            "observed_ranking": _ranking(observed, higher_is_better),
            "pairwise_ordering": comparisons, "pairwise_agreement": agreement,
            "ranked_candidates": len(predicted), "total_candidates": len(reports),
            "ranking_complete": len(predicted) == len(reports)}
