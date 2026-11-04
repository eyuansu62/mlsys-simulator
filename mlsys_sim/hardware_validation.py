"""Explicit acceptance criteria over immutable calibration and online evidence.

This workflow evaluates declared measurements. It never turns fixture data or
client chunk timing into hardware proof, and its verdict applies only to the
identities, workloads, shapes and metrics listed in its validation scope.
"""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .calibration import calibrate_dataset, validate_holdout, write_json_exclusive
from .comparison import (METRIC_UNITS, REQUEST_METRICS, _metric, _observation,
                         _request_rows, compare_runs, comparison_contract)
from .config import load_experiment
from .layout import replica_experiment
from .profiles import profile_identity, validate_dataset
from .schema import SimulationError


def _fail(message: str, code: str = "INVALID_HARDWARE_VALIDATION_PLAN") -> None:
    raise SimulationError(code, message)


def _json(value: Any) -> str:
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_HARDWARE_VALIDATION_PLAN", "Expected finite JSON data") from exc


def _object(value: Any, required: set[str], optional: set[str], label: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(f"{label} must be an object with string keys")
    if required - value.keys() or value.keys() - required - optional:
        _fail(f"{label}: missing {sorted(required - value.keys())}; "
              f"unknown {sorted(value.keys() - required - optional)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, fraction: bool = False) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be a finite nonnegative number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_HARDWARE_VALIDATION_PLAN", f"{label} must be finite") from exc
    if not math.isfinite(number) or number < 0 or (fraction and number > 1):
        _fail(f"{label} must be finite and in {'[0,1]' if fraction else '[0,infinity)'}")
    return number


def _read(path: Path) -> tuple[Any, bytes]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        payload = json.loads(raw, object_pairs_hook=unique)
        _json(payload)
        return payload, raw
    except SimulationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SimulationError("IO_ERROR", f"Cannot read validation input {path}: {exc}") from exc


def _bytes_new(path: Path, content: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError as exc:
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite {path}") from exc
    except OSError as exc:
        raise SimulationError("IO_ERROR", f"Cannot write {path}: {exc}") from exc


def _directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite {path}") from exc
    except OSError as exc:
        raise SimulationError("IO_ERROR", f"Cannot create {path}: {exc}") from exc


def _thresholds(value: Any) -> dict:
    required = {"min_holdout_records", "min_holdout_coverage", "max_step_mape_percent",
                "min_online_requests", "min_online_repeats", "max_online_relative_error"}
    thresholds = _object(value, required, {"require_sla_agreement"}, "thresholds")
    result: dict[str, Any] = {name: _integer(thresholds[name], name, 1)
                             for name in ("min_holdout_records", "min_online_requests", "min_online_repeats")}
    result["min_holdout_coverage"] = _number(thresholds["min_holdout_coverage"],
                                               "min_holdout_coverage", fraction=True)
    if result["min_holdout_coverage"] == 0:
        _fail("min_holdout_coverage must be greater than zero")
    result["max_step_mape_percent"] = _number(thresholds["max_step_mape_percent"], "max_step_mape_percent")
    limits = thresholds["max_online_relative_error"]
    if not isinstance(limits, dict) or not limits:
        _fail("max_online_relative_error must be a nonempty metric-to-limit object")
    unknown = set(limits) - (METRIC_UNITS.keys() - {"sla_satisfied", "request_count"})
    if unknown:
        _fail(f"Unknown or nonnumeric acceptance metrics: {sorted(map(str, unknown))}")
    result["max_online_relative_error"] = {metric: _number(limit, f"max_online_relative_error.{metric}")
                                            for metric, limit in limits.items()}
    agreement = thresholds.get("require_sla_agreement", False)
    if type(agreement) is not bool:
        _fail("require_sla_agreement must be a boolean")
    result["require_sla_agreement"] = agreement
    return result


def run_hardware_validation(
    plan: dict,
    output_dir: Path,
    *,
    base_dir: Path | None = None,
    allow_test_fixture: bool = False,
) -> dict:
    """Fit, hold out, simulate and compare under explicit acceptance thresholds.

    Relative inputs resolve against ``base_dir``. The output directory may have
    been reserved by the CLI; each artifact written here is exclusive. An audit
    without canonical step and online evidence produces an insufficient verdict.
    Successful evaluation is distinct from a passing hardware acceptance verdict.
    """
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be a boolean")
    _json(plan)
    value = _object(plan, {"schema_version"},
        {"steps", "online", "thresholds", "evidence_audit", "communication_dataset",
         "communication_profile_version"}, "hardware validation plan")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        _fail("schema_version must be 1")
    steps, online = value.get("steps", []), value.get("online", [])
    if not isinstance(steps, list) or not isinstance(online, list):
        _fail("steps and online must be arrays")
    incomplete = not steps or not online
    if incomplete and "evidence_audit" not in value:
        _fail("steps and online must be nonempty unless evidence_audit is supplied")
    limits = _thresholds(value["thresholds"]) if "thresholds" in value else None
    if not incomplete and limits is None:
        _fail("A complete workflow requires explicit thresholds")
    if ("communication_dataset" in value) != ("communication_profile_version" in value):
        _fail("communication_dataset and communication_profile_version must be supplied together")
    root = Path.cwd() if base_dir is None else Path(base_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(output / "validation-plan.json", plan)
    input_files: list[dict] = []

    def path_for(value: Any, label: str) -> Path:
        return (root / _text(value, label)).resolve()

    def capture(path: Path, snapshot: Path) -> Any:
        payload, raw = _read(path)
        _bytes_new(snapshot, raw)
        input_files.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                            "snapshot_file": str(snapshot.relative_to(output)), "bytes": len(raw)})
        return payload

    audit = None
    if "evidence_audit" in value:
        audit = capture(path_for(value["evidence_audit"], "evidence_audit"), output / "evidence-audit.json")
    if incomplete:
        result = {"schema_version": 1, "status": "COMPLETED", "mode": "validate-hardware",
            "verdict": "INSUFFICIENT_EVIDENCE", "criteria_passed": False, "hardware_validated": False,
            "source_kinds": [], "thresholds": limits, "input_files": input_files,
            "criteria": [{"name": "canonical_evidence", "state": "INSUFFICIENT", "passed": False,
                          "reason": "Whole-stage training/holdout and canonical online observations are required"}],
            "evidence_audit_supplied": audit is not None,
            "validation_scope": {"dimensions": [], "reason": "An audit does not supply missing measurements"}}
        write_json_exclusive(output / "acceptance.json", result)
        return result
    assert limits is not None
    criteria: list[dict] = []
    sources: set[str] = set()
    fits: dict[int, dict] = {}
    profiles: dict[int, Path] = {}
    step_reports: list[dict] = []

    def criterion(name: str, passed: bool, observed: Any, threshold: Any,
                  *, insufficient: bool = False, reason: str | None = None) -> None:
        criteria.append({"name": name, "passed": passed,
            "state": "PASS" if passed else "INSUFFICIENT" if insufficient else "FAIL",
            "observed": observed, "threshold": threshold, "reason": reason})

    for position, supplied in enumerate(steps):
        item = _object(supplied, {"replica_index", "training_dataset", "holdout_dataset", "profile_version"},
                       set(), f"steps[{position}]")
        replica = _integer(item["replica_index"], "replica_index")
        if replica in fits:
            _fail("Each replica may have only one step calibration entry")
        version = _text(item["profile_version"], "profile_version")
        directory = output / f"step-replica-{replica}"
        _directory(directory)
        training_raw = capture(path_for(item["training_dataset"], "training_dataset"), directory / "training.json")
        holdout_raw = capture(path_for(item["holdout_dataset"], "holdout_dataset"), directory / "holdout.json")
        training = validate_dataset(training_raw, allow_test_fixture=allow_test_fixture)
        holdout = validate_dataset(holdout_raw, allow_test_fixture=allow_test_fixture)
        fit = calibrate_dataset(training, profile_version=version, allow_test_fixture=allow_test_fixture)
        report = validate_holdout(fit, holdout, allow_test_fixture=allow_test_fixture, allow_extrapolation=False)
        profile_path = directory / "profile.json"
        write_json_exclusive(profile_path, fit)
        write_json_exclusive(directory / "validation.json", report)
        fits[replica], profiles[replica] = fit, profile_path.resolve()
        sources.update((training["source"]["kind"], holdout["source"]["kind"]))
        summary = report["summary"]
        criterion(f"replica-{replica}.holdout_records", summary["sample_count"] >= limits["min_holdout_records"],
                  summary["sample_count"], limits["min_holdout_records"], insufficient=True)
        coverage = summary["coverage"]
        criterion(f"replica-{replica}.holdout_coverage", coverage is not None and coverage >= limits["min_holdout_coverage"],
                  coverage, limits["min_holdout_coverage"], insufficient=True)
        mape = summary["mape_percent"]
        criterion(f"replica-{replica}.step_mape_percent", mape is not None and mape <= limits["max_step_mape_percent"],
                  mape, limits["max_step_mape_percent"], insufficient=mape is None)
        step_reports.append({"replica_index": replica, "profile_id": fit["profile_id"],
            "profile_version": version, "source": fit["source"], "identity": fit["identity"],
            "summary": summary, "holdout_shapes": sorted({row["shape_key"] for row in report["records"]}),
            "artifact_directory": directory.name})

    communication = None
    if "communication_dataset" in value:
        from .communication_calibration import _validate_dataset, calibrate_communication
        raw = capture(path_for(value["communication_dataset"], "communication_dataset"), output / "communication-dataset.json")
        data = _validate_dataset(raw, allow_test_fixture=allow_test_fixture)
        communication = calibrate_communication(data,
            profile_version=_text(value["communication_profile_version"], "communication_profile_version"),
            allow_test_fixture=allow_test_fixture)
        sources.add(data["source"]["kind"])
        write_json_exclusive(output / "communication-profile.json", communication)

    from .cli import execute
    observations_seen: set[str] = set()
    group_contracts: dict[str, str] = {}
    group_values: dict[str, dict[str, list[float]]] = {}
    group_counts: dict[str, int] = {}
    online_reports: list[dict] = []
    native_communication_inputs: list[dict] = []
    for position, supplied in enumerate(online):
        item = _object(supplied, {"experiment", "observation", "group_id"}, set(), f"online[{position}]")
        group = _text(item["group_id"], "group_id")
        experiment_path = path_for(item["experiment"], "experiment")
        experiment = load_experiment(experiment_path)
        if set(fits) != set(range(experiment.strategy.replicas)):
            _fail("steps must cover every replica of each online experiment exactly")
        for replica, fit in fits.items():
            if profile_identity(replica_experiment(experiment, replica)) != fit["identity"]:
                raise SimulationError("CALIBRATION_MISMATCH", f"Online experiment differs from replica {replica} calibration")
        directory = output / f"online-{position}"
        _directory(directory)
        observation = capture(path_for(item["observation"], "observation"), directory / "observation.json")
        if not isinstance(observation, dict):
            _fail("Online observation must be an object")
        observation = _observation(observation, allow_test_fixture)
        sources.add(observation["source"]["kind"])
        request_rows = (_request_rows(observation["requests"], observed=True)
                        if "requests" in observation else {})
        # Normalize numeric forms, row ordering and unavailable fields. Merely
        # changing source labels, 1 to 1.0 or JSON order is not a new repeat.
        evidence = {"comparison_contract": observation["comparison_contract"],
            "metrics": {name: _metric(observation["metrics"].get(name), name) for name in METRIC_UNITS},
            "requests": [{"request_id": request_id,
                **{name: _metric(row.get(name), name) for name in REQUEST_METRICS}}
                for request_id, row in sorted(request_rows.items())]}
        fingerprint = hashlib.sha256(_json(evidence).encode()).hexdigest()
        if fingerprint in observations_seen:
            raise SimulationError("DUPLICATE_EVIDENCE", "Duplicate online content cannot count as independent repetition")
        observations_seen.add(fingerprint)
        contract = comparison_contract(experiment)
        contract_hash = hashlib.sha256(_json(contract).encode()).hexdigest()
        if group in group_contracts and group_contracts[group] != contract_hash:
            _fail("Online repeat group must retain the same complete comparison contract")
        group_contracts[group] = contract_hash
        # Stage samples include local communication. Standalone communication
        # patches are retained separately and are never double-added here.
        versions = {fit["profile_version"] for fit in fits.values()}
        calibration = replace(experiment.calibration, backend="profile", allow_extrapolation=False,
            allow_test_fixture=allow_test_fixture, profile_version=next(iter(versions)) if len(versions) == 1 else "uncalibrated-v1",
            profile_path=str(profiles[0]) if experiment.strategy.replicas == 1 else None,
            replica_profile_paths=tuple(str(profiles[index]) for index in range(len(fits)))
                                  if experiment.strategy.replicas > 1 else ())
        communication_paths = (experiment.calibration.replica_native_pp_communication_profile_paths
            or ((experiment.calibration.native_pp_communication_profile_path,)
                if experiment.calibration.native_pp_communication_profile_path else ()))
        if communication_paths:
            from .pp_communication import validate_pp_communication_profile

            communication_snapshots = []
            for replica, communication_path in enumerate(communication_paths):
                snapshot = directory / f"native-communication-r{replica}.json"
                raw = capture(Path(communication_path), snapshot)
                current = validate_pp_communication_profile(raw, allow_test_fixture=allow_test_fixture)
                if current["identity"] != profile_identity(replica_experiment(experiment, replica)):
                    raise SimulationError("CALIBRATION_MISMATCH",
                                          f"Native communication Profile differs from replica {replica}")
                requested = experiment.calibration.native_pp_communication_profile_version
                if requested and requested != current["profile_version"]:
                    raise SimulationError("CALIBRATION_MISMATCH", "Native communication Profile version differs")
                sources.add(current["source"]["kind"])
                native_communication_inputs.append({"online_index": position, "replica_index": replica,
                    "profile_id": current["profile_id"], "profile_version": current["profile_version"],
                    "source_kind": current["source"]["kind"], "sha256": input_files[-1]["sha256"],
                    "snapshot_file": str(snapshot.relative_to(output))})
                communication_snapshots.append(str(snapshot.resolve()))
            calibration = (replace(calibration, native_pp_communication_profile_path=communication_snapshots[0])
                if experiment.strategy.replicas == 1 else replace(calibration,
                    replica_native_pp_communication_profile_paths=tuple(communication_snapshots)))
        effective = replace(experiment, calibration=calibration)
        effective_path = directory / "experiment.json"
        write_json_exclusive(effective_path, asdict(effective))
        input_files.append({"path": str(experiment_path),
            "effective_sha256": hashlib.sha256(_json(asdict(experiment)).encode()).hexdigest(),
            "snapshot_file": str(effective_path.relative_to(output))})
        prediction = execute("simulate", effective_path, directory / "predicted")
        criterion(f"online-{position}.prediction_complete", prediction.get("status") == "COMPLETED",
                  prediction.get("status"), "COMPLETED", insufficient=True,
                  reason=prediction.get("error_code"))
        if prediction.get("status") not in {"COMPLETED", "UNRESOLVED"}:
            online_reports.append({"group_id": group, "artifact_directory": directory.name,
                                   "prediction_status": prediction.get("status"), "comparison": None})
            continue
        comparison = compare_runs(directory / "predicted", directory / "observation.json",
                                  allow_test_fixture=allow_test_fixture)
        write_json_exclusive(directory / "comparison.json", comparison)
        sources.add(comparison["source"]["kind"])
        pairing = comparison["request_comparison"]
        paired = pairing.get("paired_count", 0)
        declared_count = comparison["sample_count"]
        count_consistent = (pairing.get("complete_pairing") is True
                            and type(declared_count) in (int, float) and declared_count == pairing.get("observed_count"))
        criterion(f"online-{position}.request_pairing", count_consistent,
                  {"declared": declared_count, "paired": paired,
                   "observed_rows": pairing.get("observed_count")}, "complete pairing and consistent count", insufficient=True)
        tail_count = 100 if any("_p99_" in name for name in limits["max_online_relative_error"]) else 0
        required_count = max(limits["min_online_requests"], tail_count)
        criterion(f"online-{position}.paired_requests", paired >= required_count,
                  paired, required_count, insufficient=True)
        group_counts[group] = group_counts.get(group, 0) + 1
        values = group_values.setdefault(group, {})
        for metric, maximum in limits["max_online_relative_error"].items():
            row = comparison["metrics"][metric]
            error = row["relative_error"]
            criterion(f"online-{position}.{metric}.relative_error", error is not None and error <= maximum,
                      error, maximum, insufficient=error is None,
                      reason=row.get("reason") or row.get("relative_error_reason"))
            if row["observed"] is not None:
                values.setdefault(metric, []).append(float(row["observed"]))
        if limits["require_sla_agreement"]:
            agreement = comparison["sla_judgment_agreement"]
            criterion(f"online-{position}.sla_agreement", agreement is True, agreement, True,
                      insufficient=agreement is None)
        online_reports.append({"group_id": group, "artifact_directory": directory.name,
            "prediction_status": prediction["status"], "observation_sha256": fingerprint,
            "contract_sha256": contract_hash, "sample_count": declared_count,
            "paired_count": paired, "source": comparison["source"],
            "metric_errors": {metric: comparison["metrics"][metric]
                              for metric in limits["max_online_relative_error"]}})

    repeated: dict[str, dict] = {}
    for group in sorted(group_contracts):
        count = group_counts.get(group, 0)
        criterion(f"group.{group}.independent_repeats", count >= limits["min_online_repeats"],
                  count, limits["min_online_repeats"], insufficient=True)
        spread = {}
        for metric, observed_values in group_values.get(group, {}).items():
            mean = float(statistics.mean(observed_values))
            spread[metric] = {"sample_count": len(observed_values), "mean": mean, "min": min(observed_values),
                "max": max(observed_values),
                "sample_stdev": float(statistics.stdev(observed_values)) if len(observed_values) > 1 else None,
                "relative_spread": (max(observed_values) - min(observed_values)) / abs(mean) if mean else None}
        repeated[group] = {"independent_repeats": count, "contract_sha256": group_contracts[group], "metrics": spread}
    passed = bool(criteria) and all(row["passed"] for row in criteria)
    fixture = "test_fixture" in sources
    insufficient = any(row["state"] == "INSUFFICIENT" for row in criteria)
    verdict = ("TEST_FIXTURE_ONLY" if fixture else "INSUFFICIENT_EVIDENCE" if insufficient
               else "PASS" if passed else "FAIL")
    scope = {"dimensions": ["step_latency", "online_metrics"], "steps": [
        {key: item[key] for key in ("replica_index", "identity", "holdout_shapes")} for item in step_reports],
        "online_contracts": sorted(set(group_contracts.values())),
        "online_metrics": sorted(limits["max_online_relative_error"]),
        "thresholds": limits, "communication_prediction_validated": False}
    if native_communication_inputs:
        scope["native_pp_communication_profiles"] = native_communication_inputs
    result = {"schema_version": 1, "status": "COMPLETED", "mode": "validate-hardware",
        "verdict": verdict, "criteria_passed": passed, "hardware_validated": verdict == "PASS",
        "source_kinds": sorted(sources), "thresholds": limits, "criteria": criteria,
        "steps": step_reports, "online": online_reports, "repeat_variation": repeated,
        "communication_profile": "communication-profile.json" if communication is not None else None,
        "input_files": input_files, "validation_scope": scope,
        "validation_scope_sha256": hashlib.sha256(_json(scope).encode()).hexdigest(),
        "limitations": [
            "The verdict applies only to the supplied identities, held-out shapes, workloads and thresholds.",
            "Declared source metadata is checked as evidence data, not independently certified physical execution.",
            "Fixture inputs never establish hardware accuracy; sparse tails do not establish general p99 behavior.",
            "A standalone communication fit is not automatically applied or accepted as held-out communication validation.",
        ]}
    write_json_exclusive(output / "acceptance.json", result)
    return result
