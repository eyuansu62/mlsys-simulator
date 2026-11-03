"""Fit versioned lookup profiles and evaluate genuinely disjoint holdouts.

These tools consume declared measurement records. They never collect GPU data,
and fixture records remain fixtures throughout fitting and validation.
"""

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from .schema import SimulationError


def _json_text(payload: Any, *, pretty: bool = False) -> str:
    try:
        return json.dumps(payload, sort_keys=True, allow_nan=False,
                          indent=2 if pretty else None,
                          separators=None if pretty else (",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_CONFIG", f"Expected finite JSON data: {exc}") from exc


def _clone(payload: Any) -> Any:
    return json.loads(_json_text(payload))


def _shape(record: dict[str, Any]) -> str:
    from .profiles import shape_key

    return shape_key(record["phase"], record["query_tokens"],
                     record["context_lengths"], record["stage_index"])


def calibrate_dataset(
    dataset: dict[str, Any],
    *,
    profile_version: str,
    parent_profile: dict[str, Any] | None = None,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    """Fit exact-shape means and observed workspace maxima into a new profile.

    A parent establishes identity/version lineage; its samples are not merged
    into the new fit. The input dictionaries and any earlier files are unchanged.
    The dataset fingerprint hashes the canonical supplied JSON, not whitespace.
    """
    from .profiles import validate_dataset, validate_profile

    data = validate_dataset(dataset, allow_test_fixture=allow_test_fixture)
    if not isinstance(profile_version, str) or not profile_version.strip():
        raise SimulationError("INVALID_CONFIG", "profile_version must be a nonempty string")
    previous_version = None
    if parent_profile is not None:
        parent = validate_profile(parent_profile, allow_test_fixture=allow_test_fixture)
        if parent["identity"] != data["identity"]:
            raise SimulationError("CALIBRATION_MISMATCH", "Parent profile identity differs from measurements")
        previous_version = parent["profile_version"]
        if previous_version == profile_version:
            raise SimulationError("PROFILE_VERSION_CONFLICT", "Calibration must produce a new profile version")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in data["records"]:
        grouped.setdefault(_shape(record), []).append(record)
    samples = []
    repeat_statistics = []
    for key, records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda record: record["record_id"])
        representative = ordered[0]
        workspace = [record["workspace_peak_bytes"] for record in ordered
                     if record["workspace_peak_bytes"] is not None]
        # statistics.mean avoids overflow from summing repeated large values.
        duration = float(statistics.mean(record["duration_ms"] for record in ordered))
        durations = [record["duration_ms"] for record in ordered]
        repeat_statistics.append({
            "shape_key": key, "sample_count": len(ordered),
            "duration_min_ms": min(durations), "duration_max_ms": max(durations),
            "duration_sample_stdev_ms": float(statistics.stdev(durations)) if len(durations) > 1 else None,
        })
        samples.append({
            "phase": representative["phase"],
            "stage_index": representative["stage_index"],
            "query_tokens": list(representative["query_tokens"]),
            "context_lengths": list(representative["context_lengths"]),
            "duration_ms": duration,
            "workspace_peak_bytes": max(workspace) if workspace else None,
            "sample_count": len(ordered),
        })
    profile = {
        "schema_version": 1,
        "profile_id": f"{data['dataset_id']}:{profile_version}",
        "profile_version": profile_version,
        "identity": _clone(data["identity"]),
        "source": _clone(data["source"]),
        "method": "lookup",
        "timing_includes_communication": True,
        "samples": samples,
        "training": {
            "dataset_sha256": hashlib.sha256(_json_text(dataset).encode("utf-8")).hexdigest(),
            "record_ids": sorted(record["record_id"] for record in data["records"]),
            "group_ids": sorted({record["group_id"] for record in data["records"]}),
            "shape_keys": sorted(grouped),
            "parent_profile_version": previous_version,
        },
        "notes": {"dataset_id": data["dataset_id"],
                  "fit": "mean_duration_and_max_observed_workspace",
                  "record_count": len(data["records"]),
                  "repeat_statistics": repeat_statistics,
                  "parent_samples_reused": False},
    }
    return validate_profile(profile, allow_test_fixture=allow_test_fixture)


def split_dataset(
    dataset: dict[str, Any],
    *,
    holdout_fraction: float = 0.25,
    seed: int = 42,
    holdout_groups: Sequence[str] | None = None,
    allow_test_fixture: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split whole connected configuration/group components, never repeats.

    The fraction is a fraction of independent components, rounded up, with at
    least one component in each partition. Explicit group selection expands to
    its entire connected component to prevent indirectly shared configurations.
    """
    from .profiles import validate_dataset

    data = validate_dataset(dataset, allow_test_fixture=allow_test_fixture)
    if (isinstance(holdout_fraction, bool) or not isinstance(holdout_fraction, (int, float))
            or not 0 < holdout_fraction < 1 or not math.isfinite(holdout_fraction)):
        raise SimulationError("INVALID_CONFIG", "holdout_fraction must be strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SimulationError("INVALID_CONFIG", "seed must be an integer")
    records = data["records"]
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        parents[find(right)] = find(left)

    first_shape: dict[str, int] = {}
    first_group: dict[str, int] = {}
    for index, record in enumerate(records):
        for lookup, key in ((first_shape, _shape(record)), (first_group, record["group_id"])):
            if key in lookup:
                union(index, lookup[key])
            else:
                lookup[key] = index
    components: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        components.setdefault(find(index), []).append(record)
    ordered = sorted(components.values(), key=lambda part: tuple(sorted(r["record_id"] for r in part)))
    if len(ordered) < 2:
        raise SimulationError("INSUFFICIENT_HOLDOUT_GROUPS",
                              "At least two independent configuration/group components are required")

    if holdout_groups is not None:
        if (isinstance(holdout_groups, (str, bytes)) or not isinstance(holdout_groups, Sequence)
                or not holdout_groups
                or any(not isinstance(group, str) or not group for group in holdout_groups)):
            raise SimulationError("INVALID_CONFIG", "holdout_groups must be a nonempty sequence of names")
        requested = set(holdout_groups)
        unknown = requested - set(first_group)
        if unknown:
            raise SimulationError("INVALID_CONFIG", f"Unknown holdout groups: {sorted(unknown)}")
        selected = {index for index, part in enumerate(ordered)
                    if any(record["group_id"] in requested for record in part)}
    else:
        indices = list(range(len(ordered)))
        random.Random(seed).shuffle(indices)
        count = min(len(ordered) - 1, max(1, math.ceil(len(ordered) * holdout_fraction)))
        selected = set(indices[:count])
    if not selected or len(selected) == len(ordered):
        raise SimulationError("INSUFFICIENT_HOLDOUT_GROUPS", "Both training and holdout must be nonempty")

    training, holdout = _clone(data), _clone(data)
    training["dataset_id"] = f"{data['dataset_id']}:training"
    holdout["dataset_id"] = f"{data['dataset_id']}:holdout"
    training["records"] = sorted([record for index, part in enumerate(ordered)
                                  if index not in selected for record in part],
                                 key=lambda record: record["record_id"])
    holdout["records"] = sorted([record for index, part in enumerate(ordered)
                                 if index in selected for record in part],
                                key=lambda record: record["record_id"])
    return (validate_dataset(training, allow_test_fixture=allow_test_fixture),
            validate_dataset(holdout, allow_test_fixture=allow_test_fixture))


def _error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    covered = [row for row in rows if row["status"] == "PREDICTED"]
    relative = [row["absolute_relative_error"] for row in covered
                if row["absolute_relative_error"] is not None]
    mean_relative = float(statistics.mean(relative)) if relative else None
    percent = mean_relative * 100 if mean_relative is not None else None
    if percent is not None and not math.isfinite(percent):
        percent = None
    return {
        "sample_count": len(rows),
        "predicted_count": len(covered),
        "rejected_count": len(rows) - len(covered),
        "coverage": len(covered) / len(rows) if rows else None,
        "extrapolated_count": sum(bool(row.get("extrapolated")) for row in covered),
        "mae_ms": float(statistics.mean(row["absolute_error_ms"] for row in covered)) if covered else None,
        "bias_ms": float(statistics.mean(row["signed_error_ms"] for row in covered)) if covered else None,
        "max_absolute_error_ms": max((row["absolute_error_ms"] for row in covered), default=None),
        "mape_percent": percent,
        "relative_error_count": len(relative),
        "mape_reason": (None if percent is not None else
                        "NO_COMPARABLE_RELATIVE_ERRORS_OR_NUMERIC_OVERFLOW"),
    }


def validate_holdout(
    profile: dict[str, Any],
    dataset: dict[str, Any],
    *,
    allow_test_fixture: bool = False,
    allow_extrapolation: bool = False,
) -> dict[str, Any]:
    """Evaluate disjoint held-out step observations, reporting rejected coverage."""
    from .profiles import estimate_profile, validate_dataset, validate_profile

    fit = validate_profile(profile, allow_test_fixture=allow_test_fixture)
    data = validate_dataset(dataset, allow_test_fixture=allow_test_fixture)
    if fit["identity"] != data["identity"]:
        raise SimulationError("CALIBRATION_MISMATCH", "Holdout identity differs from the fitted profile")
    training = fit["training"]
    overlaps = {
        "record_ids": set(training["record_ids"]) & {r["record_id"] for r in data["records"]},
        "group_ids": set(training["group_ids"]) & {r["group_id"] for r in data["records"]},
        "shape_keys": set(training["shape_keys"]) & {_shape(r) for r in data["records"]},
    }
    if any(overlaps.values()):
        counts = {name: len(values) for name, values in overlaps.items() if values}
        raise SimulationError("VALIDATION_DATA_LEAKAGE", f"Holdout overlaps fitting data: {counts}")
    rows: list[dict[str, Any]] = []
    for record in sorted(data["records"], key=lambda item: item["record_id"]):
        row = {"record_id": record["record_id"], "group_id": record["group_id"],
               "shape_key": _shape(record), "observed_ms": record["duration_ms"]}
        try:
            estimate = estimate_profile(fit, record["phase"], record["query_tokens"],
                                        record["context_lengths"], record["stage_index"],
                                        allow_extrapolation=allow_extrapolation,
                                        allow_test_fixture=allow_test_fixture)
        except SimulationError as exc:
            if exc.code != "OUT_OF_CALIBRATION_DOMAIN":
                raise
            row.update(status="OUT_OF_DOMAIN", predicted_ms=None, error_code=exc.code,
                       reason=str(exc), extrapolated=False)
        else:
            signed = estimate.duration_ms - record["duration_ms"]
            absolute = abs(signed)
            relative = absolute / record["duration_ms"]
            if not math.isfinite(relative):
                relative = None
            row.update(status="PREDICTED", predicted_ms=estimate.duration_ms,
                       signed_error_ms=signed, absolute_error_ms=absolute,
                       absolute_relative_error=relative,
                       relative_error_reason=None if relative is not None else "NUMERIC_OVERFLOW",
                       method=estimate.breakdown.get("method"),
                       extrapolated=bool(estimate.breakdown.get("extrapolated", False)),
                       error_code=None)
        rows.append(row)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    fixture = fit["source"]["kind"] == "test_fixture" or data["source"]["kind"] == "test_fixture"
    return {
        "schema_version": 1, "status": "COMPLETED",
        "validation_type": "grouped_holdout_step_latency",
        "profile_id": fit["profile_id"], "profile_version": fit["profile_version"],
        "dataset_id": data["dataset_id"], "identity": _clone(data["identity"]),
        "calibration_source": _clone(fit["source"]), "validation_source": _clone(data["source"]),
        "test_fixture_only": fixture,
        "hardware_validation": {"status": "not_asserted",
                                "reason": "The CPU tool evaluates supplied records; it does not collect hardware evidence."},
        "allow_extrapolation": allow_extrapolation,
        "summary": _error_summary(rows),
        "groups": {name: _error_summary(group_rows) for name, group_rows in sorted(groups.items())},
        "records": rows,
        "online_sla": None, "online_sla_reason": "STEP_DATASET_HAS_NO_ONLINE_TIMELINE",
        "candidate_ranking": None,
        "candidate_ranking_reason": "STEP_DATASET_HAS_NO_COMPARABLE_ONLINE_CANDIDATES",
    }


def write_json_exclusive(path: str | Path, payload: Any) -> None:
    """Write one immutable JSON artifact without overwriting any prior path."""
    serialized = _json_text(payload, pretty=True) + "\n"
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite {destination}") from exc
    except OSError as exc:
        raise SimulationError("IO_ERROR", f"Cannot write {destination}: {exc}") from exc
