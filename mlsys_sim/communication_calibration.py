"""Fit declared collective/P2P measurements into explicit alpha-beta patches.

This module never applies a fit automatically. Its parameters are conditional
on one primitive, rank count, software identity and uncontended topology. A
whole-stage timing profile already containing communication must not receive
an additional communication charge.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from .profiles import validate_identity
from .schema import SimulationError


_METHOD = "nonnegative_intercept_alpha_beta_ols"
_APPLICATION = {
    "selection_required": True,
    "automatic_application": False,
    "parent_parameters_reused": False,
    "do_not_add_to_whole_stage_lookup": True,
}


def _fail(message: str, code: str = "INVALID_COMMUNICATION_DATASET") -> None:
    raise SimulationError(code, message)


def _object(value: Any, keys: set[str], name: str, *, code: str = "INVALID_COMMUNICATION_DATASET") -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(f"{name} must be an object with string keys", code)
    if set(value) != keys:
        _fail(f"{name}: missing {sorted(keys - value.keys())}; unknown {sorted(value.keys() - keys)}", code)
    return value


def _text(value: Any, name: str, *, code: str = "INVALID_COMMUNICATION_DATASET") -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a nonempty string", code)
    return value


def _integer(value: Any, name: str, minimum: int = 0,
             *, code: str = "INVALID_COMMUNICATION_DATASET") -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"{name} must be a signed 64-bit integer >= {minimum}", code)
    return value


def _number(value: Any, name: str, *, zero: bool = False,
            code: str = "INVALID_COMMUNICATION_DATASET") -> float:
    if type(value) not in (int, float):
        _fail(f"{name} must be a finite number", code)
    try:
        result = float(value)
    except (ValueError, OverflowError):
        _fail(f"{name} must be a finite number", code)
    if not math.isfinite(result) or (result < 0 if zero else result <= 0):
        _fail(f"{name} must be finite and {'nonnegative' if zero else 'positive'}", code)
    return result


def _source(value: Any, allow_test_fixture: bool, *, code: str = "INVALID_COMMUNICATION_DATASET") -> dict:
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be a boolean", code)
    source = _object(value, {"kind", "reference"}, "source", code=code)
    _text(source["reference"], "source.reference", code=code)
    if source["kind"] not in ("measured", "test_fixture"):
        _fail("source.kind must be measured or test_fixture; simulated output is not a measurement", code)
    if source["kind"] == "test_fixture" and not allow_test_fixture:
        _fail("Test fixtures require explicit allow_test_fixture", "TEST_FIXTURE_NOT_ALLOWED")
    return dict(source)


def _identity(value: Any, source: dict, code: str) -> dict:
    try:
        return validate_identity(value, measured=source["kind"] == "measured")
    except SimulationError as exc:
        raise SimulationError(code, f"Invalid communication identity: {exc}") from exc


def _record(value: Any, devices: int) -> dict:
    row = _object(value, {"record_id", "group_id", "primitive", "ranks", "message_bytes", "duration_ms"},
                  "record")
    primitive = row["primitive"]
    if primitive not in ("allreduce", "p2p"):
        _fail("primitive must be allreduce or p2p")
    ranks = _integer(row["ranks"], "ranks", 2)
    if ranks > devices:
        _fail("ranks must not exceed the number of identity devices")
    if primitive == "p2p" and ranks != 2:
        _fail("p2p measurements must use exactly two ranks")
    return {
        "record_id": _text(row["record_id"], "record_id"),
        "group_id": _text(row["group_id"], "group_id"),
        "primitive": primitive, "ranks": ranks,
        "message_bytes": _integer(row["message_bytes"], "message_bytes"),
        "duration_ms": _number(row["duration_ms"], "duration_ms"),
    }


def _validate_dataset(data: Any, *, allow_test_fixture: bool) -> dict:
    value = _object(data, {"schema_version", "kind", "dataset_id", "identity", "source", "records"},
                    "communication dataset")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        _fail("schema_version must be 1")
    if value["kind"] != "communication_measurements":
        _fail("kind must be communication_measurements")
    source = _source(value["source"], allow_test_fixture)
    identity = _identity(value["identity"], source, "INVALID_COMMUNICATION_DATASET")
    if not isinstance(value["records"], list) or not value["records"]:
        _fail("records must be a nonempty array")
    records = [_record(row, len(identity["devices"])) for row in value["records"]]
    if len({row["record_id"] for row in records}) != len(records):
        _fail("Duplicate communication record_id")
    return {"schema_version": 1, "kind": "communication_measurements",
            "dataset_id": _text(value["dataset_id"], "dataset_id"),
            "identity": identity, "source": source, "records": records}


def load_communication_dataset(path: str | Path, *, allow_test_fixture: bool = False) -> dict:
    """Read strict JSON communication observations without changing the input."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail(f"Duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    except SimulationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SimulationError("INVALID_COMMUNICATION_DATASET", f"{path}: {exc}") from exc
    return _validate_dataset(data, allow_test_fixture=allow_test_fixture)


def _strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        _fail(f"{name} must be a nonempty array", "INVALID_COMMUNICATION_PROFILE")
    result = [_text(item, name, code="INVALID_COMMUNICATION_PROFILE") for item in value]
    if len(set(result)) != len(result):
        _fail(f"{name} contains duplicates", "INVALID_COMMUNICATION_PROFILE")
    return result


def _validate_parent(data: Any, *, allow_test_fixture: bool) -> dict:
    code = "INVALID_COMMUNICATION_PROFILE"
    profile = _object(data, {"schema_version", "kind", "profile_id", "profile_version", "method",
                             "identity", "source", "groups", "training", "application", "limitations"},
                      "communication profile", code=code)
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        _fail("Parent schema_version must be 1", code)
    if profile["kind"] != "communication_profile" or profile["method"] != _METHOD:
        _fail("Parent must be a supported communication_profile", code)
    _text(profile["profile_id"], "profile_id", code=code)
    _text(profile["profile_version"], "profile_version", code=code)
    source = _source(profile["source"], allow_test_fixture, code=code)
    identity = _identity(profile["identity"], source, code)
    if (not isinstance(profile["application"], dict)
            or any(type(value) is not bool for value in profile["application"].values())
            or profile["application"] != _APPLICATION):
        _fail("Parent application policy differs from the supported explicit-patch policy", code)
    _strings(profile["limitations"], "limitations")
    training = _object(profile["training"], {"dataset_id", "dataset_sha256", "record_ids", "group_ids",
                                            "parent_profile_version"}, "training", code=code)
    _text(training["dataset_id"], "training.dataset_id", code=code)
    if (not isinstance(training["dataset_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", training["dataset_sha256"])):
        _fail("training.dataset_sha256 must be a SHA256 hex digest", code)
    _strings(training["record_ids"], "training.record_ids")
    _strings(training["group_ids"], "training.group_ids")
    if training["parent_profile_version"] is not None:
        _text(training["parent_profile_version"], "training.parent_profile_version", code=code)
    if not isinstance(profile["groups"], list) or not profile["groups"]:
        _fail("Parent groups must be a nonempty array", code)
    group_keys = set()
    all_records: set[str] = set()
    all_groups: set[str] = set()
    required = {"primitive", "ranks", "alpha_us", "bandwidth_gbps", "beta_ms_per_byte", "intercept_ms",
                "fit_slope_ms_per_byte", "intercept_clamped", "coverage", "mae_ms", "max_absolute_error_ms",
                "record_ids", "group_ids", "infra_patch", "formula"}
    for entry in profile["groups"]:
        group = _object(entry, required, "group", code=code)
        primitive = group["primitive"]
        if primitive not in ("allreduce", "p2p"):
            _fail("Parent group primitive is unsupported", code)
        ranks = _integer(group["ranks"], "group.ranks", 2, code=code)
        if ranks > len(identity["devices"]) or (primitive == "p2p" and ranks != 2):
            _fail("Parent group rank count is incompatible with identity", code)
        key = primitive, ranks
        if key in group_keys:
            _fail("Parent contains duplicate primitive/rank groups", code)
        group_keys.add(key)
        for name in ("alpha_us", "intercept_ms", "mae_ms", "max_absolute_error_ms"):
            _number(group[name], f"group.{name}", zero=True, code=code)
        for name in ("bandwidth_gbps", "beta_ms_per_byte", "fit_slope_ms_per_byte"):
            _number(group[name], f"group.{name}", code=code)
        if type(group["intercept_clamped"]) is not bool:
            _fail("group.intercept_clamped must be boolean", code)
        coverage = _object(group["coverage"], {"message_bytes_min", "message_bytes_max", "record_count",
                                               "distinct_message_sizes"}, "group.coverage", code=code)
        minimum = _integer(coverage["message_bytes_min"], "coverage minimum", code=code)
        maximum = _integer(coverage["message_bytes_max"], "coverage maximum", code=code)
        if maximum <= minimum:
            _fail("Parent coverage must contain different message sizes", code)
        _integer(coverage["distinct_message_sizes"], "coverage distinct sizes", 2, code=code)
        count = _integer(coverage["record_count"], "coverage record count", 2, code=code)
        if coverage["distinct_message_sizes"] > count:
            _fail("Parent distinct message sizes exceed its record count", code)
        ids = _strings(group["record_ids"], "group.record_ids")
        if len(ids) != count or all_records.intersection(ids):
            _fail("Parent record coverage is inconsistent", code)
        all_records.update(ids)
        all_groups.update(_strings(group["group_ids"], "group.group_ids"))
        patch = _object(group["infra_patch"], {"link_latency_us", "link_bandwidth_gbps"},
                        "group.infra_patch", code=code)
        _number(patch["link_latency_us"], "patch latency", zero=True, code=code)
        _number(patch["link_bandwidth_gbps"], "patch bandwidth", code=code)
        if patch != {"link_latency_us": group["alpha_us"], "link_bandwidth_gbps": group["bandwidth_gbps"]}:
            _fail("Parent infra_patch must match its fitted parameters", code)
        _text(group["formula"], "group.formula", code=code)
    if all_records != set(training["record_ids"]) or all_groups != set(training["group_ids"]):
        _fail("Parent training provenance differs from fitted groups", code)
    return deepcopy(profile)


def _fit_group(rows: list[dict]) -> dict:
    sizes = [row["message_bytes"] for row in rows]
    minimum, maximum = min(sizes), max(sizes)
    if minimum == maximum:
        _fail("Each primitive/rank group needs at least two distinct message sizes",
              "INSUFFICIENT_COMMUNICATION_SAMPLES")
    # Normalize both axes before covariance; byte differences are formed while
    # still integers so large, neighbouring message sizes do not collapse.
    span = maximum - minimum
    scale_y = max(row["duration_ms"] for row in rows)
    xs = [(size - minimum) / span for size in sizes]
    ys = [row["duration_ms"] / scale_y for row in rows]
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    variance = math.fsum((x - mean_x)**2 for x in xs)
    covariance = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    normalized_slope = covariance / variance
    if not math.isfinite(normalized_slope) or normalized_slope <= 0:
        _fail("Communication fit needs a positive finite time-per-byte slope", "INVALID_COMMUNICATION_FIT")
    intercept_fraction = mean_y - normalized_slope * (mean_x + minimum / span)
    clamped = intercept_fraction < 0
    if clamped:
        # Required nonnegative-latency boundary solution: refit through origin.
        full_x = [size / maximum for size in sizes]
        normalized_slope = math.fsum(x * y for x, y in zip(full_x, ys)) / math.fsum(x*x for x in full_x)
        slope = normalized_slope * (scale_y / maximum)
        intercept = 0.0
    else:
        slope = normalized_slope * (scale_y / span)
        intercept = intercept_fraction * scale_y
    slope = _number(slope, "fitted slope", code="INVALID_COMMUNICATION_FIT")
    intercept = _number(intercept, "fitted intercept", zero=True, code="INVALID_COMMUNICATION_FIT")
    primitive, ranks = rows[0]["primitive"], rows[0]["ranks"]
    rounds = 2 * (ranks - 1) if primitive == "allreduce" else 1
    traffic_factor = 2 * (ranks - 1) / ranks if primitive == "allreduce" else 1.0
    alpha_us = _number(intercept / rounds * 1000, "alpha_us", zero=True, code="INVALID_COMMUNICATION_FIT")
    beta = _number(slope / traffic_factor, "beta_ms_per_byte", code="INVALID_COMMUNICATION_FIT")
    # Rearrange as (1 / beta) / 1e6; avoid overflowing beta * 1e6.
    try:
        bandwidth = (1.0 / beta) / 1e6
    except ZeroDivisionError as exc:
        raise SimulationError("INVALID_COMMUNICATION_FIT", "Fitted bandwidth is not representable") from exc
    bandwidth = _number(bandwidth, "bandwidth_gbps", code="INVALID_COMMUNICATION_FIT")
    _number(bandwidth * 1e6, "effective link bytes/ms", code="INVALID_COMMUNICATION_FIT")
    residuals = []
    for row in rows:
        estimate = _number(intercept + slope * row["message_bytes"], "fitted prediction", zero=True,
                           code="INVALID_COMMUNICATION_FIT")
        residuals.append(abs(estimate - row["duration_ms"]))
    return {
        "primitive": primitive, "ranks": ranks, "alpha_us": alpha_us,
        "bandwidth_gbps": bandwidth, "beta_ms_per_byte": beta,
        "intercept_ms": intercept, "fit_slope_ms_per_byte": slope,
        "intercept_clamped": clamped,
        "coverage": {"message_bytes_min": minimum, "message_bytes_max": maximum,
                     "record_count": len(rows), "distinct_message_sizes": len(set(sizes))},
        "mae_ms": float(statistics.mean(residuals)), "max_absolute_error_ms": max(residuals),
        "record_ids": sorted(row["record_id"] for row in rows),
        "group_ids": sorted({row["group_id"] for row in rows}),
        "infra_patch": {"link_latency_us": alpha_us, "link_bandwidth_gbps": bandwidth},
        "formula": ("T_ms=2*(ranks-1)*alpha_us/1000+2*(ranks-1)/ranks*message_bytes/(bandwidth_gbps*1e6)"
                    if primitive == "allreduce" else
                    "T_ms=alpha_us/1000+message_bytes/(bandwidth_gbps*1e6)"),
    }


def calibrate_communication(
    data: dict[str, Any],
    *,
    profile_version: str,
    parent_profile: dict[str, Any] | None = None,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    """Fit one uncontended alpha-beta model per primitive/rank count.

    Fit parameters are not combined across rank counts or primitives. A parent
    records version/identity lineage only; its fitted parameters are not reused.
    """
    dataset = _validate_dataset(data, allow_test_fixture=allow_test_fixture)
    version = _text(profile_version, "profile_version", code="INVALID_CONFIG")
    previous_version = None
    if parent_profile is not None:
        parent = _validate_parent(parent_profile, allow_test_fixture=allow_test_fixture)
        if parent["identity"] != dataset["identity"]:
            _fail("Parent communication profile identity differs from observations", "CALIBRATION_MISMATCH")
        previous_version = parent["profile_version"]
        if previous_version == version:
            _fail("Communication calibration must produce a new profile version", "PROFILE_VERSION_CONFLICT")
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in dataset["records"]:
        groups.setdefault((row["primitive"], row["ranks"]), []).append(row)
    fits = [_fit_group(sorted(rows, key=lambda row: row["record_id"]))
            for _, rows in sorted(groups.items())]
    try:
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise SimulationError("INVALID_COMMUNICATION_DATASET", "Measurements must be finite JSON data") from exc
    return {
        "schema_version": 1, "kind": "communication_profile", "method": _METHOD,
        "profile_id": f"{dataset['dataset_id']}:{version}", "profile_version": version,
        "identity": deepcopy(dataset["identity"]), "source": deepcopy(dataset["source"]),
        "groups": fits,
        "training": {"dataset_id": dataset["dataset_id"], "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                     "record_ids": sorted(row["record_id"] for row in dataset["records"]),
                     "group_ids": sorted({row["group_id"] for row in dataset["records"]}),
                     "parent_profile_version": previous_version},
        "application": dict(_APPLICATION),
        "limitations": [
            "AllReduce uses a Ring alpha-beta baseline; P2P uses a single-transfer baseline.",
            "Fits assume the recorded primitive/rank identity and uncontended topology; algorithm switches are not modeled.",
            "Coverage describes observed message sizes, not a validated extrapolation envelope.",
            "Select one group explicitly before applying its InfraSpec patch; do not average groups.",
            "Whole-stage timing profiles already include communication; do not add this estimate again.",
            "duration_ms must be a standalone primitive measurement, not an end-to-end or whole-stage time.",
            "Fitting measurements or fixtures does not establish held-out AMD hardware validation.",
        ],
    }
