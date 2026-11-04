"""Exact per-rank dynamic Workspace bounds from explicit Runtime ownership.

Allocator peaks do not establish attribution. Only a complete ownership ledger
whose scope excludes fixed model/KV/graph/Runtime/transport allocations can
supply the separately selected replacement for the dynamic Workspace budget.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, NoReturn

from .profiles import profile_identity, shape_key, validate_identity
from .schema import Experiment, SimulationError


ATTRIBUTION = {
    "method": "runtime_workspace_ledger_v1", "complete": True,
    "scope": "exclusive_dynamic_workspace",
    "excluded_categories": ["weights", "kv_pool", "graphs", "runtime", "pp_buffers", "communication_buffers"],
    "includes_non_torch": True,
}
_MAX_BYTES = 32 * 1024 * 1024
_MAX_RECORDS = 1000000
_METHOD = "exact_rank_max_v1"


def _fail(message: str, code: str = "INVALID_WORKSPACE_DATASET") -> NoReturn:
    raise SimulationError(code, message)


def _object(value: Any, required: set[str], optional: set[str], label: str, code: str) -> dict:
    if (not isinstance(value, dict) or any(not isinstance(key, str) for key in value)
            or required - value.keys() or value.keys() - required - optional):
        _fail(f"{label} must have exactly the supported fields", code)
    return value


def _text(value: Any, label: str, code: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\x00" in value:
        _fail(f"{label} must be a bounded nonempty string", code)
    return value


def _integer(value: Any, label: str, code: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"{label} must be an integer >= {minimum}", code)
    return value


def _json(value: Any, code: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise SimulationError(code, "Workspace data must be finite JSON") from exc


def _source(value: Any, allow_test_fixture: bool, code: str) -> dict:
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be boolean", code)
    source = _object(value, {"kind", "reference"}, set(), "source", code)
    if source["kind"] not in ("measured", "test_fixture"):
        _fail("Workspace source must be measured or test_fixture", code)
    _text(source["reference"], "source.reference", code)
    if source["kind"] == "test_fixture" and not allow_test_fixture:
        _fail("Fixture Workspace requires explicit permission", "TEST_FIXTURE_NOT_ALLOWED")
    return deepcopy(source)


def _attribution(value: Any, code: str) -> dict:
    _object(value, set(ATTRIBUTION), set(), "attribution", code)
    # Canonical JSON distinguishes Python's bool==int equality.
    if _json(value, code) != _json(ATTRIBUTION, code):
        _fail("Workspace requires complete exclusive dynamic Runtime ownership, including non-Torch allocations", code)
    return deepcopy(ATTRIBUTION)


def _identity(value: Any, source: dict, code: str) -> dict:
    try:
        identity = validate_identity(value, measured=source["kind"] == "measured")
    except SimulationError as exc:
        raise SimulationError(code, f"Invalid Workspace identity: {exc}") from exc
    if identity["parallelism"].get("replicas", 1) != 1:
        _fail("Workspace Profiles require one replica identity", code)
    return identity


def _row(value: Any, identity: dict, *, profile: bool, code: str) -> dict:
    required = {"phase", "stage_index", "query_tokens", "context_lengths", "workspace_bytes_per_rank"}
    required |= {"sample_count"} if profile else {"record_id", "group_id"}
    row = _object(value, required, {"request_phases"}, "sample" if profile else "record", code)
    if row["phase"] != "mixed" and "request_phases" in row:
        _fail("Only mixed Workspace observations may carry request_phases", code)
    if row["phase"] not in ("prefill", "decode", "recompute", "mixed"):
        _fail("Unsupported Workspace phase", code)
    try:
        shape_key(row["phase"], row["query_tokens"], row["context_lengths"], row["stage_index"],
                  request_phases=row.get("request_phases"))
    except SimulationError as exc:
        raise SimulationError(code, f"Invalid Workspace shape: {exc}") from exc
    stage = _integer(row["stage_index"], "stage_index", code)
    if stage >= identity["parallelism"]["pipeline_parallel_size"]:
        _fail("Workspace stage exceeds the identity layout", code)
    tp = identity["parallelism"]["tensor_parallel_size"]
    ranks = row["workspace_bytes_per_rank"]
    if not isinstance(ranks, list) or len(ranks) != tp:
        _fail("Workspace observations must cover every TP rank exactly once", code)
    checked = {}
    for entry in ranks:
        entry = _object(entry, {"tp_rank", "peak_bytes"}, set(), "Workspace rank", code)
        rank = _integer(entry["tp_rank"], "tp_rank", code)
        peak = _integer(entry["peak_bytes"], "peak_bytes", code)
        if rank >= tp or rank in checked:
            _fail("Workspace TP rank is duplicated or outside the layout", code)
        checked[rank] = peak
    result = {"phase": row["phase"], "stage_index": stage,
        "query_tokens": list(row["query_tokens"]), "context_lengths": list(row["context_lengths"]),
        "workspace_bytes_per_rank": [{"tp_rank": rank, "peak_bytes": checked[rank]} for rank in range(tp)]}
    if row["phase"] == "mixed":
        result["request_phases"] = list(row["request_phases"])
    if profile:
        result["sample_count"] = _integer(row["sample_count"], "sample_count", code, 1)
    else:
        result["record_id"] = _text(row["record_id"], "record_id", code)
        result["group_id"] = _text(row["group_id"], "group_id", code)
    return result


def _key(row: dict) -> str:
    return shape_key(row["phase"], row["query_tokens"], row["context_lengths"], row["stage_index"],
                     request_phases=row.get("request_phases"))


def validate_workspace_dataset(data: dict, *, allow_test_fixture: bool = False) -> dict:
    code = "INVALID_WORKSPACE_DATASET"
    value = _object(data, {"schema_version", "kind", "dataset_id", "identity", "source", "attribution", "records"},
                    {"notes"}, "Workspace dataset", code)
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != "workspace_measurements":
        _fail("Unsupported Workspace dataset kind/version", code)
    source = _source(value["source"], allow_test_fixture, code)
    identity = _identity(value["identity"], source, code)
    attribution = _attribution(value["attribution"], code)
    if not isinstance(value["records"], list) or not 1 <= len(value["records"]) <= _MAX_RECORDS:
        _fail("Workspace records must be a bounded nonempty array", code)
    records = [_row(row, identity, profile=False, code=code) for row in value["records"]]
    if len({row["record_id"] for row in records}) != len(records):
        _fail("Duplicate Workspace record_id", code)
    result = {"schema_version": 1, "kind": "workspace_measurements", "dataset_id": _text(value["dataset_id"], "dataset_id", code),
        "identity": identity, "source": source, "attribution": attribution, "records": records}
    if "notes" in value:
        if not isinstance(value["notes"], dict):
            _fail("Workspace notes must be an object", code)
        result["notes"] = json.loads(_json(value["notes"], code))
    return result


def _read(path: str | Path, code: str) -> tuple[dict, bytes]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("Duplicate Workspace JSON key", code)
            result[key] = value
        return result
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            _fail("Workspace input exceeds the 32MiB limit", code)
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: _fail("Nonfinite Workspace JSON", code))
        _json(value, code)
    except SimulationError:
        raise
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise SimulationError(code, f"Cannot read Workspace input: {exc}") from exc
    if not isinstance(value, dict):
        _fail("Workspace file must contain one JSON object", code)
    return value, raw


def load_workspace_dataset(path: str | Path, *, allow_test_fixture: bool = False) -> dict:
    value, _ = _read(path, "INVALID_WORKSPACE_DATASET")
    return validate_workspace_dataset(value, allow_test_fixture=allow_test_fixture)


def _strings(value: Any, label: str, code: str) -> list[str]:
    if not isinstance(value, list) or not value:
        _fail(f"{label} must be a nonempty string array", code)
    checked = [_text(item, label, code) for item in value]
    if len(set(checked)) != len(checked):
        _fail(f"{label} contains duplicates", code)
    return checked


def validate_workspace_profile(data: dict, *, allow_test_fixture: bool = False) -> dict:
    code = "INVALID_WORKSPACE_PROFILE"
    value = _object(data, {"schema_version", "kind", "profile_id", "profile_version", "identity", "source", "attribution",
                           "method", "samples", "training"}, {"notes"}, "Workspace Profile", code)
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != "workspace_profile" or value["method"] != _METHOD):
        _fail("Unsupported Workspace Profile kind/version/method", code)
    source = _source(value["source"], allow_test_fixture, code)
    identity = _identity(value["identity"], source, code)
    attribution = _attribution(value["attribution"], code)
    if not isinstance(value["samples"], list) or not 1 <= len(value["samples"]) <= _MAX_RECORDS:
        _fail("Workspace samples must be a bounded nonempty array", code)
    samples = [_row(row, identity, profile=True, code=code) for row in value["samples"]]
    keys = [_key(row) for row in samples]
    if len(set(keys)) != len(keys):
        _fail("Duplicate Workspace Profile shape", code)
    training = _object(value["training"], {"dataset_sha256", "record_ids", "group_ids", "shape_keys", "parent_profile_version"},
                       set(), "Workspace training", code)
    if not isinstance(training["dataset_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", training["dataset_sha256"]):
        _fail("Workspace training requires a lowercase SHA256", code)
    record_ids = _strings(training["record_ids"], "training.record_ids", code)
    group_ids = _strings(training["group_ids"], "training.group_ids", code)
    shape_keys = _strings(training["shape_keys"], "training.shape_keys", code)
    if set(shape_keys) != set(keys) or sum(row["sample_count"] for row in samples) != len(record_ids) or len(group_ids) > len(record_ids):
        _fail("Workspace training coverage/counts differ from its samples", code)
    version = _text(value["profile_version"], "profile_version", code)
    parent = training["parent_profile_version"]
    if parent is not None:
        _text(parent, "parent_profile_version", code)
        if parent == version:
            _fail("Workspace child must use a new Profile version", code)
    result = {"schema_version": 1, "kind": "workspace_profile", "profile_id": _text(value["profile_id"], "profile_id", code),
        "profile_version": version, "identity": identity, "source": source, "attribution": attribution,
        "method": _METHOD, "samples": samples, "training": {"dataset_sha256": training["dataset_sha256"],
            "record_ids": record_ids, "group_ids": group_ids, "shape_keys": shape_keys, "parent_profile_version": parent}}
    if "notes" in value:
        if not isinstance(value["notes"], dict):
            _fail("Workspace notes must be an object", code)
        result["notes"] = json.loads(_json(value["notes"], code))
    return result


def load_workspace_profile(path: str | Path, *, allow_test_fixture: bool = False) -> dict:
    value, _ = _read(path, "INVALID_WORKSPACE_PROFILE")
    return validate_workspace_profile(value, allow_test_fixture=allow_test_fixture)


def calibrate_workspace(data: dict, *, profile_version: str, parent_profile: dict | None = None,
                        allow_test_fixture: bool = False) -> dict:
    dataset = validate_workspace_dataset(data, allow_test_fixture=allow_test_fixture)
    code = "INVALID_WORKSPACE_PROFILE"
    version = _text(profile_version, "profile_version", "INVALID_CONFIG")
    previous = None
    if parent_profile is not None:
        parent = validate_workspace_profile(parent_profile, allow_test_fixture=allow_test_fixture)
        if parent["identity"] != dataset["identity"] or parent["attribution"] != dataset["attribution"]:
            _fail("Parent Workspace identity or attribution differs", "CALIBRATION_MISMATCH")
        previous = parent["profile_version"]
        if previous == version:
            _fail("Workspace calibration requires a new Profile version", "PROFILE_VERSION_CONFLICT")
    grouped: dict[str, list[dict]] = {}
    for row in dataset["records"]:
        grouped.setdefault(_key(row), []).append(row)
    samples = []
    for _, rows in sorted(grouped.items()):
        first = rows[0]
        sample = {key: deepcopy(first[key]) for key in ("phase", "stage_index", "query_tokens", "context_lengths")}
        if first["phase"] == "mixed":
            sample["request_phases"] = list(first["request_phases"])
        sample["workspace_bytes_per_rank"] = [{"tp_rank": rank,
            "peak_bytes": max(row["workspace_bytes_per_rank"][rank]["peak_bytes"] for row in rows)}
            for rank in range(dataset["identity"]["parallelism"]["tensor_parallel_size"])]
        sample["sample_count"] = len(rows)
        samples.append(sample)
    profile = {"schema_version": 1, "kind": "workspace_profile", "profile_id": f"{dataset['dataset_id']}:{version}",
        "profile_version": version, "identity": deepcopy(dataset["identity"]), "source": deepcopy(dataset["source"]),
        "attribution": deepcopy(dataset["attribution"]), "method": _METHOD, "samples": samples,
        "training": {"dataset_sha256": hashlib.sha256(_json(data, code).encode()).hexdigest(),
            "record_ids": sorted(row["record_id"] for row in dataset["records"]),
            "group_ids": sorted({row["group_id"] for row in dataset["records"]}), "shape_keys": sorted(grouped),
            "parent_profile_version": previous},
        "notes": {"dataset_id": dataset["dataset_id"], "fit": "maximum_observed_bytes_per_exact_shape_and_rank",
            "parent_samples_reused": False, "scope": "replacement_for_dynamic_workspace_only",
            "interpolation": False, "extrapolation": False}}
    return validate_workspace_profile(profile, allow_test_fixture=allow_test_fixture)


class WorkspaceModel:
    """Cache one exact identity-bound Profile without re-reading its source."""

    def __init__(self, experiment: Experiment, profile: dict | None = None):
        from .config import validate_experiment
        if experiment.strategy.replicas != 1:
            _fail("WorkspaceModel requires one replica", "INVALID_CONFIG")
        validation_experiment = experiment
        if profile is not None and experiment.calibration.workspace_profile_path is None:
            validation_experiment = replace(experiment, calibration=replace(experiment.calibration,
                workspace_profile_path="in-memory-workspace-profile"))
        validate_experiment(validation_experiment)
        if profile is None:
            path = experiment.calibration.workspace_profile_path
            if path is None:
                _fail("workspace_profile_path is required", "INVALID_CONFIG")
            value, raw = _read(path, "INVALID_WORKSPACE_PROFILE")
            path_value = str(Path(path).resolve())
            scope = "file_bytes"
        else:
            value = deepcopy(profile)
            raw = _json(value, "INVALID_WORKSPACE_PROFILE").encode()
            path_value = None
            scope = "canonical_json"
        current = validate_workspace_profile(value, allow_test_fixture=experiment.calibration.allow_test_fixture)
        if current["identity"] != profile_identity(experiment):
            _fail("Workspace Profile identity differs from its local experiment", "CALIBRATION_MISMATCH")
        requested = experiment.calibration.workspace_profile_version
        if requested and current["profile_version"] != requested:
            _fail("Workspace Profile version differs", "CALIBRATION_MISMATCH")
        self._tp = experiment.strategy.tensor_parallel_size
        self._pp = experiment.strategy.pipeline_parallel_size
        self._index = {_key(row): row for row in current["samples"]}
        self.provenance = {"profile_id": current["profile_id"], "profile_version": current["profile_version"],
            "sha256": hashlib.sha256(raw).hexdigest(), "profile_sha256": hashlib.sha256(raw).hexdigest(),
            "sha256_scope": scope, "path": path_value, "source_kind": current["source"]["kind"],
            "source_reference": current["source"]["reference"], "attribution": deepcopy(current["attribution"]),
            "method": _METHOD, "scope": "replacement_for_dynamic_workspace_only", "domain_policy": "exact_only"}

    def workspace_bytes(self, phase: str, query_tokens, context_lengths, stage_index: int, tp_rank: int,
                        request_phases=None) -> int:
        code = "OUT_OF_WORKSPACE_CALIBRATION_DOMAIN"
        if type(stage_index) is not int or not 0 <= stage_index < self._pp or type(tp_rank) is not int or not 0 <= tp_rank < self._tp:
            _fail("Workspace query rank/stage is outside the layout", code)
        key = shape_key(phase, query_tokens, context_lengths, stage_index, request_phases=request_phases)
        row = self._index.get(key)
        if row is None:
            _fail("Workspace requires an observed exact phase/shape/stage/member-role sample", code)
        return row["workspace_bytes_per_rank"][tp_rank]["peak_bytes"]
