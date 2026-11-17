"""Stationarity-gated whole-step lookup Profiles in an explicit wall boundary.

Raw order and durations are retained as consistency evidence, not a digital
signature or proof that a particular hardware process was executed.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
import math
import statistics
from typing import NoReturn

from .schema import Experiment, MemoryConfig, SimulationError


REPORTED_STEP_BOUNDARY = "reported_l2_step_hook_wall_v1"
STATIONARITY_POLICY_KIND = "reported_step_stationarity_policy_v1"
STEP_PROFILE_PROOF_KIND = "reported_step_profile_proof_v1"
STEP_PROFILE_REJECTION_KIND = "step_profile_rejected_v1"
STEP_PROFILE_DURATION_STATISTIC = "median_of_per_exact_shape_session_medians_v1"
STEP_PROFILE_FIT = "median_of_session_medians_and_max_observed_workspace"
STEP_PROFILE_ASSUMPTIONS = (
    "Each exact lookup shape gives every training session equal weight through its median.",
    "Raw observations and outliers are retained; the duration target is a session median, not a mean or tail latency.",
    "Workspace uses the maximum observed value across every retained observation.",
)
MAX_STEP_PROFILE_RECORDS = 20_000
MAX_STEP_PROFILE_JSON_BYTES = 32 * 1024 * 1024
_POLICY_FIELDS = ("scheduler", "max_batch_size", "max_batch_tokens", "max_active_requests",
                  "chunked_prefill", "prefill_chunk_tokens", "mixed_batching", "prefix_cache", "preemption")


def _error(message: str) -> NoReturn:
    raise SimulationError("INVALID_STEP_PROFILE", message)


def _sha(value: object) -> str:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        data = text.encode("utf-8")
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise SimulationError("INVALID_STEP_PROFILE", "Expected bounded finite JSON wall-step data") from exc
    if len(data) > MAX_STEP_PROFILE_JSON_BYTES:
        _error("Reported-step proof exceeds the published JSON size limit")
    return hashlib.sha256(data).hexdigest()


def _check_step_json(value: object) -> None:
    """Bound new-boundary input before the legacy JSON snapshot copies it."""
    _sha(value)


def _text(value: object, name: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > 4096
            or any(ord(char) < 32 for char in value)):
        _error(f"{name} must be bounded nonempty text")
    return str(value)


def _ratio(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{name} must be a finite ratio of at least one")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 1:
        _error(f"{name} must be a finite ratio of at least one")
    return float(value)


def freeze_stationarity_policy(*, max_within_series_median_ratio: float = 1.10,
                                max_cross_session_median_ratio: float = 1.10) -> dict:
    """Declare thresholds/counts before observing the data; old Profiles are unaffected."""
    policy = {"schema_version": 1, "kind": STATIONARITY_POLICY_KIND,
        "timing_boundary": REPORTED_STEP_BOUNDARY,
        "phases": {"prefill": {"sample_count": 100, "block_size": 20},
                   "decode": {"sample_count": 128, "block_size": 32}},
        "max_within_series_median_ratio": _ratio(max_within_series_median_ratio, "within-series threshold"),
        "max_cross_session_median_ratio": _ratio(max_cross_session_median_ratio, "cross-session threshold"),
        "minimum_training_sessions_per_group": 2,
        "aggregation": "source-ordered contiguous blocks; no sorting, dropping or trimming"}
    policy["policy_sha256"] = _sha(policy)
    return policy


def _policy(value: dict) -> dict:
    if not isinstance(value, dict):
        _error("A separately frozen stationarity policy is required")
    if _sha({key: item for key, item in value.items() if key != "policy_sha256"}) != value.get("policy_sha256"):
        _error("Stationarity policy checksum was modified")
    expected = freeze_stationarity_policy(
        max_within_series_median_ratio=_ratio(value.get("max_within_series_median_ratio"), "within-series threshold"),
        max_cross_session_median_ratio=_ratio(value.get("max_cross_session_median_ratio"), "cross-session threshold"))
    if _sha(value) != _sha(expected):
        _error("Stationarity policy, sample/block counts or policy checksum was modified")
    return deepcopy(value)


def _check_core_policy(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != set(_POLICY_FIELDS):
        _error("Wall domain must include every declared core scheduling policy field")
    if not isinstance(value["scheduler"], str) or value["scheduler"] not in {"fifo", "continuous_batching"}:
        _error("Unsupported wall-domain scheduler")
    if not isinstance(value["preemption"], str) or value["preemption"] not in {"none", "recompute", "swap"}:
        _error("Unsupported wall-domain preemption")
    for name in ("max_batch_size", "max_batch_tokens", "prefill_chunk_tokens"):
        if type(value[name]) is not int or value[name] < 1:
            _error(f"Wall-domain {name} must be a positive integer")
    active = value["max_active_requests"]
    if active is not None and (type(active) is not int or active < 1):
        _error("Wall-domain active request cap must be null or a positive integer")
    for name in ("chunked_prefill", "mixed_batching", "prefix_cache"):
        if type(value[name]) is not bool:
            _error(f"Wall-domain {name} must be boolean")
    return deepcopy(value)


def _check_memory_policy(value: object, identity: dict) -> dict:
    from .config import experiment_from_dict

    if not isinstance(value, dict) or set(value) != {field.name for field in fields(MemoryConfig)}:
        _error("Wall domain must include the complete explicit MemoryConfig")
    memory = asdict(experiment_from_dict({"strategy": {"memory": value}}).strategy.memory)
    if any(memory[name] != setting for name, setting in identity["execution"].items()):
        _error("Wall-domain memory differs from the Profile execution identity")
    return memory


def _wall_domain(identity: dict, policy: object, memory: object) -> dict:
    if identity["runtime"]["timing_boundary"] != REPORTED_STEP_BOUNDARY:
        _error("Reported-step Profiles require their explicit wall timing boundary")
    parallel = identity["parallelism"]
    if (parallel["tensor_parallel_size"] != 1 or parallel["pipeline_parallel_size"] != 1
            or parallel.get("replicas", 1) != 1):
        _error("Reported-step wall Profiles initially require TP1/PP1/one replica")
    result = {"kind": "reported_step_wall_domain_v1", "identity_sha256": _sha(identity),
              "core_policy": _check_core_policy(policy), "memory_policy": _check_memory_policy(memory, identity)}
    result["wall_domain_sha256"] = _sha(result)
    return result


def step_wall_domain(experiment: Experiment) -> dict:
    """Bind a reported-step Profile to both normal identity and wall-time policy."""
    from .config import validate_experiment
    from .profiles import profile_identity

    validate_experiment(experiment)
    if experiment.calibration.backend != "profile":
        _error("The reported step boundary is allowed only for the Profile backend")
    return _wall_domain(profile_identity(experiment),
                        {name: getattr(experiment.strategy, name) for name in _POLICY_FIELDS},
                        asdict(experiment.strategy.memory))


def _dataset_series(data: dict) -> tuple[dict, list[dict]]:
    """Check source record ordering before any normal lookup aggregation sorts IDs."""
    from .profiles import shape_key

    records = data["records"]
    if len(records) > MAX_STEP_PROFILE_RECORDS:
        _error("Reported-step records exceed the published training size limit")
    notes = data.get("notes", {})
    wall = notes.get("wall_domain")
    if not isinstance(wall, dict):
        _error("Reported-step dataset notes require an explicit wall_domain")
    expected = _wall_domain(data["identity"], wall.get("core_policy"), wall.get("memory_policy"))
    if wall != expected:
        _error("Wall-domain identity/policy/checksum does not match the dataset")
    ordered = notes.get("ordered_series")
    if not isinstance(ordered, list) or not ordered:
        _error("Reported-step dataset notes require nonempty ordered_series")
    indexed = {row["record_id"]: row for row in records}
    positions = {row["record_id"]: index for index, row in enumerate(records)}
    covered: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    checked = []
    for series in ordered:
        if not isinstance(series, dict) or set(series) != {"session_id", "group_id", "record_ids"}:
            _error("Each ordered series needs only session_id, group_id and source-ordered record_ids")
        session, group = (_text(series[name], "ordered_series." + name) for name in ("session_id", "group_id"))
        if (session, group) in pairs:
            _error("A session/group must have one complete formal series, not split fragments")
        pairs.add((session, group))
        names = series["record_ids"]
        if (not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names) or not set(names) <= set(indexed)):
            _error("Ordered-series IDs must be unique existing formal records")
        if covered & set(names):
            _error("A formal record appears in more than one ordered series")
        if [positions[name] for name in names] != sorted(positions[name] for name in names):
            _error("Ordered-series IDs do not follow source records; ID sorting is not chronology")
        covered.update(names)
        rows = [indexed[name] for name in names]
        phases = {row["phase"] for row in rows}
        if (len(phases) != 1 or not phases <= {"prefill", "decode"}
                or any(row["stage_index"] != 0 or row["group_id"] != group for row in rows)):
            _error("A formal ordered series must have one phase, stage0 and matching group_id")
        phase = next(iter(phases))
        keys = [shape_key(row["phase"], row["query_tokens"], row["context_lengths"], row["stage_index"])
                for row in rows]
        if phase == "prefill" and len(set(keys)) != 1:
            _error("A formal Prefill series must contain one exact stage/query/context shape")
        if phase == "decode" and (len({tuple(row["query_tokens"]) for row in rows}) != 1
                                   or any(query != 1 for row in rows for query in row["query_tokens"])):
            _error("A formal Decode series requires fixed batch/query shape with one token per member")
        for row in rows:
            if (len(row["query_tokens"]) > wall["core_policy"]["max_batch_size"]
                    or sum(row["query_tokens"]) > wall["core_policy"]["max_batch_tokens"]):
                _error("Formal shape exceeds its declared wall-domain batch policy")
        checked.append({"session_id": session, "group_id": group, "phase": phase,
            "record_ids": list(names), "durations_ms": [row["duration_ms"] for row in rows],
            "shape_schedule_sha256": _sha(keys)})
    if covered != set(indexed):
        _error("Ordered series must cover every selected formal record exactly once")
    _sha(data)
    return expected, checked


def validate_reported_step_dataset(data: dict) -> None:
    """Extra contract for an already normalized new-boundary dataset."""
    _dataset_series(data)


def _median(values: list[float]) -> float:
    """Avoid overflow when the middle two valid observations are both large."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else float(statistics.mean(ordered[middle - 1:middle + 1]))


def _median_ratio(values: list[float]) -> float | None:
    ratio = max(values) / min(values)
    return ratio if math.isfinite(ratio) else None


def _lookup_sessions(data: dict, series: list[dict]) -> dict[str, dict[str, list[dict]]]:
    from .profiles import shape_key

    sessions = {name: item["session_id"] for item in series for name in item["record_ids"]}
    grouped: dict[str, dict[str, list[dict]]] = {}
    # Use the original source order, including when one session has multiple groups.
    for row in data["records"]:
        key = shape_key(row["phase"], row["query_tokens"], row["context_lengths"], row["stage_index"])
        grouped.setdefault(key, {}).setdefault(sessions[row["record_id"]], []).append(row)
    return grouped


def reported_step_aggregates(data: dict) -> dict[str, dict]:
    """The explicit new-boundary target; raw outliers remain in the proof."""
    _, series = _dataset_series(data)
    result = {}
    for key, sessions in _lookup_sessions(data, series).items():
        rows = [row for members in sessions.values() for row in members]
        spaces = [row["workspace_peak_bytes"] for row in rows if row["workspace_peak_bytes"] is not None]
        result[key] = {"duration_ms": _median([_median([row["duration_ms"] for row in members])
                                              for members in sessions.values()]),
                       "sample_count": len(rows), "workspace_peak_bytes": max(spaces) if spaces else None}
    return result


def _evaluate(data: dict, policy: dict) -> dict:
    wall, series = _dataset_series(data)
    frozen = _policy(policy)
    evaluations = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in series:
        phase = item["phase"]
        expected = frozen["phases"][phase]
        values = item["durations_ms"]
        if len(values) != expected["sample_count"] or len(values) % expected["block_size"]:
            _error("Formal series has missing/excess samples or an incomplete trailing block")
        blocks = [_median(values[start:start + expected["block_size"]])
                  for start in range(0, len(values), expected["block_size"])]
        ratio = _median_ratio(blocks)
        result = {**item, "sample_count": len(values), "block_medians_ms": blocks,
                  "within_series_ratio": ratio,
                  "median_ms": _median(values),
                  "passed": ratio is not None and ratio <= frozen["max_within_series_median_ratio"]}
        evaluations.append(result)
        grouped[item["group_id"]].append(result)
    cross = []
    for group, members in sorted(grouped.items()):
        if len({(row["phase"], row["shape_schedule_sha256"]) for row in members}) != 1:
            _error("A repeated group must retain the same source-ordered phase/shape schedule")
        values = [row["median_ms"] for row in members]
        enough = len(members) >= frozen["minimum_training_sessions_per_group"]
        ratio = _median_ratio(values)
        cross.append({"group_id": group, "session_count": len(members),
            "sessions": [row["session_id"] for row in members], "session_medians_ms": values,
            "cross_session_ratio": ratio,
            "passed": enough and ratio is not None and ratio <= frozen["max_cross_session_median_ratio"]})
    lookup_shapes = []
    for key, sessions in sorted(_lookup_sessions(data, series).items()):
        per_session = []
        for session, rows in sessions.items():
            values = [row["duration_ms"] for row in rows]
            if rows[0]["phase"] == "decode" and (len(values) < 8 or len(values) % 2):
                _error("Each Decode lookup shape needs at least eight observations per session and two complete halves")
            middle = len(values) // 2
            halves = [_median(values[:middle]), _median(values[middle:])]
            ratio = _median_ratio(halves)
            per_session.append({"session_id": session, "record_ids": [row["record_id"] for row in rows],
                "sample_count": len(values), "median_ms": _median(values), "source_ordered_half_medians_ms": halves,
                "within_shape_ratio": ratio,
                "passed": ratio is not None and ratio <= frozen["max_within_series_median_ratio"]})
        values = [row["median_ms"] for row in per_session]
        ratio = _median_ratio(values)
        lookup_shapes.append({"shape_key": key, "sessions": per_session, "cross_session_ratio": ratio,
            "passed": len(per_session) >= frozen["minimum_training_sessions_per_group"]
                and all(row["passed"] for row in per_session)
                and ratio is not None and ratio <= frozen["max_cross_session_median_ratio"]})
    return {"kind": "reported_step_stationarity_report_v1", "dataset_sha256": _sha(data),
        "policy_sha256": frozen["policy_sha256"], "wall_domain_sha256": wall["wall_domain_sha256"],
        "series": evaluations, "cross_session_groups": cross, "lookup_shapes": lookup_shapes,
        "eligibility": "Prefill exact shape; Decode fixed unit query and >=8 observations per session/exact lookup shape; source-ordered halves",
        "duration_statistic": STEP_PROFILE_DURATION_STATISTIC,
        "verdict": "PASS" if all(row["passed"] for row in evaluations + cross + lookup_shapes) else "FAIL",
        "hardware_validated": False}


def check_step_stationarity(dataset: dict, policy: dict, *, allow_test_fixture: bool = False) -> dict:
    """Evaluate untrimmed source order; a failing result contains all diagnostics."""
    from .profiles import validate_dataset

    data = validate_dataset(dataset, allow_test_fixture=allow_test_fixture)
    return _evaluate(data, policy)


def calibrate_step_profile(dataset: dict, policy: dict, *, profile_version: str,
                           parent_profile: dict | None = None, allow_test_fixture: bool = False) -> dict:
    """Fit each shape's median of session medians after all stationarity gates.

    Every raw observation remains in the proof. Nonstationary data never
    produce executable samples; the target does not describe tail latency.
    """
    from .calibration import calibrate_dataset
    from .profiles import validate_dataset

    data = validate_dataset(dataset, allow_test_fixture=allow_test_fixture)
    frozen = _policy(policy)
    report = _evaluate(data, frozen)
    if report["verdict"] != "PASS":
        rejected = {"schema_version": 1, "kind": STEP_PROFILE_REJECTION_KIND, "status": "REJECTED",
            "verdict": "FAIL", "dataset_sha256": _sha(data), "stationarity_policy": frozen,
            "stationarity": report, "raw_dataset": data, "hardware_validated": False}
        rejected["rejection_sha256"] = _sha(rejected)
        return rejected
    proof = {"schema_version": 1, "kind": STEP_PROFILE_PROOF_KIND, "raw_dataset": data,
             "raw_dataset_sha256": _sha(data), "stationarity_policy": frozen, "stationarity": report,
             "duration_statistic": STEP_PROFILE_DURATION_STATISTIC}
    proof["proof_sha256"] = _sha(proof)
    return calibrate_dataset(data, profile_version=profile_version, parent_profile=parent_profile,
                             allow_test_fixture=allow_test_fixture, _reported_step_proof=proof)


def validate_reported_step_profile(profile: dict, *, allow_test_fixture: bool = False) -> None:
    """Recheck raw provenance and the explicitly declared session-median target."""
    from .profiles import shape_key, validate_dataset

    proof = profile.get("notes", {}).get("reported_step_profile")
    required = {"schema_version", "kind", "raw_dataset", "raw_dataset_sha256", "stationarity_policy",
                "stationarity", "proof_sha256", "duration_statistic"}
    if (not isinstance(proof, dict) or set(proof) != required or type(proof["schema_version"]) is not int
            or proof["schema_version"] != 1 or proof["kind"] != STEP_PROFILE_PROOF_KIND):
        _error("Reported-step Profile requires its complete stationarity/raw-data proof")
    if _sha({key: value for key, value in proof.items() if key != "proof_sha256"}) != proof["proof_sha256"]:
        _error("Reported-step Profile proof checksum changed")
    if (proof["duration_statistic"] != STEP_PROFILE_DURATION_STATISTIC
            or profile["notes"].get("duration_statistic") != STEP_PROFILE_DURATION_STATISTIC
            or profile["notes"].get("fit") != STEP_PROFILE_FIT
            or profile["notes"].get("assumptions") != list(STEP_PROFILE_ASSUMPTIONS)):
        _error("Reported-step Profile must declare its session-median duration target")
    data = validate_dataset(proof["raw_dataset"], allow_test_fixture=allow_test_fixture)
    if (data != proof["raw_dataset"] or _sha(data) != proof["raw_dataset_sha256"]
            or data["identity"] != profile["identity"] or data["source"] != profile["source"]
            or profile["training"]["dataset_sha256"] != _sha(data)):
        _error("Reported-step Profile raw data, identity, source or training hash differs")
    report = _evaluate(data, proof["stationarity_policy"])
    if report != proof["stationarity"] or report["verdict"] != "PASS":
        _error("Reported-step Profile stationarity does not match the untrimmed raw observations")
    if (profile["training"]["record_ids"] != sorted(row["record_id"] for row in data["records"])
            or profile["training"]["group_ids"] != sorted({row["group_id"] for row in data["records"]})):
        _error("Reported-step Profile training members differ from the raw proof")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in data["records"]:
        grouped[shape_key(row["phase"], row["query_tokens"], row["context_lengths"], row["stage_index"])].append(row)
    samples = {shape_key(row["phase"], row["query_tokens"], row["context_lengths"], row["stage_index"]): row
               for row in profile["samples"]}
    if set(samples) != set(grouped):
        _error("Reported-step Profile sample shapes differ from raw observations")
    aggregates = reported_step_aggregates(data)
    for key, rows in grouped.items():
        sample = samples[key]
        if (sample["duration_ms"] != aggregates[key]["duration_ms"]
                or sample["sample_count"] != len(rows)
                or sample["workspace_peak_bytes"] != aggregates[key]["workspace_peak_bytes"]):
            _error("Reported-step Profile duration/count/Workspace differs from its raw records")


def guard_step_profile_domain(experiment: Experiment, profile: dict) -> None:
    """Do not let an unchanged opaque Runtime hash bypass wall-policy matching."""
    if profile["identity"]["runtime"]["timing_boundary"] != REPORTED_STEP_BOUNDARY:
        return
    if profile["source"]["kind"] == "test_fixture" and not experiment.calibration.allow_test_fixture:
        raise SimulationError("TEST_FIXTURE_NOT_ALLOWED", "Test fixtures require explicit allow_test_fixture")
    expected = step_wall_domain(experiment)
    actual = profile["notes"]["reported_step_profile"]["raw_dataset"]["notes"]["wall_domain"]
    if actual != expected:
        raise SimulationError("CALIBRATION_MISMATCH", "Reported-step wall identity or scheduling policy differs from the Profile")
