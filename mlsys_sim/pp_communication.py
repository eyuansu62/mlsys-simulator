"""Placement-specific standalone native PP primitive calibration.

Only explicitly selected native PP boundaries consume these fits. A primitive
sample is preallocated host-dispatch-through-device-completion service, not a
vLLM dependency wait, allocator cost or a global link-parameter replacement.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, NoReturn

from .communication_calibration import _fit_group
from .profiles import profile_identity, validate_identity
from .schema import Experiment, SimulationError


_METHOD = "nonnegative_affine_ols_v1"
_LOOKUP_METHOD = "piecewise_size_lookup_v1"
_MODEL_OPTIONS = {"min_samples_per_size": 5, "max_size_ratio": 4.0,
                  "jump_ratio": 2.0, "min_jump_ms": 0.05}
_PROTOCOL_V2 = "native_pp_primitives_v2"
_PROTOCOL_V3 = "native_pp_primitives_v3"
_COMMUNICATOR_INITIALIZATION = "lazy_world_and_pair_groups_v1"
_P2P_PROTOCOL_POLICY = "simple_only_thresholds_v1"
_P2P_PROTOCOL_ENVIRONMENT = {"NCCL_P2P_LL_THRESHOLD": "0", "NCCL_P2P_LL128_THRESHOLD": "0"}
_DATASET = "native_pp_communication_measurements"
_PROFILE = "native_pp_communication_profile"
_MAX_BYTES = 32 * 1024 * 1024
_MAX_RECORDS = 1000000
_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
_BENCHMARK = {
    "protocol": "native_pp_primitives_v1",
    "timing_boundary": "synchronized_host_primitive_completion_v1",
    "p2p_direction": "unidirectional",
    "allgather_bytes": "full_output",
    "rank_reduction": "maximum_local_duration",
    "barrier_in_timing": False,
    "warmup_in_records": False,
    "startup_in_timing": False,
    "allocation_in_timing": False,
    "buffer_policy": "preallocated_out_of_place",
    "backend": "nccl",
}
_APPLICATION = {
    "explicit_native_boundary_only": True,
    "no_local_stage_collectives": True,
    "no_global_infra_patch": True,
    "parent_parameters_reused": False,
}
_LIMITATIONS = [
    "Fits describe preallocated standalone host dispatch and primitive device completion at the recorded placement.",
    "They exclude vLLM metadata ingress, upstream waits, receiver output allocation and Runtime/framework contention.",
    "AllGather message_bytes names the full logical output; P2P message_bytes names one unidirectional wire payload.",
    "No Ring algorithm, symmetric/custom communicator or shared-fabric overlap equivalence is inferred.",
    "Apply only to explicitly selected native PP boundaries, never to local whole-stage collectives or global InfraSpec.",
    "Calibration and fixtures do not establish held-out AMD hardware validity.",
]


def _fail(message: str, code: str = "INVALID_PP_COMMUNICATION_DATASET") -> NoReturn:
    raise SimulationError(code, message)


def _object(value: Any, required: set[str], optional: set[str], label: str, code: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(f"{label} must be an object with string keys", code)
    if required - value.keys() or value.keys() - required - optional:
        _fail(f"{label} has missing or unknown fields", code)
    return value


def _text(value: Any, label: str, code: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\x00" in value:
        _fail(f"{label} must be a bounded nonempty string without NUL", code)
    return value


def _integer(value: Any, label: str, code: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"{label} must be a signed 64-bit integer >= {minimum}", code)
    return value


def _number(value: Any, label: str, code: str, *, zero: bool = False) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be finite", code)
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise SimulationError(code, f"{label} must be finite") from exc
    if not math.isfinite(number) or (number < 0 if zero else number <= 0):
        _fail(f"{label} must be finite and {'nonnegative' if zero else 'positive'}", code)
    return number


def _json(value: Any, code: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise SimulationError(code, "Native PP communication data must be finite JSON") from exc


def _source(value: Any, allow_test_fixture: bool, code: str) -> dict:
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be boolean", code)
    source = _object(value, {"kind", "reference"}, set(), "source", code)
    if source["kind"] not in ("measured", "test_fixture"):
        _fail("source.kind must be measured or test_fixture", code)
    _text(source["reference"], "source.reference", code)
    if source["kind"] == "test_fixture" and not allow_test_fixture:
        _fail("Native PP communication fixtures require explicit allow_test_fixture", "TEST_FIXTURE_NOT_ALLOWED")
    return dict(source)


def _identity(value: Any, source: dict, code: str) -> dict:
    try:
        identity = validate_identity(value, measured=source["kind"] == "measured")
    except SimulationError as exc:
        raise SimulationError(code, f"Invalid native PP communication identity: {exc}") from exc
    parallel = identity["parallelism"]
    if (parallel.get("replicas", 1) != 1 or parallel.get("pipeline_transport") != "vllm_native"
            or parallel["pipeline_parallel_size"] <= 1):
        _fail("Native PP communication requires one replica with the native transport identity", code)
    model = identity["model"]
    if _DTYPE_BYTES.get(model["activation_dtype"]) != model["activation_bytes"]:
        _fail("Native PP communication requires an explicit matching activation dtype", code)
    return identity


def _benchmark(value: Any, identity: dict, source: dict, code: str) -> dict:
    expected_fields: dict[str, Any] = dict(_BENCHMARK)
    if isinstance(value, dict) and value.get("protocol") in (_PROTOCOL_V2, _PROTOCOL_V3):
        expected_fields.update(protocol=value["protocol"],
                               communicator_initialization=_COMMUNICATOR_INITIALIZATION)
        if value["protocol"] == _PROTOCOL_V3:
            expected_fields.update(p2p_protocol_policy=_P2P_PROTOCOL_POLICY,
                                   p2p_protocol_environment=_P2P_PROTOCOL_ENVIRONMENT)
    value = _object(value, set(expected_fields) | {"stack", "launch"}, set(), "benchmark", code)
    for name, expected in expected_fields.items():
        actual = value[name]
        if type(actual) is not type(expected) or actual != expected:
            _fail(f"Unsupported benchmark.{name}", code)
    for name in ("stack", "launch"):
        if not isinstance(value[name], dict) or any(not isinstance(key, str) for key in value[name]):
            _fail(f"benchmark.{name} must be an object", code)
    if value["protocol"] == _PROTOCOL_V3:
        for label, environment in (("benchmark.p2p_protocol_environment", value["p2p_protocol_environment"]),
                                   ("benchmark.stack.p2p_protocol_environment",
                                    value["stack"].get("p2p_protocol_environment"))):
            environment = _object(environment, set(_P2P_PROTOCOL_ENVIRONMENT), set(), label, code)
            if any(type(environment[name]) is not str or environment[name] != expected
                   for name, expected in _P2P_PROTOCOL_ENVIRONMENT.items()):
                _fail(f"Unsupported {label}; both actual P2P thresholds must be the string '0'", code)
    _json(value, code)
    if source["kind"] == "measured":
        stack = value["stack"]
        for name in ("torch_version", "hip_version", "rccl_version"):
            text = _text(stack.get(name), f"benchmark.stack.{name}", code)
            if text.strip().lower() in {"unknown", "unspecified", "auto", "none", "not-applicable"}:
                _fail(f"Measured benchmark requires an actual {name}", code)
        if stack["rccl_version"] != identity["runtime"]["rccl_version"]:
            _fail("Loaded benchmark RCCL version differs from target identity", "CALIBRATION_MISMATCH")
        if not isinstance(stack.get("rccl_library_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", stack["rccl_library_sha256"]):
            _fail("Measured benchmark requires the loaded RCCL library SHA256", code)
        if not isinstance(stack.get("communication_environment_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", stack["communication_environment_sha256"]):
            _fail("Measured benchmark requires its communication-environment SHA256", code)
    return deepcopy(value)


def _benchmark_semantics(value: dict) -> str:
    # Launch paths/session IDs can change between independent measurements;
    # protocol, policy, stack and communication environment must remain exact.
    return _json({key: item for key, item in value.items() if key != "launch"}, "CALIBRATION_MISMATCH")


def _placement(identity: dict, primitive: Any, stage_index: Any, tp_rank: Any, participants: Any,
               code: str) -> tuple[str, int, int | None, list[int]]:
    if primitive not in ("p2p", "allgather"):
        _fail("primitive must be p2p or allgather", code)
    parallel = identity["parallelism"]
    tp, pp = parallel["tensor_parallel_size"], parallel["pipeline_parallel_size"]
    stage = _integer(stage_index, "stage_index", code, 1)
    if stage >= pp:
        _fail("stage_index must identify a receiving pipeline stage", code)
    rank = None
    if primitive == "p2p":
        rank = _integer(tp_rank, "tp_rank", code)
        if rank >= tp:
            _fail("tp_rank exceeds the stage tensor-parallel group", code)
        expected = [(stage - 1) * tp + rank, stage * tp + rank]
    else:
        if tp_rank is not None or tp <= 1:
            _fail("AllGather requires TP > 1 and tp_rank=null", code)
        expected = list(range(stage * tp, (stage + 1) * tp))
    if not isinstance(participants, list):
        _fail("participants must be an ordered rank array", code)
    actual = [_integer(value, "participants[]", code) for value in participants]
    if actual != expected:
        _fail("Primitive participants differ from the exact native PP placement", code)
    return primitive, stage, rank, expected


def _message_bytes(identity: dict, primitive: str, value: Any, code: str) -> int:
    size = _integer(value, "message_bytes", code, 1)
    width = _DTYPE_BYTES[identity["model"]["activation_dtype"]]
    divisor = width * (identity["parallelism"]["tensor_parallel_size"] if primitive == "allgather" else 1)
    if size % divisor:
        _fail("message_bytes must cover whole elements and full AllGather output partitions", code)
    return size


def _key(primitive: str, stage_index: int, tp_rank: int | None, participants: list[int]) -> str:
    return json.dumps([primitive, stage_index, tp_rank, participants], separators=(",", ":"))


def _record(value: Any, identity: dict, code: str) -> dict:
    row = _object(value, {"record_id", "group_id", "primitive", "stage_index", "tp_rank", "participants",
        "message_bytes", "duration_ms", "rank_durations_ms", "correctness_errors"}, set(), "record", code)
    primitive, stage, rank, participants = _placement(identity, row["primitive"], row["stage_index"],
                                                     row["tp_rank"], row["participants"], code)
    size = _message_bytes(identity, primitive, row["message_bytes"], code)
    duration = _number(row["duration_ms"], "duration_ms", code)
    if _integer(row["correctness_errors"], "correctness_errors", code) != 0:
        _fail("Primitive correctness errors must be zero", code)
    supplied = row["rank_durations_ms"]
    if not isinstance(supplied, list) or len(supplied) != len(participants):
        _fail("One local duration is required per participating rank", code)
    timings = {}
    for value in supplied:
        value = _object(value, {"rank", "duration_ms"}, set(), "rank duration", code)
        current = _integer(value["rank"], "rank duration rank", code)
        if current not in participants or current in timings:
            _fail("Rank durations have missing, duplicated or foreign participants", code)
        timings[current] = _number(value["duration_ms"], "rank duration", code)
    if set(timings) != set(participants) or duration != max(timings.values()):
        _fail("duration_ms must equal the maximum complete participating-rank duration", code)
    return {"record_id": _text(row["record_id"], "record_id", code),
        "group_id": _text(row["group_id"], "group_id", code), "primitive": primitive,
        "stage_index": stage, "tp_rank": rank, "participants": participants, "message_bytes": size,
        "duration_ms": duration, "rank_durations_ms": [{"rank": current, "duration_ms": timings[current]}
            for current in participants], "correctness_errors": 0}


def validate_pp_communication_dataset(data: dict, *, allow_test_fixture: bool = False) -> dict:
    """Validate exact placement, one-way bytes and complete local timing proof."""
    code = "INVALID_PP_COMMUNICATION_DATASET"
    value = _object(data, {"schema_version", "kind", "dataset_id", "identity", "source", "records", "benchmark"},
                    {"notes"}, "native PP communication dataset", code)
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != _DATASET:
        _fail("Unsupported native PP communication dataset schema", code)
    source = _source(value["source"], allow_test_fixture, code)
    identity = _identity(value["identity"], source, code)
    benchmark = _benchmark(value["benchmark"], identity, source, code)
    if not isinstance(value["records"], list) or not 0 < len(value["records"]) <= _MAX_RECORDS:
        _fail("records must be a bounded nonempty array", code)
    records = [_record(row, identity, code) for row in value["records"]]
    if len({row["record_id"] for row in records}) != len(records):
        _fail("Duplicate native PP communication record_id", code)
    result = {"schema_version": 1, "kind": _DATASET, "dataset_id": _text(value["dataset_id"], "dataset_id", code),
        "identity": identity, "source": source, "records": records, "benchmark": benchmark}
    if "notes" in value:
        if not isinstance(value["notes"], dict):
            _fail("notes must be an object", code)
        result["notes"] = json.loads(_json(value["notes"], code))
    return result


def _read(path: str | Path, code: str) -> tuple[dict, bytes]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("Duplicate JSON key in native PP communication input", code)
            result[key] = value
        return result
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            _fail("Native PP communication file exceeds its bounded read limit", code)
        data = json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda _: _fail("Nonfinite JSON in native PP communication input", code))
    except SimulationError:
        raise
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise SimulationError(code, f"Cannot read native PP communication input {path}: {exc}") from exc
    if not isinstance(data, dict):
        _fail("Native PP communication input must be an object", code)
    _json(data, code)
    return data, raw


def load_pp_communication_dataset(path: str | Path, *, allow_test_fixture: bool = False) -> dict:
    """Load bounded JSON measurements without executing archived programs."""
    value, _ = _read(path, "INVALID_PP_COMMUNICATION_DATASET")
    return validate_pp_communication_dataset(value, allow_test_fixture=allow_test_fixture)


def _strings(value: Any, label: str, code: str) -> list[str]:
    if not isinstance(value, list) or not value:
        _fail(f"{label} must be a nonempty string array", code)
    values = [_text(item, label, code) for item in value]
    if len(set(values)) != len(values):
        _fail(f"{label} contains duplicates", code)
    return values


def _model_options(value: Any, code: str) -> dict:
    value = _object(value, set(_MODEL_OPTIONS), set(), "model_options", code)
    options: dict[str, int | float] = {"min_samples_per_size": _integer(value["min_samples_per_size"],
                                                                     "min_samples_per_size", code, 1)}
    for name in ("max_size_ratio", "jump_ratio", "min_jump_ms"):
        options[name] = _number(value[name], name, code)
    if options != _MODEL_OPTIONS:
        _fail("Unsupported piecewise size lookup model options", code)
    return dict(_MODEL_OPTIONS)


def _lookup_statistics(samples: list[dict]) -> dict:
    durations = sorted(sample["duration_ms"] for sample in samples)

    def percentile(fraction: float) -> float:
        position = (len(durations) - 1) * fraction
        lower = int(position)
        weight = position - lower
        if weight == 0:
            return durations[lower]
        return (1 - weight) * durations[lower] + weight * durations[lower + 1]

    mean = statistics.mean(durations)
    return {"duration_ms": mean, "mean_ms": mean, "median_ms": percentile(0.5),
        "min_ms": durations[0], "max_ms": durations[-1], "p10_ms": percentile(0.1),
        "p90_ms": percentile(0.9), "stddev_ms": statistics.pstdev(durations),
        "sample_count": len(samples)}


def _lookup_interval(left: dict, right: dict) -> dict:
    size_ratio = right["message_bytes"] / left["message_bytes"]
    left_mean, right_mean = left["mean_ms"], right["mean_ms"]
    reason = None
    if size_ratio > _MODEL_OPTIONS["max_size_ratio"]:
        reason = "size_gap"
    elif abs(right_mean - left_mean) >= _MODEL_OPTIONS["min_jump_ms"]:
        if left_mean / right_mean > _MODEL_OPTIONS["jump_ratio"]:
            reason = "downward_jump"
        elif right_mean / left_mean / size_ratio > _MODEL_OPTIONS["jump_ratio"]:
            reason = "upward_jump"
    return {"left_bytes": left["message_bytes"], "right_bytes": right["message_bytes"],
            "supported": reason is None, "reason": reason}


def _lookup_group(entry: dict, identity: dict, primitive: str, code: str) -> dict:
    supplied = entry["knots"]
    if not isinstance(supplied, list) or not 2 <= len(supplied) <= _MAX_RECORDS:
        _fail("Lookup requires at least two bounded message-size knots", code)
    knots = []
    previous_size = 0
    record_ids: set[str] = set()
    group_ids: set[str] = set()
    errors: list[float] = []
    stat_fields = {"duration_ms", "mean_ms", "median_ms", "min_ms", "max_ms", "p10_ms", "p90_ms",
                   "stddev_ms", "sample_count"}
    for supplied_knot in supplied:
        knot = _object(supplied_knot, stat_fields | {"message_bytes", "record_ids", "group_ids", "samples"},
                       set(), "lookup knot", code)
        size = _message_bytes(identity, primitive, knot["message_bytes"], code)
        if size <= previous_size:
            _fail("Lookup knots must have strictly increasing unique message sizes", code)
        previous_size = size
        raw_samples = knot["samples"]
        if not isinstance(raw_samples, list) or not _MODEL_OPTIONS["min_samples_per_size"] <= len(raw_samples) <= _MAX_RECORDS:
            _fail("Each lookup knot requires at least five retained samples", code)
        samples = []
        for item in raw_samples:
            item = _object(item, {"record_id", "group_id", "duration_ms"}, set(), "lookup sample", code)
            samples.append({"record_id": _text(item["record_id"], "sample.record_id", code),
                "group_id": _text(item["group_id"], "sample.group_id", code),
                "duration_ms": _number(item["duration_ms"], "sample.duration_ms", code)})
        ids = _strings(knot["record_ids"], "knot.record_ids", code)
        sample_ids = [sample["record_id"] for sample in samples]
        current_groups = _strings(knot["group_ids"], "knot.group_ids", code)
        if (len(set(sample_ids)) != len(samples) or set(sample_ids) != set(ids)
                or record_ids.intersection(ids)
                or {sample["group_id"] for sample in samples} != set(current_groups)):
            _fail("Lookup knot sample provenance is inconsistent", code)
        record_ids.update(ids)
        group_ids.update(current_groups)
        stats = _lookup_statistics(samples)
        for name in stat_fields:
            supplied_stat = (_integer(knot[name], name, code, 5) if name == "sample_count"
                             else _number(knot[name], name, code, zero=name == "stddev_ms"))
            if supplied_stat != stats[name]:
                _fail(f"Lookup knot {name} differs from retained sample statistics", code)
        errors.extend(abs(sample["duration_ms"] - stats["mean_ms"]) for sample in samples)
        knots.append({"message_bytes": size, **stats, "record_ids": list(ids),
                      "group_ids": list(current_groups), "samples": samples})
    expected_intervals = [_lookup_interval(left, right) for left, right in zip(knots, knots[1:])]
    if _json(entry["intervals"], code) != _json(expected_intervals, code):
        _fail("Lookup intervals must exactly match the training-only gap and jump rules", code)
    expected_coverage = {"message_bytes_min": knots[0]["message_bytes"],
        "message_bytes_max": knots[-1]["message_bytes"], "record_count": len(record_ids),
        "distinct_message_sizes": len(knots)}
    if (_json(entry["coverage"], code) != _json(expected_coverage, code)
            or set(_strings(entry["record_ids"], "group.record_ids", code)) != record_ids
            or set(_strings(entry["group_ids"], "group.group_ids", code)) != group_ids):
        _fail("Lookup group coverage must equal all knot sample provenance", code)
    mae, maximum_error = statistics.mean(errors), max(errors)
    if (_number(entry["mae_ms"], "mae_ms", code, zero=True) != mae
            or _number(entry["max_absolute_error_ms"], "max_absolute_error_ms", code, zero=True) != maximum_error):
        _fail("Lookup residuals differ from the complete retained observations", code)
    return {"knots": knots, "intervals": expected_intervals, "coverage": expected_coverage,
            "mae_ms": mae, "max_absolute_error_ms": maximum_error,
            "record_ids": list(entry["record_ids"]), "group_ids": list(entry["group_ids"])}


def validate_pp_communication_profile(data: dict, *, allow_test_fixture: bool = False) -> dict:
    """Validate a standalone model, its supported domain and complete lineage."""
    code = "INVALID_PP_COMMUNICATION_PROFILE"
    value = _object(data, {"schema_version", "kind", "profile_id", "profile_version", "identity", "source",
        "benchmark", "method", "groups", "training", "application", "limitations"}, {"model_options"}, "native PP profile", code)
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != _PROFILE or value["method"] not in (_METHOD, _LOOKUP_METHOD)):
        _fail("Unsupported native PP communication profile schema or method", code)
    method = value["method"]
    if method == _LOOKUP_METHOD:
        _model_options(value.get("model_options"), code)
        if len(_json(value, code).encode()) > _MAX_BYTES:
            _fail("Native PP communication Profile exceeds its bounded read limit", code)
    elif "model_options" in value:
        _fail("Affine native PP profiles do not accept lookup model_options", code)
    source = _source(value["source"], allow_test_fixture, code)
    identity = _identity(value["identity"], source, code)
    benchmark = _benchmark(value["benchmark"], identity, source, code)
    application = _object(value["application"], set(_APPLICATION), set(), "application", code)
    if any(type(application[name]) is not bool or application[name] != expected
           for name, expected in _APPLICATION.items()):
        _fail("Native PP profile application policy is not supported", code)
    training = _object(value["training"], {"dataset_id", "dataset_sha256", "record_ids", "group_ids",
        "parent_profile_version"}, set(), "training", code)
    _text(training["dataset_id"], "training.dataset_id", code)
    if not isinstance(training["dataset_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", training["dataset_sha256"]):
        _fail("training.dataset_sha256 must be a lowercase SHA256", code)
    training_ids = _strings(training["record_ids"], "training.record_ids", code)
    training_groups = _strings(training["group_ids"], "training.group_ids", code)
    parent = training["parent_profile_version"]
    if parent is not None:
        _text(parent, "training.parent_profile_version", code)
    version = _text(value["profile_version"], "profile_version", code)
    if parent == version:
        _fail("Native PP profile parent must have a different version", code)
    supplied = value["groups"]
    if not isinstance(supplied, list) or not supplied or len(supplied) > _MAX_RECORDS:
        _fail("Profile groups must be a bounded nonempty array", code)
    groups = []
    keys = set()
    all_ids: set[str] = set()
    all_group_ids: set[str] = set()
    fields = {"primitive", "stage_index", "tp_rank", "participants", "intercept_ms", "slope_ms_per_byte",
        "intercept_clamped", "coverage", "mae_ms", "max_absolute_error_ms", "record_ids", "group_ids"}
    if method == _LOOKUP_METHOD:
        fields = fields - {"intercept_ms", "slope_ms_per_byte", "intercept_clamped"} | {"knots", "intervals"}
    for entry in supplied:
        entry = _object(entry, fields, set(), "native PP profile group", code)
        primitive, stage, rank, participants = _placement(identity, entry["primitive"], entry["stage_index"],
                                                         entry["tp_rank"], entry["participants"], code)
        key = _key(primitive, stage, rank, participants)
        if key in keys:
            _fail("Duplicate primitive/placement profile group", code)
        keys.add(key)
        if method == _LOOKUP_METHOD:
            lookup = _lookup_group(entry, identity, primitive, code)
            if all_ids.intersection(lookup["record_ids"]):
                _fail("Profile groups cannot share training records", code)
            all_ids.update(lookup["record_ids"])
            all_group_ids.update(lookup["group_ids"])
            groups.append({"primitive": primitive, "stage_index": stage, "tp_rank": rank,
                           "participants": participants, **lookup})
            continue
        intercept = _number(entry["intercept_ms"], "intercept_ms", code, zero=True)
        slope = _number(entry["slope_ms_per_byte"], "slope_ms_per_byte", code)
        if type(entry["intercept_clamped"]) is not bool or entry["intercept_clamped"] and intercept != 0:
            _fail("Invalid nonnegative-intercept boundary flag", code)
        coverage = _object(entry["coverage"], {"message_bytes_min", "message_bytes_max", "record_count",
            "distinct_message_sizes"}, set(), "coverage", code)
        minimum = _message_bytes(identity, primitive, coverage["message_bytes_min"], code)
        maximum = _message_bytes(identity, primitive, coverage["message_bytes_max"], code)
        count = _integer(coverage["record_count"], "coverage.record_count", code, 2)
        distinct = _integer(coverage["distinct_message_sizes"], "coverage.distinct_message_sizes", code, 2)
        if minimum >= maximum or distinct > count:
            _fail("Profile coverage requires at least two distinct message sizes", code)
        for size in (minimum, maximum):
            _number(intercept + slope * size, "fitted coverage endpoint duration", code)
        ids = _strings(entry["record_ids"], "group.record_ids", code)
        group_ids = _strings(entry["group_ids"], "group.group_ids", code)
        if count != len(ids) or all_ids.intersection(ids) or len(group_ids) > count:
            _fail("Profile group record coverage is inconsistent", code)
        all_ids.update(ids)
        all_group_ids.update(group_ids)
        mae = _number(entry["mae_ms"], "mae_ms", code, zero=True)
        maximum_error = _number(entry["max_absolute_error_ms"], "max_absolute_error_ms", code, zero=True)
        if maximum_error < mae:
            _fail("Maximum absolute error cannot be smaller than MAE", code)
        groups.append({"primitive": primitive, "stage_index": stage, "tp_rank": rank,
            "participants": participants, "intercept_ms": intercept, "slope_ms_per_byte": slope,
            "intercept_clamped": entry["intercept_clamped"],
            "coverage": {"message_bytes_min": minimum, "message_bytes_max": maximum,
                         "record_count": count, "distinct_message_sizes": distinct},
            "mae_ms": mae, "max_absolute_error_ms": maximum_error,
            "record_ids": list(ids), "group_ids": list(group_ids)})
    if all_ids != set(training_ids) or all_group_ids != set(training_groups):
        _fail("Training provenance must exactly equal all primitive/placement groups", code)
    result = {"schema_version": 1, "kind": _PROFILE, "method": method,
        "profile_id": _text(value["profile_id"], "profile_id", code), "profile_version": version,
        "identity": identity, "source": source, "benchmark": benchmark, "groups": groups,
        "training": {"dataset_id": training["dataset_id"], "dataset_sha256": training["dataset_sha256"],
                     "record_ids": list(training_ids), "group_ids": list(training_groups),
                     "parent_profile_version": parent}, "application": dict(_APPLICATION),
        "limitations": _strings(value["limitations"], "limitations", code)}
    if method == _LOOKUP_METHOD:
        result["model_options"] = dict(_MODEL_OPTIONS)
        # The CLI writes this canonical pretty representation. Refuse a model
        # that its bounded loader could not read back, including repeated IDs.
        if len((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()) > _MAX_BYTES:
            _fail("Serialized native PP communication Profile exceeds its bounded read limit", code)
    return result


def load_pp_communication_profile(path: str | Path, *, allow_test_fixture: bool = False) -> dict:
    """Read a native communication Profile, rejecting legacy global patches."""
    value, _ = _read(path, "INVALID_PP_COMMUNICATION_PROFILE")
    return validate_pp_communication_profile(value, allow_test_fixture=allow_test_fixture)


def _fit_lookup(rows: list[dict]) -> dict:
    sizes: dict[int, list[dict]] = {}
    for row in rows:
        sizes.setdefault(row["message_bytes"], []).append(row)
    if len(sizes) < 2:
        _fail("Lookup calibration requires at least two distinct message sizes", "INVALID_COMMUNICATION_FIT")
    knots = []
    errors: list[float] = []
    for size, observations in sorted(sizes.items()):
        if len(observations) < _MODEL_OPTIONS["min_samples_per_size"]:
            _fail("Lookup calibration requires at least five samples per message size", "INVALID_COMMUNICATION_FIT")
        samples = [{"record_id": row["record_id"], "group_id": row["group_id"], "duration_ms": row["duration_ms"]}
                   for row in sorted(observations, key=lambda row: row["record_id"])]
        stats = _lookup_statistics(samples)
        knots.append({"message_bytes": size, **stats, "samples": samples,
                      "record_ids": [sample["record_id"] for sample in samples],
                      "group_ids": sorted({sample["group_id"] for sample in samples})})
        errors.extend(abs(sample["duration_ms"] - stats["mean_ms"]) for sample in samples)
    first = rows[0]
    return {"primitive": first["primitive"], "stage_index": first["stage_index"],
        "tp_rank": first["tp_rank"], "participants": list(first["participants"]),
        "knots": knots, "intervals": [_lookup_interval(left, right) for left, right in zip(knots, knots[1:])],
        "coverage": {"message_bytes_min": knots[0]["message_bytes"], "message_bytes_max": knots[-1]["message_bytes"],
                     "record_count": len(rows), "distinct_message_sizes": len(knots)},
        "mae_ms": statistics.mean(errors), "max_absolute_error_ms": max(errors),
        "record_ids": sorted(row["record_id"] for row in rows),
        "group_ids": sorted({row["group_id"] for row in rows})}


def calibrate_pp_communication(data: dict, *, profile_version: str, parent_profile: dict | None = None,
                               allow_test_fixture: bool = False, method: str = _METHOD) -> dict:
    """Fit each placement independently; lookup targets the untrimmed arithmetic mean.

    The opt-in lookup retains every sample, uses population standard deviation
    and linear order-statistic quantiles, and refuses extrapolation and guarded
    gaps. Its gap/jump policy is fixed before holdout evaluation.
    """
    if method not in (_METHOD, _LOOKUP_METHOD):
        _fail("Unsupported native PP communication calibration method", "INVALID_CONFIG")
    dataset = validate_pp_communication_dataset(data, allow_test_fixture=allow_test_fixture)
    version = _text(profile_version, "profile_version", "INVALID_CONFIG")
    previous_version = None
    source = deepcopy(dataset["source"])
    if parent_profile is not None:
        parent = validate_pp_communication_profile(parent_profile, allow_test_fixture=allow_test_fixture)
        if parent["identity"] != dataset["identity"]:
            _fail("Parent native PP profile identity differs", "CALIBRATION_MISMATCH")
        if _benchmark_semantics(parent["benchmark"]) != _benchmark_semantics(dataset["benchmark"]):
            _fail("Parent native PP profile benchmark semantics differ", "CALIBRATION_MISMATCH")
        previous_version = parent["profile_version"]
        if previous_version == version:
            _fail("Native PP calibration requires a new profile version", "PROFILE_VERSION_CONFLICT")
        if parent["source"]["kind"] == "test_fixture":
            source["kind"] = "test_fixture"
    grouped: dict[str, list[dict]] = {}
    for row in dataset["records"]:
        grouped.setdefault(_key(row["primitive"], row["stage_index"], row["tp_rank"], row["participants"]), []).append(row)
    fits = []
    for _, rows in sorted(grouped.items()):
        if method == _LOOKUP_METHOD:
            fits.append(_fit_lookup(rows))
            continue
        # Reuse the stable normalized affine solver only. Its temporary p2p/ranks
        # fields select T=a+b*bytes; no P2P semantic metadata escapes this step.
        mathematical_rows = [{"record_id": row["record_id"], "group_id": row["group_id"],
            "primitive": "p2p", "ranks": 2, "message_bytes": row["message_bytes"], "duration_ms": row["duration_ms"]}
            for row in sorted(rows, key=lambda row: row["record_id"])]
        try:
            fit = _fit_group(mathematical_rows)
        except SimulationError as exc:
            raise SimulationError(exc.code, f"Native PP primitive fit: {exc}") from exc
        first = rows[0]
        fits.append({"primitive": first["primitive"], "stage_index": first["stage_index"],
            "tp_rank": first["tp_rank"], "participants": list(first["participants"]),
            "intercept_ms": fit["intercept_ms"], "slope_ms_per_byte": fit["fit_slope_ms_per_byte"],
            "intercept_clamped": fit["intercept_clamped"], "coverage": dict(fit["coverage"]),
            "mae_ms": fit["mae_ms"], "max_absolute_error_ms": fit["max_absolute_error_ms"],
            "record_ids": list(fit["record_ids"]), "group_ids": list(fit["group_ids"])})
    code = "INVALID_PP_COMMUNICATION_DATASET"
    result = {"schema_version": 1, "kind": _PROFILE, "profile_id": f"{dataset['dataset_id']}:{version}",
        "profile_version": version, "identity": deepcopy(dataset["identity"]), "source": source,
        "benchmark": deepcopy(dataset["benchmark"]), "method": method, "groups": fits,
        "training": {"dataset_id": dataset["dataset_id"],
            "dataset_sha256": hashlib.sha256(_json(data, code).encode()).hexdigest(),
            "record_ids": sorted(row["record_id"] for row in dataset["records"]),
            "group_ids": sorted({row["group_id"] for row in dataset["records"]}),
            "parent_profile_version": previous_version}, "application": dict(_APPLICATION),
        "limitations": list(_LIMITATIONS)}
    if method == _LOOKUP_METHOD:
        result["model_options"] = dict(_MODEL_OPTIONS)
        result["limitations"].extend([
            "Lookup predicts the arithmetic mean of all retained repetitions; slow samples are not trimmed.",
            "Lookup interpolates in bytes only across adjacent knots passing fixed training-only size-gap and jump guards.",
            "Exact knots remain usable across blocked intervals; lookup never extrapolates, even when affine extrapolation is enabled.",
            "Knot p10/p90 use linear order-statistic interpolation and stddev_ms is the population standard deviation.",
        ])
    return validate_pp_communication_profile(result, allow_test_fixture=allow_test_fixture)


def _estimate(profile: dict, primitive: str, message_bytes: int, stage_index: int,
              tp_rank: int | None, *, allow_extrapolation: bool) -> dict:
    code = "OUT_OF_COMMUNICATION_CALIBRATION_DOMAIN"
    identity = profile["identity"]
    tp = identity["parallelism"]["tensor_parallel_size"]
    if primitive not in ("p2p", "allgather"):
        _fail("primitive must be p2p or allgather", code)
    stage_index = _integer(stage_index, "stage_index", code, 1)
    if primitive == "p2p":
        tp_rank = _integer(tp_rank, "tp_rank", code)
        participants = [(stage_index - 1) * tp + tp_rank, stage_index * tp + tp_rank]
    else:
        participants = list(range(stage_index * tp, (stage_index + 1) * tp))
    primitive, stage, rank, participants = _placement(identity, primitive, stage_index, tp_rank, participants, code)
    size = _message_bytes(identity, primitive, message_bytes, code)
    key = _key(primitive, stage, rank, participants)
    candidates = [group for group in profile["groups"] if _key(group["primitive"], group["stage_index"],
                  group["tp_rank"], group["participants"]) == key]
    if len(candidates) != 1:
        _fail("No standalone native PP calibration covers this exact primitive placement", code)
    group = candidates[0]
    coverage = group["coverage"]
    outside = not coverage["message_bytes_min"] <= size <= coverage["message_bytes_max"]
    source = ("measured_native_pp_communication_profile" if profile["source"]["kind"] == "measured"
              else "test_fixture_native_pp_communication_profile")
    if profile["method"] == _LOOKUP_METHOD:
        if outside:
            _fail("Piecewise size lookup never extrapolates outside its observed knots", code)
        exact = next((knot for knot in group["knots"] if knot["message_bytes"] == size), None)
        interval = None
        if exact is not None:
            contributors = [exact]
            duration = exact["mean_ms"]
        else:
            edges = [(left, right, edge) for left, right, edge in
                     zip(group["knots"], group["knots"][1:], group["intervals"])
                     if left["message_bytes"] < size < right["message_bytes"]]
            left, right, interval = edges[0]
            if not interval["supported"]:
                _fail(f"Piecewise size lookup refuses the training {interval['reason']} interval", code)
            weight = (size - left["message_bytes"]) / (right["message_bytes"] - left["message_bytes"])
            duration = (1 - weight) * left["mean_ms"] + weight * right["mean_ms"]
            contributors = [left, right]
        return {"duration_ms": _number(duration, "primitive prediction", "INVALID_COMMUNICATION_FIT"),
            "source": source, "extrapolated": False, "primitive": primitive, "stage_index": stage,
            "tp_rank": rank, "participants": participants, "message_bytes": size, "coverage": deepcopy(coverage),
            "model_method": _LOOKUP_METHOD, "exact": exact is not None,
            "interpolation": "exact" if exact is not None else "linear_bytes", "interval": deepcopy(interval),
            "contributors": [{name: knot[name] for name in ("message_bytes", "mean_ms", "sample_count")}
                             for knot in contributors]}
    if outside and not allow_extrapolation:
        _fail("Message size is outside the calibrated native PP primitive domain", code)
    duration = _number(group["intercept_ms"] + group["slope_ms_per_byte"] * size,
                       "primitive prediction", "INVALID_COMMUNICATION_FIT")
    return {"duration_ms": duration, "source": source, "extrapolated": outside,
        "primitive": primitive, "stage_index": stage, "tp_rank": rank,
        "participants": participants, "message_bytes": size, "coverage": deepcopy(coverage),
        "intercept_ms": group["intercept_ms"], "slope_ms_per_byte": group["slope_ms_per_byte"]}


class PPCommunicationModel:
    """Immutable loaded native boundary fit with conservative size-domain lookup."""

    def __init__(self, experiment: Experiment, profile: dict | None = None):
        if experiment.strategy.replicas != 1:
            _fail("Native PP communication models require one local replica", "INVALID_CONFIG")
        allow_fixture = experiment.calibration.allow_test_fixture
        if type(allow_fixture) is not bool:
            _fail("allow_test_fixture must be boolean", "INVALID_CONFIG")
        path = getattr(experiment.calibration, "native_pp_communication_profile_path", None)
        if profile is None:
            if not isinstance(path, str) or not path.strip():
                _fail("native_pp_communication_profile_path is required", "INVALID_CONFIG")
            payload, raw = _read(path, "INVALID_PP_COMMUNICATION_PROFILE")
            hash_scope = "file_bytes"
        else:
            payload = deepcopy(profile)
            raw = _json(payload, "INVALID_PP_COMMUNICATION_PROFILE").encode()
            hash_scope = "canonical_json"
        current = validate_pp_communication_profile(payload, allow_test_fixture=allow_fixture)
        if current["identity"] != profile_identity(experiment):
            _fail("Native PP communication Profile identity differs from its local experiment", "CALIBRATION_MISMATCH")
        requested = getattr(experiment.calibration, "native_pp_communication_profile_version", "")
        if requested and requested != current["profile_version"]:
            _fail("Native PP communication Profile version differs", "CALIBRATION_MISMATCH")
        self._allow_extrapolation = getattr(experiment.calibration, "native_pp_communication_allow_extrapolation", False)
        if type(self._allow_extrapolation) is not bool:
            _fail("native_pp_communication_allow_extrapolation must be boolean", "INVALID_CONFIG")
        self._profile = current
        self.provenance = {"profile_id": current["profile_id"], "profile_version": current["profile_version"],
            "profile_sha256": hashlib.sha256(raw).hexdigest(), "sha256_scope": hash_scope,
            "source_kind": current["source"]["kind"], "source_reference": current["source"]["reference"],
            "protocol": current["benchmark"]["protocol"], "timing_boundary": current["benchmark"]["timing_boundary"],
            "application": dict(_APPLICATION), "benchmark": deepcopy(current["benchmark"])}

    def estimate(self, primitive: str, message_bytes: int, stage_index: int, *, tp_rank: int | None = None) -> dict:
        result = _estimate(self._profile, primitive, message_bytes, stage_index, tp_rank,
                           allow_extrapolation=self._allow_extrapolation)
        # Events retain the immutable content/source anchor, while the complete
        # benchmark launch and stack appear once in the run-level provenance.
        result["provenance"] = {name: self.provenance[name] for name in (
            "profile_id", "profile_version", "profile_sha256", "source_kind", "source_reference",
            "protocol", "timing_boundary")}
        return result


def _validation_summary(rows: list[dict]) -> dict:
    evaluated = [row for row in rows if row["status"] == "EVALUATED"]
    errors = [row["absolute_error_ms"] for row in evaluated]
    relatives = [row["relative_error"] for row in evaluated if row["relative_error"] is not None]
    return {"coverage": {"total_records": len(rows), "evaluated_records": len(evaluated),
                         "rejected_records": len(rows) - len(evaluated),
                         "fraction": len(evaluated) / len(rows) if rows else None},
            "metrics": {"mae_ms": statistics.mean(errors) if errors else None,
                        "max_absolute_error_ms": max(errors) if errors else None,
                        "mape": statistics.mean(relatives) if relatives and len(relatives) == len(errors) else None}}


def _observed_distribution(rows: list[dict]) -> dict:
    # Reuse the lookup's untrimmed mean, linear quantiles and population stddev.
    # This describes observations even when their predictions were rejected.
    durations = sorted(row["observed_ms"] for row in rows)
    stats = _lookup_statistics([{"duration_ms": duration} for duration in durations])
    position = (len(durations) - 1) * 0.99
    lower = int(position)
    weight = position - lower
    p99 = ((1 - weight) * durations[lower] + weight * durations[lower + 1]
           if weight else durations[lower])
    return {"count": len(durations), "mean": stats["mean_ms"], "min": stats["min_ms"],
        "max": stats["max_ms"], "p10": stats["p10_ms"], "p50": stats["median_ms"],
        "p90": stats["p90_ms"], "p99": p99, "stddev": stats["stddev_ms"]}


def _validation_sizes(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["primitive"], row["stage_index"], row["tp_rank"], row["message_bytes"]), []).append(row)
    summaries = []
    for (primitive, stage, rank, size), observations in sorted(groups.items()):
        summary = _validation_summary(observations)
        observed = statistics.mean(row["observed_ms"] for row in observations)
        predicted = (statistics.mean(row["predicted_ms"] for row in observations)
                     if summary["coverage"]["rejected_records"] == 0 else None)
        absolute = abs(predicted - observed) if predicted is not None else None
        relative = absolute / observed if absolute is not None else None
        summaries.append({"primitive": primitive, "stage_index": stage, "tp_rank": rank, "message_bytes": size,
            "training_size_relation": observations[0]["training_size_relation"], **summary,
            "observed_mean_ms": observed, "predicted_mean_ms": predicted,
            "observed_distribution_ms": _observed_distribution(observations),
            "mean_target_absolute_error_ms": absolute,
            "mean_target_relative_error": relative if relative is not None and math.isfinite(relative) else None,
            "sample_mape": summary["metrics"]["mape"]})
    return summaries


def _mean_target_summary(sizes: list[dict]) -> dict:
    evaluated = [row for row in sizes if row["mean_target_absolute_error_ms"] is not None]
    relatives = [row["mean_target_relative_error"] for row in evaluated if row["mean_target_relative_error"] is not None]
    return {"total_sizes": len(sizes), "evaluated_sizes": len(evaluated),
        "rejected_sizes": len(sizes) - len(evaluated),
        "mae_ms": statistics.mean(row["mean_target_absolute_error_ms"] for row in evaluated) if evaluated else None,
        "mape": statistics.mean(relatives) if relatives and len(relatives) == len(evaluated) else None}


def validate_pp_communication_holdout(profile: dict, dataset: dict, *, allow_test_fixture: bool = False,
                                      allow_extrapolation: bool = False) -> dict:
    """Evaluate separate primitive observations without granting hardware PASS."""
    if type(allow_extrapolation) is not bool:
        _fail("allow_extrapolation must be boolean", "INVALID_CONFIG")
    current = validate_pp_communication_profile(profile, allow_test_fixture=allow_test_fixture)
    data = validate_pp_communication_dataset(dataset, allow_test_fixture=allow_test_fixture)
    if current["identity"] != data["identity"]:
        _fail("Native PP holdout identity differs from training", "CALIBRATION_MISMATCH")
    if _benchmark_semantics(current["benchmark"]) != _benchmark_semantics(data["benchmark"]):
        _fail("Native PP holdout benchmark semantics differ from training", "CALIBRATION_MISMATCH")
    leaked_ids = set(current["training"]["record_ids"]) & {row["record_id"] for row in data["records"]}
    leaked_groups = set(current["training"]["group_ids"]) & {row["group_id"] for row in data["records"]}
    if leaked_ids or leaked_groups:
        _fail("Native PP holdout shares training record_id or group_id", "CALIBRATION_LEAKAGE")
    fixture = current["source"]["kind"] == "test_fixture" or data["source"]["kind"] == "test_fixture"
    training_sizes = {_key(group["primitive"], group["stage_index"], group["tp_rank"], group["participants"]):
                      {knot["message_bytes"] for knot in group["knots"]}
                      for group in current["groups"] if "knots" in group}
    rows = []
    errors = []
    relatives = []
    for row in data["records"]:
        result = {"record_id": row["record_id"], "group_id": row["group_id"], "primitive": row["primitive"],
            "stage_index": row["stage_index"], "tp_rank": row["tp_rank"], "message_bytes": row["message_bytes"],
            "observed_ms": row["duration_ms"]}
        sizes = training_sizes.get(_key(row["primitive"], row["stage_index"], row["tp_rank"], row["participants"]))
        result["training_size_relation"] = ("exact" if row["message_bytes"] in sizes else "unseen") if sizes is not None else "unknown"
        try:
            estimate = _estimate(current, row["primitive"], row["message_bytes"], row["stage_index"],
                                 row["tp_rank"], allow_extrapolation=allow_extrapolation)
        except SimulationError as exc:
            if exc.code != "OUT_OF_COMMUNICATION_CALIBRATION_DOMAIN":
                raise
            result.update(status="OUT_OF_DOMAIN", reason=str(exc), predicted_ms=None,
                          absolute_error_ms=None, relative_error=None, extrapolated=False)
        else:
            absolute = abs(estimate["duration_ms"] - row["duration_ms"])
            relative = absolute / row["duration_ms"]
            errors.append(absolute)
            relatives.append(relative if math.isfinite(relative) else None)
            result.update(status="EVALUATED", predicted_ms=estimate["duration_ms"], absolute_error_ms=absolute,
                          relative_error=relative if math.isfinite(relative) else None,
                          extrapolated=estimate["extrapolated"])
            if current["method"] == _LOOKUP_METHOD:
                result.update(model_method=estimate["model_method"], exact=estimate["exact"],
                              interpolation=estimate["interpolation"], contributors=estimate["contributors"])
        rows.append(result)
    relative_values = [value for value in relatives if value is not None]
    by_size = _validation_sizes(rows)
    small_rows = [row for row in rows if row["message_bytes"] <= 65536]
    small_sizes = [row for row in by_size if row["message_bytes"] <= 65536]
    return {"schema_version": 1, "kind": "native_pp_communication_validation", "status": "COMPLETED",
        "profile_id": current["profile_id"], "profile_version": current["profile_version"],
        "dataset_id": data["dataset_id"], "source": {"kind": "test_fixture" if fixture else "measured",
            "reference": "Independent standalone native PP communication holdout evaluation"},
        "coverage": {"total_records": len(rows), "evaluated_records": len(errors),
            "rejected_records": len(rows) - len(errors), "fraction": len(errors) / len(rows)},
        "metrics": {"mae_ms": statistics.mean(errors) if errors else None,
            "max_absolute_error_ms": max(errors) if errors else None,
            "mape": statistics.mean(relative_values) if relative_values and len(relative_values) == len(errors) else None},
        "model_method": current["method"], "by_message_size": by_size,
        "mean_target_metrics": _mean_target_summary(by_size),
        "exact_vs_unseen": {"classification_basis": "retained_lookup_knots" if training_sizes else "unavailable_for_legacy_affine",
            **{name: _validation_summary([row for row in rows if row["training_size_relation"] == name])
               for name in ("exact", "unseen", "unknown")}},
        "small_messages": {"threshold_bytes": 65536, **_validation_summary(small_rows),
                           "mean_target_metrics": _mean_target_summary(small_sizes)},
        "records": rows, "hardware_validated": False,
        "notes": {"record_and_group_leakage_rejected": True, "training_parameters_refit": False,
                  "rate_or_runtime_accuracy_inferred": False,
                  "mean_target_requires_complete_size_coverage": True,
                  "rejected_records_are_not_zero_error": True}}
