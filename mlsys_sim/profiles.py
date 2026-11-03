"""Strict measured-profile contracts and conservative, explicit-domain lookup.

Profiles contain whole-stage time, including its collectives and output work.
They do not contain inter-stage PP transfers. No macro attribution is inferred
from a measured total, and no nearest-neighbour fallback is performed.
"""

from bisect import bisect_left
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING

from .layout import stage_layer_counts
from .schema import Experiment, SimulationError, StepEstimate

if TYPE_CHECKING:
    from .performance import AnalyticalModel


def _invalid(message: str, code: str = "INVALID_PROFILE") -> None:
    raise SimulationError(code, message)


def _object(value: Any, required: set[str], optional: set[str], name: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _invalid(f"{name} must be an object with string keys")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        _invalid(f"{name}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(f"{name} must be a nonempty string")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _invalid(f"{name} must be a signed 64-bit integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, zero: bool = False) -> float:
    if type(value) not in (int, float):
        _invalid(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        _invalid(f"{name} must be a finite number")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        _invalid(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        _invalid(f"{name} must be a nonempty array of strings")
    result = [_text(item, name) for item in value]
    if len(set(result)) != len(result):
        _invalid(f"{name} contains duplicates")
    return result


def _json_value(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError) as exc:
        raise SimulationError("INVALID_PROFILE", f"{name} must be finite JSON data") from exc


def profile_identity(experiment: Experiment) -> dict:
    """Describe the exact model, software, physical layout and timing contract."""
    strategy, infra, calibration = experiment.strategy, experiment.infra, experiment.calibration
    memory = strategy.memory
    return {
        "model": asdict(strategy.model),
        "devices": [asdict(device) for device in infra.devices],
        "parallelism": {
            "tensor_parallel_size": strategy.tensor_parallel_size,
            "pipeline_parallel_size": strategy.pipeline_parallel_size,
            "stage_layers": list(stage_layer_counts(experiment)),
            "shard_plan": "megatron_dense_no_sp",
        },
        "topology": {"name": infra.topology, "link_bandwidth_gbps": infra.link_bandwidth_gbps,
                     "link_latency_us": infra.link_latency_us},
        "runtime": {"name": calibration.runtime_name, "version": calibration.runtime_version,
                    "driver_version": calibration.driver_version, "rocm_version": calibration.rocm_version,
                    "rccl_version": calibration.rccl_version, "timing_boundary": calibration.timing_boundary},
        "execution": {"kv_block_tokens": memory.kv_block_tokens, "graph_bytes": memory.graph_bytes,
                      "runtime_bytes": memory.runtime_bytes,
                      "communication_buffer_bytes": memory.communication_buffer_bytes},
    }


def _identity(value: Any) -> dict:
    """Validate a portable identity with the same model constraints as experiments."""
    from .config import experiment_from_dict

    identity = _object(value, {"model", "devices", "parallelism", "topology", "runtime", "execution"},
                       set(), "identity")
    parallel = _object(identity["parallelism"],
                       {"tensor_parallel_size", "pipeline_parallel_size", "stage_layers", "shard_plan"},
                       set(), "identity.parallelism")
    if parallel["shard_plan"] != "megatron_dense_no_sp":
        _invalid("identity has an unsupported shard plan")
    topology = _object(identity["topology"], {"name", "link_bandwidth_gbps", "link_latency_us"},
                       set(), "identity.topology")
    runtime = _object(identity["runtime"],
                      {"name", "version", "driver_version", "rocm_version", "rccl_version", "timing_boundary"},
                      set(), "identity.runtime")
    for key, item in runtime.items():
        _text(item, f"identity.runtime.{key}")
    _text(topology["name"], "identity.topology.name")
    execution = _object(identity["execution"],
                        {"kv_block_tokens", "graph_bytes", "runtime_bytes", "communication_buffer_bytes"},
                        set(), "identity.execution")
    # Explicitly require full dataclass identity, not a default-filled partial one.
    from .schema import DeviceSpec, ModelSpec
    from dataclasses import fields
    _object(identity["model"], {field.name for field in fields(ModelSpec)}, set(), "identity.model")
    if not isinstance(identity["devices"], list) or not identity["devices"]:
        _invalid("identity.devices must be a nonempty array")
    for device in identity["devices"]:
        _object(device, {field.name for field in fields(DeviceSpec)}, set(), "identity.devices[]")
        _text(device["name"], "identity.devices[].name")
    _text(identity["model"]["revision"], "identity.model.revision")
    try:
        experiment = experiment_from_dict({
            "strategy": {"model": identity["model"],
                         "tensor_parallel_size": parallel["tensor_parallel_size"],
                         "pipeline_parallel_size": parallel["pipeline_parallel_size"],
                         "stage_layers": parallel["stage_layers"], "memory": execution},
            "infra": {"devices": identity["devices"], "topology": topology["name"],
                      "link_bandwidth_gbps": topology["link_bandwidth_gbps"],
                      "link_latency_us": topology["link_latency_us"]},
            "calibration": {"runtime_name": runtime["name"], "runtime_version": runtime["version"],
                            "driver_version": runtime["driver_version"], "rocm_version": runtime["rocm_version"],
                            "rccl_version": runtime["rccl_version"], "timing_boundary": runtime["timing_boundary"]},
        })
    except SimulationError as exc:
        raise SimulationError("INVALID_PROFILE", f"Invalid identity: {exc}") from exc
    return profile_identity(experiment)


def validate_identity(value: Any, *, measured: bool = False) -> dict:
    """Normalize a full identity; measured evidence needs explicit versions.

    Optional stack components may explicitly say ``not-applicable`` (for
    example RCCL on a non-AMD stack). Model/device/runtime identity cannot use
    that marker or default placeholders in a measured observation.
    """
    if type(measured) is not bool:
        _invalid("measured must be a boolean")
    identity = _identity(value)
    if not measured:
        return identity
    required = {
        "model.revision": identity["model"]["revision"],
        "model.weight_dtype": identity["model"]["weight_dtype"],
        "model.kv_dtype": identity["model"]["kv_dtype"],
        "model.activation_dtype": identity["model"]["activation_dtype"],
        "runtime.name": identity["runtime"]["name"],
        "runtime.version": identity["runtime"]["version"],
    }
    required.update({f"devices[{index}].name": device["name"]
                     for index, device in enumerate(identity["devices"])})
    optional_stack = {f"runtime.{name}": identity["runtime"][name]
                      for name in ("driver_version", "rocm_version", "rccl_version")}
    for name, item in {**required, **optional_stack}.items():
        normalized = item.strip().lower()
        if (normalized in {"unspecified", "unknown", "default", "toy", "auto", "none", "n/a", "na"}
                or normalized.startswith(("toy-", "toy_"))
                or (name in required and normalized == "not-applicable")):
            _invalid(f"Measured identity requires explicit non-placeholder {name}; "
                     "mark inapplicable stack components as not-applicable")
    return identity


def _source(value: Any, allow_test_fixture: bool) -> dict:
    if type(allow_test_fixture) is not bool:
        _invalid("allow_test_fixture must be a boolean")
    source = _object(value, {"kind", "reference"}, set(), "source")
    _text(source["reference"], "source.reference")
    if source["kind"] not in ("measured", "test_fixture"):
        _invalid("source.kind must be measured or test_fixture; simulated data is not measurement")
    if source["kind"] == "test_fixture" and not allow_test_fixture:
        raise SimulationError("TEST_FIXTURE_NOT_ALLOWED", "Test fixtures require explicit allow_test_fixture")
    return dict(source)


def _shape(phase: Any, query_tokens: Any, context_lengths: Any, stage_index: Any) -> tuple:
    if phase not in ("prefill", "decode", "recompute"):
        _invalid("phase must be prefill, decode or recompute")
    stage = _integer(stage_index, "stage_index")
    if (not isinstance(query_tokens, (list, tuple)) or not query_tokens
            or not isinstance(context_lengths, (list, tuple)) or len(query_tokens) != len(context_lengths)):
        _invalid("query_tokens/context_lengths must be nonempty arrays of equal length")
    queries = tuple(_integer(value, "query_tokens[]", 1) for value in query_tokens)
    contexts = tuple(_integer(value, "context_lengths[]") for value in context_lengths)
    if phase == "decode" and any(value != 1 for value in queries):
        _invalid("decode queries must each contain one token")
    if phase in ("prefill", "recompute") and any(contexts):
        _invalid("Unchunked prefill/recompute requires zero prefix context")
    return phase, stage, queries, contexts


def shape_key(phase: str, query_tokens: Sequence[int], context_lengths: Sequence[int],
              stage_index: int = 0) -> str:
    phase, stage, queries, contexts = _shape(phase, query_tokens, context_lengths, stage_index)
    return json.dumps([phase, stage, list(queries), list(contexts)], separators=(",", ":"))


def _observation(value: Any, *, profile: bool, stage_count: int) -> dict:
    required = {"phase", "stage_index", "query_tokens", "context_lengths", "duration_ms"}
    required |= {"sample_count"} if profile else {"record_id", "group_id"}
    row = _object(value, required, {"workspace_peak_bytes"}, "sample" if profile else "record")
    phase, stage, queries, contexts = _shape(row["phase"], row["query_tokens"],
                                            row["context_lengths"], row["stage_index"])
    if stage >= stage_count:
        _invalid("Observation stage_index exceeds the identity pipeline stage count")
    result: dict[str, Any] = {"phase": phase, "stage_index": stage, "query_tokens": list(queries),
              "context_lengths": list(contexts), "duration_ms": _number(row["duration_ms"], "duration_ms"),
              "workspace_peak_bytes": None}
    workspace = row.get("workspace_peak_bytes")
    if workspace is not None:
        result["workspace_peak_bytes"] = _integer(workspace, "workspace_peak_bytes")
    if profile:
        result["sample_count"] = _integer(row["sample_count"], "sample_count", 1)
    else:
        result["record_id"] = _text(row["record_id"], "record_id")
        result["group_id"] = _text(row["group_id"], "group_id")
    return result


def validate_dataset(data: dict, *, allow_test_fixture: bool = False) -> dict:
    """Validate externally supplied step observations; refuse simulated sources."""
    try:
        value = _object(data, {"schema_version", "dataset_id", "identity", "source", "records"},
                        {"notes"}, "dataset")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            _invalid("dataset.schema_version must be 1")
        source = _source(value["source"], allow_test_fixture)
        identity = validate_identity(value["identity"], measured=source["kind"] == "measured")
        if not isinstance(value["records"], list) or not value["records"]:
            _invalid("dataset.records must be a nonempty array")
        records = [_observation(row, profile=False,
                                stage_count=identity["parallelism"]["pipeline_parallel_size"])
                   for row in value["records"]]
        if len({row["record_id"] for row in records}) != len(records):
            _invalid("Duplicate dataset record_id")
        result = {"schema_version": 1, "dataset_id": _text(value["dataset_id"], "dataset_id"),
                  "identity": identity, "source": source,
                  "records": records}
        if "notes" in value:
            if not isinstance(value["notes"], dict):
                _invalid("notes must be an object")
            result["notes"] = _json_value(value["notes"], "notes")
        return result
    except SimulationError as exc:
        if exc.code == "INVALID_PROFILE":
            raise SimulationError("INVALID_DATASET", str(exc)) from exc
        raise


def validate_profile(data: dict, *, allow_test_fixture: bool = False) -> dict:
    """Validate immutable lookup-profile content and fitting provenance."""
    value = _object(data, {"schema_version", "profile_id", "profile_version", "identity", "source", "method",
                           "timing_includes_communication", "samples", "training"}, {"notes"}, "profile")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        _invalid("profile.schema_version must be 1")
    if value["method"] != "lookup":
        _invalid("Only lookup profiles are implemented")
    if value["timing_includes_communication"] is not True:
        _invalid("Lookup profiles must contain whole-stage time including communication")
    source = _source(value["source"], allow_test_fixture)
    identity = validate_identity(value["identity"], measured=source["kind"] == "measured")
    if not isinstance(value["samples"], list) or not value["samples"]:
        _invalid("profile.samples must be a nonempty array")
    samples = [_observation(row, profile=True,
                            stage_count=identity["parallelism"]["pipeline_parallel_size"])
               for row in value["samples"]]
    keys = [shape_key(row["phase"], row["query_tokens"], row["context_lengths"], row["stage_index"])
            for row in samples]
    if len(set(keys)) != len(keys):
        _invalid("Duplicate profile sample shape; aggregate repeated measurements first")
    training = _object(value["training"], {"dataset_sha256", "record_ids", "group_ids", "shape_keys",
                                            "parent_profile_version"}, set(), "training")
    if not isinstance(training["dataset_sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", training["dataset_sha256"]):
        _invalid("training.dataset_sha256 must be a SHA256 hex digest")
    parent = training["parent_profile_version"]
    if parent is not None:
        _text(parent, "training.parent_profile_version")
    training = {"dataset_sha256": training["dataset_sha256"].lower(),
                "record_ids": _string_list(training["record_ids"], "training.record_ids"),
                "group_ids": _string_list(training["group_ids"], "training.group_ids"),
                "shape_keys": _string_list(training["shape_keys"], "training.shape_keys"),
                "parent_profile_version": parent}
    if set(keys) != set(training["shape_keys"]):
        _invalid("Training shape provenance must exactly match the profile sample shapes")
    if sum(row["sample_count"] for row in samples) != len(training["record_ids"]):
        _invalid("Sample observation counts must equal the number of training records")
    if len(training["group_ids"]) > len(training["record_ids"]):
        _invalid("Training groups cannot outnumber training records")
    version = _text(value["profile_version"], "profile_version")
    if parent == version:
        _invalid("A child profile must have a new version")
    result = {"schema_version": 1, "profile_id": _text(value["profile_id"], "profile_id"),
              "profile_version": version, "identity": identity,
              "source": source, "method": "lookup",
              "timing_includes_communication": True, "samples": samples, "training": training}
    if "notes" in value:
        if not isinstance(value["notes"], dict):
            _invalid("notes must be an object")
        result["notes"] = _json_value(value["notes"], "notes")
    return result


def _read_json(path: str | Path) -> tuple[dict, bytes]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _invalid(f"Duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(value, dict):
            _invalid("Profile/dataset document must be an object")
        return value, raw
    except (OSError, ValueError, UnicodeError) as exc:
        raise SimulationError("INVALID_PROFILE", f"{path}: {exc}") from exc


def load_profile(path: str | Path, *, allow_test_fixture: bool = False) -> dict:
    return validate_profile(_read_json(path)[0], allow_test_fixture=allow_test_fixture)


def load_dataset(path: str | Path, *, allow_test_fixture: bool = False) -> dict:
    try:
        return validate_dataset(_read_json(path)[0], allow_test_fixture=allow_test_fixture)
    except SimulationError as exc:
        if exc.code == "INVALID_PROFILE":
            raise SimulationError("INVALID_DATASET", str(exc)) from exc
        raise


def _coordinate(sample: dict) -> tuple[int, int, int] | None:
    queries, contexts = sample["query_tokens"], sample["context_lengths"]
    if len(set(queries)) == len(set(contexts)) == 1:
        return len(queries), queries[0], contexts[0]
    return None


def _axis_weights(value: int, axis: list[int], allow_extrapolation: bool) -> tuple[list[tuple[int, float]], bool]:
    index = bisect_left(axis, value)
    if index < len(axis) and axis[index] == value:
        return [(value, 1.0)], False
    outside = index == 0 or index == len(axis)
    if outside and (not allow_extrapolation or len(axis) < 2):
        raise SimulationError("OUT_OF_CALIBRATION_DOMAIN", "Query is outside a calibrated axis")
    lower, upper = ((axis[0], axis[1]) if index == 0 else (axis[-2], axis[-1])
                    if index == len(axis) else (axis[index - 1], axis[index]))
    fraction = (value - lower) / (upper - lower)
    return [(lower, 1.0 - fraction), (upper, fraction)], outside


def estimate_profile(profile: dict, phase: str, query_tokens: Sequence[int],
                     context_lengths: Sequence[int], stage_index: int = 0, *,
                     allow_extrapolation: bool = False, allow_test_fixture: bool = False) -> StepEstimate:
    """Lookup or interpolate only a complete homogeneous grid cell."""
    if type(allow_extrapolation) is not bool:
        _invalid("allow_extrapolation must be a boolean")
    profile = validate_profile(profile, allow_test_fixture=allow_test_fixture)
    key = shape_key(phase, query_tokens, context_lengths, stage_index)
    samples = [row for row in profile["samples"] if row["phase"] == phase and row["stage_index"] == stage_index]
    indexed = {shape_key(row["phase"], row["query_tokens"], row["context_lengths"], stage_index): row
               for row in samples}
    exact = indexed.get(key)
    extrapolated = False
    if exact is not None:
        duration, workspace, contributors = exact["duration_ms"], exact["workspace_peak_bytes"], [key]
        method = "exact"
    else:
        coordinate = _coordinate({"query_tokens": query_tokens, "context_lengths": context_lengths})
        grid = {point: row for row in samples if (point := _coordinate(row)) is not None}
        if coordinate is None or not grid:
            raise SimulationError("OUT_OF_CALIBRATION_DOMAIN", "No exact ragged shape or homogeneous grid")
        axes = [sorted({point[axis] for point in grid}) for axis in range(3)]
        weights = []
        for value, axis in zip(coordinate, axes):
            axis_weights, outside = _axis_weights(value, axis, allow_extrapolation)
            weights.append(axis_weights)
            extrapolated |= outside
        duration = 0.0
        spaces: list[int | None] = []
        contributors = []
        for corner in product(*weights):
            corner_point = (corner[0][0], corner[1][0], corner[2][0])
            row = grid.get(corner_point)
            if row is None:
                raise SimulationError("OUT_OF_CALIBRATION_DOMAIN", "Interpolation cell has an unmeasured corner")
            duration += math.prod(item[1] for item in corner) * row["duration_ms"]
            spaces.append(row["workspace_peak_bytes"])
            contributors.append(shape_key(phase, row["query_tokens"], row["context_lengths"], stage_index))
        # Choose corner maximum instead of a weighted mean. This is a stated
        # interpolation policy, not proof that unmeasured interior peaks cannot
        # exceed it. Extrapolated workspace is explicitly unknown.
        workspace = (None if extrapolated or any(value is None for value in spaces)
                     else max(value for value in spaces if value is not None))
        method = "extrapolated" if extrapolated else "multilinear"
    if not math.isfinite(duration) or duration <= 0:
        raise SimulationError("OUT_OF_CALIBRATION_DOMAIN", "Profile prediction is not positive and finite")
    source = "measured_profile" if profile["source"]["kind"] == "measured" else "test_fixture_profile"
    return StepEstimate(duration, {"extrapolated": extrapolated,
        "method": {"exact": "exact", "multilinear": "interpolation", "extrapolated": "extrapolation"}[method],
        "workspace_peak_bytes": workspace, "profile": {
        "duration_ms": duration, "lookup": method, "profile_version": profile["profile_version"],
        "workspace_peak_bytes": workspace, "extrapolated": extrapolated,
        "timing_includes_communication": True, "contributing_shapes": contributors,
    }}, source, (
        "Whole-stage profile includes local collectives and output/sampling; do not add them again.",
        "Inter-stage PP P2P is not included in stage samples.",
        "No Attention/Linear/Misc attribution can be inferred from total step time.",
        "Interpolated workspace uses measured corner maxima, without an interior peak guarantee.",
        "Explicit exploratory extrapolation was used." if extrapolated else "Prediction is within the sampled grid domain.",
    ))


class ProfileModel:
    """Snapshot a profile once, check identity, and retain query provenance."""

    def __init__(self, experiment: Experiment):
        from .config import validate_experiment
        validate_experiment(experiment)
        if not experiment.calibration.profile_path:
            raise SimulationError("INVALID_CONFIG", "profile_path is required for the profile backend")
        self.experiment = experiment
        path = Path(experiment.calibration.profile_path)
        data, raw = _read_json(path)
        self._profile = validate_profile(data, allow_test_fixture=experiment.calibration.allow_test_fixture)
        requested_version = experiment.calibration.profile_version
        if (requested_version != "uncalibrated-v1"
                and requested_version != self._profile["profile_version"]):
            raise SimulationError("CALIBRATION_MISMATCH", "Configured profile_version differs from profile file")
        identity = profile_identity(experiment)
        if identity != self._profile["identity"]:
            changed = [key for key in identity if identity[key] != self._profile["identity"].get(key)]
            raise SimulationError("CALIBRATION_MISMATCH", f"Profile identity mismatch: {', '.join(changed)}")
        self.provenance: dict[str, Any] = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                           "profile_id": self._profile["profile_id"],
                           "profile_version": self._profile["profile_version"],
                           "source_kind": self._profile["source"]["kind"],
                           "source_reference": self._profile["source"]["reference"],
                           "timing_includes_communication": True,
                           "domain_policy": "explicit_extrapolation" if experiment.calibration.allow_extrapolation
                           else "reject_outside_grid",
                           "extrapolation_used": False}

    @property
    def profile(self) -> dict:
        return deepcopy(self._profile)

    def estimate_step(self, phase: str, query_tokens: Sequence[int], context_lengths: Sequence[int],
                      stage_index: int = 0) -> StepEstimate:
        result = estimate_profile(self._profile, phase, query_tokens, context_lengths, stage_index,
                                  allow_extrapolation=self.experiment.calibration.allow_extrapolation,
                                  allow_test_fixture=self.experiment.calibration.allow_test_fixture)
        self.provenance["extrapolation_used"] |= result.breakdown["profile"]["extrapolated"]
        return result

    def workspace_bytes(self, query_tokens: Sequence[int], phase: str, stage_index: int = 0,
                        context_lengths: Sequence[int] | None = None) -> int | None:
        if context_lengths is None:
            if phase == "decode":
                return None
            context_lengths = [0] * len(query_tokens)
        result = estimate_profile(self._profile, phase, query_tokens, context_lengths, stage_index,
                                  allow_extrapolation=self.experiment.calibration.allow_extrapolation,
                                  allow_test_fixture=self.experiment.calibration.allow_test_fixture)
        return result.breakdown["profile"]["workspace_peak_bytes"]


def create_performance_model(experiment: Experiment) -> "AnalyticalModel | ProfileModel":
    if experiment.calibration.backend == "profile":
        return ProfileModel(experiment)
    from .performance import AnalyticalModel
    return AnalyticalModel(experiment)
