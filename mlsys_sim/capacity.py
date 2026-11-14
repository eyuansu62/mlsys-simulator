"""Fresh-process capacity sweeps with typed OOM and admission kept separate.

GPU allocator observations remain raw. They are never relabeled Workspace, and
an observed transition is not an exact or generally monotone capacity threshold.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, NoReturn
import uuid

from .config import experiment_from_dict, validate_experiment
from .hardware import CollectionError, bind_inventory, probe_rocm
from .schema import Experiment, SimulationError
from .serialization import experiment_dict


PROTOCOL = "fresh_owned_runtime_capacity_v1"
_DATASET = "memory_capacity_measurements"
_CAPTURE = "memory_capacity_trial_capture"
_MAX_BYTES = 16 * 1024 * 1024
_AXES = {"input_tokens", "request_count", "kv_pool_bytes"}
_PHASES = {"startup", "load", "prepare", "run", "cleanup"}
_STATUSES = {"COMPLETED", "OOM", "ADMISSION_REJECTED", "ERROR", "TIMEOUT", "UNAVAILABLE"}
_PROFILE_FIELDS = ("profile_path", "replica_profile_paths", "native_pp_communication_profile_path",
    "replica_native_pp_communication_profile_paths", "workspace_profile_path", "replica_workspace_profile_paths")


def _fail(message: str, code: str = "INVALID_CAPACITY_DATASET") -> NoReturn:
    raise SimulationError(code, message)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise SimulationError("INVALID_CAPACITY_DATASET", "Capacity evidence must be finite JSON") from exc


def config_sha256(experiment: Experiment | dict) -> str:
    value = experiment_dict(experiment)
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _object(value: Any, required: set[str], optional: set[str], label: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(f"{label} must be an object with string keys")
    if required - value.keys() or value.keys() - required - optional:
        _fail(f"{label} has missing or unknown fields")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\x00" in value:
        _fail(f"{label} must be a bounded nonempty string")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        _fail(f"{label} must be a lowercase SHA256")
    return value


def _source(value: Any, allow_test_fixture: bool) -> dict:
    source = _object(value, {"kind", "reference"}, set(), "source")
    if source["kind"] not in {"measured", "test_fixture"}:
        _fail("Capacity sources must be measured or test_fixture")
    _text(source["reference"], "source.reference")
    if source["kind"] == "test_fixture" and not allow_test_fixture:
        _fail("Capacity fixtures require explicit permission", "TEST_FIXTURE_NOT_ALLOWED")
    return dict(source)


def _axis(value: Any) -> dict:
    axis = _object(value, {"name", "values"}, set(), "axis")
    if axis["name"] not in _AXES:
        _fail("Unsupported capacity axis")
    values = axis["values"]
    if not isinstance(values, list) or not 1 <= len(values) <= 128:
        _fail("axis.values must contain 1 to 128 positive unique integers")
    for number in values:
        _integer(number, "axis.values[]", 1)
    if len(set(values)) != len(values):
        _fail("Capacity axis values must be unique")
    return {"name": axis["name"], "values": list(values)}


def normalize_capacity_plan(experiment: Experiment, value: dict) -> dict:
    validate_experiment(experiment)
    plan = _object(value, {"schema_version", "dataset_id", "axis"},
        {"source_reference", "repeats", "timeout_s", "max_output_bytes", "raw_path", "source"}, "capacity plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        _fail("Capacity plan schema_version must be 1")
    axis = _axis(plan["axis"])
    repeats = _integer(plan.get("repeats", 2), "repeats", 1)
    timeout = plan.get("timeout_s", 120)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        _fail("timeout_s must be finite and in (0,86400]")
    limit = _integer(plan.get("max_output_bytes", 8 * 1024 * 1024), "max_output_bytes", 1)
    if repeats > 16 or repeats * len(axis["values"]) > 1024 or limit > _MAX_BYTES:
        _fail("Capacity sweep exceeds its trial or output bound")
    if axis["name"] in {"input_tokens", "request_count"} and len(experiment.workload) != 1:
        _fail("input_tokens and request_count sweeps require one base request")
    cohort_total = (sum(axis["values"]) if axis["name"] == "request_count"
                    else len(axis["values"]) * len(experiment.workload)) * repeats
    if cohort_total > 100000:
        _fail("Capacity sweep exceeds 100000 total request observations")
    for item in axis["values"]:
        capacity_experiment(experiment, axis["name"], item)
    result = {"schema_version": 1, "dataset_id": _text(plan["dataset_id"], "dataset_id"), "axis": axis,
        "source_reference": _text(plan.get("source_reference", "fresh owned-process Runtime capacity sweep"),
                                  "source_reference"),
        "repeats": repeats, "timeout_s": timeout, "max_output_bytes": limit}
    if "raw_path" in plan:
        result.update(raw_path=_text(plan["raw_path"], "raw_path"),
                      source=_source(plan.get("source"), True))
    elif "source" in plan:
        _fail("Live capacity collection derives source from execution")
    return result


def capacity_experiment(base: Experiment, axis: str, value: int) -> Experiment:
    """Change exactly the declared axis, never scheduler batch budgets."""
    base = experiment_from_dict(experiment_dict(base))
    _integer(value, "axis value", 1)
    if axis not in _AXES:
        _fail("Unsupported capacity axis")
    if axis == "kv_pool_bytes":
        current = replace(base, strategy=replace(base.strategy,
            memory=replace(base.strategy.memory, kv_pool_bytes=value)))
    else:
        if len(base.workload) != 1:
            _fail("This capacity axis requires one base request")
        request = base.workload[0]
        if axis == "input_tokens":
            if request.prompt_token_ids and value > len(request.prompt_token_ids):
                _fail("input_tokens cannot extend a supplied token template")
            current = replace(base, workload=(replace(request, input_tokens=value,
                prompt_token_ids=request.prompt_token_ids[:value] if request.prompt_token_ids else ()),))
        else:
            if value > 100000:
                _fail("request_count exceeds the bounded cohort size")
            current = replace(base, workload=tuple(replace(request,
                request_id=f"{request.request_id}/capacity-{index}") for index in range(value)))
    validate_experiment(current)
    return current


def _read(path: Path, limit: int = _MAX_BYTES) -> tuple[dict, bytes]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("Duplicate JSON key in capacity evidence")
            result[key] = value
        return result
    try:
        if path.is_symlink():
            _fail("Capacity inputs may not be symlinks")
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            _fail("Capacity evidence exceeds its bounded read size", "COLLECTION_OUTPUT_LIMIT")
        payload = json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda _: _fail("Capacity evidence must be finite JSON"))
        if not isinstance(payload, dict):
            _fail("Capacity evidence must be an object")
        canonical_json(payload)
        return payload, raw
    except SimulationError:
        raise
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise SimulationError("INVALID_CAPACITY_DATASET", f"Cannot read capacity evidence {path}: {exc}") from exc


def _selected_paths(experiment: Experiment, field: str) -> tuple[str, ...]:
    selected = getattr(experiment.calibration, field, None)
    if selected is None:
        return ()
    if field.endswith("paths"):
        if not isinstance(selected, (tuple, list)) or any(not isinstance(path, str) for path in selected):
            _fail("Selected Profile paths must be a string array")
        return tuple(selected)
    if not isinstance(selected, str):
        _fail("Selected Profile path must be a string")
    return (selected,)


def _profile_inputs(experiment: Experiment, max_bytes: int = _MAX_BYTES) -> list[dict]:
    result: list[dict] = []
    for field in _PROFILE_FIELDS:
        paths = _selected_paths(experiment, field)
        for index, path in enumerate(paths):
            used = len(canonical_json(result).encode())
            payload, raw = _read(Path(path), max(1, max_bytes - used))
            source = _source(payload.get("source"), True)
            result.append({"field": field, "index": index, "origin_path": str(Path(path).resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(), "source_kind": source["kind"],
                "raw_json": raw.decode("utf-8")})
            if len(canonical_json(result).encode()) > max_bytes:
                _fail("Capacity Profile inputs exceed the complete sweep budget", "COLLECTION_OUTPUT_LIMIT")
    return result


def _validate_inputs(experiment: Experiment, supplied: Any) -> list[dict]:
    if not isinstance(supplied, list) or len(supplied) > 192:
        _fail("profile_inputs must be a bounded array")
    expected = {}
    for field in _PROFILE_FIELDS:
        paths = _selected_paths(experiment, field)
        expected.update({(field, index): path for index, path in enumerate(paths)})
    seen = set()
    result = []
    for item in supplied:
        item = _object(item, {"field", "index", "origin_path", "sha256", "source_kind", "raw_json"}, set(), "profile input")
        field, index = item["field"], _integer(item["index"], "profile index")
        key = field, index
        if key not in expected or key in seen or not isinstance(item["raw_json"], str):
            _fail("Profile inputs do not match the selected configuration")
        raw = item["raw_json"].encode("utf-8")
        if len(raw) > _MAX_BYTES or hashlib.sha256(raw).hexdigest() != _digest(item["sha256"], "profile SHA256"):
            _fail("Profile input bytes differ from their captured SHA256")
        try:
            payload = json.loads(item["raw_json"])
        except (ValueError, TypeError) as exc:
            raise SimulationError("INVALID_CAPACITY_DATASET", "Profile snapshot must be JSON") from exc
        source = _source(payload.get("source") if isinstance(payload, dict) else None, True)
        if source["kind"] != item["source_kind"]:
            _fail("Profile input source differs from its captured bytes")
        _text(item["origin_path"], "profile origin_path")
        seen.add(key)
        result.append(deepcopy(item))
    if seen != set(expected):
        _fail("A selected Profile input is missing from the capacity evidence")
    return result


def _bind_inputs(experiment: Experiment, inputs: list[dict], directory: Path) -> Experiment:
    changes: dict[str, Any] = {}
    for item in inputs:
        path = directory / f"{item['field']}-{item['index']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(item["raw_json"].encode("utf-8"))
        if item["field"].endswith("paths"):
            changes.setdefault(item["field"], [None] * len(getattr(experiment.calibration, item["field"])))
            changes[item["field"]][item["index"]] = str(path.resolve())
        else:
            changes[item["field"]] = str(path.resolve())
    for field in tuple(changes):
        if field.endswith("paths"):
            changes[field] = tuple(changes[field])
    return replace(experiment, calibration=replace(experiment.calibration, **changes))


def _semantic_config(experiment: Experiment, inputs: list[dict]) -> str:
    config = experiment_dict(experiment)
    for item in inputs:
        if item["field"].endswith("paths"):
            selected = list(config["calibration"][item["field"]])
            selected[item["index"]] = "sha256:" + item["sha256"]
            config["calibration"][item["field"]] = selected
        else:
            config["calibration"][item["field"]] = "sha256:" + item["sha256"]
    return canonical_json(config)


def _predict(experiment: Experiment, inputs: list[dict]) -> dict:
    from .engine import simulate
    physical = {"STATIC_MEMORY_INFEASIBLE", "WORKSPACE_BUDGET_EXCEEDED"}
    domain = {"OUT_OF_CALIBRATION_DOMAIN", "OUT_OF_COMMUNICATION_CALIBRATION_DOMAIN", "OUT_OF_WORKSPACE_CALIBRATION_DOMAIN"}
    with tempfile.TemporaryDirectory(prefix="mlsys-capacity-prediction-") as temporary:
        effective = _bind_inputs(experiment, inputs, Path(temporary)) if inputs else experiment
        try:
            run = simulate(effective)
        except SimulationError as exc:
            category = "PHYSICAL_INFEASIBLE" if exc.code in physical else "CALIBRATION_DOMAIN_ERROR" if exc.code in domain else "ERROR"
            return {"category": category, "physical_feasible": False if category == "PHYSICAL_INFEASIBLE" else None,
                "status": "FAILED", "error_codes": [exc.code], "counts": {}, "calibration_source": experiment.calibration.backend}
    codes = sorted({row["error_code"] for row in run.requests if row.get("error_code")})
    if run.result.get("error_code"):
        codes = sorted(set(codes) | {run.result["error_code"]})
    if set(codes) & physical:
        category, feasible = "PHYSICAL_INFEASIBLE", False
    elif run.result["counts"]["completed"] == len(experiment.workload) and run.result["status"] == "COMPLETED":
        category, feasible = "PHYSICAL_FEASIBLE", True
    else:
        admission = {"KV_CAPACITY_EXCEEDED", "HOST_SWAP_CAPACITY_EXCEEDED", "MEMORY_NO_PROGRESS",
            "PREEMPTION_LIMIT_EXCEEDED", "REQUEST_TIMEOUT", "OUTSIDE_MEASUREMENT_WINDOW", "BATCH_TOKEN_LIMIT_EXCEEDED",
            "MEASUREMENT_WINDOW_END", "DRAIN_TIME_LIMIT", "SIMULATION_EVENT_LIMIT", "SIMULATION_TIME_LIMIT"}
        category, feasible = ("ADMISSION_LIMIT" if not codes or set(codes) <= admission else "ERROR"), None
    return {"category": category, "physical_feasible": feasible, "status": run.result["status"],
        "error_codes": codes, "counts": deepcopy(run.result["counts"]),
        "calibration_source": run.result.get("calibration_source", experiment.calibration.backend)}


def _allocator(value: Any, devices: list[dict]) -> list[dict]:
    if not isinstance(value, list):
        _fail("Allocator evidence must be an array")
    result = []
    seen = set()
    for item in value:
        item = _object(item, {"device_index", "allocated_bytes", "reserved_bytes", "free_bytes", "total_bytes",
            "peak_allocated_bytes", "peak_reserved_bytes"}, set(), "allocator observation")
        index = _integer(item["device_index"], "allocator device index")
        if index >= len(devices) or index in seen:
            _fail("Allocator observations have repeated or foreign devices")
        seen.add(index)
        row = {name: _integer(number, f"allocator {name}") for name, number in item.items()}
        if (row["allocated_bytes"] > row["reserved_bytes"] or row["free_bytes"] > row["total_bytes"]
                or row["reserved_bytes"] > row["total_bytes"] or row["peak_allocated_bytes"] < row["allocated_bytes"]
                or row["peak_reserved_bytes"] < row["reserved_bytes"]):
            _fail("Allocator counters are inconsistent")
        result.append(row)
    return result


def _normalize_observation(value: Any, experiment: Experiment, trial: dict, session: str,
                           source: dict) -> dict:
    """Malformed/missing worker proof is censored, never promoted to OOM."""
    raw = deepcopy(value)
    try:
        row = _object(value, {"schema_version", "kind", "protocol", "session_id", "worker_id", "point_index", "repeat",
            "axis", "effective_config_sha256", "source", "status", "failure_phase", "exception_class", "typed_oom",
            "stack", "devices", "metadata", "allocator", "error_code", "message", "result", "cleanup_status",
            "elapsed_ms", "kv_pool_bindings", "isolation", "oom_allocator_evidence", "capacity_oom_evidence"},
            set(), "capacity worker capture")
        if type(row["schema_version"]) is not int or row["schema_version"] != 1 or row["kind"] != _CAPTURE or row["protocol"] != PROTOCOL:
            _fail("Unsupported capacity worker schema")
        expected = {"session_id": session, "worker_id": trial["worker_id"], "point_index": trial["point_index"],
            "repeat": trial["repeat"], "axis": {"name": trial["axis_name"], "value": trial["axis_value"]},
            "effective_config_sha256": trial["effective_config_sha256"]}
        if any(canonical_json(row[key]) != canonical_json(expected_value) for key, expected_value in expected.items()):
            _fail("Worker capture does not match its fresh trial/configuration")
        current_source = _source(row["source"], True)
        if source["kind"] == "measured" and current_source["kind"] == "test_fixture":
            _fail("Worker fixture source cannot be promoted to measured")
        if row["status"] not in _STATUSES or type(row["typed_oom"]) is not bool:
            _fail("Unknown capacity worker status or typed OOM flag")
        if row["failure_phase"] is not None and row["failure_phase"] not in _PHASES:
            _fail("Unknown capacity failure phase")
        if row["isolation"] != "owned_process":
            _fail("Capacity measurement requires an owned fresh process")
        devices = row["devices"]
        if not isinstance(devices, list):
            _fail("Capacity device facts must be an array")
        for index, fact in enumerate(devices):
            fact = _object(fact, {"index", "name", "memory_bytes"}, {"architecture"}, "capacity device")
            if (type(fact["index"]) is not int or fact["index"] != index or index >= len(experiment.infra.devices)
                    or fact["name"] != experiment.infra.devices[index].name
                    or type(fact["memory_bytes"]) is not int or fact["memory_bytes"] != experiment.infra.devices[index].memory_bytes):
                _fail("Capacity worker device binding differs from the effective experiment")
        stack = row["stack"]
        if not isinstance(stack, dict):
            _fail("Capacity stack must be an object")
        canonical_json(stack)
        allocator = _object(row["allocator"], {"before", "after", "errors"}, set(), "allocator evidence")
        for key in ("before", "after"):
            _allocator(allocator[key], devices)
        if not isinstance(allocator["errors"], list):
            _fail("Allocator read errors must be retained as an array")
        status = row["status"]
        eligible = status in {"COMPLETED", "OOM"}
        if status in {"COMPLETED", "OOM", "ADMISSION_REJECTED"}:
            if len(devices) != len(experiment.infra.devices) or not stack.get("torch_version") or not stack.get("hip_version"):
                _fail("Eligible capacity evidence needs the actual HIP/device stack")
            if source["kind"] == "measured" and row["typed_oom"] and row["exception_class"] not in {
                    "torch.OutOfMemoryError", "torch.cuda.OutOfMemoryError"}:
                _fail("Measured OOM must name the actual PyTorch OOM exception class")
            metadata = row["metadata"]
            if not isinstance(metadata, dict) or metadata.get("isolation") != "owned_process":
                _fail("Missing owned Runtime metadata")
            if (metadata.get("effective_config_sha256") != trial["effective_config_sha256"]
                    or canonical_json(metadata.get("applied_axis")) != canonical_json(expected["axis"])
                    or metadata.get("runtime_name") != experiment.calibration.runtime_name
                    or metadata.get("runtime_version") != experiment.calibration.runtime_version
                    or metadata.get("execution_scope") != "complete_declared_workload"):
                _fail("Runtime metadata does not prove the effective configuration/axis")
            oom_devices = set()
            if status == "OOM":
                if row["failure_phase"] not in {"prepare", "run"}:
                    _fail("OOM before scoped workload preparation or during cleanup cannot establish an axis threshold")
                evidence = _object(row["oom_allocator_evidence"], {"phase", "per_device"}, set(), "OOM allocator evidence")
                if evidence["phase"] != row["failure_phase"] or not isinstance(evidence["per_device"], list):
                    _fail("OOM allocator evidence must bind its actual failure phase")
                seen_oom = set()
                for counter in evidence["per_device"]:
                    counter = _object(counter, {"device_index", "before", "after"}, set(), "OOM counter")
                    index = _integer(counter["device_index"], "OOM counter device")
                    if index >= len(devices) or index in seen_oom:
                        _fail("OOM allocator counters have foreign or repeated devices")
                    seen_oom.add(index)
                    before, after = counter["before"], counter["after"]
                    if before is None or after is None:
                        continue
                    before, after = _integer(before, "num_ooms before"), _integer(after, "num_ooms after")
                    if after < before:
                        _fail("OOM allocator counter moved backwards")
                    if after > before:
                        oom_devices.add(index)
                if not oom_devices:
                    _fail("Typed exception lacks an actual device allocator num_ooms increase")
            if trial["axis_name"] == "kv_pool_bytes":
                if type(metadata.get("fixed_kv_pool_bytes")) is not int or metadata["fixed_kv_pool_bytes"] != trial["axis_value"]:
                    _fail("KV pool axis needs actual applied fixed-pool evidence")
                if status == "OOM":
                    proof = _object(row["capacity_oom_evidence"],
                        {"resource", "phase", "effective_config_sha256", "attempts"}, set(), "KV allocation attempt")
                    if (proof["resource"] != "kv_pool" or proof["phase"] != row["failure_phase"]
                            or proof["effective_config_sha256"] != trial["effective_config_sha256"]
                            or not isinstance(proof["attempts"], list) or not proof["attempts"]):
                        _fail("KV-pool OOM lacks a scoped actual allocation attempt")
                    verified = set()
                    for attempt in proof["attempts"]:
                        attempt = _object(attempt, {"device_index", "shape", "element_size", "target_pool_bytes", "operation"},
                                          set(), "KV pool allocation attempt")
                        index = _integer(attempt["device_index"], "KV attempted device")
                        shape = attempt["shape"]
                        if not isinstance(shape, list) or not shape or len(shape) > 8:
                            _fail("KV allocation shape must be a bounded array")
                        size = _integer(attempt["element_size"], "KV attempted element size", 1)
                        for dimension in shape:
                            size *= _integer(dimension, "KV allocation dimension", 1)
                        if (index not in oom_devices or attempt["operation"] not in {"torch.empty", "torch.zeros"}
                                or type(attempt["target_pool_bytes"]) is not int
                                or attempt["target_pool_bytes"] != trial["axis_value"] or size != trial["axis_value"]):
                            _fail("KV allocation request differs from the failed device and full pool target")
                        verified.add(index)
                    if not verified:
                        _fail("KV OOM request has no matching allocator failure")
                else:
                    bindings = row["kv_pool_bindings"]
                    if not isinstance(bindings, list) or len(bindings) != len(devices):
                        _fail("KV pool axis lacks per-device physical allocation binding")
                    for index, binding in enumerate(bindings):
                        if (not isinstance(binding, dict) or type(binding.get("device_index")) is not int
                                or binding["device_index"] != index
                                or type(binding.get("storage_bytes")) is not int or binding["storage_bytes"] != trial["axis_value"]
                                or type(binding.get("allocation_count")) is not int or binding["allocation_count"] <= 0
                                or binding.get("method") != "observed_unique_tensor_storage"):
                            _fail("KV pool axis does not bind actual storage on every rank")
            if status == "OOM" and (row["typed_oom"] is not True or not row["exception_class"]):
                _fail("A string error cannot establish typed OOM")
            if status != "OOM" and row["typed_oom"]:
                _fail("Typed OOM flag disagrees with the capacity status")
            if status == "COMPLETED" and (row["failure_phase"] is not None or row["result"] is None
                    or row["result"].get("status") != "COMPLETED"):
                _fail("Completed capacity evidence must execute its complete workload")
            if status == "COMPLETED":
                expected_ids = {request.request_id for request in experiment.workload}
                identifiers = row["result"].get("completed_request_ids")
                if not isinstance(identifiers, list) or len(identifiers) != len(expected_ids) or set(identifiers) != expected_ids:
                    _fail("Completed capacity evidence does not cover the entire declared workload")
        elapsed = row["elapsed_ms"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            _fail("Capacity elapsed_ms must be finite and nonnegative")
        return {"status": status, "observed_status": status, "eligible_for_bracket": eligible,
            "failure_phase": row["failure_phase"], "error_code": row["error_code"],
            "exception_class": row["exception_class"], "typed_oom": row["typed_oom"], "raw": raw}
    except (SimulationError, AttributeError, TypeError, ValueError) as exc:
        return {"status": "ERROR", "observed_status": raw.get("status") if isinstance(raw, dict) else None,
            "eligible_for_bracket": False, "failure_phase": raw.get("failure_phase") if isinstance(raw, dict) else None,
            "error_code": "UNVERIFIED_CAPACITY_EVIDENCE", "exception_class": raw.get("exception_class") if isinstance(raw, dict) else None,
            "typed_oom": raw.get("typed_oom") is True if isinstance(raw, dict) else False,
            "reason": str(exc), "raw": raw}


def validate_capacity_dataset(data: dict, *, allow_test_fixture: bool = False) -> dict:
    """Check full trial identities and recompute CPU predictions from snapshots."""
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be boolean")
    canonical_json(data)
    value = _object(data, {"schema_version", "kind", "protocol", "dataset_id", "source", "session_id",
        "base_experiment", "axis", "repeats", "profile_inputs", "trials"}, {"notes"}, "capacity dataset")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != _DATASET or value["protocol"] != PROTOCOL:
        _fail("Unsupported capacity dataset schema or protocol")
    base = experiment_from_dict(value["base_experiment"])
    if canonical_json(experiment_dict(base)) != canonical_json(experiment_dict(value["base_experiment"])):
        _fail("Capacity base_experiment must be the complete expanded configuration")
    axis = _axis(value["axis"])
    repeats = _integer(value["repeats"], "repeats", 1)
    if repeats > 16 or repeats * len(axis["values"]) > 1024:
        _fail("Capacity trial bound exceeded")
    inputs = _validate_inputs(base, value["profile_inputs"])
    source = _source(value["source"], True)
    trials = value["trials"]
    if not isinstance(trials, list) or len(trials) != repeats * len(axis["values"]):
        _fail("Capacity trials must cover every point/repetition exactly")
    for item in trials:
        observation = item.get("observation") if isinstance(item, dict) else None
        observed_source = observation.get("source") if isinstance(observation, dict) else None
        if isinstance(observed_source, dict) and observed_source.get("kind") == "test_fixture":
            source["kind"] = "test_fixture"
    if any(item["source_kind"] == "test_fixture" for item in inputs):
        source["kind"] = "test_fixture"
    source = _source(source, allow_test_fixture)
    session = _text(value["session_id"], "session_id")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", session):
        _fail("session_id must be a bounded safe identifier")
    seen, workers = set(), set()
    result_trials = []
    predictions = {}
    for supplied in trials:
        trial = _object(supplied, {"point_index", "repeat", "worker_id", "axis_name", "axis_value",
            "effective_experiment", "effective_config_sha256", "observation", "prediction"}, set(), "capacity trial")
        point, repeat = _integer(trial["point_index"], "point_index"), _integer(trial["repeat"], "repeat")
        if point >= len(axis["values"]) or repeat >= repeats or (point, repeat) in seen:
            _fail("Capacity trials have repeated or foreign point/repetition")
        worker = _text(trial["worker_id"], "worker_id")
        if worker in workers:
            _fail("Each capacity trial must own a fresh worker identifier")
        if trial["axis_name"] != axis["name"] or type(trial["axis_value"]) is not int or trial["axis_value"] != axis["values"][point]:
            _fail("Capacity trial axis differs from its declared point")
        effective = capacity_experiment(base, axis["name"], trial["axis_value"])
        if canonical_json(experiment_dict(trial["effective_experiment"])) != canonical_json(experiment_dict(effective)):
            _fail("Capacity trial changed fields outside its declared axis")
        digest = config_sha256(effective)
        if _digest(trial["effective_config_sha256"], "effective_config_sha256") != digest:
            _fail("Capacity effective configuration differs from its SHA256")
        if point not in predictions:
            predictions[point] = _predict(effective, inputs)
        result_trials.append({**deepcopy(trial), "prediction": deepcopy(predictions[point])})
        seen.add((point, repeat))
        workers.add(worker)
    result = {"schema_version": 1, "kind": _DATASET, "protocol": PROTOCOL,
        "dataset_id": _text(value["dataset_id"], "dataset_id"), "source": source, "session_id": session,
        "base_experiment": experiment_dict(base), "axis": axis, "repeats": repeats, "profile_inputs": inputs,
        "trials": sorted(result_trials, key=lambda row: (row["point_index"], row["repeat"]))}
    if "notes" in value:
        if not isinstance(value["notes"], dict):
            _fail("Capacity dataset notes must be an object")
        result["notes"] = deepcopy(value["notes"])
    return result


def load_capacity_dataset(path: str | Path, *, allow_test_fixture: bool = False) -> dict:
    value, _ = _read(Path(path))
    return validate_capacity_dataset(value, allow_test_fixture=allow_test_fixture)


def validate_memory_capacity(data: dict, *, allow_test_fixture: bool = False) -> dict:
    """Report observed intervals, censoring and errors without an exact knee."""
    dataset = validate_capacity_dataset(data, allow_test_fixture=allow_test_fixture)
    axis, repeats = dataset["axis"], dataset["repeats"]
    trials = []
    confusion = {"physical_infeasible_and_typed_oom": 0, "physical_infeasible_but_completed": 0,
                 "physical_feasible_but_typed_oom": 0, "physical_feasible_and_completed": 0,
                 "excluded_admission_domain_or_censored": 0}
    for trial in dataset["trials"]:
        effective = experiment_from_dict(trial["effective_experiment"])
        observed = _normalize_observation(trial["observation"], effective, trial, dataset["session_id"], dataset["source"])
        prediction = trial["prediction"]
        feasible = prediction["physical_feasible"]
        if observed["eligible_for_bracket"] and type(feasible) is bool:
            key = ("physical_feasible_" if feasible else "physical_infeasible_") + (
                "and_completed" if feasible and observed["status"] == "COMPLETED" else
                "but_typed_oom" if feasible else
                "and_typed_oom" if observed["status"] == "OOM" else "but_completed")
            confusion[key] += 1
        else:
            confusion["excluded_admission_domain_or_censored"] += 1
        trials.append({"point_index": trial["point_index"], "repeat": trial["repeat"], "worker_id": trial["worker_id"],
            "axis_value": trial["axis_value"], "effective_config_sha256": trial["effective_config_sha256"],
            "prediction": prediction, "observation": observed})
    points = []
    for index, value in sorted(enumerate(axis["values"]), key=lambda item: item[1]):
        rows = [row for row in trials if row["point_index"] == index]
        counts = {status: sum(row["observation"]["status"] == status for row in rows) for status in sorted(_STATUSES)}
        unstable = counts["COMPLETED"] > 0 and counts["OOM"] > 0
        classification = ("UNSTABLE" if unstable else "SUCCESS" if counts["COMPLETED"] == repeats else
                          "OOM" if counts["OOM"] == repeats else "CENSORED")
        points.append({"value": value, "point_index": index, "repeats": repeats, "counts": counts,
            "classification": classification, "unstable": unstable,
            "censored": any(not row["observation"]["eligible_for_bracket"] for row in rows)})
    nonmonotone = any(left["counts"]["OOM"] and right["counts"]["COMPLETED"]
                      for index, left in enumerate(points) for right in points[index + 1:])
    brackets = [{"success_value": left["value"], "oom_value": right["value"],
        "resolution": right["value"] - left["value"], "exact_threshold": None}
        for left, right in zip(points, points[1:])
        if left["classification"] == "SUCCESS" and right["classification"] == "OOM"]
    censored = any(point["censored"] for point in points)
    unstable = any(point["unstable"] for point in points)
    status = ("NONMONOTONE" if nonmonotone else "UNSTABLE" if unstable else
              "CENSORED" if censored else "BRACKETED" if brackets else "UNBRACKETED")
    return {"schema_version": 1, "kind": "memory_capacity_validation", "status": "COMPLETED",
        "dataset_id": dataset["dataset_id"], "protocol": PROTOCOL, "source": deepcopy(dataset["source"]),
        "axis": deepcopy(axis), "points": points, "trials": trials,
        "summary": {"trial_count": len(trials), "typed_oom_count": sum(row["observation"]["status"] == "OOM" for row in trials),
            "completed_count": sum(row["observation"]["status"] == "COMPLETED" for row in trials),
            "censored_count": sum(not row["observation"]["eligible_for_bracket"] for row in trials)},
        "threshold_status": status, "observed_brackets": brackets, "exact_threshold": None,
        "nonmonotone": nonmonotone, "unstable": unstable, "censored": censored,
        "physical_confusion": confusion, "profile_input_hashes": [{key: row[key] for key in
            ("field", "index", "sha256", "source_kind")} for row in dataset["profile_inputs"]],
        "hardware_validated": False, "limitations": [
            "Each observed trial owns a new Runtime process; allocator counters are raw, not Workspace.",
            "request_count changes the request cohort, not a promised executed GPU batch.",
            "Only scoped completed/typed-OOM trials form adjacent observed brackets; monotonicity is not assumed.",
            "Admission limits, missing calibration domains, timeouts, startup failures and unverified evidence are censored.",
            "No exact capacity threshold or general hardware prediction validity is asserted."]}


def _control_observation(config: dict, source: dict, status: str, code: str, message: str) -> dict:
    return {"schema_version": 1, "kind": _CAPTURE, "protocol": PROTOCOL,
        **{key: deepcopy(config[key]) for key in ("session_id", "worker_id", "point_index", "repeat", "axis", "effective_config_sha256")},
        "source": deepcopy(source), "status": status, "failure_phase": "startup", "exception_class": None,
        "typed_oom": False, "stack": {}, "devices": [], "metadata": None,
        "allocator": {"before": [], "after": [], "errors": []}, "error_code": code, "message": message,
        "result": None, "cleanup_status": "NOT_CREATED", "elapsed_ms": 0, "kv_pool_bindings": [], "isolation": "owned_process",
        "oom_allocator_evidence": {"phase": None, "per_device": []}, "capacity_oom_evidence": None}


def collect_memory_capacity(experiment: Experiment, plan: dict, inventory: dict | None = None, *,
                            adapter_entry_point: str | None = None, base_dir: str | Path | None = None,
                            allow_test_fixture: bool = False, runner: Any = None) -> dict:
    """Collect independent capacity points; CPU/raw paths never become GPU timing."""
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be boolean")
    experiment = experiment_from_dict(experiment_dict(experiment))
    settings = normalize_capacity_plan(experiment, plan)
    root = Path.cwd() if base_dir is None else Path(base_dir)
    if "raw_path" in settings:
        value, input_bytes = _read(root / settings["raw_path"], settings["max_output_bytes"])
        dataset = validate_capacity_dataset(value, allow_test_fixture=allow_test_fixture)
        if (canonical_json(dataset["axis"]) != canonical_json(settings["axis"])
                or dataset["repeats"] != settings["repeats"]):
            _fail("Imported capacity sweep differs from its requested plan")
        imported_base = experiment_from_dict(dataset["base_experiment"])
        selected_inputs = (dataset["profile_inputs"] if canonical_json(experiment_dict(experiment)) ==
            canonical_json(dataset["base_experiment"]) else _profile_inputs(experiment))
        if _semantic_config(experiment, selected_inputs) != _semantic_config(imported_base, dataset["profile_inputs"]):
            _fail("Imported capacity target differs from the requested experiment", "CALIBRATION_MISMATCH")
        if settings["source"]["kind"] == "test_fixture" and dataset["source"]["kind"] != "test_fixture":
            dataset["source"]["kind"] = "test_fixture"
            _source(dataset["source"], allow_test_fixture)
        report = validate_memory_capacity(dataset, allow_test_fixture=allow_test_fixture)
        return {"schema_version": 1, "kind": "memory_capacity_collection", "status": "COMPLETED",
            "source": dataset["source"], "experiment": dataset["base_experiment"], "dataset": dataset, "report": report,
            "raw_records": [{"path": str((root / settings["raw_path"]).resolve()),
                "sha256": hashlib.sha256(input_bytes).hexdigest(), "bytes": len(input_bytes), "capture": value}],
            "inventory": None, "hardware_validated": False}
    if runner is not None and (not callable(runner) or not allow_test_fixture):
        _fail("Injected capacity runners require explicit fixture permission", "TEST_FIXTURE_NOT_ALLOWED")
    if not isinstance(adapter_entry_point, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*", adapter_entry_point):
        _fail("Live capacity collection requires an explicit module:factory adapter")
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        if os.environ.get(name) not in (None, "", "1"):
            _fail("Capacity sweep must launch outside an existing distributed job")
    source = {"kind": "test_fixture" if runner is not None else "measured", "reference": settings["source_reference"]}
    effective_base = experiment
    if runner is None:
        inventory = probe_rocm() if inventory is None else deepcopy(inventory)
        if inventory.get("status") != "READY":
            source["kind"] = inventory.get("source", {}).get("kind", "measured")
        else:
            effective_base = bind_inventory(experiment, inventory, runtime_name=experiment.calibration.runtime_name,
                runtime_version=experiment.calibration.runtime_version,
                require_rccl=experiment.strategy.tensor_parallel_size > 1 or experiment.strategy.pipeline_parallel_size > 1,
                allow_test_fixture=False)
    inputs = _profile_inputs(effective_base, settings["max_output_bytes"])
    if any(row["source_kind"] == "test_fixture" for row in inputs):
        source["kind"] = "test_fixture"
        _source(source, allow_test_fixture)
    from .pp_benchmark import _run_bounded
    session = uuid.uuid4().hex
    raw_records: list[dict] = []
    trials: list[dict] = []
    limit = settings["max_output_bytes"]
    used_bytes = len(canonical_json({"base_experiment": experiment_dict(effective_base), "profile_inputs": inputs,
        "axis": settings["axis"], "dataset_id": settings["dataset_id"], "source": source}).encode())

    def budget_failure() -> NoReturn:
        raise CollectionError("COLLECTION_OUTPUT_LIMIT", "Capacity sweep exceeds max_output_bytes",
            artifacts={"raw_records": deepcopy(raw_records), "partial_trials": deepcopy(trials),
                       "hardware_validated": False})

    if used_bytes >= limit:
        budget_failure()
    with tempfile.TemporaryDirectory(prefix="mlsys-capacity-") as temporary:
        directory = Path(temporary)
        effective_base = _bind_inputs(effective_base, inputs, directory / "profiles") if inputs else effective_base
        for point, value in enumerate(settings["axis"]["values"]):
            current = capacity_experiment(effective_base, settings["axis"]["name"], value)
            for repeat in range(settings["repeats"]):
                identifier = uuid.uuid4().hex
                config = {"schema_version": 1, "kind": "memory_capacity_trial_config", "protocol": PROTOCOL,
                    "session_id": session, "worker_id": identifier, "point_index": point, "repeat": repeat,
                    "axis": {"name": settings["axis"]["name"], "value": value}, "experiment": experiment_dict(current),
                    "effective_config_sha256": config_sha256(current), "adapter_entry_point": adapter_entry_point,
                    "source_reference": settings["source_reference"], "profile_input_hashes": [{key: row[key] for key in
                        ("field", "index", "sha256", "source_kind")} for row in inputs]}
                trial_dir = directory / identifier
                trial_dir.mkdir()
                config_path = trial_dir / "config.json"
                config_text = canonical_json(config)
                used_bytes += len(config_text.encode()) + 2048
                if used_bytes >= limit:
                    budget_failure()
                config_path.write_text(config_text, encoding="utf-8")
                output = trial_dir / "captures"
                output.mkdir()
                argv = [sys.executable, "-m", "mlsys_sim.capacity_worker", "--config", str(config_path), "--output-dir", str(output)]
                execute = _run_bounded if runner is None else runner
                try:
                    raw = execute(argv, timeout_s=settings["timeout_s"], max_output_bytes=max(1, (limit - used_bytes) // 3),
                                  env=dict(os.environ), output_dir=output)
                    raw_size = len(canonical_json(raw).encode())
                    if used_bytes + 2 * raw_size >= limit:
                        budget_failure()
                    used_bytes += 2 * raw_size
                    if not isinstance(raw, dict):
                        _fail("Capacity runner returned no result")
                    if raw.get("timed_out"):
                        observation = _control_observation(config, source, "TIMEOUT", "COLLECTION_TIMEOUT", "Fresh Runtime trial timed out")
                    elif raw.get("output_truncated"):
                        observation = _control_observation(config, source, "ERROR", "COLLECTION_OUTPUT_LIMIT", "Capacity output was truncated")
                    elif type(raw.get("returncode")) is not int or raw["returncode"] != 0:
                        observation = _control_observation(config, source, "ERROR", "WORKER_PROCESS_FAILED", "Capacity worker process exited abnormally")
                    else:
                        captures = raw.get("rank_captures", [])
                        observation = captures[0] if isinstance(captures, list) and len(captures) == 1 else _control_observation(
                            config, source, "ERROR", "MISSING_WORKER_CAPTURE", "Fresh Runtime did not export one complete trial")
                    if runner is not None and isinstance(observation, dict):
                        observation = deepcopy(observation)
                        observation["source"] = dict(source)
                    raw_records.append({"worker_id": identifier, "capture": raw})
                except CollectionError:
                    raise
                except (OSError, ValueError, TypeError, RuntimeError, SimulationError) as exc:
                    observation = _control_observation(config, source, "ERROR", "COLLECTION_FAILED", str(exc))
                    raw_records.append({"worker_id": identifier, "error": str(exc)})
                trials.append({"point_index": point, "repeat": repeat, "worker_id": identifier,
                    "axis_name": settings["axis"]["name"], "axis_value": value,
                    "effective_experiment": experiment_dict(current), "effective_config_sha256": config["effective_config_sha256"],
                    "observation": observation, "prediction": {}})
        dataset = validate_capacity_dataset({"schema_version": 1, "kind": _DATASET, "protocol": PROTOCOL,
            "dataset_id": settings["dataset_id"], "source": source, "session_id": session,
            "base_experiment": experiment_dict(effective_base), "axis": settings["axis"], "repeats": settings["repeats"],
            "profile_inputs": inputs, "trials": trials,
            "notes": {"request_count_is_executed_batch": False, "allocator_values_are_workspace": False,
                "prediction_uses_full_simulation": True}}, allow_test_fixture=allow_test_fixture)
    if len(canonical_json(dataset).encode()) > settings["max_output_bytes"]:
        raise CollectionError("COLLECTION_OUTPUT_LIMIT", "Capacity dataset exceeds max_output_bytes",
                              artifacts={"raw_records": raw_records, "hardware_validated": False})
    report = validate_memory_capacity(dataset, allow_test_fixture=allow_test_fixture)
    return {"schema_version": 1, "kind": "memory_capacity_collection", "status": "COMPLETED",
        "source": dataset["source"], "experiment": dataset["base_experiment"], "inventory": deepcopy(inventory),
        "dataset": dataset, "report": report, "raw_records": raw_records, "hardware_validated": False}
