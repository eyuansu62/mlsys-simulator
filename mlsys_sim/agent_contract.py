"""Strict, dependency-free validation for the versioned lab interchange contract.

This module validates data and causal evidence only.  It neither resolves a
catalog nor schedules work, and it does not import the learning-lab package.
"""

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, NoReturn

from .schema import SimulationError


SCHEMA = "mlab.simulator.v1"
MAX_REQUESTS = 10_000
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SERVING_DEFAULTS = {
    "max_concurrency": 1, "gpu_memory_utilization": 0.8,
    "kv_fraction": None, "kv_budget_gib": None,
    "chunked_prefill_tokens": None, "continuous_batching": False,
}
_REQUEST_STATUSES = {
    "completed", "failed", "rejected", "timed_out", "cancelled",
    "truncated", "blocked", "not_started", "unknown",
}


class _Check:
    def __init__(self, code: str):
        self.code = code

    def fail(self, path: str, message: str) -> NoReturn:
        raise SimulationError(self.code, f"{path}: {message}")

    def object(self, value: Any, path: str, required: set[str],
               optional: set[str] | frozenset[str] = frozenset()) -> dict:
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            self.fail(path, "must be an object with string keys")
        missing = required - value.keys()
        extra = value.keys() - required - optional
        if missing:
            self.fail(path, "missing fields: " + ", ".join(sorted(missing)))
        if extra:
            self.fail(path, "unknown fields: " + ", ".join(repr(key) for key in sorted(extra)))
        return value

    def string(self, value: Any, path: str, *, nonempty: bool = False) -> str:
        if not isinstance(value, str) or (nonempty and not value.strip()):
            self.fail(path, "must be " + ("nonempty text" if nonempty else "text"))
        return value

    def number(self, value: Any, path: str, minimum: float = 0, *, exclusive: bool = False,
               maximum: float | None = None) -> int | float:
        if type(value) not in (int, float):
            self.fail(path, "must be a finite number, not bool")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if (not finite or value < minimum or (exclusive and value == minimum)
                or (maximum is not None and value > maximum)):
            self.fail(path, "number is non-finite or outside its allowed range")
        return value

    def integer(self, value: Any, path: str, minimum: int = 0, maximum: int | None = None) -> int:
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            self.fail(path, "must be an integer within its allowed range, not bool")
        return value

    def boolean(self, value: Any, path: str) -> bool:
        if type(value) is not bool:
            self.fail(path, "must be a boolean")
        return value

    def enum(self, value: Any, path: str, choices: set[str]) -> str:
        self.string(value, path)
        if value not in choices:
            self.fail(path, "unsupported value; expected one of " + ", ".join(sorted(choices)))
        return value

    def array(self, value: Any, path: str, minimum: int = 0, maximum: int = MAX_REQUESTS) -> list:
        if not isinstance(value, list) or not minimum <= len(value) <= maximum:
            self.fail(path, f"must be an array with {minimum}..{maximum} entries")
        return value

    def strings(self, value: Any, path: str, minimum: int = 0) -> list[str]:
        rows = self.array(value, path, minimum)
        for index, row in enumerate(rows):
            self.string(row, f"{path}[{index}]", nonempty=True)
        return rows

    def unique(self, values: list[str], path: str) -> None:
        if len(values) != len(set(values)):
            self.fail(path, "values must be unique")

    def sha(self, value: Any, path: str) -> None:
        if not isinstance(value, str) or not _SHA.fullmatch(value):
            self.fail(path, "must be sha256: followed by 64 lowercase hexadecimal digits")


def _infra(value: Any, check: _Check, path: str) -> list[str]:
    value = check.object(value, path, {"id", "nodes"})
    check.string(value["id"], path + ".id", nonempty=True)
    names, devices = [], []
    for index, node in enumerate(check.array(value["nodes"], path + ".nodes", 1)):
        node_path = f"{path}.nodes[{index}]"
        node = check.object(node, node_path, {"id", "devices"})
        names.append(check.string(node["id"], node_path + ".id", nonempty=True))
        for rank, device in enumerate(check.array(node["devices"], node_path + ".devices", 1)):
            device_path = f"{node_path}.devices[{rank}]"
            device = check.object(device, device_path, {"id", "hardware"})
            devices.append(check.string(device["id"], device_path + ".id", nonempty=True))
            check.string(device["hardware"], device_path + ".hardware", nonempty=True)
    check.unique(names, path + ".nodes")
    check.unique(devices, path + ".devices")
    return devices


def _strategy(value: Any, check: _Check, path: str) -> dict[str, dict]:
    value = check.object(value, path, {"deployments"}, {"serving", "cache", "scheduling", "speculative"})
    models = {}
    for index, deployment in enumerate(check.array(value["deployments"], path + ".deployments", 1)):
        where = f"{path}.deployments[{index}]"
        deployment = check.object(deployment, where, {"model", "weight_precision", "context_tokens", "devices"},
                                  {"kv_precision", "replicas", "parallel"})
        name = check.string(deployment["model"], where + ".model", nonempty=True)
        if name in models:
            check.fail(where + ".model", "use one deployment per model and its replicas field")
        models[name] = deployment
        check.enum(deployment["weight_precision"], where + ".weight_precision", {"BF16", "FP8", "INT8", "INT4"})
        check.enum(deployment.get("kv_precision", "BF16"), where + ".kv_precision", {"BF16", "FP8", "INT8"})
        check.integer(deployment["context_tokens"], where + ".context_tokens", 1)
        devices = check.strings(deployment["devices"], where + ".devices", 1)
        check.unique(devices, where + ".devices")
        replicas = check.integer(deployment.get("replicas", 1), where + ".replicas", 1)
        parallel = check.object(deployment.get("parallel", {}), where + ".parallel", set(), {"kind", "degree"})
        kind = check.enum(parallel.get("kind", "none"), where + ".parallel.kind", {"none", "tp", "pp"})
        degree = check.integer(parallel.get("degree", 1), where + ".parallel.degree", 1)
        if kind == "none" and degree != 1:
            check.fail(where + ".parallel.degree", "none parallelism requires degree 1")
        if len(devices) != replicas * degree:
            check.fail(where + ".devices", "must contain replicas * degree devices in replica/rank order")
    serving = check.object(value.get("serving", {}), path + ".serving", set(), set(_SERVING_DEFAULTS))
    for name, default in _SERVING_DEFAULTS.items():
        item = serving.get(name, default)
        where = f"{path}.serving.{name}"
        if name == "max_concurrency":
            check.integer(item, where, 1)
        elif name == "continuous_batching":
            check.boolean(item, where)
        elif name == "chunked_prefill_tokens":
            if item is not None:
                check.integer(item, where, 1)
        elif item is not None:
            check.number(item, where, exclusive=True,
                         maximum=1 if name in {"gpu_memory_utilization", "kv_fraction"} else None)
        elif name == "gpu_memory_utilization":
            check.fail(where, "must be a positive utilization, not null")
    if serving.get("kv_fraction") is not None and serving.get("kv_budget_gib") is not None:
        check.fail(path + ".serving", "KV fraction and absolute KV budget are mutually exclusive")
    cache = check.object(value.get("cache", {}), path + ".cache", set(), {"prefix_cache"})
    for key, item in cache.items():
        check.boolean(item, path + ".cache." + key)
    scheduling = check.object(value.get("scheduling", {}), path + ".scheduling", set(), {"priority", "preemption"})
    for key, item in scheduling.items():
        check.string(item, path + ".scheduling." + key)
    check.boolean(value.get("speculative", False), path + ".speculative")
    return models


def validate_agent_request(payload: Any) -> dict:
    """Validate the v1 JSON shape/references without resolving or changing IDs.

    Optional fields retain their original omission.  Defaults are interpreted
    for cross-field checks only; no client body is reserialized or normalized.
    Catalog and service-domain support checks belong to the execution adapter.
    """
    check = _Check("INVALID_AGENT_REQUEST")
    required = {"schema", "client_request_id", "catalog_version", "task", "candidate_id", "workload",
                "infra", "strategy", "evaluation", "execution"}
    value = check.object(payload, "request", required)
    check.enum(value["schema"], "schema", {SCHEMA})
    for name in ("client_request_id", "catalog_version"):
        check.string(value[name], name, nonempty=True)
    for name in ("task", "candidate_id"):
        check.string(value[name], name)
    devices = _infra(value["infra"], check, "infra")
    models = _strategy(value["strategy"], check, "strategy")
    used: set[str] = set()
    for name, deployment in models.items():
        placement = set(deployment["devices"])
        if placement & used or not placement <= set(devices):
            check.fail("strategy.deployments." + name + ".devices", "must use exclusive known devices")
        used.update(placement)
    evaluation = check.object(value["evaluation"], "evaluation",
                              {"constraints", "duration_s", "timeout_s", "goodput_policy"})
    for index, constraint in enumerate(check.array(evaluation["constraints"], "evaluation.constraints")):
        where = f"evaluation.constraints[{index}]"
        constraint = check.object(constraint, where, {"metric", "op"}, {"value", "status", "source", "note"})
        check.string(constraint["metric"], where + ".metric")
        check.enum(constraint["op"], where + ".op", {"<", "<=", ">", ">=", "=="})
        if constraint.get("value") is not None:
            item = constraint["value"]
            # Constraints may have negative finite bounds; their evaluation is lab-owned.
            check.number(item, where + ".value", minimum=-math.inf)
        if "status" in constraint:
            check.string(constraint["status"], where + ".status")
        for name in ("source", "note"):
            if constraint.get(name) is not None:
                check.string(constraint[name], where + "." + name)
    for name in ("duration_s", "timeout_s"):
        check.number(evaluation[name], "evaluation." + name, exclusive=True)
    check.enum(evaluation["goodput_policy"], "evaluation.goodput_policy",
               {"both_thresholds_required", "ttft_only_if_tpot_tbd"})
    execution = check.object(value["execution"], "execution",
                            {"seed", "repeat", "split", "holdout", "initial_state", "warmup", "max_simulated_ms"})
    for name in ("seed", "repeat"):
        check.integer(execution[name], "execution." + name, 0, 2**63 - 1)
    check.enum(execution["split"], "execution.split", {"development", "repeat", "holdout"})
    if execution["holdout"] is not None:
        check.string(execution["holdout"], "execution.holdout")
    check.enum(execution["initial_state"], "execution.initial_state", {"empty"})
    if execution["warmup"] is not None:
        check.fail("execution.warmup", "must be null")
    check.number(execution["max_simulated_ms"], "execution.max_simulated_ms", exclusive=True)
    workload = check.object(value["workload"], "workload", {"mode", "requests", "source_sha256"})
    mode = check.enum(workload["mode"], "workload.mode", {"sequential", "open_loop", "dag"})
    check.sha(workload["source_sha256"], "workload.source_sha256")
    requests = check.array(workload["requests"], "workload.requests", 1)
    by_id = {}
    for index, row in enumerate(requests):
        where = f"workload.requests[{index}]"
        row = check.object(row, where, {"request_id", "input_tokens", "output_tokens_target"},
                           {"arrival_ms", "target_model", "depends_on", "gap_after_deps_ms", "category",
                            "prefix_group", "controller_id", "subagent_id"})
        name = check.string(row["request_id"], where + ".request_id", nonempty=True)
        if name in by_id:
            check.fail(where + ".request_id", "request IDs must be unique")
        by_id[name] = row
        check.integer(row["input_tokens"], where + ".input_tokens", 1, 65_536)
        check.integer(row["output_tokens_target"], where + ".output_tokens_target", 1, 8192)
        if row.get("arrival_ms") is not None:
            check.number(row["arrival_ms"], where + ".arrival_ms")
        gap = check.number(row.get("gap_after_deps_ms", 0), where + ".gap_after_deps_ms")
        parents = check.strings(row.get("depends_on", []), where + ".depends_on")
        check.unique(parents, where + ".depends_on")
        for label in ("target_model", "category", "prefix_group", "controller_id", "subagent_id"):
            if row.get(label) is not None:
                check.string(row[label], where + "." + label)
        if mode != "dag" and (parents or gap):
            check.fail(where, "only DAG mode accepts dependencies or relative gaps")
        if mode == "sequential" and row.get("arrival_ms") is not None:
            check.fail(where + ".arrival_ms", "sequential releases follow prior termination")
        if mode == "open_loop" and row.get("arrival_ms") is None:
            check.fail(where + ".arrival_ms", "open-loop requests require their exact arrival")
        if mode == "dag" and parents and row.get("arrival_ms") is not None:
            check.fail(where + ".arrival_ms", "DAG children use a relative dependency gap")
        if mode == "dag" and not parents and gap:
            check.fail(where + ".gap_after_deps_ms", "DAG roots have no dependency gap")
        target = row.get("target_model") or (next(iter(models)) if len(models) == 1 else None)
        if target not in models:
            check.fail(where + ".target_model", "must resolve to one deployed model")
        if row["input_tokens"] + row["output_tokens_target"] > models[target]["context_tokens"]:
            check.fail(where, "context limit cannot hold the full input plus output target")
    children: dict[str, list[str]] = {name: [] for name in by_id}
    degree = {}
    for name, row in by_id.items():
        parents = row.get("depends_on", [])
        degree[name] = len(parents)
        for parent in parents:
            if parent not in by_id or parent == name:
                check.fail("workload.requests." + name + ".depends_on", "must name distinct existing parents")
            children[parent].append(name)
    pending = deque(name for name, count in degree.items() if not count)
    visited = 0
    while pending:
        parent = pending.popleft()
        visited += 1
        for child in children[parent]:
            degree[child] -= 1
            if degree[child] == 0:
                pending.append(child)
    if visited != len(by_id):
        check.fail("workload.requests", "dependencies must form an acyclic graph")
    return deepcopy(value)


def _normalized_strategy(value: dict) -> dict:
    result = deepcopy(value)
    result["serving"] = {**_SERVING_DEFAULTS, **result.get("serving", {})}
    for name, default in (("cache", {}), ("scheduling", {}), ("speculative", False)):
        result.setdefault(name, default)
    for deployment in result["deployments"]:
        deployment.setdefault("kv_precision", "BF16")
        deployment.setdefault("replicas", 1)
        deployment["parallel"] = {"kind": "none", "degree": 1, **deployment.get("parallel", {})}
    return result


def _record(value: Any, check: _Check, path: str) -> dict:
    value = check.object(value, path,
                         {"request_id", "status", "ready_ms", "first_token_ms", "finished_ms",
                          "output_tokens", "output_tokens_target"},
                         {"started_ms", "service_ms", "deployment_model", "replica_index", "device_ids",
                          "failure_code", "blocked_by"})
    check.string(value["request_id"], path + ".request_id", nonempty=True)
    status = check.enum(value["status"], path + ".status", _REQUEST_STATUSES)
    times = []
    for name in ("ready_ms", "started_ms", "first_token_ms", "finished_ms"):
        if value.get(name) is not None:
            times.append(check.number(value[name], path + "." + name))
    if times != sorted(times):
        check.fail(path, "observed timestamps must be causally ordered")
    if value.get("service_ms") is not None:
        service = check.number(value["service_ms"], path + ".service_ms")
        if (value["ready_ms"] is not None and value["finished_ms"] is not None
                and service > value["finished_ms"] - value["ready_ms"] + 1e-6):
            check.fail(path + ".service_ms", "cannot exceed elapsed time")
    check.integer(value["output_tokens_target"], path + ".output_tokens_target", 1)
    output = value["output_tokens"]
    if output is None:
        if status != "unknown":
            check.fail(path + ".output_tokens", "only unknown observations may omit the actual output count")
    else:
        check.integer(output, path + ".output_tokens", 0, value["output_tokens_target"])
        if output == 0 and value["first_token_ms"] is not None:
            check.fail(path + ".first_token_ms", "zero-output requests cannot have a first token")
        if output > 0 and value["first_token_ms"] is None and status != "unknown":
            check.fail(path + ".first_token_ms", "observed output requires its first token timestamp")
    for name in ("deployment_model", "failure_code"):
        if value.get(name) is not None:
            check.string(value[name], path + "." + name, nonempty=True)
    if value.get("replica_index") is not None:
        check.integer(value["replica_index"], path + ".replica_index")
    devices = check.strings(value.get("device_ids", []), path + ".device_ids")
    causes = check.strings(value.get("blocked_by", []), path + ".blocked_by")
    check.unique(devices, path + ".device_ids")
    check.unique(causes, path + ".blocked_by")
    if status == "completed":
        if output != value["output_tokens_target"] or any(value[name] is None for name in
                                                         ("ready_ms", "first_token_ms", "finished_ms")):
            check.fail(path, "completed requests require full output and timing evidence")
        if value.get("deployment_model") is None or value.get("replica_index") is None or not devices:
            check.fail(path, "completed requests require actual deployment placement")
        if value.get("failure_code") is not None or causes:
            check.fail(path, "completed requests cannot have a failure cause")
    elif not value.get("failure_code"):
        check.fail(path + ".failure_code", "noncompleted requests require a machine-readable reason")
    if status in {"blocked", "not_started"} and (
            output != 0 or any(value.get(name) is not None for name in ("ready_ms", "started_ms", "first_token_ms"))):
        check.fail(path, "unreleased requests cannot have invented activity")
    if status == "blocked" and not causes:
        check.fail(path + ".blocked_by", "blocked requests require unsuccessful ancestor IDs")
    if status != "blocked" and causes:
        check.fail(path + ".blocked_by", "only blocked requests may identify dependency blockers")
    return value


def _string_map(value: Any, check: _Check, path: str) -> dict:
    if not isinstance(value, dict):
        check.fail(path, "must be an object of text values")
    for key, item in value.items():
        check.string(key, path + ".key", nonempty=True)
        check.string(item, path + "." + key, nonempty=True)
    return value


def _unique_json(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON fields")
        result[key] = value
    return result


def _nonfinite_json(value: str) -> NoReturn:
    raise ValueError("non-finite JSON constant")


def validate_agent_response(request_bytes: bytes, response: Any) -> None:
    """Check response shape, original-body identity and cross-record causality.

    Accepts explicitly marked ``mock`` protocol examples as well as simulated
    results; callers must separately require ``simulated`` for live acceptance.
    This validates evidence consistency, never model accuracy or SLA success.
    """
    check = _Check("INVALID_AGENT_RESPONSE")
    if not isinstance(request_bytes, bytes):
        check.fail("request_bytes", "must be the original UTF-8 body bytes")
    try:
        request = validate_agent_request(json.loads(request_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json, parse_constant=_nonfinite_json))
    except (ValueError, UnicodeError, RecursionError) as exc:
        check.fail("request_bytes", f"invalid paired request: {exc}")
    response = check.object(response, "response", {
        "schema", "client_request_id", "input_sha256", "status", "failure_class", "error",
        "evidence_level", "backend", "catalog_version", "model_revisions", "calibration",
        "effective_config", "requests", "per_device", "unsupported_fields", "logs"})
    check.enum(response["schema"], "schema", {SCHEMA})
    check.sha(response["input_sha256"], "input_sha256")
    for name in ("client_request_id", "catalog_version"):
        check.string(response[name], name, nonempty=True)
        if response[name] != request[name]:
            check.fail(name, "does not match the submitted request")
    if response["input_sha256"] != "sha256:" + hashlib.sha256(request_bytes).hexdigest():
        check.fail("input_sha256", "must hash the exact received body bytes")
    status = check.enum(response["status"], "status", {"succeeded", "failed", "timed_out", "cancelled"})
    check.enum(response["evidence_level"], "evidence_level", {"simulated", "mock"})
    identity = check.object(response["backend"], "backend", {"name", "version"})
    for name in identity:
        check.string(identity[name], "backend." + name, nonempty=True)
    revisions = _string_map(response["model_revisions"], check, "model_revisions")
    unsupported = _string_map(response["unsupported_fields"], check, "unsupported_fields")
    for index, row in enumerate(check.array(response["logs"], "logs")):
        check.string(row, f"logs[{index}]")
    if response["failure_class"] is not None:
        check.string(response["failure_class"], "failure_class", nonempty=True)
    error = response["error"]
    if error is not None:
        error = check.object(error, "error", {"code", "message"}, {"field"})
        for name in ("code", "message"):
            check.string(error[name], "error." + name, nonempty=True)
        if error.get("field") is not None:
            check.string(error["field"], "error.field")
    if status == "succeeded" and (error is not None or response["failure_class"] is not None):
        check.fail("status", "succeeded cannot contain a run failure")
    if status != "succeeded" and (error is None or response["failure_class"] is None):
        check.fail("status", "unsuccessful runs require a failure class and structured error")
    calibration = check.object(response["calibration"], "calibration",
                               {"version", "kind", "in_domain", "assumptions"})
    if calibration["version"] is not None:
        check.string(calibration["version"], "calibration.version")
    kind = check.enum(calibration["kind"], "calibration.kind", {"uncalibrated", "calibrated", "extrapolated"})
    if calibration["in_domain"] is not None:
        check.boolean(calibration["in_domain"], "calibration.in_domain")
    if kind == "uncalibrated" and calibration["in_domain"] is not None:
        check.fail("calibration.in_domain", "uncalibrated domain coverage is unknown")
    if kind == "extrapolated" and calibration["in_domain"] is not False:
        check.fail("calibration.in_domain", "extrapolated observations cannot claim in-domain coverage")
    check.strings(calibration["assumptions"], "calibration.assumptions")
    if response["effective_config"] is not None:
        effective = check.object(response["effective_config"], "effective_config", {"infra", "strategy"})
        _infra(effective["infra"], check, "effective_config.infra")
        _strategy(effective["strategy"], check, "effective_config.strategy")
        if (effective["infra"] != request["infra"]
                or _normalized_strategy(effective["strategy"]) != _normalized_strategy(request["strategy"])):
            check.fail("effective_config", "must not silently change the selected configuration")
    elif status == "succeeded":
        check.fail("effective_config", "successful runs must identify their effective configuration")
    deployments = {row["model"]: row for row in request["strategy"]["deployments"]}
    if status == "succeeded" and set(revisions) != set(deployments):
        check.fail("model_revisions", "every deployed model needs its resolved revision identity")
    expected = {row["request_id"]: row for row in request["workload"]["requests"]}
    observed = {}
    for index, row in enumerate(check.array(response["requests"], "requests")):
        row = _record(row, check, f"requests[{index}]")
        if row["request_id"] in observed:
            check.fail(f"requests[{index}].request_id", "duplicate request outcome")
        observed[row["request_id"]] = row
    if set(observed) != set(expected):
        check.fail("requests", "return exactly one outcome per offered request, including failures")
    if status == "succeeded" and any(row["status"] != "completed" for row in observed.values()):
        check.fail("status", "succeeded requires complete observations for all offered requests")
    previous = None
    deadline = request["execution"]["max_simulated_ms"]
    for name, source in expected.items():
        row = observed[name]
        path = "requests." + name
        if row["output_tokens_target"] != source["output_tokens_target"]:
            check.fail(path + ".output_tokens_target", "the forced output target changed")
        for field in ("ready_ms", "started_ms", "first_token_ms", "finished_ms"):
            if row.get(field) is not None and row[field] > deadline + 1e-6:
                check.fail(path + "." + field, "observation exceeds the simulation deadline")
        if row["status"] == "blocked":
            ancestors, pending = set(), list(source.get("depends_on", []))
            while pending:
                parent = pending.pop()
                if parent not in ancestors:
                    ancestors.add(parent)
                    pending.extend(expected[parent].get("depends_on", []))
            if any(parent not in ancestors or observed[parent]["status"] == "completed"
                   for parent in row.get("blocked_by", [])):
                check.fail(path + ".blocked_by", "causes must identify unsuccessful dependency ancestors")
        if row["ready_ms"] is not None:
            mode = request["workload"]["mode"]
            if mode == "sequential":
                ready = 0 if previous is None else observed[previous]["finished_ms"]
            elif mode == "open_loop":
                ready = source["arrival_ms"]
            elif source.get("depends_on"):
                parents = [observed[parent] for parent in source["depends_on"]]
                if any(parent["status"] != "completed" for parent in parents):
                    check.fail(path + ".ready_ms", "a DAG child cannot run after a parent failed")
                ready = max(parent["finished_ms"] for parent in parents) + source.get("gap_after_deps_ms", 0)
            else:
                ready = source.get("arrival_ms") or 0
            if ready is None or abs(row["ready_ms"] - ready) > 1e-6:
                check.fail(path + ".ready_ms", "violates the workload's causal release rule")
        if row["status"] == "completed":
            model = source.get("target_model") or next(iter(deployments))
            deployment = deployments[model]
            degree = deployment.get("parallel", {}).get("degree", 1)
            replica = row["replica_index"]
            start = replica * degree
            if (row["deployment_model"] != model or replica >= deployment.get("replicas", 1)
                    or row.get("device_ids", []) != deployment["devices"][start:start + degree]):
                check.fail(path, "observed placement does not match the selected deployment")
            for field in ("started_ms", "service_ms"):
                if row.get(field) is None and "requests." + field not in unsupported:
                    check.fail(path + "." + field, "missing diagnostics require an unsupported_fields reason")
        previous = name
    available = {device["id"] for node in request["infra"]["nodes"] for device in node["devices"]}
    used = {device for row in deployments.values() for device in row["devices"]}
    device_ids = []
    for index, row in enumerate(check.array(response["per_device"], "per_device")):
        path = f"per_device[{index}]"
        row = check.object(row, path, {"device_id", "capacity_bytes", "physical_peak_bytes",
                                      "kv_peak_bytes", "kv_pool_reserved_bytes"})
        device = check.string(row["device_id"], path + ".device_id", nonempty=True)
        if device not in available or device in device_ids:
            check.fail(path + ".device_id", "must identify a distinct known device")
        device_ids.append(device)
        check.integer(row["capacity_bytes"], path + ".capacity_bytes", 1)
        for field in ("physical_peak_bytes", "kv_peak_bytes", "kv_pool_reserved_bytes"):
            if row[field] is None:
                if "per_device." + field not in unsupported:
                    check.fail(path + "." + field, "unknown memory requires an unsupported_fields reason")
            else:
                check.integer(row[field], path + "." + field)
        if (row["kv_peak_bytes"] is not None and row["kv_pool_reserved_bytes"] is not None
                and row["kv_peak_bytes"] > row["kv_pool_reserved_bytes"]):
            check.fail(path + ".kv_peak_bytes", "logical KV usage cannot exceed its reserved pool")
    if not used <= set(device_ids) and "per_device" not in unsupported:
        check.fail("per_device", "missing device observations need an explicit reason")
