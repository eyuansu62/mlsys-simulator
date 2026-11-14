"""Versioned Agent data adapter over the existing serving and causal engines."""

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import sys

from .agent_catalog import (MAX_REPLICAS_PER_DEPLOYMENT, get_agent_catalog,
                            resolve_agent_device, resolve_agent_model, validate_agent_catalog_version)
from .agent_contract import validate_agent_request
from .dag import simulate_dag
from .engine import simulate
from .memory import MemoryPlanner
from .schema import (CalibrationSpec, ExecutionSpec, Experiment, InfraSpec, MemoryConfig,
                     RequestSpec, SimulationError, StrategySpec)
from .sequential import simulate_sequential
from .serialization import experiment_dict


PROTOCOL = "mlab.simulator.v1"
MAX_AGENT_BODY_BYTES = 2 * 1024 * 1024
MAX_AGENT_WALL_S = 1800
MAX_AGENT_SIMULATED_MS = 86_400_000
MAX_AGENT_EVENTS = 1_000_000
MAX_ACTIVE_REQUESTS = 1024
BATCH_TOKEN_BUDGET = 65536
RUNTIME_POLICY = "agent-analytical-v1"
GIB = 1024**3
MODELED_INITIAL_FAILURES = {"STATIC_MEMORY_INFEASIBLE", "AUTO_KV_INFEASIBLE"}


def _invalid(field: str, message: str, code: str = "UNSUPPORTED_AGENT_CONFIGURATION") -> None:
    raise SimulationError(code, f"{field}: {message}")


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _invalid(key, "duplicate JSON field", "INVALID_AGENT_JSON")
        result[key] = value
    return result


def _constant(value: str) -> None:
    _invalid("body", f"non-finite JSON constant {value}", "INVALID_AGENT_JSON")


def decode_agent_request(raw: bytes) -> dict:
    """Read one exact UTF-8 body; do not reserialize bytes used for lineage."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_AGENT_BODY_BYTES:
        _invalid("body", "request body exceeds the published byte limit", "INVALID_AGENT_JSON")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_constant)
        # Reject escaped lone surrogates before they reach HTTP UTF-8 encoding.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SimulationError("INVALID_AGENT_JSON", "body: expected one finite UTF-8 JSON object") from exc
    return validate_agent_request(value)


def _serving(request: dict) -> dict:
    return {"max_concurrency": 1, "gpu_memory_utilization": 0.8, "kv_fraction": None,
            "kv_budget_gib": None, "chunked_prefill_tokens": None, "continuous_batching": False,
            **request["strategy"].get("serving", {})}


def _device_map(request: dict) -> dict:
    return {device["id"]: resolve_agent_device(device["hardware"], device["id"])
            for node in request["infra"]["nodes"] for device in node["devices"]}


def validate_agent_support(request: dict) -> None:
    """Reject unconfirmed identities and unsupported policies without fallback."""
    validate_agent_catalog_version(request["catalog_version"])
    _device_map(request)
    if request["execution"]["max_simulated_ms"] > MAX_AGENT_SIMULATED_MS:
        _invalid("execution.max_simulated_ms", f"service limit is {MAX_AGENT_SIMULATED_MS}")
    if request["evaluation"]["timeout_s"] > MAX_AGENT_WALL_S:
        _invalid("evaluation.timeout_s", f"service limit is {MAX_AGENT_WALL_S} seconds")
    strategy = request["strategy"]
    serving = _serving(request)
    if serving["max_concurrency"] > MAX_ACTIVE_REQUESTS:
        _invalid("strategy.serving.max_concurrency", f"service limit is {MAX_ACTIVE_REQUESTS}")
    if not serving["continuous_batching"] and serving["max_concurrency"] != 1:
        _invalid("strategy.serving", "non-continuous execution requires max_concurrency=1")
    if serving["chunked_prefill_tokens"] is not None and serving["chunked_prefill_tokens"] > BATCH_TOKEN_BUDGET:
        _invalid("strategy.serving.chunked_prefill_tokens", "chunk exceeds published iteration token budget")
    if strategy.get("cache", {}).get("prefix_cache", False):
        _invalid("strategy.cache.prefix_cache", "prefix_group is a label, not token-identical prefix evidence")
    if strategy.get("speculative", False):
        _invalid("strategy.speculative", "speculative decoding has no supported catalog policy")
    scheduling = strategy.get("scheduling", {})
    if scheduling.get("priority", "fifo") != "fifo":
        _invalid("strategy.scheduling.priority", "only FIFO priority is supported")
    if scheduling.get("preemption", "none") != "none":
        _invalid("strategy.scheduling.preemption", "first protocol supports explicit no-preemption only")
    for deployment in strategy["deployments"]:
        resolve_agent_model(deployment["model"], deployment["weight_precision"],
                            deployment.get("kv_precision", "BF16"), deployment["context_tokens"])
        parallel = {"kind": "none", "degree": 1, **deployment.get("parallel", {})}
        if parallel["kind"] not in {"none", "tp"} or parallel["degree"] != 1:
            _invalid("strategy.deployments[].parallel", "no confirmed inter-device topology in this catalog",
                     "UNCONFIRMED_TOPOLOGY")
        if deployment.get("replicas", 1) > MAX_REPLICAS_PER_DEPLOYMENT:
            _invalid("strategy.deployments[].replicas", "replica count exceeds catalog limit")
    if request["workload"]["mode"] == "sequential" and len(strategy["deployments"]) != 1:
        _invalid("strategy.deployments", "sequential first integration requires one deployment with its replicas")


def _pool(request: dict, deployment: dict, devices: dict) -> Experiment:
    serving = _serving(request)
    model = resolve_agent_model(deployment["model"], deployment["weight_precision"],
                                deployment.get("kv_precision", "BF16"), deployment["context_tokens"])
    chosen = tuple(devices[name] for name in deployment["devices"])
    memory = MemoryConfig(kv_pool_bytes=1, workspace_base_bytes=64 * 1024**2,
                          workspace_bytes_per_token=1024, runtime_bytes=128 * 1024**2,
                          safety_margin_bytes=256 * 1024**2,
                          gpu_memory_utilization=serving["gpu_memory_utilization"])
    strategy = StrategySpec(model=model, replicas=deployment.get("replicas", 1),
        scheduler="continuous_batching" if serving["continuous_batching"] else "fifo",
        max_batch_size=serving["max_concurrency"], max_active_requests=serving["max_concurrency"],
        max_batch_tokens=BATCH_TOKEN_BUDGET, preemption="none", memory=memory,
        chunked_prefill=serving["chunked_prefill_tokens"] is not None,
        prefill_chunk_tokens=serving["chunked_prefill_tokens"] or 512, mixed_batching=False,
        prefix_cache=False, replica_routing="least_loaded")
    experiment = Experiment(infra=InfraSpec(devices=chosen), strategy=strategy,
        calibration=CalibrationSpec(backend="analytical", profile_version=RUNTIME_POLICY),
        execution=ExecutionSpec(seed=request["execution"]["seed"], max_events=MAX_AGENT_EVENTS,
            max_simulated_ms=request["execution"]["max_simulated_ms"], detailed_events=True))
    if serving["kv_budget_gib"] is not None:
        fraction = Fraction(str(serving["kv_budget_gib"])) * GIB
        pool = fraction.numerator // fraction.denominator
    elif serving["kv_fraction"] is not None:
        usable = Fraction(str(serving["gpu_memory_utilization"])) * chosen[0].memory_bytes
        usable_bytes = usable.numerator // usable.denominator
        fraction = Fraction(str(serving["kv_fraction"])) * usable_bytes
        pool = fraction.numerator // fraction.denominator
    else:
        # Probe one byte of KV, then reserve fixed costs plus the published worst
        # iteration Workspace. This policy never uses future output lengths.
        local = replace(experiment, infra=replace(experiment.infra, devices=(chosen[0],)),
                        strategy=replace(strategy, replicas=1))
        rank = MemoryPlanner(local).snapshot()[0]
        remaining = (rank["usable_budget_bytes"] - (rank["fixed_budget_bytes"] - 1)
                     - memory.workspace_base_bytes - BATCH_TOKEN_BUDGET * memory.workspace_bytes_per_token)
        pool = remaining // rank["kv_block_bytes"] * rank["kv_block_bytes"]
        if pool < rank["kv_block_bytes"]:
            raise SimulationError("AUTO_KV_INFEASIBLE", "Declared usable memory cannot hold one KV block")
    if not 1 <= pool <= 2**63 - 1:
        _invalid("strategy.serving", "declared KV budget must represent 1 to 2**63-1 bytes")
    return replace(experiment, strategy=replace(strategy, memory=replace(memory, kv_pool_bytes=pool)))


def _request(row: dict, when: float) -> RequestSpec:
    return RequestSpec(row["request_id"], when, row["input_tokens"], row["output_tokens_target"])


def _starts(events: list[dict]) -> dict[str, float]:
    result: dict[str, float] = {}
    for event in events:
        if event["event"] == "STEP_START":
            for name in event["request_ids"]:
                result.setdefault(name, event["timestamp_ms"])
    return result


def _execute(request: dict, pools: dict[str, Experiment]) -> tuple[list, list, dict, float, str | None]:
    sources = request["workload"]["requests"]
    mode = request["workload"]["mode"]
    if mode == "sequential":
        experiment = next(iter(pools.values()))
        outcome = simulate_sequential(experiment, tuple(_request(row, 0) for row in sources))
        run = outcome["run"]
        return run.requests, run.result["memory"], outcome["first_compute_start_ms"], outcome["elapsed_ms"], outcome["stop_reason"]
    if mode == "open_loop" and len(pools) == 1:
        experiment = next(iter(pools.values()))
        run = simulate(replace(experiment, workload=tuple(_request(row, row["arrival_ms"]) for row in sources)))
        return run.requests, run.result["memory"], _starts(run.events), run.result["simulated_ms"], run.result.get("error_code")
    single = next(iter(pools)) if len(pools) == 1 else None
    nodes = []
    for row in sources:
        nodes.append({"node_id": row["request_id"], "model_id": row.get("target_model") or single,
            "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens_target"],
            "dependencies": row.get("depends_on", []), "release_ms": row.get("arrival_ms") or 0,
            "gap_after_deps_ms": row.get("gap_after_deps_ms", 0)})
    result = simulate_dag({"schema_version": 1, "nodes": nodes,
        "pools": [{"model_id": model, "experiment": experiment_dict(exp)} for model, exp in pools.items()],
        "trace_source": request["workload"]["source_sha256"],
        "execution": {"seed": request["execution"]["seed"], "max_events": MAX_AGENT_EVENTS,
                      "max_simulated_ms": request["execution"]["max_simulated_ms"], "detailed_events": True}})
    starts = {row["request_id"]: row["first_compute_start_ms"] for row in result["nodes"]
              if row["first_compute_start_ms"] is not None}
    return result["nodes"], result["devices"], starts, result["elapsed_ms"], result["error_code"]


def _record(source: dict, raw: dict | None, request: dict, started: dict, elapsed: float,
            stop_reason: str | None) -> dict:
    name = source["request_id"]
    model = source.get("target_model") or request["strategy"]["deployments"][0]["model"]
    deployment = next(row for row in request["strategy"]["deployments"] if row["model"] == model)
    record = {"request_id": name, "status": "not_started", "ready_ms": None,
        "first_token_ms": None, "finished_ms": None, "output_tokens": 0,
        "output_tokens_target": source["output_tokens_target"], "started_ms": None,
        "service_ms": None, "deployment_model": model, "replica_index": None,
        "device_ids": [], "failure_code": stop_reason or "DEPLOYMENT_NOT_STARTED", "blocked_by": []}
    if raw is None:
        return record
    if raw["status"] == "BLOCKED_DEPENDENCY":
        record.update(status="blocked", failure_code="DEPENDENCY_FAILED", blocked_by=raw["blocked_by"])
        return record
    if raw.get("arrival_ms") is None or raw["status"] == "NOT_ADMITTED" or raw["arrival_ms"] > request["execution"]["max_simulated_ms"]:
        return record
    count = raw["output_tokens_actual"]
    native = raw["status"]
    status = {"COMPLETED": "completed", "FAILED": "failed", "TIMED_OUT": "timed_out"}.get(native)
    reason = raw.get("error_code") or stop_reason or "INCOMPLETE_EXECUTION"
    if status is None:
        status = "timed_out" if "TIME_LIMIT" in reason else "truncated" if count else "failed"
    owner = raw.get("replica_index", 0 if deployment.get("replicas", 1) == 1 else None)
    record.update(status=status, ready_ms=raw["arrival_ms"], first_token_ms=raw["first_token_ms"],
        finished_ms=raw["completion_ms"] if raw["completion_ms"] is not None else elapsed,
        output_tokens=count, started_ms=started.get(name), replica_index=owner,
        device_ids=[deployment["devices"][owner]] if owner is not None else [],
        failure_code=None if status == "completed" else reason)
    return record


def simulate_agent_request(raw: bytes) -> dict:
    """Execute exact offered rows; transport/input errors raise, modeled OOM returns a run."""
    from . import __version__
    request = decode_agent_request(raw)
    validate_agent_support(request)
    devices = _device_map(request)
    logs = [f"runtime_policy={RUNTIME_POLICY}; max_batch_tokens={BATCH_TOKEN_BUDGET}; "
            "max_concurrency is a per-replica active admission limit; preemption=none",
            "duration_s is client metric metadata; simulated deadline and worker wall timeout are distinct",
            "model/hardware are catalog-declared analytical inputs, not GPU measurements"]
    pools: dict[str, Experiment] = {}
    initial_error = None
    try:
        for deployment in request["strategy"]["deployments"]:
            pools[deployment["model"]] = _pool(request, deployment, devices)
        records, memory, starts, elapsed, stop_reason = _execute(request, pools)
    except SimulationError as error:
        if error.code not in MODELED_INITIAL_FAILURES:
            raise
        initial_error = error
        records, memory, starts, elapsed, stop_reason = [], [], {}, 0.0, error.code
    by_id = {row["request_id"]: row for row in records}
    rows = [_record(source, by_id.get(source["request_id"]), request, starts, elapsed, stop_reason)
            for source in request["workload"]["requests"]]
    completed = all(row["status"] == "completed" for row in rows)
    timeout = any(row["status"] == "timed_out" for row in rows) or stop_reason == "SIMULATION_TIME_LIMIT"
    status = "succeeded" if completed and not stop_reason else "timed_out" if timeout else "failed"
    code = stop_reason or next((row["failure_code"] for row in rows if row["failure_code"]), "EXECUTION_FAILED")
    unsupported = {"requests.service_ms": "No independently defined per-request service-only observation; queue/dependency waits are not service time."}
    ranks = {row["device_id"]: row for row in memory}
    per_device = []
    used = {name for deployment in request["strategy"]["deployments"] for name in deployment["devices"]}
    for name, device in devices.items():
        if name not in used:
            continue
        rank = ranks.get(name)
        per_device.append({"device_id": name, "capacity_bytes": device.memory_bytes,
            "physical_peak_bytes": rank["physical_peak_bytes"] if rank else None,
            "kv_peak_bytes": rank["peak_used_blocks"] * rank["kv_block_bytes"] if rank else None,
            "kv_pool_reserved_bytes": rank["kv_pool_reserved_bytes"] if rank else None})
    if not memory:
        for field in ("physical_peak_bytes", "kv_peak_bytes", "kv_pool_reserved_bytes"):
            unsupported["per_device." + field] = "Deployment initialization failed before an allocation ledger existed."
    if any(row["status"] == "completed" and row["started_ms"] is None for row in rows):
        unsupported["requests.started_ms"] = "No observed first compute start for this record."
    for model, exp in pools.items():
        logs.append(f"{model}: kv_pool_reserved_bytes_per_rank={exp.strategy.memory.kv_pool_bytes}; "
                    "auto KV, when selected, reserves fixed costs and worst iteration Workspace before allocating whole KV blocks")
    return {"schema": PROTOCOL, "client_request_id": request["client_request_id"],
        "input_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "status": status,
        "failure_class": None if status == "succeeded" else "modeled_oom" if initial_error else "simulation_limit" if timeout else "request_failure",
        "error": None if status == "succeeded" else {"code": code, "message": str(initial_error) if initial_error else "One or more offered requests did not complete", "field": None},
        "evidence_level": "simulated", "backend": {"name": "mlsys-simulator", "version": __version__},
        "catalog_version": request["catalog_version"],
        "model_revisions": {deployment["model"]: resolve_agent_model(deployment["model"], deployment["weight_precision"],
            deployment.get("kv_precision", "BF16"), deployment["context_tokens"]).revision for deployment in request["strategy"]["deployments"]},
        "calibration": get_agent_catalog()["calibration"],
        "effective_config": {"infra": request["infra"], "strategy": request["strategy"]},
        "requests": rows, "per_device": per_device, "unsupported_fields": unsupported, "logs": logs}


def main() -> int:
    try:
        response = simulate_agent_request(sys.stdin.buffer.read(MAX_AGENT_BODY_BYTES + 1))
        packet = {"ok": True, "response": response}
    except SimulationError as error:
        packet = {"ok": False, "error": {"code": error.code, "message": str(error),
                  "field": str(error).split(":", 1)[0] if ":" in str(error) else None}}
    print(json.dumps(packet, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
