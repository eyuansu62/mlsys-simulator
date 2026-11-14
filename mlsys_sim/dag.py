"""Frozen dependency traces over independently placed heterogeneous model pools.

Only an actually completed parent releases its children.  Neither output
targets nor future arrivals are registered with a serving engine in advance.
"""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from .config import _read, _value, validate_experiment
from .event_driver import EventDriver
from .schema import ExecutionSpec, Experiment, RequestSpec, SimulationError
from .serialization import experiment_dict
from .state import TERMINAL


@dataclass(frozen=True)
class DAGNode:
    node_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    dependencies: tuple[str, ...] = ()
    release_ms: float = 0.0
    priority: int = 0
    timeout_ms: float | None = None
    prompt_token_ids: tuple[int, ...] = ()
    cache_namespace: str = ""
    gap_after_deps_ms: float = 0.0


@dataclass(frozen=True)
class ModelPool:
    model_id: str
    experiment: Experiment
    role: str = "model"


@dataclass(frozen=True)
class DAGCandidate:
    candidate_id: str
    pools: tuple[ModelPool, ...]


@dataclass(frozen=True)
class DAGPlan:
    nodes: tuple[DAGNode, ...]
    pools: tuple[ModelPool, ...] = ()
    candidates: tuple[DAGCandidate, ...] = ()
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    trace_source: str = "user_supplied_trace"
    trace_description: str = ""
    schema_version: int = 1


def _invalid(message: str) -> None:
    raise SimulationError("INVALID_CONFIG", f"dag: {message}")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _request(node: DAGNode, when: float) -> RequestSpec:
    return RequestSpec(node.node_id, when, node.input_tokens, node.output_tokens,
                       node.priority, node.timeout_ms, node.prompt_token_ids, node.cache_namespace)


def _validate_pools(plan: DAGPlan, pools: tuple[ModelPool, ...]) -> None:
    ids = [pool.model_id for pool in pools]
    if not ids or len(set(ids)) != len(ids) or any(not name.strip() for name in ids):
        _invalid("model pools must have unique nonempty model_id values")
    by_id = {pool.model_id: pool for pool in pools}
    devices: set[str] = set()
    for pool in pools:
        exp = pool.experiment
        validate_experiment(exp)
        if not pool.role.strip():
            _invalid("pool role must be nonempty")
        if exp.workload:
            _invalid("pool workload must be empty; frozen requests belong in nodes")
        if exp.evaluation.measurement_window_ms is not None:
            raise SimulationError("UNSUPPORTED_MECHANISM", "Finite DAG runs do not use arrival windows")
        placement = {device.id for device in exp.infra.devices}
        if devices & placement:
            _invalid("device IDs must be exclusive across model pools")
        devices.update(placement)
    for node in plan.nodes:
        if node.model_id not in by_id:
            _invalid(f"node {node.node_id!r} references unknown model_id {node.model_id!r}")
        # Reuse all existing model-specific prompt, token, timeout and TP/PP checks.
        validate_experiment(replace(by_id[node.model_id].experiment,
                                    workload=(_request(node, node.release_ms),)))


def load_dag_plan(value: DAGPlan | dict | str | Path, *, base_dir: str | Path | None = None) -> DAGPlan:
    """Strictly validate the complete graph and every candidate before execution."""
    source = Path(value).resolve() if isinstance(value, (str, Path)) else None
    profile_root = Path(base_dir).resolve() if base_dir is not None else source.parent if source is not None else None
    mapping = _read(source) if source is not None else value
    plan: DAGPlan = _value(mapping, DAGPlan, "dag", parse=True)
    if plan.schema_version != 1:
        _invalid("only schema_version 1 is supported")
    if not plan.trace_source.strip() or not plan.nodes:
        _invalid("trace_source and at least one node are required")
    validate_experiment(Experiment(execution=plan.execution))
    if plan.execution.seed < 0:
        _invalid("execution.seed must be a nonnegative signed 64-bit integer")
    if plan.execution.warmup:
        _invalid("declare model-specific warmup in each pool experiment")
    names = [node.node_id for node in plan.nodes]
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        _invalid("node IDs must be unique and nonempty")
    by_id = {node.node_id: node for node in plan.nodes}
    children: dict[str, list[str]] = {name: [] for name in names}
    degrees = {node.node_id: len(node.dependencies) for node in plan.nodes}
    for node in plan.nodes:
        if node.gap_after_deps_ms < 0:
            _invalid(f"node {node.node_id!r} gap_after_deps_ms must be nonnegative")
        if not node.dependencies and node.gap_after_deps_ms:
            _invalid(f"root node {node.node_id!r} cannot declare a dependency gap")
        if len(set(node.dependencies)) != len(node.dependencies):
            _invalid(f"duplicate dependencies for {node.node_id!r}")
        for parent in node.dependencies:
            if parent not in by_id or parent == node.node_id:
                _invalid(f"invalid dependency {parent!r} for {node.node_id!r}")
            children[parent].append(node.node_id)
    ready = deque(name for name in names if degrees[name] == 0)
    visited = 0
    while ready:
        parent = ready.popleft()
        visited += 1
        for child in children[parent]:
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    if visited != len(names):
        _invalid("dependencies must form an acyclic graph")
    if bool(plan.pools) == bool(plan.candidates):
        _invalid("declare either pools or candidates, exclusively")
    if plan.candidates:
        ids = [candidate.candidate_id for candidate in plan.candidates]
        if len(set(ids)) != len(ids) or any(not name.strip() for name in ids):
            _invalid("candidate IDs must be unique and nonempty")
        for candidate in plan.candidates:
            _validate_pools(plan, candidate.pools)
    else:
        _validate_pools(plan, plan.pools)
    if profile_root is not None:
        def resolve(pool: ModelPool) -> ModelPool:
            calibration = pool.experiment.calibration
            paths: dict[str, Any] = {}
            for name in ("profile_path", "native_pp_communication_profile_path", "workspace_profile_path"):
                value = getattr(calibration, name)
                if value is not None:
                    paths[name] = str((profile_root / value).resolve())
            for name in ("replica_profile_paths", "replica_native_pp_communication_profile_paths",
                         "replica_workspace_profile_paths"):
                paths[name] = tuple(str((profile_root / path).resolve()) for path in getattr(calibration, name))
            return replace(pool, experiment=replace(pool.experiment,
                                                     calibration=replace(calibration, **paths)))
        plan = replace(plan, pools=tuple(resolve(pool) for pool in plan.pools),
                       candidates=tuple(replace(candidate, pools=tuple(resolve(pool) for pool in candidate.pools))
                                        for candidate in plan.candidates))
    return plan


def _node_mapping(node: DAGNode) -> dict:
    """Default-zero gaps preserve the pre-extension trace and replay contract."""
    value = asdict(node)
    if not node.gap_after_deps_ms:
        value.pop("gap_after_deps_ms")
    return value


def _trace_identity(plan: DAGPlan) -> str:
    return _sha({"trace_source": plan.trace_source, "nodes": [_node_mapping(node) for node in plan.nodes]})


def dag_plan_dict(plan: DAGPlan) -> dict:
    """Serialize a replay plan without introducing inactive optional controls."""
    def pool_value(pool: ModelPool) -> dict:
        return {"model_id": pool.model_id, "experiment": experiment_dict(pool.experiment), "role": pool.role}

    return {**asdict(plan), "nodes": [_node_mapping(node) for node in plan.nodes],
            "pools": [pool_value(pool) for pool in plan.pools],
            "candidates": [{"candidate_id": item.candidate_id, "pools": [pool_value(pool) for pool in item.pools]}
                           for item in plan.candidates]}


def _capture_profiles(pools: tuple[ModelPool, ...]) -> dict[str, list[dict]]:
    from .capacity import _profile_inputs
    return {pool.model_id: _profile_inputs(pool.experiment) for pool in pools}


def _pool_identity(pool: ModelPool, inputs: list[dict]) -> dict:
    from .capacity import _semantic_config
    return json.loads(_semantic_config(pool.experiment, inputs))


def _check_profiles(pools: tuple[ModelPool, ...], captured: dict[str, list[dict]]) -> None:
    try:
        current = _capture_profiles(pools)
    except SimulationError as exc:
        raise SimulationError("INPUT_CHANGED", "A captured model-pool Profile became unavailable") from exc
    if current != captured:
        raise SimulationError("INPUT_CHANGED", "A model-pool Profile changed during DAG prediction")


def _restore_origin_paths(value: Any, paths: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _restore_origin_paths(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_origin_paths(item, paths) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_origin_paths(item, paths) for item in value)
    return paths.get(value, value) if isinstance(value, str) else value


def _simulate_captured(plan: DAGPlan, inputs: dict[str, list[dict]]) -> dict:
    """Execute only captured bytes; live paths are evidence checks, never inputs."""
    from .capacity import _bind_inputs

    if not any(inputs.values()):
        return _simulate_bound(plan, inputs, plan.pools)
    with tempfile.TemporaryDirectory(prefix="mlsys-dag-inputs-") as temporary:
        pools = []
        paths: dict[str, str] = {}
        for index, pool in enumerate(plan.pools):
            bound = _bind_inputs(pool.experiment, inputs[pool.model_id], Path(temporary) / f"pool-{index:03d}")
            for item in inputs[pool.model_id]:
                selected = getattr(bound.calibration, item["field"])
                path = selected[item["index"]] if item["field"].endswith("paths") else selected
                Path(path).chmod(0o400)
                paths[path] = item["origin_path"]
            pools.append(replace(pool, experiment=bound))
        report = _simulate_bound(plan, inputs, tuple(pools))
        return _restore_origin_paths(report, paths)


def simulate_dag(value: DAGPlan | dict | str | Path) -> dict:
    """Run one placement on one frozen DAG; return ordinary nested pool reports."""
    plan = load_dag_plan(value)
    if plan.candidates:
        _invalid("simulate_dag expects pools; use run_dag_candidates for candidate plans")
    profile_inputs = _capture_profiles(plan.pools)
    return _simulate_captured(plan, profile_inputs)


def _simulate_bound(plan: DAGPlan, profile_inputs: dict[str, list[dict]],
                    runtime_pools: tuple[ModelPool, ...]) -> dict:
    import heapq
    from .engine import simulate

    nodes = {node.node_id: node for node in plan.nodes}
    order = {node.node_id: index for index, node in enumerate(plan.nodes)}
    children: dict[str, list[str]] = {name: [] for name in nodes}
    remaining = {node.node_id: len(node.dependencies) for node in plan.nodes}
    for node in plan.nodes:
        for parent in node.dependencies:
            children[parent].append(node.node_id)
    drivers: dict[str, EventDriver] = {}
    warmups: dict[str, dict] = {}
    for pool in runtime_pools:
        exp = pool.experiment
        if exp.execution.warmup:
            warmups[pool.model_id] = simulate(exp).result["warmup"]
        else:
            warmups[pool.model_id] = {"enabled": False, "status": "NOT_REQUESTED",
                                      "cache_effect": "none; measured run starts with empty KV"}
        drivers[pool.model_id] = EventDriver(replace(
            exp, execution=replace(exp.execution, warmup=(),
                                   detailed_events=plan.execution.detailed_events)))
    pool_order = {pool.model_id: index for index, pool in enumerate(plan.pools)}
    rows: dict[str, dict] = {
        node.node_id: {"node_id": node.node_id, "request_id": node.node_id,
                       "model_id": node.model_id, "dependencies": list(node.dependencies),
                       "input_tokens": node.input_tokens, "output_tokens_target": node.output_tokens,
                       "priority": node.priority, "timeout_ms": node.timeout_ms,
                       "release_ms": node.release_ms, "ready_ms": None,
                       "dependency_wait_ms": None, "queue_wait_ms": None,
                       "first_compute_start_ms": None, "arrival_ms": None,
                       "completion_ms": None, "status": "PENDING", "error_code": None,
                       "blocked_by": []}
        for node in plan.nodes}
    for node in plan.nodes:
        if node.gap_after_deps_ms:
            rows[node.node_id]["gap_after_deps_ms"] = node.gap_after_deps_ms
    ready_nodes: list[tuple[float, int, str]] = []
    for node in plan.nodes:
        if not node.dependencies:
            rows[node.node_id]["ready_ms"] = node.release_ms
            rows[node.node_id]["dependency_wait_ms"] = 0.0
            heapq.heappush(ready_nodes, (node.release_ms, order[node.node_id], node.node_id))
    accepted: set[str] = set()
    observed_terminal: set[str] = set()
    events: list[dict] = []
    cursors = {name: 0 for name in drivers}
    now = 0.0
    stop_reason: str | None = None

    def collect(model_id: str) -> None:
        driver = drivers[model_id]
        local = driver.events
        events.extend({**event, "model_id": model_id} for event in local[cursors[model_id]:])
        cursors[model_id] = len(local)

    def block_descendants(failed: str) -> None:
        pending = deque(children[failed])
        while pending:
            child = pending.popleft()
            row = rows[child]
            if row["status"] == "BLOCKED_DEPENDENCY":
                if failed not in row["blocked_by"]:
                    row["blocked_by"].append(failed)
                continue
            if child in accepted:
                raise SimulationError("INVALID_DRIVER_STATE", "A child ran before its parents succeeded")
            row.update(status="BLOCKED_DEPENDENCY", error_code="DEPENDENCY_FAILED",
                       blocked_by=[failed])
            pending.extend(children[child])

    def refresh() -> None:
        for model_id, driver in drivers.items():
            if driver.stopped:
                driver.finalize(reason=driver.stop_reason)
            collect(model_id)
        for name in sorted(accepted - observed_terminal, key=order.__getitem__):
            state = drivers[nodes[name].model_id].requests[name]
            if state.status not in TERMINAL:
                continue
            observed_terminal.add(name)
            rows[name].update(status=state.status, completion_ms=state.completion_ms,
                              error_code=state.error_code)
            if state.status != "COMPLETED":
                block_descendants(name)
                continue
            for child in children[name]:
                if rows[child]["status"] == "BLOCKED_DEPENDENCY":
                    continue
                remaining[child] -= 1
                if remaining[child] == 0:
                    node = nodes[child]
                    parent_end = max(rows[parent]["completion_ms"] for parent in node.dependencies)
                    after_parents = parent_end + node.gap_after_deps_ms
                    if (not math.isfinite(after_parents)
                            or node.gap_after_deps_ms > 0 and after_parents <= parent_end):
                        _invalid(f"node {child!r} dependency gap cannot advance finite representable time")
                    ready_ms = max(node.release_ms, after_parents)
                    rows[child].update(ready_ms=ready_ms,
                                       dependency_wait_ms=ready_ms - node.release_ms)
                    heapq.heappush(ready_nodes, (ready_ms, order[child], child))

    def prune_ready() -> None:
        while ready_nodes and rows[ready_nodes[0][2]]["status"] != "PENDING":
            heapq.heappop(ready_nodes)

    def next_item() -> tuple[float, int, int, int, str] | None:
        prune_ready()
        candidates = [(event[0], event[1], pool_order[model_id], event[2], model_id)
                      for model_id, driver in drivers.items()
                      if (event := driver.next_event()) is not None]
        if ready_nodes:
            when, index, name = ready_nodes[0]
            candidates.append((when, 2, len(drivers), index, name))
        return min(candidates) if candidates else None

    def event_count() -> int:
        return sum(driver.handled_events for driver in drivers.values())

    while True:
        item = next_item()
        if item is None:
            break
        when = item[0]
        if when > plan.execution.max_simulated_ms:
            now, stop_reason = plan.execution.max_simulated_ms, "SIMULATION_TIME_LIMIT"
            break
        now = when
        for driver in drivers.values():
            if not driver.stopped:
                driver.set_time(now)
        while (item := next_item()) is not None and item[0] == now:
            boundary = item[2] != len(drivers) and item[3] == -1
            if not boundary and event_count() >= plan.execution.max_events:
                stop_reason = "SIMULATION_EVENT_LIMIT"
                break
            if item[2] == len(drivers):
                _, _, name = heapq.heappop(ready_nodes)
                node, row = nodes[name], rows[name]
                driver = drivers[node.model_id]
                if driver.stopped:
                    row.update(status="UNFINISHED", error_code="MODEL_POOL_UNAVAILABLE")
                    block_descendants(name)
                else:
                    try:
                        driver.accept_request(_request(node, now))
                    except SimulationError as exc:
                        if exc.code not in {"SIMULATION_EVENT_LIMIT", "NO_AVAILABLE_REPLICA"}:
                            raise
                        row.update(status="UNFINISHED", error_code=exc.code)
                        block_descendants(name)
                    else:
                        accepted.add(name)
                        row.update(arrival_ms=now, status="ADMITTED")
            else:
                drivers[item[4]].advance_one()
            refresh()
        if stop_reason is not None:
            break
        if now >= plan.execution.max_simulated_ms:
            if (any(row["status"] == "PENDING" for row in rows.values())
                    or any(driver.has_unfinished() or driver.has_inflight()
                           for driver in drivers.values() if not driver.stopped)):
                stop_reason = "SIMULATION_TIME_LIMIT"
            break
        for driver in drivers.values():
            driver.schedule_ready()
        refresh()

    if stop_reason is None and any(row["status"] in {"PENDING", "ADMITTED"} for row in rows.values()):
        stop_reason = "DAG_NO_PROGRESS"
    pool_reports: list[dict] = []
    devices: list[dict] = []
    request_reports: dict[str, dict] = {}
    for pool in plan.pools:
        driver = drivers[pool.model_id]
        reason = stop_reason if driver.has_unfinished() or driver.has_inflight() else None
        report = driver.finalize(now_ms=now if not driver.stopped else None, reason=reason)
        report.result["warmup"] = warmups[pool.model_id]
        collect(pool.model_id)
        request_reports.update({row["request_id"]: row for row in report.requests})
        pool_reports.append({"model_id": pool.model_id, "role": pool.role,
                             "configuration_sha256": _sha(_pool_identity(pool, profile_inputs[pool.model_id])),
                             "configuration": experiment_dict(pool.experiment),
                             "profile_inputs": profile_inputs[pool.model_id],
                             "result": report.result, "requests": report.requests})
        for replica_index, engine in enumerate(driver.local_engines):
            for rank in engine.memory.snapshot():
                busy = (engine.stage_busy_ms[rank["stage_index"]]
                        if hasattr(engine, "stage_busy_ms") else engine.compute_busy_ms)
                devices.append({**rank, "model_id": pool.model_id, "replica_index": replica_index,
                                "busy_ms": busy, "utilization": busy / now if now else None,
                                "utilization_window_ms": now,
                                "utilization_scope": "modeled_step_or_stage_busy_fraction",
                                "peak_kv_used_bytes": rank["peak_used_blocks"] * rank["kv_block_bytes"]})
    refresh()
    for name, row in rows.items():
        if name in request_reports:
            original = request_reports[name]
            # Preserve DAG metadata while adding the complete ordinary request ledger.
            row.update(original)
            start = drivers[nodes[name].model_id].first_compute_start_ms.get(name)
            row["first_compute_start_ms"] = start
            row["queue_wait_ms"] = start - row["ready_ms"] if start is not None else None
        elif row["status"] == "PENDING":
            row.update(status="UNFINISHED", error_code=stop_reason or "DAG_NO_PROGRESS")
    nodes_completed = all(row["status"] == "COMPLETED" for row in rows.values())
    resources_drained = all(not driver.has_inflight() and not driver.has_unfinished()
                            for driver in drivers.values())
    successful = (nodes_completed and stop_reason is None and resources_drained
                  and all(pool["result"]["status"] == "COMPLETED" for pool in pool_reports))
    makespan = max(row["completion_ms"] for row in rows.values()) if nodes_completed else None
    chain: list[str] = []
    if nodes_completed:
        current = max(nodes, key=lambda name: (rows[name]["completion_ms"], -order[name]))
        while True:
            chain.append(current)
            dependencies = nodes[current].dependencies
            if not dependencies:
                break
            current = max(dependencies, key=lambda name: (rows[name]["completion_ms"], -order[name]))
        chain.reverse()
    counts = {status.lower(): sum(row["status"] == status for row in rows.values())
              for status in ("COMPLETED", "FAILED", "TIMED_OUT", "UNFINISHED", "BLOCKED_DEPENDENCY")}
    _check_profiles(plan.pools, profile_inputs)
    return {
        "kind": "dag", "schema_version": 1, "backend": "simulated",
        "status": "COMPLETED" if successful else "UNRESOLVED",
        "error_code": stop_reason if stop_reason else None if successful else "DAG_NODE_FAILURE",
        "hardware_validated": False, "trace_source": plan.trace_source,
        "trace_description": plan.trace_description, "trace_sha256": _trace_identity(plan),
        "configuration_sha256": _sha({"pools": [
            {"model_id": pool.model_id, "role": pool.role,
             "experiment": _pool_identity(pool, profile_inputs[pool.model_id])} for pool in plan.pools],
                                      "execution": asdict(plan.execution)}),
        "execution": asdict(plan.execution), "events_processed": event_count(),
        "elapsed_ms": now, "makespan_ms": makespan, "nodes_completed": nodes_completed,
        "resources_drained": resources_drained, "counts": {"total": len(nodes), **counts},
        "critical_path": {"nodes": chain, "elapsed_ms": makespan,
                          "definition": "observed_latest_parent_chain_including_resource_and_release_wait",
                          "isolated_service_lower_bound": False,
                          "node_elapsed_ms": [rows[name]["completion_ms"] - rows[name]["ready_ms"]
                                              for name in chain],
                          "release_wait_ms": [rows[name]["ready_ms"] -
                                              (rows[chain[index - 1]]["completion_ms"] if index else 0.0)
                                              for index, name in enumerate(chain)]},
        "frozen_nodes": [_node_mapping(node) for node in plan.nodes],
        "nodes": list(rows.values()), "pools": pool_reports, "devices": devices,
        "events": events if plan.execution.detailed_events else [],
        "limitations": ["model pools own exclusive devices and independent links, host pools and DMA lanes",
                        "dispatch, DAG dependency evaluation and inter-model data transfer have zero modeled latency",
                        "request content and target model are frozen; generated semantic quality is not evaluated",
                        "quantized byte widths are declared analytical assumptions, not quantization kernel validation",
                        "per-device utilization is modeled busy time, not measured GPU occupancy"],
    }


def run_dag_candidates(value: DAGPlan | dict | str | Path) -> dict:
    """Enumerate placements without mutating the frozen requests between runs."""
    plan = load_dag_plan(value)
    if not plan.candidates:
        _invalid("run_dag_candidates expects candidates")
    profiles = {candidate.candidate_id: _capture_profiles(candidate.pools) for candidate in plan.candidates}
    results = []
    for candidate in plan.candidates:
        result = _simulate_captured(replace(plan, pools=candidate.pools, candidates=()),
                                    profiles[candidate.candidate_id])
        result["candidate_id"] = candidate.candidate_id
        results.append(result)
        for item in plan.candidates:
            _check_profiles(item.pools, profiles[item.candidate_id])
    trace_sha = _trace_identity(plan)
    if any(result["trace_sha256"] != trace_sha for result in results):
        raise SimulationError("INVALID_DRIVER_STATE", "Candidate execution changed the frozen trace")
    ranked = sorted((result for result in results if result["status"] == "COMPLETED"),
                    key=lambda result: (result["makespan_ms"], result["candidate_id"]))
    return {"kind": "dag_candidates", "schema_version": 1, "backend": "simulated",
            "hardware_validated": False, "trace_source": plan.trace_source,
            "trace_sha256": trace_sha, "candidates": results,
            "ranking": [{"rank": index + 1, "candidate_id": result["candidate_id"],
                         "makespan_ms": result["makespan_ms"]} for index, result in enumerate(ranked)],
            "ranking_rule": "only_all_nodes_completed_candidates_ordered_by_makespan_then_id"}


def _gantt(result: dict) -> str:
    rows = result["nodes"][:500]
    width, left, span = 1200, 240, max(result["elapsed_ms"], 1.0)
    height = 70 + 24 * len(rows)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
              f'viewBox="0 0 {width} {height}" role="img" aria-label="DAG completion timeline">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<text x="16" y="24" font-family="sans-serif" font-size="16">'
              'Frozen DAG: dependency wait, queue wait and serving time</text>']
    colors = {pool["model_id"]: ("#2563eb" if index % 2 == 0 else "#c2410c")
              for index, pool in enumerate(result["pools"])}
    scale = (width - left - 30) / span
    for index, row in enumerate(rows):
        y = 48 + index * 24
        label = escape(f'{row["node_id"]} / {row["model_id"]}'[:80])
        pieces.append(f'<text x="12" y="{y + 12}" font-family="sans-serif" font-size="11">{label}</text>')
        release, ready = row["release_ms"], row["ready_ms"]
        start, end = row["first_compute_start_ms"], row["completion_ms"]
        if ready is not None and ready > release:
            pieces.append(f'<rect x="{left + release * scale:.3f}" y="{y}" '
                          f'width="{(ready - release) * scale:.3f}" height="16" fill="#e5e7eb"/>')
        if ready is not None and start is not None and start > ready:
            pieces.append(f'<rect x="{left + ready * scale:.3f}" y="{y}" '
                          f'width="{(start - ready) * scale:.3f}" height="16" fill="#fbbf24"/>')
        if start is not None:
            stop = end if end is not None else result["elapsed_ms"]
            pieces.append(f'<rect x="{left + start * scale:.3f}" y="{y}" '
                          f'width="{max(0.5, (stop - start) * scale):.3f}" height="16" '
                          f'fill="{colors[row["model_id"]]}"/>')
    pieces.append(f'<text x="16" y="{height - 8}" font-family="sans-serif" font-size="11">'
                  f'{len(rows)} of {len(result["nodes"])} nodes; horizon {span:.3f} ms; '
                  'gray: dependency wait, amber: initial queue, color: serving elapsed</text></svg>')
    return "\n".join(pieces)


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe = {key: ("'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
                          else value) for key, value in row.items()}
            writer.writerow(safe)


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.3f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th scope='col'>{escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(_display(value))}</td>" for value in row)
                   + "</tr>" for row in rows)
    return f"<div class='table-scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _report_html(result: dict) -> str:
    """Render the saved prediction without introducing another metric contract."""
    candidates = result.get("candidates", [])
    source = result["trace_source"]
    description = result.get("trace_description", candidates[0]["trace_description"] if candidates else "")
    title = "冻结 DAG 候选部署比较" if candidates else "冻结 DAG 与模型池预测"
    parts = [f"<h1>{title}</h1>",
             "<p class='lead'>同一冻结请求图上的因果调度预测。时间单位为 ms。</p>",
             "<aside>所有时间、利用率和 KV 水位均为 CPU 模拟预测；即使使用实测 Profile，"
             "本次运行也不是 W7900D 端到端真机测量。</aside>"]
    if source == "synthetic_teaching_trace":
        parts.append("<aside class='synthetic'><strong>人工教学轨迹。</strong>"
                     "此轨迹不是采集的真实 Claude / claudecode deep-research 轨迹；"
                     "示例中的 32B/7B 几何与硬件性能参数为显式解析假设。</aside>")
    parts.extend([f"<p><strong>轨迹来源：</strong>{escape(source)}</p>",
                  f"<p>{escape(description)}</p>",
                  f"<p class='hash'><strong>Trace SHA256：</strong>{escape(result['trace_sha256'])}</p>",
                  "<p><a href='report.json'>完整 JSON</a></p>"])
    if candidates:
        eligible = {item["candidate_id"] for item in candidates
                    if item["status"] == "COMPLETED" and item["resources_drained"]
                    and item["nodes_completed"] and item["error_code"] is None}
        ranks = {item["candidate_id"]: item["rank"] for item in result["ranking"]
                 if item["candidate_id"] in eligible}
        parts.extend(["<h2>候选摘要</h2>",
                      "<p>仅全部节点完成、资源排空且没有错误的候选进入排名。"
                      "截断结果可以保留任务 makespan，但不获得名次。</p>",
                      "<p><a href='candidates.csv'>候选 CSV</a></p>"])
        table_rows = []
        for item in candidates:
            table_rows.append([item["candidate_id"], ranks.get(item["candidate_id"]), item["status"],
                               item["makespan_ms"], item["elapsed_ms"], item["resources_drained"]])
        parts.append(_table(["候选", "合格名次", "整体状态", "任务 makespan", "观测 / 排空 elapsed", "资源已排空"],
                            table_rows))
        parts.append("<p>任务 makespan 以最后输出 token 为准；elapsed 包含必要的反馈与 DMA 排空。"
                     "二者分开报告，不能把 PP 反馈尾部计入用户任务墙钟。</p>")
        for index, item in enumerate(candidates):
            prefix = f"candidate-{index:03d}"
            parts.extend([f"<section><h2>{escape(item['candidate_id'])}</h2>",
                          f"<p>状态：{escape(item['status'])} · "
                          f"任务 makespan：{escape(_display(item['makespan_ms']))} ms · "
                          f"elapsed：{escape(_display(item['elapsed_ms']))} ms</p>",
                          f"<p><a href='{prefix}/report.html'>节点与设备详情</a> · "
                          f"<a href='{prefix}/gantt.svg'>甘特图 SVG</a> · "
                          f"<a href='{prefix}/nodes.csv'>节点 CSV</a></p>",
                          f"<img class='gantt' loading='lazy' src='{prefix}/gantt.svg' "
                          f"alt='{escape(item['candidate_id'], quote=True)} 的节点时间线'></section>"])
        limitations = list(dict.fromkeys(text for item in candidates for text in item["limitations"]))
    else:
        parts.append(_table(["整体状态", "任务 makespan", "观测 / 排空 elapsed", "节点全部完成", "资源已排空"],
                            [[result["status"], result["makespan_ms"], result["elapsed_ms"],
                              result["nodes_completed"], result["resources_drained"]]]))
        if result["error_code"] is not None:
            parts.append(f"<aside>运行未完整解决：{escape(result['error_code'])}。"
                         "保留的节点完成时间不能用作已成功排空的候选结果。</aside>")
        parts.extend(["<p>任务 makespan 是全部节点最后输出 token 的最晚时刻；elapsed 包含必要资源排空。"
                      "未全部完成的任务 makespan 为 —。依赖等待、初次排队与计算后的服务耗时分别记录。</p>",
                      "<p><a href='nodes.csv'>节点 CSV</a> · <a href='devices.csv'>设备 CSV</a> · "
                      "<a href='gantt.svg'>甘特图 SVG</a> · <a href='replay.plan.json'>离线重放计划</a></p>",
                      "<h2>因果时间线</h2><img class='gantt' src='gantt.svg' alt='冻结 DAG 逐节点甘特图'>",
                      "<p>灰色：依赖等待；琥珀色：初次排队；模型颜色：首次计算至请求完成的服务历时。"
                      "图最多显示 500 个节点，完整节点数据保留在 JSON / CSV 中。</p>",
                      "<h2>逐卡资源摘要</h2>"])
        parts.append(_table(["设备", "模型池", "副本", "PP stage", "TP rank", "busy ms", "利用率", "峰值 KV GiB"],
                            [[item["device_id"], item["model_id"], item["replica_index"], item["stage_index"],
                              item["tp_rank"], item["busy_ms"],
                              f"{100 * item['utilization']:.2f}%" if item["utilization"] is not None else None,
                              item["peak_kv_used_bytes"] / 1024 ** 3] for item in result["devices"]]))
        parts.extend(["<p>利用率是模型组或 PP stage 的 busy 时间除以整个观测 elapsed，"
                      "不是实测 GPU occupancy。</p>", "<h2>逐节点摘要</h2>"])
        parts.append(_table(["节点", "模型池", "状态", "ready", "依赖等待", "初次排队", "完成"],
                            [[item["node_id"], item["model_id"], item["status"], item["ready_ms"],
                              item["dependency_wait_ms"], item["queue_wait_ms"], item["completion_ms"]]
                             for item in result["nodes"]]))
        chain = " → ".join(result["critical_path"]["nodes"])
        parts.extend(["<h2>观测依赖链</h2>", f"<p>{escape(chain) if chain else '—'}</p>",
                      "<p>该链包含资源排队与 release 间隙，不是隔离执行成本的理论下界。</p>"])
        limitations = result["limitations"]
    parts.append("<h2>模型边界</h2><ul>" + "".join(f"<li>{escape(text)}</li>" for text in limitations) + "</ul>")
    styles = """
body{margin:0;background:#f4f6f8;color:#162438;font:15px/1.6 system-ui,sans-serif}
main{max-width:1200px;margin:32px auto;padding:32px;background:white;border-radius:12px}
h1{margin-top:0;font-size:30px}h2{margin-top:32px;font-size:21px}.lead{color:#465a72}
aside{padding:14px 18px;border-left:4px solid #bf7019;background:#fff6e8;margin:16px 0}
.synthetic{border-color:#2563eb;background:#eff6ff}a{color:#174f9c}p{overflow-wrap:anywhere}
.hash{font-size:12px;font-family:monospace}.table-scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3eb}
th{background:#edf2f7;white-space:nowrap}td{overflow-wrap:anywhere;max-width:360px}
.gantt{display:block;width:100%;height:auto;margin:20px 0;border:1px solid #dce3eb}
@media(max-width:700px){main{margin:0;padding:18px;border-radius:0}h1{font-size:24px}}
"""
    return ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{escape(title)}</title><style>{styles}</style></head><body><main>"
            + "\n".join(parts) + "</main></body></html>\n")


def write_dag_report(result: dict, directory: str | Path) -> Path:
    """Write bounded, escaped SVG and complete machine-readable prediction artifacts."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    saved = {**result, "stored_at": datetime.now(timezone.utc).isoformat(),
             "prediction_id": "dag-" + _sha(result)[:24]}
    (path / "report.json").write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if result["kind"] == "dag_candidates":
        summary = [{"candidate_id": item["candidate_id"], "status": item["status"],
                    "makespan_ms": item["makespan_ms"], "elapsed_ms": item["elapsed_ms"],
                    "trace_sha256": item["trace_sha256"]} for item in result["candidates"]]
        _csv(path / "candidates.csv", summary,
             ["candidate_id", "status", "makespan_ms", "elapsed_ms", "trace_sha256"])
        for index, candidate in enumerate(result["candidates"]):
            write_dag_report(candidate, path / f"candidate-{index:03d}")
    else:
        replay_pools = []
        manifest = []
        for pool_index, pool in enumerate(result["pools"]):
            experiment = deepcopy(pool["configuration"])
            for profile_index, item in enumerate(pool["profile_inputs"]):
                relative = Path("inputs") / f"pool-{pool_index:03d}" / f"profile-{profile_index:03d}.json"
                snapshot = path / relative
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(item["raw_json"].encode("utf-8"))
                manifest.append({**{key: value for key, value in item.items() if key != "raw_json"},
                                 "model_id": pool["model_id"], "role": pool["role"],
                                 "snapshot_path": relative.as_posix()})
                calibration = experiment["calibration"]
                if item["field"].endswith("paths"):
                    paths = list(calibration[item["field"]])
                    paths[item["index"]] = relative.as_posix()
                    calibration[item["field"]] = paths
                else:
                    calibration[item["field"]] = relative.as_posix()
            replay_pools.append({"model_id": pool["model_id"], "role": pool["role"],
                                 "experiment": experiment})
        if manifest:
            (path / "inputs" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        replay = {"schema_version": 1, "trace_source": result["trace_source"],
                  "trace_description": result["trace_description"], "nodes": result["frozen_nodes"],
                  "pools": replay_pools, "execution": result["execution"]}
        (path / "replay.plan.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
        _csv(path / "nodes.csv", result["nodes"],
             ["node_id", "model_id", "input_tokens", "output_tokens_target", "status",
              "release_ms", "ready_ms", "dependency_wait_ms",
              "queue_wait_ms", "first_compute_start_ms", "arrival_ms", "first_token_ms",
              "completion_ms", "error_code"])
        _csv(path / "devices.csv", result["devices"],
             ["device_id", "model_id", "replica_index", "stage_index", "tp_rank", "busy_ms",
              "utilization", "peak_kv_used_bytes", "peak_used_blocks", "workspace_peak_bytes"])
        (path / "gantt.svg").write_text(_gantt(result), encoding="utf-8")
    (path / "report.html").write_text(_report_html(result), encoding="utf-8")
    return path
