"""Opt-in joint serving constraints over existing, frozen simulation evidence.

This module evaluates observations. It never runs a simulator, reloads a Profile,
or turns an incomplete measurement into a successful constraint assessment.
"""

from copy import deepcopy
from fractions import Fraction
import math
from typing import Any, NoReturn

from .schema import SimulationError
from .state import SimulationResult


_LATENCY = {"p99_ttft_ms": "ttft", "p99_tpot_ms": "tpot", "p99_e2e_ms": "end_to_end"}
_BASES = {"measurement_window", "elapsed", "task_makespan"}
_RANK_LABELS = ("device_id", "model_id", "replica_index", "stage_index", "tp_rank")


def _invalid(message: str) -> NoReturn:
    raise SimulationError("INVALID_CONSTRAINTS", message)


def _object(value: Any, required: set[str], optional: set[str], label: str) -> dict:
    if (not isinstance(value, dict) or any(not isinstance(key, str) for key in value)
            or required - value.keys() or value.keys() - required - optional):
        _invalid(f"{label} requires {sorted(required)}; optional keys: {sorted(optional)}")
    return value


def _number(value: Any, label: str, *, zero: bool = False) -> float:
    if type(value) not in (int, float):
        _invalid(f"{label} must be a finite {'nonnegative' if zero else 'positive'} number")
    try:
        result = float(value)
    except OverflowError:
        _invalid(f"{label} exceeds finite numeric range")
        raise AssertionError("unreachable")
    if not math.isfinite(result) or (result < 0 if zero else result <= 0):
        _invalid(f"{label} must be a finite {'nonnegative' if zero else 'positive'} number")
    return result


def validate_constraints(spec: dict) -> dict:
    """Return a normalized independent specification, rejecting unknown keys."""
    _object(spec, {"schema_version"}, {"min_completed_samples", "memory", "latency", "throughput", "task"},
            "constraints")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        _invalid("constraints.schema_version must be integer 1")
    minimum = spec.get("min_completed_samples", 100)
    if type(minimum) is not int or not 1 <= minimum <= 1_000_000:
        _invalid("min_completed_samples must be an integer in [1,1000000]")
    if not {"memory", "latency", "throughput", "task"} & spec.keys():
        _invalid("At least one memory, latency, throughput or task constraint is required")
    normalized = deepcopy(spec)
    normalized["min_completed_samples"] = minimum
    if "memory" in spec:
        memory = _object(spec["memory"], set(), {"max_budget_fraction", "max_budget_bytes_per_rank"}, "memory")
        fraction = _number(memory.get("max_budget_fraction", 1), "max_budget_fraction")
        if fraction > 1:
            _invalid("max_budget_fraction cannot exceed 1")
        normalized["memory"]["max_budget_fraction"] = fraction
        if "max_budget_bytes_per_rank" in memory:
            maximum = memory["max_budget_bytes_per_rank"]
            if type(maximum) is not int or not 1 <= maximum <= 2**63 - 1:
                _invalid("max_budget_bytes_per_rank must be a positive signed 64-bit integer")
    if "latency" in spec:
        latency = _object(spec["latency"], set(), set(_LATENCY), "latency")
        if not latency:
            _invalid("latency must contain a declared P99 threshold")
        for key, value in latency.items():
            _number(value, key)
    if "throughput" in spec:
        throughput = _object(spec["throughput"], {"basis"},
                             {"min_requests_per_s", "min_output_tokens_per_s"}, "throughput")
        if throughput["basis"] not in ("measurement_window", "elapsed", "task_makespan"):
            _invalid(f"throughput.basis must be one of {sorted(_BASES)}")
        if len(throughput) == 1:
            _invalid("throughput must declare a minimum requests/s or output tokens/s")
        for key in throughput.keys() - {"basis"}:
            _number(throughput[key], key, zero=True)
    if "task" in spec:
        task = _object(spec["task"], {"max_makespan_ms"}, set(), "task")
        _number(task["max_makespan_ms"], "max_makespan_ms")
    return normalized


def _finite(value: Any, *, positive: bool = False) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and (number > 0 if positive else number >= 0) else None


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _verdict(values: list[str]) -> str:
    return "FAIL" if "FAIL" in values else "UNKNOWN" if "UNKNOWN" in values else "PASS"


def _check(name: str, category: str, scope: dict, actual: Any, operator: str,
           limit: Any, unit: str, *, verdict: str | None = None, reason: str = "",
           sample_count: int | None = None) -> dict:
    ratio = headroom = None
    if type(actual) in (int, float) and type(limit) in (int, float) and limit > 0:
        try:
            candidate = actual / limit
        except OverflowError:
            candidate = math.inf
        if math.isfinite(candidate):
            ratio = candidate
            difference = limit - actual if operator == "<=" else actual - limit
            headroom = difference / limit
    if verdict is None:
        if actual is None:
            verdict, reason = "UNKNOWN", reason or "metric_unavailable"
        else:
            passing = actual <= limit if operator == "<=" else actual >= limit
            verdict = "PASS" if passing else "FAIL"
            reason = reason or ("within_limit" if passing else "limit_violated")
    return {"id": name, "category": category, "scope": dict(scope), "actual": actual,
            "operator": operator, "limit": limit, "unit": unit, "sample_count": sample_count,
            "verdict": verdict, "reason": reason, "ratio": ratio, "normalized_headroom": headroom}


def _quantile(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * .99
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _window(result: dict) -> tuple[float, float] | None:
    contract = result.get("measurement_contract")
    selected = contract.get("arrival_window_ms") if isinstance(contract, dict) else None
    if selected is not None:
        if not isinstance(selected, (list, tuple)) or len(selected) != 2:
            return None
        start, end = (_finite(value) for value in selected)
        return None if start is None or end is None or end <= start else (start, end)
    throughput = result.get("throughput")
    if (isinstance(throughput, dict) and throughput.get("window_rule") in
            {"arrival_cohort_with_drain", "arrival_cohort_without_drain"}):
        end = _finite(throughput.get("window_ms"), positive=True)
        return None if end is None else (0.0, end)
    return None


def _throughput(rows: list[dict], interval: tuple[float, float] | None,
                *, arrival_window: bool = False) -> tuple[float | None, float | None]:
    if interval is None:
        return None, None
    start, end = interval
    if end <= start:
        return None, None
    requests = tokens = 0
    request_unknown = token_unknown = False
    for row in rows:
        if row.get("status") == "NOT_ADMITTED":
            continue
        if arrival_window:
            arrival = _finite(row.get("arrival_ms"))
            if arrival is None:
                request_unknown = token_unknown = True
                continue
            if not start <= arrival < end:
                continue
        if row.get("status") == "COMPLETED":
            completed = _finite(row.get("completion_ms"))
            if completed is None:
                request_unknown = True
            elif start <= completed <= end:
                requests += 1
        times = row.get("token_times_ms")
        actual = _count(row.get("output_tokens_actual"))
        if not isinstance(times, (list, tuple)) or actual is None or len(times) != actual:
            token_unknown = True
            continue
        timestamps = [_finite(value) for value in times]
        if any(value is None for value in timestamps):
            token_unknown = True
            continue
        valid = [value for value in timestamps if value is not None]
        if any(left > right for left, right in zip(valid, valid[1:])):
            token_unknown = True
            continue
        tokens += sum(start <= value <= end for value in valid)
    seconds = (end - start) / 1000
    if not math.isfinite(seconds) or seconds <= 0:
        return None, None
    request_rate = None if request_unknown else requests / seconds
    token_rate = None if token_unknown else tokens / seconds
    return (request_rate if request_rate is None or math.isfinite(request_rate) else None,
            token_rate if token_rate is None or math.isfinite(token_rate) else None)


def _limiting(checks: list[dict]) -> list[dict]:
    return sorted((deepcopy(check) for check in checks
                   if check["category"] in {"memory", "latency", "throughput", "task"}
                   and check["normalized_headroom"] is not None),
                  key=lambda check: (check["normalized_headroom"], check["id"],
                                     str(sorted(check["scope"].items()))))


def _assess(result: dict, rows: list[dict], ranks: Any, spec: dict, *, scope: dict,
            dag: bool = False, elapsed_ms: Any = None, makespan_ms: Any = None,
            resources_drained: Any = None, nodes_completed: Any = None,
            latency_from_rows: bool = False) -> dict:
    checks: list[dict] = []
    status = result.get("status")
    checks.append(_check("execution.status", "execution", scope, status, "==", "COMPLETED", "status",
                         verdict="UNKNOWN" if status is None else "PASS" if status == "COMPLETED" else "FAIL",
                         reason="missing_status" if status is None else "completed" if status == "COMPLETED"
                         else "run_not_completed"))
    cohort = [row for row in rows if row.get("status") != "NOT_ADMITTED"]
    completed = [row for row in cohort if row.get("status") == "COMPLETED"]
    unsuccessful = [row for row in cohort if row.get("status") not in {None, "COMPLETED"}]
    missing_status = any(row.get("status") is None for row in cohort)
    counts = result.get("counts")
    reported_failure = 0
    failure_counts_known = False
    if isinstance(counts, dict):
        failure_counts_known = all(_count(counts.get(name)) is not None
                                   for name in ("failed", "timed_out", "unfinished"))
        reported_failure = sum(_count(counts.get(name)) or 0
                               for name in ("failed", "timed_out", "unfinished", "blocked_dependency"))
    checks.append(_check("execution.unsuccessful_requests", "execution", scope,
                         max(len(unsuccessful), reported_failure), "<=", 0, "requests",
                         verdict="FAIL" if unsuccessful or reported_failure else
                         "UNKNOWN" if missing_status or not failure_counts_known else "PASS",
                         reason="unsuccessful_requests" if unsuccessful or reported_failure else
                         "request_state_missing" if missing_status or not failure_counts_known else "all_requests_completed"))
    consistent = (isinstance(counts, dict) and _count(counts.get("total")) == len(rows)
                  and _count(counts.get("completed")) == len(completed))
    checks.append(_check("execution.request_ledger", "execution", scope, consistent, "==", True, "boolean",
                         verdict="PASS" if consistent else "UNKNOWN",
                         reason="consistent_request_ledger" if consistent else "missing_or_inconsistent_request_ledger"))
    if dag:
        for name, observed in (("resources_drained", resources_drained), ("nodes_completed", nodes_completed)):
            checks.append(_check(f"execution.{name}", "execution", scope, observed, "==", True, "boolean",
                                 verdict="PASS" if observed is True else "FAIL" if observed is False else "UNKNOWN",
                                 reason=name if observed is True else f"{name}_not_established"))
    minimum = spec["min_completed_samples"]
    checks.append(_check("samples.completed", "samples", scope, len(completed), ">=", minimum, "requests",
                         verdict="PASS" if len(completed) >= minimum else "UNKNOWN",
                         reason="sufficient_completed_samples" if len(completed) >= minimum else "insufficient_completed_samples",
                         sample_count=len(completed)))
    latency = {}
    sample_counts = {}
    invalid_latency: set[str] = set()
    for metric in _LATENCY.values():
        field = "end_to_end_ms" if metric == "end_to_end" else f"{metric}_ms"
        samples = []
        for row in completed:
            value = row.get(field)
            if value is None and metric == "tpot":
                actual_tokens = _count(row.get("output_tokens_actual"))
                if actual_tokens is not None and actual_tokens <= 1:
                    continue  # One-token output legitimately has no TPOT sample.
                invalid_latency.add(metric)
                continue
            number = _finite(value)
            if number is None:
                invalid_latency.add(metric)
            else:
                samples.append(number)
        sample_counts[metric] = len(samples)
        aggregates = result.get("latency_ms")
        aggregate = aggregates.get(metric) if isinstance(aggregates, dict) else None
        aggregate = aggregate if isinstance(aggregate, dict) else {}
        raw_aggregate = aggregate.get("p99")
        if not latency_from_rows and raw_aggregate is not None and _finite(raw_aggregate) is None:
            invalid_latency.add(metric)
        latency[metric] = (None if metric in invalid_latency else
                           _quantile(samples) if latency_from_rows else _finite(raw_aggregate))
    elapsed = _finite(elapsed_ms if dag else result.get("simulated_ms"), positive=True)
    makespan = _finite(makespan_ms, positive=True) if dag else None
    intervals = {"measurement_window": _window(result) if not dag else None,
                 "elapsed": (0.0, elapsed) if elapsed is not None else None,
                 "task_makespan": (0.0, makespan) if dag and status == "COMPLETED"
                                  and resources_drained is True and nodes_completed is True
                                  and makespan is not None else None}
    rates = {basis: _throughput(cohort, interval, arrival_window=basis == "measurement_window")
             for basis, interval in intervals.items()}
    basis = spec.get("throughput", {}).get("basis")
    selected_rates = rates[basis] if basis is not None else (None, None)
    metrics = {"completed_samples": len(completed), "sample_counts": sample_counts,
               "ttft_p99_ms": latency["ttft"], "tpot_p99_ms": latency["tpot"],
               "end_to_end_p99_ms": latency["end_to_end"], "elapsed_ms": elapsed,
               "makespan_ms": makespan, "throughput_requests_per_s": selected_rates[0],
               "throughput_output_tokens_per_s": selected_rates[1]}
    for name, pair in rates.items():
        prefix = "window" if name == "measurement_window" else name
        metrics[f"{prefix}_requests_per_s"], metrics[f"{prefix}_output_tokens_per_s"] = pair
    if "memory" in spec:
        memory = spec["memory"]
        if not isinstance(ranks, list) or not ranks:
            checks.append(_check("memory.ranks", "memory", scope, None, "<=", None, "bytes",
                                 reason="per_rank_memory_evidence_missing"))
        else:
            for index, rank in enumerate(ranks):
                if not isinstance(rank, dict):
                    checks.append(_check(f"memory.rank.{index}", "memory", scope, None, "<=", None, "bytes",
                                         reason="per_rank_memory_evidence_missing"))
                    continue
                rank_scope = {**scope, **{key: rank[key] for key in _RANK_LABELS if key in rank}}
                capacity = _count(rank.get("device_memory_bytes"))
                capacity = capacity if capacity is not None and capacity > 0 else None
                budget = _count(rank.get("budget_peak_bytes"))
                fraction = Fraction(str(memory["max_budget_fraction"]))
                limit = None if capacity is None else min(
                    capacity, capacity * fraction.numerator // fraction.denominator,
                    memory.get("max_budget_bytes_per_rank", capacity))
                known = capacity is not None and budget is not None and isinstance(rank.get("device_id"), str)
                check = _check(f"memory.rank.{index}", "memory", rank_scope, budget, "<=", limit, "bytes",
                               verdict=None if known else "UNKNOWN",
                               reason="" if known else "per_rank_capacity_or_peak_missing")
                check["device_capacity_bytes"] = capacity
                check["max_budget_fraction"] = memory["max_budget_fraction"]
                checks.append(check)
    for key, limit in spec.get("latency", {}).items():
        metric = _LATENCY[key]
        count = sample_counts[metric]
        actual = latency[metric]
        checks.append(_check(f"latency.{key}", "latency", scope, actual, "<=", limit, "ms",
                             verdict="UNKNOWN" if count < minimum or actual is None else None,
                             reason="invalid_latency_evidence" if metric in invalid_latency else
                             "insufficient_metric_samples" if count < minimum else
                             "latency_metric_missing" if actual is None else "", sample_count=count))
    for key, limit in spec.get("throughput", {}).items():
        if key == "basis":
            continue
        tokens = key == "min_output_tokens_per_s"
        actual = selected_rates[1 if tokens else 0]
        checks.append(_check(f"throughput.{key}", "throughput", scope, actual, ">=", limit,
                             "output_tokens/s" if tokens else "dag_node_requests/s" if dag else "requests/s",
                             reason="throughput_basis_or_timestamps_missing" if actual is None else ""))
    if "task" in spec:
        checks.append(_check("task.max_makespan_ms", "task", scope,
                             makespan if dag and intervals["task_makespan"] is not None else None,
                             "<=", spec["task"]["max_makespan_ms"], "ms",
                             reason="" if dag and intervals["task_makespan"] is not None
                             else "completed_drained_task_makespan_required"))
    selected_interval = intervals.get(basis) if basis is not None else None
    return {"schema_version": 1, "verdict": _verdict([check["verdict"] for check in checks]),
            "constraints": deepcopy(spec), "checks": checks, "limiting_constraints": _limiting(checks),
            "metrics": metrics, "metric_contract": {
                "request_unit": "dag_node_request" if dag else "request",
                "latency_origin": "node_ready_arrival_not_dependency_release" if dag else "request_arrival",
                "quantile": "original run P99; DAG aggregate uses linear interpolation at (n-1)*0.99",
                "memory": "each rank budget_peak_bytes; includes reserved KV and safety/external budgets; no logical KV addition",
                "throughput_basis": basis,
                "throughput_window_ms": (None if selected_interval is None else
                                         selected_interval[1] - selected_interval[0]),
                "rate_intervals_ms": {name: list(interval) if interval is not None else None
                                      for name, interval in intervals.items()},
                "measurement_arrival_window": "half_open", "completion_and_emit_right_endpoint": "included",
                "output_tokens": "actual token_times_ms inside chosen interval; never cohort final token totals",
                "task_makespan": "last node output, excluding subsequent resource drain",
                "failure_policy": "any failed/timed_out/unfinished/blocked request vetoes PASS",
                "sample_policy": "declared minimum applies per assessment and each requested latency metric",
                "headroom": "upper:1-actual/limit; lower:actual/limit-1; unitless; zero limits have no ratio"}}


def evaluate_simulation_constraints(run: SimulationResult, spec: dict) -> dict:
    """Assess one ordinary run without changing its result, rows or Profiles."""
    constraints = validate_constraints(spec)
    if not isinstance(run, SimulationResult) or not isinstance(run.result, dict):
        _invalid("evaluate_simulation_constraints requires SimulationResult")
    rows = run.requests
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        _invalid("SimulationResult.requests must be a list of request records")
    return _assess(run.result, rows, run.result.get("memory"), constraints, scope={"kind": "simulation"})


def evaluate_dag_constraints(result: dict, spec: dict, *, pool_specs: dict | None = None) -> dict:
    """Assess a finite DAG overall and optional explicitly selected model pools."""
    constraints = validate_constraints(spec)
    if not isinstance(result, dict) or result.get("kind") != "dag":
        _invalid("evaluate_dag_constraints requires one DAG result")
    nodes, pools = result.get("nodes"), result.get("pools")
    if (not isinstance(nodes, list) or any(not isinstance(row, dict) for row in nodes)
            or not isinstance(pools, list) or any(not isinstance(pool, dict) for pool in pools)):
        _invalid("DAG nodes and pools must be record arrays")
    selected = {} if pool_specs is None else pool_specs
    by_id = {pool.get("model_id"): pool for pool in pools}
    if (not isinstance(selected, dict) or any(not isinstance(key, str) for key in selected)
            or selected.keys() - by_id.keys()):
        _invalid("pool_specs must name existing DAG model pools")
    normalized = {name: validate_constraints(value) for name, value in selected.items()}
    ranks = result.get("devices")
    context: dict[str, Any] = {"dag": True, "elapsed_ms": result.get("elapsed_ms"),
                               "makespan_ms": result.get("makespan_ms"),
                               "resources_drained": result.get("resources_drained"),
                               "nodes_completed": result.get("nodes_completed"), "latency_from_rows": True}
    overall = _assess(result, nodes, ranks, constraints, scope={"kind": "dag"}, **context)
    assessments = {}
    for name, pool_spec in normalized.items():
        pool_rows = [node for node in nodes if node.get("model_id") == name]
        pool_ranks = [rank for rank in ranks if rank.get("model_id") == name] if isinstance(ranks, list) else None
        assessments[name] = _assess(by_id[name].get("result", {}), pool_rows, pool_ranks, pool_spec,
                                    scope={"kind": "dag_pool", "model_id": name}, **context)
    if assessments:
        overall["overall_verdict"] = overall["verdict"]
        overall["verdict"] = _verdict([overall["verdict"], *[item["verdict"] for item in assessments.values()]])
        overall["checks"].extend(check for item in assessments.values() for check in item["checks"])
        overall["limiting_constraints"] = _limiting(overall["checks"])
        overall["pools"] = assessments
    return overall
