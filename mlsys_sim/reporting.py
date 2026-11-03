"""Shared metrics for single-group and stage-level simulations."""

import math
from typing import Any

from .state import SimulationResult


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    values = sorted(values)

    def quantile(q: float) -> float:
        position = (len(values) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"p50": quantile(0.50), "p95": quantile(0.95), "p99": quantile(0.99)}


def report_simulation(sim: Any) -> SimulationResult:
    rows: list[dict] = []
    for request in sim.requests.values():
        is_completed = request.status == "COMPLETED"
        ttft = (None if request.first_token_ms is None
                else request.first_token_ms - request.arrival_ms)
        intervals = [right - left for left, right in
                     zip(request.token_times, request.token_times[1:])]
        tpot = (sum(intervals) / len(intervals)
                if is_completed and intervals else None)
        rows.append({
            "request_id": request.request_id, "status": request.status,
            "arrival_ms": request.arrival_ms, "first_token_ms": request.first_token_ms,
            "completion_ms": request.completion_ms,
            "output_tokens_actual": request.generated,
            "num_computed_tokens": request.computed,
            "ttft_ms": ttft, "tpot_ms": tpot, "itl_ms": intervals,
            "token_times_ms": request.token_times,
            "end_to_end_ms": (request.completion_ms - request.arrival_ms
                              if is_completed and request.completion_ms is not None else None),
            "preemption_count": request.preemptions,
            "recomputed_tokens": request.recomputed_tokens,
            "recompute_ms": request.recompute_ms,
            "memory_wait_ms": request.memory_wait_ms,
            "error_code": request.error_code,
            "swap_out_count": request.swap_out_count,
            "swap_in_count": request.swap_in_count,
            "swap_out_bytes": request.swap_out_bytes,
            "swap_in_bytes": request.swap_in_bytes,
            "swap_ms": request.swap_ms,
            "kv_residency": (sim.swap.state(request.request_id)
                             if getattr(sim, "swap", None) is not None
                             and sim.swap.state(request.request_id) is not None
                             else "GPU" if request.request_id in sim.memory.allocations else "NONE"),
        })
    counts = {"total": len(rows), "completed": 0, "failed": 0,
              "timed_out": 0, "unfinished": 0, "not_admitted": 0}
    for row in rows:
        counts[row["status"].lower()] += 1
    evaluation = sim.exp.evaluation
    window = evaluation.measurement_window_ms
    cohort = [row for row in rows if row["status"] != "NOT_ADMITTED"]
    window = sim.now if window is None else window
    completed = [row for row in cohort if row["status"] == "COMPLETED"]
    sla_declared = evaluation.ttft_sla_ms is not None or evaluation.tpot_sla_ms is not None

    def eligible(row: dict) -> bool:
        if evaluation.ttft_sla_ms is not None:
            if row["ttft_ms"] is None or row["ttft_ms"] > evaluation.ttft_sla_ms:
                return False
        if evaluation.tpot_sla_ms is not None:
            if row["tpot_ms"] is None or row["tpot_ms"] > evaluation.tpot_sla_ms:
                return False
        return True

    good = [row for row in completed if eligible(row)]

    def rate(value: int) -> float | None:
        return value * 1000 / window if window > 0 else None

    provenance = getattr(sim.performance, "provenance", None)
    profile = getattr(sim.performance, "profile", None)
    calibration_source = sim.exp.calibration.backend
    profile_version = sim.exp.calibration.profile_version
    if profile is not None:
        calibration_source = ("measured_profile" if profile["source"]["kind"] == "measured"
                              else "test_fixture_profile")
        profile_version = profile["profile_version"]
    limitations = list(getattr(sim, "limitations", [
        "single replica; homogeneous phase batches; no chunked prefill",
        "fixed KV pool; allocator fragmentation is not simulated",
        "prefill-first admission; pressure drains active requests before recovery",
    ]))
    limitations.insert(0, "profile-backed CPU model; hardware holdout validation is separate"
                       if profile is not None else "uncalibrated CPU model")
    result = {
        "status": "UNRESOLVED" if sim.stop_reason else "COMPLETED",
        "error_code": sim.stop_reason,
        "backend": "simulated",
        "calibration_source": calibration_source,
        "profile_version": profile_version,
        "profile_provenance": dict(provenance) if provenance is not None else None,
        "hardware_validated": False,
        "quality_status": "not_evaluated",
        "simulated_ms": sim.now,
        "events_processed": sim.handled_events,
        "counts": counts,
        "admitted_count": len(cohort),
        "completion_rate_denominator": len(cohort),
        "completion_rate": counts["completed"] / len(cohort) if cohort else None,
        "evaluation_count": len(cohort),
        "cohort_completion_rate": len(completed) / len(cohort) if cohort else None,
        "latency_ms": {
            "ttft": _percentiles([row["ttft_ms"] for row in completed if row["ttft_ms"] is not None]),
            "tpot": _percentiles([row["tpot_ms"] for row in completed if row["tpot_ms"] is not None]),
            "itl": _percentiles([item for row in cohort for item in row["itl_ms"]]),
            "end_to_end": _percentiles([row["end_to_end_ms"] for row in completed]),
        },
        "throughput": {
            "requests_per_s": rate(len(completed)),
            "output_tokens_per_s": rate(sum(row["output_tokens_actual"] for row in cohort)),
            "window_ms": window,
            "window_rule": ("arrival_cohort_with_drain" if evaluation.drain
                            else "arrival_cohort_without_drain")
                           if evaluation.measurement_window_ms is not None else "zero_to_simulation_end",
        },
        "goodput": {
            "requests_per_s": rate(len(good)),
            "output_tokens_per_s": rate(sum(row["output_tokens_actual"] for row in good)),
            "eligible_requests": len(good), "window_ms": window,
            "sla_declared": sla_declared,
            "rule": "completed_requests_satisfying_all_declared_slas",
        },
        "sla_satisfied": (all(eligible(row) for row in completed) and len(completed) == len(cohort)
                          if sla_declared and cohort else None),
        "preemption_count": sum(row["preemption_count"] for row in rows),
        "recomputed_tokens": sum(row["recomputed_tokens"] for row in rows),
        "recompute_execution_ms": sim.recompute_execution_ms,
        "compute_busy_ms": sim.compute_busy_ms,
        "memory": sim.memory.snapshot(),
        "measurement_contract": {
            "arrival_window_ms": [0.0, evaluation.measurement_window_ms]
                                 if evaluation.measurement_window_ms is not None else None,
            "arrival_window_rule": "half_open",
            "drain": evaluation.drain,
            "max_drain_ms": evaluation.max_drain_ms,
            "completion_denominator": "requests_eligible_for_arrival_window",
            "outside_window_in_completion_denominator": False,
            "failed_timed_out_unfinished_in_completion_denominator": True,
            "initial_state": sim.exp.execution.initial_state,
        },
        "limitations": limitations,
    }
    swap = getattr(sim, "swap", None)
    if swap is not None:
        result["swap"] = swap.snapshot(sim.now)
    return SimulationResult(result, rows, sim.events)
