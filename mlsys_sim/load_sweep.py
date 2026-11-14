"""Finite-horizon offered-load/client sweeps with auditable empirical metrics."""

from copy import deepcopy
from dataclasses import asdict, replace
import csv
import heapq
from html import escape
import io
import json
import math
from pathlib import Path
from statistics import mean
import tempfile
from typing import Any

from .capacity import _bind_inputs, _profile_inputs
from .constraints import _limiting, _verdict, evaluate_simulation_constraints, validate_constraints
from .engine import simulate
from .schema import Experiment, RequestSpec, SimulationError, StepEstimate
from .serialization import experiment_dict
from .state import SimulationResult, TERMINAL
from .workload import (
    GENERATOR_VERSION, MAX_REQUESTS, _integer, _keys, _length, _positive, _rng,
    canonical_sha256, generate_workload, request_specs, validate_workload_spec,
)


SWEEP_VERSION = "finite-load-sweep-v1"
MAX_SWEEP_RUNS = 256
MAX_SWEEP_REQUESTS = 250_000
MAX_SWEEP_EVENTS = 1_000_000
_METRICS = {"p99_e2e_ms": "end_to_end", "p99_ttft_ms": "ttft", "p99_tpot_ms": "tpot"}


def _invalid(message: str) -> None:
    raise SimulationError("INVALID_LOAD_PLAN", message)


def validate_load_plan(plan: dict) -> dict:
    _keys(plan, {"schema_version", "workload", "replications", "drain_ms", "slo"},
          {"rate_grid_rps", "concurrency_grid", "min_completed_samples", "service_time",
           "backlog_growth_tolerance", "constraints"}, "load plan")
    _integer(plan["schema_version"], "schema_version", 1, 1)
    if "constraints" in plan:
        validate_constraints(plan["constraints"])
    if ("rate_grid_rps" in plan) == ("concurrency_grid" in plan):
        _invalid("Exactly one of rate_grid_rps or concurrency_grid is required")
    clients = "concurrency_grid" in plan
    key = "concurrency_grid" if clients else "rate_grid_rps"
    grid = plan[key]
    if not isinstance(grid, list) or not 1 <= len(grid) <= 256:
        _invalid(f"{key} must have 1..256 values")
    values = [(_integer(value, key, 1, 10_000) if clients else _positive(value, key)) for value in grid]
    if any(left >= right for left, right in zip(values, values[1:])):
        _invalid(f"{key} must be strictly increasing; observations are never reordered")
    repeats = _integer(plan["replications"], "replications", 1, 100)
    if repeats * len(grid) > MAX_SWEEP_RUNS:
        _invalid(f"A sweep is limited to {MAX_SWEEP_RUNS} independent runs")
    _integer(plan.get("min_completed_samples", 100), "min_completed_samples", 1, MAX_REQUESTS * 100)
    drain = _positive(plan["drain_ms"], "drain_ms", allow_zero=True)
    slo = plan["slo"]
    if not isinstance(slo, dict) or not slo or slo.keys() - _METRICS.keys():
        _invalid(f"slo must declare one or more thresholds from {sorted(_METRICS)}")
    for name, value in slo.items():
        _positive(value, name)
    if "backlog_growth_tolerance" in plan:
        _integer(plan["backlog_growth_tolerance"], "backlog_growth_tolerance", 0, MAX_REQUESTS)
    workload = deepcopy(plan["workload"])
    if not isinstance(workload, dict) or not isinstance(workload.get("arrival"), dict):
        _invalid("workload must contain an arrival specification")
    arrival = workload["arrival"]
    if clients:
        _keys(arrival, {"process", "duration_ms", "max_requests", "think_ms"}, set(), "closed-loop arrival")
        if arrival["process"] != "closed_loop":
            _invalid("concurrency_grid requires arrival.process=closed_loop")
        _positive(arrival["think_ms"], "think_ms", allow_zero=True)
        duration = _positive(arrival["duration_ms"], "duration_ms")
        workload["arrival"] = {"process": "periodic", "rate_rps": 1000 / duration,
                               "duration_ms": duration, "max_requests": arrival["max_requests"]}
        maximum = _integer(arrival["max_requests"], "max_requests", 1, MAX_REQUESTS)
        if max(values) > maximum:
            _invalid("max_requests must cover the initial clients")
        validate_workload_spec(workload)
    else:
        for value in values:
            workload["arrival"]["rate_rps"] = value
            validate_workload_spec(workload)
        duration = float(arrival["duration_ms"])
    if arrival["max_requests"] * repeats * len(grid) > MAX_SWEEP_REQUESTS:
        _invalid(f"Declared max_requests across all runs exceeds {MAX_SWEEP_REQUESTS}")
    if (not math.isfinite(duration + drain) or duration + drain < duration
            or (drain > 0 and duration + drain == duration)):
        _invalid("duration_ms + drain_ms overflows finite time")
    service = plan.get("service_time", {"kind": "deterministic"})
    if not isinstance(service, dict):
        _invalid("service_time must be an object")
    if service.get("kind") == "deterministic":
        _keys(service, {"kind"}, set(), "service_time")
    elif service.get("kind") == "empirical_step_scale":
        _keys(service, {"kind", "samples"}, set(), "service_time")
        samples = service["samples"]
        if not isinstance(samples, list) or not 1 <= len(samples) <= 4096:
            _invalid("empirical service samples must contain 1..4096 positive scales")
        for value in samples:
            _positive(value, "service time scale")
    else:
        _invalid("service_time.kind must be deterministic or empirical_step_scale")
    return plan


class _EmpiricalService:
    """Sample synthetic phase-batch duration scales; never label these as measured."""

    def __init__(self, original: Any, samples: list[float], seed: int, replication: int, replica: int):
        self.original = original
        self.samples = samples
        self.seed, self.replication, self.replica = seed, replication, replica
        self.streams: dict[int, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def estimate_step(self, phase: str, query_tokens: Any, context_lengths: Any,
                      stage_index: int | None = None, **kwargs: Any) -> StepEstimate:
        # The single-group API leaves stage_index unspecified; PP passes it positionally.
        if stage_index is None:
            base = self.original.estimate_step(phase, query_tokens, context_lengths, **kwargs)
            stage = 0
        else:
            base = self.original.estimate_step(phase, query_tokens, context_lengths, stage_index, **kwargs)
            stage = stage_index
        if stage not in self.streams:
            self.streams[stage] = _rng(self.seed, self.replication,
                                       f"empirical_service_replica_{self.replica}_stage_{stage}")
        factor = self.streams[stage].choice(self.samples)
        duration = base.duration_ms * factor
        if not math.isfinite(duration) or duration <= 0:
            _invalid("Sampled service duration overflow or underflow")
        return StepEstimate(duration, {"synthetic_sampled_ms": duration,
                                      "sampling_metadata": {"empirical_step_scale": factor,
                                                            "unscaled_duration_ms": base.duration_ms}},
                            "synthetic_empirical_step_distribution",
                            (*base.assumptions, "IID empirical scale per synthetic phase batch and PP stage."))


def _run_dynamic(experiment: Experiment, workload: dict, replication: int,
                 concurrency: int | None, service: dict) -> tuple[SimulationResult, dict]:
    from .event_driver import EventDriver

    # Match simulate(): warmup is a separate run and the measured cache starts empty.
    warmup_summary = {"enabled": False, "status": "NOT_REQUESTED",
                      "cache_effect": "none; measured run starts with empty KV"}
    if experiment.execution.warmup:
        warmup_summary = simulate(replace(experiment, workload=())).result["warmup"]
    driver = EventDriver(replace(experiment, workload=(), execution=replace(experiment.execution, warmup=())))
    if service["kind"] == "empirical_step_scale":
        for index, engine in enumerate(driver.local_engines):
            engine.performance = _EmpiricalService(engine.performance, service["samples"], workload["seed"],
                                                   replication, index)
    duration = float(workload["arrival"]["duration_ms"])
    drain = experiment.evaluation.max_drain_ms
    assert drain is not None
    cutoff = min(experiment.execution.max_simulated_ms, duration + drain)
    pending: list[tuple[float, int, RequestSpec | int]] = []
    rows: list[dict] = []
    completed_seen: set[str] = set()
    owners: dict[str, int] = {}
    if concurrency is None:
        generated = generate_workload(workload, replication=replication)
        for index, request in enumerate(request_specs(generated)):
            heapq.heappush(pending, (request.arrival_ms, index, request))
        rows = generated["requests"]
    else:
        generated = {"provenance": {"generator_version": GENERATOR_VERSION,
                     "kind": "closed_loop", "seed": workload["seed"], "replication": replication,
                     "spec_sha256": canonical_sha256(workload), "clients": concurrency,
                     "think_ms": workload["arrival"]["think_ms"],
                     "release_rule": "one_outstanding_request_per_client; next after terminal feedback",
                     "arrival_window_ms": [0.0, duration], "arrival_window_rule": "half_open"}}
        for client in range(concurrency):
            heapq.heappush(pending, (0.0, client, client))
    input_rng, output_rng = (_rng(workload["seed"], replication, label)
                             for label in ("input_lengths", "output_lengths"))
    input_range = _length(workload["tokens"]["input"], "tokens.input")
    output_range = _length(workload["tokens"]["output"], "tokens.output")
    serial = len(pending)
    reason = None

    def release_clients() -> None:
        nonlocal serial
        if concurrency is None:
            return
        for name, state in driver.requests.items():
            if name in completed_seen or state.status not in TERMINAL:
                continue
            completed_seen.add(name)
            ready = driver.now + workload["arrival"]["think_ms"]
            if (not math.isfinite(ready) or ready < driver.now or
                    (workload["arrival"]["think_ms"] > 0 and ready == driver.now)):
                _invalid("Closed-loop think deadline overflow or nonprogress")
            if ready < duration:
                if ready == state.arrival_ms:
                    _invalid("Closed-loop zero-time terminal feedback cannot advance time; configure positive think_ms")
                heapq.heappush(pending, (ready, serial, owners[name]))
                serial += 1

    while True:
        if driver.stopped:
            reason = driver.stop_reason
            break
        release_clients()
        event = driver.next_event()
        choices = ([event[0]] if event is not None else []) + ([pending[0][0]] if pending else [])
        if not choices:
            break
        when = min(choices)
        if when > cutoff:
            driver.set_time(cutoff)
            reason = "DRAIN_TIME_LIMIT" if cutoff == duration + drain else "SIMULATION_TIME_LIMIT"
            break
        # Completions/timeouts at a timestamp precede client arrivals, just as the core engine.
        while (event := driver.next_event()) is not None and event[0] == when:
            driver.advance_one()
            if driver.stopped:
                break
        if driver.stopped:
            reason = driver.stop_reason
            break
        driver.set_time(when)
        release_clients()
        offered_now = []
        while pending and pending[0][0] == when:
            _, _, value = heapq.heappop(pending)
            if concurrency is None:
                assert isinstance(value, RequestSpec)
                request = value
            else:
                assert isinstance(value, int)
                if len(rows) >= workload["arrival"]["max_requests"]:
                    _invalid("Closed-loop requests exceed max_requests; no silent truncation")
                request = RequestSpec(
                    f"{workload.get('request_prefix', 'generated')}-r{replication}-{len(rows)}", when,
                    input_rng.randint(*input_range), output_rng.randint(*output_range),
                    workload.get("priority", 0), workload.get("timeout_ms"),
                    tuple(workload.get("prompt_token_ids", ())), workload.get("cache_namespace", ""))
                timeout = request.timeout_ms
                if timeout is not None and (not math.isfinite(when + timeout) or when + timeout <= when):
                    _invalid("Closed-loop timeout deadline overflow or nonprogress")
                owners[request.request_id] = value
                row = asdict(request)
                row["prompt_token_ids"] = list(row["prompt_token_ids"])
                rows.append(row)
            offered_now.append(request)
        # Materialize every client released at this timestamp before admission
        # budgets. Already offered clients remain in the failure denominator.
        for request in offered_now:
            if driver.handled_events >= experiment.execution.max_events:
                reason = "SIMULATION_EVENT_LIMIT"
                break
            try:
                driver.accept_request(request)
            except SimulationError as error:
                if error.code not in {"NO_AVAILABLE_REPLICA", "SIMULATION_EVENT_LIMIT"}:
                    raise
                reason = error.code
                break
            if driver.stopped:
                reason = driver.stop_reason
                break
        if driver.stopped or reason is not None:
            break
        driver.schedule_ready()
        if driver.stopped:
            reason = driver.stop_reason
            break
        if driver.now >= cutoff and (driver.has_unfinished() or driver.has_inflight() or pending):
            reason = "DRAIN_TIME_LIMIT"
            break
    result = driver.finalize(reason=reason)
    result.result["warmup"] = warmup_summary
    generated["requests"] = rows
    generated["provenance"].update({"requests_sha256": canonical_sha256(rows), "request_count": len(rows),
                                    "service_time": deepcopy(service),
                                    "service_rng": "sha256 seed/replication/replica/stage substreams"})
    # Open-loop requests not reached by a global budget retain denominator membership.
    missing = [row for row in rows if row["request_id"] not in driver.requests]
    if missing:
        _account_unreached(result, missing, reason or driver.stop_reason or "SIMULATION_TIME_LIMIT")
    return result, generated


def _account_unreached(run: SimulationResult, missing: list[dict], reason: str) -> None:
    """Retain offered-but-unprocessed arrivals after a bounded dynamic run stops."""
    for spec in missing:
        run.requests.append({"request_id": spec["request_id"], "status": "UNFINISHED",
                             "arrival_ms": spec["arrival_ms"], "first_token_ms": None,
                             "completion_ms": None, "output_tokens_actual": 0, "num_computed_tokens": 0,
                             "ttft_ms": None, "tpot_ms": None, "itl_ms": [], "token_times_ms": [],
                             "end_to_end_ms": None, "error_code": reason,
                             "admission_status": "not_processed_before_global_budget"})
    result = run.result
    result["status"], result["error_code"] = "UNRESOLVED", reason
    result["counts"]["total"] += len(missing)
    result["counts"]["unfinished"] += len(missing)
    for name in ("admitted_count", "completion_rate_denominator", "evaluation_count"):
        result[name] += len(missing)
    denominator = result["completion_rate_denominator"]
    for name in ("completion_rate", "cohort_completion_rate"):
        result[name] = result["counts"]["completed"] / denominator
    if result["sla_satisfied"] is not None:
        result["sla_satisfied"] = False


def _quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    result: dict[str, float | None] = {}
    for label, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("p999", .999)):
        if not ordered:
            result[label] = None
        else:
            position = (len(ordered) - 1) * q
            lower, upper = math.floor(position), math.ceil(position)
            result[label] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


def _summary(run: SimulationResult, duration: float, plan: dict, *, closed_loop: bool) -> dict:
    cohort = [row for row in run.requests if row["status"] != "NOT_ADMITTED"]
    complete = [row for row in cohort if row["status"] == "COMPLETED"]
    failures = len(cohort) - len(complete)
    def backlog(at: float) -> int:
        return sum(row["arrival_ms"] < at and
                   (row["completion_ms"] is None or row["completion_ms"] > at) for row in cohort)
    first, last = backlog(duration * .25), backlog(duration)
    growth = last - first
    tolerance = plan.get("backlog_growth_tolerance", max(2, math.ceil(len(cohort) * .05)))
    enough = len(complete) >= plan.get("min_completed_samples", 100)
    missing_slo_metric = any(run.result["latency_ms"][_METRICS[key]]["p99"] is None
                             for key in plan["slo"])
    if failures or run.result["status"] != "COMPLETED":
        classification = "observed_failure"
    elif not enough or missing_slo_metric:
        classification = "insufficient_samples"
    elif not closed_loop and growth > tolerance:
        classification = "observed_overloaded"
    else:
        classification = "observed_stable" if not closed_loop else "observed_completed"
    passing: bool | None = None
    if classification != "insufficient_samples":
        passing = failures == 0 and run.result["status"] == "COMPLETED"
        for key, metric in _METRICS.items():
            if key in plan["slo"]:
                value = run.result["latency_ms"][metric]["p99"]
                passing = passing and value is not None and value <= plan["slo"][key]
    return {"classification": classification, "slo_pass": passing,
            "eligible_requests": len(cohort), "completed_samples": len(complete),
            "unsuccessful_requests": failures, "completion_rate_denominator": len(cohort),
            "window_completed_requests_per_s": sum(row["completion_ms"] <= duration for row in complete)
                                                   * 1000 / duration,
            "backlog": {"quarter_window": first, "window_end": last, "growth": growth,
                        "growth_tolerance": tolerance, "drained": failures == 0,
                        "criterion_applies": not closed_loop},
            "latency_ms": deepcopy(run.result["latency_ms"]),
            "throughput": deepcopy(run.result["throughput"])}


def run_load_sweep(experiment: Experiment, plan: dict) -> dict:
    """Run every declared point/replication; never refit or select favorable seeds."""
    validate_load_plan(plan)
    inputs = _profile_inputs(experiment)
    if not inputs:
        return _run_bound_sweep(experiment, plan, inputs, experiment)
    with tempfile.TemporaryDirectory(prefix="mlsys-load-inputs-") as temporary:
        bound = _bind_inputs(experiment, inputs, Path(temporary))
        paths = {}
        for item in inputs:
            selected = getattr(bound.calibration, item["field"])
            path = selected[item["index"]] if item["field"].endswith("paths") else selected
            Path(path).chmod(0o400)
            paths[path] = item["origin_path"]
        report = _run_bound_sweep(experiment, plan, inputs, bound)
        return _restore_origin_paths(report, paths)


def _restore_origin_paths(value: Any, paths: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _restore_origin_paths(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_origin_paths(item, paths) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_origin_paths(item, paths) for item in value)
    return paths.get(value, value) if isinstance(value, str) else value


def _run_bound_sweep(experiment: Experiment, plan: dict, profile_inputs: list[dict],
                     bound_experiment: Experiment) -> dict:
    clients = "concurrency_grid" in plan
    grid_key = "concurrency_grid" if clients else "rate_grid_rps"
    service = plan.get("service_time", {"kind": "deterministic"})
    if service["kind"] != "deterministic" and experiment.calibration.backend != "synthetic":
        _invalid("Empirical step scaling requires synthetic backend; Profile timing/variance is left unchanged")

    def check_frozen_inputs() -> None:
        try:
            current_inputs = _profile_inputs(experiment)
        except (SimulationError, UnicodeError) as error:
            raise SimulationError("INPUT_CHANGED", "A frozen Profile became unreadable during the load sweep") from error
        if current_inputs != profile_inputs:
            raise SimulationError("INPUT_CHANGED", "Profile bytes changed during the load sweep; mixed-source curve rejected")

    duration = float(plan["workload"]["arrival"]["duration_ms"])
    evaluation = replace(experiment.evaluation, measurement_window_ms=duration,
                         drain=True, max_drain_ms=float(plan["drain_ms"]))
    points: list[dict] = []
    constraints = plan.get("constraints")
    total_events = 0
    for point_index, value in enumerate(plan[grid_key]):
        runs: list[dict] = []
        for replication in range(plan["replications"]):
            workload = deepcopy(plan["workload"])
            if not clients:
                workload["arrival"]["rate_rps"] = value
            exp = replace(bound_experiment, evaluation=evaluation,
                          execution=replace(bound_experiment.execution, seed=workload["seed"]))
            try:
                if clients or service["kind"] != "deterministic":
                    result, generated = _run_dynamic(exp, workload, replication, value if clients else None, service)
                else:
                    generated = generate_workload(workload, replication=replication)
                    result = simulate(replace(exp, workload=request_specs(generated)))
            except SimulationError:
                check_frozen_inputs()
                raise
            check_frozen_inputs()
            total_events += result.result["events_processed"]
            if total_events > MAX_SWEEP_EVENTS:
                _invalid(f"Report events exceed total sweep bound {MAX_SWEEP_EVENTS}; reduce duration/load or disable detailed_events")
            entry: dict[str, Any] = {"replication": replication,
                                     "summary": _summary(result, duration, plan, closed_loop=clients),
                                     "workload": generated, "result": result.result, "requests": result.requests,
                                     "events": result.events}
            if constraints is not None:
                entry["constraint_assessment"] = evaluate_simulation_constraints(result, constraints)
                entry["joint_verdict"] = _joint_verdict(entry["summary"], entry["constraint_assessment"])
            runs.append(entry)
        samples: dict[str, list[float]] = {metric: [] for metric in _METRICS.values()}
        for run in runs:
            for row in run["requests"]:
                if row["status"] == "COMPLETED":
                    for metric in samples:
                        key = "end_to_end_ms" if metric == "end_to_end" else f"{metric}_ms"
                        if row[key] is not None:
                            samples[metric].append(row[key])
        labels = [run["summary"]["classification"] for run in runs]
        classification = next((label for label in ("observed_failure", "insufficient_samples", "observed_overloaded")
                               if label in labels), "observed_completed" if clients else "observed_stable")
        points.append({"point_index": point_index, "offered_rps": None if clients else value,
                       "clients": value if clients else None, "classification": classification,
                       "slo_pass": (None if "insufficient_samples" in labels else
                                    all(run["summary"]["slo_pass"] is True for run in runs)),
                       "sample_counts": {metric: len(values) for metric, values in samples.items()},
                       "pooled_latency_ms": {metric: _quantiles(values) for metric, values in samples.items()},
                       "mean_requests_per_s": mean(run["result"]["throughput"]["requests_per_s"] for run in runs),
                       "mean_window_completed_requests_per_s": mean(
                           run["summary"]["window_completed_requests_per_s"] for run in runs),
                       "replication_slo_outcomes": [run["summary"]["slo_pass"] for run in runs],
                       "mixed_replication_slo": any(run["summary"]["slo_pass"] is True for run in runs)
                                                and any(run["summary"]["slo_pass"] is False for run in runs),
                       "throughput_observed_range": [min(run["result"]["throughput"]["requests_per_s"] for run in runs),
                                                     max(run["result"]["throughput"]["requests_per_s"] for run in runs)],
                       "runs": runs})
        if constraints is not None:
            checks = []
            for run in runs:
                for check in run["constraint_assessment"]["checks"]:
                    checks.append({**deepcopy(check), "scope": {**check["scope"], "replication": run["replication"]}})
            assessment = {"verdict": _verdict([run["constraint_assessment"]["verdict"] for run in runs]),
                          "checks": checks, "limiting_constraints": _limiting(checks),
                          "replication_verdicts": [{"replication": run["replication"],
                                                    "verdict": run["constraint_assessment"]["verdict"]} for run in runs],
                          "metric_contract": {"aggregation": "every declared replication must PASS independently"}}
            joint = _verdict([run["joint_verdict"] for run in runs])
            points[-1].update(constraint_assessment=assessment, joint_verdict=joint,
                              joint_pass=True if joint == "PASS" else False if joint == "FAIL" else None)
    accepted = [point for point in points if point["slo_pass"] is True and
                point["classification"] in {"observed_stable", "observed_completed"}]
    stable = [point for point in points if point["classification"] == "observed_stable"]
    brackets = []
    slo_brackets = []
    axis = "clients" if clients else "offered_rps"
    for left, right in zip(points, points[1:]):
        if left["classification"] == "observed_stable" and right["classification"] == "observed_overloaded":
            brackets.append({"stable": left["offered_rps"], "overloaded": right["offered_rps"],
                             "resolution_rps": right["offered_rps"] - left["offered_rps"]})
        if left["slo_pass"] is True and right["slo_pass"] is False:
            slo_brackets.append({"pass": left[axis], "fail": right[axis],
                                 "resolution": right[axis] - left[axis], "axis": axis})
    report: dict[str, Any] = {"schema_version": 1, "sweep_version": SWEEP_VERSION, "backend": "simulated",
            "hardware_validated": False, "experiment_sha256": canonical_sha256(experiment_dict(experiment)),
            "profile_inputs": [{**item, "artifact_path": f"inputs/profile-{index:03d}.json"}
                               for index, item in enumerate(profile_inputs)],
            "profile_inputs_sha256": canonical_sha256(profile_inputs),
            "plan_sha256": canonical_sha256(plan), "plan": deepcopy(plan),
            "scan_axis": "clients" if clients else "offered_rps", "points": points,
            "boundaries": {"sla_max_observed": (accepted[-1]["clients" if clients else "offered_rps"]
                                                if accepted else None),
                           "last_observed_stable_rps": stable[-1]["offered_rps"] if stable else None,
                           "stable_overloaded_brackets": brackets,
                           "slo_pass_fail_brackets": slo_brackets,
                           "slo_nonmonotonic": any(left["slo_pass"] is False and right["slo_pass"] is True
                                                   for left, right in zip(points, points[1:])),
                           "p99_decreases_on_grid": any(
                               left["pooled_latency_ms"]["end_to_end"]["p99"] is not None and
                               right["pooled_latency_ms"]["end_to_end"]["p99"] is not None and
                               left["pooled_latency_ms"]["end_to_end"]["p99"] >
                               right["pooled_latency_ms"]["end_to_end"]["p99"]
                               for left, right in zip(points, points[1:])),
                           "stability_nonmonotonic": any(
                               left["classification"] == "observed_overloaded" and
                               right["classification"] == "observed_stable"
                               for left, right in zip(points, points[1:]))},
            "measurement_contract": {"arrival_window_ms": [0, duration], "drain_ms": plan["drain_ms"],
                "throughput": "original arrival-cohort completions / offered window, including drain",
                "window_completed_throughput": "requests completed at or before window end / offered window",
                "latency": "original completed-request metrics; failed/timedout/unfinished veto SLO pass",
                "quantile": "linear interpolation at (n-1)*q; pooled across all declared replications",
                "stability": "finite observation only; completed drain plus declared backlog growth test",
                "closed_loop": "client population is distinct from scheduler max_batch_size",
                "initial_state": "fresh engine and empty measured cache per point/replication",
                "service_time": deepcopy(service)},
            "limitations": ["CPU model predictions, not a hardware benchmark or mathematical stability proof",
                            "SLA boundary is observed grid only; no interpolation or extrapolation",
                            "Empirical service scaling is IID synthetic phase-batch/stage timing, not request-level variance",
                            "Low sample counts do not establish a reliable tail; all run counts remain visible"]}
    if constraints is not None:
        joint_points = [point for point in points if point["joint_verdict"] == "PASS"]
        report["boundaries"].update({
            "joint_max_observed": joint_points[-1][axis] if joint_points else None,
            "joint_pass_fail_brackets": [{"pass": left[axis], "fail": right[axis],
                                          "resolution": right[axis] - left[axis], "axis": axis}
                                         for left, right in zip(points, points[1:])
                                         if left["joint_verdict"] == "PASS" and right["joint_verdict"] == "FAIL"],
            "joint_nonmonotonic": any(left["joint_verdict"] == "FAIL" and right["joint_verdict"] == "PASS"
                                      for left, right in zip(points, points[1:]))})
    return report


def _joint_verdict(summary: dict, assessment: dict) -> str:
    if (summary["slo_pass"] is False or summary["classification"] in {"observed_failure", "observed_overloaded"}
            or assessment["verdict"] == "FAIL"):
        return "FAIL"
    if summary["slo_pass"] is None or assessment["verdict"] == "UNKNOWN":
        return "UNKNOWN"
    return "PASS"


def _csv(report: dict) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    joint = "constraints" in report["plan"]
    columns = ["offered_rps", "clients", "replication", "classification", "slo_pass",
               "eligible", "completed", "cohort_requests_per_s", "window_completed_requests_per_s",
               "ttft_p99_ms", "tpot_p99_ms", "e2e_p99_ms"]
    if joint:
        columns.extend(["constraint_verdict", "joint_verdict", "minimum_normalized_headroom"])
    writer.writerow(columns)
    for point in report["points"]:
        for run in point["runs"]:
            row = run["summary"]
            values = [point["offered_rps"], point["clients"], run["replication"], row["classification"],
                      row["slo_pass"], row["eligible_requests"], row["completed_samples"],
                      row["throughput"]["requests_per_s"], row["window_completed_requests_per_s"],
                      row["latency_ms"]["ttft"]["p99"],
                      row["latency_ms"]["tpot"]["p99"], row["latency_ms"]["end_to_end"]["p99"]]
            if joint:
                assessment = run["constraint_assessment"]
                limiting = assessment["limiting_constraints"]
                values.extend([assessment["verdict"], run["joint_verdict"],
                               limiting[0]["normalized_headroom"] if limiting else None])
            writer.writerow(values)
    return output.getvalue()


def _svg(report: dict, metric: str) -> str:
    points = report["points"]
    clients = report["scan_axis"] == "clients"
    xs = [point["clients" if clients else "offered_rps"] for point in points]
    ys = [point["mean_window_completed_requests_per_s"] if metric == "throughput" else
          point["pooled_latency_ms"][metric]["p99"] for point in points]
    threshold_key = next((key for key, name in _METRICS.items() if name == metric), None)
    threshold = report["plan"]["slo"].get(threshold_key)
    ymax = max([value for value in ys if value is not None] +
               ([threshold] if threshold is not None else []) + [1.0]) * 1.1
    xmax = max(xs) * 1.05
    title = "窗口内完成吞吐量" if metric == "throughput" else f"{metric} P99 延迟"
    units = "requests/s" if metric == "throughput" else "ms"
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 480" role="img" '
             f'aria-label="{escape(title)}"><rect width="900" height="480" fill="white"/>',
             '<g font-family="sans-serif" fill="#172536">',
             f'<text x="80" y="32" font-size="20">{escape(title)} · CPU 模型预测</text>',
             '<text x="80" y="56" font-size="12">窗口完成吞吐取重复均值；经验 P99 合并全部重复的完成样本；非真机测量</text>',
             '<path d="M80 80 V370 H840" fill="none" stroke="#657486"/>']
    for fraction in (0, .25, .5, .75, 1):
        y = 370 - fraction * 290
        items.append(f'<text x="68" y="{y+4:g}" text-anchor="end" font-size="12">{fraction*ymax:.3g}</text>')
    if threshold is not None:
        y = 370 - threshold / ymax * 290
        items.append(f'<path d="M80 {y:g} H840" stroke="#a53b31" stroke-dasharray="7 5" fill="none"/>')
        items.append(f'<text x="835" y="{y-6:g}" text-anchor="end" fill="#a53b31" font-size="12">'
                     f'SLO ≤ {threshold:g} ms</text>')
    coords = [(80 + x / xmax * 760, 370 - y / ymax * 290) if y is not None else None
              for x, y in zip(xs, ys)]
    # Separate segments around missing quantiles, never bridge censored/empty points.
    for left, right in zip(coords, coords[1:]):
        if left is not None and right is not None:
            items.append(f'<path d="M{left[0]:g} {left[1]:g} L{right[0]:g} {right[1]:g}" '
                         'stroke="#23599a" fill="none"/>')
    for index, (point, coordinate) in enumerate(zip(points, coords)):
        x = 80 + xs[index] / xmax * 760
        if coordinate is not None:
            items.append(f'<circle cx="{coordinate[0]:g}" cy="{coordinate[1]:g}" r="4" fill="#23599a"/>')
        if len(points) <= 16 or index in (0, len(points) - 1):
            count = (sum(run["summary"]["completed_samples"] for run in point["runs"])
                     if metric == "throughput" else point["sample_counts"][metric])
            items.append(f'<text x="{x:g}" y="392" text-anchor="middle" font-size="11">{xs[index]:g}</text>')
            items.append(f'<text x="{x:g}" y="409" text-anchor="middle" font-size="10">n={count}</text>')
    items.extend([f'<text x="20" y="78" font-size="12">{units}</text>',
                  f'<text x="460" y="438" text-anchor="middle" font-size="13">'
                  f'{"在线 client 数（非 batch 大小）" if clients else "开放到达率 requests/s"}</text>',
                  '<text x="80" y="464" font-size="11">分位数仅来自完成请求；失败、超时、未完成会使整点 SLA 不通过。详见 report.json/points.csv。</text>',
                  '</g></svg>'])
    return "\n".join(items)


def write_load_report(report: dict, output: str | Path) -> Path:
    """Write one complete report to a new directory; never overwrite a prior run."""
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    artifacts = {"report.json": encoded, "points.csv": _csv(report), "report.html": _html(report)}
    for metric in ("throughput", "ttft", "tpot", "end_to_end"):
        artifacts[f"{metric}.svg"] = _svg(report, metric)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in artifacts.items():
        (directory / name).write_text(content, encoding="utf-8")
    if report.get("profile_inputs"):
        inputs = directory / "inputs"
        inputs.mkdir()
        for index, item in enumerate(report["profile_inputs"]):
            # The destination depends only on our enumeration, never source paths or role text.
            (inputs / f"profile-{index:03d}.json").write_bytes(item["raw_json"].encode("utf-8"))
    runs_dir = directory / "runs"
    runs_dir.mkdir()
    for point in report["points"]:
        for run in point["runs"]:
            target = runs_dir / f"point-{point['point_index']:03d}-rep-{run['replication']:03d}.json"
            target.write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return directory


def _html(report: dict) -> str:
    clients = report["scan_axis"] == "clients"
    axis = "在线 client 数" if clients else "开放到达率 requests/s"
    rows = []
    for point in report["points"]:
        p99 = point["pooled_latency_ms"]["end_to_end"]["p99"]
        samples = point["sample_counts"]["end_to_end"]
        rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
            point["clients" if clients else "offered_rps"], point["classification"],
            point["slo_pass"], samples, f"{point['mean_window_completed_requests_per_s']:.6g}",
            "—" if p99 is None else f"{p99:.6g}", point["replication_slo_outcomes"])) + "</tr>")
    thresholds = escape(json.dumps(report["plan"]["slo"], ensure_ascii=False))
    boundaries = escape(json.dumps(report["boundaries"], ensure_ascii=False, indent=2))
    joint = _constraints_html(report) if "constraints" in report["plan"] else ""
    figures = "\n".join(f'<figure><img src="{metric}.svg" alt="{escape(label)}"/>'
                         f'<figcaption>{escape(label)}；全部重复均保留，n 为相应指标样本数（吞吐图为完成请求数）。</figcaption></figure>'
                         for metric, label in (("throughput", "窗口内完成吞吐"), ("ttft", "首 token 经验 P99"),
                                               ("tpot", "每输出 token 经验 P99"), ("end_to_end", "端到端经验 P99")))
    return f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>负载扫描实验报告</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#172536;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:.5rem;border-bottom:1px solid #ccd4dd;text-align:right}}
th:nth-child(2),td:nth-child(2){{text-align:left}}figure{{margin:1.5rem 0}}img{{max-width:100%;height:auto}}
figcaption{{color:#526273;font-size:14px}}pre{{white-space:pre-wrap;background:#f3f5f7;padding:1rem}}
</style><h1>负载扫描实验报告</h1>
<p>本报告为 CPU 模型预测；没有新增真机测量。扫描轴为 {axis}，共 {len(report['points'])} 点，
每点 {report['plan']['replications']} 次独立重复。SLO 阈值：<code>{thresholds}</code>。</p>
<p>经验 P99 仅来自完成请求，但失败、超时、未完成会否决整点 SLO；空样本或低于声明最小样本数返回未知。
各重复均须通过，不能只选健康 seed。窗口内完成吞吐按窗口内完成数 / 窗口时长计算，
原模拟器包含 drain 的到达队列吞吐另保留在 CSV / JSON，不等同于设备稳态服务率。</p>
<table><thead><tr><th>{axis}</th><th>有限观察判据</th><th>SLO 通过</th><th>完成 n</th>
<th>窗口吞吐 req/s</th><th>E2E P99 ms</th><th>各重复 SLO</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{joint}{figures}<h2>观察到的边界</h2><p>仅报告相邻网格 bracket 和实际观察到的最大通过点，不插值、不外推。
随机曲线可非单调；有限时间中的排队增长和 drain 完成情况不是数学稳定性证明。
闭环 client 数表示在线用户，独立于调度器 max_batch_size。</p><pre>{boundaries}</pre>
<p>可审计源：<a href="report.json">完整报告与生成请求</a>、<a href="points.csv">逐重复指标 CSV</a>、
<code>runs/</code> 下逐次原结果、事件及 provenance。经验分位数采用 (n−1)q 的线性插值。</p>
</html>'''


def _constraints_html(report: dict) -> str:
    parts = ["<h2>显存、延迟与吞吐联合约束</h2><p>旧 SLO 结果保留原含义；joint 仅在每个重复均满足全部约束、"
             "旧 SLO 通过且未观察到过载时通过。UNKNOWN 不获得合格名次。余量统一为无量纲值，越接近或小于 0 越限制。</p>"]
    for point in report["points"]:
        label = point["clients"] if point["clients"] is not None else point["offered_rps"]
        parts.append(f'<details><summary>负载 {escape(str(label))} · joint {point["joint_verdict"]} · '
                     f'约束 {point["constraint_assessment"]["verdict"]}</summary>')
        parts.append("<table><thead><tr><th>检查</th><th>范围</th><th>实值</th><th>关系</th><th>界限</th>"
                     "<th>单位</th><th>无量纲余量</th><th>判定</th><th>说明</th></tr></thead><tbody>")
        for check in point["constraint_assessment"]["checks"]:
            values = [check["id"], json.dumps(check["scope"], ensure_ascii=False), check["actual"],
                      check["operator"], check["limit"], check["unit"], check["normalized_headroom"],
                      check["verdict"], check["reason"]]
            parts.append("<tr>" + "".join(f"<td>{escape(str(value)) if value is not None else '—'}</td>"
                                         for value in values) + "</tr>")
        parts.append("</tbody></table></details>")
    return "\n".join(parts)
