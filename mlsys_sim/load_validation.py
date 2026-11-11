"""Frozen finite-cohort closed-loop predictions and independent load validation.

This protocol ends after a fixed number of offered requests and their drain.
It is deliberately separate from duration-limited open/closed-loop load sweeps.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import heapq
import math
from pathlib import Path
import re
import tempfile
from typing import Any, NoReturn

from .capacity import _bind_inputs, _profile_inputs
from .constraints import evaluate_simulation_constraints, validate_constraints
from .event_driver import EventDriver
from .load_sweep import _account_unreached, _restore_origin_paths
from .profiles import profile_identity
from .schema import Experiment, RequestSpec, SimulationError
from .state import SimulationResult, TERMINAL
from .workload import _rng, canonical_sha256


PROTOCOL = "finite_cohort_closed_loop_v1"
_METRICS = {"ttft_p99_ms", "tpot_p99_ms", "end_to_end_p99_ms", "elapsed_requests_per_s",
            "elapsed_output_tokens_per_s"}


def _invalid(message: str, code: str = "INVALID_LOAD_VALIDATION") -> NoReturn:
    raise SimulationError(code, message)


def _object(value: Any, required: set[str], optional: set[str], label: str) -> dict:
    if (not isinstance(value, dict) or any(not isinstance(key, str) for key in value)
            or required - value.keys() or value.keys() - required - optional):
        _invalid(f"{label} requires {sorted(required)}; optional keys: {sorted(optional)}")
    return value


def _int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _invalid(f"{label} must be an integer in [{minimum},{maximum}]")
    return value


def _num(value: Any, label: str, *, zero: bool = False) -> float:
    if type(value) not in (int, float):
        _invalid(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        _invalid(f"{label} exceeds finite numeric range")
    if not math.isfinite(result) or (result < 0 if zero else result <= 0):
        _invalid(f"{label} must be finite and {'nonnegative' if zero else 'positive'}")
    return result


def validate_closed_loop_spec(spec: dict) -> dict:
    """Normalize the finite-cohort protocol without altering the input."""
    _object(spec, {"schema_version", "study_id", "cohort", "concurrency_grid", "requests_per_run",
                   "replications", "seed", "think_ms", "input_lengths", "output_tokens", "timeout_ms", "max_run_ms"},
            {"prompt_token_id"}, "finite closed-loop study")
    _int(spec["schema_version"], "schema_version", 1, 1)
    for key in ("study_id", "cohort"):
        if not isinstance(spec[key], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", spec[key]):
            _invalid(f"{key} must be a filename-safe identifier of 1..80 characters")
    grid = spec["concurrency_grid"]
    if not isinstance(grid, list) or not 1 <= len(grid) <= 64:
        _invalid("concurrency_grid must have 1..64 strictly increasing integers")
    for value in grid:
        _int(value, "concurrency", 1, 10_000)
    if any(left >= right for left, right in zip(grid, grid[1:])):
        _invalid("concurrency_grid must be strictly increasing")
    requests = _int(spec["requests_per_run"], "requests_per_run", 1, 100_000)
    repeats = _int(spec["replications"], "replications", 1, 100)
    _int(spec["seed"], "seed", -(2**63), 2**63 - 1)
    outputs = _int(spec["output_tokens"], "output_tokens", 1, 4096)
    lengths = spec["input_lengths"]
    if not isinstance(lengths, list) or not 1 <= len(lengths) <= 1024:
        _invalid("input_lengths must contain 1..1024 distinct equally weighted categories")
    for value in lengths:
        _int(value, "input length", 1, 1_048_576)
    if len(set(lengths)) != len(lengths):
        _invalid("input_lengths categories must be distinct")
    runs = repeats * len(grid)
    if (runs > 256 or requests * runs > 250_000 or requests * outputs * runs > 4_000_000
            or requests * max(lengths) * runs > 256_000_000):
        _invalid("Study exceeds its explicit run/request/token bounds")
    result = deepcopy(spec)
    result["think_ms"] = _num(spec["think_ms"], "think_ms", zero=True)
    result["timeout_ms"] = _num(spec["timeout_ms"], "timeout_ms")
    result["max_run_ms"] = _num(spec["max_run_ms"], "max_run_ms")
    if result["max_run_ms"] > 3_600_000:
        _invalid("max_run_ms cannot exceed one hour per run")
    result["prompt_token_id"] = _int(spec.get("prompt_token_id", 100), "prompt_token_id", 0, 2**31 - 1)
    return result


def generate_request_stream(spec: dict, *, replication: int = 0) -> dict:
    """Freeze IDs and categorical lengths, independent of concurrency and timing.

    Requests have no fabricated arrival. A real or simulated client assigns
    client_id and enqueue time only after its preceding terminal feedback.
    """
    selected = validate_closed_loop_spec(spec)
    _int(replication, "replication", 0, selected["replications"] - 1)
    rng = _rng(selected["seed"], replication, f"{PROTOCOL}/categorical_input_lengths")
    rows = [{"request_id": f"{selected['study_id']}/{selected['cohort']}/r{replication}/q{index}",
             "input_tokens": rng.choice(selected["input_lengths"]), "output_tokens": selected["output_tokens"],
             "prompt_token_id": selected["prompt_token_id"], "timeout_ms": selected["timeout_ms"],
             "priority": 0} for index in range(selected["requests_per_run"])]
    return {"protocol": PROTOCOL, "replication": replication, "seed": selected["seed"],
            "requests": rows, "request_stream_sha256": canonical_sha256(rows),
            "length_distribution": "equal_weight_categorical_in_declared_order",
            "rng_stream": f"{PROTOCOL}/categorical_input_lengths",
            "prompt_rule": "repeat prompt_token_id input_tokens times; no tokenizer/chat-template inference"}


def closed_loop_policy_sha256(spec: dict, concurrency: int, replication: int) -> str:
    selected = validate_closed_loop_spec(spec)
    if type(concurrency) is not int or concurrency not in selected["concurrency_grid"]:
        _invalid("concurrency must identify one declared study cell")
    _int(replication, "replication", 0, selected["replications"] - 1)
    policy = {key: value for key, value in selected.items() if key != "concurrency_grid"}
    return canonical_sha256({"protocol": PROTOCOL, "policy": policy, "concurrency": concurrency,
                             "replication": replication, "release_rule": "terminal_feedback_then_think",
                             "stop_rule": "offer_exactly_N_then_drain; no_future_arrival_replay"})


def _failed_result(code: str) -> SimulationResult:
    empty = {"p50": None, "p95": None, "p99": None}
    return SimulationResult({"status": "UNRESOLVED", "error_code": code, "backend": "simulated",
        "hardware_validated": False, "simulated_ms": 0.0, "events_processed": 0,
        "counts": {"total": 0, "completed": 0, "failed": 0, "timed_out": 0, "unfinished": 0, "not_admitted": 0},
        "latency_ms": {name: dict(empty) for name in ("ttft", "tpot", "end_to_end", "itl")}, "memory": []}, [], [])


def _predict_cell(experiment: Experiment, spec: dict, concurrency: int, replication: int,
                  stream: dict) -> dict:
    driver: EventDriver | None = None
    ledger: list[dict] = []
    owners: dict[str, int] = {}
    seen: set[str] = set()
    sequence = 0
    issued = 0
    pending = [(0.0, client, client, None) for client in range(min(concurrency, spec["requests_per_run"]))]
    heapq.heapify(pending)
    sequence = len(pending)
    reason = None
    error = None
    prompts: dict[int, tuple[int, ...]] = {}

    def release_clients() -> None:
        nonlocal sequence
        assert driver is not None
        for name, state in driver.requests.items():
            if name in seen or state.status not in TERMINAL:
                continue
            seen.add(name)
            if issued >= spec["requests_per_run"]:
                continue
            ready = driver.now + spec["think_ms"]
            if not math.isfinite(ready) or (spec["think_ms"] > 0 and ready <= driver.now):
                _invalid("Think delay cannot advance finite representable time")
            heapq.heappush(pending, (ready, sequence, owners[name], name))
            sequence += 1

    try:
        driver = EventDriver(experiment)
        while True:
            if driver.stopped:
                reason = driver.stop_reason or "COHORT_TIME_LIMIT"
                break
            release_clients()
            if issued >= spec["requests_per_run"]:
                pending.clear()
            event = driver.next_event()
            choices = ([event[0]] if event is not None else []) + ([pending[0][0]] if pending else [])
            if not choices:
                break
            when = min(choices)
            if when > spec["max_run_ms"]:
                driver.set_time(spec["max_run_ms"])
                reason = "COHORT_TIME_LIMIT"
                break
            while (event := driver.next_event()) is not None and event[0] == when:
                driver.advance_one()
                if driver.stopped:
                    break
            if driver.stopped:
                reason = driver.stop_reason or "COHORT_TIME_LIMIT"
                break
            driver.set_time(when)
            release_clients()
            arrivals = []
            while pending and pending[0][0] == when and issued < spec["requests_per_run"]:
                _, _, client, previous = heapq.heappop(pending)
                row = stream["requests"][issued]
                deadline = when + row["timeout_ms"]
                if not math.isfinite(deadline) or deadline <= when:
                    _invalid("Request timeout deadline cannot advance finite representable time")
                if row["input_tokens"] not in prompts:
                    prompts[row["input_tokens"]] = (spec["prompt_token_id"],) * row["input_tokens"]
                request = RequestSpec(row["request_id"], when, row["input_tokens"], row["output_tokens"],
                                      row["priority"], row["timeout_ms"], prompts[row["input_tokens"]])
                ledger.append({**row, "stream_index": issued, "client_id": client, "arrival_ms": when,
                               "preceding_request_id": previous})
                owners[request.request_id] = client
                issued += 1
                arrivals.append(request)
            for request in arrivals:
                if driver.handled_events >= experiment.execution.max_events:
                    reason = "SIMULATION_EVENT_LIMIT"
                    break
                driver.accept_request(request)
            if reason is not None:
                break
            driver.schedule_ready()
        result = driver.finalize(reason=reason)
    except SimulationError as exc:
        reason, error = exc.code, str(exc)
        result = driver.finalize(reason=reason) if driver is not None else _failed_result(reason)
    if driver is not None:
        missing = [row for row in ledger if row["request_id"] not in driver.requests]
        if missing:
            _account_unreached(result, missing, reason or "COHORT_INCOMPLETE")
    by_id = {row["request_id"]: row for row in ledger}
    prompt_digests = {length: canonical_sha256(prompt) for length, prompt in prompts.items()}
    for row in result.requests:
        metadata = by_id[row["request_id"]]
        row.update(client_id=metadata["client_id"], stream_index=metadata["stream_index"],
                   input_tokens=metadata["input_tokens"], output_tokens_target=metadata["output_tokens"],
                   prompt_token_ids_sha256=prompt_digests[metadata["input_tokens"]],
                   preceding_request_id=metadata["preceding_request_id"])
    complete = result.result["counts"]["completed"]
    successful = complete == spec["requests_per_run"] and result.result["status"] == "COMPLETED"
    if not successful and result.result["status"] == "COMPLETED":
        result.result.update(status="UNRESOLVED", error_code=reason or "COHORT_INCOMPLETE")
    token_times = [time for row in result.requests for time in row.get("token_times_ms", [])]
    return {"concurrency": concurrency, "replication": replication,
            "status": "COMPLETED" if successful else "UNRESOLVED", "error": error,
            "policy_sha256": closed_loop_policy_sha256(spec, concurrency, replication),
            "request_stream_sha256": stream["request_stream_sha256"], "expected_requests": spec["requests_per_run"],
            "offered_count": issued, "not_offered_count": spec["requests_per_run"] - issued,
            "result": result.result, "requests": result.requests, "events": result.events,
            "offered_requests": ledger, "elapsed_ms": result.result["simulated_ms"],
            "last_token_ms": max(token_times) if token_times else None,
            "resources_drained": (not driver.has_inflight() and not driver.has_unfinished()) if driver is not None else False,
            "timing_boundary": "server_token_ready_v1",
            "metric_contract": {"protocol": PROTOCOL, "throughput": "finite_cohort_actual_tokens / elapsed_including_necessary_drain",
                                "steady_state_rate": False, "arrival_origin": "first_formal_enqueue_at_zero",
                                "feedback": "real_terminal_then_think; timeout/failure_release_client_but_veto_success"}}


def predict_closed_loop(experiment: Experiment, spec: dict) -> dict:
    """Freeze one finite stream per repetition and predict real completion refill."""
    from .config import validate_experiment

    selected = validate_closed_loop_spec(spec)
    validate_experiment(experiment)
    if experiment.execution.warmup or experiment.execution.initial_state != "empty_cache":
        _invalid("Finite-cohort prediction requires empty measured KV and no embedded warmup")
    if selected["prompt_token_id"] >= experiment.strategy.model.vocab_size:
        _invalid("prompt_token_id lies outside the frozen model vocabulary")
    effective = replace(experiment, workload=(),
                        evaluation=replace(experiment.evaluation, measurement_window_ms=None, drain=True, max_drain_ms=None),
                        execution=replace(experiment.execution, max_simulated_ms=selected["max_run_ms"]))
    inputs = _profile_inputs(effective)

    def predict(bound: Experiment) -> dict:
        streams = [generate_request_stream(selected, replication=index) for index in range(selected["replications"])]
        runs = []
        total_events = 0
        for concurrency in selected["concurrency_grid"]:
            for replication, stream in enumerate(streams):
                run = _predict_cell(bound, selected, concurrency, replication, stream)
                total_events += run["result"]["events_processed"]
                if total_events > 4_000_000:
                    _invalid("Finite-cohort study exceeds four million processed events")
                try:
                    unchanged = _profile_inputs(effective) == inputs
                except (SimulationError, UnicodeError) as exc:
                    raise SimulationError("INPUT_CHANGED", "Frozen Profile origin became unreadable") from exc
                if not unchanged:
                    _invalid("Profile bytes changed during finite-cohort prediction", "INPUT_CHANGED")
                runs.append(run)
        identity = profile_identity(effective)
        return {"schema_version": 1, "kind": "closed_loop_predictions", "protocol": PROTOCOL,
                "status": "COMPLETED" if all(run["status"] == "COMPLETED" for run in runs) else "UNRESOLVED",
                "study_id": selected["study_id"], "cohort": selected["cohort"], "spec": selected,
                "spec_sha256": canonical_sha256(selected), "identity": identity,
                "identity_sha256": canonical_sha256(identity), "experiment": asdict(effective),
                "experiment_sha256": canonical_sha256(asdict(effective)), "streams": streams,
                "profile_inputs": inputs, "runs": runs, "hardware_validated": False,
                "limitations": ["Model predictions use finite-cohort completion refill, not realized hardware arrivals",
                                "Finite cohort elapsed throughput is not a steady-state queue stability claim",
                                "No calibration, seed selection or holdout refitting occurs"]}

    if not inputs:
        return predict(effective)
    with tempfile.TemporaryDirectory(prefix="mlsys-finite-load-inputs-") as temporary:
        bound = _bind_inputs(effective, inputs, Path(temporary))
        paths = {}
        for item in inputs:
            value = getattr(bound.calibration, item["field"])
            path = value[item["index"]] if item["field"].endswith("paths") else value
            Path(path).chmod(0o400)
            paths[path] = item["origin_path"]
        report = predict(bound)
        # Identity/config hashes are based on the declared original paths, never random snapshots.
        report["experiment"] = asdict(effective)
        report["experiment_sha256"] = canonical_sha256(asdict(effective))
        return _restore_origin_paths(report, paths)


def _iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or len(value) > 128:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo is not None and result.utcoffset() is not None else None
    except (ValueError, OverflowError):
        return None


def _validate_contract(contract: dict) -> dict:
    _object(contract, {"schema_version", "prediction_frozen_at", "min_completed_samples", "timing_boundary",
                       "max_relative_error", "constraints"},
            {"max_absolute_error", "allow_test_fixture", "prediction_payload_sha256"}, "validation contract")
    _int(contract["schema_version"], "schema_version", 1, 1)
    if _iso(contract["prediction_frozen_at"]) is None:
        _invalid("prediction_frozen_at must be a timezone-aware ISO timestamp")
    _int(contract["min_completed_samples"], "min_completed_samples", 1, 100_000)
    if not isinstance(contract["timing_boundary"], str) or not contract["timing_boundary"]:
        _invalid("timing_boundary must be explicitly declared")
    result = deepcopy(contract)
    result["constraints"] = validate_constraints(contract["constraints"])
    result["constraints"]["min_completed_samples"] = max(result["constraints"]["min_completed_samples"],
                                                          contract["min_completed_samples"])
    if ("throughput" in result["constraints"]
            and result["constraints"]["throughput"]["basis"] != "elapsed"):
        _invalid("Finite-cohort validation throughput uses elapsed, never an arrival window or task makespan")
    thresholds = False
    for key in ("max_relative_error", "max_absolute_error"):
        selected = contract.get(key, {})
        _object(selected, set(), _METRICS, key)
        for name, limit in selected.items():
            _num(limit, f"{key}.{name}", zero=True)
            thresholds = True
    if not thresholds:
        _invalid("Validation needs at least one preregistered relative or absolute error tolerance")
    if type(contract.get("allow_test_fixture", False)) is not bool:
        _invalid("allow_test_fixture must be boolean")
    if "prediction_payload_sha256" in contract:
        digest = contract["prediction_payload_sha256"]
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            _invalid("prediction_payload_sha256 must be a lowercase SHA256")
    return result


def _merge_verdicts(values: list[str]) -> str:
    return "FAIL" if "FAIL" in values else "UNKNOWN" if "UNKNOWN" in values else "PASS"


def _guard(name: str, passing: bool | None, reason: str) -> dict:
    return {"id": name, "verdict": "PASS" if passing is True else "FAIL" if passing is False else "UNKNOWN",
            "reason": reason}


def _assess_evidence(run: dict, constraints: dict) -> dict:
    result, rows = run.get("result"), run.get("requests")
    if not isinstance(result, dict) or not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        _invalid("Each run needs a canonical result and complete request record array")
    return evaluate_simulation_constraints(SimulationResult(result, rows, []), constraints)


def _feedback_consistent(rows: list[dict], expected: list[dict], concurrency: int, think_ms: float) -> bool | None:
    order = {row["request_id"]: index for index, row in enumerate(expected)}
    by_client: dict[int, list[dict]] = {}
    for row in rows:
        client = row.get("client_id")
        if type(client) is not int:
            return None
        if not 0 <= client < min(concurrency, len(expected)):
            return False
        if not isinstance(row.get("request_id"), str) or row["request_id"] not in order:
            return False
        index = order[row["request_id"]]
        if "stream_index" in row and (type(row["stream_index"]) is not int or row["stream_index"] != index):
            return False
        if index < min(concurrency, len(expected)) and client != index:
            return False
        by_client.setdefault(client, []).append(row)
    for entries in by_client.values():
        entries.sort(key=lambda row: order[row["request_id"]])
        for previous, following in zip(entries, entries[1:]):
            end, start = previous.get("completion_ms"), following.get("arrival_ms")
            if (isinstance(end, bool) or not isinstance(end, (int, float))
                    or isinstance(start, bool) or not isinstance(start, (int, float))):
                return None
            if not math.isfinite(end) or not math.isfinite(start) or end < 0 or start < 0:
                return None
            boundary = end + think_ms
            if not math.isfinite(boundary):
                return None
            rounding = 4 * max(math.ulp(float(start)), math.ulp(float(boundary)))
            if start + rounding < boundary:
                return False
            if ("preceding_request_id" in following
                    and following["preceding_request_id"] != previous["request_id"]):
                return False
    return True


def _cohort_checks(run: dict, expected: list[dict], minimum: int, *, concurrency: int,
                   think_ms: float, max_run_ms: float) -> list[dict]:
    rows = run["requests"]
    by_id = {row["request_id"]: row for row in rows if isinstance(row.get("request_id"), str)}
    ids = {row["request_id"] for row in expected}
    complete = sum(row.get("status") == "COMPLETED" for row in rows)
    known = all(isinstance(row.get("request_id"), str) for row in rows)
    pairing = known and len(by_id) == len(rows) and set(by_id) == ids
    expected_by_id = {row["request_id"]: row for row in expected}
    inputs_match: bool | None = True
    targets_match: bool | None = True
    prompts_match: bool | None = True
    prompt_hashes = {length: canonical_sha256([expected[0]["prompt_token_id"]] * length)
                     for length in {row["input_tokens"] for row in expected}}
    for row in rows:
        name = row.get("request_id")
        target = expected_by_id.get(name) if isinstance(name, str) else None
        if target is None:
            continue
        for field, expected_value, kind in (("input_tokens", target["input_tokens"], "input"),
                                            ("output_tokens_target", target["output_tokens"], "target"),
                                            ("prompt_token_ids_sha256", prompt_hashes[target["input_tokens"]], "prompt")):
            observed_value = row.get(field, row.get("max_output_tokens") if kind == "target" else None)
            value = None if observed_value is None else (type(observed_value) is type(expected_value)
                                                        and observed_value == expected_value)
            if kind == "input":
                inputs_match = False if value is False or inputs_match is False else None if value is None else inputs_match
            elif kind == "target":
                targets_match = False if value is False or targets_match is False else None if value is None else targets_match
            else:
                prompts_match = False if value is False or prompts_match is False else None if value is None else prompts_match
    output_counts = all(row.get("status") != "COMPLETED" or
                        type(row.get("output_tokens_actual")) is int and
                        row["output_tokens_actual"] == expected[0]["output_tokens"] for row in rows)
    elapsed = run["result"].get("simulated_ms")
    time_in_budget = (None if type(elapsed) not in (int, float) or not math.isfinite(elapsed) else
                      0 <= elapsed <= max_run_ms)
    arrivals = [row.get("arrival_ms") for row in rows]
    valid_arrivals = [value for value in arrivals if isinstance(value, (int, float)) and not isinstance(value, bool)
                      and math.isfinite(value) and value >= 0]
    origin_known = bool(arrivals) and len(valid_arrivals) == len(arrivals)
    first_enqueue_zero = min(valid_arrivals) == 0 if origin_known else None
    return [_guard("cohort.complete_ids", pairing, "exactly_N_frozen_request_ids_required"),
            _guard("cohort.input_lengths", inputs_match, "actual_input_lengths_must_match_frozen_stream"),
            _guard("cohort.output_targets", targets_match, "actual_requested_output_targets_must_match_frozen_stream"),
            _guard("cohort.prompt_identity", prompts_match, "actual_prompt_token_id_hashes_must_match_frozen_stream"),
            _guard("cohort.closed_loop_feedback", _feedback_consistent(rows, expected, concurrency, think_ms),
                   "each_client_enqueue_requires_previous_terminal_feedback_plus_think"),
            _guard("cohort.time_budget", time_in_budget, "finite_cohort_elapsed_cannot_exceed_declared_max_run"),
            _guard("cohort.enqueue_origin", first_enqueue_zero, "times_must_be_rebased_to_first_formal_enqueue"),
            _guard("cohort.all_completed", complete == len(expected) and run["result"].get("status") == "COMPLETED",
                   "every_requested_cohort_member_must_complete"),
            _guard("cohort.output_lengths", output_counts, "completed_output_lengths_must_match_fixed_target"),
            _guard("cohort.drained", run.get("resources_drained"), "necessary_resource_drain_must_be_established"),
            _guard("cohort.samples", True if complete >= minimum else None, "minimum_completed_samples_required")]


def _safe_assessment(assessment: dict) -> dict:
    """Public reports retain aggregate checks, never request IDs, tokens or paths."""
    return {"verdict": assessment["verdict"],
            "checks": [{key: check[key] for key in ("id", "verdict", "reason", "actual", "operator", "limit",
                                                     "unit", "sample_count", "normalized_headroom")}
                       for check in assessment["checks"]]}


def _cell_summary(run: dict, assessment: dict) -> dict:
    counts = run["result"].get("counts")
    counts = counts if isinstance(counts, dict) else {}
    return {"status": run["result"].get("status"), "metrics": deepcopy(assessment["metrics"]),
            "counts": {name: counts.get(name)
                       for name in ("total", "completed", "failed", "timed_out", "unfinished", "not_admitted")},
            "offered_count": run.get("offered_count", len(run["requests"])),
            "not_offered_count": run.get("not_offered_count"), "constraints": _safe_assessment(assessment)}


def validate_closed_loop(predictions: dict, observations: dict, contract: dict) -> dict:
    """Compare policy-matched independent cohorts; never condition on real arrivals.

    A validation FAIL is a completed analysis result. It does not trigger fitting,
    resampling, seed deletion or changes to the frozen prediction.
    """
    selected = _validate_contract(contract)
    if not isinstance(predictions, dict) or predictions.get("kind") != "closed_loop_predictions":
        _invalid("Expected frozen closed_loop_predictions")
    raw_spec = predictions.get("spec")
    if not isinstance(raw_spec, dict):
        _invalid("Frozen predictions need their normalized finite-cohort spec")
    spec = validate_closed_loop_spec(raw_spec)
    spec_sha = canonical_sha256(spec)
    identity_sha = predictions.get("identity_sha256")
    if (predictions.get("spec_sha256") != spec_sha
            or canonical_sha256(predictions.get("identity")) != identity_sha):
        _invalid("Frozen prediction spec/identity hashes disagree", "CALIBRATION_MISMATCH")
    prediction_sha = canonical_sha256(predictions)
    if selected.get("prediction_payload_sha256", prediction_sha) != prediction_sha:
        _invalid("Prediction changed after its preregistered payload hash", "INPUT_CHANGED")
    _object(observations, {"schema_version", "study_id", "cohort", "spec_sha256", "identity_sha256", "runs"},
            {"protocol", "source", "identity"}, "closed-loop observations")
    _int(observations["schema_version"], "observations.schema_version", 1, 1)
    if (observations["study_id"] != spec["study_id"] or observations["cohort"] != spec["cohort"]
            or observations["spec_sha256"] != spec_sha or observations["identity_sha256"] != identity_sha):
        _invalid("Observed study/spec/runtime identity differs from frozen predictions", "CALIBRATION_MISMATCH")
    if "identity" in observations and canonical_sha256(observations["identity"]) != identity_sha:
        _invalid("Observed identity payload and identity_sha256 disagree", "CALIBRATION_MISMATCH")
    expected_keys = {(concurrency, replication) for concurrency in spec["concurrency_grid"]
                     for replication in range(spec["replications"])}

    def index_runs(value: Any, label: str) -> dict:
        if not isinstance(value, list) or len(value) > len(expected_keys):
            _invalid(f"{label} must be a bounded cell array")
        indexed = {}
        for run in value:
            if not isinstance(run, dict):
                _invalid(f"{label} entries must be objects")
            key = run.get("concurrency"), run.get("replication")
            if (type(key[0]) is not int or type(key[1]) is not int or key not in expected_keys or key in indexed):
                _invalid(f"{label} has duplicate or undeclared concurrency/replication cells")
            indexed[key] = run
        return indexed

    predicted = index_runs(predictions.get("runs"), "prediction runs")
    observed = index_runs(observations["runs"], "observation runs")
    if set(predicted) != expected_keys:
        _invalid("Frozen predictions must retain every declared cell")
    frozen = _iso(selected["prediction_frozen_at"])
    assert frozen is not None
    comparisons = []
    for concurrency in spec["concurrency_grid"]:
        for replication in range(spec["replications"]):
            key = concurrency, replication
            prediction = predicted[key]
            expected_stream = generate_request_stream(spec, replication=replication)
            policy_sha = closed_loop_policy_sha256(spec, concurrency, replication)
            for run in (prediction, observed.get(key)):
                if run is not None and (run.get("policy_sha256") != policy_sha or
                                         run.get("request_stream_sha256") != expected_stream["request_stream_sha256"]):
                    _invalid("Cell workload policy or fixed stream differs from frozen study", "CALIBRATION_MISMATCH")
            base = {"concurrency": concurrency, "replication": replication, "policy_sha256": policy_sha,
                    "request_stream_sha256": expected_stream["request_stream_sha256"]}
            if key not in observed:
                assessment = _assess_evidence(prediction, selected["constraints"])
                checks = _cohort_checks(prediction, expected_stream["requests"], selected["min_completed_samples"],
                                         concurrency=concurrency, think_ms=spec["think_ms"], max_run_ms=spec["max_run_ms"])
                verdict = _merge_verdicts([assessment["verdict"], *[check["verdict"] for check in checks], "UNKNOWN"])
                comparisons.append({**base, "verdict": verdict, "checks": [
                    *checks, _guard("observation.present", None, "declared_cell_not_collected")], "metrics": {},
                    "prediction_sla_verdict": assessment["verdict"], "observation_sla_verdict": "UNKNOWN",
                    "prediction": _cell_summary(prediction, assessment),
                    "observation": None})
                continue
            observation = observed[key]
            source = observation.get("source", observations.get("source"))
            if (not isinstance(source, dict) or source.get("kind") not in {"measured", "test_fixture"}
                    or not isinstance(source.get("reference"), str) or not source["reference"].strip()):
                _invalid("Each observation requires an explicit measured or test_fixture source")
            if source["kind"] == "test_fixture" and not selected.get("allow_test_fixture", False):
                _invalid("CPU fixture observations require allow_test_fixture", "TEST_FIXTURE_NOT_ALLOWED")
            if not isinstance(observation.get("session_id"), str) or not observation["session_id"]:
                _invalid("Each observed cell requires a capture session_id")
            predicted_assessment = _assess_evidence(prediction, selected["constraints"])
            observed_assessment = _assess_evidence(observation, selected["constraints"])
            start, end = _iso(observation.get("collection_started_at")), _iso(observation.get("collection_ended_at"))
            boundary_ok = (prediction.get("timing_boundary") == selected["timing_boundary"]
                           and observation.get("timing_boundary") == selected["timing_boundary"])
            checks = [_guard("prediction.frozen_before_measurement", None if start is None else start > frozen,
                             "formal_start_must_be_strictly_after_timezone_aware_prediction_freeze"),
                      _guard("measurement.time_order", None if start is None or end is None else end >= start,
                             "formal_end_must_follow_formal_start"),
                      _guard("comparison.timing_boundary", boundary_ok, "equal_declared_timing_boundary_required")]
            for label, run in (("prediction", prediction), ("observation", observation)):
                checks.extend({**check, "id": f"{label}.{check['id']}"} for check in
                              _cohort_checks(run, expected_stream["requests"], selected["min_completed_samples"],
                                             concurrency=concurrency, think_ms=spec["think_ms"], max_run_ms=spec["max_run_ms"]))
            error_rows = {}
            for metric in sorted(selected["max_relative_error"].keys() | selected.get("max_absolute_error", {}).keys()):
                p = predicted_assessment["metrics"].get(metric)
                m = observed_assessment["metrics"].get(metric)
                p_count = predicted_assessment["metrics"]["sample_counts"].get(
                    {"ttft_p99_ms": "ttft", "tpot_p99_ms": "tpot", "end_to_end_p99_ms": "end_to_end"}.get(metric),
                    predicted_assessment["metrics"]["completed_samples"])
                m_count = observed_assessment["metrics"]["sample_counts"].get(
                    {"ttft_p99_ms": "ttft", "tpot_p99_ms": "tpot", "end_to_end_p99_ms": "end_to_end"}.get(metric),
                    observed_assessment["metrics"]["completed_samples"])
                available = (boundary_ok and p is not None and m is not None
                             and p_count >= selected["min_completed_samples"] and m_count >= selected["min_completed_samples"])
                absolute = abs(p - m) if available else None
                relative = ((absolute / m if m else 0.0 if absolute == 0 else None)
                            if available else None)
                if relative is not None and not math.isfinite(relative):
                    relative = None
                metric_checks = []
                for kind, actual in (("max_relative_error", relative), ("max_absolute_error", absolute)):
                    if metric in selected.get(kind, {}):
                        limit = selected[kind][metric]
                        check = _guard(f"error.{metric}.{kind}", None if actual is None else actual <= limit,
                                       "abs_prediction_minus_measured_divided_by_measured" if kind == "max_relative_error"
                                       else "absolute_prediction_minus_measured")
                        check.update(actual=actual, limit=limit)
                        checks.append(check)
                        metric_checks.append(check["verdict"])
                error_rows[metric] = {"predicted": p, "measured": m, "absolute_error": absolute,
                                      "relative_error": relative, "predicted_sample_count": p_count,
                                      "measured_sample_count": m_count, "verdict": _merge_verdicts(metric_checks)}
            acquisition_verdict = _merge_verdicts([check["verdict"] for check in checks
                                                  if not check["id"].startswith("error.")])
            error_verdict = _merge_verdicts([row["verdict"] for row in error_rows.values()])
            acceptance = _merge_verdicts([acquisition_verdict, error_verdict, predicted_assessment["verdict"],
                                          observed_assessment["verdict"]])
            observed_summary = _cell_summary(observation, observed_assessment)
            observed_summary.update(source_kind=source["kind"], session_sha256=canonical_sha256(observation["session_id"]),
                                     timing_boundary=observation.get("timing_boundary"))
            comparisons.append({**base, "verdict": acceptance, "acquisition_verdict": acquisition_verdict,
                                "prediction_error_verdict": error_verdict,
                                "prediction_sla_verdict": predicted_assessment["verdict"],
                                "observation_sla_verdict": observed_assessment["verdict"],
                                "sla_agreement": predicted_assessment["verdict"] == observed_assessment["verdict"],
                                "checks": checks, "metrics": error_rows,
                                "prediction": _cell_summary(prediction, predicted_assessment), "observation": observed_summary})
    points = []
    for concurrency in spec["concurrency_grid"]:
        cells = [run for run in comparisons if run["concurrency"] == concurrency]
        points.append({"concurrency": concurrency, "verdict": _merge_verdicts([run["verdict"] for run in cells]),
                       "replication_verdicts": [run["verdict"] for run in cells],
                       "predicted_sla_verdict": _merge_verdicts([run.get("prediction_sla_verdict", "UNKNOWN") for run in cells]),
                       "observed_sla_verdict": _merge_verdicts([run.get("observation_sla_verdict", "UNKNOWN") for run in cells])})
    return {"schema_version": 1, "kind": "closed_loop_validation", "protocol": PROTOCOL, "status": "COMPLETED",
            "verdict": _merge_verdicts([run["verdict"] for run in comparisons]),
            "study_id": spec["study_id"], "cohort": spec["cohort"], "predictions_sha256": prediction_sha,
            "spec_sha256": spec_sha, "identity_sha256": identity_sha, "contract": selected,
            "contract_sha256": canonical_sha256(selected), "runs": comparisons, "points": points,
            "hardware_validated": False, "metric_contract": {
                "comparison": "same frozen finite-cohort policy and stream; actual arrivals and elapsed may differ",
                "elapsed_throughput": "actual emitted tokens / cohort elapsed including required drain; not steady state",
                "relative_error": "abs(predicted-measured)/measured; zero denominator undefined unless both zero",
                "pairing": "every declared concurrency/replication, no best-seed selection",
                "raw_artifacts": "not copied; only aggregate metrics, counts, hashes and checks retained",
                "freeze_timestamp": "claimed ISO ordering is checked; external remote receipt supplies independent proof",
                "no_refit": True}, "limitations": [
                    "A measured discrepancy is retained as FAIL; this function never modifies predictions or calibration",
                    "Server-ready and client/runtime delivery boundaries cannot be interchanged",
                    "A finite empirical tail does not certify broad hardware accuracy or stochastic queue stability"]}
