"""Independent, CPU-only reveal of the preregistered Qwen3-4B forward study.

This script does not import the simulator, replay measured arrivals, fit costs,
execute evidence, or contact a server. It scores every frozen cell only after
formal-status is terminal. Scheduler token-ready and API-feedback are separate
clocks/boundaries: API feedback is used for client causality and final drain.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

PROTOCOL = "finite_cohort_closed_loop_v1"
COHORTS = {"baseline": [256], "short": [64, 128], "long": [512, 1024]}
GRID = [1, 2, 4, 8, 16, 32]
METRIC_LIMITS = {"ttft_p99_ms": .25, "tpot_p99_ms": .25,
                 "end_to_end_p99_ms": .25, "elapsed_output_tokens_per_s": .15}
SLA = {"ttft_p99_ms": 500, "tpot_p99_ms": 50, "end_to_end_p99_ms": 500,
       "elapsed_output_tokens_per_s_min": 20}


class ForwardEvidenceError(ValueError):
    """Missing, changed, nonterminal, or inconsistent primary evidence."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ForwardEvidenceError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def compact_ascii(value: Any) -> bytes:
    """Runtime/freeze raw protocol uses json.dumps' ASCII-escaped default.

    This intentionally differs from workload.canonical_sha256's UTF-8 JSON
    protocol for canonical_prediction_payload_sha256. Both commitments survive
    even when a model/device/note contains non-ASCII text.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def runtime_digest(value: Any) -> str:
    return hashlib.sha256(compact_ascii(value)).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _unique(items: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def decode_json(raw: bytes) -> Any:
    def bad_constant(value: str) -> None:
        raise ForwardEvidenceError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=bad_constant)


def read_json(path: Path) -> Any:
    return decode_json(path.read_bytes())


def integer(value: Any, name: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, name + " must be an integer >= " + str(minimum))
    return value


def number(value: Any, name: str, minimum: float = 0) -> float:
    require(type(value) in (int, float), name + " must be numeric")
    result = float(value)
    require(math.isfinite(result) and result >= minimum, name + " must be finite and nonnegative")
    return result


def sha_text(value: Any, name: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), name + " requires SHA256")
    return value


def safe_path(root: Path, name: str) -> Path:
    require(isinstance(name, str) and name, "empty evidence path")
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "unsafe evidence path: " + name)
    target = root / path
    require(target.resolve().is_relative_to(root.resolve()), "evidence escapes root: " + name)
    require(not target.is_symlink(), "symlink evidence: " + name)
    return target


def verify_files(root: Path, files: dict, *, exhaustive: bool = False) -> None:
    require(isinstance(files, dict) and files, "nonempty file commitment required")
    for name, expected in files.items():
        target = safe_path(root, name)
        require(target.is_file(), "missing committed file: " + name)
        require(file_sha(target) == sha_text(expected, name), "file hash mismatch: " + name)
    if exhaustive:
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        require(actual == set(files), "job file inventory differs from committed inventory")


def utc(value: str) -> datetime:
    require(isinstance(value, str), "UTC timestamp required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162 - preserve Python 3.10 UTC parsing
    require(result.tzinfo is not None and result.utcoffset() == timedelta(0),
            "timezone-aware UTC timestamp required")
    return result


def percentile(values: list[float], fraction: float = .99) -> float:
    require(bool(values), "quantile requires observations")
    ordered = sorted(number(x, "quantile value") for x in values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def request_stream(spec: dict, replication: int) -> list[dict]:
    seed = digest(["open-loop-v1", spec["seed"], replication, PROTOCOL + "/categorical_input_lengths"])
    rng = random.Random(int(seed, 16))
    return [{"request_id": f"{spec['study_id']}/{spec['cohort']}/r{replication}/q{i}",
             "input_tokens": rng.choice(spec["input_lengths"]), "output_tokens": spec["output_tokens"],
             "prompt_token_id": spec.get("prompt_token_id", 100), "timeout_ms": float(spec["timeout_ms"]),
             "priority": 0} for i in range(spec["requests_per_run"])]


def policy_sha(spec: dict, concurrency: int, replication: int) -> str:
    normalized = {**spec, "think_ms": float(spec["think_ms"]), "timeout_ms": float(spec["timeout_ms"]),
                  "max_run_ms": float(spec["max_run_ms"]), "prompt_token_id": spec.get("prompt_token_id", 100)}
    return digest({"protocol": PROTOCOL, "policy": {k: v for k, v in normalized.items() if k != "concurrency_grid"},
                   "concurrency": concurrency, "replication": replication,
                   "release_rule": "terminal_feedback_then_think",
                   "stop_rule": "offer_exactly_N_then_drain; no_future_arrival_replay"})


def job_key(job: dict) -> tuple[str, int, int]:
    return job["cohort"], integer(job["concurrency"], "concurrency", 1), integer(job["replication"], "replication")


def _production_specs(specs: dict[str, dict]) -> set[tuple[str, int, int]]:
    require(set(specs) == set(COHORTS), "exactly three preregistered cohorts required")
    for name, spec in specs.items():
        require(spec["schema_version"] == 1 and spec["cohort"] == name and spec["input_lengths"] == COHORTS[name],
                "cohort specification differs from preregistration")
        require(spec["study_id"] == "qwen4b-online-20261117" and spec["seed"] == 2026111703,
                "study ID/seed differ from preregistration")
        require(spec["concurrency_grid"] == GRID and spec["replications"] == 3
                and spec["requests_per_run"] == 512 and spec["output_tokens"] == 16,
                "forward grid/N/output target differs from preregistration")
        require(spec["think_ms"] == 0 and spec["timeout_ms"] == 30000 and spec["max_run_ms"] == 900000,
                "forward timeout/think/drain policy differs from preregistration")
    return {(name, c, r) for name in COHORTS for c in GRID for r in range(3)}


def _same(actual: float, expected: float, name: str, tolerance: float = 1e-6) -> None:
    require(math.isclose(number(actual, name), number(expected, name), rel_tol=1e-10, abs_tol=tolerance),
            name + " disagrees with raw evidence")


def _absolute_ns(origin: int, timestamp_ms: Any) -> int:
    number(timestamp_ms, "Scheduler timestamp")
    delta = Decimal(str(timestamp_ms)) * 1_000_000
    require(delta == delta.to_integral_value(), "Scheduler timestamp lacks nanosecond precision")
    return origin + int(delta)


def _metrics(rows: list[dict], elapsed_ms: float) -> dict:
    require(elapsed_ms > 0, "positive elapsed time required")
    return {"ttft_p99_ms": percentile([r["ttft_ms"] for r in rows]),
            "tpot_p99_ms": percentile([r["tpot_ms"] for r in rows]),
            "end_to_end_p99_ms": percentile([r["end_to_end_ms"] for r in rows]),
            "elapsed_output_tokens_per_s": sum(r["output_tokens_actual"] for r in rows) * 1000 / elapsed_ms,
            "elapsed_requests_per_s": len(rows) * 1000 / elapsed_ms,
            "elapsed_ms": elapsed_ms, "completed_requests": len(rows),
            "output_tokens": sum(r["output_tokens_actual"] for r in rows)}


def recompute_server_run(capture: dict, feedback: dict, spec: dict, concurrency: int,
                         replication: int) -> dict:
    """Derive token-ready metrics independently of producer trace/results."""
    require(capture.get("source", {}).get("kind") == "measured", "Scheduler capture is not measured")
    metadata = capture["metadata"]
    require(capture.get("capture_closed") is True and not capture.get("capture_errors")
            and metadata.get("canonical_eligible") is True and metadata.get("fresh_state_verified") is True,
            "Scheduler capture must be fresh, closed and error-free")
    require(metadata.get("timing_boundary") == "server_token_ready_v1", "incorrect Scheduler boundary")
    require(isinstance(capture.get("clock_domain"), str) and capture["clock_domain"].startswith("linux-boot:"),
            "actual Linux Scheduler clock domain required")
    origin = integer(capture["clock_origin_monotonic_ns"], "Scheduler clock origin", 1)
    require(feedback.get("status") == "COMPLETED" and feedback.get("resources_drained") is True,
            "API cohort incomplete or not drained")
    require(feedback.get("timing_boundary") == "runtime_API_feedback_v1"
            and feedback.get("protocol") == PROTOCOL, "incorrect API boundary/protocol")
    clock = feedback["clock"]
    require(clock.get("name") == "perf_counter_ns" and clock.get("unit") == "nanoseconds"
            and clock.get("injected") is False, "actual monotonic API clock required")
    require(feedback["spec"] == spec and feedback["concurrency"] == concurrency
            and feedback["replication"] == replication, "API policy differs from frozen cell")
    expected = request_stream(spec, replication)
    require(feedback["request_stream"]["requests"] == expected, "API input stream differs from independent RNG")
    stream_sha = digest(expected)
    require(feedback["request_stream_sha256"] == stream_sha
            and feedback["request_stream"]["request_stream_sha256"] == stream_sha,
            "API stream checksum mismatch")
    require(feedback["policy_sha256"] == policy_sha(spec, concurrency, replication), "API policy checksum mismatch")
    n, target = spec["requests_per_run"], spec["output_tokens"]
    for key in ("expected_requests", "offered_count", "completed_count", "terminal_count"):
        require(feedback[key] == n, "incomplete API " + key)
    for key in ("not_offered_count", "failed_count", "timed_out_count", "unfinished_count"):
        require(feedback[key] == 0, "API failure/timeout: " + key)
    require(not feedback.get("errors") and feedback.get("error_code") is None
            and not feedback.get("unacknowledged_native_request_ids"), "API errors remain")
    require(feedback["client_current_outstanding"] == [0] * concurrency
            and feedback["client_peak_outstanding"] == [1] * min(n, concurrency)
            and feedback["peak_logical_outstanding"] <= concurrency
            and feedback["peak_tracked_native_outstanding"] <= concurrency,
            "API outstanding ledger inconsistent")
    start, end = integer(feedback["formal_start_ns"], "formal_start", 1), integer(feedback["formal_end_ns"], "formal_end", 1)
    require(start < end, "API formal interval reversed")
    api_rows = feedback["requests"]
    require(len(api_rows) == n, "API request count differs")
    require([r["request_id"] for r in api_rows] == [r["request_id"] for r in expected],
            "API request order differs from frozen global stream")
    native = [r["native_request_id"] for r in api_rows]
    require(all(isinstance(x, str) and x for x in native) and len(set(native)) == n,
            "native/API mapping must be one-to-one")
    cap_rows = capture["requests"]
    cap_ids = [r["request_id"] for r in cap_rows]
    require(len(cap_ids) == len(set(cap_ids)), "duplicate Scheduler request ID")
    cap_by_id = {r["request_id"]: r for r in cap_rows}
    require(set(native) <= cap_by_id.keys(), "API native ID missing in Scheduler capture")
    events: dict[str, list[dict]] = {name: [] for name in native}
    for event in capture["events"]:
        if event["request_id"] in events:
            events[event["request_id"]].append(event)
    # Audit the raw step feedback independently, including sparse terminal-only
    # API output delivery. API first-token timestamps are never server TTFT.
    steps = feedback["step_feedback"]
    require(len(steps) == feedback["step_count"] and bool(steps), "API step ledger incomplete")
    step_index = {}
    last_step_end = start
    for i, step in enumerate(steps):
        require(step["step_index"] == i, "API step ordering differs")
        before, after = integer(step["step_before_ns"], "API step start"), integer(step["feedback_ns"], "API step end")
        require(last_step_end <= before <= after <= end, "API step monotonic interval invalid")
        last_step_end = after
        records = step["records"]
        ids = [r["native_request_id"] for r in records]
        require(len(ids) == len(set(ids)) and set(ids) <= set(native), "API step has duplicate/unknown native ID")
        step_index[i] = step
    client_last: dict[int, dict] = {}
    result = []
    first_enqueue = end
    previous_submit = start
    feedback_count = 0
    for i, (row, offered) in enumerate(zip(api_rows, expected)):
        name, router = row["native_request_id"], row["request_id"]
        require(row.get("router_request_id") == router and row["stream_index"] == i, "router/stream ID mismatch")
        require(all(row.get(k) == v for k, v in offered.items()), "API offered metadata differs")
        require(row.get("status") == "COMPLETED" and row.get("output_tokens_actual") == target
                and row.get("output_tokens_target") == target and row.get("output_target_satisfied") is True
                and row.get("output_ledger_valid") is True and row.get("error_code") is None,
                "API request failed or has incomplete outputs")
        require(row.get("terminal_source") == "native_finished_feedback" and row.get("abort_acknowledged") is False,
                "unexpected API terminal source")
        before, after = integer(row["submit_before_ns"], "submit before"), integer(row["submit_after_ns"], "submit after")
        terminal = integer(row["terminal_api_feedback_ns"], "terminal feedback")
        require(previous_submit <= before <= after <= terminal <= end, "API submission/terminal clocks invalid")
        previous_submit = before
        require(row["request_deadline_ns"] == before + math.ceil(float(spec["timeout_ms"]) * 1e6)
                and terminal <= row["request_deadline_ns"], "API completion past timeout")
        client = integer(row["client_id"], "client ID")
        require(client < concurrency, "client outside concurrency")
        previous = client_last.get(client)
        require(row["preceding_request_id"] == (previous["request_id"] if previous else None),
                "client predecessor chain differs")
        if previous:
            require(before >= previous["terminal_api_feedback_ns"] + math.ceil(float(spec["think_ms"]) * 1e6),
                    "client refilled before preceding API terminal feedback")
        client_last[client] = row
        captured = cap_by_id[name]
        require(captured["status"] == "COMPLETED" and captured["input_tokens"] == offered["input_tokens"]
                and captured["max_output_tokens"] == target and captured["output_tokens_actual"] == target,
                "Scheduler request shapes/output differ from API")
        prompt = [offered["prompt_token_id"]] * offered["input_tokens"]
        require(captured.get("prompt_token_ids") == prompt
                and captured.get("prompt_token_ids_sha256") == digest(prompt), "Scheduler prompt differs")
        raw_events = events[name]
        require(raw_events and raw_events[0]["event"] == "REQUEST_ARRIVAL"
                and raw_events[-1]["event"] == "REQUEST_FINISH", "Scheduler arrival/finish missing")
        arrivals = [e for e in raw_events if e["event"] == "REQUEST_ARRIVAL"]
        finishes = [e for e in raw_events if e["event"] == "REQUEST_FINISH"]
        tokens = [e for e in raw_events if e["event"] == "TOKEN_READY"]
        require(len(arrivals) == 1 and len(finishes) == 1 and len(tokens) == target
                and len(raw_events) == target + 2, "Scheduler event ledger incomplete/unexpected")
        require([integer(e["output_token_index"], "token index", 1) for e in tokens] == list(range(1, target + 1))
                and all(integer(e["token_count"], "token count", 1) == 1 for e in tokens), "Scheduler output indices/counts differ")
        absolute = [_absolute_ns(origin, e["timestamp_ms"]) for e in raw_events]
        require(absolute == sorted(absolute), "Scheduler per-request events out of order")
        enqueue, finish = absolute[0], absolute[-1]
        token_ns = [_absolute_ns(origin, e["timestamp_ms"]) for e in tokens]
        require(before <= enqueue <= after and token_ns[-1] == finish <= terminal
                and finishes[0].get("status") == "COMPLETED", "Scheduler/API clocks fail enclosure")
        require(_absolute_ns(origin, captured["arrival_ms"]) == enqueue, "Scheduler request/event arrival differs")
        ledger = row["feedback"]
        require(ledger and len(ledger) == row["step_feedback_count"], "API per-request feedback missing")
        cumulative, last_feedback = 0, after
        first_api = None
        for j, record in enumerate(ledger):
            count = integer(record["output_tokens_actual"], "API cumulative output")
            when = integer(record["feedback_ns"], "API feedback")
            require(cumulative <= count <= target and last_feedback <= when <= terminal,
                    "API cumulative feedback moved backwards")
            require(record["delta_output_tokens"] == count - cumulative
                    and record["finished"] is (j == len(ledger) - 1), "API delta/terminal ledger inconsistent")
            step = step_index.get(record["step_index"])
            require(step is not None and when == step["feedback_ns"]
                    and any(r == {"native_request_id": name, "output_tokens_actual": count,
                                  "finished": record["finished"]} for r in step["records"]),
                    "per-request API feedback differs from raw step record")
            if count:
                require(token_ns[count - 1] <= when, "API feedback precedes server token readiness")
                first_api = when if first_api is None else first_api
            cumulative, last_feedback = count, when
        require(cumulative == target and last_feedback == terminal
                and row["first_token_api_feedback_ns"] == first_api, "API terminal output ledger differs")
        feedback_count += len(ledger)
        first_enqueue = min(first_enqueue, enqueue)
        result.append({"request_id": router, "stream_index": i, "client_id": client,
                       "preceding_request_id": row["preceding_request_id"],
                       "enqueue_ns": enqueue, "token_ready_ns": token_ns,
                       "input_tokens": offered["input_tokens"], "output_tokens_actual": target,
                       "ttft_ms": (token_ns[0] - enqueue) / 1e6,
                       "tpot_ms": (token_ns[-1] - token_ns[0]) / (target - 1) / 1e6,
                       "end_to_end_ms": (token_ns[-1] - enqueue) / 1e6})
    require(feedback_count == feedback["feedback_record_count"]
            and feedback_count == sum(len(s["records"]) for s in steps), "API feedback rows omitted/added")
    require([r["client_id"] for r in api_rows[:min(n, concurrency)]] == list(range(min(n, concurrency))),
            "initial client cohort differs from declared concurrency")
    outstanding_events = [(r["submit_before_ns"], 1) for r in api_rows] + [(r["terminal_api_feedback_ns"], -1) for r in api_rows]
    outstanding = peak = 0
    for _, change in sorted(outstanding_events):
        outstanding += change
        peak = max(peak, outstanding)
        require(0 <= outstanding <= concurrency, "actual request intervals exceed concurrency")
    require(outstanding == 0 and peak == feedback["peak_logical_outstanding"]
            == feedback["peak_tracked_native_outstanding"], "reported concurrency peak differs from raw intervals")
    # Nonformal capture members may only be the earlier fixed warmup; a hidden
    # post-start request would make this cohort's resources and denominator differ.
    for row in cap_rows:
        if row["request_id"] not in events:
            require(_absolute_ns(origin, row["arrival_ms"]) < start, "unmapped formal Scheduler request")
    require(first_enqueue >= start and end - start <= spec["max_run_ms"] * 1e6,
            "formal interval exceeds frozen run budget")
    _same(feedback["elapsed_ms"], (end - start) / 1e6, "API elapsed")
    return {"metrics": _metrics(result, (end - first_enqueue) / 1e6), "requests": result,
            "server_origin_ns": first_enqueue, "formal_end_ns": end,
            "api_timing_boundary": "runtime_API_feedback_v1", "timing_boundary": "server_token_ready_v1"}


def crosscheck_producer(raw: dict, observed: dict, trace: dict) -> None:
    """Require published producer ledgers to agree; never score their aggregates."""
    require(observed.get("timing_boundary") == trace.get("timing_boundary") == "server_token_ready_v1",
            "producer boundary differs")
    rows = observed["requests"]
    require(len(rows) == len(raw["requests"]) and len({r["request_id"] for r in rows}) == len(rows),
            "producer request count/IDs differ")
    by_id = {r["request_id"]: r for r in rows}
    origin = raw["server_origin_ns"]
    for row in raw["requests"]:
        require(row["request_id"] in by_id, "producer dropped request")
        published = by_id[row["request_id"]]
        require(published["status"] == "COMPLETED" and published["output_tokens_actual"] == row["output_tokens_actual"],
                "producer completion/output differs")
        for key in ("ttft_ms", "tpot_ms", "end_to_end_ms"):
            _same(published[key], row[key], "producer " + key)
        _same(published["arrival_ms"], (row["enqueue_ns"] - origin) / 1e6, "producer arrival")
        require(len(published["token_times_ms"]) == len(row["token_ready_ns"]), "producer tokens truncated")
        for measured, target in zip(published["token_times_ms"], row["token_ready_ns"]):
            _same(measured, (target - origin) / 1e6, "producer token time")
    metrics = raw["metrics"]
    _same(observed["result"]["observed_elapsed_ms"], metrics["elapsed_ms"], "producer elapsed")
    for metric, key in (("ttft_p99_ms", "ttft"), ("tpot_p99_ms", "tpot"), ("end_to_end_p99_ms", "end_to_end")):
        _same(observed["result"]["latency_ms"][key]["p99"], metrics[metric], "producer " + metric)
    _same(observed["result"]["throughput"]["output_tokens_per_s"], metrics["elapsed_output_tokens_per_s"],
          "producer throughput")
    require(trace["events"] == observed["events"], "producer trace/evidence events differ")
    expected_events = []
    for row in raw["requests"]:
        request_id = row["request_id"]
        expected_events.append((request_id, "REQUEST_ARRIVAL", (row["enqueue_ns"] - origin) / 1e6, 0))
        for i, when in enumerate(row["token_ready_ns"], 1):
            expected_events.append((request_id, "TOKEN_READY", (when - origin) / 1e6, i))
        expected_events.append((request_id, "REQUEST_FINISH", (row["token_ready_ns"][-1] - origin) / 1e6, 0))
    actual = [(e["request_id"], e["event"], e["timestamp_ms"], e.get("output_token_index", 0)) for e in trace["events"]]
    require(len(actual) == len(expected_events), "producer trace events omitted/added")
    # Ordering across requests can be stabilized by the producer; the tuple keys
    # preserve every event while comparing times within nanosecond tolerance.
    a = sorted(actual, key=lambda x: (x[0], x[1], x[3]))
    b = sorted(expected_events, key=lambda x: (x[0], x[1], x[3]))
    for left, right in zip(a, b):
        require(left[:2] == right[:2] and left[3] == right[3], "producer trace event identity differs")
        _same(left[2], right[2], "producer trace timestamp")


def prediction_metrics(run: dict, expected: list[dict], spec: dict, concurrency: int, replication: int) -> dict | None:
    require(run["concurrency"] == concurrency and run["replication"] == replication
            and run["request_stream_sha256"] == digest(expected)
            and run["policy_sha256"] == policy_sha(spec, concurrency, replication), "prediction cell binding differs")
    require(run.get("timing_boundary") == "server_token_ready_v1", "prediction boundary differs")
    if run.get("status") != "COMPLETED" or run.get("resources_drained") is not True:
        return None
    rows = run["requests"]
    require(len(rows) == len(expected) and {r["request_id"] for r in rows} == {r["request_id"] for r in expected},
            "complete prediction request IDs differ")
    by_id = {r["request_id"]: r for r in rows}
    ledger = run["offered_requests"]
    require([r["request_id"] for r in ledger] == [r["request_id"] for r in expected], "prediction offered stream differs")
    last_by_client = {}
    computed = []
    for offered, frozen in zip(ledger, expected):
        require(all(offered.get(k) == v for k, v in frozen.items()), "prediction offered request metadata differs")
        row = by_id[frozen["request_id"]]
        times = row["token_times_ms"]
        arrival = number(row["arrival_ms"], "predicted enqueue")
        require(row["status"] == "COMPLETED" and row["output_tokens_actual"] == spec["output_tokens"]
                and len(times) == spec["output_tokens"] and times == sorted(times), "prediction token ledger incomplete")
        require(arrival <= times[0] <= times[-1], "prediction clock reversed")
        client = integer(offered["client_id"], "prediction client")
        require(client < concurrency, "prediction client outside concurrency")
        previous = last_by_client.get(client)
        require(offered["preceding_request_id"] == (previous["request_id"] if previous else None),
                "prediction predecessor chain differs")
        if previous:
            require(arrival >= previous["completion_ms"] + spec["think_ms"], "prediction refill before terminal")
        last_by_client[client] = row
        require(row["preceding_request_id"] == offered["preceding_request_id"] and row["client_id"] == client
                and row["input_tokens"] == frozen["input_tokens"], "prediction request/offer metadata differs")
        for key, value in (("ttft_ms", times[0] - arrival),
                           ("tpot_ms", (times[-1] - times[0]) / (len(times) - 1)),
                           ("end_to_end_ms", times[-1] - arrival), ("completion_ms", times[-1])):
            _same(row[key], value, "prediction " + key)
        computed.append({"ttft_ms": times[0] - arrival, "tpot_ms": (times[-1] - times[0]) / (len(times) - 1),
                         "end_to_end_ms": times[-1] - arrival, "output_tokens_actual": len(times)})
    elapsed = number(run["elapsed_ms"], "prediction elapsed")
    require(elapsed >= max(r["token_times_ms"][-1] for r in rows), "prediction elapsed omits drain")
    require(min(r["arrival_ms"] for r in rows) == 0, "prediction origin is not its own first enqueue")
    return _metrics(computed, elapsed)


def sla_state(metrics: dict | None) -> str:
    if metrics is None:
        return "unknown"
    return "safe" if (all(metrics[k] <= SLA[k] for k in ("ttft_p99_ms", "tpot_p99_ms", "end_to_end_p99_ms"))
                      and metrics["elapsed_output_tokens_per_s"] >= SLA["elapsed_output_tokens_per_s_min"]) else "unsafe"


def compare_metrics(predicted: dict | None, measured: dict | None) -> dict:
    pred_sla, real_sla = sla_state(predicted), sla_state(measured)
    errors, checks = {}, {}
    if predicted is not None and measured is not None:
        for metric, limit in METRIC_LIMITS.items():
            actual, forecast = measured[metric], predicted[metric]
            relative = abs(forecast - actual) / actual if actual > 0 else (0 if forecast == 0 else None)
            errors[metric] = relative
            checks[metric] = relative is not None and relative <= limit
    unknown = predicted is None or measured is None
    false_safe = pred_sla == "safe" and real_sla == "unsafe"
    return {"predicted_sla": pred_sla, "measured_sla": real_sla, "unknown": unknown,
            "false_safe": false_safe, "false_unsafe": pred_sla == "unsafe" and real_sla == "safe",
            "relative_errors": errors, "metric_pass": checks,
            "status": "PASS" if not unknown and all(checks.values()) and not false_safe else "FAIL"}


def project_runner_identity_to_core(binding: dict) -> dict:
    """Bridge the two declared boundaries without dropping execution identity.

    Runner warmup captures use the original synchronized stage boundary. Their
    constructor identity may precede the actual Runtime fingerprint, which is
    recorded in metadata after binding. Only that missing SHA and the explicit
    boundary name are projected; model/device/layout/version fields stay exact.
    """
    metadata = binding["metadata"]
    config_sha = sha_text(metadata["runtime_config_sha256"], "Runner Runtime config")
    require(runtime_digest(metadata["runtime_execution_config"]) == config_sha,
            "Runner actual Runtime config checksum differs")
    expected = decode_json(canonical(binding["identity"]))
    runtime = expected["runtime"]
    require(runtime["timing_boundary"] == "step_includes_output_and_sampling_v1",
            "Runner stage timing boundary differs")
    require("runtime_config_sha256" not in runtime or runtime["runtime_config_sha256"] == config_sha,
            "Runner Runtime identity SHA differs from actual metadata")
    runtime["timing_boundary"] = "engine_core_step_wall_v1"
    runtime["runtime_config_sha256"] = config_sha
    return expected


def strategy_domain_view(strategy: dict) -> dict:
    """Restore only the two optional None controls omitted by strategy_dict."""
    view = decode_json(canonical(strategy))
    view.setdefault("max_active_requests", None)
    if isinstance(view.get("memory"), dict):
        view["memory"].setdefault("gpu_memory_utilization", None)
    return view


def validate_prediction_domain(prediction: dict, model: dict, model_raw: bytes,
                               frozen_experiment: dict | None = None) -> None:
    """Bind prediction scheduling, selectors and the original cost-file bytes."""
    experiment, domain = prediction["experiment"], model["domain"]
    require(strategy_domain_view(experiment["strategy"]) == strategy_domain_view(domain["strategy"])
            and experiment["infra"] == domain["infra"],
            "prediction strategy/infra domain differs from calibration")
    calibration = experiment["calibration"]
    require(calibration["backend"] == "runtime" and calibration["timing_boundary"] == "engine_core_step_wall_v1",
            "prediction calibration backend/boundary differs")
    for group in ("runtime", "nominal_calibration"):
        require(all(calibration.get(key) == value for key, value in domain[group].items()),
                "prediction calibration domain differs: " + group)
    if frozen_experiment is not None:
        require(strategy_domain_view(experiment["strategy"]) == strategy_domain_view(frozen_experiment["strategy"])
                and experiment["infra"] == frozen_experiment["infra"], "prediction strategy/infra differs from frozen experiment")
        require(calibration == frozen_experiment["calibration"], "prediction differs from frozen calibration configuration")
    require(prediction["experiment_sha256"] == digest(experiment), "prediction experiment checksum differs")
    inputs = prediction["profile_inputs"]
    model_sha = hashlib.sha256(model_raw).hexdigest()
    require(isinstance(inputs, list) and len(inputs) == 1, "prediction model bytes require one Runtime cost input")
    item = inputs[0]
    require(item.get("field") == "profile_path" and type(item.get("index")) is int and item["index"] == 0
            and item.get("reader") == "runtime_cost_v1" and item.get("source_kind") == "measured"
            and item.get("sha256") == model_sha and isinstance(item.get("raw_json"), str)
            and item["raw_json"].encode("utf-8") == model_raw, "prediction model bytes differ from frozen calibration")


def validate_runtime_capture(capture: dict, dataset: dict, summary: dict, receipt: dict, model: dict,
                             scheduler: dict, formal_start: int, feedback: dict | None = None,
                             runner_binding: dict | None = None) -> None:
    require(capture.get("kind") == "runtime_cost_raw_capture_v1" and capture.get("closed") is True
            and not capture.get("capture_errors"), "Runtime-cost raw capture incomplete")
    require(capture["session_id"] == summary["session_id"] == dataset["dataset_id"] == scheduler["session_id"],
            "Runtime/Scheduler session mismatch")
    acq = capture["acquisition"]
    config = receipt["runtime_execution_config"]
    prep = config["runtime_cost_preparation"]
    require(acq["preparation_policy"] == prep and acq["protocol"] == "instrumented_engine_core_step_wall_v1"
            and acq["timing_boundary"] == "engine_core_step_wall_v1"
            and acq["gpu_synchronization_added"] is False and acq["source_kind"] == "measured",
            "Runtime preparation/source/boundary differs")
    requested_cpu = prep["main_thread_cpu"]
    if requested_cpu is None:
        expected_affinity = receipt.get("native_main_thread_affinity")
        require(isinstance(expected_affinity, list) and bool(expected_affinity)
                and all(type(cpu) is int and cpu >= 0 for cpu in expected_affinity)
                and expected_affinity == sorted(set(expected_affinity)),
                "frozen native main-thread affinity declaration required")
    else:
        expected_affinity = [integer(requested_cpu, "frozen main-thread CPU")]
    require(prep["gpu_synchronization_added"] is False
            and prep["latency_adaptive"] is False and prep["formal_progress"] == "buffered_until_end"
            and prep["observer_source_sha256"] == receipt["code_files_sha256"]["runtime_cost_capture.py"],
            "forward preparation differs from frozen-affinity no-sync quiet policy")
    require(prep["warmup_batches"] == [1, 4, 8, 16] and prep["warmup_prompt_lengths"] == [64, 128, 256, 512, 1024]
            and prep["warmup_output_tokens"] == 16, "fixed20 warmup policy differs")
    require(config["environment"].get("GPU_MAX_HW_QUEUES") is None
            and config["model_config"]["enforce_eager"] is True and config["async_scheduling"] is False,
            "GPU queues/MRv1 eager Runtime domain differs")
    base = {k: v for k, v in config.items() if k not in {"actual_attention_backends", "torch_build", "sampling_policy"}}
    require(acq["core_execution_config"] == base
            and scheduler["metadata"]["actual_scheduler_config"] == base, "core/Runner/Scheduler domain differs")
    binding = acq["runner_binding"]
    require(binding["metadata"]["runtime_execution_config"] == config
            and binding["metadata"]["runtime_config_sha256"] == receipt["runtime_config_sha256"],
            "actual Runner Runtime binding differs")
    require(project_runner_identity_to_core(binding) == receipt["identity"]
            == scheduler["identity"] == summary["identity"], "Runner projection and Core identity differ")
    if runner_binding is not None:
        require(binding["sha256"] == runtime_digest(runner_binding)
                and runner_binding["metadata"] == binding["metadata"]
                and runner_binding["identity"] == binding["identity"]
                and runner_binding.get("capture_closed") is True and not runner_binding.get("capture_errors"),
                "raw Runner binding bytes/closure differ")
    require(acq["checkpoint"]["manifest_sha256"] == receipt["checkpoint_manifest_sha256"]
            and acq["checkpoint"]["manifest"] == receipt["checkpoint_manifest"], "actual checkpoint evidence differs")
    require(dataset["domain"] == model["domain"] and dataset["source"]["kind"] == "measured"
            and dataset["notes"]["raw_capture_sha256"] == runtime_digest(capture)
            and dataset["notes"]["acquisition"] == acq, "Runtime dataset/raw/domain binding differs")
    cohorts = acq["fixed_warmup_cohorts"]
    require(cohorts == summary["warmup"]["cohorts"] and [(r["batch_size"], r["input_tokens"], r["output_tokens"]) for r in cohorts]
            == [(b, n, 16) for b in (1, 4, 8, 16) for n in (64, 128, 256, 512, 1024)], "fixed20 warmup missing/reordered")
    formal = integer(acq["formal_started_ns"], "core formal start", 1)
    previous_end = 0
    for row in cohorts:
        require(previous_end <= row["started_ns"] < row["finished_ns"] < formal <= formal_start,
                "warmup/core/formal clock ordering invalid")
        previous_end = row["finished_ns"]
    for key in ("post_warmup_affinity", "post_formal_affinity"):
        affinity = acq[key]
        require(affinity["requested_cpu"] == requested_cpu and affinity["actual_main_thread_affinity"] == expected_affinity,
                "actual main-thread affinity differs from frozen preparation")
    require(summary["warmup"]["prefix_cache_disabled"] is True
            and summary["warmup"]["runner_capture_closed_before_formal"] is True
            and summary["warmup"]["initial_free_blocks"] == summary["warmup"]["post_warmup_free_blocks"],
            "warmup did not restore empty logical KV")
    iterations = capture["iterations"]
    require(iterations and [r["source_order"] for r in iterations] == list(range(1, len(iterations) + 1)),
            "Runtime raw iterations truncated/reordered")
    expected_records = []
    warm_ids = []
    last_end = 0
    formal_seen = False
    for row in iterations:
        require(row["status"] == "COMPLETED" and last_end <= row["started_ns"] <= row["finished_ns"],
                "Runtime iteration failed or clocks overlap")
        last_end = row["finished_ns"]
        _same(row["duration_ms"], (row["finished_ns"] - row["started_ns"]) / 1e6, "core wall duration")
        if row["capture_phase"] == "warmup":
            require(not formal_seen and row["finished_ns"] < formal, "warmup leaked into formal cost rows")
            warm_ids.append(row["record_id"])
            continue
        require(row["capture_phase"] == "formal" and row["started_ns"] >= formal, "unknown core phase")
        formal_seen = True
        if not row["members"]:
            continue
        members = row["members"]
        roles = [m["request_phase"] for m in members]
        phase = roles[0] if len(set(roles)) == 1 else "mixed"
        shape = {"phase": phase, "query_tokens": [m["query_tokens"] for m in members],
                 "context_lengths": [m["context_length"] for m in members]}
        if phase == "mixed":
            shape["request_phases"] = roles
        expected_records.append({"record_id": row["record_id"], "session_id": capture["session_id"],
                                 "group_id": "core-shape:" + digest(shape), "source_order": row["source_order"],
                                 **shape, "stage_index": 0, "duration_ms": row["duration_ms"], "workspace_peak_bytes": None})
    require(dataset["records"] == expected_records and dataset["notes"]["warmup_record_ids"] == warm_ids,
            "formal Runtime dataset rows differ from untrimmed raw iterations")
    require(expected_records, "no formal Runtime steps")
    if feedback is not None:
        formal_rows = [r for r in iterations if r["capture_phase"] == "formal"]
        api_returns = capture["api_returns"]
        steps = feedback["step_feedback"]
        require(len(api_returns) == len(steps), "core/API step capture count differs")
        previous_order = max((r["source_order"] for r in iterations if r["capture_phase"] == "warmup"), default=0)
        cursor = 0
        for outer, inner in zip(steps, api_returns):
            through = integer(inner["source_order_through"], "API core source order", 1)
            require(outer["step_before_ns"] <= inner["started_ns"] <= inner["finished_ns"] <= outer["feedback_ns"]
                    and through > previous_order, "API/core wrapper clock/order differs")
            selected = []
            while cursor < len(formal_rows) and formal_rows[cursor]["source_order"] <= through:
                selected.append(formal_rows[cursor])
                cursor += 1
            require(selected and all(inner["started_ns"] <= r["started_ns"] <= r["finished_ns"] <= inner["finished_ns"]
                                     for r in selected), "core steps fall outside their actual API wrapper")
            previous_order = through
        require(previous_order == iterations[-1]["source_order"], "formal core steps missing API return evidence")


def _frozen_bundle(study: Path, expected_receipt_sha256: str, code_root: Path) -> tuple[dict, dict, dict, list[dict], dict, dict]:
    frozen = study / "frozen-commitment"
    receipt_path = frozen / "freeze-receipt.json"
    require(file_sha(receipt_path) == sha_text(expected_receipt_sha256, "expected receipt"), "freeze receipt changed")
    receipt = read_json(receipt_path)
    require(receipt.get("kind") == "course_sla_prediction_freeze_receipt" and receipt.get("measurement_started") is False
            and receipt.get("prediction_artifacts_stored_locally") is True, "invalid freeze receipt")
    verify_files(frozen, receipt["files_sha256"])
    manifest = read_json(frozen / "manifest.json")
    for key, value in manifest.items():
        if key not in {"files_sha256", "kind"}:
            require(receipt.get(key) == value, "receipt/manifest differs: " + key)
    require(all(receipt["files_sha256"].get(k) == v for k, v in manifest["files_sha256"].items()),
            "receipt changed manifest file commitments")
    require(receipt["files_sha256"].get("manifest.json") == file_sha(frozen / "manifest.json"), "manifest not receipt-bound")
    verify_files(code_root, receipt["code_files_sha256"])
    require(runtime_digest(receipt["runtime_execution_config"]) == receipt["runtime_config_sha256"], "frozen Runtime config hash differs")
    require(digest(receipt["identity"]) == receipt["identity_sha256"], "frozen identity hash differs")
    require(runtime_digest(receipt["checkpoint_manifest"]) == receipt["checkpoint_manifest_sha256"], "frozen checkpoint manifest differs")
    specs = {cohort: read_json(frozen / (cohort + ".spec.json")) for cohort in COHORTS}
    expected_jobs = _production_specs(specs)
    require(file_sha(frozen / "experiment.json") == receipt["experiment_sha256"], "frozen experiment commitment differs")
    frozen_experiment = read_json(frozen / "experiment.json")
    for cohort in COHORTS:
        require(file_sha(frozen / (cohort + ".spec.json")) == receipt["specs_sha256"][cohort], "spec bytes differ")
    jobs = read_json(frozen / "jobs.json")
    keys = [job_key(job) for job in jobs]
    require(len(keys) == 54 and len(set(keys)) == 54 and set(keys) == expected_jobs, "frozen job grid incomplete/duplicated")
    model_raw = (frozen / "runtime-cost-model.json").read_bytes()
    model = decode_json(model_raw)
    require(model.get("kind") == "runtime_cost_model_v1" and model.get("source", {}).get("kind") == "measured",
            "primary model must be accepted measured Runtime costs")
    require(model["model_sha256"] == runtime_digest({k: v for k, v in model.items() if k != "model_sha256"}), "Runtime model hash differs")
    require(model["domain"]["runtime"]["runtime_config_sha256"] == receipt["runtime_config_sha256"], "model Runtime domain differs")
    training = model["raw_dataset"]
    require(training["source"]["kind"] == "measured" and training["domain"] == model["domain"]
            and runtime_digest(training) == model["training_dataset_sha256"], "model training lineage checksum/domain differs")
    training_sessions = {r["session_id"] for r in training["records"]}
    require(len(training_sessions) >= 2, "two distinct training processes required")
    predictions = {}
    for cohort in COHORTS:
        name = "primary-prediction-" + cohort + ".json.gz"
        require(name in receipt["files_sha256"], "gzip prediction bytes not remotely frozen")
        with gzip.open(frozen / name, "rb") as stream:
            raw = stream.read()
        prediction = decode_json(raw)
        require(raw == compact_ascii(prediction) + b"\n", "prediction is not compact canonical JSON+newline")
        require(hashlib.sha256(raw).hexdigest() == receipt["prediction_artifacts_sha256"][cohort]
                and digest(prediction) == receipt["canonical_prediction_payload_sha256"][cohort], "prediction reveal differs")
        require(prediction["spec"] == specs[cohort] and prediction["cohort"] == cohort
                and prediction["identity"] == receipt["identity"]
                and prediction["identity_sha256"] == receipt["identity_sha256"], "prediction identity/policy differs")
        validate_prediction_domain(prediction, model, model_raw, frozen_experiment)
        runs = prediction["runs"]
        require(len(runs) == 18 and {(r["concurrency"], r["replication"]) for r in runs} == {(c, r) for c in GRID for r in range(3)},
                "prediction cell grid incomplete/duplicated")
        for r, stream in enumerate(prediction["streams"]):
            require(stream["replication"] == r and stream["requests"] == request_stream(specs[cohort], r),
                    "frozen prediction stream differs from independent RNG")
        require(len(prediction["streams"]) == 3, "frozen prediction stream repetitions differ")
        predictions[cohort] = prediction
    return receipt, specs, predictions, jobs, model, manifest


def evaluate_forward(study_dir: Path, expected_receipt_sha256: str, code_root: Path) -> dict:
    """Audit complete attempts, reveal fixed predictions, and score raw evidence.

    Bundle/global integrity failures raise before scoring. Per-cell acquisition or
    integrity failures are retained as unknown FAIL cells; no favorable subset
    can become a passing study. The caller supplies the independently recorded
    remote receipt hash, rather than letting evidence assert its own commitment.
    """
    study = Path(study_dir)
    state = read_json(study / "formal-status.json")
    require(state.get("status") == "COMPLETED", "analysis forbidden before terminal COMPLETED status")
    require(state.get("receipt_sha256") == expected_receipt_sha256, "controller receipt differs")
    receipt, specs, predictions, jobs, model, _ = _frozen_bundle(study, expected_receipt_sha256, Path(code_root))
    attempts = state["completed_jobs"]
    require(len(attempts) == 54 and [job_key(j) for j in attempts] == [job_key(j) for j in jobs],
            "all 54 attempts must match frozen order with no trimming/retries")
    labels = [j["label"] for j in attempts]
    require(len(set(labels)) == 54 and all(isinstance(label, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", label) for label in labels),
            "job labels must be unique safe directory names")
    require((study / "formal").is_dir()
            and {p.name for p in (study / "formal").iterdir() if p.is_dir()} == set(labels),
            "formal attempt directories omitted/added relative to all frozen jobs")
    frozen_at = utc(receipt["frozen_at"])
    freeze_ns = integer(receipt["remote_freeze_monotonic_ns"], "remote freeze", 1)
    cells, sessions, verified_sessions = [], set(), set()
    training_sessions = {r["session_id"] for r in model["raw_dataset"]["records"]}
    last_attempt_end = frozen_at
    for declared, attempt in zip(jobs, attempts):
        cohort, c, r = job_key(attempt)
        require("label" not in declared or declared["label"] == attempt["label"], "controller label differs from frozen job")
        start, finish = utc(attempt["started_at_utc"]), utc(attempt["finished_at_utc"])
        require(last_attempt_end <= start < finish, "attempt UTC ordering differs from freeze/frozen sequence")
        last_attempt_end = finish
        spec = specs[cohort]
        run = next(x for x in predictions[cohort]["runs"] if (x["concurrency"], x["replication"]) == (c, r))
        predicted = prediction_metrics(run, request_stream(spec, r), spec, c, r)
        measured, problem = None, None
        try:
            folder = safe_path(study / "formal", attempt["label"])
            verify_files(folder, attempt["files_sha256"], exhaustive=True)
            require(attempt["returncode"] == 0 and attempt["worker_status"] == "COMPLETED", "worker attempt did not complete")
            required = {"summary.json", "api-feedback.json", "server-trace.json", "observed-run.json", "runner-binding.json", "runtime-cost-capture.json",
                        "runtime-cost-dataset.json", "formal-progress.json"}
            require(required <= attempt["files_sha256"].keys(), "raw evidence files missing from job commitment")
            summary = read_json(folder / "summary.json")
            feedback = read_json(folder / "api-feedback.json")
            require(summary["status"] == "COMPLETED" and summary["shutdown"] == "completed"
                    and summary["phase"] == "formal" and summary["freeze_receipt"] == receipt,
                    "worker receipt/status/shutdown differs")
            require(summary["source_sha256"] == receipt["code_files_sha256"]["vllm_sla_worker.py"], "worker source differs")
            require(summary["concurrency"] == c and summary["replication"] == r and summary["spec"] == spec,
                    "summary cell binding differs")
            require(summary["identity"] == receipt["identity"] and summary["runtime_config_sha256"] == receipt["runtime_config_sha256"]
                    and summary["runtime_execution_config"] == receipt["runtime_execution_config"], "actual Runtime identity differs")
            require(summary["session_id"] not in sessions, "Runtime session reused across independent cells")
            require(summary["session_id"] not in training_sessions, "forward Runtime session overlaps calibration training")
            sessions.add(summary["session_id"])
            require(start <= utc(summary["started_at"]) <= utc(summary["collection_started_at"])
                    <= utc(summary["collection_ended_at"]) <= utc(summary["ended_at"]) <= finish,
                    "worker/collection UTC ordering invalid")
            require(feedback["formal_start_ns"] > freeze_ns, "forward formal start does not follow remote receipt")
            cap_name = Path(summary["scheduler_capture_file"]).name
            require(bool(cap_name), "missing Scheduler capture filename")
            cap_relative = "captures/" + cap_name
            require(cap_relative in attempt["files_sha256"], "Scheduler capture not committed")
            capture = read_json(folder / cap_relative)
            require(capture["identity"] == receipt["identity"], "Scheduler identity differs")
            require(capture["clock_domain"].startswith("linux-boot:"), "Scheduler boot clock required")
            boot = capture["clock_domain"].removeprefix("linux-boot:")
            require(hashlib.sha256((boot + "\n").encode()).hexdigest() == receipt["boot_id_sha256"], "Scheduler/receipt boot clock differs")
            core = read_json(folder / "runtime-cost-capture.json")
            dataset = read_json(folder / "runtime-cost-dataset.json")
            validate_runtime_capture(core, dataset, summary, receipt, model, capture, feedback["formal_start_ns"],
                                     feedback, read_json(folder / "runner-binding.json"))
            progress = read_json(folder / "formal-progress.json")
            require(progress["quiet"] is True and bool(progress["records"]), "quiet buffered formal progress required")
            raw = recompute_server_run(capture, feedback, spec, c, r)
            require(summary["policy_sha256"] == feedback["policy_sha256"]
                    and summary["request_stream_sha256"] == feedback["request_stream_sha256"], "summary input lineage differs")
            require(file_sha(folder / "observed-run.json") == attempt["observed_run_sha256"] == summary["observed_run_sha256"],
                    "observed-run bytes differ")
            crosscheck_producer(raw, read_json(folder / "observed-run.json"), read_json(folder / "server-trace.json"))
            measured = raw["metrics"]
            verified_sessions.add(summary["session_id"])
        except (ForwardEvidenceError, KeyError, TypeError, ValueError, OSError, OverflowError) as error:
            problem = type(error).__name__ + ": " + str(error)
        comparison = compare_metrics(predicted, measured)
        cells.append({"cohort": cohort, "concurrency": c, "replication": r, "label": attempt["label"],
                      "returncode": attempt.get("returncode"), "worker_status": attempt.get("worker_status"),
                      "integrity_status": "PASS" if problem is None else "FAIL", "evidence_error": problem,
                      "predicted": predicted, "measured": measured, **comparison})
    return {"schema_version": 1, "kind": "independent_raw_forward_evaluation_v1",
            "status": "PASS" if all(c["status"] == "PASS" for c in cells) else "FAIL",
            "receipt_sha256": expected_receipt_sha256, "prediction_reveal_verified": True,
            "expected_cells": 54, "evaluated_cells": len(cells), "fresh_runtime_sessions": len(verified_sessions),
            "reported_runtime_sessions": len(sessions),
            "passing_cells": sum(c["status"] == "PASS" for c in cells),
            "false_safe_cells": sum(c["false_safe"] for c in cells),
            "false_unsafe_cells": sum(c["false_unsafe"] for c in cells), "unknown_cells": sum(c["unknown"] for c in cells),
            "measured_requests": sum(c["measured"]["completed_requests"] for c in cells if c["measured"]),
            "measured_output_tokens": sum(c["measured"]["output_tokens"] for c in cells if c["measured"]),
            "metric_limits": METRIC_LIMITS, "sla": SLA, "cells": cells,
            "boundaries": {"latency": "Scheduler enqueue to server TOKEN_READY; per-request TPOT before P99",
                           "throughput": "all cohort output tokens / (API formal_end - first Scheduler enqueue), including necessary drain",
                           "refill": "preceding terminal API feedback then think; no realized-arrival prediction replay"},
            "limitations": ["512-request empirical P99 has about five tail observations; no population confidence guarantee",
                            "Only the frozen Qwen3-4B MRv1 eager measured domain, including its declared CPU affinity, is evaluated",
                            "Integrity failures, numerical failures, false-unsafe and unknown cells are retained"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_forward(args.study, args.expected_receipt_sha256, args.code_root)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (args.out / "cells.csv").open("x", newline="") as stream:
        keys = ["cohort", "concurrency", "replication", "status", "integrity_status", "predicted_sla", "measured_sla",
                "false_safe", "false_unsafe", "unknown", "evidence_error"]
        writer = csv.DictWriter(stream, fieldnames=keys + ["error_" + m for m in METRIC_LIMITS])
        writer.writeheader()
        for cell in report["cells"]:
            writer.writerow({**{k: cell[k] for k in keys}, **{"error_" + k: cell["relative_errors"].get(k) for k in METRIC_LIMITS}})
    print(json.dumps({k: report[k] for k in ("status", "evaluated_cells", "passing_cells", "false_safe_cells", "unknown_cells")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
