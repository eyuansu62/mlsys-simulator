"""Explicit server-token traces and vLLM client-delivery observations.

The two boundaries intentionally have different output contracts. A client SSE
message may contain several tokens and never establishes their server-ready
times. Only complete, ordered server-token events become a measured-run input
to the simulator's existing comparison tools.
"""

import codecs
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable, NoReturn
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .comparison import comparison_contract, extract_metrics
from .config import validate_experiment
from .profiles import profile_identity, validate_identity
from .reporting import _percentiles, report_simulation
from .schema import Experiment, SimulationError
from .serialization import experiment_dict
from .state import RequestState, TERMINAL


def _fail(message: str, code: str = "INVALID_RUNTIME_TRACE") -> NoReturn:
    raise SimulationError(code, message)


def _object(value: Any, required: set[str], optional: set[str], label: str,
            code: str = "INVALID_RUNTIME_TRACE") -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(f"{label} must be an object with string keys", code)
    if required - value.keys() or value.keys() - required - optional:
        _fail(f"{label} has missing or unknown fields", code)
    return value


def _number(value: Any, label: str, *, positive: bool = False,
            code: str = "INVALID_RUNTIME_TRACE") -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be a finite number", code)
    try:
        result = float(value)
    except (ValueError, OverflowError):
        _fail(f"{label} must be finite", code)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        _fail(f"{label} must be finite and {'positive' if positive else 'nonnegative'}", code)
    return result


def _integer(value: Any, label: str, minimum: int = 0,
             code: str = "INVALID_RUNTIME_TRACE") -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"{label} must be a signed 64-bit integer >= {minimum}", code)
    return value


def _text(value: Any, label: str, code: str = "INVALID_RUNTIME_TRACE") -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a nonempty string", code)
    return value


def _source(value: Any, allow_test_fixture: bool) -> dict:
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be boolean")
    source = _object(value, {"kind", "reference"}, set(), "source")
    if source["kind"] not in ("measured", "test_fixture"):
        _fail("Runtime evidence must declare measured or test_fixture source")
    _text(source["reference"], "source.reference")
    if source["kind"] == "test_fixture" and not allow_test_fixture:
        _fail("Fixture evidence needs explicit permission", "TEST_FIXTURE_NOT_ALLOWED")
    return dict(source)


def _json_payload(value: str) -> dict:
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                _fail("Duplicate JSON key in runtime response")
            result[key] = item
        return result

    try:
        parsed = json.loads(value, object_pairs_hook=pairs,
                            parse_constant=lambda _: _fail("Nonfinite JSON in runtime response"))
    except (ValueError, TypeError) as exc:
        raise SimulationError("INVALID_RUNTIME_TRACE", "Invalid JSON in runtime response") from exc
    if not isinstance(parsed, dict):
        _fail("Runtime response must be a JSON object")
    return parsed


class _EmptyMemory:
    allocations: dict[str, int] = {}

    @staticmethod
    def snapshot() -> list[dict]:
        return []

    @staticmethod
    def prefix_snapshot() -> dict:
        return {"enabled": False, "evidence": "not_observed_by_token_trace"}


def normalize_runtime_trace(experiment: Experiment, raw: dict, *,
                            allow_test_fixture: bool = False,
                            include_request_evidence: bool = False) -> dict:
    """Normalize server-ready events without rewriting the supplied experiment.

    Arrivals and completed output counts must match the replay workload exactly.
    Use ``RuntimeTraceRecorder.replay_experiment`` explicitly when constructing
    a conditional replay from actual arrival/output observations.
    """
    if type(include_request_evidence) is not bool:
        _fail("include_request_evidence must be boolean")
    validate_experiment(experiment)
    trace = _object(raw, {"schema_version", "kind", "source", "timing_boundary",
                          "observed_until_ms", "events"}, {"identity", "notes"}, "runtime trace")
    if type(trace["schema_version"]) is not int or trace["schema_version"] != 1 or trace["kind"] != "runtime_trace":
        _fail("Only runtime_trace schema version 1 is supported")
    if trace["timing_boundary"] != "server_token_ready_v1":
        _fail("Client delivery/content timestamps cannot become server-token observations",
              "INCOMPATIBLE_TIMING_BOUNDARY")
    source = _source(trace["source"], allow_test_fixture)
    identity = validate_identity(profile_identity(experiment), measured=source["kind"] == "measured")
    if "identity" in trace and validate_identity(trace["identity"], measured=source["kind"] == "measured") != identity:
        _fail("Runtime trace and experiment identities differ", "CALIBRATION_MISMATCH")
    end = _number(trace["observed_until_ms"], "observed_until_ms")
    cutoff = experiment.execution.max_simulated_ms
    evaluation = experiment.evaluation
    if evaluation.measurement_window_ms is not None:
        if not evaluation.drain:
            cutoff = min(cutoff, evaluation.measurement_window_ms)
        elif evaluation.max_drain_ms is not None:
            cutoff = min(cutoff, evaluation.measurement_window_ms + evaluation.max_drain_ms)
    if end > cutoff:
        _fail("Observation horizon exceeds the experiment time/window/drain contract", "COMPARISON_MISMATCH")
    if not isinstance(trace["events"], list):
        _fail("events must be an array")
    specs = {request.request_id: request for request in experiment.workload}
    requests = {name: RequestState(name, spec.arrival_ms, spec.input_tokens, spec.priority)
                for name, spec in specs.items()}
    window = experiment.evaluation.measurement_window_ms
    for name, request in requests.items():
        if window is not None and specs[name].arrival_ms >= window:
            request.status, request.error_code = "NOT_ADMITTED", "OUTSIDE_MEASUREMENT_WINDOW"
    previous = 0.0
    owners: dict[str, int] = {}
    for item in trace["events"]:
        event = _object(item, {"event", "request_id", "timestamp_ms"},
                        {"token_count", "output_token_index", "status", "error_code", "replica_index"}, "event")
        when = _number(event["timestamp_ms"], "event.timestamp_ms")
        if when < previous or when > end:
            _fail("Runtime event timestamps must be ordered and within observed_until_ms")
        previous = when
        name = _text(event["request_id"], "event.request_id")
        if name not in requests:
            _fail("Runtime event refers to an unknown request")
        request, spec = requests[name], specs[name]
        request_deadline = spec.arrival_ms + spec.timeout_ms if spec.timeout_ms is not None else None
        if "replica_index" in event:
            owner = _integer(event["replica_index"], "replica_index")
            if owner >= experiment.strategy.replicas or (name in owners and owners[name] != owner):
                _fail("A runtime request cannot change replica ownership")
            owners[name] = owner
        kind = event["event"]
        if kind == "REQUEST_ARRIVAL":
            if request.status != "CREATED":
                _fail("Runtime arrival is duplicated or outside the admitted cohort")
            if when != spec.arrival_ms:
                _fail("Observed arrivals differ from the replay workload; explicitly bind actual arrivals first",
                      "COMPARISON_MISMATCH")
            if set(event) - {"event", "request_id", "timestamp_ms", "replica_index"}:
                _fail("Arrival event contains unrelated token or terminal metadata")
            request.status = "WAITING"
        elif kind == "TOKEN_READY":
            if request.status in TERMINAL or request.status == "CREATED":
                _fail("Token event occurs before arrival or after request termination")
            if request_deadline is not None and when > request_deadline:
                _fail("Token event exceeds the declared request timeout", "COMPARISON_MISMATCH")
            if _integer(event.get("token_count"), "token_count", 1) != 1:
                _fail("Canonical server traces need one timestamp per token; coalesced delivery has another boundary")
            if _integer(event.get("output_token_index"), "output_token_index", 1) != request.generated + 1:
                _fail("Token indices must be contiguous and cannot repeat")
            if "status" in event or "error_code" in event:
                _fail("Token event contains terminal metadata")
            request.generated += 1
            if request.generated > spec.output_tokens:
                _fail("Observed output exceeds the declared actual replay length", "COMPARISON_MISMATCH")
            request.token_times.append(when)
            request.first_token_ms = request.first_token_ms if request.first_token_ms is not None else when
            request.status = "DECODING"
        elif kind == "REQUEST_FINISH":
            if request.status in TERMINAL or request.status == "CREATED":
                _fail("Finish event occurs before arrival or after termination")
            status = event.get("status")
            if status not in ("COMPLETED", "FAILED", "TIMED_OUT", "UNFINISHED"):
                _fail("Finish event has an unsupported status")
            if status == "COMPLETED" and request_deadline is not None and when > request_deadline:
                _fail("Completion exceeds the declared request timeout", "COMPARISON_MISMATCH")
            if status == "TIMED_OUT" and (request_deadline is None or when < request_deadline):
                _fail("Timeout outcome does not match a declared elapsed request deadline", "COMPARISON_MISMATCH")
            if "token_count" in event or "output_token_index" in event:
                _fail("Finish event contains token metadata")
            if status == "COMPLETED" and request.generated != spec.output_tokens:
                _fail("Completed output count differs from actual replay length", "COMPARISON_MISMATCH")
            error = event.get("error_code")
            if error is not None:
                _text(error, "event.error_code")
            if status == "COMPLETED" and error is not None:
                _fail("Completed requests cannot declare an error")
            request.status, request.error_code = status, error
            request.completion_ms = when if status != "UNFINISHED" else None
        else:
            _fail("Unknown runtime trace event")
    for request in requests.values():
        if request.status not in TERMINAL:
            request.status, request.error_code = "UNFINISHED", "OBSERVATION_ENDED"
    facade = SimpleNamespace(exp=experiment, requests=requests, now=end, stop_reason=None,
        events=[], handled_events=len(trace["events"]), memory=_EmptyMemory(), swap=None,
        performance=SimpleNamespace(profile=None, provenance=None), compute_busy_ms=0.0,
        recompute_execution_ms=0.0, limitations=["Request timing evidence; no compute attribution"])
    report = report_simulation(facade)
    observation = {"schema_version": 1, "kind": "measured_run",
        "comparison_contract": comparison_contract(experiment), "source": source,
        "metrics": extract_metrics(report.result),
        "requests": [{key: row[key] for key in ("request_id", "ttft_ms", "tpot_ms", "end_to_end_ms")}
                     for row in report.requests],
        "notes": {"collector_timing_boundary": trace["timing_boundary"],
                  "observed_until_ms": end, "server_token_events": True,
                  "request_status_counts": report.result["counts"]}}
    bundle = {"status": "COMPLETED", "kind": "runtime_trace_collection", "source": source,
        "experiment": experiment_dict(experiment), "observation": observation,
        "raw_records": deepcopy(trace["events"]), "hardware_validated": False,
        "timing_boundary": trace["timing_boundary"], "canonical_comparable": True,
        "limitations": ["Trace identities and runtime policy are declared evidence, not remote attestation.",
                        "No stage time or Workspace is inferred from request events."]}
    if include_request_evidence:
        # This view carries observed request timing, never the facade's zero
        # compute counters or empty-memory placeholders as physical observations.
        fields = ("status", "error_code", "counts", "admitted_count", "completion_rate_denominator",
                  "completion_rate", "evaluation_count", "cohort_completion_rate", "latency_ms",
                  "throughput", "goodput", "sla_satisfied", "simulated_ms", "events_processed")
        result = {key: deepcopy(report.result[key]) for key in fields}
        result.update(backend="observed_server_trace", hardware_validated=False, memory=[],
                      observed_elapsed_ms=end, physical_memory_status="not_observed")
        bundle["request_evidence"] = {"result": result, "requests": deepcopy(report.requests),
                                      "events": deepcopy(trace["events"]),
                                      "timing_boundary": trace["timing_boundary"],
                                      "physical_memory_status": "not_observed"}
    return bundle


class RuntimeTraceRecorder:
    """Thread-safe hooks for a Runtime that exposes actual server-token readiness."""

    def __init__(self, experiment: Experiment, *, source: dict,
                 clock_ns: Callable[[], int] | None = None):
        validate_experiment(experiment)
        self.experiment = experiment
        self.source = _source(source, True)
        if clock_ns is not None:
            self.source["kind"] = "test_fixture"
        self.clock_ns = clock_ns or time.perf_counter_ns
        self.origin_ns = self._clock()
        self.events: list[dict] = []
        self._states: dict[str, str] = {}
        self._counts: dict[str, int] = {}
        self._arrivals: dict[str, float] = {}
        self._specs = {row.request_id: row for row in experiment.workload}
        self._lock = threading.Lock()

    def _clock(self) -> int:
        value = self.clock_ns()
        return _integer(value, "monotonic clock ns")

    def _time(self, timestamp_ms: float | None) -> float:
        result = ((self._clock() - self.origin_ns) / 1e6 if timestamp_ms is None
                  else _number(timestamp_ms, "timestamp_ms"))
        if result < 0 or (self.events and result < self.events[-1]["timestamp_ms"]):
            _fail("Recorder time cannot move backwards")
        return result

    def arrival(self, request_id: str, *, timestamp_ms: float | None = None) -> None:
        with self._lock:
            if request_id not in self._specs or request_id in self._states:
                _fail("Arrival must identify a known request that has not arrived")
            when = self._time(timestamp_ms)
            self._states[request_id], self._counts[request_id] = "ACTIVE", 0
            self._arrivals[request_id] = when
            self.events.append({"event": "REQUEST_ARRIVAL", "request_id": request_id, "timestamp_ms": when})

    def token_ready(self, request_id: str, *, timestamp_ms: float | None = None) -> None:
        with self._lock:
            if self._states.get(request_id) != "ACTIVE":
                _fail("Token hook requires a live arrived request")
            when = self._time(timestamp_ms)
            self._counts[request_id] += 1
            self.events.append({"event": "TOKEN_READY", "request_id": request_id, "timestamp_ms": when,
                                "token_count": 1, "output_token_index": self._counts[request_id]})

    def finish(self, request_id: str, status: str = "COMPLETED", *, timestamp_ms: float | None = None,
               error_code: str | None = None) -> None:
        with self._lock:
            if self._states.get(request_id) != "ACTIVE" or status not in ("COMPLETED", "FAILED", "TIMED_OUT", "UNFINISHED"):
                _fail("Finish hook requires a live request and supported terminal status")
            if status == "COMPLETED" and (not self._counts[request_id] or error_code is not None):
                _fail("Completed runtime work requires output and no error")
            when = self._time(timestamp_ms)
            event = {"event": "REQUEST_FINISH", "request_id": request_id, "timestamp_ms": when, "status": status}
            if error_code is not None:
                event["error_code"] = _text(error_code, "error_code")
            self.events.append(event)
            self._states[request_id] = status

    def snapshot(self, *, observed_until_ms: float | None = None) -> dict:
        with self._lock:
            end = self._time(observed_until_ms)
            return {"schema_version": 1, "kind": "runtime_trace", "source": dict(self.source),
                "timing_boundary": "server_token_ready_v1", "observed_until_ms": end,
                "identity": profile_identity(self.experiment), "events": deepcopy(self.events),
                "notes": {"arrivals_are_recorded": True, "conditional_replay_requires_explicit_binding": True}}

    def replay_experiment(self) -> Experiment:
        """Explicitly select actual arrivals/output lengths for conditional replay."""
        with self._lock:
            rows = tuple(replace(spec, arrival_ms=self._arrivals.get(spec.request_id, spec.arrival_ms),
                output_tokens=(self._counts[spec.request_id] if self._states.get(spec.request_id) == "COMPLETED"
                               else spec.output_tokens)) for spec in self.experiment.workload)
            return replace(self.experiment, workload=rows)


def _sse_payloads(chunks: Iterable[bytes], maximum: int) -> Iterable[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer, total = "", 0
    data: list[str] = []

    def lines(final: bool = False):
        nonlocal buffer
        while buffer:
            positions = [position for delimiter in ("\n", "\r") if (position := buffer.find(delimiter)) >= 0]
            if not positions:
                if final:
                    yield buffer
                    buffer = ""
                return
            end = min(positions)
            if buffer[end] == "\r" and end == len(buffer) - 1 and not final:
                return
            width = 2 if buffer[end:end + 2] == "\r\n" else 1
            line, buffer = buffer[:end], buffer[end + width:]
            yield line

    try:
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                _fail("SSE transport must yield bytes")
            total += len(chunk)
            if total > maximum:
                _fail("Runtime response exceeds max_response_bytes")
            buffer += decoder.decode(chunk)
            for line in lines():
                if not line:
                    if data:
                        yield "\n".join(data)
                        data = []
                elif line.startswith("data:"):
                    data.append(line[5:].removeprefix(" "))
        buffer += decoder.decode(b"", final=True)
        for line in lines(final=True):
            if not line and data:
                yield "\n".join(data)
                data = []
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        if data:
            _fail("SSE response ended before an event separator")
    except UnicodeError as exc:
        raise SimulationError("INVALID_RUNTIME_TRACE", "Invalid UTF-8 in SSE response") from exc


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise HTTPError(request.full_url, code, "Redirects are disabled for runtime collection", headers, fp)


class _CaptureTimeout(TimeoutError):
    code = "CAPTURE_TIME_LIMIT"


def _http_transport(url: str, body: dict | None, headers: dict, timeout_s: float) -> Iterable[bytes]:
    request = Request(url, data=json.dumps(body).encode() if body is not None else None,
                      headers=headers, method="POST" if body is not None else "GET")
    deadline = time.monotonic() + timeout_s

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("HTTP collection exceeded its total timeout")
        return value

    # urllib's default redirect handler copies Authorization to redirected
    # requests. Refuse redirects so a server cannot forward credentials.
    with build_opener(_NoRedirects()).open(request, timeout=remaining()) as response:
        while True:
            timeout = remaining()
            # CPython HTTPResponse owns the socket through this file wrapper,
            # including Connection: close responses. Tighten each blocking read
            # to the remaining total deadline, rather than renewing the timeout.
            sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            if sock is not None:
                sock.settimeout(timeout)
            read = getattr(response, "read1", None)
            chunk = read(65536) if callable(read) else response.read(1)
            remaining()
            if not chunk:
                break
            yield chunk


def _client_metrics(rows: list[dict], experiment: Experiment, end: float) -> dict:
    cohort = [row for row in rows if row["status"] != "NOT_ADMITTED"]
    completed = [row for row in cohort if row["status"] == "COMPLETED"]
    evaluation = experiment.evaluation

    def eligible(row):
        return all(limit is None or row[name] is not None and row[name] <= limit
                   for name, limit in (("ttft_ms", evaluation.ttft_sla_ms), ("tpot_ms", evaluation.tpot_sla_ms)))

    good = [row for row in completed if eligible(row)]
    window = evaluation.measurement_window_ms if evaluation.measurement_window_ms is not None else end
    def rate(count):
        return count * 1000 / window if window > 0 else None
    values = {f"{kind}_p{percentile}_ms": _percentiles([row[f"{kind}_ms"] for row in completed
              if row[f"{kind}_ms"] is not None])[f"p{percentile}"]
              for kind in ("ttft", "tpot", "end_to_end") for percentile in (50, 95, 99)}
    values.update({f"itl_p{percentile}_ms": None for percentile in (50, 95, 99)})
    values.update(throughput_requests_per_s=rate(len(completed)),
                  throughput_output_tokens_per_s=rate(sum(row["output_tokens_actual"] for row in cohort)),
                  goodput_requests_per_s=rate(len(good)),
                  goodput_output_tokens_per_s=rate(sum(row["output_tokens_actual"] for row in good)),
                  completion_rate=len(completed) / len(cohort) if cohort else None,
                  request_count=len(cohort),
                  sla_satisfied=(all(eligible(row) for row in completed) and len(completed) == len(cohort)
                      if cohort and (evaluation.ttft_sla_ms is not None or evaluation.tpot_sla_ms is not None) else None))
    return values


def collect_vllm(experiment: Experiment, plan: dict, inventory: dict | None = None, *,
                 base_dir: Path | None = None, allow_test_fixture: bool = False,
                 transport: Callable[..., Iterable[bytes]] | None = None,
                 clock_ns: Callable[[], int] | None = None) -> dict:
    """Capture vLLM delta-token SSE delivery, retaining its client boundary.

    A supplied transport or clock is always a test fixture. This capture is not
    a server token trace and cannot automatically become a Compare observation.
    ``base_dir`` is accepted for CLI plan consistency; this plan has no files.
    """
    del base_dir
    validate_experiment(experiment)
    code = "INVALID_COLLECTION_PLAN"
    if type(allow_test_fixture) is not bool or inventory is not None and not isinstance(inventory, dict):
        _fail("allow_test_fixture must be boolean and inventory must be an object", code)
    config = _object(plan, {"schema_version", "endpoint", "model", "source_reference"},
        {"timeout_s", "max_workers", "api_key_env", "max_response_bytes", "runtime_version"}, "vLLM plan", code)
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        _fail("Only collection plan schema version 1 is supported", code)
    endpoint = _text(config["endpoint"], "endpoint", code).rstrip("/")
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment or parsed.path
            or any(ord(character) < 32 for character in endpoint)):
        _fail("endpoint must be an HTTP(S) origin without credentials, query or path", code)
    try:
        parsed.port
    except ValueError:
        _fail("endpoint contains an invalid port", code)
    model = _text(config["model"], "model", code)
    reference = _text(config["source_reference"], "source_reference", code)
    timeout = _number(config.get("timeout_s", 60), "timeout_s", positive=True, code=code)
    workers = _integer(config.get("max_workers", 8), "max_workers", 1, code)
    maximum = _integer(config.get("max_response_bytes", 8 * 1024 * 1024), "max_response_bytes", 1, code)
    if timeout > 600 or workers > 256 or maximum > 64 * 1024 * 1024 or len(experiment.workload) > 10000:
        _fail("Collection exceeds bounded worker, response or request limits", code)
    if experiment.execution.warmup or not experiment.evaluation.drain or experiment.evaluation.max_drain_ms is not None:
        _fail("vLLM capture needs no embedded warmup and drain=true without max_drain_ms", code)
    if any(not request.prompt_token_ids for request in experiment.workload):
        _fail("vLLM token-ID capture requires explicit prompt_token_ids for every request", code)
    if any(request.arrival_ms > experiment.execution.max_simulated_ms for request in experiment.workload
           if experiment.evaluation.measurement_window_ms is None
           or request.arrival_ms < experiment.evaluation.measurement_window_ms):
        _fail("A planned arrival exceeds the capture execution horizon", code)
    key_name = _text(config.get("api_key_env", "OPENAI_API_KEY"), "api_key_env", code)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_name):
        _fail("api_key_env must name one environment variable", code)
    key = os.environ.get(key_name, "")
    injected = transport is not None or clock_ns is not None
    fixture = injected or inventory is not None and inventory.get("source", {}).get("kind") == "test_fixture"
    if fixture and not allow_test_fixture:
        _fail("Injected collection needs allow_test_fixture", "TEST_FIXTURE_NOT_ALLOWED")
    if not fixture:
        from .hardware import probe_rocm
        inventory = inventory if inventory is not None else probe_rocm()
        if inventory.get("status") != "READY":
            _fail("A ready ROCm inventory is required for live vLLM collection", "HARDWARE_UNAVAILABLE")
    source = {"kind": "test_fixture" if fixture else "measured", "reference": reference}
    if inventory is not None:
        inventory = deepcopy(inventory)
        if fixture:
            inventory["source"] = {"kind": "test_fixture", "reference": "Injected CPU collection dependencies"}
    send = transport or _http_transport
    clock = clock_ns or time.perf_counter_ns
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def redact(value: Any) -> str:
        result = str(value)
        return result.replace(key, "[REDACTED]") if key else result

    version_bytes = bytearray()
    try:
        for chunk in send(endpoint + "/version", None, headers, timeout):
            version_bytes.extend(chunk)
            if len(version_bytes) > 65536:
                _fail("Runtime version response exceeds 64KiB")
        version = _text(_json_payload(version_bytes.decode())["version"], "runtime version")
    except Exception as exc:
        raise SimulationError("COLLECTION_FAILED", f"Could not read vLLM version: {redact(exc)}") from None
    if key and key in version:
        _fail("Runtime metadata contains credential material and cannot be retained", "COLLECTION_FAILED")
    if "runtime_version" in config and _text(config["runtime_version"], "runtime_version", code) != version:
        _fail("Observed vLLM version differs from the plan", "CALIBRATION_MISMATCH")
    if fixture:
        effective = replace(experiment, calibration=replace(experiment.calibration,
                            runtime_name="vllm", runtime_version=version))
    else:
        from .hardware import bind_inventory
        assert inventory is not None
        effective = bind_inventory(experiment, inventory, runtime_name="vllm", runtime_version=version,
            require_rccl=experiment.strategy.tensor_parallel_size > 1 or experiment.strategy.pipeline_parallel_size > 1,
            allow_test_fixture=allow_test_fixture)
    origin = _integer(clock(), "monotonic clock ns")
    horizon = effective.execution.max_simulated_ms
    clock_lock = threading.Lock()
    last_clock = origin

    def now() -> float:
        nonlocal last_clock
        with clock_lock:
            current = _integer(clock(), "monotonic clock ns")
            if current < last_clock:
                _fail("Collection clock cannot move backwards")
            last_clock = current
            return _number((current - origin) / 1e6, "capture time")

    def work(spec) -> dict:
        row: dict[str, Any] = {"request_id": spec.request_id, "status": "FAILED", "arrival_ms": spec.arrival_ms,
               "send_time_ms": None, "send_lag_ms": None, "first_token_ms": None, "last_token_ms": None,
               "completion_ms": None, "output_tokens_actual": 0, "ttft_ms": None, "tpot_ms": None,
               "end_to_end_ms": None, "token_deliveries": [], "usage": None, "coalesced": False}
        try:
            row["send_time_ms"] = now()
            row["send_lag_ms"] = row["send_time_ms"] - spec.arrival_ms
            deadline = min(row["send_time_ms"] + timeout * 1000, horizon)
            if spec.timeout_ms is not None:
                deadline = min(deadline, spec.arrival_ms + spec.timeout_ms)
            request_timeout = (deadline - row["send_time_ms"]) / 1000
            if request_timeout <= 0:
                error = _CaptureTimeout if row["send_time_ms"] >= horizon else TimeoutError
                raise error("Request or capture deadline elapsed before send")
            body = {"model": model, "prompt": list(spec.prompt_token_ids), "max_tokens": spec.output_tokens,
                    "temperature": 0, "n": 1, "echo": False, "stream": True,
                    "return_token_ids": True, "stream_options": {"include_usage": True}}
            def guarded_chunks():
                stream = iter(send(endpoint + "/v1/completions", body, headers, request_timeout))
                try:
                    for chunk in stream:
                        current = now()
                        if current > deadline:
                            error = _CaptureTimeout if current >= horizon else TimeoutError
                            raise error("Streaming request exceeded its elapsed or capture timeout")
                        yield chunk
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()

            done, finish = False, False
            for payload in _sse_payloads(guarded_chunks(), maximum):
                when = now()
                if when > deadline:
                    raise TimeoutError("Streaming request exceeded its elapsed timeout")
                if done:
                    _fail("SSE data appeared after [DONE]")
                if payload == "[DONE]":
                    done = True
                    continue
                packet = _json_payload(payload)
                if "error" in packet:
                    _fail("Server returned an error event")
                if (packet.get("prompt_token_ids") is not None
                        and packet["prompt_token_ids"] != list(spec.prompt_token_ids)):
                    _fail("Server prompt token IDs differ from the requested prompt")
                choices = packet.get("choices")
                if not isinstance(choices, list) or len(choices) > 1:
                    _fail("Only one completion choice is supported")
                for choice in choices:
                    if (not isinstance(choice, dict) or type(choice.get("index")) is not int
                            or choice["index"] != 0 or "token_ids" not in choice):
                        _fail("vLLM choice must include index=0 and explicit delta token_ids")
                    token_ids = choice["token_ids"]
                    if not isinstance(token_ids, list) or any(type(token) is not int or not 0 <= token < effective.strategy.model.vocab_size
                                                              for token in token_ids):
                        _fail("Delta token_ids must be valid vocabulary IDs")
                    if finish and token_ids:
                        _fail("Tokens appeared after a completion finish reason")
                    if row["usage"] is not None and token_ids:
                        _fail("Tokens appeared after final usage")
                    if "prompt_token_ids" in choice and choice["prompt_token_ids"] is not None and choice["prompt_token_ids"] != list(spec.prompt_token_ids):
                        _fail("Server prompt token IDs differ from the requested prompt")
                    if token_ids:
                        row["first_token_ms"] = when if row["first_token_ms"] is None else row["first_token_ms"]
                        row["last_token_ms"] = when
                        row["output_tokens_actual"] += len(token_ids)
                        row["coalesced"] |= len(token_ids) > 1
                        row["token_deliveries"].append({"timestamp_ms": when, "token_ids": list(token_ids),
                            "payload_sha256": hashlib.sha256(payload.encode()).hexdigest()})
                        if row["output_tokens_actual"] > spec.output_tokens:
                            _fail("Server returned more tokens than max_tokens")
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        if finish or reason not in ("stop", "length"):
                            _fail("Unsupported completion finish_reason")
                        finish = True
                        row["finish_reason"] = reason
                if packet.get("usage") is not None:
                    if row["usage"] is not None:
                        _fail("Duplicate usage record")
                    if not finish:
                        _fail("Final usage appeared before the completion finish reason")
                    usage = packet["usage"]
                    if not isinstance(usage, dict):
                        _fail("Final usage must be an object")
                    prompt_count = _integer(usage.get("prompt_tokens"), "usage.prompt_tokens")
                    output_count = _integer(usage.get("completion_tokens"), "usage.completion_tokens")
                    if prompt_count != spec.input_tokens or output_count != row["output_tokens_actual"]:
                        _fail("Usage and actual delta token counts differ")
                    if "total_tokens" in usage and _integer(usage["total_tokens"], "usage.total_tokens") != prompt_count + output_count:
                        _fail("usage.total_tokens is inconsistent")
                    row["usage"] = {"prompt_tokens": prompt_count, "completion_tokens": output_count}
            if not done or not finish or row["usage"] is None or row["output_tokens_actual"] == 0:
                _fail("SSE stream lacks [DONE], finish, usage or generated token evidence")
            if row["usage"]["completion_tokens"] != row["output_tokens_actual"]:
                _fail("Final usage and accumulated token counts differ")
            if row["finish_reason"] == "length" and row["output_tokens_actual"] != spec.output_tokens:
                _fail("length finish_reason does not match max_tokens")
            row["completion_ms"] = now()
            row["ttft_ms"] = row["first_token_ms"] - spec.arrival_ms
            row["ttft_from_send_ms"] = row["first_token_ms"] - row["send_time_ms"]
            row["tpot_ms"] = ((row["last_token_ms"] - row["first_token_ms"]) / (row["output_tokens_actual"] - 1)
                              if row["output_tokens_actual"] > 1 else None)
            row["end_to_end_ms"] = row["completion_ms"] - spec.arrival_ms
            row["status"] = "COMPLETED"
        except Exception as exc:
            timed_out = isinstance(exc, TimeoutError) or isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError)
            row["status"] = "TIMED_OUT" if timed_out else "FAILED"
            row["error_code"] = getattr(exc, "code", "REQUEST_TIMEOUT" if timed_out else "RUNTIME_REQUEST_FAILED")
            row["error"] = redact(exc)
            try:
                row["completion_ms"] = now()
            except SimulationError:
                row["completion_ms"] = (last_clock - origin) / 1e6
        return row

    rows: list[dict] = []
    futures = []
    window = effective.evaluation.measurement_window_ms
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for spec in sorted(effective.workload, key=lambda request: (request.arrival_ms, request.request_id)):
            if window is not None and spec.arrival_ms >= window:
                rows.append({"request_id": spec.request_id, "status": "NOT_ADMITTED", "arrival_ms": spec.arrival_ms,
                             "output_tokens_actual": 0, "error_code": "OUTSIDE_MEASUREMENT_WINDOW"})
                continue
            while (remaining_ms := spec.arrival_ms - now()) > 0:
                time.sleep(min(.05, remaining_ms / 1000))
            current = now()
            if current >= horizon:
                rows.append({"request_id": spec.request_id, "status": "TIMED_OUT", "arrival_ms": spec.arrival_ms,
                    "output_tokens_actual": 0, "completion_ms": current, "error_code": "CAPTURE_TIME_LIMIT",
                    "ttft_ms": None, "tpot_ms": None, "end_to_end_ms": None,
                    "error": "Capture horizon elapsed before dispatch"})
            else:
                futures.append(pool.submit(work, spec))
        rows.extend(future.result() for future in futures)
    by_id = {row["request_id"]: row for row in rows}
    rows = [by_id[spec.request_id] for spec in effective.workload]
    end = max((row.get("completion_ms") or 0 for row in rows), default=0.0)
    bundle = {"status": "COMPLETED", "kind": "vllm_client_collection", "source": source,
        "timing_boundary": "client_token_delivery_v1", "experiment": experiment_dict(effective),
        "inventory": inventory, "runtime_version": version, "raw_records": rows,
        "inventory_binding": "operator_declared_server_inventory_not_remote_attestation",
        "model_binding": {"served_model": model, "declared_model": effective.strategy.model.name,
                          "declared_revision": effective.strategy.model.revision,
                          "revision_independently_verified": False},
        "server_state_independently_verified": False,
        "workload_output_tokens_semantics": "request_max_tokens; actual counts come from delta IDs and final usage",
        "client_requests": rows, "client_metrics": _client_metrics(rows, effective, end),
        "client_metric_window": {"window_ms": window if window is not None else end,
            "rule": "arrival_cohort_with_drain" if window is not None else "capture_zero_to_last_completion",
            "arrival_origin": "planned_capture_arrivals; send lag is separately retained"},
        "observation": None, "canonical_comparable": False, "hardware_validated": False,
        "coalesced_delivery": any(row.get("coalesced", False) for row in rows),
        "observed_until_ms": end,
        "limitations": ["Client token-delivery timestamps are not server token-ready or whole-stage service times.",
            "Delta token IDs determine counts; multiple tokens in one message have unknown internal generation times.",
            "No per-token ITL or Workspace is inferred. Response text is omitted; payload hashes and token IDs are retained.",
            "Model revision and hardware placement are declared inventory bindings, not remotely attested by HTTP.",
            "Replay uses an open-loop arrival plan; worker saturation is retained as client send lag."]}
    if any(row["status"] in ("FAILED", "TIMED_OUT") for row in rows):
        from .hardware import CollectionError
        bundle["status"] = "FAILED"
        raise CollectionError("COLLECTION_FAILED", "One or more streaming requests failed; inspect preserved records", bundle)
    return bundle
