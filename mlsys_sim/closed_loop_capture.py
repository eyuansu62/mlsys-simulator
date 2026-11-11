"""Finite-cohort clients driven by synchronous runtime API terminal feedback.

These timestamps are API observations, never server token-ready timestamps.
The backend owns synchronization: abort_requests must acknowledge cancellation
at a safe single-GPU step boundary before returning.
"""

from copy import deepcopy
from fractions import Fraction
import heapq
import math
import time
from typing import Any, Callable

from .load_validation import (PROTOCOL, closed_loop_policy_sha256, generate_request_stream,
                              validate_closed_loop_spec)
from .schema import SimulationError
from .workload import canonical_sha256


CAPTURE_PROTOCOL = "runtime_api_feedback_capture_v1"
TIMING_BOUNDARY = "runtime_API_feedback_v1"
_MAX_STAGNANT_CLOCK_READS = 4096
_MAX_STEPS = 1_000_000
_MAX_FEEDBACK_RECORDS = 4_000_000
_TERMINAL = {"COMPLETED", "FAILED", "TIMED_OUT", "UNFINISHED"}


class _Fault(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code, self.details = code, details


def _ns(milliseconds: float) -> int:
    # Integer nanoseconds avoid overflow and non-advancing float deadlines.
    return math.ceil(Fraction(milliseconds) * 1_000_000)


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)[:1024]}


class _Collector:
    def __init__(self, backend: Any, spec: dict, concurrency: int, replication: int,
                 clock_ns: Callable[[], int] | None, progress: Callable[[dict], None] | None):
        self.backend = backend
        self.spec, self.concurrency, self.replication = spec, concurrency, replication
        self.stream = generate_request_stream(spec, replication=replication)
        self.policy_sha = closed_loop_policy_sha256(spec, concurrency, replication)
        self.injected_clock = clock_ns is not None
        self.clock = time.perf_counter_ns if clock_ns is None else clock_ns
        self.progress = progress
        self.now: int | None = None
        self.start: int | None = None
        self.end: int | None = None
        self.deadline: int | None = None
        self.stagnant = 0
        self.requests: list[dict] = []
        self.active: dict[str, dict] = {}
        self.native_ids: set[str] = set()
        self.pending: list[tuple[int, int, int, str | None]] = []
        self.sequence = 0
        self.client_outstanding = [0] * concurrency
        self.client_peak = [0] * concurrency
        self.logical_outstanding = 0
        self.terminal_counts = {status: 0 for status in _TERMINAL}
        self.peak_outstanding = 0
        self.peak_native = 0
        self.errors: list[dict] = []
        self.steps: list[dict] = []
        self.step_count = 0
        self.feedback_count = 0
        self.progress_bucket = 0
        self.resources_drained = False

    def read_clock(self, operation: str) -> int:
        try:
            observed = self.clock()
        except Exception as exc:
            raise _Fault("CLOCK_ERROR", str(exc), operation=operation, exception_type=type(exc).__name__) from exc
        if type(observed) is not int or observed < 0:
            raise _Fault("INVALID_CLOCK", "Clock must return nonnegative integer nanoseconds",
                         operation=operation, observed=_safe(observed))
        if self.now is not None and observed < self.now:
            raise _Fault("CLOCK_ROLLBACK", "Monotonic clock moved backwards", operation=operation,
                         previous_ns=self.now, observed_ns=observed)
        self.stagnant = self.stagnant + 1 if observed == self.now else 0
        if self.stagnant >= _MAX_STAGNANT_CLOCK_READS:
            raise _Fault("CLOCK_NOT_PROGRESSING", "Clock did not advance within its bounded read budget",
                         operation=operation, observed_ns=observed)
        self.now = observed
        return observed

    def error(self, code: str, message: str, row: dict | None = None, **details: Any) -> None:
        item = {"code": code, "message": str(message)[:4096], "observed_ns": self.now, **_safe(details)}
        if row is not None:
            item.update(request_id=row["request_id"], native_request_id=row["native_request_id"],
                        stream_index=row["stream_index"], client_id=row["client_id"])
            row["error_code"] = row["error_code"] or code
        self.errors.append(item)

    def counts(self) -> dict:
        return {"offered_count": len(self.requests),
                "not_offered_count": self.spec["requests_per_run"] - len(self.requests),
                "completed_count": self.terminal_counts["COMPLETED"],
                "failed_count": self.terminal_counts["FAILED"],
                "timed_out_count": self.terminal_counts["TIMED_OUT"],
                "unfinished_count": len(self.requests) - sum(self.terminal_counts[status]
                    for status in ("COMPLETED", "FAILED", "TIMED_OUT")),
                "terminal_count": sum(self.terminal_counts.values())}

    def notify(self, stage: str) -> None:
        if self.progress is None:
            return
        try:
            self.progress({"stage": stage, "concurrency": self.concurrency, "replication": self.replication,
                           "current_ns": self.now, "step_count": self.step_count, **self.counts()})
        except Exception as exc:
            raise _Fault("PROGRESS_CALLBACK_ERROR", str(exc), stage=stage,
                         exception_type=type(exc).__name__) from exc

    def native_pending(self) -> bool:
        try:
            pending = self.backend.has_unfinished_requests()
        except Exception as exc:
            raise _Fault("BACKEND_STATE_ERROR", str(exc), exception_type=type(exc).__name__) from exc
        if type(pending) is not bool:
            raise _Fault("INVALID_BACKEND_STATE", "has_unfinished_requests must return a boolean")
        return pending

    def finish(self, row: dict, status: str, when: int | None, source: str, *, refill: bool = True,
               release_owner: bool = True) -> None:
        if row["status"] in _TERMINAL:
            return
        row.update(status=status, terminal_api_feedback_ns=when, terminal_source=source,
                   output_target_satisfied=row["output_tokens_actual"] == row["output_tokens_target"])
        self.terminal_counts[status] += 1
        native = row["native_request_id"]
        client = row["client_id"]
        if release_owner:
            if native in self.active and self.active[native] is row:
                del self.active[native]
            self.logical_outstanding -= self.client_outstanding[client]
            self.client_outstanding[client] = 0
        if refill and when is not None and len(self.requests) < self.spec["requests_per_run"]:
            ready = when + _ns(self.spec["think_ms"])
            if self.spec["think_ms"] > 0 and ready <= when:
                raise _Fault("NON_ADVANCING_THINK", "Positive think delay did not advance the clock")
            heapq.heappush(self.pending, (ready, self.sequence, client, row["request_id"]))
            self.sequence += 1

    def abort(self, rows: list[dict], status: str, code: str, *, refill: bool) -> None:
        selected = sorted((row for row in rows if row["status"] not in _TERMINAL), key=lambda row: row["stream_index"])
        if not selected:
            return
        before = self.read_clock("abort_before")
        for row in selected:
            row["abort_before_ns"] = before
        native = [row["native_request_id"] for row in selected if row["native_request_id"] is not None]
        try:
            self.backend.abort_requests(native)
        except Exception as exc:
            raise _Fault("BACKEND_ABORT_ERROR", str(exc), native_request_ids=native,
                         exception_type=type(exc).__name__) from exc
        for row in selected:
            row["abort_acknowledged"] = True
        ack = self.read_clock("abort_ack")
        for row in selected:
            row["abort_ack_ns"] = ack
            self.error(code, "Cancellation acknowledged at a synchronous API step boundary", row)
            self.finish(row, status, ack, "abort_ack", refill=refill)

    def offer(self, client: int, previous: str | None) -> None:
        if self.client_outstanding[client]:
            raise _Fault("CLIENT_OUTSTANDING_VIOLATION", "A client cannot enqueue a second outstanding request")
        before = self.read_clock("submit_before")
        assert self.deadline is not None
        if before >= self.deadline:
            raise _Fault("COHORT_TIME_LIMIT", "Formal run deadline reached before another request was offered")
        index = len(self.requests)
        spec = self.stream["requests"][index]
        row = {**spec, "router_request_id": spec["request_id"], "stream_index": index,
               "client_id": client, "preceding_request_id": previous,
               "native_request_id": None, "submit_before_ns": before, "submit_after_ns": None,
               "request_deadline_ns": before + _ns(spec["timeout_ms"]),
               "first_token_api_feedback_ns": None, "terminal_api_feedback_ns": None,
               "terminal_source": None, "status": "SUBMITTING", "error_code": None,
               "output_tokens_target": spec["output_tokens"], "output_tokens_actual": 0,
               "output_target_satisfied": False, "output_ledger_valid": True,
               "step_feedback_count": 0, "feedback": [], "abort_before_ns": None,
               "abort_ack_ns": None, "abort_acknowledged": False}
        self.requests.append(row)
        self.client_outstanding[client] = 1
        self.logical_outstanding += 1
        self.client_peak[client] = max(self.client_peak[client], 1)
        self.peak_outstanding = max(self.peak_outstanding, self.logical_outstanding)
        argument = {**deepcopy(spec), "stream_index": index, "client_id": client,
                    "preceding_request_id": previous}
        try:
            native = self.backend.add_request(argument)
        except Exception as exc:
            self.error("BACKEND_ADD_ERROR", str(exc), row, exception_type=type(exc).__name__)
            self.finish(row, "FAILED", None, "add_exception", refill=False)
            raise _Fault("BACKEND_ADD_ERROR", "Request submission raised; partial submission state is unknown") from exc
        # Keep the returned identity before reading the post-submit clock so
        # cleanup can still cancel a successful enqueue if that clock fails.
        if isinstance(native, str):
            row["native_request_id"] = native
        if not isinstance(native, str) or not native or len(native) > 4096:
            raise _Fault("INVALID_NATIVE_ID", "add_request must return a bounded nonempty native ID",
                         returned_native_id=_safe(native))
        if native in self.native_ids:
            raise _Fault("DUPLICATE_NATIVE_ID", "Native IDs must remain unique throughout the run", native_request_id=native)
        self.native_ids.add(native)
        self.active[native] = row
        row["status"] = "ACTIVE"
        self.peak_native = max(self.peak_native, len(self.active))
        row["submit_after_ns"] = self.read_clock("submit_after")

    def feedback(self, records: Any, before: int, when: int | None) -> None:
        step = {"step_index": self.step_count - 1, "step_before_ns": before,
                "feedback_ns": when, "records": _safe(records)}
        self.steps.append(step)
        if not isinstance(records, list):
            raise _Fault("INVALID_STEP_FEEDBACK", "step must return a list of cumulative request feedback")
        self.feedback_count += len(records)
        if self.feedback_count > _MAX_FEEDBACK_RECORDS:
            raise _Fault("CAPTURE_FEEDBACK_LIMIT", "Runtime feedback exceeded the bounded capture record budget")
        valid = []
        seen: set[str] = set()
        faults = []
        for record in records:
            if (not isinstance(record, dict)
                    or not {"native_request_id", "finished", "output_tokens_actual"} <= record.keys()
                    or not isinstance(record["native_request_id"], str)
                    or type(record["finished"]) is not bool or type(record["output_tokens_actual"]) is not int
                    or not 0 <= record["output_tokens_actual"] <= 2**63 - 1):
                faults.append(("INVALID_STEP_FEEDBACK", "Malformed cumulative request feedback"))
                continue
            native = record["native_request_id"]
            if native in seen:
                faults.append(("DUPLICATE_STEP_FEEDBACK", "A step returned the same native ID more than once"))
                continue
            seen.add(native)
            row = self.active.get(native)
            if row is None:
                faults.append(("UNKNOWN_NATIVE_FEEDBACK", f"Feedback references untracked native ID {native[:256]}"))
                continue
            if record["output_tokens_actual"] < row["output_tokens_actual"]:
                row["output_ledger_valid"] = False
                faults.append(("TOKEN_COUNT_ROLLBACK", "Cumulative output token count moved backwards"))
                continue
            valid.append((row, record))
        overshot = []
        for row, record in sorted(valid, key=lambda pair: pair[0]["stream_index"]):
            count = record["output_tokens_actual"]
            delta = count - row["output_tokens_actual"]
            row["feedback"].append({"step_index": self.step_count - 1, "feedback_ns": when,
                                    "finished": record["finished"], "output_tokens_actual": count,
                                    "delta_output_tokens": delta})
            row["step_feedback_count"] += 1
            row["output_tokens_actual"] = count
            row["output_target_satisfied"] = count == row["output_tokens_target"]
            if count > 0 and row["first_token_api_feedback_ns"] is None:
                row["first_token_api_feedback_ns"] = when
            if record["finished"]:
                if when is not None and when > row["request_deadline_ns"]:
                    self.error("REQUEST_TIMEOUT", "Terminal API feedback arrived after the request deadline", row)
                    status = "TIMED_OUT"
                elif not row["output_target_satisfied"]:
                    self.error("FIXED_OUTPUT_MISMATCH", "Native request ended with a different output token count", row)
                    status = "FAILED"
                else:
                    status = "COMPLETED"
                self.finish(row, status, when, "native_finished_feedback")
            elif count > row["output_tokens_target"]:
                overshot.append(row)
        if faults:
            for code, message in faults[1:]:
                self.error(code, message, operation="step_feedback")
            raise _Fault(*faults[0], operation="step_feedback")
        if overshot and when is not None:
            self.abort(overshot, "FAILED", "FIXED_OUTPUT_MISMATCH", refill=True)

    def expire(self, when: int) -> None:
        expired = [row for row in self.active.values() if row["request_deadline_ns"] <= when]
        self.abort(expired, "TIMED_OUT", "REQUEST_TIMEOUT", refill=True)

    def cleanup(self, code: str) -> None:
        """Try cancellation once; retain missing timestamps and unacknowledged ownership."""
        for row in list(self.active.values()):
            if row["abort_acknowledged"]:
                self.error(code, "Cancellation returned but its feedback clock could not be recorded", row)
                self.finish(row, "UNFINISHED", row["abort_ack_ns"], "abort_ack", refill=False)
        rows = sorted(self.active.values(), key=lambda row: row["stream_index"])
        if rows:
            before = self.now
            try:
                before = self.read_clock("cleanup_abort_before")
            except _Fault as fault:
                self.error(fault.code, str(fault), **fault.details)
            for row in rows:
                row["abort_before_ns"] = before
            try:
                self.backend.abort_requests([row["native_request_id"] for row in rows])
            except Exception as exc:
                self.error("BACKEND_ABORT_ERROR", str(exc), exception_type=type(exc).__name__)
            else:
                ack = None
                try:
                    ack = self.read_clock("cleanup_abort_ack")
                except _Fault as fault:
                    self.error(fault.code, str(fault), **fault.details)
                for row in rows:
                    row.update(abort_acknowledged=True, abort_ack_ns=ack)
                    self.error(code, "Run stopped; request cancelled at the next acknowledged API boundary", row)
                    self.finish(row, "UNFINISHED", ack, "abort_ack", refill=False)
        for row in self.requests:
            if row["status"] not in _TERMINAL:
                self.error(code, "Request has no confirmed terminal API feedback", row)
                self.finish(row, "UNFINISHED", None, "unconfirmed", refill=False, release_owner=False)
        try:
            self.resources_drained = not self.native_pending()
        except _Fault as fault:
            self.error(fault.code, str(fault), **fault.details)
            self.resources_drained = False

    def run(self) -> dict:
        try:
            self.start = self.read_clock("formal_start")
            self.deadline = self.start + _ns(self.spec["max_run_ms"])
            if self.native_pending():
                raise _Fault("BACKEND_NOT_EMPTY", "Formal collection needs a drained dedicated backend")
            for client in range(min(self.concurrency, self.spec["requests_per_run"])):
                heapq.heappush(self.pending, (self.start, self.sequence, client, None))
                self.sequence += 1
            self.notify("formal_start")
            while True:
                now = self.read_clock("loop_boundary")
                if len(self.requests) == self.spec["requests_per_run"] and not self.active:
                    if self.native_pending():
                        raise _Fault("UNTRACKED_NATIVE_WORK", "Backend still owns work after all tracked terminal feedback")
                    self.resources_drained = True
                    self.end = self.read_clock("formal_end")
                    if self.end > self.deadline:
                        raise _Fault("COHORT_TIME_LIMIT", "Formal end exceeded the total run deadline")
                    break
                if now >= self.deadline:
                    raise _Fault("COHORT_TIME_LIMIT", "Total run deadline reached at a synchronous boundary")
                self.expire(now)
                while self.pending and self.pending[0][0] <= (self.now if self.now is not None else now):
                    if len(self.requests) >= self.spec["requests_per_run"]:
                        self.pending.clear()
                        break
                    _, _, client, previous = heapq.heappop(self.pending)
                    self.offer(client, previous)
                if self.active:
                    if not self.native_pending():
                        raise _Fault("MISSING_TERMINAL_FEEDBACK", "Tracked requests disappeared without terminal API feedback")
                    if self.step_count >= _MAX_STEPS:
                        raise _Fault("CAPTURE_STEP_LIMIT", "Synchronous runtime step count exceeded the capture bound")
                    before = self.read_clock("step_before")
                    if before >= self.deadline:
                        raise _Fault("COHORT_TIME_LIMIT", "Run deadline reached before another kernel step")
                    active_count = len(self.active)
                    self.expire(before)
                    if len(self.active) != active_count:
                        # Submission itself may consume the request timeout.
                        # Cancel at this safe boundary before starting a kernel,
                        # then return eligible clients to the admission loop.
                        continue
                    self.step_count += 1
                    try:
                        records = self.backend.step()
                    except Exception as exc:
                        self.steps.append({"step_index": self.step_count - 1, "step_before_ns": before,
                                           "feedback_ns": None, "error": str(exc)[:4096],
                                           "exception_type": type(exc).__name__})
                        raise _Fault("BACKEND_STEP_ERROR", str(exc), exception_type=type(exc).__name__) from exc
                    try:
                        feedback = self.read_clock("step_feedback")
                    except _Fault:
                        # The backend already returned these counts. Preserve
                        # them even when their API feedback timestamp is lost.
                        try:
                            self.feedback(records, before, None)
                        except _Fault as secondary:
                            self.error(secondary.code, str(secondary), **secondary.details)
                        raise
                    self.feedback(records, before, feedback)
                    if feedback > self.deadline:
                        raise _Fault("COHORT_TIME_LIMIT", "An uninterruptible step crossed the total run deadline")
                    self.expire(self.now if self.now is not None else feedback)
                    bucket = self.counts()["terminal_count"] // 100
                    if bucket > self.progress_bucket:
                        self.progress_bucket = bucket
                        self.notify("progress")
                else:
                    if self.native_pending():
                        raise _Fault("UNTRACKED_NATIVE_WORK", "Backend has work without tracked ownership")
                    if not self.pending and len(self.requests) < self.spec["requests_per_run"]:
                        raise _Fault("CLOSED_LOOP_NO_PROGRESS", "No active request or pending client can offer the remaining stream")
                    if not self.injected_clock and self.pending:
                        target = min(self.pending[0][0], self.deadline)
                        remaining = max(0, target - (self.now if self.now is not None else now))
                        time.sleep(min(remaining / 1_000_000_000, .01))
        except _Fault as fault:
            self.error(fault.code, str(fault), **fault.details)
            self.cleanup(fault.code)
        except Exception as exc:
            self.error("CAPTURE_ERROR", str(exc), exception_type=type(exc).__name__)
            self.cleanup("CAPTURE_ERROR")
        if self.end is None:
            try:
                self.end = self.read_clock("formal_end")
            except _Fault as fault:
                self.error(fault.code, str(fault), **fault.details)
                self.end = self.now  # last observed valid clock, never the requested deadline
        if self.start is not None and self.end is not None and self.end <= self.start and self.requests:
            self.error("NONPOSITIVE_FORMAL_INTERVAL", "Formal API observation interval did not advance")
        try:
            self.notify("formal_end")
        except _Fault as fault:
            self.error(fault.code, str(fault), **fault.details)
        counts = self.counts()
        successful = (not self.errors and counts["completed_count"] == self.spec["requests_per_run"]
                      and counts["not_offered_count"] == 0 and self.resources_drained)
        backend_source = getattr(self.backend, "source", None)
        measured = (not self.injected_clock and isinstance(backend_source, dict)
                    and backend_source.get("kind") == "measured"
                    and isinstance(backend_source.get("reference"), str) and bool(backend_source["reference"].strip()))
        source = ({"kind": "measured", "reference": backend_source["reference"]} if measured and isinstance(backend_source, dict) else
                  {"kind": "test_fixture", "reference": "Injected clock or unattested backend; runtime API feedback only"})
        return {"schema_version": 1, "protocol": PROTOCOL, "capture_protocol": CAPTURE_PROTOCOL,
                "timing_boundary": TIMING_BOUNDARY, "source": source, "hardware_validated": False,
                "spec": deepcopy(self.spec), "input_spec_sha256": canonical_sha256(self.spec),
                "policy_sha256": self.policy_sha, "request_stream_sha256": self.stream["request_stream_sha256"],
                "request_stream": self.stream, "concurrency": self.concurrency, "replication": self.replication,
                "expected_requests": self.spec["requests_per_run"], **counts,
                "status": "COMPLETED" if successful else "FAILED",
                "error_code": self.errors[0]["code"] if self.errors else None, "errors": self.errors,
                "formal_start_ns": self.start, "formal_end_ns": self.end,
                "elapsed_ms": (self.end - self.start) / 1_000_000
                              if self.start is not None and self.end is not None else None,
                "resources_drained": self.resources_drained, "client_peak_outstanding": self.client_peak,
                "client_current_outstanding": self.client_outstanding,
                "unacknowledged_native_request_ids": list(self.active),
                "peak_logical_outstanding": self.peak_outstanding, "peak_tracked_native_outstanding": self.peak_native,
                "step_count": self.step_count, "feedback_record_count": self.feedback_count,
                "requests": self.requests, "step_feedback": self.steps,
                "clock": {"name": "injected_clock_ns" if self.injected_clock else "perf_counter_ns",
                          "unit": "nanoseconds", "injected": self.injected_clock,
                          "deadline_rounding": "positive fractional nanoseconds rounded upward"},
                "metric_contract": {"arrival": "submit_before_ns; actual API submission attempt, never a future placeholder",
                    "first_token": "first cumulative API feedback with output_tokens_actual > 0",
                    "terminal": "native finished feedback or synchronous abort acknowledgement",
                    "feedback_release": "one outstanding per client; terminal API feedback then think; fixed global stream order",
                    "timeout": "deadline observed at synchronous step boundary; no claim of kernel interruption",
                    "not_server_token_ready": True,
                    "stop_rule": "offer exactly N then drain; failures/timeouts release clients but veto overall success"}}


def collect_closed_loop(backend: Any, spec: dict, concurrency: int, *, replication: int = 0,
                        clock_ns: Callable[[], int] | None = None,
                        progress: Callable[[dict], None] | None = None) -> dict:
    """Capture runtime API observations from a dedicated synchronous backend.

    The backend exposes add_request(dict)->str, step()->list of cumulative
    feedback, has_unfinished_requests()->bool and synchronous abort_requests(list).
    Only an explicitly measured backend.source with the default real clock may
    label these observations measured. No result is labeled server-token-ready.
    """
    selected = validate_closed_loop_spec(spec)
    closed_loop_policy_sha256(selected, concurrency, replication)
    for method in ("add_request", "step", "has_unfinished_requests", "abort_requests"):
        if not callable(getattr(backend, method, None)):
            raise SimulationError("INVALID_CLOSED_LOOP_BACKEND", f"Backend needs callable {method}")
    if clock_ns is not None and not callable(clock_ns) or progress is not None and not callable(progress):
        raise SimulationError("INVALID_CLOSED_LOOP_BACKEND", "clock_ns and progress must be callable when supplied")
    return _Collector(backend, selected, concurrency, replication, clock_ns, progress).run()
