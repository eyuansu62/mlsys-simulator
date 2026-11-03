"""Deterministic batch/step DES for one GPU or one synchronous TP group.

Scheduling never receives future output lengths. The execution oracle consumes
those lengths only when committing a newly generated token.
"""

from dataclasses import dataclass, field
import heapq
import math

from .config import validate_experiment
from .memory import MemoryPlanner
from .performance import AnalyticalModel
from .schema import Experiment, SimulationError


TERMINAL = {"COMPLETED", "FAILED", "TIMED_OUT", "UNFINISHED"}


@dataclass
class SimulationResult:
    result: dict
    requests: list[dict]
    events: list[dict]


@dataclass
class _Request:
    request_id: str
    arrival_ms: float
    input_tokens: int
    priority: int
    status: str = "CREATED"
    generated: int = 0
    computed: int = 0
    first_token_ms: float | None = None
    completion_ms: float | None = None
    token_times: list[float] = field(default_factory=list)
    preemptions: int = 0
    recomputed_tokens: int = 0
    recompute_ms: float = 0.0
    memory_wait_ms: float = 0.0
    blocked_since: float | None = None
    error_code: str | None = None


@dataclass
class _Step:
    phase: str
    requests: list[_Request]
    queries: list[int]
    targets: dict[str, int]
    start_ms: float
    duration_ms: float


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


class _Simulation:
    def __init__(self, experiment: Experiment):
        validate_experiment(experiment)
        if experiment.strategy.pipeline_parallel_size != 1:
            raise SimulationError("UNSUPPORTED_MECHANISM", "PP online DES is not implemented; use predict.")
        if experiment.strategy.replicas != 1:
            raise SimulationError("UNSUPPORTED_MECHANISM", "Online simulation supports exactly one replica.")
        self.exp = experiment
        self.memory = MemoryPlanner(experiment)
        self.performance = AnalyticalModel(experiment)
        self.requests = {
            spec.request_id: _Request(spec.request_id, spec.arrival_ms,
                                      spec.input_tokens, spec.priority)
            for spec in experiment.workload
        }
        # Keep the oracle separate from the scheduler-facing request state.
        self._output_targets = {spec.request_id: spec.output_tokens for spec in experiment.workload}
        self.queue: list[tuple[float, int, int, str, object]] = []
        self.sequence = 0
        self.now = 0.0
        self.busy: _Step | None = None
        self.fifo_owner: str | None = None
        self.draining = False
        self.events: list[dict] = []
        self.handled_events = 0
        self.stop_reason: str | None = None
        self.recompute_execution_ms = 0.0
        self.compute_busy_ms = 0.0
        for spec in sorted(experiment.workload, key=lambda item: (item.arrival_ms, item.request_id)):
            self._push(spec.arrival_ms, 2, "REQUEST_ARRIVAL", spec.request_id)
            if spec.timeout_ms is not None:
                self._push(spec.arrival_ms + spec.timeout_ms, 1, "REQUEST_TIMEOUT", spec.request_id)

    def _push(self, when: float, priority: int, kind: str, payload: object) -> None:
        heapq.heappush(self.queue, (when, priority, self.sequence, kind, payload))
        self.sequence += 1

    def _log(self, kind: str, **details) -> None:
        if self.exp.execution.detailed_events:
            self.events.append({"timestamp_ms": self.now, "event": kind, **details})

    def _finish(self, request: _Request, status: str, code: str | None = None,
                release: bool = True) -> None:
        request.status = status
        request.completion_ms = self.now
        request.error_code = code
        self._end_memory_wait(request)
        if release:
            self.memory.release(request.request_id)
        self._log("REQUEST_FINISH", request_id=request.request_id,
                  status=status, error_code=code, kv_release_deferred=not release)

    def _end_memory_wait(self, request: _Request) -> None:
        if request.blocked_since is not None:
            request.memory_wait_ms += self.now - request.blocked_since
            request.blocked_since = None

    def _blocked(self, request: _Request, code: str | None) -> None:
        if request.blocked_since is None:
            request.blocked_since = self.now
            self._log("MEMORY_WAIT", request_id=request.request_id, error_code=code)

    def _commit_token(self, request: _Request) -> None:
        request.generated += 1
        request.token_times.append(self.now)
        if request.first_token_ms is None:
            request.first_token_ms = self.now
        self._log("TOKEN_READY", request_id=request.request_id,
                  output_token_index=request.generated)
        # The execution oracle is used only after the token has been committed.
        if request.generated >= self._output_targets[request.request_id]:
            self._finish(request, "COMPLETED")

    def _complete_step(self, step: _Step) -> None:
        self.busy = None
        self.compute_busy_ms += step.duration_ms
        if step.phase == "recompute":
            self.recompute_execution_ms += step.duration_ms
        self._log("STEP_COMPLETE", phase=step.phase,
                  request_ids=[request.request_id for request in step.requests])
        for request, query in zip(step.requests, step.queries):
            if request.status in TERMINAL:
                # A timed-out in-flight request stays pinned until this boundary.
                self.memory.release(request.request_id)
                self._log("DEFERRED_KV_RELEASE", request_id=request.request_id)
                continue
            request.computed = step.targets[request.request_id]
            request.status = "DECODING"
            if step.phase == "recompute":
                request.recomputed_tokens += query
                request.recompute_ms += step.duration_ms
            else:
                self._commit_token(request)

    def _handle(self, kind: str, payload: object) -> None:
        if kind == "STEP_COMPLETE":
            if not isinstance(payload, _Step):
                raise SimulationError("INVALID_EVENT", "Step completion payload is invalid.")
            self._complete_step(payload)
            return
        request = self.requests[str(payload)]
        if request.status in TERMINAL:
            return
        if kind == "REQUEST_ARRIVAL":
            request.status = "WAITING"
            self._log(kind, request_id=request.request_id)
        else:
            in_flight = self.busy is not None and request in self.busy.requests
            self._finish(request, "TIMED_OUT", "REQUEST_TIMEOUT", release=not in_flight)

    @staticmethod
    def _serving_key(request: _Request) -> tuple:
        return (-request.priority, request.arrival_ms, request.request_id)

    @staticmethod
    def _victim_key(request: _Request) -> tuple:
        return (request.priority, -request.arrival_ms, request.request_id)

    def _ready(self, status: str) -> list[_Request]:
        candidates = [request for request in self.requests.values() if request.status == status]
        return sorted(candidates, key=self._serving_key)

    def _start(self, phase: str, chosen: list[_Request], queries: list[int],
               targets: dict[str, int]) -> bool:
        contexts = [request.computed if phase == "decode" else 0 for request in chosen]
        estimate = self.performance.estimate_step(phase, queries, contexts)
        if not math.isfinite(estimate.duration_ms) or estimate.duration_ms <= 0:
            raise SimulationError("INVALID_PERFORMANCE_ESTIMATE", "A step duration must be finite and positive.")
        decision = self.memory.reserve(targets, queries, phase)
        if not decision.ok:
            raise SimulationError("RESOURCE_STATE_CHANGED", "Atomic reservation changed after scheduling checks.")
        for request in chosen:
            self._end_memory_wait(request)
            request.status = {"prefill": "PREFILL", "decode": "DECODING",
                              "recompute": "RECOMPUTING"}[phase]
        step = _Step(phase, chosen, queries, targets, self.now, estimate.duration_ms)
        self.busy = step
        self._log("STEP_START", phase=phase,
                  request_ids=[request.request_id for request in chosen],
                  query_tokens=queries, context_lengths=contexts,
                  duration_ms=estimate.duration_ms, breakdown=estimate.breakdown,
                  memory=self.memory.snapshot())
        self._push(self.now + estimate.duration_ms, 0, "STEP_COMPLETE", step)
        return True

    def _choose(self, candidates: list[_Request], phase: str,
                single: bool = False) -> tuple[list[_Request], list[int], dict[str, int]]:
        chosen: list[_Request] = []
        queries: list[int] = []
        targets: dict[str, int] = {}
        strategy = self.exp.strategy
        for request in candidates:
            if len(chosen) >= (1 if single else strategy.max_batch_size):
                break
            if phase == "prefill":
                query, target = request.input_tokens, request.input_tokens
            elif phase == "recompute":
                query = request.input_tokens + request.generated - 1
                target = query
            else:
                query, target = 1, request.computed + 1
            if query > strategy.max_batch_tokens:
                self._finish(request, "FAILED", "BATCH_TOKEN_LIMIT_EXCEEDED")
                continue
            # With no chunking, a restored prefix and its very next decode must fit.
            alone_target = target + (1 if phase == "recompute" else 0)
            alone = self.memory.check({request.request_id: alone_target}, [query], phase, alone=True)
            if not alone.ok:
                self._finish(request, "FAILED", alone.code)
                continue
            if sum(queries) + query > strategy.max_batch_tokens:
                continue
            trial_targets = {**targets, request.request_id: target}
            trial_queries = [*queries, query]
            decision = self.memory.check(trial_targets, trial_queries, phase)
            if not decision.ok:
                self._blocked(request, decision.code)
                if phase == "prefill" and self._ready("DECODING"):
                    self.draining = True
                continue
            if phase == "recompute":
                next_step = self.memory.check({request.request_id: target + 1}, [1], "decode")
                if not next_step.ok:
                    self._blocked(request, next_step.code)
                    continue
            chosen.append(request)
            queries.append(query)
            targets = trial_targets
        return chosen, queries, targets

    def _preempt(self, request: _Request) -> None:
        request.preemptions += 1
        self.memory.release(request.request_id)
        request.computed = 0
        self._log("REQUEST_PREEMPTED", request_id=request.request_id,
                  output_tokens_committed=request.generated,
                  preemption_count=request.preemptions)
        if request.preemptions > self.exp.strategy.max_preemptions_per_request:
            self._finish(request, "FAILED", "PREEMPTION_LIMIT_EXCEEDED")
        else:
            request.status = "WAITING_RECOMPUTE"
            self._blocked(request, "KV_CAPACITY_EXCEEDED")
        self.draining = True

    def _schedule_fifo(self) -> None:
        while self.busy is None:
            owner = self.requests.get(self.fifo_owner) if self.fifo_owner is not None else None
            if owner is None or owner.status in TERMINAL:
                ready = [request for request in self.requests.values()
                         if request.status not in TERMINAL | {"CREATED"}]
                if not ready:
                    return
                owner = min(ready, key=lambda request: (request.arrival_ms, request.request_id))
                self.fifo_owner = owner.request_id
            phase = "prefill" if owner.status == "WAITING" else "decode"
            chosen, queries, targets = self._choose([owner], phase, single=True)
            if chosen:
                self._start(phase, chosen, queries, targets)
                return
            if owner.status not in TERMINAL:
                self.stop_reason = "MEMORY_NO_PROGRESS"
                return

    def _schedule_batch(self) -> None:
        while self.busy is None:
            active = self._ready("DECODING")
            if not active:
                self.draining = False
            if not self.draining:
                chosen, queries, targets = self._choose(self._ready("WAITING"), "prefill")
                if chosen:
                    self._start("prefill", chosen, queries, targets)
                    return
            active = self._ready("DECODING")
            chosen, queries, targets = self._choose(active, "decode")
            if chosen:
                self._start("decode", chosen, queries, targets)
                return
            had_active = bool(active)
            active = self._ready("DECODING")
            if had_active and not active:
                # _choose may fail the last active request as individually
                # infeasible, releasing its blocks. Replan fresh admissions.
                continue
            if active:
                if self.exp.strategy.preemption == "none":
                    self.stop_reason = "MEMORY_NO_PROGRESS"
                    return
                victim = min(active, key=self._victim_key)
                self._preempt(victim)
                continue
            # A restored request gets a draining turn: no fresh admissions until
            # its next decode/continuation can actually make progress.
            chosen, queries, targets = self._choose(self._ready("WAITING_RECOMPUTE"),
                                                    "recompute", single=True)
            if chosen:
                self.draining = True
                self._start("recompute", chosen, queries, targets)
                return
            ready = self._ready("WAITING") + self._ready("WAITING_RECOMPUTE")
            if not ready:
                return
            # No resource-owning work can release anything; waiting is permanent.
            self.stop_reason = "MEMORY_NO_PROGRESS"
            return

    def run(self) -> SimulationResult:
        while self.queue:
            when = self.queue[0][0]
            if when > self.exp.execution.max_simulated_ms:
                self.now = self.exp.execution.max_simulated_ms
                self.stop_reason = "SIMULATION_TIME_LIMIT"
                break
            self.now = when
            while self.queue and self.queue[0][0] == when:
                if self.busy is None and all(request.status in TERMINAL for request in self.requests.values()):
                    self.queue.clear()
                    break
                if self.handled_events >= self.exp.execution.max_events:
                    self.stop_reason = "SIMULATION_EVENT_LIMIT"
                    break
                _, _, _, kind, payload = heapq.heappop(self.queue)
                self.handled_events += 1
                self._handle(kind, payload)
            if self.stop_reason is not None:
                break
            if self.busy is None:
                if self.exp.strategy.scheduler == "fifo":
                    self._schedule_fifo()
                else:
                    self._schedule_batch()
            if self.stop_reason is not None:
                break
            if all(request.status in TERMINAL for request in self.requests.values()) and self.busy is None:
                break
        if any(request.status not in TERMINAL for request in self.requests.values()):
            self.stop_reason = self.stop_reason or "MEMORY_NO_PROGRESS"
        if self.busy is not None:
            # A budget may interrupt virtual time between step boundaries, or
            # immediately before a completion event can be handled. Account
            # executed work without committing its unfinished token/KV state.
            elapsed = max(0.0, min(self.now - self.busy.start_ms, self.busy.duration_ms))
            self.compute_busy_ms += elapsed
            if self.busy.phase == "recompute":
                self.recompute_execution_ms += elapsed
                for request in self.busy.requests:
                    request.recompute_ms += elapsed
            self._log("STEP_INTERRUPTED", phase=self.busy.phase,
                      request_ids=[request.request_id for request in self.busy.requests],
                      elapsed_ms=elapsed, error_code=self.stop_reason)
        for request in self.requests.values():
            self._end_memory_wait(request)
            if request.status not in TERMINAL:
                request.status = "UNFINISHED"
                request.error_code = self.stop_reason
        return self._report()

    def _report(self) -> SimulationResult:
        rows: list[dict] = []
        for request in self.requests.values():
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
            })
        counts = {"total": len(rows), "completed": 0, "failed": 0,
                  "timed_out": 0, "unfinished": 0}
        for row in rows:
            counts[row["status"].lower()] += 1
        evaluation = self.exp.evaluation
        window = evaluation.measurement_window_ms
        cohort = rows if window is None else [row for row in rows if row["arrival_ms"] < window]
        window = self.now if window is None else window
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

        result = {
            "status": "UNRESOLVED" if self.stop_reason else "COMPLETED",
            "error_code": self.stop_reason,
            "backend": "simulated",
            "calibration_source": self.exp.calibration.backend,
            "profile_version": self.exp.calibration.profile_version,
            "hardware_validated": False,
            "quality_status": "not_evaluated",
            "simulated_ms": self.now,
            "events_processed": self.handled_events,
            "counts": counts,
            "completion_rate": counts["completed"] / counts["total"] if counts["total"] else None,
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
                "window_rule": "arrival_cohort_with_drain" if evaluation.measurement_window_ms is not None
                               else "zero_to_simulation_end",
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
            "recompute_execution_ms": self.recompute_execution_ms,
            "compute_busy_ms": self.compute_busy_ms,
            "memory": self.memory.snapshot(),
            "limitations": ["uncalibrated CPU model", "single replica; PP online unsupported",
                            "homogeneous phase batches; no chunked prefill",
                            "fixed KV pool; no allocator fragmentation or CPU swap",
                            "prefill-first admission; pressure drains active requests before recovery"],
        }
        return SimulationResult(result, rows, self.events)


def simulate(experiment: Experiment) -> SimulationResult:
    """Run one immutable experiment; never perform GPU execution or wall-clock sleeps."""
    return _Simulation(experiment).run()
