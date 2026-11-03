"""Deterministic batch/step DES for one GPU or one synchronous TP group.

Scheduling never receives future output lengths. The execution oracle consumes
those lengths only when committing a newly generated token.
"""

from dataclasses import dataclass, replace
import heapq
import math
from typing import TYPE_CHECKING

from .config import validate_experiment
from .memory import MemoryPlanner
from .reporting import report_simulation
from .schema import Experiment, SimulationError
from .state import RequestState as _Request, SimulationResult, TERMINAL

if TYPE_CHECKING:
    from .swap import HostSwapManager, SwapTransfer
@dataclass
class _Step:
    phase: str
    requests: list[_Request]
    queries: list[int]
    targets: dict[str, int]
    start_ms: float
    duration_ms: float

class _Simulation:
    def __init__(self, experiment: Experiment):
        from .profiles import create_performance_model

        validate_experiment(experiment)
        if experiment.strategy.pipeline_parallel_size != 1:
            raise SimulationError("UNSUPPORTED_MECHANISM", "The single-group engine requires PP=1.")
        if experiment.strategy.replicas != 1:
            raise SimulationError("UNSUPPORTED_MECHANISM", "Online simulation supports exactly one replica.")
        self.exp = experiment
        self.memory = MemoryPlanner(experiment)
        self.performance = create_performance_model(experiment)
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
        self.swap: "HostSwapManager | None" = None
        self.swap_busy: "SwapTransfer | None" = None
        if experiment.strategy.preemption == "swap":
            from .swap import HostSwapManager
            self.swap = HostSwapManager(experiment, self.memory)
        self.fifo_owner: str | None = None
        self.draining = False
        self.events: list[dict] = []
        self.handled_events = 0
        self.stop_reason: str | None = None
        self.recompute_execution_ms = 0.0
        self.compute_busy_ms = 0.0
        self.cutoff_ms = experiment.execution.max_simulated_ms
        self.cutoff_reason = "SIMULATION_TIME_LIMIT"
        window = experiment.evaluation.measurement_window_ms
        measurement_cutoff = None
        if window is not None:
            if not experiment.evaluation.drain:
                measurement_cutoff = window
                measurement_reason = "MEASUREMENT_WINDOW_END"
            elif experiment.evaluation.max_drain_ms is not None:
                measurement_cutoff = window + experiment.evaluation.max_drain_ms
                measurement_reason = "DRAIN_TIME_LIMIT"
            if measurement_cutoff is not None and measurement_cutoff < self.cutoff_ms:
                self.cutoff_ms = measurement_cutoff
                self.cutoff_reason = measurement_reason
        for spec in sorted(experiment.workload, key=lambda item: (item.arrival_ms, item.request_id)):
            if window is not None and spec.arrival_ms >= window:
                request = self.requests[spec.request_id]
                request.status = "NOT_ADMITTED"
                request.error_code = "OUTSIDE_MEASUREMENT_WINDOW"
                continue
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
        if self.swap is not None:
            residency = self.swap.state(request.request_id)
            if residency is not None:
                self.swap.cancel(request.request_id)
                if residency in {"SWAPPING_OUT", "SWAPPING_IN"}:
                    release = False
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
            if self.exp.strategy.prefix_cache and step.phase in {"prefill", "recompute"}:
                published = self.memory.publish_prefix(request.request_id, request.computed)
                self._log("PREFIX_CACHE_PUBLISH", request_id=request.request_id,
                          phase=step.phase, published_blocks=published,
                          computed_tokens=request.computed)
            if step.phase == "recompute":
                request.recomputed_tokens += query
                request.recompute_ms += step.duration_ms
                if request.computed < request.recompute_target:
                    request.status = "WAITING_RECOMPUTE"
                else:
                    request.recompute_target = 0
                    request.status = self._continuation_status(request)
            elif step.phase == "prefill":
                request.prefill_chunks += 1
                request.prefill_computed_tokens += query
                request.status = self._continuation_status(request)
                if request.computed == request.input_tokens:
                    self._commit_token(request)
            else:
                request.status = "DECODING"
                self._commit_token(request)

    def _handle(self, kind: str, payload: object) -> None:
        if kind == "SWAP_COMPLETE":
            from .swap import SwapTransfer
            if not isinstance(payload, SwapTransfer) or self.swap is None:
                raise SimulationError("INVALID_EVENT", "Swap completion payload is invalid.")
            completed = self.swap.complete(payload, self.now)
            self.swap_busy = None
            request = self.requests[payload.request_id]
            request.swap_ms += payload.end_ms - payload.start_ms
            if payload.direction == "D2H":
                request.swap_out_count += 1
                request.swap_out_bytes += payload.bytes
            else:
                request.swap_in_count += 1
                request.swap_in_bytes += payload.bytes
            self._log("SWAP_COMPLETE", **completed, memory=self.memory.snapshot(),
                      host=self.swap.snapshot(self.now))
            if request.status not in TERMINAL:
                request.status = ("SWAPPED" if payload.direction == "D2H"
                                  else self._continuation_status(request))
                self.draining = True
            return
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
            in_flight = ((self.busy is not None and request in self.busy.requests)
                         or (self.swap_busy is not None
                             and self.swap_busy.request_id == request.request_id))
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
        contexts = [targets[request.request_id] - query for request, query in zip(chosen, queries)]
        estimate = self.performance.estimate_step(phase, queries, contexts)
        if not math.isfinite(estimate.duration_ms) or estimate.duration_ms <= 0:
            raise SimulationError("INVALID_PERFORMANCE_ESTIMATE", "A step duration must be finite and positive.")
        completion_time = self.now + estimate.duration_ms
        if not math.isfinite(completion_time) or completion_time <= self.now:
            raise SimulationError("INVALID_PERFORMANCE_ESTIMATE",
                                  "A step must advance representable finite virtual time.")
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
                  source=estimate.source, assumptions=estimate.assumptions,
                  memory=self.memory.snapshot())
        self._push(completion_time, 0, "STEP_COMPLETE", step)
        return True

    @staticmethod
    def _continuation_status(request: _Request) -> str:
        if request.recompute_target > request.computed:
            return "WAITING_RECOMPUTE"
        return "PREFILLING" if request.computed < request.input_tokens else "DECODING"

    @staticmethod
    def _phase(request: _Request) -> str:
        if request.status == "WAITING_RECOMPUTE" or request.recompute_target > request.computed:
            return "recompute"
        return "prefill" if request.computed < request.input_tokens else "decode"

    def _iteration(self, request: _Request, phase: str, available: int,
                   computed: int | None = None) -> tuple[int, int]:
        current = request.computed if computed is None else computed
        if phase == "decode":
            return 1, current + 1
        goal = request.input_tokens if phase == "prefill" else request.recompute_target
        remaining = goal - current
        if self.exp.strategy.chunked_prefill:
            query = min(remaining, self.exp.strategy.prefill_chunk_tokens, available)
        else:
            query = remaining
        return query, current + query

    def _prefix_lookup(self, request: _Request) -> None:
        if request.cache_checked:
            return
        request.cache_checked = True
        if self.exp.strategy.prefix_cache:
            request.prefix_cached_tokens = self.memory.lookup_prefix(request.request_id)
            request.computed = request.prefix_cached_tokens
            self._log("PREFIX_CACHE_LOOKUP", request_id=request.request_id,
                      cached_tokens=request.prefix_cached_tokens)

    def _abandon_prefix(self, request: _Request) -> None:
        """An unlaunched cache hit may not pin memory ahead of runnable work."""
        if (request.status == "WAITING" and request.prefix_cached_tokens > 0
                and request.computed == request.prefix_cached_tokens
                and request.prefill_chunks == 0):
            self.memory.abandon_prefix_lookup(request.request_id)
            self._log("PREFIX_CACHE_LOOKUP_ABANDONED", request_id=request.request_id,
                      cached_tokens=request.prefix_cached_tokens)
            request.cache_checked = False
            request.prefix_cached_tokens = 0
            request.computed = 0

    def _choose(self, candidates: list[_Request], phase: str,
                single: bool = False) -> tuple[list[_Request], list[int], dict[str, int]]:
        chosen: list[_Request] = []
        queries: list[int] = []
        targets: dict[str, int] = {}
        strategy = self.exp.strategy
        for request in candidates:
            if (len(chosen) >= (1 if single else strategy.max_batch_size)
                    or sum(queries) >= strategy.max_batch_tokens):
                break
            if phase == "prefill" and request.status == "WAITING":
                self._prefix_lookup(request)
            if phase == "recompute" and not request.recompute_target:
                request.recompute_target = request.input_tokens + request.generated - 1
            query, target = self._iteration(request, phase, strategy.max_batch_tokens - sum(queries))
            if query <= 0:
                self._abandon_prefix(request)
                continue
            if query > strategy.max_batch_tokens:
                self._abandon_prefix(request)
                self._finish(request, "FAILED", "BATCH_TOKEN_LIMIT_EXCEEDED")
                continue
            # Chunking reduces Workspace/query size, never the final KV context
            # that must fit when this request is the only active request.
            required_kv = request.input_tokens if phase == "prefill" else target
            if phase == "recompute":
                goal = request.recompute_target
                next_phase = "prefill" if goal < request.input_tokens else "decode"
                _, required_kv = self._iteration(request, next_phase, strategy.max_batch_tokens,
                                                computed=goal)
            alone_kv = self.memory.check_kv({request.request_id: required_kv}, alone=True)
            if not alone_kv.ok:
                self._abandon_prefix(request)
                self._finish(request, "FAILED", alone_kv.code)
                continue
            alone = self.memory.check({request.request_id: target}, [query], phase, alone=True)
            if not alone.ok:
                self._abandon_prefix(request)
                self._finish(request, "FAILED", alone.code)
                continue
            if sum(queries) + query > strategy.max_batch_tokens:
                self._abandon_prefix(request)
                continue
            trial_targets = {**targets, request.request_id: target}
            trial_queries = [*queries, query]
            decision = self.memory.check(trial_targets, trial_queries, phase)
            if not decision.ok:
                self._abandon_prefix(request)
                self._blocked(request, decision.code)
                if phase == "prefill" and self._ready("DECODING"):
                    self.draining = True
                continue
            if phase == "recompute" and target == request.recompute_target:
                next_phase = "prefill" if target < request.input_tokens else "decode"
                next_query, next_target = self._iteration(request, next_phase,
                                                         strategy.max_batch_tokens, computed=target)
                next_alone = self.memory.check({request.request_id: next_target}, [next_query],
                                               next_phase, alone=True)
                if not next_alone.ok:
                    self._finish(request, "FAILED", next_alone.code)
                    continue
                next_step = self.memory.check({request.request_id: next_target}, [next_query], next_phase)
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
        request.recompute_target = max(request.computed, request.recompute_target)
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

    def _transfer(self, transfer: "SwapTransfer") -> None:
        self.swap_busy = transfer
        self.draining = True
        self._log("SWAP_START", request_id=transfer.request_id, direction=transfer.direction,
                  bytes=transfer.bytes, tokens=transfer.tokens, start_ms=transfer.start_ms,
                  end_ms=transfer.end_ms, memory=self.memory.snapshot())
        self._push(transfer.end_ms, 0, "SWAP_COMPLETE", transfer)

    def _swap_victim(self, active: list[_Request]) -> None:
        """Try safe victims in policy order; smaller feasible copies may win."""
        if self.swap is None:
            raise SimulationError("INVALID_SWAP_STATE", "Swap policy has no host manager.")
        for victim in sorted(active, key=self._victim_key):
            if victim.preemptions >= self.exp.strategy.max_preemptions_per_request:
                victim.preemptions += 1
                self._finish(victim, "FAILED", "PREEMPTION_LIMIT_EXCEEDED")
                return
            try:
                transfer = self.swap.swap_out(victim.request_id, self.now)
            except SimulationError as exc:
                if exc.code != "HOST_SWAP_CAPACITY_EXCEEDED":
                    raise
                self._log("SWAP_VICTIM_REJECTED", request_id=victim.request_id, error_code=exc.code)
                continue
            victim.preemptions += 1
            victim.status = "SWAPPING_OUT"
            self._blocked(victim, "KV_CAPACITY_EXCEEDED")
            self._log("REQUEST_PREEMPTED", request_id=victim.request_id,
                      recovery="swap", output_tokens_committed=victim.generated,
                      preemption_count=victim.preemptions)
            self._transfer(transfer)
            return
        if self.exp.strategy.swap_fallback == "recompute":
            self._log("SWAP_FALLBACK", recovery="recompute", error_code="HOST_SWAP_CAPACITY_EXCEEDED")
            self._preempt(min(active, key=self._victim_key))
        else:
            self.stop_reason = "HOST_SWAP_CAPACITY_EXCEEDED"

    def _restore_swap(self) -> bool:
        """Restore one request only after resident survivors have drained."""
        if self.swap is None:
            return False
        for request in self._ready("SWAPPED"):
            tokens = self.swap.stored_tokens(request.request_id)
            if tokens != request.computed:
                raise SimulationError("INVALID_SWAP_STATE", "Stored KV prefix differs from committed compute.")
            phase = self._phase(request)
            query, target = self._iteration(request, phase, self.exp.strategy.max_batch_tokens)
            alone = self.memory.check({request.request_id: target}, [query], phase, alone=True)
            if not alone.ok:
                self._finish(request, "FAILED", alone.code)
                continue
            try:
                transfer = self.swap.swap_in(request.request_id, self.now)
            except SimulationError as exc:
                if exc.code != "KV_CAPACITY_EXCEEDED":
                    raise
                self._blocked(request, exc.code)
                continue
            request.status = "SWAPPING_IN"
            self._transfer(transfer)
            return True
        return False

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
            phase = self._phase(owner)
            chosen, queries, targets = self._choose([owner], phase, single=True)
            if chosen:
                self._start(phase, chosen, queries, targets)
                return
            if owner.status not in TERMINAL:
                self.stop_reason = "MEMORY_NO_PROGRESS"
                return

    def _schedule_batch(self) -> None:
        if self.exp.strategy.chunked_prefill:
            self._schedule_chunked()
            return
        while self.busy is None:
            active = self._ready("DECODING")
            if not active:
                if self._restore_swap():
                    return
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
                if self.exp.strategy.preemption == "swap":
                    self._swap_victim(active)
                    if self.swap_busy is not None or self.stop_reason is not None:
                        return
                    continue
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
            ready = self._ready("WAITING") + self._ready("WAITING_RECOMPUTE") + self._ready("SWAPPED")
            if not ready:
                return
            # No resource-owning work can release anything; waiting is permanent.
            self.stop_reason = "MEMORY_NO_PROGRESS"
            return

    def _schedule_chunked(self) -> None:
        """Decode-first homogeneous batches; partial prompts retain their KV.

        Pressure recovery drains existing residents before admitting more fresh
        prompts. Recompute and DMA preserve the exact continuation position.
        """
        while self.busy is None and self.swap_busy is None:
            before = sum(r.status in TERMINAL for r in self.requests.values())
            # Decode always gets the next batch boundary when runnable.
            for phase, candidates in (
                ("decode", self._ready("DECODING")),
                ("recompute", [r for r in self._ready("WAITING_RECOMPUTE") if r.computed > 0]),
                ("prefill", self._ready("PREFILLING")
                 + [r for r in self._ready("WAITING") if r.computed > 0]),
            ):
                chosen, queries, targets = self._choose(candidates, phase)
                if chosen:
                    self._start(phase, chosen, queries, targets)
                    return
            residents = [r for r in self.requests.values()
                         if r.status not in TERMINAL and r.request_id in self.memory.allocations]
            if not residents:
                if self._restore_swap():
                    return
                recovering = self._ready("WAITING_RECOMPUTE")
                chosen, queries, targets = self._choose(recovering, "recompute", single=True)
                if chosen:
                    self.draining = True
                    self._start("recompute", chosen, queries, targets)
                    return
                self.draining = False
            if not self.draining:
                chosen, queries, targets = self._choose(self._ready("WAITING"), "prefill")
                if chosen:
                    self._start("prefill", chosen, queries, targets)
                    return
            after = sum(r.status in TERMINAL for r in self.requests.values())
            if after > before:
                continue
            residents = [r for r in self.requests.values()
                         if r.status not in TERMINAL and r.request_id in self.memory.allocations]
            if residents:
                if self.exp.strategy.preemption == "none":
                    self.stop_reason = "MEMORY_NO_PROGRESS"
                    return
                if self.exp.strategy.preemption == "swap":
                    self._swap_victim(residents)
                    if self.swap_busy is not None or self.stop_reason is not None:
                        return
                else:
                    self._preempt(min(residents, key=self._victim_key))
                continue
            pending = [r for r in self.requests.values() if r.status not in TERMINAL | {"CREATED"}]
            if pending:
                self.stop_reason = "MEMORY_NO_PROGRESS"
            return

    def run(self) -> SimulationResult:
        while self.queue:
            when = self.queue[0][0]
            if when > self.cutoff_ms:
                self.now = self.cutoff_ms
                self.stop_reason = self.cutoff_reason
                break
            self.now = when
            while self.queue and self.queue[0][0] == when:
                if (self.busy is None and self.swap_busy is None
                        and all(request.status in TERMINAL for request in self.requests.values())):
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
            if self.now >= self.cutoff_ms:
                if self.busy is not None or self.swap_busy is not None or any(request.status not in TERMINAL
                                                for request in self.requests.values()):
                    self.stop_reason = self.cutoff_reason
                break
            if self.busy is None and self.swap_busy is None:
                if self.exp.strategy.scheduler == "fifo":
                    self._schedule_fifo()
                else:
                    self._schedule_batch()
            if self.stop_reason is not None:
                break
            if (all(request.status in TERMINAL for request in self.requests.values())
                    and self.busy is None and self.swap_busy is None):
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
        if self.swap_busy is not None:
            transfer = self.swap_busy
            elapsed = max(0.0, min(self.now, transfer.end_ms) - transfer.start_ms)
            self.requests[transfer.request_id].swap_ms += elapsed
            self._log("SWAP_INTERRUPTED", request_id=transfer.request_id,
                      direction=transfer.direction, elapsed_ms=elapsed, error_code=self.stop_reason)
        for request in self.requests.values():
            self._end_memory_wait(request)
            if request.status not in TERMINAL:
                request.status = "UNFINISHED"
                request.error_code = self.stop_reason
        return self._report()

    def _report(self) -> SimulationResult:
        return report_simulation(self)


def _run_timeline(experiment: Experiment) -> SimulationResult:
    if experiment.strategy.pipeline_parallel_size > 1:
        from .pipeline import PipelineSimulation
        return PipelineSimulation(experiment).run()
    return _Simulation(experiment).run()


def simulate(experiment: Experiment) -> SimulationResult:
    """Run one immutable experiment; never perform GPU execution or wall-clock sleeps."""
    validate_experiment(experiment)
    if experiment.execution.initial_state != "empty_cache":
        raise SimulationError("UNSUPPORTED_MECHANISM", "Only empty_cache initial state is supported.")
    warmup = experiment.execution.warmup
    warmup_summary: dict = {"enabled": bool(warmup), "status": "NOT_REQUESTED",
                            "cache_effect": "none; measured run starts with empty KV"}
    if warmup:
        warmup_experiment = replace(
            experiment,
            workload=warmup,
            evaluation=replace(experiment.evaluation, measurement_window_ms=None,
                               drain=True, max_drain_ms=None),
            execution=replace(experiment.execution, warmup=()),
        )
        run = _run_timeline(warmup_experiment)
        if (run.result["status"] != "COMPLETED"
                or run.result["counts"]["completed"] != len(warmup)):
            cause = run.result["error_code"] or next(
                (row["error_code"] for row in run.requests if row["status"] != "COMPLETED"), None)
            raise SimulationError("WARMUP_FAILED",
                                  f"Warmup did not complete before measurement: {cause}; "
                                  f"counts={run.result['counts']}")
        warmup_summary.update({
            "status": "COMPLETED", "request_count": len(warmup),
            "simulated_ms": run.result["simulated_ms"],
            "compute_busy_ms": run.result["compute_busy_ms"],
            "recompute_execution_ms": run.result["recompute_execution_ms"],
            "calibration_source": run.result["calibration_source"],
            "profile_version": run.result["profile_version"],
            "profile_provenance": run.result["profile_provenance"],
            "pipeline": run.result.get("pipeline"),
            "swap": run.result.get("swap"),
        })
    measured = _run_timeline(experiment)
    measured.result["warmup"] = warmup_summary
    return measured
