"""Stage/microbatch online DES with explicit transport and token feedback.

KV is reserved across all stages before stage zero. A wave retains references
through feedback, including cancelled members. Pressure recovery first drains
all waves; DMA and stage execution do not overlap in this conservative model.
"""

from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Any

from .batching import batch_members, batch_phase
from .config import validate_experiment
from .memory import MemoryPlanner
from .profiles import create_performance_model
from .reporting import report_simulation
from .schema import Experiment, SimulationError, StepEstimate
from .state import RequestState, SimulationResult, TERMINAL


@dataclass
class _Wave:
    wave_id: int
    phase: str
    requests: tuple[str, ...]
    queries: tuple[int, ...]
    contexts: tuple[int, ...]
    targets: dict[str, int]
    estimates: tuple[StepEstimate, ...]
    request_phases: tuple[str, ...]


@dataclass(frozen=True)
class _StageTask:
    wave_id: int
    stage: int
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class _Transfer:
    wave_id: int
    boundary: int
    start_ms: float
    end_ms: float
    bytes_per_rank: int


class PipelineSimulation:
    """One model replica split into synchronous TP groups along a pipeline."""

    def __init__(self, experiment: Experiment):
        validate_experiment(experiment)
        if experiment.strategy.pipeline_parallel_size < 2:
            raise SimulationError("INVALID_CONFIG", "Pipeline engine requires at least two stages")
        self.exp = experiment
        self.memory = MemoryPlanner(experiment)
        self.performance = create_performance_model(experiment)
        self.requests = {r.request_id: RequestState(r.request_id, r.arrival_ms, r.input_tokens, r.priority)
                         for r in experiment.workload}
        # Only the execution oracle uses targets, after a token is committed.
        self._output_targets = {r.request_id: r.output_tokens for r in experiment.workload}
        self.now = 0.0
        self.stop_reason: str | None = None
        self.queue: list[tuple[float, int, int, str, Any]] = []
        self.sequence = 0
        self.handled_events = 0
        self.events: list[dict] = []
        self.pp = experiment.strategy.pipeline_parallel_size
        self.stage_queues: list[deque[int]] = [deque() for _ in range(self.pp)]
        self.stage_busy: list[_StageTask | None] = [None] * self.pp
        self.stage_busy_ms = [0.0] * self.pp
        self.stage_queue_wait_ms = [0.0] * self.pp
        self.stage_ready_at: dict[tuple[int, int], float] = {}
        self.stage_progress = {request_id: [0] * self.pp for request_id in self.requests}
        self.link_free_ms = [0.0] * (self.pp - 1)
        self.link_busy_ms = [0.0] * (self.pp - 1)
        self.link_bytes = [0] * (self.pp - 1)
        self.transfers: dict[int, _Transfer] = {}
        self.waves: dict[int, _Wave] = {}
        self.pending_requests: dict[str, int] = {}
        self.feedback_pending: dict[int, float] = {}
        self.next_wave_id = 0
        self.peak_waves = 0
        self.draining = False
        self.quiescing = False
        self.fifo_owner: str | None = None
        self.compute_busy_ms = 0.0
        self.recompute_execution_ms = 0.0
        self.mixed_batch_count = 0
        self.mixed_batch_completed_count = 0
        self.mixed_compute_busy_ms = 0.0
        self.feedback_delay_ms = experiment.strategy.pipeline_feedback_latency_us / 1000
        if (not math.isfinite(self.feedback_delay_ms)
                or (experiment.strategy.pipeline_feedback_latency_us > 0 and self.feedback_delay_ms <= 0)):
            raise SimulationError("INVALID_EVENT_TIME", "Positive feedback delay must survive unit conversion")
        self.swap: Any = None
        self.swap_transfer: Any = None
        if experiment.strategy.preemption == "swap":
            from .swap import HostSwapManager
            self.swap = HostSwapManager(experiment, self.memory)
        self.limitations = [
            "one replica; stage groups and paired TP links are independently serialized",
            ("decode-first mixed prefill/decode microbatches; recomputation stays homogeneous"
             if experiment.strategy.mixed_batching else
             "homogeneous phase microbatches; enable mixed_batching to combine prefill and decode"),
            "prefix sharing requires explicit token identities; no shared-link contention",
            "all-stage KV reservation and persistent activation-buffer bounds are conservative",
            "pressure recovery quiesces waves; DMA and GPU execution do not overlap",
            "feedback is a configured end-to-end delay without control-link contention",
        ]
        self.cutoff_ms = experiment.execution.max_simulated_ms
        self.cutoff_reason = "SIMULATION_TIME_LIMIT"
        evaluation = experiment.evaluation
        window = evaluation.measurement_window_ms
        if window is not None:
            candidate, reason = None, None
            if not evaluation.drain:
                candidate, reason = window, "MEASUREMENT_WINDOW_END"
            elif evaluation.max_drain_ms is not None:
                candidate, reason = window + evaluation.max_drain_ms, "DRAIN_TIME_LIMIT"
            if candidate is not None and candidate < self.cutoff_ms:
                self.cutoff_ms, self.cutoff_reason = candidate, str(reason)
        for spec in sorted(experiment.workload, key=lambda r: (r.arrival_ms, r.request_id)):
            if window is not None and spec.arrival_ms >= window:
                self.requests[spec.request_id].status = "NOT_ADMITTED"
                self.requests[spec.request_id].error_code = "OUTSIDE_MEASUREMENT_WINDOW"
                continue
            self._push(spec.arrival_ms, 2, "REQUEST_ARRIVAL", spec.request_id)
            if spec.timeout_ms is not None:
                self._push(spec.arrival_ms + spec.timeout_ms, 1, "REQUEST_TIMEOUT", spec.request_id)

    def _push(self, when: float, priority: int, kind: str, payload: Any) -> None:
        if not math.isfinite(when) or when < self.now:
            raise SimulationError("INVALID_EVENT_TIME", "Pipeline event time must be finite and monotonic")
        heapq.heappush(self.queue, (when, priority, self.sequence, kind, payload))
        self.sequence += 1

    def _log(self, kind: str, **details: Any) -> None:
        if self.exp.execution.detailed_events:
            self.events.append({"timestamp_ms": self.now, "event": kind, **details})

    @staticmethod
    def _serving_key(request: RequestState) -> tuple:
        return -request.priority, request.arrival_ms, request.request_id

    @staticmethod
    def _victim_key(request: RequestState) -> tuple:
        return request.priority, -request.arrival_ms, request.request_id

    def _ready(self, status: str) -> list[RequestState]:
        return sorted((r for r in self.requests.values()
                       if r.status == status and r.request_id not in self.pending_requests),
                      key=self._serving_key)

    def _blocked(self, request: RequestState, code: str | None) -> None:
        if request.blocked_since is None:
            request.blocked_since = self.now
            self._log("MEMORY_WAIT", request_id=request.request_id, error_code=code)

    def _end_wait(self, request: RequestState) -> None:
        if request.blocked_since is not None:
            request.memory_wait_ms += self.now - request.blocked_since
            request.blocked_since = None

    def _finish(self, request: RequestState, status: str, code: str | None = None) -> None:
        request.status, request.completion_ms, request.error_code = status, self.now, code
        self._end_wait(request)
        deferred = request.request_id in self.pending_requests
        if self.swap is not None and self.swap.state(request.request_id) is not None:
            state = self.swap.state(request.request_id)
            self.swap.cancel(request.request_id)
            deferred |= state in {"SWAPPING_OUT", "SWAPPING_IN"}
        if not deferred:
            self.memory.release(request.request_id)
        self._log("REQUEST_FINISH", request_id=request.request_id, status=status,
                  error_code=code, kv_release_deferred=deferred)

    def _commit_token(self, request: RequestState) -> None:
        request.generated += 1
        request.token_times.append(self.now)
        if request.first_token_ms is None:
            request.first_token_ms = self.now
        self._log("TOKEN_READY", request_id=request.request_id, output_token_index=request.generated)
        if request.generated >= self._output_targets[request.request_id]:
            self._finish(request, "COMPLETED")

    @staticmethod
    def _continuation_status(request: RequestState) -> str:
        if request.computed < request.recompute_target:
            return "WAITING_RECOMPUTE"
        return "PREFILLING" if request.computed < request.input_tokens else "DECODING"

    def _iteration(self, request: RequestState, phase: str, available: int,
                   computed: int | None = None) -> tuple[int, int]:
        current = request.computed if computed is None else computed
        if phase == "decode":
            return 1, current + 1
        goal = request.input_tokens if phase == "prefill" else request.recompute_target
        query = goal - current
        if self.exp.strategy.chunked_prefill:
            query = min(query, self.exp.strategy.prefill_chunk_tokens, available)
        return query, current + query

    def _lookup_prefix(self, request: RequestState) -> None:
        if request.cache_checked:
            return
        request.cache_checked = True
        if self.exp.strategy.prefix_cache:
            hit = self.memory.lookup_prefix(request.request_id)
            request.computed = hit
            request.prefix_cached_tokens = hit
            self.stage_progress[request.request_id] = [hit] * self.pp
            self._log("PREFIX_CACHE_LOOKUP", request_id=request.request_id,
                      cached_tokens=hit)

    def _abandon_prefix(self, request: RequestState) -> None:
        if request.status != "WAITING" or request.prefix_cached_tokens <= 0:
            return
        self.memory.abandon_prefix_lookup(request.request_id)
        request.cache_checked = False
        request.prefix_cached_tokens = 0
        request.computed = 0
        self.stage_progress[request.request_id] = [0] * self.pp
        self._log("PREFIX_CACHE_LOOKUP_ABANDONED", request_id=request.request_id,
                  reason="suffix_not_admitted")

    def _choose(self, candidates: list[RequestState], phase: str) -> tuple[list[RequestState], list[int], dict]:
        chosen: list[RequestState] = []
        queries: list[int] = []
        targets: dict[str, int] = {}
        strategy = self.exp.strategy
        maximum = 1 if strategy.scheduler == "fifo" else min(strategy.max_batch_size,
                                                              strategy.pipeline_microbatch_size)
        for request in candidates:
            if request.status in TERMINAL:
                continue
            if len(chosen) >= maximum:
                break
            available = strategy.max_batch_tokens - sum(queries)
            if available <= 0:
                break
            if phase == "prefill":
                self._lookup_prefix(request)
            query, target = self._iteration(request, phase, available)
            if query > strategy.max_batch_tokens:
                self._abandon_prefix(request)
                self._finish(request, "FAILED", "BATCH_TOKEN_LIMIT_EXCEEDED")
                continue
            required_kv = request.input_tokens if phase == "prefill" else target
            if phase == "recompute":
                next_phase = "prefill" if request.recompute_target < request.input_tokens else "decode"
                _, required_kv = self._iteration(request, next_phase, strategy.max_batch_tokens,
                                                computed=request.recompute_target)
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
            trial = {**targets, request.request_id: target}
            trial_queries = [*queries, query]
            decision = self.memory.check(trial, trial_queries, phase)
            if not decision.ok:
                self._abandon_prefix(request)
                self._blocked(request, decision.code)
                continue
            if phase == "recompute" and target == request.recompute_target:
                next_phase = "prefill" if target < request.input_tokens else "decode"
                next_query, next_target = self._iteration(request, next_phase,
                                                         strategy.max_batch_tokens, computed=target)
                feasible = self.memory.check({request.request_id: next_target}, [next_query], next_phase,
                                             alone=True)
                if not feasible.ok:
                    self._finish(request, "FAILED", feasible.code)
                    continue
                following = self.memory.check({request.request_id: next_target}, [next_query], next_phase)
                if not following.ok:
                    self._blocked(request, following.code)
                    continue
            chosen.append(request)
            queries.append(query)
            targets = trial
        return chosen, queries, targets

    def _choose_serving(self, active: list[RequestState], partial: list[RequestState],
                        fresh: list[RequestState]) -> tuple[list[RequestState], list[int], dict, list[str]]:
        """Assemble one real batch before querying its joint Workspace/Profile.

        Decode owns the first sequence and token slots. Logical KV checks keep
        candidate selection independent of any unexecuted homogeneous timing
        shape. Only the actual selected role vector queries compute Workspace.
        """
        strategy = self.exp.strategy
        maximum = min(strategy.max_batch_size, strategy.pipeline_microbatch_size)
        chosen: list[RequestState] = []
        queries: list[int] = []
        targets: dict[str, int] = {}
        members: list[str] = []
        candidates = [(request, "decode") for request in active]
        candidates += [(request, "prefill") for request in partial]
        if not self.draining:
            candidates += [(request, "prefill") for request in fresh]
        for request, member in candidates:
            if request.status in TERMINAL or request.request_id in targets:
                continue
            available = strategy.max_batch_tokens - sum(queries)
            if len(chosen) >= maximum or available <= 0:
                break
            if member == "prefill" and request.status == "WAITING":
                self._lookup_prefix(request)
            query, target = self._iteration(request, member, available)
            if query > strategy.max_batch_tokens:
                self._abandon_prefix(request)
                self._finish(request, "FAILED", "BATCH_TOKEN_LIMIT_EXCEEDED")
                continue
            if query <= 0 or query > available:
                self._abandon_prefix(request)
                continue
            required = request.input_tokens if member == "prefill" else target
            alone = self.memory.check_kv({request.request_id: required}, alone=True)
            if not alone.ok:
                self._abandon_prefix(request)
                self._finish(request, "FAILED", alone.code)
                continue
            trial = {**targets, request.request_id: target}
            decision = self.memory.check_kv(trial)
            if not decision.ok:
                self._abandon_prefix(request)
                self._blocked(request, decision.code)
                continue
            chosen.append(request)
            queries.append(query)
            targets = trial
            members.append(member)
        while chosen:
            phase = batch_phase(members)
            roles = members if phase == "mixed" else None
            decision = self.memory.check(targets, queries, phase, request_phases=roles)
            if decision.ok:
                return chosen, queries, targets, members
            # Resource pressure can reduce a real proposed batch; timing-domain
            # errors are exceptions and must never trigger a pure-phase fallback.
            request = chosen[-1]
            if len(chosen) == 1:
                alone = self.memory.check(targets, queries, phase, alone=True,
                                          request_phases=roles)
                self._abandon_prefix(request)
                if not alone.ok:
                    self._finish(request, "FAILED", alone.code)
                else:
                    self._blocked(request, decision.code)
            else:
                self._abandon_prefix(request)
                self._blocked(request, decision.code)
            targets.pop(request.request_id)
            chosen.pop()
            queries.pop()
            members.pop()
        return [], [], {}, []

    @staticmethod
    def _member_metadata(wave: _Wave) -> dict:
        return {"request_phases": list(wave.request_phases)} if wave.phase == "mixed" else {}

    def _admit(self, phase: str, chosen: list[RequestState], queries: list[int], targets: dict,
               request_phases: list[str] | None = None) -> None:
        members = batch_members(phase, queries, request_phases)
        contexts = tuple(targets[r.request_id] - query for r, query in zip(chosen, queries))
        estimates = tuple(self.performance.estimate_step(phase, queries, contexts, stage,
                          request_phases=members if phase == "mixed" else None)
                          for stage in range(self.pp))
        for estimate in estimates:
            if not math.isfinite(estimate.duration_ms) or estimate.duration_ms <= 0:
                raise SimulationError("INVALID_PERFORMANCE_ESTIMATE", "Stage duration must be positive and finite")
        decision = self.memory.reserve(targets, queries, phase,
                                       request_phases=members if phase == "mixed" else None)
        if not decision.ok:
            raise SimulationError("RESOURCE_STATE_CHANGED", "Pipeline admission reservation changed")
        wave = _Wave(self.next_wave_id, phase, tuple(r.request_id for r in chosen),
                     tuple(queries), contexts, targets, estimates, members)
        self.next_wave_id += 1
        if phase == "mixed":
            self.mixed_batch_count += 1
        self.waves[wave.wave_id] = wave
        for request, member in zip(chosen, members):
            self._end_wait(request)
            request.status = {"prefill": "PREFILL", "decode": "DECODING", "recompute": "RECOMPUTING"}[member]
            self.pending_requests[request.request_id] = wave.wave_id
        self.stage_queues[0].append(wave.wave_id)
        self.stage_ready_at[wave.wave_id, 0] = self.now
        self.peak_waves = max(self.peak_waves, len(self.waves))
        self._log("MICROBATCH_ADMITTED", microbatch_id=wave.wave_id, phase=phase,
                  request_ids=list(wave.requests), query_tokens=list(queries),
                  context_lengths=list(contexts), memory=self.memory.snapshot(),
                  **self._member_metadata(wave))

    def _start_stages(self) -> None:
        if self.swap_transfer is not None:
            return
        for stage in reversed(range(self.pp)):
            if self.stage_busy[stage] is not None or not self.stage_queues[stage]:
                continue
            wave_id = self.stage_queues[stage].popleft()
            wave = self.waves[wave_id]
            duration = wave.estimates[stage].duration_ms
            end = self.now + duration
            if not math.isfinite(end) or end <= self.now:
                raise SimulationError("INVALID_PERFORMANCE_ESTIMATE", "Stage must advance representable time")
            task = _StageTask(wave_id, stage, self.now, end)
            self.stage_busy[stage] = task
            self.stage_queue_wait_ms[stage] += self.now - self.stage_ready_at.pop((wave_id, stage))
            self._log("STAGE_START", microbatch_id=wave_id, stage=stage, phase=wave.phase,
                      request_ids=list(wave.requests), query_tokens=list(wave.queries),
                      context_lengths=list(wave.contexts), duration_ms=duration,
                      source=wave.estimates[stage].source, breakdown=wave.estimates[stage].breakdown,
                      assumptions=wave.estimates[stage].assumptions,
                      **self._member_metadata(wave))
            self._push(end, 0, "STAGE_COMPLETE", task)

    def _stage_complete(self, task: _StageTask) -> None:
        if self.stage_busy[task.stage] != task:
            raise SimulationError("INVALID_EVENT", "Stale stage completion")
        self.stage_busy[task.stage] = None
        elapsed = task.end_ms - task.start_ms
        self.stage_busy_ms[task.stage] += elapsed
        self.compute_busy_ms += elapsed
        wave = self.waves[task.wave_id]
        if wave.phase == "mixed":
            self.mixed_compute_busy_ms += elapsed
        if wave.phase == "recompute":
            self.recompute_execution_ms += elapsed
            for name in wave.requests:
                self.requests[name].recompute_ms += elapsed
        for name in wave.requests:
            self.stage_progress[name][task.stage] = wave.targets[name]
        self._log("STAGE_COMPLETE", microbatch_id=wave.wave_id, stage=task.stage,
                  phase=wave.phase, request_ids=list(wave.requests),
                  **self._member_metadata(wave))
        if task.stage == self.pp - 1:
            if wave.phase == "mixed":
                self.mixed_batch_completed_count += 1
            for name, query, member in zip(wave.requests, wave.queries, wave.request_phases):
                request = self.requests[name]
                if request.status in TERMINAL:
                    continue
                request.computed = wave.targets[name]
                if member == "recompute":
                    request.recomputed_tokens += query
                elif member == "prefill":
                    request.prefill_chunks += 1
                    request.prefill_computed_tokens += query
                    self._log("PREFILL_CHUNK_COMPLETE", microbatch_id=wave.wave_id,
                              request_id=name, query_tokens=query, computed_tokens=request.computed,
                              final=request.computed == request.input_tokens)
                    if request.computed == request.input_tokens:
                        self._log("SAMPLE_COMPLETE", microbatch_id=wave.wave_id, request_id=name)
                        self._commit_token(request)
                else:
                    self._log("SAMPLE_COMPLETE", microbatch_id=wave.wave_id, request_id=name)
                    self._commit_token(request)
                if request.status not in TERMINAL:
                    request.status = "WAITING_FEEDBACK"
            delay = self.feedback_delay_ms
            end = self.now + delay
            if not math.isfinite(end) or (delay > 0 and end <= self.now):
                raise SimulationError("INVALID_EVENT_TIME", "Feedback must have representable finite time")
            self.feedback_pending[wave.wave_id] = end
            self._log("FEEDBACK_START", microbatch_id=wave.wave_id, request_ids=list(wave.requests),
                      duration_ms=delay, **self._member_metadata(wave))
            self._push(end, 0, "FEEDBACK_COMPLETE", wave.wave_id)
            return
        model, infra = self.exp.strategy.model, self.exp.infra
        payload = int(sum(wave.queries) * model.hidden_size * model.activation_bytes)
        if self.exp.strategy.pipeline_p2p:
            bandwidth = infra.link_bandwidth_gbps * 1e6
            duration = infra.link_latency_us / 1000 + payload / bandwidth
            if not math.isfinite(bandwidth) or bandwidth <= 0 or not math.isfinite(duration) or duration <= 0:
                raise SimulationError("INVALID_PERFORMANCE_ESTIMATE", "P2P duration must be positive and finite")
        else:
            duration = 0.0
        start = max(self.now, self.link_free_ms[task.stage])
        end = start + duration
        if not math.isfinite(end) or (duration > 0 and end <= start):
            raise SimulationError("INVALID_EVENT_TIME", "P2P must advance representable time")
        transfer = _Transfer(wave.wave_id, task.stage, start, end, payload)
        self.link_free_ms[task.stage] = end
        self.transfers[wave.wave_id] = transfer
        self._push(start, 0, "TRANSFER_START", transfer)
        self._push(end, 0, "TRANSFER_COMPLETE", transfer)

    def _feedback_complete(self, wave_id: int) -> None:
        wave = self.waves.pop(wave_id)
        self.feedback_pending.pop(wave_id)
        for name, member in zip(wave.requests, wave.request_phases):
            self.pending_requests.pop(name)
            request = self.requests[name]
            if (self.exp.strategy.prefix_cache and member in {"prefill", "recompute"}
                    and request.status in {"WAITING_FEEDBACK", "COMPLETED"}):
                published = self.memory.publish_prefix(name, request.computed)
                self._log("PREFIX_CACHE_PUBLISH", request_id=name, published_blocks=published,
                          computed_tokens=request.computed)
            if request.status in TERMINAL:
                self.memory.release(name)
                self._log("DEFERRED_KV_RELEASE", request_id=name, microbatch_id=wave_id)
            else:
                if member == "recompute" and request.computed >= request.recompute_target:
                    request.recompute_target = 0
                request.status = self._continuation_status(request)
        self._log("FEEDBACK_COMPLETE", microbatch_id=wave_id, request_ids=list(wave.requests),
                  **self._member_metadata(wave))

    def _recompute(self, request: RequestState) -> None:
        request.preemptions += 1
        self.memory.release(request.request_id)
        request.recompute_target = max(request.recompute_target, request.computed)
        request.computed = 0
        self.stage_progress[request.request_id] = [0] * self.pp
        self._log("REQUEST_PREEMPTED", request_id=request.request_id,
                  mode="recompute", output_tokens_committed=request.generated)
        if request.preemptions > self.exp.strategy.max_preemptions_per_request:
            self._finish(request, "FAILED", "PREEMPTION_LIMIT_EXCEEDED")
        else:
            request.status = "WAITING_RECOMPUTE"
            self._blocked(request, "KV_CAPACITY_EXCEEDED")
        self.draining = True

    def _begin_swap(self, request: RequestState, direction: str) -> None:
        if self.waves or any(self.stage_busy) or any(self.stage_queues) or self.transfers:
            raise SimulationError("UNSAFE_PREEMPTION", "DMA requires a quiescent pipeline")
        transfer = (self.swap.swap_out(request.request_id, self.now) if direction == "D2H"
                    else self.swap.swap_in(request.request_id, self.now))
        self.swap_transfer = transfer
        if direction == "D2H":
            request.preemptions += 1
            request.status = "SWAPPING_OUT"
            self._blocked(request, "KV_CAPACITY_EXCEEDED")
        else:
            request.status = "SWAPPING_IN"
        self._log("SWAP_START", request_id=request.request_id, direction=direction,
                  bytes=transfer.bytes, tokens=transfer.tokens,
                  start_ms=transfer.start_ms, end_ms=transfer.end_ms)
        self._push(transfer.end_ms, 0, "SWAP_COMPLETE", transfer)
        self.draining = True

    def _preempt_one(self, active: list[RequestState]) -> None:
        if self.exp.strategy.preemption == "none":
            self.stop_reason = "MEMORY_NO_PROGRESS"
            return
        victims = sorted(active, key=self._victim_key)
        if self.exp.strategy.preemption == "recompute":
            self._recompute(victims[0])
            return
        last_error = "HOST_SWAP_CAPACITY_EXCEEDED"
        for victim in victims:
            if victim.preemptions >= self.exp.strategy.max_preemptions_per_request:
                victim.preemptions += 1
                self._finish(victim, "FAILED", "PREEMPTION_LIMIT_EXCEEDED")
                return
            try:
                self._begin_swap(victim, "D2H")
                return
            except SimulationError as exc:
                if exc.code != "HOST_SWAP_CAPACITY_EXCEEDED":
                    raise
                last_error = exc.code
        if self.exp.strategy.swap_fallback == "recompute":
            self._recompute(victims[0])
        else:
            self.stop_reason = last_error

    def _restore_one(self) -> bool:
        for request in self._ready("SWAPPED"):
            status = self._continuation_status(request)
            phase = {"PREFILLING": "prefill", "WAITING_RECOMPUTE": "recompute",
                     "DECODING": "decode"}[status]
            query, target = self._iteration(request, phase, self.exp.strategy.max_batch_tokens)
            next_step = self.memory.check({request.request_id: target}, [query], phase, alone=True)
            if not next_step.ok:
                self._finish(request, "FAILED", next_step.code)
                continue
            try:
                self._begin_swap(request, "H2D")
                return True
            except SimulationError as exc:
                if exc.code != "KV_CAPACITY_EXCEEDED":
                    raise
                self._blocked(request, exc.code)
        return False

    def _owner_candidates(self) -> list[RequestState] | None:
        if self.exp.strategy.scheduler != "fifo":
            return None
        owner = self.requests.get(self.fifo_owner) if self.fifo_owner else None
        if owner is not None and owner.request_id in self.pending_requests:
            return []
        if owner is None or owner.status in TERMINAL:
            ready = [r for r in self.requests.values() if r.status not in TERMINAL | {"CREATED"}]
            if not ready:
                return []
            owner = min(ready, key=lambda r: (r.arrival_ms, r.request_id))
            self.fifo_owner = owner.request_id
        return [owner]

    def _schedule(self) -> None:
        if self.swap_transfer is not None:
            return
        if self.quiescing and self.waves:
            self._start_stages()
            return
        self.quiescing = False
        # Every iteration either admits a wave, removes an impossible request,
        # frees a victim, starts a copy, or returns to a future event.
        while len(self.waves) < self.exp.strategy.pipeline_max_inflight_microbatches:
            owner = self._owner_candidates()
            if owner == []:
                break
            active = self._ready("DECODING") if owner is None else [r for r in owner if r.status == "DECODING"]
            partial = self._ready("PREFILLING") if owner is None else [r for r in owner if r.status == "PREFILLING"]
            all_active = [r for r in self.requests.values()
                          if r.status not in TERMINAL and r.request_id in self.memory.allocations]
            if (not all_active and not self._ready("SWAPPED")
                    and not self._ready("WAITING_RECOMPUTE")):
                self.draining = False
            fresh = self._ready("WAITING") if owner is None else [r for r in owner if r.status == "WAITING"]
            if self.exp.strategy.mixed_batching and owner is None:
                chosen, queries, targets, members = self._choose_serving(active, partial, fresh)
                if chosen:
                    phase = batch_phase(members)
                    self._admit(phase, chosen, queries, targets,
                                members if phase == "mixed" else None)
                    continue
                if fresh and any(r.status == "WAITING" for r in fresh):
                    self.draining = True
            else:
                if self.exp.strategy.chunked_prefill:
                    chosen, queries, targets = self._choose(active, "decode")
                    if chosen:
                        self._admit("decode", chosen, queries, targets)
                        continue
                chosen, queries, targets = self._choose(partial, "prefill")
                if chosen:
                    self._admit("prefill", chosen, queries, targets)
                    continue
                if fresh and not self.draining:
                    chosen, queries, targets = self._choose(fresh, "prefill")
                    if chosen:
                        self._admit("prefill", chosen, queries, targets)
                        continue
                    if any(r.status == "WAITING" for r in fresh):
                        self.draining = True
                chosen, queries, targets = self._choose(active, "decode")
                if chosen:
                    self._admit("decode", chosen, queries, targets)
                    continue
            active = ([r for r in self.requests.values()
                       if r.status in {"DECODING", "PREFILLING"}
                       and r.request_id not in self.pending_requests] if owner is None else
                      [r for r in owner if r.status in {"DECODING", "PREFILLING"}])
            if self.waves:
                if active or any(r.status == "WAITING" for r in fresh):
                    self.quiescing = True
                break
            if active:
                self._preempt_one(active)
                if self.stop_reason is not None or self.swap_transfer is not None:
                    break
                continue
            if self.swap is not None and self._restore_one():
                break
            recovering = (self._ready("WAITING_RECOMPUTE") if owner is None else
                          [r for r in owner if r.status == "WAITING_RECOMPUTE"])
            chosen, queries, targets = self._choose(recovering, "recompute")
            if chosen:
                self.draining = True
                self._admit("recompute", chosen, queries, targets)
                continue
            resident_recovery = [r for r in recovering if r.status == "WAITING_RECOMPUTE"
                                 and r.request_id in self.memory.allocations]
            if resident_recovery:
                self._preempt_one(resident_recovery)
                if self.stop_reason is not None or self.swap_transfer is not None:
                    break
                continue
            remaining = [r for r in self.requests.values() if r.status not in TERMINAL | {"CREATED"}]
            # _choose may have failed the final resident, clearing pressure.
            if self.draining and not self.memory.allocations and self._ready("WAITING"):
                self.draining = False
                continue
            if remaining:
                self.stop_reason = "MEMORY_NO_PROGRESS"
            break
        self._start_stages()

    def _handle(self, kind: str, payload: Any) -> None:
        if kind == "STAGE_COMPLETE":
            self._stage_complete(payload)
        elif kind == "TRANSFER_START":
            self._log(kind, microbatch_id=payload.wave_id, boundary=payload.boundary,
                      bytes_per_rank=payload.bytes_per_rank, start_ms=payload.start_ms,
                      end_ms=payload.end_ms)
        elif kind == "TRANSFER_COMPLETE":
            transfer = self.transfers.pop(payload.wave_id)
            self.link_busy_ms[transfer.boundary] += transfer.end_ms - transfer.start_ms
            self.link_bytes[transfer.boundary] += transfer.bytes_per_rank * self.exp.strategy.tensor_parallel_size
            stage = transfer.boundary + 1
            self.stage_queues[stage].append(transfer.wave_id)
            self.stage_ready_at[transfer.wave_id, stage] = self.now
            self._log(kind, microbatch_id=transfer.wave_id, boundary=transfer.boundary,
                      bytes_per_rank=transfer.bytes_per_rank)
        elif kind == "FEEDBACK_COMPLETE":
            self._feedback_complete(payload)
        elif kind == "SWAP_COMPLETE":
            details = self.swap.complete(payload, self.now)
            self.swap_transfer = None
            request = self.requests[details["request_id"]]
            duration = payload.end_ms - payload.start_ms
            request.swap_ms += duration
            if details["direction"] == "D2H":
                request.swap_out_count += 1
                request.swap_out_bytes += details["bytes"]
            else:
                request.swap_in_count += 1
                request.swap_in_bytes += details["bytes"]
            if request.status not in TERMINAL:
                request.status = ("SWAPPED" if details["direction"] == "D2H"
                                  else self._continuation_status(request))
            self._log(kind, **details)
        else:
            request = self.requests[str(payload)]
            if request.status in TERMINAL:
                return
            if kind == "REQUEST_ARRIVAL":
                request.status = "WAITING"
                self._log(kind, request_id=request.request_id)
            else:
                self._finish(request, "TIMED_OUT", "REQUEST_TIMEOUT")

    def _has_references(self) -> bool:
        return bool(self.waves or self.swap_transfer is not None)

    def run(self) -> SimulationResult:
        while self.queue:
            when = self.queue[0][0]
            if when > self.cutoff_ms:
                self.now, self.stop_reason = self.cutoff_ms, self.cutoff_reason
                break
            self.now = when
            while self.queue and self.queue[0][0] == when:
                if not self._has_references() and all(r.status in TERMINAL for r in self.requests.values()):
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
                if self._has_references() or any(r.status not in TERMINAL for r in self.requests.values()):
                    self.stop_reason = self.cutoff_reason
                break
            self._schedule()
            if self.stop_reason is not None:
                break
            if not self._has_references() and all(r.status in TERMINAL for r in self.requests.values()):
                break
        if any(r.status not in TERMINAL for r in self.requests.values()):
            self.stop_reason = self.stop_reason or "MEMORY_NO_PROGRESS"
        for task in self.stage_busy:
            if task is None:
                continue
            elapsed = max(0.0, min(self.now, task.end_ms) - task.start_ms)
            self.stage_busy_ms[task.stage] += elapsed
            self.compute_busy_ms += elapsed
            wave = self.waves[task.wave_id]
            if wave.phase == "mixed":
                self.mixed_compute_busy_ms += elapsed
            if wave.phase == "recompute":
                self.recompute_execution_ms += elapsed
                for name in wave.requests:
                    self.requests[name].recompute_ms += elapsed
            self._log("STAGE_INTERRUPTED", stage=task.stage, microbatch_id=task.wave_id,
                      elapsed_ms=elapsed, error_code=self.stop_reason,
                      **self._member_metadata(wave))
        for transfer in self.transfers.values():
            self.link_busy_ms[transfer.boundary] += max(0.0, min(self.now, transfer.end_ms) - transfer.start_ms)
        # Waiting that ends at a budget boundary is still elapsed resource wait.
        # A wave enters this queue only after its incoming transfer completes;
        # link-queued/in-transit waves must not accrue stage-queue time yet.
        for stage, queued in enumerate(self.stage_queues):
            for wave_id in queued:
                self.stage_queue_wait_ms[stage] += max(0.0, self.now - self.stage_ready_at[wave_id, stage])
        if self.swap_transfer is not None:
            self.requests[self.swap_transfer.request_id].swap_ms += max(
                0.0, min(self.now, self.swap_transfer.end_ms) - self.swap_transfer.start_ms)
        for request in self.requests.values():
            self._end_wait(request)
            if request.status not in TERMINAL:
                request.status, request.error_code = "UNFINISHED", self.stop_reason
        result = report_simulation(self)
        for row in result.requests:
            row["stage_computed_tokens"] = list(self.stage_progress[row["request_id"]])
        result.result["compute_busy_ms_semantics"] = "aggregate_stage_time"
        result.result["pipeline"] = {
            "stages": self.pp, "tensor_parallel_size": self.exp.strategy.tensor_parallel_size,
            "microbatches_admitted": self.next_wave_id, "peak_inflight_microbatches": self.peak_waves,
            "inflight_microbatches": len(self.waves),
            "stage_busy_ms": list(self.stage_busy_ms),
            "stage_queue_wait_ms": list(self.stage_queue_wait_ms),
            "stage_utilization": [busy / self.now if self.now else None for busy in self.stage_busy_ms],
            "idle_fraction": (1 - sum(self.stage_busy_ms) / (self.pp * self.now) if self.now else None),
            "idle_fraction_scope": "Includes arrival idle time, fill/drain, feedback and pressure stalls",
            "links": [{"boundary": i, "busy_ms": value, "completed_bytes": self.link_bytes[i]}
                      for i, value in enumerate(self.link_busy_ms)],
            "feedback_latency_us": self.exp.strategy.pipeline_feedback_latency_us,
            "p2p_enabled": self.exp.strategy.pipeline_p2p,
            "activation_buffer_policy": "persistent bound per stage/rank",
            "recovery_policy": "quiesce all waves before request eviction or DMA",
        }
        return result
