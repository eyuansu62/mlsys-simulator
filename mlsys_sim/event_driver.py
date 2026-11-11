"""Pinned internal adapter for causal, dynamically arriving workloads.

This module deliberately reuses the ordinary engines and replica router.  It
does not run a second scheduler.  Keep its equivalence tests with ``simulate``
when changing the private engine interfaces it adapts.
"""

import heapq
import math
from dataclasses import replace
from typing import Any

from .schema import Experiment, RequestSpec, SimulationError
from .state import SimulationResult, TERMINAL


class EventDriver:
    """Advance an initially empty resource domain by one committed event.

    A caller drains all events at a timestamp, admits causal arrivals, then
    calls :meth:`schedule_ready`.  Idle drivers remain open, retaining cache.
    """

    def __init__(self, experiment: Experiment):
        if experiment.workload or experiment.execution.warmup:
            raise SimulationError("INVALID_CONFIG", "EventDriver needs empty workload and warmup")
        from .engine import _Simulation
        from .pipeline import PipelineSimulation
        from .replicas import MultiReplicaSimulation

        self.exp = experiment
        self._multi = experiment.strategy.replicas > 1
        self._driver: Any = (MultiReplicaSimulation(experiment) if self._multi else
                            PipelineSimulation(experiment) if experiment.strategy.pipeline_parallel_size > 1
                            else _Simulation(experiment))
        domain: Any = self._driver
        self.local_engines: tuple[Any, ...] = (tuple(domain.engines) if self._multi
                                               else (domain,))
        self.first_compute_start_ms: dict[str, float] = {}
        self._result: SimulationResult | None = None

    @property
    def now(self) -> float:
        return float(self._driver.now)

    @property
    def requests(self) -> dict:
        return self._driver.requests

    @property
    def memory(self) -> Any:
        return self._driver.memory

    @property
    def events(self) -> list[dict]:
        return self._driver.events

    @property
    def handled_events(self) -> int:
        return int(self._driver.handled_events)

    @property
    def cutoff_ms(self) -> float:
        return float(self._driver.cutoff_ms)

    @property
    def cutoff_reason(self) -> str:
        return str(self._driver.cutoff_reason)

    @property
    def stop_reason(self) -> str | None:
        return self._driver.stop_reason

    @property
    def stopped(self) -> bool:
        return self._result is not None or self.stop_reason is not None

    def has_inflight(self) -> bool:
        return any(engine.has_inflight() for engine in self.local_engines)

    def has_unfinished(self) -> bool:
        return any(request.status not in TERMINAL for request in self.requests.values())

    def _observe_starts(self) -> None:
        for engine in self.local_engines:
            if hasattr(engine, "stage_busy"):
                task = engine.stage_busy[0]
                if task is not None:
                    for name in engine.waves[task.wave_id].requests:
                        self.first_compute_start_ms.setdefault(name, task.start_ms)
            elif engine.busy is not None:
                for request in engine.busy.requests:
                    self.first_compute_start_ms.setdefault(request.request_id, engine.busy.start_ms)

    def _raw_event(self) -> tuple[float, int, int, int] | None:
        if self._multi:
            return self._driver._next_event()
        engine = self._driver
        if not engine.has_inflight() and not self.has_unfinished():
            engine.queue.clear()  # only stale timeouts remain in an empty driver
        if not engine.queue:
            return None
        when, priority, sequence, _, _ = engine.queue[0]
        return when, priority, 0, sequence

    def next_event(self) -> tuple[float, int, int] | None:
        if self.stopped:
            return None
        event = self._raw_event()
        if event is None:
            return None
        when, priority, _, sequence = event
        if when > self.cutoff_ms:
            return self.cutoff_ms, 3, -1
        return when, priority, sequence

    def _set_clock(self, when: float) -> None:
        self._driver.now = when
        for engine in self.local_engines:
            if engine._final_result is None:
                engine.now = when

    def set_time(self, when: float) -> None:
        if (isinstance(when, bool) or not isinstance(when, (int, float))
                or not math.isfinite(when) or when < self.now):
            raise SimulationError("INVALID_DRIVER_STATE", "Driver clock must be finite and monotonic")
        if self.stopped:
            if when != self.now:
                raise SimulationError("INVALID_DRIVER_STATE", "A stopped driver cannot advance")
            return
        event = self.next_event()
        if event is not None and event[2] != -1 and event[0] < when:
            raise SimulationError("INVALID_DRIVER_STATE", "Cannot skip an uncommitted driver event")
        if when > self.cutoff_ms:
            self._set_clock(self.cutoff_ms)
            reason = self.cutoff_reason if self.has_inflight() or self.has_unfinished() else None
            self.finalize(reason=reason)
            return
        self._set_clock(float(when))

    def accept_request(self, spec: RequestSpec) -> None:
        if self.stopped:
            raise SimulationError("INVALID_DRIVER_STATE", "A stopped driver cannot accept requests")
        if spec.arrival_ms != self.now or spec.request_id in self.requests:
            raise SimulationError("INVALID_DRIVER_STATE", "Arrival must be new and match the driver clock")
        from .config import validate_experiment
        validate_experiment(replace(self.exp, workload=(spec,)))
        if self.handled_events >= self.exp.execution.max_events:
            self.finalize(reason="SIMULATION_EVENT_LIMIT")
            raise SimulationError("SIMULATION_EVENT_LIMIT", "Driver admission event budget exhausted")
        if self._multi:
            if not self._driver._route(spec):
                self.finalize(reason=self.stop_reason)
                raise SimulationError("NO_AVAILABLE_REPLICA", "No replica can accept the request")
        else:
            self._driver.accept_request(spec)

    def advance_one(self) -> None:
        event = self.next_event()
        if event is None:
            raise SimulationError("INVALID_DRIVER_STATE", "Driver has no next event")
        when, _, sequence = event
        self._set_clock(when)
        if sequence == -1:
            self.finalize(reason=self.cutoff_reason)
            return
        if self.handled_events >= self.exp.execution.max_events:
            self.finalize(reason="SIMULATION_EVENT_LIMIT")
            return
        raw = self._raw_event()
        assert raw is not None
        index = raw[2]
        engine = self.local_engines[index]
        _, _, _, kind, payload = heapq.heappop(engine.queue)
        engine.handle_event(kind, payload)
        if self._multi:
            self._driver.handled_events += 1
            self._driver._collect_events(index)
            self._driver._observe_resources()
            if engine.stop_reason is not None:
                self._driver._deactivate(index)
        self._observe_starts()

    def schedule_ready(self) -> None:
        if self.stopped:
            return
        if self.now >= self.cutoff_ms:
            if self.has_inflight() or self.has_unfinished():
                self.finalize(reason=self.cutoff_reason)
            return
        if self._multi:
            self._driver._schedule()
            if not self._driver._available():
                self.finalize(reason="NO_AVAILABLE_REPLICA")
        else:
            self._driver.schedule_ready()
        self._observe_starts()

    def finalize(self, now_ms: float | None = None,
                 reason: str | None = None) -> SimulationResult:
        if self._result is not None:
            return self._result
        if now_ms is not None:
            # Finalization may deliberately interrupt before a future event.
            self.set_time(now_ms)
            if self._result is not None:
                return self._result
        if self._multi:
            self._driver.stop_reason = self._driver.stop_reason or reason
            self._result = self._driver._finalize()
        else:
            self._result = self._driver.finalize(stop_reason=reason)
        return self._result
