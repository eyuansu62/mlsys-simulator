"""One causal event timeline over independently owned model replicas.

Routing observes only arrived request state and resident cache identities.
Every request remains on its selected replica through compute, feedback and
host recovery. Replica resources and DMA lanes are independent in this model.
"""

from dataclasses import replace
import heapq
from types import SimpleNamespace
from typing import Any

from .config import validate_experiment
from .layout import replica_experiment
from .reporting import report_simulation
from .schema import Experiment, RequestSpec, SimulationError
from .state import RequestState, SimulationResult, TERMINAL


_PREFIX_SUM_FIELDS = (
    "capacity_blocks", "lookup_requests", "hit_requests", "miss_requests",
    "bypassed_requests", "hit_tokens", "published_blocks", "deduplicated_blocks",
    "evicted_blocks", "abandoned_lookups", "abandoned_hit_tokens", "cached_blocks",
    "pinned_blocks", "evictable_blocks", "shared_blocks", "attached_block_references",
    "private_blocks",
)


class _ReplicaMemory:
    def __init__(self, simulation: "MultiReplicaSimulation"):
        self.simulation = simulation

    @property
    def allocations(self) -> dict[str, int]:
        return {name: tokens for engine in self.simulation.engines
                for name, tokens in engine.memory.allocations.items()}

    def snapshot(self) -> list[dict]:
        return [{**rank, "replica_index": index}
                for index, engine in enumerate(self.simulation.engines)
                for rank in engine.memory.snapshot()]

    def prefix_snapshot(self) -> dict:
        snapshots = [engine.memory.prefix_snapshot() for engine in self.simulation.engines]
        aggregate = {key: sum(snapshot.get(key, 0) for snapshot in snapshots)
                     for key in _PREFIX_SUM_FIELDS}
        eligible = aggregate["hit_requests"] + aggregate["miss_requests"]
        return {
            **aggregate,
            "enabled": self.simulation.exp.strategy.prefix_cache,
            "block_tokens": self.simulation.exp.strategy.memory.kv_block_tokens,
            "peak_cached_blocks": self.simulation.peak_cached_blocks,
            "sum_replica_peak_cached_blocks": sum(snapshot["peak_cached_blocks"] for snapshot in snapshots),
            "peak_scope": "observed_global_timeline",
            "hit_rate": aggregate["hit_requests"] / eligible if eligible else None,
            "policy": "replica_local_full_prompt_blocks_sha256_ancestry_lru",
            "identity": snapshots[0]["identity"],
            "entries": [{**entry, "replica_index": index}
                        for index, snapshot in enumerate(snapshots) for entry in snapshot["entries"]],
        }


class _ReplicaSwap:
    def __init__(self, simulation: "MultiReplicaSimulation"):
        self.simulation = simulation

    def state(self, request_id: str) -> str | None:
        owner = self.simulation.owners.get(request_id)
        if owner is None:
            return None
        manager = self.simulation.engines[owner].swap
        return manager.state(request_id) if manager is not None else None

    def snapshot(self, now_ms: float) -> dict:
        snapshots = [(index, engine.swap.snapshot(now_ms))
                     for index, engine in enumerate(self.simulation.engines) if engine.swap is not None]
        totals = {key: sum(snapshot[key] for _, snapshot in snapshots)
                  for key in ("host_capacity_bytes", "host_reserved_bytes", "host_free_bytes",
                              "dma_busy_ms", "completed_dma_busy_ms", "cancelled_transfer_bytes")}
        for key in ("completed_bytes_by_direction", "completed_transfers_by_direction",
                    "pending_transferred_bytes_by_direction"):
            totals[key] = {direction: sum(snapshot[key][direction] for _, snapshot in snapshots)
                           for direction in ("D2H", "H2D")}
        return {
            **totals,
            "host_peak_reserved_bytes": self.simulation.peak_host_reserved_bytes,
            "sum_replica_host_peak_reserved_bytes": sum(snapshot["host_peak_reserved_bytes"]
                                                       for _, snapshot in snapshots),
            "peak_scope": "observed_global_timeline",
            "busy_until_ms": max((snapshot["busy_until_ms"] for _, snapshot in snapshots), default=0.0),
            "pending_transfers": [{**transfer, "replica_index": index}
                                  for index, snapshot in snapshots for transfer in snapshot["pending_transfers"]],
            "entries": [{**entry, "replica_index": index}
                        for index, snapshot in snapshots for entry in snapshot["entries"]],
            "model": "independent_replica_serialized_bidirectional_dma",
            "capacity_scope": "sum_of_independent_per_replica_host_pools",
            "dma_busy_ms_semantics": "aggregate_replica_dma_time",
            "source": "analytical_assumption",
            "block_aligned_kv_bytes": True,
            "replicas": [{"replica_index": index, **snapshot} for index, snapshot in snapshots],
        }


class MultiReplicaSimulation:
    """Route arrivals once, then advance all local DES instances causally."""

    def __init__(self, experiment: Experiment):
        validate_experiment(experiment)
        if experiment.strategy.replicas < 2:
            raise SimulationError("INVALID_CONFIG", "Multi-replica simulation requires at least two replicas")
        from .engine import _Simulation
        from .pipeline import PipelineSimulation

        self.exp = experiment
        self.engines: list[Any] = []
        for index in range(experiment.strategy.replicas):
            local = replica_experiment(experiment, index)
            local = replace(local, workload=(), execution=replace(local.execution, warmup=()))
            engine = (PipelineSimulation(local) if local.strategy.pipeline_parallel_size > 1
                      else _Simulation(local))
            self.engines.append(engine)
        self.now = 0.0
        self.handled_events = 0
        self.stop_reason: str | None = None
        self.cutoff_ms = self.engines[0].cutoff_ms
        self.cutoff_reason = self.engines[0].cutoff_reason
        self.requests = {spec.request_id: RequestState(spec.request_id, spec.arrival_ms,
                                                      spec.input_tokens, spec.priority)
                         for spec in experiment.workload}
        self.arrivals: list[tuple[float, str, RequestSpec]] = []
        window = experiment.evaluation.measurement_window_ms
        for spec in experiment.workload:
            if window is not None and spec.arrival_ms >= window:
                request = self.requests[spec.request_id]
                request.status = "NOT_ADMITTED"
                request.error_code = "OUTSIDE_MEASUREMENT_WINDOW"
            else:
                heapq.heappush(self.arrivals, (spec.arrival_ms, spec.request_id, spec))
        self.owners: dict[str, int] = {}
        self.assignments: list[dict] = []
        self.request_count_by_replica = [0] * len(self.engines)
        self.peak_unfinished_requests_by_replica = [0] * len(self.engines)
        self.round_robin_cursor = 0
        self.local_results: list[SimulationResult | None] = [None] * len(self.engines)
        self.unavailable_replicas: set[int] = set()
        self.events: list[dict] = []
        self.event_cursors = [0] * len(self.engines)
        self.peak_cached_blocks = 0
        self.peak_host_reserved_bytes = 0
        self.memory = _ReplicaMemory(self)
        self.swap = _ReplicaSwap(self) if experiment.strategy.preemption == "swap" else None
        self.performance = SimpleNamespace(provenance=None, profile=None)
        self.compute_busy_ms = 0.0
        self.recompute_execution_ms = 0.0
        self.mixed_batch_count = 0
        self.mixed_batch_completed_count = 0
        self.mixed_compute_busy_ms = 0.0
        self.limitations = [
            "independent model replicas with replica-local KV, prefix cache, host pools and DMA lanes",
            "routing uses arrived unfinished requests, in-flight references and resident prefix identities",
            "requests remain on one replica; cross-replica migration and shared fabric contention are not modeled",
            "routing, hashing and dispatch add no separately measured CPU or network latency",
        ]
        self._result: SimulationResult | None = None

    def _available(self) -> list[int]:
        return [index for index, result in enumerate(self.local_results) if result is None]

    @staticmethod
    def _unfinished(engine: Any) -> int:
        return sum(request.status not in TERMINAL for request in engine.requests.values())

    @staticmethod
    def _inflight(engine: Any) -> int:
        if hasattr(engine, "waves"):
            return len(engine.waves) + int(engine.swap_transfer is not None)
        return int(engine.busy is not None) + int(engine.swap_busy is not None)

    @staticmethod
    def _used_blocks(engine: Any) -> int:
        return engine.memory.snapshot()[0]["used_blocks"]

    def _log(self, kind: str, **details: Any) -> None:
        if self.exp.execution.detailed_events:
            self.events.append({"timestamp_ms": self.now, "event": kind, **details})

    def _collect_events(self, index: int) -> None:
        engine = self.engines[index]
        start = self.event_cursors[index]
        self.events.extend({**event, "replica_index": index} for event in engine.events[start:])
        self.event_cursors[index] = len(engine.events)

    def _observe_resources(self) -> None:
        self.peak_cached_blocks = max(self.peak_cached_blocks,
            sum(engine.memory.prefix_snapshot()["cached_blocks"] for engine in self.engines))
        self.peak_host_reserved_bytes = max(self.peak_host_reserved_bytes,
            sum(engine.swap.host_reserved_bytes for engine in self.engines if engine.swap is not None))
        for index, engine in enumerate(self.engines):
            self.peak_unfinished_requests_by_replica[index] = max(
                self.peak_unfinished_requests_by_replica[index], self._unfinished(engine))

    def _route(self, spec: RequestSpec) -> bool:
        available = self._available()
        if not available:
            self.stop_reason = "NO_AVAILABLE_REPLICA"
            return False
        loads = {index: self._unfinished(self.engines[index]) for index in available}
        cached = {index: 0 for index in available}
        policy = self.exp.strategy.replica_routing
        if policy == "round_robin":
            count = len(self.engines)
            index = next((self.round_robin_cursor + offset) % count for offset in range(count)
                         if (self.round_robin_cursor + offset) % count in available)
            self.round_robin_cursor = (index + 1) % count
        else:
            candidates = available
            if policy == "prefix_aware":
                minimum = min(loads.values())
                candidates = [candidate for candidate in available if loads[candidate] == minimum]
                cached = {candidate: self.engines[candidate].memory.peek_prefix_tokens(spec)
                          for candidate in candidates}
            index = min(candidates, key=lambda candidate: (
                loads[candidate], -cached.get(candidate, 0), self._inflight(self.engines[candidate]),
                self._used_blocks(self.engines[candidate]), candidate))
        decision = {"request_id": spec.request_id, "replica_index": index,
                    "arrival_ms": self.now, "unfinished_requests_before": loads[index],
                    "cached_prefix_tokens": cached.get(index, 0)}
        self.assignments.append(decision)
        self._log("REQUEST_ROUTED", **decision, policy=policy)
        self.owners[spec.request_id] = index
        self.request_count_by_replica[index] += 1
        engine = self.engines[index]
        engine.now = self.now
        engine.accept_request(spec)
        self.handled_events += 1
        self.requests[spec.request_id] = engine.requests[spec.request_id]
        self._collect_events(index)
        self._observe_resources()
        return True

    def _prune_idle_queues(self) -> None:
        for index in self._available():
            engine = self.engines[index]
            if not engine.has_inflight() and not self._unfinished(engine):
                # Dynamic engines have no future arrivals; only stale timeout
                # events remain once every assigned request has terminated.
                engine.queue.clear()

    def _next_event(self) -> tuple[float, int, int, int] | None:
        self._prune_idle_queues()
        candidates = []
        if self.arrivals:
            candidates.append((self.arrivals[0][0], 2, -1, 0))
        for index in self._available():
            queue = self.engines[index].queue
            if queue:
                when, priority, sequence, _, _ = queue[0]
                candidates.append((when, priority, index, sequence))
        return min(candidates) if candidates else None

    def _deactivate(self, index: int) -> None:
        engine = self.engines[index]
        self.unavailable_replicas.add(index)
        self.local_results[index] = engine.finalize(now_ms=self.now)
        engine.queue.clear()
        self._collect_events(index)
        self._log("REPLICA_STOPPED", replica_index=index, error_code=engine.stop_reason)

    def _schedule(self) -> None:
        for index in self._available():
            engine = self.engines[index]
            engine.now = self.now
            engine.schedule_ready()
            self._collect_events(index)
            self._observe_resources()
            if engine.stop_reason is not None:
                self._deactivate(index)

    def _all_done(self) -> bool:
        return not self.arrivals and all(not engine.has_inflight() and not self._unfinished(engine)
                                         for index, engine in enumerate(self.engines)
                                         if self.local_results[index] is None)

    def run(self) -> SimulationResult:
        if self._result is not None:
            return self._result
        while True:
            event = self._next_event()
            if event is None:
                break
            when = event[0]
            if when > self.cutoff_ms:
                self.now, self.stop_reason = self.cutoff_ms, self.cutoff_reason
                break
            self.now = when
            while (event := self._next_event()) is not None and event[0] == when:
                if self.handled_events >= self.exp.execution.max_events:
                    self.stop_reason = "SIMULATION_EVENT_LIMIT"
                    break
                index = event[2]
                if index == -1:
                    _, _, spec = heapq.heappop(self.arrivals)
                    if not self._route(spec):
                        break
                else:
                    engine = self.engines[index]
                    _, _, _, kind, payload = heapq.heappop(engine.queue)
                    engine.now = self.now
                    engine.handle_event(kind, payload)
                    self.handled_events += 1
                    self._collect_events(index)
                    self._observe_resources()
                    if engine.stop_reason is not None:
                        self._deactivate(index)
            if self.stop_reason is not None:
                break
            if self.now >= self.cutoff_ms:
                if not self._all_done():
                    self.stop_reason = self.cutoff_reason
                break
            self._schedule()
            if not self._available() and self.arrivals:
                self.stop_reason = "NO_AVAILABLE_REPLICA"
                break
            if self._all_done():
                break
        return self._finalize()

    def _finalize(self) -> SimulationResult:
        for index, engine in enumerate(self.engines):
            if self.local_results[index] is None:
                reason = self.stop_reason if self._unfinished(engine) or engine.has_inflight() else None
                self.local_results[index] = engine.finalize(now_ms=self.now, stop_reason=reason)
                self._collect_events(index)
        unresolved = [result for result in self.local_results
                      if result is not None and result.result["status"] != "COMPLETED"]
        if unresolved and self.stop_reason is None:
            self.stop_reason = "REPLICA_UNRESOLVED"
        for request in self.requests.values():
            if request.status not in TERMINAL:
                self.stop_reason = self.stop_reason or "MEMORY_NO_PROGRESS"
                request.status = "UNFINISHED"
                request.error_code = self.stop_reason
        self.compute_busy_ms = sum(engine.compute_busy_ms for engine in self.engines)
        self.recompute_execution_ms = sum(engine.recompute_execution_ms for engine in self.engines)
        self.mixed_batch_count = sum(engine.mixed_batch_count for engine in self.engines)
        self.mixed_batch_completed_count = sum(engine.mixed_batch_completed_count for engine in self.engines)
        self.mixed_compute_busy_ms = sum(engine.mixed_compute_busy_ms for engine in self.engines)
        self._observe_resources()
        result = report_simulation(self)
        for row in result.requests:
            owner = self.owners.get(row["request_id"])
            row["replica_index"] = owner
            if self.exp.strategy.pipeline_parallel_size > 1:
                row["stage_computed_tokens"] = (list(self.engines[owner].stage_progress[row["request_id"]])
                                                 if owner is not None else None)
        local = [report for report in self.local_results if report is not None]
        result.result["replicas"] = [
            {"replica_index": index, "device_ids": [device.id for device in engine.exp.infra.devices],
             "result": local[index].result}
            for index, engine in enumerate(self.engines)]
        result.result["routing"] = {
            "policy": self.exp.strategy.replica_routing,
            "assigned_requests": len(self.assignments), "assignments": list(self.assignments),
            "request_count_by_replica": list(self.request_count_by_replica),
            "peak_unfinished_requests_by_replica": list(self.peak_unfinished_requests_by_replica),
            "unavailable_replicas": sorted(self.unavailable_replicas),
            "unresolved_replicas": [index for index, report in enumerate(local)
                                     if report.result["status"] != "COMPLETED"],
            "ownership_policy": "sticky_request_no_cross_replica_migration",
            "load_observation": "arrived_unfinished_requests_then_inflight_references_and_used_blocks",
            "prefix_affinity_scope": "minimum_unfinished_load_candidates_only",
            "dispatch_latency_ms": 0.0,
        }
        semantics = "aggregate_replica_stage_time"
        result.result["compute_busy_ms_semantics"] = semantics
        result.result["batching"]["compute_busy_ms_semantics"] = semantics
        if self.exp.calibration.backend == "profile":
            sources = {report.result["calibration_source"] for report in local}
            versions = {report.result["profile_version"] for report in local}
            result.result["calibration_source"] = (next(iter(sources)) if len(sources) == 1
                                                   else "mixed_replica_profiles")
            result.result["profile_version"] = next(iter(versions)) if len(versions) == 1 else "per_replica"
            result.result["profile_provenance"] = {
                "replicas": [{"replica_index": index, **(report.result["profile_provenance"] or {})}
                             for index, report in enumerate(local)]}
            result.result["limitations"][0] = "profile-backed CPU model; hardware holdout validation is separate"
        self._result = result
        return result
