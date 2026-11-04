"""Dynamic event-driver hooks preserve local scheduling and pinned lifetimes."""

from copy import deepcopy
from dataclasses import replace
import heapq
from pathlib import Path
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment,
                       InfraSpec, MemoryConfig, RequestSpec, SimulationError,
                       StrategySpec, load_experiment)
from mlsys_sim.engine import _Simulation
from mlsys_sim.pipeline import PipelineSimulation
from mlsys_sim.state import TERMINAL


ROOT = Path(__file__).resolve().parents[1]


def local(exp):
    return PipelineSimulation(exp) if exp.strategy.pipeline_parallel_size > 1 else _Simulation(exp)


def drive(exp):
    """One-replica coordinator that exposes requests only on their arrival."""
    engine = local(replace(exp, workload=()))
    arrivals = [(request.arrival_ms, 2, i, request) for i, request in enumerate(
        sorted(exp.workload, key=lambda request: (request.arrival_ms, request.request_id)))]
    heapq.heapify(arrivals)
    while arrivals or engine.queue:
        if (not arrivals and not engine.has_inflight()
                and all(request.status in TERMINAL for request in engine.requests.values())):
            break
        when = min(arrivals[0][0] if arrivals else float("inf"),
                   engine.queue[0][0] if engine.queue else float("inf"))
        if when > engine.cutoff_ms:
            return engine, engine.finalize(engine.cutoff_ms, engine.cutoff_reason)
        engine.now = when
        while True:
            arrival = arrivals[0] if arrivals and arrivals[0][0] == when else None
            queued = engine.queue[0] if engine.queue and engine.queue[0][0] == when else None
            if arrival is None and queued is None:
                break
            if engine.handled_events >= exp.execution.max_events:
                return engine, engine.finalize(stop_reason="SIMULATION_EVENT_LIMIT")
            if queued is not None and (arrival is None or queued[1] < arrival[1]):
                _, _, _, kind, payload = heapq.heappop(engine.queue)
                engine.handle_event(kind, payload)
            else:
                engine.accept_request(heapq.heappop(arrivals)[3])
        if engine.now >= engine.cutoff_ms:
            reason = (engine.cutoff_reason if engine.has_inflight() or arrivals
                      or any(request.status not in TERMINAL for request in engine.requests.values()) else None)
            return engine, engine.finalize(stop_reason=reason)
        engine.schedule_ready()
        if engine.stop_reason is not None:
            break
    return engine, engine.finalize()


def basic(pp=1, **strategy):
    values = dict(max_batch_size=2, max_batch_tokens=8, pipeline_parallel_size=pp,
                  pipeline_microbatch_size=2, pipeline_p2p=False,
                  memory=MemoryConfig(kv_block_tokens=4))
    values.update(strategy)
    return Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(pp))),
                      strategy=StrategySpec(**values),
                      calibration=CalibrationSpec(backend="synthetic", prefill_ms=2,
                                                  decode_ms=1, mixed_ms=3, recompute_ms=2))


class ReplicaDriverTests(unittest.TestCase):
    def test_dynamic_driver_matches_run_for_completion_timeout_and_arrival_ties(self):
        for pp in (1, 2):
            with self.subTest(pp=pp):
                exp = replace(basic(pp), workload=(RequestSpec("a", 0, 4, 1, timeout_ms=2),
                                                   RequestSpec("b", 2, 4, 2)))
                expected = local(exp).run()
                _, actual = drive(exp)
                self.assertEqual(actual, expected)
                self.assertEqual(actual.requests[0]["status"], "COMPLETED")

    def test_dynamic_driver_preserves_fifo_mixed_and_chunked_local_batches(self):
        for pp in (1, 2):
            for scheduler in ("fifo", "continuous_batching"):
                with self.subTest(pp=pp, scheduler=scheduler):
                    exp = replace(basic(pp, scheduler=scheduler, mixed_batching=True,
                                         chunked_prefill=True, prefill_chunk_tokens=4,
                                         max_batch_tokens=5),
                                  workload=(RequestSpec("a", 0, 4, 3), RequestSpec("b", 1, 8, 2)))
                    self.assertEqual(drive(exp)[1], local(exp).run())

    def test_existing_swap_and_prefix_examples_match_normal_local_execution(self):
        for example in ("v1_1/cpu_swap.json", "v1_1/pp_cpu_swap.json",
                        "v1_3/pp_mixed_cache_swap.json"):
            with self.subTest(example=example):
                exp = load_experiment(ROOT / "examples" / example)
                expected = local(exp).run()
                _, actual = drive(exp)
                self.assertEqual(actual, expected)
                self.assertGreater(actual.result["swap"]["completed_transfers_by_direction"]["D2H"], 0)

    def test_dynamic_driver_matches_time_and_event_budget_partial_accounting(self):
        for pp in (1, 2):
            for execution in (ExecutionSpec(max_simulated_ms=1.5), ExecutionSpec(max_events=2)):
                with self.subTest(pp=pp, execution=execution):
                    exp = replace(basic(pp), execution=execution,
                        workload=(RequestSpec("a", 0, 4, 3), RequestSpec("b", .5, 4, 2)))
                    expected = local(exp).run()
                    _, actual = drive(exp)
                    self.assertEqual(actual, expected)
                    self.assertGreater(actual.result["compute_busy_ms"], 0)
                    self.assertTrue(all(row["output_tokens_actual"] == 0 for row in actual.requests))

    def test_no_future_request_oracle_or_identity_is_registered_before_admission(self):
        for pp in (1, 2):
            with self.subTest(pp=pp):
                engine = local(basic(pp, prefix_cache=True))
                self.assertEqual(engine.requests, {})
                self.assertEqual(engine._output_targets, {})
                self.assertEqual(engine.memory._prefix_specs, {})
                spec = RequestSpec("routed", 0, 4, 1, prompt_token_ids=(0, 1, 2, 3))
                engine.accept_request(spec)
                self.assertEqual(set(engine.requests), {"routed"})
                self.assertEqual(engine._output_targets, {"routed": 1})
                self.assertEqual(set(engine.memory._prefix_specs), {"routed"})
                self.assertEqual(engine.handled_events, 1)
                self.assertEqual(engine.queue, [])

    def test_idle_replica_accepts_later_arrival_without_ending_its_lifecycle(self):
        for pp in (1, 2):
            with self.subTest(pp=pp):
                engine = local(basic(pp))
                engine.schedule_ready()
                self.assertFalse(engine.has_inflight())
                self.assertIsNone(engine.stop_reason)
                engine.now = 20
                engine.accept_request(RequestSpec("late", 20, 4, 1))
                engine.schedule_ready()
                self.assertTrue(engine.has_inflight())
                while engine.queue:
                    when, _, _, kind, payload = heapq.heappop(engine.queue)
                    engine.now = when
                    engine.handle_event(kind, payload)
                    engine.schedule_ready()
                result = engine.finalize()
                self.assertEqual(result.requests[0]["token_times_ms"], [22])
                self.assertEqual(result.result["simulated_ms"], 22)

    def test_duplicate_and_mistimed_admission_fail_without_registering_partial_state(self):
        for pp in (1, 2):
            with self.subTest(pp=pp):
                engine = local(basic(pp))
                with self.assertRaises(SimulationError) as caught:
                    engine.accept_request(RequestSpec("future", 1, 4, 1))
                self.assertEqual(caught.exception.code, "INVALID_DRIVER_STATE")
                self.assertEqual(engine.requests, {})
                self.assertEqual(engine.memory._prefix_specs, {})
                engine.accept_request(RequestSpec("a", 0, 4, 1, timeout_ms=4))
                before = deepcopy((engine.queue, engine.handled_events, engine._output_targets))
                with self.assertRaises(SimulationError):
                    engine.accept_request(RequestSpec("a", 0, 8, 99))
                self.assertEqual((engine.queue, engine.handled_events, engine._output_targets), before)

    def test_partial_finalization_is_idempotent_and_keeps_kv_reserved(self):
        for pp in (1, 2):
            with self.subTest(pp=pp):
                engine = local(basic(pp))
                engine.accept_request(RequestSpec("a", 0, 8, 2))
                engine.schedule_ready()
                first = engine.finalize(.5, "SIMULATION_TIME_LIMIT")
                before = deepcopy((engine.compute_busy_ms, engine.memory.snapshot(), engine.events))
                self.assertEqual(engine.compute_busy_ms, .5)
                self.assertTrue(engine.has_inflight())
                self.assertGreater(engine.memory.snapshot()[0]["used_blocks"], 0)
                self.assertEqual(first.requests[0]["num_computed_tokens"], 0)
                self.assertIs(engine.finalize(10, "different-reason"), first)
                self.assertIs(engine.run(), first)
                engine.schedule_ready()
                self.assertEqual((engine.compute_busy_ms, engine.memory.snapshot(), engine.events), before)
                with self.assertRaises(SimulationError):
                    engine.accept_request(RequestSpec("b", .5, 4, 1))
                with self.assertRaises(SimulationError):
                    engine.handle_event("REQUEST_TIMEOUT", "a")

    def test_finalization_time_is_monotonic_and_preserves_first_stop_reason(self):
        for pp in (1, 2):
            with self.subTest(pp=pp):
                engine = local(basic(pp))
                engine.now = 2
                for invalid in (1, float("nan"), float("inf"), True):
                    with self.subTest(invalid=invalid), self.assertRaises(SimulationError):
                        engine.finalize(invalid)
                    self.assertIsNone(engine._final_result)
                engine.stop_reason = "HOST_SWAP_CAPACITY_EXCEEDED"
                result = engine.finalize(3, "SIMULATION_EVENT_LIMIT")
                self.assertEqual(result.result["error_code"], "HOST_SWAP_CAPACITY_EXCEEDED")

    def test_pp_timeout_keeps_wave_and_prefix_ownership_until_feedback(self):
        exp = replace(basic(2, prefix_cache=True, pipeline_feedback_latency_us=1000),
                      workload=(RequestSpec("a", 0, 4, 3, timeout_ms=2.5,
                                            prompt_token_ids=(0, 1, 2, 3)),))
        engine = local(replace(exp, workload=()))
        engine.accept_request(exp.workload[0])
        engine.schedule_ready()
        while engine.queue and engine.queue[0][0] <= 2.5:
            when, _, _, kind, payload = heapq.heappop(engine.queue)
            engine.now = when
            engine.handle_event(kind, payload)
            engine.schedule_ready()
        self.assertEqual(engine.requests["a"].status, "TIMED_OUT")
        self.assertTrue(engine.has_inflight())
        self.assertEqual(engine.memory.snapshot()[0]["used_blocks"], 1)
        self.assertEqual(engine.memory.prefix_snapshot()["published_blocks"], 0)
        when, _, _, kind, payload = heapq.heappop(engine.queue)
        self.assertEqual(when, 3)
        engine.now = when
        engine.handle_event(kind, payload)
        self.assertFalse(engine.has_inflight())
        self.assertEqual(engine.memory.snapshot()[0]["used_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
