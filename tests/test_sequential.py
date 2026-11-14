"""Exact fixed streams use one persistent deployment and true terminal feedback."""

from dataclasses import replace
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment, InfraSpec,
                       MemoryConfig, RequestSpec, SimulationError, StrategySpec)
from mlsys_sim.sequential import simulate_sequential


def experiment(**strategy):
    return Experiment(strategy=StrategySpec(scheduler="continuous_batching", **strategy),
                      calibration=CalibrationSpec(backend="synthetic", prefill_ms=10, decode_ms=2))


class SequentialTests(unittest.TestCase):
    def test_original_order_and_heterogeneous_lengths_follow_terminal_feedback(self):
        requests = (RequestSpec("b", 0, 4, 3), RequestSpec("a", 0, 12, 1), RequestSpec("c", 0, 8, 2))
        report = simulate_sequential(experiment(), requests)
        rows = report["run"].requests
        self.assertEqual([row["request_id"] for row in rows], ["b", "a", "c"])
        self.assertEqual([row["arrival_ms"] for row in rows], [0, 14, 24])
        self.assertEqual([row["completion_ms"] for row in rows], [14, 24, 36])
        self.assertEqual([row["prefill_computed_tokens"] for row in rows], [4, 12, 8])
        self.assertEqual([row["output_tokens_actual"] for row in rows], [3, 1, 2])
        self.assertEqual(report["not_started"], [])
        self.assertEqual(report["first_compute_start_ms"], {"b": 0, "a": 14, "c": 24})
        self.assertTrue(report["resources_drained"])

    def test_same_deployment_cache_survives_between_requests(self):
        selected = experiment(prefix_cache=True, memory=MemoryConfig(kv_block_tokens=4))
        prompt = (1, 2, 3, 4, 5, 6, 7, 8)
        report = simulate_sequential(selected, (RequestSpec("seed", 0, 8, 1, prompt_token_ids=prompt),
                                                RequestSpec("reuse", 0, 8, 1, prompt_token_ids=prompt)))
        seed, reuse = report["run"].requests
        self.assertEqual(reuse["arrival_ms"], seed["completion_ms"])
        self.assertGreater(reuse["prefix_cached_tokens"], 0)
        self.assertEqual(reuse["prefix_cached_tokens"], 4)

    def test_request_timeout_releases_next_but_retains_pinned_compute(self):
        report = simulate_sequential(experiment(), (RequestSpec("fail", 0, 4, 2, timeout_ms=1),
                                                     RequestSpec("healthy", 0, 4, 1)))
        fail, healthy = report["run"].requests
        self.assertEqual(fail["status"], "TIMED_OUT")
        self.assertEqual(healthy["arrival_ms"], 1)
        self.assertEqual(report["first_compute_start_ms"]["healthy"], 10)
        self.assertEqual(healthy["status"], "COMPLETED")
        self.assertEqual(healthy["completion_ms"], 20)

    def test_deadline_keeps_partial_and_unoffered_ids_distinct(self):
        selected = replace(experiment(), execution=ExecutionSpec(max_simulated_ms=11))
        report = simulate_sequential(selected, (RequestSpec("partial", 0, 4, 3), RequestSpec("later", 0, 8, 1)))
        self.assertEqual(report["not_started"], ["later"])
        self.assertEqual(report["run"].requests[0]["output_tokens_actual"], 1)
        self.assertEqual(report["run"].requests[0]["status"], "UNFINISHED")
        self.assertEqual(report["elapsed_ms"], 11)
        self.assertFalse(report["resources_drained"])

    def test_last_completion_exactly_at_cutoff_is_success(self):
        selected = replace(experiment(), execution=ExecutionSpec(max_simulated_ms=20))
        report = simulate_sequential(selected, (RequestSpec("one", 0, 4, 1), RequestSpec("two", 0, 8, 1)))
        self.assertEqual([row["completion_ms"] for row in report["run"].requests], [10, 20])
        self.assertIsNone(report["stop_reason"])
        self.assertTrue(report["resources_drained"])
        self.assertEqual(report["run"].result["status"], "COMPLETED")

    def test_no_future_release_at_cutoff(self):
        selected = replace(experiment(), execution=ExecutionSpec(max_simulated_ms=10))
        report = simulate_sequential(selected, (RequestSpec("one", 0, 4, 1), RequestSpec("two", 0, 8, 1)))
        self.assertEqual(report["run"].requests[0]["status"], "COMPLETED")
        self.assertEqual(report["not_started"], ["two"])
        self.assertEqual(report["stop_reason"], "SIMULATION_TIME_LIMIT")

    def test_replica_placement_and_pipeline_resource_drain(self):
        for pp, replicas in ((1, 2), (2, 1), (2, 2)):
            with self.subTest(pp=pp, replicas=replicas):
                selected = experiment(pipeline_parallel_size=pp, replicas=replicas,
                                      pipeline_p2p=False, pipeline_feedback_latency_us=1000)
                selected = replace(selected, infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}") for i in range(pp * replicas))))
                report = simulate_sequential(selected, [RequestSpec("one", 0, 4, 1), RequestSpec("two", 0, 8, 1)])
                self.assertTrue(report["resources_drained"])
                self.assertEqual(report["run"].result["counts"]["completed"], 2)
                self.assertEqual(report["run"].requests[1]["arrival_ms"], report["run"].requests[0]["completion_ms"])
                self.assertGreaterEqual(report["elapsed_ms"], report["run"].requests[-1]["completion_ms"])
                if replicas > 1:
                    self.assertTrue(all(type(row["replica_index"]) is int for row in report["run"].requests))

    def test_invalid_inputs_and_nonempty_or_window_experiment_are_rejected(self):
        request = RequestSpec("a", 0, 4, 1)
        for requests in ([], [request, request], [replace(request, arrival_ms=1)], [replace(request, input_tokens=True)]):
            with self.subTest(requests=requests), self.assertRaises(SimulationError):
                simulate_sequential(experiment(), requests)
        with self.assertRaises(SimulationError):
            simulate_sequential(replace(experiment(), workload=(request,)), [request])
        with self.assertRaises(SimulationError):
            simulate_sequential(replace(experiment(), evaluation=replace(experiment().evaluation,
                                                                         measurement_window_ms=5)), [request])

    def test_event_budget_leaves_unreleased_list_intact(self):
        selected = replace(experiment(), execution=ExecutionSpec(max_events=1))
        report = simulate_sequential(selected, (RequestSpec("one", 0, 4, 1), RequestSpec("two", 0, 8, 1)))
        self.assertEqual(report["not_started"], ["two"])
        self.assertEqual(report["run"].result["events_processed"], 1)
        self.assertEqual(report["stop_reason"], "SIMULATION_EVENT_LIMIT")

    def test_request_local_rejection_advances_without_blocking_next(self):
        report = simulate_sequential(experiment(max_batch_tokens=16),
                                     [RequestSpec("too_large", 0, 32, 1), RequestSpec("healthy", 0, 4, 1)])
        rejected, healthy = report["run"].requests
        self.assertEqual(rejected["status"], "FAILED")
        self.assertEqual(rejected["error_code"], "BATCH_TOKEN_LIMIT_EXCEEDED")
        self.assertEqual(healthy["arrival_ms"], rejected["completion_ms"])
        self.assertEqual(healthy["status"], "COMPLETED")
        self.assertEqual(report["not_started"], [])

    def test_static_deployment_failure_is_not_fabricated_as_request_evidence(self):
        selected = replace(experiment(), infra=InfraSpec(devices=(DeviceSpec(memory_bytes=1),)))
        with self.assertRaises(SimulationError) as failure:
            simulate_sequential(selected, [RequestSpec("one", 0, 4, 1)])
        self.assertEqual(failure.exception.code, "STATIC_MEMORY_INFEASIBLE")


if __name__ == "__main__":
    unittest.main()
