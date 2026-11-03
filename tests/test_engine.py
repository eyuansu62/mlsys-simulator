"""Hand-derived event timelines and resource-pressure regression scenarios."""

from dataclasses import replace
import random
import unittest

from mlsys_sim.engine import simulate
from mlsys_sim.schema import (
    CalibrationSpec, DeviceSpec, EvaluationSpec, ExecutionSpec, Experiment,
    InfraSpec, MemoryConfig, RequestSpec, SimulationError, StrategySpec,
)


def experiment(requests=(), **strategy_options):
    return Experiment(
        workload=tuple(requests),
        strategy=StrategySpec(**strategy_options),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=10,
                                    decode_ms=1, recompute_ms=5),
    )


def request(name, arrival=0, prompt=16, output=3, timeout=None, priority=0):
    return RequestSpec(name, arrival, prompt, output, priority, timeout)


class TimelineTests(unittest.TestCase):
    def test_fifo_three_request_hand_calculation(self):
        exp = experiment([
            request("a", 0, output=3), request("b", 1, output=1),
            request("c", 12, output=2),
        ], scheduler="fifo")
        run = simulate(exp)
        a, b, c = run.requests
        self.assertEqual(a["token_times_ms"], [10, 11, 12])
        self.assertEqual(b["token_times_ms"], [22])
        self.assertEqual(c["token_times_ms"], [32, 33])
        self.assertEqual([a["ttft_ms"], b["ttft_ms"], c["ttft_ms"]], [10, 21, 20])
        self.assertIsNone(b["tpot_ms"])
        self.assertEqual(run.result["simulated_ms"], 33)
        self.assertEqual(run.result["counts"]["completed"], 3)
        self.assertEqual(run.result["status"], "COMPLETED")

    def test_continuous_batching_removes_finished_sequence(self):
        run = simulate(experiment([request("a", output=2), request("b", output=3)]))
        self.assertEqual(run.requests[0]["token_times_ms"], [10, 11])
        self.assertEqual(run.requests[1]["token_times_ms"], [10, 11, 12])
        starts = [event for event in run.events if event["event"] == "STEP_START"]
        self.assertEqual([event["request_ids"] for event in starts], [["a", "b"], ["a", "b"], ["b"]])

    def test_prefill_first_policy_at_same_timestamp_boundary(self):
        run = simulate(experiment([request("a", output=2), request("b", 10, output=1)]))
        self.assertEqual(run.requests[0]["token_times_ms"], [10, 21])
        self.assertEqual(run.requests[1]["token_times_ms"], [20])
        at_ten = [event["event"] for event in run.events if event["timestamp_ms"] == 10]
        self.assertLess(at_ten.index("STEP_COMPLETE"), at_ten.index("REQUEST_ARRIVAL"))
        self.assertLess(at_ten.index("REQUEST_ARRIVAL"), at_ten.index("STEP_START"))

    def test_timeout_during_step_pins_resources_to_completion(self):
        run = simulate(experiment([request("a", timeout=5), request("b", 6, output=1)]))
        a, b = run.requests
        self.assertEqual(a["status"], "TIMED_OUT")
        self.assertEqual(a["completion_ms"], 5)
        self.assertEqual(a["output_tokens_actual"], 0)
        self.assertEqual(b["first_token_ms"], 20)
        finish = next(event for event in run.events
                      if event["event"] == "REQUEST_FINISH" and event["request_id"] == "a")
        self.assertTrue(finish["kv_release_deferred"])
        releases = [event["timestamp_ms"] for event in run.events
                    if event["event"] == "DEFERRED_KV_RELEASE"]
        self.assertEqual(releases, [10])
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in run.result["memory"]))

    def test_completion_wins_timeout_at_same_timestamp(self):
        run = simulate(experiment([request("a", output=1, timeout=10)]))
        self.assertEqual(run.requests[0]["status"], "COMPLETED")
        self.assertEqual(run.requests[0]["completion_ms"], 10)

    def test_deterministic_replay_and_logging_do_not_change_metrics(self):
        exp = experiment([request("a"), request("b", 4)])
        first, second = simulate(exp), simulate(exp)
        self.assertEqual(first, second)
        quiet = simulate(replace(exp, execution=ExecutionSpec(detailed_events=False)))
        self.assertEqual(first.result, quiet.result)
        self.assertEqual(first.requests, quiet.requests)
        self.assertEqual(quiet.events, [])


class PressureTests(unittest.TestCase):
    def pressure(self, preemption="recompute"):
        return experiment([request("a", output=24, priority=1), request("b", output=24)],
                          memory=MemoryConfig(kv_pool_bytes=24576),
                          max_batch_size=2, preemption=preemption)

    def test_block_exhaustion_preempts_and_recomputes_without_duplicate_outputs(self):
        run = simulate(self.pressure())
        self.assertEqual(run.result["status"], "COMPLETED")
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertGreater(run.result["preemption_count"], 0)
        self.assertEqual(run.result["recomputed_tokens"], 16)
        self.assertEqual(run.result["recompute_execution_ms"], 5)
        for row in run.requests:
            self.assertEqual(row["output_tokens_actual"], 24)
            self.assertEqual(len(row["token_times_ms"]), 24)
            self.assertEqual(row["first_token_ms"], 10)
            self.assertEqual(row["num_computed_tokens"], 16 + 24 - 1)
        b = run.requests[1]
        self.assertEqual(b["preemption_count"], 1)
        self.assertGreater(max(b["itl_ms"]), 5)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in run.result["memory"]))

    def test_preemption_disabled_is_explicitly_unresolved_without_loop(self):
        run = simulate(self.pressure(preemption="none"))
        self.assertEqual(run.result["status"], "UNRESOLVED")
        self.assertEqual(run.result["error_code"], "MEMORY_NO_PROGRESS")
        self.assertEqual(run.result["preemption_count"], 0)
        self.assertLess(run.result["events_processed"], 100)

    def test_request_that_cannot_fit_alone_fails_without_preemption(self):
        run = simulate(experiment([request("too-long", prompt=49)],
                                  memory=MemoryConfig(kv_pool_bytes=24576)))
        self.assertEqual(run.result["status"], "COMPLETED")
        self.assertEqual(run.requests[0]["status"], "FAILED")
        self.assertEqual(run.requests[0]["error_code"], "KV_CAPACITY_EXCEEDED")
        self.assertEqual(run.result["preemption_count"], 0)
        self.assertEqual(run.result["counts"]["failed"], 1)

    def test_later_decode_growth_can_fail_even_when_prompt_fits(self):
        run = simulate(experiment([request("a", prompt=16, output=40)],
                                  memory=MemoryConfig(kv_pool_bytes=8192)))
        self.assertEqual(run.requests[0]["status"], "FAILED")
        self.assertEqual(run.requests[0]["output_tokens_actual"], 1)
        self.assertEqual(run.requests[0]["error_code"], "KV_CAPACITY_EXCEEDED")

    def test_failed_last_active_request_unblocks_waiting_admission(self):
        run = simulate(experiment([request("a", prompt=16, output=10),
                                   request("b", arrival=1, prompt=8, output=1)],
                                  memory=MemoryConfig(kv_pool_bytes=8192)))
        self.assertEqual(run.result["status"], "COMPLETED")
        self.assertEqual(run.requests[0]["status"], "FAILED")
        self.assertEqual(run.requests[1]["status"], "COMPLETED")
        self.assertEqual(run.requests[1]["output_tokens_actual"], 1)


class MetricsAndLimitsTests(unittest.TestCase):
    def test_empty_workload_has_null_latency_and_rates(self):
        run = simulate(experiment())
        self.assertEqual(run.result["status"], "COMPLETED")
        self.assertEqual(run.result["counts"]["total"], 0)
        self.assertIsNone(run.result["latency_ms"]["ttft"]["p99"])
        self.assertIsNone(run.result["throughput"]["requests_per_s"])

    def test_single_token_cannot_satisfy_tpot_sla(self):
        exp = replace(experiment([request("a", output=1)]),
                      evaluation=EvaluationSpec(tpot_sla_ms=2))
        run = simulate(exp)
        self.assertIsNone(run.requests[0]["tpot_ms"])
        self.assertEqual(run.result["goodput"]["eligible_requests"], 0)
        self.assertFalse(run.result["sla_satisfied"])

    def test_failure_stays_in_completion_rate_denominator(self):
        run = simulate(experiment([request("a", output=1), request("b", timeout=5)]))
        self.assertEqual(run.result["completion_rate"], 0.5)
        self.assertEqual(run.result["counts"]["timed_out"], 1)

    def test_explicit_arrival_window_includes_drain_and_excludes_later_arrivals(self):
        exp = replace(experiment([request("a", output=1), request("b", 20, output=1)]),
                      evaluation=EvaluationSpec(measurement_window_ms=5))
        run = simulate(exp)
        self.assertEqual(run.result["counts"]["completed"], 1)
        self.assertEqual(run.result["counts"]["not_admitted"], 1)
        self.assertEqual(run.requests[1]["status"], "NOT_ADMITTED")
        self.assertEqual(run.requests[1]["error_code"], "OUTSIDE_MEASUREMENT_WINDOW")
        self.assertEqual(run.result["completion_rate"], 1.0)
        self.assertEqual(run.result["completion_rate_denominator"], 1)
        self.assertEqual(run.result["simulated_ms"], 10)
        self.assertEqual(run.result["evaluation_count"], 1)
        self.assertEqual(run.result["throughput"]["requests_per_s"], 200)
        self.assertEqual(run.result["throughput"]["window_ms"], 5)

    def test_time_limit_preserves_unfinished_partial_request(self):
        exp = replace(experiment([request("a")]),
                      execution=ExecutionSpec(max_simulated_ms=10.5))
        run = simulate(exp)
        self.assertEqual(run.result["status"], "UNRESOLVED")
        self.assertEqual(run.result["simulated_ms"], 10.5)
        self.assertEqual(run.requests[0]["status"], "UNFINISHED")
        self.assertEqual(run.requests[0]["output_tokens_actual"], 1)
        self.assertEqual(run.requests[0]["first_token_ms"], 10)
        self.assertEqual(run.requests[0]["num_computed_tokens"], 16)
        self.assertEqual(run.result["compute_busy_ms"], 10.5)
        self.assertEqual(run.result["recompute_execution_ms"], 0)

    def test_time_limit_counts_partial_prefill_without_committing_token(self):
        exp = replace(experiment([request("a")]),
                      execution=ExecutionSpec(max_simulated_ms=5))
        run = simulate(exp)
        self.assertEqual(run.result["compute_busy_ms"], 5)
        self.assertEqual(run.requests[0]["output_tokens_actual"], 0)
        self.assertEqual(run.requests[0]["num_computed_tokens"], 0)
        self.assertTrue(any(rank["used_blocks"] > 0 for rank in run.result["memory"]))

    def test_time_limit_counts_partial_recompute_without_restoring_kv(self):
        exp = experiment([request("a", output=24, priority=1), request("b", output=24)],
                         memory=MemoryConfig(kv_pool_bytes=24576), max_batch_size=2)
        run = simulate(replace(exp, execution=ExecutionSpec(max_simulated_ms=35.5)))
        self.assertEqual(run.result["status"], "UNRESOLVED")
        self.assertEqual(run.result["compute_busy_ms"], 35.5)
        self.assertEqual(run.result["recompute_execution_ms"], 2.5)
        self.assertEqual(run.result["recomputed_tokens"], 0)
        b = run.requests[1]
        self.assertEqual(b["recompute_ms"], 2.5)
        self.assertEqual(b["output_tokens_actual"], 1)
        self.assertEqual(b["num_computed_tokens"], 0)

    def test_event_limit_is_explicit(self):
        exp = replace(experiment([request("a")]), execution=ExecutionSpec(max_events=1))
        run = simulate(exp)
        self.assertEqual(run.result["status"], "UNRESOLVED")
        self.assertEqual(run.result["error_code"], "SIMULATION_EVENT_LIMIT")
        self.assertEqual(run.result["events_processed"], 1)
        self.assertEqual(run.result["compute_busy_ms"], 10)
        self.assertEqual(run.requests[0]["output_tokens_actual"], 0)

    def test_event_limit_counts_partial_recompute_before_next_arrival(self):
        exp = experiment([request("a", output=24, priority=1), request("b", output=24),
                          request("c", arrival=35, output=1)],
                         memory=MemoryConfig(kv_pool_bytes=24576), max_batch_size=2)
        # Two arrivals + one prefill + 23 decode completions start recompute at
        # 33 ms. The next arrival at 35 ms hits the event budget mid-recompute.
        run = simulate(replace(exp, execution=ExecutionSpec(max_events=26)))
        self.assertEqual(run.result["error_code"], "SIMULATION_EVENT_LIMIT")
        self.assertEqual(run.result["simulated_ms"], 35)
        self.assertEqual(run.result["compute_busy_ms"], 35)
        self.assertEqual(run.result["recompute_execution_ms"], 2)
        self.assertEqual(run.result["recomputed_tokens"], 0)
        self.assertEqual(run.requests[1]["num_computed_tokens"], 0)
        self.assertEqual(run.requests[1]["output_tokens_actual"], 1)

    def test_stale_timeout_does_not_exhaust_budget_after_completion(self):
        exp = replace(experiment([request("a", output=1, timeout=10)]),
                      execution=ExecutionSpec(max_events=2))
        run = simulate(exp)
        self.assertEqual(run.result["status"], "COMPLETED")
        self.assertEqual(run.requests[0]["status"], "COMPLETED")

    def test_pp_online_is_rejected(self):
        exp = replace(experiment([request("a")], pipeline_parallel_size=2),
                      infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))))
        with self.assertRaises(SimulationError) as error:
            simulate(exp)
        self.assertEqual(error.exception.code, "UNSUPPORTED_MECHANISM")


class InvariantTests(unittest.TestCase):
    def test_bounded_random_traces_preserve_state_tokens_and_memory(self):
        rng = random.Random(731)
        for case in range(50):
            specs = [request(str(index), arrival=rng.randint(0, 15),
                             prompt=rng.randint(1, 40), output=rng.randint(1, 30),
                             timeout=rng.choice([None, None, rng.randint(1, 70)]),
                             priority=rng.randint(0, 2))
                     for index in range(rng.randint(1, 8))]
            exp = experiment(specs, max_batch_size=rng.randint(1, 4),
                             memory=MemoryConfig(kv_pool_bytes=8192 * rng.randint(1, 6)))
            run = simulate(exp)
            with self.subTest(case=case):
                self.assertEqual(run.result["status"], "COMPLETED")
                self.assertEqual(run.result["counts"]["total"], len(specs))
                self.assertEqual(sum(run.result["counts"][name] for name in
                                     ("completed", "failed", "timed_out", "unfinished")), len(specs))
                oracle = {spec.request_id: spec.output_tokens for spec in specs}
                for row in run.requests:
                    self.assertIn(row["status"], {"COMPLETED", "FAILED", "TIMED_OUT"})
                    self.assertLessEqual(row["output_tokens_actual"], oracle[row["request_id"]])
                    self.assertEqual(len(row["token_times_ms"]), row["output_tokens_actual"])
                    self.assertTrue(all(right > left for left, right in
                                        zip(row["token_times_ms"], row["token_times_ms"][1:])))
                self.assertTrue(all(rank["used_blocks"] == 0 for rank in run.result["memory"]))
                times = [event["timestamp_ms"] for event in run.events]
                self.assertEqual(times, sorted(times))
                for event in run.events:
                    if event["event"] == "STEP_START":
                        self.assertTrue(all(0 <= rank["used_blocks"] <= rank["pool_blocks"]
                                            for rank in event["memory"]))


if __name__ == "__main__":
    unittest.main()
