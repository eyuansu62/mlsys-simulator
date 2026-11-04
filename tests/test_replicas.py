"""Causal replica routing, one global metric window and resource ownership."""

from dataclasses import replace
from pathlib import Path
import random
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, EvaluationSpec, ExecutionSpec,
                       Experiment, InfraSpec, MemoryConfig, RequestSpec, StrategySpec,
                       load_experiment, simulate)
from mlsys_sim.replicas import MultiReplicaSimulation


ROOT = Path(__file__).resolve().parents[1]


def experiment(requests=(), *, replicas=2, pp=1, tp=1, policy="least_loaded", **strategy):
    return Experiment(workload=tuple(requests),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(replicas * pp * tp))),
        strategy=StrategySpec(replicas=replicas, replica_routing=policy,
                              pipeline_parallel_size=pp, tensor_parallel_size=tp, **strategy),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=10, decode_ms=1, recompute_ms=10))


def owners(result):
    return {row["request_id"]: row["replica_index"] for row in result.requests}


class ReplicaRoutingTests(unittest.TestCase):
    def test_round_robin_fifo_has_parallel_makespan_and_global_percentiles(self):
        exp = experiment([RequestSpec(str(i), 0, 4, 1) for i in range(5)],
                         policy="round_robin", scheduler="fifo")
        result = simulate(exp)
        self.assertEqual([row["replica_index"] for row in result.requests], [0, 1, 0, 1, 0])
        self.assertEqual([row["completion_ms"] for row in result.requests], [10, 10, 20, 20, 30])
        self.assertEqual(result.result["simulated_ms"], 30)
        self.assertEqual(result.result["compute_busy_ms"], 50)
        self.assertEqual(result.result["latency_ms"]["ttft"]["p50"], 20)
        self.assertAlmostEqual(result.result["throughput"]["requests_per_s"], 5 * 1000 / 30)
        self.assertEqual(result.result["routing"]["request_count_by_replica"], [3, 2])

    def test_least_loaded_uses_completions_visible_at_the_arrival(self):
        exp = experiment([RequestSpec("a-long", 0, 4, 30), RequestSpec("b-short", 0, 4, 1),
                          RequestSpec("c", 11, 4, 1)], scheduler="fifo")
        result = simulate(exp)
        self.assertEqual(owners(result), {"a-long": 0, "b-short": 1, "c": 1})
        last = result.result["routing"]["assignments"][-1]
        self.assertEqual(last["unfinished_requests_before"], 0)

    def test_completion_and_timeout_priorities_precede_same_time_arrival(self):
        exp = experiment([RequestSpec("a", 0, 4, 1), RequestSpec("b", 0, 4, 2),
                          RequestSpec("c", 10, 4, 1)], scheduler="fifo")
        result = simulate(exp)
        self.assertEqual(owners(result)["c"], 0)
        at_ten = [event for event in result.events if event["timestamp_ms"] == 10]
        routed = next(i for i, event in enumerate(at_ten) if event["event"] == "REQUEST_ROUTED")
        completions = [i for i, event in enumerate(at_ten) if event["event"] == "STEP_COMPLETE"]
        self.assertEqual(len(completions), 2)
        self.assertTrue(all(i < routed for i in completions))

    def prefix_experiment(self, *, namespace="demo", identities=True):
        rows = [RequestSpec("a", 0, 4, 1, prompt_token_ids=(100, 101, 102, 103), cache_namespace="demo"),
                RequestSpec("b", 0, 8, 1, prompt_token_ids=tuple(range(8)), cache_namespace="demo"),
                RequestSpec("c", 20, 8, 1, prompt_token_ids=tuple(range(8)) if identities else (),
                            cache_namespace=namespace)]
        return experiment(rows, policy="prefix_aware", prefix_cache=True,
                          memory=MemoryConfig(kv_block_tokens=4))

    def test_prefix_affinity_uses_only_local_cache_and_probe_does_not_count_as_lookup(self):
        result = simulate(self.prefix_experiment())
        self.assertEqual(owners(result), {"a": 0, "b": 1, "c": 1})
        self.assertEqual(result.requests[2]["prefix_cached_tokens"], 4)
        self.assertEqual(result.result["routing"]["assignments"][-1]["cached_prefix_tokens"], 4)
        self.assertEqual(result.result["prefix_cache"]["lookup_requests"], 3)
        self.assertEqual(result.result["prefix_cache"]["hit_requests"], 1)
        self.assertEqual(result.result["prefix_cache"]["hit_tokens"], 4)

    def test_prefix_affinity_does_not_overload_a_busier_cached_replica(self):
        base = self.prefix_experiment()
        rows = (*base.workload[:2],
                RequestSpec("load", 20, 8, 12, prompt_token_ids=tuple(range(8)), cache_namespace="demo"),
                replace(base.workload[2], arrival_ms=21))
        result = simulate(replace(base, workload=rows))
        self.assertEqual(owners(result)["load"], 1)
        self.assertEqual(owners(result)["c"], 0)
        self.assertEqual(result.requests[3]["prefix_cached_tokens"], 0)

    def test_prefix_namespace_and_missing_identity_cannot_create_false_affinity(self):
        for kwargs in ({"namespace": "another"}, {"identities": False}):
            with self.subTest(kwargs=kwargs):
                result = simulate(self.prefix_experiment(**kwargs))
                self.assertEqual(owners(result)["c"], 0)
                self.assertEqual(result.requests[2]["prefix_cached_tokens"], 0)
                self.assertEqual(result.result["routing"]["assignments"][-1]["cached_prefix_tokens"], 0)

    def test_round_robin_does_not_reuse_a_different_replica_cache(self):
        base = self.prefix_experiment()
        exp = replace(base, strategy=replace(base.strategy, replica_routing="round_robin"))
        result = simulate(exp)
        self.assertEqual(owners(result)["c"], 0)
        self.assertEqual(result.result["prefix_cache"]["hit_tokens"], 0)

    def test_future_output_targets_do_not_influence_simultaneous_routing(self):
        rows = [RequestSpec(str(i), 0, 4, 1) for i in range(8)]
        for policy in ("round_robin", "least_loaded", "prefix_aware"):
            with self.subTest(policy=policy):
                base = experiment(rows, policy=policy)
                longer = replace(base, workload=tuple(replace(row, output_tokens=20 + i)
                                                       for i, row in enumerate(rows)))
                self.assertEqual(owners(simulate(base)), owners(simulate(longer)))

    def test_local_engines_only_register_arrived_requests_and_oracle_targets(self):
        exp = experiment([RequestSpec("now", 0, 4, 1), RequestSpec("future", 100, 4, 90)])
        exp = replace(exp, execution=ExecutionSpec(max_simulated_ms=20))
        simulation = MultiReplicaSimulation(exp)
        self.assertTrue(all(engine._output_targets == {} for engine in simulation.engines))
        result = simulation.run()
        self.assertIsNone(owners(result)["future"])
        self.assertTrue(all("future" not in engine._output_targets for engine in simulation.engines))
        self.assertTrue(all("future" not in engine.memory._prefix_specs for engine in simulation.engines))


class ReplicaBudgetTests(unittest.TestCase):
    def test_event_budget_is_global_and_retains_uncommitted_replica_work(self):
        for budget, completed in ((1, 0), (2, 0), (3, 1)):
            with self.subTest(budget=budget):
                base = experiment([RequestSpec("a", 0, 4, 1), RequestSpec("b", 0, 4, 1)],
                                  policy="round_robin")
                exp = replace(base, execution=ExecutionSpec(max_events=budget))
                simulation = MultiReplicaSimulation(exp)
                result = simulation.run()
                self.assertEqual(result.result["error_code"], "SIMULATION_EVENT_LIMIT")
                self.assertEqual(result.result["events_processed"], budget)
                self.assertEqual(result.result["routing"]["unavailable_replicas"], [])
                self.assertEqual(sum(engine.handled_events for engine in simulation.engines), budget)
                self.assertEqual(result.result["counts"]["completed"], completed)
                if budget >= 2:
                    self.assertEqual(result.result["simulated_ms"], 10)
                    self.assertEqual(result.result["compute_busy_ms"], 20)
                else:
                    self.assertEqual(result.result["simulated_ms"], 0)
                    self.assertIsNone(owners(result)["b"])

    def test_global_time_limit_accounts_parallel_fragments_and_pins_both_pools(self):
        base = experiment([RequestSpec("a", 0, 4, 1), RequestSpec("b", 0, 4, 1)])
        result = simulate(replace(base, execution=ExecutionSpec(max_simulated_ms=5)))
        self.assertEqual(result.result["error_code"], "SIMULATION_TIME_LIMIT")
        self.assertEqual(result.result["compute_busy_ms"], 10)
        self.assertEqual(result.result["routing"]["unavailable_replicas"], [])
        self.assertEqual(result.result["routing"]["unresolved_replicas"], [0, 1])
        self.assertEqual([row["output_tokens_actual"] for row in result.requests], [0, 0])
        self.assertTrue(all(rank["used_blocks"] == 1 for rank in result.result["memory"]))

    def test_measurement_window_and_finite_drain_use_one_arrival_cohort(self):
        rows = [RequestSpec("a", 0, 4, 1), RequestSpec("b", 8, 4, 1),
                RequestSpec("outside", 10, 4, 1), RequestSpec("later", 50, 4, 1)]
        base = experiment(rows, policy="round_robin")
        exp = replace(base, calibration=replace(base.calibration, prefill_ms=20),
                      evaluation=EvaluationSpec(measurement_window_ms=10, max_drain_ms=2))
        result = simulate(exp)
        self.assertEqual(result.result["error_code"], "DRAIN_TIME_LIMIT")
        self.assertEqual(result.result["simulated_ms"], 12)
        self.assertEqual(result.result["compute_busy_ms"], 16)
        self.assertEqual(result.result["admitted_count"], 2)
        self.assertEqual(result.result["counts"]["not_admitted"], 2)
        self.assertEqual(result.result["routing"]["assigned_requests"], 2)
        self.assertTrue(all(row["replica_index"] is None for row in result.requests[2:]))

    def test_idle_replica_ignores_stale_timeout_but_accepts_a_later_arrival(self):
        rows = [RequestSpec("a", 0, 4, 1, timeout_ms=1000),
                RequestSpec("b", 20, 4, 1, timeout_ms=1000)]
        result = simulate(experiment(rows, policy="round_robin"))
        self.assertEqual(result.result["status"], "COMPLETED")
        self.assertEqual(result.result["simulated_ms"], 30)
        self.assertEqual(result.result["events_processed"], 4)

    def test_zero_timeouts_terminate_before_any_work_starts(self):
        result = simulate(experiment([RequestSpec("a", 0, 4, 1, timeout_ms=0),
                                      RequestSpec("b", 0, 4, 1, timeout_ms=0)], policy="round_robin"))
        self.assertEqual(result.result["counts"]["timed_out"], 2)
        self.assertEqual(result.result["simulated_ms"], 0)
        self.assertEqual(result.result["compute_busy_ms"], 0)
        self.assertEqual(result.result["events_processed"], 4)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))

    def test_timed_out_inflight_work_remains_visible_to_load_balancing(self):
        result = simulate(experiment([RequestSpec("a", 0, 4, 1, timeout_ms=1),
                                      RequestSpec("b", 2, 4, 1)]))
        self.assertEqual(owners(result), {"a": 0, "b": 1})
        self.assertEqual(result.result["counts"]["completed"], 1)
        self.assertEqual(result.result["counts"]["timed_out"], 1)
        self.assertEqual(result.result["compute_busy_ms"], 20)
        self.assertEqual(result.result["simulated_ms"], 12)

    def test_local_no_progress_does_not_stop_other_replicas_or_future_arrivals(self):
        rows = [RequestSpec("a", 0, 16, 24), RequestSpec("b", 0, 4, 1),
                RequestSpec("c", 0, 16, 24), RequestSpec("d", 0, 4, 1),
                RequestSpec("e", 30, 4, 1)]
        base = experiment(rows, policy="round_robin", preemption="none",
                          memory=MemoryConfig(kv_pool_bytes=24576))
        exp = replace(base, calibration=replace(base.calibration, prefill_ms=5))
        result = simulate(exp)
        self.assertEqual(result.result["status"], "UNRESOLVED")
        self.assertEqual(result.result["error_code"], "REPLICA_UNRESOLVED")
        self.assertEqual(result.result["routing"]["unavailable_replicas"], [0])
        self.assertEqual(result.result["counts"]["completed"], 3)
        self.assertEqual(owners(result)["e"], 1)
        self.assertEqual(result.requests[-1]["completion_ms"], 35)

    def test_all_unavailable_replicas_leave_later_arrivals_unassigned(self):
        rows = [RequestSpec(name, 0, 16, 24) for name in ("a", "b", "c", "d")]
        rows.append(RequestSpec("future", 100, 4, 1))
        exp = experiment(rows, policy="round_robin", preemption="none",
                         memory=MemoryConfig(kv_pool_bytes=24576))
        result = simulate(exp)
        self.assertEqual(result.result["error_code"], "NO_AVAILABLE_REPLICA")
        self.assertEqual(result.result["routing"]["unavailable_replicas"], [0, 1])
        self.assertEqual(result.result["routing"]["assigned_requests"], 4)
        self.assertIsNone(owners(result)["future"])
        self.assertEqual(result.requests[-1]["status"], "UNFINISHED")
        self.assertLess(result.result["simulated_ms"], 100)

    def test_empty_or_outside_only_workload_has_no_routing_or_clock_advance(self):
        for rows in ([], [RequestSpec("outside", 20, 4, 1)]):
            with self.subTest(rows=rows):
                exp = replace(experiment(rows), evaluation=EvaluationSpec(measurement_window_ms=10))
                result = simulate(exp)
                self.assertEqual(result.result["status"], "COMPLETED")
                self.assertEqual(result.result["simulated_ms"], 0)
                self.assertEqual(result.result["routing"]["request_count_by_replica"], [0, 0])
                self.assertEqual(result.result["admitted_count"], 0)


class ReplicaIntegrationTests(unittest.TestCase):
    def combined(self):
        base = load_experiment(ROOT / "examples/v1_3/pp_mixed_cache_swap.json")
        rows = tuple(replace(request, request_id=f"{request.request_id}-{replica}")
                     for request in base.workload for replica in range(2))
        return replace(base, workload=rows,
            infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(8))),
            strategy=replace(base.strategy, replicas=2, replica_routing="round_robin",
                tensor_parallel_size=2, pipeline_p2p=True, pipeline_feedback_latency_us=500,
                memory=replace(base.strategy.memory, kv_pool_bytes=2560)))

    def test_tp_pp_mixed_cache_and_swap_have_independent_replica_ownership(self):
        exp = self.combined()
        result = simulate(exp)
        self.assertEqual(result.result["counts"]["completed"], 10)
        self.assertEqual(result.result["routing"]["request_count_by_replica"], [5, 5])
        self.assertGreater(result.result["batching"]["mixed_batches_completed"], 0)
        self.assertGreater(result.result["prefix_cache"]["hit_tokens"], 0)
        self.assertGreaterEqual(result.result["swap"]["completed_transfers_by_direction"]["D2H"], 2)
        self.assertEqual(result.result["swap"]["host_capacity_bytes"], 2 * exp.strategy.memory.cpu_swap_bytes)
        self.assertEqual(result.result["swap"]["host_reserved_bytes"], 0)
        self.assertEqual(result.result["prefix_cache"]["attached_block_references"], 0)
        self.assertEqual(result.result["prefix_cache"]["private_blocks"], 0)
        self.assertEqual(len(result.result["memory"]), 8)
        self.assertEqual({rank["replica_index"] for rank in result.result["memory"]}, {0, 1})
        target = {request.request_id: request.output_tokens for request in exp.workload}
        self.assertTrue(all(row["output_tokens_actual"] == target[row["request_id"]] for row in result.requests))
        for index in range(0, len(result.requests), 2):
            self.assertEqual(result.requests[index]["token_times_ms"], result.requests[index + 1]["token_times_ms"])
        self.assertEqual(result.result["swap"]["peak_scope"], "observed_global_timeline")
        self.assertLessEqual(result.result["swap"]["host_peak_reserved_bytes"],
                             result.result["swap"]["sum_replica_host_peak_reserved_bytes"])

    def test_replay_and_logging_do_not_change_routing_or_metrics(self):
        exp = self.combined()
        first = simulate(exp)
        second = simulate(exp)
        quiet = simulate(replace(exp, execution=replace(exp.execution, detailed_events=False)))
        self.assertEqual(first, second)
        self.assertEqual(first.result, quiet.result)
        self.assertEqual(first.requests, quiet.requests)
        self.assertEqual(quiet.events, [])

    def test_bounded_generated_traces_preserve_global_budget_and_local_storage(self):
        rng = random.Random(831)
        for iteration in range(60):
            replicas, pp, tp = rng.choice((2, 3)), rng.choice((1, 2)), rng.choice((1, 2))
            requests = [RequestSpec(str(i), rng.randrange(13), rng.randrange(1, 21), rng.randrange(1, 6),
                                    timeout_ms=rng.choice((None, None, 5, 12)))
                        for i in range(rng.randrange(1, 9))]
            requests = [replace(row, prompt_token_ids=tuple(range(row.input_tokens))) for row in requests]
            exp = experiment(requests, replicas=replicas, pp=pp, tp=tp,
                policy=rng.choice(("round_robin", "least_loaded", "prefix_aware")),
                scheduler=rng.choice(("fifo", "continuous_batching")), max_batch_size=3,
                max_batch_tokens=12, pipeline_microbatch_size=2,
                mixed_batching=bool(iteration % 2), chunked_prefill=bool(iteration % 3),
                prefill_chunk_tokens=4, prefix_cache=bool(iteration % 2),
                preemption=rng.choice(("none", "recompute", "swap")),
                max_preemptions_per_request=10,
                memory=MemoryConfig(kv_block_tokens=4, kv_pool_bytes=rng.choice((2048, 8192, 32768)),
                                    cpu_swap_bytes=65536, guard_blocks=rng.choice((0, 1))))
            exp = replace(exp, execution=ExecutionSpec(max_events=5000, max_simulated_ms=500))
            simulation = MultiReplicaSimulation(exp)
            result = simulation.run()
            with self.subTest(iteration=iteration):
                self.assertEqual(result.result["counts"]["total"], len(requests))
                self.assertEqual(result.result["events_processed"],
                                 sum(engine.handled_events for engine in simulation.engines))
                self.assertLessEqual(result.result["events_processed"], 5000)
                targets = {row.request_id: row.output_tokens for row in requests}
                for row in result.requests:
                    self.assertLessEqual(row["output_tokens_actual"], targets[row["request_id"]])
                    self.assertTrue(all(a < b for a, b in zip(row["token_times_ms"], row["token_times_ms"][1:])))
                    if row["status"] == "COMPLETED":
                        self.assertEqual(row["output_tokens_actual"], targets[row["request_id"]])
                        self.assertEqual(row["prefill_computed_tokens"] + row["prefix_cached_tokens"],
                                         next(spec.input_tokens for spec in requests if spec.request_id == row["request_id"]))
                for rank in result.result["memory"]:
                    self.assertGreaterEqual(rank["used_blocks"], 0)
                    self.assertLessEqual(rank["peak_used_blocks"] + rank["guard_blocks"], rank["pool_blocks"])
                if "swap" in result.result:
                    self.assertLessEqual(result.result["swap"]["host_peak_reserved_bytes"],
                                         result.result["swap"]["host_capacity_bytes"])
                times = [event["timestamp_ms"] for event in result.events]
                self.assertEqual(times, sorted(times))


if __name__ == "__main__":
    unittest.main()
