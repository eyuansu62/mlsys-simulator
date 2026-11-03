"""Single-group CPU swap preserves tokens, storage lifetimes and virtual time."""

from dataclasses import replace
import random
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, EvaluationSpec, ExecutionSpec,
                       Experiment, InfraSpec, MemoryConfig, RequestSpec,
                       StrategySpec, simulate)


def pressure(*, host_bytes=32768, fallback="none", tp=1, timeout=None,
             cutoff=1000.0, policy="swap"):
    return Experiment(
        workload=(RequestSpec("a", 0.0, 16, 24, timeout_ms=timeout),
                  RequestSpec("b", 0.0, 16, 24)),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(tp))),
        strategy=StrategySpec(tensor_parallel_size=tp, max_batch_size=2,
                              preemption=policy, swap_fallback=fallback,
                              memory=MemoryConfig(kv_pool_bytes=24576 if tp == 1 else 12288,
                                                  cpu_swap_bytes=host_bytes,
                                                  swap_d2h_bandwidth_gbps=.001,
                                                  swap_h2d_bandwidth_gbps=.001,
                                                  swap_latency_us=0)),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=5,
                                    decode_ms=1, recompute_ms=5),
        execution=ExecutionSpec(max_simulated_ms=cutoff))


def transfers(run):
    return [event for event in run.events if event["event"] == "SWAP_START"]


class SingleGroupSwapTests(unittest.TestCase):
    def test_swap_preserves_outputs_without_recomputation(self):
        run = simulate(pressure())
        self.assertEqual(run.result["status"], "COMPLETED")
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertEqual([row["output_tokens_actual"] for row in run.requests], [24, 24])
        self.assertEqual(run.result["recomputed_tokens"], 0)
        self.assertEqual(run.result["recompute_execution_ms"], 0)
        copies = transfers(run)
        self.assertEqual([copy["direction"] for copy in copies], ["D2H", "H2D"])
        self.assertEqual([copy["bytes"] for copy in copies], [16384, 16384])
        self.assertAlmostEqual(run.result["simulated_ms"], 83.768)
        self.assertAlmostEqual(run.result["compute_busy_ms"], 51)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in run.result["memory"]))
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)
        self.assertEqual(run.result["swap"]["pending_transfers"], [])
        a = run.requests[0]
        self.assertEqual(a["swap_out_count"], 1)
        self.assertEqual(a["swap_in_count"], 1)
        self.assertEqual(a["swap_out_bytes"], 16384)
        self.assertEqual(a["swap_in_bytes"], 16384)
        self.assertAlmostEqual(a["swap_ms"], 32.768)

    def test_dma_intervals_do_not_overlap_gpu_steps(self):
        run = simulate(pressure())
        for copy in transfers(run):
            for step in [event for event in run.events if event["event"] == "STEP_START"]:
                start, end = step["timestamp_ms"], step["timestamp_ms"] + step["duration_ms"]
                self.assertTrue(end <= copy["start_ms"] or start >= copy["end_ms"])

    def test_inadequate_host_capacity_is_explicit_without_implicit_recompute(self):
        run = simulate(pressure(host_bytes=1))
        self.assertEqual(run.result["status"], "UNRESOLVED")
        self.assertEqual(run.result["error_code"], "HOST_SWAP_CAPACITY_EXCEEDED")
        self.assertEqual(run.result["recomputed_tokens"], 0)
        self.assertEqual(transfers(run), [])
        self.assertTrue(all(row["status"] == "UNFINISHED" for row in run.requests))

    def test_explicit_recompute_fallback_is_supported(self):
        run = simulate(pressure(host_bytes=1, fallback="recompute"))
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertGreater(run.result["recomputed_tokens"], 0)
        self.assertEqual(transfers(run), [])
        self.assertTrue(any(event["event"] == "SWAP_FALLBACK" for event in run.events))

    def test_smaller_feasible_victim_is_tried_after_host_rejection(self):
        run = simulate(pressure(host_bytes=8192))
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertEqual(transfers(run)[0]["request_id"], "b")
        self.assertEqual(transfers(run)[0]["bytes"], 8192)
        self.assertTrue(any(event["event"] == "SWAP_VICTIM_REJECTED" for event in run.events))

    def test_timeout_during_d2h_retains_source_to_completion(self):
        run = simulate(pressure(timeout=25))
        a = run.requests[0]
        self.assertEqual(a["status"], "TIMED_OUT")
        self.assertEqual(a["completion_ms"], 25)
        self.assertEqual(a["output_tokens_actual"], 17)
        finish = next(event for event in run.events if event["event"] == "REQUEST_FINISH"
                      and event["request_id"] == "a")
        self.assertTrue(finish["kv_release_deferred"])
        completion = next(event for event in run.events if event["event"] == "SWAP_COMPLETE")
        self.assertTrue(completion["cancelled"])
        self.assertAlmostEqual(completion["timestamp_ms"], 37.384)
        self.assertEqual([row["status"] for row in run.requests], ["TIMED_OUT", "COMPLETED"])
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in run.result["memory"]))

    def test_timeout_during_h2d_does_not_resume_or_free_destination_early(self):
        run = simulate(pressure(timeout=65))
        a = run.requests[0]
        self.assertEqual(a["status"], "TIMED_OUT")
        self.assertEqual(a["output_tokens_actual"], 17)
        completed = [event for event in run.events if event["event"] == "SWAP_COMPLETE"]
        self.assertEqual(len(completed), 2)
        self.assertTrue(completed[-1]["cancelled"])
        self.assertAlmostEqual(completed[-1]["timestamp_ms"], 76.768)
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)

    def test_cutoff_during_d2h_accounts_partial_copy_and_pins_both_sides(self):
        run = simulate(pressure(cutoff=22))
        self.assertEqual(run.result["status"], "UNRESOLVED")
        self.assertEqual(run.result["error_code"], "SIMULATION_TIME_LIMIT")
        self.assertAlmostEqual(run.requests[0]["swap_ms"], 1)
        self.assertEqual(run.requests[0]["swap_out_count"], 0)
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 16384)
        self.assertEqual(run.result["memory"][0]["used_blocks"], 3)
        self.assertAlmostEqual(run.result["swap"]["dma_busy_ms"], 1)

    def test_cutoff_during_h2d_keeps_host_and_gpu_destination(self):
        run = simulate(pressure(cutoff=65))
        self.assertEqual(run.result["status"], "UNRESOLVED")
        self.assertEqual(run.requests[0]["swap_in_count"], 0)
        self.assertEqual(run.result["memory"][0]["used_blocks"], 2)
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 16384)
        self.assertAlmostEqual(run.result["swap"]["dma_busy_ms"], 21)
        self.assertAlmostEqual(run.requests[0]["swap_ms"], 21)

    def test_tp_copies_include_all_rank_bytes_and_kv_replication(self):
        run = simulate(pressure(tp=4, host_bytes=65536))
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertEqual(transfers(run)[0]["bytes"], 32768)
        self.assertEqual(run.result["swap"]["completed_bytes_by_direction"],
                         {"D2H": 32768, "H2D": 32768})

    def test_swap_logging_is_observational_and_replay_is_exact(self):
        experiment = pressure()
        first, second = simulate(experiment), simulate(experiment)
        self.assertEqual(first, second)
        quiet = simulate(replace(experiment, execution=replace(experiment.execution, detailed_events=False)))
        self.assertEqual(first.result, quiet.result)
        self.assertEqual(first.requests, quiet.requests)
        self.assertEqual(quiet.events, [])

    def test_warmup_swap_is_separate_from_the_measured_run(self):
        base = pressure()
        experiment = replace(base, execution=replace(base.execution, warmup=base.workload))
        run = simulate(experiment)
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertAlmostEqual(run.result["simulated_ms"], simulate(base).result["simulated_ms"])
        self.assertTrue(run.result["warmup"]["enabled"])
        self.assertEqual(run.result["warmup"]["request_count"], 2)

    def test_bounded_drain_interrupts_a_copy_without_early_reuse(self):
        base = pressure()
        experiment = replace(base, evaluation=EvaluationSpec(measurement_window_ms=20,
                                                           max_drain_ms=2))
        run = simulate(experiment)
        self.assertEqual(run.result["error_code"], "DRAIN_TIME_LIMIT")
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 16384)
        self.assertEqual(run.result["memory"][0]["used_blocks"], 3)

    def test_generated_swap_traces_preserve_tokens_storage_and_time(self):
        rng = random.Random(7341)
        for case in range(75):
            requests = tuple(RequestSpec(f"r{i}", float(rng.randrange(10)),
                                         rng.randrange(1, 40), rng.randrange(1, 30),
                                         priority=rng.randrange(3),
                                         timeout_ms=rng.choice([None, None, 5.0, 30.0, 80.0]))
                             for i in range(rng.randrange(1, 7)))
            base = pressure(host_bytes=rng.choice([1, 8192, 16384, 65536]),
                            fallback=rng.choice(["none", "recompute"]),
                            cutoff=rng.choice([10.0, 50.0, 200.0, 1000.0]))
            experiment = replace(base, workload=requests,
                                 strategy=replace(base.strategy,
                                                  max_batch_size=rng.randrange(1, 5),
                                                  memory=replace(base.strategy.memory,
                                                                 kv_pool_bytes=8192*rng.randrange(1, 7))))
            with self.subTest(case=case):
                run = simulate(experiment)
                oracle = {request.request_id: request.output_tokens for request in requests}
                times = [event["timestamp_ms"] for event in run.events]
                self.assertEqual(times, sorted(times))
                for row in run.requests:
                    self.assertLessEqual(row["output_tokens_actual"], oracle[row["request_id"]])
                    self.assertEqual(len(row["token_times_ms"]), row["output_tokens_actual"])
                    if row["status"] == "COMPLETED":
                        self.assertEqual(row["output_tokens_actual"], oracle[row["request_id"]])
                host = run.result["swap"]
                self.assertGreaterEqual(host["host_reserved_bytes"], 0)
                self.assertLessEqual(host["host_reserved_bytes"], host["host_capacity_bytes"])
                for rank in run.result["memory"]:
                    self.assertGreaterEqual(rank["used_blocks"], 0)
                    self.assertLessEqual(rank["used_blocks"], rank["pool_blocks"])
                if run.result["status"] == "COMPLETED":
                    self.assertEqual(host["host_reserved_bytes"], 0)
                    self.assertEqual(host["pending_transfers"], [])
                    self.assertTrue(all(rank["used_blocks"] == 0 for rank in run.result["memory"]))


if __name__ == "__main__":
    unittest.main()
