"""Pipeline hand calculations and reference/resource lifetime regressions."""

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

from mlsys_sim.pipeline import PipelineSimulation
from mlsys_sim.profiles import profile_identity, shape_key
from mlsys_sim.schema import (CalibrationSpec, DeviceSpec, EvaluationSpec, ExecutionSpec,
                              Experiment, InfraSpec, MemoryConfig, ModelSpec,
                              RequestSpec, SimulationError, StrategySpec)


def experiment(requests=(), *, pp=2, tp=1, layers=2, stage_layers=(), **strategy):
    return Experiment(
        workload=tuple(requests),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(pp * tp))),
        strategy=StrategySpec(model=ModelSpec(num_layers=layers), pipeline_parallel_size=pp,
                              tensor_parallel_size=tp, stage_layers=tuple(stage_layers),
                              pipeline_p2p=False, max_batch_tokens=256,
                              pipeline_max_inflight_microbatches=8, **strategy),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=layers,
                                    decode_ms=layers, recompute_ms=layers),
    )


def run(exp):
    return PipelineSimulation(exp).run()


def request(name="a", prompt=4, output=1, arrival=0, **kwargs):
    return RequestSpec(name, arrival, prompt, output, **kwargs)


class PipelineTimelineTests(unittest.TestCase):
    def test_two_stages_three_microbatches_have_fill_and_drain(self):
        result = run(experiment([request(name) for name in ["a", "b", "c"]]))
        self.assertEqual(result.result["status"], "COMPLETED")
        self.assertEqual(result.result["simulated_ms"], 4)
        self.assertEqual([row["first_token_ms"] for row in result.requests], [2, 3, 4])
        self.assertEqual(result.result["pipeline"]["stage_busy_ms"], [3, 3])
        self.assertEqual(result.result["compute_busy_ms"], 6)
        self.assertAlmostEqual(result.result["pipeline"]["idle_fraction"], .25)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))

    def test_single_microbatch_latency_is_sum_of_stages(self):
        result = run(experiment([request()]))
        self.assertEqual(result.requests[0]["completion_ms"], 2)
        self.assertEqual(result.result["simulated_ms"], 2)

    def test_uneven_stages_expose_the_bottleneck(self):
        result = run(experiment([request(name) for name in ["a", "b", "c"]],
                                layers=3, stage_layers=(1, 2)))
        self.assertEqual([row["first_token_ms"] for row in result.requests], [3, 5, 7])
        self.assertEqual(result.result["pipeline"]["stage_busy_ms"], [3, 6])

    def test_autoregressive_token_cannot_bypass_feedback(self):
        exp = experiment([request(output=3)], pipeline_feedback_latency_us=500)
        result = run(exp)
        self.assertEqual(result.requests[0]["token_times_ms"], [2, 4.5, 7])
        self.assertEqual(result.result["simulated_ms"], 7.5)
        self.assertEqual(result.requests[0]["num_computed_tokens"], 6)
        starts = [event for event in result.events if event["event"] == "STAGE_START"
                  and event["stage"] == 0]
        self.assertEqual([event["timestamp_ms"] for event in starts], [0, 2.5, 5])

    def test_p2p_payload_and_latency_have_explicit_cost(self):
        exp = experiment([request()])
        exp = replace(exp, strategy=replace(exp.strategy, pipeline_p2p=True),
                      infra=replace(exp.infra, link_latency_us=1000, link_bandwidth_gbps=1))
        result = run(exp)
        expected = 2 + 1 + (4 * 128 * 2) / 1e6
        self.assertAlmostEqual(result.requests[0]["completion_ms"], expected)
        transfer = next(event for event in result.events if event["event"] == "TRANSFER_START")
        self.assertEqual(transfer["bytes_per_rank"], 1024)
        self.assertAlmostEqual(result.result["pipeline"]["links"][0]["busy_ms"], 1.001024)

    def test_fifo_does_not_interleave_requests(self):
        result = run(experiment([request("a", output=2), request("b")], scheduler="fifo"))
        self.assertEqual([row["token_times_ms"] for row in result.requests], [[2, 4], [6]])

    def test_microbatch_groups_are_homogeneous_and_bounded(self):
        exp = experiment([request(name, output=2) for name in ["a", "b", "c", "d"]],
                         pipeline_microbatch_size=2)
        result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 4)
        admissions = [event for event in result.events if event["event"] == "MICROBATCH_ADMITTED"]
        self.assertTrue(all(len(event["request_ids"]) <= 2 for event in admissions))
        self.assertEqual(admissions[0]["request_ids"], ["a", "b"])

    def test_tp_two_inside_two_pipeline_stages(self):
        result = run(experiment([request("a"), request("b")], tp=2))
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual(len(result.result["memory"]), 4)
        self.assertEqual(result.result["pipeline"]["tensor_parallel_size"], 2)
        self.assertEqual(result.result["simulated_ms"], 3)

    def test_inflight_bound_and_stage_nonoverlap(self):
        exp = experiment([request(str(i)) for i in range(5)])
        exp = replace(exp, strategy=replace(exp.strategy, pipeline_max_inflight_microbatches=2))
        result = run(exp)
        self.assertEqual(result.result["pipeline"]["peak_inflight_microbatches"], 2)
        busy = {}
        for event in result.events:
            if event["event"] == "STAGE_START":
                self.assertNotIn(event["stage"], busy)
                busy[event["stage"]] = event["microbatch_id"]
            elif event["event"] == "STAGE_COMPLETE":
                self.assertEqual(busy.pop(event["stage"]), event["microbatch_id"])
        self.assertEqual(busy, {})

    def test_activation_links_are_serialized_when_compute_is_faster(self):
        exp = experiment([request(str(i)) for i in range(4)])
        exp = replace(exp, strategy=replace(exp.strategy, pipeline_p2p=True),
                      infra=replace(exp.infra, link_latency_us=2000, link_bandwidth_gbps=1))
        result = run(exp)
        starts = [event for event in result.events if event["event"] == "TRANSFER_START"]
        self.assertTrue(all(right["start_ms"] >= left["end_ms"]
                            for left, right in zip(starts, starts[1:])))
        self.assertEqual(result.result["counts"]["completed"], 4)


class PipelineLifetimeTests(unittest.TestCase):
    def test_timeout_keeps_inflight_wave_until_feedback_boundary(self):
        exp = experiment([request(output=3, timeout_ms=2.5)], pipeline_feedback_latency_us=1000)
        result = run(exp)
        self.assertEqual(result.requests[0]["status"], "TIMED_OUT")
        self.assertEqual(result.requests[0]["token_times_ms"], [2])
        self.assertEqual(result.result["simulated_ms"], 3)
        release = next(event for event in result.events if event["event"] == "DEFERRED_KV_RELEASE")
        self.assertEqual(release["timestamp_ms"], 3)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))

    def test_timeout_during_stage_does_not_emit_output(self):
        result = run(experiment([request(timeout_ms=.5)]))
        self.assertEqual(result.requests[0]["status"], "TIMED_OUT")
        self.assertEqual(result.requests[0]["output_tokens_actual"], 0)
        self.assertEqual(result.result["simulated_ms"], 2)

    def test_completion_wins_same_time_timeout(self):
        result = run(experiment([request(timeout_ms=2)]))
        self.assertEqual(result.requests[0]["status"], "COMPLETED")

    def test_cutoff_retains_partial_stage_and_uncommitted_token(self):
        exp = experiment([request()])
        exp = replace(exp, execution=ExecutionSpec(max_simulated_ms=1.5))
        result = run(exp)
        self.assertEqual(result.result["status"], "UNRESOLVED")
        self.assertEqual(result.result["compute_busy_ms"], 1.5)
        self.assertEqual(result.requests[0]["stage_computed_tokens"], [4, 0])
        self.assertEqual(result.requests[0]["num_computed_tokens"], 0)
        self.assertEqual(result.requests[0]["output_tokens_actual"], 0)
        self.assertTrue(all(rank["used_blocks"] > 0 for rank in result.result["memory"]))

    def test_cutoff_counts_pending_stage_queue_wait(self):
        exp = experiment([request("a"), request("b")], layers=3, stage_layers=(1, 2))
        exp = replace(exp, execution=ExecutionSpec(max_simulated_ms=2.5))
        result = run(exp)
        self.assertEqual(result.result["error_code"], "SIMULATION_TIME_LIMIT")
        self.assertEqual(result.result["pipeline"]["stage_queue_wait_ms"], [1, .5])
        self.assertEqual(result.requests[1]["output_tokens_actual"], 0)

    def test_transfer_still_in_flight_has_no_destination_stage_queue_wait(self):
        exp = experiment([request("a"), request("b")], layers=3, stage_layers=(1, 2))
        exp = replace(exp, strategy=replace(exp.strategy, pipeline_p2p=True),
                      infra=replace(exp.infra, link_latency_us=1000, link_bandwidth_gbps=1),
                      execution=ExecutionSpec(max_simulated_ms=2.5))
        result = run(exp)
        self.assertEqual(result.result["pipeline"]["stage_queue_wait_ms"], [1, 0])

    def test_no_drain_and_excluded_arrivals(self):
        exp = experiment([request("inside", output=3), request("outside", arrival=2)])
        exp = replace(exp, evaluation=EvaluationSpec(measurement_window_ms=2, drain=False))
        result = run(exp)
        self.assertEqual(result.requests[0]["token_times_ms"], [2])
        self.assertEqual(result.requests[1]["status"], "NOT_ADMITTED")
        self.assertEqual(result.result["error_code"], "MEASUREMENT_WINDOW_END")
        self.assertEqual(result.result["admitted_count"], 1)

    def test_event_budget_stops_without_releasing_wave(self):
        exp = experiment([request()])
        exp = replace(exp, execution=ExecutionSpec(max_events=1))
        result = run(exp)
        self.assertEqual(result.result["error_code"], "SIMULATION_EVENT_LIMIT")
        self.assertEqual(result.requests[0]["output_tokens_actual"], 0)
        self.assertTrue(any(rank["used_blocks"] > 0 for rank in result.result["memory"]))

    def test_empty_workload_has_no_busy_time(self):
        result = run(experiment())
        self.assertEqual(result.result["status"], "COMPLETED")
        self.assertEqual(result.result["compute_busy_ms"], 0)
        self.assertIsNone(result.result["pipeline"]["idle_fraction"])

    def test_timeout_during_p2p_retains_references_to_last_stage(self):
        exp = experiment([request(timeout_ms=1.5)])
        exp = replace(exp, strategy=replace(exp.strategy, pipeline_p2p=True),
                      infra=replace(exp.infra, link_latency_us=1000, link_bandwidth_gbps=1))
        result = run(exp)
        self.assertEqual(result.requests[0]["status"], "TIMED_OUT")
        self.assertEqual(result.requests[0]["output_tokens_actual"], 0)
        release = next(event for event in result.events if event["event"] == "DEFERRED_KV_RELEASE")
        self.assertAlmostEqual(release["timestamp_ms"], 3.001024)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))

    def test_cutoff_does_not_charge_future_queued_link_work(self):
        exp = experiment([request(str(i)) for i in range(3)])
        exp = replace(exp, strategy=replace(exp.strategy, pipeline_p2p=True),
                      infra=replace(exp.infra, link_latency_us=10000, link_bandwidth_gbps=1),
                      execution=ExecutionSpec(max_simulated_ms=5))
        result = run(exp)
        self.assertEqual(result.result["compute_busy_ms"], 3)
        self.assertAlmostEqual(result.result["pipeline"]["links"][0]["busy_ms"], 4)
        self.assertEqual(result.result["pipeline"]["inflight_microbatches"], 3)


class PipelinePressureTests(unittest.TestCase):
    def pressure(self, *, policy="recompute", host=65536, fallback="none"):
        return experiment([request("a", prompt=16, output=24, priority=1),
                           request("b", prompt=16, output=24)],
                          preemption=policy, swap_fallback=fallback,
                          memory=MemoryConfig(kv_pool_bytes=6144, cpu_swap_bytes=host,
                                              swap_d2h_bandwidth_gbps=.001,
                                              swap_h2d_bandwidth_gbps=.001,
                                              swap_latency_us=100))

    def test_recompute_quiesces_and_preserves_output_history(self):
        result = run(self.pressure())
        self.assertEqual(result.result["status"], "COMPLETED")
        self.assertEqual([row["output_tokens_actual"] for row in result.requests], [24, 24])
        self.assertGreater(result.result["preemption_count"], 0)
        self.assertGreater(result.result["recomputed_tokens"], 0)
        pending = set()
        for event in result.events:
            if event["event"] == "MICROBATCH_ADMITTED":
                pending.add(event["microbatch_id"])
            elif event["event"] == "FEEDBACK_COMPLETE":
                pending.remove(event["microbatch_id"])
            elif event["event"] == "REQUEST_PREEMPTED":
                self.assertEqual(pending, set())

    def test_swap_moves_kv_without_recompute(self):
        result = run(self.pressure(policy="swap"))
        self.assertEqual(result.result["status"], "COMPLETED")
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual(result.result["recomputed_tokens"], 0)
        self.assertTrue(any(row["swap_out_count"] for row in result.requests))
        self.assertTrue(any(row["swap_in_count"] for row in result.requests))
        self.assertGreater(result.result["swap"]["completed_bytes_by_direction"]["D2H"], 0)
        self.assertEqual(result.result["swap"]["host_reserved_bytes"], 0)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))

    def test_host_capacity_failure_is_not_implicit_recompute(self):
        result = run(self.pressure(policy="swap", host=1))
        self.assertEqual(result.result["error_code"], "HOST_SWAP_CAPACITY_EXCEEDED")
        self.assertEqual(result.result["recomputed_tokens"], 0)

    def test_explicit_recompute_fallback_is_visible(self):
        result = run(self.pressure(policy="swap", host=1, fallback="recompute"))
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertGreater(result.result["recomputed_tokens"], 0)

    def test_no_preemption_is_bounded(self):
        result = run(self.pressure(policy="none"))
        self.assertEqual(result.result["error_code"], "MEMORY_NO_PROGRESS")

    def test_cutoff_during_swap_keeps_both_storage_reservations(self):
        exp = self.pressure(policy="swap")
        full = run(exp)
        transfer = next(event for event in full.events if event["event"] == "SWAP_START"
                        and event["direction"] == "D2H")
        cutoff = (transfer["start_ms"] + transfer["end_ms"]) / 2
        partial = run(replace(exp, execution=ExecutionSpec(max_simulated_ms=cutoff)))
        self.assertEqual(partial.result["status"], "UNRESOLVED")
        self.assertGreater(partial.result["swap"]["host_reserved_bytes"], 0)
        entry = next(row for row in partial.result["swap"]["entries"]
                     if row["request_id"] == transfer["request_id"])
        self.assertEqual(entry["state"], "SWAPPING_OUT")
        self.assertGreater(partial.result["swap"]["dma_busy_ms"], 0)
        request_row = next(row for row in partial.requests if row["request_id"] == transfer["request_id"])
        self.assertEqual(request_row["swap_out_count"], 0)
        self.assertEqual(request_row["swap_out_bytes"], 0)
        self.assertGreater(request_row["swap_ms"], 0)
        self.assertTrue(all(rank["used_blocks"] > 0 for rank in partial.result["memory"]))

    def test_timeout_during_restore_copies_to_boundary_without_new_tokens(self):
        exp = self.pressure(policy="swap")
        full = run(exp)
        transfer = next(event for event in full.events if event["event"] == "SWAP_START"
                        and event["direction"] == "H2D")
        name = transfer["request_id"]
        cutoff = (transfer["start_ms"] + transfer["end_ms"]) / 2
        requests = tuple(replace(r, timeout_ms=cutoff - r.arrival_ms) if r.request_id == name else r
                         for r in exp.workload)
        timed = run(replace(exp, workload=requests))
        row = next(row for row in timed.requests if row["request_id"] == name)
        self.assertEqual(row["status"], "TIMED_OUT")
        self.assertEqual(row["output_tokens_actual"], 1)
        self.assertEqual(timed.result["swap"]["host_reserved_bytes"], 0)
        self.assertGreater(timed.result["swap"]["cancelled_transfer_bytes"], 0)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in timed.result["memory"]))

    def test_swap_data_includes_every_tp_rank_and_stage(self):
        base = self.pressure(policy="swap")
        exp = replace(base,
                      infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(4))),
                      strategy=replace(base.strategy, tensor_parallel_size=2,
                                       memory=replace(base.strategy.memory, kv_pool_bytes=3072)))
        result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 2)
        starts = [event for event in result.events if event["event"] == "SWAP_START"
                  and event["direction"] == "D2H"]
        self.assertEqual(starts[0]["bytes"], 16 * 64 * 4)


class PipelineProfileTests(unittest.TestCase):
    def test_stage_profile_uses_exact_stage_times_and_identity(self):
        exp = experiment([request(output=2)], tp=2)
        with tempfile.TemporaryDirectory() as directory:
            samples = [{"phase": phase, "stage_index": stage, "query_tokens": [q],
                        "context_lengths": [context], "duration_ms": 3,
                        "workspace_peak_bytes": None, "sample_count": 1}
                       for stage in range(2) for phase, q, context in [("prefill", 4, 0), ("decode", 1, 4)]]
            profile = {"schema_version": 1, "profile_id": "pipeline-fixture", "profile_version": "v1",
                       "identity": profile_identity(exp),
                       "source": {"kind": "test_fixture", "reference": "pipeline tests, not GPU data"},
                       "method": "lookup", "timing_includes_communication": True, "samples": samples,
                       "training": {"dataset_sha256": "a" * 64,
                                    "record_ids": [str(i) for i in range(len(samples))],
                                    "group_ids": [str(i) for i in range(len(samples))],
                                    "shape_keys": [shape_key(s["phase"], s["query_tokens"],
                                                             s["context_lengths"], s["stage_index"])
                                                   for s in samples], "parent_profile_version": None}}
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile))
            exp = replace(exp, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                                                           allow_test_fixture=True))
            result = run(exp)
        self.assertEqual(result.requests[0]["token_times_ms"], [6, 12])
        self.assertEqual(result.result["calibration_source"], "test_fixture_profile")
        self.assertFalse(result.result["hardware_validated"])

    def test_unrepresentable_stage_duration_is_rejected(self):
        exp = experiment([request(arrival=1)])
        exp = replace(exp, calibration=replace(exp.calibration, prefill_ms=5e-324))
        with self.assertRaises(SimulationError):
            run(exp)

    def test_positive_feedback_delay_cannot_underflow_to_ideal_zero(self):
        exp = experiment([request(output=2)], pipeline_feedback_latency_us=5e-324)
        with self.assertRaises(SimulationError) as raised:
            run(exp)
        self.assertEqual(raised.exception.code, "INVALID_EVENT_TIME")


class PipelineInvariantTests(unittest.TestCase):
    def test_mixed_transport_swap_and_timeout_traces_keep_ownership(self):
        rng = random.Random(731123)
        for case in range(60):
            tp, pp = rng.choice([1, 2]), rng.choice([2, 3])
            reqs = [request(str(i), prompt=rng.choice([1, 8, 16, 17, 32]),
                            output=rng.randint(1, 25), arrival=rng.randint(0, 4),
                            priority=rng.randrange(3), timeout_ms=rng.choice([None, None, 2, 10, 40]))
                    for i in range(rng.randint(1, 6))]
            memory = MemoryConfig(kv_pool_bytes=(2048 // tp) * rng.randint(1, 6),
                                  cpu_swap_bytes=rng.choice([1, 4096, 16384, 65536]),
                                  swap_d2h_bandwidth_gbps=.01, swap_h2d_bandwidth_gbps=.02,
                                  swap_latency_us=100)
            exp = experiment(reqs, pp=pp, tp=tp, layers=pp, memory=memory,
                             preemption="swap", swap_fallback=rng.choice(["none", "recompute"]),
                             pipeline_microbatch_size=rng.choice([1, 2]),
                             pipeline_feedback_latency_us=rng.choice([0, 100]))
            exp = replace(exp, strategy=replace(exp.strategy, pipeline_p2p=True,
                                                pipeline_max_inflight_microbatches=rng.choice([1, 2, 4])),
                          infra=replace(exp.infra, link_latency_us=10, link_bandwidth_gbps=1),
                          execution=ExecutionSpec(max_simulated_ms=150, max_events=10000))
            with self.subTest(case=case):
                simulation = run(exp)
                report = simulation.result
                self.assertLess(report["events_processed"], 10000)
                self.assertLessEqual(report["swap"]["host_reserved_bytes"], memory.cpu_swap_bytes)
                self.assertTrue(all(0 <= rank["used_blocks"] <= rank["pool_blocks"]
                                    for rank in report["memory"]))
                if report["status"] == "COMPLETED":
                    self.assertEqual(report["swap"]["host_reserved_bytes"], 0)
                    self.assertTrue(all(rank["used_blocks"] == 0 for rank in report["memory"]))
                pending, dma = set(), False
                for event in simulation.events:
                    if event["event"] == "MICROBATCH_ADMITTED":
                        pending.add(event["microbatch_id"])
                    elif event["event"] == "FEEDBACK_COMPLETE":
                        pending.remove(event["microbatch_id"])
                    elif event["event"] == "SWAP_START":
                        self.assertFalse(pending)
                        dma = True
                    elif event["event"] == "SWAP_COMPLETE":
                        dma = False
                    elif event["event"] == "STAGE_START":
                        self.assertFalse(dma)

    def test_generated_traces_are_deterministic_and_conserve_memory(self):
        rng = random.Random(20261104)
        for case in range(40):
            reqs = [request(str(i), prompt=rng.choice([1, 8, 16, 17]), output=rng.randint(1, 12),
                            arrival=rng.randint(0, 6), priority=rng.randint(0, 2),
                            timeout_ms=rng.choice([None, None, 2, 10, 30]))
                    for i in range(rng.randint(1, 5))]
            exp = experiment(reqs, memory=MemoryConfig(kv_pool_bytes=4096 * rng.randint(1, 5)))
            with self.subTest(case=case):
                first = run(exp)
                second = run(replace(exp, execution=replace(exp.execution, detailed_events=False)))
                self.assertEqual(first.result, second.result)
                self.assertEqual(first.requests, second.requests)
                self.assertEqual(second.events, [])
                targets = {r.request_id: r.output_tokens for r in reqs}
                self.assertEqual(first.result["status"], "COMPLETED")
                self.assertTrue(all(rank["used_blocks"] == 0 for rank in first.result["memory"]))
                for row in first.requests:
                    self.assertLessEqual(row["output_tokens_actual"], targets[row["request_id"]])
                    self.assertEqual(row["output_tokens_actual"], len(row["token_times_ms"]))
                times = [event["timestamp_ms"] for event in first.events]
                self.assertEqual(times, sorted(times))


if __name__ == "__main__":
    unittest.main()
