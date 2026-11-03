"""Joint single-group serving batches preserve per-member execution state."""

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment, InfraSpec,
                       MemoryConfig, MemoryPlanner, RequestSpec, SimulationError, StrategySpec,
                       calibrate_dataset, profile_identity, simulate)
from mlsys_sim.engine import _Simulation


def experiment(requests, **options):
    strategy = dict(mixed_batching=True, chunked_prefill=True, prefill_chunk_tokens=4,
                    max_batch_tokens=5, max_batch_size=2)
    strategy.update(options)
    return Experiment(workload=tuple(requests), strategy=StrategySpec(**strategy),
                      calibration=CalibrationSpec(backend="synthetic", prefill_ms=2,
                                                  decode_ms=1, recompute_ms=2, mixed_ms=3))


def golden(**options):
    return experiment([RequestSpec("a", 0, 2, 3), RequestSpec("b", 2, 5, 2)], **options)


def starts(run, phase=None):
    return [event for event in run.events if event["event"] == "STEP_START"
            and (phase is None or event["phase"] == phase)]


class MixedEngineTests(unittest.TestCase):
    def test_joint_step_golden_with_no_intermediate_prefill_output(self):
        engine = _Simulation(golden())
        run = engine.run()
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertEqual([event["phase"] for event in starts(run)],
                         ["prefill", "mixed", "mixed", "decode"])
        mixed = starts(run, "mixed")
        self.assertEqual([event["request_ids"] for event in mixed], [["a", "b"], ["a", "b"]])
        self.assertEqual([event["request_phases"] for event in mixed],
                         [["decode", "prefill"], ["decode", "prefill"]])
        self.assertEqual([event["query_tokens"] for event in mixed], [[1, 4], [1, 1]])
        self.assertEqual([event["context_lengths"] for event in mixed], [[2, 0], [3, 4]])
        self.assertEqual([row["token_times_ms"] for row in run.requests], [[2, 5, 8], [8, 9]])
        self.assertEqual(run.requests[1]["prefill_computed_tokens"], 5)
        self.assertEqual(engine.mixed_batch_count, 2)
        self.assertEqual(engine.mixed_batch_completed_count, 2)
        self.assertEqual(engine.mixed_compute_busy_ms, 6)
        self.assertEqual(run.result["compute_busy_ms"], 9)

    def test_mixing_is_opt_in_and_fifo_still_owns_a_single_request(self):
        unmixed = simulate(golden(mixed_batching=False))
        self.assertFalse(starts(unmixed, "mixed"))
        self.assertEqual(unmixed.requests[0]["token_times_ms"], [2, 3, 4])
        fifo = simulate(golden(scheduler="fifo"))
        self.assertFalse(starts(fifo, "mixed"))
        self.assertTrue(all(len(event["request_ids"]) == 1 for event in starts(fifo)))
        self.assertEqual([event["request_ids"][0] for event in starts(fifo)[:3]], ["a"] * 3)

    def test_decode_members_take_budget_before_prefill_priority(self):
        base = experiment([RequestSpec("a", 0, 1, 3, priority=0),
                           RequestSpec("b", 0, 1, 3, priority=0),
                           RequestSpec("high", 2, 10, 1, priority=100)],
                          max_batch_size=3, max_batch_tokens=4, prefill_chunk_tokens=4)
        run = simulate(base)
        first = starts(run, "mixed")[0]
        self.assertEqual(first["request_ids"], ["a", "b", "high"])
        self.assertEqual(first["request_phases"], ["decode", "decode", "prefill"])
        self.assertEqual(first["query_tokens"], [1, 1, 2])
        self.assertTrue(all(sum(event["query_tokens"]) <= 4 for event in starts(run)))

    def test_whole_prefill_can_mix_without_chunking(self):
        run = simulate(experiment([RequestSpec("a", 0, 2, 3), RequestSpec("b", 2, 3, 2)],
                                  chunked_prefill=False, max_batch_tokens=4))
        self.assertEqual(starts(run, "mixed")[0]["query_tokens"], [1, 3])
        self.assertEqual([row["token_times_ms"] for row in run.requests], [[2, 5, 6], [5, 6]])

    def test_whole_prefill_does_not_get_truncated_to_leftover_budget(self):
        run = simulate(golden(chunked_prefill=False))
        self.assertFalse(starts(run, "mixed"))
        b = [event for event in starts(run) if "b" in event["request_ids"]]
        self.assertEqual(b[0]["query_tokens"], [5])
        self.assertEqual(run.requests[1]["status"], "COMPLETED")
        self.assertEqual(run.requests[1]["prefill_chunks"], 1)

    def test_batch_size_one_uses_homogeneous_timing(self):
        run = simulate(golden(max_batch_size=1))
        self.assertFalse(starts(run, "mixed"))
        self.assertEqual(run.requests[0]["token_times_ms"], [2, 3, 4])

    def test_joint_workspace_failure_trims_prefill_and_keeps_decode_progress(self):
        base = golden(memory=MemoryConfig(workspace_base_bytes=0, workspace_bytes_per_token=100,
                                         runtime_bytes=0, safety_margin_bytes=0))
        fixed = MemoryPlanner(base).snapshot()[0]["fixed_budget_bytes"]
        base = replace(base, infra=InfraSpec(devices=(DeviceSpec(memory_bytes=fixed + 400),)))
        run = simulate(base)
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertFalse(starts(run, "mixed"))
        self.assertEqual(run.requests[0]["token_times_ms"], [2, 3, 4])
        self.assertTrue(all(rank["budget_peak_bytes"] <= rank["device_memory_bytes"]
                            for rank in run.result["memory"]))

    def test_single_member_workspace_failure_is_explicit_without_preemption(self):
        base = golden(memory=MemoryConfig(workspace_base_bytes=0, workspace_bytes_per_token=100,
                                         runtime_bytes=0, safety_margin_bytes=0))
        fixed = MemoryPlanner(base).snapshot()[0]["fixed_budget_bytes"]
        base = replace(base, infra=InfraSpec(devices=(DeviceSpec(memory_bytes=fixed + 100),)))
        run = simulate(base)
        self.assertEqual(run.result["counts"]["failed"], 2)
        self.assertEqual([row["error_code"] for row in run.requests],
                         ["WORKSPACE_BUDGET_EXCEEDED"] * 2)
        self.assertEqual(run.result["preemption_count"], 0)

    def test_cutoff_keeps_both_member_reservations_and_partial_mixed_service(self):
        engine = _Simulation(replace(golden(), execution=ExecutionSpec(max_simulated_ms=3)))
        run = engine.run()
        self.assertEqual([row["num_computed_tokens"] for row in run.requests], [2, 0])
        self.assertEqual([row["output_tokens_actual"] for row in run.requests], [1, 0])
        self.assertEqual(engine.memory.allocations, {"a": 3, "b": 4})
        self.assertEqual(engine.mixed_batch_count, 1)
        self.assertEqual(engine.mixed_batch_completed_count, 0)
        self.assertEqual(engine.mixed_compute_busy_ms, 1)
        self.assertEqual(run.result["compute_busy_ms"], 3)

    def test_timeout_of_decoder_releases_only_at_joint_completion(self):
        base = golden()
        base = replace(base, workload=(replace(base.workload[0], timeout_ms=3), base.workload[1]))
        run = simulate(base)
        self.assertEqual(run.requests[0]["status"], "TIMED_OUT")
        self.assertEqual(run.requests[0]["token_times_ms"], [2])
        self.assertEqual(run.requests[1]["status"], "COMPLETED")
        release = [event for event in run.events if event["event"] == "DEFERRED_KV_RELEASE"]
        self.assertEqual([(event["request_id"], event["timestamp_ms"]) for event in release], [("a", 5)])
        self.assertEqual(run.result["memory"][0]["used_blocks"], 0)

    def test_timeout_of_prefill_does_not_emit_its_intermediate_token(self):
        base = golden()
        base = replace(base, workload=(base.workload[0], replace(base.workload[1], timeout_ms=1)))
        run = simulate(base)
        self.assertEqual(run.requests[1]["status"], "TIMED_OUT")
        self.assertEqual(run.requests[1]["output_tokens_actual"], 0)
        self.assertEqual(run.requests[1]["prefill_chunks"], 0)
        self.assertEqual(run.requests[0]["token_times_ms"], [2, 5, 6])
        self.assertEqual(run.result["memory"][0]["used_blocks"], 0)

    def test_prefix_hit_one_token_suffix_keeps_prefill_role_in_joint_batch(self):
        run = simulate(experiment([
            RequestSpec("seed", 0, 8, 1, prompt_token_ids=tuple(range(8))),
            RequestSpec("a", 5, 1, 5),
            RequestSpec("repeat", 7, 9, 1, prompt_token_ids=tuple(range(9)))],
            prefix_cache=True, memory=MemoryConfig(kv_block_tokens=4)))
        suffix = next(event for event in starts(run, "mixed") if "repeat" in event["request_ids"])
        self.assertEqual(suffix["request_phases"], ["decode", "prefill"])
        self.assertEqual(suffix["query_tokens"], [1, 1])
        self.assertEqual(suffix["context_lengths"], [1, 8])
        self.assertEqual(run.requests[2]["prefix_cached_tokens"], 8)
        self.assertEqual(run.requests[2]["prefill_computed_tokens"], 1)
        self.assertEqual(run.result["prefix_cache"]["pinned_blocks"], 0)
        self.assertEqual(run.result["prefix_cache"]["private_blocks"], 0)

    def test_failed_prefix_suffix_admission_does_not_pin_resident_decoder(self):
        base = experiment([
            RequestSpec("a", 0, 8, 1, prompt_token_ids=tuple(range(8))),
            RequestSpec("b", 11, 4, 5, prompt_token_ids=tuple(range(20, 24))),
            RequestSpec("c", 12, 12, 1, prompt_token_ids=tuple(range(12)))],
            prefix_cache=True, chunked_prefill=False, max_batch_tokens=64,
            preemption="none", memory=MemoryConfig(kv_block_tokens=4, kv_pool_bytes=6144))
        base = replace(base, calibration=replace(base.calibration, prefill_ms=10))
        run = simulate(base)
        self.assertEqual(run.result["counts"]["completed"], 3)
        self.assertEqual(run.result["preemption_count"], 0)
        self.assertEqual(run.result["prefix_cache"]["pinned_blocks"], 0)
        self.assertTrue(any(event["event"] == "PREFIX_CACHE_LOOKUP_ABANDONED" for event in run.events))

    def test_mixed_partial_prefill_recovers_with_swap_or_recompute_and_later_prefix_hit(self):
        for policy in ("swap", "recompute"):
            with self.subTest(policy=policy):
                base = experiment([
                    RequestSpec("a", 0, 8, 25, prompt_token_ids=tuple(range(8))),
                    RequestSpec("b", 2, 48, 2, prompt_token_ids=tuple(range(48))),
                    RequestSpec("c", 2, 48, 2, prompt_token_ids=tuple(range(100, 148))),
                    RequestSpec("d", 80, 48, 2, prompt_token_ids=tuple(range(48)))],
                    prefix_cache=True, preemption=policy, prefill_chunk_tokens=8,
                    max_batch_tokens=16, max_batch_size=3,
                    memory=MemoryConfig(kv_pool_bytes=32768, cpu_swap_bytes=262144, swap_latency_us=0))
                run = simulate(base)
                self.assertEqual(run.result["counts"]["completed"], 4)
                self.assertEqual([row["output_tokens_actual"] for row in run.requests], [25, 2, 2, 2])
                self.assertEqual(run.requests[3]["prefix_cached_tokens"], 32)
                self.assertEqual(run.requests[3]["prefill_computed_tokens"], 16)
                self.assertEqual(run.result["batching"]["mixed_batches_admitted"], 6)
                self.assertEqual(run.result["prefix_cache"]["pinned_blocks"], 0)
                self.assertEqual(run.result["prefix_cache"]["private_blocks"], 0)
                if policy == "swap":
                    copies = [event for event in run.events if event["event"] == "SWAP_START"]
                    self.assertEqual([(event["request_id"], event["tokens"], event["direction"])
                                      for event in copies], [("b", 32, "D2H"), ("b", 32, "H2D")])
                    self.assertEqual(run.result["recomputed_tokens"], 0)
                    self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)
                else:
                    self.assertEqual(run.result["recomputed_tokens"], 32)
                    self.assertTrue(starts(run, "recompute"))
                    self.assertTrue(all("request_phases" not in event for event in starts(run, "recompute")))

    def profile(self, base, include_mixed=True):
        shapes = [("prefill", [2], [0], None, 2, 100), ("decode", [1], [5], None, 1, 100)]
        if include_mixed:
            shapes.extend([("mixed", [1, 4], [2, 0], ["decode", "prefill"], 3, 500),
                           ("mixed", [1, 1], [3, 4], ["decode", "prefill"], 3, 200)])
        records = []
        for i, (phase, queries, contexts, members, duration, workspace) in enumerate(shapes):
            row = {"record_id": str(i), "group_id": str(i), "phase": phase,
                   "query_tokens": queries, "context_lengths": contexts, "stage_index": 0,
                   "duration_ms": duration, "workspace_peak_bytes": workspace}
            if members is not None:
                row["request_phases"] = members
            records.append(row)
        return calibrate_dataset({"schema_version": 1, "dataset_id": "mixed-golden",
            "identity": profile_identity(base), "source": {"kind": "test_fixture",
            "reference": "Hand-calculated mixed stage times, not hardware"}, "records": records},
            profile_version="mixed-test", allow_test_fixture=True)

    def test_joint_profile_does_not_require_unused_homogeneous_member_shapes(self):
        base = golden()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(self.profile(base)))
            run = simulate(replace(base, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                                      profile_version="mixed-test", allow_test_fixture=True)))
        self.assertEqual([row["token_times_ms"] for row in run.requests], [[2, 5, 8], [8, 9]])
        self.assertEqual(run.result["memory"][0]["workspace_peak_bytes"], 500)
        self.assertTrue(all(event["source"] == "test_fixture_profile" for event in starts(run)))

    def test_missing_joint_profile_propagates_without_homogeneous_timing_fallback(self):
        base = golden()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(self.profile(base, include_mixed=False)))
            with self.assertRaises(SimulationError) as raised:
                simulate(replace(base, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                                 profile_version="mixed-test", allow_test_fixture=True)))
        self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_tp_mixed_batch_uses_one_collective_payload_for_all_queries(self):
        base = replace(golden(),
            workload=(golden().workload[0], replace(golden().workload[1], arrival_ms=.000001)),
            infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(4))),
            calibration=CalibrationSpec(),
            strategy=replace(golden().strategy, tensor_parallel_size=4))
        run = simulate(base)
        mixed = starts(run, "mixed")
        self.assertTrue(mixed)
        for event in mixed:
            communication = event["breakdown"]["communication"]
            self.assertEqual(communication["allreduce_count"], 8)
            self.assertEqual(communication["allreduce_payload_bytes"],
                             sum(event["query_tokens"]) * 128 * 2)
        self.assertEqual(run.result["counts"]["completed"], 2)

    def test_generated_cache_chunk_recovery_traces_preserve_outputs_and_storage(self):
        rng = random.Random(11033)
        for case in range(120):
            requests = []
            for i in range(rng.randrange(1, 7)):
                count = rng.randrange(1, 49)
                ids = tuple(range(count)) if rng.random() < .8 else ()
                requests.append(RequestSpec(str(i), rng.randrange(15), count, rng.randrange(1, 15),
                    rng.randrange(3), rng.choice((None, None, 4, 15, 60)), ids, rng.choice(("", "", "tenant"))))
            base = experiment(requests, prefix_cache=True, chunked_prefill=bool(case % 3),
                prefill_chunk_tokens=rng.randrange(1, 17), max_batch_tokens=rng.randrange(1, 25),
                max_batch_size=rng.randrange(1, 5), preemption=rng.choice(("none", "recompute", "swap")),
                memory=MemoryConfig(kv_pool_bytes=8192 * rng.randrange(1, 8), cpu_swap_bytes=65536))
            base = replace(base, execution=ExecutionSpec(max_simulated_ms=300, max_events=2000))
            with self.subTest(case=case):
                run = simulate(base)
                oracle = {request.request_id: request for request in requests}
                for row in run.requests:
                    target = oracle[row["request_id"]]
                    self.assertLessEqual(row["output_tokens_actual"], target.output_tokens)
                    self.assertEqual(len(row["token_times_ms"]), row["output_tokens_actual"])
                    self.assertEqual(row["token_times_ms"], sorted(set(row["token_times_ms"])))
                    if row["status"] == "COMPLETED":
                        self.assertEqual(row["output_tokens_actual"], target.output_tokens)
                        self.assertEqual(row["prefill_computed_tokens"] + row["prefix_cached_tokens"],
                                         target.input_tokens)
                for rank in run.result["memory"]:
                    self.assertGreaterEqual(rank["used_blocks"], 0)
                    self.assertLessEqual(rank["used_blocks"] + rank["guard_blocks"], rank["pool_blocks"])
                for event in starts(run):
                    self.assertLessEqual(sum(event["query_tokens"]), base.strategy.max_batch_tokens)
                    self.assertLessEqual(len(event["request_ids"]), base.strategy.max_batch_size)
                if run.result["status"] == "COMPLETED":
                    self.assertEqual(run.result["prefix_cache"]["pinned_blocks"], 0)
                    self.assertEqual(run.result["prefix_cache"]["private_blocks"], 0)
                    if "swap" in run.result:
                        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
