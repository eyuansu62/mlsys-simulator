"""Joint Prefill/Decode service across PP stages, cache and recovery."""

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

from mlsys_sim import MemoryPlanner, SimulationError, calibrate_dataset, profile_identity
from mlsys_sim.pipeline import PipelineSimulation
from mlsys_sim.schema import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment,
                              InfraSpec, MemoryConfig, ModelSpec, RequestSpec, StrategySpec)


def experiment(requests, *, tp=1, **strategy):
    controls = dict(model=ModelSpec(num_layers=2), pipeline_parallel_size=2,
                    tensor_parallel_size=tp, pipeline_p2p=False,
                    pipeline_microbatch_size=2, pipeline_max_inflight_microbatches=1,
                    max_batch_size=2, max_batch_tokens=4, mixed_batching=True,
                    chunked_prefill=True, prefill_chunk_tokens=4,
                    memory=MemoryConfig(kv_block_tokens=4))
    controls.update(strategy)
    return Experiment(
        workload=tuple(requests),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(2 * tp))),
        strategy=StrategySpec(**controls),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=2, decode_ms=2,
                                    recompute_ms=2, mixed_ms=3),
    )


def run(exp):
    simulation = PipelineSimulation(exp)
    return simulation, simulation.run()


def admissions(result):
    return [event for event in result.events if event["event"] == "MICROBATCH_ADMITTED"]


def golden_requests():
    return [RequestSpec("a", 0, 2, 3), RequestSpec("b", 2, 5, 1)]


class MixedPipelineTests(unittest.TestCase):
    def test_one_joint_wave_uses_member_roles_and_only_final_prefill_samples(self):
        simulation, result = run(experiment(golden_requests()))
        self.assertEqual(result.result["counts"]["completed"], 2)
        batches = admissions(result)
        self.assertEqual([event["phase"] for event in batches], ["prefill", "mixed", "mixed"])
        self.assertEqual([event["request_ids"] for event in batches[1:]], [["a", "b"], ["a", "b"]])
        self.assertEqual([event["request_phases"] for event in batches[1:]],
                         [["decode", "prefill"], ["decode", "prefill"]])
        self.assertEqual([event["query_tokens"] for event in batches[1:]], [[1, 3], [1, 2]])
        self.assertEqual([event["context_lengths"] for event in batches[1:]], [[2, 0], [3, 3]])
        self.assertEqual([row["token_times_ms"] for row in result.requests], [[2, 5, 8], [8]])
        self.assertEqual(result.requests[1]["prefill_chunks"], 2)
        self.assertEqual(simulation.mixed_batch_count, 2)
        self.assertEqual(simulation.mixed_batch_completed_count, 2)
        self.assertEqual(simulation.mixed_compute_busy_ms, 6)
        stages = [event for event in result.events if event["event"] == "STAGE_START"
                  and event["phase"] == "mixed"]
        self.assertEqual(len(stages), 4)
        self.assertTrue(all(event["duration_ms"] == 1.5 for event in stages))
        self.assertTrue(all(event["request_phases"] == ["decode", "prefill"] for event in stages))

    def test_mixed_members_wait_for_feedback_before_continuing(self):
        _, result = run(experiment(golden_requests(), pipeline_feedback_latency_us=500))
        self.assertEqual([event["timestamp_ms"] for event in admissions(result)], [0, 2.5, 6])
        self.assertEqual([row["token_times_ms"] for row in result.requests], [[2, 5.5, 9], [9]])
        self.assertEqual(result.result["simulated_ms"], 9.5)

    def test_tp_pp_payload_and_each_rank_share_the_same_atomic_wave(self):
        exp = experiment(golden_requests(), tp=2, pipeline_p2p=True)
        simulation, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual(len(result.result["memory"]), 4)
        self.assertGreater(simulation.mixed_batch_count, 0)
        batches = admissions(result)
        expected_bytes = sum(sum(event["query_tokens"]) for event in batches) * 128 * 2 * 2
        self.assertEqual(result.result["pipeline"]["links"][0]["completed_bytes"], expected_bytes)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))

    def test_whole_prefill_suffix_is_not_silently_chunked_when_chunking_is_disabled(self):
        exp = experiment([RequestSpec("a", 0, 2, 3), RequestSpec("b", 2, 4, 1)],
                         chunked_prefill=False)
        simulation, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual(simulation.mixed_batch_count, 0)
        self.assertEqual([event["query_tokens"] for event in admissions(result)
                          if "b" in event["request_ids"]], [[4]])

    def test_mixing_also_supports_whole_prompt_when_it_fits_remaining_budget(self):
        exp = experiment([RequestSpec("a", 0, 2, 2), RequestSpec("b", 2, 3, 1)],
                         chunked_prefill=False)
        simulation, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual(simulation.mixed_batch_count, 1)
        self.assertEqual(admissions(result)[1]["query_tokens"], [1, 3])

    def test_fifo_and_single_member_microbatches_keep_their_boundaries(self):
        for controls in [{"scheduler": "fifo"}, {"pipeline_microbatch_size": 1}]:
            with self.subTest(controls=controls):
                simulation, result = run(experiment(golden_requests(), **controls))
                self.assertEqual(result.result["counts"]["completed"], 2)
                self.assertEqual(simulation.mixed_batch_count, 0)
                self.assertTrue(all(len(event["request_ids"]) == 1 for event in admissions(result)))

    def test_disabled_mixing_preserves_homogeneous_decode_priority(self):
        simulation, result = run(experiment(golden_requests(), mixed_batching=False))
        self.assertEqual(simulation.mixed_batch_count, 0)
        self.assertEqual([event["phase"] for event in admissions(result)[:3]],
                         ["prefill", "decode", "decode"])
        self.assertEqual(result.requests[0]["token_times_ms"], [2, 4, 6])

    def test_cutoff_counts_partial_joint_service_and_keeps_member_reservations(self):
        exp = replace(experiment(golden_requests()), execution=ExecutionSpec(max_simulated_ms=4.25))
        simulation, result = run(exp)
        self.assertEqual(result.result["error_code"], "SIMULATION_TIME_LIMIT")
        self.assertEqual(simulation.mixed_batch_count, 1)
        self.assertEqual(simulation.mixed_batch_completed_count, 0)
        self.assertEqual(simulation.mixed_compute_busy_ms, 2.25)
        self.assertEqual([row["num_computed_tokens"] for row in result.requests], [2, 0])
        self.assertEqual([row["stage_computed_tokens"] for row in result.requests], [[3, 2], [3, 0]])
        self.assertEqual([row["output_tokens_actual"] for row in result.requests], [1, 0])
        self.assertTrue(all(rank["used_blocks"] == 2 for rank in result.result["memory"]))

    def test_timeout_retains_member_until_shared_wave_feedback(self):
        requests = [RequestSpec("a", 0, 2, 3), RequestSpec("b", 2, 5, 1, timeout_ms=1)]
        _, result = run(experiment(requests))
        self.assertEqual(result.requests[1]["status"], "TIMED_OUT")
        self.assertEqual(result.requests[1]["output_tokens_actual"], 0)
        finish = next(event for event in result.events if event["event"] == "REQUEST_FINISH"
                      and event["request_id"] == "b")
        released = next(event for event in result.events if event["event"] == "DEFERRED_KV_RELEASE"
                        and event["request_id"] == "b")
        self.assertTrue(finish["kv_release_deferred"])
        self.assertEqual(released["timestamp_ms"], 5)
        self.assertEqual(result.requests[0]["output_tokens_actual"], 3)

    def test_prefix_hit_joins_decoder_and_publishes_only_its_prompt_role(self):
        ids = tuple(range(8))
        requests = [RequestSpec("seed", 0, 8, 1, prompt_token_ids=ids),
                    RequestSpec("a", 5, 2, 4, prompt_token_ids=(20, 21)),
                    RequestSpec("b", 7, 8, 1, prompt_token_ids=ids)]
        simulation, result = run(experiment(requests, prefix_cache=True, max_batch_tokens=6))
        self.assertEqual(result.result["counts"]["completed"], 3)
        self.assertEqual(simulation.requests["b"].prefix_cached_tokens, 4)
        mixed = [event for event in admissions(result) if event["phase"] == "mixed"]
        self.assertEqual(mixed[0]["request_ids"], ["a", "b"])
        self.assertEqual(mixed[0]["query_tokens"], [1, 4])
        self.assertEqual(mixed[0]["context_lengths"], [2, 4])
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 0)
        self.assertEqual(simulation.memory.prefix_snapshot()["private_blocks"], 0)
        self.assertEqual(len([event for event in result.events if event["event"] == "PREFIX_CACHE_PUBLISH"
                              and event["request_id"] == "a"]), 1)

    def test_cutoff_keeps_shared_prefix_pinned_for_its_mixed_wave(self):
        ids = tuple(range(8))
        requests = [RequestSpec("seed", 0, 8, 1, prompt_token_ids=ids),
                    RequestSpec("a", 5, 2, 3), RequestSpec("b", 7, 8, 1, prompt_token_ids=ids)]
        exp = replace(experiment(requests, prefix_cache=True, max_batch_tokens=6),
                      execution=ExecutionSpec(max_simulated_ms=9.25))
        simulation, result = run(exp)
        self.assertEqual(simulation.mixed_batch_count, 1)
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 1)
        self.assertEqual(simulation.requests["b"].computed, 4)
        row = next(row for row in result.requests if row["request_id"] == "b")
        self.assertEqual(row["stage_computed_tokens"], [8, 4])
        self.assertEqual(row["output_tokens_actual"], 0)

    def test_cancelled_mixed_prefill_never_publishes_its_prospective_suffix(self):
        ids = tuple(range(8))
        requests = [RequestSpec("seed", 0, 8, 1, prompt_token_ids=ids),
                    RequestSpec("a", 5, 2, 3),
                    RequestSpec("b", 7, 8, 1, timeout_ms=2, prompt_token_ids=ids)]
        simulation, result = run(experiment(requests, prefix_cache=True, max_batch_tokens=6))
        self.assertEqual(simulation.requests["b"].status, "TIMED_OUT")
        self.assertEqual(simulation.requests["b"].generated, 0)
        self.assertEqual(simulation.requests["b"].computed, 4)
        self.assertFalse(any(event["event"] == "PREFIX_CACHE_PUBLISH" and event["request_id"] == "b"
                             for event in result.events))
        released = next(event for event in result.events if event["event"] == "DEFERRED_KV_RELEASE"
                        and event["request_id"] == "b")
        self.assertEqual(released["timestamp_ms"], 10)
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 0)

    def test_rejected_shared_prefix_trial_cannot_block_selected_decoder(self):
        requests = [RequestSpec("a", 0, 8, 1, prompt_token_ids=tuple(range(8))),
                    RequestSpec("b", 11, 4, 5, prompt_token_ids=(20, 21, 22, 23)),
                    RequestSpec("c", 12, 12, 1, prompt_token_ids=tuple(range(12)))]
        exp = experiment(requests, prefix_cache=True, preemption="none", chunked_prefill=False,
                         max_batch_tokens=32,
                         memory=MemoryConfig(kv_block_tokens=4, kv_pool_bytes=1536))
        simulation, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 3)
        self.assertTrue(any(event["event"] == "PREFIX_CACHE_LOOKUP_ABANDONED" for event in result.events))
        self.assertTrue(all(request.preemptions == 0 for request in simulation.requests.values()))
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 0)

    def test_joint_workspace_failure_can_drop_prefill_without_fake_mixed_timing(self):
        exp = experiment([RequestSpec("a", 0, 2, 3), RequestSpec("b", 2, 3, 1)],
                         memory=MemoryConfig(kv_block_tokens=4, workspace_base_bytes=0,
                                             workspace_bytes_per_token=100))
        ranks = MemoryPlanner(exp).snapshot()
        exp = replace(exp, infra=replace(exp.infra, devices=tuple(
            replace(device, memory_bytes=rank["fixed_budget_bytes"] + 300)
            for device, rank in zip(exp.infra.devices, ranks))))
        simulation, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual(simulation.mixed_batch_count, 0)
        self.assertTrue(all(rank["workspace_peak_bytes"] == 300 for rank in result.result["memory"]))


class MixedPipelineRecoveryTests(unittest.TestCase):
    def test_mixed_partial_prompt_roundtrips_cpu_swap_with_prefix_cache(self):
        requests = [RequestSpec("a", 0, 4, 6, prompt_token_ids=(20, 21, 22, 23)),
                    RequestSpec("b", 2, 12, 1, prompt_token_ids=tuple(range(12)))]
        exp = experiment(requests, prefix_cache=True, preemption="swap",
                         memory=MemoryConfig(kv_block_tokens=4, kv_pool_bytes=2048, cpu_swap_bytes=65536))
        simulation, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertGreater(simulation.mixed_batch_count, 0)
        self.assertEqual(result.result["recomputed_tokens"], 0)
        self.assertEqual([row["output_tokens_actual"] for row in result.requests], [6, 1])
        swaps = [event for event in result.events if event["event"] == "SWAP_START"]
        self.assertTrue(any(event["direction"] == "D2H" and event["request_id"] == "b"
                            and event["tokens"] < 12 for event in swaps))
        self.assertEqual(simulation.swap.host_reserved_bytes, 0)
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 0)
        self.assertEqual(simulation.memory.prefix_snapshot()["private_blocks"], 0)

    def test_recompute_recovery_stays_homogeneous_after_mixed_service(self):
        requests = [RequestSpec("a", 0, 4, 6), RequestSpec("b", 2, 12, 1)]
        exp = experiment(requests, preemption="recompute",
                         memory=MemoryConfig(kv_block_tokens=4, kv_pool_bytes=2048))
        simulation, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertGreater(simulation.mixed_batch_count, 0)
        self.assertGreater(result.result["recomputed_tokens"], 0)
        self.assertEqual([row["output_tokens_actual"] for row in result.requests], [6, 1])
        self.assertTrue(all(event.get("request_phases") is None for event in admissions(result)
                            if event["phase"] == "recompute"))

    def test_generated_transport_cache_and_recovery_traces_conserve_tokens(self):
        rng = random.Random(137)
        for case in range(100):
            requests = [RequestSpec(str(index), rng.randrange(7), rng.randrange(1, 15),
                                    rng.randrange(1, 6), timeout_ms=rng.choice((None, None, 2, 12)))
                        for index in range(rng.randrange(1, 6))]
            requests = [replace(row, prompt_token_ids=tuple(range(row.input_tokens))) for row in requests]
            exp = experiment(requests, tp=rng.choice((1, 2)), prefix_cache=bool(case % 2),
                preemption=rng.choice(("none", "recompute", "swap")), pipeline_p2p=bool(case % 3),
                pipeline_feedback_latency_us=rng.choice((0, 500)),
                pipeline_max_inflight_microbatches=rng.choice((1, 2, 4)),
                memory=MemoryConfig(kv_block_tokens=4, kv_pool_bytes=rng.choice((2048, 3072, 4096)),
                                    cpu_swap_bytes=65536))
            exp = replace(exp, execution=ExecutionSpec(max_simulated_ms=100, max_events=10000))
            with self.subTest(case=case):
                simulation, result = run(exp)
                for rank in result.result["memory"]:
                    self.assertGreaterEqual(rank["used_blocks"], 0)
                    self.assertLessEqual(rank["used_blocks"], rank["pool_blocks"])
                self.assertGreaterEqual(simulation.memory.prefix_snapshot()["private_blocks"], 0)
                for row, spec in zip(result.requests, exp.workload):
                    self.assertLessEqual(row["output_tokens_actual"], spec.output_tokens)
                    self.assertEqual(len(row["token_times_ms"]), row["output_tokens_actual"])
                    self.assertTrue(all(right > left for left, right in
                                        zip(row["token_times_ms"], row["token_times_ms"][1:])))
                    if row["status"] == "COMPLETED":
                        self.assertEqual(row["prefill_computed_tokens"] + row["prefix_cached_tokens"],
                                         spec.input_tokens)
                self.assertGreaterEqual(simulation.mixed_batch_count, simulation.mixed_batch_completed_count)
                self.assertLessEqual(simulation.mixed_compute_busy_ms, simulation.compute_busy_ms + 1e-9)
                if result.result["status"] == "COMPLETED":
                    self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 0)
                    self.assertEqual(simulation.memory.prefix_snapshot()["private_blocks"], 0)


class MixedPipelineProfileTests(unittest.TestCase):
    def profile_data(self, exp, *, mixed):
        records = []
        for stage in range(2):
            records.append({"record_id": f"prefill-{stage}", "group_id": f"prefill-{stage}",
                "phase": "prefill", "stage_index": stage, "query_tokens": [2],
                "context_lengths": [0], "duration_ms": 1, "workspace_peak_bytes": 100})
            if mixed:
                records.append({"record_id": f"mixed-{stage}", "group_id": f"mixed-{stage}",
                    "phase": "mixed", "request_phases": ["decode", "prefill"], "stage_index": stage,
                    "query_tokens": [1, 3], "context_lengths": [2, 0], "duration_ms": 1.5,
                    "workspace_peak_bytes": 400 + 200 * stage})
            else:
                records.append({"record_id": f"decode-{stage}", "group_id": f"decode-{stage}",
                    "phase": "decode", "stage_index": stage, "query_tokens": [1],
                    "context_lengths": [2], "duration_ms": 1, "workspace_peak_bytes": 100})
        return {"schema_version": 1, "dataset_id": "mixed-pipeline-fixture",
                "identity": profile_identity(exp), "source": {"kind": "test_fixture",
                "reference": "Hand-calculated timing fixture, not hardware evidence"}, "records": records}

    def test_exact_joint_profile_supplies_each_stage_time_and_workspace(self):
        exp = experiment([RequestSpec("a", 0, 2, 2), RequestSpec("b", 2, 3, 1)], tp=2)
        profile = calibrate_dataset(self.profile_data(exp, mixed=True), profile_version="joint",
                                    allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile))
            exp = replace(exp, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                          profile_version="joint", allow_test_fixture=True))
            simulation, result = run(exp)
        self.assertEqual([row["token_times_ms"] for row in result.requests], [[2, 5], [5]])
        self.assertEqual(simulation.mixed_batch_count, 1)
        self.assertEqual([rank["workspace_peak_bytes"] for rank in result.result["memory"]],
                         [400, 400, 600, 600])

    def test_missing_joint_shape_never_uses_pure_phase_profiles(self):
        exp = experiment([RequestSpec("a", 0, 2, 2), RequestSpec("b", 2, 3, 1)])
        profile = calibrate_dataset(self.profile_data(exp, mixed=False), profile_version="pure-only",
                                    allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile))
            for extrapolate in [False, True]:
                configured = replace(exp, calibration=CalibrationSpec(backend="profile",
                    profile_path=str(path), profile_version="pure-only", allow_test_fixture=True,
                    allow_extrapolation=extrapolate))
                with self.subTest(extrapolate=extrapolate), self.assertRaises(SimulationError) as raised:
                    run(configured)
                self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")


if __name__ == "__main__":
    unittest.main()
