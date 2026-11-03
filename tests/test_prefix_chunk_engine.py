"""Single-group prefix reuse, partial-prompt scheduling and recovery contracts."""

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, EvaluationSpec, ExecutionSpec,
                       Experiment, InfraSpec, MemoryConfig, RequestSpec, StrategySpec,
                       calibrate_dataset, profile_identity, simulate)
from mlsys_sim.engine import _Simulation


def experiment(requests, **options):
    defaults = dict(chunked_prefill=True, prefill_chunk_tokens=8, max_batch_tokens=16,
                    max_batch_size=2)
    defaults.update(options)
    return Experiment(workload=tuple(requests), strategy=StrategySpec(**defaults),
                      calibration=CalibrationSpec(backend="synthetic", prefill_ms=2,
                                                  decode_ms=1, recompute_ms=2))


def starts(run, phase=None):
    return [event for event in run.events if event["event"] == "STEP_START"
            and (phase is None or event["phase"] == phase)]


def pressure(policy="recompute", **options):
    return experiment([RequestSpec("a", 0, 48, 2), RequestSpec("b", 0, 48, 2)],
                      preemption=policy,
                      memory=MemoryConfig(kv_pool_bytes=32768, cpu_swap_bytes=65536,
                                          swap_d2h_bandwidth_gbps=.01,
                                          swap_h2d_bandwidth_gbps=.01,
                                          swap_latency_us=0), **options)


class ChunkedPrefillTests(unittest.TestCase):
    def test_long_prompt_uses_bounded_queries_and_only_final_chunk_samples(self):
        run = simulate(experiment([RequestSpec("a", 0, 35, 3)]))
        steps = starts(run, "prefill")
        self.assertEqual([event["query_tokens"] for event in steps], [[8], [8], [8], [8], [3]])
        self.assertEqual([event["context_lengths"] for event in steps], [[0], [8], [16], [24], [32]])
        self.assertEqual(run.requests[0]["token_times_ms"], [10, 11, 12])
        self.assertEqual(run.requests[0]["prefill_chunks"], 5)
        self.assertEqual(run.requests[0]["prefill_computed_tokens"], 35)
        self.assertEqual(run.requests[0]["num_computed_tokens"], 37)
        self.assertEqual(run.result["status"], "COMPLETED")

    def test_disabled_chunking_retains_explicit_batch_limit_failure(self):
        run = simulate(experiment([RequestSpec("a", 0, 35, 3)], chunked_prefill=False))
        self.assertEqual(run.requests[0]["error_code"], "BATCH_TOKEN_LIMIT_EXCEEDED")
        self.assertEqual(starts(run), [])

    def test_chunk_budget_splits_batch_without_exceeding_global_token_budget(self):
        run = simulate(experiment([RequestSpec("a", 0, 9, 1), RequestSpec("b", 0, 9, 1)],
                                  prefill_chunk_tokens=8, max_batch_tokens=10))
        self.assertEqual(starts(run)[0]["query_tokens"], [8, 2])
        self.assertTrue(all(sum(event["query_tokens"]) <= 10 for event in starts(run)))
        self.assertEqual([r["output_tokens_actual"] for r in run.requests], [1, 1])

    def test_decode_has_priority_over_new_prefill_at_same_timestamp(self):
        run = simulate(experiment([RequestSpec("a", 0, 8, 3), RequestSpec("b", 2, 24, 1)]))
        self.assertEqual(run.requests[0]["token_times_ms"], [2, 3, 4])
        self.assertEqual([event["phase"] for event in starts(run)[:3]],
                         ["prefill", "decode", "decode"])
        self.assertEqual(run.requests[1]["first_token_ms"], 10)

    def test_fifo_retains_owner_through_every_chunk(self):
        run = simulate(experiment([RequestSpec("a", 0, 24, 2), RequestSpec("b", 1, 8, 1)],
                                  scheduler="fifo"))
        self.assertEqual([event["request_ids"] for event in starts(run)],
                         [["a"], ["a"], ["a"], ["a"], ["b"]])
        self.assertEqual(run.requests[0]["token_times_ms"], [6, 7])
        self.assertEqual(run.requests[1]["token_times_ms"], [9])

    def test_chunking_does_not_hide_a_final_context_that_cannot_fit_alone(self):
        run = simulate(experiment([RequestSpec("a", 0, 49, 1)],
                                  memory=MemoryConfig(kv_pool_bytes=24576)))
        self.assertEqual(run.requests[0]["error_code"], "KV_CAPACITY_EXCEEDED")
        self.assertEqual(starts(run), [])

    def test_cutoff_retains_reserved_fragment_without_committing_it(self):
        base = experiment([RequestSpec("a", 0, 35, 1)])
        run = simulate(replace(base, execution=ExecutionSpec(max_simulated_ms=3)))
        self.assertEqual(run.requests[0]["num_computed_tokens"], 8)
        self.assertEqual(run.requests[0]["prefill_chunks"], 1)
        self.assertEqual(run.requests[0]["prefill_computed_tokens"], 8)
        self.assertEqual(run.requests[0]["output_tokens_actual"], 0)
        self.assertEqual(run.result["compute_busy_ms"], 3)
        self.assertEqual(run.result["memory"][0]["used_blocks"], 1)

    def test_timeout_during_second_chunk_waits_for_safe_release(self):
        run = simulate(experiment([RequestSpec("a", 0, 35, 1, timeout_ms=3)]))
        self.assertEqual(run.requests[0]["status"], "TIMED_OUT")
        self.assertEqual(run.requests[0]["num_computed_tokens"], 8)
        self.assertEqual(run.requests[0]["output_tokens_actual"], 0)
        finish = next(e for e in run.events if e["event"] == "REQUEST_FINISH")
        release = next(e for e in run.events if e["event"] == "DEFERRED_KV_RELEASE")
        self.assertTrue(finish["kv_release_deferred"])
        self.assertEqual(release["timestamp_ms"], 4)
        self.assertEqual(run.result["memory"][0]["used_blocks"], 0)

    def test_partial_prefill_recompute_restores_progress_without_early_output(self):
        run = simulate(pressure())
        a = run.requests[0]
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertEqual(a["recomputed_tokens"], 32)
        self.assertEqual(a["prefill_computed_tokens"], 48)
        self.assertEqual(a["prefill_chunks"], 6)
        self.assertEqual(a["token_times_ms"], [25, 26])
        self.assertEqual([event["context_lengths"] for event in starts(run, "recompute")],
                         [[0], [8], [16], [24]])
        self.assertTrue(all(r["output_tokens_actual"] == 2 for r in run.requests))
        self.assertEqual(run.result["memory"][0]["used_blocks"], 0)

    def test_recovery_rejects_individually_infeasible_suffix_workspace_without_repreemption(self):
        # Seed a safe pressure-recovery boundary. The Profile deliberately has
        # small recompute Workspace and infeasible suffix-prefill Workspace.
        for prompt in (32, 600):
            with self.subTest(prompt=prompt), tempfile.TemporaryDirectory() as directory:
                base = experiment([RequestSpec("a", 0, prompt, 1)],
                                  chunked_prefill=False, max_batch_tokens=1024)
                dataset = {"schema_version": 1, "dataset_id": "recovery-workspace",
                           "identity": profile_identity(base),
                           "source": {"kind": "test_fixture", "reference": "unit-test shape evidence"},
                           "records": [
                               {"record_id": "r", "group_id": "r", "phase": "recompute",
                                "stage_index": 0, "query_tokens": [16], "context_lengths": [0],
                                "duration_ms": 2, "workspace_peak_bytes": 0},
                               {"record_id": "p", "group_id": "p", "phase": "prefill",
                                "stage_index": 0, "query_tokens": [prompt - 16], "context_lengths": [16],
                                "duration_ms": 2, "workspace_peak_bytes": 100 * 1024 * 1024}]}
                path = Path(directory) / "profile.json"
                path.write_text(json.dumps(calibrate_dataset(dataset, profile_version="recovery-v1",
                                                               allow_test_fixture=True)))
                base = replace(base, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                                                                  allow_test_fixture=True))
                engine = _Simulation(base)
                request = engine.requests["a"]
                request.status = "WAITING_RECOMPUTE"
                request.recompute_target = 16
                chosen, _, _ = engine._choose([request], "recompute")
                self.assertEqual(chosen, [])
                self.assertEqual(request.status, "FAILED")
                self.assertEqual(request.error_code, "WORKSPACE_BUDGET_EXCEEDED")
                self.assertEqual(request.preemptions, 0)
                self.assertEqual(engine.memory.allocations, {})

    def test_partial_prefill_swap_restores_continuation_without_recompute(self):
        run = simulate(pressure("swap"))
        a = run.requests[0]
        copies = [e for e in run.events if e["event"] == "SWAP_START"]
        self.assertEqual([e["tokens"] for e in copies], [32, 32])
        self.assertEqual([e["direction"] for e in copies], ["D2H", "H2D"])
        self.assertEqual(run.result["recomputed_tokens"], 0)
        self.assertEqual(a["prefill_computed_tokens"], 48)
        self.assertEqual(a["prefill_chunks"], 6)
        self.assertAlmostEqual(a["first_token_ms"], 20.2768)
        self.assertEqual([r["output_tokens_actual"] for r in run.requests], [2, 2])
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)
        self.assertEqual(run.result["memory"][0]["used_blocks"], 0)

    def test_partial_swap_cancellation_keeps_buffers_until_completion(self):
        base = pressure("swap")
        run = simulate(replace(base, workload=(replace(base.workload[0], timeout_ms=9),
                                               base.workload[1])))
        self.assertEqual(run.requests[0]["status"], "TIMED_OUT")
        self.assertEqual(run.requests[0]["output_tokens_actual"], 0)
        self.assertEqual(run.requests[1]["status"], "COMPLETED")
        copies = [e for e in run.events if e["event"] == "SWAP_COMPLETE"]
        self.assertEqual(len(copies), 1)
        self.assertTrue(copies[0]["cancelled"])
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)

    def test_measurement_window_cutoff_keeps_incomplete_prompt(self):
        base = experiment([RequestSpec("a", 0, 35, 1)])
        run = simulate(replace(base, evaluation=EvaluationSpec(measurement_window_ms=3, drain=False)))
        self.assertEqual(run.result["error_code"], "MEASUREMENT_WINDOW_END")
        self.assertEqual(run.requests[0]["num_computed_tokens"], 8)
        self.assertEqual(run.requests[0]["output_tokens_actual"], 0)

    def test_tp_chunk_copy_traffic_includes_all_ranks(self):
        base = pressure("swap")
        base = replace(base,
                       infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}") for i in range(4))),
                       strategy=replace(base.strategy, tensor_parallel_size=4,
                                        memory=replace(base.strategy.memory, kv_pool_bytes=16384)))
        run = simulate(base)
        copies = [e for e in run.events if e["event"] == "SWAP_START"]
        self.assertEqual([e["bytes"] for e in copies], [32768, 32768])
        self.assertEqual(run.result["counts"]["completed"], 2)


class PrefixCacheEngineTests(unittest.TestCase):
    def test_prefix_hit_reduces_work_and_keeps_one_last_query_for_sampling(self):
        ids = tuple(range(32))
        run = simulate(experiment([RequestSpec("a", 0, 32, 1, prompt_token_ids=ids),
                                   RequestSpec("b", 20, 32, 1, prompt_token_ids=ids)],
                                  prefix_cache=True))
        a, b = run.requests
        self.assertEqual(a["prefix_cached_tokens"], 0)
        self.assertEqual(b["prefix_cached_tokens"], 16)
        self.assertEqual(b["prefill_computed_tokens"], 16)
        self.assertEqual(b["prefill_chunks"], 2)
        self.assertEqual(b["ttft_ms"], 4)
        self.assertEqual(b["output_tokens_actual"], 1)
        self.assertEqual([e["context_lengths"] for e in starts(run, "prefill") if e["request_ids"] == ["b"]],
                         [[16], [24]])

    def test_exhausted_batch_budget_defers_lookup_until_request_is_considered(self):
        ids = tuple(range(33))
        run = simulate(experiment([RequestSpec("a", 0, 33, 1, prompt_token_ids=ids),
                                   RequestSpec("b", 0, 33, 1, prompt_token_ids=ids)],
                                  prefix_cache=True, prefill_chunk_tokens=16,
                                  max_batch_tokens=16))
        self.assertEqual(run.requests[1]["prefix_cached_tokens"], 32)
        lookups = [e for e in run.events if e["event"] == "PREFIX_CACHE_LOOKUP"]
        self.assertEqual([(e["request_id"], e["timestamp_ms"]) for e in lookups],
                         [("a", 0), ("b", 6)])

    def test_nonchunked_prefix_hit_processes_only_uncached_suffix(self):
        ids = tuple(range(33))
        run = simulate(experiment([RequestSpec("a", 0, 33, 1, prompt_token_ids=ids),
                                   RequestSpec("b", 20, 33, 1, prompt_token_ids=ids)],
                                  prefix_cache=True, chunked_prefill=False, max_batch_tokens=64))
        self.assertEqual(run.requests[1]["prefix_cached_tokens"], 32)
        self.assertEqual(run.requests[1]["prefill_computed_tokens"], 1)
        self.assertEqual(starts(run, "prefill")[-1]["query_tokens"], [1])
        self.assertEqual(starts(run, "prefill")[-1]["context_lengths"], [32])

    def test_identity_missing_different_or_namespace_isolated_never_false_hits(self):
        ids = tuple(range(33))
        workloads = [RequestSpec("a", 0, 33, 1, prompt_token_ids=ids),
                     RequestSpec("b", 20, 33, 1, prompt_token_ids=ids, cache_namespace="tenant"),
                     RequestSpec("c", 40, 33, 1, prompt_token_ids=(99,) + ids[1:]),
                     RequestSpec("d", 60, 33, 1)]
        run = simulate(experiment(workloads, prefix_cache=True))
        self.assertEqual([r["prefix_cached_tokens"] for r in run.requests], [0, 0, 0, 0])
        self.assertEqual([r["prefill_computed_tokens"] for r in run.requests], [33, 33, 33, 33])

    def test_failed_suffix_admission_unpins_cache_and_allows_resident_decode_to_progress(self):
        for chunking in (False, True):
            with self.subTest(chunking=chunking):
                base = experiment([
                    RequestSpec("a", 0, 8, 1, prompt_token_ids=tuple(range(8))),
                    RequestSpec("b", 11, 4, 5, prompt_token_ids=tuple(range(20, 24))),
                    RequestSpec("c", 12, 12, 1, prompt_token_ids=tuple(range(12)))],
                    prefix_cache=True, chunked_prefill=chunking, prefill_chunk_tokens=512,
                    max_batch_size=1, max_batch_tokens=64, preemption="none",
                    memory=MemoryConfig(kv_block_tokens=4, kv_pool_bytes=6144))
                base = replace(base, calibration=replace(base.calibration, prefill_ms=10))
                run = simulate(base)
                self.assertEqual(run.result["status"], "COMPLETED")
                self.assertEqual(run.result["counts"]["completed"], 3)
                self.assertEqual(run.result["preemption_count"], 0)
                self.assertEqual(run.result["prefix_cache"]["pinned_blocks"], 0)
                if not chunking:
                    self.assertTrue(any(e["event"] == "PREFIX_CACHE_LOOKUP_ABANDONED" for e in run.events))
                self.assertGreater(run.requests[2]["first_token_ms"], run.requests[1]["completion_ms"])

    def test_completed_chunks_can_be_shared_before_producer_finishes_prompt(self):
        ids = tuple(range(49))
        run = simulate(experiment([RequestSpec("a", 0, 49, 1, prompt_token_ids=ids),
                                   RequestSpec("b", 4, 33, 1, prompt_token_ids=ids[:33])],
                                  prefix_cache=True, prefill_chunk_tokens=16,
                                  max_batch_tokens=32, max_batch_size=2))
        self.assertGreaterEqual(run.requests[1]["prefix_cached_tokens"], 16)
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertEqual(run.requests[1]["prefill_computed_tokens"],
                         33 - run.requests[1]["prefix_cached_tokens"])

    def test_recompute_republishes_completed_prompt_blocks_and_restores_references(self):
        base = pressure()
        base = replace(base, workload=(replace(base.workload[0], prompt_token_ids=tuple(range(48))),
                                       replace(base.workload[1], prompt_token_ids=(99,) * 48)),
                       strategy=replace(base.strategy, prefix_cache=True))
        run = simulate(base)
        republished = [e for e in run.events if e["event"] == "PREFIX_CACHE_PUBLISH"
                       and e.get("phase") == "recompute"]
        self.assertEqual([(e["computed_tokens"], e["published_blocks"]) for e in republished],
                         [(8, 0), (16, 1), (24, 0), (32, 1)])
        continuation = next(e for e in starts(run, "recompute") if e["context_lengths"] == [16])
        self.assertGreaterEqual(continuation["memory"][0]["prefix_pinned_blocks"], 1)
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertEqual(run.result["prefix_cache"]["pinned_blocks"], 0)
        self.assertEqual(run.result["prefix_cache"]["private_blocks"], 0)
        self.assertEqual(run.requests[0]["prefill_computed_tokens"], 48)

    def test_warmup_does_not_seed_measured_prefix_cache(self):
        ids = tuple(range(33))
        spec = RequestSpec("a", 0, 33, 1, prompt_token_ids=ids)
        base = experiment([spec], prefix_cache=True)
        run = simulate(replace(base, execution=ExecutionSpec(warmup=(spec,))))
        self.assertEqual(run.requests[0]["prefix_cached_tokens"], 0)
        self.assertEqual(run.requests[0]["prefill_computed_tokens"], 33)
        self.assertEqual(run.result["warmup"]["status"], "COMPLETED")

    def test_cutoff_never_publishes_reserved_uncomputed_prompt_block(self):
        ids = tuple(range(33))
        base = experiment([RequestSpec("a", 0, 33, 1, prompt_token_ids=ids)],
                          prefix_cache=True, prefill_chunk_tokens=16)
        run = simulate(replace(base, execution=ExecutionSpec(max_simulated_ms=1)))
        publications = [e for e in run.events if e["event"] == "PREFIX_CACHE_PUBLISH"]
        self.assertEqual(publications, [])
        self.assertEqual(run.requests[0]["prefill_computed_tokens"], 0)

    def test_generated_traces_preserve_token_and_pool_bounds(self):
        rng = random.Random(5049)
        for case in range(100):
            workloads = []
            for i in range(rng.randrange(1, 6)):
                count = rng.randrange(1, 65)
                ids = tuple(range(count)) if rng.random() < .7 else ()
                workloads.append(RequestSpec(f"r{i}", float(rng.randrange(12)), count,
                                             rng.randrange(1, 12), rng.randrange(3),
                                             rng.choice([None, None, 5., 30.]), ids,
                                             rng.choice(["", "", "other"])))
            base = experiment(workloads, prefix_cache=True,
                              prefill_chunk_tokens=rng.randrange(1, 20),
                              max_batch_tokens=rng.randrange(1, 30),
                              preemption=rng.choice(["recompute", "swap", "none"]),
                              memory=MemoryConfig(kv_pool_bytes=8192 * rng.randrange(1, 8),
                                                  cpu_swap_bytes=65536))
            base = replace(base, execution=ExecutionSpec(max_simulated_ms=200, max_events=1000))
            with self.subTest(case=case):
                run = simulate(base)
                oracle = {r.request_id: r for r in workloads}
                for row in run.requests:
                    self.assertLessEqual(row["output_tokens_actual"], oracle[row["request_id"]].output_tokens)
                    self.assertEqual(len(row["token_times_ms"]), row["output_tokens_actual"])
                    self.assertEqual(row["token_times_ms"], sorted(set(row["token_times_ms"])))
                    if row["status"] == "COMPLETED":
                        self.assertEqual(row["output_tokens_actual"], oracle[row["request_id"]].output_tokens)
                        self.assertEqual(row["prefill_computed_tokens"] + row["prefix_cached_tokens"],
                                         oracle[row["request_id"]].input_tokens)
                for rank in run.result["memory"]:
                    self.assertGreaterEqual(rank["used_blocks"], 0)
                    self.assertLessEqual(rank["used_blocks"], rank["pool_blocks"])
                timestamps = [e["timestamp_ms"] for e in run.events]
                self.assertEqual(timestamps, sorted(timestamps))
                self.assertTrue(all(sum(e["query_tokens"]) <= base.strategy.max_batch_tokens
                                    for e in starts(run)))


if __name__ == "__main__":
    unittest.main()
