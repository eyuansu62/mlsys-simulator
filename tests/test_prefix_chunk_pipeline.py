"""Chunk and shared-prefix dependencies across pipeline stages and recovery."""

from dataclasses import replace
import random
import unittest

from mlsys_sim.pipeline import PipelineSimulation
from mlsys_sim.schema import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment,
                              InfraSpec, MemoryConfig, ModelSpec, RequestSpec, StrategySpec)


def experiment(requests, *, tp=1, **strategy):
    return Experiment(
        workload=tuple(requests),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(2 * tp))),
        strategy=StrategySpec(model=ModelSpec(num_layers=2), pipeline_parallel_size=2,
                              tensor_parallel_size=tp, pipeline_p2p=False,
                              max_batch_tokens=4, chunked_prefill=True, prefill_chunk_tokens=4,
                              memory=MemoryConfig(kv_block_tokens=4), **strategy),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=2, decode_ms=2,
                                    recompute_ms=2),
    )


def run(exp):
    simulation = PipelineSimulation(exp)
    return simulation, simulation.run()


def admissions(result, name=None):
    return [event for event in result.events if event["event"] == "MICROBATCH_ADMITTED"
            and (name is None or name in event["request_ids"])]


class PipelineChunkTests(unittest.TestCase):
    def test_prompt_that_cannot_fit_alone_fails_before_computing_any_chunks(self):
        exp = experiment([RequestSpec("a", 0, 17, 2)])
        exp = replace(exp, strategy=replace(exp.strategy, memory=MemoryConfig(
            kv_block_tokens=4, kv_pool_bytes=2048)))
        _, result = run(exp)
        self.assertEqual(result.requests[0]["status"], "FAILED")
        self.assertEqual(result.requests[0]["error_code"], "KV_CAPACITY_EXCEEDED")
        self.assertEqual(result.requests[0]["num_computed_tokens"], 0)
        self.assertEqual(admissions(result), [])

    def test_long_prompt_chunks_have_context_and_no_intermediate_token(self):
        simulation, result = run(experiment([RequestSpec("a", 0, 10, 2)]))
        self.assertEqual(result.result["status"], "COMPLETED")
        self.assertEqual(result.requests[0]["token_times_ms"], [6, 8])
        self.assertEqual([e["query_tokens"] for e in admissions(result)], [[4], [4], [2], [1]])
        self.assertEqual([e["context_lengths"] for e in admissions(result)], [[0], [4], [8], [10]])
        self.assertEqual(simulation.requests["a"].prefill_chunks, 3)
        self.assertEqual(simulation.requests["a"].prefill_computed_tokens, 10)
        self.assertEqual(len([e for e in result.events if e["event"] == "SAMPLE_COMPLETE"]), 2)

    def test_each_chunk_waits_for_feedback_before_its_continuation(self):
        _, result = run(experiment([RequestSpec("a", 0, 10, 2)], pipeline_feedback_latency_us=500))
        self.assertEqual([e["timestamp_ms"] for e in admissions(result)], [0, 2.5, 5, 7.5])
        self.assertEqual(result.requests[0]["token_times_ms"], [7, 9.5])
        self.assertEqual(result.result["simulated_ms"], 10)

    def test_decode_priority_precedes_new_prefill_at_feedback_boundary(self):
        _, result = run(experiment([RequestSpec("a", 0, 2, 3), RequestSpec("b", 2, 10, 1)],
                                   pipeline_max_inflight_microbatches=1))
        self.assertEqual([(e["request_ids"], e["phase"]) for e in admissions(result)[:4]],
                         [(["a"], "prefill"), (["a"], "decode"), (["a"], "decode"),
                          (["b"], "prefill")])

    def test_fifo_owner_keeps_all_chunks_until_generation_completes(self):
        _, result = run(experiment([RequestSpec("a", 0, 10, 2), RequestSpec("b", 0, 6, 1)],
                                   scheduler="fifo"))
        self.assertEqual([e["request_ids"] for e in admissions(result)], [["a"]] * 4 + [["b"]] * 2)
        self.assertEqual(result.requests[0]["token_times_ms"], [6, 8])
        self.assertEqual(result.requests[1]["token_times_ms"], [12])

    def test_timeout_on_an_intermediate_chunk_never_samples(self):
        _, result = run(experiment([RequestSpec("a", 0, 10, 2, timeout_ms=3)]))
        self.assertEqual(result.requests[0]["status"], "TIMED_OUT")
        self.assertEqual(result.requests[0]["output_tokens_actual"], 0)
        self.assertEqual(result.requests[0]["num_computed_tokens"], 4)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))

    def test_cutoff_retains_next_chunk_reservation_and_partial_stage_progress(self):
        exp = replace(experiment([RequestSpec("a", 0, 10, 2)]),
                      execution=ExecutionSpec(max_simulated_ms=3.5))
        _, result = run(exp)
        self.assertEqual(result.requests[0]["num_computed_tokens"], 4)
        self.assertEqual(result.requests[0]["stage_computed_tokens"], [8, 4])
        self.assertEqual(result.requests[0]["output_tokens_actual"], 0)
        self.assertTrue(all(rank["used_blocks"] == 2 for rank in result.result["memory"]))

    def test_batch_token_budget_slices_each_member(self):
        exp = experiment([RequestSpec("a", 0, 7, 1), RequestSpec("b", 0, 7, 1)],
                         pipeline_microbatch_size=2)
        exp = replace(exp, strategy=replace(exp.strategy, max_batch_tokens=6))
        _, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual(admissions(result)[0]["query_tokens"], [4, 2])
        self.assertTrue(all(sum(e["query_tokens"]) <= 6 for e in admissions(result)))


class PipelinePrefixTests(unittest.TestCase):
    def test_unadmitted_lookup_cannot_pin_cache_against_a_resident_decoder(self):
        exp = experiment([RequestSpec("a", 0, 8, 1, prompt_token_ids=tuple(range(8))),
                          RequestSpec("b", 11, 4, 5, prompt_token_ids=(20, 21, 22, 23)),
                          RequestSpec("c", 12, 12, 1, prompt_token_ids=tuple(range(12)))],
                         prefix_cache=True, preemption="none", max_batch_size=1,
                         pipeline_max_inflight_microbatches=2)
        exp = replace(exp, strategy=replace(exp.strategy, chunked_prefill=False, max_batch_tokens=32,
                                            memory=MemoryConfig(kv_block_tokens=4, kv_pool_bytes=1536)),
                      calibration=replace(exp.calibration, prefill_ms=10, decode_ms=1))
        simulation, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 3)
        self.assertTrue(any(e["event"] == "PREFIX_CACHE_LOOKUP_ABANDONED" for e in result.events))
        self.assertTrue(all(r.preemptions == 0 for r in simulation.requests.values()))
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 0)

    def test_waiting_prefix_owner_retries_when_resident_survivor_drains(self):
        ids = tuple(range(8))
        exp = experiment([RequestSpec("a", 0, 8, 1, prompt_token_ids=ids),
                          RequestSpec("b", 10, 8, 6, prompt_token_ids=ids),
                          RequestSpec("c", 20, 8, 1, prompt_token_ids=ids)], prefix_cache=True)
        exp = replace(exp, strategy=replace(exp.strategy, memory=MemoryConfig(
            kv_block_tokens=4, kv_pool_bytes=2048)))
        simulation, result = run(exp)
        self.assertEqual(result.result["counts"]["completed"], 3)
        self.assertEqual(simulation.requests["c"].prefix_cached_tokens, 4)
        self.assertGreater(simulation.requests["c"].memory_wait_ms, 0)
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 0)

    def test_zero_batch_budget_does_not_consume_a_later_request_lookup(self):
        simulation, result = run(experiment([
            RequestSpec("a", 0, 4, 1, prompt_token_ids=(0, 1, 2, 3)),
            RequestSpec("b", 0, 5, 1, prompt_token_ids=(0, 1, 2, 3, 4))],
            prefix_cache=True, pipeline_microbatch_size=2, pipeline_max_inflight_microbatches=1))
        self.assertEqual(simulation.requests["b"].prefix_cached_tokens, 4)
        self.assertEqual([e["query_tokens"] for e in admissions(result, "b")], [[1]])

    def test_cache_hits_only_full_prefix_blocks_and_reexecutes_final_prompt_token(self):
        requests = [RequestSpec("a", 0, 8, 1, prompt_token_ids=tuple(range(8))),
                    RequestSpec("b", 10, 8, 1, prompt_token_ids=tuple(range(8))),
                    RequestSpec("c", 20, 9, 1, prompt_token_ids=tuple(range(9)))]
        simulation, result = run(experiment(requests, prefix_cache=True))
        self.assertEqual(result.result["counts"]["completed"], 3)
        self.assertEqual(simulation.requests["a"].prefix_cached_tokens, 0)
        self.assertEqual(simulation.requests["b"].prefix_cached_tokens, 4)
        self.assertEqual(simulation.requests["c"].prefix_cached_tokens, 8)
        self.assertEqual([e["query_tokens"] for e in admissions(result, "b")], [[4]])
        self.assertEqual([e["context_lengths"] for e in admissions(result, "c")], [[8]])
        self.assertEqual([e["query_tokens"] for e in admissions(result, "c")], [[1]])
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 0)

    def test_namespace_changed_tokens_and_missing_identity_do_not_reuse(self):
        ids = tuple(range(8))
        requests = [RequestSpec("a", 0, 8, 1, prompt_token_ids=ids),
                    RequestSpec("b", 10, 8, 1, prompt_token_ids=ids, cache_namespace="tenant-b"),
                    RequestSpec("c", 20, 8, 1, prompt_token_ids=(9, *ids[1:])),
                    RequestSpec("d", 30, 8, 1)]
        simulation, result = run(experiment(requests, prefix_cache=True))
        self.assertEqual(result.result["counts"]["completed"], 4)
        self.assertTrue(all(r.prefix_cached_tokens == 0 for r in simulation.requests.values()))

    def test_prefix_reuse_is_ready_on_every_tp_rank_and_pp_stage(self):
        ids = tuple(range(8))
        simulation, result = run(experiment([RequestSpec("a", 0, 8, 1, prompt_token_ids=ids),
                                             RequestSpec("b", 10, 8, 2, prompt_token_ids=ids)],
                                            prefix_cache=True, tp=2))
        self.assertEqual(len(result.result["memory"]), 4)
        self.assertEqual(simulation.requests["b"].prefix_cached_tokens, 4)
        self.assertEqual(result.requests[1]["stage_computed_tokens"], [9, 9])
        self.assertEqual(result.requests[1]["output_tokens_actual"], 2)

    def test_cancelled_inflight_chunk_does_not_publish_uncommitted_prefix(self):
        simulation, result = run(experiment([RequestSpec("a", 0, 8, 2, timeout_ms=1.5,
                                                          prompt_token_ids=tuple(range(8)))],
                                            prefix_cache=True))
        self.assertEqual(result.requests[0]["status"], "TIMED_OUT")
        self.assertEqual(simulation.memory.prefix_snapshot()["cached_blocks"], 0)

    def test_feedback_timeout_does_not_publish_the_cancelled_chunk(self):
        simulation, result = run(experiment([RequestSpec("a", 0, 8, 2, timeout_ms=2.5,
                                                          prompt_token_ids=tuple(range(8)))],
                                            prefix_cache=True, pipeline_feedback_latency_us=1000))
        self.assertEqual(result.requests[0]["status"], "TIMED_OUT")
        self.assertEqual(simulation.memory.prefix_snapshot()["cached_blocks"], 0)

    def test_cutoff_pins_a_reused_prefix_until_wave_finishes(self):
        ids = tuple(range(8))
        exp = replace(experiment([RequestSpec("a", 0, 8, 1, prompt_token_ids=ids),
                                  RequestSpec("b", 10, 8, 1, prompt_token_ids=ids)], prefix_cache=True),
                      execution=ExecutionSpec(max_simulated_ms=11.5))
        simulation, result = run(exp)
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 1)
        self.assertEqual(result.requests[1]["num_computed_tokens"], 4)
        self.assertEqual(result.requests[1]["stage_computed_tokens"], [8, 4])
        self.assertEqual(result.requests[1]["output_tokens_actual"], 0)


class PipelineChunkRecoveryTests(unittest.TestCase):
    @staticmethod
    def pressure(preemption, *, cache=False):
        requests = [RequestSpec(name, 0, 12, 2, prompt_token_ids=tuple(range(12)) if cache else ())
                    for name in ("a", "b")]
        exp = experiment(requests, preemption=preemption, prefix_cache=cache)
        return replace(exp, strategy=replace(exp.strategy, memory=MemoryConfig(
            kv_block_tokens=4, kv_pool_bytes=2048, cpu_swap_bytes=65536)))

    def test_swap_preserves_partial_prompt_before_any_token_exists(self):
        simulation, result = run(self.pressure("swap"))
        self.assertEqual(result.result["counts"]["completed"], 2)
        swaps = [event for event in result.events if event["event"] == "SWAP_START"]
        self.assertEqual([(e["direction"], e["tokens"]) for e in swaps], [("D2H", 8), ("H2D", 8)])
        self.assertEqual(result.result["recomputed_tokens"], 0)
        self.assertEqual(simulation.requests["b"].prefill_computed_tokens, 12)
        self.assertTrue(all(row["output_tokens_actual"] == 2 for row in result.requests))
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))
        self.assertTrue(all(row["recompute_target"] == 0 for row in result.requests))

    def test_recompute_restores_partial_prompt_in_chunks_without_emitting_tokens(self):
        simulation, result = run(self.pressure("recompute"))
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual(simulation.requests["a"].recomputed_tokens, 8)
        recovery = [e for e in admissions(result, "a") if e["phase"] == "recompute"]
        self.assertEqual([e["query_tokens"] for e in recovery], [[4], [4]])
        self.assertEqual([e["context_lengths"] for e in recovery], [[0], [4]])
        self.assertTrue(all(row["output_tokens_actual"] == 2 for row in result.requests))
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))

    def test_cache_and_swap_can_share_prompt_then_release_all_request_references(self):
        simulation, result = run(self.pressure("swap", cache=True))
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual(simulation.memory.prefix_snapshot()["pinned_blocks"], 0)
        self.assertEqual(simulation.memory.prefix_snapshot()["private_blocks"], 0)
        self.assertTrue(all(row["output_tokens_actual"] == 2 for row in result.requests))

    def test_generated_chunk_cache_transport_and_timeout_traces_conserve_ownership(self):
        rng = random.Random(74)
        for iteration in range(80):
            requests = [RequestSpec(str(i), rng.randrange(8), rng.randrange(1, 18), rng.randrange(1, 5),
                                    timeout_ms=rng.choice((None, None, 2, 9)))
                        for i in range(rng.randrange(1, 6))]
            requests = [replace(row, prompt_token_ids=tuple(range(row.input_tokens))) for row in requests]
            exp = experiment(requests, prefix_cache=bool(iteration % 2),
                             preemption=rng.choice(("none", "recompute", "swap")),
                             pipeline_feedback_latency_us=rng.choice((0, 500)))
            exp = replace(exp, strategy=replace(exp.strategy, pipeline_p2p=bool(iteration % 3),
                                                memory=MemoryConfig(kv_block_tokens=4,
                                                    kv_pool_bytes=rng.choice((2048, 3072, 4096)),
                                                    cpu_swap_bytes=65536)),
                          execution=ExecutionSpec(max_simulated_ms=80, max_events=10000))
            with self.subTest(iteration=iteration):
                simulation, result = run(exp)
                for rank in result.result["memory"]:
                    self.assertGreaterEqual(rank["used_blocks"], 0)
                    self.assertLessEqual(rank["used_blocks"], rank["pool_blocks"])
                cache = simulation.memory.prefix_snapshot()
                self.assertGreaterEqual(cache["private_blocks"], 0)
                for row, spec in zip(result.requests, sorted(exp.workload, key=lambda r: r.request_id)):
                    self.assertLessEqual(row["output_tokens_actual"], spec.output_tokens)
                    self.assertEqual(len(row["token_times_ms"]), row["output_tokens_actual"])
                    self.assertTrue(all(b > a for a, b in zip(row["token_times_ms"], row["token_times_ms"][1:])))
                self.assertLess(result.result["events_processed"], 10000)
