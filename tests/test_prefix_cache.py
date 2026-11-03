"""Exact-identity prefix sharing, bounded ownership and atomic cache reclamation."""

from dataclasses import replace
import unittest

from mlsys_sim.layout import rank_kv_bytes_per_token
from mlsys_sim.memory import MemoryPlanner
from mlsys_sim.schema import (
    DeviceSpec, Experiment, InfraSpec, MemoryConfig, RequestSpec, SimulationError, StrategySpec,
)
from mlsys_sim.swap import HostSwapManager


def request(name, tokens, *, namespace="", identity=True):
    return RequestSpec(name, 0, len(tokens), 1, prompt_token_ids=tuple(tokens) if identity else (),
                       cache_namespace=namespace)


def fixture(requests=(), *, blocks=8, cache_blocks=16, tp=1, pp=1, layers=(), workspace=0):
    strategy = StrategySpec(prefix_cache=True, prefix_cache_max_blocks=cache_blocks,
                            tensor_parallel_size=tp, pipeline_parallel_size=pp, stage_layers=layers)
    exp = Experiment(workload=tuple(requests), strategy=strategy,
                     infra=InfraSpec(devices=tuple(DeviceSpec(id=f"gpu{i}") for i in range(tp * pp))))
    memory = MemoryConfig(kv_block_tokens=4,
                          kv_pool_bytes=rank_kv_bytes_per_token(exp, 0) * 4 * blocks,
                          workspace_base_bytes=128, workspace_bytes_per_token=workspace,
                          runtime_bytes=64, safety_margin_bytes=64, cpu_swap_bytes=1024 * 1024)
    return replace(exp, strategy=replace(strategy, memory=memory))


def populate(planner, spec, *, release=True):
    assert planner.reserve({spec.request_id: spec.input_tokens}, [spec.input_tokens], "prefill").ok
    planner.publish_prefix(spec.request_id, spec.input_tokens)
    if release:
        planner.release(spec.request_id)


class PrefixCacheTests(unittest.TestCase):
    def test_disabled_cache_preserves_private_accounting(self):
        spec = request("a", range(8))
        exp = fixture([spec])
        planner = MemoryPlanner(replace(exp, strategy=replace(exp.strategy, prefix_cache=False)))
        populate(planner, spec)
        self.assertEqual(planner.lookup_prefix("a"), 0)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 0)
        self.assertFalse(planner.prefix_snapshot()["enabled"])

    def test_equal_lengths_and_suffixes_do_not_establish_identity(self):
        a = request("a", range(8))
        b = request("b", (20, *range(1, 8)))
        c = request("c", range(8), namespace="another-tenant")
        d = request("d", range(8), identity=False)
        planner = MemoryPlanner(fixture([a, b, c, d]))
        populate(planner, a)
        for spec in (b, c, d):
            self.assertEqual(planner.lookup_prefix(spec.request_id), 0)
        cache = planner.prefix_snapshot()
        self.assertEqual(cache["miss_requests"], 2)
        self.assertEqual(cache["bypassed_requests"], 1)
        self.assertEqual(cache["cached_blocks"], 2)

    def test_direct_dataclass_list_tokens_are_normalized_for_cache_keys(self):
        a, b = request("a", range(9)), request("b", range(9))
        a = replace(a, prompt_token_ids=list(a.prompt_token_ids))
        b = replace(b, prompt_token_ids=list(b.prompt_token_ids))
        planner = MemoryPlanner(fixture([a, b]))
        populate(planner, a)
        self.assertEqual(planner.lookup_prefix("b"), 8)

    def test_direct_planner_rejects_invalid_token_identity_without_lookup_side_effects(self):
        for tokens in [(True, 1, 2, 3), (-1, 1, 2, 3), (256, 1, 2, 3), (2**63, 1, 2, 3)]:
            spec = request("a", tokens)
            planner = MemoryPlanner(fixture([spec]))
            before = planner.prefix_snapshot()
            with self.assertRaises(SimulationError) as error:
                planner.lookup_prefix("a")
            self.assertEqual(error.exception.code, "INVALID_CONFIG")
            self.assertEqual(planner.prefix_snapshot(), before)

    def test_long_prompts_store_constant_size_ancestry_digests_per_block(self):
        spec = request("a", tuple(index % 256 for index in range(20000)))
        planner = MemoryPlanner(fixture([spec]))
        self.assertEqual(planner.lookup_prefix("a"), 0)
        keys = planner._prefix_request_keys["a"]
        self.assertEqual(len(keys), 5000)
        self.assertTrue(all(type(key[1]) is int and len(key[2]) == 32 for key in keys))

    def test_abandoned_trial_hit_unpins_cache_and_can_be_retried(self):
        a, b = request("a", range(8)), request("b", range(9))
        planner = MemoryPlanner(fixture([a, b], blocks=3))
        populate(planner, a)
        self.assertEqual(planner.lookup_prefix("b"), 8)
        self.assertFalse(planner.check({"other": 8}, [8], "prefill").ok)
        planner.abandon_prefix_lookup("b")
        self.assertNotIn("b", planner.allocations)
        cache = planner.prefix_snapshot()
        self.assertEqual(cache["lookup_requests"], 0)
        self.assertEqual(cache["hit_requests"], 0)
        self.assertEqual(cache["hit_tokens"], 0)
        self.assertEqual(cache["abandoned_lookups"], 1)
        self.assertEqual(cache["abandoned_hit_tokens"], 8)
        self.assertEqual(cache["pinned_blocks"], 0)
        self.assertTrue(planner.check({"other": 8}, [8], "prefill").ok)
        self.assertEqual(planner.lookup_prefix("b"), 8)
        self.assertEqual(planner.prefix_snapshot()["hit_requests"], 1)

    def test_abandon_never_detaches_an_admitted_or_published_prefix(self):
        for operation in ("reserve", "reserve_kv", "publish"):
            with self.subTest(operation=operation):
                a, b = request("a", range(8)), request("b", range(9))
                planner = MemoryPlanner(fixture([a, b]))
                populate(planner, a)
                self.assertEqual(planner.lookup_prefix("b"), 8)
                if operation == "reserve":
                    planner.reserve({"b": 9}, [1], "prefill")
                elif operation == "reserve_kv":
                    planner.reserve_kv({"b": 8})
                else:
                    planner.publish_prefix("b", 8)
                before, cache = planner.snapshot(), planner.prefix_snapshot()
                with self.assertRaises(SimulationError) as error:
                    planner.abandon_prefix_lookup("b")
                self.assertEqual(error.exception.code, "INVALID_PREFIX_STATE")
                self.assertEqual(planner.snapshot(), before)
                self.assertEqual(planner.prefix_snapshot(), cache)

    def test_full_hit_keeps_final_prompt_token_for_logits(self):
        a, b = request("a", range(8)), request("b", range(8))
        c = request("c", range(9))
        planner = MemoryPlanner(fixture([a, b, c]))
        populate(planner, a)
        self.assertEqual(planner.lookup_prefix("b"), 4)
        self.assertEqual(planner.lookup_prefix("c"), 8)
        self.assertEqual(planner.allocations, {"b": 4, "c": 8})
        self.assertEqual(planner.prefix_snapshot()["hit_tokens"], 12)

    def test_longest_match_is_contiguous_and_block_aligned(self):
        a = request("a", range(12))
        b = request("b", (*range(4), 99, *range(5, 12)))
        planner = MemoryPlanner(fixture([a, b]))
        populate(planner, a)
        self.assertEqual(planner.lookup_prefix("b"), 4)

    def test_partial_blocks_and_generated_tokens_are_not_published(self):
        a = request("a", range(9))
        b = request("b", range(10))
        planner = MemoryPlanner(fixture([a, b]))
        planner.reserve({"a": 5}, [5], "prefill")
        self.assertEqual(planner.publish_prefix("a", 5), 1)
        planner.reserve({"a": 12}, [7], "prefill")
        self.assertEqual(planner.publish_prefix("a", 12), 1)
        self.assertEqual(planner.publish_prefix("a", 12), 0)
        self.assertEqual(planner.prefix_snapshot()["cached_blocks"], 2)
        self.assertEqual(planner.lookup_prefix("b"), 8)

    def test_shared_blocks_have_one_physical_owner_and_many_references(self):
        a, b = request("a", range(9)), request("b", range(9))
        planner = MemoryPlanner(fixture([a, b]))
        populate(planner, a, release=False)
        self.assertEqual(planner.lookup_prefix("b"), 8)
        planner.reserve({"b": 9}, [1], "prefill")
        snapshot = planner.snapshot()[0]
        self.assertEqual(snapshot["used_blocks"], 4)
        self.assertEqual(snapshot["private_blocks"], 2)
        self.assertEqual(snapshot["prefix_shared_blocks"], 2)
        planner.release("a")
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 3)
        self.assertEqual(planner.prefix_snapshot()["pinned_blocks"], 2)
        planner.release("b")
        planner.release("b")
        cache = planner.prefix_snapshot()
        self.assertEqual(cache["private_blocks"], 0)
        self.assertEqual(cache["pinned_blocks"], 0)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 2)

    def test_concurrent_misses_deduplicate_only_after_computation(self):
        a, b = request("a", range(8)), request("b", range(8))
        planner = MemoryPlanner(fixture([a, b]))
        self.assertEqual(planner.lookup_prefix("a"), 0)
        self.assertEqual(planner.lookup_prefix("b"), 0)
        planner.reserve({"a": 8, "b": 8}, [8, 8], "prefill")
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 4)
        self.assertEqual(planner.publish_prefix("a", 8), 2)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 4)
        self.assertEqual(planner.publish_prefix("b", 8), 2)
        cache = planner.prefix_snapshot()
        self.assertEqual(cache["published_blocks"], 2)
        self.assertEqual(cache["deduplicated_blocks"], 2)
        self.assertEqual(cache["shared_blocks"], 2)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 2)
        # Publishing must not turn an earlier miss into a later lookup hit.
        self.assertEqual(planner.lookup_prefix("b"), 0)
        self.assertEqual(planner.prefix_snapshot()["lookup_requests"], 2)

    def test_lru_only_reclaims_unreferenced_entries(self):
        a, b = request("a", range(8)), request("b", range(5))
        c = request("c", range(20, 24))
        planner = MemoryPlanner(fixture([a, b, c], cache_blocks=2))
        populate(planner, a)
        self.assertEqual(planner.lookup_prefix("b"), 4)
        populate(planner, c, release=False)
        cache = planner.prefix_snapshot()
        self.assertEqual(cache["evicted_blocks"], 1)
        self.assertEqual(cache["cached_blocks"], 2)
        self.assertEqual(cache["pinned_blocks"], 2)
        self.assertEqual(sorted(entry["prefix_tokens"] for entry in cache["entries"]), [4, 4])
        self.assertEqual(planner.allocations["b"], 4)

    def test_full_pinned_cache_leaves_remaining_blocks_private(self):
        a, b = request("a", range(8)), request("b", range(20, 24))
        planner = MemoryPlanner(fixture([a, b], cache_blocks=1))
        populate(planner, a, release=False)
        populate(planner, b, release=False)
        self.assertEqual(planner.prefix_snapshot()["cached_blocks"], 1)
        self.assertEqual(planner.prefix_snapshot()["private_blocks"], 2)
        self.assertEqual(planner.prefix_snapshot()["evicted_blocks"], 0)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 3)

    def test_checks_plan_eviction_without_mutating_and_commit_reclaims(self):
        a, b = request("a", range(8)), request("b", range(20, 28))
        planner = MemoryPlanner(fixture([a, b], blocks=2))
        populate(planner, a)
        before, cache = planner.snapshot(), planner.prefix_snapshot()
        check = planner.check({"b": 8}, [8], "prefill")
        self.assertTrue(check.ok)
        self.assertEqual(check.details["cache_eviction_blocks"], 2)
        self.assertEqual(planner.snapshot(), before)
        self.assertEqual(planner.prefix_snapshot(), cache)
        self.assertTrue(planner.reserve({"b": 8}, [8], "prefill").ok)
        self.assertEqual(planner.prefix_snapshot()["evicted_blocks"], 2)
        self.assertEqual(planner.prefix_snapshot()["cached_blocks"], 0)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 2)

    def test_pinned_cache_prevents_reservation_but_not_alone_feasibility(self):
        a = request("a", range(4))
        planner = MemoryPlanner(fixture([a], blocks=2))
        populate(planner, a, release=False)
        before, cache = planner.snapshot(), planner.prefix_snapshot()
        self.assertFalse(planner.reserve_kv({"b": 8}).ok)
        self.assertEqual(planner.snapshot(), before)
        self.assertEqual(planner.prefix_snapshot(), cache)
        self.assertTrue(planner.check_kv({"b": 8}, alone=True).ok)
        self.assertTrue(planner.check({"b": 8}, [8], "prefill", alone=True).ok)

    def test_workspace_failure_does_not_evict_cache_or_change_peaks(self):
        a, b = request("a", range(8)), request("b", range(20, 28))
        exp = fixture([a, b], blocks=2)
        fixed = MemoryPlanner(exp).snapshot()[0]["fixed_budget_bytes"]
        exp = replace(exp, infra=replace(exp.infra, devices=(replace(
            exp.infra.devices[0], memory_bytes=fixed + 128 + 1000),)))
        planner = MemoryPlanner(exp)
        populate(planner, a)
        planner.config = replace(planner.config, workspace_bytes_per_token=1000)
        before, cache = planner.snapshot(), planner.prefix_snapshot()
        decision = planner.reserve({"b": 8}, [8], "prefill")
        self.assertEqual(decision.code, "WORKSPACE_BUDGET_EXCEEDED")
        self.assertEqual(planner.snapshot(), before)
        self.assertEqual(planner.prefix_snapshot(), cache)

    def test_tp_pp_reservation_uses_smallest_stage_pool_atomically(self):
        a = request("a", range(4))
        planner = MemoryPlanner(fixture([a], tp=2, pp=2, layers=(1, 3), blocks=6))
        populate(planner, a, release=False)
        before, cache = planner.snapshot(), planner.prefix_snapshot()
        self.assertFalse(planner.reserve_kv({"b": 8}).ok)
        self.assertEqual(planner.snapshot(), before)
        self.assertEqual(planner.prefix_snapshot(), cache)
        planner.release("a")
        self.assertTrue(planner.reserve_kv({"b": 8}).ok)
        self.assertEqual([rank["used_blocks"] for rank in planner.snapshot()], [2, 2, 2, 2])
        self.assertEqual(planner.prefix_snapshot()["cached_blocks"], 0)

    def test_swap_copies_full_request_and_detaches_only_at_completion(self):
        a = request("a", range(9))
        exp = fixture([a], blocks=3)
        planner = MemoryPlanner(exp)
        populate(planner, a, release=False)
        swap = HostSwapManager(exp, planner)
        transfer = swap.swap_out("a", 0)
        self.assertEqual(transfer.bytes, 3 * planner.snapshot()[0]["kv_block_bytes"])
        self.assertEqual(planner.prefix_snapshot()["pinned_blocks"], 2)
        swap.complete(transfer, transfer.end_ms)
        self.assertEqual(planner.prefix_snapshot()["pinned_blocks"], 0)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 2)
        restored = swap.swap_in("a", transfer.end_ms)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 3)
        self.assertEqual(planner.prefix_snapshot()["cached_blocks"], 0)
        swap.complete(restored, restored.end_ms)
        self.assertEqual(swap.host_reserved_bytes, 0)
        self.assertEqual(planner.allocations, {"a": 9})

    def test_profile_workspace_receives_nonzero_prefill_and_recompute_context(self):
        planner = MemoryPlanner(fixture())
        calls = []

        class Profile:
            def workspace_bytes(self, queries, phase, stage, *, context_lengths):
                calls.append((list(queries), phase, stage, context_lengths))
                return 123

        planner._profile_model = Profile()
        for phase in ("prefill", "recompute"):
            decision = planner.check({"a": 7}, [3], phase)
            self.assertTrue(decision.ok)
            self.assertEqual(decision.details["workspace_bytes"], 123)
        self.assertEqual(calls, [([3], "prefill", 0, [4]), ([3], "recompute", 0, [4])])


if __name__ == "__main__":
    unittest.main()
