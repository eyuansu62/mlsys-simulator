"""Regression tests for physical budgets and logical KV block transactions."""

from dataclasses import replace
import unittest

from mlsys_sim.layout import rank_kv_bytes_per_token, rank_weight_bytes
from mlsys_sim.memory import MemoryPlanner
from mlsys_sim.schema import (
    DeviceSpec, Experiment, InfraSpec, MemoryConfig, ModelSpec, SimulationError,
    StrategySpec,
)


def fixture(*, tp=1, pp=1, layers=(), blocks=4, guard_blocks=0, workspace_per_token=0):
    model = ModelSpec()
    strategy = StrategySpec(
        model=model, tensor_parallel_size=tp, pipeline_parallel_size=pp,
        stage_layers=layers,
    )
    experiment = Experiment(
        strategy=strategy,
        infra=InfraSpec(devices=tuple(
            DeviceSpec(id=f"gpu{i}", memory_bytes=64 * 1024 * 1024)
            for i in range(tp * pp)
        )),
    )
    # Pool bytes are uniform per rank; PP stages may have different block sizes.
    block_bytes = rank_kv_bytes_per_token(experiment, 0) * 16
    memory = MemoryConfig(
        kv_block_tokens=16, kv_pool_bytes=block_bytes * blocks,
        workspace_base_bytes=128, workspace_bytes_per_token=workspace_per_token,
        runtime_bytes=64, safety_margin_bytes=64, guard_blocks=guard_blocks,
    )
    return replace(experiment, strategy=replace(strategy, memory=memory))


class MemoryPlannerTests(unittest.TestCase):
    def test_block_boundary_and_existing_allocations_are_retained(self):
        planner = MemoryPlanner(fixture())
        self.assertTrue(planner.reserve({"a": 16}, [16], "prefill").ok)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 1)
        self.assertTrue(planner.reserve({"a": 17}, [1], "decode").ok)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 2)
        self.assertTrue(planner.reserve({"b": 16}, [16], "prefill").ok)
        self.assertEqual(planner.snapshot()[0]["used_blocks"], 3)

    def test_release_keeps_fixed_physical_pool(self):
        planner = MemoryPlanner(fixture())
        planner.reserve({"a": 32}, [32], "prefill")
        occupied = planner.snapshot()[0]
        planner.release("a")
        planner.release("a")
        released = planner.snapshot()[0]
        self.assertEqual(released["used_blocks"], 0)
        self.assertEqual(released["free_blocks"], 4)
        self.assertEqual(occupied["physical_reserved_bytes"], released["physical_reserved_bytes"])
        self.assertEqual(occupied["kv_pool_reserved_bytes"], released["kv_pool_reserved_bytes"])
        self.assertEqual(released["peak_used_blocks"], 2)

    def test_check_and_failed_reserve_have_no_side_effects(self):
        planner = MemoryPlanner(fixture(blocks=2))
        planner.reserve({"a": 16}, [16], "prefill")
        before = planner.snapshot()
        result = planner.check({"b": 17}, [17], "prefill")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "KV_CAPACITY_EXCEEDED")
        self.assertEqual(planner.snapshot(), before)
        self.assertFalse(planner.reserve({"b": 17}, [17], "prefill").ok)
        self.assertEqual(planner.allocations, {"a": 16})
        self.assertEqual(planner.snapshot(), before)

    def test_alone_checks_infeasibility_without_releasing_other_requests(self):
        planner = MemoryPlanner(fixture(blocks=2))
        planner.reserve({"a": 16}, [16], "prefill")
        self.assertFalse(planner.check({"b": 32}, [32], "prefill").ok)
        self.assertTrue(planner.check({"b": 32}, [32], "prefill", alone=True).ok)
        self.assertFalse(planner.check({"b": 33}, [33], "prefill", alone=True).ok)
        self.assertEqual(planner.allocations, {"a": 16})

    def test_guard_blocks_are_not_allocatable(self):
        planner = MemoryPlanner(fixture(blocks=3, guard_blocks=1))
        self.assertTrue(planner.reserve({"a": 32}, [32], "prefill").ok)
        self.assertEqual(planner.snapshot()[0]["free_blocks"], 1)
        self.assertEqual(planner.snapshot()[0]["allocatable_free_blocks"], 0)
        self.assertFalse(planner.reserve({"a": 33}, [1], "decode").ok)

    def test_workspace_failure_is_distinct_from_kv_capacity(self):
        exp = fixture(blocks=4, workspace_per_token=1024)
        planner = MemoryPlanner(exp)
        fixed = planner.snapshot()[0]["fixed_budget_bytes"]
        exp = replace(exp, infra=replace(
            exp.infra, devices=(replace(exp.infra.devices[0], memory_bytes=fixed + 128 + 2048),),
        ))
        planner = MemoryPlanner(exp)
        self.assertTrue(planner.reserve({"a": 2}, [2], "prefill").ok)
        before = planner.snapshot()
        result = planner.reserve({"b": 3}, [3], "prefill")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "WORKSPACE_BUDGET_EXCEEDED")
        self.assertTrue(result.details["ranks"][0]["logical_ok"])
        self.assertEqual(planner.snapshot(), before)
        planner.release("a")
        self.assertFalse(planner.check({"b": 3}, [3], "prefill").ok)

    def test_tp_failure_is_atomic_across_ranks(self):
        exp = fixture(tp=2, workspace_per_token=1000)
        fixed = MemoryPlanner(exp).snapshot()[0]["fixed_budget_bytes"]
        devices = (exp.infra.devices[0], replace(
            exp.infra.devices[1], memory_bytes=fixed + 128 + 1000,
        ))
        planner = MemoryPlanner(replace(exp, infra=replace(exp.infra, devices=devices)))
        before = planner.snapshot()
        decision = planner.reserve({"a": 2}, [2], "prefill")
        self.assertEqual([rank["physical_ok"] for rank in decision.details["ranks"]], [True, False])
        self.assertEqual(planner.allocations, {})
        self.assertEqual(planner.snapshot(), before)

    def test_gqa_kv_heads_replicate_when_tp_exceeds_kv_heads(self):
        single = MemoryPlanner(fixture(tp=1)).snapshot()[0]
        ranks = MemoryPlanner(fixture(tp=8)).snapshot()
        self.assertEqual(len(ranks), 8)
        self.assertTrue(all(rank["kv_bytes_per_token"] == single["kv_bytes_per_token"] // 2
                            for rank in ranks))
        self.assertNotEqual(ranks[0]["kv_bytes_per_token"], single["kv_bytes_per_token"] // 8)

    def test_pp_has_stage_specific_block_sizes_and_weight_ownership(self):
        exp = fixture(tp=2, pp=2, layers=(1, 3), blocks=8)
        planner = MemoryPlanner(exp)
        ranks = planner.snapshot()
        self.assertEqual([rank["stage_index"] for rank in ranks], [0, 0, 1, 1])
        self.assertEqual(ranks[2]["kv_block_bytes"], 3 * ranks[0]["kv_block_bytes"])
        self.assertEqual(ranks[0]["weights_bytes"], rank_weight_bytes(exp, 0))
        self.assertEqual(ranks[2]["weights_bytes"], rank_weight_bytes(exp, 1))
        # Stage 1 can only hold floor(8/3)=2 blocks despite stage 0 having 8.
        result = planner.reserve({"a": 33}, [33], "prefill")
        self.assertEqual(result.code, "KV_CAPACITY_EXCEEDED")
        self.assertEqual(planner.allocations, {})

    def test_static_budget_failure(self):
        exp = fixture()
        fixed = MemoryPlanner(exp).snapshot()[0]["fixed_budget_bytes"]
        exp = replace(exp, infra=replace(
            exp.infra, devices=(replace(exp.infra.devices[0], memory_bytes=fixed + 127),),
        ))
        with self.assertRaises(SimulationError) as error:
            MemoryPlanner(exp)
        self.assertEqual(error.exception.code, "STATIC_MEMORY_INFEASIBLE")

    def test_invalid_candidates_cannot_shrink_or_use_fractional_tokens(self):
        planner = MemoryPlanner(fixture())
        planner.reserve({"a": 16}, [16], "prefill")
        for targets, queries in [({"a": 15}, [1]), ({"b": 1.5}, [1]), ({"b": 1}, [True])]:
            with self.assertRaises(SimulationError) as error:
                planner.check(targets, queries, "decode")
            self.assertEqual(error.exception.code, "INVALID_CONFIG")
        self.assertEqual(planner.allocations, {"a": 16})


if __name__ == "__main__":
    unittest.main()
