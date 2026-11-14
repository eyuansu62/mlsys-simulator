"""A usable fraction constrains budgets without changing device allocations."""

from dataclasses import replace
import unittest

from mlsys_sim import DeviceSpec, Experiment, InfraSpec, MemoryConfig, SimulationError, StrategySpec
from mlsys_sim.config import experiment_from_dict
from mlsys_sim.memory import MemoryPlanner
from mlsys_sim.profiles import profile_identity


class MemoryUtilizationTests(unittest.TestCase):
    def configured(self, utilization, capacity=16 * 1024**2):
        exp = Experiment(infra=InfraSpec(devices=(DeviceSpec(memory_bytes=capacity),)))
        return replace(exp, strategy=replace(exp.strategy,
            memory=replace(exp.strategy.memory, gpu_memory_utilization=utilization)))

    def test_fraction_changes_budget_limit_not_physical_capacity_or_allocations(self):
        exp = self.configured(0.75)
        planner = MemoryPlanner(exp)
        baseline = MemoryPlanner(self.configured(None)).snapshot()[0]
        snapshot = planner.snapshot()[0]
        self.assertEqual(snapshot["device_memory_bytes"], 16 * 1024**2)
        self.assertEqual(snapshot["usable_budget_bytes"], 12 * 1024**2)
        self.assertEqual(snapshot["physical_reserved_bytes"], baseline["physical_reserved_bytes"])
        self.assertEqual(snapshot["fixed_budget_bytes"], baseline["fixed_budget_bytes"])
        self.assertEqual(profile_identity(exp), profile_identity(self.configured(None)))

    def test_static_and_dynamic_checks_use_the_fraction(self):
        with self.assertRaises(SimulationError) as caught:
            MemoryPlanner(self.configured(0.5))
        self.assertEqual(caught.exception.code, "STATIC_MEMORY_INFEASIBLE")
        planner = MemoryPlanner(self.configured(0.65))
        decision = planner.check({"r": 2048}, [2048], "prefill", alone=True)
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "WORKSPACE_BUDGET_EXCEEDED")
        self.assertTrue(MemoryPlanner(self.configured(None)).check(
            {"r": 2048}, [2048], "prefill", alone=True).ok)

    def test_rank_limits_follow_actual_distinct_capacities(self):
        exp = Experiment(infra=InfraSpec(devices=(DeviceSpec("a", memory_bytes=16 * 1024**2),
                                                  DeviceSpec("b", memory_bytes=32 * 1024**2))),
                         strategy=StrategySpec(tensor_parallel_size=2,
                             memory=MemoryConfig(gpu_memory_utilization=0.75)))
        rows = MemoryPlanner(exp).snapshot()
        self.assertEqual([row["device_memory_bytes"] for row in rows], [16 * 1024**2, 32 * 1024**2])
        self.assertEqual([row["usable_budget_bytes"] for row in rows], [12 * 1024**2, 24 * 1024**2])

    def test_large_byte_floor_does_not_round_through_binary_float(self):
        row = MemoryPlanner(self.configured(0.5, 2**53 + 1)).snapshot()[0]
        self.assertEqual(row["usable_budget_bytes"], 2**52)

    def test_default_snapshot_and_identity_remain_legacy_shaped(self):
        self.assertNotIn("gpu_memory_utilization", MemoryPlanner(Experiment()).snapshot()[0])
        self.assertNotIn("usable_budget_bytes", MemoryPlanner(Experiment()).snapshot()[0])
        self.assertIsNone(experiment_from_dict({}).strategy.memory.gpu_memory_utilization)
        for value in (0, -1, 1.01, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(SimulationError):
                experiment_from_dict({"strategy": {"memory": {"gpu_memory_utilization": value}}})


if __name__ == "__main__":
    unittest.main()
