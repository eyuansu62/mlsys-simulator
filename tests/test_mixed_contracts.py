"""Mixed policy validation, atomic Workspace and legacy Compare compatibility."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, MemoryConfig,
                       MemoryPlanner, RequestSpec, SimulationError, StrategySpec,
                       calibrate_dataset, compare_runs, comparison_contract,
                       experiment_from_dict, write_json_exclusive)
from mlsys_sim.batching import batch_members


class MixedContractTests(unittest.TestCase):
    def test_opt_in_policy_and_synthetic_duration_are_strict(self):
        exp = experiment_from_dict({"strategy": {"mixed_batching": True},
                                    "calibration": {"backend": "synthetic", "mixed_ms": 7}})
        self.assertTrue(exp.strategy.mixed_batching)
        self.assertFalse(exp.strategy.chunked_prefill)
        self.assertEqual(exp.calibration.mixed_ms, 7)
        for value in (0, 1, "true", None):
            with self.subTest(policy=value), self.assertRaises(SimulationError):
                experiment_from_dict({"strategy": {"mixed_batching": value}})
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(duration=value), self.assertRaises(SimulationError):
                experiment_from_dict({"calibration": {"mixed_ms": value}})

    def test_mixed_roles_are_explicit_and_not_inferred_from_query_length(self):
        self.assertEqual(batch_members("mixed", [1, 1], ["decode", "prefill"]),
                         ("decode", "prefill"))
        invalid = [None, [], "decode,prefill", ["decode"], ["prefill", "prefill"],
                   ["decode", "recompute"], ["decode", {}], ["decode", ["prefill"]]]
        for members in invalid:
            with self.subTest(members=members), self.assertRaises(SimulationError) as raised:
                batch_members("mixed", [1, 1], members)
            self.assertEqual(raised.exception.code, "INVALID_CONFIG")
        with self.assertRaises(SimulationError):
            batch_members("mixed", [2, 1], ["decode", "prefill"])
        with self.assertRaises(SimulationError):
            batch_members("prefill", [1], ["prefill"])

    def test_memory_rejects_invalid_roles_and_negative_prefix_without_mutation(self):
        planner = MemoryPlanner(Experiment())
        before = planner.snapshot()
        for queries, targets, roles in [([1, 2], {"d": 2, "p": 1}, ["decode", "prefill"]),
                                        ([2, 1], {"d": 2, "p": 1}, ["decode", "prefill"]),
                                        ([1, 1], {"d": 2, "p": 1}, None)]:
            with self.subTest(queries=queries, targets=targets, roles=roles):
                with self.assertRaises(SimulationError):
                    planner.reserve(targets, queries, "mixed", request_phases=roles)
                self.assertEqual(planner.snapshot(), before)
                self.assertEqual(planner.allocations, {})

    def profiled_planner(self, root, *, infeasible):
        workload = (RequestSpec("seed", 0, 8, 1, prompt_token_ids=tuple(range(8))),
                    RequestSpec("decode", 0, 4, 2), RequestSpec("prefill", 0, 4, 1))
        base = Experiment(workload=workload,
            infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(4))),
            strategy=StrategySpec(prefix_cache=True, mixed_batching=True, tensor_parallel_size=2,
                pipeline_parallel_size=2, max_batch_tokens=8,
                memory=MemoryConfig(kv_block_tokens=4, kv_pool_bytes=1536)))
        records = []
        for stage in range(2):
            for query in (4, 8):
                records.append({"record_id": f"s{stage}-q{query}", "group_id": f"s{stage}-q{query}",
                    "phase": "prefill", "stage_index": stage, "query_tokens": [query],
                    "context_lengths": [0], "duration_ms": 1, "workspace_peak_bytes": 1})
            records.append({"record_id": f"s{stage}-mixed", "group_id": f"s{stage}-mixed",
                "phase": "mixed", "stage_index": stage, "query_tokens": [1, 4],
                "context_lengths": [4, 0], "request_phases": ["decode", "prefill"],
                "duration_ms": 2,
                "workspace_peak_bytes": (100 * 1024 * 1024 if stage == 1 and infeasible
                                          else 10 + stage * 20)})
        from mlsys_sim import profile_identity
        data = {"schema_version": 1, "dataset_id": "mixed-atomic-fixture",
                "identity": profile_identity(base), "source": {"kind": "test_fixture",
                "reference": "Atomic reservation test, not measured memory"}, "records": records}
        profile = calibrate_dataset(data, profile_version="mixed-atomic", allow_test_fixture=True)
        path = root / "profile.json"
        write_json_exclusive(path, profile)
        exp = replace(base, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                            profile_version="mixed-atomic", allow_test_fixture=True))
        planner = MemoryPlanner(exp)
        self.assertTrue(planner.reserve({"seed": 8}, [8], "prefill").ok)
        self.assertEqual(planner.publish_prefix("seed", 8), 2)
        planner.release("seed")
        self.assertTrue(planner.reserve({"decode": 4}, [4], "prefill").ok)
        return planner

    def test_joint_profile_workspace_failure_does_not_evict_or_partially_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            planner = self.profiled_planner(Path(temporary), infeasible=True)
            before = deepcopy((planner.allocations, planner.snapshot(), planner.prefix_snapshot()))
            result = planner.reserve({"decode": 5, "prefill": 4}, [1, 4], "mixed",
                                      request_phases=["decode", "prefill"])
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "WORKSPACE_BUDGET_EXCEEDED")
            self.assertEqual(result.details["cache_eviction_blocks"], 2)
            self.assertEqual((planner.allocations, planner.snapshot(), planner.prefix_snapshot()), before)

    def test_joint_reservation_evicts_idle_cache_only_after_all_ranks_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            planner = self.profiled_planner(Path(temporary), infeasible=False)
            before = deepcopy((planner.allocations, planner.snapshot(), planner.prefix_snapshot()))
            check = planner.check({"decode": 5, "prefill": 4}, [1, 4], "mixed",
                                  request_phases=["decode", "prefill"])
            self.assertTrue(check.ok)
            self.assertEqual((planner.allocations, planner.snapshot(), planner.prefix_snapshot()), before)
            self.assertTrue(planner.reserve({"decode": 5, "prefill": 4}, [1, 4], "mixed",
                                            request_phases=["decode", "prefill"]).ok)
            ranks = planner.snapshot()
            self.assertEqual([rank["workspace_peak_bytes"] for rank in ranks], [10, 10, 30, 30])
            self.assertTrue(all(rank["used_blocks"] == 3 for rank in ranks))
            self.assertEqual(planner.prefix_snapshot()["cached_blocks"], 0)
            self.assertEqual(planner.allocations, {"decode": 5, "prefill": 4})

    def test_pre_mixed_comparison_contract_gets_default_but_policy_changes_reject(self):
        current = comparison_contract(Experiment())
        legacy = deepcopy(current)
        legacy["strategy"].pop("mixed_batching")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predicted = root / "run"
            predicted.mkdir()
            (predicted / "manifest.json").write_text(json.dumps({"backend": "simulated",
                "comparison_contract": current}))
            (predicted / "result.json").write_text(json.dumps({"backend": "simulated",
                "status": "COMPLETED", "counts": {"total": 0}}))
            observation = root / "observation.json"
            observed = {"schema_version": 1, "kind": "measured_run", "comparison_contract": legacy,
                "source": {"kind": "test_fixture", "reference": "Compatibility test only"},
                "metrics": {"request_count": 0}}
            observation.write_text(json.dumps(observed))
            self.assertEqual(compare_runs(predicted, observation, allow_test_fixture=True)
                             ["metrics"]["request_count"]["absolute_error"], 0)
            observed["comparison_contract"]["strategy"]["mixed_batching"] = True
            observation.write_text(json.dumps(observed))
            with self.assertRaises(SimulationError) as raised:
                compare_runs(predicted, observation, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "COMPARISON_MISMATCH")


if __name__ == "__main__":
    unittest.main()
