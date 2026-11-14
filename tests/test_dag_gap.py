"""Relative dependency waits are causal and separate from absolute releases."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import unittest

from mlsys_sim import CalibrationSpec, Experiment, SimulationError
from mlsys_sim.dag import DAGNode, DAGPlan, ModelPool, dag_plan_dict, load_dag_plan, simulate_dag


def experiment(prefill_ms=1000):
    return Experiment(calibration=CalibrationSpec(backend="synthetic", prefill_ms=prefill_ms))


def plan(gap=100, prefill_ms=1000):
    return {
        "schema_version": 1,
        "trace_source": "explicit_relative_gap_test",
        "nodes": [
            {"node_id": "parent", "model_id": "toy", "input_tokens": 4, "output_tokens": 1},
            {"node_id": "child", "model_id": "toy", "input_tokens": 4, "output_tokens": 1,
             "dependencies": ["parent"], "gap_after_deps_ms": gap},
        ],
        "pools": [{"model_id": "toy", "experiment": asdict(experiment(prefill_ms))}],
    }


class RelativeGapTests(unittest.TestCase):
    def test_gap_follows_actual_parent_completion_and_changed_service(self):
        for service in (1000, 1400):
            with self.subTest(service=service):
                result = simulate_dag(plan(prefill_ms=service))
                parent, child = result["nodes"]
                self.assertEqual(parent["completion_ms"], service)
                self.assertEqual(child["ready_ms"], service + 100)
                self.assertEqual(child["first_compute_start_ms"], service + 100)
                self.assertEqual(child["completion_ms"], 2 * service + 100)
                self.assertEqual(child["gap_after_deps_ms"], 100)

    def test_absolute_lower_bound_and_multiple_parents(self):
        selected = plan()
        selected["nodes"].append({"node_id": "later", "model_id": "toy", "input_tokens": 4,
                                  "output_tokens": 1, "release_ms": 2500})
        selected["nodes"][1].update(dependencies=["parent", "later"], release_ms=4000)
        rows = {row["node_id"]: row for row in simulate_dag(selected)["nodes"]}
        self.assertEqual(rows["later"]["completion_ms"], 3500)
        self.assertEqual(rows["child"]["ready_ms"], 4000)
        selected["nodes"][1]["release_ms"] = 0
        rows = {row["node_id"]: row for row in simulate_dag(selected)["nodes"]}
        self.assertEqual(rows["child"]["ready_ms"], 3600)

    def test_failed_parent_blocks_child_without_false_ready(self):
        selected = plan()
        selected["nodes"][0]["timeout_ms"] = 1
        selected["nodes"].append({"node_id": "healthy", "model_id": "toy", "input_tokens": 4,
                                  "output_tokens": 1, "release_ms": 2000})
        rows = {row["node_id"]: row for row in simulate_dag(selected)["nodes"]}
        self.assertEqual(rows["child"]["status"], "BLOCKED_DEPENDENCY")
        self.assertIsNone(rows["child"]["ready_ms"])
        self.assertIsNone(rows["child"]["first_token_ms"] if "first_token_ms" in rows["child"] else None)
        self.assertEqual(rows["healthy"]["status"], "COMPLETED")

    def test_completion_exactly_at_deadline_counts_and_unstarted_child_stays_partial(self):
        selected = plan(gap=0)
        selected["execution"] = {"max_simulated_ms": 2000}
        result = simulate_dag(selected)
        self.assertTrue(result["nodes_completed"])
        self.assertEqual(result["nodes"][1]["completion_ms"], 2000)
        selected["nodes"][1]["gap_after_deps_ms"] = 1
        result = simulate_dag(selected)
        self.assertFalse(result["nodes_completed"])
        self.assertEqual(result["nodes"][1]["output_tokens_actual"], 0)

    def test_default_zero_preserves_legacy_trace_and_frozen_node_shape(self):
        selected = plan(gap=0)
        selected["nodes"][1].pop("gap_after_deps_ms")
        loaded = load_dag_plan(selected)
        legacy = [{key: value for key, value in asdict(node).items() if key != "gap_after_deps_ms"}
                  for node in loaded.nodes]
        digest = hashlib.sha256(json.dumps({"trace_source": loaded.trace_source, "nodes": legacy},
                                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        result = simulate_dag(loaded)
        self.assertEqual(result["trace_sha256"], digest)
        self.assertEqual(result["frozen_nodes"], legacy)
        self.assertTrue(all("gap_after_deps_ms" not in row for row in result["nodes"]))
        explicit_zero = deepcopy(selected)
        explicit_zero["nodes"][1]["gap_after_deps_ms"] = 0
        self.assertEqual(simulate_dag(explicit_zero), result)
        replay = dag_plan_dict(loaded)
        self.assertEqual(replay["nodes"], legacy)
        self.assertNotIn("max_active_requests", replay["pools"][0]["experiment"]["strategy"])
        self.assertNotIn("gpu_memory_utilization", replay["pools"][0]["experiment"]["strategy"]["memory"])
        self.assertEqual(simulate_dag(replay), result)

    def test_invalid_gap_roots_types_cycles_and_unknown_fields_fail(self):
        for value in (-1, float("nan"), float("inf"), True, "100"):
            with self.subTest(value=value):
                selected = plan(gap=value)
                with self.assertRaises(SimulationError):
                    load_dag_plan(selected)
        selected = plan()
        selected["nodes"][0]["gap_after_deps_ms"] = 1
        with self.assertRaises(SimulationError):
            load_dag_plan(selected)
        selected = plan()
        selected["nodes"][0]["dependencies"] = ["child"]
        with self.assertRaises(SimulationError):
            load_dag_plan(selected)
        selected = plan()
        selected["nodes"][1]["unrecognized_delay"] = 5
        with self.assertRaises(SimulationError):
            load_dag_plan(selected)

    def test_direct_dataclass_gap_is_validated(self):
        base = DAGPlan((DAGNode("p", "toy", 4, 1), DAGNode("c", "toy", 4, 1,
                        dependencies=("p",), gap_after_deps_ms=100)), (ModelPool("toy", experiment()),))
        self.assertEqual(simulate_dag(base)["nodes"][1]["ready_ms"], 1100)
        with self.assertRaises(SimulationError):
            simulate_dag(replace(base, nodes=(replace(base.nodes[0], gap_after_deps_ms=2), base.nodes[1])))

    def test_positive_gap_cannot_silently_round_to_no_wait(self):
        with self.assertRaisesRegex(SimulationError, "representable time"):
            simulate_dag(plan(gap=1e-300))


if __name__ == "__main__":
    unittest.main()
