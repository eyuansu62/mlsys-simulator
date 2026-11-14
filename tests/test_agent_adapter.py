"""Protocol execution checks; conformance workloads are not course baselines."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from mlsys_sim.agent_adapter import simulate_agent_request
from mlsys_sim.agent_catalog import CATALOG_VERSION
from mlsys_sim.agent_contract import validate_agent_response
from mlsys_sim.schema import SimulationError


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "agent"


def fixture(task="9.2"):
    return json.loads((EXAMPLES / f"task-{task}.protocol.request.json").read_text())


def execute(value, *, indent=None):
    raw = json.dumps(value, ensure_ascii=False, indent=indent).encode("utf-8")
    result = simulate_agent_request(raw)
    validate_agent_response(raw, result)
    return raw, result


class AgentAdapterTests(unittest.TestCase):
    def test_92_exact_mixed_trace_and_raw_body_lineage(self):
        request = fixture()
        original = deepcopy(request)
        raw, response = execute(request, indent=3)
        self.assertEqual(request, original)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["input_sha256"], "sha256:" + hashlib.sha256(raw).hexdigest())
        self.assertNotEqual(response["input_sha256"], request["workload"]["source_sha256"])
        self.assertEqual(len(response["requests"]), 12)
        for source, row in zip(request["workload"]["requests"], response["requests"]):
            self.assertEqual(row["request_id"], source["request_id"])
            self.assertEqual(row["ready_ms"], source["arrival_ms"])
            self.assertEqual(row["output_tokens"], source["output_tokens_target"])
            self.assertEqual(row["device_ids"], ["gpu0"])
        self.assertEqual(response["calibration"]["kind"], "uncalibrated")
        self.assertIsNone(response["calibration"]["in_domain"])
        self.assertEqual(response["evidence_level"], "simulated")
        self.assertIsNone(response["requests"][0]["service_ms"])
        self.assertIn("requests.service_ms", response["unsupported_fields"])

    def test_repeat_and_duration_metadata_do_not_regenerate_workload(self):
        request = fixture()
        _, first = execute(request)
        _, repeat = execute(request)
        self.assertEqual(first, repeat)
        request["execution"].update(repeat=7, split="holdout", holdout="short")
        request["evaluation"]["duration_s"] = 1234
        _, changed = execute(request)
        self.assertEqual(first["requests"], changed["requests"])
        self.assertEqual(first["per_device"], changed["per_device"])
        self.assertNotEqual(first["input_sha256"], changed["input_sha256"])

    def test_91_fixed_sequential_lengths_follow_terminal_feedback(self):
        request = fixture("9.1")
        _, response = execute(request)
        rows = response["requests"]
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["ready_ms"], 0)
        for previous, following in zip(rows, rows[1:]):
            self.assertEqual(following["ready_ms"], previous["finished_ms"])
        self.assertEqual([r["output_tokens_target"] for r in rows],
                         [r["output_tokens_target"] for r in request["workload"]["requests"]])

    def test_93_relative_gaps_and_observed_replica_placement(self):
        request = fixture("9.3")
        _, response = execute(request)
        rows = {row["request_id"]: row for row in response["requests"]}
        self.assertEqual(len(rows), 10)
        self.assertEqual(response["status"], "succeeded")
        for source in request["workload"]["requests"]:
            if source.get("depends_on"):
                ready = max(rows[parent]["finished_ms"] for parent in source["depends_on"])
                self.assertAlmostEqual(rows[source["request_id"]]["ready_ms"], ready + source.get("gap_after_deps_ms", 0))
        self.assertEqual(len(response["per_device"]), 4)
        replicas = {row["replica_index"] for row in rows.values() if row["deployment_model"] == "qwen3-0.6b"}
        self.assertGreater(len(replicas), 1)
        for row in response["per_device"]:
            self.assertEqual(row["capacity_bytes"], 16 * 1024**3)
            self.assertLessEqual(row["kv_peak_bytes"], row["kv_pool_reserved_bytes"])

    def test_unconfirmed_originals_are_rejected_without_alias_inference(self):
        for task in ("9.1", "9.2", "9.3"):
            request = json.loads((EXAMPLES / f"task-{task}.unconfirmed.request.json").read_text())
            with self.subTest(task=task), self.assertRaises(SimulationError) as caught:
                simulate_agent_request(json.dumps(request).encode())
            self.assertEqual(caught.exception.code, "CATALOG_VERSION_MISMATCH")
            request["catalog_version"] = CATALOG_VERSION
            with self.assertRaises(SimulationError) as caught:
                simulate_agent_request(json.dumps(request).encode())
            self.assertIn(caught.exception.code, {"UNKNOWN_MODEL_ID", "UNKNOWN_HARDWARE_ID"})

    def test_modeled_initial_oom_has_all_not_started_and_unknown_memory(self):
        request = fixture()
        request["strategy"]["serving"]["gpu_memory_utilization"] = 0.01
        _, response = execute(request)
        self.assertEqual(response["status"], "failed")
        self.assertEqual(response["failure_class"], "modeled_oom")
        self.assertEqual(len(response["requests"]), 12)
        self.assertTrue(all(row["status"] == "not_started" for row in response["requests"]))
        self.assertIsNone(response["per_device"][0]["physical_peak_bytes"])

    def test_request_level_sequential_failure_advances_next_request(self):
        request = fixture("9.1")
        request["workload"]["requests"] = [
            {"request_id": "large", "input_tokens": 32, "output_tokens_target": 2},
            {"request_id": "small", "input_tokens": 1, "output_tokens_target": 2}]
        request["strategy"]["serving"]["kv_budget_gib"] = 2 / 1024
        _, response = execute(request)
        failed, completed = response["requests"]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["ready_ms"], failed["finished_ms"])
        self.assertEqual(response["status"], "failed")

    def test_dag_failure_blocks_only_descendants(self):
        request = fixture("9.3")
        request["workload"]["requests"] = [
            {"request_id": "parent", "target_model": "qwen3-4b", "input_tokens": 16, "output_tokens_target": 2},
            {"request_id": "child", "target_model": "qwen3-0.6b", "input_tokens": 1,
             "output_tokens_target": 2, "depends_on": ["parent"], "gap_after_deps_ms": 100},
            {"request_id": "independent", "target_model": "qwen3-0.6b", "input_tokens": 1, "output_tokens_target": 2}]
        request["strategy"]["serving"]["kv_budget_gib"] = 2 / 1024
        _, response = execute(request)
        rows = {row["request_id"]: row for row in response["requests"]}
        self.assertEqual(rows["child"]["status"], "blocked")
        self.assertEqual(rows["child"]["blocked_by"], ["parent"])
        self.assertIsNone(rows["child"]["ready_ms"])
        self.assertEqual(rows["independent"]["status"], "completed")

    def test_deadline_preserves_partial_tokens_and_future_offered_ids(self):
        request = fixture()
        request["execution"]["max_simulated_ms"] = 50
        _, response = execute(request)
        self.assertEqual(response["status"], "timed_out")
        first = response["requests"][0]
        self.assertGreater(first["output_tokens"], 0)
        self.assertLess(first["output_tokens"], first["output_tokens_target"])
        self.assertEqual(first["finished_ms"], 50)
        self.assertEqual(response["requests"][-1]["status"], "not_started")
        self.assertEqual(len(response["requests"]), 12)

    def test_no_chunking_is_silently_enabled_for_long_prompts(self):
        request = fixture("9.3")
        _, response = execute(request)
        self.assertIsNone(response["effective_config"]["strategy"]["serving"]["chunked_prefill_tokens"])
        self.assertEqual(response["requests"][-1]["status"], "completed")
        self.assertIn("max_batch_tokens=65536", response["logs"][0])

    def test_unsupported_policies_and_topology_are_not_substituted(self):
        for section, key, value in (("cache", "prefix_cache", True), ("scheduling", "priority", "short_first"),
                                    ("scheduling", "preemption", "restart")):
            request = fixture()
            request["strategy"].setdefault(section, {})[key] = value
            with self.subTest(key=key), self.assertRaises(SimulationError):
                simulate_agent_request(json.dumps(request).encode())
        request = fixture()
        request["infra"]["nodes"][0]["devices"].append({"id": "gpu1", "hardware": "rx9070xt"})
        request["strategy"]["deployments"][0]["devices"].append("gpu1")
        request["strategy"]["deployments"][0]["parallel"] = {"kind": "tp", "degree": 2}
        with self.assertRaises(SimulationError) as caught:
            simulate_agent_request(json.dumps(request).encode())
        self.assertEqual(caught.exception.code, "UNCONFIRMED_TOPOLOGY")

    def test_tp_degree_one_needs_no_unknown_inter_device_topology(self):
        request = fixture()
        request["strategy"]["deployments"][0]["parallel"] = {"kind": "tp", "degree": 1}
        _, response = execute(request)
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["effective_config"]["strategy"], request["strategy"])

    def test_optional_defaults_and_null_diagnostics_remain_valid(self):
        request = fixture("9.1")
        request["strategy"]["deployments"][0]["parallel"] = {}
        request["strategy"]["deployments"][0].pop("replicas")
        request["strategy"].pop("cache")
        request["strategy"].pop("scheduling")
        _, response = execute(request)
        self.assertEqual(response["effective_config"]["strategy"], request["strategy"])
        self.assertEqual(response["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
