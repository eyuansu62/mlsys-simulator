"""Keep captured protocol evidence distinct from unresolved course baselines."""

import hashlib
import json
from pathlib import Path
import unittest

from mlsys_sim.agent_contract import validate_agent_response


ROOT = Path(__file__).resolve().parents[1] / "examples" / "agent"


class AgentEvidenceTests(unittest.TestCase):
    def test_captured_http_bytes_bind_complete_protocol_responses(self):
        report = json.loads((ROOT / "evidence/http-verification.json").read_text())
        for case in report["cases"]:
            if case["kind"] != "explicit_different_catalog_protocol_conformance":
                continue
            with self.subTest(task=case["task"]):
                request = (ROOT / case["request_file"]).read_bytes()
                response = (ROOT / "evidence" / f"task-{case['task']}.protocol.response.json").read_bytes()
                self.assertEqual(hashlib.sha256(request).hexdigest(), case["request_sha256"])
                self.assertEqual(hashlib.sha256(response).hexdigest(), case["response_sha256"])
                validate_agent_response(request, json.loads(response))

    def test_original_handoff_files_remain_exact_and_unaccepted(self):
        source = json.loads((ROOT / "sources.json").read_text())
        evidence = json.loads((ROOT / "evidence/http-verification.json").read_text())
        rejected = [row for row in evidence["cases"] if row["kind"] == "original_unconfirmed_baseline"]
        self.assertEqual(len(rejected), 3)
        for record in source["records"]:
            data = (ROOT / f"task-{record['task']}.unconfirmed.request.json").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["original_request_sha256"])
            case = next(row for row in rejected if row["task"] == record["task"])
            self.assertEqual(case["http_status"], 422)
            self.assertFalse(case["original_baseline_accepted"])

    def test_protocol_variants_do_not_resample_or_rewrite_request_streams(self):
        mapping = {"llm-3b": "qwen3-1.7b", "llm-32b": "qwen3-4b", "llm-7b": "qwen3-0.6b"}
        for task in ("9.1", "9.2", "9.3"):
            original = json.loads((ROOT / f"task-{task}.unconfirmed.request.json").read_text())
            variant = json.loads((ROOT / f"task-{task}.protocol.request.json").read_text())
            expected = original["workload"]
            for row in expected["requests"]:
                if "target_model" in row:
                    row["target_model"] = mapping[row["target_model"]]
            self.assertEqual(expected, variant["workload"])
            self.assertEqual(original["evaluation"], variant["evaluation"])
            self.assertEqual(original["execution"], variant["execution"])
            self.assertIn("not-course-baseline", variant["candidate_id"])


if __name__ == "__main__":
    unittest.main()
