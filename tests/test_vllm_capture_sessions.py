"""Independent real-capture cohorts must not collide on local iteration IDs.

These source-shaped records are explicit CPU fixtures. Session identity does
not relax the existing record, group or shape holdout leakage checks.
"""

import json
import unittest

from mlsys_sim import SimulationError, calibrate_dataset, validate_holdout
from mlsys_sim.vllm_capture import normalize_vllm_capture
from test_vllm_capture import capture, experiment, record


def dataset(session_id, shapes):
    exp = experiment()
    records = [record(iteration=index, queries=[query], prompts=[query], duration=float(query))
               for index, query in enumerate(shapes)]
    channel = capture(exp, records=records)
    channel["session_id"] = session_id
    return normalize_vllm_capture(exp, [channel], allow_test_fixture=True)["dataset"]


def profile(data):
    return calibrate_dataset(data, profile_version="session-regression", allow_test_fixture=True)


class VllmCaptureSessionTests(unittest.TestCase):
    def test_independent_sessions_reusing_local_iterations_can_evaluate_disjoint_shapes(self):
        training = dataset("train-cohort", [4, 8])
        heldout = dataset("holdout-cohort", [6])
        train_ids = {row["record_id"] for row in training["records"]}
        self.assertTrue(train_ids.isdisjoint(row["record_id"] for row in heldout["records"]))
        report = validate_holdout(profile(training), heldout, allow_test_fixture=True)
        self.assertEqual(report["summary"]["coverage"], 1.0)
        self.assertEqual(report["summary"]["mape_percent"], 0.0)
        self.assertEqual(report["records"][0]["predicted_ms"], 6.0)
        identifier = heldout["records"][0]["record_id"]
        self.assertEqual(json.loads(identifier.removeprefix("vllm-record:")),
                         ["holdout-cohort", 0, 0, "iteration-0"])
        self.assertEqual(heldout["source"]["kind"], "test_fixture")

    def test_same_session_and_local_iteration_still_fail_record_leakage(self):
        training = dataset("same-cohort", [4, 8])
        heldout = dataset("same-cohort", [6])
        with self.assertRaises(SimulationError) as raised:
            validate_holdout(profile(training), heldout, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "VALIDATION_DATA_LEAKAGE")
        self.assertIn("record_ids", str(raised.exception))
        self.assertNotIn("shape_keys", str(raised.exception))

    def test_different_sessions_do_not_relax_shape_leakage(self):
        training = dataset("train-cohort", [4, 8])
        heldout = dataset("holdout-cohort", [4])
        with self.assertRaises(SimulationError) as raised:
            validate_holdout(profile(training), heldout, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "VALIDATION_DATA_LEAKAGE")
        self.assertIn("shape_keys", str(raised.exception))
        self.assertNotIn("record_ids", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
