"""Published source, frozen parameters and actual core predictions stay aligned."""

from dataclasses import replace
import gzip
import json
from pathlib import Path
import unittest

from mlsys_sim.analytical_cost import analytical_cost_identity
from mlsys_sim.config import load_experiment
from mlsys_sim.cost_calibration import validate_analytical_costs
from mlsys_sim.performance import AnalyticalModel
from mlsys_sim.schema import SimulationError


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/runtime_cost_20261116"


def compressed(path):
    return json.loads(gzip.decompress(path.read_bytes()))


class PublishedCostStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = compressed(EXAMPLES / "dataset.json.gz")
        cls.artifact = compressed(EXAMPLES / "fit.json.gz")
        cls.report = compressed(ROOT / "docs/runtime-cost-20261116/report.json.gz")

    def test_frozen_validation_reproduces_every_published_value(self):
        self.assertEqual(validate_analytical_costs(self.dataset, self.artifact), self.report)
        self.assertEqual(self.report["holdout"]["declared_record_count"], 684)
        self.assertEqual(self.report["holdout"]["acquisition_session_count"], 3)
        self.assertEqual(self.report["holdout"]["calibrated"]["cell_count"], 6)
        self.assertEqual(self.report["repeats"]["declared_record_count"], 1368)
        self.assertEqual(len(self.dataset["records"]), 6174)

    def test_actual_core_matches_all_heldout_and_repeat_predictions(self):
        index = json.loads((EXAMPLES / "models.json").read_text())
        models = {key: AnalyticalModel(load_experiment(EXAMPLES / paths["experiment"]))
                  for key, paths in index.items()}
        expected = {row["record_id"]: row["predicted_ms"] for split in ("holdout", "repeats")
                    for row in self.report[split]["calibrated"]["records"]}
        seen = set()
        for row in self.dataset["records"]:
            if row["record_id"] not in expected:
                continue
            prediction = models[row["model_id"]].estimate_step(row["phase"], row["query_tokens"], row["context_lengths"])
            self.assertAlmostEqual(prediction.duration_ms, expected[row["record_id"]], places=10)
            seen.add(row["record_id"])
        self.assertEqual(seen, set(expected))

    def test_eager_and_changed_backend_never_borrow_configured_graph_costs(self):
        index = json.loads((EXAMPLES / "models.json").read_text())
        for paths in index.values():
            exp = load_experiment(EXAMPLES / paths["experiment"])
            identity = analytical_cost_identity(exp)
            self.assertTrue(any(c.identity_sha256 == identity for c in exp.calibration.analytical_costs))
            for field, value in (("prefill_execution_mode", "eager"), ("prefill_attention_backend", "other-backend")):
                changed = replace(exp, calibration=replace(exp.calibration, **{field: value}))
                with self.assertRaises(SimulationError) as caught:
                    AnalyticalModel(changed).estimate_step("prefill", [512], [0])
                self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")
        self.assertEqual(self.dataset["source"]["original_archive_sha256"],
                         "32567daea53b3a0700d07be9a0f332577fddb2c87070f6515d43df15cb49bb85")
        self.assertEqual(self.dataset["source"]["archive_verification"], "verified_original_bytes_and_consumed_files")


if __name__ == "__main__":
    unittest.main()
