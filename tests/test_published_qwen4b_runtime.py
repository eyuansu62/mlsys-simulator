"""Replay the original published AMD cost artifact through the CPU public API.

These checks preserve frozen coefficients; they perform no calibration fit,
full online cohort, or hardware measurement.
"""

import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim.config import load_experiment
from mlsys_sim.runtime_cost import RuntimeCostModel


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/qwen4b_online_20261117"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def replay_entry(inventory, target):
    selected = [row for row in inventory["artifacts"] if row["replay_target"] == target]
    if len(selected) != 1:
        raise AssertionError(f"Expected exactly one original replay entry for {target}")
    return selected[0]


def payload(entry):
    relative = Path(entry["published_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise AssertionError("Published replay paths must stay portable inside the example")
    stored = (DATA / relative).read_bytes()
    if sha(stored) != entry["published_sha256"]:
        raise AssertionError("Published storage bytes differ from replay inventory")
    raw = gzip.decompress(stored) if entry["storage"] == "gzip" else stored
    if sha(raw) != entry["source_raw_sha256"]:
        raise AssertionError("Published storage changed original decoded bytes")
    return raw


class PublishedQwen4bRuntimeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory_path = DATA / "replay-inventory.json"
        cls.inventory = json.loads(cls.inventory_path.read_bytes())
        cls.cases_path = DATA / "cpu-regression-cases.json"
        cls.cases = json.loads(cls.cases_path.read_bytes())
        cls.model_entry = replay_entry(cls.inventory, "replay/runtime-cost-model.json")
        cls.experiment_entry = replay_entry(cls.inventory, "replay/experiment.json")
        cls.model_gzip_path = DATA / cls.model_entry["published_file"]
        cls.experiment_path = DATA / cls.experiment_entry["published_file"]
        cls.input_sha = {path: sha(path.read_bytes()) for path in
                         (cls.inventory_path, cls.cases_path, cls.model_gzip_path, cls.experiment_path)}
        cls.raw_model = payload(cls.model_entry)
        if sha(cls.raw_model) != cls.cases["runtime_model"]["raw_sha256"]:
            raise AssertionError("Published gzip did not preserve the accepted original model bytes")
        cls.artifact = json.loads(cls.raw_model)
        cls.temporary = tempfile.TemporaryDirectory(prefix="published-qwen4b-model-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.materialized = Path(cls.temporary.name) / "runtime-cost-model.json"
        cls.materialized.write_bytes(cls.raw_model)
        # Materialize the publisher's unmodified relative-path experiment next
        # to the unmodified model. No model or hardware values are reconstructed.
        cls.materialized_experiment = Path(cls.temporary.name) / "experiment.json"
        cls.raw_experiment = payload(cls.experiment_entry)
        cls.materialized_experiment.write_bytes(cls.raw_experiment)
        cls.experiment = load_experiment(cls.materialized_experiment)
        cls.fit_guard = patch("mlsys_sim.runtime_cost.fit_runtime_costs",
            side_effect=AssertionError("Published artifact regression must not fit new parameters"))
        cls.fit_mock = cls.fit_guard.start()
        cls.addClassCleanup(cls.fit_guard.stop)
        cls.model = RuntimeCostModel(cls.experiment)
        cls.snapshot = cls.model.profile

    def test_original_model_bytes_and_all_frozen_parameters_are_retained(self):
        expected = self.cases["runtime_model"]
        self.assertEqual(sha(self.raw_model), expected["raw_sha256"])
        self.assertEqual(self.artifact["model_sha256"], expected["model_sha256"])
        self.assertEqual(self.model.provenance["sha256"], expected["raw_sha256"])
        self.assertEqual(self.model.provenance["model_sha256"], expected["model_sha256"])
        self.assertEqual(len(self.artifact["raw_dataset"]["records"]), self.cases["training_record_count"])
        self.assertEqual(len(self.artifact["parameters"]), self.cases["parameter_count"])
        self.assertEqual(self.artifact["domain"]["timing_boundary"], self.cases["timing_boundary"])
        self.assertEqual(self.cases["source_commit"], "4169cbafc7b614bf5c3540647f81ab7ae45944db")
        self.assertFalse(self.cases["hardware_validated"])
        self.assertEqual(self.artifact["source"]["kind"], "measured")
        self.assertEqual(self.snapshot["parameters"], self.artifact["parameters"])
        self.assertEqual(sha(canonical(self.snapshot["parameters"])), expected["parameters_sha256"])
        self.assertEqual(self.snapshot["model_sha256"], expected["model_sha256"])
        self.fit_mock.assert_not_called()

    def test_real_prefill_decode_and_ragged_mixed_geometries_match_frozen_replay(self):
        cases = self.cases["queries"]
        self.assertEqual({row["phase"] for row in cases}, {"prefill", "decode", "mixed"})
        self.assertEqual({len(row["query_tokens"]) for row in cases}, {4, 5})
        self.assertEqual(len(cases), 6)
        parameters = {(row["phase"], row["batch_size"]): row for row in self.artifact["parameters"]}
        for row in cases:
            with self.subTest(phase=row["phase"], batch=len(row["query_tokens"])):
                estimate = self.model.estimate_step(row["phase"], row["query_tokens"], row["context_lengths"],
                    row["stage_index"], request_phases=row.get("request_phases"))
                self.assertTrue(math.isclose(estimate.duration_ms, row["duration_ms"],
                    rel_tol=1e-10, abs_tol=1e-12), (estimate.duration_ms, row["duration_ms"]))
                self.assertEqual(estimate.source, "runtime_cost_calibrated")
                info = estimate.breakdown["runtime_cost"]
                self.assertEqual(info["model_sha256"], self.cases["runtime_model"]["model_sha256"])
                self.assertTrue(info["shape_observed_in_training"])
                self.assertFalse(info["hardware_validated"])
                frozen = parameters[row["phase"], len(row["query_tokens"])]
                self.assertEqual(float(info["scale"]).hex(), float(frozen["scale"]).hex())
                self.assertEqual(float(info["intercept_ms"]).hex(), float(frozen["intercept_ms"]).hex())
                self.assertIsNone(row["workspace_peak_bytes"])
                self.assertIsNone(self.model.workspace_bytes(row["query_tokens"], row["phase"], row["stage_index"],
                    context_lengths=row["context_lengths"], request_phases=row.get("request_phases")))
        self.assertEqual(sha(self.materialized.read_bytes()), self.cases["runtime_model"]["raw_sha256"])
        self.assertEqual(self.materialized_experiment.read_bytes(), self.raw_experiment)
        for path, before in self.input_sha.items():
            self.assertEqual(sha(path.read_bytes()), before, str(path))
        self.assertEqual(sha(canonical(self.model.profile["parameters"])), self.cases["runtime_model"]["parameters_sha256"])
        self.fit_mock.assert_not_called()


class PublishedQwen4bForwardReportTests(unittest.TestCase):
    def test_original_failure_and_all_preregistered_cells_are_retained(self):
        doc = ROOT / "docs/qwen4b-online-20261117"
        report = json.loads(gzip.decompress((doc / "report.json.gz").read_bytes()))
        commitments = json.loads((doc / "source-byte-commitments.json").read_bytes())
        self.assertEqual(commitments["evaluation_report_raw_sha256"],
            "678e7acdefccd936e198a75074ec6cafa1e3272c93c76c5ad7553d8a3a5c86b0")
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["passing_cells"], 53)
        self.assertEqual(report["measured_requests"], 27648)
        self.assertEqual(report["measured_output_tokens"], 442368)
        self.assertTrue(report["prediction_reveal_verified"])
        self.assertEqual(report["unknown_cells"], 0)
        self.assertEqual(report["false_safe_cells"], 0)
        self.assertEqual(report["false_unsafe_cells"], 0)
        cells = report["cells"]
        self.assertEqual(len(cells), 54)
        self.assertEqual({(c["cohort"], c["concurrency"], c["replication"]) for c in cells},
            {(h, c, r) for h in ("baseline", "short", "long")
             for c in (1, 2, 4, 8, 16, 32) for r in range(3)})
        self.assertTrue(all(c["integrity_status"] == "PASS" for c in cells))
        failures = [c for c in cells if c["status"] != "PASS"]
        self.assertEqual([c["label"] for c in failures], ["41-short-c2-r2"])
        self.assertFalse(failures[0]["metric_pass"]["ttft_p99_ms"])
        self.assertTrue(math.isclose(failures[0]["relative_errors"]["ttft_p99_ms"],
            0.30588205882648445, rel_tol=1e-12))

    def test_public_checksums_cover_every_published_artifact(self):
        for directory in (DATA, ROOT / "docs/qwen4b-online-20261117"):
            with self.subTest(directory=directory.name):
                hashes = json.loads((directory / "sha256.json").read_bytes())
                actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                          if p.is_file() and p.name != "sha256.json"}
                self.assertEqual(set(hashes), actual)
                for name, expected in hashes.items():
                    self.assertEqual(sha((directory / name).read_bytes()), expected, name)


if __name__ == "__main__":
    unittest.main()
