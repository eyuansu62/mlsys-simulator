"""End-to-end CLI criteria, fixture permission, export and phase scope checks."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim.analytical_cost import analytical_cost_identity
from mlsys_sim.schema import AnalyticalCostSpec, CalibrationSpec, Experiment
from mlsys_sim.serialization import experiment_dict
from test_cost_calibration import fixture


ROOT = Path(__file__).resolve().parents[1]


class CostCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="mlsys-cost-cli-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def run_cli(self, *args):
        env = dict(os.environ)
        env.pop("MLSYS_VLLM_CAPTURE_CONFIG", None)
        return subprocess.run([sys.executable, "-m", "mlsys_sim.cost_cli", *map(str, args)],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)

    def inputs(self, data=None):
        data = fixture() if data is None else data
        dataset = self.write("dataset.json", data)
        membership = self.write("membership.json", {
            "training_record_ids": ["gpu1", "wall1", "gpu3", "wall3"],
            "holdout_record_ids": ["heldout2"], "repeat_record_ids": ["repeat1", "repeat3"]})
        return dataset, membership

    def freeze(self, dataset, membership):
        out = self.root / "freeze"
        process = self.run_cli("freeze", "--dataset", dataset, "--membership", membership,
                               "--max-relative-error", "0.25", "--out", out)
        self.assertEqual(process.returncode, 0, process.stderr)
        frozen = json.loads((out / "split.json").read_text())
        self.assertEqual(frozen["acceptance"]["max_relative_error"], 0.25)
        return out / "split.json"

    def fit(self, dataset, frozen):
        out = self.root / "fit"
        process = self.run_cli("fit", "--dataset", dataset, "--split", frozen,
                               "--calibration-id", "fixture-cost", "--allow-test-fixture", "--out", out)
        self.assertEqual(process.returncode, 0, process.stderr)
        return out / "fit.json"

    def test_full_cli_chain_requires_explicit_fixture_permission(self):
        dataset, membership = self.inputs()
        frozen = self.freeze(dataset, membership)
        denied = self.run_cli("fit", "--dataset", dataset, "--split", frozen,
                              "--calibration-id", "fixture-cost", "--out", self.root / "denied")
        self.assertEqual(denied.returncode, 2)
        self.assertEqual(json.loads(denied.stderr)["code"], "TEST_FIXTURE_NOT_ALLOWED")
        self.assertFalse((self.root / "denied").exists())
        fitted = self.fit(dataset, frozen)
        out = self.root / "validation"
        process = self.run_cli("validate", "--dataset", dataset, "--fit", fitted, "--out", out)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["max_relative_error"], 0.25)
        self.assertFalse(report["hardware_validated"])
        self.assertTrue((out / "manifest.json").is_file())

    def test_existing_output_is_never_overwritten(self):
        dataset, membership = self.inputs()
        frozen = self.freeze(dataset, membership)
        before = frozen.read_bytes()
        process = self.run_cli("freeze", "--dataset", dataset, "--membership", membership,
                               "--out", frozen.parent)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stderr)["code"], "COST_OUTPUT_EXISTS")
        self.assertEqual(frozen.read_bytes(), before)

    def test_negative_holdout_returns_three_and_preserves_failed_report(self):
        data = fixture()
        next(row for row in data["records"] if row["record_id"] == "heldout2")["observed_ms"] *= 20
        dataset, membership = self.inputs(data)
        fitted = self.fit(dataset, self.freeze(dataset, membership))
        out = self.root / "failed-validation"
        process = self.run_cli("validate", "--dataset", dataset, "--fit", fitted, "--out", out)
        self.assertEqual(process.returncode, 3, process.stderr)
        self.assertEqual(json.loads(process.stdout)["verdict"], "FAIL")
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["holdout"]["calibrated"]["records"][0]["record_id"], "heldout2")
        self.assertFalse(report["holdout"]["calibrated"]["records"][0]["passed"])
        self.assertTrue((out / "cells.csv").is_file())
        self.assertTrue((out / "holdout-errors.svg").is_file())

    def test_source_and_membership_tampering_return_two(self):
        dataset, membership = self.inputs()
        frozen = self.freeze(dataset, membership)
        altered = json.loads(dataset.read_text())
        altered["records"][0]["observed_attention_ms"] += 1
        changed = self.write("changed-dataset.json", altered)
        process = self.run_cli("fit", "--dataset", changed, "--split", frozen,
                               "--calibration-id", "changed", "--allow-test-fixture", "--out", self.root / "bad-source")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stderr)["code"], "INVALID_COST_CALIBRATION")
        changed_plan = json.loads(frozen.read_text())
        changed_plan["training_record_ids"].append("unused2")
        changed_split = self.write("changed-split.json", changed_plan)
        process = self.run_cli("fit", "--dataset", dataset, "--split", changed_split,
                               "--calibration-id", "changed", "--allow-test-fixture", "--out", self.root / "bad-members")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stderr)["code"], "INVALID_COST_CALIBRATION")

    def cost_experiment(self):
        calibration = CalibrationSpec(runtime_name="fixture-runtime", runtime_version="fixture-v1",
            runtime_config_sha256="a" * 64, analytical_cost_timing_boundary="fixture_step_wall_v1",
            prefill_execution_mode="eager", decode_execution_mode="eager",
            prefill_attention_backend="fixture-attention", decode_attention_backend="fixture-attention",
            allow_test_fixture=True)
        experiment = Experiment(calibration=calibration)
        cost = AnalyticalCostSpec(phase="prefill", execution_mode="eager", attention_backend="fixture-attention",
            identity_sha256=analytical_cost_identity(experiment), nonattention_scale=0, attention_scale=0,
            attention_fixed_us_per_layer=250, runtime_fixed_us=4000,
            min_batch_size=1, max_batch_size=1, min_query_tokens=1, max_query_tokens=3,
            min_context_tokens=0, max_context_tokens=0,
            calibration_id="phase-fixture", training_dataset_sha256="b" * 64,
            training_record_ids=("gpu1", "wall1"), source_kind="test_fixture")
        return replace(experiment, calibration=replace(calibration, analytical_costs=(cost,)))

    def test_phase_predict_uses_actual_scoped_cost_without_querying_other_phase(self):
        experiment = self.cost_experiment()
        source = self.write("experiment.json", experiment_dict(experiment))
        out = self.root / "predict"
        process = self.run_cli("predict", "--experiment", source, "--phase", "prefill",
                               "--query-tokens", "2", "--context-lengths", "0", "--out", out)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads((out / "prediction.json").read_text())
        self.assertAlmostEqual(result["estimate"]["duration_ms"], 5)
        self.assertEqual(result["estimate"]["source"], "analytical_cost_calibrated")
        self.assertIn("not_assessed", result["memory_feasibility"])
        self.assertFalse(result["hardware_validated"])

    def test_predict_rejects_mode_shape_and_fixture_scope_mismatch(self):
        configured = self.cost_experiment()
        for name, experiment, query in (
            ("mode", replace(configured, calibration=replace(configured.calibration,
                prefill_execution_mode="graph_full_and_piecewise_configured")), "2"),
            ("shape", configured, "4"),
            ("fixture", replace(configured, calibration=replace(configured.calibration, allow_test_fixture=False)), "2")):
            with self.subTest(name=name):
                source = self.write(name + ".json", experiment_dict(experiment))
                process = self.run_cli("predict", "--experiment", source, "--phase", "prefill",
                                       "--query-tokens", query, "--context-lengths", "0", "--out", self.root / name)
                self.assertEqual(process.returncode, 2)
                self.assertIn(json.loads(process.stderr)["code"], {"CALIBRATION_MISMATCH", "TEST_FIXTURE_NOT_ALLOWED"})

    def test_fit_exports_replayable_calibration_for_matching_experiment(self):
        experiment = self.cost_experiment()
        experiment = replace(experiment, strategy=replace(experiment.strategy,
            model=replace(experiment.strategy.model, num_layers=2)),
            calibration=replace(experiment.calibration, analytical_costs=(), allow_test_fixture=False))
        data = fixture()
        identity = analytical_cost_identity(experiment)
        for row in data["records"]:
            row["domain"]["identity_sha256"] = identity
        data["experiments"] = {"fixture-model": experiment_dict(experiment)}
        dataset, membership = self.inputs(data)
        fitted = self.fit(dataset, self.freeze(dataset, membership))
        index = json.loads((fitted.parent / "models.json").read_text())
        exported = json.loads((fitted.parent / index["fixture-model"]["experiment"]).read_text())
        calibration = exported["calibration"]
        self.assertTrue(calibration["allow_test_fixture"])
        self.assertEqual(len(calibration["analytical_costs"]), 1)
        self.assertEqual(calibration["analytical_costs"][0]["identity_sha256"], identity)
        self.assertEqual(calibration["analytical_costs"][0]["source_kind"], "test_fixture")
        self.assertEqual(calibration["profile_version"], "fixture-cost")
        source = self.write("export-base.json", experiment_dict(experiment))
        prediction = self.root / "export-predict"
        process = self.run_cli("predict", "--experiment", source,
            "--calibration", fitted.parent / index["fixture-model"]["calibration"],
            "--phase", "prefill", "--query-tokens", "2", "--context-lengths", "0", "--out", prediction)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads((prediction / "prediction.json").read_text())["estimate"]["source"],
                         "analytical_cost_calibrated")


if __name__ == "__main__":
    unittest.main()
