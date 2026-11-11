"""Real public CLI artifacts for the opt-in course workflows."""

import hashlib
import io
import json
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim.calibration import calibrate_dataset
import mlsys_sim.cli as cli
from mlsys_sim.cli import main
from mlsys_sim.dag import run_dag_candidates, simulate_dag
from mlsys_sim.load_sweep import run_load_sweep
from mlsys_sim.profiles import profile_identity
from mlsys_sim.schema import DeviceSpec, Experiment, InfraSpec


ROOT = Path(__file__).resolve().parents[1]


def traffic():
    return {"schema_version": 1, "seed": 42,
            "arrival": {"process": "periodic", "rate_rps": 100, "duration_ms": 30, "max_requests": 10},
            "tokens": {"input": 2, "output": 1}}


def dag_plan():
    return {"nodes": [
        {"node_id": "a", "model_id": "large", "input_tokens": 2, "output_tokens": 1},
        {"node_id": "b", "model_id": "small", "input_tokens": 2, "output_tokens": 1, "dependencies": ["a"]},
        {"node_id": "c", "model_id": "large", "input_tokens": 2, "output_tokens": 1, "dependencies": ["b"]}],
        "pools": [{"model_id": name, "experiment": {
            "infra": {"devices": [{"id": f"{name}-gpu"}]},
            "calibration": {"backend": "synthetic", "prefill_ms": duration}}}
                  for name, duration in (("large", 2), ("small", 5))], "candidates": []}


class CourseCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_cli(self, mode, plan, *, experiment=None, name="run", expected=0):
        path = self.root / f"{name}-plan.json"
        path.write_text(json.dumps(plan))
        command = [sys.executable, "-m", "mlsys_sim", mode, "--plan", str(path),
                   "--out", str(self.root / name)]
        if experiment is not None:
            exp_path = self.root / f"{name}-experiment.json"
            exp_path.write_text(json.dumps(experiment))
            command += ["--experiment", str(exp_path)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        output = self.root / name
        result = json.loads((output / "result.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertFalse(result["hardware_validated"])
        self.assertFalse(manifest["hardware_validated"])
        self.assertEqual(manifest["mode"], mode)
        recorded = next(row for row in manifest["input_files"] if row["path"] == str(path.resolve()))
        self.assertEqual(recorded["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        return output, result

    def test_chapter1_has_eighteen_points_and_embedded_same_source_figures(self):
        plan = json.loads((ROOT / "examples/course/chapter1.plan.json").read_text())
        out, report = self.run_cli("chapter1", plan)
        self.assertEqual(report["point_count"], 18)
        self.assertEqual(sum(len(row["components"]) for row in report["points"]), 54)
        child = out / "artifacts"
        self.assertEqual(json.loads((child / "report.json").read_text()), report)
        self.assertEqual(len(list(child.glob("*.svg"))), 4)
        self.assertIn("roofline-rx9070xt.svg", (child / "report.html").read_text())
        self.assertIn("historical_prediction_before_measurement_established", report["provenance"])
        self.assertFalse(report["provenance"]["historical_prediction_before_measurement_established"])

    def test_generate_workload_retains_replayable_jsonl_and_seed(self):
        out, report = self.run_cli("generate-workload", traffic())
        rows = [json.loads(line) for line in (out / "workload.jsonl").read_text().splitlines()]
        self.assertEqual([row["arrival_ms"] for row in rows], [0, 10, 20])
        self.assertEqual(rows, report["requests"])
        self.assertEqual(report["provenance"]["seed"], 42)
        _, repeat = self.run_cli("generate-workload", traffic(), name="repeat")
        self.assertEqual(report, repeat)

    def test_load_sweep_reports_empirical_grid_without_changing_single_run_schema(self):
        plan = {"schema_version": 1, "workload": traffic(), "rate_grid_rps": [50, 100],
                "replications": 2, "drain_ms": 100, "min_completed_samples": 1,
                "slo": {"p99_e2e_ms": 20}}
        experiment = {"strategy": {"scheduler": "fifo"},
                      "calibration": {"backend": "synthetic", "prefill_ms": 2}}
        out, report = self.run_cli("sweep", plan, experiment=experiment)
        self.assertEqual(len(report["points"]), 2)
        self.assertEqual([len(row["runs"]) for row in report["points"]], [2, 2])
        self.assertEqual(report["boundaries"]["sla_max_observed"], 100)
        self.assertTrue((out / "artifacts/report.html").is_file())
        self.assertTrue((out / "artifacts/points.csv").is_file())

    def test_dag_children_arrive_on_actual_parent_completion(self):
        out, report = self.run_cli("dag", dag_plan())
        self.assertEqual(report["status"], "COMPLETED")
        self.assertEqual(report["makespan_ms"], 9)
        self.assertTrue(report["resources_drained"])
        self.assertEqual([row["ready_ms"] for row in report["nodes"]], [0, 2, 7])
        self.assertEqual([row["completion_ms"] for row in report["nodes"]], [2, 7, 9])
        self.assertTrue(list((out / "artifacts").glob("*.svg")))

    def test_invalid_dependency_is_structured_and_existing_output_is_immutable(self):
        plan = dag_plan()
        plan["nodes"][0]["dependencies"] = ["c"]
        _, report = self.run_cli("dag", plan, name="invalid", expected=2)
        self.assertEqual(report["error_code"], "INVALID_CONFIG")
        out, _ = self.run_cli("generate-workload", traffic())
        before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", "generate-workload", "--plan",
            str(self.root / "run-plan.json"), "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        self.assertEqual({str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}, before)

    def test_manifest_saves_consumed_profile_bytes_when_source_changes_between_cli_and_api(self):
        base = Experiment(infra=InfraSpec(devices=(DeviceSpec(id="course-gpu"),)))
        dataset = {"schema_version": 1, "dataset_id": "course-cli-fixture", "identity": profile_identity(base),
                   "source": {"kind": "test_fixture", "reference": "hand-derived two ms"},
                   "records": [{"record_id": "r", "group_id": "g", "phase": "prefill", "stage_index": 0,
                                "query_tokens": [2], "context_lengths": [0], "duration_ms": 2,
                                "workspace_peak_bytes": 100}]}
        original = json.dumps(calibrate_dataset(dataset, profile_version="cli-v1", allow_test_fixture=True)).encode()
        consumed = original + b"\n"
        profile = self.root / "mutable-profile.json"
        experiment = asdict(base)
        experiment["calibration"].update(backend="profile", profile_path=str(profile), allow_test_fixture=True)
        for name, api in (("sweep", run_load_sweep), ("dag-single", simulate_dag),
                          ("dag-candidates", run_dag_candidates)):
            with self.subTest(workflow=name):
                profile.write_bytes(original)
                output = self.root / name
                mode = "sweep" if name == "sweep" else "dag"
                plan_path = self.root / f"{name}.json"
                if mode == "sweep":
                    plan = {"schema_version": 1, "workload": traffic(), "rate_grid_rps": [50],
                            "replications": 1, "drain_ms": 100, "min_completed_samples": 1,
                            "slo": {"p99_e2e_ms": 20}}
                    exp_path = self.root / "profile-experiment.json"
                    exp_path.write_text(json.dumps(experiment))
                    extra = ["--experiment", str(exp_path)]
                    api_path = "mlsys_sim.load_sweep.run_load_sweep"
                else:
                    pools = [{"model_id": "large", "experiment": experiment}]
                    plan = {"nodes": [{"node_id": "a", "model_id": "large",
                                       "input_tokens": 2, "output_tokens": 1}]}
                    if name == "dag-single":
                        plan["pools"] = pools
                    else:
                        plan["candidates"] = [{"candidate_id": "candidate", "pools": pools}]
                    extra = []
                    api_path = "mlsys_sim.dag." + api.__name__
                plan_path.write_text(json.dumps(plan))

                def changed_during_capture(*args, **kwargs):
                    profile.write_bytes(consumed)
                    try:
                        return api(*args, **kwargs)
                    finally:
                        profile.unlink()

                with patch(api_path, side_effect=changed_during_capture), redirect_stdout(io.StringIO()):
                    code = main([mode, "--plan", str(plan_path), "--out", str(output), *extra])
                self.assertEqual(code, 0, (output / "result.json").read_text())
                manifest = json.loads((output / "manifest.json").read_text())
                selected = manifest["workflow_profiles"][0]
                self.assertEqual(selected["sha256"], hashlib.sha256(consumed).hexdigest())
                self.assertEqual((output / selected["snapshot_file"]).read_bytes(), consumed)
                recorded = next(item for item in manifest["input_files"] if item["path"] == str(profile.resolve()))
                self.assertEqual(recorded["sha256"], selected["sha256"])

    def test_dag_executes_captured_plan_when_live_file_is_replaced(self):
        plan = dag_plan()
        path = self.root / "captured-plan.json"
        path.write_text(json.dumps(plan))
        captured_bytes = path.read_bytes()
        replacement = dag_plan()
        replacement["nodes"][0]["node_id"] = "replacement"
        replacement["nodes"][1]["dependencies"] = ["replacement"]
        original_read = cli._read_json

        def replace_after_read(selected):
            value = original_read(selected)
            path.write_text(json.dumps(replacement))
            return value

        out = self.root / "captured-plan-run"
        with patch("mlsys_sim.cli._read_json", side_effect=replace_after_read), redirect_stdout(io.StringIO()):
            code = main(["dag", "--plan", str(path), "--out", str(out)])
        self.assertEqual(code, 0, (out / "result.json").read_text())
        result = json.loads((out / "result.json").read_text())
        self.assertEqual([row["node_id"] for row in result["nodes"]], ["a", "b", "c"])
        self.assertEqual(json.loads((out / "plan.json").read_text()), plan)
        manifest = json.loads((out / "manifest.json").read_text())
        recorded = next(item for item in manifest["input_files"] if item["path"] == str(path.resolve()))
        self.assertEqual(recorded["sha256"], hashlib.sha256(captured_bytes).hexdigest())


if __name__ == "__main__":
    unittest.main()
