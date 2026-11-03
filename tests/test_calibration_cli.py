"""Exercise the sourced dataset -> profile -> simulation -> comparison loop."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim.comparison import extract_metrics
from mlsys_sim.profiles import profile_identity
from mlsys_sim.schema import DeviceSpec, Experiment, InfraSpec, StrategySpec


ROOT = Path(__file__).resolve().parents[1]


def fixture_dataset() -> dict:
    records = []
    for name, phase, query, context, duration in [
        ("p16-a", "prefill", 16, 0, 9), ("p16-b", "prefill", 16, 0, 11),
        ("p24", "prefill", 24, 0, 15), ("p32", "prefill", 32, 0, 20),
        ("d16", "decode", 1, 16, 2), ("d17", "decode", 1, 17, 2),
    ]:
        records.append({"record_id": name, "group_id": f"{phase}-{query}-{context}",
                        "phase": phase, "stage_index": 0, "query_tokens": [query],
                        "context_lengths": [context], "duration_ms": duration,
                        "workspace_peak_bytes": 4096})
    return {"schema_version": 1, "dataset_id": "cli-fixture",
            "identity": profile_identity(Experiment()),
            "source": {"kind": "test_fixture", "reference": "Hand-defined CLI test data; not GPU measurements"},
            "records": records}


class CalibrationCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dataset = self.directory / "dataset.json"
        self.dataset.write_text(json.dumps(fixture_dataset()))

    def command(self, mode, *arguments, out="run", expected=0, wrapper=False):
        output = self.directory / out
        entry = [str(ROOT / f"{mode}.py")] if wrapper else ["-m", "mlsys_sim", mode]
        process = subprocess.run([sys.executable, *entry, *map(str, arguments), "--out", str(output)],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        summary = json.loads(process.stdout)
        self.assertTrue((output / "result.json").exists(), process.stdout + process.stderr)
        self.assertTrue((output / "logs.txt").exists())
        self.assertEqual(summary["result_path"], str((output / "result.json").resolve()))
        return output, json.loads((output / "result.json").read_text())

    def calibrated(self, name="fit", version="fixture-v1"):
        return self.command("calibrate", "--dataset", self.dataset, "--profile-version", version,
                            "--allow-test-fixture", out=name, wrapper=True)[0]

    def experiment(self, profile, name="experiment.json"):
        path = self.directory / name
        path.write_text(json.dumps({
            "workload": [{"request_id": "r", "arrival_ms": 0, "input_tokens": 16, "output_tokens": 3}],
            "calibration": {"backend": "profile", "profile_path": str(profile), "allow_test_fixture": True},
        }))
        return path

    def observed(self, run, name="observed.json"):
        manifest = json.loads((run / "manifest.json").read_text())
        result = json.loads((run / "result.json").read_text())
        rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
        data = {"schema_version": 1, "kind": "measured_run",
                "comparison_contract": manifest["comparison_contract"],
                "source": {"kind": "test_fixture", "reference": "Simulated values used only as explicit CLI fixtures"},
                "metrics": extract_metrics(result),
                "requests": [{key: row[key] for key in ("request_id", "ttft_ms", "tpot_ms", "end_to_end_ms")}
                             for row in rows]}
        path = self.directory / name
        path.write_text(json.dumps(data))
        return path

    def test_calibrate_to_profile_simulation_and_compare(self):
        fit = self.calibrated()
        profile = json.loads((fit / "profile.json").read_text())
        self.assertEqual(profile["source"]["kind"], "test_fixture")
        experiment = self.experiment(fit / "profile.json")
        run, result = self.command("simulate", "--experiment", experiment, out="simulation")
        self.assertEqual(result["counts"]["completed"], 1)
        rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["first_token_ms"], 10)
        self.assertEqual(rows[0]["completion_ms"], 14)
        self.assertFalse(result["hardware_validated"])
        self.assertEqual((run / "profile.json").read_bytes(), (fit / "profile.json").read_bytes())
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["profile_version"], "fixture-v1")
        self.assertTrue(manifest["profile_sha256"])
        self.assertIn("comparison_contract", manifest)
        observed = self.observed(run)
        _, compared = self.command("compare", "--predicted", run, "--observed", observed,
                                    "--allow-test-fixture", out="comparison", wrapper=True)
        self.assertEqual(compared["metrics"]["ttft_p50_ms"]["absolute_error"], 0)
        self.assertEqual(compared["request_comparison"]["paired_count"], 1)
        self.assertEqual(compared["source"]["kind"], "test_fixture")
        self.assertFalse(compared["hardware_validated"])

    def test_holdout_new_fit_and_existing_profile_validation(self):
        run, result = self.command("validate", "--dataset", self.dataset, "--profile-version", "held-out-v1",
                                    "--holdout-group", "prefill-24-0", "--allow-test-fixture",
                                    out="holdout", wrapper=True)
        self.assertEqual(result["summary"]["coverage"], 1)
        self.assertEqual(result["summary"]["mae_ms"], 0)
        self.assertTrue(result["test_fixture_only"])
        for name in ("train.json", "holdout.json", "profile.json", "manifest.json"):
            self.assertTrue((run / name).exists())
        _, repeated = self.command("validate", "--dataset", run / "holdout.json", "--profile", run / "profile.json",
                                   "--allow-test-fixture", out="existing-profile")
        self.assertEqual(repeated["summary"], result["summary"])

    def test_default_fixture_rejection_and_failed_outputs(self):
        run, result = self.command("calibrate", "--dataset", self.dataset, "--profile-version", "v1",
                                    out="no-fixture-permission", expected=2)
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(result["error_code"])
        self.assertTrue((run / "manifest.json").exists())
        self.assertFalse((run / "profile.json").exists())

    def test_validation_rejects_fitting_records_and_parent_version_reuse(self):
        fit = self.calibrated()
        previous = (fit / "profile.json").read_bytes()
        _, leaked = self.command("validate", "--dataset", self.dataset, "--profile", fit / "profile.json",
                                 "--allow-test-fixture", out="leaked", expected=2)
        self.assertEqual(leaked["error_code"], "VALIDATION_DATA_LEAKAGE")
        _, duplicate = self.command("calibrate", "--dataset", self.dataset, "--profile-version", "fixture-v1",
                                    "--parent-profile", fit / "profile.json", "--allow-test-fixture",
                                    out="duplicate-version", expected=2)
        self.assertEqual(duplicate["error_code"], "PROFILE_VERSION_CONFLICT")
        self.assertEqual((fit / "profile.json").read_bytes(), previous)

    def test_profile_snapshot_and_config_hash_track_content(self):
        fit = self.calibrated()
        shared = self.directory / "shared-profile.json"
        shared.write_bytes((fit / "profile.json").read_bytes())
        experiment = self.experiment(shared)
        first, _ = self.command("simulate", "--experiment", experiment, out="first")
        old_snapshot = (first / "profile.json").read_bytes()
        original = json.loads(shared.read_text())
        updated = deepcopy(original)
        updated["notes"]["annotation"] = "external edit"
        shared.write_text(json.dumps(updated))
        second, _ = self.command("simulate", "--experiment", experiment, out="second")
        first_manifest = json.loads((first / "manifest.json").read_text())
        second_manifest = json.loads((second / "manifest.json").read_text())
        self.assertNotEqual(first_manifest["config_sha256"], second_manifest["config_sha256"])
        self.assertEqual((first / "profile.json").read_bytes(), old_snapshot)
        self.assertNotEqual((second / "profile.json").read_bytes(), old_snapshot)

    def test_comparison_pairs_resolve_relative_to_pairs_file(self):
        fit = self.calibrated()
        experiment = self.experiment(fit / "profile.json")
        first, _ = self.command("simulate", "--experiment", experiment, out="a")
        second, _ = self.command("simulate", "--experiment", experiment, out="b")
        self.observed(first, "a-observed.json")
        self.observed(second, "b-observed.json")
        pairs = self.directory / "pairs.json"
        pairs.write_text(json.dumps([{"candidate_id": name, "predicted": name,
                                      "observed": f"{name}-observed.json"} for name in ("a", "b")]))
        _, result = self.command("compare", "--pairs", pairs, "--allow-test-fixture", out="pairs-comparison")
        self.assertEqual(result["ranked_candidates"], 2)
        self.assertEqual(result["predicted_ranking"], [["a", "b"]])
        self.assertEqual(result["pairwise_agreement"], 1)

    def test_conflicting_or_invalid_cli_arguments_preserve_diagnostics(self):
        fit = self.calibrated()
        for index, extra in enumerate([
            ["--profile", fit / "profile.json", "--profile-version", "new"],
            ["--profile", fit / "profile.json", "--holdout-group", "prefill-24-0"],
            ["--profile-version", "new", "--holdout-fraction", "nan"],
        ]):
            _, result = self.command("validate", "--dataset", self.dataset, *extra,
                                      "--allow-test-fixture", out=f"invalid-{index}", expected=2)
            self.assertEqual(result["status"], "FAILED")
        pairs = self.directory / "pairs.json"
        pairs.write_text("[]")
        _, result = self.command("compare", "--pairs", pairs, "--observed", "irrelevant.json",
                                  out="conflicting-compare", expected=2)
        self.assertEqual(result["error_code"], "INVALID_ARGUMENT")

    def test_every_workflow_refuses_existing_output_directory(self):
        fit = self.calibrated()
        previous = (fit / "result.json").read_bytes()
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", "calibrate", "--dataset",
                                  str(self.dataset), "--profile-version", "other", "--out", str(fit),
                                  "--allow-test-fixture"], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["error_code"], "OUTPUT_EXISTS")
        self.assertEqual((fit / "result.json").read_bytes(), previous)

    def test_communication_calibration_is_a_separate_explicit_workflow(self):
        experiment = Experiment(infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))),
                                strategy=StrategySpec(tensor_parallel_size=2))
        data = {"schema_version": 1, "kind": "communication_measurements",
                "dataset_id": "communication-fixture", "identity": profile_identity(experiment),
                "source": {"kind": "test_fixture", "reference": "Hand-defined communication fixture"},
                "records": [
                    {"record_id": "a", "group_id": "p2p", "primitive": "p2p", "ranks": 2,
                     "message_bytes": 0, "duration_ms": 0.005},
                    {"record_id": "b", "group_id": "p2p", "primitive": "p2p", "ranks": 2,
                     "message_bytes": 1000000, "duration_ms": 0.055},
                ]}
        path = self.directory / "communication.json"
        path.write_text(json.dumps(data))
        output, report = self.command("calibrate", "--communication", "--dataset", path,
                                      "--profile-version", "comm-v1", "--allow-test-fixture", out="comm")
        self.assertEqual(report["communication_group_count"], 1)
        profile = json.loads((output / "profile.json").read_text())
        self.assertEqual(profile["kind"], "communication_profile")
        self.assertIn("infra_patch", profile["groups"][0])
        self.assertEqual(report["source"]["kind"], "test_fixture")


if __name__ == "__main__":
    unittest.main()
