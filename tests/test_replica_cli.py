"""Per-replica Profile snapshots and sourced multi-replica CLI artifacts."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, ModelSpec,
                       RequestSpec, StrategySpec, calibrate_dataset)
from mlsys_sim.comparison import extract_metrics
from mlsys_sim.layout import replica_experiment
from mlsys_sim.profiles import profile_identity


ROOT = Path(__file__).resolve().parents[1]


class ReplicaCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.sections = self.root / "sections"
        (self.sections / "profiles").mkdir(parents=True)
        self.base = Experiment(
            workload=(RequestSpec("a", 0, 4, 2), RequestSpec("b", 0, 4, 2)),
            infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}", name=f"fixture-device-{i}") for i in range(2))),
            strategy=StrategySpec(replicas=2, replica_routing="round_robin", scheduler="fifo",
                model=ModelSpec(name="fixture-dense", revision="fixture-revision",
                                weight_dtype="float16", kv_dtype="float16", activation_dtype="float16")),
            calibration=CalibrationSpec(runtime_name="fixture-runtime", runtime_version="fixture-v1",
                driver_version="not-applicable", rocm_version="not-applicable", rccl_version="not-applicable"))
        self.paths = []
        for replica in range(2):
            records = []
            for index, (phase, queries, contexts, duration) in enumerate([
                ("prefill", [4], [0], 2 + 3 * replica), ("decode", [1], [4], 1),
                ("prefill", [4, 4], [0, 0], 6 + replica), ("decode", [1, 1], [4, 4], 2),
            ]):
                records.append({"record_id": f"r{replica}-{index}", "group_id": f"r{replica}-{index}",
                    "phase": phase, "stage_index": 0, "query_tokens": queries, "context_lengths": contexts,
                    "duration_ms": duration, "workspace_peak_bytes": 4096})
            dataset = {"schema_version": 1, "dataset_id": f"replica-{replica}-fixture",
                "identity": profile_identity(replica_experiment(self.base, replica)),
                "source": {"kind": "test_fixture", "reference": "Hand-defined CLI values, not GPU evidence"},
                "records": records}
            profile = calibrate_dataset(dataset, profile_version=f"replica-v{replica}", allow_test_fixture=True)
            path = self.sections / "profiles" / f"{replica}.json"
            path.write_text(json.dumps(profile, indent=replica + 1) + "\n")
            self.paths.append(path)
        self.config = self.root / "experiment.json"
        self.calibration = self.sections / "calibration.json"
        calibration = asdict(self.base.calibration)
        calibration.update(backend="profile", profile_path=None, replica_profile_paths=["profiles/0.json", "profiles/1.json"],
                           allow_test_fixture=True)
        self.calibration.write_text(json.dumps(calibration))
        config = asdict(self.base)
        config["calibration"] = "sections/calibration.json"
        self.config.write_text(json.dumps(config))

    def run_cli(self, mode="simulate", name="run", expected=0, *extra):
        output = self.root / name
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", mode,
            "--experiment", str(self.config), "--out", str(output), *map(str, extra)], cwd=ROOT,
            capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        result = json.loads((output / "result.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertTrue((output / "logs.txt").exists())
        return output, result, manifest

    def test_profile_inputs_resolve_by_section_and_snapshot_exact_raw_bytes(self):
        output, result, manifest = self.run_cli()
        self.assertEqual(result["counts"]["completed"], 2)
        self.assertEqual(result["profile_version"], "per_replica")
        self.assertEqual(manifest["profile_version"], "per_replica")
        self.assertEqual(manifest["calibration_source"], "test_fixture_profile")
        self.assertTrue(manifest["comparison_contract"]["strategy"]["replicas"] == 2)
        for replica, origin in enumerate(self.paths):
            snapshot_file = f"profiles/replica-{replica}.json"
            self.assertEqual((output / snapshot_file).read_bytes(), origin.read_bytes())
            item = manifest["replica_profiles"][replica]
            self.assertEqual(item["snapshot_file"], snapshot_file)
            self.assertEqual(item["origin_profile_path"], str(origin))
            self.assertEqual(item["sha256"], hashlib.sha256(origin.read_bytes()).hexdigest())
            self.assertEqual(item["source_kind"], "test_fixture")
            provenance = result["profile_provenance"]["replicas"][replica]
            self.assertEqual(provenance["snapshot_file"], snapshot_file)
            self.assertEqual(provenance["origin_profile_path"], str(origin))
        inputs = {entry["path"]: entry["sha256"] for entry in manifest["input_files"]}
        self.assertIn(str(self.calibration), inputs)
        self.assertTrue(all(str(path) in inputs for path in self.paths))
        self.assertFalse(result["hardware_validated"])

    def test_prediction_uses_each_snapshot_and_retains_per_replica_provenance(self):
        output, result, manifest = self.run_cli("predict", "prediction")
        self.assertEqual(result["online_metrics_reason"], "SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING")
        self.assertEqual(len(result["replica_predictions"]), 2)
        self.assertEqual(len(result["profile_provenance"]["replicas"]), 2)
        self.assertTrue(all(value is None for value in result["online_metrics"].values()))
        for item in result["replica_predictions"]:
            replica = item["replica_index"]
            provenance = item["prediction"]["profile_provenance"]
            self.assertEqual(provenance["snapshot_file"], f"profiles/replica-{replica}.json")
            self.assertEqual(provenance["origin_profile_path"], str(self.paths[replica]))
        self.assertEqual(manifest["calibration_source"], "test_fixture_profile")
        self.assertTrue((output / "profiles/replica-1.json").exists())

    def test_source_edit_changes_config_hash_without_mutating_earlier_snapshot(self):
        first, _, first_manifest = self.run_cli(name="before")
        old_bytes = (first / "profiles/replica-1.json").read_bytes()
        profile = json.loads(self.paths[1].read_text())
        profile.setdefault("notes", {})["annotation"] = "external edit after first run"
        self.paths[1].write_text(json.dumps(profile))
        second, _, second_manifest = self.run_cli(name="after")
        self.assertNotEqual(first_manifest["config_sha256"], second_manifest["config_sha256"])
        self.assertEqual((first / "profiles/replica-1.json").read_bytes(), old_bytes)
        self.assertNotEqual((second / "profiles/replica-1.json").read_bytes(), old_bytes)

    def test_local_profiles_cannot_be_swapped_between_device_identities(self):
        calibration = json.loads(self.calibration.read_text())
        calibration["replica_profile_paths"].reverse()
        self.calibration.write_text(json.dumps(calibration))
        for mode in ("simulate", "predict"):
            _, result, manifest = self.run_cli(mode, f"wrong-{mode}", 2)
            self.assertEqual(result["error_code"], "CALIBRATION_MISMATCH")
            self.assertTrue(manifest["input_files"])

    def test_missing_path_and_fixture_permission_fail_with_diagnostics(self):
        calibration = json.loads(self.calibration.read_text())
        calibration["allow_test_fixture"] = False
        self.calibration.write_text(json.dumps(calibration))
        _, result, _ = self.run_cli(name="fixture-denied", expected=2)
        self.assertEqual(result["error_code"], "TEST_FIXTURE_NOT_ALLOWED")
        calibration["allow_test_fixture"] = True
        calibration["replica_profile_paths"][1] = "profiles/missing.json"
        self.calibration.write_text(json.dumps(calibration))
        _, result, _ = self.run_cli(name="missing", expected=2)
        self.assertEqual(result["error_code"], "IO_ERROR")

    def test_requested_version_must_match_every_replica(self):
        calibration = json.loads(self.calibration.read_text())
        calibration["profile_version"] = "replica-v0"
        self.calibration.write_text(json.dumps(calibration))
        output, result, manifest = self.run_cli(name="version-mismatch", expected=2)
        self.assertEqual(result["error_code"], "CALIBRATION_MISMATCH")
        self.assertEqual(len(manifest["replica_profiles"]), 1)
        self.assertEqual((output / "profiles/replica-0.json").read_bytes(), self.paths[0].read_bytes())

    def test_existing_output_cannot_be_overwritten(self):
        output, _, _ = self.run_cli()
        original = (output / "profiles/replica-0.json").read_bytes()
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", "simulate", "--experiment",
            str(self.config), "--out", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["error_code"], "OUTPUT_EXISTS")
        self.assertEqual((output / "profiles/replica-0.json").read_bytes(), original)

    def test_global_comparison_contract_roundtrips_through_cli(self):
        output, result, manifest = self.run_cli()
        observed = self.root / "observation.json"
        observed.write_text(json.dumps({"schema_version": 1, "kind": "measured_run",
            "comparison_contract": manifest["comparison_contract"],
            "source": {"kind": "test_fixture", "reference": "Simulation copied for CLI contract testing only"},
            "metrics": extract_metrics(result)}))
        compare = self.root / "comparison"
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", "compare", "--predicted", str(output),
            "--observed", str(observed), "--allow-test-fixture", "--out", str(compare)], cwd=ROOT,
            capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((compare / "result.json").read_text())
        self.assertEqual(report["metrics"]["request_count"]["absolute_error"], 0)
        self.assertFalse(report["hardware_validated"])

    def test_mixed_declared_sources_are_not_promoted_to_validated_hardware(self):
        # These are contract-test values with explicit provenance, not GPU data.
        profile = json.loads(self.paths[0].read_text())
        profile["source"] = {"kind": "measured", "reference": "Measured-source metadata contract fixture; no GPU evidence"}
        self.paths[0].write_text(json.dumps(profile))
        _, result, manifest = self.run_cli(name="mixed-sources")
        self.assertEqual(manifest["calibration_source"], "mixed_replica_profiles")
        self.assertEqual(result["calibration_source"], "mixed_replica_profiles")
        self.assertEqual([item["source_kind"] for item in manifest["replica_profiles"]], ["measured", "test_fixture"])
        self.assertFalse(result["hardware_validated"])


if __name__ == "__main__":
    unittest.main()
