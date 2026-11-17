"""Replay the prepared Prefill study without GPUs, fitting or network access."""

from dataclasses import replace
from datetime import datetime
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from statistics import mean, median
import unittest

from mlsys_sim.config import load_experiment
from mlsys_sim.profiles import ProfileModel, validate_profile
from mlsys_sim.schema import SimulationError
from mlsys_sim.step_profile import check_step_stationarity


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/prefill_state_20261117"
REPORT = ROOT / "docs/prefill-state-20261117"


def raw(path):
    content = path.read_bytes()
    return gzip.decompress(content) if path.suffix == ".gz" else content


def read(path):
    return json.loads(raw(path))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class PublishedPrefillStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = read(DATA / "frozen/inputs.json.gz")
        cls.predictions = read(DATA / "frozen/predictions.json.gz")["predictions"]
        cls.report = read(DATA / "forward-report.json.gz")
        cls.experiment = load_experiment(DATA / "experiment.json")

    def test_released_bytes_match_manifests_and_frozen_commitment(self):
        for directory in (DATA, REPORT):
            manifest = read(directory / "sha256.json")
            files = {str(p.relative_to(directory)) for p in directory.rglob("*")
                     if p.is_file() and p.name != "sha256.json"}
            self.assertEqual(set(manifest), files)
            for name, expected in manifest.items():
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected, name)
        commitment = read(DATA / "frozen/commitment.json")
        for name, expected in commitment["files"].items():
            payload = raw(DATA / "frozen" / (name + ".gz"))
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected["raw_sha256"])
            self.assertEqual(hashlib.sha256(canonical(json.loads(payload))).hexdigest(),
                             expected["canonical_payload_sha256"])
        self.assertEqual(raw(DATA / "frozen/profile.json.gz"), (DATA / "profile.json").read_bytes())
        receipt = read(DATA / "receipts/freeze-receipt.json")
        self.assertEqual(receipt["prediction_commitment"], commitment)
        self.assertFalse(receipt["private_source_exported"])
        frozen_at = datetime.fromisoformat(receipt["committed_at_utc"])
        progress = read(DATA / "receipts/progress.json")
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(len(progress["attempts"]), 3)
        for job in progress["attempts"]:
            self.assertLess(frozen_at, datetime.fromisoformat(job["started_at_utc"]))
            self.assertLess(receipt["committed_monotonic_ns"], job["started_monotonic_ns"])
            self.assertEqual(job["returncode"], 0)
        # Historical source hashes identify the forecast commit; later compatible
        # source edits are allowed when the frozen predictions still replay.
        self.assertEqual(self.report["source_modules_sha256"], self.inputs["source_modules_sha256"])
        self.assertEqual(self.report["simulator_commit"], commitment["simulator_commit"])
        for name, digest in self.inputs["source_modules_sha256"].items():
            self.assertTrue(name.startswith("mlsys_sim/") and name.endswith(".py"))
            self.assertEqual(len(bytes.fromhex(digest)), 32)

    def test_training_proof_retains_all_observations_and_stable_sessions(self):
        dataset = read(DATA / "dataset.json")
        profile = validate_profile(read(DATA / "profile.json"))
        proof = profile["notes"]["reported_step_profile"]
        self.assertEqual(proof["raw_dataset"], dataset)
        self.assertEqual(len(dataset["records"]), 1968)
        self.assertEqual(len(profile["samples"]), 54)
        self.assertEqual(dataset["source"]["kind"], "measured")
        self.assertEqual(check_step_stationarity(dataset, read(DATA / "stationarity-policy.json")),
                         proof["stationarity"])
        self.assertEqual(proof["stationarity"]["verdict"], "PASS")
        self.assertEqual({r["session_id"] for r in dataset["notes"]["ordered_series"]},
                         {"train-a", "train-b"})
        self.assertTrue(all(r["workspace_peak_bytes"] is None for r in dataset["records"]))

    def test_all_71_predictions_replay_from_the_installed_public_profile(self):
        model = ProfileModel(self.experiment)
        self.assertEqual(len(self.predictions), 71)
        for row in self.predictions:
            prediction = model.estimate_step(row["phase"], row["query_tokens"],
                                             row["context_lengths_representative"])
            self.assertEqual(prediction.duration_ms, row["calibrated_ms"])
        for field, value in (("kv_pool_bytes", 4 * 1024**3), ("cpu_swap_bytes", 1024**3)):
            changed = replace(self.experiment, strategy=replace(self.experiment.strategy,
                memory=replace(self.experiment.strategy.memory, **{field: value})))
            with self.assertRaises(SimulationError):
                ProfileModel(changed).estimate_step("prefill", [768], [0])
        with self.assertRaises(SimulationError):
            model.estimate_step("prefill", [1600], [0])

    def test_all_33_cells_recompute_from_raw_steps_without_refitting(self):
        report = self.report
        predictions = {(r["phase"], r["shape"], r["context_sum"]): r for r in self.predictions}
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(len(report["cells"]), 33)
        self.assertEqual(report["covered_steps"], 3636)
        self.assertFalse(report["refitted"])
        self.assertFalse(report["hardware_validated"])
        self.assertTrue(report["scoped_hardware_measurement_performed"])
        errors, count = [], 0
        for cell in report["cells"]:
            steps = cell["steps"]
            self.assertEqual(len(steps), 100 if cell["phase"] == "prefill" else 128)
            count += len(steps)
            predicted = []
            for step in steps:
                row = predictions[cell["phase"], cell["shape"], step["context_sum"]]
                self.assertEqual(step["calibrated_ms"], row["calibrated_ms"])
                self.assertEqual(step["observed_ms"], step["raw_step"]["step_ms"])
                self.assertEqual(step["errors"], [])
                predicted.append(row["calibrated_ms"])
            observed = median(s["observed_ms"] for s in steps)
            error = abs(median(predicted) - observed) / observed
            self.assertEqual(observed, cell["observed_median_ms"])
            self.assertEqual(error, cell["calibrated_relative_error"])
            self.assertLessEqual(error, self.inputs["acceptance"]["max_relative_error"])
            errors.append(error)
        self.assertEqual(count, 3636)
        self.assertEqual(mean(errors), report["calibrated_cell_mape"])
        data = read(DATA / "forward-stationarity-dataset.json.gz")
        stable = check_step_stationarity(data, read(DATA / "stationarity-policy.json"))
        self.assertEqual(stable, read(DATA / "forward-stationarity.json"))
        self.assertEqual(stable["verdict"], "PASS")
        self.assertEqual(read(REPORT / "report.json.gz"), report)

    def test_warmups_are_retained_and_separate_from_accuracy_samples(self):
        warmups = list(csv.DictReader(io.StringIO(raw(REPORT / "warmups.csv.gz").decode())))
        steps = list(csv.DictReader(io.StringIO(raw(REPORT / "steps.csv.gz").decode())))
        self.assertEqual(len(warmups), 2688)
        self.assertEqual(len(steps), 3636)
        expected = {(f"forward-{session}", shape) for session in (1, 2, 3)
                    for shape in self.inputs["sessions"][0]["shapes"] if shape.startswith("S=")}
        self.assertEqual({(r["session"], r["shape"]) for r in warmups}, expected)
        for session, shape in expected:
            rows = [r for r in warmups if (r["session"], r["shape"]) == (session, shape)]
            self.assertEqual([int(r["index"]) for r in rows], list(range(128)))
        for row in warmups:
            self.assertEqual(row["role"], "warmup_not_accuracy_sample")
            self.assertEqual((row["cpu_before"], row["cpu_after"]), ("10", "10"))
            self.assertEqual(json.loads(row["errors"]), [])


if __name__ == "__main__":
    unittest.main()
