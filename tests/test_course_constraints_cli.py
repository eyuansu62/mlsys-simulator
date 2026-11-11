"""Installed-style CLI checks for joint screening and imported course evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mlsys_sim.cli as cli
from mlsys_sim.calibration import calibrate_dataset
from mlsys_sim.profiles import profile_identity
from mlsys_sim.schema import Experiment


ROOT = Path(__file__).resolve().parents[1]


def screening_plan():
    experiment = {"workload": [{"request_id": "r", "arrival_ms": 0,
                                 "input_tokens": 2, "output_tokens": 1}],
                  "strategy": {"scheduler": "fifo"},
                  "calibration": {"backend": "synthetic", "prefill_ms": 2}}
    slow = deepcopy(experiment)
    slow["calibration"]["prefill_ms"] = 20
    return {"schema_version": 1, "kind": "simulation", "source": "synthetic_teaching_trace",
            "constraints": {"schema_version": 1, "min_completed_samples": 1,
                            "memory": {"max_budget_fraction": 1},
                            "latency": {"p99_e2e_ms": 5},
                            "throughput": {"basis": "elapsed", "min_requests_per_s": 100}},
            "objective": {"metric": "elapsed_ms", "direction": "min"},
            "candidates": [{"candidate_id": "fast", "experiment": experiment},
                           {"candidate_id": "slow", "experiment": slow}]}


class CourseConstraintsCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_cli(self, args, output, expected=0):
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", *args, "--out", str(output)],
                                 cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        result = json.loads((output / "result.json").read_text())
        self.assertFalse(result["hardware_validated"])
        return result

    def test_screen_ranks_only_jointly_passing_candidates(self):
        path = self.root / "screen.json"
        path.write_text(json.dumps(screening_plan()))
        out = self.root / "screen"
        result = self.run_cli(["screen", "--plan", str(path)], out)
        self.assertEqual([item["candidate_id"] for item in result["ranking"]], ["fast"])
        self.assertEqual(len(result["candidates"]), 2)
        self.assertTrue((out / "artifacts/report.html").is_file())
        self.assertTrue((out / "artifacts/replay.plan.json").is_file())
        before = {str(item.relative_to(out)): item.read_bytes() for item in out.rglob("*") if item.is_file()}
        self.run_cli(["screen", "--plan", str(path)], out, expected=2)
        self.assertEqual({str(item.relative_to(out)): item.read_bytes() for item in out.rglob("*")
                          if item.is_file()}, before)

    def test_screen_consumes_the_initial_captured_plan(self):
        path = self.root / "mutable-plan.json"
        plan = screening_plan()
        path.write_text(json.dumps(plan))
        raw = path.read_bytes()
        read = cli._read_json

        def replace_source(selected):
            value = read(selected)
            changed = screening_plan()
            changed["candidates"] = changed["candidates"][1:]
            path.write_text(json.dumps(changed))
            return value

        out = self.root / "captured-plan"
        with patch("mlsys_sim.cli._read_json", side_effect=replace_source), redirect_stdout(io.StringIO()):
            code = cli.main(["screen", "--plan", str(path), "--out", str(out)])
        self.assertEqual(code, 0)
        result = json.loads((out / "result.json").read_text())
        self.assertEqual([item["candidate_id"] for item in result["ranking"]], ["fast"])
        self.assertEqual(json.loads((out / "plan.json").read_text()), plan)
        manifest = json.loads((out / "manifest.json").read_text())
        item = next(item for item in manifest["input_files"] if item["path"] == str(path.resolve()))
        self.assertEqual(item["sha256"], hashlib.sha256(raw).hexdigest())

    def test_import_retains_raw_bytes_patch_and_calibrated_grid(self):
        raw = (ROOT / "examples/course/course-calibration.fixture.json").read_bytes()
        path = self.root / "course.json"
        path.write_bytes(raw)
        out = self.root / "import"
        result = self.run_cli(["import-course-calibration", "--input", str(path),
                               "--source-kind", "test_fixture", "--chapter1-plan",
                               str(ROOT / "examples/course/chapter1.plan.json"),
                               "--expected-sha256", hashlib.sha256(raw).hexdigest()], out)
        self.assertEqual((out / "inputs/course-calibration.json").read_bytes(), raw)
        self.assertEqual(result["calibrated_chapter1"]["point_count"], 18)
        self.assertFalse(result["calibration"]["hardware_validated"])
        self.assertTrue((out / "artifacts/calibration_patch.json").is_file())
        self.assertTrue((out / "artifacts/chapter1-calibrated/report.html").is_file())
        self.assertEqual(len(list((out / "artifacts/chapter1-calibrated").glob("*.svg"))), 4)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["course_calibration_sha256"], hashlib.sha256(raw).hexdigest())

    def test_screen_manifest_uses_consumed_profile_after_origin_removal(self):
        from mlsys_sim.screening import run_candidate_screening

        base = Experiment()
        dataset = {"schema_version": 1, "dataset_id": "screen-cli-fixture", "identity": profile_identity(base),
                   "source": {"kind": "test_fixture", "reference": "two ms serving step"},
                   "records": [{"record_id": "r", "group_id": "g", "phase": "prefill", "stage_index": 0,
                                "query_tokens": [2], "context_lengths": [0], "duration_ms": 2,
                                "workspace_peak_bytes": 100}]}
        raw = json.dumps(calibrate_dataset(dataset, profile_version="screen-cli-v1", allow_test_fixture=True)).encode()
        consumed = raw + b"\n"
        profile = self.root / "ephemeral-profile.json"
        profile.write_bytes(raw)
        exp = asdict(base)
        exp["workload"] = [{"request_id": "r", "arrival_ms": 0, "input_tokens": 2, "output_tokens": 1}]
        exp["calibration"].update(backend="profile", profile_path="ephemeral-profile.json",
                                  profile_version="screen-cli-v1", allow_test_fixture=True)
        plan = screening_plan()
        plan["candidates"] = [{"candidate_id": "profiled", "experiment": exp}]
        path = self.root / "profile-screen.json"
        path.write_text(json.dumps(plan))

        def consumed_new_source(*args, **kwargs):
            profile.write_bytes(consumed)
            try:
                return run_candidate_screening(*args, **kwargs)
            finally:
                profile.unlink()

        out = self.root / "profile-screen"
        with patch("mlsys_sim.screening.run_candidate_screening", side_effect=consumed_new_source), \
                redirect_stdout(io.StringIO()):
            code = cli.main(["screen", "--plan", str(path), "--out", str(out)])
        self.assertEqual(code, 0, (out / "result.json").read_text())
        manifest = json.loads((out / "manifest.json").read_text())
        snapshot = manifest["workflow_profiles"][0]
        self.assertEqual(snapshot["sha256"], hashlib.sha256(consumed).hexdigest())
        self.assertEqual((out / snapshot["snapshot_file"]).read_bytes(), consumed)

    def test_import_hash_mismatch_and_missing_fixture_declaration_are_structured(self):
        path = ROOT / "examples/course/course-calibration.fixture.json"
        result = self.run_cli(["import-course-calibration", "--input", str(path),
                               "--source-kind", "test_fixture", "--expected-sha256", "0" * 64],
                              self.root / "bad-sha", expected=2)
        self.assertEqual(result["status"], "FAILED")
        result = self.run_cli(["import-course-calibration", "--input", str(path)],
                              self.root / "undeclared-fixture", expected=2)
        self.assertEqual(result["status"], "FAILED")

    def test_import_uses_captured_bytes_after_source_is_deleted(self):
        from mlsys_sim.course_calibration import import_course_calibration

        path = self.root / "ephemeral-course.json"
        raw = (ROOT / "examples/course/course-calibration.fixture.json").read_bytes()
        path.write_bytes(raw)

        def delete_before_import(consumed, **kwargs):
            if path.exists():
                path.unlink()
            self.assertEqual(consumed, raw)
            return import_course_calibration(consumed, **kwargs)

        out = self.root / "captured-import"
        with patch("mlsys_sim.course_calibration.import_course_calibration", side_effect=delete_before_import), \
                redirect_stdout(io.StringIO()):
            code = cli.main(["import-course-calibration", "--input", str(path), "--source-kind", "test_fixture",
                             "--out", str(out)])
        self.assertEqual(code, 0)
        self.assertEqual((out / "inputs/course-calibration.json").read_bytes(), raw)
        self.assertEqual((out / "artifacts/source.json").read_bytes(), raw)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["course_calibration_sha256"], hashlib.sha256(raw).hexdigest())

    def test_import_bounds_source_before_hashing_or_parsing(self):
        path = self.root / "oversized.json"
        path.write_bytes(b" " * (1024 * 1024 + 1))
        out = self.root / "oversized"
        result = self.run_cli(["import-course-calibration", "--input", str(path)], out, expected=2)
        self.assertEqual(result["error_code"], "INVALID_COURSE_CALIBRATION")
        self.assertEqual(json.loads((out / "manifest.json").read_text())["input_files"], [])


if __name__ == "__main__":
    unittest.main()
