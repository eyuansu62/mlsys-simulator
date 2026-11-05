"""Installed-style Workspace repeat workflow using explicit CPU fixtures."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim.cli import main
from test_workspace import experiment, workspace_record
from test_workspace_repeats import observations


class WorkspaceRepeatCLITests(unittest.TestCase):
    def test_calibrate_snapshot_then_validate_keeps_rank_errors_and_provenance(self):
        exp = experiment(tp=2, pp=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for session, peaks in (("train", (4, 8)), ("repeat", (0, 9))):
                rows = [workspace_record(exp, stage=stage, phase="decode", queries=(1,),
                        contexts=(16,), peaks=peaks, identifier=f"s{stage}") for stage in range(2)]
                (root / f"{session}.json").write_text(json.dumps(observations(session, exp, rows)))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["calibrate", "--workspace", "--dataset", str(root / "train.json"),
                    "--profile-version", "cpu-fixture-v1", "--allow-test-fixture", "--out", str(root / "fit")]), 0)
                profile_before = (root / "fit/profile.json").read_bytes()
                self.assertEqual(main(["validate", "--workspace-repeats", "--profile", str(root / "fit/profile.json"),
                    "--training-dataset", str(root / "fit/dataset.json"), "--dataset", str(root / "repeat.json"),
                    "--allow-test-fixture", "--out", str(root / "validation")]), 0)
            report = json.loads((root / "validation/result.json").read_text())
            self.assertEqual(report["summary"]["coverage"], 1)
            self.assertEqual(report["summary"]["rank_coverage"], 1)
            self.assertEqual(report["summary"]["underprediction_count"], 2)
            self.assertFalse(report["hardware_validated"])
            self.assertEqual((root / "fit/profile.json").read_bytes(), profile_before)
            self.assertTrue(all((root / "validation" / name).is_file()
                for name in ("training.json", "repeat.json", "profile.json", "manifest.json")))
            manifest = json.loads((root / "validation/manifest.json").read_text())
            self.assertEqual(manifest["profile_kind"], "workspace_profile")

    def test_missing_training_and_extrapolation_are_explicit_argument_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, extra in enumerate(([], ["--training-dataset", "missing.json", "--allow-extrapolation"])):
                out = root / str(index)
                with redirect_stdout(io.StringIO()):
                    result = main(["validate", "--workspace-repeats", "--profile", "missing-profile.json",
                        "--dataset", "missing-data.json", "--out", str(out), *extra])
                self.assertEqual(result, 2)
                self.assertEqual(json.loads((out / "result.json").read_text())["error_code"], "INVALID_ARGUMENT")


if __name__ == "__main__":
    unittest.main()
