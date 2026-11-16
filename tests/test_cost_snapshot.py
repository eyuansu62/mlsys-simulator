"""New phase predictions bind the exact input bytes consumed, even on mutation."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import Experiment
from mlsys_sim.cost_cli import main
from mlsys_sim.performance import AnalyticalModel


class CostSnapshotTests(unittest.TestCase):
    def test_prediction_and_manifest_use_one_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / "experiment.json", root / "run"
            raw = json.dumps(asdict(Experiment())).encode()
            path.write_bytes(raw)
            original = Path.read_bytes
            reads = []

            def mutate_after_snapshot(candidate):
                value = original(candidate)
                if candidate == path:
                    reads.append(value)
                    path.write_bytes(b'{"changed":"after consumption"}')
                return value

            with patch.object(Path, "read_bytes", mutate_after_snapshot):
                status = main(["predict", "--experiment", str(path), "--phase", "prefill",
                               "--query-tokens", "16", "--context-lengths", "0", "--out", str(output)])
            self.assertEqual(status, 0)
            self.assertEqual(len(reads), 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["input_bytes_sha256"]["experiment"], hashlib.sha256(raw).hexdigest())
            result = json.loads((output / "prediction.json").read_text())
            expected = AnalyticalModel(Experiment()).estimate_step("prefill", [16], [0])
            self.assertEqual(result["estimate"]["duration_ms"], expected.duration_ms)


if __name__ == "__main__":
    unittest.main()
