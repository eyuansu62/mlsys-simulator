"""Runtime snapshots retain the model reader's file limit and escaping budget."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import RequestSpec, RuntimeCostModel, SimulationError
from mlsys_sim.capacity import (_bind_inputs, _profile_inputs, _runtime_cost_budgets,
                                 _validate_inputs, canonical_json)
from mlsys_sim.load_validation import predict_closed_loop
from mlsys_sim.serialization import experiment_dict

try:
    from .test_runtime_cost_integration import fixture, study
except ImportError:
    from test_runtime_cost_integration import fixture, study


ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024


class RuntimeCostBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        exp, artifact = fixture()
        self.path = self.root / "runtime.json"
        self.small = json.dumps(artifact).encode()
        self.path.write_bytes(self.small)
        self.exp = replace(exp, calibration=replace(exp.calibration, profile_path=str(self.path)))

    def test_large_valid_model_through_prediction_cli_and_offline_rebind(self):
        expected = RuntimeCostModel(self.exp).estimate_step("prefill", [4], [0]).duration_ms
        # Legal JSON whitespace changes file bytes, not any fitted observation.
        # Tabs deliberately double in embedded raw_json to exercise both budgets.
        raw = b"\t" * (17 * MIB) + self.small
        self.path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        self.assertEqual(RuntimeCostModel(self.exp).estimate_step("prefill", [4], [0]).duration_ms, expected)
        inputs = _profile_inputs(self.exp)
        self.assertEqual(inputs[0]["sha256"], digest)
        self.assertEqual(inputs[0]["raw_json"].encode(), raw)
        self.assertGreater(len(canonical_json(inputs).encode()), 32 * MIB)
        self.assertEqual(_validate_inputs(self.exp, inputs), inputs)
        prediction = predict_closed_loop(self.exp, study())
        self.assertEqual(prediction["status"], "COMPLETED")
        self.assertEqual(prediction["profile_inputs"], inputs)

        configured = replace(self.exp, workload=(RequestSpec("r", 0, 4, 2),))
        path = self.root / "experiment.json"
        path.write_text(json.dumps(experiment_dict(configured)))
        commands = [(["-m", "mlsys_sim.cost_cli", "predict", "--phase", "prefill",
                      "--query-tokens", "4", "--context-lengths", "0"], "phase"),
                    (["-m", "mlsys_sim", "predict"], "shape"),
                    (["-m", "mlsys_sim", "simulate"], "simulation")]
        for command, name in commands:
            output = self.root / name
            process = subprocess.run([sys.executable, *command, "--experiment", str(path), "--out", str(output)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual((output / "runtime-cost.json").read_bytes(), raw)
        self.path.unlink()
        rebound = _bind_inputs(self.exp, inputs, self.root / "offline")
        self.assertEqual(Path(rebound.calibration.profile_path).read_bytes(), raw)
        self.assertEqual(RuntimeCostModel(rebound).estimate_step("prefill", [4], [0]).duration_ms, expected)

    def test_explicit_snapshot_limit_and_old_default_remain_bounded(self):
        raw = b"\t" * (17 * MIB) + self.small
        self.path.write_bytes(raw)
        with self.assertRaises(SimulationError) as caught:
            _profile_inputs(self.exp, max_bytes=len(raw) + 1024)
        self.assertEqual(caught.exception.code, "COLLECTION_OUTPUT_LIMIT")
        legacy = replace(self.exp, calibration=replace(self.exp.calibration, backend="profile",
            timing_boundary="step_includes_output_and_sampling_v1"))
        with self.assertRaises(SimulationError) as caught:
            _profile_inputs(legacy)
        self.assertEqual(caught.exception.code, "COLLECTION_OUTPUT_LIMIT")

    def test_runtime_read_and_offline_snapshot_use_same_reader_limit(self):
        file_limit, snapshot_limit = _runtime_cost_budgets()
        self.assertEqual(file_limit, 128 * MIB)
        self.assertGreater(snapshot_limit, 2 * file_limit)
        from mlsys_sim import runtime_cost
        with patch.object(runtime_cost, "_MAX_BYTES", len(self.small) - 1):
            with self.assertRaises(SimulationError) as caught:
                _profile_inputs(self.exp)
            self.assertEqual(caught.exception.code, "COLLECTION_OUTPUT_LIMIT")
        inputs = _profile_inputs(self.exp)
        # The offline path must enforce the same raw bound, never trust a SHA
        # as permission to bypass it or partially write a replay file.
        with patch.object(runtime_cost, "_MAX_BYTES", len(self.small) - 1):
            output = self.root / "rejected-offline"
            with self.assertRaises(SimulationError) as caught:
                _bind_inputs(self.exp, inputs, output)
            self.assertEqual(caught.exception.code, "COLLECTION_OUTPUT_LIMIT")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
