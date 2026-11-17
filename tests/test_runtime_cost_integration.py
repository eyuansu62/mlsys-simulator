"""Runtime costs use their own reader through simulation, snapshots and CLIs."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import (RequestSpec, RuntimeCostModel, SimulationError,
    create_performance_model, fit_runtime_costs, simulate)
from mlsys_sim.capacity import _bind_inputs, _profile_inputs, _validate_inputs
from mlsys_sim.comparison import comparison_contract, compare_runs
from mlsys_sim.config import experiment_from_dict, load_experiment, validate_experiment
from mlsys_sim.load_validation import predict_closed_loop
from mlsys_sim.performance import AnalyticalModel
from mlsys_sim.profiles import load_profile, profile_identity, validate_identity
from mlsys_sim.serialization import experiment_dict
try:
    from .test_comparison import prediction as comparison_prediction
    from .test_runtime_cost import fixture as runtime_fixture
except ImportError:
    from test_comparison import prediction as comparison_prediction
    from test_runtime_cost import fixture as runtime_fixture


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = "engine_core_step_wall_v1"


def fixture():
    exp, data, policy = runtime_fixture()
    artifact = fit_runtime_costs(data, policy, calibration_id="runtime-integration-v1", allow_test_fixture=True)
    assert artifact["kind"] == "runtime_cost_model_v1"
    return exp, artifact


def study():
    return {"schema_version": 1, "study_id": "runtime-integration", "cohort": "baseline",
        "concurrency_grid": [1], "requests_per_run": 3, "replications": 1, "seed": 1,
        "think_ms": 0, "input_lengths": [4], "output_tokens": 2, "timeout_ms": 1000,
        "max_run_ms": 10000}


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        exp, self.artifact = fixture()
        self.path = self.root / "cost.json"
        self.raw = (json.dumps(self.artifact, indent=1) + "\n").encode()
        self.path.write_bytes(self.raw)
        self.exp = replace(exp, calibration=replace(exp.calibration, profile_path=str(self.path)))

    def run_cli(self, *args, cost=False):
        return subprocess.run([sys.executable, "-m", "mlsys_sim.cost_cli" if cost else "mlsys_sim",
                               *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=30)

    def test_config_requires_explicit_new_backend_and_boundary(self):
        validate_experiment(self.exp)
        self.assertEqual(experiment_from_dict(experiment_dict(self.exp)), self.exp)
        changes = [{"backend": "analytical"}, {"backend": "profile"},
                   {"timing_boundary": "step_includes_output_and_sampling_v1"},
                   {"profile_path": None}, {"runtime_config_sha256": ""}, {"allow_extrapolation": True}]
        for fields in changes:
            with self.subTest(fields=fields), self.assertRaises(SimulationError):
                validate_experiment(replace(self.exp, calibration=replace(self.exp.calibration, **fields)))
        with self.assertRaises(SimulationError):
            validate_experiment(replace(self.exp, strategy=replace(self.exp.strategy, replicas=2)))
        with self.assertRaises(SimulationError):
            AnalyticalModel(self.exp)
        with self.assertRaises(SimulationError):
            load_profile(self.path, allow_test_fixture=True)

    def test_factory_mixed_engine_and_provenance(self):
        model = create_performance_model(self.exp)
        self.assertIsInstance(model, RuntimeCostModel)
        prefill = model.estimate_step("prefill", [4], [0]).duration_ms
        decode = model.estimate_step("decode", [1], [4]).duration_ms
        exp = replace(self.exp, workload=(RequestSpec("a", 0, 4, 3),
            RequestSpec("b", prefill + decode / 2, 4, 3)))
        run = simulate(exp)
        self.assertEqual(run.result["counts"]["completed"], 2)
        self.assertGreater(run.result["batching"]["mixed_batches_admitted"], 0)
        self.assertEqual(run.result["calibration_source"], "test_fixture_runtime_cost")
        self.assertEqual(run.result["profile_version"], "runtime-integration-v1")
        self.assertEqual(run.result["profile_provenance"]["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertFalse(run.result["hardware_validated"])
        with self.assertRaises(SimulationError):
            create_performance_model(replace(self.exp, strategy=replace(self.exp.strategy, prefill_chunk_tokens=8)))

    def test_exact_raw_reader_snapshot_and_offline_rebind_after_origin_deleted(self):
        inputs = _profile_inputs(self.exp)
        self.assertEqual(inputs[0]["reader"], "runtime_cost_v1")
        self.assertEqual(inputs[0]["raw_json"].encode(), self.raw)
        self.assertEqual(_validate_inputs(self.exp, inputs), inputs)
        for reader in (None, "profile_v1"):
            corrupted = deepcopy(inputs)
            if reader is None:
                corrupted[0].pop("reader")
            else:
                corrupted[0]["reader"] = reader
            with self.subTest(reader=reader), self.assertRaises(SimulationError):
                _bind_inputs(self.exp, corrupted, self.root / str(reader))
        self.path.unlink()
        rebound = _bind_inputs(self.exp, inputs, self.root / "offline")
        self.assertEqual(Path(rebound.calibration.profile_path).read_bytes(), self.raw)
        estimate = create_performance_model(rebound).estimate_step("prefill", [4], [0])
        self.assertGreater(estimate.duration_ms, 0)

    def test_closed_loop_binds_snapshot_through_origin_ABA_and_rejects_missing_phase(self):
        first = predict_closed_loop(self.exp, study())
        self.assertEqual(first["status"], "COMPLETED")
        original = RuntimeCostModel.__init__
        observed = []
        def initialize(model, configured):
            observed.append(configured.calibration.profile_path)
            self.path.write_bytes(b'{"changed":"during-consumption"}')
            try:
                original(model, configured)
            finally:
                self.path.write_bytes(self.raw)
        with patch.object(RuntimeCostModel, "__init__", initialize):
            self.assertEqual(first, predict_closed_loop(self.exp, study()))
        self.assertTrue(observed)
        self.assertTrue(all(Path(path).resolve() != self.path.resolve() for path in observed))
        self.assertEqual(first["profile_inputs"][0]["reader"], "runtime_cost_v1")
        bad_spec = study()
        bad_spec["concurrency_grid"] = [2]
        bad = predict_closed_loop(self.exp, bad_spec)
        self.assertEqual(bad["status"], "UNRESOLVED")
        self.assertEqual(bad["runs"][0]["result"]["error_code"], "OUT_OF_CALIBRATION_DOMAIN")

    def test_general_cli_preserves_runtime_raw_snapshot_and_manifest(self):
        configured = replace(self.exp, workload=(RequestSpec("r", 0, 4, 2),))
        path = self.root / "experiment.json"
        mapping = experiment_dict(configured)
        mapping["calibration"]["profile_path"] = "cost.json"
        path.write_text(json.dumps(mapping))
        self.assertEqual(load_experiment(path), configured)
        out = self.root / "run"
        process = self.run_cli("simulate", "--experiment", path, "--out", out)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual((out / "runtime-cost.json").read_bytes(), self.raw)
        self.assertFalse((out / "profile.json").exists())
        result = json.loads((out / "result.json").read_text())
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["runtime_cost_reader"], "runtime_cost_v1")
        self.assertEqual(manifest["runtime_cost_sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(result["profile_provenance"]["snapshot_file"], "runtime-cost.json")
        self.assertEqual(result["counts"]["completed"], 1)
        predicted = self.root / "shape-prediction"
        process = self.run_cli("predict", "--experiment", path, "--out", predicted)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual((predicted / "runtime-cost.json").read_bytes(), self.raw)
        result = json.loads((predicted / "result.json").read_text())
        self.assertEqual(result["source"], "runtime")
        self.assertEqual(result["profile_provenance"]["reader"], "runtime_cost_v1")

    def test_phase_cli_supports_explicit_mixed_and_relative_runtime_path(self):
        path = self.root / "experiment.json"
        mapping = experiment_dict(self.exp)
        mapping["calibration"]["profile_path"] = "cost.json"
        path.write_text(json.dumps(mapping))
        out = self.root / "phase"
        process = self.run_cli("predict", "--experiment", path, "--phase", "mixed",
            "--query-tokens", 1, 4, "--context-lengths", 4, 0,
            "--request-phases", "decode", "prefill", "--out", out, cost=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        manifest = json.loads((out / "manifest.json").read_text())
        result = json.loads((out / "prediction.json").read_text())
        self.assertEqual(manifest["input_bytes_sha256"]["runtime_cost"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual((out / "runtime-cost.json").read_bytes(), self.raw)
        self.assertEqual(json.loads((out / "runtime-cost-input.json").read_text())["reader"], "runtime_cost_v1")
        self.assertEqual(result["request_phases"], ["decode", "prefill"])
        self.assertEqual(result["estimate"]["source"], "runtime_cost_calibrated")

    def test_comparison_keeps_core_boundary_and_distinguishes_memory_domain(self):
        identity = profile_identity(self.exp)
        self.assertEqual(validate_identity(identity), identity)
        contract = comparison_contract(self.exp)
        self.assertEqual(contract["timing_boundary"], BOUNDARY)
        self.assertEqual(contract["identity"]["runtime"]["timing_boundary"], BOUNDARY)
        changed = replace(self.exp, strategy=replace(self.exp.strategy,
            memory=replace(self.exp.strategy.memory, kv_pool_bytes=8 * 1024**2)))
        other = comparison_contract(changed)
        self.assertNotEqual(contract, other)
        output = self.root / "predicted"
        output.mkdir()
        (output / "result.json").write_text(json.dumps(comparison_prediction()))
        (output / "manifest.json").write_text(json.dumps({"backend": "simulated", "comparison_contract": contract}))
        observed = self.root / "observation.json"
        observed.write_text(json.dumps({"schema_version": 1, "kind": "measured_run", "comparison_contract": other,
            "source": {"kind": "test_fixture", "reference": "CPU fixture"}, "metrics": {"request_count": 2}}))
        with self.assertRaises(SimulationError):
            compare_runs(output, observed, allow_test_fixture=True)


if __name__ == "__main__":
    unittest.main()
