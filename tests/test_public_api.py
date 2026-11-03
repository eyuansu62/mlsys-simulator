"""Examples that consume the library exclusively through its public imports."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import mlsys_sim
from mlsys_sim import (
    AnalyticalModel,
    CalibrationSpec,
    DeviceSpec,
    Experiment,
    InfraSpec,
    MemoryPlanner,
    RequestSpec,
    SimulationError,
    SimulationResult,
    StepEstimate,
    StrategySpec,
    experiment_from_dict,
    load_experiment,
    predict_experiment,
    simulate,
    validate_experiment,
)


class PublicAPITests(unittest.TestCase):
    def test_construct_and_simulate_through_package_imports(self):
        experiment = Experiment(
            workload=(RequestSpec("request", 0.0, 16, 3),),
            strategy=StrategySpec(scheduler="fifo"),
            calibration=CalibrationSpec(backend="synthetic", prefill_ms=10.0, decode_ms=2.0),
        )
        validate_experiment(experiment)
        run = simulate(experiment)
        self.assertIsInstance(run, SimulationResult)
        self.assertEqual(run.result["counts"]["completed"], 1)
        self.assertEqual(run.requests[0]["first_token_ms"], 10.0)
        self.assertEqual(run.requests[0]["completion_ms"], 14.0)

    def test_public_prediction_and_memory_apis_accept_same_experiment(self):
        experiment = experiment_from_dict({
            "workload": [{"request_id": "request", "arrival_ms": 0,
                          "input_tokens": 16, "output_tokens": 2}],
        })
        prediction = predict_experiment(experiment)
        estimate = AnalyticalModel(experiment).estimate_step("prefill", [16], [0])
        self.assertIsInstance(estimate, StepEstimate)
        self.assertGreater(estimate.duration_ms, 0)
        self.assertEqual(prediction["source"], "analytical")
        self.assertFalse(prediction["hardware_validated"])
        planner = MemoryPlanner(experiment)
        self.assertTrue(planner.reserve({"request": 16}, [16], "prefill").ok)
        planner.release("request")
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in planner.snapshot()))

    def test_load_configuration_without_internal_module_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps({"strategy": {"scheduler": "fifo"}}), encoding="utf-8")
            from_path = load_experiment(path)
            from_string = load_experiment(str(path))
        self.assertEqual(from_path, from_string)
        self.assertEqual(from_path.strategy.scheduler, "fifo")

    def test_pp_api_reports_analysis_but_rejects_online_simulation(self):
        experiment = replace(
            Experiment(),
            infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))),
            strategy=StrategySpec(pipeline_parallel_size=2),
        )
        report = predict_experiment(experiment)
        self.assertEqual(report["online_metrics_reason"], "PP_ONLINE_DES_NOT_IMPLEMENTED")
        with self.assertRaises(SimulationError) as raised:
            simulate(experiment)
        self.assertEqual(raised.exception.code, "UNSUPPORTED_MECHANISM")

    def test_declared_exports_are_available(self):
        self.assertEqual(len(mlsys_sim.__all__), len(set(mlsys_sim.__all__)))
        self.assertTrue(all(hasattr(mlsys_sim, name) for name in mlsys_sim.__all__))
        self.assertFalse(any(name.startswith("_") and name != "__version__"
                             for name in mlsys_sim.__all__))
        self.assertRegex(mlsys_sim.__version__, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
