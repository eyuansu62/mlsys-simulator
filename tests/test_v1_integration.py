"""Cross-module V1 contracts: referenced profiles, workspace and strict controls."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec,
                       MemoryPlanner, SimulationError, StrategySpec,
                       calibrate_dataset, experiment_from_dict, load_experiment,
                       profile_identity, write_json_exclusive)


class V1IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def profiled(self, experiment=None, spaces=(1048576,)):
        base = experiment or Experiment()
        data = {"schema_version": 1, "dataset_id": "workspace-fixture",
                "identity": profile_identity(base),
                "source": {"kind": "test_fixture", "reference": "unit-test values, not hardware"},
                "records": [{"record_id": f"r{stage}", "group_id": f"g{stage}",
                             "phase": "prefill", "stage_index": stage,
                             "query_tokens": [4], "context_lengths": [0],
                             "duration_ms": 1, "workspace_peak_bytes": space}
                            for stage, space in enumerate(spaces)]}
        path = self.root / "profile.json"
        write_json_exclusive(path, calibrate_dataset(data, profile_version="workspace-v1",
                                                     allow_test_fixture=True))
        return replace(base, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                                                         profile_version="workspace-v1",
                                                         allow_test_fixture=True))

    def test_profile_workspace_can_reject_an_analytically_small_batch(self):
        experiment = self.profiled(spaces=(100 * 1024 * 1024,))
        planner = MemoryPlanner(experiment)
        before = planner.snapshot()
        decision = planner.reserve({"a": 4}, [4], "prefill")
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "WORKSPACE_BUDGET_EXCEEDED")
        self.assertEqual(decision.details["ranks"][0]["workspace_source"], "profile")
        self.assertEqual(planner.allocations, {})
        self.assertEqual(planner.snapshot(), before)

    def test_missing_workspace_is_an_explicit_assumption(self):
        planner = MemoryPlanner(self.profiled(spaces=(None,)))
        decision = planner.reserve({"a": 4}, [4], "prefill")
        self.assertTrue(decision.ok)
        self.assertEqual(decision.details["ranks"][0]["workspace_source"], "analytical_assumption")
        self.assertEqual(planner.snapshot()[0]["workspace_source"], "analytical_assumption")

    def test_smaller_profile_observation_does_not_relabel_an_analytical_peak(self):
        from dataclasses import asdict
        base = experiment_from_dict({"strategy": {"memory": {"workspace_base_bytes": 0,
                                                                 "workspace_bytes_per_token": 100}}})
        data = {"schema_version": 1, "dataset_id": "mixed-workspace-fixture",
                "identity": profile_identity(base),
                "source": {"kind": "test_fixture", "reference": "unit-test fixture"},
                "records": [{"record_id": str(q), "group_id": str(q), "phase": "prefill",
                             "stage_index": 0, "query_tokens": [q], "context_lengths": [0],
                             "duration_ms": 1, "workspace_peak_bytes": value}
                            for q, value in [(4, None), (8, 100)]]}
        path = self.root / "mixed.json"
        write_json_exclusive(path, calibrate_dataset(data, profile_version="v1", allow_test_fixture=True))
        values = asdict(base)
        values["calibration"].update(backend="profile", profile_path=str(path), allow_test_fixture=True)
        planner = MemoryPlanner(experiment_from_dict(values))
        self.assertTrue(planner.reserve({"r": 4}, [4], "prefill").ok)
        planner.release("r")
        self.assertTrue(planner.reserve({"r": 8}, [8], "prefill").ok)
        snapshot = planner.snapshot()[0]
        self.assertEqual(snapshot["workspace_peak_bytes"], 400)
        self.assertEqual(snapshot["workspace_source"], "analytical_assumption")
        self.assertEqual(snapshot["workspace_last_source"], "profile")

    def test_pp_workspace_peaks_remain_stage_specific(self):
        experiment = Experiment(infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
                                strategy=StrategySpec(pipeline_parallel_size=2))
        planner = MemoryPlanner(self.profiled(experiment, spaces=(1048576, 2097152)))
        self.assertTrue(planner.reserve({"r": 4}, [4], "prefill").ok)
        self.assertEqual([row["workspace_peak_bytes"] for row in planner.snapshot()], [1048576, 2097152])

    def test_profile_and_warmup_references_use_their_section_directories(self):
        directory = self.root / "sections"
        directory.mkdir()
        (directory / "calibration.json").write_text(json.dumps({
            "backend": "profile", "profile_path": "data/profile.json"}))
        (directory / "warmup.json").write_text(json.dumps([
            {"request_id": "warm", "arrival_ms": 0, "input_tokens": 4, "output_tokens": 1}]))
        (directory / "execution.json").write_text(json.dumps({"warmup": "warmup.json"}))
        main = self.root / "experiment.json"
        main.write_text(json.dumps({"calibration": "sections/calibration.json",
                                    "execution": "sections/execution.json"}))
        experiment = load_experiment(main)
        self.assertEqual(experiment.calibration.profile_path, str((directory / "data/profile.json").resolve()))
        self.assertEqual(experiment.execution.warmup[0].request_id, "warm")

    def test_invalid_profile_and_measurement_controls_are_not_ignored(self):
        invalid = [
            {"calibration": {"backend": "profile"}},
            {"calibration": {"allow_extrapolation": True}},
            {"calibration": {"runtime_name": " "}},
            {"calibration": {"timing_boundary": "kernel_only"}},
            {"evaluation": {"drain": False}},
            {"evaluation": {"max_drain_ms": 10}},
            {"evaluation": {"measurement_window_ms": 10, "drain": False, "max_drain_ms": 1}},
            {"execution": {"initial_state": "warm_prefix_cache"}},
            {"execution": {"warmup": [{"request_id": "bad", "arrival_ms": 0,
                                         "input_tokens": -1, "output_tokens": 1}]}},
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(SimulationError):
                experiment_from_dict(config)


if __name__ == "__main__":
    unittest.main()
