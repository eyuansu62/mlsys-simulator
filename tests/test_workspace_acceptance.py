"""Workspace fixtures cannot inherit measured acceptance from other inputs.

All measured labels here exercise schema-only arithmetic, without GPU execution.
The native-stage holdout stub is inherited from the existing source-boundary
fixture; actual Workspace validation, snapshots, simulation and Compare execute.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_pp_communication_acceptance as acceptance_fixture
from mlsys_sim import SimulationError, replica_experiment
from mlsys_sim.cli import execute
from mlsys_sim.workspace import ATTRIBUTION, calibrate_workspace


class WorkspaceAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.support = acceptance_fixture.NativePPCommunicationAcceptanceTests()
        self.support.setUp()
        self.addCleanup(self.support.doCleanups)
        self.root = self.support.root

    def inputs(self, replicas=1, fixture_replica=0):
        plan, exp, _, _ = self.support.inputs(replicas=replicas, boundary_fixture_replica=-1)
        paths = []
        for replica in range(replicas):
            local = replica_experiment(exp, replica)
            stage = json.loads((self.root / f"train-{replica}.json").read_text())
            records = []
            for row in stage["records"]:
                records.append({key: deepcopy(row[key]) for key in (
                    "record_id", "group_id", "phase", "stage_index", "query_tokens", "context_lengths")})
                records[-1]["workspace_bytes_per_rank"] = [
                    {"tp_rank": rank, "peak_bytes": 32 + rank}
                    for rank in range(local.strategy.tensor_parallel_size)]
            source = {"kind": "test_fixture" if replica == fixture_replica else "measured",
                      "reference": "Schema-only Workspace arithmetic, no physical measurement"}
            data = {"schema_version": 1, "kind": "workspace_measurements", "dataset_id": f"workspace-{replica}",
                "identity": stage["identity"], "source": source, "attribution": deepcopy(ATTRIBUTION), "records": records}
            profile = calibrate_workspace(data, profile_version="workspace-acceptance", allow_test_fixture=True)
            path = self.support.write(f"workspace-{replica}.json", profile)
            paths.append(str(path))
        calibration = (replace(exp.calibration, workspace_profile_path=paths[0]) if replicas == 1 else
                       replace(exp.calibration, replica_workspace_profile_paths=tuple(paths)))
        exp = replace(exp, calibration=calibration)
        self.support.write("experiment.json", asdict(exp))
        return plan, paths

    def test_workspace_fixture_contaminates_single_and_replica_hardware_acceptance(self):
        for replicas in (1, 2):
            with self.subTest(replicas=replicas):
                plan, paths = self.inputs(replicas, fixture_replica=replicas - 1)
                result = self.support.run_plan(plan, name=f"acceptance-{replicas}")
                self.assertTrue(result["criteria_passed"])
                self.assertEqual(result["verdict"], "TEST_FIXTURE_ONLY")
                self.assertFalse(result["hardware_validated"])
                self.assertEqual(result["source_kinds"], ["measured", "test_fixture"])
                bindings = result["validation_scope"]["workspace_profiles"]
                self.assertEqual(len(bindings), replicas)
                for index, binding in enumerate(bindings):
                    original = Path(paths[index]).read_bytes()
                    self.assertEqual(binding["sha256"], hashlib.sha256(original).hexdigest())
                    snapshot = self.root / f"acceptance-{replicas}" / binding["snapshot_file"]
                    self.assertEqual(snapshot.read_bytes(), original)

    def test_source_permission_and_version_mismatch_are_rejected(self):
        plan, paths = self.inputs()
        with self.assertRaises(SimulationError) as raised:
            self.support.run_plan(plan, name="no-fixtures", fixtures=False)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        experiment = json.loads((self.root / "experiment.json").read_text())
        experiment["calibration"]["workspace_profile_version"] = "wrong"
        self.support.write("experiment.json", experiment)
        with self.assertRaises(SimulationError) as raised:
            self.support.run_plan(plan, name="wrong-version")
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
        self.assertTrue(Path(paths[0]).is_file())

    def test_read_once_snapshot_survives_original_workspace_file_change(self):
        plan, paths = self.inputs()
        original = Path(paths[0]).read_bytes()
        def change_after_capture(mode, experiment_path, output):
            Path(paths[0]).write_text("broken source after gate capture")
            return execute(mode, experiment_path, output)
        with patch("mlsys_sim.cli.execute", side_effect=change_after_capture):
            result = self.support.run_plan(plan, name="immutable")
        self.assertTrue(result["criteria_passed"])
        self.assertEqual(result["verdict"], "TEST_FIXTURE_ONLY")
        binding = result["validation_scope"]["workspace_profiles"][0]
        self.assertEqual((self.root / "immutable" / binding["snapshot_file"]).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
