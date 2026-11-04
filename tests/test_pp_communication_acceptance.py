"""Native boundary fixtures cannot acquire a hardware PASS through measured labels.

Every declared measured source below is schema-only unit arithmetic, not GPU
measurement. Only the stage holdout summary is mocked to isolate acceptance
source propagation; native Profile validation, snapshots, simulation and Compare
execute normally.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import (CalibrationSpec, DeviceSpec, EvaluationSpec, Experiment, InfraSpec,
    ModelSpec, RequestSpec, SimulationError, StrategySpec, comparison_contract,
    profile_identity, replica_experiment)
from mlsys_sim.hardware_validation import run_hardware_validation
from mlsys_sim.pp_communication import calibrate_pp_communication
from mlsys_sim.pp_transport import native_pp_descriptor
from test_pp_communication_integration import communication_dataset


class NativePPCommunicationAcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def inputs(self, *, replicas=1, boundary_fixture_replica=0):
        model = ModelSpec(name="schema-only-model", revision="schema-only-revision",
            weight_dtype="float16", kv_dtype="float16", activation_dtype="float16")
        exp = Experiment(workload=tuple(RequestSpec(chr(97 + i), 0, 4, 3) for i in range(replicas)),
            infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}", name=f"AMD-schema-only-{i}")
                                         for i in range(2 * replicas))),
            strategy=StrategySpec(model=model, replicas=replicas, pipeline_parallel_size=2,
                pipeline_transport="vllm_native", pipeline_max_inflight_microbatches=1,
                max_batch_tokens=16, max_batch_size=1, scheduler="fifo", replica_routing="round_robin"),
            calibration=CalibrationSpec(runtime_name="schema-only-runtime", runtime_version="1.0",
                driver_version="schema-only-driver", rocm_version="schema-only-rocm", rccl_version="schema-only-rccl"),
            evaluation=EvaluationSpec(ttft_sla_ms=20, tpot_sla_ms=5))
        measured = {"kind": "measured", "reference": "Schema-only unit arithmetic; no hardware measurement"}
        steps, comm_paths = [], []
        original_profiles = []
        for replica in range(replicas):
            local = replica_experiment(exp, replica)
            rows = []
            for stage in range(2):
                for phase, query, context in (("prefill", 4, 0), ("decode", 1, 4), ("decode", 1, 5)):
                    descriptor = native_pp_descriptor(local, query)
                    rows.append({"record_id": f"s{stage}-{phase}-{context}",
                        "group_id": f"s{stage}-{phase}-{context}", "phase": phase, "stage_index": stage,
                        "query_tokens": [query], "context_lengths": [context], "duration_ms": 1.0,
                        "workspace_peak_bytes": None,
                        "pp_transport": {"outgoing" if stage == 0 else "incoming": descriptor}})
            dataset = {"schema_version": 1, "dataset_id": f"schema-step-{replica}",
                "identity": profile_identity(local), "source": measured, "records": rows}
            holdout = deepcopy(dataset)
            holdout["dataset_id"] += "-holdout"
            holdout["records"] = []
            for stage in range(2):
                holdout["records"].append({"record_id": f"held-{stage}", "group_id": f"held-{stage}",
                    "phase": "prefill", "stage_index": stage, "query_tokens": [6], "context_lengths": [0],
                    "duration_ms": 1.0, "workspace_peak_bytes": None,
                    "pp_transport": {"outgoing" if stage == 0 else "incoming": native_pp_descriptor(local, 6)}})
            self.write(f"train-{replica}.json", dataset)
            self.write(f"hold-{replica}.json", holdout)
            steps.append({"replica_index": replica, "training_dataset": f"train-{replica}.json",
                "holdout_dataset": f"hold-{replica}.json", "profile_version": "schema-step-v1"})
            comm_data = communication_dataset(local, label=f"schema-comm-{replica}")
            if replica != boundary_fixture_replica:
                comm_data["source"] = measured
                comm_data["benchmark"]["stack"] = {"torch_version": "schema-only-torch",
                    "hip_version": "schema-only-hip", "rccl_version": "schema-only-rccl",
                    "rccl_library_sha256": "a" * 64, "communication_environment_sha256": "b" * 64}
            profile = calibrate_pp_communication(comm_data, profile_version="schema-comm-v1", allow_test_fixture=True)
            path = self.write(f"comm-{replica}.json", profile)
            comm_paths.append(str(path))
            original_profiles.append(path.read_bytes())
        calibration = (replace(exp.calibration, native_pp_communication_profile_path=comm_paths[0])
            if replicas == 1 else replace(exp.calibration, replica_native_pp_communication_profile_paths=tuple(comm_paths)))
        exp = replace(exp, calibration=calibration)
        self.write("experiment.json", asdict(exp))
        # One per-stage millisecond plus two directional primitive messages.
        ttft = 2 + 2 * (.01 + 1e-6 * 4 * model.hidden_size * model.activation_bytes)
        tpot = 2 + 2 * (.01 + 1e-6 * model.hidden_size * model.activation_bytes)
        observed = {"schema_version": 1, "kind": "measured_run", "source": measured,
            "comparison_contract": comparison_contract(exp),
            "metrics": {"ttft_p50_ms": ttft, "tpot_p50_ms": tpot,
                "completion_rate": 1, "request_count": replicas, "sla_satisfied": True},
            "requests": [{"request_id": chr(97 + i), "ttft_ms": ttft, "tpot_ms": tpot,
                "end_to_end_ms": ttft + 2 * tpot} for i in range(replicas)]}
        self.write("observed.json", observed)
        plan = {"schema_version": 1, "steps": steps,
            "online": [{"experiment": "experiment.json", "observation": "observed.json", "group_id": "schema-workload"}],
            "thresholds": {"min_holdout_records": 1, "min_holdout_coverage": 1,
                "max_step_mape_percent": 1, "min_online_requests": replicas, "min_online_repeats": 1,
                "max_online_relative_error": {"ttft_p50_ms": .001, "tpot_p50_ms": .001},
                "require_sla_agreement": True}}
        return plan, exp, comm_paths, original_profiles

    @staticmethod
    def passing_holdout(*args, **kwargs):
        # The native stage lookup remains exact-only in production. This mock
        # only lets the source-boundary acceptance regression reach all-pass.
        return {"summary": {"sample_count": 2, "coverage": 1.0, "mape_percent": 0.0},
                "records": [{"shape_key": '["prefill",0,[6],[0]]'}]}

    def run_plan(self, plan, *, name="acceptance", fixtures=True):
        with patch("mlsys_sim.hardware_validation.validate_holdout", side_effect=self.passing_holdout):
            return run_hardware_validation(plan, self.root / name, base_dir=self.root,
                                           allow_test_fixture=fixtures)

    def test_fixture_boundary_prevents_otherwise_measured_acceptance_and_keeps_bytes(self):
        plan, _, paths, originals = self.inputs()
        result = self.run_plan(plan)
        self.assertTrue(result["criteria_passed"])
        self.assertEqual(result["verdict"], "TEST_FIXTURE_ONLY")
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(result["source_kinds"], ["measured", "test_fixture"])
        self.assertFalse(result["validation_scope"]["communication_prediction_validated"])
        bindings = result["validation_scope"]["native_pp_communication_profiles"]
        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings[0]["source_kind"], "test_fixture")
        expected_hash = hashlib.sha256(originals[0]).hexdigest()
        self.assertEqual(bindings[0]["sha256"], expected_hash)
        snapshot = self.root / "acceptance/online-0/native-communication-r0.json"
        self.assertEqual(snapshot.read_bytes(), originals[0])
        effective = json.loads((self.root / "acceptance/online-0/experiment.json").read_text())
        self.assertEqual(effective["calibration"]["native_pp_communication_profile_path"], str(snapshot.resolve()))
        predicted_snapshot = self.root / "acceptance/online-0/predicted/communication-profiles/replica-0.json"
        self.assertEqual(predicted_snapshot.read_bytes(), originals[0])
        self.assertIn(expected_hash, {item["sha256"] for item in result["input_files"] if "sha256" in item})
        self.assertEqual(Path(paths[0]).read_bytes(), originals[0])

    def test_one_fixture_among_replica_boundary_profiles_taints_complete_verdict(self):
        plan, _, _, originals = self.inputs(replicas=2, boundary_fixture_replica=1)
        result = self.run_plan(plan)
        self.assertTrue(result["criteria_passed"])
        self.assertEqual(result["verdict"], "TEST_FIXTURE_ONLY")
        bindings = result["validation_scope"]["native_pp_communication_profiles"]
        self.assertEqual([row["source_kind"] for row in bindings], ["measured", "test_fixture"])
        for replica in range(2):
            path = self.root / f"acceptance/online-0/native-communication-r{replica}.json"
            self.assertEqual(path.read_bytes(), originals[replica])
        effective = json.loads((self.root / "acceptance/online-0/experiment.json").read_text())
        self.assertEqual(effective["calibration"]["replica_native_pp_communication_profile_paths"],
            [str((self.root / f"acceptance/online-0/native-communication-r{replica}.json").resolve()) for replica in range(2)])

    def test_fixture_permission_and_native_identity_or_version_are_enforced_before_prediction(self):
        plan, exp, paths, _ = self.inputs()
        with self.assertRaises(SimulationError) as raised:
            self.run_plan(plan, fixtures=False)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        current = json.loads(Path(paths[0]).read_text())
        current["identity"]["devices"].reverse()
        Path(paths[0]).write_text(json.dumps(current))
        with self.assertRaises(SimulationError) as raised:
            self.run_plan(plan, name="bad-identity")
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
        _, exp, _, _ = self.inputs()
        self.write("experiment.json", asdict(replace(exp, calibration=replace(exp.calibration,
            native_pp_communication_profile_version="wrong-version"))))
        with self.assertRaises(SimulationError) as raised:
            self.run_plan(plan, name="bad-version")
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_checked_profile_snapshot_survives_original_mutation_before_execute(self):
        plan, _, paths, originals = self.inputs()
        from mlsys_sim.cli import execute

        def mutate_then_execute(*args, **kwargs):
            Path(paths[0]).write_text("changed after acceptance captured original bytes")
            return execute(*args, **kwargs)

        with patch("mlsys_sim.cli.execute", side_effect=mutate_then_execute):
            result = self.run_plan(plan)
        self.assertTrue(result["criteria_passed"])
        self.assertEqual(result["verdict"], "TEST_FIXTURE_ONLY")
        self.assertEqual((self.root / "acceptance/online-0/native-communication-r0.json").read_bytes(), originals[0])
        self.assertEqual((self.root / "acceptance/online-0/predicted/communication-profiles/replica-0.json").read_bytes(), originals[0])


if __name__ == "__main__":
    unittest.main()
