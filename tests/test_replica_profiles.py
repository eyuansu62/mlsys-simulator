"""Cluster online identities and local stage evidence stay distinct."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (AnalyticalModel, CalibrationSpec, DeviceSpec, Experiment,
                       InfraSpec, ProfileModel, RequestSpec, SimulationError,
                       StrategySpec, calibrate_dataset, compare_runs,
                       comparison_contract, predict_experiment, profile_identity,
                       write_json_exclusive)
from mlsys_sim.layout import replica_experiment
from mlsys_sim.profiles import validate_dataset, validate_identity, validate_profile


def cluster(*, replicas=2, tp=1, pp=1, workload=None):
    return Experiment(workload=workload or (RequestSpec("r", 0, 8, 2),),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(replicas * tp * pp))),
        strategy=StrategySpec(replicas=replicas, tensor_parallel_size=tp,
                               pipeline_parallel_size=pp))


def dataset(exp, *, prefill=5, decode=1):
    return {"schema_version": 1, "dataset_id": "replica-local-fixture",
            "identity": profile_identity(exp), "source": {"kind": "test_fixture",
            "reference": "Hand calculations, not hardware observations"},
            "records": [{"record_id": phase, "group_id": phase, "phase": phase,
                         "stage_index": 0, "query_tokens": [query], "context_lengths": [context],
                         "duration_ms": duration, "workspace_peak_bytes": 100}
                        for phase, query, context, duration in
                        [("prefill", 8, 0, prefill), ("decode", 1, 8, decode)]]}


class ReplicaIdentityTests(unittest.TestCase):
    def test_single_replica_identity_has_no_new_field_and_normalizes_explicit_one(self):
        identity = profile_identity(Experiment())
        self.assertEqual(set(identity["parallelism"]), {"tensor_parallel_size",
            "pipeline_parallel_size", "stage_layers", "shard_plan"})
        original = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        explicit = deepcopy(identity)
        explicit["parallelism"]["replicas"] = 1
        self.assertEqual(validate_identity(explicit), identity)
        self.assertEqual(json.dumps(profile_identity(Experiment()), sort_keys=True,
                                    separators=(",", ":")), original)

    def test_cluster_identity_preserves_replica_count_and_all_device_placement(self):
        exp = cluster(tp=2, pp=2)
        identity = profile_identity(exp)
        self.assertEqual(identity["parallelism"]["replicas"], 2)
        self.assertEqual([device["id"] for device in identity["devices"]], [f"gpu{i}" for i in range(8)])
        self.assertEqual(validate_identity(identity), identity)
        for replicas in (0, True, 3):
            invalid = deepcopy(identity)
            invalid["parallelism"]["replicas"] = replicas
            with self.subTest(replicas=replicas), self.assertRaises(SimulationError) as raised:
                validate_identity(invalid)
            self.assertEqual(raised.exception.code, "INVALID_PROFILE")

    def test_routing_is_online_contract_policy_not_stage_measurement_identity(self):
        exp = cluster()
        alternative = replace(exp, strategy=replace(exp.strategy, replica_routing="round_robin"))
        self.assertEqual(profile_identity(exp), profile_identity(alternative))
        self.assertNotEqual(comparison_contract(exp), comparison_contract(alternative))

    def comparison(self, root, predicted_contract, observed_contract):
        predicted = root / "run"
        predicted.mkdir()
        (predicted / "manifest.json").write_text(json.dumps({"backend": "simulated",
            "comparison_contract": predicted_contract}))
        (predicted / "result.json").write_text(json.dumps({"backend": "simulated",
            "status": "COMPLETED", "counts": {"total": 1}}))
        observed = root / "observed.json"
        observed.write_text(json.dumps({"schema_version": 1, "kind": "measured_run",
            "comparison_contract": observed_contract, "source": {"kind": "test_fixture",
            "reference": "Online comparison contract fixture only"}, "metrics": {"request_count": 1}}))
        return compare_runs(predicted, observed, allow_test_fixture=True)

    def test_online_compare_accepts_cluster_and_rejects_routing_or_placement_change(self):
        exp = cluster()
        base = comparison_contract(exp)
        with tempfile.TemporaryDirectory() as temporary:
            result = self.comparison(Path(temporary), base, base)
            self.assertEqual(result["metrics"]["request_count"]["absolute_error"], 0)
        changed_policy = comparison_contract(replace(exp,
            strategy=replace(exp.strategy, replica_routing="prefix_aware")))
        changed_devices = comparison_contract(replace(exp,
            infra=replace(exp.infra, devices=tuple(reversed(exp.infra.devices)))))
        changed_layout = comparison_contract(cluster(replicas=1, tp=2))
        for changed in (changed_policy, changed_devices, changed_layout):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(SimulationError) as raised:
                    self.comparison(Path(temporary), base, changed)
                self.assertEqual(raised.exception.code, "COMPARISON_MISMATCH")

    def test_original_comparison_contract_defaults_routing_policy(self):
        base = comparison_contract(Experiment())
        legacy = deepcopy(base)
        legacy["strategy"].pop("replica_routing")
        with tempfile.TemporaryDirectory() as temporary:
            result = self.comparison(Path(temporary), base, legacy)
        self.assertEqual(result["metrics"]["request_count"]["absolute_error"], 0)

    def test_step_datasets_and_profiles_require_one_replica_identity(self):
        data = dataset(cluster())
        with self.assertRaises(SimulationError) as raised:
            validate_dataset(data, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "INVALID_DATASET")
        local = replica_experiment(cluster(), 0)
        profile = calibrate_dataset(dataset(local), profile_version="local", allow_test_fixture=True)
        profile["identity"] = profile_identity(cluster())
        with self.assertRaises(SimulationError) as raised:
            validate_profile(profile, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "INVALID_PROFILE")


class ReplicaPredictionTests(unittest.TestCase):
    def profiled(self, root):
        exp = cluster()
        paths = []
        for index, duration in enumerate((5, 9)):
            local = replica_experiment(exp, index)
            profile = calibrate_dataset(dataset(local, prefill=duration, decode=index + 1),
                profile_version=f"local-{index}", allow_test_fixture=True)
            path = root / f"replica-{index}.json"
            write_json_exclusive(path, profile)
            paths.append(str(path))
        return replace(exp, calibration=CalibrationSpec(backend="profile",
            replica_profile_paths=tuple(paths), allow_test_fixture=True))

    def test_stage_model_constructors_refuse_unsliced_clusters(self):
        with self.assertRaises(SimulationError) as raised:
            AnalyticalModel(cluster())
        self.assertEqual(raised.exception.code, "UNSUPPORTED_MECHANISM")
        with tempfile.TemporaryDirectory() as temporary:
            exp = self.profiled(Path(temporary))
            with self.assertRaises(SimulationError) as raised:
                ProfileModel(exp)
            self.assertEqual(raised.exception.code, "UNSUPPORTED_MECHANISM")

    def test_shape_predictions_use_heterogeneous_local_devices_without_online_routing(self):
        exp = cluster(workload=(RequestSpec("r", 0, 256, 2),))
        devices = (replace(exp.infra.devices[0], compute_tflops=1, memory_bandwidth_gbps=1e6),
                   replace(exp.infra.devices[1], compute_tflops=10, memory_bandwidth_gbps=1e6))
        result = predict_experiment(replace(exp, infra=replace(exp.infra, devices=devices)))
        self.assertFalse(result["replica_routing_applied"])
        self.assertEqual(result["replica_count"], 2)
        self.assertTrue(all(value is None for value in result["online_metrics"].values()))
        self.assertEqual(result["online_metrics_reason"], "SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING")
        first, second = [item["prediction"] for item in result["replica_predictions"]]
        self.assertGreater(first["isolated_prefill_ms"], second["isolated_prefill_ms"])
        self.assertEqual([item["device_ids"] for item in result["replica_predictions"]], [["gpu0"], ["gpu1"]])
        self.assertEqual([rank["replica_index"] for rank in result["memory"]], [0, 1])
        self.assertNotIn("isolated_prefill_ms", result)

    def test_each_local_profile_retains_device_identity_duration_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            exp = self.profiled(Path(temporary))
            result = predict_experiment(exp)
        self.assertEqual([item["prediction"]["isolated_prefill_ms"]
                          for item in result["replica_predictions"]], [5, 9])
        self.assertEqual(result["profile_version"], "per_replica")
        self.assertEqual(result["calibration_source"], "test_fixture_profile")
        provenance = result["profile_provenance"]["replicas"]
        self.assertEqual([item["replica_index"] for item in provenance], [0, 1])
        self.assertEqual([item["profile_version"] for item in provenance], ["local-0", "local-1"])
        self.assertNotEqual(provenance[0]["sha256"], provenance[1]["sha256"])
        self.assertFalse(result["hardware_validated"])

    def test_swapped_profile_files_and_requested_version_cannot_silently_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            exp = self.profiled(Path(temporary))
            swapped = replace(exp, calibration=replace(exp.calibration,
                replica_profile_paths=tuple(reversed(exp.calibration.replica_profile_paths))))
            wrong_version = replace(exp, calibration=replace(exp.calibration, profile_version="missing"))
            for invalid in (swapped, wrong_version):
                with self.subTest(invalid=invalid), self.assertRaises(SimulationError) as raised:
                    predict_experiment(invalid)
                self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")


if __name__ == "__main__":
    unittest.main()
