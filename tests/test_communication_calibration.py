"""Hand-derived alpha-beta fits, provenance and communication input boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import DeviceSpec, Experiment, InfraSpec, SimulationError, StrategySpec, ring_allreduce_ms
from mlsys_sim.communication_calibration import calibrate_communication, load_communication_dataset
from mlsys_sim.profiles import profile_identity


def fixture(primitive="p2p", ranks=2, sizes=(0, 1000, 1000000), alpha_us=5.0, bandwidth_gbps=20.0):
    exp = Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(4))),
                     strategy=StrategySpec(tensor_parallel_size=4))
    rounds = 2 * (ranks - 1) if primitive == "allreduce" else 1
    traffic = 2 * (ranks - 1) / ranks if primitive == "allreduce" else 1.0
    return {
        "schema_version": 1, "kind": "communication_measurements", "dataset_id": "unit-fixture",
        "identity": profile_identity(exp),
        "source": {"kind": "test_fixture", "reference": "Hand-generated unit fixture; no hardware measurement"},
        "records": [
            {"record_id": f"{primitive}-{ranks}-{i}", "group_id": f"{primitive}-{ranks}",
             "primitive": primitive, "ranks": ranks, "message_bytes": size,
             "duration_ms": rounds * alpha_us / 1000 + traffic * size / (bandwidth_gbps * 1e6)}
            for i, size in enumerate(sizes)
        ],
    }


def fit(data, version="v1", parent=None):
    return calibrate_communication(data, profile_version=version, parent_profile=parent,
                                   allow_test_fixture=True)


class CommunicationFitTests(unittest.TestCase):
    def test_p2p_hand_calculation_and_si_units(self):
        data = fixture()
        result = fit(data)
        group = result["groups"][0]
        self.assertEqual(result["kind"], "communication_profile")
        self.assertAlmostEqual(group["alpha_us"], 5, places=10)
        self.assertAlmostEqual(group["bandwidth_gbps"], 20, places=9)
        self.assertAlmostEqual(group["beta_ms_per_byte"], 1 / (20 * 1e6), places=16)
        self.assertEqual(group["coverage"], {"message_bytes_min": 0, "message_bytes_max": 1000000,
                                            "record_count": 3, "distinct_message_sizes": 3})
        self.assertLess(group["mae_ms"], 1e-15)
        self.assertEqual(group["infra_patch"], {"link_latency_us": group["alpha_us"],
                                                "link_bandwidth_gbps": group["bandwidth_gbps"]})
        self.assertTrue(result["application"]["selection_required"])
        self.assertFalse(result["application"]["automatic_application"])
        self.assertTrue(result["application"]["do_not_add_to_whole_stage_lookup"])

    def test_ring_divides_rounds_and_per_rank_traffic_correctly(self):
        data = fixture("allreduce", 4, alpha_us=7, bandwidth_gbps=40)
        group = fit(data)["groups"][0]
        self.assertAlmostEqual(group["alpha_us"], 7, places=9)
        self.assertAlmostEqual(group["bandwidth_gbps"], 40, places=8)
        for record in data["records"]:
            predicted = ring_allreduce_ms(4, record["message_bytes"],
                                         group["bandwidth_gbps"], group["alpha_us"])
            self.assertAlmostEqual(predicted, record["duration_ms"], places=12)

    def test_primitive_and_rank_groups_are_not_mixed(self):
        data = fixture("p2p", 2, alpha_us=3, bandwidth_gbps=10)
        data["records"].extend(fixture("allreduce", 2, alpha_us=5, bandwidth_gbps=20)["records"])
        data["records"].extend(fixture("allreduce", 4, alpha_us=9, bandwidth_gbps=30)["records"])
        groups = {(group["primitive"], group["ranks"]): group for group in fit(data)["groups"]}
        self.assertEqual(len(groups), 3)
        for key, alpha, bandwidth in [(('p2p', 2), 3, 10), (('allreduce', 2), 5, 20),
                                      (('allreduce', 4), 9, 30)]:
            self.assertAlmostEqual(groups[key]["alpha_us"], alpha, places=9)
            self.assertAlmostEqual(groups[key]["bandwidth_gbps"], bandwidth, places=8)

    def test_negative_intercept_is_clamped_and_refit_through_origin(self):
        data = fixture(sizes=(1, 2, 3))
        for record, duration in zip(data["records"], [1, 3, 5]):
            record["duration_ms"] = duration
        group = fit(data)["groups"][0]
        # Unconstrained line is 2*x - 1; boundary solution slope is sum(x*y)/sum(x*x).
        expected = (1 * 1 + 2 * 3 + 3 * 5) / (1**2 + 2**2 + 3**2)
        self.assertTrue(group["intercept_clamped"])
        self.assertEqual(group["alpha_us"], 0)
        self.assertAlmostEqual(group["fit_slope_ms_per_byte"], expected)
        self.assertAlmostEqual(group["bandwidth_gbps"], 1 / (expected * 1e6))

    def test_repeats_contribute_to_ols_and_provenance(self):
        data = fixture(sizes=(0, 0, 1000, 1000))
        group = fit(data)["groups"][0]
        self.assertEqual(group["coverage"]["record_count"], 4)
        self.assertEqual(group["coverage"]["distinct_message_sizes"], 2)
        self.assertEqual(group["record_ids"], sorted(row["record_id"] for row in data["records"]))

    def test_new_version_preserves_parent_and_does_not_reuse_parameters(self):
        data = fixture()
        parent = fit(data)
        original, original_data = deepcopy(parent), deepcopy(data)
        child = fit(data, "v2", parent)
        self.assertEqual(parent, original)
        self.assertEqual(data, original_data)
        self.assertEqual(child["training"]["parent_profile_version"], "v1")
        self.assertEqual(child["training"]["dataset_sha256"], parent["training"]["dataset_sha256"])
        self.assertEqual(len(child["training"]["dataset_sha256"]), 64)
        self.assertFalse(child["application"]["parent_parameters_reused"])
        with self.assertRaises(SimulationError) as raised:
            fit(data, "v1", parent)
        self.assertEqual(raised.exception.code, "PROFILE_VERSION_CONFLICT")

    def test_parent_identity_mismatch_is_not_silently_recalibrated(self):
        data = fixture()
        parent = fit(data)
        data["identity"]["topology"]["name"] = "different-node-topology"
        with self.assertRaises(SimulationError) as raised:
            fit(data, "v2", parent)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")


class CommunicationValidationTests(unittest.TestCase):
    def test_fixture_requires_opt_in_and_never_becomes_measured(self):
        data = fixture()
        with self.assertRaises(SimulationError) as raised:
            calibrate_communication(data, profile_version="v1")
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        self.assertEqual(fit(data)["source"]["kind"], "test_fixture")
        data["source"]["kind"] = "simulated"
        with self.assertRaises(SimulationError) as raised:
            fit(data)
        self.assertEqual(raised.exception.code, "INVALID_COMMUNICATION_DATASET")

    def test_measured_source_requires_full_explicit_identity(self):
        data = fixture()
        data["source"]["kind"] = "measured"
        with self.assertRaises(SimulationError) as raised:
            calibrate_communication(data, profile_version="v1")
        self.assertEqual(raised.exception.code, "INVALID_COMMUNICATION_DATASET")
        self.assertIn("Measured identity", str(raised.exception))

    def test_rank_and_primitive_limits(self):
        for primitive, ranks in [('allreduce', 1), ('allreduce', 5), ('p2p', 3), ('unknown', 2),
                                 ('p2p', True), ('allreduce', 2.0)]:
            with self.subTest(primitive=primitive, ranks=ranks):
                data = fixture()
                data["records"][0].update(primitive=primitive, ranks=ranks)
                with self.assertRaises(SimulationError) as raised:
                    fit(data)
                self.assertEqual(raised.exception.code, "INVALID_COMMUNICATION_DATASET")

    def test_unknown_fields_types_duplicate_ids_and_nonfinite_durations(self):
        mutations = [
            lambda data: data.update(notes="unsupported metadata"),
            lambda data: data["source"].update(extra="unknown"),
            lambda data: data["records"][0].update(extra="unknown"),
            lambda data: data["records"][1].update(record_id=data["records"][0]["record_id"]),
            lambda data: data["records"][0].update(message_bytes=True),
            lambda data: data["records"][0].update(message_bytes=-1),
            lambda data: data["records"][0].update(message_bytes=2**63),
            lambda data: data["records"][0].update(duration_ms=float('nan')),
            lambda data: data["records"][0].update(duration_ms=float('inf')),
            lambda data: data["records"][0].update(duration_ms=0),
            lambda data: data["records"][0].update(duration_ms=True),
        ]
        for mutation in mutations:
            data = fixture()
            mutation(data)
            with self.subTest(data=data):
                with self.assertRaises(SimulationError) as raised:
                    fit(data)
                self.assertEqual(raised.exception.code, "INVALID_COMMUNICATION_DATASET")

    def test_singular_or_nonpositive_slope_is_rejected(self):
        for sizes, durations, code in [((10, 10), (1, 2), "INSUFFICIENT_COMMUNICATION_SAMPLES"),
                                       ((0, 1000), (1, 1), "INVALID_COMMUNICATION_FIT"),
                                       ((0, 1000), (2, 1), "INVALID_COMMUNICATION_FIT")]:
            data = fixture(sizes=sizes)
            for record, duration in zip(data["records"], durations):
                record["duration_ms"] = duration
            with self.subTest(sizes=sizes, durations=durations):
                with self.assertRaises(SimulationError) as raised:
                    fit(data)
                self.assertEqual(raised.exception.code, code)

    def test_malformed_parent_is_rejected(self):
        data = fixture()
        for mutate in [lambda parent: parent.update(extra="unknown"),
                       lambda parent: parent['training'].update(dataset_sha256='not-a-hash'),
                       lambda parent: parent['groups'][0]['infra_patch'].update(link_latency_us=999),
                       lambda parent: parent['groups'][0].update(ranks=5),
                       lambda parent: parent['application'].update(selection_required=1),
                       lambda parent: parent['groups'][0]['coverage'].update(distinct_message_sizes=999)]:
            parent = fit(data)
            mutate(parent)
            with self.subTest(parent=parent):
                with self.assertRaises(SimulationError) as raised:
                    fit(data, "v2", parent)
                self.assertEqual(raised.exception.code, "INVALID_COMMUNICATION_PROFILE")

    def test_loader_rejects_duplicate_json_keys_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'measurements.json'
            path.write_text(json.dumps(fixture()), encoding='utf-8')
            with self.assertRaises(SimulationError) as raised:
                load_communication_dataset(path)
            self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
            loaded = load_communication_dataset(path, allow_test_fixture=True)
            self.assertEqual(loaded['source']['kind'], 'test_fixture')
            path.write_text('{"schema_version":1,"schema_version":1}', encoding='utf-8')
            with self.assertRaises(SimulationError) as raised:
                load_communication_dataset(path, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, 'INVALID_COMMUNICATION_DATASET')

    def test_file_errors_and_empty_or_malformed_collections_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SimulationError) as raised:
                load_communication_dataset(Path(directory)/'missing.json')
            self.assertEqual(raised.exception.code, 'INVALID_COMMUNICATION_DATASET')
        for value in [[], {}, dict(fixture(), records=[]), dict(fixture(), schema_version=True),
                      dict(fixture(), kind='measured_run')]:
            with self.subTest(value=value):
                with self.assertRaises(SimulationError):
                    fit(value)


if __name__ == '__main__':
    unittest.main()
