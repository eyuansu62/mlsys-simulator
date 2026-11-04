"""Explicit P2P policy domains and complete observed latency distributions."""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from mlsys_sim import SimulationError
from mlsys_sim.pp_communication import (PPCommunicationModel, calibrate_pp_communication,
    load_pp_communication_profile, validate_pp_communication_dataset,
    validate_pp_communication_holdout, validate_pp_communication_profile)
import test_pp_communication as legacy
import test_pp_communication_lookup as lookup


ENVIRONMENT = {"NCCL_P2P_LL_THRESHOLD": "0", "NCCL_P2P_LL128_THRESHOLD": "0"}


def policy_data(*, protocol=3, **kwargs):
    data = lookup.capture(**kwargs)
    if protocol >= 2:
        data["benchmark"].update(protocol=f"native_pp_primitives_v{protocol}",
                                communicator_initialization="lazy_world_and_pair_groups_v1")
    if protocol == 3:
        data["benchmark"].update(p2p_protocol_policy="simple_only_thresholds_v1",
                                p2p_protocol_environment=deepcopy(ENVIRONMENT))
        data["benchmark"]["stack"]["p2p_protocol_environment"] = deepcopy(ENVIRONMENT)
    return data


def calibrate(data, *, method="piecewise_size_lookup_v1", version="policy-v1", parent=None):
    return calibrate_pp_communication(data, profile_version=version, method=method,
                                      parent_profile=parent, allow_test_fixture=True)


class P2PPolicyDomainTests(unittest.TestCase):
    def test_v1_v2_profiles_preserve_existing_schema_and_estimates(self):
        baseline = None
        for protocol in (1, 2):
            data = policy_data(protocol=protocol)
            for method in ("nonnegative_affine_ols_v1", "piecewise_size_lookup_v1"):
                with self.subTest(protocol=protocol, method=method):
                    profile = calibrate(data, method=method)
                    self.assertEqual(validate_pp_communication_profile(profile, allow_test_fixture=True), profile)
                    self.assertNotIn("p2p_protocol_policy", profile["benchmark"])
                    self.assertNotIn("p2p_protocol_environment", profile["benchmark"])
                    estimate = PPCommunicationModel(legacy.experiment(tp=1), profile).estimate("p2p", 768, 1, tp_rank=0)
                    if method == "nonnegative_affine_ols_v1":
                        if baseline is None:
                            baseline = estimate["duration_ms"]
                        self.assertEqual(estimate["duration_ms"], baseline)
                        self.assertNotIn("model_method", estimate)
            self.assertNotIn("p2p_protocol_policy", data["benchmark"])

    def test_v3_roundtrip_only_adds_explicit_domain_without_changing_model(self):
        for method in ("nonnegative_affine_ols_v1", "piecewise_size_lookup_v1"):
            v2 = calibrate(policy_data(protocol=2), method=method)
            data = policy_data()
            before = deepcopy(data)
            profile = calibrate(data, method=method)
            self.assertEqual(data, before)
            self.assertEqual(profile["method"], method)
            self.assertEqual(profile["groups"], v2["groups"])
            self.assertEqual(profile.get("model_options"), v2.get("model_options"))
            self.assertEqual(profile["benchmark"]["p2p_protocol_environment"], ENVIRONMENT)
            self.assertEqual(profile["benchmark"]["stack"]["p2p_protocol_environment"], ENVIRONMENT)
            self.assertEqual(validate_pp_communication_profile(profile, allow_test_fixture=True), profile)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "v3.json"
                path.write_text(json.dumps(profile))
                self.assertEqual(load_pp_communication_profile(path, allow_test_fixture=True), profile)
            a, b = (PPCommunicationModel(legacy.experiment(tp=1), p).estimate("p2p", 768, 1, tp_rank=0)
                    for p in (v2, profile))
            self.assertEqual(a["duration_ms"], b["duration_ms"])
            self.assertEqual(b["provenance"]["protocol"], "native_pp_primitives_v3")

    def test_v3_requires_policy_and_both_complete_actual_environments(self):
        mutations = [
            lambda b: b.pop("p2p_protocol_policy"),
            lambda b: b.pop("p2p_protocol_environment"),
            lambda b: b.update(p2p_protocol_policy="NCCL_PROTO_Simple"),
            lambda b: b.pop("communicator_initialization"),
            lambda b: b["stack"].pop("p2p_protocol_environment"),
            lambda b: b["p2p_protocol_environment"].pop("NCCL_P2P_LL128_THRESHOLD"),
            lambda b: b["stack"]["p2p_protocol_environment"].pop("NCCL_P2P_LL_THRESHOLD"),
            lambda b: b["p2p_protocol_environment"].update(unrecognized="0"),
            lambda b: b["stack"]["p2p_protocol_environment"].update(unrecognized="0"),
        ]
        for mutation in mutations:
            data = policy_data()
            mutation(data["benchmark"])
            with self.subTest(mutation=mutation), self.assertRaises(SimulationError):
                validate_pp_communication_dataset(data, allow_test_fixture=True)
        for target in ("top", "stack"):
            for name in ENVIRONMENT:
                for invalid in (0, False, None, "", "4096", "00", "0 ", 0.0):
                    data = policy_data()
                    environment = (data["benchmark"]["p2p_protocol_environment"] if target == "top"
                                   else data["benchmark"]["stack"]["p2p_protocol_environment"])
                    environment[name] = invalid
                    with self.subTest(target=target, name=name, invalid=invalid), self.assertRaises(SimulationError):
                        validate_pp_communication_dataset(data, allow_test_fixture=True)
            for invalid in (None, [], "0", False):
                data = policy_data()
                if target == "top":
                    data["benchmark"]["p2p_protocol_environment"] = invalid
                else:
                    data["benchmark"]["stack"]["p2p_protocol_environment"] = invalid
                with self.subTest(target=target, invalid=invalid), self.assertRaises(SimulationError):
                    validate_pp_communication_dataset(data, allow_test_fixture=True)

    def test_policy_cannot_be_smuggled_into_legacy_top_level_benchmarks(self):
        for protocol in (1, 2):
            for name, value in (("p2p_protocol_policy", "simple_only_thresholds_v1"),
                                ("p2p_protocol_environment", ENVIRONMENT)):
                data = policy_data(protocol=protocol)
                data["benchmark"][name] = deepcopy(value)
                with self.subTest(protocol=protocol, field=name), self.assertRaises(SimulationError):
                    validate_pp_communication_dataset(data, allow_test_fixture=True)

    def test_profile_checks_actual_stack_policy_even_after_calibration(self):
        good = calibrate(policy_data())
        for invalid in ({}, {**ENVIRONMENT, "NCCL_P2P_LL_THRESHOLD": "4096"},
                        {**ENVIRONMENT, "NCCL_P2P_LL128_THRESHOLD": False}):
            profile = deepcopy(good)
            profile["benchmark"]["stack"]["p2p_protocol_environment"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(SimulationError):
                validate_pp_communication_profile(profile, allow_test_fixture=True)

    def test_parent_and_holdout_cannot_cross_protocol_even_with_identical_stack(self):
        v3 = policy_data()
        parent = calibrate(v3)
        for protocol in (1, 2):
            data = policy_data(protocol=protocol, dataset_id="independent")
            data["benchmark"]["stack"] = deepcopy(v3["benchmark"]["stack"])
            for action in (lambda: calibrate(data, parent=parent, version="new"),
                           lambda: validate_pp_communication_holdout(parent, data, allow_test_fixture=True)):
                with self.subTest(protocol=protocol), self.assertRaises(SimulationError) as raised:
                    action()
                self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_environment_hash_and_other_stack_fields_are_still_exact(self):
        parent = calibrate(policy_data())
        for name, value in (("communication_environment_sha256", "a" * 64), ("fixture", "other build")):
            data = policy_data(dataset_id="independent")
            data["benchmark"]["stack"][name] = value
            for action in (lambda: calibrate(data, parent=parent, version="new"),
                           lambda: validate_pp_communication_holdout(parent, data, allow_test_fixture=True)):
                with self.subTest(field=name), self.assertRaises(SimulationError) as raised:
                    action()
                self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
        valid = policy_data(dataset_id="independent")
        valid["benchmark"]["launch"] = {"session": "new"}
        self.assertEqual(validate_pp_communication_holdout(parent, valid, allow_test_fixture=True)
                         ["coverage"]["fraction"], 1)


class ObservedDistributionTests(unittest.TestCase):
    def report(self, values, *, sizes=(512, 768)):
        train = policy_data(curve=lambda size, repeat: 1.0)
        data = policy_data(curve=lambda size, repeat: values[repeat], repeats=len(values),
                           sizes=sizes, dataset_id="independent")
        return validate_pp_communication_holdout(calibrate(train), data, allow_test_fixture=True)

    def test_outlier_is_retained_in_distribution_and_existing_metrics(self):
        report = self.report([1.0, 1.0, 1.0, 1.0, 101.0])
        self.assertFalse(report["hardware_validated"])
        self.assertEqual(report["coverage"]["total_records"], 10)
        for row in report["by_message_size"]:
            distribution = row["observed_distribution_ms"]
            self.assertEqual(set(distribution), {"count", "mean", "min", "max", "p10", "p50", "p90", "p99", "stddev"})
            self.assertEqual(distribution["count"], 5)
            self.assertEqual(distribution["mean"], 21.0)
            self.assertEqual(distribution["min"], 1.0)
            self.assertEqual(distribution["max"], 101.0)
            self.assertEqual(distribution["p10"], 1.0)
            self.assertEqual(distribution["p50"], 1.0)
            self.assertAlmostEqual(distribution["p90"], 61.0)
            self.assertAlmostEqual(distribution["p99"], 97.0)
            self.assertEqual(distribution["stddev"], 40.0)
            self.assertEqual(distribution["mean"], row["observed_mean_ms"])
            self.assertNotIn("bimodal", row)
        self.assertEqual(report["metrics"]["max_absolute_error_ms"], 100.0)

    def test_rejected_sizes_still_describe_every_observation(self):
        report = self.report([1.0, 2.0, 3.0, 4.0, 100.0], sizes=(256, 4096))
        self.assertEqual(report["coverage"]["fraction"], 0)
        self.assertIsNone(report["metrics"]["mape"])
        for row in report["by_message_size"]:
            self.assertEqual(row["observed_distribution_ms"]["count"], 5)
            self.assertEqual(row["observed_distribution_ms"]["max"], 100)
            self.assertEqual(row["observed_distribution_ms"]["mean"], 22)
            self.assertIsNone(row["predicted_mean_ms"])

    def test_singleton_and_constant_samples_have_legitimate_zero_stddev(self):
        for values in ([0.25], [0.25] * 6):
            with self.subTest(samples=len(values)):
                report = self.report(values)
                for row in report["by_message_size"]:
                    dist = row["observed_distribution_ms"]
                    self.assertEqual(dist["count"], len(values))
                    self.assertEqual(dist["stddev"], 0.0)
                    for name in ("mean", "min", "max", "p10", "p50", "p90", "p99"):
                        self.assertEqual(dist[name], 0.25)

    def test_extreme_finite_positive_durations_remain_finite_json(self):
        for values in ([1e-300] * 6, [1e308] * 6,
                       [sys.float_info.max] * 5, [1e-300, 1e-300, 1e-300, 1e308, 1e308]):
            with self.subTest(values=values):
                report = self.report(values)
                json.dumps(report, allow_nan=False)
                for row in report["by_message_size"]:
                    dist = row["observed_distribution_ms"]
                    self.assertTrue(all(math.isfinite(value) for value in dist.values()))
                    self.assertEqual(dist["max"], max(values))


if __name__ == "__main__":
    unittest.main()
