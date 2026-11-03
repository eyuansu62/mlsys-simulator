"""Calibration and grouped holdout correctness, including provenance failures."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim.calibration import (calibrate_dataset, split_dataset,
                                    validate_holdout, write_json_exclusive)
from mlsys_sim.profiles import estimate_profile, profile_identity, shape_key
from mlsys_sim.schema import Experiment, SimulationError


def record(name, query, duration, group=None, workspace=None):
    return {"record_id": name, "group_id": group or name, "phase": "prefill",
            "stage_index": 0, "query_tokens": [query], "context_lengths": [0],
            "duration_ms": duration, "workspace_peak_bytes": workspace}


def dataset(records):
    return {"schema_version": 1, "dataset_id": "fixture-observations",
            "identity": profile_identity(Experiment()),
            "source": {"kind": "test_fixture", "reference": "Unit-test hand calculations only"},
            "records": records}


def fit(data=None, version="v1", **kwargs):
    if data is None:
        data = dataset([record("low-1", 4, 10, "low", 6),
                        record("low-2", 4, 14, "low", 8),
                        record("high", 8, 20, "high", 12)])
    return calibrate_dataset(data, profile_version=version, allow_test_fixture=True, **kwargs)


class CalibrationTests(unittest.TestCase):
    def test_repeat_fit_mean_maximum_hash_and_input_immutability(self):
        data = dataset([record("a", 4, 10, "configuration", 6),
                        record("b", 4, 14, "configuration", 8),
                        record("c", 8, 20, "other")])
        before = copy.deepcopy(data)
        profile = fit(data)
        self.assertEqual(data, before)
        self.assertEqual(profile["source"]["kind"], "test_fixture")
        sample = next(sample for sample in profile["samples"] if sample["query_tokens"] == [4])
        self.assertEqual(sample["duration_ms"], 12)
        self.assertEqual(sample["workspace_peak_bytes"], 8)
        self.assertEqual(sample["sample_count"], 2)
        variation = next(item for item in profile["notes"]["repeat_statistics"]
                         if item["shape_key"] == shape_key("prefill", [4], [0]))
        self.assertEqual((variation["duration_min_ms"], variation["duration_max_ms"]), (10, 14))
        self.assertAlmostEqual(variation["duration_sample_stdev_ms"], 8 ** 0.5)
        self.assertEqual(profile["training"]["record_ids"], ["a", "b", "c"])
        self.assertEqual(profile["training"]["group_ids"], ["configuration", "other"])
        canonical = json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":"))
        self.assertEqual(profile["training"]["dataset_sha256"],
                         hashlib.sha256(canonical.encode()).hexdigest())

    def test_parent_version_lineage_is_immutable_and_must_change(self):
        parent = fit()
        before = copy.deepcopy(parent)
        updated = fit(version="v2", parent_profile=parent)
        self.assertEqual(parent, before)
        self.assertEqual(updated["training"]["parent_profile_version"], "v1")
        self.assertNotEqual(parent["profile_id"], updated["profile_id"])
        with self.assertRaises(SimulationError) as raised:
            fit(parent_profile=parent)
        self.assertEqual(raised.exception.code, "PROFILE_VERSION_CONFLICT")

    def test_parent_identity_mismatch_is_rejected(self):
        parent = fit()
        parent["identity"]["runtime"]["version"] = "different-runtime-version"
        with self.assertRaises(SimulationError) as raised:
            fit(version="v2", parent_profile=parent)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_fixture_source_requires_explicit_opt_in_and_is_preserved(self):
        data = dataset([record("a", 4, 10)])
        with self.assertRaises(SimulationError):
            calibrate_dataset(data, profile_version="v1")
        self.assertEqual(fit(data)["source"], data["source"])
        data["source"]["kind"] = "simulated"
        with self.assertRaises(SimulationError):
            fit(data)

    def test_invalid_versions_and_durations_are_rejected(self):
        for version in [None, 1, "", " "]:
            with self.subTest(version=version), self.assertRaises(SimulationError):
                fit(version=version)
        for duration in [0, -1, float("nan"), float("inf"), True]:
            with self.subTest(duration=duration), self.assertRaises(SimulationError):
                fit(dataset([record("invalid", 4, duration)]))

    def test_extreme_finite_repeats_do_not_overflow_or_underflow_mean(self):
        for duration in [1e308, 5e-324]:
            with self.subTest(duration=duration):
                profile = fit(dataset([record("a", 4, duration), record("b", 4, duration)]))
                self.assertEqual(profile["samples"][0]["duration_ms"], duration)

    def test_ragged_shapes_remain_exact_shapes(self):
        a, b = record("ragged-a", 4, 20, "ragged"), record("ragged-b", 4, 24, "ragged")
        for item in [a, b]:
            item["query_tokens"], item["context_lengths"] = [4, 8], [0, 0]
        profile = fit(dataset([a, b]))
        estimate = estimate_profile(profile, "prefill", [4, 8], [0, 0], allow_test_fixture=True)
        self.assertEqual(estimate.duration_ms, 22)
        self.assertEqual(profile["training"]["shape_keys"], [shape_key("prefill", [4, 8], [0, 0])])


class GroupedSplitTests(unittest.TestCase):
    def test_deterministic_component_split_is_independent_of_input_order(self):
        data = dataset([record(f"r{i}", i + 1, i + 10, f"g{i}") for i in range(8)])
        before = copy.deepcopy(data)
        first = split_dataset(data, seed=9, holdout_fraction=0.25, allow_test_fixture=True)
        second = split_dataset(data, seed=9, holdout_fraction=0.25, allow_test_fixture=True)
        reordered = copy.deepcopy(data)
        reordered["records"].reverse()
        third = split_dataset(reordered, seed=9, holdout_fraction=0.25, allow_test_fixture=True)
        self.assertEqual(data, before)
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(len(first[0]["records"]), 6)
        self.assertEqual(len(first[1]["records"]), 2)
        self.assertEqual(first[0]["source"], data["source"])

    def test_shape_or_group_transitive_component_cannot_leak(self):
        data = dataset([record("a", 4, 10, "group-a"), record("b", 4, 12, "group-b"),
                        record("c", 8, 20, "group-b"), record("d", 16, 40, "independent")])
        training, holdout = split_dataset(data, holdout_groups=["group-a"], allow_test_fixture=True)
        self.assertEqual([r["record_id"] for r in training["records"]], ["d"])
        self.assertEqual([r["record_id"] for r in holdout["records"]], ["a", "b", "c"])
        self.assertFalse({_shape_for_test(r) for r in training["records"]}
                         & {_shape_for_test(r) for r in holdout["records"]})

    def test_unsplittable_or_invalid_partition_is_rejected(self):
        data = dataset([record("a", 4, 10), record("b", 8, 20)])
        for fraction in [0, 1, -0.5, float("nan"), True, 10 ** 400]:
            with self.subTest(fraction=fraction), self.assertRaises(SimulationError):
                split_dataset(data, holdout_fraction=fraction, allow_test_fixture=True)
        for groups in [[], ["missing"], ["a", "b"], "a"]:
            with self.subTest(groups=groups), self.assertRaises(SimulationError):
                split_dataset(data, holdout_groups=groups, allow_test_fixture=True)
        with self.assertRaises(SimulationError):
            split_dataset(data, seed=True, allow_test_fixture=True)
        with self.assertRaises(SimulationError) as raised:
            split_dataset(dataset([record("a", 4, 10), record("b", 4, 12)]), allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "INSUFFICIENT_HOLDOUT_GROUPS")


def _shape_for_test(row):
    return shape_key(row["phase"], row["query_tokens"], row["context_lengths"], row["stage_index"])


class HoldoutValidationTests(unittest.TestCase):
    def test_hand_computed_interpolated_holdout_reports_error_and_group_counts(self):
        holdout = dataset([record("mid-1", 6, 16, "middle"), record("mid-2", 6, 20, "middle")])
        report = validate_holdout(fit(), holdout, allow_test_fixture=True)
        self.assertEqual(report["summary"]["coverage"], 1)
        self.assertEqual(report["summary"]["mae_ms"], 2)
        self.assertAlmostEqual(report["summary"]["mape_percent"], 10)
        self.assertEqual(report["groups"]["middle"]["sample_count"], 2)
        self.assertTrue(report["test_fixture_only"])
        self.assertIsNone(report["online_sla"])
        self.assertIsNone(report["candidate_ranking"])
        self.assertEqual(report["records"][0]["method"], "interpolation")

    def test_record_group_and_configuration_leaks_are_independently_rejected(self):
        for observed in [record("low-1", 6, 16, "unseen"),
                         record("unseen", 6, 16, "low"),
                         record("unseen", 4, 12, "unseen")]:
            with self.subTest(record=observed), self.assertRaises(SimulationError) as raised:
                validate_holdout(fit(), dataset([observed]), allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "VALIDATION_DATA_LEAKAGE")

    def test_out_of_domain_is_reported_not_silently_extrapolated(self):
        report = validate_holdout(fit(), dataset([record("outside", 10, 24)]), allow_test_fixture=True)
        self.assertEqual(report["summary"]["coverage"], 0)
        self.assertIsNone(report["summary"]["mae_ms"])
        self.assertIsNone(report["summary"]["mape_percent"])
        self.assertEqual(report["records"][0]["error_code"], "OUT_OF_CALIBRATION_DOMAIN")
        self.assertIsNone(report["records"][0]["predicted_ms"])

    def test_explicit_extrapolation_is_counted(self):
        report = validate_holdout(fit(), dataset([record("outside", 10, 24)]),
                                  allow_test_fixture=True, allow_extrapolation=True)
        self.assertEqual(report["summary"]["coverage"], 1)
        self.assertEqual(report["summary"]["extrapolated_count"], 1)
        self.assertEqual(report["summary"]["mae_ms"], 0)

    def test_partial_coverage_keeps_rejections_and_reasons(self):
        report = validate_holdout(fit(), dataset([record("middle", 6, 16), record("outside", 10, 24)]),
                                  allow_test_fixture=True)
        self.assertEqual(report["summary"]["coverage"], 0.5)
        self.assertEqual(report["summary"]["predicted_count"], 1)
        self.assertEqual(report["summary"]["rejected_count"], 1)
        self.assertEqual(report["groups"]["outside"]["coverage"], 0)

    def test_holdout_identity_and_default_fixture_gate_are_enforced(self):
        profile = fit()
        holdout = dataset([record("middle", 6, 16)])
        with self.assertRaises(SimulationError):
            validate_holdout(profile, holdout)
        holdout["identity"]["runtime"]["version"] = "different-runtime"
        with self.assertRaises(SimulationError) as raised:
            validate_holdout(profile, holdout, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_relative_overflow_is_null_and_json_stays_finite(self):
        profile = fit(dataset([record("low", 4, 1e308), record("high", 8, 1e308)]))
        report = validate_holdout(profile, dataset([record("middle", 6, 5e-324)]), allow_test_fixture=True)
        self.assertIsNone(report["summary"]["mape_percent"])
        self.assertIsNone(report["records"][0]["absolute_relative_error"])
        json.dumps(report, allow_nan=False)


class ExclusiveArtifactTests(unittest.TestCase):
    def test_writing_refuses_overwrite_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new" / "v1.json"
            write_json_exclusive(destination, {"version": "v1"})
            previous = destination.read_bytes()
            with self.assertRaises(SimulationError) as raised:
                write_json_exclusive(destination, {"version": "changed"})
            self.assertEqual(raised.exception.code, "OUTPUT_EXISTS")
            self.assertEqual(destination.read_bytes(), previous)
            invalid = Path(directory) / "invalid.json"
            with self.assertRaises(SimulationError):
                write_json_exclusive(invalid, {"value": float("nan")})
            self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
