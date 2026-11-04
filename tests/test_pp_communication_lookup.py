"""Size lookup preserves measured nonmonotonicity and honest holdout coverage."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import SimulationError
from mlsys_sim.pp_communication import (PPCommunicationModel, calibrate_pp_communication,
    load_pp_communication_profile, validate_pp_communication_dataset,
    validate_pp_communication_holdout, validate_pp_communication_profile)
import test_pp_communication as legacy


METHOD = "piecewise_size_lookup_v1"


def capture(curve=None, *, sizes=(512, 1024, 2048), repeats=5, dataset_id="train", exp=None):
    data = legacy.dataset(exp or legacy.experiment(tp=1), sizes=sizes, repeats=repeats, dataset_id=dataset_id)
    if curve is not None:
        for row in data["records"]:
            repeat = int(row["record_id"].rsplit("repeat", 1)[1])
            value = curve(row["message_bytes"], repeat)
            row["duration_ms"] = value
            row["rank_durations_ms"] = [{"rank": rank, "duration_ms": value} for rank in row["participants"]]
    return data


def lookup(data=None, **kwargs):
    return calibrate_pp_communication(capture() if data is None else data, profile_version="lookup-v1",
                                      allow_test_fixture=True, method=METHOD, **kwargs)


def validate(profile):
    return validate_pp_communication_profile(profile, allow_test_fixture=True)


def protocol_v2(data):
    data = deepcopy(data)
    data["benchmark"].update(protocol="native_pp_primitives_v2",
                            communicator_initialization="lazy_world_and_pair_groups_v1")
    return data


class PiecewiseSizeLookupTests(unittest.TestCase):
    def test_default_affine_schema_and_opt_in_lookup_are_distinct(self):
        data = capture()
        profile = legacy.fit(data)
        self.assertEqual(profile, calibrate_pp_communication(data, profile_version="comm-v1",
                         method="nonnegative_affine_ols_v1", allow_test_fixture=True))
        self.assertNotIn("model_options", profile)
        self.assertNotIn("knots", profile["groups"][0])
        new = lookup(data)
        self.assertEqual(new["method"], METHOD)
        self.assertNotIn("intercept_ms", new["groups"][0])
        self.assertEqual(new, validate(new))
        with self.assertRaises(SimulationError):
            calibrate_pp_communication(data, profile_version="bad", method="auto", allow_test_fixture=True)
        profile["model_options"] = new["model_options"]
        with self.assertRaises(SimulationError):
            validate(profile)

    def test_mean_retains_slow_samples_and_recomputable_distribution(self):
        data = capture(lambda size, repeat: [0.1, 0.1, 0.1, 0.1, 1.1][repeat])
        before = deepcopy(data)
        profile = lookup(data)
        self.assertEqual(data, before)
        group = profile["groups"][0]
        for knot in group["knots"]:
            self.assertAlmostEqual(knot["duration_ms"], 0.3)
            self.assertEqual(knot["duration_ms"], knot["mean_ms"])
            self.assertEqual(knot["median_ms"], 0.1)
            self.assertEqual(knot["min_ms"], 0.1)
            self.assertEqual(knot["max_ms"], 1.1)
            self.assertAlmostEqual(knot["p10_ms"], 0.1)
            self.assertAlmostEqual(knot["p90_ms"], 0.7)
            self.assertAlmostEqual(knot["stddev_ms"], 0.4)
            self.assertEqual(knot["sample_count"], 5)
            self.assertEqual(len(knot["samples"]), 5)
        self.assertEqual(group["coverage"]["record_count"], len(data["records"]))
        self.assertAlmostEqual(group["mae_ms"], 0.32)
        self.assertAlmostEqual(group["max_absolute_error_ms"], 0.8)
        self.assertEqual(set(group["record_ids"]), {row["record_id"] for row in data["records"]})

    def test_flat_and_decreasing_curves_do_not_require_a_positive_global_slope(self):
        exp = legacy.experiment(tp=1)
        for values in ((0.2, 0.2, 0.2), (0.3, 0.2, 0.15), (0.3, 0.2, 0.25)):
            with self.subTest(values=values):
                curve = dict(zip((512, 1024, 2048), values))
                profile = lookup(capture(lambda size, repeat: curve[size]))
                model = PPCommunicationModel(exp, profile)
                result = model.estimate("p2p", 768, 1, tp_rank=0)
                self.assertAlmostEqual(result["duration_ms"], (values[0] + values[1]) / 2)
                self.assertFalse(result["exact"])
                self.assertEqual(result["interpolation"], "linear_bytes")
                self.assertTrue(all(edge["supported"] for edge in profile["groups"][0]["intervals"]))

    def test_exact_and_interpolated_predictions_carry_only_compact_contributors(self):
        profile = lookup()
        model = PPCommunicationModel(legacy.experiment(tp=1), profile)
        exact = model.estimate("p2p", 1024, 1, tp_rank=0)
        interior = model.estimate("p2p", 1536, 1, tp_rank=0)
        self.assertTrue(exact["exact"])
        self.assertIsNone(exact["interval"])
        self.assertEqual(exact["model_method"], METHOD)
        self.assertEqual(exact["interpolation"], "exact")
        self.assertEqual(len(exact["contributors"]), 1)
        self.assertEqual(len(interior["contributors"]), 2)
        self.assertEqual(set(interior["contributors"][0]), {"message_bytes", "mean_ms", "sample_count"})
        self.assertNotIn("samples", json.dumps(interior))
        self.assertNotIn("benchmark", interior["provenance"])
        self.assertEqual(interior["provenance"]["profile_sha256"], model.provenance["profile_sha256"])
        self.assertEqual(interior["source"], "test_fixture_native_pp_communication_profile")
        self.assertFalse(interior["extrapolated"])

    def test_gap_and_jump_guards_keep_endpoints_but_never_fill_holes(self):
        cases = [((512, 4096), (0.1, 0.2), "size_gap"),
                 ((512, 1024), (1.0, 0.1), "downward_jump"),
                 ((512, 1024), (0.1, 1.0), "upward_jump")]
        base = legacy.experiment(tp=1)
        exp = replace(base, calibration=replace(base.calibration, native_pp_communication_allow_extrapolation=True))
        for sizes, times, reason in cases:
            with self.subTest(reason=reason):
                curve = dict(zip(sizes, times))
                profile = lookup(capture(lambda size, repeat: curve[size], sizes=sizes))
                self.assertEqual(profile["groups"][0]["intervals"][0]["reason"], reason)
                model = PPCommunicationModel(exp, profile)
                for size, duration in curve.items():
                    self.assertEqual(model.estimate("p2p", size, 1, tp_rank=0)["duration_ms"], duration)
                for size in ((sizes[0] + sizes[1]) // 2, 256, 8192):
                    with self.assertRaises(SimulationError) as error:
                        model.estimate("p2p", size, 1, tp_rank=0)
                    self.assertEqual(error.exception.code, "OUT_OF_COMMUNICATION_CALIBRATION_DOMAIN")

    def test_upward_guard_normalizes_by_size_and_small_time_difference_is_ignored(self):
        for sizes, times in (((512, 2048), (0.1, 0.6)), ((512, 1024), (0.001, 0.049))):
            with self.subTest(sizes=sizes, times=times):
                curve = dict(zip(sizes, times))
                profile = lookup(capture(lambda size, repeat: curve[size], sizes=sizes))
                edge = profile["groups"][0]["intervals"][0]
                self.assertEqual(edge, {"left_bytes": sizes[0], "right_bytes": sizes[1], "supported": True, "reason": None})

    def test_each_placement_needs_two_sizes_with_five_observations_each(self):
        for data in (capture(sizes=(512,)), capture(repeats=4), capture()):
            if len(data["records"]) == 15:
                data["records"].pop()
            with self.subTest(records=len(data["records"])), self.assertRaises(SimulationError):
                lookup(data)
        data = capture(exp=legacy.experiment(tp=2, pp=3))
        profile = lookup(data)
        self.assertEqual(len(profile["groups"]), 6)
        data["records"].pop()
        with self.assertRaises(SimulationError):
            lookup(data)

    def test_all_placements_and_dtype_divisibility_remain_strict(self):
        exp = legacy.experiment(tp=2, pp=3)
        profile = lookup(capture(exp=exp))
        model = PPCommunicationModel(exp, profile)
        for stage in (1, 2):
            for rank in (0, 1):
                self.assertEqual(model.estimate("p2p", 768, stage, tp_rank=rank)["participants"],
                                 [(stage - 1) * 2 + rank, stage * 2 + rank])
            self.assertEqual(model.estimate("allgather", 768, stage)["participants"], [stage * 2, stage * 2 + 1])
        for primitive, size, stage, rank in (("p2p", 769, 1, 0), ("allgather", 770, 1, None),
                                              ("p2p", 768, 3, 0), ("p2p", 768, 1, 2)):
            with self.subTest(primitive=primitive, size=size, stage=stage, rank=rank), self.assertRaises(SimulationError):
                model.estimate(primitive, size, stage, tp_rank=rank)

    def test_strict_statistics_sample_lineage_options_and_interval_recomputation(self):
        good = lookup(capture(lambda size, repeat: 0.1 + repeat / 100))
        mutations = [
            lambda p: p["model_options"].update(min_samples_per_size=4),
            lambda p: p["model_options"].update(max_size_ratio=True),
            lambda p: p["model_options"].update(extra=1),
            lambda p: p["groups"][0]["knots"][0].update(mean_ms=0.11),
            lambda p: p["groups"][0]["knots"][0].update(duration_ms=0.11),
            lambda p: p["groups"][0]["knots"][0].update(p10_ms=0.11),
            lambda p: p["groups"][0]["knots"][0].update(sample_count=True),
            lambda p: p["groups"][0]["knots"][0].update(stddev_ms=0),
            lambda p: p["groups"][0]["knots"][0]["samples"].pop(),
            lambda p: p["groups"][0]["knots"][0]["samples"][0].update(group_id="foreign"),
            lambda p: p["groups"][0]["knots"][0]["samples"][0].update(duration_ms=True),
            lambda p: p["groups"][0]["knots"][0]["record_ids"].pop(),
            lambda p: p["groups"][0]["intervals"][0].update(supported=1),
            lambda p: p["groups"][0]["intervals"][0].update(reason="fake"),
            lambda p: p["groups"][0]["intervals"].reverse(),
            lambda p: p["groups"][0]["knots"].reverse(),
            lambda p: p["groups"][0].update(mae_ms=0),
            lambda p: p["groups"][0]["coverage"].update(record_count=15.0),
            lambda p: p["training"]["group_ids"].pop(),
            lambda p: p["groups"][0].update(intercept_ms=0),
        ]
        for mutation in mutations:
            profile = deepcopy(good)
            mutation(profile)
            with self.subTest(mutation=mutation), self.assertRaises(SimulationError):
                validate(profile)
        blocked = lookup(capture(lambda size, repeat: 1.0 if size == 512 else 0.1))
        blocked["groups"][0]["intervals"][0].update(supported=True, reason=None)
        with self.assertRaises(SimulationError):
            validate(blocked)

    def test_group_ids_can_span_sizes_but_records_cannot_be_reused(self):
        data = capture()
        for row in data["records"]:
            row["group_id"] = "one-independent-session"
        profile = lookup(data)
        self.assertEqual(profile["training"]["group_ids"], ["one-independent-session"])
        self.assertEqual(validate(profile), profile)
        bad = deepcopy(profile)
        source = bad["groups"][0]["knots"][0]
        target = bad["groups"][0]["knots"][1]
        target["samples"][0]["record_id"] = source["samples"][0]["record_id"]
        target["record_ids"][0] = source["record_ids"][0]
        with self.assertRaises(SimulationError):
            validate(bad)

    def test_bounded_profile_roundtrip_and_extreme_finite_timings(self):
        for value in (1e-300, 1e308):
            with self.subTest(value=value):
                profile = lookup(capture(lambda size, repeat: value, repeats=6))
                estimate = PPCommunicationModel(legacy.experiment(tp=1), profile).estimate("p2p", 768, 1, tp_rank=0)
                self.assertTrue(math.isfinite(estimate["duration_ms"]))
                self.assertAlmostEqual(estimate["duration_ms"] / value, 1)
        profile = lookup()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lookup.json"
            path.write_text(json.dumps(profile))
            self.assertEqual(load_pp_communication_profile(path, allow_test_fixture=True), profile)
        with patch("mlsys_sim.pp_communication._MAX_BYTES", 1024):
            with self.assertRaises(SimulationError):
                lookup()
            with self.assertRaises(SimulationError):
                validate(profile)


class PiecewiseHoldoutTests(unittest.TestCase):
    def test_unseen_size_and_new_session_means_are_distinct_from_sample_mape(self):
        profile = lookup(capture(lambda size, repeat: 0.3))
        heldout = capture(lambda size, repeat: [0.1, 0.1, 0.1, 0.1, 1.1][repeat],
                          sizes=(512, 768), dataset_id="blind-new-session")
        before = deepcopy(profile)
        report = validate_pp_communication_holdout(profile, heldout, allow_test_fixture=True)
        self.assertEqual(profile, before)
        self.assertFalse(report["hardware_validated"])
        self.assertEqual(report["coverage"]["fraction"], 1)
        self.assertEqual(report["exact_vs_unseen"]["exact"]["coverage"]["evaluated_records"], 5)
        self.assertEqual(report["exact_vs_unseen"]["unseen"]["coverage"]["evaluated_records"], 5)
        expected = statistics.mean([2, 2, 2, 2, 0.8 / 1.1])
        for size in report["by_message_size"]:
            self.assertAlmostEqual(size["mean_target_absolute_error_ms"], 0)
            self.assertAlmostEqual(size["sample_mape"], expected)
        self.assertAlmostEqual(report["mean_target_metrics"]["mae_ms"], 0)
        self.assertAlmostEqual(report["small_messages"]["metrics"]["mape"], expected)

    def test_rejected_sizes_stay_in_coverage_and_have_null_mean_error(self):
        profile = lookup(capture(lambda size, repeat: 1.0 if size == 512 else 0.1,
                                 sizes=(512, 1024, 65536, 131072)))
        data = capture(lambda size, repeat: 0.2, sizes=(512, 768, 2048, 65536, 98304, 262144),
                       dataset_id="blind", repeats=3)
        report = validate_pp_communication_holdout(profile, data, allow_test_fixture=True, allow_extrapolation=True)
        self.assertEqual(report["coverage"], {"total_records": 18, "evaluated_records": 9,
                                             "rejected_records": 9, "fraction": 0.5})
        self.assertEqual(report["exact_vs_unseen"]["unseen"]["coverage"]["rejected_records"], 9)
        self.assertEqual(report["mean_target_metrics"]["evaluated_sizes"], 3)
        self.assertEqual(report["mean_target_metrics"]["rejected_sizes"], 3)
        self.assertEqual(report["small_messages"]["coverage"]["rejected_records"], 6)
        self.assertEqual(report["small_messages"]["coverage"]["total_records"], 12)
        rejected = [row for row in report["by_message_size"] if row["coverage"]["rejected_records"]]
        for row in rejected:
            self.assertIsNone(row["predicted_mean_ms"])
            self.assertIsNone(row["mean_target_absolute_error_ms"])
            self.assertIsNone(row["sample_mape"])

    def test_entire_holdout_rejected_has_no_fabricated_error(self):
        profile = lookup()
        data = capture(sizes=(256, 4096), dataset_id="outside")
        report = validate_pp_communication_holdout(profile, data, allow_test_fixture=True)
        self.assertEqual(report["coverage"]["fraction"], 0)
        self.assertIsNone(report["metrics"]["mae_ms"])
        self.assertIsNone(report["metrics"]["mape"])
        self.assertIsNone(report["mean_target_metrics"]["mape"])
        self.assertEqual(report["mean_target_metrics"]["rejected_sizes"], 2)

    def test_legacy_holdout_does_not_invent_training_size_membership(self):
        report = validate_pp_communication_holdout(legacy.fit(), legacy.dataset(dataset_id="blind"),
                                                  allow_test_fixture=True)
        self.assertEqual(report["exact_vs_unseen"]["classification_basis"], "unavailable_for_legacy_affine")
        self.assertEqual(report["exact_vs_unseen"]["unknown"]["coverage"]["fraction"], 1)
        self.assertEqual(report["exact_vs_unseen"]["exact"]["coverage"]["total_records"], 0)

    def test_holdout_leakage_and_identity_checks_apply_to_lookup(self):
        data = capture()
        profile = lookup(data)
        shared_group = capture(dataset_id="blind")
        shared_group["records"][0]["group_id"] = data["records"][0]["group_id"]
        different_identity = capture(dataset_id="blind")
        different_identity["identity"]["parallelism"]["pipeline_parallel_size"] = 3
        for bad in (deepcopy(data), shared_group, different_identity):
            with self.assertRaises(SimulationError):
                validate_pp_communication_holdout(profile, bad, allow_test_fixture=True)


class ProtocolSemanticIsolationTests(unittest.TestCase):
    def test_v1_and_v2_have_explicit_distinct_initialization_domains(self):
        data = capture()
        v2 = protocol_v2(data)
        self.assertEqual(validate_pp_communication_dataset(v2, allow_test_fixture=True), v2)
        bad = deepcopy(data)
        bad["benchmark"]["communicator_initialization"] = "lazy_world_and_pair_groups_v1"
        for value in (bad, protocol_v2(data)):
            if value["benchmark"]["protocol"] == "native_pp_primitives_v2":
                value["benchmark"]["communicator_initialization"] = "eager"
            with self.assertRaises(SimulationError):
                validate_pp_communication_dataset(value, allow_test_fixture=True)
        bad = deepcopy(v2)
        del bad["benchmark"]["communicator_initialization"]
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(bad, allow_test_fixture=True)
        parent = lookup(data)
        with self.assertRaises(SimulationError) as error:
            lookup(v2, parent_profile=parent)
        self.assertEqual(error.exception.code, "CALIBRATION_MISMATCH")
        heldout = protocol_v2(capture(dataset_id="blind"))
        with self.assertRaises(SimulationError) as error:
            validate_pp_communication_holdout(parent, heldout, allow_test_fixture=True)
        self.assertEqual(error.exception.code, "CALIBRATION_MISMATCH")

    def test_launch_may_differ_while_full_stack_and_environment_stay_exact(self):
        data = protocol_v2(capture())
        parent = lookup(data)
        heldout = protocol_v2(capture(dataset_id="blind"))
        heldout["benchmark"]["launch"] = {"capture_session": "new-independent-session"}
        report = validate_pp_communication_holdout(parent, heldout, allow_test_fixture=True)
        self.assertEqual(report["coverage"]["fraction"], 1)
        child = calibrate_pp_communication(heldout, profile_version="child", parent_profile=parent,
                                           method=METHOD, allow_test_fixture=True)
        self.assertEqual(child["training"]["parent_profile_version"], "lookup-v1")
        self.assertEqual(child["source"]["kind"], "test_fixture")
        heldout["benchmark"]["stack"]["extra_environment_policy"] = "changed"
        for action in (lambda: validate_pp_communication_holdout(parent, heldout, allow_test_fixture=True),
                       lambda: calibrate_pp_communication(heldout, profile_version="child", parent_profile=parent,
                                                          method=METHOD, allow_test_fixture=True)):
            with self.assertRaises(SimulationError) as error:
                action()
            self.assertEqual(error.exception.code, "CALIBRATION_MISMATCH")


if __name__ == "__main__":
    unittest.main()
