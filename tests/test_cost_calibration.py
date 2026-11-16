"""Independent known-coefficient, lineage and held-out-failure cost checks."""

from copy import deepcopy
import unittest

from mlsys_sim.cost_calibration import (cost_data_sha256, fit_analytical_costs,
    freeze_cost_split, validate_analytical_costs, validate_cost_dataset)
from mlsys_sim.schema import SimulationError


def fixture():
    domain = {"identity_sha256": "a" * 64, "execution_mode": "eager",
              "attention_backend": "fixture-attention", "timing_boundary": "fixture_step_wall_v1"}
    records = []
    def row(name, q, kind, session):
        value = {"record_id": name, "kind": kind, "session_id": session,
            "group_id": f"shape-{q}", "model_id": "fixture-model", "phase": "prefill",
            "shape": f"q{q}", "query_tokens": [q], "context_lengths": [0], "layers": 2,
            "nominal_nonattention_ms": 2.0 * q, "nominal_attention_ms": float(q),
            "domain": deepcopy(domain)}
        if kind == "gpu_components":
            value.update(observed_nonattention_ms=4.0 * q, observed_attention_ms=3.0 * q + 1)
        else:
            value.update(observed_ms=4.0 * q + 3.0 * q + 1 + 4,
                         default_ms=2.0 * (2.0 * q + q) + 0.3)
        records.append(value)
    for q in (1, 3):
        row(f"gpu{q}", q, "gpu_components", "profile")
        row(f"wall{q}", q, "step_wall", "fit")
        row(f"repeat{q}", q, "step_wall", "holdout")
    row("heldout2", 2, "step_wall", "holdout")
    row("unused2", 2, "step_wall", "fit")
    return {"schema_version": 1, "kind": "analytical_cost_dataset_v1",
            "source": {"kind": "test_fixture", "reference": "Known coefficient CPU fixture"},
            "records": records}


def split(data, **changes):
    kwargs = {"training_record_ids": ["gpu1", "wall1", "gpu3", "wall3"],
              "holdout_record_ids": ["heldout2"], "repeat_record_ids": ["repeat1", "repeat3"]}
    kwargs.update(changes)
    return freeze_cost_split(data, **kwargs)


def artifact(data=None):
    data = fixture() if data is None else data
    return fit_analytical_costs(data, split(data), calibration_id="known-coefficients")


def record(data, name):
    return next(row for row in data["records"] if row["record_id"] == name)


def rehash(fitted):
    fitted["fit_sha256"] = cost_data_sha256({key: value for key, value in fitted.items() if key != "fit_sha256"})


class CostCalibrationTests(unittest.TestCase):
    def test_known_independent_component_and_wall_coefficients_recovered(self):
        data = fixture()
        before = deepcopy(data)
        fitted = artifact(data)
        cost = fitted["cost_specs"][0]
        self.assertAlmostEqual(cost["nonattention_scale"], 2)
        self.assertAlmostEqual(cost["attention_scale"], 3)
        self.assertAlmostEqual(cost["attention_fixed_us_per_layer"], 500)
        self.assertAlmostEqual(cost["runtime_fixed_us"], 4000)
        self.assertEqual(cost["source_kind"], "test_fixture")
        result = validate_analytical_costs(data, fitted)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["holdout"]["calibrated"]["cell_mape"], 0)
        self.assertEqual(result["repeats"]["calibrated"]["cell_mape"], 0)
        self.assertEqual(result["unused_record_ids"], ["unused2"])
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(data, before)

    def test_heldout_targets_never_change_fitted_parameters(self):
        first = fixture()
        second = deepcopy(first)
        record(second, "heldout2")["observed_ms"] *= 20
        left, right = artifact(first), artifact(second)
        self.assertEqual(left["cost_specs"], right["cost_specs"])
        self.assertNotEqual(left["fit_sha256"], right["fit_sha256"])
        failed = validate_analytical_costs(second, right)
        self.assertEqual(failed["verdict"], "FAIL")
        self.assertEqual(failed["holdout"]["calibrated"]["cells_passed"], 0)
        self.assertEqual(len(failed["holdout"]["calibrated"]["records"]), 1)
        with self.assertRaises(SimulationError):
            validate_analytical_costs(second, left)

    def test_rank_deficient_attention_fit_fails_explicitly(self):
        data = fixture()
        record(data, "gpu3")["nominal_attention_ms"] = 1
        with self.assertRaisesRegex(SimulationError, "not identifiable"):
            artifact(data)

    def test_negative_attention_intercept_uses_nonnegative_boundary_fit(self):
        data = fixture()
        record(data, "gpu1").update(nominal_attention_ms=2, observed_attention_ms=1)
        record(data, "gpu3").update(nominal_attention_ms=4, observed_attention_ms=7)
        fitted = artifact(data)
        self.assertAlmostEqual(fitted["cost_specs"][0]["attention_scale"], 1.5)
        self.assertEqual(fitted["cost_specs"][0]["attention_fixed_us_per_layer"], 0)
        self.assertLess(fitted["diagnostics"][0]["attention"]["unconstrained_intercept_ms"], 0)
        self.assertTrue(fitted["diagnostics"][0]["attention"]["constraint_active"])

    def test_negative_wall_residual_is_retained_not_fitted_as_negative_cpu(self):
        data = fixture()
        for name in ("wall1", "wall3"):
            record(data, name)["observed_ms"] = 1
        fitted = artifact(data)
        self.assertEqual(fitted["cost_specs"][0]["runtime_fixed_us"], 0)
        self.assertTrue(fitted["diagnostics"][0]["runtime_intercept_clamped"])
        self.assertLess(fitted["diagnostics"][0]["unclamped_runtime_intercept_ms"], 0)

    def test_membership_and_raw_source_are_frozen_before_fit(self):
        data = fixture()
        frozen = split(data)
        data["records"][0]["observed_attention_ms"] += 1
        with self.assertRaisesRegex(SimulationError, "modified"):
            fit_analytical_costs(data, frozen, calibration_id="changed")
        data = fixture()
        frozen = split(data)
        frozen["training_record_ids"].append("unused2")
        with self.assertRaises(SimulationError):
            fit_analytical_costs(data, frozen, calibration_id="changed")

    def test_acquisition_and_group_leakage_rejected(self):
        for field, value in (("session_id", "fit"), ("group_id", "shape-1"), ("shape", "q1")):
            data = fixture()
            record(data, "heldout2")[field] = value
            with self.subTest(field=field), self.assertRaises(SimulationError):
                split(data)
        with self.assertRaises(SimulationError):
            split(fixture(), holdout_record_ids=["wall1"])

    def test_repeat_group_overlap_is_allowed_only_with_new_session(self):
        data = fixture()
        self.assertEqual(validate_analytical_costs(data, artifact(data))["repeats"]["verdict"], "PASS")
        record(data, "repeat1")["session_id"] = "fit"
        with self.assertRaises(SimulationError):
            split(data)

    def test_backend_and_identity_domains_are_not_transferred(self):
        for field in ("execution_mode", "attention_backend", "timing_boundary", "identity_sha256"):
            data = fixture()
            record(data, "heldout2")["domain"][field] = "b" * 64 if field == "identity_sha256" else "different"
            with self.subTest(field=field), self.assertRaises(SimulationError):
                split(data)

    def test_out_of_domain_shape_is_retained_as_coverage_failure(self):
        data = fixture()
        record(data, "heldout2")["query_tokens"] = [4]
        result = validate_analytical_costs(data, artifact(data))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["holdout"]["coverage"], 0)
        self.assertEqual(result["holdout"]["declared_record_count"], 1)
        self.assertEqual(result["holdout"]["rejected"][0]["record_id"], "heldout2")

    def test_threshold_cannot_be_relaxed_after_fitting(self):
        data = fixture()
        frozen = split(data, max_relative_error=0.25)
        fitted = fit_analytical_costs(data, frozen, calibration_id="threshold")
        self.assertEqual(validate_analytical_costs(data, fitted)["max_relative_error"], 0.25)
        with self.assertRaisesRegex(SimulationError, "threshold differs"):
            validate_analytical_costs(data, fitted, max_relative_error=0.5)

    def test_cost_artifact_and_training_lineage_tampering_rejected(self):
        data = fixture()
        fitted = artifact(data)
        fitted["cost_specs"][0]["nonattention_scale"] += 1
        with self.assertRaises(SimulationError):
            validate_analytical_costs(data, fitted)
        fitted = artifact(data)
        cost = fitted["cost_specs"][0]
        cost["training_record_ids"].append("heldout2")
        cost["training_dataset_sha256"] = cost_data_sha256([record(data, name) for name in cost["training_record_ids"]])
        fitted["fit_sha256"] = cost_data_sha256({k: v for k, v in fitted.items() if k != "fit_sha256"})
        with self.assertRaisesRegex(SimulationError, "lineage"):
            validate_analytical_costs(data, fitted)

    def test_rehashed_cost_domain_must_match_diagnostic_and_training(self):
        data = fixture()
        for field, changed in (("phase", "decode"), ("identity_sha256", "b" * 64),
                               ("execution_mode", "graph"), ("attention_backend", "other-backend")):
            fitted = artifact(data)
            fitted["cost_specs"][0][field] = changed
            rehash(fitted)
            with self.subTest(field=field), self.assertRaisesRegex(SimulationError, "training domain"):
                validate_analytical_costs(data, fitted)
        for field, changed in (("phase", "decode"), ("model_id", "different-model")):
            fitted = artifact(data)
            fitted["diagnostics"][0][field] = changed
            rehash(fitted)
            with self.subTest(diagnostic=field), self.assertRaisesRegex(SimulationError, "training domain"):
                validate_analytical_costs(data, fitted)
        fitted = artifact(data)
        fitted["diagnostics"][0]["domain"]["timing_boundary"] = "different-boundary"
        rehash(fitted)
        with self.assertRaisesRegex(SimulationError, "training domain"):
            validate_analytical_costs(data, fitted)

    def test_rehashed_support_bounds_must_come_from_training(self):
        data = fixture()
        changes = {"min_batch_size": 0, "max_batch_size": 2, "min_query_tokens": 2,
                   "max_query_tokens": 30, "min_context_tokens": -1, "max_context_tokens": 4}
        for field, changed in changes.items():
            fitted = artifact(data)
            fitted["cost_specs"][0][field] = changed
            rehash(fitted)
            with self.subTest(field=field), self.assertRaisesRegex(SimulationError, "training support"):
                validate_analytical_costs(data, fitted)
        fitted = artifact(data)
        fitted["cost_specs"][0]["min_batch_size"] = True
        rehash(fitted)
        with self.assertRaisesRegex(SimulationError, "training support"):
            validate_analytical_costs(data, fitted)

    def test_layer_count_cannot_change_within_a_fitted_identity(self):
        data = fixture()
        record(data, "heldout2")["layers"] = 3
        fitted = artifact(data)
        with self.assertRaisesRegex(SimulationError, "layer count"):
            validate_analytical_costs(data, fitted)

    def test_all_training_domains_must_have_exactly_one_cost(self):
        data = fixture()
        extra = []
        for source in data["records"]:
            if source["record_id"] in {"gpu1", "wall1", "gpu3", "wall3"}:
                row = deepcopy(source)
                row["record_id"] = "other-" + row["record_id"]
                row["domain"]["identity_sha256"] = "b" * 64
                extra.append(row)
        data["records"].extend(extra)
        frozen = split(data, training_record_ids=["gpu1", "wall1", "gpu3", "wall3"]
                       + [row["record_id"] for row in extra])
        fitted = fit_analytical_costs(data, frozen, calibration_id="two-domains")
        self.assertEqual(len(fitted["cost_specs"]), 2)
        fitted["cost_specs"].pop()
        fitted["diagnostics"].pop()
        rehash(fitted)
        with self.assertRaisesRegex(SimulationError, "missing fitted cost domains"):
            validate_analytical_costs(data, fitted)
        fitted = artifact()
        fitted["cost_specs"].append(deepcopy(fitted["cost_specs"][0]))
        fitted["diagnostics"].append(deepcopy(fitted["diagnostics"][0]))
        rehash(fitted)
        with self.assertRaisesRegex(SimulationError, "Duplicate"):
            validate_analytical_costs(fixture(), fitted)

    def test_calibration_patch_family_and_identity_cannot_disagree(self):
        data = fixture()
        for container, field, changed in (("calibration_patch", "backend", "synthetic"),
            ("calibration_patch", "analytical_cost_timing_boundary", "different-boundary"),
            (None, "algorithm", "different-algorithm"), (None, "calibration_id", "different-id")):
            fitted = artifact(data)
            target = fitted if container is None else fitted[container]
            target[field] = changed
            rehash(fitted)
            with self.subTest(field=field), self.assertRaises(SimulationError):
                validate_analytical_costs(data, fitted)
        fitted = artifact(data)
        fitted["cost_specs"][0]["not_a_cost_field"] = 42
        rehash(fitted)
        with self.assertRaisesRegex(SimulationError, "fields"):
            validate_analytical_costs(data, fitted)

    def test_optional_repeat_partition_is_not_a_successful_validation(self):
        data = fixture()
        frozen = split(data, repeat_record_ids=[])
        fitted = fit_analytical_costs(data, frozen, calibration_id="no-repeats")
        report = validate_analytical_costs(data, fitted)
        self.assertEqual(report["repeats"]["verdict"], "NOT_REQUESTED")
        self.assertEqual(report["repeats"]["declared_record_count"], 0)
        self.assertEqual(report["verdict"], "PASS")

    def test_invalid_zero_nonfinite_duplicate_and_unspecified_observations(self):
        for key, value in (("nominal_nonattention_ms", 0), ("observed_ms", float("nan")),
                           ("observed_ms", 0), ("layers", True)):
            data = fixture()
            record(data, "wall1")[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(SimulationError):
                validate_cost_dataset(data)
        data = fixture()
        data["records"].append(deepcopy(data["records"][0]))
        with self.assertRaises(SimulationError):
            validate_cost_dataset(data)
        data = fixture()
        data["records"][0]["domain"]["execution_mode"] = "unspecified"
        with self.assertRaises(SimulationError):
            validate_cost_dataset(data)

    def test_steps_are_nested_under_session_shape_median(self):
        data = fixture()
        duplicate = deepcopy(record(data, "heldout2"))
        duplicate["record_id"] = "heldout2-extra"
        duplicate["observed_ms"] *= 2
        data["records"].append(duplicate)
        frozen = split(data, holdout_record_ids=["heldout2", "heldout2-extra"])
        fitted = fit_analytical_costs(data, frozen, calibration_id="nested")
        report = validate_analytical_costs(data, fitted)
        self.assertEqual(report["holdout"]["calibrated"]["record_count"], 2)
        self.assertEqual(report["holdout"]["calibrated"]["cell_count"], 1)
        self.assertEqual(report["holdout"]["calibrated"]["cells"][0]["sample_count"], 2)
        self.assertIn("nested", report["acceptance"]["independent_unit"])


if __name__ == "__main__":
    unittest.main()
