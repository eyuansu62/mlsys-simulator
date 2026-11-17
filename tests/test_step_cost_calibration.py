"""Whole-step-only calibration with independent process and policy checks."""

from copy import deepcopy
import unittest

from mlsys_sim.cost_calibration import (cost_data_sha256, fit_analytical_costs, freeze_cost_split,
                                       validate_analytical_costs)
from mlsys_sim.schema import SimulationError
from mlsys_sim.step_cost_calibration import (REJECTION_KIND, STEP_ALGORITHM, fit_step_costs,
                                            freeze_step_fit_spec, validate_step_fit_metadata)


def fixture(slope=2, intercept=4):
    domain = {"identity_sha256": "a" * 64, "execution_mode": "eager",
              "attention_backend": "fixture-backend", "timing_boundary": "fixture_whole_step_v1"}
    rows = []
    for session in ("train-a", "train-b", "heldout"):
        shapes = (1, 3) if session != "heldout" else (1, 2)
        for shape in shapes:
            for sample in range(2):
                rows.append({"record_id": f"{session}-{shape}-{sample}", "kind": "step_wall",
                    "session_id": session, "group_id": f"shape-{shape}", "model_id": "fixture-model",
                    "phase": "decode", "shape": f"B{shape}", "query_tokens": [1] * shape,
                    "context_lengths": [100] * shape, "layers": 2,
                    "nominal_nonattention_ms": shape * 0.75, "nominal_attention_ms": shape * 0.25,
                    "observed_ms": slope * shape + intercept, "default_ms": 1.5 * shape + 0.2,
                    "domain": deepcopy(domain)})
    return {"schema_version": 1, "kind": "analytical_cost_dataset_v1",
            "source": {"kind": "test_fixture", "reference": "Synthetic complete step timings"}, "records": rows}


def split(data, **changes):
    fields = {"training_record_ids": [row["record_id"] for row in data["records"] if row["session_id"].startswith("train")],
        "holdout_record_ids": [row["record_id"] for row in data["records"] if row["session_id"] == "heldout" and row["shape"] == "B2"],
        "repeat_record_ids": [row["record_id"] for row in data["records"] if row["session_id"] == "heldout" and row["shape"] == "B1"]}
    fields.update(changes)
    return freeze_cost_split(data, **fields)


def fitted(data=None, threshold=1.25):
    data = fixture() if data is None else data
    frozen = split(data)
    policy = freeze_step_fit_spec(frozen, max_training_process_ratio=threshold)
    return fit_step_costs(data, frozen, calibration_id="step-fixture", fit_spec=policy)


def rehash(value):
    value["fit_sha256"] = cost_data_sha256({key: item for key, item in value.items() if key != "fit_sha256"})


class StepCostCalibrationTests(unittest.TestCase):
    def test_complete_steps_need_no_component_labels(self):
        data = fixture()
        before = deepcopy(data)
        result = fitted(data)
        self.assertEqual(result["algorithm"], STEP_ALGORITHM)
        cost = result["cost_specs"][0]
        self.assertAlmostEqual(cost["nonattention_scale"], 2)
        self.assertEqual(cost["nonattention_scale"], cost["attention_scale"])
        self.assertEqual(cost["attention_fixed_us_per_layer"], 0)
        self.assertAlmostEqual(cost["runtime_fixed_us"], 4000)
        self.assertTrue(all(row["kind"] == "step_wall" for row in data["records"]))
        validate_step_fit_metadata(data, result)
        validation = validate_analytical_costs(data, result)
        self.assertEqual(validation["verdict"], "PASS")
        self.assertEqual(validation["holdout"]["calibrated"]["cell_mape"], 0)
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(data, before)

    def test_validation_targets_cannot_change_fit_or_stability(self):
        left = fixture()
        right = deepcopy(left)
        for row in right["records"]:
            if row["session_id"] == "heldout":
                row["observed_ms"] *= 50
        first, second = fitted(left), fitted(right)
        self.assertEqual(first["cost_specs"], second["cost_specs"])
        self.assertEqual(first["diagnostics"], second["diagnostics"])
        self.assertNotEqual(first["fit_sha256"], second["fit_sha256"])
        self.assertEqual(validate_analytical_costs(right, second)["verdict"], "FAIL")
        with self.assertRaises(SimulationError):
            validate_analytical_costs(right, first)

    def test_each_session_shape_median_gets_equal_weight(self):
        data = fixture()
        data["records"] = [row for row in data["records"] if not (
            row["session_id"] == "train-b" and row["shape"] == "B3")]
        template = deepcopy(next(row for row in data["records"] if row["session_id"] == "train-a" and row["shape"] == "B3"))
        for sample in range(100):
            row = deepcopy(template)
            row.update(record_id=f"many-{sample}", session_id="train-b", observed_ms=12)
            data["records"].append(row)
        result = fitted(data)
        self.assertAlmostEqual(result["cost_specs"][0]["nonattention_scale"], 2.5)
        self.assertAlmostEqual(result["cost_specs"][0]["runtime_fixed_us"], 3500)
        self.assertEqual(result["diagnostics"][0]["whole_step_affine"]["equal_weight_cell_count"], 4)

    def test_negative_intercept_uses_nonnegative_boundary(self):
        result = fitted(fixture(slope=3, intercept=-2))
        cost = result["cost_specs"][0]
        self.assertAlmostEqual(cost["nonattention_scale"], 2.2)
        self.assertEqual(cost["runtime_fixed_us"], 0)
        self.assertLess(result["diagnostics"][0]["whole_step_affine"]["numeric_diagnostics"]["unconstrained_intercept_ms"], 0)

    def test_negative_slope_uses_constant_nonnegative_model(self):
        result = fitted(fixture(slope=-1, intercept=10))
        self.assertEqual(result["cost_specs"][0]["nonattention_scale"], 0)
        self.assertAlmostEqual(result["cost_specs"][0]["runtime_fixed_us"], 8000)

    def test_unstable_training_rejected_with_all_process_diagnostics(self):
        data = fixture()
        for row in data["records"]:
            if row["session_id"] == "train-b":
                row["observed_ms"] *= 2
        result = fitted(data)
        self.assertEqual(result["kind"], REJECTION_KIND)
        self.assertEqual(result["status"], "REJECTED")
        self.assertFalse(result["fit_attempted"])
        self.assertNotIn("cost_specs", result)
        self.assertNotIn("calibration_patch", result)
        self.assertIn("UNSTABLE_TRAINING_PROCESSES", {row["code"] for row in result["rejections"]})
        cells = result["diagnostics"][0]["training_cells"]
        self.assertEqual(len(cells), 4)
        self.assertEqual({cell["session_id"] for cell in cells}, {"train-a", "train-b"})
        self.assertEqual({row["max_median_over_min"] for row in result["diagnostics"][0]["stability"]}, {2})

    def test_stability_threshold_is_inclusive_and_bound_before_fit(self):
        data = fixture()
        for row in data["records"]:
            if row["session_id"] == "train-b":
                row["observed_ms"] *= 1.25
        self.assertEqual(fitted(data)["kind"], "analytical_cost_fit_v1")
        frozen = split(data)
        policy = freeze_step_fit_spec(frozen)
        policy["stability"]["max_training_process_ratio"] = 2
        with self.assertRaisesRegex(SimulationError, "policy"):
            fit_step_costs(data, frozen, calibration_id="changed", fit_spec=policy)
        result = fitted(data)
        changed = result["step_fit_spec"]
        changed["stability"]["max_training_process_ratio"] = 1.1
        changed["fit_spec_sha256"] = cost_data_sha256({key: value for key, value in changed.items() if key != "fit_spec_sha256"})
        result["fit_spec_sha256"] = changed["fit_spec_sha256"]
        rehash(result)
        with self.assertRaisesRegex(SimulationError, "stability"):
            validate_step_fit_metadata(data, result)

    def test_one_process_or_one_shape_does_not_make_identifiable_stable_fit(self):
        for selected in (
            lambda row: row["session_id"] == "train-a",
            lambda row: row["session_id"].startswith("train") and row["shape"] == "B1"):
            data = fixture()
            frozen = split(data, training_record_ids=[row["record_id"] for row in data["records"] if selected(row)])
            result = fit_step_costs(data, frozen, calibration_id="insufficient", fit_spec=freeze_step_fit_spec(frozen))
            self.assertEqual(result["kind"], REJECTION_KIND)
            self.assertNotIn("cost_specs", result)
        data = fixture()
        for row in data["records"]:
            if row["session_id"].startswith("train"):
                row.update(nominal_nonattention_ms=0.75, nominal_attention_ms=0.25)
        result = fitted(data)
        self.assertIn("NON_IDENTIFIABLE_NOMINAL_FEATURES", {row["code"] for row in result["rejections"]})

    def test_each_shape_needs_process_repeats_for_stability_gate(self):
        data = fixture()
        selected = [row["record_id"] for row in data["records"] if row["session_id"] == "train-a"
                    or (row["session_id"] == "train-b" and row["shape"] == "B1")]
        frozen = split(data, training_record_ids=selected)
        result = fit_step_costs(data, frozen, calibration_id="missing-shape", fit_spec=freeze_step_fit_spec(frozen))
        self.assertIn("INSUFFICIENT_SHAPE_REPEATS", {row["code"] for row in result["rejections"]})

    def test_old_freeze_leakage_guards_and_source_hashes_are_reused(self):
        data = fixture()
        with self.assertRaises(SimulationError):
            split(data, holdout_record_ids=["train-a-1-0"])
        frozen = split(data)
        policy = freeze_step_fit_spec(frozen)
        changed = deepcopy(data)
        changed["records"][0]["observed_ms"] += 1
        with self.assertRaises(SimulationError):
            fit_step_costs(changed, frozen, calibration_id="changed", fit_spec=policy)
        policy["split_sha256"] = "b" * 64
        with self.assertRaises(SimulationError):
            fit_step_costs(data, frozen, calibration_id="changed", fit_spec=policy)

    def test_component_input_and_invalid_policy_are_not_silently_ignored(self):
        data = fixture()
        data["records"][0].update(kind="gpu_components", observed_nonattention_ms=1, observed_attention_ms=1)
        frozen = split(data)
        with self.assertRaisesRegex(SimulationError, "step_wall"):
            fit_step_costs(data, frozen, calibration_id="components", fit_spec=freeze_step_fit_spec(frozen))
        for value in (True, 0, 0.99, float("inf")):
            with self.subTest(value=value), self.assertRaises(SimulationError):
                freeze_step_fit_spec(frozen, max_training_process_ratio=value)

    def test_whole_step_coupling_cannot_be_relabelled_component_calibration(self):
        data = fixture()
        result = fitted(data)
        result["cost_specs"][0]["attention_scale"] += 1
        rehash(result)
        with self.assertRaisesRegex(SimulationError, "shared scale"):
            validate_step_fit_metadata(data, result)
        result = fitted(data)
        result["diagnostics"][0]["training_cells"][0]["observed_ms"] += 1
        rehash(result)
        with self.assertRaisesRegex(SimulationError, "median diagnostics"):
            validate_step_fit_metadata(data, result)

    def test_default_component_family_is_unchanged_and_still_requires_components(self):
        from test_cost_calibration import fixture as component_fixture, split as component_split

        data = component_fixture()
        original = fit_analytical_costs(data, component_split(data), calibration_id="old-family")
        before = deepcopy(original)
        fitted()
        self.assertEqual(original, before)
        self.assertNotIn("step_fit_spec", original)
        self.assertNotEqual(original["algorithm"], STEP_ALGORITHM)
        self.assertAlmostEqual(original["cost_specs"][0]["attention_scale"], 3)
        self.assertEqual(validate_analytical_costs(data, original)["verdict"], "PASS")
        data = fixture()
        with self.assertRaisesRegex(SimulationError, "component shapes"):
            fit_analytical_costs(data, split(data), calibration_id="not-new-family")


if __name__ == "__main__":
    unittest.main()
