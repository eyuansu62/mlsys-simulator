"""Course raw-data integrity, two-sided calibration and opt-in application."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from mlsys_sim.course_calibration import (CALIBRATION_PROTOCOL, apply_course_calibration,
    import_course_calibration, validate_course_calibration_report, write_course_calibration_report)
from mlsys_sim.course_roofline import (CALIBRATED_PROTOCOL, PROTOCOL, run_calibrated_chapter1,
    run_chapter1, write_chapter1_report)
from mlsys_sim.schema import CalibrationSpec, Experiment, SimulationError
from mlsys_sim.config import load_experiment
from mlsys_sim.engine import simulate


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/course/course-calibration.fixture.json"
PLAN = ROOT / "examples/course/chapter1.plan.json"
SOURCE = {"kind": "test_fixture", "label": "hand-calculated fixture; no GPU measurements"}


def data():
    return json.loads(FIXTURE.read_bytes())


def encoded(value):
    return json.dumps(value, allow_nan=True).encode()


def imported(value=None):
    return import_course_calibration(FIXTURE.read_bytes() if value is None else encoded(value), source=SOURCE)


def plan():
    return json.loads(PLAN.read_bytes())


def rehash(report):
    report["data_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in report.items() if k != "data_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return report


def change_measurement(row, multiplier):
    """Independent fixture mutation that preserves every source-derived field."""
    measured = round(row["median_ms"] * multiplier, 5)
    row["median_ms"] = measured
    row["per_round_ms"] = [round(measured * .98, 5), measured, round(measured * 1.02, 5)]
    tc, tm = row["flops"] / 195e9, row["bytes"] / 640e6
    row["ratio_meas_over_nom"] = round(measured / max(tc, tm), 3)
    if "eta_point" in row:
        row["eta_point"] = round(tc / measured, 4)
    elif "beta_point" in row:
        row["beta_point"] = round(tm / measured, 4)
    else:
        row["achieved_tflops"] = round(row["flops"] / measured / 1e9, 1)
        row["achieved_bw_gbps"] = round(row["bytes"] / measured / 1e6, 1)
    if "t_pred_cal_ms" in row:
        row["pred_err_pct"] = round((measured / row["t_pred_cal_ms"] - 1) * 100, 1)


class CourseCalibrationImportTests(unittest.TestCase):
    def test_fixture_import_has_explicit_source_and_reproducible_fit(self):
        result = imported()
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["protocol"], CALIBRATION_PROTOCOL)
        self.assertEqual(result["coefficients"], {"eta": .5, "beta": .8})
        self.assertEqual(result["source_kind"], "test_fixture")
        self.assertFalse(result["hardware_validated"])
        self.assertFalse(result["historical_prediction_before_measurement_established"])
        self.assertEqual(len(result["calibration_records"]), 8)
        self.assertEqual(len(result["holdout_records"]), 5)
        self.assertEqual(result["raw_sha256"], hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
        self.assertEqual(result["fit"]["source"]["fit_points"],
            {"eta": ["4096^3", "5120^3"], "beta": ["ln_8192x4096", "softmax_8192x4096"]})
        source = data()
        ef = [r for r in source["calibration"]["gemm"].values() if r["fit"]]
        actual = sum(r["t_comp_ms"] for r in ef) / sum(r["median_ms"] for r in ef)
        self.assertEqual(result["fit"]["recomputed_from_source_rounded_times"]["eta"], actual)
        self.assertAlmostEqual(result["fit"]["recomputed_from_source_rounded_times"]["beta"], .8, places=4)
        validate_course_calibration_report(result)

    def test_both_sides_are_recomputed_when_calibration_changes_the_bound(self):
        result = imported()
        attention = next(r for r in result["holdout_records"] if r["id"] == "attn_mha16_s512")
        self.assertEqual(attention["source_record"]["binding"], "memory")
        self.assertEqual(attention["recomputed_bound"], "compute")
        self.assertAlmostEqual(attention["recomputed_prediction_ms"], 2_147_483_648 / 195e9 / .5)
        self.assertGreater(attention["recomputed_prediction_ms"], attention["source_prediction_ms"])
        self.assertEqual(attention["source_prediction_ms"], data()["holdout_prediction"]["predictions_ms"]["attn_mha16_s512"]["t_pred_cal_ms"])
        self.assertEqual(attention["recomputed_prediction_ms"],
                         max(attention["recomputed_t_compute_ms"], attention["recomputed_t_memory_ms"]))
        expected = (attention["source_record"]["median_ms"] / attention["recomputed_prediction_ms"] - 1) * 100
        self.assertEqual(attention["recomputed_error_pct"], expected)
        self.assertIn("measured_ms / prediction_ms", attention["error_definition"])

    def test_nonfit_beta_above_one_is_preserved_without_clamping_or_fitting(self):
        result = imported()
        point = next(r for r in result["calibration_records"] if r["id"] == "softmax_4096x4096")
        self.assertFalse(point["source_record"]["fit"])
        self.assertGreater(point["source_record"]["beta_point"], 1)
        self.assertEqual(point["source_record"]["beta_point"], data()["calibration"]["softmax"]["4096x4096"]["beta_point"])
        self.assertEqual(result["coefficients"]["beta"], .8)

    def test_holdout_measurements_do_not_refit_the_coefficients(self):
        changed = data()
        for key, row in changed["holdout_measured"].items():
            change_measurement(row, 1.5)
            changed["holdout_err_pct"][key] = row["pred_err_pct"]
        result = imported(changed)
        self.assertEqual(result["coefficients"], imported()["coefficients"])
        self.assertNotEqual(result["holdout_records"][0]["recomputed_error_pct"],
                            imported()["holdout_records"][0]["recomputed_error_pct"])

    def test_derived_coefficients_are_checked_not_blindly_trusted(self):
        for name in ("eta", "beta"):
            changed = data()
            changed["fit"][name] += .01
            with self.subTest(name=name), self.assertRaises(SimulationError):
                imported(changed)
        changed = data()
        changed["calibration"]["gemm"]["2048^3"]["fit"] = True
        with self.assertRaises(SimulationError):
            imported(changed)
        changed = data()
        changed["holdout_measured"]["gemm_6144"]["fit"] = True
        with self.assertRaises(SimulationError):
            imported(changed)

    def test_strict_source_kind_and_expected_raw_checksum(self):
        raw = FIXTURE.read_bytes()
        with self.assertRaises(SimulationError):
            import_course_calibration(raw, source={"kind": "course_external_report", "label": "cannot upgrade fixture"})
        result = import_course_calibration(raw, source={**SOURCE, "expected_sha256": hashlib.sha256(raw).hexdigest()})
        self.assertEqual(result["source_kind"], "test_fixture")
        for source in ({**SOURCE, "expected_sha256": "0" * 64}, {**SOURCE, "unexpected": True},
                       {"kind": "test_fixture"}, {**SOURCE, "kind": "measured"}):
            with self.subTest(source=source), self.assertRaises(SimulationError):
                import_course_calibration(raw, source=source)
        # Absence of a fixture marker allows only an external declaration, still never hardware PASS.
        external = data()
        external.pop("test_fixture")
        result = import_course_calibration(encoded(external), source={"kind": "course_external_report", "label": "unverified external declaration"})
        self.assertFalse(result["hardware_validated"])

    def test_rejects_duplicate_json_nonfinite_bool_and_unsupported_fields(self):
        raw = FIXTURE.read_bytes().replace(b'"test_fixture": true,', b'"test_fixture": false, "test_fixture": true,', 1)
        with self.assertRaises(SimulationError):
            import_course_calibration(raw, source=SOURCE)
        for field, value in (("median_ms", True), ("median_ms", float("nan")), ("median_ms", 0),
                             ("flops", 1), ("binding", "memory"), ("fit", 1), ("per_round_ms", [1, 2])):
            changed = data()
            changed["calibration"]["gemm"]["4096^3"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SimulationError):
                imported(changed)
        changed = data()
        changed["unknown"] = 1
        with self.assertRaises(SimulationError):
            imported(changed)
        for raw in (b'', b' ' * (1024 * 1024 + 1), b'{bad json', b'\xff', b'[]'):
            with self.subTest(raw_length=len(raw)), self.assertRaises(SimulationError):
                import_course_calibration(raw, source=SOURCE)

    def test_rejects_wrong_signed_error_or_huge_integer_without_overflow(self):
        for value in (True, float("inf"), 10**400, 12345):
            changed = data()
            changed["holdout_measured"]["gemm_6144"]["pred_err_pct"] = value
            with self.subTest(error=value), self.assertRaises(SimulationError):
                imported(changed)
        changed = data()
        changed["holdout_err_pct"]["gemm_6144"] = 10**400
        with self.assertRaises(SimulationError):
            imported(changed)

    def test_current_import_time_and_historical_source_times_are_separate(self):
        result = imported()
        self.assertEqual(result["fit"]["source"]["stored_at"], "2026-01-01T00:00:00Z")
        self.assertNotEqual(result["stored_at"], result["fit"]["source"]["stored_at"])
        with self.assertRaises(SimulationError):
            import_course_calibration(FIXTURE.read_bytes(), source=SOURCE, stored_at="2026-01-01T00:00:00Z")
        archived = rehash({**result, "stored_at": "2026-01-01T00:00:00Z"})
        validate_course_calibration_report(archived)
        for invalid in (None, True, 42, "not UTC", "2026-01-01T00:00:00"):
            with self.subTest(stored_at=invalid), self.assertRaises(SimulationError):
                validate_course_calibration_report(rehash({**result, "stored_at": invalid}))

    def test_tampered_report_fails_even_if_its_untrusted_hash_is_recomputed(self):
        result = imported()
        result["coefficients"]["eta"] = .9
        with self.assertRaises(SimulationError):
            validate_course_calibration_report(result)
        with self.assertRaises(SimulationError):
            validate_course_calibration_report(rehash(result))


class CourseCalibrationApplicationTests(unittest.TestCase):
    def test_patch_is_opt_in_and_never_contains_launch_backend_profile_or_pp_fields(self):
        result = imported()
        patch = result["calibration_patch"]
        self.assertFalse(patch["apply_automatically"])
        self.assertEqual(patch["requires"], {"backend": "analytical"})
        self.assertEqual(set(patch["patch"]), {"compute_efficiency", "bandwidth_efficiency", "profile_version"})
        original = Experiment(calibration=CalibrationSpec(launch_overhead_us=7,
                                                         framework_overhead_us=13))
        applied = apply_course_calibration(original, result, allow_test_fixture=True)
        self.assertEqual((applied.calibration.compute_efficiency, applied.calibration.bandwidth_efficiency), (.5, .8))
        self.assertEqual(applied.calibration.launch_overhead_us, 7)
        self.assertEqual(applied.calibration.framework_overhead_us, 13)
        self.assertEqual(original.calibration.bandwidth_efficiency, .5)
        self.assertEqual(applied.strategy, original.strategy)
        self.assertEqual(applied.infra, original.infra)

    def test_application_rejects_nonanalytical_backends_and_requires_fixture_opt_in(self):
        result = imported()
        for backend in ("synthetic", "profile"):
            original = Experiment(calibration=CalibrationSpec(backend=backend))
            with self.subTest(backend=backend), self.assertRaises(SimulationError):
                apply_course_calibration(original, result, allow_test_fixture=True)
        with self.assertRaises(SimulationError):
            apply_course_calibration(Experiment(), result)

    def test_only_three_fields_change_with_native_pp_workspace_and_all_strategy_features(self):
        base = load_experiment(ROOT / "examples/v1_6/qwen3-r2-tp2-pp2.json")
        for replicas in (1, 2):
            strategy = replace(base.strategy, replicas=replicas, pipeline_transport="vllm_native", pipeline_p2p=True)
            calibration = CalibrationSpec(compute_efficiency=.25, bandwidth_efficiency=.4,
                profile_version="before-import", launch_overhead_us=7, framework_overhead_us=13,
                native_pp_communication_profile_path="native.json" if replicas == 1 else None,
                replica_native_pp_communication_profile_paths=("native-r0.json", "native-r1.json") if replicas == 2 else (),
                native_pp_communication_profile_version="native-v1", native_pp_communication_allow_extrapolation=True,
                workspace_profile_path="workspace.json" if replicas == 1 else None,
                replica_workspace_profile_paths=("workspace-r0.json", "workspace-r1.json") if replicas == 2 else (),
                workspace_profile_version="workspace-v1", allow_test_fixture=True)
            original = replace(base, strategy=strategy, calibration=calibration,
                               infra=replace(base.infra, devices=base.infra.devices[:replicas * 4]))
            applied = apply_course_calibration(original, imported(), allow_test_fixture=True)
            before, after = asdict(original.calibration), asdict(applied.calibration)
            changed = {key for key in before if before[key] != after[key]}
            self.assertEqual(changed, {"compute_efficiency", "bandwidth_efficiency", "profile_version"})
            self.assertEqual(applied.strategy, original.strategy)
            self.assertEqual(applied.infra, original.infra)
            self.assertEqual(applied.workload, original.workload)
            self.assertEqual(applied.execution, original.execution)

    def test_multi_replica_tp_pp_cache_swap_mixed_analytical_integration_remains_valid(self):
        base = load_experiment(ROOT / "examples/v1_6/qwen3-r2-tp2-pp2.json")
        original = replace(base, calibration=CalibrationSpec(backend="analytical"))
        applied = apply_course_calibration(original, imported(), allow_test_fixture=True)
        result = simulate(applied)
        self.assertEqual(result.result["status"], "COMPLETED")
        self.assertEqual(result.result["counts"]["completed"], len(original.workload))
        self.assertEqual(len(result.result["memory"]), 8)
        self.assertEqual(result.result["prefix_cache"]["attached_block_references"], 0)
        self.assertEqual(result.result["prefix_cache"]["private_blocks"], 0)
        self.assertEqual(result.result["swap"]["host_reserved_bytes"], 0)
        for request in result.requests:
            self.assertEqual(request["stage_computed_tokens"], [request["num_computed_tokens"]] * 2)

    def test_new_18_point_protocol_is_explicit_and_nominal_output_contract_stays_intact(self):
        raw_plan = plan()
        nominal = run_chapter1(raw_plan)
        result = run_calibrated_chapter1(raw_plan, imported())
        self.assertEqual(result["protocol"], CALIBRATED_PROTOCOL)
        self.assertEqual(result["base_formula_version"], PROTOCOL)
        self.assertEqual(result["point_count"], 18)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(nominal["protocol"], PROTOCOL)
        self.assertNotIn("applied_calibration", nominal)
        self.assertNotIn("nominal", nominal["points"][0]["components"][0])
        for old, new in zip(nominal["points"], result["points"]):
            self.assertEqual(new["nominal_t_step_ms"], old["t_step_ms"])
            self.assertNotEqual(new["prediction_id"], old["prediction_id"])
            self.assertEqual(new["global_flops"], old["global_flops"])
            self.assertEqual(new["global_bytes"], old["global_bytes"])
            for before, after in zip(old["components"], new["components"]):
                self.assertEqual(after["t_compute_ms"], before["t_compute_ms"] / .5)
                self.assertEqual(after["t_memory_ms"], before["t_memory_ms"] / .8)
                self.assertEqual(after["t_ms"], max(after["t_compute_ms"], after["t_memory_ms"]))
                self.assertEqual(after["ridge"], before["ridge"] * .5 / .8)
                self.assertEqual(after["nominal"]["bound"], before["bound"])
            self.assertEqual(new["t_step_ms"], math.fsum(c["t_ms"] for c in new["components"]))
        self.assertEqual(result["reconciliation"]["time_rows_scope"], "nominal_baseline_before_eta_beta_transfer")
        self.assertFalse(result["applied_calibration"]["whole_model_validation_established"])

    def test_calibration_hardware_mismatch_or_tampering_is_rejected(self):
        changed = plan()
        changed["hardware"]["bandwidth_gbps"] = 600
        with self.assertRaises(SimulationError):
            run_calibrated_chapter1(changed, imported())
        result = imported()
        result["coefficients"]["beta"] = .99
        with self.assertRaises(SimulationError):
            run_calibrated_chapter1(plan(), rehash(result))

    def test_dense_component_bound_can_switch_after_eta_beta_scaling(self):
        changed = plan()
        changed["counting"]["prefill_gemm_activation_bytes_per_layer_hidden"] = 80
        old = run_chapter1(changed)["points"][1]["components"][0]
        new = run_calibrated_chapter1(changed, imported())["points"][1]["components"][0]
        self.assertEqual(old["bound"], "memory")
        self.assertEqual(new["bound"], "compute")
        self.assertEqual(new["nominal"]["bound"], "memory")
        self.assertGreater(new["t_compute_ms"], new["t_memory_ms"])
        self.assertEqual(new["t_ms"], old["t_compute_ms"] / .5)

    def test_transfer_report_does_not_alias_input_calibration_metadata(self):
        calibration = imported()
        report = run_calibrated_chapter1(plan(), calibration)
        calibration["source"]["label"] = "mutated after prediction"
        calibration["fitting_domain"]["tp"] = 8
        self.assertNotEqual(report["applied_calibration"]["source"]["label"], calibration["source"]["label"])
        self.assertEqual(report["applied_calibration"]["fitting_domain"]["tp"], 1)


class CourseCalibrationReportTests(unittest.TestCase):
    def test_writer_preserves_exact_source_and_outputs_the_unapplied_patch(self):
        raw = b'\r\n ' + FIXTURE.read_bytes() + b' \r\n'
        report = import_course_calibration(raw, source=SOURCE)
        with tempfile.TemporaryDirectory() as directory:
            assets = write_course_calibration_report(report, Path(directory) / "report")
            self.assertEqual(assets["source.json"].read_bytes(), raw)
            self.assertEqual(json.loads(assets["report.json"].read_bytes()), report)
            self.assertEqual(json.loads(assets["calibration_patch.json"].read_bytes()), report["calibration_patch"])
            with assets["holdout_points.csv"].open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[3]["nominal_bound"], "memory")
            self.assertEqual(rows[3]["recomputed_bound"], "compute")
            svg = assets["calibration.svg"].read_text()
            ET.fromstring(svg)
            self.assertIn(report["data_sha256"], svg)
            self.assertIn("test_fixture", svg)
            html = assets["report.html"].read_text()
            self.assertIn("无硬件PASS", html)
            self.assertIn("未使用留出点重新拟合", html)
            self.assertEqual(html.count('<table>'), 2)

    def test_new_report_can_render_all_18_points_and_nominal_comparison_csv(self):
        result = run_calibrated_chapter1(plan(), imported())
        with tempfile.TemporaryDirectory() as directory:
            assets = write_chapter1_report(result, Path(directory) / "chapter1-calibrated")
            with assets["components.csv"].open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 54)
            self.assertIn("nominal_t_compute_ms", rows[0])
            self.assertEqual(rows[0]["eta_applied"], "0.5")
            roofline = assets["roofline-rx9070xt.svg"].read_text()
            ET.fromstring(roofline)
            self.assertIn("primitive η/β 迁移", roofline)
            self.assertNotIn("nominal η=1", roofline)
            self.assertIn("97.5 TFLOP/s", roofline)
            self.assertIn("AI=190.4297", roofline)
            html = assets["report.html"].read_text()
            self.assertIn("校准前nominal基线", html)
            self.assertIn("η=0.5000，β=0.8000", html)
            self.assertIn("Nominal ms", html)
            self.assertIn("迁移后 ms", html)
            self.assertIn("变化 %", html)
            self.assertIn('<details><summary>完整适用域与证据限制', html)
            self.assertIn("非新真机测量", html)

    def test_writer_refuses_overwrite_or_edited_data_before_writing(self):
        result = imported()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing"
            target.mkdir()
            marker = target / "keep"
            marker.write_text("preserve")
            with self.assertRaises(SimulationError):
                write_course_calibration_report(result, target)
            self.assertEqual(marker.read_text(), "preserve")
            bad = deepcopy(result)
            bad["coefficients"]["eta"] = .7
            with self.assertRaises(SimulationError):
                write_course_calibration_report(rehash(bad), Path(directory) / "bad")
            self.assertFalse((Path(directory) / "bad").exists())

    def test_frozen_report_rerenders_deterministically(self):
        report = imported()
        report["stored_at"] = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        rehash(report)
        with tempfile.TemporaryDirectory() as directory:
            first = write_course_calibration_report(report, Path(directory) / "first")
            second = write_course_calibration_report(report, Path(directory) / "second")
            for name in first:
                self.assertEqual(first[name].read_bytes(), second[name].read_bytes())
