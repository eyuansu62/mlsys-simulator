"""Hand-checked study rendering: overload is distinct from bad acquisition."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from mlsys_sim.load_validation_report import _canonical, write_closed_loop_study_report
from mlsys_sim.load_validation import predict_closed_loop, validate_closed_loop
from mlsys_sim.schema import CalibrationSpec, Experiment, SimulationError, StrategySpec


BASELINE = [1, 2, 4, 8, 16, 32, 64]
HOLDOUT = [8, 16, 32]
METRICS = ("ttft_p99_ms", "tpot_p99_ms", "end_to_end_p99_ms", "elapsed_output_tokens_per_s")


def cell(concurrency, replication):
    p = {"ttft_p99_ms": 10 * concurrency + replication, "tpot_p99_ms": 5 + replication / 10,
         "end_to_end_p99_ms": 20 * concurrency + replication, "elapsed_output_tokens_per_s": 200 + concurrency * 10,
         "completed_samples": 1000, "sample_counts": {"ttft": 1000, "tpot": 1000, "end_to_end": 1000}}
    m = {**p, "sample_counts": dict(p["sample_counts"])}
    for metric in METRICS:
        m[metric] = p[metric] * (.9 if metric == "elapsed_output_tokens_per_s" else 1.1)
    counts = {"total": 1000, "completed": 1000, "failed": 0, "timed_out": 0, "unfinished": 0, "not_admitted": 0}
    ps = "PASS" if p["ttft_p99_ms"] <= 250 and p["end_to_end_p99_ms"] <= 500 else "FAIL"
    ms = "PASS" if m["ttft_p99_ms"] <= 250 and m["end_to_end_p99_ms"] <= 500 else "FAIL"
    return {"concurrency": concurrency, "replication": replication, "policy_sha256": "b" * 64,
        "request_stream_sha256": "c" * 64, "verdict": "PASS" if ps == ms == "PASS" else "FAIL",
        "acquisition_verdict": "PASS", "prediction_error_verdict": "PASS", "prediction_sla_verdict": ps,
        "observation_sla_verdict": ms, "sla_agreement": ps == ms, "checks": [],
        "metrics": {metric: {"predicted": p[metric], "measured": m[metric],
            "absolute_error": abs(p[metric] - m[metric]), "relative_error": abs(p[metric] - m[metric]) / m[metric],
            "predicted_sample_count": 1000, "measured_sample_count": 1000, "verdict": "PASS"} for metric in METRICS},
        "prediction": {"status": "COMPLETED", "metrics": p, "counts": dict(counts), "offered_count": 1000,
            "not_offered_count": 0, "constraints": {"verdict": ps, "checks": []}},
        "observation": {"status": "COMPLETED", "metrics": m, "counts": dict(counts), "offered_count": 1000,
            "not_offered_count": 0, "constraints": {"verdict": ms, "checks": []}, "source_kind": "test_fixture",
            "session_sha256": "d" * 64, "timing_boundary": "server_token_ready_v1"}}


def validation(cohort, grid):
    return {"schema_version": 1, "kind": "closed_loop_validation", "protocol": "finite_cohort_closed_loop_v1",
        "status": "COMPLETED", "verdict": "FAIL", "study_id": "hand-calculated-study", "cohort": cohort,
        "predictions_sha256": "a" * 64, "spec_sha256": "b" * 64, "identity_sha256": "c" * 64,
        "contract_sha256": "d" * 64, "hardware_validated": False,
        "contract": {"schema_version": 1, "min_completed_samples": 1000,
            "constraints": {"latency": {"p99_ttft_ms": 250, "p99_tpot_ms": 20, "p99_e2e_ms": 500}},
            "max_relative_error": {metric: .2 for metric in METRICS}, "timing_boundary": "server_token_ready_v1"},
        "runs": [cell(c, r) for c in grid for r in range(3)],
        "points": [{"concurrency": c, "verdict": "PASS" if c <= 16 else "FAIL"} for c in grid],
        "metric_contract": {"no_refit": True}, "limitations": ["Hand-calculated CPU fixture, not hardware sampling"]}


def study():
    return {"schema_version": 1, "study_id": "report-fixture", "title": "有限cohort课程验证：手算夹具",
        "evidence_kind": "test_fixture", "metadata": {"model": "hand-calculated dense model fixture",
            "device": "CPU fixture only", "runtime": "test-fixture; no Runtime ran",
            "protocol": "finite_cohort_closed_loop_v1", "seed": 20261111, "sampling_seed": 0,
            "replication_stream": "SHA-derived base_seed/replication length stream, not seed+r",
            "input_lengths": {"baseline": [256], "short": [64, 128], "long": [512, 1024]}, "output_tokens": 16},
        "freeze": {"frozen_at": "2026-11-11T07:32:21Z", "receipt_sha256": "e" * 64,
            "raw_prediction_sha256": {name: "a" * 64 for name in ("baseline", "short", "long")},
            "canonical_prediction_sha256": {name: "b" * 64 for name in ("baseline", "short", "long")},
            "collection_source_sha256": "f" * 64},
        "selection": {"predicted_safe_cap": 16, "holdout_concurrencies": HOLDOUT, "replications": 3},
        "cohorts": {name: validation(name, BASELINE if name == "baseline" else HOLDOUT)
                    for name in ("baseline", "short", "long")},
        "attempt_summary": {"expected_cells": 39, "attempted_cells": 39, "completed_cells": 39, "failed_cells": 0},
        "limitations": ["本例所有数字均手算；没有采样GPU，也没有真实收据。"]}


def render(value):
    with tempfile.TemporaryDirectory() as directory:
        files = write_closed_loop_study_report(value, Path(directory) / "report")
        return {name: path.read_text(encoding="utf-8") for name, path in files.items()}


def summary(files):
    return json.loads(files["summary.json"])


class ClosedLoopStudySummaryTests(unittest.TestCase):
    def test_deliberate_overload_is_not_an_acquisition_or_accuracy_failure(self):
        source = study()
        before = deepcopy(source)
        files = render(source)
        report = summary(files)
        self.assertEqual(report["acquisition_verdict"], "PASS")
        self.assertEqual(report["prediction_error_verdict"], "PASS")
        self.assertTrue(all(value["verdict"] == "FAIL" for value in source["cohorts"].values()))
        self.assertEqual(report["sla_agreement"], {"agree": 39, "known": 39, "unknown": 0,
            "false_safe_or_unconfirmed": 0, "false_safe_observed_fail": 0, "predicted_pass_observed_unknown": 0})
        self.assertEqual(report["cohorts"]["baseline"]["largest_observed_safe_grid_point"], 16)
        self.assertFalse(report["cohorts"]["baseline"]["safe_grid_nonmonotonic"])
        self.assertFalse(report["hardware_validated"])
        self.assertEqual(json.loads(files["report.json"]), source)
        self.assertEqual(source, before)

    def test_missing_repeat_is_unknown_false_safe_unconfirmed_and_never_zero(self):
        source = study()
        row = source["cohorts"]["baseline"]["runs"][2]
        row["observation"] = None
        row["observation_sla_verdict"] = "UNKNOWN"
        row["metrics"] = {}
        for key in ("acquisition_verdict", "prediction_error_verdict", "sla_agreement"):
            row.pop(key)
        files = render(source)
        report = summary(files)
        self.assertEqual(report["observed_cells"], 38)
        self.assertEqual(report["missing_cells"], 1)
        self.assertEqual(report["acquisition_verdict"], "UNKNOWN")
        self.assertEqual(report["prediction_error_verdict"], "UNKNOWN")
        self.assertEqual(report["sla_agreement"]["predicted_pass_observed_unknown"], 1)
        self.assertEqual(report["sla_agreement"]["false_safe_observed_fail"], 0)
        rows = list(csv.DictReader(files["cells.csv"].splitlines()))
        self.assertEqual(len(rows), 39)
        self.assertEqual(rows[2]["observed_ttft_p99_ms"], "")
        self.assertIn("缺测", rows[2]["flags"])
        svg = files["baseline-latency.svg"]
        self.assertIn("n=2/3", svg)
        self.assertNotIn("base seed 20261111 子流） observation ttft_p99_ms=13.2", svg)
        self.assertIn("base seed 20261111 子流） prediction ttft_p99_ms=12", svg)

    def test_censored_repeat_stays_in_table_but_not_in_tail_plot_or_safe_set(self):
        source = study()
        row = source["cohorts"]["baseline"]["runs"][4]
        observed = row["observation"]
        observed["status"] = "UNRESOLVED"
        observed["counts"].update(completed=999, unfinished=1)
        observed["metrics"]["completed_samples"] = 999
        observed["metrics"]["sample_counts"] = {name: 999 for name in ("ttft", "tpot", "end_to_end")}
        row.update(acquisition_verdict="FAIL", prediction_error_verdict="UNKNOWN", observation_sla_verdict="FAIL")
        files = render(source)
        report = summary(files)
        self.assertEqual(report["censored_cells"], 1)
        self.assertEqual(report["acquisition_verdict"], "FAIL")
        point = report["cohorts"]["baseline"]["points"][1]
        self.assertFalse(point["observed_all_repetitions_pass"])
        rows = list(csv.DictReader(files["cells.csv"].splitlines()))
        self.assertEqual(rows[4]["completed_samples"], "999")
        self.assertNotEqual(rows[4]["observed_ttft_p99_ms"], "")
        self.assertIn("删失", rows[4]["flags"])
        self.assertIn("完成样本不足", rows[4]["flags"])
        self.assertNotIn("C2 r1（base seed 20261111 子流） observation", files["baseline-latency.svg"])

    def test_sla_false_safe_can_exist_even_when_the_error_target_passes(self):
        source = study()
        for row in source["cohorts"]["baseline"]["runs"]:
            if row["concurrency"] == 16:
                row["observation_sla_verdict"] = "FAIL"
        source["cohorts"]["baseline"]["contract"]["constraints"]["latency"]["p99_ttft_ms"] = 170
        files = render(source)
        report = summary(files)
        self.assertEqual(report["prediction_error_verdict"], "PASS")
        self.assertEqual(report["acquisition_verdict"], "PASS")
        self.assertEqual(report["sla_agreement"]["false_safe_observed_fail"], 3)
        self.assertEqual(report["sla_agreement"]["agree"], 36)
        self.assertEqual(report["cohorts"]["baseline"]["largest_observed_safe_grid_point"], 8)
        self.assertIn('stroke="#b42538"', files["baseline-latency.svg"])

    def test_nonmonotonic_safe_grid_never_becomes_interpolated_capacity(self):
        source = study()
        for row in source["cohorts"]["baseline"]["runs"]:
            if row["concurrency"] == 4:
                row["observation_sla_verdict"] = "FAIL"
                row["observation"]["metrics"]["ttft_p99_ms"] = 1000
                row["prediction_error_verdict"] = "FAIL"
        files = render(source)
        report = summary(files)
        baseline = report["cohorts"]["baseline"]
        self.assertTrue(baseline["safe_grid_nonmonotonic"])
        self.assertEqual(baseline["largest_observed_safe_grid_point"], 16)
        self.assertFalse(baseline["interpolated"])
        self.assertFalse(baseline["global_capacity_established"])
        self.assertIn("FAIL后再PASS", files["report.html"])
        self.assertIn("不是全局容量", files["report.md"])

    def test_missing_entire_cell_retains_the_predeclared_row(self):
        source = study()
        source["cohorts"]["short"]["runs"].pop(1)
        files = render(source)
        report = summary(files)
        self.assertEqual(report["expected_cells"], 39)
        self.assertEqual(report["missing_cells"], 1)
        rows = list(csv.DictReader(files["cells.csv"].splitlines()))
        missing = next(r for r in rows if r["cohort"] == "short" and r["clients"] == "8" and r["replication"] == "1")
        self.assertEqual(missing["acquisition"], "UNKNOWN")
        self.assertEqual(missing["predicted_ttft_p99_ms"], "")

    def test_all_missing_observations_do_not_invent_a_safe_point_or_measurements(self):
        source = study()
        for value in source["cohorts"].values():
            for row in value["runs"]:
                row["observation"] = None
                row["observation_sla_verdict"] = "UNKNOWN"
                row["acquisition_verdict"] = row["prediction_error_verdict"] = "UNKNOWN"
        files = render(source)
        report = summary(files)
        self.assertEqual(report["missing_cells"], 39)
        self.assertTrue(all(c["largest_observed_safe_grid_point"] is None for c in report["cohorts"].values()))
        self.assertNotIn("） observation ", files["baseline-latency.svg"])
        self.assertIn("n=0/3", files["baseline-latency.svg"])


class ClosedLoopStudyArtifactTests(unittest.TestCase):
    def test_digest_labels_identify_canonical_input_instead_of_pretty_file_bytes(self):
        files = render(study())
        loaded = json.loads(files["report.json"])
        canonical_digest = hashlib.sha256(_canonical(loaded)).hexdigest()
        pretty_digest = hashlib.sha256(files["report.json"].encode("utf-8")).hexdigest()
        self.assertEqual(summary(files)["study_sha256"], canonical_digest)
        self.assertNotEqual(canonical_digest, pretty_digest)
        for name in ("report.html", "report.md", "baseline-latency.svg", "cohort-throughput.svg", "holdout-e2e.svg"):
            self.assertIn("canonical report输入SHA256", files[name])
            self.assertIn(canonical_digest, files[name])
            self.assertNotIn("report.json SHA256", files[name])
            self.assertNotIn("report.json sha256=", files[name])

    def test_real_validator_sanitized_cpu_output_is_accepted_without_touching_the_collector(self):
        source = study()
        source["selection"].update(holdout_concurrencies=[1, 2], replications=2)
        source["attempt_summary"].update(expected_cells=12, attempted_cells=12, completed_cells=12)
        source["metadata"].update(input_lengths={name: [16, 32] for name in ("baseline", "short", "long")}, output_tokens=3)
        experiment = Experiment(strategy=StrategySpec(scheduler="fifo"),
                                calibration=CalibrationSpec(backend="synthetic", prefill_ms=10, decode_ms=1))
        for name in ("baseline", "short", "long"):
            spec = {"schema_version": 1, "study_id": "cpu-validator-fixture", "cohort": name,
                "concurrency_grid": [1, 2], "requests_per_run": 6, "replications": 2, "seed": 20261111,
                "think_ms": 0, "input_lengths": [16, 32], "output_tokens": 3, "timeout_ms": 1000, "max_run_ms": 10000}
            predictions = predict_closed_loop(experiment, spec)
            runs = deepcopy(predictions["runs"])
            for index, row in enumerate(runs):
                row.update(source={"kind": "test_fixture", "reference": "CPU simulation, not a GPU acquisition"},
                    session_id=f"fixture-{index}", collection_started_at="2026-11-11T08:00:00Z",
                    collection_ended_at="2026-11-11T08:01:00Z")
            observed = {"schema_version": 1, "study_id": predictions["study_id"], "cohort": name,
                "spec_sha256": predictions["spec_sha256"], "identity_sha256": predictions["identity_sha256"], "runs": runs}
            contract = {"schema_version": 1, "prediction_frozen_at": source["freeze"]["frozen_at"],
                "min_completed_samples": 6, "timing_boundary": "server_token_ready_v1",
                "max_relative_error": {metric: 0 for metric in METRICS}, "allow_test_fixture": True,
                "constraints": {"schema_version": 1, "min_completed_samples": 6,
                    "latency": {"p99_ttft_ms": 1000, "p99_tpot_ms": 1000, "p99_e2e_ms": 1000}}}
            source["cohorts"][name] = validate_closed_loop(predictions, observed, contract)
        files = render(source)
        self.assertEqual(summary(files)["expected_cells"], 12)
        self.assertEqual(summary(files)["acquisition_verdict"], "PASS")
        self.assertEqual(summary(files)["prediction_error_verdict"], "PASS")
        self.assertEqual(summary(files)["sla_agreement"]["agree"], 12)

    def test_same_source_hash_all_repeat_markers_exact_values_and_non_ci_labels(self):
        source = study()
        files = render(source)
        digest = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(summary(files)["study_sha256"], digest)
        self.assertEqual(set(files), {"report.json", "summary.json", "cells.csv", "report.html", "report.md",
            "baseline-latency.svg", "cohort-throughput.svg", "holdout-e2e.svg"})
        expected_markers = {"baseline-latency.svg": 7 * 3 * 2 * 3, "cohort-throughput.svg": 39 * 2,
                            "holdout-e2e.svg": 18 * 2}
        for name, count in expected_markers.items():
            svg = files[name]
            parsed = ET.fromstring(svg)
            self.assertEqual(len(parsed.findall("{http://www.w3.org/2000/svg}g")), count)
            self.assertIn(digest, svg)
            self.assertIn("min–max，非置信区间", svg)
            self.assertIn("hardware_validated=false", svg)
        self.assertIn("C1 r0（base seed 20261111 子流） observation ttft_p99_ms=11", files["baseline-latency.svg"])
        self.assertIn("C1 r1（base seed 20261111 子流） observation ttft_p99_ms=12.1", files["baseline-latency.svg"])
        self.assertIn("C1 r2（base seed 20261111 子流） observation ttft_p99_ms=13.2", files["baseline-latency.svg"])
        self.assertIn("不是不同Runtime采样seed", files["report.html"])
        self.assertEqual(files["report.html"].count('<img '), 3)
        self.assertIn('<details><summary>', files["report.html"])

    def test_full_39_rows_have_samples_metrics_error_and_check_trace(self):
        source = study()
        row = source["cohorts"]["long"]["runs"][0]
        row["checks"] = [{"id": "measurement.time_order", "verdict": "UNKNOWN", "reason": "fixture-example"}]
        row["acquisition_verdict"] = "UNKNOWN"
        files = render(source)
        rows = list(csv.DictReader(files["cells.csv"].splitlines()))
        self.assertEqual(len(rows), 39)
        target = next(r for r in rows if r["cohort"] == "long" and r["clients"] == "8" and r["replication"] == "0")
        self.assertEqual(target["completed_samples"], "1000")
        self.assertEqual(target["ttft_samples"], "1000")
        self.assertEqual(float(target["relative_error_ttft_p99_ms"]), abs(80 - 88) / 88)
        self.assertEqual(json.loads(target["nonpass_checks"])[0]["id"], "measurement.time_order")
        self.assertIn("measured_sample_count", json.loads(target["metric_errors_json"])["ttft_p99_ms"])

    def test_html_escapes_title_and_metadata_without_executing_markup(self):
        source = study()
        source["title"] = '<script>alert("fixture")</script>'
        source["metadata"]["model"] = '<img src=x onerror=alert(1)>'
        html = render(source)["report.html"]
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<img src=x', html)

    def test_frozen_study_renders_byte_reproducibly(self):
        source = study()
        first, second = render(source), render(source)
        self.assertEqual(first, second)

    def test_existing_directory_or_symlink_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "existing"
            target.mkdir()
            marker = target / "keep"
            marker.write_text("keep")
            with self.assertRaises(SimulationError) as raised:
                write_closed_loop_study_report(study(), target)
            self.assertEqual(raised.exception.code, "OUTPUT_EXISTS")
            self.assertEqual(marker.read_text(), "keep")
            link = root / "link"
            link.symlink_to(root / "absent")
            with self.assertRaises(SimulationError):
                write_closed_loop_study_report(study(), link)


class ClosedLoopStudyInputTests(unittest.TestCase):
    def test_rejects_nonfinite_invalid_counts_duplicates_raw_payload_and_cell_mismatch(self):
        for field, value in (("ttft_p99_ms", float("nan")), ("tpot_p99_ms", True),
                             ("elapsed_output_tokens_per_s", float("inf")), ("ttft_p99_ms", 1e308)):
            source = study()
            source["cohorts"]["baseline"]["runs"][0]["observation"]["metrics"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SimulationError):
                render(source)
        for mutate in (lambda s: s["cohorts"]["baseline"]["runs"].append(deepcopy(s["cohorts"]["baseline"]["runs"][0])),
                       lambda s: s["cohorts"]["baseline"]["runs"][0].update(requests=[{"private": "raw request"}]),
                       lambda s: s["attempt_summary"].update(expected_cells=38),
                       lambda s: s["cohorts"]["baseline"]["runs"][0]["observation"]["counts"].update(completed=True)):
            source = study()
            mutate(source)
            with self.assertRaises(SimulationError):
                render(source)

    def test_rejects_private_freeze_paths_bad_hashes_naive_time_and_fixture_upgrade(self):
        for field, value in (("receipt_sha256", "/private/user/path"), ("collection_source_sha256", "x" * 64),
                             ("frozen_at", "2026-11-11T07:32:21")):
            source = study()
            source["freeze"][field] = value
            with self.subTest(field=field), self.assertRaises(SimulationError):
                render(source)
        source = study()
        source["evidence_kind"] = "measured"
        with self.assertRaises(SimulationError):
            render(source)
        source = study()
        source["cohorts"]["baseline"]["hardware_validated"] = True
        with self.assertRaises(SimulationError):
            render(source)

    def test_unknown_metadata_or_missing_seed_is_shown_without_guessing(self):
        source = study()
        source["metadata"].pop("seed")
        files = render(source)
        self.assertIn("base seed未提供", files["cells.csv"])
        self.assertNotIn("base seed 20261111 子流", files["cells.csv"])
