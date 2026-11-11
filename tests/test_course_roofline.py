"""Chapter 1's frozen paper protocol, independent arithmetic and artifact safety."""

from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from mlsys_sim.course_roofline import PROTOCOL, run_chapter1, write_chapter1_report
from mlsys_sim.schema import SimulationError


PLAN = Path(__file__).resolve().parents[1] / "examples" / "course" / "chapter1.plan.json"


def plan():
    return json.loads(PLAN.read_text(encoding="utf-8"))


def point(report, model="Qwen3-0.6B", phase="prefill", size=512):
    return next(p for p in report["points"] if p["model_name"] == model and p["phase"] == phase
                and p["batch_size" if phase == "decode" else "sequence_length"] == size)


class Chapter1FormulaTests(unittest.TestCase):
    def setUp(self):
        self.report = run_chapter1(plan())

    def test_grid_is_the_complete_course_18_points(self):
        report = self.report
        self.assertEqual(report["point_count"], 18)
        self.assertEqual(len(report["points"]), 18)
        self.assertEqual(Counter(p["model_name"] for p in report["points"]),
                         {"Qwen3-0.6B": 6, "Qwen3-1.7B": 6, "Qwen3-4B": 6})
        self.assertEqual(len({p["prediction_id"] for p in report["points"]}), 18)
        for name in ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"):
            for size in (128, 512, 2048):
                self.assertEqual(point(report, name, "prefill", size)["batch_size"], 1)
            for batch in (1, 4, 16):
                self.assertEqual(point(report, name, "decode", batch)["sequence_length"], 1024)

    def test_full_square_attention_anchor_and_separate_rooflines(self):
        gemm, attention, vector = point(self.report)["components"]
        self.assertEqual(attention["flops"], 60_129_542_144)
        self.assertEqual(attention["bytes"], 176_160_768)
        self.assertAlmostEqual(attention["ai"], 341 + 1 / 3)
        self.assertAlmostEqual(attention["ridge"], 304.6875)
        self.assertAlmostEqual(vector["ridge"], 76.09375)
        self.assertEqual((gemm["pipeline"], attention["pipeline"], vector["pipeline"]),
                         ("matrix", "matrix", "vector"))
        self.assertEqual(attention["bound"], "compute")
        self.assertEqual(vector["bound"], "memory")
        # This is deliberately not half-square causal work or hidden_size=1024 work.
        causal = 4 * 28 * 2048 * (512 * 513 // 2)
        self.assertGreater(attention["flops"], causal * 1.99)

    def test_decode_context_is_inclusive_and_attention_ai_does_not_increase_with_batch(self):
        for model, ratio in (("Qwen3-0.6B", 2), ("Qwen3-1.7B", 2), ("Qwen3-4B", 4)):
            for batch in (1, 4, 16):
                with self.subTest(model=model, batch=batch):
                    p = point(self.report, model, "decode", batch)
                    gemm, attn, _ = p["components"]
                    self.assertEqual(attn["ai"], ratio)
                    self.assertEqual(gemm["ai"], batch)
                    self.assertTrue(p["context_includes_current"])
                    self.assertEqual(gemm["bound"], "memory")
        attn = point(self.report, phase="decode", size=1)["components"][1]
        self.assertEqual(attn["flops"], 234_881_024)
        self.assertEqual(attn["bytes"], 117_440_512)
        self.assertEqual(attn["t_memory_ms"], 0.1835008)

    def test_exact_qwen_parameters_and_default_head_policy(self):
        prefill = point(self.report)
        decode = point(self.report, phase="decode", size=1)
        self.assertEqual(prefill["linear_parameters"], 440_401_920)
        self.assertEqual(prefill["head_parameters"], 155_582_464)
        self.assertEqual(prefill["components"][0]["flops"], 2 * 440_401_920 * 512)
        self.assertEqual(decode["components"][0]["bytes"], 1_191_968_768)
        self.assertEqual(decode["components"][0]["flops"], 1_191_968_768)
        # Tied embeddings share storage but the output projection still executes once.
        self.assertEqual(decode["linear_parameters"] + decode["head_parameters"], 595_984_384)

    def test_last_token_head_is_explicitly_optional_and_does_not_change_decode(self):
        modified = plan()
        modified["counting"]["prefill_lm_head"] = "last_token"
        included = run_chapter1(modified)
        omitted_gemm = point(self.report)["components"][0]
        included_gemm = point(included)["components"][0]
        self.assertEqual(included_gemm["flops"] - omitted_gemm["flops"], 311_164_928)
        self.assertEqual(included_gemm["bytes"] - omitted_gemm["bytes"], 311_164_928)
        self.assertEqual(point(included, phase="decode", size=1)["components"],
                         point(self.report, phase="decode", size=1)["components"])
        self.assertNotEqual(point(included)["prediction_id"], point(self.report)["prediction_id"])

    def test_step_is_sum_of_component_rooflines_and_global_ai_has_no_bound(self):
        for p in self.report["points"]:
            with self.subTest(config=p["config_id"]):
                self.assertEqual(p["t_step_ms"], math.fsum(c["t_ms"] for c in p["components"]))
                self.assertNotIn("global_bound", p)
                for c in p["components"]:
                    self.assertEqual(c["t_ms"], max(c["t_compute_ms"], c["t_memory_ms"]))
        p = point(self.report)
        wrong_global_max = max(p["global_flops"] / 195e9, p["global_bytes"] / 640e6)
        self.assertGreater(p["t_step_ms"], wrong_global_max)

    def test_all_boundaries_match_the_frozen_hand_calculations(self):
        required = {"component", "flops", "bytes", "ai", "ridge", "bound", "t_compute_ms", "t_memory_ms", "t_ms"}
        for p in self.report["points"]:
            expected = "memory" if p["phase"] == "decode" or p["sequence_length"] == 128 else "compute"
            self.assertEqual([c["bound"] for c in p["components"]], [expected, expected, "memory"])
            for c in p["components"]:
                self.assertTrue(required <= c.keys())

    def test_primitives_and_naive_asymptotic_disagreement_are_honest(self):
        matmul, norm, flash, naive = self.report["primitives"]
        self.assertEqual((matmul["flops"], matmul["bytes"]), (137_438_953_472, 100_663_296))
        self.assertAlmostEqual(matmul["ai"], 1365 + 1 / 3)
        self.assertEqual(matmul["bound"], "compute")
        self.assertEqual((norm["flops"], norm["bytes"], norm["ai"]), (8_388_608, 8_388_608, 1))
        self.assertEqual(norm["pipeline"], "vector")
        self.assertEqual((flash["bound"], naive["bound"]), ("compute", "memory"))
        self.assertEqual(naive["bytes"], 645_922_816)
        self.assertAlmostEqual(naive["ai"], 93.0909090909091)
        reconciliation = self.report["reconciliation"]
        self.assertEqual(reconciliation["naive_attention"]["formula_derived_asymptotic_ai"], 128)
        self.assertEqual(reconciliation["naive_attention"]["status"], "course_formula_inconsistency")
        self.assertEqual(len(reconciliation["time_rows"]), 6)
        self.assertTrue(any(not r["matches_one_decimal_rounding"] for r in reconciliation["time_rows"]))
        self.assertEqual(reconciliation["kind"], "formula_and_rounding_reconciliation_not_hardware_error")

    def test_provenance_is_reproducible_without_claiming_a_measurement(self):
        report = self.report
        expected = hashlib.sha256(json.dumps(plan(), sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(report["provenance"]["plan_sha256"], expected)
        self.assertEqual(report["formula_version"], PROTOCOL)
        self.assertFalse(report["hardware_validated"])
        self.assertEqual(report["provenance"]["measurement_sources"], [])
        self.assertFalse(report["provenance"]["historical_prediction_before_measurement_established"])
        self.assertEqual(len(report["provenance"]["model_config_sha256"]), 3)
        self.assertEqual(report["plan"]["pending_hardware"]["status"], "pending_parameters")
        self.assertIsNone(report["plan"]["pending_hardware"]["matrix_tflops"])
        self.assertLess(abs((datetime.fromisoformat(report["stored_at"].replace("Z", "+00:00"))
                            - datetime.now(timezone.utc)).total_seconds()), 10)
        for p in report["points"]:
            self.assertEqual(p["stored_at"], report["stored_at"])
            self.assertEqual(p["calibration_version"], "nominal")
            self.assertFalse(p["hardware_validated"])
            self.assertTrue(p["prediction_id"].startswith("pred_ch1_"))

    def test_input_and_returned_formula_maps_do_not_alias(self):
        source = plan()
        before = deepcopy(source)
        report = run_chapter1(source)
        self.assertEqual(source, before)
        source["models"][0]["hf_config"]["hidden_size"] = 1
        self.assertEqual(report["plan"]["models"][0]["hf_config"]["hidden_size"], 1024)
        report["formulas"]["step_time"] = "changed"
        self.assertNotEqual(run_chapter1(before)["formulas"]["step_time"], "changed")


class Chapter1ValidationTests(unittest.TestCase):
    def test_rejects_nonfinite_invalid_unknown_or_unbounded_plans(self):
        changes = [("protocol", "other"), ("schema_version", True), ("experiment_id", "../escape"),
                   ("experiment_id", "=CSV()"), ("models", []), ("models", [None] * 3),
                   ("axes", {"prefill_lengths": "128", "decode_batch_sizes": [1, 4, 16], "decode_context": 1024}),
                   ("calibration", {"version": "nominal", "eta": True, "t_fixed_ms": 0}),
                   ("extra", "not accepted")]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(SimulationError):
                run_chapter1({**plan(), key: value})
        for value in (True, None, -1, 0, float("nan"), float("inf"), 10**400, 1e-100):
            source = plan()
            source["hardware"]["matrix_tflops"] = value
            with self.subTest(peak=value), self.assertRaises(SimulationError):
                run_chapter1(source)
        source = plan()
        source["hardware"]["source"] = "x" * (1024 * 1024)
        with self.assertRaises(SimulationError):
            run_chapter1(source)

    def test_rejects_wrong_geometry_dtype_unknown_pending_hardware_and_formula_policy(self):
        modifications = [("num_attention_heads", 8), ("hidden_size", 2048), ("torch_dtype", "float16"),
                         ("tie_word_embeddings", False), ("head_dim", 64)]
        for key, value in modifications:
            source = plan()
            source["models"][0]["hf_config"][key] = value
            with self.subTest(field=key), self.assertRaises(SimulationError):
                run_chapter1(source)
        for key, value in (("prefill_attention", "causal"), ("decode_context_includes_current", False),
                           ("prefill_lm_head", "every_token"), ("elementwise_bytes_per_layer_hidden", 0)):
            source = plan()
            source["counting"][key] = value
            with self.subTest(field=key), self.assertRaises(SimulationError):
                run_chapter1(source)
        source = plan()
        source["pending_hardware"]["matrix_tflops"] = 100
        with self.assertRaises(SimulationError):
            run_chapter1(source)

    def test_accepts_only_current_utc_timestamps_and_forbids_historical_backdating(self):
        now = datetime.now(timezone.utc)
        report = run_chapter1(plan(), stored_at=now.isoformat())
        self.assertTrue(report["stored_at"].endswith("Z"))
        for value in ((now - timedelta(days=1)).isoformat(), (now + timedelta(days=1)).isoformat(),
                      now.replace(tzinfo=None).isoformat(), "2020-01-01", "not-a-date", True):
            with self.subTest(timestamp=value), self.assertRaises(SimulationError):
                run_chapter1(plan(), stored_at=value)


class Chapter1ReportTests(unittest.TestCase):
    def test_report_writes_all_figures_and_exact_54_component_csv_from_one_source(self):
        report = run_chapter1(plan())
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "chapter1"
            assets = write_chapter1_report(report, out)
            self.assertEqual(set(assets), {"report.json", "components.csv", "report.md", "report.html",
                "roofline-rx9070xt.svg", "roofline-w7900d-pending.svg", "batching.svg", "prefill.svg"})
            self.assertEqual(json.loads(assets["report.json"].read_text()), report)
            with assets["components.csv"].open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 54)
            for row, component in zip(rows, [c for p in report["points"] for c in p["components"]]):
                self.assertEqual(float(row["t_ms"]), component["t_ms"])
                self.assertEqual(float(row["flops"]), component["flops"])
                self.assertEqual(float(row["bytes"]), component["bytes"])
                self.assertEqual(row["hardware_validated"], "False")
            for name in ("roofline-rx9070xt.svg", "roofline-w7900d-pending.svg", "batching.svg", "prefill.svg"):
                svg = assets[name].read_text()
                ET.fromstring(svg)
                self.assertIn(report["data_sha256"], svg)
                self.assertIn("hardware_validated=false", svg)
                self.assertIn("无实测点", svg)
                self.assertIn('![', assets["report.md"].read_text())
            for p in report["points"]:
                self.assertIn(p["config_id"], assets["roofline-rx9070xt.svg"].read_text())
                self.assertIn(p["config_id"], assets["roofline-w7900d-pending.svg"].read_text())
                self.assertIn(p["config_id"], assets["batching.svg" if p["phase"] == "decode" else "prefill.svg"].read_text())
            roofline = assets["roofline-rx9070xt.svg"].read_text()
            root = ET.fromstring(roofline)
            self.assertEqual(len(root.findall("{http://www.w3.org/2000/svg}g")), 54)
            texts = [node.text for node in root.findall("{http://www.w3.org/2000/svg}text")]
            for number in range(1, 19):
                for component in ("G", "A", "E"):
                    self.assertIn(f"{component}{number}: AI=", roofline)
                    self.assertNotIn(f"{component}{number}", texts)
            self.assertNotIn("合并编号", roofline)
            pending = assets["roofline-w7900d-pending.svg"].read_text()
            self.assertIn("无硬件 Roofline", pending)
            self.assertNotIn("polyline", pending)
            html = assets["report.html"].read_text()
            self.assertEqual(html.count('<img '), 4)
            self.assertEqual(html.count('<table>'), 3)
            self.assertIn("不标作硬件测量误差", html)

    def test_refuses_existing_output_and_symlinks_without_overwriting(self):
        report = run_chapter1(plan())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "existing"
            target.mkdir()
            marker = target / "report.json"
            marker.write_text("keep")
            with self.assertRaises(SimulationError) as raised:
                write_chapter1_report(report, target)
            self.assertEqual(raised.exception.code, "OUTPUT_EXISTS")
            self.assertEqual(marker.read_text(), "keep")
            dangling = root / "dangling"
            dangling.symlink_to(root / "absent")
            with self.assertRaises(SimulationError):
                write_chapter1_report(report, dangling)
            # Real macOS temporary directories often have a symlink ancestor.
            link = root / "ancestor"
            link.symlink_to(target, target_is_directory=True)
            write_chapter1_report(report, link / "fresh")
            self.assertTrue((target / "fresh" / "report.json").is_file())

    def test_edited_report_is_rejected_before_creating_output(self):
        report = run_chapter1(plan())
        report["points"][0]["t_step_ms"] = 0
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "bad"
            with self.assertRaises(SimulationError):
                write_chapter1_report(report, out)
            self.assertFalse(out.exists())

    def test_rendering_a_frozen_report_is_byte_reproducible(self):
        report = run_chapter1(plan())
        with tempfile.TemporaryDirectory() as directory:
            first = write_chapter1_report(report, Path(directory) / "first")
            second = write_chapter1_report(report, Path(directory) / "second")
            for name in first:
                self.assertEqual(first[name].read_bytes(), second[name].read_bytes())

    def test_explicit_alternative_positive_coefficients_remain_finite_and_plotted(self):
        source = plan()
        source["counting"]["elementwise_flops_per_layer_hidden"] = 1e-8
        source["counting"]["elementwise_bytes_per_layer_hidden"] = 4096
        report = run_chapter1(source)
        with tempfile.TemporaryDirectory() as directory:
            assets = write_chapter1_report(report, Path(directory) / "alternative")
            ET.fromstring(assets["roofline-rx9070xt.svg"].read_text())
            pending = assets["roofline-w7900d-pending.svg"].read_text()
            self.assertEqual(pending.count("Elementwise"), 1)
            self.assertEqual(pending.count("elementwise:"), 18)
