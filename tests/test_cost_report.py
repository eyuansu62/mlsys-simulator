"""Report consistency, failure retention and escaped external labels."""

from copy import deepcopy
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from mlsys_sim.cost_calibration import fit_analytical_costs, freeze_cost_split, validate_analytical_costs
from mlsys_sim.cost_report import write_cost_validation_report
from test_cost_calibration import fixture


class TableCells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.current = ""

    def handle_endtag(self, tag):
        if tag == "td":
            self.cells.append(self.current)
            self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current += data


def known_report(*, default_available=True):
    data = fixture()
    bad = deepcopy(next(row for row in data["records"] if row["record_id"] == "heldout2"))
    bad.update(record_id="heldout2-bad", session_id="holdout-other", observed_ms=38)
    data["records"].append(bad)
    for row in data["records"]:
        row["model_id"] = "<script>alert('model')</script>"
        if not default_available:
            row.pop("default_ms", None)
    split = freeze_cost_split(data, training_record_ids=["gpu1", "wall1", "gpu3", "wall3"],
                              holdout_record_ids=["heldout2", "heldout2-bad"], repeat_record_ids=[])
    fit = fit_analytical_costs(data, split, calibration_id="report-fixture")
    return validate_analytical_costs(data, fit)


class CostReportTests(unittest.TestCase):
    def test_two_known_cells_match_json_csv_html_and_svg(self):
        report = known_report()
        with tempfile.TemporaryDirectory(prefix="mlsys-cost-report-") as temp:
            out = Path(temp) / "report"
            write_cost_validation_report(report, out)
            self.assertEqual(json.loads((out / "report.json").read_text()), report)
            with (out / "cells.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(float(row["observed_ms"]) for row in rows), [19, 38])
            self.assertEqual({float(row["predicted_ms"]) for row in rows}, {19})
            self.assertEqual({row["passed"] for row in rows}, {"True", "False"})
            html = (out / "report.html").read_text()
            self.assertIn("主留出判定：FAIL", html)
            parser = TableCells()
            parser.feed(html)
            for row in rows:
                self.assertIn(f"{float(row['observed_ms']):.3f}", parser.cells)
                self.assertIn(f"{float(row['predicted_ms']):.3f}", parser.cells)
            svg = ET.fromstring((out / "holdout-errors.svg").read_text())
            titles = [node.text for node in svg.iter() if node.tag.endswith("title")]
            self.assertTrue(any("+0.00%" in title for title in titles))
            self.assertTrue(any("-50.00%" in title for title in titles))
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn("<script>", html)
            self.assertNotIn("<script>", (out / "holdout-errors.svg").read_text())

    def test_unknown_before_values_remain_blank_not_zero(self):
        report = known_report(default_available=False)
        self.assertIsNone(report["holdout"]["legacy_default"])
        with tempfile.TemporaryDirectory(prefix="mlsys-cost-report-") as temp:
            out = Path(temp) / "report"
            write_cost_validation_report(report, out)
            with (out / "cells.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["before_ms"] for row in rows}, {""})
            self.assertTrue(all(row["predicted_ms"] for row in rows))
            self.assertIn("—", (out / "report.html").read_text())
            self.assertTrue((out / "holdout-errors.svg").is_file())

    def test_failed_coverage_and_rejection_reason_survive_report(self):
        report = known_report()
        report["holdout"]["declared_record_count"] += 1
        report["holdout"]["coverage"] = 2 / 3
        report["holdout"]["rejected"].append({"record_id": "missing-record", "reason": "outside_declared_training_ranges"})
        with tempfile.TemporaryDirectory(prefix="mlsys-cost-report-") as temp:
            out = Path(temp) / "report"
            write_cost_validation_report(report, out)
            saved = json.loads((out / "report.json").read_text())
            self.assertEqual(saved["holdout"]["rejected"], report["holdout"]["rejected"])
            self.assertEqual(saved["verdict"], "FAIL")
            self.assertIn("2/3", (out / "report.html").read_text())

    def test_report_destination_cannot_overwrite_a_frozen_output(self):
        with tempfile.TemporaryDirectory(prefix="mlsys-cost-report-") as temp:
            out = Path(temp) / "report"
            report = known_report()
            write_cost_validation_report(report, out)
            original = (out / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                write_cost_validation_report(report, out)
            self.assertEqual((out / "report.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
