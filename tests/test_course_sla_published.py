"""Keep the published historical study and its negative conclusions consistent."""

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/course-sla-9070xt-20261111'
FROZEN = ROOT / 'examples/course/9070-sla/frozen-commitment'


class PublishedCourseSlaTests(unittest.TestCase):
    def test_received_commitment_and_all_public_frozen_input_bytes_are_unchanged(self):
        raw = (FROZEN / 'freeze-receipt.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         '8a04caf27814ac43ee7ad21a4a9d783b1b58b24c15f118d31760c96bf0bd171f')
        receipt = json.loads(raw)
        for name, expected in receipt['files_sha256'].items():
            self.assertEqual(hashlib.sha256((FROZEN / name).read_bytes()).hexdigest(), expected, name)
        self.assertTrue(receipt['prediction_artifacts_stored_locally'])
        self.assertEqual(len(json.loads((FROZEN / 'jobs.json').read_text())), 39)

    def test_all_published_cells_retain_samples_and_measured_error_denominator(self):
        study = json.loads((REPORT / 'report.json').read_text())
        rows = [row for cohort in study['cohorts'].values() for row in cohort['runs']]
        self.assertEqual(len(rows), 39)
        self.assertEqual(sum(row['observation']['counts']['completed'] for row in rows), 39000)
        for row in rows:
            self.assertEqual(row['acquisition_verdict'], 'PASS')
            self.assertEqual(row['prediction_error_verdict'], 'FAIL')
            for metric in row['metrics'].values():
                self.assertEqual(metric['measured_sample_count'], 1000)
                expected = abs(metric['predicted'] - metric['measured']) / metric['measured']
                self.assertTrue(math.isclose(metric['relative_error'], expected, rel_tol=1e-12, abs_tol=1e-12))
        with (REPORT / 'cells.csv').open(newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 39)

    def test_negative_sla_conclusions_are_not_replaced_by_best_repeat_or_joint_failure(self):
        summary = json.loads((REPORT / 'summary.json').read_text())
        self.assertEqual(summary['acquisition_counts'], {'PASS': 39})
        self.assertEqual(summary['prediction_error_counts'], {'FAIL': 39})
        self.assertEqual(summary['sla_agreement']['agree'], 29)
        self.assertEqual(summary['sla_agreement']['false_safe_observed_fail'], 10)
        self.assertEqual(summary['cohorts']['baseline']['largest_observed_safe_grid_point'], 4)
        for name in ('short', 'long'):
            self.assertIsNone(summary['cohorts'][name]['largest_observed_safe_grid_point'])
        point = next(item for item in summary['cohorts']['short']['points'] if item['concurrency'] == 8)
        self.assertEqual(point['observed_repetition_verdicts'], ['PASS', 'FAIL', 'PASS'])
        self.assertFalse(point['observed_all_repetitions_pass'])
        self.assertFalse(summary['hardware_validated'])


if __name__ == '__main__':
    unittest.main()
