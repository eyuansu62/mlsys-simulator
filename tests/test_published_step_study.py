"""Replay the measured controlled fit and all frozen forward evidence on CPU."""

from dataclasses import replace
from datetime import datetime
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import unittest

from mlsys_sim import AnalyticalModel, load_experiment
from mlsys_sim.analytical_cost import analytical_cost_identity
from mlsys_sim.cost_calibration import cost_data_sha256, validate_analytical_costs
from mlsys_sim.schema import SimulationError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'examples/runtime_step_20261117'


def compressed(name):
    return json.loads(gzip.decompress((DATA / name).read_bytes()))


class PublishedStepStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = compressed('dataset.json.gz')
        cls.fit = compressed('fit.json.gz')
        cls.inputs = compressed('frozen/inputs.json.gz')
        cls.table = compressed('frozen/predictions.json.gz')['predictions']
        cls.report = compressed('forward-report.json.gz')
        cls.experiment = load_experiment(DATA / 'experiments/model-0.json')

    def test_published_artifact_manifests_match_checkout_bytes(self):
        for directory in (DATA, ROOT / 'docs/runtime-step-20261117'):
            manifest = json.loads((directory / 'sha256.json').read_text())
            for name, expected in manifest.items():
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected, name)

    def test_fit_replays_only_its_real_training_members_and_internal_validation(self):
        result = validate_analytical_costs(self.dataset, self.fit)
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(len(self.dataset['records']), 1824)
        self.assertEqual(len(self.fit['split']['training_record_ids']), 912)
        self.assertEqual(result['holdout']['declared_record_count'], 456)
        self.assertEqual(result['repeats']['declared_record_count'], 456)
        selected = {row['record_id']: row for row in self.dataset['records']}
        sessions = {selected[name]['session_id'] for name in self.fit['split']['training_record_ids']}
        self.assertEqual(sessions, {'train-a', 'train-b'})
        self.assertEqual(self.fit['algorithm'], 'whole_step_nonnegative_affine_session_medians_v1')
        for cost in self.fit['cost_specs']:
            self.assertEqual(cost['nonattention_scale'], cost['attention_scale'])
            self.assertEqual(cost['attention_fixed_us_per_layer'], 0)
        self.assertFalse(self.fit['hardware_validated'])

    def test_actual_core_matches_every_one_of308_frozen_predictions(self):
        model = AnalyticalModel(self.experiment)
        legacy = AnalyticalModel(replace(self.experiment, calibration=replace(
            self.experiment.calibration, analytical_costs=(), profile_version='uncalibrated-v1')))
        self.assertEqual(len(self.table), 308)
        self.assertEqual(analytical_cost_identity(self.experiment), self.inputs['cost_identity_sha256'])
        for row in self.table:
            q, c = row['query_tokens'], row['context_lengths_representative']
            self.assertEqual(sum(c), row['context_sum'])
            self.assertAlmostEqual(model.estimate_step(row['phase'], q, c).duration_ms,
                                   row['calibrated_ms'], places=10)
            self.assertAlmostEqual(legacy.estimate_step(row['phase'], q, c).duration_ms,
                                   row['legacy_default_ms'], places=10)

    def test_all24cells_and2736_steps_recalculate_without_refitting(self):
        lookup = {(r['phase'], r['shape'], r['context_sum']): r for r in self.table}
        helper_spec = importlib.util.spec_from_file_location('published_step_contexts', ROOT / 'scripts/import_cost_trace.py')
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        errors, total = [], 0
        self.assertEqual(len(self.report['cells']), 24)
        for cell in self.report['cells']:
            schedules, values, predictions = {}, [], []
            self.assertEqual(len(cell['steps']), 100 if cell['phase'] == 'prefill' else 128)
            for step in cell['steps']:
                raw = step['raw_step']
                self.assertEqual(step['observed_ms'], raw['step_ms'])
                self.assertEqual(step['query_tokens'], raw['num_scheduled'])
                contexts = step['context_lengths']
                if cell['phase'] == 'decode':
                    generation = raw['generate_index']
                    if raw.get('all_steps_num_scheduled') is not None:
                        schedules[generation] = helper.reconstruct_contexts(
                            raw['all_steps_num_scheduled'], len(step['query_tokens']), prompt=1024)
                    self.assertEqual(contexts, schedules[generation][raw['step_index']])
                self.assertEqual(min(contexts), raw['context_min'])
                self.assertEqual(max(contexts), raw['context_max'])
                expected = lookup[cell['phase'], cell['shape'], sum(contexts)]
                self.assertEqual(step['calibrated_ms'], expected['calibrated_ms'])
                values.append(step['observed_ms'])
                predictions.append(expected['calibrated_ms'])
                total += 1
            actual = statistics.median(values)
            predicted = statistics.median(predictions)
            error = abs(predicted - actual) / actual
            self.assertEqual(actual, cell['observed_median_ms'])
            self.assertEqual(predicted, cell['calibrated_median_ms'])
            self.assertEqual(error, cell['calibrated_relative_error'])
            self.assertLessEqual(error, .3)
            errors.append(error)
        self.assertEqual(total, 2736)
        self.assertEqual(statistics.mean(errors), self.report['calibrated_cell_mape'])
        self.assertEqual(self.report['accuracy_pass_cells'], 24)
        self.assertTrue(self.report['scoped_hardware_measurement_performed'])
        self.assertFalse(self.report['refitted'])
        self.assertFalse(self.report['hardware_validated'])
        for group in self.report['groups'].values():
            self.assertEqual(group['accuracy_pass_cells'], 12)
            self.assertEqual(group['covered_steps'], 1368)

    def test_measurement_host_commitment_predates_all_new_sessions(self):
        commitment = json.loads((DATA / 'frozen/commitment.json').read_text())
        receipt = json.loads((DATA / 'receipts/freeze-receipt.json').read_text())
        progress = json.loads((DATA / 'receipts/progress.json').read_text())
        self.assertEqual(commitment, receipt['prediction_commitment'])
        self.assertEqual(self.fit['fit_sha256'], commitment['fit_sha256'])
        for name in ['inputs.json', 'predictions.json']:
            raw = gzip.decompress((DATA / 'frozen' / (name + '.gz')).read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), commitment['files'][name]['raw_sha256'])
            self.assertEqual(cost_data_sha256(json.loads(raw)),
                             commitment['files'][name]['canonical_payload_sha256'])
        self.assertEqual(hashlib.sha256((DATA / 'receipts/freeze-receipt.json').read_bytes()).hexdigest(),
                         progress['receipt_sha256'])
        self.assertEqual(progress['status'], 'completed')
        self.assertEqual(progress['completed_sessions'], 3)
        previous = receipt['committed_monotonic_ns']
        for attempt in progress['attempts']:
            self.assertLess(previous, attempt['started_monotonic_ns'])
            self.assertLess(datetime.fromisoformat(receipt['committed_at_utc']),
                            datetime.fromisoformat(attempt['started_at_utc']))
            previous = attempt['finished_monotonic_ns']
            self.assertEqual(attempt['status'], 'completed')
        self.assertEqual(self.inputs['expected_controls']['process_environment']['GPU_MAX_HW_QUEUES'], '1')

    def test_uncontrolled_or_different_execution_domain_cannot_borrow_costs(self):
        variants = [replace(self.experiment, calibration=replace(self.experiment.calibration,
                        runtime_config_sha256='0' * 64)),
                    replace(self.experiment, calibration=replace(self.experiment.calibration,
                        decode_execution_mode='eager')),
                    replace(self.experiment, strategy=replace(self.experiment.strategy, prefix_cache=True))]
        for exp in variants:
            with self.assertRaises(SimulationError) as caught:
                AnalyticalModel(exp).estimate_step('decode', [1, 1], [1024, 1024])
            self.assertEqual(caught.exception.code, 'CALIBRATION_MISMATCH')


if __name__ == '__main__':
    unittest.main()
