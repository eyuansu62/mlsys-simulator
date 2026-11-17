"""Explicit whole-step CLI selection, rejection persistence and exported core use."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim import CalibrationSpec, Experiment, fit_step_costs, freeze_step_fit_spec
from mlsys_sim.analytical_cost import analytical_cost_identity, nominal_cost_features
from mlsys_sim.config import load_experiment
from mlsys_sim.performance import AnalyticalModel
from mlsys_sim.serialization import experiment_dict
from test_step_cost_calibration import fixture, split


class StepCostCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'mlsys_sim.cost_cli', *map(str, args)],
                              capture_output=True, text=True, timeout=30)

    def inputs(self, *, unstable=False, export=False):
        data = fixture()
        if unstable:
            for row in data['records']:
                if row['session_id'] == 'train-b':
                    row['observed_ms'] *= 2
        if export:
            exp = Experiment(calibration=CalibrationSpec(runtime_name='fixture', runtime_version='1',
                runtime_config_sha256='b' * 64, analytical_cost_timing_boundary='fixture_whole_step_v1',
                prefill_execution_mode='eager', decode_execution_mode='eager',
                prefill_attention_backend='fixture-backend', decode_attention_backend='fixture-backend'))
            mapping = experiment_dict(exp)
            mapping['strategy']['model']['num_layers'] = 2
            data['experiments'] = {'fixture-model': mapping}
            from mlsys_sim.config import experiment_from_dict
            identity = analytical_cost_identity(experiment_from_dict(mapping))
            for row in data['records']:
                row['domain']['identity_sha256'] = identity
        membership = split(data)
        data['recommended_split'] = {k: deepcopy(membership[k]) for k in
            ('training_record_ids', 'holdout_record_ids', 'repeat_record_ids')}
        path = self.root / 'dataset.json'
        path.write_text(json.dumps(data))
        return path

    def freeze(self, dataset):
        out = self.root / 'frozen'
        result = self.run_cli('freeze', '--dataset', dataset, '--recommended', '--step-affine', '--out', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((out / 'fit-spec.json').exists())
        policy = json.loads((out / 'fit-spec.json').read_text())
        self.assertEqual(policy['stability']['max_training_process_ratio'], 1.25)
        return out

    def fit(self, dataset, frozen, *extra):
        out = self.root / 'fit'
        result = self.run_cli('fit', '--dataset', dataset, '--split', frozen / 'split.json',
            '--fit-spec', frozen / 'fit-spec.json', '--calibration-id', 'whole-step-fixture',
            '--allow-test-fixture', '--out', out, *extra)
        return result, out

    def test_end_to_end_fit_validate_and_exported_python_prediction(self):
        dataset = self.inputs(export=True)
        result, out = self.fit(dataset, self.freeze(dataset))
        self.assertEqual(result.returncode, 0, result.stderr)
        validation = self.run_cli('validate', '--dataset', dataset, '--fit', out / 'fit.json',
                                  '--out', self.root / 'validation')
        self.assertEqual(validation.returncode, 0, validation.stderr)
        self.assertEqual(json.loads((self.root / 'validation/report.json').read_text())['verdict'], 'PASS')
        index = json.loads((out / 'models.json').read_text())
        exp = load_experiment(out / index['fixture-model']['experiment'])
        features = nominal_cost_features(exp, 'decode', [1, 1], [100, 100])
        prediction = AnalyticalModel(exp).estimate_step('decode', [1, 1], [100, 100])
        self.assertAlmostEqual(prediction.duration_ms,
                               2 * (features['nonattention_ms'] + features['attention_ms']) + 4)
        self.assertTrue(callable(fit_step_costs) and callable(freeze_step_fit_spec))

    def test_unstable_fit_saves_all_diagnostics_without_executable_calibration(self):
        dataset = self.inputs(unstable=True)
        result, out = self.fit(dataset, self.freeze(dataset))
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'REJECTED')
        rejected = json.loads((out / 'fit-rejected.json').read_text())
        self.assertEqual(len(rejected['diagnostics'][0]['training_cells']), 4)
        self.assertNotIn('cost_specs', rejected)
        self.assertFalse((out / 'fit.json').exists())
        self.assertFalse((out / 'calibrations').exists())
        self.assertTrue((out / 'manifest.json').exists())

    def test_step_fit_is_never_silently_selected_by_default_component_fit(self):
        dataset = self.inputs()
        frozen = self.freeze(dataset)
        result = self.run_cli('fit', '--dataset', dataset, '--split', frozen / 'split.json',
            '--calibration-id', 'not-implicit', '--allow-test-fixture', '--out', self.root / 'default-fit')
        self.assertEqual(result.returncode, 2)
        self.assertIn('component', json.loads(result.stderr)['message'])

    def test_stability_flag_without_explicit_family_is_rejected(self):
        dataset = self.inputs()
        result = self.run_cli('freeze', '--dataset', dataset, '--recommended',
            '--max-training-process-ratio', '1.25', '--out', self.root / 'bad-freeze')
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / 'bad-freeze').exists())


if __name__ == '__main__':
    unittest.main()
