"""Opt-in reported-wall Profile CLI, phase prediction and comparison contracts."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim.comparison import comparison_contract, compare_runs
from mlsys_sim.serialization import experiment_dict
from mlsys_sim.step_profile import freeze_stationarity_policy
from test_step_profile import fixture


ROOT = Path(__file__).resolve().parents[1]


class StepProfileCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def run_cli(self, *args, cost=False):
        return subprocess.run([sys.executable, '-m', 'mlsys_sim.cost_cli' if cost else 'mlsys_sim',
                               *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=30)

    def test_calibration_and_relative_path_phase_prediction(self):
        exp, data = fixture()
        dataset = self.write('dataset.json', data)
        policy = self.write('policy.json', freeze_stationarity_policy())
        out = self.root / 'fit'
        result = self.run_cli('calibrate', '--dataset', dataset, '--profile-version', 'reported-v1',
            '--stationarity-policy', policy, '--allow-test-fixture', '--out', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        profile = json.loads((out / 'profile.json').read_text())
        self.assertEqual(profile['notes']['reported_step_profile']['stationarity']['verdict'], 'PASS')
        exp = replace(exp, calibration=replace(exp.calibration, profile_path='fit/profile.json',
                                               profile_version='reported-v1'))
        experiment = self.write('experiment.json', experiment_dict(exp))
        prediction = self.root / 'prediction'
        result = self.run_cli('predict', '--experiment', experiment, '--phase', 'prefill',
            '--query-tokens', '6', '--context-lengths', '0', '--out', prediction, cost=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads((prediction / 'prediction.json').read_text())
        self.assertEqual(value['estimate']['duration_ms'], 13)
        manifest = json.loads((prediction / 'manifest.json').read_text())
        self.assertEqual(manifest['input_bytes_sha256']['profile'],
                         hashlib.sha256((out / 'profile.json').read_bytes()).hexdigest())

    def test_nonstationary_training_preserves_rejection_without_profile(self):
        _, data = fixture()
        names = set(data['notes']['ordered_series'][0]['record_ids'][:40])
        for row in data['records']:
            if row['record_id'] in names:
                row['duration_ms'] *= 2
        dataset = self.write('dataset.json', data)
        policy = self.write('policy.json', freeze_stationarity_policy())
        out = self.root / 'rejected'
        result = self.run_cli('calibrate', '--dataset', dataset, '--profile-version', 'rejected-v1',
            '--stationarity-policy', policy, '--allow-test-fixture', '--out', out)
        self.assertEqual(result.returncode, 3, result.stderr)
        rejected = json.loads((out / 'profile-rejected.json').read_text())
        self.assertNotIn('samples', rejected)
        self.assertEqual(json.loads((out / 'result.json').read_text())['status'], 'REJECTED')
        self.assertFalse((out / 'profile.json').exists())
        self.assertTrue((out / 'dataset.json').exists())

    def test_reported_wall_comparison_contract_is_not_relabelled(self):
        exp, _ = fixture()
        contract = comparison_contract(exp)
        self.assertEqual(contract['timing_boundary'], 'reported_l2_step_hook_wall_v1')
        self.assertEqual(contract['identity']['runtime']['timing_boundary'], contract['timing_boundary'])

    def test_compare_keeps_full_memory_policy_in_new_boundary_contract(self):
        from test_comparison import prediction
        exp, _ = fixture()
        changed = replace(exp, strategy=replace(exp.strategy,
            memory=replace(exp.strategy.memory, kv_pool_bytes=8 * 1024**2)))
        first, second = comparison_contract(exp), comparison_contract(changed)
        self.assertNotEqual(first, second)
        directory = self.root / 'predicted'
        directory.mkdir()
        (directory / 'result.json').write_text(json.dumps(prediction()))
        (directory / 'manifest.json').write_text(json.dumps({'backend': 'simulated', 'comparison_contract': first}))
        observed = self.write('observation.json', {'schema_version': 1, 'kind': 'measured_run',
            'comparison_contract': second, 'source': {'kind': 'test_fixture', 'reference': 'synthetic'},
            'metrics': {'request_count': 2}})
        from mlsys_sim import SimulationError
        with self.assertRaises(SimulationError):
            compare_runs(directory, observed, allow_test_fixture=True)

    def test_policy_is_not_accepted_for_communication_fit(self):
        _, data = fixture()
        dataset = self.write('dataset.json', data)
        policy = self.write('policy.json', freeze_stationarity_policy())
        result = self.run_cli('calibrate', '--dataset', dataset, '--profile-version', 'bad',
            '--communication', '--stationarity-policy', policy, '--out', self.root / 'bad')
        self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
