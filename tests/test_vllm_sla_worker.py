"""CPU checks for formal-ID selection and actual Scheduler timing boundaries."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import Experiment, SimulationError
from mlsys_sim.profiles import profile_identity
from mlsys_sim.vllm_sla_worker import (_captured_json, _study_inputs, formal_server_evidence,
                                       verify_checkpoint)


def evidence():
    origin = 2**60
    prompt_hash = hashlib.sha256(json.dumps([100, 100], separators=(',', ':')).encode()).hexdigest()
    capture = {'source': {'kind': 'test_fixture'}, 'clock_origin_monotonic_ns': origin,
               'identity': profile_identity(Experiment()),
               'metadata': {'canonical_eligible': True, 'fresh_state_verified': True},
               'capture_errors': [], 'capture_closed': True,
               'requests': [{'request_id': 'native', 'input_tokens': 2, 'max_output_tokens': 2,
                             'prompt_token_ids_sha256': prompt_hash}],
               'events': [{'event': 'REQUEST_ARRIVAL', 'request_id': 'warmup', 'timestamp_ms': 0},
                          {'event': 'REQUEST_ARRIVAL', 'request_id': 'native', 'timestamp_ms': 5},
                          {'event': 'TOKEN_READY', 'request_id': 'native', 'timestamp_ms': 10,
                           'token_count': 1, 'output_token_index': 1},
                          {'event': 'TOKEN_READY', 'request_id': 'native', 'timestamp_ms': 13,
                           'token_count': 1, 'output_token_index': 2},
                          {'event': 'REQUEST_FINISH', 'request_id': 'native', 'timestamp_ms': 13,
                           'status': 'COMPLETED'}]}
    feedback = {'timing_boundary': 'runtime_API_feedback_v1', 'formal_end_ns': origin + 14000000,
                'requests': [{'request_id': 'router', 'native_request_id': 'native', 'input_tokens': 2,
                              'output_tokens': 2, 'timeout_ms': 1000, 'prompt_token_id': 100,
                              'submit_before_ns': origin + 4000000, 'submit_after_ns': origin + 6000000}]}
    return capture, feedback


class VllmSlaEvidenceTests(unittest.TestCase):
    def test_selects_formal_ids_rebases_actual_events_and_excludes_warmup(self):
        capture, feedback = evidence()
        before = deepcopy(capture)
        selected = formal_server_evidence(Experiment(), capture, feedback, allow_test_fixture=True)
        self.assertEqual(capture, before)
        self.assertEqual([row['timestamp_ms'] for row in selected['trace']['events']], [0, 5, 8, 8])
        self.assertEqual(selected['trace']['observed_until_ms'], 9)
        row = selected['evidence']['requests'][0]
        self.assertEqual(row['request_id'], 'router')
        self.assertEqual(row['token_times_ms'], [5, 8])
        self.assertEqual(row['input_tokens'], 2)
        self.assertEqual(row['output_tokens_target'], 2)
        self.assertEqual(row['prompt_token_ids_sha256'], capture['requests'][0]['prompt_token_ids_sha256'])
        self.assertFalse(selected['normalization']['hardware_validated'])

    def test_fixture_and_invalid_capture_do_not_become_measured(self):
        capture, feedback = evidence()
        with self.assertRaises(SimulationError):
            formal_server_evidence(Experiment(), capture, feedback)
        for change in ('fresh', 'closed', 'errors', 'coalesced'):
            raw = deepcopy(capture)
            if change == 'fresh':
                raw['metadata']['fresh_state_verified'] = False
            elif change == 'closed':
                raw['capture_closed'] = False
            elif change == 'errors':
                raw['capture_errors'] = [{'error_code': 'CAPTURE_LIMIT'}]
            else:
                raw['events'][2]['token_count'] = 2
            with self.subTest(change=change), self.assertRaises(SimulationError):
                formal_server_evidence(Experiment(), raw, feedback, allow_test_fixture=True)

    def test_prompt_identity_and_actual_enqueue_call_are_checked(self):
        capture, feedback = evidence()
        capture['requests'][0]['prompt_token_ids_sha256'] = '0' * 64
        with self.assertRaisesRegex(SimulationError, 'prompt identity'):
            formal_server_evidence(Experiment(), capture, feedback, allow_test_fixture=True)
        capture, feedback = evidence()
        feedback['requests'][0]['submit_after_ns'] = 2**60 + 4500000
        with self.assertRaisesRegex(SimulationError, 'outside actual'):
            formal_server_evidence(Experiment(), capture, feedback, allow_test_fixture=True)

    def test_mismatched_or_missing_actual_identity_is_not_relabelled(self):
        for change in ('missing', 'model', 'runtime'):
            capture, feedback = evidence()
            if change == 'missing':
                del capture['identity']
            elif change == 'model':
                capture['identity']['model']['name'] = 'different-runtime-model'
            else:
                capture['identity']['runtime']['version'] = 'different-runtime-version'
            with self.subTest(change=change), self.assertRaises(SimulationError):
                formal_server_evidence(Experiment(), capture, feedback, allow_test_fixture=True)

    def test_runner_and_scheduler_fingerprints_use_their_distinct_scopes(self):
        capture, feedback = evidence()
        base = {'scheduler_config': {'max_num_seqs': 8}}
        full = {**base, 'actual_attention_backends': ['fixture'], 'torch_build': 'fixture',
                'sampling_policy': {'temperature': 0}}
        digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        exp = Experiment(calibration=replace(Experiment().calibration, runtime_config_sha256=digest))
        capture['identity'] = profile_identity(exp)
        capture['metadata']['actual_scheduler_config'] = deepcopy(base)
        formal_server_evidence(exp, capture, feedback, allow_test_fixture=True,
                               runtime_execution_config=full)
        capture['metadata']['actual_scheduler_config']['scheduler_config']['max_num_seqs'] = 4
        with self.assertRaisesRegex(SimulationError, 'Scheduler execution'):
            formal_server_evidence(exp, capture, feedback, allow_test_fixture=True,
                                   runtime_execution_config=full)

    def test_inline_inputs_consume_captured_bytes_when_file_changes_after_read(self):
        spec = {'schema_version': 1, 'study_id': 'fixture', 'cohort': 'base', 'concurrency_grid': [1],
                'requests_per_run': 2, 'replications': 1, 'seed': 42, 'think_ms': 0,
                'input_lengths': [2], 'output_tokens': 2, 'timeout_ms': 100, 'max_run_ms': 1000}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp_path, spec_path = root / 'exp.json', root / 'spec.json'
            a = json.dumps({'strategy': {'max_batch_size': 8}}).encode()
            exp_path.write_bytes(a)
            spec_path.write_text(json.dumps(spec))
            original = Path.read_bytes
            reads = []
            def mutate(path):
                data = original(path)
                reads.append(path)
                if path == exp_path:
                    exp_path.write_text(json.dumps({'strategy': {'max_batch_size': 2}}))
                return data
            with patch.object(Path, 'read_bytes', mutate):
                exp, _, consumed, _ = _study_inputs(exp_path, spec_path)
            self.assertEqual(exp.strategy.max_batch_size, 8)
            self.assertEqual(consumed, a)
            self.assertEqual(reads, [exp_path, spec_path])
        with self.assertRaises(SimulationError):
            _captured_json(b'{"x":1,"x":2}')

    def test_actual_checkpoint_files_are_rehashed_not_just_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            path = model / 'model.safetensors'
            path.write_bytes(b'fixed-fixture-weights')
            manifest = [{'file': path.name, 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
            digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            verify_checkpoint(model, manifest, digest)
            path.write_bytes(b'other-fixture-weights')
            with self.assertRaisesRegex(SimulationError, 'bytes differ'):
                verify_checkpoint(model, manifest, digest)


if __name__ == '__main__':
    unittest.main()
