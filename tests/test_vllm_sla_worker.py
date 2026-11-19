"""CPU checks for formal-ID selection and actual Scheduler timing boundaries."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlsys_sim import Experiment, SimulationError
from mlsys_sim.profiles import profile_identity
from mlsys_sim.vllm_sla_worker import (_VllmBackend, _captured_json, _study_inputs, formal_server_evidence,
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


class FakeEngine:
    """Pinned engine semantics: external IDs are translated unless internal=True."""
    def __init__(self, *, clock=None, finish=True):
        self.external_to_internal = {}
        self.pending = {}
        self.aborts = []
        self.clock, self.finish = clock, finish
        self.sequence = 0

    def add_request(self, external, prompt, parameters):
        native = f'{external}-native-{self.sequence}'
        self.sequence += 1
        self.external_to_internal[external] = native
        self.pending[native] = (external, prompt, parameters)
        return native

    def abort_request(self, request_ids, *, internal=False):
        self.aborts.append((list(request_ids), internal))
        selected = request_ids if internal else [self.external_to_internal.get(name) for name in request_ids]
        for native in selected:
            row = self.pending.pop(native, None)
            if row is not None:
                self.external_to_internal.pop(row[0], None)

    def has_unfinished_requests(self):
        return bool(self.pending)

    def step(self):
        if self.clock is not None:
            self.clock.now += 2_000_000
        outputs = [SimpleNamespace(request_id=external, finished=self.finish,
            outputs=[SimpleNamespace(token_ids=[7, 8] if self.finish else [])])
            for external, _, _ in self.pending.values()]
        if self.finish:
            self.pending.clear()
            self.external_to_internal.clear()
        return outputs


class VllmSlaBackendTests(unittest.TestCase):
    def test_internal_abort_does_not_translate_a_colliding_external_id(self):
        engine = FakeEngine()
        backend = _VllmBackend(engine, None)
        request = {'request_id': 'q0', 'prompt_token_id': 100, 'input_tokens': 4}
        target = backend.add_request(request)
        other = backend.add_request({**request, 'request_id': target})
        self.assertNotEqual(target, request['request_id'])
        self.assertNotEqual(other, target)
        backend.abort_requests([target])
        self.assertEqual(engine.aborts, [([target], True)])
        self.assertNotIn(target, engine.pending)
        self.assertEqual(set(engine.pending), {other})
        self.assertTrue(backend.has_unfinished_requests())
        self.assertEqual(backend.step(), [{'native_request_id': other, 'finished': True,
                                         'output_tokens_actual': 2}])
        self.assertFalse(backend.has_unfinished_requests())

    def test_collector_timeout_cancels_native_request_and_drains_backend(self):
        from mlsys_sim.closed_loop_capture import collect_closed_loop
        clock = SimpleNamespace(now=0)
        engine = FakeEngine(clock=clock, finish=False)
        backend = _VllmBackend(engine, None)
        spec = {'schema_version': 1, 'study_id': 'native-abort', 'cohort': 'fixture',
            'concurrency_grid': [1], 'requests_per_run': 1, 'replications': 1, 'seed': 1,
            'think_ms': 0, 'input_lengths': [4], 'output_tokens': 2, 'timeout_ms': 1,
            'max_run_ms': 100, 'prompt_token_id': 100}
        result = collect_closed_loop(backend, spec, 1, clock_ns=lambda: clock.now)
        row = result['requests'][0]
        self.assertNotEqual(row['request_id'], row['native_request_id'])
        self.assertEqual(engine.aborts, [([row['native_request_id']], True)])
        self.assertEqual(row['status'], 'TIMED_OUT')
        self.assertEqual(result['timed_out_count'], 1)
        self.assertTrue(result['resources_drained'])
        self.assertFalse(engine.pending)

    def test_ordinary_completion_keeps_output_mapping_and_never_aborts(self):
        engine = FakeEngine()
        parameters = object()
        backend = _VllmBackend(engine, parameters)
        native = backend.add_request({'request_id': 'router', 'prompt_token_id': 100, 'input_tokens': 3})
        self.assertEqual(engine.pending[native], ('router', {'prompt_token_ids': [100, 100, 100]}, parameters))
        self.assertEqual(backend.step(), [{'native_request_id': native, 'finished': True,
                                         'output_tokens_actual': 2}])
        self.assertFalse(backend.has_unfinished_requests())
        self.assertFalse(engine.aborts)


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
