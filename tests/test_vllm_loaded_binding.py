"""New loaded-storage proof cannot be promoted while checkpoint tying is pending."""

from copy import deepcopy
import unittest

from mlsys_sim import SimulationError
from test_vllm_capture import capture, experiment, normalize


class LoadedBindingImportTests(unittest.TestCase):
    def test_completed_capture_needs_prepared_resolved_effective_policy(self):
        exp = experiment()
        original = capture(exp)
        original['metadata']['model_binding'] = {'word_embedding_tying': {
            'verification_phase': 'prepared', 'pending_loaded_storage_proof': False,
            'effective_tie_word_embeddings': True}}
        self.assertTrue(normalize(exp, [original])['canonical_stage_ready'])
        for field, value, code in (
            ('verification_phase', 'construction', 'HARDWARE_METADATA_INCOMPLETE'),
            ('pending_loaded_storage_proof', True, 'HARDWARE_METADATA_INCOMPLETE'),
            ('pending_loaded_storage_proof', 0, 'HARDWARE_METADATA_INCOMPLETE'),
            ('effective_tie_word_embeddings', False, 'CALIBRATION_MISMATCH'),
            ('effective_tie_word_embeddings', 1, 'CALIBRATION_MISMATCH')):
            changed = deepcopy(original)
            changed['metadata']['model_binding']['word_embedding_tying'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SimulationError) as raised:
                normalize(exp, [changed])
            self.assertEqual(raised.exception.code, code)

    def test_pending_only_raw_iteration_remains_diagnostic(self):
        exp = experiment()
        channel = capture(exp)
        channel['records'][0]['status'] = 'UNSUPPORTED'
        channel['metadata']['model_binding'] = {'word_embedding_tying': {
            'verification_phase': 'construction', 'pending_loaded_storage_proof': True,
            'effective_tie_word_embeddings': None}}
        result = normalize(exp, [channel])
        self.assertEqual(result['datasets'], [])
        self.assertFalse(result['canonical_stage_ready'])
        self.assertEqual(result['raw_records'][0]['metadata']['model_binding'], channel['metadata']['model_binding'])


if __name__ == '__main__':
    unittest.main()
