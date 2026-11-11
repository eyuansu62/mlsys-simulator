"""Full observed token times are opt-in and never imply memory observations."""

from copy import deepcopy
import unittest

from mlsys_sim import CalibrationSpec, Experiment, RequestSpec, SimulationError
from mlsys_sim.runtime_adapter import normalize_runtime_trace


class ServerRequestEvidenceTests(unittest.TestCase):
    def test_default_contract_stays_small_and_opt_in_preserves_token_times(self):
        exp = Experiment(workload=(RequestSpec('r', 0, 2, 2),),
                         calibration=CalibrationSpec(backend='synthetic'))
        raw = {'schema_version': 1, 'kind': 'runtime_trace',
               'source': {'kind': 'test_fixture', 'reference': 'hand-derived server events'},
               'timing_boundary': 'server_token_ready_v1', 'observed_until_ms': 8,
               'events': [{'event': 'REQUEST_ARRIVAL', 'request_id': 'r', 'timestamp_ms': 0},
                          {'event': 'TOKEN_READY', 'request_id': 'r', 'timestamp_ms': 5,
                           'token_count': 1, 'output_token_index': 1},
                          {'event': 'TOKEN_READY', 'request_id': 'r', 'timestamp_ms': 8,
                           'token_count': 1, 'output_token_index': 2},
                          {'event': 'REQUEST_FINISH', 'request_id': 'r', 'timestamp_ms': 8,
                           'status': 'COMPLETED'}]}
        before = deepcopy(raw)
        old = normalize_runtime_trace(exp, raw, allow_test_fixture=True)
        new = normalize_runtime_trace(exp, raw, allow_test_fixture=True, include_request_evidence=True)
        self.assertEqual(raw, before)
        self.assertNotIn('request_evidence', old)
        evidence = new.pop('request_evidence')
        self.assertEqual(old, new)
        self.assertEqual(evidence['requests'][0]['token_times_ms'], [5, 8])
        self.assertEqual(evidence['requests'][0]['tpot_ms'], 3)
        self.assertEqual(evidence['result']['backend'], 'observed_server_trace')
        self.assertEqual(evidence['result']['memory'], [])
        self.assertNotIn('compute_busy_ms', evidence['result'])
        self.assertEqual(evidence['physical_memory_status'], 'not_observed')

    def test_non_boolean_opt_in_is_rejected(self):
        with self.assertRaises(SimulationError):
            normalize_runtime_trace(Experiment(), {}, include_request_evidence=1)


if __name__ == '__main__':
    unittest.main()
