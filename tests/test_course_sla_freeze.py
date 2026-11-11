"""CPU-only checks for remote hash commitment receipt creation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('course_sla_freeze', ROOT / 'scripts/freeze_course_sla_commitment.py')
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CourseSlaFreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        self.code = self.root / 'code'
        self.code.mkdir()
        (self.code / 'worker.py').write_text('# CPU fixture\n')
        template = self.bundle / 'validation-contract.template.json'
        template.write_text(json.dumps({'schema_version': 1, 'prediction_frozen_at': 'pending'}))
        self.manifest = {'prediction_artifacts_stored_locally': True,
                         'prediction_artifacts_sha256': {name: '1' * 64 for name in ('baseline', 'short', 'long')},
                         'canonical_prediction_payload_sha256': {name: '2' * 64 for name in ('baseline', 'short', 'long')},
                         'files_sha256': {template.name: MODULE.sha(template)},
                         'code_files_sha256': {'worker.py': MODULE.sha(self.code / 'worker.py')}}
        self.path = self.bundle / 'manifest.json'
        self.path.write_text(json.dumps(self.manifest))

    def test_receipt_commits_all_cohorts_without_claiming_remote_full_payloads(self):
        expected = MODULE.sha(self.path)
        result = MODULE.freeze_bundle(self.bundle, self.code, expected)
        receipt = json.loads((self.bundle / 'freeze-receipt.json').read_text())
        self.assertFalse(result['measurement_started'])
        self.assertTrue(result['prediction_artifacts_stored_locally'])
        self.assertGreater(receipt['remote_freeze_monotonic_ns'], 0)
        self.assertEqual(receipt['prediction_artifacts_sha256'], self.manifest['prediction_artifacts_sha256'])
        for cohort in ('baseline', 'short', 'long'):
            contract = json.loads((self.bundle / (cohort + '.validation-contract.json')).read_text())
            self.assertEqual(contract['prediction_frozen_at'], receipt['frozen_at'])
            self.assertEqual(contract['prediction_payload_sha256'], '2' * 64)
        self.assertEqual(result['receipt_sha256'], hashlib.sha256((self.bundle / 'freeze-receipt.json').read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            MODULE.freeze_bundle(self.bundle, self.code, expected)

    def test_wrong_manifest_or_modified_source_prevents_freeze(self):
        with self.assertRaises(ValueError):
            MODULE.freeze_bundle(self.bundle, self.code, '0' * 64)
        (self.code / 'worker.py').write_text('# changed fixture\n')
        with self.assertRaises(ValueError):
            MODULE.freeze_bundle(self.bundle, self.code, MODULE.sha(self.path))
        self.assertFalse((self.bundle / 'freeze-receipt.json').exists())

    def test_missing_cohort_commitment_is_not_silently_omitted(self):
        del self.manifest['prediction_artifacts_sha256']['long']
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            MODULE.freeze_bundle(self.bundle, self.code, MODULE.sha(self.path))


if __name__ == '__main__':
    unittest.main()
