"""Receive a prediction commitment on the measurement host before GPU work.

Full predictions may remain at the producer. This receipt commits to their
raw and canonical hashes, not to remote storage of those full artifacts.
"""

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_bundle(bundle: Path, code_root: Path, expected_manifest_sha256: str) -> dict:
    raw = (bundle / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError('Received manifest differs from its expected commitment')
    manifest = json.loads(raw)
    if manifest.get('prediction_artifacts_stored_locally') is not True:
        raise ValueError('This receiver requires explicit locally archived prediction commitments')
    cohorts = {'baseline', 'short', 'long'}
    if (set(manifest['prediction_artifacts_sha256']) != cohorts
            or set(manifest['canonical_prediction_payload_sha256']) != cohorts):
        raise ValueError('Every declared cohort needs both complete-payload commitments')
    for name in ('prediction_artifacts_sha256', 'canonical_prediction_payload_sha256'):
        for digest in manifest[name].values():
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Prediction commitments must be lowercase SHA256')
    for name, digest in manifest['files_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha(bundle / path) != digest:
            raise ValueError('Received frozen input differs: ' + name)
    for name, digest in manifest['code_files_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha(code_root / path) != digest:
            raise ValueError('Received prediction/collection code differs: ' + name)
    paths = [bundle / 'freeze-receipt.json', *[bundle / (name + '.validation-contract.json') for name in sorted(cohorts)]]
    if any(path.exists() for path in paths):
        raise FileExistsError('Freeze outputs already exist; preserve the previous attempt')
    stamp = datetime.now(timezone.utc).isoformat()
    files = dict(manifest['files_sha256'])
    files['manifest.json'] = hashlib.sha256(raw).hexdigest()
    template = json.loads((bundle / 'validation-contract.template.json').read_text())
    for cohort in sorted(cohorts):
        contract = {**template, 'prediction_frozen_at': stamp,
                    'prediction_payload_sha256': manifest['canonical_prediction_payload_sha256'][cohort]}
        path = bundle / (cohort + '.validation-contract.json')
        with path.open('x') as stream:
            json.dump(contract, stream, indent=2, allow_nan=False)
            stream.write('\n')
        path.chmod(0o444)
        files[path.name] = sha(path)
    receipt = {**manifest, 'kind': 'course_sla_prediction_freeze_receipt', 'frozen_at': stamp,
               'files_sha256': files, 'remote_freeze_monotonic_ns': time.perf_counter_ns(),
               'measurement_started': False}
    boot = Path('/proc/sys/kernel/random/boot_id')
    receipt['boot_id_sha256'] = sha(boot) if boot.is_file() else None
    path = bundle / 'freeze-receipt.json'
    with path.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o444)
    return {'receipt_sha256': sha(path), 'frozen_at': stamp, 'measurement_started': False,
            'prediction_artifacts_stored_locally': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_bundle(args.bundle, args.code_root, args.expected_manifest_sha256)))


if __name__ == '__main__':
    main()
