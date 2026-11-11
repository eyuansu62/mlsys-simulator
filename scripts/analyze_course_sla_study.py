"""Reveal committed CPU predictions and validate private real closed-loop cohorts.

Run after all GPU sessions have ended. This script performs CPU-only analysis;
it never replays measured arrivals into the predictor or refits its parameters.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def read_json(path, expected_sha256=None):
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Evidence bytes differ from their commitment: ' + path.name)
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate evidence JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def analyze(root: Path, expected_receipt_sha256: str, out: Path) -> dict:
    from mlsys_sim import experiment_from_dict
    from mlsys_sim.load_validation import predict_closed_loop, validate_closed_loop
    from mlsys_sim.workload import canonical_sha256

    frozen = root / 'frozen-commitment'
    receipt_path = frozen / 'freeze-receipt.json'
    receipt = read_json(receipt_path, expected_receipt_sha256)
    inputs = {name: read_json(frozen / name, digest) for name, digest in receipt['files_sha256'].items()}
    module_file = sys.modules['mlsys_sim'].__file__
    if module_file is None:
        raise ValueError('Installed simulator source location is required')
    for name, digest in receipt['code_files_sha256'].items():
        if sha(Path(module_file).parent / name) != digest:
            raise ValueError('Frozen collection/prediction source changed: ' + name)
    state = read_json(root / 'formal-status.json')
    jobs = inputs['jobs.json']
    if state['status'] != 'COMPLETED' or len(state['completed_jobs']) != len(jobs):
        raise ValueError('Every declared GPU cell must be attempted before analysis')
    if state['receipt_sha256'] != expected_receipt_sha256:
        raise ValueError('Controller did not use the same freeze receipt')
    out.mkdir(parents=True, exist_ok=False)
    experiment = experiment_from_dict(inputs['experiment.json'])
    results = {}
    sessions = set()
    attempts = []
    for cohort in ('baseline', 'short', 'long'):
        spec = inputs[cohort + '.spec.json']
        # This is a cryptographic reveal: identical frozen code and inputs must
        # reproduce both commitments received remotely before GPU measurement.
        prediction = predict_closed_loop(experiment, spec)
        canonical = canonical_sha256(prediction)
        raw = (json.dumps(prediction, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if canonical != receipt['canonical_prediction_payload_sha256'][cohort]:
            raise ValueError('Canonical prediction reveal failed: ' + cohort)
        if hashlib.sha256(raw).hexdigest() != receipt['prediction_artifacts_sha256'][cohort]:
            raise ValueError('Raw prediction reveal failed: ' + cohort)
        observed = {'schema_version': 1, 'study_id': spec['study_id'], 'cohort': cohort,
                    'spec_sha256': prediction['spec_sha256'], 'identity_sha256': prediction['identity_sha256'],
                    'identity': prediction['identity'], 'runs': []}
        for job in (item for item in state['completed_jobs'] if item['cohort'] == cohort):
            folder = root / 'formal' / job['label']
            attempt = {key: job.get(key) for key in ('cohort', 'concurrency', 'replication', 'returncode',
                                                    'worker_status', 'error_code', 'observed_run_sha256')}
            attempts.append(attempt)
            if not (folder / 'observed-run.json').is_file():
                continue
            summary = read_json(folder / 'summary.json')
            feedback = read_json(folder / 'api-feedback.json')
            if summary['observed_run_sha256'] != job['observed_run_sha256']:
                raise ValueError('Observed request evidence bytes changed')
            if summary['source_sha256'] != receipt['code_files_sha256']['vllm_sla_worker.py']:
                raise ValueError('Worker source differs from the frozen collection code')
            if canonical_sha256(summary['identity']) != prediction['identity_sha256']:
                raise ValueError('Actual guarded Runtime identity differs from frozen prediction')
            if summary['runtime_config_sha256'] != receipt['runtime_config_sha256']:
                raise ValueError('Actual Runtime execution fingerprint differs from freeze')
            if summary['session_id'] in sessions:
                raise ValueError('Distinct fresh Runtime sessions are required for every cell')
            sessions.add(summary['session_id'])
            if feedback['formal_start_ns'] <= receipt['remote_freeze_monotonic_ns']:
                raise ValueError('Formal monotonic start does not follow remote freeze')
            if summary['freeze_receipt'] != receipt:
                raise ValueError('Worker did not bind this exact receipt')
            evidence = read_json(folder / 'observed-run.json', job['observed_run_sha256'])
            observed['runs'].append({
                'concurrency': job['concurrency'], 'replication': job['replication'],
                'policy_sha256': summary['policy_sha256'],
                'request_stream_sha256': summary['request_stream_sha256'],
                'source': summary['source'], 'session_id': summary['session_id'],
                'collection_started_at': summary['collection_started_at'],
                'collection_ended_at': summary['collection_ended_at'],
                'timing_boundary': summary['timing_boundary'],
                'resources_drained': feedback['resources_drained'],
                'offered_count': feedback['offered_count'], 'not_offered_count': feedback['not_offered_count'],
                'result': evidence['result'], 'requests': evidence['requests']})
        contract = inputs[cohort + '.validation-contract.json']
        validation = validate_closed_loop(prediction, observed, contract)
        write(out / (cohort + '.validation.json'), validation)
        results[cohort] = validation
    selection = inputs['selection.json']
    summary = {'schema_version': 1, 'study_id': receipt['study_id'], 'cohorts': results,
               'selection': selection, 'attempts': attempts, 'fresh_runtime_sessions': len(sessions),
               'freeze': {key: receipt[key] for key in ('frozen_at', 'prediction_artifacts_sha256',
                   'canonical_prediction_payload_sha256', 'local_manifest_sha256', 'runtime_config_sha256',
                   'scheduler_config_sha256', 'checkpoint_manifest_sha256', 'code_files_sha256')},
               'receipt_sha256': expected_receipt_sha256, 'prediction_reveal_verified': True,
               'experiment': asdict(experiment), 'hardware_validated': False}
    write(out / 'study-analysis.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.expected_receipt_sha256, args.out)
    print(json.dumps({'prediction_reveal_verified': True, 'cohorts': list(result['cohorts']),
                      'fresh_runtime_sessions': result['fresh_runtime_sessions'],
                      'hardware_validated': False}))


if __name__ == '__main__':
    main()
