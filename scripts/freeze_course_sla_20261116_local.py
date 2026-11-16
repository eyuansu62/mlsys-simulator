"""Local (CPU-only) freeze for study course9070sla20261116.

Generates the new low-concurrency + long-input blind holdout bundle under
examples/course/9070-sla-20261116/frozen-commitment/ and commits to the raw
and canonical SHA256 of each cohort's complete prediction payload.

This is the LOCAL half of the freeze. Host-side fields (model checkpoint
manifest, actual vLLM execution config, host design contract) are marked
REMOTE_PENDING here and are completed on the 9070XT by the host-side producer
before scripts/freeze_course_sla_commitment.py issues the receipt. Nothing
host-side is fabricated locally. The prediction commitments (the core of the
blind holdout) are fully local and deterministic: reveal re-runs
predict_closed_loop on the frozen experiment + spec and must reproduce both
SHAs (see scripts/analyze_course_sla_study.py:70-75 for the authoritative
formula used below).
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('/mnt/luyuzhou/hello-mlsys/mlsys-simulator')
sys.path.insert(0, str(REPO))
from mlsys_sim import load_experiment                                  # noqa: E402
from mlsys_sim.load_validation import predict_closed_loop              # noqa: E402
from mlsys_sim.workload import canonical_sha256                        # noqa: E402

STUDY = 'course9070sla20261116'
GRID = [1, 2, 4]                 # NEW pre-registered grid; disjoint from consumed 11-11 {8,16,32}
SEED = 20261116
SRC = REPO / 'examples/course/9070-sla/frozen-commitment'
DST = REPO / 'examples/course/9070-sla-20261116/frozen-commitment'
ARCHIVE = Path('/home/luyzh/.claude/projects/-mnt-luyuzhou-hello-mlsys-mlsys-simulator/'
               'b965358a-4b0d-430a-9c1a-532933163ab0/scratch/step2-20261116-payloads')

COHORTS = {
    'baseline': [256],
    'short':    [64, 128],
    'long':     [512, 1024],
}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_path(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def make_spec(cohort: str, lengths: list[int]) -> dict:
    return {
        'schema_version': 1,
        'study_id': STUDY,
        'concurrency_grid': list(GRID),
        'requests_per_run': 1000,
        'replications': 3,
        'seed': SEED,
        'think_ms': 0,
        'output_tokens': 16,
        'timeout_ms': 10000,
        'max_run_ms': 600000,
        'prompt_token_id': 100,
        'cohort': cohort,
        'input_lengths': list(lengths),
    }


def raw_payload_bytes(prediction: dict) -> bytes:
    # Authoritative reveal encoding (analyze_course_sla_study.py:71):
    # canonical compact sorted JSON + a single trailing newline.
    return (json.dumps(prediction, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def sla_verdict(run: dict) -> bool:
    lat = run['result']['latency_ms']
    thr = run['result']['throughput']
    return (lat['end_to_end']['p99'] <= 500
            and lat['ttft']['p99'] <= 500
            and lat['tpot']['p99'] <= 50
            and thr['output_tokens_per_s'] >= 20)


def main() -> None:
    experiment_path = SRC / 'experiment.json'
    experiment = load_experiment(str(experiment_path))

    DST.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    # Copy the reused frozen experiment + the identical validation template.
    (DST / 'experiment.json').write_bytes(experiment_path.read_bytes())
    (DST / 'validation-contract.template.json').write_bytes(
        (SRC / 'validation-contract.template.json').read_bytes())

    predictions = {}
    raw_sha = {}
    canonical_sha = {}
    for cohort, lengths in COHORTS.items():
        spec = make_spec(cohort, lengths)
        (DST / f'{cohort}.spec.json').write_text(
            json.dumps(spec, indent=2, allow_nan=False) + '\n')
        prediction = predict_closed_loop(experiment, spec)
        raw = raw_payload_bytes(prediction)
        raw_sha[cohort] = sha_bytes(raw)
        canonical_sha[cohort] = canonical_sha256(prediction)
        (ARCHIVE / f'{cohort}.prediction-payload.json').write_bytes(raw)
        predictions[cohort] = prediction
        print(f'[{cohort}] runs={len(prediction["runs"])} status={prediction["status"]} '
              f'raw={raw_sha[cohort][:16]}… canonical={canonical_sha[cohort][:16]}…')

    # predictions-summary.json (compact per-run digest, mirrors 11-11).
    summary = {}
    for cohort in COHORTS:
        pred = predictions[cohort]
        summary[cohort] = {
            'spec_sha256': pred['spec_sha256'],
            'identity_sha256': pred['identity_sha256'],
            'runs': [{
                'concurrency': run['concurrency'],
                'replication': run['replication'],
                'policy_sha256': run['policy_sha256'],
                'request_stream_sha256': run['request_stream_sha256'],
                'status': run['status'],
                'latency_ms': run['result']['latency_ms'],
                'elapsed_ms': run['elapsed_ms'],
                'throughput': run['result']['throughput'],
                'counts': run['result']['counts'],
            } for run in pred['runs']],
        }
    (DST / 'predictions-summary.json').write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n')

    # selection.json: pre-registered blind grid (record only, no selection).
    base_runs = predictions['baseline']['runs']
    baseline_verdicts = []
    for c in GRID:
        reps = [r for r in base_runs if r['concurrency'] == c]
        reps.sort(key=lambda r: r['replication'])
        verdicts = ['PASS' if sla_verdict(r) else 'FAIL' for r in reps]
        baseline_verdicts.append({
            'concurrency': c,
            'predicted_joint_verdict': 'PASS' if all(v == 'PASS' for v in verdicts) else 'FAIL',
            'replication_verdicts': verdicts,
        })
    selection = {
        'rule': ('pre-registered low-concurrency blind grid {1,2,4}; no cap search and no '
                 'selection from observed data (design doc sections 2 and 6)'),
        'pre_registered_grid': list(GRID),
        'grid_disjoint_from_consumed_1111_holdout': True,
        'consumed_1111_grid': [8, 16, 32],
        'selection_from_prediction': False,
        'baseline_predicted_verdicts_record_only': baseline_verdicts,
        'no_observed_data_used': True,
    }
    (DST / 'selection.json').write_text(
        json.dumps(selection, indent=2, allow_nan=False) + '\n')

    # jobs.json: deterministic execution order for the 27 runtimes.
    jobs = [{'cohort': cohort, 'concurrency': c, 'replication': rep}
            for cohort in ('baseline', 'short', 'long')
            for rep in range(3)
            for c in GRID]
    (DST / 'jobs.json').write_text(json.dumps(jobs, indent=2, allow_nan=False) + '\n')

    # code_files_sha256: the frozen simulator source for this study.
    code_files = {}
    for py in sorted((REPO / 'mlsys_sim').glob('*.py')):
        code_files[py.name] = sha_path(py)

    # files_sha256: the eight bundle inputs the remote receiver re-verifies.
    bundle_files = ['experiment.json', 'baseline.spec.json', 'short.spec.json',
                    'long.spec.json', 'jobs.json', 'predictions-summary.json',
                    'selection.json', 'validation-contract.template.json']
    files_sha = {name: sha_path(DST / name) for name in bundle_files}

    identity = predictions['baseline']['identity']
    manifest = {
        'kind': 'course_sla_local_prediction_manifest',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'study_id': STUDY,
        'experiment_sha256': files_sha['experiment.json'],
        'specs_sha256': {c: files_sha[f'{c}.spec.json'] for c in COHORTS},
        'files_sha256': files_sha,
        'code_files_sha256': code_files,
        # Host-side: completed on the 9070XT before freeze_bundle; not fabricated here.
        'checkpoint_manifest': 'REMOTE_PENDING',
        'checkpoint_manifest_sha256': 'REMOTE_PENDING',
        'runtime_execution_config': 'REMOTE_PENDING',
        'runtime_config_sha256': 'REMOTE_PENDING',
        'scheduler_config_sha256': 'REMOTE_PENDING',
        'design_contract_sha256': 'REMOTE_PENDING',
        'identity': identity,
        'identity_sha256': predictions['baseline']['identity_sha256'],
        'prediction_artifacts_stored_locally': True,
        'prediction_artifacts_sha256': raw_sha,
        'canonical_prediction_payload_sha256': canonical_sha,
        'remote_freeze_pending': True,
        'local_manifest_sha256': None,
        'freeze_semantics': ('local CPU freeze: complete prediction payloads + raw/canonical '
                             'SHA256 committed; host-side checkpoint/runtime/design fields are '
                             'REMOTE_PENDING and completed on the 9070XT before '
                             'freeze_course_sla_commitment.py issues the receipt; reveal '
                             're-runs predict_closed_loop on the frozen inputs after measurement'),
    }
    manifest['local_manifest_sha256'] = canonical_sha256(
        {k: v for k, v in manifest.items() if k != 'local_manifest_sha256'})
    (DST / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')

    print('\nBUNDLE', DST)
    for name in sorted(p.name for p in DST.iterdir()):
        print(f'  {name:36s} {files_sha.get(name, sha_path(DST / name)) if name in files_sha else sha_path(DST / name)}')
    print('\nCOMMITMENTS (prediction payloads, archived at', ARCHIVE, ')')
    for c in COHORTS:
        print(f'  {c:9s} raw       {raw_sha[c]}')
        print(f'  {c:9s} canonical {canonical_sha[c]}')
    print(f'  manifest    {sha_path(DST / "manifest.json")}')
    print(f'  identity    {manifest["identity_sha256"]}')
    print('\nhardware_validated=false  remote_freeze_pending=true')


if __name__ == '__main__':
    main()
