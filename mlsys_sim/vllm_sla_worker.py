"""Optional pinned-vLLM worker for a frozen finite closed-loop course study.

GPU imports occur only in main(). API feedback drives clients; the existing
Scheduler hook supplies the separate server-token observation boundary.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any
import uuid

from .config import experiment_from_dict
from .load_validation import closed_loop_policy_sha256, generate_request_stream, validate_closed_loop_spec
from .profiles import profile_identity, validate_identity
from .runtime_adapter import normalize_runtime_trace
from .schema import Experiment, RequestSpec, SimulationError
from .serialization import experiment_dict


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _captured_json(raw: bytes) -> dict:
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SimulationError('INVALID_STUDY_INPUT', 'Duplicate JSON field: ' + key)
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise SimulationError('INVALID_STUDY_INPUT', 'Study input must be an inline JSON object')
    return value


def _study_inputs(experiment_path: Path, spec_path: Path) -> tuple[Experiment, dict, bytes, bytes]:
    experiment_raw, spec_raw = experiment_path.read_bytes(), spec_path.read_bytes()
    return (experiment_from_dict(_captured_json(experiment_raw)),
            validate_closed_loop_spec(_captured_json(spec_raw)), experiment_raw, spec_raw)


def verify_checkpoint(model: Path, manifest: list[dict], expected_sha256: str) -> None:
    """Bind the actually read local checkpoint files before each formal load."""
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != expected_sha256 or not manifest:
        raise SimulationError('CHECKPOINT_CHANGED', 'Frozen checkpoint manifest identity is invalid')
    selected_files = {str(path.relative_to(model)) for path in model.rglob('*')
                       if path.is_file() and (path.suffix in {'.json', '.safetensors'}
                                              or path.name.endswith('.metadata'))}
    if selected_files != {item['file'] for item in manifest}:
        raise SimulationError('CHECKPOINT_CHANGED', 'Checkpoint file set differs from frozen manifest')
    for item in manifest:
        relative = Path(item['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise SimulationError('CHECKPOINT_CHANGED', 'Checkpoint filename is outside its declared directory')
        path = model / relative
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise SimulationError('CHECKPOINT_CHANGED', 'Checkpoint file is missing or has changed size')
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha.update(chunk)
        if sha.hexdigest() != item['sha256']:
            raise SimulationError('CHECKPOINT_CHANGED', 'Checkpoint file bytes differ from the frozen manifest')


def formal_server_evidence(experiment, capture: dict, feedback: dict, *, allow_test_fixture: bool = False,
                           runtime_execution_config: dict | None = None) -> dict:
    """Select actual formal IDs from an unmodified constructor-origin capture."""
    source_kind = capture.get('source', {}).get('kind')
    if type(allow_test_fixture) is not bool or (source_kind != 'measured'
            and not (source_kind == 'test_fixture' and allow_test_fixture)):
        raise SimulationError('INVALID_SLA_CAPTURE', 'Measured source or explicit CPU fixture required')
    if (
            capture.get('metadata', {}).get('canonical_eligible') is not True
            or capture.get('metadata', {}).get('fresh_state_verified') is not True
            or capture.get('capture_errors') or not capture.get('capture_closed')):
        raise SimulationError('INVALID_SLA_CAPTURE', 'Complete fresh-origin measured Scheduler evidence required')
    actual_identity = validate_identity(capture.get('identity'), measured=source_kind == 'measured')
    if actual_identity != profile_identity(experiment):
        raise SimulationError('CALIBRATION_MISMATCH', 'Actual Scheduler identity differs from frozen experiment')
    expected_runtime = experiment.calibration.runtime_config_sha256
    if expected_runtime:
        snapshot = capture.get('metadata', {}).get('actual_scheduler_config')
        if (not isinstance(runtime_execution_config, dict)
                or hashlib.sha256(json.dumps(runtime_execution_config, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest() != expected_runtime):
            raise SimulationError('CALIBRATION_MISMATCH', 'Actual Runner execution config differs from freeze')
        scheduler_expected = {key: value for key, value in runtime_execution_config.items()
                              if key not in {'actual_attention_backends', 'torch_build', 'sampling_policy'}}
        if not isinstance(snapshot, dict) or snapshot != scheduler_expected:
            raise SimulationError('CALIBRATION_MISMATCH', 'Actual Scheduler execution config differs from freeze')
    if feedback.get('timing_boundary') != 'runtime_API_feedback_v1':
        raise SimulationError('INVALID_SLA_CAPTURE', 'Explicit API feedback ledger required')
    admissions = feedback['requests']
    mapping = {row['native_request_id']: row for row in admissions if row.get('native_request_id') is not None}
    if len(mapping) != len(admissions):
        raise SimulationError('INVALID_SLA_CAPTURE', 'Each offered request needs one actual native identity')
    captured = {row['request_id']: row for row in capture['requests']}
    if not mapping.keys() <= captured.keys():
        raise SimulationError('INVALID_SLA_CAPTURE', 'An API request has no Scheduler enqueue')
    origin = capture['clock_origin_monotonic_ns']
    selected = [event for event in capture['events'] if event['request_id'] in mapping]
    arrivals = {event['request_id']: event for event in selected if event['event'] == 'REQUEST_ARRIVAL'}
    if len(arrivals) != len(mapping) or not arrivals:
        raise SimulationError('INVALID_SLA_CAPTURE', 'Formal arrivals are incomplete or empty')
    start_ns = min(origin + round(event['timestamp_ms'] * 1e6) for event in arrivals.values())
    specs = []
    for native, row in mapping.items():
        actual = captured[native]
        if (actual['input_tokens'] != row['input_tokens']
                or actual['max_output_tokens'] != row['output_tokens']):
            raise SimulationError('INVALID_SLA_CAPTURE', 'Actual prompt/output targets differ from frozen stream')
        expected_hash = hashlib.sha256(json.dumps([row['prompt_token_id']] * row['input_tokens'],
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if actual.get('prompt_token_ids_sha256') != expected_hash:
            raise SimulationError('INVALID_SLA_CAPTURE', 'Actual prompt identity differs from frozen stream')
        at_ns = origin + round(arrivals[native]['timestamp_ms'] * 1e6)
        if not row['submit_before_ns'] <= at_ns <= row['submit_after_ns']:
            raise SimulationError('INVALID_SLA_CAPTURE', 'Scheduler enqueue is outside actual add_request call')
        specs.append(RequestSpec(row['request_id'], (at_ns - start_ns) / 1e6,
                                 row['input_tokens'], row['output_tokens'], timeout_ms=row['timeout_ms'],
                                 prompt_token_ids=(row['prompt_token_id'],) * row['input_tokens']))
    events = []
    for event in selected:
        events.append({**event, 'request_id': mapping[event['request_id']]['request_id'],
                       'timestamp_ms': (origin + round(event['timestamp_ms'] * 1e6) - start_ns) / 1e6})
    observed_until = (feedback['formal_end_ns'] - start_ns) / 1e6
    if observed_until < max(event['timestamp_ms'] for event in events):
        raise SimulationError('INVALID_SLA_CAPTURE', 'Feedback ended before the observed server events')
    measured_exp = replace(experiment, workload=tuple(specs))
    trace = {'schema_version': 1, 'kind': 'runtime_trace',
             'source': {'kind': source_kind, 'reference': 'pinned Scheduler events; formal ID/clock selection'},
             'timing_boundary': 'server_token_ready_v1', 'observed_until_ms': observed_until,
             'identity': actual_identity, 'events': events,
             'notes': {'selection': 'actual native IDs from the independently driven finite client policy',
                       'normalization_only': 'actual arrivals normalize measured metrics; predictions are never replayed from them'}}
    bundle = normalize_runtime_trace(measured_exp, trace, include_request_evidence=True,
                                     allow_test_fixture=allow_test_fixture)
    descriptions = {row.request_id: row for row in specs}
    feedback_by_router = {row['request_id']: row for row in admissions}
    for row in bundle['request_evidence']['requests']:
        descriptor = descriptions[row['request_id']]
        row['input_tokens'] = descriptor.input_tokens
        row['output_tokens_target'] = descriptor.output_tokens
        row['prompt_token_ids_sha256'] = hashlib.sha256(json.dumps(list(descriptor.prompt_token_ids),
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        original = feedback_by_router[row['request_id']]
        for name in ('client_id', 'stream_index', 'preceding_request_id'):
            row[name] = original.get(name)
    return {'trace': trace, 'normalization': bundle, 'formal_origin_monotonic_ns': start_ns,
            'evidence': bundle['request_evidence']}


class _VllmBackend:
    source = {'kind': 'measured', 'reference': 'actual pinned synchronous vLLM engine API feedback'}

    def __init__(self, engine: Any, parameters: Any):
        self.engine, self.parameters = engine, parameters
        self.native_by_router: dict[str, str] = {}

    def add_request(self, request: dict) -> str:
        native = self.engine.add_request(request['request_id'],
            {'prompt_token_ids': [request['prompt_token_id']] * request['input_tokens']}, self.parameters)
        self.native_by_router[request['request_id']] = native
        return native

    def step(self) -> list[dict]:
        rows = []
        for output in self.engine.step():
            if len(output.outputs) != 1:
                raise SimulationError('UNSUPPORTED_RUNTIME_OUTPUT', 'Exactly one greedy candidate is required')
            native = self.native_by_router.get(output.request_id, output.request_id)
            rows.append({'native_request_id': native, 'finished': output.finished,
                         'output_tokens_actual': len(output.outputs[0].token_ids)})
        return rows

    def has_unfinished_requests(self) -> bool:
        return self.engine.has_unfinished_requests()

    def abort_requests(self, native_ids: list[str]) -> None:
        self.engine.abort_request(native_ids)


def _freeze_check(args, experiment_raw: bytes, spec_raw: bytes) -> dict | None:
    if args.phase == 'pilot':
        return None
    if args.freeze_receipt is None:
        raise SimulationError('MISSING_PREDICTION_FREEZE', 'Formal collection requires a freeze receipt')
    receipt = _captured_json(args.freeze_receipt.read_bytes())
    root = args.freeze_receipt.parent
    if receipt.get('kind') != 'course_sla_prediction_freeze_receipt':
        raise SimulationError('INVALID_PREDICTION_FREEZE', 'Unexpected prediction freeze receipt')
    for filename, expected in receipt['files_sha256'].items():
        if _sha(root / filename) != expected:
            raise SimulationError('INPUT_CHANGED', 'Frozen prediction input changed: ' + filename)
    if hashlib.sha256(experiment_raw).hexdigest() != receipt['experiment_sha256']:
        raise SimulationError('INPUT_CHANGED', 'Experiment differs from the prediction freeze')
    if hashlib.sha256(spec_raw).hexdigest() not in receipt['specs_sha256'].values():
        raise SimulationError('INPUT_CHANGED', 'Study policy differs from the prediction freeze')
    for filename, expected in receipt['code_files_sha256'].items():
        if _sha(Path(__file__).resolve().parent / filename) != expected:
            raise SimulationError('INPUT_CHANGED', 'Runtime collection code differs from the freeze')
    if time.perf_counter_ns() <= receipt['remote_freeze_monotonic_ns']:
        raise SimulationError('INVALID_PREDICTION_FREEZE', 'Formal collection must follow remote prediction freeze')
    verify_checkpoint(args.model, receipt['checkpoint_manifest'], receipt['checkpoint_manifest_sha256'])
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--replication', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('pilot', 'formal'), required=True)
    parser.add_argument('--freeze-receipt', type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    experiment, spec, experiment_raw, spec_raw = _study_inputs(args.experiment, args.spec)
    if args.phase == 'formal' and not experiment.calibration.runtime_config_sha256:
        raise SimulationError('MISSING_RUNTIME_BINDING', 'Formal prediction must bind the actual Runner execution config')
    if (not args.model.is_dir() or experiment.strategy.replicas != 1
            or experiment.strategy.tensor_parallel_size != 1 or experiment.strategy.pipeline_parallel_size != 1
            or experiment.strategy.prefix_cache or not experiment.strategy.mixed_batching
            or experiment.strategy.model.weight_dtype != 'bfloat16'):
        raise SimulationError('UNSUPPORTED_STUDY_RUNTIME', 'This worker supports one BF16 synchronous dense model')
    receipt = _freeze_check(args, experiment_raw, spec_raw)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    session = 'sla-' + uuid.uuid4().hex
    capture_config = {'schema_version': 1, 'experiment': experiment_dict(experiment), 'session_id': session,
                      'output_dir': str(args.out / 'captures'), 'capture_scheduler': True,
                      'record_prompt_token_ids': True, 'max_records': 100000, 'max_events': 1000000,
                      'source_reference': 'actual single-card course closed-loop validation; ' + args.phase}
    _write(args.out / 'capture-config.json', capture_config)
    os.environ['MLSYS_VLLM_CAPTURE_CONFIG'] = str(args.out / 'capture-config.json')
    summary: dict = {'status': 'FAILED', 'phase': args.phase, 'session_id': session, 'pid': os.getpid(),
                    'started_at': _now(), 'spec': spec, 'identity': profile_identity(experiment),
                    'concurrency': args.concurrency, 'replication': args.replication,
                    'policy_sha256': closed_loop_policy_sha256(spec, args.concurrency, args.replication),
                    'request_stream_sha256': generate_request_stream(spec, replication=args.replication)['request_stream_sha256'],
                    'hardware_validated': False, 'source': {'kind': 'measured', 'reference': 'actual owned fresh Runtime'},
                    'source_sha256': _sha(Path(__file__)), 'freeze_receipt': receipt}
    llm = None
    try:
        from .closed_loop_capture import collect_closed_loop
        from .vllm_hooks import register_vllm_hooks
        register_vllm_hooks()
        import torch
        from vllm import LLM, SamplingParams
        props = torch.cuda.get_device_properties(0)
        if ('gfx1201' not in props.gcnArchName or props.total_memory != experiment.infra.devices[0].memory_bytes
                or torch.version.hip is None):
            raise SimulationError('HARDWARE_IDENTITY_MISMATCH', 'Actual single-device ROCm identity differs')
        strategy, memory = experiment.strategy, experiment.strategy.memory
        llm = LLM(str(args.model), tensor_parallel_size=1, pipeline_parallel_size=1,
                  distributed_executor_backend='uni',
                  worker_extension_cls='mlsys_sim.vllm_hooks.VllmCaptureWorkerExtension',
                  dtype='bfloat16', enforce_eager=True, async_scheduling=False,
                  max_model_len=2048, max_num_batched_tokens=strategy.max_batch_tokens,
                  max_num_seqs=strategy.max_batch_size, block_size=memory.kv_block_tokens,
                  enable_prefix_caching=False, enable_chunked_prefill=strategy.chunked_prefill,
                  kv_cache_memory_bytes=memory.kv_pool_bytes, disable_custom_all_reduce=True,
                  skip_tokenizer_init=True, generation_config='vllm', seed=0)
        parameters = SamplingParams(temperature=0, max_tokens=spec['output_tokens'], ignore_eos=True, detokenize=False)
        engine = llm.llm_engine
        core = engine.engine_core.engine_core
        scheduler = core.scheduler
        scheduler_capture = scheduler._mlsys_scheduler_capture
        initial_free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        warmup = []
        for length in (64, 128, 256, 512, 1024):
            answers = llm.generate([{'prompt_token_ids': [spec['prompt_token_id']] * length}
                                    for _ in range(min(args.concurrency, 16))], parameters, use_tqdm=False)
            if len(answers) != min(args.concurrency, 16) or any(len(row.outputs[0].token_ids) != spec['output_tokens']
                                                              or not row.finished for row in answers):
                raise SimulationError('WARMUP_FAILED', 'Warmup did not produce every fixed-length output')
            warmup.append({'input_tokens': length, 'requests': len(answers)})
        free_after = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        if (engine.has_unfinished_requests() or scheduler.requests or scheduler.running or scheduler.waiting
                or initial_free != free_after):
            raise SimulationError('WARMUP_NOT_DRAINED', 'Warmup must release every request and restore logical KV')
        binding = llm.collective_rpc('mlsys_capture_close', timeout=60)
        if (len(binding) != 1 or binding[0]['capture'].get('capture_closed') is not True
                or binding[0]['capture'].get('capture_errors') or not binding[0]['capture'].get('records')
                or any(row.get('status') != 'COMPLETED' or row.get('profile_eligible') is not True
                       for row in binding[0]['capture']['records'])):
            raise SimulationError('INVALID_RUNTIME_BINDING', 'Complete single-rank initial binding required')
        _write(args.out / 'runner-binding.json', binding[0]['capture'])
        summary['runtime_config_sha256'] = binding[0]['capture']['metadata']['runtime_config_sha256']
        summary['runtime_execution_config'] = binding[0]['capture']['metadata']['runtime_execution_config']
        if (experiment.calibration.runtime_config_sha256
                and summary['runtime_config_sha256'] != experiment.calibration.runtime_config_sha256):
            raise SimulationError('CALIBRATION_MISMATCH', 'Actual Runner execution config differs from freeze')
        summary['warmup'] = {'cohorts': warmup, 'initial_free_blocks': initial_free,
                             'post_warmup_free_blocks': free_after, 'prefix_cache_disabled': True,
                             'runner_capture_closed_before_formal': True}
        torch.cuda.reset_peak_memory_stats()
        summary['collection_started_at'] = _now()
        def progress(state):
            print(json.dumps({'kind': 'sla_progress', 'cohort': spec['cohort'], 'concurrency': args.concurrency,
                              'replication': args.replication, **state}), flush=True)
        feedback = collect_closed_loop(_VllmBackend(engine, parameters), spec, args.concurrency,
                                       replication=args.replication, progress=progress)
        summary['collection_ended_at'] = _now()
        _write(args.out / 'api-feedback.json', feedback)
        summary['api_feedback_status'] = feedback['status']
        summary['allocator_diagnostics'] = {'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'scope': 'raw Torch allocator diagnostics; not complete physical memory/Workspace proof'}
        summary['scheduler_capture_file'] = scheduler_capture.flush()
        capture = scheduler_capture.export_capture()
        selected = formal_server_evidence(experiment, capture, feedback,
                                          runtime_execution_config=summary['runtime_execution_config'])
        _write(args.out / 'server-trace.json', selected['trace'])
        _write(args.out / 'observed-run.json', selected['evidence'])
        summary['observed_run_sha256'] = _sha(args.out / 'observed-run.json')
        summary['timing_boundary'] = 'server_token_ready_v1'
        summary['status'] = 'COMPLETED' if feedback['status'] == 'COMPLETED' else 'PARTIAL'
    except BaseException as error:
        summary.update(error_code=getattr(error, 'code', type(error).__name__), error=str(error),
                       traceback=traceback.format_exc())
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
                summary['shutdown'] = 'completed'
            except Exception as error:
                summary['shutdown'] = 'failed: ' + str(error)
        summary['ended_at'] = _now()
        _write(args.out / 'summary.json', summary)
    return 0 if summary['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
