"""Strict aggregation of versioned vLLM internal capture channels.

Per-rank synchronized host intervals are reduced by maximum duration. Absolute
timestamps from different ranks are never subtracted to manufacture service
time. Incomplete, overlapping or unsupported samples remain raw diagnostics.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, NoReturn

from .config import validate_experiment
from .layout import replica_experiment, stage_layer_counts
from .profiles import profile_identity, shape_key, validate_dataset, validate_identity
from .runtime_adapter import normalize_runtime_trace
from .schema import Experiment, RequestSpec, SimulationError


_REVISION = "98dff2a81"
_VERSION = "0.29.1.dev0+g98dff2a81"
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_CHANNELS = 1024
_MAX_RECORDS = 1000000
_BOUNDARY = "step_includes_output_and_sampling_v1"
_NAMESPACE_PREFIX = "vllm-salt-sha256:"


def _fail(message: str, code: str = "INVALID_VLLM_CAPTURE") -> NoReturn:
    raise SimulationError(code, message)


def _object(value: Any, required: set[str], optional: set[str], label: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(f"{label} must be an object with string keys")
    if required - value.keys() or value.keys() - required - optional:
        _fail(f"{label} has missing or unknown fields")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        _fail(f"{label} must be a bounded nonempty string")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be finite")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        _fail(f"{label} must be finite")
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        _fail(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise SimulationError("INVALID_VLLM_CAPTURE", "Capture must be finite JSON data") from exc


def _source(value: Any, allow_test_fixture: bool) -> dict:
    source = _object(value, {"kind", "reference"}, set(), "capture source")
    if source["kind"] not in ("measured", "test_fixture"):
        _fail("Capture source must be measured or test_fixture")
    _text(source["reference"], "source.reference")
    if source["kind"] == "test_fixture" and not allow_test_fixture:
        _fail("Fixture captures require explicit permission", "TEST_FIXTURE_NOT_ALLOWED")
    return dict(source)


def _opaque_namespace(value: str) -> bool:
    return (value.startswith(_NAMESPACE_PREFIX) and len(value) == len(_NAMESPACE_PREFIX) + 64
            and all(character in "0123456789abcdef" for character in value[len(_NAMESPACE_PREFIX):]))


def _expected_namespace(value: str) -> str:
    # RequestSpec namespaces are literal domains. Guessing an already-encoded
    # value from its spelling would collapse two distinct native salt values.
    # A previously bound replay must opt into binding again to use captured
    # opaque domains, rather than being reinterpreted implicitly.
    from .vllm_hooks import cache_namespace_from_vllm_salt
    return cache_namespace_from_vllm_salt(value)


def _supported_version(value: Any) -> str:
    version = _text(value, "runtime_version")
    if version != _VERSION and not version.startswith(_VERSION + "."):
        _fail("Capture runtime does not match the pinned vLLM source revision", "UNSUPPORTED_VLLM_VERSION")
    return version


def load_vllm_captures(paths: Sequence[str | Path]) -> list[dict]:
    """Read bounded complete JSON exports and retain their original file hashes."""
    if not isinstance(paths, (list, tuple)) or not paths or len(paths) > _MAX_CHANNELS:
        _fail("capture paths must be a bounded nonempty array")
    captures = []
    total = 0

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("Duplicate JSON key in capture")
            result[key] = value
        return result

    for supplied in paths:
        if not isinstance(supplied, (str, Path)):
            _fail("Each capture path must be a path or string")
        path = Path(supplied)
        try:
            if path.is_symlink():
                _fail("Capture files may not be symlinks")
            with path.open("rb") as stream:
                raw = stream.read(_MAX_FILE_BYTES + 1)
            total += len(raw)
            if len(raw) > _MAX_FILE_BYTES or total > _MAX_TOTAL_BYTES:
                _fail("Capture input exceeds the bounded read size")
            payload = json.loads(raw, object_pairs_hook=pairs,
                                 parse_constant=lambda _: _fail("Nonfinite JSON in capture"))
        except SimulationError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise SimulationError("INVALID_VLLM_CAPTURE", f"Cannot read capture {path}: {exc}") from exc
        if not isinstance(payload, dict) or "_capture_input" in payload:
            _fail("Capture file must contain one exported object without reserved metadata")
        _json(payload)
        captures.append({**payload, "_capture_input": {"path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}})
    return captures


def _header(experiment: Experiment, value: Any, allow_test_fixture: bool) -> tuple[dict, dict]:
    required = {"schema_version", "kind", "source", "session_id", "runtime_version", "api_revision",
                "clock_domain", "replica_index", "stage_index", "tp_rank", "tensor_parallel_size",
                "pipeline_parallel_size", "identity", "experiment", "records", "events", "metadata",
                "hardware_validated"}
    capture = _object(value, required, {"_capture_input", "capture_errors", "capture_closed"}, "runner capture")
    if type(capture["schema_version"]) is not int or capture["schema_version"] != 1:
        _fail("Capture schema_version must be 1")
    if capture["kind"] != "vllm_internal_capture":
        _fail("A runner capture must have kind=vllm_internal_capture")
    if capture["api_revision"] != _REVISION:
        _fail("Unknown vLLM capture ABI revision", "UNSUPPORTED_VLLM_VERSION")
    _supported_version(capture["runtime_version"])
    source = _source(capture["source"], allow_test_fixture)
    _text(capture["session_id"], "session_id")
    _text(capture["clock_domain"], "clock_domain")
    replica = _integer(capture["replica_index"], "replica_index")
    stage = _integer(capture["stage_index"], "stage_index")
    rank = _integer(capture["tp_rank"], "tp_rank")
    strategy = experiment.strategy
    if replica >= strategy.replicas or stage >= strategy.pipeline_parallel_size or rank >= strategy.tensor_parallel_size:
        _fail("Capture rank coordinate is outside the experiment layout")
    if (_integer(capture["tensor_parallel_size"], "tensor_parallel_size", 1) != strategy.tensor_parallel_size
            or _integer(capture["pipeline_parallel_size"], "pipeline_parallel_size", 1) != strategy.pipeline_parallel_size):
        _fail("Capture TP/PP dimensions differ from the experiment", "CALIBRATION_MISMATCH")
    local = replica_experiment(experiment, replica)
    identity = validate_identity(capture["identity"], measured=source["kind"] == "measured")
    actual_hash = identity["runtime"].get("runtime_config_sha256", "")
    expected_hash = getattr(local.calibration, "runtime_config_sha256", "")
    if not expected_hash and actual_hash:
        local = replace(local, calibration=replace(local.calibration, runtime_config_sha256=actual_hash))
    if identity != profile_identity(local):
        _fail("Capture identity differs from the configured local replica", "CALIBRATION_MISMATCH")
    from .config import experiment_from_dict
    captured_exp = experiment_from_dict(capture["experiment"])
    if profile_identity(captured_exp) != identity:
        _fail("Capture experiment and identity disagree", "CALIBRATION_MISMATCH")
    if capture["hardware_validated"] is not False:
        _fail("Raw collection must not claim prediction validation")
    if not isinstance(capture["records"], list) or not isinstance(capture["events"], list):
        _fail("Capture records and events must be arrays")
    if "capture_errors" in capture and not isinstance(capture["capture_errors"], list):
        _fail("capture_errors must be an array")
    if "capture_closed" in capture and type(capture["capture_closed"]) is not bool:
        _fail("capture_closed must be boolean")
    metadata = capture["metadata"]
    if not isinstance(metadata, dict):
        _fail("Capture metadata must be an object")
    if metadata.get("instrumentation_source") != "vllm_internal_runner" or metadata.get("synchronized") is not True:
        _fail("Complete synchronized runner instrumentation must be declared")
    execution = metadata.get("runtime_execution_config")
    if execution is not None:
        if not isinstance(execution, dict) or hashlib.sha256(_json(execution).encode()).hexdigest() != actual_hash:
            _fail("Actual runtime execution snapshot and identity hash differ", "CALIBRATION_MISMATCH")
    if metadata.get("runtime_config_sha256", actual_hash) != actual_hash:
        _fail("Runtime metadata hash and identity disagree", "CALIBRATION_MISMATCH")
    eligible_records = any(isinstance(record, dict) and record.get("status") == "COMPLETED"
                           and record.get("profile_eligible") is not False for record in capture["records"])
    if source["kind"] == "measured" and eligible_records and (not actual_hash or execution is None):
        _fail("Measured capture requires its actual execution snapshot and identity hash", "HARDWARE_METADATA_INCOMPLETE")
    binding = metadata.get("stage_layer_binding")
    if binding is not None:
        fact = _object(binding, {"start_layer", "end_layer", "layer_count", "method"}, set(), "stage layer binding")
        start = _integer(fact["start_layer"], "start_layer")
        end = _integer(fact["end_layer"], "end_layer")
        count = _integer(fact["layer_count"], "layer_count", 1)
        if fact["method"] != "loaded_qwen_model_start_end_layer" or end - start != count:
            _fail("Stage binding does not describe an observed Qwen layer range")
        layers = stage_layer_counts(local)
        expected_start, expected_end = sum(layers[:stage]), sum(layers[:stage + 1])
        if eligible_records and (start != expected_start or end != expected_end):
            _fail("Loaded PP stage layers differ from the configured layout", "CALIBRATION_MISMATCH")
    elif (source["kind"] == "measured" and eligible_records
          and strategy.pipeline_parallel_size > 1):
        _fail("Measured PP capture lacks observed loaded stage-layer bounds", "HARDWARE_METADATA_INCOMPLETE")
    if source["kind"] == "measured" and strategy.tensor_parallel_size > 1:
        binding = metadata.get("rccl_binding")
        if not isinstance(binding, dict) or binding.get("verified") is not True:
            _fail("Measured TP capture lacks a loaded RCCL binding", "HARDWARE_METADATA_INCOMPLETE")
        version = binding.get("runtime_version")
        digest = binding.get("library_sha256")
        if version != local.calibration.rccl_version:
            _fail("Loaded RCCL version differs from capture identity", "CALIBRATION_MISMATCH")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            _fail("Loaded RCCL binding requires a library SHA256", "HARDWARE_METADATA_INCOMPLETE")
        if binding.get("verification_scope") != "local_proc_maps_loaded_library_ncclGetVersion":
            _fail("RCCL binding does not identify the loaded Runtime library", "HARDWARE_METADATA_INCOMPLETE")
    return capture, source


def _record(value: Any, stage: int, pp: int) -> dict:
    required = {"sample_id", "iteration_id", "request_ids", "phase", "query_tokens", "context_lengths",
                "prompt_lengths", "start_monotonic_ns", "end_monotonic_ns", "duration_ms", "status",
                "timing_boundary", "capture_boundary", "forward_complete", "sampling_required",
                "sampling_complete", "workspace_peak_bytes", "allocator_baseline", "allocator_after"}
    optional = {"request_phases", "error_code", "message", "async_output", "deferred_output_resolved",
                "output_history_lengths", "resumed_request_ids", "unsupported_reason", "capture_quality",
                "profile_eligible", "canonical_rejection_reason", "transport_metadata"}
    optional.add("sampled_token_counts")
    minimal = {"sample_id", "iteration_id", "status"}
    record = _object(value, minimal, (required - minimal) | optional, "capture record")
    _text(record["sample_id"], "sample_id")
    _integer(record["iteration_id"], "iteration_id")
    if record["status"] not in ("COMPLETED", "FAILED", "INCOMPLETE", "UNSUPPORTED"):
        _fail("Capture record has an unknown status")
    for flag in ("forward_complete", "sampling_required", "sampling_complete"):
        if flag in record and type(record[flag]) is not bool:
            _fail(f"{flag} must be boolean")
    if "sampling_required" in record and record["sampling_required"] != (stage == pp - 1):
        _fail("Sampling requirement must match the PP stage")
    if record["status"] != "COMPLETED":
        # Failed/unsupported iterations can lack a usable descriptor or ending
        # timestamp. They remain preserved and never produce a fake shape.
        return record
    if required - record.keys():
        _fail("A completed capture must contain its complete descriptor and timing evidence")
    identifiers = record["request_ids"]
    if (not isinstance(identifiers, list) or not identifiers
            or any(not isinstance(item, str) or not item for item in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        _fail("Completed capture request_ids must be a unique nonempty array")
    if not isinstance(record["prompt_lengths"], list) or len(record["prompt_lengths"]) != len(identifiers):
        _fail("One prompt length is required per completed request")
    for prompt in record["prompt_lengths"]:
        _integer(prompt, "prompt_length", 1)
    try:
        shape_key(record["phase"], record["query_tokens"], record["context_lengths"], stage,
                  request_phases=record.get("request_phases"))
    except SimulationError as exc:
        raise SimulationError("INVALID_VLLM_CAPTURE", f"Invalid captured shape: {exc}") from exc
    if record["phase"] not in ("prefill", "decode", "mixed"):
        _fail("Captured recomputation requires a separate explicit Runtime contract")
    if len(record["query_tokens"]) != len(identifiers):
        _fail("Captured shape arrays must align with request_ids")
    if record["phase"] != "mixed" and "request_phases" in record:
        _fail("Homogeneous capture must omit request_phases")
    roles = record.get("request_phases", [record["phase"]] * len(identifiers))
    for query, context, prompt, role in zip(record["query_tokens"], record["context_lengths"],
                                          record["prompt_lengths"], roles):
        if role == "prefill" and (context >= prompt or context + query > prompt):
            _fail("Prefill capture must process an unfinished prompt suffix")
        if role == "decode" and context < prompt:
            _fail("One-token unfinished prefill cannot be relabeled as Decode")
    start = _integer(record["start_monotonic_ns"], "start_monotonic_ns")
    end = _integer(record["end_monotonic_ns"], "end_monotonic_ns")
    duration = _number(record["duration_ms"], "duration_ms", positive=True)
    if end <= start or not math.isclose(duration, (end - start) / 1e6, rel_tol=1e-12, abs_tol=1e-9):
        _fail("Captured duration does not match its monotonic interval")
    if record["timing_boundary"] != _BOUNDARY:
        _fail("Unknown captured stage timing boundary")
    if not isinstance(record["allocator_baseline"], list) or not isinstance(record["allocator_after"], list):
        _fail("Allocator evidence must remain raw per-device arrays")
    for flag in ("async_output", "deferred_output_resolved", "profile_eligible"):
        if flag in record and type(record[flag]) is not bool:
            _fail(f"{flag} must be boolean")
    history = record.get("output_history_lengths")
    if history is not None:
        if not isinstance(history, list) or len(history) != len(identifiers):
            _fail("Output history lengths must align with captured requests")
        for count in history:
            _integer(count, "output_history_length")
    resumed = record.get("resumed_request_ids", [])
    if not isinstance(resumed, list) or any(item not in identifiers for item in resumed):
        _fail("Resumed request IDs must belong to the captured batch")
    counts = record.get("sampled_token_counts")
    if counts is not None:
        if not isinstance(counts, list) or (len(counts) != len(identifiers)
                                            and not (stage < pp - 1 and counts == [])):
            _fail("Sampled token counts must align with captured requests")
        for count in counts:
            _integer(count, "sampled_token_count")
    return record


def _shape_signature(record: dict) -> str:
    return _json({key: record.get(key) for key in ("sample_id", "request_ids", "phase", "query_tokens",
        "context_lengths", "prompt_lengths", "request_phases", "output_history_lengths", "resumed_request_ids")})


def _scheduler_header(experiment: Experiment, value: Any, allow_test_fixture: bool) -> tuple[dict, dict]:
    required = {"schema_version", "kind", "source", "session_id", "runtime_version", "api_revision",
                "clock_domain", "clock_origin_monotonic_ns", "replica_index", "identity", "experiment",
                "metadata", "hardware_validated", "events", "requests", "capture_errors", "capture_closed",
                "observed_until_ms"}
    capture = _object(value, required, {"_capture_input"}, "scheduler capture")
    if (type(capture["schema_version"]) is not int or capture["schema_version"] != 1
            or capture["kind"] != "vllm_scheduler_capture" or capture["api_revision"] != _REVISION):
        _fail("Unknown scheduler capture schema or ABI revision")
    _supported_version(capture["runtime_version"])
    source = _source(capture["source"], allow_test_fixture)
    _text(capture["session_id"], "session_id")
    _text(capture["clock_domain"], "clock_domain")
    _integer(capture["clock_origin_monotonic_ns"], "clock_origin_monotonic_ns")
    _number(capture["observed_until_ms"], "observed_until_ms")
    replica = _integer(capture["replica_index"], "replica_index")
    if replica >= experiment.strategy.replicas:
        _fail("Scheduler replica index is outside the experiment")
    local = replica_experiment(experiment, replica)
    identity = validate_identity(capture["identity"], measured=source["kind"] == "measured")
    expected = profile_identity(local)
    declared_hash = expected["runtime"].pop("runtime_config_sha256", "")
    captured = deepcopy(identity)
    actual_hash = captured["runtime"].pop("runtime_config_sha256", "")
    if captured != expected or declared_hash and actual_hash and declared_hash != actual_hash:
        _fail("Scheduler capture identity differs from the local experiment", "CALIBRATION_MISMATCH")
    from .config import experiment_from_dict
    if profile_identity(experiment_from_dict(capture["experiment"])) != identity:
        _fail("Scheduler experiment and identity disagree", "CALIBRATION_MISMATCH")
    if capture["hardware_validated"] is not False or type(capture["capture_closed"]) is not bool:
        _fail("Scheduler capture must declare unvalidated collection and boolean closure")
    for name in ("events", "requests", "capture_errors"):
        if not isinstance(capture[name], list):
            _fail(f"Scheduler {name} must be an array")
    metadata = capture["metadata"]
    if (not isinstance(metadata, dict) or metadata.get("instrumentation_source") != "vllm_internal_scheduler"
            or metadata.get("timing_boundary") != "server_token_ready_v1"
            or metadata.get("observation_point") != "scheduler_enqueue_to_engine_core_output_dispatch"):
        _fail("Unknown scheduler instrumentation boundary")
    if type(metadata.get("canonical_eligible")) is not bool:
        _fail("Scheduler metadata must declare canonical eligibility")
    return capture, source


def _scheduler_replay(experiment: Experiment, channels: list[dict], source: dict,
                      *, bind_runtime_workload: bool) -> tuple[Experiment, dict | None, dict]:
    provenance: dict[str, Any] = {"applied": False, "requested": bind_runtime_workload,
        "semantics": "conditional_replay_of_recorded_scheduler_arrivals_and_actual_outputs"}
    if not channels:
        return experiment, None, {**provenance, "reason": "No scheduler lifecycle channel was supplied"}
    if len({capture["replica_index"] for capture in channels}) != len(channels):
        _fail("Duplicate scheduler channel for a replica")
    if {capture["replica_index"] for capture in channels} != set(range(experiment.strategy.replicas)):
        return experiment, None, {**provenance, "reason": "Scheduler lifecycle does not cover every configured replica"}
    if any(capture["capture_errors"] or not capture["capture_closed"]
           or capture["metadata"]["canonical_eligible"] is not True for capture in channels):
        return experiment, None, {**provenance, "reason": "Scheduler capture is incomplete or marked ineligible"}
    if experiment.execution.warmup:
        return experiment, None, {**provenance, "reason": "Scheduler capture has no explicit warmup/measurement separation"}
    origin = min(capture["clock_origin_monotonic_ns"] for capture in channels)
    originals = {request.request_id: request for request in experiment.workload}
    specs: dict[str, RequestSpec] = {}
    descriptor_counts: dict[str, int] = {}
    descriptor_statuses: dict[str, str] = {}
    events = []
    ends = []
    actual_hash = getattr(experiment.calibration, "runtime_config_sha256", "")
    for channel in channels:
        channel_hash = channel["identity"]["runtime"].get("runtime_config_sha256", "")
        if channel_hash and channel_hash != actual_hash:
            _fail("Scheduler execution identity conflicts with the paired runner", "CALIBRATION_MISMATCH")
        offset = (channel["clock_origin_monotonic_ns"] - origin) / 1e6
        ends.append(offset + channel["observed_until_ms"])
        replica = channel["replica_index"]
        metadata = channel["metadata"]
        previous_native: dict[int, tuple[float, str]] = {}
        if metadata.get("priority_semantics") != "simulator_priority_equals_negated_vllm_priority":
            return experiment, None, {**provenance, "reason": "Scheduler priority semantics are not established"}
        for supplied in channel["requests"]:
            row = _object(supplied, {"request_id", "input_tokens", "arrival_ms", "priority", "native_priority",
                "max_output_tokens", "output_tokens_actual", "status"},
                {"finish_reason", "prompt_token_ids", "cache_namespace", "prompt_token_ids_sha256", "error_code",
                 "native_arrival_time"},
                "scheduler request")
            name = _text(row["request_id"], "request_id")
            if name in specs:
                _fail("Scheduler request ID is duplicated across replica channels")
            prompt = _integer(row["input_tokens"], "input_tokens", 1)
            arrival = offset + _number(row["arrival_ms"], "arrival_ms")
            maximum = _integer(row["max_output_tokens"], "max_output_tokens", 1)
            actual = _integer(row["output_tokens_actual"], "output_tokens_actual")
            if actual > maximum:
                _fail("Scheduler output count exceeds its declared maximum")
            priority, native = row["priority"], row["native_priority"]
            if (type(priority) is not int or type(native) is not int
                    or not -(2**63) <= priority <= 2**63 - 1 or priority != -native):
                _fail("Scheduler priority conversion is inconsistent")
            native_arrival = row.get("native_arrival_time")
            if native_arrival is not None:
                order = _number(native_arrival, "native_arrival_time"), name
                if native in previous_native and order < previous_native[native]:
                    return experiment, None, {**provenance,
                        "reason": "Native arrival order differs from the captured enqueue order"}
                previous_native[native] = order
            if row["status"] not in ("COMPLETED", "FAILED", "TIMED_OUT"):
                return experiment, None, {**provenance, "reason": "Scheduler lifecycle lacks a terminal outcome"}
            if row["status"] == "COMPLETED" and actual == 0:
                _fail("Completed scheduler request has no published output")
            original = originals.get(name)
            ids = row.get("prompt_token_ids")
            identity_hash = row.get("prompt_token_ids_sha256")
            if identity_hash is not None and (not isinstance(identity_hash, str) or len(identity_hash) != 64
                    or any(character not in "0123456789abcdef" for character in identity_hash)):
                _fail("Captured prompt identity hash must be a SHA256 digest")
            if ids is not None:
                if (not isinstance(ids, list) or len(ids) != prompt
                        or any(type(token) is not int or not 0 <= token < experiment.strategy.model.vocab_size for token in ids)):
                    _fail("Captured prompt token IDs do not match the request geometry")
                token_ids = tuple(ids)
                if identity_hash is not None and hashlib.sha256(_json(ids).encode()).hexdigest() != identity_hash:
                    _fail("Captured prompt token IDs contradict their identity hash", "COMPARISON_MISMATCH")
                if original is not None and original.prompt_token_ids and tuple(original.prompt_token_ids) != token_ids:
                    _fail("Captured prompt IDs differ from the supplied workload", "COMPARISON_MISMATCH")
            else:
                token_ids = tuple(original.prompt_token_ids) if original is not None else ()
                if token_ids:
                    if identity_hash is None:
                        return experiment, None, {**provenance,
                            "reason": "Original prompt IDs cannot be borrowed without a captured token-array hash"}
                    if hashlib.sha256(_json(list(token_ids)).encode()).hexdigest() != identity_hash:
                        _fail("Captured prompt identity differs from the supplied workload", "COMPARISON_MISMATCH")
            if experiment.strategy.prefix_cache and not token_ids:
                return experiment, None, {**provenance, "reason": "Prefix Cache replay requires captured or matching original prompt IDs"}
            if "cache_namespace" not in row and (experiment.strategy.prefix_cache
                                                   or original is not None and original.cache_namespace):
                return experiment, None, {**provenance, "reason": "Captured native cache namespace is required for prefix identity"}
            namespace = row.get("cache_namespace", "")
            if not isinstance(namespace, str):
                _fail("Captured cache namespace must be a string")
            if "cache_namespace" in row:
                if metadata.get("cache_namespace_encoding") != "vllm_nonempty_salt_sha256_v1":
                    return experiment, None, {**provenance, "reason": "Captured cache namespace encoding is unknown"}
                if namespace and not _opaque_namespace(namespace):
                    _fail("Captured namespace is not a normalized opaque vLLM salt domain")
            if original is not None and not bind_runtime_workload and namespace != _expected_namespace(original.cache_namespace):
                _fail("Captured native cache namespace differs from the supplied workload", "COMPARISON_MISMATCH")
            output = actual if actual else maximum
            if original is not None and original.input_tokens != prompt:
                _fail("Captured input length differs from the supplied workload", "COMPARISON_MISMATCH")
            if not bind_runtime_workload:
                if (original is None or original.arrival_ms != arrival or original.output_tokens != output
                        or original.priority != priority):
                    return experiment, None, {**provenance,
                        "reason": "Actual scheduler workload differs; enable explicit bind_runtime_workload for conditional replay"}
                spec = original
            else:
                if original is not None and original.timeout_ms is not None:
                    # A captured server protocol does not recover the caller's
                    # original relative timeout after changing its arrival.
                    return experiment, None, {**provenance, "reason": "Conditional replay cannot infer captured request timeout policy"}
                spec = RequestSpec(name, arrival, prompt, output, priority=priority,
                                   prompt_token_ids=token_ids, cache_namespace=namespace)
            specs[name] = spec
            descriptor_counts[name] = actual
            descriptor_statuses[name] = row["status"]
        for supplied in channel["events"]:
            if not isinstance(supplied, dict):
                _fail("Scheduler event must be an object")
            event = deepcopy(supplied)
            name = _text(event.get("request_id"), "event.request_id")
            if name not in specs:
                _fail("Scheduler event has no request descriptor")
            event["timestamp_ms"] = offset + _number(event.get("timestamp_ms"), "event.timestamp_ms")
            if event.get("replica_index", replica) != replica:
                _fail("Scheduler event changes request replica ownership")
            event["replica_index"] = replica
            events.append(event)
    if not specs:
        return experiment, None, {**provenance, "reason": "Scheduler capture has no requests"}
    for name in specs:
        related = [event for event in events if event["request_id"] == name]
        arrivals = [event for event in related if event.get("event") == "REQUEST_ARRIVAL"]
        finished = [event for event in related if event.get("event") == "REQUEST_FINISH"]
        if len(arrivals) != 1 or len(finished) != 1:
            return experiment, None, {**provenance, "reason": "Scheduler lifecycle lacks one ingress and terminal event per request"}
        tokens = [event for event in related if event.get("event") == "TOKEN_READY"]
        if len(tokens) != descriptor_counts[name] or finished[0].get("status") != descriptor_statuses[name]:
            _fail("Scheduler descriptors contradict their published lifecycle events")
    if not bind_runtime_workload and set(specs) != {request.request_id for request in experiment.workload
        if experiment.evaluation.measurement_window_ms is None or request.arrival_ms < experiment.evaluation.measurement_window_ms}:
        return experiment, None, {**provenance, "reason": "Scheduler cohort differs from the supplied workload"}
    effective = (replace(experiment, workload=tuple(sorted(specs.values(), key=lambda request: (request.arrival_ms, request.request_id))))
                 if bind_runtime_workload else experiment)
    validate_experiment(effective)
    events.sort(key=lambda event: event["timestamp_ms"])
    trace = {"schema_version": 1, "kind": "runtime_trace", "source": source,
        "timing_boundary": "server_token_ready_v1", "observed_until_ms": max(ends), "events": events,
        "identity": profile_identity(effective), "notes": {"instrumentation_source": "vllm_internal_scheduler",
            "observation_point": "scheduler_enqueue_to_engine_core_output_dispatch",
            "session_clock_origin_monotonic_ns": origin}}
    provenance.update(applied=bind_runtime_workload,
        original_workload_sha256=hashlib.sha256(_json(asdict(experiment)["workload"]).encode()).hexdigest(),
        replay_workload_sha256=hashlib.sha256(_json(asdict(effective)["workload"]).encode()).hexdigest(),
        cache_namespace_binding="vllm_nonempty_salt_sha256_v1", reason=None)
    return effective, trace, provenance


def _quarantine(reasons: list[str], replica: int, stage: int, iteration: int,
                records: list[dict], **extra: Any) -> dict:
    return {"replica_index": replica, "stage_index": stage, "iteration_id": iteration,
            "sample_id": records[0]["sample_id"] if records else None,
            "reasons": sorted(set(reasons)), **extra}


def normalize_vllm_capture(experiment: Experiment, captures: list[dict], *,
                           allow_test_fixture: bool = False,
                           runtime_trace: dict | None = None,
                           bind_runtime_workload: bool = False) -> dict:
    """Validate complete local rank groups and preserve unsupported raw evidence.

    Canonical stage datasets are per replica. An online observation requires an
    explicit complete server trace; runner output alone never supplies arrivals
    or terminal lifecycle events. This function does not rewrite the experiment.
    """
    validate_experiment(experiment)
    if type(allow_test_fixture) is not bool or type(bind_runtime_workload) is not bool:
        _fail("Fixture and workload-binding options must be boolean")
    if not isinstance(captures, list) or not captures or len(captures) > _MAX_CHANNELS:
        _fail("captures must be a bounded nonempty array")
    _json(captures)
    runners = [value for value in captures if isinstance(value, dict) and value.get("kind") == "vllm_internal_capture"]
    schedulers = [value for value in captures if isinstance(value, dict) and value.get("kind") == "vllm_scheduler_capture"]
    if len(runners) + len(schedulers) != len(captures) or not runners:
        _fail("Capture input must contain runner channels and only recognized runner/scheduler exports")
    if schedulers and runtime_trace is not None:
        _fail("An explicit runtime_trace cannot replace supplied scheduler lifecycle channels")
    validated = [_header(experiment, value, allow_test_fixture) for value in runners]
    validated_schedulers = [_scheduler_header(experiment, value, allow_test_fixture) for value in schedulers]
    all_channels = [*validated, *validated_schedulers]
    sessions = {capture["session_id"] for capture, _ in all_channels}
    versions = {capture["runtime_version"] for capture, _ in all_channels}
    clocks = {capture["clock_domain"] for capture, _ in all_channels}
    if len(sessions) != 1 or len(versions) != 1 or len(clocks) != 1:
        _fail("Capture channels must share session, runtime version and clock domain")
    actual_hashes = {capture["identity"]["runtime"].get("runtime_config_sha256", "")
                     for capture, _ in validated} - {""}
    if len(actual_hashes) > 1:
        _fail("Capture channels have incompatible actual runtime configurations", "CALIBRATION_MISMATCH")
    actual_hash = next(iter(actual_hashes), "")
    original_hash = getattr(experiment.calibration, "runtime_config_sha256", "")
    effective = (replace(experiment, calibration=replace(experiment.calibration, runtime_config_sha256=actual_hash))
                 if not original_hash and actual_hash else experiment)
    trace_source = None
    if runtime_trace is not None:
        if not isinstance(runtime_trace, dict):
            _fail("runtime_trace must be an object")
        trace_source = _source(runtime_trace.get("source"), allow_test_fixture)
    fixture = (any(source["kind"] == "test_fixture" for _, source in all_channels)
               or trace_source is not None and trace_source["kind"] == "test_fixture")
    source = {"kind": "test_fixture" if fixture else "measured",
              "reference": f"vLLM internal capture session {next(iter(sessions))}"}
    groups: dict[tuple[int, int, int], dict[int, dict]] = {}
    channels: set[tuple[int, int, int]] = set()
    overlaps: set[tuple[int, int, int, int]] = set()
    count = 0
    inputs = []
    for capture, _ in validated:
        replica, stage, rank = capture["replica_index"], capture["stage_index"], capture["tp_rank"]
        coordinate = replica, stage, rank
        if coordinate in channels:
            _fail("Duplicate capture channel for one replica/stage/TP rank")
        channels.add(coordinate)
        if "_capture_input" in capture:
            inputs.append(deepcopy(capture["_capture_input"]))
        else:
            inputs.append({"sha256": hashlib.sha256(_json(capture).encode()).hexdigest(), "scope": "canonical_json"})
        completed = []
        for supplied in capture["records"]:
            count += 1
            if count > _MAX_RECORDS:
                _fail("Capture record count exceeds the bounded limit")
            record = _record(supplied, stage, experiment.strategy.pipeline_parallel_size)
            iteration = record["iteration_id"]
            ranks = groups.setdefault((replica, stage, iteration), {})
            if rank in ranks:
                _fail("Duplicate iteration in one rank channel")
            ranks[rank] = record
            if record["status"] == "COMPLETED":
                completed.append(record)
        ordered = sorted(completed, key=lambda record: record["start_monotonic_ns"])
        previous = None
        for record in ordered:
            if previous is not None and record["start_monotonic_ns"] < previous["end_monotonic_ns"]:
                overlaps.add((*coordinate, record["iteration_id"]))
                overlaps.add((*coordinate, previous["iteration_id"]))
            if previous is None or record["end_monotonic_ns"] > previous["end_monotonic_ns"]:
                previous = record
    unbound_channels = {(capture["replica_index"], capture["stage_index"], capture["tp_rank"])
                        for capture, _ in validated if not capture["identity"]["runtime"].get("runtime_config_sha256")
                        or capture["metadata"].get("runtime_execution_config") is None}
    accepted: dict[int, list[dict]] = {}
    quarantined = []
    expected_ranks = set(range(experiment.strategy.tensor_parallel_size))
    signatures: dict[tuple[int, int], str] = {}
    for (replica, stage, iteration), ranked in sorted(groups.items()):
        records = [ranked[rank] for rank in sorted(ranked)]
        reasons = []
        if set(ranked) != expected_ranks:
            reasons.append("missing_tp_rank")
        if any(record["status"] != "COMPLETED" for record in records):
            reasons.append("incomplete_or_failed_iteration")
        complete = [record for record in records if record["status"] == "COMPLETED"]
        if complete:
            shape_signatures = {_shape_signature(record) for record in complete}
            if len(shape_signatures) != 1:
                _fail("TP ranks disagree on the captured batch descriptor", "CALIBRATION_MISMATCH")
            signature = next(iter(shape_signatures))
            across_stage = signatures.get((replica, iteration))
            if across_stage is not None and across_stage != signature:
                _fail("PP stage counters do not identify the same captured iteration", "CALIBRATION_MISMATCH")
            signatures[replica, iteration] = signature
        for rank, record in ranked.items():
            if record["status"] != "COMPLETED":
                continue
            if not record["forward_complete"]:
                reasons.append("forward_not_complete")
            if (replica, stage, rank) in unbound_channels:
                reasons.append("runtime_execution_configuration_unobserved")
            required = stage == experiment.strategy.pipeline_parallel_size - 1
            expected_boundary = "runner_execute_to_sample_host_ready" if required else "runner_execute_host_ready"
            if record["capture_boundary"] != expected_boundary:
                reasons.append("unsupported_capture_boundary")
            if required and not record["sampling_complete"]:
                reasons.append("sampling_not_complete")
            if not required and record["sampling_complete"]:
                reasons.append("unexpected_sampling_on_nonfinal_stage")
            if (*((replica, stage, rank)), iteration) in overlaps:
                reasons.append("overlapping_host_service_intervals")
            if record.get("async_output", False):
                reasons.append("async_consumption_is_not_exclusive_stage_service")
            if record.get("profile_eligible") is False:
                reasons.append("hook_marked_sample_ineligible")
            if experiment.strategy.pipeline_parallel_size > 1:
                transport = record.get("transport_metadata")
                required_directions = {"incoming"} | ({"outgoing"} if stage < experiment.strategy.pipeline_parallel_size - 1 else set())
                if not isinstance(transport, dict) or required_directions - transport.keys():
                    reasons.append("pp_transport_scope_unproven")
                else:
                    for direction, info in transport.items():
                        if direction not in {"incoming", "outgoing"} or not isinstance(info, dict):
                            _fail("Captured PP transport metadata has an invalid direction or descriptor")
                        if (info.get("profile_transport_compatible") is not True
                                or info.get("incoming_communication_excluded") is not True
                                or info.get("lazy_incoming") is not False
                                or info.get("default_tp_sliced_p2p_receiver_allgather") is not False
                                or info.get("modeled_transport") != "one_unpadded_full_hidden_tensor_per_rank"):
                            reasons.append("pp_transport_not_isolated_or_not_modeled")
                            continue
                        if direction == "incoming" and stage == 0 and info.get("object_type") == "builtins.NoneType":
                            continue
                        if (info.get("materialized_tensor_keys") != ["hidden_states"]
                                or info.get("materialized_shape") != [sum(record["query_tokens"]), experiment.strategy.model.hidden_size]):
                            reasons.append("pp_intermediate_payload_not_modeled")
            if any(count > 1 for count in record.get("sampled_token_counts", [])):
                reasons.append("multi_token_speculative_output_unsupported")
            if record["workspace_peak_bytes"] is not None:
                reasons.append("workspace_has_no_runtime_attribution")
            if record.get("resumed_request_ids"):
                reasons.append("resumed_recompute_boundary_unsupported")
            if record.get("output_history_lengths") is not None:
                roles = record.get("request_phases", [record["phase"]] * len(record["request_ids"]))
                if any(history and role == "prefill" for history, role in zip(record["output_history_lengths"], roles)):
                    reasons.append("prefill_with_existing_output_history")
        if reasons:
            quarantined.append(_quarantine(reasons, replica, stage, iteration, records,
                                          present_tp_ranks=sorted(ranked), expected_tp_ranks=sorted(expected_ranks)))
            continue
        representative = records[0]
        key = shape_key(representative["phase"], representative["query_tokens"],
                        representative["context_lengths"], stage,
                        request_phases=representative.get("request_phases"))
        row = {"record_id": f"replica-{replica}/stage-{stage}/{representative['sample_id']}",
               "group_id": f"vllm/{next(iter(sessions))}/r{replica}/{hashlib.sha256(key.encode()).hexdigest()[:16]}",
               "phase": representative["phase"], "stage_index": stage,
               "query_tokens": list(representative["query_tokens"]),
               "context_lengths": list(representative["context_lengths"]),
               "duration_ms": max(record["duration_ms"] for record in records), "workspace_peak_bytes": None}
        if representative["phase"] == "mixed":
            row["request_phases"] = list(representative["request_phases"])
        accepted.setdefault(replica, []).append(row)
    datasets: list[dict[str, Any]] = []
    for replica, records in sorted(accepted.items()):
        dataset = validate_dataset({"schema_version": 1,
            "dataset_id": f"vllm-{next(iter(sessions))}-replica-{replica}",
            "identity": profile_identity(replica_experiment(effective, replica)), "source": source,
            "records": records, "notes": {"collector": "vllm_internal_runner",
                "runtime_version": next(iter(versions)), "api_revision": _REVISION,
                "rank_reduction": "maximum_synchronized_per_rank_host_duration",
                "absolute_cross_rank_time_subtraction": False, "workspace_policy": "unknown_without_runtime_attribution",
                "capture_inputs": inputs, "excluded_groups": len(quarantined)}},
            allow_test_fixture=allow_test_fixture)
        datasets.append({"replica_index": replica, "dataset": dataset})
    observation = None
    online_reason: str | None = "Runner records do not establish scheduler arrivals and terminal lifecycle"
    workload_binding: dict[str, Any] = {"applied": False, "requested": bind_runtime_workload}
    if validated_schedulers:
        scheduler_channels = [capture for capture, _ in validated_schedulers]
        effective, runtime_trace, workload_binding = _scheduler_replay(effective, scheduler_channels, source,
            bind_runtime_workload=bind_runtime_workload)
        online_reason = workload_binding.get("reason")
        for capture in scheduler_channels:
            if "_capture_input" in capture:
                inputs.append(deepcopy(capture["_capture_input"]))
            else:
                inputs.append({"sha256": hashlib.sha256(_json(capture).encode()).hexdigest(), "scope": "canonical_json"})
    if runtime_trace is not None:
        trace = deepcopy(runtime_trace)
        if fixture:
            trace["source"] = source
        normalized = normalize_runtime_trace(effective, trace, allow_test_fixture=allow_test_fixture)
        observation = normalized["observation"]
        if validated_schedulers and effective.evaluation.measurement_window_ms is None:
            for name in ("throughput_requests_per_s", "throughput_output_tokens_per_s",
                         "goodput_requests_per_s", "goodput_output_tokens_per_s"):
                observation["metrics"][name] = None
            observation["notes"]["rate_denominator_status"] = "unavailable_without_explicit_measurement_window"
            observation["notes"]["capture_close_is_not_cohort_rate_end"] = True
        online_reason = None
    expected_channels = {(replica, stage, rank)
                         for replica in range(experiment.strategy.replicas)
                         for stage in range(experiment.strategy.pipeline_parallel_size)
                         for rank in range(experiment.strategy.tensor_parallel_size)}
    missing_channels = sorted(expected_channels - channels)
    accepted_stages = {(replica, record["stage_index"])
                       for replica, records in accepted.items() for record in records}
    expected_stages = {(replica, stage) for replica in range(experiment.strategy.replicas)
                       for stage in range(experiment.strategy.pipeline_parallel_size)}
    return {"schema_version": 1, "status": "COMPLETED", "kind": "vllm_hook_collection",
        "source": source, "experiment": asdict(effective), "datasets": datasets,
        **({"dataset": datasets[0]["dataset"]} if experiment.strategy.replicas == 1 and datasets else {}),
        "observation": observation, "canonical_comparable": observation is not None,
        "canonical_stage_ready": (bool(datasets) and not missing_channels and accepted_stages == expected_stages
                                  and not quarantined and not any(capture.get("capture_errors") for capture, _ in validated)),
        "canonical_stage_records_available": bool(datasets),
        "online_readiness": {"ready": observation is not None, "reason": online_reason},
        "capture_summary": {"channels": len(captures), "runner_channels": len(runners),
                            "scheduler_channels": len(schedulers), "raw_records": count,
                            "accepted_groups": sum(len(item["dataset"]["records"]) for item in datasets),
                            "quarantined_groups": len(quarantined),
                            "missing_channels": [{"replica_index": replica, "stage_index": stage, "tp_rank": rank}
                                                 for replica, stage, rank in missing_channels]},
        "quarantined": quarantined, "raw_records": deepcopy(captures),
        "provenance": {"session_id": next(iter(sessions)), "runtime_version": next(iter(versions)),
                       "api_revision": _REVISION, "clock_domain": next(iter(clocks)), "capture_inputs": inputs,
                       "runtime_config_binding": {"declared_sha256": original_hash,
                                                  "observed_sha256": actual_hash,
                                                  "derived_from_capture": bool(not original_hash and actual_hash)},
                       "runtime_workload_binding": workload_binding},
        "hardware_validated": False,
        "limitations": ["Internal hooks perturb timing; no live GPU accuracy is established by collection.",
                        "Only complete synchronized exclusive stage boundaries become step records.",
                        "Async consumption intervals, allocator peaks and runner tokens are not inferred stage/Workspace/lifecycle evidence."]}
