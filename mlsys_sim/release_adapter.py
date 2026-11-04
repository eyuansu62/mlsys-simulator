"""Read-only, bounded audits of published request-service reference packages.

These packages contain client measurements, not whole-stage Profiles. This
adapter never executes bundled code, reads prompt/output text, or manufactures
token events and memory attribution from coarser measurements.
"""

from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, NoReturn

from .qwen3 import model_from_qwen3_config, normalized_qwen3_architecture
from .schema import SimulationError


_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_EXPANDED_BYTES = 128 * 1024 * 1024
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_ROWS = 250_000
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:/+\-]{1,512}\Z")
_UNITS = {"ms", "s", "byte", "token", "count", "%", "tokens/s", "requests/s", "MiB", "GiB"}
_NUMERIC_COLUMN = re.compile(r"(?:^|_)(?:ms|s|tokens?|bytes?|mib|gib|pct|concurrency|rep|count)(?:_|$)")
_NUMERIC_NAMES = {"n_measure", "n_ok", "n_reps", "request_no", "order", "num_gpu_blocks",
                  "kv_num_gpu_blocks", "preemptions", "preemptions_total", "running_requests", "waiting_requests"}


def _fail(message: str, code: str = "INVALID_REFERENCE_DATA") -> NoReturn:
    raise SimulationError(code, message)


def _relative(name: Any) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or PurePosixPath(name).is_absolute() or any(part in {".", ".."} for part in name.split("/"))
            or any(not part for part in name.split("/")) or re.match(r"^[A-Za-z]:", name)):
        _fail("Reference paths must be safe relative POSIX paths", "UNSAFE_REFERENCE_PATH")
    return name


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        _fail(f"{name} must be a bounded identifier")
    return value


def _number(value: Any, name: str, *, optional: bool = False, integer: bool = False,
            minimum: float = 0) -> Any:
    if optional and value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        _fail(f"{name} must be a finite number")
    if integer:
        try:
            exact = Decimal(str(value))
        except InvalidOperation:
            _fail(f"{name} must be a finite integer")
        if not exact.is_finite() or not minimum <= exact <= 2**63 - 1 or exact != exact.to_integral_value():
            _fail(f"{name} must be an integer >= {minimum} in the signed 64-bit range")
        return int(exact)
    try:
        result = float(value)
    except (ValueError, OverflowError):
        _fail(f"{name} must be a finite number")
    if not math.isfinite(result) or result < minimum:
        _fail(f"{name} must be finite and >= {minimum}")
    return result


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _safe_fields(value: dict, keys: tuple[str, ...]) -> dict:
    return {key: value[key] for key in keys if key in value}


def _unique_pairs(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON key in reference data")
        result[key] = value
    return result


class _Reader:
    def __init__(self, path: Path):
        if path.is_symlink() or not path.is_dir():
            _fail("Reference import expects an extracted directory, not an archive or symlink",
                  "UNSAFE_REFERENCE_PATH")
        self.root = path.resolve()
        self.files: dict[str, dict] = {}
        self.total = 0
        self.tables: list[dict] = []

    def path(self, name: str) -> Path:
        name = _relative(name)
        target = self.root
        for part in PurePosixPath(name).parts:
            target = target / part
            if target.is_symlink():
                _fail("Reference data may not follow symlinks", "UNSAFE_REFERENCE_PATH")
        if not target.resolve().is_relative_to(self.root):
            _fail("Reference path escapes its input directory", "UNSAFE_REFERENCE_PATH")
        return target

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def raw(self, name: str) -> bytes:
        target = self.path(name)
        if not target.is_file():
            _fail(f"Required reference file is missing: {name}")
        if target.stat().st_size > _MAX_FILE_BYTES:
            _fail(f"Reference file exceeds the input limit: {name}", "REFERENCE_SIZE_LIMIT")
        with target.open("rb") as stream:
            raw = stream.read(_MAX_FILE_BYTES + 1)
        if len(raw) > _MAX_FILE_BYTES:
            _fail(f"Reference file exceeds the input limit: {name}", "REFERENCE_SIZE_LIMIT")
        if name not in self.files:
            self.total += len(raw)
            if self.total > _MAX_TOTAL_BYTES:
                _fail("Reference inputs exceed the total size limit", "REFERENCE_SIZE_LIMIT")
        self.files[name] = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return raw

    def text(self, name: str) -> str:
        raw = self.raw(name)
        if name.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(_MAX_EXPANDED_BYTES + 1)
            if len(raw) > _MAX_EXPANDED_BYTES:
                _fail(f"Expanded reference data exceeds the limit: {name}", "REFERENCE_SIZE_LIMIT")
        try:
            return raw.decode("utf-8")
        except UnicodeError as exc:
            raise SimulationError("INVALID_REFERENCE_DATA", f"Reference data is not UTF-8: {name}") from exc

    def json(self, name: str) -> Any:
        def nonfinite(_):
            _fail(f"Non-finite JSON value in {name}")

        value = json.loads(self.text(name), object_pairs_hook=_unique_pairs, parse_constant=nonfinite)
        def finite(item, depth=0):
            if depth > 40:
                _fail("Reference metadata is nested too deeply")
            if isinstance(item, float) and not math.isfinite(item):
                _fail(f"Non-finite JSON value in {name}")
            if isinstance(item, dict):
                for nested in item.values():
                    finite(nested, depth + 1)
            elif isinstance(item, list):
                for nested in item:
                    finite(nested, depth + 1)
        finite(value)
        return value

    def csv(self, name: str, required: tuple[str, ...]) -> list[dict]:
        parser = csv.DictReader(io.StringIO(self.text(name)))
        fields = parser.fieldnames or []
        if (len(fields) > 128 or len(fields) != len(set(fields)) or not set(required) <= set(fields)
                or any(not re.fullmatch(r"[A-Za-z0-9_#]{1,128}", field) for field in fields)):
            _fail(f"Invalid or missing CSV columns in {name}")
        rows: list[dict] = []
        for row in parser:
            if None in row or len(rows) >= _MAX_ROWS:
                _fail(f"Malformed or excessive rows in {name}")
            for key, value in row.items():
                if value not in (None, "") and (_NUMERIC_COLUMN.search(key) or key in _NUMERIC_NAMES):
                    _number(value, f"{name}:{key}", minimum=-math.inf)
            rows.append(row)
        self.tables.append({"path": name, "columns": fields, "rows": len(rows), "units": {}})
        return rows


def _source(manifest: dict, metadata: dict, allow_fixture: bool, fingerprint: str) -> dict:
    kinds = []
    for document in (manifest, metadata):
        source = document.get("source")
        if isinstance(source, dict) and "kind" in source:
            kinds.append(source["kind"])
        if "source_kind" in document:
            kinds.append(document["source_kind"])
    if any(kind not in ("measured", "test_fixture") for kind in kinds):
        _fail("Reference source kind must be measured or test_fixture")
    kind = "test_fixture" if "test_fixture" in kinds else "measured"
    if kind == "test_fixture" and not allow_fixture:
        _fail("Reference fixtures require explicit opt-in", "TEST_FIXTURE_NOT_ALLOWED")
    return {"kind": kind, "reference": f"external-reference-manifest-sha256:{fingerprint}"}


def _manifest_hashes(manifest: dict, release: bool) -> dict[str, str]:
    entries = manifest.get("files") if release else manifest.get("sha256")
    result: dict[str, str] = {}
    if release:
        if not isinstance(entries, list):
            _fail("Release MANIFEST requires files entries")
        pairs = [(item.get("path"), item.get("sha256")) for item in entries if isinstance(item, dict)]
        if len(pairs) != len(entries):
            _fail("Invalid release file manifest")
    else:
        if not isinstance(entries, dict):
            _fail("Legacy manifest requires a sha256 map")
        pairs = list(entries.items())
    for path, digest in pairs:
        path = _relative(path)
        if path in result or not isinstance(digest, str) or not _SHA.fullmatch(digest):
            _fail("Invalid or duplicate file checksum in reference manifest")
        result[path] = digest.lower()
    return result


def _checksums(reader: _Reader) -> dict[str, str]:
    if not reader.exists("CHECKSUMS.sha256"):
        return {}
    result = {}
    for line in reader.text("CHECKSUMS.sha256").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA.fullmatch(parts[0]):
            _fail("Invalid CHECKSUMS.sha256 line")
        path = _relative(parts[1].lstrip(" *"))
        if path in result:
            _fail("Duplicate CHECKSUMS.sha256 path")
        result[path] = parts[0].lower()
    return result


def _check_hashes(reader: _Reader, expected: dict[str, str], checksums: dict[str, str]) -> dict:
    verified = []
    mismatches = []
    for name, item in sorted(reader.files.items()):
        for label, declared in (("manifest", expected), ("checksums", checksums)):
            if name not in declared:
                continue
            if declared[name] == item["sha256"]:
                verified.append({"path": name, "index": label})
            else:
                mismatches.append({"path": name, "index": label,
                                   "expected_sha256": declared[name], "actual_sha256": item["sha256"]})
    controls = {"MANIFEST.json", "manifest.json", "CHECKSUMS.sha256"}
    unverified = sorted(set(reader.files) - set(expected) - set(checksums) - controls)
    return {"scope": "known_data_files_and_readme_only", "verified": verified, "mismatches": mismatches,
            "checksums_present": bool(checksums), "manifest_entries": len(expected),
            "unverified_data_files": unverified,
            "unread_manifest_entries": len(set(expected) - reader.files.keys()),
            "passed": not mismatches and not unverified, "source_bytes_repaired": False}


def _model_summary(name: str, metadata: dict, config: dict | None) -> dict:
    result = {"name": name, **_safe_fields(metadata, (
        "served_name", "repository", "revision", "hf_revision", "config_json_sha256",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
        "hidden_size", "torch_dtype", "kv_bytes_per_token_bf16", "decode_attention_impl"))}
    reasons = []
    if config is not None:
        result.update(_safe_fields(config, ("num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "torch_dtype",
            "tie_word_embeddings", "use_sliding_window", "sliding_window")))
        result["quantization"] = config.get("quantization_config") is not None
        revision = metadata.get("hf_revision", metadata.get("revision"))
        served = metadata.get("served_name", name)
        try:
            if not isinstance(revision, str):
                raise SimulationError("INVALID_QWEN3_CONFIG", "A declared model revision is required")
            model = model_from_qwen3_config(config, name=served, revision=revision)
        except SimulationError as exc:
            result["model_import_error_code"] = exc.code
            if config.get("model_type") != "qwen3" or config.get("architectures") != ["Qwen3ForCausalLM"]:
                reasons.append("DENSE_QWEN3_FAMILY_NOT_DECLARED")
            if "tie_word_embeddings" not in config:
                reasons.append("TIED_EMBEDDING_STATE_NOT_PROVIDED")
            if not revision:
                reasons.append("MODEL_REVISION_NOT_PROVIDED")
            if (config.get("use_sliding_window") is True or
                    config.get("use_sliding_window") is not False and config.get("sliding_window") is not None):
                reasons.append("SLIDING_WINDOW_NOT_SUPPORTED")
            if result["quantization"]:
                reasons.append("QUANTIZED_FORMAT_NOT_SUPPORTED")
            if config.get("dtype", config.get("torch_dtype")) not in {"float16", "bfloat16", "float32", "float64"}:
                reasons.append("STORAGE_DTYPE_NOT_SUPPORTED_OR_DECLARED")
            if not reasons:
                reasons.append("QWEN3_MODEL_CONFIG_INVALID_OR_UNSUPPORTED")
        else:
            from dataclasses import asdict

            result["model_spec"] = asdict(model)
            result["query_projection_size"] = model.query_projection_size
            result["normalized_architecture"] = normalized_qwen3_architecture(config)
            result["architecture_config_sha256"] = model.architecture_config_sha256
    else:
        reasons.extend(("MODEL_GEOMETRY_NOT_PROVIDED", "TIED_EMBEDDING_STATE_NOT_PROVIDED"))
    result["simulator_model_compatible"] = not reasons
    result["geometry_supported"] = not reasons
    result["compatibility_scope"] = "declared_geometry_and_storage_only"
    result["compatibility_reasons"] = reasons
    return result


def _release_requests(rows: list[dict]) -> list[dict]:
    result = []
    seen = set()
    identifiers = ("uid", "matrix", "run_id", "request_id", "config_id", "model", "trace_id",
                   "arrival_pattern", "feature_scope", "protocol")
    integers = ("input_tokens", "prompt_tokens", "output_tokens_target", "output_tokens_actual", "cached_tokens")
    times = ("arrival_time_ms", "send_time_ms", "first_token_time_ms", "last_token_time_ms",
             "ttft_ms", "ttft_from_send_ms", "tpot_ms", "e2e_ms", "ttfc_ms")
    for row in rows:
        record: dict[str, Any] = {key: _identifier(row[key], key) for key in identifiers if key in row}
        uid = record["uid"]
        if uid in seen:
            _fail("Duplicate reference request UID")
        seen.add(uid)
        phase, status = row["phase"], row["status"]
        if phase not in ("warmup", "measure") or status not in (
                "ok", "timeout_request", "timeout_round", "error", "not_sent", "http_error",
                "stream_error", "conn_error", "parse_error"):
            _fail("Unsupported request phase or status")
        record.update(phase=phase, status=status, integrity_issue=bool(row.get("integrity_note")))
        for key in integers:
            record[key] = _number(row.get(key), key, integer=True,
                                  minimum=1 if key in {"input_tokens", "output_tokens_target"} else 0,
                                  optional=status != "ok" and key not in {"input_tokens", "output_tokens_target"})
        for key in times:
            record[key] = _number(row.get(key), key, optional=key == "ttfc_ms" or status != "ok")
        for key in ("concurrency", "n_content_chunks", "rep"):
            if key in row:
                record[key] = _number(row[key], key, optional=True, integer=True)
        if status == "ok":
            arrival, send, first, last = (record[key] for key in times[:4])
            if any(left > right + .002 for left, right in ((arrival, send), (send, first), (first, last))):
                _fail("Request client timestamps are not monotonic")
            for actual, derived in ((record["ttft_ms"], first - arrival),
                                    (record["ttft_from_send_ms"], first - send),
                                    (record["e2e_ms"], last - arrival)):
                if abs(actual - derived) > .003:
                    _fail("Request timing does not match its declared client boundary")
            output = record["output_tokens_actual"]
            if output < 1 or record["cached_tokens"] > record["prompt_tokens"]:
                _fail("Invalid successful request token counts")
            if output > 1 and abs(record["tpot_ms"] - (last - first) / (output - 1)) > .003:
                _fail("TPOT does not match usage count and client content endpoints")
        record["first_content_time_ms"] = record.pop("first_token_time_ms")
        record["last_content_time_ms"] = record.pop("last_token_time_ms")
        result.append(record)
    return result


def _timestamps(reader: _Reader, records: list[dict]) -> dict:
    name = "02_measured/token_timestamps.jsonl.gz"
    if not reader.exists(name):
        return {"present": False, "kind": "not_provided", "token_events_available": False}
    by_uid = {row["uid"]: row for row in records}
    seen = set()
    missing = differing = count = 0
    for line in reader.text(name).splitlines():
        if not line.strip():
            continue
        if count >= _MAX_ROWS:
            _fail("Excessive timestamp rows", "REFERENCE_SIZE_LIMIT")
        row = _object(json.loads(line, object_pairs_hook=_unique_pairs), "timestamp row")
        uid = _identifier(row.get("uid"), "timestamp uid")
        if uid in seen or uid not in by_uid:
            _fail("Duplicate or unpaired timestamp UID")
        seen.add(uid)
        count += 1
        offsets = row.get("chunk_offsets_ms_from_send")
        request = by_uid[uid]
        if offsets is None:
            if not request["integrity_issue"]:
                _fail("Missing chunk offsets require a published integrity note")
            missing += 1
            continue
        if not isinstance(offsets, list) or len(offsets) > 100_000:
            _fail("Invalid chunk offset array")
        values = [_number(value, "chunk offset") for value in offsets]
        if values != sorted(values):
            _fail("Chunk offsets are not monotonic")
        if request.get("n_content_chunks") is not None and len(values) != request["n_content_chunks"]:
            _fail("Chunk offset count differs from the declared content chunk count")
        differing += len(values) != request["output_tokens_actual"]
    reader.tables.append({"path": name, "columns": ["uid", "chunk_offsets_ms_from_send"],
                          "rows": count, "units": {"chunk_offsets_ms_from_send": "ms"}})
    return {"present": True, "kind": "client_nonempty_content_sse_chunks", "rows": count,
            "missing_offset_arrays": missing, "unpaired_request_rows": len(by_uid) - len(seen),
            "chunk_count_differs_from_output_tokens": differing, "token_events_available": False}


def _counter(rows: list[dict], key: str) -> dict:
    return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))


def _open_loop_metadata(reader: _Reader) -> list[dict]:
    directory = reader.path("01_workloads/open_loop")
    if not directory.is_dir():
        return []
    names = sorted(path.name for path in directory.glob("*.meta.json"))
    if len(names) > 64:
        _fail("Too many open-loop metadata files", "REFERENCE_SIZE_LIMIT")
    result = []
    for name in names:
        data = _object(reader.json(f"01_workloads/open_loop/{name}"), "open-loop metadata")
        process = _object(data.get("process"), "arrival process")
        current = {"name": _identifier(data.get("name"), "arrival metadata name"),
                   "trace_id": _identifier(data.get("trace_id"), "arrival trace_id"),
                   "n_requests": _number(data.get("n_requests"), "n_requests", integer=True, minimum=1),
                   "arrival_window_s": _number(data.get("arrival_window_s"), "arrival_window_s"),
                   "process_type": _identifier(process.get("type"), "arrival process type")}
        for key in ("rate_req_s", "burst_size", "burst_gap_s", "intra_burst_ms", "seed"):
            if key in process:
                current[key] = _number(process[key], key, optional=True)
        result.append(current)
    return result


def _release(reader: _Reader) -> tuple[dict, dict, list[dict], dict, dict]:
    manifest = _object(reader.json("MANIFEST.json"), "release manifest")
    hardware = _object(reader.json("00_environment/hardware.json"), "hardware")
    software = _object(reader.json("00_environment/software.json"), "software")
    models = _object(reader.json("00_environment/models.json"), "models")
    configs = _object(reader.json("00_environment/server_configs.json"), "server configs")
    schema = _object(reader.json("02_measured/schema.json"), "schema")
    if not models or not configs or not isinstance(hardware.get("target_gpu"), dict):
        _fail("Release hardware/model/config metadata is incomplete")
    if not software.get("vllm_actual") or not software.get("torch"):
        _fail("Release requires actual Runtime and framework versions")
    rows = reader.csv("02_measured/requests.csv.gz", ("uid", "matrix", "run_id", "request_id", "phase",
        "input_tokens", "prompt_tokens", "output_tokens_target", "output_tokens_actual", "cached_tokens",
        "arrival_time_ms", "send_time_ms", "first_token_time_ms", "last_token_time_ms", "ttft_ms",
        "ttft_from_send_ms", "tpot_ms", "e2e_ms", "status"))
    records = _release_requests(rows)
    runs = reader.csv("02_measured/runs.csv", ("run_id", "matrix", "protocol", "mode"))
    cells = reader.csv("02_measured/cells.csv", ("matrix",))
    run_ids = {_identifier(row["run_id"], "run_id") for row in runs}
    if len(run_ids) != len(runs) or any(row["run_id"] not in run_ids for row in records):
        _fail("Duplicate run IDs or request rows without an associated run")
    model_summaries = []
    for name, value in models.items():
        metadata = _object(value, "model metadata")
        served = metadata.get("served_name", name)
        if not isinstance(served, str) or not _IDENTIFIER.fullmatch(served):
            _fail("Model served_name must be a safe identifier")
        path = f"00_environment/model_configs/{served}/config.json"
        config = _object(reader.json(path), "model config") if reader.exists(path) else None
        model_summaries.append(_model_summary(name, metadata, config))
    gpu = hardware["target_gpu"]
    hardware_summary: dict[str, Any] = {"vendor": "AMD" if any("AMD" in str(name) for name in gpu.get("name", [])) else "unknown",
        "target_gpu": _safe_fields(gpu, ("name", "arch", "pci", "pci_device_id", "vram_total_mib",
                                         "hip_visible_devices")),
        "host": _safe_fields(hardware.get("host", {}), ("os",)),
        "parallelism": _safe_fields(hardware.get("parallelism", {}),
                                     ("gpus", "tensor_parallel", "pipeline_parallel", "inter_gpu_communication_reason"))}
    hardware_summary["target_gpu"]["torch_enumeration"] = [
        _safe_fields(_object(item, "device enumeration"), ("idx", "name", "arch", "vram", "pci_bus_id"))
        for item in gpu.get("torch_enumeration", [])]
    hardware_summary["host"]["kernel_amdgpu_by_launch"] = [
        _safe_fields(_object(item, "host version"), ("amdgpu_version", "kernel"))
        for item in hardware.get("host", {}).get("kernel_amdgpu_by_launch", [])]
    software_summary = _safe_fields(software, ("vllm_actual", "torch", "hip", "python_in_container",
        "client_python", "vllm_engine", "attention_backends_in_logs", "image_lock_ok_all_launches",
        "vllm_version_lock_ok_all_launches"))
    software_summary["runtime_version_priority"] = "actual_vllm_over_image_tag"
    timestamps = _timestamps(reader, records)
    published = {}
    if reader.exists("05_provenance/integrity_check.json"):
        item = _object(reader.json("05_provenance/integrity_check.json"), "published integrity")
        issues = item.get("known_issue_lines", item.get("unparsable_lines", []))
        published = {"passed": item.get("passed"), "known_issue_count": len(issues),
            "measured_issue_count": sum(issue.get("measure_record") is True for issue in issues if isinstance(issue, dict))}
    optional_views = (("03_calibration_views/service_time_c1.csv", ("run_id", "ttft_ms", "tpot_ms")),
                      ("03_calibration_views/decode_context_c1.csv", ("model",)),
                      ("03_calibration_views/memory_capacity.csv", ("model",)),
                      ("03_calibration_views/online_validation_index.csv", ("run_id",)),
                      ("00_environment/memory_ledger.csv", ("model",)))
    for path, required in optional_views:
        if reader.exists(path):
            reader.csv(path, required)
    workload_protocols = []
    if reader.exists("01_workloads/closed_loop_specs.json"):
        specs = reader.json("01_workloads/closed_loop_specs.json")
        if not isinstance(specs, list):
            _fail("Closed-loop specifications must be an array")
        workload_protocols = [_safe_fields(_object(spec, "closed-loop specification"),
            ("matrix", "protocol", "concurrency", "repeats", "reset_prefix_cache_after_warmup",
             "measurement_mode", "metrics_poll_s", "request_timeout_s", "round_timeout_s")) for spec in specs]
    coverage = reader.json("03_calibration_views/coverage.json") if reader.exists("03_calibration_views/coverage.json") else {}
    counts = {"requests": len(records), "measure": sum(row["phase"] == "measure" for row in records),
              "warmup": sum(row["phase"] == "warmup" for row in records), "runs": len(runs), "cells": len(cells)}
    mismatches = []
    tables = schema.get("tables", {})
    for table in reader.tables:
        declared = tables.get(table["path"], {}) if isinstance(tables, dict) else {}
        if "rows" in declared and declared["rows"] != table["rows"]:
            mismatches.append({"path": table["path"], "declared_rows": declared["rows"], "actual_rows": table["rows"]})
        table["units"] = {column["name"]: column["unit"] for column in declared.get("columns", [])
                          if isinstance(column, dict) and column.get("name") in table["columns"]
                          and column.get("unit") in _UNITS}
    details = {"format": "mlsys_9070xt_v1_release", "hardware": hardware_summary,
        "software": software_summary, "models": model_summaries, "counts": counts,
        "protocols": {"matrix": _counter(runs, "matrix"), "mode": _counter(runs, "mode"),
            "protocol": _counter(runs, "protocol"), "arrival_pattern": _counter(runs, "arrival_pattern"),
            "closed_loop": workload_protocols, "closed_loop_replay_scope": "conditional_measured_arrival_replay_only",
            "open_loop": _open_loop_metadata(reader),
            "throughput_window": "last_content_end_minus_first_actual_send"},
        "integrity": {"published": published, "count_mismatches": mismatches,
            "request_rows_with_integrity_note": sum(row["integrity_issue"] for row in records), "timestamps": timestamps},
        "coverage": _safe_fields(_object(coverage, "coverage"), ("input_tokens_overall", "uncached_tokens_overall",
            "output_tokens_targets", "max_context_tokens", "closed_loop_concurrency", "preemptions_observed",
            "overload_reached", "dtype", "block_size", "max_num_seqs", "max_num_batched_tokens"))}
    return manifest, hardware, records, details, schema


def _legacy(reader: _Reader) -> tuple[dict, dict, list[dict], dict, dict]:
    manifest = _object(reader.json("manifest.json"), "legacy manifest")
    experiment = _object(reader.json("experiment.json"), "legacy experiment")
    if manifest.get("format") != "mlsys-data-share-v1":
        _fail("Unsupported legacy reference format")
    hardware = _object(experiment.get("hardware"), "legacy hardware")
    software = _object(experiment.get("software"), "legacy software")
    models = _object(experiment.get("models"), "legacy models")
    if not hardware.get("gpu") or not software.get("vllm") or not models:
        _fail("Legacy hardware/model/software metadata is incomplete")
    runs = reader.csv("tables/runs.csv", ("record", "model", "trace", "config", "concurrency"))
    run_map = {row["record"]: row for row in runs}
    if len(run_map) != len(runs):
        _fail("Duplicate legacy run record")
    rows = reader.csv("tables/requests.csv", ("record", "request_no", "input_tokens", "output_tokens",
                                               "cached_tokens", "ttft_ms", "tpot_ms", "e2e_ms"))
    records, seen = [], set()
    all_success = experiment.get("replay", {}).get("all_recorded_requests_successful") is True
    for row in rows:
        run_id = _identifier(row["record"], "record")
        if run_id not in run_map:
            _fail("Legacy request refers to an unknown run")
        number = _number(row["request_no"], "request_no", integer=True)
        uid = f"{run_id}:{number}"
        if uid in seen:
            _fail("Duplicate legacy request ID")
        seen.add(uid)
        run = run_map[run_id]
        record: dict[str, Any] = {"uid": uid, "request_id": uid, "run_id": run_id, "phase": "measure",
            "model": _identifier(run["model"], "model"), "trace_id": _identifier(run["trace"], "trace"),
            "config_id": _identifier(run["config"], "config"), "protocol": "closed_loop_completion_refill",
            "concurrency": _number(run["concurrency"], "concurrency", integer=True, minimum=1),
            "status": "reported_success" if all_success else "unknown", "integrity_issue": False}
        for source, target in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens_actual"),
                               ("cached_tokens", "cached_tokens")):
            record[target] = _number(row[source], source, integer=True, minimum=0 if source == "cached_tokens" else 1)
        if record["cached_tokens"] > record["input_tokens"]:
            _fail("Legacy cached token count exceeds input length")
        for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
            record[key] = _number(row[key], key)
        record.update(arrival_time_ms=None, send_time_ms=None, first_content_time_ms=None, last_content_time_ms=None)
        records.append(record)
    for name, required in (("tables/profiling.csv", ("profile", "record")), ("tables/engine.csv", ("record",)),
                           ("tables/gpu.csv", ("record",))):
        if reader.exists(name):
            reader.csv(name, required)
    claimed = _object(manifest.get("counts", {}), "manifest counts")
    mismatches = []
    for key, count in (("runs", len(runs)), ("measurement_requests", len(records))):
        if key in claimed and claimed[key] != count:
            mismatches.append({"table": key, "declared_rows": claimed[key], "actual_rows": count})
    gpu_name = str(hardware["gpu"])
    details = {"format": "mlsys-data-share-v1",
        "hardware": {"vendor": "NVIDIA" if "NVIDIA" in gpu_name else "AMD" if "AMD" in gpu_name else "unknown",
                     **_safe_fields(hardware, ("gpu", "count", "memory_mib"))},
        "software": _safe_fields(software, ("vllm", "torch", "cuda_runtime", "transformers", "tokenizers")),
        "models": [_model_summary(name, _object(value, "legacy model"), None) for name, value in models.items()],
        "counts": {"requests": len(records), "measure": len(records), "warmup": 0,
                   "runs": len(runs), "cells": None},
        "protocols": {"mode": {"closed_loop": len(runs)}, "arrival_times_available": False,
            "measurement_phase_only": True, "warmup_rows_available": False,
            "request_status_evidence": "declared_all_successful" if all_success else "not_available",
            "closed_loop_replay_scope": "arrival_times_not_provided",
            "throughput_window": "published_measurement_window_not_reconstructible_from_request_rows",
            "replay": _safe_fields(experiment.get("replay", {}), ("policy", "concurrency", "warmup_requests",
                "measurement_requests", "preemptions_per_run"))},
        "integrity": {"count_mismatches": mismatches, "request_rows_with_integrity_note": 0,
            "timestamps": {"present": False, "kind": "not_provided", "token_events_available": False}},
        "coverage": {"profiling_is_separate_from_low_interference_timing": True}}
    return manifest, experiment, records, details, {}


def import_reference_data(path: str | Path, *, allow_test_fixture: bool = False) -> dict:
    """Audit known extracted reference formats without promoting their evidence.

    Checksum and declared-count mismatches are retained as audit failures rather
    than repaired. Unsafe paths, malformed metadata and invalid numeric evidence
    are rejected. No canonical stage dataset or observed-run contract is made.
    """
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be a boolean")
    try:
        reader = _Reader(Path(path))
        release = reader.exists("00_environment/hardware.json") and reader.exists("02_measured/schema.json")
        if release:
            manifest, source_metadata, records, audit, _ = _release(reader)
            manifest_path = "MANIFEST.json"
        elif reader.exists("manifest.json") and reader.exists("experiment.json"):
            manifest, source_metadata, records, audit, _ = _legacy(reader)
            manifest_path = "manifest.json"
        else:
            _fail("Directory does not contain a recognized reference-data layout")
        expected = _manifest_hashes(manifest, release)
        checksums = _checksums(reader)
        if reader.exists("README.md"):
            reader.raw("README.md")  # Hash only; bundled prose is never an instruction.
        source = _source(manifest, source_metadata, allow_test_fixture, reader.files[manifest_path]["sha256"])
        audit["manifest"] = _check_hashes(reader, expected, checksums)
        audit["source_path"] = str(reader.root)
        audit["source_files"] = [reader.files[name] for name in sorted(reader.files)]
        audit["source_sha256"] = hashlib.sha256(json.dumps(audit["source_files"],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        audit["source_assurance"] = "declared_external_measurement_source_not_independent_hardware_authentication"
        audit["archive"] = {"input_kind": "extracted_directory", "nested_archives_opened": False}
        geometry_supported = bool(audit["models"]) and all(
            model.get("simulator_model_compatible", False) for model in audit["models"])
        reasons = ["NO_PER_STEP_BATCH_COMPOSITION_AND_TIMING", "NO_SERVER_TOKEN_ID_EVENTS",
            "CLIENT_CONTENT_TIMING_BOUNDARY_DIFFERS_FROM_STAGE_BOUNDARY",
            "REQUEST_REPLAY_POLICY_AND_WINDOW_NEED_EXPLICIT_ADAPTER"]
        if not geometry_supported:
            reasons.append("MODEL_IDENTITY_INCOMPLETE_OR_UNSUPPORTED")
        audit["compatibility"] = {
            "request_service_reference": True, "canonical_stage_profile": False, "canonical_token_trace": False,
            "canonical_online_comparison": False, "prompt_token_ids_available": False,
            "workspace_attribution_available": False, "model_adapter_required": not geometry_supported,
            "model_geometry_supported": geometry_supported,
            "amd_hardware_evidence": audit["hardware"]["vendor"] == "AMD",
            "reasons": reasons,
        }
        audit["validation_readiness"] = {"ready": False, "hardware_validated": False,
            "usable_for": ["client_request_service_reference", "declared_runtime_coverage_audit"],
            "missing_evidence": ["stage_service_samples", "token_level_server_trace", "workspace_attribution",
                                 "independent_heldout_validation", "multi_gpu_communication_and_recovery_evidence"],
            "integrity_checks_passed": (audit["manifest"]["passed"] and not audit["integrity"]["count_mismatches"]
                and audit["integrity"].get("published", {}).get("passed") is not False
                and not audit["integrity"]["timestamps"].get("unpaired_request_rows", 0))}
        return {"schema_version": 1, "status": "COMPLETED", "kind": "reference_import", "source": source,
                "audit": audit, "request_service_records": records,
                "raw_records": {"tables": reader.tables, "content_included": False,
                                "prompt_text_included": False, "output_text_included": False},
                "hardware_validated": False}
    except SimulationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, csv.Error, EOFError) as exc:
        raise SimulationError("INVALID_REFERENCE_DATA", "Malformed or unreadable reference-data package") from exc
