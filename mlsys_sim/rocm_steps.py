"""Whole-stage timing around a user supplied inference Runtime on PyTorch HIP.

The Runtime owns model execution and existing KV. This adapter owns the timing
boundary, allocator evidence and provenance. It never substitutes a synthetic
kernel for a model, or derives Workspace from total process allocator usage.
PyTorch is imported only when live collection is requested.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import time
from typing import Any, NoReturn

from .config import validate_experiment
from .hardware import (CollectionError, bind_inventory, integer, object_fields,
                       text)
from .profiles import profile_identity, shape_key, validate_dataset
from .schema import Experiment, SimulationError


_BOUNDARY = "step_includes_output_and_sampling_v1"
_LIMITATIONS = [
    "Runtime callbacks must execute the declared complete stage, including local collectives and output/sampling.",
    "Host wall time includes Runtime/framework service and synchronization, not only GPU kernel time.",
    "Allocator counters include persistent weights, KV and outputs; they are not Workspace attribution.",
    "No hardware prediction accuracy is inferred from collection alone.",
]


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _invalid(message: str) -> NoReturn:
    raise SimulationError("INVALID_COLLECTION_PLAN", message)


def _plan(value: dict, experiment: Experiment) -> dict:
    plan = object_fields(value, {"schema_version", "dataset_id", "source_reference", "samples"},
                         {"warmup_iterations", "iterations"}, "step collection plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        _invalid("step collection plan schema_version must be 1")
    dataset_id = text(plan["dataset_id"], "dataset_id")
    reference = text(plan["source_reference"], "source_reference")
    warmup = integer(plan.get("warmup_iterations", 3), "warmup_iterations")
    iterations = integer(plan.get("iterations", 10), "iterations", minimum=1)
    if not isinstance(plan["samples"], list) or not plan["samples"]:
        _invalid("samples must be a nonempty array")
    if len(plan["samples"]) > 10000 or iterations * len(plan["samples"]) > 1000000 or warmup > 1000000:
        _invalid("collection exceeds the bounded sample/iteration limits")
    samples = []
    identifiers = set()
    for value in plan["samples"]:
        row = object_fields(value, {"sample_id", "group_id", "phase", "stage_index",
                                   "query_tokens", "context_lengths"}, {"request_phases"}, "step sample")
        sample_id = text(row["sample_id"], "sample_id")
        if sample_id in identifiers:
            _invalid("sample_id must be unique within a collection")
        identifiers.add(sample_id)
        group_id = text(row["group_id"], "group_id")
        try:
            key = shape_key(row["phase"], row["query_tokens"], row["context_lengths"],
                            row["stage_index"], request_phases=row.get("request_phases"))
        except SimulationError as exc:
            raise SimulationError("INVALID_COLLECTION_PLAN", f"Invalid sample {sample_id}: {exc}") from exc
        if row["phase"] != "mixed" and "request_phases" in row:
            _invalid("request_phases must be omitted for homogeneous samples")
        if row["stage_index"] >= experiment.strategy.pipeline_parallel_size:
            _invalid("sample stage_index exceeds the experiment stage count")
        current = {"sample_id": sample_id, "group_id": group_id, "phase": row["phase"],
                   "stage_index": row["stage_index"], "query_tokens": list(row["query_tokens"]),
                   "context_lengths": list(row["context_lengths"]), "shape_key": key}
        if row["phase"] == "mixed":
            current["request_phases"] = list(row["request_phases"])
        samples.append(current)
    return {"dataset_id": dataset_id, "source_reference": reference, "samples": samples,
            "warmup_iterations": warmup, "iterations": iterations}


def _metadata(adapter: Any) -> dict:
    method = getattr(adapter, "metadata", None)
    if not callable(method):
        _invalid("Runtime adapter must expose metadata()")
    value = deepcopy(object_fields(method(), {"runtime_name", "runtime_version", "timing_boundary",
                                             "includes_local_collectives", "workspace_attribution"},
                                   {"rccl_runtime_version", "rccl_library_sha256", "rccl_binding_reference"},
                                   "Runtime metadata"))
    text(value["runtime_name"], "runtime_name")
    text(value["runtime_version"], "runtime_version")
    if value["timing_boundary"] != _BOUNDARY or value["includes_local_collectives"] is not True:
        _invalid("Runtime must declare the supported complete-stage boundary and included local collectives")
    if value["workspace_attribution"] not in ("none", "runtime"):
        _invalid("workspace_attribution must be none or runtime")
    if value.get("rccl_runtime_version") is not None:
        text(value["rccl_runtime_version"], "rccl_runtime_version")
    if value.get("rccl_library_sha256") is not None:
        digest = text(value["rccl_library_sha256"], "rccl_library_sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            _invalid("rccl_library_sha256 must be a SHA256 hexadecimal digest")
        value["rccl_library_sha256"] = digest.lower()
    if "rccl_binding_reference" in value:
        text(value["rccl_binding_reference"], "rccl_binding_reference")
    for name in ("prepare", "run_step"):
        if not callable(getattr(adapter, name, None)):
            _invalid(f"Runtime adapter must expose {name}(sample)")
    return value


def _check_rccl_binding(metadata: dict, inventory: dict, *, require_rccl: bool) -> None:
    """A queried library is not proof of the Runtime's loaded collective library.

    Live TP adapters must obtain the version and file hash from their actual
    loaded RCCL library. torch.cuda.nccl.version() build macros are insufficient.
    This is a Runtime-supplied binding declaration, retained with the capture.
    """
    if not require_rccl or inventory.get("source", {}).get("kind") != "measured":
        return
    version = metadata.get("rccl_runtime_version")
    digest = metadata.get("rccl_library_sha256")
    library = inventory.get("rccl_library")
    observed_version = inventory.get("versions", {}).get("rccl_version")
    observed_digest = library.get("sha256") if isinstance(library, dict) else None
    missing = (not isinstance(version, str) or version.strip().lower() in
               {"", "unknown", "unspecified", "not-applicable", "n/a"}
               or not isinstance(digest, str) or not isinstance(observed_version, str)
               or observed_version.strip().lower() in {"", "unknown", "unspecified", "not-applicable", "n/a"}
               or not isinstance(observed_digest, str) or len(observed_digest) != 64
               or any(character not in "0123456789abcdefABCDEF" for character in observed_digest))
    if missing:
        raise SimulationError("HARDWARE_METADATA_INCOMPLETE",
                              "Live TP collection requires the Runtime's loaded RCCL version and library SHA256")
    if version != observed_version or digest != str(observed_digest).lower():
        raise SimulationError("HARDWARE_IDENTITY_MISMATCH",
                              "Runtime loaded RCCL version/library differ from the queried inventory library")


def _counter(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise SimulationError("INVALID_ALLOCATOR_SAMPLE", f"{name} must be a nonnegative 64-bit integer")
    return value


def _allocator(cuda: Any, indices: list[int], *, baseline: bool) -> list[dict]:
    records = []
    for index in indices:
        row = {"device_index": index,
               "allocated_bytes": _counter(cuda.memory_allocated(index), "memory_allocated"),
               "reserved_bytes": _counter(cuda.memory_reserved(index), "memory_reserved")}
        if not baseline:
            row["peak_allocated_bytes"] = _counter(cuda.max_memory_allocated(index), "max_memory_allocated")
            row["peak_reserved_bytes"] = _counter(cuda.max_memory_reserved(index), "max_memory_reserved")
        records.append(row)
    return records


def _workspace(adapter: Any, sample: dict, indices: list[int], metadata: dict) -> tuple[int | None, str]:
    method = getattr(adapter, "workspace_peak_bytes", None)
    if metadata["workspace_attribution"] != "runtime" or not callable(method):
        return None, "not_attributed"
    value = method(deepcopy(sample))
    if value is None:
        return None, "not_attributed"
    if isinstance(value, Mapping):
        if any(type(index) is not int for index in value) or set(value) != set(indices):
            raise SimulationError("INVALID_ALLOCATOR_SAMPLE", "Workspace mapping must cover exactly the participating devices")
        return max(_counter(value[index], "workspace_peak_bytes") for index in indices), "runtime_reported_rank_max"
    return _counter(value, "workspace_peak_bytes"), "runtime_reported_group_bound"


def _clock(clock_ns: Callable[[], int]) -> int:
    value = clock_ns()
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise SimulationError("INVALID_TIMING_SAMPLE", "Monotonic clock must return nonnegative 64-bit integer ns")
    return value


def _check_backend(torch: Any, inventory: dict, devices: int) -> None:
    cuda = getattr(torch, "cuda", None)
    hip = getattr(getattr(torch, "version", None), "hip", None)
    if not hip or cuda is None or not cuda.is_available():
        raise SimulationError("HARDWARE_UNAVAILABLE", "Collection requires an available PyTorch HIP device")
    if inventory.get("status") != "READY" or not isinstance(inventory.get("devices"), list):
        raise SimulationError("HARDWARE_UNAVAILABLE", "A ready ROCm inventory is required")
    if cuda.device_count() < devices or len(inventory["devices"]) < devices:
        raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "The Runtime does not expose every declared stage/TP device")
    for index in range(devices):
        properties = cuda.get_device_properties(index)
        captured = inventory["devices"][index]
        if (captured.get("index") != index or captured.get("name") != properties.name
                or captured.get("memory_bytes") != properties.total_memory):
            raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Current device properties differ from the captured inventory")
    captured_torch = inventory.get("torch", {})
    if captured_torch.get("hip_version") != str(hip) or captured_torch.get("version") != str(torch.__version__):
        raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Current PyTorch/HIP build differs from the captured inventory")


def collect_steps(experiment: Experiment, plan: dict, adapter: Any, inventory: dict, *,
                  torch_module: Any = None, clock_ns: Callable[[], int] | None = None,
                  allow_test_fixture: bool = False) -> dict:
    """Measure complete stage calls, returning canonical records and raw evidence.

    An adapter provides ``metadata()``, ``prepare(sample)`` and
    ``run_step(sample)``. An optional ``workspace_peak_bytes(sample)`` may return
    a conservative per-rank bound or a mapping over participating device indices
    only when metadata explicitly declares Runtime attribution. ``close()`` is
    called if present. The adapter is responsible for group-level completion
    inside its callable; participant device synchronization brackets the call.

    Supplied ``torch_module`` or ``clock_ns`` dependencies are test injection:
    their records are always fixtures, regardless of the input inventory label.
    Live TP metadata must also declare ``rccl_runtime_version`` and
    ``rccl_library_sha256`` for the actually loaded library, matching inventory.
    """
    if type(allow_test_fixture) is not bool:
        _invalid("allow_test_fixture must be boolean")
    bundle: dict[str, Any] = {"status": "FAILED", "kind": "step_collection",
                              "hardware_validated": False, "raw_records": [],
                              "limitations": list(_LIMITATIONS)}
    error: Exception | None = None
    completed: dict | None = None
    try:
        validate_experiment(experiment)
        if experiment.strategy.replicas != 1:
            _invalid("Stage collection requires one replica; slice the experiment before collecting")
        parsed = _plan(plan, experiment)
        captured = deepcopy(inventory)
        injected = torch_module is not None or clock_ns is not None
        if injected:
            captured["source"] = {"kind": "test_fixture",
                                  "reference": "Injected measurement dependencies; no live hardware evidence"}
        bundle["inventory"] = captured
        if torch_module is None:
            try:
                torch_module = importlib.import_module("torch")
            except ImportError as exc:
                raise SimulationError("HARDWARE_UNAVAILABLE", "PyTorch with ROCm is not installed") from exc
        torch = torch_module
        devices = experiment.strategy.tensor_parallel_size * experiment.strategy.pipeline_parallel_size
        _check_backend(torch, captured, devices)
        metadata = _metadata(adapter)
        bundle["runtime_metadata"] = metadata
        _check_rccl_binding(metadata, captured, require_rccl=experiment.strategy.tensor_parallel_size > 1)
        effective = bind_inventory(experiment, captured, runtime_name=metadata["runtime_name"],
                                   runtime_version=metadata["runtime_version"],
                                   require_rccl=experiment.strategy.tensor_parallel_size > 1,
                                   allow_test_fixture=allow_test_fixture)
        source = {"kind": captured["source"]["kind"], "reference": parsed["source_reference"]}
        bundle.update(source=source, experiment=asdict(effective), runtime_metadata=metadata,
                      diagnostics={"injected_dependencies": injected,
                                   "timing_boundary": "synchronized_participant_host_wall_time",
                                   "workspace_policy": "runtime_attribution_or_null"},
                      provenance={"inventory_sha256": _sha256(captured), "plan_sha256": _sha256(plan),
                                  "experiment_sha256": _sha256(asdict(effective)),
                                  "runtime_metadata_sha256": _sha256(metadata)})
        clock = time.perf_counter_ns if clock_ns is None else clock_ns
        records = []
        previous_end: int | None = None
        cuda = torch.cuda
        tp = effective.strategy.tensor_parallel_size
        for shape in parsed["samples"]:
            indices = list(range(shape["stage_index"] * tp, (shape["stage_index"] + 1) * tp))
            bundle["diagnostics"].update(current_sample_id=shape["sample_id"], current_operation="warmup")
            for _ in range(parsed["warmup_iterations"]):
                adapter.prepare(deepcopy(shape))
                for index in indices:
                    cuda.synchronize(index)
                adapter.run_step(deepcopy(shape))
                for index in indices:
                    cuda.synchronize(index)
            for repetition in range(parsed["iterations"]):
                bundle["diagnostics"]["current_operation"] = "measurement"
                record_id = f"{shape['sample_id']}/repeat-{repetition + 1}"
                raw = {"record_id": record_id, "group_id": shape["group_id"], "sample": deepcopy(shape),
                       "iteration": repetition + 1, "device_indices": list(indices), "status": "PREPARING"}
                bundle["raw_records"].append(raw)
                adapter.prepare(deepcopy(shape))
                for index in indices:
                    cuda.synchronize(index)
                raw["allocator_baseline"] = _allocator(cuda, indices, baseline=True)
                for index in indices:
                    cuda.reset_peak_memory_stats(index)
                raw["status"] = "RUNNING"
                start = _clock(clock)
                raw["start_monotonic_ns"] = start
                if previous_end is not None and start < previous_end:
                    raise SimulationError("INVALID_TIMING_SAMPLE", "The monotonic clock moved backwards between stages")
                adapter.run_step(deepcopy(shape))
                for index in indices:
                    cuda.synchronize(index)
                end = _clock(clock)
                raw["end_monotonic_ns"] = end
                duration = (end - start) / 1e6
                if not math.isfinite(duration) or duration <= 0:
                    raise SimulationError("INVALID_TIMING_SAMPLE", "A measured stage must advance the monotonic clock")
                raw["duration_ms"] = duration
                previous_end = end
                raw["allocator_after"] = _allocator(cuda, indices, baseline=False)
                workspace, workspace_source = _workspace(adapter, shape, indices, metadata)
                raw.update(status="COMPLETED", workspace_peak_bytes=workspace,
                           workspace_source=workspace_source)
                record = {"record_id": record_id, "group_id": shape["group_id"], "phase": shape["phase"],
                          "stage_index": shape["stage_index"], "query_tokens": list(shape["query_tokens"]),
                          "context_lengths": list(shape["context_lengths"]), "duration_ms": duration,
                          "workspace_peak_bytes": workspace}
                if shape["phase"] == "mixed":
                    record["request_phases"] = list(shape["request_phases"])
                records.append(record)
        dataset = validate_dataset({"schema_version": 1, "dataset_id": parsed["dataset_id"],
            "identity": profile_identity(effective), "source": source, "records": records,
            "notes": {"collector": "pytorch_rocm_callable_stage", "runtime_metadata": metadata,
                      "collection_provenance": bundle["provenance"],
                      "warmup_iterations_per_shape": parsed["warmup_iterations"],
                      "iterations_per_shape": parsed["iterations"],
                      "timing_boundary": "synchronized_participant_host_wall_time",
                      "workspace_policy": "runtime_attribution_or_null"}},
            allow_test_fixture=allow_test_fixture)
        completed = {**bundle, "status": "COMPLETED", "dataset": dataset}
    except Exception as exc:
        error = exc
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                bundle["close_error"] = {"type": type(exc).__name__, "message": str(exc)}
                error = error or exc
    if error is not None:
        if bundle["raw_records"] and bundle["raw_records"][-1]["status"] != "COMPLETED":
            bundle["raw_records"][-1].update(status="FAILED", error_type=type(error).__name__, error=str(error))
        code = error.code if isinstance(error, SimulationError) else "COLLECTION_FAILED"
        raise CollectionError(code, str(error), artifacts=bundle) from error
    if completed is None:
        raise CollectionError("COLLECTION_FAILED", "Stage collection did not produce records", artifacts=bundle)
    return completed
