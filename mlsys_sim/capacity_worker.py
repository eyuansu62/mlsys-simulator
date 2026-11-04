"""One fresh owned-process Runtime capacity trial; Torch imports only on execution."""

import argparse
from copy import deepcopy
import importlib
from pathlib import Path
import time
import traceback
from typing import Any, Callable

from .capacity import (PROTOCOL, _allocator, _control_observation, _digest, _integer, _object,
                       _profile_inputs, _read, canonical_json, config_sha256)
from .config import experiment_from_dict
from .schema import SimulationError


_ADMISSION = {"KV_CAPACITY_EXCEEDED", "KV_ADMISSION_REJECTED", "ADMISSION_REJECTED", "HOST_SWAP_CAPACITY_EXCEEDED"}


def _class_name(value: Any) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


def _counters(torch: Any, devices: list[dict], errors: list[dict]) -> list[dict]:
    result = []
    for fact in devices:
        index = fact["index"]
        try:
            free, total = torch.cuda.mem_get_info(index)
            row = {"device_index": index, "allocated_bytes": torch.cuda.memory_allocated(index),
                "reserved_bytes": torch.cuda.memory_reserved(index), "free_bytes": free, "total_bytes": total,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(index)}
            result.extend(_allocator([row], devices))
        except Exception as exc:
            errors.append({"device_index": index, "exception_class": _class_name(exc), "message": str(exc)[:4096]})
    return result


def _oom_counters(torch: Any, devices: list[dict]) -> dict[int, int | None]:
    result = {}
    for fact in devices:
        index = fact["index"]
        try:
            value = torch.cuda.memory_stats(index).get("num_ooms")
            result[index] = value if type(value) is int and value >= 0 else None
        except Exception:
            result[index] = None
    return result


def _kv_bindings(adapter: Any, experiment: Any) -> list[dict]:
    method = getattr(adapter, "fixed_kv_pool_allocations", None)
    if not callable(method):
        raise SimulationError("KV_POOL_BINDING_UNAVAILABLE", "KV-pool sweeps need actual fixed_kv_pool_allocations()")
    allocations = method()
    expected = set(range(len(experiment.infra.devices)))
    if not isinstance(allocations, dict) or any(type(key) is not int for key in allocations) or set(allocations) != expected:
        raise SimulationError("KV_POOL_BINDING_UNAVAILABLE", "Fixed KV pool allocations must cover every visible rank")
    rows = []
    for index in sorted(allocations):
        buffers = allocations[index]
        if not isinstance(buffers, (list, tuple)) or not buffers:
            raise SimulationError("KV_POOL_BINDING_UNAVAILABLE", "Each fixed KV pool needs actual storage tensors")
        storages: dict[int, int] = {}
        for tensor in buffers:
            device = getattr(tensor, "device", None)
            if getattr(device, "type", None) != "cuda" or getattr(device, "index", None) != index:
                raise SimulationError("KV_POOL_BINDING_MISMATCH", "Fixed KV pool tensor device differs from its rank")
            storage = tensor.untyped_storage()
            pointer, size = _integer(storage.data_ptr(), "storage pointer", 1), _integer(storage.nbytes(), "storage bytes", 1)
            if pointer in storages and storages[pointer] != size:
                raise SimulationError("KV_POOL_BINDING_MISMATCH", "Aliased KV storage has inconsistent byte extent")
            storages[pointer] = size
        total = sum(storages.values())
        if total != experiment.strategy.memory.kv_pool_bytes:
            raise SimulationError("KV_POOL_BINDING_MISMATCH", "Actual fixed KV storage differs from the effective pool budget")
        rows.append({"device_index": index, "storage_bytes": total, "allocation_count": len(storages),
                     "method": "observed_unique_tensor_storage"})
    return rows


def _runtime_metadata(adapter: Any, config: dict, experiment: Any) -> dict:
    method = getattr(adapter, "metadata", None)
    if not callable(method):
        raise SimulationError("INVALID_CAPACITY_ADAPTER", "Runtime adapter must provide metadata()")
    value = method()
    if not isinstance(value, dict):
        raise SimulationError("INVALID_CAPACITY_ADAPTER", "Runtime metadata must be a JSON object")
    canonical_json(value)
    if (value.get("isolation") != "owned_process"
            or value.get("effective_config_sha256") != config["effective_config_sha256"]
            or canonical_json(value.get("applied_axis")) != canonical_json(config["axis"])
            or value.get("runtime_name") != experiment.calibration.runtime_name
            or value.get("runtime_version") != experiment.calibration.runtime_version
            or value.get("execution_scope") != "complete_declared_workload"):
        raise SimulationError("CAPACITY_CONFIG_UNPROVEN", "Runtime metadata does not prove the owned effective workload/configuration")
    if config["axis"]["name"] == "kv_pool_bytes" and (
            type(value.get("fixed_kv_pool_bytes")) is not int
            or value["fixed_kv_pool_bytes"] != experiment.strategy.memory.kv_pool_bytes):
        raise SimulationError("KV_POOL_BINDING_MISMATCH", "Runtime metadata does not report the actual fixed KV pool")
    return deepcopy(value)


def _error_code(exc: Exception, torch: Any) -> tuple[str, bool]:
    oom_class = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    typed = isinstance(oom_class, type) and isinstance(exc, oom_class)
    if typed:
        return "PYTORCH_ALLOCATOR_OOM", True
    if isinstance(exc, SimulationError):
        return exc.code, False
    if isinstance(exc, AssertionError):
        for frame, _ in traceback.walk_tb(exc.__traceback__):
            owner = frame.f_locals.get("self")
            if type(owner).__name__ == "WorkspaceManager" or frame.f_globals.get("__name__") == "vllm.v1.worker.workspace":
                return "WORKSPACE_MANAGER_ASSERTION", False
        return "ASSERTION_FAILURE", False
    if isinstance(exc, TimeoutError):
        return "RUNTIME_TIMEOUT", False
    if isinstance(exc, ChildProcessError):
        return "PEER_FAILURE", False
    backend_error = getattr(getattr(torch, "distributed", None), "DistBackendError", None)
    if isinstance(backend_error, type) and isinstance(exc, backend_error):
        return "DISTRIBUTED_BACKEND_ERROR", False
    return "RUNTIME_ERROR", False


def run_capacity_trial(config: dict, *, torch_module: Any = None, factory: Callable | None = None,
                       clock_ns: Callable[[], int] | None = None) -> dict:
    """Classify a single isolated trial; injected execution always stays fixture."""
    value = _object(config, {"schema_version", "kind", "protocol", "session_id", "worker_id", "point_index", "repeat",
        "axis", "experiment", "effective_config_sha256", "adapter_entry_point", "source_reference", "profile_input_hashes"},
        set(), "capacity trial config")
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != "memory_capacity_trial_config" or value["protocol"] != PROTOCOL):
        raise SimulationError("INVALID_CAPACITY_CONFIG", "Unknown capacity trial configuration")
    experiment = experiment_from_dict(value["experiment"])
    if _digest(value["effective_config_sha256"], "effective config SHA") != config_sha256(experiment):
        raise SimulationError("INVALID_CAPACITY_CONFIG", "Worker received a mismatched effective configuration")
    selected = [{key: row[key] for key in ("field", "index", "sha256", "source_kind")}
                for row in _profile_inputs(experiment)]
    if canonical_json(selected) != canonical_json(value["profile_input_hashes"]):
        raise SimulationError("CAPACITY_PROFILE_INPUT_CHANGED", "Worker Profile inputs differ from the captured bytes")
    fixture = any(item is not None for item in (torch_module, factory, clock_ns))
    source = {"kind": "test_fixture" if fixture else "measured", "reference": value["source_reference"]}
    result = _control_observation(value, source, "ERROR", "TRIAL_NOT_COMPLETED", "Fresh trial has not completed")
    result["oom_allocator_evidence"] = {"phase": None, "per_device": []}
    result["capacity_oom_evidence"] = None
    clock = time.perf_counter_ns if clock_ns is None else clock_ns
    start = clock()
    phase, adapter, torch = "startup", None, torch_module
    before_oom: dict[int, int | None] = {}
    oom = False
    try:
        torch = importlib.import_module("torch") if torch is None else torch
        if not getattr(torch.version, "hip", None) or not torch.cuda.is_available() or torch.cuda.device_count() < len(experiment.infra.devices):
            raise SimulationError("HARDWARE_UNAVAILABLE", "Capacity collection requires an available PyTorch HIP backend")
        result["stack"] = {"torch_version": str(torch.__version__), "hip_version": str(torch.version.hip)}
        for index, expected in enumerate(experiment.infra.devices):
            properties = torch.cuda.get_device_properties(index)
            fact = {"index": index, "name": str(properties.name), "memory_bytes": int(properties.total_memory),
                    "architecture": getattr(properties, "gcnArchName", None)}
            if fact["name"] != expected.name or fact["memory_bytes"] != expected.memory_bytes:
                raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Actual capacity-worker GPU differs from the bound rank")
            result["devices"].append(fact)
            torch.cuda.reset_peak_memory_stats(index)
        result["allocator"]["before"] = _counters(torch, result["devices"], result["allocator"]["errors"])
        phase = "load"
        before_oom = _oom_counters(torch, result["devices"])
        if factory is None:
            module, name = value["adapter_entry_point"].split(":")
            factory = getattr(importlib.import_module(module), name)
        adapter = factory(experiment)
        result["metadata"] = _runtime_metadata(adapter, value, experiment)
        for name in ("prepare", "run"):
            if not callable(getattr(adapter, name, None)):
                raise SimulationError("INVALID_CAPACITY_ADAPTER", f"Runtime adapter must expose {name}()")
        phase = "prepare"
        before_oom = _oom_counters(torch, result["devices"])
        adapter.prepare()
        if value["axis"]["name"] == "kv_pool_bytes":
            result["kv_pool_bindings"] = _kv_bindings(adapter, experiment)
        phase = "run"
        before_oom = _oom_counters(torch, result["devices"])
        actual = adapter.run()
        if not isinstance(actual, dict) or actual.get("status") not in {"COMPLETED", "ADMISSION_REJECTED"}:
            raise SimulationError("INVALID_CAPACITY_ADAPTER", "run() must return a complete workload or explicit admission refusal")
        canonical_json(actual)
        if actual["status"] == "COMPLETED":
            expected_ids = {request.request_id for request in experiment.workload}
            ids = actual.get("completed_request_ids")
            if not isinstance(ids, list) or len(ids) != len(expected_ids) or set(ids) != expected_ids:
                raise SimulationError("INCOMPLETE_CAPACITY_WORKLOAD", "Completed trial must name the entire declared request cohort")
        for index in range(len(experiment.infra.devices)):
            torch.cuda.synchronize(index)
        result.update(status=actual["status"], failure_phase=None if actual["status"] == "COMPLETED" else "run",
            error_code=None if actual["status"] == "COMPLETED" else "ADMISSION_REJECTED", message=actual.get("reason"),
            result=deepcopy(actual), exception_class=None, typed_oom=False)
    except Exception as exc:
        code, oom = _error_code(exc, torch)
        if phase == "startup" and isinstance(exc, (ImportError, OSError)):
            code = "HARDWARE_UNAVAILABLE"
        after_oom = _oom_counters(torch, result["devices"]) if torch is not None else {}
        evidence = [{"device_index": index, "before": before_oom.get(index), "after": after_oom.get(index)}
                    for index in sorted(after_oom)]
        if oom and adapter is not None and value["axis"]["name"] == "kv_pool_bytes" and phase in {"prepare", "run"}:
            evidence_method = getattr(adapter, "capacity_oom_evidence", None)
            if callable(evidence_method):
                try:
                    result["capacity_oom_evidence"] = deepcopy(evidence_method())
                    canonical_json(result["capacity_oom_evidence"])
                except Exception as proof_exc:
                    result["capacity_oom_evidence"] = {"error": str(proof_exc)[:4096]}
        result.update(status="OOM" if oom else "ADMISSION_REJECTED" if code in _ADMISSION else
            "UNAVAILABLE" if code == "HARDWARE_UNAVAILABLE" else "TIMEOUT" if code == "RUNTIME_TIMEOUT" else "ERROR",
            failure_phase=phase, exception_class=_class_name(exc), typed_oom=oom,
            error_code=code, message=str(exc)[:4096], oom_allocator_evidence={"phase": phase, "per_device": evidence})
    finally:
        if torch is not None and result["devices"]:
            result["allocator"]["after"] = _counters(torch, result["devices"], result["allocator"]["errors"])
        if adapter is not None and not oom:
            try:
                phase = "cleanup"
                before_oom = _oom_counters(torch, result["devices"])
                close = getattr(adapter, "close", None)
                if callable(close):
                    close()
                result["cleanup_status"] = "COMPLETED"
            except Exception as exc:
                code, cleanup_oom = _error_code(exc, torch)
                after_oom = _oom_counters(torch, result["devices"])
                result.update(status="OOM" if cleanup_oom else "ERROR", failure_phase="cleanup", error_code=code,
                    exception_class=_class_name(exc), typed_oom=cleanup_oom, message=str(exc)[:4096],
                    cleanup_status="ERROR", oom_allocator_evidence={"phase": "cleanup", "per_device": [
                        {"device_index": index, "before": before_oom.get(index), "after": after_oom.get(index)}
                        for index in sorted(after_oom)]})
        else:
            result["cleanup_status"] = "SKIPPED_AFTER_OOM" if oom else "NOT_CREATED"
        elapsed = clock() - start
        result["elapsed_ms"] = elapsed / 1e6
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config, _ = _read(args.config, 4 * 1024 * 1024)
    try:
        result = run_capacity_trial(config)
    except Exception as exc:
        result = _control_observation(config, {"kind": "measured", "reference": config.get("source_reference", "capacity worker")},
            "ERROR", "WORKER_STARTUP_ERROR", str(exc)[:4096])
        result["oom_allocator_evidence"] = {"phase": "startup", "per_device": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "rank-0.json"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(result))
    # A typed OOM/admission refusal is a completed observation, not a launcher failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
