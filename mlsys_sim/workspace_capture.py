"""Runtime-owned dynamic Workspace attribution, separate from allocator peaks.

The recorder accounts physical scratch storage lifetimes. It does not classify
arbitrary tensors, infer ownership from stack names, or subtract a baseline from
process allocator peaks. Optional vLLM manager observation is explicitly partial.
"""

from copy import deepcopy
import functools
import importlib
import inspect
import json
import threading
import time
from typing import Any, Callable, NoReturn

from .config import validate_experiment
from .hardware import CollectionError, bind_inventory, object_fields, text
from .profiles import profile_identity
from .rocm_steps import (_allocator, _check_backend, _check_rccl_binding, _clock,
                         _plan, _sha256)
from .schema import Experiment, SimulationError
from .serialization import experiment_dict


_EXCLUDED = ["weights", "kv_pool", "graphs", "runtime", "pp_buffers", "communication_buffers"]
_ATTRIBUTION = {"method": "runtime_workspace_ledger_v1", "complete": True,
                "scope": "exclusive_dynamic_workspace", "excluded_categories": _EXCLUDED,
                "includes_non_torch": True}
_MISSING = object()
_MAX_RECORDS = 10000
_MAX_TOTAL_EVENTS = 1000000
_MAX_RAW_BYTES = 32 * 1024 * 1024


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise SimulationError("INVALID_WORKSPACE_CAPTURE", f"{name} must be an integer >= {minimum}")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or "\x00" in value:
        raise SimulationError("INVALID_WORKSPACE_CAPTURE", f"{name} must be a bounded nonempty string")
    return value


class WorkspaceRecorder:
    """One explicit stage/epoch with rank-complete physical scratch lifetimes.

    ``release_completed`` means the storage can actually be reused, including
    asynchronous users. ``request_free`` alone never reduces live bytes. Every
    rank must call ``complete_rank`` even when its true Workspace peak is zero.
    No Tensor/storage reference is retained by ``record_tensor``.
    """

    def __init__(self, tensor_parallel_size: int, *, stage_index: int = 0,
                 epoch_id: str = "workspace-epoch", max_events: int = 100000,
                 device_indices: list[int] | tuple[int, ...] | None = None,
                 max_event_bytes: int = 4 * 1024 * 1024):
        self.tensor_parallel_size = _integer(tensor_parallel_size, "tensor_parallel_size", 1)
        if self.tensor_parallel_size > 1024:
            raise SimulationError("INVALID_WORKSPACE_CAPTURE", "Workspace ranks exceed the bounded limit")
        self.stage_index = _integer(stage_index, "stage_index")
        self.epoch_id = _identifier(epoch_id, "epoch_id")
        self._max = _integer(max_events, "max_events", 1)
        if self._max > 1000000:
            raise SimulationError("INVALID_WORKSPACE_CAPTURE", "max_events exceeds the bounded limit")
        self._max_event_bytes = _integer(max_event_bytes, "max_event_bytes", 1)
        if self._max_event_bytes > 4 * 1024 * 1024:
            raise SimulationError("INVALID_WORKSPACE_CAPTURE", "Event byte budget exceeds 4MiB")
        self._event_bytes = 0
        devices = list(range(self.tensor_parallel_size)) if device_indices is None else list(device_indices)
        devices = [_integer(device, "device_index") for device in devices]
        if len(devices) != self.tensor_parallel_size or len(set(devices)) != len(devices):
            raise SimulationError("INVALID_WORKSPACE_CAPTURE", "One unique device is required per TP rank")
        self._devices = [_integer(device, "device_index") for device in devices]
        self._allocations: dict[tuple[int, str], tuple[str, int]] = {}
        self._storage: dict[tuple[int, str], set[str]] = {}
        self._storage_sizes: dict[tuple[int, str], int] = {}
        self._used_ids: set[tuple[int, str]] = set()
        self._requested: set[tuple[int, str]] = set()
        self._tensor_active: dict[tuple[int, str, int], str] = {}
        self._tensor_keys: dict[tuple[int, str], tuple[int, str, int]] = {}
        self._tensor_serial = 0
        self._live = [0] * self.tensor_parallel_size
        self._peaks = [0] * self.tensor_parallel_size
        self._completed: set[int] = set()
        self._events: list[dict] = []
        self._sealed = False
        self._error: dict | None = None
        self._lock = threading.RLock()

    def _reject(self, message: str, code: str = "INVALID_WORKSPACE_CAPTURE") -> NoReturn:
        if self._error is None:
            self._error = {"error_code": code, "message": message}
        self._sealed = True
        raise SimulationError(self._error["error_code"], self._error["message"])

    def _rank(self, value: Any) -> int:
        if self._sealed:
            raise SimulationError("WORKSPACE_CAPTURE_CLOSED", "Workspace epoch is already sealed")
        if type(value) is not int or not 0 <= value < self.tensor_parallel_size:
            self._reject("Unknown or non-integer Workspace TP rank")
        if value in self._completed:
            self._reject("Workspace rank has already completed its epoch")
        return value

    def _event(self, event: dict) -> None:
        if len(self._events) >= self._max:
            self._reject("Workspace event limit reached; evidence is incomplete", "WORKSPACE_CAPTURE_LIMIT")
        size = len(json.dumps(event, separators=(",", ":"), allow_nan=False).encode()) + 1
        if self._event_bytes + size > self._max_event_bytes:
            self._reject("Workspace event byte limit reached; evidence is incomplete", "WORKSPACE_CAPTURE_LIMIT")
        self._events.append(event)
        self._event_bytes += size

    def _validated(self, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except SimulationError as exc:
            self._reject(str(exc), exc.code)

    def allocate(self, tp_rank: int, allocation_id: str, storage_id: str, size_bytes: int) -> None:
        self._allocate(tp_rank, allocation_id, storage_id, size_bytes)

    def _allocate(self, tp_rank: int, allocation_id: str, storage_id: str, size_bytes: int,
                  *, _tensor_owned: bool = False) -> None:
        with self._lock:
            rank = self._rank(tp_rank)
            allocation = self._validated(lambda: _identifier(allocation_id, "allocation_id"))
            storage = self._validated(lambda: _identifier(storage_id, "storage_id"))
            if storage.startswith("@tensor-storage:") and not _tensor_owned:
                self._reject("Manual storage IDs cannot use the reserved Tensor storage namespace")
            size = self._validated(lambda: _integer(size_bytes, "size_bytes"))
            key, storage_key = (rank, allocation), (rank, storage)
            if key in self._used_ids:
                self._reject("An allocation_id cannot be reused within one epoch")
            previous = self._storage_sizes.get(storage_key)
            if previous is not None and previous != size:
                self._reject("One storage_id cannot change its reserved byte size")
            next_live = self._live[rank] + (size if storage_key not in self._storage else 0)
            if next_live > 2**63 - 1:
                self._reject("Workspace live storage exceeds the integer bound")
            self._event({"action": "allocate", "tp_rank": rank, "allocation_id": allocation,
                         "storage_id": storage, "size_bytes": size})
            self._used_ids.add(key)
            self._storage_sizes[storage_key] = size
            self._storage.setdefault(storage_key, set()).add(allocation)
            self._allocations[key] = storage, size
            self._live[rank] = next_live
            self._peaks[rank] = max(self._peaks[rank], next_live)

    def request_free(self, tp_rank: int, allocation_id: str) -> None:
        with self._lock:
            rank = self._rank(tp_rank)
            allocation = self._validated(lambda: _identifier(allocation_id, "allocation_id"))
            key = rank, allocation
            if key not in self._allocations or key in self._requested:
                self._reject("Free request must refer to one live, not-yet-requested allocation")
            self._event({"action": "free_requested", "tp_rank": rank, "allocation_id": allocation})
            self._requested.add(key)

    def release_completed(self, tp_rank: int, allocation_id: str) -> None:
        with self._lock:
            rank = self._rank(tp_rank)
            allocation = self._validated(lambda: _identifier(allocation_id, "allocation_id"))
            key = rank, allocation
            if key not in self._allocations:
                self._reject("Completed release must refer to a live allocation exactly once")
            storage, size = self._allocations[key]
            self._event({"action": "release_completed", "tp_rank": rank, "allocation_id": allocation})
            del self._allocations[key]
            self._requested.discard(key)
            storage_key = rank, storage
            aliases = self._storage[storage_key]
            aliases.remove(allocation)
            if not aliases:
                del self._storage[storage_key]
                self._live[rank] -= size
                tensor_key = self._tensor_keys.pop(storage_key, None)
                if tensor_key is not None:
                    self._tensor_active.pop(tensor_key, None)

    def record_tensor(self, tp_rank: int, allocation_id: str, tensor: Any) -> None:
        """Account actual backing storage once across views; never keep Tensor refs."""
        with self._lock:
            rank = self._rank(tp_rank)
            try:
                device = tensor.device
                if getattr(device, "type", None) not in ("cuda", "hip") or getattr(device, "index", None) != self._devices[rank]:
                    self._reject("Tensor storage is not on its participating HIP/CUDA device")
                storage = tensor.untyped_storage()
                size = _integer(storage.nbytes(), "storage.nbytes")
                pointer = _integer(storage.data_ptr(), "storage.data_ptr")
                tensor_key = rank, str(device), pointer
                storage_id = self._tensor_active.get(tensor_key)
                if storage_id is None:
                    self._tensor_serial += 1
                    storage_id = f"@tensor-storage:{self._tensor_serial}"
                self._allocate(rank, allocation_id, storage_id, size, _tensor_owned=True)
                self._tensor_active[tensor_key] = storage_id
                self._tensor_keys[rank, storage_id] = tensor_key
            except Exception as exc:
                self._reject(str(exc), exc.code if isinstance(exc, SimulationError) else "INVALID_WORKSPACE_CAPTURE")

    def complete_rank(self, tp_rank: int) -> None:
        with self._lock:
            rank = self._rank(tp_rank)
            if any(key[0] == rank for key in self._allocations):
                self._reject("Workspace rank ended with live or pending-free scratch storage", "WORKSPACE_EPOCH_LEAK")
            self._event({"action": "rank_complete", "tp_rank": rank})
            self._completed.add(rank)

    mark_rank_complete = complete_rank

    def seal(self) -> list[dict]:
        with self._lock:
            if self._error is not None:
                self._reject(self._error["message"], self._error["error_code"])
            if not self._sealed:
                if self._completed != set(range(self.tensor_parallel_size)) or self._allocations:
                    self._reject("Workspace epoch lacks complete, leak-free evidence for every TP rank", "WORKSPACE_EPOCH_INCOMPLETE")
                self._sealed = True
            return [{"tp_rank": rank, "peak_bytes": value} for rank, value in enumerate(self._peaks)]

    def abort(self, reason: str = "Workspace collection did not complete") -> None:
        with self._lock:
            if not self._sealed:
                self._error = {"error_code": "WORKSPACE_EPOCH_INCOMPLETE", "message": str(reason)[:4096]}
                self._sealed = True

    def export_capture(self) -> dict:
        with self._lock:
            return {"schema_version": 1, "epoch_id": self.epoch_id, "stage_index": self.stage_index,
                "tensor_parallel_size": self.tensor_parallel_size,
                "status": "INVALID" if self._error else "COMPLETED" if self._sealed else "OPEN",
                "sealed": self._sealed, "events": deepcopy(self._events), "error": deepcopy(self._error),
                "completed_tp_ranks": sorted(self._completed),
                "peak_bytes_per_rank": [{"tp_rank": rank, "peak_bytes": value}
                                         for rank, value in enumerate(self._peaks)],
                "live_bytes_per_rank": list(self._live), "live_allocation_count": len(self._allocations)}


def _metadata(adapter: Any) -> tuple[dict, bool]:
    method = getattr(adapter, "metadata", None)
    if not callable(method):
        raise SimulationError("INVALID_COLLECTION_PLAN", "Workspace adapter must expose metadata()")
    metadata = deepcopy(object_fields(method(), {"runtime_name", "runtime_version", "timing_boundary",
        "includes_local_collectives", "workspace_attribution"}, {"complete_workspace_attribution",
        "includes_non_torch", "excluded_categories", "rccl_runtime_version", "rccl_library_sha256",
        "rccl_binding_reference", "fixed_memory_bytes_per_rank"}, "Workspace Runtime metadata"))
    text(metadata["runtime_name"], "runtime_name")
    text(metadata["runtime_version"], "runtime_version")
    if (metadata["timing_boundary"] != "step_includes_output_and_sampling_v1"
            or metadata["includes_local_collectives"] is not True):
        raise SimulationError("INVALID_COLLECTION_PLAN", "Workspace adapter must execute the supported full stage")
    if metadata["workspace_attribution"] not in ("runtime_ledger", "managed_scratch_capacity", "none"):
        raise SimulationError("INVALID_COLLECTION_PLAN", "Unknown Workspace attribution scope")
    for flag in ("complete_workspace_attribution", "includes_non_torch"):
        if flag in metadata and type(metadata[flag]) is not bool:
            raise SimulationError("INVALID_COLLECTION_PLAN", f"{flag} must be boolean")
    excluded = metadata.get("excluded_categories")
    if excluded is not None and (not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded)
                                 or len(set(excluded)) != len(excluded)):
        raise SimulationError("INVALID_COLLECTION_PLAN", "excluded_categories must be unique strings")
    complete = (metadata["workspace_attribution"] == "runtime_ledger"
        and metadata.get("complete_workspace_attribution") is True
        and metadata.get("includes_non_torch") is True and set(excluded or ()) == set(_EXCLUDED))
    for name in ("prepare", "run_step"):
        if not callable(getattr(adapter, name, None)):
            raise SimulationError("INVALID_COLLECTION_PLAN", f"Workspace adapter must expose {name}(sample)")
    if complete and any(not callable(getattr(adapter, name, None)) for name in
                        ("begin_workspace_capture", "end_workspace_capture")):
        raise SimulationError("INVALID_COLLECTION_PLAN", "Complete attribution requires begin/end Workspace instrumentation")
    return metadata, complete


def collect_workspace(experiment: Experiment, plan: dict, adapter: Any, inventory: dict, *,
                      torch_module: Any = None, clock_ns: Callable[[], int] | None = None,
                      allow_test_fixture: bool = False) -> dict:
    """Collect Runtime-owned storage lifetimes; raw allocator peaks stay separate."""
    bundle: dict[str, Any] = {"status": "FAILED", "kind": "workspace_collection", "dataset": None,
        "hardware_validated": False, "raw_records": [], "attribution_complete": False,
        "limitations": ["Workspace ownership is a Runtime instrumentation contract, not inferred allocator attribution.",
            "Managed scratch capacity and allocator peaks alone cannot establish complete dynamic Workspace.",
            "No live GPU capacity or prediction accuracy follows from collection alone."]}
    error: Exception | None = None
    completed: dict | None = None
    recorder: WorkspaceRecorder | None = None
    torch = torch_module
    try:
        if type(allow_test_fixture) is not bool:
            raise SimulationError("INVALID_COLLECTION_PLAN", "allow_test_fixture must be boolean")
        validate_experiment(experiment)
        if experiment.strategy.replicas != 1:
            raise SimulationError("INVALID_COLLECTION_PLAN", "Workspace collection requires one local replica")
        if not isinstance(plan, dict):
            raise SimulationError("INVALID_COLLECTION_PLAN", "Workspace plan must be an object")
        supplied = dict(plan)
        max_events = _integer(supplied.pop("max_events_per_sample", 100000), "max_events_per_sample", 1)
        if max_events > 100000:
            raise SimulationError("INVALID_COLLECTION_PLAN", "max_events_per_sample exceeds 100000")
        if len(json.dumps(supplied, allow_nan=False).encode()) > 4 * 1024 * 1024:
            raise SimulationError("WORKSPACE_CAPTURE_LIMIT", "Workspace plan exceeds 4MiB")
        parsed = _plan(supplied, experiment)
        if len(parsed["samples"]) * parsed["iterations"] > _MAX_RECORDS:
            raise SimulationError("WORKSPACE_CAPTURE_LIMIT", "Workspace collection exceeds 10000 records")
        if (parsed["warmup_iterations"] > 1000
                or len(parsed["samples"]) * (parsed["iterations"] + parsed["warmup_iterations"]) > 1000000):
            raise SimulationError("WORKSPACE_CAPTURE_LIMIT", "Workspace operation count exceeds the bounded limit")
        if len(json.dumps(inventory, allow_nan=False).encode()) > 16 * 1024 * 1024:
            raise SimulationError("WORKSPACE_CAPTURE_LIMIT", "Workspace inventory exceeds 16MiB")
        captured = deepcopy(inventory)
        injected = torch_module is not None or clock_ns is not None
        if injected:
            captured["source"] = {"kind": "test_fixture", "reference": "Injected Workspace backend; no live hardware evidence"}
        bundle["inventory"] = captured
        if torch is None:
            try:
                torch = importlib.import_module("torch")
            except ImportError as exc:
                raise SimulationError("HARDWARE_UNAVAILABLE", "PyTorch HIP is required for live Workspace sampling") from exc
        count = experiment.strategy.tensor_parallel_size * experiment.strategy.pipeline_parallel_size
        _check_backend(torch, captured, count)
        metadata, complete = _metadata(adapter)
        if len(json.dumps(metadata, allow_nan=False).encode()) > 1024 * 1024:
            raise SimulationError("WORKSPACE_CAPTURE_LIMIT", "Workspace Runtime metadata exceeds 1MiB")
        bundle["runtime_metadata"] = metadata
        require_rccl = experiment.strategy.tensor_parallel_size > 1 or experiment.strategy.pipeline_parallel_size > 1
        _check_rccl_binding(metadata, captured, require_rccl=require_rccl)
        effective = bind_inventory(experiment, captured, runtime_name=metadata["runtime_name"],
            runtime_version=metadata["runtime_version"], require_rccl=require_rccl,
            allow_test_fixture=allow_test_fixture)
        source = {"kind": captured["source"]["kind"], "reference": parsed["source_reference"]}
        bundle.update(source=source, experiment=experiment_dict(effective), attribution_complete=complete,
            diagnostics={"injected_dependencies": injected, "allocator_peak_scope": "since_process_or_external_reset",
                "workspace_policy": "runtime_owned_complete_ledger_or_no_dataset"},
            provenance={"inventory_sha256": _sha256(captured), "plan_sha256": _sha256(plan),
                "experiment_sha256": _sha256(experiment_dict(effective)), "runtime_metadata_sha256": _sha256(metadata)})
        clock = time.perf_counter_ns if clock_ns is None else clock_ns
        previous_end = 0
        records = []
        total_events = 0
        raw_bytes = 0
        tp, cuda = effective.strategy.tensor_parallel_size, torch.cuda
        for sample in parsed["samples"]:
            indices = list(range(sample["stage_index"] * tp, (sample["stage_index"] + 1) * tp))
            bundle["diagnostics"].update(current_sample_id=sample["sample_id"],
                current_stage_index=sample["stage_index"], current_operation="warmup")
            for _ in range(parsed["warmup_iterations"]):
                adapter.prepare(deepcopy(sample))
                for index in indices:
                    cuda.synchronize(index)
                adapter.run_step(deepcopy(sample))
                for index in indices:
                    cuda.synchronize(index)
            for iteration in range(parsed["iterations"]):
                if total_events >= _MAX_TOTAL_EVENTS or raw_bytes >= _MAX_RAW_BYTES:
                    raise SimulationError("WORKSPACE_CAPTURE_LIMIT", "Workspace collection evidence limit reached")
                identifier = f"{sample['sample_id']}/repeat-{iteration + 1}"
                raw: dict[str, Any] = {"record_id": identifier, "group_id": sample["group_id"],
                    "sample": deepcopy(sample), "device_indices": indices, "status": "PREPARING"}
                bundle["raw_records"].append(raw)
                bundle["diagnostics"]["current_operation"] = "prepare"
                recorder = WorkspaceRecorder(tp, stage_index=sample["stage_index"], epoch_id=identifier,
                    device_indices=indices, max_events=min(max_events, _MAX_TOTAL_EVENTS - total_events),
                    max_event_bytes=min(4 * 1024 * 1024, _MAX_RAW_BYTES - raw_bytes))
                adapter.prepare(deepcopy(sample))
                for index in indices:
                    cuda.synchronize(index)
                raw["allocator_baseline"] = _allocator(cuda, indices, baseline=True)
                begin = getattr(adapter, "begin_workspace_capture", None)
                end = getattr(adapter, "end_workspace_capture", None)
                began = False
                step_error: BaseException | None = None
                try:
                    if callable(begin):
                        began = True
                        bundle["diagnostics"]["current_operation"] = "begin_workspace_capture"
                        begin(deepcopy(sample), recorder)
                    raw["status"] = "RUNNING"
                    start = _clock(clock)
                    if start < previous_end:
                        raise SimulationError("INVALID_TIMING_SAMPLE", "Workspace clock moved backwards")
                    raw["start_monotonic_ns"] = start
                    bundle["diagnostics"]["current_operation"] = "run_step"
                    adapter.run_step(deepcopy(sample))
                    for index in indices:
                        cuda.synchronize(index)
                    stop = _clock(clock)
                    if stop <= start:
                        raise SimulationError("INVALID_TIMING_SAMPLE", "Workspace capture must advance the clock")
                    raw.update(end_monotonic_ns=stop, duration_ms=(stop - start) / 1e6)
                    previous_end = stop
                    raw["allocator_after"] = _allocator(cuda, indices, baseline=False)
                except BaseException as exc:
                    step_error = exc
                    raise
                finally:
                    try:
                        if began and callable(end):
                            bundle["diagnostics"]["current_operation"] = "end_workspace_capture"
                            end(deepcopy(sample), recorder)
                    except Exception as exc:
                        raw["end_capture_error"] = {"type": type(exc).__name__, "message": str(exc)}
                        if step_error is None:
                            raise
                    finally:
                        if step_error is not None:
                            recorder.abort(str(step_error))
                        raw["workspace_ledger"] = recorder.export_capture()
                if complete:
                    peaks = recorder.seal()
                    row = {"record_id": identifier, "group_id": sample["group_id"], "phase": sample["phase"],
                        "stage_index": sample["stage_index"], "query_tokens": list(sample["query_tokens"]),
                        "context_lengths": list(sample["context_lengths"]), "workspace_bytes_per_rank": peaks}
                    if sample["phase"] == "mixed":
                        row["request_phases"] = list(sample["request_phases"])
                    records.append(row)
                    raw["workspace_bytes_per_rank"] = peaks
                else:
                    recorder.abort("Runtime declares partial or unknown Workspace coverage")
                    evidence = getattr(adapter, "workspace_evidence", None)
                    if callable(evidence):
                        payload = json.dumps(evidence(deepcopy(sample)), allow_nan=False)
                        if len(payload.encode()) > 1024 * 1024:
                            raise SimulationError("WORKSPACE_CAPTURE_LIMIT", "Partial Workspace evidence exceeds 1MiB")
                        raw["partial_workspace_evidence"] = json.loads(payload)
                raw.update(status="COMPLETED", attribution_complete=complete, workspace_ledger=recorder.export_capture())
                total_events += len(raw["workspace_ledger"]["events"])
                size = len(json.dumps(raw, allow_nan=False).encode())
                if raw_bytes + size > _MAX_RAW_BYTES:
                    raw.pop("workspace_ledger", None)
                    raw.pop("partial_workspace_evidence", None)
                    raw["workspace_evidence_truncated"] = True
                    raw["attribution_complete"] = False
                    raw["status"] = "INCOMPLETE"
                    raise SimulationError("WORKSPACE_CAPTURE_LIMIT", "Workspace raw evidence exceeds 32MiB")
                raw_bytes += size
        dataset = None
        if complete:
            from .workspace import validate_workspace_dataset
            dataset = validate_workspace_dataset({"schema_version": 1, "kind": "workspace_measurements",
                "dataset_id": parsed["dataset_id"], "identity": profile_identity(effective), "source": source,
                "records": records, "attribution": deepcopy(_ATTRIBUTION)}, allow_test_fixture=allow_test_fixture)
        completed = {**bundle, "status": "COMPLETED", "dataset": dataset}
    except Exception as exc:
        error = exc
    finally:
        if error is not None and recorder is not None:
            recorder.abort(str(error))
            if bundle["raw_records"] and not bundle["raw_records"][-1].get("workspace_evidence_truncated"):
                bundle["raw_records"][-1]["workspace_ledger"] = recorder.export_capture()
        close = getattr(adapter, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                bundle["close_error"] = {"type": type(exc).__name__, "message": str(exc)}
                error = error or exc
    if error is not None:
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        is_oom = isinstance(oom_type, type) and isinstance(error, oom_type)
        bundle["failure"] = {"error_type": type(error).__name__, "message": str(error),
            "failure_kind": "OUT_OF_MEMORY" if is_oom else "RUNTIME_OR_CAPTURE_ERROR"}
        if bundle["raw_records"] and bundle["raw_records"][-1]["status"] != "COMPLETED":
            bundle["raw_records"][-1].update(status="FAILED", error_type=type(error).__name__, error=str(error),
                failure_kind="OUT_OF_MEMORY" if is_oom else "RUNTIME_OR_CAPTURE_ERROR")
        code = "WORKSPACE_OUT_OF_MEMORY" if is_oom else error.code if isinstance(error, SimulationError) else "COLLECTION_FAILED"
        raise CollectionError(code, str(error), artifacts=bundle) from error
    if completed is None:
        raise CollectionError("COLLECTION_FAILED", "Workspace collection did not complete", artifacts=bundle)
    return completed


class PinnedVllmWorkspaceObserver:
    """Observe managed buffer capacity only; never claim full dynamic Workspace."""

    def __init__(self, manager: Any, *, runtime_version: str, allow_test_fixture: bool = False,
                 max_records: int = 10000):
        from .vllm_hooks import API_REVISION, SUPPORTED_VERSION_PREFIX
        if (not isinstance(runtime_version, str) or runtime_version != SUPPORTED_VERSION_PREFIX
                and not runtime_version.startswith(SUPPORTED_VERSION_PREFIX + ".")):
            raise SimulationError("UNSUPPORTED_VLLM_VERSION", "Workspace observer requires the pinned vLLM Runtime")
        fixture = type(manager).__module__ != "vllm.v1.worker.workspace" or type(manager).__name__ != "WorkspaceManager"
        if fixture and not allow_test_fixture:
            raise SimulationError("TEST_FIXTURE_NOT_ALLOWED", "Non-live WorkspaceManager observation requires fixture permission")
        if not fixture:
            actual = importlib.import_module("vllm")
            if getattr(actual, "__version__", None) != runtime_version:
                raise SimulationError("UNSUPPORTED_VLLM_VERSION", "Installed Workspace Runtime differs from the declared version")
            if getattr(getattr(manager, "_device", None), "type", None) not in ("cuda", "hip"):
                raise SimulationError("HARDWARE_UNAVAILABLE", "Live WorkspaceManager must own a device buffer")
        original = getattr(manager, "_ensure_workspace_size", None)
        if not callable(original) or tuple(inspect.signature(original).parameters) != ("required_bytes",):
            raise SimulationError("UNSUPPORTED_VLLM_ABI", "WorkspaceManager allocation signature differs")
        if getattr(manager, "_mlsys_workspace_observer", None) is not None:
            raise SimulationError("CAPTURE_ALREADY_INSTALLED", "WorkspaceManager is already observed")
        self.manager = manager
        self._max = _integer(max_records, "max_records", 1)
        if self._max > 100000:
            raise SimulationError("INVALID_WORKSPACE_CAPTURE", "Observer record bound is too large")
        self._records: list[dict] = []
        self._dropped = 0
        self._closed = False
        self._saved = vars(manager).get("_ensure_workspace_size", _MISSING)
        self._header = {"schema_version": 1, "scope": "managed_scratch_capacity", "complete": False,
            "source": {"kind": "test_fixture" if fixture else "measured",
                       "reference": "Pinned WorkspaceManager storage observation; incomplete dynamic attribution"},
            "api_revision": API_REVISION, "runtime_version": runtime_version,
            "old_retained_storage_observable": False,
            "limitations": ["Only currently referenced manager slots are visible; old views may retain replaced storage.",
                            "Unmanaged activations, backend scratch, graph pools and direct device allocations are not covered."]}
        self._snapshot()
        @functools.wraps(original)
        def observe(required_bytes):
            try:
                output = original(required_bytes)
            except BaseException as exc:
                self._record({"required_bytes": required_bytes if type(required_bytes) is int
                              and 0 <= required_bytes <= 2**63 - 1 else None,
                              "status": "FAILED", "error_type": type(exc).__name__})
                raise
            try:
                self._record({"required_bytes": _integer(required_bytes, "required_bytes"),
                              "status": "COMPLETED", "snapshot": self._snapshot()})
            except Exception as exc:
                self._record({"status": "OBSERVATION_FAILED", "error_type": type(exc).__name__})
            return output
        try:
            manager._ensure_workspace_size = observe
            manager._mlsys_workspace_observer = self
        except Exception:
            self.close()
            raise

    def _snapshot(self) -> dict:
        manager = self.manager
        count = _integer(getattr(manager, "_num_ubatches", None), "num_ubatches", 1)
        lanes = _integer(getattr(manager, "_num_lanes", None), "num_lanes", 1)
        slots = vars(manager).get("_current_workspaces")
        if count * lanes > 1024 or not isinstance(slots, list) or len(slots) != count * lanes or type(manager._locked) is not bool:
            raise SimulationError("UNSUPPORTED_VLLM_ABI", "WorkspaceManager slots or lock differ from the pinned ABI")
        capacities = []
        for index, tensor in enumerate(slots):
            capacity = (0 if tensor is None else _integer(
                _integer(tensor.numel(), "workspace.numel") * _integer(tensor.element_size(), "workspace.element_size", 1),
                "workspace.capacity_bytes"))
            capacities.append({"slot": index, "capacity_bytes": capacity})
        return {"num_ubatches": count, "num_lanes": lanes, "locked": manager._locked,
                "slots": capacities, "current_managed_capacity_bytes": _integer(
                    sum(item["capacity_bytes"] for item in capacities), "current_managed_capacity_bytes")}

    def _record(self, value: dict) -> None:
        if self._closed:
            return
        if len(self._records) < self._max:
            self._records.append(value)
        else:
            self._dropped += 1

    def export_capture(self) -> dict:
        return {**deepcopy(self._header), "records": deepcopy(self._records), "dropped_records": self._dropped,
                "closed": self._closed, "hardware_validated": False}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._saved is _MISSING:
            vars(self.manager).pop("_ensure_workspace_size", None)
        else:
            self.manager._ensure_workspace_size = self._saved
        if getattr(self.manager, "_mlsys_workspace_observer", None) is self:
            delattr(self.manager, "_mlsys_workspace_observer")
