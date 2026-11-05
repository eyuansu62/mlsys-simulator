"""Pinned GPUWorker raw Workspace observer; imports vLLM only in a worker.

No individual observation from this module establishes Workspace completeness.
The worker retains original model/PP calls and records their actual storage and
allocator lifetimes. Formal execution never receives an extra communication
wait, reap or device synchronization from this observer.
"""

from __future__ import annotations

from copy import deepcopy
import ctypes
import functools
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from vllm.v1.worker.gpu_worker import Worker


UINT64_MAX = 2**64 - 1
MAX_TRANSPORT_EVENTS = 4096
_MISSING = object()


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_gzip(path: Path, value: Any) -> None:
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding(function: Any) -> dict[str, Any]:
    original = inspect.unwrap(function)
    path = Path(inspect.getfile(original)).resolve()
    return {
        "qualified_name": original.__module__ + "." + original.__qualname__,
        "source_reference": str(path),
        "source_sha256": hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        "file_sha256": file_sha256(path),
    }


def compact_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Remove stack text only; preserve every segment, block and history event."""
    result = dict(snapshot)
    result["segments"] = [
        {**segment, "blocks": [{k: v for k, v in block.items() if k != "frames"}
                               for block in segment["blocks"]]}
        for segment in snapshot["segments"]
    ]
    result["device_traces"] = [
        [{k: v for k, v in event.items() if k != "frames"} for event in events]
        for events in snapshot["device_traces"]
    ]
    return result


def tensor_descriptor(tensor: Any, path: str, *, tensor_name: str | None = None,
                      cursor: Any = None) -> dict[str, Any]:
    """Read actual Tensor/storage metadata; return no live Tensor references."""
    before = cursor() if cursor is not None else None
    storage = tensor.untyped_storage()
    result = {
        "path": path, "tensor_name": tensor_name,
        "pointer": int(storage.data_ptr()), "bytes": int(storage.nbytes()),
        "tensor_data_ptr": int(tensor.data_ptr()), "shape": list(tensor.shape),
        "stride": list(tensor.stride()), "tensor_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype), "element_size": int(tensor.element_size()),
        "device_index": tensor.device.index,
    }
    after = cursor() if cursor is not None else None
    result.update(metadata_cursor_before=before, metadata_cursor_after=after,
                  metadata_cursor=after,
                  metadata_cursor_stable=before is not None and before == after)
    return result


def group_descriptor(group: Any) -> dict[str, Any]:
    return {"object_id": id(group), "class_name": type(group).__module__ + "." + type(group).__name__,
            "ranks": list(group.ranks), "rank_in_group": int(group.rank_in_group),
            "world_size": int(group.world_size), "unique_name": getattr(group, "unique_name", None)}


def mapped_library_segments(library_path: str, *, maps_text: str | None = None) -> list[dict[str, Any]]:
    """Observe the exact loaded library's file mappings; never infer code ranges."""
    target = Path(library_path).resolve(strict=True)
    family = target.name.split(".so", 1)[0] + ".so"
    text = Path("/proc/self/maps").read_text() if maps_text is None else maps_text
    if len(text.encode()) > 4 * 1024 * 1024:
        raise ValueError("Process mapping table exceeds the observation bound")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[-1].startswith("/"):
            continue
        original_path = parts[-1]
        candidate = Path(original_path.removesuffix(" (deleted)"))
        if not candidate.name.startswith(family):
            continue
        if original_path.endswith(" (deleted)"):
            raise ValueError("A mapped runtime library was deleted; current file bytes cannot bind that mapping")
        if candidate.resolve(strict=True) != target:
            continue
        start, end = (int(value, 16) for value in parts[0].split("-"))
        offset = int(parts[2], 16)
        if not 0 < start < end or len(parts[1]) != 4:
            raise ValueError("Actual library mapping has invalid bounds or permissions")
        rows.append({"start": start, "end": end, "offset": offset,
                     "permissions": parts[1], "path": str(target)})
    if not rows or not any("x" in row["permissions"] for row in rows):
        raise ValueError("The exact loaded runtime file has no observed executable mapping")
    return sorted(rows, key=lambda row: row["start"])


def observe_hardware_inventory(loaded_rccl: dict[str, Any]) -> dict[str, Any]:
    """Keep the existing probe's actual facts/source, including unavailable data."""
    from mlsys_sim.hardware import probe_rocm

    try:
        return probe_rocm(rccl_library=loaded_rccl["path"])
    except Exception as exc:
        # An exception is raw failed evidence, never a fabricated measured/READY
        # inventory. The root merge must reject absent required hardware facts.
        return {"kind": "rocm_inventory_probe_failure", "status": "ERROR",
                "error": {"type": type(exc).__name__, "message": str(exc)}}


def cpu_return_witness(value: Any, runner: Any) -> dict[str, Any]:
    """Observe only stored fields, never a lazy .tensors property."""
    import torch

    seen: set[int] = set()

    def count(item: Any) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            return int(item.device.type == "cuda")
        if isinstance(item, dict):
            return sum(count(child) for child in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(child) for child in item)
        if type(item).__module__.startswith("vllm.") and hasattr(item, "__dict__"):
            return sum(count(child) for child in vars(item).values())
        return 0

    return {"type": type(value).__module__ + "." + type(value).__name__,
            "gpu_tensor_count": count(value),
            "execute_state_is_none": runner.execute_model_state is None,
            "prev_sampled_is_none": runner.input_batch.prev_sampled_token_ids is None}


def install_workerproc_shutdown_observer(worker_proc_type: Any) -> dict[str, Any]:
    """Observe the original MP teardown once; never take over Runtime cleanup."""
    original = worker_proc_type.shutdown
    if getattr(original, "_mlsys_workspace_shutdown_observer", False):
        return deepcopy(original._mlsys_workspace_original_shutdown_source)
    binding = source_binding(original)

    @functools.wraps(original)
    def observe(proc: Any, *args: Any, **kwargs: Any) -> Any:
        wrapper = vars(proc).get("worker")
        actual = vars(wrapper).get("worker") if hasattr(wrapper, "__dict__") else None
        receipt: dict[str, Any] = {"process_id": os.getpid(), "thread_id": threading.get_native_id(),
            "source": deepcopy(binding), "original_returned": False,
            "clock": "CLOCK_MONOTONIC",
            "native_clock": "CLOCK_MONOTONIC_RAW",
            "started_monotonic_ns": time.perf_counter_ns(),
            "started_monotonic_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
            "scope": "original_WorkerProc_shutdown_including_model_and_distributed_environment"}
        try:
            result = original(proc, *args, **kwargs)
            receipt["original_returned"] = True
            receipt["return_type"] = type(result).__module__ + "." + type(result).__name__
            return result
        except BaseException as exc:
            receipt["exception"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            receipt["ended_monotonic_ns"] = time.perf_counter_ns()
            receipt["ended_monotonic_raw_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            if actual is not None and getattr(actual, "_ws_deferred_payload", None) is not None:
                try:
                    actual._ws_after_natural_shutdown(receipt)
                except BaseException as exc:
                    # Preserve the original Runtime outcome. Missing/failed
                    # final observation remains a failed capture for the parent.
                    try:
                        actual._ws_write_shutdown_observer_error(receipt, exc)
                    except BaseException:
                        pass

    observe._mlsys_workspace_shutdown_observer = True  # type: ignore[attr-defined]
    observe._mlsys_workspace_original_shutdown_source = deepcopy(binding)  # type: ignore[attr-defined]
    worker_proc_type.shutdown = observe
    return binding


class WorkspaceWorker(Worker):
    """Explicit worker_cls subclass; all actual base Worker calls run once."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        plan_path = Path(os.environ["MLSYS_WORKSPACE_STAGE_PLAN"]).resolve()
        self._ws_plan_path = plan_path
        self._ws_plan_sha = file_sha256(plan_path)
        self._ws_plan = json.loads(plan_path.read_text())
        from workspace_stage_session import validate_native_finalization

        self._ws_native_finalization = validate_native_finalization(
            self._ws_plan["mode"], self._ws_plan["cohort"], self._ws_plan["repeats"],
            self._ws_plan.get("native_finalization", "worker_shutdown"))
        self._ws_root = Path(self._ws_plan["out"]) / ("worker-pid-" + str(os.getpid()))
        self._ws_root.mkdir()
        self._ws_native: Any = None
        self._ws_bridge: Any = None
        self._ws_cohort = "startup"
        self._ws_current: dict[str, Any] | None = None
        self._ws_native_active: int | None = None
        self._ws_torch_active: int | None = None
        self._ws_records: list[dict[str, Any]] = []
        self._ws_events: list[dict[str, Any]] = []
        self._ws_event_counter = 0
        self._ws_errors: list[str] = []
        self._ws_ctor: list[dict[str, Any]] = []
        self._ws_pg_scopes: list[dict[str, Any]] = []
        self._ws_pg_prewarm: list[dict[str, Any]] = []
        self._ws_pending_names: dict[tuple[int, ...], list[str]] = {}
        self._ws_transport_sources: dict[str, Any] = {}
        self._ws_pp: Any = None
        self._ws_hook: Any = None
        self._ws_closed = False
        self._ws_closed_payload: dict[str, Any] | None = None
        self._ws_closed_summary: dict[str, Any] | None = None
        self._ws_deferred_payload: dict[str, Any] | None = None
        self._ws_pending_close_ack: dict[str, Any] | None = None
        self._ws_attribute_patches: list[tuple[Any, str, Any, Any]] = []
        self._ws_stdthread_control: dict[str, Any] | None = None
        self._ws_index = 0
        self._ws_sources = {"worker_source_reference": str(Path(__file__).resolve()),
                            "worker_source_sha256": file_sha256(Path(__file__).resolve()),
                            "base_worker": source_binding(Worker.execute_model)}
        super().__init__(*args, **kwargs)
        self._ws_shutdown_source = None
        if self._ws_plan["mode"] != "single":
            from vllm.v1.executor.multiproc_executor import WorkerProc

            self._ws_shutdown_source = install_workerproc_shutdown_observer(WorkerProc)

    def _ws_load_native(self) -> Any:
        if self._ws_native is None:
            lib = ctypes.CDLL(self._ws_plan["native_probe_library"])
            lib.mlsys_probe_begin.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
            lib.mlsys_probe_end.argtypes = [ctypes.c_uint64]
            lib.mlsys_probe_write.argtypes = [ctypes.c_char_p]
            self._ws_native = lib
        return self._ws_native

    def _ws_begin_native(self, epoch: int, label: str) -> None:
        if self._ws_native_active is not None:
            raise RuntimeError("Native Workspace observation epochs cannot nest")
        if self._ws_load_native().mlsys_probe_begin(epoch, label.encode()) != 0:
            raise RuntimeError("Native Workspace epoch begin failed")
        self._ws_native_active = epoch

    def _ws_end_native(self, epoch: int) -> None:
        if self._ws_native_active != epoch or self._ws_native.mlsys_probe_end(epoch) != 0:
            raise RuntimeError("Native Workspace epoch end failed")
        self._ws_native_active = None

    def init_device(self) -> None:
        import torch
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        self._ws_prime_stdthread_control()
        original = PyNcclCommunicator.__init__
        constructor_source = source_binding(original)

        @functools.wraps(original)
        def observe(comm: Any, *args: Any, **kwargs: Any) -> None:
            # Bind the actual source signature before calling it; no opaque
            # constructor argument or communicator field is inferred.
            inspect.signature(original).bind(comm, *args, **kwargs)
            epoch = 6001 + len(self._ws_ctor)
            label = "explicit_PyNcclCommunicator_constructor_" + str(epoch)
            self._ws_begin_native(epoch, label)
            row: dict[str, Any] = {"epoch_id": epoch, "epoch_label": label,
                "entrypoint": constructor_source["qualified_name"], "comm_only": True,
                "constructor_source": constructor_source, "status": "FAILED"}
            try:
                original(comm, *args, **kwargs)
                actual = vars(comm)
                row.update(world_size=actual.get("world_size"), rank=actual.get("rank"),
                    available=actual.get("available"), disabled=actual.get("disabled"))
                handle = actual.get("comm")
                if handle is not None:
                    if not isinstance(handle, ctypes.c_void_p) or type(handle.value) is not int:
                        raise RuntimeError("Actual ncclComm_t is not the pinned nonempty void pointer")
                    row["communicator_handle"] = handle.value
                else:
                    row["communicator_handle"] = None
                device = actual.get("device")
                row["logical_device_index"] = getattr(device, "index", None)
                # This is a declared startup constructor control, never formal
                # PP receive synchronization. The original constructor itself
                # includes a one-element all-reduce and stream synchronization.
                torch.cuda.synchronize(torch.cuda.current_device())
                row["status"] = "RETURNED"
            except BaseException as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)
                raise
            finally:
                self._ws_end_native(epoch)
                self._ws_ctor.append(row)

        PyNcclCommunicator.__init__ = observe
        try:
            super().init_device()
        finally:
            PyNcclCommunicator.__init__ = original

    def _ws_prime_stdthread_control(self) -> None:
        """Pure CPU native dispatch control, before device/NCCL initialization."""
        native = self._ws_load_native()
        native.mlsys_probe_stdthread_control.argtypes = []
        native.mlsys_probe_stdthread_control.restype = ctypes.c_int
        code = int(native.mlsys_probe_stdthread_control())
        if code != 0:
            raise RuntimeError("Observed std::thread dispatch control failed: " + str(code))
        native.mlsys_probe_write_stdthread_control.argtypes = [ctypes.c_char_p]
        native.mlsys_probe_write_stdthread_control.restype = ctypes.c_int
        path = self._ws_root / "stdthread-control.json"
        code = int(native.mlsys_probe_write_stdthread_control(str(path).encode()))
        if code != 0:
            raise RuntimeError("Observed std::thread dispatch control output failed: " + str(code))
        control = json.loads(path.read_text())
        header = Path(control["header_reference"]).resolve(strict=True)
        if file_sha256(header) != control["header_sha256"]:
            raise RuntimeError("Native std::thread ABI header differs from its actual pinned source")
        module = Path(control["module_path"]).resolve(strict=True)
        self._ws_stdthread_control = {"control": control,
            "header": {"path": control["header_reference"], "sha256": file_sha256(header)},
            "runtime_library": {"path": control["module_path"], "sha256": file_sha256(module),
                                "mapped_segments": mapped_library_segments(str(module))}}

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        import torch
        from mlsys_sim import load_experiment
        from mlsys_sim.vllm_hooks import install_runner_hook
        from vllm.distributed import get_pp_group, get_tp_group

        if load_dummy_weights:
            raise RuntimeError("Raw live Workspace acquisition cannot use dummy weights")
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="all", max_entries=100000, clear_history=True)
        self._ws_begin_native(5001, "explicit_ATen_LT_workspace_initialization")
        lib = ctypes.CDLL(str(Path(torch.__file__).parent / "lib/libtorch_hip.so"))
        self._ws_getptr = getattr(lib, "_ZN2at4cuda22getCUDABlasLtWorkspaceEv")
        self._ws_getptr.argtypes = []
        self._ws_getptr.restype = ctypes.c_void_p
        self._ws_getsize = getattr(lib, "_ZN2at4cuda26getCUDABlasLtWorkspaceSizeEv")
        self._ws_getsize.argtypes = []
        self._ws_getsize.restype = ctypes.c_size_t
        self._ws_lt = {"pointer": self._ws_getptr(), "bytes": self._ws_getsize(),
                       "thread_id": threading.get_native_id()}
        torch.cuda.synchronize(self.device)
        self._ws_end_native(5001)
        self._ws_prewarm_process_groups()
        super().load_model(load_dummy_weights=load_dummy_weights)
        experiment = load_experiment(self._ws_plan["bound_experiment"])
        pp, tp = get_pp_group(), get_tp_group()
        self._ws_pp = pp
        self._ws_experiment = experiment
        self._ws_placement = {"process_id": os.getpid(), "global_rank": int(self.rank),
            "stage_index": int(pp.rank_in_group), "tp_rank": int(tp.rank_in_group),
            "tensor_parallel_size": int(tp.world_size), "pipeline_parallel_size": int(pp.world_size),
            "world_size": int(tp.world_size * pp.world_size), "logical_device_index": int(self.device.index)}
        self._ws_hook = install_runner_hook(self.model_runner, experiment,
            session_id=self._ws_plan["session_id"], stage_index=pp.rank_in_group,
            tp_rank=tp.rank_in_group, output_dir=self._ws_root / "stage-captures", max_records=256,
            source_reference="Actual Worker-level Workspace observation; component remains raw")
        self._ws_install_transport_observers()
        self._ws_install_pg_post_observers()

    def _ws_prewarm_process_groups(self) -> None:
        """Controlled communication before model load; no formal-path changes."""
        import torch
        from vllm.distributed import get_pp_group, get_tp_group

        tp, pp = get_tp_group(), get_pp_group()
        if tp.world_size == pp.world_size == 1:
            return
        if (tp.world_size, pp.world_size) not in {(2, 1), (1, 2)}:
            raise RuntimeError("Communication prewarm supports only the actual two-card TP2 or PP2 domains")
        if self._ws_native_active is not None:
            raise RuntimeError("Communication prewarm cannot nest an SDK epoch")
        coordinator = tp if tp.world_size == 2 else pp
        group = coordinator.device_group
        ranks = list(torch.distributed.get_process_group_ranks(group))
        rank = int(torch.distributed.get_rank(group))
        backend = str(torch.distributed.get_backend(group))
        if backend != "nccl" or len(ranks) != 2 or rank != int(coordinator.rank_in_group):
            raise RuntimeError("Controlled communication requires the real two-rank ProcessGroupNCCL")
        kind = "tp_allgather" if tp.world_size == 2 else "pp_p2p"
        epoch = 6501 if kind == "tp_allgather" else 6502
        label = "controlled_torch_process_group_" + kind + "_prewarm_v1"
        functions = ([torch.distributed.all_gather_into_tensor] if kind == "tp_allgather"
                     else [torch.distributed.isend if rank == 0 else torch.distributed.irecv])
        bindings = [source_binding(function) for function in functions]
        row: dict[str, Any] = {"kind": "torch_process_group_prewarm", "control": kind,
            "epoch_id": epoch, "epoch_label": label, "entrypoint": bindings[0]["qualified_name"],
            "source_binding": bindings[0], "comm_only": True, "process_id": os.getpid(),
            "logical_device_index": int(self.device.index),
            "pg": {"object_id": id(group), "backend": backend,
                   "group_name": getattr(group, "group_name", None), "global_ranks": ranks,
                   "world_size": len(ranks), "rank_in_group": rank},
            "operations": [], "input_buffers": [], "output_buffers": [], "status": "FAILED",
            "scope": "independent_startup_communication_before_model_load"}
        inputs: list[Any] = []
        outputs: list[Any] = []
        works: list[Any] = []
        try:
            # All allocations and GPU fills complete before opening the SDK
            # constructor/first-use scope. The buffers are control data only.
            if kind == "tp_allgather":
                inputs = [torch.full((75968,), float(rank + 1), dtype=torch.bfloat16, device=self.device)]
                outputs = [torch.empty((151936,), dtype=torch.bfloat16, device=self.device)]
                row["input_buffers"] = [tensor_descriptor(inputs[0], "control.tp.input", tensor_name="input")]
                row["output_buffers"] = [tensor_descriptor(outputs[0], "control.tp.output", tensor_name="output")]
                row["operations"].append({"qualified_name": bindings[0]["qualified_name"],
                    "source_binding": bindings[0], "async_op": False, "send_count": 75968,
                    "receive_count": 151936, "group_object_id": id(group)})
            else:
                buffers = [torch.full((512, 2048), value if rank == 0 else -1.0,
                    dtype=torch.bfloat16, device=self.device) for value in (17.0, 19.0)]
                if rank == 0:
                    inputs = buffers
                else:
                    outputs = buffers
                names = ("hidden_states", "residual")
                descriptors = [tensor_descriptor(tensor, "control.pp." + name, tensor_name=name)
                               for name, tensor in zip(names, buffers)]
                row["input_buffers" if rank == 0 else "output_buffers"] = descriptors
                for name in names:
                    row["operations"].append({"qualified_name": bindings[0]["qualified_name"],
                        "source_binding": bindings[0], "async_op": True, "tensor_name": name,
                        "peer": ranks[1] if rank == 0 else ranks[0],
                        "global_dst" if rank == 0 else "global_src": ranks[1] if rank == 0 else ranks[0],
                        "tag": 0, "element_count": 512 * 2048, "group_object_id": id(group)})
            torch.cuda.synchronize(self.device)
            row["preallocated_before_epoch"] = True
            row["preallocation_completed_monotonic_raw_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            self._ws_begin_native(epoch, label)
            try:
                if kind == "tp_allgather":
                    torch.distributed.all_gather_into_tensor(outputs[0], inputs[0], group=group, async_op=False)
                elif rank == 0:
                    for tensor in inputs:
                        works.append(torch.distributed.isend(tensor, dst=ranks[1], group=group, tag=0))
                else:
                    for tensor in outputs:
                        works.append(torch.distributed.irecv(tensor, src=ranks[0], group=group, tag=0))
                for work in works:
                    work.wait()
                if works:
                    del work
                torch.cuda.synchronize(self.device)
                row["device_sync_completed"] = True
            finally:
                self._ws_end_native(epoch)
            # Validation is a real full CPU copy after the communication epoch.
            # It adds no validation kernels or allocator activity to that scope.
            checks = []
            if kind == "tp_allgather":
                values = outputs[0].cpu().reshape(-1).tolist()
                for peer in range(2):
                    chunk = values[peer * 75968:(peer + 1) * 75968]
                    expected = float(peer + 1)
                    checks.append({"peer_rank": peer, "expected_count": 75968, "observed_count": len(chunk),
                        "expected_value": expected, "expected_checksum": expected * 75968,
                        "observed_checksum": sum(chunk), "mismatch_count": sum(value != expected for value in chunk)})
                del values, chunk
            else:
                for name, tensor, expected in zip(("hidden_states", "residual"), inputs or outputs, (17.0, 19.0)):
                    values = tensor.cpu().reshape(-1).tolist()
                    checks.append({"tensor_name": name, "expected_count": 512 * 2048,
                        "observed_count": len(values), "expected_value": expected,
                        "expected_checksum": expected * 512 * 2048, "observed_checksum": sum(values),
                        "mismatch_count": sum(value != expected for value in values)})
                    del values
                del tensor
            success = all(check["expected_count"] == check["observed_count"]
                          and check["expected_checksum"] == check["observed_checksum"]
                          and check["mismatch_count"] == 0 for check in checks)
            row["payload_validation"] = {"success": success,
                "method": "full_CPU_copy_compare_after_native_epoch", "checks": checks}
            if not success:
                raise RuntimeError("Actual startup communication payload failed the full correctness check")
            row["status"] = "COMPLETED"
        except BaseException as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            # No control Tensor, Work or helper closure enters the model or the
            # formal PP queue. Their actual frees stay in the complete trace.
            inputs.clear()
            outputs.clear()
            works.clear()
            if kind == "pp_p2p" and "buffers" in locals():
                buffers.clear()
            self._ws_pg_prewarm.append(row)
            write_new(self._ws_root / ("communication-prewarm-" + kind + ".json"), row)

    def _ws_install_pg_post_observers(self) -> None:
        """Observe actual lazy PG initialization without an added wait/sync."""
        import torch

        if self._ws_pp.world_size == 1:
            return
        for operation in ("isend", "irecv"):
            original = getattr(torch.distributed, operation)
            binding = source_binding(original)

            def make_observer(method: Any, name: str, source: dict[str, Any]) -> Any:
                @functools.wraps(method)
                def observe(*args: Any, **kwargs: Any) -> Any:
                    bound = inspect.signature(method).bind(*args, **kwargs)
                    group = bound.arguments.get("group")
                    if group is not self._ws_pp.device_group:
                        return method(*args, **kwargs)
                    scope: dict[str, Any] = {"operation": name, "cohort": self._ws_cohort,
                        "source": source, "process_id": os.getpid(),
                        "group_object_id": id(group), "backend": str(torch.distributed.get_backend(group)),
                        "group_name": getattr(group, "group_name", None),
                        "group_ranks": list(torch.distributed.get_process_group_ranks(group)),
                        "added_device_synchronization": False,
                        "scope": "host_API_return_not_device_completion", "status": "STARTED"}
                    dedicated = self._ws_cohort != "formal" and self._ws_native_active is None
                    epoch = 7001 + len(self._ws_pg_scopes)
                    if dedicated:
                        scope["epoch_id"] = epoch
                        scope["epoch_label"] = "controlled_actual_PP_" + name + "_host_post_" + str(epoch)
                        self._ws_begin_native(epoch, scope["epoch_label"])
                    else:
                        scope["epoch_id"] = self._ws_native_active
                        scope["dedicated_initialization_epoch"] = False
                    try:
                        result = method(*args, **kwargs)
                        scope.update(status="RETURNED", returned_handle_id=id(result),
                            returned_handle_class=type(result).__module__ + "." + type(result).__name__)
                        return result
                    except BaseException as exc:
                        scope.update(status="RAISED", error=type(exc).__name__ + ": " + str(exc))
                        raise
                    finally:
                        if dedicated:
                            self._ws_end_native(epoch)
                        self._ws_pg_scopes.append(scope)
                return observe

            wrapper = make_observer(original, operation, binding)
            self._ws_attribute_patches.append((torch.distributed, operation,
                vars(torch.distributed).get(operation, _MISSING), wrapper))
            setattr(torch.distributed, operation, wrapper)

    def _ws_cursor(self) -> int | None:
        if self._ws_bridge is None:
            return None
        value = int(self._ws_bridge.mlsys_torch_probe_cursor())
        if value == UINT64_MAX:
            raise RuntimeError("Torch observer cursor is uncertain during concurrent callbacks")
        return value

    def _ws_install_bridge(self) -> None:
        if self._ws_bridge is not None:
            return

        bridge = ctypes.CDLL(self._ws_plan["torch_probe_library"])
        bridge.mlsys_torch_probe_install.argtypes = [ctypes.c_uint64]
        bridge.mlsys_torch_probe_begin.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        bridge.mlsys_torch_probe_end.argtypes = [ctypes.c_uint64]
        bridge.mlsys_torch_probe_write.argtypes = [ctypes.c_char_p]
        bridge.mlsys_torch_probe_cursor.argtypes = []
        bridge.mlsys_torch_probe_cursor.restype = ctypes.c_uint64
        if bridge.mlsys_torch_probe_install(131072) != 0:
            raise RuntimeError("Torch allocator observer could not install on the actual inference thread")
        self._ws_bridge = bridge
        self._ws_install_thread = threading.get_native_id()
        initial = {"process_id": os.getpid(), "thread_id": self._ws_install_thread,
                   **self._ws_take_snapshot(), "pending_pp": self._ws_pending()}
        write_gzip(self._ws_root / "startup.json.gz", initial)

    def _ws_take_snapshot(self) -> dict[str, Any]:
        import torch

        before = self._ws_cursor()
        snapshot = compact_snapshot(torch.cuda.memory._snapshot())
        after = self._ws_cursor()
        return {"snapshot": snapshot, "snapshot_cursor_before": before,
                "snapshot_cursor_after": after, "cursor": after}

    def _ws_pending(self) -> list[dict[str, Any]]:
        if self._ws_pp is None:
            return []
        pending = vars(self._ws_pp).get("_pending_isends", ())
        result = []
        for index, (handles, tensors) in enumerate(pending):
            ids = [id(handle) for handle in handles]
            names = self._ws_pending_names.get(tuple(ids), [])
            result.append({"entry_index": index, "handle_ids": ids,
                "tensor_descriptors": [tensor_descriptor(tensor, "pp.pending[" + str(index) + "][" + str(i) + "]",
                    tensor_name=names[i] if i < len(names) else None, cursor=self._ws_cursor)
                    for i, tensor in enumerate(tensors)]})
        return result

    def _ws_describe_dict(self, value: Any, prefix: str) -> list[dict[str, Any]]:
        import torch

        if not isinstance(value, dict):
            raise RuntimeError("Pinned PP tensor dictionary is not a plain dict")
        return [tensor_descriptor(tensor, prefix + "." + str(name), tensor_name=name, cursor=self._ws_cursor)
                for name, tensor in value.items() if isinstance(tensor, torch.Tensor)]

    def _ws_install_transport_observers(self) -> None:
        group = self._ws_pp
        self._ws_group_descriptor = group_descriptor(group)
        if group.world_size == 1:
            return
        for operation in ("isend_tensor_dict", "irecv_tensor_dict", "_reap_completed_isends"):
            original = getattr(group, operation)
            binding = source_binding(original)
            self._ws_transport_sources[operation] = binding

            def make_observer(method: Any, name: str, source: dict[str, Any]) -> Any:
                @functools.wraps(method)
                def observe(*args: Any, **kwargs: Any) -> Any:
                    self._ws_event_counter += 1
                    event: dict[str, Any] = {"event_id": self._ws_event_counter,
                        "operation": name, "cohort": self._ws_cohort,
                        "epoch_id": self._ws_current["epoch_id"] if self._ws_current else None,
                        "observer_epoch_id": self._ws_torch_active,
                        "capture_iteration": self._ws_index, "group": group_descriptor(group),
                        "source_method_sha256": source["source_sha256"], "tensors": [],
                        "status": "STARTED"}
                    try:
                        event["before_pending"] = self._ws_pending()
                        if name == "isend_tensor_dict":
                            bound = inspect.signature(method).bind(*args, **kwargs)
                            event["tensors"] = self._ws_describe_dict(bound.arguments["tensor_dict"], "pp.isend")
                        event["before_cursor"] = self._ws_cursor()
                    except Exception as exc:
                        event["metadata_error_before"] = type(exc).__name__ + ": " + str(exc)
                        self._ws_errors.append(event["metadata_error_before"])
                    try:
                        result = method(*args, **kwargs)
                    except BaseException as exc:
                        event.update(status="RAISED", error=type(exc).__name__ + ": " + str(exc))
                        raise
                    else:
                        event["status"] = "RETURNED"
                        try:
                            event["after_cursor"] = self._ws_cursor()
                            if name == "isend_tensor_dict":
                                event["handles"] = [id(handle) for handle in result]
                                names = [row["tensor_name"] for row in event["tensors"] if all(row["shape"])]
                                self._ws_pending_names[tuple(event["handles"])] = names
                                event["actual_dict_tensor_names_in_order"] = names
                            elif name == "irecv_tensor_dict":
                                event["tensors"] = self._ws_describe_dict(result[0], "pp.irecv")
                                event["handles"] = [id(handle) for handle in result[1]]
                            event["after_pending"] = self._ws_pending()
                        except Exception as exc:
                            event["metadata_error_after"] = type(exc).__name__ + ": " + str(exc)
                            self._ws_errors.append(event["metadata_error_after"])
                        return result
                    finally:
                        if len(self._ws_events) >= MAX_TRANSPORT_EVENTS:
                            raise RuntimeError("PP transport observation limit exceeded; original data remains raw")
                        self._ws_events.append(event)
                return observe

            wrapper = make_observer(original, operation, binding)
            self._ws_attribute_patches.append((group, operation, vars(group).get(operation, _MISSING), wrapper))
            setattr(group, operation, wrapper)

    def _ws_release_observer_references(self) -> list[dict[str, Any]]:
        """Restore only observer-added attributes; leave original Runtime state."""
        receipt = []
        for owner, name, previous, wrapper in reversed(self._ws_attribute_patches):
            restored = vars(owner).get(name, _MISSING) is wrapper
            if restored:
                if previous is _MISSING:
                    vars(owner).pop(name, None)
                else:
                    setattr(owner, name, previous)
            else:
                self._ws_errors.append("Observer attribute changed before restore: " + name)
            receipt.append({"owner_class": type(owner).__module__ + "." + type(owner).__name__,
                "attribute": name, "had_own_attribute": previous is not _MISSING,
                "previous_value_type": None if previous is _MISSING else type(previous).__name__,
                "restored": restored})
        self._ws_attribute_patches.clear()
        self._ws_pending_names.clear()
        # Both references were introduced only by instrumentation. Natural
        # VLLM globals/worker state continue owning their original objects.
        self._ws_pp = None
        self._ws_hook = None
        return receipt

    def _ws_census(self) -> list[dict[str, Any]]:
        import torch

        runner = self.model_runner
        owners: list[dict[str, Any]] = []
        seen: set[int] = set()

        def tensor(value: Any, path: str, category: str) -> None:
            if value.device.type == "cuda":
                owners.append({**tensor_descriptor(value, path), "category": category})

        def walk(value: Any, path: str, category: str, depth: int = 0) -> None:
            if isinstance(value, torch.Tensor):
                tensor(value, path, category)
                return
            if depth > 7 or id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, path + "[" + str(key) + "]", category, depth + 1)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    walk(child, path + "[" + str(index) + "]", category, depth + 1)
            elif (not isinstance(value, torch.nn.Module)
                  and (value is runner or type(value).__module__.startswith("vllm."))
                  and hasattr(value, "__dict__")):
                for key, child in vars(value).items():
                    if key not in {"model", "kv_caches", "_mlsys_internal_capture", "vllm_config"}:
                        walk(child, path + "." + key, category, depth + 1)

        for name, value in runner.model.named_parameters():
            tensor(value, "model.parameter:" + name, "weights")
        for name, value in runner.model.named_buffers():
            tensor(value, "model.buffer:" + name, "runtime")
        walk(runner.kv_caches, "runner.kv_caches", "kv_pool")
        walk(runner, "runner", "runtime")
        before = torch.cuda.memory._snapshot()
        pointer, size = self._ws_getptr(), self._ws_getsize()
        after = torch.cuda.memory._snapshot()
        index = int(self.device.index)
        delta = after["device_traces"][index][len(before["device_traces"][index]):]
        if any(row["action"] != "snapshot" for row in delta):
            raise RuntimeError("LT Workspace getter caused an allocator mutation during owner observation")
        if pointer != self._ws_lt["pointer"] or size != self._ws_lt["bytes"]:
            raise RuntimeError("LT getter binding changed from its actual constructor")
        owners.append({"path": "ATen.getCUDABlasLtWorkspace", "category": "runtime",
                       "pointer": pointer, "bytes": size, "element_size": 1,
                       "binding": "verified_exported_getter_no_allocation"})
        return owners

    def _ws_start(self) -> None:
        self._ws_install_bridge()
        if self._ws_current is not None:
            raise RuntimeError("Another Workspace step began before the prior sampling boundary")
        if threading.get_native_id() != self._ws_install_thread:
            raise RuntimeError("Actual inference thread differs from Torch observer installation")
        self._ws_index += 1
        epoch = 10000 + self._ws_index
        owners = self._ws_census()
        snapshot = self._ws_take_snapshot()
        row = {"epoch_id": epoch, "capture_iteration": self._ws_index,
               "cohort": self._ws_cohort, "process_id": os.getpid(),
               "thread_id": threading.get_native_id(), "start_monotonic_ns": time.perf_counter_ns(),
               "owners_before": owners, "pending_pp_before": self._ws_pending(),
               "before": snapshot["snapshot"], "before_cursor": snapshot["cursor"],
               "before_snapshot_cursor_before": snapshot["snapshot_cursor_before"],
               "before_snapshot_cursor_after": snapshot["snapshot_cursor_after"], "native_epoch_id": None,
               "status": "RAW_STEP_STARTED", "complete_workspace": False}
        if self._ws_bridge.mlsys_torch_probe_begin(epoch, ("worker_step:" + str(epoch)).encode()) != 0:
            raise RuntimeError("Torch Workspace epoch begin failed")
        self._ws_torch_active = epoch
        self._ws_current = row
        # Warmup is retained in the complete SDK trace, while native startup
        # communicator scopes may be observed without nesting a model epoch.
        if self._ws_cohort == "formal":
            self._ws_begin_native(epoch, "formal_worker_step:" + str(epoch))
            row["native_epoch_id"] = epoch

    def _ws_finish(self, value: Any, *, error: BaseException | None = None) -> None:
        row = self._ws_current
        if row is None:
            return
        try:
            row["return_witness"] = cpu_return_witness(value, self.model_runner)
            row["pending_pp_after"] = self._ws_pending()
            row["owners_after"] = self._ws_census()
            snapshot = self._ws_take_snapshot()
            row["after"] = snapshot["snapshot"]
            row["after_cursor"] = snapshot["cursor"]
            row["after_snapshot_cursor_before"] = snapshot["snapshot_cursor_before"]
            row["after_snapshot_cursor_after"] = snapshot["snapshot_cursor_after"]
            if self._ws_hook.records:
                row["stage_record_index"] = len(self._ws_hook.records) - 1
                row["stage_record"] = deepcopy(self._ws_hook.records[-1])
                for field in ("iteration_id", "phase", "query_tokens", "context_lengths", "request_phases"):
                    if field in row["stage_record"]:
                        row[field] = deepcopy(row["stage_record"][field])
            row["status"] = "RAW_STEP_CAPTURED" if error is None else "RAW_STEP_FAILED"
            if error is not None:
                row["error"] = type(error).__name__ + ": " + str(error)
        except Exception as exc:
            row.update(status="RAW_OBSERVER_FAILED", observer_error=type(exc).__name__ + ": " + str(exc))
            self._ws_errors.append(row["observer_error"])
        finally:
            if self._ws_bridge.mlsys_torch_probe_end(row["epoch_id"]) != 0:
                self._ws_errors.append("Torch Workspace epoch end failed")
            self._ws_torch_active = None
            if row["native_epoch_id"] is not None:
                self._ws_end_native(row["native_epoch_id"])
            row["end_monotonic_ns"] = time.perf_counter_ns()
            filename = "epoch-" + str(row["epoch_id"]) + ".json.gz"
            write_gzip(self._ws_root / filename, row)
            self._ws_records.append({key: deepcopy(value) for key, value in row.items()
                                    if key not in {"before", "after", "owners_before", "owners_after"}})
            self._ws_records[-1]["epoch_file"] = str(self._ws_root / filename)
            self._ws_current = None

    def execute_model(self, scheduler_output: Any) -> Any:
        positive = scheduler_output.total_num_scheduled_tokens > 0
        if positive:
            self._ws_start()
        try:
            value = super().execute_model(scheduler_output)
        except BaseException as exc:
            if positive:
                self._ws_finish(None, error=exc)
            raise
        if positive and self._ws_placement["stage_index"] < self._ws_placement["pipeline_parallel_size"] - 1:
            self._ws_finish(value)
        return value

    def sample_tokens(self, grammar_output: Any) -> Any:
        try:
            value = super().sample_tokens(grammar_output)
        except BaseException as exc:
            self._ws_finish(None, error=exc)
            raise
        self._ws_finish(value)
        return value

    def mlsys_workspace_arm(self, cohort: str) -> dict[str, Any]:
        if (cohort not in {"warmup", "formal"} or self._ws_current is not None or self._ws_closed
                or self._ws_pending_close_ack is not None):
            raise RuntimeError("Cannot arm this Workspace cohort at the current boundary")
        if file_sha256(self._ws_plan_path) != self._ws_plan_sha:
            raise RuntimeError("Immutable Workspace acquisition plan changed")
        self._ws_cohort = cohort
        return {"cohort": cohort, "process_id": os.getpid(), "placement": self._ws_placement,
                "armed_monotonic_ns": time.perf_counter_ns(), "complete_workspace": False}

    def mlsys_workspace_export(self) -> dict[str, Any]:
        if self._ws_closed_payload is not None:
            return deepcopy(self._ws_closed_payload)
        if self._ws_deferred_payload is not None:
            return deepcopy(self._ws_deferred_payload)
        import torch
        import vllm
        from mlsys_sim.vllm_hooks import _loaded_rccl

        properties = torch.cuda.get_device_properties(self.device)
        loaded_rccl = _loaded_rccl()
        try:
            loaded_rccl["mapped_segments"] = mapped_library_segments(loaded_rccl["path"])
        except Exception as exc:
            loaded_rccl["mapped_segments"] = []
            loaded_rccl["mapped_segments_error"] = {"type": type(exc).__name__, "message": str(exc)}
            self._ws_errors.append("Actual RCCL executable mappings were not established: " + str(exc))
        hardware_inventory = observe_hardware_inventory(loaded_rccl)
        if hardware_inventory.get("status") == "ERROR":
            self._ws_errors.append("Observed hardware inventory probe failed: " +
                                   json.dumps(hardware_inventory["error"], sort_keys=True))
        vocabulary_modules = []
        vocabulary_fields = ("padding_size", "num_embeddings", "org_num_embeddings", "embedding_dim",
            "num_embeddings_padded", "org_vocab_size_padded", "added_vocab_size_padded",
            "num_embeddings_per_partition", "tp_size", "tp_rank", "vocab_start_index", "vocab_end_index")
        for name, module in self.model_runner.model.named_modules():
            if name.split(".")[-1] not in {"embed_tokens", "lm_head"}:
                continue
            actual = vars(module)
            fields = {key: actual[key] for key in vocabulary_fields if key in actual}
            if any(type(value) not in (int, float, str, bool, type(None)) for value in fields.values()):
                raise RuntimeError("Vocabulary module contains an unsupported non-scalar metadata field")
            row = {"module_name": name, "class_name": type(module).__module__ + "." + type(module).__name__,
                   "fields": fields, "constructor_source": source_binding(type(module).__init__)}
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor):
                row["weight"] = tensor_descriptor(weight, name + ".weight", cursor=self._ws_cursor)
            vocabulary_modules.append(row)
        return {"schema_version": 1, "kind": "vllm_worker_workspace_raw",
            "session_id": self._ws_plan["session_id"], "process_id": os.getpid(),
            **self._ws_placement, "placement": dict(self._ws_placement),
            "source_bindings": self._ws_sources, "plan_sha256": self._ws_plan_sha,
            "experiment": deepcopy(self._ws_hook.header["experiment"]),
            "identity": deepcopy(self._ws_hook.header["identity"]),
            "runtime_config": deepcopy(self._ws_hook.header["metadata"].get("runtime_execution_config")),
            "runtime_config_sha256": self._ws_hook.header["metadata"].get("runtime_config_sha256"),
            "model_binding": deepcopy(self._ws_hook.header["metadata"].get("model_binding")),
            "runtime_version": str(vllm.__version__), "torch_version": str(torch.__version__),
            "torch_git": str(torch.version.git_version), "hip_build_version": str(torch.version.hip),
            "allocator_backend": torch.cuda.memory.get_allocator_backend(),
            "observed_device": {"name": properties.name, "memory_bytes": int(properties.total_memory),
                "architecture": getattr(properties, "gcnArchName", None), "visibility": {
                    name: os.environ.get(name) for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                                                           "CUDA_VISIBLE_DEVICES")}},
            "observed_pp_group": self._ws_group_descriptor,
            "constructor": self._ws_lt, "communicator_constructors": self._ws_ctor,
            "transport_sources": self._ws_transport_sources, "transport_events": deepcopy(self._ws_events),
            "pg_initialization_scopes": self._ws_pg_scopes, "records": deepcopy(self._ws_records),
            "communication_prewarm": deepcopy(self._ws_pg_prewarm),
            "loaded_rccl": loaded_rccl, "observed_hardware_inventory": hardware_inventory,
            "stdthread_control": deepcopy(self._ws_stdthread_control),
            "history_max_entries": 100000,
            "observed_vocabulary_modules": vocabulary_modules,
            "startup_snapshot": str(self._ws_root / "startup.json.gz"),
            "stage_capture": self._ws_hook.export_capture(),
            "native_output_template": os.environ.get("MLSYS_NATIVE_PROBE_OUT"),
            "native_final_file": os.environ.get("MLSYS_NATIVE_PROBE_OUT", "").replace("%p", str(os.getpid())),
            "torch_probe_file": str(self._ws_root / "torch-native-records.json"),
            "capture_errors": list(self._ws_errors), "complete_workspace": False, "hardware_validated": False,
            "capture_closed": self._ws_closed,
            "native_finalization_policy": self._ws_native_finalization,
            "limitations": ["Raw worker observation requires independent owner/native/transport/root audits.",
                "Nonfinal Worker return is an asynchronous PP handoff, not a CPU-only release proof.",
                "Warmup native callbacks retain global startup scope; only formal rows have model native epochs."]}

    def mlsys_workspace_close(self) -> dict[str, Any]:
        import torch

        if self._ws_closed_summary is not None:
            return deepcopy(self._ws_closed_summary)
        if self._ws_pending_close_ack is not None:
            return deepcopy(self._ws_pending_close_ack)
        if self._ws_current is not None:
            raise RuntimeError("Workspace close requires one closed model/sampling boundary")
        if self._ws_bridge is None or threading.get_native_id() != self._ws_install_thread:
            raise RuntimeError("Workspace close must run on the actual inference thread")
        self._ws_cohort = "teardown"
        epoch = 9001
        self._ws_begin_native(epoch, "explicit_post_cohort_PP_teardown")
        if self._ws_bridge.mlsys_torch_probe_begin(epoch, b"explicit_post_cohort_PP_teardown") != 0:
            raise RuntimeError("Teardown Torch epoch begin failed")
        self._ws_torch_active = epoch
        before = {**self._ws_take_snapshot(), "pending_pp": self._ws_pending()}
        works = getattr(self, "_pp_send_work", [])
        for handle in works:
            handle.wait()
        if works:
            del handle
        self._pp_send_work: list[Any] = []
        works.clear()
        torch.cuda.synchronize(self.device)
        if self._ws_pp.world_size > 1:
            self._ws_pp._reap_completed_isends()
        # Explicit post-cohort cleanup only. It completes deferred allocator
        # releases so the full tape can audit transport generations; no formal
        # row's peak, original wait/reap order or endpoint is changed.
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        after = {**self._ws_take_snapshot(), "pending_pp": self._ws_pending()}
        if self._ws_bridge.mlsys_torch_probe_end(epoch) != 0:
            raise RuntimeError("Teardown Torch epoch end failed")
        self._ws_torch_active = None
        self._ws_end_native(epoch)
        write_gzip(self._ws_root / "teardown.json.gz", {"epoch_id": epoch, "before": before, "after": after,
            "policy": "after_complete_cohort_only_wait_retained_device_work_then_natural_reap_and_empty_cache"})
        if self._ws_bridge.mlsys_torch_probe_write(str(self._ws_root / "torch-native-records.json").encode()) != 0:
            raise RuntimeError("Torch observer output failed")
        self._ws_hook.flush()
        # Gather every potentially HIP-backed metadata read before closing the
        # SDK window. Only CPU serialization/RPC return follows finalization.
        raw = self.mlsys_workspace_export()
        if self._ws_native.mlsys_probe_write(str(self._ws_root / "native-checkpoint.json").encode()) != 0:
            raise RuntimeError("Native observer checkpoint output failed")
        raw["observer_reference_release"] = self._ws_release_observer_references()
        raw["capture_errors"] = list(self._ws_errors)
        raw["teardown_snapshot"] = str(self._ws_root / "teardown.json.gz")
        if self._ws_plan["mode"] != "single":
            finalization = getattr(self, "_ws_native_finalization", "worker_shutdown")
            session_name = ("session-process-exit-diagnostic.json"
                            if finalization == "process_exit" else "session.json")
            raw.update(capture_closed=False, native_finalize_deferred=True,
                       native_finalization_policy=finalization,
                       deferred_close_saved_monotonic_ns=time.perf_counter_ns(),
                       lifecycle_clock="CLOCK_MONOTONIC",
                       natural_shutdown_source=deepcopy(self._ws_shutdown_source))
            self._ws_deferred_payload = deepcopy(raw)
            write_new(self._ws_root / "session-pending.json", raw)
            ack = {"process_id": os.getpid(), "placement": dict(self._ws_placement),
                "session_file": str(self._ws_root / session_name),
                "pending_session_file": str(self._ws_root / "session-pending.json"),
                "process_exit_payload_file": str(self._ws_root / "session-process-exit-pending.json"),
                "native_finalization_policy": finalization,
                "record_count": len(self._ws_records), "capture_errors": list(self._ws_errors),
                "finalize_deferred": True, "capture_closed": False,
                "complete_workspace": False, "hardware_validated": False}
            if finalization == "process_exit":
                ack["diagnostic_session_file"] = ack["session_file"]
            self._ws_pending_close_ack = deepcopy(ack)
            return ack
        return self._ws_finalize_cpu_payload(raw)

    def _ws_finalize_cpu_payload(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Only native observer stop/finalize plus CPU output; never HIP reads."""
        if self._ws_closed_summary is not None:
            return deepcopy(self._ws_closed_summary)
        stop_status = int(self._ws_native.mlsys_probe_stop())
        finalize_status = None
        errors = list(raw.get("capture_errors", []))
        if stop_status == 0:
            try:
                self._ws_native.mlsys_probe_finalize.argtypes = []
                self._ws_native.mlsys_probe_finalize.restype = ctypes.c_int
                finalize_status = int(self._ws_native.mlsys_probe_finalize())
            except Exception as exc:
                errors.append("Official SDK finalizer exception: " + type(exc).__name__ + ": " + str(exc))
        if stop_status != 0:
            errors.append("Native SDK stop/drain failed with original code " + str(stop_status))
        if finalize_status != 0:
            errors.append("Official SDK finalizer did not succeed: " + str(finalize_status))
        original_returned = raw.get("runtime_shutdown", {}).get("original_returned", True)
        self._ws_closed = stop_status == 0 and finalize_status == 0 and original_returned
        raw.update(capture_closed=self._ws_closed, native_sdk_stop_status=stop_status,
                   native_sdk_finalize_status=finalize_status, capture_errors=errors)
        self._ws_closed_payload = deepcopy(raw)
        write_new(self._ws_root / "session.json", raw)
        summary = {"process_id": os.getpid(), "placement": dict(self._ws_placement),
                "session_file": str(self._ws_root / "session.json"),
                "record_count": len(self._ws_records), "capture_errors": errors,
                "capture_closed": self._ws_closed, "complete_workspace": False, "hardware_validated": False}
        self._ws_closed_summary = deepcopy(summary)
        return summary

    def _ws_after_natural_shutdown(self, receipt: dict[str, Any]) -> None:
        if self._ws_closed_summary is not None or self._ws_deferred_payload is None:
            return
        raw = deepcopy(self._ws_deferred_payload)
        raw["runtime_shutdown"] = deepcopy(receipt)
        if receipt.get("original_returned") is not True:
            raw["capture_errors"].append("Natural WorkerProc.shutdown did not return successfully")
        path = self._ws_root / "natural-shutdown-receipt.json"
        write_new(path, receipt)
        raw["natural_shutdown_receipt_file"] = str(path)
        raw["natural_shutdown_receipt_sha256"] = file_sha256(path)
        if getattr(self, "_ws_native_finalization", "worker_shutdown") == "process_exit":
            # Diagnostic only: leave the real native observer active through
            # normal process teardown. No manual stop or SDK finalize occurs.
            raw.update(capture_closed=False, native_sdk_stop_status=None,
                native_sdk_finalize_status=None, native_sdk_closure_pending=True,
                diagnostic_only=True, complete_workspace=False, hardware_validated=False,
                native_observation_policy="automatic_SDK_tool_fini_at_actual_process_exit")
            self._ws_closed_payload = deepcopy(raw)
            write_new(self._ws_root / "session-process-exit-pending.json", raw)
            self._ws_closed_summary = {"process_id": os.getpid(), "capture_closed": False,
                "diagnostic_only": True, "complete_workspace": False, "hardware_validated": False}
            return
        self._ws_finalize_cpu_payload(raw)

    def _ws_write_shutdown_observer_error(self, receipt: dict[str, Any], error: BaseException) -> None:
        write_new(self._ws_root / "shutdown-observer-error.json", {"runtime_shutdown": receipt,
            "observer_error": {"type": type(error).__name__, "message": str(error)},
            "capture_closed": False, "complete_workspace": False, "hardware_validated": False})
