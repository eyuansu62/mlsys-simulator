"""Replay actual PP Tensor storage generations across model-step boundaries.

This component proves a transport classification, never complete Workspace or
native API coverage. Inputs remain raw observations; a caller-supplied verdict
or an address without its allocation generation cannot exclude storage.
"""

from collections import defaultdict
import hashlib
import json
from typing import Any


_ACTIONS = ("alloc", "free_requested", "free_completed", "segment_alloc", "segment_free",
            "segment_map", "segment_unmap", "snapshot", "oom")
_OPERATIONS = {"isend_tensor_dict", "irecv_tensor_dict", "_reap_completed_isends"}
_NAMES = {"hidden_states", "residual"}
_LIMIT = 262144


class TransportInputError(ValueError):
    pass


def _int(value: Any, label: str, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise TransportInputError(label + ": missing, uncertain or invalid integer")
    return value


def _object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise TransportInputError(label + ": expected object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\0" in value:
        raise TransportInputError(label + ": expected bounded nonempty string")
    return value


def _hash(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise TransportInputError(label + ": expected actual lowercase source SHA256")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or _json(value) != _json(expected):
        raise TransportInputError(label + ": observed value differs")


def _pool(value: Any) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or any(type(x) is not int for x in value):
        raise TransportInputError("Native allocator pool is missing")
    if value != [0, 0]:
        raise TransportInputError("Private allocator pools are outside transport replay")
    return 0, 0


def _history_key(row: dict) -> tuple:
    name = row.get("action")
    if name not in _ACTIONS:
        raise TransportInputError("Unknown memory-history action")
    _same(row.get("compile_context") in ("N/A", ""), True, "history compilation context")
    return (name, _int(row.get("addr"), "history address"), _int(row.get("size"), "history bytes"),
            _int(row.get("stream"), "history stream"), _pool(row.get("pool_id")),
            row.get("user_metadata"))


def _native_key(row: dict) -> tuple:
    return (row["action_name"], row["address"], row["size_bytes"], row["stream"],
            _pool(row["mempool"]), row["user_metadata"])


def _blocks(snapshot: dict, device: int) -> dict[int, dict]:
    output = {}
    ranges = []
    segments = snapshot.get("segments")
    if not isinstance(segments, list):
        raise TransportInputError("Snapshot has no actual segments")
    for segment in segments:
        _same(segment.get("device"), device, "snapshot device")
        _pool(segment.get("segment_pool_id"))
        _same(segment.get("is_expandable"), False, "snapshot expandable mapping")
        start = _int(segment.get("address"), "segment address", 1)
        total = _int(segment.get("total_size"), "segment capacity", 1)
        cursor = start
        ranges.append((start, start + total))
        for block in segment["blocks"]:
            address = _int(block.get("address", cursor), "block address", 1)
            size = _int(block.get("size"), "block capacity", 1)
            _same(address, cursor, "contiguous block")
            cursor += size
            state = block.get("state")
            if state == "inactive":
                continue
            if state not in ("active_allocated", "active_awaiting_free", "active_pending_free"):
                raise TransportInputError("Snapshot block has unknown lifetime state")
            requested = _int(block.get("requested_size"), "requested block bytes", 1, size)
            if address in output:
                raise TransportInputError("Snapshot active storage aliases another allocator block")
            output[address] = {"bytes": requested, "state": state}
        _same(cursor, start + total, "snapshot segment coverage")
    if any(a[1] > b[0] for a, b in zip(sorted(ranges), sorted(ranges)[1:])):
        raise TransportInputError("Snapshot segments overlap")
    return output


def _tape(probe: dict, device: int, process_id: int) -> tuple[list[dict], dict[int, dict]]:
    expected = {"schema_version": 1, "kind": "torch_allocator_trace_probe", "pid": process_id,
        "complete_workspace": False, "allocation_owner_classification_performed": False,
        "checkpoint_stable": True, "checkpoint_is_prefix_only": True,
        "global_observation_issues_present": False, "callback_gpu_calls": False,
        "callback_gil": False, "callback_extra_locks": False, "abi_cxx11": 1,
        "clock": "CLOCK_MONOTONIC_RAW", "metadata_encoding": "byte_preserving_latin1",
        "owner_policy": "active_epoch_install_thread_token_only", "hook_detachable": False,
        "callback_scope": "all_allocator_tracker_callbacks_since_explicit_install"}
    for field, value in expected.items():
        _same(probe.get(field), value, "torch_probe." + field)
    for field in ("dropped_events", "unknown_actions", "truncated_metadata", "clock_failures",
                  "thread_id_failures", "api_errors", "boundary_callbacks", "callbacks_inflight_at_write_end"):
        _same(probe.get(field), 0, "torch_probe." + field)
    events = probe.get("events")
    if not isinstance(events, list) or len(events) > _LIMIT:
        raise TransportInputError("Complete bounded native Torch event tape is required")
    for field in ("retained_events", "attempted_events_at_cutoff", "attempted_events_at_write_end"):
        _same(probe.get(field), len(events), "torch_probe." + field)
    limit = _int(probe.get("event_limit"), "native event limit", 1, _LIMIT)
    if len(events) >= limit:
        raise TransportInputError("Native event limit was reached")
    install_thread = _int(probe.get("install_thread_id"), "installed inference thread", 1)
    epochs = {}
    for epoch in probe.get("epochs", []):
        identifier = _int(epoch.get("id"), "native epoch id", 1)
        if identifier in epochs:
            raise TransportInputError("Duplicate native epoch")
        epochs[identifier] = epoch
        _same(epoch.get("ended"), True, "closed native epoch")
        _same(epoch.get("owner_thread_id"), install_thread, "native epoch thread")
        for field in ("dropped_events", "unknown_actions", "unowned_allocations", "cross_thread_allocations",
                      "truncated_metadata", "boundary_overlap", "boundary_callbacks", "clock_failures",
                      "thread_id_failures", "oom_events", "mapping_events", "observation_issue_mask"):
            _same(epoch.get(field), 0, "native epoch." + field)
        _same(epoch.get("observation_issues_present"), False, "native epoch observations")
        first = _int(epoch.get("begin_sequence"), "epoch start cursor", maximum=len(events))
        last = _int(epoch.get("end_sequence"), "epoch end cursor", maximum=len(events))
        begin = _int(epoch.get("begin_ns"), "epoch begin clock", 1)
        end = _int(epoch.get("end_ns"), "epoch end clock", 1)
        if first > last or begin >= end:
            raise TransportInputError("Native epoch bounds differ")
    counts: dict[int, int] = defaultdict(int)
    outside_allocations = 0
    for index, row in enumerate(events, 1):
        _same(row.get("sequence"), index, "complete native sequence")
        action = _int(row.get("action"), "native action", maximum=len(_ACTIONS) - 1)
        _same(row.get("action_name"), _ACTIONS[action], "native action name")
        if action in (5, 6, 8):
            raise TransportInputError("Mapping or OOM occurred in the complete transport tape")
        if action != 7:
            _same(row.get("device"), device, "native device")
        else:
            _int(row.get("device"), "native global snapshot metadata device", maximum=1)
        for field in ("address", "size_bytes", "stream", "owner_token"):
            _int(row.get(field), "native " + field)
        _pool(row.get("mempool"))
        metadata = row.get("user_metadata")
        if not isinstance(metadata, str):
            raise TransportInputError("Native metadata bytes are absent")
        size = len(metadata.encode("latin1"))
        _same(row.get("user_metadata_size"), size, "native metadata bytes")
        _same(row.get("retained_metadata_size"), size, "complete native metadata")
        thread = _int(row.get("thread_id"), "native event thread", 1)
        timestamp = _int(row.get("timestamp_ns"), "native event timestamp", 1)
        identifier = _int(row.get("epoch_id"), "native event epoch")
        flags = _int(row.get("flags"), "native observation flags")
        matching = [ep for ep in epochs.values() if ep["begin_sequence"] < index <= ep["end_sequence"]]
        if len(matching) > 1 or (matching[0]["id"] if matching else 0) != identifier:
            raise TransportInputError("Native callback crosses an epoch boundary")
        if identifier:
            ep = epochs[identifier]
            counts[identifier] += 1
            if not ep["begin_ns"] <= timestamp <= ep["end_ns"]:
                raise TransportInputError("Native callback clock is outside its epoch")
            expected_flags = 0 if thread == install_thread else 2
            _same(flags, expected_flags, "native callback thread flags")
            _same(row["owner_token"], identifier if thread == install_thread else 0, "native owner token")
        else:
            _same(flags, 4, "native outside-epoch flags")
            _same(row["owner_token"], 0, "outside-epoch owner token")
            outside_allocations += int(action in (0, 3))
        if action in (0, 3) and thread != install_thread:
            raise TransportInputError("A background allocation has no supported semantic owner")
    for identifier, epoch in epochs.items():
        _same(epoch.get("observed_events"), counts[identifier], "native epoch event count")
    _same(probe.get("outside_epoch_allocations"), outside_allocations, "outside-epoch allocation count")
    return events, epochs


def _marker(snapshot: dict, before: Any, after: Any, events: list[dict], device: int) -> int:
    left = _int(before, "snapshot cursor before", maximum=len(events))
    right = _int(after, "snapshot cursor after", maximum=len(events))
    if left > right:
        raise TransportInputError("Snapshot cursor interval is reversed")
    allocated = sum(block["size"] for segment in snapshot.get("segments", []) if segment.get("device") == device
                    for block in segment["blocks"] if block["state"] != "inactive")
    # The installed allocator emits one callback per visible-device snapshot,
    # with device_ left at its default0. It records active rounded bytes and
    # appends the history marker after copying the returned trace. The observed
    # call interval and unique nonzero allocated-byte value identify this rank;
    # snapshot metadata is never used to infer another device's allocations.
    matches = [event for event in events[left:right] if event["action_name"] == "snapshot"
               and event["size_bytes"] == allocated and allocated > 0]
    if len(matches) != 1:
        raise TransportInputError("Snapshot needs exactly one actual native snapshot marker")
    traces = snapshot.get("device_traces")
    if not isinstance(traces, list) or len(traces) <= device or not traces[device]:
        raise TransportInputError("Snapshot needs full device memory history")
    return matches[0]["sequence"]


def audit_workspace_transport(*, torch_probe: dict, transport_context: dict, placement: dict) -> dict:
    """Derive generation-bound transport exclusions from complete raw evidence.

    Context contains the worker session, loaded startup/teardown envelopes, and
    every loaded model-epoch record, including warmup. No filesystem is read.
    """
    result: dict[str, Any] = {"schema_version": 1, "kind": "torch_pp_transport_generation_audit",
        "status": "PARTIAL", "complete_workspace": False, "hardware_validated": False,
        "issues": [], "transport_generations": [], "epochs": {},
        "remaining_obligations": ["Independently bound actual source/build/process/rank evidence",
            "Unchanged fixed weight/KV/Runtime owners and complete non-Torch coverage",
            "PP fixed-target plus observed transport-peak budget admission"]}
    try:
        _audit(result, torch_probe, transport_context, placement)
    except (TransportInputError, ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError) as exc:
        result["issues"].append({"code": "INVALID_OR_INCOMPLETE_TRANSPORT_EVIDENCE", "message": str(exc)})
    return result


def _audit(result: dict, probe: dict, context: dict, placement: dict) -> None:
    # Local import avoids a module-import cycle; this validator is also called
    # by the rank-local Torch audit, which replays these raw inputs itself.
    from workspace_torch_audit import _placement
    from mlsys_sim.config import experiment_from_dict
    from mlsys_sim.pp_transport import validate_native_pp_descriptor

    device = _int(placement.get("device_index"), "placement device")
    layout = _placement(placement, device)
    if layout["tensor_parallel_size"] != 1 or layout["pipeline_parallel_size"] != 2:
        raise TransportInputError("Only actual two-device TP1/PP2 transport is supported")
    result["raw_evidence_sha256"] = hashlib.sha256(_json((probe, context, layout)).encode()).hexdigest()
    session = _object(context.get("session"), "worker session")
    _same(session.get("schema_version"), 1, "worker schema")
    _same(session.get("kind"), "vllm_worker_workspace_raw", "worker kind")
    process_id = _int(session.get("process_id"), "worker process", 1)
    _text(session.get("session_id"), "actual session")
    _same(session.get("capture_closed"), True, "worker closed capture")
    _same(session.get("capture_errors"), [], "worker capture errors")
    for field in ("tensor_parallel_size", "pipeline_parallel_size", "stage_index", "tp_rank"):
        _same(session.get(field), layout[field], "worker " + field)
        _same(session.get("placement", {}).get(field), layout[field], "worker placement " + field)
    _same(session.get("logical_device_index"), device, "logical device")
    _same(session.get("placement", {}).get("logical_device_index"), device, "placement logical device")
    _same(session.get("placement", {}).get("process_id"), process_id, "placement process")
    _same(session.get("global_rank"), layout["stage_index"], "global rank")
    _same(session.get("placement", {}).get("global_rank"), layout["stage_index"], "placement global rank")
    _same(session.get("world_size"), 2, "actual world size")
    _same(session.get("placement", {}).get("world_size"), 2, "placement world size")
    bindings = _object(session.get("source_bindings"), "actual worker source bindings")
    _hash(bindings.get("worker_source_sha256"), "worker source")
    _text(bindings.get("worker_source_reference"), "worker source file")
    base = _object(bindings.get("base_worker"), "base Worker method source")
    _same(base.get("qualified_name"), "vllm.v1.worker.gpu_worker.Worker.execute_model", "base Worker entrypoint")
    _hash(base.get("source_sha256"), "base Worker source")
    _hash(base.get("file_sha256"), "base Worker file")
    _text(base.get("source_reference"), "base Worker source file")
    sources = _object(session.get("transport_sources"), "actual transport method sources")
    if set(sources) != _OPERATIONS:
        raise TransportInputError("Actual transport methods are not all bound")
    for name, source in sources.items():
        _same(source.get("qualified_name"), "vllm.distributed.parallel_state.GroupCoordinator." + name,
              "actual PP method")
        _hash(source.get("source_sha256"), "actual PP method hash")
        _hash(source.get("file_sha256"), "actual PP file hash")
        _text(source.get("source_reference"), "actual PP source file")
    group = _object(session.get("observed_pp_group"), "actual PP group")
    _int(group.get("object_id"), "PP group object", 1)
    _same(group.get("class_name"), "vllm.distributed.parallel_state.GroupCoordinator", "PP group class")
    _same(group.get("ranks"), [0, 1], "PP group ranks")
    _same(group.get("rank_in_group"), layout["stage_index"], "PP group local rank")
    _same(group.get("world_size"), 2, "PP group size")
    stage_capture = _object(session.get("stage_capture"), "actual stage capture")
    experiment = experiment_from_dict(_object(stage_capture.get("experiment"), "actual stage experiment"))
    _same(experiment.strategy.pipeline_transport, "vllm_native", "actual stage transport strategy")
    _same(experiment.strategy.pipeline_parallel_size, 2, "actual stage experiment PP")
    _same(experiment.strategy.tensor_parallel_size, 1, "actual stage experiment TP")
    _same(stage_capture.get("session_id"), session["session_id"], "stage session")
    for field in ("tensor_parallel_size", "pipeline_parallel_size", "stage_index", "tp_rank"):
        _same(stage_capture.get(field), layout[field], "stage " + field)
    native_binding = stage_capture.get("metadata", {}).get("native_pp_backend_binding", {})
    _same(native_binding.get("backend"), "nccl", "actual native PP backend")
    _same(native_binding.get("use_cpu_custom_send_recv"), False, "actual native PP custom path")
    if layout["stage_index"] == 1:
        persistent = _object(stage_capture.get("metadata", {}).get("native_pp_persistent_binding"),
                             "actual loaded PP persistent targets")
        _same(persistent.get("max_num_tokens"), 512, "loaded PP persistent token capacity")
        _same(persistent.get("scope"), "loaded_runner_intermediate_buffers", "loaded PP target scope")
        descriptor = validate_native_pp_descriptor(experiment, persistent.get("descriptor"))
        _same(descriptor["padded_tokens"], 512, "loaded persistent PP descriptor capacity")
    events, native_epochs = _tape(probe, device, process_id)
    startup = _object(context.get("startup"), "startup snapshot envelope")
    teardown = _object(context.get("teardown"), "teardown envelope")
    _same(startup.get("process_id"), process_id, "startup process")
    _same(startup.get("thread_id"), probe["install_thread_id"], "startup actual inference thread")
    start_mark = _marker(startup["snapshot"], startup.get("snapshot_cursor_before"),
                         startup.get("snapshot_cursor_after"), events, device)
    _same(startup.get("cursor"), startup.get("snapshot_cursor_after"), "startup cursor")
    _same(startup.get("pending_pp"), [], "startup pending PP needs observed generations")
    finish = _object(teardown.get("after"), "final teardown snapshot")
    finish_mark = _marker(finish["snapshot"], finish.get("snapshot_cursor_before"),
                          finish.get("snapshot_cursor_after"), events, device)
    _same(finish.get("cursor"), finish.get("snapshot_cursor_after"), "teardown cursor")
    _same(finish.get("pending_pp"), [], "final pending PP")
    if finish_mark <= start_mark:
        raise TransportInputError("Full transport acquisition window is empty or reversed")
    bound = _int(session.get("history_max_entries"), "history limit", 1, 1000000)
    old_trace = startup["snapshot"]["device_traces"][device]
    final_trace = finish["snapshot"]["device_traces"][device]
    if len(old_trace) >= bound or len(final_trace) >= bound or len(final_trace) < len(old_trace):
        raise TransportInputError("Memory history wrapped, shrank or reached its bound")
    if [_history_key(x) for x in old_trace] != [_history_key(x) for x in final_trace[:len(old_trace)]]:
        raise TransportInputError("Initial memory history is not an unchanged final prefix")
    delta = [_history_key(x) for x in final_trace[len(old_trace):] if x["action"] != "snapshot"]
    native_delta = [_native_key(x) for x in events[start_mark:finish_mark] if x["action_name"] != "snapshot"]
    if delta != native_delta:
        raise TransportInputError("Full-session memory history differs from the native event tape")
    frames = context.get("epochs")
    if not isinstance(frames, list) or not frames or len(frames) > 63:
        raise TransportInputError("All bounded warmup/formal model frames are required")
    indexed = {}
    snapshots = {}
    source_slots = set()
    raw_records = session.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(frames):
        raise TransportInputError("Loaded epoch records do not cover the worker index")
    raw_index = {row["epoch_id"]: row for row in raw_records}
    for frame in frames:
        identifier = _int(frame.get("epoch_id"), "frame epoch", 1)
        iteration = _int(frame.get("capture_iteration"), "frame iteration", 1)
        if iteration in indexed or identifier not in native_epochs or identifier not in raw_index:
            raise TransportInputError("Frame/rank/native epoch identity is missing or duplicated")
        indexed[iteration] = frame
        _same(frame.get("process_id"), process_id, "frame process")
        _same(frame.get("thread_id"), probe["install_thread_id"], "frame actual inference thread")
        _same(frame.get("status"), "RAW_STEP_CAPTURED", "actual frame completion")
        for field, value in raw_index[identifier].items():
            if field != "epoch_file":
                _same(frame.get(field), value, "worker index " + field)
        slot = _int(frame.get("stage_record_index"), "source stage record index")
        records = stage_capture.get("records", [])
        if slot in source_slots or slot >= len(records) or frame.get("stage_record") != records[slot]:
            raise TransportInputError("Model frame does not bind the actual stage record")
        source_slots.add(slot)
        for field in ("iteration_id", "phase", "query_tokens", "context_lengths"):
            _same(frame.get(field), records[slot].get(field), "actual stage/frame " + field)
        _same(frame.get("iteration_id"), slot, "source stage iteration index")
        _text(frame.get("cohort"), "actual frame cohort")
        left = _marker(frame["before"], frame.get("before_snapshot_cursor_before"),
                       frame.get("before_snapshot_cursor_after"), events, device)
        right = _marker(frame["after"], frame.get("after_snapshot_cursor_before"),
                        frame.get("after_snapshot_cursor_after"), events, device)
        if not start_mark <= left <= right <= finish_mark:
            raise TransportInputError("Frame snapshots escape the complete acquisition window")
        _same(frame.get("before_cursor"), frame.get("before_snapshot_cursor_after"), "frame before cursor")
        _same(frame.get("after_cursor"), frame.get("after_snapshot_cursor_after"), "frame after cursor")
        epoch = native_epochs[identifier]
        if left > epoch["begin_sequence"] or epoch["end_sequence"] < right:
            raise TransportInputError("Frame snapshots disagree with the native model epoch")
        snapshots[identifier] = {"before_sequence": left, "after_sequence": right,
                               "before": _blocks(frame["before"], device), "after": _blocks(frame["after"], device)}
    _replay(result, events, native_epochs, indexed, snapshots, session, layout, experiment,
            _blocks(startup["snapshot"], device), start_mark, finish_mark)


def _replay(result: dict, events: list[dict], native_epochs: dict, frames: dict, snapshots: dict,
            session: dict, layout: dict, experiment: Any, baseline: dict, start_mark: int, finish_mark: int) -> None:
    from mlsys_sim.pp_transport import validate_native_pp_descriptor
    device = layout["device_index"]
    generations: list[dict] = []
    by_address: dict[int, list[dict]] = defaultdict(list)
    live: dict[int, dict] = {}
    sequence_generation = {}

    def allocate(address: int, size: int, sequence: int, stream: int, epoch: int, baseline_state=None):
        if address in live or any(address < p + v["bytes"] and p < address + size for p, v in live.items()):
            raise TransportInputError("Allocator generation overlaps live requested storage")
        current = {"generation_id": f"d{device}-a{address:x}-s{sequence}", "device": device,
            "address": address, "bytes": size, "alloc_sequence": sequence, "allocation_epoch_id": epoch,
            "stream": stream, "free_requested_sequence": None, "free_completed_sequence": None,
            "registrations": [], "event_sequences": [], "baseline": baseline_state is not None}
        if baseline_state in ("active_awaiting_free", "active_pending_free"):
            raise TransportInputError("Initial pending storage has no observed free-request generation")
        live[address] = current
        generations.append(current)
        by_address[address].append(current)
        return current

    for event in events:
        sequence, action = event["sequence"], event["action_name"]
        address, size, stream = event["address"], event["size_bytes"], event["stream"]
        if action == "alloc":
            if not address or not size:
                raise TransportInputError("Allocation requires actual positive storage")
            current = allocate(address, size, sequence, stream, event["epoch_id"])
        elif action in ("free_requested", "free_completed"):
            current = live.get(address)
            if current is None or current["bytes"] != size or (not current["baseline"] and current["stream"] != stream):
                raise TransportInputError("Free has no exact live generation/size/stream")
            if action == "free_requested":
                if current["free_requested_sequence"] is not None:
                    raise TransportInputError("Duplicate free request")
                current["free_requested_sequence"] = sequence
            else:
                if current["free_requested_sequence"] is None:
                    raise TransportInputError("Completed free lacks a real free request")
                current["free_completed_sequence"] = sequence
                del live[address]
        else:
            current = None
        if current is not None:
            current["event_sequences"].append(sequence)
            sequence_generation[sequence] = current
        if sequence == start_mark:
            for pointer, block in baseline.items():
                if pointer in live:
                    _same(live[pointer]["bytes"], block["bytes"], "initial native/snapshot storage")
                else:
                    allocate(pointer, block["bytes"], 0, 0, 0, block["state"])

    def at(cursor: int, address: int, size: int) -> dict:
        values = [g for g in by_address[address] if g["alloc_sequence"] <= cursor
                  and (g["free_completed_sequence"] is None or cursor < g["free_completed_sequence"])]
        if len(values) != 1 or values[0]["bytes"] != size:
            raise TransportInputError("Tensor metadata has no unique exact live allocation generation")
        return values[0]

    def descriptor(value: dict, *, flattened: bool = False) -> dict:
        pointer = _int(value.get("pointer"), "transport storage pointer", 1)
        size = _int(value.get("bytes"), "transport storage bytes", 1)
        _same(value.get("device_index"), device, "transport Tensor device")
        _same(value.get("dtype"), "torch.bfloat16", "transport Tensor dtype")
        _same(value.get("element_size"), 2, "transport Tensor element width")
        _same(value.get("tensor_offset"), 0, "transport full-storage offset")
        _same(value.get("tensor_data_ptr"), pointer, "transport Tensor/data storage binding")
        shape = value.get("shape")
        if not isinstance(shape, list):
            raise TransportInputError("Transport Tensor shape is absent")
        flat = (flattened and len(shape) == 1 and type(shape[0]) is int
                and 2048 <= shape[0] <= 512 * 2048 and shape[0] % 2048 == 0)
        matrix = (len(shape) == 2 and type(shape[0]) is int
                  and 1 <= shape[0] <= 512 and type(shape[1]) is int and shape[1] == 2048)
        if not (flat or matrix):
            raise TransportInputError("Transport Tensor has unsupported physical shape")
        _same(value.get("stride"), [1] if flat else [2048, 1], "contiguous full PP Tensor")
        _same(size, shape[0] * 2 if flat else shape[0] * 2048 * 2, "full PP Tensor capacity")
        if value.get("tensor_name") not in _NAMES:
            raise TransportInputError("Actual PP Tensor dictionary name is missing")
        before = _int(value.get("metadata_cursor_before"), "Tensor metadata start cursor", maximum=len(events))
        after = _int(value.get("metadata_cursor_after"), "Tensor metadata end cursor", maximum=len(events))
        _same(value.get("metadata_cursor"), after, "Tensor metadata cursor")
        if before > after:
            raise TransportInputError("Tensor metadata cursor interval is reversed")
        first, last = at(before, pointer, size), at(after, pointer, size)
        if first is not last or first["baseline"]:
            raise TransportInputError("Tensor metadata crosses a reused address or lacks observed birth")
        if first["free_requested_sequence"] is not None and first["free_requested_sequence"] <= after:
            raise TransportInputError("Tensor metadata refers to storage whose release was already requested")
        return first

    records = session.get("transport_events")
    if not isinstance(records, list) or len(records) > 4096:
        raise TransportInputError("Bounded actual PP calls are required")
    seen_ids = set()
    registrations: dict[int, list[dict]] = defaultdict(list)
    for position, event in enumerate(records):
        identifier = _int(event.get("event_id"), "unique transport event id", 1)
        if identifier in seen_ids:
            raise TransportInputError("Duplicate transport call identity")
        seen_ids.add(identifier)
        operation = event.get("operation")
        if operation not in _OPERATIONS:
            raise TransportInputError("Unknown transport operation")
        _same(event.get("status"), "RETURNED", "original transport return")
        if "metadata_error_before" in event or "metadata_error_after" in event:
            raise TransportInputError("Transport metadata acquisition was incomplete")
        _same(event.get("group"), session["observed_pp_group"], "actual PP group object/rank")
        _same(event.get("source_method_sha256"), session["transport_sources"][operation]["source_sha256"],
              "actual transport source")
        before = _int(event.get("before_cursor"), "transport call start cursor", maximum=len(events))
        after = _int(event.get("after_cursor"), "transport call end cursor", maximum=len(events))
        if before > after:
            raise TransportInputError("Transport operation cursor interval is reversed")
        observer = _int(event.get("observer_epoch_id"), "transport native observer epoch", 1)
        if observer not in native_epochs:
            raise TransportInputError("Transport call has no real enclosing Torch epoch")
        epoch = native_epochs[observer]
        if before < epoch["begin_sequence"] or after > epoch["end_sequence"]:
            raise TransportInputError("Transport call escaped its real model/teardown epoch")
        if operation == "_reap_completed_isends":
            _same(event.get("tensors"), [], "natural reap has no invented Tensor return")
            if observer == 9001:
                _same(event.get("cohort"), "teardown", "explicit teardown reap")
                _same(event.get("epoch_id"), None, "teardown is not a model frame")
            else:
                iteration = _int(event.get("capture_iteration"), "natural reap model iteration", 1)
                frame = frames.get(iteration)
                if (layout["stage_index"] != 0 or frame is None or observer != frame["epoch_id"]
                        or event.get("epoch_id") != observer or event.get("cohort") != frame["cohort"]):
                    raise TransportInputError("Natural reap is not the actual send frame or explicit teardown")
            continue
        iteration = _int(event.get("capture_iteration"), "transport model iteration", 1)
        frame = frames.get(iteration)
        if frame is None or frame["epoch_id"] != observer or event.get("epoch_id") != observer:
            raise TransportInputError("Transport call does not bind an actual model frame")
        _same(event.get("cohort"), frame["cohort"], "transport model cohort")
        window = snapshots[observer]
        if before < window["before_sequence"] or after > window["after_sequence"]:
            raise TransportInputError("Actual PP call escapes its raw frame snapshots")
        expected = "isend_tensor_dict" if layout["stage_index"] == 0 else "irecv_tensor_dict"
        _same(operation, expected, "PP direction on this stage")
        handles = event.get("handles")
        if (not isinstance(handles, list) or len(handles) != (3 if operation == "isend_tensor_dict" else 2)
                or len(set(handles)) != len(handles) or any(type(x) is not int or x <= 0 for x in handles)):
            raise TransportInputError("Actual asynchronous PP handles are not completely observed")
        tensors = event.get("tensors")
        if not isinstance(tensors, list) or len(tensors) != 2 or {x.get("tensor_name") for x in tensors} != _NAMES:
            raise TransportInputError("Actual PP call needs both exact named tensors")
        direction = "outgoing" if operation == "isend_tensor_dict" else "incoming"
        native = frame["stage_record"].get("transport_metadata", {}).get(direction, {}).get("native_descriptor")
        native = validate_native_pp_descriptor(experiment, _object(native, "actual stage PP physical descriptor"))
        physical = _int(native["padded_tokens"], "actual PP physical tokens", 1, 512)
        native_tensors = {x["name"]: x for x in native["tensors"]}
        for tensor in tensors:
            generation = descriptor(tensor, flattened=operation == "irecv_tensor_dict")
            if operation == "isend_tensor_dict" and tensor["metadata_cursor_after"] > before:
                raise TransportInputError("Send argument metadata was not observed before the original call")
            if operation == "irecv_tensor_dict" and tensor["metadata_cursor_before"] < after:
                raise TransportInputError("Receive result metadata predates the original return")
            wanted_shape = [physical * 2048] if len(tensor["shape"]) == 1 else [physical, 2048]
            _same(tensor["shape"], wanted_shape, "PP Tensor/actual forward physical shape")
            declared = native_tensors[tensor["tensor_name"]]
            for field, value in (("shape", [physical, 2048]), ("dtype", "bfloat16"), ("element_size", 2),
                                 ("full_bytes", tensor["bytes"]), ("tp_sliced", False),
                                 ("receiver_allgather", False)):
                _same(declared.get(field), value, "physical PP descriptor " + field)
            if generation["allocation_epoch_id"] != observer:
                raise TransportInputError("PP storage was not born in this actual forward/receive epoch")
            if operation == "irecv_tensor_dict" and not before < generation["alloc_sequence"] <= after:
                raise TransportInputError("Receive storage was not allocated by the actual receive call")
            if operation == "isend_tensor_dict" and generation["alloc_sequence"] > before:
                raise TransportInputError("Send input storage was not live before the original send")
            if at(after, generation["address"], generation["bytes"]) is not generation:
                raise TransportInputError("Transport return has a different storage generation")
            if operation == "isend_tensor_dict" and at(before, generation["address"], generation["bytes"]) is not generation:
                raise TransportInputError("Send storage was not live at the actual original call")
            if any(reg["event_id"] != identifier for reg in generation["registrations"]):
                raise TransportInputError("A transport generation was reused by a different PP call")
            registration = {"event_id": identifier, "operation": operation, "return_position": position,
                "capture_iteration": iteration, "epoch_id": observer, "tensor_name": tensor["tensor_name"],
                "after_cursor": after, "source_method_sha256": event["source_method_sha256"]}
            generation["registrations"].append(registration)
        registrations[iteration].append(event)
    if any(len(registrations[iteration]) != 1 for iteration in frames):
        raise TransportInputError("Every actual warmup/formal frame needs its complete PP call")
    proven = [g for g in generations if g["registrations"]]
    if not proven:
        raise TransportInputError("No generation-bound PP storage was observed")
    for generation in proven:
        if generation["free_requested_sequence"] is None or generation["free_completed_sequence"] is None:
            raise TransportInputError("PP generation remains live or pending at final teardown")
        if generation["free_completed_sequence"] > finish_mark:
            raise TransportInputError("PP generation was released beyond the final snapshot")

    def pending(value: Any, position: int, *, after: bool, current: dict) -> tuple:
        if not isinstance(value, list) or len(value) > 128:
            raise TransportInputError("Actual bounded self-retained PP queue is missing")
        output = []
        for index, entry in enumerate(value):
            _same(entry.get("entry_index"), index, "actual PP FIFO position")
            handles = entry.get("handle_ids")
            tensors = entry.get("tensor_descriptors")
            if (not isinstance(handles, list) or len(handles) != 3 or len(set(handles)) != 3
                    or any(type(x) is not int or x <= 0 for x in handles)
                    or not isinstance(tensors, list) or len(tensors) != 2
                    or {x.get("tensor_name") for x in tensors} != _NAMES):
                raise TransportInputError("PP FIFO entry lacks exact handles/tensor names")
            identities = []
            for tensor in tensors:
                generation = descriptor(tensor, flattened=True)
                if after and tensor["metadata_cursor_before"] < current["after_cursor"]:
                    raise TransportInputError("Retained Tensor metadata predates the observed queue return")
                if not after and tensor["metadata_cursor_after"] > current["before_cursor"]:
                    raise TransportInputError("Retained Tensor metadata follows the observed queue entry")
                matching = [reg for reg in generation["registrations"] if reg["tensor_name"] == tensor["tensor_name"]]
                if len(matching) != 1:
                    raise TransportInputError("PP FIFO storage has no real send registration")
                reg = matching[0]
                if (reg["operation"] != "isend_tensor_dict"
                        or reg["return_position"] > position
                        or (reg["return_position"] == position and not (after and current["operation"] == "isend_tensor_dict"))
                        or reg["after_cursor"] > tensor["metadata_cursor_after"]):
                    raise TransportInputError("PP FIFO borrows a future/receive registration")
                _same(reg["tensor_name"], tensor["tensor_name"], "retained PP tensor name")
                origin = records[reg["return_position"]]
                _same(origin["handles"], handles, "retained actual send handles")
                identities.append(generation["generation_id"])
            output.append((tuple(handles), tuple(identities)))
        return tuple(output)

    for position, event in enumerate(records):
        pending_before = pending(event.get("before_pending"), position, after=False, current=event)
        pending_after = pending(event.get("after_pending"), position, after=True, current=event)
        operation = event["operation"]
        if operation == "irecv_tensor_dict":
            if pending_before or pending_after:
                raise TransportInputError("Last PP rank unexpectedly retains send buffers")
        elif operation == "_reap_completed_isends":
            if len(pending_after) > len(pending_before) or (pending_after and pending_before[-len(pending_after):] != pending_after):
                raise TransportInputError("Natural reap changed/appended/reordered the actual FIFO")
        else:
            if not pending_after or len(pending_after) > len(pending_before) + 1:
                raise TransportInputError("Original send did not append exactly one retained entry")
            retained = pending_after[:-1]
            if retained and pending_before[-len(retained):] != retained:
                raise TransportInputError("Send changed older FIFO entries rather than naturally reaping a prefix")
            _same(list(pending_after[-1][0]), event["handles"], "new actual send queue entry")
    cpp_sequences = {sequence for generation in proven for sequence in generation["event_sequences"]}
    if any(event["flags"] == 2 and event["sequence"] not in cpp_sequences for event in events):
        raise TransportInputError("A background free has no proved PP generation")
    cross_thread_free = [event["sequence"] for event in events if event["sequence"] in cpp_sequences
                         and event["action_name"] in ("free_requested", "free_completed") and event["flags"] == 2]
    for identifier, bundle in snapshots.items():
        left, right = bundle["before_sequence"], bundle["after_sequence"]
        selected = {}
        for side, cursor in (("before", left), ("after", right)):
            rows = []
            for generation in proven:
                if generation["alloc_sequence"] <= cursor < generation["free_completed_sequence"]:
                    block = bundle[side].get(generation["address"])
                    if block is None or block["bytes"] != generation["bytes"]:
                        raise TransportInputError("Live PP generation is absent from its actual snapshot")
                    requested = generation["free_requested_sequence"] <= cursor
                    if requested:
                        if block["state"] not in ("active_awaiting_free", "active_pending_free"):
                            raise TransportInputError("PP requested/completed free snapshot state differs")
                    else:
                        _same(block["state"], "active_allocated", "PP active snapshot state")
                    rows.append({"generation_id": generation["generation_id"], "address": generation["address"],
                                 "bytes": generation["bytes"], "state": block["state"]})
            selected[side] = rows
        live_bytes = sum(row["bytes"] for row in selected["before"])
        peak = live_bytes
        for event in events[left:right]:
            if event["sequence"] in cpp_sequences:
                if event["action_name"] == "alloc":
                    live_bytes += event["size_bytes"]
                elif event["action_name"] == "free_completed":
                    live_bytes -= event["size_bytes"]
                peak = max(peak, live_bytes)
        _same(live_bytes, sum(row["bytes"] for row in selected["after"]), "PP endpoint live-byte replay")
        result["epochs"][str(identifier)] = {"before_sequence": left, "after_sequence": right,
            "before": selected["before"], "after": selected["after"], "peak_pp_requested_bytes": peak,
            "excluded_native_sequences": sorted(sequence for sequence in cpp_sequences if left < sequence <= right),
            "allowed_cross_thread_free_sequences": [sequence for sequence in cross_thread_free if left < sequence <= right]}
    result.update(status="TRANSPORT_GENERATIONS_PROVEN", placement=layout, process_id=session["process_id"],
        session_id=session["session_id"], transport_generations=proven,
        excluded_native_sequences=sorted(cpp_sequences), allowed_cross_thread_free_sequences=cross_thread_free,
        global_transport_peak_requested_bytes=max(row["peak_pp_requested_bytes"] for row in result["epochs"].values()),
        history_cross_check={"initial_snapshot_sequence": start_mark, "final_snapshot_sequence": finish_mark,
                             "full_native_events": len(events)})
