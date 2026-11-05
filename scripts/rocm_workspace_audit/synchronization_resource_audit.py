"""Restricted, generation-bound HIP event/HSA signal resource admission.

This proves control-resource lifetimes, never zero GPU memory. Every underlying
allocation, pool, mapping, private scratch and native GPU backing still requires
the independent native memory audit and its separate fixed-budget proof.
"""

from collections import defaultdict
import hashlib
import json
from typing import Any


_CREATE = {"hipEventCreateWithFlags", "hipEventCreate"}
_RECORD = {"hipEventRecord", "hipEventRecord_spt", "hipEventRecordWithFlags"}
_WAIT = {"hipStreamWaitEvent", "hipStreamWaitEvent_spt"}
_EVENT = _CREATE | _RECORD | _WAIT | {"hipEventDestroy", "hipEventSynchronize", "hipEventQuery"}
_DEVICE_QUERY = {"hipGetDevice", "hipCtxGetDevice"}
_DEVICE_SET = {"hipSetDevice"}
_INVALIDATE = {"hipCtxSetCurrent", "hipCtxPushCurrent", "hipCtxPopCurrent", "hipCtxCreate", "hipCtxDestroy"}
_FENCE = {"hipDeviceSynchronize", "hipStreamSynchronize"}
_SIGNAL = {"hsa_signal_create", "hsa_signal_destroy"}


class ResourceEvidenceError(ValueError):
    pass


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**64 - 1:
        raise ResourceEvidenceError(name + ": missing/invalid actual integer")
    return value


def _same(value: Any, expected: Any, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ResourceEvidenceError(name + ": observed value differs")


def _success(call: dict, status_type: int = 1) -> None:
    _same(call["exit"].get("status_type"), status_type, call["operation"] + " typed status")
    _same(call["exit"].get("status"), 0, call["operation"] + " success")


def _payload(call: dict, name: str = "raw_hip_control_payload", phase: str = "exit") -> dict:
    value = call[phase].get(name)
    if not isinstance(value, dict):
        raise ResourceEvidenceError(call["operation"] + ": actual " + name + " is missing")
    return value


def _proof(call: dict, generation: str, fence: dict | None = None) -> dict:
    a, b = call["enter"], call["exit"]
    return {"operation": call["operation"], "domain": a["domain"], "cid": a["cid"],
        "tid": a["tid"], "epoch": a["epoch"], "enter_sequence": a["seq"], "exit_sequence": b["seq"],
        "resource_generation_id": generation,
        "completion_fence_cid": None if fence is None else fence["enter"]["cid"]}


def _calls(data: dict) -> tuple[list[dict], dict[int, dict]]:
    for key, value in {"schema_version": 1, "kind": "native_allocation_api_diagnostic",
        "configured": True, "init_called": True, "ready": True, "bounds_valid": True,
        "write_complete": True, "finalize_callback_called": True, "metadata_truncated": False,
        "filter_policy": "all_APIs_global"}.items():
        _same(data.get(key), value, "native report." + key)
    for key in ("dropped_events", "recursive_suppressed", "unknown_events", "missing_user_data",
                "filtered_nonmutator_events", "callbacks_inflight"):
        _same(data.get(key), 0, "native report." + key)
    stop = data.get("stop_evidence", {})
    for key, value in {"invoked": True, "completed": True, "observation_window_closed": True,
        "stop_result": 0, "drain_status": 0, "callbacks_after_stop": 0, "actual_sdk_stop_calls": 1}.items():
        _same(stop.get(key), value, "actual SDK stop." + key)
    events = data.get("events")
    if not isinstance(events, list) or len(events) > 4194304:
        raise ResourceEvidenceError("Full bounded raw native event tape is missing")
    for key in ("retained_events", "attempted_events", "observed_callbacks"):
        _same(data.get(key), len(events), "native report." + key)
    if len(events) >= _int(data.get("max_events"), "native event bound", 1):
        raise ResourceEvidenceError("Native observation bound was reached")
    operations = {(i, op["id"]): op["name"] for i, domain in enumerate(data.get("domains", []))
                  for op in domain["operations"]}
    wanted = _EVENT | _DEVICE_QUERY | _DEVICE_SET | _INVALIDATE | _FENCE | _SIGNAL
    pending = {}
    calls = []
    by_cid = {}
    for index, event in enumerate(events, 1):
        _same(event.get("seq"), index, "complete raw callback sequence")
        operation = operations.get((event.get("domain"), event.get("op")))
        if operation is None:
            raise ResourceEvidenceError("Callback operation is absent from the actual SDK table")
        if operation not in wanted:
            continue
        domain = event["domain"]
        if (operation in _SIGNAL and domain != 2) or (operation not in _SIGNAL and domain != 0):
            raise ResourceEvidenceError("Synchronization API has an unexpected SDK domain")
        cid, tid = _int(event.get("cid"), "actual correlation", 1), _int(event.get("tid"), "actual thread", 1)
        _int(event.get("epoch"), "actual epoch")
        _int(event.get("ns"), "actual callback clock", 1)
        call_key = domain, event["op"], cid, tid
        if event.get("phase") == 1:
            if call_key in pending:
                raise ResourceEvidenceError("Synchronization call has duplicate entry")
            pending[call_key] = event
        elif event.get("phase") == 2:
            before = pending.pop(call_key, None)
            if before is None or before["ns"] > event["ns"]:
                raise ResourceEvidenceError("Synchronization return lacks its real ordered entry")
            for field in ("ancestor", "epoch", "flags"):
                _same(event.get(field), before.get(field), "paired synchronization " + field)
            call = {"operation": operation, "enter": before, "exit": event}
            if cid in by_cid:
                raise ResourceEvidenceError("Synchronization correlation was reused")
            by_cid[cid] = call
            calls.append(call)
        else:
            raise ResourceEvidenceError("Synchronization API has an unknown callback phase")
    if pending:
        raise ResourceEvidenceError("Synchronization calls remain unpaired at SDK stop")
    return sorted(calls, key=lambda value: value["enter"]["seq"]), by_cid


def audit_synchronization_resources(native_report: dict, *, target_epochs: list[int], device_index: int,
                                     natural_shutdown_evidence: dict | None = None) -> dict:
    """Recompute restricted event/signal proofs from actual raw callbacks.

    Startup events may remain live as fixed synchronization resources. Formal
    acquisitions/destructions and event-linked formal HSA signals require real
    generation closure and a subsequent observed completion fence. No byte
    value is inferred from absent memory callbacks.
    """
    result: dict[str, Any] = {"schema_version": 1, "kind": "synchronization_resource_lifecycle_audit",
        "status": "PARTIAL", "complete_workspace": False, "hardware_validated": False,
        "allowed_call_proofs": [], "event_generations": [], "signal_generations": [],
        "startup_live_resources": [], "issues": [], "gpu_storage_bytes": None,
        "scope": "HIP_DisableTiming_event_and_event_linked_HSA_signal_resources_only",
        "remaining_obligations": ["All native GPU backing must independently match registered owners and budgets",
            "No allocation/free/pool/VMM/IPC/scratch operation is admitted by this helper",
            "Driver and opaque pool capacity is not inferred from zero allocation callbacks"]}
    try:
        _int(device_index, "target logical device")
        if (not isinstance(target_epochs, list) or not target_epochs
                or any(type(value) is not int or value <= 0 for value in target_epochs)
                or len(set(target_epochs)) != len(target_epochs)):
            raise ResourceEvidenceError("Explicit unique target model epochs are required")
        json.dumps(native_report, allow_nan=False)
        epochs = native_report.get("epochs")
        if not isinstance(epochs, list):
            raise ResourceEvidenceError("Actual native epoch windows are missing")
        windows = {}
        for epoch in epochs:
            identifier = _int(epoch.get("id"), "actual native epoch id", 1)
            if identifier in windows:
                raise ResourceEvidenceError("Native epoch identities are duplicated")
            _same(epoch.get("ended"), True, "actual closed native epoch")
            begin, end = _int(epoch.get("begin_ns"), "epoch begin", 1), _int(epoch.get("end_ns"), "epoch end", 1)
            if begin >= end:
                raise ResourceEvidenceError("Actual native epoch clock window is reversed")
            windows[identifier] = begin, end
        if not set(target_epochs) <= windows.keys():
            raise ResourceEvidenceError("Requested model epochs are absent from the actual trace")
        calls, by_cid = _calls(native_report)
        for call in calls:
            identifier = call["enter"]["epoch"]
            if identifier and (identifier not in windows
                    or not windows[identifier][0] <= call["enter"]["ns"] <= call["exit"]["ns"] <= windows[identifier][1]):
                raise ResourceEvidenceError("Synchronization call is outside its actual epoch window")
        shutdown = None
        if natural_shutdown_evidence is not None:
            evidence = natural_shutdown_evidence
            for key, value in {"process_id": native_report.get("pid"), "native_clock": "CLOCK_MONOTONIC_RAW",
                "original_returned": True, "return_type": "builtins.NoneType",
                "scope": "original_WorkerProc_shutdown_including_model_and_distributed_environment"}.items():
                _same(evidence.get(key), value, "actual natural shutdown." + key)
            _int(evidence.get("process_id"), "actual shutdown process", 1)
            _int(evidence.get("thread_id"), "actual shutdown thread", 1)
            if "exception" in evidence:
                raise ResourceEvidenceError("Original shutdown raised rather than completing naturally")
            source = evidence.get("source", {})
            _same(source.get("qualified_name"), "vllm.v1.executor.multiproc_executor.WorkerProc.shutdown",
                  "actual original shutdown method")
            _same(source.get("source_sha256"), "36f51b0ac5471bc230ccfce7dfc6ced2d16d2c73e35b10a3ab81eae1ec592140",
                  "pinned original shutdown method bytes")
            _same(source.get("file_sha256"), "3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a",
                  "pinned original shutdown module bytes")
            if not isinstance(source.get("source_reference"), str) or not source["source_reference"]:
                raise ResourceEvidenceError("Actual original shutdown source location is missing")
            begin = _int(evidence.get("started_monotonic_raw_ns"), "actual RAW shutdown start", 1)
            end = _int(evidence.get("ended_monotonic_raw_ns"), "actual RAW shutdown end", 1)
            if begin >= end:
                raise ResourceEvidenceError("Actual natural shutdown RAW window is reversed")
            shutdown = evidence
            result["natural_shutdown_evidence_sha256"] = hashlib.sha256(
                json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        _audit_resources(result, calls, by_cid, set(target_epochs), device_index, shutdown)
    except (ResourceEvidenceError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        result["issues"].append({"code": "UNPROVEN_SYNCHRONIZATION_RESOURCE", "message": str(exc)})
        result["allowed_call_proofs"] = []
    return result


def _audit_resources(result: dict, calls: list[dict], by_cid: dict[int, dict], targets: set[int], device: int,
                     shutdown: dict | None) -> None:
    target_calls = [call for call in calls if call["enter"]["epoch"] in targets]
    first_target = min((call["enter"]["seq"] for call in target_calls), default=None)
    if first_target is None:
        raise ResourceEvidenceError("Target epoch has no actual synchronization observation")
    contexts: dict[int, list[tuple[int, int | None, int | None]]] = defaultdict(list)
    for call in sorted(calls, key=lambda value: value["exit"]["seq"]):
        operation, a, b = call["operation"], call["enter"], call["exit"]
        if operation in _INVALIDATE:
            contexts[a["tid"]].append((a["seq"], None, None))
        elif operation in _DEVICE_QUERY | _DEVICE_SET:
            if b.get("status_type") != 1 or b.get("status") != 0:
                contexts[a["tid"]].append((b["seq"], None, None))
                continue
            payload = _payload(call)
            _same(payload.get("kind"), "device_query" if operation in _DEVICE_QUERY else "device_set", "actual device observation")
            actual = _int(payload.get("device_index"), "actual device result")
            if operation in _DEVICE_SET:
                _same(_payload(call, phase="enter").get("device_index"), actual, "actual set-device argument")
            else:
                _int(payload.get("output_address"), "actual device output pointer", 1)
            contexts[a["tid"]].append((b["seq"], actual, a["cid"]))
    for values in contexts.values():
        values.sort()

    def context_at(call: dict, phase: str = "enter") -> tuple[int | None, int | None]:
        row = call[phase]
        values = [value for value in contexts[row["tid"]] if value[0] < row["seq"]]
        return (values[-1][1], values[-1][2]) if values else (None, None)

    def observed_device(call: dict, phase: str = "exit") -> int | None:
        payload = _payload(call, phase=phase)
        actual, source = context_at(call, phase)
        if payload.get("device_index") is not None:
            _same(payload.get("device_index"), actual, "independent device context replay")
            _same(payload.get("device_source_cid"), source, "independent device source correlation")
        return actual

    generations: list[dict] = []
    by_handle: dict[int, list[dict]] = defaultdict(list)
    live: dict[int, dict] = {}
    call_events: dict[int, dict] = {}
    stream_devices: dict[int, int | None] = {}
    for call in calls:
        operation, a, b = call["operation"], call["enter"], call["exit"]
        if operation not in _EVENT:
            continue
        if operation in _CREATE:
            _success(call)
            entry, payload = _payload(call, phase="enter"), _payload(call)
            _same(entry.get("kind"), "event", "HIP event entry payload")
            _same(payload.get("kind"), "event", "HIP event return payload")
            _same(entry.get("output_address"), payload.get("output_address"), "actual event output pointer")
            _int(payload.get("output_address"), "actual event output pointer", 1)
            flags = _int(payload.get("flags"), "actual event flags")
            _same(entry.get("flags"), flags, "actual event creation flags")
            handle = _int(payload.get("object_handle"), "actual returned HIP event", 1)
            if handle in live:
                raise ResourceEvidenceError("HIP event handle was recreated before its actual destroy")
            generation = {"generation_id": "hip-event-" + str(handle) + "-c" + str(a["cid"]),
                "object_handle": handle, "flags": flags, "device_index": observed_device(call),
                "create": call, "destroy": None, "uses": [], "streams": [], "formal": a["epoch"] in targets,
                "startup": b["seq"] < first_target, "completion_fence": None}
            live[handle] = generation
            by_handle[handle].append(generation)
            generations.append(generation)
            call_events[a["cid"]] = generation
            continue
        payload, entry = _payload(call), _payload(call, phase="enter")
        _same(payload.get("kind"), "event", "actual event use payload")
        handle = _int(payload.get("object_handle"), "actual referenced HIP event", 1)
        _same(entry.get("object_handle"), handle, "paired actual HIP event handle")
        found_event = live.get(handle)
        if found_event is None:
            if a["epoch"] in targets:
                raise ResourceEvidenceError("Formal event reference lacks its global-startup generation")
            continue
        generation = found_event
        relevant = a["epoch"] in targets
        if relevant and generation["flags"] != 2:
            raise ResourceEvidenceError("Only exact hipEventDisableTiming flags2 are admitted")
        if operation == "hipEventQuery":
            _same(b.get("status_type"), 1, "typed HIP event query")
            if type(b.get("status")) is not int or b["status"] not in (0, 600):
                raise ResourceEvidenceError("HIP event query failed outside the known pending status")
        else:
            _success(call)
        if operation in _RECORD | _WAIT:
            if operation == "hipEventRecordWithFlags" or operation in _WAIT:
                _same(payload.get("flags"), 0, "event recording/wait flags stay in the supported eager domain")
                _same(entry.get("flags"), 0, "paired event recording/wait flags")
            stream = _int(payload.get("stream_handle"), "actual event stream")
            _same(entry.get("stream_handle"), stream, "paired actual event stream")
            current_device = observed_device(call)
            if current_device is None and relevant:
                raise ResourceEvidenceError("Formal event recording/wait has no actual device observation")
            if generation["device_index"] is None:
                generation["device_index"] = current_device
            elif current_device is not None and generation["device_index"] != current_device:
                raise ResourceEvidenceError("Event crosses an unproven device context")
            if stream in stream_devices and stream_devices[stream] != generation["device_index"]:
                raise ResourceEvidenceError("Stream changed device without a source-bound lifecycle")
            stream_devices[stream] = generation["device_index"]
            if stream not in generation["streams"]:
                generation["streams"].append(stream)
        generation["uses"].append(call)
        generation["formal"] = generation["formal"] or relevant
        call_events[a["cid"]] = generation
        if operation == "hipEventDestroy":
            generation["destroy"] = call
            del live[handle]
    fences = []
    for call in calls:
        if call["operation"] in _FENCE:
            if call["exit"].get("status_type") != 1 or call["exit"].get("status") != 0:
                continue
            payload = _payload(call)
            expected = "device_sync" if call["operation"] == "hipDeviceSynchronize" else "stream_sync"
            _same(payload.get("kind"), expected, "actual completion fence")
            fence_device = observed_device(call)
            if fence_device == device:
                fences.append(call)

    event_fences = []
    for call in calls:
        if call["operation"] not in {"hipEventSynchronize", "hipEventQuery"}:
            continue
        if call["exit"].get("status_type") != 1 or call["exit"].get("status") != 0:
            continue
        event = call_events.get(call["enter"]["cid"])
        if event is None or event["device_index"] != device:
            continue
        records = [value for value in event["uses"] if value["operation"] in _RECORD
                   and value["exit"]["seq"] < call["enter"]["seq"]]
        if records:
            event_fences.append((call, records[-1], event))

    def fence_after(sequence: int, generation: dict) -> dict | None:
        for fence in fences:
            if fence["enter"]["seq"] <= sequence:
                continue
            if fence["operation"] == "hipDeviceSynchronize":
                return fence
            stream = _payload(fence).get("stream_handle")
            if len(generation["streams"]) == 1 and stream == generation["streams"][0]:
                return fence
        for fence, record, event in event_fences:
            if (event is not generation and record["enter"]["seq"] > sequence
                    and len(generation["streams"]) == 1
                    and _payload(record)["stream_handle"] == generation["streams"][0]):
                return fence
        return None

    def completion_fence(destroy: dict, generation: dict, *, parent_use: dict | None = None) -> tuple[dict | None, str | None]:
        after = fence_after(destroy["exit"]["seq"], generation)
        if after is not None:
            return after, "destroy_then_observed_completion_fence"
        if shutdown is None:
            return None, None
        first, last = shutdown["started_monotonic_raw_ns"], shutdown["ended_monotonic_raw_ns"]
        if not first <= destroy["enter"]["ns"] <= destroy["exit"]["ns"] <= last:
            return None, None
        if destroy["enter"]["tid"] != shutdown["thread_id"]:
            return None, None
        submissions = [value for value in generation["uses"] if value["operation"] in _RECORD | _WAIT
                       and value["exit"]["seq"] < destroy["enter"]["seq"]]
        origin = generation["create"]["exit"]["seq"]
        last_use = max([origin, *[value["exit"]["seq"] for value in submissions],
                        *([] if parent_use is None else [parent_use["exit"]["seq"]])])
        pending_queries = [value for value in generation["uses"] if value["operation"] == "hipEventQuery"
                           and value["exit"].get("status") == 600]
        for fence in reversed(fences):
            if not last_use < fence["enter"]["seq"] < fence["exit"]["seq"] < destroy["enter"]["seq"]:
                continue
            if fence["operation"] == "hipStreamSynchronize":
                if len(generation["streams"]) != 1 or _payload(fence)["stream_handle"] != generation["streams"][0]:
                    continue
            if any(fence["exit"]["seq"] < query["exit"]["seq"] < destroy["enter"]["seq"] for query in pending_queries):
                continue
            return fence, "last_GPU_use_then_completion_fence_then_original_CPU_shutdown_destroy"
        for fence, record, event in reversed(event_fences):
            if (event is generation or record["enter"]["seq"] <= last_use
                    or fence["exit"]["seq"] >= destroy["enter"]["seq"]
                    or len(generation["streams"]) != 1
                    or _payload(record)["stream_handle"] != generation["streams"][0]):
                continue
            if any(fence["exit"]["seq"] < query["exit"]["seq"] < destroy["enter"]["seq"] for query in pending_queries):
                continue
            return fence, "last_GPU_use_then_recorded_event_fence_then_original_CPU_shutdown_destroy"
        return None, None

    proofs = []
    relevant_generations = [generation for generation in generations if generation["formal"]]
    for generation in relevant_generations:
        if generation["flags"] != 2 or generation["device_index"] != device:
            raise ResourceEvidenceError("Formal resource lacks exact DisableTiming/target-device binding")
        destroy = generation["destroy"]
        if destroy is None:
            if not generation["startup"]:
                raise ResourceEvidenceError("Formal HIP event acquisition remains live at the closed observation boundary")
            state = "LIVE_FIXED_STARTUP_SYNCHRONIZATION_RESOURCE"
            result["startup_live_resources"].append(generation["generation_id"])
        else:
            fence, fence_order = completion_fence(destroy, generation)
            if fence is None:
                raise ResourceEvidenceError("HIP destroy returned without an observed later completion fence")
            generation["completion_fence"] = fence
            generation["completion_order"] = fence_order
            state = "DESTROYED_AND_COMPLETION_FENCED"
        for call in [generation["create"], *generation["uses"]]:
            if call["enter"]["epoch"] in targets:
                proofs.append(_proof(call, generation["generation_id"], generation["completion_fence"]))
        result["event_generations"].append({"generation_id": generation["generation_id"],
            "object_handle": generation["object_handle"], "flags": generation["flags"],
            "device_index": generation["device_index"], "state": state,
            "create": _proof(generation["create"], generation["generation_id"]),
            "destroy": None if destroy is None else _proof(destroy, generation["generation_id"]),
            "uses": [_proof(call, generation["generation_id"]) for call in generation["uses"]],
            "completion_fence": None if generation["completion_fence"] is None
                else _proof(generation["completion_fence"], generation["generation_id"]),
            "completion_order": generation.get("completion_order")})

    def parent_event(call: dict) -> tuple[dict | None, dict | None]:
        child = call
        seen = set()
        while child["enter"].get("ancestor"):
            cid = _int(child["enter"]["ancestor"], "actual signal ancestor", 1)
            if cid in seen:
                raise ResourceEvidenceError("Signal ancestry cycles")
            seen.add(cid)
            parent = by_cid.get(cid)
            if parent is None:
                return None, None
            if (parent["enter"]["tid"] != call["enter"]["tid"]
                    or not parent["enter"]["seq"] < child["enter"]["seq"] < child["exit"]["seq"] < parent["exit"]["seq"]):
                raise ResourceEvidenceError("Signal ancestry is not the actual nested same-thread call")
            if parent["operation"] == "hipDeviceSynchronize":
                return None, None  # Existing direct-device-sync rule stays separate.
            if parent["operation"] in _EVENT:
                return call_events.get(cid), parent
            child = parent
        return None, None

    signal_live = {}
    signal_generations = []
    for call in calls:
        if call["operation"] not in _SIGNAL:
            continue
        a, b = call["enter"], call["exit"]
        if call["operation"] == "hsa_signal_create":
            event_generation, parent = parent_event(call)
            if event_generation is not None and parent is None:
                raise ResourceEvidenceError("Event-linked signal has no actual parent call")
            _success(call, 2)
            payload = _payload(call, "raw_hsa_signal_payload")
            _same(payload.get("operation"), "create", "actual HSA signal operation")
            handle = _int(payload.get("signal_handle"), "actual returned HSA signal handle", 1)
            _int(payload.get("output_address"), "actual HSA signal output pointer", 1)
            _same(payload.get("consumers_observed"), True, "actual HSA consumers read")
            count = _int(payload.get("num_consumers"), "actual HSA consumer count")
            consumers = payload.get("consumer_agents")
            if (count > 4 or not isinstance(consumers, list) or len(consumers) != count
                    or any(type(value) is not int or value <= 0 for value in consumers)):
                raise ResourceEvidenceError("HSA consumer agents were not completely observed")
            initial = payload.get("initial_value")
            if type(initial) is not int or not -(2**63) <= initial <= 2**63 - 1:
                raise ResourceEvidenceError("Actual initial signal value is missing")
            entry = _payload(call, "raw_hsa_signal_payload", "enter")
            for field in ("initial_value", "num_consumers", "output_address"):
                _same(entry.get(field), payload.get(field), "paired HSA signal argument " + field)
            if handle in signal_live:
                raise ResourceEvidenceError("HSA signal handle recreated before actual destroy")
            if parent is not None:
                _success(parent)
            generation = {"generation_id": "hsa-signal-" + str(handle) + "-c" + str(a["cid"]),
                "signal_handle": handle, "create": call, "destroy": None,
                "parent": parent, "event_generation": event_generation,
                "initial_value": initial, "num_consumers": count, "consumer_agents": consumers}
            signal_live[handle] = generation
            signal_generations.append(generation)
        else:
            payload = _payload(call, "raw_hsa_signal_payload")
            handle = _int(payload.get("signal_handle"), "actual HSA signal destroy handle", 1)
            found_signal = signal_live.get(handle)
            if found_signal is None:
                continue
            generation = found_signal
            _success(call, 2)
            _same(payload.get("operation"), "destroy", "actual HSA signal destroy")
            _same(_payload(call, "raw_hsa_signal_payload", "enter").get("signal_handle"), handle,
                  "paired HSA signal destroy handle")
            generation["destroy"] = call
            del signal_live[handle]
    for generation in signal_generations:
        create, destroy = generation["create"], generation["destroy"]
        if create["enter"]["epoch"] not in targets and (destroy is None or destroy["enter"]["epoch"] not in targets):
            continue
        event_generation = generation["event_generation"]
        if event_generation is None:
            continue  # A different, unchanged native rule must admit this call.
        if event_generation not in relevant_generations or event_generation["device_index"] != device:
            raise ResourceEvidenceError("Formal HSA signal does not belong to an admitted target-device event")
        if destroy is None:
            raise ResourceEvidenceError("Formal event-linked HSA signal remains live at SDK stop")
        fence, fence_order = completion_fence(destroy, event_generation, parent_use=generation["parent"])
        if fence is None:
            raise ResourceEvidenceError("HSA signal destroy has no observed later completion fence")
        for call in (create, destroy):
            if call["enter"]["epoch"] in targets:
                proofs.append(_proof(call, generation["generation_id"], fence))
        result["signal_generations"].append({"generation_id": generation["generation_id"],
            "signal_handle": generation["signal_handle"], "initial_value": generation["initial_value"],
            "num_consumers": generation["num_consumers"], "consumer_agents": generation["consumer_agents"],
            "parent_event_generation_id": event_generation["generation_id"],
            "parent_call": _proof(generation["parent"], event_generation["generation_id"]),
            "create": _proof(create, generation["generation_id"]), "destroy": _proof(destroy, generation["generation_id"]),
            "completion_fence": _proof(fence, generation["generation_id"]),
            "completion_order": fence_order,
            "gpu_storage_bytes": None})
    result.update(status="SYNCHRONIZATION_RESOURCES_PROVEN", allowed_call_proofs=proofs)
