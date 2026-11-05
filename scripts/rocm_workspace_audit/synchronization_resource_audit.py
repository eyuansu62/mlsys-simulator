"""Restricted, generation-bound HIP event/HSA signal resource admission.

This proves control-resource lifetimes, never zero GPU memory. Every underlying
allocation, pool, mapping, private scratch and native GPU backing still requires
the independent native memory audit and its separate fixed-budget proof.
"""

from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable


_CREATE = {"hipEventCreateWithFlags", "hipEventCreate"}
_RECORD = {"hipEventRecord", "hipEventRecord_spt", "hipEventRecordWithFlags"}
_WAIT = {"hipStreamWaitEvent", "hipStreamWaitEvent_spt"}
_EVENT = _CREATE | _RECORD | _WAIT | {"hipEventDestroy", "hipEventSynchronize", "hipEventQuery"}
_DEVICE_QUERY = {"hipGetDevice", "hipCtxGetDevice"}
_DEVICE_SET = {"hipSetDevice"}
_INVALIDATE = {"hipCtxSetCurrent", "hipCtxPushCurrent", "hipCtxPopCurrent", "hipCtxCreate", "hipCtxDestroy"}
_FENCE = {"hipDeviceSynchronize", "hipStreamSynchronize"}
_SIGNAL = {"hsa_signal_create", "hsa_signal_destroy"}
_HOST_QUERY = {"hsa_amd_pointer_info": (3, "raw_hsa_pointer_info_payload"),
    "hsa_agent_get_info": (2, "raw_hsa_agent_info_payload"),
    "hsa_amd_signal_get_event_id": (3, "raw_hsa_signal_event_payload")}


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


def _calls(data: dict, include_host: bool = False) -> tuple[list[dict], dict[int, dict]]:
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
    if include_host:
        wanted |= _HOST_QUERY.keys()
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
        expected_domain = _HOST_QUERY[operation][0] if operation in _HOST_QUERY else (2 if operation in _SIGNAL else 0)
        if domain != expected_domain:
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


def _artifact(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ResourceEvidenceError(label + ": source-bound artifact is missing")
    digest = value.get("sha256")
    if (not isinstance(value.get("path"), str) or not value["path"] or not isinstance(digest, str)
            or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ResourceEvidenceError(label + ": actual path/SHA256 is missing")


def _source_record(value: Any, suffix: str, worker: dict) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("qualified_name"), str) or not value["qualified_name"].endswith(suffix):
        raise ResourceEvidenceError("Actual worker source method binding differs: " + suffix)
    _artifact({"path": value.get("source_reference"), "sha256": value.get("source_sha256")}, "actual worker method source")
    _same(value.get("source_reference"), worker.get("worker_source_reference"), "worker method/source file identity")
    _same(value.get("file_sha256"), worker.get("worker_source_sha256"), "worker method/source file bytes")


def _registry_at(native: dict, by_cid: dict[int, dict], export: dict, label: str) -> dict:
    observed = _live_export(export, native, label)
    cursor = export["event_cursor"]
    if cursor > len(native["events"]):
        raise ResourceEvidenceError("Live registry cursor lies outside the intact actual SDK tape")
    live: dict[int, dict] = {}
    counts: dict[int, int] = defaultdict(int)
    unknown = 0
    for call in sorted(by_cid.values(), key=lambda value: value["exit"]["seq"]):
        if call["operation"] not in _SIGNAL or call["exit"]["seq"] > cursor:
            continue
        if call["exit"].get("status_type") != 2 or call["exit"].get("status") != 0:
            continue
        payload = _payload(call, "raw_hsa_signal_payload")
        handle = _int(payload.get("signal_handle"), "actual registry signal", 1)
        if call["operation"] == "hsa_signal_destroy":
            if handle in live:
                del live[handle]
            else:
                unknown += 1
            continue
        if handle in live:
            raise ResourceEvidenceError("Raw registry replay recreates a still-live signal generation")
        counts[handle] += 1
        a, b = call["enter"], call["exit"]
        live[handle] = {"signal_handle": handle, "generation": counts[handle], "create_cid": a["cid"],
            "create_seq": b["seq"], "create_epoch": a["epoch"], "create_tid": a["tid"],
            **{key: payload.get(key) for key in ("initial_value", "num_consumers", "consumers_observed", "consumer_agents")}}
    _same(observed, live, label + " equals full SDK logical-generation replay at its actual cursor")
    _same(export.get("unknown_destroy_count"), unknown, label + " actual unknown older destroys")
    if cursor and native["events"][cursor - 1]["ns"] > export["snapshot_monotonic_raw_ns"]:
        raise ResourceEvidenceError("Registry snapshot clock precedes its actual committed callback")
    return live


def _unwrap_worker_host_evidence(native: dict, evidence: dict, by_cid: dict[int, dict], targets: set[int],
                                 device: int, shutdown: dict | None) -> tuple[dict, dict | None]:
    if evidence.get("kind") != "public_hsa_signal_backing_worker_evidence":
        return evidence, None
    expected: dict[str, Any] = {"schema_version": 1, "process_id": native.get("pid"),
        "status": "CAPTURE_RETURNED_REQUIRES_INDEPENDENT_NATIVE_AUDIT", "complete_workspace": False,
        "hardware_validated": False}
    for key, value in expected.items():
        _same(evidence.get(key), value, "actual worker host-evidence envelope." + key)
    pid = _int(evidence.get("process_id"), "actual worker process", 1)
    tid = _int(evidence.get("thread_id"), "actual worker host-evidence thread", 1)
    capture = evidence.get("capture")
    if not isinstance(capture, dict):
        raise ResourceEvidenceError("Original public capture object is missing from worker envelope")
    _artifact(evidence.get("capture_artifact"), "original public capture artifact")
    _int(evidence["capture_artifact"].get("bytes"), "original public capture artifact bytes", 1)
    for key in ("process_id", "thread_id", "source_fence", "source_scope"):
        _same(capture.get(key), evidence.get(key), "worker/original public capture " + key)
    scope, fence = evidence.get("source_scope"), evidence.get("source_fence")
    if not isinstance(scope, dict) or not isinstance(fence, dict):
        raise ResourceEvidenceError("Source-bound pre-shutdown query scope/fence is missing")
    scope_expected = {"kind": "post_formal_pre_natural_shutdown_public_signal_queries", "process_id": pid,
        "thread_id": tid, "target_device_index": device, "native_clock": "CLOCK_MONOTONIC_RAW",
        "formal_native_epochs": sorted(targets), "query_started_monotonic_raw_ns": evidence.get("query_started_monotonic_raw_ns")}
    for key, value in scope_expected.items():
        _same(scope.get(key), value, "actual worker query scope." + key)
    if not isinstance(scope.get("session_id"), str) or not scope["session_id"]:
        raise ResourceEvidenceError("Actual worker session identity is missing")
    worker = scope.get("worker_source")
    if not isinstance(worker, dict):
        raise ResourceEvidenceError("Actual worker source bytes are not bound")
    _artifact({"path": worker.get("worker_source_reference"), "sha256": worker.get("worker_source_sha256")}, "worker module")
    _source_record(scope.get("source"), ".WorkspaceWorker._ws_capture_host_signals", worker)
    _source_record(fence.get("source"), ".WorkspaceWorker.mlsys_workspace_close", worker)
    for key, value in {"process_id": pid, "thread_id": tid, "target_device_index": device,
        "native_clock": "CLOCK_MONOTONIC_RAW", "native_epoch_id": 9001,
        "method": "existing_post_cohort_last_target_device_synchronize"}.items():
        _same(fence.get(key), value, "original target-device source fence." + key)
    if fence.get("observation_errors"):
        raise ResourceEvidenceError("Original target-device fence observations contain errors")
    before, after = (_int(fence.get("native_cursor_" + key), "actual fence " + key + " cursor") for key in ("before", "after"))
    start, end = (_int(fence.get(key + "_monotonic_raw_ns"), "actual fence RAW " + key, 1) for key in ("started", "ended"))
    candidates = [call for call in by_cid.values() if call["operation"] == "hipDeviceSynchronize"
        and before < call["enter"]["seq"] < call["exit"]["seq"] <= after
        and start <= call["enter"]["ns"] <= call["exit"]["ns"] <= end
        and call["enter"]["tid"] == tid and call["enter"]["epoch"] == 9001]
    if len(candidates) != 1:
        raise ResourceEvidenceError("Worker fence does not uniquely identify its actual typed SDK device synchronize")
    actual_fence = candidates[0]
    _success(actual_fence)
    qstart, qend = (_int(evidence.get("query_" + key + "_monotonic_raw_ns"), "actual worker query RAW " + key, 1)
                    for key in ("started", "ended"))
    cstart, cend = (_int(capture.get("capture_" + key + "_monotonic_raw_ns"), "actual provider RAW " + key, 1)
                    for key in ("started", "ended"))
    epochs = {row["id"]: row for row in native["epochs"]}
    if 9001 not in epochs or not all(epochs[key]["end_ns"] < start for key in targets):
        raise ResourceEvidenceError("Worker host queries do not follow the complete formal cohort")
    if not start <= end <= epochs[9001]["end_ns"] <= qstart <= cstart <= cend <= qend:
        raise ResourceEvidenceError("Original fence/teardown/query RAW windows are out of order")
    left, right = (_int(evidence.get("native_cursor_" + key), "actual worker query " + key + " cursor") for key in ("before", "after"))
    a, b = capture.get("before_live"), capture.get("after_live")
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ResourceEvidenceError("Worker capture lacks complete original A/B live exports")
    for key in ("before_live_artifact", "after_live_artifact"):
        _artifact(capture.get(key), "original " + key)
    live_a, live_b = _registry_at(native, by_cid, a, "A before query"), _registry_at(native, by_cid, b, "B after query")
    _same(live_a, live_b, "actual A/B stable standard-signal logical generation set")
    if not after <= left <= a["event_cursor"] <= b["event_cursor"] <= right <= len(native["events"]):
        raise ResourceEvidenceError("Worker/source-fence/A/B callback cursor windows are out of order")
    if not cstart <= a["snapshot_monotonic_raw_ns"] <= b["snapshot_monotonic_raw_ns"] <= cend:
        raise ResourceEvidenceError("A/B original RAW snapshots lie outside the actual provider window")
    post = evidence.get("after_natural_shutdown_live")
    if post is not None:
        if not isinstance(post, dict) or shutdown is None:
            raise ResourceEvidenceError("Post-shutdown registry snapshot has no actual original shutdown receipt")
        for key, value in {"process_id": pid, "kind": "post_natural_shutdown_observed_signal_generations",
            "status": "OBSERVED_REGISTRY_SNAPSHOT_ONLY", "new_signal_queries_performed": False,
            "complete_workspace": False, "hardware_validated": False, "export_returncode": 0}.items():
            _same(post.get(key), value, "actual post-shutdown registry receipt." + key)
        _same(post.get("thread_id"), shutdown["thread_id"], "post-shutdown registry actual original thread")
        _artifact(post, "original post-shutdown registry artifact")
        _int(post.get("bytes"), "post-shutdown registry artifact bytes", 1)
        c = post.get("observed_live")
        if not isinstance(c, dict):
            raise ResourceEvidenceError("Actual C post-shutdown registry export is missing")
        _registry_at(native, by_cid, c, "C after natural shutdown")
        if not (qend < shutdown["started_monotonic_raw_ns"] <= shutdown["ended_monotonic_raw_ns"]
                <= c["snapshot_monotonic_raw_ns"] < native["stop_evidence"].get("begin_ns", 0)
                and right <= c["event_cursor"] <= len(native["events"])):
            raise ResourceEvidenceError("Natural shutdown/C registry/SDK-stop RAW boundaries are out of order")
    elif shutdown is not None:
        raise ResourceEvidenceError("Actual worker shutdown has no original C registry observation")
    kernel = evidence.get("kfd_event_evidence")
    if kernel is not None:
        if not isinstance(kernel, dict):
            raise ResourceEvidenceError("Worker/capture KFD evidence has no original common object")
        original = {key: value for key, value in kernel.items() if key not in ("final_report_artifact", "final_snapshot_observation")}
        if capture.get("kfd_event_evidence") is None:
            _same(original, {}, "unbound final KFD diagnostic cannot invent a query-period source binding")
        else:
            _same(original, capture["kfd_event_evidence"], "worker preserves original KFD query evidence byte-for-byte")
        capture = {**capture, "kfd_event_evidence": kernel}
    return capture, {"fence_cid": actual_fence["enter"]["cid"], "source_scope": scope,
        "query_started_monotonic_raw_ns": qstart, "query_ended_monotonic_raw_ns": qend,
        "post_shutdown_registry_observed": post is not None}


def _adapt_host_evidence(native: dict, evidence: dict, by_cid: dict[int, dict], targets: set[int]) -> dict | None:
    """Join the public provider's bounded windows to immutable actual SDK calls."""
    if evidence.get("kind") != "public_hsa_signal_backing_capture":
        return evidence
    if evidence.get("status") == "NO_LIVE_FORMAL_TARGETS":
        _same(evidence.get("process_id"), native.get("pid"), "not-applicable host evidence process")
        _int(native.get("pid"), "actual not-applicable native process", 1)
        _same(evidence.get("records"), [], "no per-generation host claims when not applicable")
        _same(evidence.get("issues"), [], "not-applicable host producer has no capture error")
        _same(evidence.get("ordinary_signal_control"), None, "not-applicable host evidence has no invented control")
        before = _live_export(evidence.get("before_live"), native, "not-applicable before")
        after = _live_export(evidence.get("after_live"), native, "not-applicable after")
        _same(before, after, "not-applicable stable exported resources")
        live: dict[int, dict] = {}
        cursor = evidence["before_live"]["event_cursor"]
        for call in sorted(by_cid.values(), key=lambda value: value["enter"]["seq"]):
            if call["operation"] not in _SIGNAL or call["exit"]["seq"] > cursor:
                continue
            _success(call, 2)
            handle = _int(_payload(call, "raw_hsa_signal_payload").get("signal_handle"), "actual no-target signal", 1)
            if call["operation"] == "hsa_signal_create":
                if handle in live:
                    raise ResourceEvidenceError("Not-applicable resource tape recreates a still-live signal")
                live[handle] = call
            else:
                live.pop(handle, None)
        if any(call["enter"]["epoch"] in targets for call in live.values()) or any(row.get("create_epoch") in targets for row in before.values()):
            raise ResourceEvidenceError("Not-applicable host evidence conceals an actual live formal signal")
        return None  # The unchanged destroyed-resource rule must prove every real target.
    expected_fields: dict[str, Any] = {"schema_version": 1, "process_id": native.get("pid"), "status": "CAPTURED",
        "issues": [], "complete_workspace": False, "hardware_validated": False}
    for key, expected in expected_fields.items():
        _same(evidence.get(key), expected, "actual public provider capture." + key)
    _artifact(evidence.get("provider"), "public provider source")
    sources, libraries = evidence.get("sources"), evidence.get("runtime_dsos")
    if not isinstance(sources, list) or not sources or not isinstance(libraries, list) or not libraries:
        raise ResourceEvidenceError("Actual compiled public provider/runtime source bindings are missing")
    for artifact in [*sources, *libraries]:
        _artifact(artifact, "actual public provider/runtime artifact")
    abi = dict(evidence.get("initialization", {}).get("abi", {}))
    compiled_sources = [value for value in sources if value["path"].endswith(".cpp")]
    if len(compiled_sources) != 1:
        raise ResourceEvidenceError("Actual compiled public ABI provider source is not unique")
    compiler_source = compiled_sources[0]
    if evidence.get("compiler_source") is not None:
        actual = evidence["compiler_source"]
        _artifact(actual, "actual compiled provider __FILE__ artifact")
        _same(actual.get("sha256"), compiler_source.get("sha256"), "actual compiled/frozen provider source SHA256")
        # __FILE__ names the actual build input; the frozen sibling is a
        # separate expected-byte copy. Authenticate both without inventing a
        # path identity between the two real source files.
        built_bytes = _read_artifact(actual, "actual compiled provider source", json_value=False)
        frozen_bytes = _read_artifact(compiler_source, "frozen expected provider source", json_value=False)
        _same(len(built_bytes), len(frozen_bytes), "actual compiled/frozen provider source extent")
        _same(built_bytes, frozen_bytes, "actual compiled/frozen provider source bytes")
        declared = evidence.get("initialization", {}).get("compiled_source_path")
        if (not isinstance(declared, str) or not Path(declared).is_absolute()
                or Path(declared).resolve(strict=True) != Path(actual["path"]).resolve(strict=True)):
            raise ResourceEvidenceError("Actual provider __FILE__ does not resolve to its independently authenticated build-input artifact")
        compiler_source = actual
    abi.update(compiler_source=compiler_source, headers=evidence.get("abi_headers"), runtime_dsos=libraries)

    def join(observed: dict) -> dict:
        operation = observed.get("api")
        if operation not in _HOST_QUERY and operation not in _SIGNAL:
            raise ResourceEvidenceError("Provider names an unsupported public signal API")
        before = _int(observed.get("before_cursor"), "provider actual before cursor")
        after = _int(observed.get("after_cursor"), "provider actual after cursor")
        begin = _int(observed.get("started_monotonic_raw_ns"), "provider actual query RAW start", 1)
        end = _int(observed.get("ended_monotonic_raw_ns"), "provider actual query RAW end", 1)
        if before > after or begin > end or after > len(native["events"]):
            raise ResourceEvidenceError("Provider public API has invalid actual callback/clock bounds")
        candidates = [call for call in by_cid.values() if call["operation"] == operation
            and before < call["enter"]["seq"] < call["exit"]["seq"] <= after
            and begin <= call["enter"]["ns"] <= call["exit"]["ns"] <= end]
        if len(candidates) != 1:
            raise ResourceEvidenceError("Provider query window has no unique actual SDK API call")
        call = candidates[0]
        binding = observed.get("api_binding", {})
        _same(binding.get("symbol_name"), operation, "provider actual resolved API symbol")
        _int(binding.get("address"), "actual public API address", 1)
        _int(binding.get("module_base"), "actual public API module base", 1)
        if not any(binding.get("module_path") in (row.get("path"), row.get("loaded_path")) for row in libraries):
            raise ResourceEvidenceError("Public API is not bound to the observed runtime DSO artifact")
        input_value, output = observed.get("input"), observed.get("output")
        if not isinstance(input_value, dict) or not isinstance(output, dict):
            raise ResourceEvidenceError("Provider actual public API arguments/results are missing")
        _same(output.get("status_type"), 2, "provider actual typed HSA return")
        _same(output.get("status"), call["exit"].get("status"), "provider return matches actual SDK typed result")
        field = _HOST_QUERY[operation][1] if operation in _HOST_QUERY else "raw_hsa_signal_payload"
        entry, payload = _payload(call, field, "enter"), _payload(call, field)
        for key, value in input_value.items():
            if key not in entry:
                raise ResourceEvidenceError("Provider argument is absent from the actual typed SDK payload: " + key)
            # The SDK only reads the const consumer list after successful
            # return; it never invents an entry-time list for an unread pointer.
            actual = payload[key] if operation == "hsa_signal_create" and key == "consumer_agents" else entry[key]
            _same(value, actual, "provider/SDK actual input " + key)
        for key, value in output.items():
            if key not in ("status_type", "status"):
                if key not in payload:
                    raise ResourceEvidenceError("Provider result is absent from actual SDK payload: " + key)
                _same(value, payload[key], "provider/SDK actual output " + key)
        _success(call, 2)
        return call

    def sample(observed: dict, handle: int) -> dict:
        _same(observed.get("process_id"), native.get("pid"), "actual query sample process")
        _same(observed.get("signal_handle"), handle, "actual sample signal handle")
        _same(observed.get("abi_signal_address"), handle, "installed unhinted ABI sample address")
        _same(observed.get("address_rule"), "unhinted_amd_signal_handle", "actual installed ABI address rule")
        api_calls = observed.get("api_calls")
        if not isinstance(api_calls, list) or len(api_calls) != 3:
            raise ResourceEvidenceError("Provider sample does not contain the three actual public queries")
        linked = [join(value) for value in api_calls]
        if {value["operation"] for value in linked} != _HOST_QUERY.keys():
            raise ResourceEvidenceError("Provider sample does not uniquely query pointer/owner/event identity")
        for call in linked:
            _same(observed.get("thread_id"), call["enter"]["tid"], "actual provider sample thread")
        result: dict[str, Any] = {key: next(value["enter"]["cid"] for value in linked if value["operation"] == operation)
            for key, operation in {"pointer_info_cid": "hsa_amd_pointer_info", "owner_device_cid": "hsa_agent_get_info",
                                   "event_id_cid": "hsa_amd_signal_get_event_id"}.items()}
        if "kfd_window" in observed:
            window = observed["kfd_window"]
            if not isinstance(window, dict):
                raise ResourceEvidenceError("Original per-sample KFD observation window is missing")
            first = _int(window.get("before_cursor"), "actual before-query KFD cursor")
            last = _int(window.get("after_cursor"), "actual after-query KFD cursor")
            begin = _int(window.get("started_monotonic_raw_ns"), "actual KFD sample RAW start", 1)
            end = _int(window.get("ended_monotonic_raw_ns"), "actual KFD sample RAW end", 1)
            if (first > last or last == 2**64 - 1 or begin > end
                    or any(not begin <= value["enter"]["ns"] <= value["exit"]["ns"] <= end for value in linked)):
                raise ResourceEvidenceError("KFD sample cursor/RAW window does not enclose its actual SDK queries")
            result["kfd_window"] = window
        return result

    records = evidence.get("records")
    if not isinstance(records, list):
        raise ResourceEvidenceError("Actual provider per-generation records are missing")
    adapted = []
    for record in records:
        if not isinstance(record.get("samples"), list):
            raise ResourceEvidenceError("Actual provider per-generation samples are missing")
        adapted.append({**record, "samples": [sample(value, record["signal_handle"]) for value in record["samples"]]})
    control = evidence.get("ordinary_signal_control", {})
    create, destroy = join(control.get("create_call", {})), join(control.get("destroy_call", {}))
    handle = _int(_payload(create, "raw_hsa_signal_payload").get("signal_handle"), "actual control signal", 1)
    adapted_control = {"method": "same_process_traced_standard_signal_control", "process_id": native.get("pid"),
        "create_cid": create["enter"]["cid"], "destroy_cid": destroy["enter"]["cid"],
        "samples": [sample(value, handle) for value in control.get("samples", [])]}
    return {**evidence, "kind": "public_hsa_signal_host_backing_evidence", "abi": abi,
        "records": adapted, "ordinary_signal_control": adapted_control}


def _host_manifest(native: dict, evidence: dict) -> dict:
    """Bindings are authenticated against artifact bytes by the session assembler.

    This independently checks their internal agreement and the actual callback
    outputs. A provider assertion about ownership is never used as a proof.
    """
    _same(evidence.get("schema_version"), 1, "host signal evidence schema")
    _same(evidence.get("kind"), "public_hsa_signal_host_backing_evidence", "host signal evidence kind")
    _same(evidence.get("process_id"), native.get("pid"), "host signal query process")
    _int(native.get("pid"), "actual native process", 1)
    _artifact(evidence.get("provider"), "callback-outside provider source")
    abi = evidence.get("abi")
    if not isinstance(abi, dict):
        raise ResourceEvidenceError("Compiled public signal ABI manifest is missing")
    _artifact(abi.get("compiler_source"), "compiled ABI probe source")
    headers, libraries = abi.get("headers"), abi.get("runtime_dsos")
    if not isinstance(headers, list) or not isinstance(libraries, list) or not headers or not libraries:
        raise ResourceEvidenceError("Actual signal ABI headers/runtime DSO identities are missing")
    for artifact in [*headers, *libraries]:
        _artifact(artifact, "public ABI/runtime")
    # This narrowly supported installed ABI has no low-bit handle hints/helper.
    # Other ABI revisions require their own compiled decoder/control, not a guess.
    if not any(row["path"].endswith("/amd_hsa_signal.h") and row["sha256"] ==
            "a8606371af1c186417963f640a0d2a418805a313b0c8c0fe23aa42b0d813cecd" for row in headers):
        raise ResourceEvidenceError("Installed amd_signal_t address ABI is not supported")
    _same(abi.get("signal_handle_address_mask"), 2**64 - 1, "installed public signal address decoding")
    span = _int(abi.get("amd_signal_size"), "actual compiled signal span", 1)
    alignment = _int(abi.get("amd_signal_alignment"), "actual compiled signal alignment", 1)
    if span % alignment or alignment & (alignment - 1):
        raise ResourceEvidenceError("Compiled signal span/alignment is invalid")
    table = native.get("hsa_query_storage")
    if not isinstance(table, dict):
        raise ResourceEvidenceError("Bounded native public-query storage evidence is missing")
    rows = [row for row in native["events"] if any(key in row for _, key in _HOST_QUERY.values())]
    for field, expected in {"schema_version": 1, "method": "bounded_public_hsa_query_payloads_v1",
        "capacity": 8192, "retained_records": len(rows), "dropped_records": 0, "read_errors": 0}.items():
        _same(table.get(field), expected, "actual bounded query table." + field)
    if len(rows) > 8192:
        raise ResourceEvidenceError("Native public-query storage capacity was exceeded")
    _same(table.get("storage_capacity_bytes"), 8192 * _int(table.get("record_size_bytes"),
        "actual public-query record size", 1), "actual public-query storage bytes")
    for left, right in {"device_attribute": "agent_info_device_attribute", "cpu_device_enum": "cpu_device_type",
        "pointer_type_hsa": "pointer_type_hsa", "pointer_type_locked": "pointer_type_locked",
        "pointer_info_size": "pointer_info_struct_size", "pointer_info_fields": "pointer_info_fields"}.items():
        _same(abi.get(left), table.get(right), "independently compiled public ABI." + left)
    fields, size = abi.get("pointer_info_fields"), _int(abi.get("pointer_info_size"), "pointer-info ABI size", 1)
    if not isinstance(fields, dict) or not {"type", "host_base_address", "size_bytes", "owner_agent"} <= fields.keys():
        raise ResourceEvidenceError("Pointer-info ABI field extents are missing")
    occupied: set[int] = set()
    for offset_width in fields.values():
        if not isinstance(offset_width, list) or len(offset_width) != 2:
            raise ResourceEvidenceError("Pointer-info ABI field has no actual extent")
        offset, width = _int(offset_width[0], "ABI offset"), _int(offset_width[1], "ABI width", 1)
        covered = set(range(offset, offset + width))
        if offset + width > size or occupied & covered:
            raise ResourceEvidenceError("Pointer-info ABI fields overlap or exceed the actual structure")
        occupied |= covered
    for key in ("device_attribute", "cpu_device_enum", "pointer_type_hsa", "pointer_type_locked"):
        _int(abi.get(key), "actual compiled constant " + key)
    return abi


def _public_query(by_cid: dict[int, dict], cid: Any, operation: str) -> tuple[dict, dict, dict]:
    call = by_cid.get(_int(cid, "actual public-query correlation", 1))
    if call is None or call["operation"] != operation:
        raise ResourceEvidenceError("Public-query correlation does not identify its actual API")
    _success(call, 2)
    _same(call["enter"].get("ancestor"), 0, "public query is outside runtime callbacks")
    field = _HOST_QUERY[operation][1]
    entry, payload = _payload(call, field, "enter"), _payload(call, field)
    for value in (entry, payload):
        _same(value.get("payload_observed"), True, "actual callback public-query payload")
        _int(value.get("output_address" if operation != "hsa_amd_pointer_info" else "info_address"),
             "actual query output address", 1)
    inputs = ("pointer_address", "info_address") if operation == "hsa_amd_pointer_info" else (
        ("agent_handle", "attribute", "output_address") if operation == "hsa_agent_get_info" else ("signal_handle", "output_address"))
    for key in inputs:
        _same(entry.get(key), payload.get(key), "paired raw public-query input " + key)
        _int(payload.get(key), "actual query input " + key)
    return call, entry, payload


def _ordinary_control(native: dict, evidence: dict, by_cid: dict[int, dict], abi: dict) -> None:
    control = evidence.get("ordinary_signal_control")
    if not isinstance(control, dict):
        raise ResourceEvidenceError("Actual ordinary-signal ABI positive control is missing")
    if control.get("method") == "same_process_traced_standard_signal_control":
        _same(control.get("process_id"), native.get("pid"), "ordinary control actual process")
        create = by_cid.get(_int(control.get("create_cid"), "actual ordinary create correlation", 1))
        destroy = by_cid.get(_int(control.get("destroy_cid"), "actual ordinary destroy correlation", 1))
        if create is None or destroy is None or create["operation"] != "hsa_signal_create" or destroy["operation"] != "hsa_signal_destroy":
            raise ResourceEvidenceError("Ordinary control has no actual standard signal lifecycle")
        _success(create, 2)
        _success(destroy, 2)
        for call in (create, destroy):
            _same(call["enter"].get("ancestor"), 0, "ordinary control is outside runtime callbacks")
        payload, ending = _payload(create, "raw_hsa_signal_payload"), _payload(destroy, "raw_hsa_signal_payload")
        handle = _int(payload.get("signal_handle"), "actual ordinary control handle", 1)
        count = _int(payload.get("num_consumers"), "ordinary control actual consumer count")
        _same(payload.get("consumers_observed"), True, "ordinary control actual consumers observed")
        consumers = payload.get("consumer_agents")
        if count > 4 or not isinstance(consumers, list) or len(consumers) != count:
            raise ResourceEvidenceError("Ordinary control actual consumers were not completely read")
        for key in ("initial_value", "num_consumers", "output_address"):
            _same(_payload(create, "raw_hsa_signal_payload", "enter").get(key), payload.get(key),
                  "ordinary control paired actual constructor " + key)
        _same(ending.get("signal_handle"), handle, "ordinary control destroys the same actual signal")
        _same(_payload(destroy, "raw_hsa_signal_payload", "enter").get("signal_handle"), handle, "ordinary control destroy argument")
        samples = control.get("samples")
        if not isinstance(samples, list) or len(samples) != 2:
            raise ResourceEvidenceError("Ordinary control requires two actual public ABI samples")
        for row in samples:
            sample = _cpu_abi_query(row, by_cid, abi, handle, require_no_hw=False)
            if not create["exit"]["seq"] < sample["first_sequence"] <= sample["last_sequence"] < destroy["enter"]["seq"]:
                raise ResourceEvidenceError("Ordinary control queries are not inside its actual signal lifetime")
        return
    for key in ("source", "result_artifact"):
        _artifact(control.get(key), "ordinary-signal control " + key)
    digest = hashlib.sha256(json.dumps(abi, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _same(control.get("abi_sha256"), digest, "ordinary control uses the same compiled ABI/header identity")
    _same(control.get("runtime_dsos"), abi["runtime_dsos"], "ordinary control uses the same actual runtime DSO bytes")
    receipt = control.get("receipt")
    if not isinstance(receipt, dict):
        raise ResourceEvidenceError("Actual ordinary-signal compile/run receipt is missing")
    _same(receipt.get("compile_returncode"), 0, "ordinary control actual compilation")
    _same(receipt.get("run_returncode"), 0, "ordinary control actual exit")
    value = receipt.get("result")
    if not isinstance(value, dict):
        raise ResourceEvidenceError("Ordinary-signal actual public API outputs are missing")
    for key in ("init_status", "create_status", "pointer_info_status", "owner_status", "event_status", "destroy_status"):
        _same(value.get(key), 0, "ordinary control actual HSA result " + key)
    for left, right in {"amd_signal_size": "amd_signal_size", "amd_signal_alignment": "amd_signal_alignment",
        "device_attribute": "device_attribute", "cpu_enum": "cpu_device_enum", "hsa_type_enum": "pointer_type_hsa",
        "locked_type_enum": "pointer_type_locked", "owner_device": "cpu_device_enum"}.items():
        _same(value.get(left), abi[right], "ordinary control actual compiled ABI " + left)
    returned = _int(value.get("pointer_info_size"), "ordinary control actual returned extent", 1)
    if returned > abi["pointer_info_size"] or any(sum(abi["pointer_info_fields"][key]) > returned
            for key in ("type", "host_base_address", "size_bytes", "owner_agent")):
        raise ResourceEvidenceError("Ordinary control pointer-info result is truncated")
    if value.get("pointer_info_type") not in (abi["pointer_type_hsa"], abi["pointer_type_locked"]):
        raise ResourceEvidenceError("Ordinary control did not positively identify known allocation backing")
    handle, base = _int(value.get("signal_handle"), "ordinary control actual signal", 1), _int(value.get("host_base"), "ordinary control host alias", 1)
    size = _int(value.get("size_bytes"), "ordinary control observed region", 1)
    if (handle % abi["amd_signal_alignment"] or base + size > 2**64
            or not base <= handle < handle + abi["amd_signal_size"] <= base + size):
        raise ResourceEvidenceError("Ordinary control host alias does not contain its actual ABI body")
    _int(value.get("owner_handle"), "ordinary control actual preferred-access owner", 1)
    _int(value.get("event_id"), "ordinary control actual event ID")  # A destroyed positive control may be HW-backed.
    if not any(row["path"] == value.get("module") or row.get("loaded_path") == value.get("module") for row in abi["runtime_dsos"]):
        raise ResourceEvidenceError("Ordinary control did not resolve the same actual runtime module")


def _native_cpu_mutations(native: dict) -> list[dict]:
    """Recompute successful normalized storage identities, never infer an internal allocator."""
    opmap = {(i, row["id"]): row["name"] for i, domain in enumerate(native["domains"]) for row in domain["operations"]}
    allocate = {"hsa_memory_allocate", "hsa_amd_memory_pool_allocate"}
    free = {"hsa_memory_free", "hsa_amd_memory_pool_free"}
    normalized = {"MEMORY_ALLOCATION_ALLOCATE", "MEMORY_ALLOCATION_FREE"}
    names = allocate | free | normalized
    pending: dict[tuple, dict] = {}
    pairs: list[dict] = []
    by_cid: dict[int, list[dict]] = defaultdict(list)
    for row in native["events"]:
        operation = opmap[(row["domain"], row["op"])]
        if operation not in names:
            continue
        expected = 6 if operation in normalized else (2 if operation.startswith("hsa_memory_") else 3)
        _same(row["domain"], expected, "actual allocator SDK domain")
        key = row["domain"], row["op"], row["cid"], row["tid"]
        if row["phase"] == 1:
            if key in pending:
                raise ResourceEvidenceError("Storage callback has duplicate entry")
            pending[key] = row
        elif row["phase"] == 2:
            start = pending.pop(key, None)
            if start is None or start["ns"] > row["ns"]:
                raise ResourceEvidenceError("Storage callback has no ordered actual entry")
            call = {"operation": operation, "enter": start, "exit": row}
            (pairs if operation in normalized else by_cid[row["cid"]]).append(call)
        else:
            raise ResourceEvidenceError("Storage callback has unknown phase")
    if pending:
        raise ResourceEvidenceError("Storage API remains open at closed SDK boundary")
    agents = {value["handle"]: value for value in native.get("agent_inventory", {}).get("agents", [])}
    mutations = []
    for call in pairs:
        a, b = call["enter"], call["exit"]
        acquiring = call["operation"] == "MEMORY_ALLOCATION_ALLOCATE"
        _same(a.get("flags"), 1 if acquiring else 2, "normalized storage entry flags")
        _same(b.get("flags"), a["flags"], "normalized storage return flags")
        candidates = [value for value in by_cid[b["cid"]] if value["operation"] in (allocate if acquiring else free)]
        if len(candidates) != 1:
            raise ResourceEvidenceError("Storage backing has no unique successful actual HSA API correlation")
        api = candidates[0]
        _success(api, 2)
        if api["exit"]["tid"] != b["tid"] or not api["enter"]["ns"] <= b["ns"] <= api["exit"]["ns"]:
            raise ResourceEvidenceError("Normalized backing is not inside its same-thread actual HSA API")
        payload = _payload(call, "raw_allocation_payload")
        _same(payload.get("success_proven"), False, "normalized payload does not self-assert API success")
        address = _int(payload.get("address_or_handle"), "actual backing address")
        size, agent = _int(payload.get("size_bytes"), "actual backing bytes"), _int(payload.get("agent_handle"), "actual SDK backing agent")
        if acquiring and (not address or not size or address + size > 2**64 or agent not in agents):
            raise ResourceEvidenceError("Actual successful backing range/agent is invalid")
        mutations.append({"action": "allocate" if acquiring else "free", "address": address, "bytes": size,
            "agent_handle": agent, "sequence": b["seq"], "cid": b["cid"]})
    live: dict[int, dict] = {}
    counts: dict[int, int] = defaultdict(int)
    for row in sorted(mutations, key=lambda value: value["sequence"]):
        address = row["address"]
        if row["action"] == "allocate":
            if address in live or any(value["agent_handle"] == row["agent_handle"]
                    and max(address, value["address"]) < min(address + row["bytes"], value["address"] + value["bytes"])
                    for value in live.values()):
                raise ResourceEvidenceError("Native backing is reused/overlapped without its actual free")
            counts[address] += 1
            row.update(generation=counts[address], storage_id=f"agent{row['agent_handle']}:0x{address:x}:generation{counts[address]}", free_sequence=None)
            live[address] = row
        elif address in live:
            old = live.pop(address)
            old["free_sequence"] = row["sequence"]
    return mutations


def _host_page_range(pointer: int, size: int, page: int) -> tuple[int, int]:
    if pointer + size > 2**64:
        raise ResourceEvidenceError("Actual host-memory extent overflows")
    return pointer // page * page, (pointer + size + page - 1) // page * page


def _stream_owner_lifecycle_calls(native: dict) -> list[dict]:
    """Keep possible stream/context invalidators from the full SDK tape."""
    opmap = {(i, row["id"]): row["name"] for i, domain in enumerate(native["domains"])
             for row in domain["operations"]}
    context = _INVALIDATE | {"hipDeviceReset", "hipDevicePrimaryCtxReset", "hipDevicePrimaryCtxRelease"}
    pending: dict[tuple, dict] = {}
    seen: set[int] = set()
    calls = []
    for row in native["events"]:
        name = opmap[(row["domain"], row["op"])]
        stream = name.startswith(("hipStreamCreate", "hipExtStreamCreate", "hipStreamDestroy"))
        if not stream and name not in context:
            continue
        _same(row["domain"], 0, "actual stream/context lifecycle HIP domain")
        cid, tid = _int(row.get("cid"), "actual stream/context lifecycle correlation", 1), _int(row.get("tid"), "actual stream/context lifecycle thread", 1)
        call_key = row["domain"], row["op"], cid, tid
        if row["phase"] == 1:
            if cid in seen:
                raise ResourceEvidenceError("Stream/context lifecycle correlation was reused")
            seen.add(cid)
            pending[call_key] = row
        elif row["phase"] == 2:
            first = pending.pop(call_key, None)
            if first is None or not (first["seq"] < row["seq"] and first["ns"] <= row["ns"]):
                raise ResourceEvidenceError("Stream/context lifecycle lacks its ordered actual entry")
            for key in ("ancestor", "epoch", "flags"):
                _same(first.get(key), row.get(key), "paired stream/context lifecycle " + key)
            calls.append({"operation": name, "enter": first, "exit": row, "stream_operation": stream})
        else:
            raise ResourceEvidenceError("Stream/context lifecycle has an unknown callback phase")
    if pending:
        raise ResourceEvidenceError("Stream/context lifecycle remains open at the closed SDK boundary")
    return calls


def _hsa_host_memory_calls(native: dict, opmap: dict) -> list[dict]:
    """Read the complete bounded public inputs; an unlock has no size output."""
    operations = {"hsa_amd_memory_lock": "lock", "hsa_amd_memory_lock_to_pool": "lock_to_pool",
                  "hsa_amd_memory_unlock": "unlock", "hsa_amd_agents_allow_access": "allow_access"}
    rows = [row for row in native["events"] if opmap[(row["domain"], row["op"])] in operations
            or "raw_hsa_host_memory_payload" in row]
    if not rows:
        return []
    table = native.get("hsa_host_memory_storage")
    if not isinstance(table, dict):
        raise ResourceEvidenceError("Actual bounded HSA host-memory storage is missing")
    for key, expected in {"schema_version": 1, "method": "bounded_public_hsa_host_memory_payloads_v1",
            "capacity": 8192, "agent_capacity": 4, "retained_records": len(rows), "dropped_records": 0,
            "read_errors": 0, "incomplete_agent_arrays": 0}.items():
        _same(table.get(key), expected, "actual HSA host-memory storage." + key)
    record_size = _int(table.get("record_size_bytes"), "actual HSA host-memory record bytes", 1)
    _same(table.get("storage_capacity_bytes"), 8192 * record_size, "actual HSA host-memory storage bytes")
    if len(rows) > 8192:
        raise ResourceEvidenceError("Actual HSA host-memory payload bound was exceeded")
    if "supported_operations" in table or any(opmap[(row["domain"], row["op"])] == "hsa_amd_agents_allow_access" for row in rows):
        _same(table.get("supported_operations"), ["hsa_amd_memory_unlock", "hsa_amd_memory_lock",
            "hsa_amd_memory_lock_to_pool", "hsa_amd_agents_allow_access"], "actual HSA host-memory supported operations")
    inputs = ("operation", "host_ptr", "size_bytes", "agents_address", "num_agents", "agents_observed",
              "agent_handles", "pool_handle", "flags", "agent_ptr_output_address")
    access_inputs = ("operation", "ptr", "agents_address", "num_agents", "agents_observed", "agent_handles",
                     "flags_address", "flags_values", "flags_observed", "size_bytes", "pool_handle", "flags",
                     "agent_ptr_output_address", "agent_alias")
    pending: dict[tuple, dict] = {}
    calls = []
    seen_cids: set[int] = set()
    for row in rows:
        name = opmap[(row["domain"], row["op"])]
        if name not in operations:
            raise ResourceEvidenceError("HSA host-memory payload belongs to an unrelated operation")
        _same(row["domain"], 3, "actual HSA host-memory SDK domain")
        access = name == "hsa_amd_agents_allow_access"
        _same(row.get("flags"), 0 if access else 32, "actual HSA host-memory surface flags")
        cid, tid = _int(row.get("cid"), "actual HSA host-memory correlation", 1), _int(row.get("tid"), "actual HSA host-memory thread", 1)
        for key in ("seq", "ns"):
            _int(row.get(key), "actual HSA host-memory " + key, 1)
        for key in ("ancestor", "epoch"):
            _int(row.get(key), "actual HSA host-memory " + key)
        payload = row.get("raw_hsa_host_memory_payload")
        if not isinstance(payload, dict):
            raise ResourceEvidenceError("HSA host-memory API lacks its actual public payload")
        actual_inputs = access_inputs if access else inputs
        expected_fields = actual_inputs + (("payload_observed",) if access else ("payload_observed", "agent_alias"))
        if any(key not in payload for key in expected_fields):
            raise ResourceEvidenceError("HSA host-memory payload omits an actual public argument field")
        _same(payload.get("payload_observed"), True, "actual HSA host-memory public input read")
        _same(payload.get("operation"), operations[name], "actual HSA host-memory operation")
        pointer = _int(payload.get("ptr" if access else "host_ptr"), "actual HSA memory input pointer", 1)
        if access:
            count = _int(payload.get("num_agents"), "actual HSA access agent count")
            handles = payload.get("agent_handles")
            if count > 4 or not isinstance(handles, list) or len(handles) != count:
                raise ResourceEvidenceError("Actual HSA access agent array was not completely read")
            _same(payload.get("agents_observed"), True, "actual HSA access complete agent inputs")
            _int(payload.get("agents_address"), "actual HSA access agent-array address", 1 if count else 0)
            for handle in handles:
                _int(handle, "actual HSA access runtime agent handle", 1)
            _same(payload.get("flags_address"), 0, "HSA access reserved flags require an actual null pointer")
            _same(payload.get("flags_values"), None, "HSA access null flags have no invented values")
            _same(payload.get("flags_observed"), True, "HSA access reserved flags were actually observed null")
            for key in ("size_bytes", "pool_handle", "flags", "agent_ptr_output_address", "agent_alias"):
                _same(payload.get(key), None, "HSA access has no actual " + key)
        elif name == "hsa_amd_memory_unlock":
            for key in inputs[2:] + ("agent_alias",):
                _same(payload.get(key), None, "unlock has no actual " + key)
        else:
            length = _int(payload.get("size_bytes"), "actual HSA locked extent", 1)
            if pointer + length > 2**64:
                raise ResourceEvidenceError("Actual HSA locked host extent overflows")
            count = _int(payload.get("num_agents"), "actual HSA lock agent count")
            handles = payload.get("agent_handles")
            if count > 4 or not isinstance(handles, list) or len(handles) != count:
                raise ResourceEvidenceError("Actual HSA lock agent array was not completely read")
            _same(payload.get("agents_observed"), True, "actual HSA lock complete agent inputs")
            _int(payload.get("agents_address"), "actual HSA lock agent-array address", 1 if count else 0)
            for handle in handles:
                _int(handle, "actual runtime HSA agent handle", 1)
            _int(payload.get("agent_ptr_output_address"), "actual HSA lock output address", 1)
            for key in ("pool_handle", "flags"):
                if name == "hsa_amd_memory_lock_to_pool":
                    _int(payload.get(key), "actual HSA lock-to-pool " + key)
                else:
                    _same(payload.get(key), None, "ordinary HSA lock has no " + key)
            if row["phase"] == 2:
                alias = _int(payload.get("agent_alias"), "actual successful HSA device alias", 1)
                if alias + length > 2**64:
                    raise ResourceEvidenceError("Actual HSA device alias extent overflows")
            else:
                _same(payload.get("agent_alias"), None, "HSA lock entry has no output alias")
        call_key = row["domain"], row["op"], cid, tid
        if row["phase"] == 1:
            if call_key in pending or cid in seen_cids:
                raise ResourceEvidenceError("HSA host-memory correlation was reused")
            _same(row.get("status_type"), 0, "HSA host-memory entry has no return type")
            _same(row.get("status"), None, "HSA host-memory entry has no return value")
            seen_cids.add(cid)
            pending[call_key] = row
        elif row["phase"] == 2:
            start = pending.pop(call_key, None)
            if start is None or not (start["seq"] < row["seq"] and start["ns"] <= row["ns"]):
                raise ResourceEvidenceError("HSA host-memory return lacks an ordered actual entry")
            for field in ("ancestor", "epoch", "flags"):
                _same(start.get(field), row.get(field), "paired HSA host-memory " + field)
            before = start["raw_hsa_host_memory_payload"]
            for field in actual_inputs:
                _same(before.get(field), payload.get(field), "paired HSA host-memory input " + field)
            call = {"operation": name, "enter": start, "exit": row}
            _success(call, 2)
            calls.append(call)
        else:
            raise ResourceEvidenceError("HSA host-memory API has an unknown callback phase")
    if pending:
        raise ResourceEvidenceError("HSA host-memory API remains open at the actual SDK stop")
    return sorted(calls, key=lambda call: call["enter"]["seq"])


def _disjoint_host_unregister_sequences(native: dict, base: int, size: int, shutdown: dict | None,
                                        agent_base: int | None = None, *, proofs: list[dict] | None = None,
                                        hsa_calls: list[dict] | None = None) -> set[int]:
    """Prove exact successful host ranges do not touch the queried CPU region.

    This is an address-effect proof only. Repeated HSA locks keep every possible
    extent until their reference count reaches zero; no particular lock or
    physical allocation free is assigned to an unlock.
    """
    names = {"hipHostRegister", "hipHostUnregister"}
    opmap = {(i, row["id"]): row["name"] for i, domain in enumerate(native["domains"])
             for row in domain["operations"]}
    pending: dict[tuple, dict] = {}
    calls = []
    for row in native["events"]:
        name = opmap[(row["domain"], row["op"])]
        if name not in names:
            continue
        _same(row["domain"], 0, "host registration actual HIP domain")
        key = row["cid"], row["tid"], name
        if row["phase"] == 1:
            if key in pending:
                raise ResourceEvidenceError("Host registration has a duplicate actual entry")
            pending[key] = row
        elif row["phase"] == 2:
            first = pending.pop(key, None)
            if first is None or not first["ns"] <= row["ns"]:
                raise ResourceEvidenceError("Host registration lacks an ordered actual entry")
            for field in ("ancestor", "epoch", "flags"):
                _same(first.get(field), row.get(field), "paired host registration " + field)
            call = {"operation": name, "enter": first, "exit": row}
            _success(call)
            a, b = _payload(call, phase="enter"), _payload(call)
            _same(a, b, "paired actual host-registration arguments")
            _same(b.get("kind"), "host_registration" if name == "hipHostRegister" else "host_unregistration",
                  "actual host-registration effect")
            _same(first.get("flags"), 32, "actual host registration surface flags")
            _int(b.get("pointer"), "actual host registration pointer", 1)
            calls.append(call)
        else:
            raise ResourceEvidenceError("Host registration has an unknown callback phase")
    if pending:
        raise ResourceEvidenceError("Host registration remains unpaired at SDK stop")
    live: dict[int, dict] = {}
    allowed: set[int] = set()
    registrations: dict[int, dict] = {}
    cleanup: dict[int, dict] = {}
    page = None

    def identity(call: dict) -> dict:
        a, b = call["enter"], call["exit"]
        return {"operation": call["operation"], "cid": a["cid"], "ancestor_cid": a["ancestor"],
            "tid": a["tid"], "epoch": a["epoch"], "enter_sequence": a["seq"], "exit_sequence": b["seq"],
            "enter_monotonic_raw_ns": a["ns"], "exit_monotonic_raw_ns": b["ns"]}

    for call in sorted(calls, key=lambda value: value["enter"]["seq"]):
        a, b = call["enter"], call["exit"]
        payload = _payload(call)
        pointer = payload["pointer"]
        if call["operation"] == "hipHostRegister":
            length = _int(payload.get("size_bytes"), "actual registered host extent", 1)
            if pointer + length > 2**64 or pointer in live:
                raise ResourceEvidenceError("Host registration range overflows or reuses a live registration")
            live[pointer] = call
            if a["cid"] in registrations:
                raise ResourceEvidenceError("Host registration correlation was reused")
            registrations[a["cid"]] = call
            continue
        registered = live.pop(pointer, None)
        if registered is None or not registered["exit"]["seq"] < a["seq"]:
            raise ResourceEvidenceError("Host unregister has no unique prior live registered range")
        length = _payload(registered)["size_bytes"]
        page = _int(native.get("hsa_host_memory_storage", {}).get("system_page_size"),
                    "actual host-registration system page size", 1)
        if page & (page - 1) or page > 2**30:
            raise ResourceEvidenceError("Actual host-registration page granule is unsupported")
        region_start, region_end = _host_page_range(base, size, page)
        host_start, host_end = _host_page_range(pointer, length, page)
        if max(region_start, host_start) < min(region_end, host_end):
            continue  # The ordinary region guard will reject this overlapping operation.
        if shutdown is None:
            continue
        _same(shutdown.get("process_id"), native.get("pid"), "host cleanup original shutdown process")
        if not (a["epoch"] == b["epoch"] == 0
                and shutdown["started_monotonic_raw_ns"] <= a["ns"] <= b["ns"]
                    <= shutdown["ended_monotonic_raw_ns"]):
            continue
        allowed.update((a["seq"], b["seq"], registered["enter"]["seq"], registered["exit"]["seq"]))
        if a["cid"] in cleanup:
            raise ResourceEvidenceError("Host unregistration correlation was reused")
        cleanup[a["cid"]] = {"call": call, "registered": registered}
        if proofs is not None:
            proofs.append({"kind": "HIP_HOST_RANGE_EXCLUSION", "registration": identity(registered),
                "unregistration": identity(call), "host_ptr": pointer, "size_bytes": length,
                "system_page_size": page, "host_page_range": [host_start, host_end],
                "cpu_query_page_range": [region_start, region_end]})
    if hsa_calls is None:
        hsa_calls = _hsa_host_memory_calls(native, opmap)
    candidates: dict[int, list[dict]] = defaultdict(list)
    remaining: dict[int, int] = defaultdict(int)
    staged: dict[int, set[int]] = defaultdict(set)
    staged_unlocks: dict[int, list[dict]] = defaultdict(list)

    def parent_of(call: dict, parents: dict[int, dict]) -> dict | None:
        a, b = call["enter"], call["exit"]
        parent = parents.get(a["ancestor"])
        if parent is None:
            return None
        p, q = parent["enter"], parent["exit"]
        if not (p["tid"] == a["tid"] == b["tid"] == q["tid"] and p["epoch"] == a["epoch"] == q["epoch"]
                and p["seq"] < a["seq"] < b["seq"] < q["seq"]
                and p["ns"] <= a["ns"] <= b["ns"] <= q["ns"]):
            return None
        return parent

    for call in hsa_calls:
        if call["operation"] == "hsa_amd_agents_allow_access":
            continue  # Its allocation scope is independently checked below.
        a, b = call["enter"], call["exit"]
        payload = b["raw_hsa_host_memory_payload"]
        pointer = payload["host_ptr"]
        if call["operation"] != "hsa_amd_memory_unlock":
            if staged[pointer]:
                raise ResourceEvidenceError("A new HSA lock crosses an incompletely observed unlock group")
            candidates[pointer].append(call)
            remaining[pointer] += 1
            continue
        ending = cleanup.get(a["ancestor"])
        parents = {} if ending is None else {a["ancestor"]: ending["call"]}
        parent = parent_of(call, parents)
        locks = candidates[pointer]
        if ending is None or parent is None or not locks or not remaining[pointer]:
            continue  # Unknown or source-unbound unlock remains subject to the ordinary region guard.
        registered = ending["registered"]
        if not isinstance(registered, dict):
            raise ResourceEvidenceError("HSA unlock has no actual HIP registration record")
        if page is None:
            raise ResourceEvidenceError("HSA unlock has no actual host-memory page granule")
        region = _host_page_range(base, size, page)
        host = _payload(registered)
        host_range = _host_page_range(host["pointer"], host["size_bytes"], page)
        for lock in locks:
            value = lock["exit"]["raw_hsa_host_memory_payload"]
            owner = parent_of(lock, registrations)
            if owner is not registered or lock["exit"]["seq"] >= a["seq"]:
                raise ResourceEvidenceError("HSA unlock lacks its independently observed HIP registration ancestry")
            extent = _host_page_range(value["host_ptr"], value["size_bytes"], page)
            if not host_range[0] <= extent[0] < extent[1] <= host_range[1]:
                raise ResourceEvidenceError("Actual HSA lock extent exceeds its parent HIP registered page range")
            # Runtime HSA agent handles are not SDK agent IDs. An arbitrary GPU
            # alias cannot be compared to a CPU virtual address by integer value.
            # Restrict this address-effect proof to observed identity mappings.
            if agent_base != base or value["agent_alias"] != value["host_ptr"]:
                raise ResourceEvidenceError("HSA host cleanup has no observed common identity-mapped VA namespace")
            if max(region[0], extent[0]) < min(region[1], extent[1]):
                raise ResourceEvidenceError("Actual HSA host range shares the live CPU query pages")
            staged[pointer].update((lock["enter"]["seq"], lock["exit"]["seq"]))
        staged[pointer].update((a["seq"], b["seq"]))
        staged_unlocks[pointer].append(identity(call))
        remaining[pointer] -= 1
        if not remaining[pointer]:
            allowed.update(staged.pop(pointer))
            if proofs is not None:
                proofs.append({"kind": "HSA_HOST_INPUT_REFERENCE_GROUP_EXCLUSION", "host_ptr": pointer,
                    "registration": identity(registered), "unregistration": identity(parent),
                    "candidate_locks": [{**identity(lock), "raw_hsa_host_memory_payload":
                            dict(lock["exit"]["raw_hsa_host_memory_payload"]),
                        "host_page_range": list(_host_page_range(pointer,
                            lock["exit"]["raw_hsa_host_memory_payload"]["size_bytes"], page))}
                        for lock in locks], "unlocks": staged_unlocks[pointer], "reference_count": len(locks),
                    "cpu_query_page_range": list(region), "system_page_size": page,
                    "claim": "all candidate extents are disjoint; individual unlock-to-lock or physical free is unknown"})
            staged_unlocks.pop(pointer)
            candidates.pop(pointer)
    return allowed


def _disjoint_host_allocation_sequences(native: dict, base: int, size: int, shutdown: dict | None,
                                        mutations: list[dict], access_calls: list[dict],
                                        proofs: list[dict]) -> set[int]:
    """Exclude only actual HIP host generations with completely covered SDK effects."""
    hip = {"hipHostMalloc", "hipHostFree"}
    allocate = {"hsa_memory_allocate", "hsa_amd_memory_pool_allocate"}
    release = {"hsa_memory_free", "hsa_amd_memory_pool_free"}
    normalized = {"MEMORY_ALLOCATION_ALLOCATE", "MEMORY_ALLOCATION_FREE"}
    opmap = {(i, row["id"]): row["name"] for i, domain in enumerate(native["domains"])
             for row in domain["operations"]}
    hip_rows = [row for row in native["events"] if opmap[(row["domain"], row["op"])] in hip]
    if not hip_rows:
        return set()
    table = native.get("hip_host_memory_payloads", {})
    if not isinstance(table, dict):
        raise ResourceEvidenceError("Actual HIP host-memory payload metadata is missing")
    for key, expected in {"schema_version": 1, "method": "existing_native_event_storage_v1",
            "supported_operations": ["hipHostMalloc", "hipHostFree"], "retained_records": len(hip_rows),
            "safe_read_errors": 0, "uses_native_event_storage": True}.items():
        _same(table.get(key), expected, "actual HIP host-memory payload metadata." + key)
    pending: dict[tuple, dict] = {}
    seen_calls: set[tuple] = set()
    calls = []
    for row in native["events"]:
        name = opmap[(row["domain"], row["op"])]
        if name not in hip | allocate | release | normalized:
            continue
        expected = 0 if name in hip else 6 if name in normalized else 2 if name.startswith("hsa_memory_") else 3
        _same(row["domain"], expected, "actual HIP host allocation/effect domain")
        for key in ("cid", "tid", "seq", "ns"):
            _int(row.get(key), "actual HIP host allocation/effect " + key, 1)
        for key in ("ancestor", "epoch"):
            _int(row.get(key), "actual HIP host allocation/effect " + key)
        call_key = row["domain"], row["op"], row["cid"], row["tid"]
        if row["phase"] == 1:
            identity = row["domain"], row["cid"]
            if call_key in pending or identity in seen_calls:
                raise ResourceEvidenceError("Host allocation/effect has a duplicate actual entry")
            seen_calls.add(identity)
            pending[call_key] = row
        elif row["phase"] == 2:
            first = pending.pop(call_key, None)
            if first is None or not (first["seq"] < row["seq"] and first["ns"] <= row["ns"]):
                raise ResourceEvidenceError("Host allocation/effect lacks an ordered actual entry")
            for key in ("ancestor", "epoch", "flags"):
                _same(first.get(key), row.get(key), "paired host allocation/effect " + key)
            calls.append({"operation": name, "enter": first, "exit": row})
        else:
            raise ResourceEvidenceError("Host allocation/effect has an unknown callback phase")
    if pending:
        raise ResourceEvidenceError("Host allocation/effect remains open at SDK stop")
    host_calls = sorted((call for call in calls if call["operation"] in hip), key=lambda call: call["enter"]["seq"])
    live: dict[int, dict] = {}
    allowed: set[int] = set()
    seen_cids: set[int] = set()
    agents = {row["handle"]: row for row in native.get("agent_inventory", {}).get("agents", [])}

    def contains(parent: dict, child: dict) -> bool:
        a, b, c, d = parent["enter"], parent["exit"], child["enter"], child["exit"]
        return (a["tid"] == b["tid"] == c["tid"] == d["tid"] and a["epoch"] == b["epoch"] == c["epoch"] == d["epoch"]
            and a["seq"] < c["seq"] < d["seq"] < b["seq"] and a["ns"] <= c["ns"] <= d["ns"] <= b["ns"])

    def effect(parent: dict, acquiring: bool, pointer: int, length: int) -> tuple[dict, dict, set[int]]:
        a, b = parent["enter"], parent["exit"]
        apis = [call for call in calls if call["operation"] in (allocate if acquiring else release)
            and call["enter"]["ancestor"] == a["cid"] and contains(parent, call)]
        if len(apis) != 1:
            raise ResourceEvidenceError("HIP host generation lacks a unique actual contained HSA storage effect")
        api = apis[0]
        _success(api, 2)
        _same(api["enter"].get("flags"), (1 if acquiring else 2) | (16 if "_pool_" in api["operation"] else 0),
              "actual HSA host storage surface flags")
        _same(api["exit"].get("ancestor"), a["cid"], "actual HSA effect HIP parent")
        norms = [call for call in calls if call["operation"] == ("MEMORY_ALLOCATION_ALLOCATE" if acquiring else "MEMORY_ALLOCATION_FREE")
            and call["enter"]["cid"] == api["enter"]["cid"] and contains(api, call)]
        if len(norms) != 1:
            raise ResourceEvidenceError("HIP host effect lacks a unique actual same-correlation normalized storage pair")
        norm = norms[0]
        for row in (norm["enter"], norm["exit"]):
            _same(row.get("ancestor"), a["cid"], "actual normalized storage HIP parent")
            _same(row.get("flags"), 1 if acquiring else 2, "actual normalized host storage effect flags")
            _same(row.get("status_type"), 0, "normalized storage does not claim a typed API status")
            _same(row.get("status"), None, "normalized storage does not invent API success")
        before, after = _payload(norm, "raw_allocation_payload", "enter"), _payload(norm, "raw_allocation_payload")
        for value in (before, after):
            _same(value.get("success_proven"), False, "normalized storage requires independent HSA success")
        _same(after.get("address_or_handle"), pointer, "actual HIP/native storage base identity")
        if acquiring:
            _same(before.get("address_or_handle"), 0, "normalized allocation entry has no returned address")
            _same(before.get("size_bytes"), after.get("size_bytes"), "actual normalized allocation requested extent")
            _same(before.get("agent_handle"), after.get("agent_handle"), "actual normalized allocation SDK agent")
            candidates = [row for row in mutations if row["action"] == "allocate" and row["cid"] == api["enter"]["cid"]
                and row["sequence"] == norm["exit"]["seq"] and row["address"] == pointer and row["bytes"] >= length]
            if len(candidates) != 1:
                raise ResourceEvidenceError("HIP host range has no unique complete actual native allocation extent")
            allocation = candidates[0]
            agent = agents.get(allocation["agent_handle"], {})
            _same(agent.get("type"), 1, "independent actual SDK host allocation CPU type")
            _same(agent.get("type_name"), "CPU", "independent actual SDK host allocation CPU agent")
        else:
            _same(before.get("address_or_handle"), pointer, "paired actual normalized free pointer")
            for value in (before, after):
                _same(value.get("size_bytes"), 0, "normalized free has no invented extent")
                _same(value.get("agent_handle"), 0, "normalized free has no invented SDK agent")
            candidates = [row for row in mutations if row["action"] == "allocate" and row["address"] == pointer
                and row.get("free_sequence") == norm["exit"]["seq"]]
            if len(candidates) != 1:
                raise ResourceEvidenceError("HIP host free has no unique previously live native allocation generation")
            allocation = candidates[0]
        sequences = {row["seq"] for call in (api, norm) for row in (call["enter"], call["exit"])}
        access_proofs = []
        if acquiring:
            accesses = [call for call in access_calls if call["enter"]["ancestor"] == a["cid"] and contains(parent, call)]
            for access in accesses:
                value = _payload(access, "raw_hsa_host_memory_payload")
                _same(value.get("ptr"), pointer, "actual HSA access scope is the returned HIP host allocation")
                if access["enter"]["seq"] <= api["exit"]["seq"]:
                    raise ResourceEvidenceError("HSA access precedes its independently observed host allocation")
                sequences.update((access["enter"]["seq"], access["exit"]["seq"]))
                access_proofs.append({**_proof(access, allocation["storage_id"]), "raw_hsa_host_memory_payload": dict(value)})
        # _calls already checks the complete 1-based raw sequence, so this is
        # the exact callback interval rather than a process-wide rescan.
        nested = {row["seq"] for row in native["events"][a["seq"]:b["seq"] - 1] if row["tid"] == a["tid"]}
        if nested != sequences:
            raise ResourceEvidenceError("HIP host allocation/free contains an uncovered actual nested effect")
        return allocation, {"hsa_api": _proof(api, allocation["storage_id"]),
            "normalized_storage": _proof(norm, allocation["storage_id"]), "access_calls": access_proofs}, sequences

    for call in host_calls:
        a, b = call["enter"], call["exit"]
        if a["cid"] in seen_cids:
            raise ResourceEvidenceError("HIP host allocation/free correlation was reused")
        seen_cids.add(a["cid"])
        _success(call)
        _same(a.get("status_type"), 0, "HIP host entry has no typed return")
        _same(a.get("status"), None, "HIP host entry has no return value")
        entry, output = _payload(call, phase="enter"), _payload(call)
        acquiring = call["operation"] == "hipHostMalloc"
        for value in (entry, output):
            if any(key not in value for key in ("kind", "pointer", "pointer_observed", "output_address", "size_bytes", "flags", "payload_observed")):
                raise ResourceEvidenceError("HIP host allocation/free omits an actual argument/output field")
            _same(value.get("payload_observed"), True, "actual HIP host payload read")
            _same(value.get("kind"), "host_allocation" if acquiring else "host_free", "actual HIP host operation kind")
        _same(a.get("flags"), 33 if acquiring else 34, "actual HIP host allocation/free flags")
        pointer = _int(output.get("pointer"), "actual HIP host pointer", 1)
        _same(output.get("pointer_observed"), True, "actual HIP host output/input pointer observation")
        if acquiring:
            _same(entry.get("pointer"), None, "HIP allocation entry has no returned pointer")
            _same(entry.get("pointer_observed"), False, "HIP allocation entry has no observed output")
            for key in ("output_address", "size_bytes", "flags"):
                _same(entry.get(key), output.get(key), "paired HIP host allocation input " + key)
                _int(output.get(key), "actual HIP host allocation " + key, 0 if key == "flags" else 1)
            length = output["size_bytes"]
            if pointer + length > 2**64 or any(max(pointer, old["pointer"]) < min(pointer + length,
                    old["pointer"] + old["length"]) for old in live.values()):
                raise ResourceEvidenceError("HIP host allocation reuses/overlaps a live host generation")
            allocation, detail, sequences = effect(call, True, pointer, length)
            live[pointer] = {"call": call, "pointer": pointer, "length": length,
                "allocation": allocation, "detail": detail, "sequences": sequences}
            continue
        for value in (entry, output):
            _same(value.get("pointer"), pointer, "paired actual HIP host free pointer")
            _same(value.get("pointer_observed"), True, "actual HIP host free argument observation")
            _same(value.get("output_address"), 0, "HIP host free has no output argument")
            for key in ("size_bytes", "flags"):
                _same(value.get(key), None, "HIP host free has no actual " + key)
        acquired = live.pop(pointer, None)
        if acquired is None or acquired["call"]["exit"]["seq"] >= a["seq"]:
            raise ResourceEvidenceError("HIP host free lacks its unique prior live actual HIP allocation")
        allocation, detail, sequences = effect(call, False, pointer, acquired["length"])
        if allocation is not acquired["allocation"]:
            raise ResourceEvidenceError("HIP host allocation/free crosses an independently observed native generation")
        if shutdown is None:
            continue
        _same(shutdown.get("process_id"), native.get("pid"), "host allocation cleanup original shutdown process")
        if not (a["epoch"] == b["epoch"] == 0 and shutdown["started_monotonic_raw_ns"] <= a["ns"] <= b["ns"]
                <= shutdown["ended_monotonic_raw_ns"]):
            continue
        page = _int(native.get("hsa_host_memory_storage", {}).get("system_page_size"), "actual HIP host allocation page size", 1)
        if page & (page - 1) or page > 2**30:
            raise ResourceEvidenceError("Actual HIP host allocation page granule is unsupported")
        region = _host_page_range(base, size, page)
        ranges = [_host_page_range(pointer, acquired["length"], page), _host_page_range(pointer, allocation["bytes"], page)]
        if any(max(region[0], extent[0]) < min(region[1], extent[1]) for extent in ranges):
            continue
        allowed.update(acquired["sequences"] | sequences | {row["seq"] for value in (acquired["call"], call)
            for row in (value["enter"], value["exit"])})
        generation = "hip-host-allocation-c" + str(acquired["call"]["enter"]["cid"])
        proofs.append({"kind": "HIP_HOST_ALLOCATION_RANGE_EXCLUSION", "hip_generation_id": generation,
            "host_ptr": pointer, "requested_size_bytes": acquired["length"], "flags": _payload(acquired["call"])["flags"],
            "allocation": _proof(acquired["call"], generation), "free": _proof(call, generation),
            "native_storage_id": allocation["storage_id"], "sdk_cpu_agent": allocation["agent_handle"],
            "actual_native_size_bytes": allocation["bytes"], "allocation_effects": acquired["detail"],
            "free_effects": detail, "system_page_size": page, "host_page_ranges": [list(value) for value in ranges],
            "cpu_query_page_range": list(region), "claim": "this separate host generation is disjoint; no CPU query-region allocation lifetime is inferred"})
    return allowed


def _cpu_backing_identity(native: dict, locator: Any, sample: dict, create_sequence: int,
                          shutdown: dict | None = None) -> dict:
    base, size = sample["host_base_address"], sample["bytes"]
    mutations = _native_cpu_mutations(native)
    for row in mutations:
        if row["sequence"] <= create_sequence:
            continue
        if row["action"] == "allocate" and max(base, row["address"]) < min(base + size, row["address"] + row["bytes"]):
            raise ResourceEvidenceError("Observed backing allocation overlaps an already-live signal ABI region")
        if row["action"] == "free" and base <= row["address"] < base + size:
            raise ResourceEvidenceError("A still-live signal ABI region has an actual storage free")
    opmap = {(i, row["id"]): row["name"] for i, domain in enumerate(native["domains"]) for row in domain["operations"]}
    host_cleanup_proofs: list[dict] = []
    host_memory_calls = _hsa_host_memory_calls(native, opmap)
    disjoint_host_cleanup = _disjoint_host_unregister_sequences(native, base, size, shutdown,
        sample.get("agent_base_address"), proofs=host_cleanup_proofs, hsa_calls=host_memory_calls)
    disjoint_host_cleanup.update(_disjoint_host_allocation_sequences(native, base, size, shutdown, mutations,
        [call for call in host_memory_calls if call["operation"] == "hsa_amd_agents_allow_access"], host_cleanup_proofs))
    # No source-unbound address-changing operation may be waved away because it
    # is outside a model epoch. Query-region identity is not allocation lifetime.
    for row in native["events"]:
        if row["seq"] <= create_sequence:
            continue
        if row["seq"] in disjoint_host_cleanup:
            continue
        name = opmap[(row["domain"], row["op"])]
        lower = name.lower()
        if (row.get("flags", 0) & (4 | 8 | 32 | 256) or name in {"hipDeviceReset", "hipCtxDestroy",
                "hipDevicePrimaryCtxReset", "hipDevicePrimaryCtxRelease"}
                or any(token in lower for token in ("migrate", "svm", "unmap", "memadvise", "setaccess", "memory_unlock", "agents_allow_access", "shut_down"))):
            raise ResourceEvidenceError("Unresolved storage migration/unmap/rebind/lifecycle may affect live CPU query region: " + name)
    cleanup_scope = {} if not disjoint_host_cleanup else {"host_cleanup_noninterference": {
        "method": "actual_HIP_host_ranges_and_complete_independent_native_effects",
        "allowed_callback_sequences": sorted(disjoint_host_cleanup),
        "exclusions": host_cleanup_proofs,
        "hsa_alias_support": "HSA lock/unlock: observed host_ptr=agent_alias and query host_base=agent_base identity mappings only",
        "host_allocation_support": "actual HIP host generations with complete CPU SDK storage and same-allocation HSA access effects",
        "claim": "queried CPU pages are disjoint; no physical unlock/free or allocation generation is assigned to the CPU query region"}}
    if locator is None:
        return {"kind": "CPU_QUERY_REGION", "storage_id": f"cpu-query-region:0x{base:x}:bytes{size}",
            "address": base, "bytes": size, "owner_agent": sample["owner_agent"],
            "allocation_generation": None, "born_sequence": None, "free_sequence": None,
            "lifetime_claim": "positive CPU preferred-access range observation; allocation birth/free are unknown",
            **cleanup_scope}
    if not isinstance(locator, dict):
        raise ResourceEvidenceError("Tracked CPU backing locator is invalid")
    candidates = [row for row in mutations if row["action"] == "allocate"
        and row["cid"] == locator.get("allocation_cid") and row["sequence"] == locator.get("normalized_allocate_exit_sequence")]
    if len(candidates) != 1:
        raise ResourceEvidenceError("CPU backing locator has no unique successful native allocation")
    allocation = candidates[0]
    agent: dict = next((value for value in native.get("agent_inventory", {}).get("agents", [])
                  if value.get("handle") == allocation["agent_handle"]), {})
    _same(agent.get("type_name"), "CPU", "actual SDK backing agent is independently CPU")
    _same(agent.get("type"), 1, "actual SDK CPU agent type")
    _same(allocation["address"], base, "exact actual CPU backing host range")
    _same(allocation["bytes"], size, "exact actual CPU backing byte range")
    if allocation["sequence"] > create_sequence or allocation["free_sequence"] is not None:
        raise ResourceEvidenceError("Actual CPU backing generation is not continuously live with the resource")
    return {"kind": "TRACKED_CPU_ALLOCATION", "storage_id": allocation["storage_id"], "address": base,
        "bytes": size, "owner_agent": sample["owner_agent"], "sdk_agent_handle": allocation["agent_handle"],
        "allocation_generation": allocation["generation"], "born_sequence": allocation["sequence"],
        "free_sequence": None, "allocation_cid": allocation["cid"], **cleanup_scope}


def _cpu_abi_query(record: dict, by_cid: dict[int, dict], abi: dict, handle: int, *, require_no_hw: bool = True) -> dict:
    pointer, entry, payload = _public_query(by_cid, record.get("pointer_info_cid"), "hsa_amd_pointer_info")
    owner, _, owner_payload = _public_query(by_cid, record.get("owner_device_cid"), "hsa_agent_get_info")
    event, _, event_payload = _public_query(by_cid, record.get("event_id_cid"), "hsa_amd_signal_get_event_id")
    address = handle & abi["signal_handle_address_mask"]
    if not address or address % abi["amd_signal_alignment"] or address + abi["amd_signal_size"] > 2**64:
        raise ResourceEvidenceError("Public signal ABI address/span is invalid or overflows")
    _same(payload.get("pointer_address"), address, "query uses actual public-ABI signal address")
    _same(entry.get("input_size"), abi["pointer_info_size"], "actual pointer-info caller structure extent")
    _same(payload.get("input_size"), None, "pointer-info return retains no guessed caller size")
    returned = _int(payload.get("returned_size"), "actual pointer-info returned extent", 1)
    if returned > abi["pointer_info_size"] or any(sum(abi["pointer_info_fields"][key]) > returned
            for key in ("type", "agent_base_address", "host_base_address", "size_bytes", "owner_agent")):
        raise ResourceEvidenceError("Actual pointer-info return truncates ownership/range fields")
    if type(payload.get("type")) is not int or payload["type"] not in (abi["pointer_type_hsa"], abi["pointer_type_locked"]):
        raise ResourceEvidenceError("Signal ABI address has no positive known HSA/LOCKED allocation type")
    base = _int(payload.get("host_base_address"), "actual CPU host alias", 1)
    agent_base = _int(payload.get("agent_base_address"), "actual public query agent alias", 1)
    size = _int(payload.get("size_bytes"), "actual CPU region size", 1)
    if (base + size > 2**64 or agent_base + size > 2**64
            or not base <= address < address + abi["amd_signal_size"] <= base + size):
        raise ResourceEvidenceError("Actual host allocation does not contain the full public signal ABI span")
    _same(owner_payload.get("agent_handle"), payload.get("owner_agent"), "queried actual preferred-access owner")
    _int(payload.get("owner_agent"), "actual HSA preferred-access owner", 1)
    _same(owner_payload.get("attribute"), abi["device_attribute"], "compiled HSA device attribute")
    _same(owner_payload.get("device_type"), abi["cpu_device_enum"], "actual CPU preferred-access owner")
    _same(event_payload.get("signal_handle"), handle, "actual queried live signal")
    _int(event_payload.get("event_id"), "actual hardware-event query result")
    if require_no_hw:
        _same(event_payload.get("event_id"), 0, "signal is not hardware-event backed")
    if not pointer["exit"]["seq"] < owner["enter"]["seq"]:
        raise ResourceEvidenceError("Owner query does not follow the actual pointer-info result")
    if len({value["enter"]["tid"] for value in (pointer, owner, event)}) != 1:
        raise ResourceEvidenceError("Callback-outside query sample crosses unbound provider threads")
    return {"address": address, "abi_span_bytes": abi["amd_signal_size"], "host_base_address": base,
        "agent_base_address": agent_base,
        "bytes": size, "owner_agent": payload["owner_agent"], "pointer_type": payload["type"],
        "event_id": event_payload["event_id"],
        "first_sequence": min(value["enter"]["seq"] for value in (pointer, owner, event)),
        "last_sequence": max(value["exit"]["seq"] for value in (pointer, owner, event)),
        "calls": [_proof(value, "public-query") for value in (pointer, owner, event)]}


def _live_export(value: Any, native: dict, label: str) -> dict:
    if not isinstance(value, dict):
        raise ResourceEvidenceError(label + ": actual live-signal export is missing")
    for key, expected in {"schema_version": 1, "kind": "observed_live_hsa_signal_generations",
        "pid": native["pid"], "complete_workspace": False, "hardware_validated": False,
        "dropped_events": 0, "reused_live_handle_count": 0}.items():
        _same(value.get(key), expected, label + "." + key)
    _int(value.get("snapshot_monotonic_raw_ns"), label + " actual clock", 1)
    _int(value.get("event_cursor"), label + " actual cursor")
    rows = value.get("generations")
    if not isinstance(rows, list):
        raise ResourceEvidenceError(label + ": live generation list is missing")
    found = {}
    for row in rows:
        handle = _int(row.get("signal_handle"), "actual exported signal", 1)
        if handle in found:
            raise ResourceEvidenceError("Live export repeats a signal handle")
        found[handle] = row
    return found


def _live_host_signal(native: dict, evidence: dict, generation: dict, generations: list[dict],
                      by_cid: dict[int, dict], fence: dict, load_kfd: Callable[[], dict],
                      shutdown: dict | None = None) -> dict:
    abi = _host_manifest(native, evidence)
    before, after = evidence.get("before_live"), evidence.get("after_live")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ResourceEvidenceError("Stable actual live exports are missing")
    live_a, live_b = _live_export(before, native, "before query"), _live_export(after, native, "after query")
    _same(live_a, live_b, "stable live-signal generation set during public queries")
    _same(before.get("unknown_destroy_count"), after.get("unknown_destroy_count"), "no unknown destroy during query")
    if not before["event_cursor"] <= after["event_cursor"] <= len(native["events"]):
        raise ResourceEvidenceError("Public queries are not inside the closed native callback window")
    create, handle = generation["create"], generation["signal_handle"]
    counter = sum(value["signal_handle"] == handle and value["create"]["exit"]["seq"] <= create["exit"]["seq"]
                  for value in generations)
    row = live_a.get(handle)
    expected = {"signal_handle": handle, "generation": counter, "create_cid": create["enter"]["cid"],
        "create_seq": create["exit"]["seq"], "create_epoch": create["enter"]["epoch"],
        "create_tid": create["enter"]["tid"], "initial_value": generation["initial_value"],
        "num_consumers": generation["num_consumers"], "consumers_observed": True,
        "consumer_agents": generation["consumer_agents"]}
    _same(row, expected, "actual signal logical generation identity")
    records = evidence.get("records")
    if not isinstance(records, list):
        raise ResourceEvidenceError("Per-generation public query records are missing")
    records = [value for value in records if value.get("signal_handle") == handle
               and value.get("create_cid") == create["enter"]["cid"]]
    if len(records) != 1:
        raise ResourceEvidenceError("Live signal has no unique actual per-generation query record")
    record = records[0]
    for key in ("signal_handle", "generation", "create_cid", "create_seq"):
        _same(record.get(key), expected[key], "query record generation " + key)
    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) != 2:
        raise ResourceEvidenceError("Live resource requires two stable positive public-ABI query samples")
    first, last = [_cpu_abi_query(value, by_cid, abi, handle, require_no_hw=False) for value in samples]
    for key in ("address", "abi_span_bytes", "host_base_address", "agent_base_address", "bytes", "owner_agent", "pointer_type", "event_id"):
        _same(first[key], last[key], "stable actual CPU ABI backing " + key)
    if not (fence["exit"]["seq"] <= before["event_cursor"] < first["first_sequence"]
            <= first["last_sequence"] < last["first_sequence"] <= last["last_sequence"] <= after["event_cursor"]):
        raise ResourceEvidenceError("Public CPU queries do not follow the last GPU fence inside stable live exports")
    if not before["snapshot_monotonic_raw_ns"] <= after["snapshot_monotonic_raw_ns"]:
        raise ResourceEvidenceError("Actual live export clocks are reversed")
    stop_ns = _int(native["stop_evidence"].get("begin_ns"), "actual SDK stop entry clock", 1)
    if not after["snapshot_monotonic_raw_ns"] < stop_ns:
        raise ResourceEvidenceError("Stable live resource query window is not before actual SDK stop")
    for value in [*first["calls"], *last["calls"]]:
        call = by_cid[value["cid"]]
        if not before["snapshot_monotonic_raw_ns"] <= call["enter"]["ns"] <= call["exit"]["ns"] <= after["snapshot_monotonic_raw_ns"]:
            raise ResourceEvidenceError("Actual public query clocks are outside the stable live-export window")
    _ordinary_control(native, evidence, by_cid, abi)
    backing = _cpu_backing_identity(native, record.get("backing"), first, create["exit"]["seq"], shutdown)
    hardware = None
    if first["event_id"]:
        kfd = load_kfd()  # Always recompute from the original typed tapes, never a producer verdict.
        proofs = []
        for sample in samples:
            matches = [value for value in kfd["event_proofs"] if value.get("event_query_cid") == sample["event_id_cid"]
                and value.get("create_cid") == create["enter"]["cid"] and value.get("signal_handle") == handle
                and value.get("event_id") == first["event_id"]]
            if len(matches) != 1:
                raise ResourceEvidenceError("Live signal has no unique actual same-generation KFD GTT event-page proof at both queries")
            proofs.append(matches[0])
        hardware = {"kind": "PUBLIC_HARDWARE_EVENT_WITH_OBSERVED_GTT_EVENT_PAGE", "event_id": first["event_id"],
            "query_proofs": proofs, "lifetime_claim": "actual event-page backing at the two public queries; no logical signal destroy is claimed"}
    return {"method": "positive_public_HSA_query_CPU_preferred_access_ABI_body",
        "signal_generation": expected, "abi_address": first["address"], "abi_span_bytes": first["abi_span_bytes"],
        "samples": [first, last], "backing": backing, "hardware_event_backing": hardware,
        "limitations": ["CPU preferred-access owner is not a statement of zero physical VRAM use",
            "The public ABI body is not the full private runtime object or host synchronization cost",
            "The signal remains live; no destroy or backing release is claimed",
            "Every other GPU allocation/pool/mapping/scratch remains subject to the independent native gate"]}


def _record_cpu_region(result: dict, region: dict) -> None:
    for index, existing in enumerate(result["observed_cpu_backing_regions"]):
        if max(existing["address"], region["address"]) >= min(existing["address"] + existing["bytes"], region["address"] + region["bytes"]):
            continue
        for key in ("address", "bytes", "owner_agent"):
            _same(existing[key], region[key], "exact CPU slab alias identity " + key)
        if existing["kind"] == region["kind"]:
            _same(existing, region, "deduplicated CPU backing identity")
        elif region["kind"] == "TRACKED_CPU_ALLOCATION":
            result["observed_cpu_backing_regions"][index] = region
        return
    result["observed_cpu_backing_regions"].append(region)
    result["observed_cpu_backing_bytes"] += region["bytes"]


def _read_artifact(reference: Any, label: str, *, json_value: bool = True) -> Any:
    _artifact(reference, label)
    path = Path(reference["path"])
    if not path.is_absolute() or not path.is_file():
        raise ResourceEvidenceError(label + ": original absolute artifact is unavailable")
    size = reference.get("size_bytes", reference.get("bytes"))
    size = _int(size, label + " actual file bytes", 1)
    if "size_bytes" in reference and "bytes" in reference:
        _same(reference["bytes"], size, label + " original byte-count aliases")
    if size > 2**31 or path.stat().st_size != size:
        raise ResourceEvidenceError(label + ": original bounded artifact extent differs")
    encoded = path.read_bytes()
    if len(encoded) != size or hashlib.sha256(encoded).hexdigest() != reference["sha256"]:
        raise ResourceEvidenceError(label + ": original file bytes/SHA256 differ")
    return json.loads(encoded) if json_value else encoded


def _kfd_host_backing(native: dict, evidence: dict, by_cid: dict[int, dict], targets: set[int],
                       worker_binding: dict | None) -> dict:
    """Load original tapes and independently rerun the driver event-page audit."""
    kernel = evidence.get("kfd_event_evidence")
    if not isinstance(kernel, dict):
        raise ResourceEvidenceError("Hardware-backed live signal has no original passive KFD evidence")
    for key, value in {"schema_version": 1, "kind": "passive_kfd_event_evidence"}.items():
        _same(kernel.get(key), value, "original passive KFD evidence." + key)
    if worker_binding is None:
        raise ResourceEvidenceError("Hardware event-page admission requires the source-bound worker SDK-stop prefix")
    source = kernel.get("source_binding")
    if not isinstance(source, dict):
        raise ResourceEvidenceError("Actual KFD source/header/DSO/driver binding is missing")
    reports = {key: _read_artifact(kernel.get(key + "_artifact"), "original KFD " + key)
               for key in ("pre_query_report", "post_query_report", "final_report")}
    _same(kernel.get("report_artifact"), kernel.get("post_query_report_artifact"), "original query-period KFD report reference")
    pre, post, final = (reports[key] for key in ("pre_query_report", "post_query_report", "final_report"))
    for report in (pre, post, final):
        if not isinstance(report, dict):
            raise ResourceEvidenceError("Original KFD prefix is not a JSON object")
        _same(report.get("pid"), native.get("pid"), "actual native/KFD worker process")
        _same(report.get("abi"), source.get("abi"), "actual source-bound KFD ABI")
        rows = report.get("events")
        if not isinstance(rows, list):
            raise ResourceEvidenceError("Original KFD bounded raw prefix is missing")
        _same(report.get("last_sequence"), len(rows), "original KFD prefix cursor/retained rows")
    if not (pre["events"] == post["events"][:len(pre["events"])]
            and post["events"] == final["events"][:len(post["events"])]) or not (
            pre["snapshot_ns"] <= post["snapshot_ns"] <= final["snapshot_ns"]):
        raise ResourceEvidenceError("KFD pre/post/final are not immutable ordered prefixes of one actual tape")
    observation = kernel.get("final_snapshot_observation")
    if not isinstance(observation, dict):
        raise ResourceEvidenceError("Successful SDK-stop to actual final KFD prefix receipt is missing")
    for key, value in {"process_id": native["pid"], "native_stop_returncode": 0, "write_returncode": 0,
        "prefix_only": True, "complete_workspace": False, "hardware_validated": False,
        "method": "passive_KFD_registry_prefix_after_natural_shutdown_and_successful_SDK_stop",
        "status": "OBSERVED_PASSIVE_PREFIX_REQUIRES_INDEPENDENT_AUDIT"}.items():
        _same(observation.get(key), value, "actual final KFD prefix receipt." + key)
    _int(observation.get("thread_id"), "actual final KFD receipt thread", 1)
    _source_record(observation.get("source"), ".WorkspaceWorker._ws_capture_final_kfd_prefix",
                   worker_binding["source_scope"]["worker_source"])
    begin = _int(observation.get("started_monotonic_raw_ns"), "actual final KFD RAW start", 1)
    end = _int(observation.get("ended_monotonic_raw_ns"), "actual final KFD RAW end", 1)
    left, right = (_int(observation.get(key + "_cursor"), "actual final KFD " + key + " cursor") for key in ("before", "after"))
    if (right == 2**64 - 1 or not left <= final["last_sequence"] <= right
            or not native["stop_evidence"].get("end_ns", 2**64) <= begin <= final["snapshot_ns"] <= end):
        raise ResourceEvidenceError("Final KFD prefix does not cover the actual successfully closed SDK window")
    manifest = _read_artifact(kernel.get("namespace_control_artifact"), "original cold namespace-control manifest")
    _same(manifest, kernel.get("namespace_control"), "original embedded cold namespace-control manifest bytes")
    if not isinstance(manifest, dict):
        raise ResourceEvidenceError("Original cold namespace-control manifest is missing")
    for key, value in {"schema_version": 1, "kind": "cold_hsa_kfd_namespace_control_manifest"}.items():
        _same(manifest.get(key), value, "actual cold namespace-control manifest." + key)
    cold_native = _read_artifact(manifest.get("native_report"), "original cold native tape")
    cold_kfd = _read_artifact(manifest.get("kfd_report"), "original cold KFD tape")
    control_bytes = _read_artifact(manifest.get("control_jsonl"), "original cold control JSONL", json_value=False)
    control = [json.loads(line) for line in control_bytes.splitlines() if line.strip()]
    if not control or not isinstance(control[0], dict) or not isinstance(control[-1], dict):
        raise ResourceEvidenceError("Actual independent cold ordinary-signal control output is missing")
    _same(control[0].get("pid"), cold_native.get("pid"), "actual cold control/native process")
    _same(control[0].get("kind"), "cold_ordinary_signal_control", "actual cold control entry")
    for key in ("init_status", "native_begin", "provider_init_status"):
        _same(control[0].get(key), 0, "actual cold control initialization " + key)
    for key in ("native_end", "native_stop", "native_finalize", "kfd_write", "failures"):
        _same(control[-1].get(key), 0, "actual independent cold control completion " + key)
    build = _read_artifact(manifest.get("build_receipt"), "original cold Native/KFD shared-DSO build receipt")
    _same(build.get("returncode"), 0, "actual shared Native/KFD DSO compiler return")
    cold_source = manifest.get("source_binding")
    if not isinstance(cold_source, dict):
        raise ResourceEvidenceError("Actual cold source binding is missing")
    _same(build.get("binary_sha256"), cold_source.get("probe_library_sha256"), "actual cold observer shared-DSO bytes")
    source_hashes = build.get("sources_sha256")
    if not isinstance(source_hashes, dict) or not all(cold_source.get(key) in source_hashes.values()
            for key in ("probe_source_sha256", "native_probe_source_sha256")):
        raise ResourceEvidenceError("Actual cold DSO compiler receipt does not bind both observer source files")
    namespace = {"native_report": cold_native, "kfd_report": cold_kfd, "source_binding": cold_source,
        **{key: manifest.get(key) for key in ("create_cid", "event_query_cid", "kfd_create_event_call_id", "target_epochs")}}
    locators = []
    for record in evidence.get("records", []):
        for sample in record.get("samples", []):
            query, _, payload = _public_query(by_cid, sample.get("event_id_cid"), "hsa_amd_signal_get_event_id")
            event_id = _int(payload.get("event_id"), "actual public hardware-event ID")
            if event_id == 0:
                continue
            window = sample.get("kfd_window")
            if not isinstance(window, dict):
                raise ResourceEvidenceError("Hardware event query has no actual passive KFD cursor/RAW window")
            before, after = (_int(window.get(key + "_cursor"), "actual per-query KFD " + key + " cursor") for key in ("before", "after"))
            start, finish = (_int(window.get(key + "_monotonic_raw_ns"), "actual per-query KFD RAW " + key, 1) for key in ("started", "ended"))
            if not (pre["last_sequence"] <= before <= after <= post["last_sequence"]
                    and pre["snapshot_ns"] <= start <= query["enter"]["ns"] <= query["exit"]["ns"] <= finish <= post["snapshot_ns"]):
                raise ResourceEvidenceError("Hardware query is not inside the original passive KFD prefix/cursor windows")
            locators.append({"create_cid": record.get("create_cid"), "event_query_cid": query["enter"]["cid"],
                "kfd_cursor_before": before, "kfd_cursor_after": after})
    path = Path(__file__).with_name("kfd_event_audit.py")
    spec = importlib.util.spec_from_file_location("_mlsys_independent_kfd_event_audit", path)
    if spec is None or spec.loader is None:
        raise ResourceEvidenceError("Independent pure KFD event-page audit is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit_kfd_events(final, native_report=native, event_observations=locators,
        source_binding=source, namespace_control=namespace, target_epochs=sorted(targets))
    if not isinstance(result, dict) or result.get("status") != "KFD_EVENT_GTT_BACKING_PROVEN" or result.get("issues"):
        raise ResourceEvidenceError("Independent KFD event-page replay remains partial: " + str(result.get("issues") if isinstance(result, dict) else result))
    _same(result.get("complete_workspace"), False, "KFD proof is not a complete Workspace verdict")
    _same(result.get("hardware_validated"), False, "KFD proof is not a hardware-wide validation verdict")
    return result


def audit_synchronization_resources(native_report: dict, *, target_epochs: list[int], device_index: int,
                                     natural_shutdown_evidence: dict | None = None,
                                     host_signal_evidence: dict | None = None) -> dict:
    """Recompute restricted event/signal proofs from actual raw callbacks.

    Startup events may remain live as fixed synchronization resources. Formal
    resources require actual generation closure/completion, or the restricted
    positive CPU ABI-body evidence for a still-live event-linked or directly
    stream-synchronize-owned HSA signal.
    The caller authenticates original artifact/source/module bytes and their
    session bindings; this audit independently replays actual API generations,
    outputs and fences. No byte value is inferred from absent memory callbacks.
    """
    result: dict[str, Any] = {"schema_version": 1, "kind": "synchronization_resource_lifecycle_audit",
        "status": "PARTIAL", "complete_workspace": False, "hardware_validated": False,
        "allowed_call_proofs": [], "event_generations": [], "signal_generations": [],
        "startup_live_resources": [], "issues": [], "gpu_storage_bytes": None,
        "observed_cpu_backing_regions": [], "observed_cpu_backing_bytes": 0,
        "scope": "HIP_DisableTiming_event_event_linked_and_stream_fence_owned_HSA_signal_resources_only",
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
        if host_signal_evidence is not None:
            json.dumps(host_signal_evidence, allow_nan=False)
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
        calls, by_cid = _calls(native_report, host_signal_evidence is not None)
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
        worker_binding = None
        if host_signal_evidence is not None:
            host_signal_evidence, worker_binding = _unwrap_worker_host_evidence(native_report, host_signal_evidence,
                by_cid, set(target_epochs), device_index, shutdown)
            host_signal_evidence = _adapt_host_evidence(native_report, host_signal_evidence, by_cid, set(target_epochs))
        _audit_resources(result, calls, by_cid, set(target_epochs), device_index, shutdown,
                         native_report, host_signal_evidence, worker_binding)
    except (ResourceEvidenceError, ValueError, TypeError, KeyError, AttributeError, IndexError, OSError, ImportError,
            OverflowError, RecursionError) as exc:
        result["issues"].append({"code": "UNPROVEN_SYNCHRONIZATION_RESOURCE", "message": str(exc)})
        result["allowed_call_proofs"] = []
    return result


def _audit_resources(result: dict, calls: list[dict], by_cid: dict[int, dict], targets: set[int], device: int,
                     shutdown: dict | None, native: dict, host_evidence: dict | None,
                     worker_binding: dict | None) -> None:
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
    stream_records: dict[int, list[dict]] = defaultdict(list)
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
            if operation in _RECORD:
                stream_records[stream].append(call)
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
    worker_fence = None
    if worker_binding is not None:
        worker_fence = next((value for value in fences if value["enter"]["cid"] == worker_binding["fence_cid"]), None)
        if worker_fence is None:
            raise ResourceEvidenceError("Worker source fence lacks independently replayed actual target-device context")
        result["worker_host_evidence_binding"] = {**worker_binding, "completion_fence": _proof(worker_fence, "worker-last-target-fence")}

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
    kfd_result = None

    def load_kfd() -> dict:
        nonlocal kfd_result
        if kfd_result is None:
            if host_evidence is None:
                raise ResourceEvidenceError("Actual passive KFD evidence is missing")
            kfd_result = _kfd_host_backing(native, host_evidence, by_cid, targets, worker_binding)
        return kfd_result
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

    owner_lifecycle: list[dict] | None = None

    def stream_fence_scope(create: dict, parent: dict) -> tuple[dict, dict, dict]:
        """A direct stream-sync owner is a separate scope, never a HIP event."""
        nonlocal owner_lifecycle
        a, b = parent["enter"], parent["exit"]
        c, d = create["enter"], create["exit"]
        _same(c.get("ancestor"), a["cid"], "direct stream-owned signal parent correlation")
        if not (a["tid"] == c["tid"] == d["tid"] == b["tid"]
                and a["epoch"] == c["epoch"] == d["epoch"] == b["epoch"]
                and a["seq"] < c["seq"] < d["seq"] < b["seq"]
                and a["ns"] <= c["ns"] <= d["ns"] <= b["ns"]):
            raise ResourceEvidenceError("Stream-owned signal lacks its actual same-thread/epoch nested parent scope")
        _success(parent)
        entry, output = _payload(parent, phase="enter"), _payload(parent)
        for value in (entry, output):
            _same(value.get("kind"), "stream_sync", "actual stream-owned signal parent kind")
            _int(value.get("stream_handle"), "actual stream-owned signal parent stream")
        _same(entry["stream_handle"], output["stream_handle"], "paired actual parent stream")
        for phase in ("enter", "exit"):
            _same(observed_device(parent, phase), device, "independently replayed stream-owned target device")
        stream = entry["stream_handle"]
        _same(stream_devices.get(stream), device, "independently replayed actual parent stream device")
        records = [value for value in stream_records[stream] if value["exit"]["seq"] < a["seq"]
            and value["exit"]["ns"] <= a["ns"]
            and call_events[value["enter"]["cid"]]["device_index"] == device]
        if not records:
            raise ResourceEvidenceError("Stream-owned signal has no prior successful target-device event record on its actual stream")
        if owner_lifecycle is None:
            owner_lifecycle = _stream_owner_lifecycle_calls(native)
        valid_record = None
        lifecycle_proof = None
        for record in reversed(records):
            r = record["exit"]
            unrelated = []
            for mutation in owner_lifecycle:
                first, last = mutation["enter"], mutation["exit"]
                if (last["seq"] <= r["seq"] and last["ns"] <= r["ns"]
                        or first["seq"] >= a["seq"] and first["ns"] >= a["ns"]):
                    continue
                before, after = first.get("raw_hip_control_payload"), last.get("raw_hip_control_payload")
                # An error may report earlier asynchronous work. No failed or
                # untyped return is assumed to imply absence of a side effect.
                successful = (type(last.get("status_type")) is int and last["status_type"] == 1
                    and type(last.get("status")) is int and last["status"] == 0)
                exact_other = (mutation["stream_operation"] and successful
                    and isinstance(before, dict) and isinstance(after, dict)
                    and type(before.get("stream_handle")) is int and 0 < before["stream_handle"] < 2**64
                    and type(after.get("stream_handle")) is int
                    and before["stream_handle"] == after["stream_handle"] != stream)
                if not exact_other:
                    break
                unrelated.append({"operation": mutation["operation"], "cid": first["cid"],
                    "stream_handle": after["stream_handle"], "enter_sequence": first["seq"],
                    "exit_sequence": last["seq"]})
            else:
                valid_record = record
                lifecycle_proof = {"method": "full_native_tape_without_stream_or_context_lifecycle_invalidation",
                    "record_exit_sequence": r["seq"], "parent_enter_sequence": a["seq"],
                    "record_exit_monotonic_raw_ns": r["ns"], "parent_enter_monotonic_raw_ns": a["ns"],
                    "precise_other_stream_operations": unrelated}
                break
        if valid_record is None or lifecycle_proof is None:
            raise ResourceEvidenceError("Stream-owned signal's prior records are invalidated by unresolved stream/context lifecycle")
        if (worker_binding is None or worker_fence is None
                or worker_fence["operation"] != "hipDeviceSynchronize"
                or worker_fence["enter"]["seq"] <= b["seq"]
                or worker_fence["enter"]["ns"] < b["ns"]):
            raise ResourceEvidenceError("Stream-owned live signal lacks a later source-bound target-device worker fence")
        return worker_fence, valid_record, lifecycle_proof

    signal_live = {}
    signal_generations = []
    for call in calls:
        if call["operation"] not in _SIGNAL:
            continue
        a, b = call["enter"], call["exit"]
        if call["operation"] == "hsa_signal_create":
            direct_parent = by_cid.get(a.get("ancestor"))
            stream_owned = direct_parent is not None and direct_parent["operation"] == "hipStreamSynchronize"
            if stream_owned:
                event_generation, parent = None, direct_parent
            else:
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
            if parent is not None and not stream_owned:
                _success(parent)
            generation = {"generation_id": "hsa-signal-" + str(handle) + "-c" + str(a["cid"]),
                "signal_handle": handle, "create": call, "destroy": None,
                "parent": parent, "event_generation": event_generation, "stream_owned": stream_owned,
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
        stream_owned = generation["stream_owned"]
        if event_generation is None and not stream_owned:
            continue  # A different, unchanged native rule must admit this call.
        host = None
        parent_stream_binding = None
        parent_stream_lifecycle = None
        if stream_owned:
            if destroy is not None or host_evidence is None or generation["parent"] is None:
                raise ResourceEvidenceError("Stream-owned HSA admission requires an observed still-live CPU-backed generation")
            fence, stream_record, parent_stream_lifecycle = stream_fence_scope(create, generation["parent"])
            parent_stream_binding = _proof(stream_record,
                call_events[stream_record["enter"]["cid"]]["generation_id"])
            host = _live_host_signal(native, host_evidence, generation, signal_generations, by_cid, fence, load_kfd,
                                     shutdown)
            fence_order = "stream_sync_parent_then_source_bound_target_device_fence_then_positive_CPU_ABI_backing_query"
            parent_scope = "HIP_STREAM_SYNCHRONIZE_INTERNAL_HSA_SIGNAL"
            parent_scope_id = "hip-stream-sync-c" + str(generation["parent"]["enter"]["cid"])
        elif event_generation is None or event_generation not in relevant_generations or event_generation["device_index"] != device:
            raise ResourceEvidenceError("Formal HSA signal does not belong to an admitted target-device event")
        elif destroy is None:
            if host_evidence is None:
                raise ResourceEvidenceError("Formal event-linked HSA signal remains live at SDK stop")
            last_use = max([generation["parent"]["exit"]["seq"], create["exit"]["seq"],
                *[value["exit"]["seq"] for value in event_generation["uses"] if value["operation"] in _RECORD | _WAIT
                   or value["operation"] == "hipEventQuery" and value["exit"]["status"] == 600]])
            fence = worker_fence if worker_fence is not None else fence_after(last_use, event_generation)
            if fence is not None and fence["enter"]["seq"] <= last_use:
                fence = None
            if fence is None:
                raise ResourceEvidenceError("Live signal has no observed fence after its last GPU use")
            host = _live_host_signal(native, host_evidence, generation, signal_generations, by_cid, fence, load_kfd,
                                     shutdown)
            fence_order = "last_GPU_use_then_completion_fence_then_positive_CPU_ABI_backing_query"
        else:
            fence, fence_order = completion_fence(destroy, event_generation, parent_use=generation["parent"])
        if not stream_owned:
            parent_scope = "HIP_EVENT_INTERNAL_HSA_SIGNAL"
            parent_scope_id = event_generation["generation_id"]
        if fence is None:
            raise ResourceEvidenceError("HSA signal destroy has no observed later completion fence")
        for call in (create, destroy):
            if call is not None and call["enter"]["epoch"] in targets:
                proofs.append(_proof(call, generation["generation_id"], fence))
        result["signal_generations"].append({"generation_id": generation["generation_id"],
            "signal_handle": generation["signal_handle"], "initial_value": generation["initial_value"],
            "num_consumers": generation["num_consumers"], "consumer_agents": generation["consumer_agents"],
            "parent_scope": parent_scope, "parent_scope_id": parent_scope_id,
            "parent_stream_binding": parent_stream_binding,
            "parent_stream_lifecycle": parent_stream_lifecycle,
            "parent_event_generation_id": None if stream_owned else event_generation["generation_id"],
            "parent_call": _proof(generation["parent"], parent_scope_id),
            "create": _proof(create, generation["generation_id"]),
            "destroy": None if destroy is None else _proof(destroy, generation["generation_id"]),
            "state": "LIVE_CPU_BACKED_ABI_SIGNAL_RESOURCE" if host is not None else "DESTROYED_AND_COMPLETION_FENCED",
            "completion_fence": _proof(fence, generation["generation_id"]),
            "completion_order": fence_order,
            "gpu_storage_bytes": None, "host_backing_evidence": host})
        if host is not None:
            _record_cpu_region(result, host["backing"])
    if kfd_result is not None:
        result["gtt_event_page_backing"] = kfd_result
        result["observed_gtt_event_page_bytes"] = kfd_result["gtt_page_bytes"]
    result.update(status="SYNCHRONIZATION_RESOURCES_PROVEN", allowed_call_proofs=proofs)
