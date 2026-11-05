"""CPU-only resource/control observations; no GPU or driver is exercised."""

from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

_PATH = Path(__file__).resolve().parents[1] / "scripts/rocm_workspace_audit/synchronization_resource_audit.py"
_SPEC = importlib.util.spec_from_file_location("synchronization_resource_audit", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOL)
audit_synchronization_resources = _TOOL.audit_synchronization_resources


def resource_fixture():
    events: list[dict] = []
    operations: list[dict[str, int]] = [{}, {}, {}]

    def row(operation, cid, phase, epoch, payload, *, ancestor=0, status=0):
        domain = 2 if operation.startswith("hsa_") else 0
        table = operations[domain]
        if operation not in table:
            table[operation] = len(table)
        sequence = len(events) + 1
        value = {"seq": sequence, "callback_seq": sequence, "ns": sequence * 10,
            "domain": domain, "op": table[operation], "cid": cid, "tid": 77,
            "phase": phase, "epoch": epoch, "ancestor": ancestor,
            "flags": 64 if operation in {"hipEventCreateWithFlags", "hipEventDestroy", "hsa_signal_create", "hsa_signal_destroy"} else 0,
            "status_type": (2 if domain == 2 else 1) if phase == 2 else 0,
            "status": status if phase == 2 else None,
            "raw_hsa_signal_payload" if domain == 2 else "raw_hip_control_payload": payload}
        events.append(value)
        return value

    def hip(kind, handle=None, stream=0, flags=0, output=0):
        return {"kind": kind, "object_handle": handle, "stream_handle": stream,
                "flags": flags, "output_address": output, "device_index": 0, "device_source_cid": 1}

    def call(name, cid, epoch, entry, exit=None, **kwargs):
        row(name, cid, 1, epoch, deepcopy(entry), **kwargs)
        row(name, cid, 2, epoch, deepcopy(entry if exit is None else exit), **kwargs)

    call("hipSetDevice", 1, 0, hip("device_set"))
    call("hipEventCreateWithFlags", 2, 0, hip("event", flags=2, output=100), hip("event", 10, flags=2, output=100))
    call("hipEventRecord", 3, 1, hip("event", 10, 101))
    call("hipEventQuery", 4, 1, hip("event", 10), status=600)
    call("hipEventCreateWithFlags", 5, 1, hip("event", flags=2, output=200), hip("event", 20, flags=2, output=200))
    call("hipEventRecord", 6, 1, hip("event", 20, 101))
    row("hipStreamWaitEvent", 7, 1, 1, hip("event", 20, 102))
    signal: dict = {"operation": "create", "signal_handle": None, "output_address": 300,
              "initial_value": 1, "num_consumers": 0, "consumers_observed": True, "consumer_agents": []}
    call("hsa_signal_create", 8, 1, signal, {**signal, "signal_handle": 30}, ancestor=7)
    row("hipStreamWaitEvent", 7, 2, 1, hip("event", 20, 102))
    call("hipEventDestroy", 9, 1, hip("event", 20))
    call("hsa_signal_destroy", 10, 1, {"operation": "destroy", "signal_handle": 30})
    call("hipDeviceSynchronize", 11, 2, hip("device_sync"))
    domains = [{"operations": [{"id": number, "name": name} for name, number in table.items()]}
               for table in operations]
    data = {"schema_version": 1, "kind": "native_allocation_api_diagnostic", "configured": True,
        "init_called": True, "ready": True, "bounds_valid": True, "write_complete": True,
        "finalize_callback_called": True, "metadata_truncated": False, "filter_policy": "all_APIs_global",
        "dropped_events": 0, "recursive_suppressed": 0, "unknown_events": 0, "missing_user_data": 0,
        "filtered_nonmutator_events": 0, "callbacks_inflight": 0,
        "retained_events": len(events), "attempted_events": len(events), "observed_callbacks": len(events),
        "max_events": 1024, "events": events, "domains": domains,
        "epochs": [{"id": 1, "begin_ns": 45, "end_ns": 205, "ended": True},
                   {"id": 2, "begin_ns": 205, "end_ns": 225, "ended": True}],
        "stop_evidence": {"invoked": True, "completed": True, "observation_window_closed": True,
            "stop_result": 0, "drain_status": 0, "callbacks_after_stop": 0, "actual_sdk_stop_calls": 1}}
    return data


def named(data, name, phase=2):
    for value in data["events"]:
        operation = next(item["name"] for item in data["domains"][value["domain"]]["operations"] if item["id"] == value["op"])
        if operation == name and value["phase"] == phase:
            return value
    raise AssertionError(name)


def natural_shutdown_fixture():
    data = resource_fixture()
    data["pid"] = 77
    pending = data["events"]
    before = [row for row in pending if row["cid"] not in (9, 10, 11)]
    fence = [row for row in pending if row["cid"] == 11]
    destroys = [row for row in pending if row["cid"] in (9, 10)]
    for row in destroys:
        row["epoch"] = 0
    data["events"] = before + fence + destroys
    for index, row in enumerate(data["events"], 1):
        row.update(seq=index, callback_seq=index, ns=index * 10)
    data["epochs"] = [{"id": 1, "begin_ns": 45, "end_ns": 165, "ended": True},
                      {"id": 2, "begin_ns": 165, "end_ns": 185, "ended": True}]
    receipt = {"process_id": 77, "thread_id": 77, "original_returned": True,
        "return_type": "builtins.NoneType", "native_clock": "CLOCK_MONOTONIC_RAW",
        "started_monotonic_raw_ns": 185, "ended_monotonic_raw_ns": 225,
        "clock": "CLOCK_MONOTONIC", "started_monotonic_ns": 50000, "ended_monotonic_ns": 60000,
        "scope": "original_WorkerProc_shutdown_including_model_and_distributed_environment",
        "source": {"qualified_name": "vllm.v1.executor.multiproc_executor.WorkerProc.shutdown",
            "source_sha256": "36f51b0ac5471bc230ccfce7dfc6ced2d16d2c73e35b10a3ab81eae1ec592140",
            "file_sha256": "3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a",
            "source_reference": "cpu-fixture-pinned-original-method.py"}}
    return data, receipt


def host_signal_fixture():
    data = resource_fixture()
    data["pid"] = 77
    data["events"] = [row for row in data["events"] if row["cid"] != 10]
    for row in data["events"]:
        if row["cid"] == 8 and row["phase"] == 2:
            row["raw_hsa_signal_payload"]["signal_handle"] = 32768
    for index, row in enumerate(data["events"], 1):
        row.update(seq=index, callback_seq=index, ns=index * 10)
    data["epochs"] = [{"id": 1, "begin_ns": 45, "end_ns": 185, "ended": True},
                      {"id": 2, "begin_ns": 185, "end_ns": 205, "ended": True}]
    fields = {"returned_size": [0, 4], "type": [4, 4], "agent_base_address": [8, 8],
        "host_base_address": [16, 8], "size_bytes": [24, 8], "owner_agent": [40, 8],
        "global_flags": [48, 4], "registered": [52, 1], "alloc_flags": [56, 4]}
    def artifact(path, digest="a" * 64):
        return {"path": path, "sha256": digest}
    abi = {"compiler_source": artifact("/actual/probe.cpp"),
        "headers": [artifact("/actual/amd_hsa_signal.h", "a8606371af1c186417963f640a0d2a418805a313b0c8c0fe23aa42b0d813cecd")],
        "runtime_dsos": [artifact("/actual/libhsa-runtime64.so.1")],
        "signal_handle_address_mask": 2**64 - 1, "amd_signal_size": 64, "amd_signal_alignment": 64,
        "pointer_info_size": 64, "pointer_info_fields": fields, "device_attribute": 17,
        "cpu_device_enum": 0, "pointer_type_hsa": 1, "pointer_type_locked": 2}
    before_cursor = len(data["events"])
    data["domains"].append({"operations": []})
    def query(name, cid, entry, output):
        domain = 2 if name == "hsa_agent_get_info" else 3
        table = data["domains"][domain]["operations"]
        op = next((row["id"] for row in table if row["name"] == name), len(table))
        if not any(row["name"] == name for row in table):
            table.append({"id": op, "name": name})
        payload_key = {"hsa_agent_get_info": "raw_hsa_agent_info_payload", "hsa_amd_pointer_info": "raw_hsa_pointer_info_payload",
                       "hsa_amd_signal_get_event_id": "raw_hsa_signal_event_payload"}[name]
        for phase, payload in ((1, entry), (2, output)):
            sequence = len(data["events"]) + 1
            data["events"].append({"seq": sequence, "callback_seq": sequence, "ns": sequence * 10,
                "domain": domain, "op": op, "cid": cid, "tid": 77, "epoch": 0, "ancestor": 0,
                "phase": phase, "flags": 512, "status_type": 2 if phase == 2 else 0,
                "status": 0 if phase == 2 else None, payload_key: {"payload_observed": True, **payload}})
    samples = []
    for cid in (20, 23):
        query("hsa_amd_pointer_info", cid, {"pointer_address": 32768, "info_address": 5000, "input_size": 64},
              {"pointer_address": 32768, "info_address": 5000, "input_size": None,
               "returned_size": 64, "type": 1, "host_base_address": 32768, "agent_base_address": 32768,
               "size_bytes": 4096, "owner_agent": 500})
        query("hsa_agent_get_info", cid + 1, {"agent_handle": 500, "attribute": 17, "output_address": 6000},
              {"agent_handle": 500, "attribute": 17, "output_address": 6000, "device_type": 0})
        query("hsa_amd_signal_get_event_id", cid + 2, {"signal_handle": 32768, "output_address": 7000},
              {"signal_handle": 32768, "output_address": 7000, "event_id": 0})
        samples.append({"pointer_info_cid": cid, "owner_device_cid": cid + 1, "event_id_cid": cid + 2})
    data["hsa_query_storage"] = {"schema_version": 1, "method": "bounded_public_hsa_query_payloads_v1",
        "capacity": 8192, "retained_records": 12, "record_size_bytes": 128, "storage_capacity_bytes": 8192 * 128,
        "dropped_records": 0, "read_errors": 0, "pointer_info_struct_size": 64, "pointer_info_fields": fields,
        "agent_info_device_attribute": 17, "cpu_device_type": 0, "pointer_type_hsa": 1, "pointer_type_locked": 2}
    for key in ("retained_events", "attempted_events", "observed_callbacks"):
        data[key] = len(data["events"])
    data["stop_evidence"]["begin_ns"] = 1000
    generation = {"signal_handle": 32768, "generation": 1, "create_cid": 8,
        "create_seq": next(row["seq"] for row in data["events"] if row["cid"] == 8 and row["phase"] == 2),
        "create_epoch": 1, "create_tid": 77, "initial_value": 1, "num_consumers": 0,
        "consumers_observed": True, "consumer_agents": []}
    before = {"schema_version": 1, "kind": "observed_live_hsa_signal_generations", "pid": 77,
        "complete_workspace": False, "hardware_validated": False, "dropped_events": 0,
        "reused_live_handle_count": 0, "unknown_destroy_count": 5,
        "snapshot_monotonic_raw_ns": before_cursor * 10 + 1, "event_cursor": before_cursor,
        "generations": [generation]}
    after = {**deepcopy(before), "snapshot_monotonic_raw_ns": len(data["events"]) * 10 + 1, "event_cursor": len(data["events"])}
    control = {"source": artifact("/actual/control.cpp"), "result_artifact": artifact("/actual/control.json"),
        "abi_sha256": hashlib.sha256(json.dumps(abi, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "runtime_dsos": abi["runtime_dsos"], "receipt": {"compile_returncode": 0, "run_returncode": 0,
            "result": {"init_status": 0, "create_status": 0, "pointer_info_status": 0, "owner_status": 0,
                "event_status": 0, "destroy_status": 0, "amd_signal_size": 64, "amd_signal_alignment": 64,
                "device_attribute": 17, "cpu_enum": 0, "hsa_type_enum": 1, "locked_type_enum": 2,
                "owner_device": 0, "pointer_info_size": 64, "pointer_info_type": 1, "signal_handle": 65536,
                "host_base": 65536, "size_bytes": 4096, "owner_handle": 500, "event_id": 3,
                "module": "/actual/libhsa-runtime64.so.1"}}}
    evidence = {"schema_version": 1, "kind": "public_hsa_signal_host_backing_evidence", "process_id": 77,
        "provider": artifact("/actual/provider.cpp"), "abi": abi, "before_live": before, "after_live": after,
        "ordinary_signal_control": control,
        "records": [{key: generation[key] for key in ("signal_handle", "generation", "create_cid", "create_seq")} | {"samples": samples}]}
    return data, evidence


def public_provider_fixture():
    data, evidence = host_signal_fixture()
    control_rows = []
    for phase in (1, 2):
        row = deepcopy(next(row for row in data["events"] if row["cid"] == 8 and row["phase"] == phase))
        row.update(cid=40, epoch=0, ancestor=0)
        row["raw_hsa_signal_payload"].update(output_address=9000, signal_handle=None if phase == 1 else 65536)
        control_rows.append(row)
    for old_cid in range(20, 26):
        for phase in (1, 2):
            row = deepcopy(next(row for row in data["events"] if row["cid"] == old_cid and row["phase"] == phase))
            row["cid"] = old_cid + 21
            for field in ("raw_hsa_pointer_info_payload", "raw_hsa_signal_event_payload"):
                payload = row.get(field)
                if payload is not None:
                    for key in ("pointer_address", "host_base_address", "signal_handle"):
                        if key in payload:
                            payload[key] = 65536
                    if "event_id" in payload:
                        payload["event_id"] = 3
            control_rows.append(row)
    operation = next(row["id"] for row in data["domains"][2]["operations"] if row["name"] == "hsa_signal_destroy")
    for phase in (1, 2):
        control_rows.append({"domain": 2, "op": operation, "cid": 47, "tid": 77, "phase": phase,
            "epoch": 0, "ancestor": 0, "flags": 64, "status_type": 2 if phase == 2 else 0,
            "status": 0 if phase == 2 else None, "raw_hsa_signal_payload": {"operation": "destroy", "signal_handle": 65536}})
    data["events"].extend(control_rows)
    for index, row in enumerate(data["events"], 1):
        row.update(seq=index, callback_seq=index, ns=index * 10)
    for key in ("retained_events", "attempted_events", "observed_callbacks"):
        data[key] = len(data["events"])
    data["hsa_query_storage"]["retained_records"] = 24
    evidence["after_live"].update(event_cursor=len(data["events"]), snapshot_monotonic_raw_ns=len(data["events"]) * 10 + 1)
    def observed(cid):
        rows = [row for row in data["events"] if row["cid"] == cid]
        a, b = rows
        name = next(item["name"] for item in data["domains"][a["domain"]]["operations"] if item["id"] == a["op"])
        field = next(key for key in a if key.startswith("raw_"))
        return {"api": name, "before_cursor": a["seq"] - 1, "after_cursor": b["seq"],
            "started_monotonic_raw_ns": a["ns"] - 1, "ended_monotonic_raw_ns": b["ns"] + 1,
            "input": {key: value for key, value in a[field].items() if key != "payload_observed"},
            "output": {**{key: value for key, value in b[field].items() if key != "payload_observed"},
                       "status_type": b["status_type"], "status": b["status"]},
            "api_binding": {"address": 10000, "module_base": 9000,
                "module_path": "/actual/libhsa-runtime64.so.1", "symbol_name": name}}
    def sample(cids, handle):
        return {"process_id": 77, "thread_id": 77, "signal_handle": handle, "abi_signal_address": handle,
            "address_rule": "unhinted_amd_signal_handle", "api_calls": [observed(cid) for cid in cids]}
    producer = {**evidence, "kind": "public_hsa_signal_backing_capture", "status": "CAPTURED", "issues": [],
        "complete_workspace": False, "hardware_validated": False, "sources": [evidence["abi"]["compiler_source"]],
        "abi_headers": evidence["abi"]["headers"], "runtime_dsos": evidence["abi"]["runtime_dsos"],
        "initialization": {"abi": {key: value for key, value in evidence["abi"].items()
                                   if key not in ("compiler_source", "headers", "runtime_dsos")}},
        "records": [{**evidence["records"][0], "samples": [sample((20, 21, 22), 32768), sample((23, 24, 25), 32768)]}],
        "ordinary_signal_control": {"create_call": observed(40), "samples": [sample((41, 42, 43), 65536),
            sample((44, 45, 46), 65536)], "destroy_call": observed(47)}}
    producer.pop("abi")
    return data, producer


def tracked_cpu_fixture():
    data, evidence = host_signal_fixture()
    while len(data["domains"]) <= 6:
        data["domains"].append({"operations": []})
    ext = data["domains"][3]["operations"]
    allocate_op = len(ext)
    ext.append({"id": allocate_op, "name": "hsa_amd_memory_pool_allocate"})
    data["domains"][6]["operations"].append({"id": 0, "name": "MEMORY_ALLOCATION_ALLOCATE"})
    prefix = []
    for domain, operation, phase in ((3, allocate_op, 1), (6, 0, 1), (6, 0, 2), (3, allocate_op, 2)):
        row: dict = {"domain": domain, "op": operation, "cid": 100, "tid": 77, "phase": phase,
            "epoch": 0, "ancestor": 0, "flags": 1, "status_type": 2 if domain == 3 and phase == 2 else 0,
            "status": 0 if domain == 3 and phase == 2 else None}
        if domain == 6:
            row["raw_allocation_payload"] = {"address_or_handle": 32768, "size_bytes": 4096,
                "agent_handle": 700, "success_proven": False}
        prefix.append(row)
    data["events"] = prefix + data["events"]
    for index, row in enumerate(data["events"], 1):
        row.update(seq=index, callback_seq=index, ns=index * 10)
    for key in ("retained_events", "attempted_events", "observed_callbacks"):
        data[key] = len(data["events"])
    for epoch in data["epochs"]:
        epoch["begin_ns"] += 40
        epoch["end_ns"] += 40
    for key in ("before_live", "after_live"):
        evidence[key]["event_cursor"] += 4
        evidence[key]["snapshot_monotonic_raw_ns"] += 40
        evidence[key]["generations"][0]["create_seq"] += 4
    evidence["records"][0]["create_seq"] += 4
    evidence["records"][0]["backing"] = {"allocation_cid": 100, "normalized_allocate_exit_sequence": 3}
    data["agent_inventory"] = {"agents": [{"handle": 700, "type": 1, "type_name": "CPU"}]}
    return data, evidence


def worker_host_fixture():
    data, capture = public_provider_fixture()
    for row in data["events"]:
        if row["cid"] == 11:
            row["epoch"] = 9001
    data["epochs"][1]["id"] = 9001
    for key in ("before_live", "after_live"):
        capture[key]["unknown_destroy_count"] = 0
    capture["before_live"]["snapshot_monotonic_raw_ns"] = 209
    capture.update(thread_id=77, capture_started_monotonic_raw_ns=208, capture_ended_monotonic_raw_ns=482,
        before_live_artifact={"path": "/actual/live-before.json", "sha256": "a" * 64},
        after_live_artifact={"path": "/actual/live-after.json", "sha256": "a" * 64})
    worker = {"worker_source_reference": "/actual/worker_workspace_capture.py", "worker_source_sha256": "b" * 64}
    def source(method):
        return {"qualified_name": "worker_workspace_capture.WorkspaceWorker." + method,
            "source_reference": worker["worker_source_reference"], "file_sha256": worker["worker_source_sha256"],
            "source_sha256": "c" * 64}
    fence = {"native_cursor_before": 18, "native_cursor_after": 20, "started_monotonic_raw_ns": 189,
        "ended_monotonic_raw_ns": 201, "native_epoch_id": 9001, "process_id": 77, "thread_id": 77,
        "target_device_index": 0, "native_clock": "CLOCK_MONOTONIC_RAW",
        "method": "existing_post_cohort_last_target_device_synchronize", "source": source("mlsys_workspace_close")}
    scope = {"kind": "post_formal_pre_natural_shutdown_public_signal_queries", "session_id": "cpu-fixture-session",
        "process_id": 77, "thread_id": 77, "target_device_index": 0, "native_clock": "CLOCK_MONOTONIC_RAW",
        "source": source("_ws_capture_host_signals"), "worker_source": worker,
        "formal_native_epochs": [1], "query_started_monotonic_raw_ns": 206}
    capture.update(source_fence=deepcopy(fence), source_scope=deepcopy(scope))
    post = {"process_id": 77, "thread_id": 77, "kind": "post_natural_shutdown_observed_signal_generations",
        "path": "/actual/live-after-shutdown.json", "sha256": "d" * 64, "bytes": 300,
        "new_signal_queries_performed": False, "complete_workspace": False, "hardware_validated": False,
        "export_returncode": 0, "status": "OBSERVED_REGISTRY_SNAPSHOT_ONLY",
        "observed_live": {**deepcopy(capture["after_live"]), "snapshot_monotonic_raw_ns": 501}}
    envelope = {"schema_version": 1, "kind": "public_hsa_signal_backing_worker_evidence", "process_id": 77,
        "thread_id": 77, "source_fence": fence, "source_scope": scope, "query_started_monotonic_raw_ns": 206,
        "query_ended_monotonic_raw_ns": 483, "native_cursor_before": 20, "native_cursor_after": 48,
        "complete_workspace": False, "hardware_validated": False,
        "status": "CAPTURE_RETURNED_REQUIRES_INDEPENDENT_NATIVE_AUDIT", "capture": capture,
        "capture_artifact": {"path": "/actual/capture.json", "sha256": "e" * 64, "bytes": 3000},
        "after_natural_shutdown_live": post}
    _, shutdown = natural_shutdown_fixture()
    shutdown.update(started_monotonic_raw_ns=485, ended_monotonic_raw_ns=500)
    return data, envelope, shutdown


def stream_owned_signal(data, *, stream=101):
    """Keep the same captured live signal, with an actual direct stream-sync parent."""
    operations = data["domains"][0]["operations"]
    op = len(operations)
    operations.append({"id": op, "name": "hipStreamSynchronize"})
    for row in data["events"]:
        if row["cid"] == 7:
            row["op"] = op
            row["raw_hip_control_payload"].update(kind="stream_sync", object_handle=None, stream_handle=stream)


def insert_stream_lifecycle(data, envelope, *, cut=6, first_ns=61, handle=101, status_type=1, status=0,
                            operations=("hipStreamDestroy", "hipStreamCreate")):
    additions = []
    table = data["domains"][0]["operations"]
    for index, name in enumerate(operations):
        op = len(table)
        table.append({"id": op, "name": name})
        for phase in (1, 2):
            payload = {} if handle is None else {"stream_handle": handle}
            additions.append({"domain": 0, "op": op, "cid": 60 + index, "tid": 88,
                "ns": first_ns + len(additions), "epoch": 0, "ancestor": 0, "phase": phase,
                "flags": 64, "status_type": status_type if phase == 2 else 0,
                "status": status if phase == 2 else None, "raw_hip_control_payload": payload})
    data["events"][cut:cut] = additions
    for index, row in enumerate(data["events"], 1):
        row.update(seq=index, callback_seq=index)
    for key in ("retained_events", "attempted_events", "observed_callbacks"):
        data[key] = len(data["events"])
    cursor_keys = {"before_cursor", "after_cursor", "event_cursor", "create_seq", "native_cursor_before", "native_cursor_after"}

    def shift(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in cursor_keys and type(item) is int and item > cut:
                    value[key] += len(additions)
                else:
                    shift(item)
        elif isinstance(value, list):
            for item in value:
                shift(item)
    shift(envelope)


def hardware_worker_fixture(directory):
    # Reuse the independent driver's UAPI graph fixture, including cold causal
    # CREATE_EVENT/get_event_id control, rather than mocking its audit verdict.
    from tests import test_rocm_native_backing_audit as native_tests
    spec = importlib.util.spec_from_file_location("sync_kfd_cpu_fixture", Path(__file__).with_name("test_rocm_kfd_events.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"test_rocm_native_backing_audit": native_tests}):
        spec.loader.exec_module(module)
    cold = module.KfdFixture()
    cold.finish()
    cold.native["epochs"] = [{"id": 8001, "begin_ns": 35, "end_ns": 190, "ended": True}]
    data, envelope, shutdown = worker_host_fixture()
    data["stop_evidence"]["end_ns"] = 1001
    for row in data["events"]:
        if row["cid"] in (22, 25) and row["phase"] == 2:
            row["raw_hsa_signal_event_payload"]["event_id"] = 7
    capture = envelope["capture"]
    for sample in capture["records"][0]["samples"]:
        for observed in sample["api_calls"]:
            if observed["api"] == "hsa_amd_signal_get_event_id":
                observed["output"]["event_id"] = 7
        sample["kfd_window"] = {"before_cursor": 8, "after_cursor": 8,
            "started_monotonic_raw_ns": min(call["started_monotonic_raw_ns"] for call in sample["api_calls"]),
            "ended_monotonic_raw_ns": max(call["ended_monotonic_raw_ns"] for call in sample["api_calls"])}
    worker_report = deepcopy(cold.report)
    worker_report["pid"] = 77
    for row in worker_report["events"]:
        row.update(pid=77, tid=77)
    def artifact(name, value, *, encoded=None):
        content = json.dumps(value, sort_keys=True).encode() if encoded is None else encoded
        path = Path(directory) / name
        path.write_bytes(content)
        return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    source = deepcopy(cold.source)
    pre = {**deepcopy(worker_report), "snapshot_ns": 209}
    post = {**deepcopy(worker_report), "snapshot_ns": 481}
    final = {**deepcopy(worker_report), "snapshot_ns": 1003}
    control_output = [{"kind": "cold_ordinary_signal_control", "pid": 100, "init_status": 0,
        "native_begin": 0, "provider_init_status": 0}, {"native_end": 0, "native_stop": 0,
        "native_finalize": 0, "kfd_write": 0, "failures": 0}]
    manifest = {"schema_version": 1, "kind": "cold_hsa_kfd_namespace_control_manifest",
        "native_report": artifact("cold-native.json", cold.native), "kfd_report": artifact("cold-kfd.json", cold.report),
        "control_jsonl": artifact("cold-control.jsonl", None,
            encoded=b"\n".join(json.dumps(row).encode() for row in control_output) + b"\n"),
        "build_receipt": artifact("cold-build.json", {"returncode": 0, "binary_sha256": "a" * 64,
            "sources_sha256": {"native.cpp": "a" * 64, "kfd.cpp": "a" * 64}}),
        "source_binding": deepcopy(source), "create_cid": 1, "event_query_cid": 2,
        "kfd_create_event_call_id": cold.created, "target_epochs": [8001]}
    pre_ref, post_ref = artifact("worker-kfd-pre.json", pre), artifact("worker-kfd-post.json", post)
    kernel = {"schema_version": 1, "kind": "passive_kfd_event_evidence", "source_binding": source,
        "namespace_control": manifest, "namespace_control_artifact": artifact("cold-manifest.json", manifest),
        "pre_query_report_artifact": pre_ref, "post_query_report_artifact": post_ref, "report_artifact": deepcopy(post_ref)}
    capture["kfd_event_evidence"] = deepcopy(kernel)
    worker = envelope["source_scope"]["worker_source"]
    observation = {"process_id": 77, "thread_id": 77, "native_stop_returncode": 0, "write_returncode": 0,
        "prefix_only": True, "method": "passive_KFD_registry_prefix_after_natural_shutdown_and_successful_SDK_stop",
        "complete_workspace": False, "hardware_validated": False,
        "status": "OBSERVED_PASSIVE_PREFIX_REQUIRES_INDEPENDENT_AUDIT", "before_cursor": 8, "after_cursor": 8,
        "started_monotonic_raw_ns": 1002, "ended_monotonic_raw_ns": 1004,
        "source": {"qualified_name": "worker_workspace_capture.WorkspaceWorker._ws_capture_final_kfd_prefix",
            "source_reference": worker["worker_source_reference"], "file_sha256": worker["worker_source_sha256"],
            "source_sha256": "c" * 64}}
    envelope["kfd_event_evidence"] = {**kernel, "final_report_artifact": artifact("worker-kfd-final.json", final),
        "final_snapshot_observation": observation}
    return data, envelope, shutdown


class SynchronizationResourceTests(unittest.TestCase):
    def host_cleanup_fixture(self, *, pointer=65536, size=4096):
        data, _ = host_signal_fixture()
        data["hsa_host_memory_storage"] = {"system_page_size": 4096}
        table = data["domains"][0]["operations"]
        for name, cid, kind in (("hipHostRegister", 90, "host_registration"),
                                ("hipHostUnregister", 91, "host_unregistration")):
            op = len(table)
            table.append({"id": op, "name": name})
            for phase in (1, 2):
                sequence = len(data["events"]) + 1
                data["events"].append({"seq": sequence, "callback_seq": sequence, "ns": sequence * 10,
                    "domain": 0, "op": op, "cid": cid, "tid": 88, "epoch": 0, "ancestor": 0,
                    "phase": phase, "flags": 32, "status_type": 1 if phase == 2 else 0,
                    "status": 0 if phase == 2 else None,
                    "raw_hip_control_payload": {"kind": kind, "pointer": pointer,
                        "size_bytes": size if cid == 90 else 0, "flags": 0, "device_index": 0}})
        shutdown = {"process_id": data["pid"], "started_monotonic_raw_ns": data["events"][-2]["ns"],
                    "ended_monotonic_raw_ns": data["events"][-1]["ns"]}
        return data, shutdown

    def cpu_identity(self, data, shutdown):
        return _TOOL._cpu_backing_identity(data, None,
            {"host_base_address": 32768, "agent_base_address": 32768,
             "bytes": 4096, "owner_agent": 500}, 16, shutdown)

    def host_allocation_cleanup_fixture(self, *, pointer=65536, size=4100, native_size=None):
        data, _ = host_signal_fixture()
        while len(data["domains"]) <= 6:
            data["domains"].append({"operations": []})
        operations = {}
        for domain, names in ((0, ("hipHostMalloc", "hipHostFree")),
                (3, ("hsa_amd_memory_pool_allocate", "hsa_amd_memory_pool_free", "hsa_amd_agents_allow_access")),
                (6, ("MEMORY_ALLOCATION_ALLOCATE", "MEMORY_ALLOCATION_FREE"))):
            table = data["domains"][domain]["operations"]
            for name in names:
                operations[name] = domain, len(table)
                table.append({"id": len(table), "name": name})
        native_size = size if native_size is None else native_size

        def row(name, cid, phase, flags, payload=None, ancestor=0):
            domain, op = operations[name]
            sequence = len(data["events"]) + 1
            value = {"domain": domain, "op": op, "cid": cid, "tid": 88, "epoch": 0, "ancestor": ancestor,
                "seq": sequence, "callback_seq": sequence, "ns": sequence * 10, "phase": phase, "flags": flags,
                "status_type": (1 if domain == 0 else 2) if phase == 2 and domain != 6 else 0,
                "status": 0 if phase == 2 and domain != 6 else None}
            if payload is not None:
                field = "raw_hip_control_payload" if domain == 0 else "raw_allocation_payload" if domain == 6 else "raw_hsa_host_memory_payload"
                value[field] = deepcopy(payload)
            data["events"].append(value)

        allocated = {"kind": "host_allocation", "output_address": 8192, "size_bytes": size, "flags": 1,
            "pointer": None, "pointer_observed": False, "payload_observed": True}
        row("hipHostMalloc", 90, 1, 33, allocated)
        row("hsa_amd_memory_pool_allocate", 100, 1, 17, ancestor=90)
        normalized = {"address_or_handle": 0, "size_bytes": native_size, "agent_handle": 700, "success_proven": False}
        row("MEMORY_ALLOCATION_ALLOCATE", 100, 1, 1, normalized, ancestor=90)
        row("MEMORY_ALLOCATION_ALLOCATE", 100, 2, 1, {**normalized, "address_or_handle": pointer}, ancestor=90)
        row("hsa_amd_memory_pool_allocate", 100, 2, 17, ancestor=90)
        access = {"operation": "allow_access", "ptr": pointer, "agents_address": 12288, "num_agents": 1,
            "agents_observed": True, "agent_handles": [900001], "flags_address": 0, "flags_values": None,
            "flags_observed": True, "payload_observed": True, **{key: None for key in (
                "size_bytes", "pool_handle", "flags", "agent_ptr_output_address", "agent_alias")}}
        row("hsa_amd_agents_allow_access", 101, 1, 0, access, ancestor=90)
        row("hsa_amd_agents_allow_access", 101, 2, 0, access, ancestor=90)
        row("hipHostMalloc", 90, 2, 33, {**allocated, "pointer": pointer, "pointer_observed": True})
        shutdown_start = (len(data["events"]) + 1) * 10
        released = {"kind": "host_free", "pointer": pointer, "pointer_observed": True, "payload_observed": True,
                    "output_address": 0, "size_bytes": None, "flags": None}
        row("hipHostFree", 91, 1, 34, released)
        row("hsa_amd_memory_pool_free", 102, 1, 18, ancestor=91)
        normalized = {"address_or_handle": pointer, "size_bytes": 0, "agent_handle": 0, "success_proven": False}
        row("MEMORY_ALLOCATION_FREE", 102, 1, 2, normalized, ancestor=91)
        row("MEMORY_ALLOCATION_FREE", 102, 2, 2, normalized, ancestor=91)
        row("hsa_amd_memory_pool_free", 102, 2, 18, ancestor=91)
        row("hipHostFree", 91, 2, 34, released)
        data["agent_inventory"] = {"agents": [{"handle": 700, "type": 1, "type_name": "CPU"}]}
        data["hip_host_memory_payloads"] = {"schema_version": 1, "method": "existing_native_event_storage_v1",
            "supported_operations": ["hipHostMalloc", "hipHostFree"], "retained_records": 4,
            "safe_read_errors": 0, "uses_native_event_storage": True}
        data["hsa_host_memory_storage"] = {"schema_version": 1, "method": "bounded_public_hsa_host_memory_payloads_v1",
            "capacity": 8192, "agent_capacity": 4, "retained_records": 2, "record_size_bytes": 104,
            "storage_capacity_bytes": 8192 * 104, "dropped_records": 0, "read_errors": 0,
            "incomplete_agent_arrays": 0, "system_page_size": 4096,
            "supported_operations": ["hsa_amd_memory_unlock", "hsa_amd_memory_lock", "hsa_amd_memory_lock_to_pool", "hsa_amd_agents_allow_access"]}
        for key in ("retained_events", "attempted_events", "observed_callbacks"):
            data[key] = len(data["events"])
        return data, {"process_id": data["pid"], "started_monotonic_raw_ns": shutdown_start,
                      "ended_monotonic_raw_ns": data["events"][-1]["ns"]}

    def test_disjoint_actual_hip_host_generation_covers_cpu_native_storage_and_access_effects(self):
        data, shutdown = self.host_allocation_cleanup_fixture(native_size=8192)
        before = deepcopy(data)
        result = self.cpu_identity(data, shutdown)
        self.assertEqual(data, before)
        self.assertEqual(result["kind"], "CPU_QUERY_REGION")
        self.assertIsNone(result["allocation_generation"])
        self.assertIsNone(result["free_sequence"])
        scope = result["host_cleanup_noninterference"]
        proof = scope["exclusions"][0]
        self.assertEqual(proof["kind"], "HIP_HOST_ALLOCATION_RANGE_EXCLUSION")
        self.assertEqual(proof["requested_size_bytes"], 4100)
        self.assertEqual(proof["actual_native_size_bytes"], 8192)
        self.assertEqual(proof["sdk_cpu_agent"], 700)
        self.assertEqual(proof["allocation_effects"]["access_calls"][0]["cid"], 101)
        self.assertEqual(proof["free_effects"]["normalized_storage"]["cid"], 102)
        self.assertEqual(len(scope["allowed_callback_sequences"]), 14)

    def test_hip_host_cleanup_rejects_missing_mutated_untyped_or_wrong_generation_facts(self):
        for mode in ("missing_malloc", "missing_parent_ptr", "wrong_free_ptr", "failed_malloc", "failed_free", "untyped_free",
                     "changed_size", "changed_flags", "unread_output", "free_invented_size", "gpu_agent",
                     "wrong_hsa_ancestor", "failed_hsa_free", "wrong_native_free", "native_too_small",
                     "wrong_access_ptr", "untyped_access", "changed_access_agents", "truncated_access_agents",
                     "nonnull_reserved_flags", "invented_access_extent", "bad_count", "read_error"):
            data, shutdown = self.host_allocation_cleanup_fixture(native_size=4096 if mode == "native_too_small" else None)
            if mode == "missing_malloc":
                data["events"] = [row for row in data["events"] if row["cid"] != 90]
                data["hip_host_memory_payloads"]["retained_records"] = 2
            elif mode == "gpu_agent":
                data["agent_inventory"]["agents"][0].update(type=2, type_name="GPU")
            elif mode == "bad_count":
                data["hip_host_memory_payloads"]["retained_records"] -= 1
            elif mode == "read_error":
                data["hip_host_memory_payloads"]["safe_read_errors"] = 1
            for row in data["events"]:
                payload = row.get("raw_hip_control_payload", {})
                if row["cid"] == 90 and row["phase"] == 2:
                    if mode == "failed_malloc":
                        row["status"] = 1
                    elif mode == "changed_size":
                        payload["size_bytes"] += 1
                    elif mode == "changed_flags":
                        payload["flags"] += 1
                    elif mode == "unread_output":
                        payload["pointer_observed"] = False
                elif row["cid"] == 91:
                    if mode == "missing_parent_ptr":
                        payload.pop("pointer")
                    elif mode == "wrong_free_ptr":
                        payload["pointer"] += 4096
                    elif mode == "failed_free" and row["phase"] == 2:
                        row["status"] = 1
                    elif mode == "untyped_free" and row["phase"] == 2:
                        row.update(status_type=0, status=None)
                    elif mode == "free_invented_size":
                        payload["size_bytes"] = 4096
                elif row["cid"] == 102:
                    if mode == "wrong_hsa_ancestor" and row["domain"] == 3:
                        row["ancestor"] = 999
                    elif mode == "failed_hsa_free" and row["domain"] == 3 and row["phase"] == 2:
                        row["status"] = 1
                    elif mode == "wrong_native_free" and row["domain"] == 6:
                        row["raw_allocation_payload"]["address_or_handle"] += 4096
                elif row["cid"] == 101:
                    value = row["raw_hsa_host_memory_payload"]
                    if mode == "wrong_access_ptr":
                        value["ptr"] = 32768
                    elif mode == "untyped_access" and row["phase"] == 2:
                        row.update(status_type=0, status=None)
                    elif mode == "changed_access_agents" and row["phase"] == 2:
                        value["agent_handles"] = [999999]
                    elif mode == "truncated_access_agents":
                        value["num_agents"] = 5
                    elif mode == "nonnull_reserved_flags":
                        value["flags_address"] = 4096
                        value["flags_observed"] = False
                    elif mode == "invented_access_extent":
                        value["size_bytes"] = 4100
            with self.subTest(mode=mode), self.assertRaises(_TOOL.ResourceEvidenceError):
                self.cpu_identity(data, shutdown)

    def test_reused_hip_host_pointer_requires_its_own_complete_live_allocation_generation(self):
        data, shutdown = self.host_allocation_cleanup_fixture()
        second_group = deepcopy([row for row in data["events"] if row["cid"] >= 90])
        for row in second_group:
            row["cid"] += 200
            if row["ancestor"]:
                row["ancestor"] += 200
        data["events"].extend(second_group)

        def renumber():
            for index, row in enumerate(data["events"], 1):
                row.update(seq=index, callback_seq=index, ns=index * 10)
            for key in ("retained_events", "attempted_events", "observed_callbacks"):
                data[key] = len(data["events"])
            data["hip_host_memory_payloads"]["retained_records"] = sum(row["domain"] == 0 and row["cid"] >= 90 for row in data["events"])
            data["hsa_host_memory_storage"]["retained_records"] = sum("raw_hsa_host_memory_payload" in row for row in data["events"])
            shutdown["ended_monotonic_raw_ns"] = data["events"][-1]["ns"]

        renumber()
        result = self.cpu_identity(data, shutdown)
        proofs = result["host_cleanup_noninterference"]["exclusions"]
        self.assertEqual(len(proofs), 2)
        self.assertNotEqual(proofs[0]["hip_generation_id"], proofs[1]["hip_generation_id"])
        self.assertTrue(proofs[0]["native_storage_id"].endswith("generation1"))
        self.assertTrue(proofs[1]["native_storage_id"].endswith("generation2"))
        data["events"] = [row for row in data["events"] if row["cid"] != 290]
        renumber()
        with self.assertRaises(_TOOL.ResourceEvidenceError):
            self.cpu_identity(data, shutdown)

    def test_hip_host_cleanup_requires_disjoint_complete_pages_shutdown_and_known_nested_effects(self):
        for mode in ("overlap", "shared_page", "missing_shutdown", "wrong_process", "outside_window", "unknown_lower"):
            data, shutdown = self.host_allocation_cleanup_fixture(
                pointer=36863 if mode in ("overlap", "shared_page") else 65536,
                size=2 if mode == "overlap" else 1 if mode == "shared_page" else 4100)
            if mode == "missing_shutdown":
                shutdown = None
            elif mode == "wrong_process":
                shutdown["process_id"] += 1
            elif mode == "outside_window":
                shutdown["ended_monotonic_raw_ns"] -= 1
            elif mode == "unknown_lower":
                table = data["domains"][3]["operations"]
                op = len(table)
                table.append({"id": op, "name": "unknown_nested_host_effect"})
                insertion = next(i for i, row in enumerate(data["events"]) if row["cid"] == 90 and row["phase"] == 2)
                extra = [{"domain": 3, "op": op, "cid": 103, "tid": 88, "ancestor": 90,
                    "epoch": 0, "phase": phase, "flags": 0, "status_type": 0, "status": None} for phase in (1, 2)]
                data["events"][insertion:insertion] = extra
                for i, row in enumerate(data["events"], 1):
                    row.update(seq=i, callback_seq=i, ns=i * 10)
                for key in ("retained_events", "attempted_events", "observed_callbacks"):
                    data[key] = len(data["events"])
                shutdown["started_monotonic_raw_ns"] += 20
                shutdown["ended_monotonic_raw_ns"] += 20
            with self.subTest(mode=mode), self.assertRaises(_TOOL.ResourceEvidenceError):
                if mode == "shared_page":
                    _TOOL._cpu_backing_identity(data, None, {"host_base_address": 32769,
                        "agent_base_address": 32769, "bytes": 4094, "owner_agent": 500}, 16, shutdown)
                else:
                    self.cpu_identity(data, shutdown)

    def hsa_host_cleanup_fixture(self, *, lock_sizes=(4100, 4100), unlock_count=None,
                                 pointer=65536, registered_size=4100, lock_pointer=None):
        """Two agents may pin one host input, with no particular physical free pairing."""
        data, _ = host_signal_fixture()
        lock_pointer = pointer if lock_pointer is None else lock_pointer
        unlock_count = len(lock_sizes) if unlock_count is None else unlock_count
        operations = {}
        for domain, names in ((0, ("hipHostRegister", "hipHostUnregister")),
                (3, ("hsa_amd_memory_lock", "hsa_amd_memory_unlock", "hsa_amd_memory_lock_to_pool"))):
            table = data["domains"][domain]["operations"]
            for name in names:
                operations[name] = domain, len(table)
                table.append({"id": len(table), "name": name})

        def row(name, cid, phase, payload, *, ancestor=0):
            domain, op = operations[name]
            sequence = len(data["events"]) + 1
            field = "raw_hsa_host_memory_payload" if domain == 3 else "raw_hip_control_payload"
            data["events"].append({"seq": sequence, "callback_seq": sequence, "ns": sequence * 10,
                "domain": domain, "op": op, "cid": cid, "tid": 88, "epoch": 0, "ancestor": ancestor,
                "phase": phase, "flags": 32, "status_type": (2 if domain == 3 else 1) if phase == 2 else 0,
                "status": 0 if phase == 2 else None, field: deepcopy(payload)})

        registered = {"kind": "host_registration", "pointer": pointer, "size_bytes": registered_size,
                      "flags": 0, "device_index": 0}
        unregistered = {**registered, "kind": "host_unregistration", "size_bytes": 0}
        row("hipHostRegister", 90, 1, registered)
        for index, length in enumerate(lock_sizes):
            name = "hsa_amd_memory_lock" if index % 2 == 0 else "hsa_amd_memory_lock_to_pool"
            payload = {"operation": "lock" if index % 2 == 0 else "lock_to_pool", "host_ptr": lock_pointer,
                "payload_observed": True, "size_bytes": length, "agents_address": 8192 + index * 8,
                "num_agents": 1, "agents_observed": True, "agent_handles": [700 + index],
                "pool_handle": None if index % 2 == 0 else 900, "flags": None if index % 2 == 0 else 0,
                "agent_ptr_output_address": 12288 + index * 8, "agent_alias": None}
            row(name, 100 + index, 1, payload, ancestor=90)
            row(name, 100 + index, 2, {**payload, "agent_alias": lock_pointer}, ancestor=90)
        row("hipHostRegister", 90, 2, registered)
        shutdown_start = (len(data["events"]) + 1) * 10
        row("hipHostUnregister", 91, 1, unregistered)
        for index in range(unlock_count):
            payload = {"operation": "unlock", "host_ptr": lock_pointer, "payload_observed": True,
                **{key: None for key in ("size_bytes", "agents_address", "num_agents", "agents_observed",
                    "agent_handles", "pool_handle", "flags", "agent_ptr_output_address", "agent_alias")}}
            row("hsa_amd_memory_unlock", 110 + index, 1, payload, ancestor=91)
            row("hsa_amd_memory_unlock", 110 + index, 2, payload, ancestor=91)
        row("hipHostUnregister", 91, 2, unregistered)
        count = sum("raw_hsa_host_memory_payload" in event for event in data["events"])
        data["hsa_host_memory_storage"] = {"schema_version": 1,
            "method": "bounded_public_hsa_host_memory_payloads_v1", "capacity": 8192, "agent_capacity": 4,
            "retained_records": count, "record_size_bytes": 104, "storage_capacity_bytes": 8192 * 104,
            "dropped_records": 0, "read_errors": 0, "incomplete_agent_arrays": 0, "system_page_size": 4096}
        for key in ("retained_events", "attempted_events", "observed_callbacks"):
            data[key] = len(data["events"])
        return data, {"process_id": data["pid"], "started_monotonic_raw_ns": shutdown_start,
                      "ended_monotonic_raw_ns": data["events"][-1]["ns"]}

    def test_nested_repeated_actual_hsa_locks_preserve_only_disjoint_cpu_region(self):
        data, shutdown = self.hsa_host_cleanup_fixture(lock_sizes=(4100, 8192))
        before = deepcopy(data)
        result = self.cpu_identity(data, shutdown)
        self.assertEqual(result["kind"], "CPU_QUERY_REGION")
        self.assertIsNone(result["allocation_generation"])
        self.assertIsNone(result["free_sequence"])
        scope = result["host_cleanup_noninterference"]
        self.assertIn("identity mappings only", scope["hsa_alias_support"])
        self.assertIn("no physical unlock/free", scope["claim"])
        self.assertEqual(len(scope["allowed_callback_sequences"]), 12)
        self.assertEqual(data, before)
        # An aligned-down real HSA input is admissible only inside the actual
        # parent registration's page extent, independently of requested bytes.
        data, shutdown = self.hsa_host_cleanup_fixture(pointer=65537, lock_pointer=65536)
        self.assertEqual(self.cpu_identity(data, shutdown)["kind"], "CPU_QUERY_REGION")

    def test_hsa_unlock_requires_all_lock_references_and_actual_parent_generation(self):
        for mode in ("missing_lock", "missing_unlock", "extra_unlock", "wrong_lock_ancestor",
                     "wrong_unlock_ancestor", "wrong_thread", "outside_parent", "wrong_unlock_pointer"):
            data, shutdown = self.hsa_host_cleanup_fixture(
                lock_sizes=() if mode == "missing_lock" else (4100, 4100),
                unlock_count=1 if mode in ("missing_lock", "missing_unlock") else 3 if mode == "extra_unlock" else 2)
            for event in data["events"]:
                if mode in ("wrong_lock_ancestor", "wrong_unlock_ancestor", "wrong_thread", "outside_parent"):
                    target = 110 if mode == "wrong_unlock_ancestor" else 100
                    if event["cid"] == target:
                        if mode.endswith("ancestor"):
                            event["ancestor"] = 999
                        elif mode == "wrong_thread":
                            event["tid"] += 1
                        else:
                            event["ns"] = 1
                if mode == "wrong_unlock_pointer" and event["cid"] == 110:
                    event["raw_hsa_host_memory_payload"]["host_ptr"] += 4096
            with self.subTest(mode=mode), self.assertRaises(_TOOL.ResourceEvidenceError):
                self.cpu_identity(data, shutdown)

    def test_reused_host_pointer_cannot_borrow_an_old_completed_lock_group(self):
        first, shutdown = self.hsa_host_cleanup_fixture(lock_sizes=(4100,))
        second, _ = self.hsa_host_cleanup_fixture(lock_sizes=(8192,))
        second_group = deepcopy([row for row in second["events"] if row["cid"] >= 90])
        for row in second_group:
            row["cid"] += 100
            if row["ancestor"]:
                row["ancestor"] += 100
        first["events"].extend(second_group)

        def renumber(data):
            for index, row in enumerate(data["events"], 1):
                row.update(seq=index, callback_seq=index, ns=index * 10)
            data["hsa_host_memory_storage"]["retained_records"] = sum(
                "raw_hsa_host_memory_payload" in row for row in data["events"])
            for key in ("retained_events", "attempted_events", "observed_callbacks"):
                data[key] = len(data["events"])
            shutdown["ended_monotonic_raw_ns"] = data["events"][-1]["ns"]

        renumber(first)
        self.assertEqual(self.cpu_identity(first, shutdown)["kind"], "CPU_QUERY_REGION")
        # The original first generation is fully observed and closed. A later
        # successful unlock with identical pointer needs its own actual lock.
        first["events"] = [row for row in first["events"] if row["cid"] != 200]
        renumber(first)
        with self.assertRaises(_TOOL.ResourceEvidenceError):
            self.cpu_identity(first, shutdown)

    def test_hsa_extent_inputs_success_and_sparse_storage_must_be_complete(self):
        for mode in ("changed_extent", "failed_lock", "failed_unlock", "untyped_unlock", "missing_payload",
                     "invented_unlock_size", "unread_input", "changed_agents", "truncated_agents",
                     "out_of_parent_range", "dropped", "read_error", "array_error", "count", "capacity"):
            data, shutdown = self.hsa_host_cleanup_fixture()
            for event in data["events"]:
                if event["cid"] == 100:
                    payload = event["raw_hsa_host_memory_payload"]
                    if mode == "changed_extent" and event["phase"] == 2:
                        payload["size_bytes"] += 1
                    elif mode == "failed_lock" and event["phase"] == 2:
                        event["status"] = 4097
                        payload["agent_alias"] = None
                    elif mode == "unread_input":
                        payload["payload_observed"] = False
                    elif mode == "changed_agents" and event["phase"] == 2:
                        payload["agent_handles"] = [999]
                    elif mode == "truncated_agents":
                        payload["num_agents"] = 5
                    elif mode == "out_of_parent_range":
                        payload["size_bytes"] = 12288
                elif event["cid"] == 110 and event["phase"] == 2:
                    if mode == "failed_unlock":
                        event["status"] = 4097
                    elif mode == "untyped_unlock":
                        event["status_type"] = 0
                    elif mode == "missing_payload":
                        event.pop("raw_hsa_host_memory_payload")
                    elif mode == "invented_unlock_size":
                        event["raw_hsa_host_memory_payload"]["size_bytes"] = 4096
            table = data["hsa_host_memory_storage"]
            for candidate, field in (("dropped", "dropped_records"), ("read_error", "read_errors"),
                    ("array_error", "incomplete_agent_arrays"), ("count", "retained_records"), ("capacity", "capacity")):
                if mode == candidate:
                    table[field] += 1
            with self.subTest(mode=mode), self.assertRaises(_TOOL.ResourceEvidenceError):
                self.cpu_identity(data, shutdown)

    def test_hsa_page_expansion_and_all_same_pointer_candidate_ranges_are_checked(self):
        for sizes in ((4096, 36864), (36864, 4096)):
            data, shutdown = self.hsa_host_cleanup_fixture(pointer=1, registered_size=4096, lock_sizes=sizes)
            with self.subTest(sizes=sizes), self.assertRaises(_TOOL.ResourceEvidenceError):
                self.cpu_identity(data, shutdown)
        data, shutdown = self.hsa_host_cleanup_fixture(pointer=32767, registered_size=1, lock_sizes=(1, 1))
        with self.assertRaises(_TOOL.ResourceEvidenceError):
            _TOOL._cpu_backing_identity(data, None, {"host_base_address": 32766, "agent_base_address": 32766,
                "bytes": 1, "owner_agent": 500}, 16, shutdown)

    def test_different_hsa_alias_namespaces_are_unsupported_even_when_numbers_are_disjoint(self):
        for alias in (32768, 1048576):
            data, shutdown = self.hsa_host_cleanup_fixture()
            next(event for event in data["events"] if event["cid"] == 100 and event["phase"] == 2)[
                "raw_hsa_host_memory_payload"]["agent_alias"] = alias
            with self.subTest(alias=alias), self.assertRaises(_TOOL.ResourceEvidenceError):
                self.cpu_identity(data, shutdown)
        for agent_base in (None, 1048576):
            data, shutdown = self.hsa_host_cleanup_fixture()
            with self.subTest(agent_base=agent_base), self.assertRaises(_TOOL.ResourceEvidenceError):
                _TOOL._cpu_backing_identity(data, None, {"host_base_address": 32768,
                    "agent_base_address": agent_base, "bytes": 4096, "owner_agent": 500}, 16, shutdown)

    def test_exact_disjoint_host_unregister_preserves_cpu_query_region(self):
        data, shutdown = self.host_cleanup_fixture()
        before = deepcopy(data)
        result = self.cpu_identity(data, shutdown)
        self.assertEqual(result["kind"], "CPU_QUERY_REGION")
        self.assertIsNone(result["allocation_generation"])
        self.assertEqual(data, before)

    def test_host_unregister_one_byte_overlap_or_missing_shutdown_cannot_be_ignored(self):
        data, shutdown = self.host_cleanup_fixture(pointer=32768 + 4095, size=2)
        with self.assertRaises(_TOOL.ResourceEvidenceError):
            self.cpu_identity(data, shutdown)
        data, shutdown = self.host_cleanup_fixture()
        for bad in (None, {**shutdown, "process_id": 99},
                    {**shutdown, "ended_monotonic_raw_ns": shutdown["started_monotonic_raw_ns"]}):
            with self.subTest(shutdown=bad), self.assertRaises(_TOOL.ResourceEvidenceError):
                self.cpu_identity(data, bad)

    def test_unregistered_pointer_mutated_arguments_or_failed_host_cleanup_is_rejected(self):
        for change in ("missing_registration", "wrong_pointer", "changed_argument", "failed_status"):
            data, shutdown = self.host_cleanup_fixture()
            if change == "missing_registration":
                data["events"] = [r for r in data["events"] if r["cid"] != 90]
            elif change == "wrong_pointer":
                for row in data["events"]:
                    if row["cid"] == 91:
                        row["raw_hip_control_payload"]["pointer"] += 1
            elif change == "changed_argument":
                data["events"][-1]["raw_hip_control_payload"]["pointer"] += 1
            else:
                data["events"][-1]["status"] = 1
            with self.subTest(change=change), self.assertRaises(_TOOL.ResourceEvidenceError):
                self.cpu_identity(data, shutdown)

    def test_host_cleanup_requires_actual_page_granule_and_rejects_shared_page(self):
        for page in (None, 0, True, 3000, 2**31):
            data, shutdown = self.host_cleanup_fixture()
            data["hsa_host_memory_storage"]["system_page_size"] = page
            with self.subTest(page=page), self.assertRaises(_TOOL.ResourceEvidenceError):
                self.cpu_identity(data, shutdown)
        data, shutdown = self.host_cleanup_fixture(pointer=32768, size=1)
        with self.assertRaises(_TOOL.ResourceEvidenceError):
            _TOOL._cpu_backing_identity(data, None,
                {"host_base_address": 32769, "bytes": 4094, "owner_agent": 500}, 16, shutdown)

    def host_audit(self, data, evidence):
        before = deepcopy((data, evidence))
        result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                                                 host_signal_evidence=evidence)
        self.assertEqual((data, evidence), before)
        return result

    def test_stream_owned_live_signal_has_separate_scope_and_exact_cpu_backing_proof(self):
        data, envelope, shutdown = worker_host_fixture()
        stream_owned_signal(data)
        before = deepcopy((data, envelope, shutdown))
        result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
            natural_shutdown_evidence=shutdown, host_signal_evidence=envelope)
        self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
        self.assertEqual((data, envelope, shutdown), before)
        signal = result["signal_generations"][0]
        self.assertEqual(signal["parent_scope"], "HIP_STREAM_SYNCHRONIZE_INTERNAL_HSA_SIGNAL")
        self.assertIsNone(signal["parent_event_generation_id"])
        self.assertIsNone(signal["destroy"])
        self.assertEqual(signal["parent_call"]["cid"], 7)
        self.assertEqual(signal["parent_stream_binding"]["operation"], "hipEventRecord")
        self.assertEqual(signal["completion_fence"]["cid"], 11)
        self.assertEqual(signal["host_backing_evidence"]["backing"]["kind"], "CPU_QUERY_REGION")
        proof = [value for value in result["allowed_call_proofs"] if value["cid"] == 8]
        self.assertEqual(len(proof), 1)
        self.assertEqual(proof[0]["operation"], "hsa_signal_create")
        self.assertEqual(proof[0]["completion_fence_cid"], 11)
        self.assertEqual(len(result["event_generations"]), 2)

    def test_stream_owned_signal_requires_actual_parent_scope_stream_success_and_device(self):
        for mode in ("changed_stream", "unknown_stream", "invalid_stream", "wait_only_stream", "wrong_device",
                     "wrong_thread", "wrong_epoch", "not_contained", "failed_parent", "untyped_parent", "wrong_kind"):
            data, envelope, shutdown = worker_host_fixture()
            stream_owned_signal(data, stream=102 if mode == "wait_only_stream" else 101)
            for row in data["events"]:
                if row["cid"] != 7:
                    continue
                payload = row["raw_hip_control_payload"]
                if mode == "changed_stream" and row["phase"] == 2:
                    payload["stream_handle"] += 1
                elif mode == "unknown_stream":
                    payload["stream_handle"] = 999
                elif mode == "invalid_stream":
                    payload["stream_handle"] = True
                elif mode == "wrong_device":
                    payload["device_index"] = 1
                elif mode == "wrong_thread":
                    row["tid"] += 1
                elif mode == "wrong_epoch":
                    row["epoch"] = 0
                elif mode == "not_contained" and row["phase"] == 2:
                    row["ns"] = 140
                elif mode == "failed_parent" and row["phase"] == 2:
                    row["status"] = 1
                elif mode == "untyped_parent" and row["phase"] == 2:
                    row["status_type"] = 0
                elif mode == "wrong_kind":
                    payload["kind"] = "device_sync"
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                natural_shutdown_evidence=shutdown, host_signal_evidence=envelope)
            with self.subTest(mode=mode):
                self.assertEqual(result["status"], "PARTIAL", result)
                self.assertEqual(result["allowed_call_proofs"], [])

    def test_stream_owned_signal_cannot_use_unknown_ancestor_or_unbound_later_fence(self):
        data, envelope, shutdown = worker_host_fixture()
        stream_owned_signal(data)
        for row in data["events"]:
            if row["cid"] == 8:
                row["ancestor"] = 999
        result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
            natural_shutdown_evidence=shutdown, host_signal_evidence=envelope)
        self.assertNotIn(8, {value["cid"] for value in result["allowed_call_proofs"]})
        data, envelope, shutdown = worker_host_fixture()
        stream_owned_signal(data)
        # All actual device-fence callbacks are still present, but the original
        # worker source binding is absent. A parent label cannot substitute.
        result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
            host_signal_evidence=envelope["capture"])
        self.assertEqual(result["status"], "PARTIAL", result)
        self.assertEqual(result["allowed_call_proofs"], [])

    def test_current_target_device_and_cross_device_wait_do_not_establish_stream_ownership(self):
        for mode in ("wait_only", "foreign_record"):
            data, envelope, shutdown = worker_host_fixture()
            stream = 102 if mode == "wait_only" else 202
            stream_owned_signal(data, stream=stream)

            def append_pair(name, cid, tid, payload, output=None):
                operations = data["domains"][0]["operations"]
                op = next(row["id"] for row in operations if row["name"] == name)
                for phase in (1, 2):
                    sequence = len(data["events"]) + 1
                    data["events"].append({"domain": 0, "op": op, "cid": cid, "tid": tid,
                        "seq": sequence, "callback_seq": sequence, "ns": 600 + sequence,
                        "epoch": 0, "ancestor": 0, "phase": phase,
                        "flags": 64 if name == "hipEventCreateWithFlags" else 0,
                        "status_type": 1 if phase == 2 else 0, "status": 0 if phase == 2 else None,
                        "raw_hip_control_payload": deepcopy(output if phase == 2 and output is not None else payload)})

            if mode == "wait_only":
                # A successful wait uses the existing event but cannot establish
                # that its opaque stream belongs to the current device.
                append_pair("hipStreamWaitEvent", 60, 77, {"kind": "event", "object_handle": 10,
                    "stream_handle": stream, "flags": 0, "device_index": 0, "device_source_cid": 1})
                expected = "no prior successful target-device event record"
            else:
                # A different thread records an independently device-1 event on
                # this stream while the parent thread's actual context stays 0.
                append_pair("hipSetDevice", 60, 88, {"kind": "device_set", "device_index": 1})
                creation = {"kind": "event", "object_handle": None, "output_address": 900,
                    "stream_handle": 0, "flags": 2, "device_index": 1, "device_source_cid": 60}
                append_pair("hipEventCreateWithFlags", 61, 88, creation, {**creation, "object_handle": 99})
                append_pair("hipEventRecord", 62, 88, {"kind": "event", "object_handle": 99,
                    "stream_handle": stream, "device_index": 1, "device_source_cid": 60})
                expected = "actual parent stream device"
            for key in ("retained_events", "attempted_events", "observed_callbacks"):
                data[key] = len(data["events"])
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                natural_shutdown_evidence=shutdown, host_signal_evidence=envelope)
            with self.subTest(mode=mode):
                self.assertEqual(result["status"], "PARTIAL", result)
                self.assertEqual(result["allowed_call_proofs"], [])
                self.assertIn(expected, result["issues"][0]["message"])

    def test_stream_owned_signal_replays_nonzero_event_id_through_independent_kfd_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            data, envelope, shutdown = hardware_worker_fixture(directory)
            stream_owned_signal(data)
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                natural_shutdown_evidence=shutdown, host_signal_evidence=envelope)
            self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
            signal = result["signal_generations"][0]
            self.assertEqual(signal["parent_scope"], "HIP_STREAM_SYNCHRONIZE_INTERNAL_HSA_SIGNAL")
            self.assertEqual(signal["host_backing_evidence"]["hardware_event_backing"]["event_id"], 7)
            for row in data["events"]:
                if row["cid"] in (22, 25) and row["phase"] == 2:
                    row["raw_hsa_signal_event_payload"]["event_id"] = 8
            for sample in envelope["capture"]["records"][0]["samples"]:
                for observed in sample["api_calls"]:
                    if observed["api"] == "hsa_amd_signal_get_event_id":
                        observed["output"]["event_id"] = 8
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                natural_shutdown_evidence=shutdown, host_signal_evidence=envelope)
            self.assertEqual(result["status"], "PARTIAL", result)
            self.assertEqual(result["allowed_call_proofs"], [])

    def test_stream_record_owner_cannot_cross_destroy_recreate_or_unknown_lifecycle(self):
        for mode in ("same_handle", "missing_handle", "unknown_status", "failed_status", "context_reset"):
            data, envelope, shutdown = worker_host_fixture()
            stream_owned_signal(data)
            for row in data["events"]:
                if row["cid"] == 3:
                    row["epoch"] = 0
                elif row["cid"] == 6:
                    row["raw_hip_control_payload"]["stream_handle"] = 103
            data["epochs"][0]["begin_ns"] = 65
            insert_stream_lifecycle(data, envelope,
                handle=None if mode == "missing_handle" else 101,
                status_type=0 if mode == "unknown_status" else 1,
                status=None if mode == "unknown_status" else 1 if mode == "failed_status" else 0,
                operations=("hipDeviceReset",) if mode == "context_reset" else ("hipStreamDestroy", "hipStreamCreate"))
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                natural_shutdown_evidence=shutdown, host_signal_evidence=envelope)
            with self.subTest(mode=mode):
                self.assertEqual(result["status"], "PARTIAL", result)
                self.assertEqual(result["allowed_call_proofs"], [])
                self.assertIn("invalidated by unresolved stream/context lifecycle", result["issues"][0]["message"])

    def test_precise_other_stream_or_lifecycle_outside_owner_interval_does_not_invalidate_record(self):
        for mode in ("other_handle", "before_record", "after_parent", "fresh_record"):
            data, envelope, shutdown = worker_host_fixture()
            stream_owned_signal(data)
            for row in data["events"]:
                if row["cid"] == 3:
                    row["epoch"] = 0
                elif row["cid"] == 6 and mode != "fresh_record":
                    row["raw_hip_control_payload"]["stream_handle"] = 103
            data["epochs"][0]["begin_ns"] = 65
            cut, started = (4, 41) if mode == "before_record" else (16, 161) if mode == "after_parent" else (6, 61)
            insert_stream_lifecycle(data, envelope, cut=cut, first_ns=started,
                handle=202 if mode == "other_handle" else 101)
            before = deepcopy((data, envelope))
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                natural_shutdown_evidence=shutdown, host_signal_evidence=envelope)
            with self.subTest(mode=mode):
                self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
                self.assertEqual((data, envelope), before)
                signal = result["signal_generations"][0]
                self.assertEqual(signal["parent_stream_binding"]["cid"], 6 if mode == "fresh_record" else 3)
                excluded = signal["parent_stream_lifecycle"]["precise_other_stream_operations"]
                self.assertEqual(len(excluded), 2 if mode == "other_handle" else 0)

    def test_positive_cpu_query_region_is_live_and_does_not_invent_allocation_birth_or_free(self):
        data, evidence = host_signal_fixture()
        self.partial(data)
        result = self.host_audit(data, evidence)
        self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
        signal = result["signal_generations"][0]
        self.assertEqual(signal["state"], "LIVE_CPU_BACKED_ABI_SIGNAL_RESOURCE")
        self.assertIsNone(signal["destroy"])
        self.assertEqual(signal["host_backing_evidence"]["abi_span_bytes"], 64)
        backing = signal["host_backing_evidence"]["backing"]
        self.assertEqual(backing["kind"], "CPU_QUERY_REGION")
        for key in ("allocation_generation", "born_sequence", "free_sequence"):
            self.assertIsNone(backing[key])
        self.assertEqual(result["observed_cpu_backing_bytes"], 4096)
        self.assertIsNone(result["gpu_storage_bytes"])
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["hardware_validated"])
        self.assertNotIn("hsa_signal_destroy", {row["operation"] for row in result["allowed_call_proofs"]})

    def test_unknown_truncated_gpu_owner_hw_event_wrong_span_or_untyped_query_cannot_admit_live_resource(self):
        for payload_name, mutate in (
            ("raw_hsa_pointer_info_payload", lambda p: p.update(type=0)),
            ("raw_hsa_pointer_info_payload", lambda p: p.update(returned_size=24)),
            ("raw_hsa_pointer_info_payload", lambda p: p.update(host_base_address=32769)),
            ("raw_hsa_pointer_info_payload", lambda p: p.update(size_bytes=63)),
            ("raw_hsa_pointer_info_payload", lambda p: p.update(size_bytes=2**64 - 1)),
            ("raw_hsa_agent_info_payload", lambda p: p.update(device_type=1)),
            ("raw_hsa_signal_event_payload", lambda p: p.update(event_id=3)),
            ("raw_hsa_signal_event_payload", lambda p: p.update(signal_handle=64)),
        ):
            data, evidence = host_signal_fixture()
            payload = next(row[payload_name] for row in data["events"] if row["phase"] == 2 and payload_name in row)
            mutate(payload)
            result = self.host_audit(data, evidence)
            self.assertEqual(result["status"], "PARTIAL", (payload_name, result))
            self.assertEqual(result["allowed_call_proofs"], [])
        data, evidence = host_signal_fixture()
        next(row for row in data["events"] if row["cid"] == 20 and row["phase"] == 2)["status_type"] = 0
        self.assertEqual(self.host_audit(data, evidence)["status"], "PARTIAL")

    def test_host_proof_requires_stable_generation_actual_fence_and_bound_control(self):
        for mutate in (
            lambda d, e: e.update(process_id=78),
            lambda d, e: e["after_live"]["generations"][0].update(generation=2),
            lambda d, e: e["records"][0].update(create_cid=9),
            lambda d, e: e["records"][0].update(samples=e["records"][0]["samples"][:1]),
            lambda d, e: e["before_live"].update(event_cursor=0),
            lambda d, e: e["after_live"].update(snapshot_monotonic_raw_ns=1001),
            lambda d, e: e["abi"]["headers"][0].update(sha256="0" * 64),
            lambda d, e: e.pop("ordinary_signal_control"),
            lambda d, e: e["ordinary_signal_control"]["receipt"]["result"].update(destroy_status=1),
            lambda d, e: e["ordinary_signal_control"].update(runtime_dsos=[]),
            lambda d, e: d["hsa_query_storage"].update(dropped_records=1),
            lambda d, e: d["hsa_query_storage"].update(pointer_info_struct_size=128),
        ):
            data, evidence = host_signal_fixture()
            mutate(data, evidence)
            result = self.host_audit(data, evidence)
            self.assertEqual(result["status"], "PARTIAL", result)
            self.assertEqual(result["allowed_call_proofs"], [])

    def test_nonformal_migration_or_unknown_unmap_is_not_hidden_by_cpu_query_owner(self):
        for name, flags in (("hsa_amd_memory_migrate", 0), ("hipMemUnmap", 4),
                            ("hsa_amd_memory_unlock", 32), ("unknown_memory_operation", 256)):
            data, evidence = host_signal_fixture()
            table = data["domains"][3]["operations"]
            op = len(table)
            table.append({"id": op, "name": name})
            for phase in (1, 2):
                sequence = len(data["events"]) + 1
                data["events"].append({"seq": sequence, "callback_seq": sequence, "ns": sequence * 10,
                    "domain": 3, "op": op, "cid": 90, "tid": 77, "epoch": 0, "ancestor": 0,
                    "phase": phase, "flags": flags, "status_type": 2 if phase == 2 else 0,
                    "status": 0 if phase == 2 else None})
            for key in ("retained_events", "attempted_events", "observed_callbacks"):
                data[key] = len(data["events"])
            result = self.host_audit(data, evidence)
            self.assertEqual(result["status"], "PARTIAL", result)
            self.assertEqual(result["allowed_call_proofs"], [])

    def test_public_provider_windows_are_uniquely_joined_to_actual_sdk_cids(self):
        data, evidence = public_provider_fixture()
        result = self.host_audit(data, evidence)
        self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
        self.assertEqual(result["signal_generations"][0]["host_backing_evidence"]["samples"][0]["calls"][0]["cid"], 20)
        for mutate in (
            lambda e: e["records"][0]["samples"][0]["api_calls"][0].update(before_cursor=999),
            lambda e: e["records"][0]["samples"][0]["api_calls"][0]["output"].update(type=0),
            lambda e: e["records"][0]["samples"][0]["api_calls"][0]["api_binding"].update(module_path="/different/runtime.so"),
            lambda e: e["records"][0]["samples"][0].update(thread_id=78),
            lambda e: e["ordinary_signal_control"]["destroy_call"]["output"].update(signal_handle=65537),
        ):
            bad = deepcopy(evidence)
            mutate(bad)
            result = self.host_audit(data, bad)
            self.assertEqual(result["status"], "PARTIAL", result)
            self.assertEqual(result["allowed_call_proofs"], [])

    def test_actual_compiler_file_and_frozen_sibling_may_be_distinct_same_byte_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            data, evidence = public_provider_fixture()
            content = b"// same independently verified compiled and frozen source\n"
            built = Path(directory) / "hsa_signal_backing_probe.actual-build.cpp"
            frozen = Path(directory) / "hsa_signal_backing_probe.cpp"
            for path in (built, frozen):
                path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            actual = {"path": str(built), "sha256": digest, "size_bytes": len(content)}
            expected = {"path": str(frozen), "sha256": digest, "size_bytes": len(content)}
            evidence["compiler_source"] = actual
            evidence["sources"] = [expected]
            evidence["initialization"]["compiled_source_path"] = str(built)
            result = self.host_audit(data, evidence)
            self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
            self.assertNotEqual(actual["path"], expected["path"])
            for mutate in (
                lambda e: e["sources"][0].update(path=str(built), sha256="f" * 64),
                lambda e: e["compiler_source"].update(sha256="f" * 64),
                lambda e: e["sources"][0].update(size_bytes=len(content) + 1),
                lambda e: e["initialization"].update(compiled_source_path=str(frozen)),
            ):
                bad = deepcopy(evidence)
                mutate(bad)
                rejected = self.host_audit(data, bad)
                self.assertEqual(rejected["status"], "PARTIAL", rejected)
                self.assertEqual(rejected["allowed_call_proofs"], [])

    def test_tracked_cpu_backing_uses_real_allocator_generation_and_independent_agent_namespaces(self):
        data, evidence = tracked_cpu_fixture()
        result = self.host_audit(data, evidence)
        self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
        backing = result["observed_cpu_backing_regions"][0]
        self.assertEqual(backing["kind"], "TRACKED_CPU_ALLOCATION")
        self.assertEqual(backing["allocation_generation"], 1)
        self.assertEqual(backing["born_sequence"], 3)
        self.assertEqual(backing["owner_agent"], 500)  # Runtime HSA handle.
        self.assertEqual(backing["sdk_agent_handle"], 700)  # Independent rocprofiler agent ID.
        self.assertNotIn("hsa_amd_memory_pool_allocate", {row["operation"] for row in result["allowed_call_proofs"]})
        for mutate in (
            lambda d, e: e["records"][0]["backing"].update(allocation_cid=101),
            lambda d, e: e["records"][0]["backing"].update(normalized_allocate_exit_sequence=4),
            lambda d, e: d["agent_inventory"]["agents"][0].update(type=2, type_name="GPU"),
            lambda d, e: d["events"][2]["raw_allocation_payload"].update(success_proven=True),
            lambda d, e: d["events"][2]["raw_allocation_payload"].update(size_bytes=8192),
        ):
            bad_data, bad_evidence = deepcopy((data, evidence))
            mutate(bad_data, bad_evidence)
            bad = self.host_audit(bad_data, bad_evidence)
            self.assertEqual(bad["status"], "PARTIAL", bad)
            self.assertEqual(bad["allowed_call_proofs"], [])

    def test_an_actual_free_of_a_live_signal_region_after_queries_still_invalidates_admission(self):
        data, evidence = tracked_cpu_fixture()
        ext = data["domains"][3]["operations"]
        free_op = len(ext)
        ext.append({"id": free_op, "name": "hsa_amd_memory_pool_free"})
        data["domains"][6]["operations"].append({"id": 1, "name": "MEMORY_ALLOCATION_FREE"})
        for domain, operation, phase in ((3, free_op, 1), (6, 1, 1), (6, 1, 2), (3, free_op, 2)):
            sequence = len(data["events"]) + 1
            row: dict = {"seq": sequence, "callback_seq": sequence, "ns": sequence * 10,
                "domain": domain, "op": operation, "cid": 101, "tid": 77, "phase": phase,
                "epoch": 0, "ancestor": 0, "flags": 2, "status_type": 2 if domain == 3 and phase == 2 else 0,
                "status": 0 if domain == 3 and phase == 2 else None}
            if domain == 6:
                row["raw_allocation_payload"] = {"address_or_handle": 32768, "size_bytes": 0,
                    "agent_handle": 0, "success_proven": False}
            data["events"].append(row)
        for key in ("retained_events", "attempted_events", "observed_callbacks"):
            data[key] = len(data["events"])
        for locator in (True, False):
            bad = deepcopy(evidence)
            if not locator:
                bad["records"][0].pop("backing")
            result = self.host_audit(data, bad)
            self.assertEqual(result["status"], "PARTIAL", result)
            self.assertEqual(result["allowed_call_proofs"], [])

    def test_malformed_host_records_fail_closed_without_changing_raw_evidence(self):
        for mutate in (
            lambda e: e.update(records=[None]), lambda e: e.update(before_live=[]),
            lambda e: e.update(ordinary_signal_control=True),
            lambda e: e["abi"].update(amd_signal_size=float("nan")),
        ):
            data, evidence = host_signal_fixture()
            mutate(evidence)
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                                                     host_signal_evidence=evidence)
            self.assertEqual(result["status"], "PARTIAL", result)
            self.assertEqual(result["allowed_call_proofs"], [])

    def test_not_applicable_provider_preserves_closed_rule_only_if_raw_has_no_live_formal_signal(self):
        data = resource_fixture()
        data["pid"] = 77
        export = {"schema_version": 1, "kind": "observed_live_hsa_signal_generations", "pid": 77,
            "complete_workspace": False, "hardware_validated": False, "dropped_events": 0,
            "reused_live_handle_count": 0, "unknown_destroy_count": 0,
            "snapshot_monotonic_raw_ns": 225, "event_cursor": 22, "generations": []}
        evidence = {"schema_version": 1, "kind": "public_hsa_signal_backing_capture", "process_id": 77,
            "status": "NO_LIVE_FORMAL_TARGETS", "issues": [], "records": [], "ordinary_signal_control": None,
            "before_live": export, "after_live": deepcopy(export)}
        result = self.host_audit(data, evidence)
        self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
        live, host = host_signal_fixture()
        evidence.update(before_live=host["before_live"], after_live=host["after_live"])
        bad = self.host_audit(live, evidence)
        self.assertEqual(bad["status"], "PARTIAL", bad)
        self.assertEqual(bad["allowed_call_proofs"], [])

    def test_worker_envelope_replays_source_fence_and_original_a_b_c_logical_generations(self):
        data, envelope, shutdown = worker_host_fixture()
        before = deepcopy((data, envelope, shutdown))
        result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
            host_signal_evidence=envelope, natural_shutdown_evidence=shutdown)
        self.assertEqual((data, envelope, shutdown), before)
        self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
        self.assertEqual(result["worker_host_evidence_binding"]["completion_fence"]["cid"], 11)
        self.assertTrue(result["worker_host_evidence_binding"]["post_shutdown_registry_observed"])
        self.assertEqual(result["signal_generations"][0]["state"], "LIVE_CPU_BACKED_ABI_SIGNAL_RESOURCE")
        envelope["kfd_event_evidence"] = {"final_report_artifact": {"path": "/diagnostic/unused.json"},
            "final_snapshot_observation": {"status": "DIAGNOSTIC_ONLY_NOT_USED_FOR_EVENT_ID_ZERO"}}
        unaffected = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
            host_signal_evidence=envelope, natural_shutdown_evidence=shutdown)
        self.assertEqual(unaffected["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", unaffected["issues"])

    def test_worker_envelope_cannot_replace_actual_fence_or_hide_changed_generation_and_scope(self):
        for mutate in (
            lambda d, e, s: e["capture"].update(source_scope={}),
            lambda d, e, s: e["source_fence"].update(native_cursor_before=0, native_cursor_after=2),
            lambda d, e, s: e["source_scope"].update(formal_native_epochs=[2]),
            lambda d, e, s: e["after_natural_shutdown_live"]["observed_live"]["generations"][0].update(create_cid=9),
            lambda d, e, s: e["after_natural_shutdown_live"].update(new_signal_queries_performed=True),
            lambda d, e, s: e["after_natural_shutdown_live"]["observed_live"].update(snapshot_monotonic_raw_ns=499),
            lambda d, e, s: e["capture"]["before_live"].update(unknown_destroy_count=1),
            lambda d, e, s: e["source_fence"]["source"].update(file_sha256="f" * 64),
            lambda d, e, s: e.update(query_started_monotonic_raw_ns=180),
        ):
            data, envelope, shutdown = worker_host_fixture()
            mutate(data, envelope, shutdown)
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                host_signal_evidence=envelope, natural_shutdown_evidence=shutdown)
            self.assertEqual(result["status"], "PARTIAL", result)
            self.assertEqual(result["allowed_call_proofs"], [])

    def test_nonzero_hardware_event_requires_positive_cpu_body_and_independently_replayed_gtt_page(self):
        with tempfile.TemporaryDirectory() as directory:
            data, envelope, shutdown = hardware_worker_fixture(directory)
            before = deepcopy((data, envelope, shutdown))
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                host_signal_evidence=envelope, natural_shutdown_evidence=shutdown)
            self.assertEqual((data, envelope, shutdown), before)
            self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
            self.assertEqual(result["observed_cpu_backing_bytes"], 4096)
            self.assertEqual(result["observed_gtt_event_page_bytes"], 32768)
            self.assertEqual(len(result["gtt_event_page_backing"]["gtt_pages"]), 1)
            self.assertEqual(len(result["gtt_event_page_backing"]["event_proofs"]), 2)
            resource = result["signal_generations"][0]["host_backing_evidence"]
            self.assertEqual(resource["hardware_event_backing"]["event_id"], 7)
            self.assertIsNone(result["signal_generations"][0]["destroy"])
            self.assertIsNone(result["gpu_storage_bytes"])
            self.assertFalse(result["complete_workspace"])

    def test_kfd_artifact_tampering_missing_namespace_or_nonclosing_final_prefix_cannot_admit_hardware_event(self):
        with tempfile.TemporaryDirectory() as directory:
            data, envelope, shutdown = hardware_worker_fixture(directory)
            for mutate in (
                lambda e: e["kfd_event_evidence"]["final_report_artifact"].update(sha256="f" * 64),
                lambda e: e["kfd_event_evidence"].pop("final_report_artifact"),
                lambda e: e["kfd_event_evidence"]["final_snapshot_observation"].update(native_stop_returncode=2),
                lambda e: e["kfd_event_evidence"]["final_snapshot_observation"].update(started_monotonic_raw_ns=999),
                lambda e: e["kfd_event_evidence"]["namespace_control_artifact"].update(path="/missing/cold-control.json"),
                lambda e: e["capture"]["records"][0]["samples"][0].pop("kfd_window"),
            ):
                bad = deepcopy(envelope)
                mutate(bad)
                result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                    host_signal_evidence=bad, natural_shutdown_evidence=shutdown)
                self.assertEqual(result["status"], "PARTIAL", result)
                self.assertEqual(result["allowed_call_proofs"], [])

    def test_kfd_producer_success_label_does_not_replace_actual_gtt_type_or_driver_namespace_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            data, envelope, shutdown = hardware_worker_fixture(directory)
            kernel = envelope["kfd_event_evidence"]
            for key in ("pre_query_report_artifact", "post_query_report_artifact", "final_report_artifact"):
                reference = kernel[key]
                path = Path(reference["path"])
                report = json.loads(path.read_bytes())
                for row in report["events"]:
                    if row["operation"] == 3:
                        row["values"][5] = 1  # VRAM, despite a fabricated success label.
                encoded = json.dumps(report, sort_keys=True).encode()
                path.write_bytes(encoded)
                reference.update(sha256=hashlib.sha256(encoded).hexdigest(), size_bytes=len(encoded))
            kernel["report_artifact"] = deepcopy(kernel["post_query_report_artifact"])
            kernel["status"] = "KFD_EVENT_GTT_BACKING_PROVEN"
            envelope["capture"]["kfd_event_evidence"] = {key: deepcopy(value) for key, value in kernel.items()
                if key not in ("final_report_artifact", "final_snapshot_observation")}
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                host_signal_evidence=envelope, natural_shutdown_evidence=shutdown)
            self.assertEqual(result["status"], "PARTIAL", result)
            self.assertEqual(result["allowed_call_proofs"], [])
            self.assertIn("not positively GTT-only", result["issues"][0]["message"])

    def run_audit(self, data):
        before = deepcopy(data)
        result = audit_synchronization_resources(data, target_epochs=[1], device_index=0)
        self.assertEqual(data, before)
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["hardware_validated"])
        self.assertIsNone(result["gpu_storage_bytes"])
        return result

    def closed(self, data):
        result = self.run_audit(data)
        self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
        return result

    def partial(self, data):
        result = self.run_audit(data)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["allowed_call_proofs"], [])
        self.assertTrue(result["issues"])

    def test_exact_successful_disable_timing_event_and_signal_lifetimes(self):
        result = self.closed(resource_fixture())
        self.assertEqual(len(result["event_generations"]), 2)
        self.assertEqual(len(result["signal_generations"]), 1)
        self.assertEqual(result["startup_live_resources"], ["hip-event-10-c2"])
        transient = next(value for value in result["event_generations"] if value["object_handle"] == 20)
        self.assertEqual(transient["state"], "DESTROYED_AND_COMPLETION_FENCED")
        self.assertEqual(transient["completion_fence"]["cid"], 11)
        signal = result["signal_generations"][0]
        self.assertEqual(signal["parent_call"]["cid"], 7)
        allowed = {value["operation"] for value in result["allowed_call_proofs"]}
        self.assertTrue({"hipEventCreateWithFlags", "hipEventDestroy", "hsa_signal_create", "hsa_signal_destroy"} <= allowed)
        self.assertNotIn("hipMalloc", allowed)

    def test_pending_query_is_retained_and_never_a_completion_fence(self):
        result = self.closed(resource_fixture())
        query = next(row for row in result["allowed_call_proofs"] if row["operation"] == "hipEventQuery")
        self.assertIsNone(query["completion_fence_cid"])
        data = resource_fixture()
        named(data, "hipDeviceSynchronize")["status"] = 600
        self.partial(data)

    def test_exact_handle_and_creation_flags_cannot_be_replaced_by_balanced_counts(self):
        for mutate in (
            lambda d: named(d, "hipEventDestroy")["raw_hip_control_payload"].update(object_handle=99),
            lambda d: named(d, "hipEventCreateWithFlags")["raw_hip_control_payload"].update(flags=6),
            lambda d: named(d, "hipEventCreateWithFlags").update(status_type=0, status=None),
            lambda d: named(d, "hipEventDestroy").pop("raw_hip_control_payload"),
        ):
            data = resource_fixture()
            mutate(data)
            self.partial(data)

    def test_startup_resource_must_have_an_observed_create_before_formal_use(self):
        data = resource_fixture()
        named(data, "hipEventRecord")["raw_hip_control_payload"]["object_handle"] = 99
        self.partial(data)
        data = resource_fixture()
        named(data, "hipEventCreateWithFlags")["raw_hip_control_payload"].update(object_handle=0)
        self.partial(data)

    def test_signal_handle_consumer_list_and_actual_ancestry_are_required(self):
        for mutate in (
            lambda d: named(d, "hsa_signal_destroy")["raw_hsa_signal_payload"].update(signal_handle=31),
            lambda d: named(d, "hsa_signal_create")["raw_hsa_signal_payload"].update(consumers_observed=False),
            lambda d: named(d, "hsa_signal_create")["raw_hsa_signal_payload"].update(num_consumers=1),
            lambda d: named(d, "hsa_signal_create")["raw_hsa_signal_payload"].pop("initial_value"),
            lambda d: named(d, "hsa_signal_create").update(ancestor=999),
        ):
            data = resource_fixture()
            mutate(data)
            self.partial(data)

    def test_device_fence_needs_independent_observed_context(self):
        data = resource_fixture()
        named(data, "hipDeviceSynchronize")["raw_hip_control_payload"].update(device_index=1)
        self.partial(data)
        data = resource_fixture()
        named(data, "hipDeviceSynchronize")["raw_hip_control_payload"].update(device_source_cid=999)
        self.partial(data)

    def test_real_later_event_fence_on_same_stream_can_close_resources(self):
        data = resource_fixture()
        for row in data["events"]:
            operation = next(item["name"] for item in data["domains"][row["domain"]]["operations"]
                             if item["id"] == row["op"])
            if operation == "hipStreamWaitEvent":
                row["raw_hip_control_payload"]["stream_handle"] = 101
        record_op = next(item["id"] for item in data["domains"][0]["operations"] if item["name"] == "hipEventRecord")
        for phase in (1, 2):
            row = named(data, "hipDeviceSynchronize", phase)
            row["op"] = record_op
            row["raw_hip_control_payload"].update(kind="event", object_handle=10, stream_handle=101)
        sync_op = len(data["domains"][0]["operations"])
        data["domains"][0]["operations"].append({"id": sync_op, "name": "hipEventSynchronize"})
        for phase in (1, 2):
            sequence = len(data["events"]) + 1
            data["events"].append({"seq": sequence, "callback_seq": sequence, "ns": sequence * 10,
                "domain": 0, "op": sync_op, "cid": 12, "tid": 77, "epoch": 2, "ancestor": 0,
                "phase": phase, "flags": 0, "status_type": 1 if phase == 2 else 0,
                "status": 0 if phase == 2 else None,
                "raw_hip_control_payload": {"kind": "event", "object_handle": 10, "flags": 0,
                    "stream_handle": 0, "output_address": 0, "device_index": 0, "device_source_cid": 1}})
        for field in ("retained_events", "attempted_events", "observed_callbacks"):
            data[field] = len(data["events"])
        data["epochs"][1]["end_ns"] = len(data["events"]) * 10 + 5
        result = self.closed(data)
        transient = next(row for row in result["event_generations"] if row["object_handle"] == 20)
        self.assertEqual(transient["completion_fence"]["cid"], 12)

    def test_memory_operations_are_never_a_resource_admission(self):
        data = resource_fixture()
        index = len(data["domains"][0]["operations"])
        data["domains"][0]["operations"].append({"id": index, "name": "hipMalloc"})
        for phase in (1, 2):
            sequence = len(data["events"]) + 1
            data["events"].append({"seq": sequence, "callback_seq": sequence, "ns": sequence * 10,
                "domain": 0, "op": index, "cid": 20, "tid": 77, "epoch": 0, "ancestor": 0,
                "phase": phase, "flags": 1, "status_type": 1 if phase == 2 else 0,
                "status": 0 if phase == 2 else None})
        for field in ("retained_events", "attempted_events", "observed_callbacks"):
            data[field] = len(data["events"])
        result = self.closed(data)
        self.assertNotIn("hipMalloc", {row["operation"] for row in result["allowed_call_proofs"]})

    def test_scope_loss_unknown_operation_or_missing_epoch_never_admits_resources(self):
        for mutate in (
            lambda d: d.update(dropped_events=1), lambda d: d.update(filtered_nonmutator_events=1),
            lambda d: d["events"][0].update(op=999), lambda d: d.update(epochs=[]),
            lambda d: d["events"][-1].update(seq=999),
        ):
            data = resource_fixture()
            mutate(data)
            self.partial(data)

    def test_last_actual_gpu_use_fence_then_source_bound_cpu_shutdown_destroy(self):
        data, receipt = natural_shutdown_fixture()
        self.partial(data)
        before = deepcopy((data, receipt))
        result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                                                 natural_shutdown_evidence=receipt)
        self.assertEqual((data, receipt), before)
        self.assertEqual(result["status"], "SYNCHRONIZATION_RESOURCES_PROVEN", result["issues"])
        row = next(row for row in result["event_generations"] if row["object_handle"] == 20)
        self.assertEqual(row["completion_order"], "last_GPU_use_then_completion_fence_then_original_CPU_shutdown_destroy")
        self.assertEqual(row["completion_fence"]["cid"], 11)
        self.assertEqual(len(result["signal_generations"]), 1)

    def test_old_clock_missing_raw_wrong_process_source_or_window_cannot_close(self):
        for mutate in (
            lambda r: r.pop("native_clock"), lambda r: r.pop("started_monotonic_raw_ns"),
            lambda r: r.update(process_id=78), lambda r: r.update(thread_id=78),
            lambda r: r["source"].update(source_sha256="f" * 64),
            lambda r: r.update(started_monotonic_raw_ns=220),
            lambda r: r.update(original_returned=False),
        ):
            data, receipt = natural_shutdown_fixture()
            mutate(receipt)
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                                                     natural_shutdown_evidence=receipt)
            self.assertEqual(result["status"], "PARTIAL", result)
            self.assertEqual(result["allowed_call_proofs"], [])

    def test_an_early_fence_does_not_cover_a_later_rerecord_or_pending_query(self):
        for operation, status in (("hipEventRecord", 0), ("hipEventQuery", 600)):
            data, receipt = natural_shutdown_fixture()
            op = next(row["id"] for row in data["domains"][0]["operations"] if row["name"] == operation)
            extra = []
            for phase in (1, 2):
                extra.append({"domain": 0, "op": op, "cid": 30, "tid": 77, "epoch": 0, "ancestor": 0,
                    "phase": phase, "flags": 0, "status_type": 1 if phase == 2 else 0,
                    "status": status if phase == 2 else None,
                    "raw_hip_control_payload": {"kind": "event", "object_handle": 20, "stream_handle": 101,
                        "flags": 0, "output_address": 0, "device_index": 0, "device_source_cid": 1}})
            data["events"][18:18] = extra
            for index, row in enumerate(data["events"], 1):
                row.update(seq=index, callback_seq=index, ns=index * 10)
            for field in ("retained_events", "attempted_events", "observed_callbacks"):
                data[field] = len(data["events"])
            receipt["ended_monotonic_raw_ns"] = 245
            result = audit_synchronization_resources(data, target_epochs=[1], device_index=0,
                                                     natural_shutdown_evidence=receipt)
            self.assertEqual(result["status"], "PARTIAL", result)


if __name__ == "__main__":
    unittest.main()
