"""CPU-only resource/control observations; no GPU or driver is exercised."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

_PATH = Path(__file__).resolve().parents[1] / "scripts/rocm_workspace_audit/synchronization_resource_audit.py"
_SPEC = importlib.util.spec_from_file_location("synchronization_resource_audit", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOL)
audit_synchronization_resources = _TOOL.audit_synchronization_resources


def resource_fixture():
    events, operations = [], [{}, {}, {}]

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
    signal = {"operation": "create", "signal_handle": None, "output_address": 300,
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


class SynchronizationResourceTests(unittest.TestCase):
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
