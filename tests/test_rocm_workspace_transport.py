"""CPU-only actual-order-shaped transport observations; no GPU was measured."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import unittest

from test_rocm_workspace_placement import rank_fixture
from test_rocm_workspace_torch_audit import audit_torch_workspace
from mlsys_sim import DeviceSpec, Experiment, InfraSpec, MemoryConfig, ModelSpec, StrategySpec
from mlsys_sim.config import experiment_from_dict
from mlsys_sim.pp_transport import native_pp_descriptor


_PATH = Path(__file__).resolve().parents[1] / "scripts/rocm_workspace_audit/workspace_transport_audit.py"
_SPEC = importlib.util.spec_from_file_location("workspace_transport_audit", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)
audit_workspace_transport = _TOOL.audit_workspace_transport


def transport_fixture(cycles=2, defer_transport_free=False):
    rank = rank_fixture(pp=2)
    experiment = Experiment(infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))),
        strategy=StrategySpec(model=ModelSpec(**rank["model"]), pipeline_parallel_size=2,
            pipeline_transport="vllm_native", pipeline_transport_max_tokens=512,
            max_batch_size=1, max_batch_tokens=512, memory=MemoryConfig(kv_pool_bytes=268435456)))
    segments = deepcopy(rank["before_snapshot"]["segments"])
    arena = segments[-1]["address"]
    segments[-1]["total_size"] = 32768
    addresses = [arena + value * 4096 for value in range(5)]
    live = {}
    history = [{"action": "snapshot", "addr": 0, "size": 0, "stream": 0,
                "pool_id": [0, 0], "user_metadata": "", "compile_context": "N/A"}]
    events = []
    epochs = []

    def emit(action, address=0, size=0, epoch=0, thread=77):
        sequence = len(events) + 1
        history.append({"action": action, "addr": address, "size": size, "stream": 0,
                        "pool_id": [0, 0], "user_metadata": "", "compile_context": "N/A"})
        event = {"sequence": sequence, "timestamp_ns": 1000 + sequence * 10,
            "thread_id": thread, "epoch_id": epoch, "owner_token": epoch if thread == 77 else 0,
            "action": _TOOL._ACTIONS.index(action), "action_name": action, "device": 0,
            "address": address, "size_bytes": size, "stream": 0, "mempool": [0, 0],
            "flags": (0 if thread == 77 else 2) if epoch else 4,
            "user_metadata_size": 0, "retained_metadata_size": 0, "user_metadata": ""}
        events.append(event)
        if action == "alloc":
            live[address] = {"bytes": size, "state": "active_allocated"}
        elif action == "free_requested":
            live[address]["state"] = "active_awaiting_free"
        elif action == "free_completed":
            del live[address]
        return sequence

    def snapshot(epoch=0):
        before = len(events)
        copied_history = deepcopy(history)
        allocated = sum(block["size"] for segment in segments[:-1] for block in segment["blocks"]
                        if block["state"] == "active_allocated")
        allocated += sum(((value["bytes"] + 511) // 512) * 512 for value in live.values())
        emit("snapshot", size=allocated, epoch=epoch)
        emit("snapshot", size=0, epoch=epoch)
        current = deepcopy(segments)
        blocks = []
        cursor = arena
        for address, value in sorted(live.items()):
            if address > cursor:
                blocks.append({"address": cursor, "size": address - cursor, "requested_size": 0,
                               "state": "inactive"})
            rounded = ((value["bytes"] + 511) // 512) * 512
            blocks.append({"address": address, "size": rounded, "requested_size": value["bytes"],
                           "state": value["state"]})
            cursor = address + rounded
        if cursor < arena + 32768:
            blocks.append({"address": cursor, "size": arena + 32768 - cursor,
                           "requested_size": 0, "state": "inactive"})
        current[-1]["blocks"] = blocks
        return {"snapshot": {"segments": current, "device_traces": [copied_history]},
                "cursor": len(events), "snapshot_cursor_before": before,
                "snapshot_cursor_after": len(events), "pending_pp": []}

    def descriptor(address, name, cursor):
        return {"path": "pp.actual." + name, "tensor_name": name, "pointer": address,
            "bytes": 4096, "tensor_data_ptr": address, "shape": [1, 2048], "stride": [2048, 1],
            "tensor_offset": 0, "dtype": "torch.bfloat16", "element_size": 2, "device_index": 0,
            "metadata_cursor_before": cursor, "metadata_cursor_after": cursor, "metadata_cursor": cursor}

    def pending(pair, handles, cursor):
        tensors = [descriptor(pair[0], "hidden_states", cursor), descriptor(pair[1], "residual", cursor)]
        for tensor in tensors:
            tensor.update(shape=[2048], stride=[1])
        return [{"entry_index": 0, "handle_ids": handles, "tensor_descriptors": tensors}]

    def epoch_info(identifier, first, last):
        zero = {name: 0 for name in ("dropped_events", "unknown_actions", "unowned_allocations",
            "cross_thread_allocations", "truncated_metadata", "boundary_overlap", "boundary_callbacks",
            "clock_failures", "thread_id_failures", "oom_events", "mapping_events", "observation_issue_mask")}
        return {"id": identifier, "owner_thread_id": 77, "begin_sequence": first, "end_sequence": last,
                "begin_ns": 1000 + first * 10 + 1, "end_ns": 1000 + last * 10 + 1,
                "ended": True, "observed_events": last - first,
                "observation_issues_present": False, **zero}

    group = {"object_id": 123, "class_name": "vllm.distributed.parallel_state.GroupCoordinator",
             "ranks": [0, 1], "rank_in_group": 0, "world_size": 2, "unique_name": "pp:cpu-fixture"}
    sources = {name: {"qualified_name": "vllm.distributed.parallel_state.GroupCoordinator." + name,
                "source_sha256": str(index) * 64, "source_reference": "cpu-fixture-source.py",
                "file_sha256": "f" * 64} for index, name in enumerate(sorted(_TOOL._OPERATIONS), 1)}
    transport_events = []
    frames = []
    startup = snapshot()
    startup.update(process_id=77, thread_id=77)
    previous = None
    old_handles = None
    for iteration in range(1, cycles + 1):
        identifier = 10000 + iteration
        before = snapshot()
        first = len(events)
        pair = addresses[:2] if iteration % 2 else addresses[2:4]
        for address in pair:
            emit("alloc", address, 4096, identifier)
        emit("alloc", addresses[4], 64 * iteration, identifier)
        emit("free_requested", addresses[4], 64 * iteration, identifier)
        emit("free_completed", addresses[4], 64 * iteration, identifier)
        send_cursor = len(events)
        pp_before = pending(previous, old_handles, send_cursor) if previous else []
        if previous:
            for address in previous:
                emit("free_requested", address, 4096, identifier)
                if not (defer_transport_free and address == previous[-1] and iteration == 2):
                    emit("free_completed", address, 4096, identifier)
        reap_end = len(events)
        handles = [iteration * 100 + offset for offset in (1, 2, 3)]
        pp_after = pending(pair, handles, reap_end)
        common = {"cohort": "warmup" if iteration == 1 else "formal", "epoch_id": identifier,
                  "observer_epoch_id": identifier, "capture_iteration": iteration, "group": group,
                  "status": "RETURNED"}
        # Nested natural reap returns before its enclosing send. IDs are
        # reserved on call entry and therefore need not be sorted in the log.
        transport_events.append({**deepcopy(common), "event_id": iteration * 2,
            "operation": "_reap_completed_isends", "source_method_sha256": sources["_reap_completed_isends"]["source_sha256"],
            "before_cursor": send_cursor, "after_cursor": reap_end,
            "before_pending": deepcopy(pp_before), "after_pending": [], "tensors": []})
        transport_events.append({**deepcopy(common), "event_id": iteration * 2 - 1,
            "operation": "isend_tensor_dict", "source_method_sha256": sources["isend_tensor_dict"]["source_sha256"],
            "before_cursor": send_cursor, "after_cursor": reap_end,
            "before_pending": deepcopy(pp_before), "after_pending": deepcopy(pp_after),
            "handles": handles, "tensors": [descriptor(pair[0], "hidden_states", send_cursor),
                                             descriptor(pair[1], "residual", send_cursor)]})
        after = snapshot(identifier)
        if defer_transport_free and previous and iteration == 2:
            emit("free_completed", previous[-1], 4096, identifier, thread=88)
        last = len(events)
        epochs.append(epoch_info(identifier, first, last))
        native = native_pp_descriptor(experiment, 1)
        stage = {"iteration_id": iteration - 1, "phase": "decode", "query_tokens": [1],
            "context_lengths": [15 + iteration], "transport_metadata": {"outgoing": {"native_descriptor": native}}}
        frame = {"epoch_id": identifier, "capture_iteration": iteration, "cohort": common["cohort"],
            "process_id": 77, "thread_id": 77, "status": "RAW_STEP_CAPTURED", "stage_record_index": iteration - 1,
            "stage_record": stage, "before": before["snapshot"], "after": after["snapshot"],
            "iteration_id": iteration - 1, "phase": "decode", "query_tokens": [1],
            "context_lengths": [15 + iteration],
            "before_cursor": before["cursor"], "after_cursor": after["cursor"],
            "before_snapshot_cursor_before": before["snapshot_cursor_before"],
            "before_snapshot_cursor_after": before["snapshot_cursor_after"],
            "after_snapshot_cursor_before": after["snapshot_cursor_before"],
            "after_snapshot_cursor_after": after["snapshot_cursor_after"],
            "owners_before": deepcopy(rank["owners_before"]), "owners_after": deepcopy(rank["owners_after"])}
        frames.append(frame)
        previous, old_handles = pair, handles
    teardown_first = len(events)
    before = snapshot(9001)
    reap_first = len(events)
    old_pending = pending(previous, old_handles, reap_first)
    for address in previous:
        emit("free_requested", address, 4096, 9001)
    reap_last = len(events)
    for address in previous:
        emit("free_completed", address, 4096, 9001)
    after = snapshot(9001)
    epochs.append(epoch_info(9001, teardown_first, len(events)))
    transport_events.append({"event_id": cycles * 2 + 1, "operation": "_reap_completed_isends", "cohort": "teardown",
        "epoch_id": None, "observer_epoch_id": 9001, "capture_iteration": cycles, "group": group,
        "source_method_sha256": sources["_reap_completed_isends"]["source_sha256"], "status": "RETURNED",
        "before_cursor": reap_first, "after_cursor": reap_last, "before_pending": old_pending,
        "after_pending": [], "tensors": []})
    teardown = {"epoch_id": 9001, "before": before, "after": after}
    probe = deepcopy(rank["torch_probe"])
    probe.update(pid=77, events=events, epochs=epochs, event_limit=131072, outside_epoch_allocations=0,
                 retained_events=len(events), attempted_events_at_cutoff=len(events),
                 attempted_events_at_write_end=len(events), clock="CLOCK_MONOTONIC_RAW",
                 metadata_encoding="byte_preserving_latin1", owner_policy="active_epoch_install_thread_token_only",
                 hook_detachable=False, callback_scope="all_allocator_tracker_callbacks_since_explicit_install")
    source = {"worker_source_sha256": "a" * 64, "worker_source_reference": "cpu-worker-fixture.py",
        "base_worker": {"qualified_name": "vllm.v1.worker.gpu_worker.Worker.execute_model",
            "source_sha256": "b" * 64, "source_reference": "cpu-base-fixture.py", "file_sha256": "c" * 64}}
    session = {"schema_version": 1, "kind": "vllm_worker_workspace_raw", "session_id": "cpu-pp-fixture",
        "process_id": 77, "tensor_parallel_size": 1, "pipeline_parallel_size": 2, "stage_index": 0,
        "tp_rank": 0, "logical_device_index": 0, "global_rank": 0, "world_size": 2,
        "placement": {"process_id": 77, "tensor_parallel_size": 1, "pipeline_parallel_size": 2,
                      "stage_index": 0, "tp_rank": 0, "logical_device_index": 0, "global_rank": 0, "world_size": 2},
        "capture_closed": True, "capture_errors": [], "source_bindings": source,
        "observed_pp_group": group, "transport_sources": sources, "transport_events": transport_events,
        "history_max_entries": 100000,
        "stage_capture": {"session_id": "cpu-pp-fixture", "tensor_parallel_size": 1,
            "pipeline_parallel_size": 2, "stage_index": 0, "tp_rank": 0,
            "experiment": asdict(experiment),
            "source": {"kind": "test_fixture", "reference": "CPU-only integer observations"},
            "metadata": {"native_pp_backend_binding": {"backend": "nccl", "use_cpu_custom_send_recv": False}},
            "records": [deepcopy(frame["stage_record"]) for frame in frames]},
        "records": [{key: deepcopy(value) for key, value in frame.items()
                     if key not in {"before", "after", "owners_before", "owners_after"}} for frame in frames]}
    return {"torch_probe": probe, "transport_context": {"session": session, "startup": startup,
            "teardown": teardown, "epochs": frames}, "placement": rank["placement"]}


class WorkspaceTransportTests(unittest.TestCase):
    def audit(self, data):
        original = deepcopy(data)
        result = audit_workspace_transport(**data)
        self.assertEqual(data, original)
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["hardware_validated"])
        return result

    def closed(self, data):
        result = self.audit(data)
        self.assertEqual(result["status"], "TRANSPORT_GENERATIONS_PROVEN", result["issues"])
        return result

    def partial(self, data):
        result = self.audit(data)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertTrue(result["issues"])
        return result

    def test_cross_epoch_generations_keep_actual_fifo_overlap_until_completed_free(self):
        result = self.closed(transport_fixture())
        self.assertEqual(len(result["transport_generations"]), 4)
        first, second = result["epochs"]["10001"], result["epochs"]["10002"]
        self.assertEqual((len(first["before"]), len(first["after"])), (0, 2))
        self.assertEqual((len(second["before"]), len(second["after"])), (2, 2))
        self.assertEqual((first["peak_pp_requested_bytes"], second["peak_pp_requested_bytes"]), (8192, 16384))
        self.assertTrue(all(row["free_completed_sequence"] for row in result["transport_generations"]))

    def test_receiver_storage_requires_actual_receive_birth_and_has_no_self_retained_send_queue(self):
        data = transport_fixture()
        context = data["transport_context"]
        session = context["session"]
        place = rank_fixture(pp=2, stage=1, device=1)["placement"]
        data["placement"] = place
        session.update(stage_index=1, global_rank=1, logical_device_index=1)
        session["placement"].update(stage_index=1, global_rank=1, logical_device_index=1)
        session["observed_pp_group"]["rank_in_group"] = 1
        session["stage_capture"]["stage_index"] = 1
        session["stage_capture"]["metadata"]["native_pp_persistent_binding"] = {
            "max_num_tokens": 512, "scope": "loaded_runner_intermediate_buffers",
            "descriptor": native_pp_descriptor(experiment_from_dict(session["stage_capture"]["experiment"]), 512)}
        selected = []
        for event in session["transport_events"]:
            event["group"]["rank_in_group"] = 1
            if event["operation"] == "_reap_completed_isends" and event["cohort"] != "teardown":
                continue
            event.update(before_pending=[], after_pending=[])
            if event["operation"] == "isend_tensor_dict":
                event["operation"] = "irecv_tensor_dict"
                event["source_method_sha256"] = session["transport_sources"]["irecv_tensor_dict"]["source_sha256"]
                frame = context["epochs"][event["capture_iteration"] - 1]
                event["before_cursor"] = frame["before_cursor"]
                event["handles"] = event["handles"][:2]
                for tensor in event["tensors"]:
                    tensor.update(device_index=1, metadata_cursor_before=event["after_cursor"],
                        metadata_cursor_after=event["after_cursor"], metadata_cursor=event["after_cursor"],
                        shape=[2048], stride=[1])
            selected.append(event)
        session["transport_events"] = selected
        for event in data["torch_probe"]["events"]:
            if event["action_name"] != "snapshot":
                event["device"] = 1
        all_snapshots = [context["startup"]["snapshot"], context["teardown"]["before"]["snapshot"],
                         context["teardown"]["after"]["snapshot"]]
        for frame in context["epochs"]:
            frame["stage_record"]["transport_metadata"]["incoming"] = frame["stage_record"]["transport_metadata"].pop("outgoing")
            all_snapshots.extend([frame["before"], frame["after"]])
        session["stage_capture"]["records"] = [deepcopy(frame["stage_record"]) for frame in context["epochs"]]
        session["records"] = [{key: deepcopy(value) for key, value in frame.items()
                               if key not in {"before", "after", "owners_before", "owners_after"}}
                              for frame in context["epochs"]]
        for snapshot in all_snapshots:
            for segment in snapshot["segments"]:
                segment["device"] = 1
            snapshot["device_traces"] = [[], snapshot["device_traces"][0]]
        result = self.closed(data)
        self.assertTrue(all(event["device"] == 0 for event in data["torch_probe"]["events"]
                            if event["action_name"] == "snapshot"))
        self.assertTrue(all(reg["operation"] == "irecv_tensor_dict" for row in result["transport_generations"]
                            for reg in row["registrations"]))

    def test_drops_missing_pid_unknowns_and_uncertain_snapshot_cursor_fail_closed(self):
        for mutate in (
            lambda d: d["torch_probe"].update(dropped_events=1),
            lambda d: d["torch_probe"].pop("pid"),
            lambda d: d["torch_probe"].update(pid=78),
            lambda d: d["torch_probe"]["events"][3].update(action_name="unknown"),
            lambda d: d["transport_context"]["startup"].update(snapshot_cursor_before=2**64 - 1),
            lambda d: d["transport_context"]["epochs"].pop(0),
        ):
            data = transport_fixture()
            mutate(data)
            self.partial(data)

    def test_source_group_and_frame_binding_cannot_be_asserted_by_a_boolean(self):
        for mutate in (
            lambda s: s["transport_events"][1].update(source_method_sha256="d" * 64),
            lambda s: s["transport_events"][1]["group"].update(object_id=999),
            lambda s: s["transport_events"][1].update(capture_iteration=2),
            lambda s: s["transport_events"][1].update(epoch_id=10002),
            lambda s: s.update(transport_sources={"verified": True}),
            lambda s: s["transport_events"][1].update(status="RAISED"),
        ):
            data = transport_fixture()
            mutate(data["transport_context"]["session"])
            self.partial(data)

    def test_address_and_capacity_without_current_generation_are_not_enough(self):
        for mutate in (
            lambda t: t.update(pointer=t["pointer"] + 512),
            lambda t: t.update(bytes=8192),
            lambda t: t.update(tensor_offset=1),
            lambda t: t.update(stride=[1, 1]),
            lambda t: t.update(tensor_name=None),
            lambda t: t.update(metadata_cursor_before=0),
            lambda t: t.update(metadata_cursor_after=2**64 - 1),
        ):
            data = transport_fixture()
            mutate(data["transport_context"]["session"]["transport_events"][1]["tensors"][0])
            self.partial(data)

    def test_real_address_reuse_needs_distinct_new_generation_registration(self):
        data = transport_fixture(cycles=3)
        result = self.closed(data)
        generations = result["transport_generations"]
        first = next(row for row in generations if row["allocation_epoch_id"] == 10001)
        reused = next(row for row in generations if row["allocation_epoch_id"] == 10003
                      and row["address"] == first["address"])
        self.assertNotEqual(first["generation_id"], reused["generation_id"])
        self.assertLess(first["free_completed_sequence"], reused["alloc_sequence"])
        actual = data["transport_context"]["session"]["transport_events"]
        old_cursor = actual[1]["tensors"][0]["metadata_cursor"]
        current = actual[5]["tensors"][0]
        current["metadata_cursor_before"] = old_cursor
        self.partial(data)

    def test_fifo_must_retain_actual_send_tensors_and_handles(self):
        for mutate in (
            lambda e: e.update(after_pending=[]),
            lambda e: e["after_pending"][0]["handle_ids"].__setitem__(0, 999),
            lambda e: e["after_pending"][0]["tensor_descriptors"][0].update(tensor_name="residual"),
            lambda e: e["after_pending"][0].update(entry_index=2),
        ):
            data = transport_fixture()
            mutate(data["transport_context"]["session"]["transport_events"][1])
            self.partial(data)

    def test_all_registered_storage_must_be_released_at_explicit_teardown(self):
        data = transport_fixture()
        end = data["transport_context"]["teardown"]["after"]
        end["pending_pp"] = [{"unknown": True}]
        self.partial(data)
        data = transport_fixture()
        # A final snapshot cannot hide a missing real release.
        probe = data["torch_probe"]
        target = next(x for x in reversed(probe["events"]) if x["action_name"] == "free_completed")
        target.update(action=7, action_name="snapshot", address=0, size_bytes=0)
        self.partial(data)

    def test_full_memory_history_and_native_tape_are_both_required(self):
        data = transport_fixture()
        trace = data["transport_context"]["teardown"]["after"]["snapshot"]["device_traces"][0]
        next(row for row in trace if row["action"] != "snapshot")["size"] += 1
        self.partial(data)
        data = transport_fixture()
        data["transport_context"]["startup"]["snapshot"]["device_traces"][0][0]["user_metadata"] = "changed"
        self.partial(data)

    def test_background_free_allowed_only_for_a_proven_transport_generation(self):
        data = transport_fixture()
        result = self.closed(data)
        sequence = next(row["free_completed_sequence"] for row in result["transport_generations"]
                        if row["allocation_epoch_id"] == 10001)
        event = data["torch_probe"]["events"][sequence - 1]
        event.update(thread_id=88, owner_token=0, flags=2)
        result = self.closed(data)
        self.assertIn(sequence, result["epochs"]["10002"]["allowed_cross_thread_free_sequences"])
        data = transport_fixture()
        event = next(row for row in data["torch_probe"]["events"] if row["action_name"] == "free_completed"
                     and row["size_bytes"] == 128)
        event.update(thread_id=88, owner_token=0, flags=2)
        self.partial(data)

    def test_torch_audit_excludes_only_replayed_transport_and_keeps_dynamic_peak(self):
        data = transport_fixture()
        rank = rank_fixture(pp=2)
        frame = data["transport_context"]["epochs"][1]
        arguments = {"model": rank["model"], "before_snapshot": frame["before"],
            "after_snapshot": frame["after"], "owners_before": frame["owners_before"],
            "owners_after": frame["owners_after"], "torch_probe": data["torch_probe"],
            "epoch_id": frame["epoch_id"], "history_max_entries": 100000,
            "placement": data["placement"]}
        before = deepcopy(arguments)
        self.assertEqual(audit_torch_workspace(**arguments)["status"], "PARTIAL")
        result = audit_torch_workspace(**arguments, transport_context=data["transport_context"])
        self.assertEqual(arguments, before)
        self.assertEqual(result["status"], "CLOSED_CANDIDATE", result["issues"])
        self.assertEqual(result["requested_storage_peak"], 128)
        self.assertEqual(result["pp_transport_peak_bytes"], 16384)
        self.assertEqual(len(result["ordered_ledger"]), 3)
        self.assertEqual(len(result["excluded_transport_ledger"]), 6)
        self.assertFalse(result["complete_workspace"])
        spoofed = deepcopy(data["transport_context"])
        spoofed["session"]["transport_events"][1]["source_method_sha256"] = "d" * 64
        result = audit_torch_workspace(**arguments, transport_context=spoofed)
        self.assertEqual(result["status"], "PARTIAL")

    def test_torch_audit_requires_same_raw_frame_and_does_not_hide_unknown_persistent_storage(self):
        data = transport_fixture()
        rank = rank_fixture(pp=2)
        frame = data["transport_context"]["epochs"][1]
        arguments = {"model": rank["model"], "before_snapshot": frame["before"],
            "after_snapshot": deepcopy(frame["after"]), "owners_before": frame["owners_before"],
            "owners_after": frame["owners_after"], "torch_probe": data["torch_probe"],
            "epoch_id": frame["epoch_id"], "history_max_entries": 100000,
            "placement": data["placement"], "transport_context": data["transport_context"]}
        arguments["after_snapshot"]["segments"][0]["blocks"][0]["requested_size"] -= 1
        result = audit_torch_workspace(**arguments)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIn("Transport frame disagrees", result["issues"][-1]["message"])

    def test_torch_audit_preserves_transport_background_free_flags_without_broad_thread_relaxation(self):
        data = transport_fixture()
        proof = self.closed(data)
        sequence = next(row["free_completed_sequence"] for row in proof["transport_generations"]
                        if row["allocation_epoch_id"] == 10001)
        event = data["torch_probe"]["events"][sequence - 1]
        event.update(thread_id=88, owner_token=0, flags=2)
        rank = rank_fixture(pp=2)
        frame = data["transport_context"]["epochs"][1]
        result = audit_torch_workspace(model=rank["model"], before_snapshot=frame["before"],
            after_snapshot=frame["after"], owners_before=frame["owners_before"], owners_after=frame["owners_after"],
            torch_probe=data["torch_probe"], epoch_id=frame["epoch_id"], history_max_entries=100000,
            placement=data["placement"], transport_context=data["transport_context"])
        self.assertEqual(result["status"], "CLOSED_CANDIDATE", result["issues"])
        copied = next(row for row in result["excluded_transport_ledger"] if row["native_sequence"] == sequence)
        self.assertEqual((copied["flags"], copied["thread_id"]), (2, 88))

    def test_proven_delayed_free_after_snapshot_does_not_change_fixed_or_dynamic_storage(self):
        data = transport_fixture(defer_transport_free=True)
        proof = self.closed(data)
        row = proof["epochs"]["10002"]
        self.assertEqual(len(row["after"]), 3)
        self.assertTrue(any(value["state"] == "active_awaiting_free" for value in row["after"]))
        rank = rank_fixture(pp=2)
        frame = data["transport_context"]["epochs"][1]
        result = audit_torch_workspace(model=rank["model"], before_snapshot=frame["before"],
            after_snapshot=frame["after"], owners_before=frame["owners_before"], owners_after=frame["owners_after"],
            torch_probe=data["torch_probe"], epoch_id=frame["epoch_id"], history_max_entries=100000,
            placement=data["placement"], transport_context=data["transport_context"])
        self.assertEqual(result["status"], "CLOSED_CANDIDATE", result["issues"])
        self.assertEqual(result["requested_storage_peak"], 128)
        self.assertEqual(result["pp_transport_peak_bytes"], 16384)


if __name__ == "__main__":
    unittest.main()
