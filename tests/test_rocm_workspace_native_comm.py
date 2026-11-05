"""CPU communication-backing contracts and the actual native path formatter.

These are synthetic SDK records. No test creates a measured hardware verdict.
"""

from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_rocm_native_backing_audit import Fixture, audit_native_backing, _load_tool

analyze = _load_tool("analyze_native_probe").analyze


class CommunicationFixture(Fixture):
    def __init__(self, *, grouped=False):
        super().__init__()
        self.native["pid"] = 1
        self.native["epochs"].insert(1, dict(
            id=6001, label="controlled_pynccl_constructor", begin_ns=310, end_ns=390, ended=True))
        self.op(8, 1, "ncclCommInitRank", 64, 3)
        self.op(8, 2, "ncclAllReduce", 64, 3)
        self.op(8, 3, "ncclGroupStart", 64, 3)
        self.op(8, 4, "ncclGroupEnd", 64, 3)
        self.op(8, 5, "ncclSend", 64, 3)
        self.op(8, 6, "ncclRecv", 64, 3)
        self.op(0, 1, "hipStreamSynchronize", 0, 1)
        self.init_cid = self.call(8, 1, 320, 355, epoch=6001)
        for row in self.events:
            if row["cid"] == self.init_cid:
                row["raw_rccl_payload"] = dict(kind="init", comm_output_address=0xD000,
                    world_size=2, rank=0, comm_handle=0xC000 if row["phase"] == 2 else None)
        self.allocation_cid = self.alloc(0xB000, 256, 335, epoch=6001)
        for row in self.events:
            if row["cid"] == self.allocation_cid:
                row["ancestor"] = self.init_cid
        self.call(0, 0, 365, 375, epoch=6001)
        self.formal_cids = []
        if grouped:
            self.group_start = self.rccl(3, 430, 435, dict(kind="group", comm_handle=None))
            self.formal_cids.append(self.rccl(5, 440, 445, self.payload(peer=1)))
            self.formal_cids.append(self.rccl(6, 450, 455, self.payload(peer=1)))
            self.group_end = self.rccl(4, 460, 475, dict(kind="group", comm_handle=None))
        else:
            self.formal_cids.append(self.rccl(2, 430, 480, self.payload()))
        self.communication = dict(
            schema_version=1, method="controlled_rccl_constructor_v1",
            driver_source_sha256="a" * 64, driver_source_reference="CPU fixture driver",
            constructor_source_sha256="b" * 64, constructor_source_reference="CPU fixture constructor",
            runtime_version="0.29.1.dev0+g98dff2a81.d20261002",
            loaded_rccl=dict(verified=True, runtime_version="2.30.7", library_sha256="c" * 64,
                path="/fixture/librccl.so", version_code=23007,
                verification_scope="local_proc_maps_loaded_library_ncclGetVersion"),
            rank=dict(process_id=1, global_rank=0, tp_rank=0, stage_index=0,
                tensor_parallel_size=2, pipeline_parallel_size=1, world_size=2,
                logical_device_index=0, agent_handle=1),
            initialization_epochs=[dict(epoch_id=6001, epoch_label="controlled_pynccl_constructor",
                entrypoint="vllm.distributed.device_communicators.pynccl.PyNcclCommunicator.__init__",
                comm_only=True, communicator_handle=0xC000, world_size=2, rank=0)],
            registered_backings=[dict(address=0xB000, bytes=256, agent_handle=1,
                allocation_cid=self.allocation_cid, born_epoch=6001, init_cid=self.init_cid)])

    @staticmethod
    def payload(**kwargs):
        return dict(kind="collective", comm_handle=0xC000, stream_handle=0,
                    count=128, datatype=9, **kwargs)

    def rccl(self, op, start, end, payload, *, epoch=16):
        cid = self.call(8, op, start, end, epoch=epoch)
        for row in self.events:
            if row["cid"] == cid:
                row["raw_rccl_payload"] = deepcopy(payload)
        return cid

    def audit_communication(self, binding=None):
        return audit_native_backing(self.finish(), self.snapshots, device_to_agent={0: 1},
            ctor_binding=self.binding, communication_binding=self.communication if binding is None else binding)


class NativeCommunicationBackingTests(unittest.TestCase):
    def assert_partial(self, result):
        self.assertEqual(result["status"], "PARTIAL", result)
        self.assertIsNone(result["native_fixed_runtime_bytes"])
        self.assertIsNone(result.get("native_fixed_communication_bytes"))
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["hardware_validated"])

    def test_exact_communicator_ancestry_and_registration_are_separate_fixed_budget(self):
        f = CommunicationFixture()
        before = deepcopy(f.communication)
        result = f.audit_communication()
        self.assertEqual(f.communication, before)
        self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
        self.assertEqual(result["native_fixed_runtime_bytes"], 4480)
        self.assertEqual(result["native_fixed_communication_bytes"], 256)
        self.assertFalse(result["complete_workspace"])
        self.assertEqual(result["communication_constructor_candidates"], f.communication["registered_backings"])
        epoch = result["epochs"][0]
        self.assertEqual(epoch["allowed_communication_calls"][0]["operation"], "ncclAllReduce")
        self.assertEqual(epoch["before"]["native_fixed_communication_component_sum_bytes"], 256)
        self.assertEqual(epoch["before"]["native_fixed_runtime_component_sum_bytes"], 4480)
        self.assertTrue(any(row["owner"] == "controlled_rccl_ctor_scope_private_communication_backing"
                            for row in epoch["before"]["owner_table"]))

    def test_without_explicit_binding_collectives_and_private_backing_remain_partial(self):
        f = CommunicationFixture()
        result = audit_native_backing(f.finish(), f.snapshots, device_to_agent={0: 1}, ctor_binding=f.binding)
        self.assert_partial(result)
        self.assertNotIn("native_fixed_communication_bytes", result)
        self.assertIn("formal_native_mutator_not_proven_sync_signal", result["problem_counts"])

    def test_actual_process_rank_library_and_constructor_facts_are_required(self):
        for change in (
            lambda b: b["rank"].update(process_id=2),
            lambda b: b["rank"].update(global_rank=1),
            lambda b: b["rank"].update(agent_handle=2),
            lambda b: b["loaded_rccl"].update(verified=False),
            lambda b: b["loaded_rccl"].pop("library_sha256"),
            lambda b: b.update(constructor_source_sha256="unknown"),
            lambda b: b["initialization_epochs"][0].update(entrypoint="arbitrary_init_device"),
            lambda b: b["initialization_epochs"][0].update(comm_only=False),
            lambda b: b["initialization_epochs"][0].update(epoch_id=16),
        ):
            f = CommunicationFixture()
            change(f.communication)
            with self.subTest(change=change):
                self.assert_partial(f.audit_communication())

    def test_epoch_tag_without_exact_successful_init_ancestry_cannot_assign_owner(self):
        for change in (
            lambda f: [r.update(ancestor=0) for r in f.events if r["cid"] == f.allocation_cid],
            lambda f: [r.update(tid=2) for r in f.events if r["cid"] == f.allocation_cid],
            lambda f: [r.update(status=1) for r in f.events if r["cid"] == f.init_cid and r["phase"] == 2],
            lambda f: [r.pop("raw_rccl_payload", None) for r in f.events if r["cid"] == f.init_cid],
            lambda f: [r["raw_rccl_payload"].update(comm_handle=0xA000)
                       for r in f.events if r["cid"] == f.init_cid and r["phase"] == 2],
        ):
            f = CommunicationFixture()
            change(f)
            with self.subTest(change=change):
                self.assert_partial(f.audit_communication())

    def test_registration_must_replay_exact_address_size_agent_and_correlation(self):
        for field in ("address", "bytes", "agent_handle", "allocation_cid", "born_epoch", "init_cid"):
            f = CommunicationFixture()
            f.communication["registered_backings"][0][field] += 1
            with self.subTest(field=field):
                self.assert_partial(f.audit_communication())
        f = CommunicationFixture()
        f.communication["registered_backings"] = []
        result = f.audit_communication()
        self.assert_partial(result)
        self.assertEqual(result["communication_constructor_candidates"][0]["address"], 0xB000)

    def test_torch_reserved_backing_has_priority_and_cannot_be_registered_twice(self):
        f = CommunicationFixture()
        for snap in f.snapshots[16].values():
            snap["segments"].append(dict(device=0, address=0xB000, total_size=256))
        result = f.audit_communication()
        self.assert_partial(result)
        self.assertEqual(result["communication_constructor_candidates"], [])
        self.assertIn("communication_backing_registrations_differ_from_observed_private_constructor_set",
                      result["problem_counts"])

    def test_unregistered_handle_failed_status_unknown_collective_and_new_native_growth_reject(self):
        for change in (
            lambda f: [r["raw_rccl_payload"].update(comm_handle=0xA000)
                       for r in f.events if r["cid"] in f.formal_cids],
            lambda f: [r.update(status=1) for r in f.events if r["cid"] in f.formal_cids and r["phase"] == 2],
            lambda f: f.op(8, 2, "ncclAllToAll", 64, 3),
            lambda f: f.alloc(0xE000, 64, 460, epoch=16),
        ):
            f = CommunicationFixture()
            change(f)
            with self.subTest(change=change):
                self.assert_partial(f.audit_communication())

    def test_group_send_receive_are_typed_bound_and_closed_within_epoch_thread(self):
        f = CommunicationFixture(grouped=True)
        result = f.audit_communication()
        self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
        self.assertEqual([r["operation"] for r in result["epochs"][0]["allowed_communication_calls"]],
                         ["ncclGroupStart", "ncclSend", "ncclRecv", "ncclGroupEnd"])
        for change in (
            lambda f: f.events.__setitem__(slice(None), [r for r in f.events if r["cid"] != f.group_end]),
            lambda f: [r.update(tid=2) for r in f.events if r["cid"] == f.group_end],
            lambda f: [r["raw_rccl_payload"].update(peer=2)
                       for r in f.events if r["cid"] in f.formal_cids],
        ):
            f = CommunicationFixture(grouped=True)
            change(f)
            with self.subTest(change=change):
                self.assert_partial(f.audit_communication())

    def test_communication_signal_without_real_handle_lifetime_evidence_remains_partial(self):
        for broken in (False, True):
            f = CommunicationFixture()
            parent = f.call(0, 1, 440, 470, epoch=16, ancestor=f.formal_cids[0])
            f.call(2, 0, 450, 455, epoch=16, ancestor=0 if broken else parent)
            result = f.audit_communication()
            with self.subTest(broken=broken):
                self.assert_partial(result)

    def test_fixed_private_communication_set_must_not_disappear_between_formal_epochs(self):
        f = CommunicationFixture()
        f.native["epochs"].append(dict(id=17, label="decode-17", begin_ns=610, end_ns=670, ended=True))
        f.snapshots[17] = deepcopy(f.snapshots[16])
        f.free(0xB000, 605, epoch=0)
        f.call(0, 0, 640, 650, epoch=17)
        result = f.audit_communication()
        self.assert_partial(result)
        self.assertIn("communication_private_backing_set_not_fixed_across_formal_boundaries", result["problem_counts"])

    def test_ctor_first_use_requires_exact_known_comm_control_and_typed_module_launch(self):
        for broken in (None, "count", "datatype", "launch_status"):
            f = CommunicationFixture()
            for row in f.events:
                if row["cid"] == f.init_cid and row["phase"] == 2:
                    row["ns"] = 330
            first = f.rccl(2, 332, 360, f.payload() | {"count": 1, "datatype": 7}, epoch=6001)
            f.op(0, 2, "hipExtModuleLaunchKernel", 0, 1)
            launch = f.call(0, 2, 333, 359, epoch=6001, ancestor=first)
            for row in f.events:
                if row["cid"] == f.allocation_cid:
                    row["ancestor"] = launch
                if row["cid"] == first and broken in ("count", "datatype"):
                    row["raw_rccl_payload"][broken] += 1
                if row["cid"] == launch and row["phase"] == 2 and broken == "launch_status":
                    row.update(status_type=0, status=None)
            f.communication["registered_backings"][0].update(
                proof_kind="ctor-known-first-use", first_use_cid=first, launch_cid=launch)
            result = f.audit_communication()
            with self.subTest(broken=broken):
                if broken:
                    self.assert_partial(result)
                else:
                    self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
                    self.assertEqual(result["communication_constructor_candidates"][0]["proof_kind"],
                                     "ctor-known-first-use")

    def test_proxy_thread_needs_actual_init_birth_library_mapping_and_exact_child_allocator(self):
        for broken in (None, "parent", "library", "mapping", "return", "tid", "origin", "std_control", "std_foreign_dispatch"):
            f = CommunicationFixture()
            f.op(0, 2, "hipExtMallocWithFlags", 1, 1)
            allocator = f.call(0, 2, 333, 340, epoch=6001)
            for row in f.events:
                if row["cid"] in (allocator, f.allocation_cid):
                    row.update(tid=2, thread_origin_id=1)
                if row["cid"] == f.allocation_cid:
                    row["ancestor"] = allocator
            origin = dict(id=1, parent_init_cid=f.init_cid, parent_epoch=6001, parent_tid=1,
                init_output_address=0xD000, start_address=0x400100, module_base=0x400000,
                module_offset=0x100, argument_address=0xE000, create_begin_ns=325, create_end_ns=326,
                create_returncode=0, child_tid=2, child_start_ns=327, child_end_ns=0,
                start_routine_invocations=1, module_known=True, module_path_canonical=True,
                module_path="/fixture/librccl.so")
            f.native["rccl_thread_origins"] = dict(schema_version=1,
                method="pthread_create_from_observed_rccl_init_v1", capacity=256, record_count=1,
                dropped_records=0, observation_errors=0, init_stack_errors=0, records=[origin])
            f.communication["loaded_rccl"]["mapped_segments"] = [dict(start=0x400000, end=0x402000,
                offset=0, permissions="r-xp", path="/fixture/librccl.so")]
            f.communication["registered_backings"][0].update(proof_kind="rccl-born-proxy-thread", thread_origin_id=1)
            if broken in {"std_control", "std_foreign_dispatch"}:
                origin.update(module_path="/fixture/libstdc++.so", module_base=0x800000,
                    start_address=0x800100, module_offset=0x100, stdthread_trampoline_matched=True,
                    state_dispatch_observed=True, state_vtable=0x400200, state_typeinfo=0x400300,
                    state_dispatch=0x400400, run_member_offset=16,
                    vtable_module="/fixture/librccl.so", typeinfo_module="/fixture/librccl.so",
                    dispatch_module="/fixture/librccl.so", vtable_module_base=0x400000,
                    typeinfo_module_base=0x400000, dispatch_module_base=0x400000)
                f.communication["stdthread_control"] = dict(
                    control=dict(schema_version=1, method="observed_stdthread_state_dispatch_v1", pid=1,
                        attempted=True, completed=True, control_returncode=0, trampoline_observed=True,
                        start_address=0x800100, module_base=0x800000, module_offset=0x100,
                        run_member_offset=16, member_pointer_size=16, state_size=8, glibcxx_release=13,
                        header_reference="/usr/include/c++/13/bits/std_thread.h",
                        header_sha256="bfaeaa41bf061e826c96cbee7ee3c453250918b7c3e7cc96b01693ee31751233",
                        module_path="/fixture/libstdc++.so"),
                    header=dict(path="/usr/include/c++/13/bits/std_thread.h",
                        sha256="bfaeaa41bf061e826c96cbee7ee3c453250918b7c3e7cc96b01693ee31751233"),
                    runtime_library=dict(path="/fixture/libstdc++.so", sha256="d" * 64,
                        mapped_segments=[dict(start=0x800000, end=0x802000, offset=0,
                            permissions="r-xp", path="/fixture/libstdc++.so")]))
                if broken == "std_foreign_dispatch":
                    origin["dispatch_module"] = "/fixture/libstdc++.so"
            if broken == "parent":
                origin["parent_init_cid"] += 1
            elif broken == "library":
                origin["module_path"] = "/fixture/libother.so"
            elif broken == "mapping":
                f.communication["loaded_rccl"]["mapped_segments"][0]["permissions"] = "rw-p"
            elif broken == "return":
                origin["create_returncode"] = 1
            elif broken == "tid":
                origin["child_tid"] = 3
            elif broken == "origin":
                for row in f.events:
                    if row["cid"] == f.allocation_cid:
                        row["thread_origin_id"] = 0
            result = f.audit_communication()
            with self.subTest(broken=broken):
                if broken and broken != "std_control":
                    self.assert_partial(result)
                else:
                    self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
                    self.assertEqual(result["communication_constructor_candidates"][0]["proof_kind"],
                                     "rccl-born-proxy-thread")

    def test_process_group_scope_binds_real_buffers_validation_source_and_native_init(self):
        for broken in (None, "validation", "pointer", "dtype", "backend", "sync"):
            f = CommunicationFixture()
            for row in f.events:
                if row["epoch"] == 16:
                    row["ns"] += 300
            formal = next(e for e in f.native["epochs"] if e["id"] == 16)
            formal.update(begin_ns=700, end_ns=900)
            f.native["stop_evidence"].update(begin_ns=1000, end_ns=1010)
            f.native["finalize_ns"] = 1100
            f.native["epochs"].insert(2, dict(id=6501, label="known_tp_prewarm", begin_ns=400, end_ns=600, ended=True))
            f.op(8, 7, "ncclCommInitRankConfig", 64, 3)
            f.op(8, 8, "ncclAllGather", 64, 3)
            init = f.call(8, 7, 430, 470, epoch=6501)
            for row in f.events:
                if row["cid"] == init:
                    row["raw_rccl_payload"] = dict(kind="init", comm_output_address=0xE000,
                        world_size=2, rank=0, comm_handle=0xD000 if row["phase"] == 2 else None)
            alloc = f.alloc(0xF000, 128, 440, epoch=6501)
            for row in f.events:
                if row["cid"] == alloc:
                    row["ancestor"] = init
            payload = f.payload() | dict(comm_handle=0xD000, count=75968,
                send_buffer_address=0x100000, receive_buffer_address=0x200000)
            f.rccl(8, 480, 520, payload, epoch=6501)
            f.call(0, 0, 560, 580, epoch=6501)
            f.rccl(8, 735, 755, payload)
            source = dict(qualified_name="torch.distributed.distributed_c10d.all_gather_into_tensor",
                file_sha256="e" * 64, source_sha256="f" * 64, source_reference="/fixture/distributed_c10d.py")
            def descriptor(pointer, count):
                return dict(shape=[count], dtype="torch.bfloat16", element_size=2, device_index=0,
                    bytes=count * 2, tensor_offset=0, pointer=pointer, tensor_data_ptr=pointer)
            scope = dict(kind="torch_process_group_prewarm", control="tp_allgather", epoch_id=6501,
                epoch_label="known_tp_prewarm", entrypoint=source["qualified_name"], source_binding=source,
                comm_only=True, process_id=1, logical_device_index=0, status="COMPLETED",
                scope="independent_startup_communication_before_model_load", preallocated_before_epoch=True,
                preallocation_completed_monotonic_raw_ns=395, device_sync_completed=True,
                pg=dict(object_id=123, backend="nccl", group_name="group0", global_ranks=[0, 1],
                    world_size=2, rank_in_group=0), input_buffers=[descriptor(0x100000, 75968)],
                output_buffers=[descriptor(0x200000, 151936)],
                payload_validation=dict(success=True, method="full_CPU_copy_compare_after_native_epoch",
                    checks=[dict(expected_count=75968, observed_count=75968, mismatch_count=0,
                        expected_value=value, expected_checksum=value * 75968, observed_checksum=value * 75968)
                        for value in (1.0, 2.0)]))
            if broken == "validation":
                scope["payload_validation"]["success"] = False
            elif broken == "pointer":
                scope["input_buffers"][0].update(pointer=0x300000, tensor_data_ptr=0x300000)
            elif broken == "dtype":
                scope["input_buffers"][0]["dtype"] = "torch.float32"
            elif broken == "backend":
                scope["pg"]["backend"] = "gloo"
            elif broken == "sync":
                scope["device_sync_completed"] = False
            f.communication["initialization_epochs"].append(dict(kind="torch_process_group_prewarm",
                epoch_id=6501, epoch_label="known_tp_prewarm", entrypoint=source["qualified_name"],
                comm_only=True, communicator_handle=0xD000, world_size=2, rank=0, pg_scope=scope))
            f.communication["registered_backings"].append(dict(address=0xF000, bytes=128, agent_handle=1,
                allocation_cid=alloc, born_epoch=6501, init_cid=init))
            result = f.audit_communication()
            with self.subTest(broken=broken):
                if broken:
                    self.assert_partial(result)
                else:
                    self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
                    self.assertEqual(result["native_fixed_communication_bytes"], 384)

    def test_startup_pool_metadata_and_host_registration_need_exact_observed_effect_and_source(self):
        for broken in (None, "export", "device", "threshold", "host_flags", "origin"):
            f = CommunicationFixture()
            f.op(0, 2, "hipMemPoolCreate", 16, 1)
            f.op(0, 3, "hipMemPoolSetAttribute", 16, 1)
            f.op(0, 4, "hipHostRegister", 32, 1)
            create = f.call(0, 2, 323, 326, epoch=6001, ancestor=f.init_cid)
            attribute = f.call(0, 3, 327, 329, epoch=6001, ancestor=f.init_cid)
            host = f.call(0, 4, 330, 334, epoch=6001, ancestor=f.init_cid)
            for row in f.events:
                if row["cid"] == create:
                    row["raw_hip_control_payload"] = dict(kind="pool_create",
                        object_handle=0xE100 if row["phase"] == 2 else None,
                        output_address=0xD100, properties_address=0xD200, properties_observed=row["phase"] == 2,
                        reserved_zero=True, alloc_type=1, handle_types=0, location_type=1, location_id=0, max_size_bytes=0)
                    if broken == "export":
                        row["raw_hip_control_payload"]["handle_types"] = 1
                    elif broken == "device":
                        row["raw_hip_control_payload"]["location_id"] = 1
                if row["cid"] == attribute:
                    row["raw_hip_control_payload"] = dict(kind="pool_attribute", object_handle=0xE100,
                        attribute=4, pointer=0xD300, value=2**64 - 1 if row["phase"] == 2 else None)
                    if broken == "threshold":
                        row["raw_hip_control_payload"]["value"] = 0
                if row["cid"] == host:
                    row["raw_hip_control_payload"] = dict(kind="host_registration", pointer=0x800000,
                        size_bytes=1024, flags=0x80000003)
                    if broken == "host_flags":
                        row["raw_hip_control_payload"]["flags"] = 4
                if broken == "origin" and row["cid"] in (create, attribute, host):
                    row["ancestor"] = 0
            result = f.audit_communication()
            with self.subTest(broken=broken):
                if broken:
                    self.assert_partial(result)
                else:
                    self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
                    controls = result["startup_communication_control_audit"]
                    self.assertEqual(len(controls), 3)
                    self.assertTrue(all(c["accepted_startup_effect"] for c in controls))
                    self.assertTrue(all(c["gpu_backing_replay_mutation_applied"] is False for c in controls))

    def test_async_copy_host_unpin_is_effect_only_outside_formal(self):
        f = CommunicationFixture()
        f.op(0, 2, "hipMemcpyAsync", 0, 1)
        f.op(3, 2, "hsa_amd_memory_unlock", 32, 2)
        parent = f.call(0, 2, 301, 309, epoch=0)
        f.call(3, 2, 303, 305, epoch=0, ancestor=parent)
        result = f.audit_communication()
        self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
        audit = result["host_registration_cleanup_audit"][-1]
        self.assertEqual(audit["parent_operation"], "hipMemcpyAsync")
        self.assertFalse(audit["argument_pointer_observed"])
        self.assertFalse(audit["host_registration_lifetime_verified"])

    def test_natural_teardown_requires_raw_window_and_exact_registered_pool_host_objects(self):
        for broken in (None, "raw_clock", "pool_handle", "host_pointer", "outside_window",
                       "rccl_cleanup_thread", "cleanup_error", "cleanup_wrong_tid"):
            f = CommunicationFixture()
            f.native["stop_evidence"].update(begin_ns=1000, end_ns=1010)
            f.native["finalize_ns"] = 1100
            f.op(0, 2, "hipMemPoolCreate", 16, 1)
            f.op(0, 3, "hipHostRegister", 32, 1)
            f.op(0, 4, "hipMemPoolDestroy", 16, 1)
            f.op(0, 5, "hipHostUnregister", 32, 1)
            f.op(3, 2, "hsa_amd_memory_unlock", 32, 2)
            create = f.call(0, 2, 323, 326, epoch=6001, ancestor=f.init_cid)
            register = f.call(0, 3, 327, 329, epoch=6001, ancestor=f.init_cid)
            unregister = f.call(0, 5, 750, 760, epoch=0)
            f.call(3, 2, 754, 756, epoch=0, ancestor=unregister)
            destroy = f.call(0, 4, 780, 790, epoch=0)
            if broken in {"rccl_cleanup_thread", "cleanup_error", "cleanup_wrong_tid"}:
                f.op(8, 7, "ncclCommAbort", 64, 3)
                cleanup = f.call(8, 7, 740, 800, epoch=0, status=1 if broken == "cleanup_error" else 0)
                for row in f.events:
                    if row["cid"] in (unregister, destroy, cleanup) or row["ancestor"] == unregister:
                        row["tid"] = 2
                    if row["cid"] in (unregister, destroy):
                        row["ancestor"] = cleanup
                    if row["cid"] == cleanup and broken == "cleanup_wrong_tid":
                        row["tid"] = 3
            for row in f.events:
                if row["cid"] == create:
                    row["raw_hip_control_payload"] = dict(kind="pool_create",
                        object_handle=0xE100 if row["phase"] == 2 else None, output_address=0xD100,
                        properties_address=0xD200, properties_observed=row["phase"] == 2,
                        reserved_zero=True, alloc_type=1, handle_types=0, location_type=1, location_id=0, max_size_bytes=0)
                if row["cid"] == register:
                    row["raw_hip_control_payload"] = dict(kind="host_registration", pointer=0x800000,
                        size_bytes=1024, flags=0x80000003)
                if row["cid"] == unregister:
                    row["raw_hip_control_payload"] = dict(kind="host_unregistration",
                        pointer=0x900000 if broken == "host_pointer" else 0x800000)
                if row["cid"] == destroy:
                    row["raw_hip_control_payload"] = dict(kind="pool_destroy",
                        object_handle=0xF100 if broken == "pool_handle" else 0xE100)
            f.communication["natural_shutdown_evidence"] = dict(process_id=1, thread_id=1,
                native_clock="CLOCK_MONOTONIC_RAW", original_returned=True, return_type="builtins.NoneType",
                scope="original_WorkerProc_shutdown_including_model_and_distributed_environment",
                started_monotonic_raw_ns=700, ended_monotonic_raw_ns=900,
                source=dict(qualified_name="vllm.v1.executor.multiproc_executor.WorkerProc.shutdown",
                    source_sha256="36f51b0ac5471bc230ccfce7dfc6ced2d16d2c73e35b10a3ab81eae1ec592140",
                    file_sha256="3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a",
                    source_reference="/fixture/multiproc_executor.py"))
            if broken == "raw_clock":
                f.communication["natural_shutdown_evidence"].pop("native_clock")
            elif broken == "outside_window":
                f.communication["natural_shutdown_evidence"]["started_monotonic_raw_ns"] = 770
            result = f.audit_communication()
            with self.subTest(broken=broken):
                if broken and broken != "rccl_cleanup_thread":
                    self.assert_partial(result)
                else:
                    self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
                    controls = result["startup_communication_control_audit"]
                    cleanups = [c for c in controls if "accepted_natural_teardown_effect" in c]
                    self.assertEqual(len(cleanups), 2)
                    self.assertTrue(all(c["accepted_natural_teardown_effect"] for c in cleanups))
                    self.assertTrue(result["host_registration_cleanup_audit"][-1]["explained_host_registration_cleanup"])


class PublicHsaQueryEvidenceTests(unittest.TestCase):
    @staticmethod
    def fixture():
        f = Fixture()
        f.op(3, 7, "hsa_amd_pointer_info", 512, 2)
        f.op(2, 7, "hsa_agent_get_info", 512, 2)
        f.op(3, 8, "hsa_amd_signal_get_event_id", 0, 2)
        pointer = f.call(3, 7, 620, 625, epoch=0)
        owner = f.call(2, 7, 630, 635, epoch=0)
        event = f.call(3, 8, 640, 645, epoch=0)
        fields = dict(returned_size=[0, 4], type=[4, 4], agent_base_address=[8, 8],
            host_base_address=[16, 8], size_bytes=[24, 8], owner_agent=[40, 8],
            global_flags=[48, 4], registered=[52, 1], alloc_flags=[56, 4])
        for row in f.events:
            exited = row["phase"] == 2
            if row["cid"] == pointer:
                payload = dict(pointer_address=0x8010, info_address=0xD000, input_size=None if exited else 64,
                    payload_observed=True, **dict.fromkeys(fields))
                if exited:
                    payload.update(returned_size=64, type=1, agent_base_address=0x8000,
                        host_base_address=0x8000, size_bytes=64, owner_agent=999,
                        global_flags=1, registered=True, alloc_flags=16)
                row["raw_hsa_pointer_info_payload"] = payload
            elif row["cid"] == owner:
                row["raw_hsa_agent_info_payload"] = dict(agent_handle=999, attribute=17,
                    output_address=0xD100, device_type=0 if exited else None, payload_observed=True)
            elif row["cid"] == event:
                row["raw_hsa_signal_event_payload"] = dict(signal_handle=0x8010, output_address=0xD200,
                    event_id=0 if exited else None, payload_observed=True)
        f.native["hsa_query_storage"] = dict(schema_version=1, method="bounded_public_hsa_query_payloads_v1",
            capacity=8192, retained_records=6, record_size_bytes=112, storage_capacity_bytes=8192 * 112,
            dropped_records=0, read_errors=0, agent_info_device_attribute=17, cpu_device_type=0,
            pointer_type_hsa=1, pointer_type_locked=2, pointer_info_struct_size=64, pointer_info_fields=fields)
        return f, pointer, owner, event

    def test_public_query_outputs_remain_diagnostic_without_cross_api_agent_identity_inference(self):
        f, _, _, _ = self.fixture()
        data = f.finish()
        before = deepcopy(data)
        result = analyze(data)
        self.assertEqual(result["global_failures"], [])
        self.assertEqual(data, before)
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["hardware_validated"])
        # Runtime hsa_agent_t=999 is intentionally not any SDK inventory ID.
        backing = audit_native_backing(data, f.snapshots, device_to_agent={0: 1}, ctor_binding=f.binding)
        self.assertEqual(backing["status"], "NATIVE_BACKING_CHECKS_PASSED", backing["issues"])
        self.assertFalse(backing["complete_workspace"])

    def test_complete_public_inputs_capture_null_optional_addresses_and_consumer_pointer(self):
        for broken in (None, "missing_optional", "optional_changed", "missing_consumers", "consumer_changed", "null_consumers"):
            f, pointer, _, _ = self.fixture()
            f.native["hsa_query_storage"]["complete_public_query_inputs"] = True
            for row in f.events:
                if row["cid"] == pointer:
                    row["raw_hsa_pointer_info_payload"].update(alloc_address=0, num_agents_address=0, accessible_address=0)
                    if broken == "missing_optional":
                        row["raw_hsa_pointer_info_payload"].pop("alloc_address")
                    elif broken == "optional_changed" and row["phase"] == 2:
                        row["raw_hsa_pointer_info_payload"]["accessible_address"] = 0xF000
                if row["domain"] == 2 and row["op"] == 0:
                    row["raw_hsa_signal_payload"] = dict(output_address=0xD900,
                        initial_value=0, num_consumers=1, consumers_address=0xD800)
                    if broken == "missing_consumers":
                        row["raw_hsa_signal_payload"].pop("consumers_address")
                    elif broken == "consumer_changed" and row["phase"] == 2:
                        row["raw_hsa_signal_payload"]["consumers_address"] += 8
                    elif broken == "null_consumers":
                        row["raw_hsa_signal_payload"]["consumers_address"] = 0
            failures = analyze(f.finish())["global_failures"]
            with self.subTest(broken=broken):
                if broken:
                    self.assertTrue(any(key.startswith("hsa_query_") for key in failures), failures)
                else:
                    self.assertEqual(failures, [])

    def test_unknown_truncated_and_failed_outputs_never_read_undefined_fields(self):
        for mode in ("unknown", "truncated", "failed"):
            f, cid, _, _ = self.fixture()
            row = next(r for r in f.events if r["cid"] == cid and r["phase"] == 2)
            payload = row["raw_hsa_pointer_info_payload"]
            if mode == "unknown":
                payload.update(dict.fromkeys(f.native["hsa_query_storage"]["pointer_info_fields"]))
                payload.update(returned_size=64, type=0)
            elif mode == "truncated":
                payload.update(returned_size=48, global_flags=None, registered=None, alloc_flags=None)
            else:
                row["status"] = 4097
                payload.update(dict.fromkeys(f.native["hsa_query_storage"]["pointer_info_fields"]))
            with self.subTest(mode=mode):
                self.assertEqual(analyze(f.finish())["global_failures"], [])
                payload["alloc_flags"] = 16
                self.assertIn("hsa_query_output_outside_returned_contract" if mode != "failed"
                    else "hsa_query_failed_output_not_null", analyze(f.finish())["global_failures"])

    def test_query_loss_bad_abi_mutated_pairs_and_untyped_outputs_fail_closed(self):
        for mode in ("drop", "read", "count", "layout", "missing", "untyped", "extent", "pair", "pending"):
            f, cid, _, event = self.fixture()
            table = f.native["hsa_query_storage"]
            row = next(r for r in f.events if r["cid"] == cid and r["phase"] == 2)
            if mode == "drop":
                table["dropped_records"] = 1
            elif mode == "read":
                table["read_errors"] = 1
            elif mode == "count":
                table["retained_records"] = 5
            elif mode == "layout":
                table["pointer_info_fields"]["owner_agent"] = [63, 8]
            elif mode == "missing":
                row.pop("raw_hsa_pointer_info_payload")
                table["retained_records"] = 5
            elif mode == "untyped":
                row.update(status_type=0, status=None)
            elif mode == "extent":
                row["raw_hsa_pointer_info_payload"]["returned_size"] = 65
            elif mode == "pair":
                row["raw_hsa_pointer_info_payload"]["pointer_address"] += 1
            else:
                row = next(r for r in f.events if r["cid"] == event and r["phase"] == 2)
                row["status"] = 1  # Failed query cannot disclose a stale event id.
            with self.subTest(mode=mode):
                self.assertTrue(any(k.startswith("hsa_query_") for k in analyze(f.finish())["global_failures"]))


class HsaHostMemoryPayloadTests(unittest.TestCase):
    @staticmethod
    def fixture():
        f = Fixture()
        cids = []
        for op, name, start in ((7, "hsa_amd_memory_lock", 610), (8, "hsa_amd_memory_unlock", 620),
                (9, "hsa_amd_memory_lock_to_pool", 630)):
            f.op(3, op, name, 32, 2)
            cid = f.call(3, op, start, start + 5, epoch=0)
            cids.append(cid)
            for row in f.events:
                if row["cid"] != cid:
                    continue
                unlock = op == 8
                row["raw_hsa_host_memory_payload"] = dict(operation="unlock" if unlock else "lock" if op == 7 else "lock_to_pool",
                    host_ptr=0x700000, payload_observed=True, size_bytes=None if unlock else 4097,
                    agents_address=None if unlock else 0xD000, num_agents=None if unlock else 1,
                    agent_handles=None if unlock else [0xA001], agents_observed=None if unlock else True,
                    pool_handle=0xA100 if op == 9 else None, flags=0 if op == 9 else None,
                    agent_ptr_output_address=None if unlock else 0xD100,
                    agent_alias=0x900000 if not unlock and row["phase"] == 2 else None)
        f.native["hsa_host_memory_storage"] = dict(schema_version=1,
            method="bounded_public_hsa_host_memory_payloads_v1", capacity=8192, agent_capacity=4,
            retained_records=6, record_size_bytes=104, storage_capacity_bytes=8192 * 104,
            dropped_records=0, read_errors=0, incomplete_agent_arrays=0, system_page_size=4096)
        return f, cids

    def test_actual_host_and_distinct_device_alias_inputs_remain_diagnostic_and_immutable(self):
        f, _ = self.fixture()
        data = f.finish()
        before = deepcopy(data)
        result = analyze(data)
        self.assertEqual(result["global_failures"], [])
        self.assertEqual(data, before)
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["hardware_validated"])

    def test_host_pointer_status_extent_arrays_and_output_observation_fail_closed(self):
        for mode in ("changed_unlock", "missing_unlock", "untyped_unlock", "invented_unlock_size",
                "failed_lock_alias", "enter_alias", "array_truncated", "array_changed", "safe_read_failed", "range_overflow"):
            f, cids = self.fixture()
            for row in f.events:
                payload = row.get("raw_hsa_host_memory_payload")
                if payload is None:
                    continue
                if row["cid"] == cids[1] and row["phase"] == 2:
                    if mode == "changed_unlock":
                        payload["host_ptr"] += 4096
                    elif mode == "missing_unlock":
                        row.pop("raw_hsa_host_memory_payload")
                    elif mode == "untyped_unlock":
                        row.update(status_type=0, status=None)
                    elif mode == "invented_unlock_size":
                        payload["size_bytes"] = 4096
                if row["cid"] == cids[0]:
                    if mode == "failed_lock_alias" and row["phase"] == 2:
                        row["status"] = 4097
                    elif mode == "enter_alias" and row["phase"] == 1:
                        payload["agent_alias"] = 0x900000
                    elif mode == "array_truncated":
                        payload.update(num_agents=5, agent_handles=[1, 2, 3, 4])
                    elif mode == "array_changed" and row["phase"] == 2:
                        payload["agent_handles"][0] += 1
                    elif mode == "safe_read_failed":
                        payload["payload_observed"] = False
                    elif mode == "range_overflow":
                        payload["host_ptr"] = 2**64 - 1
            with self.subTest(mode=mode):
                self.assertTrue(any(key.startswith("hsa_host_memory_") for key in analyze(f.finish())["global_failures"]))

    def test_sparse_budget_page_size_and_failed_output_are_explicit(self):
        for mode in ("drop", "read", "array", "count", "missing_table", "page_zero", "page_not_power_two"):
            f, _ = self.fixture()
            table = f.native["hsa_host_memory_storage"]
            if mode in ("drop", "read", "array"):
                table[{"drop": "dropped_records", "read": "read_errors", "array": "incomplete_agent_arrays"}[mode]] = 1
            elif mode == "count":
                table["retained_records"] -= 1
            elif mode == "missing_table":
                f.native.pop("hsa_host_memory_storage")
            else:
                table["system_page_size"] = 0 if mode == "page_zero" else 6000
            with self.subTest(mode=mode):
                self.assertTrue(any(key.startswith("hsa_host_memory_") for key in analyze(f.finish())["global_failures"]))
        f, cids = self.fixture()
        for row in f.events:
            if row["cid"] == cids[0] and row["phase"] == 2:
                row.update(status=4097)
                row["raw_hsa_host_memory_payload"]["agent_alias"] = None
        self.assertEqual(analyze(f.finish())["global_failures"], [])

    def test_agents_allow_access_retains_exact_pointer_agents_and_reserved_null_flags(self):
        for mode in (None, "pointer_changed", "array_changed", "array_truncated", "flags_pointer", "flags_scalar",
                "status_untyped", "missing_pointer", "safe_read"):
            f, _ = self.fixture()
            f.native["hsa_host_memory_storage"]["supported_operations"] = ["hsa_amd_memory_unlock",
                "hsa_amd_memory_lock", "hsa_amd_memory_lock_to_pool", "hsa_amd_agents_allow_access"]
            f.native["hsa_host_memory_storage"]["retained_records"] += 2
            f.op(3, 10, "hsa_amd_agents_allow_access", 0, 2)
            cid = f.call(3, 10, 640, 645, epoch=0)
            for row in f.events:
                if row["cid"] != cid:
                    continue
                payload = dict(operation="allow_access", ptr=0x700000, payload_observed=True,
                    size_bytes=None, agents_address=0xD000, num_agents=1, agents_observed=True,
                    agent_handles=[0xA001], pool_handle=None, flags=None, agent_ptr_output_address=None,
                    agent_alias=None, flags_address=0, flags_values=None, flags_observed=True)
                row["raw_hsa_host_memory_payload"] = payload
                if mode == "pointer_changed" and row["phase"] == 2:
                    payload["ptr"] += 4096
                elif mode == "array_changed" and row["phase"] == 2:
                    payload["agent_handles"][0] += 1
                elif mode == "array_truncated":
                    payload.update(num_agents=5, agent_handles=[1, 2, 3, 4])
                elif mode == "flags_pointer":
                    payload.update(flags_address=0xD200, flags_observed=False)
                elif mode == "flags_scalar":
                    payload["flags_values"] = 0
                elif mode == "status_untyped" and row["phase"] == 2:
                    row.update(status_type=0, status=None)
                elif mode == "missing_pointer":
                    payload.pop("ptr")
                elif mode == "safe_read":
                    payload["payload_observed"] = False
            with self.subTest(mode=mode):
                failures = analyze(f.finish())["global_failures"]
                if mode:
                    self.assertTrue(any(code.startswith("hsa_host_memory_") for code in failures))
                else:
                    self.assertEqual(failures, [])


class HipHostMemoryPayloadTests(unittest.TestCase):
    @staticmethod
    def fixture():
        f = Fixture()
        f.op(0, 7, "hipHostMalloc", 33, 1)
        f.op(0, 8, "hipHostFree", 34, 1)
        allocate = f.call(0, 7, 610, 615, epoch=0)
        free = f.call(0, 8, 620, 625, epoch=0)
        for row in f.events:
            if row["cid"] == allocate:
                row["raw_hip_control_payload"] = dict(kind="host_allocation", output_address=0xD000,
                    pointer=0x700000 if row["phase"] == 2 else None, size_bytes=8193, flags=3,
                    pointer_observed=row["phase"] == 2, payload_observed=True)
            elif row["cid"] == free:
                row["raw_hip_control_payload"] = dict(kind="host_free", output_address=0,
                    pointer=0x700000, size_bytes=None, flags=None, pointer_observed=True, payload_observed=True)
        f.native["hip_host_memory_payloads"] = dict(schema_version=1, method="existing_native_event_storage_v1",
            supported_operations=["hipHostMalloc", "hipHostFree"], retained_records=4,
            safe_read_errors=0, uses_native_event_storage=True)
        return f, allocate, free

    def test_actual_parent_pointer_and_requested_extent_keep_original_flags_without_ownership_inference(self):
        f, _, _ = self.fixture()
        data = f.finish()
        before = deepcopy(data)
        result = analyze(data)
        self.assertEqual(result["global_failures"], [])
        self.assertEqual(data, before)
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["hardware_validated"])

    def test_changed_inputs_missing_parent_pointer_failed_output_and_bad_extents_fail_closed(self):
        for mode in ("missing_free", "changed_free", "free_extent", "untyped", "malloc_size_changed",
                "malloc_flags_changed", "stale_failed_output", "enter_output", "safe_read", "overflow"):
            f, allocate, free = self.fixture()
            for row in f.events:
                payload = row.get("raw_hip_control_payload")
                if payload is None:
                    continue
                if row["cid"] == free and row["phase"] == 2:
                    if mode == "missing_free":
                        payload.pop("pointer")
                    elif mode == "changed_free":
                        payload["pointer"] += 4096
                    elif mode == "free_extent":
                        payload["size_bytes"] = 0
                    elif mode == "untyped":
                        row.update(status_type=0, status=None)
                if row["cid"] == allocate:
                    if mode == "enter_output" and row["phase"] == 1:
                        payload.update(pointer=0x700000, pointer_observed=True)
                    elif row["phase"] == 2:
                        if mode == "malloc_size_changed":
                            payload["size_bytes"] += 1
                        elif mode == "malloc_flags_changed":
                            payload["flags"] += 1
                        elif mode == "stale_failed_output":
                            row["status"] = 1
                        elif mode == "safe_read":
                            payload["payload_observed"] = False
                        elif mode == "overflow":
                            payload["pointer"] = 2**64 - 1
            with self.subTest(mode=mode):
                self.assertTrue(any(key.startswith("hip_host_memory_") for key in analyze(f.finish())["global_failures"]))

    def test_budget_counters_are_explicit_and_failed_allocations_do_not_disclose_unproven_output(self):
        f, allocate, _ = self.fixture()
        for row in f.events:
            if row["cid"] == allocate and row["phase"] == 2:
                row["status"] = 1
                row["raw_hip_control_payload"].update(pointer=None, pointer_observed=False)
        self.assertEqual(analyze(f.finish())["global_failures"], [])
        for key, value in (("safe_read_errors", 1), ("retained_records", 3), ("uses_native_event_storage", False)):
            bad = deepcopy(f)
            bad.native["hip_host_memory_payloads"][key] = value
            with self.subTest(key=key):
                self.assertTrue(any(code.startswith("hip_host_memory_") for code in analyze(bad.finish())["global_failures"]))


class NativeOutputPathFormatterTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix" and shutil.which("c++"), "Optional native formatter needs a C++ compiler")
    def test_actual_formatter_only_accepts_bounded_literal_or_pid_paths(self):
        source = (Path(__file__).resolve().parents[1] / "scripts/rocm_workspace_audit/native_probe.cpp").read_text()
        formatter = source.split("// MLSYS_OUTPUT_PATH_FORMATTER_BEGIN\n", 1)[1].split(
            "// MLSYS_OUTPUT_PATH_FORMATTER_END", 1)[0]
        harness = """#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cstdlib>
""" + formatter + """
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    size_t capacity = std::strtoul(argv[2], nullptr, 10);
    if (capacity > 65536) return 3;
    std::vector<char> buffer(capacity ? capacity : 1, '\\0');
    bool valid = expand_output_path(buffer.data(), capacity, argv[1]);
    std::cout << (valid ? 1 : 0) << "\\n" << buffer.data() << "\\n" << getpid() << "\\n";
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cpp, binary = root / "formatter.cpp", root / "formatter"
            toolchain = subprocess.run([shutil.which("c++") or "c++", "-std=c++17", "-x", "c++",
                "-fsyntax-only", "-"], input="#include <cstdio>\n", capture_output=True, text=True, timeout=30)
            if toolchain.returncode:
                self.skipTest("Optional native formatter needs usable C++ standard-library headers")
            cpp.write_text(harness)
            compiled = subprocess.run([shutil.which("c++") or "c++", "-std=c++17", str(cpp), "-o", str(binary)],
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            def run(value, size=128):
                result = subprocess.run([str(binary), value, str(size)], check=True, capture_output=True,
                                        text=True, timeout=5).stdout.splitlines()
                return int(result[0]), result[1], result[2]
            for value in ("/tmp/plain.json", "/tmp/worker-%p.json", "%p/%p.json", ""):
                valid, expanded, pid = run(value)
                self.assertEqual((valid, expanded), (1, value.replace("%p", pid)))
            for value, size in (("%s", 128), ("%d", 128), ("%%", 128), ("tail%", 128),
                                ("a" * 128, 128), ("short", 0), ("x%p", 3)):
                valid, expanded, _ = run(value, size)
                self.assertEqual((valid, expanded), (0, ""))


class OfficialClientFinalizeEvidenceTests(unittest.TestCase):
    @staticmethod
    def fixture():
        data = Fixture().finish()
        data["explicit_client_finalize"] = dict(schema_version=1, requested=True,
            official_callback_available=True, actual_client_id_saved=True, request_begin_ns=720,
            actual_tool_fini_callback_count=1, api="rocprofiler_client_finalize_t",
            sdk_return_type="void", sdk_return_status=None)
        return data

    def test_actual_callback_and_void_api_facts_are_distinct_from_a_synthetic_success_status(self):
        data = self.fixture()
        self.assertEqual(analyze(data)["global_failures"], [])
        self.assertFalse(analyze(data)["complete_workspace"])
        for change in (
            lambda d: d["explicit_client_finalize"].update(sdk_return_status=0),
            lambda d: d["explicit_client_finalize"].update(actual_tool_fini_callback_count=0),
            lambda d: d["explicit_client_finalize"].update(actual_tool_fini_callback_count=2),
            lambda d: d["explicit_client_finalize"].update(official_callback_available=False),
            lambda d: d["explicit_client_finalize"].update(actual_client_id_saved=False),
            lambda d: d["explicit_client_finalize"].update(request_begin_ns=600),
            lambda d: d["explicit_client_finalize"].update(request_begin_ns=900),
            lambda d: d.update(finalize_callback_called=False),
        ):
            bad = deepcopy(data)
            change(bad)
            with self.subTest(change=change):
                self.assertTrue(analyze(bad)["global_failures"])

    def test_default_sdk_exit_path_remains_valid_and_request_facts_cannot_be_backfilled(self):
        data = self.fixture()
        data["explicit_client_finalize"].update(requested=False, request_begin_ns=0)
        self.assertEqual(analyze(data)["global_failures"], [])
        data["explicit_client_finalize"]["request_begin_ns"] = 720
        self.assertIn("unrequested_client_finalize_has_request_timestamp", analyze(data)["global_failures"])
        data.pop("explicit_client_finalize")
        self.assertEqual(analyze(data)["global_failures"], [])

    def test_open_api_drain_retains_late_exit_and_closes_only_after_ordinary_calls_finish(self):
        data = self.fixture()
        data["api_call_drain_evidence"] = dict(method="ordinary_api_enter_exit_counter_v1",
            ordinary_domains=[0, 1, 2, 3, 4, 5, 8], normalized_and_scratch_counted=False,
            sdk_stop_return_ns=705, window_closed_ns=710, open_calls_at_sdk_stop_return=1,
            open_calls_at_window_close=0, counter_underflows=0, api_enters_after_sdk_stop_return=0,
            wait_iterations=12, timeout_ns=1000000000)
        self.assertEqual(analyze(data)["global_failures"], [])
        for change in (
            lambda d: d["api_call_drain_evidence"].update(open_calls_at_window_close=1),
            lambda d: d["api_call_drain_evidence"].update(counter_underflows=1),
            lambda d: d["api_call_drain_evidence"].update(api_enters_after_sdk_stop_return=1),
            lambda d: d["api_call_drain_evidence"].update(normalized_and_scratch_counted=True),
            lambda d: d["api_call_drain_evidence"].update(window_closed_ns=706),
            lambda d: d["api_call_drain_evidence"].update(sdk_stop_return_ns=711),
        ):
            bad = deepcopy(data)
            change(bad)
            with self.subTest(change=change):
                self.assertTrue(analyze(bad)["global_failures"])


if __name__ == "__main__":
    unittest.main()
