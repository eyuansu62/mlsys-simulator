"""Synthetic CPU records exercise the UAPI graph; they are not hardware samples."""

from copy import deepcopy
import unittest

from test_rocm_native_backing_audit import _load_tool

audit = _load_tool("kfd_event_audit").audit_kfd_events


def identity(fd, kind=1, minor=0):
    return dict(fd=fd, kind=kind, observed=True, path_complete=True, dev=4,
        rdev=5000 + minor, inode=100 + fd, mode=0o020600, rdev_major=226 if kind == 2 else 511,
        rdev_minor=minor, path="/dev/kfd" if kind == 1 else "/dev/dri/renderD" + str(minor))


class KfdFixture:
    def __init__(self):
        self.rows = []
        self.call_id = 0
        self.abi = dict(machine="linux_x86_64_sysv", pointer_bytes=8, off_t_bytes=8, off64_t_bytes=8,
            create_event_bytes=32, allocate_bytes=40, requests={str(i): 1000 + i for i in range(1, 7)},
            acquire_vm_request=1011, request_operations={str(1000 + i): i for i in range(1, 18)},
            gtt_flag=2, vram_flag=1, userptr_flag=4, doorbell_flag=8, mmio_flag=16,
            map_shared=1, map_fixed=16, prot_read=1, prot_write=2, signal_event_type=0,
            runtime_enable_mask=1, runtime_ttmp_mask=2,
            offsets={"create_id": [24, 4], "create_slot": [28, 4]})
        self.report = dict(schema_version=1, kind="passive_kfd_event_api_tape", complete_workspace=False,
            hardware_validated=False, write_complete=True, scope="observed_libc_kfd_and_mapping_prefix",
            clock="CLOCK_MONOTONIC_RAW", pid=100, snapshot_ns=200, capacity=131072,
            record_size_bytes=1000, storage_capacity_bytes=131072000, dropped_rows=0,
            read_errors=0, forward_errors=0, reentrant_calls=0, active_mutators=0,
            fork_observation=dict(atfork_registered=True, inherited_kfd_resources_unsupported=False,
                parent_pid=0, discarded_parent_rows=0), abi=self.abi, events=self.rows)
        self.source = {key: "a" * 64 for key in ("probe_source_sha256", "native_probe_source_sha256",
            "probe_library_sha256", "hsa_runtime_sha256", "libc_sha256", "kfd_header_sha256", "drm_header_sha256")}
        self.source.update(kernel_release="6.14.0-fixture", amdgpu_module_build_id="b" * 40, abi=deepcopy(self.abi),
            device_nodes=[dict(gpu_id=77, rdev_major=226, rdev_minor=128, pci_bus_id="0000:01:00.0")])
        self.source["artifacts"] = {name: dict(path="/fixture/" + name, sha256=self.source[name + "_sha256"], size_bytes=100)
            for name in ("probe_source", "native_probe_source", "probe_library", "hsa_runtime", "libc", "kfd_header", "drm_header")}
        self.report.update(compiled_source_path="/fixture/probe_source", cplusplus=201703, compiler_version="fixture compiler",
            libc_bindings={name: dict(address=0x710000 + i * 64, module_base=0x700000,
                dladdr_success=True, module_path="/fixture/libc", symbol_name="__" + name)
                for i, name in enumerate(("ioctl", "mmap", "mmap64", "munmap", "close", "dup2", "dup3", "mremap", "close_range"))})
        self.acquire = self.call(1, 11, 10, 12, [8, 77], associated_fd_identity=identity(8, 2, 128))
        self.allocate = self.call(1, 3, 20, 22, [0x100000, 32768, 0, 0, 77, 2],
            output=[0x100000, 32768, 0xAA, 0xB000, 77, 2])
        self.mapping = self.call(2, 0, 30, 32, [], retval=0x200000, fd_identity=identity(8, 2, 128),
            length=32768, offset=0xB000, protection=3, flags=1)
        self.created = self.call(1, 1, 40, 45, [0xAA, 0, 0, 1, 0, 0, 0],
            output=[0x80000000, 7, 0, 1, 0, 7, 3])  # event ID deliberately differs from slot index.
        self.native = dict(pid=100, write_complete=True, filter_policy="all_APIs_global",
            stop_evidence=dict(observation_window_closed=True, stop_result=0), dropped_events=0,
            recursive_suppressed=0, unknown_events=0, missing_user_data=0,
            domains=[dict(operations=[]) for _ in range(4)], events=[])
        self.native["domains"][2]["operations"] = [dict(id=1, name="hsa_signal_create"), dict(id=2, name="hsa_signal_destroy")]
        self.native["domains"][3]["operations"] = [dict(id=1, name="hsa_amd_signal_get_event_id")]
        self.native_call(1, 2, 1, 35, 50, dict(signal_handle=0xA000, initial_value=0, num_consumers=0))
        self.native_call(2, 3, 1, 60, 62, dict(signal_handle=0xA000, event_id=7, payload_observed=True))
        self.native_call(3, 2, 2, 70, 75, dict(signal_handle=0xA000))
        self.native_call(4, 2, 1, 100, 110, dict(signal_handle=0xA040, initial_value=0, num_consumers=0))
        self.native_call(5, 3, 1, 120, 122, dict(signal_handle=0xA040, event_id=7, payload_observed=True))
        self.observations = [dict(create_cid=4, event_query_cid=5, kfd_cursor_before=8, kfd_cursor_after=8)]
        self.target_epochs = None

    def call(self, api, op, start, end, values, *, output=None, retval=0, **changes):
        self.call_id += 1
        row = dict(call_id=self.call_id, tid=101, pid=100, api=api, operation=op,
            fd_identity=identity(7), associated_fd_identity={}, request=1000 + op if api == 1 else 0,
            argument_address=0xFFFF if api == 1 else 0, address=0, length=0, offset=0,
            protection=0, flags=0, target_fd=0, values=values + [0] * (8 - len(values)),
            payload_observed=True, output_valid=False, errno=0, retval=0)
        row.update(changes)
        a = deepcopy(row)
        a.update(ns=start, phase=1)
        b = deepcopy(row)
        b.update(ns=end, phase=2, retval=retval, output_valid=api == 1 and retval == 0)
        if output is not None:
            b["values"] = output + [0] * (8 - len(output))
        self.rows.extend((a, b))
        return self.call_id

    def native_call(self, cid, domain, op, start, end, payload):
        key = "raw_hsa_signal_event_payload" if domain == 3 else "raw_hsa_signal_payload"
        for phase, ns in ((1, start), (2, end)):
            actual = deepcopy(payload)
            if domain == 3:
                actual["output_address"] = 0xD000 + cid * 8
                if phase == 1:
                    actual["event_id"] = None
            elif op == 1 and phase == 1:
                actual["signal_handle"] = None
            self.native["events"].append(dict(cid=cid, tid=101, domain=domain, op=op, phase=phase, ns=ns,
                status_type=2 if phase == 2 else 0, status=0 if phase == 2 else None, **{key: actual}))

    def finish(self):
        self.rows.sort(key=lambda row: row["ns"])
        self.native["events"].sort(key=lambda row: row["ns"])
        for tape in (self.rows, self.native["events"]):
            for index, row in enumerate(tape, 1):
                row["seq"] = index
        self.report.update(attempted_rows=len(self.rows), retained_rows=len(self.rows), last_sequence=len(self.rows))
        return self.report

    def run(self):
        self.finish()
        control = dict(native_report=self.native, kfd_report=self.report, source_binding=self.source,
            create_cid=1, event_query_cid=2, kfd_create_event_call_id=self.created, target_epochs=self.target_epochs)
        return audit(self.report, native_report=self.native, event_observations=self.observations,
            source_binding=self.source, namespace_control=control, target_epochs=self.target_epochs)


class KfdEventAuditTests(unittest.TestCase):
    def partial(self, fixture):
        value = fixture.run()
        self.assertEqual(value["status"], "PARTIAL", value)
        self.assertTrue(value["issues"])
        self.assertEqual(value["event_proofs"], [])
        self.assertFalse(value["complete_workspace"])
        self.assertFalse(value["hardware_validated"])

    def test_cold_namespace_control_then_cached_event_borrow_use_original_gtt_page(self):
        fixture = KfdFixture()
        fixture.finish()
        before = deepcopy(fixture.report)
        result = fixture.run()
        self.assertEqual(result["status"], "KFD_EVENT_GTT_BACKING_PROVEN", result)
        self.assertEqual(result["event_proofs"][0]["event_id"], 7)
        self.assertEqual(result["event_proofs"][0]["slot_index"], 3)
        self.assertEqual(result["event_proofs"][0]["slot_address"], 0x200018)
        self.assertEqual(result["gtt_page_bytes"], 32768)
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(before["events"], fixture.report["events"])

    def test_shared_hardware_event_page_is_counted_once_for_multiple_signal_queries(self):
        f = KfdFixture()
        f.native_call(6, 3, 1, 130, 132, dict(signal_handle=0xA040, event_id=7, payload_observed=True))
        f.observations.append(dict(create_cid=4, event_query_cid=6, kfd_cursor_before=8, kfd_cursor_after=8))
        result = f.run()
        self.assertEqual(result["status"], "KFD_EVENT_GTT_BACKING_PROVEN", result)
        self.assertEqual(len(result["event_proofs"]), 2)
        self.assertEqual(len(result["gtt_pages"]), 1)

    def test_typed_event_state_controls_and_readonly_xnack_do_not_invent_storage(self):
        f = KfdFixture()
        f.call(1, 12, 51, 52, [7, 0])
        f.call(1, 13, 53, 54, [7, 0])
        f.call(1, 14, 55, 56, [2**64 - 1], output=[0])
        f.observations[0].update(kfd_cursor_before=14, kfd_cursor_after=14)
        result = f.run()
        self.assertEqual(result["status"], "KFD_EVENT_GTT_BACKING_PROVEN", result)
        self.assertEqual(result["gtt_page_bytes"], 32768)
        for mode in ("unknown_event", "pad", "xnack_mutator"):
            broken = deepcopy(f)
            for row in broken.rows:
                if row["operation"] == 12 and mode in ("unknown_event", "pad"):
                    row["values"][0 if mode == "unknown_event" else 1] += 1
                elif row["operation"] == 14 and mode == "xnack_mutator" and row["phase"] == 1:
                    row["values"][0] = 0
            with self.subTest(mode=mode):
                self.partial(broken)

    def test_runtime_and_scratch_controls_need_actual_closed_epoch_startup_boundary(self):
        for mode in (None, "late", "no_epoch", "mask", "scratch_gpu", "scratch_pad"):
            f = KfdFixture()
            f.native["epochs"] = [dict(id=8001, begin_ns=35, end_ns=190, ended=True)]
            f.target_epochs = [8001]
            f.call(1, 15, 13, 14, [0x8800, 3, 0], output=[0x8800, 3, 1])
            f.call(1, 16, 15, 16, [0x9000, 77, 0])
            f.observations[0].update(kfd_cursor_before=12, kfd_cursor_after=12)
            if mode == "no_epoch":
                f.target_epochs = None
            for row in f.rows:
                if mode == "late" and row["operation"] == 16:
                    row["ns"] += 25
                elif mode == "mask" and row["operation"] == 15:
                    row["values"][1] |= 4
                elif mode in ("scratch_gpu", "scratch_pad") and row["operation"] == 16:
                    row["values"][1 if mode == "scratch_gpu" else 2] += 1
            with self.subTest(mode=mode):
                if mode:
                    self.partial(f)
                else:
                    result = f.run()
                    self.assertEqual(result["status"], "KFD_EVENT_GTT_BACKING_PROVEN", result)
                    self.assertEqual(len(result["startup_control_obligations"]), 2)
                    self.assertTrue(all(row["gpu_backing_zero_claimed"] is False
                        for row in result["startup_control_obligations"]))

    def test_available_memory_is_exact_readonly_evidence_and_never_a_workspace_budget(self):
        for available in (0, 256, 2**64 - 1):
            f = KfdFixture()
            f.call(1, 17, 115, 116, [77, 0, 0], output=[77, available, 0])
            f.observations[0].update(kfd_cursor_before=10, kfd_cursor_after=10)
            result = f.run()
            with self.subTest(available=available):
                self.assertEqual(result["status"], "KFD_EVENT_GTT_BACKING_PROVEN", result)
                self.assertEqual(result["gtt_page_bytes"], 32768)
                self.assertEqual(result["observed_available_memory_queries"][0]["available_bytes"], available)
                self.assertFalse(result["observed_available_memory_queries"][0]["used_for_workspace_bytes"])
        for mode in ("wrong_gpu", "changed_gpu", "pad", "unobserved"):
            f = KfdFixture()
            cid = f.call(1, 17, 115, 116, [77, 0, 0], output=[77, 100, 0])
            f.observations[0].update(kfd_cursor_before=10, kfd_cursor_after=10)
            for row in f.rows:
                if row["call_id"] == cid:
                    if mode == "wrong_gpu":
                        row["values"][0] = 88
                    elif mode == "changed_gpu" and row["phase"] == 2:
                        row["values"][0] = 88
                    elif mode == "pad":
                        row["values"][2] = 1
                    elif mode == "unobserved":
                        row["payload_observed"] = False
            with self.subTest(mode=mode):
                self.partial(f)

    def test_vram_unknown_userptr_and_mixed_memory_types_cannot_be_called_cpu_backing(self):
        for flags in (0, 1, 4, 3, 8, 16):
            f = KfdFixture()
            for row in f.rows:
                if row["call_id"] == f.allocate:
                    row["values"][5] = flags
            with self.subTest(flags=flags):
                self.partial(f)

    def test_free_destroy_partial_unmap_fixed_overlap_and_moved_mapping_invalidate_generations(self):
        for mode in ("free", "destroy", "unmap", "fixed", "mremap"):
            f = KfdFixture()
            if mode == "free":
                f.call(1, 4, 115, 116, [0xAA])
            elif mode == "destroy":
                f.call(1, 2, 115, 116, [7])
            elif mode == "unmap":
                f.call(4, 0, 115, 116, [], address=0x200010, length=8)
            elif mode == "fixed":
                f.call(2, 0, 115, 116, [], retval=0x200000, address=0x200000, length=4096, flags=16)
            else:
                f.call(8, 0, 115, 116, [65536, 0], retval=0x300000, address=0x200000, length=32768)
            f.observations[0].update(kfd_cursor_before=10, kfd_cursor_after=10)
            with self.subTest(mode=mode):
                self.partial(f)

    def test_fd_close_dup_rebinding_and_close_range_cannot_reuse_old_namespace(self):
        for api in (5, 6, 7, 9):
            f = KfdFixture()
            changes = dict(fd_identity=identity(7)) if api == 5 else (
                dict(address=7, length=7) if api == 9 else dict(fd_identity=identity(10), target_fd=7))
            f.call(api, 0, 115, 116, [], retval=0 if api in (5, 9) else 7, **changes)
            f.observations[0].update(kfd_cursor_before=10, kfd_cursor_after=10)
            with self.subTest(api=api):
                self.partial(f)

    def test_destroy_cannot_hide_a_real_event_destruction_behind_a_different_entry_id(self):
        f = KfdFixture()
        f.call(1, 2, 115, 116, [999, 0], output=[7, 0])
        f.observations[0].update(kfd_cursor_before=10, kfd_cursor_after=10)
        result = f.run()
        self.assertEqual(result["status"], "PARTIAL", result)
        self.assertTrue(any("DESTROY_EVENT input-only" in issue for issue in result["issues"]))
        self.assertEqual(result["event_proofs"], [])

    def test_free_cannot_hide_page_release_behind_another_live_allocation_handle(self):
        f = KfdFixture()
        f.call(1, 3, 90, 92, [0x300000, 4096, 0, 0, 77, 2],
            output=[0x300000, 4096, 0xBB, 0xB100, 77, 2])
        f.call(1, 4, 115, 116, [0xBB], output=[0xAA])
        f.observations[0].update(kfd_cursor_before=12, kfd_cursor_after=12)
        result = f.run()
        self.assertEqual(result["status"], "PARTIAL", result)
        self.assertTrue(any("FREE_MEMORY input-only" in issue for issue in result["issues"]))
        self.assertEqual(result["event_proofs"], [])

    def test_event_slot_not_id_bounds_alias_offset_and_gpu_render_identity_are_exact(self):
        for mode in ("slot", "handle", "offset", "render", "gpu", "pci"):
            f = KfdFixture()
            if mode in ("slot", "handle"):
                for row in f.rows:
                    if row["call_id"] == f.created:
                        if mode == "slot" and row["phase"] == 2:
                            row["values"][6] = 4096
                        elif mode == "handle" and row["phase"] == 1:
                            row["values"][0] = 0xBB
            elif mode in ("offset", "render"):
                for row in f.rows:
                    if row["call_id"] == f.mapping:
                        if mode == "offset":
                            row["offset"] += 4096
                        else:
                            row["fd_identity"] = identity(9, 2, 129)
            elif mode == "gpu":
                f.source["device_nodes"][0]["gpu_id"] = 78
            else:
                f.source["device_nodes"][0]["pci_bus_id"] = "not-a-pci-device"
            with self.subTest(mode=mode):
                self.partial(f)

    def test_native_queries_require_actual_status_generation_and_one_namespace(self):
        for mode in ("status", "handle", "getid", "reused", "pid", "missing_control_create"):
            f = KfdFixture()
            if mode == "status":
                next(row for row in f.native["events"] if row["cid"] == 5 and row["phase"] == 2)["status"] = 1
            elif mode in ("handle", "getid"):
                for row in f.native["events"]:
                    if row["cid"] == 5 and (mode == "handle" or row["phase"] == 2):
                        row["raw_hsa_signal_event_payload"]["signal_handle" if mode == "handle" else "event_id"] += 1
            elif mode == "reused":
                f.native_call(6, 2, 2, 115, 116, dict(signal_handle=0xA040))
            elif mode == "pid":
                f.native["pid"] += 1
            else:
                for row in f.rows:
                    if row["call_id"] == f.created:
                        row["ns"] += 15
            with self.subTest(mode=mode):
                self.partial(f)

    def test_loss_fork_inheritance_stale_cursor_bad_abi_and_missing_source_stay_partial(self):
        for mode in ("drop", "read", "forward", "reentrant", "fork", "oldpid", "cursor", "header", "abi"):
            f = KfdFixture()
            if mode in ("drop", "read", "forward", "reentrant"):
                f.report[{"drop": "dropped_rows", "read": "read_errors", "forward": "forward_errors", "reentrant": "reentrant_calls"}[mode]] = 1
            elif mode == "fork":
                f.report["fork_observation"]["inherited_kfd_resources_unsupported"] = True
            elif mode == "oldpid":
                f.rows[0]["pid"] -= 1
            elif mode == "cursor":
                f.observations[0]["kfd_cursor_before"] = 100
            elif mode == "header":
                f.source.pop("kfd_header_sha256")
            else:
                f.abi["gtt_flag"] = 0
                f.source["abi"] = deepcopy(f.abi)
            with self.subTest(mode=mode):
                self.partial(f)

    def test_actual_forward_target_source_bytes_and_loaded_aliases_are_bound(self):
        for mode in (None, "unknown_target", "unresolved_used", "source_path", "source_hash"):
            f = KfdFixture()
            if mode is None:
                f.source["artifacts"]["libc"].update(loaded_path="/lib/actual-libc.so", realpath="/usr/lib/actual-libc.so")
                for binding in f.report["libc_bindings"].values():
                    binding["module_path"] = "/lib/actual-libc.so"
            elif mode == "unknown_target":
                f.report["libc_bindings"]["ioctl"]["module_path"] = "/tmp/other.so"
            elif mode == "unresolved_used":
                f.report["libc_bindings"]["ioctl"].update(address=None, dladdr_success=False)
            elif mode == "source_path":
                f.report["compiled_source_path"] = "/other/source.cpp"
            else:
                f.source["artifacts"]["probe_source"]["sha256"] = "f" * 64
            with self.subTest(mode=mode):
                if mode:
                    self.partial(f)
                else:
                    self.assertEqual(f.run()["status"], "KFD_EVENT_GTT_BACKING_PROVEN")


if __name__ == "__main__":
    unittest.main()
