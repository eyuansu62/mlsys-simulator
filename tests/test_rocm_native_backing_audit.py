"""CPU fixtures for the backing component, never real GPU measurements."""

from copy import deepcopy
import unittest

import importlib.util
from pathlib import Path
import sys

_TOOL_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "rocm_workspace_audit"


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, _TOOL_ROOT / (name + ".py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load packaged ROCm audit source: " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_tool("analyze_native_probe")
audit_native_backing = _load_tool("native_backing_audit").audit_native_backing
report_v2 = _load_tool("_native_probe_fixture").report_v2


class Fixture:
    def __init__(self):
        self.native = report_v2(sentinel=False)
        self.native.update(init_ns=10, finalize_ns=800, max_events=1000)
        self.native["stop_evidence"].update(begin_ns=700, end_ns=710)
        self.native["epochs"] = [
            dict(id=5001, label="controlled_lt_only", begin_ns=100, end_ns=300, ended=True),
            dict(id=16, label="prefill-16", begin_ns=400, end_ns=600, ended=True),
        ]
        self.events = []
        self.ids = 200
        self.op(0, 0, "hipDeviceSynchronize", 0, 1)
        self.op(2, 0, "hsa_signal_create", 64, 2)
        self.op(3, 0, "hsa_amd_memory_pool_allocate", 17, 2)
        self.op(3, 1, "hsa_amd_memory_pool_free", 18, 2)
        self.op(6, 0, "MEMORY_ALLOCATION_ALLOCATE", 1)
        self.op(6, 1, "MEMORY_ALLOCATION_FREE", 2)
        self.op(7, 0, "SCRATCH_MEMORY_ALLOC", 1024)
        self.op(7, 1, "SCRATCH_MEMORY_FREE", 1024)
        self.op(8, 0, "ncclGetVersion", 512, 3)
        self.call(8, 0, 20, 30, epoch=0)
        self.alloc(0x1000, 1024, 40, epoch=0)
        self.alloc(0x2000, 256, 120, epoch=5001)
        self.alloc(0x3000, 128, 140, epoch=5001)
        self.alloc(0x4000, 512, 160, epoch=5001)
        self.alloc(0x8000, 64, 180, epoch=5001, agent=0)
        self.scratch(4096, 200, epoch=5001)
        self.call(0, 0, 250, 260, epoch=5001)
        sync_id = self.call(0, 0, 500, 530, epoch=16)
        self.call(2, 0, 510, 520, epoch=16, ancestor=sync_id)
        snap = {
            "segments": [
                {"device": 0, "address": 0x1000, "total_size": 1024},
                {"device": 0, "address": 0x4000, "total_size": 512},
            ]
        }
        self.snapshots = {16: {"before": deepcopy(snap), "after": deepcopy(snap)}}
        self.binding = {
            "epoch_id": 5001,
            "epoch_label": "controlled_lt_only",
            "entrypoint": "ATen.getCUDABlasLtWorkspace",
            "lt_only": True,
            "driver_source_sha256": "a" * 64,
            "driver_source_reference": "CPU fixture driver",
        }

    def op(self, domain, ident, name, flags, status_type=0):
        ops = self.native["domains"][domain]["operations"]
        ops[:] = [o for o in ops if o["id"] != ident]
        ops.append(
            dict(
                id=ident,
                name=name,
                flags=flags,
                decoded_exit_status_type=status_type,
                enter_count=0,
                exit_count=0,
                other_phase_count=0,
                filtered_nonmutator_events=0,
            )
        )

    def event(self, domain, op, phase, ns, epoch, cid, ancestor=0, status=0, **payload):
        meta = next(o for o in self.native["domains"][domain]["operations"] if o["id"] == op)
        st = meta["decoded_exit_status_type"] if phase == 2 else 0
        row = dict(
            domain=domain,
            op=op,
            phase=phase,
            ns=ns,
            epoch=epoch,
            cid=cid,
            ancestor=ancestor,
            tid=1,
            flags=meta["flags"],
            status_type=st,
            status=status if st else None,
            **payload,
        )
        self.events.append(row)
        return row

    def call(self, domain, op, start, end, *, epoch, ancestor=0, status=0):
        self.ids += 1
        cid = self.ids
        self.event(domain, op, 1, start, epoch, cid, ancestor)
        self.event(domain, op, 2, end, epoch, cid, ancestor, status=status)
        return cid

    def alloc(self, address, size, start, *, epoch, agent=1, status=0):
        self.ids += 1
        cid = self.ids
        self.event(3, 0, 1, start, epoch, cid)
        self.event(
            6,
            0,
            1,
            start + 1,
            epoch,
            cid,
            raw_allocation_payload=dict(
                address_or_handle=0, size_bytes=size, agent_handle=agent, success_proven=False
            ),
        )
        self.event(
            6,
            0,
            2,
            start + 2,
            epoch,
            cid,
            raw_allocation_payload=dict(
                address_or_handle=address, size_bytes=size, agent_handle=agent, success_proven=False
            ),
        )
        self.event(3, 0, 2, start + 3, epoch, cid, status=status)
        return cid

    def free(self, address, start, *, epoch, status=0):
        self.ids += 1
        cid = self.ids
        self.event(3, 1, 1, start, epoch, cid)
        for phase in (1, 2):
            self.event(
                6,
                1,
                phase,
                start + phase,
                epoch,
                cid,
                raw_allocation_payload=dict(
                    address_or_handle=address, size_bytes=0, agent_handle=0, success_proven=False
                ),
            )
        self.event(3, 1, 2, start + 3, epoch, cid, status=status)

    def scratch(self, size, start, *, epoch, agent=1, queue=7, free=False):
        self.ids += 1
        for phase in (1, 2):
            self.event(
                7,
                int(free),
                phase,
                start + phase,
                epoch,
                self.ids,
                raw_scratch_payload=dict(
                    agent_handle=agent,
                    queue_handle=queue,
                    flags=0,
                    args_kind=phase,
                    allocation_size_known=not free and phase == 2,
                    allocation_size_bytes=size if not free and phase == 2 else None,
                    workspace_attribution_supported=False,
                ),
            )

    def finish(self):
        self.events.sort(key=lambda r: r["ns"])
        for i, row in enumerate(self.events, 1):
            row.update(seq=i, callback_seq=i)
        self.native["events"] = self.events
        count = len(self.events)
        self.native.update(
            observed_callbacks=count,
            attempted_events=count,
            retained_events=count,
            filtered_nonmutator_events=0,
        )
        for domain in self.native["domains"]:
            domain["operation_count"] = len(domain["operations"])
            for op in domain["operations"]:
                rows = [r for r in self.events if r["domain"] == domain["index"] and r["op"] == op["id"]]
                op.update(
                    enter_count=sum(r["phase"] == 1 for r in rows),
                    exit_count=sum(r["phase"] == 2 for r in rows),
                )
        for epoch in self.native["epochs"]:
            rows = [r for r in self.events if r["epoch"] == epoch["id"]]
            sync = [
                r["ns"]
                for r in rows
                if r["domain"] == 0 and r["op"] == 0 and r["phase"] == 2 and r["status"] == 0
            ]
            epoch.update(
                api_events=len(rows),
                mutator_events=sum(bool(r["flags"] & 1535) for r in rows),
                unsupported_events=sum(bool(r["flags"] & 1532) for r in rows),
                dropped_at_begin=0,
                dropped_at_end=0,
                recursive_at_begin=0,
                recursive_at_end=0,
                successful_device_sync_exit_ns=max(sync) if sync else 0,
            )
        return self.native

    def audit(self, **kwargs):
        return audit_native_backing(
            self.finish(),
            self.snapshots,
            device_to_agent=kwargs.get("device_to_agent", {0: 1}),
            ctor_binding=kwargs.get("ctor_binding", self.binding),
        )


class NativeBackingAuditTests(unittest.TestCase):
    def test_exact_backing_and_controlled_ctor_are_separate_from_torch_ownership(self):
        fixture = Fixture()
        before = deepcopy(fixture.snapshots)
        result = fixture.audit()
        self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(result["native_fixed_runtime_bytes"], 256 + 128 + 4096)
        self.assertEqual(result["unknown_allocations"], [])
        rows = result["epochs"][0]["before"]["owner_table"]
        owners = {r["address"]: r["owner"] for r in rows}
        self.assertEqual(owners[0x1000], "torch_reserved_segment_backing")
        self.assertEqual(owners[0x4000], "torch_reserved_segment_backing")
        self.assertEqual(owners[0x2000], "controlled_lt_ctor_scope_private_backing")
        self.assertEqual(owners[0x8000], "excluded_actual_cpu_backing")
        self.assertEqual(len(result["epochs"][0]["allowed_fence_signal_calls"]), 1)
        self.assertEqual(fixture.snapshots, before)

    def test_unknown_gpu_backing_is_not_relabelled_runtime(self):
        fixture = Fixture()
        fixture.alloc(0x5000, 64, 60, epoch=0)
        result = fixture.audit()
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["native_fixed_runtime_bytes"])
        self.assertTrue(any(r["address"] == 0x5000 for r in result["unknown_allocations"]))

    def test_controlled_ctor_requires_label_source_and_lt_only_binding(self):
        for field, value in (
            ("lt_only", False),
            ("epoch_label", "guessed"),
            ("driver_source_sha256", "bad"),
            ("entrypoint", "getCurrentBlasHandle"),
            ("epoch_id", 5001.0),
        ):
            with self.subTest(field=field):
                fixture = Fixture()
                fixture.binding[field] = value
                result = fixture.audit()
                self.assertEqual(result["status"], "PARTIAL")
                self.assertIsNone(result["native_fixed_runtime_bytes"])

    def test_segment_matching_is_exact_and_uses_actual_gpu_agent(self):
        fixture = Fixture()
        fixture.snapshots[16]["before"]["segments"][0]["total_size"] += 1
        self.assertEqual(fixture.audit()["status"], "PARTIAL")
        self.assertEqual(Fixture().audit(device_to_agent={0: 0})["status"], "PARTIAL")

    def test_failed_allocation_or_free_does_not_mutate_the_successful_ledger(self):
        fixture = Fixture()
        cid = next(
            r["cid"]
            for r in fixture.events
            if r.get("raw_allocation_payload", {}).get("address_or_handle") == 0x2000
        )
        next(r for r in fixture.events if r["domain"] == 3 and r["phase"] == 2 and r["cid"] == cid)[
            "status"
        ] = 1001
        result = fixture.audit()
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["successful_normalized_call_counts"]["MEMORY_ALLOCATION_ALLOCATE"], 4)
        fixture = Fixture()
        fixture.free(0x2000, 320, epoch=0, status=1001)
        result = fixture.audit()
        self.assertEqual(result["status"], "PARTIAL")
        self.assertTrue(any(r["address"] == 0x2000 for r in result["epochs"][0]["before"]["owner_table"]))

    def test_unpaired_and_unlinked_records_remain_partial(self):
        fixture = Fixture()
        fixture.events.remove(next(r for r in fixture.events if r["domain"] == 6 and r["phase"] == 1))
        self.assertEqual(fixture.audit()["status"], "PARTIAL")
        fixture = Fixture()
        next(r for r in fixture.events if r["domain"] == 6 and r["phase"] == 2)["cid"] = 9999
        self.assertEqual(fixture.audit()["status"], "PARTIAL")

    def test_address_reuse_gets_a_new_generation(self):
        fixture = Fixture()
        fixture.alloc(0x5000, 64, 60, epoch=0)
        fixture.free(0x5000, 80, epoch=0)
        fixture.alloc(0x5000, 128, 220, epoch=5001)
        result = fixture.audit()
        self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
        row = next(r for r in result["epochs"][0]["before"]["owner_table"] if r["address"] == 0x5000)
        self.assertEqual(row["generation"], 2)

    def test_pool_growth_is_whole_capacity_not_incremental_sum(self):
        fixture = Fixture()
        fixture.scratch(8192, 220, epoch=5001)
        result = fixture.audit()
        self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
        self.assertEqual(result["epochs"][0]["before"]["private_scratch_pool_reserve_bytes"], 8192)
        self.assertEqual(result["native_fixed_runtime_bytes"], 256 + 128 + 8192)

    def test_any_formal_private_scratch_or_native_allocation_rejects_fixed_reserve(self):
        fixture = Fixture()
        fixture.scratch(8192, 450, epoch=16)
        self.assertEqual(fixture.audit()["status"], "PARTIAL")
        fixture = Fixture()
        fixture.alloc(0x5000, 64, 450, epoch=16)
        self.assertEqual(fixture.audit()["status"], "PARTIAL")

    def test_signal_requires_typed_success_and_real_device_sync_ancestor(self):
        for mutation in (
            lambda r: r.update(ancestor=0),
            lambda r: r.update(status_type=0, status=None),
            lambda r: r.update(status=1001),
        ):
            fixture = Fixture()
            for row in fixture.events:
                if row["domain"] == 2 and row["phase"] == 2:
                    mutation(row)
            self.assertEqual(fixture.audit()["status"], "PARTIAL")

    def test_vmm_ipc_and_async_pools_are_never_accepted_from_a_plausible_peak(self):
        for name, flags in (
            ("hsa_amd_vmem_handle_create", 5),
            ("hipIpcOpenMemHandle", 8),
            ("hipMallocAsync", 17),
        ):
            fixture = Fixture()
            fixture.op(0, 1, name, flags, 1)
            fixture.call(0, 1, 320, 325, epoch=0)
            result = fixture.audit()
            self.assertEqual(result["status"], "PARTIAL")
            self.assertIn("unsupported_native_memory_surface", result["problem_counts"])

    def test_bad_setup_loss_and_nonfinite_input_fail_closed(self):
        for field, value in (("dropped_events", 1), ("stop_status", 2), ("unknown_events", 1)):
            fixture = Fixture()
            fixture.native[field] = value
            self.assertEqual(fixture.audit()["status"], "PARTIAL")
        fixture = Fixture()
        next(
            r
            for r in fixture.events
            if r.get("raw_allocation_payload", {}).get("address_or_handle") == 0x2000
        )["raw_allocation_payload"]["size_bytes"] = float("nan")
        self.assertEqual(fixture.audit()["status"], "PARTIAL")

    def test_scratch_repeated_none_correlation_pairs_use_actual_agent_queue(self):
        fixture = Fixture()
        fixture.scratch(8192, 220, epoch=5001)
        for row in fixture.events:
            if row["domain"] == 7:
                row["cid"] = 0
        result = fixture.audit()
        self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
        self.assertEqual(result["native_fixed_runtime_bytes"], 256 + 128 + 8192)
        self.assertEqual(len(result["epochs"][0]["before"]["private_scratch_pool_table"]), 1)

    def test_unknown_free_and_overlapping_backing_are_not_repaired(self):
        fixture = Fixture()
        fixture.free(0xDEAD, 320, epoch=0)
        self.assertEqual(fixture.audit()["status"], "PARTIAL")
        fixture = Fixture()
        fixture.alloc(0x1100, 512, 220, epoch=5001)
        result = fixture.audit()
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIn("overlapping_native_backing_ranges", result["problem_counts"])

    def test_normalized_none_correlation_and_signal_flags_are_not_trusted(self):
        fixture = Fixture()
        cid = next(
            r["cid"]
            for r in fixture.events
            if r.get("raw_allocation_payload", {}).get("address_or_handle") == 0x2000
        )
        for row in fixture.events:
            if row["cid"] == cid:
                row["cid"] = 0
        self.assertEqual(fixture.audit()["status"], "PARTIAL")
        fixture = Fixture()
        fixture.op(2, 0, "hsa_signal_create", 65, 2)
        for row in fixture.events:
            if row["domain"] == 2:
                row["flags"] = 65
        self.assertEqual(fixture.audit()["status"], "PARTIAL")

    def test_whole_pool_agent_and_byte_values_must_be_observed_and_valid(self):
        for mutation in (
            lambda p: p.update(agent_handle=0),
            lambda p: p.update(allocation_size_known=False),
            lambda p: p.update(allocation_size_bytes=True),
        ):
            fixture = Fixture()
            mutation(
                next(r for r in fixture.events if r["domain"] == 7 and r["phase"] == 2)["raw_scratch_payload"]
            )
            self.assertEqual(fixture.audit()["status"], "PARTIAL")

    def test_exact_hsa_queries_preserve_flags_and_do_not_claim_unobserved_status(self):
        fixture = Fixture()
        fixture.op(3, 2, "hsa_amd_agent_iterate_memory_pools", 16, 0)
        fixture.op(3, 3, "hsa_amd_memory_get_preferred_copy_engine", 128, 2)
        fixture.call(3, 2, 310, 315, epoch=0)
        fixture.call(3, 3, 320, 325, epoch=0)
        before = deepcopy(fixture.finish())
        result = fixture.audit()
        self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
        self.assertEqual(fixture.native, before)
        audit = {row["operation"]: row for row in result["explicit_hsa_query_surface_audit"]}
        enum = audit["hsa_amd_agent_iterate_memory_pools"]
        self.assertEqual(enum["raw_flags"], 16)
        self.assertFalse(enum["successful_query_return_proven"])
        self.assertFalse(enum["nested_allocation_events_discarded"])
        self.assertTrue(audit["hsa_amd_memory_get_preferred_copy_engine"]["typed_exit_status_observed"])

    def test_query_whitelist_does_not_discard_nested_real_gpu_allocation(self):
        fixture = Fixture()
        fixture.op(3, 2, "hsa_amd_agent_iterate_memory_pools", 16, 0)
        parent = fixture.call(3, 2, 310, 350, epoch=0)
        cid = fixture.alloc(0x5000, 64, 320, epoch=0)
        for row in fixture.events:
            if row["cid"] == cid:
                row["ancestor"] = parent
        result = fixture.audit()
        self.assertEqual(result["status"], "PARTIAL")
        self.assertTrue(any(r["address"] == 0x5000 for r in result["unknown_allocations"]))

    def test_query_exception_requires_exact_api_phase_status_and_outside_formal(self):
        for name, flags, status_type, epoch, start in (
            ("hsa_amd_agent_iterate_memory_pools_future", 16, 0, 0, 310),
            ("hsa_amd_memory_get_preferred_copy_engine", 128, 0, 0, 310),
            ("hsa_amd_agent_iterate_memory_pools", 16, 0, 16, 450),
            ("hsa_amd_memory_get_preferred_copy_engine", 132, 2, 0, 310),
        ):
            fixture = Fixture()
            fixture.op(3, 2, name, flags, status_type)
            fixture.call(3, 2, start, start + 5, epoch=epoch)
            self.assertEqual(fixture.audit()["status"], "PARTIAL")
        fixture = Fixture()
        fixture.op(3, 2, "hsa_amd_memory_get_preferred_copy_engine", 128, 2)
        cid = fixture.call(3, 2, 310, 315, epoch=0)
        fixture.events.remove(next(r for r in fixture.events if r["cid"] == cid and r["phase"] == 2))
        self.assertEqual(fixture.audit()["status"], "PARTIAL")

    def test_host_unlock_is_only_explained_in_paired_successful_copy_scope(self):
        fixture = Fixture()
        fixture.op(0, 1, "hipMemcpyWithStream", 0, 1)
        fixture.op(3, 2, "hsa_amd_memory_unlock", 32, 2)
        parent = fixture.call(0, 1, 310, 360, epoch=0)
        fixture.call(3, 2, 320, 330, epoch=0, ancestor=parent)
        result = fixture.audit()
        self.assertEqual(result["status"], "NATIVE_BACKING_CHECKS_PASSED", result["issues"])
        row = result["host_registration_cleanup_audit"][0]
        self.assertTrue(row["explained_host_registration_cleanup"])
        self.assertTrue(row["argument_owner_not_inferred"])
        self.assertFalse(row["host_registration_lifetime_verified"])
        self.assertFalse(row["gpu_backing_replay_mutation_applied"])
        self.assertEqual(row["raw_flags"], 32)
        self.assertEqual(result["native_fixed_runtime_bytes"], 256 + 128 + 4096)

    def test_host_unlock_missing_or_wrong_parent_failed_or_formal_is_partial(self):
        for parent_name, parent_status, unlock_status, epoch, begin in (
            ("hipFree", 0, 0, 0, 310),
            ("hipMemcpyWithStream", 1, 0, 0, 310),
            ("hipMemcpyWithStream", 0, 1001, 0, 310),
            ("hipMemcpyWithStream", 0, 0, 16, 450),
        ):
            fixture = Fixture()
            fixture.op(0, 1, parent_name, 0, 1)
            fixture.op(3, 2, "hsa_amd_memory_unlock", 32, 2)
            parent = fixture.call(0, 1, begin, begin + 30, epoch=epoch, status=parent_status)
            fixture.call(3, 2, begin + 10, begin + 20, epoch=epoch, ancestor=parent, status=unlock_status)
            self.assertEqual(fixture.audit()["status"], "PARTIAL")
        fixture = Fixture()
        fixture.op(3, 2, "hsa_amd_memory_unlock", 32, 2)
        fixture.call(3, 2, 310, 315, epoch=0)
        self.assertEqual(fixture.audit()["status"], "PARTIAL")

    def test_host_cleanup_exception_never_hides_vmm_pool_or_unknown_free(self):
        for operation, flags in (("hipMemMap", 4), ("hipIpcOpenMemHandle", 8), ("hipMallocAsync", 17)):
            fixture = Fixture()
            fixture.op(0, 1, "hipMemcpyWithStream", 0, 1)
            fixture.op(3, 2, "hsa_amd_memory_unlock", 32, 2)
            fixture.op(0, 2, operation, flags, 1)
            parent = fixture.call(0, 1, 310, 360, epoch=0)
            fixture.call(3, 2, 320, 330, epoch=0, ancestor=parent)
            fixture.call(0, 2, 340, 345, epoch=0, ancestor=parent)
            self.assertEqual(fixture.audit()["status"], "PARTIAL")
        fixture = Fixture()
        fixture.op(0, 1, "hipMemcpyWithStream", 0, 1)
        fixture.op(3, 2, "hsa_amd_memory_unlock", 32, 2)
        parent = fixture.call(0, 1, 310, 360, epoch=0)
        fixture.call(3, 2, 320, 330, epoch=0, ancestor=parent)
        fixture.free(0xDEAD, 340, epoch=0)
        self.assertEqual(fixture.audit()["status"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()
