"""Storage-owner evidence and source-shaped Workspace collection on CPU."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import gc

from mlsys_sim import SimulationError
from mlsys_sim.hardware import CollectionError
from mlsys_sim.workspace_capture import WorkspaceRecorder, collect_workspace, PinnedVllmWorkspaceObserver
from test_rocm_steps import FakeClock
import test_rocm_steps as rocm_fixture


EXCLUDED = ["weights", "kv_pool", "graphs", "runtime", "pp_buffers", "communication_buffers"]


class FakeOOM(RuntimeError):
    pass


class LedgerAdapter:
    def __init__(self, tp=1, *, complete=True, zero=False):
        self.tp, self.complete, self.zero = tp, complete, zero
        self.recorder = None
        self.runs = self.begins = self.ends = self.closes = 0
        self.failure = None
        self.fail_end = False
        self.fail_begin = False
        self.skip_rank = None
    def metadata(self):
        return {"runtime_name": "workspace-fixture-runtime", "runtime_version": "fixture-v1",
            "timing_boundary": "step_includes_output_and_sampling_v1", "includes_local_collectives": True,
            "workspace_attribution": "runtime_ledger", "complete_workspace_attribution": self.complete,
            "includes_non_torch": True, "excluded_categories": list(EXCLUDED)}
    def prepare(self, sample):
        pass
    def begin_workspace_capture(self, sample, recorder):
        self.begins += 1
        self.recorder = recorder
        if self.fail_begin:
            raise RuntimeError("begin failed after partial installation")
    def run_step(self, sample):
        self.runs += 1
        if self.recorder is not None and not self.zero:
            for rank in range(self.tp):
                self.recorder.allocate(rank, "a", "same-backing", 40 + rank * 20)
                self.recorder.allocate(rank, "view", "same-backing", 40 + rank * 20)
                self.recorder.request_free(rank, "a")
                self.recorder.release_completed(rank, "view")
                self.recorder.release_completed(rank, "a")
        if self.failure:
            raise self.failure
    def end_workspace_capture(self, sample, recorder):
        self.ends += 1
        self.recorder = None
        for rank in range(self.tp):
            if rank != self.skip_rank:
                recorder.complete_rank(rank)
        if self.fail_end:
            raise RuntimeError("end failed")
    def close(self):
        self.closes += 1


class WorkspaceRecorderTests(unittest.TestCase):
    def test_aliases_deduplicate_full_storage_and_pending_free_stays_live(self):
        recorder = WorkspaceRecorder(2)
        recorder.allocate(0, "view-a", "storage-a", 100)
        recorder.allocate(0, "view-b", "storage-a", 100)
        recorder.request_free(0, "view-a")
        recorder.release_completed(0, "view-b")
        recorder.allocate(0, "other", "storage-b", 50)
        self.assertEqual(recorder.export_capture()["live_bytes_per_rank"], [150, 0])
        recorder.release_completed(0, "view-a")
        recorder.release_completed(0, "other")
        recorder.complete_rank(0)
        recorder.complete_rank(1)
        self.assertEqual(recorder.seal(), [{"tp_rank": 0, "peak_bytes": 150}, {"tp_rank": 1, "peak_bytes": 0}])
        self.assertEqual(recorder.seal()[0]["peak_bytes"], 150)
        sealed = recorder.export_capture()
        with self.assertRaises(SimulationError):
            recorder.allocate(0, "late", "late", 20)
        self.assertEqual(recorder.export_capture(), sealed)

    def test_unknown_zero_and_epoch_leaks_do_not_become_success(self):
        recorder = WorkspaceRecorder(2)
        recorder.complete_rank(0)
        with self.assertRaises(SimulationError):
            recorder.seal()
        self.assertEqual(recorder.export_capture()["status"], "INVALID")
        recorder = WorkspaceRecorder(1)
        recorder.allocate(0, "a", "storage", 10)
        recorder.request_free(0, "a")
        with self.assertRaises(SimulationError) as raised:
            recorder.complete_rank(0)
        self.assertEqual(raised.exception.code, "WORKSPACE_EPOCH_LEAK")
        self.assertEqual(recorder.export_capture()["live_bytes_per_rank"], [10])

    def test_bad_events_seal_epoch_and_do_not_allow_recovery(self):
        operations = [lambda r: r.allocate(True, "x", "x", 1), lambda r: r.allocate(2, "x", "x", 1),
            lambda r: r.allocate(0, "x", "x", True), lambda r: r.allocate(0, "x", "x", -1),
            lambda r: r.release_completed(0, "unknown"), lambda r: r.request_free(0, "unknown")]
        for operation in operations:
            recorder = WorkspaceRecorder(1)
            with self.subTest(operation=operation), self.assertRaises(SimulationError):
                operation(recorder)
            self.assertEqual(recorder.export_capture()["status"], "INVALID")
            with self.assertRaises(SimulationError):
                recorder.seal()
        recorder = WorkspaceRecorder(1)
        recorder.allocate(0, "a", "same", 10)
        with self.assertRaises(SimulationError):
            recorder.allocate(0, "b", "same", 20)
        recorder = WorkspaceRecorder(1)
        recorder.allocate(0, "a", "same", 10)
        recorder.release_completed(0, "a")
        with self.assertRaises(SimulationError):
            recorder.release_completed(0, "a")

    def test_record_tensor_counts_backing_and_retains_no_tensor_reference(self):
        class Storage:
            def nbytes(self):
                return 1024
            def data_ptr(self):
                return 1000
        class Tensor:
            device = SimpleNamespace(type="cuda", index=3)
            def untyped_storage(self):
                return Storage()
        recorder = WorkspaceRecorder(1, device_indices=[3])
        first, second = Tensor(), Tensor()
        reference = weakref.ref(first)
        recorder.record_tensor(0, "tensor", first)
        recorder.record_tensor(0, "view", second)
        del first
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(recorder.export_capture()["live_bytes_per_rank"], [1024])
        recorder.release_completed(0, "tensor")
        recorder.release_completed(0, "view")
        recorder.record_tensor(0, "reused-address", Tensor())
        recorder.release_completed(0, "reused-address")
        recorder.complete_rank(0)
        self.assertEqual(recorder.seal()[0]["peak_bytes"], 1024)
        bad = WorkspaceRecorder(1)
        with self.assertRaises(SimulationError):
            bad.record_tensor(0, "wrong-device", Tensor())

    def test_event_count_and_byte_limits_never_publish_partial_zero(self):
        for options in ({"max_events": 1}, {"max_event_bytes": 10}):
            recorder = WorkspaceRecorder(1, **options)
            with self.subTest(options=options), self.assertRaises(SimulationError):
                recorder.allocate(0, "a", "storage", 4)
                recorder.release_completed(0, "a")
            self.assertEqual(recorder.export_capture()["status"], "INVALID")
            self.assertLessEqual(len(recorder.export_capture()["events"]), 1)

    def test_tensor_storage_namespace_cannot_alias_an_unrelated_native_storage(self):
        class Storage:
            def nbytes(self):
                return 10
            def data_ptr(self):
                return 1000
        class Tensor:
            device = SimpleNamespace(type="cuda", index=0)
            def untyped_storage(self):
                return Storage()
        for tensor_first in (False, True):
            recorder = WorkspaceRecorder(1)
            if tensor_first:
                recorder.record_tensor(0, "torch", Tensor())
            recorder.allocate(0, "native", "tensor-storage-1", 10)
            if not tensor_first:
                recorder.record_tensor(0, "torch", Tensor())
            recorder.release_completed(0, "native")
            recorder.release_completed(0, "torch")
            recorder.complete_rank(0)
            self.assertEqual(recorder.seal()[0]["peak_bytes"], 20)
        recorder = WorkspaceRecorder(1)
        recorder.record_tensor(0, "torch", Tensor())
        with self.assertRaises(SimulationError):
            recorder.allocate(0, "collision", "@tensor-storage:1", 10)
        self.assertEqual(recorder.export_capture()["status"], "INVALID")


class WorkspaceCollectionTests(unittest.TestCase):
    def environment(self, tp=1, pp=1):
        exp, inventory, torch, cuda, log = rocm_fixture.RocmStepTests().environment(tp=tp, pp=pp)
        cuda.OutOfMemoryError = FakeOOM
        return exp, inventory, torch, cuda, log
    def plan(self, **changes):
        value = rocm_fixture.RocmStepTests().plan(**changes)
        value["dataset_id"] = "workspace-test"
        return value
    def collect(self, exp, inventory, torch, adapter, plan, log):
        return collect_workspace(exp, plan, adapter, inventory, torch_module=torch,
            clock_ns=FakeClock(list(range(100)), log), allow_test_fixture=True)

    def test_complete_rank_ledger_is_canonical_and_allocator_peaks_stay_raw(self):
        exp, inventory, torch, cuda, log = self.environment(tp=2, pp=2)
        adapter = LedgerAdapter(tp=2)
        result = self.collect(exp, inventory, torch, adapter, self.plan(stage=1, warmup=1, iterations=2, mixed=True), log)
        self.assertEqual(adapter.runs, 3)
        self.assertEqual((adapter.begins, adapter.ends, adapter.closes), (2, 2, 1))
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(result["source"]["kind"], "test_fixture")
        records = result["dataset"]["records"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["workspace_bytes_per_rank"], [{"tp_rank": 0, "peak_bytes": 40}, {"tp_rank": 1, "peak_bytes": 60}])
        self.assertEqual(records[0]["request_phases"], ["decode", "prefill"])
        self.assertEqual(result["raw_records"][0]["allocator_after"][0]["peak_allocated_bytes"], 12002)
        self.assertEqual(result["raw_records"][0]["device_indices"], [2, 3])
        self.assertFalse(any(item[0] == "reset_peak" for item in log))

    def test_zero_requires_explicit_rank_completion_and_partial_has_no_dataset(self):
        exp, inventory, torch, cuda, log = self.environment()
        adapter = LedgerAdapter(zero=True)
        result = self.collect(exp, inventory, torch, adapter, self.plan(), log)
        self.assertEqual(result["dataset"]["records"][0]["workspace_bytes_per_rank"], [{"tp_rank": 0, "peak_bytes": 0}])
        adapter = LedgerAdapter(complete=False)
        result = self.collect(exp, inventory, torch, adapter, self.plan(), log)
        self.assertIsNone(result["dataset"])
        self.assertFalse(result["attribution_complete"])
        adapter = LedgerAdapter(zero=True)
        adapter.skip_rank = 0
        with self.assertRaises(CollectionError) as raised:
            self.collect(exp, inventory, torch, adapter, self.plan(), log)
        self.assertIsNone(raised.exception.artifacts["dataset"])
        self.assertEqual(raised.exception.artifacts["raw_records"][0]["workspace_ledger"]["status"], "INVALID")

    def test_partial_manager_declaration_never_qualifies_as_dynamic_workspace(self):
        exp, inventory, torch, cuda, log = self.environment()
        adapter = LedgerAdapter()
        original = adapter.metadata
        adapter.metadata = lambda: {**original(), "workspace_attribution": "managed_scratch_capacity"}
        adapter.workspace_evidence = lambda sample: {"scope": "managed_scratch_capacity", "capacity_bytes": 9000, "complete": False}
        result = self.collect(exp, inventory, torch, adapter, self.plan(), log)
        self.assertIsNone(result["dataset"])
        self.assertEqual(result["raw_records"][0]["partial_workspace_evidence"]["capacity_bytes"], 9000)
        adapter = LedgerAdapter()
        original = adapter.metadata
        adapter.metadata = lambda: {**original(), "includes_non_torch": False}
        self.assertIsNone(self.collect(exp, inventory, torch, adapter, self.plan(), log)["dataset"])

    def test_errors_preserve_original_type_and_always_uninstall_and_close(self):
        for failure in (FakeOOM("typed allocator failure"), RuntimeError("out of memory string is not typed OOM"),
                        AssertionError("Workspace is locked")):
            exp, inventory, torch, cuda, log = self.environment()
            adapter = LedgerAdapter()
            adapter.failure = failure
            adapter.fail_end = True
            with self.subTest(failure=failure), self.assertRaises(CollectionError) as raised:
                self.collect(exp, inventory, torch, adapter, self.plan(), log)
            self.assertEqual((adapter.ends, adapter.closes), (1, 1))
            self.assertIs(raised.exception.__cause__, failure)
            self.assertEqual(raised.exception.code == "WORKSPACE_OUT_OF_MEMORY", isinstance(failure, FakeOOM))
            self.assertEqual(raised.exception.artifacts["failure"]["error_type"], type(failure).__name__)
        exp, inventory, torch, cuda, log = self.environment()
        adapter = LedgerAdapter()
        adapter.fail_begin = True
        with self.assertRaises(CollectionError):
            self.collect(exp, inventory, torch, adapter, self.plan(), log)
        self.assertEqual((adapter.ends, adapter.closes), (1, 1))

    def test_fixture_permission_backend_and_warmup_failures_are_explicit(self):
        exp, inventory, torch, cuda, log = self.environment()
        adapter = LedgerAdapter()
        with self.assertRaises(CollectionError) as raised:
            collect_workspace(exp, self.plan(), adapter, inventory, torch_module=torch)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        self.assertEqual(adapter.closes, 1)
        torch.version.hip = None
        with self.assertRaises(CollectionError) as raised:
            self.collect(exp, inventory, torch, LedgerAdapter(), self.plan(), log)
        self.assertEqual(raised.exception.code, "HARDWARE_UNAVAILABLE")
        torch.version.hip = "fixture-hip-v1"
        adapter = LedgerAdapter()
        adapter.failure = FakeOOM("warmup")
        with self.assertRaises(CollectionError) as raised:
            self.collect(exp, inventory, torch, adapter, self.plan(warmup=1), log)
        self.assertEqual(raised.exception.code, "WORKSPACE_OUT_OF_MEMORY")
        self.assertEqual(raised.exception.artifacts["diagnostics"]["current_operation"], "warmup")

    def test_collection_limits_preserve_incomplete_evidence(self):
        exp, inventory, torch, cuda, log = self.environment()
        plan = self.plan(iterations=2)
        with patch("mlsys_sim.workspace_capture._MAX_TOTAL_EVENTS", 8), self.assertRaises(CollectionError) as raised:
            self.collect(exp, inventory, torch, LedgerAdapter(), plan, log)
        self.assertEqual(raised.exception.code, "WORKSPACE_CAPTURE_LIMIT")
        self.assertIsNone(raised.exception.artifacts["dataset"])
        with patch("mlsys_sim.workspace_capture._MAX_RAW_BYTES", 1024), self.assertRaises(CollectionError) as raised:
            self.collect(exp, inventory, torch, LedgerAdapter(), self.plan(), log)
        self.assertTrue(raised.exception.artifacts["raw_records"][0]["workspace_evidence_truncated"])
        self.assertNotIn("workspace_ledger", raised.exception.artifacts["raw_records"][0])
        plan = self.plan(iterations=10001)
        with self.assertRaises(CollectionError):
            self.collect(exp, inventory, torch, LedgerAdapter(), plan, log)


class ManagedWorkspaceTests(unittest.TestCase):
    class Tensor:
        def __init__(self, size):
            self.size = size
        def numel(self):
            return self.size
        def element_size(self):
            return 1
    class Manager:
        def __init__(self):
            self._num_ubatches = self._num_lanes = 1
            self._locked = False
            self._current_workspaces = [None]
            self.calls = 0
        def _ensure_workspace_size(self, required_bytes):
            self.calls += 1
            current = self._current_workspaces[0]
            if current is None or current.numel() < required_bytes:
                if self._locked:
                    raise AssertionError("Workspace is locked")
                current = ManagedWorkspaceTests.Tensor(required_bytes)
                self._current_workspaces[0] = current
            return current

    def test_manager_observer_is_partial_reversible_bounded_and_preserves_exception(self):
        manager = self.Manager()
        original = manager._ensure_workspace_size
        observer = PinnedVllmWorkspaceObserver(manager, runtime_version="0.29.1.dev0+g98dff2a81", allow_test_fixture=True, max_records=1)
        tensor = manager._ensure_workspace_size(100)
        self.assertIs(manager._ensure_workspace_size(20), tensor)
        manager._locked = True
        with self.assertRaises(AssertionError):
            manager._ensure_workspace_size(200)
        result = observer.export_capture()
        self.assertFalse(result["complete"])
        self.assertFalse(result["old_retained_storage_observable"])
        self.assertEqual(result["scope"], "managed_scratch_capacity")
        self.assertEqual(result["source"]["kind"], "test_fixture")
        self.assertEqual(result["records"][0]["snapshot"]["current_managed_capacity_bytes"], 100)
        self.assertEqual(result["dropped_records"], 2)
        observer.close()
        self.assertEqual(manager._ensure_workspace_size, original)
        sealed = observer.export_capture()
        manager._ensure_workspace_size(20)
        self.assertEqual(observer.export_capture(), sealed)
        self.assertEqual(manager.calls, 4)

    def test_unpinned_or_unapproved_fixture_manager_is_not_patched(self):
        manager = self.Manager()
        for version in ("main", "0.29.1.dev0+g98dff2a82"):
            with self.assertRaises(SimulationError):
                PinnedVllmWorkspaceObserver(manager, runtime_version=version, allow_test_fixture=True)
            self.assertNotIn("_ensure_workspace_size", vars(manager))
        with self.assertRaises(SimulationError):
            PinnedVllmWorkspaceObserver(manager, runtime_version="0.29.1.dev0+g98dff2a81")


if __name__ == "__main__":
    unittest.main()
