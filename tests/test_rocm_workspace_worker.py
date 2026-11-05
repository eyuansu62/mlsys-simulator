"""CPU-only protocol checks for raw worker capture; never GPU measurements."""

from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import weakref

from mlsys_sim import DeviceSpec, Experiment, InfraSpec, MemoryConfig, ModelSpec, StrategySpec


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "scripts/rocm_workspace_audit"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BaseWorker:
    def execute_model(self, scheduler_output):
        return scheduler_output


def worker_module():
    names = ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu_worker")
    modules = {name: types.ModuleType(name) for name in names}
    modules[names[-1]].Worker = BaseWorker
    with patch.dict(sys.modules, modules):
        return load("fixture_raw_workspace_worker", "worker_workspace_capture.py")


WORKER = worker_module()
LAUNCHER = load("fixture_workspace_stage_launcher", "workspace_stage_session.py")


class Storage:
    def data_ptr(self):
        return 4096

    def nbytes(self):
        return 16


class Tensor:
    shape = (2, 4)
    dtype = "torch.bfloat16"
    device = types.SimpleNamespace(type="cuda", index=1)

    def untyped_storage(self):
        return Storage()

    def data_ptr(self):
        return 4098

    def stride(self):
        return (4, 1)

    def storage_offset(self):
        return 1

    def element_size(self):
        return 2


def fake_torch():
    return types.SimpleNamespace(Tensor=Tensor)


def experiment(mode="single"):
    tp, pp = {"single": (1, 1), "tp2": (2, 1), "pp2": (1, 2)}[mode]
    return Experiment(infra=InfraSpec(devices=tuple(DeviceSpec("g" + str(i)) for i in range(tp * pp))),
        strategy=StrategySpec(model=ModelSpec(weight_dtype="bfloat16", kv_dtype="bfloat16",
            activation_dtype="bfloat16"), tensor_parallel_size=tp, pipeline_parallel_size=pp,
            scheduler="fifo", max_batch_size=1, max_batch_tokens=512, memory=MemoryConfig(kv_pool_bytes=268435456),
            pipeline_transport="vllm_native" if mode == "pp2" else "full_hidden",
            pipeline_transport_max_tokens=512 if mode == "pp2" else 0))


class MetadataTests(unittest.TestCase):
    def test_rccl_mappings_bind_exact_file_offsets_executable_range_and_deleted_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "librccl.so.1"
            path.write_bytes(b"CPU fixture library bytes, never loaded")
            text = (f"1000-2000 r--p 00000000 00:00 1 {path}\n"
                    f"2000-5000 r-xp 00001000 00:00 1 {path}\n"
                    f"5000-6000 rw-p 00004000 00:00 1 {path}\n"
                    "8000-9000 r-xp 00000000 00:00 2 /unrelated/module.so\n")
            rows = WORKER.mapped_library_segments(str(path), maps_text=text)
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["start"] - row["offset"] for row in rows}, {0x1000})
            self.assertEqual(rows[1], {"start": 0x2000, "end": 0x5000, "offset": 0x1000,
                                       "permissions": "r-xp", "path": str(path.resolve())})
            with self.assertRaisesRegex(ValueError, "deleted"):
                WORKER.mapped_library_segments(str(path), maps_text=text.replace(str(path), str(path) + " (deleted)"))
            with self.assertRaisesRegex(ValueError, "executable"):
                WORKER.mapped_library_segments(str(path), maps_text=text.replace("r-xp", "r--p"))

    def test_hardware_probe_receives_loaded_library_and_preserves_actual_source_and_errors(self):
        inventory = {"kind": "rocm_inventory", "status": "READY", "source": {"kind": "measured"},
                     "versions": {"driver_version": "actual-driver", "rocm_version": "actual-rocm",
                                  "rccl_version": "actual-rccl"},
                     "rccl_library": {"sha256": "a" * 64}}
        with patch("mlsys_sim.hardware.probe_rocm", return_value=inventory) as probe:
            self.assertIs(WORKER.observe_hardware_inventory({"path": "/actual/librccl.so.1"}), inventory)
            probe.assert_called_once_with(rccl_library="/actual/librccl.so.1")
        fixture = {"status": "UNAVAILABLE", "source": {"kind": "test_fixture"}, "issues": ["kept"]}
        with patch("mlsys_sim.hardware.probe_rocm", return_value=fixture):
            self.assertEqual(WORKER.observe_hardware_inventory({"path": "/fixture/library"}), fixture)
        with patch("mlsys_sim.hardware.probe_rocm", side_effect=RuntimeError("actual failure")):
            failure = WORKER.observe_hardware_inventory({"path": "/actual/library"})
        self.assertEqual(failure["status"], "ERROR")
        self.assertEqual(failure["error"], {"type": "RuntimeError", "message": "actual failure"})
        self.assertNotIn("source", failure)
        self.assertNotIn("versions", failure)

    def test_snapshot_retains_unknown_fields_and_complete_event_arrays_without_mutation(self):
        before = {"other": 7, "segments": [{"address": 42, "blocks": [{"frames": ["text"], "size": 4}]}],
                  "device_traces": [[{"action": "alloc", "frames": ["text"], "unknown": 9}], []]}
        original = deepcopy(before)
        value = WORKER.compact_snapshot(before)
        self.assertEqual(before, original)
        self.assertEqual(value, {"other": 7, "segments": [{"address": 42, "blocks": [{"size": 4}]}],
                                 "device_traces": [[{"action": "alloc", "unknown": 9}], []]})

    def test_storage_metadata_has_actual_view_capacity_and_never_retains_tensor(self):
        tensor = Tensor()
        reference = weakref.ref(tensor)
        counters = iter((12, 12))
        row = WORKER.tensor_descriptor(tensor, "actual", tensor_name="residual", cursor=lambda: next(counters))
        self.assertEqual(row["pointer"], 4096)
        self.assertEqual(row["bytes"], 16)
        self.assertEqual(row["tensor_data_ptr"], 4098)
        self.assertEqual(row["tensor_offset"], 1)
        self.assertEqual(row["tensor_name"], "residual")
        self.assertEqual(row["device_index"], 1)
        self.assertEqual(row["metadata_cursor"], 12)
        self.assertTrue(row["metadata_cursor_stable"])
        json.dumps(row)
        del tensor
        gc.collect()
        self.assertIsNone(reference())

    def test_changing_cursor_is_preserved_and_never_retried(self):
        cursors = iter((10, 11))
        row = WORKER.tensor_descriptor(Tensor(), "actual", cursor=lambda: next(cursors))
        self.assertEqual((row["metadata_cursor_before"], row["metadata_cursor_after"]), (10, 11))
        self.assertFalse(row["metadata_cursor_stable"])
        instance = object.__new__(WORKER.WorkspaceWorker)
        instance._ws_bridge = types.SimpleNamespace(mlsys_torch_probe_cursor=lambda: WORKER.UINT64_MAX)
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            instance._ws_cursor()

    def test_source_binding_uses_real_unwrapped_function_bytes(self):
        row = WORKER.source_binding(BaseWorker.execute_model)
        import inspect
        self.assertEqual(row["source_sha256"], hashlib.sha256(
            inspect.getsource(BaseWorker.execute_model).encode()).hexdigest())
        self.assertEqual(row["file_sha256"], hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def test_cpu_witness_does_not_evaluate_lazy_tensors(self):
        class Lazy:
            __module__ = "vllm.fixture"

            @property
            def tensors(self):
                raise AssertionError("Observer consumed a lazy receive")

        runner = types.SimpleNamespace(execute_model_state=None,
                                       input_batch=types.SimpleNamespace(prev_sampled_token_ids=None))
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            value = WORKER.cpu_return_witness(Lazy(), runner)
        self.assertEqual(value["gpu_tensor_count"], 0)
        self.assertTrue(value["execute_state_is_none"])


class ProtocolTests(unittest.TestCase):
    def test_process_exit_policy_is_only_one_pp2_pilot_and_default_policy_stays_available(self):
        self.assertEqual(LAUNCHER.validate_native_finalization("pp2", "pilot", 1, "process_exit"), "process_exit")
        for mode, cohort, repeats in (("single", "pilot", 1), ("tp2", "pilot", 1),
                                      ("pp2", "train", 1), ("pp2", "repeat", 1),
                                      ("pp2", "pilot", 2), ("pp2", "pilot", True)):
            with self.subTest(mode=mode, cohort=cohort, repeats=repeats), self.assertRaises(ValueError):
                LAUNCHER.validate_native_finalization(mode, cohort, repeats, "process_exit")
            self.assertEqual(LAUNCHER.validate_native_finalization(mode, cohort, repeats, "worker_shutdown"),
                             "worker_shutdown")
        with self.assertRaises(ValueError):
            LAUNCHER.validate_native_finalization("pp2", "pilot", 1, "force_success")

    def test_native_header_reader_observes_top_metadata_without_synthesizing_event_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native-42.json"
            data = {"pid": 42, "text": 'string with "events": [ and punctuation',
                    "nested": {"events": [7]}, "stop_evidence": {"origin": 2, "stop_result": 17},
                    "events": [{"seq": 1}], "write_complete": True}
            path.write_text(json.dumps(data))
            header = LAUNCHER.read_native_header(path)
            self.assertEqual(header["pid"], 42)
            self.assertEqual(header["nested"], {"events": [7]})
            self.assertEqual(header["stop_evidence"]["stop_result"], 17)
            self.assertNotIn("events", header)
            self.assertNotIn("write_complete", header)

    def test_process_exit_join_keeps_actual_sdk_codes_and_never_promotes_capture(self):
        for code in (0, 17):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                pending = root / "session-process-exit-pending.json"
                native = root / "native-42.json"
                actual = {"pid": 42, "stop_status": code, "finalize_callback_called": True,
                    "finalize_ns": 250, "stop_evidence": {"origin": 2, "stop_result": code},
                    "events": [], "write_complete": True}
                native.write_text(json.dumps(actual))
                pending.write_text(json.dumps({"process_id": 42, "native_final_file": str(native),
                    "capture_closed": False, "complete_workspace": False, "capture_errors": []}))
                ack = {"process_id": 42, "process_exit_payload_file": str(pending),
                       "session_file": str(root / "session-process-exit-diagnostic.json")}
                parent = {"original_returned": True, "started_monotonic_raw_ns": 100,
                    "ended_monotonic_raw_ns": 300,
                    "owned_worker_processes": [{"pid": 42, "alive": False, "exit_code": 0}]}
                joined = LAUNCHER.join_process_exit_diagnostic(ack, parent)
                self.assertFalse(joined["capture_closed"])
                self.assertFalse(joined["complete_workspace"])
                self.assertFalse(joined["hardware_validated"])
                self.assertIsNone(joined["native_sdk_stop_status"])
                self.assertIsNone(joined["native_sdk_finalize_status"])
                observed = joined["process_exit_diagnostic"]
                self.assertEqual(observed["native_observed_header"]["stop_evidence"], actual["stop_evidence"])
                self.assertEqual(observed["native_final_sha256"], hashlib.sha256(native.read_bytes()).hexdigest())
                self.assertEqual(observed["parent_natural_shutdown"], parent)

    def test_process_exit_missing_final_is_a_real_failure_despite_checkpoint_or_zero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "native-42.json"
            checkpoint = root / "native-checkpoint.json"
            checkpoint.write_text(json.dumps({"pid": 42, "finalize_callback_called": True, "events": []}))
            pending = root / "session-process-exit-pending.json"
            pending.write_text(json.dumps({"process_id": 42, "native_final_file": str(native)}))
            output = root / "session-process-exit-diagnostic.json"
            ack = {"process_id": 42, "process_exit_payload_file": str(pending), "session_file": str(output)}
            parent = {"original_returned": True,
                      "owned_worker_processes": [{"pid": 42, "alive": False, "exit_code": 0}]}
            with self.assertRaisesRegex(RuntimeError, "missing"):
                LAUNCHER.join_process_exit_diagnostic(ack, parent)
            failure = json.loads(output.read_text())
            self.assertFalse(failure["process_exit_diagnostic"]["native_final_file_observed"])
            self.assertFalse(failure["capture_closed"])
            output.unlink()
            parent["owned_worker_processes"][0]["exit_code"] = -9
            with self.assertRaisesRegex(RuntimeError, "successful exit"):
                LAUNCHER.join_process_exit_diagnostic(ack, parent)
            self.assertFalse(output.exists())

    def test_prewarm_allocates_outside_epoch_and_validates_actual_payload_after_scope(self):
        for mode, rank in (("tp2", 0), ("pp2", 0), ("pp2", 1)):
            with self.subTest(mode=mode, rank=rank), tempfile.TemporaryDirectory() as temporary:
                calls = []
                state = {"epoch": None, "receive_index": 0}
                references = []

                class ControlTensor(Tensor):
                    def __init__(self, shape, value):
                        self.shape = shape
                        self.values = [value] * (shape[0] if len(shape) == 1 else shape[0] * shape[1])
                        references.append(weakref.ref(self))

                    def untyped_storage(self):
                        return types.SimpleNamespace(data_ptr=lambda: id(self), nbytes=lambda: len(self.values) * 2)

                    def data_ptr(self):
                        return id(self)

                    def stride(self):
                        return (1,) if len(self.shape) == 1 else (self.shape[1], 1)

                    def storage_offset(self):
                        return 0

                    def cpu(self):
                        self_test.assertIsNone(state["epoch"], "Payload validation entered native epoch")
                        calls.append("CPU-copy")
                        return self

                    def reshape(self, *shape):
                        return self

                    def tolist(self):
                        return self.values

                self_test = self
                pg = types.SimpleNamespace(group_name="actual-fixture-pg")

                class Work:
                    def wait(self):
                        self_test.assertEqual(state["epoch"], 6502)
                        calls.append("wait")

                class Distributed:
                    def get_process_group_ranks(self, group):
                        self_test.assertIs(group, pg)
                        return [4, 9]

                    def get_rank(self, group):
                        return rank

                    def get_backend(self, group):
                        return "nccl"

                    def all_gather_into_tensor(self, output, input_, *, group, async_op):
                        self_test.assertEqual(state["epoch"], 6501)
                        self_test.assertFalse(async_op)
                        self_test.assertEqual(len(input_.values), 75968)
                        output.values = [1.0] * 75968 + [2.0] * 75968
                        calls.append("gather")

                    def isend(self, tensor, *, dst, group, tag):
                        self_test.assertEqual((state["epoch"], dst, tag), (6502, 9, 0))
                        calls.append("send")
                        return Work()

                    def irecv(self, tensor, *, src, group, tag):
                        self_test.assertEqual((state["epoch"], src, tag), (6502, 4, 0))
                        tensor.values = [17.0 + state["receive_index"] * 2] * (512 * 2048)
                        state["receive_index"] += 1
                        calls.append("receive")
                        return Work()

                def allocate(shape, value, **kwargs):
                    self.assertIsNone(state["epoch"], "GPU allocation/fill entered native epoch")
                    calls.append("allocate")
                    return ControlTensor(shape, value)

                fake = types.SimpleNamespace(bfloat16="fixture-bf16", distributed=Distributed(),
                    full=allocate, empty=lambda shape, **kwargs: allocate(shape, -1.0, **kwargs),
                    cuda=types.SimpleNamespace(synchronize=lambda device: calls.append(("sync", state["epoch"]))))
                local = types.SimpleNamespace(device_group=pg, world_size=2, rank_in_group=rank)
                singleton = types.SimpleNamespace(world_size=1)
                distributed_module = types.ModuleType("vllm.distributed")
                distributed_module.get_tp_group = lambda: local if mode == "tp2" else singleton
                distributed_module.get_pp_group = lambda: local if mode == "pp2" else singleton
                instance = object.__new__(WORKER.WorkspaceWorker)
                instance._ws_root = Path(temporary)
                instance._ws_native_active = None
                instance._ws_pg_prewarm = []
                instance.device = types.SimpleNamespace(index=1)

                def begin(epoch, label):
                    self.assertIsNone(state["epoch"])
                    state["epoch"] = epoch
                    calls.append("begin")

                def end(epoch):
                    self.assertEqual(state["epoch"], epoch)
                    state["epoch"] = None
                    calls.append("end")

                instance._ws_begin_native = begin
                instance._ws_end_native = end
                with patch.dict(sys.modules, {"torch": fake, "vllm.distributed": distributed_module}):
                    instance._ws_prewarm_process_groups()
                row = instance._ws_pg_prewarm[0]
                self.assertEqual(row["status"], "COMPLETED")
                self.assertTrue(row["payload_validation"]["success"])
                self.assertEqual(row["pg"]["global_ranks"], [4, 9])
                self.assertTrue(row["preallocated_before_epoch"])
                self.assertTrue(row["device_sync_completed"])
                self.assertLess(calls.index("end"), calls.index("CPU-copy"))
                self.assertIn(("sync", 6501 if mode == "tp2" else 6502), calls)
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references), "Control tensors leaked into model")

    def test_native_budget_is_explicit_bounded_and_never_silently_clamped(self):
        self.assertEqual(LAUNCHER.native_event_budget(1048576), 1048576)
        self.assertEqual(LAUNCHER.native_event_budget(2097152), 2097152)
        self.assertEqual(LAUNCHER.native_event_budget(4194304), 4194304)
        for value in (0, -1, 4194305, True, "2097152", 2097152.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LAUNCHER.native_event_budget(value)
        self.assertEqual(LAUNCHER.shutdown_grace(60), 60)
        for value in (0, 4, 121, True, 60.0):
            with self.subTest(grace=value), self.assertRaises(ValueError):
                LAUNCHER.shutdown_grace(value)

    def test_fixed_live_domains_and_explicit_unsupported_policies(self):
        for mode in ("single", "tp2", "pp2"):
            LAUNCHER.validate_plan_experiment(experiment(mode), mode)
        exp = experiment()
        for field in ("prefix_cache", "chunked_prefill", "mixed_batching"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                LAUNCHER.validate_plan_experiment(replace(exp, strategy=replace(exp.strategy, **{field: True})), "single")
        with self.assertRaises(ValueError):
            LAUNCHER.validate_plan_experiment(replace(exp, strategy=replace(exp.strategy,
                memory=replace(exp.strategy.memory, cpu_swap_bytes=1))), "single")
        with self.assertRaises(ValueError):
            LAUNCHER.validate_plan_experiment(exp, "tp2")

    def test_actual_output_count_and_lengths_are_required(self):
        output = types.SimpleNamespace(finished=True, request_id="actual", prompt_token_ids=[1] * 16,
                                       outputs=[types.SimpleNamespace(token_ids=[1, 2, 3, 4])])
        self.assertEqual(LAUNCHER.validate_outputs([output], [16])[0]["output_token_ids"], [1, 2, 3, 4])
        for values, lengths in (([], [16]), ([output], [32])):
            with self.assertRaises(RuntimeError):
                LAUNCHER.validate_outputs(values, lengths)
        output.finished = False
        with self.assertRaises(RuntimeError):
            LAUNCHER.validate_outputs([output], [16])

    def test_supervisor_module_import_does_not_import_torch_or_vllm(self):
        program = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('launcher',{str(TOOLS / 'workspace_stage_session.py')!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules and 'vllm' not in sys.modules")
        result = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_transport_calls_once_without_waits_and_names_follow_actual_dict(self):
        class Group:
            ranks = [0, 1]
            rank_in_group = 0
            world_size = 2
            unique_name = "fixture-pp"

            def __init__(self):
                self._pending_isends = []
                self.calls = []

            def isend_tensor_dict(self, tensor_dict):
                self.calls.append("isend")
                self._reap_completed_isends()
                handles = [object(), object(), object()]
                self._pending_isends.append((handles, list(tensor_dict.values())))
                return handles

            def irecv_tensor_dict(self):
                self.calls.append("irecv")
                return {"hidden_states": Tensor(), "residual": Tensor()}, [object(), object()], []

            def _reap_completed_isends(self):
                self.calls.append("reap")
                self._pending_isends.clear()

        instance = object.__new__(WORKER.WorkspaceWorker)
        instance._ws_pp = group = Group()
        instance._ws_transport_sources = {}
        instance._ws_attribute_patches = []
        instance._ws_pending_names = {}
        instance._ws_events = []
        instance._ws_event_counter = 0
        instance._ws_errors = []
        instance._ws_cohort = "formal"
        instance._ws_current = {"epoch_id": 10001}
        instance._ws_torch_active = 10001
        instance._ws_index = 1
        instance._ws_cursor = lambda: 10
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            instance._ws_install_transport_observers()
            handles = group.isend_tensor_dict({"hidden_states": Tensor(), "residual": Tensor()})
            received = group.irecv_tensor_dict()
            group._reap_completed_isends()
        self.assertEqual(group.calls, ["isend", "reap", "irecv", "reap"])
        self.assertEqual(len(handles), 3)
        self.assertEqual(set(received[0]), {"hidden_states", "residual"})
        self.assertFalse(instance._ws_errors)
        self.assertEqual(len(instance._ws_events), 4)
        self.assertEqual([event["event_id"] for event in instance._ws_events], [2, 1, 3, 4])
        sent = next(event for event in instance._ws_events if event["operation"] == "isend_tensor_dict")
        self.assertEqual(sent["status"], "RETURNED")
        self.assertEqual(sent["epoch_id"], 10001)
        self.assertEqual([row["tensor_name"] for row in sent["after_pending"][0]["tensor_descriptors"]],
                         ["hidden_states", "residual"])
        self.assertEqual(instance._ws_events[-1]["after_pending"], [])
        receipt = instance._ws_release_observer_references()
        self.assertTrue(all(row["restored"] and not row["had_own_attribute"] for row in receipt))
        self.assertNotIn("isend_tensor_dict", vars(group))
        self.assertNotIn("irecv_tensor_dict", vars(group))
        self.assertNotIn("_reap_completed_isends", vars(group))
        self.assertIsNone(instance._ws_pp)

    def test_observer_restore_keeps_exact_preexisting_instance_attribute(self):
        owner = types.SimpleNamespace()
        old = object()
        wrapper = object()
        owner.method = wrapper
        instance = object.__new__(WORKER.WorkspaceWorker)
        instance._ws_attribute_patches = [(owner, "method", old, wrapper)]
        instance._ws_pending_names = {}
        instance._ws_errors = []
        instance._ws_pp = object()
        instance._ws_hook = object()
        receipt = instance._ws_release_observer_references()
        self.assertIs(owner.method, old)
        self.assertTrue(receipt[0]["had_own_attribute"])
        self.assertTrue(receipt[0]["restored"])
        self.assertEqual(instance._ws_attribute_patches, [])
        self.assertIsNone(instance._ws_pp)
        self.assertIsNone(instance._ws_hook)

    def test_natural_shutdown_wrapper_preserves_original_return_exception_and_call_count(self):
        original_error = RuntimeError("actual Runtime failure")
        token = object()
        for raises in (False, True):
            with self.subTest(raises=raises):
                calls = []
                receipts = []
                errors = []

                class Actual:
                    _ws_deferred_payload = {"capture_closed": False}

                    def _ws_after_natural_shutdown(self, receipt):
                        calls.append("observe")
                        receipts.append(receipt)
                        if raises:
                            raise ValueError("observer failure must not mask Runtime failure")

                    def _ws_write_shutdown_observer_error(self, receipt, error):
                        errors.append(type(error).__name__)

                class Proc:
                    def __init__(self):
                        self.worker = types.SimpleNamespace(worker=Actual())

                    def shutdown(self):
                        calls.append("original")
                        if raises:
                            raise original_error
                        return token

                first = WORKER.install_workerproc_shutdown_observer(Proc)
                self.assertEqual(WORKER.install_workerproc_shutdown_observer(Proc), first)
                if raises:
                    with self.assertRaises(RuntimeError) as caught:
                        Proc().shutdown()
                    self.assertIs(caught.exception, original_error)
                    self.assertEqual(errors, ["ValueError"])
                else:
                    self.assertIs(Proc().shutdown(), token)
                self.assertEqual(calls, ["original", "observe"])
                self.assertEqual(receipts[0]["original_returned"], not raises)
                self.assertEqual(receipts[0]["source"], first)
                self.assertLessEqual(receipts[0]["started_monotonic_raw_ns"], receipts[0]["ended_monotonic_raw_ns"])

    def test_nonzero_real_stop_code_remains_failed_without_fabricated_finalize(self):
        with tempfile.TemporaryDirectory() as temporary:
            instance = object.__new__(WORKER.WorkspaceWorker)
            instance._ws_root = Path(temporary)
            instance._ws_closed_summary = None
            instance._ws_placement = {"stage_index": 0, "tp_rank": 0}
            instance._ws_records = []
            instance._ws_native = types.SimpleNamespace(mlsys_probe_stop=lambda: 17)
            result = instance._ws_finalize_cpu_payload({"capture_errors": [], "complete_workspace": False})
            self.assertFalse(result["capture_closed"])
            raw = json.loads((Path(temporary) / "session.json").read_text())
            self.assertEqual(raw["native_sdk_stop_status"], 17)
            self.assertIsNone(raw["native_sdk_finalize_status"])
            self.assertFalse(raw["capture_closed"])
            self.assertTrue(any("17" in value for value in raw["capture_errors"]))

    def test_stdthread_prime_is_cpu_only_and_binds_actual_header_and_runtime_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            header = root / "std_thread.h"
            module = root / "libstdc++.so.6"
            header.write_bytes(b"CPU fixture header")
            module.write_bytes(b"CPU fixture runtime bytes, never loaded")
            control = {"header_reference": str(header), "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
                       "module_path": str(module), "attempted": True, "completed": True}
            calls = []

            class Call:
                def __init__(self, operation):
                    self.operation = operation

                def __call__(self, *args):
                    calls.append(self.operation)
                    if self.operation == "write":
                        Path(args[0].decode()).write_text(json.dumps(control))
                    return 0

            native = types.SimpleNamespace(mlsys_probe_stdthread_control=Call("prime"),
                mlsys_probe_write_stdthread_control=Call("write"))
            instance = object.__new__(WORKER.WorkspaceWorker)
            instance._ws_root = root
            instance._ws_load_native = lambda: native
            mapping = [{"start": 4096, "end": 8192, "offset": 0,
                        "permissions": "r-xp", "path": str(module)}]
            with patch.object(WORKER, "mapped_library_segments", return_value=mapping) as mapped:
                instance._ws_prime_stdthread_control()
            self.assertEqual(calls, ["prime", "write"])
            mapped.assert_called_once_with(str(module.resolve()))
            bound = instance._ws_stdthread_control
            self.assertEqual(bound["control"], control)
            self.assertEqual(bound["header"]["sha256"], hashlib.sha256(header.read_bytes()).hexdigest())
            self.assertEqual(bound["runtime_library"]["sha256"], hashlib.sha256(module.read_bytes()).hexdigest())
            self.assertEqual(bound["runtime_library"]["mapped_segments"], mapping)

    def test_close_finalizes_once_after_all_hip_and_metadata_reads(self):
        self._exercise_close("single")

    def test_mp_close_defers_until_real_shutdown_and_uses_only_cached_cpu_payload(self):
        self._exercise_close("tp2")

    def test_process_exit_close_does_not_call_manual_stop_or_finalize(self):
        self._exercise_close("pp2", "process_exit")

    def _exercise_close(self, mode, policy="worker_shutdown"):
        events = []
        state = {"finalized": False}

        class Call:
            def __init__(self, name, value=0):
                self.name, self.value = name, value

            def __call__(self, *args):
                events.append(self.name)
                if self.name == "finalize":
                    state["finalized"] = True
                return self.value

        def hip(name, value=None):
            def invoke(*args):
                self.assertFalse(state["finalized"], name)
                events.append(name)
                return value
            return invoke

        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
            synchronize=hip("sync"), empty_cache=hip("empty_cache"),
            memory=types.SimpleNamespace(_snapshot=hip("snapshot", {"segments": [], "device_traces": [[]]}))))
        native = types.SimpleNamespace(mlsys_probe_begin=Call("native_begin"), mlsys_probe_end=Call("native_end"),
            mlsys_probe_write=Call("native_write"), mlsys_probe_stop=Call("stop"),
            mlsys_probe_finalize=Call("finalize"))
        bridge = types.SimpleNamespace(mlsys_torch_probe_begin=Call("torch_begin"),
            mlsys_torch_probe_end=Call("torch_end"), mlsys_torch_probe_write=Call("torch_write"))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"torch": fake}):
            instance = object.__new__(WORKER.WorkspaceWorker)
            instance._ws_root = Path(temporary)
            instance._ws_current = None
            instance._ws_closed_summary = None
            instance._ws_closed_payload = None
            instance._ws_deferred_payload = None
            instance._ws_pending_close_ack = None
            instance._ws_attribute_patches = []
            instance._ws_pending_names = {}
            instance._ws_plan = {"mode": mode}
            instance._ws_native_finalization = policy
            instance._ws_shutdown_source = {"qualified_name": "CPU.fixture.shutdown"}
            instance._ws_bridge = bridge
            instance._ws_native = native
            instance._ws_native_active = None
            instance._ws_install_thread = threading.get_native_id()
            instance._ws_pp = types.SimpleNamespace(world_size=1)
            instance._ws_hook = types.SimpleNamespace(flush=Call("hook_flush"))
            instance._ws_cursor = lambda: 1
            instance._ws_pending = lambda: []
            instance._ws_records = []
            instance._ws_errors = []
            instance._ws_placement = {"stage_index": 0, "tp_rank": 0}
            instance.device = 0
            instance._pp_send_work = []

            def export():
                self.assertFalse(state["finalized"])
                events.append("metadata")
                return {"complete_workspace": False, "hardware_validated": False}

            instance.mlsys_workspace_export = export
            result = instance.mlsys_workspace_close()
            self.assertEqual(instance.mlsys_workspace_close(), result)
            if mode != "single":
                self.assertTrue(result["finalize_deferred"])
                self.assertFalse(result["capture_closed"])
                self.assertFalse((Path(temporary) / "session.json").exists())
                self.assertFalse(state["finalized"])
                self.assertNotIn("stop", events)
                pending = json.loads((Path(temporary) / "session-pending.json").read_text())
                self.assertFalse(pending["capture_closed"])

                def forbidden_metadata():
                    raise AssertionError("Metadata was rebuilt after natural Runtime shutdown")

                instance.mlsys_workspace_export = forbidden_metadata
                instance._ws_after_natural_shutdown({"original_returned": True, "source": instance._ws_shutdown_source})
                if policy == "process_exit":
                    self.assertNotIn("stop", events)
                    self.assertNotIn("finalize", events)
                    self.assertFalse(state["finalized"])
                    self.assertFalse((Path(temporary) / "session.json").exists())
                    payload = json.loads((Path(temporary) / "session-process-exit-pending.json").read_text())
                    self.assertFalse(payload["capture_closed"])
                    self.assertIsNone(payload["native_sdk_stop_status"])
                    self.assertIsNone(payload["native_sdk_finalize_status"])
                    self.assertEqual(result["diagnostic_session_file"], result["session_file"])
                    return
            raw = json.loads((Path(temporary) / "session.json").read_text())
            self.assertEqual(raw["native_sdk_finalize_status"], 0)
            self.assertFalse(raw["complete_workspace"])
            self.assertLess(events.index("metadata"), events.index("stop"))
            self.assertEqual(events[-1], "finalize")
            self.assertEqual(events.count("finalize"), 1)


if __name__ == "__main__":
    unittest.main()
