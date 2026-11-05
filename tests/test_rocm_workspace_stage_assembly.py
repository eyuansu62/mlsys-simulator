"""CPU orchestration tests; component auditors are mocked, never GPU evidence.

The measured-kind canonical objects below test promotion gates and serialization
only. Dummy source/DSO files are read as bytes and must never be executed.
"""

from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim.config import experiment_from_dict
from mlsys_sim.layout import rank_weight_bytes
from test_rocm_workspace_stage_domain import inputs as domain_inputs

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/rocm_workspace_audit"
_SPEC = importlib.util.spec_from_file_location("assemble_stage_workspace", _SCRIPTS / "assemble_stage_workspace.py")
assert _SPEC is not None and _SPEC.loader is not None
_ASSEMBLY = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_SCRIPTS))
try:
    _SPEC.loader.exec_module(_ASSEMBLY)
finally:
    sys.path.remove(str(_SCRIPTS))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as stream:
            json.dump(value, stream)
    else:
        path.write_text(json.dumps(value, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, mode="single", repeats=3, cohort="train"):
    base = root / "acquisition"
    base.mkdir()
    shared = root / "shared-sources"
    shared.mkdir()
    for name in ("worker_workspace_capture.py", "workspace_stage_session.py"):
        (shared / name).write_text('raise RuntimeError("CPU source fixture must never be executed")\n')
    for name in ("native_probe.so", "torch_allocator_probe.so"):
        (shared / name).write_bytes(b"CPU fixture, not an executable DSO\n")
    plan, workers, experiment = domain_inputs(mode, repeats)
    plan.update(out=str(base), cohort=cohort, bound_experiment=str(base / "bound-experiment.json"),
        worker_source=str(shared / "worker_workspace_capture.py"),
        worker_source_sha256=sha(shared / "worker_workspace_capture.py"),
        launcher_source_sha256=sha(shared / "workspace_stage_session.py"),
        native_probe_library=str(shared / "native_probe.so"),
        native_probe_library_sha256=sha(shared / "native_probe.so"),
        torch_probe_library=str(shared / "torch_allocator_probe.so"),
        torch_probe_library_sha256=sha(shared / "torch_allocator_probe.so"))
    write(base / "bound-experiment.json", experiment)
    plan["bound_experiment_sha256"] = sha(base / "bound-experiment.json")
    write(base / "session-plan.json", plan)
    references = []
    shutdown_source = shared / "multiproc_executor.py"
    shutdown_source.write_text("class WorkerProc:\n    def shutdown(self):\n        return None\n")
    for worker in workers:
        pid = worker["process_id"]
        folder = base / f"worker-pid-{pid}"
        folder.mkdir()
        worker.update(plan_sha256=sha(base / "session-plan.json"),
            startup_snapshot=str(folder / "startup.json.gz"), teardown_snapshot=str(folder / "teardown.json.gz"),
            torch_probe_file=str(folder / "torch-native-records.json"), native_final_file=str(base / f"native-{pid}.json"),
            history_max_entries=100000)
        worker["source_bindings"].update(worker_source_reference=plan["worker_source"],
            worker_source_sha256=plan["worker_source_sha256"])
        write(folder / "startup.json.gz", {"process_id": pid, "thread_id": pid + 5000, "snapshot": {}})
        write(folder / "teardown.json.gz", {"epoch_id": 9001, "before": {}, "after": {}})
        write(folder / "torch-native-records.json", {"install_thread_id": pid + 5000})
        write(base / f"native-{pid}.json", {"pid": pid, "status": "FORGED_PRODUCER_STATUS_IS_NOT_AN_AUDIT",
            "agent_inventory": {"agents": [{"handle": pid + 10000, "type_name": "GPU",
                "product_name": worker["observed_device"]["name"]}]}})
        reference = {"process_id": pid, "session_file": str(folder / "session.json")}
        if mode != "single":
            source = {"qualified_name": "vllm.v1.executor.multiproc_executor.WorkerProc.shutdown",
                "source_reference": str(shutdown_source), "file_sha256": sha(shutdown_source),
                "source_sha256": hashlib.sha256(b"    def shutdown(self):\n        return None\n").hexdigest()}
            receipt_value = {"process_id": pid, "thread_id": pid + 5000, "clock": "CLOCK_MONOTONIC",
                "source": source, "original_returned": True, "started_monotonic_ns": 200,
                "ended_monotonic_ns": 300, "return_type": "builtins.NoneType",
                "scope": "original_WorkerProc_shutdown_including_model_and_distributed_environment"}
            receipt_file = folder / "natural-shutdown-receipt.json"
            write(receipt_file, receipt_value)
            worker.update(native_finalize_deferred=True, natural_shutdown_source=source,
                lifecycle_clock="CLOCK_MONOTONIC", deferred_close_saved_monotonic_ns=100,
                runtime_shutdown=receipt_value, natural_shutdown_receipt_file=str(receipt_file),
                natural_shutdown_receipt_sha256=sha(receipt_file))
            reference.update(finalize_deferred=True, capture_closed=False)
        for index, row in enumerate(worker["records"]):
            row["thread_id"] = pid + 5000
            row["epoch_file"] = str(folder / f"epoch-{row['epoch_id']}.json.gz")
            frame = {key: deepcopy(value) for key, value in row.items() if key != "epoch_file"}
            frame.update(before={}, after={}, owners_before=[], owners_after=[])
            write(Path(row["epoch_file"]), frame)
        write(folder / "session.json", worker)
        references.append(reference)
    write(base / "session-status.json", {"status": "RAW_ACQUISITION_COMPLETED_AUDIT_REQUIRED",
        "session_id": plan["session_id"], "worker_exit_code": 0})
    summary = {"status": "RAW_ACQUISITION_COMPLETED_AUDIT_REQUIRED",
        "session_id": plan["session_id"], "ranks": references}
    if mode != "single":
        summary["natural_parent_shutdown"] = {"original_returned": True,
            "owned_worker_processes": [{"pid": w["process_id"], "exit_code": 0, "alive": False} for w in workers]}
    write(base / "launcher-summary.json", summary)
    receipt = root / "host-receipt.json"
    write(receipt, {"returncode": 0})
    return base, receipt, plan, workers, experiment


def native_ok(*args, **kwargs):
    return {"status": "NATIVE_BACKING_CHECKS_PASSED", "native_fixed_runtime_bytes": 100,
            "native_fixed_communication_bytes": 0, "hardware_validated": False}


def torch_ok(**kwargs):
    from dataclasses import asdict
    from mlsys_sim import CalibrationSpec, DeviceSpec, Experiment, InfraSpec, ModelSpec, StrategySpec
    layout = kwargs["placement"]
    model = ModelSpec(**kwargs["model"])
    exp = Experiment(strategy=StrategySpec(model=model, tensor_parallel_size=layout["tensor_parallel_size"],
        pipeline_parallel_size=layout["pipeline_parallel_size"]), calibration=CalibrationSpec(),
        infra=InfraSpec(devices=tuple(DeviceSpec(str(i)) for i in range(
            layout["tensor_parallel_size"] * layout["pipeline_parallel_size"]))))
    # The values are CPU fixture arithmetic, not actual fixed storage.
    categories = {"weights": rank_weight_bytes(exp, layout["stage_index"]),
                  "kv_pool": 268435456, "runtime": 256}
    if layout["pipeline_parallel_size"] > 1 and layout["stage_index"]:
        categories["pp_buffers"] = 4194304
    assert asdict(model) == kwargs["model"]
    return {"status": "CLOSED_CANDIDATE", "requested_storage_peak": 0 if layout["tp_rank"] == 0 else 37,
            "fixed_requested_bytes_by_category": categories, "pp_transport_peak_bytes": 1234,
            "complete_workspace": False}


class StageAssemblyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def assemble(self, base, receipt, *, native=native_ok, torch=torch_ok, transport=None, out="assembled"):
        if transport is None:
            transport = {"status": "TRANSPORT_GENERATIONS_PROVEN"}
        with patch.object(_ASSEMBLY, "audit_native_backing", side_effect=native) as n, \
             patch.object(_ASSEMBLY, "audit_torch_workspace", side_effect=torch) as t, \
             patch.object(_ASSEMBLY, "audit_workspace_transport", return_value=transport):
            report = _ASSEMBLY.assemble_stage_workspace(base, receipt, self.root / out)
            return report, n, t

    def test_full_single_cohort_promotes_only_after_independent_audits_and_keeps_real_zero(self):
        base, receipt, plan, workers, exp = fixture(self.root)
        report, native, torch = self.assemble(base, receipt)
        self.assertTrue(report["attribution_complete"])
        self.assertFalse(report["hardware_validated"])
        self.assertEqual(report["canonical_record_count"], 24)
        self.assertEqual(native.call_count, 1)
        self.assertEqual(torch.call_count, 24)
        self.assertTrue(all(call.kwargs["placement"]["device_index"] == 0 for call in torch.call_args_list))
        data = json.loads((self.root / "assembled/dataset.json").read_text())
        self.assertEqual(sum(row["phase"] == "decode" for row in data["records"]), 18)
        self.assertTrue(all(row["workspace_bytes_per_rank"] == [{"tp_rank": 0, "peak_bytes": 0}]
                            for row in data["records"]))
        self.assertEqual(data["notes"]["acquisition"]["session_id"], plan["session_id"])
        hashes = data["notes"]["acquisition"]["capture_input_sha256"]
        self.assertNotIn(sha(base / "bound-experiment.json"), hashes)
        self.assertNotIn(sha(receipt), hashes)
        self.assertNotIn(plan["worker_source_sha256"], hashes)
        self.assertIn(str((base / "bound-experiment.json").resolve()), report["raw_input_sha256"])
        self.assertIn(plan["worker_source"], report["source_and_binary_sha256"])
        self.assertEqual(report["fixed_budget_maxima_by_rank"][0]["proved_budget_epochs"], 24)

    def test_producer_success_strings_never_replace_native_or_torch_replay(self):
        for name, native, torch in (
            ("native", lambda *a, **k: {"status": "PARTIAL", "native_fixed_runtime_bytes": None}, torch_ok),
            ("torch", native_ok, lambda **k: {"status": "PARTIAL", "requested_storage_peak": None}),
        ):
            location = self.root / name
            location.mkdir()
            base, receipt, _, _, _ = fixture(location)
            with patch.object(_ASSEMBLY, "audit_native_backing", side_effect=native), \
                 patch.object(_ASSEMBLY, "audit_torch_workspace", side_effect=torch):
                report = _ASSEMBLY.assemble_stage_workspace(base, receipt, location / "out")
            self.assertEqual(report["status"], "PARTIAL")
            self.assertFalse((location / "out/dataset.json").exists())
            self.assertTrue((location / "out/rank-0/native-audit.json").exists())
            self.assertTrue((location / "out/rank-0/torch-audit-10009.json").exists())

    def test_pilot_or_short_cohort_remains_diagnostic_even_when_mocked_components_pass(self):
        base, receipt, _, _, _ = fixture(self.root, repeats=1, cohort="pilot")
        report, _, _ = self.assemble(base, receipt)
        self.assertEqual(report["status"], "PARTIAL")
        self.assertFalse((self.root / "assembled/dataset.json").exists())
        self.assertIn("DIAGNOSTIC_COHORT_NOT_ELIGIBLE_FOR_CALIBRATION", {row["code"] for row in report["issues"]})
        self.assertEqual(len(report["component_epoch_results"]), 8)

    def test_multicard_missing_native_comm_binding_stays_partial_without_max_rank_collapse(self):
        base, receipt, _, _, _ = fixture(self.root, mode="tp2")
        report, _, torch = self.assemble(base, receipt)
        self.assertFalse(report["attribution_complete"])
        self.assertFalse((self.root / "assembled/dataset.json").exists())
        self.assertEqual(torch.call_count, 48)
        self.assertEqual({row["global_rank"] for row in report["component_epoch_results"]}, {0, 1})
        peaks = {row["global_rank"]: row["component_candidate_peak_bytes"] for row in report["component_epoch_results"]}
        self.assertEqual(peaks, {0: 0, 1: 37})
        self.assertEqual(sum(row["code"] == "COMMUNICATION_BINDING_MISSING" for row in report["issues"]), 2)

    def test_host_envelope_is_forwarded_unchanged_and_never_overrides_independent_native_partial(self):
        base, receipt, plan, workers, _ = fixture(self.root, mode="pp2")
        library = self.root / "shared-sources/librccl.so"
        library.write_bytes(b"CPU-only fixture, never loaded")
        bindings = {}
        for worker in workers:
            worker["loaded_rccl"].update(path=str(library), library_sha256=sha(library))
            worker["stage_capture"]["metadata"]["rccl_binding"] = deepcopy(worker["loaded_rccl"])
            worker["observed_hardware_inventory"]["rccl_library"]["sha256"] = sha(library)
            worker["host_signal_evidence"] = {"kind": "CPU forwarding fixture", "status": "FORGED_PASS",
                "process_id": worker["process_id"], "original_integer_id": 37}
            write(Path(worker["startup_snapshot"]).with_name("session.json"), worker)
            bindings[str(worker["process_id"])] = {"driver_source_reference": plan["worker_source"],
                "driver_source_sha256": plan["worker_source_sha256"], "loaded_rccl": worker["loaded_rccl"],
                "rank": {**worker["placement"], "agent_handle": worker["process_id"] + 10000},
                "initialization_epochs": []}
        path = self.root / "communication.json"
        write(path, bindings)
        with patch.object(_ASSEMBLY, "_host_signal_source_gate") as file_gate, \
             patch.object(_ASSEMBLY, "audit_native_backing", return_value={"status": "PARTIAL"}) as native, \
             patch.object(_ASSEMBLY, "audit_torch_workspace", side_effect=torch_ok), \
             patch.object(_ASSEMBLY, "audit_workspace_transport", return_value={"status": "TRANSPORT_GENERATIONS_PROVEN"}):
            result = _ASSEMBLY.assemble_stage_workspace(base, receipt, self.root / "out", communication_bindings=path)
        self.assertEqual(file_gate.call_count, 2)
        self.assertEqual(native.call_count, 2)
        for observed, worker in zip(native.call_args_list, workers):
            self.assertEqual(observed.kwargs["communication_binding"]["host_signal_evidence"], worker["host_signal_evidence"])
        self.assertFalse(result["attribution_complete"])
        self.assertFalse((self.root / "out/dataset.json").exists())

    def test_multicard_final_native_window_requires_actual_natural_shutdown_receipts(self):
        for broken in ("missing", "changed", "failed", "pending", "alive"):
            with self.subTest(broken=broken):
                location = self.root / broken
                location.mkdir()
                base, receipt, plan, workers, _ = fixture(location, mode="tp2")
                worker = workers[0]
                session_file = Path(worker["startup_snapshot"]).with_name("session.json")
                summary = json.loads((base / "launcher-summary.json").read_text())
                if broken == "missing":
                    worker.pop("natural_shutdown_receipt_file")
                elif broken == "changed":
                    Path(worker["natural_shutdown_receipt_file"]).write_text("{}")
                elif broken == "failed":
                    worker["runtime_shutdown"]["original_returned"] = False
                elif broken == "pending":
                    summary["ranks"][0]["capture_closed"] = True
                else:
                    summary["natural_parent_shutdown"]["owned_worker_processes"][0]["alive"] = True
                write(session_file, worker)
                write(base / "launcher-summary.json", summary)
                with patch.object(_ASSEMBLY, "audit_native_backing", side_effect=native_ok) as native:
                    result = _ASSEMBLY.assemble_stage_workspace(base, receipt, location / "out")
                self.assertFalse(result["attribution_complete"])
                self.assertEqual(native.call_count, 0)
                self.assertFalse((location / "out/dataset.json").exists())

    def test_missing_worker_rank_is_rejected_by_actual_domain_before_any_promoting_audit(self):
        base, receipt, _, _, _ = fixture(self.root, mode="tp2")
        summary = json.loads((base / "launcher-summary.json").read_text())
        summary["ranks"].pop()
        write(base / "launcher-summary.json", summary)
        report, native, torch = self.assemble(base, receipt)
        self.assertEqual(report["status"], "PARTIAL")
        self.assertEqual(native.call_count, 0)
        self.assertEqual(torch.call_count, 0)
        self.assertFalse((self.root / "assembled/dataset.json").exists())

    def test_pp_transport_context_must_be_independently_proven_and_peak_not_defaulted_to_zero(self):
        base, receipt, _, _, _ = fixture(self.root, mode="pp2")
        report, _, torch = self.assemble(base, receipt, transport={"status": "PARTIAL"})
        self.assertFalse(report["attribution_complete"])
        self.assertIn("PP_TRANSPORT_AUDIT_PARTIAL", {row["code"] for row in report["issues"]})
        self.assertTrue(all(call.kwargs["transport_context"] is not None for call in torch.call_args_list))
        self.assertTrue(all(len(call.kwargs["transport_context"]["epochs"]) == 32 for call in torch.call_args_list))
        self.assertFalse((self.root / "assembled/dataset.json").exists())
        exp = experiment_from_dict(json.loads((base / "bound-experiment.json").read_text()))
        placement = {"stage_index": 0}
        result = {"fixed_requested_bytes_by_category": {"weights": rank_weight_bytes(exp, 0),
            "runtime": 83886080, "kv_pool": 268435456}}
        with self.assertRaises(ValueError):
            _ASSEMBLY._rank_budget(exp, placement, result, native_ok(), {"pipeline_buffer_reserved_bytes": 8388608})

    def test_weight_padding_uses_runtime_spare_and_unknown_native_never_becomes_zero(self):
        base, receipt, _, _, raw_exp = fixture(self.root)
        exp = experiment_from_dict(raw_exp)
        exp = replace(exp, strategy=replace(exp.strategy, memory=replace(exp.strategy.memory, runtime_bytes=100)))
        tensor = {"fixed_requested_bytes_by_category": {"weights": rank_weight_bytes(exp, 0) + 64,
                  "runtime": 90, "kv_pool": 268435456}}
        native = {"status": "NATIVE_BACKING_CHECKS_PASSED", "native_fixed_runtime_bytes": 10}
        result = _ASSEMBLY._rank_budget(exp, {"stage_index": 0}, tensor, native,
                                       {"pipeline_buffer_reserved_bytes": 0})
        self.assertEqual(result["fixed_padding_excess_bytes"], 64)
        self.assertEqual(result["runtime_component_bound_bytes"], 164)
        self.assertFalse(result["all_fixed_budgets_admitted"])
        self.assertEqual(result["native_fixed_communication_bytes"], 0)
        native["native_fixed_runtime_bytes"] = None
        with self.assertRaises(ValueError):
            _ASSEMBLY._rank_budget(exp, {"stage_index": 0}, tensor, native,
                                   {"pipeline_buffer_reserved_bytes": 0})

    def test_nonzero_parent_return_and_changed_executed_source_cannot_emit_dataset(self):
        for label in ("returncode", "source"):
            place = self.root / label
            place.mkdir()
            base, receipt, plan, _, _ = fixture(place)
            if label == "returncode":
                write(receipt, {"returncode": 1})
            else:
                Path(plan["worker_source"]).write_text("different executed source")
            report = _ASSEMBLY.assemble_stage_workspace(base, receipt, place / "out")
            self.assertEqual(report["status"], "PARTIAL")
            self.assertFalse((place / "out/dataset.json").exists())

    def test_session_bound_parent_receipt_cannot_be_reused_for_another_plan(self):
        base, receipt, plan, _, _ = fixture(self.root)
        write(receipt, {"returncode": 0, "session_id": plan["session_id"],
            "plan_sha256": "f" * 64, "session": str(base)})
        report, native, _ = self.assemble(base, receipt)
        self.assertFalse(report["attribution_complete"])
        self.assertEqual(native.call_count, 0)
        self.assertFalse((self.root / "assembled/dataset.json").exists())

    def test_raw_frame_mismatch_and_path_escape_keep_partial(self):
        base, receipt, _, workers, _ = fixture(self.root)
        frame = Path(workers[0]["records"][8]["epoch_file"])
        with gzip.open(frame, "rt") as stream:
            value = json.load(stream)
        value["phase"] = "decode"
        write(frame, value)
        report, _, _ = self.assemble(base, receipt)
        self.assertEqual(report["status"], "PARTIAL")
        self.assertIn("epoch bytes", report["issues"][-1]["message"])
        inputs = _ASSEMBLY.Inputs(base)
        with self.assertRaises(ValueError):
            inputs.path(str(self.root / "outside.json"))

    def test_duplicate_json_keys_and_existing_output_cannot_be_overwritten(self):
        base, receipt, _, _, _ = fixture(self.root)
        (base / "session-status.json").write_text('{"status": "first", "status": "second"}')
        report = _ASSEMBLY.assemble_stage_workspace(base, receipt, self.root / "out")
        self.assertEqual(report["status"], "PARTIAL")
        original = (self.root / "out/attribution-report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            _ASSEMBLY.assemble_stage_workspace(base, receipt, self.root / "out")
        self.assertEqual((self.root / "out/attribution-report.json").read_bytes(), original)


class CommunicationSourceGateTests(unittest.TestCase):
    """Source authentication is independent from the synthetic native replay."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / "distributed_c10d.py"
        source.write_text("def all_gather_into_tensor(output, input):\n    return output\n")
        binding = {"qualified_name": "torch.distributed.distributed_c10d.all_gather_into_tensor",
            "source_reference": str(source), "source_sha256": sha(source), "file_sha256": sha(source)}
        self.scope = {"kind": "torch_process_group_prewarm", "epoch_id": 6501,
            "epoch_label": "actual prewarm", "entrypoint": binding["qualified_name"],
            "source_binding": binding, "status": "COMPLETED", "comm_only": True}
        library = self.root / "librccl.so"
        library.write_bytes(b"not a DSO: only read as CPU fixture bytes")
        self.plan = {"worker_source": str(self.root / "worker_workspace_capture.py"),
            "worker_source_sha256": "a" * 64}
        loaded = {"path": str(library), "library_sha256": sha(library)}
        self.worker = {"placement": {"process_id": 12}, "loaded_rccl": loaded,
            "communicator_constructors": [], "communication_prewarm": [self.scope]}
        self.communication = {"driver_source_reference": self.plan["worker_source"],
            "driver_source_sha256": self.plan["worker_source_sha256"], "loaded_rccl": loaded,
            "rank": {"process_id": 12, "agent_handle": 7}, "initialization_epochs": [{
                "kind": "torch_process_group_prewarm", "epoch_id": 6501,
                "epoch_label": "actual prewarm", "entrypoint": binding["qualified_name"],
                "comm_only": True, "communicator_handle": 98, "world_size": 2, "rank": 0,
                "pg_scope": deepcopy(self.scope)}]}

    def gate(self):
        inputs = _ASSEMBLY.Inputs(self.root)
        _ASSEMBLY._communication_source_gate(self.communication, self.worker, self.plan, 7, inputs)
        return inputs

    def test_actual_pg_prewarm_has_its_own_source_gate(self):
        inputs = self.gate()
        self.assertIn(self.scope["source_binding"]["source_reference"], inputs.shared)

    def test_pg_scope_cannot_replace_raw_observation_or_actual_function_source(self):
        for change in ("scope", "source", "epoch"):
            with self.subTest(change=change):
                old = deepcopy(self.communication)
                if change == "scope":
                    self.communication["initialization_epochs"][0]["pg_scope"]["comm_only"] = False
                elif change == "source":
                    Path(self.scope["source_binding"]["source_reference"]).write_text("def other():\n    pass\n")
                else:
                    self.communication["initialization_epochs"][0]["epoch_id"] = 6502
                with self.assertRaises(ValueError):
                    self.gate()
                self.communication = old
                Path(self.scope["source_binding"]["source_reference"]).write_text(
                    "def all_gather_into_tensor(output, input):\n    return output\n")

    def test_natural_shutdown_binding_cannot_substitute_its_own_receipt(self):
        self.worker["runtime_shutdown"] = {"process_id": 12, "original_returned": True}
        self.communication["natural_shutdown_evidence"] = deepcopy(self.worker["runtime_shutdown"])
        self.gate()
        self.communication["natural_shutdown_evidence"]["process_id"] = 13
        with self.assertRaisesRegex(ValueError, "shutdown evidence differs"):
            self.gate()

    def test_stdthread_control_checks_raw_control_header_and_loaded_runtime_bytes(self):
        folder = self.root / "worker-pid-12"
        folder.mkdir()
        self.worker["startup_snapshot"] = str(folder / "startup.json.gz")
        header, runtime = self.root / "std_thread.h", self.root / "libstdc++.so"
        header.write_text("CPU header fixture")
        runtime.write_bytes(b"CPU runtime fixture")
        control = {"pid": 12, "header_reference": str(header), "header_sha256": sha(header),
            "module_path": str(runtime)}
        proof = {"control": control, "header": {"path": str(header), "sha256": sha(header)},
            "runtime_library": {"path": str(runtime), "sha256": sha(runtime)}}
        write(folder / "stdthread-control.json", control)
        self.worker["stdthread_control"] = deepcopy(proof)
        self.communication["stdthread_control"] = deepcopy(proof)
        inputs = self.gate()
        self.assertIn(str((folder / "stdthread-control.json").resolve()), inputs.capture)
        runtime.write_bytes(b"changed after capture")
        with self.assertRaises(ValueError):
            self.gate()


class HostSignalSourceGateTests(unittest.TestCase):
    """Only file/source admission is exercised; no callback ownership is mocked as real."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shared, session = self.root / "shared", self.root / "session"
        shared.mkdir()
        session.mkdir()
        worker_file = shared / "worker_workspace_capture.py"
        worker_file.write_text("class WorkspaceWorker:\n"
            "    def _ws_capture_host_signals(self):\n        return None\n"
            "    def mlsys_workspace_close(self):\n        return None\n"
            "    def _ws_capture_final_kfd_prefix(self):\n        return None\n")
        helper = shared / "hsa_signal_backing_capture.py"
        helper.write_text("def capture_hsa_signal_backings():\n    return {}\n")
        cpp = shared / "hsa_signal_backing_probe.cpp"
        cpp.write_text("// CPU fixture source; never compiled\n")
        compiled = shared / "actual-compiled-source.cpp"
        compiled.write_bytes(cpp.read_bytes())
        provider, runtime = shared / "provider.so", shared / "runtime.so"
        provider.write_bytes(b"CPU fixture provider, not executable")
        runtime.write_bytes(b"CPU fixture runtime, not executable")
        alias = shared / "runtime-loaded-alias.so"
        alias.symlink_to(runtime)
        headers = []
        for name in ("amd_hsa_signal.h", "hsa.h", "hsa_ext_amd.h", "amd_hsa_common.h", "amd_hsa_queue.h"):
            path = shared / name
            path.write_text("// CPU fixture header " + name)
            headers.append(self.artifact(path))
        self.plan = {"mode": "pp2", "native_finalization": "worker_shutdown", "session_id": "cpu-source-gate",
            "worker_source": str(worker_file), "worker_source_sha256": sha(worker_file),
            "host_signal_provider": {"provider_path": str(provider), "provider_sha256": sha(provider),
                "rocr_path": str(runtime), "rocr_sha256": sha(runtime),
                "abi_header_path": str(shared / "amd_hsa_signal.h"),
                "abi_header_sha256": sha(shared / "amd_hsa_signal.h")}}
        source = {"worker_source_reference": str(worker_file), "worker_source_sha256": sha(worker_file)}
        self.worker = {"process_id": 12, "logical_device_index": 1, "source_bindings": source,
            "startup_snapshot": str(session / "startup.json.gz"),
            "runtime_shutdown": {"thread_id": 19},
            "records": [{"cohort": "formal", "native_epoch_id": 10001}]}
        scope = {"kind": "post_formal_pre_natural_shutdown_public_signal_queries", "session_id": self.plan["session_id"],
            "process_id": 12, "thread_id": 17, "target_device_index": 1, "native_clock": "CLOCK_MONOTONIC_RAW",
            "worker_source": source, "formal_native_epochs": [10001], "query_started_monotonic_raw_ns": 300,
            "source": self.method_source(worker_file, "_ws_capture_host_signals")}
        fence = {"method": "existing_post_cohort_last_target_device_synchronize", "process_id": 12,
            "thread_id": 17, "target_device_index": 1, "native_clock": "CLOCK_MONOTONIC_RAW",
            "native_epoch_id": 9001, "native_cursor_before": 10, "native_cursor_after": 12,
            "started_monotonic_raw_ns": 100, "ended_monotonic_raw_ns": 200,
            "source": self.method_source(worker_file, "mlsys_workspace_close")}
        capture = {"schema_version": 1, "kind": "public_hsa_signal_backing_capture",
            "process_id": 12, "thread_id": 17, "status": "CAPTURED", "issues": [],
            "complete_workspace": False, "hardware_validated": False,
            "provider": self.artifact(provider), "sources": [self.artifact(helper), self.artifact(cpp)],
            "compiler_source": self.artifact(compiled), "abi_headers": headers,
            "runtime_dsos": [{**self.artifact(runtime), "loaded_path": str(alias)}],
            "initialization": {"compiled_source_path": str(compiled), "api_bindings": {
                "hsa_amd_pointer_info": {"symbol_name": "hsa_amd_pointer_info", "dladdr_success": True,
                    "module_path": str(alias), "address": 8193, "module_base": 8192}}},
            "source_binding": {"qualified_name": "hsa_signal_backing_capture.capture_hsa_signal_backings",
                "source_sha256": sha(helper)}, "source_scope": scope, "source_fence": fence,
            "capture_started_monotonic_raw_ns": 310, "capture_ended_monotonic_raw_ns": 390}
        for phase, cursor in (("before", 13), ("after", 18)):
            data = {"schema_version": 1, "kind": "observed_live_hsa_signal_generations", "pid": 12,
                "event_cursor": cursor, "generations": []}
            path = session / ("live-" + phase + ".json")
            write(path, data)
            capture[phase + "_live"] = data
            capture[phase + "_live_artifact"] = self.artifact(path)
        capture_file = session / "capture.json"
        write(capture_file, capture)
        evidence = {"schema_version": 1, "kind": "public_hsa_signal_backing_worker_evidence",
            "process_id": 12, "thread_id": 17, "status": "CAPTURE_RETURNED_REQUIRES_INDEPENDENT_NATIVE_AUDIT",
            "source_scope": scope, "source_fence": fence, "native_cursor_before": 12, "native_cursor_after": 19,
            "query_started_monotonic_raw_ns": 300, "query_ended_monotonic_raw_ns": 400,
            "capture": capture, "capture_artifact": {"path": str(capture_file), "sha256": sha(capture_file),
                "bytes": capture_file.stat().st_size}, "complete_workspace": False, "hardware_validated": False}
        write(session / "host-signal-worker-evidence.json", evidence)
        observed = {"schema_version": 1, "kind": "observed_live_hsa_signal_generations", "pid": 12,
            "event_cursor": 20, "generations": []}
        post_file = session / "host-signals-after-natural-shutdown.json"
        write(post_file, observed)
        evidence["after_natural_shutdown_live"] = {"kind": "post_natural_shutdown_observed_signal_generations",
            "process_id": 12, "thread_id": 19, "path": str(post_file), "sha256": sha(post_file),
            "bytes": post_file.stat().st_size, "observed_live": observed, "new_signal_queries_performed": False}
        self.worker["host_signal_evidence"] = evidence
        self.inputs = _ASSEMBLY.Inputs(session)

    @staticmethod
    def artifact(path):
        return {"path": str(path.resolve()), "sha256": sha(path), "size_bytes": path.stat().st_size}

    @staticmethod
    def method_source(path, name):
        return {"source_reference": str(path), "file_sha256": sha(path),
            "qualified_name": "worker_workspace_capture.WorkspaceWorker." + name,
            "source_sha256": hashlib.sha256(("    def " + name + "(self):\n        return None\n").encode()).hexdigest()}

    def gate(self):
        _ASSEMBLY._host_signal_source_gate(self.plan, self.worker, self.inputs, inference_thread=17)

    def refresh_metadata_artifacts(self):
        evidence = self.worker["host_signal_evidence"]
        path = Path(evidence["capture_artifact"]["path"])
        write(path, evidence["capture"])
        evidence["capture_artifact"].update(sha256=sha(path), bytes=path.stat().st_size)
        write(self.root / "session/host-signal-worker-evidence.json",
              {key: value for key, value in evidence.items()
               if key not in {"after_natural_shutdown_live", "kfd_event_evidence"}})

    def add_kfd(self):
        """Only source/artifact fixture data; these are not valid hardware tapes."""
        shared, session, cold = self.root / "shared", self.root / "session", self.root / "cold-control"
        cold.mkdir()
        artifacts = {}
        for name, filename in {"probe_source": "kfd_event_probe.cpp", "native_probe_source": "native_probe.cpp",
            "probe_library": "combined.so", "libc": "libc.so", "kfd_header": "kfd_ioctl.h", "drm_header": "drm.h"}.items():
            path = shared / filename
            path.write_text("CPU-only fixture " + name)
            artifacts[name] = self.artifact(path)
        artifacts["hsa_runtime"] = self.artifact(shared / "runtime.so")
        library = shared / "combined.so"
        self.plan.update(native_probe_library=str(library), native_probe_library_sha256=sha(library))
        build = cold / "build.json"
        write(build, {"returncode": 0, "binary_sha256": sha(library), "sources_sha256": {
            Path(artifacts[name]["path"]).name: artifacts[name]["sha256"] for name in ("probe_source", "native_probe_source")}})
        abi = {"fixture": "not hardware ABI"}
        binding = {name + "_sha256": value["sha256"] for name, value in artifacts.items()}
        binding.update(artifacts=artifacts, abi=abi, build_receipt_artifact=self.artifact(build))
        system = session / "system"
        system.mkdir()
        snapshots = []
        files = {"/sys/class/kfd/kfd/topology/nodes/1/gpu_id": b"77\n",
            "/sys/class/kfd/kfd/topology/nodes/1/properties": b"drm_render_minor 128\n",
            "/sys/class/drm/renderD128/dev": b"226:128\n",
            "/sys/class/drm/renderD128/device/uevent": b"PCI_SLOT_NAME=0000:01:00.0\n",
            "/sys/module/amdgpu/notes/.note.gnu.build-id": struct.pack("<III", 4, 20, 3) + b"GNU\0" + bytes.fromhex("a" * 40)}
        for index, (name, data) in enumerate(files.items()):
            path = system / (str(index) + ".json")
            doc = {"schema_version": 1, "kind": "actual_system_file_bytes", "original_path": name,
                "contents_hex": data.hex(), "contents_size_bytes": len(data), "contents_sha256": hashlib.sha256(data).hexdigest()}
            write(path, doc)
            snapshots.append({**{key: doc[key] for key in ("original_path", "contents_size_bytes", "contents_sha256")},
                "snapshot_artifact": self.artifact(path)})
        uname = system / "uname.json"
        write(uname, {"schema_version": 1, "kind": "actual_uname_result", "release": "CPU-fixture"})
        snapshots.append({"source": "os.uname", "snapshot_artifact": self.artifact(uname)})
        binding.update(kernel_release="CPU-fixture", amdgpu_module_build_id="a" * 40,
            device_nodes=[{"gpu_id": 77, "rdev_major": 226, "rdev_minor": 128, "pci_bus_id": "0000:01:00.0"}],
            system_observation_artifacts=snapshots)

        def report(pid):
            return {"pid": pid, "abi": abi, "compiled_source_path": artifacts["probe_source"]["path"],
                "libc_bindings": {"ioctl": {"address": 8193, "module_base": 8192,
                    "dladdr_success": True, "module_path": artifacts["libc"]["path"]}, "unused": {"address": None}}}

        cold_native, cold_kfd, control = cold / "native.json", cold / "kfd.json", cold / "control.jsonl"
        write(cold_native, {"pid": 99, "fixture": "not a Native SDK tape"})
        write(cold_kfd, report(99))
        control.write_text('{"original_control_line":1}\n{"original_control_line":2}\n')
        manifest = {"schema_version": 1, "kind": "cold_hsa_kfd_namespace_control_manifest",
            "source_binding": binding, "native_report": self.artifact(cold_native),
            "kfd_report": self.artifact(cold_kfd), "control_jsonl": self.artifact(control),
            "build_receipt": self.artifact(build), "create_cid": 137, "event_query_cid": 139,
            "kfd_create_event_call_id": 3, "target_epochs": [8001]}
        manifest_file = cold / "manifest.json"
        write(manifest_file, manifest)
        self.plan["host_signal_provider"].update(namespace_control_path=str(manifest_file),
            namespace_control_sha256=sha(manifest_file))
        kernel = {"schema_version": 1, "kind": "passive_kfd_event_evidence", "source_binding": binding,
            "namespace_control": manifest, "namespace_control_artifact": self.artifact(manifest_file)}
        for key in ("report_artifact", "pre_query_report_artifact", "post_query_report_artifact"):
            path = session / (key + ".json")
            write(path, report(12))
            kernel[key] = self.artifact(path)
        self.worker["host_signal_evidence"]["capture"]["kfd_event_evidence"] = deepcopy(kernel)
        self.refresh_metadata_artifacts()
        self.worker["native_sdk_stop_status"] = 0
        final = session / "kfd-final.json"
        write(final, report(12))
        kernel["final_report_artifact"] = self.artifact(final)
        kernel["final_snapshot_observation"] = {"process_id": 12, "thread_id": 19,
            "native_stop_returncode": 0, "write_returncode": 0, "prefix_only": True,
            "method": "passive_KFD_registry_prefix_after_natural_shutdown_and_successful_SDK_stop",
            "source": self.method_source(Path(self.plan["worker_source"]), "_ws_capture_final_kfd_prefix"),
            "complete_workspace": False, "hardware_validated": False}
        self.worker["host_signal_evidence"]["kfd_event_evidence"] = kernel

    def test_original_artifacts_sources_and_same_byte_runtime_aliases_are_verified_without_rewriting(self):
        before = deepcopy(self.worker)
        self.gate()
        self.assertEqual(before, self.worker)
        self.assertIn(str((self.root / "session/capture.json").resolve()), self.inputs.capture)
        self.assertIn(str(self.root / "shared/runtime-loaded-alias.so"), self.inputs.shared)

    def test_changed_capture_registry_or_size_is_rejected(self):
        for what in ("capture", "before", "after_shutdown", "size"):
            with self.subTest(what=what):
                self.setUp()
                evidence = self.worker["host_signal_evidence"]
                if what == "size":
                    evidence["capture_artifact"]["bytes"] += 1
                else:
                    artifact = (evidence["capture_artifact"] if what == "capture" else
                        evidence["capture"]["before_live_artifact"] if what == "before" else
                        evidence["after_natural_shutdown_live"])
                    Path(artifact["path"]).write_text("{}")
                with self.assertRaises(ValueError):
                    self.gate()

    def test_scope_fence_function_and_compiled_source_cannot_be_substituted(self):
        for what in ("session", "device", "body", "compiled_source", "runtime"):
            with self.subTest(what=what):
                self.setUp()
                evidence = self.worker["host_signal_evidence"]
                if what == "session":
                    evidence["source_scope"]["session_id"] = "other-session"
                elif what == "device":
                    evidence["source_fence"]["target_device_index"] = 0
                elif what == "body":
                    evidence["source_fence"]["source"]["source_sha256"] = "f" * 64
                elif what == "compiled_source":
                    Path(evidence["capture"]["compiler_source"]["path"]).write_text("different CPP source")
                else:
                    (self.root / "shared/runtime.so").write_bytes(b"different actual library")
                if what in {"session", "device", "body"}:
                    self.refresh_metadata_artifacts()
                with self.assertRaises(ValueError):
                    self.gate()

    def test_unconfigured_or_missing_evidence_is_rejected_and_legacy_absence_is_unchanged(self):
        evidence = self.worker.pop("host_signal_evidence")
        with self.assertRaises(ValueError):
            self.gate()
        self.plan.pop("host_signal_provider")
        self.gate()
        self.worker["host_signal_evidence"] = evidence
        with self.assertRaises(ValueError):
            self.gate()

    def test_no_live_targets_need_no_provider_initialization_but_cannot_hide_queried_records(self):
        capture = self.worker["host_signal_evidence"]["capture"]
        capture.pop("initialization")
        capture.pop("compiler_source")
        capture.update(status="NO_LIVE_FORMAL_TARGETS", records=[], ordinary_signal_control=None)
        self.refresh_metadata_artifacts()
        self.gate()  # Independent Native replay must still establish the actual empty target set.
        capture["records"] = [{"signal_handle": 37}]
        self.refresh_metadata_artifacts()
        with self.assertRaises(ValueError):
            self.gate()

    def test_capture_function_body_and_complete_public_headers_are_not_status_claims(self):
        for what in ("body", "header", "module"):
            with self.subTest(what=what):
                self.setUp()
                capture = self.worker["host_signal_evidence"]["capture"]
                if what == "body":
                    capture["source_binding"]["source_sha256"] = "f" * 64
                elif what == "header":
                    capture["abi_headers"].pop()
                else:
                    capture["initialization"]["api_bindings"]["hsa_amd_pointer_info"]["module_path"] = "/unbound/module.so"
                self.refresh_metadata_artifacts()
                with self.assertRaises(ValueError):
                    self.gate()

    def test_raw_capture_cannot_escape_its_session_and_registry_c_never_queries_a_handle(self):
        evidence = self.worker["host_signal_evidence"]
        evidence["after_natural_shutdown_live"]["new_signal_queries_performed"] = True
        with self.assertRaises(ValueError):
            self.gate()
        evidence["after_natural_shutdown_live"]["new_signal_queries_performed"] = False
        other = self.root / "shared/capture.json"
        write(other, evidence["capture"])
        evidence["capture_artifact"].update(path=str(other), sha256=sha(other), bytes=other.stat().st_size)
        write(self.root / "session/host-signal-worker-evidence.json",
              {key: value for key, value in evidence.items() if key != "after_natural_shutdown_live"})
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.gate()

    def test_kfd_original_files_and_common_cold_control_are_separate_from_fresh_capture_hashes(self):
        self.add_kfd()
        before = deepcopy(self.worker)
        self.gate()
        self.assertEqual(before, self.worker)
        kernel = self.worker["host_signal_evidence"]["kfd_event_evidence"]
        self.assertIn(kernel["final_report_artifact"]["path"], self.inputs.capture)
        for key in ("native_report", "kfd_report", "control_jsonl", "build_receipt"):
            path = kernel["namespace_control"][key]["path"]
            self.assertIn(path, self.inputs.raw)
            self.assertNotIn(path, self.inputs.capture)
        for row in kernel["source_binding"]["system_observation_artifacts"]:
            self.assertNotIn(row["snapshot_artifact"]["path"], self.inputs.capture)

    def test_na_final_kfd_diagnostic_does_not_invent_a_query_source_or_namespace(self):
        self.add_kfd()
        evidence = self.worker["host_signal_evidence"]
        evidence["capture"].pop("kfd_event_evidence")
        evidence["capture"].update(status="NO_LIVE_FORMAL_TARGETS", records=[], ordinary_signal_control=None)
        evidence["kfd_event_evidence"] = {key: value for key, value in evidence["kfd_event_evidence"].items()
            if key in {"final_report_artifact", "final_snapshot_observation"}}
        self.refresh_metadata_artifacts()
        self.gate()  # Source admission only; no hardware-event backing is claimed.

    def test_kfd_tape_source_and_build_bytes_cannot_be_backfilled_or_replaced(self):
        for what in ("worker_report", "cold_report", "source", "cpp_path", "libc", "build"):
            with self.subTest(what=what):
                self.setUp()
                self.add_kfd()
                kernel = self.worker["host_signal_evidence"]["kfd_event_evidence"]
                if what == "worker_report":
                    Path(kernel["final_report_artifact"]["path"]).write_text("{}")
                elif what == "cold_report":
                    Path(kernel["namespace_control"]["native_report"]["path"]).write_text("{}")
                elif what == "source":
                    Path(kernel["source_binding"]["artifacts"]["native_probe_source"]["path"]).write_text("different source")
                elif what == "cpp_path":
                    path = Path(kernel["final_report_artifact"]["path"])
                    value = json.loads(path.read_text())
                    value["compiled_source_path"] = kernel["source_binding"]["artifacts"]["native_probe_source"]["path"]
                    write(path, value)
                    kernel["final_report_artifact"] = self.artifact(path)
                elif what == "libc":
                    Path(kernel["source_binding"]["artifacts"]["libc"]["path"]).write_text("different libc bytes")
                else:
                    Path(kernel["source_binding"]["build_receipt_artifact"]["path"]).write_text("{}")
                with self.assertRaises(ValueError):
                    self.gate()


if __name__ == "__main__":
    unittest.main()
