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


if __name__ == "__main__":
    unittest.main()
