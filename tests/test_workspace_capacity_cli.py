"""Workspace/Capacity CLI artifacts remain explicit, immutable and scoped."""

from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import json
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

from mlsys_sim import calibrate_dataset, experiment_from_dict, profile_identity, replica_experiment
from mlsys_sim.pp_communication import calibrate_pp_communication
from mlsys_sim.pp_transport import native_pp_descriptor
from mlsys_sim.workspace import ATTRIBUTION, calibrate_workspace
from test_pp_communication_cli import dataset as communication_dataset, experiment, SOURCE
import test_rocm_steps as rocm_fixtures


ROOT = Path(__file__).resolve().parents[1]


def workspace_dataset(exp, *, identifier="workspace-cli", peak=100):
    records = []
    for stage in range(exp.strategy.pipeline_parallel_size):
        for phase, query, context in (("prefill", 4, 0), ("decode", 1, 4)):
            for repeat in range(2):
                records.append({"record_id": f"s{stage}-{phase}-r{repeat}", "group_id": f"s{stage}-{phase}",
                    "phase": phase, "stage_index": stage, "query_tokens": [query], "context_lengths": [context],
                    "workspace_bytes_per_rank": [{"tp_rank": rank,
                        "peak_bytes": 0 if rank == 0 else peak + stage + repeat}
                        for rank in range(exp.strategy.tensor_parallel_size)]})
    return {"schema_version": 1, "kind": "workspace_measurements", "dataset_id": identifier,
        "identity": profile_identity(exp), "source": SOURCE, "attribution": deepcopy(ATTRIBUTION), "records": records}


def stage_profile(exp):
    records = []
    for stage in range(exp.strategy.pipeline_parallel_size):
        for phase, query, context, duration in (("prefill", 4, 0, 2.5), ("decode", 1, 4, .5)):
            records.append({"record_id": f"s{stage}-{phase}", "group_id": phase, "phase": phase,
                "stage_index": stage, "query_tokens": [query], "context_lengths": [context], "duration_ms": duration,
                "workspace_peak_bytes": 1024 * 1024,
                "pp_transport": {"outgoing" if stage == 0 else "incoming": native_pp_descriptor(exp, query)}})
    return calibrate_dataset({"schema_version": 1, "dataset_id": "stage-ws-cli", "identity": profile_identity(exp),
        "source": SOURCE, "records": records}, profile_version="stage-cli", allow_test_fixture=True)


def capacity_dataset():
    from mlsys_sim.capacity import capacity_experiment, config_sha256, validate_capacity_dataset

    base = experiment_from_dict(asdict(experiment()))
    trials = []
    for point, tokens in enumerate((4, 8)):
        effective = capacity_experiment(base, "input_tokens", tokens)
        digest = config_sha256(effective)
        for repeat in range(2):
            worker = f"worker-{point}-{repeat}"
            oom = tokens == 8
            devices = [{"index": index, "name": device.name, "memory_bytes": device.memory_bytes}
                       for index, device in enumerate(effective.infra.devices)]
            allocator = [{"device_index": row["index"], "allocated_bytes": 100, "reserved_bytes": 200,
                "free_bytes": row["memory_bytes"] - 200, "total_bytes": row["memory_bytes"],
                "peak_allocated_bytes": 150, "peak_reserved_bytes": 200} for row in devices]
            observed = {"schema_version": 1, "kind": "memory_capacity_trial_capture",
                "protocol": "fresh_owned_runtime_capacity_v1", "session_id": "capacity-cli-fixture", "worker_id": worker,
                "point_index": point, "repeat": repeat, "axis": {"name": "input_tokens", "value": tokens},
                "effective_config_sha256": digest, "source": SOURCE,
                "status": "OOM" if oom else "COMPLETED", "failure_phase": "run" if oom else None,
                "exception_class": "torch.cuda.OutOfMemoryError" if oom else None, "typed_oom": oom,
                "stack": {"torch_version": "fixture-torch", "hip_version": "fixture-hip"}, "devices": devices,
                "metadata": {"isolation": "owned_process", "effective_config_sha256": digest,
                    "applied_axis": {"name": "input_tokens", "value": tokens},
                    "execution_scope": "complete_declared_workload",
                    "runtime_name": effective.calibration.runtime_name, "runtime_version": effective.calibration.runtime_version},
                "allocator": {"before": allocator, "after": allocator, "errors": []},
                "error_code": "OOM" if oom else None, "message": "CPU typed-OOM fixture" if oom else None,
                "result": None if oom else {"status": "COMPLETED",
                    "completed_request_ids": [request.request_id for request in effective.workload]},
                "cleanup_status": "COMPLETED", "elapsed_ms": 1.0, "kv_pool_bindings": [], "isolation": "owned_process",
                "oom_allocator_evidence": {"phase": "run" if oom else None,
                    "per_device": [{"device_index": row["index"], "before": 0,
                        "after": 1 if row["index"] == 0 else 0} for row in devices] if oom else []},
                "capacity_oom_evidence": None}
            trials.append({"point_index": point, "repeat": repeat, "worker_id": worker,
                "axis_name": "input_tokens", "axis_value": tokens, "effective_experiment": asdict(effective),
                "effective_config_sha256": digest, "observation": observed, "prediction": {}})
    return validate_capacity_dataset({"schema_version": 1, "kind": "memory_capacity_measurements",
        "protocol": "fresh_owned_runtime_capacity_v1", "dataset_id": "capacity-cli-fixture", "source": SOURCE,
        "session_id": "capacity-cli-fixture", "base_experiment": asdict(base),
        "axis": {"name": "input_tokens", "values": [4, 8]}, "repeats": 2,
        "profile_inputs": [], "trials": trials}, allow_test_fixture=True)


class WorkspaceCapacityCLITests(unittest.TestCase):
    def test_shipped_capacity_fixture_tracks_expanded_configuration_defaults(self):
        from mlsys_sim import load_experiment
        from mlsys_sim.capacity import collect_memory_capacity

        root = ROOT / "examples/v1_9"
        exp = load_experiment(root / "capacity-base.fixture.json")
        plan = json.loads((root / "capacity.plan.fixture.json").read_text())
        collected = collect_memory_capacity(exp, plan, base_dir=root, allow_test_fixture=True)
        self.assertEqual(collected["status"], "COMPLETED")
        self.assertEqual(collected["source"]["kind"], "test_fixture")
        self.assertFalse(collected["hardware_validated"])
        self.assertEqual(len(collected["dataset"]["trials"]), 4)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def cli(self, target, *arguments, expected=0):
        output = self.root / target
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", *map(str, arguments), "--out", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        result = json.loads((output / "result.json").read_text()) if (output / "result.json").exists() else None
        return output, result

    def configured(self, exp=None):
        exp = exp or experiment()
        paths = []
        for replica in range(exp.strategy.replicas):
            local = replica_experiment(exp, replica)
            profile = calibrate_workspace(workspace_dataset(local, identifier=f"ws-r{replica}"),
                profile_version="workspace-cli-v1", allow_test_fixture=True)
            paths.append(str(self.write(f"source/workspace-{replica}.json", profile)))
        calibration = replace(exp.calibration, allow_test_fixture=True, workspace_profile_version="workspace-cli-v1")
        calibration = (replace(calibration, workspace_profile_path=paths[0]) if exp.strategy.replicas == 1 else
            replace(calibration, replica_workspace_profile_paths=tuple(paths)))
        return replace(exp, calibration=calibration), paths

    def test_explicit_workspace_factory_collects_complete_fixture_ledger(self):
        from mlsys_sim.cli import main
        from mlsys_sim.workspace_capture import collect_workspace

        exp, inventory, torch, cuda, log = rocm_fixtures.RocmStepTests().environment(tp=2, pp=2)
        plan = rocm_fixtures.RocmStepTests().plan(iterations=1, warmup=0)
        config_path = self.write("collect-experiment.json", asdict(exp))
        inventory_path = self.write("inventory.json", inventory)
        plan_path = self.write("workspace-plan.json", plan)

        class Adapter:
            recorder = None
            def metadata(self):
                return {"runtime_name": "fixture-runtime", "runtime_version": "fixture-runtime-v1",
                    "timing_boundary": "step_includes_output_and_sampling_v1", "includes_local_collectives": True,
                    "workspace_attribution": "runtime_ledger", "complete_workspace_attribution": True,
                    "includes_non_torch": True, "excluded_categories": deepcopy(ATTRIBUTION["excluded_categories"])}
            def prepare(self, sample):
                cuda.after = False
            def begin_workspace_capture(self, sample, recorder):
                self.recorder = recorder
            def run_step(self, sample):
                cuda.after = True
                self.recorder.allocate(1, "temporary", "exclusive-scratch", 37)
            def end_workspace_capture(self, sample, recorder):
                recorder.request_free(1, "temporary")
                recorder.release_completed(1, "temporary")
                recorder.complete_rank(0)
                recorder.complete_rank(1)
            def close(self):
                pass

        module = ModuleType("workspace_cli_fixture")
        adapter_file = self.root / "workspace_cli_fixture.py"
        adapter_file.write_text("# CPU Runtime adapter fixture; injected backend only\n")
        module.__file__ = str(adapter_file)
        module.factory = lambda unused: Adapter()
        def injected(local_exp, local_plan, adapter, local_inventory, **kwargs):
            return collect_workspace(local_exp, local_plan, adapter, local_inventory, torch_module=torch,
                clock_ns=rocm_fixtures.FakeClock([1_000_000, 2_000_000], log), **kwargs)
        output = self.root / "workspace-collected"
        with patch.dict(sys.modules, {module.__name__: module}), patch(
                "mlsys_sim.workspace_capture.collect_workspace", side_effect=injected), redirect_stdout(StringIO()):
            code = main(["collect", "--kind", "workspace", "--experiment", str(config_path), "--plan", str(plan_path),
                "--adapter", f"{module.__name__}:factory", "--inventory", str(inventory_path), "--allow-test-fixture",
                "--out", str(output)])
        self.assertEqual(code, 0, (output / "result.json").read_text())
        collected = json.loads((output / "dataset.json").read_text())
        self.assertEqual(collected["source"]["kind"], "test_fixture")
        self.assertEqual(collected["records"][0]["workspace_bytes_per_rank"],
                         [{"tp_rank": 0, "peak_bytes": 0}, {"tp_rank": 1, "peak_bytes": 37}])
        self.assertFalse(json.loads((output / "result.json").read_text())["hardware_validated"])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertIn(hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
                      {item["sha256"] for item in manifest["input_files"]})
        self.cli("collected-fit", "calibrate", "--workspace", "--dataset", output / "dataset.json",
                 "--profile-version", "collected-ws-v1", "--allow-test-fixture")

    def test_workspace_calibration_parent_and_simulation_prediction_use_exact_snapshots(self):
        exp = experiment()
        data = self.write("workspace.json", workspace_dataset(exp))
        fitted, result = self.cli("fit", "calibrate", "--workspace", "--dataset", data,
            "--profile-version", "workspace-cli-v1", "--allow-test-fixture")
        self.assertEqual(result["profile_kind"], "workspace_profile")
        profile = json.loads((fitted / "profile.json").read_text())
        self.assertEqual(profile["samples"][0]["workspace_bytes_per_rank"][0]["peak_bytes"], 0)
        self.assertTrue(any(row["workspace_bytes_per_rank"][1]["peak_bytes"] == 101 for row in profile["samples"]))
        parent, result = self.cli("parent", "calibrate", "--workspace", "--dataset", data,
            "--parent-profile", fitted / "profile.json", "--profile-version", "workspace-cli-v2", "--allow-test-fixture")
        self.assertEqual(result["parent_profile_version"], "workspace-cli-v1")
        exp = replace(exp, calibration=replace(exp.calibration, allow_test_fixture=True,
            workspace_profile_path=str(parent / "profile.json"), workspace_profile_version="workspace-cli-v2"))
        config = self.write("selected.json", asdict(exp))
        for mode in ("simulate", "predict"):
            output, result = self.cli(mode, mode, "--experiment", config)
            self.assertFalse(result["hardware_validated"])
            self.assertEqual((output / "workspace-profiles/replica-0.json").read_bytes(),
                             (parent / "profile.json").read_bytes())
            for row in result["memory"]:
                self.assertEqual(row["workspace_profile_provenance"]["snapshot_file"],
                                 "workspace-profiles/replica-0.json")
            if mode == "simulate":
                self.assertEqual(result["counts"]["completed"], 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["workspace_profiles"][0]["sha256"],
                             hashlib.sha256((parent / "profile.json").read_bytes()).hexdigest())

    def test_triple_profiles_are_independent_and_each_content_hash_is_in_fingerprint(self):
        exp, workspace_paths = self.configured()
        stage = self.write("source/stage.json", stage_profile(exp))
        comm = self.write("source/communication.json", calibrate_pp_communication(
            communication_dataset(exp), profile_version="comm-cli-v1", allow_test_fixture=True))
        exp = replace(exp, calibration=replace(exp.calibration, backend="profile", profile_path=str(stage),
            profile_version="stage-cli", native_pp_communication_profile_path=str(comm),
            native_pp_communication_profile_version="comm-cli-v1"))
        config = self.write("triple.json", asdict(exp))
        output, result = self.cli("triple", "simulate", "--experiment", config)
        self.assertEqual(result["counts"]["completed"], 1)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertTrue((output / "profile.json").exists())
        self.assertTrue((output / "communication-profiles/replica-0.json").exists())
        self.assertTrue((output / "workspace-profiles/replica-0.json").exists())
        previous = manifest["config_sha256"]
        for index, path in enumerate((stage, comm, Path(workspace_paths[0]))):
            path.write_bytes(path.read_bytes() + b"\n")
            changed, _ = self.cli(f"triple-changed-{index}", "simulate", "--experiment", config)
            current = json.loads((changed / "manifest.json").read_text())["config_sha256"]
            self.assertNotEqual(previous, current)
            previous = current
        self.assertEqual((output / "workspace-profiles/replica-0.json").read_bytes(),
                         Path(workspace_paths[0]).read_bytes().rstrip(b"\n"))

    def test_per_replica_workspace_source_and_snapshot_cannot_be_swapped(self):
        exp, paths = self.configured(experiment(replicas=2))
        config = self.write("cluster.json", asdict(exp))
        output, result = self.cli("cluster", "simulate", "--experiment", config)
        self.assertEqual(result["counts"]["completed"], 2)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["workspace_profiles"]), 2)
        for row in result["memory"]:
            self.assertEqual(row["workspace_profile_provenance"]["snapshot_file"],
                             f"workspace-profiles/replica-{row['replica_index']}.json")
        bad = replace(exp, calibration=replace(exp.calibration, replica_workspace_profile_paths=tuple(reversed(paths))))
        _, failed = self.cli("cluster-swapped", "predict", "--experiment", self.write("swapped.json", asdict(bad)), expected=2)
        self.assertEqual(failed["error_code"], "CALIBRATION_MISMATCH")

    def test_workspace_fixture_identity_domain_and_cli_kind_guards(self):
        exp, paths = self.configured()
        denied = replace(exp, calibration=replace(exp.calibration, allow_test_fixture=False))
        _, result = self.cli("fixture-denied", "predict", "--experiment", self.write("denied.json", asdict(denied)), expected=2)
        self.assertEqual(result["error_code"], "TEST_FIXTURE_NOT_ALLOWED")
        mismatch = replace(exp, calibration=replace(exp.calibration, workspace_profile_version="other-version"))
        _, result = self.cli("version-denied", "predict", "--experiment", self.write("mismatch.json", asdict(mismatch)), expected=2)
        self.assertEqual(result["error_code"], "CALIBRATION_MISMATCH")
        missing = replace(exp, workload=(replace(exp.workload[0], input_tokens=8),))
        output, result = self.cli("domain-denied", "predict", "--experiment", self.write("missing.json", asdict(missing)), expected=2)
        self.assertEqual(result["error_code"], "OUT_OF_WORKSPACE_CALIBRATION_DOMAIN")
        self.assertTrue((output / "workspace-profiles/replica-0.json").exists())
        dataset = self.write("measurements.json", workspace_dataset(experiment()))
        self.cli("mixed-kinds", "calibrate", "--workspace", "--native-pp-communication", "--dataset", dataset,
                 "--profile-version", "bad", "--allow-test-fixture", expected=2)
        _, result = self.cli("missing-profile", "validate", "--dataset", dataset, "--allow-test-fixture", expected=2)
        self.assertEqual(result["error_code"], "INVALID_ARGUMENT")
        self.assertEqual(json.loads(Path(paths[0]).read_text())["source"]["kind"], "test_fixture")

    def test_workspace_path_reference_and_content_change_keep_source_bytes(self):
        exp, paths = self.configured()
        calibration = asdict(exp.calibration)
        calibration["workspace_profile_path"] = "../source/workspace-0.json"
        section = self.write("settings/calibration.json", calibration)
        config = asdict(exp)
        config["calibration"] = "settings/calibration.json"
        source = self.write("reference.json", config)
        first, _ = self.cli("reference", "predict", "--experiment", source)
        before = json.loads((first / "manifest.json").read_text())
        self.assertIn(str(section.resolve()), {item["path"] for item in before["input_files"]})
        initial = Path(paths[0]).read_bytes()
        Path(paths[0]).write_bytes(initial + b"\n")
        second, _ = self.cli("reference-changed", "predict", "--experiment", source)
        after = json.loads((second / "manifest.json").read_text())
        self.assertNotEqual(before["config_sha256"], after["config_sha256"])
        self.assertEqual((first / "workspace-profiles/replica-0.json").read_bytes(), initial)

    def test_unselected_workspace_keeps_v1_8_communication_fingerprint(self):
        exp = experiment()
        comm = self.write("comm-only.json", calibrate_pp_communication(communication_dataset(exp),
            profile_version="comm-only", allow_test_fixture=True))
        selected = replace(exp, calibration=replace(exp.calibration, allow_test_fixture=True,
            native_pp_communication_profile_path=str(comm), native_pp_communication_profile_version="comm-only"))
        output, _ = self.cli("comm-only", "predict", "--experiment", self.write("comm-config.json", asdict(selected)))
        manifest = json.loads((output / "manifest.json").read_text())
        previous = deepcopy(manifest["effective_config"])
        for field in ("workspace_profile_path", "replica_workspace_profile_paths", "workspace_profile_version"):
            previous["calibration"].pop(field)
        recipe = {"experiment": previous, "native_pp_communication_profile_sha256": [hashlib.sha256(comm.read_bytes()).hexdigest()]}
        expected = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False,
                                             separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(manifest["config_sha256"], expected)
        self.assertNotIn("workspace_profiles", manifest)

    def test_capacity_raw_import_validates_without_profile_and_preserves_observed_bracket(self):
        data = capacity_dataset()
        raw = self.write("capacity-input/raw.json", data)
        config = self.write("capacity-experiment.json", data["base_experiment"])
        plan = self.write("capacity-input/plan.json", {"schema_version": 1, "dataset_id": "capacity-cli-fixture",
            "axis": data["axis"], "repeats": data["repeats"], "raw_path": "raw.json", "source": SOURCE})
        collected, result = self.cli("capacity-collected", "collect", "--kind", "memory-capacity",
            "--experiment", config, "--plan", plan, "--allow-test-fixture")
        self.assertFalse(result["hardware_validated"])
        self.assertTrue((collected / "capacity_report.json").exists())
        manifest = json.loads((collected / "manifest.json").read_text())
        self.assertIn(hashlib.sha256(raw.read_bytes()).hexdigest(),
                      {entry["sha256"] for entry in manifest["input_files"]})
        validated, result = self.cli("capacity-validated", "validate", "--memory-capacity", "--dataset",
            collected / "dataset.json", "--allow-test-fixture")
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(result["threshold_status"], "BRACKETED")
        self.assertEqual(result["summary"]["typed_oom_count"], 2)
        self.assertEqual(result["summary"]["completed_count"], 2)
        self.assertIsNone(result["exact_threshold"])
        self.assertEqual(result["observed_brackets"], [{"success_value": 4, "oom_value": 8,
                                                       "resolution": 4, "exact_threshold": None}])
        self.assertEqual(json.loads((validated / "capacity_report.json").read_text()), result)
        _, failed = self.cli("capacity-fixture-denied", "validate", "--memory-capacity", "--dataset", raw, expected=2)
        self.assertEqual(failed["error_code"], "TEST_FIXTURE_NOT_ALLOWED")

    def test_capacity_untyped_oom_and_profile_options_do_not_establish_thresholds(self):
        data = capacity_dataset()
        data["trials"][-1]["observation"].update(typed_oom=False, exception_class="RuntimeError",
                                                   message="out of memory text only")
        source = self.write("capacity-untyped.json", data)
        _, report = self.cli("capacity-censored", "validate", "--memory-capacity", "--dataset", source,
            "--allow-test-fixture")
        self.assertEqual(report["threshold_status"], "CENSORED")
        self.assertEqual(report["observed_brackets"], [])
        self.assertEqual(report["summary"]["typed_oom_count"], 1)
        self.assertIsNone(report["exact_threshold"])
        _, failed = self.cli("capacity-profile-option", "validate", "--memory-capacity", "--dataset", source,
            "--profile", source, "--allow-test-fixture", expected=2)
        self.assertEqual(failed["error_code"], "INVALID_ARGUMENT")
        self.cli("capacity-conflicting-kind", "validate", "--memory-capacity", "--native-pp-communication",
            "--dataset", source, "--allow-test-fixture", expected=2)


if __name__ == "__main__":
    unittest.main()
