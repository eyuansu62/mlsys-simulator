"""Native communication CLI provenance, strict selection and workflow regressions."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, ModelSpec,
                       RequestSpec, StrategySpec, calibrate_dataset, replica_experiment)
from mlsys_sim.pp_benchmark import (benchmark_cases, normalize_pp_benchmark_plan,
                                  sample_descriptor, worker_launch_contract)
from mlsys_sim.pp_communication import calibrate_pp_communication
from mlsys_sim.pp_transport import native_pp_descriptor
from mlsys_sim.profiles import profile_identity


ROOT = Path(__file__).resolve().parents[1]
SOURCE = {"kind": "test_fixture", "reference": "CLI protocol fixture; no GPU or communication measurement"}
STACK = {"torch_version": "fixture-torch", "hip_version": "fixture-hip", "rccl_version": "fixture-rccl",
         "rccl_library_sha256": "a" * 64}
BENCHMARK = {"protocol": "native_pp_primitives_v1", "timing_boundary": "synchronized_host_primitive_completion_v1",
    "p2p_direction": "unidirectional", "allgather_bytes": "full_output", "rank_reduction": "maximum_local_duration",
    "barrier_in_timing": False, "warmup_in_records": False, "startup_in_timing": False,
    "allocation_in_timing": False, "buffer_policy": "preallocated_out_of_place", "backend": "nccl",
    "stack": STACK, "launch": {"fixture": True}}


def experiment(replicas=1):
    return Experiment(workload=tuple(RequestSpec(f"r{i}", 0, 4, 2) for i in range(replicas)),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}", memory_bytes=128 * 1024**2)
                                     for i in range(4 * replicas))),
        strategy=StrategySpec(model=ModelSpec(activation_dtype="float16"),
            tensor_parallel_size=2, pipeline_parallel_size=2, replicas=replicas, replica_routing="round_robin",
            max_batch_size=1, max_batch_tokens=8, pipeline_microbatch_size=1,
            pipeline_max_inflight_microbatches=1, pipeline_transport="vllm_native"),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=5, decode_ms=1))


def service(case, size):
    if case["primitive"] == "allgather":
        return .003 + .000002 * size
    return (.002 + .000001 * size) if case["tp_rank"] == 0 else (.004 + .00000125 * size)


def dataset(exp, identifier="train", sizes=(128, 2048)):
    records = []
    for case in benchmark_cases(exp):
        for size in sizes:
            maximum = service(case, size)
            name = f"{identifier}/{case['primitive']}/{case['tp_rank']}/{size}"
            records.append({**case, "record_id": name, "group_id": name, "message_bytes": size,
                "duration_ms": maximum, "correctness_errors": 0,
                "rank_durations_ms": [{"rank": rank, "duration_ms": maximum if rank == case["participants"][-1]
                    else maximum * .9} for rank in case["participants"]]})
    return {"schema_version": 1, "kind": "native_pp_communication_measurements", "dataset_id": identifier,
        "identity": profile_identity(exp), "source": SOURCE, "records": records, "benchmark": BENCHMARK}


def raw_capture(exp, plan):
    normalized = normalize_pp_benchmark_plan(exp, plan)
    captures = []
    for rank, device in enumerate(exp.infra.devices):
        records = []
        for case in benchmark_cases(exp):
            if rank not in case["participants"]:
                continue
            for size in normalized["message_bytes"]:
                maximum = service(case, size)
                records.append({**sample_descriptor(normalized, case, size, 0, 0),
                    "duration_ms": maximum if rank == case["participants"][-1] else maximum * .9,
                    "correctness_errors": 0})
        captures.append({"schema_version": 1, "kind": "native_pp_primitive_rank_capture",
            "session_id": "cli-fixture-session",
            "protocol": "native_pp_primitives_v1", "rank": rank, "world_size": len(exp.infra.devices),
            "identity": profile_identity(exp), "source": SOURCE, "status": "COMPLETED", "stack": STACK,
            "device": {"index": rank, "name": device.name, "memory_bytes": device.memory_bytes},
            "launch": worker_launch_contract(exp, normalized), "records": records})
    return {"schema_version": 1, "kind": "native_pp_primitives_capture", "returncode": 0,
        "stdout": "", "stderr": "", "timed_out": False, "output_truncated": False, "rank_captures": captures}


class PPCommunicationCLITests(unittest.TestCase):
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
        for index in range(exp.strategy.replicas):
            local = replica_experiment(exp, index)
            profile = calibrate_pp_communication(dataset(local, f"train-{index}"),
                profile_version="native-cli-v1", allow_test_fixture=True)
            paths.append(str(self.write(f"source/communication-{index}.json", profile)))
        calibration = replace(exp.calibration, allow_test_fixture=True,
            native_pp_communication_profile_version="native-cli-v1")
        calibration = (replace(calibration, native_pp_communication_profile_path=paths[0])
            if exp.strategy.replicas == 1 else
            replace(calibration, replica_native_pp_communication_profile_paths=tuple(paths)))
        return replace(exp, calibration=calibration), paths

    def test_collect_import_fit_independent_holdout_and_predict_simulate(self):
        exp = experiment()
        config = self.write("experiment.json", asdict(exp))
        plan = {"schema_version": 1, "dataset_id": "cli-collected", "message_bytes": [128, 2048],
                "warmup_iterations": 1, "iterations": 1, "repeats": 1}
        raw = self.write("input/rank-captures.json", raw_capture(exp, plan))
        plan.update(raw_path="rank-captures.json", source=SOURCE)
        plan_path = self.write("input/plan.json", plan)
        collected, result = self.cli("collected", "collect", "--kind", "pp-communication",
            "--experiment", config, "--plan", plan_path, "--allow-test-fixture")
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(json.loads((collected / "dataset.json").read_text())["source"]["kind"], "test_fixture")
        manifest = json.loads((collected / "manifest.json").read_text())
        self.assertIn(hashlib.sha256(raw.read_bytes()).hexdigest(), {item["sha256"] for item in manifest["input_files"]})
        fitted, result = self.cli("fitted", "calibrate", "--native-pp-communication", "--dataset",
            collected / "dataset.json", "--profile-version", "native-cli-v1", "--allow-test-fixture")
        self.assertEqual(result["communication_group_count"], 3)
        holdout = self.write("holdout.json", dataset(exp, "independent-holdout", (256, 1024)))
        validated, result = self.cli("validated", "validate", "--native-pp-communication", "--profile",
            fitted / "profile.json", "--dataset", holdout, "--allow-test-fixture")
        self.assertFalse(result["hardware_validated"])
        self.assertTrue((validated / "holdout.json").exists())
        configured = replace(exp, calibration=replace(exp.calibration, allow_test_fixture=True,
            native_pp_communication_profile_path=str(fitted / "profile.json"),
            native_pp_communication_profile_version="native-cli-v1"))
        selected = self.write("selected.json", asdict(configured))
        predicted, result = self.cli("prediction", "predict", "--experiment", selected)
        # Two tensors; each charges maximum paired-rank P2P plus one receiver AG.
        expected_transfer = 2 * ((.004 + .00000125 * 512) + (.003 + .000002 * 1024))
        self.assertAlmostEqual(result["isolated_prefill_ms"], 5 + expected_transfer)
        self.assertEqual(result["communication"]["p2p_source"], "test_fixture_native_pp_communication_profile")
        self.assertEqual(result["communication"]["profile_provenance"]["snapshot_file"],
                         "communication-profiles/replica-0.json")
        self.assertTrue((predicted / "communication-profiles/replica-0.json").exists())
        _, simulated = self.cli("simulation", "simulate", "--experiment", selected)
        self.assertEqual(simulated["counts"]["completed"], 1)
        self.assertEqual(simulated["pipeline"]["communication_profile_provenance"]["snapshot_file"],
                         "communication-profiles/replica-0.json")

    def test_profile_bytes_change_config_hash_and_keep_exact_snapshot(self):
        exp, paths = self.configured()
        config = self.write("selected.json", asdict(exp))
        first, _ = self.cli("first", "predict", "--experiment", config)
        first_manifest = json.loads((first / "manifest.json").read_text())
        source = Path(paths[0])
        first_bytes = source.read_bytes()
        self.assertEqual((first / "communication-profiles/replica-0.json").read_bytes(), first_bytes)
        source.write_bytes(first_bytes + b"\n")
        second, _ = self.cli("second", "predict", "--experiment", config)
        second_manifest = json.loads((second / "manifest.json").read_text())
        self.assertNotEqual(first_manifest["config_sha256"], second_manifest["config_sha256"])
        self.assertNotEqual(first_manifest["native_pp_communication_profiles"][0]["sha256"],
                            second_manifest["native_pp_communication_profiles"][0]["sha256"])
        self.assertEqual((first / "communication-profiles/replica-0.json").read_bytes(), first_bytes)

    def test_stage_and_boundary_profiles_are_both_snapshotted_and_not_double_charged(self):
        exp, _ = self.configured()
        records = []
        for stage in range(2):
            for phase, query, context, duration in (("prefill", 4, 0, 2.5), ("decode", 1, 4, .5)):
                records.append({"record_id": f"stage{stage}-{phase}", "group_id": phase,
                    "phase": phase, "stage_index": stage, "query_tokens": [query], "context_lengths": [context],
                    "duration_ms": duration, "workspace_peak_bytes": None,
                    "pp_transport": {"outgoing" if stage == 0 else "incoming": native_pp_descriptor(exp, query)}})
        stage_data = {"schema_version": 1, "dataset_id": "stage-fixture", "identity": profile_identity(exp),
                      "source": SOURCE, "records": records}
        stage = calibrate_dataset(stage_data, profile_version="stage-cli-v1", allow_test_fixture=True)
        stage_path = self.write("source/stage.json", stage)
        exp = replace(exp, calibration=replace(exp.calibration, backend="profile", profile_path=str(stage_path),
            profile_version="stage-cli-v1"))
        config = self.write("both.json", asdict(exp))
        output, result = self.cli("both", "simulate", "--experiment", config)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertTrue((output / "profile.json").exists())
        self.assertTrue((output / "communication-profiles/replica-0.json").exists())
        self.assertIn("profile_sha256", manifest)
        self.assertEqual(len(manifest["native_pp_communication_profiles"]), 1)
        prefill = 5 + 2 * ((.004 + .00000125 * 512) + (.003 + .000002 * 1024))
        decode = 1 + 2 * ((.004 + .00000125 * 128) + (.003 + .000002 * 256))
        self.assertAlmostEqual(result["simulated_ms"], prefill + decode)

    def test_each_replica_uses_its_own_immutable_communication_profile(self):
        exp, paths = self.configured(experiment(replicas=2))
        config = self.write("cluster.json", asdict(exp))
        for mode in ("simulate", "predict"):
            output, result = self.cli(f"cluster-{mode}", mode, "--experiment", config)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["native_pp_communication_profiles"]), 2)
            key, report_key, section, provenance = (("replicas", "result", "pipeline", "communication_profile_provenance")
                if mode == "simulate" else ("replica_predictions", "prediction", "communication", "profile_provenance"))
            self.assertEqual(len(result[key]), 2)
            for item in result[key]:
                replica = item["replica_index"]
                self.assertEqual(item[report_key][section][provenance]["snapshot_file"],
                                 f"communication-profiles/replica-{replica}.json")
                self.assertTrue((output / f"communication-profiles/replica-{replica}.json").exists())
        second = Path(paths[1])
        second.write_bytes(second.read_bytes() + b"\n")
        changed, _ = self.cli("cluster-second-source-changed", "predict", "--experiment", config)
        changed_manifest = json.loads((changed / "manifest.json").read_text())
        self.assertNotEqual(manifest["config_sha256"], changed_manifest["config_sha256"])
        self.assertEqual(manifest["native_pp_communication_profiles"][0]["sha256"],
                         changed_manifest["native_pp_communication_profiles"][0]["sha256"])
        self.assertNotEqual(manifest["native_pp_communication_profiles"][1]["sha256"],
                            changed_manifest["native_pp_communication_profiles"][1]["sha256"])

    def test_communication_path_resolves_relative_to_referenced_calibration_section(self):
        exp, paths = self.configured()
        calibration = asdict(exp.calibration)
        calibration["native_pp_communication_profile_path"] = "../source/communication-0.json"
        section = self.write("settings/calibration.json", calibration)
        config = asdict(exp)
        config["calibration"] = "settings/calibration.json"
        output, _ = self.cli("referenced-config", "predict", "--experiment", self.write("referenced.json", config))
        manifest = json.loads((output / "manifest.json").read_text())
        sources = {item["path"] for item in manifest["input_files"]}
        self.assertIn(str(section.resolve()), sources)
        self.assertIn(str(Path(paths[0]).resolve()), sources)
        self.assertEqual((output / "communication-profiles/replica-0.json").read_bytes(), Path(paths[0]).read_bytes())

    def test_fixture_gates_placement_and_domain_failures_are_explicit(self):
        exp, paths = self.configured()
        original = json.loads(Path(paths[0]).read_text())
        no_opt_in = replace(exp, calibration=replace(exp.calibration, allow_test_fixture=False))
        _, result = self.cli("fixture-denied", "predict", "--experiment", self.write("denied.json", asdict(no_opt_in)), expected=2)
        self.assertEqual(result["error_code"], "TEST_FIXTURE_NOT_ALLOWED")
        changed = deepcopy(original)
        changed["groups"][0]["participants"] = [0, 3]
        Path(paths[0]).write_text(json.dumps(changed))
        _, result = self.cli("placement-denied", "predict", "--experiment", self.write("placement.json", asdict(exp)), expected=2)
        self.assertIn("PP_COMMUNICATION", result["error_code"])
        Path(paths[0]).write_text(json.dumps(original))
        # Refit an honest narrower range so the strict domain failure follows valid profile validation.
        narrow = calibrate_pp_communication(dataset(replace(exp, calibration=replace(exp.calibration,
            native_pp_communication_profile_path=None, native_pp_communication_profile_version="",
            allow_test_fixture=False)), "narrow", (512, 2048)), profile_version="native-cli-v1", allow_test_fixture=True)
        Path(paths[0]).write_text(json.dumps(narrow))
        output, result = self.cli("domain-denied", "predict", "--experiment", self.write("domain.json", asdict(exp)), expected=2)
        self.assertEqual(result["error_code"], "OUT_OF_COMMUNICATION_CALIBRATION_DOMAIN")
        self.assertTrue((output / "communication-profiles/replica-0.json").exists())
        self.assertTrue((output / "logs.txt").read_text())
        before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", "predict", "--experiment",
            str(self.root / "domain.json"), "--out", str(output)], cwd=ROOT,
            capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertIn("OUTPUT_EXISTS", process.stdout)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()})

    def test_native_flags_do_not_accept_legacy_profiles_or_implicit_split(self):
        exp = experiment()
        path = self.write("measurements.json", dataset(exp))
        self.cli("conflicting-flags", "calibrate", "--communication", "--native-pp-communication",
            "--dataset", path, "--profile-version", "bad", "--allow-test-fixture", expected=2)
        _, result = self.cli("split-denied", "validate", "--native-pp-communication", "--dataset", path,
            "--profile-version", "bad", "--allow-test-fixture", expected=2)
        self.assertEqual(result["error_code"], "INVALID_ARGUMENT")

    def test_unselected_boundary_settings_preserve_the_previous_config_fingerprint(self):
        exp = experiment()
        output, _ = self.cli("legacy-fingerprint", "predict", "--experiment",
                             self.write("legacy.json", asdict(exp)))
        manifest = json.loads((output / "manifest.json").read_text())
        previous = deepcopy(manifest["effective_config"])
        for key in ("native_pp_communication_profile_path", "replica_native_pp_communication_profile_paths",
                    "native_pp_communication_profile_version", "native_pp_communication_allow_extrapolation"):
            previous["calibration"].pop(key)
        expected = hashlib.sha256(json.dumps(previous, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
        self.assertEqual(manifest["config_sha256"], expected)
        self.assertNotIn("native_pp_communication_profiles", manifest)


if __name__ == "__main__":
    unittest.main()
