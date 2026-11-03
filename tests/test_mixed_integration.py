"""Online joint profiles and replayable PP/cache/swap mixed CLI evidence."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment, InfraSpec,
                       RequestSpec, SimulationError, StrategySpec, calibrate_dataset,
                       load_experiment, profile_identity, simulate, write_json_exclusive)


ROOT = Path(__file__).resolve().parents[1]


class MixedIntegrationTests(unittest.TestCase):
    def base(self, pp=1, tp=1):
        return Experiment(workload=(RequestSpec("decode-owner", 0, 4, 2),
                                     RequestSpec("prompt", 2, 4, 1)),
            infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(pp * tp))),
            strategy=StrategySpec(mixed_batching=True, max_batch_size=2, max_batch_tokens=5,
                pipeline_parallel_size=pp, tensor_parallel_size=tp,
                pipeline_microbatch_size=2, pipeline_p2p=False))

    def profiled(self, root, base, *, include_mixed=True):
        pp = base.strategy.pipeline_parallel_size
        records = []
        for stage in range(pp):
            records.append({"record_id": f"initial-{stage}", "group_id": f"initial-{stage}",
                "phase": "prefill", "stage_index": stage, "query_tokens": [4],
                "context_lengths": [0], "duration_ms": 2 / pp, "workspace_peak_bytes": 100})
            if include_mixed:
                records.append({"record_id": f"mixed-{stage}", "group_id": f"mixed-{stage}",
                    "phase": "mixed", "stage_index": stage, "query_tokens": [1, 4],
                    "context_lengths": [4, 0], "request_phases": ["decode", "prefill"],
                    "duration_ms": 3 / pp, "workspace_peak_bytes": 400 + stage})
        data = {"schema_version": 1, "dataset_id": "online-joint-stage-fixture",
                "identity": profile_identity(base), "source": {"kind": "test_fixture",
                "reference": "Hand-calculated service times, not GPU measurements"}, "records": records}
        profile = calibrate_dataset(data, profile_version="online-mixed", allow_test_fixture=True)
        path = root / "profile.json"
        write_json_exclusive(path, profile)
        return replace(base, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
            profile_version="online-mixed", allow_test_fixture=True)), path

    def test_runtime_profiles_need_no_fictional_pure_decode_member_lookup(self):
        for pp, tp in ((1, 1), (2, 2)):
            with self.subTest(pp=pp, tp=tp), tempfile.TemporaryDirectory() as temporary:
                exp, _ = self.profiled(Path(temporary), self.base(pp, tp))
                result = simulate(exp)
            self.assertEqual(result.result["counts"]["completed"], 2)
            self.assertEqual([row["token_times_ms"] for row in result.requests], [[2, 5], [5]])
            self.assertEqual(result.result["batching"]["mixed_batches_admitted"], 1)
            self.assertEqual(result.result["batching"]["mixed_batches_completed"], 1)
            self.assertEqual(result.result["batching"]["mixed_compute_busy_ms"], 3)
            self.assertEqual([rank["workspace_peak_bytes"] for rank in result.result["memory"]],
                             [400 + stage for stage in range(pp) for _ in range(tp)])

    def test_runtime_missing_joint_shape_is_not_a_pure_phase_timing_fallback(self):
        for pp in (1, 2):
            with self.subTest(pp=pp), tempfile.TemporaryDirectory() as temporary:
                exp, _ = self.profiled(Path(temporary), self.base(pp), include_mixed=False)
                with self.assertRaises(SimulationError) as raised:
                    simulate(exp)
                self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_profile_cli_snapshots_ordered_roles_and_emits_joint_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp, profile_path = self.profiled(root, self.base())
            config = root / "experiment.json"
            config.write_text(json.dumps(asdict(exp)))
            output = root / "run"
            process = subprocess.run([sys.executable, "-m", "mlsys_sim", "simulate",
                "--experiment", str(config), "--out", str(output)], cwd=ROOT,
                capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual((output / "profile.json").read_bytes(), profile_path.read_bytes())
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            mixed = [event for event in events if event["event"] == "STEP_START" and event["phase"] == "mixed"]
            self.assertEqual(len(mixed), 1)
            self.assertEqual(mixed[0]["request_ids"], ["decode-owner", "prompt"])
            self.assertEqual(mixed[0]["request_phases"], ["decode", "prefill"])
            self.assertEqual(mixed[0]["context_lengths"], [4, 0])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["comparison_contract"]["strategy"]["mixed_batching"])
            self.assertFalse(manifest["hardware_validated"])

    def test_warmup_mixed_service_does_not_double_count_the_measured_timeline(self):
        base = self.base()
        exp = replace(base, calibration=CalibrationSpec(backend="synthetic", prefill_ms=2, mixed_ms=3),
                      execution=ExecutionSpec(warmup=base.workload))
        result = simulate(exp)
        self.assertEqual(result.result["batching"]["mixed_batches_completed"], 1)
        self.assertEqual(result.result["warmup"]["batching"]["mixed_batches_completed"], 1)
        self.assertEqual(result.result["compute_busy_ms"], 5)
        self.assertEqual(result.result["warmup"]["compute_busy_ms"], 5)

    def test_example_cli_replays_and_combines_real_mixed_cache_and_swap_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, requests in (("mixed_batching", 2), ("pp_mixed_cache_swap", 5)):
                outputs = []
                for replay in range(2):
                    output = Path(temporary) / f"{name}-{replay}"
                    config = ROOT / "examples" / "v1_3" / f"{name}.json"
                    process = subprocess.run([sys.executable, "-m", "mlsys_sim", "simulate",
                        "--experiment", str(config), "--out", str(output)], cwd=ROOT,
                        capture_output=True, text=True, check=False)
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    result = json.loads((output / "result.json").read_text())
                    self.assertEqual(result["counts"]["completed"], requests)
                    self.assertGreater(result["batching"]["mixed_batches_completed"], 0)
                    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
                    mixed = [event for event in events if event.get("phase") == "mixed"]
                    self.assertTrue(mixed)
                    for event in mixed:
                        if "query_tokens" in event:
                            self.assertEqual(len(event["query_tokens"]), len(event["request_phases"]))
                            self.assertEqual(set(event["request_phases"]), {"decode", "prefill"})
                    if requests == 5:
                        self.assertEqual(result["prefix_cache"]["hit_tokens"], 12)
                        self.assertEqual(result["prefix_cache"]["attached_block_references"], 0)
                        self.assertEqual(result["prefix_cache"]["private_blocks"], 0)
                        self.assertEqual(result["swap"]["completed_transfers_by_direction"], {"D2H": 1, "H2D": 1})
                        self.assertEqual(result["swap"]["host_reserved_bytes"], 0)
                        self.assertEqual(result["recomputed_tokens"], 0)
                    outputs.append(output)
                for artifact in ("result.json", "requests.jsonl", "events.jsonl"):
                    self.assertEqual((outputs[0] / artifact).read_bytes(),
                                     (outputs[1] / artifact).read_bytes(), (name, artifact))

    def test_all_features_remain_compatible_on_tp_two_pp_two_with_transport(self):
        base = load_experiment(ROOT / "examples/v1_3/pp_mixed_cache_swap.json")
        exp = replace(base, infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(4))),
            strategy=replace(base.strategy, tensor_parallel_size=2, pipeline_p2p=True,
                pipeline_feedback_latency_us=500,
                memory=replace(base.strategy.memory, kv_pool_bytes=2560)))
        result = simulate(exp)
        self.assertEqual(result.result["counts"]["completed"], 5)
        self.assertGreater(result.result["batching"]["mixed_batches_completed"], 0)
        self.assertGreater(result.result["prefix_cache"]["hit_tokens"], 0)
        self.assertGreater(result.result["swap"]["completed_transfers_by_direction"]["D2H"], 0)
        self.assertEqual(result.result["swap"]["host_reserved_bytes"], 0)
        self.assertEqual(result.result["prefix_cache"]["attached_block_references"], 0)


if __name__ == "__main__":
    unittest.main()
