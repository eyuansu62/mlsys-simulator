"""Public collection commands retain evidence and never promote CPU fixtures."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim import CalibrationSpec, DeviceSpec, Experiment, InfraSpec, RequestSpec, StrategySpec


ROOT = Path(__file__).resolve().parents[1]


class HardwareCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload))
        return path

    def command(self, name, *arguments, expected=0):
        output = self.root / name
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", *map(str, arguments), "--out", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return output, json.loads((output / "result.json").read_text())

    def experiment(self):
        exp = Experiment(workload=(RequestSpec("r", 0, 4, 2),),
                         calibration=CalibrationSpec(backend="synthetic", prefill_ms=2, decode_ms=1))
        return self.write("experiment.json", asdict(exp))

    def trace(self, boundary="server_token_ready_v1"):
        return self.write("trace.json", {"schema_version": 1, "kind": "runtime_trace",
            "source": {"kind": "test_fixture", "reference": "CPU CLI hand-calculated token timing"},
            "timing_boundary": boundary, "observed_until_ms": 3,
            "events": [{"event": "REQUEST_ARRIVAL", "request_id": "r", "timestamp_ms": 0},
                {"event": "TOKEN_READY", "request_id": "r", "timestamp_ms": 2,
                 "token_count": 1, "output_token_index": 1},
                {"event": "TOKEN_READY", "request_id": "r", "timestamp_ms": 3,
                 "token_count": 1, "output_token_index": 2},
                {"event": "REQUEST_FINISH", "request_id": "r", "timestamp_ms": 3, "status": "COMPLETED"}]})

    def test_package_import_does_not_load_optional_torch(self):
        process = subprocess.run([sys.executable, "-c", "import sys, mlsys_sim; print('torch' in sys.modules)"],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "False")

    def test_probe_is_a_readiness_report_with_explicit_raw_inventory(self):
        output, result = self.command("probe", "collect", "--kind", "probe")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertIs(type(result["hardware_available"]), bool)
        self.assertFalse(result["hardware_validated"])
        inventory = json.loads((output / "inventory.json").read_text())
        self.assertEqual(inventory["kind"], "rocm_inventory")
        self.assertIn("pytorch_collective_build_version", inventory["torch"])

    def test_missing_hardware_prevents_factory_import_and_keeps_diagnostics(self):
        inventory = self.write("inventory.json", {"schema_version": 1, "kind": "rocm_inventory",
            "status": "UNAVAILABLE", "source": {"kind": "test_fixture", "reference": "CPU environment"}})
        plan = self.write("plan.json", {"schema_version": 1})
        output, result = self.command("no-gpu", "collect", "--kind", "steps", "--experiment", self.experiment(),
            "--plan", plan, "--inventory", inventory, "--adapter", "module_that_must_not_be_imported:factory",
            "--allow-test-fixture", expected=2)
        self.assertEqual(result["error_code"], "HARDWARE_UNAVAILABLE")
        self.assertTrue((output / "inventory.json").is_file())
        self.assertTrue((output / "logs.txt").read_text())
        self.assertFalse((output / "dataset.json").exists())

    def test_server_trace_import_simulation_and_compare_form_a_fixture_loop(self):
        config = self.experiment()
        output, result = self.command("observed", "collect", "--kind", "runtime-trace", "--experiment", config,
                                     "--raw", self.trace(), "--allow-test-fixture")
        self.assertTrue(result["canonical_comparable"])
        self.assertFalse(result["hardware_validated"])
        observation = json.loads((output / "observed.json").read_text())
        self.assertEqual(observation["source"]["kind"], "test_fixture")
        predicted, _ = self.command("predicted", "simulate", "--experiment", config)
        _, comparison = self.command("comparison", "compare", "--predicted", predicted,
            "--observed", output / "observed.json", "--allow-test-fixture")
        self.assertEqual(comparison["metrics"]["ttft_p50_ms"]["absolute_error"], 0)
        self.assertEqual(comparison["metrics"]["tpot_p50_ms"]["absolute_error"], 0)
        self.assertFalse(comparison["hardware_validated"])

    def test_client_chunks_and_unpermitted_fixtures_do_not_become_server_observations(self):
        output, result = self.command("wrong-boundary", "collect", "--kind", "runtime-trace",
            "--experiment", self.experiment(), "--raw", self.trace("client_content_delivery_v1"),
            "--allow-test-fixture", expected=2)
        self.assertEqual(result["error_code"], "INCOMPATIBLE_TIMING_BOUNDARY")
        self.assertFalse((output / "observed.json").exists())
        _, result = self.command("unpermitted", "collect", "--kind", "runtime-trace",
            "--experiment", self.experiment(), "--raw", self.trace(), expected=2)
        self.assertEqual(result["error_code"], "TEST_FIXTURE_NOT_ALLOWED")

    def test_fixture_acceptance_is_explicit_and_output_is_immutable(self):
        plan = ROOT / "examples/v1_5/fixture-validation.plan.json"
        output, result = self.command("acceptance", "validate-hardware", "--plan", plan, "--allow-test-fixture")
        self.assertEqual(result["verdict"], "TEST_FIXTURE_ONLY")
        self.assertTrue(result["criteria_passed"])
        self.assertFalse(result["hardware_validated"])
        self.assertFalse(json.loads((output / "manifest.json").read_text())["hardware_validated"])
        previous = (output / "result.json").read_bytes()
        _, repeated = self.command("acceptance", "validate-hardware", "--plan", plan,
                                    "--allow-test-fixture", expected=2)
        # An existing directory must survive the rejected second invocation.
        self.assertEqual((output / "result.json").read_bytes(), previous)
        self.assertEqual(repeated["verdict"], "TEST_FIXTURE_ONLY")

    def test_audit_only_validation_reports_missing_evidence(self):
        audit = self.write("audit.json", {"source": "reference audit", "canonical_stage_profile": False})
        plan = self.write("audit.plan.json", {"schema_version": 1, "evidence_audit": audit.name})
        _, result = self.command("audit-acceptance", "validate-hardware", "--plan", plan)
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(result["hardware_validated"])

    def test_required_measured_pass_exits_nonzero_for_a_passing_fixture(self):
        _, result = self.command("required-acceptance", "validate-hardware", "--plan",
            ROOT / "examples/v1_5/fixture-validation.plan.json", "--allow-test-fixture", "--require-pass", expected=3)
        self.assertTrue(result["criteria_passed"])
        self.assertEqual(result["verdict"], "TEST_FIXTURE_ONLY")
        self.assertFalse(result["hardware_validated"])

    def test_wrong_kind_flags_are_not_silently_ignored(self):
        _, result = self.command("bad-args", "collect", "--kind", "probe", "--raw", self.trace(), expected=2)
        self.assertEqual(result["error_code"], "INVALID_ARGUMENT")

    def rccl(self, wrong=0):
        raw = "\n".join(["# Collective test starting: all_reduce_perf",
            "# nThread 1 nGpus 2 minBytes 1024 maxBytes 1024 step: 1(bytes) warmup iters: 1 iters: 2 agg iters: 1 validation: 1 graph: 0",
            "rccl-tests: Version fixture-source", "# RCCL version 2.30.7",
            "# Rank 0 Group 0 Pid 123 on fixture device 0 [0000:01:00] AMD Fixture 0",
            "# Rank 1 Group 0 Pid 123 on fixture device 1 [0000:02:00] AMD Fixture 1",
            "# out-of-place", "# size count type redop root time algbw busbw #wrong",
            "# (B) (elements) (us) (GB/s) (GB/s)", f"1024 256 float sum -1 10 100 50 {wrong}",
            "# Out of bounds values : 0 OK", "# Avg bus bandwidth : 50",
            "# Collective test concluded: all_reduce_perf", ""])
        path = self.root / "rccl.log"
        path.write_text(raw)
        exp = Experiment(infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
                         strategy=StrategySpec(tensor_parallel_size=2))
        config = self.write("rccl-experiment.json", asdict(exp))
        plan = self.write("rccl.plan.json", {"schema_version": 1, "dataset_id": "cpu-rccl-fixture",
            "ranks": 2, "mode": "out_of_place", "raw_path": path.name,
            "source": {"kind": "test_fixture", "reference": "Synthetic CPU parser fixture"},
            "algorithm": "Ring", "rank_aggregation": "max"})
        return path, config, plan

    def test_raw_rccl_import_keeps_fixture_source_and_exact_log_hash(self):
        raw, config, plan = self.rccl()
        output, result = self.command("rccl", "collect", "--kind", "rccl", "--experiment", config,
                                     "--plan", plan, "--allow-test-fixture")
        dataset = json.loads((output / "dataset.json").read_text())
        self.assertEqual(dataset["source"]["kind"], "test_fixture")
        self.assertEqual(dataset["records"][0]["duration_ms"], .01)
        self.assertFalse(result["hardware_validated"])
        captured = json.loads((output / "raw_records.json").read_text())
        self.assertEqual(captured[0]["sha256"], hashlib.sha256(raw.read_bytes()).hexdigest())

    def test_failed_rccl_parse_retains_partial_raw_without_a_success_dataset(self):
        _, config, plan = self.rccl(wrong=1)
        output, result = self.command("failed-rccl", "collect", "--kind", "rccl", "--experiment", config,
                                     "--plan", plan, "--allow-test-fixture", expected=2)
        self.assertEqual(result["error_code"], "INVALID_RCCL_OUTPUT")
        self.assertTrue((output / "raw_records.json").is_file())
        self.assertFalse((output / "dataset.json").exists())


if __name__ == "__main__":
    unittest.main()
