"""Hand-computed queues, causal feedback, all-seed accounting and artifacts."""

from copy import deepcopy
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from mlsys_sim.calibration import calibrate_dataset
from mlsys_sim.config import load_experiment
from mlsys_sim.engine import simulate
import mlsys_sim.load_sweep as load_sweep
from mlsys_sim.load_sweep import run_load_sweep, write_load_report
from mlsys_sim.profiles import profile_identity
import mlsys_sim.profiles as profiles
from mlsys_sim.schema import (
    CalibrationSpec, DeviceSpec, Experiment, InfraSpec, RequestSpec, SimulationError, StrategySpec,
)


def experiment(**strategy):
    return Experiment(strategy=StrategySpec(scheduler="fifo", **strategy),
                      calibration=CalibrationSpec(backend="synthetic", prefill_ms=10, decode_ms=1))


def plan(clients=False):
    arrival = ({"process": "closed_loop", "duration_ms": 50, "max_requests": 100, "think_ms": 0}
               if clients else {"process": "periodic", "rate_rps": 100, "duration_ms": 100, "max_requests": 100})
    return {"schema_version": 1, "workload": {"schema_version": 1, "seed": 42, "arrival": arrival,
            "tokens": {"input": 16, "output": 1}}, "replications": 2, "drain_ms": 200,
            "min_completed_samples": 2, "slo": {"p99_e2e_ms": 30},
            ("concurrency_grid" if clients else "rate_grid_rps"): [1, 2] if clients else [100, 200]}


class LoadSweepTests(unittest.TestCase):
    def test_hand_computed_periodic_stable_and_overloaded(self):
        spec = plan()
        before = deepcopy(spec)
        report = run_load_sweep(experiment(), spec)
        self.assertEqual(spec, before)
        stable, overloaded = report["points"]
        self.assertEqual(stable["classification"], "observed_stable")
        self.assertEqual(overloaded["classification"], "observed_overloaded")
        self.assertEqual(stable["pooled_latency_ms"]["end_to_end"]["p99"], 10)
        self.assertEqual(overloaded["runs"][0]["summary"]["latency_ms"]["end_to_end"]["p99"], 104.05)
        self.assertEqual(stable["mean_requests_per_s"], 100)
        # Original cohort throughput includes drain: not silently capped at device service rate.
        self.assertEqual(overloaded["mean_requests_per_s"], 200)
        self.assertEqual(overloaded["mean_window_completed_requests_per_s"], 100)
        self.assertEqual(report["boundaries"]["sla_max_observed"], 100)
        self.assertEqual(report["boundaries"]["stable_overloaded_brackets"],
                         [{"stable": 100, "overloaded": 200, "resolution_rps": 100}])
        self.assertEqual(report["boundaries"]["slo_pass_fail_brackets"],
                         [{"pass": 100, "fail": 200, "resolution": 100, "axis": "offered_rps"}])
        self.assertEqual(report, run_load_sweep(experiment(), spec))

    def test_incomplete_drain_and_failures_veto_slo(self):
        spec = plan()
        spec["drain_ms"] = 0
        result = run_load_sweep(experiment(), spec)
        point = result["points"][1]
        self.assertEqual(point["classification"], "observed_failure")
        self.assertFalse(point["slo_pass"])
        self.assertEqual(point["runs"][0]["summary"]["completion_rate_denominator"], 20)
        spec["workload"]["timeout_ms"] = 5
        spec["slo"]["p99_e2e_ms"] = 1e6
        self.assertFalse(run_load_sweep(experiment(), spec)["points"][0]["slo_pass"])

    def test_closed_loop_uses_real_feedback_and_clients_are_not_batch_size(self):
        result = run_load_sweep(experiment(), plan(clients=True))
        one, two = result["points"]
        self.assertEqual(one["classification"], "observed_completed")
        self.assertEqual([row["arrival_ms"] for row in one["runs"][0]["workload"]["requests"]],
                         [0, 10, 20, 30, 40])
        self.assertEqual([row["arrival_ms"] for row in two["runs"][0]["workload"]["requests"]],
                         [0, 0, 10, 20, 30, 40])
        self.assertEqual(two["runs"][0]["result"]["counts"]["completed"], 6)
        self.assertEqual(two["pooled_latency_ms"]["end_to_end"]["p99"], 20)

    def test_closed_loop_timeout_releases_client_and_retains_failures(self):
        spec = plan(clients=True)
        spec["concurrency_grid"] = [1]
        spec["workload"]["timeout_ms"] = 5
        run = run_load_sweep(experiment(), spec)["points"][0]["runs"][0]
        self.assertEqual([row["arrival_ms"] for row in run["workload"]["requests"]], list(range(0, 50, 5)))
        self.assertEqual(run["result"]["counts"]["timed_out"], 10)
        self.assertFalse(run["summary"]["slo_pass"])

    def test_closed_loop_rejects_unrepresentable_positive_think_delay(self):
        spec = plan(clients=True)
        spec["workload"]["arrival"]["think_ms"] = 5e-324
        with self.assertRaisesRegex(SimulationError, "think deadline overflow or nonprogress"):
            run_load_sweep(experiment(), spec)

    def test_empirical_service_is_seeded_and_synthetic_only(self):
        spec = plan(clients=True)
        spec["service_time"] = {"kind": "empirical_step_scale", "samples": [.5, 1, 2]}
        first = run_load_sweep(experiment(), spec)
        self.assertEqual(first, run_load_sweep(experiment(), spec))
        runs = first["points"][0]["runs"]
        self.assertNotEqual(runs[0]["requests"], runs[1]["requests"])
        self.assertFalse(first["hardware_validated"])
        analytical = replace(experiment(), calibration=CalibrationSpec())
        with self.assertRaisesRegex(SimulationError, "requires synthetic"):
            run_load_sweep(analytical, spec)

    def test_empirical_duration_overflow_and_underflow_are_errors(self):
        for prefill, scale in ((1e308, 2), (5e-324, .1)):
            spec = plan(clients=True)
            spec["service_time"] = {"kind": "empirical_step_scale", "samples": [scale]}
            exp = replace(experiment(), calibration=CalibrationSpec(backend="synthetic", prefill_ms=prefill))
            with self.subTest(prefill=prefill, scale=scale), self.assertRaisesRegex(SimulationError, "overflow or underflow"):
                run_load_sweep(exp, spec)

    def test_profile_tp_pp_closed_loop_keeps_measured_shape_and_provenance(self):
        base = Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(id=f"gpu{i}") for i in range(4))),
                          strategy=StrategySpec(tensor_parallel_size=2, pipeline_parallel_size=2,
                                                pipeline_p2p=False, scheduler="fifo"))
        data = {"schema_version": 1, "dataset_id": "load-stage-fixture", "identity": profile_identity(base),
                "source": {"kind": "test_fixture", "reference": "hand-derived 5 ms/stage"},
                "records": [{"record_id": f"r{stage}", "group_id": f"g{stage}", "phase": "prefill",
                             "stage_index": stage, "query_tokens": [16], "context_lengths": [0],
                             "duration_ms": 5, "workspace_peak_bytes": 100} for stage in range(2)]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            original_bytes = (json.dumps(calibrate_dataset(data, profile_version="load-v1",
                                                           allow_test_fixture=True), indent=3) + "\r\n").encode()
            path.write_bytes(original_bytes)
            exp = replace(base, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                                                            allow_test_fixture=True))
            report = run_load_sweep(exp, plan(clients=True))
            point = report["points"][0]
            self.assertEqual(report["profile_inputs"][0]["raw_json"].encode(), original_bytes)
            self.assertEqual(report["profile_inputs"][0]["field"], "profile_path")
            self.assertEqual(report["profile_inputs"][0]["source_kind"], "test_fixture")
            write_load_report(report, Path(temp) / "frozen-output")
            self.assertEqual((Path(temp) / "frozen-output/inputs/profile-000.json").read_bytes(), original_bytes)
            self.assertEqual(report, run_load_sweep(exp, plan(clients=True)))
            self.assertNotIn("mlsys-load-inputs-", json.dumps(report))
            original_read = profiles._read_json
            reads = []
            def transient_original_change(selected):
                # The old live-file implementation read B while recording A.
                # A frozen sweep instead reads its A snapshot while origin is B.
                reads.append(str(Path(selected).resolve()))
                path.write_bytes(original_bytes + b"\n")
                try:
                    return original_read(selected)
                finally:
                    path.write_bytes(original_bytes)
            with patch("mlsys_sim.profiles._read_json", side_effect=transient_original_change):
                aba_report = run_load_sweep(exp, plan(clients=True))
            self.assertEqual(aba_report, report)
            self.assertTrue(reads)
            self.assertTrue(all(selected != str(path.resolve()) for selected in reads))
            expected_sha = report["profile_inputs"][0]["sha256"]
            self.assertTrue(all(run["result"]["profile_provenance"]["sha256"] == expected_sha
                                for candidate in aba_report["points"] for run in candidate["runs"]))
            original_run = load_sweep._run_dynamic
            def mutate_after_run(*args, **kwargs):
                outcome = original_run(*args, **kwargs)
                path.write_bytes(original_bytes + b"\n")
                return outcome
            with patch("mlsys_sim.load_sweep._run_dynamic", side_effect=mutate_after_run):
                with self.assertRaisesRegex(SimulationError, "Profile bytes changed") as raised:
                    run_load_sweep(exp, plan(clients=True))
            self.assertEqual(raised.exception.code, "INPUT_CHANGED")
        self.assertEqual(point["pooled_latency_ms"]["end_to_end"]["p99"], 10)
        self.assertEqual(point["runs"][0]["result"]["calibration_source"], "test_fixture_profile")
        self.assertIsNotNone(point["runs"][0]["result"]["profile_provenance"])

    def test_native_pp_and_workspace_frozen_sources_keep_reproducible_paths(self):
        root = Path(__file__).resolve().parents[1]
        exp = load_experiment(root / "examples/v1_9/native-pp-workspace.fixture.json")
        spec = plan()
        spec["rate_grid_rps"] = [10]
        spec["replications"] = 1
        spec["min_completed_samples"] = 1
        spec["workload"]["tokens"] = {"input": 4, "output": 2}
        report = run_load_sweep(exp, spec)
        self.assertEqual({item["field"] for item in report["profile_inputs"]},
                         {"native_pp_communication_profile_path", "workspace_profile_path"})
        self.assertEqual(report, run_load_sweep(exp, spec))
        self.assertNotIn("mlsys-load-inputs-", json.dumps(report))

    def test_all_single_and_replica_profile_roles_bind_one_readonly_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            changes = {}
            origins = {}
            for field in ("profile_path", "replica_profile_paths", "native_pp_communication_profile_path",
                          "replica_native_pp_communication_profile_paths", "workspace_profile_path",
                          "replica_workspace_profile_paths"):
                paths = []
                for index in range(2 if field.endswith("paths") else 1):
                    path = Path(temp) / f"{field}-{index}.json"
                    raw = json.dumps({"source": {"kind": "test_fixture", "reference": field}}).encode() + b"\r\n"
                    path.write_bytes(raw)
                    paths.append(str(path.resolve()))
                    origins[(field, index)] = (str(path.resolve()), raw)
                changes[field] = tuple(paths) if field.endswith("paths") else paths[0]
            exp = replace(experiment(), calibration=replace(experiment().calibration, **changes))
            captured_paths = []
            def inspect_bound(original, selected_plan, inputs, bound):
                self.assertEqual(original, exp)
                for item in inputs:
                    value = getattr(bound.calibration, item["field"])
                    selected = value[item["index"]] if item["field"].endswith("paths") else value
                    origin, raw = origins[(item["field"], item["index"])]
                    self.assertNotEqual(selected, origin)
                    self.assertEqual(Path(selected).read_bytes(), raw)
                    self.assertEqual(Path(selected).stat().st_mode & 0o222, 0)
                    captured_paths.append(selected)
                return asdict(bound)
            with patch("mlsys_sim.load_sweep._run_bound_sweep", side_effect=inspect_bound):
                restored = run_load_sweep(exp, plan())
            self.assertEqual(restored, asdict(exp))
            self.assertTrue(all(not Path(path).exists() for path in captured_paths))

    def test_warmup_isolated_from_measured_prefix_cache(self):
        exp = experiment(prefix_cache=True)
        exp = replace(exp, execution=replace(exp.execution, warmup=(
            RequestSpec("warm", 0, 32, 1, prompt_token_ids=tuple(range(32))),)))
        spec = plan(clients=True)
        spec["concurrency_grid"] = [1]
        spec["workload"]["tokens"]["input"] = 32
        spec["workload"]["prompt_token_ids"] = list(range(32))
        run = run_load_sweep(exp, spec)["points"][0]["runs"][0]
        self.assertEqual(run["requests"][0]["prefix_cached_tokens"], 0)
        self.assertEqual(run["result"]["warmup"]["status"], "COMPLETED")

    def test_nonmonotonic_metrics_and_mixed_seed_failure_remain_visible(self):
        spec = plan()
        calls = 0
        def varied_service(exp):
            nonlocal calls
            # Deliberately varying known service domains tests reporting rather
            # than imposing an unjustified monotonicity constraint on samples.
            duration = (10, 40, 1, 1)[calls]
            calls += 1
            return simulate(replace(exp, calibration=replace(exp.calibration, prefill_ms=duration)))
        with patch("mlsys_sim.load_sweep.simulate", side_effect=varied_service):
            report = run_load_sweep(experiment(), spec)
        self.assertTrue(report["points"][0]["mixed_replication_slo"])
        self.assertFalse(report["points"][0]["slo_pass"])
        self.assertTrue(report["boundaries"]["slo_nonmonotonic"])
        self.assertTrue(report["boundaries"]["p99_decreases_on_grid"])
        self.assertEqual([point["offered_rps"] for point in report["points"]], [100, 200])

    def test_short_runs_are_explicitly_insufficient(self):
        spec = plan()
        spec["min_completed_samples"] = 100
        result = run_load_sweep(experiment(), spec)
        self.assertIsNone(result["points"][0]["slo_pass"])
        self.assertEqual(result["points"][0]["classification"], "insufficient_samples")
        self.assertIsNone(result["boundaries"]["sla_max_observed"])

    def test_missing_declared_tpot_samples_is_unknown(self):
        spec = plan()
        spec["slo"] = {"p99_tpot_ms": 10}
        point = run_load_sweep(experiment(), spec)["points"][0]
        self.assertIsNone(point["slo_pass"])
        self.assertEqual(point["classification"], "insufficient_samples")

    def test_same_timestamp_released_clients_survive_admission_budget(self):
        exp = experiment()
        exp = replace(exp, execution=replace(exp.execution, max_events=1))
        spec = plan(clients=True)
        spec["concurrency_grid"] = [2]
        run = run_load_sweep(exp, spec)["points"][0]["runs"][0]
        self.assertEqual(len(run["workload"]["requests"]), 2)
        self.assertEqual(run["result"]["counts"]["unfinished"], 2)
        self.assertEqual(run["summary"]["completion_rate_denominator"], 2)
        self.assertFalse(run["summary"]["slo_pass"])

    def test_total_event_bound_applies_with_logging_disabled(self):
        exp = experiment()
        exp = replace(exp, execution=replace(exp.execution, detailed_events=False))
        with patch("mlsys_sim.load_sweep.MAX_SWEEP_EVENTS", 5):
            with self.assertRaisesRegex(SimulationError, "total sweep bound"):
                run_load_sweep(exp, plan())

    def test_tp_pp_cache_swap_mixed_replica_compatibility(self):
        base = Experiment(calibration=CalibrationSpec(backend="synthetic", prefill_ms=1, decode_ms=.5))
        strategy = replace(base.strategy, tensor_parallel_size=2, pipeline_parallel_size=2,
                           replicas=2, chunked_prefill=True, prefill_chunk_tokens=8,
                           mixed_batching=True, prefix_cache=True, preemption="swap",
                           memory=replace(base.strategy.memory, cpu_swap_bytes=4 * 1024**2))
        devices = tuple(DeviceSpec(id=f"gpu{i}") for i in range(8))
        exp = replace(base, strategy=strategy, infra=InfraSpec(devices=devices))
        spec = plan(clients=True)
        spec["workload"]["tokens"]["input"] = 32
        spec["workload"]["prompt_token_ids"] = list(range(32))
        spec["workload"]["tokens"]["output"] = 3
        spec["service_time"] = {"kind": "empirical_step_scale", "samples": [.8, 1.2]}
        result = run_load_sweep(exp, spec)
        self.assertTrue(all(run["result"]["counts"]["unfinished"] == 0
                            for point in result["points"] for run in point["runs"]))
        self.assertTrue(any(run["result"]["prefix_cache"]["hit_requests"] > 0
                            for point in result["points"] for run in point["runs"]))

    def test_artifacts_share_samples_and_refuse_overwrite(self):
        report = run_load_sweep(experiment(), plan())
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report"
            write_load_report(report, output)
            saved = json.loads((output / "report.json").read_text())
            self.assertEqual(saved, json.loads(json.dumps(report)))
            with (output / "points.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]["e2e_p99_ms"]), report["points"][0]["runs"][0]["summary"]
                             ["latency_ms"]["end_to_end"]["p99"])
            self.assertEqual(len(list((output / "runs").glob("*.json"))), 4)
            html = (output / "report.html").read_text()
            self.assertEqual(html.count("<figure>"), 4)
            self.assertIn("slo_pass_fail_brackets", html)
            for path in output.glob("*.svg"):
                ET.fromstring(path.read_text())
                self.assertIn("n=", path.read_text())
            self.assertIn("SLO ≤ 30 ms", (output / "end_to_end.svg").read_text())
            self.assertIn(">n=0</text>", (output / "tpot.svg").read_text())
            with self.assertRaises(FileExistsError):
                write_load_report(report, output)

    def test_invalid_plan_and_bounded_client_generation(self):
        for key, value in (("rate_grid_rps", []), ("rate_grid_rps", [2, 1]), ("rate_grid_rps", [1, 1]),
                           ("rate_grid_rps", [True]), ("replications", True), ("drain_ms", float("inf")),
                           ("slo", {}), ("slo", {"p99_e2e_ms": 0}), ("extra", 1)):
            with self.subTest(key=key, value=value), self.assertRaises(SimulationError):
                spec = plan()
                spec[key] = value
                run_load_sweep(experiment(), spec)
        spec = plan(clients=True)
        spec["workload"]["arrival"]["max_requests"] = 2
        with self.assertRaisesRegex(SimulationError, "no silent truncation"):
            run_load_sweep(experiment(), spec)
        for invalid in ("100", None, [], True):
            spec = plan(clients=True)
            spec["workload"]["arrival"]["max_requests"] = invalid
            with self.subTest(max_requests=invalid), self.assertRaises(SimulationError):
                run_load_sweep(experiment(), spec)
        spec = plan()
        spec["workload"]["arrival"]["max_requests"] = 100_000
        with self.assertRaisesRegex(SimulationError, "across all runs"):
            run_load_sweep(experiment(), spec)


if __name__ == "__main__":
    unittest.main()
