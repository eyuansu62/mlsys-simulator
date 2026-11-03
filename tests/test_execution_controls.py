"""Measurement-window and isolated warmup controls have explicit semantics."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, EvaluationSpec, ExecutionSpec, Experiment,
                       InfraSpec, MemoryConfig, RequestSpec, SimulationError, StrategySpec, simulate)
from mlsys_sim.profiles import profile_identity, shape_key


def experiment(requests=(), *, window=None, drain=True, max_drain=None, warmup=()):
    return Experiment(
        workload=tuple(requests),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=10,
                                    decode_ms=2, recompute_ms=5),
        evaluation=EvaluationSpec(measurement_window_ms=window, drain=drain,
                                  max_drain_ms=max_drain),
        execution=ExecutionSpec(warmup=tuple(warmup)),
    )


class MeasurementWindowTests(unittest.TestCase):
    def test_half_open_window_keeps_exclusions_auditable_without_execution(self):
        run = simulate(experiment([
            RequestSpec("inside", 0, 16, 1), RequestSpec("boundary", 5, 16, 1),
            RequestSpec("outside", 10, 16, 1),
        ], window=5))
        self.assertEqual(run.result["counts"], {
            "total": 3, "completed": 1, "failed": 0, "timed_out": 0,
            "unfinished": 0, "not_admitted": 2,
        })
        self.assertEqual(run.result["completion_rate"], 1)
        self.assertEqual(run.result["admitted_count"], 1)
        self.assertEqual(run.result["simulated_ms"], 10)
        self.assertEqual([row["status"] for row in run.requests],
                         ["COMPLETED", "NOT_ADMITTED", "NOT_ADMITTED"])
        self.assertFalse(any(event.get("request_id") in {"boundary", "outside"}
                             for event in run.events))
        self.assertFalse(run.result["measurement_contract"]["outside_window_in_completion_denominator"])

    def test_no_drain_interrupts_prefill_and_pins_inflight_memory(self):
        run = simulate(experiment([RequestSpec("a", 0, 16, 2)], window=5, drain=False))
        self.assertEqual(run.result["status"], "UNRESOLVED")
        self.assertEqual(run.result["error_code"], "MEASUREMENT_WINDOW_END")
        self.assertEqual(run.result["simulated_ms"], 5)
        self.assertEqual(run.result["compute_busy_ms"], 5)
        self.assertEqual(run.requests[0]["status"], "UNFINISHED")
        self.assertEqual(run.requests[0]["output_tokens_actual"], 0)
        self.assertGreater(run.result["memory"][0]["used_blocks"], 0)

    def test_completion_at_cutoff_commits_but_no_new_step_starts(self):
        run = simulate(experiment([RequestSpec("a", 0, 16, 3)], window=10, drain=False))
        self.assertEqual(run.requests[0]["output_tokens_actual"], 1)
        self.assertEqual(run.result["compute_busy_ms"], 10)
        self.assertEqual(run.result["simulated_ms"], 10)
        self.assertEqual([event["phase"] for event in run.events if event["event"] == "STEP_START"],
                         ["prefill"])

    def test_request_finishing_exactly_at_cutoff_is_successful(self):
        run = simulate(experiment([RequestSpec("a", 0, 16, 1)], window=10, drain=False))
        self.assertEqual(run.result["status"], "COMPLETED")
        self.assertEqual(run.result["counts"]["completed"], 1)
        self.assertEqual(run.result["compute_busy_ms"], 10)

    def test_finite_drain_is_relative_to_arrival_window_end(self):
        run = simulate(experiment([RequestSpec("a", 0, 16, 5)], window=5, max_drain=6))
        self.assertEqual(run.result["error_code"], "DRAIN_TIME_LIMIT")
        self.assertEqual(run.result["simulated_ms"], 11)
        self.assertEqual(run.result["compute_busy_ms"], 11)
        self.assertEqual(run.requests[0]["token_times_ms"], [10])
        self.assertEqual(run.requests[0]["num_computed_tokens"], 16)

    def test_hard_time_budget_can_end_before_drain_limit(self):
        exp = experiment([RequestSpec("a", 0, 16, 5)], window=5, max_drain=20)
        run = simulate(replace(exp, execution=replace(exp.execution, max_simulated_ms=7)))
        self.assertEqual(run.result["error_code"], "SIMULATION_TIME_LIMIT")
        self.assertEqual(run.result["simulated_ms"], 7)

    def test_sla_excludes_outside_requests_but_keeps_failures_in_denominator(self):
        exp = experiment([RequestSpec("a", 0, 16, 1), RequestSpec("timeout", 0, 16, 2, timeout_ms=1),
                          RequestSpec("outside", 5, 16, 1)], window=5)
        run = simulate(replace(exp, evaluation=replace(exp.evaluation, ttft_sla_ms=20)))
        self.assertEqual(run.result["completion_rate"], 0.5)
        self.assertEqual(run.result["completion_rate_denominator"], 2)
        self.assertFalse(run.result["sla_satisfied"])

    def test_empty_cohort_has_null_completion_and_declared_zero_rates(self):
        run = simulate(experiment([RequestSpec("outside", 5, 16, 1)], window=5))
        self.assertEqual(run.result["status"], "COMPLETED")
        self.assertIsNone(run.result["completion_rate"])
        self.assertIsNone(run.result["cohort_completion_rate"])
        self.assertEqual(run.result["throughput"]["requests_per_s"], 0)
        self.assertEqual(run.result["counts"]["not_admitted"], 1)

    def test_positive_duration_below_clock_resolution_cannot_create_zero_time_steps(self):
        exp = experiment([RequestSpec("a", 1.0, 16, 2)])
        exp = replace(exp, calibration=replace(exp.calibration, prefill_ms=5e-324))
        with self.assertRaises(SimulationError) as raised:
            simulate(exp)
        self.assertEqual(raised.exception.code, "INVALID_PERFORMANCE_ESTIMATE")


class WarmupTests(unittest.TestCase):
    def test_warmup_is_separate_from_measured_timeline_rates_and_memory(self):
        measured = RequestSpec("measured", 0, 16, 2)
        exp = experiment([measured], warmup=[RequestSpec("warm", 0, 32, 3)])
        warmed, cold = simulate(exp), simulate(replace(exp, execution=replace(exp.execution, warmup=())))
        self.assertEqual(warmed.requests, cold.requests)
        self.assertEqual(warmed.events, cold.events)
        self.assertEqual(warmed.result["simulated_ms"], 12)
        self.assertEqual(warmed.result["counts"]["total"], 1)
        self.assertEqual(warmed.result["warmup"]["request_count"], 1)
        self.assertEqual(warmed.result["warmup"]["simulated_ms"], 14)
        self.assertEqual(warmed.result["warmup"]["compute_busy_ms"], 14)
        self.assertEqual(warmed.result["warmup"]["status"], "COMPLETED")
        self.assertEqual(warmed.result["memory"], cold.result["memory"])
        self.assertEqual(warmed.result["latency_ms"], cold.result["latency_ms"])

    def test_warmup_ignores_main_measurement_window(self):
        exp = experiment([RequestSpec("measured", 0, 16, 1)], window=5,
                         warmup=[RequestSpec("warm", 8, 16, 1)])
        run = simulate(exp)
        self.assertEqual(run.result["warmup"]["simulated_ms"], 18)
        self.assertEqual(run.result["counts"]["completed"], 1)

    def test_failed_warmup_prevents_measurement(self):
        exp = experiment([RequestSpec("measured", 0, 1, 1)],
                         warmup=[RequestSpec("warm", 0, 17, 1)])
        exp = replace(exp, strategy=StrategySpec(memory=MemoryConfig(kv_pool_bytes=8192)))
        with self.assertRaises(SimulationError) as raised:
            simulate(exp)
        self.assertEqual(raised.exception.code, "WARMUP_FAILED")
        self.assertIn("KV_CAPACITY_EXCEEDED", str(raised.exception))

    def test_warmup_time_budget_failure_is_explicit(self):
        exp = experiment([RequestSpec("measured", 0, 1, 1)], warmup=[RequestSpec("warm", 0, 16, 2)])
        exp = replace(exp, execution=replace(exp.execution, max_simulated_ms=5))
        with self.assertRaises(SimulationError) as raised:
            simulate(exp)
        self.assertEqual(raised.exception.code, "WARMUP_FAILED")

    def test_unknown_initial_cache_is_not_silently_emulated(self):
        exp = experiment([RequestSpec("a", 0, 16, 1)])
        with self.assertRaises(SimulationError) as raised:
            simulate(replace(exp, execution=replace(exp.execution, initial_state="prefix_cache")))
        self.assertEqual(raised.exception.code, "UNSUPPORTED_MECHANISM")


class ProfileEngineTests(unittest.TestCase):
    def profile_experiment(self, directory, base, samples, *, extrapolate=False):
        path = Path(directory) / "profile.json"
        profile = {
            "schema_version": 1, "profile_id": "test-only", "profile_version": "fixture-v1",
            "identity": profile_identity(base),
            "source": {"kind": "test_fixture", "reference": "engine unit fixture; no hardware measurement"},
            "method": "lookup", "timing_includes_communication": True,
            "samples": [dict(sample, workspace_peak_bytes=None, sample_count=1) for sample in samples],
            "training": {
                "dataset_sha256": "0" * 64,
                "record_ids": [f"sample-{i}" for i in range(len(samples))],
                "group_ids": [f"group-{i}" for i in range(len(samples))],
                "shape_keys": [shape_key(sample["phase"], sample["query_tokens"],
                                         sample["context_lengths"], sample["stage_index"])
                               for sample in samples],
                "parent_profile_version": None,
            },
        }
        path.write_text(json.dumps(profile), encoding="utf-8")
        return replace(base, calibration=replace(base.calibration, backend="profile",
                                                profile_path=str(path), allow_test_fixture=True,
                                                allow_extrapolation=extrapolate))

    def test_tp_profile_is_not_charged_analytical_communication_again(self):
        base = replace(experiment([RequestSpec("a", 0, 16, 2)]),
                       infra=InfraSpec(devices=(DeviceSpec("g0"), DeviceSpec("g1")),
                                       link_latency_us=100000),
                       strategy=StrategySpec(tensor_parallel_size=2))
        samples = [
            {"phase": "prefill", "stage_index": 0, "query_tokens": [16],
             "context_lengths": [0], "duration_ms": 7},
            {"phase": "decode", "stage_index": 0, "query_tokens": [1],
             "context_lengths": [16], "duration_ms": 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            run = simulate(self.profile_experiment(directory, base, samples))
        self.assertEqual(run.requests[0]["token_times_ms"], [7, 10])
        self.assertEqual(run.result["calibration_source"], "test_fixture_profile")
        self.assertEqual(run.result["profile_version"], "fixture-v1")
        self.assertFalse(run.result["hardware_validated"])
        self.assertEqual(run.result["profile_provenance"]["source_kind"], "test_fixture")
        self.assertEqual(len(run.result["profile_provenance"]["sha256"]), 64)

    def test_uncovered_decode_does_not_silently_become_analytical(self):
        base = experiment([RequestSpec("a", 0, 16, 2)])
        samples = [{"phase": "prefill", "stage_index": 0, "query_tokens": [16],
                    "context_lengths": [0], "duration_ms": 7}]
        with tempfile.TemporaryDirectory() as directory:
            exp = self.profile_experiment(directory, base, samples)
            with self.assertRaises(SimulationError) as raised:
                simulate(exp)
        self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_explicit_extrapolation_is_reported_by_actual_execution(self):
        base = experiment([RequestSpec("a", 0, 24, 1)])
        samples = [
            {"phase": "prefill", "stage_index": 0, "query_tokens": [8],
             "context_lengths": [0], "duration_ms": 4},
            {"phase": "prefill", "stage_index": 0, "query_tokens": [16],
             "context_lengths": [0], "duration_ms": 8},
        ]
        with tempfile.TemporaryDirectory() as directory:
            exp = self.profile_experiment(directory, base, samples)
            with self.assertRaises(SimulationError) as raised:
                simulate(exp)
            self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")
            run = simulate(replace(exp, calibration=replace(exp.calibration, allow_extrapolation=True)))
        self.assertEqual(run.requests[0]["first_token_ms"], 12)
        self.assertTrue(run.result["profile_provenance"]["extrapolation_used"])


if __name__ == "__main__":
    unittest.main()
