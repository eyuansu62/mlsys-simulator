"""Finite fixed-N feedback, frozen streams and non-refitting holdout validation."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim.calibration import calibrate_dataset
from mlsys_sim.load_validation import (
    closed_loop_policy_sha256, generate_request_stream, predict_closed_loop,
    validate_closed_loop, validate_closed_loop_spec,
)
from mlsys_sim.profiles import profile_identity
import mlsys_sim.profiles as profiles
from mlsys_sim.schema import CalibrationSpec, DeviceSpec, Experiment, InfraSpec, SimulationError, StrategySpec
from mlsys_sim.workload import canonical_sha256


def study(**changes):
    return {"schema_version": 1, "study_id": "lesson", "cohort": "short", "concurrency_grid": [1, 2],
            "requests_per_run": 6, "replications": 2, "seed": 42, "think_ms": 0,
            "input_lengths": [16, 32], "output_tokens": 3, "timeout_ms": 1000, "max_run_ms": 10_000,
            **changes}


def experiment(**strategy):
    return Experiment(strategy=StrategySpec(scheduler="fifo", **strategy),
                      calibration=CalibrationSpec(backend="synthetic", prefill_ms=10, decode_ms=1))


def validation_contract(**changes):
    return {"schema_version": 1, "prediction_frozen_at": "2026-11-11T01:00:00Z", "min_completed_samples": 6,
            "timing_boundary": "server_token_ready_v1", "max_relative_error": {
                "ttft_p99_ms": 0, "tpot_p99_ms": 0, "end_to_end_p99_ms": 0, "elapsed_output_tokens_per_s": 0},
            "constraints": {"schema_version": 1, "min_completed_samples": 6,
                "latency": {"p99_ttft_ms": 1000, "p99_tpot_ms": 1000, "p99_e2e_ms": 1000},
                "throughput": {"basis": "elapsed", "min_output_tokens_per_s": 1}},
            "allow_test_fixture": True, **changes}


def observations(predictions):
    runs = deepcopy(predictions["runs"])
    for index, run in enumerate(runs):
        run.update(source={"kind": "test_fixture", "reference": "CPU protocol fixture, never GPU evidence"},
                   session_id=f"fixture-{index}", collection_started_at="2026-11-11T02:00:00+00:00",
                   collection_ended_at="2026-11-11T02:01:00+00:00")
    return {"schema_version": 1, "study_id": predictions["study_id"], "cohort": predictions["cohort"],
            "spec_sha256": predictions["spec_sha256"], "identity_sha256": predictions["identity_sha256"], "runs": runs}


def dilate(run, factor):
    for row in run["requests"]:
        for name in ("arrival_ms", "first_token_ms", "completion_ms", "ttft_ms", "tpot_ms", "end_to_end_ms"):
            if row.get(name) is not None:
                row[name] *= factor
        for name in ("token_times_ms", "itl_ms"):
            row[name] = [value * factor for value in row[name]]
    result = run["result"]
    result["simulated_ms"] *= factor
    for values in result["latency_ms"].values():
        for key in values:
            if values[key] is not None:
                values[key] *= factor
    run["elapsed_ms"] *= factor


class FiniteStreamTests(unittest.TestCase):
    def test_frozen_stream_replay_uses_categorical_lengths_and_is_concurrency_independent(self):
        spec = study(requests_per_run=100)
        before = deepcopy(spec)
        first = generate_request_stream(spec, replication=0)
        self.assertEqual(first, generate_request_stream(spec, replication=0))
        self.assertEqual(len(first["requests"]), 100)
        self.assertEqual(set(row["input_tokens"] for row in first["requests"]), {16, 32})
        self.assertTrue(all("arrival_ms" not in row and "client_id" not in row for row in first["requests"]))
        different_grid = deepcopy(spec)
        different_grid["concurrency_grid"] = [1, 4, 8]
        self.assertEqual(first, generate_request_stream(different_grid, replication=0))
        self.assertNotEqual(first, generate_request_stream(spec, replication=1))
        self.assertNotEqual(closed_loop_policy_sha256(spec, 1, 0), closed_loop_policy_sha256(spec, 2, 0))
        self.assertEqual(spec, before)

    def test_invalid_specs_and_aggregate_bounds(self):
        for changes in ({"seed": True}, {"requests_per_run": 0}, {"requests_per_run": True},
                        {"replications": 101}, {"concurrency_grid": [2, 1]}, {"concurrency_grid": [True]},
                        {"input_lengths": []}, {"input_lengths": [16, 16]}, {"input_lengths": [True]},
                        {"think_ms": -1}, {"timeout_ms": float("nan")}, {"max_run_ms": float("inf")},
                        {"max_run_ms": 3_600_001}, {"prompt_token_id": False}, {"extra": 1},
                        {"requests_per_run": 100_000, "replications": 3}):
            with self.subTest(changes=changes), self.assertRaises(SimulationError):
                validate_closed_loop_spec(study(**changes))


class FinitePredictionTests(unittest.TestCase):
    def test_hand_computed_real_feedback_and_exact_N_then_drain(self):
        prediction = predict_closed_loop(experiment(), study())
        one, _, two, _ = prediction["runs"]
        self.assertEqual(prediction["status"], "COMPLETED")
        self.assertEqual(one["elapsed_ms"], 72)
        self.assertEqual(two["elapsed_ms"], 72)
        self.assertEqual([row["arrival_ms"] for row in one["requests"]], [0, 12, 24, 36, 48, 60])
        self.assertEqual([row["arrival_ms"] for row in two["requests"]], [0, 0, 12, 24, 36, 48])
        self.assertEqual([row["client_id"] for row in two["requests"]], [0, 1, 0, 1, 0, 1])
        self.assertEqual(two["not_offered_count"], 0)
        self.assertEqual(one["request_stream_sha256"], two["request_stream_sha256"])
        self.assertEqual(prediction, predict_closed_loop(experiment(), study()))

    def test_finite_target_smaller_than_clients_and_positive_think(self):
        spec = study(concurrency_grid=[10], requests_per_run=2, replications=1, think_ms=5)
        result = predict_closed_loop(experiment(), spec)["runs"][0]
        self.assertEqual([row["client_id"] for row in result["requests"]], [0, 1])
        self.assertEqual(result["offered_count"], 2)
        spec = study(concurrency_grid=[1], requests_per_run=3, replications=1, think_ms=5)
        result = predict_closed_loop(experiment(), spec)["runs"][0]
        self.assertEqual([row["arrival_ms"] for row in result["requests"]], [0, 17, 34])
        self.assertEqual(result["elapsed_ms"], 46)

    def test_timeout_refills_but_does_not_become_success(self):
        result = predict_closed_loop(experiment(), study(concurrency_grid=[1], replications=1, timeout_ms=5))["runs"][0]
        self.assertEqual([row["arrival_ms"] for row in result["requests"]], [0, 5, 10, 15, 20, 25])
        self.assertEqual(result["offered_count"], 6)
        self.assertEqual(result["result"]["counts"]["timed_out"], 6)
        self.assertEqual(result["status"], "UNRESOLVED")

    def test_budget_stops_future_release_and_retains_same_timestamp_offers(self):
        spec = study(concurrency_grid=[1], replications=1, max_run_ms=20)
        result = predict_closed_loop(experiment(), spec)["runs"][0]
        self.assertEqual(result["offered_count"], 2)
        self.assertEqual(result["not_offered_count"], 4)
        self.assertEqual(result["status"], "UNRESOLVED")
        exp = experiment()
        exp = replace(exp, execution=replace(exp.execution, max_events=1))
        result = predict_closed_loop(exp, study(concurrency_grid=[2], replications=1))["runs"][0]
        self.assertEqual(len(result["requests"]), 2)
        self.assertEqual(result["offered_count"], 2)
        self.assertEqual(result["not_offered_count"], 4)
        self.assertEqual(result["result"]["counts"]["unfinished"], 2)

    def test_tp_pp_replicas_cache_swap_mixed_mechanisms_remain_in_original_engine(self):
        exp = Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(id=f"g{i}") for i in range(8))),
            strategy=StrategySpec(tensor_parallel_size=2, pipeline_parallel_size=2, replicas=2,
                                  prefix_cache=True, chunked_prefill=True, prefill_chunk_tokens=16,
                                  mixed_batching=True, preemption="swap"),
            calibration=CalibrationSpec(backend="synthetic", prefill_ms=1, decode_ms=1))
        exp = replace(exp, strategy=replace(exp.strategy, memory=replace(exp.strategy.memory, cpu_swap_bytes=2**20)))
        predicted = predict_closed_loop(exp, study(concurrency_grid=[2], replications=1, requests_per_run=10))
        self.assertEqual(predicted["status"], "COMPLETED")
        self.assertTrue(predicted["runs"][0]["resources_drained"])
        self.assertGreater(predicted["runs"][0]["result"]["prefix_cache"]["hit_requests"], 0)

    def test_profile_is_snapshotted_and_source_ABA_cannot_change_prediction(self):
        base = experiment()
        data = {"schema_version": 1, "dataset_id": "finite-fixture", "identity": profile_identity(base),
                "source": {"kind": "test_fixture", "reference": "CPU service, not GPU"},
                "records": [{"record_id": "prefill", "group_id": "prefill", "phase": "prefill", "stage_index": 0,
                             "query_tokens": [16], "context_lengths": [0], "duration_ms": 10}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            raw = json.dumps(calibrate_dataset(data, profile_version="finite-v1", allow_test_fixture=True)).encode()
            path.write_bytes(raw)
            exp = replace(base, calibration=CalibrationSpec(backend="profile", profile_path=str(path), allow_test_fixture=True))
            spec = study(input_lengths=[16], output_tokens=1)
            first = predict_closed_loop(exp, spec)
            self.assertEqual(first, predict_closed_loop(exp, spec))
            reader = profiles._read_json
            def temporary_origin_change(selected):
                path.write_bytes(raw + b"\n")
                try:
                    self.assertNotEqual(Path(selected).resolve(), path.resolve())
                    return reader(selected)
                finally:
                    path.write_bytes(raw)
            with patch("mlsys_sim.profiles._read_json", side_effect=temporary_origin_change):
                self.assertEqual(first, predict_closed_loop(exp, spec))
            self.assertNotIn("mlsys-finite-load-inputs-", json.dumps(first))


class IndependentValidationTests(unittest.TestCase):
    def setUp(self):
        self.predicted = predict_closed_loop(experiment(), study())
        self.observed = observations(self.predicted)

    def test_equal_policy_metric_validation_is_sanitized_and_read_only(self):
        before = deepcopy(self.predicted), deepcopy(self.observed)
        report = validate_closed_loop(self.predicted, self.observed, validation_contract())
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["status"], "COMPLETED")
        self.assertEqual(len(report["runs"]), 4)
        self.assertEqual((self.predicted, self.observed), before)
        public = json.dumps(report)
        self.assertNotIn('"requests":', public)
        self.assertNotIn('"token_times_ms":', public)
        self.assertNotIn('"prompt_token_id":', public)
        self.assertNotIn("CPU protocol fixture", public)

    def test_different_actual_arrivals_are_allowed_and_error_denominator_is_measured(self):
        for run in self.observed["runs"]:
            dilate(run, 2)
        contract = validation_contract(max_relative_error={name: .6 for name in
            ("ttft_p99_ms", "tpot_p99_ms", "end_to_end_p99_ms")})
        report = validate_closed_loop(self.predicted, self.observed, contract)
        self.assertEqual(report["verdict"], "PASS")
        self.assertAlmostEqual(report["runs"][0]["metrics"]["ttft_p99_ms"]["relative_error"], .5)
        contract["max_relative_error"] = {"ttft_p99_ms": .1}
        failed = validate_closed_loop(self.predicted, self.observed, contract)
        self.assertEqual(failed["status"], "COMPLETED")
        self.assertEqual(failed["verdict"], "FAIL")
        self.assertEqual(len(failed["runs"]), 4)

    def test_failed_seed_low_sample_and_missing_cell_are_never_selected_away(self):
        row = self.observed["runs"][1]["requests"][0]
        row["status"] = "TIMED_OUT"
        self.observed["runs"][1]["result"]["counts"]["completed"] -= 1
        self.observed["runs"][1]["result"]["counts"]["timed_out"] += 1
        report = validate_closed_loop(self.predicted, self.observed, validation_contract())
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["points"][0]["replication_verdicts"], ["PASS", "FAIL"])
        self.observed = observations(self.predicted)
        self.observed["runs"].pop()
        report = validate_closed_loop(self.predicted, self.observed, validation_contract())
        self.assertEqual(report["verdict"], "UNKNOWN")
        self.assertEqual(len(report["runs"]), 4)
        report = validate_closed_loop(self.predicted, observations(self.predicted),
                                       validation_contract(min_completed_samples=100))
        self.assertEqual(report["verdict"], "UNKNOWN")

    def test_boundary_and_freeze_order_cannot_be_forged_into_pass(self):
        self.observed["runs"][0]["timing_boundary"] = "client_token_delivery_v1"
        report = validate_closed_loop(self.predicted, self.observed, validation_contract())
        self.assertEqual(report["verdict"], "FAIL")
        self.observed = observations(self.predicted)
        self.observed["runs"][0]["collection_started_at"] = "2026-11-11T00:59:59Z"
        self.assertEqual(validate_closed_loop(self.predicted, self.observed, validation_contract())["verdict"], "FAIL")
        self.observed["runs"][0]["collection_started_at"] = "2026-11-11T02:00:00"
        self.assertEqual(validate_closed_loop(self.predicted, self.observed, validation_contract())["verdict"], "UNKNOWN")

    def test_stream_policy_identity_payload_and_fixture_source_are_strict(self):
        for key in ("policy_sha256", "request_stream_sha256"):
            broken = deepcopy(self.observed)
            broken["runs"][0][key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(SimulationError):
                validate_closed_loop(self.predicted, broken, validation_contract())
        broken = deepcopy(self.observed)
        broken["identity_sha256"] = "0" * 64
        with self.assertRaises(SimulationError):
            validate_closed_loop(self.predicted, broken, validation_contract())
        with self.assertRaises(SimulationError):
            validate_closed_loop(self.predicted, self.observed, validation_contract(allow_test_fixture=False))
        with self.assertRaises(SimulationError):
            validate_closed_loop(self.predicted, self.observed, validation_contract(prediction_payload_sha256="0" * 64))
        contract = validation_contract(prediction_payload_sha256=canonical_sha256(self.predicted))
        self.assertEqual(validate_closed_loop(self.predicted, self.observed, contract)["verdict"], "PASS")

    def test_actual_prompt_and_output_evidence_is_checked_not_only_declared_hash(self):
        self.observed["runs"][0]["requests"][0]["input_tokens"] = 999
        self.assertEqual(validate_closed_loop(self.predicted, self.observed, validation_contract())["verdict"], "FAIL")
        self.observed = observations(self.predicted)
        self.observed["runs"][0]["requests"][0].pop("prompt_token_ids_sha256")
        self.assertEqual(validate_closed_loop(self.predicted, self.observed, validation_contract())["verdict"], "UNKNOWN")

    def test_precomputed_early_arrival_and_missing_client_feedback_cannot_pass(self):
        self.observed["runs"][0]["requests"][1]["arrival_ms"] = 0
        report = validate_closed_loop(self.predicted, self.observed, validation_contract())
        self.assertEqual(report["verdict"], "FAIL")
        self.assertTrue(any(check["id"] == "observation.cohort.closed_loop_feedback" and check["verdict"] == "FAIL"
                            for check in report["runs"][0]["checks"]))
        self.observed = observations(self.predicted)
        self.observed["runs"][0]["requests"][0].pop("client_id")
        self.assertEqual(validate_closed_loop(self.predicted, self.observed, validation_contract())["verdict"], "UNKNOWN")

    def test_hard_cohort_time_budget_and_known_prediction_failure_remain_fail(self):
        self.observed["runs"][0]["result"]["simulated_ms"] = 10_001
        self.assertEqual(validate_closed_loop(self.predicted, self.observed, validation_contract())["verdict"], "FAIL")
        spec = study(max_run_ms=20)
        prediction = predict_closed_loop(experiment(), spec)
        observation = observations(prediction)
        observation["runs"] = []
        report = validate_closed_loop(prediction, observation, validation_contract())
        self.assertEqual(report["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
