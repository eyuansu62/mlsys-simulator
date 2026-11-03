"""Observed runs must be comparable, sourced and complete about missing data."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim.comparison import (comparison_contract, compare_candidates, compare_runs,
                                  extract_metrics)
from mlsys_sim.config import experiment_from_dict
from mlsys_sim.schema import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, ModelSpec,
                              RequestSpec, SimulationError, StrategySpec)


def contract() -> dict:
    return comparison_contract(Experiment(
        infra=InfraSpec(devices=(DeviceSpec(name="example-device"),)),
        strategy=StrategySpec(model=ModelSpec(name="example-model", revision="rev1",
                                              weight_dtype="float16", kv_dtype="float16",
                                              activation_dtype="float16")),
        calibration=CalibrationSpec(runtime_name="example-runtime", runtime_version="v1",
                                    driver_version="not-applicable", rocm_version="not-applicable",
                                    rccl_version="not-applicable")))


def prediction() -> dict:
    return {
        "status": "COMPLETED", "backend": "simulated", "calibration_source": "analytical",
        "counts": {"total": 2}, "completion_rate": 1.0, "sla_satisfied": True,
        "latency_ms": {"ttft": {"p50": 12, "p95": 14, "p99": 15},
                       "tpot": {"p50": 2, "p95": 3, "p99": 4}},
        "throughput": {"requests_per_s": 20, "output_tokens_per_s": 40},
        "goodput": {"requests_per_s": 10, "output_tokens_per_s": 20},
    }


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save(self, name="first", *, score=10, observed_score=8, source="test_fixture",
             comparison=None, observed_comparison=None, observed_requests=None):
        directory = self.root / name
        directory.mkdir()
        result = prediction()
        result["goodput"]["requests_per_s"] = score
        current = comparison or contract()
        (directory / "result.json").write_text(json.dumps(result))
        (directory / "manifest.json").write_text(json.dumps({
            "backend": "simulated", "comparison_contract": current}))
        (directory / "requests.jsonl").write_text('\n'.join(json.dumps(row) for row in [
            {"request_id": "a", "ttft_ms": 12, "tpot_ms": 2, "end_to_end_ms": 16},
            {"request_id": "b", "ttft_ms": 14, "tpot_ms": 3, "end_to_end_ms": 20},
        ]) + '\n')
        observation = {
            "schema_version": 1, "kind": "measured_run",
            "comparison_contract": observed_comparison or current,
            "source": {"kind": source, "reference": "unit-test-fixture-not-hardware"},
            "metrics": {"ttft_p50_ms": 10, "goodput_requests_per_s": observed_score,
                        "completion_rate": 0.5, "sla_satisfied": False, "request_count": 2},
        }
        if observed_requests is not None:
            observation["requests"] = observed_requests
        observed_path = self.root / f"{name}-observed.json"
        observed_path.write_text(json.dumps(observation))
        return directory, observed_path

    def test_absolute_relative_error_and_sla_disagreement(self):
        directory, observed = self.save()
        report = compare_runs(directory, observed, allow_test_fixture=True)
        self.assertEqual(report["metrics"]["ttft_p50_ms"]["absolute_error"], 2)
        self.assertEqual(report["metrics"]["ttft_p50_ms"]["relative_error"], 0.2)
        self.assertEqual(report["metrics"]["ttft_p50_ms"]["unit"], "ms")
        self.assertFalse(report["sla_judgment_agreement"])
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["tail_latency_evidence"], "limited")
        self.assertEqual(report["source"]["kind"], "test_fixture")
        self.assertFalse(report["hardware_validated"])

    def test_zero_observation_and_missing_metrics_are_not_perfect_predictions(self):
        directory, observed = self.save(observed_score=0)
        report = compare_runs(directory / "result.json", observed, allow_test_fixture=True)
        zero = report["metrics"]["goodput_requests_per_s"]
        self.assertEqual(zero["absolute_error"], 10)
        self.assertIsNone(zero["relative_error"])
        self.assertEqual(zero["relative_error_reason"], "zero_observation")
        missing = report["metrics"]["tpot_p50_ms"]
        self.assertIsNone(missing["absolute_error"])
        self.assertEqual(missing["reason"], "observation_unavailable")
        self.assertFalse(report["coverage"]["complete"])
        self.assertLess(report["coverage"]["fraction"], 1)

    def test_source_is_required_and_fixture_is_explicit(self):
        directory, observed = self.save()
        with self.assertRaises(SimulationError) as raised:
            compare_runs(directory, observed)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        for source in ("simulated", "synthetic", "analytical"):
            directory, observed = self.save(source, source=source)
            with self.assertRaises(SimulationError) as raised:
                compare_runs(directory, observed, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "INVALID_OBSERVATION_SOURCE")
        directory, observed = self.save("measured", source="measured")
        self.assertEqual(compare_runs(directory, observed)["source"]["kind"], "measured")

    def test_simulator_result_cannot_be_used_as_observation(self):
        directory, observed = self.save()
        observed.write_text((directory / "result.json").read_text())
        with self.assertRaises(SimulationError) as raised:
            compare_runs(directory, observed, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "INVALID_OBSERVATION_SOURCE")

    def test_identity_policy_window_and_timing_must_match(self):
        for key in ("identity", "strategy", "evaluation", "execution"):
            changed = deepcopy(contract())
            if key == "identity":
                changed[key]["runtime"]["version"] = "v2"
            elif key == "strategy":
                changed[key]["max_batch_size"] += 1
            elif key == "evaluation":
                changed[key]["ttft_sla_ms"] = 1.0
            else:
                changed[key]["seed"] += 1
            directory, observed = self.save(key, observed_comparison=changed)
            with self.assertRaises(SimulationError) as raised:
                compare_runs(directory, observed, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "COMPARISON_MISMATCH")

    def test_unknown_identity_and_timing_are_not_comparable_by_agreement(self):
        for field in ("identity", "timing_boundary"):
            unknown = deepcopy(contract())
            unknown[field] = {"model": "fixture"} if field == "identity" else "unknown-timing"
            directory, observed = self.save(field, comparison=unknown)
            with self.assertRaises(SimulationError):
                compare_runs(directory, observed, allow_test_fixture=True)

    def test_agreeing_but_incomplete_contracts_are_rejected(self):
        incomplete = deepcopy(contract())
        del incomplete["execution"]
        directory, observed = self.save(comparison=incomplete)
        with self.assertRaises(SimulationError) as raised:
            compare_runs(directory, observed, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "INVALID_COMPARISON_INPUT")

    def test_partial_strategy_and_inconsistent_layout_are_rejected(self):
        for name in ("partial", "layout"):
            changed = deepcopy(contract())
            if name == "partial":
                changed["strategy"] = {"scheduler": "fifo"}
            else:
                changed["strategy"]["tensor_parallel_size"] = 2
            directory, observed = self.save(name, comparison=changed)
            with self.assertRaises(SimulationError) as raised:
                compare_runs(directory, observed, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "INVALID_COMPARISON_INPUT")

    def test_requests_are_joined_by_id_and_unpaired_rows_are_retained(self):
        directory, observed = self.save(observed_requests=[
            {"request_id": "c", "ttft_ms": 100},
            {"request_id": "a", "ttft_ms": 10, "tpot_ms": None, "end_to_end_ms": 15},
        ])
        report = compare_runs(directory, observed, allow_test_fixture=True)["request_comparison"]
        self.assertEqual(report["paired_count"], 1)
        self.assertEqual(report["unpaired_predicted_ids"], ["b"])
        self.assertEqual(report["unpaired_observed_ids"], ["c"])
        self.assertEqual(report["rows"][0]["request_id"], "a")
        self.assertEqual(report["rows"][0]["metrics"]["ttft_ms"]["absolute_error"], 2)
        self.assertFalse(report["complete_pairing"])

    def test_duplicate_request_ids_and_unknown_metrics_are_rejected(self):
        directory, observed = self.save(observed_requests=[{"request_id": "a"}, {"request_id": "a"}])
        with self.assertRaises(SimulationError):
            compare_runs(directory, observed, allow_test_fixture=True)
        directory, observed = self.save("unknown")
        data = json.loads(observed.read_text())
        data["metrics"]["latency_seconds"] = 10
        observed.write_text(json.dumps(data))
        with self.assertRaises(SimulationError):
            compare_runs(directory, observed, allow_test_fixture=True)

    def test_nonfinite_negative_boolean_and_out_of_range_metrics_are_rejected(self):
        for index, (name, value) in enumerate([
            ("ttft_p50_ms", float("nan")), ("ttft_p50_ms", -1),
            ("ttft_p50_ms", True), ("completion_rate", 1.1),
            ("sla_satisfied", 1), ("request_count", 1.5),
        ]):
            directory, observed = self.save(str(index))
            data = json.loads(observed.read_text())
            data["metrics"][name] = value
            observed.write_text(json.dumps(data))
            with self.assertRaises(SimulationError):
                compare_runs(directory, observed, allow_test_fixture=True)

    def test_rankings_preserve_ties_and_measure_pairwise_ordering(self):
        pairs = []
        for name, score, observed_score in [("a", 10, 20), ("b", 10, 20), ("c", 5, 30)]:
            predicted, observed = self.save(name, score=score, observed_score=observed_score)
            pairs.append({"candidate_id": name, "predicted": predicted, "observed": observed})
        report = compare_candidates(pairs, allow_test_fixture=True)
        self.assertEqual(report["predicted_ranking"], [["a", "b"], ["c"]])
        self.assertEqual(report["observed_ranking"], [["c"], ["a", "b"]])
        self.assertAlmostEqual(report["pairwise_agreement"], 1 / 3)
        self.assertTrue(report["ranking_complete"])
        self.assertEqual(report["unit"], "requests/s")

    def test_candidate_missing_score_is_excluded_instead_of_ranked_as_zero(self):
        pairs = []
        for name in ("a", "b"):
            predicted, observed = self.save(name)
            if name == "b":
                data = json.loads(observed.read_text())
                data["metrics"]["goodput_requests_per_s"] = None
                observed.write_text(json.dumps(data))
            pairs.append({"candidate_id": name, "predicted": str(predicted), "observed": str(observed)})
        report = compare_candidates(pairs, allow_test_fixture=True)
        self.assertEqual(report["ranked_candidates"], 1)
        self.assertEqual(report["predicted_ranking"], [["a"]])
        self.assertFalse(report["ranking_complete"])
        self.assertIsNone(report["pairwise_agreement"])

    def test_candidate_pairs_need_common_workload_and_evaluation(self):
        a, a_obs = self.save("a")
        different = deepcopy(contract())
        different["workload_sha256"] = "b" * 64
        b, b_obs = self.save("b", comparison=different)
        with self.assertRaises(SimulationError) as raised:
            compare_candidates([
                {"candidate_id": "a", "predicted": str(a), "observed": str(a_obs)},
                {"candidate_id": "b", "predicted": str(b), "observed": str(b_obs)},
            ], allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "COMPARISON_MISMATCH")

    def test_contract_ignores_calibration_fit_and_logging_but_tracks_workload_and_warmup(self):
        original = Experiment(workload=(RequestSpec("a", 0, 16, 2),))
        base = comparison_contract(original)
        fit_changed = replace(original, calibration=replace(original.calibration, compute_efficiency=0.25))
        self.assertEqual(base, comparison_contract(fit_changed))
        logging_changed = replace(original, execution=replace(original.execution, detailed_events=False,
                                                                max_events=123))
        self.assertEqual(base, comparison_contract(logging_changed))
        warmup_changed = replace(original, execution=replace(original.execution,
                                                             warmup=(RequestSpec("warm", 0, 8, 1),)))
        self.assertNotEqual(base, comparison_contract(warmup_changed))
        runtime_changed = replace(original, calibration=CalibrationSpec(runtime_version="new-runtime"))
        self.assertNotEqual(base, comparison_contract(runtime_changed))

    def test_extract_metrics_uses_declared_units_and_preserves_null(self):
        metrics = extract_metrics(prediction())
        self.assertEqual(metrics["throughput_output_tokens_per_s"], 40)
        self.assertEqual(metrics["ttft_p50_ms"], 12)
        self.assertIsNone(metrics["itl_p99_ms"])
        self.assertIs(metrics["sla_satisfied"], True)

    def test_metric_request_count_uses_measurement_cohort(self):
        run = prediction()
        run["counts"]["total"] = 100
        run["evaluation_count"] = 2
        run["cohort_completion_rate"] = 0.5
        self.assertEqual(extract_metrics(run)["request_count"], 2)
        self.assertEqual(extract_metrics(run)["completion_rate"], 0.5)

    def test_dataclass_and_loader_numeric_types_have_the_same_contract(self):
        direct = Experiment(workload=(RequestSpec("r", 0, 16, 3),))
        loaded = experiment_from_dict({"workload": [{"request_id": "r", "arrival_ms": 0.0,
                                                     "input_tokens": 16, "output_tokens": 3}]})
        self.assertEqual(comparison_contract(direct), comparison_contract(loaded))
        changed = deepcopy(contract())
        changed["identity"]["devices"][0]["compute_tflops"] = 1
        changed["identity"]["topology"]["link_bandwidth_gbps"] = 20
        changed["execution"]["max_simulated_ms"] = 180000
        directory, observed = self.save(observed_comparison=changed)
        report = compare_runs(directory, observed, allow_test_fixture=True)
        self.assertEqual(report["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
