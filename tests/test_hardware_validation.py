"""Acceptance predicates are separate from fixture success and tool execution."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, EvaluationSpec, Experiment, InfraSpec,
                       ModelSpec, RequestSpec, SimulationError, StrategySpec,
                       comparison_contract, profile_identity, replica_experiment)
from mlsys_sim.hardware_validation import run_hardware_validation


class HardwareValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def inputs(self, *, replicas=1, kind="test_fixture"):
        # These values exercise declared-source acceptance logic only; no GPU
        # measurement is performed by this unit test, including kind=measured.
        base = Experiment(
            workload=tuple(RequestSpec(chr(97 + i), 0, 4, 3) for i in range(replicas)),
            infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}", name=f"AMD-unit-contract-{i}")
                                         for i in range(replicas))),
            strategy=StrategySpec(replicas=replicas, replica_routing="round_robin", scheduler="fifo",
                model=ModelSpec(name="unit-contract-model", revision="unit-contract-revision",
                                weight_dtype="float16", kv_dtype="float16", activation_dtype="float16")),
            calibration=CalibrationSpec(runtime_name="unit-contract-runtime", runtime_version="1.0",
                driver_version="unit-driver", rocm_version="unit-rocm", rccl_version="not-applicable"),
            evaluation=EvaluationSpec(ttft_sla_ms=20, tpot_sla_ms=2))
        source = {"kind": kind, "reference": "Unit contract arithmetic only, not hardware evidence"}
        steps = []
        for index in range(replicas):
            local = replica_experiment(base, index)
            records = []
            for name, phase, query, context, duration in [
                ("low", "prefill", 2, 0, 1 + index), ("high", "prefill", 6, 0, 3 + index),
                ("decode-4", "decode", 1, 4, 1), ("decode-5", "decode", 1, 5, 1)]:
                records.append({"record_id": name, "group_id": name, "phase": phase,
                    "stage_index": 0, "query_tokens": [query], "context_lengths": [context],
                    "duration_ms": duration, "workspace_peak_bytes": None})
            dataset = {"schema_version": 1, "dataset_id": f"unit-replica-{index}",
                       "identity": profile_identity(local), "source": source, "records": records}
            holdout = deepcopy(dataset)
            holdout["dataset_id"] += "-holdout"
            holdout["records"] = [{"record_id": "middle", "group_id": "middle", "phase": "prefill",
                "stage_index": 0, "query_tokens": [4], "context_lengths": [0],
                "duration_ms": 2 + index, "workspace_peak_bytes": None}]
            self.write(f"training-{index}.json", dataset)
            self.write(f"holdout-{index}.json", holdout)
            steps.append({"replica_index": index, "training_dataset": f"training-{index}.json",
                          "holdout_dataset": f"holdout-{index}.json", "profile_version": f"unit-v{index}"})
        self.write("experiment.json", asdict(base))
        observed = {"schema_version": 1, "kind": "measured_run",
            "comparison_contract": comparison_contract(base), "source": source,
            "metrics": {"ttft_p50_ms": 2 + (replicas - 1) / 2, "tpot_p50_ms": 1,
                        "completion_rate": 1, "request_count": replicas, "sla_satisfied": True},
            "requests": [{"request_id": chr(97 + index), "ttft_ms": 2 + index,
                           "tpot_ms": 1, "end_to_end_ms": 4 + index} for index in range(replicas)]}
        self.write("observation.json", observed)
        plan = {"schema_version": 1, "steps": steps,
            "online": [{"experiment": "experiment.json", "observation": "observation.json", "group_id": "same-workload"}],
            "thresholds": {"min_holdout_records": 1, "min_holdout_coverage": 1,
                "max_step_mape_percent": 1, "min_online_requests": replicas, "min_online_repeats": 1,
                "max_online_relative_error": {"ttft_p50_ms": .1, "tpot_p50_ms": .1},
                "require_sla_agreement": True}}
        return plan, base, observed

    def run_plan(self, plan, name="acceptance", *, fixtures=True):
        return run_hardware_validation(plan, self.root / name, base_dir=self.root,
                                       allow_test_fixture=fixtures)

    def test_fixture_loop_passes_criteria_without_claiming_hardware_validation(self):
        plan, _, _ = self.inputs()
        result = self.run_plan(plan)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["verdict"], "TEST_FIXTURE_ONLY")
        self.assertTrue(result["criteria_passed"])
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(result["steps"][0]["summary"]["coverage"], 1)
        self.assertEqual(result["steps"][0]["summary"]["mape_percent"], 0)
        self.assertEqual(result["online"][0]["paired_count"], 1)
        self.assertEqual(result["repeat_variation"]["same-workload"]["independent_repeats"], 1)
        self.assertEqual(len(result["validation_scope_sha256"]), 64)
        self.assertTrue((self.root / "acceptance/online-0/predicted/profile.json").exists())

    def test_declared_measured_success_is_scoped_to_inputs_not_a_general_verdict(self):
        plan, _, _ = self.inputs(kind="measured")
        result = self.run_plan(plan, fixtures=False)
        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(result["hardware_validated"])
        self.assertEqual(result["validation_scope"]["dimensions"], ["step_latency", "online_metrics"])
        self.assertEqual(result["validation_scope"]["online_metrics"], ["tpot_p50_ms", "ttft_p50_ms"])
        self.assertFalse(result["validation_scope"]["communication_prediction_validated"])

    def test_one_fixture_source_prevents_measured_promotion(self):
        plan, _, observation = self.inputs(kind="measured")
        observation["source"]["kind"] = "test_fixture"
        self.write("observation.json", observation)
        result = self.run_plan(plan)
        self.assertEqual(result["source_kinds"], ["measured", "test_fixture"])
        self.assertTrue(result["criteria_passed"])
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(result["verdict"], "TEST_FIXTURE_ONLY")

    def test_fixture_permission_is_required_at_the_evidence_boundary(self):
        plan, _, _ = self.inputs()
        with self.assertRaises(SimulationError) as caught:
            self.run_plan(plan, fixtures=False)
        self.assertEqual(caught.exception.code, "TEST_FIXTURE_NOT_ALLOWED")

    def test_error_threshold_failure_is_a_completed_failed_acceptance(self):
        plan, _, observation = self.inputs(kind="measured")
        observation["metrics"]["ttft_p50_ms"] = 4
        observation["requests"][0]["ttft_ms"] = 4
        self.write("observation.json", observation)
        result = self.run_plan(plan, fixtures=False)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["criteria_passed"])
        self.assertFalse(result["hardware_validated"])

    def test_insufficient_holdout_coverage_cannot_pass_on_covered_rows(self):
        plan, _, _ = self.inputs(kind="measured")
        path = self.root / "holdout-0.json"
        holdout = json.loads(path.read_text())
        holdout["records"][0]["query_tokens"] = [8]
        path.write_text(json.dumps(holdout))
        result = self.run_plan(plan, fixtures=False)
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["steps"][0]["summary"]["coverage"], 0)
        self.assertIsNone(result["steps"][0]["summary"]["mape_percent"])

    def test_missing_or_zero_relative_metric_is_not_zero_error(self):
        for label, value in (("missing", None), ("zero", 0)):
            plan, _, observed = self.inputs(kind="measured")
            observed["metrics"]["ttft_p50_ms"] = value
            self.write("observation.json", observed)
            result = self.run_plan(plan, name=label, fixtures=False)
            self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
            criterion = next(row for row in result["criteria"] if row["name"] == "online-0.ttft_p50_ms.relative_error")
            self.assertIsNone(criterion["observed"])

    def test_aggregate_request_count_does_not_replace_actual_paired_rows(self):
        plan, _, observed = self.inputs(kind="measured")
        observed.pop("requests")
        observed["metrics"]["request_count"] = 1000
        self.write("observation.json", observed)
        result = self.run_plan(plan, fixtures=False)
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["online"][0]["paired_count"], 0)

    def test_p99_requires_at_least_one_hundred_paired_samples(self):
        plan, _, observed = self.inputs(kind="measured")
        plan["thresholds"]["max_online_relative_error"] = {"ttft_p99_ms": .1}
        observed["metrics"]["ttft_p99_ms"] = 2
        self.write("observation.json", observed)
        result = self.run_plan(plan, fixtures=False)
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        sample = next(row for row in result["criteria"] if row["name"] == "online-0.paired_requests")
        self.assertEqual(sample["threshold"], 100)

    def test_distinct_repeats_preserve_observed_spread(self):
        plan, _, observed = self.inputs()
        second = deepcopy(observed)
        second["metrics"]["ttft_p50_ms"] = 2.02
        second["requests"][0]["ttft_ms"] = 2.02
        second["requests"][0]["end_to_end_ms"] = 4.02
        self.write("observation-2.json", second)
        plan["online"].append({"experiment": "experiment.json", "observation": "observation-2.json", "group_id": "same-workload"})
        plan["thresholds"]["min_online_repeats"] = 2
        result = self.run_plan(plan)
        self.assertTrue(result["criteria_passed"])
        values = result["repeat_variation"]["same-workload"]
        self.assertEqual(values["independent_repeats"], 2)
        self.assertAlmostEqual(values["metrics"]["ttft_p50_ms"]["mean"], 2.01)
        self.assertGreater(values["metrics"]["ttft_p50_ms"]["sample_stdev"], 0)

    def test_duplicate_evidence_cannot_be_disguised_by_source_reference(self):
        plan, _, observed = self.inputs()
        observed["source"]["reference"] = "Different filename and label, same measurement content"
        self.write("duplicate.json", observed)
        plan["online"].append({"experiment": "experiment.json", "observation": "duplicate.json", "group_id": "same-workload"})
        with self.assertRaises(SimulationError) as caught:
            self.run_plan(plan)
        self.assertEqual(caught.exception.code, "DUPLICATE_EVIDENCE")

    def test_holdout_shape_leakage_remains_a_hard_error(self):
        plan, _, _ = self.inputs()
        path = self.root / "training-0.json"
        training = json.loads(path.read_text())
        training["records"].append({"record_id": "different-id", "group_id": "different-group",
            "phase": "prefill", "stage_index": 0, "query_tokens": [4], "context_lengths": [0],
            "duration_ms": 2, "workspace_peak_bytes": None})
        path.write_text(json.dumps(training))
        with self.assertRaises(SimulationError) as caught:
            self.run_plan(plan)
        self.assertEqual(caught.exception.code, "VALIDATION_DATA_LEAKAGE")

    def test_numeric_forms_and_row_order_do_not_create_independent_repeats(self):
        plan, _, observed = self.inputs(replicas=2)
        observed["metrics"]["request_count"] = 2.0
        observed["metrics"]["completion_rate"] = 1.0
        observed["requests"].reverse()
        for row in observed["requests"]:
            row["ttft_ms"] = float(row["ttft_ms"])
        self.write("numeric-duplicate.json", observed)
        plan["online"].append({"experiment": "experiment.json", "observation": "numeric-duplicate.json",
                               "group_id": "same-workload"})
        with self.assertRaises(SimulationError) as caught:
            self.run_plan(plan)
        self.assertEqual(caught.exception.code, "DUPLICATE_EVIDENCE")

    def test_repeat_group_cannot_mix_different_arrival_contracts(self):
        plan, base, observed = self.inputs()
        changed = replace(base, workload=(replace(base.workload[0], arrival_ms=1),))
        self.write("changed-experiment.json", asdict(changed))
        observed["comparison_contract"] = comparison_contract(changed)
        self.write("changed-observation.json", observed)
        plan["online"].append({"experiment": "changed-experiment.json", "observation": "changed-observation.json",
                               "group_id": "same-workload"})
        with self.assertRaises(SimulationError) as caught:
            self.run_plan(plan)
        self.assertEqual(caught.exception.code, "INVALID_HARDWARE_VALIDATION_PLAN")

    def test_communication_fit_is_preserved_without_automatic_double_charge(self):
        plan, base, _ = self.inputs(replicas=2)
        communication = {"schema_version": 1, "kind": "communication_measurements",
            "dataset_id": "standalone-unit-communication", "identity": profile_identity(base),
            "source": {"kind": "test_fixture", "reference": "Hand arithmetic, not RCCL measurements"},
            "records": [{"record_id": str(size), "group_id": "ring-two", "primitive": "allreduce",
                         "ranks": 2, "message_bytes": size, "duration_ms": .02 + size / 20e6}
                        for size in (1000, 1000000)]}
        self.write("communication.json", communication)
        plan.update(communication_dataset="communication.json", communication_profile_version="comm-v1")
        result = self.run_plan(plan)
        self.assertTrue(result["criteria_passed"])
        self.assertEqual(result["communication_profile"], "communication-profile.json")
        fit = json.loads((self.root / "acceptance/communication-profile.json").read_text())
        self.assertFalse(fit["application"]["automatic_application"])
        self.assertAlmostEqual(fit["groups"][0]["alpha_us"], 10)
        effective = json.loads((self.root / "acceptance/online-0/experiment.json").read_text())
        self.assertEqual(effective["infra"]["link_latency_us"], base.infra.link_latency_us)
        self.assertFalse(result["validation_scope"]["communication_prediction_validated"])

    def test_multi_replica_profiles_are_fitted_and_snapshotted_separately(self):
        plan, _, _ = self.inputs(replicas=2)
        result = self.run_plan(plan)
        self.assertTrue(result["criteria_passed"])
        self.assertEqual(len(result["steps"]), 2)
        predicted = json.loads((self.root / "acceptance/online-0/predicted/result.json").read_text())
        self.assertEqual(predicted["counts"]["completed"], 2)
        self.assertEqual(predicted["profile_version"], "per_replica")
        self.assertTrue((self.root / "acceptance/online-0/predicted/profiles/replica-1.json").is_file())
        plan["steps"].pop()
        with self.assertRaises(SimulationError):
            self.run_plan(plan, name="missing-replica")

    def test_prediction_domain_failure_retains_diagnostics_and_cannot_pass(self):
        plan, _, _ = self.inputs(kind="measured")
        path = self.root / "training-0.json"
        training = json.loads(path.read_text())
        training["records"] = [row for row in training["records"] if row["record_id"] != "decode-5"]
        path.write_text(json.dumps(training))
        result = self.run_plan(plan, fixtures=False)
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["online"][0]["prediction_status"], "FAILED")
        self.assertTrue((self.root / "acceptance/online-0/predicted/logs.txt").exists())

    def test_artifacts_are_exclusive_and_raw_input_bytes_are_retained(self):
        plan, _, _ = self.inputs()
        self.run_plan(plan)
        output = self.root / "acceptance"
        original = (output / "acceptance.json").read_bytes()
        self.assertEqual((output / "step-replica-0/training.json").read_bytes(),
                         (self.root / "training-0.json").read_bytes())
        self.assertEqual((output / "online-0/observation.json").read_bytes(),
                         (self.root / "observation.json").read_bytes())
        with self.assertRaises(SimulationError) as caught:
            self.run_plan(plan)
        self.assertEqual(caught.exception.code, "OUTPUT_EXISTS")
        self.assertEqual((output / "acceptance.json").read_bytes(), original)

    def test_audit_only_is_explicitly_insufficient_and_does_not_invent_inputs(self):
        self.write("audit.json", {"status": "COMPLETED", "canonical_step_evidence": False})
        result = self.run_plan({"schema_version": 1, "evidence_audit": "audit.json"})
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(result["validation_scope"]["dimensions"], [])
        self.assertFalse((self.root / "acceptance/step-replica-0").exists())

    def test_invalid_thresholds_and_unknown_plan_fields_do_not_run(self):
        plan, _, _ = self.inputs()
        invalid = []
        for name, value in (("min_online_requests", 0), ("min_online_repeats", True),
                            ("min_holdout_coverage", 0), ("min_holdout_coverage", 1.1),
                            ("max_step_mape_percent", float("nan")), ("require_sla_agreement", 1)):
            current = deepcopy(plan)
            current["thresholds"][name] = value
            invalid.append(current)
        current = deepcopy(plan)
        current["thresholds"]["max_online_relative_error"] = {"unknown": 1}
        invalid.append(current)
        invalid.append({**plan, "ignore_bad_data": True})
        for index, current in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(SimulationError):
                self.run_plan(current, name=f"bad-{index}")


if __name__ == "__main__":
    unittest.main()
