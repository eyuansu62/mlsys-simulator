"""Joint constraints retain physical ranks, sample validity and explicit time bases."""

from copy import deepcopy
import math
import unittest

from mlsys_sim.constraints import (
    evaluate_dag_constraints, evaluate_simulation_constraints, validate_constraints,
)
from mlsys_sim.engine import simulate
from mlsys_sim.schema import CalibrationSpec, EvaluationSpec, Experiment, RequestSpec, SimulationError


def run(output=4, window=20):
    exp = Experiment(workload=(RequestSpec("r", 0, 16, output),),
                     calibration=CalibrationSpec(backend="synthetic", prefill_ms=10, decode_ms=5),
                     evaluation=EvaluationSpec(measurement_window_ms=window, max_drain_ms=100))
    return simulate(exp)


def spec(**options):
    return {"schema_version": 1, "min_completed_samples": 1, **options}


def dag():
    rows = []
    for name, model, arrival, first, end in (("a", "large", 0, 10, 20), ("b", "small", 20, 25, 30)):
        rows.append({"node_id": name, "request_id": name, "model_id": model, "status": "COMPLETED",
                     "ready_ms": arrival, "arrival_ms": arrival, "completion_ms": end,
                     "first_token_ms": first, "ttft_ms": first - arrival, "tpot_ms": end - first,
                     "end_to_end_ms": end - arrival, "dependency_wait_ms": arrival,
                     "token_times_ms": [first, end], "output_tokens_actual": 2})
    counts = {"total": 2, "completed": 2, "failed": 0, "timed_out": 0, "unfinished": 0, "blocked_dependency": 0}
    ranks = [{"model_id": model, "device_id": model, "stage_index": 0, "tp_rank": 0, "replica_index": 0,
              "device_memory_bytes": 100, "budget_peak_bytes": 50, "physical_peak_bytes": 40}
             for model in ("large", "small")]
    return {"kind": "dag", "status": "COMPLETED", "counts": counts, "nodes": rows, "devices": ranks,
            "nodes_completed": True, "resources_drained": True, "makespan_ms": 30, "elapsed_ms": 40,
            "pools": [{"model_id": model, "result": {"status": "COMPLETED", "counts": {
                **counts, "total": 1, "completed": 1}, "memory": [rank]}, "requests": [row]}
                      for model, row, rank in zip(("large", "small"), rows, ranks)]}


class ConstraintSchemaTests(unittest.TestCase):
    def test_defaults_strict_schema_and_no_mutation(self):
        original = {"schema_version": 1, "memory": {}}
        before = deepcopy(original)
        normalized = validate_constraints(original)
        self.assertEqual(normalized["min_completed_samples"], 100)
        self.assertEqual(normalized["memory"]["max_budget_fraction"], 1)
        self.assertEqual(original, before)

    def test_invalid_schema_thresholds_and_unknown_keys(self):
        invalid = [None, [], {}, {"schema_version": True, "memory": {}}, {"schema_version": 1},
                   spec(min_completed_samples=True, memory={}), spec(min_completed_samples=0, memory={}),
                   spec(memory={"max_budget_fraction": 1.1}), spec(memory={"max_budget_fraction": math.nan}),
                   spec(memory={"max_budget_bytes_per_rank": 2.0}), spec(memory={"max_budget_bytes_per_rank": True}),
                   spec(memory={"max_budget_bytes_per_rank": 0}), spec(latency={}), spec(latency={"p99_e2e_ms": 0}),
                   spec(latency={"p99_e2e_ms": math.inf}), spec(latency={"p99_e2e_ms": 10**1000}),
                   spec(latency={"p90_e2e_ms": 20}), spec(throughput={"min_requests_per_s": 1}),
                   spec(throughput={"basis": "elapsed"}), spec(throughput={"basis": "cohort", "min_requests_per_s": 1}),
                   spec(throughput={"basis": "elapsed", "min_output_tokens_per_s": -1}),
                   spec(task={"max_makespan_ms": False}), spec(task={"max_makespan_ms": 1, "extra": 2}),
                   spec(memory={}, extra=2)]
        for value in invalid:
            with self.subTest(value=str(value)[:100]), self.assertRaises(SimulationError) as raised:
                validate_constraints(value)
            self.assertEqual(raised.exception.code, "INVALID_CONSTRAINTS")


class SimulationConstraintTests(unittest.TestCase):
    def test_joint_window_constraints_count_actual_emits_before_completion(self):
        observation = run()
        before = deepcopy(observation)
        result = evaluate_simulation_constraints(observation, spec(
            memory={}, latency={"p99_e2e_ms": 25, "p99_ttft_ms": 10, "p99_tpot_ms": 5},
            throughput={"basis": "measurement_window", "min_requests_per_s": 0, "min_output_tokens_per_s": 150}))
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["metrics"]["window_requests_per_s"], 0)
        self.assertEqual(result["metrics"]["window_output_tokens_per_s"], 150)
        self.assertEqual(result["metrics"]["elapsed_output_tokens_per_s"], 160)
        self.assertEqual(observation, before)
        self.assertTrue(all(check["normalized_headroom"] is not None for check in result["limiting_constraints"]))

    def test_explicit_throughput_window_fallback_is_not_elapsed_fallback(self):
        observation = run()
        observation.result.pop("measurement_contract")
        constraints = spec(throughput={"basis": "measurement_window", "min_output_tokens_per_s": 150})
        self.assertEqual(evaluate_simulation_constraints(observation, constraints)["verdict"], "PASS")
        observation.result["throughput"]["window_rule"] = "zero_to_simulation_end"
        self.assertEqual(evaluate_simulation_constraints(observation, constraints)["verdict"], "UNKNOWN")

    def test_minimum_throughput_is_independent_of_drain_inflated_cohort(self):
        observation = run()
        constraints = spec(throughput={"basis": "measurement_window", "min_requests_per_s": 40})
        self.assertEqual(observation.result["throughput"]["requests_per_s"], 50)
        assessment = evaluate_simulation_constraints(observation, constraints)
        self.assertEqual(assessment["verdict"], "FAIL")
        self.assertEqual(assessment["metrics"]["throughput_requests_per_s"], 0)

    def test_missing_token_timestamps_cannot_pass_from_final_token_count(self):
        observation = run()
        observation.requests[0].pop("token_times_ms")
        constraints = spec(throughput={"basis": "elapsed", "min_output_tokens_per_s": 1})
        self.assertEqual(evaluate_simulation_constraints(observation, constraints)["verdict"], "UNKNOWN")

    def test_each_rank_budget_is_checked_without_double_counting_logical_kv(self):
        observation = run()
        observation.result["memory"] = [
            {"device_id": "small", "device_memory_bytes": 100, "budget_peak_bytes": 90,
             "kv_used_bytes": 1000, "physical_peak_bytes": 80},
            {"device_id": "large", "device_memory_bytes": 1000, "budget_peak_bytes": 100,
             "kv_used_bytes": 1000, "physical_peak_bytes": 80}]
        self.assertEqual(evaluate_simulation_constraints(observation, spec(memory={}))["verdict"], "PASS")
        result = evaluate_simulation_constraints(observation, spec(memory={"max_budget_fraction": .8}))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["limiting_constraints"][0]["scope"]["device_id"], "small")
        result = evaluate_simulation_constraints(observation, spec(memory={"max_budget_bytes_per_rank": 89}))
        self.assertEqual(result["verdict"], "FAIL")

    def test_integer_byte_comparison_preserves_one_byte_over_2_to_53(self):
        observation = run()
        observation.result["memory"] = [{"device_id": "big", "device_memory_bytes": 2**53,
                                          "budget_peak_bytes": 2**53 + 1}]
        result = evaluate_simulation_constraints(observation, spec(memory={}))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["limiting_constraints"][0]["actual"], 2**53 + 1)
        self.assertLess(result["limiting_constraints"][0]["normalized_headroom"], 0)
        observation.result["memory"] = [{"device_id": "decimal", "device_memory_bytes": 10,
                                          "budget_peak_bytes": 3}]
        self.assertEqual(evaluate_simulation_constraints(observation, spec(memory={"max_budget_fraction": .3}))
                         ["verdict"], "PASS")

    def test_failed_timedout_and_unfinished_never_pass(self):
        for status in ("FAILED", "TIMED_OUT", "UNFINISHED"):
            observation = run()
            observation.requests[0]["status"] = status
            observation.result["counts"]["completed"] = 0
            observation.result["counts"][status.lower()] = 1
            with self.subTest(status=status):
                self.assertEqual(evaluate_simulation_constraints(observation, spec(memory={}))["verdict"], "FAIL")

    def test_empty_low_sample_missing_metric_and_missing_rank_are_unknown(self):
        observation = run(output=1)
        self.assertEqual(evaluate_simulation_constraints(observation, {"schema_version": 1, "memory": {}})
                         ["verdict"], "UNKNOWN")
        self.assertEqual(evaluate_simulation_constraints(observation, spec(latency={"p99_tpot_ms": 10}))
                         ["verdict"], "UNKNOWN")
        observation.result["memory"] = []
        self.assertEqual(evaluate_simulation_constraints(observation, spec(memory={}))["verdict"], "UNKNOWN")
        empty = simulate(Experiment())
        self.assertEqual(evaluate_simulation_constraints(empty, spec(memory={}))["verdict"], "UNKNOWN")

    def test_invalid_requested_latency_is_not_dropped_when_remaining_samples_suffice(self):
        for invalid in (math.nan, math.inf, -1, True):
            observation = run()
            second = deepcopy(observation.requests[0])
            second["request_id"] = "invalid"
            second["end_to_end_ms"] = invalid
            observation.requests.append(second)
            observation.result["counts"]["total"] = observation.result["counts"]["completed"] = 2
            with self.subTest(invalid=invalid):
                result = evaluate_simulation_constraints(observation, spec(latency={"p99_e2e_ms": 30}))
                self.assertEqual(result["verdict"], "UNKNOWN")
                self.assertTrue(any(check["reason"] == "invalid_latency_evidence" for check in result["checks"]))

    def test_missing_tpot_is_legitimate_only_for_at_most_one_actual_token(self):
        observation = run()
        second = deepcopy(observation.requests[0])
        second["request_id"] = "missing-tpot"
        second["tpot_ms"] = None
        observation.requests.append(second)
        observation.result["counts"]["total"] = observation.result["counts"]["completed"] = 2
        constraints = spec(latency={"p99_tpot_ms": 5})
        self.assertEqual(evaluate_simulation_constraints(observation, constraints)["verdict"], "UNKNOWN")
        observation.requests[-1] = deepcopy(run(output=1).requests[0])
        observation.requests[-1]["request_id"] = "single-token"
        self.assertEqual(evaluate_simulation_constraints(observation, constraints)["verdict"], "PASS")

    def test_task_constraint_does_not_synthesize_ordinary_makespan(self):
        self.assertEqual(evaluate_simulation_constraints(run(), spec(task={"max_makespan_ms": 100}))
                         ["verdict"], "UNKNOWN")


class DAGConstraintTests(unittest.TestCase):
    def test_shared_elapsed_pool_throughput_and_task_makespan_excludes_drain(self):
        observation = dag()
        constraints = spec(task={"max_makespan_ms": 35},
                           throughput={"basis": "elapsed", "min_requests_per_s": 50})
        result = evaluate_dag_constraints(observation, constraints, pool_specs={
            "small": spec(throughput={"basis": "elapsed", "min_requests_per_s": 25})})
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["metrics"]["throughput_requests_per_s"], 50)
        self.assertEqual(result["pools"]["small"]["metrics"]["throughput_requests_per_s"], 25)
        self.assertEqual(result["metric_contract"]["request_unit"], "dag_node_request")
        self.assertEqual(result["metrics"]["makespan_ms"], 30)
        self.assertEqual(result["metrics"]["elapsed_ms"], 40)
        task_rates = evaluate_dag_constraints(observation, spec(
            throughput={"basis": "task_makespan", "min_requests_per_s": 60}))
        self.assertEqual(task_rates["verdict"], "PASS")
        self.assertAlmostEqual(task_rates["metrics"]["throughput_requests_per_s"], 200 / 3)

    def test_node_latency_excludes_dependency_wait_and_per_pool_failure_cannot_hide(self):
        observation = dag()
        result = evaluate_dag_constraints(observation, spec(latency={"p99_e2e_ms": 20}), pool_specs={
            "large": spec(latency={"p99_e2e_ms": 15})})
        self.assertEqual(result["overall_verdict"], "PASS")
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["pools"]["large"]["metrics"]["end_to_end_p99_ms"], 20)
        self.assertEqual(result["pools"]["large"]["metric_contract"]["latency_origin"],
                         "node_ready_arrival_not_dependency_release")
        self.assertAlmostEqual(result["metrics"]["end_to_end_p99_ms"], 19.9)

    def test_blocked_node_undrained_state_unknown_pool_and_missing_task_horizon(self):
        observation = dag()
        observation["resources_drained"] = False
        self.assertEqual(evaluate_dag_constraints(observation, spec(memory={}))["verdict"], "FAIL")
        self.assertIsNone(evaluate_dag_constraints(observation, spec(memory={}))
                          ["metrics"]["task_makespan_requests_per_s"])
        observation = dag()
        observation["nodes"][1]["status"] = "BLOCKED_DEPENDENCY"
        self.assertEqual(evaluate_dag_constraints(observation, spec(memory={}))["verdict"], "FAIL")
        with self.assertRaises(SimulationError):
            evaluate_dag_constraints(dag(), spec(memory={}), pool_specs={"unknown": spec(memory={})})
        self.assertEqual(evaluate_dag_constraints(dag(), spec(
            throughput={"basis": "measurement_window", "min_requests_per_s": 0}))["verdict"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
