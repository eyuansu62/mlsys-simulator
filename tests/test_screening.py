"""Joint candidate feasibility, frozen workloads and exact Profile replay."""

from copy import deepcopy
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment,
                       InfraSpec, RequestSpec, SimulationError, StrategySpec, calibrate_dataset,
                       load_experiment)
from mlsys_sim.screening import run_candidate_screening, write_screening_report


ROOT = Path(__file__).resolve().parents[1]


def experiment(prefill=.5, decode=.1):
    return Experiment(workload=tuple(RequestSpec(str(index), 0, 16, 100 if index == 99 else 1)
                                     for index in range(100)),
                      strategy=StrategySpec(scheduler="fifo"),
                      calibration=CalibrationSpec(backend="synthetic", prefill_ms=prefill, decode_ms=decode),
                      execution=ExecutionSpec(max_events=10000, detailed_events=False))


def simulation_plan():
    baseline = experiment()
    return {"schema_version": 1, "kind": "simulation", "source": "synthetic_teaching_trace",
            "constraints": {"schema_version": 1, "min_completed_samples": 100, "memory": {},
                            "latency": {"p99_e2e_ms": 70},
                            "throughput": {"basis": "elapsed", "min_requests_per_s": 1000}},
            "objective": {"metric": "elapsed_ms", "direction": "min"},
            "candidates": [{"candidate_id": "pass", "experiment": asdict(baseline)},
                           {"candidate_id": "memory", "experiment": asdict(replace(baseline,
                               infra=InfraSpec(devices=(DeviceSpec(memory_bytes=1024 * 1024),))))},
                           {"candidate_id": "latency", "experiment": asdict(experiment(prefill=.8))},
                           {"candidate_id": "throughput", "experiment": asdict(experiment(decode=1))}]}


def load_plan():
    return {"schema_version": 1, "workload": {"schema_version": 1, "seed": 42,
                "arrival": {"process": "periodic", "rate_rps": 20, "duration_ms": 100, "max_requests": 20},
                "tokens": {"input": 16, "output": 3}},
            "rate_grid_rps": [20, 40], "replications": 2, "drain_ms": 1000,
            "min_completed_samples": 1, "slo": {"p99_e2e_ms": 1000}}


def profile_experiment(directory):
    exp = load_experiment(ROOT / "examples/v1/profile_serving.json")
    dataset = json.loads((ROOT / "examples/v1/step_measurements.fixture.json").read_text())
    profile = calibrate_dataset(dataset, profile_version=exp.calibration.profile_version, allow_test_fixture=True)
    path = Path(directory) / "profile.json"
    path.write_text(json.dumps(profile) + "\n")
    return replace(exp, calibration=replace(exp.calibration, profile_path=str(path)),
                   execution=ExecutionSpec(max_events=10000, detailed_events=False))


class ScreeningTests(unittest.TestCase):
    def test_memory_latency_throughput_failures_are_preserved(self):
        plan = simulation_plan()
        frozen = deepcopy(plan)
        result = run_candidate_screening(plan)
        self.assertEqual(plan, frozen)
        self.assertEqual(result["counts"], {"pass": 1, "fail": 3, "unknown": 0})
        self.assertEqual([row["candidate_id"] for row in result["ranking"]], ["pass"])
        by_id = {row["candidate_id"]: row for row in result["candidates"]}
        self.assertEqual(by_id["memory"]["execution"]["error_code"], "STATIC_MEMORY_INFEASIBLE")
        for name, category in (("latency", "latency"), ("throughput", "throughput")):
            checks = by_id[name]["constraint_assessment"]["checks"]
            self.assertTrue(any(check["category"] == category and check["verdict"] == "FAIL" for check in checks))
            other = "throughput" if category == "latency" else "latency"
            self.assertTrue(all(check["verdict"] == "PASS" for check in checks if check["category"] == other))
        self.assertLess(by_id["pass"]["constraint_assessment"]["limiting_constraints"][0]["normalized_headroom"], 1)

    def test_exact_ties_are_retained_without_favorable_id_tiebreak(self):
        plan = simulation_plan()
        first = plan["candidates"][0]
        plan["candidates"] = [{**deepcopy(first), "candidate_id": "z"}, {**deepcopy(first), "candidate_id": "a"}]
        result = run_candidate_screening(plan)
        self.assertEqual([item["candidate_id"] for item in result["ranking"]], ["z", "a"])
        self.assertEqual([item["rank"] for item in result["ranking"]], [1, 1])
        self.assertEqual([item["tie_count"] for item in result["ranking"]], [2, 2])

    def test_missing_objective_missing_window_and_low_samples_remain_unknown(self):
        for mode in ("objective", "window", "samples"):
            with self.subTest(mode=mode):
                plan = simulation_plan()
                plan["candidates"] = plan["candidates"][:1]
                if mode == "objective":
                    del plan["objective"]
                elif mode == "window":
                    plan["objective"] = {"metric": "window_output_tokens_per_s", "direction": "max"}
                else:
                    plan["constraints"]["min_completed_samples"] = 1000
                result = run_candidate_screening(plan)
                self.assertEqual(result["candidates"][0]["verdict"], "UNKNOWN")
                self.assertEqual(result["ranking"], [])

    def test_actual_window_tokens_do_not_count_outputs_after_drain(self):
        exp = replace(experiment(prefill=2, decode=2), workload=(RequestSpec("a", 0, 16, 3),),
                      evaluation=replace(experiment().evaluation, measurement_window_ms=5))
        plan = {"schema_version": 1, "kind": "simulation", "source": "synthetic_teaching_trace",
                "constraints": {"schema_version": 1, "min_completed_samples": 1, "memory": {}},
                "objective": {"metric": "window_output_tokens_per_s", "direction": "max"},
                "candidates": [{"candidate_id": "one", "experiment": asdict(exp)}]}
        report = run_candidate_screening(plan)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["execution"]["requests"][0]["token_times_ms"], [2, 4, 6])
        self.assertEqual(candidate["objective"]["value"], 400)
        self.assertEqual(candidate["constraint_assessment"]["metrics"]["elapsed_output_tokens_per_s"], 500)

    def test_runtime_workspace_failure_is_not_hidden_or_ranked(self):
        plan = simulation_plan()
        candidate = deepcopy(plan["candidates"][0])
        candidate["candidate_id"] = "workspace"
        candidate["experiment"]["strategy"]["memory"]["workspace_bytes_per_token"] = 64 * 1024 * 1024
        plan["candidates"] = [candidate, plan["candidates"][0]]
        result = run_candidate_screening(plan)
        failed = result["candidates"][0]
        self.assertEqual(failed["verdict"], "FAIL")
        self.assertTrue(any(row["error_code"] == "WORKSPACE_BUDGET_EXCEEDED" for row in failed["execution"]["requests"]))
        self.assertEqual([row["candidate_id"] for row in result["ranking"]], ["pass"])

    def test_changed_request_seed_measurement_or_kind_is_rejected_before_running(self):
        for key in ("output", "arrival", "seed", "window", "extra", "kind", "budget"):
            with self.subTest(key=key):
                plan = simulation_plan()
                if key == "output":
                    plan["candidates"][1]["experiment"]["workload"][0]["output_tokens"] += 1
                elif key == "arrival":
                    plan["candidates"][1]["experiment"]["workload"][0]["arrival_ms"] = 1
                elif key == "seed":
                    plan["candidates"][1]["experiment"]["execution"]["seed"] = 99
                elif key == "window":
                    plan["candidates"][1]["experiment"]["evaluation"]["measurement_window_ms"] = 100
                elif key == "extra":
                    plan["candidates"][1]["load_plan"] = load_plan()
                elif key == "budget":
                    for candidate in plan["candidates"]:
                        candidate["experiment"]["execution"]["max_events"] = 2_000_000
                else:
                    plan["kind"] = ["simulation", "dag"]
                with patch("mlsys_sim.screening.simulate") as simulated, self.assertRaises(SimulationError):
                    run_candidate_screening(plan)
                simulated.assert_not_called()

    def test_sweep_all_points_and_replications_are_required(self):
        exp = replace(experiment(), workload=())
        plan = {"schema_version": 1, "kind": "sweep", "source": "synthetic_teaching_trace",
                "constraints": {"schema_version": 1, "min_completed_samples": 1, "memory": {},
                                "latency": {"p99_e2e_ms": 1000},
                                "throughput": {"basis": "measurement_window", "min_requests_per_s": 0}},
                "objective": {"metric": "min_window_output_tokens_per_s", "direction": "max"},
                "candidates": [{"candidate_id": "all", "experiment": asdict(exp), "load_plan": load_plan()},
                               {"candidate_id": "static", "experiment": asdict(replace(exp,
                                   infra=InfraSpec(devices=(DeviceSpec(memory_bytes=1024),)))), "load_plan": load_plan()}]}
        result = run_candidate_screening(plan)
        good, failed = result["candidates"]
        self.assertEqual(good["verdict"], "PASS")
        self.assertEqual(len(good["run_assessments"]), 4)
        self.assertEqual(len(failed["execution"]["planned_runs"]), 4)
        values = [run["assessment"]["metrics"]["window_output_tokens_per_s"] for run in good["run_assessments"]]
        self.assertEqual(good["objective"]["value"], min(values))
        self.assertEqual([row["candidate_id"] for row in result["ranking"]], ["all"])
        bad_plan = deepcopy(plan)
        bad_plan["candidates"][1]["load_plan"]["workload"]["seed"] = 7
        with self.assertRaises(SimulationError):
            run_candidate_screening(bad_plan)

    def test_dag_pool_constraints_and_resource_drain_veto_ranking(self):
        dag = json.loads((ROOT / "examples/course/dag-fork-join.plan.json").read_text())
        dag["execution"]["max_events"] = 10000
        slow = deepcopy(dag)
        slow["pools"][1]["experiment"]["calibration"]["prefill_ms"] = 8
        plan = {"schema_version": 1, "kind": "dag", "source": "synthetic_teaching_trace",
                "constraints": {"schema_version": 1, "min_completed_samples": 1, "memory": {},
                                "task": {"max_makespan_ms": 50}},
                "pool_constraints": {"small": {"schema_version": 1, "min_completed_samples": 1,
                                               "latency": {"p99_e2e_ms": 10}}},
                "objective": {"metric": "makespan_ms", "direction": "min"},
                "candidates": [{"candidate_id": "good", "dag_plan": dag}, {"candidate_id": "pool-latency", "dag_plan": slow}]}
        result = run_candidate_screening(plan)
        self.assertEqual([row["candidate_id"] for row in result["ranking"]], ["good"])
        self.assertEqual(result["candidates"][0]["objective"]["value"], 29)
        self.assertEqual(result["candidates"][1]["constraint_assessment"]["pools"]["small"]["verdict"], "FAIL")
        changed = deepcopy(plan)
        changed["candidates"][1]["dag_plan"]["nodes"][0]["output_tokens"] += 1
        with self.assertRaises(SimulationError):
            run_candidate_screening(changed)

    def test_one_failing_load_point_cannot_be_hidden_by_the_low_load_point(self):
        load = load_plan()
        load["rate_grid_rps"] = [20, 200]
        load["workload"]["arrival"]["max_requests"] = 30
        load["workload"]["tokens"]["output"] = 1
        load["slo"]["p99_e2e_ms"] = 15
        fast = replace(experiment(prefill=1), workload=())
        slow = replace(experiment(prefill=10), workload=())
        plan = {"schema_version": 1, "kind": "sweep", "source": "synthetic_teaching_trace",
                "constraints": {"schema_version": 1, "min_completed_samples": 1,
                                "latency": {"p99_e2e_ms": 15},
                                "throughput": {"basis": "measurement_window", "min_requests_per_s": 0}},
                "objective": {"metric": "min_window_throughput_rps", "direction": "max"},
                "candidates": [{"candidate_id": "fast", "experiment": asdict(fast), "load_plan": deepcopy(load)},
                               {"candidate_id": "slow", "experiment": asdict(slow), "load_plan": deepcopy(load)}]}
        report = run_candidate_screening(plan)
        slow_runs = report["candidates"][1]["run_assessments"]
        self.assertEqual([item["assessment"]["verdict"] for item in slow_runs], ["PASS", "PASS", "FAIL", "FAIL"])
        self.assertEqual([item["candidate_id"] for item in report["ranking"]], ["fast"])

    def test_sweep_limiting_resources_exclude_exact_minimum_sample_gate(self):
        load = load_plan()
        load["workload"]["arrival"].update(rate_rps=100, duration_ms=1000, max_requests=100)
        load["workload"]["tokens"]["output"] = 1
        load.update(rate_grid_rps=[100], replications=1, min_completed_samples=100)
        exp = replace(experiment(prefill=.1), workload=())
        plan = {"schema_version": 1, "kind": "sweep", "source": "synthetic_teaching_trace",
                "constraints": {"schema_version": 1, "min_completed_samples": 100,
                                "memory": {}, "latency": {"p99_e2e_ms": 1000},
                                "throughput": {"basis": "measurement_window", "min_requests_per_s": 1}},
                "objective": {"metric": "min_window_throughput_rps", "direction": "max"},
                "candidates": [{"candidate_id": "one", "experiment": asdict(exp), "load_plan": load}]}
        report = run_candidate_screening(plan)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "PASS")
        assessment = candidate["constraint_assessment"]
        sample = next(item for item in assessment["checks"] if item["id"] == "samples.completed")
        self.assertEqual(sample["normalized_headroom"], 0)
        tight = assessment["limiting_constraints"]
        self.assertEqual(tight[0]["category"], "memory")
        self.assertTrue(all(item["category"] in {"memory", "latency", "throughput", "task"} for item in tight))
        original = candidate["run_assessments"][0]["assessment"]["limiting_constraints"][0]
        self.assertEqual(tight[0]["normalized_headroom"], original["normalized_headroom"])

    def test_corrupt_profile_is_fatal_even_when_an_earlier_candidate_is_static_oom(self):
        with tempfile.TemporaryDirectory() as temporary:
            exp = profile_experiment(temporary)
            path = Path(exp.calibration.profile_path)
            path.write_text(json.dumps({"source": {"kind": "test_fixture", "reference": "invalid Profile"}}))
            oom = replace(exp, infra=InfraSpec(devices=(DeviceSpec(memory_bytes=1024),)),
                calibration=replace(exp.calibration, backend="synthetic", profile_path=None, allow_test_fixture=False))
            plan = {"schema_version": 1, "kind": "simulation", "source": "synthetic_teaching_trace",
                    "constraints": {"schema_version": 1, "min_completed_samples": 1, "memory": {}},
                    "objective": {"metric": "elapsed_ms", "direction": "min"},
                    "candidates": [{"candidate_id": "oom", "experiment": asdict(oom)},
                                   {"candidate_id": "bad-profile", "experiment": asdict(exp)}]}
            with patch("mlsys_sim.screening.simulate") as simulated, self.assertRaises(SimulationError) as caught:
                run_candidate_screening(plan)
            simulated.assert_not_called()
            self.assertNotEqual(caught.exception.code, "STATIC_MEMORY_INFEASIBLE")

    def test_profile_aba_preserves_full_report_for_all_kinds_and_candidates(self):
        from mlsys_sim import profiles
        for kind in ("simulation", "sweep", "dag"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                exp = profile_experiment(temporary)
                source = Path(exp.calibration.profile_path)
                raw = source.read_bytes()
                if kind == "simulation":
                    candidate = {"experiment": asdict(exp)}
                elif kind == "sweep":
                    candidate = {"experiment": asdict(replace(exp, workload=())), "load_plan": load_plan()}
                else:
                    candidate = {"dag_plan": {"nodes": [{"node_id": "a", "model_id": "profiled", "input_tokens": 16, "output_tokens": 3}],
                        "pools": [{"model_id": "profiled", "experiment": asdict(replace(exp, workload=()))}],
                        "execution": {"max_events": 10000}, "trace_source": "synthetic_teaching_trace"}}
                plan = {"schema_version": 1, "kind": kind, "source": "synthetic_teaching_trace",
                        "constraints": {"schema_version": 1, "min_completed_samples": 1, "memory": {}},
                        "objective": {"metric": "min_window_output_tokens_per_s" if kind == "sweep" else "elapsed_ms", "direction": "max"},
                        "candidates": [{"candidate_id": "one", **deepcopy(candidate)}, {"candidate_id": "two", **deepcopy(candidate)}]}
                baseline = run_candidate_screening(plan)
                original = profiles._read_json
                reads = []
                def transient(path):
                    source.write_bytes(raw + b"\n")
                    try:
                        value, actual = original(path)
                        reads.append((str(path), hashlib.sha256(actual).hexdigest()))
                        return value, actual
                    finally:
                        source.write_bytes(raw)
                with patch.object(profiles, "_read_json", side_effect=transient):
                    actual = run_candidate_screening(plan)
                self.assertEqual(actual, baseline)
                self.assertTrue(reads)
                self.assertTrue(all(path != str(source) and digest == hashlib.sha256(raw).hexdigest() for path, digest in reads))
                self.assertNotIn("mlsys-screen-inputs-", json.dumps(actual))

    def test_persistent_profile_change_is_fatal(self):
        from mlsys_sim.screening import simulate as ordinary_simulate
        with tempfile.TemporaryDirectory() as temporary:
            exp = profile_experiment(temporary)
            source = Path(exp.calibration.profile_path)
            raw = source.read_bytes()
            plan = {"schema_version": 1, "kind": "simulation", "source": "synthetic_teaching_trace",
                    "constraints": {"schema_version": 1, "min_completed_samples": 1, "memory": {}},
                    "objective": {"metric": "elapsed_ms", "direction": "min"},
                    "candidates": [{"candidate_id": "one", "experiment": asdict(exp)}]}
            def changed(runtime):
                result = ordinary_simulate(runtime)
                source.write_bytes(raw + b"\n")
                return result
            with patch("mlsys_sim.screening.simulate", side_effect=changed), self.assertRaises(SimulationError) as caught:
                run_candidate_screening(plan)
            self.assertEqual(caught.exception.code, "INPUT_CHANGED")

    def test_report_links_escaping_csv_and_source_deleted_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            exp = profile_experiment(temporary)
            source = Path(exp.calibration.profile_path)
            plan = {"schema_version": 1, "kind": "simulation", "source": "synthetic_teaching_trace",
                    "description": "<script>unsafe text</script>",
                    "constraints": {"schema_version": 1, "min_completed_samples": 1, "memory": {}},
                    "objective": {"metric": "elapsed_ms", "direction": "min"},
                    "candidates": [{"candidate_id": "=<script>../one</script>", "experiment": asdict(exp)}]}
            report = run_candidate_screening(plan)
            output = write_screening_report(report, Path(temporary) / "artifacts")
            html = (output / "report.html").read_text()
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("href='candidate-000.json'", html)
            with (output / "candidates.csv").open() as stream:
                self.assertTrue(next(csv.DictReader(stream))["candidate_id"].startswith("'="))
            self.assertIn("stored_at", json.loads((output / "report.json").read_text()))
            source.unlink()
            replay = run_candidate_screening(output / "replay.plan.json")
            self.assertEqual(replay["configuration_sha256"], report["configuration_sha256"])
            self.assertEqual(replay["workload_sha256"], report["workload_sha256"])
            self.assertEqual(replay["ranking"], report["ranking"])
            self.assertEqual(replay["candidates"][0]["execution"]["requests"], report["candidates"][0]["execution"]["requests"])
            with self.assertRaises(FileExistsError):
                write_screening_report(report, output)

    def test_all_shipped_screening_plans_execute_and_retain_rejections(self):
        expected = {"simulation": (1, 3), "sweep": (1, 1), "dag": (2, 1)}
        for kind, (passed, failed) in expected.items():
            with self.subTest(kind=kind):
                report = run_candidate_screening(ROOT / f"examples/course/screen-{kind}.plan.json")
                self.assertEqual(report["counts"], {"pass": passed, "fail": failed, "unknown": 0})
                self.assertEqual(len(report["ranking"]), passed)
                self.assertEqual(report["source_kind"], "simulated")
                self.assertFalse(report["hardware_validated"])


if __name__ == "__main__":
    unittest.main()
