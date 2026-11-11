"""Hand-derived heterogeneous fork/join timelines and strict frozen-trace checks."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from mlsys_sim import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment,
                       InfraSpec, MemoryConfig, RequestSpec, SimulationError, StrategySpec,
                       load_experiment, simulate)
from mlsys_sim.dag import (DAGNode, DAGPlan, ModelPool, load_dag_plan,
                           run_dag_candidates, simulate_dag, write_dag_report)


def pool(name, *, prefill=10, replicas=1, tp=1, pp=1, **strategy):
    return ModelPool(name, Experiment(
        infra=InfraSpec(devices=tuple(DeviceSpec(f"{name}-{i}") for i in range(replicas * tp * pp))),
        strategy=StrategySpec(scheduler="fifo", tensor_parallel_size=tp,
                              pipeline_parallel_size=pp, replicas=replicas, **strategy),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=prefill, decode_ms=1)))


def node(name, model="large", parents=(), release=0, output=1, timeout=None, **kwargs):
    return DAGNode(name, model, 4, output, tuple(parents), release, timeout_ms=timeout, **kwargs)


class DAGTimelineTests(unittest.TestCase):
    def test_two_pool_fork_join_uses_actual_parent_completion(self):
        plan = DAGPlan((node("a"), node("b", "small", ("a",), output=2),
                        node("c", "small", ("a",)), node("d", parents=("b", "c"))),
                       (pool("large"), pool("small", prefill=4)))
        result = simulate_dag(plan)
        rows = {row["node_id"]: row for row in result["nodes"]}
        self.assertEqual([rows[name]["completion_ms"] for name in "abcd"], [10, 15, 19, 29])
        self.assertEqual([rows[name]["ready_ms"] for name in "abcd"], [0, 10, 10, 19])
        self.assertEqual(rows["c"]["queue_wait_ms"], 5)
        self.assertEqual(rows["d"]["dependency_wait_ms"], 19)
        self.assertEqual(result["makespan_ms"], 29)
        self.assertEqual(result["critical_path"]["nodes"], ["a", "c", "d"])
        self.assertEqual(result["critical_path"]["node_elapsed_ms"], [10, 9, 10])
        devices = {row["device_id"]: row for row in result["devices"]}
        self.assertAlmostEqual(devices["large-0"]["utilization"], 20 / 29)
        self.assertAlmostEqual(devices["small-0"]["utilization"], 9 / 29)

    def test_heterogeneous_tp2_large_and_two_small_replicas(self):
        plan = DAGPlan((node("a"), node("b", "small", ("a",)), node("c", "small", ("a",)),
                        node("d", parents=("b", "c"))),
                       (pool("large", tp=2), pool("small", prefill=4, replicas=2)))
        result = simulate_dag(plan)
        self.assertEqual([row["completion_ms"] for row in result["nodes"]], [10, 14, 14, 24])
        self.assertEqual([row.get("replica_index") for row in result["nodes"]], [None, 0, 1, None])
        self.assertEqual(len(result["devices"]), 4)
        self.assertTrue(all(row["peak_kv_used_bytes"] > 0 for row in result["devices"]))

    def test_no_dependencies_single_node_equals_ordinary_engine(self):
        for tp, pp, replicas in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 2, 2)):
            with self.subTest(tp=tp, pp=pp, replicas=replicas):
                model = pool("large", tp=tp, pp=pp, replicas=replicas)
                plan = DAGPlan((node("a", release=7, output=3),), (model,))
                result = simulate_dag(plan)
                ordinary = simulate(replace(model.experiment, workload=(RequestSpec("a", 7, 4, 3),)))
                row = result["nodes"][0]
                for field, value in ordinary.requests[0].items():
                    self.assertEqual(row[field], value)
                self.assertEqual(result["makespan_ms"], ordinary.result["simulated_ms"])
                self.assertEqual(result["pools"][0]["result"], ordinary.result)

    def test_completion_before_same_time_release_and_no_future_targets(self):
        plan = DAGPlan((node("parent"), node("child", parents=("parent",)),
                        node("other", release=10)), (pool("large", replicas=2),))
        result = simulate_dag(plan)
        rows = {row["node_id"]: row for row in result["nodes"]}
        self.assertEqual(rows["child"]["arrival_ms"], 10)
        self.assertEqual(rows["other"]["arrival_ms"], 10)
        at_ten = [event for event in result["events"] if event["timestamp_ms"] == 10]
        completed = next(i for i, event in enumerate(at_ten) if event["event"] == "STEP_COMPLETE")
        arrivals = [i for i, event in enumerate(at_ten) if event["event"] == "REQUEST_ROUTED"]
        self.assertTrue(all(completed < index for index in arrivals))
        longer = replace(plan, nodes=tuple(replace(n, output_tokens=100) if n.node_id == "other" else n
                                           for n in plan.nodes))
        second = simulate_dag(longer)
        self.assertEqual([row.get("replica_index") for row in result["nodes"]],
                         [row.get("replica_index") for row in second["nodes"]])

    def test_timeout_from_ready_blocks_descendants_and_drains_pinned_work(self):
        plan = DAGPlan((node("parent"), node("fail", "small", ("parent",), timeout=1),
                        node("blocked", parents=("fail",)), node("healthy", release=20)),
                       (pool("large"), pool("small", prefill=4)))
        result = simulate_dag(plan)
        rows = {row["node_id"]: row for row in result["nodes"]}
        self.assertEqual(rows["fail"]["arrival_ms"], 10)
        self.assertEqual(rows["fail"]["completion_ms"], 11)
        self.assertEqual(rows["blocked"]["status"], "BLOCKED_DEPENDENCY")
        self.assertIsNone(rows["blocked"]["arrival_ms"])
        self.assertEqual(rows["healthy"]["completion_ms"], 30)
        self.assertEqual(result["elapsed_ms"], 30)
        self.assertIsNone(result["makespan_ms"])
        self.assertEqual(result["pools"][1]["result"]["compute_busy_ms"], 4)

    def test_global_cutoff_keeps_partial_tokens_and_memory_ownership(self):
        plan = DAGPlan((node("a"), node("b", "small"), node("child", parents=("a",))),
                       (pool("large"), pool("small")), execution=ExecutionSpec(max_simulated_ms=5))
        result = simulate_dag(plan)
        self.assertEqual(result["error_code"], "SIMULATION_TIME_LIMIT")
        self.assertEqual(result["elapsed_ms"], 5)
        self.assertIsNone(result["makespan_ms"])
        self.assertEqual(result["nodes"][2]["status"], "BLOCKED_DEPENDENCY")
        self.assertTrue(all(row["used_blocks"] == 1 for row in result["devices"]))
        self.assertTrue(all(row["busy_ms"] == 5 for row in result["devices"]))
        self.assertTrue(all(row.get("output_tokens_actual", 0) == 0 for row in result["nodes"]))

    def test_global_event_budget_never_admits_future_nodes(self):
        plan = DAGPlan((node("a"), node("future", release=100)), (pool("large"),),
                       execution=ExecutionSpec(max_events=1))
        result = simulate_dag(plan)
        self.assertEqual(result["events_processed"], 1)
        self.assertEqual(result["error_code"], "SIMULATION_EVENT_LIMIT")
        self.assertIsNone(result["nodes"][1]["arrival_ms"])
        self.assertEqual(result["nodes"][0]["output_tokens_actual"], 0)

    def test_quiet_logging_preserves_metrics_and_queue_wait(self):
        plan = DAGPlan((node("a"), node("b")), (pool("large"),))
        detailed = simulate_dag(plan)
        quiet = simulate_dag(replace(plan, execution=ExecutionSpec(detailed_events=False)))
        self.assertEqual(quiet["nodes"], detailed["nodes"])
        self.assertEqual(quiet["devices"], detailed["devices"])
        self.assertEqual(quiet["makespan_ms"], detailed["makespan_ms"])
        self.assertEqual(quiet["events"], [])

    def test_warmup_isolated_from_measured_cache(self):
        model = pool("large", prefix_cache=True, memory=MemoryConfig(kv_block_tokens=4))
        warmup = RequestSpec("warm", 0, 4, 1, prompt_token_ids=(1, 2, 3, 4))
        model = replace(model, experiment=replace(model.experiment,
                                                  execution=ExecutionSpec(warmup=(warmup,))))
        result = simulate_dag(DAGPlan((node("a", prompt_token_ids=(1, 2, 3, 4)),), (model,)))
        self.assertEqual(result["pools"][0]["result"]["warmup"]["status"], "COMPLETED")
        self.assertEqual(result["nodes"][0]["prefix_cached_tokens"], 0)
        self.assertEqual(result["nodes"][0]["completion_ms"], 10)

    def test_pp_feedback_tail_is_drained_separately_from_task_makespan(self):
        model = pool("large", pp=2, pipeline_feedback_latency_us=10000)
        for cutoff, status in ((11, "UNRESOLVED"), (30, "COMPLETED")):
            with self.subTest(cutoff=cutoff):
                plan = DAGPlan((node("a"),), (model,),
                               execution=ExecutionSpec(max_simulated_ms=cutoff))
                result = simulate_dag(plan)
                self.assertEqual(result["status"], status)
                self.assertTrue(result["nodes_completed"])
                self.assertAlmostEqual(result["makespan_ms"], 10.0050512)
                self.assertEqual(result["resources_drained"], status == "COMPLETED")
                if status == "UNRESOLVED":
                    self.assertEqual(result["elapsed_ms"], 11)
                    self.assertTrue(all(row["used_blocks"] == 1 for row in result["devices"]))
                else:
                    self.assertAlmostEqual(result["elapsed_ms"], 20.0050512)
                mapping = asdict(plan)
                mapping["candidates"] = [{"candidate_id": "candidate", "pools": mapping.pop("pools")}]
                ranked = run_dag_candidates(mapping)
                self.assertEqual(len(ranked["ranking"]), int(status == "COMPLETED"))

    def test_idle_pool_local_cutoff_does_not_fail_finished_requests(self):
        fast = pool("fast", prefill=10)
        fast = replace(fast, experiment=replace(fast.experiment,
                                                execution=ExecutionSpec(max_simulated_ms=15)))
        plan = DAGPlan((node("a", "fast"), node("b", "slow")),
                       (fast, pool("slow", prefill=20)))
        result = simulate_dag(plan)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["makespan_ms"], 20)
        self.assertTrue(all(row["result"]["status"] == "COMPLETED" for row in result["pools"]))
        late = simulate_dag(replace(plan, nodes=(*plan.nodes, node("late", "fast", ("b",)))))
        self.assertEqual(late["status"], "UNRESOLVED")
        self.assertEqual(late["nodes"][-1]["error_code"], "MODEL_POOL_UNAVAILABLE")

    def test_shipped_52_node_three_candidate_trace_completes(self):
        path = Path(__file__).resolve().parents[1] / "examples/course/dag-candidates.plan.json"
        result = run_dag_candidates(path)
        self.assertEqual(len(result["ranking"]), 3)
        self.assertEqual(len({row["trace_sha256"] for row in result["candidates"]}), 1)
        self.assertTrue(all(row["counts"]["completed"] == 52 for row in result["candidates"]))
        self.assertTrue(all(len(row["devices"]) == 4 for row in result["candidates"]))
        self.assertEqual(result["trace_source"], "synthetic_teaching_trace")

    def test_tp_pp_replicas_cache_chunking_swap_and_mixed_batches_reuse_core(self):
        root = Path(__file__).resolve().parents[1]
        exp = load_experiment(root / "examples/v1_4/replicas_pp_mixed_cache_swap.json")
        exp = replace(exp, infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(8))),
                      strategy=replace(exp.strategy, tensor_parallel_size=2,
                                       memory=replace(exp.strategy.memory, kv_pool_bytes=2560)))
        specs = sorted(exp.workload, key=lambda spec: (spec.arrival_ms, spec.request_id))
        nodes = tuple(DAGNode(spec.request_id, "combined", spec.input_tokens, spec.output_tokens,
                              release_ms=spec.arrival_ms, priority=spec.priority, timeout_ms=spec.timeout_ms,
                              prompt_token_ids=spec.prompt_token_ids, cache_namespace=spec.cache_namespace)
                      for spec in specs)
        result = simulate_dag(DAGPlan(nodes, (ModelPool("combined", replace(exp, workload=())),),
                                      execution=exp.execution))
        ordinary = simulate(replace(exp, workload=tuple(specs)))
        self.assertEqual(result["pools"][0]["result"], ordinary.result)
        self.assertEqual(result["pools"][0]["requests"], ordinary.requests)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertGreater(result["pools"][0]["result"]["prefix_cache"]["hit_tokens"], 0)
        self.assertGreater(result["pools"][0]["result"]["batching"]["mixed_batches_completed"], 0)
        self.assertGreater(result["pools"][0]["result"]["swap"]["completed_bytes_by_direction"]["D2H"], 0)
        self.assertEqual(len(result["devices"]), 8)

    def test_profile_pool_preserves_strict_identity_checks(self):
        root = Path(__file__).resolve().parents[1]
        exp = load_experiment(root / "examples/v1/profile_serving.json")
        spec = exp.workload[0]
        model = ModelPool("profiled", replace(exp, workload=()))
        target = DAGNode("a", "profiled", spec.input_tokens, spec.output_tokens)
        result = simulate_dag(DAGPlan((target,), (model,)))
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["pools"][0]["result"]["calibration_source"], "test_fixture_profile")
        mismatched = replace(model, experiment=replace(model.experiment,
            strategy=replace(model.experiment.strategy,
                             model=replace(model.experiment.strategy.model, name="different-model"))))
        with self.assertRaises(SimulationError):
            simulate_dag(DAGPlan((target,), (mismatched,)))


class DAGValidationAndReportTests(unittest.TestCase):
    def mapping(self):
        return asdict(DAGPlan((node("a"), node("b", parents=("a",))), (pool("large"),)))

    def test_unknown_fields_types_cycles_models_and_device_aliases_rejected(self):
        invalid = []
        mapping = self.mapping()
        mapping["unknown"] = 1
        invalid.append(mapping)
        for field, value in (("input_tokens", True), ("release_ms", float("nan")),
                             ("dependencies", ("missing",)), ("model_id", "missing"),
                             ("node_id", "b")):
            mapping = self.mapping()
            mapping["nodes"][0][field] = value
            invalid.append(mapping)
        mapping = self.mapping()
        mapping["nodes"][0]["dependencies"] = ("b",)
        invalid.append(mapping)
        mapping = self.mapping()
        mapping["pools"] = (*mapping["pools"], {**mapping["pools"][0], "model_id": "other"})
        invalid.append(mapping)
        for mapping in invalid:
            with self.subTest(mapping=mapping), self.assertRaises(SimulationError):
                load_dag_plan(mapping)

    def test_candidates_freeze_trace_exclude_failures_from_ranking_and_do_not_mutate(self):
        mapping = self.mapping()
        pools = mapping.pop("pools")
        mapping["candidates"] = [{"candidate_id": "slow", "pools": pools},
                                  {"candidate_id": "fast", "pools": deepcopy(pools)},
                                  {"candidate_id": "fail", "pools": deepcopy(pools)}]
        mapping["candidates"][1]["pools"][0]["experiment"]["calibration"]["prefill_ms"] = 5
        mapping["candidates"][2]["pools"][0]["experiment"]["execution"]["max_simulated_ms"] = 1
        frozen = deepcopy(mapping)
        result = run_dag_candidates(mapping)
        self.assertEqual(mapping, frozen)
        self.assertEqual([row["candidate_id"] for row in result["ranking"]], ["fast", "slow"])
        self.assertEqual(len({row["trace_sha256"] for row in result["candidates"]}), 1)
        self.assertIsNone(result["candidates"][2]["makespan_ms"])

    def test_report_escapes_labels_and_uses_fixed_safe_names(self):
        name = "<script>alert(1)</script>"
        plan = DAGPlan((node(name),), (pool("large"),))
        result = simulate_dag(plan)
        with tempfile.TemporaryDirectory() as temporary:
            path = write_dag_report(result, Path(temporary) / "artifacts")
            svg = (path / "gantt.svg").read_text()
            ET.fromstring(svg)
            self.assertNotIn("<script>", svg)
            saved = json.loads((path / "report.json").read_text())
            self.assertIn("stored_at", saved)
            self.assertIn("prediction_id", saved)
            self.assertTrue((path / "nodes.csv").exists())
            self.assertTrue((path / "devices.csv").exists())
            html = (path / "report.html").read_text()
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("src='gantt.svg'", html)
            self.assertIn("href='replay.plan.json'", html)
            self.assertIn("任务 makespan", html)
            self.assertIn("elapsed", html)
            with self.assertRaises(FileExistsError):
                write_dag_report(result, path)

    def test_candidate_html_uses_numeric_links_and_does_not_rank_partial_results(self):
        mapping = self.mapping()
        mapping["trace_source"] = "synthetic_teaching_trace"
        pools = mapping.pop("pools")
        mapping["candidates"] = [{"candidate_id": "<script>../first</script>", "pools": pools},
                                  {"candidate_id": "partial", "pools": deepcopy(pools)}]
        mapping["candidates"][1]["pools"][0]["experiment"]["execution"]["max_simulated_ms"] = 1
        result = run_dag_candidates(mapping)
        self.assertEqual([row["candidate_id"] for row in result["ranking"]], ["<script>../first</script>"])
        with tempfile.TemporaryDirectory() as temporary:
            path = write_dag_report(result, Path(temporary) / "artifacts")
            html = (path / "report.html").read_text()
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;../first&lt;/script&gt;", html)
            self.assertIn("href='candidate-000/report.html'", html)
            self.assertIn("href='candidate-001/report.html'", html)
            self.assertIn("src='candidate-000/gantt.svg'", html)
            self.assertIn("不是采集的真实 Claude", html)
            self.assertIn("不是 W7900D 端到端真机测量", html)
            self.assertIn("<td>partial</td><td>—</td><td>UNRESOLVED</td>", html)
            self.assertTrue((path / "candidate-001/report.html").exists())

    def test_plan_relative_profile_paths_resolve_without_cwd_dependence(self):
        mapping = self.mapping()
        mapping["pools"][0]["experiment"]["calibration"]["profile_path"] = "profiles/model.json"
        mapping["pools"][0]["experiment"]["calibration"]["backend"] = "profile"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "plan.json"
            source.write_text(json.dumps(mapping))
            plan = load_dag_plan(source)
            self.assertEqual(plan.pools[0].experiment.calibration.profile_path,
                             str((source.parent / "profiles/model.json").resolve()))
            captured = load_dag_plan(mapping, base_dir=source.parent)
            self.assertEqual(captured, plan)

    def test_profile_bytes_are_frozen_hashed_and_replay_without_original_file(self):
        root = Path(__file__).resolve().parents[1]
        exp = load_experiment(root / "examples/v1/profile_serving.json")
        raw = Path(exp.calibration.profile_path).read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.json"
            profile.write_bytes(raw)
            exp = replace(exp, workload=(), calibration=replace(exp.calibration, profile_path=str(profile)))
            plan = DAGPlan((DAGNode("a", "profiled", 16, 3),), (ModelPool("profiled", exp),))
            first = simulate_dag(plan)
            captured = first["pools"][0]["profile_inputs"][0]
            self.assertEqual(captured["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(captured["raw_json"].encode(), raw)
            profile.write_bytes(raw + b"\n")
            second = simulate_dag(plan)
            self.assertNotEqual(first["configuration_sha256"], second["configuration_sha256"])
            self.assertEqual(first["nodes"], second["nodes"])
            output = write_dag_report(first, Path(temporary) / "saved")
            profile.unlink()
            replay = simulate_dag(output / "replay.plan.json")
            self.assertEqual(replay["configuration_sha256"], first["configuration_sha256"])
            self.assertEqual(replay["trace_sha256"], first["trace_sha256"])
            self.assertEqual(replay["nodes"], first["nodes"])
            self.assertEqual((output / "inputs/pool-000/profile-000.json").read_bytes(), raw)

    def test_profile_changed_during_prediction_or_between_candidates_is_rejected(self):
        from mlsys_sim.dag import _simulate_captured
        from mlsys_sim.event_driver import EventDriver
        root = Path(__file__).resolve().parents[1]
        exp = load_experiment(root / "examples/v1/profile_serving.json")
        raw = Path(exp.calibration.profile_path).read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.json"
            profile.write_bytes(raw)
            exp = replace(exp, workload=(), calibration=replace(exp.calibration, profile_path=str(profile)))
            plan = DAGPlan((DAGNode("a", "profiled", 16, 3),), (ModelPool("profiled", exp),))
            original = EventDriver.advance_one
            def changed(driver):
                original(driver)
                profile.write_bytes(raw + b"\n")
            with patch.object(EventDriver, "advance_one", changed), self.assertRaises(SimulationError) as caught:
                simulate_dag(plan)
            self.assertEqual(caught.exception.code, "INPUT_CHANGED")
            profile.write_bytes(raw)
            mapping = asdict(plan)
            pools = mapping.pop("pools")
            mapping["candidates"] = [{"candidate_id": "one", "pools": pools},
                                      {"candidate_id": "two", "pools": deepcopy(pools)}]
            def changed_candidate(value, inputs):
                result = _simulate_captured(value, inputs)
                profile.write_bytes(raw + b"\n")
                return result
            with patch("mlsys_sim.dag._simulate_captured", side_effect=changed_candidate), self.assertRaises(SimulationError) as caught:
                run_dag_candidates(mapping)
            self.assertEqual(caught.exception.code, "INPUT_CHANGED")

    def test_timing_profile_aba_uses_captured_bytes_in_warmup_and_all_candidates(self):
        from mlsys_sim import profiles
        root = Path(__file__).resolve().parents[1]
        exp = load_experiment(root / "examples/v1/profile_serving.json")
        raw = Path(exp.calibration.profile_path).read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "profile.json"
            source.write_bytes(raw)
            exp = replace(exp, workload=(), calibration=replace(exp.calibration, profile_path=str(source)),
                          execution=replace(exp.execution, warmup=(RequestSpec("warm", 0, 16, 3),)))
            plan = DAGPlan((DAGNode("a", "profiled", 16, 3),), (ModelPool("profiled", exp),))
            baseline = simulate_dag(plan)
            original = profiles._read_json
            reads = []
            def transient(path):
                source.write_bytes(raw + b"\n")
                try:
                    payload, actual = original(path)
                    reads.append((str(path), hashlib.sha256(actual).hexdigest()))
                    return payload, actual
                finally:
                    source.write_bytes(raw)
            with patch.object(profiles, "_read_json", side_effect=transient):
                actual = simulate_dag(plan)
            self.assertEqual(actual, baseline)
            self.assertTrue(reads)
            self.assertTrue(all(path != str(source) for path, _ in reads))
            expected = hashlib.sha256(raw).hexdigest()
            self.assertTrue(all(digest == expected for _, digest in reads))
            self.assertEqual(actual["pools"][0]["result"]["profile_provenance"]["sha256"], expected)
            self.assertNotIn("mlsys-dag-inputs-", json.dumps(actual))
            mapping = asdict(plan)
            pools = mapping.pop("pools")
            mapping["candidates"] = [{"candidate_id": "one", "pools": pools},
                                      {"candidate_id": "two", "pools": deepcopy(pools)}]
            baseline_candidates = run_dag_candidates(mapping)
            with patch.object(profiles, "_read_json", side_effect=transient):
                actual_candidates = run_dag_candidates(mapping)
            self.assertEqual(actual_candidates, baseline_candidates)
            self.assertNotIn("mlsys-dag-inputs-", json.dumps(actual_candidates))

    def test_replica_timing_communication_and_workspace_snapshots_survive_source_aba(self):
        from mlsys_sim import calibrate_dataset, pp_communication, profile_identity, profiles, workspace
        from mlsys_sim.layout import replica_experiment
        from tests.test_native_pp_transport import dataset, experiment, record
        from tests.test_workspace import configured, workspace_record

        local = experiment(tp=2, pp=2)
        cluster = replace(local, workload=(),
            infra=replace(local.infra, devices=tuple(replace(device, id=f"replica{replica}-{device.id}")
                for replica in range(2) for device in local.infra.devices)),
            strategy=replace(local.strategy, replicas=2))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_paths, communication_paths, workspace_paths = [], [], []
            fixture = Path(__file__).resolve().parents[1] / "examples/v1_8/communication.profile.fixture.json"
            for replica in range(2):
                local = replica_experiment(cluster, replica)
                stages = [record(local, stage, query=(4,), context=(0,), padded=4, duration=5)
                          for stage in range(2)]
                stage_path = root / f"stage-{replica}.json"
                stage_path.write_text(json.dumps(calibrate_dataset(dataset(local, stages),
                    profile_version="dag-aba-stage", allow_test_fixture=True)))
                stage_paths.append(str(stage_path))
                communication = json.loads(fixture.read_text())
                communication["identity"] = profile_identity(local)
                communication_path = root / f"communication-{replica}.json"
                communication_path.write_text(json.dumps(communication))
                communication_paths.append(str(communication_path))
                rows = [workspace_record(local, stage=stage, queries=(4,), contexts=(0,), peaks=(2, 3),
                                         identifier=f"workspace-{stage}") for stage in range(2)]
                _, workspace_path = configured(local, root, rows, version=f"workspace-{replica}")
                workspace_paths.append(str(workspace_path))
            cluster = replace(cluster, calibration=replace(cluster.calibration, backend="profile",
                profile_version="dag-aba-stage", replica_profile_paths=tuple(stage_paths), allow_test_fixture=True,
                replica_native_pp_communication_profile_paths=tuple(communication_paths),
                replica_workspace_profile_paths=tuple(workspace_paths)))
            plan = DAGPlan((DAGNode("a", "replicas", 4, 1), DAGNode("b", "replicas", 4, 1)),
                           (ModelPool("replicas", cluster),))
            baseline = simulate_dag(plan)
            sources = {path: Path(path).read_bytes() for path in (*stage_paths, *communication_paths, *workspace_paths)}
            observed = []
            def transient(original):
                def read(path, *args, **kwargs):
                    for source, raw in sources.items():
                        Path(source).write_bytes(raw + b"\n")
                    try:
                        payload, raw = original(path, *args, **kwargs)
                        observed.append((str(path), hashlib.sha256(raw).hexdigest()))
                        return payload, raw
                    finally:
                        for source, raw in sources.items():
                            Path(source).write_bytes(raw)
                return read
            with patch.object(profiles, "_read_json", side_effect=transient(profiles._read_json)), \
                    patch.object(pp_communication, "_read", side_effect=transient(pp_communication._read)), \
                    patch.object(workspace, "_read", side_effect=transient(workspace._read)):
                actual = simulate_dag(plan)
            self.assertEqual(actual, baseline)
            self.assertEqual(actual["status"], "COMPLETED")
            captured = actual["pools"][0]["profile_inputs"]
            self.assertEqual({item["field"] for item in captured},
                             {"replica_profile_paths", "replica_native_pp_communication_profile_paths",
                              "replica_workspace_profile_paths"})
            self.assertEqual(len(captured), 6)
            self.assertTrue(observed)
            self.assertTrue(all(path not in sources for path, _ in observed))
            self.assertEqual({digest for _, digest in observed}, {item["sha256"] for item in captured})
            self.assertNotIn("mlsys-dag-inputs-", json.dumps(actual))


if __name__ == "__main__":
    unittest.main()
