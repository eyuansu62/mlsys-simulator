"""Fresh Runtime capacity evidence, typed allocator OOM and censoring contracts."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from mlsys_sim import CalibrationSpec, Experiment, MemoryConfig, RequestSpec, SimulationError, StrategySpec
from mlsys_sim.capacity import (PROTOCOL, capacity_experiment, collect_memory_capacity, config_sha256,
    load_capacity_dataset, normalize_capacity_plan, validate_capacity_dataset, validate_memory_capacity)
from mlsys_sim.capacity_worker import run_capacity_trial
from mlsys_sim.hardware import CollectionError
from mlsys_sim.pp_benchmark import _run_bounded


ROOT = Path(__file__).resolve().parents[1]


class FakeOOM(RuntimeError):
    pass


FakeOOM.__module__ = "torch"
FakeOOM.__qualname__ = "OutOfMemoryError"


def experiment():
    return Experiment(workload=(RequestSpec("r", 0.0, 4, 1),),
        strategy=StrategySpec(max_batch_tokens=16, memory=MemoryConfig(kv_block_tokens=1)),
        calibration=CalibrationSpec(backend="synthetic", runtime_name="fixture-runtime", runtime_version="1.0"))


class FakeCuda:
    OutOfMemoryError = FakeOOM
    def __init__(self, exp):
        self.exp = exp
        self.ooms = {index: 0 for index in range(len(exp.infra.devices))}
        self.syncs = []
    def is_available(self):
        return True
    def device_count(self):
        return len(self.exp.infra.devices)
    def get_device_properties(self, index):
        device = self.exp.infra.devices[index]
        return SimpleNamespace(name=device.name, total_memory=device.memory_bytes)
    def reset_peak_memory_stats(self, index):
        pass
    def memory_stats(self, index):
        return {"num_ooms": self.ooms[index]}
    def memory_allocated(self, index):
        return 100
    def memory_reserved(self, index):
        return 200
    def max_memory_allocated(self, index):
        return 150
    def max_memory_reserved(self, index):
        return 200
    def mem_get_info(self, index):
        total = self.exp.infra.devices[index].memory_bytes
        return total - 200, total
    def synchronize(self, index):
        self.syncs.append(index)


class FakeTorch:
    __version__ = "fixture-torch"
    version = SimpleNamespace(hip="fixture-hip")
    def __init__(self, exp):
        self.cuda = FakeCuda(exp)


class PoolTensor:
    def __init__(self, index, size, pointer):
        self.device = SimpleNamespace(type="cuda", index=index)
        self.size, self.pointer = size, pointer
    def untyped_storage(self):
        return self
    def nbytes(self):
        return self.size
    def data_ptr(self):
        return self.pointer


def worker_config(exp, *, axis="input_tokens", value=4, point=0, repeat=0, identifier="worker-0"):
    return {"schema_version": 1, "kind": "memory_capacity_trial_config", "protocol": PROTOCOL,
        "session_id": "capacity-unit-session", "worker_id": identifier, "point_index": point, "repeat": repeat,
        "axis": {"name": axis, "value": value}, "experiment": asdict(exp), "effective_config_sha256": config_sha256(exp),
        "adapter_entry_point": "fixture:factory", "source_reference": "CPU source-shaped capacity fixture; no GPU evidence",
        "profile_input_hashes": []}


def configured_factory(config, torch, *, outcome="COMPLETED", phase="run", oom_counter=True, close_log=None,
                       kv_proof=True, remote=False):
    def factory(exp):
        def failure():
            if outcome == "OOM":
                if oom_counter:
                    torch.cuda.ooms[0] += 1
                raise FakeOOM("CPU injected typed allocator exception")
            if outcome == "STRING_OOM":
                raise RuntimeError("CUDA out of memory")
            if outcome == "PEER":
                raise ChildProcessError("peer failed")
            if outcome == "TIMEOUT":
                raise TimeoutError("Runtime timed out")
            if outcome == "ASSERTION":
                class WorkspaceManager:
                    def lock(self):
                        assert False, "workspace is locked"
                WorkspaceManager().lock()
        if phase == "load":
            failure()

        class Adapter:
            def metadata(self):
                return {"isolation": "remote" if remote else "owned_process",
                    "effective_config_sha256": config["effective_config_sha256"], "applied_axis": config["axis"],
                    "runtime_name": exp.calibration.runtime_name, "runtime_version": exp.calibration.runtime_version,
                    "execution_scope": "complete_declared_workload", "fixed_kv_pool_bytes": exp.strategy.memory.kv_pool_bytes,
                    "owner_pid": os.getpid()}
            def prepare(self):
                if phase == "prepare":
                    failure()
            def run(self):
                if phase == "run":
                    failure()
                return {"status": "ADMISSION_REJECTED" if outcome == "ADMISSION" else "COMPLETED",
                    "reason": "KV admission refused" if outcome == "ADMISSION" else None,
                    "completed_request_ids": [request.request_id for request in exp.workload]}
            def close(self):
                if close_log is not None:
                    close_log.append(config["worker_id"])
                if phase == "cleanup":
                    failure()
            def fixed_kv_pool_allocations(self):
                if not kv_proof:
                    return {}
                # Alias appears twice but may only be charged once.
                tensor = PoolTensor(0, exp.strategy.memory.kv_pool_bytes, 123)
                return {0: [tensor, tensor]}
            def capacity_oom_evidence(self):
                return {"resource": "kv_pool", "phase": phase, "effective_config_sha256": config["effective_config_sha256"],
                    "attempts": [{"device_index": 0, "shape": [exp.strategy.memory.kv_pool_bytes],
                        "element_size": 1, "target_pool_bytes": exp.strategy.memory.kv_pool_bytes, "operation": "torch.empty"}]}
        return Adapter()
    return factory


def fixture_dataset(values=(4, 8), *, outcomes=None, repeats=2, axis="input_tokens", base=None):
    base = experiment() if base is None else base
    trials = []
    for point, value in enumerate(values):
        effective = capacity_experiment(base, axis, value)
        for repeat in range(repeats):
            config = worker_config(effective, axis=axis, value=value, point=point, repeat=repeat,
                                   identifier=f"worker-{point}-{repeat}")
            current = outcomes[point][repeat] if outcomes else "COMPLETED" if point == 0 else "OOM"
            torch = FakeTorch(effective)
            kwargs = current if isinstance(current, dict) else {"outcome": current}
            observation = run_capacity_trial(config, torch_module=torch,
                factory=configured_factory(config, torch, **kwargs))
            trials.append({"point_index": point, "repeat": repeat, "worker_id": config["worker_id"],
                "axis_name": axis, "axis_value": value, "effective_experiment": asdict(effective),
                "effective_config_sha256": config["effective_config_sha256"], "observation": observation, "prediction": {}})
    return {"schema_version": 1, "kind": "memory_capacity_measurements", "protocol": PROTOCOL,
        "dataset_id": "capacity-fixture", "source": {"kind": "test_fixture", "reference": "CPU fixtures, no GPU"},
        "session_id": "capacity-unit-session", "base_experiment": asdict(base), "axis": {"name": axis, "values": list(values)},
        "repeats": repeats, "profile_inputs": [], "trials": trials}


class CapacityTests(unittest.TestCase):
    def test_axes_change_only_declared_cohort_input_or_pool(self):
        base = experiment()
        counted = capacity_experiment(base, "request_count", 3)
        self.assertEqual(len(counted.workload), 3)
        self.assertEqual(len({row.request_id for row in counted.workload}), 3)
        self.assertEqual(counted.strategy, base.strategy)
        self.assertTrue(all(row.arrival_ms == base.workload[0].arrival_ms for row in counted.workload))
        token_base = replace(base, workload=(replace(base.workload[0], prompt_token_ids=(1, 2, 3, 4)),))
        self.assertEqual(capacity_experiment(token_base, "input_tokens", 2).workload[0].prompt_token_ids, (1, 2))
        with self.assertRaises(SimulationError):
            capacity_experiment(token_base, "input_tokens", 5)
        pool = capacity_experiment(base, "kv_pool_bytes", 4096)
        self.assertEqual(pool.workload, base.workload)
        self.assertEqual(pool.strategy.memory.kv_pool_bytes, 4096)
        self.assertEqual(pool.strategy.max_batch_tokens, base.strategy.max_batch_tokens)

    def test_plan_exact_types_and_total_cohort_bound_are_checked(self):
        base = experiment()
        plan = {"schema_version": 1, "dataset_id": "test", "axis": {"name": "input_tokens", "values": [4, 8]}}
        for mutate in (lambda p: p["axis"].update(values=[True]), lambda p: p["axis"].update(values=[4, 4]),
                       lambda p: p.update(repeats=True), lambda p: p.update(timeout_s=float("nan")),
                       lambda p: p.update(max_output_bytes=16 * 1024 * 1024 + 1)):
            changed = deepcopy(plan)
            mutate(changed)
            with self.assertRaises(SimulationError):
                normalize_capacity_plan(base, changed)
        plan["axis"] = {"name": "request_count", "values": [100000, 99999]}
        with self.assertRaises(SimulationError):
            normalize_capacity_plan(base, plan)

    def test_typed_allocator_oom_produces_observed_bracket_not_exact_threshold(self):
        data = fixture_dataset()
        result = validate_memory_capacity(data, allow_test_fixture=True)
        self.assertEqual(result["threshold_status"], "BRACKETED")
        self.assertEqual(result["observed_brackets"], [{"success_value": 4, "oom_value": 8, "resolution": 4, "exact_threshold": None}])
        self.assertEqual(result["summary"]["typed_oom_count"], 2)
        self.assertIsNone(result["exact_threshold"])
        self.assertFalse(result["hardware_validated"])
        with self.assertRaises(SimulationError) as raised:
            validate_memory_capacity(data)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")

    def test_manual_typed_oom_missing_allocator_counter_and_cleanup_oom_are_censored(self):
        for kwargs in ({"outcome": "OOM", "oom_counter": False}, {"outcome": "OOM", "phase": "cleanup"},
                       {"outcome": "OOM", "phase": "load"}):
            data = fixture_dataset((4,), outcomes=[[kwargs]], repeats=1)
            result = validate_memory_capacity(data, allow_test_fixture=True)
            observed = result["trials"][0]["observation"]
            self.assertEqual(observed["observed_status"], "OOM")
            self.assertTrue(observed["typed_oom"])
            self.assertEqual(observed["status"], "ERROR")
            self.assertFalse(observed["eligible_for_bracket"])
            self.assertEqual(result["threshold_status"], "CENSORED")

    def test_runtime_text_assertion_peer_timeout_and_admission_never_become_oom(self):
        cases = (("STRING_OOM", "ERROR", "RUNTIME_ERROR"), ("ASSERTION", "ERROR", "WORKSPACE_MANAGER_ASSERTION"),
                 ("PEER", "ERROR", "PEER_FAILURE"), ("TIMEOUT", "TIMEOUT", "RUNTIME_TIMEOUT"),
                 ("ADMISSION", "ADMISSION_REJECTED", "ADMISSION_REJECTED"))
        for outcome, status, code in cases:
            data = fixture_dataset((4,), outcomes=[[outcome]], repeats=1)
            observed = validate_memory_capacity(data, allow_test_fixture=True)["trials"][0]["observation"]
            self.assertEqual((observed["status"], observed["error_code"]), (status, code))
            self.assertFalse(observed["typed_oom"])

    def test_raw_scope_hash_axis_session_and_complete_cohort_proof_are_required(self):
        good = fixture_dataset((4,), repeats=1)
        mutations = [lambda r: r.update(effective_config_sha256="a" * 64), lambda r: r.update(session_id="other"),
            lambda r: r.update(schema_version=True), lambda r: r["metadata"].update(isolation="remote"),
            lambda r: r["metadata"].update(applied_axis={"name": "input_tokens", "value": True}),
            lambda r: r["result"].pop("completed_request_ids"), lambda r: r["result"].update(completed_request_ids=["r", "r"])]
        for mutate in mutations:
            data = deepcopy(good)
            mutate(data["trials"][0]["observation"])
            result = validate_memory_capacity(data, allow_test_fixture=True)
            self.assertEqual(result["trials"][0]["observation"]["status"], "ERROR")
            self.assertFalse(result["trials"][0]["observation"]["eligible_for_bracket"])
        bad = deepcopy(good)
        bad["trials"][0]["effective_experiment"]["strategy"]["max_batch_tokens"] += 1
        with self.assertRaises(SimulationError):
            validate_capacity_dataset(bad, allow_test_fixture=True)

    def test_instability_nonmonotonicity_and_censoring_do_not_fabricate_a_knee(self):
        data = fixture_dataset((8, 4), outcomes=[["COMPLETED", "COMPLETED"], ["OOM", "COMPLETED"]])
        result = validate_memory_capacity(data, allow_test_fixture=True)
        self.assertTrue(result["nonmonotone"])
        self.assertTrue(result["unstable"])
        self.assertEqual(result["threshold_status"], "NONMONOTONE")
        self.assertEqual(result["observed_brackets"], [])
        data = fixture_dataset((4, 8, 16), outcomes=[["COMPLETED"], ["TIMEOUT"], ["OOM"]], repeats=1)
        result = validate_memory_capacity(data, allow_test_fixture=True)
        self.assertEqual(result["threshold_status"], "CENSORED")
        self.assertEqual(result["observed_brackets"], [])
        data = fixture_dataset((4, 8), outcomes=[["COMPLETED"], ["COMPLETED"]], repeats=1)
        self.assertEqual(validate_memory_capacity(data, allow_test_fixture=True)["threshold_status"], "UNBRACKETED")

    def test_kv_pool_requires_actual_storage_or_scoped_failed_allocation_and_counter(self):
        data = fixture_dataset((4096, 8192), axis="kv_pool_bytes", repeats=1,
            outcomes=[["COMPLETED"], [{"outcome": "OOM", "phase": "prepare"}]])
        result = validate_memory_capacity(data, allow_test_fixture=True)
        self.assertEqual(result["threshold_status"], "BRACKETED")
        self.assertEqual(data["trials"][0]["observation"]["kv_pool_bindings"][0]["allocation_count"], 1)
        for target in ("capacity_oom_evidence", "oom_allocator_evidence"):
            bad = deepcopy(data)
            bad["trials"][1]["observation"][target] = None
            report = validate_memory_capacity(bad, allow_test_fixture=True)
            self.assertEqual(report["summary"]["typed_oom_count"], 0)
        missing = fixture_dataset((4096,), axis="kv_pool_bytes", repeats=1,
            outcomes=[[{"outcome": "COMPLETED", "kv_proof": False}]])
        self.assertEqual(validate_memory_capacity(missing, allow_test_fixture=True)["points"][0]["classification"], "CENSORED")

    def test_logical_pool_admission_is_not_a_physical_oom_prediction(self):
        tiny_pool = replace(experiment(), strategy=replace(experiment().strategy,
            memory=replace(experiment().strategy.memory, kv_pool_bytes=1)))
        data = fixture_dataset((4,), repeats=1, base=tiny_pool)
        result = validate_memory_capacity(data, allow_test_fixture=True)
        self.assertEqual(result["trials"][0]["prediction"]["category"], "ADMISSION_LIMIT")
        self.assertEqual(result["physical_confusion"]["excluded_admission_domain_or_censored"], 1)
        small_device = replace(experiment(), infra=replace(experiment().infra,
            devices=(replace(experiment().infra.devices[0], memory_bytes=1024 * 1024),)))
        data = fixture_dataset((4,), repeats=1, base=small_device)
        result = validate_memory_capacity(data, allow_test_fixture=True)
        self.assertEqual(result["trials"][0]["prediction"]["category"], "PHYSICAL_INFEASIBLE")
        self.assertEqual(result["physical_confusion"]["physical_infeasible_but_completed"], 1)

    def test_fresh_collector_trials_recover_after_oom_and_enforce_complete_budget(self):
        identifiers, closed = [], []
        def runner(argv, **kwargs):
            config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
            identifiers.append(config["worker_id"])
            exp = experiment_from_config(config)
            torch = FakeTorch(exp)
            outcome = "OOM" if config["axis"]["value"] == 4 else "COMPLETED"
            capture = run_capacity_trial(config, torch_module=torch,
                factory=configured_factory(config, torch, outcome=outcome, close_log=closed))
            return {"returncode": 0, "timed_out": False, "output_truncated": False, "rank_captures": [capture]}
        plan = {"schema_version": 1, "dataset_id": "fresh-fixture", "axis": {"name": "input_tokens", "values": [4, 8]}, "repeats": 2}
        bundle = collect_memory_capacity(experiment(), plan, adapter_entry_point="fixture:factory",
                                          allow_test_fixture=True, runner=runner)
        self.assertEqual(len(set(identifiers)), 4)
        self.assertEqual(len(closed), 2)  # OOM runtimes are never reused or closed through unsafe GPU paths.
        self.assertEqual(bundle["report"]["summary"]["completed_count"], 2)
        self.assertTrue(bundle["report"]["nonmonotone"])
        with self.assertRaises(CollectionError) as raised:
            collect_memory_capacity(experiment(), {**plan, "max_output_bytes": 12000},
                adapter_entry_point="fixture:factory", allow_test_fixture=True, runner=runner)
        self.assertEqual(raised.exception.code, "COLLECTION_OUTPUT_LIMIT")
        self.assertLess(len(json.dumps(raised.exception.artifacts)), 12000)

    def test_raw_import_checks_target_and_retains_raw_allocator_without_workspace_claim(self):
        data = fixture_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.json"
            path.write_text(json.dumps(data))
            plan = {"schema_version": 1, "dataset_id": "import", "axis": data["axis"], "repeats": 2,
                "raw_path": str(path), "source": data["source"]}
            bundle = collect_memory_capacity(experiment(), plan, allow_test_fixture=True)
            self.assertEqual(bundle["report"]["threshold_status"], "BRACKETED")
            self.assertEqual(load_capacity_dataset(path, allow_test_fixture=True)["source"]["kind"], "test_fixture")
            changed = replace(experiment(), strategy=replace(experiment().strategy, max_batch_tokens=8))
            with self.assertRaises(SimulationError) as raised:
                collect_memory_capacity(changed, plan, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
            row = bundle["dataset"]["trials"][0]["observation"]["allocator"]["after"][0]
            self.assertIn("peak_reserved_bytes", row)
            self.assertNotIn("workspace_peak_bytes", row)

    def test_process_timeout_and_abnormal_exit_are_censored(self):
        def timeout_runner(argv, **kwargs):
            return _run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], **{**kwargs, "timeout_s": .05})
        plan = {"schema_version": 1, "dataset_id": "timeout", "axis": {"name": "input_tokens", "values": [4]}, "repeats": 1}
        bundle = collect_memory_capacity(experiment(), plan, adapter_entry_point="fixture:factory",
            allow_test_fixture=True, runner=timeout_runner)
        self.assertEqual(bundle["report"]["trials"][0]["observation"]["status"], "TIMEOUT")
        def bad_exit(argv, **kwargs):
            config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
            torch = FakeTorch(experiment_from_config(config))
            capture = run_capacity_trial(config, torch_module=torch, factory=configured_factory(config, torch))
            return {"returncode": 1, "timed_out": False, "rank_captures": [capture]}
        bundle = collect_memory_capacity(experiment(), plan, adapter_entry_point="fixture:factory",
            allow_test_fixture=True, runner=bad_exit)
        self.assertEqual(bundle["report"]["summary"]["completed_count"], 0)
        self.assertEqual(bundle["report"]["trials"][0]["observation"]["error_code"], "WORKER_PROCESS_FAILED")

    def test_source_shaped_oom_process_exits_before_next_fresh_runtime(self):
        script = '''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from test_capacity import FakeTorch, configured_factory, experiment_from_config
from mlsys_sim.capacity_worker import run_capacity_trial
config=json.loads(Path(sys.argv[2]).read_text())
torch=FakeTorch(experiment_from_config(config))
outcome="OOM" if config["axis"]["value"]==4 else "COMPLETED"
capture=run_capacity_trial(config,torch_module=torch,factory=configured_factory(config,torch,outcome=outcome))
(Path(sys.argv[3])/"rank-0.json").write_text(json.dumps(capture))
'''
        def runner(argv, **kwargs):
            config = argv[argv.index("--config") + 1]
            return _run_bounded([sys.executable, "-c", script, str(ROOT / "tests"), config,
                                 str(kwargs["output_dir"])], **kwargs)
        plan = {"schema_version": 1, "dataset_id": "process-fixture", "axis": {"name": "input_tokens", "values": [4, 8]}, "repeats": 2}
        bundle = collect_memory_capacity(experiment(), plan, adapter_entry_point="fixture:factory",
            allow_test_fixture=True, runner=runner)
        observed = [row["observation"] for row in bundle["dataset"]["trials"]]
        self.assertEqual([row["status"] for row in observed], ["OOM", "OOM", "COMPLETED", "COMPLETED"])
        self.assertEqual(len({row["metadata"]["owner_pid"] for row in observed}), 4)
        self.assertEqual([row["cleanup_status"] for row in observed[:2]], ["SKIPPED_AFTER_OOM"] * 2)

    def test_profile_bytes_are_shared_immutably_and_embedded_for_portable_prediction(self):
        from mlsys_sim import calibrate_dataset, profile_identity
        base = experiment()
        profile = calibrate_dataset({"schema_version": 1, "dataset_id": "capacity-profile",
            "identity": profile_identity(base), "source": {"kind": "test_fixture", "reference": "CPU profile fixture"},
            "records": [{"record_id": str(n), "group_id": str(n), "phase": "prefill", "stage_index": 0,
                "query_tokens": [n], "context_lengths": [0], "duration_ms": float(n), "workspace_peak_bytes": None}
                for n in (4, 8)]}, profile_version="capacity-stage-v1", allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as temporary:
            origin = Path(temporary) / "profile.json"
            origin.write_text(json.dumps(profile))
            original = origin.read_bytes()
            configured = replace(base, calibration=replace(base.calibration, backend="profile", profile_path=str(origin),
                profile_version="capacity-stage-v1", allow_test_fixture=True))
            def runner(argv, **kwargs):
                config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
                origin.write_text("original changed after capture")
                exp = experiment_from_config(config)
                self.assertEqual(Path(exp.calibration.profile_path).read_bytes(), original)
                torch = FakeTorch(exp)
                capture = run_capacity_trial(config, torch_module=torch, factory=configured_factory(config, torch))
                return {"returncode": 0, "rank_captures": [capture]}
            plan = {"schema_version": 1, "dataset_id": "profiles", "axis": {"name": "input_tokens", "values": [4, 8]}, "repeats": 1}
            bundle = collect_memory_capacity(configured, plan, adapter_entry_point="fixture:factory",
                allow_test_fixture=True, runner=runner)
            self.assertEqual(bundle["report"]["summary"]["completed_count"], 2)
            self.assertEqual(bundle["dataset"]["profile_inputs"][0]["raw_json"].encode(), original)
            report = validate_memory_capacity(bundle["dataset"], allow_test_fixture=True)
            self.assertTrue(all(row["prediction"]["category"] == "PHYSICAL_FEASIBLE" for row in report["trials"]))


def experiment_from_config(config):
    from mlsys_sim import experiment_from_dict
    return experiment_from_dict(config["experiment"])


if __name__ == "__main__":
    unittest.main()
