"""Simple-only P2P policy is explicit, inherited and isolated from old captures."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import DeviceSpec, InfraSpec, SimulationError
from mlsys_sim.config import experiment_from_dict
from mlsys_sim.hardware import CollectionError
from mlsys_sim.pp_benchmark import (COMMUNICATOR_INITIALIZATION, P2P_PROTOCOL_ENVIRONMENT,
    P2P_PROTOCOL_POLICY, PROTOCOL, PROTOCOL_V2, PROTOCOL_V3, collect_pp_communication,
    normalize_pp_benchmark_plan, worker_launch_contract)
from mlsys_sim.pp_benchmark_worker import run_worker

from test_pp_benchmark import FakeTorch, envelope, experiment, plan, rank_captures


def v3_plan():
    return {**plan(), "protocol": PROTOCOL_V3}


def v3_runner(argv, **kwargs):
    config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
    target = experiment_from_dict(config["experiment"])
    captures = rank_captures(target, config["plan"], config["session_id"])
    environment = kwargs["env"]
    actual = {name: environment.get(name) for name in P2P_PROTOCOL_ENVIRONMENT}
    digest = hashlib.sha256(json.dumps({name: value for name, value in sorted(environment.items())
        if name.startswith(("NCCL_", "RCCL_", "TORCH_NCCL_"))},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for capture in captures:
        capture["protocol"] = PROTOCOL_V3
        capture["stack"].update(p2p_protocol_environment=dict(actual), communication_environment_sha256=digest)
    return {**envelope(captures), "protocol": PROTOCOL_V3}


class SimplePolicyCollectionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_v3_is_explicit_and_cannot_override_its_fixed_policy_in_the_plan(self):
        self.assertNotIn("protocol", normalize_pp_benchmark_plan(experiment(), plan()))
        self.assertEqual(normalize_pp_benchmark_plan(experiment(), v3_plan())["protocol"], PROTOCOL_V3)
        for override in ({"p2p_protocol_policy": "different"},
                         {"p2p_protocol_environment": {"NCCL_P2P_LL_THRESHOLD": "4096"}},
                         {"message_bytes": [0, 16]}, {"warmup_iterations": 0},
                         {"iterations": 1000000}, {"max_output_bytes": 32 * 1024 * 1024}):
            with self.subTest(override=override), self.assertRaises(SimulationError):
                normalize_pp_benchmark_plan(experiment(), {**v3_plan(), **override})

    def test_live_child_environment_and_every_metadata_layer_agree(self):
        calls = []
        def inspect(argv, **kwargs):
            calls.append(kwargs["env"])
            return v3_runner(argv, **kwargs)
        selected = {"NCCL_SHM_USE_CUDA_MEMCPY": "1", "NCCL_SHM_MEMCPY_MODE": "2", "NCCL_PROTO": "Simple"}
        with patch.dict(os.environ, {**selected, "TASK_SECRET": "not-in-launch-audit"}):
            result = collect_pp_communication(experiment(), v3_plan(), runner=inspect, allow_test_fixture=True)
            self.assertTrue(all(name not in os.environ for name in P2P_PROTOCOL_ENVIRONMENT))
        self.assertEqual({name: calls[0][name] for name in P2P_PROTOCOL_ENVIRONMENT}, P2P_PROTOCOL_ENVIRONMENT)
        benchmark = result["dataset"]["benchmark"]
        self.assertEqual(benchmark["protocol"], PROTOCOL_V3)
        self.assertEqual(benchmark["communicator_initialization"], COMMUNICATOR_INITIALIZATION)
        self.assertEqual(benchmark["p2p_protocol_policy"], P2P_PROTOCOL_POLICY)
        self.assertEqual(benchmark["p2p_protocol_environment"], P2P_PROTOCOL_ENVIRONMENT)
        contract = benchmark["launch"]["rank_contract"]
        self.assertEqual(contract, worker_launch_contract(experiment(), normalize_pp_benchmark_plan(experiment(), v3_plan())))
        self.assertIs(contract["world_init_device_id"], False)
        self.assertEqual(contract["p2p_protocol_policy"], P2P_PROTOCOL_POLICY)
        self.assertEqual(contract["p2p_protocol_environment"], P2P_PROTOCOL_ENVIRONMENT)
        self.assertEqual(benchmark["launch"]["environment"], {
            **selected, **P2P_PROTOCOL_ENVIRONMENT, "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"})
        self.assertEqual(benchmark["stack"]["p2p_protocol_environment"], P2P_PROTOCOL_ENVIRONMENT)
        raw = result["raw_records"][0]
        self.assertEqual(raw["protocol"], PROTOCOL_V3)
        for capture in raw["rank_captures"]:
            self.assertEqual(capture["protocol"], PROTOCOL_V3)
            self.assertEqual(capture["launch"], contract)
            self.assertEqual(capture["stack"]["p2p_protocol_environment"], P2P_PROTOCOL_ENVIRONMENT)
        self.assertEqual(result["provenance"]["protocol"], PROTOCOL_V3)
        self.assertEqual(result["source"]["kind"], "test_fixture")
        self.assertFalse(result["hardware_validated"])

    def test_conflicting_inherited_thresholds_fail_before_probe_or_launch(self):
        for name in P2P_PROTOCOL_ENVIRONMENT:
            for value in ("4096", "16384", "-1", "", "00", "0x0"):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                    with patch("mlsys_sim.pp_benchmark.probe_rocm") as probe:
                        with self.assertRaises(CollectionError) as raised:
                            collect_pp_communication(experiment(), v3_plan())
                    self.assertEqual(raised.exception.code, "INVALID_COLLECTION_PLAN")
                    probe.assert_not_called()
                    self.assertEqual(os.environ[name], value)
        with patch.dict(os.environ, P2P_PROTOCOL_ENVIRONMENT):
            result = collect_pp_communication(experiment(), v3_plan(), runner=v3_runner, allow_test_fixture=True)
        self.assertEqual(result["dataset"]["benchmark"]["p2p_protocol_environment"], P2P_PROTOCOL_ENVIRONMENT)

    def test_raw_import_does_not_relabel_old_missing_or_conflicting_domain_evidence(self):
        original = collect_pp_communication(experiment(), v3_plan(), runner=v3_runner,
                                            allow_test_fixture=True)["raw_records"][0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v3.json"
            options = {**v3_plan(), "raw_path": path.name,
                "source": {"kind": "test_fixture", "reference": "Retained v3 CPU fixture"}}
            path.write_text(json.dumps(original))
            imported = collect_pp_communication(experiment(), options, base_dir=temporary, allow_test_fixture=True)
            self.assertEqual(imported["raw_records"][0]["raw_input"]["bytes"], path.stat().st_size)
            mutations = [lambda value: value.pop("protocol"), lambda value: value.update(protocol=PROTOCOL_V2),
                lambda value: value["rank_captures"][0].update(protocol=PROTOCOL_V2),
                lambda value: value["rank_captures"][0]["launch"].pop("p2p_protocol_policy"),
                lambda value: value["rank_captures"][0]["launch"].update(p2p_protocol_policy="assumed_simple"),
                lambda value: value["rank_captures"][0]["stack"].pop("p2p_protocol_environment"),
                lambda value: value["rank_captures"][0]["stack"]["p2p_protocol_environment"].update(
                    NCCL_P2P_LL128_THRESHOLD="16384"),
                lambda value: value["rank_captures"][0]["stack"]["p2p_protocol_environment"].update(
                    NCCL_P2P_LL_THRESHOLD=0),
                lambda value: value["rank_captures"][0]["stack"]["p2p_protocol_environment"].update(EXTRA="0")]
            for mutation in mutations:
                changed = deepcopy(original)
                mutation(changed)
                path.write_text(json.dumps(changed))
                with self.subTest(mutation=mutation), self.assertRaises(CollectionError) as raised:
                    collect_pp_communication(experiment(), options, base_dir=temporary, allow_test_fixture=True)
                self.assertTrue(raised.exception.artifacts["raw_records"])
            path.write_text(json.dumps(original))
            for protocol in (PROTOCOL, PROTOCOL_V2):
                with self.subTest(protocol=protocol), self.assertRaises(CollectionError):
                    collect_pp_communication(experiment(), {**options, "protocol": protocol},
                                             base_dir=temporary, allow_test_fixture=True)

    def test_v3_keeps_complete_rank_correctness_source_and_output_bounds(self):
        mutations = [lambda value: value["rank_captures"].pop(),
            lambda value: value["rank_captures"][0].update(session_id="borrowed-session"),
            lambda value: value["rank_captures"][0]["records"][0].update(correctness_errors=1),
            lambda value: value["rank_captures"][0]["records"][0].update(duration_ms=float("nan"))]
        for mutation in mutations:
            def changed(argv, **kwargs):
                value = v3_runner(argv, **kwargs)
                mutation(value)
                return value
            with self.subTest(mutation=mutation), self.assertRaises(CollectionError):
                collect_pp_communication(experiment(), v3_plan(), runner=changed, allow_test_fixture=True)
        with self.assertRaises(SimulationError) as raised:
            collect_pp_communication(experiment(), v3_plan(), runner=v3_runner)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        with self.assertRaises(CollectionError) as raised:
            collect_pp_communication(experiment(), {**v3_plan(), "max_output_bytes": 128},
                                     runner=v3_runner, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "COLLECTION_OUTPUT_LIMIT")


class SimplePolicyWorkerTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, P2P_PROTOCOL_ENVIRONMENT, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def run_rank(self, target, rank, *, corrupt=False, startup_change=False):
        backend = FakeTorch(target, rank)
        calls = []
        initialize = backend.initialize
        def record_initialize(**kwargs):
            calls.append(kwargs)
            initialize(**kwargs)
            if startup_change:
                os.environ["NCCL_P2P_LL_THRESHOLD"] = "4096"
        backend.distributed.init_process_group = record_initialize
        if corrupt:
            receive = backend.receive
            def bad_receive(tensor, **kwargs):
                result = receive(tensor, **kwargs)
                tensor.values[0] = -7
                return result
            backend.distributed.irecv = bad_receive
        ticks = 0
        def clock():
            nonlocal ticks
            ticks += 1
            backend.log.append("start" if ticks % 2 else "end")
            return ticks * 1_000_000
        capture = run_worker({"schema_version": 1, "experiment": asdict(target), "plan": v3_plan(),
            "session_id": "simple-policy-cpu", "rendezvous": "/tmp/unused-simple-policy-store"},
            torch_module=backend, clock_ns=clock,
            binding_provider=lambda: {"runtime_version": "2.30.7", "library_sha256": "a" * 64},
            rank=rank, local_rank=rank)
        return capture, backend, calls

    def test_every_tp_pp_rank_uses_lazy_groups_and_reports_actual_environment(self):
        for tp, pp in ((1, 2), (2, 2), (2, 3)):
            base = experiment(tp)
            target = replace(base, infra=InfraSpec(devices=tuple(DeviceSpec(f"d{i}", name=f"AMD CPU fixture {i}")
                for i in range(tp * pp))), strategy=replace(base.strategy, pipeline_parallel_size=pp))
            groups = []
            for rank in range(tp * pp):
                with self.subTest(tp=tp, pp=pp, rank=rank):
                    capture, backend, calls = self.run_rank(target, rank)
                    self.assertEqual(capture["status"], "COMPLETED", capture)
                    self.assertEqual(capture["source"]["kind"], "test_fixture")
                    self.assertEqual(capture["stack"]["p2p_protocol_environment"], P2P_PROTOCOL_ENVIRONMENT)
                    self.assertNotIn("device_id", calls[0])
                    groups.append([item for item in backend.log if isinstance(item, tuple)])
            self.assertTrue(all(group == groups[0] for group in groups))

    def test_missing_or_conflicting_env_fails_before_torch_import_and_gpu_calls(self):
        for name in P2P_PROTOCOL_ENVIRONMENT:
            for value in (None, "4096"):
                with self.subTest(name=name, value=value), patch.dict(os.environ, dict(P2P_PROTOCOL_ENVIRONMENT), clear=True):
                    if value is None:
                        os.environ.pop(name)
                    else:
                        os.environ[name] = value
                    config = {"schema_version": 1, "experiment": asdict(experiment()), "plan": v3_plan(),
                        "session_id": "bad-inherited-env", "rendezvous": "/tmp/unused-simple-policy-store"}
                    with patch("mlsys_sim.pp_benchmark_worker.importlib.import_module") as imported:
                        capture = run_worker(config, rank=0, local_rank=0)
                    imported.assert_not_called()
                    self.assertEqual(capture["status"], "FAILED")
                    self.assertEqual(capture["error_code"], "CALIBRATION_MISMATCH")
                    self.assertEqual(capture["stack"]["p2p_protocol_environment"][name], value)
                    self.assertEqual(capture["records"], [])

    def test_startup_cannot_replace_the_threshold_before_measured_operations(self):
        capture, backend, _ = self.run_rank(experiment(), 0, startup_change=True)
        self.assertEqual(capture["status"], "FAILED")
        self.assertEqual(capture["error_code"], "CALIBRATION_MISMATCH")
        self.assertEqual(capture["stack"]["p2p_protocol_environment"]["NCCL_P2P_LL_THRESHOLD"], "4096")
        self.assertNotIn("isend", backend.log)

    def test_original_warmup_device_completion_and_payload_checks_are_preserved(self):
        capture, backend, _ = self.run_rank(experiment(), 2)
        self.assertEqual(capture["status"], "COMPLETED", capture)
        self.assertEqual(backend.log.count("irecv"), 10)
        self.assertLess(backend.log.index("irecv"), backend.log.index("start"))
        windows, current = [], None
        for event in backend.log:
            if event == "start":
                current = []
            elif event == "end":
                windows.append(current)
                current = None
            elif current is not None:
                current.append(event)
        self.assertEqual(len(windows), len(capture["records"]))
        self.assertTrue(all(window in (["irecv", "wait", "sync"], ["allgather", "sync"])
                            for window in windows))
        self.assertEqual({row["duration_ms"] for row in capture["records"]}, {1.0})
        failed, _, _ = self.run_rank(experiment(), 2, corrupt=True)
        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(failed["error_code"], "COMMUNICATION_CORRECTNESS_FAILED")
        self.assertEqual(failed["records"], [])


if __name__ == "__main__":
    unittest.main()
