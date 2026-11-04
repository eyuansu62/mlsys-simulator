"""Explicit lazy native-PP collection remains isolated from the v1 protocol."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import SimulationError
from mlsys_sim.config import experiment_from_dict
from mlsys_sim.hardware import CollectionError
from mlsys_sim.pp_benchmark import (COMMUNICATOR_INITIALIZATION, PROTOCOL, PROTOCOL_V2,
    collect_pp_communication, normalize_pp_benchmark_plan, worker_launch_contract)
from mlsys_sim.pp_benchmark_worker import run_worker
from mlsys_sim.pp_communication import validate_pp_communication_dataset

from test_pp_benchmark import FakeTorch, envelope, experiment, plan, rank_captures, runner


def v2_plan():
    return {**plan(), "protocol": PROTOCOL_V2}


def protocol_runner(argv, **kwargs):
    config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
    target = experiment_from_dict(config["experiment"])
    captures = rank_captures(target, config["plan"], config["session_id"])
    protocol = config["plan"].get("protocol", PROTOCOL)
    for capture in captures:
        capture["protocol"] = protocol
    result = envelope(captures)
    if protocol == PROTOCOL_V2:
        result["protocol"] = protocol
    return result


class ProtocolSelectionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_default_v1_plan_and_contract_payloads_stay_unchanged(self):
        target = experiment()
        normalized = normalize_pp_benchmark_plan(target, plan())
        self.assertEqual(normalized, {**plan(), "timeout_s": 120, "max_output_bytes": 8 * 1024 * 1024})
        self.assertEqual(worker_launch_contract(target, normalized), {
            "protocol": "native_pp_primitives_v1", "world_size": 4, "activation_dtype": "float16",
            "message_bytes": [16, 32], "warmup_iterations": 1, "iterations": 2, "repeats": 2,
            "control_backend": "gloo", "primitive_backend": "nccl",
            "group_creation": "all_world_ranks_same_order", "p2p_api": "isend_irecv",
            "allgather_api": "all_gather_into_tensor", "buffer_policy": "preallocated_out_of_place"})
        result = collect_pp_communication(target, plan(), runner=runner, allow_test_fixture=True)
        self.assertNotIn("protocol", result["raw_records"][0])
        self.assertNotIn("communicator_initialization", result["dataset"]["benchmark"])
        self.assertNotIn("world_init_device_id", result["dataset"]["benchmark"]["launch"]["rank_contract"])
        self.assertEqual(result["dataset"]["benchmark"]["protocol"], PROTOCOL)
        self.assertEqual(result["provenance"]["protocol"], PROTOCOL)

    def test_protocol_is_explicit_and_does_not_reorder_message_sizes(self):
        options = {**v2_plan(), "message_bytes": [32, 16]}
        normalized = normalize_pp_benchmark_plan(experiment(), options)
        self.assertEqual(normalized["protocol"], PROTOCOL_V2)
        self.assertEqual(normalized["message_bytes"], [32, 16])
        explicit_v1 = normalize_pp_benchmark_plan(experiment(), {**plan(), "protocol": PROTOCOL})
        self.assertEqual(explicit_v1["protocol"], PROTOCOL)
        for protocol in (None, True, 2, "v2", "native_pp_primitives_v3", {"protocol": PROTOCOL_V2}):
            with self.subTest(protocol=protocol), self.assertRaises(SimulationError):
                normalize_pp_benchmark_plan(experiment(), {**plan(), "protocol": protocol})

    def test_v2_collection_metadata_all_selects_one_protocol(self):
        target, options = experiment(), v2_plan()
        result = collect_pp_communication(target, options, runner=protocol_runner, allow_test_fixture=True)
        benchmark = result["dataset"]["benchmark"]
        contract = worker_launch_contract(target, normalize_pp_benchmark_plan(target, options))
        self.assertEqual(contract["protocol"], PROTOCOL_V2)
        self.assertIs(contract["world_init_device_id"], False)
        self.assertEqual(contract["communicator_initialization"], COMMUNICATOR_INITIALIZATION)
        self.assertEqual(benchmark["protocol"], PROTOCOL_V2)
        self.assertEqual(benchmark["communicator_initialization"], COMMUNICATOR_INITIALIZATION)
        self.assertEqual(benchmark["launch"]["rank_contract"], contract)
        self.assertEqual(result["provenance"]["protocol"], PROTOCOL_V2)
        raw = result["raw_records"][0]
        self.assertEqual(raw["protocol"], PROTOCOL_V2)
        self.assertTrue(all(capture["protocol"] == PROTOCOL_V2 and capture["launch"] == contract
                            for capture in raw["rank_captures"]))
        first = result["dataset"]["records"][0]
        self.assertEqual(first["duration_ms"], max(row["duration_ms"] for row in first["rank_durations_ms"]))
        self.assertFalse(result["hardware_validated"])

    def test_v2_raw_roundtrip_requires_matching_envelope_rank_and_launch_protocol(self):
        target = experiment()
        result = collect_pp_communication(target, v2_plan(), runner=protocol_runner, allow_test_fixture=True)
        original = result["raw_records"][0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            options = {**v2_plan(), "raw_path": path.name,
                "source": {"kind": "test_fixture", "reference": "CPU protocol-domain fixture"}}
            path.write_text(json.dumps(original))
            imported = collect_pp_communication(target, options, base_dir=temporary, allow_test_fixture=True)
            self.assertEqual(imported["dataset"]["records"], result["dataset"]["records"])
            self.assertEqual(imported["raw_records"][0]["raw_input"]["bytes"], path.stat().st_size)
            changes = [lambda value: value.pop("protocol"),
                lambda value: value.update(protocol=PROTOCOL),
                lambda value: value["rank_captures"][0].update(protocol=PROTOCOL),
                lambda value: value["rank_captures"][0]["launch"].update(protocol=PROTOCOL),
                lambda value: value["rank_captures"][0]["launch"].update(world_init_device_id=True),
                lambda value: value["rank_captures"][0]["launch"].update(
                    communicator_initialization="observed_fast_algorithm")]
            for change in changes:
                changed = deepcopy(original)
                change(changed)
                path.write_text(json.dumps(changed))
                with self.subTest(change=change), self.assertRaises(CollectionError):
                    collect_pp_communication(target, options, base_dir=temporary, allow_test_fixture=True)
            path.write_text(json.dumps(original))
            with self.assertRaises(CollectionError):
                collect_pp_communication(target, {key: value for key, value in options.items() if key != "protocol"},
                                         base_dir=temporary, allow_test_fixture=True)

    def test_benchmark_initialization_metadata_is_versioned_and_fixed(self):
        v1 = collect_pp_communication(experiment(), plan(), runner=runner, allow_test_fixture=True)["dataset"]
        v2 = collect_pp_communication(experiment(), v2_plan(), runner=protocol_runner,
                                     allow_test_fixture=True)["dataset"]
        for change in (lambda value: value["benchmark"].pop("communicator_initialization"),
                       lambda value: value["benchmark"].update(communicator_initialization="eager")):
            changed = deepcopy(v2)
            change(changed)
            with self.subTest(change=change), self.assertRaises(SimulationError):
                validate_pp_communication_dataset(changed, allow_test_fixture=True)
        v1["benchmark"]["communicator_initialization"] = COMMUNICATOR_INITIALIZATION
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(v1, allow_test_fixture=True)

    def test_selected_actual_environment_is_auditable_without_dumping_all_environment(self):
        selected = {"TORCH_NCCL_BLOCKING_WAIT": "0", "NCCL_BLOCKING_WAIT": "1", "NCCL_P2P_DISABLE": "1",
            "NCCL_PROTO": "Simple", "NCCL_P2P_LEVEL": "LOC", "NCCL_SHM_DISABLE": "0", "NCCL_NET": "Socket",
            "NCCL_IB_DISABLE": "1", "HIP_VISIBLE_DEVICES": "1,0"}
        with patch.dict(os.environ, {**selected, "TASK_SECRET": "must-not-be-captured",
                                     "NCCL_DEBUG_FILE": "/tmp/private-log-path"}):
            result = collect_pp_communication(experiment(), v2_plan(), runner=protocol_runner,
                                             allow_test_fixture=True)
        expected = {**selected, "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"}
        self.assertEqual(result["raw_records"][0]["environment"], expected)
        self.assertEqual(result["dataset"]["benchmark"]["launch"]["environment"], expected)
        observed = []
        def inspect(argv, **kwargs):
            observed.append(dict(kwargs["env"]))
            return protocol_runner(argv, **kwargs)
        result = collect_pp_communication(experiment(), v2_plan(), runner=inspect, allow_test_fixture=True)
        self.assertNotIn("TORCH_NCCL_BLOCKING_WAIT", observed[0])
        self.assertNotIn("NCCL_BLOCKING_WAIT", observed[0])
        self.assertNotIn("NCCL_P2P_DISABLE", observed[0])
        self.assertNotIn("TORCH_NCCL_BLOCKING_WAIT", result["dataset"]["benchmark"]["launch"]["environment"])

    def test_v2_preserves_complete_stack_source_session_and_output_guards(self):
        changes = [lambda value: value["rank_captures"].pop(),
            lambda value: value["rank_captures"][0].update(session_id="borrowed-session"),
            lambda value: value["rank_captures"][0]["stack"].update(communication_environment_sha256="b" * 64),
            lambda value: value["rank_captures"][0]["records"][0].update(correctness_errors=1)]
        for change in changes:
            def changed_runner(argv, **kwargs):
                value = protocol_runner(argv, **kwargs)
                change(value)
                return value
            with self.subTest(change=change), self.assertRaises(CollectionError) as raised:
                collect_pp_communication(experiment(), v2_plan(), runner=changed_runner, allow_test_fixture=True)
            self.assertTrue(raised.exception.artifacts["raw_records"])
        with self.assertRaises(SimulationError) as raised:
            collect_pp_communication(experiment(), v2_plan(), runner=protocol_runner)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        with self.assertRaises(CollectionError) as raised:
            collect_pp_communication(experiment(), {**v2_plan(), "max_output_bytes": 128},
                                     runner=protocol_runner, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "COLLECTION_OUTPUT_LIMIT")
        for change in ({"warmup_iterations": 0}, {"iterations": 1000000}, {"message_bytes": [16, 17]},
                       {"timeout_s": float("inf")}, {"max_output_bytes": 32 * 1024 * 1024}):
            with self.subTest(change=change), self.assertRaises(SimulationError):
                normalize_pp_benchmark_plan(experiment(), {**v2_plan(), **change})


class LazyWorkerTests(unittest.TestCase):
    def run_rank(self, options, rank=2, *, corrupt=False):
        target = experiment()
        backend = FakeTorch(target, rank)
        calls = []
        original = backend.initialize
        def initialize(**kwargs):
            calls.append(kwargs)
            original(**kwargs)
        backend.distributed.init_process_group = initialize
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
        result = run_worker({"schema_version": 1, "experiment": asdict(target), "plan": options,
            "session_id": "cpu-lazy-worker", "rendezvous": "/tmp/unused-lazy-fixture-store"},
            torch_module=backend, clock_ns=clock,
            binding_provider=lambda: {"runtime_version": "2.30.7", "library_sha256": "a" * 64},
            rank=rank, local_rank=rank)
        return result, backend, calls

    def test_only_explicit_v2_omits_world_device_id(self):
        for options in (plan(), {**plan(), "protocol": PROTOCOL}, v2_plan()):
            with self.subTest(protocol=options.get("protocol")):
                result, backend, calls = self.run_rank(options)
                self.assertEqual(result["status"], "COMPLETED", result)
                self.assertEqual(len(calls), 1)
                if options.get("protocol") == PROTOCOL_V2:
                    self.assertNotIn("device_id", calls[0])
                    self.assertIs(result["launch"]["world_init_device_id"], False)
                else:
                    self.assertEqual(calls[0]["device_id"], 2)
                    self.assertNotIn("world_init_device_id", result["launch"])
                self.assertEqual(calls[0]["backend"], "nccl")
                self.assertEqual([entry[1] for entry in backend.log if isinstance(entry, tuple)],
                                 ["gloo", "nccl", "nccl", "nccl"])
                self.assertEqual(result["protocol"], options.get("protocol", PROTOCOL))

    def test_v2_warms_real_p2p_and_keeps_original_completion_timing_boundary(self):
        result, backend, _ = self.run_rank(v2_plan())
        self.assertEqual(result["status"], "COMPLETED", result)
        self.assertEqual(backend.log.count("irecv"), 10)  # Two real warmups plus eight measured receives.
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
        self.assertEqual(len(windows), len(result["records"]))
        self.assertTrue(all(window in (["irecv", "wait", "sync"], ["allgather", "sync"])
                            for window in windows))
        self.assertEqual({row["duration_ms"] for row in result["records"]}, {1.0})
        self.assertEqual(backend.log[-1], "destroy")
        failed, _, _ = self.run_rank(v2_plan(), corrupt=True)
        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(failed["error_code"], "COMMUNICATION_CORRECTNESS_FAILED")
        self.assertEqual(failed["records"], [])


if __name__ == "__main__":
    unittest.main()
