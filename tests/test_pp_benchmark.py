"""CPU-only primitive protocol, complete-rank reduction and process boundaries."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlsys_sim import CalibrationSpec, DeviceSpec, Experiment, InfraSpec, ModelSpec, SimulationError, StrategySpec
from mlsys_sim.hardware import CollectionError
from mlsys_sim.pp_benchmark import (PROTOCOL, _run_bounded, benchmark_cases, collect_pp_communication,
    normalize_pp_benchmark_plan, sample_descriptor, worker_launch_contract)
from mlsys_sim.pp_benchmark_worker import _launch_world, run_worker
from mlsys_sim.profiles import profile_identity


def experiment(tp=2):
    return Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"d{i}", name=f"AMD CPU fixture {i}")
                                                   for i in range(tp * 2))),
        strategy=StrategySpec(tensor_parallel_size=tp, pipeline_parallel_size=2,
            pipeline_transport="vllm_native", model=ModelSpec(activation_dtype="float16")),
        calibration=CalibrationSpec(rccl_version="2.30.7"))


def plan():
    return {"schema_version": 1, "dataset_id": "isolated-fixture", "message_bytes": [16, 32],
            "warmup_iterations": 1, "iterations": 2, "repeats": 2,
            "source_reference": "CPU injected primitive protocol; no GPU measurement"}


def rank_captures(exp, options, session_id="fixture-session"):
    options = normalize_pp_benchmark_plan(exp, options)
    result = []
    for rank, device in enumerate(exp.infra.devices):
        rows = []
        for case in benchmark_cases(exp):
            if rank not in case["participants"]:
                continue
            for size in options["message_bytes"]:
                for repeat in range(options["repeats"]):
                    for iteration in range(options["iterations"]):
                        rows.append({**sample_descriptor(options, case, size, repeat, iteration),
                            "duration_ms": .01 + size / 1e6 + rank / 1000,
                            "correctness_errors": 0})
        result.append({"schema_version": 1, "kind": "native_pp_primitive_rank_capture", "protocol": PROTOCOL,
            "session_id": session_id, "rank": rank, "world_size": len(exp.infra.devices), "identity": profile_identity(exp),
            "source": {"kind": "test_fixture", "reference": "CPU fixture rank"}, "status": "COMPLETED",
            "stack": {"torch_version": "fixture-torch", "hip_version": "fixture-hip",
                      "rccl_version": "2.30.7", "rccl_library_sha256": "a" * 64},
            "device": {"index": rank, "name": device.name, "memory_bytes": device.memory_bytes},
            "launch": worker_launch_contract(exp, options), "records": rows})
    return result


def envelope(captures):
    return {"schema_version": 1, "kind": "native_pp_primitives_capture", "returncode": 0,
        "stdout": "", "stderr": "", "timed_out": False, "output_truncated": False,
        "rank_captures": captures}


def runner(argv, **kwargs):
    config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
    from mlsys_sim.config import experiment_from_dict
    exp = experiment_from_dict(config["experiment"])
    return envelope(rank_captures(exp, config["plan"], session_id=config["session_id"]))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_complete_one_way_and_full_output_records_reduce_by_maximum(self):
        exp, settings = experiment(), plan()
        calls = []
        def execute(argv, **kwargs):
            calls.append((argv, kwargs))
            return runner(argv, **kwargs)
        collected = collect_pp_communication(exp, settings, runner=execute, allow_test_fixture=True)
        dataset = collected["dataset"]
        self.assertEqual(len(dataset["records"]), 24)
        self.assertEqual(dataset["source"]["kind"], "test_fixture")
        self.assertFalse(collected["hardware_validated"])
        first = dataset["records"][0]
        self.assertAlmostEqual(first["duration_ms"], .012016)
        self.assertEqual(first["message_bytes"], 16)
        self.assertEqual([r["rank"] for r in first["rank_durations_ms"]], [0, 2])
        gather = next(row for row in dataset["records"] if row["primitive"] == "allgather")
        self.assertEqual(gather["message_bytes"], 16)  # Full output, never divided a second time.
        self.assertEqual(gather["participants"], [2, 3])
        self.assertFalse(dataset["benchmark"]["allocation_in_timing"])
        self.assertIn("--launch-world", calls[0][0])
        self.assertEqual(calls[0][1]["env"]["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "1")
        self.assertIn("output_dir", calls[0][1])

    def test_tp_one_collects_only_directional_p2p(self):
        result = collect_pp_communication(experiment(tp=1), plan(), runner=runner, allow_test_fixture=True)
        self.assertEqual({row["primitive"] for row in result["dataset"]["records"]}, {"p2p"})

    def test_injection_requires_permission_and_never_promotes_measured_tag(self):
        with self.assertRaises(SimulationError) as raised:
            collect_pp_communication(experiment(), plan(), runner=runner)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        def measured_tag(argv, **kwargs):
            value = runner(argv, **kwargs)
            for capture in value["rank_captures"]:
                capture["source"]["kind"] = "measured"
            return value
        result = collect_pp_communication(experiment(), plan(), runner=measured_tag, allow_test_fixture=True)
        self.assertEqual(result["dataset"]["source"]["kind"], "test_fixture")

    def test_plan_rejects_unwarmed_misaligned_unbounded_or_ambiguous_work(self):
        changes = [{"warmup_iterations": 0}, {"message_bytes": [16, 17]}, {"message_bytes": [16, 16]},
                   {"message_bytes": [16]}, {"iterations": True}, {"iterations": 1000000},
                   {"timeout_s": float("inf")}, {"max_output_bytes": 32 * 1024 * 1024},
                   {"source": {"kind": "measured", "reference": "cannot label a live launch"}}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(SimulationError):
                normalize_pp_benchmark_plan(experiment(), {**plan(), **change})

    def test_wrong_direction_bytes_missing_rank_and_bad_correctness_are_rejected(self):
        def mutate(change):
            def execute(argv, **kwargs):
                value = runner(argv, **kwargs)
                change(value)
                return value
            return execute
        changes = [lambda value: value["rank_captures"].pop(),
            lambda value: value["rank_captures"][0]["records"][0].update(participants=[2, 0]),
            lambda value: value["rank_captures"][1]["records"][0].update(participants=[True, 3]),
            lambda value: value["rank_captures"][0]["records"][0].update(message_bytes=8),
            lambda value: value["rank_captures"][0]["records"].pop(),
            lambda value: value["rank_captures"][0]["records"][0].update(correctness_errors=1),
            lambda value: value["rank_captures"][0]["records"][0].update(duration_ms=float("nan")),
            lambda value: value["rank_captures"][0]["launch"].update(p2p_api="pingpong_half_rtt"),
            lambda value: value["rank_captures"][0]["launch"].update(warmup_iterations=True)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(CollectionError) as raised:
                collect_pp_communication(experiment(), plan(), runner=mutate(change), allow_test_fixture=True)
            self.assertIn("raw_records", raised.exception.artifacts)

    def test_sessions_are_required_and_cannot_be_mixed_or_borrowed_by_live_launch(self):
        exp, settings = experiment(), plan()
        original = rank_captures(exp, settings)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.json"
            options = {**settings, "raw_path": path.name,
                       "source": {"kind": "test_fixture", "reference": "Cohort rejection fixture"}}
            for change in (lambda values: values[0].pop("session_id"),
                           lambda values: values[0].update(session_id="other-session"),
                           lambda values: values[0].update(session_id="bad/session")):
                changed = json.loads(json.dumps(original))
                change(changed)
                path.write_text(json.dumps(envelope(changed)))
                with self.subTest(change=change), self.assertRaises(CollectionError):
                    collect_pp_communication(exp, options, base_dir=temporary, allow_test_fixture=True)
        def wrong_launch(argv, **kwargs):
            value = runner(argv, **kwargs)
            for capture in value["rank_captures"]:
                capture["session_id"] = "borrowed-cohort"
            return value
        with self.assertRaises(CollectionError) as raised:
            collect_pp_communication(exp, settings, runner=wrong_launch, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_failed_or_truncated_launch_keeps_partial_rank_evidence(self):
        for field, value, code in (("returncode", 3, "COLLECTION_FAILED"),
                                   ("timed_out", True, "COLLECTION_TIMEOUT"),
                                   ("output_truncated", True, "COLLECTION_OUTPUT_LIMIT")):
            def execute(argv, **kwargs):
                result = runner(argv, **kwargs)
                result[field] = value
                result["rank_captures"] = result["rank_captures"][:1]
                return result
            with self.subTest(field=field), self.assertRaises(CollectionError) as raised:
                collect_pp_communication(experiment(), plan(), runner=execute, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(len(raised.exception.artifacts["raw_records"][0]["rank_captures"]), 1)

    def test_raw_import_checks_source_and_retains_file_hash_without_local_gpu(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(json.dumps([envelope(rank_captures(exp, plan()))]))
            options = {**plan(), "raw_path": path.name,
                       "source": {"kind": "test_fixture", "reference": "Preserved CPU fixture"}}
            unavailable = {"status": "UNAVAILABLE"}
            result = collect_pp_communication(exp, options, unavailable, base_dir=temporary, allow_test_fixture=True)
            self.assertEqual(result["dataset"]["source"]["kind"], "test_fixture")
            self.assertEqual(result["raw_records"][0]["raw_input"]["bytes"], path.stat().st_size)
            with self.assertRaises(CollectionError) as raised:
                collect_pp_communication(exp, options, unavailable, base_dir=temporary)
            self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")

    def test_raw_import_duplicate_json_and_output_bound_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"kind":"native_pp_primitives_capture","kind":"other"}')
            options = {**plan(), "raw_path": path.name,
                       "source": {"kind": "test_fixture", "reference": "Invalid CPU fixture"}}
            with self.assertRaises(CollectionError):
                collect_pp_communication(experiment(), options, base_dir=temporary, allow_test_fixture=True)
            path.write_text("x" * 1000)
            with self.assertRaises(CollectionError) as raised:
                collect_pp_communication(experiment(), {**options, "max_output_bytes": 100},
                                         base_dir=temporary, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "COLLECTION_OUTPUT_LIMIT")

    def test_nested_distributed_launcher_is_not_reinterpreted(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}), self.assertRaises(CollectionError):
            collect_pp_communication(experiment(), plan(), runner=runner, allow_test_fixture=True)


class Tensor:
    def __init__(self, size, log):
        self.values = [0] * size
        self.log = log
    def fill_(self, value):
        self.log.append("fill")
        self.values[:] = [value] * len(self.values)
        return self
    def __ne__(self, value):
        self.log.append("correctness")
        return [item != value for item in self.values]
    def __getitem__(self, key):
        result = Tensor(0, self.log)
        result.values = self.values[key]
        return result


class FakeTorch:
    __version__ = "fixture-torch"
    version = SimpleNamespace(hip="fixture-hip")
    float16 = "float16"
    def __init__(self, exp, rank):
        self.log = []
        self.rank = rank
        self.exp = exp
        self.initialized = False
        self.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: len(exp.infra.devices),
            set_device=lambda value: None, synchronize=lambda value: self.log.append("sync"),
            get_device_properties=lambda value: SimpleNamespace(name=exp.infra.devices[value].name,
                                                               total_memory=exp.infra.devices[value].memory_bytes))
        self.distributed = SimpleNamespace(is_nccl_available=lambda: True, is_gloo_available=lambda: True,
            is_initialized=lambda: self.initialized, init_process_group=self.initialize,
            get_world_size=lambda: len(exp.infra.devices), get_backend=lambda: "nccl",
            new_group=self.new_group, barrier=self.barrier, isend=self.send, irecv=self.receive,
            all_gather_into_tensor=self.gather, destroy_process_group=lambda: self.log.append("destroy"))
    def initialize(self, **kwargs):
        self.initialized = True
        self.log.append("init")
    def new_group(self, **kwargs):
        self.log.append(("group", kwargs["backend"], kwargs["ranks"]))
        assert kwargs["use_local_synchronization"] is False
        return kwargs
    def barrier(self, group):
        assert group["backend"] == "gloo"
        self.log.append("barrier")
    def send(self, tensor, dst, group):
        self.log.append("isend")
        return SimpleNamespace(wait=lambda: self.log.append("wait"))
    def receive(self, tensor, src, group):
        self.log.append("irecv")
        tensor.values[:] = [(src + 1) % 32] * len(tensor.values)
        return SimpleNamespace(wait=lambda: self.log.append("wait"))
    def gather(self, output, input_tensor, group):
        self.log.append("allgather")
        output.values[:] = [((rank + 1) % 32) for rank in group["ranks"] for _ in input_tensor.values]
    def device(self, kind, index):
        return index
    def empty(self, size, **kwargs):
        self.log.append("allocate")
        return Tensor(size, self.log)
    def count_nonzero(self, values):
        return SimpleNamespace(item=lambda: sum(values))


class WorkerTests(unittest.TestCase):
    def config(self):
        return {"schema_version": 1, "experiment": asdict(experiment()), "plan": plan(),
                "session_id": "cpu-worker", "rendezvous": "/tmp/unused-fixture-store"}

    def run_rank(self, rank, *, change=None):
        config = self.config()
        backend = FakeTorch(experiment(), rank)
        if change is not None:
            change(backend)
        ticks = 0
        def clock():
            nonlocal ticks
            ticks += 1
            backend.log.append("start" if ticks % 2 else "end")
            return ticks * 1_000_000
        result = run_worker(config, torch_module=backend, clock_ns=clock,
            binding_provider=lambda: {"runtime_version": "2.30.7", "library_sha256": "a" * 64},
            rank=rank, local_rank=rank)
        return result, backend

    def test_all_world_ranks_create_identical_groups_and_only_participants_operate(self):
        first, a = self.run_rank(0)
        second, b = self.run_rank(2)
        self.assertEqual(first["status"], "COMPLETED", first)
        self.assertEqual(second["status"], "COMPLETED", second)
        self.assertEqual([x for x in a.log if isinstance(x, tuple)], [x for x in b.log if isinstance(x, tuple)])
        self.assertEqual({r["primitive"] for r in first["records"]}, {"p2p"})
        self.assertEqual({r["primitive"] for r in second["records"]}, {"p2p", "allgather"})
        self.assertEqual(first["source"]["kind"], "test_fixture")
        self.assertEqual(first["session_id"], "cpu-worker")
        self.assertEqual(len(first["stack"]["communication_environment_sha256"]), 64)

    def test_barrier_allocation_warmup_and_correctness_stay_outside_timer(self):
        capture, backend = self.run_rank(2)
        inside = False
        windows = []
        current = []
        for event in backend.log:
            if event == "start":
                self.assertFalse(inside)
                inside, current = True, []
            elif event == "end":
                self.assertTrue(inside)
                inside = False
                windows.append(current)
            elif inside:
                current.append(event)
        self.assertEqual(len(windows), len(capture["records"]))
        self.assertTrue(all("barrier" not in row and "fill" not in row and "allocate" not in row
                            and "correctness" not in row for row in windows))
        self.assertTrue(all(row[-1] == "sync" for row in windows))
        self.assertTrue(any(row == ["irecv", "wait", "sync"] for row in windows))
        self.assertTrue(any(row == ["allgather", "sync"] for row in windows))
        self.assertEqual({r["duration_ms"] for r in capture["records"]}, {1.0})
        # Each participating cell has one true warmup in addition to timed calls.
        self.assertEqual(backend.log.count("irecv"), 10)

    def test_receiver_checks_every_payload_and_keeps_failure_evidence(self):
        def corrupt(backend):
            original = backend.receive
            def receive(tensor, **kwargs):
                result = original(tensor, **kwargs)
                tensor.values[0] = -7
                return result
            backend.distributed.irecv = receive
        capture, _ = self.run_rank(2, change=corrupt)
        self.assertEqual(capture["status"], "FAILED")
        self.assertEqual(capture["error_code"], "COMMUNICATION_CORRECTNESS_FAILED")
        self.assertEqual(capture["records"], [])  # Failed warmup is never promoted.

    def test_cuda_only_or_wrong_device_never_starts_a_primitive(self):
        for change in (lambda backend: setattr(backend, "version", SimpleNamespace(hip=None)),
                       lambda backend: setattr(backend.cuda, "get_device_properties",
                            lambda _: SimpleNamespace(name="wrong", total_memory=1))):
            capture, backend = self.run_rank(0, change=change)
            self.assertEqual(capture["status"], "FAILED")
            self.assertNotIn("isend", backend.log)

    def test_supervisor_polls_every_rank_before_deciding_a_failure(self):
        class Process:
            def __init__(self, code):
                self.code, self.returncode, self.polled = code, None, 0
            def poll(self):
                self.polled += 1
                self.returncode = self.code
                return self.code
            def terminate(self):
                self.code = -15
            def kill(self):
                self.code = -9
            def wait(self, **kwargs):
                self.returncode = self.code
                return self.code
        workers = [Process(None), Process(1), Process(0), Process(0)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker.json"
            path.write_text(json.dumps(self.config()))
            with patch("mlsys_sim.pp_benchmark_worker.subprocess.Popen", side_effect=workers):
                self.assertEqual(_launch_world(path, Path(temporary) / "out"), 1)
        self.assertTrue(all(worker.polled for worker in workers))


@unittest.skipUnless(os.name == "posix", "Owned POSIX process sessions")
class BoundedLauncherTests(unittest.TestCase):
    def test_timeout_terminates_a_term_ignoring_parent_and_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            child_pid = directory / "child.pid"
            heartbeat = directory / "heartbeat"
            child_code = ("import pathlib,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                f"path=pathlib.Path({str(heartbeat)!r});"
                "exec('while True:\\n path.write_text(str(time.monotonic_ns()))\\n time.sleep(.005)')")
            code = ("import pathlib,signal,subprocess,sys,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
                f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid));time.sleep(30)")
            result = _run_bounded([sys.executable, "-c", code], timeout_s=1, max_output_bytes=8192,
                                  env=dict(os.environ), output_dir=directory)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["returncode"], 0)
            self.assertTrue(child_pid.exists())
            self.assertTrue(heartbeat.exists())
            before = heartbeat.read_text()
            time.sleep(.03)
            self.assertEqual(heartbeat.read_text(), before)

    def test_output_bound_truncates_evidence_and_stops_the_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = _run_bounded([sys.executable, "-c", "import time;print('x'*10000,flush=True);time.sleep(30)"],
                timeout_s=5, max_output_bytes=128, env=dict(os.environ), output_dir=Path(temporary))
        self.assertTrue(result["output_truncated"])
        self.assertLessEqual(len(result["stdout"].encode()) + len(result["stderr"].encode()), 128)


if __name__ == "__main__":
    unittest.main()
