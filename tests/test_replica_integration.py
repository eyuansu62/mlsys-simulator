"""Global metric arithmetic, warmup isolation and public multi-replica examples."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment, InfraSpec,
                       MemoryConfig, RequestSpec, StrategySpec, load_experiment, simulate)


ROOT = Path(__file__).resolve().parents[1]


class ReplicaIntegrationTests(unittest.TestCase):
    def test_global_percentile_is_recomputed_and_uses_one_common_window(self):
        exp = Experiment(workload=tuple(RequestSpec(name, 0, 4, 1) for name in ("a", "b", "c")),
            infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
            strategy=StrategySpec(replicas=2, replica_routing="round_robin", scheduler="fifo"),
            calibration=CalibrationSpec(backend="synthetic", prefill_ms=2))
        run = simulate(exp)
        self.assertEqual([row["token_times_ms"] for row in run.requests], [[2], [2], [4]])
        self.assertEqual([row["replica_index"] for row in run.requests], [0, 1, 0])
        self.assertEqual(run.result["latency_ms"]["ttft"]["p50"], 2)
        local_medians = [replica["result"]["latency_ms"]["ttft"]["p50"]
                         for replica in run.result["replicas"]]
        self.assertEqual(local_medians, [3, 2])
        self.assertNotEqual(sum(local_medians) / 2, run.result["latency_ms"]["ttft"]["p50"])
        self.assertEqual(run.result["throughput"]["window_ms"], 4)
        self.assertEqual(run.result["throughput"]["requests_per_s"], 750)
        self.assertEqual(run.result["counts"]["completed"], 3)

    def test_warmup_discards_replica_caches_and_resets_round_robin_cursor(self):
        ids = tuple(range(8))
        warmup = tuple(RequestSpec(name, 0, 8, 1, prompt_token_ids=ids) for name in ("w0", "w1", "w2"))
        measured = tuple(RequestSpec(name, 0, 8, 1, prompt_token_ids=ids) for name in ("m0", "m1"))
        exp = Experiment(workload=measured, infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
            strategy=StrategySpec(replicas=2, replica_routing="round_robin", prefix_cache=True,
                                  memory=MemoryConfig(kv_block_tokens=4)),
            execution=ExecutionSpec(warmup=warmup), calibration=CalibrationSpec(backend="synthetic", prefill_ms=2))
        run = simulate(exp)
        self.assertEqual(run.result["warmup"]["routing"]["request_count_by_replica"], [2, 1])
        self.assertEqual(run.result["routing"]["request_count_by_replica"], [1, 1])
        self.assertEqual([row["replica_index"] for row in run.requests], [0, 1])
        self.assertEqual([row["prefix_cached_tokens"] for row in run.requests], [0, 0])
        self.assertGreater(run.result["warmup"]["replicas"][0]["result"]["prefix_cache"]["cached_blocks"], 0)
        self.assertEqual(run.result["prefix_cache"]["hit_tokens"], 0)

    def test_public_examples_replay_and_report_actual_owned_resources(self):
        cases = (("round_robin", 4, [2, 2]), ("prefix_aware", 3, [1, 2]),
                 ("replicas_pp_mixed_cache_swap", 10, [5, 5]))
        with tempfile.TemporaryDirectory() as temporary:
            for name, completed, assignments in cases:
                outputs = []
                for replay in range(2):
                    output = Path(temporary) / f"{name}-{replay}"
                    process = subprocess.run([sys.executable, "-m", "mlsys_sim", "simulate",
                        "--experiment", str(ROOT / "examples/v1_4" / f"{name}.json"), "--out", str(output)],
                        cwd=ROOT, capture_output=True, text=True, check=False)
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    result = json.loads((output / "result.json").read_text())
                    self.assertEqual(result["counts"]["completed"], completed)
                    self.assertEqual(result["routing"]["request_count_by_replica"], assignments)
                    self.assertEqual(result["prefix_cache"]["attached_block_references"], 0)
                    self.assertEqual(result["prefix_cache"]["private_blocks"], 0)
                    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
                    owners = {row["request_id"]: row["replica_index"] for row in rows}
                    self.assertEqual(len(owners), completed)
                    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
                    self.assertEqual([event["timestamp_ms"] for event in events],
                                     sorted(event["timestamp_ms"] for event in events))
                    for event in events:
                        if "request_id" in event and event["request_id"] in owners:
                            self.assertEqual(event["replica_index"], owners[event["request_id"]])
                    if completed == 10:
                        self.assertEqual(result["batching"]["mixed_batches_completed"], 4)
                        self.assertEqual(result["prefix_cache"]["hit_tokens"], 24)
                        self.assertEqual(result["swap"]["completed_transfers_by_direction"], {"D2H": 2, "H2D": 2})
                        self.assertEqual(result["swap"]["host_reserved_bytes"], 0)
                        self.assertEqual(result["recomputed_tokens"], 0)
                        self.assertEqual(len(result["memory"]), 4)
                    outputs.append(output)
                for artifact in ("result.json", "requests.jsonl", "events.jsonl"):
                    self.assertEqual((outputs[0] / artifact).read_bytes(),
                                     (outputs[1] / artifact).read_bytes(), (name, artifact))

    def test_logs_are_observational_and_tp_pp_routing_keeps_each_requests_owner(self):
        base = load_experiment(ROOT / "examples/v1_4/replicas_pp_mixed_cache_swap.json")
        exp = replace(base, infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(8))),
            strategy=replace(base.strategy, tensor_parallel_size=2, pipeline_p2p=True,
                pipeline_feedback_latency_us=500,
                memory=replace(base.strategy.memory, kv_pool_bytes=2560)))
        run = simulate(exp)
        quiet = simulate(replace(exp, execution=replace(exp.execution, detailed_events=False)))
        self.assertEqual(run.result, quiet.result)
        self.assertEqual(run.requests, quiet.requests)
        self.assertEqual(quiet.events, [])
        self.assertEqual(run.result["counts"]["completed"], 10)
        self.assertGreater(run.result["batching"]["mixed_batches_completed"], 0)
        self.assertGreater(run.result["prefix_cache"]["hit_tokens"], 0)
        self.assertGreater(run.result["swap"]["completed_transfers_by_direction"]["D2H"], 0)
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)
        self.assertEqual(len(run.result["memory"]), 8)
        self.assertEqual({row["replica_index"] for row in run.requests}, {0, 1})


if __name__ == "__main__":
    unittest.main()
