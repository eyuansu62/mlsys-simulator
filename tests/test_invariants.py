"""Generated workloads exercise interactions between events and KV pressure."""

from dataclasses import replace
import random
import unittest

from mlsys_sim.engine import simulate
from mlsys_sim.schema import (CalibrationSpec, ExecutionSpec, Experiment, MemoryConfig,
                              RequestSpec, StrategySpec)


class GeneratedInvariantTests(unittest.TestCase):
    def test_generated_workloads_preserve_tokens_time_and_resources(self):
        generator = random.Random(20261103)
        for case in range(80):
            specs = tuple(RequestSpec(
                request_id=f"r{i}", arrival_ms=float(generator.randrange(6)),
                input_tokens=generator.choice([1, 8, 16, 17, 32]),
                output_tokens=generator.choice([1, 2, 4, 20]),
                priority=generator.randrange(3),
                timeout_ms=generator.choice([None, None, 0.0, 5.0, 40.0]))
                for i in range(generator.randrange(1, 6)))
            experiment = Experiment(
                workload=specs,
                strategy=StrategySpec(
                    scheduler=generator.choice(["fifo", "continuous_batching"]),
                    max_batch_size=generator.randrange(1, 5),
                    max_batch_tokens=generator.choice([16, 32, 64, 256]),
                    preemption=generator.choice(["none", "recompute"]),
                    memory=MemoryConfig(kv_pool_bytes=8192 * generator.randrange(1, 9))),
                calibration=CalibrationSpec(backend="synthetic", prefill_ms=3.0,
                                            decode_ms=1.0, recompute_ms=2.0),
                execution=ExecutionSpec(max_simulated_ms=200.0, max_events=2000))
            with self.subTest(case=case):
                simulation = simulate(experiment)
                rows, events, result = simulation.requests, simulation.events, simulation.result
                self.assertEqual(sum(result["counts"][key] for key in
                                     ["completed", "failed", "timed_out", "unfinished"]), len(specs))
                previous = -1.0
                for event in events:
                    self.assertGreaterEqual(event["timestamp_ms"], previous)
                    previous = event["timestamp_ms"]
                    if event["event"] == "STEP_START":
                        for rank in event["memory"]:
                            self.assertLessEqual(rank["used_blocks"] + rank["guard_blocks"],
                                                 rank["pool_blocks"])
                            self.assertLessEqual(rank["budget_peak_bytes"], rank["device_memory_bytes"])
                targets = {spec.request_id: spec for spec in specs}
                for row in rows:
                    self.assertLessEqual(row["output_tokens_actual"], targets[row["request_id"]].output_tokens)
                    times = row["token_times_ms"]
                    self.assertEqual(len(times), row["output_tokens_actual"])
                    self.assertEqual(times, sorted(times))
                    self.assertTrue(all(time >= row["arrival_ms"] for time in times))
                    if row["status"] == "COMPLETED":
                        self.assertEqual(row["output_tokens_actual"], targets[row["request_id"]].output_tokens)
                        self.assertEqual(row["num_computed_tokens"],
                                         targets[row["request_id"]].input_tokens + len(times) - 1)
                if result["status"] == "COMPLETED":
                    self.assertTrue(all(rank["used_blocks"] == 0 for rank in result["memory"]))
                # Recording is observational: disabling it cannot change state or metrics.
                quiet = simulate(replace(experiment, execution=replace(experiment.execution,
                                                                      detailed_events=False)))
                self.assertEqual(result, quiet.result)
                self.assertEqual(rows, quiet.requests)
                self.assertEqual(quiet.events, [])


if __name__ == "__main__":
    unittest.main()
