"""Active sequence admission is distinct from per-step batch membership."""

from dataclasses import replace
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, RequestSpec,
                       SimulationError, StrategySpec, simulate)
from mlsys_sim.config import experiment_from_dict


def configured(*, limit=None, batch=1, requests=None, **options):
    return Experiment(
        workload=tuple(requests or [RequestSpec(name, 0, 16, 3) for name in ("a", "b", "c")]),
        strategy=StrategySpec(max_batch_size=batch, max_active_requests=limit, **options),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=10, decode_ms=1,
                                    mixed_ms=10, recompute_ms=5))


class AdmissionLimitTests(unittest.TestCase):
    def test_limit_one_serializes_residency_without_changing_arrival(self):
        unrestricted = simulate(configured())
        limited = simulate(configured(limit=1))
        before = [event["timestamp_ms"] for event in unrestricted.events
                  if event["event"] == "STEP_START" and event["phase"] == "prefill"]
        after = [event["timestamp_ms"] for event in limited.events
                 if event["event"] == "STEP_START" and event["phase"] == "prefill"]
        self.assertEqual(before, [0, 10, 20])
        self.assertEqual(after, [0, 12, 24])
        self.assertEqual([row["arrival_ms"] for row in limited.requests], [0, 0, 0])
        self.assertEqual(limited.result["counts"]["completed"], 3)

    def test_provisional_batch_members_share_the_admission_limit(self):
        run = simulate(configured(limit=2, batch=8))
        starts = [event for event in run.events if event["event"] == "STEP_START"]
        self.assertEqual(starts[0]["request_ids"], ["a", "b"])
        self.assertLessEqual(max(len(event["request_ids"]) for event in starts), 2)
        self.assertEqual(run.result["counts"]["completed"], 3)

    def test_failed_and_timed_out_requests_release_the_slot(self):
        requests = [RequestSpec("a", 0, 16, 3, timeout_ms=5), RequestSpec("b", 0, 16, 1)]
        run = simulate(configured(limit=1, requests=requests))
        self.assertEqual([row["status"] for row in run.requests], ["TIMED_OUT", "COMPLETED"])
        self.assertEqual(run.requests[1]["first_token_ms"], 20)
        requests = [RequestSpec("a", 0, 17, 1), RequestSpec("b", 0, 16, 1)]
        run = simulate(configured(limit=1, requests=requests, max_batch_tokens=16))
        self.assertEqual([row["status"] for row in run.requests], ["FAILED", "COMPLETED"])

    def test_chunked_and_mixed_paths_preserve_one_active_sequence(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                run = simulate(configured(limit=1, batch=8, chunked_prefill=True,
                                          prefill_chunk_tokens=8, mixed_batching=mixed))
                active = set()
                for event in run.events:
                    if event["event"] == "STEP_START":
                        active.update(event["request_ids"])
                        self.assertLessEqual(len(active), 1)
                    if event["event"] == "REQUEST_FINISH":
                        active.discard(event["request_id"])
                self.assertEqual(run.result["counts"]["completed"], 3)

    def test_pipeline_and_replica_limits_are_local(self):
        pipeline = configured(limit=1, batch=2, pipeline_parallel_size=2,
                              pipeline_p2p=False, pipeline_max_inflight_microbatches=4)
        pipeline = replace(pipeline, infra=InfraSpec(devices=(DeviceSpec("p0"), DeviceSpec("p1"))))
        run = simulate(pipeline)
        active = set()
        for event in run.events:
            if event["event"] == "MICROBATCH_ADMITTED":
                active.update(event["request_ids"])
                self.assertLessEqual(len(active), 1)
            if event["event"] == "REQUEST_FINISH":
                active.discard(event["request_id"])
        self.assertEqual(run.result["counts"]["completed"], 3)
        replica = configured(limit=1, replicas=2, replica_routing="round_robin",
                             requests=[RequestSpec(str(i), 0, 16, 3) for i in range(4)])
        replica = replace(replica, infra=InfraSpec(devices=(DeviceSpec("r0"), DeviceSpec("r1"))))
        run = simulate(replica)
        self.assertEqual(run.result["counts"]["completed"], 4)
        self.assertEqual(sorted(row["first_token_ms"] for row in run.requests), [10, 10, 22, 22])

    def test_limit_is_opt_in_and_strictly_validated(self):
        self.assertIsNone(experiment_from_dict({}).strategy.max_active_requests)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(SimulationError):
                experiment_from_dict({"strategy": {"max_active_requests": value}})


if __name__ == "__main__":
    unittest.main()
