"""The dynamic adapter must retain the ordinary engine's causal semantics."""

from dataclasses import replace
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, ExecutionSpec, Experiment,
                       InfraSpec, RequestSpec, SimulationError, StrategySpec, simulate)
from mlsys_sim.event_driver import EventDriver


def experiment(*, pp=1, replicas=1, detailed=True):
    return Experiment(
        infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(pp * replicas))),
        strategy=StrategySpec(pipeline_parallel_size=pp, replicas=replicas, scheduler="fifo"),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=10, decode_ms=1),
        execution=ExecutionSpec(detailed_events=detailed))


def drive(exp, requests):
    driver = EventDriver(exp)
    arrivals = sorted(requests, key=lambda spec: (spec.arrival_ms, spec.request_id))
    while arrivals or driver.next_event() is not None:
        event = driver.next_event()
        when = min(arrivals[0].arrival_ms if arrivals else float("inf"),
                   event[0] if event is not None else float("inf"))
        while (event := driver.next_event()) is not None and event[0] == when:
            driver.advance_one()
        driver.set_time(when)
        while arrivals and arrivals[0].arrival_ms == when:
            driver.accept_request(arrivals.pop(0))
        while (event := driver.next_event()) is not None and event[0] == when:
            driver.advance_one()
        driver.schedule_ready()
        if driver.stopped:
            break
    return driver, driver.finalize()


class EventDriverTests(unittest.TestCase):
    def test_single_pp_and_replica_equivalence_with_timeouts_and_idle_gap(self):
        rows = (RequestSpec("a", 0, 4, 2), RequestSpec("b", 10, 4, 1),
                RequestSpec("c", 50, 4, 1, timeout_ms=1000))
        for pp, replicas in ((1, 1), (2, 1), (1, 2), (2, 2)):
            with self.subTest(pp=pp, replicas=replicas):
                exp = experiment(pp=pp, replicas=replicas)
                _, dynamic = drive(exp, rows)
                ordinary = simulate(replace(exp, workload=rows))
                self.assertEqual(dynamic.requests, ordinary.requests)
                self.assertEqual(dynamic.result, {k: v for k, v in ordinary.result.items() if k != "warmup"})
                self.assertEqual(dynamic.events, ordinary.events)

    def test_quiet_first_compute_start_is_observed_without_events(self):
        for pp in (1, 2):
            with self.subTest(pp=pp):
                driver, result = drive(experiment(pp=pp, detailed=False),
                                       (RequestSpec("a", 0, 4, 1), RequestSpec("b", 0, 4, 1)))
                self.assertEqual(driver.first_compute_start_ms,
                                 {"a": 0, "b": result.requests[0]["completion_ms"]})
                self.assertEqual(result.events, [])

    def test_invalid_state_and_pending_events_cannot_be_skipped(self):
        exp = experiment()
        with self.assertRaises(SimulationError):
            EventDriver(replace(exp, workload=(RequestSpec("a", 0, 4, 1),)))
        driver = EventDriver(exp)
        driver.accept_request(RequestSpec("a", 0, 4, 1))
        driver.schedule_ready()
        for when in (-1, float("inf"), True, 11):
            with self.subTest(when=when), self.assertRaises(SimulationError):
                driver.set_time(when)
        with self.assertRaises(SimulationError):
            driver.accept_request(RequestSpec("future", 1, 4, 1))
        driver.advance_one()
        first = driver.finalize()
        self.assertIs(driver.finalize(), first)
        with self.assertRaises(SimulationError):
            driver.accept_request(RequestSpec("b", 10, 4, 1))

    def test_local_time_budget_accounts_partial_work_without_committing(self):
        exp = replace(experiment(replicas=2), execution=ExecutionSpec(max_simulated_ms=5))
        driver = EventDriver(exp)
        driver.accept_request(RequestSpec("a", 0, 4, 1))
        driver.accept_request(RequestSpec("b", 0, 4, 1))
        driver.schedule_ready()
        self.assertEqual(driver.next_event(), (5, 3, -1))
        driver.advance_one()
        result = driver.finalize()
        self.assertEqual(result.result["error_code"], "SIMULATION_TIME_LIMIT")
        self.assertEqual(result.result["compute_busy_ms"], 10)
        self.assertEqual([row["output_tokens_actual"] for row in result.requests], [0, 0])
        self.assertTrue(all(row["used_blocks"] == 1 for row in result.result["memory"]))

    def test_local_event_budget_counts_admission_once(self):
        exp = replace(experiment(), execution=ExecutionSpec(max_events=1))
        driver = EventDriver(exp)
        driver.accept_request(RequestSpec("a", 0, 4, 1))
        driver.schedule_ready()
        driver.advance_one()
        result = driver.finalize()
        self.assertEqual(result.result["events_processed"], 1)
        self.assertEqual(result.result["error_code"], "SIMULATION_EVENT_LIMIT")
        self.assertEqual(result.requests[0]["output_tokens_actual"], 0)

    def test_completion_at_cutoff_cannot_be_skipped_by_clock_advance(self):
        driver = EventDriver(replace(experiment(), execution=ExecutionSpec(max_simulated_ms=10)))
        driver.accept_request(RequestSpec("a", 0, 4, 1))
        driver.schedule_ready()
        with self.assertRaises(SimulationError):
            driver.set_time(11)
        driver.set_time(10)
        driver.advance_one()
        self.assertEqual(driver.finalize().requests[0]["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
