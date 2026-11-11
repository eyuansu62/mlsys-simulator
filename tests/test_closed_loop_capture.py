"""CPU-only runtime clients: true terminal refill and honest API boundaries."""

from copy import deepcopy
import unittest

from mlsys_sim.closed_loop_capture import TIMING_BOUNDARY, collect_closed_loop
from mlsys_sim.load_validation import generate_request_stream
from mlsys_sim.schema import SimulationError


def spec(*, requests=7, think=0, timeout=100, maximum=1000, outputs=2):
    return {"schema_version": 1, "study_id": "cpu-capture", "cohort": "frozen",
            "concurrency_grid": [1, 2, 3, 8], "requests_per_run": requests,
            "replications": 2, "seed": 42, "think_ms": think,
            "input_lengths": [4, 8, 16], "output_tokens": outputs,
            "timeout_ms": timeout, "max_run_ms": maximum, "prompt_token_id": 100}


class Clock:
    def __init__(self, read_advance=0):
        self.now = 0
        self.read_advance = read_advance
        self.reads = 0
        self.fail_at = None

    def __call__(self):
        self.reads += 1
        if self.reads == self.fail_at:
            raise RuntimeError("injected clock failure")
        self.now += self.read_advance
        return self.now


class Backend:
    def __init__(self, clock, *, cycles=2, step_ns=1_000_000, submit_ns=1000,
                 abort_ns=200_000, reverse=False, early_tokens=False):
        self.clock = clock
        self.cycles = cycles
        self.step_ns, self.submit_ns, self.abort_ns = step_ns, submit_ns, abort_ns
        self.reverse, self.early_tokens = reverse, early_tokens
        self.pending = {}
        self.log = []
        self.max_pending = 0
        self.steps = 0
        self.raise_add = None
        self.raise_step = None
        self.raise_abort = False
        self.short = False
        self.overshoot = False
        self.rollback_add = False
        self.bad_feedback = None

    def add_request(self, request):
        self.log.append(("add", request["stream_index"], request["client_id"], self.clock.now))
        if request["stream_index"] == self.raise_add:
            raise RuntimeError("injected submission failure")
        native = f"native-{request['stream_index']}"
        self.pending[native] = {"request": deepcopy(request), "ticks": 0}
        self.max_pending = max(self.max_pending, len(self.pending))
        self.clock.now += self.submit_ns
        if self.rollback_add:
            self.clock.now -= 10_000
        return native

    def step(self):
        self.steps += 1
        self.log.append(("step", tuple(self.pending), self.clock.now))
        self.clock.now += self.step_ns
        if self.steps == self.raise_step:
            raise RuntimeError("injected step failure")
        records = []
        for native, value in list(self.pending.items()):
            value["ticks"] += 1
            target = value["request"]["output_tokens"]
            count = target if self.early_tokens else min(target, target * value["ticks"] // self.cycles)
            finished = value["ticks"] >= self.cycles
            if self.short:
                count = max(0, target - 1)
            if self.overshoot:
                count, finished = target + 1, False
            records.append({"native_request_id": native, "finished": finished, "output_tokens_actual": count})
            if finished:
                del self.pending[native]
        if self.reverse:
            records.reverse()
        if self.bad_feedback is not None:
            records = self.bad_feedback(records)
        return records

    def has_unfinished_requests(self):
        return bool(self.pending)

    def abort_requests(self, native_ids):
        self.log.append(("abort", list(native_ids), self.clock.now))
        if self.raise_abort:
            raise RuntimeError("injected abort failure")
        self.clock.now += self.abort_ns
        for native in native_ids:
            self.pending.pop(native, None)


class ClosedLoopCaptureTests(unittest.TestCase):
    def test_initial_clients_are_all_admitted_before_step_and_never_queue_all_N(self):
        selected = spec(requests=7)
        snapshots = []
        for concurrency in (1, 3):
            clock = Clock()
            backend = Backend(clock)
            report = collect_closed_loop(backend, selected, concurrency, clock_ns=clock)
            self.assertEqual(report["status"], "COMPLETED")
            self.assertEqual(report["offered_count"], 7)
            self.assertEqual(report["completed_count"], 7)
            self.assertEqual(report["not_offered_count"], 0)
            self.assertEqual([event[0] for event in backend.log[:concurrency + 1]], ["add"] * concurrency + ["step"])
            self.assertEqual(backend.max_pending, concurrency)
            self.assertEqual(report["peak_logical_outstanding"], concurrency)
            self.assertEqual(report["client_peak_outstanding"], [1] * concurrency)
            self.assertEqual(report["client_current_outstanding"], [0] * concurrency)
            self.assertEqual(report["request_stream"], generate_request_stream(selected))
            self.assertEqual(report["timing_boundary"], TIMING_BOUNDARY)
            self.assertEqual(report["source"]["kind"], "test_fixture")
            self.assertTrue(report["resources_drained"])
            for row in report["requests"]:
                self.assertNotIn("token_times_ms", row)
                self.assertIsNotNone(row["native_request_id"])
            snapshots.append(report)
        self.assertEqual(snapshots[0]["request_stream_sha256"], snapshots[1]["request_stream_sha256"])
        self.assertNotEqual([row["submit_before_ns"] for row in snapshots[0]["requests"]],
                            [row["submit_before_ns"] for row in snapshots[1]["requests"]])
        self.assertNotEqual(snapshots[0]["policy_sha256"], snapshots[1]["policy_sha256"])

    def test_same_step_terminal_feedback_refills_in_original_stream_order(self):
        clock = Clock()
        backend = Backend(clock, cycles=1, reverse=True)
        report = collect_closed_loop(backend, spec(requests=5), 2, clock_ns=clock)
        self.assertEqual(report["status"], "COMPLETED")
        self.assertEqual([row["client_id"] for row in report["requests"][:4]], [0, 1, 0, 1])
        for row in report["requests"][2:]:
            previous = next(item for item in report["requests"] if item["request_id"] == row["preceding_request_id"])
            self.assertGreaterEqual(row["submit_before_ns"], previous["terminal_api_feedback_ns"])

    def test_target_token_count_without_finished_feedback_does_not_release_client(self):
        clock = Clock()
        backend = Backend(clock, cycles=3, early_tokens=True)
        report = collect_closed_loop(backend, spec(requests=2), 1, clock_ns=clock)
        first, second = report["requests"]
        self.assertEqual(first["step_feedback_count"], 3)
        self.assertLess(first["first_token_api_feedback_ns"], first["terminal_api_feedback_ns"])
        self.assertEqual([item["delta_output_tokens"] for item in first["feedback"]], [2, 0, 0])
        self.assertGreaterEqual(second["submit_before_ns"], first["terminal_api_feedback_ns"])
        self.assertEqual(backend.max_pending, 1)

    def test_positive_think_delay_is_honored_from_real_terminal_feedback(self):
        clock = Clock(read_advance=200_000)
        report = collect_closed_loop(Backend(clock, cycles=1), spec(requests=5, think=2), 2, clock_ns=clock)
        self.assertEqual(report["status"], "COMPLETED")
        for row in report["requests"]:
            if row["preceding_request_id"] is not None:
                previous = next(item for item in report["requests"] if item["request_id"] == row["preceding_request_id"])
                self.assertGreaterEqual(row["submit_before_ns"], previous["terminal_api_feedback_ns"] + 2_000_000)

    def test_zero_clock_and_nonadvancing_think_wait_fail_without_hanging(self):
        clock = Clock()
        report = collect_closed_loop(Backend(clock, cycles=1, step_ns=0, submit_ns=0), spec(requests=1), 1, clock_ns=clock)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error_code"], "NONPOSITIVE_FORMAL_INTERVAL")
        clock = Clock()
        report = collect_closed_loop(Backend(clock, cycles=1), spec(requests=2, think=1), 1, clock_ns=clock)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error_code"], "CLOCK_NOT_PROGRESSING")
        self.assertEqual(report["offered_count"], 1)
        self.assertLess(clock.reads, 4200)

    def test_request_timeout_aborts_at_step_boundary_then_reuses_client(self):
        clock = Clock()
        backend = Backend(clock, cycles=4)
        report = collect_closed_loop(backend, spec(requests=2, timeout=1.5), 1, clock_ns=clock)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["timed_out_count"], 2)
        self.assertEqual(report["offered_count"], 2)
        self.assertEqual(report["not_offered_count"], 0)
        self.assertTrue(report["resources_drained"])
        first, second = report["requests"]
        self.assertGreater(first["abort_before_ns"], first["request_deadline_ns"])
        self.assertGreater(first["abort_ack_ns"], first["abort_before_ns"])
        self.assertEqual(first["terminal_api_feedback_ns"], first["abort_ack_ns"])
        self.assertGreaterEqual(second["submit_before_ns"], first["abort_ack_ns"])
        self.assertEqual(first["output_tokens_actual"], 1)

    def test_completion_wins_exact_deadline_and_late_finished_is_timeout(self):
        for duration, expected in ((1_000_000, "COMPLETED"), (1_100_000, "TIMED_OUT")):
            with self.subTest(duration=duration):
                clock = Clock()
                backend = Backend(clock, step_ns=duration, submit_ns=0)
                report = collect_closed_loop(backend, spec(requests=1, timeout=2), 1, clock_ns=clock)
                self.assertEqual(report["requests"][0]["status"], expected)
                self.assertEqual(report["requests"][0]["terminal_source"], "native_finished_feedback")

    def test_slow_initial_submission_times_out_at_boundary_before_any_kernel(self):
        clock = Clock()
        backend = Backend(clock, submit_ns=1_000_000)
        report = collect_closed_loop(backend, spec(requests=2, timeout=.5), 2, clock_ns=clock)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["timed_out_count"], 2)
        self.assertEqual([event[0] for event in backend.log], ["add", "add", "abort"])
        self.assertEqual(report["step_count"], 0)
        self.assertTrue(report["resources_drained"])

    def test_global_budget_records_actual_late_boundary_and_unoffered_remainder(self):
        clock = Clock()
        backend = Backend(clock, step_ns=2_000_000)
        report = collect_closed_loop(backend, spec(requests=5, maximum=1.5), 2, clock_ns=clock)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error_code"], "COHORT_TIME_LIMIT")
        self.assertEqual(report["offered_count"], 2)
        self.assertEqual(report["not_offered_count"], 3)
        self.assertEqual(report["unfinished_count"], 2)
        self.assertGreater(report["elapsed_ms"], 1.5)
        self.assertTrue(report["resources_drained"])
        self.assertTrue(all(row["abort_acknowledged"] for row in report["requests"]))
        self.assertTrue(all("submit_before_ns" not in row for row in report["request_stream"]["requests"]))

    def test_short_and_overshoot_outputs_are_failed_without_truncating_actuals(self):
        for overshoot in (False, True):
            with self.subTest(overshoot=overshoot):
                clock = Clock()
                backend = Backend(clock, cycles=1)
                backend.short, backend.overshoot = not overshoot, overshoot
                report = collect_closed_loop(backend, spec(requests=3), 1, clock_ns=clock)
                self.assertEqual(report["failed_count"], 3)
                self.assertEqual(report["completed_count"], 0)
                self.assertEqual(report["offered_count"], 3)
                self.assertEqual(report["status"], "FAILED")
                self.assertTrue(report["resources_drained"])
                self.assertTrue(all(row["output_tokens_actual"] == (3 if overshoot else 1) for row in report["requests"]))

    def test_submission_and_step_exceptions_preserve_partial_counts_and_cleanup(self):
        for operation in ("add", "step"):
            with self.subTest(operation=operation):
                clock = Clock()
                backend = Backend(clock, cycles=3)
                if operation == "add":
                    backend.raise_add = 1
                else:
                    backend.raise_step = 2
                report = collect_closed_loop(backend, spec(requests=5), 2, clock_ns=clock)
                self.assertEqual(report["status"], "FAILED")
                self.assertEqual(report["offered_count"], 2)
                self.assertEqual(report["not_offered_count"], 3)
                self.assertEqual(report["error_code"], f"BACKEND_{operation.upper()}_ERROR")
                self.assertTrue(report["resources_drained"])
                self.assertTrue(any(event[0] == "abort" for event in backend.log))
                if operation == "add":
                    self.assertIsNone(report["requests"][1]["native_request_id"])
                else:
                    self.assertEqual(report["requests"][0]["step_feedback_count"], 1)

    def test_abort_failure_retains_unacknowledged_ownership(self):
        clock = Clock()
        backend = Backend(clock, cycles=5)
        backend.raise_abort = True
        report = collect_closed_loop(backend, spec(requests=2, timeout=.5), 1, clock_ns=clock)
        self.assertEqual(report["status"], "FAILED")
        self.assertFalse(report["resources_drained"])
        self.assertEqual(report["client_current_outstanding"], [1])
        self.assertEqual(report["unacknowledged_native_request_ids"], ["native-0"])
        self.assertFalse(report["requests"][0]["abort_acknowledged"])

    def test_clock_rollback_keeps_native_identity_and_does_not_invent_submit_end(self):
        clock = Clock()
        clock.now = 100_000
        backend = Backend(clock)
        backend.rollback_add = True
        report = collect_closed_loop(backend, spec(requests=2), 1, clock_ns=clock)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error_code"], "CLOCK_ROLLBACK")
        self.assertEqual(report["requests"][0]["native_request_id"], "native-0")
        self.assertIsNone(report["requests"][0]["submit_after_ns"])
        self.assertTrue(report["resources_drained"])

    def test_clock_failure_after_step_preserves_returned_actual_counts_without_timestamp(self):
        clock = Clock()
        clock.fail_at = 6
        report = collect_closed_loop(Backend(clock, cycles=1), spec(requests=1), 1, clock_ns=clock)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error_code"], "CLOCK_ERROR")
        self.assertEqual(report["requests"][0]["output_tokens_actual"], 2)
        self.assertIsNone(report["requests"][0]["terminal_api_feedback_ns"])
        self.assertIsNone(report["step_feedback"][0]["feedback_ns"])
        self.assertEqual(report["step_feedback"][0]["records"][0]["output_tokens_actual"], 2)

    def test_unknown_duplicate_and_decreasing_feedback_are_explicit_protocol_failures(self):
        for bad, code in ((lambda rows: rows + [{"native_request_id": "unknown", "finished": True, "output_tokens_actual": 2}],
                           "UNKNOWN_NATIVE_FEEDBACK"), (lambda rows: rows + rows, "DUPLICATE_STEP_FEEDBACK")):
            with self.subTest(code=code):
                clock = Clock()
                backend = Backend(clock, cycles=1)
                backend.bad_feedback = bad
                report = collect_closed_loop(backend, spec(requests=1), 1, clock_ns=clock)
                self.assertEqual(report["status"], "FAILED")
                self.assertEqual(report["error_code"], code)
                self.assertEqual(report["requests"][0]["output_tokens_actual"], 2)
        clock = Clock()
        backend = Backend(clock, cycles=3, early_tokens=True)
        def rollback(rows):
            if backend.steps == 2:
                rows[0]["output_tokens_actual"] = 1
            return rows
        backend.bad_feedback = rollback
        report = collect_closed_loop(backend, spec(requests=1), 1, clock_ns=clock)
        self.assertEqual(report["error_code"], "TOKEN_COUNT_ROLLBACK")
        self.assertEqual(report["requests"][0]["output_tokens_actual"], 2)
        self.assertFalse(report["requests"][0]["output_ledger_valid"])

    def test_C_above_N_and_progress_stages_do_not_offer_extra_requests(self):
        clock = Clock()
        backend = Backend(clock, cycles=1)
        stages = []
        report = collect_closed_loop(backend, spec(requests=3), 8, clock_ns=clock, progress=stages.append)
        self.assertEqual(report["offered_count"], 3)
        self.assertEqual(report["client_peak_outstanding"], [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual([item["stage"] for item in stages], ["formal_start", "formal_end"])
        clock = Clock()
        stages = []
        report = collect_closed_loop(Backend(clock, cycles=1), spec(requests=205), 8, clock_ns=clock, progress=stages.append)
        self.assertEqual(report["status"], "COMPLETED")
        self.assertEqual(sum(item["stage"] == "progress" for item in stages), 2)
        self.assertEqual(stages[-1]["completed_count"], 205)

    def test_progress_exceptions_and_preexisting_native_work_are_not_hidden(self):
        clock = Clock()
        def broken(_):
            raise RuntimeError("callback failure")
        report = collect_closed_loop(Backend(clock), spec(), 1, clock_ns=clock, progress=broken)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error_code"], "PROGRESS_CALLBACK_ERROR")
        self.assertEqual(report["offered_count"], 0)
        clock = Clock()
        backend = Backend(clock)
        backend.pending["unrelated"] = {}
        report = collect_closed_loop(backend, spec(), 1, clock_ns=clock)
        self.assertEqual(report["error_code"], "BACKEND_NOT_EMPTY")
        self.assertFalse(report["resources_drained"])
        self.assertIn("unrelated", backend.pending)
        self.assertFalse(any(event[0] == "abort" for event in backend.log))

    def test_injected_clock_never_claims_measured_source_and_invalid_interface_fails_early(self):
        clock = Clock()
        backend = Backend(clock, cycles=1)
        backend.source = {"kind": "measured", "reference": "attested adapter"}
        report = collect_closed_loop(backend, spec(requests=1), 1, clock_ns=clock)
        self.assertEqual(report["source"]["kind"], "test_fixture")
        self.assertFalse(report["hardware_validated"])
        self.assertTrue(report["metric_contract"]["not_server_token_ready"])
        with self.assertRaises(SimulationError):
            collect_closed_loop(object(), spec(), 1)
        with self.assertRaises(SimulationError):
            collect_closed_loop(backend, spec(), 7, clock_ns=clock)


if __name__ == "__main__":
    unittest.main()
