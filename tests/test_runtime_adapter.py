"""Server trace lifecycle and token-ID SSE capture without GPU dependencies."""

from copy import deepcopy
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import patch

from mlsys_sim import (CalibrationSpec, DeviceSpec, EvaluationSpec, ExecutionSpec, Experiment, InfraSpec,
                       RequestSpec, SimulationError, StrategySpec, compare_runs,
                       comparison_contract, simulate)
from mlsys_sim.hardware import CollectionError
from mlsys_sim.runtime_adapter import (RuntimeTraceRecorder, _sse_payloads,
                                       collect_vllm, normalize_runtime_trace)


SOURCE = {"kind": "test_fixture", "reference": "Synthetic protocol test, not GPU evidence"}


def experiment():
    return Experiment(workload=(RequestSpec("a", 0, 4, 2), RequestSpec("b", 1, 4, 1)),
        strategy=StrategySpec(scheduler="fifo"),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=2, decode_ms=1))


def trace(exp=None):
    exp = exp or experiment()
    recorder = RuntimeTraceRecorder(exp, source=SOURCE)
    recorder.arrival("a", timestamp_ms=0)
    recorder.arrival("b", timestamp_ms=1)
    recorder.token_ready("a", timestamp_ms=2)
    recorder.token_ready("a", timestamp_ms=3)
    recorder.finish("a", timestamp_ms=3)
    recorder.token_ready("b", timestamp_ms=5)
    recorder.finish("b", timestamp_ms=5)
    return recorder.snapshot(observed_until_ms=5)


def vllm_experiment(**changes):
    base = Experiment(workload=(RequestSpec("a", 0, 3, 3, prompt_token_ids=(1, 2, 3)),))
    return replace(base, **changes)


def plan(**changes):
    result = {"schema_version": 1, "endpoint": "http://127.0.0.1:8000", "model": "served-model",
              "source_reference": "SSE fixture, not measured hardware", "max_workers": 1}
    return {**result, **changes}


def packets(batches=((11,), (12, 13)), *, usage=3, finish="length", text="", include_ids=True):
    result = []
    for index, batch in enumerate(batches):
        choice = {"index": 0, "text": text, "finish_reason": finish if index == len(batches) - 1 else None}
        if include_ids:
            choice["token_ids"] = list(batch)
        result.append(json.dumps({"choices": [choice]}, ensure_ascii=False))
    result.append(json.dumps({"choices": [], "usage": {"prompt_tokens": 3,
                                                      "completion_tokens": usage, "total_tokens": 3 + usage}}))
    result.append("[DONE]")
    return result


def encode_payloads(values, newline="\n"):
    return "".join(f"data: {value}{newline}{newline}" for value in values).encode()


class Clock:
    def __init__(self):
        self.values = itertools.count()

    def __call__(self):
        return next(self.values) * 1_000_000


class Transport:
    def __init__(self, payloads=None, fragment=0):
        self.payloads = payloads or packets()
        self.fragment = fragment
        self.calls = []

    def __call__(self, url, body, headers, timeout_s):
        self.calls.append((url, deepcopy(body), dict(headers), timeout_s))
        if body is None:
            yield b'{"version":"0.29.1.fixture"}'
            return
        data = encode_payloads(self.payloads)
        size = self.fragment or len(data)
        for index in range(0, len(data), size):
            yield data[index:index + size]


class RuntimeTraceTests(unittest.TestCase):
    def test_server_trace_normalizes_to_existing_compare_contract(self):
        exp = experiment()
        bundle = normalize_runtime_trace(exp, trace(), allow_test_fixture=True)
        observation = bundle["observation"]
        self.assertEqual(observation["source"], SOURCE)
        self.assertEqual(observation["metrics"]["ttft_p50_ms"], 3)
        self.assertEqual(observation["metrics"]["tpot_p50_ms"], 1)
        self.assertEqual(observation["metrics"]["throughput_requests_per_s"], 400)
        self.assertFalse(bundle["hardware_validated"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predicted = root / "predicted"
            predicted.mkdir()
            run = simulate(exp)
            (predicted / "result.json").write_text(json.dumps(run.result))
            (predicted / "manifest.json").write_text(json.dumps({"backend": "simulated",
                "comparison_contract": comparison_contract(exp)}))
            (predicted / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in run.requests))
            observed = root / "observed.json"
            observed.write_text(json.dumps(observation))
            comparison = compare_runs(predicted, observed, allow_test_fixture=True)
            self.assertEqual(comparison["metrics"]["ttft_p50_ms"]["absolute_error"], 0)
            self.assertEqual(comparison["request_comparison"]["paired_count"], 2)

    def test_fixture_permission_and_injected_clock_never_upgrade_source(self):
        with self.assertRaises(SimulationError) as raised:
            normalize_runtime_trace(experiment(), trace())
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        recorder = RuntimeTraceRecorder(experiment(), source={"kind": "measured", "reference": "declared"},
                                        clock_ns=Clock())
        self.assertEqual(recorder.source["kind"], "test_fixture")

    def test_actual_arrival_and_output_rebinding_is_explicit(self):
        exp = Experiment(workload=(RequestSpec("a", 0, 4, 1),))
        recorder = RuntimeTraceRecorder(exp, source=SOURCE)
        recorder.arrival("a", timestamp_ms=2)
        recorder.token_ready("a", timestamp_ms=3)
        recorder.token_ready("a", timestamp_ms=4)
        recorder.finish("a", timestamp_ms=4)
        raw = recorder.snapshot(observed_until_ms=4)
        with self.assertRaises(SimulationError) as raised:
            normalize_runtime_trace(exp, raw, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "COMPARISON_MISMATCH")
        bound = recorder.replay_experiment()
        self.assertEqual((bound.workload[0].arrival_ms, bound.workload[0].output_tokens), (2, 2))
        self.assertNotEqual(comparison_contract(exp), comparison_contract(bound))
        self.assertEqual(normalize_runtime_trace(bound, raw, allow_test_fixture=True)
                         ["observation"]["requests"][0]["ttft_ms"], 1)

    def test_client_boundary_or_coalesced_tokens_cannot_fake_server_itl(self):
        for boundary, count in (("client_token_delivery_v1", 1), ("server_token_ready_v1", 2)):
            raw = trace()
            raw["timing_boundary"] = boundary
            raw["events"][2]["token_count"] = count
            with self.subTest(boundary=boundary, count=count), self.assertRaises(SimulationError):
                normalize_runtime_trace(experiment(), raw, allow_test_fixture=True)

    def test_lifecycle_duplicate_skipped_index_after_finish_and_unknown_ids_reject(self):
        variants = []
        raw = trace()
        raw["events"][3]["output_token_index"] = 1
        variants.append(raw)
        raw = trace()
        raw["events"][3]["timestamp_ms"] = .5
        variants.append(raw)
        raw = trace()
        raw["events"].append({"event": "TOKEN_READY", "request_id": "a", "timestamp_ms": 5,
                              "token_count": 1, "output_token_index": 3})
        variants.append(raw)
        raw = trace()
        raw["events"][0]["request_id"] = "unknown"
        variants.append(raw)
        raw = trace()
        raw["events"][1] = deepcopy(raw["events"][0])
        variants.append(raw)
        raw = trace()
        raw["observed_until_ms"] = 4
        variants.append(raw)
        for variant in variants:
            with self.subTest(raw=variant), self.assertRaises(SimulationError):
                normalize_runtime_trace(experiment(), variant, allow_test_fixture=True)

    def test_failed_and_unfinished_requests_remain_in_denominator(self):
        raw = trace()
        raw["events"] = raw["events"][:3]
        raw["events"].append({"event": "REQUEST_FINISH", "request_id": "a", "timestamp_ms": 3,
                              "status": "TIMED_OUT", "error_code": "REQUEST_TIMEOUT"})
        exp = experiment()
        exp = replace(exp, workload=(replace(exp.workload[0], timeout_ms=3), exp.workload[1]))
        bundle = normalize_runtime_trace(exp, raw, allow_test_fixture=True)
        self.assertEqual(bundle["observation"]["metrics"]["request_count"], 2)
        self.assertEqual(bundle["observation"]["metrics"]["completion_rate"], 0)
        self.assertIsNone(bundle["observation"]["metrics"]["tpot_p50_ms"])

    def test_execution_window_and_drain_cutoffs_cannot_be_silently_extended(self):
        base = experiment()
        cases = [replace(base, execution=ExecutionSpec(max_simulated_ms=4)),
                 replace(base, evaluation=EvaluationSpec(measurement_window_ms=4, drain=False)),
                 replace(base, evaluation=EvaluationSpec(measurement_window_ms=2, max_drain_ms=2))]
        for exp in cases:
            with self.subTest(exp=exp), self.assertRaises(SimulationError) as raised:
                normalize_runtime_trace(exp, trace(), allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "COMPARISON_MISMATCH")
        exact = replace(base, evaluation=EvaluationSpec(measurement_window_ms=2, max_drain_ms=3))
        self.assertTrue(normalize_runtime_trace(exact, trace(), allow_test_fixture=True)["canonical_comparable"])

    def test_request_deadlines_apply_to_tokens_completion_and_timeout_outcomes(self):
        base = experiment()
        exp = replace(base, workload=(replace(base.workload[0], timeout_ms=2.5), base.workload[1]))
        with self.assertRaises(SimulationError):
            normalize_runtime_trace(exp, trace(), allow_test_fixture=True)
        exp = replace(base, workload=(replace(base.workload[0], timeout_ms=3), base.workload[1]))
        self.assertTrue(normalize_runtime_trace(exp, trace(), allow_test_fixture=True)["canonical_comparable"])
        late_finish = trace()
        late_finish["events"][4]["timestamp_ms"] = 4
        with self.assertRaises(SimulationError):
            normalize_runtime_trace(exp, late_finish, allow_test_fixture=True)
        for deadline in (None, 4):
            raw = trace()
            raw["events"] = raw["events"][:3] + [{"event": "REQUEST_FINISH", "request_id": "a",
                "timestamp_ms": 3, "status": "TIMED_OUT"}]
            current = replace(base, workload=(replace(base.workload[0], timeout_ms=deadline), base.workload[1]))
            with self.subTest(deadline=deadline), self.assertRaises(SimulationError):
                normalize_runtime_trace(current, raw, allow_test_fixture=True)

    def test_outside_window_and_changing_replica_owner_are_not_accepted(self):
        exp = replace(experiment(), evaluation=EvaluationSpec(measurement_window_ms=.5))
        with self.assertRaises(SimulationError):
            normalize_runtime_trace(exp, trace(), allow_test_fixture=True)
        exp = replace(experiment(), infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
                      strategy=StrategySpec(replicas=2))
        raw = trace()
        raw["events"][0]["replica_index"] = 0
        raw["events"][2]["replica_index"] = 1
        with self.assertRaises(SimulationError):
            normalize_runtime_trace(exp, raw, allow_test_fixture=True)

    def test_recorder_rejects_duplicate_or_late_hooks_and_snapshot_is_independent(self):
        recorder = RuntimeTraceRecorder(experiment(), source=SOURCE)
        recorder.arrival("a", timestamp_ms=0)
        with self.assertRaises(SimulationError):
            recorder.arrival("a", timestamp_ms=0)
        with self.assertRaises(SimulationError):
            recorder.token_ready("b", timestamp_ms=1)
        recorder.token_ready("a", timestamp_ms=2)
        recorder.finish("a", timestamp_ms=2)
        with self.assertRaises(SimulationError):
            recorder.token_ready("a", timestamp_ms=2)
        snapshot = recorder.snapshot(observed_until_ms=3)
        snapshot["events"].clear()
        self.assertEqual(len(recorder.events), 3)


class VllmCaptureTests(unittest.TestCase):
    def collect(self, transport=None, **options):
        return collect_vllm(vllm_experiment(), plan(), transport=transport or Transport(),
                            clock_ns=Clock(), allow_test_fixture=True, **options)

    def test_token_ids_drive_counts_even_with_empty_text_and_coalescing(self):
        transport = Transport()
        bundle = self.collect(transport)
        row = bundle["client_requests"][0]
        self.assertEqual(row["status"], "COMPLETED")
        self.assertEqual(row["output_tokens_actual"], 3)
        self.assertEqual(len(row["token_deliveries"]), 2)
        self.assertTrue(row["coalesced"])
        self.assertIsNotNone(row["ttft_ms"])
        self.assertIsNone(bundle["client_metrics"]["itl_p50_ms"])
        self.assertIsNone(bundle["observation"])
        self.assertFalse(bundle["canonical_comparable"])
        self.assertEqual(bundle["source"]["kind"], "test_fixture")
        self.assertEqual(bundle["timing_boundary"], "client_token_delivery_v1")
        body = transport.calls[1][1]
        self.assertEqual(body["prompt"], [1, 2, 3])
        self.assertTrue(body["return_token_ids"])
        self.assertEqual(body["n"], 1)

    def test_sse_utf8_byte_fragmentation_crlf_comments_and_multiline_data(self):
        values = packets(text="你好")
        body = encode_payloads(values, "\r\n")
        body = b': heartbeat\r\n\r\n' + body
        self.assertEqual(list(_sse_payloads((body[index:index + 1] for index in range(len(body))), len(body))), values)
        self.assertEqual(list(_sse_payloads([b'data: {\n', b'data: "choices": []}\n\n'], 100)),
                         ['{\n"choices": []}'])
        self.assertEqual(self.collect(Transport(fragment=1))["client_requests"][0]["output_tokens_actual"], 3)

    def test_missing_ids_usage_mismatch_finish_or_done_preserve_failed_artifacts(self):
        values = packets()
        cases = [packets(include_ids=False), packets(usage=2), values[:-1],
                 [values[0], values[1].replace('"length"', 'null'), *values[2:]],
                 [*values, values[0]],
                 [json.dumps({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 0,
                                                       "total_tokens": 3}}), *values]]
        for payloads in cases:
            with self.subTest(payloads=payloads), self.assertRaises(CollectionError) as raised:
                self.collect(Transport(payloads))
            self.assertEqual(raised.exception.artifacts["source"]["kind"], "test_fixture")
            self.assertEqual(raised.exception.artifacts["client_requests"][0]["status"], "FAILED")
            self.assertEqual(raised.exception.artifacts["status"], "FAILED")
            self.assertFalse(raised.exception.artifacts["hardware_validated"])

    def test_protocol_limits_duplicates_nonfinite_and_invalid_utf8_reject(self):
        with self.assertRaises(SimulationError):
            list(_sse_payloads([b'data: {}\n\n'], 1))
        for invalid in ['{"choices":[],"choices":[]}', '{"x":NaN}']:
            with self.subTest(invalid=invalid), self.assertRaises(CollectionError):
                self.collect(Transport([invalid, '[DONE]']))
        for data in (b'data: \xff\n\n', b'data: [DONE]'):
            with self.subTest(data=data), self.assertRaises(SimulationError):
                list(_sse_payloads([data], 100))

    def test_credentials_never_enter_bundle_or_error_artifacts(self):
        secret = "secret-value-that-must-not-be-recorded"
        def broken(url, body, headers, timeout):
            if body is None:
                return [b'{"version":"fixture"}']
            raise RuntimeError(f"transport failed with Bearer {secret}")
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
            success = self.collect()
            self.assertNotIn(secret, json.dumps(success))
            with self.assertRaises(CollectionError) as raised:
                self.collect(broken)
            self.assertNotIn(secret, json.dumps(raised.exception.artifacts))
            self.assertNotIn(secret, str(raised.exception))
            def echo_version(url, body, headers, timeout):
                return [json.dumps({"version": secret}).encode()]
            with self.assertRaises(SimulationError) as echoed:
                self.collect(echo_version)
            self.assertNotIn(secret, str(echoed.exception))

    def test_version_and_plan_boundary_errors_fail_before_requests(self):
        for config in [plan(runtime_version="wrong"), plan(extra=True), plan(endpoint="http://user:secret@localhost"),
                       plan(endpoint="http://localhost?api_key=x"), plan(max_workers=0), plan(timeout_s=float("nan"))]:
            transport = Transport()
            with self.subTest(config=config), self.assertRaises(SimulationError):
                collect_vllm(vllm_experiment(), config, transport=transport, clock_ns=Clock(), allow_test_fixture=True)
            self.assertLessEqual(len(transport.calls), 1)
        no_ids = Experiment(workload=(RequestSpec("a", 0, 3, 3),))
        with self.assertRaises(SimulationError):
            collect_vllm(no_ids, plan(), transport=Transport(), allow_test_fixture=True)
        with self.assertRaises(SimulationError):
            collect_vllm(replace(vllm_experiment(), evaluation=EvaluationSpec(measurement_window_ms=1, drain=False)),
                         plan(), transport=Transport(), allow_test_fixture=True)

    def test_injection_cannot_promote_inventory_and_live_cpu_never_sends_http(self):
        claimed = {"status": "READY", "source": {"kind": "measured", "reference": "declared"}}
        bundle = self.collect(inventory=claimed)
        self.assertEqual(bundle["source"]["kind"], "test_fixture")
        self.assertEqual(bundle["inventory"]["source"]["kind"], "test_fixture")
        with self.assertRaises(SimulationError) as raised:
            collect_vllm(vllm_experiment(), plan(), transport=Transport())
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        with patch('mlsys_sim.hardware.probe_rocm', return_value={"status": "UNAVAILABLE"}), \
             patch('mlsys_sim.runtime_adapter._http_transport') as opening:
            with self.assertRaises(SimulationError) as raised:
                collect_vllm(vllm_experiment(), plan())
            self.assertEqual(raised.exception.code, "HARDWARE_UNAVAILABLE")
            opening.assert_not_called()

    def test_deadline_before_send_records_timeout_and_outside_window_is_not_sent(self):
        exp = vllm_experiment(workload=(RequestSpec("a", 0, 3, 3, timeout_ms=0, prompt_token_ids=(1, 2, 3)),))
        transport = Transport()
        with self.assertRaises(CollectionError) as raised:
            collect_vllm(exp, plan(), transport=transport, clock_ns=Clock(), allow_test_fixture=True)
        self.assertEqual(raised.exception.artifacts["client_requests"][0]["status"], "TIMED_OUT")
        self.assertEqual(len(transport.calls), 1)
        exp = vllm_experiment(evaluation=EvaluationSpec(measurement_window_ms=1),
                             workload=(RequestSpec("a", 2, 3, 3, prompt_token_ids=(1, 2, 3)),))
        transport = Transport()
        bundle = collect_vllm(exp, plan(), transport=transport, clock_ns=Clock(), allow_test_fixture=True)
        self.assertEqual(bundle["client_requests"][0]["status"], "NOT_ADMITTED")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(bundle["client_metrics"]["request_count"], 0)

    def test_heartbeat_bytes_obey_total_timeout_before_a_data_event_exists(self):
        emitted = []
        closed = []
        def heartbeat(url, body, headers, timeout):
            if body is None:
                yield b'{"version":"fixture"}'
                return
            try:
                for index in range(20):
                    emitted.append(index)
                    yield b': heartbeat\n\n'
            finally:
                closed.append(True)
        ticks = itertools.count()
        with self.assertRaises(CollectionError) as raised:
            collect_vllm(vllm_experiment(), plan(timeout_s=.05), transport=heartbeat,
                         clock_ns=lambda: next(ticks) * 30_000_000, allow_test_fixture=True)
        self.assertLess(len(emitted), 20)
        self.assertTrue(closed)
        self.assertEqual(raised.exception.artifacts["client_requests"][0]["status"], "TIMED_OUT")

    def test_queued_requests_never_send_after_the_global_capture_horizon(self):
        requests = tuple(RequestSpec(name, 0, 3, 3, prompt_token_ids=(1, 2, 3)) for name in ('a', 'b'))
        exp = vllm_experiment(workload=requests, execution=ExecutionSpec(max_simulated_ms=6))
        transport = Transport(fragment=1)
        with self.assertRaises(CollectionError) as raised:
            collect_vllm(exp, plan(), transport=transport, clock_ns=Clock(), allow_test_fixture=True)
        post_requests = [call for call in transport.calls if call[1] is not None]
        self.assertLessEqual(len(post_requests), 1)
        self.assertTrue(all(row['status'] == 'TIMED_OUT' for row in raised.exception.artifacts['client_requests']))

    def test_local_http_server_exercises_actual_standard_library_transport(self):
        payload = encode_payloads(packets())
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"version":"local-fixture"}')

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if body['return_token_ids'] is not True:
                    self.send_error(400)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(payload)

        try:
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        except PermissionError:
            self.skipTest('Loopback listeners are disabled by the execution sandbox')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            inventory = {"source": dict(SOURCE), "status": "UNAVAILABLE"}
            bundle = collect_vllm(vllm_experiment(), plan(endpoint=f'http://127.0.0.1:{server.server_port}'),
                                 inventory, allow_test_fixture=True)
            self.assertEqual(bundle["source"]["kind"], "test_fixture")
            self.assertEqual(bundle["runtime_version"], "local-fixture")
            self.assertEqual(bundle["client_requests"][0]["output_tokens_actual"], 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_real_transport_refuses_redirect_without_forwarding_credentials(self):
        class Target(BaseHTTPRequestHandler):
            received = []
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.received.append(self.headers.get('Authorization'))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"version":"redirect-target"}')

        servers = []
        threads = []
        try:
            try:
                target = ThreadingHTTPServer(('127.0.0.1', 0), Target)
            except PermissionError:
                self.skipTest('Loopback listeners are disabled by the execution sandbox')
            servers.append(target)
            class Redirect(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_GET(self):
                    self.send_response(302)
                    self.send_header('Location', f'http://127.0.0.1:{target.server_port}/target')
                    self.end_headers()

            origin = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
            servers.append(origin)
            for server in servers:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                threads.append(thread)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'must-not-cross-origins'}):
                with self.assertRaises(SimulationError) as raised:
                    collect_vllm(vllm_experiment(), plan(endpoint=f'http://127.0.0.1:{origin.server_port}'),
                                 {'source': dict(SOURCE)}, allow_test_fixture=True)
                self.assertEqual(raised.exception.code, 'COLLECTION_FAILED')
            self.assertEqual(Target.received, [])
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
