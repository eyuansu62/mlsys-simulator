"""Real loopback Agent interchange and bounded binary worker checks."""

import hashlib
import http.client
import json
import os
import subprocess
import threading
import unittest
from unittest.mock import patch

from mlsys_sim.agent_catalog import CATALOG_VERSION
from mlsys_sim.agent_contract import validate_agent_response
from mlsys_sim.agent_service import (MAX_AGENT_BODY_BYTES, MAX_AGENT_WALL_S,
                                      create_agent_server, get_agent_service_catalog, run_agent_isolated)


def agent_fixture():
    return {
        "schema": "mlab.simulator.v1", "client_request_id": "接口-α", "catalog_version": CATALOG_VERSION,
        "task": "transport-conformance", "candidate_id": "known-Qwen-BF16-not-an-original-baseline",
        "workload": {"mode": "open_loop", "source_sha256": "sha256:" + "a" * 64,
                     "requests": [{"request_id": "原始-r1", "input_tokens": 16,
                                   "output_tokens_target": 2, "arrival_ms": 0}]},
        "infra": {"id": "course-nominal", "nodes": [{"id": "node0", "devices": [
            {"id": "gpu0", "hardware": "rx9070xt"}]}]},
        "strategy": {"deployments": [{"model": "qwen3-0.6b", "weight_precision": "BF16",
                                       "context_tokens": 1024, "devices": ["gpu0"]}],
                     "serving": {"max_concurrency": 2, "continuous_batching": True, "kv_budget_gib": 0.125}},
        "evaluation": {"constraints": [], "duration_s": 10, "timeout_s": 2,
                       "goodput_policy": "both_thresholds_required"},
        "execution": {"seed": 42, "repeat": 0, "split": "development", "holdout": None,
                      "initial_state": "empty", "warmup": None, "max_simulated_ms": 10000},
    }


def body_fixture():
    return json.dumps(agent_fixture(), ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


class AgentWorkerTests(unittest.TestCase):
    def test_binary_worker_preserves_unicode_and_original_body_in_ascii_environment(self):
        raw = body_fixture()
        with patch.dict(os.environ, {"PYTHONIOENCODING": "ascii"}):
            packet = run_agent_isolated(raw, 5)
        self.assertTrue(packet["ok"], packet)
        response = packet["response"]
        self.assertEqual(response["client_request_id"], "接口-α")
        self.assertEqual(response["input_sha256"], "sha256:" + hashlib.sha256(raw).hexdigest())
        self.assertEqual(response["requests"][0]["request_id"], "原始-r1")
        validate_agent_response(raw, response)

    def test_worker_timeout_and_failed_launch_never_publish_partial_or_private_data(self):
        cases = ((subprocess.TimeoutExpired("worker", 2, stderr=b"/private/secret traceback"), "AGENT_TIMEOUT"),
                 (OSError("/private/secret"), "AGENT_WORKER_FAILED"))
        for exception, code in cases:
            with self.subTest(code=code), patch("mlsys_sim.agent_service.subprocess.run", side_effect=exception):
                packet = run_agent_isolated(body_fixture(), 2)
                self.assertEqual(packet["error"]["code"], code)
                self.assertNotIn("response", packet)
                self.assertNotIn("private", json.dumps(packet))

    def test_worker_invalid_exit_envelope_or_response_is_service_failure(self):
        outputs = ((1, b"private traceback"), (0, b"no-json"), (0, b"{}"),
                   (0, b'{"ok":1,"response":{}}'), (0, b'{"ok":true,"response":{}}'),
                   (0, b'{"ok":false,"error":{"message":"private"}}'),
                   (0, b'{"ok":false,"error":{"code":"x","message":"x","traceback":"private"}}'))
        for code, stdout in outputs:
            with self.subTest(code=code, stdout=stdout):
                result = subprocess.CompletedProcess(["worker"], code, stdout=stdout, stderr=b"private")
                with patch("mlsys_sim.agent_service.subprocess.run", return_value=result):
                    packet = run_agent_isolated(body_fixture(), 2)
                self.assertEqual(packet["error"]["code"], "AGENT_WORKER_FAILED")
                self.assertNotIn("private", json.dumps(packet))

    def test_worker_rejects_out_of_domain_timeout_and_body_before_launch(self):
        with patch("mlsys_sim.agent_service.subprocess.run") as spawn:
            for timeout in (0, -1, True, float("inf"), MAX_AGENT_WALL_S + 1, 10**10_000):
                self.assertFalse(run_agent_isolated(body_fixture(), timeout)["ok"])
            self.assertFalse(run_agent_isolated(b"", 1)["ok"])
            self.assertFalse(run_agent_isolated(b" " * (MAX_AGENT_BODY_BYTES + 1), 1)["ok"])
            spawn.assert_not_called()

    def test_catalog_limits_are_fresh_and_have_no_public_listener(self):
        first = get_agent_service_catalog()
        self.assertEqual(first["http"]["bind"], "127.0.0.1")
        self.assertEqual(first["http"]["max_body_bytes"], 2 * 1024**2)
        self.assertEqual(first["http"]["max_wall_timeout_s"], 1800)
        first["http"]["max_concurrent_runs"] = 999
        self.assertEqual(get_agent_service_catalog()["http"]["max_concurrent_runs"], 1)


class AgentHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.server = create_agent_server(0)
        except PermissionError as error:
            raise unittest.SkipTest(f"Loopback socket unavailable: {error}")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.headers), response.read()
        finally:
            connection.close()

    def post(self, raw=None):
        return self.request("POST", "/v1/simulate", raw if raw is not None else body_fixture(),
                            {"Content-Type": "application/json"})

    def test_catalog_and_head_are_local_json_with_shared_security_headers(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        status, headers, body = self.request("GET", "/v1/catalog")
        self.assertEqual(status, 200)
        catalog = json.loads(body)
        self.assertEqual(catalog["catalog_version"], CATALOG_VERSION)
        self.assertEqual(catalog["http"]["simulate_path"], "/v1/simulate")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        status, headers, body = self.request("HEAD", "/v1/catalog")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")
        self.assertGreater(int(headers["Content-Length"]), 0)

    def test_real_http_execution_preserves_exact_utf8_body_and_original_ids(self):
        raw = body_fixture()
        status, _, body = self.post(raw)
        self.assertEqual(status, 200, body)
        response = json.loads(body)
        self.assertNotIn("ok", response)
        self.assertEqual(response["evidence_level"], "simulated")
        self.assertEqual(response["status"], "succeeded")
        self.assertEqual(response["requests"][0]["output_tokens"], 2)
        self.assertEqual(response["input_sha256"], "sha256:" + hashlib.sha256(raw).hexdigest())
        validate_agent_response(raw, response)

    def test_handled_modeled_oom_is_http200_with_failed_records(self):
        request = agent_fixture()
        request["strategy"]["serving"]["kv_budget_gib"] = 16
        raw = json.dumps(request).encode()
        status, _, body = self.post(raw)
        self.assertEqual(status, 200, body)
        response = json.loads(body)
        self.assertEqual(response["status"], "failed")
        self.assertEqual(len(response["requests"]), 1)
        self.assertEqual(response["requests"][0]["status"], "not_started")
        validate_agent_response(raw, response)

    def test_unconfirmed_ids_and_unsupported_policy_are_422_not_fallback(self):
        changes = (("catalog", "to-be-agreed", "CATALOG_VERSION_MISMATCH"),
                   ("model", "llm-3b", "UNKNOWN_MODEL_ID"),
                   ("hardware", "w7900d", "UNKNOWN_HARDWARE_ID"),
                   ("priority", "short_first", "UNSUPPORTED_AGENT_CONFIGURATION"))
        for kind, replacement, code in changes:
            with self.subTest(kind=kind):
                request = agent_fixture()
                if kind == "catalog":
                    request["catalog_version"] = replacement
                elif kind == "model":
                    request["strategy"]["deployments"][0]["model"] = replacement
                elif kind == "hardware":
                    request["infra"]["nodes"][0]["devices"][0]["hardware"] = replacement
                else:
                    request["strategy"]["scheduling"] = {"priority": replacement}
                status, _, body = self.post(json.dumps(request).encode())
                self.assertEqual(status, 422, body)
                self.assertEqual(json.loads(body)["error"]["code"], code)
                self.assertNotIn("requests", json.loads(body))

    def test_invalid_json_utf8_duplicates_and_shapes_have_explicit_errors(self):
        for raw in (b"{", b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1,"x":2}', b"\xff"):
            with self.subTest(raw=raw):
                status, _, body = self.post(raw)
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body)["error"]["code"], "INVALID_AGENT_JSON")
        status, _, body = self.post(b"[]")
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"]["code"], "INVALID_AGENT_REQUEST")

    def test_header_body_limits_and_closed_path_allowlist(self):
        for path in ("/", "/api/presets", "/api/simulate", "/../pyproject.toml", "/v1/catalog?path=x"):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(json.loads(body)["error"]["code"], "NOT_FOUND")
        status, _, _ = self.request("POST", "/v1/simulate", b"{}", {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        status, _, _ = self.request("POST", "/v1/simulate", b"{}",
                                    {"Content-Type": "application/json", "Content-Length": str(MAX_AGENT_BODY_BYTES + 1)})
        self.assertEqual(status, 413)
        for length in ("-1", "x", "9" * 5000):
            status, _, _ = self.request("POST", "/v1/simulate", b"{}",
                {"Content-Type": "application/json", "Content-Length": length})
            self.assertIn(status, (400, 413))

    def test_duplicate_length_host_origin_headers_and_transfer_encoding_are_rejected(self):
        port = self.server.server_port
        cases = (([("Content-Length", "2"), ("Content-Length", "2")], 400),
                 ([("Content-Length", "2"), ("Transfer-Encoding", "chunked")], 400),
                 ([("Content-Length", "2"), ("Host", f"127.0.0.1:{port}")], 403),
                 ([("Content-Length", "2"), ("Origin", f"http://127.0.0.1:{port}"),
                   ("Origin", f"http://127.0.0.1:{port}")], 403))
        for headers, expected in cases:
            with self.subTest(headers=headers):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    connection.putrequest("POST", "/v1/simulate")
                    connection.putheader("Content-Type", "application/json")
                    for name, value in headers:
                        connection.putheader(name, value)
                    connection.endheaders(b"{}")
                    response = connection.getresponse()
                    self.assertEqual(response.status, expected)
                    response.read()
                finally:
                    connection.close()

    def test_cross_origin_access_and_cors_are_not_enabled(self):
        for headers in ({"Host": "example.com"}, {"Origin": "https://example.com"},
                        {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                status, _, _ = self.request("GET", "/v1/catalog", headers=headers)
                self.assertEqual(status, 403)
        status, headers, _ = self.request("OPTIONS", "/v1/simulate")
        self.assertEqual(status, 405)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_busy_response_has_retry_after_without_queuing_a_second_run(self):
        original = self.server.agent_runner
        started, release = threading.Event(), threading.Event()
        first = []

        def slow_runner(raw, timeout):
            started.set()
            release.wait(timeout=5)
            return original(raw, timeout)

        self.server.agent_runner = slow_runner
        thread = threading.Thread(target=lambda: first.append(self.post()))
        thread.start()
        try:
            self.assertTrue(started.wait(timeout=3))
            status, headers, body = self.post()
            self.assertEqual(status, 429)
            self.assertEqual(headers["Retry-After"], "1")
            self.assertEqual(json.loads(body)["error"]["code"], "AGENT_BUSY")
        finally:
            release.set()
            thread.join(timeout=5)
            self.server.agent_runner = original
        self.assertEqual(first[0][0], 200)

    def test_wall_timeout_and_worker_fault_are_transport_failures_not_fake_run_records(self):
        original = self.server.agent_runner
        try:
            for code, expected in (("AGENT_TIMEOUT", 504), ("AGENT_WORKER_FAILED", 500)):
                self.server.agent_runner = lambda raw, timeout: {"ok": False, "error": {
                    "code": code, "message": "Bounded worker failure", "field": None}}
                status, _, body = self.post()
                self.assertEqual(status, expected)
                response = json.loads(body)
                self.assertEqual(response["error"]["code"], code)
                self.assertNotIn("requests", response)
        finally:
            self.server.agent_runner = original


if __name__ == "__main__":
    unittest.main()
