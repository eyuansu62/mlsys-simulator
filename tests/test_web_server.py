"""Loopback-only transport and bounded worker behavior for the Web UI."""

import http.client
from importlib.resources import files
import json
import subprocess
import threading
import unittest
from unittest.mock import patch

from mlsys_sim.web import MAX_BODY_BYTES, create_server, run_isolated
from mlsys_sim.web_simulation import presets, run_web_simulation


def small_configuration():
    payload = presets()["defaults"]
    payload["workload"].update(requests=2, input_tokens=16, output_tokens=2)
    return payload


class WorkerTests(unittest.TestCase):
    def test_isolated_worker_runs_without_gpu_dependencies(self):
        result = run_isolated(small_configuration())
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["result"]["counts"]["completed"], 2)
        self.assertFalse(result["meta"]["hardware_validated"])

    def test_private_worker_json_preserves_unicode_under_ascii_stdio(self):
        payload = small_configuration()
        payload.pop("model_id")
        payload["model"] = {"name": "课堂模型 λ"}
        with patch.dict("os.environ", {"PYTHONIOENCODING": "ascii"}):
            result = run_isolated(payload)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["experiment"]["strategy"]["model"]["name"], "课堂模型 λ")
        self.assertEqual(result["result"]["counts"]["completed"], 2)

    def test_timeout_is_reported_without_partial_fake_metrics(self):
        with patch("mlsys_sim.web.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 20)):
            result = run_isolated(small_configuration())
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "WEB_TIMEOUT")
        self.assertNotIn("charts", result)

    def test_invalid_worker_exit_and_output_are_not_relabelled_success(self):
        for exit_code, stdout in ((1, "private traceback"), (0, "not json"), (0, "{}")):
            with self.subTest(exit_code=exit_code, stdout=stdout):
                returned = subprocess.CompletedProcess(["worker"], exit_code, stdout=stdout, stderr="private")
                with patch("mlsys_sim.web.subprocess.run", return_value=returned):
                    result = run_isolated(small_configuration())
                self.assertEqual(result["error"]["code"], "WEB_WORKER_FAILED")
                self.assertNotIn("private", json.dumps(result))


class WebServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.server = create_server(0, runner=run_web_simulation)
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

    def test_server_only_binds_ipv4_loopback(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_preset_and_packaged_assets_are_local_and_cache_free(self):
        status, headers, body = self.request("GET", "/api/presets")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(json.loads(body)["defaults"]["model_id"], "qwen3-1.7b")
        for path, filename, content_type in (("/", "index.html", "text/html"),
                                              ("/app.js", "app.js", "text/javascript"),
                                              ("/style.css", "style.css", "text/css")):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertTrue(headers["Content-Type"].startswith(content_type))
                self.assertEqual(body, files("mlsys_sim").joinpath("web_assets", filename).read_bytes())
                self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_post_runs_actual_adapter_and_retains_errors(self):
        status, _, body = self.request("POST", "/api/simulate", json.dumps(small_configuration()),
                                      {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["result"]["counts"]["completed"], 2)
        configuration = small_configuration()
        configuration["hardware"]["memory_gib"] = 0.25
        status, _, body = self.request("POST", "/api/simulate", json.dumps(configuration),
                                      {"Content-Type": "application/json"})
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"]["code"], "STATIC_MEMORY_INFEASIBLE")
        self.assertNotIn("charts", json.loads(body))

    def test_host_and_origin_guards_prevent_remote_page_access(self):
        host = f"127.0.0.1:{self.server.server_port}"
        cases = ({"Host": "example.com"}, {"Origin": "https://example.com"}, {"Origin": "null"},
                 {"Origin": f"http://localhost:{self.server.server_port}"},
                 {"Sec-Fetch-Site": "cross-site"})
        for headers in cases:
            with self.subTest(headers=headers):
                status, _, body = self.request("GET", "/api/presets", headers=headers)
                self.assertEqual(status, 403)
                self.assertFalse(json.loads(body)["ok"])
        status, _, _ = self.request("GET", "/api/presets", headers={"Origin": f"http://{host}"})
        self.assertEqual(status, 200)

    def test_no_arbitrary_static_paths_or_cors(self):
        for path in ("/../pyproject.toml", "/%2e%2e/pyproject.toml", "/api/presets?path=x", "/nope"):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertEqual(json.loads(body)["error"]["code"], "NOT_FOUND")
        status, headers, _ = self.request("OPTIONS", "/api/simulate")
        self.assertEqual(status, 405)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_body_limits_and_finite_unique_json_are_enforced(self):
        for body in ('{"model_id":"qwen3-1.7b","model_id":"qwen3-4b"}',
                     '{"x":NaN}', '{"x":Infinity}', '[]', '{'):
            with self.subTest(body=body):
                status, _, data = self.request("POST", "/api/simulate", body,
                                              {"Content-Type": "application/json"})
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(data)["error"]["code"], "INVALID_JSON")
        status, _, _ = self.request("POST", "/api/simulate", "{}", {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        status, _, _ = self.request("POST", "/api/simulate", "{}",
                                    {"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)})
        self.assertEqual(status, 413)

    def test_only_one_simulation_runs_at_once(self):
        started, release = threading.Event(), threading.Event()
        original = self.server.runner
        first = []

        def slow_runner(payload):
            started.set()
            release.wait(timeout=5)
            return original(payload)

        def initial():
            first.append(self.request("POST", "/api/simulate", json.dumps(small_configuration()),
                                      {"Content-Type": "application/json"}))

        self.server.runner = slow_runner
        thread = threading.Thread(target=initial)
        thread.start()
        try:
            self.assertTrue(started.wait(timeout=3))
            status, _, body = self.request("POST", "/api/simulate", json.dumps(small_configuration()),
                                          {"Content-Type": "application/json"})
            self.assertEqual(status, 429)
            self.assertEqual(json.loads(body)["error"]["code"], "WEB_BUSY")
        finally:
            release.set()
            thread.join(timeout=5)
            self.server.runner = original
        self.assertEqual(first[0][0], 200)


if __name__ == "__main__":
    unittest.main()
