"""Dependency-free loopback Web UI for bounded analytical CPU experiments."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import socket
import subprocess
import sys
import threading
from typing import Callable
from urllib.parse import urlsplit

from .web_simulation import presets


MAX_BODY_BYTES = 64 * 1024
WALL_TIMEOUT_S = 20
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}


def run_isolated(payload: dict) -> dict:
    """A separate process makes the wall-time limit independent of engine progress."""
    try:
        process = subprocess.run([sys.executable, "-m", "mlsys_sim.web_simulation"],
            input=json.dumps(payload, allow_nan=False), capture_output=True, text=True,
            timeout=WALL_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": {"code": "WEB_TIMEOUT",
                "message": f"Simulation exceeded {WALL_TIMEOUT_S}s; reduce requests or token lengths"}}
    if process.returncode != 0:
        return {"ok": False, "error": {"code": "WEB_WORKER_FAILED", "message": "Simulation worker failed"}}
    try:
        response = json.loads(process.stdout)
        if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
            raise ValueError("Invalid worker response")
        return response
    except (ValueError, TypeError):
        return {"ok": False, "error": {"code": "WEB_WORKER_FAILED", "message": "Invalid simulation response"}}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


class WebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, runner: Callable[[dict], dict] = run_isolated):
        self.runner = runner
        self.simulation_slot = threading.BoundedSemaphore(1)
        super().__init__(("127.0.0.1", port), WebHandler)

    def get_request(self) -> tuple[socket.socket, tuple]:
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address


class WebHandler(BaseHTTPRequestHandler):
    server: WebServer
    server_version = "MLSysLocal"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                         "base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, value: dict) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {"ok": False, "error": {"code": code, "message": message}})

    def _local(self) -> bool:
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update({"127.0.0.1", "localhost"})
        host = self.headers.get("Host", "")
        if host not in allowed or len(self.headers.get_all("Host", [])) != 1:
            self._error(403, "LOCAL_ONLY", "Use the printed localhost URL")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{host}":
            self._error(403, "INVALID_ORIGIN", "Cross-origin requests are not allowed")
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._error(403, "INVALID_ORIGIN", "Cross-site requests are not allowed")
            return False
        return True

    def do_GET(self) -> None:
        if not self._local():
            return
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            self._error(404, "NOT_FOUND", "Unknown endpoint")
            return
        if parsed.path == "/api/presets":
            self._json(200, presets())
            return
        asset = ASSETS.get(parsed.path)
        if asset is None:
            self._error(404, "NOT_FOUND", "Unknown endpoint")
            return
        name, content_type = asset
        try:
            body = files("mlsys_sim").joinpath("web_assets", name).read_bytes()
        except FileNotFoundError:
            self._error(503, "ASSET_MISSING", "Packaged Web UI asset is missing")
            return
        self._send(200, body, content_type)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        if not self._local():
            return
        if self.path != "/api/simulate":
            self._error(404, "NOT_FOUND", "Unknown endpoint")
            return
        if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
            self._error(400, "INVALID_BODY", "One Content-Length header is required")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "INVALID_BODY", "Invalid Content-Length")
            return
        if not 0 < length <= MAX_BODY_BYTES:
            self._error(413, "BODY_TOO_LARGE", f"JSON body must be 1–{MAX_BODY_BYTES} bytes")
            return
        if self.headers.get_content_type() != "application/json":
            self._error(415, "INVALID_CONTENT_TYPE", "Content-Type must be application/json")
            return
        if not self.server.simulation_slot.acquire(blocking=False):
            self._error(429, "WEB_BUSY", "A simulation is already running; try again after it finishes")
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete JSON body")
            payload = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            if not isinstance(payload, dict):
                raise ValueError("Configuration must be an object")
            response = self.server.runner(payload)
            self._json(200 if response.get("ok") else 422, response)
        except (ValueError, UnicodeError, RecursionError, TimeoutError):
            self._error(400, "INVALID_JSON", "Body must be a complete finite JSON object without duplicate fields")
        finally:
            self.server.simulation_slot.release()

    def do_OPTIONS(self) -> None:
        self._error(405, "METHOD_NOT_ALLOWED", "Only local GET and JSON POST are supported")


def create_server(port: int = 8765, *, runner: Callable[[dict], dict] = run_isolated) -> WebServer:
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Port must be an integer in [0, 65535]")
    return WebServer(port, runner)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Web interface for analytical CPU simulation")
    parser.add_argument("--port", type=int, default=8765, help="Loopback port (default: 8765)")
    args = parser.parse_args(argv)
    try:
        server = create_server(args.port)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"MLSys Simulator: http://127.0.0.1:{server.server_port}", flush=True)
    print("Analytical CPU predictions; no model weights or GPU runtime are loaded. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
