"""Loopback HTTP transport for the versioned Agent simulation protocol.

The local Web form remains separate.  This service accepts exact JSON bytes and
runs the existing Agent adapter in a bounded child process, without importing a
course-platform package or exposing filesystem/profile execution inputs.
"""

import argparse
import json
import math
import subprocess
import sys
from typing import Callable, TypeGuard
from urllib.parse import urlsplit

from .agent_adapter import MAX_AGENT_BODY_BYTES, MAX_AGENT_WALL_S
from .agent_catalog import get_agent_catalog
from .agent_contract import validate_agent_response
from .schema import SimulationError
from .web import WebHandler, WebServer


def _worker_error(code: str, message: str, field: str | None = None) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, "field": field}}


def _valid_error(value: object) -> TypeGuard[dict]:
    return (isinstance(value, dict) and not set(value) - {"code", "message", "field"}
            and all(isinstance(value.get(name), str) and value[name].strip() for name in ("code", "message"))
            and (value.get("field") is None or isinstance(value["field"], str)))


def run_agent_isolated(raw: bytes, timeout_s: float) -> dict:
    """Preserve exact body bytes and keep the wall limit independent of CPU work."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_AGENT_BODY_BYTES:
        return _worker_error("INVALID_AGENT_JSON", "Expected a bounded UTF-8 JSON body", "body")
    if (type(timeout_s) not in (int, float) or not 0 < timeout_s <= MAX_AGENT_WALL_S
            or not math.isfinite(timeout_s)):
        return _worker_error("INVALID_AGENT_REQUEST", "Wall timeout exceeds the service domain",
                             "evaluation.timeout_s")
    try:
        process = subprocess.run([sys.executable, "-m", "mlsys_sim.agent_adapter"],
                                 input=raw, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return _worker_error("AGENT_TIMEOUT", "Simulation exceeded its requested wall-time limit",
                             "evaluation.timeout_s")
    except OSError:
        return _worker_error("AGENT_WORKER_FAILED", "Unable to start the simulation worker")
    if process.returncode != 0:
        return _worker_error("AGENT_WORKER_FAILED", "Simulation worker failed")
    try:
        packet = json.loads(process.stdout)
        if not isinstance(packet, dict) or type(packet.get("ok")) is not bool:
            raise ValueError("Invalid worker envelope")
        if packet["ok"]:
            if set(packet) != {"ok", "response"}:
                raise ValueError("Invalid success envelope")
            validate_agent_response(raw, packet["response"])
        elif set(packet) != {"ok", "error"} or not _valid_error(packet["error"]):
            raise ValueError("Invalid error envelope")
        return packet
    except (ValueError, TypeError, UnicodeError, RecursionError):
        # Never publish worker stderr, tracebacks or partial invalid records.
        return _worker_error("AGENT_WORKER_FAILED", "Invalid simulation worker response")


def _status_for_error(code: str) -> int:
    if code == "INVALID_AGENT_JSON":
        return 400
    if code == "AGENT_TIMEOUT":
        return 504
    if code == "INVALID_AGENT_REQUEST" or code.startswith(("UNKNOWN_", "UNSUPPORTED_", "UNCONFIRMED_")):
        return 422
    if code in {"CATALOG_VERSION_MISMATCH", "INVALID_CONFIG", "INVALID_DEVICE_ID"}:
        return 422
    return 500


def _simulation_error(error: SimulationError) -> dict:
    message = str(error)
    field = None
    if ":" in message:
        prefix = message.split(":", 1)[0]
        if prefix and len(prefix) <= 256 and not any(char.isspace() for char in prefix):
            field = prefix
    return {"code": error.code, "message": message, "field": field}


def get_agent_service_catalog() -> dict:
    catalog = get_agent_catalog()
    catalog["http"] = {
        "simulate_path": "/v1/simulate", "catalog_path": "/v1/catalog",
        "max_body_bytes": MAX_AGENT_BODY_BYTES, "max_wall_timeout_s": MAX_AGENT_WALL_S,
        "max_concurrent_runs": 1, "bind": "127.0.0.1",
        "authentication": "none; loopback-only",
        "busy": {"status": 429, "code": "AGENT_BUSY", "retry_after_s": 1},
        "execution": "synchronous; no automatic retry, cancellation or durable job identity",
    }
    return catalog


class AgentServer(WebServer):
    def __init__(self, port: int, runner: Callable[[bytes, float], dict] = run_agent_isolated):
        super().__init__(port)
        self.agent_runner = runner
        self.RequestHandlerClass = AgentHandler


class AgentHandler(WebHandler):
    server: AgentServer
    server_version = "MLSysAgentLocal"
    _retry_after: str | None = None

    def end_headers(self) -> None:
        if self._retry_after is not None:
            self.send_header("Retry-After", self._retry_after)
        super().end_headers()

    def _error(self, status: int, code: str, message: str, field: str | None = None) -> None:
        self._retry_after = "1" if status == 429 else None
        try:
            self._json(status, {"error": {"code": code, "message": message, "field": field}})
        finally:
            self._retry_after = None

    def _local(self) -> bool:
        if len(self.headers.get_all("Origin", [])) > 1:
            self._error(403, "INVALID_ORIGIN", "One Origin header is allowed")
            return False
        return super()._local()

    def do_GET(self) -> None:
        if not self._local():
            return
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or parsed.path != "/v1/catalog":
            self._error(404, "NOT_FOUND", "Unknown Agent endpoint")
            return
        self._json(200, get_agent_service_catalog())

    def do_POST(self) -> None:
        from .agent_adapter import decode_agent_request, validate_agent_support

        if not self._local():
            return
        if self.path != "/v1/simulate":
            self._error(404, "NOT_FOUND", "Unknown Agent endpoint")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
            self._error(400, "INVALID_AGENT_BODY", "One Content-Length header is required", "body")
            return
        length_text = lengths[0]
        if not length_text.isascii() or not length_text.isdigit():
            self._error(400, "INVALID_AGENT_BODY", "Content-Length must be decimal bytes", "body")
            return
        if len(length_text) > 20:
            self._error(413, "AGENT_BODY_TOO_LARGE", "Content-Length exceeds the service domain", "body")
            return
        length = int(length_text)
        if not 0 < length <= MAX_AGENT_BODY_BYTES:
            self._error(413, "AGENT_BODY_TOO_LARGE", f"JSON body must be 1–{MAX_AGENT_BODY_BYTES} bytes", "body")
            return
        if len(self.headers.get_all("Content-Type", [])) != 1 or self.headers.get_content_type() != "application/json":
            self._error(415, "INVALID_CONTENT_TYPE", "One application/json Content-Type header is required", "body")
            return
        if not self.server.simulation_slot.acquire(blocking=False):
            self._error(429, "AGENT_BUSY", "An Agent simulation is already running; retry after it finishes")
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                self._error(400, "INVALID_AGENT_BODY", "Incomplete request body", "body")
                return
            request = decode_agent_request(raw)
            validate_agent_support(request)
            packet = self.server.agent_runner(raw, request["evaluation"]["timeout_s"])
            if packet.get("ok") is True:
                validate_agent_response(raw, packet["response"])
                self._json(200, packet["response"])
            else:
                error = packet.get("error")
                if not _valid_error(error):
                    self._error(500, "AGENT_WORKER_FAILED", "Invalid simulation worker response")
                    return
                self._error(_status_for_error(error["code"]), error["code"], error["message"], error.get("field"))
        except SimulationError as error:
            detail = _simulation_error(error)
            self._error(_status_for_error(detail["code"]), detail["code"], detail["message"], detail["field"])
        except OSError:
            # A disconnected client does not cancel the completed/running job.
            self.close_connection = True
        except Exception:
            self._error(500, "AGENT_SERVICE_FAILED", "Agent service failed")
        finally:
            self.server.simulation_slot.release()

    def do_OPTIONS(self) -> None:
        if self._local():
            self._error(405, "METHOD_NOT_ALLOWED", "Only local GET and JSON POST are supported")


def create_agent_server(port: int = 8877, *,
                        runner: Callable[[bytes, float], dict] = run_agent_isolated) -> AgentServer:
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Port must be an integer in [0, 65535]")
    return AgentServer(port, runner)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local versioned Agent simulation service")
    parser.add_argument("--port", type=int, default=8877, help="Loopback port (default: 8877)")
    args = parser.parse_args(argv)
    try:
        server = create_agent_server(args.port)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"MLSys Agent: http://127.0.0.1:{server.server_port}/v1/simulate", flush=True)
    print("GET /v1/catalog for explicit identities and limits. Analytical CPU simulation only. Ctrl+C to stop.",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
