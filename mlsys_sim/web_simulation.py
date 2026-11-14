"""Bounded form adapter for the local UI; all service costs use the core engine."""

from dataclasses import asdict, fields
import json
import math
import sys
import time
from typing import Any

from .config import experiment_from_dict
from .engine import simulate
from .memory import MemoryPlanner
from .qwen3 import model_from_qwen3_config
from .schema import ModelSpec, SimulationError
from .serialization import experiment_dict
from .workload import GENERATOR_VERSION, _rng, canonical_sha256


GIB = 1024**3
MAX_REQUESTS = 500
MAX_EVENTS = 100_000
MAX_SIMULATED_MS = 600_000
MAX_MEMORY_POINTS = 2048
_GEOMETRIES = (
    ("qwen3-0.6b", "Qwen3-0.6B", 28, 1024, 3072, 16, 8),
    ("qwen3-1.7b", "Qwen3-1.7B", 28, 2048, 6144, 16, 8),
    ("qwen3-4b", "Qwen3-4B", 36, 2560, 9728, 32, 8),
)


def _models() -> list[dict]:
    models = []
    for identity, name, layers, hidden, intermediate, heads, kv_heads in _GEOMETRIES:
        config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                  "num_hidden_layers": layers, "hidden_size": hidden,
                  "intermediate_size": intermediate, "num_attention_heads": heads,
                  "num_key_value_heads": kv_heads, "head_dim": 128, "vocab_size": 151936,
                  "tie_word_embeddings": True, "torch_dtype": "bfloat16",
                  "hidden_act": "silu", "attention_bias": False}
        model = model_from_qwen3_config(config, name=name, revision="course-7df2dae493eb-geometry")
        models.append({"id": identity, "label": name, "model": asdict(model),
                       "source": "Hello MLSys course declared geometry; no weights loaded"})
    return models


def presets() -> dict:
    """Return independent JSON-ready defaults; preset rates are declared assumptions."""
    return {
        "models": _models(),
        "hardware": [
            {"id": "rx9070xt", "label": "RX9070XT · 课程名义参数", "memory_gib": 16,
             "compute_tflops": 195, "bandwidth_gbps": 640},
            {"id": "teaching24", "label": "24 GiB · 教学配置", "memory_gib": 24,
             "compute_tflops": 195, "bandwidth_gbps": 640},
            {"id": "teaching48", "label": "48 GiB · 教学配置", "memory_gib": 48,
             "compute_tflops": 195, "bandwidth_gbps": 640},
        ],
        "defaults": {
            "model_id": "qwen3-1.7b",
            "hardware": {"memory_gib": 16, "compute_tflops": 195, "bandwidth_gbps": 640},
            "workload": {"requests": 48, "input_tokens": 256, "output_tokens": 32,
                         "arrival": "periodic", "rate_rps": 20, "seed": 42},
            "strategy": {"scheduler": "continuous_batching", "max_batch_size": 8,
                         "max_batch_tokens": 2048, "chunked_prefill": True,
                         "prefill_chunk_tokens": 512, "mixed_batching": True},
            "memory": {"kv_pool_gib": 2},
        },
        "limits": {"max_requests": MAX_REQUESTS, "max_input_tokens": 8192,
                   "max_output_tokens": 512, "max_batch_size": 128,
                   "max_batch_tokens": 32768, "max_events": MAX_EVENTS,
                   "max_simulated_ms": MAX_SIMULATED_MS, "wall_timeout_s": 20},
        "source": "analytical_uncalibrated", "hardware_validated": False,
        "hardware_note": "Compute/bandwidth are course nominal assumptions, not measurements.",
    }


def _error(message: str) -> None:
    raise SimulationError("INVALID_WEB_CONFIG", message)


def _object(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) - allowed:
        _error(f"{name} must be an object; allowed fields: {', '.join(sorted(allowed))}")
    return value


def _number(value: Any, name: str, low: float, high: float, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or not low <= value <= high or (integer and not isinstance(value, int)):
        _error(f"{name} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        _error(f"{name} must be true or false")
    return value


def build_experiment(payload: dict) -> tuple[dict, dict]:
    """Build a fixed-size open-loop cohort; preserve the original engine contract."""
    _object(payload, "configuration", {"model_id", "model", "hardware", "workload", "strategy", "memory"})
    defaults = presets()["defaults"]
    model = next((item["model"] for item in _models() if item["id"] == payload.get("model_id")), None)
    if "model" in payload:
        model = _object(payload["model"], "model", {item.name for item in fields(ModelSpec)})
        for name, default in asdict(ModelSpec()).items():
            value = model.get(name, default)
            if isinstance(default, bool):
                _boolean(value, f"model.{name}")
            elif isinstance(default, int):
                _number(value, f"model.{name}", 1, 1_000_000, integer=True)
            elif isinstance(default, float):
                _number(value, f"model.{name}", 0.125, 8)
            elif not isinstance(value, str) or len(value) > 256 or any(ord(char) < 32 for char in value):
                _error(f"model.{name} must be bounded text without control characters")
        if model.get("num_layers", 4) > 256:
            _error("model.num_layers must not exceed 256")
    elif model is None:
        _error("Select a supported model_id or supply custom model geometry")

    sections = {}
    for section in ("hardware", "workload", "strategy", "memory"):
        value = _object(payload.get(section, {}), section, set(defaults[section]))
        sections[section] = {**defaults[section], **value}
    hardware, load, strategy, memory = (sections[key] for key in ("hardware", "workload", "strategy", "memory"))
    memory_gib = _number(hardware["memory_gib"], "hardware.memory_gib", 0.25, 1024)
    compute = _number(hardware["compute_tflops"], "hardware.compute_tflops", 0.01, 10000)
    bandwidth = _number(hardware["bandwidth_gbps"], "hardware.bandwidth_gbps", 0.1, 100000)
    count = int(_number(load["requests"], "workload.requests", 1, MAX_REQUESTS, integer=True))
    inputs = int(_number(load["input_tokens"], "workload.input_tokens", 1, 8192, integer=True))
    outputs = int(_number(load["output_tokens"], "workload.output_tokens", 1, 512, integer=True))
    seed = int(_number(load["seed"], "workload.seed", -(2**53 - 1), 2**53 - 1, integer=True))
    rate = _number(load["rate_rps"], "workload.rate_rps", 0.1, 10000)
    arrival = load["arrival"]
    if arrival not in ("periodic", "poisson", "burst"):
        _error("workload.arrival must be periodic, poisson or burst")
    if strategy["scheduler"] not in ("continuous_batching", "fifo"):
        _error("strategy.scheduler must be continuous_batching or fifo")
    for name, high in (("max_batch_size", 128), ("max_batch_tokens", 32768), ("prefill_chunk_tokens", 8192)):
        _number(strategy[name], f"strategy.{name}", 1, high, integer=True)
    for name in ("chunked_prefill", "mixed_batching"):
        _boolean(strategy[name], f"strategy.{name}")
    kv_gib = _number(memory["kv_pool_gib"], "memory.kv_pool_gib", 1 / 1024, 1024)

    clock = 0.0
    rng = _rng(seed, 0, "arrivals")
    rows = []
    for index in range(count):
        if arrival == "poisson":
            step = -math.log1p(-rng.random()) * 1000 / rate
            if step <= 0 or clock + step <= clock:
                _error("Poisson arrival step cannot advance the clock")
            clock += step
        elif arrival == "periodic":
            clock = index * 1000 / rate
        if clock >= MAX_SIMULATED_MS:
            _error(f"Workload arrival span exceeds {MAX_SIMULATED_MS} ms; increase the offered rate")
        rows.append({"request_id": f"web-{index + 1:04d}", "arrival_ms": clock,
                     "input_tokens": inputs, "output_tokens": outputs})
    experiment = {
        "schema_version": 1, "workload": rows,
        "infra": {"devices": [{"id": "gpu0", "name": "Declared single GPU",
                                "memory_bytes": int(memory_gib * GIB), "compute_tflops": compute,
                                "memory_bandwidth_gbps": bandwidth}]},
        "strategy": {"model": model, **strategy, "memory": {
            "kv_pool_bytes": int(kv_gib * GIB), "workspace_base_bytes": 64 * 1024**2,
            "workspace_bytes_per_token": 1024, "runtime_bytes": 128 * 1024**2,
            "safety_margin_bytes": 256 * 1024**2}},
        "calibration": {"backend": "analytical", "profile_version": "web-uncalibrated-v1",
                        "compute_efficiency": 0.5, "bandwidth_efficiency": 0.5},
        "evaluation": {"drain": True},
        "execution": {"seed": seed, "max_events": MAX_EVENTS,
                      "max_simulated_ms": MAX_SIMULATED_MS, "detailed_events": True},
    }
    validated = experiment_from_dict(experiment)
    return experiment_dict(validated), {
        "generator_version": GENERATOR_VERSION, "workload_rule": "fixed_request_count_then_drain",
        "arrival_process": arrival, "seed": seed, "request_count": count,
        "requests_sha256": canonical_sha256(rows),
    }


def throughput_chart(requests: list[dict], elapsed_ms: float, bins: int = 40) -> list[dict]:
    """Count actual emission timestamps; every bin's denominator is its true width."""
    if elapsed_ms <= 0:
        return []
    width = elapsed_ms / bins
    counts = [0] * bins
    for request in requests:
        for when in request["token_times_ms"]:
            if 0 <= when <= elapsed_ms:
                index = min(bins - 1, int(when / width))
                counts[index] += 1
    rows = []
    for index, count in enumerate(counts):
        start, end = index * width, min((index + 1) * width, elapsed_ms)
        rows.append({"start_ms": start, "end_ms": end, "tokens": count,
                     "tokens_per_s": count * 1000 / (end - start)})
    return rows


def _memory_row(when: float, row: dict) -> dict:
    names = ("device_id", "budget_peak_bytes", "physical_peak_bytes", "physical_reserved_bytes",
             "fixed_budget_bytes", "workspace_peak_bytes", "weights_bytes", "device_memory_bytes")
    return {"time_ms": when, **{name: row[name] for name in names},
            "used_kv_bytes": row["kv_used_bytes"], "kv_pool_bytes": row["kv_pool_reserved_bytes"]}


def memory_chart(initial: list[dict], events: list[dict], final: list[dict], elapsed_ms: float) -> tuple[list, bool]:
    points = [_memory_row(0.0, row) for row in initial]
    points.extend(_memory_row(event["timestamp_ms"], row)
                  for event in events for row in event.get("memory", []))
    points.extend(_memory_row(elapsed_ms, row) for row in final)
    if len(points) <= MAX_MEMORY_POINTS:
        return points, False
    # Preserve both ends and every bucket's KV and budget maxima; no averaging of peaks.
    chosen = {0, len(points) - 1}
    bucket = math.ceil(len(points) / ((MAX_MEMORY_POINTS - 2) // 2))
    for start in range(0, len(points), bucket):
        indices = range(start, min(start + bucket, len(points)))
        chosen.add(max(indices, key=lambda index: points[index]["used_kv_bytes"]))
        chosen.add(max(indices, key=lambda index: points[index]["budget_peak_bytes"]))
    return [points[index] for index in sorted(chosen)], True


def run_web_simulation(payload: dict) -> dict:
    """Return an original engine result plus presentation views, never a GPU claim."""
    started = time.monotonic()
    experiment = None
    try:
        experiment, provenance = build_experiment(payload)
        configured = experiment_from_dict(experiment)
        initial = MemoryPlanner(configured).snapshot()
        run = simulate(configured)
        result = run.result
        elapsed = result["simulated_ms"]
        memory, thinned = memory_chart(initial, run.events, result["memory"], elapsed)
        status = ("COMPLETED" if result["status"] == "COMPLETED"
                  and result["counts"]["completed"] == len(run.requests) else "FAILED")
        return {
            "ok": True, "experiment": experiment, "result": result, "requests": run.requests,
            "charts": {
                "latency": {"ttft_ms": result["latency_ms"]["ttft"],
                            "tpot_ms": result["latency_ms"]["tpot"],
                            "e2e_ms": result["latency_ms"]["end_to_end"]},
                "throughput": throughput_chart(run.requests, elapsed), "memory": memory,
                "memory_capacity_bytes": {row["device_id"]: row["device_memory_bytes"] for row in initial},
            },
            "meta": {"source": "analytical_uncalibrated", "hardware_validated": False,
                     "execution_outcome": status, "elapsed_ms": elapsed,
                     "wall_elapsed_ms": (time.monotonic() - started) * 1000,
                     "throughput_rule": "actual_token_emissions_per_bin_width",
                     "memory_rule": "cumulative_budget_peak_and_logical_kv_not_additive",
                     "memory_thinned": thinned, "events_processed": result["events_processed"],
                     "workload": provenance},
        }
    except SimulationError as error:
        packet: dict[str, Any] = {"ok": False, "error": {"code": error.code, "message": str(error)}}
        if experiment is not None:
            packet["experiment"] = experiment
        return packet


def main() -> int:
    """Private JSON worker used by the loopback server's wall-clock timeout."""
    try:
        payload = json.load(sys.stdin)
        response = run_web_simulation(payload)
    except (ValueError, TypeError, RecursionError):
        response = {"ok": False, "error": {"code": "INVALID_WEB_CONFIG", "message": "Invalid JSON configuration"}}
    # The private pipe must work under non-UTF-8 locales; JSON reconstructs Unicode on decode.
    print(json.dumps(response, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
