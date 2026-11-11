"""Bounded, replayable open-loop traffic generation, independent of simulation."""

from dataclasses import asdict
import hashlib
import json
import math
import random
from typing import Any

from .schema import RequestSpec, SimulationError


GENERATOR_VERSION = "open-loop-v1"
MAX_REQUESTS = 100_000


def canonical_sha256(value: Any) -> str:
    """Hash JSON data using a portable representation; reject NaN and infinity."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _error(message: str) -> None:
    raise SimulationError("INVALID_WORKLOAD", message)


def _keys(value: Any, required: set[str], optional: set[str], name: str) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        _error(f"{name} requires {sorted(required)}; optional keys: {sorted(optional)}")
    return value


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        _error(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
    try:
        result = float(value)
    except OverflowError:
        _error(f"{name} exceeds finite floating-point range")
        raise AssertionError("unreachable")
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        _error(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
    return result


def _length(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, dict):
        _keys(value, {"min", "max"}, set(), name)
        low = _integer(value["min"], f"{name}.min", 1, 2**31 - 1)
        high = _integer(value["max"], f"{name}.max", low, 2**31 - 1)
        return low, high
    number = _integer(value, name, 1, 2**31 - 1)
    return number, number


def _rng(seed: int, replication: int, stream: str) -> random.Random:
    digest = canonical_sha256([GENERATOR_VERSION, seed, replication, stream])
    return random.Random(int(digest, 16))


def validate_workload_spec(spec: dict) -> dict:
    """Validate explicit finite bounds without changing the caller's dictionary."""
    _keys(spec, {"schema_version", "seed", "arrival", "tokens"},
          {"request_prefix", "priority", "timeout_ms", "prompt_token_ids", "cache_namespace"}, "workload")
    _integer(spec["schema_version"], "schema_version", 1, 1)
    _integer(spec["seed"], "seed", -(2**63), 2**63 - 1)
    arrival = _keys(spec["arrival"], {"process", "rate_rps", "duration_ms", "max_requests"},
                    set(), "arrival")
    if arrival["process"] not in ("poisson", "periodic"):
        _error("arrival.process must be poisson or periodic (open-loop)")
    rate = _positive(arrival["rate_rps"], "rate_rps")
    duration = _positive(arrival["duration_ms"], "duration_ms")
    maximum = _integer(arrival["max_requests"], "max_requests", 1, MAX_REQUESTS)
    interval = 1000.0 / rate
    if not math.isfinite(interval) or interval == 0 or duration + interval == duration:
        _error("arrival rate/duration cannot represent advancing millisecond timestamps")
    expected = rate * duration / 1000.0
    if not math.isfinite(expected) or expected > maximum:
        _error("expected requests exceed max_requests; increase the explicit bound or reduce load")
    tokens = _keys(spec["tokens"], {"input", "output"}, set(), "tokens")
    low, high = _length(tokens["input"], "tokens.input")
    _length(tokens["output"], "tokens.output")
    prefix = spec.get("request_prefix", "generated")
    if not isinstance(prefix, str) or not prefix or len(prefix) > 128:
        _error("request_prefix must be a nonempty string of at most 128 characters")
    namespace = spec.get("cache_namespace", "")
    if not isinstance(namespace, str):
        _error("cache_namespace must be a string")
    _integer(spec.get("priority", 0), "priority", -(2**31), 2**31 - 1)
    if "timeout_ms" in spec:
        _positive(spec["timeout_ms"], "timeout_ms")
    if "prompt_token_ids" in spec:
        prompt = spec["prompt_token_ids"]
        if not isinstance(prompt, list) or low != high or len(prompt) != low:
            _error("prompt_token_ids requires a fixed input length and exactly that many token IDs")
        for token in prompt:
            _integer(token, "prompt token ID", 0, 2**63 - 1)
    return spec


def request_specs(generated: dict) -> tuple[RequestSpec, ...]:
    """Convert an expanded generator result to the existing immutable contract."""
    return tuple(RequestSpec(**{**row, "prompt_token_ids": tuple(row["prompt_token_ids"])})
                 for row in generated["requests"])


def generate_workload(spec: dict, *, replication: int = 0) -> dict:
    """Expand [0,duration) arrivals; exceeding a declared bound is an error."""
    validate_workload_spec(spec)
    _integer(replication, "replication", 0, 99_999)
    arrival = spec["arrival"]
    rate = float(arrival["rate_rps"])
    duration = float(arrival["duration_ms"])
    seed = spec["seed"]
    arrivals = _rng(seed, replication, "arrivals")
    inputs, outputs = (_rng(seed, replication, label) for label in ("input_lengths", "output_lengths"))
    input_range = _length(spec["tokens"]["input"], "tokens.input")
    output_range = _length(spec["tokens"]["output"], "tokens.output")
    interval = 1000.0 / rate
    when = 0.0
    rows: list[dict] = []
    index = 0
    while True:
        if arrival["process"] == "poisson":
            # random() can return zero: -log1p(-0) would produce no advance.
            draw = arrivals.random()
            step = -math.log1p(-draw) * interval
            next_when = when + step
            if not math.isfinite(next_when) or step <= 0 or next_when <= when:
                _error("Poisson arrival increment is not representable; no silent zero-time retry")
            when = next_when
        else:
            when = index * interval
            if not math.isfinite(when) or (index and when <= rows[-1]["arrival_ms"]):
                _error("Periodic arrival timestamp overflow or nonprogress")
        if when >= duration:
            break
        if index >= arrival["max_requests"]:
            _error("Generated requests exceed max_requests; no silent truncation")
        timeout = spec.get("timeout_ms")
        if timeout is not None and (not math.isfinite(when + timeout) or when + timeout <= when):
            _error("Request timeout deadline overflow or nonprogress")
        request = RequestSpec(
            f"{spec.get('request_prefix', 'generated')}-r{replication}-{index}", when,
            inputs.randint(*input_range), outputs.randint(*output_range),
            spec.get("priority", 0), timeout, tuple(spec.get("prompt_token_ids", ())),
            spec.get("cache_namespace", ""),
        )
        row = asdict(request)
        row["prompt_token_ids"] = list(row["prompt_token_ids"])
        rows.append(row)
        index += 1
    return {"requests": rows, "provenance": {
        "generator_version": GENERATOR_VERSION, "kind": "open_loop",
        "spec_sha256": canonical_sha256(spec), "requests_sha256": canonical_sha256(rows),
        "seed": seed, "replication": replication, "request_count": len(rows),
        "arrival_process": arrival["process"], "arrival_window_ms": [0.0, duration],
        "arrival_window_rule": "half_open", "rng": "python_random_sha256_independent_substreams_v1",
        "stream_names": ["arrivals", "input_lengths", "output_lengths"],
    }}
