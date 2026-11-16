"""Portable experiment mappings preserving default legacy control semantics."""

from dataclasses import asdict
from copy import deepcopy

from .schema import CalibrationSpec, Experiment, StrategySpec


def _omit_default_costs(value: dict) -> dict:
    if value.get("analytical_costs", ()) in ((), []):
        value.pop("analytical_costs", None)
    for name in ("prefill_execution_mode", "decode_execution_mode",
                 "prefill_attention_backend", "decode_attention_backend", "analytical_cost_timing_boundary"):
        if value.get(name, "unspecified") == "unspecified":
            value.pop(name, None)
    return value


def calibration_dict(calibration: CalibrationSpec) -> dict:
    """Keep opt-in cost profiles explicit without adding defaults to old inputs."""
    return _omit_default_costs(asdict(calibration))


def _omit_default_controls(value: dict) -> dict:
    if value.get("max_active_requests") is None:
        value.pop("max_active_requests", None)
    memory = value.get("memory")
    if isinstance(memory, dict) and memory.get("gpu_memory_utilization") is None:
        memory.pop("gpu_memory_utilization", None)
    return value


def strategy_dict(strategy: StrategySpec) -> dict:
    """Keep enabled controls explicit without changing old default snapshots."""
    return _omit_default_controls(asdict(strategy))


def experiment_dict(experiment: Experiment | dict) -> dict:
    """Return a fresh mapping suitable for replay, emitted configs and hashes."""
    value = asdict(experiment) if isinstance(experiment, Experiment) else deepcopy(experiment)
    if isinstance(value.get("strategy"), dict):
        _omit_default_controls(value["strategy"])
    if isinstance(value.get("calibration"), dict):
        _omit_default_costs(value["calibration"])
    return value
