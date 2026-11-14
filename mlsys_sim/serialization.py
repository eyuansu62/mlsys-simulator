"""Portable experiment mappings preserving default legacy control semantics."""

from dataclasses import asdict
from copy import deepcopy

from .schema import Experiment, StrategySpec


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
    return value
