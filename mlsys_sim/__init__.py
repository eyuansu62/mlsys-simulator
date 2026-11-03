"""Public Python API for CPU-only, deterministic LLM serving simulation.

Construct an :class:`Experiment` directly or load a configuration with
:func:`load_experiment`, then call :func:`simulate` or :func:`predict_experiment`.
The package is pre-1.0; changes to this import surface are recorded in the
changelog. Analytical and synthetic timings are not hardware-validated.
"""

__version__ = "0.1.0"

from .config import experiment_from_dict, load_experiment, validate_experiment
from .engine import SimulationResult, simulate
from .memory import MemoryDecision, MemoryPlanner
from .performance import AnalyticalModel, predict_experiment, ring_allreduce_ms
from .schema import (
    CalibrationSpec,
    DeviceSpec,
    EvaluationSpec,
    ExecutionSpec,
    Experiment,
    InfraSpec,
    MemoryConfig,
    ModelSpec,
    RequestSpec,
    SimulationError,
    StepEstimate,
    StrategySpec,
)

__all__ = [
    "__version__",
    "AnalyticalModel",
    "CalibrationSpec",
    "DeviceSpec",
    "EvaluationSpec",
    "ExecutionSpec",
    "Experiment",
    "InfraSpec",
    "MemoryConfig",
    "MemoryDecision",
    "MemoryPlanner",
    "ModelSpec",
    "RequestSpec",
    "SimulationError",
    "SimulationResult",
    "StepEstimate",
    "StrategySpec",
    "experiment_from_dict",
    "load_experiment",
    "predict_experiment",
    "ring_allreduce_ms",
    "simulate",
    "validate_experiment",
]
