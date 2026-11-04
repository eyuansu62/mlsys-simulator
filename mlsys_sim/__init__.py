"""Public Python API for CPU-only, deterministic LLM serving simulation.

Construct an :class:`Experiment` directly or load a configuration with
:func:`load_experiment`, then call :func:`simulate` or :func:`predict_experiment`.
Changes to this import surface and the versioned data contracts are recorded
in the changelog. Calibration provenance and hardware validation are separate.
"""

__version__ = "1.4.0"

from .config import experiment_from_dict, load_experiment, validate_experiment
from .engine import SimulationResult, simulate
from .memory import MemoryDecision, MemoryPlanner
from .layout import replica_experiment
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
from .profiles import (ProfileModel, create_performance_model, estimate_profile,
                       load_dataset, load_profile, profile_identity)
from .calibration import (calibrate_dataset, split_dataset, validate_holdout,
                          write_json_exclusive)
from .comparison import (compare_candidates, compare_runs, comparison_contract,
                         extract_metrics)
from .communication_calibration import calibrate_communication, load_communication_dataset
from .swap import HostSwapManager, SwapTransfer

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
    "replica_experiment",
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
    "ProfileModel",
    "create_performance_model",
    "estimate_profile",
    "load_dataset",
    "load_profile",
    "profile_identity",
    "calibrate_dataset",
    "split_dataset",
    "validate_holdout",
    "write_json_exclusive",
    "compare_candidates",
    "compare_runs",
    "comparison_contract",
    "extract_metrics",
    "calibrate_communication",
    "load_communication_dataset",
    "HostSwapManager",
    "SwapTransfer",
]
