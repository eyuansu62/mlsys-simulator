"""Public API for deterministic LLM simulation and optional ROCm collection.

Construct an :class:`Experiment` directly or load a configuration with
:func:`load_experiment`, then call :func:`simulate` or :func:`predict_experiment`.
Changes to this import surface and the versioned data contracts are recorded
in the changelog. Calibration provenance and hardware validation are separate.
"""

__version__ = "1.7.0"

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
from .hardware import CollectionError, bind_inventory, probe_rocm
from .rocm_steps import collect_steps
from .rccl_adapter import collect_rccl, parse_rccl_output
from .runtime_adapter import RuntimeTraceRecorder, collect_vllm, normalize_runtime_trace
from .hardware_validation import run_hardware_validation
from .release_adapter import import_reference_data
from .qwen3 import (load_qwen3_model, model_from_qwen3_config,
                    normalized_qwen3_architecture, qwen3_architecture_fingerprint)
from .vllm_hooks import (RunnerCapture, SchedulerCapture, VllmCaptureWorkerExtension,
                         install_runner_hook, install_scheduler_hook, register_vllm_hooks,
                         uninstall_vllm_hooks_plugin)
from .vllm_capture import load_vllm_captures, normalize_vllm_capture
from .pp_transport import (estimate_pipeline_transfer, native_pp_descriptor,
                           validate_native_pp_descriptor)

__all__ = [
    "estimate_pipeline_transfer",
    "native_pp_descriptor",
    "validate_native_pp_descriptor",
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
    "CollectionError",
    "probe_rocm",
    "bind_inventory",
    "collect_steps",
    "collect_rccl",
    "parse_rccl_output",
    "RuntimeTraceRecorder",
    "collect_vllm",
    "normalize_runtime_trace",
    "run_hardware_validation",
    "import_reference_data",
    "model_from_qwen3_config",
    "load_qwen3_model",
    "normalized_qwen3_architecture",
    "qwen3_architecture_fingerprint",
    "RunnerCapture",
    "SchedulerCapture",
    "VllmCaptureWorkerExtension",
    "install_runner_hook",
    "install_scheduler_hook",
    "register_vllm_hooks",
    "uninstall_vllm_hooks_plugin",
    "load_vllm_captures",
    "normalize_vllm_capture",
]
