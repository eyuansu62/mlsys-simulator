"""Shared immutable experiment and prediction contracts (all durations in ms)."""

from dataclasses import dataclass, field


class SimulationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ModelSpec:
    name: str = "toy-dense-gqa"
    num_layers: int = 4
    hidden_size: int = 128
    intermediate_size: int = 512
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 16
    vocab_size: int = 256
    weight_bytes: float = 2.0
    kv_bytes: float = 2.0
    activation_bytes: float = 2.0
    revision: str = "unspecified"
    weight_dtype: str = "auto"
    kv_dtype: str = "auto"
    activation_dtype: str = "auto"


@dataclass(frozen=True)
class DeviceSpec:
    id: str = "gpu0"
    memory_bytes: int = 64 * 1024 * 1024
    compute_tflops: float = 1.0
    memory_bandwidth_gbps: float = 10.0
    name: str = "toy-device"


@dataclass(frozen=True)
class InfraSpec:
    devices: tuple[DeviceSpec, ...] = field(default_factory=lambda: (DeviceSpec(),))
    link_bandwidth_gbps: float = 20.0
    link_latency_us: float = 5.0
    topology: str = "single-node-uncontended"


@dataclass(frozen=True)
class MemoryConfig:
    kv_block_tokens: int = 16
    kv_pool_bytes: int = 4 * 1024 * 1024
    workspace_base_bytes: int = 1024 * 1024
    workspace_bytes_per_token: int = 1024
    graph_bytes: int = 0
    runtime_bytes: int = 1024 * 1024
    communication_buffer_bytes: int = 0
    other_bytes: int = 0
    safety_margin_bytes: int = 1024 * 1024
    guard_blocks: int = 0
    cpu_swap_bytes: int = 0
    swap_d2h_bandwidth_gbps: float = 12.0
    swap_h2d_bandwidth_gbps: float = 12.0
    swap_latency_us: float = 10.0


@dataclass(frozen=True)
class StrategySpec:
    model: ModelSpec = field(default_factory=ModelSpec)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    stage_layers: tuple[int, ...] = ()
    replicas: int = 1
    scheduler: str = "continuous_batching"
    max_batch_size: int = 8
    max_batch_tokens: int = 4096
    preemption: str = "recompute"
    max_preemptions_per_request: int = 100
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    pipeline_microbatch_size: int = 1
    pipeline_max_inflight_microbatches: int = 8
    pipeline_feedback_latency_us: float = 0.0
    pipeline_p2p: bool = True
    swap_fallback: str = "none"
    prefix_cache: bool = False
    prefix_cache_max_blocks: int = 128
    chunked_prefill: bool = False
    prefill_chunk_tokens: int = 512


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ms: float
    input_tokens: int
    output_tokens: int
    priority: int = 0
    timeout_ms: float | None = None
    prompt_token_ids: tuple[int, ...] = ()
    cache_namespace: str = ""


@dataclass(frozen=True)
class CalibrationSpec:
    backend: str = "analytical"
    profile_version: str = "uncalibrated-v1"
    compute_efficiency: float = 0.5
    bandwidth_efficiency: float = 0.5
    launch_overhead_us: float = 2.0
    framework_overhead_us: float = 5.0
    prefill_ms: float = 10.0
    decode_ms: float = 1.0
    recompute_ms: float = 10.0
    profile_path: str | None = None
    allow_extrapolation: bool = False
    allow_test_fixture: bool = False
    runtime_name: str = "unspecified"
    runtime_version: str = "unspecified"
    driver_version: str = "unspecified"
    rocm_version: str = "unspecified"
    rccl_version: str = "unspecified"
    timing_boundary: str = "step_includes_output_and_sampling_v1"


@dataclass(frozen=True)
class EvaluationSpec:
    ttft_sla_ms: float | None = None
    tpot_sla_ms: float | None = None
    measurement_window_ms: float | None = None
    drain: bool = True
    max_drain_ms: float | None = None


@dataclass(frozen=True)
class ExecutionSpec:
    seed: int = 42
    max_simulated_ms: float = 180000.0
    max_events: int = 1000000
    detailed_events: bool = True
    initial_state: str = "empty_cache"
    warmup: tuple[RequestSpec, ...] = ()


@dataclass(frozen=True)
class Experiment:
    workload: tuple[RequestSpec, ...] = ()
    infra: InfraSpec = field(default_factory=InfraSpec)
    strategy: StrategySpec = field(default_factory=StrategySpec)
    calibration: CalibrationSpec = field(default_factory=CalibrationSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    schema_version: int = 1


@dataclass(frozen=True)
class StepEstimate:
    duration_ms: float
    breakdown: dict
    source: str
    assumptions: tuple[str, ...] = ()
