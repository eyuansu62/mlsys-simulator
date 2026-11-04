"""Opt-in, source-pinned single-GPU Qwen3 physical KV-pool capacity adapter.

Importing this module imports neither Torch nor vLLM. The factory is intended
for capacity_worker owned processes, never an existing application Runtime.
"""

import ast
from copy import deepcopy
from dataclasses import asdict
import functools
import hashlib
from importlib import import_module
import inspect
import os
from pathlib import Path
import textwrap
from typing import Any, NoReturn

from .capacity import config_sha256
from .config import validate_experiment
from .qwen3 import load_qwen3_model
from .schema import Experiment, SimulationError
from .vllm_hooks import _shape_model_matches, _signature, _version


MODEL_PATH_ENV = "MLSYS_VLLM_CAPACITY_MODEL_PATH"
_ALLOCATOR_SOURCE_SHA256 = "567365d6ba1f8cfbf8e87d070e116d81984bc4a03ab7bd50517345e4e16363a2"
_ALLOCATOR_PARAMETERS = ("kv_cache_config", "device", "layout", "kernel_block_sizes")
_INITIALIZER_PARAMETERS = ("self", "kv_cache_config", "is_profiling", "kv_cache_allocation_context")


def _fail(message: str, code: str = "UNSUPPORTED_CAPACITY_RUNTIME") -> NoReturn:
    raise SimulationError(code, message)


def _scope(experiment: Experiment) -> None:
    validate_experiment(experiment)
    strategy, memory = experiment.strategy, experiment.strategy.memory
    if (strategy.tensor_parallel_size != 1 or strategy.pipeline_parallel_size != 1 or strategy.replicas != 1
            or len(experiment.infra.devices) != 1 or strategy.stage_layers
            or strategy.pipeline_transport != "full_hidden" or strategy.max_batch_size != 1
            or strategy.scheduler != "fifo" or strategy.preemption != "recompute"
            or strategy.swap_fallback != "none" or memory.cpu_swap_bytes != 0 or memory.graph_bytes != 0
            or memory.kv_block_tokens != 16
            or strategy.prefix_cache or strategy.chunked_prefill or strategy.mixed_batching):
        _fail("KV-pool adapter requires one FIFO Qwen3 TP1/PP1 replica, block16, eager and no cache/chunk/mixed/swap")
    if len(experiment.workload) != 1:
        _fail("KV-pool adapter executes exactly one declared request")
    request = experiment.workload[0]
    if (request.arrival_ms != 0 or request.priority != 0 or request.timeout_ms is not None or request.cache_namespace
            or len(request.prompt_token_ids) != request.input_tokens or not request.prompt_token_ids
            or request.output_tokens <= 0 or request.input_tokens + request.output_tokens > strategy.max_batch_tokens):
        _fail("KV-pool adapter needs one immediate untimed token-ID request within the fixed model/token limit")
    if (experiment.execution.warmup or experiment.execution.initial_state != "empty_cache"
            or experiment.evaluation.measurement_window_ms is not None or not experiment.evaluation.drain
            or experiment.evaluation.max_drain_ms is not None):
        _fail("Capacity trials require an empty fresh Runtime and a complete unwindowed workload")
    model = strategy.model
    if (model.weight_dtype not in {"float16", "bfloat16"} or model.kv_dtype != model.weight_dtype
            or model.activation_dtype != model.weight_dtype or not model.qk_norm or not model.final_norm
            or not model.architecture_config_sha256 or model.weight_bytes != 2 or model.kv_bytes != 2
            or model.activation_bytes != 2):
        _fail("Capacity adapter requires explicit unquantized FP16/BF16 full-attention Qwen3 identity")
    if experiment.calibration.runtime_name != "vllm":
        _fail("Capacity adapter supports only the pinned vLLM V1 Runtime")
    _version(experiment.calibration.runtime_version)
    if experiment.calibration.runtime_config_sha256:
        _fail("Capacity adapter cannot apply an externally pinned Runtime execution-config digest")


def _allocator_guard(function: Any) -> tuple[Any, int]:
    _signature(function, _ALLOCATOR_PARAMETERS, "allocate_kv_cache")
    if (not inspect.isfunction(function) or function.__module__ != "vllm.v1.worker.utils"
            or function.__name__ != "allocate_kv_cache" or hasattr(function, "__wrapped__")):
        _fail("Native KV allocator must be the unmodified pinned function", "UNSUPPORTED_VLLM_ABI")
    try:
        lines, first = inspect.getsourcelines(function)
        source = "".join(lines)
        if hashlib.sha256(source.encode()).hexdigest() != _ALLOCATOR_SOURCE_SHA256:
            _fail("Native KV allocation source differs from fixed revision 98dff2a81", "UNSUPPORTED_VLLM_ABI")
        parsed = ast.parse(textwrap.dedent(source))
        calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == "torch" and node.func.attr == "zeros"]
        if len(calls) != 1:
            _fail("Native KV allocator has no unique backing zeros call", "UNSUPPORTED_VLLM_ABI")
        return function.__code__, first + calls[0].lineno - 1
    except (OSError, TypeError, SyntaxError) as exc:
        raise SimulationError("UNSUPPORTED_VLLM_ABI", "Cannot verify the native KV allocation source") from exc


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an actual integer >= {minimum}", "KV_POOL_BINDING_MISMATCH")
    return value


def _device_index(device: Any) -> int:
    if getattr(device, "type", None) != "cuda" or getattr(device, "index", None) != 0:
        _fail("Native KV allocation must target the sole bound HIP device", "KV_POOL_BINDING_MISMATCH")
    return 0


class VllmKVPoolAdapter:
    """One-shot adapter; prediction-only reserves are not Runtime allocations."""

    def __init__(self, experiment: Experiment, *, model_path: str | Path):
        _scope(experiment)
        self.experiment = experiment
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            _fail("Capacity model_path must be an existing local model directory")
        imported = load_qwen3_model(self.model_path / "config.json", name=experiment.strategy.model.name,
                                   revision=experiment.strategy.model.revision, kv_dtype=experiment.strategy.model.kv_dtype)
        if (imported["model"] != asdict(experiment.strategy.model)
                or experiment.strategy.model.revision != "local-config-sha256:" + imported["provenance"]["source_sha256"]):
            _fail("Local Qwen3 configuration differs from the declared model identity", "HARDWARE_IDENTITY_MISMATCH")
        self._model_provenance = imported["provenance"]
        self._state = "new"
        self._llm: Any = None
        self._runner: Any = None
        self._torch: Any = None
        self._vllm: Any = None
        self._proof: dict | None = None
        self._pool: dict | None = None
        self._runtime: dict | None = None
        self._version: str | None = None
        self._oom = False
        for name in ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_V2_MODEL_RUNNER"):
            if os.environ.get(name, "0") != "0":
                _fail(f"Capacity adapter requires {name}=0 before importing vLLM")
        if os.environ.get("MLSYS_VLLM_CAPTURE_CONFIG"):
            _fail("Capacity adapter cannot run with a separately installed internal-capture session")

    def metadata(self) -> dict:
        if self._vllm is None:
            if os.environ.get("MLSYS_VLLM_CAPTURE_CONFIG"):
                _fail("Capacity adapter cannot share an internal-capture session")
            for name in ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_V2_MODEL_RUNNER"):
                if os.environ.get(name, "0") != "0":
                    _fail(f"Capacity adapter requires {name}=0 before importing vLLM")
                os.environ[name] = "0"
            self._vllm = import_module("vllm")
            self._version = _version(getattr(self._vllm, "__version__", ""))
            if self._version != self.experiment.calibration.runtime_version:
                _fail("Loaded vLLM version differs from the complete experiment", "HARDWARE_IDENTITY_MISMATCH")
        return {"isolation": "owned_process", "execution_scope": "complete_declared_workload",
            "effective_config_sha256": config_sha256(self.experiment),
            "runtime_name": "vllm", "runtime_version": self._version,
            "applied_axis": {"name": "kv_pool_bytes", "value": self.experiment.strategy.memory.kv_pool_bytes},
            "fixed_kv_pool_bytes": self.experiment.strategy.memory.kv_pool_bytes,
            "model_config_source_sha256": self._model_provenance["source_sha256"],
            "native_kv_allocator_source_sha256": _ALLOCATOR_SOURCE_SHA256,
            "preparation_scope": "runtime_startup_with_exact_native_kv_backing_failure_proof",
            "prediction_only_fields": ["analytical_efficiency", "memory_category_reserves", "evaluation_slas",
                                       "simulation_time_and_event_limits"],
            "weights_revision_scope": "local_config_identity_and_loaded_architecture_not_full_weight_content_hash"}

    def _runtime_binding(self, runner: Any) -> dict:
        exp = self.experiment
        config = runner.vllm_config
        model, scheduler, cache = config.model_config, config.scheduler_config, config.cache_config
        parallel, compilation = config.parallel_config, config.compilation_config
        expected = [(model, "enforce_eager", True), (model, "enable_sleep_mode", False),
            (model, "max_model_len", exp.strategy.max_batch_tokens),
            (scheduler, "policy", "fcfs"), (scheduler, "max_num_seqs", 1),
            (scheduler, "max_num_batched_tokens", exp.strategy.max_batch_tokens),
            (scheduler, "enable_chunked_prefill", False), (scheduler, "async_scheduling", False),
            (cache, "enable_prefix_caching", False), (cache, "block_size", exp.strategy.memory.kv_block_tokens),
            (cache, "kv_cache_memory_bytes", exp.strategy.memory.kv_pool_bytes),
            (cache, "num_gpu_blocks_override", None),
            (parallel, "tensor_parallel_size", 1), (parallel, "pipeline_parallel_size", 1),
            (parallel, "data_parallel_size", 1), (parallel, "decode_context_parallel_size", 1),
            (parallel, "prefill_context_parallel_size", 1), (parallel, "use_ubatching", False),
            (parallel, "enable_expert_parallel", False), (parallel, "disable_custom_all_reduce", True)]
        for owner, name, value in expected:
            actual = getattr(owner, name, object())
            if actual != value or (type(value) in (bool, int) and type(actual) is not type(value)):
                _fail(f"Actual vLLM {name} differs from the supported capacity configuration", "CAPACITY_CONFIG_UNPROVEN")
        offload = config.offload_config
        if offload.uva.cpu_offload_gb != 0 or offload.prefetch.offload_group_size != 0:
            _fail("Actual capacity Runtime must disable CPU offload", "CAPACITY_CONFIG_UNPROVEN")
        for name in ("mode", "cudagraph_mode"):
            value = getattr(compilation, name, None)
            resolved = getattr(value, "value", value)
            if type(resolved) is not int or resolved != 0:
                _fail("Actual capacity Runtime must disable compilation and graphs", "CAPACITY_CONFIG_UNPROVEN")
        for name in ("lora_config", "speculative_config", "kv_transfer_config", "kv_events_config"):
            if getattr(config, name, None) is not None:
                _fail(f"Actual capacity Runtime has unsupported {name}", "CAPACITY_CONFIG_UNPROVEN")
        binding = _shape_model_matches(runner, exp, _phase="prepared")
        return {"model": binding, "max_model_len": model.max_model_len, "scheduler": "fcfs",
            "max_num_seqs": 1, "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "enforce_eager": True, "compilation_and_graphs": "disabled",
            "kv_cache_memory_bytes": cache.kv_cache_memory_bytes, "block_size": cache.block_size}

    def _pool_binding(self, config: Any, device: Any) -> dict:
        _device_index(device)
        tensors = config.kv_cache_tensors
        if not isinstance(tensors, list) or not tensors:
            _fail("Native KV config has no physical tensor descriptors", "KV_POOL_BINDING_MISMATCH")
        sizes = {_integer(tensor.size, "KV descriptor size") for tensor in tensors}
        blocks = _integer(config.num_blocks, "KV block count")
        if len(sizes) != 1:
            _fail("Native KV tensors must share one backing extent", "KV_POOL_BINDING_MISMATCH")
        raw = sizes.pop()
        quantum, remainder = divmod(raw, blocks)
        target = self.experiment.strategy.memory.kv_pool_bytes
        if remainder or quantum <= 0 or raw != target or (raw + 4095) // 4096 * 4096 != target:
            _fail(f"Requested KV bytes must match actual block and page alignment (raw={raw}, blocks={blocks})",
                  "KV_POOL_BINDING_MISMATCH")
        return {"device_index": 0, "num_blocks": blocks, "block_quantum_bytes": quantum,
                "raw_bytes": raw, "storage_bytes": target}

    def prepare(self) -> None:
        if self._state != "new":
            _fail("Capacity Runtime can only be prepared once")
        self._state = "preparing"
        self.metadata()
        self._torch = import_module("torch")
        if (not getattr(self._torch.version, "hip", None) or not self._torch.cuda.is_available()
                or self._torch.cuda.device_count() != 1):
            _fail("Capacity adapter requires a verified available HIP Torch backend", "HARDWARE_UNAVAILABLE")
        module: Any = import_module("vllm.v1.worker.gpu_model_runner")
        original = module.allocate_kv_cache
        code, zeros_line = _allocator_guard(original)
        initializer = module.GPUModelRunner.initialize_kv_cache
        _signature(initializer, _INITIALIZER_PARAMETERS, "GPUModelRunner.initialize_kv_cache")
        initializer_code = getattr(initializer, "__code__", None)

        @functools.wraps(original)
        def observe(kv_cache_config, device, layout, kernel_block_sizes=None):
            frame = inspect.currentframe()
            try:
                frame = frame.f_back if frame is not None else None
                while frame is not None and frame.f_code is not initializer_code:
                    frame = frame.f_back
                if frame is None or type(frame.f_locals.get("is_profiling")) is not bool:
                    _fail("KV allocation escaped the pinned runner initialization scope", "UNSUPPORTED_VLLM_ABI")
                if frame.f_locals["is_profiling"]:
                    return original(kv_cache_config, device, layout, kernel_block_sizes)
                runner = frame.f_locals.get("self")
                if type(runner) is not module.GPUModelRunner or self._pool is not None:
                    _fail("Capacity trial has an unexpected/repeated physical KV initialization", "UNSUPPORTED_VLLM_ABI")
                self._runtime = self._runtime_binding(runner)
                self._pool = self._pool_binding(kv_cache_config, device)
                self._runner = runner
                try:
                    return original(kv_cache_config, device, layout, kernel_block_sizes)
                except self._torch.cuda.OutOfMemoryError as exc:
                    trace = exc.__traceback__
                    while trace is not None:
                        if trace.tb_frame.f_code is code and trace.tb_lineno == zeros_line:
                            actual = trace.tb_frame.f_locals
                            size = actual.get("buf_size")
                            if (type(size) is int and size == self._pool["storage_bytes"]
                                    and actual.get("kv_cache_config") is kv_cache_config):
                                index = _device_index(actual.get("device"))
                                self._proof = {"resource": "kv_pool", "phase": "prepare",
                                    "effective_config_sha256": config_sha256(self.experiment),
                                    "attempts": [{"device_index": index, "shape": [size], "element_size": 1,
                                        "target_pool_bytes": size, "operation": "torch.zeros"}]}
                            break
                        trace = trace.tb_next
                    raise
            finally:
                del frame

        module.allocate_kv_cache = observe
        try:
            exp = self.experiment
            self._llm = self._vllm.LLM(str(self.model_path), dtype=exp.strategy.model.weight_dtype,
                enforce_eager=True, distributed_executor_backend="uni", tensor_parallel_size=1,
                pipeline_parallel_size=1, max_model_len=exp.strategy.max_batch_tokens,
                max_num_batched_tokens=exp.strategy.max_batch_tokens, max_num_seqs=1,
                block_size=exp.strategy.memory.kv_block_tokens, enable_prefix_caching=False,
                enable_chunked_prefill=False, async_scheduling=False,
                kv_cache_memory_bytes=exp.strategy.memory.kv_pool_bytes,
                kv_cache_dtype=exp.strategy.model.kv_dtype, cpu_offload_gb=0, scheduling_policy="fcfs",
                compilation_config=0, disable_custom_all_reduce=True, skip_tokenizer_init=True,
                generation_config="vllm", seed=exp.execution.seed, trust_remote_code=False, enable_sleep_mode=False)
            actual_runner = self._llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
            if actual_runner is not self._runner or self._pool is None:
                _fail("Loaded Runtime did not expose the observed actual KV allocation", "KV_POOL_BINDING_UNAVAILABLE")
            self.fixed_kv_pool_allocations()
            self._state = "prepared"
        except Exception as exc:
            self._oom = isinstance(exc, self._torch.cuda.OutOfMemoryError)
            self._state = "oom" if self._oom else "failed"
            raise
        finally:
            module.allocate_kv_cache = original

    def fixed_kv_pool_allocations(self) -> dict:
        if self._runner is None or self._pool is None or self._oom:
            _fail("Actual KV storage is unavailable", "KV_POOL_BINDING_UNAVAILABLE")
        values = self._runner.kv_caches
        if not isinstance(values, list) or not values:
            _fail("Runtime exposes no bound KV storage views", "KV_POOL_BINDING_UNAVAILABLE")
        # The capacity worker verifies devices, aliases and complete storage bytes.
        return {0: list(values)}

    def capacity_oom_evidence(self) -> dict | None:
        return deepcopy(self._proof)

    def run(self) -> dict:
        if self._state != "prepared":
            _fail("Capacity run needs one newly prepared Runtime")
        self._state = "running"
        try:
            request = self.experiment.workload[0]
            sampling = self._vllm.SamplingParams(temperature=0, max_tokens=request.output_tokens,
                ignore_eos=True, detokenize=False, seed=self.experiment.execution.seed)
            output = self._llm.generate([{"prompt_token_ids": list(request.prompt_token_ids)}], sampling, use_tqdm=False)
            if (not isinstance(output, list) or len(output) != 1 or output[0].finished is not True
                    or len(output[0].outputs) != 1
                    or len(output[0].outputs[0].token_ids) != request.output_tokens
                    or any(type(token) is not int or not 0 <= token < self.experiment.strategy.model.vocab_size
                           for token in output[0].outputs[0].token_ids)):
                _fail("Actual LLM did not finish the complete declared token cohort", "INCOMPLETE_CAPACITY_WORKLOAD")
            self._state = "completed"
            return {"status": "COMPLETED", "completed_request_ids": [request.request_id],
                    "runtime_binding": deepcopy(self._runtime), "kv_pool": deepcopy(self._pool)}
        except Exception as exc:
            self._proof = None
            self._oom = isinstance(exc, self._torch.cuda.OutOfMemoryError)
            self._state = "oom" if self._oom else "failed"
            raise

    def close(self) -> None:
        if self._state == "closed":
            return
        if self._llm is not None and not self._oom:
            self._llm.llm_engine.engine_core.shutdown()
        # OOM-mutated state stays owned by the worker until process exit.
        self._state = "closed"


def kv_pool_factory(experiment: Experiment) -> VllmKVPoolAdapter:
    """Capacity-worker factory; the operator supplies an explicit local path."""
    path = os.environ.get(MODEL_PATH_ENV)
    if not path:
        _fail(f"Set {MODEL_PATH_ENV} to the already downloaded local Qwen3 directory")
    return VllmKVPoolAdapter(experiment, model_path=path)
