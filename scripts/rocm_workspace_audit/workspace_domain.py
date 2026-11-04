"""Exact supported live Workspace domain; observations must come from the worker."""

import hashlib
import json
from typing import Any

RUNTIME_VERSION = "0.29.1.dev0+g98dff2a81.d20261002"
TORCH_VERSION = "2.13.0+rocm10.1.0"
TORCH_GIT = "a8ec1f47c5b233927bf23f5cd8f68db3475af226"
HIP_VERSION = "7.16.26385"
DEVICE_NAME = "AMD Radeon RX 9070 XT"
KV_BYTES = 268435456
RUNNER_CLASS = "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
_MISSING = object()


class WorkspaceDomainError(ValueError):
    pass


def plain(value: Any) -> Any:
    """Normalize JSON-supported enums, never replace absent observations."""
    return json.loads(json.dumps(value, allow_nan=False))


def config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _get(root: Any, path: str) -> Any:
    value = root
    for name in path.split("."):
        if not isinstance(value, dict) or name not in value:
            return _MISSING
        value = value[name]
    return value


def _expect(issues: list[str], root: Any, path: str, expected: Any, prefix: str) -> None:
    value = _get(root, path)
    if value is _MISSING:
        issues.append(prefix + ":MISSING:" + path)
    elif type(value) is not type(expected) or value != expected:
        issues.append(prefix + ":MISMATCH:" + path)


def declared_domain_issues(experiment: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    expected = {
        "strategy.tensor_parallel_size": 1,
        "strategy.pipeline_parallel_size": 1,
        "strategy.replicas": 1,
        "strategy.max_batch_tokens": 512,
        "strategy.max_batch_size": 1,
        "strategy.prefix_cache": False,
        "strategy.chunked_prefill": False,
        "strategy.mixed_batching": False,
        "strategy.pipeline_transport": "full_hidden",
        "strategy.memory.kv_pool_bytes": KV_BYTES,
        "strategy.memory.kv_block_tokens": 16,
        "strategy.memory.graph_bytes": 0,
        "strategy.memory.cpu_swap_bytes": 0,
        "strategy.model.name": "Qwen3-1.7B",
        "strategy.model.weight_dtype": "bfloat16",
        "strategy.model.kv_dtype": "bfloat16",
        "strategy.model.activation_dtype": "bfloat16",
        "strategy.model.tie_word_embeddings": True,
        "strategy.model.qk_norm": True,
        "strategy.model.final_norm": True,
        "calibration.runtime_name": "vllm",
        "calibration.runtime_version": RUNTIME_VERSION,
        "calibration.driver_version": "6.16.13",
        "calibration.rocm_version": "10.1.0",
        "calibration.rccl_version": "2.30.7",
        "calibration.timing_boundary": "step_includes_output_and_sampling_v1",
    }
    for path, value in expected.items():
        _expect(issues, experiment, path, value, "DECLARED_DOMAIN")
    devices = experiment.get("infra", {}).get("devices")
    if not isinstance(devices, (list, tuple)) or len(devices) != 1:
        issues.append("DECLARED_DOMAIN:ONE_DEVICE_REQUIRED")
    else:
        _expect(issues, devices[0], "name", DEVICE_NAME, "DECLARED_DEVICE")
        capacity = devices[0].get("memory_bytes")
        if type(capacity) is not int or capacity <= 0:
            issues.append("DECLARED_DEVICE:INVALID_CAPACITY")
    return issues


def observed_domain_issues(
    experiment: dict[str, Any],
    observation: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    require_runner: bool = True,
) -> list[str]:
    issues = declared_domain_issues(experiment)
    expected = {
        "schema_version": 1,
        "runtime_name": "vllm",
        "runtime_version": RUNTIME_VERSION,
        "torch_version": TORCH_VERSION,
        "torch_git": TORCH_GIT,
        "hip_build_version": HIP_VERSION,
        "visible_device_count": 1,
        "allocator_backend": "native",
        "device.index": 0,
        "device.name": DEVICE_NAME,
    }
    for path, value in expected.items():
        _expect(issues, observation, path, value, "OBSERVED_DOMAIN")
    devices = experiment.get("infra", {}).get("devices", [])
    if isinstance(devices, (list, tuple)) and len(devices) == 1:
        _expect(issues, observation, "device.memory_bytes", devices[0].get("memory_bytes"), "OBSERVED_DEVICE")
    architecture = _get(observation, "device.architecture")
    if not isinstance(architecture, str) or architecture.split(":")[0] != "gfx1201":
        issues.append("OBSERVED_DEVICE:UNSUPPORTED_ARCHITECTURE")
    if require_runner:
        runner = {
            "class_name": RUNNER_CLASS,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "config_tensor_parallel_size": 1,
            "config_pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "use_async_scheduling": False,
        }
        for path, value in runner.items():
            _expect(issues, observation, "runner." + path, value, "OBSERVED_RUNNER")
        hardware = {
            "source_kind": "measured",
            "status": "READY",
            "driver_version": "6.16.13",
            "rocm_version": "10.1.0",
            "rccl_version": "2.30.7",
        }
        for path, value in hardware.items():
            _expect(issues, observation, "hardware." + path, value, "OBSERVED_HARDWARE")
        digest = _get(observation, "hardware.rccl_library_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            issues.append("OBSERVED_HARDWARE:MISSING_RCCL_LIBRARY_DIGEST")
    if config is not None:
        expected_config = {
            "api_revision": "98dff2a81",
            "async_scheduling": False,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "model_config.enforce_eager": True,
            "model_config.max_model_len": 512,
            "scheduler_config.policy": "fcfs",
            "scheduler_config.max_num_batched_tokens": 512,
            "scheduler_config.max_num_seqs": 1,
            "scheduler_config.enable_chunked_prefill": False,
            "cache_config.cache_dtype": "auto",
            "cache_config.block_size": 16,
            "cache_config.enable_prefix_caching": False,
            "cache_config.kv_sharing_fast_prefill": False,
            "cache_config.kv_cache_memory_bytes": KV_BYTES,
            "cache_config.swap_space": None,
            "cache_config.cpu_offload_gb": None,
            "parallel_config.data_parallel_size": 1,
            "parallel_config.decode_context_parallel_size": 1,
            "parallel_config.prefill_context_parallel_size": 1,
            "parallel_config.use_ubatching": False,
            "parallel_config.enable_expert_parallel": False,
            "compilation_config.mode": 0,
            "compilation_config.cudagraph_mode": "NONE",
            "environment.VLLM_USE_V2_MODEL_RUNNER": "0",
        }
        for path, value in expected_config.items():
            _expect(issues, config, path, value, "ACTUAL_RUNTIME_CONFIG")
    return sorted(set(issues))


def require_domain(issues: list[str]) -> None:
    if issues:
        raise WorkspaceDomainError("Unsupported or incomplete Workspace domain: " + "; ".join(issues))


def bound_config_sha256(experiment: dict[str, Any], config: dict[str, Any]) -> str:
    digest = config_sha256(config)
    expected = _get(experiment, "calibration.runtime_config_sha256")
    if type(expected) is not str or expected not in ("", digest):
        raise WorkspaceDomainError("Declared runtime_config_sha256 differs from the actual Runner snapshot")
    return digest


def recorded_session_domain_issues(session: dict[str, Any]) -> list[str]:
    experiment = session.get("experiment")
    observation = session.get("observed_runtime_domain")
    config = session.get("runtime_config")
    if not isinstance(experiment, dict):
        return ["RECORDED_DOMAIN:MISSING_EXPERIMENT"]
    issues = declared_domain_issues(experiment)
    if not isinstance(observation, dict):
        return sorted(set(issues + ["RECORDED_DOMAIN:MISSING_OBSERVED_RUNTIME_DOMAIN"]))
    if not isinstance(config, dict):
        return sorted(set(issues + ["RECORDED_DOMAIN:MISSING_RUNTIME_CONFIG"]))
    issues += observed_domain_issues(experiment, observation, config)
    for key, observed_key in (
        ("observed_runtime_version", "runtime_version"),
        ("torch_version", "torch_version"),
        ("torch_git", "torch_git"),
    ):
        _expect(issues, session, key, observation.get(observed_key), "RECORDED_DOMAIN")
    digest = _get(experiment, "calibration.runtime_config_sha256")
    if not isinstance(digest, str) or digest != config_sha256(config):
        issues.append("RECORDED_DOMAIN:RUNTIME_CONFIG_SHA256_MISMATCH")
    return sorted(set(issues))
