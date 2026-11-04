"""Opt-in internal capture for the source-pinned vLLM V1 runner ABI.

No vLLM or Torch import occurs until explicit live installation. Async outputs
are observed only when their real caller consumes them; their consumer-delayed
intervals remain raw evidence rather than exclusive stage service samples.
"""

import atexit
from copy import deepcopy
from dataclasses import asdict, replace
import functools
import hashlib
import importlib
import inspect
import json
import math
import operator
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, NoReturn
import uuid

from .batching import batch_phase
from .config import experiment_from_dict, load_experiment, validate_experiment
from .hardware import object_fields, read_json, text
from .layout import replica_experiment, stage_layer_counts
from .pp_transport import pipeline_transport_max_tokens, validate_native_pp_descriptor
from .profiles import profile_identity, validate_identity
from .schema import Experiment, SimulationError


API_REVISION = "98dff2a81"
SUPPORTED_VERSION_PREFIX = "0.29.1.dev0+g98dff2a81"
_BOUNDARY = "step_includes_output_and_sampling_v1"
_MISSING = object()
_PLUGIN_PATCHES: list[tuple[type, Any]] = []
_ACTIVE_CAPTURES: list[Any] = []
_PLUGIN_CONFIG: dict | None = None


def _fail(code: str, message: str) -> NoReturn:
    raise SimulationError(code, message)


def _int(value: Any, label: str, minimum: int = 0) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise SimulationError("UNSUPPORTED_VLLM_ABI", f"{label} must be an integer") from exc
    if isinstance(value, bool) or not minimum <= result <= 2**63 - 1:
        _fail("UNSUPPORTED_VLLM_ABI", f"{label} is outside its integer range")
    return result


def _list(value: Any, label: str) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        _fail("UNSUPPORTED_VLLM_ABI", f"{label} must expose a one-dimensional array")
    return list(value)


def _version(value: str) -> str:
    if not isinstance(value, str) or not (value == SUPPORTED_VERSION_PREFIX
            or value.startswith(SUPPORTED_VERSION_PREFIX + ".")):
        _fail("UNSUPPORTED_VLLM_VERSION", f"Internal capture supports vLLM {SUPPORTED_VERSION_PREFIX} only")
    return value


def _signature(target: Any, expected: tuple[str, ...], label: str) -> None:
    if not callable(target):
        _fail("UNSUPPORTED_VLLM_ABI", f"{label} is not callable")
    try:
        parameters = tuple(inspect.signature(target).parameters.values())
    except (ValueError, TypeError) as exc:
        raise SimulationError("UNSUPPORTED_VLLM_ABI", f"Cannot inspect {label}") from exc
    if tuple(parameter.name for parameter in parameters) != expected or any(
            parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD) for parameter in parameters):
        _fail("UNSUPPORTED_VLLM_ABI", f"{label} signature differs from source revision {API_REVISION}")


def _clock_domain() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError as exc:
        raise SimulationError("HARDWARE_METADATA_INCOMPLETE", "A shared host monotonic clock domain is required") from exc
    return "linux-boot:" + value


def _runtime_version() -> str:
    module = importlib.import_module("vllm")
    return _version(getattr(module, "__version__", ""))


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    return str(value)


def _execution_config(config: Any, experiment: Experiment, async_scheduling: bool) -> dict:
    """Retain configured execution policy, excluding rank/path-only metadata."""
    result: dict[str, Any] = {"api_revision": API_REVISION, "async_scheduling": bool(async_scheduling),
        "tensor_parallel_size": experiment.strategy.tensor_parallel_size,
        "pipeline_parallel_size": experiment.strategy.pipeline_parallel_size}
    sections = {
        "scheduler_config": ("policy", "max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill"),
        "cache_config": ("cache_dtype", "block_size", "enable_prefix_caching", "kv_sharing_fast_prefill",
            "gpu_memory_utilization", "num_gpu_blocks_override", "kv_cache_memory_bytes", "swap_space", "cpu_offload_gb"),
        "compilation_config": ("mode", "cudagraph_mode", "backend", "compile_sizes",
            "cudagraph_capture_sizes", "max_cudagraph_capture_size", "splitting_ops", "custom_ops",
            "cudagraph_num_of_warmups", "cudagraph_copy_inputs"),
        "parallel_config": ("data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size",
                            "use_ubatching", "enable_expert_parallel"),
        "attention_config": ("backend", "attention_backend"),
        "model_config": ("enforce_eager", "max_model_len"),
    }
    for section, names in sections.items():
        owner = getattr(config, section, None)
        result[section] = {name: _plain(getattr(owner, name, None)) for name in names}
    passes = getattr(getattr(config, "compilation_config", None), "pass_config", None)
    result["compilation_pass_config"] = ({key: _plain(value) for key, value in sorted(vars(passes).items())
        if not key.startswith("_")} if passes is not None and hasattr(passes, "__dict__") else None)
    result["environment"] = {name: os.environ.get(name) for name in
        ("VLLM_ATTENTION_BACKEND", "VLLM_USE_V2_MODEL_RUNNER", "NCCL_ALGO", "NCCL_PROTO",
         "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MHA", "VLLM_ROCM_USE_AITER_PAGED_ATTN",
         "VLLM_ROCM_USE_AITER_RMSNORM", "VLLM_ROCM_USE_AITER_ROPE", "VLLM_ROCM_USE_AITER_LINEAR",
         "VLLM_USE_TRITON_FLASH_ATTN")}
    # Preserve older unset-policy identities, but never merge explicit transport
    # choices such as host-staged P2P into the default Runtime domain.
    for name in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "NCCL_NET",
                 "TORCH_NCCL_BLOCKING_WAIT", "NCCL_P2P_LL_THRESHOLD", "NCCL_P2P_LL128_THRESHOLD",
                 "NCCL_SHM_USE_CUDA_MEMCPY", "NCCL_SHM_MEMCPY_MODE"):
        if name in os.environ:
            result["environment"][name] = os.environ[name]
    return result


def _config_digest(snapshot: dict) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def cache_namespace_from_vllm_salt(salt: str | None) -> str:
    """Opaque equivalent domain for vLLM's truthy first-block cache salt."""
    if salt is None or salt == "":
        return ""
    if not isinstance(salt, str) or len(salt) > 4096:
        _fail("UNSUPPORTED_VLLM_ABI", "Request cache_salt must be a bounded string or None")
    return "vllm-salt-sha256:" + hashlib.sha256(salt.encode("utf-8")).hexdigest()


def _attention_backends(runner: Any) -> list[str]:
    result: set[str] = set()
    for groups in getattr(runner, "attn_groups", []):
        for group in groups:
            backend = getattr(group, "backend", getattr(group, "attn_backend", None))
            if backend is not None:
                method = getattr(backend, "get_name", None)
                result.add(str(method() if callable(method) else getattr(backend, "__name__", type(backend).__name__)))
    return sorted(result)


def _loaded_rccl() -> dict:
    """Query the shared RCCL actually mapped in this Linux worker process."""
    from .hardware import _rccl_version
    try:
        with Path("/proc/self/maps").open() as stream:
            maps = stream.read(4 * 1024 * 1024)
        paths = {Path(line.split(maxsplit=5)[-1]).resolve()
                 for line in maps.splitlines() if len(line.split(maxsplit=5)) == 6
                 and line.split(maxsplit=5)[-1].startswith("/")
                 and Path(line.split(maxsplit=5)[-1]).name.startswith("librccl.so")}
        if len(paths) != 1:
            _fail("HARDWARE_METADATA_INCOMPLETE", "Exactly one loaded librccl.so mapping is required for live TP capture")
        fact = _rccl_version(next(iter(paths)))
    except (OSError, AttributeError, ValueError) as exc:
        raise SimulationError("HARDWARE_METADATA_INCOMPLETE", f"Cannot establish the loaded RCCL library: {exc}") from exc
    return {"verified": True, "runtime_version": fact["version"], "library_sha256": fact["sha256"],
            "path": fact["path"], "version_code": fact["version_code"],
            "verification_scope": "local_proc_maps_loaded_library_ncclGetVersion"}


def _stage_layer_binding(runner: Any, experiment: Experiment, stage_index: int) -> dict:
    model = getattr(runner, "model", None)
    candidates = []
    visited: set[int] = set()
    for _ in range(8):
        if model is None or id(model) in visited:
            break
        visited.add(id(model))
        if hasattr(model, "start_layer") and hasattr(model, "end_layer"):
            candidates.append((_int(model.start_layer, "start_layer"), _int(model.end_layer, "end_layer")))
        original = getattr(model, "_orig_mod", None)
        model = original if original is not None else getattr(model, "model", None)
    if len(set(candidates)) != 1:
        _fail("HARDWARE_METADATA_INCOMPLETE", "Loaded Qwen3 stage layer range is unavailable or ambiguous")
    start, end = candidates[0]
    layers = stage_layer_counts(experiment)
    expected_start = sum(layers[:stage_index])
    if start != expected_start or end != expected_start + layers[stage_index]:
        _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded Qwen3 layer range differs from the configured PP stage")
    return {"start_layer": start, "end_layer": end, "layer_count": end - start,
            "method": "loaded_qwen_model_start_end_layer"}


def _check_unscaled_qwen_rope(hf: Any) -> None:
    """Accept HF's default-RoPE attribute alias without accepting scaling."""
    thetas: list[int | float] = []

    def theta(value: Any) -> None:
        if type(value) not in (int, float):
            _fail("UNSUPPORTED_VLLM_FEATURE", "Default Qwen3 RoPE theta must be finite and positive")
        try:
            valid = math.isfinite(float(value)) and value > 0
        except (OverflowError, ValueError):
            valid = False
        if not valid:
            _fail("UNSUPPORTED_VLLM_FEATURE", "Default Qwen3 RoPE theta must be finite and positive")
        thetas.append(value)

    for name in ("rope_scaling", "rope_parameters"):
        parameters = getattr(hf, name, None)
        if parameters is None:
            continue
        default_type = None if name == "rope_scaling" else "default"
        if (not isinstance(parameters, dict) or set(parameters) - {"rope_type", "rope_theta"}
                or parameters.get("rope_type", default_type) != "default"):
            _fail("UNSUPPORTED_VLLM_FEATURE", "Only unscaled default Qwen3 RoPE parameters are supported")
        if "rope_theta" in parameters:
            theta(parameters["rope_theta"])
    top_theta = getattr(hf, "rope_theta", None)
    if top_theta is not None:
        theta(top_theta)
    if any(value != thetas[0] for value in thetas[1:]):
        _fail("HARDWARE_IDENTITY_MISMATCH", "Observed Qwen3 RoPE theta aliases disagree")


def _loaded_embedding_alias(runner: Any, experiment: Experiment) -> dict:
    """Observe local Qwen weights without comparing or copying their contents."""
    root = getattr(runner, "model", None)
    pairs: dict[tuple[int, int], tuple[Any, Any]] = {}
    visited: set[int] = set()
    for _ in range(8):
        if root is None or id(root) in visited:
            break
        visited.add(id(root))
        embedding = getattr(getattr(root, "model", None), "embed_tokens", None)
        head = getattr(root, "lm_head", None)
        first, second = getattr(embedding, "weight", None), getattr(head, "weight", None)
        if first is not None and second is not None:
            pairs[id(first), id(second)] = first, second
        original = getattr(root, "_orig_mod", None)
        root = original if original is not None else getattr(root, "model", None)
    if len(pairs) != 1:
        _fail("HARDWARE_METADATA_INCOMPLETE", "Loaded local embedding/head weights are unavailable or ambiguous")
    embedding_weight, head_weight = next(iter(pairs.values()))
    expected_device = getattr(runner, "device", None)
    if expected_device is None:
        _fail("HARDWARE_METADATA_INCOMPLETE", "Loaded embedding proof requires the Runner device")

    def describe(weight: Any) -> dict:
        try:
            shape = [_int(size, "embedding weight shape", 1) for size in tuple(weight.shape)]
            stride = [_int(size, "embedding weight stride") for size in tuple(weight.stride())]
            device = weight.device
            dtype = str(weight.dtype).removeprefix("torch.")
            storage = weight.untyped_storage()
            info: dict[str, Any] = {"shape": shape, "stride": stride, "dtype": dtype, "device": str(device),
                "storage_data_ptr": _int(storage.data_ptr(), "embedding storage pointer", 1),
                "storage_bytes": _int(storage.nbytes(), "embedding storage bytes", 1),
                "tensor_data_ptr": _int(weight.data_ptr(), "embedding tensor pointer", 1),
                "storage_offset": _int(weight.storage_offset(), "embedding storage offset"),
                "element_size": _int(weight.element_size(), "embedding element size", 1)}
            minimum_rows = ((experiment.strategy.model.vocab_size + experiment.strategy.tensor_parallel_size - 1)
                            // experiment.strategy.tensor_parallel_size)
            if (len(shape) != 2 or len(stride) != 2 or shape[0] < minimum_rows
                    or shape[1] != experiment.strategy.model.hidden_size
                    or dtype != experiment.strategy.model.weight_dtype
                    or info["element_size"] != int(experiment.strategy.model.weight_bytes)
                    or getattr(device, "type", None) != "cuda" or str(device) != str(expected_device)
                    or info["storage_bytes"] < shape[0] * shape[1] * info["element_size"]
                    or info["tensor_data_ptr"] != info["storage_data_ptr"] + info["storage_offset"] * info["element_size"]):
                _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded embedding/head storage differs from its model/device identity")
            return info
        except SimulationError:
            raise
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise SimulationError("HARDWARE_METADATA_INCOMPLETE", "Loaded embedding/head storage cannot be observed") from exc

    first_info, second_info = describe(embedding_weight), describe(head_weight)
    same_storage = (first_info["device"] == second_info["device"]
                    and first_info["storage_data_ptr"] == second_info["storage_data_ptr"]
                    and first_info["storage_bytes"] == second_info["storage_bytes"])
    same_view = first_info == second_info
    return {"same_parameter": embedding_weight is head_weight, "same_storage": same_storage,
            "same_tensor_view": same_view, "effective_tie_word_embeddings": same_view,
            "embedding": first_info, "lm_head": second_info,
            "method": "loaded_local_parameter_and_storage_alias", "cross_rank_alias_verified": False}


def _shape_model_matches(runner: Any, experiment: Experiment, *, _phase: str = "strict") -> dict:
    """Check known architectural fields, never infer hidden width from heads."""
    config = runner.model_config
    hf = config.hf_config
    model = experiment.strategy.model
    if _phase not in {"strict", "construction", "prepared"}:
        _fail("UNSUPPORTED_VLLM_ABI", "Unknown model-binding observation phase")
    hf_tied = getattr(hf, "tie_word_embeddings", _MISSING)
    checkpoint_untied = getattr(config, "word_embeddings_untied_by_checkpoint", False)
    if type(hf_tied) is not bool or type(checkpoint_untied) is not bool:
        _fail("HARDWARE_IDENTITY_MISMATCH", "Word-embedding tying metadata must be boolean")
    pending = checkpoint_untied and not hf_tied and model.tie_word_embeddings
    effective_tied = hf_tied
    tying: dict[str, Any] = {"startup_hf_tie_word_embeddings": hf_tied,
        "startup_word_embeddings_untied_by_checkpoint": checkpoint_untied,
        "word_embeddings_untied_by_checkpoint": checkpoint_untied,
        "verification_phase": _phase, "pending_loaded_storage_proof": False,
        "effective_tie_word_embeddings": hf_tied, "local_alias": None}
    if _phase == "construction" and pending:
        effective_tied = model.tie_word_embeddings
        tying.update(pending_loaded_storage_proof=True, effective_tie_word_embeddings=None)
    elif _phase == "prepared":
        if experiment.strategy.pipeline_parallel_size == 1:
            alias = _loaded_embedding_alias(runner, experiment)
            effective_tied = alias["effective_tie_word_embeddings"]
            if effective_tied != hf_tied and not (pending and effective_tied):
                _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded word-embedding storage contradicts its configuration")
            tying.update(local_alias=alias, effective_tie_word_embeddings=effective_tied)
        elif pending:
            _fail("HARDWARE_METADATA_INCOMPLETE", "Checkpoint-untied PP endpoints lack a local tied-weight proof")
        else:
            tying["verification_scope"] = "configured_distinct_pp_endpoint_copies_no_cross_rank_alias_claim"
    fields = {"num_hidden_layers": model.num_layers, "hidden_size": model.hidden_size,
              "intermediate_size": model.intermediate_size, "num_attention_heads": model.num_attention_heads,
              "num_key_value_heads": model.num_kv_heads, "head_dim": model.head_dim,
              "vocab_size": model.vocab_size}
    if getattr(hf, "model_type", None) != "qwen3" or not model.qk_norm or not model.final_norm:
        _fail("HARDWARE_IDENTITY_MISMATCH", "This Qwen3 hook requires Q/K Norm and final RMSNorm identity")
    for field, expected in fields.items():
        if getattr(hf, field, _MISSING) != expected:
            _fail("HARDWARE_IDENTITY_MISMATCH", f"Loaded Qwen3 {field} differs from the experiment")
    if effective_tied != model.tie_word_embeddings:
        _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded Qwen3 tie_word_embeddings differs from the experiment")
    window = getattr(hf, "sliding_window", None)
    window_enabled = getattr(hf, "use_sliding_window", False)
    if window is not None:
        _int(window, "sliding_window", 1)
        if not hasattr(hf, "use_sliding_window"):
            _fail("UNSUPPORTED_VLLM_FEATURE", "An inactive configured window needs explicit use_sliding_window=false")
    if (getattr(hf, "attention_bias", False) or type(window_enabled) is not bool or window_enabled
            or getattr(hf, "quantization_config", None) or getattr(config, "quantization", None) is not None
            or getattr(hf, "hidden_act", "silu") != "silu"):
        _fail("UNSUPPORTED_VLLM_FEATURE", "Loaded model has unsupported attention, quantization or activation behavior")
    _check_unscaled_qwen_rope(hf)
    layer_types = getattr(hf, "layer_types", _MISSING)
    if layer_types is not _MISSING and (not isinstance(layer_types, list)
            or len(layer_types) != model.num_layers or any(value != "full_attention" for value in layer_types)):
        _fail("UNSUPPORTED_VLLM_FEATURE", "Only full-attention Qwen3 layers are supported")
    dtype = str(config.dtype).removeprefix("torch.")
    if dtype != model.weight_dtype or dtype != model.activation_dtype:
        _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded model dtype differs from the declared weight/activation identity")
    if str(getattr(runner, "kv_cache_dtype", "")).removeprefix("torch.") != model.kv_dtype:
        _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded KV cache dtype differs from the declared model identity")
    commit = getattr(hf, "_commit_hash", None)
    requested = getattr(config, "revision", None)
    observed = commit or requested
    if observed is not None and (not isinstance(observed, str) or observed != model.revision):
        _fail("HARDWARE_IDENTITY_MISMATCH", "Visible model revision differs from the declared identity")
    if model.architecture_config_sha256 and not (_phase == "construction" and pending):
        from .qwen3 import model_from_qwen3_config
        raw = deepcopy(hf.to_dict() if callable(getattr(hf, "to_dict", None)) else dict(vars(hf)))
        # The loader may restore physical tying without resetting HF's flag.
        # Only loaded storage evidence may override that field in this copy.
        if _phase == "prepared":
            raw["tie_word_embeddings"] = effective_tied
        imported = model_from_qwen3_config(raw, name=model.name, revision=model.revision, kv_dtype=model.kv_dtype)
        if imported.architecture_config_sha256 != model.architecture_config_sha256:
            _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded Qwen3 architecture fingerprint differs")
    tying["architecture_fingerprint_status"] = ("pending_loaded_storage_proof"
        if model.architecture_config_sha256 and _phase == "construction" and pending else
        "verified" if model.architecture_config_sha256 else "not_requested")
    return {"declared_name": model.name, "declared_revision": model.revision,
            "observed_config_commit_hash": commit, "observed_requested_revision": requested,
            "configuration_revision_verified": observed is not None,
            "revision_verified": observed is not None,
            "weights_revision_independently_verified": False,
            "word_embedding_tying": tying,
            "verification_scope": "visible_configuration_revision_not_full_weight_hash" if observed is not None
                                  else "operator_declared_revision_only"}


class RunnerCapture:
    """Reversible instance hooks around this revision's real execution methods."""

    def __init__(self, runner: Any, experiment: Experiment, *, session_id: str,
                 replica_index: int = 0, stage_index: int = 0, tp_rank: int = 0,
                 runtime_version: str | None = None, clock_domain: str | None = None,
                 torch_module: Any = None, clock_ns: Callable[[], int] | None = None,
                 synchronize: Callable[[], None] | None = None,
                 synchronize_current_stream: Callable[[], None] | None = None,
                 allocator_snapshot: Callable[[], Any] | None = None,
                 source_reference: str = "opt-in pinned vLLM internal runner capture",
                 output_dir: str | Path | None = None, rccl_metadata: dict | None = None,
                 max_records: int = 100000):
        validate_experiment(experiment)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", text(session_id, "session_id")):
            _fail("INVALID_COLLECTION_PLAN", "session_id must be a bounded filename-safe identifier")
        replica_index = _int(replica_index, "replica_index")
        if replica_index >= experiment.strategy.replicas:
            _fail("INVALID_COLLECTION_PLAN", "replica_index is outside the experiment")
        local = replica_experiment(experiment, replica_index)
        stage_index, tp_rank = _int(stage_index, "stage_index"), _int(tp_rank, "tp_rank")
        if stage_index >= local.strategy.pipeline_parallel_size or tp_rank >= local.strategy.tensor_parallel_size:
            _fail("INVALID_COLLECTION_PLAN", "Stage or TP rank is outside its local layout")
        if getattr(runner, "_mlsys_internal_capture", None) is not None:
            _fail("CAPTURE_ALREADY_INSTALLED", "This runner already has a capture installed")
        _signature(getattr(runner, "execute_model", None), ("scheduler_output", "intermediate_tensors"), "execute_model")
        _signature(getattr(runner, "sample_tokens", None), ("grammar_output",), "sample_tokens")
        _signature(getattr(runner, "_prepare_inputs", None), ("scheduler_output", "num_scheduled_tokens"), "_prepare_inputs")
        for name in ("input_batch", "requests", "execute_model_state", "use_async_scheduling"):
            if not hasattr(runner, name):
                _fail("UNSUPPORTED_VLLM_ABI", f"Runner has no {name}")
        self.fixture = any(value is not None for value in (
            torch_module, clock_ns, synchronize, synchronize_current_stream, allocator_snapshot))
        if not self.fixture and (type(runner).__module__ != "vllm.v1.worker.gpu_model_runner"
                or type(runner).__name__ != "GPUModelRunner"):
            _fail("UNSUPPORTED_VLLM_ABI", "Live capture requires the pinned MRv1 GPUModelRunner implementation")
        actual_clock_domain = "fixture-clock" if self.fixture else _clock_domain()
        if clock_domain is not None and not self.fixture and clock_domain != actual_clock_domain:
            _fail("HARDWARE_IDENTITY_MISMATCH", "Live clock_domain must match the actual host boot clock")
        version = _version(runtime_version) if runtime_version is not None else _runtime_version()
        if not self.fixture and runtime_version is not None and _runtime_version() != version:
            _fail("HARDWARE_IDENTITY_MISMATCH", "Declared and installed vLLM versions differ")
        self.runner, self.exp = runner, local
        self.replica_index, self.stage_index, self.tp_rank = replica_index, stage_index, tp_rank
        self._clock = time.perf_counter_ns if clock_ns is None else clock_ns
        self._last_ns = 0
        self._sync: Callable[[], None]
        self._sync_current: Callable[[], None]
        self._allocator: Callable[[], Any]
        self._native_stream_proven = self.fixture
        metadata: dict[str, Any] = {"instrumentation_source": "vllm_internal_runner", "synchronized": True,
            "profile_scope": "synchronous_exclusive_runner_host_service_only",
            "async_profile_eligible": False, "workspace_attribution": "none",
            "sampling_method": "observe_real_sample_return_or_caller_get_output",
            "runtime_policy": {"async_scheduling": bool(runner.use_async_scheduling),
                               "broadcast_pp_output": bool(getattr(runner, "broadcast_pp_output", False))},
            "workload_payload_included": False}
        if getattr(local.strategy, "pipeline_transport", "full_hidden") == "vllm_native":
            metadata["profile_scope"] = (
                "synchronous_exclusive_runner_host_service_excludes_natural_pp_receive_and_receiver_allgather")
        if self.fixture:
            self._sync = synchronize or (lambda: None)
            self._sync_current = synchronize_current_stream or (lambda: None)
            self._allocator = allocator_snapshot or (lambda: [])
            if hasattr(runner, "model_config"):
                metadata["model_binding"] = _shape_model_matches(runner, local, _phase="construction")
        else:
            torch = importlib.import_module("torch")
            if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
                _fail("HARDWARE_UNAVAILABLE", "Internal capture requires live PyTorch HIP hardware")
            current = int(getattr(runner.device, "index", torch.cuda.current_device()))
            properties = torch.cuda.get_device_properties(current)
            expected = local.infra.devices[stage_index * local.strategy.tensor_parallel_size + tp_rank]
            if properties.name != expected.name or properties.total_memory != expected.memory_bytes:
                _fail("HARDWARE_IDENTITY_MISMATCH", "Current rank device differs from the supplied bound experiment")
            if local.calibration.runtime_name != "vllm" or local.calibration.runtime_version != version:
                _fail("HARDWARE_IDENTITY_MISMATCH", "Experiment must retain the actual vLLM runtime identity")
            metadata["model_binding"] = _shape_model_matches(runner, local, _phase="construction")
            actual_parallel = runner.parallel_config
            if (getattr(actual_parallel, "tensor_parallel_size", None) != local.strategy.tensor_parallel_size
                    or getattr(actual_parallel, "pipeline_parallel_size", None) != local.strategy.pipeline_parallel_size):
                _fail("HARDWARE_IDENTITY_MISMATCH", "Actual TP/PP sizes differ from the capture layout")
            validate_identity(profile_identity(local), measured=True)
            if (local.strategy.tensor_parallel_size > 1
                    or (getattr(local.strategy, "pipeline_transport", "full_hidden") == "vllm_native"
                        and local.strategy.pipeline_parallel_size > 1)):
                proof = _loaded_rccl()
                if proof["runtime_version"] != local.calibration.rccl_version:
                    _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded RCCL version differs from the bound experiment")
                expected_digest = (rccl_metadata or {}).get("rccl_library_sha256")
                if expected_digest is not None and expected_digest != proof["library_sha256"]:
                    _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded RCCL library differs from the supplied library digest")
                metadata["rccl_binding"] = proof
            self._sync = lambda: torch.cuda.synchronize(current)
            self._sync_current = lambda: torch.cuda.current_stream(current).synchronize()
            if self._native_pp():
                distributed = importlib.import_module("vllm.distributed")
                group = distributed.get_pp_group()
                tp_group = distributed.get_tp_group()
                backend = str(torch.distributed.get_backend(group.device_group))
                self._native_stream_proven = (type(group).__module__ == "vllm.distributed.parallel_state"
                    and type(group).__name__ == "GroupCoordinator"
                    and getattr(group, "use_cpu_custom_send_recv", None) is False and backend == "nccl"
                    and getattr(group, "world_size", None) == local.strategy.pipeline_parallel_size
                    and getattr(tp_group, "world_size", None) == local.strategy.tensor_parallel_size)
                metadata["native_pp_backend_binding"] = {"backend": backend,
                    "use_cpu_custom_send_recv": getattr(group, "use_cpu_custom_send_recv", None),
                    "api_revision": API_REVISION, "torch_version": str(torch.__version__)}
                metadata["native_pp_stream_binding"] = {
                    "standard_process_group_nccl": self._native_stream_proven,
                    "actual_backend": backend,
                    "group_type": type(group).__module__ + "." + type(group).__name__,
                    "contract": "current_to_communication_stream_until_work_wait_v1"}
            def counters() -> list[dict]:
                return [{"device_index": current, "allocated_bytes": int(torch.cuda.memory_allocated(current)),
                         "reserved_bytes": int(torch.cuda.memory_reserved(current)),
                         "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(current)),
                         "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(current))}]
            self._allocator = counters
            metadata["device"] = {"visible_index": current, "name": properties.name,
                "memory_bytes": int(properties.total_memory), "architecture": getattr(properties, "gcnArchName", None),
                "visibility": {name: os.environ.get(name) for name in
                               ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}}
            metadata["torch_version"] = str(torch.__version__)
            metadata["hip_build_version"] = str(torch.version.hip)
        self.header: dict[str, Any] = {"schema_version": 1, "kind": "vllm_internal_capture",
            "source": {"kind": "test_fixture" if self.fixture else "measured",
                       "reference": text(source_reference, "source_reference")},
            "session_id": session_id, "runtime_version": version, "api_revision": API_REVISION,
            "clock_domain": clock_domain or actual_clock_domain,
            "replica_index": replica_index, "stage_index": stage_index, "tp_rank": tp_rank,
            "tensor_parallel_size": local.strategy.tensor_parallel_size,
            "pipeline_parallel_size": local.strategy.pipeline_parallel_size,
            "identity": profile_identity(local),
            "experiment": asdict(replace(local, workload=(), execution=replace(local.execution, warmup=()))),
            "metadata": metadata, "hardware_validated": False}
        self.records: list[dict] = []
        self.errors: list[dict] = []
        self._pending: dict | None = None
        self._active: dict | None = None
        self._counter = 0
        self._max = _int(max_records, "max_records", 1)
        self._closed = False
        self._flushed: Path | None = None
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._saved: dict[str, Any] = {}
        self._future_methods: list[tuple[Any, Any]] = []
        self._receive_methods: list[tuple[Any, Any, Any]] = []
        self._lock = threading.RLock()
        self._limit_reached = False
        self._runtime_bound = False
        self._loaded_tying_bound: dict | None = None
        self._expected_runtime_hash = local.calibration.runtime_config_sha256
        self._install()

    def _bind_runtime_config(self, sampling_policy: dict | None) -> None:
        config = getattr(self.runner, "vllm_config", self.runner)
        snapshot = _execution_config(config, self.exp, bool(self.runner.use_async_scheduling))
        snapshot["actual_attention_backends"] = _attention_backends(self.runner)
        snapshot["torch_build"] = {key: self.header["metadata"].get(key) for key in ("torch_version", "hip_build_version")}
        snapshot["sampling_policy"] = sampling_policy
        if not self.fixture and not snapshot["actual_attention_backends"]:
            _fail("HARDWARE_METADATA_INCOMPLETE", "Actual attention backend was not available at the prepared-batch boundary")
        digest = _config_digest(snapshot)
        if self._expected_runtime_hash and self._expected_runtime_hash != digest:
            _fail("HARDWARE_IDENTITY_MISMATCH", "Configured Runtime execution fingerprint differs from the actual runner")
        if self._runtime_bound and self.header["metadata"]["runtime_config_sha256"] != digest:
            _fail("HARDWARE_IDENTITY_MISMATCH", "Runtime execution configuration changed during capture")
        if not self._runtime_bound:
            original = self.exp.calibration.runtime_config_sha256
            self.exp = replace(self.exp, calibration=replace(self.exp.calibration, runtime_config_sha256=digest))
            self.header["identity"] = profile_identity(self.exp)
            self.header["experiment"] = asdict(replace(self.exp, workload=(), execution=replace(self.exp.execution, warmup=())))
            self.header["metadata"].update(runtime_execution_config=snapshot, runtime_config_sha256=digest,
                runtime_config_binding={"input_sha256": original, "actual_sha256": digest,
                                        "bound_at": "first_prepared_batch"})
            self._runtime_bound = True

    def _now(self) -> int:
        value = _int(self._clock(), "monotonic clock")
        if value < self._last_ns:
            _fail("INVALID_TIMING_SAMPLE", "Capture clock moved backwards")
        self._last_ns = value
        return value

    def _error(self, frame: dict | None, exc: Exception, status: str = "UNSUPPORTED") -> None:
        code = exc.code if isinstance(exc, SimulationError) else "CAPTURE_FAILED"
        data = {"error_code": code, "message": str(exc)}
        if frame is not None:
            frame.update(status=status, profile_eligible=False, **data)
        else:
            if len(self.errors) < min(self._max, 1000):
                self.errors.append(data)
            else:
                metadata = self.header["metadata"]
                metadata["dropped_capture_errors"] = metadata.get("dropped_capture_errors", 0) + 1

    def _install(self) -> None:
        runner = self.runner
        for name in ("execute_model", "sample_tokens", "_prepare_inputs"):
            self._saved[name] = vars(runner).get(name, _MISSING)
        execute, sample, prepare = runner.execute_model, runner.sample_tokens, runner._prepare_inputs
        @functools.wraps(prepare)
        def prepare_wrapper(scheduler_output, num_scheduled_tokens):
            if self._active is not None:
                try:
                    self._snapshot(self._active, scheduler_output, num_scheduled_tokens)
                except Exception as exc:
                    self._error(self._active, exc)
            return prepare(scheduler_output, num_scheduled_tokens)
        @functools.wraps(execute)
        def execute_wrapper(scheduler_output, intermediate_tensors=None):
            with self._lock:
                frame = self._begin(scheduler_output)
                self._active = frame
                release_receive: Callable[[], None] | None = None
                try:
                    if (frame is not None and frame.get("status") != "UNSUPPORTED"
                            and frame.get("profile_eligible") is True
                            and self._native_pp() and self.stage_index > 0):
                        try:
                            release_receive = self._observe_native_receive(frame, intermediate_tensors)
                        except Exception as exc:
                            self._error(frame, exc)
                    try:
                        output = execute(scheduler_output, intermediate_tensors)
                    finally:
                        if release_receive is not None:
                            release_receive()
                except BaseException as exc:
                    if frame is not None and not self._closed:
                        self._error(frame, exc if isinstance(exc, Exception) else RuntimeError(type(exc).__name__), "FAILED")
                    self._active = None
                    raise
                self._active = None
                if self._closed:
                    return output
                if frame is not None:
                    self._safe_transport(frame, intermediate_tensors, "incoming")
                    if frame.get("status") == "UNSUPPORTED":
                        self._finish_clock(frame)
                    elif self.stage_index != self.exp.strategy.pipeline_parallel_size - 1:
                        if output is None or runner.execute_model_state is not None:
                            self._error(frame, SimulationError("UNSUPPORTED_VLLM_ABI", "Nonfinal stage did not return completed forward tensors"))
                        else:
                            self._safe_transport(frame, output, "outgoing")
                            self._complete(frame, output, sampling=False)
                    elif output is None:
                        self._pending = frame
                    else:
                        # This ABI's positive final-stage path transfers state
                        # to sample_tokens. Other early-return paths are raw only.
                        self._error(frame, SimulationError("UNSUPPORTED_VLLM_BOUNDARY", "Final stage bypassed the supported sampling boundary"))
                        self._finish_clock(frame)
                return output
        @functools.wraps(sample)
        def sample_wrapper(grammar_output):
            with self._lock:
                frame = self._pending
                self._pending = None
                if frame is not None and grammar_output is not None:
                    self._error(frame, SimulationError("UNSUPPORTED_VLLM_FEATURE", "Structured-output sampling is outside this capture contract"))
                try:
                    output = sample(grammar_output)
                except BaseException as exc:
                    if frame is not None:
                        self._error(frame, exc if isinstance(exc, Exception) else RuntimeError(type(exc).__name__), "FAILED")
                    raise
                if frame is not None:
                    if callable(getattr(output, "get_output", None)):
                        try:
                            self._observe_future(frame, output)
                        except Exception as exc:
                            self._error(frame, exc)
                    else:
                        self._complete(frame, output, sampling=True)
                return output
        try:
            runner.execute_model = execute_wrapper
            runner.sample_tokens = sample_wrapper
            runner._prepare_inputs = prepare_wrapper
            runner._mlsys_internal_capture = self
        except Exception:
            self.close()
            raise

    def _begin(self, scheduler_output: Any) -> dict | None:
        if self._closed:
            return None
        if self._pending is not None:
            self._error(self._pending, SimulationError("INCOMPLETE_VLLM_ITERATION", "Another execute began before the pending sample completed"), "INCOMPLETE")
            self._pending = None
        try:
            scheduled = scheduler_output.num_scheduled_tokens
            if not isinstance(scheduled, dict):
                _fail("UNSUPPORTED_VLLM_ABI", "SchedulerOutput.num_scheduled_tokens must be a dictionary")
            if not scheduled:
                return None
            if len(self.records) >= self._max:
                if not self._limit_reached:
                    self._limit_reached = True
                    self._error(None, SimulationError("CAPTURE_LIMIT", "Capture record limit reached"))
                return None
            frame: dict[str, Any] = {"sample_id": f"iteration-{self._counter}", "iteration_id": self._counter,
                "request_ids": [], "query_tokens": [], "context_lengths": [], "prompt_lengths": [],
                "phase": None, "status": "INCOMPLETE", "forward_complete": False,
                "sampling_required": self.stage_index == self.exp.strategy.pipeline_parallel_size - 1,
                "sampling_complete": False, "async_output": bool(self.runner.use_async_scheduling),
                "deferred_output_resolved": False, "profile_eligible": not self.runner.use_async_scheduling,
                "timing_boundary": _BOUNDARY,
                "capture_boundary": "runner_execute_to_sample_host_ready" if
                    self.stage_index == self.exp.strategy.pipeline_parallel_size - 1 else "runner_execute_host_ready",
                "workspace_peak_bytes": None, "allocator_baseline": [], "allocator_after": [],
                "resumed_request_ids": [], "output_history_lengths": [], "sampled_token_counts": []}
            self._counter += 1
            self.records.append(frame)
            if (getattr(self.runner, "speculative_config", None) is not None
                    or getattr(self.runner, "is_pooling_model", False)
                    or getattr(self.runner, "broadcast_pp_output", False)
                    or getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
                    or getattr(scheduler_output, "scheduled_encoder_inputs", {})
                    or getattr(scheduler_output, "kv_connector_metadata", None) is not None
                    or getattr(scheduler_output, "ec_connector_metadata", None) is not None):
                _fail("UNSUPPORTED_VLLM_FEATURE", "Speculation, pooling, broadcast or external/multimodal KV are unsupported")
            config = getattr(self.runner, "vllm_config", None)
            if self._native_pp() and not self._native_stream_proven:
                _fail("UNSUPPORTED_VLLM_FEATURE", "Native PP capture requires the standard ProcessGroupNCCL stream contract")
            passes = getattr(getattr(config, "compilation_config", None), "pass_config", None)
            if (getattr(config, "lora_config", None) is not None
                    or getattr(config, "kv_transfer_config", None) is not None
                    or getattr(config, "ec_transfer_config", None) is not None
                    or getattr(passes, "enable_sequence_parallelism", False)
                    or getattr(passes, "enable_sp", False)):
                _fail("UNSUPPORTED_VLLM_FEATURE", "LoRA, transfer connectors or sequence parallelism are outside this capture contract")
            parallel = getattr(self.runner, "parallel_config", None)
            if parallel is not None and (getattr(parallel, "data_parallel_size", 1) != 1
                    or getattr(parallel, "use_ubatching", False)
                    or getattr(parallel, "enable_expert_parallel", False)
                    or getattr(parallel, "decode_context_parallel_size", 1) != 1
                    or getattr(parallel, "prefill_context_parallel_size", 1) != 1):
                _fail("UNSUPPORTED_VLLM_FEATURE", "Native DP/EP/context parallelism or DBO is outside the independent-stage capture model")
            cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
            resumed: set[str] = getattr(cached, "resumed_req_ids", set()) if cached is not None else set()
            if resumed:
                frame["resumed_request_ids"] = sorted(resumed)
                _fail("UNSUPPORTED_VLLM_FEATURE", "Resumed/recomputed history is retained as raw evidence")
            # Worker posted irecv before entering the Runner. Synchronizing here
            # would consume its wait before the Runner's natural lazy boundary.
            if not (self._native_pp() and self.stage_index > 0):
                self._sync()
            frame["allocator_baseline"] = deepcopy(self._allocator())
            frame["start_monotonic_ns"] = self._now()
            return frame
        except Exception as exc:
            if "frame" in locals():
                self._error(frame, exc)
                return frame
            self._error(None, exc)
            return None

    def _native_pp(self) -> bool:
        return (getattr(self.exp.strategy, "pipeline_transport", "full_hidden") == "vllm_native"
                and self.exp.strategy.pipeline_parallel_size > 1)

    def _observe_native_receive(self, frame: dict, value: Any) -> Callable[[], None]:
        """Observe the original lazy consumer without evaluating its tensors."""
        names = {base.__name__ for base in type(value).__mro__}
        lazy = "AsyncIntermediateTensors" in names
        stored = vars(value) if hasattr(value, "__dict__") else {}
        info = frame.setdefault("transport_metadata", {}).setdefault("incoming", {})
        info.update(lazy_incoming=lazy, natural_receive_observed=False, receive_wait_completed=False)
        if not lazy:
            if not isinstance(stored.get("tensors"), dict):
                _fail("UNSUPPORTED_VLLM_ABI", "Native PP requires observable intermediate tensor storage")
            self._external_receive_ready(frame, info)
            return lambda: None
        if not self.fixture and (type(value).__module__ != "vllm.v1.worker.gpu_worker"
                or type(value).__name__ != "AsyncIntermediateTensors"):
            _fail("UNSUPPORTED_VLLM_ABI", "Native lazy PP requires the pinned AsyncIntermediateTensors class")
        if ({"_comm_waited", "_comm_handles", "_comm_postprocess", "tensors"} - stored.keys()
                or type(stored.get("_comm_waited")) is not bool
                or not isinstance(stored.get("tensors"), dict)
                or not all(stored.get(name) is None or isinstance(stored.get(name), list)
                           for name in ("_comm_handles", "_comm_postprocess"))):
            _fail("UNSUPPORTED_VLLM_ABI", "Native lazy PP synchronization storage differs from the pinned ABI")
        if stored["_comm_waited"]:
            self._external_receive_ready(frame, info)
            if not self._closed:
                info["already_waited_external"] = True
            return lambda: None
        original = value.wait_for_comm
        _signature(original, (), "AsyncIntermediateTensors.wait_for_comm")
        previous = stored.get("wait_for_comm", _MISSING)

        def release() -> None:
            if vars(value).get("wait_for_comm") is observe:
                if previous is _MISSING:
                    vars(value).pop("wait_for_comm", None)
                else:
                    value.wait_for_comm = previous
            self._receive_methods = [entry for entry in self._receive_methods if entry[0] is not value]

        @functools.wraps(original)
        def observe():
            # Repeated calls retain the source method's idempotence and never
            # create additional intervals for its no-op path.
            if self._closed or vars(value).get("_comm_waited") is True:
                return original()
            start = None
            try:
                if info.get("natural_receive_observed"):
                    _fail("UNSUPPORTED_VLLM_BOUNDARY", "A native receive restarted after completion")
                # ProcessGroupNCCL posts irecv on its own communication stream.
                # Before Work.wait the current compute stream does not depend
                # on that receive. Drain prepared H2D work here so the later
                # receive completion sync cannot subtract it from stage cost.
                self._sync_current()
                if not self._closed:
                    info["pre_wait_compute_stream_synchronized"] = True
                    start = self._now()
            except Exception as exc:
                if not self._closed:
                    self._error(frame, exc)
            try:
                result = original()
            except BaseException as exc:
                if not self._closed:
                    self._error(frame, exc if isinstance(exc, Exception) else RuntimeError(type(exc).__name__), "FAILED")
                raise
            if not self._closed and start is not None:
                try:
                    if vars(value).get("_comm_waited") is not True:
                        _fail("UNSUPPORTED_VLLM_ABI", "Natural lazy receive did not mark communication complete")
                    # Original waits and postprocess must run before completion
                    # synchronization. The following local input copy remains
                    # inside the independent stage service interval.
                    self._sync()
                    if self._closed:
                        return result
                    end = self._now()
                    if end < start:
                        _fail("INVALID_TIMING_SAMPLE", "Native receive interval moved backwards")
                    frame["pp_receive_timing"] = {"method": "natural_wait_for_comm_v1",
                        "intervals": [{"start_monotonic_ns": start, "end_monotonic_ns": end}],
                        "excluded_ns": end - start}
                    info.update(natural_receive_observed=True, receive_wait_completed=True,
                                receive_completion_scope="natural_wait_for_comm")
                except Exception as exc:
                    self._error(frame, exc)
            return result

        try:
            value.wait_for_comm = observe
            self._receive_methods.append((value, previous, observe))
        except Exception:
            release()
            raise
        return release

    def _external_receive_ready(self, frame: dict, info: dict) -> None:
        """Exclude already-enqueued external communication before stage entry."""
        self._sync()
        if self._closed:
            return
        baseline = deepcopy(self._allocator())
        if self._closed:
            return
        frame["allocator_baseline"] = baseline
        frame["start_monotonic_ns"] = self._now()
        info.update(receive_wait_completed=True, receive_completion_scope="before_runner_entry",
                    external_receive_synchronized=True)

    def _native_descriptor(self, tensors: Any) -> dict:
        if not isinstance(tensors, dict) or set(tensors) != {"hidden_states", "residual"}:
            _fail("UNSUPPORTED_VLLM_ABI", "Native PP requires actual hidden_states and residual tensors")
        entries: list[dict[str, Any]] = []
        tp = self.exp.strategy.tensor_parallel_size
        padded_tokens = None
        for name in ("hidden_states", "residual"):
            tensor = tensors[name]
            shape = [_int(size, f"{name}.shape", 1) for size in tuple(getattr(tensor, "shape", ()))]
            if len(shape) != 2 or (padded_tokens is not None and shape[0] != padded_tokens):
                _fail("UNSUPPORTED_VLLM_ABI", "Native PP tensor shapes must share their physical token rows")
            padded_tokens = shape[0]
            dtype = str(getattr(tensor, "dtype", "")).removeprefix("torch.")
            if not callable(getattr(tensor, "numel", None)) or not callable(getattr(tensor, "element_size", None)):
                _fail("UNSUPPORTED_VLLM_ABI", "Native PP tensor metadata methods are missing")
            numel, element_size = _int(tensor.numel(), f"{name}.numel", 1), _int(
                tensor.element_size(), f"{name}.element_size", 1)
            full_bytes = numel * element_size
            sliced = tp > 1 and numel % tp == 0
            entries.append({"name": name, "shape": shape, "dtype": dtype,
                "element_size": element_size, "numel": numel, "full_bytes": full_bytes,
                "tp_sliced": sliced, "wire_bytes_per_rank": full_bytes // tp if sliced else full_bytes,
                "receiver_allgather": sliced})
        descriptor = {"contract": "vllm_native_pp_v1", "padded_tokens": padded_tokens, "tensors": entries,
            "resident_bytes_per_rank": sum(entry["full_bytes"] for entry in entries),
            "wire_bytes_per_rank": sum(entry["wire_bytes_per_rank"] for entry in entries)}
        return validate_native_pp_descriptor(self.exp, descriptor)

    def _bind_native_persistent(self) -> None:
        """Bind actual loaded receive buffers, not just the current small batch."""
        if not self._native_pp() or self.stage_index == 0:
            return
        capacity = _int(getattr(self.runner, "max_num_tokens", None), "runner.max_num_tokens", 1)
        if capacity > pipeline_transport_max_tokens(self.exp):
            _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded native PP persistent capacity exceeds the configured bound")
        value = getattr(self.runner, "intermediate_tensors", None)
        stored = vars(value) if value is not None and hasattr(value, "__dict__") else {}
        descriptor = self._native_descriptor(stored.get("tensors"))
        if descriptor["padded_tokens"] != capacity:
            _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded native PP persistent shapes differ from runner.max_num_tokens")
        binding = {"max_num_tokens": capacity, "descriptor": descriptor,
                   "scope": "loaded_runner_intermediate_buffers"}
        previous = self.header["metadata"].get("native_pp_persistent_binding")
        if previous is not None and previous != binding:
            _fail("HARDWARE_IDENTITY_MISMATCH", "Native PP persistent buffer capacity changed during capture")
        self.header["metadata"]["native_pp_persistent_binding"] = binding

    def _transport(self, frame: dict, value: Any, direction: str) -> None:
        """Inspect only stored metadata; never dereference lazy .tensors."""
        if self.exp.strategy.pipeline_parallel_size == 1:
            return
        names = {base.__name__ for base in type(value).__mro__} if value is not None else set()
        lazy = "AsyncIntermediateTensors" in names
        stored = vars(value) if value is not None and hasattr(value, "__dict__") else {}
        info: dict[str, Any]
        if self._native_pp():
            info = frame.setdefault("transport_metadata", {}).setdefault(direction, {})
            info.update(object_type=type(value).__module__ + "." + type(value).__name__,
                        lazy_incoming=lazy, modeled_transport="vllm_native_pp_v1",
                        profile_transport_compatible=False, incoming_communication_excluded=False)
            self.header["metadata"]["pp_transport_scope"] = (
                "native_two_tensor_sliced_p2p_receiver_allgather_observe_natural_receive")
            if direction == "incoming" and self.stage_index == 0 and value is None:
                info.update(profile_transport_compatible=True, incoming_communication_excluded=True,
                            default_tp_sliced_p2p_receiver_allgather=False)
                return
            descriptor = self._native_descriptor(stored.get("tensors"))
            info.update(native_descriptor=descriptor,
                        materialized_tensor_keys=[entry["name"] for entry in descriptor["tensors"]],
                        default_tp_sliced_p2p_receiver_allgather=any(
                            entry["tp_sliced"] for entry in descriptor["tensors"]))
            if (not frame.get("query_tokens")
                    or descriptor["padded_tokens"] < sum(frame["query_tokens"])):
                _fail("UNSUPPORTED_VLLM_BOUNDARY", "Native PP payload does not cover the prepared query tokens")
            persistent = self.header["metadata"].get("native_pp_persistent_binding")
            if self.stage_index > 0 and (not self.fixture or persistent is not None):
                if (not isinstance(persistent, dict)
                        or descriptor["padded_tokens"] > persistent["max_num_tokens"]):
                    _fail("UNSUPPORTED_VLLM_BOUNDARY", "Native PP payload exceeds the actual loaded persistent capacity")
            if direction == "incoming" and info.get("receive_wait_completed") is not True:
                frame.update(profile_eligible=False, canonical_rejection_reason="NATIVE_PP_RECEIVE_NOT_OBSERVED")
                return
            info.update(profile_transport_compatible=True, incoming_communication_excluded=True)
            return
        tensors = stored.get("tensors") if not lazy else None
        keys = sorted(str(key) for key, tensor in tensors.items() if tensor is not None) if isinstance(tensors, dict) else None
        info = {"object_type": type(value).__module__ + "." + type(value).__name__,
            "lazy_incoming": lazy, "materialized_tensor_keys": keys,
            "modeled_transport": "one_unpadded_full_hidden_tensor_per_rank",
            "default_tp_sliced_p2p_receiver_allgather": self.exp.strategy.tensor_parallel_size > 1,
            "profile_transport_compatible": False, "incoming_communication_excluded": False}
        frame.setdefault("transport_metadata", {})[direction] = info
        self.header["metadata"]["pp_transport_scope"] = "inspect_without_wait_or_tensor_property_access"
        reason = None
        if self.exp.strategy.tensor_parallel_size > 1:
            reason = "TP_PP_SLICED_TRANSPORT_NOT_MODELED"
        elif lazy:
            reason = "LAZY_PP_INCOMING_IN_TIMED_RUNNER"
        elif direction == "incoming" and self.stage_index == 0 and value is None:
            info.update(profile_transport_compatible=True, incoming_communication_excluded=True)
            return
        elif keys != ["hidden_states"]:
            reason = "PP_INTERMEDIATE_PAYLOAD_NOT_MODELED"
        else:
            assert isinstance(tensors, dict)
            tensor = tensors["hidden_states"]
            shape = tuple(getattr(tensor, "shape", ()))
            info["materialized_shape"] = list(shape)
            if (len(shape) != 2 or shape[-1] != self.exp.strategy.model.hidden_size
                    or not frame.get("query_tokens") or shape[0] != sum(frame["query_tokens"])):
                reason = "PP_PADDED_OR_UNKNOWN_PAYLOAD_NOT_MODELED"
        if reason is not None:
            frame.update(profile_eligible=False, canonical_rejection_reason=reason)
        else:
            info.update(profile_transport_compatible=True, incoming_communication_excluded=True)

    def _safe_transport(self, frame: dict, value: Any, direction: str) -> None:
        try:
            self._transport(frame, value, direction)
        except Exception as exc:
            self._error(frame, exc)

    def _snapshot(self, frame: dict, scheduler_output: Any, num_scheduled_tokens: Any) -> None:
        previous_binding = self.header["metadata"].get("model_binding")
        if previous_binding is not None:
            binding = _shape_model_matches(self.runner, self.exp, _phase="prepared")
            tying = binding["word_embedding_tying"]
            prior_tying = previous_binding["word_embedding_tying"]
            tying["startup_hf_tie_word_embeddings"] = prior_tying["startup_hf_tie_word_embeddings"]
            tying["startup_word_embeddings_untied_by_checkpoint"] = prior_tying[
                "startup_word_embeddings_untied_by_checkpoint"]
            tying["constructor_pending_loaded_storage_proof"] = prior_tying.get(
                "constructor_pending_loaded_storage_proof", prior_tying["pending_loaded_storage_proof"])
            if self._loaded_tying_bound is not None and self._loaded_tying_bound != tying:
                _fail("HARDWARE_IDENTITY_MISMATCH", "Loaded word-embedding storage binding changed during capture")
            self._loaded_tying_bound = deepcopy(tying)
            self.header["metadata"]["model_binding"] = binding
        if not self.fixture:
            self.header["metadata"]["stage_layer_binding"] = _stage_layer_binding(self.runner, self.exp, self.stage_index)
            self._bind_native_persistent()
        batch = self.runner.input_batch
        count = _int(batch.num_reqs, "input_batch.num_reqs", 1)
        ids = _list(batch.req_ids, "input_batch.req_ids")[:count]
        queries = [_int(value, "query_tokens", 1) for value in _list(num_scheduled_tokens, "num_scheduled_tokens")[:count]]
        contexts = [_int(value, "context_lengths") for value in _list(batch.num_computed_tokens_cpu,
                                                                                   "num_computed_tokens_cpu")[:count]]
        if len(ids) != count or len(queries) != count or len(contexts) != count or len(set(ids)) != count:
            _fail("UNSUPPORTED_VLLM_ABI", "Prepared input arrays do not cover the same unique request batch")
        scheduled = scheduler_output.num_scheduled_tokens
        if set(ids) != set(scheduled) or any(scheduled[name] != query for name, query in zip(ids, queries)):
            _fail("UNSUPPORTED_VLLM_ABI", "Prepared order/query lengths disagree with the scheduler output")
        prompts, histories, roles = [], [], []
        policies: list[dict | None] = []
        for name, query, context in zip(ids, queries, contexts):
            state = self.runner.requests[name]
            if (getattr(state, "mm_features", []) or getattr(state, "lora_request", None) is not None
                    or getattr(state, "prompt_embeds", None) is not None or getattr(state, "pooling_params", None) is not None):
                _fail("UNSUPPORTED_VLLM_FEATURE", "Request carries unsupported multimodal, LoRA, embedding or pooling state")
            prompt = _int(state.num_prompt_tokens, "num_prompt_tokens", 1)
            history = len(_list(state.output_token_ids, "output_token_ids"))
            params = getattr(state, "sampling_params", None)
            if params is not None and (getattr(params, "structured_outputs", None) is not None
                    or getattr(params, "logits_processors", None) or getattr(params, "allowed_token_ids", None)):
                _fail("UNSUPPORTED_VLLM_FEATURE", "Per-request constrained sampling is outside the supported domain")
            policies.append({name: _plain(getattr(params, name, None)) for name in
                ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
                 "repetition_penalty", "logprobs", "prompt_logprobs")} if params is not None else None)
            if history and context < prompt:
                _fail("UNSUPPORTED_VLLM_FEATURE", "A history-bearing re-prefill is not a new prompt")
            role = "prefill" if context < prompt else "decode"
            if role == "prefill" and context + query > prompt or role == "decode" and query != 1:
                _fail("UNSUPPORTED_VLLM_FEATURE", "Prepared request crosses unsupported prefill/decode or speculative boundaries")
            prompts.append(prompt)
            histories.append(history)
            roles.append(role)
        frame.update(request_ids=ids, query_tokens=queries, context_lengths=contexts,
                     prompt_lengths=prompts, output_history_lengths=histories, phase=batch_phase(roles))
        if any(policy != policies[0] for policy in policies):
            _fail("UNSUPPORTED_VLLM_FEATURE", "Heterogeneous sampling policies require separate captures")
        self._bind_runtime_config(policies[0])
        if frame["phase"] == "mixed":
            frame["request_phases"] = roles

    def _finish_clock(self, frame: dict) -> None:
        if self._closed:
            return
        try:
            # An unconsumed native lazy receive must never be consumed by the
            # instrumentation's final synchronization, even for a raw sample.
            incoming = frame.get("transport_metadata", {}).get("incoming", {})
            if not (self._native_pp() and self.stage_index > 0) or incoming.get("receive_wait_completed") is True:
                self._sync()
            if self._closed:
                return
            end = self._now()
            frame["end_monotonic_ns"] = end
            if "start_monotonic_ns" in frame:
                wall_ns = end - frame["start_monotonic_ns"]
                receive_timing = frame.get("pp_receive_timing")
                excluded_ns = receive_timing["excluded_ns"] if receive_timing is not None else 0
                duration = (wall_ns - excluded_ns) / 1e6
                if not math.isfinite(duration) or duration <= 0:
                    _fail("INVALID_TIMING_SAMPLE", "Captured service must advance the monotonic clock")
                frame["duration_ms"] = duration
                if receive_timing is not None:
                    receive_timing["wall_duration_ms"] = wall_ns / 1e6
            frame["allocator_after"] = deepcopy(self._allocator())
        except Exception as exc:
            self._error(frame, exc, "FAILED")

    def _complete(self, frame: dict, output: Any, *, sampling: bool) -> None:
        if self._closed:
            return
        try:
            if not frame.get("request_ids"):
                _fail("UNSUPPORTED_VLLM_ABI", "No actual prepared batch was observed")
            if sampling:
                ids = _list(output.req_ids, "ModelRunnerOutput.req_ids")
                values = _list(output.sampled_token_ids, "ModelRunnerOutput.sampled_token_ids")
                if ids != frame["request_ids"] or len(values) != len(ids):
                    _fail("UNSUPPORTED_VLLM_ABI", "Sample output does not match the captured input order")
                counts = [len(_list(tokens, "sampled tokens")) for tokens in values]
                expected = [int(context + query >= prompt) for context, query, prompt in
                            zip(frame["context_lengths"], frame["query_tokens"], frame["prompt_lengths"])]
                if counts != expected:
                    _fail("UNSUPPORTED_VLLM_BOUNDARY", "Output sampling counts do not match final-prompt/decode membership")
                frame["sampled_token_counts"] = counts
            self._finish_clock(frame)
            if self._closed:
                return
            if frame.get("status") not in ("UNSUPPORTED", "FAILED"):
                frame.update(status="COMPLETED", forward_complete=True, sampling_complete=sampling)
                if frame["async_output"]:
                    frame.update(profile_eligible=False, canonical_rejection_reason="ASYNC_CONSUMPTION_BOUNDARY")
        except Exception as exc:
            self._error(frame, exc)
            self._finish_clock(frame)

    def _observe_future(self, frame: dict, output: Any) -> None:
        frame["async_output"] = True
        frame["profile_eligible"] = False
        original = output.get_output
        previous = vars(output).get("get_output", _MISSING)
        captured = False
        def release() -> None:
            with self._lock:
                if previous is _MISSING:
                    vars(output).pop("get_output", None)
                else:
                    output.get_output = previous
                self._future_methods = [entry for entry in self._future_methods if entry[0] is not output]
        @functools.wraps(original)
        def observe():
            nonlocal captured
            try:
                result = original()
            except BaseException as exc:
                with self._lock:
                    if not captured and not self._closed:
                        self._error(frame, exc if isinstance(exc, Exception) else RuntimeError(type(exc).__name__), "FAILED")
                        captured = True
                release()
                raise
            with self._lock:
                if not captured and not self._closed:
                    frame["deferred_output_resolved"] = True
                    self._complete(frame, result, sampling=True)
                    captured = True
            release()
            return result
        try:
            output.get_output = observe
            self._future_methods.append((output, previous))
        except Exception as exc:
            self._error(frame, SimulationError("UNSUPPORTED_VLLM_ABI", f"Cannot observe the real async result: {exc}"))

    def export_capture(self) -> dict:
        with self._lock:
            return {**deepcopy(self.header), "records": deepcopy(self.records), "events": [],
                    "capture_errors": deepcopy(self.errors), "capture_closed": self._closed}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._pending is not None:
                self._error(self._pending, SimulationError("INCOMPLETE_VLLM_ITERATION", "Capture closed before sampling"), "INCOMPLETE")
                self._pending = None
            if self._active is not None:
                self._error(self._active, SimulationError("INCOMPLETE_VLLM_ITERATION", "Capture closed during execution"), "INCOMPLETE")
                self._active = None
            for name, original in self._saved.items():
                if original is _MISSING:
                    vars(self.runner).pop(name, None)
                else:
                    setattr(self.runner, name, original)
            if getattr(self.runner, "_mlsys_internal_capture", None) is self:
                delattr(self.runner, "_mlsys_internal_capture")
            for future, previous in self._future_methods:
                if previous is _MISSING:
                    vars(future).pop("get_output", None)
                else:
                    future.get_output = previous
            self._future_methods.clear()
            for value, previous, observe in self._receive_methods:
                if vars(value).get("wait_for_comm") is observe:
                    if previous is _MISSING:
                        vars(value).pop("wait_for_comm", None)
                    else:
                        value.wait_for_comm = previous
            self._receive_methods.clear()

    def flush(self) -> str | None:
        with self._lock:
            if self._flushed is not None:
                return str(self._flushed)
            self.close()
            if self._output_dir is None:
                return None
            self._output_dir.mkdir(parents=True, exist_ok=True)
            path = self._output_dir / (f"{self.header['session_id']}-r{self.replica_index}-s{self.stage_index}"
                f"-tp{self.tp_rank}-pid{os.getpid()}-{uuid.uuid4().hex}.json")
            with path.open("x", encoding="utf-8") as stream:
                json.dump(self.export_capture(), stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            self._flushed = path
            return str(path)


def install_runner_hook(runner: Any, experiment: Experiment, **kwargs: Any) -> RunnerCapture:
    return RunnerCapture(runner, experiment, **kwargs)


class SchedulerCapture:
    """Observe real enqueue and published EngineCoreOutput lifecycle events."""

    def __init__(self, scheduler: Any, experiment: Experiment, *, session_id: str,
                 replica_index: int = 0, runtime_version: str | None = None,
                 clock_domain: str | None = None, clock_ns: Callable[[], int] | None = None,
                 source_reference: str = "opt-in pinned vLLM Scheduler capture",
                 output_dir: str | Path | None = None, record_prompt_token_ids: bool = False,
                 max_events: int = 1000000, _constructor_origin: bool = False):
        validate_experiment(experiment)
        local = replica_experiment(experiment, replica_index)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", text(session_id, "session_id")):
            _fail("INVALID_COLLECTION_PLAN", "session_id must be filename safe")
        if type(record_prompt_token_ids) is not bool:
            _fail("INVALID_COLLECTION_PLAN", "record_prompt_token_ids must be boolean")
        if getattr(scheduler, "_mlsys_scheduler_capture", None) is not None:
            _fail("CAPTURE_ALREADY_INSTALLED", "This Scheduler already has a capture installed")
        _signature(getattr(scheduler, "add_request", None), ("request",), "Scheduler.add_request")
        _signature(getattr(scheduler, "update_from_output", None),
                   ("scheduler_output", "model_runner_output"), "Scheduler.update_from_output")
        _signature(getattr(scheduler, "finish_requests", None),
                   ("request_ids", "finished_status"), "Scheduler.finish_requests")
        fixture = clock_ns is not None
        actual_clock_domain = "fixture-clock" if fixture else _clock_domain()
        if clock_domain is not None and not fixture and clock_domain != actual_clock_domain:
            _fail("HARDWARE_IDENTITY_MISMATCH", "Live Scheduler clock_domain must match the actual host boot clock")
        version = _version(runtime_version) if runtime_version is not None else _runtime_version()
        if not fixture:
            if runtime_version is not None and version != _runtime_version():
                _fail("HARDWARE_IDENTITY_MISMATCH", "Installed and declared vLLM versions differ")
            validate_identity(profile_identity(local), measured=True)
        self.scheduler, self.exp = scheduler, local
        self._clock = time.perf_counter_ns if clock_ns is None else clock_ns
        self._last_ns = 0
        self._origin = self._now_ns()
        self._end_ns = self._origin
        self._max = _int(max_events, "max_events", 1)
        self._limit_reached = False
        self._record_ids = record_prompt_token_ids
        self._closed = False
        self._flushed: Path | None = None
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._saved: dict[str, Any] = {}
        self._native_order: dict[int, tuple[float, str]] = {}
        self.requests: dict[str, dict] = {}
        self.events: list[dict] = []
        self.errors: list[dict] = []
        self._lock = threading.RLock()
        self.header: dict[str, Any] = {"schema_version": 1, "kind": "vllm_scheduler_capture",
            "source": {"kind": "test_fixture" if fixture else "measured",
                       "reference": text(source_reference, "source_reference")},
            "session_id": session_id, "runtime_version": version, "api_revision": API_REVISION,
            "clock_domain": clock_domain or actual_clock_domain,
            "clock_origin_monotonic_ns": self._origin, "replica_index": replica_index,
            "identity": profile_identity(local),
            "experiment": asdict(replace(local, workload=(), execution=replace(local.execution, warmup=()))),
            "metadata": {"instrumentation_source": "vllm_internal_scheduler",
                "timing_boundary": "server_token_ready_v1",
                "observation_point": "scheduler_enqueue_to_engine_core_output_dispatch",
                "priority_semantics": "simulator_priority_equals_negated_vllm_priority",
                "prompt_token_ids_recorded": record_prompt_token_ids,
                "cache_namespace_encoding": "vllm_nonempty_salt_sha256_v1",
                "runtime_config_binding": "input_identity_or_same_session_runner_capture",
                "canonical_eligible": True}, "hardware_validated": False}
        config = getattr(scheduler, "vllm_config", None)
        self.header["metadata"]["actual_scheduler_config"] = _execution_config(config, local,
            bool(getattr(getattr(config, "scheduler_config", None), "async_scheduling", False)))
        existing = getattr(scheduler, "requests", {})
        fresh = not existing and (_constructor_origin or fixture)
        self.header["metadata"].update(install_origin="post_constructor" if _constructor_origin else "programmatic",
            fresh_state_verified=fresh, initial_cache_state="empty" if fresh else "unverified")
        if not fresh:
            self._error(SimulationError("UNVERIFIED_INITIAL_STATE", "Programmatic attach cannot assume empty pre-existing request/cache state"))
        self._check_policy(config, fixture)
        self._install()

    def _check_policy(self, config: Any, fixture: bool) -> None:
        if config is None and fixture:
            return
        scheduling = getattr(config, "scheduler_config", None)
        cache = getattr(config, "cache_config", None)
        strategy = self.exp.strategy
        mismatches = []
        expected = (("max_num_batched_tokens", strategy.max_batch_tokens),
                    ("max_num_seqs", strategy.max_batch_size),
                    ("enable_chunked_prefill", strategy.chunked_prefill))
        for name, value in expected:
            if getattr(scheduling, name, _MISSING) != value:
                mismatches.append(name)
        if getattr(cache, "enable_prefix_caching", _MISSING) != strategy.prefix_cache:
            mismatches.append("enable_prefix_caching")
        if getattr(self.scheduler, "block_size", getattr(cache, "block_size", _MISSING)) != strategy.memory.kv_block_tokens:
            mismatches.append("block_size")
        raw_dtype = str(getattr(cache, "cache_dtype", "unknown"))
        effective_dtype = self.exp.strategy.model.weight_dtype if raw_dtype == "auto" else raw_dtype
        if effective_dtype != strategy.model.kv_dtype:
            mismatches.append("cache_dtype")
        policy = str(getattr(scheduling, "policy", "unknown")).lower()
        if policy not in {"fcfs", "priority"}:
            mismatches.append("policy")
        if strategy.scheduler == "fifo":
            if strategy.max_batch_size != 1 or policy != "fcfs":
                mismatches.append("fifo_single_owner_mapping")
        elif not strategy.mixed_batching:
            mismatches.append("vllm_decode_first_joint_packing_mapping")
        parallel = getattr(config, "parallel_config", None)
        if (getattr(parallel, "tensor_parallel_size", _MISSING) != strategy.tensor_parallel_size
                or getattr(parallel, "pipeline_parallel_size", _MISSING) != strategy.pipeline_parallel_size
                or getattr(parallel, "data_parallel_size", 1) != 1
                or getattr(parallel, "use_ubatching", False)
                or getattr(parallel, "enable_expert_parallel", False)):
            mismatches.append("parallelism")
        self.header["metadata"].update(policy_mismatches=mismatches, actual_native_policy=policy)
        if mismatches:
            self._error(SimulationError("RUNTIME_POLICY_MISMATCH", "Observed Scheduler policy differs: " + ", ".join(mismatches)))

    def _now_ns(self) -> int:
        value = _int(self._clock(), "monotonic clock")
        if value < self._last_ns:
            _fail("INVALID_TIMING_SAMPLE", "Scheduler capture clock moved backwards")
        self._last_ns = value
        return value

    def _now(self) -> float:
        self._end_ns = self._now_ns()
        return (self._end_ns - self._origin) / 1e6

    def _error(self, exc: Exception) -> None:
        if len(self.errors) < 1000:
            self.errors.append({"error_code": getattr(exc, "code", "CAPTURE_FAILED"), "message": str(exc)})
        self.header["metadata"]["canonical_eligible"] = False

    def _event(self, kind: str, name: str, when: float, **kwargs: Any) -> None:
        if len(self.events) >= self._max:
            _fail("CAPTURE_LIMIT", "Scheduler event capture limit reached")
        self.events.append({"event": kind, "request_id": name, "timestamp_ms": when, **kwargs})

    def _capture_available(self) -> bool:
        if self._limit_reached:
            return False
        if len(self.events) >= self._max:
            self._limit_reached = True
            self._error(SimulationError("CAPTURE_LIMIT", "Scheduler event capture limit reached"))
            return False
        return True

    def _arrival(self, request: Any) -> None:
        if not self._capture_available():
            return
        when = self._now()
        name = text(request.request_id, "request_id")
        if name in self.requests or getattr(request, "resumable", False):
            _fail("UNSUPPORTED_VLLM_FEATURE", "Reused IDs and streaming-input sessions are not canonical replay")
        ids = request.prompt_token_ids
        if not isinstance(ids, list) or not ids or getattr(request, "mm_features", []) or getattr(request, "lora_request", None):
            _fail("UNSUPPORTED_VLLM_FEATURE", "Scheduler request must be ordinary text tokens without LoRA")
        if any(type(token) is not int or not 0 <= token < self.exp.strategy.model.vocab_size for token in ids):
            _fail("UNSUPPORTED_VLLM_ABI", "Scheduler prompt tokens are outside the declared vocabulary")
        params = request.sampling_params
        maximum = _int(params.max_tokens, "max_tokens", 1)
        priority = operator.index(getattr(request, "priority", 0))
        if isinstance(getattr(request, "priority", 0), bool) or not -(2**63) < priority < 2**63:
            _fail("UNSUPPORTED_VLLM_ABI", "Native priority cannot be represented by the replay contract")
        if priority and self.header["metadata"].get("actual_native_policy") == "fcfs":
            _fail("RUNTIME_POLICY_MISMATCH", "FCFS ignores nonzero native priorities but this replay would reorder them")
        native_arrival = getattr(request, "arrival_time", None)
        native_number = (float(native_arrival) if isinstance(native_arrival, (int, float))
                         and not isinstance(native_arrival, bool) else None)
        native_valid = native_number is not None and math.isfinite(native_number)
        namespace = cache_namespace_from_vllm_salt(getattr(request, "cache_salt", None))
        row = {"request_id": name, "input_tokens": len(ids), "arrival_ms": when,
               "priority": -priority, "native_priority": priority, "max_output_tokens": maximum,
               "native_arrival_time": native_number if native_valid else None,
               "cache_namespace": namespace,
               "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
               "output_tokens_actual": 0, "status": "WAITING"}
        if self._record_ids:
            row["prompt_token_ids"] = list(ids)
        self.requests[name] = row
        self._event("REQUEST_ARRIVAL", name, when)
        if self.header["metadata"].get("actual_native_policy") == "priority":
            if not native_valid:
                self._error(SimulationError("RUNTIME_POLICY_MISMATCH", "Native priority tie ordering needs observed native arrival_time"))
            else:
                assert native_number is not None
                key = native_number, name
                previous = self._native_order.get(priority)
                if previous is not None and key < previous:
                    self._error(SimulationError("RUNTIME_POLICY_MISMATCH", "Native priority arrival order differs from captured enqueue order"))
                self._native_order[priority] = max(key, previous) if previous is not None else key

    @staticmethod
    def _reason(reason: Any) -> str:
        value = getattr(reason, "name", None)
        return str(value if value is not None else reason).upper()

    def _published(self, outputs: Any) -> None:
        if not self._capture_available():
            return
        when = self._now()
        if not isinstance(outputs, dict):
            _fail("UNSUPPORTED_VLLM_ABI", "Scheduler did not return EngineCoreOutputs by client")
        for channel in outputs.values():
            for output in channel.outputs:
                name = output.request_id
                if name not in self.requests:
                    _fail("UNSUPPORTED_VLLM_ABI", "Published output has no captured enqueue")
                row = self.requests[name]
                if row["status"] in {"COMPLETED", "FAILED"}:
                    _fail("UNSUPPORTED_VLLM_ABI", "Published output appeared after termination")
                ids = _list(output.new_token_ids, "EngineCoreOutput.new_token_ids")
                if len(ids) > 1:
                    _fail("UNSUPPORTED_VLLM_FEATURE", "Multi-token published output has no separate server-token timestamps")
                for token in ids:
                    _int(token, "published token")
                    if token >= self.exp.strategy.model.vocab_size:
                        _fail("UNSUPPORTED_VLLM_ABI", "Published token is outside the vocabulary")
                    row["output_tokens_actual"] += 1
                    self._event("TOKEN_READY", name, when, token_count=1,
                                output_token_index=row["output_tokens_actual"])
                reason = output.finish_reason
                if reason is not None:
                    label = self._reason(reason)
                    if label in {"STOP", "LENGTH"}:
                        status, error = "COMPLETED", None
                    elif label in {"ABORT", "ERROR"}:
                        status, error = "FAILED", "RUNTIME_" + label
                    else:
                        _fail("UNSUPPORTED_VLLM_ABI", "Unknown EngineCore finish reason")
                    row.update(status=status, finish_reason=label)
                    details = {"status": status}
                    if error is not None:
                        details["error_code"] = error
                    self._event("REQUEST_FINISH", name, when, **details)

    def _install(self) -> None:
        scheduler = self.scheduler
        for name in ("add_request", "update_from_output", "finish_requests"):
            self._saved[name] = vars(scheduler).get(name, _MISSING)
        add, update, finish = scheduler.add_request, scheduler.update_from_output, scheduler.finish_requests
        @functools.wraps(add)
        def add_wrapper(request):
            result = add(request)
            with self._lock:
                try:
                    self._arrival(request)
                except Exception as exc:
                    self._error(exc)
            return result
        @functools.wraps(update)
        def update_wrapper(scheduler_output, model_runner_output):
            result = update(scheduler_output, model_runner_output)
            with self._lock:
                try:
                    if getattr(scheduler_output, "scheduled_spec_decode_tokens", {}):
                        _fail("UNSUPPORTED_VLLM_FEATURE", "Speculative scheduler output cannot become canonical tokens")
                    self._published(result)
                except Exception as exc:
                    self._error(exc)
            return result
        @functools.wraps(finish)
        def finish_wrapper(request_ids, finished_status):
            result = finish(request_ids, finished_status)
            with self._lock:
                try:
                    if not self._capture_available():
                        return result
                    when = self._now()
                    for request in result:
                        row = self.requests.get(request.request_id)
                        if row is not None and row["status"] not in {"COMPLETED", "FAILED"}:
                            label = self._reason(finished_status)
                            row.update(status="FAILED", finish_reason=label)
                            self._event("REQUEST_FINISH", request.request_id, when, status="FAILED",
                                        error_code="RUNTIME_" + label)
                except Exception as exc:
                    self._error(exc)
            return result
        try:
            scheduler.add_request, scheduler.update_from_output, scheduler.finish_requests = add_wrapper, update_wrapper, finish_wrapper
            scheduler._mlsys_scheduler_capture = self
        except Exception:
            self.close()
            raise

    def export_capture(self) -> dict:
        with self._lock:
            if not self._closed:
                try:
                    self._now()
                except Exception as exc:
                    self._error(exc)
            return {**deepcopy(self.header), "events": deepcopy(self.events),
                    "requests": deepcopy(list(self.requests.values())), "capture_errors": deepcopy(self.errors),
                    "capture_closed": self._closed, "observed_until_ms": (self._end_ns - self._origin) / 1e6}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._now()
            except Exception as exc:
                self._error(exc)
            self._closed = True
            for name, original in self._saved.items():
                if original is _MISSING:
                    vars(self.scheduler).pop(name, None)
                else:
                    setattr(self.scheduler, name, original)
            if getattr(self.scheduler, "_mlsys_scheduler_capture", None) is self:
                delattr(self.scheduler, "_mlsys_scheduler_capture")

    def flush(self) -> str | None:
        with self._lock:
            if self._flushed is not None:
                return str(self._flushed)
            self.close()
            if self._output_dir is None:
                return None
            self._output_dir.mkdir(parents=True, exist_ok=True)
            path = self._output_dir / (f"{self.header['session_id']}-scheduler-r{self.header['replica_index']}"
                                      f"-pid{os.getpid()}-{uuid.uuid4().hex}.json")
            with path.open("x", encoding="utf-8") as stream:
                json.dump(self.export_capture(), stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            self._flushed = path
            return str(path)


def install_scheduler_hook(scheduler: Any, experiment: Experiment, **kwargs: Any) -> SchedulerCapture:
    return SchedulerCapture(scheduler, experiment, **kwargs)


class VllmCaptureWorkerExtension:
    """New RPC methods only; does not override Worker load/init methods."""

    model_runner: Any

    def mlsys_capture_install(self, config: dict) -> dict:
        data = object_fields(config, {"experiment", "session_id"},
            {"schema_version", "output_dir", "source_reference", "replica_index", "rccl_metadata", "max_records"}, "worker capture config")
        if "schema_version" in data and (type(data["schema_version"]) is not int or data["schema_version"] != 1):
            _fail("INVALID_COLLECTION_PLAN", "Worker capture config schema_version must be 1")
        experiment = experiment_from_dict(data["experiment"])
        distributed = importlib.import_module("vllm.distributed")
        hook = install_runner_hook(self.model_runner, experiment,
            session_id=data["session_id"], replica_index=data.get("replica_index", 0),
            stage_index=distributed.get_pp_group().rank_in_group,
            tp_rank=distributed.get_tp_group().rank_in_group,
            output_dir=data.get("output_dir"), source_reference=data.get("source_reference", "explicit worker capture RPC"),
            rccl_metadata=data.get("rccl_metadata"), max_records=data.get("max_records", 100000))
        _ACTIVE_CAPTURES.append(hook)
        self._mlsys_worker_capture = hook
        return {"installed": True, "stage_index": hook.stage_index, "tp_rank": hook.tp_rank,
                "runtime_version": hook.header["runtime_version"]}

    def mlsys_capture_export(self) -> dict:
        hook = getattr(self.model_runner, "_mlsys_internal_capture", getattr(self, "_mlsys_worker_capture", None))
        if hook is None:
            _fail("CAPTURE_NOT_INSTALLED", "Install the worker capture before exporting")
        return hook.export_capture()

    def mlsys_capture_close(self) -> dict:
        hook = getattr(self.model_runner, "_mlsys_internal_capture", getattr(self, "_mlsys_worker_capture", None))
        if hook is None:
            _fail("CAPTURE_NOT_INSTALLED", "No worker capture is installed")
        return {"capture_file": hook.flush(), "capture": hook.export_capture()}


def _flush_active() -> None:
    for hook in list(_ACTIVE_CAPTURES):
        try:
            hook.flush()
        except Exception:
            # Never turn interpreter teardown into a forged successful sample.
            # Explicit RPC close/export is the observable, error-reporting path.
            pass


def register_vllm_hooks() -> None:
    """Dormant general-plugin entrypoint; explicit environment activates capture."""
    global _PLUGIN_CONFIG
    filename = os.environ.get("MLSYS_VLLM_CAPTURE_CONFIG")
    if not filename:
        return
    if _PLUGIN_CONFIG is not None:
        return
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "").lower() in {"1", "true"}:
        _fail("UNSUPPORTED_VLLM_ABI", "Internal capture requires MRv1; set VLLM_USE_V2_MODEL_RUNNER=0")
    config = object_fields(read_json(filename), {"schema_version", "experiment", "session_id", "output_dir"},
        {"source_reference", "replica_index", "rccl_metadata", "max_records", "capture_scheduler",
         "record_prompt_token_ids", "max_events"}, "vLLM plugin config")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        _fail("INVALID_COLLECTION_PLAN", "Plugin config schema_version must be 1")
    root = Path(filename).resolve().parent
    experiment = (load_experiment(root / config["experiment"]) if isinstance(config["experiment"], str)
                  else experiment_from_dict(config["experiment"]))
    version = _runtime_version()
    module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    runner_type = module.GPUModelRunner
    _signature(runner_type.__init__, ("self", "vllm_config", "device"), "GPUModelRunner.__init__")
    capture_scheduler = config.get("capture_scheduler", True)
    if type(capture_scheduler) is not bool:
        _fail("INVALID_COLLECTION_PLAN", "capture_scheduler must be boolean")
    scheduler_type = None
    scheduler_original = None
    if capture_scheduler:
        scheduler_module = importlib.import_module("vllm.v1.core.sched.scheduler")
        scheduler_type = scheduler_module.Scheduler
        scheduler_original = scheduler_type.__init__
        _signature(scheduler_original, ("self", "vllm_config", "kv_cache_config", "structured_output_manager",
            "block_size", "hash_block_size", "mm_registry", "include_finished_set", "log_stats"), "Scheduler.__init__")
    if _PLUGIN_PATCHES:
        _fail("CAPTURE_ALREADY_INSTALLED", "Plugin constructor already patched")
    original = runner_type.__init__
    output = root / text(config["output_dir"], "output_dir")
    @functools.wraps(original)
    def initialize(runner, vllm_config, device):
        if getattr(vllm_config, "use_v2_model_runner", False):
            _fail("UNSUPPORTED_VLLM_ABI", "The configured runtime selects unsupported MRv2")
        original(runner, vllm_config, device)
        distributed = importlib.import_module("vllm.distributed")
        hook = install_runner_hook(runner, experiment, session_id=config["session_id"],
            replica_index=config.get("replica_index", 0), stage_index=distributed.get_pp_group().rank_in_group,
            tp_rank=distributed.get_tp_group().rank_in_group, runtime_version=version,
            output_dir=output, source_reference=config.get("source_reference", "explicit general-plugin capture"),
            rccl_metadata=config.get("rccl_metadata"), max_records=config.get("max_records", 100000))
        _ACTIVE_CAPTURES.append(hook)
    runner_type.__init__ = initialize
    _PLUGIN_PATCHES.append((runner_type, original))
    if scheduler_type is not None and scheduler_original is not None:
        @functools.wraps(scheduler_original)
        def initialize_scheduler(scheduler, *args, **kwargs):
            assert scheduler_original is not None
            actual_config = args[0] if args else kwargs.get("vllm_config")
            if getattr(actual_config, "use_v2_model_runner", False):
                _fail("UNSUPPORTED_VLLM_ABI", "The configured runtime selects unsupported MRv2; use MRv1 capture")
            scheduler_original(scheduler, *args, **kwargs)
            hook = install_scheduler_hook(scheduler, experiment, session_id=config["session_id"],
                replica_index=config.get("replica_index", 0), runtime_version=version,
                source_reference=config.get("source_reference", "explicit general-plugin Scheduler capture"),
                output_dir=output, record_prompt_token_ids=config.get("record_prompt_token_ids", False),
                max_events=config.get("max_events", 1000000), _constructor_origin=True)
            _ACTIVE_CAPTURES.append(hook)
        scheduler_type.__init__ = initialize_scheduler
        _PLUGIN_PATCHES.append((scheduler_type, scheduler_original))
    _PLUGIN_CONFIG = deepcopy(config)
    atexit.register(_flush_active)


def uninstall_vllm_hooks_plugin() -> None:
    """Explicitly close captures and restore constructor patches in this process."""
    global _PLUGIN_CONFIG
    for hook in list(_ACTIVE_CAPTURES):
        hook.close()
    _ACTIVE_CAPTURES.clear()
    for target, original in reversed(_PLUGIN_PATCHES):
        target.__init__ = original
    _PLUGIN_PATCHES.clear()
    _PLUGIN_CONFIG = None
