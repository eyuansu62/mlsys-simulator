"""Versioned, explicit identities for the Agent service's first supported domain.

This catalog copies already-shipped course geometry and nominal hardware
declarations. It neither discovers physical devices nor invents aliases from a
parameter count. Changing a declaration or support policy requires a new
catalog version.
"""

from dataclasses import asdict

from .qwen3 import model_from_qwen3_config
from .schema import DeviceSpec, ModelSpec, SimulationError


CATALOG_VERSION = "mlsys-agent-catalog-v1"
MODEL_REVISION = "course-7df2dae493eb-geometry"
MAX_CONTEXT_TOKENS = 40960
MAX_REPLICAS_PER_DEPLOYMENT = 16
_COURSE_COMMIT = "7df2dae493eb30c0f7e845b6d587b905288c0d7a"
_MODEL_SOURCE = ("https://github.com/amdjiahangpan/hello-mlsys/blob/"
                 + _COURSE_COMMIT + "/docs/part0-intro/chapter0/index.md")
_HARDWARE_SOURCE = ("https://github.com/amdjiahangpan/hello-mlsys/blob/"
                    + _COURSE_COMMIT + "/docs/part1-modeling/chapter1/index.md")
_GEOMETRIES = (
    ("qwen3-0.6b", "Qwen3-0.6B", 28, 1024, 3072, 16, 8),
    ("qwen3-1.7b", "Qwen3-1.7B", 28, 2048, 6144, 16, 8),
    ("qwen3-4b", "Qwen3-4B", 36, 2560, 9728, 32, 8),
)


def validate_agent_catalog_version(version: str) -> None:
    """Require the published catalog version rather than a negotiation label."""
    if not isinstance(version, str) or version != CATALOG_VERSION:
        raise SimulationError("CATALOG_VERSION_MISMATCH",
                              f"catalog_version must be {CATALOG_VERSION!r}")


def _model(model_id: str) -> ModelSpec:
    declaration = next((row for row in _GEOMETRIES if row[0] == model_id), None)
    if not isinstance(model_id, str) or declaration is None:
        raise SimulationError("UNKNOWN_MODEL_ID", "deployments[].model must name a published catalog model")
    _, name, layers, hidden, intermediate, heads, kv_heads = declaration
    config = {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "num_hidden_layers": layers, "hidden_size": hidden, "intermediate_size": intermediate,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": 128,
        "vocab_size": 151936, "tie_word_embeddings": True, "torch_dtype": "bfloat16",
        "hidden_act": "silu", "attention_bias": False, "attention_dropout": 0.0,
        "use_cache": True, "use_sliding_window": False, "sliding_window": None,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000,
        "max_position_embeddings": MAX_CONTEXT_TOKENS, "rope_scaling": None,
    }
    return model_from_qwen3_config(config, name=name, revision=MODEL_REVISION)


def resolve_agent_model(model_id: str, weight_precision: str, kv_precision: str,
                        context_tokens: int) -> ModelSpec:
    """Resolve exact BF16 geometry; unknown precision/context never falls back."""
    model = _model(model_id)
    if weight_precision != "BF16" or kv_precision != "BF16":
        raise SimulationError("UNSUPPORTED_PRECISION",
                              "deployments[] requires weight_precision=BF16 and kv_precision=BF16 "
                              "in this catalog")
    if type(context_tokens) is not int or not 1 <= context_tokens <= MAX_CONTEXT_TOKENS:
        raise SimulationError("UNSUPPORTED_CONTEXT_LENGTH",
                              f"deployments[].context_tokens must be an integer in [1, {MAX_CONTEXT_TOKENS}]")
    return model


def resolve_agent_device(hardware_id: str, device_id: str) -> DeviceSpec:
    """Resolve a declared device without changing its true catalog capacity."""
    if hardware_id != "rx9070xt":
        raise SimulationError("UNKNOWN_HARDWARE_ID", "infra device hardware must name a published catalog device")
    if (not isinstance(device_id, str) or not device_id.strip() or device_id != device_id.strip()
            or len(device_id) > 256 or any(ord(char) < 32 for char in device_id)):
        raise SimulationError("INVALID_DEVICE_ID", "infra device ID must be bounded nonempty text")
    return DeviceSpec(id=device_id, name="RX9070XT · course nominal",
                      memory_bytes=16 * 1024**3, compute_tflops=195.0,
                      memory_bandwidth_gbps=640.0)


def get_agent_catalog() -> dict:
    """Return fresh JSON data; nominal declarations are not measured profiles."""
    models = []
    for model_id, name, *_ in _GEOMETRIES:
        model = _model(model_id)
        models.append({
            "id": model_id, "name": name, "revision": MODEL_REVISION,
            "identity_scope": "course-declared geometry, not a downloaded checkpoint revision",
            "geometry": asdict(model), "max_context_tokens": MAX_CONTEXT_TOKENS,
            "supported_weight_precisions": ["BF16"], "supported_kv_precisions": ["BF16"],
            "tp_degrees": [1],
            "source": {"kind": "course_geometry", "reference": _MODEL_SOURCE,
                       "shipped_declaration": "examples/course/chapter1.plan.json"},
        })
    device = resolve_agent_device("rx9070xt", "catalog-device")
    return {
        "schema": "mlsys.agent.catalog.v1", "catalog_version": CATALOG_VERSION,
        "evidence_level": "simulated", "hardware_validated": False,
        "calibration": {"kind": "uncalibrated", "version": "agent-analytical-v1", "in_domain": None,
                        "assumptions": ["Model geometry and device rates are declared inputs, not GPU measurements.",
                                        "No measured Profile or inter-device communication topology is selected."]},
        "models": models,
        "hardware": [{"id": "rx9070xt", "name": "RX9070XT · course nominal",
                      "capacity_bytes": device.memory_bytes, "compute_tflops": device.compute_tflops,
                      "memory_bandwidth_gbps": device.memory_bandwidth_gbps,
                      "evidence": "course_nominal",
                      "source": {"kind": "course_nominal", "reference": _HARDWARE_SOURCE,
                                 "capacity_declaration": "Existing single-GPU Web preset: 16 GiB"}}],
        "communication": None,
        "unsupported_fields": {
            "communication": "No confirmed link topology or communication Profile is supplied; TP degree >1 "
                             "and PP are not exposed by this catalog.",
        },
        "support": {"tp_degrees": [1], "pp_degrees": [1],
                    "max_replicas_per_deployment": MAX_REPLICAS_PER_DEPLOYMENT,
                    "replica_semantics": "Independent single-GPU replicas on explicitly declared devices",
                    "workload_modes": ["open_loop", "sequential", "dag"]},
    }
