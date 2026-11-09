"""Strict, dependency-free import of dense Hugging Face Qwen3 geometry.

Configuration import describes a model. It does not load weights, execute
remote code, discover a Runtime version or establish hardware accuracy.
"""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, NoReturn

from .schema import ModelSpec, SimulationError


_MAX_BYTES = 1024 * 1024
_WIDTHS = {"float16": 2.0, "bfloat16": 2.0, "float32": 4.0, "float64": 8.0}
_GEOMETRY = {
    "num_hidden_layers": "num_layers", "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size", "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_kv_heads", "head_dim": "head_dim", "vocab_size": "vocab_size",
}
_MAPPED = set(_GEOMETRY) | {"tie_word_embeddings", "dtype", "torch_dtype"}
_BEHAVIOR = {"model_type", "architectures", "hidden_act", "attention_bias", "attention_dropout",
             "quantization_config", "use_sliding_window", "sliding_window", "layer_types",
             "rope_scaling", "rope_parameters", "use_cache", "qk_norm", "final_norm"}
_METADATA = {
    "initializer_range", "bos_token_id", "eos_token_id", "pad_token_id", "max_position_embeddings",
    "max_window_layers", "rms_norm_eps", "rope_theta", "transformers_version", "_name_or_path",
    "_commit_hash", "return_dict", "output_hidden_states", "output_attentions", "torchscript",
    "pruned_heads", "chunk_size_feed_forward", "is_encoder_decoder", "is_decoder",
    "add_cross_attention", "tie_encoder_decoder", "task_specific_params", "problem_type",
    "id2label", "label2id", "finetuning_task", "tokenizer_class", "prefix", "_attn_implementation",
    "_attn_implementation_autoset", "architectures_version",
    "_attn_implementation_internal", "_auto_class", "tf_legacy_loss", "use_bfloat16",
    "sep_token_id", "decoder_start_token_id", "max_length", "min_length", "do_sample",
    "early_stopping", "num_beams", "num_beam_groups", "diversity_penalty", "temperature",
    "top_k", "top_p", "typical_p", "repetition_penalty", "length_penalty", "no_repeat_ngram_size",
    "bad_words_ids", "num_return_sequences", "output_scores", "return_dict_in_generate",
    "forced_bos_token_id", "forced_eos_token_id", "remove_invalid_values",
    "exponential_decay_length_penalty", "suppress_tokens", "begin_suppress_tokens",
    "encoder_no_repeat_ngram_size", "renormalize_logits", "sequence_bias",
}
_MOE = {"num_experts", "num_experts_per_tok", "moe_intermediate_size", "decoder_sparse_step",
        "router_aux_loss_coef", "output_router_logits", "num_local_experts"}
_KNOWN = _MAPPED | _BEHAVIOR | _METADATA | _MOE | {"auto_map"}
_ARCH_METADATA = {"rms_norm_eps", "rope_theta", "max_position_embeddings"}


def _fail(message: str, *, unsupported: bool = False) -> NoReturn:
    raise SimulationError("UNSUPPORTED_MECHANISM" if unsupported else "INVALID_QWEN3_CONFIG", message)


def _text(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > 1024
            or any(ord(char) < 32 for char in value)):
        _fail(f"{label} must be a bounded nonempty string without control characters")
    if value.strip().lower() in {"unspecified", "unknown", "auto", "default", "none", "n/a"}:
        _fail(f"{label} must be explicitly supplied, not a placeholder")
    return value


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"{label} must be an integer in [{minimum}, 2**63-1]")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be finite")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(f"{label} must be finite")
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        _fail(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _canonical(config: Any) -> bytes:
    try:
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False,
                             ensure_ascii=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise SimulationError("INVALID_QWEN3_CONFIG", "Config must be finite JSON data") from exc
    if len(encoded) > _MAX_BYTES:
        _fail("Qwen3 configuration exceeds the 1MiB limit")
    return encoded


def _validated(config: dict) -> tuple[dict, str]:
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        _fail("Qwen3 config must be an object with string keys")
    _canonical(config)
    unknown = set(config) - _KNOWN
    if unknown:
        _fail(f"Unknown Qwen3 configuration fields: {', '.join(sorted(unknown))}")
    if config.get("model_type") != "qwen3" or config.get("architectures") != ["Qwen3ForCausalLM"]:
        _fail("Only explicitly declared dense Qwen3ForCausalLM configurations are supported", unsupported=True)
    if any(key in config for key in _MOE):
        _fail("MoE/router configuration is outside the dense Qwen3 model", unsupported=True)
    if config.get("auto_map") not in (None, {}):
        _fail("Custom auto_map code is not a supported Qwen3 implementation", unsupported=True)
    missing = set(_GEOMETRY) | {"tie_word_embeddings", "hidden_act", "attention_bias"}
    missing -= config.keys()
    if missing:
        _fail(f"Missing explicit Qwen3 fields: {', '.join(sorted(missing))}")
    for name in _GEOMETRY:
        _integer(config[name], name)
    if config["head_dim"] % 2:
        _fail("Qwen3 full rotary attention requires an even head_dim", unsupported=True)
    if type(config["tie_word_embeddings"]) is not bool:
        _fail("tie_word_embeddings must be an explicit boolean")
    if config["hidden_act"] != "silu":
        _fail("Only the dense gated SiLU Qwen3 MLP is modeled", unsupported=True)
    if config["attention_bias"] is not False:
        _fail("Qwen3 attention bias is not modeled", unsupported=True)
    if config.get("attention_dropout", 0) != 0 or type(config.get("attention_dropout", 0)) not in (int, float):
        _fail("Nonzero or invalid attention dropout is not an inference contract", unsupported=True)
    if config.get("quantization_config") is not None:
        _fail("Quantized Qwen3 storage and metadata are not imported; declare the effective "
              "weight_bytes/kv_bytes/activation_bytes (including block/scale overhead) and "
              "weight_dequant explicitly in the experiment strategy; see docs/quantization.md",
              unsupported=True)
    window = config.get("use_sliding_window", False)
    if type(window) is not bool:
        _fail("use_sliding_window must be a boolean")
    if window:
        _fail("Sliding-window attention is not modeled", unsupported=True)
    if config.get("sliding_window") is not None:
        # A disabled, configured window is inactive, as in HF configurations.
        _integer(config["sliding_window"], "sliding_window")
        if "use_sliding_window" not in config:
            _fail("A configured sliding_window needs explicit use_sliding_window=false", unsupported=True)
    if "layer_types" in config:
        kinds = config["layer_types"]
        if (not isinstance(kinds, list) or len(kinds) != config["num_hidden_layers"]
                or any(kind != "full_attention" for kind in kinds)):
            _fail("Only an explicit full-attention layer list is supported", unsupported=True)
    if config.get("rope_scaling") is not None:
        _fail("Scaled RoPE configuration is not represented by this Qwen3 importer", unsupported=True)
    if "rope_parameters" in config and config["rope_parameters"] is not None:
        parameters = config["rope_parameters"]
        if (not isinstance(parameters, dict) or set(parameters) - {"rope_type", "rope_theta"}
                or parameters.get("rope_type", "default") != "default"):
            _fail("Only default RoPE parameters are supported", unsupported=True)
        if "rope_theta" in parameters:
            _finite(parameters["rope_theta"], "rope_parameters.rope_theta", positive=True)
            if "rope_theta" in config and parameters["rope_theta"] != config["rope_theta"]:
                _fail("Top-level and nested RoPE theta disagree")
    for name in ("qk_norm", "final_norm", "use_cache"):
        if name in config and config[name] is not True:
            _fail(f"Qwen3 {name} must be enabled", unsupported=True)
    for name in ("is_encoder_decoder", "add_cross_attention", "tie_encoder_decoder"):
        if name in config and config[name] is not False:
            _fail("Encoder or cross-attention configurations are unsupported", unsupported=True)
    for name in ("rms_norm_eps", "rope_theta"):
        if name in config:
            _finite(config[name], name, positive=True)
    if "initializer_range" in config:
        _finite(config["initializer_range"], "initializer_range")
    for name in ("max_position_embeddings", "max_window_layers"):
        if name in config:
            _integer(config[name], name)
    storage = [config[key] for key in ("dtype", "torch_dtype") if config.get(key) is not None]
    if not storage or any(not isinstance(value, str) or value not in _WIDTHS for value in storage):
        _fail("dtype or torch_dtype must explicitly declare float16, bfloat16, float32 or float64", unsupported=True)
    if len(set(storage)) != 1:
        _fail("dtype and torch_dtype disagree")
    return config, storage[0]


def model_from_qwen3_config(config: dict, *, name: str, revision: str,
                           kv_dtype: str | None = None) -> ModelSpec:
    """Map exact declared geometry without changing residual or attention width.

    KV defaults to the explicit model storage dtype. A Runtime using a different
    KV cache dtype must pass it explicitly and validate its own actual setup.
    """
    config, dtype = _validated(config)
    model_name, model_revision = _text(name, "name"), _text(revision, "revision")
    kv = dtype if kv_dtype is None else kv_dtype
    if not isinstance(kv, str) or kv not in _WIDTHS:
        _fail("kv_dtype must be an explicit supported unquantized float dtype", unsupported=True)
    values = {destination: config[source] for source, destination in _GEOMETRY.items()}
    model = ModelSpec(**values, name=model_name, revision=model_revision,
        tie_word_embeddings=config["tie_word_embeddings"], qk_norm=True, final_norm=True,
        architecture_config_sha256=hashlib.sha256(_canonical(_architecture(config, dtype))).hexdigest(),
        weight_dtype=dtype, weight_bytes=_WIDTHS[dtype], activation_dtype=dtype,
        activation_bytes=_WIDTHS[dtype], kv_dtype=kv, kv_bytes=_WIDTHS[kv])
    # Reuse the core model constraints without allocating weights or KV storage.
    from .config import validate_experiment
    from .schema import Experiment, StrategySpec
    validate_experiment(Experiment(strategy=StrategySpec(model=model)))
    return model


def _architecture(config: dict, dtype: str) -> dict:
    """Canonical supported architecture, using pinned Qwen3Config defaults.

    Defaults follow the Qwen3 HF configuration introduced in Transformers 4.51.
    Runtime config objects expose the resolved values, so metadata/version/path
    additions do not change this fingerprint. Inactive sliding windows cannot
    alter the full-attention execution contract.
    """
    nested = config.get("rope_parameters") or {}
    theta = config.get("rope_theta", nested.get("rope_theta", 10000.0))
    return {
        "model_type": "qwen3", "architecture": "Qwen3ForCausalLM",
        **{key: config[key] for key in _GEOMETRY},
        "tie_word_embeddings": config["tie_word_embeddings"],
        "qk_norm": True, "final_norm": True, "dtype": dtype,
        "hidden_act": "silu", "attention_bias": False, "attention_dropout": 0.0,
        "rms_norm_eps": _finite(config.get("rms_norm_eps", 1e-6), "rms_norm_eps", positive=True),
        "rope_type": "default", "rope_theta": _finite(theta, "rope_theta", positive=True),
        "rope_scaling": None,
        "max_position_embeddings": _integer(config.get("max_position_embeddings", 32768),
                                              "max_position_embeddings"),
        "use_cache": True, "layer_attention": "full_attention", "use_sliding_window": False,
        "sliding_window": None,
    }


def normalized_qwen3_architecture(config: dict) -> dict:
    """Return the validated, metadata-independent Qwen3 architecture mapping."""
    value, dtype = _validated(config)
    return _architecture(value, dtype)


def qwen3_architecture_fingerprint(config: dict) -> str:
    """Fingerprint resolved model behavior while excluding paths and versions."""
    return hashlib.sha256(_canonical(normalized_qwen3_architecture(config))).hexdigest()


def load_qwen3_model(path: str | Path, *, name: str, revision: str,
                     kv_dtype: str | None = None) -> dict:
    """Return a JSON-ready model import and immutable raw/canonical provenance."""
    source = Path(path)
    try:
        with source.open("rb") as stream:
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            _fail("Qwen3 configuration exceeds the 1MiB limit")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    _fail("Qwen3 configuration contains duplicate JSON keys")
                result[key] = value
            return result
        def constant(_):
            _fail("Qwen3 configuration contains a non-finite value")
        config = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except SimulationError:
        raise
    except (OSError, ValueError, UnicodeError) as exc:
        raise SimulationError("INVALID_QWEN3_CONFIG", "Cannot read a valid UTF-8 Qwen3 JSON configuration") from exc
    model = model_from_qwen3_config(config, name=name, revision=revision, kv_dtype=kv_dtype)
    return {"schema_version": 1, "kind": "qwen3_model_import", "model": asdict(model),
        "provenance": {"source_path": str(source.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
            "architecture_config_sha256": model.architecture_config_sha256,
            "normalized_architecture": normalized_qwen3_architecture(config),
            "mapped_fields": sorted(_MAPPED & config.keys()),
            "validated_behavior_fields": sorted((_BEHAVIOR | _ARCH_METADATA) & config.keys()),
            "ignored_metadata_fields": sorted((_METADATA - _ARCH_METADATA) & config.keys()),
            "kv_dtype_binding": "explicit_override" if kv_dtype is not None else "declared_model_storage_default",
            "scope": "dense_Qwen3_geometry_and_storage; no_weight_loading_or_runtime_version_inference"},
        "hardware_validated": False}
