"""Strict, dependency-free JSON/TOML experiment loading and validation."""

from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
import types
import typing

from .schema import Experiment, SimulationError


def _invalid(path, message):
    raise SimulationError("INVALID_CONFIG", f"{path}: {message}")


def _value(value, annotation, path, *, parse):
    origin = typing.get_origin(annotation)
    if origin is types.UnionType:
        for option in typing.get_args(annotation):
            try:
                return _value(value, option, path, parse=parse)
            except SimulationError:
                pass
        _invalid(path, "value has an invalid type")
    if annotation is type(None):
        if value is not None:
            _invalid(path, "expected null")
        return None
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            _invalid(path, "expected an array")
        item_type = typing.get_args(annotation)[0]
        return tuple(_value(v, item_type, f"{path}[{i}]", parse=parse)
                     for i, v in enumerate(value))
    if isinstance(annotation, type) and is_dataclass(annotation):
        if isinstance(value, annotation):
            mapping = {f.name: getattr(value, f.name) for f in fields(annotation)}
        elif parse and isinstance(value, dict):
            mapping = value
        else:
            _invalid(path, f"expected {annotation.__name__}")
        valid_fields = {f.name for f in fields(annotation)}
        unknown = set(mapping) - valid_fields
        if unknown:
            _invalid(path, f"unknown fields: {', '.join(sorted(map(str, unknown)))}")
        hints = typing.get_type_hints(annotation)
        parsed = {k: _value(v, hints[k], f"{path}.{k}", parse=parse)
                  for k, v in mapping.items()}
        try:
            return annotation(**parsed)
        except TypeError as exc:
            _invalid(path, str(exc))
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _invalid(path, "expected a finite number")
        try:
            number = float(value)
        except (ValueError, OverflowError):
            _invalid(path, "expected a finite number")
        if not math.isfinite(number):
            _invalid(path, "expected a finite number")
        return number
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            _invalid(path, "expected an integer")
        if not -(2 ** 63) <= value <= 2 ** 63 - 1:
            _invalid(path, "integer must fit the signed 64-bit range")
    elif not isinstance(value, annotation):
        _invalid(path, f"expected {annotation.__name__}")
    return value


def _positive(value, path, *, zero=False):
    if value < 0 if zero else value <= 0:
        _invalid(path, "must be nonnegative" if zero else "must be positive")


def validate_experiment(experiment: Experiment) -> None:
    """Validate both parsed files and direct Python dataclass construction."""
    exp = _value(experiment, Experiment, "experiment", parse=False)
    if exp.schema_version != 1:
        _invalid("schema_version", "only version 1 is supported")
    strategy, infra = exp.strategy, exp.infra
    model, memory, calibration = strategy.model, strategy.memory, exp.calibration
    for name in ("num_layers", "hidden_size", "intermediate_size",
                 "num_attention_heads", "num_kv_heads", "head_dim", "vocab_size"):
        _positive(getattr(model, name), f"strategy.model.{name}")
    for name in ("weight_bytes", "kv_bytes", "activation_bytes"):
        if getattr(model, name) not in (2.0, 4.0, 8.0):
            raise SimulationError(
                "UNSUPPORTED_MECHANISM",
                f"strategy.model.{name}: only non-quantized 2/4/8-byte storage is "
                "supported; 8-bit/packed formats and their metadata are not modeled",
            )
    if not model.name:
        _invalid("strategy.model.name", "must not be empty")
    if model.hidden_size != model.num_attention_heads * model.head_dim:
        _invalid("strategy.model", "hidden_size must equal query heads × head_dim")
    if model.num_attention_heads % model.num_kv_heads:
        _invalid("strategy.model", "query heads must be divisible by KV heads")
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "replicas",
                 "max_batch_size", "max_batch_tokens", "max_preemptions_per_request"):
        _positive(getattr(strategy, name), f"strategy.{name}")
    if strategy.replicas != 1:
        raise SimulationError("UNSUPPORTED_MECHANISM", "multiple replicas are not implemented")
    if strategy.scheduler not in ("fifo", "continuous_batching"):
        raise SimulationError("UNSUPPORTED_MECHANISM", "unsupported scheduler")
    if strategy.preemption not in ("none", "recompute"):
        raise SimulationError("UNSUPPORTED_MECHANISM", "only none/recompute preemption is supported")
    tp, pp = strategy.tensor_parallel_size, strategy.pipeline_parallel_size
    if tp not in (1, 2, 4):
        raise SimulationError("UNSUPPORTED_MECHANISM", "supported TP degrees are 1, 2, and 4")
    for name in ("num_attention_heads", "hidden_size", "intermediate_size", "vocab_size"):
        if getattr(model, name) % tp:
            _invalid(f"strategy.model.{name}", "must be divisible by TP")
    if (model.num_kv_heads % tp if tp < model.num_kv_heads else tp % model.num_kv_heads):
        _invalid("strategy.model.num_kv_heads", "KV head sharding/replication is not integral")
    if pp > model.num_layers:
        _invalid("strategy.pipeline_parallel_size", "cannot exceed number of layers")
    if strategy.stage_layers:
        if (len(strategy.stage_layers) != pp or min(strategy.stage_layers) <= 0
                or sum(strategy.stage_layers) != model.num_layers):
            _invalid("strategy.stage_layers", "positive counts must cover all layers exactly")
    if len(infra.devices) != tp * pp:
        _invalid("infra.devices", "must contain exactly TP × PP devices in stage-major order")
    ids = [device.id for device in infra.devices]
    if len(set(ids)) != len(ids) or not all(ids):
        _invalid("infra.devices", "device IDs must be unique and nonempty")
    for i, device in enumerate(infra.devices):
        for name in ("memory_bytes", "compute_tflops", "memory_bandwidth_gbps"):
            _positive(getattr(device, name), f"infra.devices[{i}].{name}")
    # Compute/bandwidth differences are conservatively modeled by slowest rank.
    _positive(infra.link_bandwidth_gbps, "infra.link_bandwidth_gbps")
    _positive(infra.link_latency_us, "infra.link_latency_us", zero=True)
    for f in fields(memory):
        _positive(getattr(memory, f.name), f"strategy.memory.{f.name}",
                  zero=f.name not in ("kv_block_tokens", "kv_pool_bytes"))
    if calibration.backend not in ("analytical", "synthetic"):
        raise SimulationError("UNSUPPORTED_MECHANISM", "measured/lookup backends are not implemented")
    if not calibration.profile_version:
        _invalid("calibration.profile_version", "must not be empty")
    for name in ("compute_efficiency", "bandwidth_efficiency"):
        value = getattr(calibration, name)
        if not 0 < value <= 1:
            _invalid(f"calibration.{name}", "must be in (0, 1]")
    for name in ("launch_overhead_us", "framework_overhead_us"):
        _positive(getattr(calibration, name), f"calibration.{name}", zero=True)
    for name in ("prefill_ms", "decode_ms", "recompute_ms"):
        _positive(getattr(calibration, name), f"calibration.{name}")
    request_ids = set()
    for i, request in enumerate(exp.workload):
        path = f"workload[{i}]"
        if not request.request_id or request.request_id in request_ids:
            _invalid(path, "request IDs must be nonempty and unique")
        request_ids.add(request.request_id)
        _positive(request.arrival_ms, f"{path}.arrival_ms", zero=True)
        _positive(request.input_tokens, f"{path}.input_tokens")
        _positive(request.output_tokens, f"{path}.output_tokens")
        if request.timeout_ms is not None:
            _positive(request.timeout_ms, f"{path}.timeout_ms", zero=True)
    for f in fields(exp.evaluation):
        value = getattr(exp.evaluation, f.name)
        if value is not None:
            _positive(value, f"evaluation.{f.name}", zero=f.name != "measurement_window_ms")
    _positive(exp.execution.max_simulated_ms, "execution.max_simulated_ms")
    _positive(exp.execution.max_events, "execution.max_events")


def experiment_from_dict(mapping: dict[str, typing.Any]) -> Experiment:
    experiment = _value(mapping, Experiment, "experiment", parse=True)
    validate_experiment(experiment)
    return experiment


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _invalid("config", f"duplicate key {key!r}")
        result[key] = value
    return result


def _read(path: Path):
    try:
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".toml":
            import tomllib
            return tomllib.loads(content)
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line, object_pairs_hook=_no_duplicate_keys)
                    for line in content.splitlines() if line.strip()]
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise SimulationError("INVALID_CONFIG", "YAML requires installed PyYAML; use JSON/TOML otherwise") from exc
            class StrictLoader(yaml.SafeLoader):
                pass
            def construct_mapping(loader, node, deep=False):
                return _no_duplicate_keys(
                    (loader.construct_object(k, deep=deep), loader.construct_object(v, deep=deep))
                    for k, v in node.value)
            StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                                         construct_mapping)
            try:
                return yaml.load(content, Loader=StrictLoader)
            except yaml.YAMLError as exc:
                raise SimulationError("INVALID_CONFIG", f"{path}: {exc}") from exc
        return json.loads(content, object_pairs_hook=_no_duplicate_keys)
    except SimulationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SimulationError("INVALID_CONFIG", f"{path}: {exc}") from exc


def load_experiment(path: str | Path) -> Experiment:
    """Load a config; section references resolve relative to its own directory."""
    source = Path(path)
    mapping = _read(source)
    if not isinstance(mapping, dict):
        _invalid(str(source), "experiment must be an object")
    mapping = dict(mapping)
    strategy_root = source.parent
    for section in ("workload", "infra", "strategy", "calibration", "evaluation", "execution"):
        if isinstance(mapping.get(section), str):
            referenced_path = source.parent / mapping[section]
            mapping[section] = _read(referenced_path)
            if section == "strategy":
                strategy_root = referenced_path.parent
    if isinstance(mapping.get("strategy"), dict):
        strategy = dict(mapping["strategy"])
        if isinstance(strategy.get("model"), str):
            strategy["model"] = _read(strategy_root / strategy["model"])
        mapping["strategy"] = strategy
    return experiment_from_dict(mapping)
