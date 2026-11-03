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
    dtype_bytes = {"float16": 2.0, "bfloat16": 2.0, "float32": 4.0, "float64": 8.0}
    for kind in ("weight", "kv", "activation"):
        dtype = getattr(model, f"{kind}_dtype")
        if dtype != "auto" and (dtype not in dtype_bytes or dtype_bytes[dtype] != getattr(model, f"{kind}_bytes")):
            _invalid(f"strategy.model.{kind}_dtype", "must be auto or an explicit float dtype matching its byte width")
    if not model.name.strip() or not model.revision.strip():
        _invalid("strategy.model", "name and revision must not be empty")
    if model.hidden_size != model.num_attention_heads * model.head_dim:
        _invalid("strategy.model", "hidden_size must equal query heads × head_dim")
    if model.num_attention_heads % model.num_kv_heads:
        _invalid("strategy.model", "query heads must be divisible by KV heads")
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "replicas",
                 "max_batch_size", "max_batch_tokens", "max_preemptions_per_request",
                 "pipeline_microbatch_size", "pipeline_max_inflight_microbatches"):
        _positive(getattr(strategy, name), f"strategy.{name}")
    if strategy.pipeline_microbatch_size > strategy.max_batch_size:
        _invalid("strategy.pipeline_microbatch_size", "cannot exceed max_batch_size")
    _positive(strategy.pipeline_feedback_latency_us, "strategy.pipeline_feedback_latency_us", zero=True)
    if strategy.replicas != 1:
        raise SimulationError("UNSUPPORTED_MECHANISM", "multiple replicas are not implemented")
    if strategy.scheduler not in ("fifo", "continuous_batching"):
        raise SimulationError("UNSUPPORTED_MECHANISM", "unsupported scheduler")
    if strategy.preemption not in ("none", "recompute", "swap"):
        raise SimulationError("UNSUPPORTED_MECHANISM", "supported preemption policies are none/recompute/swap")
    if strategy.swap_fallback not in ("none", "recompute"):
        raise SimulationError("UNSUPPORTED_MECHANISM", "swap_fallback must be none or recompute")
    if strategy.swap_fallback != "none" and strategy.preemption != "swap":
        _invalid("strategy.swap_fallback", "requires preemption=swap")
    if strategy.preemption == "swap" and memory.cpu_swap_bytes <= 0:
        _invalid("strategy.memory.cpu_swap_bytes", "swap requires a positive host capacity")
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
        if not device.name.strip():
            _invalid(f"infra.devices[{i}].name", "must not be empty")
        for name in ("memory_bytes", "compute_tflops", "memory_bandwidth_gbps"):
            _positive(getattr(device, name), f"infra.devices[{i}].{name}")
    # Compute/bandwidth differences are conservatively modeled by slowest rank.
    _positive(infra.link_bandwidth_gbps, "infra.link_bandwidth_gbps")
    _positive(infra.link_latency_us, "infra.link_latency_us", zero=True)
    if not infra.topology.strip():
        _invalid("infra.topology", "must not be empty")
    for f in fields(memory):
        _positive(getattr(memory, f.name), f"strategy.memory.{f.name}",
                  zero=f.name not in ("kv_block_tokens", "kv_pool_bytes", "swap_d2h_bandwidth_gbps",
                                      "swap_h2d_bandwidth_gbps"))
    if calibration.backend not in ("analytical", "synthetic", "profile"):
        raise SimulationError("UNSUPPORTED_MECHANISM", "supported backends are analytical, synthetic and profile")
    if calibration.backend == "profile":
        if calibration.profile_path is None or not calibration.profile_path.strip():
            _invalid("calibration.profile_path", "profile backend requires a profile file")
    elif calibration.profile_path is not None or calibration.allow_extrapolation or calibration.allow_test_fixture:
        _invalid("calibration", "profile file and domain/source overrides require backend=profile")
    for name in ("runtime_name", "runtime_version", "driver_version", "rocm_version", "rccl_version",
                 "timing_boundary"):
        if not getattr(calibration, name).strip():
            _invalid(f"calibration.{name}", "must not be empty")
    if calibration.timing_boundary != "step_includes_output_and_sampling_v1":
        raise SimulationError("UNSUPPORTED_MECHANISM", "unsupported step timing boundary")
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
    warmup_ids = set()
    for i, request in enumerate(exp.execution.warmup):
        path = f"execution.warmup[{i}]"
        if not request.request_id.strip() or request.request_id in warmup_ids:
            _invalid(path, "warmup request IDs must be nonempty and unique")
        warmup_ids.add(request.request_id)
        _positive(request.arrival_ms, f"{path}.arrival_ms", zero=True)
        _positive(request.input_tokens, f"{path}.input_tokens")
        _positive(request.output_tokens, f"{path}.output_tokens")
        if request.timeout_ms is not None:
            _positive(request.timeout_ms, f"{path}.timeout_ms", zero=True)
    if exp.execution.initial_state != "empty_cache":
        raise SimulationError("UNSUPPORTED_MECHANISM", "only empty_cache initial state is supported")
    for f in fields(exp.evaluation):
        value = getattr(exp.evaluation, f.name)
        if f.name != "drain" and value is not None:
            _positive(value, f"evaluation.{f.name}", zero=f.name != "measurement_window_ms")
    if exp.evaluation.max_drain_ms is not None:
        if exp.evaluation.measurement_window_ms is None or not exp.evaluation.drain:
            _invalid("evaluation.max_drain_ms", "requires a measurement window and drain=true")
    if not exp.evaluation.drain and exp.evaluation.measurement_window_ms is None:
        _invalid("evaluation.drain", "drain=false requires a measurement window")
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
    calibration_root = source.parent
    execution_root = source.parent
    for section in ("workload", "infra", "strategy", "calibration", "evaluation", "execution"):
        if isinstance(mapping.get(section), str):
            referenced_path = source.parent / mapping[section]
            mapping[section] = _read(referenced_path)
            if section == "strategy":
                strategy_root = referenced_path.parent
            elif section == "calibration":
                calibration_root = referenced_path.parent
            elif section == "execution":
                execution_root = referenced_path.parent
    if isinstance(mapping.get("strategy"), dict):
        strategy = dict(mapping["strategy"])
        if isinstance(strategy.get("model"), str):
            strategy["model"] = _read(strategy_root / strategy["model"])
        mapping["strategy"] = strategy
    if isinstance(mapping.get("calibration"), dict):
        calibration = dict(mapping["calibration"])
        if isinstance(calibration.get("profile_path"), str):
            calibration["profile_path"] = str((calibration_root / calibration["profile_path"]).resolve())
        mapping["calibration"] = calibration
    if isinstance(mapping.get("execution"), dict):
        execution = dict(mapping["execution"])
        if isinstance(execution.get("warmup"), str):
            execution["warmup"] = _read(execution_root / execution["warmup"])
        mapping["execution"] = execution
    return experiment_from_dict(mapping)
