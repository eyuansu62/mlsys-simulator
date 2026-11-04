"""The supported dense GQA shard plan, shared by timing and memory models."""

from dataclasses import replace

from .schema import Experiment, ModelSpec, SimulationError


def replica_experiment(experiment: Experiment, replica_index: int) -> Experiment:
    """Slice replica-major/stage-major/TP-rank placement into a local model."""
    from .config import validate_experiment

    validate_experiment(experiment)
    replicas = experiment.strategy.replicas
    if type(replica_index) is not int or not 0 <= replica_index < replicas:
        raise SimulationError("INVALID_CONFIG", "replica_index is outside the configured replica set")
    width = experiment.strategy.tensor_parallel_size * experiment.strategy.pipeline_parallel_size
    devices = experiment.infra.devices[replica_index * width:(replica_index + 1) * width]
    calibration = experiment.calibration
    if replicas > 1 and calibration.backend == "profile":
        if len(calibration.replica_profile_paths) != replicas:
            raise SimulationError("INVALID_CONFIG", "one Profile path is required per replica")
        calibration = replace(calibration, profile_path=calibration.replica_profile_paths[replica_index],
                              replica_profile_paths=())
    return replace(experiment, infra=replace(experiment.infra, devices=devices),
                   strategy=replace(experiment.strategy, replicas=1), calibration=calibration)


def stage_layer_counts(experiment: Experiment) -> tuple[int, ...]:
    strategy = experiment.strategy
    if strategy.stage_layers:
        return strategy.stage_layers
    count, remainder = divmod(strategy.model.num_layers, strategy.pipeline_parallel_size)
    return tuple(count + (i < remainder) for i in range(strategy.pipeline_parallel_size))


def local_kv_heads(model: ModelSpec, tp: int) -> int:
    return max(1, model.num_kv_heads // tp)


def layer_parameters(model: ModelSpec, tp: int) -> dict[str, int]:
    hidden = model.hidden_size
    q_o = 2 * hidden * model.query_projection_size // tp
    k_v = 2 * hidden * local_kv_heads(model, tp) * model.head_dim
    ffn = 3 * hidden * model.intermediate_size // tp
    # Qwen3 normalizes each head with one shared head_dim-sized Q/K scale.
    # Like residual RMSNorm scales, those vectors are replicated on TP ranks.
    norms = 2 * hidden + (2 * model.head_dim if model.qk_norm else 0)
    return {"linear": q_o + k_v + ffn, "norm": norms}


def rank_weight_bytes(experiment: Experiment, stage: int) -> int:
    strategy = experiment.strategy
    model, tp = strategy.model, strategy.tensor_parallel_size
    params = layer_parameters(model, tp)
    total = stage_layer_counts(experiment)[stage] * sum(params.values())
    # Tied vocabulary weights alias on the same rank only. Separate PP
    # endpoints still need physical copies, even though their values agree.
    if stage == 0:
        total += model.vocab_size * model.hidden_size // tp
    if stage == strategy.pipeline_parallel_size - 1:
        if not model.tie_word_embeddings or strategy.pipeline_parallel_size > 1:
            total += model.vocab_size * model.hidden_size // tp
        if model.final_norm:
            total += model.hidden_size
    return total * int(model.weight_bytes)


def rank_kv_bytes_per_token(experiment: Experiment, stage: int) -> int:
    strategy = experiment.strategy
    model = strategy.model
    return (2 * stage_layer_counts(experiment)[stage]
            * local_kv_heads(model, strategy.tensor_parallel_size)
            * model.head_dim * int(model.kv_bytes))
