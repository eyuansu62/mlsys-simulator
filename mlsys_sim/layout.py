"""The supported dense GQA shard plan, shared by timing and memory models."""

from .schema import Experiment, ModelSpec


def stage_layer_counts(experiment: Experiment) -> tuple[int, ...]:
    strategy = experiment.strategy
    if strategy.stage_layers:
        return strategy.stage_layers
    count, remainder = divmod(strategy.model.num_layers, strategy.pipeline_parallel_size)
    return tuple(count + (i < remainder) for i in range(strategy.pipeline_parallel_size))


def local_kv_heads(model: ModelSpec, tp: int) -> int:
    return max(1, model.num_kv_heads // tp)


def layer_parameters(model: ModelSpec, tp: int) -> dict[str, float]:
    hidden = model.hidden_size
    q_o = 2 * hidden * hidden / tp
    k_v = 2 * hidden * local_kv_heads(model, tp) * model.head_dim
    ffn = 3 * hidden * model.intermediate_size / tp
    return {"linear": q_o + k_v + ffn, "norm": 2 * hidden}


def rank_weight_bytes(experiment: Experiment, stage: int) -> int:
    strategy = experiment.strategy
    model, tp = strategy.model, strategy.tensor_parallel_size
    params = layer_parameters(model, tp)
    total = stage_layer_counts(experiment)[stage] * sum(params.values())
    # Untied embedding and LM head, partitioned along the vocabulary dimension.
    if stage == 0:
        total += model.vocab_size * model.hidden_size / tp
    if stage == strategy.pipeline_parallel_size - 1:
        total += model.vocab_size * model.hidden_size / tp
    return int(total * model.weight_bytes)


def rank_kv_bytes_per_token(experiment: Experiment, stage: int) -> int:
    strategy = experiment.strategy
    model = strategy.model
    return int(2 * stage_layer_counts(experiment)[stage]
               * local_kv_heads(model, strategy.tensor_parallel_size)
               * model.head_dim * model.kv_bytes)
