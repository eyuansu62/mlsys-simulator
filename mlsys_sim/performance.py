"""Explicit, uncalibrated dense-GQA roofline and communication estimates."""

from collections.abc import Sequence
import math
from typing import Any

from .batching import batch_members
from .config import validate_experiment
from .layout import layer_parameters, local_kv_heads, stage_layer_counts
from .schema import Experiment, SimulationError, StepEstimate


def _finite(value, name, *, allow_zero=False):
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise SimulationError(
            "INVALID_CONFIG", f"{name} must remain finite and "
            f"{'nonnegative' if allow_zero else 'positive'} after unit conversion")
    return value


def ring_allreduce_ms(ranks: int, message_bytes: float, bandwidth_gbps: float,
                      latency_us: float) -> float:
    """Ring baseline; message_bytes is the full logical tensor on each rank."""
    if ranks <= 1:
        return 0.0
    bandwidth = _finite(bandwidth_gbps * 1e6, "effective link bytes/ms")
    return _finite(2 * (ranks - 1) * latency_us / 1000
                   + 2 * (ranks - 1) / ranks * message_bytes / bandwidth,
                   "allreduce duration", allow_zero=True)


def _allgather_ms(ranks, total_bytes, bandwidth_gbps, latency_us):
    if ranks <= 1:
        return 0.0
    bandwidth = _finite(bandwidth_gbps * 1e6, "effective link bytes/ms")
    return _finite((ranks - 1) * latency_us / 1000
                   + (ranks - 1) / ranks * total_bytes / bandwidth,
                   "allgather duration", allow_zero=True)


class AnalyticalModel:
    """Three serialized macro components, without kernel or overlap simulation."""

    def __init__(self, experiment: Experiment) -> None:
        validate_experiment(experiment)
        if experiment.strategy.replicas != 1:
            raise SimulationError("UNSUPPORTED_MECHANISM",
                                  "AnalyticalModel describes one replica; use replica_experiment first")
        if experiment.calibration.backend == "profile":
            raise SimulationError("UNSUPPORTED_MECHANISM",
                                  "Use create_performance_model for whole-stage profile predictions")
        self.experiment = experiment

    def estimate_step(
        self,
        phase: str,
        query_tokens: Sequence[int],
        context_lengths: Sequence[int],
        stage_index: int = 0,
        *,
        request_phases: Sequence[str] | None = None,
    ) -> StepEstimate:
        exp = self.experiment
        strategy, profile, infra = exp.strategy, exp.calibration, exp.infra
        model, tp = strategy.model, strategy.tensor_parallel_size
        queries, contexts = tuple(query_tokens), tuple(context_lengths)
        if not queries or len(queries) != len(contexts):
            raise SimulationError("INVALID_CONFIG", "query/context arrays must be nonempty and equal length")
        if any(type(q) is not int or q <= 0 for q in queries):
            raise SimulationError("INVALID_CONFIG", "query lengths must be positive integers")
        if any(type(c) is not int or c < 0 for c in contexts):
            raise SimulationError("INVALID_CONFIG", "context lengths must be nonnegative integers")
        members = batch_members(phase, queries, request_phases)
        counts = stage_layer_counts(exp)
        if type(stage_index) is not int or not 0 <= stage_index < len(counts):
            raise SimulationError("INVALID_CONFIG", "invalid stage index")
        layers = counts[stage_index]
        if profile.backend == "synthetic":
            duration = _finite(getattr(profile, f"{phase}_ms") * (layers / model.num_layers),
                               "synthetic stage duration")
            return StepEstimate(duration, {"synthetic_fixed_batch_ms": duration},
                                "synthetic", ("Fixed duration per phase batch; not a hardware prediction.",))

        devices = infra.devices[stage_index * tp:(stage_index + 1) * tp]
        compute = _finite(min(d.compute_tflops for d in devices) * 1e9
                          * profile.compute_efficiency, "effective FLOPs/ms")
        bandwidth = _finite(min(d.memory_bandwidth_gbps for d in devices) * 1e6
                            * profile.bandwidth_efficiency, "effective HBM bytes/ms")
        tokens, batch = sum(queries), len(queries)
        local_params = layer_parameters(model, tp)

        def macro(flops, hbm_bytes, launches):
            compute_ms, memory_ms = flops / compute, hbm_bytes / bandwidth
            overhead_ms = launches * profile.launch_overhead_us / 1000
            return {"flops_per_rank": flops, "hbm_bytes_per_rank": hbm_bytes,
                    "compute_ms": compute_ms, "memory_ms": memory_ms,
                    "overhead_ms": overhead_ms,
                    "duration_ms": max(compute_ms, memory_ms) + overhead_ms}

        # One joint batch reuses weights across both serving phases when mixed.
        linear_flops = 2 * layers * local_params["linear"] * tokens
        linear_bytes = (layers * local_params["linear"] * model.weight_bytes
                        + 4 * layers * tokens * model.hidden_size * model.activation_bytes)
        linear = macro(linear_flops, linear_bytes, layers)
        # Causal attention includes the prefix and lower triangle of new queries.
        pairs = sum(q * c + q * (q + 1) / 2 for q, c in zip(queries, contexts))
        attn_flops = 4 * layers * (model.num_attention_heads / tp) * model.head_dim * pairs
        kv_bytes = (2 * layers * local_kv_heads(model, tp) * model.head_dim
                    * model.kv_bytes * sum(c + q for c, q in zip(contexts, queries)))
        query_io = 2 * layers * tokens * model.hidden_size / tp * model.activation_bytes
        attention = macro(attn_flops, kv_bytes + query_io, layers)
        misc = macro(12 * layers * tokens * model.hidden_size,
                     8 * layers * tokens * model.hidden_size * model.activation_bytes,
                     layers)

        output_flops = 0.0
        output_bytes = 0.0
        if stage_index == 0:
            output_bytes += tokens * model.hidden_size * model.weight_bytes / tp
        if stage_index == len(counts) - 1:
            # Only last-position logits per sequence are materialized.
            output_flops += 2 * batch * model.vocab_size * model.hidden_size / tp
            output_bytes += model.vocab_size * model.hidden_size * model.weight_bytes / tp
            output_bytes += batch * model.vocab_size * model.activation_bytes / tp
        output = macro(output_flops, output_bytes, 1 if output_bytes else 0)
        payload = tokens * model.hidden_size * model.activation_bytes
        collective_ms = 2 * layers * ring_allreduce_ms(
            tp, payload, infra.link_bandwidth_gbps, infra.link_latency_us)
        logits_gather_ms = 0.0
        if stage_index == len(counts) - 1:
            logits_gather_ms = _allgather_ms(
                tp, batch * model.vocab_size * model.activation_bytes,
                infra.link_bandwidth_gbps, infra.link_latency_us)
        framework_ms = profile.framework_overhead_us / 1000 if stage_index == 0 else 0.0
        breakdown = {"linear": linear, "attention": attention, "misc": misc,
                     "embedding_and_output": output,
                     "communication": {"duration_ms": collective_ms + logits_gather_ms,
                                       "allreduce_count": 2 * layers if tp > 1 else 0,
                                       "allreduce_payload_bytes": payload,
                                       "allreduce_ms": collective_ms,
                                       "logits_allgather_ms": logits_gather_ms},
                     "framework": {"duration_ms": framework_ms}}
        duration = sum(part["duration_ms"] for part in breakdown.values())
        if phase == "mixed":
            breakdown["batch"] = {"phase": phase, "request_phases": list(members),
                                  "shared_macro_execution": True}
        _finite(duration, "performance estimate")
        return StepEstimate(duration, breakdown, "analytical_uncalibrated", (
            "Uncalibrated effective compute/bandwidth and serialized macro operations.",
            "Ideal weight and GQA KV reuse; attention IO is an optimistic lower-bound approximation.",
            "Three macro launch terms per block; no overlap or link contention.",
            "Untied, vocabulary-sharded embedding/output; final logits use AllGather.",
        ))


def predict_experiment(experiment: Experiment) -> dict[str, Any]:
    """Shape-only C01/C02/C04 report, including analytical PP stage costs."""
    from .memory import MemoryPlanner
    from .profiles import create_performance_model

    validate_experiment(experiment)
    if experiment.strategy.replicas > 1:
        return _predict_replicas(experiment)
    exp, strategy = experiment, experiment.strategy
    predictor = create_performance_model(exp)
    planner = MemoryPlanner(exp)
    requests = exp.workload
    queries = [r.input_tokens for r in requests] or [1]
    contexts = [0] * len(queries)
    stages = []
    prefill_total, decode_total = 0.0, 0.0
    for stage, layers in enumerate(stage_layer_counts(exp)):
        prefill = predictor.estimate_step("prefill", queries, contexts, stage)
        decode = predictor.estimate_step("decode", [1] * len(queries), queries, stage)
        stages.append({"stage": stage, "layers": layers,
                       "device_ids": [d.id for d in exp.infra.devices[
                           stage * strategy.tensor_parallel_size:(stage + 1) * strategy.tensor_parallel_size]],
                       "prefill_ms": prefill.duration_ms, "decode_step_ms": decode.duration_ms,
                       "prefill_source": prefill.source, "decode_source": decode.source,
                       "prefill_breakdown": prefill.breakdown,
                       "decode_breakdown": decode.breakdown})
        prefill_total += prefill.duration_ms
        decode_total += decode.duration_ms
    boundaries = strategy.pipeline_parallel_size - 1
    prefill_payload = sum(queries) * strategy.model.hidden_size * strategy.model.activation_bytes
    decode_payload = len(queries) * strategy.model.hidden_size * strategy.model.activation_bytes

    def transfer(payload):
        if not boundaries or not strategy.pipeline_p2p:
            return 0.0
        bandwidth = _finite(exp.infra.link_bandwidth_gbps * 1e6, "effective P2P bytes/ms")
        return _finite(boundaries * (exp.infra.link_latency_us / 1000
                                     + payload / bandwidth), "P2P duration", allow_zero=True)

    targets = {r.request_id: r.input_tokens for r in requests} or {"shape": 1}
    memory = planner.check(targets, queries, "prefill")
    prefill_p2p, decode_p2p = transfer(prefill_payload), transfer(decode_payload)
    isolated_prefill = _finite(prefill_total + prefill_p2p, "isolated prefill duration")
    isolated_decode = _finite(decode_total + decode_p2p, "isolated decode duration")
    provenance = getattr(predictor, "provenance", None)
    return {"status": "COMPLETED", "mode": "predict", "backend": "simulated",
            "source": exp.calibration.backend, "hardware_validated": False,
            "profile_version": (provenance["profile_version"] if provenance
                                else exp.calibration.profile_version),
            "profile_provenance": dict(provenance) if provenance else None,
            "shape": {"query_tokens": queries, "context_lengths": contexts,
                      "serving_cache_and_chunk_policies_applied": False,
                      "requests_in_batch": len(queries),
                      "within_batch_token_limit": sum(queries) <= strategy.max_batch_tokens,
                      "within_batch_size_limit": len(queries) <= strategy.max_batch_size},
            "stages": stages, "memory": planner.snapshot(),
            "prefill_memory_feasible": memory.ok, "memory_error_code": memory.code,
            "memory_check_details": memory.details,
            "communication": {"pp_boundaries": boundaries,
                              "p2p_source": ("analytical_uncalibrated" if strategy.pipeline_p2p
                                             else "explicit_ideal_zero_cost"),
                              "prefill_p2p_bytes_per_rank_per_boundary": prefill_payload,
                              "decode_p2p_bytes_per_rank_per_boundary": decode_payload,
                              "prefill_p2p_ms": prefill_p2p,
                              "decode_p2p_ms": decode_p2p},
            "isolated_prefill_ms": isolated_prefill,
            "isolated_decode_step_ms": isolated_decode,
            "online_metrics": {"ttft_ms": None, "tpot_ms": None, "p99_ms": None,
                               "goodput_requests_per_s": None},
            "online_metrics_reason": "SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING",
            "assumptions": [
                "Analyzes one phase batch containing every workload row, ignoring arrival times.",
                "PP activations replicated per TP rank; identical independent links, no contention.",
                "No sampling/feedback latency included in PP forward path.",
                ("Stage timings use the supplied whole-stage profile; PP transfers remain analytical."
                 if provenance else "Stage timings are analytical/synthetic, without measured calibration."),
                "Profile calibration does not itself establish held-out hardware prediction accuracy.",
            ]}


def _predict_replicas(experiment: Experiment) -> dict[str, Any]:
    """Compare each local shape without assigning online workload arrivals."""
    from .layout import replica_experiment

    predictions: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    versions: list[dict[str, Any]] = []
    sources: set[str] = set()
    for index in range(experiment.strategy.replicas):
        local = replica_experiment(experiment, index)
        result = predict_experiment(local)
        predictions.append({"replica_index": index,
                            "device_ids": [device.id for device in local.infra.devices],
                            "prediction": result})
        memory.extend({"replica_index": index, **rank} for rank in result["memory"])
        versions.append({"replica_index": index, "profile_version": result["profile_version"]})
        if result["profile_provenance"] is not None:
            item = result["profile_provenance"]
            provenance.append({"replica_index": index, **item})
            sources.add("measured_profile" if item["source_kind"] == "measured"
                        else "test_fixture_profile")
    if not sources:
        sources.add(experiment.calibration.backend)
    version_set = {item["profile_version"] for item in versions}
    return {
        "status": "COMPLETED", "mode": "predict", "backend": "simulated",
        "source": experiment.calibration.backend,
        "calibration_source": next(iter(sources)) if len(sources) == 1 else "mixed_replica_profiles",
        "hardware_validated": False,
        "replica_count": experiment.strategy.replicas,
        "replica_predictions": predictions,
        "profile_version": next(iter(version_set)) if len(version_set) == 1 else "per_replica",
        "profile_versions": versions,
        "profile_provenance": {"replicas": provenance} if provenance else None,
        "memory": memory,
        "prefill_memory_feasible": all(item["prediction"]["prefill_memory_feasible"]
                                        for item in predictions),
        "online_metrics": {"ttft_ms": None, "tpot_ms": None, "p99_ms": None,
                           "goodput_requests_per_s": None},
        "online_metrics_reason": "SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING",
        "replica_routing_applied": False,
        "assumptions": [
            "Every workload row forms the same isolated phase batch on each replica for shape comparison.",
            "No arrival routing, queueing, cache reuse, chunking or mixed online batching is executed.",
            "Replica resources and paired TP/PP links are independent; shared fabric contention is not modeled.",
            "Profile calibration does not itself establish held-out hardware prediction accuracy.",
        ],
    }
