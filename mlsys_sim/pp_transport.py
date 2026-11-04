"""Explicit PP payload geometry and uncalibrated boundary communication costs.

Native receiver waits establish dependency completion, not independent network
service samples. This module estimates wire P2P and receiver AllGather once from
the observed/configured tensor geometry and the experiment's link parameters.
"""

from copy import deepcopy
import math
from typing import Any, NoReturn

from .schema import Experiment, SimulationError, StepEstimate


_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
_CONTRACT = "vllm_native_pp_v1"


def _fail(message: str, code: str = "INVALID_PP_TRANSPORT") -> NoReturn:
    raise SimulationError(code, message)


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


def pipeline_transport_max_tokens(experiment: Experiment) -> int:
    """Resolve the native payload/buffer bound without changing legacy defaults."""
    strategy = experiment.strategy
    configured = strategy.pipeline_transport_max_tokens
    if type(configured) is not int or configured < 0:
        _fail("pipeline_transport_max_tokens must be a nonnegative integer")
    maximum = _positive_integer(strategy.max_batch_tokens, "max_batch_tokens")
    if configured and configured < maximum:
        _fail("pipeline_transport_max_tokens must cover max_batch_tokens")
    return configured or maximum


def native_pp_descriptor(experiment: Experiment, padded_tokens: int) -> dict:
    """Describe the two full resident tensors and each rank's actual wire slice."""
    strategy, model = experiment.strategy, experiment.strategy.model
    if strategy.pipeline_transport != "vllm_native" or strategy.pipeline_parallel_size <= 1:
        _fail("Native PP descriptors require vllm_native transport and PP > 1")
    padded = _positive_integer(padded_tokens, "padded_tokens")
    if padded > pipeline_transport_max_tokens(experiment):
        _fail("padded_tokens exceeds pipeline_transport_max_tokens")
    hidden = _positive_integer(model.hidden_size, "hidden_size")
    tp = _positive_integer(strategy.tensor_parallel_size, "tensor_parallel_size")
    element_size = _DTYPE_BYTES.get(model.activation_dtype)
    if element_size is None or model.activation_bytes != element_size:
        _fail("Native PP requires an explicit activation dtype matching activation_bytes")
    numel = padded * hidden
    sliced = tp > 1 and numel % tp == 0
    full_bytes = numel * element_size
    wire = full_bytes // tp if sliced else full_bytes
    tensors = [{"name": name, "shape": [padded, hidden], "dtype": model.activation_dtype,
        "element_size": element_size, "numel": numel, "full_bytes": full_bytes,
        "tp_sliced": sliced, "wire_bytes_per_rank": wire, "receiver_allgather": sliced}
        for name in ("hidden_states", "residual")]
    return {"contract": _CONTRACT, "padded_tokens": padded, "tensors": tensors,
            "resident_bytes_per_rank": 2 * full_bytes, "wire_bytes_per_rank": 2 * wire}


def validate_native_pp_descriptor(experiment: Experiment, value: Any) -> dict:
    """Reject missing, invented or loosely typed payload evidence."""
    fields = {"contract", "padded_tokens", "tensors", "resident_bytes_per_rank", "wire_bytes_per_rank"}
    if not isinstance(value, dict) or set(value) != fields:
        _fail("Native PP descriptor has missing or unknown fields")
    expected = native_pp_descriptor(experiment, value["padded_tokens"])
    for name in ("resident_bytes_per_rank", "wire_bytes_per_rank"):
        _positive_integer(value[name], name)
    tensors = value["tensors"]
    if not isinstance(tensors, list) or len(tensors) != 2:
        _fail("Native PP requires hidden_states and residual tensor descriptors")
    for supplied, required in zip(tensors, expected["tensors"]):
        if not isinstance(supplied, dict) or set(supplied) != set(required):
            _fail("Native PP tensor descriptor has missing or unknown fields")
        shape = supplied["shape"]
        if not isinstance(shape, list) or len(shape) != 2:
            _fail("Native PP tensor shape must be [padded_tokens, hidden_size]")
        for dimension in shape:
            _positive_integer(dimension, "tensor.shape[]")
        for name in ("element_size", "numel", "full_bytes", "wire_bytes_per_rank"):
            _positive_integer(supplied[name], f"tensor.{name}")
        for name in ("tp_sliced", "receiver_allgather"):
            if type(supplied[name]) is not bool:
                _fail(f"tensor.{name} must be boolean")
    if value != expected:
        _fail("Native PP tensor geometry differs from the experiment or slicing contract")
    return deepcopy(expected)


def estimate_pipeline_transfer(experiment: Experiment, query_tokens,
                               stage_estimate: StepEstimate | None = None, *, stage_index: int = 0,
                               communication_model: Any = None) -> dict:
    """Estimate one boundary, preserving the original full-hidden default cost."""
    queries = tuple(query_tokens)
    if not queries:
        _fail("A PP transfer requires a nonempty query batch")
    for query in queries:
        _positive_integer(query, "query_tokens[]")
    strategy, model, infra = experiment.strategy, experiment.strategy.model, experiment.infra
    if type(stage_index) is not int or not 0 <= stage_index < strategy.pipeline_parallel_size - 1:
        _fail("PP transfer stage_index must identify a source boundary")
    descriptor = None
    if strategy.pipeline_transport == "vllm_native":
        transport = stage_estimate.breakdown.get("pp_transport") if stage_estimate is not None else None
        if transport is not None:
            if not isinstance(transport, dict) or "outgoing" not in transport:
                _fail("A native PP boundary requires its stage's outgoing descriptor")
            descriptor = validate_native_pp_descriptor(experiment, transport["outgoing"])
            payload_source = "observed_profile_descriptor"
        else:
            if stage_estimate is not None and stage_estimate.source in ("measured_profile", "test_fixture_profile"):
                _fail("Native Profile stage has no outgoing PP descriptor")
            descriptor = native_pp_descriptor(experiment, sum(queries))
            payload_source = "configured_logical_tokens"
        if descriptor["padded_tokens"] < sum(queries):
            _fail("Native PP padded_tokens cannot be smaller than the logical query count")
        payload = descriptor["wire_bytes_per_rank"]
    elif strategy.pipeline_transport == "full_hidden":
        payload = int(sum(queries) * model.hidden_size * model.activation_bytes)
        payload_source = "configured_logical_tokens"
    else:
        _fail("Unsupported pipeline_transport")
    if not strategy.pipeline_p2p:
        return {"duration_ms": 0.0, "bytes_per_rank": payload,
                "source": "explicit_ideal_zero_cost", "descriptor": descriptor,
                "breakdown": {"duration_ms": 0.0, "payload_source": payload_source,
                              "p2p_enabled": False}}
    if communication_model is None and experiment.calibration.native_pp_communication_profile_path:
        from .pp_communication import PPCommunicationModel
        communication_model = PPCommunicationModel(experiment)
    if communication_model is not None and descriptor is None:
        _fail("Communication Profiles are confined to native PP boundaries")
    bandwidth = infra.link_bandwidth_gbps * 1e6
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        _fail("Effective PP link bandwidth must be positive and finite", "INVALID_CONFIG")
    latency = infra.link_latency_us / 1000
    if not math.isfinite(latency) or latency < 0:
        _fail("PP link latency must be nonnegative and finite", "INVALID_CONFIG")
    parts = []
    if descriptor is None:
        duration = latency + payload / bandwidth
        breakdown = {"duration_ms": duration, "p2p_ms": duration, "receiver_allgather_ms": 0.0,
                     "payload_source": payload_source, "p2p_enabled": True}
    else:
        tp = strategy.tensor_parallel_size
        for tensor in descriptor["tensors"]:
            component: dict[str, Any] = {"name": tensor["name"]}
            if communication_model is not None:
                estimates = [communication_model.estimate("p2p", tensor["wire_bytes_per_rank"],
                             stage_index + 1, tp_rank=rank) for rank in range(tp)]
                p2p = max(estimate["duration_ms"] for estimate in estimates)
                gather_estimate = (communication_model.estimate("allgather", tensor["full_bytes"],
                                    stage_index + 1) if tensor["receiver_allgather"] else None)
                gather = gather_estimate["duration_ms"] if gather_estimate is not None else 0.0
                component.update(p2p_rank_estimates=estimates, receiver_allgather_estimate=gather_estimate)
            else:
                p2p = latency + tensor["wire_bytes_per_rank"] / bandwidth
                gather = ((tp - 1) * latency + (tp - 1) / tp * tensor["full_bytes"] / bandwidth
                          if tensor["receiver_allgather"] else 0.0)
            component.update(p2p_ms=p2p, receiver_allgather_ms=gather, duration_ms=p2p + gather)
            parts.append(component)
        duration = sum(part["duration_ms"] for part in parts)
        breakdown = {"duration_ms": duration, "p2p_ms": sum(part["p2p_ms"] for part in parts),
            "receiver_allgather_ms": sum(part["receiver_allgather_ms"] for part in parts),
            "tensors": parts, "payload_source": payload_source, "p2p_enabled": True,
            "communication_calibrated": communication_model is not None, "receiver_wait_used_as_network_service": False,
            "serialization_policy": "two_p2p_messages_and_required_receiver_allgathers"}
        if communication_model is not None:
            breakdown["communication_profile_provenance"] = {
                key: deepcopy(value) for key, value in communication_model.provenance.items()
                if key not in ("benchmark", "application")}
    if not math.isfinite(duration) or duration <= 0:
        _fail("PP transfer duration must be positive and finite", "INVALID_CONFIG")
    return {"duration_ms": duration, "bytes_per_rank": payload,
            "source": ("analytical_uncalibrated" if communication_model is None else
                       parts[0]["p2p_rank_estimates"][0]["source"]),
            "breakdown": breakdown, "descriptor": descriptor}
