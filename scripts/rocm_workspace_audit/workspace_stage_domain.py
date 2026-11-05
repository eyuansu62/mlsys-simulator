"""Pure-data gates for pinned Worker Workspace acquisitions.

These checks bind actual prepared-batch metadata and rank/shape domains. They do
not authenticate producer files or establish allocator/native/PP attribution;
the assembler must independently verify those sources and complete lifetimes.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from typing import Any, cast

from mlsys_sim.config import experiment_from_dict
from mlsys_sim.profiles import profile_identity
from mlsys_sim.vllm_capture import normalize_vllm_capture
from mlsys_sim.schema import SimulationError
from workspace_domain import RUNTIME_VERSION, TORCH_VERSION, TORCH_GIT, HIP_VERSION
from workspace_stage_session import validate_plan_experiment

_DEVICES = {
    "AMD Radeon RX 7900 XTX": (25753026560, "gfx1100"),
    "AMD Radeon RX 9070 XT": (17095983104, "gfx1201"),
}
_GEOMETRY = {"num_layers": 28, "hidden_size": 2048, "intermediate_size": 6144,
    "num_attention_heads": 16, "num_kv_heads": 8, "head_dim": 128, "vocab_size": 151936}
_STACK = {"runtime_name": "vllm", "runtime_version": RUNTIME_VERSION,
          "driver_version": "6.16.13", "rocm_version": "10.1.0", "rccl_version": "2.30.7",
          "timing_boundary": "step_includes_output_and_sampling_v1"}
_SAMPLING = {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0,
             "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
             "logprobs": None, "prompt_logprobs": None}
_RCCL_IDENTITY_FIELDS = ("verified", "runtime_version", "library_sha256", "path", "version_code", "verification_scope")


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Workspace domain input must be finite JSON") from exc


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    _need(type(value) is int and minimum <= value <= 2**63 - 1, label + " must be a bounded integer")
    return value


def _sha(value: Any, label: str) -> str:
    _need(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
          label + " must be a lowercase SHA256")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    _need(isinstance(value, dict), label + " must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    _need(isinstance(value, list), label + " must be an array")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    _need(_json(actual) == _json(expected), label + " differs")


def _config(config: dict[str, Any], tp: int, pp: int) -> None:
    expected = {
        "api_revision": "98dff2a81", "async_scheduling": False,
        "tensor_parallel_size": tp, "pipeline_parallel_size": pp,
        "model_config": {"enforce_eager": True, "max_model_len": 512},
        "scheduler_config": {"policy": "fcfs", "max_num_batched_tokens": 512,
            "max_num_seqs": 1, "enable_chunked_prefill": False},
        "cache_config": {"cache_dtype": "auto", "block_size": 16,
            "enable_prefix_caching": False, "kv_sharing_fast_prefill": False,
            "kv_cache_memory_bytes": 268435456, "swap_space": None, "cpu_offload_gb": None},
        "parallel_config": {"data_parallel_size": 1, "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1, "use_ubatching": False, "enable_expert_parallel": False},
        "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
        "torch_build": {"torch_version": TORCH_VERSION, "hip_build_version": HIP_VERSION},
    }
    for key, value in expected.items():
        if isinstance(value, dict):
            section = _object(config.get(key), "Runtime " + key)
            for field, expected_value in value.items():
                _need(field in section, "Runtime field missing: " + key + "." + field)
                _equal(section[field], expected_value, "Runtime " + key + "." + field)
        else:
            _equal(config.get(key), value, "Runtime " + key)
    backends = config.get("actual_attention_backends")
    _need(isinstance(backends, list) and bool(backends)
          and all(isinstance(item, str) and item for item in backends)
          and len(set(backends)) == len(backends), "Actual attention backend observation is missing or duplicated")
    sampling = _object(config.get("sampling_policy"), "sampling policy")
    _need(set(sampling) == set(_SAMPLING), "Sampling policy fields differ from the fixed greedy domain")
    for key, expected_value in _SAMPLING.items():
        value = sampling[key]
        if expected_value is None:
            _need(value is None, "Sampling " + key + " must be absent")
        else:
            _need(type(value) in (int, float) and math.isfinite(value) and value == expected_value,
                  "Sampling " + key + " differs from the fixed greedy domain")
    environment = _object(config.get("environment"), "Runtime environment")
    _need(environment.get("VLLM_USE_V2_MODEL_RUNNER") == "0", "Prepared Runtime is not explicit MRv1")


def _device(value: Any, expected: Any, label: str) -> None:
    observed = _object(value, label)
    _need(expected.name in _DEVICES, "Unsupported physical GPU in declared Workspace domain")
    capacity, architecture = _DEVICES[expected.name]
    _equal(observed.get("name"), expected.name, label + " name")
    _need(type(observed.get("memory_bytes")) is int and observed["memory_bytes"] == expected.memory_bytes == capacity,
          label + " capacity differs from actual/declaration")
    arch = observed.get("architecture")
    _need(isinstance(arch, str) and arch.split(":")[0] == architecture, label + " architecture differs")


def _worker_hardware(worker: dict[str, Any], experiment: Any, global_rank: int, logical: int) -> None:
    _equal(worker.get("runtime_version"), RUNTIME_VERSION, "Actual vLLM version")
    _equal(worker.get("torch_version"), TORCH_VERSION, "Actual Torch version")
    _equal(worker.get("torch_git"), TORCH_GIT, "Actual Torch source build")
    _equal(worker.get("hip_build_version"), HIP_VERSION, "Actual HIP build")
    _device(worker.get("observed_device"), experiment.infra.devices[global_rank], "Worker GPU")
    hardware = _object(worker.get("observed_hardware_inventory"), "observed hardware inventory")
    _need(hardware.get("kind") == "rocm_inventory" and type(hardware.get("schema_version")) is int
          and hardware["schema_version"] == 1, "A versioned actual ROCm inventory is required")
    _need(hardware.get("status") == "READY" and hardware.get("source", {}).get("kind") == "measured",
          "Actual measured READY hardware inventory is required")
    for key in ("driver_version", "rocm_version", "rccl_version"):
        expected = getattr(experiment.calibration, key)
        _need(hardware.get("versions", {}).get(key) == expected == _STACK[key],
              "Observed/declaration software component differs: " + key)
    torch_fact = _object(hardware.get("torch"), "inventory Torch")
    _equal(torch_fact.get("version"), TORCH_VERSION, "Inventory Torch version")
    _equal(torch_fact.get("hip_version"), HIP_VERSION, "Inventory HIP version")
    _need(torch_fact.get("cuda_available") is True, "Observed HIP GPU is unavailable")
    devices = _array(hardware.get("devices"), "Inventory devices")
    _need(len(devices) == len(experiment.infra.devices),
          "Inventory must describe the full visible local GPU cohort")
    indices = [_integer(_object(item, "Inventory GPU").get("index"), "Inventory GPU index") for item in devices]
    _need(sorted(indices) == list(range(len(devices))), "Actual visible GPU inventory indices are incomplete or duplicated")
    matches = [item for item in devices if isinstance(item, dict) and item.get("index") == logical]
    _need(len(matches) == 1, "Worker logical GPU index is not unique in actual inventory")
    _device(matches[0], experiment.infra.devices[global_rank], "Inventory worker GPU")
    loaded = _object(worker.get("loaded_rccl"), "loaded RCCL")
    _need(loaded.get("verified") is True and loaded.get("runtime_version") == experiment.calibration.rccl_version
          and loaded.get("verification_scope") == "local_proc_maps_loaded_library_ncclGetVersion"
          and type(loaded.get("version_code")) is int and loaded["version_code"] == 23007
          and isinstance(loaded.get("path"), str) and loaded["path"].startswith("/"),
          "Actual loaded RCCL binding is missing")
    digest = _sha(loaded.get("library_sha256"), "loaded RCCL digest")
    library = _object(hardware.get("rccl_library"), "inventory RCCL library")
    _need(library.get("version") == loaded["runtime_version"] and library.get("sha256") == digest,
          "Inventory and loaded RCCL library differ")


def _validate_stage_workspace_domain(
    plan: dict[str, Any], workers: list[dict[str, Any]], *, experiment: dict[str, Any],
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate raw Worker identity and exact prepared-shape cohort, without fitting.

    ``experiment`` must be read from the caller-verified bound-experiment file.
    File contents, receipts, actual probe finalization and storage attribution
    remain independent assembler obligations; no path is read by this helper.
    """
    _json((plan, workers, experiment))
    plan = _object(plan, "session plan")
    _need(type(plan.get("schema_version")) is int and plan["schema_version"] == 1, "Unknown session plan schema")
    raw_mode = plan.get("mode")
    _need(raw_mode in ("single", "tp2", "pp2"), "Unsupported Workspace placement")
    mode = cast(str, raw_mode)
    _need(plan.get("cohort") in ("train", "repeat", "pilot"), "Unknown acquisition cohort")
    repeats = _integer(plan.get("repeats"), "repeats", 1)
    _need(repeats <= 3, "Workspace supports at most three predeclared repetitions")
    session = plan.get("session_id")
    _need(isinstance(session, str) and bool(session), "Session identity is missing")
    clock = plan.get("clock_domain")
    _need(isinstance(clock, str) and clock.startswith("linux-boot:"), "Actual shared Linux boot clock is missing")
    _equal(plan.get("clock"), "CLOCK_MONOTONIC", "prepared clock")
    _equal(plan.get("native_clock"), "CLOCK_MONOTONIC_RAW", "native clock")
    _need(type(plan.get("expected_formal_records_per_rank")) is int
          and plan["expected_formal_records_per_rank"] == 8 * repeats, "Declared formal cohort size differs")
    try:
        declared = experiment_from_dict(experiment)
        validate_plan_experiment(declared, mode)
    except (SimulationError, ValueError, KeyError) as exc:
        raise ValueError("Unsupported declared Workspace experiment: " + str(exc)) from exc
    tp, pp = declared.strategy.tensor_parallel_size, declared.strategy.pipeline_parallel_size
    world = tp * pp
    for key, expected in _STACK.items():
        _equal(getattr(declared.calibration, key), expected, "Declared " + key)
    model = declared.strategy.model
    _need(model.name == "Qwen3-1.7B" and model.tie_word_embeddings is (pp == 1)
          and model.qk_norm is True and model.final_norm is True, "Unsupported loaded model/tying domain")
    for field, value in _GEOMETRY.items():
        _need(type(getattr(model, field)) is int and getattr(model, field) == value,
              "Unsupported Qwen geometry: " + field)
    _need(not declared.strategy.stage_layers or tuple(declared.strategy.stage_layers) == (14, 14),
          "Only the fixed even PP layer split is supported")
    _need(isinstance(workers, list) and len(workers) == world, "Missing or extra physical worker ranks")
    if plan_sha256 is not None:
        _sha(plan_sha256, "actual plan SHA")
    worker_source = _sha(plan.get("worker_source_sha256"), "planned worker source SHA")
    coordinates: dict[int, dict[str, Any]] = {}
    placements: dict[int, dict[str, int]] = {}
    pids: set[int] = set()
    logical_devices: set[int] = set()
    plan_hashes: set[str] = set()
    visible_inventory: set[str] = set()
    loaded_rccl_digests: set[str] = set()
    raw_captures: list[dict[str, Any]] = []
    formal_by_rank: dict[str, list[dict[str, Any]]] = {}
    expected_shapes = Counter({("prefill", q, 0): repeats for q in (16, 32)})
    expected_shapes.update({("decode", 1, q + offset): repeats for q in (16, 32) for offset in range(3)})
    for worker in workers:
        worker = _object(worker, "raw worker")
        _need(worker.get("kind") == "vllm_worker_workspace_raw" and type(worker.get("schema_version")) is int
              and worker["schema_version"] == 1, "Unsupported raw worker kind/schema")
        _need(worker.get("capture_closed") is True and worker.get("capture_errors") == []
              and worker.get("complete_workspace") is False and worker.get("hardware_validated") is False,
              "Raw worker is open, failed or falsely claims completed attribution")
        for key in ("native_sdk_stop_status", "native_sdk_finalize_status"):
            _need(type(worker.get(key)) is int and worker[key] == 0, "Raw worker lacks successful " + key)
        _equal(worker.get("session_id"), session, "Worker session")
        raw_place = _object(worker.get("placement"), "actual placement")
        stage = _integer(raw_place.get("stage_index"), "stage")
        rank = _integer(raw_place.get("tp_rank"), "TP rank")
        global_rank = _integer(raw_place.get("global_rank"), "global rank")
        logical = _integer(raw_place.get("logical_device_index"), "logical device")
        pid = _integer(raw_place.get("process_id"), "actual process ID", 1)
        _need(stage < pp and rank < tp and global_rank == stage * tp + rank and logical < world,
              "Actual rank/device is outside the declared layout")
        _need(global_rank not in coordinates and pid not in pids and logical not in logical_devices,
              "Worker rank, process or logical device is duplicated")
        for key, value in {"stage_index": stage, "tp_rank": rank, "global_rank": global_rank,
                           "logical_device_index": logical, "process_id": pid,
                           "tensor_parallel_size": tp, "pipeline_parallel_size": pp, "world_size": world}.items():
            _equal(worker.get(key), value, "Worker " + key)
            _equal(raw_place.get(key), value, "Placement " + key)
        digest = _sha(worker.get("plan_sha256"), "Worker plan SHA")
        if plan_sha256 is not None:
            _equal(digest, plan_sha256, "Worker immutable plan SHA")
        plan_hashes.add(digest)
        sources = _object(worker.get("source_bindings"), "worker source binding")
        _equal(sources.get("worker_source_sha256"), worker_source, "Executed worker source SHA")
        _equal(sources.get("worker_source_reference"), plan.get("worker_source"), "Executed worker source path")
        _worker_hardware(worker, declared, global_rank, logical)
        loaded_rccl_digests.add(worker["loaded_rccl"]["library_sha256"])
        visible_inventory.add(_json(sorted(worker["observed_hardware_inventory"]["devices"],
                                           key=lambda item: item["index"])))
        capture = _object(worker.get("stage_capture"), "actual stage capture")
        _need(capture.get("source", {}).get("kind") == "measured" and capture.get("capture_errors") == [],
              "Actual measured stage capture is required")
        for key, header_value in {"session_id": session, "clock_domain": clock, "stage_index": stage,
                           "tp_rank": rank, "tensor_parallel_size": tp, "pipeline_parallel_size": pp,
                           "replica_index": 0, "runtime_version": RUNTIME_VERSION}.items():
            _equal(capture.get(key), header_value, "Stage header " + key)
        metadata = _object(capture.get("metadata"), "prepared metadata")
        _equal(metadata.get("torch_version"), TORCH_VERSION, "Prepared Torch build")
        _equal(metadata.get("hip_build_version"), HIP_VERSION, "Prepared HIP build")
        if world > 1:
            prepared_rccl = _object(metadata.get("rccl_binding"), "Prepared loaded RCCL")
            # Executable segment maps are additional native-origin evidence.
            # The prepared hook and final worker export share the exact loaded
            # library identity, even when only the export records those maps.
            _equal({key: prepared_rccl.get(key) for key in _RCCL_IDENTITY_FIELDS},
                   {key: worker["loaded_rccl"].get(key) for key in _RCCL_IDENTITY_FIELDS},
                   "Prepared/exported actual loaded RCCL")
        config = _object(worker.get("runtime_config"), "actual prepared Runtime config")
        _config(config, tp, pp)
        actual_hash = hashlib.sha256(_json(config).encode()).hexdigest()
        _equal(worker.get("runtime_config_sha256"), actual_hash, "Worker prepared Runtime SHA")
        _equal(metadata.get("runtime_config_sha256"), actual_hash, "Prepared header Runtime SHA")
        _equal(metadata.get("runtime_execution_config"), config, "Prepared Runtime config")
        _equal(worker.get("identity"), capture.get("identity"), "Worker/header identity")
        _equal(worker.get("experiment"), capture.get("experiment"), "Worker/header experiment")
        _equal(worker.get("model_binding"), metadata.get("model_binding"), "Loaded model binding")
        device_meta = _object(metadata.get("device"), "Prepared device metadata")
        _equal(device_meta.get("visible_index"), logical, "Prepared logical device")
        _device(device_meta, declared.infra.devices[global_rank], "Prepared GPU")
        _equal(worker["observed_device"].get("visibility"), device_meta.get("visibility"), "Actual visibility")
        _need(worker["observed_device"].get("visibility", {}).get("HIP_VISIBLE_DEVICES") == plan.get("visible_devices"),
              "Actual visible devices differ from the selected acquisition plan")
        tying = _object(metadata.get("model_binding", {}).get("word_embedding_tying"), "loaded tying proof")
        _need(tying.get("verification_phase") == "prepared" and tying.get("pending_loaded_storage_proof") is False
              and tying.get("effective_tie_word_embeddings") is model.tie_word_embeddings
              and tying.get("architecture_fingerprint_status") == "verified", "Loaded tying/architecture proof differs")
        if pp == 1:
            alias = _object(tying.get("local_alias"), "loaded tied storage alias")
            _need(alias.get("same_parameter") is True and alias.get("same_storage") is True
                  and alias.get("same_tensor_view") is True
                  and alias.get("effective_tie_word_embeddings") is True,
                  "Single-stage vocabulary tying requires actual local Parameter/storage proof")
        else:
            _need(tying.get("verification_scope") == "configured_distinct_pp_endpoint_copies_no_cross_rank_alias_claim",
                  "PP endpoints must retain their distinct local storage scope")
        binding = _object(metadata.get("stage_layer_binding"), "actual loaded stage layers")
        start, end = stage * 28 // pp, (stage + 1) * 28 // pp
        _equal(binding, {"start_layer": start, "end_layer": end, "layer_count": end - start,
                        "method": "loaded_qwen_model_start_end_layer"}, "Loaded stage layer bounds")
        rows = _array(worker.get("records"), "Raw epoch records")
        stage_rows = _array(capture.get("records"), "Prepared records")
        _need(1 <= len(rows) <= 256
              and len(rows) == len(stage_rows), "Raw Workspace and prepared stage rows do not cover the same calls")
        formal = []
        seen_epochs: set[int] = set()
        for index, row in enumerate(rows):
            row = _object(row, "raw epoch")
            _need(row.get("status") == "RAW_STEP_CAPTURED", "Raw Workspace epoch failed")
            _equal(row.get("process_id"), pid, "Epoch process")
            _equal(row.get("stage_record_index"), index, "Epoch prepared record index")
            _equal(row.get("capture_iteration"), index + 1, "Epoch capture sequence")
            _equal(row.get("stage_record"), stage_rows[index], "Epoch prepared record payload")
            actual = _object(stage_rows[index], "prepared stage record")
            _equal(row.get("iteration_id"), actual.get("iteration_id"), "Actual source iteration")
            _equal(actual.get("iteration_id"), index, "Prepared iteration counter")
            epoch = _integer(row.get("epoch_id"), "epoch ID", 1)
            _need(epoch not in seen_epochs, "Duplicate Workspace epoch ID")
            seen_epochs.add(epoch)
            _need(row.get("cohort") in ("startup", "warmup", "formal"), "Unknown Workspace cohort")
            if row["cohort"] != "formal":
                continue
            _equal(row.get("native_epoch_id"), epoch, "Formal native epoch binding")
            for key in ("phase", "query_tokens", "context_lengths"):
                _equal(row.get(key), actual.get(key), "Epoch prepared " + key)
            _need("request_phases" not in actual and "request_phases" not in row,
                  "Homogeneous acquisition cannot contain mixed member roles")
            prompts = _array(actual.get("prompt_lengths"), "Prepared prompt lengths")
            queries = _array(actual.get("query_tokens"), "Prepared queries")
            contexts = _array(actual.get("context_lengths"), "Prepared contexts")
            _need(len(prompts) == 1 and prompts[0] in (16, 32)
                  and len(queries) == 1 and len(contexts) == 1,
                  "Formal prepared shape is outside the fixed single-request grid")
            q, c, prompt = queries[0], contexts[0], prompts[0]
            _integer(q, "prepared query", 1)
            _integer(c, "prepared context")
            _need((actual.get("phase") == "prefill" and q == prompt and c == 0)
                  or (actual.get("phase") == "decode" and q == 1 and prompt <= c <= prompt + 2),
                  "Formal phase/query/context does not describe the natural four-token request")
            counts = [1] if stage == pp - 1 else []
            _equal(actual.get("sampled_token_counts"), counts, "Actual sampling count")
            left = _integer(row.get("start_monotonic_ns"), "Worker epoch start")
            right = _integer(row.get("end_monotonic_ns"), "Worker epoch end")
            prepared_start = _integer(actual.get("start_monotonic_ns"), "prepared start")
            prepared_end = _integer(actual.get("end_monotonic_ns"), "prepared end")
            _need(left <= prepared_start < prepared_end <= right,
                  "Prepared interval is not contained in its actual Worker epoch")
            witness = _object(row.get("return_witness"), "Worker return witness")
            expected_type = "vllm.v1.outputs.ModelRunnerOutput" if stage == pp - 1 else "builtins.NoneType"
            _need(witness.get("type") == expected_type and type(witness.get("gpu_tensor_count")) is int
                  and witness["gpu_tensor_count"] == 0 and witness.get("execute_state_is_none") is True
                  and witness.get("prev_sampled_is_none") is True, "Worker output boundary is unsupported")
            formal.append(row)
        shapes = Counter((row["phase"], row["query_tokens"][0], row["context_lengths"][0]) for row in formal)
        _equal(sorted(shapes.items()), sorted(expected_shapes.items()), "Exact formal phase/context cohort")
        requests: dict[str, list[dict[str, Any]]] = {}
        for row in formal:
            actual = row["stage_record"]
            ids = _array(actual.get("request_ids"), "Actual prepared request IDs")
            _need(len(ids) == 1 and isinstance(ids[0], str) and bool(ids[0]), "One actual request ID is required")
            requests.setdefault(ids[0], []).append(actual)
        _need(len(requests) == 2 * repeats, "Formal cohort does not contain the complete actual requests")
        for request in requests.values():
            prompt = request[0]["prompt_lengths"][0]
            actual_sequence = [(row["phase"], row["query_tokens"], row["context_lengths"],
                                row.get("output_history_lengths")) for row in request]
            expected_sequence = [("prefill", [prompt], [0], [0])]
            expected_sequence += [("decode", [1], [prompt + offset], [offset + 1]) for offset in range(3)]
            _equal(actual_sequence, expected_sequence, "Actual request prefill/decode continuity")
        formal_by_rank[str(global_rank)] = deepcopy(formal)
        placements[global_rank] = {"tensor_parallel_size": tp, "pipeline_parallel_size": pp,
            "stage_index": stage, "tp_rank": rank, "device_index": logical,
            "start_layer": binding["start_layer"], "end_layer": binding["end_layer"]}
        pids.add(pid)
        logical_devices.add(logical)
        coordinates[global_rank] = worker
        raw_captures.append(capture)
    _need(set(coordinates) == set(range(world)) and len(plan_hashes) == 1, "Incomplete worker/plan cohort")
    _need(len(visible_inventory) == 1, "Workers disagree on their actual visible physical GPU inventory")
    _need(len(loaded_rccl_digests) == 1, "Workers have different actual loaded RCCL binaries")
    try:
        normalized = normalize_vllm_capture(declared, raw_captures)
    except SimulationError as exc:
        raise ValueError("Prepared rank/native payload validation failed: " + str(exc)) from exc
    _need(normalized.get("canonical_stage_ready") is True and not normalized.get("quarantined"),
          "Raw prepared captures contain missing, failed or unsupported rank/payload evidence")
    effective = experiment_from_dict(normalized["experiment"])
    return {"status": "SUPPORTED_WORKSPACE_STAGE_DOMAIN", "hardware_validated": False,
        "attribution_complete": False, "experiment": asdict(effective), "identity": profile_identity(effective),
        "placements": [placements[rank] for rank in range(world)], "formal_by_rank": formal_by_rank,
        "worker_process_ids": [coordinates[rank]["process_id"] for rank in range(world)],
        "formal_repetitions": repeats, "formal_full_three_repetitions": repeats == 3,
        "source_session": session, "clock_domain": clock,
        "normalization": normalized,
        "limitations": ["Domain metadata does not establish allocator/native/transport attribution.",
            "Nonfinal CPU Worker return does not release retained PP tensor leases.",
            "File hashes, process exits and actual SDK final files must be checked by the assembler."]}


def validate_stage_workspace_domain(
    plan: dict[str, Any], workers: list[dict[str, Any]], *, experiment: dict[str, Any],
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Apply the pinned pure-data gates, rejecting malformed or missing facts.

    The caller supplies the separately file-verified bound ``experiment``.
    The returned normalization is shape/rank evidence, never Workspace bytes.
    """
    try:
        return _validate_stage_workspace_domain(plan, workers, experiment=experiment, plan_sha256=plan_sha256)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed or incomplete Workspace domain evidence: " + str(exc)) from exc
