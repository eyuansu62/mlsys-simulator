"""Task-local, fail-closed Torch storage audit; never a complete Workspace claim.

Inputs are JSON-like observations only, never Tensor references. Native SDK
coverage, effective Runtime/source binding and fixed-budget admission are the
caller's separate obligations. This module does not infer owners from frames.
"""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from mlsys_sim.workspace_capture import WorkspaceRecorder

_ACTIONS = (
    "alloc",
    "free_requested",
    "free_completed",
    "segment_alloc",
    "segment_free",
    "segment_map",
    "segment_unmap",
    "snapshot",
    "oom",
)
_WIDTHS = {"bfloat16": 2, "float32": 4, "int32": 4, "int64": 8, "bool": 1, "uint8": 1}
_GEOMETRY = {
    "num_layers": 28,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_attention_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}
_RUNTIME = {
    "runner.input_batch.block_table.block_tables[0].block_table.gpu": ([1, 32], "int32"),
    "runner.input_batch.block_table.block_tables[0].slot_mapping.gpu": ([512], "int64"),
    "runner.input_batch.temperature": ([1], "float32"),
    "runner.input_batch.top_p": ([1], "float32"),
    "runner.input_batch.top_k": ([1], "int32"),
    "runner.input_batch.frequency_penalties": ([1], "float32"),
    "runner.input_batch.presence_penalties": ([1], "float32"),
    "runner.input_batch.repetition_penalties": ([1], "float32"),
    "runner.input_batch.sampling_metadata.frequency_penalties": ([1], "float32"),
    "runner.input_batch.sampling_metadata.presence_penalties": ([1], "float32"),
    "runner.input_batch.sampling_metadata.repetition_penalties": ([1], "float32"),
    "runner.input_batch.logitsprocs.argmax_invariant[0].min_p_device": ([1], "float32"),
    "runner.input_batch.logitsprocs.non_argmax_invariant[0].neg_inf_tensor": ([], "float32"),
    "runner.input_ids.gpu": ([512], "int32"),
    "runner.positions": ([512], "int64"),
    "runner.query_start_loc.gpu": ([2], "int32"),
    "runner.seq_lens": ([1], "int32"),
    "runner.num_computed_tokens": ([1], "int32"),
    "runner.prev_num_draft_tokens.gpu": ([1], "int32"),
    "runner.req_indices.gpu": ([512], "int64"),
    "runner.prev_positions.gpu": ([1], "int64"),
    "runner.num_scheduled_tokens.gpu": ([1], "int32"),
    "runner.encoder_seq_lens.gpu": ([1], "int32"),
    "runner.inputs_embeds.gpu": ([512, 2048], "bfloat16"),
    "runner.is_token_ids.gpu": ([512], "bool"),
    "runner.discard_request_mask.gpu": ([1], "bool"),
    "runner.num_decode_draft_tokens.gpu": ([1], "int32"),
    "runner.num_accepted_tokens.gpu": ([1], "int32"),
    "runner.query_pos.gpu": ([512], "int64"),
}


class AuditInputError(ValueError):
    pass


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise AuditInputError(f"{label}: expected bounded integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\0" in value:
        raise AuditInputError(f"{label}: expected nonempty bounded string")
    return value


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise AuditInputError("Inputs must be finite JSON observations, not live objects") from exc


def _issue(result: dict, code: str, **details: Any) -> None:
    entry = {"code": code, **details}
    if entry not in result["issues"]:
        result["issues"].append(entry)


def _model(model: dict) -> None:
    if not isinstance(model, dict):
        raise AuditInputError("model must be asdict(ModelSpec)")
    for name, value in _GEOMETRY.items():
        if type(model.get(name)) is not int or model[name] != value:
            raise AuditInputError(f"Unsupported Qwen3 model geometry: {name}")
    if model.get("name") != "Qwen3-1.7B":
        raise AuditInputError("Only the fixed Qwen3-1.7B domain is supported")
    for name in ("tie_word_embeddings", "qk_norm", "final_norm"):
        if model.get(name) is not True:
            raise AuditInputError(f"The fixed domain requires {name}=true")
    for name in ("weight_dtype", "kv_dtype", "activation_dtype"):
        if model.get(name) != "bfloat16":
            raise AuditInputError(f"The fixed domain requires {name}=bfloat16")
    for name in ("weight_bytes", "kv_bytes", "activation_bytes"):
        if type(model.get(name)) not in (int, float) or model[name] != 2:
            raise AuditInputError(f"The fixed domain requires {name}=2")


def _weight_shapes() -> dict[str, list[int]]:
    result = {"model.embed_tokens.weight": [151936, 2048], "model.norm.weight": [2048]}
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            result[prefix + suffix] = [2048]
        for suffix in ("q_norm.weight", "k_norm.weight"):
            result[prefix + "self_attn." + suffix] = [128]
        for suffix, shape in (
            ("self_attn.qkv_proj.weight", [4096, 2048]),
            ("self_attn.o_proj.weight", [2048, 2048]),
            ("mlp.gate_up_proj.weight", [12288, 2048]),
            ("mlp.down_proj.weight", [2048, 6144]),
        ):
            result[prefix + suffix] = shape
    return result


def _expected_owner(path: str) -> tuple[str, list[int] | None, str | None, str] | None:
    if path.startswith("model.parameter:"):
        name = path.removeprefix("model.parameter:")
        shape = _weight_shapes().get(name)
        if name == "lm_head.weight":
            shape = [151936, 2048]
        return ("weights", shape, "bfloat16", "weight") if shape is not None else None
    if re.fullmatch(r"runner\.kv_caches\[(?:[0-9]|1[0-9]|2[0-7])\]", path):
        return "kv_pool", None, "bfloat16", "kv"
    if re.fullmatch(
        r"model\.buffer:model\.layers\.(?:[0-9]|1[0-9]|2[0-7])\.self_attn\.rotary_emb\.cos_sin_cache", path
    ):
        return "runtime", [40960, 128], "bfloat16", "rope"
    if re.fullmatch(
        r"model\.buffer:model\.layers\.(?:[0-9]|1[0-9]|2[0-7])\.self_attn\.attn\._(?:q|k|v|prob)_scale", path
    ):
        return "runtime", [], "float32", "scale"
    if path == "ATen.getCUDABlasLtWorkspace":
        return "runtime", None, None, "blaslt"
    if path == "runner.input_batch.logitsprocs.argmax_invariant[0].min_p":
        return "runtime", [0], "float32", "min_p_empty_alias"
    if path in _RUNTIME:
        shape, dtype = _RUNTIME[path]
        return "runtime", shape, dtype, "runtime_tensor"
    return None


def _registry(owners: list[dict], result: dict, label: str) -> list[dict]:
    if not isinstance(owners, list) or not 0 < len(owners) <= 20000:
        raise AuditInputError(f"{label}: expected bounded nonempty owner array")
    checked: list[dict[str, Any]] = []
    paths: set[str] = set()
    for row in owners:
        if not isinstance(row, dict):
            raise AuditInputError(f"{label}: owner must be an object")
        path = _text(row.get("path"), "owner.path")
        if path in paths:
            _issue(result, "DUPLICATE_OWNER_PATH", registry=label, path=path)
        paths.add(path)
        category = _text(row.get("category"), "owner.category")
        pointer, size = _int(row.get("pointer"), "owner.pointer"), _int(row.get("bytes"), "owner.bytes")
        if size and not pointer:
            raise AuditInputError(f"{label}: positive owner storage needs a pointer")
        expected = _expected_owner(path)
        current: dict[str, Any] = {
            "path": path,
            "category": category,
            "pointer": pointer,
            "bytes": size,
            "known_path": expected is not None,
            "valid_owner": False,
        }
        if expected and expected[3] == "blaslt":
            current["binding"] = row.get("binding")
            current["valid_owner"] = (
                category == "runtime"
                and size == 83886080
                and row.get("binding") == "verified_exported_getter_no_allocation"
            )
        else:
            shape = row.get("shape")
            if not isinstance(shape, list) or len(shape) > 8:
                raise AuditInputError(f"{label}: Tensor owner requires shape")
            shape = [_int(value, "owner.shape") for value in shape]
            dtype = _text(row.get("dtype"), "owner.dtype").removeprefix("torch.")
            element = _int(row.get("element_size"), "owner.element_size", 1)
            offset = row.get("storage_offset", row.get("tensor_offset"))
            offset = _int(offset, "owner.storage_offset")
            if (
                "storage_offset" in row
                and "tensor_offset" in row
                and row["storage_offset"] != row["tensor_offset"]
            ):
                raise AuditInputError("Conflicting Tensor storage offsets")
            stride = row.get("stride")
            if not isinstance(stride, list) or len(stride) != len(shape):
                raise AuditInputError(f"{label}: Tensor owner requires matching stride")
            stride = [_int(value, "owner.stride") for value in stride]
            if dtype not in _WIDTHS or element != _WIDTHS[dtype]:
                raise AuditInputError(f"{label}: unsupported/mismatched Tensor dtype width")
            logical = math.prod(shape) * element
            extent = (
                0
                if logical == 0
                else (offset + sum((dim - 1) * step for dim, step in zip(shape, stride)) + 1) * element
            )
            if extent > size or (not size and logical):
                raise AuditInputError(f"{label}: Tensor view escapes its full underlying storage")
            current.update(
                shape=shape,
                dtype=dtype,
                element_size=element,
                storage_offset=offset,
                stride=stride,
                logical_bytes=logical,
                view_end_bytes=extent,
            )
            if expected:
                wanted_category, wanted_shape, wanted_dtype, kind = expected
                valid = category == wanted_category and dtype == wanted_dtype
                if kind == "min_p_empty_alias":
                    # Greedy/default-min-p keeps an empty view into the 4-byte
                    # min_p_device storage. It may identify an alias, never a
                    # new allocation owner. The actual owner is checked below.
                    current["alias_only"] = True
                    current["valid_alias"] = (
                        valid and shape == [0] and stride == [1] and offset == 0 and size == 4
                    )
                    valid = False
                elif kind == "kv":
                    per_block = 28 * 2 * 16 * 8 * 128 * 2
                    blocks = size // per_block
                    valid = valid and size % per_block == 0 and blocks > 0 and shape == [blocks, 2, 16, 1024]
                else:
                    valid = valid and shape == wanted_shape and logical == size and offset == 0
                current["valid_owner"] = valid
        if expected and not current["valid_owner"] and not current.get("valid_alias", False):
            _issue(result, "INVALID_KNOWN_OWNER", registry=label, path=path)
        if not size:
            result["diagnostics"]["zero_storage_paths"].append({"registry": label, "path": path})
        checked.append(current)
    min_p_alias = next((row for row in checked if row.get("alias_only")), None)
    if min_p_alias is not None:
        device_path = "runner.input_batch.logitsprocs.argmax_invariant[0].min_p_device"
        min_p_device = next(
            (row for row in checked if row["path"] == device_path and row["valid_owner"]), None
        )
        if (
            min_p_device is None
            or not min_p_alias.get("valid_alias")
            or any(
                min_p_alias[name] != min_p_device[name]
                for name in ("pointer", "bytes", "dtype", "element_size")
            )
        ):
            _issue(result, "MIN_P_EMPTY_ALIAS_BINDING_MISMATCH", registry=label)
    required = {"model.parameter:" + name for name in _weight_shapes()}
    required |= {f"runner.kv_caches[{i}]" for i in range(28)}
    required |= {
        f"model.buffer:model.layers.{i}.self_attn.attn._{scale}_scale"
        for i in range(28)
        for scale in ("q", "k", "v", "prob")
    }
    required.add("ATen.getCUDABlasLtWorkspace")
    missing = required - {row["path"] for row in checked if row["valid_owner"]}
    if missing:
        _issue(result, "MISSING_REQUIRED_OWNERS", registry=label, paths=sorted(missing))
    rope = [row for row in checked if row["valid_owner"] and "rotary_emb.cos_sin_cache" in row["path"]]
    if not rope:
        _issue(result, "MISSING_ROPE_OWNER", registry=label)
    kv = [row for row in checked if row["valid_owner"] and row["path"].startswith("runner.kv_caches[")]
    if kv:
        if len({(row["pointer"], row["bytes"]) for row in kv}) != 1:
            _issue(result, "KV_POOL_IS_NOT_ONE_SHARED_STORAGE", registry=label)
        blocks = kv[0]["shape"][0]
        layer_elements = blocks * 2 * 16 * 1024
        # Both proved layouts tile a complete contiguous physical layer:
        # [layer, blocks, K/V, token, head] or [layer, K/V, blocks, token, head].
        # A bounding interval alone cannot rule out internal overlap or holes.
        layouts = {(32768, 16384, 1024, 1), (16384, blocks * 16384, 1024, 1)}
        observed_layouts = {tuple(row["stride"]) for row in kv}
        if (
            len(observed_layouts) != 1
            or not observed_layouts <= layouts
            or any(
                row["shape"] != [blocks, 2, 16, 1024]
                or row["dtype"] != "bfloat16"
                or row["element_size"] != 2
                for row in kv
            )
        ):
            _issue(result, "KV_LAYOUT_NOT_PROVED", registry=label, strides=sorted(observed_layouts))
        if (
            len(kv) != 28
            or {row["storage_offset"] for row in kv} != {layer * layer_elements for layer in range(28)}
            or kv[0]["bytes"] != 28 * layer_elements * 2
        ):
            _issue(result, "KV_VIEWS_DO_NOT_PARTITION_POOL", registry=label)
    embed = next((row for row in checked if row["path"] == "model.parameter:model.embed_tokens.weight"), None)
    head = next((row for row in checked if row["path"] == "model.parameter:lm_head.weight"), None)
    if head and embed and (head["pointer"], head["bytes"]) != (embed["pointer"], embed["bytes"]):
        _issue(result, "TIED_EMBEDDING_ALIAS_MISMATCH", registry=label)
    return checked


def _blocks(snapshot: dict, device: int, result: dict, label: str) -> dict[int, dict]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("segments"), list):
        raise AuditInputError(f"{label}: missing snapshot segments")
    active: dict[int, dict] = {}
    ranges = []
    for segment in snapshot["segments"]:
        current_device = _int(segment.get("device"), "segment.device")
        if current_device != device:
            _issue(result, "OTHER_DEVICE_SEGMENT", snapshot=label, device=current_device)
            continue
        address = _int(segment.get("address"), "segment.address", 1)
        total = _int(segment.get("total_size"), "segment.total_size", 1)
        pool = segment.get("segment_pool_id")
        if _pool(pool) != (0, 0) or segment.get("is_expandable", False) is not False:
            _issue(result, "UNSUPPORTED_POOL_OR_MAPPING", snapshot=label, address=address)
        ranges.append((address, address + total))
        cursor = address
        if not isinstance(segment.get("blocks"), list):
            raise AuditInputError("Snapshot blocks must be an array")
        for block in segment["blocks"]:
            pointer = _int(block.get("address", cursor), "block.address", 1)
            size = _int(block.get("size"), "block.size", 1)
            if pointer != cursor:
                raise AuditInputError("Snapshot blocks are not contiguous within the segment")
            cursor = pointer + size
            state = block.get("state")
            if state not in ("inactive", "active_allocated", "active_awaiting_free"):
                raise AuditInputError("Unknown snapshot block state")
            if state == "inactive":
                continue
            requested = _int(block.get("requested_size"), "block.requested_size", 1)
            if requested > size or pointer in active:
                raise AuditInputError("Snapshot active block size/address is inconsistent")
            if state == "active_awaiting_free":
                _issue(result, "PENDING_FREE_AT_BOUNDARY", snapshot=label, address=pointer)
            active[pointer] = {
                "address": pointer,
                "requested_bytes": requested,
                "rounded_block_bytes": size,
                "state": state,
                "pool": list(_pool(pool)),
            }
        if cursor != address + total:
            raise AuditInputError("Snapshot segment byte coverage is inconsistent")
    for left, right in zip(sorted(ranges), sorted(ranges)[1:]):
        if left[1] > right[0]:
            raise AuditInputError("Snapshot segments overlap")
    return active


def _pool(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AuditInputError("Pool must be an exact pair of integers")
    return _int(value[0], "pool[0]"), _int(value[1], "pool[1]")


def _categorize(blocks: dict[int, dict], owners: list[dict], result: dict, label: str) -> list[dict]:
    by_pointer: dict[int, list[dict]] = defaultdict(list)
    for owner in owners:
        if owner["bytes"]:
            by_pointer[owner["pointer"]].append(owner)
    for pointer, rows in by_pointer.items():
        if len({row["bytes"] for row in rows}) != 1 or len({row["category"] for row in rows}) != 1:
            _issue(result, "ALIAS_CATEGORY_OR_SIZE_CONFLICT", registry=label, pointer=pointer)
    output = []
    seen_paths: set[str] = set()
    for pointer, block in sorted(blocks.items()):
        matches = [
            owner
            for owner in owners
            if owner["bytes"]
            and pointer <= owner["pointer"]
            and owner["pointer"] + owner["bytes"] <= pointer + block["requested_bytes"]
        ]
        known = [owner for owner in matches if owner["valid_owner"]]
        categories = {owner["category"] for owner in matches}
        if not known or len(categories) != 1:
            _issue(
                result,
                "UNATTRIBUTED_ENTRY_STORAGE" if not known else "ALIAS_CATEGORY_CONFLICT",
                registry=label,
                address=pointer,
            )
        full = [
            owner
            for owner in known
            if owner["pointer"] == pointer and owner["bytes"] == block["requested_bytes"]
        ]
        if not full:
            _issue(result, "OWNER_DOES_NOT_COVER_REQUESTED_STORAGE", registry=label, address=pointer)
        unknown_aliases = [owner["path"] for owner in matches if not owner["known_path"]]
        if unknown_aliases:
            result["diagnostics"]["unknown_alias_paths"].append(
                {"registry": label, "address": pointer, "paths": sorted(unknown_aliases)}
            )
        seen_paths.update(owner["path"] for owner in matches)
        output.append(
            {
                **block,
                "category": next(iter(categories)) if len(categories) == 1 and known else "unknown",
                "reported_fieldpaths": sorted(owner["path"] for owner in matches),
                "verified_owner_paths": sorted(owner["path"] for owner in known),
            }
        )
    unbound = [owner["path"] for owner in owners if owner["bytes"] and owner["path"] not in seen_paths]
    if unbound:
        _issue(result, "OWNER_STORAGE_NOT_IN_ACTIVE_SNAPSHOT", registry=label, paths=sorted(unbound))
    return output


def _history_key(row: dict) -> tuple:
    if not isinstance(row, dict) or row.get("action") not in _ACTIONS:
        raise AuditInputError("Unknown/missing memory-history action")
    action = row["action"]
    address = _int(row.get("addr"), "history.addr")
    size = _int(row.get("size"), "history.size")
    stream = _int(row.get("stream"), "history.stream")
    pool = _pool(row.get("pool_id"))
    metadata = row.get("user_metadata")
    context = row.get("compile_context")
    if not isinstance(metadata, str) or not isinstance(context, str):
        raise AuditInputError("History needs explicit user_metadata and compile_context")
    return action, address, size, stream, pool, metadata, context


def _delta(before: dict, after: dict, device: int, bound: int, supplied: list | None) -> list[dict]:
    traces = []
    for value in (before, after):
        current = value.get("device_traces")
        if not isinstance(current, list) or len(current) <= device or not isinstance(current[device], list):
            raise AuditInputError("Missing full device memory history")
        trace = current[device]
        if len(trace) >= bound:
            raise AuditInputError("HISTORY_BOUND_REACHED: trace may have wrapped")
        traces.append(trace)
    left, right = traces
    if len(right) < len(left) or [_history_key(row) for row in left] != [
        _history_key(row) for row in right[: len(left)]
    ]:
        raise AuditInputError("HISTORY_PREFIX_CHANGED: ordered stable fields differ or trace shrank")
    delta = right[len(left) :]
    if supplied is not None and (
        not isinstance(supplied, list)
        or [_history_key(row) for row in supplied] != [_history_key(row) for row in delta]
    ):
        raise AuditInputError("Supplied memory-history delta differs from stable snapshot suffix")
    return delta


def _native_events(probe: dict, epoch_id: int, device: int, result: dict) -> list[dict]:
    if not isinstance(probe, dict) or probe.get("kind") != "torch_allocator_trace_probe":
        raise AuditInputError("torch_probe must be the native Torch bridge report, not SDK output")
    expected = {
        "schema_version": 1,
        "complete_workspace": False,
        "allocation_owner_classification_performed": False,
        "checkpoint_stable": True,
        "checkpoint_is_prefix_only": True,
        "global_observation_issues_present": False,
        "callback_gpu_calls": False,
        "callback_gil": False,
        "callback_extra_locks": False,
        "abi_cxx11": 1,
    }
    for name, wanted in expected.items():
        if type(probe.get(name)) is not type(wanted) or probe[name] != wanted:
            _issue(result, "INVALID_TORCH_PROBE_HEADER", field=name)
    for name in (
        "dropped_events",
        "unknown_actions",
        "truncated_metadata",
        "clock_failures",
        "thread_id_failures",
        "api_errors",
        "boundary_callbacks",
        "callbacks_inflight_at_write_end",
    ):
        if type(probe.get(name)) is not int or probe[name] != 0:
            _issue(result, "TORCH_PROBE_LOSS_OR_UNKNOWN", field=name)
    events = probe.get("events")
    if not isinstance(events, list) or len(events) > 262144:
        raise AuditInputError("Native Torch events must be bounded")
    if any(
        type(probe.get(name)) is not int or probe[name] != len(events)
        for name in ("retained_events", "attempted_events_at_cutoff", "attempted_events_at_write_end")
    ):
        _issue(result, "TORCH_PROBE_EVENT_COUNT_MISMATCH")
    epochs = probe.get("epochs")
    if not isinstance(epochs, list):
        raise AuditInputError("Native Torch epochs are missing")
    matches = [epoch for epoch in epochs if type(epoch.get("id")) is int and epoch["id"] == epoch_id]
    if len(matches) != 1:
        raise AuditInputError("Need exactly one matching native Torch epoch")
    epoch = matches[0]
    owner_thread = _int(epoch.get("owner_thread_id"), "epoch.owner_thread_id", 1)
    if (
        type(probe.get("install_thread_id")) is not int
        or owner_thread != probe["install_thread_id"]
        or epoch.get("ended") is not True
    ):
        _issue(result, "TORCH_EPOCH_OWNER_OR_CLOSE_MISMATCH")
    for name in (
        "dropped_events",
        "unknown_actions",
        "unowned_allocations",
        "cross_thread_allocations",
        "truncated_metadata",
        "boundary_overlap",
        "boundary_callbacks",
        "clock_failures",
        "thread_id_failures",
        "oom_events",
        "mapping_events",
        "observation_issue_mask",
    ):
        if type(epoch.get(name)) is not int or epoch[name] != 0:
            _issue(result, "TORCH_EPOCH_ISSUES", field=name)
    if epoch.get("observation_issues_present") is not False:
        _issue(result, "TORCH_EPOCH_ISSUES", field="observation_issues_present")
    begin = _int(epoch.get("begin_ns"), "epoch.begin_ns", 1)
    end = _int(epoch.get("end_ns"), "epoch.end_ns", 1)
    first = _int(epoch.get("begin_sequence"), "epoch.begin_sequence")
    last = _int(epoch.get("end_sequence"), "epoch.end_sequence")
    if begin >= end or first > last or last > len(events):
        raise AuditInputError("Invalid native Torch epoch bounds")
    selected = []
    for index, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or type(event.get("sequence")) is not int
            or event["sequence"] != index + 1
        ):
            raise AuditInputError("Native Torch event sequence is incomplete or reordered")
        in_epoch = event.get("epoch_id") == epoch_id
        in_range = first < event["sequence"] <= last
        if in_epoch != in_range:
            _issue(result, "CROSS_BOUNDARY_TORCH_EVENT", sequence=event["sequence"])
        if not in_epoch:
            continue
        action = _int(event.get("action"), "native.action")
        if action >= len(_ACTIONS) or event.get("action_name") != _ACTIONS[action]:
            raise AuditInputError("Unknown/conflicting native Torch action")
        if (
            type(event.get("thread_id")) is not int
            or event["thread_id"] != owner_thread
            or type(event.get("owner_token")) is not int
            or event["owner_token"] != epoch_id
            or type(event.get("flags")) is not int
            or event["flags"] != 0
        ):
            _issue(result, "CROSS_THREAD_OR_UNOWNED_TORCH_EVENT", sequence=event["sequence"])
        stamp = _int(event.get("timestamp_ns"), "native.timestamp_ns", 1)
        if not begin <= stamp <= end or type(event.get("device")) is not int or event["device"] != device:
            _issue(result, "TORCH_EVENT_TIME_OR_DEVICE_MISMATCH", sequence=event["sequence"])
        metadata = event.get("user_metadata")
        metadata_size = _int(event.get("user_metadata_size"), "native.user_metadata_size")
        retained_size = _int(event.get("retained_metadata_size"), "native.retained_metadata_size")
        if not isinstance(metadata, str) or metadata_size != len(metadata.encode("latin1")):
            raise AuditInputError("Native metadata size does not match byte-preserving value")
        if retained_size != metadata_size:
            _issue(result, "NATIVE_METADATA_TRUNCATED", sequence=event["sequence"])
        selected.append(event)
    if type(epoch.get("observed_events")) is not int or epoch["observed_events"] != len(selected):
        _issue(result, "TORCH_EPOCH_EVENT_COUNT_MISMATCH")
    return selected


def _cross_check(delta: list[dict], native: list[dict], result: dict) -> list[tuple[dict, dict]]:
    history = [row for row in delta if _history_key(row)[0] != "snapshot"]
    observed = [row for row in native if row["action_name"] != "snapshot"]
    pairs = []
    if len(history) != len(observed):
        _issue(result, "HISTORY_NATIVE_COUNT_MISMATCH", history=len(history), native=len(observed))
    for index, (row, event) in enumerate(zip(history, observed)):
        key = _history_key(row)
        other = (
            event["action_name"],
            _int(event.get("address"), "native.address"),
            _int(event.get("size_bytes"), "native.size_bytes"),
            _int(event.get("stream"), "native.stream"),
            _pool(event.get("mempool")),
            event["user_metadata"],
        )
        if key[:6] != other:
            _issue(result, "HISTORY_NATIVE_ORDERED_KEY_MISMATCH", index=index)
        if key[6] not in ("N/A", ""):
            _issue(result, "UNSUPPORTED_COMPILE_CONTEXT", index=index)
        if key[4] != (0, 0):
            _issue(result, "UNSUPPORTED_POOL", index=index)
        if key[0] in ("segment_map", "segment_unmap", "oom"):
            _issue(result, "UNSUPPORTED_MAPPING_OR_OOM", action=key[0], index=index)
        pairs.append((row, event))
    return pairs


def audit_torch_workspace(
    *,
    model: dict,
    before_snapshot: dict,
    after_snapshot: dict,
    owners_before: list[dict],
    owners_after: list[dict],
    torch_probe: dict,
    epoch_id: int,
    history_max_entries: int,
    history_delta: list[dict] | None = None,
    device_index: int = 0,
) -> dict:
    """Audit one exact TP1, eager Qwen3 epoch, without claiming native coverage."""
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "torch_workspace_epoch_audit",
        "status": "PARTIAL",
        "torch_side_closed_candidate": False,
        "complete_workspace": False,
        "includes_non_torch_complete": False,
        "byte_basis": "owned_requested_storage_capacity",
        "requested_storage_peak": None,
        "observed_requested_storage_peak": None,
        "fixed_requested_bytes_by_category": {},
        "entry_categories": [],
        "ordered_ledger": [],
        "issues": [],
        "diagnostics": {"unknown_alias_paths": [], "zero_storage_paths": []},
        "remaining_obligations": [
            "Native allocation/resource coverage and explicit fixed-owner backing registration",
            "Effective Runtime/source/rank/shape and returned-output binding",
            "Deduplicated fixed-budget admission",
            "Independent Workspace calibration/validation and physical OOM/headroom evidence",
        ],
    }
    recorder = None
    try:
        _model(model)
        _int(epoch_id, "epoch_id", 1)
        _int(device_index, "device_index")
        _int(history_max_entries, "history_max_entries", 1)
        if history_max_entries > 1000000:
            raise AuditInputError("History bound exceeds supported audit limit")
        _json(
            (model, before_snapshot, after_snapshot, owners_before, owners_after, torch_probe, history_delta)
        )
        result.update(
            epoch_id=epoch_id,
            device_index=device_index,
            model_sha256=hashlib.sha256(_json(model).encode()).hexdigest(),
        )
        before_owners = _registry(owners_before, result, "before")
        after_owners = _registry(owners_after, result, "after")
        before = _blocks(before_snapshot, device_index, result, "before")
        after = _blocks(after_snapshot, device_index, result, "after")
        categories = _categorize(before, before_owners, result, "before")
        _categorize(after, after_owners, result, "after")
        result["entry_categories"] = categories
        fixed: dict[str, int] = defaultdict(int)
        for row in categories:
            fixed[row["category"]] += row["requested_bytes"]
        result["fixed_requested_bytes_by_category"] = dict(sorted(fixed.items()))
        if _json(sorted(before_owners, key=lambda row: row["path"])) != _json(
            sorted(after_owners, key=lambda row: row["path"])
        ):
            _issue(result, "FIXED_OWNER_BINDINGS_CHANGED")
        if before != after:
            _issue(
                result,
                "FIXED_ACTIVE_STORAGE_CHANGED",
                new_addresses=sorted(after.keys() - before.keys()),
                missing_addresses=sorted(before.keys() - after.keys()),
            )
        delta = _delta(before_snapshot, after_snapshot, device_index, history_max_entries, history_delta)
        native = _native_events(torch_probe, epoch_id, device_index, result)
        pairs = _cross_check(delta, native, result)
        result["history_cross_check"] = {
            "before_entries": len(before_snapshot["device_traces"][device_index]),
            "after_entries": len(after_snapshot["device_traces"][device_index]),
            "delta_entries": len(delta),
            "native_epoch_events": len(native),
            "cross_checked_non_snapshot_events": len(pairs),
            "identity_fields_exclude_frames_and_time_us": True,
        }
        recorder = WorkspaceRecorder(
            1,
            epoch_id=f"torch-epoch-{epoch_id}",
            device_indices=[device_index],
            max_events=100000,
            max_event_bytes=4 * 1024 * 1024,
        )
        active: dict[int, dict] = {}
        generations: Counter = Counter()
        for row, event in pairs:
            action, address, size, stream, pool, metadata, _ = _history_key(row)
            if action in ("segment_alloc", "segment_free"):
                continue
            if action not in ("alloc", "free_requested", "free_completed"):
                _issue(result, "UNSUPPORTED_WORKSPACE_ACTION", action=action)
                continue
            if any(
                address < pointer + block["rounded_block_bytes"] and pointer < address + max(size, 1)
                for pointer, block in before.items()
            ):
                _issue(result, "FIXED_STORAGE_LIFETIME_MUTATED", action=action, address=address)
                continue
            if not size or not address:
                raise AuditInputError("Dynamic storage event needs positive requested size/address")
            if action == "alloc":
                if address in active:
                    raise AuditInputError("ALLOC overlaps a live generation")
                for pointer, lease in active.items():
                    if address < pointer + lease["size"] and pointer < address + size:
                        raise AuditInputError("Dynamic allocation address ranges overlap")
                generations[address] += 1
                allocation = f"e{epoch_id}-s{event['sequence']}"
                storage = f"e{epoch_id}-p{address:x}-g{generations[address]}"
                active[address] = {
                    "size": size,
                    "stream": stream,
                    "pool": pool,
                    "allocation": allocation,
                    "storage": storage,
                    "generation": generations[address],
                    "requested": False,
                }
                recorder.allocate(0, allocation, storage, size)
            else:
                if address not in active:
                    raise AuditInputError("Free has no matching new storage generation")
                lease = active[address]
                if (size, stream, pool) != (lease["size"], lease["stream"], lease["pool"]):
                    raise AuditInputError("Free requested-size/stream/pool differs from its allocation")
                if action == "free_requested":
                    if lease["requested"]:
                        raise AuditInputError("Duplicate free request")
                    lease["requested"] = True
                    recorder.request_free(0, lease["allocation"])
                else:
                    if not lease["requested"]:
                        raise AuditInputError("Completed free lacks a prior request")
                    recorder.release_completed(0, lease["allocation"])
            lease = active[address]
            result["ordered_ledger"].append(
                {
                    "action": {
                        "alloc": "allocate",
                        "free_requested": "free_requested",
                        "free_completed": "release_completed",
                    }[action],
                    "original_action": action,
                    "native_sequence": event["sequence"],
                    "thread_id": event["thread_id"],
                    "owner_token": event["owner_token"],
                    "allocation_id": lease["allocation"],
                    "storage_id": lease["storage"],
                    "generation": lease["generation"],
                    "address": address,
                    "requested_bytes": size,
                    "stream": stream,
                    "pool": list(pool),
                    "user_metadata": metadata,
                }
            )
            if action == "free_completed":
                del active[address]
        if active:
            _issue(result, "NEW_PERSISTENT_OR_PENDING_STORAGE", addresses=sorted(active))
        else:
            recorder.complete_rank(0)
            recorder.seal()
        capture = recorder.export_capture()
        result["recorder_capture"] = capture
        result["observed_requested_storage_peak"] = capture["peak_bytes_per_rank"][0]["peak_bytes"]
        result["new_storage_generations"] = sum(generations.values())
        if not result["issues"] and capture["status"] == "COMPLETED":
            result.update(
                status="CLOSED_CANDIDATE",
                torch_side_closed_candidate=True,
                requested_storage_peak=result["observed_requested_storage_peak"],
            )
    except (
        AuditInputError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        UnicodeError,
        RecursionError,
    ) as exc:
        _issue(result, "INVALID_OR_INCOMPLETE_TORCH_EVIDENCE", message=str(exc))
        if recorder is not None:
            recorder.abort(str(exc))
            result["recorder_capture"] = recorder.export_capture()
    return result
