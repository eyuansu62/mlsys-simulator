"""Task-local native backing audit; never a complete Workspace/hardware verdict.

Only structured SDK callback evidence, exact allocator segment identities and a
caller-supplied controlled-constructor binding are used. No frame/size heuristics.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any

from analyze_native_probe import analyze, MUTATOR_MASK

MAX_EVENTS = 4194304
MAX_ISSUES = 4096
ALLOCATE_APIS = {"hsa_memory_allocate", "hsa_amd_memory_pool_allocate"}
FREE_APIS = {"hsa_memory_free", "hsa_amd_memory_pool_free"}
NORMAL_ALLOCATE = {"MEMORY_ALLOCATION_ALLOCATE"}
NORMAL_FREE = {"MEMORY_ALLOCATION_FREE"}
SIGNALS = {"hsa_signal_create", "hsa_signal_destroy"}
EXPECTED_CTOR_ENTRYPOINT = "ATen.getCUDABlasLtWorkspace"
RCCL_FORMAL_APIS = {
    "ncclAllReduce", "ncclAllGather", "ncclBroadcast", "ncclReduce", "ncclReduceScatter",
    "ncclSend", "ncclRecv", "ncclGroupStart", "ncclGroupEnd",
}
RCCL_INIT_APIS = {"ncclCommInitRank", "ncclCommInitRankConfig"}
RCCL_COMM_APIS = RCCL_FORMAL_APIS - {"ncclGroupStart", "ncclGroupEnd"}
EXPECTED_COMM_CTOR = "vllm.distributed.device_communicators.pynccl.PyNcclCommunicator.__init__"
PG_ENTRYPOINTS = {
    "torch.distributed.distributed_c10d.all_gather_into_tensor",
    "torch.distributed.distributed_c10d.isend", "torch.distributed.distributed_c10d.irecv",
}
SYNC_RESOURCE_APIS = {"hipEventCreate", "hipEventCreateWithFlags", "hipEventDestroy",
                      "hsa_signal_create", "hsa_signal_destroy"}
# Exact public API effects, not substring overrides of unknown memory surfaces.
# Raw conservative probe flags remain unchanged in the input and audit records.
READ_ONLY_HSA_APIS: dict[str, dict[str, Any]] = {
    "hsa_amd_agent_iterate_memory_pools": {
        "flags": {0, 16, 512},
        "typed_status_required": False,
        "effect": "pool_enumeration_surface_only_nested_calls_still_audited",
    },
    "hsa_amd_memory_get_preferred_copy_engine": {
        "flags": {0, 128, 512},
        "typed_status_required": True,
        "effect": "preferred_sdma_engine_mask_query",
    },
}
HOST_UNPIN_APIS = {"hsa_amd_memory_unlock"}
HOST_UNPIN_COPY_PARENTS = {"hipMemcpyWithStream", "hipMemcpyAsync"}
COMM_STARTUP_CONTROL_APIS = {"hipHostRegister", "hipMemPoolCreate", "hipMemPoolSetAttribute"}
COMM_TEARDOWN_CONTROL_APIS = {"hipHostUnregister", "hipMemPoolDestroy"}
HSA_EFFECT_SOURCE = "https://github.com/ROCm/rocm-systems/blob/fe3963abc496ae779963b9a2534c140b56510fe9/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h"


def _integer(value, label, minimum=0, maximum=2**64 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be a bounded integer >= {minimum}")
    return value


class Problems:
    def __init__(self):
        self.rows = []
        self.counts: Counter[str] = Counter()

    def add(self, code, **detail):
        self.counts[code] += 1
        if len(self.rows) < MAX_ISSUES:
            self.rows.append({"code": code, **detail})

    def __bool__(self):
        return bool(self.counts)


def _base_result():
    return {
        "schema_version": 1,
        "kind": "native_backing_owner_audit",
        "method": "successful_hsa_backing_replay_v1",
        "status": "PARTIAL",
        "complete_workspace": False,
        "hardware_validated": False,
        "native_fixed_runtime_bytes": None,
        "unknown_allocations": [],
        "epochs": [],
        "scope": "Native GPU backing identities at explicitly supplied synchronous formal epoch boundaries",
        "limitations": [
            "Torch lease ownership, allocator-history completeness, driver scope and fixed-budget validation are separate proofs.",
            "Controlled constructor purpose is supplied by the executed-driver binding, not inferred from an allocation's size or frames.",
            "Private scratch reserve is the sum of observed whole-pool peaks by agent/queue, not a physical alias-deduplicated total.",
            "No memory peak-minus-baseline calculation or automatic Workspace/hardware acceptance is performed.",
        ],
    }


def _finish(result, problems):
    result["issues"] = problems.rows
    result["problem_counts"] = dict(sorted(problems.counts.items()))
    result["issues_truncated"] = sum(problems.counts.values()) > len(problems.rows)
    result["status"] = "PARTIAL" if problems else "NATIVE_BACKING_CHECKS_PASSED"
    if problems:
        result["native_fixed_runtime_bytes"] = None
        if "native_fixed_communication_bytes" in result:
            result["native_fixed_communication_bytes"] = None
    return result


def _pg_scope_valid(scope, constructor, placement, epoch):
    if not isinstance(scope, dict) or not isinstance(epoch, dict):
        return False
    source = scope.get("source_binding", {})
    pg = scope.get("pg", {})
    validation = scope.get("payload_validation", {})
    if (scope.get("kind") != "torch_process_group_prewarm" or scope.get("status") != "COMPLETED"
            or scope.get("scope") != "independent_startup_communication_before_model_load"
            or scope.get("epoch_id") != constructor["epoch_id"]
            or scope.get("epoch_label") != constructor["epoch_label"]
            or scope.get("entrypoint") != constructor["entrypoint"]
            or scope.get("process_id") != placement["process_id"]
            or scope.get("logical_device_index") != placement["logical_device_index"]
            or scope.get("comm_only") is not True or scope.get("preallocated_before_epoch") is not True
            or scope.get("device_sync_completed") is not True
            or type(scope.get("preallocation_completed_monotonic_raw_ns")) is not int
            or not 0 < scope["preallocation_completed_monotonic_raw_ns"] < epoch["begin_ns"]):
        return False
    if (not isinstance(source, dict) or source.get("qualified_name") != constructor["entrypoint"]
            or not isinstance(source.get("source_reference"), str) or not source["source_reference"].startswith("/")
            or any(not isinstance(source.get(k), str) or not re.fullmatch(r"[0-9a-f]{64}", source[k])
                   for k in ("source_sha256", "file_sha256"))):
        return False
    if (not isinstance(pg, dict) or pg.get("backend") != "nccl" or pg.get("global_ranks") != [0, 1]
            or pg.get("world_size") != constructor["world_size"] or pg.get("rank_in_group") != constructor["rank"]
            or type(pg.get("object_id")) is not int or pg["object_id"] <= 0
            or not isinstance(pg.get("group_name"), str) or not pg["group_name"]):
        return False
    inputs, outputs = scope.get("input_buffers"), scope.get("output_buffers")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return False
    control = scope.get("control")
    if control == "tp_allgather":
        if constructor["entrypoint"] != "torch.distributed.distributed_c10d.all_gather_into_tensor":
            return False
        shapes = ([[75968]], [[151936]])
    elif control == "pp_p2p":
        expected = "torch.distributed.distributed_c10d.isend" if constructor["rank"] == 0 else "torch.distributed.distributed_c10d.irecv"
        if constructor["entrypoint"] != expected:
            return False
        shapes = ([[512, 2048], [512, 2048]], []) if constructor["rank"] == 0 else ([], [[512, 2048], [512, 2048]])
    else:
        return False
    for buffers, required in zip((inputs, outputs), shapes):
        if len(buffers) != len(required):
            return False
        for descriptor, shape in zip(buffers, required):
            count = 1
            for extent in shape:
                count *= extent
            if (not isinstance(descriptor, dict) or descriptor.get("shape") != shape
                    or descriptor.get("dtype") != "torch.bfloat16" or descriptor.get("element_size") != 2
                    or descriptor.get("device_index") != placement["logical_device_index"]
                    or descriptor.get("bytes") != count * 2 or descriptor.get("tensor_offset") != 0
                    or type(descriptor.get("pointer")) is not int or descriptor["pointer"] <= 0
                    or descriptor.get("tensor_data_ptr") != descriptor["pointer"]):
                return False
    checks = validation.get("checks") if isinstance(validation, dict) else None
    if (validation.get("success") is not True or validation.get("method") != "full_CPU_copy_compare_after_native_epoch"
            or not isinstance(checks, list) or len(checks) != 2):
        return False
    expected_count = 75968 if control == "tp_allgather" else 512 * 2048
    expected_values = (1.0, 2.0) if control == "tp_allgather" else (17.0, 19.0)
    return all(isinstance(c, dict) and c.get("expected_count") == c.get("observed_count") == expected_count
               and c.get("mismatch_count") == 0 and c.get("expected_value") == value
               and c.get("expected_checksum") == c.get("observed_checksum") == value * expected_count
               for c, value in zip(checks, expected_values))


def _communication_binding(binding, epochs, formal_ids, lt_id, mapping, problems, *, pid=None):
    """Validate the opt-in contract; no constructor purpose is inferred here."""
    if binding is None:
        return None
    initial_count = sum(problems.counts.values())
    if not isinstance(binding, dict):
        raise ValueError("communication_binding must be an object")
    if (type(binding.get("schema_version")) is not int or binding["schema_version"] != 1
            or binding.get("method") != "controlled_rccl_constructor_v1"):
        problems.add("unsupported_communication_binding_method")
    for field in ("driver_source_sha256", "constructor_source_sha256"):
        if not isinstance(binding.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", binding[field]):
            problems.add("communication_source_digest_missing", field=field)
    for field in ("driver_source_reference", "constructor_source_reference"):
        if not isinstance(binding.get(field), str) or not 0 < len(binding[field].strip()) <= 4096:
            problems.add("communication_source_reference_missing", field=field)
    if binding.get("runtime_version") != "0.29.1.dev0+g98dff2a81.d20261002":
        problems.add("unsupported_communication_runtime_version")
    loaded = binding.get("loaded_rccl", {})
    if not isinstance(loaded, dict):
        raise ValueError("loaded_rccl must preserve the actual loaded-library probe")
    if (loaded.get("verified") is not True or loaded.get("runtime_version") != "2.30.7"
            or type(loaded.get("version_code")) is not int or loaded["version_code"] != 23007
            or loaded.get("verification_scope") != "local_proc_maps_loaded_library_ncclGetVersion"
            or not isinstance(loaded.get("path"), str) or not loaded["path"].startswith("/")):
        problems.add("actual_loaded_communication_library_not_proven")
    if not isinstance(loaded.get("library_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", loaded["library_sha256"]
    ):
        problems.add("actual_loaded_communication_library_digest_missing")
    rank = binding.get("rank")
    if not isinstance(rank, dict):
        raise ValueError("communication_binding requires actual process/rank/device facts")
    fields = ("process_id", "global_rank", "tp_rank", "stage_index", "tensor_parallel_size",
              "pipeline_parallel_size", "world_size", "logical_device_index", "agent_handle")
    for field in fields:
        _integer(rank[field], "communication rank." + field)
    tp, pp, world = rank["tensor_parallel_size"], rank["pipeline_parallel_size"], rank["world_size"]
    if ((tp, pp) not in ((2, 1), (1, 2)) or world != tp * pp
            or not 0 <= rank["tp_rank"] < tp or not 0 <= rank["stage_index"] < pp
            or rank["global_rank"] != rank["stage_index"] * tp + rank["tp_rank"]
            or rank["process_id"] != pid or rank["process_id"] <= 0
            or mapping != {rank["logical_device_index"]: rank["agent_handle"]}):
        problems.add("communication_rank_process_or_actual_device_mapping_mismatch")
    init = binding.get("initialization_epochs")
    if not isinstance(init, list) or not 1 <= len(init) <= 16:
        raise ValueError("one to sixteen controlled communication initialization epochs are required")
    constructors, handles = {}, {}
    for row in init:
        if not isinstance(row, dict):
            raise ValueError("communication initialization epoch must be an object")
        identifier = _integer(row["epoch_id"], "communication initialization epoch", 1)
        handle = _integer(row["communicator_handle"], "actual initialized communicator handle", 1)
        size = _integer(row["world_size"], "actual communicator world size", 1)
        local_rank = _integer(row["rank"], "actual communicator local rank")
        epoch = epochs.get(identifier)
        known_source = row.get("entrypoint") == EXPECTED_COMM_CTOR
        if row.get("kind") == "torch_process_group_prewarm":
            known_source = row.get("entrypoint") in PG_ENTRYPOINTS and _pg_scope_valid(row.get("pg_scope"), row, rank, epoch)
        if identifier in constructors or handle in handles:
            problems.add("duplicate_communication_constructor_epoch_or_handle")
        if (identifier in formal_ids or identifier == lt_id or row.get("comm_only") is not True
                or not known_source or not isinstance(row.get("epoch_label"), str)
                or not epoch or epoch.get("label") != row["epoch_label"]
                or epoch.get("ended") is not True or size != 2 or not 0 <= local_rank < size):
            problems.add("controlled_communication_constructor_binding_missing_or_mismatched", epoch_id=identifier)
        elif (not epoch["begin_ns"] < epoch.get("successful_device_sync_exit_ns", 0) < epoch["end_ns"]
                or any(epoch["end_ns"] >= epochs[i]["begin_ns"] for i in formal_ids if i in epochs)):
            problems.add("communication_constructor_not_synchronously_complete_before_formal", epoch_id=identifier)
        constructors[identifier] = deepcopy(row)
        handles[handle] = deepcopy(row)
    registered = binding.get("registered_backings")
    if not isinstance(registered, list) or len(registered) > 65536:
        raise ValueError("registered communication backing table is required and bounded")
    registrations = {}
    expected_fields = {"address", "bytes", "agent_handle", "allocation_cid", "born_epoch", "init_cid"}
    for row in registered:
        proof = row.get("proof_kind", "init-ancestor") if isinstance(row, dict) else None
        extra = ({"proof_kind", "first_use_cid", "launch_cid"} if proof == "ctor-known-first-use"
                 else {"proof_kind", "thread_origin_id"} if proof == "rccl-born-proxy-thread" else set())
        if not isinstance(row, dict) or proof not in {"init-ancestor", "ctor-known-first-use", "rccl-born-proxy-thread"} or set(row) != expected_fields | extra:
            raise ValueError("communication backing requires exact address/size/agent/allocation/init identities")
        for field in expected_fields:
            _integer(row[field], "registered communication backing." + field, 0 if field == "born_epoch" and proof == "rccl-born-proxy-thread" else 1)
        for field in extra - {"proof_kind"}:
            _integer(row[field], "registered communication proof." + field, 1)
        key = (row["agent_handle"], row["address"], row["bytes"])
        if key in registrations or proof != "rccl-born-proxy-thread" and row["born_epoch"] not in constructors:
            problems.add("duplicate_or_unbound_communication_backing_registration", address=row["address"])
        if row["agent_handle"] != rank["agent_handle"] or row["address"] + row["bytes"] > 2**64:
            problems.add("communication_backing_registration_device_or_range_mismatch", address=row["address"])
        registrations[key] = deepcopy(row)
    if sum(problems.counts.values()) != initial_count:
        return None
    return {"constructors": constructors, "handles": handles, "registrations": registrations,
            "rank": deepcopy(rank), "init_calls": {}, "first_uses": [], "thread_origins": {},
            "loaded_rccl": deepcopy(loaded), "pg_calls": defaultdict(list),
            "pg_group_calls": defaultdict(list), "verified_pg_epochs": set(), "pg_group_ends": [],
            "stdthread_control": deepcopy(binding.get("stdthread_control")),
            "natural_shutdown_evidence": deepcopy(binding.get("natural_shutdown_evidence")),
            "host_signal_evidence": deepcopy(binding.get("host_signal_evidence"))}


def _ancestor_call(child, ancestor_cid, call_pairs):
    """Follow only unique, contained, same-thread actual API parent intervals."""
    seen = set()
    chain: list[tuple[dict, dict]] = []
    current = child
    while current[0].get("ancestor"):
        cid = current[0]["ancestor"]
        if cid in seen or len(chain) >= 64:
            return None
        seen.add(cid)
        candidates = call_pairs.get(cid, [])
        if len(candidates) != 1:
            return None
        parent = candidates[0]
        a, b = current
        pa, pb = parent
        if (a["ancestor"] != b["ancestor"] or pa["tid"] != pb["tid"] or pa["tid"] != a["tid"]
                or pa["epoch"] != pb["epoch"] or pa["epoch"] != a["epoch"] or a["epoch"] != b["epoch"]
                or not pa["seq"] < a["seq"] < b["seq"] < pb["seq"]
                or not pa["ns"] <= a["ns"] <= b["ns"] <= pb["ns"]):
            return None
        chain.append(parent)
        if cid == ancestor_cid:
            return chain
        current = parent
    return None


def _communication_calls(call_pairs, communication, epochs, formal_ids, opmap, problems):
    accepted = {}
    formal_calls = []
    for candidates in call_pairs.values():
        if len(candidates) != 1:
            continue
        a, b = candidates[0]
        if a["domain"] != 8:
            continue
        name = opmap[(a["domain"], a["op"])]
        if name not in RCCL_INIT_APIS | RCCL_FORMAL_APIS:
            continue
        pair_valid = (a["epoch"] == b["epoch"] and a["flags"] == b["flags"] == 64
                      and b["status_type"] == 3 and b["status"] == 0)
        pa, pb = a.get("raw_rccl_payload"), b.get("raw_rccl_payload")
        pair_valid = pair_valid and isinstance(pa, dict) and isinstance(pb, dict)
        if name in RCCL_INIT_APIS and a["epoch"] in communication["constructors"]:
            constructor = communication["constructors"][a["epoch"]]
            pair_valid = bool(pair_valid and pa.get("kind") == pb.get("kind") == "init"
                and pa.get("comm_handle") is None and pb.get("comm_handle") == constructor["communicator_handle"]
                and pa.get("comm_output_address") == pb.get("comm_output_address")
                and type(pb.get("comm_output_address")) is int and pb["comm_output_address"] > 0
                and pa.get("world_size") == pb.get("world_size") == constructor["world_size"]
                and pa.get("rank") == pb.get("rank") == constructor["rank"])
            if pair_valid:
                if a["epoch"] in communication["init_calls"]:
                    problems.add("communication_constructor_init_call_not_unique", epoch_id=a["epoch"])
                communication["init_calls"][a["epoch"]] = (a, b)
                accepted[(a["domain"], a["op"], a["cid"], a["tid"])] = (name, a, b)
            else:
                problems.add("communication_constructor_actual_init_payload_mismatch", cid=a["cid"])
        elif (name == "ncclAllReduce" and a["epoch"] in communication["constructors"]
              and communication["constructors"][a["epoch"]].get("kind") != "torch_process_group_prewarm"):
            constructor = communication["constructors"][a["epoch"]]
            initialization = communication["init_calls"].get(a["epoch"])
            if (pair_valid and pa == pb and pb.get("kind") == "collective"
                    and pb.get("comm_handle") == constructor["communicator_handle"]
                    and pb.get("count") == 1 and pb.get("datatype") == 7
                    and initialization is not None and initialization[1]["seq"] < a["seq"]):
                communication["first_uses"].append((a, b))
            else:
                problems.add("communication_constructor_first_use_not_exact_known_control", cid=a["cid"])
        elif (name in {"ncclGroupStart", "ncclGroupEnd"} and a["epoch"] in communication["constructors"]
              and communication["constructors"][a["epoch"]].get("kind") == "torch_process_group_prewarm"):
            if pair_valid and pa == pb and pb.get("kind") == "group":
                communication["pg_group_calls"][a["epoch"]].append((name, a, b))
            else:
                problems.add("process_group_prewarm_group_not_successful", cid=a["cid"])
        elif (name in RCCL_COMM_APIS and a["epoch"] in communication["constructors"]
              and communication["constructors"][a["epoch"]].get("kind") == "torch_process_group_prewarm"):
            constructor = communication["constructors"][a["epoch"]]
            initialization = communication["init_calls"].get(a["epoch"])
            if (pair_valid and pa == pb and pb.get("kind") == "collective"
                    and pb.get("comm_handle") == constructor["communicator_handle"]
                    and initialization is not None and initialization[1]["seq"] < a["seq"]):
                communication["pg_calls"][a["epoch"]].append((name, a, b))
            else:
                problems.add("process_group_prewarm_call_not_bound_to_actual_init", cid=a["cid"])
        elif name in RCCL_FORMAL_APIS and a["epoch"] in formal_ids:
            pair_valid = bool(pair_valid and pa == pb)
            if name in RCCL_COMM_APIS:
                handle = pb.get("comm_handle") if isinstance(pb, dict) else None
                constructor = communication["handles"].get(handle) if type(handle) is int else None
                pair_valid = bool(pair_valid and constructor and pb.get("kind") == "collective"
                    and type(pb.get("stream_handle")) is int and 0 <= pb["stream_handle"] < 2**64
                    and type(pb.get("count")) is int and 0 <= pb["count"] < 2**64
                    and type(pb.get("datatype")) is int and 0 <= pb["datatype"] <= 9)
                if name in {"ncclSend", "ncclRecv"}:
                    pair_valid = bool(pair_valid and type(pb.get("peer")) is int
                                      and constructor and 0 <= pb["peer"] < constructor["world_size"])
                elif name in {"ncclBroadcast", "ncclReduce"}:
                    pair_valid = bool(pair_valid and type(pb.get("root")) is int
                                      and constructor and 0 <= pb["root"] < constructor["world_size"])
            else:
                pair_valid = bool(pair_valid and pb.get("kind") == "group")
            if pair_valid:
                key = (a["domain"], a["op"], a["cid"], a["tid"])
                accepted[key] = (name, a, b)
                formal_calls.append((name, a, b))
            else:
                problems.add("formal_communication_call_not_bound_or_successful", cid=a["cid"], operation=name)
    for identifier in communication["constructors"]:
        if identifier not in communication["init_calls"]:
            problems.add("communication_constructor_actual_init_call_missing", epoch_id=identifier)
        constructor = communication["constructors"][identifier]
        if constructor.get("kind") != "torch_process_group_prewarm":
            continue
        scope = constructor["pg_scope"]
        calls = communication["pg_calls"].get(identifier, [])
        if scope["control"] == "tp_allgather":
            good = (len(calls) == 1 and calls[0][0] == "ncclAllGather"
                and calls[0][2]["raw_rccl_payload"].get("count") == 75968
                and calls[0][2]["raw_rccl_payload"].get("datatype") == 9
                and calls[0][2]["raw_rccl_payload"].get("send_buffer_address") == scope["input_buffers"][0]["tensor_data_ptr"]
                and calls[0][2]["raw_rccl_payload"].get("receive_buffer_address") == scope["output_buffers"][0]["tensor_data_ptr"])
        else:
            sending = constructor["rank"] == 0
            buffers = scope["input_buffers"] if sending else scope["output_buffers"]
            pointer_field = "send_buffer_address" if sending else "receive_buffer_address"
            good = (len(calls) == 2 and all(call[0] == ("ncclSend" if sending else "ncclRecv")
                and call[2]["raw_rccl_payload"].get("count") == 512 * 2048
                and call[2]["raw_rccl_payload"].get("datatype") == 9
                and call[2]["raw_rccl_payload"].get("peer") == 1 - constructor["rank"]
                and call[2]["raw_rccl_payload"].get(pointer_field) == descriptor["tensor_data_ptr"]
                for call, descriptor in zip(calls, buffers)))
        if not good:
            problems.add("process_group_prewarm_actual_buffers_or_collective_parameters_mismatch", epoch_id=identifier)
        else:
            communication["verified_pg_epochs"].add(identifier)
            groups = []
            for name, a, b in sorted(calls + communication["pg_group_calls"].get(identifier, []), key=lambda c: c[1]["seq"]):
                if name == "ncclGroupStart":
                    groups.append([a, b, 0])
                elif name == "ncclGroupEnd":
                    if groups:
                        _, start_exit, members = groups.pop()
                        if members and start_exit["seq"] < a["seq"]:
                            communication["pg_group_ends"].append((a, b))
                else:
                    for group in groups:
                        group[2] += 1
    stacks: dict[tuple[int, int], list[list[Any]]] = defaultdict(list)
    for name, a, b in sorted(formal_calls, key=lambda call: call[1]["seq"]):
        group_key = (a["epoch"], a["tid"])
        if name == "ncclGroupStart":
            if len(stacks[group_key]) >= 16:
                problems.add("formal_communication_group_nesting_exceeds_bound", cid=a["cid"])
            stacks[group_key].append([a, b, 0])
        elif name == "ncclGroupEnd":
            if not stacks[group_key]:
                problems.add("formal_communication_group_end_without_start", cid=a["cid"])
            else:
                start, end, members = stacks[group_key].pop()
                if not end["seq"] < a["seq"] or not members:
                    problems.add("formal_communication_group_order_or_members_missing", cid=a["cid"])
        else:
            for group in stacks[group_key]:
                group[2] += 1
    if any(stacks.values()):
        problems.add("formal_communication_group_not_closed")
    return accepted


def _module_ranges(library):
    if not isinstance(library, dict) or not isinstance(library.get("path"), str):
        return None, []
    mappings = library.get("mapped_segments", [])
    if not isinstance(mappings, list):
        raise ValueError("actual loaded library mappings must be an array")
    segments = []
    for m in mappings:
        if (not isinstance(m, dict) or m.get("path") != library["path"]
                or not isinstance(m.get("permissions"), str)):
            continue
        start = _integer(m["start"], "loaded mapping start", 1)
        end = _integer(m["end"], "loaded mapping end", start + 1)
        offset = _integer(m["offset"], "loaded mapping offset")
        if offset <= start:
            segments.append((start, end, offset, m["permissions"]))
    return min((start - offset for start, _, offset, _ in segments), default=None), segments


def _stdthread_dispatch_proven(origin, communication, pid, rccl_base, rccl_segments):
    proof = communication.get("stdthread_control")
    if not isinstance(proof, dict):
        return False
    control, header, runtime = proof.get("control"), proof.get("header"), proof.get("runtime_library")
    if not isinstance(control, dict) or not isinstance(header, dict) or not isinstance(runtime, dict):
        return False
    if (control.get("schema_version") != 1 or control.get("method") != "observed_stdthread_state_dispatch_v1"
            or control.get("pid") != pid or control.get("attempted") is not True
            or control.get("completed") is not True or control.get("trampoline_observed") is not True
            or control.get("control_returncode") != 0 or control.get("glibcxx_release") != 13
            or control.get("member_pointer_size") != 16 or control.get("state_size") != 8
            or header.get("path") != control.get("header_reference")
            or header.get("sha256") != control.get("header_sha256")
            or header.get("sha256") != "bfaeaa41bf061e826c96cbee7ee3c453250918b7c3e7cc96b01693ee31751233"
            or runtime.get("path") != control.get("module_path")
            or not isinstance(runtime.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", runtime["sha256"])):
        return False
    base, segments = _module_ranges(runtime)
    start = control.get("start_address")
    if (type(start) is not int or base is None or control.get("module_base") != base
            or control.get("module_offset") != start - base
            or not any(a <= start < b and "x" in permissions for a, b, _, permissions in segments)
            or origin.get("start_address") != start or origin.get("module_path") != runtime["path"]
            or origin.get("module_base") != base or origin.get("module_offset") != start - base
            or origin.get("stdthread_trampoline_matched") is not True
            or origin.get("state_dispatch_observed") is not True
            or type(control.get("run_member_offset")) is not int
            or not 0 <= control["run_member_offset"] <= 256 or control["run_member_offset"] % 8
            or origin.get("run_member_offset") != control["run_member_offset"]):
        return False
    library = communication["loaded_rccl"]["path"]
    for field, path_field, base_field, permission in (
        ("state_vtable", "vtable_module", "vtable_module_base", "r"),
        ("state_typeinfo", "typeinfo_module", "typeinfo_module_base", "r"),
        ("state_dispatch", "dispatch_module", "dispatch_module_base", "x"),
    ):
        address = origin.get(field)
        if (type(address) is not int or not address or origin.get(path_field) != library
                or origin.get(base_field) != rccl_base
                or not any(a <= address < b and permission in perms for a, b, _, perms in rccl_segments)):
            return False
    return True


def _validate_thread_origins(data, communication, call_pairs, opmap, problems):
    trace = data.get("rccl_thread_origins")
    if trace is None:
        return
    if not isinstance(trace, dict):
        raise ValueError("rccl_thread_origins must be structured observer facts")
    library = communication["loaded_rccl"]["path"]
    base, segments = _module_ranges(communication["loaded_rccl"])
    for origin in trace.get("records", []):
        if not isinstance(origin, dict):
            raise ValueError("thread origin must be an object")
        identifier = _integer(origin["id"], "thread origin id", 1)
        parent = communication["init_calls"].get(origin.get("parent_epoch"))
        if parent is None:
            continue
        a, b = parent
        direct_routine = bool(origin.get("module_path") == library and origin.get("module_base") == base
            and type(origin.get("start_address")) is int and type(origin.get("module_offset")) is int
            and base is not None and origin["start_address"] - base == origin["module_offset"]
            and any(start <= origin["start_address"] < end and "x" in perms for start, end, _, perms in segments))
        proven_dispatch = direct_routine or _stdthread_dispatch_proven(origin, communication, data.get("pid"), base, segments)
        valid = bool(origin.get("parent_init_cid") == a["cid"]
            and origin.get("parent_tid") == a["tid"] == b["tid"]
            and origin.get("init_output_address") == a.get("raw_rccl_payload", {}).get("comm_output_address")
            and origin.get("create_returncode") == 0 and type(origin.get("create_returncode")) is int
            and origin.get("start_routine_invocations") == 1
            and type(origin.get("child_tid")) is int and origin["child_tid"] > 0
            and origin.get("module_known") is True and origin.get("module_path_canonical") is True
            and proven_dispatch
            and type(origin.get("create_begin_ns")) is int and type(origin.get("create_end_ns")) is int
            and a["ns"] <= origin["create_begin_ns"] <= origin["create_end_ns"] <= b["ns"]
            and type(origin.get("child_start_ns")) is int
            and origin["child_start_ns"] >= origin["create_begin_ns"])
        if valid:
            communication["thread_origins"][identifier] = deepcopy(origin)


def _startup_allocation_proof(api_pair, normalized_pair, communication, call_pairs, opmap, formal_begin):
    api_a, api_b = api_pair
    norm_a, norm_b = normalized_pair
    initialization = communication["init_calls"].get(norm_b["epoch"])
    if initialization is not None and norm_a["epoch"] == norm_b["epoch"] == api_a["epoch"] == api_b["epoch"]:
        chain = _ancestor_call(api_pair, initialization[0]["cid"], call_pairs)
        if chain:
            return {"kind": "init-ancestor", "init_cid": initialization[0]["cid"], "chain": chain}
        for first in communication["first_uses"]:
            if first[0]["epoch"] != norm_b["epoch"]:
                continue
            chain = _ancestor_call(api_pair, first[0]["cid"], call_pairs)
            if (chain and len(chain) == 2 and chain[0][1]["domain"] == 0
                    and opmap[(chain[0][1]["domain"], chain[0][1]["op"])] == "hipExtModuleLaunchKernel"
                    and chain[0][1]["status_type"] == 1 and chain[0][1]["status"] == 0
                    and chain[0][0]["flags"] == chain[0][1]["flags"] == 0):
                return {"kind": "ctor-known-first-use", "init_cid": initialization[0]["cid"],
                        "first_use_cid": first[0]["cid"], "launch_cid": chain[0][0]["cid"], "chain": chain}
    origin_id = api_a.get("thread_origin_id", 0)
    origin = communication["thread_origins"].get(origin_id)
    if (origin and origin_id == api_b.get("thread_origin_id") == norm_a.get("thread_origin_id") == norm_b.get("thread_origin_id")
            and api_a["tid"] == api_b["tid"] == norm_a["tid"] == norm_b["tid"] == origin["child_tid"]
            and origin["child_start_ns"] <= api_a["ns"] <= api_b["ns"] < formal_begin
            and (origin.get("child_end_ns", 0) == 0 or api_b["ns"] <= origin["child_end_ns"])):
        parents = call_pairs.get(api_a["ancestor"], [])
        if len(parents) == 1:
            parent = parents[0]
            name = opmap[(parent[1]["domain"], parent[1]["op"])]
            chain = _ancestor_call(api_pair, parent[0]["cid"], call_pairs)
            if (chain and len(chain) == 1 and parent[1]["domain"] == 0
                    and name in {"hipMalloc", "hipExtMallocWithFlags"}
                    and parent[1]["status_type"] == 1 and parent[1]["status"] == 0):
                return {"kind": "rccl-born-proxy-thread", "init_cid": origin["parent_init_cid"],
                        "thread_origin_id": origin_id, "chain": chain}
    return None


def _controlled_startup_call(pair, communication, call_pairs, formal_begin):
    a, b = pair
    if not a["ns"] <= b["ns"] < formal_begin or a["epoch"] != b["epoch"]:
        return None
    initialization = communication["init_calls"].get(a["epoch"])
    if initialization is not None and _ancestor_call(pair, initialization[0]["cid"], call_pairs):
        return {"kind": "init-ancestor", "init_cid": initialization[0]["cid"]}
    if initialization is not None and a["epoch"] in communication["verified_pg_epochs"]:
        parents = [p[1:] for p in communication["pg_calls"].get(a["epoch"], [])]
        parents.extend(p for p in communication["pg_group_ends"] if p[0]["epoch"] == a["epoch"])
        for parent in parents:
            if _ancestor_call(pair, parent[0]["cid"], call_pairs):
                return {"kind": "source_bound_pg_prewarm_first_use", "init_cid": initialization[0]["cid"],
                        "first_use_cid": parent[0]["cid"]}
    origin_id = a.get("thread_origin_id", 0)
    origin = communication["thread_origins"].get(origin_id)
    if (origin and origin_id == b.get("thread_origin_id") and a["tid"] == b["tid"] == origin["child_tid"]
            and origin["child_start_ns"] <= a["ns"] and (not origin.get("child_end_ns") or b["ns"] <= origin["child_end_ns"])):
        return {"kind": "rccl-born-proxy-thread", "init_cid": origin["parent_init_cid"], "thread_origin_id": origin_id}
    return None


def _natural_teardown_window(communication, formal_end, stop_begin):
    evidence = communication.get("natural_shutdown_evidence")
    if not isinstance(evidence, dict):
        return None
    source = evidence.get("source", {})
    begin, end = evidence.get("started_monotonic_raw_ns"), evidence.get("ended_monotonic_raw_ns")
    if (evidence.get("process_id") != communication["rank"]["process_id"]
            or evidence.get("native_clock") != "CLOCK_MONOTONIC_RAW" or evidence.get("original_returned") is not True
            or evidence.get("return_type") != "builtins.NoneType" or "exception" in evidence
            or evidence.get("scope") != "original_WorkerProc_shutdown_including_model_and_distributed_environment"
            or type(evidence.get("thread_id")) is not int or evidence["thread_id"] <= 0
            or not isinstance(source, dict)
            or source.get("qualified_name") != "vllm.v1.executor.multiproc_executor.WorkerProc.shutdown"
            or source.get("source_sha256") != "36f51b0ac5471bc230ccfce7dfc6ced2d16d2c73e35b10a3ab81eae1ec592140"
            or source.get("file_sha256") != "3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a"
            or not isinstance(source.get("source_reference"), str) or not source["source_reference"]
            or type(begin) is not int or type(end) is not int or not formal_end < begin < end < stop_begin):
        return None
    return begin, end


def _rccl_cleanup_ancestor(pair, call_pairs, opmap):
    """A proven destructor call can release known objects on its own cleanup thread."""
    current, seen = pair, set()
    for _ in range(64):
        cid = current[0].get("ancestor")
        if not cid or cid in seen:
            return None
        seen.add(cid)
        candidates = call_pairs.get(cid, [])
        if len(candidates) != 1 or not _ancestor_call(current, cid, call_pairs):
            return None
        a, b = candidates[0]
        name = opmap[(b["domain"], b["op"])]
        if (a["domain"] == b["domain"] == 8 and name in {"ncclCommAbort", "ncclCommDestroy", "ncclCommFinalize"}
                and a["flags"] == b["flags"] == 64 and b["status_type"] == 3 and b["status"] == 0):
            return {"kind": "actual_successful_rccl_teardown_ancestor", "operation": name,
                    "cid": cid, "tid": b["tid"], "enter_sequence": a["seq"], "exit_sequence": b["seq"]}
        current = a, b
    return None


def _startup_control_proofs(rows, communication, call_pairs, opmap, formal_begin, formal_end, stop_begin,
                           allocations, agents, result, problems):
    if communication is None:
        return set()
    accepted: set[int] = set()
    controls, pools, host_ranges = [], {}, {}
    for candidates in sorted(call_pairs.values(), key=lambda group: group[0][0]["seq"] if group else 0):
        if len(candidates) != 1:
            continue
        a, b = candidates[0]
        name = opmap[(b["domain"], b["op"])]
        if b["domain"] != 0 or name not in COMM_STARTUP_CONTROL_APIS:
            continue
        pa, pb = a.get("raw_hip_control_payload"), b.get("raw_hip_control_payload")
        origin = _controlled_startup_call((a, b), communication, call_pairs, formal_begin)
        good = bool(origin and b["status_type"] == 1 and b["status"] == 0 and isinstance(pa, dict) and isinstance(pb, dict))
        effect = None
        if name == "hipHostRegister":
            good = bool(good and a["flags"] == b["flags"] == 32 and pa == pb
                and pb.get("kind") == "host_registration"
                and type(pb.get("pointer")) is int and pb["pointer"] > 0
                and type(pb.get("size_bytes")) is int and 0 < pb["size_bytes"] < 2**64
                and pb["pointer"] + pb["size_bytes"] <= 2**64
                and pb.get("flags") == 0x80000003)
            if good:
                good = not any(agents[x["agent_handle"]]["type_name"] == "GPU"
                    and max(pb["pointer"], x["address"]) < min(pb["pointer"] + pb["size_bytes"], x["address"] + x["bytes"])
                    for x in allocations)
            effect = "register_existing_host_range_portable_mapped_uncached_no_gpu_storage_owner_inferred"
        elif name == "hipMemPoolCreate":
            good = bool(good and a["flags"] == b["flags"] == 16 and pb.get("kind") == "pool_create"
                and pa.get("object_handle") is None and type(pb.get("object_handle")) is int and pb["object_handle"] > 0
                and pa.get("output_address") == pb.get("output_address") and pb.get("output_address", 0) > 0
                and pa.get("properties_address") == pb.get("properties_address") and pb.get("properties_address", 0) > 0
                and pb.get("properties_observed") is True and pb.get("reserved_zero") is True
                and pb.get("alloc_type") == 1 and pb.get("handle_types") == 0
                and pb.get("location_type") == 1 and pb.get("location_id") == communication["rank"]["logical_device_index"]
                and pb.get("max_size_bytes") == 0)
            if good:
                key = (a["epoch"], pb["object_handle"])
                good = key not in pools
                pools[key] = (a, b)
            effect = "create_nonexportable_device_pool_metadata_backing_capacity_remains_independently_audited"
        else:
            pool = pools.get((a["epoch"], pb.get("object_handle"))) if isinstance(pb, dict) else None
            good = bool(good and a["flags"] == b["flags"] == 16 and pb.get("kind") == "pool_attribute"
                and pa.get("object_handle") == pb.get("object_handle") and pool and pool[1]["seq"] < a["seq"]
                and pa.get("attribute") == pb.get("attribute") == 4 and pb.get("value") == 2**64 - 1
                and pa.get("pointer") == pb.get("pointer") and pb.get("pointer", 0) > 0)
            effect = "set_known_pool_release_threshold_only_no_claim_of_zero_gpu_backing"
        controls.append({"operation": name, "cid": b["cid"], "epoch_id": b["epoch"],
            "enter_sequence": a["seq"], "exit_sequence": b["seq"], "raw_flags": b["flags"],
            "actual_payload": deepcopy(pb), "origin_proof": origin, "accepted_startup_effect": good,
            "effect": effect, "gpu_backing_replay_mutation_applied": False, "host_cost_or_lifetime_claimed": False})
        if good:
            accepted.update((a["seq"], b["seq"]))
            if name == "hipHostRegister":
                if pb["pointer"] in host_ranges:
                    problems.add("host_registration_range_reused_without_known_release", pointer=pb["pointer"])
                host_ranges[pb["pointer"]] = (a, b)
        else:
            problems.add("startup_communication_control_effect_not_proven", operation=name, cid=b["cid"])
    teardown_window = _natural_teardown_window(communication, formal_end, stop_begin)
    accepted_host_unregistration_cids = set()
    destroyed_pools, released_hosts = set(), set()
    for candidates in sorted(call_pairs.values(), key=lambda group: group[0][0]["seq"] if group else 0):
        if len(candidates) != 1:
            continue
        a, b = candidates[0]
        name = opmap[(b["domain"], b["op"])]
        if b["domain"] != 0 or name not in COMM_TEARDOWN_CONTROL_APIS:
            continue
        pa, pb = a.get("raw_hip_control_payload"), b.get("raw_hip_control_payload")
        thread_bound = a["tid"] == communication.get("natural_shutdown_evidence", {}).get("thread_id")
        cleanup_origin = None
        if not thread_bound:
            origin = communication["thread_origins"].get(a.get("thread_origin_id"))
            thread_bound = bool(origin and a.get("thread_origin_id") == b.get("thread_origin_id")
                                and a["tid"] == b["tid"] == origin["child_tid"])
        if not thread_bound:
            cleanup_origin = _rccl_cleanup_ancestor((a, b), call_pairs, opmap)
            thread_bound = cleanup_origin is not None
        good = bool(teardown_window and thread_bound and a["epoch"] == b["epoch"] == 0
            and teardown_window[0] <= a["ns"] <= b["ns"] <= teardown_window[1]
            and b["status_type"] == 1 and b["status"] == 0
            and isinstance(pa, dict) and isinstance(pb, dict) and pa == pb)
        if name == "hipHostUnregister":
            pointer = pb.get("pointer") if isinstance(pb, dict) else None
            previous = host_ranges.get(pointer)
            good = bool(good and a["flags"] == b["flags"] == 32 and pb.get("kind") == "host_unregistration"
                        and previous and previous[1]["seq"] < a["seq"] and pointer not in released_hosts)
            if good:
                released_hosts.add(pointer)
                accepted_host_unregistration_cids.add(a["cid"])
        else:
            handle = pb.get("object_handle") if isinstance(pb, dict) else None
            previous_pools = [v for (_, key), v in pools.items() if key == handle]
            good = bool(good and a["flags"] == b["flags"] == 16 and pb.get("kind") == "pool_destroy"
                        and len(previous_pools) == 1 and previous_pools[0][1]["seq"] < a["seq"]
                        and handle not in destroyed_pools)
            if good:
                destroyed_pools.add(handle)
        controls.append({"operation": name, "cid": b["cid"], "epoch_id": b["epoch"],
            "enter_sequence": a["seq"], "exit_sequence": b["seq"], "raw_flags": b["flags"],
            "actual_payload": deepcopy(pb), "accepted_natural_teardown_effect": good,
            "cleanup_origin_proof": cleanup_origin,
            "gpu_backing_replay_mutation_applied": False, "host_cost_or_lifetime_claimed": False})
        if good:
            accepted.update((a["seq"], b["seq"]))
        else:
            problems.add("natural_communication_teardown_control_not_proven", operation=name, cid=a["cid"])
    communication["accepted_host_unregistration_cids"] = accepted_host_unregistration_cids
    for sequence, row in rows.items():
        if sequence not in accepted:
            problems.add("unsupported_native_memory_surface", sequence=sequence,
                         operation=opmap[(row["domain"], row["op"])])
    result["startup_communication_control_audit"] = controls
    return accepted


def _synchronization_proofs(data, ids, communication, opmap, call_pairs, result, problems):
    def new_resource(row):
        name = opmap.get((row["domain"], row["op"]))
        if row["epoch"] not in ids or not row["flags"] & MUTATOR_MASK or name not in SYNC_RESOURCE_APIS:
            return False
        if name not in SIGNALS:
            return True
        parents = call_pairs.get(row["ancestor"], [])
        return not (len(parents) == 1 and parents[0][0]["domain"] == 0
                    and opmap[(parents[0][0]["domain"], parents[0][0]["op"])] == "hipDeviceSynchronize")
    needed = any(new_resource(row) for row in data["events"])
    if communication is None or not needed:
        return {}
    path = Path(__file__).with_name("synchronization_resource_audit.py")
    if not path.is_file():
        problems.add("synchronization_resource_auditor_missing")
        return {}
    name = "_mlsys_synchronization_resource_audit"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            problems.add("synchronization_resource_auditor_unloadable")
            return {}
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    options = {}
    if communication.get("natural_shutdown_evidence") is not None:
        options["natural_shutdown_evidence"] = communication["natural_shutdown_evidence"]
    if communication.get("host_signal_evidence") is not None:
        options["host_signal_evidence"] = communication["host_signal_evidence"]
    audit = module.audit_synchronization_resources(data, target_epochs=ids,
        device_index=communication["rank"]["logical_device_index"], **options)
    result["synchronization_resource_audit"] = audit
    if audit.get("status") != "SYNCHRONIZATION_RESOURCES_PROVEN" or audit.get("complete_workspace") is not False:
        problems.add("synchronization_resource_lifetimes_not_proven")
        return {}
    accepted = {}
    for proof in audit.get("allowed_call_proofs", []):
        if proof.get("operation") in {"hipEventRecord", "hipEventRecord_spt", "hipEventRecordWithFlags",
                "hipEventQuery", "hipEventSynchronize", "hipStreamSynchronize", "hipDeviceSynchronize",
                "hipStreamWaitEvent", "hipStreamWaitEvent_spt"}:
            continue  # These establish the helper's proof; no mutator exemption is requested here.
        if proof.get("operation") not in SYNC_RESOURCE_APIS or proof.get("epoch") not in ids:
            problems.add("invalid_synchronization_resource_call_proof")
            continue
        candidates = call_pairs.get(proof.get("cid"), [])
        matched = [p for p in candidates if p[0]["domain"] == proof.get("domain")
            and p[0]["tid"] == p[1]["tid"] == proof.get("tid")
            and p[0]["epoch"] == p[1]["epoch"] == proof.get("epoch")
            and p[0]["seq"] == proof.get("enter_sequence") and p[1]["seq"] == proof.get("exit_sequence")
            and opmap[(p[0]["domain"], p[0]["op"])] == proof.get("operation")]
        if len(matched) != 1:
            problems.add("synchronization_resource_proof_does_not_match_raw_call", cid=proof.get("cid"))
            continue
        a, b = matched[0]
        accepted[(a["domain"], a["op"], a["cid"], a["tid"])] = deepcopy(proof)
    return accepted


def audit_native_backing(
    native_report, snapshots_by_epoch, *, device_to_agent, lt_ctor_epoch=5001, ctor_binding=None,
    communication_binding=None,
):
    """Audit exact GPU backing/constructor identities, returning fail-closed facts.

    snapshots_by_epoch maps integer epoch IDs to {"before": snapshot, "after": snapshot}.
    device_to_agent maps Torch logical device index to an actual SDK GPU agent handle.
    ctor_binding requires epoch_id, epoch_label, entrypoint, lt_only, and executed
    driver_source_sha256/driver_source_reference. The caller authenticates that source.
    All returned bytes concern fixed native backing; complete_workspace stays false.
    """
    result, problems = _base_result(), Problems()
    try:
        return _audit(
            native_report, snapshots_by_epoch, device_to_agent, lt_ctor_epoch, ctor_binding, result, problems,
            communication_binding,
        )
    except (ValueError, KeyError, TypeError, OverflowError, IndexError, AttributeError) as error:
        problems.add("invalid_input", message=str(error))
        return _finish(result, problems)


def _audit(data, snapshots, device_to_agent, ctor_id, binding, result, problems, communication_binding):
    if not isinstance(data, dict) or not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("native report and nonempty epoch snapshot map are required")
    if (
        data.get("probe_protocol_version") != 2
        or data.get("probe_protocol") != "native_allocation_api_probe_v2"
    ):
        problems.add("native_v2_required")
    if data.get("retention_mode") != "all_callbacks_v1" or data.get("filter_policy") != "all_APIs_global":
        problems.add("full_global_retention_required")
    if len(data.get("events", [])) > MAX_EVENTS:
        raise ValueError("native events exceed the audited helper bound")
    validation = analyze(data)
    result["native_protocol_validation"] = {
        key: validation[key]
        for key in ("probe_protocol_version", "global_failures", "rccl_callback_evidence", "stop_evidence")
    }
    for issue in validation["global_failures"]:
        problems.add("native_protocol_failure", reason=issue)
    # Expected allocation/signal events make the generic diagnostic PARTIAL. Only
    # its separate global setup/loss/finalization failures invalidate setup here.
    epochs = {row["id"]: row for row in data.get("epochs", [])}
    validation_epochs = {row["epoch_id"]: row for row in validation["epochs"]}
    ctor_id = _integer(ctor_id, "lt_ctor_epoch", 1)
    ids = sorted(_integer(value, "formal epoch", 1) for value in snapshots)
    if ctor_id in ids:
        raise ValueError("constructor epoch cannot be a formal epoch")
    for identifier in ids:
        if identifier not in epochs:
            problems.add("formal_epoch_missing", epoch_id=identifier)
    if ctor_id not in epochs:
        problems.add("constructor_epoch_missing", epoch_id=ctor_id)
    binding_valid = isinstance(binding, dict)
    if binding_valid:
        binding_valid = (
            type(binding.get("epoch_id")) is int
            and binding.get("epoch_id") == ctor_id
            and binding.get("lt_only") is True
            and binding.get("entrypoint") == EXPECTED_CTOR_ENTRYPOINT
            and isinstance(binding.get("epoch_label"), str)
            and bool(binding["epoch_label"])
            and isinstance(binding.get("driver_source_sha256"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", binding["driver_source_sha256"]))
            and isinstance(binding.get("driver_source_reference"), str)
            and bool(binding["driver_source_reference"].strip())
            and ctor_id in epochs
            and epochs[ctor_id]["label"] == binding["epoch_label"]
        )
    if not binding_valid:
        problems.add("controlled_lt_constructor_binding_missing_or_mismatched")
    result["constructor_binding"] = deepcopy(binding) if isinstance(binding, dict) else None
    result["constructor_binding_valid"] = binding_valid
    result["constructor_proof_scope"] = (
        "Matches the caller's executed-driver epoch/source binding; does not independently authenticate or execute the driver."
    )
    expected_generic_reasons = {
        "allocation_mutator_or_runtime_lifecycle_surface_observed",
        "unsupported_memory_or_lifecycle_surface_observed",
    }
    for identifier in [ctor_id, *ids]:
        if identifier not in validation_epochs:
            continue
        for reason in validation_epochs[identifier]["reasons"]:
            if reason not in expected_generic_reasons:
                problems.add("epoch_boundary_or_completion_failure", epoch_id=identifier, reason=reason)
    if ctor_id in epochs:
        for identifier in ids:
            if identifier in epochs and not epochs[ctor_id]["end_ns"] < epochs[identifier]["begin_ns"]:
                problems.add("constructor_not_completed_before_formal_epoch", epoch_id=identifier)
    agents = {row["handle"]: row for row in data.get("agent_inventory", {}).get("agents", [])}
    if not isinstance(device_to_agent, dict) or not device_to_agent:
        raise ValueError("an explicit device_to_agent mapping is required")
    mapping = {}
    for device, handle in device_to_agent.items():
        device, handle = _integer(device, "Torch device"), _integer(handle, "SDK agent")
        if handle not in agents or agents[handle].get("type_name") != "GPU":
            problems.add("device_mapping_not_actual_gpu_agent", device=device, agent_handle=handle)
        mapping[device] = handle
    if len(set(mapping.values())) != len(mapping):
        problems.add("device_mapping_reuses_agent")
    result["device_to_agent"] = dict(mapping)
    communication = _communication_binding(
        communication_binding, epochs, ids, ctor_id, mapping, problems, pid=data.get("pid")
    )
    if communication_binding is not None:
        result.update(
            communication_binding=deepcopy(communication_binding),
            communication_binding_valid=communication is not None,
            native_fixed_communication_bytes=None,
            communication_constructor_candidates=[],
            communication_proof_scope=(
                "Exact normalized successful HSA backing, real RCCL init ancestry and fixed formal boundaries; "
                "the caller separately authenticates executed constructor/source/rank/library observations."
            ),
        )
    if communication is not None:
        for identifier in communication["constructors"]:
            for reason in validation_epochs.get(identifier, {}).get("reasons", []):
                if reason not in expected_generic_reasons:
                    problems.add("communication_constructor_boundary_or_completion_failure",
                                 epoch_id=identifier, reason=reason)
    opmap = {
        (index, row["id"]): row["name"]
        for index, domain in enumerate(data["domains"])
        for row in domain["operations"]
    }
    events = data["events"]
    relevant_names = (
        ALLOCATE_APIS
        | FREE_APIS
        | SIGNALS
        | set(READ_ONLY_HSA_APIS)
        | HOST_UNPIN_APIS
        | HOST_UNPIN_COPY_PARENTS
        | {"hipDeviceSynchronize"}
        | (RCCL_FORMAL_APIS | RCCL_INIT_APIS | COMM_TEARDOWN_CONTROL_APIS if communication is not None else set())
    )
    pairs = defaultdict(list)
    normalized = defaultdict(list)
    scratch_events_global = []
    events_by_epoch = defaultdict(list)
    deferred_semantic_surfaces = {}
    deferred_startup_controls = {}
    all_api_rows = defaultdict(list)
    for row in events:
        for field in (
            "seq",
            "callback_seq",
            "ns",
            "tid",
            "cid",
            "ancestor",
            "epoch",
            "domain",
            "op",
            "phase",
            "flags",
            "status_type",
        ):
            _integer(row[field], "event." + field)
        if row["phase"] not in (1, 2):
            problems.add("unsupported_callback_phase", sequence=row["seq"], phase=row["phase"])
        if row["status_type"] not in (0, 1, 2, 3):
            problems.add("unknown_typed_status_kind", sequence=row["seq"], status_type=row["status_type"])
        if row["status_type"]:
            _integer(row["status"], "typed EXIT status", -(2**31), 2**31 - 1)
            if row["phase"] != 2:
                problems.add("typed_status_outside_exit", sequence=row["seq"])
            pending_event_query = (row["domain"] == 0 and row["status_type"] == 1 and row["status"] == 600
                and opmap.get((row["domain"], row["op"])) == "hipEventQuery")
            if row["status"] != 0 and not pending_event_query:
                problems.add(
                    "typed_api_failure",
                    sequence=row["seq"],
                    operation=opmap.get((row["domain"], row["op"])),
                    status=row["status"],
                )
        elif row.get("status") is not None:
            problems.add("undecoded_status_not_null", sequence=row["seq"])
        name = opmap.get((row["domain"], row["op"]))
        if name is None:
            problems.add("unknown_native_operation", sequence=row["seq"])
            continue
        call_key = (row["domain"], row["op"], row["cid"], row["tid"])
        if communication is not None and row["domain"] in (0, 1, 2, 3, 4, 5, 8):
            all_api_rows[call_key].append(row)
        if row["domain"] == 6:
            normalized[call_key].append(row)
        elif row["domain"] == 7:
            scratch_events_global.append(row)
        elif name in relevant_names:
            pairs[call_key].append(row)
        if row["epoch"]:
            events_by_epoch[row["epoch"]].append(row)
        # Plain HSA pool backing alloc/free is supported. Stream-ordered pools,
        # VMM/imports and managed/registered aliases need different state machines.
        unsupported = row["flags"] & (4 | 8 | 128 | 256)
        unsupported |= bool(row["flags"] & 16 and name not in ALLOCATE_APIS | FREE_APIS)
        unsupported |= (
            name
            in {
                "hipMallocManaged",
                "hipHostRegister",
                "hipHostUnregister",
                "hsa_amd_memory_lock",
                "hsa_amd_memory_unlock",
            }
            or name.startswith("hsa_amd_svm_")
            or name.startswith("hipMemPool")
            or name in {"hipMallocAsync", "hipFreeAsync", "hipMallocFromPoolAsync"}
        )
        if unsupported:
            if communication is not None and row["domain"] == 0 and name in COMM_STARTUP_CONTROL_APIS | COMM_TEARDOWN_CONTROL_APIS:
                deferred_startup_controls[row["seq"]] = row
            elif name in set(READ_ONLY_HSA_APIS) | HOST_UNPIN_APIS and row["domain"] == 3:
                deferred_semantic_surfaces[row["seq"]] = row
            else:
                problems.add("unsupported_native_memory_surface", sequence=row["seq"], operation=name)

    def pair(rows, category):
        enters = [r for r in rows if r["phase"] == 1]
        exits = [r for r in rows if r["phase"] == 2]
        if len(enters) != 1 or len(exits) != 1 or len(rows) != 2:
            problems.add("unpaired_native_call", category=category, sequences=[r["seq"] for r in rows])
            return None
        a, b = enters[0], exits[0]
        if not a["seq"] < b["seq"] or not a["ns"] <= b["ns"] or a["ancestor"] != b["ancestor"]:
            problems.add("invalid_native_call_order", category=category, sequences=[a["seq"], b["seq"]])
            return None
        return a, b

    api_pairs_by_cid = defaultdict(list)
    ancestry_pairs = defaultdict(list)
    if communication is not None:
        for rows in all_api_rows.values():
            matched = pair(rows, "communication ancestry")
            if matched is not None:
                ancestry_pairs[matched[1]["cid"]].append(matched)
    communication_calls = {}
    if communication is not None:
        communication_calls = _communication_calls(
            ancestry_pairs, communication, epochs, ids, opmap, problems
        )
        _validate_thread_origins(data, communication, ancestry_pairs, opmap, problems)
    sync_resource_calls = _synchronization_proofs(
        data, ids, communication, opmap, ancestry_pairs, result, problems
    )
    signal_pairs, sync_pairs = {}, {}
    validated_semantic_sequences: set[int]
    query_pairs, host_unpin_pairs, validated_semantic_sequences = [], [], set()
    copy_pairs_by_cid = defaultdict(list)
    for call_key, rows in pairs.items():
        matched = pair(rows, "API")
        if matched is None:
            continue
        a, b = matched
        name = opmap[(b["domain"], b["op"])]
        if name in ALLOCATE_APIS | FREE_APIS:
            api_pairs_by_cid[b["cid"]].append((name, a, b))
        elif name in SIGNALS:
            signal_pairs[call_key] = (name, a, b)
        elif name == "hipDeviceSynchronize":
            sync_pairs[b["cid"]] = (a, b)
        elif name in HOST_UNPIN_COPY_PARENTS | {"hipHostUnregister"}:
            copy_pairs_by_cid[b["cid"]].append((name, a, b))
        elif name in HOST_UNPIN_APIS:
            host_unpin_pairs.append((name, a, b))
        elif name in READ_ONLY_HSA_APIS:
            policy = READ_ONLY_HSA_APIS[name]
            outside_formal = all(
                not (a["ns"] < epochs[i]["end_ns"] and b["ns"] > epochs[i]["begin_ns"])
                and a["epoch"] != i
                and b["epoch"] != i
                for i in ids
                if i in epochs
            )
            status_valid = b["status_type"] == 2 and b["status"] == 0
            status_unobserved = b["status_type"] == 0 and b.get("status") is None
            accepted = (
                a["domain"] == b["domain"] == 3
                and a["flags"] == b["flags"]
                and b["flags"] in policy["flags"]
                and outside_formal
                and (status_valid or not policy["typed_status_required"] and status_unobserved)
            )
            query_pairs.append(
                {
                    "operation": name,
                    "cid": b["cid"],
                    "enter_sequence": a["seq"],
                    "exit_sequence": b["seq"],
                    "raw_flags": b["flags"],
                    "accepted_nonmutating_surface": accepted,
                    "outside_formal": outside_formal,
                    "effect": policy["effect"],
                    "typed_exit_status_observed": status_valid,
                    "successful_query_return_proven": status_valid,
                    "source_contract": HSA_EFFECT_SOURCE,
                    "nested_allocation_events_discarded": False,
                }
            )
            if accepted:
                validated_semantic_sequences.update((a["seq"], b["seq"]))
            else:
                problems.add("read_only_hsa_surface_constraints_not_proven", operation=name, cid=b["cid"])
    def finish_host_cleanup():
        explained_host_cleanup = []
        for name, a, b in host_unpin_pairs:
            candidates = copy_pairs_by_cid.get(a["ancestor"], [])
            outside_formal = all(
                not (a["ns"] < epochs[i]["end_ns"] and b["ns"] > epochs[i]["begin_ns"])
                and a["epoch"] != i
                and b["epoch"] != i
                for i in ids
                if i in epochs
            )
            parent = candidates[0] if len(candidates) == 1 else None
            parent_effect_proven = bool(parent and (
                parent[0] in HOST_UNPIN_COPY_PARENTS and parent[1]["flags"] == parent[2]["flags"] == 0
                or parent[0] == "hipHostUnregister" and communication is not None
                    and parent[1]["flags"] == parent[2]["flags"] == 32
                    and parent[1]["cid"] in communication.get("accepted_host_unregistration_cids", set())))
            accepted = bool(
                parent
                and a["domain"] == b["domain"] == 3
                and a["cid"]
                and a["ancestor"]
                and a["flags"] == b["flags"] == 32
                and outside_formal
                and b["status_type"] == 2
                and b["status"] == 0
                and parent[1]["domain"] == parent[2]["domain"] == 0
                and parent_effect_proven
                and parent[1]["tid"] == parent[2]["tid"] == a["tid"]
                and parent[1]["epoch"] == parent[2]["epoch"] == a["epoch"] == b["epoch"]
                and parent[2]["status_type"] == 1
                and parent[2]["status"] == 0
                and parent[1]["seq"] < a["seq"] < b["seq"] < parent[2]["seq"]
                and parent[1]["ns"] <= a["ns"] <= b["ns"] <= parent[2]["ns"]
            )
            explained_host_cleanup.append(
                {
                    "operation": name,
                    "cid": b["cid"],
                    "parent_cid": a["ancestor"],
                    "parent_operation": parent[0] if parent else None,
                    "enter_sequence": a["seq"],
                    "exit_sequence": b["seq"],
                    "raw_flags": b["flags"],
                    "outside_formal": outside_formal,
                    "explained_host_registration_cleanup": accepted,
                    "typed_hsa_exit_status": b["status"] if b["status_type"] == 2 else None,
                    "typed_copy_exit_status": parent[2]["status"]
                    if parent and parent[0] in HOST_UNPIN_COPY_PARENTS and parent[2]["status_type"] == 1
                    else None,
                    "argument_owner_not_inferred": True,
                    "argument_pointer_observed": False,
                    "host_registration_lifetime_verified": False,
                    "gpu_backing_replay_mutation_applied": False,
                    "effect": "host_registration_unpin_only_by_public_api_contract",
                    "source_contract": HSA_EFFECT_SOURCE,
                }
            )
            if accepted:
                validated_semantic_sequences.update((a["seq"], b["seq"]))
            else:
                problems.add("host_unpin_cleanup_scope_or_status_not_proven", cid=b["cid"], operation=name)
        result["host_registration_cleanup_audit"] = explained_host_cleanup
        for sequence, row in deferred_semantic_surfaces.items():
            if sequence not in validated_semantic_sequences:
                problems.add(
                    "unsupported_native_memory_surface",
                    sequence=sequence,
                    operation=opmap.get((row["domain"], row["op"])),
                )
        result["explicit_hsa_query_surface_audit"] = query_pairs
        result["effect_classification_policy"] = "exact_hsa_query_and_copy_host_cleanup_v1"
    normalized_call_counts: Counter[str]
    mutations, normalized_call_counts = [], Counter()
    for call_key, rows in normalized.items():
        matched = pair(rows, "normalized allocation")
        if matched is None:
            continue
        a, b = matched
        name = opmap[(b["domain"], b["op"])]
        if name not in NORMAL_ALLOCATE | NORMAL_FREE:
            problems.add("unsupported_normalized_allocation_operation", sequence=b["seq"], operation=name)
            continue
        if not b["cid"]:
            problems.add("normalized_allocation_correlation_is_none", sequence=b["seq"])
            continue
        if a["flags"] != b["flags"] or b["flags"] != (1 if name in NORMAL_ALLOCATE else 2):
            problems.add("normalized_allocation_flags_differ_from_operation", sequence=b["seq"])
            continue
        candidates = api_pairs_by_cid.get(b["cid"], [])
        expected = ALLOCATE_APIS if name in NORMAL_ALLOCATE else FREE_APIS
        candidates = [(n, start, end) for n, start, end in candidates if n in expected]
        if len(candidates) != 1:
            problems.add(
                "normalized_api_correlation_not_unique",
                sequence=b["seq"],
                cid=b["cid"],
                candidate_count=len(candidates),
            )
            continue
        api_name, api_start, api_end = candidates[0]
        if api_end["status_type"] != 2 or api_end["status"] != 0 or api_end["tid"] != b["tid"]:
            problems.add("normalized_call_success_not_proven", sequence=b["seq"], api_operation=api_name)
            continue
        if not (api_start["ns"] <= b["ns"] <= api_end["ns"]):
            problems.add("normalized_call_not_inside_matched_api", sequence=b["seq"])
            continue
        payload = b.get("raw_allocation_payload")
        if not isinstance(payload, dict) or payload.get("success_proven") is not False:
            problems.add("normalized_payload_missing_or_claims_success", sequence=b["seq"])
            continue
        address = _integer(payload["address_or_handle"], "allocation address")
        size = _integer(payload["size_bytes"], "allocation size")
        agent = _integer(payload["agent_handle"], "allocation agent")
        if name in NORMAL_ALLOCATE:
            if not address or not size or address + size > 2**64:
                problems.add("invalid_successful_allocation_range", sequence=b["seq"])
                continue
            if agent not in agents:
                problems.add(
                    "allocation_agent_not_in_actual_inventory", sequence=b["seq"], agent_handle=agent
                )
                continue
        mutation = {
            "action": "allocate" if name in NORMAL_ALLOCATE else "free",
            "sequence": b["seq"],
            "timestamp_ns": b["ns"],
            "address": address,
            "bytes": size,
            "agent_handle": agent,
            "born_epoch": b["epoch"],
            "cid": b["cid"],
            "api_operation": api_name,
            "api_enter_sequence": api_start["seq"],
            "api_exit_sequence": api_end["seq"],
            "constructor_scope_proven": bool(
                binding_valid
                and name in NORMAL_ALLOCATE
                and a["epoch"] == b["epoch"] == api_start["epoch"] == api_end["epoch"] == ctor_id
            ),
        }
        if communication is not None and name in NORMAL_ALLOCATE:
            proof = _startup_allocation_proof((api_start, api_end), (a, b), communication, ancestry_pairs,
                                              opmap, min(epochs[i]["begin_ns"] for i in ids if i in epochs))
            if proof is not None:
                mutation["communication_init_cid"] = proof["init_cid"]
                mutation["communication_proof_kind"] = proof["kind"]
                mutation["communication_proof"] = {key: value for key, value in proof.items() if key != "chain"}
                mutation["communication_init_ancestry"] = [
                    {"cid": end["cid"], "operation": opmap[(end["domain"], end["op"])],
                     "enter_sequence": start["seq"], "exit_sequence": end["seq"]}
                    for start, end in proof["chain"]
                ]
        mutations.append(mutation)
        normalized_call_counts[name] += 1
    # An HSA allocator call without its normalized event is an observation gap.
    linked_cids = {m["cid"] for m in mutations}
    for cid, calls in api_pairs_by_cid.items():
        if cid not in linked_cids:
            problems.add(
                "hsa_memory_api_without_successful_normalized_record",
                cid=cid,
                operations=[name for name, _, _ in calls],
            )
    mutations.sort(key=lambda row: row["sequence"])
    live: dict[int, dict[str, Any]]
    generations: Counter[int]
    live, generations, all_allocations = {}, Counter(), []
    for mutation in mutations:
        address = mutation["address"]
        if mutation["action"] == "allocate":
            if address in live:
                problems.add(
                    "allocation_address_reused_before_free", address=address, sequence=mutation["sequence"]
                )
                continue
            overlap = [
                r
                for r in live.values()
                if r["agent_handle"] == mutation["agent_handle"]
                and max(address, r["address"]) < min(address + mutation["bytes"], r["address"] + r["bytes"])
            ]
            if overlap:
                problems.add(
                    "overlapping_native_backing_ranges", address=address, sequence=mutation["sequence"]
                )
                continue
            generations[address] += 1
            row = {
                **mutation,
                "generation": generations[address],
                "storage_id": f"agent{mutation['agent_handle']}:0x{address:x}:generation{generations[address]}",
            }
            mutation["storage_id"] = row["storage_id"]
            live[address] = row
            all_allocations.append(row)
        elif not address:
            mutation["diagnostic_noop"] = "successful_null_free_has_no_storage_identity"
        elif address not in live:
            problems.add("unknown_free_address", address=address, sequence=mutation["sequence"])
        else:
            mutation["storage_id"] = live[address]["storage_id"]
            live.pop(address)
    result["successful_normalized_call_counts"] = dict(normalized_call_counts)
    result["successful_native_allocation_generations"] = len(all_allocations)
    result["diagnostic_noop_frees"] = [m for m in mutations if m.get("diagnostic_noop")]
    _startup_control_proofs(deferred_startup_controls, communication, ancestry_pairs, opmap,
        min(epochs[i]["begin_ns"] for i in ids if i in epochs),
        max(epochs[i]["end_ns"] for i in ids if i in epochs), data.get("stop_evidence", {}).get("begin_ns", 0),
        all_allocations, agents, result, problems)
    finish_host_cleanup()
    # Whole private-scratch pool sizes replace the capacity observation for a
    # queue. Summing growth events would double-charge one pool. For reservation,
    # retain each observed per-queue maximum, including pools later freed.
    scratch_observations = []
    pending_scratch: dict[tuple[Any, Any], dict[str, Any]]
    scratch_pairs, pending_scratch = [], {}
    for event in scratch_events_global:
        scratch_key = (event["op"], event["tid"])
        if event["phase"] == 1:
            if scratch_key in pending_scratch:
                problems.add(
                    "overlapping_private_scratch_callbacks",
                    sequences=[pending_scratch[scratch_key]["seq"], event["seq"]],
                )
            pending_scratch[scratch_key] = event
        elif event["phase"] == 2:
            start = pending_scratch.pop(scratch_key, None)
            if start is None:
                problems.add("unpaired_private_scratch_exit", sequence=event["seq"])
            elif (
                start["cid"] != event["cid"]
                or start["ancestor"] != event["ancestor"]
                or start["ns"] > event["ns"]
            ):
                problems.add(
                    "private_scratch_pair_identity_or_order_differs", sequences=[start["seq"], event["seq"]]
                )
            else:
                scratch_pairs.append((start, event))
    for event in pending_scratch.values():
        problems.add("unpaired_private_scratch_enter", sequence=event["seq"])
    for a, b in scratch_pairs:
        name = opmap[(b["domain"], b["op"])]
        payload = b.get("raw_scratch_payload")
        if not isinstance(payload, dict) or payload.get("workspace_attribution_supported") is not False:
            problems.add("private_scratch_payload_missing_or_promoted", sequence=b["seq"])
            continue
        entry_payload = a.get("raw_scratch_payload", {})
        if any(entry_payload.get(field) != payload.get(field) for field in ("agent_handle", "queue_handle")):
            problems.add("private_scratch_agent_queue_changes_within_pair", sequence=b["seq"])
            continue
        agent = _integer(payload["agent_handle"], "scratch agent")
        queue = _integer(payload["queue_handle"], "scratch queue")
        if agent not in agents or agents[agent].get("type_name") != "GPU":
            problems.add("private_scratch_agent_not_actual_gpu", sequence=b["seq"], agent_handle=agent)
            continue
        observed_size = None
        if name == "SCRATCH_MEMORY_ALLOC":
            if payload.get("allocation_size_known") is not True:
                problems.add("private_scratch_whole_pool_size_missing", sequence=b["seq"])
                continue
            observed_size = _integer(payload["allocation_size_bytes"], "scratch whole pool size")
        elif name not in {"SCRATCH_MEMORY_FREE", "SCRATCH_MEMORY_ASYNC_RECLAIM"}:
            problems.add("unknown_private_scratch_operation", sequence=b["seq"], operation=name)
            continue
        scratch_observations.append(
            {
                "agent_handle": agent,
                "queue_handle": queue,
                "sequence": b["seq"],
                "timestamp_ns": b["ns"],
                "epoch_id": b["epoch"],
                "operation": name,
                "whole_pool_capacity_bytes": observed_size,
            }
        )
    result["private_scratch_observations"] = scratch_observations

    allocations_by_id = {row["storage_id"]: row for row in all_allocations}

    def state_at(timestamp):
        state = {}
        for mutation in mutations:
            if mutation["timestamp_ns"] >= timestamp or "storage_id" not in mutation:
                continue
            if mutation["action"] == "allocate":
                state[mutation["address"]] = allocations_by_id[mutation["storage_id"]]
            elif (
                mutation["address"] in state
                and state[mutation["address"]]["storage_id"] == mutation["storage_id"]
            ):
                state.pop(mutation["address"])
            elif mutation["address"]:
                problems.add(
                    "boundary_native_replay_lifetime_mismatch",
                    sequence=mutation["sequence"],
                    boundary_ns=timestamp,
                )
        return list(state.values())

    def scratch_bound_at(timestamp):
        pools: dict[tuple[Any, Any], dict[str, Any]] = {}
        for observation in scratch_observations:
            if observation["timestamp_ns"] >= timestamp:
                continue
            key = (observation["agent_handle"], observation["queue_handle"])
            item = pools.setdefault(
                key,
                {
                    "agent_handle": key[0],
                    "queue_handle": key[1],
                    "peak_whole_pool_capacity_bytes": 0,
                    "last_operation": None,
                },
            )
            size = observation["whole_pool_capacity_bytes"]
            if size is not None:
                item["peak_whole_pool_capacity_bytes"] = max(item["peak_whole_pool_capacity_bytes"], size)
            item["last_operation"] = observation["operation"]
        return list(pools.values())

    def segments_for(snapshot, identifier, boundary):
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("segments"), list):
            raise ValueError("each before/after snapshot must contain segments")
        segments = {}
        for segment in snapshot["segments"]:
            device = _integer(segment["device"], "snapshot device")
            if device not in mapping:
                problems.add("snapshot_device_not_explicitly_mapped", epoch_id=identifier, device=device)
                continue
            address = _integer(segment["address"], "segment address", 1)
            size = _integer(segment["total_size"], "segment capacity", 1)
            if address + size > 2**64:
                raise ValueError("snapshot segment range overflow")
            key = (mapping[device], address, size)
            if key in segments:
                problems.add(
                    "duplicate_snapshot_segment", epoch_id=identifier, boundary=boundary, address=address
                )
            segments[key] = {
                "device": device,
                "address": address,
                "bytes": size,
                "agent_handle": mapping[device],
            }
        return segments

    fixed_signatures, scratch_signatures, fixed_candidates = [], [], []
    communication_signatures, communication_candidates, communication_bytes = [], {}, []
    for identifier in ids:
        if identifier not in epochs:
            continue
        epoch = epochs[identifier]
        bundle = snapshots[identifier]
        if not isinstance(bundle, dict) or bundle.get("epoch_id", identifier) != identifier:
            raise ValueError("snapshot bundle epoch binding differs")
        formal_rows = events_by_epoch[identifier]
        allowed_signals, rejected_formal, allowed_communication_calls = [], [], []
        for row in formal_rows:
            if not row["flags"] & MUTATOR_MASK:
                continue
            name = opmap.get((row["domain"], row["op"]))
            call_key = (row["domain"], row["op"], row["cid"], row["tid"])
            signal = signal_pairs.get(call_key)
            allowed = False
            communication_call = communication_calls.get(call_key)
            if communication_call is not None:
                allowed = True
                if row["phase"] == 2:
                    allowed_communication_calls.append({
                        "operation": name, "cid": row["cid"], "typed_rccl_status": row["status"],
                        "enter_sequence": communication_call[1]["seq"], "exit_sequence": row["seq"],
                        "raw_rccl_payload": deepcopy(row.get("raw_rccl_payload")),
                    })
            if name in SIGNALS and row["domain"] == 2 and signal is not None:
                _, a, b = signal
                sync_parent = sync_pairs.get(a["ancestor"])
                allowed = bool(
                    sync_parent
                    and a["flags"] == b["flags"] == 64
                    and a["epoch"] == b["epoch"] == identifier
                    and a["ancestor"] == b["ancestor"]
                    and b["status_type"] == 2
                    and b["status"] == 0
                    and sync_parent[0]["epoch"] == sync_parent[1]["epoch"] == identifier
                    and sync_parent[0]["tid"] == sync_parent[1]["tid"] == a["tid"]
                    and sync_parent[1]["status_type"] == 1
                    and sync_parent[1]["status"] == 0
                    and sync_parent[0]["seq"] < a["seq"] < b["seq"] < sync_parent[1]["seq"]
                )
                if allowed and row["phase"] == 2:
                    allowed_signals.append(
                        {
                            "operation": name,
                            "cid": b["cid"],
                            "parent_sync_cid": a["ancestor"],
                            "typed_hsa_status": b["status"],
                            "enter_sequence": a["seq"],
                            "exit_sequence": b["seq"],
                        }
                    )
                if not allowed and call_key in sync_resource_calls:
                    for parent_name, parent_a, parent_b in communication_calls.values():
                        if (parent_a["epoch"] != identifier
                                or parent_name not in RCCL_COMM_APIS | {"ncclGroupEnd"}):
                            continue
                        chain = _ancestor_call((a, b), parent_a["cid"], ancestry_pairs)
                        allowed = bool(chain and a["flags"] == b["flags"] == 64
                            and a["epoch"] == b["epoch"] == identifier
                            and b["status_type"] == 2 and b["status"] == 0
                            and all(start["domain"] == end["domain"] == 0
                                and opmap[(end["domain"], end["op"])] in {
                                    "hipDeviceSynchronize", "hipStreamSynchronize", "hipMemcpyWithStream"}
                                and start["flags"] == end["flags"] == 0
                                and end["status_type"] == 1 and end["status"] == 0
                                for start, end in chain[:-1]))
                        if allowed:
                            if row["phase"] == 2:
                                allowed_signals.append({
                                    "operation": name, "cid": b["cid"],
                                    "parent_rccl_cid": parent_a["cid"], "parent_rccl_operation": parent_name,
                                    "typed_hsa_status": b["status"], "enter_sequence": a["seq"],
                                    "exit_sequence": b["seq"], "ancestry_cids": [end["cid"] for _, end in chain],
                                })
                            break
            if not allowed and call_key in sync_resource_calls:
                allowed = True
                if row["phase"] == 2 and name in SIGNALS:
                    allowed_signals.append({"operation": name, "cid": row["cid"],
                                            "resource_lifetime_proof": sync_resource_calls[call_key]})
            if not allowed:
                rejected_formal.append(
                    {"sequence": row["seq"], "operation": name, "cid": row["cid"], "flags": row["flags"]}
                )
                problems.add(
                    "formal_native_mutator_not_proven_sync_signal",
                    epoch_id=identifier,
                    sequence=row["seq"],
                    operation=name,
                )
        scratch_events = [r for r in formal_rows if r["domain"] == 7]
        if scratch_events:
            problems.add(
                "formal_private_scratch_event_prevents_fixed_reserve",
                epoch_id=identifier,
                event_count=len(scratch_events),
            )
        entry = {
            "epoch_id": identifier,
            "epoch_label": epoch["label"],
            "allowed_fence_signal_calls": allowed_signals,
            "rejected_formal_events": rejected_formal,
            "private_scratch_event_count": len(scratch_events),
        }
        if communication is not None:
            entry["allowed_communication_calls"] = allowed_communication_calls
        for boundary, timestamp in (("before", epoch["begin_ns"]), ("after", epoch["end_ns"])):
            state = state_at(timestamp)
            segments = segments_for(bundle[boundary], identifier, boundary)
            matched_segments, owner_rows, private, communication_private = set(), [], [], []
            for allocation in state:
                agent = agents[allocation["agent_handle"]]
                row = {
                    k: allocation[k]
                    for k in (
                        "storage_id",
                        "address",
                        "bytes",
                        "agent_handle",
                        "generation",
                        "born_epoch",
                        "cid",
                        "api_operation",
                    )
                }
                if agent["type_name"] == "CPU":
                    row["owner"] = "excluded_actual_cpu_backing"
                elif allocation["agent_handle"] not in mapping.values():
                    row["owner"] = "unknown_unmapped_gpu"
                    problems.add(
                        "native_gpu_agent_outside_explicit_mapping",
                        epoch_id=identifier,
                        boundary=boundary,
                        **row,
                    )
                    result["unknown_allocations"].append(
                        {"epoch_id": identifier, "boundary": boundary, **row}
                    )
                else:
                    key = (allocation["agent_handle"], allocation["address"], allocation["bytes"])
                    if key in segments:
                        row["owner"] = "torch_reserved_segment_backing"
                        row["torch_device"] = segments[key]["device"]
                        matched_segments.add(key)
                    elif allocation["constructor_scope_proven"]:
                        row["owner"] = "controlled_lt_ctor_scope_private_backing"
                        row["constructor_epoch"] = ctor_id
                        private.append(row)
                    elif communication is not None and allocation.get("communication_init_cid"):
                        registration = {
                            "address": allocation["address"], "bytes": allocation["bytes"],
                            "agent_handle": allocation["agent_handle"], "allocation_cid": allocation["cid"],
                            "born_epoch": allocation["born_epoch"], "init_cid": allocation["communication_init_cid"],
                        }
                        proof = allocation.get("communication_proof", {})
                        if proof.get("kind") in {"ctor-known-first-use", "rccl-born-proxy-thread"}:
                            registration["proof_kind"] = proof["kind"]
                            for field in ("first_use_cid", "launch_cid", "thread_origin_id"):
                                if field in proof:
                                    registration[field] = proof[field]
                        communication_candidates[key] = registration
                        expected = communication["registrations"].get(key)
                        if expected == registration:
                            row["owner"] = "controlled_rccl_ctor_scope_private_communication_backing"
                            row["initialization_epoch"] = allocation["born_epoch"]
                            row["init_cid"] = allocation["communication_init_cid"]
                            row["init_ancestry"] = allocation["communication_init_ancestry"]
                            row["allocation_purpose_proof"] = deepcopy(proof)
                            communication_private.append(row)
                        else:
                            row["owner"] = "unregistered_communication_constructor_candidate"
                            problems.add("communication_private_backing_registration_missing_or_mismatched",
                                         epoch_id=identifier, boundary=boundary, **registration)
                            result["unknown_allocations"].append({"epoch_id": identifier, "boundary": boundary, **row})
                    else:
                        row["owner"] = "unknown_gpu_backing"
                        problems.add("unknown_gpu_backing", epoch_id=identifier, boundary=boundary, **row)
                        result["unknown_allocations"].append(
                            {"epoch_id": identifier, "boundary": boundary, **row}
                        )
                owner_rows.append(row)
            for key, segment in segments.items():
                if key not in matched_segments:
                    problems.add(
                        "torch_reserved_segment_has_no_exact_native_backing",
                        epoch_id=identifier,
                        boundary=boundary,
                        **segment,
                    )
            pools = scratch_bound_at(timestamp)
            selected_pools = [p for p in pools if p["agent_handle"] in mapping.values()]
            if len(pools) != len(selected_pools):
                problems.add(
                    "private_scratch_pool_outside_device_mapping", epoch_id=identifier, boundary=boundary
                )
            ctor_bytes = sum(row["bytes"] for row in private)
            scratch_bytes = sum(pool["peak_whole_pool_capacity_bytes"] for pool in selected_pools)
            fixed_candidates.append(ctor_bytes + scratch_bytes)
            if communication is not None:
                communication_bytes.append(sum(row["bytes"] for row in communication_private))
                communication_signatures.append(
                    tuple(sorted((row["storage_id"], row["bytes"]) for row in communication_private))
                )
            fixed_signatures.append(tuple(sorted((r["storage_id"], r["bytes"]) for r in private)))
            scratch_signatures.append(
                tuple(
                    sorted(
                        (p["agent_handle"], p["queue_handle"], p["peak_whole_pool_capacity_bytes"])
                        for p in selected_pools
                    )
                )
            )
            entry[boundary] = {
                "owner_table": owner_rows,
                "gpu_live_backing_count": sum(
                    r["owner"] != "excluded_actual_cpu_backing" for r in owner_rows
                ),
                "torch_reserved_segment_count": len(segments),
                "matched_torch_segment_count": len(matched_segments),
                "lt_private_runtime_bytes": ctor_bytes,
                "private_scratch_pool_reserve_bytes": scratch_bytes,
                "private_scratch_pool_table": selected_pools,
                "native_fixed_runtime_component_sum_bytes": ctor_bytes + scratch_bytes,
            }
            if communication is not None:
                entry[boundary]["native_fixed_communication_component_sum_bytes"] = communication_bytes[-1]
        result["epochs"].append(entry)
    if fixed_signatures and any(signature != fixed_signatures[0] for signature in fixed_signatures):
        problems.add("lt_private_backing_set_not_fixed_across_formal_boundaries")
    if scratch_signatures and any(signature != scratch_signatures[0] for signature in scratch_signatures):
        problems.add("private_scratch_reserve_not_fixed_across_formal_boundaries")
    if communication is not None:
        result["communication_constructor_candidates"] = [communication_candidates[k] for k in sorted(communication_candidates)]
        if communication["registrations"] != communication_candidates:
            problems.add("communication_backing_registrations_differ_from_observed_private_constructor_set")
        if communication_signatures and any(signature != communication_signatures[0] for signature in communication_signatures):
            problems.add("communication_private_backing_set_not_fixed_across_formal_boundaries")
        result["known_native_fixed_communication_component_sum_bytes"] = max(communication_bytes, default=0)
        result["native_fixed_communication_bytes"] = max(communication_bytes, default=0)
    if fixed_candidates:
        result["known_native_fixed_runtime_component_sum_bytes"] = max(fixed_candidates)
        result["native_fixed_runtime_bytes"] = max(fixed_candidates)
    result["native_fixed_runtime_reserve_method"] = (
        "controlled_ctor_live_backing_plus_sum_of_per_agent_queue_whole_pool_peaks"
    )
    result["scope_completion_requires_root_merge"] = [
        "Torch allocator lease and static owner registry",
        "effective model/runtime/sample/source binding",
        "fixed Runtime budget covering the native component sum",
        "private scratch alias/disjointness or conservative budget policy",
        "normal process and device completion evidence",
    ]
    return _finish(result, problems)
