"""Task-local native backing audit; never a complete Workspace/hardware verdict.

Only structured SDK callback evidence, exact allocator segment identities and a
caller-supplied controlled-constructor binding are used. No frame/size heuristics.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import re
from typing import Any

from analyze_native_probe import analyze, MUTATOR_MASK

MAX_EVENTS = 1048576
MAX_ISSUES = 4096
ALLOCATE_APIS = {"hsa_memory_allocate", "hsa_amd_memory_pool_allocate"}
FREE_APIS = {"hsa_memory_free", "hsa_amd_memory_pool_free"}
NORMAL_ALLOCATE = {"MEMORY_ALLOCATION_ALLOCATE"}
NORMAL_FREE = {"MEMORY_ALLOCATION_FREE"}
SIGNALS = {"hsa_signal_create", "hsa_signal_destroy"}
EXPECTED_CTOR_ENTRYPOINT = "ATen.getCUDABlasLtWorkspace"
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
HOST_UNPIN_COPY_PARENTS = {"hipMemcpyWithStream"}
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
    return result


def audit_native_backing(
    native_report, snapshots_by_epoch, *, device_to_agent, lt_ctor_epoch=5001, ctor_binding=None
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
            native_report, snapshots_by_epoch, device_to_agent, lt_ctor_epoch, ctor_binding, result, problems
        )
    except (ValueError, KeyError, TypeError, OverflowError, IndexError, AttributeError) as error:
        problems.add("invalid_input", message=str(error))
        return _finish(result, problems)


def _audit(data, snapshots, device_to_agent, ctor_id, binding, result, problems):
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
    )
    pairs = defaultdict(list)
    normalized = defaultdict(list)
    scratch_events_global = []
    events_by_epoch = defaultdict(list)
    deferred_semantic_surfaces = {}
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
            if row["status"] != 0:
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
            if name in set(READ_ONLY_HSA_APIS) | HOST_UNPIN_APIS and row["domain"] == 3:
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
        elif name in HOST_UNPIN_COPY_PARENTS:
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
            and parent[1]["flags"] == parent[2]["flags"] == 0
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
                if parent and parent[2]["status_type"] == 1
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
    for identifier in ids:
        if identifier not in epochs:
            continue
        epoch = epochs[identifier]
        bundle = snapshots[identifier]
        if not isinstance(bundle, dict) or bundle.get("epoch_id", identifier) != identifier:
            raise ValueError("snapshot bundle epoch binding differs")
        formal_rows = events_by_epoch[identifier]
        allowed_signals, rejected_formal = [], []
        for row in formal_rows:
            if not row["flags"] & MUTATOR_MASK:
                continue
            name = opmap.get((row["domain"], row["op"]))
            call_key = (row["domain"], row["op"], row["cid"], row["tid"])
            signal = signal_pairs.get(call_key)
            allowed = False
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
        for boundary, timestamp in (("before", epoch["begin_ns"]), ("after", epoch["end_ns"])):
            state = state_at(timestamp)
            segments = segments_for(bundle[boundary], identifier, boundary)
            matched_segments, owner_rows, private = set(), [], []
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
        result["epochs"].append(entry)
    if fixed_signatures and any(signature != fixed_signatures[0] for signature in fixed_signatures):
        problems.add("lt_private_backing_set_not_fixed_across_formal_boundaries")
    if scratch_signatures and any(signature != scratch_signatures[0] for signature in scratch_signatures):
        problems.add("private_scratch_reserve_not_fixed_across_formal_boundaries")
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
