"""Fail-closed analysis of the task-local SDK diagnostic, never a Workspace verdict."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

MUTATOR_MASK = 1535
UNSUPPORTED_MASK = 1532
DOMAIN_NAMES = [
    "HIP_RUNTIME_API",
    "HIP_COMPILER_API",
    "HSA_CORE_API",
    "HSA_AMD_EXT_API",
    "HSA_IMAGE_EXT_API",
    "HSA_FINALIZE_EXT_API",
    "MEMORY_ALLOCATION",
    "SCRATCH_MEMORY",
]
DOMAIN_NAMES_V2 = DOMAIN_NAMES + ["RCCL_API"]


def analyze(data):
    failures = []
    if data.get("kind") != "native_allocation_api_diagnostic" or data.get("schema_version") != 1:
        raise ValueError("unsupported diagnostic kind/schema")
    protocol_version = data.get("probe_protocol_version", 1)
    if type(protocol_version) is not int or protocol_version not in (1, 2):
        raise ValueError("unsupported probe protocol version")
    expected_protocol_names = (
        (None, "native_allocation_api_probe_v1")
        if protocol_version == 1
        else ("native_allocation_api_probe_v2",)
    )
    if data.get("probe_protocol") not in expected_protocol_names:
        failures.append("probe_protocol_name_mismatch")
    expected_domains = DOMAIN_NAMES if protocol_version == 1 else DOMAIN_NAMES_V2
    if data.get("complete_workspace") is not False or data.get("hardware_validated") is not False:
        raise ValueError("diagnostic must not advertise complete Workspace or hardware validation")
    for key in (
        "configured",
        "init_called",
        "ready",
        "bounds_valid",
        "write_complete",
        "finalize_callback_called",
    ):
        if data.get(key) is not True:
            failures.append(key + "_not_true")
    for key in (
        "create_status",
        "valid_status",
        "start_status",
        "stop_status",
        "callbacks_inflight",
        "dropped_events",
        "recursive_suppressed",
        "unknown_events",
        "missing_user_data",
    ):
        if type(data.get(key)) is not int or data[key] != 0:
            failures.append(key + "_not_zero")
    if data.get("context_valid") != 1:
        failures.append("context_not_valid")
    stop = data.get("stop_evidence", {})
    for key in ("invoked", "completed", "observation_window_closed", "fini_skipped_second_stop"):
        if stop.get(key) is not True:
            failures.append("stop_" + key + "_not_true")
    for key in ("stop_result", "drain_status", "callbacks_after_stop", "rejected_requests"):
        if type(stop.get(key)) is not int or stop[key] != 0:
            failures.append("stop_" + key + "_not_zero")
    if stop.get("origin") != 1 or stop.get("actual_sdk_stop_calls") != 1:
        failures.append("explicit_single_sdk_stop_not_proven")
    if not stop.get("sdk_status_name"):
        failures.append("actual_stop_status_name_missing")
    if not 0 < stop.get("begin_ns", 0) <= stop.get("end_ns", 0) < data.get("finalize_ns", 0):
        failures.append("stop_and_sdk_finalize_order_not_proven")
    drain = data.get("api_call_drain_evidence")
    if drain is not None:
        if (not isinstance(drain, dict) or drain.get("method") != "ordinary_api_enter_exit_counter_v1"
                or drain.get("ordinary_domains") != [0, 1, 2, 3, 4, 5, 8]
                or drain.get("normalized_and_scratch_counted") is not False):
            failures.append("ordinary_api_drain_contract_invalid")
        else:
            for field in ("open_calls_at_window_close", "counter_underflows", "api_enters_after_sdk_stop_return"):
                if type(drain.get(field)) is not int or drain[field] != 0:
                    failures.append("ordinary_api_drain_" + field + "_not_zero")
            for field in ("wait_iterations", "open_calls_at_sdk_stop_return", "sdk_stop_return_ns", "window_closed_ns"):
                if type(drain.get(field)) is not int or drain[field] < 0:
                    failures.append("ordinary_api_drain_" + field + "_invalid")
            if (not 0 < stop.get("begin_ns", 0) <= drain.get("sdk_stop_return_ns", 0)
                    <= drain.get("window_closed_ns", 0) == stop.get("end_ns", 0)
                    or drain.get("timeout_ns") != 1000000000):
                failures.append("ordinary_api_drain_window_order_invalid")
    explicit_finalize = data.get("explicit_client_finalize")
    if explicit_finalize is not None:
        if not isinstance(explicit_finalize, dict):
            failures.append("explicit_client_finalize_facts_invalid")
        else:
            if (type(explicit_finalize.get("schema_version")) is not int
                    or explicit_finalize["schema_version"] != 1
                    or type(explicit_finalize.get("requested")) is not bool
                    or explicit_finalize.get("actual_tool_fini_callback_count") != 1
                    or type(explicit_finalize.get("actual_tool_fini_callback_count")) is not int
                    or explicit_finalize.get("api") != "rocprofiler_client_finalize_t"
                    or explicit_finalize.get("sdk_return_type") != "void"
                    or explicit_finalize.get("sdk_return_status", "missing") is not None):
                failures.append("explicit_client_finalize_facts_invalid")
            if explicit_finalize.get("requested") is True:
                if (explicit_finalize.get("official_callback_available") is not True
                        or explicit_finalize.get("actual_client_id_saved") is not True
                        or type(explicit_finalize.get("request_begin_ns")) is not int
                        or not stop.get("end_ns", 0) <= explicit_finalize.get("request_begin_ns", 0)
                            <= data.get("finalize_ns", 0)):
                    failures.append("official_client_finalize_invocation_not_proven")
            elif explicit_finalize.get("request_begin_ns") != 0:
                failures.append("unrequested_client_finalize_has_request_timestamp")
    if data.get("metadata_truncated") is not False:
        failures.append("metadata_truncated")
    origins = data.get("rccl_thread_origins")
    if origins is not None:
        if (not isinstance(origins, dict) or origins.get("schema_version") != 1
                or origins.get("method") != "pthread_create_from_observed_rccl_init_v1"
                or not isinstance(origins.get("records"), list)
                or origins.get("record_count") != len(origins.get("records", []))
                or type(origins.get("capacity")) is not int
                or not 0 <= len(origins.get("records", [])) <= origins.get("capacity", -1) <= 256):
            failures.append("rccl_thread_origin_metadata_invalid")
        else:
            for key in ("dropped_records", "observation_errors", "init_stack_errors"):
                if type(origins.get(key)) is not int or origins[key] != 0:
                    failures.append("rccl_thread_origin_" + key + "_not_zero")
    if data.get("buffered_services_used") is not False or data.get("buffer_flush_applicable") is not False:
        failures.append("unexpected_buffered_service")
    if data.get("sdk_buffer_drop_count", "missing") is not None:
        failures.append("callback_only_sdk_drop_field_must_be_null")
    inventory = data.get("agent_inventory", {})
    agents = inventory.get("agents", [])
    if (
        inventory.get("query_status") != 0
        or inventory.get("callback_count") != 1
        or inventory.get("version") != 0
    ):
        failures.append("agent_query_setup_failed")
    if (
        inventory.get("invalid") is not False
        or not agents
        or inventory.get("retained_count") != len(agents)
        or inventory.get("reported_count") != len(agents)
        or len(agents) > inventory.get("max_agents", -1)
    ):
        failures.append("agent_inventory_incomplete_or_truncated")
    handles = [a.get("handle") for a in agents]
    if len(set(handles)) != len(handles):
        failures.append("duplicate_agent_handle")
    for agent in agents:
        if (
            agent.get("type") not in (1, 2)
            or agent.get("type_name") != {1: "CPU", 2: "GPU"}.get(agent.get("type"))
            or not agent.get("name")
        ):
            failures.append("unknown_agent_type_or_name")
        for key in ("handle", "node_id", "gpu_id", "logical_node_id", "logical_node_type_id"):
            if type(agent.get(key)) is not int:
                failures.append("invalid_agent_field:" + key)
    domains = data.get("domains", [])
    names = [d.get("name") for d in domains]
    if names != expected_domains:
        failures.append("required_domains_differ")
    opmap: dict[tuple[Any, Any], dict[str, Any]] = {}
    for i, domain in enumerate(domains):
        if (
            domain.get("index") != i
            or domain.get("enumerate_status") != 0
            or domain.get("configure_status") != 0
        ):
            failures.append("domain_setup_failure:" + str(i))
        operations = domain.get("operations", [])
        if not operations or domain.get("operation_count") != len(operations):
            failures.append("operation_enumeration_incomplete:" + str(i))
        for op in operations:
            operation_key = (i, op["id"])
            if operation_key in opmap:
                failures.append("duplicate_operation")
            opmap[operation_key] = op
    full_retention = False
    if protocol_version == 1:
        if data.get("filter_policy") != "all_mutators_global_and_all_APIs_in_or_entered_epoch":
            failures.append("unknown_filter_policy_for_legacy_v1")
    else:
        retention_mode = data.get("retention_mode")
        full_retention = retention_mode == "all_callbacks_v1"
        if (
            retention_mode not in ("all_callbacks_v1", "mutators_and_epoch_v1")
            or data.get("retain_all_option_valid") is not True
        ):
            failures.append("invalid_retention_mode_or_option")
        expected_filter = (
            "all_APIs_global" if full_retention else "all_mutators_global_and_all_APIs_in_or_entered_epoch"
        )
        if data.get("filter_policy") != expected_filter:
            failures.append("retention_mode_filter_policy_mismatch")
        if full_retention and data.get("filtered_nonmutator_events") != 0:
            failures.append("full_retention_cannot_filter_callbacks")
    events = data.get("events", [])
    if data.get("observed_callbacks") != data.get("attempted_events", 0) + data.get(
        "filtered_nonmutator_events", 0
    ):
        failures.append("observed_callback_accounting_mismatch")
    observed_from_ops = sum(
        op.get("enter_count", 0) + op.get("exit_count", 0) + op.get("other_phase_count", 0)
        for op in opmap.values()
    )
    if observed_from_ops != data.get("observed_callbacks"):
        failures.append("global_operation_count_mismatch")
    if sum(op.get("filtered_nonmutator_events", 0) for op in opmap.values()) != data.get(
        "filtered_nonmutator_events"
    ):
        failures.append("filtered_operation_count_mismatch")
    if any(
        op.get("flags", 0) & MUTATOR_MASK and op.get("filtered_nonmutator_events", 0) for op in opmap.values()
    ):
        failures.append("allocation_mutator_was_filtered")
    if len(events) != data.get("retained_events") or len(events) != data.get("attempted_events"):
        failures.append("event_count_mismatch_or_loss")
    if len(events) > data.get("max_events", -1):
        failures.append("event_limit_exceeded")
    epoch_map = {e["id"]: e for e in data.get("epochs", [])}
    if len(epoch_map) != len(data.get("epochs", [])) or 0 in epoch_map:
        failures.append("duplicate_or_zero_epoch")
    retained_by_op = Counter((e.get("domain"), e.get("op")) for e in events)
    for operation_key, op in opmap.items():
        total = op.get("enter_count", 0) + op.get("exit_count", 0) + op.get("other_phase_count", 0)
        if retained_by_op[operation_key] + op.get("filtered_nonmutator_events", 0) != total:
            failures.append("retained_operation_count_mismatch")
    epoch_events = defaultdict(list)
    open_calls = defaultdict(list)
    pairs, unmatched_exits, invalid_events = [], [], []
    for index, event in enumerate(events, 1):
        if event.get("callback_seq", 0) <= (events[index - 2].get("callback_seq", 0) if index > 1 else 0):
            failures.append("nonincreasing_callback_sequence")
        if full_retention and event.get("callback_seq") != index:
            failures.append("full_retention_callback_sequence_gap")
        if event.get("seq") != index:
            failures.append("noncontiguous_sequence")
        op = opmap.get((event.get("domain"), event.get("op")))
        if op is None or op.get("flags") != event.get("flags"):
            failures.append("unknown_operation_or_flags")
        if event.get("epoch"):
            if event["epoch"] not in epoch_map:
                failures.append("event_has_unknown_epoch")
            epoch_events[event["epoch"]].append(event)
        call_key = (event.get("domain"), event.get("op"), event.get("cid"), event.get("tid"))
        if event.get("phase") == 1:
            open_calls[call_key].append(event)
        elif event.get("phase") == 2:
            if open_calls[call_key]:
                pairs.append((open_calls[call_key].pop(), event))
            else:
                unmatched_exits.append(event)
        else:
            invalid_events.append(event)
    unclosed = [event for stack in open_calls.values() for event in stack]
    rccl_evidence = {
        "required_by_protocol": protocol_version == 2,
        "configured_domain_present": "RCCL_API" in names,
        "observed_callback_events": sum(r.get("domain") == 8 for r in events),
        "successful_known_exit_events": 0,
    }
    if protocol_version == 2:
        # A configured callback table alone does not show that the loaded RCCL
        # runtime registered with the SDK. Require an actually observed, paired,
        # typed successful RCCL call (e.g. the ncclGetVersion control).
        rccl_evidence["successful_known_exit_events"] = sum(
            b.get("domain") == 8 and b.get("status_type") == 3 and b.get("status") == 0 for a, b in pairs
        )
        if not rccl_evidence["observed_callback_events"] or not rccl_evidence["successful_known_exit_events"]:
            failures.append("rccl_runtime_callback_binding_not_observed")
    reports = []
    for identifier, epoch in epoch_map.items():
        reasons = list(dict.fromkeys(failures))
        rows = epoch_events[identifier]
        begin, end = epoch.get("begin_ns", 0), epoch.get("end_ns", 0)
        if epoch.get("ended") is not True or not 0 < begin < end:
            reasons.append("unclosed_or_invalid_epoch")
        sync = epoch.get("successful_device_sync_exit_ns", 0)
        if not begin < sync < end:
            reasons.append("no_observed_successful_device_synchronize_exit")
        observed_syncs = [
            r["ns"]
            for r in rows
            if r.get("phase") == 2
            and r.get("status_type") == 1
            and r.get("status") == 0
            and opmap.get((r.get("domain"), r.get("op")), {}).get("name") == "hipDeviceSynchronize"
        ]
        if not observed_syncs or max(observed_syncs) != sync:
            reasons.append("device_sync_summary_does_not_match_raw_success")
        if not 0 < data.get("init_ns", 0) < begin or not end < data.get("finalize_ns", 0):
            reasons.append("epoch_not_inside_initialized_and_finalized_trace")
        if not end < stop.get("begin_ns", 0):
            reasons.append("epoch_not_closed_before_explicit_stop")
        if epoch.get("api_events") != len(rows):
            reasons.append("epoch_event_count_mismatch")
        if any(not begin <= row.get("ns", 0) <= end for row in rows):
            reasons.append("event_outside_epoch_timestamp_range")
        mutators = [r for r in rows if r.get("flags", 0) & MUTATOR_MASK]
        unsupported = [r for r in rows if r.get("flags", 0) & UNSUPPORTED_MASK]
        if mutators:
            reasons.append("allocation_mutator_or_runtime_lifecycle_surface_observed")
        if unsupported:
            reasons.append("unsupported_memory_or_lifecycle_surface_observed")
        if epoch.get("mutator_events") != len(mutators) or epoch.get("unsupported_events") != len(
            unsupported
        ):
            reasons.append("epoch_mutation_counter_mismatch")
        crossing = sum(
            a["ns"] < begin <= b["ns"]
            or a["ns"] <= end < b["ns"]
            or ((a.get("epoch") == identifier) != (b.get("epoch") == identifier))
            for a, b in pairs
            if a.get("flags", 0) & MUTATOR_MASK
        )
        # An unmatched pre-entry ENTER may still be active during this epoch.
        # An unmatched EXIT before entry is outside the epoch, but retained in diagnostics.
        unresolved = sum(r["ns"] <= end for r in unclosed if r.get("flags", 0) & MUTATOR_MASK)
        unresolved += sum(
            begin <= r["ns"] <= end
            for r in unmatched_exits + invalid_events
            if r.get("flags", 0) & MUTATOR_MASK
        )
        if crossing or unresolved:
            reasons.append("cross_boundary_or_unmatched_mutator_api_lifetime")
        if epoch.get("dropped_at_begin") != 0 or epoch.get("dropped_at_end") != 0:
            reasons.append("epoch_or_prior_event_loss")
        if epoch.get("recursive_at_begin") != 0 or epoch.get("recursive_at_end") != 0:
            reasons.append("epoch_or_prior_callback_suppression")
        counts = Counter(
            opmap.get((r["domain"], r["op"]), {}).get("name", "UNKNOWN") for r in rows if r["phase"] == 1
        )
        mutations = Counter(
            opmap.get((r["domain"], r["op"]), {}).get("name", "UNKNOWN") for r in mutators if r["phase"] == 1
        )
        reasons = list(dict.fromkeys(reasons))
        allocation_rows = [r for r in rows if r.get("domain") == 6]
        scratch_rows = [r for r in rows if r.get("domain") == 7]
        signal_rows = [
            r
            for r in rows
            if opmap.get((r.get("domain"), r.get("op")), {}).get("name")
            in ("hsa_signal_create", "hsa_signal_destroy")
        ]
        sync_correlations = {
            r.get("cid")
            for r in rows
            if opmap.get((r.get("domain"), r.get("op")), {}).get("name") == "hipDeviceSynchronize"
        }
        signal_sync_descendants = sum(
            r.get("ancestor") in sync_correlations and r.get("ancestor") != 0 for r in signal_rows
        )
        domain_coverage_valid = (
            not failures
            and epoch.get("ended") is True
            and begin < sync < end < stop.get("begin_ns", 0)
            and observed_syncs
            and max(observed_syncs) == sync
            and not crossing
            and not unresolved
        )
        reports.append(
            {
                "epoch_id": identifier,
                "label": epoch.get("label"),
                "status": "PARTIAL" if reasons else "NO_NATIVE_ALLOCATION_MUTATOR_API_CALLS_OBSERVED",
                "reasons": reasons,
                "api_event_count": len(rows),
                "mutator_event_count": len(mutators),
                "unsupported_event_count": len(unsupported),
                "normalized_allocation_event_count": len(allocation_rows),
                "private_scratch_event_count": len(scratch_rows),
                "signal_lifecycle_event_count": len(signal_rows),
                "signal_events_with_direct_device_sync_ancestor": signal_sync_descendants,
                "private_scratch_growth_status": (
                    "OBSERVED_UNSUPPORTED_SCRATCH_EVENTS"
                    if scratch_rows
                    else "NO_SCRATCH_EVENTS_OBSERVED_IN_CONFIGURED_DOMAIN"
                    if domain_coverage_valid
                    else "INSUFFICIENT_EVIDENCE"
                ),
                "cross_boundary_api_count": crossing,
                "unresolved_api_count": unresolved,
                "enter_counts_by_operation": dict(sorted(counts.items())),
                "mutator_enter_counts_by_operation": dict(sorted(mutations.items())),
                "complete_workspace": False,
                "includes_non_torch_complete": False,
            }
        )
    return {
        "schema_version": 1,
        "kind": "native_allocation_api_diagnostic_analysis",
        "status": "PARTIAL"
        if failures or not reports or any(e["status"] == "PARTIAL" for e in reports)
        else "DIAGNOSTIC_COMPLETE",
        "complete_workspace": False,
        "hardware_validated": False,
        "probe_protocol_version": protocol_version,
        "rccl_callback_evidence": rccl_evidence,
        "global_failures": list(dict.fromkeys(failures)),
        "agent_inventory": inventory,
        "stop_evidence": stop,
        "filter_policy": data.get("filter_policy"),
        "filtered_nonmutator_events": data.get("filtered_nonmutator_events"),
        "unmatched_global_enter_count": len(unclosed),
        "unmatched_global_exit_count": len(unmatched_exits),
        "epochs": reports,
        "claim_scope": "Only callbacks in the initialized-to-explicit-stop observation window and declared epochs. Signal/lifecycle records are retained, private-scratch events remain unsupported, and absence of normalized backing allocation records does not prove absence of storage leases, pool reuse, device heap usage or driver allocations.",
        "remaining_proofs": [
            "full Torch lease history and static owner inventory",
            "no uninstrumented non-Torch pool/suballocator",
            "driver/device-side allocation coverage or explicit scope exclusion",
            "effective Runtime and sample binding",
            "actual process and GPU completion/finalization provenance",
        ],
    }


def allocation_scope_paths(data, addresses):
    """Show SDK correlation ancestry for selected raw allocation EXIT records.

    This does not infer owner/purpose or classify a normalized record as success.
    It never substitutes timing proximity or backtrace text for correlation IDs.
    """
    names = {
        (i, op["id"]): op["name"]
        for i, d in enumerate(data.get("domains", []))
        for op in d.get("operations", [])
    }
    events = data.get("events", [])
    selected = [
        r
        for r in events
        if r.get("domain") == 6
        and r.get("phase") == 2
        and r.get("flags", 0) & 1
        and r.get("raw_allocation_payload", {}).get("address_or_handle") in addresses
    ]
    frontier = {value for r in selected for value in (r.get("cid", 0), r.get("ancestor", 0)) if value}
    nodes: dict[tuple[Any, ...], dict[str, Any]]
    seen, nodes = set(), {}
    depth_limited = False
    for _ in range(64):
        if not frontier:
            break
        seen.update(frontier)
        next_ids = set()
        for r in events:
            if r.get("cid") not in frontier or r.get("domain") not in (0, 1, 2, 3, 4, 5, 8):
                continue
            key = (r["cid"], r["domain"], r["op"], r.get("ancestor", 0), r.get("tid"))
            node = nodes.setdefault(
                key,
                {
                    "cid": r["cid"],
                    "ancestor": r.get("ancestor", 0),
                    "domain": r["domain"],
                    "operation": names.get((r["domain"], r["op"]), "UNKNOWN"),
                    "thread_id": r.get("tid"),
                    "enter_count": 0,
                    "exit_count": 0,
                    "typed_exit_status": None,
                    "status_type": None,
                },
            )
            if r.get("phase") == 1:
                node["enter_count"] += 1
            elif r.get("phase") == 2:
                node["exit_count"] += 1
                if r.get("status_type"):
                    node["typed_exit_status"] = r.get("status")
                    node["status_type"] = r.get("status_type")
            if r.get("ancestor") and r["ancestor"] not in seen:
                next_ids.add(r["ancestor"])
        frontier = next_ids
    else:
        depth_limited = bool(frontier)
    by_id = defaultdict(list)
    for node in nodes.values():
        node["phase_pair_complete"] = node["enter_count"] == 1 and node["exit_count"] == 1
        by_id[node["cid"]].append(node)
    output = []
    for record in selected:
        pending = [value for value in (record.get("cid", 0), record.get("ancestor", 0)) if value]
        visited, missing, candidates = set(), set(), []
        while pending:
            cid = pending.pop(0)
            if cid in visited:
                continue
            visited.add(cid)
            matches = by_id.get(cid, [])
            if not matches:
                missing.add(cid)
                continue
            candidates.extend(matches)
            pending.extend(n["ancestor"] for n in matches if n["ancestor"])
        output.append(
            {
                "allocation_record_sequence": record.get("seq"),
                "raw_allocation_payload": record.get("raw_allocation_payload"),
                "record_correlation": record.get("cid"),
                "record_ancestor": record.get("ancestor"),
                "candidate_api_scopes": candidates,
                "missing_correlation_ids": sorted(missing),
                "ambiguous_correlation_ids": sorted(cid for cid in visited if len(by_id.get(cid, [])) > 1),
                "scope_chain_complete": bool(candidates)
                and not missing
                and not depth_limited
                and all(len(by_id.get(cid, [])) == 1 for cid in visited)
                and all(n["phase_pair_complete"] for n in candidates),
                "success_or_purpose_inferred": False,
            }
        )
    return {
        "requested_addresses": sorted(addresses),
        "matched_allocation_records": len(selected),
        "full_global_retention_reported": data.get("filter_policy") == "all_APIs_global",
        "depth_limit_reached": depth_limited,
        "allocations": output,
        "scope": "SDK API correlation/ancestor structure only; complete ownership and successful allocation need separate proof.",
    }


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--allocation-address",
        action="append",
        default=[],
        type=lambda value: int(value, 0),
        help="Show SDK scope ancestry for this raw allocation address/handle (repeatable; hex accepted)",
    )
    args = parser.parse_args()
    if args.report.stat().st_size > 1024 * 1024 * 1024:
        raise ValueError("diagnostic exceeds 1GiB analysis bound")
    data = json.loads(
        args.report.read_text(),
        object_pairs_hook=unique_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )
    result = analyze(data)
    if args.allocation_address:
        if any(not 0 < value < 2**64 for value in args.allocation_address):
            raise ValueError("allocation addresses must be positive unsigned 64-bit values")
        result["selected_allocation_scopes"] = allocation_scope_paths(data, set(args.allocation_address))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
