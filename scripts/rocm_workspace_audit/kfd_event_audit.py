"""Replay passive KFD event-page evidence, never a complete GPU memory verdict.

The caller authenticates the original artifacts and source/build receipts. This
component independently joins their typed native/UAPI records, without reading
private ROCr objects or trusting a producer's ownership or success booleans.
"""

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any


class KfdEvidenceError(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise KfdEvidenceError(message)


def _int(value, name, low=0):
    _need(type(value) is int and low <= value < 2**64, name + " is not an actual integer")
    return value


def _source(binding, abi):
    _need(isinstance(binding, dict), "Original source/build binding is missing")
    hashes = ("probe_source_sha256", "native_probe_source_sha256", "probe_library_sha256",
        "hsa_runtime_sha256", "libc_sha256", "kfd_header_sha256", "drm_header_sha256")
    for key in hashes:
        _need(isinstance(binding.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", binding[key]), key + " is missing")
    _need(isinstance(binding.get("kernel_release"), str) and bool(binding["kernel_release"]), "Kernel identity is missing")
    _need(isinstance(binding.get("amdgpu_module_build_id"), str)
        and re.fullmatch(r"[0-9a-f]{40}", binding["amdgpu_module_build_id"]), "Actual amdgpu build ID is missing")
    _need(binding.get("abi") == abi, "Source-bound compiled ABI differs from raw tape")
    _need(isinstance(binding.get("device_nodes"), list) and binding["device_nodes"], "Actual render-node topology is missing")
    return {key: binding[key] for key in hashes + ("kernel_release", "amdgpu_module_build_id")}


def _bindings(report, source):
    artifacts = source.get("artifacts")
    names = ("probe_source", "native_probe_source", "probe_library", "hsa_runtime", "libc", "kfd_header", "drm_header")
    _need(isinstance(artifacts, dict) and set(artifacts) == set(names), "Seven original source/binary/header artifacts are required")
    for name in names:
        artifact = artifacts[name]
        _need(isinstance(artifact, dict) and isinstance(artifact.get("path"), str) and artifact["path"].startswith("/")
            and artifact.get("sha256") == source[name + "_sha256"]
            and _int(artifact.get("size_bytes"), "Original artifact byte count", 1) > 0,
            "Source artifact differs from its actual hash/path: " + name)
        for field in ("loaded_path", "realpath"):
            if field in artifact:
                _need(isinstance(artifact[field], str) and artifact[field].startswith("/"), "Invalid observed artifact alias")
    probe = artifacts["probe_source"]
    _need(report.get("compiled_source_path") == probe.get("loaded_path", probe["path"]),
        "Compiled KFD source differs from the authenticated original file")
    _need(type(report.get("cplusplus")) is int and report["cplusplus"] >= 201703
        and isinstance(report.get("compiler_version"), str) and report["compiler_version"], "Actual compiler identity is absent")
    bindings = report.get("libc_bindings")
    _need(isinstance(bindings, dict), "Actual resolved libc function bindings are absent")
    symbols = {1: "ioctl", 2: "mmap", 3: "mmap64", 4: "munmap", 5: "close", 6: "dup2", 7: "dup3", 8: "mremap", 9: "close_range"}
    libc = artifacts["libc"]
    paths = {libc.get(key) for key in ("path", "loaded_path", "realpath") if libc.get(key)}
    for name in {symbols[row["api"]] for row in report["events"]}:
        binding = bindings.get(name, {})
        _need(binding.get("dladdr_success") is True
            and _int(binding.get("address"), "Actual libc function address", 1) >= _int(binding.get("module_base"), "Actual libc module base", 1)
            and binding.get("module_path") in paths, "Observed libc target is not the authenticated loaded file: " + name)


def _tape(report):
    _need(isinstance(report, dict), "Original KFD tape is missing")
    for key, expected in {"schema_version": 1, "kind": "passive_kfd_event_api_tape",
        "complete_workspace": False, "hardware_validated": False, "write_complete": True,
        "scope": "observed_libc_kfd_and_mapping_prefix", "clock": "CLOCK_MONOTONIC_RAW"}.items():
        _need(type(report.get(key)) is type(expected) and report[key] == expected, "KFD tape contract: " + key)
    pid = _int(report.get("pid"), "KFD process", 1)
    for key in ("dropped_rows", "read_errors", "forward_errors", "reentrant_calls", "active_mutators"):
        _need(type(report.get(key)) is int and report[key] == 0, "KFD observer uncertainty: " + key)
    fork = report.get("fork_observation", {})
    _need(fork.get("atfork_registered") is True and fork.get("inherited_kfd_resources_unsupported") is False,
        "Inherited KFD resources or unobserved fork are unsupported")
    rows, abi = report.get("events"), report.get("abi")
    _need(isinstance(rows, list) and isinstance(abi, dict), "Bounded raw events/compiled ABI are missing")
    _need(abi.get("machine") == "linux_x86_64_sysv" and abi.get("pointer_bytes") == 8
        and abi.get("off_t_bytes") == abi.get("off64_t_bytes") == 8, "Unsupported ioctl/mmap ABI")
    flags = [_int(abi.get(key), "Compiled memory-type flag", 1)
        for key in ("gtt_flag", "vram_flag", "userptr_flag", "doorbell_flag", "mmio_flag")]
    _need(len(set(flags)) == len(flags) and all(flag & (flag - 1) == 0 for flag in flags),
        "Memory-type flags are not distinct positive bits")
    _need(abi.get("create_event_bytes") == 32 and abi.get("allocate_bytes") == 40,
        "Public KFD fixed-width request structures differ")
    _need(0 < len(rows) < _int(report.get("capacity"), "KFD row capacity", 1) <= 131072, "KFD row bound reached")
    for key in ("attempted_rows", "retained_rows", "last_sequence"):
        _need(type(report.get(key)) is int and report[key] == len(rows), "KFD row accounting differs: " + key)
    width = _int(report.get("record_size_bytes"), "KFD record size", 1)
    _need(report.get("storage_capacity_bytes") == report["capacity"] * width, "KFD static byte budget differs")
    pending, pairs = {}, []
    for sequence, row in enumerate(rows, 1):
        _need(isinstance(row, dict) and row.get("seq") == sequence and row.get("pid") == pid,
            "Raw KFD sequence/PID was lost, reordered or inherited")
        _int(row.get("ns"), "KFD raw clock", 1)
        call_key = _int(row.get("call_id"), "KFD call", 1), _int(row.get("tid"), "KFD thread", 1)
        _need(_int(row.get("api"), "Actual libc API", 1) in range(1, 10), "Unknown observed libc API")
        _need(_int(row.get("operation"), "Actual compiled KFD operation") < 18, "Unsupported compiled KFD class")
        _need(_int(row.get("phase"), "Actual observation phase", 1) in (1, 2), "Unknown observation phase")
        if row["api"] == 1:
            _need(row.get("operation") == abi.get("request_operations", {}).get(str(row.get("request")), 0),
                "Ioctl class differs from the actual compiled UAPI table")
        _need(isinstance(row.get("values"), list) and len(row["values"]) == 8,
            "Original typed UAPI values are missing")
        for value in row["values"]:
            _int(value, "Typed UAPI value")
        if row.get("phase") == 1:
            _need(call_key not in pending, "Duplicate KFD API entry")
            pending[call_key] = row
        elif row.get("phase") == 2:
            before = pending.pop(call_key, None)
            _need(isinstance(before, dict) and before["ns"] <= row["ns"], "KFD API entry/exit mismatch")
            for field in ("api", "operation", "request", "argument_address", "address", "length",
                    "offset", "protection", "flags", "target_fd"):
                _need(before.get(field) == row.get(field), "KFD API original arguments differ: " + field)
            _need(type(row.get("retval")) is int and type(row.get("errno")) is int, "Actual libc result/errno is missing")
            pairs.append((before, row))
        else:
            raise KfdEvidenceError("Invalid KFD callback phase")
    _need(all(row.get("api") == 1 and row.get("operation") == 7 for row in pending.values()),
        "A mutating/unknown libc call remains open")
    return pid, abi, sorted(pairs, key=lambda pair: pair[1]["seq"])


def _native(report):
    _need(isinstance(report, dict) and report.get("write_complete") is True
        and report.get("filter_policy") == "all_APIs_global", "Original full native tape is missing")
    _need(report.get("stop_evidence", {}).get("observation_window_closed") is True
        and report.get("stop_evidence", {}).get("stop_result") == 0, "Native SDK observation is not closed")
    for key in ("dropped_events", "recursive_suppressed", "unknown_events", "missing_user_data"):
        _need(type(report.get(key)) is int and report[key] == 0, "Native observer uncertainty: " + key)
    operations = {(i, row["id"]): row["name"] for i, domain in enumerate(report.get("domains", []))
        for row in domain["operations"]}
    wanted = {"hsa_signal_create", "hsa_signal_destroy", "hsa_amd_signal_get_event_id"}
    pending, calls = {}, {}
    for sequence, row in enumerate(report.get("events", []), 1):
        _need(row.get("seq") == sequence, "Native sequence is not complete")
        name = operations.get((row.get("domain"), row.get("op")))
        if name not in wanted:
            continue
        _need(row.get("domain") == (3 if name == "hsa_amd_signal_get_event_id" else 2), "Wrong HSA query domain")
        call_key = row.get("cid"), row.get("tid")
        if row.get("phase") == 1:
            _need(call_key not in pending, "Duplicate HSA API entry")
            pending[call_key] = row
        elif row.get("phase") == 2:
            before = pending.pop(call_key, None)
            _need(isinstance(before, dict) and before["ns"] <= row["ns"], "HSA API is not paired")
            _need(row["cid"] not in calls, "HSA CID is not unique")
            calls[row["cid"]] = (name, before, row)
    _need(not pending, "HSA signal/query API remains open")
    return calls


def _signal_query(calls, create_cid, query_cid):
    create = calls.get(create_cid)
    query = calls.get(query_cid)
    _need(create is not None and create[0] == "hsa_signal_create"
        and query is not None and query[0] == "hsa_amd_signal_get_event_id", "Actual HSA create/query locators do not resolve")
    for call in (create, query):
        _need(call[2].get("status_type") == 2 and call[2].get("status") == 0, "HSA create/query typed success is absent")
    signal = create[2].get("raw_hsa_signal_payload", {})
    payload = query[2].get("raw_hsa_signal_event_payload", {})
    query_input = query[1].get("raw_hsa_signal_event_payload", {})
    handle = _int(signal.get("signal_handle"), "Actual HSA signal handle", 1)
    _need(payload.get("signal_handle") == handle and payload.get("payload_observed") is True
        and create[2]["ns"] < query[1]["ns"], "Public event query is not bound to the observed signal generation")
    _need(query_input.get("signal_handle") == handle and query_input.get("event_id") is None
        and query_input.get("payload_observed") is True
        and _int(query_input.get("output_address"), "Actual public query output address", 1) == payload.get("output_address"),
        "Public event-query original input differs")
    for name, _, end in calls.values():
        other = end.get("raw_hsa_signal_payload", {})
        if (name in {"hsa_signal_create", "hsa_signal_destroy"} and end.get("status_type") == 2 and end.get("status") == 0
                and other.get("signal_handle") == handle and create[2]["seq"] < end["seq"] <= query[2]["seq"]):
            raise KfdEvidenceError("HSA signal was destroyed or reused before the public query")
    return create, query, handle, _int(payload.get("event_id"), "Actual public event ID", 1)


def _identity(row, kind):
    value = row.get("fd_identity", {})
    _need(value.get("observed") is True and value.get("path_complete") is True and value.get("kind") == kind,
        "Actual KFD/render fd identity is absent")
    expected = "/dev/kfd" if kind == 1 else r"/dev/dri/renderD[0-9]+"
    _need(re.fullmatch(expected, value.get("path", "")), "Device path identity differs")
    _need(_int(value.get("mode"), "Device file mode") & 0o170000 == 0o020000, "FD is not a character device")
    return tuple(_int(value.get(key), "Actual fd." + key) for key in ("fd", "dev", "rdev", "inode", "mode"))


def _overlap(a, size, b, width):
    return size > 0 and width > 0 and a < b + width and b < a + size


def _startup_cutoff(native, target_epochs):
    if target_epochs is None:
        return None
    _need(isinstance(target_epochs, list) and target_epochs and len(set(target_epochs)) == len(target_epochs),
        "Target native epoch locators are missing or duplicate")
    epochs = {row["id"]: row for row in native.get("epochs", [])}
    selected = []
    for identifier in target_epochs:
        _int(identifier, "Target native epoch", 1)
        epoch = epochs.get(identifier, {})
        _need(epoch.get("ended") is True and 0 < epoch.get("begin_ns", 0) < epoch.get("end_ns", 0),
            "Target epoch is not an actual closed native interval")
        selected.append(epoch["begin_ns"])
    return min(selected)


def _replay(report, source, startup_cutoff=None):
    pid, abi, pairs = _tape(report)
    _source(source, abi)
    _bindings(report, source)
    generations: dict[tuple, int] = defaultdict(int)
    fd_mutations = []
    for a, b in pairs:
        if a["api"] == 5 and b["retval"] != 0 and a.get("fd_identity", {}).get("kind") in (1, 2):
            raise KfdEvidenceError("A failed device close leaves FD lifetime uncertain")
        if a["api"] == 5 and b["retval"] == 0:
            fd_mutations.append((a, b, a["fd_identity"]["fd"], a["fd_identity"]["fd"]))
        elif a["api"] in (6, 7) and b["retval"] >= 0 and a["fd_identity"]["fd"] != a["target_fd"]:
            fd_mutations.append((a, b, a["target_fd"], a["target_fd"]))
        elif a["api"] == 9 and b["retval"] == 0:
            _need(a["flags"] == 0, "close_range flags/descriptor-table changes are unsupported")
            fd_mutations.append((a, b, a["address"], a["length"]))

    def fd_key(row, kind):
        identity = _identity(row, kind)
        fd = identity[0]
        epoch = sum(b["seq"] < row["seq"] for _, b, first, last in fd_mutations if first <= fd <= last)
        return identity + (epoch,)

    def stable_fd(a, b, kind):
        key = fd_key(a, kind)
        _need(key == fd_key(b, kind), "FD was closed, rebound or changed during the original call")
        _need(not any(first <= key[0] <= last and x["ns"] <= b["ns"] and a["ns"] <= y["ns"]
            for x, y, first, last in fd_mutations), "Concurrent FD replacement makes the call ambiguous")
        return key

    allocations: dict[Any, Any] = {}
    events: dict[Any, Any] = {}
    pages: dict[Any, Any] = {}
    acquires: dict[Any, Any] = {}
    mappings: list[dict] = []
    all_allocations, all_events, startup_controls, memory_queries = [], [], [], []
    for a, b in pairs:
        api, op, result = a["api"], a["operation"], b["retval"]
        if api == 1:
            key = stable_fd(a, b, 1)
            _need(op != 0, "Unclassified KFD ioctl may change the event namespace")
            typed = op <= 6 or 11 <= op <= 17
            if typed:
                _need(a.get("payload_observed") is True, "Original typed KFD arguments were not observed")
                request = (abi.get("acquire_vm_request") if op == 11 else abi.get("requests", {}).get(str(op))) if op <= 11 else a["request"]
                _need(a["request"] == request, "Typed KFD operation differs from the compiled UAPI request")
            if result != 0:
                continue
            if typed:
                _need(b.get("payload_observed") is True and b.get("output_valid") is True,
                    "Successful typed KFD output was not observed")
            values = b["values"]
            if op == 11:
                _need(a["values"][:2] == values[:2], "ACQUIRE_VM original arguments changed")
                associated = a.get("associated_fd_identity", {})
                render_row = dict(seq=a["seq"], fd_identity=associated)
                render_key = fd_key(render_row, 2)
                _need(values[0] == render_key[0] and associated == b.get("associated_fd_identity"),
                    "ACQUIRE_VM actual render FD identity differs")
                old = acquires.get((key, values[1]))
                _need(old is None or old[0] == render_key, "ACQUIRE_VM changed the actual render FD binding")
                acquires[(key, values[1])] = (render_key, a, b)
            elif op in (9, 10, 15, 16):
                _need(type(startup_cutoff) is int and b["ns"] < startup_cutoff,
                    "KFD runtime/queue/scratch control is not proven before all target epochs")
                if op == 15:
                    mask = _int(abi.get("runtime_enable_mask"), "Runtime enable mask", 1) | _int(abi.get("runtime_ttmp_mask"), "Runtime TTMP mask", 1)
                    _need(a["values"][0:2] == values[0:2] and a["values"][1] & ~mask == 0
                        and (not a["values"][1] & abi["runtime_enable_mask"] or a["values"][0] > 0),
                        "Runtime-enable public inputs are not supported")
                elif op == 16:
                    _need(a["values"][:3] == values[:3] and values[0] > 0 and values[1] > 0 and values[2] == 0
                        and (key, values[1]) in acquires, "Scratch virtual-address startup inputs are not supported")
                startup_controls.append(dict(call_id=a["call_id"], operation=op, input_values=deepcopy(a["values"]),
                    output_values=deepcopy(values), actual_exit_ns=b["ns"], gpu_backing_zero_claimed=False))
            elif op == 14:
                _need(a["values"][0] == 2**64 - 1 and values[0] in (0, 1),
                    "XNACK operation is not the public -1 read-only query")
            elif op == 17:
                _need(a["values"][0] == values[0] and values[0] > 0 and a["values"][2] == values[2] == 0
                    and any(node.get("gpu_id") == values[0] for node in source["device_nodes"]),
                    "Available-memory query does not match an actual GPU ID/public input")
                memory_queries.append(dict(call_id=a["call_id"], gpu_id=values[0], available_bytes=values[1],
                    actual_exit_ns=b["ns"], used_for_workspace_bytes=False))
            elif op == 3:
                _need([a["values"][i] for i in (0, 1, 4, 5)] == [values[i] for i in (0, 1, 4, 5)],
                    "GPU allocation original request changed")
                handle = values[2]
                identity = key, handle
                _need(handle > 0 and identity not in allocations, "Allocation handle is missing or reused while live")
                _need(values[1] > 0 and values[0] + values[1] <= 2**64, "Allocation range is invalid")
                generations[identity] += 1
                allocation = dict(kfd_key=key, handle=handle, generation=generations[identity],
                    address=values[0], bytes=values[1], mmap_offset=values[3], gpu_id=values[4], flags=values[5],
                    enter=a, exit=b, freed=None)
                allocations[identity] = allocation
                all_allocations.append(allocation)
            elif op == 4:
                _need(a["values"] == values, "FREE_MEMORY input-only handle changed across the original call")
                allocation = allocations.pop((key, a["values"][0]), None)
                _need(allocation is not None, "Unobserved allocation/free generation")
                allocation["freed"] = a
            elif op == 1:
                _need(a["values"][2:5] == values[2:5], "CREATE_EVENT original request changed")
                incoming = a["values"][0]
                if incoming:
                    page = allocations.get((key, incoming))
                    _need(page is not None, "CREATE_EVENT page input has no live original allocation handle")
                    _need(key not in pages or pages[key] is page, "KFD event page was rebound")
                    pages[key] = page
                event_id = values[5]
                identity = key, event_id
                _need(event_id > 0 and identity not in events, "Hardware event ID is reused while live")
                event = dict(event_id=event_id, slot=values[6], output_page_offset=values[0],
                    event_type=values[2], page=pages.get(key), kfd_key=key, enter=a, exit=b, destroyed=None)
                events[identity] = event
                all_events.append(event)
            elif op == 2:
                _need(a["values"] == values, "DESTROY_EVENT input-only event ID/pad changed across the original call")
                event = events.pop((key, a["values"][0]), None)
                if event is not None:
                    event["destroyed"] = a
            elif op in (12, 13):
                _need(a["values"][:2] == values[:2] and values[1] == 0 and (key, values[0]) in events,
                    "SET/RESET_EVENT does not refer to a proven live same-namespace event")
        elif api in (2, 3) and result != -1:
            _need(0 <= result < 2**64 and 0 < a["length"] <= 2**64 - result, "Successful mmap extent is invalid")
            for old in mappings:
                if old["invalidated"] is None and _overlap(old["address"], old["bytes"], result, a["length"]):
                    old["invalidated"] = a
            mapping = dict(address=result, bytes=a["length"], enter=a, exit=b, invalidated=None, fd_key=None)
            kind = a.get("fd_identity", {}).get("kind")
            if kind in (1, 2):
                mapping["fd_key"] = stable_fd(a, b, kind)
            mappings.append(mapping)
        elif api == 4 and result == 0:
            for old in mappings:
                if old["invalidated"] is None and _overlap(old["address"], old["bytes"], a["address"], a["length"]):
                    old["invalidated"] = a
        elif api == 8 and result != -1:
            for old in mappings:
                if (old["invalidated"] is None and (_overlap(old["address"], old["bytes"], a["address"], a["length"])
                        or _overlap(old["address"], old["bytes"], result, a["values"][0]))):
                    old["invalidated"] = a  # Moved mappings are intentionally not registered as event pages.
    return dict(pid=pid, abi=abi, pairs=pairs, events=all_events, allocations=all_allocations,
        acquires=acquires, mappings=mappings, fd_mutations=fd_mutations, startup_controls=startup_controls,
        memory_queries=memory_queries)


def _namespace(control, source):
    _need(isinstance(control, dict), "Independent cold ordinary-signal namespace control is missing")
    replay = _replay(control.get("kfd_report"), control.get("source_binding"),
        _startup_cutoff(control.get("native_report", {}), control.get("target_epochs")))
    _need(_source(control["source_binding"], replay["abi"]) == _source(source, source["abi"]),
        "Namespace control uses different actual binaries, headers, driver or kernel")
    native = control.get("native_report", {})
    _need(native.get("pid") == replay["pid"], "Namespace control PID differs")
    create, query, _, event_id = _signal_query(_native(native), control.get("create_cid"), control.get("event_query_cid"))
    payload = create[2]["raw_hsa_signal_payload"]
    _need(payload.get("initial_value") == 0 and payload.get("num_consumers") == 0,
        "Namespace control is not an ordinary all-consumer signal")
    matches = [event for event in replay["events"] if event["enter"]["call_id"] == control.get("kfd_create_event_call_id")
        and event["event_id"] == event_id and event["event_type"] == replay["abi"].get("signal_event_type")
        and event["enter"]["tid"] == create[1]["tid"]
        and create[1]["ns"] <= event["enter"]["ns"] <= event["exit"]["ns"] <= create[2]["ns"]]
    _need(len(matches) == 1 and query[1]["ns"] > create[2]["ns"],
        "Public get_event_id is not causally tied to one cold original KFD CREATE_EVENT")
    return dict(process_id=replay["pid"], create_cid=control["create_cid"], event_query_cid=control["event_query_cid"],
        kfd_create_event_call_id=matches[0]["enter"]["call_id"], actual_event_id=event_id)


def audit_kfd_events(report, *, native_report, event_observations, source_binding, namespace_control, target_epochs=None):
    result: dict[str, Any] = dict(status="PARTIAL", complete_workspace=False, hardware_validated=False,
        scope="actual_gtt_event_page_backing_at_public_hsa_queries", issues=[], event_proofs=[], gtt_pages=[])
    try:
        replay = _replay(report, source_binding, _startup_cutoff(native_report, target_epochs))
        _need(native_report.get("pid") == replay["pid"], "Native/KFD process identity differs")
        result["namespace_control"] = _namespace(namespace_control, source_binding)
        calls = _native(native_report)
        _need(isinstance(event_observations, list) and 0 < len(event_observations) <= 4096,
            "Bounded actual HSA query locators are required")
        abi, pages, seen = replay["abi"], {}, set()
        for observed in event_observations:
            create, query, handle, event_id = _signal_query(calls, observed["create_cid"], observed["event_query_cid"])
            _need(query[2]["cid"] not in seen, "Duplicate HSA event observation")
            seen.add(query[2]["cid"])
            before, after = (_int(observed.get(key), "Actual KFD query cursor")
                for key in ("kfd_cursor_before", "kfd_cursor_after"))
            _need(before <= after <= report["last_sequence"], "KFD query cursor lies outside the raw prefix")
            matches = [event for event in replay["events"] if event["event_id"] == event_id
                and event["event_type"] == abi.get("signal_event_type")
                and event["exit"]["ns"] <= query[1]["ns"] and event["exit"]["seq"] <= before
                and (event["destroyed"] is None or event["destroyed"]["ns"] > query[2]["ns"])]
            _need(len(matches) == 1, "Public event ID does not identify one active KFD namespace/generation")
            event = matches[0]
            page = event["page"]
            _need(page is not None and page["exit"]["ns"] <= event["enter"]["ns"]
                and (page["freed"] is None or page["freed"]["ns"] > query[2]["ns"]), "Event-page backing was absent, freed or reused")
            _need(query[2]["ns"] <= report["snapshot_ns"], "Public query occurs after the KFD prefix snapshot")
            _need(all(row["ns"] <= query[1]["ns"] for row in report["events"][:before]),
                "KFD before-query cursor includes a later record")
            memory_types = sum(abi[key] for key in ("gtt_flag", "vram_flag", "userptr_flag", "doorbell_flag", "mmio_flag"))
            _need(page["flags"] & memory_types == abi["gtt_flag"], "Event page is not positively GTT-only backing")
            acquire = replay["acquires"].get((page["kfd_key"], page["gpu_id"]))
            _need(acquire is not None and acquire[2]["ns"] <= page["enter"]["ns"], "Original GPU_ID to render-FD ACQUIRE_VM is absent")
            render, _, _ = acquire
            topology = [node for node in source_binding["device_nodes"] if node.get("gpu_id") == page["gpu_id"]
                and (node.get("rdev_major"), node.get("rdev_minor")) == (
                    acquire[1]["associated_fd_identity"].get("rdev_major"), acquire[1]["associated_fd_identity"].get("rdev_minor"))
                and isinstance(node.get("pci_bus_id"), str) and re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", node["pci_bus_id"])]
            _need(len(topology) == 1, "Actual render-node GPU/PCI topology does not match ACQUIRE_VM")
            maps = [mapping for mapping in replay["mappings"] if mapping["exit"]["ns"] <= query[1]["ns"]
                and mapping["exit"]["seq"] <= before
                and (mapping["invalidated"] is None or mapping["invalidated"]["ns"] > query[2]["ns"])
                and ((mapping["fd_key"] == render and mapping["enter"]["offset"] == page["mmap_offset"])
                    or (mapping["fd_key"] == page["kfd_key"] and mapping["enter"]["offset"] == event["output_page_offset"]))]
            maps = [mapping for mapping in maps if mapping["bytes"] >= page["bytes"]
                and mapping["enter"]["flags"] & abi["map_shared"]
                and mapping["enter"]["protection"] & (abi["prot_read"] | abi["prot_write"]) == (abi["prot_read"] | abi["prot_write"])
                and event["slot"] * 8 + 8 <= page["bytes"]]
            _need(len(maps) == 1, "No unique live same-device mmap covers the complete GTT event page/slot")
            mapping = maps[0]
            for start, end, first, last in replay["fd_mutations"]:
                _need(not (first <= page["kfd_key"][0] <= last and event["exit"]["ns"] <= start["ns"] <= query[2]["ns"]),
                    "KFD fd namespace was closed or rebound before the query")
                _need(not (first <= render[0] <= last and acquire[2]["ns"] <= start["ns"] <= mapping["exit"]["ns"]),
                    "Render FD was closed or rebound between ACQUIRE_VM and mmap")
            _need(not any(a["seq"] > before and b["seq"] <= after and a["api"] == 1
                and a["operation"] in (1, 2, 3, 4, 11) for a, b in replay["pairs"]),
                "Event namespace/backing changed inside the public-query cursor window")
            page_id = (page["kfd_key"], page["handle"], page["generation"])
            proof = dict(signal_handle=handle, create_cid=create[2]["cid"], event_query_cid=query[2]["cid"],
                event_id=event_id, kfd_create_event_call_id=event["enter"]["call_id"], slot_index=event["slot"],
                slot_address=mapping["address"] + event["slot"] * 8, slot_bytes=8,
                allocate_call_id=page["enter"]["call_id"], allocate_handle=page["handle"],
                allocation_generation=page["generation"], mmap_call_id=mapping["enter"]["call_id"],
                backing_class="driver_gtt_event_page", logical_signal_destruction_claimed=False, host_cost_is_zero=False)
            result["event_proofs"].append(proof)
            pages[page_id] = dict(address=mapping["address"], bytes=page["bytes"], gpu_id=page["gpu_id"],
                allocate_handle=page["handle"], allocation_generation=page["generation"],
                allocate_call_id=page["enter"]["call_id"], mmap_call_id=mapping["enter"]["call_id"],
                classification="GTT_system_memory", charged_to_gpu_requested_storage=False)
        result["gtt_pages"] = list(pages.values())
        result["startup_control_obligations"] = replay["startup_controls"]
        result["observed_available_memory_queries"] = replay["memory_queries"]
        result["gtt_page_bytes"] = sum(page["bytes"] for page in result["gtt_pages"])
        result["status"] = "KFD_EVENT_GTT_BACKING_PROVEN"
    except (KfdEvidenceError, KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        result["issues"].append(str(error))
        result["event_proofs"] = []
        result["gtt_pages"] = []
    return deepcopy(result)
