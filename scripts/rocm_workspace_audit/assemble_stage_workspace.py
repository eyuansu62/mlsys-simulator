"""Assemble pinned per-rank Workspace evidence; unknowns never become zero.

Input paths and source files are read as data only. Component status strings in
an acquisition are not accepted as audit results: native, Torch and optional PP
transport evidence are independently replayed before any canonical output.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import gc
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path
import struct
from typing import Any

from mlsys_sim.config import experiment_from_dict
from mlsys_sim.layout import rank_weight_bytes
from mlsys_sim.memory import MemoryPlanner
from mlsys_sim.profiles import shape_key
from mlsys_sim.schema import SimulationError
from mlsys_sim.workspace import ATTRIBUTION, validate_workspace_dataset
from native_backing_audit import audit_native_backing
from workspace_stage_domain import validate_stage_workspace_domain
from workspace_torch_audit import audit_torch_workspace
from workspace_transport_audit import audit_workspace_transport

_MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
_MAX_JSON_BYTES = 2 * 1024 * 1024 * 1024


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError(label + " requires a bounded integer")
    return value


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked evidence/source file: " + str(path))
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Inputs:
    """Keep raw acquisition hashes separate from shared source/binary hashes."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.raw: dict[str, str] = {}
        self.shared: dict[str, str] = {}
        self.capture: dict[str, str] = {}

    def path(self, supplied: Any) -> Path:
        if not isinstance(supplied, str) or not supplied:
            raise ValueError("Acquisition path is missing")
        path = Path(supplied)
        if not path.is_absolute():
            path = self.root / path
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Acquisition path escapes its actual session directory: " + str(path))
        return path

    def shared_file(self, supplied: Any, expected: Any) -> None:
        if not isinstance(supplied, str) or not isinstance(expected, str):
            raise ValueError("Source/binary path and SHA are required")
        actual = _sha(Path(supplied))
        self.shared[supplied] = actual
        if actual != expected:
            raise ValueError("Executed source/binary SHA mismatch: " + supplied)

    def shared_alias_file(self, supplied: Any, expected: Any) -> Path:
        """Verify a loader/compiler spelling and its actual resolved file bytes."""
        if not isinstance(supplied, str) or not Path(supplied).is_absolute():
            raise ValueError("Actual loaded/source alias must be an absolute path")
        resolved = Path(supplied).resolve(strict=True)
        self.shared_file(str(resolved), expected)
        self.shared[supplied] = self.shared[str(resolved)]
        return resolved

    def read(self, path: Path, *, capture: bool = True) -> Any:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked raw evidence: " + str(path))
        if path.stat().st_size > _MAX_INPUT_BYTES:
            raise ValueError("Raw input exceeds bounded file size: " + str(path))
        with path.open("rb") as stream:
            encoded = stream.read(_MAX_INPUT_BYTES + 1)
        if len(encoded) > _MAX_INPUT_BYTES:
            raise ValueError("Raw input grew beyond its bounded file size: " + str(path))
        digest = hashlib.sha256(encoded).hexdigest()
        self.raw[str(path.resolve())] = digest
        if capture:
            self.capture[str(path.resolve())] = digest
        if path.suffix == ".gz":
            with gzip.GzipFile(fileobj=io.BytesIO(encoded), mode="rb") as decoded_stream:
                raw = decoded_stream.read(_MAX_JSON_BYTES + 1)
        else:
            raw = encoded
        if len(raw) > _MAX_JSON_BYTES:
            raise ValueError("Raw input exceeds bounded decoded size: " + str(path))

        def unique(pairs: Any) -> dict:
            value: dict = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("Duplicate evidence JSON key: " + key)
                value[key] = item
            return value

        def nonfinite(value: str) -> None:
            raise ValueError("Nonfinite evidence JSON: " + value)

        return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def _source_gate(plan: dict, inputs: Inputs) -> None:
    worker = Path(plan["worker_source"])
    if worker.name != "worker_workspace_capture.py":
        raise ValueError("Unsupported Worker observer source filename")
    inputs.shared_file(str(worker), plan["worker_source_sha256"])
    inputs.shared_file(str(worker.with_name("workspace_stage_session.py")), plan["launcher_source_sha256"])
    for name in ("native_probe_library", "torch_probe_library"):
        inputs.shared_file(plan[name], plan[name + "_sha256"])


def _constructor_source(path: Path, qualified: str) -> str:
    """Use AST/tokenization to read a known method; never import producer code."""
    constructor = "vllm.distributed.device_communicators.pynccl.PyNcclCommunicator.__init__"
    shutdown = "vllm.v1.executor.multiproc_executor.WorkerProc.shutdown"
    worker_prefix = "worker_workspace_capture.WorkspaceWorker."
    worker_methods = {worker_prefix + name for name in
                      ("_ws_capture_host_signals", "mlsys_workspace_close", "_ws_capture_final_kfd_prefix")}
    host_capture = "hsa_signal_backing_capture.capture_hsa_signal_backings"
    pg_prefix = "torch.distributed.distributed_c10d."
    pg_functions = {pg_prefix + name for name in ("all_gather_into_tensor", "isend", "irecv")}
    if qualified not in {constructor, shutdown, host_capture} | worker_methods | pg_functions:
        raise ValueError("Unknown communication constructor entrypoint")
    text = path.read_text()
    tree = ast.parse(text)
    if qualified in {constructor, shutdown} | worker_methods:
        class_name, method_name = qualified.split(".")[-2:]
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        methods = [node for cls in classes for node in cls.body
                   if isinstance(node, ast.FunctionDef) and node.name == method_name]
    else:
        methods = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == qualified.rsplit(".", 1)[-1]]
    if len(methods) != 1:
        raise ValueError("Pinned communicator constructor is not unique in its actual source")
    method = methods[0]
    first = min([method.lineno, *[node.lineno for node in method.decorator_list]]) - 1
    block = "".join(inspect.getblock(text.splitlines(keepends=True)[first:]))
    return hashlib.sha256(block.encode()).hexdigest()


def _verify_callable_source(source: dict, inputs: Inputs) -> None:
    reference = source["source_reference"]
    inputs.shared_file(reference, source["file_sha256"])
    if _constructor_source(Path(reference), source["qualified_name"]) != source["source_sha256"]:
        raise ValueError("Actual communication function body SHA differs")


def _kfd_source_gate(plan: dict, worker: dict, evidence: dict, inputs: Inputs) -> None:
    """Authenticate passive/cold-control files without interpreting ID namespaces."""
    def same(actual: Any, expected: Any, label: str) -> None:
        if _json(actual) != _json(expected):
            raise ValueError("KFD evidence differs: " + label)

    def raw(ref: Any, *, shared: bool = False, json_value: bool = True) -> Any:
        if not isinstance(ref, dict):
            raise ValueError("KFD original artifact reference is missing")
        path = Path(ref["path"]) if shared else inputs.path(ref["path"])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("KFD original artifact must be an actual absolute regular file")
        same(path.stat().st_size, _integer(ref.get("size_bytes"), "KFD artifact bytes", 1), "artifact bytes")
        if path.stat().st_size > _MAX_INPUT_BYTES:
            raise ValueError("KFD original artifact exceeds the bounded input size")
        if json_value:
            value = inputs.read(path, capture=not shared)
        else:
            inputs.raw[str(path.resolve())] = _sha(path)
            if not shared:
                inputs.capture[str(path.resolve())] = inputs.raw[str(path.resolve())]
            value = None  # Preserve control JSONL as bytes, never synthesize a JSON report.
        same(inputs.raw[str(path.resolve())], ref["sha256"], "original artifact SHA")
        return value

    def source(binding: Any, report: dict, *, primary: bool) -> None:
        if not isinstance(binding, dict) or not isinstance(binding.get("artifacts"), dict):
            raise ValueError("KFD actual source artifacts are missing")
        names = ("probe_source", "native_probe_source", "probe_library", "hsa_runtime", "libc", "kfd_header", "drm_header")
        paths = {}
        for name in names:
            row = binding["artifacts"][name]
            same(row["sha256"], binding[name + "_sha256"], "source artifact " + name)
            path = inputs.shared_alias_file(row["path"], row["sha256"])
            for alias in ("loaded_path", "realpath"):
                if alias in row:
                    inputs.shared_alias_file(row[alias], row["sha256"])
            same(path.stat().st_size, _integer(row.get("size_bytes"), "KFD source artifact bytes", 1), "source bytes")
            paths[name] = path
        same(binding.get("abi"), report.get("abi"), "original compiled ABI object")
        inputs.shared_alias_file(report["compiled_source_path"], binding["probe_source_sha256"])
        if primary:
            same(binding["probe_library_sha256"], plan["native_probe_library_sha256"], "actual combined Native/KFD DSO")
            inputs.shared_alias_file(plan["native_probe_library"], binding["probe_library_sha256"])
            same(binding["hsa_runtime_sha256"], plan["host_signal_provider"]["rocr_sha256"], "actual queried HSA Runtime")
        modules = report.get("libc_bindings")
        if not isinstance(modules, dict) or not modules:
            raise ValueError("KFD actual libc function bindings are missing")
        observed = False
        for row in modules.values():
            if row.get("address") is None:
                continue  # Untouched symbols are not actively resolved for an audit.
            _integer(row.get("address"), "KFD actual libc function address", 1)
            _integer(row.get("module_base"), "KFD actual libc module base", 1)
            same(row.get("dladdr_success"), True, "actual resolved libc module")
            inputs.shared_alias_file(row["module_path"], binding["libc_sha256"])
            observed = True
        if not observed:
            raise ValueError("KFD tape has no actual resolved libc binding")
        build = raw(binding.get("build_receipt_artifact"), shared=True)
        if type(build.get("returncode")) is not int or build["returncode"] != 0:
            raise ValueError("KFD combined observer build did not actually succeed")
        same(build.get("binary_sha256"), binding["probe_library_sha256"], "actual built DSO bytes")
        hashes = build.get("sources_sha256")
        if not isinstance(hashes, dict):
            raise ValueError("KFD combined observer build source hashes are missing")
        for name in ("probe_source", "native_probe_source"):
            expected = binding[name + "_sha256"]
            if hashes.get(str(paths[name])) != expected and hashes.get(paths[name].name) != expected:
                raise ValueError("KFD combined build source differs: " + name)
        observations = binding.get("system_observation_artifacts")
        if observations is None and not primary:
            return  # Older independently captured cold metadata remains a shared source binding.
        if not isinstance(observations, list) or not observations:
            raise ValueError("KFD current kernel/sysfs byte observations are missing")
        files: dict[str, bytes] = {}
        unames = []
        for row in observations:
            # Kernel/header/topology facts can have identical bytes in a fresh
            # Runtime; retain their hashes without treating them as model tape reuse.
            doc = raw(row.get("snapshot_artifact"), shared=True)
            same(doc.get("schema_version"), 1, "system observation schema")
            if row.get("source") == "os.uname":
                same(doc.get("kind"), "actual_uname_result", "uname observation kind")
                unames.append(doc)
                continue
            same(doc.get("kind"), "actual_system_file_bytes", "system-file observation kind")
            name = row.get("original_path")
            if not isinstance(name, str) or not Path(name).is_absolute() or name in files:
                raise ValueError("KFD original system-file path is missing or repeated")
            same(doc.get("original_path"), name, "original system-file identity")
            data = bytes.fromhex(doc["contents_hex"])
            for value in (row, doc):
                same(_integer(value.get("contents_size_bytes"), "Observed system-file bytes"), len(data), "system-file length")
                same(value.get("contents_sha256"), hashlib.sha256(data).hexdigest(), "system-file bytes SHA")
            files[name] = data
        if len(unames) != 1:
            raise ValueError("KFD actual uname observation is missing or ambiguous")
        same(unames[0].get("release"), binding.get("kernel_release"), "actual kernel release")
        notes = [data for name, data in files.items() if name.endswith("/module/amdgpu/notes/.note.gnu.build-id")]
        if len(notes) != 1:
            raise ValueError("KFD actual amdgpu build note is missing or ambiguous")
        note, offset, ids = notes[0], 0, []
        while offset < len(note):
            if len(note) - offset < 12:
                raise ValueError("KFD actual module ELF note is truncated")
            namesz, descsz, kind = struct.unpack_from("<III", note, offset)
            begin = offset + 12
            desc = begin + ((namesz + 3) & ~3)
            end = desc + ((descsz + 3) & ~3)
            if end > len(note):
                raise ValueError("KFD actual module ELF note exceeds its bytes")
            if note[begin:begin + namesz].rstrip(b"\0") == b"GNU" and kind == 3:
                if descsz != 20:
                    raise ValueError("KFD actual module build ID has unsupported extent")
                ids.append(note[desc:desc + descsz].hex())
            offset = end
        same(ids, [binding.get("amdgpu_module_build_id")], "actual amdgpu module build ID")
        devices = []
        for name in sorted(files):
            if "/class/kfd/kfd/topology/nodes/" not in name or not name.endswith("/gpu_id"):
                continue
            gpu_id = int(files[name].decode().strip())
            if not gpu_id:
                continue
            properties = dict(line.split(" ", 1) for line in files[str(Path(name).with_name("properties"))].decode().splitlines())
            minor = int(properties["drm_render_minor"])
            prefix = name.split("/class/kfd/", 1)[0] + "/class/drm/renderD" + str(minor)
            major, actual_minor = map(int, files[prefix + "/dev"].decode().strip().split(":"))
            same(actual_minor, minor, "actual render-node minor")
            uevent = dict(line.split("=", 1) for line in files[prefix + "/device/uevent"].decode().splitlines() if "=" in line)
            devices.append({"gpu_id": gpu_id, "rdev_major": major, "rdev_minor": minor,
                            "pci_bus_id": uevent["PCI_SLOT_NAME"].lower()})
        same(devices, binding.get("device_nodes"), "actual kernel GPU/render topology")

    original = worker["host_signal_evidence"]["capture"].get("kfd_event_evidence")
    final_keys = {"final_report_artifact", "final_snapshot_observation"}
    diagnostic_only = original is None and set(evidence) <= final_keys
    if not diagnostic_only:
        if (type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1
                or evidence.get("kind") != "passive_kfd_event_evidence" or not isinstance(original, dict)):
            raise ValueError("KFD final evidence lacks its original query capture refs")
        same({key: value for key, value in evidence.items() if key not in final_keys}, original,
             "original passive query evidence")
    report = raw(evidence.get("final_report_artifact"))
    same(report.get("pid"), worker["process_id"], "worker final passive-tape process")
    observation = evidence.get("final_snapshot_observation")
    if not isinstance(observation, dict):
        raise ValueError("KFD final prefix lacks its actual source-bound export observation")
    for key, expected in {"process_id": worker["process_id"], "thread_id": worker["runtime_shutdown"]["thread_id"],
        "native_stop_returncode": worker["native_sdk_stop_status"], "write_returncode": 0, "prefix_only": True,
        "method": "passive_KFD_registry_prefix_after_natural_shutdown_and_successful_SDK_stop",
        "complete_workspace": False, "hardware_validated": False}.items():
        same(observation.get(key), expected, "actual final-prefix observation " + key)
    src = observation["source"]
    same(src.get("qualified_name"), "worker_workspace_capture.WorkspaceWorker._ws_capture_final_kfd_prefix", "export method")
    same(src.get("source_reference"), plan["worker_source"], "export method file")
    same(src.get("file_sha256"), plan["worker_source_sha256"], "export method file SHA")
    _verify_callable_source(src, inputs)
    if diagnostic_only:
        return  # N/A/event-zero must still pass the independent actual signal replay.
    for key in ("report_artifact", "pre_query_report_artifact", "post_query_report_artifact"):
        before = raw(evidence.get(key))
        same(before.get("pid"), worker["process_id"], "worker query-prefix process")
        same(before.get("abi"), report.get("abi"), "query/final compiled ABI")
        inputs.shared_alias_file(before["compiled_source_path"], evidence["source_binding"]["probe_source_sha256"])
    source(evidence.get("source_binding"), report, primary=True)
    control = evidence.get("namespace_control")
    if not isinstance(control, dict):
        raise ValueError("KFD independent cold namespace control is missing")
    same(raw(evidence.get("namespace_control_artifact"), shared=True), control, "original cold manifest")
    config = plan["host_signal_provider"]
    same(evidence["namespace_control_artifact"]["sha256"], config.get("namespace_control_sha256"), "planned cold manifest")
    inputs.shared_alias_file(config["namespace_control_path"], config["namespace_control_sha256"])
    for key, expected in {"schema_version": 1, "kind": "cold_hsa_kfd_namespace_control_manifest"}.items():
        same(control.get(key), expected, "cold manifest " + key)
    native = raw(control.get("native_report"), shared=True)
    cold = raw(control.get("kfd_report"), shared=True)
    raw(control.get("control_jsonl"), shared=True, json_value=False)
    raw(control.get("build_receipt"), shared=True)
    same(native.get("pid"), cold.get("pid"), "cold Native/KFD process")
    source(control.get("source_binding"), cold, primary=False)
    # CID, KFD call ID, slot index, GPU ID and HSA agent identities stay in
    # their original namespaces. Only the independent replay may join them.


def _host_signal_source_gate(plan: dict, worker: dict, inputs: Inputs, *, inference_thread: int) -> None:
    """Authenticate optional raw query artifacts; resource ownership stays independent."""
    config, evidence = plan.get("host_signal_provider"), worker.get("host_signal_evidence")
    if config is None and evidence is None:
        return
    if (not isinstance(config, dict) or not isinstance(evidence, dict) or plan.get("mode") != "pp2"
            or plan.get("native_finalization", "worker_shutdown") != "worker_shutdown"):
        raise ValueError("Host signal evidence requires its actual planned PP2 worker-shutdown provider")

    def same(actual: Any, expected: Any, label: str) -> None:
        if _json(actual) != _json(expected):
            raise ValueError("Host signal evidence differs: " + label)

    def artifact(value: Any, *, raw: bool = False, embedded: Any = None, size_key: str = "size_bytes") -> Path:
        if not isinstance(value, dict):
            raise ValueError("Host signal evidence artifact is missing")
        if raw:
            path = inputs.path(value["path"])
            loaded = inputs.read(path)
            same(inputs.raw[str(path.resolve())], value["sha256"], "raw artifact SHA")
            same(loaded, embedded, "raw artifact contents")
        else:
            path = inputs.shared_alias_file(value["path"], value["sha256"])
        same(path.stat().st_size, _integer(value.get(size_key), "Host signal artifact bytes", 1), "artifact size")
        return path

    pid, device = worker["process_id"], worker["logical_device_index"]
    for key, expected in {"schema_version": 1, "kind": "public_hsa_signal_backing_worker_evidence",
        "process_id": pid, "thread_id": inference_thread, "complete_workspace": False,
        "hardware_validated": False}.items():
        same(evidence.get(key), expected, "worker envelope " + key)
    sidecar = inputs.read(inputs.path(worker["startup_snapshot"]).with_name("host-signal-worker-evidence.json"))
    extensions = {"after_natural_shutdown_live", "kfd_event_evidence"}
    same({key: value for key, value in sidecar.items() if key not in extensions},
         {key: value for key, value in evidence.items() if key not in extensions},
         "original pre-shutdown worker envelope")
    if "kfd_event_evidence" in sidecar:
        current = evidence.get("kfd_event_evidence")
        if not isinstance(current, dict):
            raise ValueError("Host signal original KFD extension is missing")
        same(sidecar["kfd_event_evidence"], {key: value for key, value in current.items()
             if key not in {"final_report_artifact", "final_snapshot_observation"}}, "original KFD extension")
    capture = evidence.get("capture")
    if not isinstance(capture, dict):
        raise ValueError("Host signal evidence lacks its original public capture")
    artifact(evidence.get("capture_artifact"), raw=True, embedded=capture, size_key="bytes")
    for key in ("process_id", "thread_id", "source_scope", "source_fence"):
        same(capture.get(key), evidence.get(key), "capture/envelope " + key)
    scope, fence = evidence.get("source_scope"), evidence.get("source_fence")
    if not isinstance(scope, dict) or not isinstance(fence, dict):
        raise ValueError("Host signal source scope/fence is missing")
    epochs = sorted({row["native_epoch_id"] for row in worker["records"] if row["cohort"] == "formal"})
    for key, expected in {"kind": "post_formal_pre_natural_shutdown_public_signal_queries",
        "session_id": plan["session_id"], "process_id": pid, "thread_id": inference_thread,
        "target_device_index": device, "native_clock": "CLOCK_MONOTONIC_RAW", "formal_native_epochs": epochs,
        "worker_source": worker["source_bindings"],
        "query_started_monotonic_raw_ns": evidence.get("query_started_monotonic_raw_ns")}.items():
        same(scope.get(key), expected, "actual source scope " + key)
    for key, expected in {"method": "existing_post_cohort_last_target_device_synchronize", "process_id": pid,
        "thread_id": inference_thread, "target_device_index": device, "native_clock": "CLOCK_MONOTONIC_RAW",
        "native_epoch_id": 9001}.items():
        same(fence.get(key), expected, "actual source fence " + key)
    for value, name in ((scope, "_ws_capture_host_signals"), (fence, "mlsys_workspace_close")):
        source = value["source"]
        same(source.get("qualified_name"), "worker_workspace_capture.WorkspaceWorker." + name, "worker method")
        same(source.get("source_reference"), plan["worker_source"], "worker method file")
        same(source.get("file_sha256"), plan["worker_source_sha256"], "worker method file SHA")
        _verify_callable_source(source, inputs)

    # These are original windows, not correlations or ownership claims. The
    # synchronization auditor joins them to unique typed SDK calls/context.
    for value, prefix in ((fence, ""), (evidence, "query_"), (capture, "capture_")):
        start = _integer(value.get(prefix + "started_monotonic_raw_ns"), "Host query RAW start", 1)
        end = _integer(value.get(prefix + "ended_monotonic_raw_ns"), "Host query RAW end", 1)
        if start > end:
            raise ValueError("Host signal RAW window is reversed")
    for value in (fence, evidence):
        before = _integer(value.get("native_cursor_before"), "Host query before cursor")
        after = _integer(value.get("native_cursor_after"), "Host query after cursor")
        if before > after:
            raise ValueError("Host signal cursor window is reversed")
    for phase in ("before", "after"):
        artifact(capture.get(phase + "_live_artifact"), raw=True, embedded=capture.get(phase + "_live"))
    post = evidence.get("after_natural_shutdown_live")
    if not isinstance(post, dict):
        raise ValueError("Host signal evidence lacks the actual post-shutdown registry artifact")
    for key, expected in {"kind": "post_natural_shutdown_observed_signal_generations", "process_id": pid,
        "thread_id": worker["runtime_shutdown"]["thread_id"], "new_signal_queries_performed": False}.items():
        same(post.get(key), expected, "post-shutdown registry " + key)
    artifact(post, raw=True, embedded=post.get("observed_live"), size_key="bytes")

    for name in ("provider", "rocr", "abi_header"):
        inputs.shared_alias_file(config[name + "_path"], config[name + "_sha256"])
    artifact(capture.get("provider"))
    same(capture["provider"]["sha256"], config["provider_sha256"], "planned provider bytes")
    def table(key: str) -> list[dict]:
        rows = capture.get(key)
        if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
            raise ValueError("Host signal source/header/runtime artifact table is missing")
        for row in rows:
            artifact(row)
        return rows

    sources, headers, libraries = (table(key) for key in ("sources", "abi_headers", "runtime_dsos"))
    helper = [row for row in sources if Path(row["path"]).name == "hsa_signal_backing_capture.py"]
    cpp = [row for row in sources if Path(row["path"]).name == "hsa_signal_backing_probe.cpp"]
    if len(helper) != 1 or len(cpp) != 1:
        raise ValueError("Host signal helper and sibling CPP source are not unique")
    binding = capture["source_binding"]
    same(binding.get("qualified_name"), "hsa_signal_backing_capture.capture_hsa_signal_backings", "capture function")
    same(_constructor_source(Path(helper[0]["path"]), binding["qualified_name"]), binding.get("source_sha256"),
         "actual capture function body SHA")
    required_headers = {"amd_hsa_signal.h", "hsa.h", "hsa_ext_amd.h", "amd_hsa_common.h", "amd_hsa_queue.h"}
    names = [Path(row["path"]).name for row in headers]
    if not required_headers <= set(names) or any(names.count(name) != 1 for name in required_headers):
        raise ValueError("Host signal public ABI headers are missing or ambiguous")
    same(next(row["sha256"] for row in headers if Path(row["path"]).name == "amd_hsa_signal.h"),
         config["abi_header_sha256"], "planned ABI header bytes")
    module_hashes: dict[str, str] = {}
    for row in libraries:
        for path in (row["path"], row["loaded_path"]):
            inputs.shared_alias_file(path, row["sha256"])
            if path in module_hashes and module_hashes[path] != row["sha256"]:
                raise ValueError("Host signal runtime alias has conflicting bytes")
            module_hashes[path] = row["sha256"]
    if config["rocr_sha256"] not in module_hashes.values():
        raise ValueError("Host signal actual configured ROCr bytes are absent")
    init = capture.get("initialization")
    if init is not None or capture.get("compiler_source") is not None:
        if not isinstance(init, dict):
            raise ValueError("Host signal provider initialization is missing")
        compiled = capture["compiler_source"]
        artifact(compiled)
        same(compiled["sha256"], cpp[0]["sha256"], "actual compiled CPP versus sibling source bytes")
        inputs.shared_alias_file(init["compiled_source_path"], compiled["sha256"])
        bindings = init.get("api_bindings")
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError("Host signal actual API module bindings are missing")
        for name, row in bindings.items():
            same(row.get("symbol_name"), name, "resolved public API name")
            same(row.get("dladdr_success"), True, "resolved public API module")
            _integer(row.get("address"), "Public API function address", 1)
            _integer(row.get("module_base"), "Public API module base", 1)
            if row.get("module_path") not in module_hashes:
                raise ValueError("Host signal public API module lacks its actual DSO artifact")
    elif capture.get("records") or capture.get("ordinary_signal_control") is not None:
        raise ValueError("Host signal queried records lack their actual compiled provider source")
    kfd = evidence.get("kfd_event_evidence")
    if kfd is not None:
        if not isinstance(kfd, dict):
            raise ValueError("KFD worker evidence is malformed")
        _kfd_source_gate(plan, worker, kfd, inputs)


def _communication_source_gate(binding: dict, worker: dict, plan: dict, agent: int, inputs: Inputs) -> None:
    if binding.get("driver_source_reference") != plan["worker_source"] or binding.get("driver_source_sha256") != plan["worker_source_sha256"]:
        raise ValueError("Communication driver source is not the actual Workspace Worker")
    if _json(binding.get("loaded_rccl")) != _json(worker["loaded_rccl"]):
        raise ValueError("Communication library differs from the actual worker loaded-RCCL probe")
    rank = {**worker["placement"], "agent_handle": agent}
    if _json(binding.get("rank")) != _json(rank):
        raise ValueError("Communication binding rank/process/device differs from actual Worker")
    if ("natural_shutdown_evidence" in binding
            and _json(binding["natural_shutdown_evidence"]) != _json(worker.get("runtime_shutdown"))):
        raise ValueError("Communication natural shutdown evidence differs from actual Worker receipt")
    if ("host_signal_evidence" in binding
            and _json(binding["host_signal_evidence"]) != _json(worker.get("host_signal_evidence"))):
        raise ValueError("Communication host signal evidence differs from actual Worker envelope")
    ctors = {row["epoch_id"]: row for row in worker.get("communicator_constructors", [])}
    pg_scopes = {row["epoch_id"]: row for row in worker.get("communication_prewarm", [])}
    for row in binding.get("initialization_epochs", []):
        if row.get("kind") == "torch_process_group_prewarm":
            observed = pg_scopes.get(row.get("epoch_id"))
            if (not isinstance(observed, dict) or observed.get("status") != "COMPLETED"
                    or _json(row.get("pg_scope")) != _json(observed)):
                raise ValueError("ProcessGroup initialization differs from its actual successful prewarm observation")
            for key in ("epoch_id", "epoch_label", "entrypoint", "comm_only"):
                if _json(row.get(key)) != _json(observed.get(key)):
                    raise ValueError("ProcessGroup prewarm observation differs: " + key)
            _verify_callable_source(observed["source_binding"], inputs)
            continue
        observed = ctors.get(row.get("epoch_id"))
        if not isinstance(observed, dict) or observed.get("status") != "RETURNED":
            raise ValueError("Communication initialization has no actual successful constructor observation")
        for key in ("epoch_id", "epoch_label", "entrypoint", "comm_only", "communicator_handle", "world_size", "rank"):
            if _json(row.get(key)) != _json(observed.get(key)):
                raise ValueError("Communication constructor observation differs: " + key)
        source = observed["constructor_source"]
        reference = source["source_reference"]
        if binding.get("constructor_source_reference") != reference:
            raise ValueError("Communication constructor source reference differs")
        _verify_callable_source(source, inputs)
        if (source["source_sha256"] != binding.get("constructor_source_sha256")
                or ("constructor_source_file_sha256" in binding
                    and binding["constructor_source_file_sha256"] != source["file_sha256"])):
            raise ValueError("Actual communication constructor body SHA differs")
    inputs.shared_file(worker["loaded_rccl"]["path"], worker["loaded_rccl"]["library_sha256"])
    control = worker.get("stdthread_control")
    if control is not None or binding.get("stdthread_control") is not None:
        if not isinstance(control, dict) or _json(binding.get("stdthread_control")) != _json(control):
            raise ValueError("Communication std::thread control differs from actual Worker")
        actual_control = inputs.read(inputs.path(worker["startup_snapshot"]).with_name("stdthread-control.json"))
        if _json(control["control"]) != _json(actual_control):
            raise ValueError("Communication std::thread control differs from actual native CPU control")
        for field in ("header", "runtime_library"):
            inputs.shared_file(control[field]["path"], control[field]["sha256"])


def _natural_shutdown_gate(plan: dict, workers: list[dict], launcher: dict, inputs: Inputs) -> None:
    """Bind an MP final trace to the original Runtime teardown and actual exits."""
    if plan["mode"] == "single":
        return
    parent = launcher.get("natural_parent_shutdown", {})
    if parent.get("original_returned") is not True:
        raise ValueError("Multicard capture lacks successful natural parent shutdown")
    exits = parent.get("owned_worker_processes")
    if (not isinstance(exits, list) or len(exits) != len(workers)
            or {row.get("pid") for row in exits} != {worker["process_id"] for worker in workers}
            or any(row.get("alive") is not False or type(row.get("exit_code")) is not int
                   or row["exit_code"] != 0 for row in exits)):
        raise ValueError("Multicard natural shutdown has missing, live or unsuccessful actual worker exits")
    acknowledgements = {row["process_id"]: row for row in launcher["ranks"]}
    for worker in workers:
        pid = worker["process_id"]
        ack = acknowledgements[pid]
        if (ack.get("finalize_deferred") is not True or ack.get("capture_closed") is not False
                or worker.get("native_finalize_deferred") is not True):
            raise ValueError("Multicard close acknowledgement did not defer final native closure")
        path = inputs.path(worker["natural_shutdown_receipt_file"])
        receipt = inputs.read(path)
        if inputs.raw[str(path.resolve())] != worker["natural_shutdown_receipt_sha256"]:
            raise ValueError("Natural Worker shutdown receipt bytes changed")
        if (_json(receipt) != _json(worker.get("runtime_shutdown"))
                or receipt.get("process_id") != pid or receipt.get("original_returned") is not True
                or _json(receipt.get("source")) != _json(worker.get("natural_shutdown_source"))
                or receipt.get("scope") != "original_WorkerProc_shutdown_including_model_and_distributed_environment"):
            raise ValueError("Natural Worker shutdown receipt differs from actual final session")
        if receipt["source"].get("qualified_name") != "vllm.v1.executor.multiproc_executor.WorkerProc.shutdown":
            raise ValueError("Natural Worker shutdown source is not the original WorkerProc.shutdown")
        if worker.get("lifecycle_clock") != "CLOCK_MONOTONIC" or receipt.get("clock") != "CLOCK_MONOTONIC":
            raise ValueError("Natural Worker shutdown uses an unsupported lifecycle clock")
        saved = _integer(worker.get("deferred_close_saved_monotonic_ns"), "Deferred close time", 1)
        started = _integer(receipt.get("started_monotonic_ns"), "Natural shutdown start", 1)
        ended = _integer(receipt.get("ended_monotonic_ns"), "Natural shutdown end", 1)
        if not saved <= started <= ended:
            raise ValueError("Natural Worker shutdown predates the deferred close or has reversed bounds")
        if "native_clock" in receipt:
            if receipt["native_clock"] != "CLOCK_MONOTONIC_RAW":
                raise ValueError("Natural Worker shutdown native clock differs from the SDK clock")
            raw_start = _integer(receipt.get("started_monotonic_raw_ns"), "Native-clock shutdown start", 1)
            raw_end = _integer(receipt.get("ended_monotonic_raw_ns"), "Native-clock shutdown end", 1)
            if raw_start > raw_end:
                raise ValueError("Natural Worker shutdown native-clock bounds are reversed")
        _verify_callable_source(receipt["source"], inputs)


def _rank_budget(experiment: Any, placement: dict, torch_result: dict, native_result: dict,
                 rank_reservation: dict) -> dict:
    """Separate actual fixed padding/communication/transport from dynamic scratch."""
    values = torch_result.get("fixed_requested_bytes_by_category")
    if not isinstance(values, dict):
        raise ValueError("Fixed Torch storage categories are missing")
    allowed = {"weights", "kv_pool", "runtime", "graphs", "pp_buffers", "communication_buffers"}
    if set(values) - allowed:
        raise ValueError("Unknown fixed Torch category cannot be admitted")
    for name, value in values.items():
        _integer(value, "Fixed " + name)
    for name in ("weights", "kv_pool", "runtime"):
        _integer(values.get(name), "Required fixed " + name, 1)
    native_runtime = _integer(native_result.get("native_fixed_runtime_bytes"), "Native fixed Runtime")
    if "native_fixed_communication_bytes" in native_result:
        native_communication = _integer(native_result["native_fixed_communication_bytes"], "Native fixed communication")
    elif experiment.strategy.tensor_parallel_size == experiment.strategy.pipeline_parallel_size == 1:
        native_communication = 0  # Only a successful no-communicator audit permits this observed zero.
    else:
        raise ValueError("Multicard native communication storage remains unknown")
    mathematical = rank_weight_bytes(experiment, placement["stage_index"])
    observed_weights = values["weights"]
    padding = max(0, observed_weights - mathematical)
    runtime = values["runtime"] + native_runtime + padding
    communication = values.get("communication_buffers", 0) + native_communication
    static_pp = values.get("pp_buffers", 0)
    if experiment.strategy.pipeline_parallel_size > 1:
        dynamic_pp = _integer(torch_result.get("pp_transport_peak_bytes"), "Observed PP transport peak")
    else:
        dynamic_pp = 0
    pp = static_pp + dynamic_pp
    memory = experiment.strategy.memory
    checks = {"runtime": runtime <= memory.runtime_bytes,
              "communication_buffers": communication <= memory.communication_buffer_bytes,
              "kv_pool": values["kv_pool"] <= memory.kv_pool_bytes,
              "graphs": values.get("graphs", 0) <= memory.graph_bytes,
              "pp_buffers": pp <= rank_reservation["pipeline_buffer_reserved_bytes"]}
    return {"mathematical_weight_bytes": mathematical, "observed_fixed_weight_bytes": observed_weights,
        "fixed_padding_excess_bytes": padding, "fixed_padding_accounted_in": "runtime_reserve",
        "torch_fixed_categories": deepcopy(values), "native_fixed_runtime_bytes": native_runtime,
        "native_fixed_communication_bytes": native_communication,
        "runtime_component_bound_bytes": runtime, "communication_component_bound_bytes": communication,
        "pp_static_requested_bytes": static_pp, "pp_dynamic_transport_peak_bytes": dynamic_pp,
        "pp_component_bound_bytes": pp, "pp_reserved_bytes": rank_reservation["pipeline_buffer_reserved_bytes"],
        "checks": checks, "all_fixed_budgets_admitted": all(checks.values())}


def _assemble(base: Path, receipt_path: Path, out: Path, bindings_path: Path | None,
              inputs: Inputs, issues: list[dict], report: dict) -> list[dict]:
    plan_path = base / "session-plan.json"
    plan = inputs.read(plan_path)
    status = inputs.read(base / "session-status.json")
    launcher = inputs.read(base / "launcher-summary.json")
    receipt = inputs.read(receipt_path, capture=False)
    for key, expected in (("returncode", 0),):
        if type(receipt.get(key)) is not int or receipt[key] != expected:
            raise ValueError("Actual parent process exit is not proven zero")
    # New controllers bind the receipt to its immutable plan. Older standalone
    # receipts retain their original return-code contract.
    for key, expected_value in (("session_id", plan.get("session_id")),
                                ("plan_sha256", inputs.raw[str(plan_path.resolve())]),
                                ("session", str(base))):
        if key in receipt and receipt[key] != expected_value:
            raise ValueError("Actual parent process receipt differs from acquisition " + key)
    if (status.get("status") != "RAW_ACQUISITION_COMPLETED_AUDIT_REQUIRED"
            or type(status.get("worker_exit_code")) is not int or status["worker_exit_code"] != 0
            or launcher.get("status") != "RAW_ACQUISITION_COMPLETED_AUDIT_REQUIRED"
            or launcher.get("shutdown_error") is not None):
        raise ValueError("Actual bounded worker/launcher acquisition is incomplete")
    if status.get("session_id") != plan.get("session_id") or launcher.get("session_id") != plan.get("session_id"):
        raise ValueError("Parent/launcher source session mismatch")
    if inputs.root != Path(plan["out"]).resolve():
        raise ValueError("Acquisition root differs from its actual immutable plan")
    _source_gate(plan, inputs)
    exp_path = inputs.path(plan["bound_experiment"])
    supplied = inputs.read(exp_path, capture=False)
    if inputs.raw[str(exp_path.resolve())] != plan["bound_experiment_sha256"]:
        raise ValueError("Bound experiment bytes differ from the immutable plan")
    references = launcher.get("ranks")
    if not isinstance(references, list):
        raise ValueError("Actual worker exports are missing")
    workers = [inputs.read(inputs.path(row["session_file"])) for row in references]
    domain = validate_stage_workspace_domain(plan, workers, experiment=supplied,
                                            plan_sha256=inputs.raw[str(plan_path.resolve())])
    _natural_shutdown_gate(plan, workers, launcher, inputs)
    _write(out / "domain-audit.json", domain)
    report.update(source_session=domain["source_session"], clock_domain=domain["clock_domain"],
                  formal_repetitions=domain["formal_repetitions"], identity=domain["identity"])
    experiment = experiment_from_dict(domain["experiment"])
    neutral = replace(experiment, calibration=replace(experiment.calibration, backend="analytical",
        profile_path=None, workspace_profile_path=None, replica_profile_paths=(), replica_workspace_profile_paths=()))
    reservations = MemoryPlanner(neutral).snapshot()
    comm = inputs.read(bindings_path, capture=False) if bindings_path is not None else {}
    if not isinstance(comm, dict):
        raise ValueError("Communication bindings must map actual process IDs to binding objects")
    pids = {str(worker["process_id"]) for worker in workers}
    if set(comm) - pids:
        raise ValueError("Communication binding supplied for an unobserved process")
    by_rank = {worker["global_rank"]: worker for worker in workers}
    grouped: dict[tuple[int, int], dict[int, tuple[dict, int]]] = defaultdict(dict)
    budgets = []
    epoch_results = []
    for global_rank, placement in enumerate(domain["placements"]):
        worker = by_rank[global_rank]
        pid = worker["process_id"]
        worker_out = out / ("rank-" + str(global_rank))
        worker_out.mkdir()
        frames = [inputs.read(inputs.path(row["epoch_file"])) for row in worker["records"]]
        for index, frame in enumerate(frames):
            # The compact index omits only these four raw storage arrays.
            indexed = {key: deepcopy(value) for key, value in frame.items()
                       if key not in {"before", "after", "owners_before", "owners_after"}}
            indexed["epoch_file"] = worker["records"][index]["epoch_file"]
            if _json(indexed) != _json(worker["records"][index]):
                raise ValueError("Loaded epoch bytes and the raw session index differ")
        startup = inputs.read(inputs.path(worker["startup_snapshot"]))
        teardown = inputs.read(inputs.path(worker["teardown_snapshot"]))
        torch_probe = inputs.read(inputs.path(worker["torch_probe_file"]))
        native = inputs.read(inputs.path(worker["native_final_file"]))
        if native.get("pid") != pid or startup.get("process_id") != pid:
            raise ValueError("Actual native/startup process differs from exported worker")
        inference_thread = _integer(startup.get("thread_id"), "Actual inference thread", 1)
        if torch_probe.get("install_thread_id") != inference_thread or any(
                row.get("thread_id") != inference_thread for row in frames):
            raise ValueError("Torch observer and actual inference thread differ")
        formal = [frame for frame in frames if frame["cohort"] == "formal"]
        snapshots = {frame["epoch_id"]: {"before": frame["before"], "after": frame["after"]} for frame in formal}
        expected_name = experiment.infra.devices[global_rank].name
        agents = [agent for agent in native.get("agent_inventory", {}).get("agents", [])
                  if agent.get("type_name") == "GPU" and agent.get("product_name") == expected_name]
        if len(agents) != 1:
            raise ValueError("Actual native GPU agent mapping is missing or ambiguous")
        agent = _integer(agents[0].get("handle"), "Native GPU agent", 1)
        communication = comm.get(str(pid))
        _host_signal_source_gate(plan, worker, inputs, inference_thread=inference_thread)
        if worker.get("host_signal_evidence") is not None and communication is not None:
            if ("host_signal_evidence" in communication
                    and _json(communication["host_signal_evidence"]) != _json(worker["host_signal_evidence"])):
                raise ValueError("Communication host signal evidence differs from actual Worker envelope")
            communication = {**communication, "host_signal_evidence": deepcopy(worker["host_signal_evidence"])}
        if experiment.strategy.tensor_parallel_size * experiment.strategy.pipeline_parallel_size > 1:
            if communication is None:
                issues.append({"code": "COMMUNICATION_BINDING_MISSING", "global_rank": global_rank})
            else:
                _communication_source_gate(communication, worker, plan, agent, inputs)
        constructor = {"epoch_id": 5001, "epoch_label": "explicit_ATen_LT_workspace_initialization",
            "entrypoint": "ATen.getCUDABlasLtWorkspace", "lt_only": True,
            "driver_source_sha256": plan["worker_source_sha256"], "driver_source_reference": plan["worker_source"]}
        native_result = audit_native_backing(native, snapshots,
            device_to_agent={placement["device_index"]: agent}, ctor_binding=constructor,
            communication_binding=communication)
        _write(worker_out / "native-audit.json", native_result)
        # Native SDK traces may approach2GiB. Retain the derived report, not
        # simultaneous raw SDK dictionaries for the two physical ranks.
        del native
        gc.collect()
        if native_result.get("status") != "NATIVE_BACKING_CHECKS_PASSED":
            issues.append({"code": "NATIVE_BACKING_AUDIT_PARTIAL", "global_rank": global_rank})
        transport_context = None
        if experiment.strategy.pipeline_parallel_size > 1:
            transport_context = {"session": worker, "startup": startup, "teardown": teardown, "epochs": frames}
            transport = audit_workspace_transport(torch_probe=torch_probe,
                transport_context=transport_context, placement=placement)
            _write(worker_out / "transport-audit.json", transport)
            if transport.get("status") != "TRANSPORT_GENERATIONS_PROVEN":
                issues.append({"code": "PP_TRANSPORT_AUDIT_PARTIAL", "global_rank": global_rank})
        for frame in formal:
            result = audit_torch_workspace(model=asdict(experiment_from_dict(worker["experiment"]).strategy.model),
                before_snapshot=frame["before"], after_snapshot=frame["after"],
                owners_before=frame["owners_before"], owners_after=frame["owners_after"],
                torch_probe=torch_probe, epoch_id=frame["epoch_id"], history_max_entries=worker["history_max_entries"],
                device_index=placement["device_index"], placement=placement, transport_context=transport_context)
            _write(worker_out / ("torch-audit-" + str(frame["epoch_id"]) + ".json"), result)
            epoch_results.append({"global_rank": global_rank, "epoch_id": frame["epoch_id"],
                "phase": frame["phase"], "query_tokens": frame["query_tokens"], "context_lengths": frame["context_lengths"],
                "torch_status": result.get("status"), "native_status": native_result.get("status"),
                "component_candidate_peak_bytes": result.get("requested_storage_peak")})
            if result.get("status") != "CLOSED_CANDIDATE":
                issues.append({"code": "TORCH_AUDIT_PARTIAL", "global_rank": global_rank, "epoch_id": frame["epoch_id"]})
                continue
            peak = _integer(result.get("requested_storage_peak"), "Proven requested Workspace peak")
            grouped[placement["stage_index"], frame["iteration_id"]][placement["tp_rank"]] = (frame, peak)
            if native_result.get("status") == "NATIVE_BACKING_CHECKS_PASSED":
                try:
                    budget = _rank_budget(experiment, placement, result, native_result, reservations[global_rank])
                    budgets.append({"global_rank": global_rank, "epoch_id": frame["epoch_id"], **budget})
                    if not budget["all_fixed_budgets_admitted"]:
                        issues.append({"code": "FIXED_BUDGET_EXCEEDED", "global_rank": global_rank,
                                       "epoch_id": frame["epoch_id"], "checks": budget["checks"]})
                except (ValueError, TypeError, KeyError) as exc:
                    issues.append({"code": "FIXED_BUDGET_NOT_PROVEN", "global_rank": global_rank,
                                   "epoch_id": frame["epoch_id"], "message": str(exc)})
    report["fixed_budget_observations"] = budgets
    maxima = []
    for global_rank in range(len(domain["placements"])):
        selected = [row for row in budgets if row["global_rank"] == global_rank]
        maxima.append({"global_rank": global_rank,
            "expected_formal_epochs": 8 * plan["repeats"], "proved_budget_epochs": len(selected),
            "all_fixed_budgets_admitted": (len(selected) == 8 * plan["repeats"]
                and all(row["all_fixed_budgets_admitted"] for row in selected)),
            **{field: max((row[field] for row in selected), default=None) for field in
               ("observed_fixed_weight_bytes", "fixed_padding_excess_bytes", "runtime_component_bound_bytes",
                "communication_component_bound_bytes", "pp_static_requested_bytes",
                "pp_dynamic_transport_peak_bytes", "pp_component_bound_bytes")}})
    report["fixed_budget_maxima_by_rank"] = maxima
    report["component_epoch_results"] = epoch_results
    if plan.get("cohort") not in ("train", "repeat") or plan.get("repeats") != 3:
        issues.append({"code": "DIAGNOSTIC_COHORT_NOT_ELIGIBLE_FOR_CALIBRATION"})
    records = []
    tp = experiment.strategy.tensor_parallel_size
    for (stage, iteration), ranked in sorted(grouped.items()):
        if set(ranked) != set(range(tp)):
            issues.append({"code": "PROVEN_WORKSPACE_TP_RANK_MISSING", "stage_index": stage, "iteration_id": iteration})
            continue
        frame = ranked[0][0]
        for other, _ in ranked.values():
            for field in ("phase", "query_tokens", "context_lengths"):
                if _json(other[field]) != _json(frame[field]):
                    raise ValueError("Rank-local proven Workspace records describe different prepared shapes")
        key = shape_key(frame["phase"], frame["query_tokens"], frame["context_lengths"], stage)
        records.append({"record_id": "workspace:" + _json([plan["session_id"], stage, iteration]),
            "group_id": "workspace/" + plan["session_id"] + "/s" + str(stage) + "/" + hashlib.sha256(key.encode()).hexdigest()[:16],
            "phase": frame["phase"], "stage_index": stage, "query_tokens": list(frame["query_tokens"]),
            "context_lengths": list(frame["context_lengths"]),
            "workspace_bytes_per_rank": [{"tp_rank": rank, "peak_bytes": ranked[rank][1]} for rank in range(tp)]})
    expected = 8 * plan["repeats"] * experiment.strategy.pipeline_parallel_size
    if len(records) != expected:
        issues.append({"code": "CANONICAL_WORKSPACE_COHORT_INCOMPLETE", "expected_records": expected,
                       "proven_shape_rank_groups": len(records)})
    report["proven_shape_rank_group_count"] = len(records)
    return records


def assemble_stage_workspace(session: str | Path, receipt: str | Path, out: str | Path,
                             *, communication_bindings: str | Path | None = None) -> dict:
    """Replay raw evidence into a new directory; partial results never emit data."""
    base, target = Path(session).resolve(), Path(out).resolve()
    target.mkdir(parents=True, exist_ok=False)
    inputs = Inputs(base)
    issues: list[dict] = []
    report: dict[str, Any] = {"schema_version": 1, "kind": "workspace_stage_attribution_report",
        "status": "PARTIAL", "attribution_complete": False, "hardware_validated": False,
        "byte_basis": "owned_requested_storage_capacity", "issues": issues,
        "scope": "pinned eager Qwen3-1.7B BF16 single/TP2/PP2 prefill16/32 plus natural Decode contexts; exact per rank",
        "fixed_budget_observations": [], "component_epoch_results": [],
        "limitations": ["Allocator rounding/reserved arenas, fragmentation and driver headroom remain separate capacity evidence.",
            "These observations do not prove unseen-shape bounds or serving/SLA accuracy."]}
    records: list[dict] = []
    try:
        records = _assemble(base, Path(receipt).resolve(), target,
            Path(communication_bindings).resolve() if communication_bindings is not None else None,
            inputs, issues, report)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError,
            SyntaxError, UnicodeError, RecursionError, SimulationError) as exc:
        issues.append({"code": "MISSING_OR_UNSUPPORTED_ASSEMBLY_EVIDENCE", "message": type(exc).__name__ + ": " + str(exc)})
    report["raw_input_sha256"] = inputs.raw
    report["source_and_binary_sha256"] = inputs.shared
    report["capture_input_sha256"] = inputs.capture
    dataset = None
    if not issues:
        try:
            dataset = validate_workspace_dataset({"schema_version": 1, "kind": "workspace_measurements",
                "dataset_id": "workspace-stage-" + report["source_session"], "identity": report["identity"],
                "source": {"kind": "measured", "reference": "Actual pinned Worker/native/Torch/PP storage ledgers; exact observed domain"},
                "attribution": deepcopy(ATTRIBUTION), "records": records,
                "notes": {"byte_basis": report["byte_basis"], "scope": report["scope"],
                    "acquisition": {"session_id": report["source_session"], "clock_domain": report["clock_domain"],
                        "capture_input_sha256": sorted(set(inputs.capture.values()))}}})
        except (ValueError, TypeError, KeyError, SimulationError) as exc:
            issues.append({"code": "CANONICAL_WORKSPACE_DATASET_REJECTED", "message": str(exc)})
    report["canonical_record_count"] = len(records) if not issues else 0
    if not issues:
        report.update(status="ATTRIBUTION_COMPLETE_SUPPORTED_DOMAIN", attribution_complete=True)
    _write(target / "attribution-report.json", report)
    if not issues and dataset is not None:
        dataset["notes"]["attribution_report_sha256"] = _sha(target / "attribution-report.json")
        _write(target / "dataset.json", dataset)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--communication-bindings", type=Path)
    args = parser.parse_args()
    result = assemble_stage_workspace(args.session, args.receipt, args.out, communication_bindings=args.communication_bindings)
    print(_json({"status": result["status"], "issues": result["issues"],
                 "canonical_record_count": result["canonical_record_count"], "hardware_validated": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
