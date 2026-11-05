"""Collect public HSA backing observations outside SDK callbacks.

This producer never grants admission, invents callback correlations, frees a
Runtime signal or declares allocation birth. The independent synchronization
audit joins each observed cursor window to the complete raw SDK tape.
"""

from copy import deepcopy
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import threading
import time
from typing import Any


_UNCERTAIN = 2**64 - 1
_BUFFER_BYTES = 262144


def _artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1048576):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


def _write_new(path: Path, data: dict) -> None:
    encoded = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def _raw_clock() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _UNCERTAIN:
        raise ValueError(name + " must be an actual bounded integer")
    return value


def _checked_artifact(reference: dict) -> dict:
    if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
            or not Path(reference["path"]).is_absolute()):
        raise ValueError("Original artifact must have an actual absolute path")
    actual = _artifact(Path(reference["path"]))
    if (actual["sha256"] != reference.get("sha256")
            or type(reference.get("size_bytes")) is not int or actual["size_bytes"] != reference["size_bytes"]):
        raise ValueError("Original artifact bytes differ: " + reference["path"])
    return actual


def _namespace_manifest(path: Path) -> tuple[dict, dict]:
    reference = _artifact(path)
    data = json.loads(path.read_text())
    if (type(data.get("schema_version")) is not int or data["schema_version"] != 1
            or data.get("kind") != "cold_hsa_kfd_namespace_control_manifest"
            or not isinstance(data.get("source_binding"), dict)):
        raise ValueError("Actual cold HSA/KFD namespace manifest is missing")
    for name in ("native_report", "kfd_report", "control_jsonl", "build_receipt"):
        _checked_artifact(data.get(name))
    for name in ("create_cid", "event_query_cid", "kfd_create_event_call_id"):
        _integer(data.get(name), "Actual namespace locator " + name, 1)
    epochs = data.get("target_epochs")
    if (not isinstance(epochs, list) or not epochs or len(set(epochs)) != len(epochs)
            or any(type(value) is not int or value <= 0 for value in epochs)):
        raise ValueError("Cold namespace control lacks actual source-bound epoch identities")
    return data, reference


def _gnu_build_id(note: bytes) -> str:
    offset, found = 0, []
    order = "<" if sys.byteorder == "little" else ">"
    while offset < len(note):
        if len(note) - offset < 12:
            raise ValueError("Truncated actual amdgpu ELF note")
        names, descriptors, kind = struct.unpack_from(order + "III", note, offset)
        name_start = offset + 12
        desc_start = name_start + ((names + 3) & ~3)
        end = desc_start + ((descriptors + 3) & ~3)
        if end > len(note):
            raise ValueError("Actual amdgpu ELF note exceeds its bytes")
        if note[name_start:name_start + names].rstrip(b"\0") == b"GNU" and kind == 3:
            if descriptors != 20:
                raise ValueError("Actual amdgpu build ID is not the supported 20-byte descriptor")
            found.append(note[desc_start:desc_start + descriptors].hex())
        offset = end
    if len(found) != 1:
        raise ValueError("Actual amdgpu GNU build ID is missing or ambiguous")
    return found[0]


def _system_identity(*, sys_root: Path = Path("/sys"), dev_root: Path = Path("/dev"),
                     kernel_release: str | None = None, stat_device: Any = os.stat,
                     observations_dir: Path | None = None) -> dict:
    """Read current kernel/module/topology facts; never reuse control labels."""
    note = sys_root / "module/amdgpu/notes/.note.gnu.build-id"
    devices = []
    observations: list[dict[str, Any]] = []
    if observations_dir is not None:
        observations_dir.mkdir(mode=0o700)

    def observe(path: Path) -> bytes:
        raw = path.read_bytes()
        if observations_dir is not None:
            document = {"schema_version": 1, "kind": "actual_system_file_bytes",
                "original_path": str(path), "contents_hex": raw.hex(),
                "contents_size_bytes": len(raw), "contents_sha256": hashlib.sha256(raw).hexdigest()}
            target = observations_dir / ("file-" + str(len(observations)) + ".json")
            _write_new(target, document)
            observations.append({"original_path": str(path), "contents_size_bytes": len(raw),
                "contents_sha256": document["contents_sha256"], "snapshot_artifact": _artifact(target)})
        return raw

    for node in sorted((sys_root / "class/kfd/kfd/topology/nodes").iterdir()):
        gpu = int(observe(node / "gpu_id").decode().strip())
        if not gpu:
            continue
        _integer(gpu, "Actual topology GPU ID", 1)
        properties = {}
        for line in observe(node / "properties").decode().splitlines():
            key, _, value = line.partition(" ")
            if key in properties:
                raise ValueError("Actual KFD topology repeats a property")
            properties[key] = value.strip()
        minor = int(properties["drm_render_minor"])
        _integer(minor, "Actual DRM render minor")
        name = "renderD" + str(minor)
        device = stat_device(dev_root / "dri" / name)
        if not stat.S_ISCHR(device.st_mode):
            raise ValueError("Actual DRM topology path is not a character device")
        device_numbers = observe(sys_root / "class/drm" / name / "dev").decode().strip().split(":")
        if (len(device_numbers) != 2 or int(device_numbers[1]) != minor
                or (int(device_numbers[0]), int(device_numbers[1])) != (os.major(device.st_rdev), os.minor(device.st_rdev))):
            raise ValueError("Actual DRM character-device numbers differ from sysfs topology")
        values = {}
        for line in observe(sys_root / "class/drm" / name / "device/uevent").decode().splitlines():
            key, separator, value = line.partition("=")
            if separator:
                if key in values:
                    raise ValueError("Actual DRM uevent repeats an identity field")
                values[key] = value
        pci = values.get("PCI_SLOT_NAME", "")
        if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", pci):
            raise ValueError("Actual DRM PCI identity is missing")
        devices.append({"gpu_id": gpu, "rdev_major": os.major(device.st_rdev),
                        "rdev_minor": os.minor(device.st_rdev), "pci_bus_id": pci.lower()})
    if not devices or len({row["gpu_id"] for row in devices}) != len(devices):
        raise ValueError("Actual GPU/render topology is empty or ambiguous")
    uname = os.uname()
    release = uname.release if kernel_release is None else kernel_release
    if observations_dir is not None:
        target = observations_dir / "uname.json"
        _write_new(target, {"schema_version": 1, "kind": "actual_uname_result", "sysname": uname.sysname,
                           "release": release, "version": uname.version, "machine": uname.machine})
        observations.append({"source": "os.uname", "snapshot_artifact": _artifact(target)})
    result = {"kernel_release": release, "amdgpu_module_build_id": _gnu_build_id(observe(note)), "device_nodes": devices}
    if observations_dir is not None:
        result["system_observation_artifacts"] = observations
    return result


class _KfdObserver:
    def __init__(self, native: Any):
        self.native = native
        native.mlsys_kfd_probe_cursor.argtypes = []
        native.mlsys_kfd_probe_cursor.restype = ctypes.c_uint64
        native.mlsys_kfd_probe_write.argtypes = [ctypes.c_char_p]
        native.mlsys_kfd_probe_write.restype = ctypes.c_int

    def cursor(self) -> int:
        value = _UNCERTAIN
        for _ in range(64):
            value = int(self.native.mlsys_kfd_probe_cursor())
            if value != _UNCERTAIN:
                break
            time.sleep(0)  # CPU scheduling only; never a device/API fence.
        return value

    def snapshot(self, path: Path) -> dict:
        code = int(self.native.mlsys_kfd_probe_write(os.fsencode(path)))
        if code:
            raise RuntimeError("Actual KFD snapshot export failed: " + str(code))
        data = json.loads(path.read_text())
        if (data.get("schema_version") != 1 or data.get("kind") != "passive_kfd_event_api_tape"
                or data.get("pid") != os.getpid() or data.get("clock") != "CLOCK_MONOTONIC_RAW"
                or data.get("write_complete") is not True or data.get("complete_workspace") is not False
                or data.get("hardware_validated") is not False):
            raise ValueError("KFD prefix is not an actual intact snapshot for this process")
        for name in ("dropped_rows", "read_errors", "forward_errors", "reentrant_calls", "active_mutators"):
            if type(data.get(name)) is not int or data[name] != 0:
                raise ValueError("KFD snapshot retains uncertainty: " + name)
        return data


def _sample_with_kfd(provider: Any, handle: int, observer: _KfdObserver | None) -> dict:
    if observer is None:
        return provider.sample(handle)
    before, started = observer.cursor(), _raw_clock()
    sample = provider.sample(handle)
    ended, after = _raw_clock(), observer.cursor()
    sample["kfd_window"] = {"before_cursor": before, "after_cursor": after,
        "started_monotonic_raw_ns": started, "ended_monotonic_raw_ns": ended}
    return sample


def _kfd_source_binding(snapshot: dict, manifest: dict, native: Any, initialization: dict,
                        environment_dir: Path | None = None) -> dict:
    expected = manifest["source_binding"]
    names = ("probe_source", "native_probe_source", "probe_library", "hsa_runtime", "libc", "kfd_header", "drm_header")
    templates = expected.get("artifacts")
    if not isinstance(templates, dict) or set(templates) != set(names):
        raise ValueError("Cold control requires seven exact original source/binary/header artifacts")
    for reference in templates.values():
        _checked_artifact(reference)
    api_modules = {binding["module_path"] for binding in snapshot.get("libc_bindings", {}).values()
                   if binding.get("dladdr_success") is True and binding.get("address")}
    if not api_modules:
        raise ValueError("Actual KFD libc function modules are missing")
    modules = {Path(module).resolve(strict=True) for module in api_modules}
    if len(modules) != 1:
        raise ValueError("Observed KFD functions belong to different actual libc files")
    libc_loaded = sorted(api_modules)[0]
    overrides = {
        "probe_source": snapshot["compiled_source_path"], "probe_library": native._name,
        "hsa_runtime": initialization["api_bindings"]["hsa_amd_pointer_info"]["module_path"],
        "libc": libc_loaded,
    }
    artifacts = {}
    binding = {}
    for name in names:
        original = overrides.get(name, templates[name].get("loaded_path", templates[name]["path"]))
        if not isinstance(original, str) or not Path(original).is_absolute():
            raise ValueError("Observed KFD source path is not absolute: " + name)
        actual = {**_artifact(Path(original)), "loaded_path": original, "realpath": str(Path(original).resolve())}
        if actual["sha256"] != expected.get(name + "_sha256"):
            raise ValueError("Current KFD source/binary/header differs from the cold control: " + name)
        artifacts[name] = actual
        binding[name + "_sha256"] = actual["sha256"]
    paths = {artifacts["libc"][name] for name in ("path", "loaded_path", "realpath")}
    if not api_modules <= paths:
        raise ValueError("All observed libc aliases must be authenticated explicitly")
    identity = _system_identity() if environment_dir is None else _system_identity(observations_dir=environment_dir)
    for name in ("kernel_release", "amdgpu_module_build_id"):
        if identity[name] != expected.get(name):
            raise ValueError("Current actual kernel/module differs from the cold control: " + name)
    if snapshot.get("abi") != expected.get("abi"):
        raise ValueError("Current KFD compiled ABI differs from the cold control")
    binding.update(identity)
    binding.update(abi=deepcopy(snapshot["abi"]), artifacts=artifacts)
    reference = expected.get("build_receipt_artifact", manifest["build_receipt"])
    binding["build_receipt_artifact"] = _checked_artifact(reference)
    return binding


def _live(native_bridge: Any, path: Path) -> dict:
    function = native_bridge.mlsys_probe_write_live_signals
    function.argtypes = [ctypes.c_char_p]
    function.restype = ctypes.c_int
    code = int(function(os.fsencode(path)))
    if code:
        raise RuntimeError("Native live generation export failed: " + str(code))
    data = json.loads(path.read_text())
    if (data.get("schema_version") != 1 or data.get("kind") != "observed_live_hsa_signal_generations"
            or data.get("pid") != os.getpid() or data.get("dropped_events") != 0
            or data.get("reused_live_handle_count") != 0
            or data.get("complete_workspace") is not False or data.get("hardware_validated") is not False
            or not isinstance(data.get("generations"), list)):
        raise ValueError("Native live generation export is not an intact observation for this process")
    _integer(data.get("event_cursor"), "live event cursor")
    _integer(data.get("snapshot_monotonic_raw_ns"), "live snapshot RAW clock", 1)
    identities = set()
    for row in data["generations"]:
        key = tuple(_integer(row.get(name), name, 1) for name in
                    ("signal_handle", "generation", "create_cid", "create_seq", "create_tid"))
        _integer(row.get("create_epoch"), "create epoch")
        count = _integer(row.get("num_consumers"), "signal consumer count")
        if (count > 4 or row.get("consumers_observed") is not True
                or not isinstance(row.get("consumer_agents"), list) or len(row["consumer_agents"]) != count
                or type(row.get("initial_value")) is not int or not -(2**63) <= row["initial_value"] < 2**63):
            raise ValueError("Native signal constructor arguments were not completely observed")
        for agent in row["consumer_agents"]:
            _integer(agent, "signal consumer agent", 1)
        if key[0] in identities:
            raise ValueError("Native live export repeats a signal handle")
        identities.add(key[0])
    return data


class _Provider:
    def __init__(self, path: Path, rocr: Path, native: Any):
        self.library = ctypes.CDLL(str(path))
        self.library.mlsys_hsa_signal_backing_init.argtypes = [
            ctypes.c_char_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.library.mlsys_hsa_signal_backing_init.restype = ctypes.c_int
        self.library.mlsys_hsa_signal_backing_sample.argtypes = [
            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.library.mlsys_hsa_signal_backing_sample.restype = ctypes.c_int
        self.library.mlsys_hsa_signal_backing_control.argtypes = [
            ctypes.c_int64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        self.library.mlsys_hsa_signal_backing_control.restype = ctypes.c_int
        native.mlsys_probe_cursor.argtypes = []
        native.mlsys_probe_cursor.restype = ctypes.c_uint64
        pointer = ctypes.cast(native.mlsys_probe_cursor, ctypes.c_void_p).value
        if not pointer:
            raise ValueError("Actual Native cursor export is missing")
        self.initialization = self._call(self.library.mlsys_hsa_signal_backing_init, os.fsencode(rocr), pointer)

    @staticmethod
    def _call(function: Any, *arguments: Any) -> dict:
        output = ctypes.create_string_buffer(_BUFFER_BYTES)
        code = int(function(*arguments, output, len(output)))
        if code:
            raise RuntimeError("HSA provider protocol failed: " + str(code))
        data = json.loads(output.value)
        if not isinstance(data, dict):
            raise ValueError("HSA provider did not return a JSON object")
        return data

    def sample(self, handle: int) -> dict:
        return self._call(self.library.mlsys_hsa_signal_backing_sample, handle)

    def control(self, initial: int, consumers: list[int]) -> dict:
        values = (ctypes.c_uint64 * len(consumers))(*consumers)
        return self._call(self.library.mlsys_hsa_signal_backing_control, initial, values, len(consumers))


def _calls(sample: dict) -> list[dict]:
    calls = sample.get("api_calls")
    if not isinstance(calls, list) or "error" in sample:
        raise ValueError("Provider sample is partial or lacks API observations")
    return calls


def _check_windows(sample: dict, pid: int, tid: int) -> None:
    if sample.get("process_id") != pid or sample.get("thread_id") != tid:
        raise ValueError("Provider sample changed its actual process or thread")
    for call in _calls(sample):
        _check_call_window(call)


def _check_call_window(call: dict) -> None:
    before = _integer(call.get("before_cursor"), "before API cursor")
    after = _integer(call.get("after_cursor"), "after API cursor")
    begin = _integer(call.get("started_monotonic_raw_ns"), "API RAW start", 1)
    end = _integer(call.get("ended_monotonic_raw_ns"), "API RAW end", 1)
    if before == _UNCERTAIN or after == _UNCERTAIN or before >= after or begin > end:
        raise ValueError("Provider API cursor/clock window is uncertain or reversed")
    if "cid" in call or "correlation_id" in call:
        raise ValueError("Producer must not invent a Native callback correlation")


def capture_hsa_signal_backings(*, native_bridge: Any, provider_path: Path | str,
                                rocr_path: Path | str, abi_header_path: Path | str,
                                output_dir: Path | str, target_epochs: list[int],
                                source_fence: dict, source_scope: dict,
                                namespace_control: Path | str | None = None) -> dict:
    """Export live generations, own-control and two samples without a GPU call.

    Caller must use a paused, source-bound pre-shutdown CPU window after its
    existing target-device fence. Fence truth, complete SDK closure, correlations
    and backing admission are independently rechecked against raw evidence.
    """
    if (not isinstance(target_epochs, list) or not target_epochs
            or any(type(value) is not int or value <= 0 for value in target_epochs)
            or len(set(target_epochs)) != len(target_epochs)):
        raise ValueError("Explicit unique formal target epochs are required")
    if not isinstance(source_fence, dict) or not isinstance(source_scope, dict):
        raise ValueError("Actual caller fence and source scope observations are required")
    provider_path, rocr_path = Path(provider_path).resolve(strict=True), Path(rocr_path).resolve(strict=True)
    header = Path(abi_header_path).resolve(strict=True)
    directory = Path(output_dir)
    directory.mkdir(mode=0o700)  # Never overwrite another capture or its files.
    pid, tid = os.getpid(), threading.get_native_id()
    result: dict[str, Any] = {
        "schema_version": 1, "kind": "public_hsa_signal_backing_capture", "status": "PARTIAL",
        "process_id": pid, "thread_id": tid, "provider": _artifact(provider_path),
        "sources": [_artifact(Path(__file__)), _artifact(Path(__file__).with_name("hsa_signal_backing_probe.cpp"))],
        "abi_headers": [_artifact(header)],
        "runtime_dsos": [{**_artifact(rocr_path), "loaded_path": os.fspath(rocr_path)}],
        "source_fence": deepcopy(source_fence), "source_scope": deepcopy(source_scope),
        "source_binding": {"qualified_name": __name__ + ".capture_hsa_signal_backings",
            "source_sha256": hashlib.sha256(inspect.getsource(capture_hsa_signal_backings).encode()).hexdigest()},
        "capture_started_monotonic_raw_ns": _raw_clock(), "capture_ended_monotonic_raw_ns": None,
        "before_live": None, "after_live": None, "records": [], "ordinary_signal_control": None,
        "issues": [], "complete_workspace": False, "hardware_validated": False,
        "scope": "public_query_observations_only_no_allocation_birth_or_release_assertion",
    }
    for name in ("hsa.h", "hsa_ext_amd.h", "amd_hsa_common.h", "amd_hsa_queue.h"):
        sibling = header.with_name(name)
        if sibling.is_file():
            result["abi_headers"].append(_artifact(sibling))
    before_path, after_path = directory / "live-before.json", directory / "live-after.json"
    kfd: _KfdObserver | None = None
    kfd_observations: dict[str, Any] = {}
    try:
        result["before_live"] = _live(native_bridge, before_path)
        result["before_live_artifact"] = _artifact(before_path)
        rows = [row for row in result["before_live"]["generations"] if row["create_epoch"] in target_epochs]
        if not rows:
            result["status"] = "NO_LIVE_FORMAL_TARGETS"
            return result
        provider = _Provider(provider_path, rocr_path, native_bridge)
        result["initialization"] = provider.initialization
        result["compiler_source"] = _artifact(Path(provider.initialization["compiled_source_path"]))
        if (provider.initialization.get("ready") is not True
                or provider.initialization.get("process_id") != pid
                or provider.initialization.get("thread_id") != tid):
            raise ValueError("Provider did not bind this actual process/thread and its initialized Runtime")
        if (hasattr(native_bridge, "mlsys_kfd_probe_cursor") and hasattr(native_bridge, "mlsys_kfd_probe_write")):
            kfd = _KfdObserver(native_bridge)
            path = directory / "kfd-pre-query.json"
            kfd.snapshot(path)
            kfd_observations["pre_query_report_artifact"] = _artifact(path)
        control = provider.control(rows[0]["initial_value"], rows[0]["consumer_agents"])
        result["ordinary_signal_control"] = control
        if (control.get("process_id") != pid or control.get("thread_id") != tid
                or control.get("create_call", {}).get("output", {}).get("status") != 0
                or control.get("destroy_call", {}).get("output", {}).get("status") != 0
                or not isinstance(control.get("samples"), list) or len(control["samples"]) != 2):
            raise ValueError("Own ordinary-signal control did not preserve its real successful create/query/destroy")
        for sample in control["samples"]:
            _check_windows(sample, pid, tid)
        _check_call_window(control["create_call"])
        _check_call_window(control["destroy_call"])
        for row in rows:
            observation = {**deepcopy(row), "samples": []}
            result["records"].append(observation)
            for _ in range(2):
                sample = _sample_with_kfd(provider, row["signal_handle"], kfd)
                observation["samples"].append(sample)
                _check_windows(sample, pid, tid)
                if sample.get("signal_handle") != row["signal_handle"]:
                    raise ValueError("Provider queried a different actual signal")
                if kfd is not None:
                    window = sample["kfd_window"]
                    first = _integer(window.get("before_cursor"), "before KFD cursor")
                    last = _integer(window.get("after_cursor"), "after KFD cursor")
                    if first == _UNCERTAIN or last == _UNCERTAIN or first > last:
                        raise ValueError("Actual KFD sample cursor window is uncertain or reversed")
        modules = {item["loaded_path"] for item in result["runtime_dsos"]}
        for binding in provider.initialization.get("api_bindings", {}).values():
            if binding.get("dladdr_success") is not True:
                raise ValueError("Actual API function has no dladdr module identity")
            loaded = binding["module_path"]
            if loaded not in modules:
                result["runtime_dsos"].append({**_artifact(Path(loaded)), "loaded_path": loaded})
                modules.add(loaded)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError, OverflowError) as exc:
        result["issues"].append({"code": "PUBLIC_HSA_SIGNAL_CAPTURE_PARTIAL", "message": str(exc)})
    finally:
        try:
            if kfd is not None:
                path = directory / "kfd-post-query.json"
                snapshot = kfd.snapshot(path)
                artifact = _artifact(path)
                kfd_observations.update(post_query_report_artifact=artifact, report_artifact=artifact)
                if namespace_control is not None:
                    manifest, reference = _namespace_manifest(Path(namespace_control))
                    binding = _kfd_source_binding(snapshot, manifest, native_bridge, result["initialization"],
                                                  directory / "current-environment")
                    result["kfd_event_evidence"] = {"schema_version": 1, "kind": "passive_kfd_event_evidence",
                        **kfd_observations, "source_binding": binding, "namespace_control": manifest,
                        "namespace_control_artifact": reference}
            hardware_backed = any(call.get("api") == "hsa_amd_signal_get_event_id"
                and call.get("output", {}).get("status") == 0
                and type(call.get("output", {}).get("event_id")) is int and call["output"]["event_id"] > 0
                for row in result["records"] for sample in row["samples"] for call in sample.get("api_calls", []))
            if hardware_backed and "kfd_event_evidence" not in result:
                raise ValueError("Hardware-backed target signals require the actual cold KFD namespace/source evidence")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError, OverflowError) as exc:
            result["issues"].append({"code": "KFD_EVENT_EVIDENCE_PARTIAL", "message": str(exc)})
        if kfd_observations and "kfd_event_evidence" not in result:
            result["kfd_snapshot_observations"] = kfd_observations  # Diagnostics, never an admitted source binding.
        try:
            result["after_live"] = _live(native_bridge, after_path)
            result["after_live_artifact"] = _artifact(after_path)
            if result["before_live"] is not None:
                before = sorted(result["before_live"]["generations"], key=lambda row: row["signal_handle"])
                after = sorted(result["after_live"]["generations"], key=lambda row: row["signal_handle"])
                if before != after:
                    raise ValueError("Live signal generations changed during the CPU query window")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError, OverflowError) as exc:
            result["issues"].append({"code": "LIVE_SIGNAL_EXPORT_CHANGED_OR_MISSING", "message": str(exc)})
        result["capture_ended_monotonic_raw_ns"] = _raw_clock()
        if result["issues"]:
            result["status"] = "PARTIAL"
        elif result["status"] != "NO_LIVE_FORMAL_TARGETS":
            result["status"] = "CAPTURED"
        _write_new(directory / "capture.json", result)
    return result
