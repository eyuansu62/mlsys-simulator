"""Optional ROCm inventory and evidence binding, without a GPU dependency at import.

Unknown versions remain unknown. PyTorch's collective build version is recorded
separately from the version queried from an explicitly selected RCCL library.
"""

import ctypes
from dataclasses import replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, NoReturn

from .schema import Experiment, SimulationError


class CollectionError(SimulationError):
    def __init__(self, code: str, message: str, artifacts: dict | None = None):
        super().__init__(code, message)
        self.artifacts = artifacts or {}


def _invalid(message: str) -> NoReturn:
    raise SimulationError("INVALID_COLLECTION_PLAN", message)


def object_fields(value: Any, required: set[str], optional: set[str], label: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _invalid(f"{label} must be an object with string keys")
    if required - value.keys() or value.keys() - required - optional:
        _invalid(f"{label}: missing {sorted(required - value.keys())}; "
                 f"unknown {sorted(value.keys() - required - optional)}")
    return value


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(f"{label} must be a nonempty string")
    return value


def integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _invalid(f"{label} must be an integer in [{minimum}, 2**63-1]")
    return value


def number(value: Any, label: str, minimum: float = 0, positive: bool = False) -> float:
    if type(value) not in (int, float):
        _invalid(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        _invalid(f"{label} must be finite")
    if not math.isfinite(result) or result < minimum or (positive and result <= 0):
        _invalid(f"{label} must be finite and meet its lower bound")
    return result


def read_json(path: str | Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _invalid(f"Duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda value: _invalid(f"Non-finite JSON value {value}"))
        json.dumps(value, allow_nan=False)
        return value
    except SimulationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SimulationError("INVALID_COLLECTION_PLAN", f"Cannot read {path}: {exc}") from exc


def _read_version(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as stream:
            value = stream.read(4096).strip()
        return value.splitlines()[0] if value else None
    except (OSError, UnicodeError):
        return None


def _version_fields(value: Any, keys: set[str]) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(char for char in key.lower() if char.isalnum())
            if normalized in keys and isinstance(item, str) and item.strip().lower() not in {
                "", "n/a", "unknown", "not available"}:
                result.add(item.strip())
            result.update(_version_fields(item, keys))
    elif isinstance(value, list):
        for item in value:
            result.update(_version_fields(item, keys))
    return result


def _rccl_version(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("RCCL library must be a regular file")
    library = ctypes.CDLL(str(resolved))
    function = library.ncclGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    code = ctypes.c_int()
    status = function(ctypes.byref(code))
    if status != 0 or code.value < 2000:
        raise ValueError(f"ncclGetVersion failed or returned an invalid code: {status}/{code.value}")
    if code.value >= 10000:
        major, minor, patch = code.value // 10000, code.value // 100 % 100, code.value % 100
    else:
        major, minor, patch = code.value // 1000, code.value // 100 % 10, code.value % 100
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(resolved), "sha256": digest.hexdigest(), "version_code": code.value,
            "version": f"{major}.{minor}.{patch}", "method": "selected_library_ncclGetVersion",
            "loaded_runtime_equivalence": "must_be_established_by_runtime_adapter"}


def probe_rocm(*, torch_module: Any = None, rccl_library: str | Path | None = None,
               rocm_root: str | Path = "/opt/rocm") -> dict:
    """Observe local availability. Supplying a test module always marks a fixture."""
    fixture = torch_module is not None
    inventory: dict[str, Any] = {
        "schema_version": 1, "kind": "rocm_inventory", "status": "UNAVAILABLE",
        "source": {"kind": "test_fixture" if fixture else "measured",
                   "reference": "injected CPU test backend" if fixture else "local ROCm inventory probe"},
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version()},
        "torch": {"version": None, "hip_version": None, "cuda_available": False,
                  "pytorch_collective_build_version": None},
        "versions": {"rocm_version": None, "driver_version": None, "rccl_version": None},
        "devices": [], "tools": {}, "issues": [], "rccl_library": None,
    }
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except (ImportError, OSError, RuntimeError) as exc:
            inventory["issues"].append({"code": "TORCH_UNAVAILABLE", "message": str(exc)})
    if torch_module is not None:
        try:
            inventory["torch"].update(version=str(torch_module.__version__),
                                      hip_version=getattr(torch_module.version, "hip", None),
                                      cuda_available=bool(torch_module.cuda.is_available()))
            if not inventory["torch"]["hip_version"] or not inventory["torch"]["cuda_available"]:
                inventory["issues"].append({"code": "HIP_UNAVAILABLE",
                    "message": "Collection requires both a HIP build and an available GPU"})
            else:
                for index in range(torch_module.cuda.device_count()):
                    properties = torch_module.cuda.get_device_properties(index)
                    inventory["devices"].append({"index": index, "name": str(properties.name),
                        "memory_bytes": int(properties.total_memory),
                        "architecture": getattr(properties, "gcnArchName", None)})
                if inventory["devices"]:
                    inventory["status"] = "READY"
                try:
                    version = torch_module.cuda.nccl.version()
                    inventory["torch"]["pytorch_collective_build_version"] = (
                        ".".join(str(part) for part in version) if isinstance(version, (list, tuple)) else str(version))
                except (AttributeError, RuntimeError, OSError):
                    pass
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            inventory["status"] = "UNAVAILABLE"
            inventory["issues"].append({"code": "HIP_PROBE_FAILED", "message": str(exc)})
    root = Path(rocm_root)
    for candidate in (root / ".info/version", root / ".info/version-dev"):
        value = _read_version(candidate)
        if value is not None:
            inventory["versions"]["rocm_version"] = value
            inventory["tools"]["rocm_version_file"] = str(candidate)
            break
    inventory["versions"]["driver_version"] = _read_version(Path("/sys/module/amdgpu/version"))
    binary = shutil.which("amd-smi")
    if binary:
        try:
            result = subprocess.run([binary, "version", "--json"], capture_output=True,
                                    text=True, timeout=10, check=False, shell=False)
            captured = {"argv": [binary, "version", "--json"], "returncode": result.returncode,
                        "stdout": result.stdout[:65536], "stderr": result.stderr[:65536]}
            inventory["tools"]["amd_smi_version"] = captured
            if result.returncode == 0 and len(result.stdout) <= 65536:
                payload = json.loads(result.stdout)
                for field, keys in (("rocm_version", {"rocmversion"}),
                                    ("driver_version", {"driverversion", "amdgpuversion"})):
                    values = _version_fields(payload, keys)
                    if len(values) == 1 and inventory["versions"][field] is None:
                        inventory["versions"][field] = next(iter(values))
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            inventory["issues"].append({"code": "AMD_SMI_UNAVAILABLE", "message": str(exc)})
    if rccl_library is not None:
        try:
            info = _rccl_version(Path(rccl_library))
            inventory["rccl_library"] = info
            inventory["versions"]["rccl_version"] = info["version"]
        except (OSError, AttributeError, ValueError) as exc:
            inventory["issues"].append({"code": "RCCL_VERSION_UNAVAILABLE", "message": str(exc)})
    return inventory


def bind_inventory(experiment: Experiment, inventory: dict, *, runtime_name: str,
                   runtime_version: str, require_rccl: bool = False,
                   allow_test_fixture: bool = False) -> Experiment:
    """Bind observed device/software facts while retaining declared model identity."""
    from .config import validate_experiment
    from .profiles import profile_identity, validate_identity

    validate_experiment(experiment)
    if type(allow_test_fixture) is not bool or type(require_rccl) is not bool:
        _invalid("Hardware source overrides must be booleans")
    if (not isinstance(inventory, dict) or inventory.get("kind") != "rocm_inventory"
            or type(inventory.get("schema_version")) is not int or inventory["schema_version"] != 1):
        raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "A versioned ROCm inventory is required")
    source = inventory.get("source")
    torch_info = inventory.get("torch")
    versions = inventory.get("versions")
    if not isinstance(source, dict) or not isinstance(torch_info, dict) or not isinstance(versions, dict):
        raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Inventory source, torch and versions must be objects")
    text(source.get("reference"), "inventory.source.reference")
    fixture = source.get("kind") == "test_fixture"
    if fixture and not allow_test_fixture:
        raise SimulationError("TEST_FIXTURE_NOT_ALLOWED", "Injected hardware inventory requires explicit fixture permission")
    if source.get("kind") not in ("measured", "test_fixture"):
        raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Inventory source must be declared")
    if (inventory.get("status") != "READY" or not isinstance(torch_info.get("hip_version"), str)
            or not torch_info["hip_version"].strip() or torch_info.get("cuda_available") is not True):
        raise SimulationError("HARDWARE_UNAVAILABLE", "An available AMD GPU and a PyTorch HIP build are required")
    facts = inventory.get("devices")
    if not isinstance(facts, list) or len(facts) < len(experiment.infra.devices):
        raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Inventory does not cover every experiment device")
    devices = []
    for index, device in enumerate(experiment.infra.devices):
        fact = facts[index]
        if not isinstance(fact, dict) or type(fact.get("index")) is not int or fact["index"] != index:
            raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Inventory must retain ordered visible device indices")
        devices.append(replace(device, name=text(fact.get("name"), "device.name"),
                               memory_bytes=integer(fact.get("memory_bytes"), "device.memory_bytes", 1)))
    actual: dict[str, str] = {}
    for name in ("rocm_version", "driver_version", "rccl_version"):
        value = versions.get(name)
        if value is None and name == "rccl_version" and not require_rccl:
            value = "not-applicable"
        if value is None:
            if not fixture:
                raise SimulationError("HARDWARE_METADATA_INCOMPLETE", f"No observed {name}; no version is inferred")
            value = "unknown"
        if (not fixture and (name != "rccl_version" or require_rccl)
                and str(value).strip().lower() in {"unknown", "unspecified", "not-applicable", "n/a"}):
            raise SimulationError("HARDWARE_METADATA_INCOMPLETE", f"No applicable observed {name}")
        actual[name] = text(value, name)
    calibration = replace(experiment.calibration, runtime_name=text(runtime_name, "runtime_name"),
                          runtime_version=text(runtime_version, "runtime_version"),
                          rocm_version=actual["rocm_version"], driver_version=actual["driver_version"],
                          rccl_version=actual["rccl_version"])
    effective = replace(experiment, infra=replace(experiment.infra, devices=tuple(devices)), calibration=calibration)
    validate_experiment(effective)
    validate_identity(profile_identity(effective), measured=not fixture)
    return effective
