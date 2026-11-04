from pathlib import Path
import ctypes
import torch
import json
import hashlib
import os


def _mapped_rccl_libraries() -> set[Path]:
    return {
        Path(parts[-1]).resolve(strict=True)
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(parts := line.split(maxsplit=5)) == 6
        and parts[-1].startswith("/")
        and Path(parts[-1]).name.startswith("librccl.so")
    }


def _select_rccl_library(mapped: set[Path], provided: str | None) -> tuple[Path, str]:
    if provided is not None:
        requested = Path(provided)
        if not requested.is_absolute():
            raise ValueError("MLSYS_RCCL_LIBRARY must name an explicit absolute librccl.so path")
        selected = requested.resolve(strict=True)
        if not selected.is_file() or not selected.name.startswith("librccl.so"):
            raise ValueError("MLSYS_RCCL_LIBRARY must identify the actual librccl.so library")
        if mapped and mapped != {selected}:
            raise ValueError("MLSYS_RCCL_LIBRARY conflicts with the actual mapped RCCL libraries")
        return selected, "explicit_MLSYS_RCCL_LIBRARY"
    if len(mapped) != 1:
        raise ValueError("Provide MLSYS_RCCL_LIBRARY when exactly one actual RCCL library is not mapped")
    return next(iter(mapped)), "actual_proc_maps"


root = Path(os.environ.get("MLSYS_CONTROL_OUTPUT_DIR", "/app/work")).resolve()
root.mkdir(parents=True, exist_ok=True)
result_path = root / "native2-control-result.json"
checkpoint_path = root / "native2-controls-checkpoint.json"
for output_path in (result_path, checkpoint_path):
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Control output already exists: {output_path}")
torch.cuda.init()
torch.cuda.synchronize(0)
probe = ctypes.CDLL("/app/work/native_probe.so")
probe.mlsys_probe_begin.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
probe.mlsys_probe_end.argtypes = [ctypes.c_uint64]
probe.mlsys_probe_write.argtypes = [ctypes.c_char_p]
assert probe.mlsys_probe_begin(1, b"negative_no_device_alloc") == 0
torch.cuda.synchronize(0)
assert probe.mlsys_probe_end(1) == 0
paths = {
    s.split(maxsplit=5)[-1]
    for s in Path("/proc/self/maps").read_text().splitlines()
    if len(s.split(maxsplit=5)) == 6 and "/libamdhip64.so" in s
}
assert len(paths) == 1
hip = ctypes.CDLL(next(iter(paths)))
hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
hip.hipFree.argtypes = [ctypes.c_void_p]
assert probe.mlsys_probe_begin(2, b"positive_direct_hip_malloc_free") == 0
pointer = ctypes.c_void_p()
rc1 = hip.hipMalloc(ctypes.byref(pointer), 8 * 1024 * 1024)
assert rc1 == 0 and pointer.value
rc2 = hip.hipFree(pointer)
assert rc2 == 0
torch.cuda.synchronize(0)
assert probe.mlsys_probe_end(2) == 0
rccl_path, rccl_selection = _select_rccl_library(
    _mapped_rccl_libraries(), os.environ.get("MLSYS_RCCL_LIBRARY")
)
rccl = ctypes.CDLL(str(rccl_path))
if _mapped_rccl_libraries() != {rccl_path}:
    raise ValueError("The selected RCCL library is not the unique actual loaded mapping")
rccl.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
rccl.ncclGetVersion.restype = ctypes.c_int
version = ctypes.c_int()
assert probe.mlsys_probe_begin(3, b"positive_known_rccl_ncclGetVersion") == 0
try:
    rccl_returncode = rccl.ncclGetVersion(ctypes.byref(version))
finally:
    assert probe.mlsys_probe_end(3) == 0
with rccl_path.open("rb") as library_stream:
    rccl_sha256 = hashlib.file_digest(library_stream, "sha256").hexdigest()
assert probe.mlsys_probe_write(str(checkpoint_path).encode()) == 0
assert probe.mlsys_probe_stop() == 0
with result_path.open("x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "COMPLETED" if rccl_returncode == 0 and version.value > 0 else "FAILED",
            "direct_hip_malloc_returncode": rc1,
            "direct_hip_free_returncode": rc2,
            "rccl_library": str(rccl_path),
            "rccl_library_sha256": rccl_sha256,
            "rccl_library_selection": rccl_selection,
            "nccl_get_version_returncode": rccl_returncode,
            "nccl_get_version_code": version.value,
            "rccl_query_control_epoch_id": 3,
            "complete_workspace": False,
            "hardware_validated": False,
            "diagnostic_only": True,
        },
        stream,
    )
assert rccl_returncode == 0 and version.value > 0
