"""RCCL AllReduce capture with explicit units, ranks and correctness evidence.

The supported text format is rccl-tests' single-process AllReduce table. Its
``time`` column is used directly; neither reported bandwidth nor process wall
time is substituted for primitive service time. SendRecv is intentionally not
interpreted as the simulator's isolated, unidirectional PP transfer.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, NoReturn

from .communication_calibration import _validate_dataset
from .config import validate_experiment
from .profiles import profile_identity, validate_identity
from .schema import Experiment, SimulationError


_LIMIT = 2 * 1024 * 1024
_MAX_LIMIT = 16 * 1024 * 1024
_MODE = {"out_of_place": "out-of-place", "in_place": "in-place"}
_WIDTHS = {"int8": 1, "uint8": 1, "half": 2, "bfloat16": 2,
           "float": 4, "int32": 4, "uint32": 4, "double": 8, "int64": 8, "uint64": 8}
_ACTIVATION_DTYPES = {"float16": "half", "bfloat16": "bfloat16", "float32": "float", "float64": "double"}


def _fail(message: str, code: str = "INVALID_RCCL_OUTPUT") -> NoReturn:
    raise SimulationError(code, message)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(f"{label} must be a signed 64-bit integer >= {minimum}", "INVALID_COLLECTION_PLAN")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be a finite number", "INVALID_COLLECTION_PLAN")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_COLLECTION_PLAN", f"{label} must be finite") from exc
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        _fail(f"{label} must be finite and {'positive' if positive else 'nonnegative'}", "INVALID_COLLECTION_PLAN")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        _fail(f"{label} must be a nonempty string without NUL", "INVALID_COLLECTION_PLAN")
    return value


def _table_number(value: str, label: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_RCCL_OUTPUT", f"{label} is not numeric") from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        _fail(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _communication_dtype(experiment: Experiment, *, fixture: bool) -> dict:
    dtype = experiment.strategy.model.activation_dtype
    if dtype in _ACTIVATION_DTYPES:
        return {"activation_dtype": dtype, "rccl_datatype": _ACTIVATION_DTYPES[dtype],
                "binding": "explicit_activation_dtype"}
    if dtype == "auto" and fixture:
        return {"activation_dtype": "auto", "rccl_datatype": "float",
                "binding": "fixture_default_float_not_hardware_evidence"}
    _fail("RCCL hardware collection needs an explicit supported activation_dtype",
          "HARDWARE_METADATA_INCOMPLETE")


def parse_rccl_output(output: str, *, ranks: int, mode: str = "out_of_place") -> dict:
    """Parse a complete checked AllReduce table, preserving raw column facts.

    Rank aggregation and algorithm are launch facts not printed by every table;
    callers must also preserve the command/environment that established them.
    All row correctness columns are checked, including an unselected mode.
    """
    _integer(ranks, "ranks", 2)
    if mode not in ("out_of_place", "in_place"):
        _fail("mode must be in_place or out_of_place", "INVALID_COLLECTION_PLAN")
    if not isinstance(output, str) or not output.strip():
        _fail("RCCL stdout must be nonempty text")
    if re.search(r"\b(?:sendrecv|send_recv|all_gather|reduce_scatter|broadcast)_perf\b", output, re.I):
        _fail("Only all_reduce_perf is supported; SendRecv is not isolated PP P2P")
    starts = re.findall(r"^#\s*Collective test starting:\s*(\S+)\s*$", output, re.M)
    ends = re.findall(r"^#\s*Collective test concluded:\s*(\S+)\s*$", output, re.M)
    if (len(starts) != 1 or len(ends) != 1 or starts != ends
            or Path(starts[0]).name != "all_reduce_perf"):
        _fail("Expected one complete all_reduce_perf start/conclusion pair")
    if re.search(r"\b(?:FAILED|ABORTED|TIMEOUT)\b", output, re.I):
        _fail("RCCL output reports a failed or incomplete benchmark")
    settings = re.findall(
        r"^#\s*nThread\s+(\d+)\s+nGpus\s+(\d+)\s+.*?validation:\s*(\d+)\s+graph:\s*(\d+)\s*$",
        output, re.M)
    if len(settings) != 1:
        _fail("Missing or repeated nThread/nGpus/validation/graph launch header")
    threads, gpus, validation, graph = map(int, settings[0])
    if threads != 1 or gpus != ranks or validation < 1 or graph != 0:
        _fail("Capture requires one process/thread, gpus=ranks, correctness enabled and no graph replay")
    rank_rows = re.findall(
        r"^#\s*Rank\s+(\d+)\s+Group\s+(\d+)\s+Pid\s+(\d+)\s+on\s+\S+\s+device\s+(\d+)\s+\[[^\]]+\]\s+(.+?)\s*$",
        output, re.M)
    if (len(rank_rows) != ranks or {int(row[0]) for row in rank_rows} != set(range(ranks))
            or {int(row[1]) for row in rank_rows} != {0}
            or len({row[2] for row in rank_rows}) != 1
            or {int(row[3]) for row in rank_rows} != set(range(ranks))):
        _fail("Rank/device evidence does not describe one complete unsplit single-process group")
    for rank, _, _, device, _ in rank_rows:
        if int(rank) != int(device):
            _fail("Rank-to-device order differs from the supported visible-device order")
    errors = re.findall(r"^#\s*Out of bounds values\s*:\s*(\d+)\s+(\S+)\s*$", output, re.M)
    if len(errors) != 1 or errors[0] != ("0", "OK"):
        _fail("RCCL correctness summary must explicitly report zero out-of-bounds values")
    versions = re.findall(r"^rccl-tests:\s*Version\s+(\S+)\s*$", output, re.M)
    if len(versions) > 1:
        _fail("Conflicting rccl-tests version headers")
    library_versions = set(re.findall(r"\b(?:NCCL|RCCL)\s+version\s*[:=]?\s*([0-9]+\.[0-9]+\.[0-9]+)", output, re.I))
    if len(library_versions) > 1:
        _fail("Conflicting loaded RCCL library banners")
    modes: list[str] = []
    header: list[str] | None = None
    units: list[str] | None = None
    records = []
    started = False
    finished = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Out of bounds values"):
            finished = True
        if finished:
            continue
        if "out-of-place" in line or "in-place" in line:
            if header is not None:
                _fail("Repeated table mode header")
            modes = []
            if "out-of-place" in line:
                modes.append("out_of_place")
            if "in-place" in line.replace("out-of-place", ""):
                modes.append("in_place")
        if stripped.startswith("#") and re.match(r"#\s*size\s+count\s+type\s+redop\s+root\s+", stripped):
            if header is not None:
                _fail("Repeated RCCL table header")
            header = stripped[1:].split()
            expected = ["size", "count", "type", "redop", "root"] + ["time", "algbw", "busbw", "#wrong"] * len(modes)
            if (not modes or mode not in modes
                    or header not in (expected, expected + ["algo", "proto", "nchannels"])):
                _fail("Unsupported RCCL columns, CPU time, or missing selected placement mode")
            continue
        if header is not None and stripped.startswith("#") and "(B)" in stripped:
            if units is not None:
                _fail("Repeated table units")
            units = re.findall(r"\((us|ms|ns|s)\)", stripped)
            if len(units) != len(modes):
                _fail("Every RCCL time column needs an explicit supported unit")
            started = True
            continue
        if not started or not stripped or stripped.startswith("#"):
            continue
        columns = stripped.split()
        if header is None or units is None or len(columns) != len(header):
            _fail("Malformed RCCL numeric table row")
        try:
            size, count, root = int(columns[0]), int(columns[1]), int(columns[4])
        except ValueError as exc:
            raise SimulationError("INVALID_RCCL_OUTPUT", "Invalid RCCL size/count/root") from exc
        datatype = columns[2]
        if (not 0 < size <= 2**63 - 1 or not 0 < count <= 2**63 - 1
                or datatype not in _WIDTHS or size != count * _WIDTHS[datatype]
                or columns[3] != "sum" or root not in (-1, 0)):
            _fail("RCCL payload must describe a positive exact-byte supported Sum AllReduce")
        values = {}
        for position, placement in enumerate(modes):
            offset = 5 + position * 4
            duration = _table_number(columns[offset], "RCCL time", positive=True)
            factor = {"ns": 1e-6, "us": .001, "ms": 1.0, "s": 1000.0}[units[position]]
            milliseconds = duration * factor
            if not math.isfinite(milliseconds) or milliseconds <= 0:
                _fail("RCCL time is not representable in milliseconds")
            algbw = _table_number(columns[offset + 1], "algbw")
            busbw = _table_number(columns[offset + 2], "busbw")
            if columns[offset + 3] != "0":
                _fail("RCCL row has unchecked or nonzero correctness errors")
            values[placement] = {"duration_ms": milliseconds, "reported_time": duration,
                                 "reported_time_unit": units[position], "algbw_gbps": algbw,
                                 "busbw_gbps": busbw}
        if header[-3:] == ["algo", "proto", "nchannels"]:
            if columns[-3].lower() != "ring" or not columns[-1].isdigit() or int(columns[-1]) <= 0:
                _fail("Reported communication algorithm must be Ring with positive channels")
        records.append({"message_bytes": size, "count": count, "datatype": datatype,
                        "mode": mode, **values[mode], "all_modes": values})
    if not records or header is None or units is None:
        _fail("No complete RCCL timing table was found")
    start_position = re.search(r"^#\s*Collective test starting:", output, re.M)
    table_position = re.search(r"^#\s*size\s+count\s+type\s+redop\s+root", output, re.M)
    error_position = re.search(r"^#\s*Out of bounds values", output, re.M)
    end_position = re.search(r"^#\s*Collective test concluded:", output, re.M)
    if (start_position is None or table_position is None or error_position is None or end_position is None
            or not start_position.start() < table_position.start() < error_position.start() < end_position.start()):
        _fail("RCCL start, table, correctness summary and conclusion are out of order")
    return {"records": records, "metadata": {
        "primitive": "allreduce", "ranks": ranks, "threads": threads,
        "gpus_per_thread": gpus, "processes": 1, "group": 0,
        "mode": mode, "correctness_errors": 0,
        "rccl_tests_version": versions[0] if versions else None,
        "rccl_library_version": next(iter(library_versions)) if library_versions else None,
        "devices": [{"rank": int(row[0]), "device_index": int(row[3]), "name": row[4]}
                    for row in sorted(rank_rows, key=lambda row: int(row[0]))],
        "rank_aggregation_observed_in_table": False,
        "algorithm_observed_in_table": header[-3:] == ["algo", "proto", "nchannels"],
    }}


def _run_bounded(argv: list[str], *, timeout_s: float, max_output_bytes: int, env: dict[str, str]) -> dict:
    start = time.monotonic()
    timed_out = False
    truncated = False
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env, shell=False)
        while process.poll() is None:
            if os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size > max_output_bytes:
                truncated = True
                process.kill()
                break
            if time.monotonic() - start >= timeout_s:
                timed_out = True
                process.kill()
                break
            time.sleep(.01)
        process.wait()
        stdout.seek(0)
        stderr.seek(0)
        left = stdout.read(max_output_bytes + 1)
        right = stderr.read(max_output_bytes + 1)
    truncated |= len(left) + len(right) > max_output_bytes
    return {"returncode": process.returncode, "stdout": left[:max_output_bytes].decode("utf-8", errors="replace"),
            "stderr": right[:max(0, max_output_bytes - len(left))].decode("utf-8", errors="replace"),
            "timed_out": timed_out, "output_truncated": truncated,
            "process_elapsed_ms": (time.monotonic() - start) * 1000}


def _source(value: Any, allow_test_fixture: bool) -> dict:
    if (not isinstance(value, dict) or set(value) != {"kind", "reference"}
            or value.get("kind") not in ("measured", "test_fixture")):
        _fail("source must declare measured or test_fixture and a reference", "INVALID_COLLECTION_PLAN")
    _text(value["reference"], "source.reference")
    if value["kind"] == "test_fixture" and not allow_test_fixture:
        _fail("Fixture collection requires allow_test_fixture", "TEST_FIXTURE_NOT_ALLOWED")
    return dict(value)


def collect_rccl(experiment: Experiment, plan: dict, inventory: dict | None = None, *,
                 base_dir: str | Path | None = None, allow_test_fixture: bool = False,
                 runner: Any = None) -> dict:
    """Capture checked single-node AllReduce or import its preserved text log.

    Live runners are bounded, shell-free and pin Ring/max-rank/validation flags.
    Raw imports explicitly declare launch facts absent from text tables; those
    declarations remain in the raw evidence instead of becoming observed facts.
    A custom runner is a fixture injection and can never emit measured records.
    """
    from .hardware import CollectionError, bind_inventory, probe_rocm

    validate_experiment(experiment)
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be boolean", "INVALID_COLLECTION_PLAN")
    if not isinstance(plan, dict):
        _fail("RCCL plan must be an object", "INVALID_COLLECTION_PLAN")
    if runner is not None and not callable(runner):
        _fail("Injected runner must be callable", "INVALID_COLLECTION_PLAN")
    if inventory is not None and not isinstance(inventory, dict):
        _fail("Inventory must be an object", "HARDWARE_IDENTITY_MISMATCH")
    required = {"schema_version", "dataset_id", "ranks", "mode"}
    optional = {"source_reference", "source", "binary", "message_bytes", "repeats", "iterations",
                "warmup_iterations", "timeout_s", "max_output_bytes", "raw_path", "algorithm",
                "rank_aggregation", "primitive"}
    if required - plan.keys() or plan.keys() - required - optional:
        _fail("RCCL plan has missing or unknown fields", "INVALID_COLLECTION_PLAN")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        _fail("RCCL plan schema_version must be 1", "INVALID_COLLECTION_PLAN")
    dataset_id = _text(plan["dataset_id"], "dataset_id")
    ranks = _integer(plan["ranks"], "ranks", 2)
    mode = plan["mode"]
    if mode not in ("in_place", "out_of_place") or plan.get("primitive", "allreduce") != "allreduce":
        _fail("Only in_place/out_of_place AllReduce is supported; SendRecv is not PP P2P", "INVALID_COLLECTION_PLAN")
    if experiment.strategy.replicas != 1 or len(experiment.infra.devices) != ranks:
        _fail("RCCL captures require one replica identity with exactly ranks devices", "INVALID_COLLECTION_PLAN")
    limit = _integer(plan.get("max_output_bytes", _LIMIT), "max_output_bytes", 1)
    if limit > _MAX_LIMIT:
        _fail("max_output_bytes exceeds the 16MiB capture bound", "INVALID_COLLECTION_PLAN")
    base = Path.cwd() if base_dir is None else Path(base_dir)
    raw_records: list[dict] = []
    parsed_records: list[dict] = []
    metadata: list[dict] = []
    source: dict
    effective = experiment
    datatype_binding: dict = {}

    def failure(code: str, message: str) -> NoReturn:
        raise CollectionError(code, message, artifacts={
            "kind": "rccl_collection", "inventory": inventory,
            "experiment": asdict(effective), "raw_records": raw_records,
            "communication_datatype": datatype_binding,
            "hardware_validated": False})

    def consume(raw: dict, *, expected_size: int | None = None) -> None:
        raw_records.append(raw)
        if type(raw.get("returncode")) is not int:
            failure("COLLECTION_FAILED", "Runner did not return an integer process exit code")
        if raw.get("timed_out"):
            failure("COLLECTION_TIMEOUT", "RCCL benchmark exceeded timeout_s")
        if raw.get("output_truncated"):
            failure("COLLECTION_OUTPUT_LIMIT", "RCCL benchmark exceeded the output capture bound")
        if raw["returncode"] != 0:
            failure("COLLECTION_FAILED", f"RCCL benchmark exited with code {raw['returncode']}")
        output, error = raw.get("stdout"), raw.get("stderr", "")
        if not isinstance(output, str) or not isinstance(error, str):
            failure("COLLECTION_FAILED", "Runner stdout/stderr must be text")
        if len(output.encode()) + len(error.encode()) > limit:
            failure("COLLECTION_OUTPUT_LIMIT", "Runner output exceeded the declared capture bound")
        if re.search(r"\b(?:ERROR|FAILED|ABORTED)\b", error):
            failure("COLLECTION_FAILED", "RCCL stderr reports a benchmark error")
        try:
            parsed = parse_rccl_output(output, ranks=ranks, mode=mode)
        except SimulationError as exc:
            failure(exc.code, str(exc))
        if expected_size is not None and (len(parsed["records"]) != 1
                                          or parsed["records"][0]["message_bytes"] != expected_size):
            failure("INVALID_RCCL_OUTPUT", "Benchmark output does not match its one-size command")
        if any(row["datatype"] != datatype_binding.get("rccl_datatype") for row in parsed["records"]):
            failure("HARDWARE_IDENTITY_MISMATCH", "RCCL table datatype differs from the experiment activation dtype")
        parsed["metadata"]["communication_datatype"] = dict(datatype_binding)
        metadata.append(parsed["metadata"])
        parsed_records.extend(parsed["records"])

    if "raw_path" in plan:
        if runner is not None or any(key in plan for key in ("binary", "message_bytes", "repeats", "iterations", "warmup_iterations", "timeout_s")):
            _fail("raw_path cannot be combined with live-run controls", "INVALID_COLLECTION_PLAN")
        if plan.get("algorithm") != "Ring" or plan.get("rank_aggregation") != "max":
            _fail("Raw logs must explicitly declare algorithm=Ring and rank_aggregation=max", "INVALID_COLLECTION_PLAN")
        source = _source(plan.get("source"), allow_test_fixture)
        datatype_binding = _communication_dtype(experiment, fixture=source["kind"] == "test_fixture")
        path = base / _text(plan["raw_path"], "raw_path")
        try:
            with path.open("rb") as stream:
                data = stream.read(limit + 1)
        except OSError as exc:
            failure("IO_ERROR", f"Cannot read RCCL capture: {exc}")
        if len(data) > limit:
            failure("COLLECTION_OUTPUT_LIMIT", "RCCL input exceeds max_output_bytes")
        try:
            output = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            failure("INVALID_RCCL_OUTPUT", f"RCCL capture is not UTF-8: {exc}")
        consume({"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                 "stdout": output, "stderr": "", "returncode": 0,
                 "timed_out": False, "output_truncated": False,
                 "launch_contract": {"algorithm": "Ring", "rank_aggregation": "max", "source": "declared_import_metadata"}})
        if source["kind"] == "measured":
            observed = metadata[0]
            if observed["rccl_tests_version"] is None or observed["rccl_library_version"] is None:
                failure("HARDWARE_METADATA_INCOMPLETE", "Measured import needs benchmark and loaded RCCL version banners")
            if observed["rccl_library_version"] != effective.calibration.rccl_version:
                failure("HARDWARE_IDENTITY_MISMATCH", "Imported RCCL banner differs from declared library identity")
            if effective.calibration.rocm_version.strip().lower() in {"unknown", "unspecified", "not-applicable", "n/a"}:
                failure("HARDWARE_METADATA_INCOMPLETE", "RCCL import requires an applicable observed ROCm version")
            for index, device in enumerate(observed["devices"]):
                name = device["name"]
                if name != effective.infra.devices[index].name or not any(
                        vendor in name.upper() for vendor in ("AMD", "RADEON", "INSTINCT")):
                    failure("HARDWARE_IDENTITY_MISMATCH", "Measured RCCL device names must match the declared AMD devices")
            effective = replace(effective, calibration=replace(effective.calibration,
                runtime_name="rccl-tests", runtime_version=observed["rccl_tests_version"]))
        validate_identity(profile_identity(effective), measured=source["kind"] == "measured")
    else:
        if "source" in plan:
            _fail("Live capture source is derived from its execution, not a plan label", "INVALID_COLLECTION_PLAN")
        if plan.get("algorithm", "Ring") != "Ring" or plan.get("rank_aggregation", "max") != "max":
            _fail("Live capture pins Ring and max-rank timing", "INVALID_COLLECTION_PLAN")
        reference = _text(plan.get("source_reference"), "source_reference")
        binary = _text(plan.get("binary"), "binary")
        if Path(binary).name != "all_reduce_perf":
            _fail("binary must be all_reduce_perf; SendRecv is not isolated PP P2P", "INVALID_COLLECTION_PLAN")
        sizes = plan.get("message_bytes")
        if not isinstance(sizes, list) or not sizes:
            _fail("message_bytes must be a nonempty unique integer array", "INVALID_COLLECTION_PLAN")
        datatype_binding = _communication_dtype(experiment, fixture=runner is not None)
        datatype = datatype_binding["rccl_datatype"]
        element_bytes = _WIDTHS[datatype]
        for size in sizes:
            _integer(size, "message_bytes[]", element_bytes)
            if size % element_bytes:
                _fail("AllReduce payload bytes must align with the activation datatype", "INVALID_COLLECTION_PLAN")
        if len(set(sizes)) != len(sizes):
            _fail("message_bytes must not contain duplicates", "INVALID_COLLECTION_PLAN")
        repeats = _integer(plan.get("repeats", 1), "repeats", 1)
        iterations = _integer(plan.get("iterations", 20), "iterations", 1)
        warmup = _integer(plan.get("warmup_iterations", 5), "warmup_iterations")
        timeout = _number(plan.get("timeout_s", 60), "timeout_s", positive=True)
        if repeats * len(sizes) > 10000 or iterations > 10_000_000 or warmup > 10_000_000:
            _fail("RCCL plan exceeds bounded collection size", "INVALID_COLLECTION_PLAN")
        for key in ("NCCL_TESTS_SPLIT", "NCCL_TESTS_SPLIT_MASK"):
            if os.environ.get(key):
                _fail("Split RCCL groups are unsupported; clear NCCL_TESTS_SPLIT settings", "INVALID_COLLECTION_PLAN")
        if os.environ.get("NCCL_TESTS_DEVICE", "0") != "0":
            _fail("NCCL_TESTS_DEVICE must not remap the visible-device order", "INVALID_COLLECTION_PLAN")
        for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE"):
            if os.environ.get(key) not in (None, "", "1"):
                _fail("Live capture is a single-process benchmark, not an MPI launch", "INVALID_COLLECTION_PLAN")
        if runner is not None:
            if not allow_test_fixture:
                _fail("Injected runners require allow_test_fixture", "TEST_FIXTURE_NOT_ALLOWED")
            inventory = deepcopy(inventory) if inventory is not None else probe_rocm()
            inventory["source"] = {"kind": "test_fixture", "reference": reference}
        elif inventory is None:
            inventory = probe_rocm()
        if inventory.get("status") != "READY":
            failure("HARDWARE_UNAVAILABLE", "RCCL collection requires available AMD HIP devices")
        if runner is None and inventory.get("source", {}).get("kind") != "measured":
            _fail("Live execution cannot promote fixture inventory", "TEST_FIXTURE_NOT_ALLOWED")
        source = {"kind": "test_fixture" if runner is not None else "measured", "reference": reference}
        if runner is None:
            candidate = (base / binary).resolve() if "/" in binary else None
            located = str(candidate) if candidate is not None else shutil.which(binary)
            if located is None or not os.access(located, os.X_OK):
                failure("HARDWARE_UNAVAILABLE", "all_reduce_perf executable was not found")
            binary = located
        env = dict(os.environ)
        env["NCCL_ALGO"] = "Ring"
        env["NCCL_DEBUG"] = "VERSION"
        captured_env = {key: env[key] for key in ("NCCL_ALGO", "NCCL_DEBUG", "HIP_VISIBLE_DEVICES",
                                                "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if key in env}
        executable_hash = None
        if Path(binary).is_file():
            digest = hashlib.sha256()
            with Path(binary).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            executable_hash = digest.hexdigest()
        execute = _run_bounded if runner is None else runner
        # Establish the known library metadata before executing the benchmark.
        # Its separate benchmark identity is bound only after observing its header.
        versions_object = inventory.get("versions")
        library_version = versions_object.get("rccl_version") if isinstance(versions_object, dict) else None
        effective = bind_inventory(experiment, inventory, runtime_name="RCCL",
                                   runtime_version=library_version or "unknown", require_rccl=True,
                                   allow_test_fixture=allow_test_fixture)
        for repeat in range(repeats):
            for size in sizes:
                argv = [binary, "-b", str(size), "-e", str(size), "-g", str(ranks), "-t", "1",
                        "-a", "3", "-c", "1", "-C", "0", "-G", "0", "-N", "1",
                        "-O", "1" if mode == "out_of_place" else "0", "-d", datatype, "-o", "sum",
                        "-n", str(iterations), "-w", str(warmup)]
                try:
                    captured = execute(argv, timeout_s=timeout, max_output_bytes=limit, env=env)
                except subprocess.TimeoutExpired as exc:
                    partial = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
                    error = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
                    raw_records.append({"argv": argv, "stdout": partial[:limit], "stderr": error[:limit],
                                        "returncode": -1, "timed_out": True})
                    failure("COLLECTION_TIMEOUT", "RCCL runner timed out")
                except (OSError, RuntimeError) as exc:
                    raw_records.append({"argv": argv, "stdout": "", "stderr": str(exc),
                                        "returncode": None, "execution_error": type(exc).__name__})
                    failure("COLLECTION_FAILED", f"Cannot execute RCCL benchmark: {exc}")
                if not isinstance(captured, dict):
                    failure("COLLECTION_FAILED", "Runner must return a capture object")
                raw = {**captured, "argv": argv, "environment": captured_env,
                       "executable_sha256": executable_hash, "repeat": repeat,
                       "requested_message_bytes": size,
                       "communication_datatype": dict(datatype_binding),
                       "launch_contract": {"algorithm": "Ring", "rank_aggregation": "max", "source": "fixed_runner_arguments"}}
                consume(raw, expected_size=size)
        versions = {item["rccl_tests_version"] for item in metadata}
        if len(versions) != 1 or None in versions:
            failure("HARDWARE_METADATA_INCOMPLETE", "Capture needs one observed rccl-tests version")
        for item in metadata:
            if item["rccl_library_version"] is None:
                failure("HARDWARE_METADATA_INCOMPLETE", "No loaded RCCL version banner was captured")
            if item["rccl_library_version"] != inventory["versions"]["rccl_version"]:
                failure("HARDWARE_IDENTITY_MISMATCH", "Benchmark RCCL banner differs from the selected library version")
            for index, device in enumerate(item["devices"]):
                if device["name"] != inventory["devices"][index]["name"]:
                    failure("HARDWARE_IDENTITY_MISMATCH", "Benchmark devices differ from the probed visible-device order")
        effective = bind_inventory(experiment, inventory, runtime_name="rccl-tests",
                                   runtime_version=next(iter(versions)), require_rccl=True,
                                   allow_test_fixture=allow_test_fixture)
    records = [{"record_id": f"{dataset_id}:{index}", "group_id": f"allreduce-r{ranks}-{mode}",
                "primitive": "allreduce", "ranks": ranks, "message_bytes": row["message_bytes"],
                "duration_ms": row["duration_ms"]} for index, row in enumerate(parsed_records)]
    dataset = _validate_dataset({"schema_version": 1, "kind": "communication_measurements",
        "dataset_id": dataset_id, "identity": profile_identity(effective), "source": source,
        "records": records}, allow_test_fixture=allow_test_fixture)
    return {"status": "COMPLETED", "kind": "rccl_collection", "source": source,
            "inventory": inventory, "experiment": asdict(effective), "dataset": dataset,
            "raw_records": raw_records, "capture_metadata": metadata,
            "communication_datatype": datatype_binding,
            "hardware_validated": False,
            "limitations": [
                "Only checked single-node one-process Ring AllReduce is collected.",
                "Raw-import algorithm and max-rank launch facts are explicit declarations, not inferred from table timings.",
                "The rccl-tests time column is a benchmark iteration statistic, not process wall time or model-stage time.",
                "Communication samples are bound to the explicit activation datatype; a fit is not universal across datatypes.",
                "SendRecv is bidirectional ring traffic and is not imported as isolated PP P2P.",
                "A successful capture does not establish hardware prediction accuracy.",
            ]}
