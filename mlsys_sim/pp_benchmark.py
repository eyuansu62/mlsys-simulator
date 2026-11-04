"""Bounded, isolated native PP primitive collection and strict rank reduction.

The benchmark never times a Runtime receive wait or divides a round trip. Each
sample is one completed unidirectional P2P or full-output AllGather operation.
PyTorch is needed only in the explicitly launched live worker processes.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, NoReturn
import uuid

from .config import validate_experiment
from .hardware import CollectionError, bind_inventory, integer, number, object_fields, probe_rocm, text
from .profiles import profile_identity, validate_identity
from .schema import Experiment, SimulationError


PROTOCOL = "native_pp_primitives_v1"
PROTOCOL_V2 = "native_pp_primitives_v2"
PROTOCOL_V3 = "native_pp_primitives_v3"
COMMUNICATOR_INITIALIZATION = "lazy_world_and_pair_groups_v1"
P2P_PROTOCOL_POLICY = "simple_only_thresholds_v1"
P2P_PROTOCOL_ENVIRONMENT = {"NCCL_P2P_LL_THRESHOLD": "0", "NCCL_P2P_LL128_THRESHOLD": "0"}
TIMING_BOUNDARY = "synchronized_host_primitive_completion_v1"
_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
_MAX_BYTES = 16 * 1024 * 1024
_MAX_MESSAGE = 256 * 1024 * 1024


def _fail(message: str, code: str = "INVALID_COLLECTION_PLAN") -> NoReturn:
    raise SimulationError(code, message)


def _session_id(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        _fail("session_id must be a bounded filename-safe cohort identifier", "INVALID_PP_COMMUNICATION_CAPTURE")
    return value


def benchmark_cases(experiment: Experiment) -> list[dict]:
    """Ordered placement cases; every world rank creates these groups in order."""
    tp, pp = experiment.strategy.tensor_parallel_size, experiment.strategy.pipeline_parallel_size
    result = []
    for stage in range(1, pp):
        for rank in range(tp):
            result.append({"primitive": "p2p", "stage_index": stage, "tp_rank": rank,
                "participants": [(stage - 1) * tp + rank, stage * tp + rank]})
        if tp > 1:
            result.append({"primitive": "allgather", "stage_index": stage, "tp_rank": None,
                "participants": list(range(stage * tp, (stage + 1) * tp))})
    return result


def normalize_pp_benchmark_plan(experiment: Experiment, value: Any) -> dict:
    """Validate finite workload bounds before importing or launching anything."""
    validate_experiment(experiment)
    strategy = experiment.strategy
    if (strategy.replicas != 1 or strategy.pipeline_parallel_size <= 1
            or strategy.pipeline_transport != "vllm_native"):
        _fail("PP primitive collection requires one replica with native PP transport")
    width = _DTYPE_BYTES.get(strategy.model.activation_dtype)
    if width is None or width != strategy.model.activation_bytes:
        _fail("PP primitive collection requires an explicit matching activation dtype")
    world = strategy.tensor_parallel_size * strategy.pipeline_parallel_size
    if world > 32 or len(experiment.infra.devices) != world:
        _fail("PP primitive collection supports exactly one local device per rank, at most 32 ranks")
    plan = object_fields(value, {"schema_version", "dataset_id", "message_bytes"},
        {"source_reference", "warmup_iterations", "iterations", "repeats", "timeout_s",
         "max_output_bytes", "raw_path", "source", "protocol"}, "PP communication plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        _fail("PP communication plan schema_version must be 1")
    if plan.get("protocol", PROTOCOL) not in (PROTOCOL, PROTOCOL_V2, PROTOCOL_V3):
        _fail("PP communication plan protocol must select native_pp_primitives_v1, v2 or v3")
    identifier = text(plan["dataset_id"], "dataset_id")
    if len(identifier) > 128 or any(character in identifier for character in "\x00\r\n"):
        _fail("dataset_id must be a bounded nonempty identifier")
    sizes = plan["message_bytes"]
    if not isinstance(sizes, list) or not 2 <= len(sizes) <= 32:
        _fail("message_bytes needs between 2 and 32 distinct sizes")
    alignment = width * strategy.tensor_parallel_size
    for size in sizes:
        integer(size, "message_bytes[]", 1)
        if size > _MAX_MESSAGE or size % alignment:
            _fail("Message sizes must be <=256MiB and align to TP times activation element size")
    if len(set(sizes)) != len(sizes):
        _fail("message_bytes must contain distinct sizes")
    warmup = integer(plan.get("warmup_iterations", 3), "warmup_iterations", 1)
    iterations = integer(plan.get("iterations", 10), "iterations", 1)
    repeats = integer(plan.get("repeats", 3), "repeats", 1)
    timeout = number(plan.get("timeout_s", 120), "timeout_s", positive=True)
    limit = integer(plan.get("max_output_bytes", 8 * 1024 * 1024), "max_output_bytes", 1)
    if timeout > 86400 or limit > _MAX_BYTES or warmup > 10000:
        _fail("PP benchmark timeout, output or warmup exceeds its bound")
    rank_samples = sum(len(case["participants"]) for case in benchmark_cases(experiment))
    if rank_samples * len(sizes) * iterations * repeats > 100000:
        _fail("PP benchmark exceeds 100000 per-rank samples")
    if rank_samples * len(sizes) * warmup > 100000:
        _fail("PP benchmark warmup exceeds 100000 per-rank operations")
    result = {"schema_version": 1, "dataset_id": identifier, "message_bytes": list(sizes),
        "source_reference": text(plan.get("source_reference", "isolated native PP communication microbenchmark"),
                                 "source_reference"),
        "warmup_iterations": warmup, "iterations": iterations, "repeats": repeats,
        "timeout_s": timeout, "max_output_bytes": limit}
    # Absent protocol preserves the original normalized plan and fixture bytes.
    if "protocol" in plan:
        result["protocol"] = plan["protocol"]
    if "raw_path" in plan:
        result["raw_path"] = text(plan["raw_path"], "raw_path")
        result["source"] = _source(plan.get("source"))
    elif "source" in plan:
        _fail("Live collection derives source from execution rather than a plan label")
    return result


def _source(value: Any) -> dict:
    source = object_fields(value, {"kind", "reference"}, set(), "capture source")
    if source["kind"] not in ("measured", "test_fixture"):
        _fail("Capture source must be measured or test_fixture")
    text(source["reference"], "source.reference")
    return dict(source)


def worker_launch_contract(experiment: Experiment, plan: dict) -> dict:
    protocol = plan.get("protocol", PROTOCOL)
    contract = {"protocol": protocol, "world_size": len(experiment.infra.devices),
        "activation_dtype": experiment.strategy.model.activation_dtype,
        "message_bytes": list(plan["message_bytes"]), "warmup_iterations": plan["warmup_iterations"],
        "iterations": plan["iterations"], "repeats": plan["repeats"],
        "control_backend": "gloo", "primitive_backend": "nccl",
        "group_creation": "all_world_ranks_same_order", "p2p_api": "isend_irecv",
        "allgather_api": "all_gather_into_tensor", "buffer_policy": "preallocated_out_of_place"}
    if protocol in (PROTOCOL_V2, PROTOCOL_V3):
        contract.update(world_init_device_id=False,
                        communicator_initialization=COMMUNICATOR_INITIALIZATION)
    if protocol == PROTOCOL_V3:
        contract.update(p2p_protocol_policy=P2P_PROTOCOL_POLICY,
                        p2p_protocol_environment=dict(P2P_PROTOCOL_ENVIRONMENT))
    return contract


def sample_descriptor(plan: dict, case: dict, size: int, repeat: int, iteration: int) -> dict:
    local_rank = case["tp_rank"] if case["tp_rank"] is not None else "all"
    cell = f"{plan['dataset_id']}/{case['primitive']}/s{case['stage_index']}/tp{local_rank}/b{size}"
    return {**case, "message_bytes": size, "record_id": f"{cell}/r{repeat}/i{iteration}",
            "group_id": cell, "repeat": repeat, "iteration": iteration}


def _read_bounded_json(path: Path, limit: int) -> tuple[Any, dict]:
    if path.is_symlink():
        _fail("Capture files may not be symlinks")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("Duplicate JSON key in PP capture")
            result[key] = value
        return result
    try:
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            _fail("PP capture exceeds max_output_bytes", "COLLECTION_OUTPUT_LIMIT")
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: _fail("PP capture must contain finite JSON"))
        json.dumps(value, allow_nan=False)
    except SimulationError:
        raise
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise SimulationError("INVALID_PP_COMMUNICATION_CAPTURE", f"Cannot read PP capture: {exc}") from exc
    return value, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _stop_process_group(process: subprocess.Popen) -> None:
    # The launcher owns a new session, so descendants share this process group.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass
    process.wait()


def _run_bounded(argv: list[str], *, timeout_s: float, max_output_bytes: int,
                 env: dict[str, str], output_dir: Path) -> dict:
    """Bound the owned benchmark session, including workers and capture files."""
    if os.name != "posix":
        _fail("Live HIP primitive launch requires a POSIX process-group boundary", "HARDWARE_UNAVAILABLE")
    start, timed_out, truncated = time.monotonic(), False, False
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env,
                                   shell=False, start_new_session=True)
        try:
            while process.poll() is None:
                size = os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size
                size += sum(path.stat().st_size for path in output_dir.glob("rank-*.json") if path.is_file())
                if size > max_output_bytes:
                    truncated = True
                    break
                if time.monotonic() - start >= timeout_s:
                    timed_out = True
                    break
                time.sleep(.01)
        finally:
            # Also reap an orphaned child if a launcher exited unexpectedly.
            _stop_process_group(process)
        stdout.seek(0)
        stderr.seek(0)
        left = stdout.read(max_output_bytes + 1)
        right = stderr.read(max_output_bytes + 1)
    captures, inputs = [], []
    consumed = len(left) + len(right)
    truncated |= consumed > max_output_bytes
    for path in sorted(output_dir.glob("rank-*.json")):
        try:
            value, info = _read_bounded_json(path, max(1, max_output_bytes - min(consumed, max_output_bytes)))
        except SimulationError as exc:
            captures.append({"path": str(path), "capture_read_error": str(exc)})
            truncated |= exc.code == "COLLECTION_OUTPUT_LIMIT"
            continue
        consumed += info["bytes"]
        captures.append(value)
        inputs.append(info)
    return {"returncode": process.returncode, "stdout": left[:max_output_bytes].decode("utf-8", errors="replace"),
        "stderr": right[:max(0, max_output_bytes - len(left))].decode("utf-8", errors="replace"),
        "timed_out": timed_out, "output_truncated": truncated or consumed > max_output_bytes,
        "process_elapsed_ms": (time.monotonic() - start) * 1000,
        "rank_captures": captures, "rank_capture_inputs": inputs}


def _aggregate(experiment: Experiment, plan: dict, values: Any, source: dict,
               inventory: dict | None, *, expected_session: str | None = None) -> tuple[list[dict], dict, dict, str]:
    world = len(experiment.infra.devices)
    if not isinstance(values, list) or len(values) != world:
        _fail("A complete capture from every world rank is required", "INVALID_PP_COMMUNICATION_CAPTURE")
    expected_launch = worker_launch_contract(experiment, plan)
    expected_identity = profile_identity(experiment)
    ranks: dict[int, dict] = {}
    stacks = []
    sessions: set[str] = set()
    for value in values:
        capture = object_fields(value, {"schema_version", "kind", "protocol", "session_id", "rank", "world_size", "identity",
            "source", "status", "stack", "device", "launch", "records"}, {"error_code", "message"}, "rank capture")
        rank = integer(capture["rank"], "rank")
        if rank >= world or rank in ranks or capture["world_size"] != world or type(capture["world_size"]) is not int:
            _fail("Missing, repeated or conflicting world ranks", "INVALID_PP_COMMUNICATION_CAPTURE")
        if (type(capture["schema_version"]) is not int or capture["schema_version"] != 1
                or capture["kind"] != "native_pp_primitive_rank_capture"
                or capture["protocol"] != plan.get("protocol", PROTOCOL)
                or capture["status"] != "COMPLETED"):
            _fail("Only complete versioned primitive rank captures can form measurements", "INVALID_PP_COMMUNICATION_CAPTURE")
        declared_source = _source(capture["source"])
        session = _session_id(capture["session_id"])
        sessions.add(session)
        if expected_session is not None and session != expected_session:
            _fail("Rank capture does not belong to this launch session", "CALIBRATION_MISMATCH")
        if declared_source["kind"] == "test_fixture":
            source = {"kind": "test_fixture", "reference": source["reference"]}
        if (validate_identity(capture["identity"]) != expected_identity
                or json.dumps(capture["launch"], sort_keys=True, separators=(",", ":"), allow_nan=False)
                != json.dumps(expected_launch, sort_keys=True, separators=(",", ":"), allow_nan=False)):
            _fail("Rank capture target identity or primitive launch differs from the requested plan", "CALIBRATION_MISMATCH")
        fact = object_fields(capture["device"], {"index", "name", "memory_bytes"}, {"architecture"}, "rank device")
        expected_device = experiment.infra.devices[rank]
        if (type(fact["index"]) is not int or fact["index"] != rank or type(fact["memory_bytes"]) is not int
                or fact["name"] != expected_device.name or fact["memory_bytes"] != expected_device.memory_bytes):
            _fail("Rank's actual device differs from its ordered visible-device binding", "HARDWARE_IDENTITY_MISMATCH")
        stack = capture["stack"]
        if not isinstance(stack, dict):
            _fail("Rank stack must remain an object", "INVALID_PP_COMMUNICATION_CAPTURE")
        if plan.get("protocol", PROTOCOL) == PROTOCOL_V3:
            policy_environment = stack.get("p2p_protocol_environment")
            if type(policy_environment) is not dict or policy_environment != P2P_PROTOCOL_ENVIRONMENT:
                _fail("Every v3 rank must observe both P2P thresholds as canonical string zero",
                      "CALIBRATION_MISMATCH")
        json.dumps(stack, allow_nan=False)
        stacks.append(stack)
        if not isinstance(capture["records"], list):
            _fail("Rank records must be an array", "INVALID_PP_COMMUNICATION_CAPTURE")
        ranks[rank] = capture
    if len(sessions) != 1:
        _fail("Rank captures must belong to one complete communication session", "CALIBRATION_MISMATCH")
    first_stack = json.dumps(stacks[0], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if any(json.dumps(stack, sort_keys=True, separators=(",", ":"), allow_nan=False) != first_stack
           for stack in stacks[1:]):
        _fail("Primitive ranks must share an observed software/library stack", "HARDWARE_IDENTITY_MISMATCH")
    stack = deepcopy(stacks[0])
    if source["kind"] == "measured":
        required = ("torch_version", "hip_version", "rccl_version", "rccl_library_sha256",
                    "communication_environment_sha256")
        if any(not isinstance(stack.get(key), str) or not stack[key] for key in required):
            _fail("Measured primitives require actual Torch/HIP/loaded-RCCL facts", "HARDWARE_METADATA_INCOMPLETE")
        if stack["rccl_version"] != experiment.calibration.rccl_version:
            _fail("Worker loaded RCCL version differs from the target experiment", "HARDWARE_IDENTITY_MISMATCH")
        digest = stack["rccl_library_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            _fail("Worker loaded RCCL requires a SHA256 digest", "HARDWARE_METADATA_INCOMPLETE")
        fingerprint = stack["communication_environment_sha256"]
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            _fail("Communication environment must have its actual SHA256 digest", "HARDWARE_METADATA_INCOMPLETE")
        if inventory is not None and inventory.get("status") == "READY":
            library = inventory.get("rccl_library")
            if not isinstance(library, dict) or library.get("sha256") != digest:
                _fail("Worker's mapped RCCL differs from the probed library", "HARDWARE_IDENTITY_MISMATCH")
            torch_info = inventory.get("torch", {})
            if stack["torch_version"] != torch_info.get("version") or stack["hip_version"] != torch_info.get("hip_version"):
                _fail("Worker Torch/HIP differs from the probe", "HARDWARE_IDENTITY_MISMATCH")
    expected: dict[str, dict] = {}
    for case in benchmark_cases(experiment):
        for size in plan["message_bytes"]:
            for repeat in range(plan["repeats"]):
                for iteration in range(plan["iterations"]):
                    row = sample_descriptor(plan, case, size, repeat, iteration)
                    expected[row["record_id"]] = row
    gathered: dict[str, dict[int, dict]] = {}
    required_row = {"record_id", "group_id", "primitive", "stage_index", "tp_rank", "participants",
                    "message_bytes", "repeat", "iteration", "duration_ms", "correctness_errors"}
    for rank, capture in ranks.items():
        expected_count = sum(rank in row["participants"] for row in expected.values())
        if len(capture["records"]) != expected_count:
            _fail("Rank records omit or add benchmark iterations", "INVALID_PP_COMMUNICATION_CAPTURE")
        for value in capture["records"]:
            row = object_fields(value, required_row, set(), "primitive rank sample")
            identifier = row["record_id"]
            if not isinstance(identifier, str) or identifier not in expected:
                _fail("Unexpected primitive iteration", "INVALID_PP_COMMUNICATION_CAPTURE")
            target = expected[identifier]
            if not isinstance(row["participants"], list) or any(type(participant) is not int
                                                               for participant in row["participants"]):
                _fail("Primitive participants must be exact integer ranks", "INVALID_PP_COMMUNICATION_CAPTURE")
            if any(type(row[key]) is not type(target[key]) or row[key] != target[key] for key in target):
                _fail("Primitive rank sample differs from its declared one-way/full-output operation", "INVALID_PP_COMMUNICATION_CAPTURE")
            if rank not in target["participants"] or rank in gathered.setdefault(identifier, {}):
                _fail("Duplicate or nonparticipant primitive sample", "INVALID_PP_COMMUNICATION_CAPTURE")
            if type(row["correctness_errors"]) is not int or row["correctness_errors"] != 0:
                _fail("Every primitive sample must pass complete payload correctness checks", "COMMUNICATION_CORRECTNESS_FAILED")
            number(row["duration_ms"], "duration_ms", positive=True)
            gathered[identifier][rank] = row
    records = []
    for identifier, expected_row in expected.items():
        measured = gathered.get(identifier, {})
        if set(measured) != set(expected_row["participants"]):
            _fail("Primitive sample lacks a participant timing", "INVALID_PP_COMMUNICATION_CAPTURE")
        durations = [{"rank": rank, "duration_ms": measured[rank]["duration_ms"]}
                     for rank in expected_row["participants"]]
        row = {key: deepcopy(value) for key, value in expected_row.items() if key not in ("repeat", "iteration")}
        row.update(duration_ms=max(item["duration_ms"] for item in durations),
                   rank_durations_ms=durations, correctness_errors=0)
        records.append(row)
    return records, stack, source, next(iter(sessions))


def collect_pp_communication(experiment: Experiment, plan: dict, inventory: dict | None = None, *,
                             base_dir: str | Path | None = None, allow_test_fixture: bool = False,
                             runner: Any = None) -> dict:
    """Run or import complete isolated primitive captures without hardware claims."""
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be boolean")
    settings = normalize_pp_benchmark_plan(experiment, plan)
    if runner is not None and not callable(runner):
        _fail("Injected runner must be callable")
    if runner is not None and (not allow_test_fixture or "raw_path" in settings):
        _fail("Injected runners require fixture opt-in and cannot import raw files", "TEST_FIXTURE_NOT_ALLOWED")
    effective = experiment
    raw_records: list[dict] = []
    reference = settings["source_reference"]
    source = {"kind": "test_fixture" if runner is not None else "measured", "reference": reference}
    expected_session: str | None = None

    def failure(code: str, message: str) -> NoReturn:
        raise CollectionError(code, message, artifacts={"kind": "native_pp_communication_collection",
            "experiment": asdict(effective), "inventory": deepcopy(inventory),
            "raw_records": deepcopy(raw_records), "hardware_validated": False})

    try:
        if "raw_path" in settings:
            source = settings["source"]
            if source["kind"] == "test_fixture" and not allow_test_fixture:
                _fail("Fixture imports require explicit permission", "TEST_FIXTURE_NOT_ALLOWED")
            base = Path.cwd() if base_dir is None else Path(base_dir)
            value, info = _read_bounded_json(base / settings["raw_path"], settings["max_output_bytes"])
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if not isinstance(value, dict) or value.get("kind") != "native_pp_primitives_capture":
                _fail("Raw input must be one exported native PP primitive capture", "INVALID_PP_COMMUNICATION_CAPTURE")
            captured = deepcopy(value)
            captured["raw_input"] = info
        else:
            for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
                if os.environ.get(name) not in (None, "", "1"):
                    _fail("Launch from outside an existing distributed process group")
            environment = dict(os.environ)
            if settings.get("protocol", PROTOCOL) == PROTOCOL_V3:
                for name, expected in P2P_PROTOCOL_ENVIRONMENT.items():
                    if environment.get(name, expected) != expected:
                        _fail(f"v3 requires {name}=0; an existing conflicting value cannot be overridden")
                    environment[name] = expected
            if runner is None:
                inventory = probe_rocm() if inventory is None else deepcopy(inventory)
                if inventory.get("source", {}).get("kind") != "measured":
                    _fail("Live execution cannot promote a fixture inventory", "TEST_FIXTURE_NOT_ALLOWED")
                effective = bind_inventory(experiment, inventory,
                    runtime_name=experiment.calibration.runtime_name,
                    runtime_version=experiment.calibration.runtime_version, require_rccl=True)
            elif inventory is not None:
                inventory = deepcopy(inventory)
                inventory["source"] = {"kind": "test_fixture", "reference": reference}
            environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
            captured_env = {key: environment[key] for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_DEBUG",
                "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_SHM_DISABLE", "NCCL_NET", "NCCL_IB_DISABLE",
                "NCCL_P2P_LL_THRESHOLD", "NCCL_P2P_LL128_THRESHOLD", "NCCL_SHM_USE_CUDA_MEMCPY", "NCCL_SHM_MEMCPY_MODE",
                "TORCH_NCCL_BLOCKING_WAIT", "NCCL_BLOCKING_WAIT", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                "NCCL_ASYNC_ERROR_HANDLING",
                "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
                if key in environment}
            with tempfile.TemporaryDirectory(prefix="mlsys-pp-primitives-") as temporary:
                root = Path(temporary)
                config = root / "config.json"
                output = root / "rank-captures"
                output.mkdir()
                expected_session = uuid.uuid4().hex
                config.write_text(json.dumps({"schema_version": 1, "experiment": asdict(effective),
                    "plan": {key: value for key, value in settings.items() if key not in ("raw_path", "source")},
                    "session_id": expected_session, "rendezvous": str(root / "rendezvous")},
                    allow_nan=False), encoding="utf-8")
                argv = [sys.executable, "-m", "mlsys_sim.pp_benchmark_worker", "--launch-world",
                    "--config", str(config), "--output-dir", str(output)]
                execute = _run_bounded if runner is None else runner
                captured = execute(argv, timeout_s=settings["timeout_s"],
                    max_output_bytes=settings["max_output_bytes"], env=environment, output_dir=output)
                if not isinstance(captured, dict):
                    _fail("Primitive runner must return a capture object", "COLLECTION_FAILED")
                captured = deepcopy(captured)
                captured.update(argv=argv, environment=captured_env)
                if settings.get("protocol", PROTOCOL) in (PROTOCOL_V2, PROTOCOL_V3) and "protocol" not in captured:
                    captured["protocol"] = settings["protocol"]
        captured.update(schema_version=1, kind="native_pp_primitives_capture")
        raw_records.append(captured)
        if captured.get("protocol", PROTOCOL) != settings.get("protocol", PROTOCOL):
            failure("CALIBRATION_MISMATCH", "Capture envelope protocol differs from the requested plan")
        if type(captured.get("returncode")) is not int:
            failure("COLLECTION_FAILED", "Primitive launcher did not provide an integer exit code")
        if captured.get("timed_out"):
            failure("COLLECTION_TIMEOUT", "Primitive launcher exceeded timeout_s")
        if captured.get("output_truncated"):
            failure("COLLECTION_OUTPUT_LIMIT", "Primitive launcher exceeded its complete-output bound")
        if captured["returncode"] != 0:
            failure("COLLECTION_FAILED", f"Primitive launcher exited with code {captured['returncode']}")
        if len(json.dumps(captured, allow_nan=False).encode()) > settings["max_output_bytes"]:
            failure("COLLECTION_OUTPUT_LIMIT", "Primitive evidence exceeds max_output_bytes")
        records, stack, source, session = _aggregate(effective, settings, captured.get("rank_captures"), source,
                                                    inventory, expected_session=expected_session)
        if "session_id" in captured and captured["session_id"] != session:
            failure("CALIBRATION_MISMATCH", "Capture envelope and rank session disagree")
        captured["session_id"] = session
        if source["kind"] == "test_fixture" and not allow_test_fixture:
            failure("TEST_FIXTURE_NOT_ALLOWED", "Any injected/fixture rank keeps the complete dataset a fixture")
        validate_identity(profile_identity(effective), measured=source["kind"] == "measured")
        benchmark = {"protocol": settings.get("protocol", PROTOCOL), "timing_boundary": TIMING_BOUNDARY,
            "p2p_direction": "unidirectional", "allgather_bytes": "full_output",
            "rank_reduction": "maximum_local_duration", "barrier_in_timing": False,
            "warmup_in_records": False, "startup_in_timing": False, "allocation_in_timing": False,
            "buffer_policy": "preallocated_out_of_place", "backend": "nccl", "stack": stack,
            "launch": {"session_id": session, "rank_contract": worker_launch_contract(effective, settings),
                       "argv": captured.get("argv"), "environment": captured.get("environment", {})}}
        if settings.get("protocol", PROTOCOL) in (PROTOCOL_V2, PROTOCOL_V3):
            benchmark["communicator_initialization"] = COMMUNICATOR_INITIALIZATION
        if settings.get("protocol", PROTOCOL) == PROTOCOL_V3:
            benchmark.update(p2p_protocol_policy=P2P_PROTOCOL_POLICY,
                             p2p_protocol_environment=dict(P2P_PROTOCOL_ENVIRONMENT))
        dataset = {"schema_version": 1, "kind": "native_pp_communication_measurements",
            "dataset_id": settings["dataset_id"], "identity": profile_identity(effective),
            "source": source, "records": records, "benchmark": benchmark,
            "notes": {"target_identity_scope": "declared_model_runtime_target_not_model_execution",
                "timer": "one_local_completed_primitive_per_iteration",
                "allgather_allocation_excluded": True, "source_wait_used_as_network_latency": False}}
        from .pp_communication import validate_pp_communication_dataset
        dataset = validate_pp_communication_dataset(dataset, allow_test_fixture=allow_test_fixture)
    except CollectionError:
        raise
    except SimulationError as exc:
        failure(exc.code, str(exc))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        failure("COLLECTION_FAILED", f"Primitive collection failed: {exc}")
    return {"schema_version": 1, "status": "COMPLETED", "kind": "native_pp_communication_collection",
        "source": source, "experiment": asdict(effective), "inventory": deepcopy(inventory),
        "dataset": dataset, "raw_records": raw_records, "hardware_validated": False,
        "provenance": {"protocol": settings.get("protocol", PROTOCOL), "session_id": session,
            "rank_aggregation": "maximum_local_duration",
            "target_identity_scope": "declared_model_runtime_target_not_model_execution"},
        "limitations": ["A synchronized standalone primitive is not a Runtime dependency wait or round-trip estimate.",
            "Output allocation, metadata ingress and upstream/previous-send waiting are excluded.",
            "Independent uncontended primitive measurements do not establish end-to-end hardware accuracy."]}
