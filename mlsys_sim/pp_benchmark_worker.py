"""One-GPU-per-rank native PP microbenchmark worker and owned local launcher.

Importing this module does not import Torch or initialize distributed state.
The parent launcher gives every rank the same session, allowing its bounded
supervisor to terminate all workers together on failure or timeout.
"""

import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from .config import experiment_from_dict
from .hardware import integer, object_fields
from .pp_benchmark import (P2P_PROTOCOL_ENVIRONMENT, PROTOCOL, PROTOCOL_V3,
                           _read_bounded_json, _session_id, benchmark_cases,
                           normalize_pp_benchmark_plan, sample_descriptor, worker_launch_contract)
from .profiles import profile_identity
from .schema import SimulationError


def _correctness(torch: Any, tensor: Any, expected: int) -> int:
    return int(torch.count_nonzero(tensor != expected).item())


def _execute_case(torch: Any, dist: Any, device: Any, rank: int, case: dict,
                  group: Any, control: Any, size: int, width: int, plan: dict,
                  clock: Callable[[], int], records: list[dict], progress: Callable[[], None]) -> None:
    """Preallocate, warm real operations, then measure individual completions."""
    participants = case["participants"]
    member = rank in participants
    dtype = getattr(torch, plan["activation_dtype"])
    input_tensor: Any = None
    output_tensor: Any = None
    if member:
        if case["primitive"] == "p2p":
            input_tensor = torch.empty(size // width, dtype=dtype, device=device)
        else:
            input_tensor = torch.empty(size // (width * len(participants)), dtype=dtype, device=device)
            output_tensor = torch.empty(size // width, dtype=dtype, device=device)

    def reset() -> None:
        if not member:
            return
        if case["primitive"] == "p2p":
            input_tensor.fill_((participants[0] + 1) % 32 if rank == participants[0] else -1)
        else:
            input_tensor.fill_((rank + 1) % 32)
            output_tensor.fill_(-1)
        torch.cuda.synchronize(device)

    def operation() -> None:
        if not member:
            return
        if case["primitive"] == "p2p":
            if rank == participants[0]:
                work = dist.isend(input_tensor, dst=participants[1], group=group)
            else:
                work = dist.irecv(input_tensor, src=participants[0], group=group)
            work.wait()
        else:
            # Full output bytes define the fit axis; each input is one TP slice.
            dist.all_gather_into_tensor(output_tensor, input_tensor, group=group)
        torch.cuda.synchronize(device)

    def correctness() -> int:
        if not member:
            return 0
        if case["primitive"] == "p2p":
            return _correctness(torch, input_tensor, (participants[0] + 1) % 32)
        chunk = size // (width * len(participants))
        return sum(_correctness(torch, output_tensor[index * chunk:(index + 1) * chunk],
                                (participant + 1) % 32) for index, participant in enumerate(participants))

    # A group collective does not initialize unbatched P2P's pair communicator.
    # At least one real directional operation is therefore mandatory warmup.
    for _ in range(plan["warmup_iterations"]):
        reset()
        dist.barrier(group=control)
        operation()
        if correctness():
            raise SimulationError("COMMUNICATION_CORRECTNESS_FAILED", "Primitive warmup payload was incorrect")
        dist.barrier(group=control)
    for repeat in range(plan["repeats"]):
        for iteration in range(plan["iterations"]):
            reset()
            # Gloo control keeps idle/nonparticipant GPU ranks off the fabric.
            dist.barrier(group=control)
            if member:
                start = integer(clock(), "primitive clock")
                operation()
                end = integer(clock(), "primitive clock")
                if end <= start:
                    raise SimulationError("INVALID_TIMING_SAMPLE", "Primitive completion must advance its local clock")
                errors = correctness()
                descriptor = sample_descriptor(plan, case, size, repeat, iteration)
                records.append({**descriptor, "duration_ms": (end - start) / 1e6,
                                "correctness_errors": errors})
                if errors:
                    raise SimulationError("COMMUNICATION_CORRECTNESS_FAILED", "Primitive payload was incorrect")
            dist.barrier(group=control)
        progress()


def run_worker(config: dict, *, torch_module: Any = None, clock_ns: Callable[[], int] | None = None,
               binding_provider: Callable[[], dict] | None = None,
               rank: int | None = None, local_rank: int | None = None,
               progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Execute the fixed protocol. Any backend/clock/rank injection is a fixture."""
    fixture = any(value is not None for value in (torch_module, clock_ns, binding_provider, rank, local_rank))
    value = object_fields(config, {"schema_version", "experiment", "plan", "session_id", "rendezvous"},
                          set(), "worker config")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise SimulationError("INVALID_COLLECTION_PLAN", "Worker config schema_version must be 1")
    experiment = experiment_from_dict(value["experiment"])
    plan = normalize_pp_benchmark_plan(experiment, value["plan"])
    session = _session_id(value["session_id"])
    if "raw_path" in plan:
        raise SimulationError("INVALID_COLLECTION_PLAN", "A worker cannot import captures")
    rank = integer(int(os.environ.get("RANK", "-1")) if rank is None else rank, "rank")
    local_rank = integer(int(os.environ.get("LOCAL_RANK", "-1")) if local_rank is None else local_rank, "local_rank")
    world = len(experiment.infra.devices)
    if rank >= world or rank != local_rank:
        raise SimulationError("INVALID_COLLECTION_PLAN", "Worker requires ordered single-node one-GPU ranks")
    source = {"kind": "test_fixture" if fixture else "measured",
              "reference": plan["source_reference"]}
    capture: dict[str, Any] = {"schema_version": 1, "kind": "native_pp_primitive_rank_capture",
        "protocol": plan.get("protocol", PROTOCOL), "session_id": session, "rank": rank, "world_size": world,
        "identity": profile_identity(experiment),
        "source": source, "status": "INCOMPLETE", "stack": {}, "device": {},
        "launch": worker_launch_contract(experiment, plan), "records": []}
    torch = torch_module
    initialized = False

    def progress() -> None:
        if progress_callback is not None:
            progress_callback(deepcopy(capture))

    try:
        if plan.get("protocol", PROTOCOL) == PROTOCOL_V3:
            observed = {name: os.environ.get(name) for name in P2P_PROTOCOL_ENVIRONMENT}
            capture["stack"]["p2p_protocol_environment"] = observed
            if observed != P2P_PROTOCOL_ENVIRONMENT:
                raise SimulationError("CALIBRATION_MISMATCH",
                    "v3 workers require both inherited P2P thresholds to be canonical string zero")
        torch = importlib.import_module("torch") if torch is None else torch
        if not getattr(torch.version, "hip", None) or not torch.cuda.is_available() or torch.cuda.device_count() < world:
            raise SimulationError("HARDWARE_UNAVAILABLE", "Primitive workers require available PyTorch HIP devices")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(local_rank)
        expected = experiment.infra.devices[rank]
        capture["device"] = {"index": local_rank, "name": str(properties.name),
            "memory_bytes": int(properties.total_memory), "architecture": getattr(properties, "gcnArchName", None)}
        if properties.name != expected.name or properties.total_memory != expected.memory_bytes:
            raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Actual worker GPU differs from the target rank")
        dist = torch.distributed
        if not dist.is_nccl_available() or not dist.is_gloo_available():
            raise SimulationError("HARDWARE_UNAVAILABLE", "NCCL/RCCL primitives and Gloo host control are required")
        if dist.is_initialized():
            raise SimulationError("INVALID_COLLECTION_PLAN", "Primitive worker needs a new distributed group")
        timeout = timedelta(seconds=max(1.0, min(plan["timeout_s"], 120.0)))
        rendezvous = Path(value["rendezvous"]).resolve()
        world_options: dict[str, Any] = {"backend": "nccl", "init_method": rendezvous.as_uri(), "rank": rank,
                                         "world_size": world, "timeout": timeout}
        if plan.get("protocol", PROTOCOL) == PROTOCOL:
            world_options["device_id"] = device
        # v2/v3 follow native vLLM's lazy world/group creation API. This records
        # the initialization request, not an observed RCCL algorithm choice.
        dist.init_process_group(**world_options)
        initialized = True
        if dist.get_world_size() != world or str(dist.get_backend()) != "nccl":
            raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Actual distributed backend/layout differs")
        control = dist.new_group(ranks=list(range(world)), backend="gloo", timeout=timeout,
                                 use_local_synchronization=False)
        cases = benchmark_cases(experiment)
        groups = [dist.new_group(ranks=case["participants"], backend="nccl", timeout=timeout,
                                 use_local_synchronization=False) for case in cases]
        if binding_provider is None:
            from .vllm_hooks import _loaded_rccl
            binding = _loaded_rccl()
        else:
            binding = binding_provider()
        capture["stack"] = {"torch_version": str(torch.__version__), "hip_version": str(torch.version.hip),
            "rccl_version": binding["runtime_version"], "rccl_library_sha256": binding["library_sha256"],
            "communication_environment_sha256": hashlib.sha256(json.dumps(
                {name: value for name, value in sorted(os.environ.items())
                 if name.startswith(("NCCL_", "RCCL_", "TORCH_NCCL_"))},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        if plan.get("protocol", PROTOCOL) == PROTOCOL_V3:
            observed = {name: os.environ.get(name) for name in P2P_PROTOCOL_ENVIRONMENT}
            capture["stack"]["p2p_protocol_environment"] = observed
            if observed != P2P_PROTOCOL_ENVIRONMENT:
                raise SimulationError("CALIBRATION_MISMATCH", "v3 P2P thresholds changed during worker startup")
        if binding["runtime_version"] != experiment.calibration.rccl_version and not fixture:
            raise SimulationError("HARDWARE_IDENTITY_MISMATCH", "Mapped RCCL differs from the bound experiment")
        progress()
        clock = time.perf_counter_ns if clock_ns is None else clock_ns
        width = int(experiment.strategy.model.activation_bytes)
        active_plan = {**plan, "activation_dtype": experiment.strategy.model.activation_dtype}
        for case, group in zip(cases, groups):
            for size in plan["message_bytes"]:
                _execute_case(torch, dist, device, rank, case, group, control, size, width,
                              active_plan, clock, capture["records"], progress)
        capture["status"] = "COMPLETED"
    except Exception as exc:
        capture.update(status="FAILED", error_code=exc.code if isinstance(exc, SimulationError) else "COLLECTION_FAILED",
                       message=str(exc))
    finally:
        progress()
        if initialized and capture["status"] == "COMPLETED":
            torch.distributed.destroy_process_group()
    return capture


def _write_capture(directory: Path, rank: int, value: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"rank-{rank:04d}.json"
    temporary = directory / f"rank-{rank:04d}.json.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)


def _launch_world(config_path: Path, output: Path) -> int:
    """Own direct workers which inherit the caller's bounded process session."""
    config, _ = _read_bounded_json(config_path, 4 * 1024 * 1024)
    experiment = experiment_from_dict(config["experiment"])
    normalize_pp_benchmark_plan(experiment, config["plan"])
    world = len(experiment.infra.devices)
    workers = []
    try:
        for rank in range(world):
            env = dict(os.environ)
            env.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world), LOCAL_WORLD_SIZE=str(world))
            workers.append(subprocess.Popen([sys.executable, "-m", "mlsys_sim.pp_benchmark_worker",
                "--config", str(config_path), "--output-dir", str(output), "--rank", str(rank)],
                env=env, shell=False))
        while True:
            statuses = [worker.poll() for worker in workers]
            if any(status not in (None, 0) for status in statuses):
                return 1
            if all(status is not None for status in statuses):
                return 0
            time.sleep(.01)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        deadline = time.monotonic() + .2
        for worker in workers:
            if worker.poll() is None:
                try:
                    worker.wait(timeout=max(.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    worker.kill()
            worker.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launch-world", action="store_true")
    parser.add_argument("--rank", type=int)
    args = parser.parse_args(argv)
    if args.launch_world:
        if args.rank is not None:
            parser.error("--launch-world cannot be combined with --rank")
        return _launch_world(args.config, args.output_dir)
    rank = int(os.environ.get("RANK", "-1"))
    if args.rank is None or args.rank != rank:
        parser.error("Worker rank must agree with its owned launcher environment")
    config, _ = _read_bounded_json(args.config, 4 * 1024 * 1024)
    capture = run_worker(config, progress_callback=lambda value: _write_capture(args.output_dir, rank, value))
    return 0 if capture["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
