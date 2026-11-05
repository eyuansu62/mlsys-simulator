"""Bounded fresh single/TP2/PP2 Workspace raw acquisition; no fitting/verdict.

The supervisor imports only stdlib and the dependency-free simulator config.
Only its owned child imports vLLM, and actual GPU allocation occurs in the
explicit WorkspaceWorker instances. Native SDK files use the process template.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any
import uuid


MAX_LOG_BYTES = 64 * 1024 * 1024
MAX_NATIVE_HEADER_BYTES = 16 * 1024 * 1024


def native_event_budget(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 4194304:
        raise ValueError("native-max-events must be an explicit integer in1..4194304")
    return value


def shutdown_grace(value: int) -> int:
    if type(value) is not int or not 5 <= value <= 120:
        raise ValueError("shutdown-grace-s must be an integer in5..120")
    return value


def validate_native_finalization(mode: str, cohort: str, repeats: int, policy: str) -> str:
    if policy not in {"worker_shutdown", "process_exit"}:
        raise ValueError("Unknown native finalization policy")
    if policy == "process_exit" and (mode != "pp2" or cohort != "pilot" or type(repeats) is not int or repeats != 1):
        raise ValueError("process_exit is diagnostic-only and restricted to one PP2 pilot repetition")
    return policy


def read_native_header(path: Path) -> dict[str, Any]:
    """Read actual top-level metadata before the huge event table, not a trace verdict."""
    with path.open(encoding="utf-8") as stream:
        text = stream.read(MAX_NATIVE_HEADER_BYTES)
    decoder = json.JSONDecoder()
    position = 0

    def whitespace(index: int) -> int:
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    position = whitespace(position)
    if position >= len(text) or text[position] != "{":
        raise ValueError("Actual Native final file is not a JSON object")
    position += 1
    header: dict[str, Any] = {}
    while position < len(text):
        position = whitespace(position)
        key, position = decoder.raw_decode(text, position)
        if not isinstance(key, str) or key in header:
            raise ValueError("Native header contains an invalid or duplicate field")
        position = whitespace(position)
        if position >= len(text) or text[position] != ":":
            raise ValueError("Native header field lacks a value separator")
        position = whitespace(position + 1)
        if key == "events":
            if position >= len(text) or text[position] != "[":
                raise ValueError("Native final event table is not an array")
            return header
        value, position = decoder.raw_decode(text, position)
        header[key] = value
        position = whitespace(position)
        if position >= len(text) or text[position] != ",":
            raise ValueError("Native final header ended before its event table")
        position += 1
    raise ValueError("Native final metadata exceeds the bounded header reader")


def join_process_exit_diagnostic(ack: dict[str, Any], parent_shutdown: dict[str, Any]) -> dict[str, Any]:
    """Join real post-exit artifacts without asserting an SDK success or canonical close."""
    payload_path = Path(ack["process_exit_payload_file"])
    raw = json.loads(payload_path.read_text())
    pid = raw["process_id"]
    entries = [row for row in parent_shutdown["owned_worker_processes"] if row["pid"] == pid]
    if (pid != ack["process_id"] or len(entries) != 1 or entries[0]["alive"] is not False
            or entries[0]["exit_code"] != 0 or parent_shutdown.get("original_returned") is not True):
        raise RuntimeError("Diagnostic Native join requires the actual owned worker's successful exit")
    evidence: dict[str, Any] = {"diagnostic_only": True,
        "owned_worker_exit": deepcopy_json(entries[0]),
        "parent_natural_shutdown": deepcopy_json(parent_shutdown),
        "native_trace_validation": "not_asserted_header_observation_only"}
    native = Path(raw["native_final_file"])
    evidence["native_final_file_observed"] = native.is_file() and not native.is_symlink()
    error = None
    try:
        if native.is_symlink() or not native.is_file():
            raise RuntimeError("Actual automatic SDK final file is missing after owned zero exit")
        if "checkpoint" in native.name:
            raise RuntimeError("A Native checkpoint cannot substitute for the actual SDK final file")
        with native.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        evidence.update(native_final_bytes=native.stat().st_size, native_final_sha256=digest)
        header = read_native_header(native)
        if header.get("pid") != pid:
            raise RuntimeError("Actual automatic SDK file PID differs from the owned exited worker")
        fields = ("schema_version", "kind", "probe_protocol_version", "pid", "clock", "configured",
            "init_called", "ready", "init_ns", "start_status", "stop_status", "finalize_callback_called",
            "finalize_ns", "callbacks_inflight", "retained_events", "dropped_events", "stop_evidence",
            "explicit_client_finalize")
        evidence.update(native_final_file=str(native), native_final_file_observed=True,
            native_final_bytes=native.stat().st_size, native_final_sha256=digest,
            native_observed_header={key: header[key] for key in fields if key in header},
            header_reader_scope="top_level_metadata_before_events_no_synthesized_event_array")
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        evidence.update(native_final_file=str(native), error=error)
    raw.update(process_exit_diagnostic=evidence, capture_closed=False, complete_workspace=False,
        hardware_validated=False, diagnostic_only=True, native_sdk_stop_status=None,
        native_sdk_finalize_status=None,
        native_closure_acceptance="NOT_ASSERTED_PROCESS_EXIT_DIAGNOSTIC")
    write_new(Path(ack["session_file"]), raw)
    if error is not None:
        raise RuntimeError(error)
    return raw


def deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan_experiment(exp: Any, mode: str) -> None:
    expected = {"single": (1, 1), "tp2": (2, 1), "pp2": (1, 2)}[mode]
    s = exp.strategy
    if ((s.tensor_parallel_size, s.pipeline_parallel_size) != expected or s.replicas != 1
            or len(exp.infra.devices) != expected[0] * expected[1]
            or s.scheduler != "fifo" or s.max_batch_tokens != 512 or s.max_batch_size != 1
            or s.prefix_cache or s.chunked_prefill or s.mixed_batching
            or s.preemption != "recompute" or s.memory.kv_pool_bytes != 268435456
            or s.memory.kv_block_tokens != 16 or s.memory.graph_bytes or s.memory.cpu_swap_bytes
            or exp.execution.initial_state != "empty_cache" or exp.execution.warmup
            or any(getattr(s.model, field) != "bfloat16"
                   for field in ("weight_dtype", "kv_dtype", "activation_dtype"))):
        raise ValueError("Experiment does not match BF16 eager FIFO seq1/512-token/256MiB-KV Workspace capture")
    if mode == "pp2" and (s.pipeline_transport != "vllm_native"
                           or s.pipeline_transport_max_tokens < 512):
        raise ValueError("PP2 requires native transport and a bound covering the actual persistent512 buffers")


def validate_outputs(outputs: Any, lengths: list[int]) -> list[dict[str, Any]]:
    if len(outputs) != len(lengths):
        raise RuntimeError("Actual model output cohort is incomplete")
    rows = []
    for output, length in zip(outputs, lengths):
        if (not output.finished or len(output.prompt_token_ids) != length
                or len(output.outputs) != 1 or len(output.outputs[0].token_ids) != 4):
            raise RuntimeError("Actual request did not finish with four generated tokens")
        rows.append({"request_id": output.request_id, "input_tokens": length,
                     "output_token_ids": list(output.outputs[0].token_ids), "finished": True})
    return rows


def prepare(args: Any) -> dict[str, Any]:
    from mlsys_sim import load_experiment

    if sys.platform != "linux" or not hasattr(time, "CLOCK_MONOTONIC"):
        raise ValueError("Live Workspace capture requires Linux/POSIX; use CPU fixtures elsewhere")
    exp = load_experiment(args.experiment)
    validate_plan_experiment(exp, args.mode)
    if args.repeats not in (1, 2, 3) or not 30 <= args.timeout_s <= 600:
        raise ValueError("Use1..3 repetitions (formal target3), timeout30..600 seconds")
    budget = native_event_budget(args.native_max_events)
    grace = shutdown_grace(args.shutdown_grace_s)
    finalization = validate_native_finalization(args.mode, args.cohort, args.repeats, args.native_finalization)
    model = Path(args.model).resolve()
    if not model.is_dir():
        raise ValueError("Model must be an existing local directory; no network model resolution")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = {"schema_version": 1, "mode": args.mode, "cohort": args.cohort,
        "repeats": args.repeats, "model": str(model), "out": str(out),
        "timeout_s": args.timeout_s, "session_id": "workspace-" + args.mode + "-" + uuid.uuid4().hex[:16],
        "clock_domain": "linux-boot:" + Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "clock": "CLOCK_MONOTONIC", "native_clock": "CLOCK_MONOTONIC_RAW",
        "supervisor_pid": os.getpid(), "bound_experiment": str(out / "bound-experiment.json"),
        "experiment_input": str(args.experiment.resolve()), "experiment_input_sha256": file_sha256(args.experiment),
        "native_probe_library": str(args.native_probe.resolve()),
        "torch_probe_library": str(args.torch_probe.resolve()),
        "worker_source": str(Path(__file__).with_name("worker_workspace_capture.py").resolve()),
        "worker_source_sha256": file_sha256(Path(__file__).with_name("worker_workspace_capture.py")),
        "launcher_source_sha256": file_sha256(Path(__file__).resolve()),
        "visible_devices": args.visible_devices or ("1" if args.mode == "single" else "0,1"),
        "native_max_events": budget,
        "worker_shutdown_grace_seconds": grace,
        "native_finalization": finalization,
        "expected_formal_records_per_rank": 8 * args.repeats,
        "formal_only_output": False, "hardware_validated": False}
    for name in ("native_probe_library", "torch_probe_library"):
        path = Path(plan[name])
        if not path.is_file():
            raise ValueError("The actual compiled probe is missing: " + str(path))
        plan[name + "_sha256"] = file_sha256(path)
    write_new(out / "bound-experiment.json", asdict(exp))
    plan["bound_experiment_sha256"] = file_sha256(out / "bound-experiment.json")
    write_new(out / "session-plan.json", plan)
    return plan


def run_worker(plan: dict[str, Any]) -> int:
    """Child launcher: importing Torch is allowed; no parent CUDA API calls."""
    out = Path(plan["out"])
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.pop("MLSYS_VLLM_CAPTURE_CONFIG", None)
    os.environ["MLSYS_WORKSPACE_STAGE_PLAN"] = str(out / "session-plan.json")
    summary: dict[str, Any] = {"status": "FAILED", "session_id": plan["session_id"],
        "launcher_pid": os.getpid(), "parent_pid": os.getppid(), "ranks": [],
        "complete_workspace": False, "hardware_validated": False}
    llm = None
    shutdown_attempted = False
    try:
        from mlsys_sim import load_experiment
        from vllm import LLM, SamplingParams

        if file_sha256(Path(plan["bound_experiment"])) != plan["bound_experiment_sha256"]:
            raise RuntimeError("Bound actual experiment changed before child execution")
        if file_sha256(Path(plan["worker_source"])) != plan["worker_source_sha256"]:
            raise RuntimeError("Worker observation source changed before execution")
        exp = load_experiment(plan["bound_experiment"])
        validate_plan_experiment(exp, plan["mode"])
        finalization = validate_native_finalization(plan["mode"], plan["cohort"], plan["repeats"],
            plan.get("native_finalization", "worker_shutdown"))
        llm = LLM(plan["model"], tensor_parallel_size=exp.strategy.tensor_parallel_size,
            pipeline_parallel_size=exp.strategy.pipeline_parallel_size, dtype="bfloat16",
            distributed_executor_backend="uni" if plan["mode"] == "single" else "mp",
            worker_cls="worker_workspace_capture.WorkspaceWorker",
            max_model_len=512, max_num_batched_tokens=512, max_num_seqs=1,
            block_size=16, enable_prefix_caching=False, enable_chunked_prefill=False,
            async_scheduling=False, enforce_eager=True, kv_cache_memory_bytes=268435456,
            disable_custom_all_reduce=True, skip_tokenizer_init=True,
            generation_config="vllm", seed=0)
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, detokenize=False)
        warm = [16, 32]
        summary["warmup_arming"] = llm.collective_rpc("mlsys_workspace_arm", args=("warmup",), timeout=60)
        summary["warmup_outputs"] = validate_outputs(llm.generate(
            [{"prompt_token_ids": [100] * n} for n in warm], params, use_tqdm=False), warm)
        formal = [16, 32] * plan["repeats"]
        summary["formal_arming"] = llm.collective_rpc("mlsys_workspace_arm", args=("formal",), timeout=60)
        start = time.perf_counter_ns()
        answers = llm.generate([{ "prompt_token_ids": [100] * n} for n in formal], params, use_tqdm=False)
        end = time.perf_counter_ns()
        summary["formal_outputs"] = validate_outputs(answers, formal)
        summary["formal_parent_bounds"] = {"clock": "CLOCK_MONOTONIC", "start_ns": start, "end_ns": end,
            "warning": "Do not compare with SDK/Torch native CLOCK_MONOTONIC_RAW timestamps"}
        summary["ranks"] = llm.collective_rpc("mlsys_workspace_close", timeout=90)
        expected = exp.strategy.tensor_parallel_size * exp.strategy.pipeline_parallel_size
        if plan["mode"] != "single":
            if any(row.get("finalize_deferred") is not True or row.get("capture_closed") is not False
                   for row in summary["ranks"]):
                raise RuntimeError("MP close acknowledgement must defer SDK finalization to natural shutdown")
            core = llm.llm_engine.engine_core
            executor = core.engine_core.model_executor
            owned = [item.proc for item in executor.workers]
            summary["natural_parent_shutdown"] = {"started_monotonic_ns": time.perf_counter_ns(),
                "started_monotonic_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                "clock": "CLOCK_MONOTONIC", "native_clock": "CLOCK_MONOTONIC_RAW",
                "worker_shutdown_grace_seconds": plan["worker_shutdown_grace_seconds"],
                "policy": "original_engine_shutdown_and_its_original_worker_termination_policy"}
            shutdown_attempted = True
            try:
                core.shutdown()
            except BaseException as exc:
                summary["natural_parent_shutdown"].update(original_returned=False,
                    ended_monotonic_ns=time.perf_counter_ns(),
                    ended_monotonic_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                    exception={"type": type(exc).__name__, "message": str(exc)})
                raise
            else:
                summary["natural_parent_shutdown"].update(original_returned=True,
                    ended_monotonic_ns=time.perf_counter_ns(),
                    ended_monotonic_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                    owned_worker_processes=[{"pid": proc.pid, "exit_code": proc.exitcode,
                                             "alive": proc.is_alive()} for proc in owned])
                if any(proc.is_alive() or proc.exitcode != 0 for proc in owned):
                    raise RuntimeError("Owned MP workers did not exit successfully after their natural shutdown")
                llm = None
        coordinates = set()
        for row in summary["ranks"]:
            if finalization == "process_exit":
                data = join_process_exit_diagnostic(row, summary["natural_parent_shutdown"])
            else:
                data = json.loads(Path(row["session_file"]).read_text())
            placement = data["placement"]
            key = placement["stage_index"], placement["tp_rank"]
            if key in coordinates or data["session_id"] != plan["session_id"]:
                raise RuntimeError("Duplicate or conflicting actual rank/session")
            if finalization != "process_exit" and data.get("capture_closed") is not True:
                raise RuntimeError("Actual SDK window did not close successfully after the declared Runtime lifecycle")
            if row.get("finalize_deferred") and data.get("runtime_shutdown", {}).get("original_returned") is not True:
                raise RuntimeError("Deferred MP session lacks a successful actual WorkerProc.shutdown receipt")
            coordinates.add(key)
            selected = [record for record in data["records"] if record["cohort"] == "formal"]
            if len(selected) != plan["expected_formal_records_per_rank"]:
                raise RuntimeError("Actual formal Workspace row count differs from the declared cohort")
            if data["capture_errors"] or any(record["status"] != "RAW_STEP_CAPTURED" for record in selected):
                raise RuntimeError("Worker retained an unsupported or incomplete raw observation")
        if len(coordinates) != expected:
            raise RuntimeError("Raw acquisition is missing worker ranks")
        summary["status"] = "RAW_ACQUISITION_COMPLETED_AUDIT_REQUIRED"
        if finalization == "process_exit":
            summary.update(diagnostic_only=True, capture_closed=False,
                native_closure_acceptance="NOT_ASSERTED_PROCESS_EXIT_DIAGNOSTIC")
    except BaseException as exc:
        summary.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if llm is not None and not shutdown_attempted:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception as exc:
                summary.update(status="FAILED", shutdown_error=type(exc).__name__ + ": " + str(exc))
        write_new(out / "launcher-summary.json", summary)
    return 0 if summary["status"] == "RAW_ACQUISITION_COMPLETED_AUDIT_REQUIRED" else 1


def terminate_owned(process: subprocess.Popen[Any]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=.2)
            break
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def supervise(plan: dict[str, Any]) -> int:
    out = Path(plan["out"])
    env = os.environ.copy()
    for name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(name, None)
    env["HIP_VISIBLE_DEVICES"] = plan["visible_devices"]
    env["MLSYS_NATIVE_PROBE_OUT"] = str(out / "native-%p.json")
    env["MLSYS_NATIVE_PROBE_RETAIN_ALL"] = "1"
    env["MLSYS_NATIVE_PROBE_MAX_EVENTS"] = str(native_event_budget(plan["native_max_events"]))
    env["ROCP_TOOL_LIBRARIES"] = plan["native_probe_library"]
    if env.get("LD_PRELOAD") not in (None, "", plan["native_probe_library"]):
        raise RuntimeError("Conflicting preloaded libraries require an explicit separate capture domain")
    env["LD_PRELOAD"] = plan["native_probe_library"]
    env["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] = str(shutdown_grace(plan["worker_shutdown_grace_seconds"]))
    root = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result: dict[str, Any] = {"session_id": plan["session_id"], "status": "FAILED",
                            "hardware_validated": False, "complete_workspace": False}
    with (out / "launcher.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--_worker-plan",
            str(out / "session-plan.json")], env=env, stdout=log, stderr=subprocess.STDOUT,
            shell=False, start_new_session=True)
        start = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - start > plan["timeout_s"] or log.tell() > MAX_LOG_BYTES:
                result["error"] = "OWNED_SESSION_TIMEOUT_OR_LOG_LIMIT"
                terminate_owned(process)
                break
            time.sleep(.05)
        result["worker_exit_code"] = process.returncode
        terminate_owned(process)
    if process.returncode == 0:
        result["status"] = "RAW_ACQUISITION_COMPLETED_AUDIT_REQUIRED"
    write_new(out / "session-status.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if process.returncode == 0 else 1


def main(argv: list[str] | None = None) -> int:
    supplied = sys.argv[1:] if argv is None else argv
    if len(supplied) == 2 and supplied[0] == "--_worker-plan":
        return run_worker(json.loads(Path(supplied[1]).read_text()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "tp2", "pp2"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", choices=("train", "repeat", "pilot"), default="pilot")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="/app/models/Qwen3-1.7B")
    parser.add_argument("--native-probe", type=Path, default=Path("/app/work/native_probe.so"))
    parser.add_argument("--torch-probe", type=Path, default=Path("/app/work/torch_allocator_probe.so"))
    parser.add_argument("--visible-devices")
    parser.add_argument("--native-max-events", type=int, default=1048576,
                        help="Explicit SDK event budget1..4194304; compiled native probe must support it")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--shutdown-grace-s", type=int, default=60,
                        help="Bound original MP cleanup/file publication grace5..120 seconds")
    parser.add_argument("--native-finalization", choices=("worker_shutdown", "process_exit"),
                        default="worker_shutdown", help="process_exit is restricted to one PP2 pilot diagnostic")
    try:
        return supervise(prepare(parser.parse_args(supplied)))
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": type(exc).__name__ + ": " + str(exc),
                          "complete_workspace": False, "hardware_validated": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
