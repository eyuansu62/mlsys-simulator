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
    pg_prefix = "torch.distributed.distributed_c10d."
    pg_functions = {pg_prefix + name for name in ("all_gather_into_tensor", "isend", "irecv")}
    if qualified not in {constructor, shutdown} and qualified not in pg_functions:
        raise ValueError("Unknown communication constructor entrypoint")
    text = path.read_text()
    tree = ast.parse(text)
    if qualified in {constructor, shutdown}:
        class_name, method_name = qualified.split(".")[-2:]
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        methods = [node for cls in classes for node in cls.body
                   if isinstance(node, ast.FunctionDef) and node.name == method_name]
    else:
        methods = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == qualified.removeprefix(pg_prefix)]
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
