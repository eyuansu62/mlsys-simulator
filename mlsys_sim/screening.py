"""Constrained candidate comparison over one immutable workload contract.

Profiles are captured for the complete candidate set before execution. Every
engine reads read-only copies of those bytes, while source changes veto the
entire report. Configuration errors are never relabeled as physical OOM.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
from html import escape
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from .capacity import _bind_inputs, _profile_inputs, _semantic_config
from .config import _read, experiment_from_dict
from .constraints import (evaluate_dag_constraints, evaluate_simulation_constraints,
                          validate_constraints)
from .dag import (DAGPlan, _restore_origin_paths, _simulate_bound, _trace_identity,
                  dag_plan_dict, load_dag_plan)
from .engine import simulate
from .layout import replica_experiment
from .load_sweep import _run_bound_sweep, validate_load_plan
from .schema import Experiment, SimulationError
from .serialization import experiment_dict
from .state import SimulationResult
from .workload import canonical_sha256


MAX_CANDIDATES = 32
MAX_TOTAL_REQUESTS = 250_000
MAX_TOTAL_EVENTS = 5_000_000
MAX_PROFILE_BYTES = 64 * 1024 * 1024
_METRICS = {
    "simulation": {"elapsed_ms", "p99_e2e_ms", "throughput_rps", "goodput_rps",
                   "elapsed_output_tokens_per_s", "window_output_tokens_per_s"},
    "sweep": {"worst_p99_e2e_ms", "min_throughput_rps", "min_window_throughput_rps",
              "min_window_output_tokens_per_s"},
    "dag": {"makespan_ms", "elapsed_ms"},
}
_PHYSICAL_ERRORS = {"STATIC_MEMORY_INFEASIBLE"}


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    configuration: dict
    experiment: Experiment | None = None
    load_plan: dict | None = None
    dag_plan: DAGPlan | None = None


def _invalid(message: str) -> None:
    raise SimulationError("INVALID_SCREENING_PLAN", message)


def _object(value: Any, required: set[str], optional: set[str], label: str) -> dict:
    if (not isinstance(value, dict) or any(not isinstance(key, str) for key in value)
            or required - value.keys() or value.keys() - required - optional):
        _invalid(f"{label} requires {sorted(required)}; optional keys: {sorted(optional)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\x00" in value:
        _invalid(f"{label} must be a bounded nonempty string")
    return value


def _resolve_experiment(value: dict, root: Path | None) -> Experiment:
    exp = experiment_from_dict(value)
    if root is None:
        return exp
    changes: dict[str, Any] = {}
    for field in ("profile_path", "native_pp_communication_profile_path", "workspace_profile_path"):
        path = getattr(exp.calibration, field)
        if path is not None:
            changes[field] = str((root / path).resolve())
    for field in ("replica_profile_paths", "replica_native_pp_communication_profile_paths",
                  "replica_workspace_profile_paths"):
        changes[field] = tuple(str((root / path).resolve()) for path in getattr(exp.calibration, field))
    return replace(exp, calibration=replace(exp.calibration, **changes))


def _normalize(value: dict | str | Path, base_dir: Path | None) -> tuple[dict, list[_Candidate]]:
    source = Path(value).resolve() if isinstance(value, (str, Path)) else None
    root = Path(base_dir).resolve() if base_dir is not None else source.parent if source is not None else None
    raw = _read(source) if source is not None else deepcopy(value)
    plan = _object(raw, {"schema_version", "kind", "source", "constraints", "candidates"},
                   {"description", "objective", "pool_constraints"}, "screening plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        _invalid("schema_version must be integer 1")
    kind = plan["kind"]
    if not isinstance(kind, str) or kind not in _METRICS:
        _invalid("kind must be simulation, sweep or dag; mixed kinds cannot be ranked")
    _text(plan["source"], "source")
    if "description" in plan and not isinstance(plan["description"], str):
        _invalid("description must be text")
    constraints = validate_constraints(plan["constraints"])
    pool_specs = plan.get("pool_constraints", {})
    if not isinstance(pool_specs, dict) or any(not isinstance(name, str) for name in pool_specs):
        _invalid("pool_constraints must map model IDs to constraint specifications")
    if "pool_constraints" in plan and kind != "dag":
        _invalid("pool_constraints is available only for DAG candidates")
    pool_specs = {name: validate_constraints(spec) for name, spec in pool_specs.items()}
    objective = plan.get("objective")
    if objective is not None:
        _object(objective, {"metric", "direction"}, set(), "objective")
        if (not isinstance(objective["metric"], str) or objective["metric"] not in _METRICS[kind]
                or objective["direction"] not in ("min", "max")):
            _invalid(f"objective requires a supported {kind} metric and direction min/max")
    supplied = plan["candidates"]
    if not isinstance(supplied, list) or not 1 <= len(supplied) <= MAX_CANDIDATES:
        _invalid(f"candidates must contain 1..{MAX_CANDIDATES} complete configurations")
    candidates = []
    ids: set[str] = set()
    identities = []
    total_requests = total_events = 0
    for candidate in supplied:
        required = {"candidate_id", "dag_plan"} if kind == "dag" else {"candidate_id", "experiment"}
        if kind == "sweep":
            required.add("load_plan")
        _object(candidate, required, set(), "candidate")
        name = _text(candidate["candidate_id"], "candidate_id")
        if name in ids:
            _invalid("candidate IDs must be unique")
        ids.add(name)
        if kind == "dag":
            if not isinstance(candidate["dag_plan"], dict):
                _invalid("dag_plan must be one inline complete DAG object")
            dag = load_dag_plan(candidate["dag_plan"], base_dir=root)
            if dag.candidates:
                _invalid("Each screening dag_plan must contain one complete pools placement")
            if pool_specs.keys() - {pool.model_id for pool in dag.pools}:
                _invalid("pool_constraints must name pools present in every candidate")
            identities.append(canonical_sha256({"trace_sha256": _trace_identity(dag),
                                                "execution": asdict(dag.execution)}))
            total_requests += len(dag.nodes) + sum(len(pool.experiment.execution.warmup) for pool in dag.pools)
            total_events += dag.execution.max_events + sum(pool.experiment.execution.max_events
                for pool in dag.pools if pool.experiment.execution.warmup)
            configuration = {"candidate_id": name, "dag_plan": dag_plan_dict(dag)}
            candidates.append(_Candidate(name, configuration, dag_plan=dag))
        else:
            exp = _resolve_experiment(candidate["experiment"], root)
            identity = {"workload": [asdict(spec) for spec in exp.workload],
                        "seed": exp.execution.seed, "evaluation": asdict(exp.evaluation)}
            load = None
            runs = 1
            requests = len(exp.workload)
            if kind == "sweep":
                if exp.workload:
                    _invalid("sweep experiment.workload must be empty; load_plan is the frozen workload")
                load = deepcopy(validate_load_plan(candidate["load_plan"]))
                if (load.get("service_time", {}).get("kind", "deterministic") != "deterministic"
                        and exp.calibration.backend != "synthetic"):
                    _invalid("Empirical service scaling requires synthetic candidate backends")
                identity["load_plan"] = load
                grid = load["rate_grid_rps"] if "rate_grid_rps" in load else load["concurrency_grid"]
                runs = len(grid) * load["replications"]
                requests = load["workload"]["arrival"]["max_requests"]
            identities.append(canonical_sha256(identity))
            total_requests += runs * (requests + len(exp.execution.warmup))
            total_events += runs * exp.execution.max_events * (2 if exp.execution.warmup else 1)
            configuration = {"candidate_id": name, "experiment": experiment_dict(exp)}
            if load is not None:
                configuration["load_plan"] = load
            candidates.append(_Candidate(name, configuration, experiment=exp, load_plan=load))
    if len(set(identities)) != 1:
        _invalid("All candidates must preserve the complete request trace, seed and measurement/load contract")
    if total_requests > MAX_TOTAL_REQUESTS or total_events > MAX_TOTAL_EVENTS:
        _invalid(f"Declared total exceeds {MAX_TOTAL_REQUESTS} requests or {MAX_TOTAL_EVENTS} events; "
                 "reduce per-run bounds, scan size or candidate count")
    normalized = {"schema_version": 1, "kind": kind, "source": plan["source"],
                  "description": plan.get("description", ""), "constraints": constraints,
                  "objective": deepcopy(objective), "candidates": [candidate.configuration for candidate in candidates]}
    if pool_specs:
        normalized["pool_constraints"] = pool_specs
    normalized["workload_sha256"] = identities[0]
    normalized["declared_budget"] = {"requests": total_requests, "events": total_events,
                                      "max_requests": MAX_TOTAL_REQUESTS, "max_events": MAX_TOTAL_EVENTS}
    return normalized, candidates


def _experiments(candidate: _Candidate) -> list[tuple[str | None, str | None, Experiment]]:
    if candidate.dag_plan is not None:
        return [(pool.model_id, pool.role, pool.experiment) for pool in candidate.dag_plan.pools]
    assert candidate.experiment is not None
    return [(None, None, candidate.experiment)]


def _check_sources(candidates: list[_Candidate], captured: dict[str, list[dict]]) -> None:
    for candidate in candidates:
        current: list[dict] = []
        try:
            for model_id, role, exp in _experiments(candidate):
                current.extend({**item, **({"model_id": model_id, "role": role} if model_id is not None else {})}
                               for item in _profile_inputs(exp))
        except (SimulationError, UnicodeError) as exc:
            raise SimulationError("INPUT_CHANGED", "A captured screening Profile became unavailable") from exc
        if current != captured[candidate.candidate_id]:
            raise SimulationError("INPUT_CHANGED", "A Profile changed during candidate screening")


def _preflight_profiles(exp: Experiment) -> None:
    """Check every selected Profile before physical infeasibility can hide it."""
    from .profiles import create_performance_model
    from .pp_communication import PPCommunicationModel
    from .workspace import WorkspaceModel

    for index in range(exp.strategy.replicas):
        local = replica_experiment(exp, index)
        if local.calibration.backend == "profile":
            create_performance_model(local)
        if local.calibration.native_pp_communication_profile_path:
            PPCommunicationModel(local)
        if local.calibration.workspace_profile_path:
            WorkspaceModel(local)


def _bind(candidate: _Candidate, inputs: list[dict], directory: Path,
          paths: dict[str, str]) -> _Candidate:
    bound = []
    for index, (model_id, _, exp) in enumerate(_experiments(candidate)):
        selected = [item for item in inputs if item.get("model_id") == model_id]
        runtime = _bind_inputs(exp, selected, directory / f"pool-{index:03d}")
        for item in selected:
            value = getattr(runtime.calibration, item["field"])
            path = value[item["index"]] if item["field"].endswith("paths") else value
            Path(path).chmod(0o400)
            paths[path] = item["origin_path"]
        bound.append(runtime)
    if candidate.dag_plan is not None:
        pools = tuple(replace(pool, experiment=exp) for pool, exp in zip(candidate.dag_plan.pools, bound))
        return replace(candidate, dag_plan=replace(candidate.dag_plan, pools=pools))
    return replace(candidate, experiment=bound[0])


def _merge(assessments: list[dict]) -> dict:
    checks = [check for assessment in assessments for check in assessment["checks"]]
    verdicts = [assessment["verdict"] for assessment in assessments]
    verdict = "FAIL" if "FAIL" in verdicts else "UNKNOWN" if not verdicts or "UNKNOWN" in verdicts else "PASS"
    limiting = sorted((deepcopy(check) for check in checks
                       if check["category"] in {"memory", "latency", "throughput", "task"}
                       and check.get("normalized_headroom") is not None),
                      key=lambda check: (check["normalized_headroom"], check["id"], str(check["scope"])))
    return {"schema_version": 1, "verdict": verdict, "checks": checks, "limiting_constraints": limiting,
            "metric_contract": {"aggregation": "ALL declared points and ALL replications",
                                "individual_contracts": [item["metric_contract"] for item in assessments]}}


def _failure(code: str, message: str, *, scope: dict | None = None) -> dict:
    check = {"id": "execution.startup", "category": "execution", "scope": scope or {},
             "actual": code, "operator": "==", "limit": "COMPLETED", "unit": "status",
             "sample_count": 0, "verdict": "FAIL", "reason": message,
             "ratio": None, "normalized_headroom": None}
    return {"schema_version": 1, "verdict": "FAIL", "checks": [check], "limiting_constraints": [],
            "metrics": {}, "metric_contract": {"observation": "not_started_static_infeasible",
                                                   "no_invented_memory_or_latency_observations": True}}


def _sweep_assessment(report: dict, spec: dict) -> tuple[dict, list[dict]]:
    assessments = []
    runs = []
    for point in report["points"]:
        for run in point["runs"]:
            assessment = evaluate_simulation_constraints(SimulationResult(run["result"], run["requests"], run["events"]), spec)
            scope = {"point_index": point["point_index"], "replication": run["replication"]}
            for check in assessment["checks"]:
                check["scope"].update(scope)
            label = run["summary"]["classification"]
            slo = run["summary"]["slo_pass"]
            for name, actual, passing in (("classification", label, label in {"observed_stable", "observed_completed"}),
                                           ("slo", slo, slo is True)):
                unknown = label == "insufficient_samples" or actual is None
                verdict = "UNKNOWN" if unknown else "PASS" if passing else "FAIL"
                assessment["checks"].append({"id": f"load.{name}", "category": "load", "scope": scope,
                    "actual": actual, "operator": "==", "limit": True if name == "slo" else "observed_stable_or_completed",
                    "unit": "boolean" if name == "slo" else "status", "sample_count": assessment["metrics"]["completed_samples"],
                    "verdict": verdict, "reason": "declared_load_plan_gate", "ratio": None, "normalized_headroom": None})
            values = [check["verdict"] for check in assessment["checks"]]
            assessment["verdict"] = "FAIL" if "FAIL" in values else "UNKNOWN" if "UNKNOWN" in values else "PASS"
            assessments.append(assessment)
            runs.append({**scope, "assessment": assessment})
    combined = _merge(assessments)
    return combined, runs


def _objective(kind: str, objective: dict | None, assessment: dict, execution: dict,
               runs: list[dict]) -> dict:
    if objective is None:
        return {"metric": None, "direction": None, "value": None, "unit": None,
                "verdict": "UNKNOWN", "reason": "objective_not_declared"}
    metric = objective["metric"]
    metrics = assessment.get("metrics", {})
    value = None
    unit = "ms" if metric.endswith("_ms") else "output_tokens/s" if "tokens_per_s" in metric else "requests/s"
    if kind == "sweep":
        selected = {"worst_p99_e2e_ms": "end_to_end_p99_ms", "min_throughput_rps": "throughput_requests_per_s",
                    "min_window_throughput_rps": "window_requests_per_s",
                    "min_window_output_tokens_per_s": "window_output_tokens_per_s"}[metric]
        values = [run["assessment"]["metrics"].get(selected) for run in runs]
        if values and all(isinstance(number, (int, float)) and not isinstance(number, bool)
                          and math.isfinite(number) for number in values):
            value = max(values) if metric == "worst_p99_e2e_ms" else min(values)
    elif metric == "goodput_rps":
        value = execution.get("result", {}).get("goodput", {}).get("requests_per_s")
    else:
        selected = {"p99_e2e_ms": "end_to_end_p99_ms", "throughput_rps": "throughput_requests_per_s"}.get(metric, metric)
        value = metrics.get(selected)
    known = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return {"metric": metric, "direction": objective["direction"], "value": value if known else None,
            "unit": unit, "verdict": "KNOWN" if known else "UNKNOWN",
            "reason": "observed_metric" if known else "metric_or_denominator_unavailable",
            "aggregation": "maximum run P99" if metric == "worst_p99_e2e_ms" else
                           "minimum over every declared point/replication" if kind == "sweep" else "one complete run"}


def run_candidate_screening(value: dict | str | Path, *, base_dir: Path | None = None) -> dict:
    """Evaluate every candidate fairly; rank only jointly passing observations."""
    plan, candidates = _normalize(value, base_dir)
    captured: dict[str, list[dict]] = {}
    consumed = 0
    for candidate in candidates:
        inputs: list[dict] = []
        for model_id, role, exp in _experiments(candidate):
            rows = _profile_inputs(exp, max_bytes=min(16 * 1024 * 1024, max(1, MAX_PROFILE_BYTES - consumed)))
            consumed += sum(len(row["raw_json"].encode("utf-8")) for row in rows)
            if consumed > MAX_PROFILE_BYTES:
                _invalid("Total captured Profile bytes exceed the 64 MiB screening bound")
            inputs.extend({**row, **({"model_id": model_id, "role": role} if model_id is not None else {})} for row in rows)
        captured[candidate.candidate_id] = inputs
    results = []
    with tempfile.TemporaryDirectory(prefix="mlsys-screen-inputs-") as temporary:
        paths: dict[str, str] = {}
        runtime = [_bind(candidate, captured[candidate.candidate_id], Path(temporary) / f"candidate-{index:03d}", paths)
                   for index, candidate in enumerate(candidates)]
        for candidate in runtime:
            for _, _, exp in _experiments(candidate):
                _preflight_profiles(exp)
        _check_sources(candidates, captured)
        for original, bound in zip(candidates, runtime):
            inputs = captured[original.candidate_id]
            runs: list[dict] = []
            try:
                if plan["kind"] == "dag":
                    assert original.dag_plan is not None and bound.dag_plan is not None
                    profile_inputs = {pool.model_id: [item for item in inputs if item["model_id"] == pool.model_id]
                                      for pool in original.dag_plan.pools}
                    # The original DAG helper expects role-free Profile evidence.
                    profile_inputs = {name: [{key: val for key, val in row.items() if key not in {"model_id", "role"}}
                                              for row in rows] for name, rows in profile_inputs.items()}
                    dag = _simulate_bound(original.dag_plan, profile_inputs, bound.dag_plan.pools)
                    execution = {"kind": "dag", "report": dag}
                    assessment = evaluate_dag_constraints(dag, plan["constraints"], pool_specs=plan.get("pool_constraints"))
                elif plan["kind"] == "sweep":
                    assert original.experiment is not None and bound.experiment is not None and original.load_plan is not None
                    sweep = _run_bound_sweep(original.experiment, original.load_plan, inputs, bound.experiment)
                    execution = {"kind": "sweep", "report": sweep}
                    assessment, runs = _sweep_assessment(sweep, plan["constraints"])
                else:
                    assert bound.experiment is not None
                    result = simulate(bound.experiment)
                    execution = {"kind": "simulation", "result": result.result,
                                 "requests": result.requests, "events": result.events}
                    assessment = evaluate_simulation_constraints(result, plan["constraints"])
            except SimulationError as exc:
                _check_sources(candidates, captured)
                if exc.code not in _PHYSICAL_ERRORS:
                    raise
                assessment = _failure(exc.code, str(exc))
                execution = {"kind": plan["kind"], "status": "NOT_STARTED_STATIC_INFEASIBLE",
                             "error_code": exc.code, "error_message": str(exc)}
                if original.load_plan is not None:
                    grid = (original.load_plan["rate_grid_rps"] if "rate_grid_rps" in original.load_plan
                            else original.load_plan["concurrency_grid"])
                    runs = [{"point_index": point, "replication": rep,
                             "assessment": _failure(exc.code, str(exc), scope={"point_index": point, "replication": rep})}
                            for point in range(len(grid)) for rep in range(original.load_plan["replications"])]
                    assessment = _merge([run["assessment"] for run in runs])
                    execution["planned_runs"] = [{"point_index": run["point_index"], "replication": run["replication"],
                                                 "status": "NOT_STARTED_STATIC_INFEASIBLE"} for run in runs]
            _check_sources(candidates, captured)
            objective = _objective(plan["kind"], plan["objective"], assessment, execution, runs)
            verdict = ("FAIL" if assessment["verdict"] == "FAIL" else "UNKNOWN"
                       if assessment["verdict"] == "UNKNOWN" or objective["verdict"] == "UNKNOWN" else "PASS")
            semantic = []
            for model_id, role, exp in _experiments(original):
                selected = [item for item in inputs if item.get("model_id") == model_id]
                semantic.append({"model_id": model_id, "role": role,
                                 "experiment": json.loads(_semantic_config(exp, selected))})
            results.append({"candidate_id": original.candidate_id, "verdict": verdict,
                            "configuration": original.configuration,
                            "configuration_sha256": canonical_sha256({"experiments": semantic,
                                "load_plan": original.load_plan,
                                "dag_execution": asdict(original.dag_plan.execution) if original.dag_plan is not None else None}),
                            "profile_inputs": inputs, "constraint_assessment": assessment,
                            "objective": objective, "run_assessments": runs, "execution": execution})
        results = _restore_origin_paths(results, paths)
    accepted = [candidate for candidate in results if candidate["verdict"] == "PASS"]
    if plan["objective"] is not None:
        accepted.sort(key=lambda candidate: candidate["objective"]["value"], reverse=plan["objective"]["direction"] == "max")
    ranking = []
    previous = None
    rank = 0
    for index, candidate in enumerate(accepted):
        score = candidate["objective"]["value"]
        if index == 0 or score != previous:
            rank = index + 1
        ranking.append({"rank": rank, "candidate_id": candidate["candidate_id"], "value": score,
                        "unit": candidate["objective"]["unit"],
                        "tie_count": sum(item["objective"]["value"] == score for item in accepted)})
        previous = score
    all_inputs = [{"candidate_id": candidate["candidate_id"], **item}
                  for candidate in results for item in candidate["profile_inputs"]]
    return {"schema_version": 1, "kind": "candidate_screening", "screen_kind": plan["kind"],
            "backend": "simulated", "source_kind": "simulated", "hardware_validated": False, "source": plan["source"],
            "description": plan["description"], "workload_sha256": plan["workload_sha256"],
            "plan": plan, "configuration_sha256": canonical_sha256({
                "kind": plan["kind"], "source": plan["source"], "constraints": plan["constraints"],
                "pool_constraints": plan.get("pool_constraints"), "objective": plan["objective"],
                "workload_sha256": plan["workload_sha256"],
                "candidates": [{"candidate_id": item["candidate_id"], "sha256": item["configuration_sha256"]}
                               for item in results]}),
            "profile_inputs": all_inputs, "candidates": results, "ranking": ranking,
            "counts": {status.lower(): sum(item["verdict"] == status for item in results)
                       for status in ("PASS", "FAIL", "UNKNOWN")},
            "ranking_rule": "only PASS constraints and known objective; exact ties retained with competition ranks",
            "limitations": ["CPU model prediction; constraint PASS is not hardware validation or an optimality proof",
                            "all candidates preserve one frozen request/load contract; only declared configurations differ",
                            "sweep objectives use every declared point and replication, never a favorable subset",
                            "static infeasibility prevents startup; no missing memory/latency observations are invented"]}


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rendered = {key: _display(row.get(key)) for key in fields}
            writer.writerow({key: "'" + value if value.startswith(("=", "+", "-", "@")) else value
                             for key, value in rendered.items()})


def _html(report: dict) -> str:
    rows = []
    ranks = {item["candidate_id"]: item["rank"] for item in report["ranking"]}
    sections = []
    for index, candidate in enumerate(report["candidates"]):
        name = escape(candidate["candidate_id"])
        objective = candidate["objective"]
        rows.append("<tr>" + "".join(f"<td>{escape(_display(value))}</td>" for value in
            [candidate["candidate_id"], candidate["verdict"], ranks.get(candidate["candidate_id"]),
             objective["metric"], objective["value"], objective["unit"]]) + "</tr>")
        checks = candidate["constraint_assessment"]["checks"]
        check_rows = ["<tr>" + "".join(f"<td>{escape(_display(value))}</td>" for value in
            [item["id"], item["scope"], item["actual"], item["operator"], item["limit"], item["unit"],
             item["sample_count"], item["verdict"], item["normalized_headroom"], item["reason"]]) + "</tr>"
                      for item in checks]
        tight = candidate["constraint_assessment"]["limiting_constraints"][:3]
        tight_text = "; ".join(f"{item['id']} @ {_display(item['scope'])}: {_display(item['normalized_headroom'])}"
                              for item in tight)
        sections.append(f"<section><h2>{name}</h2><p>约束：{escape(candidate['constraint_assessment']['verdict'])}；"
                        f"选择：{escape(candidate['verdict'])}；目标：{escape(objective['reason'])}。</p>"
                        f"<p>最紧约束：{escape(tight_text) if tight_text else '无可计算的数值 headroom'}</p>"
                        f"<p><a href='candidate-{index:03d}.json'>完整候选 JSON</a></p>"
                        "<div class='scroll'><table class='checks'><thead><tr><th>约束</th><th>范围</th><th>actual</th><th>操作</th>"
                        "<th>limit</th><th>单位</th><th>样本数</th><th>判定</th><th>归一化 headroom</th><th>原因</th>"
                        "</tr></thead><tbody>" + "".join(check_rows) + "</tbody></table></div></section>")
    return ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'><title>候选配置联合约束筛选</title>"
            "<style>body{background:#f3f5f8;color:#18293d;font:15px/1.6 system-ui;margin:0}"
            "main{max-width:1250px;margin:32px auto;padding:28px;background:white}"
            "aside{padding:14px;background:#fff4db;border-left:4px solid #b97816}"
            "table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px;text-align:left;"
            "border-bottom:1px solid #dde4ed;overflow-wrap:anywhere}th{background:#edf2f7}.scroll{overflow-x:auto}"
            "table.checks{min-width:1450px}table.checks th{white-space:nowrap}"
            "table.checks td:nth-child(n+3):nth-child(-n+9){white-space:nowrap;overflow-wrap:normal}"
            "table.checks td:first-child{min-width:150px}"
            "table.checks td:nth-child(2){min-width:210px;max-width:320px}"
            "table.checks td:last-child{min-width:190px;max-width:340px}"
            "a{color:#20599d}h2{margin-top:36px}p{overflow-wrap:anywhere}</style></head><body><main>"
            "<h1>候选配置联合约束筛选</h1><aside>所有数值为 CPU 模拟预测。PASS 只表示所声明约束在当前模型观测中成立，"
            "不是硬件实测、SLA 认证或全局最优证明。</aside>"
            f"<p>来源：{escape(report['source'])} · 类型：{escape(report['screen_kind'])}</p>"
            f"<p>{escape(report['description'])}</p><p>同负载 SHA256：{escape(report['workload_sha256'])}</p>"
            "<p>只对全部约束 PASS 且目标可观测的候选排名；FAIL 与 UNKNOWN 保留，不进入排名。"
            "相同目标值保留并列名次，缺失目标或分母不会被记成零。</p>"
            "<p><a href='report.json'>完整 JSON</a> · <a href='candidates.csv'>候选 CSV</a> · "
            "<a href='checks.csv'>约束 CSV</a> · <a href='replay.plan.json'>离线重放计划</a></p>"
            "<div class='scroll'><table><thead><tr><th>候选</th><th>选择判定</th><th>名次</th><th>目标</th>"
            "<th>actual</th><th>单位</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            + "".join(sections) + "<h2>解释边界</h2><ul>"
            + "".join(f"<li>{escape(item)}</li>" for item in report["limitations"]) + "</ul></main></body></html>\n")


def write_screening_report(report: dict, out: str | Path) -> Path:
    """Save one report and the exact Profile bytes needed for offline replay."""
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=False)
    saved = {**report, "stored_at": datetime.now(timezone.utc).isoformat(),
             "prediction_id": "screen-" + canonical_sha256(report)[:24]}
    (directory / "report.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    replay = deepcopy(report["plan"])
    replay.pop("workload_sha256", None)
    replay.pop("declared_budget", None)
    manifest = []
    summaries = []
    checks: list[dict] = []
    for index, candidate in enumerate(report["candidates"]):
        (directory / f"candidate-{index:03d}.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2,
                                                                        allow_nan=False) + "\n", encoding="utf-8")
        summaries.append({"candidate_id": candidate["candidate_id"], "verdict": candidate["verdict"],
                          "constraint_verdict": candidate["constraint_assessment"]["verdict"],
                          "objective": candidate["objective"]["metric"], "actual": candidate["objective"]["value"],
                          "unit": candidate["objective"]["unit"],
                          "rank": next((item["rank"] for item in report["ranking"]
                                        if item["candidate_id"] == candidate["candidate_id"]), None)})
        checks.extend({"candidate_id": candidate["candidate_id"], **check}
                      for check in candidate["constraint_assessment"]["checks"])
        for number, item in enumerate(candidate["profile_inputs"]):
            relative = Path("inputs") / f"candidate-{index:03d}" / f"profile-{number:03d}.json"
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["raw_json"].encode("utf-8"))
            manifest.append({"candidate_id": candidate["candidate_id"], "snapshot_path": relative.as_posix(),
                             **{key: value for key, value in item.items() if key != "raw_json"}})
            configuration = replay["candidates"][index]
            if "model_id" in item:
                pool = next(pool for pool in configuration["dag_plan"]["pools"] if pool["model_id"] == item["model_id"])
                calibration = pool["experiment"]["calibration"]
            else:
                calibration = configuration["experiment"]["calibration"]
            if item["field"].endswith("paths"):
                selected = list(calibration[item["field"]])
                selected[item["index"]] = relative.as_posix()
                calibration[item["field"]] = selected
            else:
                calibration[item["field"]] = relative.as_posix()
    if manifest:
        (directory / "inputs/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "replay.plan.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _csv(directory / "candidates.csv", summaries, ["candidate_id", "verdict", "constraint_verdict", "objective", "actual", "unit", "rank"])
    _csv(directory / "checks.csv", checks, ["candidate_id", "id", "category", "scope", "actual", "operator", "limit",
                                           "unit", "sample_count", "verdict", "normalized_headroom", "reason"])
    (directory / "report.html").write_text(_html(report), encoding="utf-8")
    return directory
