"""Create a new current-code CPU local draft using the historical 27-job recipe.

This never regenerates the immutable November 16 commitment or its raw SHAs.
Full payloads stay in the explicit new --out/predictions directory. Hardware
fields and the remote freeze remain pending; no GPU is contacted.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mlsys_sim import load_experiment  # noqa: E402
from mlsys_sim.load_validation import predict_closed_loop, validate_closed_loop_spec  # noqa: E402
from mlsys_sim.workload import canonical_sha256  # noqa: E402

GRID = [1, 2, 4]
SEED = 20261116
SRC = REPO / "examples/course/9070-sla/frozen-commitment"
HISTORICAL = REPO / "examples/course/9070-sla-20261116/frozen-commitment"
COHORTS = {"baseline": [256], "short": [64, 128], "long": [512, 1024]}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_path(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def make_spec(cohort: str, lengths: list[int], *, study_id: str, requests_per_run: int = 1000,
              replications: int = 3) -> dict[str, Any]:
    return validate_closed_loop_spec({"schema_version": 1, "study_id": study_id,
        "concurrency_grid": list(GRID), "requests_per_run": requests_per_run,
        "replications": replications, "seed": SEED, "think_ms": 0, "output_tokens": 16,
        "timeout_ms": 10000, "max_run_ms": 600000, "prompt_token_id": 100,
        "cohort": cohort, "input_lengths": list(lengths)})


def raw_payload_bytes(prediction: dict) -> bytes:
    return (json.dumps(prediction, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sla_verdict(run: dict, expected_requests: int) -> bool:
    result = run.get("result", {})
    counts = result.get("counts", {})
    if (type(expected_requests) is not int or expected_requests < 1 or run.get("status") != "COMPLETED"
            or result.get("status") != "COMPLETED" or run.get("resources_drained") is not True
            or run.get("error") is not None or result.get("error_code") is not None):
        return False
    expected_run = {"expected_requests": expected_requests, "offered_count": expected_requests, "not_offered_count": 0}
    expected_counts = {"total": expected_requests, "completed": expected_requests,
                       "failed": 0, "timed_out": 0, "unfinished": 0, "not_admitted": 0}
    if (any(type(run.get(key)) is not int or run[key] != value for key, value in expected_run.items())
            or any(type(counts.get(key)) is not int or counts[key] != value for key, value in expected_counts.items())):
        return False
    requests = run.get("requests")
    if (not isinstance(requests, list) or len(requests) != expected_requests
            or any(not isinstance(row, dict) or row.get("status") != "COMPLETED" for row in requests)):
        return False
    latency, throughput = result["latency_ms"], result["throughput"]
    metrics = (latency["end_to_end"]["p99"], latency["ttft"]["p99"], latency["tpot"]["p99"],
               throughput["output_tokens_per_s"])
    return all(value is not None for value in metrics) and (
        metrics[0] <= 500 and metrics[1] <= 500 and metrics[2] <= 50 and metrics[3] >= 20)


def _new_output(out: Path) -> Path:
    expanded = out.expanduser()
    if expanded.is_symlink():
        raise ValueError("--out must be a new directory, not a symlink alias")
    destination = expanded.resolve()
    protected_roots = (SRC.resolve(strict=True), HISTORICAL.resolve(strict=True))
    for protected in protected_roots:
        if destination == protected or destination.is_relative_to(protected):
            raise ValueError("--out cannot be a protected historical snapshot or any child/symlink alias")
    # resolve() can preserve a differently cased spelling on case-insensitive
    # filesystems. Check identities of existing ancestors before any mkdir.
    for ancestor in (destination, *destination.parents):
        try:
            ancestor.stat()
        except FileNotFoundError:
            continue
        # Other stat/samefile errors fail closed rather than granting permission.
        if any(ancestor.samefile(protected) for protected in protected_roots):
            raise ValueError("--out has a protected historical directory identity in its ancestry")
    if destination.exists():
        raise ValueError("--out already exists; a new exclusive output directory is required")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _code_files_sha256() -> dict[str, str]:
    return {str(path.relative_to(REPO)): sha_path(path) for path in sorted((REPO / "mlsys_sim").rglob("*.py"))}


def freeze_local(out: Path, *, study_id: str, requests_per_run: int = 1000,
                 replications: int = 3) -> dict[str, Any]:
    """Write a new draft, preserving the source experiment and all old bundles."""
    if not isinstance(study_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", study_id):
        raise ValueError("--study-id must be a bounded explicit identifier")
    source_specs = [json.loads(path.read_bytes()) for directory in (SRC, HISTORICAL)
                    for path in directory.glob("*.spec.json")]
    if study_id in {spec["study_id"] for spec in source_specs}:
        raise ValueError("--study-id cannot reuse a historical study identifier")
    specs = {name: make_spec(name, lengths, study_id=study_id, requests_per_run=requests_per_run,
                           replications=replications) for name, lengths in COHORTS.items()}
    experiment_path = SRC / "experiment.json"
    experiment_raw = experiment_path.read_bytes()
    experiment = load_experiment(experiment_path)
    template = json.loads((SRC / "validation-contract.template.json").read_bytes())
    template["min_completed_samples"] = requests_per_run
    template["constraints"]["min_completed_samples"] = requests_per_run
    code_files = _code_files_sha256()
    destination = _new_output(Path(out))
    (destination / "predictions").mkdir(exist_ok=False)
    _write_new(destination / "experiment.json", experiment_raw)
    _write_new(destination / "validation-contract.template.json", _json_bytes(template))
    predictions, raw_sha, canonical_sha, artifact_paths = {}, {}, {}, {}
    for name, spec in specs.items():
        _write_new(destination / f"{name}.spec.json", _json_bytes(spec))
        prediction = predict_closed_loop(experiment, spec)
        raw = raw_payload_bytes(prediction)
        relative = f"predictions/{name}.prediction-payload.json"
        _write_new(destination / relative, raw)
        predictions[name], raw_sha[name], canonical_sha[name] = prediction, sha_bytes(raw), canonical_sha256(prediction)
        artifact_paths[name] = relative
    summary = {name: {"spec_sha256": prediction["spec_sha256"], "identity_sha256": prediction["identity_sha256"],
        "runs": [{"concurrency": run["concurrency"], "replication": run["replication"],
            "policy_sha256": run["policy_sha256"], "request_stream_sha256": run["request_stream_sha256"],
            "status": run["status"], "latency_ms": run["result"]["latency_ms"], "elapsed_ms": run["elapsed_ms"],
            "throughput": run["result"]["throughput"], "counts": run["result"]["counts"]}
            for run in prediction["runs"]]} for name, prediction in predictions.items()}
    _write_new(destination / "predictions-summary.json", _json_bytes(summary))
    baseline_verdicts = []
    for concurrency in GRID:
        runs = sorted((row for row in predictions["baseline"]["runs"] if row["concurrency"] == concurrency),
                      key=lambda row: row["replication"])
        verdicts = ["PASS" if sla_verdict(row, requests_per_run) else "FAIL" for row in runs]
        baseline_verdicts.append({"concurrency": concurrency, "replication_verdicts": verdicts,
                                 "replication_statuses": [row["status"] for row in runs],
                                 "predicted_joint_verdict": "PASS" if all(value == "PASS" for value in verdicts) else "FAIL"})
    jobs = [{"cohort": cohort, "concurrency": concurrency, "replication": repetition}
            for cohort in COHORTS for repetition in range(replications) for concurrency in GRID]
    selection = {"rule": "Declared grid; prediction verdicts recorded only; no observed-data selection",
        "pre_registered_grid": list(GRID), "requests_per_run": requests_per_run, "replications": replications,
        "jobs_count": len(jobs), "baseline_configuration_role": "historical_retest",
        "new_blind_holdout_claim": False, "grid_disjoint_from_consumed_1111_holdout": False,
        "historical_reference_grid": json.loads((SRC / "baseline.spec.json").read_bytes())["concurrency_grid"],
        "selection_from_prediction": False, "no_observed_data_used": True,
        "baseline_predicted_verdicts_record_only": baseline_verdicts}
    _write_new(destination / "selection.json", _json_bytes(selection))
    _write_new(destination / "jobs.json", _json_bytes(jobs))
    if experiment_path.read_bytes() != experiment_raw or _code_files_sha256() != code_files:
        raise ValueError("Source experiment or simulator modules changed during the CPU draft")
    names = [str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()]
    files_sha = {name: sha_path(destination / name) for name in sorted(names)}
    manifest = {"kind": "course_sla_current_code_local_draft_v1", "status": "LOCAL_DRAFT",
        "generated_at": datetime.now(timezone.utc).isoformat(), "study_id": study_id,
        "requests_per_run": requests_per_run, "replications": replications, "jobs_count": len(jobs),
        "experiment_sha256": sha_bytes(experiment_raw), "source_experiment": str(experiment_path.relative_to(REPO)),
        "specs_sha256": {name: files_sha[f"{name}.spec.json"] for name in COHORTS},
        "files_sha256": files_sha, "code_files_sha256": code_files,
        "checkpoint_manifest": "REMOTE_PENDING", "checkpoint_manifest_sha256": "REMOTE_PENDING",
        "runtime_execution_config": "REMOTE_PENDING", "runtime_config_sha256": "REMOTE_PENDING",
        "scheduler_config_sha256": "REMOTE_PENDING", "design_contract_sha256": "REMOTE_PENDING",
        "identity": predictions["baseline"]["identity"], "identity_sha256": predictions["baseline"]["identity_sha256"],
        "prediction_artifacts_stored_locally": True, "prediction_artifact_paths": artifact_paths,
        "prediction_artifacts_sha256": raw_sha, "canonical_prediction_payload_sha256": canonical_sha,
        "hardware_validated": False, "remote_freeze_pending": True, "historical_reproduction": False,
        "freeze_semantics": "Current-code CPU local draft only; new raw/canonical hashes do not reproduce the historical 2026-11-16 commitment. Remote checkpoint/runtime/design receipt remains pending.",
        "local_manifest_sha256": None}
    manifest["local_manifest_sha256"] = canonical_sha256({key: value for key, value in manifest.items()
                                                        if key != "local_manifest_sha256"})
    _write_new(destination / "manifest.json", _json_bytes(manifest))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New exclusive output directory")
    parser.add_argument("--study-id", required=True, help="New draft identifier; historical IDs forbidden")
    parser.add_argument("--requests-per-run", type=int, default=1000)
    parser.add_argument("--replications", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        manifest = freeze_local(args.out, study_id=args.study_id, requests_per_run=args.requests_per_run,
                                replications=args.replications)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"LOCAL DRAFT {args.out} jobs={manifest['jobs_count']}")
    for name, digest in manifest["prediction_artifacts_sha256"].items():
        print(f"  {name}: raw={digest} canonical={manifest['canonical_prediction_payload_sha256'][name]}")
    print("hardware_validated=false remote_freeze_pending=true historical_reproduction=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
