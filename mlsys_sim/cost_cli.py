"""Explicit freeze, fit, validate and phase-cost prediction command line."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from .analytical_cost import analytical_cost_identity
from .config import experiment_from_dict
from .cost_calibration import (cost_data_sha256, fit_analytical_costs, freeze_cost_split,
                               validate_analytical_costs)
from .cost_report import write_cost_validation_report
from .performance import AnalyticalModel
from .schema import SimulationError
from .serialization import calibration_dict, experiment_dict
from .step_cost_calibration import fit_step_costs, freeze_step_fit_spec


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key}")
        result[key] = value
    return result


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    decoded = gzip.decompress(raw) if path.suffix == ".gz" else raw
    value = json.loads(decoded.decode("utf-8"), object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value, digest


def _write(path: Path, value: dict, *, frozen=False) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if frozen:
        path.chmod(0o444)


def _manifest(out: Path, command: str, inputs: dict) -> None:
    _write(out / "manifest.json", {"command": command, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_bytes_sha256": inputs, "hardware_validated": False})


def _fit_exports(dataset: dict, artifact: dict, out: Path, allow_fixture: bool) -> None:
    if not dataset.get("experiments"):
        return
    (out / "calibrations").mkdir()
    (out / "experiments").mkdir()
    index = {}
    for number, (model, value) in enumerate(sorted(dataset["experiments"].items())):
        experiment = experiment_from_dict(value)
        identity = analytical_cost_identity(experiment)
        selected = [cost for cost in artifact["cost_specs"] if cost["identity_sha256"] == identity]
        if not selected:
            raise SimulationError("CALIBRATION_MISMATCH", "No fitted cost identity matches an exported experiment")
        calibration = calibration_dict(experiment.calibration)
        calibration.update(artifact["calibration_patch"])
        calibration.update(analytical_costs=selected, profile_version=artifact["calibration_id"])
        if dataset["source"]["kind"] == "test_fixture":
            calibration["allow_test_fixture"] = allow_fixture
        mapping = experiment_dict(experiment)
        mapping["calibration"] = calibration
        configured = experiment_from_dict(mapping)
        name = f"model-{number}.json"
        _write(out / "calibrations" / name, calibration)
        _write(out / "experiments" / name, experiment_dict(configured))
        index[model] = {"calibration": "calibrations/" + name, "experiment": "experiments/" + name,
                        "identity_sha256": identity}
    _write(out / "models.json", index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze, fit and validate explicit Runtime-scoped analytical costs")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Freeze explicit membership and acceptance policy before fitting")
    freeze.add_argument("--dataset", required=True, type=Path)
    source = freeze.add_mutually_exclusive_group(required=True)
    source.add_argument("--membership", type=Path)
    source.add_argument("--recommended", action="store_true", help="Explicitly use the importer's label-based split")
    freeze.add_argument("--max-relative-error", type=float, default=0.3)
    freeze.add_argument("--step-affine", action="store_true",
                        help="Also freeze a whole-step affine fit policy; leaves the default component fit unchanged")
    freeze.add_argument("--max-training-process-ratio", type=float,
                        help="Maximum same-shape training process median ratio for --step-affine")
    fit = commands.add_parser("fit", help="Fit frozen training records only")
    fit.add_argument("--dataset", required=True, type=Path)
    fit.add_argument("--split", required=True, type=Path)
    fit.add_argument("--calibration-id", required=True)
    fit.add_argument("--allow-test-fixture", action="store_true")
    fit.add_argument("--fit-spec", type=Path,
                     help="Explicit frozen whole-step affine fit policy from freeze --step-affine")
    validate = commands.add_parser("validate", help="Validate frozen held-out records, without refitting")
    validate.add_argument("--dataset", required=True, type=Path)
    validate.add_argument("--fit", required=True, type=Path)
    predict = commands.add_parser("predict", help="Predict one declared phase and shape")
    predict.add_argument("--experiment", required=True, type=Path, help="Inline JSON experiment (or JSON.gz), no file references")
    predict.add_argument("--calibration", type=Path)
    predict.add_argument("--phase", choices=["prefill", "decode", "mixed"], required=True)
    predict.add_argument("--query-tokens", type=int, nargs="+", required=True)
    predict.add_argument("--context-lengths", type=int, nargs="+", required=True)
    predict.add_argument("--request-phases", choices=["prefill", "decode"], nargs="+",
                         help="Explicit ordered member phases for a mixed batch")
    for command in (freeze, fit, validate, predict):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise SimulationError("COST_OUTPUT_EXISTS", "Output directory must be new; frozen results are not overwritten")
        inputs = {}
        if args.command == "predict":
            runtime_snapshot = None
            mapping, inputs["experiment"] = _read(args.experiment)
            if any(isinstance(mapping.get(key), str) for key in
                   ("workload", "infra", "strategy", "calibration", "evaluation", "execution")):
                raise SimulationError("INVALID_COST_INPUT", "Phase prediction requires inline experiment sections")
            strategy = mapping.get("strategy", {})
            if isinstance(strategy, dict) and isinstance(strategy.get("model"), str):
                raise SimulationError("INVALID_COST_INPUT", "Phase prediction requires inline model geometry")
            experiment = experiment_from_dict(mapping)
            if args.calibration:
                calibration, inputs["calibration"] = _read(args.calibration)
                mapping = experiment_dict(experiment)
                mapping["calibration"] = calibration
                experiment = experiment_from_dict(mapping)
            if args.request_phases is not None and args.phase != "mixed":
                raise SimulationError("INVALID_COST_INPUT", "Member phases require --phase mixed")
            if experiment.calibration.backend in {"profile", "runtime"}:
                from .profiles import create_performance_model
                profile_root = (args.calibration or args.experiment).parent
                path = Path(experiment.calibration.profile_path or "")
                experiment = replace(experiment, calibration=replace(experiment.calibration,
                    profile_path=str((profile_root / path).resolve())))
                if experiment.calibration.backend == "runtime":
                    from .capacity import _bind_inputs, _profile_inputs
                    snapshots = _profile_inputs(experiment)
                    runtime_snapshot = next(item for item in snapshots if item["field"] == "profile_path")
                    with tempfile.TemporaryDirectory(prefix="mlsys-runtime-phase-") as temporary:
                        bound = _bind_inputs(experiment, snapshots, Path(temporary))
                        model = create_performance_model(bound)
                else:
                    model = create_performance_model(experiment)
                provenance = model.provenance
                if not isinstance(provenance, dict):
                    raise SimulationError("INVALID_COST_INPUT", "Selected cost model has no input provenance")
                role = "runtime_cost" if experiment.calibration.backend == "runtime" else "profile"
                inputs[role] = provenance["sha256"]
                estimate = model.estimate_step(args.phase, args.query_tokens, args.context_lengths,
                                               request_phases=args.request_phases)
            else:
                estimate = AnalyticalModel(experiment).estimate_step(args.phase, args.query_tokens, args.context_lengths,
                                                                    request_phases=args.request_phases)
            result = {"kind": "phase_cost_prediction_v1", "phase": args.phase,
                "query_tokens": args.query_tokens, "context_lengths": args.context_lengths,
                "estimate": asdict(estimate), "hardware_validated": False,
                "memory_feasibility": "not_assessed; isolated phase cost only"}
            if args.phase == "mixed":
                result["request_phases"] = args.request_phases
            args.out.mkdir(parents=True)
            if runtime_snapshot is not None:
                (args.out / "runtime-cost.json").write_bytes(runtime_snapshot["raw_json"].encode("utf-8"))
                _write(args.out / "runtime-cost-input.json", {
                    key: value for key, value in runtime_snapshot.items() if key != "raw_json"})
            _write(args.out / "prediction.json", result)
            _manifest(args.out, args.command, inputs)
            print(json.dumps({"status": "COMPLETED", "duration_ms": estimate.duration_ms,
                              "source": estimate.source, "out": str(args.out)}))
            return 0
        dataset, inputs["dataset"] = _read(args.dataset)
        if args.command == "freeze":
            if args.max_training_process_ratio is not None and not args.step_affine:
                raise SimulationError("INVALID_COST_INPUT", "Training stability policy requires --step-affine")
            if args.membership:
                membership, inputs["membership"] = _read(args.membership)
            else:
                membership = dataset["recommended_split"]
            result = freeze_cost_split(dataset, training_record_ids=membership["training_record_ids"],
                holdout_record_ids=membership["holdout_record_ids"],
                repeat_record_ids=membership.get("repeat_record_ids", []), max_relative_error=args.max_relative_error)
            fit_spec = (freeze_step_fit_spec(result,
                max_training_process_ratio=(1.25 if args.max_training_process_ratio is None
                                            else args.max_training_process_ratio)) if args.step_affine else None)
            args.out.mkdir(parents=True)
            _write(args.out / "split.json", result, frozen=True)
            if fit_spec is not None:
                _write(args.out / "fit-spec.json", fit_spec, frozen=True)
        elif args.command == "fit":
            if dataset["source"]["kind"] == "test_fixture" and not args.allow_test_fixture:
                raise SimulationError("TEST_FIXTURE_NOT_ALLOWED", "Fitting test fixtures requires --allow-test-fixture")
            split, inputs["split"] = _read(args.split)
            if args.fit_spec:
                fit_spec, inputs["fit_spec"] = _read(args.fit_spec)
                result = fit_step_costs(dataset, split, calibration_id=args.calibration_id, fit_spec=fit_spec)
            else:
                result = fit_analytical_costs(dataset, split, calibration_id=args.calibration_id)
            args.out.mkdir(parents=True)
            if result.get("kind") == "step_cost_fit_rejected_v1":
                _write(args.out / "fit-rejected.json", result, frozen=True)
                _manifest(args.out, args.command, inputs)
                print(json.dumps({"status": "REJECTED", "command": args.command,
                                  "out": str(args.out), "verdict": "FAIL"}))
                return 3
            _fit_exports(dataset, result, args.out, args.allow_test_fixture)
            _write(args.out / "fit.json", result, frozen=True)
        else:
            artifact, inputs["fit"] = _read(args.fit)
            result = validate_analytical_costs(dataset, artifact)
            write_cost_validation_report(result, args.out)
        _manifest(args.out, args.command, inputs)
        print(json.dumps({"status": "COMPLETED", "command": args.command, "out": str(args.out),
                          "verdict": result.get("verdict"), "artifact_sha256": cost_data_sha256(result)}))
        return 3 if result.get("verdict") == "FAIL" else 0
    except (SimulationError, ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAILED", "code": getattr(error, "code", "INVALID_COST_INPUT"),
                          "message": str(error)}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
