"""Immutable CLI artifacts for simulation, profiles and validation workflows."""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, NoReturn

from . import __version__
from .config import load_experiment
from .schema import SimulationError


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_CONFIG", f"Expected finite JSON data: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    _json(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fresh_output(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite {path}") from exc


def _record_input(manifest: dict, path: str | Path, raw: bytes | None = None) -> None:
    source = Path(path).resolve()
    content = source.read_bytes() if raw is None else raw
    entry = {"path": str(source), "sha256": hashlib.sha256(content).hexdigest()}
    existing = [item for item in manifest.setdefault("input_files", []) if item["path"] != str(source)]
    manifest["input_files"] = sorted([*existing, entry], key=lambda item: item["path"])


def _experiment_inputs(manifest: dict, path: Path, experiment: Any) -> None:
    from .config import _read

    _record_input(manifest, path)
    original = _read(path)
    strategy_root, execution_root = path.parent, path.parent
    strategy, execution = original.get("strategy", {}), original.get("execution", {})
    for name in ("workload", "infra", "strategy", "calibration", "evaluation", "execution"):
        reference = original.get(name)
        if isinstance(reference, str):
            referenced = path.parent / reference
            _record_input(manifest, referenced)
            if name == "strategy":
                strategy, strategy_root = _read(referenced), referenced.parent
            if name == "execution":
                execution, execution_root = _read(referenced), referenced.parent
    if isinstance(strategy, dict) and isinstance(strategy.get("model"), str):
        _record_input(manifest, strategy_root / strategy["model"])
    if isinstance(execution, dict) and isinstance(execution.get("warmup"), str):
        _record_input(manifest, execution_root / execution["warmup"])
    if experiment.calibration.profile_path:
        _record_input(manifest, experiment.calibration.profile_path)
    for profile_path in experiment.calibration.replica_profile_paths:
        _record_input(manifest, profile_path)
    if experiment.calibration.native_pp_communication_profile_path:
        _record_input(manifest, experiment.calibration.native_pp_communication_profile_path)
    for profile_path in experiment.calibration.replica_native_pp_communication_profile_paths:
        _record_input(manifest, profile_path)


def _read_json(path: Path) -> tuple[Any, bytes]:
    def unique(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise SimulationError("INVALID_CONFIG", f"Duplicate JSON key {key!r} in {path}")
            data[key] = value
        return data
    try:
        raw = path.read_bytes()
        payload = json.loads(raw, object_pairs_hook=unique)
        _json(payload)
        return payload, raw
    except (OSError, ValueError) as exc:
        if isinstance(exc, SimulationError):
            raise
        raise SimulationError("INVALID_CONFIG", f"Cannot read {path}: {exc}") from exc


def _simulation(args: argparse.Namespace, manifest: dict) -> dict:
    from .comparison import comparison_contract

    original = load_experiment(args.experiment)
    _experiment_inputs(manifest, args.experiment, original)
    expanded = asdict(original)
    fingerprint_config = expanded
    if (original.calibration.native_pp_communication_profile_path is None
            and not original.calibration.replica_native_pp_communication_profile_paths):
        # New opt-in boundary settings do not change pre-existing run hashes.
        fingerprint_config = {**expanded, "calibration": {
            key: value for key, value in expanded["calibration"].items() if key not in {
                "native_pp_communication_profile_path", "replica_native_pp_communication_profile_paths",
                "native_pp_communication_profile_version", "native_pp_communication_allow_extrapolation"}}}
    manifest.update({"backend": "simulated", "experiment_path": str(args.experiment.resolve()),
                     "effective_config": expanded, "seed": original.execution.seed,
                     "comparison_contract": comparison_contract(original),
                     "calibration_source": original.calibration.backend,
                     "profile_version": original.calibration.profile_version})
    experiment, profile = original, None
    replica_profiles: list[dict] = []
    origin = None
    if original.calibration.backend == "profile" and original.strategy.replicas > 1:
        from .layout import replica_experiment
        from .profiles import profile_identity, validate_profile

        snapshots = []
        (args.out / "profiles").mkdir()
        for replica_index, profile_path in enumerate(original.calibration.replica_profile_paths):
            source = Path(profile_path).resolve()
            payload, raw = _read_json(source)
            current = validate_profile(payload, allow_test_fixture=original.calibration.allow_test_fixture)
            expected = profile_identity(replica_experiment(original, replica_index))
            if current["identity"] != expected:
                raise SimulationError("CALIBRATION_MISMATCH",
                                      f"Profile identity does not match replica {replica_index}")
            requested_version = original.calibration.profile_version
            if requested_version != "uncalibrated-v1" and requested_version != current["profile_version"]:
                raise SimulationError("CALIBRATION_MISMATCH",
                                      f"Profile version does not match replica {replica_index}")
            snapshot_file = f"profiles/replica-{replica_index}.json"
            snapshot = args.out / snapshot_file
            with snapshot.open("xb") as stream:
                stream.write(raw)
            profile_hash = hashlib.sha256(raw).hexdigest()
            _record_input(manifest, source, raw)
            replica_profiles.append({
                "replica_index": replica_index, "origin_profile_path": str(source),
                "snapshot_file": snapshot_file, "sha256": profile_hash,
                "profile_id": current["profile_id"], "profile_version": current["profile_version"],
                "source_kind": current["source"]["kind"], "source_reference": current["source"]["reference"],
            })
            # Keep partial snapshot provenance available if a later input fails.
            manifest["replica_profiles"] = replica_profiles
            snapshots.append(str(snapshot.resolve()))
        kinds = {item["source_kind"] for item in replica_profiles}
        versions = {item["profile_version"] for item in replica_profiles}
        manifest.update(
            profile_version=next(iter(versions)) if len(versions) == 1 else "per_replica",
            calibration_source=("measured_profile" if kinds == {"measured"} else
                                "test_fixture_profile" if kinds == {"test_fixture"} else
                                "mixed_replica_profiles"),
        )
        experiment = replace(original, calibration=replace(original.calibration,
                                                           replica_profile_paths=tuple(snapshots)))
        manifest["config_sha256"] = hashlib.sha256(_json({
            "experiment": fingerprint_config,
            "replica_profile_sha256": [item["sha256"] for item in replica_profiles],
        }).encode()).hexdigest()
    elif original.calibration.backend == "profile":
        from .profiles import validate_profile

        if original.calibration.profile_path is None:
            raise SimulationError("INVALID_CONFIG", "profile_path is required for the profile backend")
        origin = Path(original.calibration.profile_path).resolve()
        payload, raw = _read_json(origin)
        profile = validate_profile(payload, allow_test_fixture=original.calibration.allow_test_fixture)
        # Timing and workspace both consume this exact immutable copy.
        snapshot = args.out / "profile.json"
        with snapshot.open("xb") as stream:
            stream.write(raw)
        profile_hash = hashlib.sha256(raw).hexdigest()
        _record_input(manifest, origin, raw)
        manifest.update(profile_sha256=profile_hash, profile_snapshot="profile.json",
                        origin_profile_path=str(origin), profile_version=profile["profile_version"],
                        calibration_source=("measured_profile" if profile["source"]["kind"] == "measured"
                                            else "test_fixture_profile"))
        experiment = replace(original, calibration=replace(original.calibration,
                                                           profile_path=str(snapshot.resolve())))
        manifest["config_sha256"] = hashlib.sha256(
            _json({"experiment": fingerprint_config, "profile_sha256": profile_hash}).encode()).hexdigest()
    else:
        manifest["config_sha256"] = hashlib.sha256(_json(fingerprint_config).encode()).hexdigest()
    communication_profiles = []
    communication_paths = (original.calibration.replica_native_pp_communication_profile_paths
        or ((original.calibration.native_pp_communication_profile_path,)
            if original.calibration.native_pp_communication_profile_path else ()))
    if communication_paths:
        from .layout import replica_experiment
        from .pp_communication import validate_pp_communication_profile
        from .profiles import profile_identity

        (args.out / "communication-profiles").mkdir()
        communication_snapshots = []
        for replica_index, communication_path in enumerate(communication_paths):
            source = Path(communication_path).resolve()
            payload, raw = _read_json(source)
            current = validate_pp_communication_profile(payload,
                allow_test_fixture=original.calibration.allow_test_fixture)
            if current["identity"] != profile_identity(replica_experiment(original, replica_index)):
                raise SimulationError("CALIBRATION_MISMATCH",
                                      f"Communication Profile identity does not match replica {replica_index}")
            requested = original.calibration.native_pp_communication_profile_version
            if requested and requested != current["profile_version"]:
                raise SimulationError("CALIBRATION_MISMATCH", "Native PP communication Profile version differs")
            snapshot_file = f"communication-profiles/replica-{replica_index}.json"
            snapshot = args.out / snapshot_file
            with snapshot.open("xb") as stream:
                stream.write(raw)
            _record_input(manifest, source, raw)
            communication_profiles.append({"replica_index": replica_index,
                "origin_profile_path": str(source), "snapshot_file": snapshot_file,
                "sha256": hashlib.sha256(raw).hexdigest(), "profile_id": current["profile_id"],
                "profile_version": current["profile_version"], "source_kind": current["source"]["kind"],
                "source_reference": current["source"]["reference"]})
            communication_snapshots.append(str(snapshot.resolve()))
            manifest["native_pp_communication_profiles"] = communication_profiles
        calibration = (replace(experiment.calibration,
            native_pp_communication_profile_path=communication_snapshots[0]) if original.strategy.replicas == 1 else
            replace(experiment.calibration,
                replica_native_pp_communication_profile_paths=tuple(communication_snapshots)))
        experiment = replace(experiment, calibration=calibration)
        fingerprint: dict[str, Any] = {"experiment": expanded,
            "native_pp_communication_profile_sha256": [item["sha256"] for item in communication_profiles]}
        if profile is not None:
            fingerprint["profile_sha256"] = manifest["profile_sha256"]
        if replica_profiles:
            fingerprint["replica_profile_sha256"] = [item["sha256"] for item in replica_profiles]
        manifest["config_sha256"] = hashlib.sha256(_json(fingerprint).encode()).hexdigest()
    _write_json(args.out / "manifest.json", manifest)
    if args.mode == "simulate":
        from .engine import simulate

        run = simulate(experiment)
        result, requests, events = run.result, run.requests, run.events
    else:
        from .performance import predict_experiment

        result, requests, events = predict_experiment(experiment), [], []
        result.setdefault("status", "COMPLETED")
        result.setdefault("backend", "simulated")
    if profile is not None:
        provenance = result.get("profile_provenance")
        if isinstance(provenance, dict):
            provenance.update(origin_profile_path=str(origin), snapshot_file="profile.json")
            manifest["profile_provenance"] = dict(provenance)
    if replica_profiles:
        indexed = {item["replica_index"]: item for item in replica_profiles}

        def annotate(provenance, replica_index):
            if isinstance(provenance, dict) and replica_index in indexed:
                item = indexed[replica_index]
                provenance.update(origin_profile_path=item["origin_profile_path"],
                                  snapshot_file=item["snapshot_file"])

        provenance = result.get("profile_provenance")
        if isinstance(provenance, dict):
            for item in provenance.get("replicas", []):
                if isinstance(item, dict):
                    annotate(item, item.get("replica_index"))
            manifest["profile_provenance"] = dict(provenance)
        for collection, local_key in (("replicas", "result"), ("replica_predictions", "prediction")):
            for item in result.get(collection, []):
                if isinstance(item, dict) and isinstance(item.get(local_key), dict):
                    annotate(item[local_key].get("profile_provenance"), item.get("replica_index"))
    if communication_profiles:
        communication_index = {item["replica_index"]: item for item in communication_profiles}

        def annotate_communication(current, replica_index):
            if replica_index not in communication_index or not isinstance(current, dict):
                return
            item = communication_index[replica_index]
            for section, key in (("communication", "profile_provenance"),
                                 ("pipeline", "communication_profile_provenance")):
                provenance = current.get(section, {}).get(key)
                if isinstance(provenance, dict):
                    provenance.update(origin_profile_path=item["origin_profile_path"],
                                      snapshot_file=item["snapshot_file"])

        if original.strategy.replicas == 1:
            annotate_communication(result, 0)
        for collection, local_key in (("replicas", "result"), ("replica_predictions", "prediction")):
            for item in result.get(collection, []):
                if isinstance(item, dict):
                    annotate_communication(item.get(local_key), item.get("replica_index"))
    for name, rows in (("requests.jsonl", requests), ("events.jsonl", events)):
        if name == "events.jsonl" and not experiment.execution.detailed_events:
            continue
        with (args.out / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(_json(row) + "\n")
    return result


def _calibrate(args: argparse.Namespace, manifest: dict) -> dict:
    from .calibration import calibrate_dataset
    from .profiles import load_dataset, load_profile

    _record_input(manifest, args.dataset)
    native = getattr(args, "native_pp_communication", False)
    if native:
        from .pp_communication import calibrate_pp_communication, load_pp_communication_dataset
        dataset = load_pp_communication_dataset(args.dataset, allow_test_fixture=args.allow_test_fixture)
    elif args.communication:
        from .communication_calibration import calibrate_communication, load_communication_dataset

        dataset = load_communication_dataset(args.dataset, allow_test_fixture=args.allow_test_fixture)
    else:
        dataset = load_dataset(args.dataset, allow_test_fixture=args.allow_test_fixture)
    parent = None
    if args.parent_profile is not None:
        _record_input(manifest, args.parent_profile)
        if native or args.communication:
            parent, _ = _read_json(args.parent_profile)
            if not isinstance(parent, dict):
                raise SimulationError("INVALID_ARGUMENT", "Parent profile must be a JSON object")
        else:
            parent = load_profile(args.parent_profile, allow_test_fixture=args.allow_test_fixture)
    fit = calibrate_pp_communication if native else calibrate_communication if args.communication else calibrate_dataset
    profile = fit(dataset, profile_version=args.profile_version,
                  parent_profile=parent, allow_test_fixture=args.allow_test_fixture)
    _write_json(args.out / "dataset.json", dataset)
    _write_json(args.out / "profile.json", profile)
    manifest.update(profile_version=profile["profile_version"], source=profile["source"],
                    dataset_sha256=profile["training"]["dataset_sha256"],
                    profile_sha256=hashlib.sha256((args.out / "profile.json").read_bytes()).hexdigest())
    return {"status": "COMPLETED", "mode": "calibrate", "profile_id": profile["profile_id"],
            "profile_version": profile["profile_version"], "source": profile["source"],
            "training_record_count": len(profile["training"]["record_ids"]),
            "shape_count": len(profile.get("samples", [])),
            "communication_group_count": len(profile.get("groups", [])),
            "profile_kind": profile.get("kind", "step_profile"),
            "parent_profile_version": profile["training"]["parent_profile_version"]}


def _validate(args: argparse.Namespace, manifest: dict) -> dict:
    from .calibration import calibrate_dataset, split_dataset, validate_holdout
    from .profiles import load_dataset, load_profile

    if getattr(args, "native_pp_communication", False):
        from .pp_communication import (load_pp_communication_dataset, load_pp_communication_profile,
                                       validate_pp_communication_holdout)
        if args.profile is None or any(value is not None for value in
                                      (args.holdout_group, args.holdout_fraction, args.seed)):
            raise SimulationError("INVALID_ARGUMENT", "Native communication validation requires --profile "
                                  "and an independent holdout dataset; split options are unsupported")
        _record_input(manifest, args.dataset)
        _record_input(manifest, args.profile)
        dataset = load_pp_communication_dataset(args.dataset, allow_test_fixture=args.allow_test_fixture)
        profile = load_pp_communication_profile(args.profile, allow_test_fixture=args.allow_test_fixture)
        _write_json(args.out / "dataset.json", dataset)
        _write_json(args.out / "holdout.json", dataset)
        _write_json(args.out / "profile.json", profile)
        manifest.update(profile_version=profile["profile_version"], source=profile["source"],
            profile_sha256=hashlib.sha256((args.out / "profile.json").read_bytes()).hexdigest(),
            dataset_sha256=profile["training"]["dataset_sha256"])
        return validate_pp_communication_holdout(profile, dataset, allow_test_fixture=args.allow_test_fixture,
                                                 allow_extrapolation=args.allow_extrapolation)
    if args.profile is not None and any(value is not None for value in
                                       (args.holdout_group, args.holdout_fraction, args.seed)):
        raise SimulationError("INVALID_ARGUMENT", "Existing-profile validation consumes a holdout dataset; "
                              "split options require --profile-version")
    _record_input(manifest, args.dataset)
    dataset = load_dataset(args.dataset, allow_test_fixture=args.allow_test_fixture)
    _write_json(args.out / "dataset.json", dataset)
    if args.profile is not None:
        _record_input(manifest, args.profile)
        profile = load_profile(args.profile, allow_test_fixture=args.allow_test_fixture)
        holdout = dataset
    else:
        fraction = 0.25 if args.holdout_fraction is None else args.holdout_fraction
        seed = 42 if args.seed is None else args.seed
        training, holdout = split_dataset(dataset, holdout_fraction=fraction, seed=seed,
                                          holdout_groups=args.holdout_group,
                                          allow_test_fixture=args.allow_test_fixture)
        profile = calibrate_dataset(training, profile_version=args.profile_version,
                                    allow_test_fixture=args.allow_test_fixture)
        _write_json(args.out / "train.json", training)
        manifest["split"] = {"holdout_fraction": fraction, "seed": seed,
                             "holdout_groups": args.holdout_group}
    _write_json(args.out / "holdout.json", holdout)
    _write_json(args.out / "profile.json", profile)
    manifest.update(profile_version=profile["profile_version"], source=profile["source"],
                    profile_sha256=hashlib.sha256((args.out / "profile.json").read_bytes()).hexdigest(),
                    dataset_sha256=profile["training"]["dataset_sha256"])
    return validate_holdout(profile, holdout, allow_test_fixture=args.allow_test_fixture,
                            allow_extrapolation=args.allow_extrapolation)


def _comparison_inputs(manifest: dict, predicted: Path, observed: Path) -> None:
    directory = predicted if predicted.is_dir() else predicted.parent
    _record_input(manifest, directory / "manifest.json")
    _record_input(manifest, directory / "result.json" if predicted.is_dir() else predicted)
    if (directory / "requests.jsonl").exists():
        _record_input(manifest, directory / "requests.jsonl")
    _record_input(manifest, observed)


def _compare(args: argparse.Namespace, manifest: dict) -> dict:
    from .comparison import compare_candidates, compare_runs

    if args.pairs is not None:
        if args.observed is not None:
            raise SimulationError("INVALID_ARGUMENT", "--observed cannot be combined with --pairs")
        pairs, raw = _read_json(args.pairs)
        _record_input(manifest, args.pairs, raw)
        if not isinstance(pairs, list):
            raise SimulationError("INVALID_ARGUMENT", "The pairs JSON file must contain an array")
        resolved = []
        for pair in pairs:
            if (not isinstance(pair, dict) or not isinstance(pair.get("predicted"), str)
                    or not isinstance(pair.get("observed"), str)):
                raise SimulationError("INVALID_ARGUMENT", "Each pair needs predicted/observed file paths")
            current = dict(pair)
            current["predicted"] = str((args.pairs.parent / pair["predicted"]).resolve())
            current["observed"] = str((args.pairs.parent / pair["observed"]).resolve())
            _comparison_inputs(manifest, Path(current["predicted"]), Path(current["observed"]))
            resolved.append(current)
        return compare_candidates(resolved, metric=args.metric, higher_is_better=not args.lower_is_better,
                                  allow_test_fixture=args.allow_test_fixture)
    if args.predicted is None or args.observed is None:
        raise SimulationError("INVALID_ARGUMENT", "Single-run comparison requires --predicted and --observed")
    _comparison_inputs(manifest, args.predicted, args.observed)
    result = compare_runs(args.predicted, args.observed, allow_test_fixture=args.allow_test_fixture)
    result["focus_metric"] = {"name": args.metric, "higher_is_better": not args.lower_is_better,
                              "comparison": result["metrics"][args.metric]}
    return result


def _save_collection(bundle: dict, output: Path, manifest: dict) -> dict:
    """Keep raw evidence in fixed artifact paths, separate from the summary."""
    if bundle.get("kind") == "native_pp_communication_collection":
        for record in bundle.get("raw_records", []):
            captured_input = record.get("raw_input") if isinstance(record, dict) else None
            if isinstance(captured_input, dict) and "path" in captured_input and "sha256" in captured_input:
                entry = {key: captured_input[key] for key in ("path", "sha256")}
                # Use the collector's bounded read, including on partial failure,
                # rather than a later version of a mutable source file.
                previous = [item for item in manifest.setdefault("input_files", [])
                            if item["path"] != entry["path"]]
                manifest["input_files"] = sorted([*previous, entry], key=lambda item: item["path"])
    json_files = {"inventory": "inventory.json", "experiment": "experiment.json",
                  "effective_experiment": "effective_experiment.json", "dataset": "dataset.json",
                  "observation": "observed.json", "audit": "audit.json", "raw_records": "raw_records.json",
                  "client_metrics": "client_metrics.json"}
    line_files = {"request_service_records": "request_service.jsonl", "client_requests": "client_requests.jsonl"}
    artifacts: dict[str, Any] = {}
    for key, filename in json_files.items():
        if bundle.get(key) is not None:
            destination = output / filename
            if destination.exists():
                raise SimulationError("OUTPUT_EXISTS", f"Collection artifact already exists: {destination}")
            _write_json(destination, bundle[key])
            artifacts[key] = {"file": filename, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    for key, filename in line_files.items():
        if bundle.get(key) is not None:
            destination = output / filename
            with destination.open("x", encoding="utf-8") as stream:
                for row in bundle[key]:
                    stream.write(_json(row) + "\n")
            artifacts[key] = {"file": filename, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    if bundle.get("datasets") is not None:
        directory = output / "datasets"
        directory.mkdir()
        datasets = []
        for item in bundle["datasets"]:
            replica = item["replica_index"]
            if type(replica) is not int or replica < 0:
                raise SimulationError("INVALID_VLLM_CAPTURE", "Dataset replica index must be nonnegative")
            destination = directory / f"replica-{replica}.json"
            if destination.exists():
                raise SimulationError("OUTPUT_EXISTS", "A replica dataset artifact already exists")
            _write_json(destination, item["dataset"])
            datasets.append({"replica_index": replica, "file": str(destination.relative_to(output)),
                             "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        artifacts["datasets"] = datasets
    manifest["collection_artifacts"] = artifacts
    if "source" in bundle:
        manifest["source"] = bundle["source"]
    result = {key: value for key, value in bundle.items()
              if key not in json_files and key not in line_files and key != "datasets"}
    result["artifacts"] = artifacts
    result["hardware_validated"] = False
    return result


def _collection(args: argparse.Namespace, manifest: dict) -> dict:
    from .hardware import CollectionError, probe_rocm, read_json

    allowed = {
        "probe": {"rccl_library", "rocm_root"},
        "steps": {"experiment", "plan", "adapter", "inventory", "rccl_library", "rocm_root"},
        "rccl": {"experiment", "plan", "inventory", "rccl_library", "rocm_root"},
        "pp-communication": {"experiment", "plan", "inventory", "rccl_library", "rocm_root"},
        "runtime-trace": {"experiment", "raw"},
        "vllm": {"experiment", "plan", "inventory", "rccl_library", "rocm_root"},
        "vllm-hook": {"experiment", "plan", "raw"},
        "reference": {"reference_dir"},
    }[args.kind]
    for field in ("experiment", "plan", "adapter", "raw", "reference_dir", "inventory", "rccl_library", "rocm_root"):
        if getattr(args, field, None) is not None and field not in allowed:
            raise SimulationError("INVALID_ARGUMENT", f"--{field.replace('_', '-')} does not apply to {args.kind}")
    if args.inventory is not None and (args.rccl_library is not None or args.rocm_root is not None):
        raise SimulationError("INVALID_ARGUMENT", "Use an existing --inventory or fresh probe options, not both")
    if args.kind == "probe":
        inventory = probe_rocm(rccl_library=args.rccl_library, rocm_root=args.rocm_root or Path("/opt/rocm"))
        return _save_collection({"status": "COMPLETED", "kind": "rocm_probe", "inventory": inventory,
                                 "hardware_available": inventory["status"] == "READY",
                                 "source": inventory["source"]}, args.out, manifest)
    if args.kind == "reference":
        if args.reference_dir is None:
            raise SimulationError("INVALID_ARGUMENT", "Reference import requires --reference-dir")
        from .release_adapter import import_reference_data
        bundle = import_reference_data(args.reference_dir, allow_test_fixture=args.allow_test_fixture)
        return _save_collection(bundle, args.out, manifest)
    if args.experiment is None:
        raise SimulationError("INVALID_ARGUMENT", f"{args.kind} requires --experiment")
    experiment = load_experiment(args.experiment)
    _experiment_inputs(manifest, args.experiment, experiment)
    if args.kind == "runtime-trace":
        if args.raw is None:
            raise SimulationError("INVALID_ARGUMENT", "Runtime trace import requires --raw")
        from .runtime_adapter import normalize_runtime_trace
        _record_input(manifest, args.raw)
        bundle = normalize_runtime_trace(experiment, read_json(args.raw), allow_test_fixture=args.allow_test_fixture)
        return _save_collection(bundle, args.out, manifest)
    if args.kind == "vllm-hook":
        from .hardware import object_fields
        from .vllm_capture import load_vllm_captures, normalize_vllm_capture

        if (args.raw is None) == (args.plan is None):
            raise SimulationError("INVALID_ARGUMENT", "Hook import needs exactly one of --raw or --plan")
        runtime_trace = None
        bind_runtime_workload = False
        if args.plan is not None:
            _record_input(manifest, args.plan)
            plan = object_fields(read_json(args.plan), {"schema_version", "captures"},
                                 {"runtime_trace", "bind_runtime_workload"}, "vLLM hook import plan")
            if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
                raise SimulationError("INVALID_COLLECTION_PLAN", "Hook plan schema_version must be 1")
            if (not isinstance(plan["captures"], list) or not plan["captures"]
                    or any(not isinstance(value, str) or not value.strip() for value in plan["captures"])):
                raise SimulationError("INVALID_COLLECTION_PLAN", "captures must contain capture-file paths")
            paths = [args.plan.parent / value for value in plan["captures"]]
            bind_runtime_workload = plan.get("bind_runtime_workload", False)
            if type(bind_runtime_workload) is not bool:
                raise SimulationError("INVALID_COLLECTION_PLAN", "bind_runtime_workload must be boolean")
            if "runtime_trace" in plan:
                if not isinstance(plan["runtime_trace"], str) or not plan["runtime_trace"].strip():
                    raise SimulationError("INVALID_COLLECTION_PLAN", "runtime_trace must be a path")
                path = args.plan.parent / plan["runtime_trace"]
                _record_input(manifest, path)
                runtime_trace = read_json(path)
        else:
            paths = [args.raw]
        captures = load_vllm_captures(paths)
        for capture in captures:
            provenance = capture.get("_capture_input", {})
            if "path" in provenance and "sha256" in provenance:
                manifest["input_files"].append({"path": provenance["path"], "sha256": provenance["sha256"]})
        bundle = normalize_vllm_capture(experiment, captures, allow_test_fixture=args.allow_test_fixture,
                                        runtime_trace=runtime_trace, bind_runtime_workload=bind_runtime_workload)
        return _save_collection(bundle, args.out, manifest)
    if args.plan is None:
        raise SimulationError("INVALID_ARGUMENT", f"{args.kind} requires --plan")
    _record_input(manifest, args.plan)
    plan = read_json(args.plan)
    if not isinstance(plan, dict):
        raise SimulationError("INVALID_COLLECTION_PLAN", "Collection plan must be an object")
    if args.kind == "pp-communication" and isinstance(plan.get("raw_path"), str):
        raw_path = args.plan.parent / plan["raw_path"]
        with raw_path.open("rb") as stream:
            raw = stream.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise SimulationError("COLLECTION_OUTPUT_LIMIT", "Communication raw input exceeds 16MiB")
        _record_input(manifest, raw_path, raw)
    if args.inventory is not None:
        _record_input(manifest, args.inventory)
        inventory = read_json(args.inventory)
    else:
        inventory = probe_rocm(rccl_library=args.rccl_library, rocm_root=args.rocm_root or Path("/opt/rocm"))
    # Preserve the actual readiness result even when collection cannot start.
    _write_json(args.out / "inventory.json", inventory)
    manifest["inventory_sha256"] = hashlib.sha256((args.out / "inventory.json").read_bytes()).hexdigest()
    try:
        if args.kind == "steps":
            if args.adapter is None or args.adapter.count(":") != 1:
                raise SimulationError("INVALID_ARGUMENT", "Steps require --adapter module:factory")
            if not isinstance(inventory, dict) or inventory.get("status") != "READY":
                raise SimulationError("HARDWARE_UNAVAILABLE", "ROCm step collection needs an available HIP GPU")
            module_name, factory_name = args.adapter.split(":")
            try:
                module = importlib.import_module(module_name)
                factory = getattr(module, factory_name)
                if not callable(factory):
                    raise TypeError("Adapter factory is not callable")
                adapter = factory(experiment)
            except Exception as exc:
                raise SimulationError("COLLECTION_FAILED", f"Cannot create Runtime adapter: {exc}") from exc
            manifest["runtime_adapter"] = {"entry_point": args.adapter}
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).is_file():
                _record_input(manifest, filename)
            from .rocm_steps import collect_steps
            bundle = collect_steps(experiment, plan, adapter, inventory, allow_test_fixture=args.allow_test_fixture)
        elif args.kind == "rccl":
            from .rccl_adapter import collect_rccl
            bundle = collect_rccl(experiment, plan, inventory, base_dir=args.plan.parent,
                                  allow_test_fixture=args.allow_test_fixture)
        elif args.kind == "pp-communication":
            from .pp_benchmark import collect_pp_communication
            bundle = collect_pp_communication(experiment, plan, inventory, base_dir=args.plan.parent,
                                               allow_test_fixture=args.allow_test_fixture)
        else:
            from .runtime_adapter import collect_vllm
            bundle = collect_vllm(experiment, plan, inventory, base_dir=args.plan.parent,
                                 allow_test_fixture=args.allow_test_fixture)
    except CollectionError as exc:
        if exc.artifacts:
            # inventory.json already exists; preserve a single immutable copy.
            partial = {key: value for key, value in exc.artifacts.items() if key != "inventory"}
            _save_collection(partial, args.out, manifest)
        raise
    # The readiness snapshot was already saved before invoking the adapter.
    bundle = dict(bundle)
    captured_inventory = bundle.pop("inventory", None)
    if captured_inventory is not None and captured_inventory != inventory:
        _write_json(args.out / "execution_inventory.json", captured_inventory)
        manifest["execution_inventory_sha256"] = hashlib.sha256(
            (args.out / "execution_inventory.json").read_bytes()).hexdigest()
    result = _save_collection(bundle, args.out, manifest)
    result["artifacts"]["inventory"] = {"file": "inventory.json", "sha256": manifest["inventory_sha256"]}
    return result


def _hardware_validation(args: argparse.Namespace, manifest: dict) -> dict:
    from .hardware import read_json
    from .hardware_validation import run_hardware_validation

    _record_input(manifest, args.plan)
    result = run_hardware_validation(read_json(args.plan), args.out, base_dir=args.plan.parent,
                                     allow_test_fixture=args.allow_test_fixture)
    manifest["validation_scope"] = result.get("validation_scope")
    manifest["hardware_validated"] = result.get("hardware_validated", False)
    return result


def _model_import(args: argparse.Namespace, manifest: dict) -> dict:
    from .qwen3 import load_qwen3_model

    bundle = load_qwen3_model(args.config, name=args.name, revision=args.revision, kv_dtype=args.kv_dtype)
    provenance = bundle["provenance"]
    manifest["input_files"].append({"path": provenance["source_path"], "sha256": provenance["source_sha256"]})
    manifest["model_provenance"] = provenance
    _write_json(args.out / "model.json", bundle["model"])
    _write_json(args.out / "normalized-architecture.json", provenance["normalized_architecture"])
    model = bundle["model"]
    return {"status": "COMPLETED", "kind": bundle["kind"], "model_file": "model.json",
            "architecture_file": "normalized-architecture.json", "provenance": provenance,
            "dimensions": {"hidden_size": model["hidden_size"],
                           "query_projection_size": model["num_attention_heads"] * model["head_dim"],
                           "kv_projection_size": model["num_kv_heads"] * model["head_dim"]},
            "hardware_validated": False}


def _execute(args: argparse.Namespace) -> dict:
    _fresh_output(args.out)
    manifest = {"schema_version": 1, "simulator_version": __version__,
                "source_sha256": _source_hash(), "mode": args.mode,
                "backend": "simulated" if args.mode in ("simulate", "predict") else "tool",
                "hardware_validated": False, "input_files": []}
    try:
        if args.mode in ("simulate", "predict"):
            result = _simulation(args, manifest)
        elif args.mode == "calibrate":
            result = _calibrate(args, manifest)
        elif args.mode == "validate":
            result = _validate(args, manifest)
        elif args.mode == "collect":
            result = _collection(args, manifest)
        elif args.mode == "validate-hardware":
            result = _hardware_validation(args, manifest)
        elif args.mode == "import-model":
            result = _model_import(args, manifest)
        else:
            result = _compare(args, manifest)
        result.setdefault("status", "COMPLETED")
        result.setdefault("hardware_validated", False)
        _write_json(args.out / "manifest.json", manifest)
        _write_json(args.out / "result.json", result)
        message = ("Collection adapter execution; raw evidence and measurement boundaries are preserved.\n"
                   if args.mode == "collect" else
                   "Explicit scoped hardware-validation criteria evaluated; inspect verdict and evidence.\n"
                   if args.mode == "validate-hardware" else
                   "CPU tool execution; hardware validation is not inferred from supplied records.\n")
        (args.out / "logs.txt").write_text(message, encoding="utf-8")
        return result
    except (SimulationError, OSError) as exc:
        code = exc.code if isinstance(exc, SimulationError) else "IO_ERROR"
        result = {"status": "FAILED", "error_code": code, "message": str(exc)}
        if args.mode in ("collect", "validate-hardware"):
            result["hardware_validated"] = False
            manifest["hardware_validated"] = False
        _write_json(args.out / "manifest.json", manifest)
        _write_json(args.out / "result.json", result)
        (args.out / "logs.txt").write_text(f"{code}: {exc}\n", encoding="utf-8")
        return result


def execute(mode: str, experiment_path: Path, output_path: Path) -> dict:
    """Compatibility entry point for source-checkout simulate/predict commands."""
    if mode not in ("simulate", "predict"):
        raise SimulationError("INVALID_ARGUMENT", "execute expects simulate or predict")
    return _execute(argparse.Namespace(mode=mode, experiment=experiment_path, out=output_path))


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise SimulationError("INVALID_ARGUMENT", message)


def _arguments(parser: argparse.ArgumentParser, mode: str) -> None:
    parser.add_argument("--out", type=Path, required=True)
    if mode in ("simulate", "predict"):
        parser.add_argument("--experiment", type=Path, required=True)
        return
    if mode == "import-model":
        parser.add_argument("--kind", choices=("qwen3",), required=True)
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--name", required=True)
        parser.add_argument("--revision", required=True)
        parser.add_argument("--kv-dtype", choices=("float16", "bfloat16", "float32", "float64"))
        return
    parser.add_argument("--allow-test-fixture", action="store_true")
    if mode == "collect":
        parser.add_argument("--kind", choices=("probe", "steps", "rccl", "pp-communication", "runtime-trace", "vllm", "vllm-hook", "reference"),
                            required=True)
        for name in ("experiment", "plan", "raw", "reference-dir", "inventory", "rccl-library"):
            parser.add_argument(f"--{name}", type=Path)
        parser.add_argument("--adapter", help="Explicit Runtime factory as module:callable")
        parser.add_argument("--rocm-root", type=Path, help="ROCm installation root (default: /opt/rocm)")
        return
    if mode == "validate-hardware":
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--require-pass", action="store_true",
                            help="Exit 3 unless scoped measured hardware acceptance passes")
        return
    if mode in ("calibrate", "validate"):
        parser.add_argument("--dataset", type=Path, required=True)
        if mode == "calibrate":
            parser.add_argument("--profile-version", required=True)
            parser.add_argument("--parent-profile", type=Path)
            communication = parser.add_mutually_exclusive_group()
            communication.add_argument("--communication", action="store_true")
            communication.add_argument("--native-pp-communication", action="store_true")
        else:
            profile = parser.add_mutually_exclusive_group(required=True)
            profile.add_argument("--profile", type=Path)
            profile.add_argument("--profile-version")
            parser.add_argument("--holdout-group", action="append")
            parser.add_argument("--holdout-fraction", type=float, help="Component fraction (default: 0.25)")
            parser.add_argument("--seed", type=int, help="Deterministic split seed (default: 42)")
            parser.add_argument("--allow-extrapolation", action="store_true")
            parser.add_argument("--native-pp-communication", action="store_true")
        return
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predicted", type=Path)
    source.add_argument("--pairs", type=Path)
    parser.add_argument("--observed", type=Path)
    from .comparison import METRIC_UNITS
    parser.add_argument("--metric", choices=[name for name in METRIC_UNITS
                                             if name not in ("sla_satisfied", "request_count")],
                        default="goodput_requests_per_s")
    parser.add_argument("--lower-is-better", action="store_true")


def _requested_output(arguments: list[str]) -> Path | None:
    for index, value in enumerate(arguments):
        if value.startswith("--out=") and value[6:]:
            return Path(value[6:])
        if value == "--out" and index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
            return Path(arguments[index + 1])
    return None


def main(argv: list[str] | None = None, *, mode: str | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    output = _requested_output(arguments)
    parsed = False
    try:
        parser = _Parser(description="CPU LLM simulation, calibration and validation tools")
        modes = ("simulate", "predict", "calibrate", "compare", "validate", "collect", "validate-hardware", "import-model")
        if mode is None:
            commands = parser.add_subparsers(dest="mode", required=True, parser_class=_Parser)
            for name in modes:
                _arguments(commands.add_parser(name), name)
        else:
            if mode not in modes:
                raise SimulationError("INVALID_ARGUMENT", f"Unknown mode {mode}")
            parser.set_defaults(mode=mode)
            _arguments(parser, mode)
        args = parser.parse_args(arguments)
        output, parsed = args.out, True
        result = _execute(args)
    except (SimulationError, OSError) as exc:
        result = {"status": "FAILED", "error_code": getattr(exc, "code", "IO_ERROR"), "message": str(exc)}
        if not parsed and output is not None:
            try:
                _fresh_output(output)
                _write_json(output / "manifest.json", {"schema_version": 1, "mode": mode,
                                                        "simulator_version": __version__})
                _write_json(output / "result.json", result)
                (output / "logs.txt").write_text(f"{result['error_code']}: {exc}\n", encoding="utf-8")
            except (SimulationError, OSError) as output_error:
                result = {"status": "FAILED", "error_code": getattr(output_error, "code", "IO_ERROR"),
                          "message": str(output_error)}
    summary = {"status": result.get("status"),
                      "result_path": str((output / "result.json").resolve()) if output is not None else None,
                      "error_code": result.get("error_code"),
                      "next_actions": [] if result.get("status") == "COMPLETED"
                      else ["Inspect result.json and logs.txt; use a fresh output directory."]}
    acceptance_failed = False
    if parsed and args.mode == "validate-hardware":
        summary.update(verdict=result.get("verdict"), hardware_validated=result.get("hardware_validated", False))
        acceptance_failed = args.require_pass and not (result.get("verdict") == "PASS"
                                                       and result.get("hardware_validated") is True)
        if acceptance_failed and result.get("status") == "COMPLETED":
            summary["next_actions"] = ["Inspect acceptance.json; fixture or insufficient evidence is not a measured PASS."]
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    if acceptance_failed and result.get("status") == "COMPLETED":
        return 3
    return 0 if result.get("status") == "COMPLETED" else (3 if result.get("status") == "UNRESOLVED" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
