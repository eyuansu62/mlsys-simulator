"""CLI and immutable, machine-readable experiment output."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from . import __version__
from .config import load_experiment
from .schema import SimulationError


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
                    + "\n", encoding="utf-8")


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def execute(mode: str, experiment_path: Path, output_path: Path) -> dict:
    """Create a fresh run directory; preserve diagnostics on configuration errors."""
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite {output_path}") from exc
    manifest = {
        "schema_version": 1, "simulator_version": __version__,
        "source_sha256": _source_hash(), "mode": mode,
        "backend": "simulated", "experiment_path": str(experiment_path.resolve()),
    }
    try:
        experiment = load_experiment(experiment_path)
        expanded = asdict(experiment)
        canonical = json.dumps(expanded, sort_keys=True, separators=(",", ":"), allow_nan=False)
        manifest.update({
            "effective_config": expanded,
            "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "seed": experiment.execution.seed,
            "calibration_source": experiment.calibration.backend,
            "profile_version": experiment.calibration.profile_version,
            "hardware_validated": False,
        })
        _write_json(output_path / "manifest.json", manifest)
        if mode == "simulate":
            from .engine import simulate
            simulation = simulate(experiment)
            result, requests, events = simulation.result, simulation.requests, simulation.events
        else:
            from .performance import predict_experiment
            result = predict_experiment(experiment)
            result.setdefault("status", "COMPLETED")
            result.setdefault("backend", "simulated")
            requests, events = [], []
        _write_json(output_path / "result.json", result)
        for name, rows in (("requests.jsonl", requests), ("events.jsonl", events)):
            if name == "events.jsonl" and not experiment.execution.detailed_events:
                continue
            with (output_path / name).open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        (output_path / "logs.txt").write_text(
            "CPU simulation; timing profiles are not hardware validated.\n", encoding="utf-8")
        return result
    except (SimulationError, OSError) as exc:
        code = exc.code if isinstance(exc, SimulationError) else "IO_ERROR"
        result = {"status": "FAILED", "error_code": code, "message": str(exc)}
        _write_json(output_path / "manifest.json", manifest)
        _write_json(output_path / "result.json", result)
        (output_path / "logs.txt").write_text(f"{code}: {exc}\n", encoding="utf-8")
        return result


def main(argv: list[str] | None = None, *, mode: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic CPU LLM serving simulator")
    if mode is None:
        parser.add_argument("mode", choices=("simulate", "predict"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    selected_mode = mode or args.mode
    try:
        result = execute(selected_mode, args.experiment, args.out)
    except (SimulationError, OSError) as exc:
        result = {"status": "FAILED", "error_code": getattr(exc, "code", "IO_ERROR"),
                  "message": str(exc)}
    print(json.dumps({"status": result.get("status"),
                      "result_path": str((args.out / "result.json").resolve()),
                      "error_code": result.get("error_code"),
                      "next_actions": [] if result.get("status") == "COMPLETED"
                      else ["Inspect result.json and logs.txt; use a fresh output directory."]},
                     sort_keys=True, allow_nan=False))
    return 0 if result.get("status") == "COMPLETED" else (3 if result.get("status") == "UNRESOLVED" else 2)


if __name__ == "__main__":
    sys.exit(main())
