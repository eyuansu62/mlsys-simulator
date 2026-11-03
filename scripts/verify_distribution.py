"""Verify distribution contents and an installed wheel outside the checkout."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import venv
import zipfile


SMOKE_CODE = '''
from dataclasses import asdict
from importlib.metadata import metadata, version
import json
from pathlib import Path
import sys
import mlsys_sim
from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec,
                       MemoryConfig, RequestSpec, StrategySpec,
                       predict_experiment, simulate, profile_identity, comparison_contract)

assert Path(mlsys_sim.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert version("mlsys-simulator") == mlsys_sim.__version__
assert not metadata("mlsys-simulator").get_all("Requires-Dist")
exp = Experiment(
    workload=tuple(RequestSpec(name, 0.0, 16, 24) for name in ("a", "b")),
    strategy=StrategySpec(max_batch_size=2, memory=MemoryConfig(kv_pool_bytes=24576)),
    calibration=CalibrationSpec(backend="synthetic", prefill_ms=5,
                                decode_ms=1, recompute_ms=5))
simulation = simulate(exp)
assert simulation.result["counts"]["completed"] == 2
assert simulation.result["preemption_count"] == 1
assert [row["output_tokens_actual"] for row in simulation.requests] == [24, 24]
Path("experiment.json").write_text(json.dumps(asdict(exp)))
pp = Experiment(workload=(RequestSpec("shape", 0.0, 16, 2),),
                infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))),
                strategy=StrategySpec(pipeline_parallel_size=2))
assert predict_experiment(pp)["online_metrics_reason"] == "PP_ONLINE_DES_NOT_IMPLEMENTED"
Path("pp.json").write_text(json.dumps(asdict(pp)))
profile_base = Experiment(workload=(RequestSpec("profile-demo", 0, 16, 3),),
                          strategy=StrategySpec(scheduler="fifo"))
records = []
for name, phase, query, context, duration in [
    ("low", "prefill", 8, 0, 4), ("middle", "prefill", 16, 0, 6.2),
    ("high", "prefill", 24, 0, 8), ("decode-a", "decode", 1, 16, 1),
    ("decode-b", "decode", 1, 17, 1.2)]:
    records.append({"record_id": name, "group_id": name, "phase": phase,
                    "stage_index": 0, "query_tokens": [query], "context_lengths": [context],
                    "duration_ms": duration, "workspace_peak_bytes": None})
data = {"schema_version": 1, "dataset_id": "installed-wheel-fixture",
        "identity": profile_identity(profile_base),
        "source": {"kind": "test_fixture", "reference": "Hand calculations, not hardware."},
        "records": records}
Path("measurements.json").write_text(json.dumps(data))
profile_config = asdict(profile_base)
profile_config["calibration"].update(backend="profile", profile_path="fit/profile.json",
                                       profile_version="smoke-v1", allow_test_fixture=True)
Path("profile-experiment.json").write_text(json.dumps(profile_config))
observation = {"schema_version": 1, "kind": "measured_run",
               "comparison_contract": comparison_contract(profile_base),
               "source": {"kind": "test_fixture", "reference": "Hand calculations, not hardware."},
               "metrics": {"ttft_p50_ms": 6.2, "tpot_p50_ms": 1.1,
                           "completion_rate": 1, "request_count": 1}}
Path("observation.json").write_text(json.dumps(observation))
communication = {"schema_version": 1, "kind": "communication_measurements",
                 "dataset_id": "installed-communication-fixture", "identity": profile_identity(pp),
                 "source": {"kind": "test_fixture", "reference": "Hand calculations, not RCCL measurements."},
                 "records": [{"record_id": str(size), "group_id": "p2p-two-ranks",
                              "primitive": "p2p", "ranks": 2, "message_bytes": size,
                              "duration_ms": .005 + size / 20e6} for size in [1000, 1000000]]}
Path("communication.json").write_text(json.dumps(communication))
print(json.dumps({"version": mlsys_sim.__version__, "wheel_api": "passed"}))
'''


def run(command: list[str], cwd: Path, env: dict[str, str], expected: int = 0) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if result.returncode != expected:
        raise RuntimeError(f"{command[0]} exited {result.returncode}, expected {expected}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


def verify(dist: Path) -> dict:
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected one wheel and one source distribution in a clean dist directory.")
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "mlsys_sim/py.typed" in names, "Wheel is missing its typing marker"
        assert "mlsys_sim/__init__.py" in names
        assert not any(name.startswith(("tests/", "sample-runs/", "work/")) for name in names)
        assert archive.testzip() is None
    with tarfile.open(sdists[0]) as archive:
        names = {"/".join(member.name.split("/")[1:]) for member in archive.getmembers()}
        for required in ("pyproject.toml", "README.md", "README.zh-CN.md",
                         "examples/memory_pressure.json", "tests/test_public_api.py",
                         "examples/v1/step_measurements.fixture.json", "calibrate.py", "validate.py",
                         "scripts/verify_distribution.py"):
            assert required in names, f"Source distribution missing {required}"
        assert not any(name.startswith(("sample-runs/", "work/")) for name in names)

    with tempfile.TemporaryDirectory(prefix="mlsys-installed-") as temporary:
        outside = Path(temporary)
        environment = outside / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        binaries = environment / ("Scripts" if os.name == "nt" else "bin")
        python = binaries / ("python.exe" if os.name == "nt" else "python")
        console = binaries / ("mlsys-sim.exe" if os.name == "nt" else "mlsys-sim")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        run([str(python), "-I", "-m", "pip", "install", "--no-deps", str(wheel)], outside, env)
        run([str(python), "-I", "-c", SMOKE_CODE], outside, env)
        simulated = json.loads(run([str(console), "simulate", "--experiment", "experiment.json",
                                   "--out", "simulation"], outside, env))
        assert simulated["status"] == "COMPLETED"
        prediction = json.loads(run([str(console), "predict", "--experiment", "pp.json",
                                    "--out", "prediction"], outside, env))
        assert prediction["status"] == "COMPLETED"
        failed = json.loads(run([str(console), "simulate", "--experiment", "pp.json",
                                "--out", "unsupported"], outside, env, expected=2))
        assert failed["error_code"] == "UNSUPPORTED_MECHANISM"
        run([str(console), "calibrate", "--dataset", "measurements.json", "--profile-version",
             "smoke-v1", "--out", "fit", "--allow-test-fixture"], outside, env)
        run([str(console), "simulate", "--experiment", "profile-experiment.json",
             "--out", "profile-run"], outside, env)
        comparison = json.loads(run([str(console), "compare", "--predicted", "profile-run",
                                     "--observed", "observation.json", "--out", "comparison",
                                     "--allow-test-fixture"], outside, env))
        assert comparison["status"] == "COMPLETED"
        run([str(console), "validate", "--dataset", "measurements.json", "--profile-version",
             "heldout-v1", "--holdout-group", "middle", "--out", "heldout",
             "--allow-test-fixture"], outside, env)
        report = json.loads((outside / "heldout/result.json").read_text())
        assert report["summary"]["coverage"] == 1
        assert report["test_fixture_only"] is True
        assert report["hardware_validation"]["status"] == "not_asserted"
        run([str(console), "calibrate", "--communication", "--dataset", "communication.json",
             "--profile-version", "communication-v1", "--out", "communication-fit",
             "--allow-test-fixture"], outside, env)
        fit = json.loads((outside / "communication-fit/profile.json").read_text())
        assert fit["application"]["automatic_application"] is False
        assert 19.9 < fit["groups"][0]["infra_patch"]["link_bandwidth_gbps"] < 20.1
        run([str(python), "-I", "-m", "mlsys_sim", "--help"], outside, env)
    return {"wheel": wheel.name, "sdist": sdists[0].name,
            "distribution_contents": "passed", "installed_api_and_cli": "passed",
            "installed_calibration_comparison_holdout": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path(__file__).resolve().parents[1] / "dist")
    args = parser.parse_args()
    print(json.dumps(verify(args.dist), sort_keys=True))


if __name__ == "__main__":
    main()
