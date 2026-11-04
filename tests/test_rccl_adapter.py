"""Strict RCCL table semantics and bounded CPU-only collector injection."""

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, ModelSpec,
                       SimulationError, StrategySpec, calibrate_communication)
from mlsys_sim.hardware import CollectionError
from mlsys_sim.rccl_adapter import _run_bounded, collect_rccl, parse_rccl_output


def output(size=1024, *, modes="both", unit="us", out_time="10", in_time="8", out_wrong="0", in_wrong="0", datatype="float"):
    placements = "out-of-place                       in-place" if modes == "both" else modes
    columns = "time algbw busbw #wrong"
    header = f"# size count type redop root {columns}"
    units = f"# (B) (elements) ( {unit} ) (GB/s) (GB/s)".replace(f"( {unit} )", f"({unit})")
    width = {"float": 4, "half": 2, "bfloat16": 2, "double": 8}[datatype]
    row = f"{size} {size // width} {datatype} sum -1 {out_time} 100 50 {out_wrong}"
    if modes == "both":
        header += " time algbw busbw #wrong"
        units += f" ({unit}) (GB/s) (GB/s)"
        row += f" {in_time} 200 100 {in_wrong}"
    elif modes == "in-place":
        row = f"{size} {size // width} {datatype} sum -1 {in_time} 200 100 {in_wrong}"
    return "\n".join([
        "# Collective test starting: all_reduce_perf",
        f"# nThread 1 nGpus 2 minBytes {size} maxBytes {size} step: 1(bytes) warmup iters: 1 iters: 2 agg iters: 1 validation: 1 graph: 0",
        "rccl-tests: Version fixture-git-hash",
        "# RCCL version 2.30.7",
        "#  Rank 0 Group 0 Pid 123 on fixture device 0 [0000:01:00] AMD Fixture 0",
        "#  Rank 1 Group 0 Pid 123 on fixture device 1 [0000:02:00] AMD Fixture 1",
        f"# {placements}", header, units, row,
        "# Out of bounds values : 0 OK",
        "# Avg bus bandwidth : 50",
        "# Collective test concluded: all_reduce_perf", "",
    ])


def experiment(*, measured=False):
    model = ModelSpec(name="declared-dense", revision="revision-a", weight_dtype="float16",
                      activation_dtype="float16", kv_dtype="float16") if measured else ModelSpec()
    calibration = CalibrationSpec(runtime_name="declared-runtime", runtime_version="1.0",
        driver_version="fixture-driver", rocm_version="10.1.0", rccl_version="2.30.7") if measured else CalibrationSpec()
    return Experiment(infra=InfraSpec(devices=(DeviceSpec("a", name="AMD Fixture 0"),
                                             DeviceSpec("b", name="AMD Fixture 1"))),
                      strategy=StrategySpec(tensor_parallel_size=2, model=model), calibration=calibration)


def inventory(*, measured=False):
    return {"schema_version": 1, "kind": "rocm_inventory", "status": "READY",
        "source": {"kind": "measured" if measured else "test_fixture", "reference": "CPU test inventory"},
        "torch": {"version": "fixture-torch", "hip_version": "fixture-hip", "cuda_available": True},
        "versions": {"driver_version": "fixture-driver", "rocm_version": "10.1.0", "rccl_version": "2.30.7"},
        "devices": [{"index": index, "name": f"AMD Fixture {index}", "memory_bytes": 64 * 1024 * 1024}
                    for index in range(2)]}


def plan():
    return {"schema_version": 1, "dataset_id": "rccl-fixture", "source_reference": "CPU runner fixture only",
            "ranks": 2, "mode": "out_of_place", "binary": "all_reduce_perf",
            "message_bytes": [1024, 2048], "iterations": 2, "warmup_iterations": 1, "repeats": 2}


def runner(argv, **kwargs):
    size = int(argv[argv.index("-b") + 1])
    datatype = argv[argv.index("-d") + 1]
    return {"returncode": 0, "stdout": output(size, modes="out-of-place", out_time=str(size / 102.4), datatype=datatype),
            "stderr": "", "process_elapsed_ms": 999999}


class RCCLParserTests(unittest.TestCase):
    def test_selects_real_time_column_and_converts_explicit_units(self):
        for unit, factor in (("us", .001), ("ms", 1), ("ns", .000001), ("s", 1000)):
            with self.subTest(unit=unit):
                parsed = parse_rccl_output(output(unit=unit), ranks=2)
                self.assertAlmostEqual(parsed["records"][0]["duration_ms"], 10 * factor)
                self.assertEqual(parsed["records"][0]["algbw_gbps"], 100)
                self.assertEqual(parsed["records"][0]["busbw_gbps"], 50)
                inplace = parse_rccl_output(output(unit=unit), ranks=2, mode="in_place")
                self.assertAlmostEqual(inplace["records"][0]["duration_ms"], 8 * factor)
        self.assertFalse(parsed["metadata"]["rank_aggregation_observed_in_table"])

    def test_single_placement_and_one_byte_unit_header_are_supported(self):
        parsed = parse_rccl_output(output(modes="in-place"), ranks=2, mode="in_place")
        self.assertEqual(parsed["records"][0]["duration_ms"], .008)
        with self.assertRaises(SimulationError):
            parse_rccl_output(output(modes="in-place"), ranks=2)

    def test_rejects_wrong_counts_unchecked_data_and_failed_footer(self):
        malformed = [output(out_wrong="1"), output(in_wrong="1"), output(out_wrong="N/A"),
                     output().replace("0 OK", "1 FAILED"), output().replace("validation: 1", "validation: 0"),
                     output().replace("graph: 0", "graph: 1")]
        for value in malformed:
            with self.subTest(value=value[-140:]), self.assertRaises(SimulationError):
                parse_rccl_output(value, ranks=2)

    def test_rejects_nonfinite_time_bandwidth_and_impossible_payload(self):
        malformed = [output(out_time="nan"), output(out_time="0"), output(out_time="inf"),
                     output().replace("10 100 50 0", "10 nan 50 0"),
                     output().replace("1024 256 float", "1024 255 float"),
                     output().replace("1024 256 float", "1024 256 unknown")]
        for value in malformed:
            with self.subTest(value=value[-180:]), self.assertRaises(SimulationError):
                parse_rccl_output(value, ranks=2)

    def test_rejects_missing_units_cpu_time_and_truncated_output(self):
        malformed = [output().replace("(us)", "(ticks)"), output().replace("time algbw", "cputime algbw"),
                     output().replace("# Out of bounds values : 0 OK", ""),
                     output().replace("# Collective test concluded: all_reduce_perf", ""),
                     output().replace("# out-of-place                       in-place", "# unknown-placement")]
        for value in malformed:
            with self.subTest(value=value[-200:]), self.assertRaises(SimulationError):
                parse_rccl_output(value, ranks=2)

    def test_rejects_split_mpi_or_incorrect_rank_device_evidence(self):
        malformed = [output().replace("Rank 1 Group 0", "Rank 1 Group 1"),
                     output().replace("Rank 1 Group 0 Pid 123", "Rank 1 Group 0 Pid 124"),
                     output().replace("device 1", "device 0"),
                     output().replace("nThread 1 nGpus 2", "nThread 2 nGpus 1"),
                     output().replace("#  Rank 1 Group 0 Pid 123 on fixture device 1 [0000:02:00] AMD Fixture 1", "")]
        for value in malformed:
            with self.subTest(value=value[:140]), self.assertRaises(SimulationError):
                parse_rccl_output(value, ranks=2)
        with self.assertRaises(SimulationError):
            parse_rccl_output(output(), ranks=3)

    def test_sendrecv_is_never_mapped_to_unidirectional_p2p(self):
        with self.assertRaises(SimulationError) as raised:
            parse_rccl_output(output().replace("all_reduce_perf", "sendrecv_perf"), ranks=2)
        self.assertIn("not isolated", str(raised.exception))

    def test_malformed_mode_and_rank_types_raise_contract_errors(self):
        for kwargs in ({"ranks": True}, {"ranks": 2, "mode": []}, {"ranks": 2, "mode": "p2p"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(SimulationError):
                parse_rccl_output(output(), **kwargs)


class RCCLCollectionTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_injected_runner_is_fixture_and_fixed_command_preserves_evidence(self):
        commands = []
        original = inventory(measured=True)
        before = deepcopy(original)
        def capture(argv, **kwargs):
            commands.append((argv, kwargs))
            return runner(argv, **kwargs)
        result = collect_rccl(experiment(), plan(), original, runner=capture, allow_test_fixture=True)
        self.assertEqual(original, before)
        self.assertEqual(result["dataset"]["source"]["kind"], "test_fixture")
        self.assertFalse(result["hardware_validated"])
        self.assertEqual([row["duration_ms"] for row in result["dataset"]["records"]], [.01, .02, .01, .02])
        self.assertEqual(len(commands), 4)
        for argv, kwargs in commands:
            for flag, expected in (("-a", "3"), ("-c", "1"), ("-C", "0"), ("-t", "1"), ("-g", "2"), ("-O", "1")):
                self.assertEqual(argv[argv.index(flag) + 1], expected)
            self.assertEqual(kwargs["env"]["NCCL_ALGO"], "Ring")
            self.assertEqual(kwargs["env"]["NCCL_DEBUG"], "VERSION")
        profile = calibrate_communication(result["dataset"], profile_version="comm-fixture", allow_test_fixture=True)
        self.assertEqual(profile["source"]["kind"], "test_fixture")

    def test_runner_injection_requires_fixture_permission(self):
        with self.assertRaises(SimulationError) as raised:
            collect_rccl(experiment(), plan(), inventory(measured=True), runner=runner)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")

    def test_raw_import_requires_declared_launch_facts_and_fixture_permission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw.txt").write_text(output())
            raw = {"schema_version": 1, "dataset_id": "import", "ranks": 2, "mode": "in_place",
                "raw_path": "raw.txt", "source": {"kind": "test_fixture", "reference": "Hand-written log"},
                "algorithm": "Ring", "rank_aggregation": "max"}
            with self.assertRaises(SimulationError):
                collect_rccl(experiment(), raw, base_dir=root)
            result = collect_rccl(experiment(), raw, base_dir=root, allow_test_fixture=True)
            self.assertEqual(result["dataset"]["records"][0]["duration_ms"], .008)
            self.assertEqual(len(result["raw_records"][0]["sha256"]), 64)
            for key in ("algorithm", "rank_aggregation"):
                missing = {name: value for name, value in raw.items() if name != key}
                with self.subTest(key=key), self.assertRaises(SimulationError):
                    collect_rccl(experiment(), missing, base_dir=root, allow_test_fixture=True)

    def test_declared_measured_import_preserves_observed_benchmark_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw.txt").write_text(output(datatype="half"))
            raw = {"schema_version": 1, "dataset_id": "declared-import", "ranks": 2, "mode": "out_of_place",
                "raw_path": "raw.txt", "source": {"kind": "measured", "reference": "Declared external capture (unit validation)"},
                "algorithm": "Ring", "rank_aggregation": "max"}
            result = collect_rccl(experiment(measured=True), raw, base_dir=root)
            self.assertEqual(result["source"]["kind"], "measured")
            self.assertEqual(result["dataset"]["identity"]["runtime"]["name"], "rccl-tests")
            self.assertEqual(result["dataset"]["identity"]["runtime"]["version"], "fixture-git-hash")
            self.assertFalse(result["hardware_validated"])
            (root / "raw.txt").write_text(output(datatype="half").replace("# RCCL version 2.30.7", ""))
            with self.assertRaises(CollectionError) as raised:
                collect_rccl(experiment(measured=True), raw, base_dir=root)
            self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")

    def test_live_datatype_follows_explicit_precision_and_checks_returned_rows(self):
        for dtype, rccl_type, width in (("float16", "half", 2), ("bfloat16", "bfloat16", 2),
                                        ("float32", "float", 4), ("float64", "double", 8)):
            base = experiment()
            exp = replace(base, strategy=replace(base.strategy,
                model=replace(base.strategy.model, activation_dtype=dtype, activation_bytes=width)))
            commands = []
            def capture(argv, **kwargs):
                commands.append(argv)
                return runner(argv, **kwargs)
            result = collect_rccl(exp, plan(), inventory(), runner=capture, allow_test_fixture=True)
            self.assertTrue(all(argv[argv.index("-d") + 1] == rccl_type for argv in commands))
            self.assertEqual(result["dataset"]["identity"]["model"]["activation_dtype"], dtype)
            self.assertEqual(result["communication_datatype"], {
                "activation_dtype": dtype, "rccl_datatype": rccl_type, "binding": "explicit_activation_dtype"})
            self.assertEqual(result["capture_metadata"][0]["communication_datatype"]["rccl_datatype"], rccl_type)
        base = experiment()
        bf16 = replace(base, strategy=replace(base.strategy, model=replace(base.strategy.model,
            activation_dtype="bfloat16", activation_bytes=2)))
        with self.assertRaises(CollectionError) as raised:
            collect_rccl(bf16, plan(), inventory(),
                runner=lambda *args, **kwargs: {"returncode": 0, "stdout": output(modes="out-of-place"), "stderr": ""},
                allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "HARDWARE_IDENTITY_MISMATCH")

    def test_measured_raw_import_rejects_a_different_communication_dtype(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = {"schema_version": 1, "dataset_id": "precision-import", "ranks": 2, "mode": "out_of_place",
                "raw_path": "raw.txt", "source": {"kind": "measured", "reference": "Declared precision validation only"},
                "algorithm": "Ring", "rank_aggregation": "max"}
            (root / "raw.txt").write_text(output(datatype="float"))
            with self.assertRaises(CollectionError) as raised:
                collect_rccl(experiment(measured=True), raw, base_dir=root)
            self.assertEqual(raised.exception.code, "HARDWARE_IDENTITY_MISMATCH")
            (root / "raw.txt").write_text(output(datatype="half"))
            result = collect_rccl(experiment(measured=True), raw, base_dir=root)
            self.assertEqual(result["dataset"]["identity"]["model"]["activation_dtype"], "float16")
            self.assertEqual(result["communication_datatype"]["rccl_datatype"], "half")

    def test_process_failures_timeout_and_partial_capture_are_not_datasets(self):
        for capture, code in [({"returncode": 1, "stdout": "partial", "stderr": ""}, "COLLECTION_FAILED"),
                              ({"returncode": -9, "stdout": "partial", "timed_out": True}, "COLLECTION_TIMEOUT"),
                              ({"returncode": 0, "stdout": output(), "output_truncated": True}, "COLLECTION_OUTPUT_LIMIT"),
                              ({"returncode": 0, "stdout": output().replace("0 OK", "1 FAILED")}, "INVALID_RCCL_OUTPUT")]:
            with self.subTest(code=code), self.assertRaises(CollectionError) as raised:
                collect_rccl(experiment(), plan(), inventory(), runner=lambda *args, **kwargs: capture,
                             allow_test_fixture=True)
            self.assertEqual(raised.exception.code, code)
            self.assertTrue(raised.exception.artifacts["raw_records"])
            self.assertNotIn("dataset", raised.exception.artifacts)

    def test_timeout_exception_preserves_the_attempted_command_and_partial_output(self):
        def timeout(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 1, output=b"partial stdout", stderr=b"partial stderr")
        with self.assertRaises(CollectionError) as raised:
            collect_rccl(experiment(), plan(), inventory(), runner=timeout, allow_test_fixture=True)
        raw = raised.exception.artifacts["raw_records"][0]
        self.assertEqual(raw["stdout"], "partial stdout")
        self.assertTrue(raw["argv"])

    def test_library_banner_device_and_requested_size_must_match(self):
        for transform, code in [(lambda text: text.replace("2.30.7", "2.30.6"), "HARDWARE_IDENTITY_MISMATCH"),
                                (lambda text: text.replace("# RCCL version 2.30.7", ""), "HARDWARE_METADATA_INCOMPLETE"),
                                (lambda text: text.replace("AMD Fixture 1", "other device"), "HARDWARE_IDENTITY_MISMATCH"),
                                (lambda text: text.replace("1024 256", "2048 512"), "INVALID_RCCL_OUTPUT")]:
            one_size = {**plan(), "message_bytes": [1024], "repeats": 1}
            with self.subTest(code=code), self.assertRaises(CollectionError) as raised:
                collect_rccl(experiment(), one_size, inventory(),
                    runner=lambda *args, **kwargs: {"returncode": 0, "stdout": transform(output(modes="out-of-place")), "stderr": ""},
                    allow_test_fixture=True)
            self.assertEqual(raised.exception.code, code)

    def test_unknown_controls_invalid_sizes_and_split_groups_are_rejected(self):
        invalid = [{**plan(), "primitive": "p2p"}, {**plan(), "binary": "sendrecv_perf"},
                   {**plan(), "message_bytes": [[]]}, {**plan(), "message_bytes": [True]},
                   {**plan(), "message_bytes": [3]}, {**plan(), "message_bytes": [1024, 1024]},
                   {**plan(), "mode": []}, {**plan(), "extra": 1}, {**plan(), "timeout_s": float("nan")}]
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(SimulationError):
                collect_rccl(experiment(), candidate, inventory(), runner=runner, allow_test_fixture=True)
        for variable, value in (("NCCL_TESTS_SPLIT", "MOD 2"), ("WORLD_SIZE", "2"), ("NCCL_TESTS_DEVICE", "1")):
            with self.subTest(variable=variable), patch.dict(os.environ, {variable: value}), self.assertRaises(SimulationError):
                collect_rccl(experiment(), plan(), inventory(), runner=runner, allow_test_fixture=True)

    def test_real_cpu_process_capture_is_bounded_and_shell_free(self):
        literal = "literal; not a shell command"
        with patch("mlsys_sim.rccl_adapter.subprocess.Popen", wraps=subprocess.Popen) as spawn:
            result = _run_bounded([sys.executable, "-c", "import sys; print(sys.argv[1])", literal],
                                  timeout_s=2, max_output_bytes=1024, env={})
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"].strip(), literal)
        self.assertIs(spawn.call_args.kwargs["shell"], False)
        limited = _run_bounded([sys.executable, "-c", "print('x' * 100000)"],
                               timeout_s=2, max_output_bytes=128, env={})
        self.assertTrue(limited["output_truncated"])
        self.assertLessEqual(len(limited["stdout"].encode()) + len(limited["stderr"].encode()), 128)
        timed_out = _run_bounded([sys.executable, "-c", "import time; print('partial', flush=True); time.sleep(1)"],
                                 timeout_s=.3, max_output_bytes=1024, env={})
        self.assertTrue(timed_out["timed_out"])
        self.assertIn("partial", timed_out["stdout"])


if __name__ == "__main__":
    unittest.main()
