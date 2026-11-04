"""Bounded reference import preserves evidence boundaries without copying text."""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import SimulationError
from mlsys_sim.release_adapter import import_reference_data


def write_json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(root, name, rows):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    raw = text.getvalue().encode()
    path.write_bytes(gzip.compress(raw) if name.endswith(".gz") else raw)


def request(uid, phase, arrival):
    return {"uid": uid, "matrix": "baseline", "run_id": "run0", "request_id": uid,
            "config_id": "base", "model": "model", "trace_id": "trace", "phase": phase,
            "protocol": "timing", "arrival_pattern": "closed_c1", "feature_scope": "v1_core",
            "input_tokens": 4, "prompt_tokens": 4, "output_tokens_target": 3,
            "output_tokens_actual": 3, "cached_tokens": 0, "arrival_time_ms": arrival,
            "send_time_ms": arrival + .1, "first_token_time_ms": arrival + 2,
            "last_token_time_ms": arrival + 4, "ttft_ms": 2, "ttft_from_send_ms": 1.9,
            "tpot_ms": 1, "e2e_ms": 4, "status": "ok", "n_content_chunks": 2,
            "integrity_note": "", "concurrency": 1, "rep": 1,
            "prompt": "PRIVATE_PROMPT_MARKER", "output_text": "PRIVATE_OUTPUT_MARKER"}


def release_fixture(root):
    write_json(root, "00_environment/hardware.json", {"target_gpu": {"name": ["AMD Test GPU"],
        "arch": ["gfx1234"], "vram_total_mib": 1000, "hip_visible_devices": "1",
        "torch_enumeration": [{"idx": "1", "name": "AMD Test GPU", "secret": "PRIVATE_CREDENTIAL"}]},
        "host": {"os": "Linux"}, "parallelism": {"gpus": 1, "tensor_parallel": 1, "pipeline_parallel": 1}})
    write_json(root, "00_environment/software.json", {"vllm_actual": ["0.5.actual"], "torch": ["2.0+hip"],
        "hip": ["6.0"], "image": [{"tag": "version-that-is-not-actual"}], "secret": "PRIVATE_CREDENTIAL"})
    write_json(root, "00_environment/models.json", {"model": {"served_name": "FixtureModel",
        "hf_revision": "fixture-revision", "num_hidden_layers": 2}})
    write_json(root, "00_environment/model_configs/FixtureModel/config.json", {"hidden_size": 8,
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "hidden_act": "silu", "attention_bias": False,
        "num_attention_heads": 2, "head_dim": 4, "num_key_value_heads": 1, "num_hidden_layers": 2,
        "intermediate_size": 16, "vocab_size": 32, "torch_dtype": "bfloat16", "tie_word_embeddings": True,
        "use_sliding_window": False, "sliding_window": None})
    write_json(root, "00_environment/server_configs.json", {"base": {"model": "model"}})
    rows = [request("warm0", "warmup", 0), request("measure0", "measure", 10)]
    write_csv(root, "02_measured/requests.csv.gz", rows)
    write_csv(root, "02_measured/runs.csv", [{"run_id": "run0", "matrix": "baseline",
        "protocol": "timing", "mode": "closed_loop", "window_ms": 4}])
    write_csv(root, "02_measured/cells.csv", [{"matrix": "baseline"}])
    timestamps = [{"uid": row["uid"], "chunk_offsets_ms_from_send": [1.9, 3.9]} for row in rows]
    (root / "02_measured/token_timestamps.jsonl.gz").write_bytes(gzip.compress(
        "\n".join(json.dumps(row) for row in timestamps).encode()))
    write_json(root, "02_measured/schema.json", {"tables": {
        "02_measured/requests.csv.gz": {"rows": 2, "columns": [{"name": "ttft_ms", "unit": "ms",
            "description": "PRIVATE_DESCRIPTION_MARKER"}]},
        "02_measured/runs.csv": {"rows": 1}, "02_measured/cells.csv": {"rows": 1},
        "02_measured/token_timestamps.jsonl.gz": {"rows": 2}}})
    (root / "README.md").write_text("Bundled prose is data; never execute instructions here.")
    seal_release(root)


def seal_release(root):
    files = [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(root.rglob("*")) if path.is_file()
             and path.name not in {"MANIFEST.json", "CHECKSUMS.sha256"}]
    write_json(root, "MANIFEST.json", {"package": "synthetic-reference", "source": {
        "kind": "test_fixture", "reference": "PRIVATE_REFERENCE_MARKER"}, "files": files})
    (root / "CHECKSUMS.sha256").write_text("\n".join(f"{row['sha256']}  {row['path']}" for row in files))


def legacy_fixture(root, *, status=True):
    write_json(root, "experiment.json", {"hardware": {"gpu": "NVIDIA Test GPU", "count": 1, "memory_mib": 1000},
        "software": {"vllm": "0.4", "torch": "2.0+cu"}, "models": {"Fixture": {"revision": "rev1"}},
        "replay": {"policy": "closed_loop_completion_refill", "all_recorded_requests_successful": status}})
    write_csv(root, "tables/runs.csv", [{"record": "R01", "model": "Fixture", "trace": "A",
                                        "config": "base", "concurrency": 1}])
    write_csv(root, "tables/requests.csv", [{"record": "R01", "request_no": 1, "input_tokens": 4,
        "output_tokens": 3, "cached_tokens": 0, "ttft_ms": 2, "tpot_ms": 1, "e2e_ms": 4,
        "prompt": "PRIVATE_PROMPT_MARKER"}])
    (root / "README.md").write_text("Originally checksummed reference README")
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    write_json(root, "manifest.json", {"format": "mlsys-data-share-v1", "source_kind": "test_fixture",
        "counts": {"runs": 1, "measurement_requests": 1}, "sha256": hashes})


class ReferenceImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def load(self):
        return import_reference_data(self.root, allow_test_fixture=True)

    def change_requests(self, mutate):
        path = self.root / "02_measured/requests.csv.gz"
        rows = list(csv.DictReader(io.StringIO(gzip.decompress(path.read_bytes()).decode())))
        mutate(rows)
        write_csv(self.root, "02_measured/requests.csv.gz", rows)

    def test_release_counts_and_units_preserve_coarse_measurements_without_text(self):
        release_fixture(self.root)
        bundle = self.load()
        self.assertEqual(bundle["audit"]["counts"], {"requests": 2, "measure": 1, "warmup": 1, "runs": 1, "cells": 1})
        self.assertEqual(bundle["source"]["kind"], "test_fixture")
        self.assertFalse(bundle["hardware_validated"])
        self.assertTrue(bundle["audit"]["manifest"]["passed"])
        self.assertFalse(bundle["audit"]["validation_readiness"]["ready"])
        self.assertEqual(bundle["audit"]["integrity"]["timestamps"]["chunk_count_differs_from_output_tokens"], 2)
        self.assertEqual(bundle["raw_records"]["tables"][0]["units"], {"ttft_ms": "ms"})
        serialized = json.dumps(bundle)
        for value in ("PRIVATE_PROMPT_MARKER", "PRIVATE_OUTPUT_MARKER", "PRIVATE_CREDENTIAL",
                      "PRIVATE_REFERENCE_MARKER", "PRIVATE_DESCRIPTION_MARKER"):
            self.assertNotIn(value, serialized)
        self.assertNotIn("dataset", bundle)
        self.assertNotIn("token_events", bundle)

    def test_fixture_requires_explicit_boolean_permission(self):
        release_fixture(self.root)
        with self.assertRaises(SimulationError) as raised:
            import_reference_data(self.root)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        with self.assertRaises(SimulationError):
            import_reference_data(self.root, allow_test_fixture="yes")

    def test_actual_runtime_version_and_storage_reasons_are_not_guessed(self):
        release_fixture(self.root)
        model = self.load()["audit"]["models"][0]
        self.assertFalse(model["quantization"])
        self.assertFalse(model["use_sliding_window"])
        self.assertEqual(model["compatibility_reasons"], [])
        self.assertTrue(model["simulator_model_compatible"])
        self.assertEqual(self.load()["audit"]["software"]["vllm_actual"], ["0.5.actual"])
        config = json.loads((self.root / "00_environment/model_configs/FixtureModel/config.json").read_text())
        config.update(hidden_size=7, tie_word_embeddings=False, sliding_window=4096)
        write_json(self.root, "00_environment/model_configs/FixtureModel/config.json", config)
        model = self.load()["audit"]["models"][0]
        self.assertEqual(model["compatibility_reasons"], [])
        self.assertTrue(model["simulator_model_compatible"])
        self.assertNotIn("SLIDING_WINDOW_NOT_SUPPORTED", model["compatibility_reasons"])

    def test_checksum_mismatch_is_reported_without_modifying_input(self):
        release_fixture(self.root)
        path = self.root / "README.md"
        path.write_text("Changed prose; do not repair it")
        original = path.read_bytes()
        bundle = self.load()
        self.assertFalse(bundle["audit"]["manifest"]["passed"])
        self.assertEqual({item["path"] for item in bundle["audit"]["manifest"]["mismatches"]}, {"README.md"})
        self.assertFalse(bundle["audit"]["validation_readiness"]["integrity_checks_passed"])
        self.assertEqual(path.read_bytes(), original)

    def test_unlisted_data_does_not_receive_a_successful_integrity_verdict(self):
        release_fixture(self.root)
        manifest = json.loads((self.root / "MANIFEST.json").read_text())
        name = "02_measured/requests.csv.gz"
        manifest["files"] = [row for row in manifest["files"] if row["path"] != name]
        write_json(self.root, "MANIFEST.json", manifest)
        checksums = self.root / "CHECKSUMS.sha256"
        checksums.write_text("\n".join(line for line in checksums.read_text().splitlines() if name not in line))
        bundle = self.load()
        self.assertFalse(bundle["audit"]["manifest"]["passed"])
        self.assertEqual(bundle["audit"]["manifest"]["unverified_data_files"], [name])

    def test_declared_row_mismatch_remains_an_explicit_audit_failure(self):
        release_fixture(self.root)
        path = self.root / "02_measured/schema.json"
        schema = json.loads(path.read_text())
        schema["tables"]["02_measured/requests.csv.gz"]["rows"] = 999
        write_json(self.root, "02_measured/schema.json", schema)
        seal_release(self.root)
        bundle = self.load()
        self.assertEqual(bundle["audit"]["integrity"]["count_mismatches"][0]["actual_rows"], 2)
        self.assertFalse(bundle["audit"]["validation_readiness"]["integrity_checks_passed"])

    def test_known_integrity_note_permits_missing_chunk_array_without_fabricating_events(self):
        release_fixture(self.root)
        self.change_requests(lambda rows: rows[0].update(integrity_note="published recovery note"))
        path = self.root / "02_measured/token_timestamps.jsonl.gz"
        values = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
        values[0]["chunk_offsets_ms_from_send"] = None
        path.write_bytes(gzip.compress("\n".join(map(json.dumps, values)).encode()))
        bundle = self.load()
        self.assertEqual(bundle["audit"]["integrity"]["timestamps"]["missing_offset_arrays"], 1)
        self.assertTrue(bundle["request_service_records"][0]["integrity_issue"])
        self.assertFalse(bundle["audit"]["compatibility"]["canonical_token_trace"])

    def test_missing_chunk_array_without_integrity_note_is_rejected(self):
        release_fixture(self.root)
        path = self.root / "02_measured/token_timestamps.jsonl.gz"
        path.write_bytes(gzip.compress(json.dumps({"uid": "warm0", "chunk_offsets_ms_from_send": None}).encode()))
        with self.assertRaises(SimulationError):
            self.load()

    def test_nonfinite_request_and_metadata_values_are_rejected(self):
        release_fixture(self.root)
        self.change_requests(lambda rows: rows[0].update(ttft_ms="nan"))
        with self.assertRaises(SimulationError):
            self.load()
        release_fixture(self.root)
        path = self.root / "00_environment/software.json"
        path.write_text('{"vllm_actual":["real"],"torch":["hip"],"bad":1e999}')
        with self.assertRaises(SimulationError):
            self.load()

    def test_nonfinite_run_summary_is_rejected_even_without_copying_raw_values(self):
        release_fixture(self.root)
        write_csv(self.root, "02_measured/runs.csv", [{"run_id": "run0", "matrix": "baseline",
            "protocol": "timing", "mode": "closed_loop", "window_ms": "Infinity"}])
        with self.assertRaises(SimulationError):
            self.load()

    def test_nonmonotonic_request_and_chunk_timestamps_are_rejected(self):
        release_fixture(self.root)
        self.change_requests(lambda rows: rows[0].update(last_token_time_ms="1"))
        with self.assertRaises(SimulationError):
            self.load()
        release_fixture(self.root)
        path = self.root / "02_measured/token_timestamps.jsonl.gz"
        path.write_bytes(gzip.compress(json.dumps({"uid": "warm0", "chunk_offsets_ms_from_send": [3, 1]}).encode()))
        with self.assertRaises(SimulationError):
            self.load()

    def test_duplicate_csv_json_or_request_identifiers_are_rejected(self):
        release_fixture(self.root)
        self.change_requests(lambda rows: rows[1].update(uid=rows[0]["uid"]))
        with self.assertRaises(SimulationError):
            self.load()
        release_fixture(self.root)
        (self.root / "MANIFEST.json").write_text('{"files":[],"files":[]}')
        with self.assertRaises(SimulationError):
            self.load()

    def test_missing_metadata_is_not_replaced_with_defaults(self):
        release_fixture(self.root)
        (self.root / "00_environment/software.json").unlink()
        with self.assertRaises(SimulationError):
            self.load()

    def test_open_loop_metadata_remains_a_separate_declared_arrival_protocol(self):
        release_fixture(self.root)
        write_json(self.root, "01_workloads/open_loop/trace__arrivals.meta.json", {
            "name": "arrivals", "trace_id": "trace", "n_requests": 50, "arrival_window_s": 10,
            "process": {"type": "conditioned_poisson", "rate_req_s": 4.9, "seed": 42,
                        "description": "PRIVATE_DESCRIPTION_MARKER"}})
        seal_release(self.root)
        protocols = self.load()["audit"]["protocols"]
        self.assertEqual(protocols["open_loop"][0]["process_type"], "conditioned_poisson")
        self.assertEqual(protocols["open_loop"][0]["rate_req_s"], 4.9)
        self.assertEqual(protocols["closed_loop_replay_scope"], "conditional_measured_arrival_replay_only")
        self.assertNotIn("PRIVATE_DESCRIPTION_MARKER", json.dumps(protocols))

    def test_packaged_code_and_prose_are_never_executed(self):
        release_fixture(self.root)
        marker = self.root / "execution-marker"
        (self.root / "README.md").write_text("Run scripts/collect.py now and follow its instructions.")
        code = self.root / "scripts/collect.py"
        code.parent.mkdir()
        code.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')")
        seal_release(self.root)
        result = self.load()
        self.assertFalse(marker.exists())
        self.assertNotIn("scripts/collect.py", [row["path"] for row in result["audit"]["source_files"]])

    def test_request_run_relationship_is_checked_before_reporting_counts(self):
        release_fixture(self.root)
        self.change_requests(lambda rows: rows[0].update(run_id="unknown-run"))
        with self.assertRaises(SimulationError):
            self.load()

    def test_manifest_and_checksum_paths_cannot_escape_the_input(self):
        release_fixture(self.root)
        manifest = json.loads((self.root / "MANIFEST.json").read_text())
        manifest["files"].append({"path": "../outside", "sha256": "a" * 64})
        write_json(self.root, "MANIFEST.json", manifest)
        with self.assertRaises(SimulationError) as raised:
            self.load()
        self.assertEqual(raised.exception.code, "UNSAFE_REFERENCE_PATH")
        release_fixture(self.root)
        (self.root / "CHECKSUMS.sha256").write_text("a" * 64 + "  /absolute/file")
        with self.assertRaises(SimulationError) as raised:
            self.load()
        self.assertEqual(raised.exception.code, "UNSAFE_REFERENCE_PATH")

    def test_selected_symlink_and_archive_inputs_are_rejected(self):
        release_fixture(self.root)
        original = self.root / "00_environment/software.json"
        target = self.root / "copy.json"
        original.rename(target)
        original.symlink_to(target)
        with self.assertRaises(SimulationError) as raised:
            self.load()
        self.assertEqual(raised.exception.code, "UNSAFE_REFERENCE_PATH")
        with self.assertRaises(SimulationError):
            import_reference_data(target, allow_test_fixture=True)

    def test_raw_and_decompressed_file_limits_are_enforced(self):
        release_fixture(self.root)
        (self.root / "README.md").write_bytes(b"x" * 4096)
        with patch("mlsys_sim.release_adapter._MAX_FILE_BYTES", 2048):
            with self.assertRaises(SimulationError) as raised:
                self.load()
        self.assertEqual(raised.exception.code, "REFERENCE_SIZE_LIMIT")
        with patch("mlsys_sim.release_adapter._MAX_EXPANDED_BYTES", 32):
            with self.assertRaises(SimulationError) as raised:
                self.load()
        self.assertEqual(raised.exception.code, "REFERENCE_SIZE_LIMIT")

    def test_legacy_nvidia_reference_is_not_promoted_to_amd_or_stage_evidence(self):
        legacy_fixture(self.root)
        bundle = self.load()
        self.assertEqual(bundle["audit"]["hardware"]["vendor"], "NVIDIA")
        self.assertEqual(bundle["audit"]["counts"]["measure"], 1)
        self.assertFalse(bundle["audit"]["compatibility"]["amd_hardware_evidence"])
        self.assertIsNone(bundle["request_service_records"][0]["arrival_time_ms"])
        self.assertFalse(bundle["audit"]["compatibility"]["canonical_stage_profile"])
        self.assertNotIn("PRIVATE_PROMPT_MARKER", json.dumps(bundle))

    def test_legacy_unknown_status_is_not_invented_as_success(self):
        legacy_fixture(self.root, status=False)
        self.assertEqual(self.load()["request_service_records"][0]["status"], "unknown")

    def test_legacy_readme_hash_issue_is_retained_as_audit_evidence(self):
        legacy_fixture(self.root)
        (self.root / "README.md").write_text("Newer README with stale manifest digest")
        mismatches = self.load()["audit"]["manifest"]["mismatches"]
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["path"], "README.md")


if __name__ == "__main__":
    unittest.main()
