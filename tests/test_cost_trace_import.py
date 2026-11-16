"""Retrospective grouping and trace-derived contexts without private fixtures."""

from dataclasses import asdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import Experiment


PATH = Path(__file__).resolve().parents[1] / "scripts" / "import_cost_trace.py"
SPEC = importlib.util.spec_from_file_location("cost_trace_import_test_target", PATH)
assert SPEC is not None and SPEC.loader is not None
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def dump(path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in value) if jsonl else json.dumps(value))


def schedule(batch):
    prefix = ([[1024] * 8, [1] * 8 + [1024] * 7, [1] * 15 + [1024]] if batch == 16
              else [[1024] * batch])
    return prefix + [[1] * batch for _ in range(19)] + [[1]]


def fixture(root):
    model = "toy"
    prefix = root / "5_raw/real_machine/run_20261116a/raw" / model
    context = {
        "engine": {"runner_class": "FixtureRunner", "use_v2_model_runner": True,
                   "async_scheduling": True, "compilation_mode": "3",
                   "cudagraph_mode": "FULL_AND_PIECEWISE", "attention_impl": "fixture.RocmAttentionImpl",
                   "kv_cache_shape_layer0": [100, 2, 16, 32]},
        "environment": {"model": {"key": model, "hf_revision": "unspecified"}, "vllm": "fixture-v1",
                        "torch": "fixture-torch", "hip": "fixture-hip", "python": "fixture-python",
                        "image_id": "fixture-image", "hostname": "do-not-publish-host",
                        "device": {"arch": "fixture", "total_memory": 64 * 1024**2}},
        "llm_kwargs": {"model": "/do-not-publish/model", "max_num_batched_tokens": 32768,
                       "max_num_seqs": 32, "enable_prefix_caching": False}}
    exp = asdict(Experiment())
    dump(root / f"5_raw/simulator_runs/experiments/{model}_rx9070xt_prefill_128_nominal.json", exp)
    for repeat in (1, 2, 3):
        run = prefix / f"timing_rep{repeat}"
        current = deepcopy(context)
        current["engine"]["kv_cache_shape_layer0"][0] += repeat
        dump(run / "run_context.json", current)
        rows = []
        for length in (128, 512, 2048):
            steps = [{"num_scheduled": [length], "context_min": 0, "context_max": 0,
                      "step_ms": 10 + length / 1000, "generate_index": i} for i in range(100)]
            rows.append({"model": model, "phase": "prefill", "shape": f"S={length}",
                         "kind": "step_hooked", "n": 100, "steps": steps})
        for batch in (1, 4, 16):
            full = IMPORTER.reconstruct_contexts(schedule(batch), batch)
            steps = []
            for generation in range(8):
                for index in range(16):
                    step = {"num_scheduled": [1] * batch, "context_min": min(full[index]),
                            "context_max": max(full[index]), "step_ms": 2 + batch / 100,
                            "generate_index": generation, "step_index": index,
                            "all_steps_num_scheduled": schedule(batch) if index == 0 else None}
                    steps.append(step)
            rows.append({"model": model, "phase": "decode", "shape": f"B={batch},C=1024",
                         "kind": "step_hooked", "n": 128, "steps": steps})
        dump(run / "l2_timing.jsonl", rows, jsonl=True)
    profile = prefix / "profile"
    context["llm_kwargs"]["profiler_config"] = {"profiler": "fixture", "torch_profiler_dir": "/do-not-publish/profiler"}
    dump(profile / "run_context.json", context)
    summaries, schedules = [], []
    for phase, shape in [("prefill", f"S={x}") for x in (128, 512, 2048)] + [("decode", f"B={b},C=1024") for b in (1, 4, 16)]:
        summaries.append({"phase": phase, "shape": shape, "target_steps": 4 if phase == "prefill" else 16,
                          "category_ms_per_step": {"GEMM": 1.0, "Element-wise": 0.1, "Attention": 0.2}})
        raw = [[int(shape.split("=")[1])]] * 4 if phase == "prefill" else schedule(int(shape.split(",")[0].split("=")[1]))
        schedules.append({"phase": phase, "shape": shape, "all_steps_num_scheduled": raw})
    dump(profile / "profile_summary.json", summaries)
    dump(profile / "l2_profile.jsonl", schedules, jsonl=True)
    return prefix


class CostTraceImportTests(unittest.TestCase):
    def test_reconstructed_multiset_matches_mixed_preamble_without_claiming_ids(self):
        values = IMPORTER.reconstruct_contexts(schedule(16), 16)
        self.assertEqual(values[0], [1024] + [1025] * 7 + [1026] * 8)
        self.assertEqual(values[15], [1039] + [1040] * 7 + [1041] * 8)
        self.assertEqual(len(values), 19)

    def test_ambiguous_preamble_and_resumed_full_batches_are_rejected(self):
        for raw in ([[1024] * 4, [1] * 3 + [1024]],
                    [[1024] * 4, [1] * 4, [1] * 2, [1] * 4]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                IMPORTER.reconstruct_contexts(raw, 4)

    def test_complete_groups_sessions_and_raw_rows_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            data = IMPORTER.build_dataset(root, models=("toy",))
        self.assertEqual(len(data["records"]), 2058)
        self.assertEqual({key: len(values) for key, values in data["recommended_split"].items()},
                         {"training_record_ids": 916, "holdout_record_ids": 228,
                          "repeat_record_ids": 456, "unused_record_ids": 458})
        by_id = {r["record_id"]: r for r in data["records"]}
        train = [by_id[x] for x in data["recommended_split"]["training_record_ids"]]
        holdout = [by_id[x] for x in data["recommended_split"]["holdout_record_ids"]]
        self.assertFalse({x["session_id"] for x in train} & {x["session_id"] for x in holdout})
        self.assertFalse({x["group_id"] for x in train} & {x["group_id"] for x in holdout})
        self.assertEqual(data["source"]["kind"], "test_fixture")
        self.assertIn("retrospective", data["source"]["study_design"])
        encoded = json.dumps(data)
        self.assertNotIn("/do-not-publish", encoded)
        self.assertNotIn("do-not-publish-host", encoded)

    def test_profile_context_proxy_is_not_timing_context_or_cpu_overhead(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            data = IMPORTER.build_dataset(root, models=("toy",))
        row = next(r for r in data["records"] if r["kind"] == "gpu_components" and r["shape"] == "B=16,C=1024")
        self.assertTrue(row["context_evidence"]["approximate"])
        self.assertEqual(row["context_evidence"]["selected_indices"], "not_observed")
        self.assertEqual(row["context_lengths"], [1044] * 16)
        self.assertEqual(row["source_timing_boundary"], "separate_process_profile_kernel_sum_v1")
        self.assertIn("not the hooked wall interval", row["transfer_assumption"])

    def test_context_bound_drift_is_rejected_instead_of_invented_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = fixture(root)
            path = prefix / "timing_rep3/l2_timing.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[-1]["steps"][0]["context_max"] = 9999
            dump(path, rows, jsonl=True)
            with self.assertRaisesRegex(ValueError, "disagree with observed bounds"):
                IMPORTER.build_dataset(root, models=("toy",))

    def test_configured_runtime_drift_is_not_mixed_into_one_cost_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = fixture(root)
            path = prefix / "timing_rep2/run_context.json"
            value = json.loads(path.read_text())
            value["engine"]["async_scheduling"] = False
            dump(path, value)
            with self.assertRaisesRegex(ValueError, "Configured runtime controls changed"):
                IMPORTER.build_dataset(root, models=("toy",))


if __name__ == "__main__":
    unittest.main()
