"""Prompt identities, positive-context profiles and legacy comparison contracts."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, Experiment, MemoryPlanner, RequestSpec,
                       SimulationError, StrategySpec, calibrate_dataset,
                       compare_runs, comparison_contract, estimate_profile,
                       experiment_from_dict, predict_experiment, profile_identity,
                       simulate, write_json_exclusive)


class PrefixChunkContractTests(unittest.TestCase):
    def test_policy_defaults_and_prompt_identity_roundtrip(self):
        exp = experiment_from_dict({"strategy": {"prefix_cache": True, "chunked_prefill": True},
            "workload": [{"request_id": "r", "arrival_ms": 0, "input_tokens": 3,
                          "output_tokens": 1, "prompt_token_ids": [0, 1, 255],
                          "cache_namespace": "tenant-a"}]})
        self.assertEqual(exp.workload[0].prompt_token_ids, (0, 1, 255))
        self.assertEqual(exp.strategy.prefill_chunk_tokens, 512)
        self.assertEqual(exp.strategy.prefix_cache_max_blocks, 128)
        self.assertEqual(experiment_from_dict(asdict(exp)), exp)

    def test_invalid_policy_controls_are_rejected(self):
        for policy in [{"prefix_cache": 1}, {"chunked_prefill": "true"},
                       {"prefix_cache_max_blocks": 0}, {"prefix_cache_max_blocks": True},
                       {"prefill_chunk_tokens": -1}, {"prefill_chunk_tokens": 1.5}]:
            with self.subTest(policy=policy), self.assertRaises(SimulationError):
                experiment_from_dict({"strategy": policy})

    def test_bad_prompt_ids_rejected_for_workload_and_warmup(self):
        base = {"request_id": "r", "arrival_ms": 0, "input_tokens": 2, "output_tokens": 1}
        for identities in [[1], [1, 256], [-1, 0], [True, 2], [1.0, 2], "12"]:
            for warmup in [False, True]:
                row = {**base, "prompt_token_ids": identities}
                config = {"execution": {"warmup": [row]}} if warmup else {"workload": [row]}
                with self.subTest(ids=identities, warmup=warmup), self.assertRaises(SimulationError):
                    experiment_from_dict(config)
        with self.assertRaises(SimulationError):
            experiment_from_dict({"workload": [{**base, "cache_namespace": None}]})

    def test_legacy_workload_digest_is_stable_and_identities_change_it(self):
        exp = Experiment(workload=(RequestSpec("r", 0.0, 2, 1),))
        legacy_row = asdict(exp.workload[0])
        legacy_row.pop("prompt_token_ids")
        legacy_row.pop("cache_namespace")
        canonical = json.dumps([legacy_row], sort_keys=True, separators=(",", ":"))
        digest = comparison_contract(exp)["workload_sha256"]
        self.assertEqual(digest, hashlib.sha256(canonical.encode()).hexdigest())
        for row in [replace(exp.workload[0], prompt_token_ids=(1, 2)),
                    replace(exp.workload[0], cache_namespace="tenant")]:
            self.assertNotEqual(comparison_contract(replace(exp, workload=(row,)))["workload_sha256"], digest)

    def test_pre_feature_comparison_contract_has_additive_defaults(self):
        current = comparison_contract(Experiment())
        legacy = deepcopy(current)
        for name in ("prefix_cache", "prefix_cache_max_blocks", "chunked_prefill", "prefill_chunk_tokens"):
            legacy["strategy"].pop(name)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predicted = root / "run"
            predicted.mkdir()
            (predicted / "manifest.json").write_text(json.dumps({"backend": "simulated",
                "comparison_contract": current}))
            (predicted / "result.json").write_text(json.dumps({"backend": "simulated",
                "status": "COMPLETED", "counts": {"total": 0}}))
            observation = root / "observation.json"
            observation.write_text(json.dumps({"schema_version": 1, "kind": "measured_run",
                "comparison_contract": legacy, "source": {"kind": "test_fixture",
                "reference": "Compatibility fixture only"}, "metrics": {"request_count": 0}}))
            self.assertEqual(compare_runs(predicted, observation, allow_test_fixture=True)
                             ["metrics"]["request_count"]["absolute_error"], 0)

    def dataset(self, exp, shapes):
        return {"schema_version": 1, "dataset_id": "chunk-context-fixture",
                "identity": profile_identity(exp),
                "source": {"kind": "test_fixture", "reference": "Hand calculations, not hardware"},
                "records": [{"record_id": f"r{i}", "group_id": f"g{i}", "phase": phase,
                             "stage_index": 0, "query_tokens": [query], "context_lengths": [context],
                             "duration_ms": duration, "workspace_peak_bytes": workspace}
                            for i, (phase, query, context, duration, workspace) in enumerate(shapes)]}

    def test_profiles_keep_positive_prefill_and_recompute_contexts_distinct(self):
        data = self.dataset(Experiment(), [("prefill", 4, 0, 1, 100), ("prefill", 4, 4, 3, 300),
                                           ("recompute", 4, 4, 2, 200)])
        profile = calibrate_dataset(data, profile_version="positive-context", allow_test_fixture=True)
        estimate = estimate_profile(profile, "prefill", [4], [2], allow_test_fixture=True)
        self.assertEqual(estimate.duration_ms, 2)
        self.assertEqual(estimate.breakdown["workspace_peak_bytes"], 300)
        self.assertEqual(estimate_profile(profile, "recompute", [4], [4],
                                         allow_test_fixture=True).duration_ms, 2)
        self.assertEqual(len(set(profile["training"]["shape_keys"])), 3)

    def test_context_zero_profile_does_not_silently_model_prefix_hit(self):
        profile = calibrate_dataset(self.dataset(Experiment(), [("prefill", 4, 0, 1, 100)]),
                                    profile_version="zero-only", allow_test_fixture=True)
        for extrapolation in [False, True]:
            with self.subTest(extrapolation=extrapolation), self.assertRaises(SimulationError) as raised:
                estimate_profile(profile, "prefill", [4], [4], allow_extrapolation=extrapolation,
                                 allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_chunked_profile_simulation_and_workspace_use_same_context(self):
        exp = Experiment(workload=(RequestSpec("r", 0, 8, 1),),
                         strategy=StrategySpec(chunked_prefill=True, prefill_chunk_tokens=4,
                                               max_batch_tokens=4))
        profile = calibrate_dataset(self.dataset(exp, [("prefill", 4, 0, 1, 100),
                                                       ("prefill", 4, 4, 3, 300)]),
                                    profile_version="chunks", allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            write_json_exclusive(path, profile)
            exp = replace(exp, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                            profile_version="chunks", allow_test_fixture=True))
            planner = MemoryPlanner(exp)
            decision = planner.check({"r": 8}, [4], "prefill")
            self.assertEqual(decision.details["workspace_bytes"], 300)
            result = simulate(exp)
        self.assertEqual(result.requests[0]["token_times_ms"], [4])
        self.assertEqual(result.requests[0]["prefill_computed_tokens"], 8)
        self.assertEqual(result.result["memory"][0]["workspace_peak_bytes"], 300)

    def test_predict_explains_that_cache_and_chunks_are_not_applied(self):
        exp = Experiment(workload=(RequestSpec("r", 0, 8, 1, prompt_token_ids=tuple(range(8))),),
                         strategy=StrategySpec(prefix_cache=True, chunked_prefill=True,
                                               prefill_chunk_tokens=4, max_batch_tokens=4))
        result = predict_experiment(exp)
        self.assertEqual(result["shape"]["query_tokens"], [8])
        self.assertFalse(result["shape"]["serving_cache_and_chunk_policies_applied"])
        self.assertFalse(result["shape"]["within_batch_token_limit"])


if __name__ == "__main__":
    unittest.main()
