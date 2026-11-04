"""Qwen3 matrix geometry, tied storage and additive identity contracts."""

from copy import deepcopy
from dataclasses import asdict, replace
import math
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (AnalyticalModel, CalibrationSpec, DeviceSpec, Experiment, InfraSpec, MemoryConfig,
                       MemoryPlanner, ModelSpec, ProfileModel, SimulationError, StrategySpec,
                       calibrate_dataset, experiment_from_dict, profile_identity, validate_experiment,
                       write_json_exclusive)
from mlsys_sim.layout import layer_parameters, rank_kv_bytes_per_token, rank_weight_bytes
from mlsys_sim.profiles import validate_identity


def qwen(size="4b"):
    fields = ({"num_layers": 28, "hidden_size": 2048, "num_attention_heads": 16,
               "intermediate_size": 6144} if size == "1.7b" else
              {"num_layers": 36, "hidden_size": 2560, "num_attention_heads": 32,
               "intermediate_size": 9728})
    return ModelSpec(name=f"Qwen3-{size}", revision="geometry-golden", num_kv_heads=8,
                     head_dim=128, vocab_size=151936, weight_dtype="bfloat16",
                     kv_dtype="bfloat16", activation_dtype="bfloat16",
                     tie_word_embeddings=True, qk_norm=True, final_norm=True, **fields)


def experiment(model, tp=1, pp=1, stage_layers=()):
    return Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}", memory_bytes=32 * 1024**3)
                                                    for i in range(tp * pp))),
                      strategy=StrategySpec(model=model, tensor_parallel_size=tp,
                                            pipeline_parallel_size=pp, stage_layers=stage_layers))


class QwenGeometryTests(unittest.TestCase):
    def test_real_qwen_logical_parameter_and_kv_goldens(self):
        for size, parameters, kv in (("1.7b", 1720574976, 114688), ("4b", 4022468096, 147456)):
            with self.subTest(size=size):
                exp = experiment(qwen(size))
                validate_experiment(exp)
                self.assertEqual(rank_weight_bytes(exp, 0), parameters * 2)
                self.assertEqual(rank_kv_bytes_per_token(exp, 0), kv)
                self.assertEqual(exp.strategy.model.query_projection_size,
                                 exp.strategy.model.num_attention_heads * exp.strategy.model.head_dim)

    def test_four_b_keeps_residual_and_query_width_independent(self):
        model = qwen()
        self.assertEqual((model.hidden_size, model.query_projection_size), (2560, 4096))
        params = layer_parameters(model, 1)
        self.assertEqual(params["linear"], 100925440)
        self.assertEqual(params["norm"], 5376)
        validate_experiment(experiment(model, tp=4, pp=2, stage_layers=(13, 23)))

    def test_tied_same_rank_storage_deduplicates_but_does_not_remove_output_work(self):
        tied = experiment(qwen("1.7b"), tp=2)
        untied = replace(tied, strategy=replace(tied.strategy,
            model=replace(tied.strategy.model, tie_word_embeddings=False)))
        vocabulary_shard = 151936 * 2048 // 2 * 2
        self.assertEqual(rank_weight_bytes(untied, 0) - rank_weight_bytes(tied, 0), vocabulary_shard)
        left = AnalyticalModel(tied).estimate_step("prefill", [4], [0]).breakdown
        right = AnalyticalModel(untied).estimate_step("prefill", [4], [0]).breakdown
        self.assertEqual(left["embedding_and_output"], right["embedding_and_output"])

    def test_tied_pp_endpoints_have_distinct_physical_copies(self):
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                tied = experiment(qwen(), tp=tp, pp=2, stage_layers=(13, 23))
                untied = replace(tied, strategy=replace(tied.strategy,
                    model=replace(tied.strategy.model, tie_word_embeddings=False)))
                for stage in range(2):
                    self.assertEqual(rank_weight_bytes(tied, stage), rank_weight_bytes(untied, stage))
                # TP partitions vocabulary storage, while replicated RMSNorms
                # remain separately owned on every rank.
                total = tp * sum(rank_weight_bytes(tied, stage) for stage in range(2))
                logical = 8044936192
                duplicate_vocab = 151936 * 2560 * 2
                replicated_norms = (tp - 1) * ((36 * 5376 + 2560) * 2)
                self.assertEqual(total, logical + duplicate_vocab + replicated_norms)

    def test_single_stage_tp_storage_shards_matrices_and_replicates_norms(self):
        for size, logical in (("1.7b", 3441149952), ("4b", 8044936192)):
            model = qwen(size)
            norm_parameters = model.num_layers * (2 * model.hidden_size + 2 * model.head_dim) + model.hidden_size
            for tp in (1, 2, 4):
                with self.subTest(size=size, tp=tp):
                    exp = experiment(model, tp=tp)
                    physical = tp * rank_weight_bytes(exp, 0)
                    self.assertEqual(physical, logical + (tp - 1) * norm_parameters * 2)

    def test_qk_norm_is_head_dim_shared_weights_replicated_on_tp_ranks(self):
        base = ModelSpec(qk_norm=True)
        plain = replace(base, qk_norm=False)
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                self.assertEqual(layer_parameters(base, tp)["norm"] - layer_parameters(plain, tp)["norm"], 32)
        # TP=4 replicates the two KV heads and their projection weights.
        replicated = experiment(base, tp=4)
        self.assertEqual(rank_kv_bytes_per_token(replicated, 0), 256)
        self.assertGreater(4 * layer_parameters(base, 4)["linear"], layer_parameters(base, 1)["linear"])

    def test_final_norm_is_last_stage_only_and_replicated_per_rank(self):
        for pp in (1, 2):
            for tp in (1, 2, 4):
                with self.subTest(pp=pp, tp=tp):
                    with_norm = experiment(qwen(), tp=tp, pp=pp)
                    without = replace(with_norm, strategy=replace(with_norm.strategy,
                        model=replace(with_norm.strategy.model, final_norm=False)))
                    differences = [rank_weight_bytes(with_norm, stage) - rank_weight_bytes(without, stage)
                                   for stage in range(pp)]
                    self.assertEqual(differences, [0] * (pp - 1) + [2560 * 2])

    def test_new_flags_are_strict_booleans_and_query_width_is_derived(self):
        for field in ("tie_word_embeddings", "qk_norm", "final_norm"):
            for value in (0, 1, None, "true"):
                with self.subTest(field=field, value=value), self.assertRaises(SimulationError):
                    experiment_from_dict({"strategy": {"model": {field: value}}})
        with self.assertRaises(SimulationError):
            experiment_from_dict({"strategy": {"model": {"query_projection_size": 128}}})
        validate_experiment(experiment(replace(ModelSpec(), hidden_size=96)))

    def test_legacy_identity_omits_additive_false_flags(self):
        legacy = profile_identity(Experiment())
        self.assertTrue(all(field not in legacy["model"] for field in
                            ("tie_word_embeddings", "qk_norm", "final_norm")))
        self.assertNotIn("architecture_config_sha256", legacy["model"])
        explicit = deepcopy(legacy)
        explicit["model"].update(tie_word_embeddings=False, qk_norm=False, final_norm=False,
                                  architecture_config_sha256="")
        self.assertEqual(validate_identity(explicit), legacy)
        incomplete = deepcopy(legacy)
        incomplete["model"].pop("head_dim")
        with self.assertRaises(SimulationError):
            validate_identity(incomplete)

    def test_nondefault_model_mechanisms_are_identity_bound(self):
        identity = profile_identity(experiment(qwen()))
        for field in ("tie_word_embeddings", "qk_norm", "final_norm"):
            self.assertIs(identity["model"][field], True)
            changed = deepcopy(identity)
            changed["model"][field] = False
            self.assertNotEqual(validate_identity(changed), identity)
        self.assertEqual(validate_identity(identity), identity)

    def test_architecture_override_digest_is_strict_and_identity_bound(self):
        for value in ("short", "A" * 64, "g" * 64, None, True):
            with self.subTest(value=value), self.assertRaises(SimulationError):
                experiment_from_dict({"strategy": {"model": {"architecture_config_sha256": value}}})
        first = experiment(replace(qwen(), architecture_config_sha256="a" * 64))
        second = replace(first, strategy=replace(first.strategy,
            model=replace(first.strategy.model, architecture_config_sha256="b" * 64)))
        self.assertNotEqual(profile_identity(first), profile_identity(second))
        self.assertEqual(validate_identity(profile_identity(first)), profile_identity(first))

    def test_runtime_configuration_digest_keeps_legacy_empty_identity(self):
        identity = profile_identity(Experiment())
        self.assertNotIn("runtime_config_sha256", identity["runtime"])
        explicit = deepcopy(identity)
        explicit["runtime"]["runtime_config_sha256"] = ""
        self.assertEqual(validate_identity(explicit), identity)

    def test_runtime_configuration_digest_is_strict_and_identity_bound(self):
        for value in ("short", "A" * 64, "g" * 64, None, True):
            with self.subTest(value=value), self.assertRaises(SimulationError):
                experiment_from_dict({"calibration": {"runtime_config_sha256": value}})
        first = Experiment(calibration=CalibrationSpec(runtime_config_sha256="a" * 64))
        second = replace(first, calibration=replace(first.calibration, runtime_config_sha256="b" * 64))
        self.assertEqual(validate_identity(profile_identity(first)), profile_identity(first))
        self.assertNotEqual(profile_identity(first), profile_identity(second))

    def test_profile_rejects_different_runtime_configuration_at_same_runtime_version(self):
        first = Experiment(calibration=CalibrationSpec(runtime_config_sha256="a" * 64))
        dataset = {"schema_version": 1, "dataset_id": "runtime-configuration-fixture",
            "identity": profile_identity(first),
            "source": {"kind": "test_fixture", "reference": "Same runtime version, distinct execution policies"},
            "records": [{"record_id": "r", "group_id": "shape", "phase": "prefill", "stage_index": 0,
                         "query_tokens": [4], "context_lengths": [0], "duration_ms": 1,
                         "workspace_peak_bytes": None}]}
        profile = calibrate_dataset(dataset, profile_version="config-v1", allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            write_json_exclusive(path, profile)
            calibrated = replace(first, calibration=replace(first.calibration, backend="profile",
                profile_path=str(path), profile_version="config-v1", allow_test_fixture=True))
            self.assertEqual(ProfileModel(calibrated).estimate_step("prefill", [4], [0]).duration_ms, 1)
            changed = replace(calibrated, calibration=replace(calibrated.calibration, runtime_config_sha256="b" * 64))
            with self.assertRaises(SimulationError) as raised:
                ProfileModel(changed)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_default_macro_timings_remain_numerically_unchanged(self):
        model = AnalyticalModel(Experiment())
        cases = (
            ("prefill", [8, 16], [0, 0], None, 0.49692000000000003),
            ("decode", [1, 1], [8, 16], None, 0.43292),
            ("mixed", [1, 8], [16, 4], ["decode", "prefill"], 0.453656),
        )
        for phase, query, context, roles, expected in cases:
            with self.subTest(phase=phase):
                actual = model.estimate_step(phase, query, context, request_phases=roles).duration_ms
                # Permit only rounding-level variation across supported Python versions.
                self.assertAlmostEqual(actual, expected, delta=4 * math.ulp(expected))

    def test_weight_accounting_uses_exact_integer_matrix_sizes(self):
        hidden = 1000000001
        model = ModelSpec(num_layers=1, hidden_size=hidden, intermediate_size=1,
                          num_attention_heads=1, num_kv_heads=1, head_dim=1000000000, vocab_size=1)
        exp = experiment(model)
        validate_experiment(exp)
        parameters = 4 * hidden * 1000000000 + 3 * hidden + 2 * hidden + 2 * hidden
        self.assertEqual(rank_weight_bytes(exp, 0), parameters * 2)

    def test_kv_accounting_uses_exact_integer_head_sizes(self):
        heads = 100000001
        model = ModelSpec(num_layers=1, num_attention_heads=heads, num_kv_heads=heads, head_dim=heads)
        exp = experiment(model)
        validate_experiment(exp)
        self.assertEqual(rank_kv_bytes_per_token(exp, 0), 4 * heads * heads)

    def test_independent_query_width_changes_attention_io_but_not_kv_or_pp_payload(self):
        base = ModelSpec(hidden_size=96, num_attention_heads=8, num_kv_heads=2, head_dim=16)
        wider = replace(base, head_dim=32)
        first = experiment(base, tp=2, pp=2)
        second = experiment(wider, tp=2, pp=2)
        left = AnalyticalModel(first).estimate_step("prefill", [4], [0])
        right = AnalyticalModel(second).estimate_step("prefill", [4], [0])
        self.assertGreater(right.breakdown["linear"]["flops_per_rank"], left.breakdown["linear"]["flops_per_rank"])
        self.assertGreater(right.breakdown["attention"]["hbm_bytes_per_rank"], left.breakdown["attention"]["hbm_bytes_per_rank"])
        self.assertEqual(right.breakdown["communication"]["allreduce_payload_bytes"],
                         left.breakdown["communication"]["allreduce_payload_bytes"])
        self.assertEqual(right.breakdown["communication"]["allreduce_payload_bytes"], 4 * 96 * 2)

    def test_norm_costs_are_explicit_but_uncalibrated(self):
        base = experiment(ModelSpec())
        normalized = replace(base, strategy=replace(base.strategy,
            model=replace(base.strategy.model, qk_norm=True, final_norm=True)))
        left = AnalyticalModel(base).estimate_step("decode", [1], [8])
        right = AnalyticalModel(normalized).estimate_step("decode", [1], [8])
        self.assertGreater(right.breakdown["misc"]["flops_per_rank"], left.breakdown["misc"]["flops_per_rank"])
        self.assertGreater(right.breakdown["misc"]["hbm_bytes_per_rank"], left.breakdown["misc"]["hbm_bytes_per_rank"])
        self.assertEqual(right.source, "analytical_uncalibrated")

    def test_memory_planner_uses_tied_weight_savings_and_unchanged_kv_geometry(self):
        exp = experiment(qwen("1.7b"))
        exp = replace(exp, strategy=replace(exp.strategy, memory=MemoryConfig(kv_pool_bytes=8 * 1024 * 1024)))
        planner = MemoryPlanner(exp)
        rank = planner.snapshot()[0]
        self.assertEqual(rank["weights_bytes"], 3441149952)
        self.assertEqual(rank["kv_bytes_per_token"], 114688)
        self.assertEqual(asdict(exp.strategy.model)["hidden_size"], 2048)


if __name__ == "__main__":
    unittest.main()
