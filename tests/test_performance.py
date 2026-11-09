"""Physical units, input shapes and parallel prediction assumptions."""

from dataclasses import replace
import unittest

from mlsys_sim.config import experiment_from_dict
from mlsys_sim.layout import layer_parameters, local_kv_heads, rank_weight_bytes
from mlsys_sim.performance import AnalyticalModel, predict_experiment, ring_allreduce_ms
from mlsys_sim.schema import Experiment, SimulationError


class PerformanceTests(unittest.TestCase):
    def test_equal_total_tokens_have_distinct_attention_cost(self):
        predictor = AnalyticalModel(Experiment())
        split = predictor.estimate_step("prefill", [512, 512], [0, 0])
        long = predictor.estimate_step("prefill", [1024], [0])
        self.assertEqual(split.breakdown["linear"]["flops_per_rank"],
                         long.breakdown["linear"]["flops_per_rank"])
        self.assertGreater(long.breakdown["attention"]["flops_per_rank"],
                           split.breakdown["attention"]["flops_per_rank"])

    def test_decode_context_increases_kv_traffic(self):
        predictor = AnalyticalModel(Experiment())
        short = predictor.estimate_step("decode", [1], [16])
        long = predictor.estimate_step("decode", [1], [1024])
        self.assertGreater(long.breakdown["attention"]["hbm_bytes_per_rank"],
                           short.breakdown["attention"]["hbm_bytes_per_rank"])

    def test_allreduce_units_and_tp_one(self):
        self.assertEqual(ring_allreduce_ms(1, 1000, 1, 1), 0)
        # TP2, logical tensor 1MB, effective 1GB/s, round startup 10us.
        self.assertAlmostEqual(ring_allreduce_ms(2, 1_000_000, 1, 10), 1.02)

    def test_effective_rates_reject_floating_underflow_and_overflow(self):
        for device_field, efficiency_field in (
            ("compute_tflops", "compute_efficiency"),
            ("memory_bandwidth_gbps", "bandwidth_efficiency"),
        ):
            for rate, efficiency in ((5e-324, 5e-324), (1e308, 0.5)):
                with self.subTest(device_field=device_field, rate=rate):
                    exp = experiment_from_dict({
                        "infra": {"devices": [{device_field: rate}]},
                        "calibration": {efficiency_field: efficiency},
                    })
                    with self.assertRaises(SimulationError) as caught:
                        AnalyticalModel(exp).estimate_step("prefill", [1], [0])
                    self.assertEqual(caught.exception.code, "INVALID_CONFIG")

    def test_link_rate_conversion_and_transfer_overflow_are_rejected(self):
        for rate in (0.0, 1e308, 5e-324):
            with self.subTest(rate=rate):
                with self.assertRaises(SimulationError) as caught:
                    ring_allreduce_ms(2, 1_000_000, rate, 10)
                self.assertEqual(caught.exception.code, "INVALID_CONFIG")
        exp = experiment_from_dict({
            "strategy": {"pipeline_parallel_size": 2},
            "infra": {"devices": [{"id": "a"}, {"id": "b"}],
                      "link_bandwidth_gbps": 1e308},
            "calibration": {"backend": "synthetic"},
        })
        with self.assertRaises(SimulationError) as caught:
            predict_experiment(exp)
        self.assertEqual(caught.exception.code, "INVALID_CONFIG")

    def test_synthetic_stage_duration_underflow_is_explicit(self):
        exp = experiment_from_dict({
            "strategy": {"pipeline_parallel_size": 2, "stage_layers": [1, 3]},
            "infra": {"devices": [{"id": "a"}, {"id": "b"}]},
            "calibration": {"backend": "synthetic", "prefill_ms": 5e-324},
        })
        with self.assertRaises(SimulationError) as caught:
            AnalyticalModel(exp).estimate_step("prefill", [1], [0])
        self.assertEqual(caught.exception.code, "INVALID_CONFIG")

    def test_tp_counts_collectives_and_retains_gqa_replicas(self):
        exp = experiment_from_dict({"strategy": {"tensor_parallel_size": 4},
                                    "infra": {"devices": [{"id": str(i)} for i in range(4)]}})
        estimate = AnalyticalModel(exp).estimate_step("decode", [1], [32])
        self.assertEqual(estimate.breakdown["communication"]["allreduce_count"], 8)
        self.assertEqual(local_kv_heads(exp.strategy.model, 4), 1)
        two = replace(exp, strategy=replace(exp.strategy, tensor_parallel_size=2))
        self.assertEqual(local_kv_heads(two.strategy.model, 2), 1)
        self.assertGreater(rank_weight_bytes(exp, 0), rank_weight_bytes(Experiment(), 0) / 4)

    def test_synthetic_is_fixed_per_batch_and_marked(self):
        exp = experiment_from_dict({"calibration": {"backend": "synthetic", "decode_ms": 7}})
        predictor = AnalyticalModel(exp)
        self.assertEqual(predictor.estimate_step("decode", [1], [1]).duration_ms, 7)
        self.assertEqual(predictor.estimate_step("decode", [1, 1], [30, 40]).duration_ms, 7)
        self.assertEqual(predictor.estimate_step("prefill", [2], [0]).source, "synthetic")

    def test_invalid_shapes_are_rejected(self):
        predictor = AnalyticalModel(Experiment())
        for args in [("decode", [2], [0]), ("prefill", [], []),
                     ("prefill", [1], [-1]), ("prefill", [1], [0, 1])]:
            with self.assertRaises(SimulationError):
                predictor.estimate_step(*args)

    def test_pp_report_sums_stage_latency_and_p2p_but_no_online_sla(self):
        exp = experiment_from_dict({
            "strategy": {"pipeline_parallel_size": 2, "stage_layers": [1, 3]},
            "infra": {"devices": [{"id": "a"}, {"id": "b"}]},
            "calibration": {"backend": "synthetic", "prefill_ms": 12},
        })
        result = predict_experiment(exp)
        self.assertEqual([s["prefill_ms"] for s in result["stages"]], [3, 9])
        self.assertAlmostEqual(result["isolated_prefill_ms"],
                               12 + result["communication"]["prefill_p2p_ms"])
        self.assertTrue(all(v is None for v in result["online_metrics"].values()))
        self.assertEqual(result["online_metrics_reason"], "SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING")
        self.assertFalse(result["hardware_validated"])


class QuantizedStoragePerformanceTests(unittest.TestCase):
    def step(self, phase, queries, contexts, **model):
        exp = experiment_from_dict({"strategy": {"model": model}})
        return AnalyticalModel(exp).estimate_step(phase, queries, contexts), exp

    def test_fused_dequant_charges_work_without_extra_weight_traffic(self):
        baseline, _ = self.step("decode", [1], [0])
        fused, exp = self.step("decode", [1], [0], weight_bytes=0.5, weight_dtype="int4",
                               weight_dequant="fused")
        model = exp.strategy.model
        layers = model.num_layers
        dequant = layers * layer_parameters(model, 1)["linear"]
        linear = fused.breakdown["linear"]
        self.assertEqual(linear["dequant_flops_per_rank"], dequant)
        self.assertEqual(linear["flops_per_rank"],
                         baseline.breakdown["linear"]["flops_per_rank"] + dequant)
        activation_io = layers * 1 * (4 * model.hidden_size) * model.activation_bytes
        self.assertEqual(linear["hbm_bytes_per_rank"],
                         layers * layer_parameters(model, 1)["linear"] * 0.5 + activation_io)
        self.assertNotIn("dequant", fused.breakdown)
        self.assertTrue(any("Fused weight dequant" in note for note in fused.assumptions))
        self.assertTrue(any("no accuracy" in note for note in fused.assumptions))

    def test_materialized_dequant_is_a_separate_pass_with_expanded_gemm(self):
        baseline, _ = self.step("decode", [1], [0])
        # 0.515625 is the GPTQ-128 effective width (group scales included), so
        # no raw dtype label matches it; auto keeps the bytes as source of truth.
        exp = experiment_from_dict({"strategy": {
            "model": {"weight_bytes": 0.515625, "weight_dequant": "materialized"}}})
        estimate = AnalyticalModel(exp).estimate_step("decode", [1], [0])
        model = exp.strategy.model
        params = layer_parameters(model, 1)["linear"]
        dequant = estimate.breakdown["dequant"]
        self.assertEqual(dequant["flops_per_rank"], 4 * params)
        self.assertEqual(dequant["hbm_bytes_per_rank"], 4 * params * (0.515625 + 2.0))
        self.assertEqual(dequant["peak_expanded_weights_bytes_per_rank"], 475136)
        self.assertAlmostEqual(dequant["overhead_ms"], 4 * exp.calibration.launch_overhead_us / 1000)
        linear = estimate.breakdown["linear"]
        # The GEMM reads expanded activation-width weights, so its bytes match
        # the unquantized baseline; the packed-read/write pass is separate.
        self.assertEqual(linear["hbm_bytes_per_rank"],
                         baseline.breakdown["linear"]["hbm_bytes_per_rank"])
        self.assertEqual(linear["flops_per_rank"],
                         baseline.breakdown["linear"]["flops_per_rank"])
        self.assertNotIn("dequant_flops_per_rank", linear)
        self.assertTrue(any("Materialized weight dequant" in note for note in estimate.assumptions))

    def test_kv_dequant_is_fused_into_attention_at_declared_rate(self):
        baseline, _ = self.step("decode", [1], [16])
        estimate, exp = self.step("decode", [1], [16], kv_bytes=1.0, kv_dtype="int8")
        model = exp.strategy.model
        kv_elements = 2 * 4 * local_kv_heads(model, 1) * model.head_dim * 17
        attention = estimate.breakdown["attention"]
        self.assertEqual(attention["flops_per_rank"],
                         baseline.breakdown["attention"]["flops_per_rank"] + kv_elements)
        self.assertEqual(attention["hbm_bytes_per_rank"],
                         2 * 4 * local_kv_heads(model, 1) * model.head_dim * 1.0 * 17
                         + baseline.breakdown["attention"]["hbm_bytes_per_rank"]
                         - 2 * 4 * local_kv_heads(model, 1) * model.head_dim * 2.0 * 17)
        self.assertTrue(any("KV dequant is fused on-the-fly" in note for note in estimate.assumptions))

    def test_legacy_float_kv_widths_do_not_acquire_quantization_work(self):
        # All nine pairs were valid before quantized storage was introduced.
        for kv in (2.0, 4.0, 8.0):
            for activation in (2.0, 4.0, 8.0):
                exp = experiment_from_dict({"strategy": {"model": {
                    "kv_bytes": kv, "activation_bytes": activation}}})
                expensive = replace(exp, calibration=replace(exp.calibration, dequant_flops_per_weight=1000))
                for phase, queries, contexts, pairs in (
                    ("prefill", [2, 3], [0, 0], 9),
                    ("decode", [1, 1], [16, 32], 50),
                    ("mixed", [1, 3], [16, 0], 23),
                ):
                    roles = {"request_phases": ["decode", "prefill"]} if phase == "mixed" else {}
                    with self.subTest(kv=kv, activation=activation, phase=phase):
                        result = AnalyticalModel(exp).estimate_step(phase, queries, contexts, **roles)
                        self.assertEqual(result.breakdown["attention"]["flops_per_rank"],
                                         4 * 4 * 8 * 16 * pairs)
                        self.assertEqual(result, AnalyticalModel(expensive).estimate_step(
                            phase, queries, contexts, **roles))
                        self.assertFalse(any("dequant" in note.lower() for note in result.assumptions))

    def test_effective_compressed_kv_widths_still_charge_dequant(self):
        for width in (0.5, 0.515625, 1.0, 2.5):
            exp = experiment_from_dict({"strategy": {"model": {
                "kv_bytes": width, "activation_bytes": 4.0}},
                "calibration": {"dequant_flops_per_weight": 3}})
            result = AnalyticalModel(exp).estimate_step("decode", [1], [16])
            with self.subTest(width=width):
                self.assertEqual(result.breakdown["attention"]["flops_per_rank"],
                                 4 * 4 * 8 * 16 * 17 + 2 * 4 * 2 * 16 * 17 * 3)
                self.assertTrue(any("KV dequant" in note for note in result.assumptions))


if __name__ == "__main__":
    unittest.main()
