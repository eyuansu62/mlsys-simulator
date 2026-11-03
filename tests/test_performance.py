"""Physical units, input shapes and parallel prediction assumptions."""

from dataclasses import replace
import unittest

from mlsys_sim.config import experiment_from_dict
from mlsys_sim.layout import local_kv_heads, rank_weight_bytes
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
        self.assertEqual(result["online_metrics_reason"], "PP_ONLINE_DES_NOT_IMPLEMENTED")
        self.assertFalse(result["hardware_validated"])


if __name__ == "__main__":
    unittest.main()
