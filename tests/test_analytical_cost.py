"""Explicit analytical corrections must never leak across runtime domains."""

from dataclasses import asdict, replace
import unittest

from mlsys_sim.analytical_cost import analytical_cost_identity, nominal_cost_features
from mlsys_sim.config import experiment_from_dict
from mlsys_sim.performance import AnalyticalModel
from mlsys_sim.schema import AnalyticalCostSpec, CalibrationSpec, Experiment, SimulationError


def domain():
    return Experiment(calibration=CalibrationSpec(
        runtime_name="fixture-runtime", runtime_version="1", runtime_config_sha256="a" * 64,
        prefill_execution_mode="eager", decode_execution_mode="graph_configured",
        prefill_attention_backend="fixture-attention", decode_attention_backend="fixture-paged",
        analytical_cost_timing_boundary="fixture_step_wall_v1", allow_test_fixture=True))


def calibrated(exp=None, phase="prefill", **changes):
    exp = exp or domain()
    selectors = exp.calibration
    spec = AnalyticalCostSpec(
        phase=phase, execution_mode=getattr(selectors, phase + "_execution_mode"),
        attention_backend=getattr(selectors, phase + "_attention_backend"),
        identity_sha256=analytical_cost_identity(exp),
        calibration_id="fixture-cost-v1", training_dataset_sha256="b" * 64,
        training_record_ids=("train-1", "train-2"), source_kind="test_fixture",
        min_batch_size=1, max_batch_size=4, min_query_tokens=1,
        max_query_tokens=512 if phase == "prefill" else 1,
        min_context_tokens=0 if phase == "prefill" else 1024,
        max_context_tokens=0 if phase == "prefill" else 1041,
        nonattention_scale=1.4, attention_scale=3.0,
        attention_fixed_us_per_layer=12.0, runtime_fixed_us=1200.0,
    )
    return replace(exp, calibration=replace(selectors, analytical_costs=(replace(spec, **changes),)))


class AnalyticalCostTests(unittest.TestCase):
    def test_nominal_features_use_original_counts_without_default_overheads(self):
        exp = Experiment()
        feature = nominal_cost_features(exp, "prefill", [128], [0])
        plain = replace(exp, calibration=replace(exp.calibration, compute_efficiency=1,
                        bandwidth_efficiency=1, launch_overhead_us=0, framework_overhead_us=0))
        step = AnalyticalModel(plain).estimate_step("prefill", [128], [0])
        self.assertEqual(feature["attention_ms"], step.breakdown["attention"]["duration_ms"])
        self.assertEqual(feature["nonattention_ms"], sum(step.breakdown[key]["duration_ms"]
                         for key in ("linear", "misc", "embedding_and_output")))
        self.assertAlmostEqual(sum(feature[key] for key in ("attention_ms", "nonattention_ms",
                               "communication_ms", "dequant_ms")), step.duration_ms)

    def test_opt_in_formula_adds_fixed_costs_and_scales_attention_once(self):
        exp = calibrated()
        spec = exp.calibration.analytical_costs[0]
        feature = nominal_cost_features(exp, "prefill", [128], [0])
        step = AnalyticalModel(exp).estimate_step("prefill", [128], [0])
        expected = (1.4 * feature["nonattention_ms"] + 3 * feature["attention_ms"]
                    + feature["layers"] * 12 / 1000 + 1.2)
        self.assertAlmostEqual(step.duration_ms, expected)
        self.assertAlmostEqual(step.breakdown["attention"]["duration_ms"],
                               3 * feature["attention_ms"] + feature["layers"] * .012)
        self.assertEqual(step.breakdown["runtime_fixed"]["duration_ms"], 1.2)
        self.assertEqual(step.breakdown["cost_calibration"]["calibration_id"], spec.calibration_id)
        self.assertEqual(step.source, "analytical_cost_calibrated")
        self.assertIn("not a direct CPU", " ".join(step.assumptions))

    def test_legacy_efficiencies_and_overheads_are_not_double_charged(self):
        exp = calibrated()
        first = AnalyticalModel(exp).estimate_step("prefill", [128], [0])
        changed = replace(exp, calibration=replace(exp.calibration, compute_efficiency=.01,
                          bandwidth_efficiency=.2, launch_overhead_us=9000, framework_overhead_us=7000))
        second = AnalyticalModel(changed).estimate_step("prefill", [128], [0])
        self.assertEqual(first, second)

    def test_phase_execution_backend_and_runtime_are_independent_keys(self):
        exp = calibrated()
        changes = [
            {"prefill_execution_mode": "graph_configured"},
            {"prefill_attention_backend": "other-attention"},
            {"runtime_config_sha256": "c" * 64},
            {"analytical_cost_timing_boundary": "other_wall_v1"},
        ]
        for fields in changes:
            with self.subTest(fields=fields), self.assertRaises(SimulationError) as caught:
                AnalyticalModel(replace(exp, calibration=replace(exp.calibration, **fields))).estimate_step(
                    "prefill", [128], [0])
            self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")
        with self.assertRaises(SimulationError) as caught:
            AnalyticalModel(exp).estimate_step("decode", [1], [1024])
        self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_exact_model_and_device_identity_cannot_be_reused(self):
        exp = calibrated()
        variants = [replace(exp, strategy=replace(exp.strategy, model=replace(exp.strategy.model, hidden_size=256))),
                    replace(exp, infra=replace(exp.infra, devices=(replace(exp.infra.devices[0], compute_tflops=2),)))]
        for changed in variants:
            with self.assertRaises(SimulationError) as caught:
                AnalyticalModel(changed).estimate_step("prefill", [128], [0])
            self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_bounds_include_member_context_and_batch_not_only_token_sum(self):
        exp = calibrated()
        for queries, contexts in (([513], [0]), ([32], [1]), ([1] * 5, [0] * 5)):
            with self.subTest(queries=queries), self.assertRaises(SimulationError) as caught:
                AnalyticalModel(exp).estimate_step("prefill", queries, contexts)
            self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")
        decode = calibrated(phase="decode")
        AnalyticalModel(decode).estimate_step("decode", [1, 1], [1024, 1041])
        with self.assertRaises(SimulationError):
            AnalyticalModel(decode).estimate_step("decode", [1], [1042])

    def test_wall_cost_does_not_cross_unmeasured_core_policy_changes(self):
        exp = calibrated()
        for change in ({"prefix_cache": True}, {"chunked_prefill": True},
                       {"mixed_batching": True}, {"scheduler": "fifo"},
                       {"max_batch_tokens": 8192}, {"max_active_requests": 2}):
            changed = replace(exp, strategy=replace(exp.strategy, **change))
            with self.subTest(change=change), self.assertRaises(SimulationError) as caught:
                AnalyticalModel(changed).estimate_step("prefill", [128], [0])
            self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_mixed_and_recompute_have_no_unmeasured_fallback(self):
        model = AnalyticalModel(calibrated())
        for args in (("recompute", [16], [0], None), ("mixed", [1, 16], [1024, 0], ["decode", "prefill"])):
            with self.subTest(phase=args[0]), self.assertRaises(SimulationError) as caught:
                model.estimate_step(args[0], args[1], args[2], request_phases=args[3])
            self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_unmeasured_parallel_or_replica_layout_is_explicitly_rejected(self):
        exp = calibrated()
        devices = tuple(replace(exp.infra.devices[0], id=f"gpu{index}") for index in range(2))
        for field in ("tensor_parallel_size", "pipeline_parallel_size", "replicas"):
            changed = replace(exp, infra=replace(exp.infra, devices=devices),
                              strategy=replace(exp.strategy, **{field: 2}))
            with self.subTest(field=field), self.assertRaises(SimulationError) as caught:
                AnalyticalModel(changed)
            self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_invalid_coefficients_duplicate_domains_and_provenance_rejected(self):
        for changes in ({"attention_scale": -1.}, {"nonattention_scale": float("nan")},
                        {"runtime_fixed_us": -1.}, {"min_batch_size": 3, "max_batch_size": 2},
                        {"training_dataset_sha256": "not-a-sha"}, {"training_record_ids": ("same", "same")},
                        {"calibration_id": ""}):
            with self.subTest(changes=changes), self.assertRaises(SimulationError):
                AnalyticalModel(calibrated(**changes))
        exp = calibrated()
        with self.assertRaises(SimulationError):
            AnalyticalModel(replace(exp, calibration=replace(exp.calibration,
                analytical_costs=exp.calibration.analytical_costs * 2)))

    def test_fixture_opt_in_and_measured_profile_exclusivity(self):
        exp = calibrated()
        with self.assertRaises(SimulationError) as caught:
            AnalyticalModel(replace(exp, calibration=replace(exp.calibration, allow_test_fixture=False)))
        self.assertEqual(caught.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        with self.assertRaises(SimulationError):
            AnalyticalModel(replace(exp, calibration=replace(exp.calibration, backend="synthetic")))

    def test_embedded_roundtrip_and_provenance_do_not_require_a_mutable_file(self):
        exp = calibrated()
        loaded = experiment_from_dict(asdict(exp))
        self.assertEqual(AnalyticalModel(exp).estimate_step("prefill", [128], [0]),
                         AnalyticalModel(loaded).estimate_step("prefill", [128], [0]))
        provenance = AnalyticalModel(loaded).provenance
        self.assertFalse(provenance["hardware_validated"])
        self.assertEqual(provenance["source"]["kind"], "test_fixture")
        self.assertIn("b" * 64, provenance["training_dataset_sha256"])

    def test_explicit_runtime_identity_is_required(self):
        exp = calibrated()
        for field in ("runtime_name", "runtime_version", "runtime_config_sha256",
                      "analytical_cost_timing_boundary"):
            value = "" if field == "runtime_config_sha256" else "unspecified"
            with self.subTest(field=field), self.assertRaises(SimulationError):
                AnalyticalModel(replace(exp, calibration=replace(exp.calibration, **{field: value})))


if __name__ == "__main__":
    unittest.main()
