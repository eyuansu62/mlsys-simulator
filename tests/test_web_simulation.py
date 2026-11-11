"""Behavior checks for the UI adapter, not a second performance model."""

from copy import deepcopy
import math
import unittest

from mlsys_sim.config import experiment_from_dict
from mlsys_sim.engine import simulate
from mlsys_sim.web_simulation import (build_experiment, memory_chart, presets,
                                      run_web_simulation, throughput_chart)


class WebSimulationTests(unittest.TestCase):
    def configuration(self):
        data = presets()["defaults"]
        data["workload"].update(requests=6, input_tokens=16, output_tokens=4)
        return data

    def test_preset_qwen_query_geometry_and_tied_storage_are_preserved(self):
        model = next(row["model"] for row in presets()["models"] if row["id"] == "qwen3-4b")
        self.assertEqual(model["hidden_size"], 2560)
        self.assertEqual(model["num_attention_heads"] * model["head_dim"], 4096)
        self.assertTrue(model["tie_word_embeddings"])
        self.assertTrue(model["qk_norm"] and model["final_norm"])

    def test_default_presets_are_independent(self):
        first = presets()
        first["defaults"]["workload"]["requests"] = 1
        first["models"][0]["model"]["hidden_size"] = 1
        self.assertEqual(presets()["defaults"]["workload"]["requests"], 48)
        self.assertEqual(presets()["models"][0]["model"]["hidden_size"], 1024)

    def test_adapter_reports_original_engine_values(self):
        payload = self.configuration()
        response = run_web_simulation(payload)
        self.assertTrue(response["ok"], response)
        actual = simulate(experiment_from_dict(response["experiment"]))
        self.assertEqual(response["result"], actual.result)
        self.assertEqual(response["requests"], actual.requests)
        self.assertEqual(response["charts"]["latency"]["e2e_ms"], actual.result["latency_ms"]["end_to_end"])
        self.assertEqual(response["meta"]["execution_outcome"], "COMPLETED")
        self.assertFalse(response["meta"]["hardware_validated"])

    def test_one_output_has_missing_tpot(self):
        payload = self.configuration()
        payload["workload"]["output_tokens"] = 1
        response = run_web_simulation(payload)
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["charts"]["latency"]["tpot_ms"], {"p50": None, "p95": None, "p99": None})

    def test_fixed_count_poisson_is_seeded_and_never_clipped(self):
        payload = self.configuration()
        payload["workload"].update(requests=20, arrival="poisson")
        first, provenance = build_experiment(payload)
        again, _ = build_experiment(payload)
        self.assertEqual(first, again)
        self.assertEqual(len(first["workload"]), 20)
        self.assertGreater(first["workload"][0]["arrival_ms"], 0)
        self.assertTrue(all(left["arrival_ms"] < right["arrival_ms"]
                            for left, right in zip(first["workload"], first["workload"][1:])))
        payload["workload"]["seed"] += 1
        different, _ = build_experiment(payload)
        self.assertNotEqual(first["workload"], different["workload"])
        self.assertEqual(provenance["workload_rule"], "fixed_request_count_then_drain")

    def test_burst_and_periodic_keep_exact_requested_count(self):
        payload = self.configuration()
        for arrival in ("burst", "periodic"):
            with self.subTest(arrival=arrival):
                payload["workload"].update(arrival=arrival, requests=7, rate_rps=10)
                experiment, _ = build_experiment(payload)
                expected = [0] * 7 if arrival == "burst" else list(range(0, 700, 100))
                self.assertEqual([row["arrival_ms"] for row in experiment["workload"]], expected)

    def test_invalid_input_never_falls_back_to_defaults(self):
        invalid = [("workload", "requests", 0), ("workload", "requests", 501),
                   ("workload", "requests", 2.0), ("workload", "input_tokens", True),
                   ("workload", "output_tokens", 513), ("workload", "seed", 2**60),
                   ("workload", "rate_rps", float("nan")), ("hardware", "memory_gib", float("inf")),
                   ("strategy", "mixed_batching", "false"), ("memory", "kv_pool_gib", -1)]
        for section, name, value in invalid:
            with self.subTest(section=section, name=name, value=value):
                payload = self.configuration()
                payload[section][name] = value
                result = run_web_simulation(payload)
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"]["code"], "INVALID_WEB_CONFIG")

    def test_no_arbitrary_profile_or_path_input(self):
        for patch in ({"profile_path": "/tmp/example"}, {"calibration": {"profile_path": "/tmp/example"}}):
            payload = {**self.configuration(), **patch}
            result = run_web_simulation(payload)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["code"], "INVALID_WEB_CONFIG")

    def test_static_memory_infeasible_is_an_error_with_original_experiment(self):
        payload = self.configuration()
        payload["model_id"] = "qwen3-4b"
        payload["hardware"]["memory_gib"] = 0.25
        response = run_web_simulation(payload)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "STATIC_MEMORY_INFEASIBLE")
        self.assertIn("experiment", response)
        self.assertNotIn("charts", response)

    def test_custom_model_uses_core_geometry_validation(self):
        payload = self.configuration()
        del payload["model_id"]
        payload["model"] = {"name": "custom", "num_attention_heads": 3, "num_kv_heads": 2}
        response = run_web_simulation(payload)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "INVALID_CONFIG")

    def test_tokens_in_drain_are_counted_at_actual_emission_times(self):
        requests = [{"token_times_ms": [10, 100, 200]}, {"token_times_ms": [250]}]
        points = throughput_chart(requests, 250, bins=5)
        self.assertEqual([row["tokens"] for row in points], [1, 0, 1, 0, 2])
        self.assertEqual(points[-1]["end_ms"], 250)
        self.assertEqual(sum(row["tokens"] for row in points), 4)
        self.assertAlmostEqual(sum(row["tokens_per_s"] * (row["end_ms"] - row["start_ms"]) / 1000
                                   for row in points), 4)
        self.assertEqual(points[-1]["tokens_per_s"], 40)
        self.assertEqual(throughput_chart([], 0), [])

    def test_memory_keeps_pool_and_logical_usage_separate_and_preserves_peaks(self):
        response = run_web_simulation(self.configuration())
        self.assertTrue(response["ok"], response)
        points = response["charts"]["memory"]
        self.assertEqual(points[0]["used_kv_bytes"], 0)
        self.assertEqual(points[-1]["used_kv_bytes"], 0)
        self.assertGreater(max(point["used_kv_bytes"] for point in points), 0)
        self.assertTrue(all(point["budget_peak_bytes"] == point["fixed_budget_bytes"]
                            + point["workspace_peak_bytes"] for point in points))
        self.assertEqual(max(point["budget_peak_bytes"] for point in points),
                         response["result"]["memory"][0]["budget_peak_bytes"])
        self.assertEqual(sum(row["tokens"] for row in response["charts"]["throughput"]), 24)

    def test_memory_thinning_preserves_logical_and_budget_maxima(self):
        base = {"device_id": "gpu0", "budget_peak_bytes": 100, "physical_peak_bytes": 90,
                "physical_reserved_bytes": 80, "fixed_budget_bytes": 90, "workspace_peak_bytes": 10,
                "weights_bytes": 40, "device_memory_bytes": 1000, "kv_used_bytes": 0,
                "kv_pool_reserved_bytes": 20}
        events = []
        for index in range(10000):
            row = {**base, "kv_used_bytes": 999 if index == 5001 else index % 19,
                   "budget_peak_bytes": 100 + index}
            events.append({"timestamp_ms": index, "memory": [row]})
        points, thinned = memory_chart([base], events, [{**base, "budget_peak_bytes": 10099}], 10000)
        self.assertTrue(thinned)
        self.assertLessEqual(len(points), 2048)
        self.assertEqual(max(point["used_kv_bytes"] for point in points), 999)
        self.assertEqual(max(point["budget_peak_bytes"] for point in points), 10099)

    def test_payload_is_not_modified(self):
        payload = self.configuration()
        before = deepcopy(payload)
        run_web_simulation(payload)
        self.assertEqual(payload, before)
        self.assertTrue(math.isfinite(run_web_simulation(payload)["meta"]["wall_elapsed_ms"]))


if __name__ == "__main__":
    unittest.main()
