"""Confirmed catalog identities cannot be inferred from parameter-count aliases."""

from dataclasses import asdict
import json
from pathlib import Path
import unittest

from mlsys_sim.agent_catalog import (
    CATALOG_VERSION, get_agent_catalog, resolve_agent_device, resolve_agent_model,
    validate_agent_catalog_version,
)
from mlsys_sim.qwen3 import model_from_qwen3_config
from mlsys_sim.schema import SimulationError


class AgentCatalogTests(unittest.TestCase):
    def assert_code(self, code, function, *args):
        with self.assertRaises(SimulationError) as raised:
            function(*args)
        self.assertEqual(raised.exception.code, code)

    def test_geometry_is_exactly_the_preexisting_course_configuration(self):
        course = json.loads((Path(__file__).resolve().parents[1]
                             / "examples/course/chapter1.plan.json").read_text())
        ids = {"qwen3_0_6b": "qwen3-0.6b", "qwen3_1_7b": "qwen3-1.7b",
               "qwen3_4b": "qwen3-4b"}
        for declaration in course["models"]:
            with self.subTest(model=declaration["id"]):
                expected = model_from_qwen3_config(declaration["hf_config"],
                                                  name=declaration["name"], revision=declaration["revision"])
                actual = resolve_agent_model(ids[declaration["id"]], "BF16", "BF16", 40960)
                self.assertEqual(asdict(actual), asdict(expected))

    def test_query_width_and_tied_storage_flags_are_not_simplified(self):
        small = resolve_agent_model("qwen3-0.6b", "BF16", "BF16", 32768)
        large = resolve_agent_model("qwen3-4b", "BF16", "BF16", 32768)
        self.assertEqual((small.hidden_size, small.query_projection_size), (1024, 2048))
        self.assertEqual((large.hidden_size, large.query_projection_size), (2560, 4096))
        for model in (small, large):
            self.assertTrue(model.tie_word_embeddings and model.qk_norm and model.final_norm)
            self.assertEqual(model.weight_dtype, "bfloat16")
            self.assertEqual(model.kv_dtype, "bfloat16")

    def test_unknown_and_placeholder_models_are_never_size_guessed(self):
        for identity in ("llm-3b", "llm-7b", "llm-32b", "llm-1.7b", "Qwen3-4B",
                         "qwen3-4b ", "qwen3-0.5b", None, {}):
            with self.subTest(identity=identity):
                self.assert_code("UNKNOWN_MODEL_ID", resolve_agent_model, identity, "BF16", "BF16", 1024)

    def test_only_declared_precision_combinations_are_available(self):
        for weight, kv in (("FP8", "BF16"), ("BF16", "FP8"), ("FP16", "BF16"),
                           ("bf16", "BF16"), (None, "BF16")):
            with self.subTest(weight=weight, kv=kv):
                self.assert_code("UNSUPPORTED_PRECISION", resolve_agent_model,
                                 "qwen3-1.7b", weight, kv, 1024)

    def test_context_length_is_explicit_and_bounded_by_source(self):
        for value in (0, -1, True, 40961, 40960.0, "40960", None):
            with self.subTest(value=value):
                self.assert_code("UNSUPPORTED_CONTEXT_LENGTH", resolve_agent_model,
                                 "qwen3-1.7b", "BF16", "BF16", value)

    def test_hardware_is_explicit_nominal_not_a_live_device_probe(self):
        device = resolve_agent_device("rx9070xt", "node0/gpu3")
        self.assertEqual(device.id, "node0/gpu3")
        self.assertEqual(device.memory_bytes, 16 * 1024**3)
        self.assertEqual(device.compute_tflops, 195)
        self.assertEqual(device.memory_bandwidth_gbps, 640)
        catalog = get_agent_catalog()
        self.assertEqual(catalog["hardware"][0]["evidence"], "course_nominal")
        self.assertFalse(catalog["hardware_validated"])
        self.assertEqual(catalog["calibration"]["kind"], "uncalibrated")
        self.assertIsNone(catalog["calibration"]["in_domain"])

    def test_unconfirmed_hardware_and_device_identifiers_fail_explicitly(self):
        for identity in ("w7900d", "rx7900xtx", "RX9070XT", None, {}):
            self.assert_code("UNKNOWN_HARDWARE_ID", resolve_agent_device, identity, "gpu0")
        for identity in ("", " gpu0", "gpu0\n", True, None, "x" * 257):
            self.assert_code("INVALID_DEVICE_ID", resolve_agent_device, "rx9070xt", identity)

    def test_version_placeholders_are_not_executable_catalogs(self):
        validate_agent_catalog_version(CATALOG_VERSION)
        for version in ("to-be-agreed", "EXAMPLE_ONLY", "", None, {}, CATALOG_VERSION + "-other"):
            self.assert_code("CATALOG_VERSION_MISMATCH", validate_agent_catalog_version, version)

    def test_catalog_is_json_ready_independent_and_has_no_unconfirmed_topology(self):
        first = get_agent_catalog()
        self.assertEqual(first["catalog_version"], CATALOG_VERSION)
        self.assertIsNone(first["communication"])
        self.assertIn("communication", first["unsupported_fields"])
        self.assertEqual(first["support"]["tp_degrees"], [1])
        self.assertEqual(first["support"]["max_replicas_per_deployment"], 16)
        self.assertEqual({item["id"] for item in first["models"]},
                         {"qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"})
        json.dumps(first, allow_nan=False)
        first["models"][0]["geometry"]["hidden_size"] = 1
        first["hardware"][0]["capacity_bytes"] = 1
        second = get_agent_catalog()
        self.assertEqual(second["models"][0]["geometry"]["hidden_size"], 1024)
        self.assertEqual(second["hardware"][0]["capacity_bytes"], 16 * 1024**3)


if __name__ == "__main__":
    unittest.main()
