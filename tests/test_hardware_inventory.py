"""Optional HIP discovery and strict source/software identity binding."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlsys_sim import DeviceSpec, Experiment, InfraSpec, ModelSpec, SimulationError, StrategySpec
from mlsys_sim.hardware import bind_inventory, probe_rocm, read_json


def fake_torch():
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(name=f"AMD test device {index}",
            total_memory=64 * 1024 * 1024, gcnArchName="gfx-test"),
        nccl=SimpleNamespace(version=lambda: (2, 18, 3)))
    return SimpleNamespace(__version__="fixture-torch", version=SimpleNamespace(hip="fixture-hip"), cuda=cuda)


class HardwareInventoryTests(unittest.TestCase):
    def probe(self, module):
        with patch("mlsys_sim.hardware.shutil.which", return_value=None), \
                patch("mlsys_sim.hardware._read_version", return_value=None):
            return probe_rocm(torch_module=module)

    def test_optional_torch_absence_is_diagnostic_without_cpu_measurement_fallback(self):
        with patch("mlsys_sim.hardware.shutil.which", return_value=None), \
                patch("mlsys_sim.hardware._read_version", return_value=None), \
                patch("mlsys_sim.hardware.importlib.import_module", side_effect=ImportError("No torch")):
            inventory = probe_rocm()
        self.assertEqual(inventory["status"], "UNAVAILABLE")
        self.assertEqual(inventory["devices"], [])
        self.assertEqual(inventory["issues"][0]["code"], "TORCH_UNAVAILABLE")
        self.assertIsNone(inventory["versions"]["rocm_version"])

    def test_nvidia_cuda_availability_does_not_prove_hip(self):
        module = fake_torch()
        module.version.hip = None
        inventory = self.probe(module)
        self.assertTrue(inventory["torch"]["cuda_available"])
        self.assertEqual(inventory["status"], "UNAVAILABLE")
        self.assertEqual(inventory["devices"], [])
        with self.assertRaises(SimulationError) as raised:
            bind_inventory(Experiment(), inventory, runtime_name="fixture", runtime_version="v1", allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "HARDWARE_UNAVAILABLE")

    def test_broken_optional_torch_import_is_a_readiness_diagnostic(self):
        with patch("mlsys_sim.hardware.shutil.which", return_value=None), \
                patch("mlsys_sim.hardware._read_version", return_value=None), \
                patch("mlsys_sim.hardware.importlib.import_module", side_effect=RuntimeError("HIP library unavailable")):
            inventory = probe_rocm()
        self.assertEqual(inventory["status"], "UNAVAILABLE")
        self.assertEqual(inventory["issues"][0]["code"], "TORCH_UNAVAILABLE")

    def test_injected_backend_is_fixture_and_collective_build_is_not_runtime_version(self):
        inventory = self.probe(fake_torch())
        self.assertEqual(inventory["status"], "READY")
        self.assertEqual(inventory["source"]["kind"], "test_fixture")
        self.assertEqual(inventory["torch"]["pytorch_collective_build_version"], "2.18.3")
        self.assertIsNone(inventory["versions"]["rccl_version"])
        with self.assertRaises(SimulationError):
            bind_inventory(Experiment(), inventory, runtime_name="fixture", runtime_version="v1")
        bound = bind_inventory(Experiment(), inventory, runtime_name="fixture", runtime_version="v1",
                               allow_test_fixture=True)
        self.assertEqual(bound.infra.devices[0].name, "AMD test device 0")
        self.assertEqual(bound.calibration.rccl_version, "not-applicable")
        self.assertEqual(bound.calibration.rocm_version, "unknown")

    def test_binding_retains_placement_and_never_mutates_original_identity(self):
        inventory = self.probe(fake_torch())
        exp = Experiment(infra=InfraSpec(devices=(DeviceSpec("logical-a"), DeviceSpec("logical-b"))),
                         strategy=StrategySpec(tensor_parallel_size=2))
        before = deepcopy(inventory)
        bound = bind_inventory(exp, inventory, runtime_name="fixture", runtime_version="v1",
                               require_rccl=True, allow_test_fixture=True)
        self.assertEqual([device.id for device in bound.infra.devices], ["logical-a", "logical-b"])
        self.assertEqual([device.name for device in bound.infra.devices], ["AMD test device 0", "AMD test device 1"])
        self.assertEqual(inventory, before)
        self.assertEqual(exp.infra.devices[0].name, "toy-device")

    def measured(self):
        inventory = self.probe(fake_torch())
        inventory["source"] = {"kind": "measured", "reference": "Declared identity contract test; no GPU run"}
        inventory["versions"].update(rocm_version="test-rocm-version", driver_version="test-driver-version")
        model = ModelSpec(name="dense-reference", revision="revision-1", weight_dtype="bfloat16",
                          kv_dtype="bfloat16", activation_dtype="bfloat16")
        return inventory, Experiment(strategy=StrategySpec(model=model))

    def test_required_versions_and_model_identity_cannot_be_inferred(self):
        inventory, exp = self.measured()
        for field in ("rocm_version", "driver_version"):
            invalid = deepcopy(inventory)
            invalid["versions"][field] = None
            with self.subTest(field=field), self.assertRaises(SimulationError) as raised:
                bind_inventory(exp, invalid, runtime_name="actual-runtime", runtime_version="v1")
            self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")
        with self.assertRaises(SimulationError) as raised:
            bind_inventory(exp, inventory, runtime_name="actual-runtime", runtime_version="v1", require_rccl=True)
        self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")
        with self.assertRaises(SimulationError):
            bind_inventory(Experiment(), inventory, runtime_name="actual-runtime", runtime_version="v1")
        bound = bind_inventory(exp, inventory, runtime_name="actual-runtime", runtime_version="v1")
        self.assertEqual(bound.strategy.model, exp.strategy.model)

    def test_malformed_inventory_and_non_boolean_overrides_fail_explicitly(self):
        inventory = self.probe(fake_torch())
        for change in ({"schema_version": True}, {"source": None}, {"torch": []}, {"versions": None}):
            with self.subTest(change=change), self.assertRaises(SimulationError):
                bind_inventory(Experiment(), {**inventory, **change}, runtime_name="fixture", runtime_version="v1",
                               allow_test_fixture=True)
        with self.assertRaises(SimulationError):
            bind_inventory(Experiment(), inventory, runtime_name="fixture", runtime_version="v1", allow_test_fixture="yes")
        swapped = deepcopy(inventory)
        swapped["devices"].reverse()
        with self.assertRaises(SimulationError):
            bind_inventory(Experiment(), swapped, runtime_name="fixture", runtime_version="v1", allow_test_fixture=True)

    def test_selected_rccl_library_has_its_own_version_and_content_hash(self):
        class Version:
            def __call__(self, pointer):
                pointer._obj.value = 23007
                return 0
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "librccl.so"
            library.write_bytes(b"fixture-library-bytes")
            with patch("mlsys_sim.hardware.ctypes.CDLL", return_value=SimpleNamespace(ncclGetVersion=Version())), \
                    patch("mlsys_sim.hardware.shutil.which", return_value=None), \
                    patch("mlsys_sim.hardware._read_version", return_value=None):
                inventory = probe_rocm(torch_module=fake_torch(), rccl_library=library)
            self.assertEqual(inventory["versions"]["rccl_version"], "2.30.7")
            self.assertEqual(inventory["rccl_library"]["sha256"], hashlib.sha256(library.read_bytes()).hexdigest())
            self.assertEqual(inventory["source"]["kind"], "test_fixture")
            self.assertIn("must_be_established", inventory["rccl_library"]["loaded_runtime_equivalence"])

    def test_collection_json_rejects_duplicates_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            for content in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(SimulationError):
                    read_json(path)
            path.write_text(json.dumps({"schema_version": 1, "value": 2}))
            self.assertEqual(read_json(path)["value"], 2)


if __name__ == "__main__":
    unittest.main()
