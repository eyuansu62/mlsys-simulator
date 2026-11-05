"""CPU producer/protocol checks; these are not HSA hardware observations."""

from copy import deepcopy
import ctypes
import io
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_rocm_native_backing_audit import _load_tool


capture = _load_tool("hsa_signal_backing_capture")


class CaptureFixture:
    def __init__(self, root):
        self.root = root
        self.provider = root / "provider.so"
        self.runtime = root / "runtime.so"
        self.header = root / "amd_hsa_signal.h"
        for path in (self.provider, self.runtime, self.header):
            path.write_bytes(b"CPU fixture artifact, not an installed GPU library")
        self.live = [{"signal_handle": 4096, "generation": 1, "create_cid": 101,
            "create_seq": 202, "create_epoch": 10001, "create_tid": threading.get_native_id(),
            "initial_value": 0, "num_consumers": 0, "consumers_observed": True, "consumer_agents": []}]
        self.after = None
        self.exports = 0
        self.samples = []
        self.controls = []
        self.mode = None

        def export(path):
            self.exports += 1
            rows = self.live if self.exports == 1 or self.after is None else self.after
            data = {"schema_version": 1, "kind": "observed_live_hsa_signal_generations",
                "pid": os.getpid(), "snapshot_monotonic_raw_ns": 1000 + self.exports,
                "event_cursor": 100 * self.exports, "dropped_events": 0,
                "reused_live_handle_count": 0, "unknown_destroy_count": 2,
                "generations": rows, "complete_workspace": False, "hardware_validated": False}
            Path(os.fsdecode(path)).write_text(json.dumps(data))
            return 0

        self.bridge = SimpleNamespace(mlsys_probe_write_live_signals=export)
        owner = self

        class Provider:
            def __init__(self, *args):
                self.initialization = {"ready": True, "process_id": os.getpid(),
                    "thread_id": threading.get_native_id(), "abi": {"amd_signal_size": 64},
                    "compiled_source_path": str(Path(capture.__file__).with_name("hsa_signal_backing_probe.cpp")),
                    "api_bindings": {"hsa_amd_pointer_info": {"dladdr_success": True,
                        "module_path": str(owner.runtime)}}}

            def sample(self, handle):
                owner.samples.append(handle)
                out = {"process_id": os.getpid(), "thread_id": threading.get_native_id(),
                    "signal_handle": handle, "abi_signal_address": handle,
                    "api_calls": [owner.call("hsa_amd_pointer_info"), owner.call("hsa_agent_get_info"),
                                  owner.call("hsa_amd_signal_get_event_id")]}
                out["api_calls"][-1]["output"]["event_id"] = 3 if owner.mode == "hardware" else 0
                if owner.mode == "uncertain":
                    out["api_calls"][0]["before_cursor"] = 2**64 - 1
                if owner.mode == "wrong_handle":
                    out["signal_handle"] += 64
                if owner.mode == "wrong_thread":
                    out["thread_id"] += 1
                if owner.mode == "invented_cid":
                    out["api_calls"][0]["cid"] = 1234
                return out

            def control(self, initial, consumers):
                owner.controls.append((initial, consumers))
                out = {"process_id": os.getpid(), "thread_id": threading.get_native_id(),
                    "create_call": owner.call("hsa_signal_create"),
                    "samples": [owner.sample(8192), owner.sample(8192)],
                    "destroy_call": owner.call("hsa_signal_destroy")}
                if owner.mode == "failed_control_destroy":
                    out["destroy_call"]["output"]["status"] = 7
                return out

        self.factory = Provider

    @staticmethod
    def call(name):
        return {"api": name, "before_cursor": 100, "after_cursor": 102,
            "started_monotonic_raw_ns": 10, "ended_monotonic_raw_ns": 11,
            "input": {"output_address": 32768},
            "output": {"status_type": 2, "status": 0, "event_id": 3}}

    def sample(self, handle):
        return {"process_id": os.getpid(), "thread_id": threading.get_native_id(), "signal_handle": handle,
            "api_calls": [self.call("hsa_amd_pointer_info"), self.call("hsa_agent_get_info"),
                          self.call("hsa_amd_signal_get_event_id")]}

    def collect(self, output="capture"):
        with patch.object(capture, "_Provider", self.factory):
            return capture.capture_hsa_signal_backings(native_bridge=self.bridge, provider_path=self.provider,
                rocr_path=self.runtime, abi_header_path=self.header, output_dir=self.root / output,
                target_epochs=[10001], source_fence={"actual_fence_fixture": "not verified by producer"},
                source_scope={"actual_scope_fixture": "not verified by producer"})


class HsaSignalCaptureTests(unittest.TestCase):
    def test_two_samples_and_real_live_exports_never_grant_hardware_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            report = fixture.collect()
            self.assertEqual(report["status"], "CAPTURED", report["issues"])
            self.assertEqual(fixture.exports, 2)
            self.assertEqual(fixture.samples, [4096, 4096])
            self.assertEqual(fixture.controls, [(0, [])])
            self.assertEqual(report["records"][0]["create_cid"], 101)
            self.assertEqual(report["before_live"]["generations"], report["after_live"]["generations"])
            self.assertFalse(report["complete_workspace"])
            self.assertFalse(report["hardware_validated"])
            self.assertEqual(json.loads((Path(temporary) / "capture/capture.json").read_text()), report)

    def test_nonzero_event_id_is_preserved_for_independent_fail_closed_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            fixture.mode = "hardware"
            report = fixture.collect()
            self.assertEqual(report["status"], "PARTIAL")
            self.assertEqual(report["ordinary_signal_control"]["samples"][0]["api_calls"][-1]["output"]["event_id"], 3)
            self.assertEqual(report["records"][0]["samples"][0]["api_calls"][-1]["output"]["event_id"], 3)
            self.assertNotIn("pointer_info_cid", report["records"][0]["samples"][0])

    def test_hardware_without_namespace_does_not_spoil_eventless_or_not_applicable_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            self.assertEqual(fixture.collect()["status"], "CAPTURED")
            fixture.mode = "hardware"
            report = fixture.collect("hardware")
            self.assertEqual(report["status"], "PARTIAL")
            self.assertEqual(report["issues"][0]["code"], "KFD_EVENT_EVIDENCE_PARTIAL")

    def test_no_live_formal_generation_keeps_auditable_not_applicable_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            fixture.live[0]["create_epoch"] = 0
            with patch.object(capture, "_Provider", side_effect=AssertionError("Provider must not run")):
                report = capture.capture_hsa_signal_backings(native_bridge=fixture.bridge,
                    provider_path=fixture.provider, rocr_path=fixture.runtime, abi_header_path=fixture.header,
                    output_dir=fixture.root / "capture", target_epochs=[10001], source_fence={}, source_scope={})
            self.assertEqual(report["status"], "NO_LIVE_FORMAL_TARGETS")
            self.assertEqual(report["records"], [])
            self.assertIsNone(report["ordinary_signal_control"])
            self.assertEqual(fixture.exports, 2)

    def test_changed_generation_after_queries_is_partial_with_original_facts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            fixture.after = deepcopy(fixture.live)
            fixture.after[0]["create_cid"] += 1
            report = fixture.collect()
            self.assertEqual(report["status"], "PARTIAL")
            self.assertEqual(report["records"][0]["create_cid"], 101)
            self.assertNotEqual(report["before_live"]["generations"], report["after_live"]["generations"])

    def test_uncertain_cursor_wrong_handle_thread_or_invented_cid_remains_partial(self):
        for mode in ("uncertain", "wrong_handle", "wrong_thread", "invented_cid"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = CaptureFixture(Path(temporary))
                fixture.mode = mode
                report = fixture.collect()
                self.assertEqual(report["status"], "PARTIAL")
                self.assertTrue(report["issues"])
                self.assertEqual(fixture.exports, 2)

    def test_failed_own_control_destroy_is_retained_not_relabelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            fixture.mode = "failed_control_destroy"
            report = fixture.collect()
            self.assertEqual(report["status"], "PARTIAL")
            self.assertEqual(report["ordinary_signal_control"]["destroy_call"]["output"]["status"], 7)
            self.assertEqual(fixture.samples, [])

    def test_constructor_and_generation_metadata_cannot_be_backfilled(self):
        for name, value in (("consumers_observed", False), ("generation", True), ("num_consumers", 5)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = CaptureFixture(Path(temporary))
                fixture.live[0][name] = value
                report = fixture.collect()
                self.assertEqual(report["status"], "PARTIAL")
                self.assertEqual(fixture.samples, [])

    def test_existing_output_and_invalid_target_epochs_do_not_execute_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            fixture.collect()
            with self.assertRaises(FileExistsError):
                fixture.collect()
            self.assertEqual(fixture.exports, 2)
            for epochs in ([], [True], [10001, 10001]):
                with self.assertRaises(ValueError):
                    capture.capture_hsa_signal_backings(native_bridge=fixture.bridge,
                        provider_path=fixture.provider, rocr_path=fixture.runtime, abi_header_path=fixture.header,
                        output_dir=fixture.root / "invalid", target_epochs=epochs, source_fence={}, source_scope={})
            self.assertFalse((fixture.root / "invalid").exists())

    def test_loaded_module_alias_is_preserved_beside_actual_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            alias = fixture.root / "runtime-alias.so"
            alias.symlink_to(fixture.runtime)
            parent = fixture.factory

            class AliasProvider(parent):
                def __init__(self, *args):
                    super().__init__(*args)
                    self.initialization["api_bindings"]["hsa_amd_pointer_info"]["module_path"] = str(alias)

            fixture.factory = AliasProvider
            report = fixture.collect()
            aliases = [entry for entry in report["runtime_dsos"] if entry["loaded_path"] == str(alias)]
            self.assertEqual(len(aliases), 1)
            self.assertEqual(aliases[0]["path"], str(fixture.runtime.resolve()))


class KfdSourceCaptureTests(unittest.TestCase):
    def test_artifact_size_counts_actual_read_bytes_without_virtual_stat_size(self):
        class VirtualFile:
            def resolve(self, strict):
                return self

            def open(self, mode):
                return io.BytesIO(b"short sysfs bytes\n")

            def stat(self):
                raise AssertionError("Virtual file stat size is not its content size")

            def __str__(self):
                return "/sys/fixture-virtual-file"

        actual = capture._artifact(VirtualFile())
        self.assertEqual(actual["size_bytes"], len(b"short sysfs bytes\n"))

    @staticmethod
    def fixture(root):
        paths = {}
        for name in ("probe_source", "native_probe_source", "probe_library", "hsa_runtime", "libc", "kfd_header", "drm_header"):
            path = root / name
            path.write_bytes(("actual CPU fixture bytes " + name).encode())
            paths[name] = capture._artifact(path)
        receipt = root / "build-receipt.json"
        receipt.write_text('{"fixture":true}')
        source = {name + "_sha256": value["sha256"] for name, value in paths.items()}
        identity = {"kernel_release": "fixture-kernel", "amdgpu_module_build_id": "a" * 40,
            "device_nodes": [{"gpu_id": 17, "rdev_major": 226, "rdev_minor": 128,
                              "pci_bus_id": "0000:11:00.0"}]}
        source.update(identity)
        source.update(artifacts=paths, abi={"fixture_abi": 1}, build_receipt_artifact=capture._artifact(receipt))
        manifest = {"schema_version": 1, "kind": "cold_hsa_kfd_namespace_control_manifest",
            "source_binding": source, "build_receipt": capture._artifact(receipt),
            "create_cid": 1, "event_query_cid": 2, "kfd_create_event_call_id": 3, "target_epochs": [8001]}
        for key in ("native_report", "kfd_report", "control_jsonl"):
            path = root / key
            path.write_text('{"cpu_fixture":true}')
            manifest[key] = capture._artifact(path)
        path = root / "namespace.json"
        path.write_text(json.dumps(manifest))
        snapshot = {"compiled_source_path": paths["probe_source"]["path"], "abi": source["abi"],
            "libc_bindings": {"ioctl": {"address": 1000, "dladdr_success": True,
                                        "module_path": paths["libc"]["path"]}}}
        native = SimpleNamespace(_name=paths["probe_library"]["path"])
        init = {"api_bindings": {"hsa_amd_pointer_info": {"module_path": paths["hsa_runtime"]["path"]}}}
        return manifest, path, snapshot, native, init, identity

    def test_current_source_binding_is_rehashed_and_identity_is_fresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = self.fixture(Path(temporary))
            manifest, path, snapshot, native, init, identity = values
            loaded, artifact = capture._namespace_manifest(path)
            self.assertEqual(loaded, manifest)
            self.assertEqual(artifact["sha256"], capture._artifact(path)["sha256"])
            with patch.object(capture, "_system_identity", return_value=identity) as observed:
                binding = capture._kfd_source_binding(snapshot, loaded, native, init)
            observed.assert_called_once_with()
            self.assertEqual(binding["kernel_release"], identity["kernel_release"])
            self.assertEqual(binding["probe_library_sha256"], manifest["source_binding"]["probe_library_sha256"])
            self.assertEqual(binding["artifacts"]["probe_source"]["loaded_path"], snapshot["compiled_source_path"])

    def test_changed_raw_artifact_is_rejected_without_loading_its_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, path, *_ = self.fixture(Path(temporary))
            Path(manifest["native_report"]["path"]).write_text("changed raw bytes")
            with self.assertRaisesRegex(ValueError, "artifact bytes differ"):
                capture._namespace_manifest(path)

    def test_wrong_current_library_kernel_buildid_or_compiled_abi_never_copies_cold_labels(self):
        for change in ("library", "kernel", "buildid", "abi"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                manifest, _, snapshot, native, init, identity = self.fixture(Path(temporary))
                if change == "library":
                    Path(native._name).write_bytes(b"different actually loaded library")
                elif change == "kernel":
                    identity["kernel_release"] = "new-kernel"
                elif change == "buildid":
                    identity["amdgpu_module_build_id"] = "b" * 40
                else:
                    snapshot["abi"] = {"fixture_abi": 2}
                with patch.object(capture, "_system_identity", return_value=identity), self.assertRaises(ValueError):
                    capture._kfd_source_binding(snapshot, manifest, native, init)

    def test_source_binding_keeps_actual_loaded_alias_without_basename_matching(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, snapshot, native, init, identity = self.fixture(Path(temporary))
            alias = Path(temporary) / "libc-alias"
            alias.symlink_to(Path(manifest["source_binding"]["artifacts"]["libc"]["path"]))
            snapshot["libc_bindings"]["ioctl"]["module_path"] = str(alias)
            with patch.object(capture, "_system_identity", return_value=identity):
                binding = capture._kfd_source_binding(snapshot, manifest, native, init)
            self.assertEqual(binding["artifacts"]["libc"]["loaded_path"], str(alias))
            self.assertNotEqual(binding["artifacts"]["libc"]["path"], str(alias))

    def test_current_sysfs_gpu_render_pci_and_elf_build_id_are_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = root / "sys/class/kfd/kfd/topology/nodes/1"
            node.mkdir(parents=True)
            (node / "gpu_id").write_text("17\n")
            (node / "properties").write_text("drm_render_minor 128\n")
            dev = root / "sys/class/drm/renderD128/device"
            dev.mkdir(parents=True)
            (dev.parent / "dev").write_text("22:128\n")
            (dev / "uevent").write_text("PCI_SLOT_NAME=0000:11:00.0\n")
            notes = root / "sys/module/amdgpu/notes"
            notes.mkdir(parents=True)
            note = struct.pack(("<" if sys.byteorder == "little" else ">") + "III", 4, 20, 3) + b"GNU\0" + bytes(range(20))
            (notes / ".note.gnu.build-id").write_bytes(note)
            def fake(path):
                # Keep fixture dev_t positive on older macOS Python builds.
                return SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(22, 128))
            actual = capture._system_identity(sys_root=root / "sys", dev_root=root / "dev",
                                             kernel_release="actual-fixture-kernel", stat_device=fake,
                                             observations_dir=root / "environment")
            self.assertEqual(actual["kernel_release"], "actual-fixture-kernel")
            self.assertEqual(actual["amdgpu_module_build_id"], bytes(range(20)).hex())
            self.assertEqual(actual["device_nodes"][0]["rdev_minor"], 128)
            self.assertEqual(actual["device_nodes"][0]["gpu_id"], 17)
            observations = actual["system_observation_artifacts"]
            self.assertEqual(len(observations), 6)
            captured = json.loads(Path(observations[0]["snapshot_artifact"]["path"]).read_text())
            self.assertEqual(bytes.fromhex(captured["contents_hex"]), b"17\n")
            self.assertEqual(captured["contents_size_bytes"], 3)
            (dev / "uevent").write_text("PCI_SLOT_NAME=not-a-PCI-identity\n")
            with self.assertRaisesRegex(ValueError, "PCI identity"):
                capture._system_identity(sys_root=root / "sys", dev_root=root / "dev",
                                         kernel_release="actual-fixture-kernel", stat_device=fake)

    def test_kfd_capture_preserves_original_refs_and_all_query_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CaptureFixture(root)
            fixture.mode = "hardware"
            manifest, path, snapshot, native, _, identity = self.fixture(root)
            source = manifest["source_binding"]
            source["artifacts"]["hsa_runtime"] = capture._artifact(fixture.runtime)
            source["hsa_runtime_sha256"] = source["artifacts"]["hsa_runtime"]["sha256"]
            path.write_text(json.dumps(manifest))
            fixture.bridge._name = native._name

            def cursor():
                return 98

            def export(name):
                data = {**deepcopy(snapshot), "schema_version": 1, "kind": "passive_kfd_event_api_tape",
                    "pid": os.getpid(), "clock": "CLOCK_MONOTONIC_RAW", "write_complete": True,
                    "complete_workspace": False, "hardware_validated": False,
                    **{key: 0 for key in ("dropped_rows", "read_errors", "forward_errors", "reentrant_calls", "active_mutators")}}
                Path(os.fsdecode(name)).write_text(json.dumps(data))
                return 0

            fixture.bridge.mlsys_kfd_probe_cursor = cursor
            fixture.bridge.mlsys_kfd_probe_write = export
            with patch.object(capture, "_Provider", fixture.factory), patch.object(capture, "_system_identity", return_value=identity):
                report = capture.capture_hsa_signal_backings(native_bridge=fixture.bridge,
                    provider_path=fixture.provider, rocr_path=fixture.runtime, abi_header_path=fixture.header,
                    output_dir=root / "capture", target_epochs=[10001], source_fence={}, source_scope={},
                    namespace_control=path)
            self.assertEqual(report["status"], "CAPTURED", report["issues"])
            evidence = report["kfd_event_evidence"]
            self.assertEqual(evidence["namespace_control"], manifest)
            self.assertEqual(evidence["report_artifact"], evidence["post_query_report_artifact"])
            self.assertNotIn("events", evidence)
            self.assertNotIn("native_report", evidence)
            self.assertEqual(evidence["source_binding"]["kernel_release"], identity["kernel_release"])
            for sample in report["records"][0]["samples"]:
                self.assertEqual(sample["kfd_window"]["before_cursor"], 98)
                self.assertEqual(sample["kfd_window"]["after_cursor"], 98)
            self.assertEqual(report["issues"], [])

    def test_truncated_or_duplicate_module_buildid_is_rejected(self):
        note = struct.pack(("<" if sys.byteorder == "little" else ">") + "III", 4, 20, 3) + b"GNU\0" + bytes(range(20))
        for raw in (b"not an ELF note", note[:-1], note + note):
            with self.assertRaises(ValueError):
                capture._gnu_build_id(raw)

    def test_sample_kfd_window_keeps_real_equal_cursor_without_fake_correlation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            provider = fixture.factory()
            observer = SimpleNamespace(cursor=lambda: 98)
            sample = capture._sample_with_kfd(provider, 4096, observer)
            self.assertEqual(sample["kfd_window"]["before_cursor"], 98)
            self.assertEqual(sample["kfd_window"]["after_cursor"], 98)
            self.assertNotIn("cid", sample["kfd_window"])


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("c++"), "Optional provider C ABI needs Linux C++")
class HsaSignalProviderCppTests(unittest.TestCase):
    def test_actual_provider_uses_public_outputs_and_destroys_only_its_own_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inc = root / "include/hsa"
            inc.mkdir(parents=True)
            (inc / "hsa.h").write_text("""#pragma once
#include <stdint.h>
typedef struct { uint64_t handle; } hsa_agent_t;
typedef struct { uint64_t handle; } hsa_signal_t;
typedef int64_t hsa_signal_value_t;
typedef enum { HSA_STATUS_SUCCESS=0, HSA_STATUS_ERROR=9 } hsa_status_t;
typedef enum { HSA_AGENT_INFO_DEVICE=17 } hsa_agent_info_t;
typedef enum { HSA_DEVICE_TYPE_CPU=0, HSA_DEVICE_TYPE_GPU=1 } hsa_device_type_t;
extern "C" {
hsa_status_t hsa_agent_get_info(hsa_agent_t,hsa_agent_info_t,void*);
hsa_status_t hsa_signal_create(hsa_signal_value_t,uint32_t,const hsa_agent_t*,hsa_signal_t*);
hsa_status_t hsa_signal_destroy(hsa_signal_t);
}
""")
            (inc / "hsa_ext_amd.h").write_text("""#pragma once
#include <stddef.h>
#include "hsa.h"
typedef enum { HSA_EXT_POINTER_TYPE_UNKNOWN=0,HSA_EXT_POINTER_TYPE_HSA=1,HSA_EXT_POINTER_TYPE_LOCKED=2 } hsa_amd_pointer_type_t;
typedef struct { uint32_t size; hsa_amd_pointer_type_t type; void* agentBaseAddress; void* hostBaseAddress;
size_t sizeInBytes; void* userData; hsa_agent_t agentOwner; uint32_t global_flags; bool registered; uint32_t alloc_flags; } hsa_amd_pointer_info_t;
extern "C" {
hsa_status_t hsa_amd_pointer_info(const void*,hsa_amd_pointer_info_t*,void*(*)(size_t),uint32_t*,hsa_agent_t**);
hsa_status_t hsa_amd_signal_get_event_id(hsa_signal_t,uint32_t*);
}
""")
            (inc / "amd_hsa_signal.h").write_text("#pragma once\nstruct alignas(64) amd_signal_t { unsigned char bytes[64]; };\n")
            fake = root / "runtime.cpp"
            fake.write_text("""#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_signal.h>
#include <stddef.h>
static amd_signal_t objects[32]; static uint64_t sequence=0; static int mode=0,created=0,destroyed=0,owner_queries=0;
extern "C" {
uint64_t fixture_cursor(){return sequence;}
uint64_t fixture_external(){return (uint64_t)&objects[0];}
void fixture_mode(int m){mode=m;}
int fixture_created(){return created;} int fixture_destroyed(){return destroyed;} int fixture_owner_queries(){return owner_queries;}
hsa_status_t hsa_signal_create(hsa_signal_value_t,uint32_t,const hsa_agent_t*,hsa_signal_t* p){sequence+=2;p->handle=(uint64_t)&objects[++created];return HSA_STATUS_SUCCESS;}
hsa_status_t hsa_signal_destroy(hsa_signal_t s){sequence+=2;if(s.handle==(uint64_t)&objects[0])return HSA_STATUS_ERROR;++destroyed;return HSA_STATUS_SUCCESS;}
hsa_status_t hsa_agent_get_info(hsa_agent_t,hsa_agent_info_t,void* p){sequence+=2;++owner_queries;*(hsa_device_type_t*)p=HSA_DEVICE_TYPE_CPU;return HSA_STATUS_SUCCESS;}
hsa_status_t hsa_amd_pointer_info(const void*,hsa_amd_pointer_info_t* i,void*(*)(size_t),uint32_t*,hsa_agent_t**){
sequence+=2;i->type=mode==1?HSA_EXT_POINTER_TYPE_UNKNOWN:HSA_EXT_POINTER_TYPE_HSA;i->hostBaseAddress=objects;i->agentBaseAddress=objects;i->sizeInBytes=sizeof(objects);i->agentOwner.handle=77;i->global_flags=1;i->registered=true;i->alloc_flags=1;
if(mode==2)i->size=offsetof(hsa_amd_pointer_info_t,agentOwner);if(mode==3)return HSA_STATUS_ERROR;return HSA_STATUS_SUCCESS;}
hsa_status_t hsa_amd_signal_get_event_id(hsa_signal_t,uint32_t* id){sequence+=2;*id=3;return HSA_STATUS_SUCCESS;}
}
""")
            runtime, provider = root / "runtime.so", root / "provider.so"
            subprocess.run(["c++", "-shared", "-fPIC", "-std=c++17", "-I", str(inc.parent), str(fake),
                            "-o", str(runtime)], check=True, capture_output=True)
            source = Path(capture.__file__).with_name("hsa_signal_backing_probe.cpp")
            subprocess.run(["c++", "-shared", "-fPIC", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(inc.parent), str(source), "-o", str(provider), "-ldl"],
                           check=True, capture_output=True)
            library = ctypes.CDLL(str(runtime))
            library.fixture_cursor.restype = ctypes.c_uint64
            library.fixture_external.restype = ctypes.c_uint64
            native = SimpleNamespace(mlsys_probe_cursor=library.fixture_cursor)
            probe = capture._Provider(provider, runtime, native)
            self.assertTrue(probe.initialization["ready"])
            self.assertEqual(probe.initialization["abi"]["amd_signal_size"], 64)
            control = probe.control(0, [])
            self.assertEqual(control["destroy_call"]["output"]["status"], 0)
            self.assertEqual([row["api_calls"][-1]["output"]["event_id"] for row in control["samples"]], [3, 3])
            self.assertEqual(library.fixture_created(), 1)
            self.assertEqual(library.fixture_destroyed(), 1)
            external = library.fixture_external()
            for mode in (0, 1, 2, 3):
                library.fixture_mode(mode)
                owners = library.fixture_owner_queries()
                sample = probe.sample(external)
                pointer = sample["api_calls"][0]["output"]
                self.assertEqual(library.fixture_destroyed(), 1)
                if mode == 0:
                    self.assertEqual(pointer["owner_agent"], 77)
                else:
                    self.assertIsNone(pointer["owner_agent"])
                    self.assertEqual(library.fixture_owner_queries(), owners)
                if mode == 1:
                    self.assertEqual(pointer["type"], 0)
                    self.assertIsNone(pointer["host_base_address"])
                if mode == 3:
                    self.assertEqual(pointer["status"], 9)
                    self.assertIsNone(pointer["returned_size"])
            output = ctypes.create_string_buffer(1)
            self.assertNotEqual(probe.library.mlsys_hsa_signal_backing_control(0, None, 0, output, 1), 0)
            self.assertEqual(library.fixture_destroyed(), 2)
            self.assertEqual(library.fixture_created(), 2)


if __name__ == "__main__":
    unittest.main()
