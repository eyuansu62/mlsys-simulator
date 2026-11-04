"""CPU observations only: no Torch/GPU object is constructed by these tests."""

import math
import unittest
from copy import deepcopy
from dataclasses import asdict

from mlsys_sim import ModelSpec
import importlib.util
from pathlib import Path
import sys

_TOOL_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "rocm_workspace_audit"


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, _TOOL_ROOT / (name + ".py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load packaged ROCm audit source: " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit_torch_workspace = _load_tool("workspace_torch_audit").audit_torch_workspace


def fixture():
    model = asdict(
        ModelSpec(
            name="Qwen3-1.7B",
            num_layers=28,
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=16,
            num_kv_heads=8,
            head_dim=128,
            vocab_size=151936,
            weight_dtype="bfloat16",
            kv_dtype="bfloat16",
            activation_dtype="bfloat16",
            qk_norm=True,
            final_norm=True,
            tie_word_embeddings=True,
            revision="cpu-fixture",
            architecture_config_sha256="a" * 64,
        )
    )
    owners, segments = [], []
    cursor = 0x100000000

    def stride(shape):
        output: list[int]
        value, output = 1, []
        for dim in reversed(shape):
            output.insert(0, value)
            value *= dim
        return output

    def storage(path, category, shape, dtype="bfloat16", size=None):
        nonlocal cursor
        width = {"bfloat16": 2, "float32": 4, "int32": 4, "int64": 8}[dtype]
        size = math.prod(shape) * width if size is None else size
        owner = {
            "path": path,
            "category": category,
            "pointer": cursor,
            "bytes": size,
            "shape": shape,
            "dtype": "torch." + dtype,
            "element_size": width,
            "storage_offset": 0,
            "stride": stride(shape),
        }
        owners.append(owner)
        rounded = ((size + 511) // 512) * 512
        segments.append(
            {
                "device": 0,
                "address": cursor,
                "total_size": rounded,
                "segment_pool_id": [0, 0],
                "is_expandable": False,
                "blocks": [
                    {
                        "address": cursor,
                        "size": rounded,
                        "requested_size": size,
                        "state": "active_allocated",
                        "frames": [],
                    }
                ],
            }
        )
        cursor += rounded + 4096
        return owner

    storage("model.parameter:model.embed_tokens.weight", "weights", [151936, 2048])
    storage("model.parameter:model.norm.weight", "weights", [2048])
    for layer in range(28):
        prefix = f"model.parameter:model.layers.{layer}."
        for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            storage(prefix + suffix, "weights", [2048])
        for suffix in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            storage(prefix + suffix, "weights", [128])
        for suffix, shape in (
            ("self_attn.qkv_proj.weight", [4096, 2048]),
            ("self_attn.o_proj.weight", [2048, 2048]),
            ("mlp.gate_up_proj.weight", [12288, 2048]),
            ("mlp.down_proj.weight", [2048, 6144]),
        ):
            storage(prefix + suffix, "weights", shape)
        for scale in ("q", "k", "v", "prob"):
            storage(
                f"model.buffer:model.layers.{layer}.self_attn.attn._{scale}_scale", "runtime", [], "float32"
            )
    storage("model.buffer:model.layers.0.self_attn.rotary_emb.cos_sin_cache", "runtime", [40960, 128])
    pool = storage("runner.kv_caches[0]", "kv_pool", [146, 2, 16, 1024], size=267911168)
    elements = math.prod(pool["shape"])
    for layer in range(1, 28):
        current = deepcopy(pool)
        current.update(path=f"runner.kv_caches[{layer}]", storage_offset=elements * layer)
        owners.append(current)
    blas = storage("ATen.getCUDABlasLtWorkspace", "runtime", [83886080 // 4], "float32")
    for key in ("shape", "dtype", "element_size", "storage_offset", "stride"):
        blas.pop(key)
    blas["binding"] = "verified_exported_getter_no_allocation"
    storage("runner.positions", "runtime", [512], "int64")
    arena = cursor
    segments.append(
        {
            "device": 0,
            "address": arena,
            "total_size": 4096,
            "segment_pool_id": [0, 0],
            "is_expandable": False,
            "blocks": [
                {"address": arena, "size": 4096, "requested_size": 0, "state": "inactive", "frames": []}
            ],
        }
    )

    def history(action, address, size):
        return {
            "action": action,
            "addr": address,
            "size": size,
            "stream": 0,
            "pool_id": [0, 0],
            "user_metadata": "",
            "compile_context": "N/A",
            "time_us": 1234,
            "frames": [{"name": "not-owner-identity"}],
        }

    a, b = arena, arena + 512
    specs = [
        ("alloc", a, 64),
        ("alloc", b, 128),
        ("free_requested", a, 64),
        ("free_completed", a, 64),
        ("alloc", a, 96),
        ("free_requested", b, 128),
        ("free_completed", b, 128),
        ("free_requested", a, 96),
        ("free_completed", a, 96),
    ]
    delta = [history(*row) for row in specs]
    before_trace = [history("snapshot", 0, 999)]
    before = {"segments": segments, "device_traces": [before_trace]}
    after_trace = deepcopy(before_trace) + delta + [history("snapshot", 0, 999)]
    # The actual serializer can convert old timestamps slightly differently.
    after_trace[0]["time_us"] += 1
    after_trace[0]["frames"] = [{"name": "resymbolized-after-snapshot"}]
    after = {"segments": deepcopy(segments), "device_traces": [after_trace]}

    def event(row, sequence, epoch=7):
        return {
            "sequence": sequence,
            "timestamp_ns": 100 + sequence if epoch else 90 if sequence == 1 else 201,
            "thread_id": 77,
            "epoch_id": epoch,
            "owner_token": epoch,
            "action": (
                "alloc",
                "free_requested",
                "free_completed",
                "segment_alloc",
                "segment_free",
                "segment_map",
                "segment_unmap",
                "snapshot",
                "oom",
            ).index(row["action"]),
            "action_name": row["action"],
            "device": 0,
            "address": row["addr"],
            "size_bytes": row["size"],
            "stream": row["stream"],
            "mempool": row["pool_id"],
            "flags": 0 if epoch else 4,
            "user_metadata": row["user_metadata"],
            "user_metadata_size": 0,
            "retained_metadata_size": 0,
        }

    events = [event(before_trace[0], 1, 0)] + [event(row, i + 2) for i, row in enumerate(delta)]
    events.append(event(history("snapshot", 0, 999), len(events) + 1, 0))
    epoch = {
        "id": 7,
        "owner_thread_id": 77,
        "begin_ns": 100,
        "end_ns": 200,
        "begin_sequence": 1,
        "end_sequence": len(events) - 1,
        "ended": True,
        "observed_events": len(delta),
        "dropped_events": 0,
        "unknown_actions": 0,
        "unowned_allocations": 0,
        "cross_thread_allocations": 0,
        "truncated_metadata": 0,
        "boundary_overlap": 0,
        "boundary_callbacks": 0,
        "clock_failures": 0,
        "thread_id_failures": 0,
        "oom_events": 0,
        "mapping_events": 0,
        "observation_issue_mask": 0,
        "observation_issues_present": False,
    }
    probe = {
        "schema_version": 1,
        "kind": "torch_allocator_trace_probe",
        "complete_workspace": False,
        "allocation_owner_classification_performed": False,
        "checkpoint_stable": True,
        "checkpoint_is_prefix_only": True,
        "global_observation_issues_present": False,
        "callback_gpu_calls": False,
        "callback_gil": False,
        "callback_extra_locks": False,
        "abi_cxx11": 1,
        "install_thread_id": 77,
        "retained_events": len(events),
        "attempted_events_at_cutoff": len(events),
        "attempted_events_at_write_end": len(events),
        "dropped_events": 0,
        "unknown_actions": 0,
        "truncated_metadata": 0,
        "clock_failures": 0,
        "thread_id_failures": 0,
        "api_errors": 0,
        "boundary_callbacks": 0,
        "callbacks_inflight_at_write_end": 0,
        "events": events,
        "epochs": [epoch],
    }
    return {
        "model": model,
        "before_snapshot": before,
        "after_snapshot": after,
        "owners_before": owners,
        "owners_after": deepcopy(owners),
        "torch_probe": probe,
        "epoch_id": 7,
        "history_max_entries": 100000,
    }


class TorchWorkspaceAuditTests(unittest.TestCase):
    def audit(self, data):
        before = deepcopy(data)
        result = audit_torch_workspace(**data)
        self.assertEqual(data, before, "the audit must not mutate observations")
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["includes_non_torch_complete"])
        return result

    def closed(self, data):
        result = self.audit(data)
        self.assertEqual(result["status"], "CLOSED_CANDIDATE", result["issues"])
        self.assertTrue(result["torch_side_closed_candidate"])
        return result

    def partial(self, data, code=None):
        result = self.audit(data)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["requested_storage_peak"])
        if code:
            self.assertIn(code, {row["code"] for row in result["issues"]})
        return result

    def test_requested_peak_real_order_generation_and_shared_kv_dedup(self):
        data = fixture()
        result = self.closed(data)
        self.assertEqual(result["requested_storage_peak"], 224)
        self.assertEqual(result["observed_requested_storage_peak"], 224)
        self.assertEqual(result["new_storage_generations"], 3)
        allocations = [row for row in result["ordered_ledger"] if row["action"] == "allocate"]
        self.assertEqual([row["generation"] for row in allocations], [1, 1, 2])
        self.assertEqual(len({row["storage_id"] for row in allocations}), 3)
        self.assertEqual(result["fixed_requested_bytes_by_category"]["kv_pool"], 267911168)
        self.assertEqual(sum(row["category"] == "kv_pool" for row in result["entry_categories"]), 1)
        self.assertEqual(sum(row["category"] == "weights" for row in result["entry_categories"]), 226)
        self.assertEqual(result["recorder_capture"]["status"], "COMPLETED")
        self.assertEqual(result["history_cross_check"]["cross_checked_non_snapshot_events"], 9)

    def test_free_requested_does_not_lower_live_storage_before_completion(self):
        data = fixture()
        trace = data["after_snapshot"]["device_traces"][0]
        native = data["torch_probe"]["events"]
        a, b, c = trace[1]["addr"], trace[2]["addr"], trace[1]["addr"] + 1024
        specs = [
            ("alloc", a, 64),
            ("alloc", b, 128),
            ("free_requested", a, 64),
            ("alloc", c, 256),
            ("free_completed", a, 64),
            ("free_requested", c, 256),
            ("free_completed", c, 256),
            ("free_requested", b, 128),
            ("free_completed", b, 128),
        ]
        for index, (action, address, size) in enumerate(specs, 1):
            trace[index].update(action=action, addr=address, size=size)
            native[index].update(
                action=("alloc", "free_requested", "free_completed").index(action),
                action_name=action,
                address=address,
                size_bytes=size,
            )
        result = self.closed(data)
        self.assertEqual(result["requested_storage_peak"], 448)

    def test_tied_lm_head_alias_is_optional_and_deduplicated(self):
        data = fixture()
        original = self.closed(data)["fixed_requested_bytes_by_category"]["weights"]
        alias = deepcopy(data["owners_before"][0])
        alias["path"] = "model.parameter:lm_head.weight"
        data["owners_before"].append(alias)
        data["owners_after"].append(deepcopy(alias))
        self.assertEqual(self.closed(data)["fixed_requested_bytes_by_category"]["weights"], original)
        data["owners_before"][-1]["pointer"] += 4096
        self.partial(data, "TIED_EMBEDDING_ALIAS_MISMATCH")

    def test_unknown_alias_diagnostic_does_not_create_an_owner(self):
        data = fixture()
        alias = deepcopy(next(row for row in data["owners_before"] if row["path"] == "runner.positions"))
        alias["path"] = "runner.not_in_white_list"
        data["owners_before"].append(alias)
        data["owners_after"].append(deepcopy(alias))
        result = self.closed(data)
        self.assertTrue(result["diagnostics"]["unknown_alias_paths"])
        for key in ("owners_before", "owners_after"):
            data[key] = [row for row in data[key] if row["path"] != "runner.positions"]
        self.partial(data, "UNATTRIBUTED_ENTRY_STORAGE")

    def test_alias_category_and_capacity_conflicts_fail(self):
        for field, value in (("category", "weights"), ("bytes", 8192)):
            data = fixture()
            alias = deepcopy(data["owners_before"][-1])
            alias.update(path="runner.unknown_alias")
            alias[field] = value
            data["owners_before"].append(alias)
            data["owners_after"].append(deepcopy(alias))
            self.partial(data, "ALIAS_CATEGORY_OR_SIZE_CONFLICT")

    def test_known_path_wrong_shape_dtype_category_or_view_extent_fails(self):
        for field, value in (
            ("shape", [2048, 151936]),
            ("category", "runtime"),
            ("dtype", "torch.float32"),
            ("storage_offset", 1),
            ("element_size", True),
        ):
            data = fixture()
            data["owners_before"][0][field] = value
            self.partial(data)

    def test_missing_or_fake_blas_getter_binding_fails(self):
        for field, value in (("binding", "guessed_from_stack"), ("bytes", 83886079)):
            data = fixture()
            row = next(row for row in data["owners_before"] if row["path"] == "ATen.getCUDABlasLtWorkspace")
            row[field] = value
            self.partial(data, "INVALID_KNOWN_OWNER")

    def test_kv_layout_storage_offset_and_pool_alias_proofs_are_required(self):
        for mutation in (
            lambda row: row.update(shape=[2, 146, 16, 1024]),
            lambda row: row.update(storage_offset=0),
            lambda row: row.update(pointer=row["pointer"] + 4096),
        ):
            data = fixture()
            row = next(row for row in data["owners_before"] if row["path"] == "runner.kv_caches[1]")
            mutation(row)
            self.partial(data)

    def test_actual_kv_axis_permutation_has_a_mathematically_complete_layer_tile(self):
        data = fixture()
        original = self.closed(data)["fixed_requested_bytes_by_category"]
        for registry in ("owners_before", "owners_after"):
            for row in data[registry]:
                if row["path"].startswith("runner.kv_caches["):
                    row["stride"] = [16384, 146 * 16384, 1024, 1]
        result = self.closed(data)
        self.assertEqual(result["fixed_requested_bytes_by_category"], original)
        self.assertEqual(result["requested_storage_peak"], 224)

    def test_kv_bounds_do_not_substitute_for_no_overlap_and_no_holes_proof(self):
        cases = (
            (32769, 16239, 1024, 1),  # Same bounding span, different interior.
            (28 * 32768, 16384, 1024, 1),
        )  # Unobserved block-major layout.
        for strides in cases:
            data = fixture()
            for registry in ("owners_before", "owners_after"):
                for row in data[registry]:
                    if row["path"].startswith("runner.kv_caches["):
                        row["stride"] = list(strides)
                        if strides[0] == 28 * 32768:
                            layer = int(row["path"].split("[")[1].split("]")[0])
                            row["storage_offset"] = layer * 32768
            self.partial(data, "KV_LAYOUT_NOT_PROVED")

    def test_mixed_proved_kv_layouts_are_not_one_fixed_layout(self):
        data = fixture()
        for registry in ("owners_before", "owners_after"):
            row = next(row for row in data[registry] if row["path"] == "runner.kv_caches[0]")
            row["stride"] = [16384, 146 * 16384, 1024, 1]
        self.partial(data, "KV_LAYOUT_NOT_PROVED")

    def test_fixed_binding_changes_and_pending_entry_frees_are_partial(self):
        data = fixture()
        data["owners_after"][-1]["path"] = "runner.positions_replaced"
        self.partial(data, "FIXED_OWNER_BINDINGS_CHANGED")
        data = fixture()
        data["before_snapshot"]["segments"][0]["blocks"][0]["state"] = "active_awaiting_free"
        self.partial(data, "PENDING_FREE_AT_BOUNDARY")

    def test_fixed_storage_freed_and_reallocated_is_not_hidden_by_equal_snapshots(self):
        data = fixture()
        address = data["before_snapshot"]["segments"][0]["address"]
        data["after_snapshot"]["device_traces"][0][1]["addr"] = address
        data["torch_probe"]["events"][1]["address"] = address
        self.partial(data, "FIXED_STORAGE_LIFETIME_MUTATED")

    def test_history_wrap_prefix_and_supplied_delta_mismatch_fail(self):
        data = fixture()
        data["history_max_entries"] = len(data["after_snapshot"]["device_traces"][0])
        self.partial(data)
        data = fixture()
        data["after_snapshot"]["device_traces"][0][0]["user_metadata"] = "changed"
        self.partial(data)
        data = fixture()
        data["history_delta"] = deepcopy(data["after_snapshot"]["device_traces"][0][1:])
        self.closed(data)
        data["history_delta"][0]["size"] += 1
        self.partial(data)

    def test_raw_native_order_size_and_metadata_must_exactly_match_history(self):
        for field, value in (
            ("size_bytes", 65),
            ("stream", 99),
            ("address", 123),
            ("mempool", [1, 0]),
            ("user_metadata", "x"),
        ):
            data = fixture()
            event = data["torch_probe"]["events"][1]
            event[field] = value
            if field == "user_metadata":
                event.update(user_metadata_size=1, retained_metadata_size=1)
            self.partial(data)

    def test_drop_unknown_cross_thread_and_bad_boundary_are_partial(self):
        mutations = [
            lambda p: p.update(dropped_events=1),
            lambda p: p.pop("thread_id_failures"),
            lambda p: p["events"][1].update(thread_id=78),
            lambda p: p["events"][1].update(owner_token=0),
            lambda p: p["events"][1].update(flags=2),
            lambda p: p["events"][1].update(timestamp_ns=201),
            lambda p: p["epochs"][0].update(observation_issue_mask=1),
            lambda p: p["events"][1].update(sequence=100),
            lambda p: p["events"][1].update(epoch_id=0),
        ]
        for mutation in mutations:
            data = fixture()
            mutation(data["torch_probe"])
            self.partial(data)

    def test_free_request_without_completion_and_new_survivors_are_partial(self):
        data = fixture()
        trace = data["after_snapshot"]["device_traces"][0]
        trace.pop(-2)
        probe = data["torch_probe"]
        probe["events"].pop(-2)
        probe["events"][-1]["sequence"] -= 1
        for field in ("retained_events", "attempted_events_at_cutoff", "attempted_events_at_write_end"):
            probe[field] -= 1
        probe["epochs"][0]["observed_events"] -= 1
        probe["epochs"][0]["end_sequence"] -= 1
        self.partial(data, "NEW_PERSISTENT_OR_PENDING_STORAGE")

    def test_unmatched_duplicate_free_and_overlap_fail(self):
        for replacement in ("free_requested", "free_completed", "alloc"):
            data = fixture()
            trace = data["after_snapshot"]["device_traces"][0]
            event = data["torch_probe"]["events"][2]
            trace[2]["action"] = replacement
            event.update(
                action=("alloc", "free_requested", "free_completed").index(replacement),
                action_name=replacement,
            )
            if replacement == "alloc":
                trace[2]["addr"] = trace[1]["addr"]
                event["address"] = trace[1]["addr"]
            self.partial(data)

    def test_unsupported_pool_mapping_oom_and_model_are_partial(self):
        data = fixture()
        data["before_snapshot"]["segments"][0]["segment_pool_id"] = [1, 0]
        self.partial(data, "UNSUPPORTED_POOL_OR_MAPPING")
        for action, index in (("segment_map", 5), ("oom", 8)):
            data = fixture()
            data["after_snapshot"]["device_traces"][0][1]["action"] = action
            data["torch_probe"]["events"][1].update(action=index, action_name=action)
            self.partial(data, "UNSUPPORTED_MAPPING_OR_OOM")
        for field, value in (("num_layers", 27), ("tie_word_embeddings", False), ("activation_bytes", True)):
            data = fixture()
            data["model"][field] = value
            self.partial(data)

    def test_zero_byte_unknown_views_do_not_add_capacity_or_create_owner(self):
        data = fixture()
        empty = {
            "path": "runner.input_batch.logitsprocs.non_argmax_invariant[0].logits_slice[0]",
            "category": "runtime",
            "pointer": 0,
            "bytes": 0,
            "shape": [0],
            "dtype": "torch.float32",
            "element_size": 4,
            "storage_offset": 0,
            "stride": [1],
        }
        data["owners_before"].append(empty)
        data["owners_after"].append(deepcopy(empty))
        result = self.closed(data)
        self.assertEqual(result["requested_storage_peak"], 224)
        self.assertTrue(result["diagnostics"]["zero_storage_paths"])

    def test_actual_empty_min_p_view_keeps_four_byte_backing_without_becoming_owner(self):
        data = self.min_p_fixture()
        result = self.closed(data)
        storage = next(
            row
            for row in result["entry_categories"]
            if any(path.endswith(".min_p_device") for path in row["verified_owner_paths"])
        )
        self.assertEqual(storage["requested_bytes"], 4)
        self.assertEqual(len(storage["verified_owner_paths"]), 1)
        self.assertEqual(len(storage["reported_fieldpaths"]), 2)
        self.assertFalse(
            any(row["path"].endswith(".min_p") for row in result["diagnostics"]["zero_storage_paths"])
        )

    def min_p_fixture(self):
        data = fixture()
        pointer = data["before_snapshot"]["segments"][-1]["address"] + 8192
        owner = {
            "path": "runner.input_batch.logitsprocs.argmax_invariant[0].min_p_device",
            "category": "runtime",
            "pointer": pointer,
            "bytes": 4,
            "shape": [1],
            "dtype": "torch.float32",
            "element_size": 4,
            "storage_offset": 0,
            "stride": [1],
        }
        alias = {**owner, "path": "runner.input_batch.logitsprocs.argmax_invariant[0].min_p", "shape": [0]}
        segment = {
            "device": 0,
            "address": pointer,
            "total_size": 512,
            "segment_pool_id": [0, 0],
            "is_expandable": False,
            "blocks": [{"address": pointer, "size": 512, "requested_size": 4, "state": "active_allocated"}],
        }
        for registry in ("owners_before", "owners_after"):
            data[registry].extend([deepcopy(owner), deepcopy(alias)])
        for snapshot in ("before_snapshot", "after_snapshot"):
            data[snapshot]["segments"].append(deepcopy(segment))
        return data

    def test_empty_min_p_alias_cannot_own_separate_storage_or_replace_device_owner(self):
        for mutation in ("missing_device", "separate_storage", "nonempty", "wrong_stride"):
            data = self.min_p_fixture()
            for registry in ("owners_before", "owners_after"):
                alias = data[registry][-1]
                if mutation == "missing_device":
                    data[registry] = [
                        row for row in data[registry] if not row["path"].endswith(".min_p_device")
                    ]
                elif mutation == "separate_storage":
                    alias["pointer"] += 1024
                elif mutation == "nonempty":
                    alias["shape"] = [1]
                else:
                    alias["stride"] = [0]
            if mutation == "separate_storage":
                for snapshot in ("before_snapshot", "after_snapshot"):
                    segment = deepcopy(data[snapshot]["segments"][-1])
                    segment["address"] += 1024
                    segment["blocks"][0]["address"] += 1024
                    data[snapshot]["segments"].append(segment)
            self.partial(data, "MIN_P_EMPTY_ALIAS_BINDING_MISMATCH")


if __name__ == "__main__":
    unittest.main()
