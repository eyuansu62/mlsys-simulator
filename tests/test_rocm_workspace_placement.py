"""CPU-only rank-local storage observations, never multi-GPU measurements."""

from copy import deepcopy
import math
import re
import unittest

from test_rocm_workspace_torch_audit import audit_torch_workspace, fixture


def layout(tp=1, pp=1, stage=0, rank=0, device=0):
    return {"tensor_parallel_size": tp, "pipeline_parallel_size": pp, "stage_index": stage,
            "tp_rank": rank, "device_index": device, "start_layer": stage * (28 // pp),
            "end_layer": (stage + 1) * (28 // pp)}


def contiguous(shape):
    result, step = [], 1
    for dim in reversed(shape):
        result.insert(0, step)
        step *= dim
    return result


def rank_fixture(tp=1, pp=1, stage=0, rank=0, device=0):
    data = fixture()
    data["placement"] = layout(tp, pp, stage, rank, device)
    data["device_index"] = device
    if pp == 2:
        data["model"]["tie_word_embeddings"] = False
    count = 28 // pp
    features = 1024 // tp
    blocks = 146 * tp * pp
    retained = []
    for original in data["owners_before"]:
        row = deepcopy(original)
        path = row["path"]
        match = re.search(r"model\.layers\.(\d+)\.", path)
        if match and not stage * count <= int(match[1]) < (stage + 1) * count:
            if "rotary_emb" in path:
                row["path"] = path.replace("model.layers.0.", f"model.layers.{stage * count}.")
            else:
                continue
        if path == "model.parameter:model.norm.weight" and stage != pp - 1:
            continue
        if path == "model.parameter:model.embed_tokens.weight":
            if pp == 2 and stage == 1:
                row["path"] = "model.parameter:lm_head.weight"
            row["shape"][0] //= tp
        elif path.endswith(("qkv_proj.weight", "gate_up_proj.weight")):
            row["shape"][0] //= tp
        elif path.endswith(("o_proj.weight", "down_proj.weight")):
            row["shape"][1] //= tp
        if path.startswith("model.parameter:"):
            row["bytes"] = math.prod(row["shape"]) * 2
            row["stride"] = contiguous(row["shape"])
        kv = re.fullmatch(r"runner.kv_caches\[(\d+)\]", path)
        if kv:
            index = int(kv[1])
            if index >= count:
                continue
            row["shape"] = [blocks, 2, 16, features]
            row["stride"] = contiguous(row["shape"])
            row["storage_offset"] = index * blocks * 2 * 16 * features
        retained.append(row)
    by_pointer = {row["pointer"]: row for row in retained}
    for name in ("before_snapshot", "after_snapshot"):
        snapshot = data[name]
        segments = []
        for segment in snapshot["segments"]:
            block = segment["blocks"][0]
            if block["state"] != "inactive":
                owner = by_pointer.get(block["address"])
                if owner is None:
                    continue
                block["requested_size"] = owner["bytes"]
                block["size"] = ((owner["bytes"] + 511) // 512) * 512
                segment["total_size"] = block["size"]
            segment["device"] = device
            segments.append(segment)
        snapshot["segments"] = segments
        trace = snapshot["device_traces"][0]
        snapshot["device_traces"] = [[] for _ in range(device)] + [trace]
    for event in data["torch_probe"]["events"]:
        event["device"] = device
    if pp == 2 and stage == 1:
        arena = data["before_snapshot"]["segments"][-1]["address"]
        for index, name in enumerate(("hidden_states", "residual")):
            pointer = arena + (32 + index * 4) * 1024 * 1024
            retained.append({"path": "runner.intermediate_tensors.tensors[" + name + "]",
                "category": "runtime", "pointer": pointer, "bytes": 2097152,
                "shape": [512, 2048], "stride": [2048, 1], "dtype": "torch.bfloat16",
                "element_size": 2, "storage_offset": 0})
            for field in ("before_snapshot", "after_snapshot"):
                data[field]["segments"].append({"device": device, "address": pointer,
                    "total_size": 2097152, "segment_pool_id": [0, 0], "is_expandable": False,
                    "blocks": [{"address": pointer, "size": 2097152, "requested_size": 2097152,
                                "state": "active_allocated"}]})
    data["owners_before"] = retained
    data["owners_after"] = deepcopy(retained)
    return data


class WorkspacePlacementTests(unittest.TestCase):
    def audit(self, data):
        before = deepcopy(data)
        result = audit_torch_workspace(**data)
        self.assertEqual(data, before)
        self.assertFalse(result["complete_workspace"])
        self.assertFalse(result["includes_non_torch_complete"])
        return result

    def closed(self, data):
        result = self.audit(data)
        self.assertEqual(result["status"], "CLOSED_CANDIDATE", result["issues"])
        self.assertEqual(result["requested_storage_peak"], 224)
        self.assertEqual(result["fixed_requested_bytes_by_category"]["kv_pool"], 267911168)
        return result

    def partial(self, data, code=None):
        result = self.audit(data)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["requested_storage_peak"])
        if code:
            self.assertIn(code, {row["code"] for row in result["issues"]})
        return result

    def test_legacy_default_and_explicit_single_rank_have_identical_audit_results(self):
        old = self.closed(fixture())
        current = self.closed(rank_fixture())
        self.assertEqual(current.pop("placement"), layout())
        self.assertEqual(old, current)

    def test_tp2_shards_each_rank_and_keeps_layer_norms_replicated(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                result = self.closed(rank_fixture(tp=2, rank=rank, device=rank))
                self.assertEqual(result["placement"]["tp_rank"], rank)
                self.assertEqual(result["fixed_requested_bytes_by_category"]["weights"], 1720698880)
                self.assertEqual(sum(row["category"] == "weights" for row in result["entry_categories"]), 226)

    def test_pp2_global_weight_layers_and_local_kv_views_have_separate_endpoints(self):
        for stage, expected in ((0, 2031737856), (1, 2031741952)):
            with self.subTest(stage=stage):
                result = self.closed(rank_fixture(pp=2, stage=stage, device=stage))
                self.assertEqual(result["fixed_requested_bytes_by_category"]["weights"], expected)
                self.assertEqual(result["placement"]["start_layer"], stage * 14)
                self.assertEqual(sum(row["category"] == "weights" for row in result["entry_categories"]), 113 + stage)
                if stage == 1:
                    self.assertEqual(result["fixed_requested_bytes_by_category"]["pp_buffers"], 4194304)

    def test_both_proved_kv_layouts_tile_the_exact_rank_local_pool(self):
        for args in (dict(tp=2), dict(pp=2), dict(pp=2, stage=1)):
            with self.subTest(args=args):
                data = rank_fixture(**args)
                for registry in ("owners_before", "owners_after"):
                    for row in data[registry]:
                        if row["path"].startswith("runner.kv_caches["):
                            blocks, _, _, features = row["shape"]
                            row["stride"] = [16 * features, blocks * 16 * features, features, 1]
                self.closed(data)

    def test_two_card_evidence_requires_complete_valid_local_placement(self):
        for change in (
            lambda p: p.pop("tp_rank"), lambda p: p.update(tp_rank=2),
            lambda p: p.update(stage_index=1), lambda p: p.update(device_index=1),
            lambda p: p.update(end_layer=14), lambda p: p.update(start_layer=1),
            lambda p: p.update(tensor_parallel_size=True), lambda p: p.update(extra="declared"),
            lambda p: p.update(pipeline_parallel_size=2, end_layer=14),
        ):
            data = rank_fixture(tp=2)
            change(data["placement"])
            self.partial(data, "INVALID_OR_INCOMPLETE_TORCH_EVIDENCE")
        for args in (dict(tp=2), dict(pp=2), dict(pp=2, stage=1)):
            data = rank_fixture(**args)
            data.pop("placement")
            self.partial(data)

    def test_missing_local_layer_or_endpoint_and_unsharded_weight_are_partial(self):
        data = rank_fixture(pp=2, stage=1)
        for path in ("model.parameter:lm_head.weight", "model.parameter:model.norm.weight",
                     "model.parameter:model.layers.14.self_attn.qkv_proj.weight"):
            case = deepcopy(data)
            for registry in ("owners_before", "owners_after"):
                case[registry] = [row for row in case[registry] if row["path"] != path]
            self.partial(case, "MISSING_REQUIRED_OWNERS")
        data = rank_fixture(tp=2)
        for registry in ("owners_before", "owners_after"):
            next(row for row in data[registry] if row["path"].endswith("qkv_proj.weight"))["shape"] = [4096, 2048]
        self.partial(data)

    def test_wrong_stage_weight_or_kv_alias_is_not_accepted_as_an_unknown_alias(self):
        for owner_path in ("model.parameter:lm_head.weight", "runner.kv_caches[14]",
                           "model.buffer:model.layers.27.self_attn.attn._q_scale"):
            data = rank_fixture(pp=2)
            for registry in ("owners_before", "owners_after"):
                if "kv_caches" in owner_path:
                    source = next(row for row in data[registry] if row["path"].startswith("runner.kv_caches["))
                else:
                    source = data[registry][0]
                alias = deepcopy(source)
                alias["path"] = owner_path
                data[registry].append(alias)
            self.partial(data, "UNEXPECTED_PLACEMENT_OWNER")

    def test_wrong_tied_alias_or_kv_partition_cannot_become_workspace(self):
        data = rank_fixture(tp=2)
        for registry in ("owners_before", "owners_after"):
            alias = deepcopy(data[registry][0])
            alias.update(path="model.parameter:lm_head.weight", pointer=alias["pointer"] + 512)
            data[registry].append(alias)
        self.partial(data, "TIED_EMBEDDING_ALIAS_MISMATCH")
        for change in (lambda rows: rows[-1].update(storage_offset=0),
                       lambda rows: rows[0]["stride"].__setitem__(0, 1)):
            data = rank_fixture(pp=2)
            for registry in ("owners_before", "owners_after"):
                rows = [row for row in data[registry] if row["path"].startswith("runner.kv_caches[")]
                change(rows)
            self.partial(data)

    def test_unknown_pp_buffers_and_new_persistent_storage_still_prevent_closure(self):
        data = rank_fixture(pp=2, stage=1)
        for registry in ("owners_before", "owners_after"):
            source = next(row for row in data[registry] if row["path"] == "runner.positions")
            source["path"] = "runner.intermediate_tensors.tensors[unproved_field]"
            source["category"] = "pp_buffers"
        self.partial(data, "UNATTRIBUTED_ENTRY_STORAGE")
        data = rank_fixture(tp=2)
        data["after_snapshot"]["device_traces"][0].pop(-2)
        probe = data["torch_probe"]
        probe["events"].pop(-2)
        probe["events"][-1]["sequence"] -= 1
        for field in ("retained_events", "attempted_events_at_cutoff", "attempted_events_at_write_end"):
            probe[field] -= 1
        probe["epochs"][0]["observed_events"] -= 1
        probe["epochs"][0]["end_sequence"] -= 1
        self.partial(data, "NEW_PERSISTENT_OR_PENDING_STORAGE")

    def test_sampling_metadata_empty_to_full_views_require_the_same_real_base_storage(self):
        data = fixture()
        arena = data["before_snapshot"]["segments"][-1]["address"]
        for index, name in enumerate(("frequency_penalties", "presence_penalties", "repetition_penalties")):
            pointer = arena + (64 + index) * 4096
            base = {"path": "runner.input_batch." + name, "category": "runtime", "pointer": pointer,
                    "bytes": 4, "shape": [1], "stride": [1], "dtype": "torch.float32",
                    "element_size": 4, "storage_offset": 0}
            for side, owner_field in (("before_snapshot", "owners_before"), ("after_snapshot", "owners_after")):
                data[owner_field].append(deepcopy(base))
                alias = {**base, "path": "runner.input_batch.sampling_metadata." + name,
                         "shape": [0] if side == "before_snapshot" else [1]}
                data[owner_field].append(alias)
                data[side]["segments"].append({"device": 0, "address": pointer, "total_size": 512,
                    "segment_pool_id": [0, 0], "is_expandable": False,
                    "blocks": [{"address": pointer, "size": 512, "requested_size": 4, "state": "active_allocated"}]})
        self.closed(data)
        case = deepcopy(data)
        for field in ("owners_before", "owners_after"):
            case[field] = [row for row in case[field] if row["path"] != "runner.input_batch.frequency_penalties"]
        self.partial(case, "SAMPLING_VIEW_ALIAS_BINDING_MISMATCH")
        case = deepcopy(data)
        for field in ("owners_before", "owners_after"):
            next(row for row in case[field] if row["path"] == "runner.input_batch.sampling_metadata.frequency_penalties")["pointer"] += 512
        self.partial(case, "SAMPLING_VIEW_ALIAS_BINDING_MISMATCH")


if __name__ == "__main__":
    unittest.main()
