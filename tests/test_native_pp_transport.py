"""Native PP geometry, exact calibration and conservative boundary accounting."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, MemoryPlanner, ModelSpec,
                       RequestSpec, SimulationError, StrategySpec, calibrate_dataset,
                       comparison_contract, experiment_from_dict, load_experiment, profile_identity,
                       simulate, validate_experiment)
from mlsys_sim.comparison import _contract
from mlsys_sim.layout import replica_experiment
from mlsys_sim.performance import predict_experiment
from mlsys_sim.pipeline import PipelineSimulation
from mlsys_sim.pp_transport import (estimate_pipeline_transfer, native_pp_descriptor,
                                   validate_native_pp_descriptor)
from mlsys_sim.profiles import estimate_profile, validate_dataset, validate_identity


ROOT = Path(__file__).resolve().parents[1]


def experiment(*, tp=2, pp=2, **changes):
    strategy = StrategySpec(model=ModelSpec(activation_dtype="float16"),
        pipeline_parallel_size=pp, tensor_parallel_size=tp, max_batch_tokens=16,
        pipeline_transport="vllm_native", preemption="none")
    return Experiment(strategy=replace(strategy, **changes),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(tp * pp)),
                        link_latency_us=1000, link_bandwidth_gbps=1),
        workload=(RequestSpec("request", 0, 4, 1),),
        calibration=CalibrationSpec(backend="synthetic", prefill_ms=10, decode_ms=5))


def record(exp, stage, *, query=(4,), context=(0,), padded=8, phase="prefill", roles=None, duration=3):
    directions = ({"incoming"} if stage > 0 else set()) | (
        {"outgoing"} if stage < exp.strategy.pipeline_parallel_size - 1 else set())
    row = {"record_id": f"{phase}-{stage}", "group_id": f"{phase}-{stage}", "phase": phase,
        "stage_index": stage, "query_tokens": list(query), "context_lengths": list(context),
        "duration_ms": duration, "workspace_peak_bytes": None,
        "pp_transport": {direction: native_pp_descriptor(exp, padded) for direction in sorted(directions)}}
    if roles is not None:
        row["request_phases"] = list(roles)
    return row


def dataset(exp, records):
    return {"schema_version": 1, "dataset_id": "native-pp-fixture", "identity": profile_identity(exp),
        "source": {"kind": "test_fixture", "reference": "Hand-declared geometry and timing; no GPU evidence"},
        "records": records}


class NativePPGeometryTests(unittest.TestCase):
    def test_native_dtype_pp_and_token_bound_are_explicit(self):
        valid = experiment()
        validate_experiment(valid)
        for changed in (replace(valid.strategy, pipeline_transport="unknown"),
                        replace(valid.strategy, pipeline_transport_max_tokens=15),
                        replace(valid.strategy, model=replace(valid.strategy.model, activation_dtype="auto"))):
            with self.subTest(changed=changed), self.assertRaises(SimulationError):
                validate_experiment(replace(valid, strategy=changed))
        with self.assertRaises(SimulationError):
            validate_experiment(experiment(pp=1))
        for value in (True, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(SimulationError):
                experiment_from_dict({"strategy": {"pipeline_transport_max_tokens": value}})

    def test_two_tensors_keep_full_resident_bytes_and_slice_only_wire(self):
        for tp, sliced, wire in ((1, False, 1536), (2, True, 768), (4, True, 384)):
            with self.subTest(tp=tp):
                descriptor = native_pp_descriptor(experiment(tp=tp), 3)
                self.assertEqual(descriptor["resident_bytes_per_rank"], 1536)
                self.assertEqual(descriptor["wire_bytes_per_rank"], wire)
                self.assertEqual([row["name"] for row in descriptor["tensors"]], ["hidden_states", "residual"])
                for tensor in descriptor["tensors"]:
                    self.assertEqual(tensor["shape"], [3, 128])
                    self.assertEqual(tensor["numel"], 384)
                    self.assertEqual(tensor["full_bytes"], 768)
                    self.assertEqual(tensor["tp_sliced"], sliced)
                    self.assertEqual(tensor["receiver_allgather"], sliced)
                self.assertEqual(validate_native_pp_descriptor(experiment(tp=tp), descriptor), descriptor)

    def test_unshardable_geometry_falls_back_to_full_tensor_wire(self):
        # The helper stays conservative even for a future caller outside the
        # engine's stricter hidden_size-divisible-by-TP configuration domain.
        exp = experiment()
        exp = replace(exp, strategy=replace(exp.strategy, model=replace(exp.strategy.model, hidden_size=129)))
        descriptor = native_pp_descriptor(exp, 1)
        self.assertEqual(descriptor["wire_bytes_per_rank"], 516)
        self.assertEqual(descriptor["resident_bytes_per_rank"], 516)
        self.assertTrue(all(not tensor["tp_sliced"] and not tensor["receiver_allgather"]
                            for tensor in descriptor["tensors"]))

    def test_descriptor_rejects_misleading_shape_dtype_bytes_slice_and_types(self):
        exp = experiment()
        good = native_pp_descriptor(exp, 4)
        mutations = []
        for key, value in (("wire_bytes_per_rank", 2048), ("resident_bytes_per_rank", True),
                           ("contract", "other"), ("padded_tokens", False)):
            bad = deepcopy(good)
            bad[key] = value
            mutations.append(bad)
        for key, value in (("shape", [4, 129]), ("dtype", "float32"), ("numel", 513),
                           ("full_bytes", 1023), ("wire_bytes_per_rank", 1024),
                           ("tp_sliced", 1), ("receiver_allgather", False), ("element_size", 2.0)):
            bad = deepcopy(good)
            bad["tensors"][0][key] = value
            mutations.append(bad)
        bad = deepcopy(good)
        bad["tensors"].reverse()
        mutations.append(bad)
        for bad in mutations:
            with self.subTest(bad=bad), self.assertRaises(SimulationError):
                validate_native_pp_descriptor(exp, bad)
        self.assertEqual(good, native_pp_descriptor(exp, 4))

    def test_element_and_wire_byte_accounting_remains_integer_exact(self):
        exp = experiment(max_batch_tokens=1)
        model = replace(exp.strategy.model, hidden_size=2**53 + 2)
        exp = replace(exp, strategy=replace(exp.strategy, model=model))
        descriptor = native_pp_descriptor(exp, 1)
        self.assertEqual(descriptor["resident_bytes_per_rank"], 4 * (2**53 + 2))
        self.assertEqual(descriptor["wire_bytes_per_rank"], 2 * (2**53 + 2))
        self.assertIs(type(descriptor["tensors"][0]["full_bytes"]), int)

    def test_wire_p2p_and_receiver_allgather_are_charged_once_per_tensor(self):
        for tp in (1, 2):
            with self.subTest(tp=tp):
                exp = experiment(tp=tp)
                transfer = estimate_pipeline_transfer(exp, [4])
                wire = 1024 // tp
                p2p = 2 * (1 + wire / 1e6)
                gather = 0 if tp == 1 else 2 * (1 + 512 / 1e6)
                self.assertAlmostEqual(transfer["duration_ms"], p2p + gather)
                self.assertAlmostEqual(transfer["breakdown"]["p2p_ms"], p2p)
                self.assertAlmostEqual(transfer["breakdown"]["receiver_allgather_ms"], gather)
                self.assertEqual(transfer["source"], "analytical_uncalibrated")
                self.assertFalse(transfer["breakdown"]["communication_calibrated"])
                self.assertFalse(transfer["breakdown"]["receiver_wait_used_as_network_service"])

    def test_legacy_identity_and_single_full_hidden_cost_are_unchanged(self):
        exp = experiment(pipeline_transport="full_hidden")
        transfer = estimate_pipeline_transfer(exp, [4])
        self.assertEqual(transfer["duration_ms"], 1 + 1024 / 1e6)
        self.assertEqual(transfer["bytes_per_rank"], 1024)
        identity = profile_identity(exp)
        self.assertNotIn("pipeline_transport", identity["parallelism"])
        self.assertNotIn("pipeline_transport_max_tokens", identity["parallelism"])
        explicit = deepcopy(identity)
        explicit["parallelism"].update(pipeline_transport="full_hidden", pipeline_transport_max_tokens=0)
        self.assertEqual(validate_identity(explicit), identity)

    def test_native_identity_binds_effective_padding_bound_and_compares(self):
        exp = experiment()
        identity = profile_identity(exp)
        self.assertEqual(identity["parallelism"]["pipeline_transport_max_tokens"], 16)
        self.assertEqual(validate_identity(identity), identity)
        self.assertEqual(profile_identity(replace(exp, strategy=replace(exp.strategy,
            pipeline_transport_max_tokens=16))), identity)
        self.assertNotEqual(profile_identity(replace(exp, strategy=replace(exp.strategy,
            pipeline_transport_max_tokens=32))), identity)
        contract = comparison_contract(exp)
        self.assertEqual(_contract(contract, "fixture")["identity"], identity)
        old = comparison_contract(experiment(pipeline_transport="full_hidden"))
        old["strategy"].pop("pipeline_transport")
        old["strategy"].pop("pipeline_transport_max_tokens")
        self.assertEqual(_contract(old, "legacy")["identity"], old["identity"])

    def test_native_buffer_reserves_both_full_tensors_at_padding_bound(self):
        exp = experiment(pp=3, pipeline_transport_max_tokens=32)
        ranks = MemoryPlanner(exp).snapshot()
        one_boundary = 2 * 32 * 128 * 2 * exp.strategy.pipeline_max_inflight_microbatches
        persistent = 2 * 32 * 128 * 2
        self.assertEqual([rank["pipeline_buffer_reserved_bytes"] for rank in ranks],
                         [one_boundary] * 2 + [3 * one_boundary + persistent] * 2
                         + [2 * one_boundary + persistent] * 2)
        components = ranks[2]["pipeline_buffer_breakdown"]
        self.assertEqual(components["incoming_slice_backing_bytes"], one_boundary)
        self.assertEqual(components["persistent_input_copy_bytes"], persistent)
        unsliced = MemoryPlanner(experiment(tp=1, pp=3, pipeline_transport_max_tokens=32)).snapshot()
        self.assertEqual([rank["pipeline_buffer_reserved_bytes"] for rank in unsliced],
                         [one_boundary, 2 * one_boundary + persistent, one_boundary + persistent])
        self.assertEqual(unsliced[1]["pipeline_buffer_breakdown"]["incoming_slice_backing_bytes"], 0)
        legacy = replace(exp, strategy=replace(exp.strategy, pipeline_transport="full_hidden"))
        self.assertEqual(MemoryPlanner(legacy).snapshot()[0]["pipeline_buffer_reserved_bytes"],
                         one_boundary // 4)


class NativePPProfileTests(unittest.TestCase):
    def test_calibration_preserves_padding_descriptor_and_rejects_conflicting_repeats(self):
        exp = experiment()
        first = record(exp, 0)
        repeated = deepcopy(first)
        repeated.update(record_id="repeat", duration_ms=5)
        data = dataset(exp, [first, repeated])
        original = deepcopy(data)
        profile = calibrate_dataset(data, profile_version="native", allow_test_fixture=True)
        self.assertEqual(profile["samples"][0]["duration_ms"], 4)
        self.assertEqual(profile["samples"][0]["pp_transport"], first["pp_transport"])
        self.assertEqual(data, original)
        repeated["pp_transport"]["outgoing"] = native_pp_descriptor(exp, 16)
        with self.assertRaises(SimulationError) as raised:
            calibrate_dataset(dataset(exp, [first, repeated]), profile_version="conflict", allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_native_rows_require_exact_directions_and_valid_padded_size(self):
        exp = experiment()
        good = record(exp, 0)
        invalid = []
        missing = deepcopy(good)
        missing.pop("pp_transport")
        invalid.append(missing)
        extra = deepcopy(good)
        extra["pp_transport"]["incoming"] = native_pp_descriptor(exp, 8)
        invalid.append(extra)
        for padded in (2, 17):
            bad = deepcopy(good)
            # Use a permissive builder only to produce an invalid supplied row.
            bound = replace(exp, strategy=replace(exp.strategy, pipeline_transport_max_tokens=32))
            bad["pp_transport"]["outgoing"] = native_pp_descriptor(bound, padded)
            invalid.append(bad)
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(SimulationError):
                validate_dataset(dataset(exp, [row]), allow_test_fixture=True)
        legacy = experiment(pipeline_transport="full_hidden")
        with self.assertRaises(SimulationError):
            validate_dataset(dataset(legacy, [good]), allow_test_fixture=True)

    def test_native_profile_is_exact_even_if_extrapolation_is_requested(self):
        exp = experiment()
        rows = [record(exp, 0, query=(q,), padded=8) for q in (2, 6)]
        rows[1]["record_id"] = "other"
        profile = calibrate_dataset(dataset(exp, rows), profile_version="grid", allow_test_fixture=True)
        exact = estimate_profile(profile, "prefill", [2], [0], allow_test_fixture=True)
        self.assertEqual(exact.breakdown["pp_transport"]["outgoing"]["padded_tokens"], 8)
        for allow in (False, True):
            with self.subTest(allow=allow), self.assertRaises(SimulationError) as raised:
                estimate_profile(profile, "prefill", [4], [0], allow_test_fixture=True, allow_extrapolation=allow)
            self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def profile(self, exp, *, mismatched_incoming=False):
        rows = [record(exp, stage, duration=3 if stage == 0 else 7) for stage in range(2)]
        if mismatched_incoming:
            rows[1]["pp_transport"]["incoming"] = native_pp_descriptor(exp, 4)
        rows += [record(exp, stage, phase="decode", query=(1,), context=(4,), padded=8,
                        duration=2 if stage == 0 else 3) for stage in range(2)]
        return calibrate_dataset(dataset(exp, rows), profile_version="native", allow_test_fixture=True)

    def test_profile_stage_plus_observed_payload_analytical_boundary_has_no_double_charge(self):
        exp = experiment()
        profile = self.profile(exp)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile))
            configured = replace(exp, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                profile_version="native", allow_test_fixture=True))
            result = simulate(configured)
            prediction = predict_experiment(configured)
        edge = 4 + 4096 / 1e6
        self.assertAlmostEqual(result.requests[0]["completion_ms"], 10 + edge)
        for observed, expected in zip(result.result["pipeline"]["stage_busy_ms"], (3, 7)):
            self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(prediction["isolated_prefill_ms"], 10 + edge)
        transfer = next(event for event in result.events if event["event"] == "TRANSFER_START")
        self.assertEqual(transfer["bytes_per_rank"], 2048)
        self.assertEqual(transfer["pp_transport"]["padded_tokens"], 8)
        self.assertEqual(transfer["source"], "analytical_uncalibrated")
        self.assertEqual(transfer["breakdown"]["payload_source"], "observed_profile_descriptor")
        self.assertEqual(prediction["communication"]["prefill_p2p_bytes_per_rank_per_boundary"], 2048)

    def test_ideal_zero_disables_native_edge_but_keeps_real_geometry(self):
        exp = experiment(pipeline_p2p=False)
        result = simulate(exp)
        self.assertEqual(result.requests[0]["completion_ms"], 10)
        edge = next(event for event in result.events if event["event"] == "TRANSFER_START")
        self.assertEqual(edge["source"], "explicit_ideal_zero_cost")
        self.assertEqual(edge["bytes_per_rank"], 1024)
        self.assertEqual(edge["start_ms"], edge["end_ms"])

    def test_adjacent_profile_geometry_mismatch_fails_before_reservation(self):
        exp = experiment()
        profile = self.profile(exp, mismatched_incoming=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile))
            configured = replace(exp, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                allow_test_fixture=True))
            engine = PipelineSimulation(configured)
            with self.assertRaises(SimulationError) as raised:
                engine.run()
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
            self.assertFalse(engine.memory.allocations)
            self.assertFalse(engine.waves)
            self.assertFalse(engine.transfers)
            with self.assertRaises(SimulationError):
                predict_experiment(configured)


class NativePPServingTests(unittest.TestCase):
    def test_r2_tp2_pp2_cache_chunk_swap_mixed_and_profile_remain_composable(self):
        base = load_experiment(ROOT / "examples/v1_6/qwen3-r2-tp2-pp2.json")
        exp = replace(base, strategy=replace(base.strategy, pipeline_transport="vllm_native",
                                             pipeline_p2p=True, pipeline_transport_max_tokens=8))
        baseline = simulate(exp)
        profiles = []
        for replica in range(2):
            local = replica_experiment(exp, replica)
            rows = []
            for index, event in enumerate(baseline.events):
                if event["event"] != "STAGE_START" or event["replica_index"] != replica:
                    continue
                row = record(local, event["stage"], phase=event["phase"], query=event["query_tokens"],
                    context=event["context_lengths"], padded=sum(event["query_tokens"]),
                    roles=event.get("request_phases"), duration=event["duration_ms"])
                row.update(record_id=f"declared-fixture-{index}", group_id=f"shape-{index}")
                rows.append(row)
            profiles.append(calibrate_dataset(dataset(local, rows), profile_version="combo", allow_test_fixture=True))
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for replica, profile in enumerate(profiles):
                path = Path(temporary) / f"replica-{replica}.json"
                path.write_text(json.dumps(profile))
                paths.append(str(path))
            calibrated = replace(exp, calibration=replace(exp.calibration, backend="profile",
                replica_profile_paths=tuple(paths), profile_version="combo", allow_test_fixture=True))
            replay = simulate(calibrated)
        self.assertEqual(replay.requests, baseline.requests)
        for result in (baseline, replay):
            self.assertEqual(result.result["counts"]["completed"], 10)
            self.assertGreater(result.result["batching"]["mixed_batches_completed"], 0)
            self.assertGreater(result.result["prefix_cache"]["hit_tokens"], 0)
            self.assertEqual(result.result["swap"]["completed_transfers_by_direction"], {"D2H": 2, "H2D": 2})
            self.assertEqual(result.result["recomputed_tokens"], 0)
            self.assertEqual(result.result["prefix_cache"]["attached_block_references"], 0)
            self.assertEqual(result.result["prefix_cache"]["private_blocks"], 0)
            self.assertEqual(result.result["swap"]["host_reserved_bytes"], 0)
            self.assertTrue(all(rank["used_blocks"] == 0 for rank in result.result["memory"]))
            edges = [event for event in result.events if event["event"] == "TRANSFER_START"]
            self.assertTrue(edges)
            self.assertTrue(all(len(event["pp_transport"]["tensors"]) == 2 for event in edges))


if __name__ == "__main__":
    unittest.main()
