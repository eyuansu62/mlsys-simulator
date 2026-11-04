"""Explicit native PP protocol policy and stage identity remain separate domains.

All observations are CPU fixtures. Existing lookup lifecycle tests are replayed
with v3 communication metadata rather than duplicating their serving scenarios.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import SimulationError, calibrate_dataset, load_experiment, simulate
from mlsys_sim.pp_benchmark import normalize_pp_benchmark_plan, worker_launch_contract
from mlsys_sim.pp_communication import (load_pp_communication_dataset, load_pp_communication_profile,
                                      validate_pp_communication_holdout)
from mlsys_sim.vllm_hooks import (_config_digest, _execution_config, install_runner_hook,
                                SUPPORTED_VERSION_PREFIX)
from test_native_pp_transport import dataset as stage_dataset, experiment as native_experiment
from test_native_pp_transport import record as stage_record
import test_pp_small_messages_integration as lookup_integration
from test_vllm_hooks import Clock, Runner, experiment as hook_experiment, schedule, state


POLICY = "simple_only_thresholds_v1"
ENVIRONMENT = {"NCCL_P2P_LL_THRESHOLD": "0", "NCCL_P2P_LL128_THRESHOLD": "0"}
TRANSPORT_KEYS = (*ENVIRONMENT, "NCCL_SHM_USE_CUDA_MEMCPY", "NCCL_SHM_MEMCPY_MODE")
ORIGINAL_LOOKUP_DATASET = lookup_integration.lookup_dataset


def policy_dataset(*args, **kwargs):
    data = ORIGINAL_LOOKUP_DATASET(*args, **kwargs)
    benchmark = data["benchmark"]
    benchmark.update(protocol="native_pp_primitives_v3",
        communicator_initialization="lazy_world_and_pair_groups_v1",
        p2p_protocol_policy=POLICY, p2p_protocol_environment=dict(ENVIRONMENT))
    benchmark["stack"]["p2p_protocol_environment"] = dict(ENVIRONMENT)
    return data


class RuntimePolicyIdentityTests(unittest.TestCase):
    def test_unset_transport_options_preserve_the_prior_snapshot_digest(self):
        with patch.dict(os.environ, {}, clear=True):
            snapshot = _execution_config(Runner().vllm_config, hook_experiment(), False)
        self.assertEqual(_config_digest(snapshot),
                         "0823d81d7bcad2a95505532e32012106d51d3067076586e19226520e02709223")
        self.assertTrue(all(name not in snapshot["environment"] for name in TRANSPORT_KEYS))

    def test_only_explicit_transport_values_are_bound_including_zero_and_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            default = _execution_config(Runner().vllm_config, hook_experiment(), False)
            for key in TRANSPORT_KEYS:
                for value in ("0", "1", ""):
                    with self.subTest(key=key, value=value), patch.dict(os.environ,
                            {key: value, "UNRELATED_PRIVATE_VALUE": "must-not-appear"}):
                        snapshot = _execution_config(Runner().vllm_config, hook_experiment(), False)
                        self.assertEqual(snapshot["environment"][key], value)
                        self.assertNotEqual(_config_digest(snapshot), _config_digest(default))
                        self.assertNotIn("must-not-appear", json.dumps(snapshot))
                        self.assertEqual(set(snapshot["environment"]) - set(default["environment"]), {key})

    def test_transport_changes_during_capture_quarantine_later_samples(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in TRANSPORT_KEYS:
                with self.subTest(key=key):
                    runner = Runner()
                    runner.requests = {"r": state(4)}
                    hook = install_runner_hook(runner, hook_experiment(), session_id="policy-cpu-fixture",
                        runtime_version=SUPPORTED_VERSION_PREFIX + ".d20261002", clock_ns=Clock())
                    self.addCleanup(hook.close)
                    runner.execute_model(schedule({"r": 4}))
                    runner.sample_tokens(None)
                    previous = hook.header["metadata"]["runtime_config_sha256"]
                    with patch.dict(os.environ, {key: "0"}):
                        runner.execute_model(schedule({"r": 4}))
                        runner.sample_tokens(None)
                    self.assertEqual(hook.header["metadata"]["runtime_config_sha256"], previous)
                    self.assertEqual(hook.records[1]["error_code"], "HARDWARE_IDENTITY_MISMATCH")
                    hook.close()

    def test_stage_profiles_cannot_cross_default_and_explicit_transport_domains(self):
        exp = native_experiment()
        with patch.dict(os.environ, {}, clear=True):
            default_hash = _config_digest(_execution_config(Runner().vllm_config, exp, False))
            with patch.dict(os.environ, ENVIRONMENT):
                policy_hash = _config_digest(_execution_config(Runner().vllm_config, exp, False))
        self.assertNotEqual(default_hash, policy_hash)
        with tempfile.TemporaryDirectory() as temporary:
            for label, selected, other in (("default", default_hash, policy_hash),
                                            ("simple", policy_hash, default_hash)):
                local = replace(exp, calibration=replace(exp.calibration, runtime_config_sha256=selected))
                rows = [stage_record(local, stage, padded=4) for stage in range(2)]
                rows += [stage_record(local, stage, phase="decode", query=(1,), context=(4,), padded=1)
                         for stage in range(2)]
                profile = calibrate_dataset(stage_dataset(local, rows), profile_version=label,
                                            allow_test_fixture=True)
                path = Path(temporary) / f"{label}.json"
                path.write_text(json.dumps(profile))
                selected_exp = replace(local, calibration=replace(local.calibration, backend="profile",
                    profile_path=str(path), profile_version=label, allow_test_fixture=True))
                self.assertEqual(simulate(selected_exp).result["counts"]["completed"], 1)
                mismatched = replace(selected_exp, calibration=replace(selected_exp.calibration,
                    runtime_config_sha256=other))
                with self.subTest(profile=label), self.assertRaises(SimulationError) as raised:
                    simulate(mismatched)
                self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")


class ProtocolPolicyCompositionTests(unittest.TestCase):
    def replay(self, method):
        case = lookup_integration.SmallMessageIntegrationTests()
        with patch.object(lookup_integration, "lookup_dataset", policy_dataset):
            getattr(case, method)()

    def test_v3_lookup_counts_two_tensors_paired_rank_max_and_allgather_once(self):
        self.replay("test_exact_and_interpolated_boundaries_charge_each_tensor_and_pair_max_once")
        self.replay("test_stage_profile_padding_and_native_lookup_do_not_double_charge_communication")

    def test_v3_missing_or_guarded_domain_still_fails_before_kv_mutation(self):
        self.replay("test_discontinuity_and_size_gaps_fail_before_kv_even_with_extrapolation")

    def test_v3_lookup_preserves_three_profiles_replicas_cache_swap_and_mixed(self):
        self.replay("test_dense_lookup_composes_with_three_profiles_replicas_cache_chunk_swap_and_mixed")

    def test_the_legacy_replay_fixture_remains_v1_without_policy_metadata(self):
        before = ORIGINAL_LOOKUP_DATASET(native_experiment())
        selected = policy_dataset(native_experiment())
        after = ORIGINAL_LOOKUP_DATASET(native_experiment())
        self.assertEqual(before, after)
        self.assertEqual(before["benchmark"]["protocol"], "native_pp_primitives_v1")
        self.assertNotIn("p2p_protocol_policy", before["benchmark"])
        self.assertEqual(selected["benchmark"]["p2p_protocol_environment"], ENVIRONMENT)
        self.assertEqual(selected["records"], before["records"])

    def test_public_v3_fixture_and_live_template_have_the_same_explicit_policy(self):
        root = Path(__file__).resolve().parents[1] / "examples/v1_11"
        exp = load_experiment(root / "native-pp-collect.fixture.json")
        plan = normalize_pp_benchmark_plan(exp, json.loads((root / "collector.plan.live-template.json").read_text()))
        contract = worker_launch_contract(exp, plan)
        self.assertEqual(contract["protocol"], "native_pp_primitives_v3")
        self.assertEqual(contract["p2p_protocol_policy"], POLICY)
        self.assertEqual(contract["p2p_protocol_environment"], ENVIRONMENT)
        self.assertFalse(contract["world_init_device_id"])
        profile = load_pp_communication_profile(root / "communication.profile.fixture.json", allow_test_fixture=True)
        heldout = load_pp_communication_dataset(root / "communication-holdout.fixture.json", allow_test_fixture=True)
        report = validate_pp_communication_holdout(profile, heldout, allow_test_fixture=True)
        self.assertEqual(report["coverage"]["fraction"], 1)
        self.assertLess(report["metrics"]["mae_ms"], 1e-15)
        self.assertTrue(all(row["training_size_relation"] == "unseen" for row in report["records"]))
        self.assertFalse(report["hardware_validated"])


if __name__ == "__main__":
    unittest.main()
