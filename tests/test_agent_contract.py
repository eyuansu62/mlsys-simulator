"""Interchange conformance checks using explicit mock records, not timing fits."""

from copy import deepcopy
import hashlib
import json
import math
import unittest

from mlsys_sim.agent_contract import SCHEMA, validate_agent_request, validate_agent_response
from mlsys_sim.schema import SimulationError


def request_fixture(mode="open_loop"):
    rows = [{"request_id": "parent", "input_tokens": 16, "output_tokens_target": 2},
            {"request_id": "child", "input_tokens": 32, "output_tokens_target": 2}]
    if mode == "open_loop":
        rows[0]["arrival_ms"], rows[1]["arrival_ms"] = 0, 5
    elif mode == "dag":
        rows[1].update(depends_on=["parent"], gap_after_deps_ms=2)
    return {
        "schema": SCHEMA, "client_request_id": "fixture-调用", "catalog_version": "fixture-v1",
        "task": "schema-conformance", "candidate_id": "fixture-not-performance",
        "workload": {"mode": mode, "source_sha256": "sha256:" + "a" * 64, "requests": rows},
        "infra": {"id": "fixture", "nodes": [{"id": "n0", "devices": [{"id": "gpu0", "hardware": "fixture"}]}]},
        "strategy": {"deployments": [{"model": "model-a", "weight_precision": "BF16",
                                       "context_tokens": 128, "devices": ["gpu0"]}]},
        "evaluation": {"constraints": [{"metric": "example", "op": "<=", "value": None}],
                       "duration_s": 30, "timeout_s": 10, "goodput_policy": "both_thresholds_required"},
        "execution": {"seed": 42, "repeat": 9, "split": "development", "holdout": None,
                      "initial_state": "empty", "warmup": None, "max_simulated_ms": 100},
    }


def response_fixture(request):
    body = json.dumps(request, ensure_ascii=False, indent=2).encode("utf-8")
    second_ready = {"open_loop": 5, "sequential": 10, "dag": 12}[request["workload"]["mode"]]
    rows = []
    for name, ready, first, end in (("parent", 0, 3, 10), ("child", second_ready, second_ready + 2, second_ready + 4)):
        rows.append({"request_id": name, "status": "completed", "ready_ms": ready,
                     "started_ms": ready, "first_token_ms": first, "finished_ms": end,
                     "service_ms": end - ready, "output_tokens": 2, "output_tokens_target": 2,
                     "deployment_model": "model-a", "replica_index": 0, "device_ids": ["gpu0"]})
    return body, {
        "schema": SCHEMA, "client_request_id": request["client_request_id"],
        "input_sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
        "status": "succeeded", "failure_class": None, "error": None, "evidence_level": "mock",
        "backend": {"name": "explicit-fixture-not-an-engine", "version": "1"},
        "catalog_version": request["catalog_version"], "model_revisions": {"model-a": "fixture-revision"},
        "calibration": {"version": None, "kind": "uncalibrated", "in_domain": None,
                        "assumptions": ["Mock protocol records; no performance evidence"]},
        "effective_config": {"infra": deepcopy(request["infra"]), "strategy": deepcopy(request["strategy"])},
        "requests": rows,
        "per_device": [{"device_id": "gpu0", "capacity_bytes": 4096, "physical_peak_bytes": 2048,
                        "kv_peak_bytes": 512, "kv_pool_reserved_bytes": 1024}],
        "unsupported_fields": {}, "logs": [],
    }


def set_path(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


class AgentRequestContractTests(unittest.TestCase):
    def assert_invalid(self, request, field=None):
        with self.assertRaises(SimulationError) as raised:
            validate_agent_request(request)
        self.assertEqual(raised.exception.code, "INVALID_AGENT_REQUEST")
        if field is not None:
            self.assertIn(field, str(raised.exception))

    def test_three_modes_preserve_raw_values_and_return_an_independent_copy(self):
        for mode in ("open_loop", "sequential", "dag"):
            with self.subTest(mode=mode):
                original = request_fixture(mode)
                before = deepcopy(original)
                validated = validate_agent_request(original)
                self.assertEqual(validated, before)
                validated["workload"]["requests"][0]["input_tokens"] = 64
                self.assertEqual(original, before)
                self.assertNotIn("serving", original["strategy"])
                self.assertEqual(original["execution"]["seed"], 42)
                self.assertEqual(original["execution"]["repeat"], 9)

    def test_catalog_placeholders_and_policy_names_are_structural_not_resolution(self):
        request = request_fixture()
        request["catalog_version"] = "to-be-agreed"
        request["strategy"]["deployments"][0]["model"] = "llm-3b"
        request["strategy"]["scheduling"] = {"priority": "critical_path", "preemption": "restart"}
        request["strategy"]["cache"] = {"prefix_cache": True}
        self.assertEqual(validate_agent_request(request), request)

    def test_types_finiteness_bounds_and_unknown_fields_are_strict(self):
        mutations = [
            (("schema",), "future-schema"), (("client_request_id",), ""),
            (("execution", "seed"), True), (("execution", "repeat"), 0.0),
            (("execution", "seed"), -1), (("execution", "seed"), 2**63),
            (("execution", "warmup"), []), (("execution", "initial_state"), "retained"),
            (("execution", "max_simulated_ms"), math.inf),
            (("evaluation", "duration_s"), 0), (("evaluation", "timeout_s"), math.nan),
            (("evaluation", "constraints", 0, "value"), True),
            (("workload", "requests", 0, "input_tokens"), 0),
            (("workload", "requests", 0, "input_tokens"), 65_537),
            (("workload", "requests", 0, "output_tokens_target"), 8193),
            (("workload", "requests", 0, "arrival_ms"), -1),
            (("workload", "requests", 0, "input_tokens"), True),
            (("workload", "source_sha256"), "a" * 64),
            (("strategy", "deployments", 0, "context_tokens"), 16),
            (("strategy", "deployments", 0, "weight_precision"), "FP4"),
        ]
        for path, replacement in mutations:
            with self.subTest(path=path, replacement=replacement):
                request = request_fixture()
                set_path(request, path, replacement)
                self.assert_invalid(request)
        for location in ((), ("workload",), ("strategy",), ("strategy", "deployments", 0)):
            with self.subTest(location=location):
                request = request_fixture()
                target = request
                for key in location:
                    target = target[key]
                target["unexpected"] = 1
                self.assert_invalid(request, "unknown fields")

    def test_missing_fields_duplicate_ids_and_large_request_count_are_rejected(self):
        request = request_fixture()
        del request["execution"]["holdout"]
        self.assert_invalid(request, "holdout")
        request = request_fixture()
        request["workload"]["requests"][1]["request_id"] = "parent"
        self.assert_invalid(request, "unique")
        request = request_fixture()
        request["workload"]["requests"] *= 5001
        self.assert_invalid(request, "10000")

    def test_fixed_release_rules_do_not_coerce_modes(self):
        cases = []
        request = request_fixture("sequential")
        request["workload"]["requests"][0]["arrival_ms"] = 0
        cases.append(request)
        request = request_fixture()
        del request["workload"]["requests"][0]["arrival_ms"]
        cases.append(request)
        request = request_fixture()
        request["workload"]["requests"][1]["gap_after_deps_ms"] = 1
        cases.append(request)
        request = request_fixture("dag")
        request["workload"]["requests"][1]["arrival_ms"] = 0
        cases.append(request)
        request = request_fixture("dag")
        request["workload"]["requests"][0]["gap_after_deps_ms"] = 1
        cases.append(request)
        for request in cases:
            self.assert_invalid(request)

    def test_dependency_graph_is_validated_without_requiring_topological_array_order(self):
        request = request_fixture("dag")
        request["workload"]["requests"].reverse()
        self.assertEqual(validate_agent_request(request), request)
        for parents in (["missing"], ["parent", "parent"], ["child"]):
            request = request_fixture("dag")
            request["workload"]["requests"][1]["depends_on"] = parents
            self.assert_invalid(request)
        request = request_fixture("dag")
        request["workload"]["requests"][0]["depends_on"] = ["child"]
        self.assert_invalid(request, "acyclic")

    def test_replica_tp_placement_and_model_context_are_checked(self):
        request = request_fixture()
        devices = [{"id": f"gpu{index}", "hardware": "fixture"} for index in range(4)]
        request["infra"]["nodes"][0]["devices"] = devices
        deployment = request["strategy"]["deployments"][0]
        deployment.update(devices=[row["id"] for row in devices], replicas=2, parallel={"kind": "tp", "degree": 2})
        self.assertEqual(validate_agent_request(request), request)
        for mutation in (lambda r: r["strategy"]["deployments"][0]["devices"].pop(),
                         lambda r: r["infra"]["nodes"][0]["devices"].append({"id": "gpu0", "hardware": "x"}),
                         lambda r: r["strategy"]["deployments"].append(deepcopy(r["strategy"]["deployments"][0])),
                         lambda r: r["workload"]["requests"][0].update(target_model="missing")):
            changed = deepcopy(request)
            mutation(changed)
            self.assert_invalid(changed)
        changed = deepcopy(request)
        changed["strategy"]["deployments"][0]["parallel"]["kind"] = "none"
        self.assert_invalid(changed, "degree")

    def test_multiple_deployments_require_unambiguous_targets_and_exclusive_devices(self):
        request = request_fixture("dag")
        request["infra"]["nodes"][0]["devices"].append({"id": "gpu1", "hardware": "fixture"})
        second = deepcopy(request["strategy"]["deployments"][0])
        second.update(model="model-b", devices=["gpu1"])
        request["strategy"]["deployments"].append(second)
        self.assert_invalid(request, "target_model")
        for index, row in enumerate(request["workload"]["requests"]):
            row["target_model"] = ["model-a", "model-b"][index]
        self.assertEqual(validate_agent_request(request), request)
        second["devices"] = ["gpu0"]
        self.assert_invalid(request, "exclusive")

    def test_serving_and_policy_shapes_are_strict_but_do_not_enforce_runtime_support(self):
        for serving in ({"max_concurrency": True}, {"gpu_memory_utilization": None},
                        {"gpu_memory_utilization": 0}, {"gpu_memory_utilization": 1.01},
                        {"kv_fraction": 0.5, "kv_budget_gib": 1},
                        {"chunked_prefill_tokens": 2.0}, {"continuous_batching": 1}, {"max_batch_size": 4}):
            request = request_fixture()
            request["strategy"]["serving"] = serving
            self.assert_invalid(request)
        for section, value in (("cache", {"prefix_cache": 1}), ("cache", {"prefix_group": "x"}),
                               ("scheduling", {"priority": 0}), ("scheduling", {"future": "fifo"}),
                               ("speculative", "false")):
            request = request_fixture()
            request["strategy"][section] = value
            self.assert_invalid(request)


class AgentResponseContractTests(unittest.TestCase):
    def assert_invalid(self, body, response, field=None):
        with self.assertRaises(SimulationError) as raised:
            validate_agent_response(body, response)
        self.assertEqual(raised.exception.code, "INVALID_AGENT_RESPONSE")
        if field is not None:
            self.assertIn(field, str(raised.exception))

    def test_all_three_modes_accept_consistent_mock_evidence_without_modifying_it(self):
        for mode in ("open_loop", "sequential", "dag"):
            body, response = response_fixture(request_fixture(mode))
            before = deepcopy(response)
            self.assertIsNone(validate_agent_response(body, response))
            self.assertEqual(response, before)

    def test_raw_utf8_body_hash_is_not_a_reserialized_json_hash(self):
        body, response = response_fixture(request_fixture())
        compact = json.dumps(json.loads(body), separators=(",", ":")).encode()
        self.assertNotEqual(body, compact)
        self.assert_invalid(compact, response, "input_sha256")
        response["input_sha256"] = "sha256:" + hashlib.sha256(compact).hexdigest()
        validate_agent_response(compact, response)
        self.assert_invalid(body.replace(b'"seed": 42', b'"seed": 42, "seed": 42'), response, "request_bytes")

    def test_response_identity_target_and_record_completeness_cannot_change(self):
        mutations = [
            (("client_request_id",), "other"), (("catalog_version",), "other"),
            (("requests", 0, "output_tokens_target"), 3),
            (("requests", 0, "request_id"), "other"),
            (("model_revisions",), {}), (("effective_config",), None),
        ]
        for path, replacement in mutations:
            body, response = response_fixture(request_fixture())
            set_path(response, path, replacement)
            self.assert_invalid(body, response)
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.append(deepcopy(rows[0]))):
            body, response = response_fixture(request_fixture())
            mutation(response["requests"])
            self.assert_invalid(body, response, "requests")

    def test_timestamp_placement_and_completed_output_evidence_are_required(self):
        mutations = [
            ("ready_ms", 1), ("first_token_ms", None), ("first_token_ms", math.nan),
            ("started_ms", 4), ("finished_ms", 101), ("output_tokens", 1),
            ("output_tokens", True), ("replica_index", 1), ("replica_index", False),
            ("deployment_model", "other"), ("device_ids", ["gpu1"]),
            ("failure_code", "FAKE_FAILURE"), ("service_ms", 20),
        ]
        for field, value in mutations:
            body, response = response_fixture(request_fixture())
            response["requests"][0][field] = value
            self.assert_invalid(body, response)

    def test_sequential_failure_continues_after_terminal_not_as_dependency_block(self):
        body, response = response_fixture(request_fixture("sequential"))
        response.update(status="failed", failure_class="request_error", error={"code": "OOM", "message": "fixture"})
        first, second = response["requests"]
        first.update(status="failed", started_ms=None, first_token_ms=None, finished_ms=1,
                     output_tokens=0, service_ms=None, failure_code="OOM")
        second.update(ready_ms=1, started_ms=1, first_token_ms=2, finished_ms=3, service_ms=2)
        validate_agent_response(body, response)
        second.update(ready_ms=2, started_ms=2, first_token_ms=3, finished_ms=4)
        self.assert_invalid(body, response, "causal release")

    def test_dag_gap_uses_actual_parent_completion_and_preserves_dependency_failures(self):
        body, response = response_fixture(request_fixture("dag"))
        response["requests"][1]["ready_ms"] = 10
        self.assert_invalid(body, response, "causal release")
        body, response = response_fixture(request_fixture("dag"))
        response.update(status="failed", failure_class="oom", error={"code": "OOM", "message": "fixture"})
        parent, child = response["requests"]
        parent.update(status="failed", first_token_ms=None, output_tokens=0, finished_ms=1,
                      service_ms=None, failure_code="OOM")
        self.assert_invalid(body, response, "parent failed")
        child.update(status="blocked", ready_ms=None, started_ms=None, first_token_ms=None,
                     finished_ms=1, service_ms=None, output_tokens=0, failure_code="DEPENDENCY_FAILED",
                     blocked_by=["parent"])
        validate_agent_response(body, response)
        for blocker in ("missing", "child"):
            child["blocked_by"] = [blocker]
            self.assert_invalid(body, response, "ancestors")

    def test_deadline_is_inclusive_but_later_observations_are_rejected(self):
        request = request_fixture("dag")
        request["execution"]["max_simulated_ms"] = 16
        body, response = response_fixture(request)
        validate_agent_response(body, response)
        response["requests"][1]["finished_ms"] = 16.0001
        self.assert_invalid(body, response, "deadline")

    def test_unknown_and_not_started_records_preserve_unknown_vs_zero(self):
        for status, output in (("unknown", None), ("not_started", 0)):
            body, response = response_fixture(request_fixture())
            response.update(status="failed", failure_class="backend_error",
                            error={"code": "LOST", "message": "fixture"}, effective_config=None)
            response["requests"] = [{"request_id": row["request_id"], "status": status,
                "ready_ms": None, "started_ms": None, "first_token_ms": None, "finished_ms": None,
                "output_tokens": output, "output_tokens_target": 2, "failure_code": "LOST"}
                for row in response["requests"]]
            validate_agent_response(body, response)
            response["requests"][0]["first_token_ms"] = 0
            if status == "not_started":
                self.assert_invalid(body, response)
            else:
                validate_agent_response(body, response)  # Some timestamps may survive a lost count.

    def test_missing_diagnostics_require_reasons_and_memory_is_not_double_counted(self):
        body, response = response_fixture(request_fixture())
        for row in response["requests"]:
            row["service_ms"] = None
        self.assert_invalid(body, response, "missing diagnostics")
        response["unsupported_fields"]["requests.service_ms"] = "No per-request service accounting"
        validate_agent_response(body, response)
        response["per_device"][0]["physical_peak_bytes"] = None
        self.assert_invalid(body, response, "unknown memory")
        response["unsupported_fields"]["per_device.physical_peak_bytes"] = "No physical observation"
        validate_agent_response(body, response)
        response["per_device"][0]["kv_peak_bytes"] = 1025
        self.assert_invalid(body, response, "reserved pool")

    def test_changed_effective_configuration_is_rejected_but_schema_defaults_match(self):
        body, response = response_fixture(request_fixture())
        response["effective_config"]["strategy"]["deployments"][0].update(
            kv_precision="BF16", replicas=1, parallel={"kind": "none", "degree": 1})
        response["effective_config"]["strategy"]["serving"] = {"max_concurrency": 1}
        validate_agent_response(body, response)
        response["effective_config"]["strategy"]["serving"]["max_concurrency"] = 2
        self.assert_invalid(body, response, "silently change")

    def test_calibration_scope_and_run_failure_status_are_not_promoted(self):
        for changes in ({"calibration": {"version": None, "kind": "uncalibrated", "in_domain": True, "assumptions": []}},
                        {"calibration": {"version": "x", "kind": "extrapolated", "in_domain": None, "assumptions": []}},
                        {"error": {"code": "OOM", "message": "fixture"}},
                        {"status": "failed"}, {"evidence_level": "measured"}):
            body, response = response_fixture(request_fixture())
            response.update(changes)
            self.assert_invalid(body, response)
        body, response = response_fixture(request_fixture())
        response["per_device"].append(deepcopy(response["per_device"][0]))
        self.assert_invalid(body, response, "distinct known")


if __name__ == "__main__":
    unittest.main()
