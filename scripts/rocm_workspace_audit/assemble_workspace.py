"""Merge independently checked live storage evidence; never patch missing facts."""

from pathlib import Path
import argparse
import gzip
import hashlib
import json
from mlsys_sim.config import experiment_from_dict
from mlsys_sim.profiles import profile_identity
from mlsys_sim.workspace import validate_workspace_dataset
from workspace_torch_audit import audit_torch_workspace
from native_backing_audit import audit_native_backing
from workspace_domain import recorded_session_domain_issues


def load(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--native", required=True)
    p.add_argument("--receipt", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    base = Path(a.session)
    out = Path(a.out)
    out.mkdir()
    s = load(base / "session.json")
    native = load(Path(a.native))
    tp = load(base / "torch-native-records.json")
    receipt = load(Path(a.receipt))
    e = experiment_from_dict(s["experiment"])
    issues = recorded_session_domain_issues(s)
    epochs = {}
    raw = []
    if receipt.get("returncode") != 0:
        issues.append("PROCESS_EXIT_NOT_PROVEN_ZERO")
    if (
        receipt.get("driver_source_sha256") != s["source_sha256"]
        or not receipt.get("driver_source_reference")
        or sha(Path(receipt["driver_source_reference"])) != s["source_sha256"]
    ):
        issues.append("EXECUTED_DRIVER_SOURCE_NOT_BOUND")
    if len(s["records"]) != 6 or any(
        x["phase"] != "prefill"
        or x["stage_index"] != 0
        or x["query_tokens"] not in [[16], [32]]
        or x["context_lengths"] != [0]
        for x in s["records"]
    ):
        issues.append("UNSUPPORTED_ACTUAL_SHAPE_COHORT")
    if s["fresh_pid"] != native["pid"] or s["fresh_pid"] != tp["install_thread_id"]:
        issues.append("PROCESS_BINDING_MISMATCH")
    if s["torch_git"] != "a8ec1f47c5b233927bf23f5cd8f68db3475af226":
        issues.append("UNSUPPORTED_TORCH_BUILD")
    for record in s["records"]:
        ep = load(base / ("epoch-" + str(record["epoch_id"]) + ".json.gz"))
        if any(
            ep.get(key) != record.get(key)
            for key in (
                "record_id",
                "group_id",
                "epoch_id",
                "phase",
                "stage_index",
                "query_tokens",
                "context_lengths",
            )
        ):
            issues.append("EPOCH_SESSION_RECORD_BINDING_MISMATCH:" + str(record["epoch_id"]))
        epochs[record["epoch_id"]] = {"before": ep["before"], "after": ep["after"]}
        raw.append(ep)
    agents = [
        z
        for z in native["agent_inventory"]["agents"]
        if z["type_name"] == "GPU" and z["product_name"] == e.infra.devices[0].name
    ]
    if len(agents) != 1:
        issues.append("DEVICE_MAPPING_NOT_UNIQUE")
    binding = {
        "epoch_id": 5001,
        "epoch_label": "explicit_ATen_LT_workspace_initialization",
        "entrypoint": "ATen.getCUDABlasLtWorkspace",
        "lt_only": True,
        "driver_source_sha256": s["source_sha256"],
        "driver_source_reference": receipt.get("driver_source_reference", ""),
    }
    n = audit_native_backing(
        native,
        epochs,
        device_to_agent={0: agents[0]["handle"]} if len(agents) == 1 else {},
        ctor_binding=binding,
    )
    write(out / "native-audit.json", n)
    if n["status"] != "NATIVE_BACKING_CHECKS_PASSED":
        issues.append("NATIVE_BACKING_AUDIT_NOT_CLOSED")
    records = []
    fixed = []
    audits = []
    for ep in raw:
        t = audit_torch_workspace(
            model=s["experiment"]["strategy"]["model"],
            before_snapshot=ep["before"],
            after_snapshot=ep["after"],
            owners_before=ep["owners_before"],
            owners_after=ep["owners_after"],
            torch_probe=tp,
            epoch_id=ep["epoch_id"],
            history_max_entries=s["history_max_entries"],
        )
        write(out / ("torch-audit-" + str(ep["epoch_id"]) + ".json"), t)
        audits.append(t)
        if t["status"] != "CLOSED_CANDIDATE":
            issues.append("TORCH_AUDIT_NOT_CLOSED:" + str(ep["epoch_id"]))
            continue
        values = t["fixed_requested_bytes_by_category"]
        runtime = values.get("runtime", 0) + (n["native_fixed_runtime_bytes"] or 0)
        fixed.append(runtime)
        if runtime > e.strategy.memory.runtime_bytes:
            issues.append("FIXED_RUNTIME_BUDGET_EXCEEDED")
        if values.get("kv_pool", 0) > e.strategy.memory.kv_pool_bytes:
            issues.append("KV_BUDGET_EXCEEDED")
        witness = [v for v in s["return_witness"] if v["epoch_id"] == ep["epoch_id"]]
        if (
            len(witness) != 1
            or witness[0]["type"] != "vllm.v1.outputs.ModelRunnerOutput"
            or witness[0]["gpu_tensor_count"] != 0
            or not witness[0]["execute_state_is_none"]
            or not witness[0]["prev_sampled_is_none"]
        ):
            issues.append("CPU_RETURN_NOT_CLOSED")
        records.append(
            {
                k: ep[k]
                for k in ["record_id", "group_id", "phase", "stage_index", "query_tokens", "context_lengths"]
            }
        )
        records[-1]["workspace_bytes_per_rank"] = [{"tp_rank": 0, "peak_bytes": t["requested_storage_peak"]}]
    input_paths = [
        base / "session.json",
        base / "torch-native-records.json",
        Path(a.native),
        Path(a.receipt),
        *[base / ("epoch-" + str(v["epoch_id"]) + ".json.gz") for v in s["records"]],
    ]
    summary = {
        "status": "PARTIAL" if issues else "ATTRIBUTION_COMPLETE_SUPPORTED_DOMAIN",
        "attribution_complete": not issues,
        "hardware_validated": False,
        "issues": sorted(set(issues)),
        "supported_domain": "pinned Qwen3-1.7B BF16 TP1PP1 synchronous eager singleprefill q16/32 context0 greedy1",
        "byte_basis": "owned_requested_storage_capacity",
        "fixed_runtime_component_bounds": fixed,
        "runtime_budget_bytes": e.strategy.memory.runtime_bytes,
        "input_sha256": {p.name: sha(p) for p in input_paths},
        "source_session": s["session_id"],
    }
    write(out / "attribution-report.json", summary)
    if not issues:
        data = {
            "schema_version": 1,
            "kind": "workspace_measurements",
            "dataset_id": "actual-workspace-" + s["cohort"] + "-" + s["session_id"],
            "identity": profile_identity(e),
            "source": {
                "kind": "measured",
                "reference": "Actual pinned Qwen3 Runtime allocator/native owner ledger; complete only at measured requested-storage domain",
            },
            "records": records,
            "attribution": {
                "method": "runtime_workspace_ledger_v1",
                "complete": True,
                "scope": "exclusive_dynamic_workspace",
                "excluded_categories": [
                    "weights",
                    "kv_pool",
                    "graphs",
                    "runtime",
                    "pp_buffers",
                    "communication_buffers",
                ],
                "includes_non_torch": True,
            },
            "notes": {
                "byte_basis": summary["byte_basis"],
                "domain": summary["supported_domain"],
                "acquisition": {
                    "session_id": s["session_id"],
                    "clock_domain": s["clock_domain"],
                    "capture_input_sha256": sorted(set(summary["input_sha256"].values())),
                },
                "attribution_report_sha256": sha(out / "attribution-report.json"),
                "fixed_runtime_component_bound": max(fixed),
            },
        }
        data = validate_workspace_dataset(data)
        write(out / "dataset.json", data)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "issues": summary["issues"],
                "peaks": [r["workspace_bytes_per_rank"] for r in records],
            }
        )
    )


if __name__ == "__main__":
    main()
