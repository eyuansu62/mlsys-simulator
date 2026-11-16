"""Import the supplied step-cost archive without executing its programs.

The split is retrospective: the aggregate report was observed before this
work.  Rep3 remains acquisition-session-disjoint from fitting, and its middle
shapes are excluded from every fitted component.  No row is trimmed.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile


MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b")
ARCHIVE_SHA256 = "32567daea53b3a0700d07be9a0f332577fddb2c87070f6515d43df15cb49bb85"
TIMING_BOUNDARY = "reported_l2_step_hook_wall_v1"
EXECUTION_MODE = "graph_full_and_piecewise_configured"
ENDPOINTS = {"S=128", "S=2048", "B=1,C=1024", "B=16,C=1024"}
MIDDLES = {"S=512", "B=4,C=1024"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reconstruct_contexts(schedule: list[list[int]], batch: int, prompt: int = 1024) -> list[list[int]]:
    """Derive full-decode context multisets under a checked uniform-prompt protocol.

    Each pre-full step must contain every existing sequence once as q=1; all
    other members must be new uniform prompts.  This does not assign request
    identities or infer contexts when subsets/reordering become ambiguous.
    """
    _require(type(batch) is int and batch > 0 and isinstance(schedule, list), "Invalid decode schedule")
    contexts: list[int] = []
    full: list[list[int]] = []
    tail = False
    for queries in schedule:
        _require(isinstance(queries, list) and bool(queries) and all(type(q) is int and q > 0 for q in queries),
                 "Invalid scheduled query lengths")
        is_full = len(queries) == batch and all(q == 1 for q in queries)
        if tail:
            _require(not is_full, "Full batch resumed after an ambiguous generation tail")
            continue
        if len(contexts) == batch and is_full:
            full.append(sorted(contexts))
            contexts = [c + 1 for c in contexts]
            continue
        if full:
            # Generation tails have fewer sequences; selected full-batch steps
            # cannot be inferred through them.  They are not cost observations.
            tail = True
            continue
        decodes = sum(q == 1 for q in queries)
        prefills = [q for q in queries if q != 1]
        _require(decodes == len(contexts), "Ambiguous preamble: a prior sequence was omitted")
        _require(all(q == prompt for q in prefills), "Preamble is not the declared uniform prompt")
        _require(len(contexts) + len(prefills) <= batch, "Preamble exceeds declared batch")
        contexts = [c + 1 for c in contexts] + [prompt] * len(prefills)
    _require(bool(full), "No full decode batch in schedule")
    return full


class _Source:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files: dict[str, str] = {}

    def read(self, relative: str) -> bytes:
        path = (self.root / relative).resolve()
        _require(path.is_relative_to(self.root), "Source path leaves the supplied package")
        data = path.read_bytes()
        digest = _sha(data)
        if relative in self.files:
            _require(self.files[relative] == digest, "Source changed during import")
        self.files[relative] = digest
        return data

    def json(self, relative: str):
        return json.loads(self.read(relative))

    def jsonl(self, relative: str):
        return [json.loads(line) for line in self.read(relative).splitlines() if line.strip()]

    def verify(self) -> None:
        for name, digest in self.files.items():
            _require(_sha((self.root / name).read_bytes()) == digest, "Source changed during import")

    def verify_archive(self, archive: Path) -> str:
        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        _require(digest.hexdigest() == ARCHIVE_SHA256, "Original archive SHA does not match this import protocol")
        with zipfile.ZipFile(archive) as package:
            for name, expected in self.files.items():
                _require(_sha(package.read("summary_package_20261116/" + name)) == expected,
                         "Extracted source does not match original archive bytes")
        return digest.hexdigest()


def _configured_controls(context: dict) -> dict:
    engine, environment = context["engine"], context["environment"]
    _require(engine["cudagraph_mode"] == "FULL_AND_PIECEWISE", "Unexpected configured graph mode")
    kwargs = deepcopy(context["llm_kwargs"])
    kwargs.pop("model", None)  # machine-local path is not an execution control
    kwargs.pop("profiler_config", None)  # separately labelled component-proxy instrumentation
    return {
        "scope": "configured_controls_v1",
        "engine": {key: engine[key] for key in ("runner_class", "use_v2_model_runner", "async_scheduling",
                                                "compilation_mode", "cudagraph_mode", "attention_impl")},
        "software": {key: environment.get(key) for key in ("vllm", "torch", "hip", "python", "image_id")},
        "model": {key: environment["model"].get(key) for key in ("key", "hf_revision")},
        "llm_kwargs": kwargs,
    }


def _experiment(source: _Source, model: str, context: dict) -> dict:
    name = f"5_raw/simulator_runs/experiments/{model}_rx9070xt_prefill_128_nominal.json"
    exp = deepcopy(source.json(name))
    environment = context["environment"]
    _require(exp["strategy"]["model"]["revision"] == environment["model"]["hf_revision"], "Model revision mismatch")
    exp["workload"] = []
    exp["strategy"].update({
        "max_batch_tokens": context["llm_kwargs"]["max_num_batched_tokens"],
        "max_batch_size": context["llm_kwargs"]["max_num_seqs"],
        "preemption": "none", "prefix_cache": False, "chunked_prefill": False, "mixed_batching": False,
    })
    exp["calibration"].update({
        "compute_efficiency": 0.5, "bandwidth_efficiency": 0.5,
        "launch_overhead_us": 2.0, "framework_overhead_us": 5.0,
        "runtime_name": "vllm", "runtime_version": environment["vllm"],
        "runtime_config_sha256": _sha(_canonical(_configured_controls(context))),
        "prefill_execution_mode": EXECUTION_MODE, "decode_execution_mode": EXECUTION_MODE,
        "prefill_attention_backend": context["engine"]["attention_impl"].split(".")[-1],
        "decode_attention_backend": context["engine"]["attention_impl"].split(".")[-1],
        "analytical_cost_timing_boundary": TIMING_BOUNDARY,
    })
    return exp


def _base_record(model: str, phase: str, shape: str, session: str, source_file: str,
                 source_sha: str, domain: dict, queries: list[int], contexts: list[int],
                 features: dict, group: str) -> dict:
    return {"model_id": model, "phase": phase, "shape": shape, "session_id": session,
            "group_id": f"{model}:{phase}:{shape}", "source_file": source_file, "source_sha256": source_sha,
            "query_tokens": queries, "context_lengths": contexts, "layers": features["layers"],
            "nominal_nonattention_ms": features["nonattention_ms"],
            "nominal_attention_ms": features["attention_ms"], "domain": domain,
            "split_group": group}


def build_dataset(package_root: str | Path, *, models: tuple[str, ...] = MODELS,
                  archive: str | Path | None = None) -> dict:
    """Build sanitized records and an explicit immutable-membership recommendation."""
    from mlsys_sim.analytical_cost import analytical_cost_identity, nominal_cost_features
    from mlsys_sim.config import experiment_from_dict
    from mlsys_sim.performance import AnalyticalModel

    source = _Source(Path(package_root))
    records, experiments, sessions = [], {}, {}
    split: dict[str, list[str]] = {name: [] for name in
        ("training_record_ids", "holdout_record_ids", "repeat_record_ids", "unused_record_ids")}
    feature_cache: dict[tuple, dict] = {}
    for model in models:
        _require(re.fullmatch(r"[a-z0-9][a-z0-9.-]*", model) is not None, "Invalid model directory ID")
        prefix = f"5_raw/real_machine/run_20261116a/raw/{model}"
        contexts = {run: source.json(f"{prefix}/{run}/run_context.json")
                    for run in ("timing_rep1", "timing_rep2", "timing_rep3", "profile")}
        exp_json = _experiment(source, model, contexts["timing_rep1"])
        exp = experiment_from_dict(exp_json)
        predictor = AnalyticalModel(exp)
        experiments[model] = exp_json
        controls = _configured_controls(contexts["timing_rep1"])
        identity = analytical_cost_identity(exp)
        domain_base = {"identity_sha256": identity, "execution_mode": EXECUTION_MODE,
                       "attention_backend": contexts["timing_rep1"]["engine"]["attention_impl"].split(".")[-1],
                       "timing_boundary": TIMING_BOUNDARY}

        def features(phase: str, queries: list[int], lengths: list[int]) -> dict:
            key = (model, phase, tuple(queries), tuple(lengths))
            if key not in feature_cache:
                feature_cache[key] = {**nominal_cost_features(exp, phase, queries, lengths),
                                      "default_ms": predictor.estimate_step(phase, queries, lengths).duration_ms}
            return feature_cache[key]

        def session(run: str) -> str:
            context = contexts[run]
            _require(_configured_controls(context) == controls, "Configured runtime controls changed between sessions")
            name = f"{prefix}/{run}/run_context.json"
            identity = f"{model}:{run}:{source.files[name]}"
            sessions[identity] = {"source_file": name, "source_sha256": source.files[name],
                "configured_controls": controls,
                "diagnostics": {"kv_cache_shape_layer0": context["engine"].get("kv_cache_shape_layer0"),
                                "observed_device": context["environment"].get("device")},
                "instrumentation": {key: value for key, value in context["llm_kwargs"].get("profiler_config", {}).items()
                                    if key != "torch_profiler_dir"},
                "identity_scope": "configured_controls_v1; KV allocation shape is an observed outcome, not a bound control",
                "actual_graph_dispatch": "not_observed"}
            return identity

        for repeat in (1, 2, 3):
            run = f"timing_rep{repeat}"
            session_id = session(run)
            path = f"{prefix}/{run}/l2_timing.jsonl"
            all_rows = source.jsonl(path)
            rows = [row for row in all_rows if row.get("kind") == "step_hooked"]
            _require(len(rows) == 6, "Expected all six untrimmed hooked shapes per timing process")
            for raw_index, row in enumerate(rows):
                phase, shape = row["phase"], row["shape"]
                _require(row["model"] == model and phase in ("prefill", "decode")
                         and shape in ENDPOINTS | MIDDLES, "Unexpected trace model/phase/shape")
                samples = row["steps"]
                _require(len(samples) == row["n"] and len(samples) == (100 if phase == "prefill" else 128),
                         "Missing or extra formal timing samples")
                role = ("training" if repeat in (1, 2) and shape in ENDPOINTS else
                        "holdout" if repeat == 3 and shape in MIDDLES else
                        "repeat" if repeat == 3 else "unused")
                schedules: dict[int, list[list[int]]] = {}
                for index, step in enumerate(samples):
                    queries = step["num_scheduled"]
                    if phase == "prefill":
                        _require(len(queries) == 1 and step["context_min"] == step["context_max"] == 0,
                                 "Unexpected Prefill shape/context")
                        _require(queries == [int(shape.split("=")[1])], "Prefill shape label does not match queries")
                        lengths = [0]
                        evidence = {"method": "observed_zero_context", "exact_multiset": True}
                    else:
                        _require(all(q == 1 for q in queries), "Selected decode sample contains Prefill")
                        _require(len(queries) == int(shape.split(",", 1)[0].split("=")[1]),
                                 "Decode shape label does not match batch")
                        generation = step["generate_index"]
                        if step.get("all_steps_num_scheduled") is not None:
                            _require(generation not in schedules, "Duplicate generation schedule")
                            schedules[generation] = reconstruct_contexts(step["all_steps_num_scheduled"], len(queries))
                        _require(generation in schedules, "Missing decode generation preamble")
                        position = step["step_index"]
                        _require(type(position) is int and 0 <= position < len(schedules[generation]),
                                 "Decode sample index exceeds validated schedule")
                        lengths = schedules[generation][position]
                        _require(min(lengths) == step["context_min"] and max(lengths) == step["context_max"],
                                 "Reconstructed contexts disagree with observed bounds")
                        evidence = {"method": "reconstructed_multiset_not_per_request_observation",
                            "exact_multiset": True, "observed_min": step["context_min"],
                            "observed_max": step["context_max"], "generate_index": generation,
                            "selected_step_index": position,
                            "assumption": "uniform1024 prompts; all prior active sequences retained in checked preamble"}
                    record = _base_record(model, phase, shape, session_id, path, source.files[path],
                                          domain_base, queries, lengths, features(phase, queries, lengths), role)
                    record.update(record_id=f"{model}:{run}:{raw_index}:{index}", kind="step_wall",
                                  observed_ms=step["step_ms"], context_evidence=evidence,
                                  default_ms=features(phase, queries, lengths)["default_ms"],
                                  raw_record_index=all_rows.index(row), raw_sample_index=index)
                    records.append(record)
                    split[role + "_record_ids"].append(record["record_id"])

        profile_session = session("profile")
        summary_path = f"{prefix}/profile/profile_summary.json"
        summaries = source.json(summary_path)
        schedule_path = f"{prefix}/profile/l2_profile.jsonl"
        profile_schedules = {(row["phase"], row["shape"]): row for row in source.jsonl(schedule_path)}
        _require(len(summaries) == 6, "Expected every profile component cell")
        for index, row in enumerate(summaries):
            phase, shape = row["phase"], row["shape"]
            schedule = profile_schedules[(phase, shape)]["all_steps_num_scheduled"]
            if phase == "prefill":
                queries, lengths = schedule[0], [0]
                _require(all(item == queries for item in schedule) and len(queries) == 1,
                         "Ambiguous Prefill profile schedule")
                context_evidence = {"method": "observed_profile_uniform_prefill", "approximate": False}
            else:
                batch = int(shape.split(",", 1)[0].split("=")[1])
                candidates = reconstruct_contexts(schedule, batch)
                upper = max(max(item) for item in candidates)
                queries, lengths = [1] * batch, [upper] * batch
                context_evidence = {"method": "profile_aggregate_upper_representative",
                    "approximate": True, "context_min": min(min(item) for item in candidates),
                    "context_max": upper, "candidate_steps": len(candidates),
                    "aggregate_steps": row["target_steps"], "selected_indices": "not_observed",
                    "assumption": "conservative upper context from this profile's own schedule; not timing-process contexts"}
            role = "training" if shape in ENDPOINTS else "unused"
            record = _base_record(model, phase, shape, profile_session, summary_path, source.files[summary_path],
                                  domain_base, queries, lengths, features(phase, queries, lengths), role)
            categories = row["category_ms_per_step"]
            record.update(record_id=f"{model}:profile:{index}", kind="gpu_components",
                observed_nonattention_ms=categories["GEMM"] + categories["Element-wise"],
                observed_attention_ms=categories["Attention"], context_evidence=context_evidence,
                source_timing_boundary="separate_process_profile_kernel_sum_v1",
                transfer_assumption="GPU component calibration proxy; not the hooked wall interval or directly measured CPU overhead",
                source_schedule_file=schedule_path, source_schedule_sha256=source.files[schedule_path],
                raw_record_index=index)
            records.append(record)
            split[role + "_record_ids"].append(record["record_id"])

    source.verify()
    if archive is not None:
        source.verify_archive(Path(archive))
    return {"schema_version": 1, "kind": "analytical_cost_dataset_v1", "records": records,
        "experiments": experiments, "recommended_split": split,
        "source": {"kind": "provided_trace" if archive is not None else "test_fixture", "original_archive_sha256": ARCHIVE_SHA256,
            "archive_verification": "verified_original_bytes_and_consumed_files" if archive is not None else "reference_only_not_recomputed",
            "raw_files": source.files, "sessions": sessions,
            "study_design": "retrospective_group_holdout_not_new_blind_acquisition",
            "prior_information": "Aggregate all18-shape results were read before this improvement task.",
            "split_policy": "reps1+2 endpoint walls and profile endpoints train; rep3 middle shapes holdout; rep3 endpoints repeat; retain all unused",
            "actual_graph_dispatch": "not_observed; configured FULL_AND_PIECEWISE only",
            "runtime_state_scope": "Configured controls bound; allocation outcome1423/1782 differences retained as diagnostics without causal attribution.",
            "measurement_limit": "Original step-hook implementation absent; reported boundary retained, not recertified.",
            "component_limit": "Profile components come from a different instrumented process; context aggregate alignment is approximate for Decode."}}


def import_cost_trace(package_root: str | Path, outdir: str | Path, *, archive: str | Path | None = None) -> dict:
    dataset = build_dataset(package_root, archive=archive)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "dataset.json").write_text(json.dumps(dataset, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (destination / "recommended-split.json").write_text(json.dumps(dataset["recommended_split"], indent=2) + "\n")
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="Verify the original supplied ZIP and consumed bytes")
    args = parser.parse_args()
    dataset = import_cost_trace(args.package_root, args.out, archive=args.archive)
    print(json.dumps({"records": len(dataset["records"]), "out": str(args.out),
                      "split_counts": {name: len(values) for name, values in dataset["recommended_split"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
