"""An explicit, dependency-free implementation of Hello MLSys chapter 1.

This nominal teaching protocol intentionally counts full-square attention.
It neither changes the serving simulator's causal attention nor fits a Profile.
"""

import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
import re
from typing import Any, NoReturn

from .layout import layer_parameters
from .qwen3 import model_from_qwen3_config
from .schema import ModelSpec, SimulationError


PROTOCOL = "hello_mlsys_ch1_nominal_v1"
CALIBRATED_PROTOCOL = "hello_mlsys_ch1_eta_beta_transfer_v1"
_MAX_BYTES = 1024 * 1024
_COURSE_COMMIT = "7df2dae493eb30c0f7e845b6d587b905288c0d7a"
_COURSE_PATH = "docs/part1-modeling/chapter1/index.md"
_COURSE_SHA = "0c0c01a90d35b1b107cc88ccd2c48e776de141bb698356d4b8696baa401ec231"
_MODELS = {
    "qwen3_0_6b": ("Qwen3-0.6B", 28, 1024, 3072, 16, 8, 128, 151936),
    "qwen3_1_7b": ("Qwen3-1.7B", 28, 2048, 6144, 16, 8, 128, 151936),
    "qwen3_4b": ("Qwen3-4B", 36, 2560, 9728, 32, 8, 128, 151936),
}
FORMULAS = {
    "units": "ms, bytes, decimal GB/s and TFLOP/s; BF16=2 bytes; FMA=2 FLOPs",
    "linear_parameters": "P_linear=L*(2*h*d_q+2*h*d_kv+3*h*intermediate); P_head=h*vocab",
    "prefill_gemm": "F=2*P_linear*S; D=2*P_linear+a_prefill*L*h*S; optional last-token head adds 2*P_head to F,D",
    "decode_gemm": "F=2*(P_linear+P_head)*B; D=2*(P_linear+P_head)+a_decode*L*h*B",
    "prefill_attention": "F=4*L*d_q*S*S; D=2*S*(2*d_q+2*d_kv)*L; full square, not causal triangle",
    "decode_attention": "F=4*L*B*ctx*d_q; D=4*L*B*ctx*d_kv; ctx includes current position",
    "elementwise": "F=f_elementwise*L*h*tokens; D=d_elementwise*L*h*tokens; vector pipeline",
    "component_time": "t_ms=max(F/(pipeline_TFLOPS*1e9),D/(bandwidth_GBps*1e6)); eta=1",
    "step_time": "t_step_ms=sum(component.t_ms)+t_fixed_ms; global AI is display only",
    "naive_attention": "D_naive=D_flash+4*L*q_heads*S*S; same declared F; asymptotic AI=head_dim",
    "matmul": "M=N=K=4096; F=2*M*N*K; D=2*(M*K+K*N+M*N)",
    "layernorm": "S=2048,h=1024; F=4*S*h; D=4*S*h; affine/reduction extras omitted teaching count",
}
_WARNINGS = [
    "解析 nominal 下界；未执行 GPU、未拟合效率、hardware_validated=false。",
    "本次 stored_at 是当前运行时间，不证明早于课程已有实测；未来测量须另行关联 prediction_id。",
    "本协议使用 full-square attention；默认服务模拟器和既有 Profile 仍使用原口径。",
    "GEMM 激活及 elementwise 是 plan 中公开的流量假设，并非从课程近似时间反推的效率。",
    "W7900D 算力/带宽待确认；AI 示意不是该硬件的 Roofline 或时间预测。",
    "本章时间不包含排队、launch、通信或容量/OOM 判定；带宽受限不等于显存装不下。",
]


def _fail(message: str) -> NoReturn:
    raise SimulationError("INVALID_COURSE_PLAN", message)


def _canonical(value: Any) -> bytes:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                             ensure_ascii=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise SimulationError("INVALID_COURSE_PLAN", "Course data must be finite JSON") from exc
    if len(encoded) > _MAX_BYTES:
        _fail("Course data exceeds the 1MiB limit")
    return encoded


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _object(value: Any, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(f"{label} must explicitly contain exactly: {', '.join(sorted(keys))}")
    return value


def _text(value: Any, label: str, limit: int = 512) -> str:
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or len(value) > limit or any(ord(c) < 32 for c in value)):
        _fail(f"{label} must be bounded text without control characters")
    return value


def _number(value: Any, label: str, *, positive: bool = False, maximum: float = 1e6) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        _fail(f"{label} must be a finite number")
    if not math.isfinite(result) or not (1e-9 <= result <= maximum if positive else 0 <= result <= maximum):
        _fail(f"{label} must be in [{'1e-9' if positive else '0'}, {maximum}]")
    return result


def _validate_plan(plan: dict) -> list[ModelSpec]:
    _canonical(plan)
    _object(plan, {"schema_version", "protocol", "experiment_id", "calibration", "course_source",
                   "hardware", "pending_hardware", "axes", "counting", "models"}, "plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1 or plan["protocol"] != PROTOCOL:
        _fail("Only explicit schema_version=1 and hello_mlsys_ch1_nominal_v1 are supported")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", _text(plan["experiment_id"], "experiment_id")):
        _fail("experiment_id must be a safe lowercase identifier")
    calibration = _object(plan["calibration"], {"version", "eta", "t_fixed_ms"}, "calibration")
    if (calibration["version"] != "nominal" or type(calibration["eta"]) not in (int, float)
            or calibration["eta"] != 1 or type(calibration["t_fixed_ms"]) not in (int, float)
            or calibration["t_fixed_ms"] != 0):
        _fail("This chapter freezes nominal eta=1 and uncalibrated t_fixed_ms=0")
    source = _object(plan["course_source"], {"repository", "commit_sha", "path", "sha256"}, "course_source")
    if source != {"repository": "amdjiahangpan/hello-mlsys", "commit_sha": _COURSE_COMMIT,
                  "path": _COURSE_PATH, "sha256": _COURSE_SHA}:
        _fail("Course source must match the pinned chapter revision for this formula protocol")
    hardware = _object(plan["hardware"], {"id", "name", "status", "matrix_tflops", "vector_tflops",
                                         "bandwidth_gbps", "source"}, "hardware")
    if hardware["id"] != "rx9070xt" or hardware["status"] != "course_nominal":
        _fail("The primary chapter grid must explicitly identify RX9070XT course_nominal hardware")
    for key in ("name", "source"):
        _text(hardware[key], f"hardware.{key}")
    for key in ("matrix_tflops", "vector_tflops", "bandwidth_gbps"):
        _number(hardware[key], f"hardware.{key}", positive=True)
    pending = _object(plan["pending_hardware"], {"id", "name", "status", "matrix_tflops", "vector_tflops",
                                                "bandwidth_gbps", "source"}, "pending_hardware")
    if (pending["id"] != "w7900d" or pending["status"] != "pending_parameters"
            or any(pending[k] is not None for k in ("matrix_tflops", "vector_tflops", "bandwidth_gbps"))):
        _fail("W7900D parameters must remain explicitly null/pending in this course protocol")
    for key in ("name", "source"):
        _text(pending[key], f"pending_hardware.{key}")
    axes = _object(plan["axes"], {"prefill_lengths", "decode_batch_sizes", "decode_context"}, "axes")
    if (axes != {"prefill_lengths": [128, 512, 2048], "decode_batch_sizes": [1, 4, 16], "decode_context": 1024}
            or type(axes["decode_context"]) is not int
            or any(type(x) is not int for k in ("prefill_lengths", "decode_batch_sizes") for x in axes[k])):
        _fail("Chapter 1 freezes prefill 128/512/2048, decode B1/4/16 and ctx1024")
    counting = _object(plan["counting"], {"prefill_attention", "prefill_lm_head", "bf16_bytes",
        "decode_context_includes_current", "prefill_gemm_activation_bytes_per_layer_hidden",
        "decode_gemm_activation_bytes_per_layer_hidden", "elementwise_flops_per_layer_hidden",
        "elementwise_bytes_per_layer_hidden"}, "counting")
    if (counting["prefill_attention"] != "full_square" or type(counting["bf16_bytes"]) is not int
            or counting["bf16_bytes"] != 2 or counting["decode_context_includes_current"] is not True
            or counting["prefill_lm_head"] not in ("omitted", "last_token")):
        _fail("Explicit BF16/full-square/inclusive-context and omitted or last_token head are required")
    for key in ("prefill_gemm_activation_bytes_per_layer_hidden", "decode_gemm_activation_bytes_per_layer_hidden",
                "elementwise_flops_per_layer_hidden", "elementwise_bytes_per_layer_hidden"):
        _number(counting[key], f"counting.{key}", positive=key.startswith("elementwise"), maximum=4096)
    rows = plan["models"]
    if not isinstance(rows, list) or len(rows) != 3:
        _fail("Exactly three declared Qwen3 models are required")
    models = []
    for row, (model_id, geometry) in zip(rows, _MODELS.items()):
        _object(row, {"id", "name", "revision", "source", "hf_config"}, "model")
        if row["id"] != model_id or row["name"] != geometry[0]:
            _fail("Models must be the ordered Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B grid")
        _text(row["source"], "model.source")
        _text(row["revision"], "model.revision")
        model = model_from_qwen3_config(row["hf_config"], name=row["name"], revision=row["revision"])
        actual = (model.name, model.num_layers, model.hidden_size, model.intermediate_size,
                  model.num_attention_heads, model.num_kv_heads, model.head_dim, model.vocab_size)
        if actual != geometry or model.weight_dtype != "bfloat16" or not model.tie_word_embeddings:
            _fail("Model names must match their declared course BF16 dense Qwen3 geometry and tied embeddings")
        models.append(model)
    return models


def _timestamp(stored_at: str | None) -> str:
    now = datetime.now(timezone.utc)
    if stored_at is None:
        return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    _text(stored_at, "stored_at", 64)
    try:
        parsed = datetime.fromisoformat(stored_at.replace("Z", "+00:00"))
        delta = (parsed - now).total_seconds()
    except (ValueError, TypeError):
        _fail("stored_at must be a current UTC timestamp")
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0 or not -60 <= delta <= 5:
        _fail("stored_at must be current UTC (at most 60s old / 5s clock skew); historical backdating is forbidden")
    return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _component(name: str, flops: int | float, byte_count: int | float, hardware: dict,
               pipeline: str = "matrix") -> dict:
    peak = hardware[f"{pipeline}_tflops"]
    bandwidth = hardware["bandwidth_gbps"]
    tc, tm = flops / (peak * 1e9), byte_count / (bandwidth * 1e6)
    return {"component": name, "pipeline": pipeline, "flops": flops, "bytes": byte_count,
            "ai": flops / byte_count, "ridge": peak * 1000 / bandwidth,
            "bound": "compute" if tc >= tm else "memory", "t_compute_ms": tc,
            "t_memory_ms": tm, "t_ms": max(tc, tm)}


def _attention(model: ModelSpec, phase: str, size: int, context: int, hardware: dict) -> dict:
    layers, dq, dkv = model.num_layers, model.query_projection_size, model.num_kv_heads * model.head_dim
    if phase == "prefill":
        flops, traffic = 4 * layers * dq * size**2, 4 * size * (dq + dkv) * layers
    else:
        flops, traffic = 4 * layers * size * context * dq, 4 * layers * size * context * dkv
    return _component("attention", flops, traffic, hardware)


def _point(plan: dict, model: ModelSpec, index: int, phase: str, size: int, stored_at: str) -> dict:
    row, hardware, count = plan["models"][index], plan["hardware"], plan["counting"]
    linear = layer_parameters(model, 1)["linear"] * model.num_layers
    head = model.hidden_size * model.vocab_size
    lh = model.num_layers * model.hidden_size
    if phase == "prefill":
        head_used = head if count["prefill_lm_head"] == "last_token" else 0
        f_gemm = 2 * (linear * size + head_used)
        d_gemm = 2 * (linear + head_used) + size * lh * count["prefill_gemm_activation_bytes_per_layer_hidden"]
        batch, sequence = 1, size
    else:
        f_gemm = 2 * (linear + head) * size
        d_gemm = 2 * (linear + head) + size * lh * count["decode_gemm_activation_bytes_per_layer_hidden"]
        batch, sequence = size, plan["axes"]["decode_context"]
    components = [_component("gemm", f_gemm, d_gemm, hardware),
                  _attention(model, phase, size, sequence, hardware),
                  _component("elementwise", size * lh * count["elementwise_flops_per_layer_hidden"],
                             size * lh * count["elementwise_bytes_per_layer_hidden"], hardware, "vector")]
    step = math.fsum(c["t_ms"] for c in components) + plan["calibration"]["t_fixed_ms"]
    flops, traffic = sum(c["flops"] for c in components), sum(c["bytes"] for c in components)
    config_id = f"{row['id']}_{phase}_{'s' if phase == 'prefill' else 'b'}{size}"
    return {"config_id": config_id, "prediction_id": f"pred_ch1_{plan['experiment_id']}_{config_id}_{_sha(plan)[:12]}",
            "stored_at": stored_at, "calibration_version": "nominal", "formula_version": PROTOCOL,
            "hardware_validated": False, "model_id": row["id"], "model_name": row["name"],
            "phase": phase, "batch_size": batch, "sequence_length": sequence,
            "context_includes_current": phase == "decode", "linear_parameters": linear,
            "head_parameters": head, "components": components, "t_step_ms": step,
            "aggregate_tokens_per_s": (size if phase == "decode" else sequence) * 1000 / step,
            "per_request_tokens_per_s": (1 if phase == "decode" else sequence) * 1000 / step,
            "global_flops": flops, "global_bytes": traffic, "global_ai_display_only": flops / traffic,
            "dominant_component": max(components, key=lambda c: c["t_ms"])["component"]}


def _primitives(model: ModelSpec, hardware: dict) -> list[dict]:
    n, s, h = 4096, 2048, 1024
    flash = _attention(model, "prefill", 512, 512, hardware)
    naive = _component("attention_naive", flash["flops"],
                       flash["bytes"] + 4 * model.num_layers * model.num_attention_heads * 512**2, hardware)
    return [_component("matmul_4096_bf16", 2 * n**3, 6 * n**2, hardware),
            _component("layernorm_2048_1024_bf16", 4 * s * h, 4 * s * h, hardware, "vector"),
            {**flash, "component": "attention_flash_s512_28_layers"}, naive]


def _reconciliation(points: list[dict], primitives: list[dict], count: dict) -> dict:
    rows = []
    anchors = {("decode", 1): 2.1, ("decode", 4): 2.6, ("decode", 16): 4.9,
               ("prefill", 128): 1.5, ("prefill", 512): 2.7, ("prefill", 2048): 14.0}
    for point in points[:6]:
        phase = point["phase"]
        size = point["batch_size"] if phase == "decode" else point["sequence_length"]
        approximate = anchors[phase, size]
        rows.append({"config_id": point["config_id"], "course_approximate_ms": approximate,
                     "nominal_ms": point["t_step_ms"], "difference_ms": point["t_step_ms"] - approximate,
                     "matches_one_decimal_rounding": round(point["t_step_ms"], 1) == approximate})
    return {"kind": "formula_and_rounding_reconciliation_not_hardware_error", "time_rows": rows,
            "naive_attention": {"course_claimed_asymptotic_ai": 0.5, "formula_derived_asymptotic_ai": 128,
                "s512_nominal_ai": primitives[3]["ai"], "status": "course_formula_inconsistency",
                "explanation": "原文 F=4*L*q_heads*head_dim*S²、额外 D=4*L*q_heads*S²，故极限为 head_dim=128；不能同时得到0.5。"},
            "notes": [
                "时间对账使用课程算例 B/C 的约数，未将其当作标定目标或测量数据。",
                "0.6B 精确 P_linear=440401920、P_linear+P_head=595984384；课程写约440M/596M。",
                "prefill 激活流量系数与 elementwise 是明确假设；各组件求和可能无法复现近似表的单步舍入。",
                f"prefill LM head={count['prefill_lm_head']}；课程主项省略末 token head，任务表泛称包含 head。",
                "不对 FlashAttention 统一乘1.13：原文13%是外部算子测例比值，非本模型冻结公式。",
            ]}


def run_chapter1(plan: dict, *, stored_at: str | None = None) -> dict:
    """Return all 18 nominal predictions; a supplied timestamp must be current UTC.

    Inputs are inline JSON only. No file paths, remote code or GPU APIs are read
    from the plan. The timestamp cannot be used to create historical evidence.
    """
    models = _validate_plan(plan)
    plan = json.loads(_canonical(plan))
    timestamp = _timestamp(stored_at)
    points: list[dict] = []
    for index, model in enumerate(models):
        for phase, sizes in (("prefill", plan["axes"]["prefill_lengths"]),
                             ("decode", plan["axes"]["decode_batch_sizes"])):
            points.extend(_point(plan, model, index, phase, size, timestamp) for size in sizes)
    primitives = _primitives(models[0], plan["hardware"])
    report = {"schema_version": 1, "protocol": PROTOCOL, "formula_version": PROTOCOL,
              "report_id": f"ch1_{_sha(plan)[:16]}", "stored_at": timestamp, "hardware_validated": False,
              "status": "COMPLETED", "prediction_status": "nominal_predictions_complete_with_course_reconciliation",
              "plan": plan,
              "point_count": len(points), "points": points, "primitives": primitives,
              "formulas": FORMULAS.copy(), "warnings": _WARNINGS.copy(),
              "provenance": {"plan_sha256": _sha(plan), "formulas_sha256": _sha(FORMULAS),
                  "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "course_source": plan["course_source"],
                  "model_config_sha256": {row["id"]: _sha(row["hf_config"]) for row in plan["models"]},
                  "measurement_sources": [], "historical_prediction_before_measurement_established": False},
              "reconciliation": _reconciliation(points, primitives, plan["counting"])}
    report["data_sha256"] = _sha(report)
    return report


def run_calibrated_chapter1(plan: dict, calibration: dict, *, stored_at: str | None = None) -> dict:
    """Explicitly transfer imported primitive coefficients to all 18 dense points.

    This is a separate protocol and a domain-transfer experiment, not a claim
    that primitive holdouts establish calibrated whole-model hardware accuracy.
    The original nominal entry point and its returned fields stay unchanged.
    """
    from .course_calibration import validate_course_calibration_report

    validate_course_calibration_report(calibration)
    calibration = json.loads(_canonical(calibration))
    report = run_chapter1(plan, stored_at=stored_at)
    hardware, constants = plan["hardware"], calibration["nominal_constants"]
    if (hardware["matrix_tflops"] != constants["peak_tflops"]
            or hardware["bandwidth_gbps"] != constants["bw_gbps"]):
        _fail("Primitive calibration and chapter plan must declare the same matrix peak and bandwidth")
    eta, beta = calibration["coefficients"]["eta"], calibration["coefficients"]["beta"]
    nominal_hash = report["data_sha256"]
    identity = _sha({"plan": plan, "raw_calibration_sha256": calibration["raw_sha256"], "eta": eta, "beta": beta})[:16]

    def transform(component: dict) -> None:
        component["nominal"] = {k: component[k] for k in ("t_compute_ms", "t_memory_ms", "t_ms", "ridge", "bound")}
        component["t_compute_ms"] /= eta
        component["t_memory_ms"] /= beta
        component["t_ms"] = max(component["t_compute_ms"], component["t_memory_ms"])
        component["bound"] = "compute" if component["t_compute_ms"] >= component["t_memory_ms"] else "memory"
        component["ridge"] *= eta / beta
        component["eta_applied"], component["beta_applied"] = eta, beta

    for point in report["points"]:
        point["nominal_t_step_ms"] = point["t_step_ms"]
        for component in point["components"]:
            transform(component)
        point["t_step_ms"] = math.fsum(c["t_ms"] for c in point["components"]) + plan["calibration"]["t_fixed_ms"]
        tokens = point["batch_size"] if point["phase"] == "decode" else point["sequence_length"]
        point["aggregate_tokens_per_s"] = tokens * 1000 / point["t_step_ms"]
        point["per_request_tokens_per_s"] = (1 if point["phase"] == "decode" else tokens) * 1000 / point["t_step_ms"]
        point["dominant_component"] = max(point["components"], key=lambda c: c["t_ms"])["component"]
        point["calibration_version"] = calibration["calibration_id"]
        point["formula_version"] = CALIBRATED_PROTOCOL
        point["prediction_id"] = f"pred_ch2_{plan['experiment_id']}_{point['config_id']}_{identity}"
        point["validation_scope"] = "primitive_to_dense_model_domain_transfer_not_hardware_validation"
    for primitive in report["primitives"]:
        transform(primitive)
    report["protocol"] = report["formula_version"] = CALIBRATED_PROTOCOL
    report["base_formula_version"] = PROTOCOL
    report["report_id"] = f"ch1_eta_beta_{identity}"
    report["prediction_status"] = "primitive_to_dense_domain_transfer_complete_not_whole_model_validation"
    report["applied_calibration"] = {"calibration_id": calibration["calibration_id"], "eta": eta, "beta": beta,
        "source_kind": calibration["source_kind"], "source": calibration["source"],
        "raw_sha256": calibration["raw_sha256"], "import_report_sha256": calibration["data_sha256"],
        "fitting_domain": calibration["fitting_domain"], "hardware_validated": False,
        "launch_framework_overheads_identified": False, "whole_model_validation_established": False}
    report["formulas"]["component_time"] = "t_ms=max(t_compute_nominal/eta,t_memory_nominal/beta); recompute bound and effective ridge"
    report["formulas"]["effective_ridge"] = "pipeline_peak_TFLOPS*1000*eta/(bandwidth_GBps*beta)"
    report["warnings"][0] = "η/β 从 primitive 迁移到 dense 全模型，属于域外假设；无新 GPU 实测或全模型硬件PASS。"
    report["warnings"] += calibration["limitations"] + [
        "同一 GEMM η 同时用于矩阵/向量计算是显式迁移假设；向量计算峰值效率并未单独识别。",
        "课程 B/C 近似表的 reconciliation 保留 nominal 基线；不能用校准后结果反标近似表的计数正确性。",
    ]
    report["reconciliation"]["time_rows_scope"] = "nominal_baseline_before_eta_beta_transfer"
    report["provenance"]["nominal_data_sha256"] = nominal_hash
    report["provenance"]["course_calibration_raw_sha256"] = calibration["raw_sha256"]
    report["provenance"]["course_calibration_report_sha256"] = calibration["data_sha256"]
    report["provenance"]["formulas_sha256"] = _sha(report["formulas"])
    report["data_sha256"] = _sha({k: v for k, v in report.items() if k != "data_sha256"})
    return report


def _transferred(report: dict) -> bool:
    return report["protocol"] == CALIBRATED_PROTOCOL


def _plot_hardware(report: dict) -> dict:
    hardware = report["plan"]["hardware"].copy()
    if _transferred(report):
        calibration = report["applied_calibration"]
        hardware["matrix_tflops"] *= calibration["eta"]
        hardware["vector_tflops"] *= calibration["eta"]
        hardware["bandwidth_gbps"] *= calibration["beta"]
    return hardware


def _svg_open(report: dict, title: str, width: int, height: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title, quote=True)}">',
            f'<title>{escape(title)}</title>', f'<metadata>report.json sha256={report["data_sha256"]}; '
            f'protocol={report["protocol"]}; hardware_validated=false</metadata>',
            '<style>text{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;'
            'fill:#172b3a;font-size:13px} .small{font-size:11px} .title{font-size:21px;font-weight:600}'
            '.grid{stroke:#d9e2e8;stroke-width:1} .axis{stroke:#607785;stroke-width:1}</style>',
            f'<rect width="{width}" height="{height}" fill="#fff"/>',
            f'<text x="32" y="34" class="title">{escape(title)}</text>']


def _label(x: float, y: float, value: str, extra: str = "") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" {extra}>{escape(value)}</text>'


def _marker(x: float, y: float, color: str, title: str, kind: int = 0) -> str:
    if kind == 1:
        shape = f'<path d="M{x:.2f},{y-4:.2f}l4,4l-4,4l-4,-4z"'
    elif kind == 2:
        shape = f'<rect x="{x-3:.2f}" y="{y-3:.2f}" width="6" height="6"'
    else:
        shape = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5"'
    return f'<g><title>{escape(title)}</title>{shape} fill="white" stroke="{color}" stroke-width="1.8"/></g>'


def _source_footer(report: dict, y: int) -> str:
    mode = "primitive η/β 迁移，非全模型验证" if _transferred(report) else "nominal η=1"
    return _label(32, y, f"数据：report.json · {report['data_sha256'][:16]} · {mode} · 无实测点", 'class="small"')


def _ai_limits(report: dict) -> tuple[float, float]:
    ais = [c["ai"] for p in report["points"] for c in p["components"]]
    return min(math.log10(0.5), math.floor(math.log10(min(ais)))), max(math.log10(5000), math.ceil(math.log10(max(ais))))


def _roofline_svg(report: dict) -> str:
    title = "RX 9070 XT：矩阵 / 向量双屋顶与 18 点组件"
    if _transferred(report):
        title += "（primitive η/β 迁移）"
    svg = _svg_open(report, title, 1250, 650)
    x0, x1, y0, y1 = 82, 750, 95, 545
    loglo, loghi = _ai_limits(report)
    hardware = _plot_hardware(report)
    min_tf = min(c["flops"] / c["t_ms"] / 1e9 for p in report["points"] for c in p["components"])
    log_ylo = min(-1, math.floor(math.log10(min_tf)))
    log_yhi = max(math.log10(400), math.log10(max(hardware["matrix_tflops"], hardware["vector_tflops"]) * 1.5))
    def xpos(ai: float) -> float:
        return x0 + (math.log10(ai) - loglo) / (loghi - loglo) * (x1 - x0)

    def ypos(tf: float) -> float:
        return y1 - (math.log10(tf) - log_ylo) / (log_yhi - log_ylo) * (y1 - y0)
    exponents = range(math.ceil(loglo), math.ceil(loghi), max(1, math.ceil((loghi - loglo) / 5)))
    for ai in (10.0**exponent for exponent in exponents):
        x = xpos(ai)
        svg += [f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y0}" y2="{y1}" class="grid"/>',
                _label(x - 9, y1 + 22, str(ai))]
    exponents = range(math.ceil(log_ylo), math.ceil(log_yhi), max(1, math.ceil((log_yhi - log_ylo) / 5)))
    for tf in (10.0**exponent for exponent in exponents):
        y = ypos(tf)
        svg += [f'<line x1="{x0}" x2="{x1}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>',
                _label(37, y + 4, str(tf))]
    for pipeline, color, text_y in (("matrix", "#176b9b", 65), ("vector", "#a45518", 83)):
        peak = hardware[f"{pipeline}_tflops"]
        ridge = peak * 1000 / hardware["bandwidth_gbps"]
        samples = [10.0**loglo, 10.0**loghi]
        if samples[0] < ridge < samples[-1]:
            samples.insert(1, ridge)
        coords = [(xpos(ai), ypos(max(10.0**log_ylo, min(peak, ai * hardware["bandwidth_gbps"] / 1000))))
                  for ai in samples]
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
        dash = ' stroke-dasharray="7,4"' if pipeline == "vector" else ""
        svg += [f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>',
                _label(82, text_y, f"{'矩阵' if pipeline == 'matrix' else '向量'}：{peak:g} TFLOP/s；拐点 AI={ridge:.4f}")]
    colors = ("#176b9b", "#b0521c", "#27734d")
    for number, point in enumerate(report["points"], 1):
        for kind, component in enumerate(point["components"]):
            ai = component["ai"]
            peak = hardware["vector_tflops" if kind == 2 else "matrix_tflops"]
            tf = min(peak, ai * hardware["bandwidth_gbps"] / 1000)
            name = ("G", "A", "E")[kind] + str(number)
            svg.append(_marker(xpos(ai), ypos(tf), colors[kind],
                               f"{name}: AI={ai:.9g}; {point['config_id']}", kind))
    svg += [_label(250, 590, "算术强度 AI（FLOP/byte，对数）"),
            _label(33, 62, "TFLOP/s"), _label(805, 70, "编号 / 配置                 AI：G / A / E")]
    for number, point in enumerate(report["points"], 1):
        axis = f"S{point['sequence_length']}" if point["phase"] == "prefill" else f"B{point['batch_size']} ctx1024"
        ais = " / ".join(f"{c['ai']:.1f}" for c in point["components"])
        svg.append(_label(805, 94 + (number - 1) * 25,
                          f"{number:02d}  {point['model_name']} {axis}  {ais}", 'class="small"'))
    svg += [_label(805, 568, "G ○ GEMM；A ◇ Attention；E □ Elementwise"),
            _label(805, 589, "54 个组件点；精确重合处覆盖，编号在右表", 'class="small"'),
            _label(805, 608, "完整 ID / 精确 AI 见 CSV 与各点悬浮标题", 'class="small"'),
            _source_footer(report, 627), "</svg>"]
    return "\n".join(svg)


def _pending_svg(report: dict) -> str:
    svg = _svg_open(report, "W7900D：待确认参数（仅展示 18 点 AI）", 1150, 715)
    svg += [_label(32, 65, "峰值算力与带宽未知：无硬件 Roofline、无时间/吞吐预测，纵轴仅为配置索引。"),
            _label(32, 87, "AI 来自算子计数，适用同一冻结 BF16 口径；此图不能替代 W7900D 验证。")]
    loglo, loghi = _ai_limits(report)
    def xpos(ai: float) -> float:
        return 390 + (math.log10(ai) - loglo) / (loghi - loglo) * 680
    exponents = range(math.ceil(loglo), math.ceil(loghi), max(1, math.ceil((loghi - loglo) / 5)))
    for ai in (10.0**exponent for exponent in exponents):
        x = xpos(ai)
        svg += [f'<line x1="{x:.2f}" x2="{x:.2f}" y1="110" y2="625" class="grid"/>',
                _label(x - 8, 649, str(ai))]
    for number, point in enumerate(report["points"], 1):
        y = 127 + (number - 1) * 28
        axis = f"S{point['sequence_length']}" if point["phase"] == "prefill" else f"B{point['batch_size']} ctx1024"
        svg.append(_label(32, y + 4, f"{number:02d}  {point['model_name']} {point['phase']} {axis}"))
        for kind, component in enumerate(point["components"]):
            svg.append(_marker(xpos(component["ai"]), y, ("#176b9b", "#b0521c", "#27734d")[kind],
                               f"{point['config_id']} {component['component']}: AI={component['ai']:.9g}", kind))
    svg += [_label(412, 677, "算术强度 AI（FLOP/byte，对数）；○ GEMM  ◇ Attention  □ Elementwise"),
            _source_footer(report, 701), "</svg>"]
    return "\n".join(svg)


def _curve_svg(report: dict, phase: str) -> str:
    decode = phase == "decode"
    title = "Decode batching：step 与聚合吞吐" if decode else "Prefill：序列长度与 step 下界"
    if _transferred(report):
        title = title.replace("下界", "预测") + "（η/β 迁移）"
    width, height = (1150 if decode else 900), 550
    svg = _svg_open(report, title, width, height)
    panels = [("t_step_ms", "单步时间（ms）", 82, 520 if decode else 840)]
    if decode:
        panels.append(("aggregate_tokens_per_s", "聚合吞吐上限（token/s）", 660, 1090))
    points = [p for p in report["points"] if p["phase"] == phase]
    sizes = report["plan"]["axes"]["decode_batch_sizes" if decode else "prefill_lengths"]
    for field, ylabel, x0, x1 in panels:
        ymax = max(p[field] for p in points) * 1.15
        def xpos(x: float) -> float:
            return x0 + (x - sizes[0]) / (sizes[-1] - sizes[0]) * (x1 - x0)

        def ypos(y: float) -> float:
            return 435 - y / ymax * 300
        svg.append(_label(x0, 110, ylabel))
        for step in range(5):
            value = ymax * step / 4
            y = ypos(value)
            svg += [f'<line x1="{x0}" x2="{x1}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>',
                    _label(x0 - 53, y + 4, f"{value:.2g}")]
        for size in sizes:
            x = xpos(size)
            svg += [f'<line x1="{x:.2f}" x2="{x:.2f}" y1="135" y2="435" class="grid"/>',
                    _label(x - 10, 458, str(size))]
        for index, (model_id, geometry) in enumerate(_MODELS.items()):
            subset = [p for p in points if p["model_id"] == model_id]
            color = ("#176b9b", "#b0521c", "#27734d")[index]
            coords = [(xpos(p["batch_size"] if decode else p["sequence_length"]), ypos(p[field])) for p in subset]
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
            svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
            for point, (x, y) in zip(subset, coords):
                svg.append(_marker(x, y, color, f"{point['config_id']}: {field}={point[field]:.9g}", index))
        svg.append(_label(x0 + 70, 489, "Batch size B（ctx=1024，含当前 token）" if decode else "输入长度 S（B=1）"))
    for index, geometry in enumerate(_MODELS.values()):
        svg.append(_label(32 + index * 250, 75, f"{'○◇□'[index]} {geometry[0]}"))
    svg += [_source_footer(report, 529), "</svg>"]
    return "\n".join(svg)


def _csv(report: dict) -> str:
    buffer = io.StringIO(newline="")
    fields = ["prediction_id", "stored_at", "model_name", "phase", "batch_size", "sequence_length",
              "component", "pipeline", "flops", "bytes", "ai", "ridge", "bound",
              "t_compute_ms", "t_memory_ms", "t_ms", "t_step_ms", "formula_version", "hardware_validated"]
    if _transferred(report):
        fields += ["nominal_t_compute_ms", "nominal_t_memory_ms", "nominal_t_ms", "nominal_bound", "eta_applied", "beta_applied"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for point in report["points"]:
        for component in point["components"]:
            writer.writerow({key: component["nominal"][key.removeprefix("nominal_")] if key.startswith("nominal_")
                             else component[key] if key in component else point[key] for key in fields})
    return buffer.getvalue()


def _markdown(report: dict) -> str:
    mode = "primitive η/β 迁移" if _transferred(report) else "nominal"
    rows = [f"# 第 1 章：18 点 {mode} 实验报告", "", f"协议 `{report['protocol']}`；创建于 `{report['stored_at']}`。",
            "", "**解析预测完成；不代表课程的历史实测、先预测后测量纪律或 W7900D 硬件验收已完成。**", ""]
    rows += [f"- {warning}" for warning in report["warnings"]]
    rows += ["", "## 计数与瓶颈", "", "![RX9070XT 双屋顶与全部18点组件](roofline-rx9070xt.svg)", "",
             f"图中保留全部54个 {mode} 组件点；精确重合的 marker 会覆盖，编号与配置在右表，完整数值见 [components.csv](components.csv)。",
             "每个组件先选矩阵/向量峰值，取 max(compute, memory)，step 再求和；全局 AI 不用于判定。", "",
             "| 模型 | Phase | B | S / ctx | GEMM | Attention | Elementwise | Step (ms) | 聚合 token/s |",
             "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |"]
    for p in report["points"]:
        bounds = " | ".join(c["bound"] for c in p["components"])
        rows.append(f"| {p['model_name']} | {p['phase']} | {p['batch_size']} | {p['sequence_length']} | "
                    f"{bounds} | {p['t_step_ms']:.6f} | {p['aggregate_tokens_per_s']:.2f} |")
    rows += ["", "Prefill token/s 是输入处理率；decode token/s 是生成速率，二者不可混作同一服务吞吐。", "",
             "![Batching 曲线](batching.svg)", "", "固定 ctx 后，batching 复用 GEMM 权重，但 KV 扫描仍随 B 增长。", "",
             "![Prefill 曲线](prefill.svg)", "", "固定 B=1；长序列的 full-square attention 计算项增长。", "",
             "## W7900D：待确认参数", "", "![W7900D 待确认，仅 AI 索引](roofline-w7900d-pending.svg)", "",
             "参数未知，无法画有数值意义的该硬件屋顶线；此图只展示同一 18 点的 AI，纵轴不是吞吐。", "",
             *( ["η/β 迁移报告中，下表保留校准前 nominal 基线；上面的18点与曲线使用校准后时间。", ""] if _transferred(report) else []),
             "## 手算与近似表对账", "", "| 0.6B 配置 | 课程约数 ms | 本协议精确 ms | 差值 ms | 一位小数相同 |",
             "| --- | ---: | ---: | ---: | --- |"]
    for r in report["reconciliation"]["time_rows"]:
        rows.append(f"| {r['config_id']} | {r['course_approximate_ms']:.1f} | {r['nominal_ms']:.6f} | "
                    f"{r['difference_ms']:+.6f} | {'是' if r['matches_one_decimal_rounding'] else '否'} |")
    rows += ["", "这些差异属于公式、流量假设与近似表口径，**不标作硬件测量误差或校准通过**。", ""]
    rows += [f"- {note}" for note in report["reconciliation"]["notes"]]
    rows += [f"- {report['reconciliation']['naive_attention']['explanation']}", "", "## Primitive 直觉练习", "",
             "| 算子 | FLOPs | Bytes | AI | 判定 | t_compute ms | t_memory ms | t ms |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for c in report["primitives"]:
        rows.append(f"| {c['component']} | {c['flops']:.0f} | {c['bytes']:.0f} | {c['ai']:.6f} | {c['bound']} | "
                    f"{c['t_compute_ms']:.6f} | {c['t_memory_ms']:.6f} | {c['t_ms']:.6f} |")
    rows += ["", "## 可复现底稿", "", f"- report data SHA256：`{report['data_sha256']}`",
             f"- plan SHA256：`{report['provenance']['plan_sha256']}`",
             f"- formulas SHA256：`{report['provenance']['formulas_sha256']}`",
             f"- 课程原文：[{_COURSE_COMMIT[:12]} / 第 1 章](https://github.com/amdjiahangpan/hello-mlsys/blob/"
             f"{_COURSE_COMMIT}/{_COURSE_PATH})；SHA256 `{_COURSE_SHA}`。",
             "- [report.json](report.json) 保存全部公式、plan、逐点 prediction_id / stored_at、模型输入 hash。",
             "- 以上图、表和 CSV 都从该 report 对象生成；未读取或重标课程已有测量为本次实验。", ""]
    return "\n".join(rows)


def _html(report: dict) -> str:
    def table(headers: list[str], rows: list[list[str]]) -> str:
        return ('<div class="table"><table><thead><tr>' + ''.join(f'<th>{escape(s)}</th>' for s in headers)
                + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(s)}</td>' for s in row)
                + '</tr>' for row in rows) + '</tbody></table></div>')

    def picture(name: str, title: str) -> str:
        return f'<figure><img src="{name}" alt="{title}"><figcaption>{title}</figcaption></figure>'

    summary = table(["模型", "Phase", "B", "S / ctx", "GEMM", "Attention", "Elementwise", "Step ms", "聚合 token/s"],
        [[p["model_name"], p["phase"], str(p["batch_size"]), str(p["sequence_length"]),
          *(c["bound"] for c in p["components"]), f'{p["t_step_ms"]:.6f}', f'{p["aggregate_tokens_per_s"]:.2f}']
         for p in report["points"]])
    transfer_intro = ''
    if _transferred(report):
        calibration = report["applied_calibration"]
        transfer_intro = (f'<p><strong>η={calibration["eta"]:.4f}，β={calibration["beta"]:.4f}</strong>；'
            f'来源：{escape(calibration["source"]["label"])}（<code>{escape(calibration["source_kind"])}</code>）。'
            'primitive 系数迁移到 dense 模型，非新真机测量；未证明全模型精度。</p>')
        summary = table(["模型", "Phase", "B", "S / ctx", "GEMM", "Attention", "Elementwise",
                         "Nominal ms", "迁移后 ms", "变化 %", "聚合 token/s"],
            [[p["model_name"], p["phase"], str(p["batch_size"]), str(p["sequence_length"]),
              *(c["bound"] for c in p["components"]), f'{p["nominal_t_step_ms"]:.6f}', f'{p["t_step_ms"]:.6f}',
              f'{(p["t_step_ms"] / p["nominal_t_step_ms"] - 1) * 100:+.1f}', f'{p["aggregate_tokens_per_s"]:.2f}']
             for p in report["points"]])
    reconciled = table(["0.6B 配置", "课程约数 ms", "本协议精确 ms", "差值 ms", "一位小数相同"],
        [[r["config_id"], f'{r["course_approximate_ms"]:.1f}', f'{r["nominal_ms"]:.6f}',
          f'{r["difference_ms"]:+.6f}', "是" if r["matches_one_decimal_rounding"] else "否"]
         for r in report["reconciliation"]["time_rows"]])
    primitives = table(["算子", "FLOPs", "Bytes", "AI", "判定", "t_compute ms", "t_memory ms", "t ms"],
        [[c["component"], f'{c["flops"]:.0f}', f'{c["bytes"]:.0f}', f'{c["ai"]:.6f}', c["bound"],
          f'{c["t_compute_ms"]:.6f}', f'{c["t_memory_ms"]:.6f}', f'{c["t_ms"]:.6f}'] for c in report["primitives"]])
    warning_list = '<ul>' + ''.join(f'<li>{escape(w)}</li>' for w in report["warnings"]) + '</ul>'
    if _transferred(report):
        warning_list = '<details><summary>完整适用域与证据限制（展开）</summary>' + warning_list + '</details>'
    notes = report["reconciliation"]["notes"] + [report["reconciliation"]["naive_attention"]["explanation"]]
    notes_list = '<ul>' + ''.join(f'<li>{escape(w)}</li>' for w in notes) + '</ul>'
    formulas = '<dl>' + ''.join(f'<dt>{escape(k)}</dt><dd><code>{escape(v)}</code></dd>'
                                for k, v in report["formulas"].items()) + '</dl>'
    mode = "primitive η/β 迁移" if _transferred(report) else "nominal"
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>第1章 {mode} 实验报告</title><style>'
            'body{font:16px/1.65 system-ui,"PingFang SC",sans-serif;color:#172b3a;max-width:1250px;'
            'margin:32px auto;padding:0 24px}img{display:block;width:100%;height:auto}figure{margin:32px 0}'
            'a{color:#176b9b}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}'
            '.table{overflow-x:auto}th,td{padding:9px 10px;border-bottom:1px solid #d9e2e8;text-align:right;'
            'white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#f3f6f8}'
            'code,dd{overflow-wrap:anywhere}dt{font-weight:600;margin-top:16px}figcaption{color:#526775;'
            f'font-size:14px}}h2{{margin-top:42px}}</style><h1>第1章：18点 {mode} 实验报告</h1>'
            f'<p>协议 <code>{report["protocol"]}</code>；创建于 <code>{escape(report["stored_at"])}</code>。</p>'
            '<p><strong>解析预测完成；不代表历史实测、先预测后测量纪律或 W7900D 硬件验收已完成。</strong></p>'
            + transfer_intro
            + '<p><a href="report.json">JSON 底稿</a> · <a href="components.csv">组件 CSV</a> · '
            '<a href="report.md">Markdown 完整报告</a></p>' + warning_list
            + '<h2>组件瓶颈与18点结果</h2>' + picture("roofline-rx9070xt.svg", f"双屋顶：保留全部54个 {mode} 组件点；精确重合位置覆盖，编号与配置见右侧表。")
            + '<p>每个组件选择矩阵/向量峰值，取 max(compute, memory)，step 再求和。全局 AI 只供展示。</p>'
            + summary + '<p>Prefill token/s 是输入处理率；decode token/s 是生成速率，不能混作服务吞吐。</p>'
            + picture("batching.svg", "固定 ctx 后 batching 复用 GEMM 权重；KV 扫描仍随 B 增长。")
            + picture("prefill.svg", "B=1；长序列的 full-square attention 计算项增长。")
            + '<h2>W7900D：待确认参数</h2>'
            + picture("roofline-w7900d-pending.svg", "参数未知，无硬件屋顶线；仅标18点的 AI，纵轴不是吞吐。")
            + '<h2>手算与近似表对账</h2><p>差异属于公式、流量假设与近似表口径，不标作硬件测量误差或校准通过。</p>'
            + ('<p>此表为校准前nominal基线；18点结果与曲线使用校准后时间。</p>' if _transferred(report) else '')
            + reconciled + notes_list + '<h2>Primitive 直觉练习</h2>' + primitives
            + '<h2>可复现公式与来源</h2>' + formulas
            + f'<p>report data SHA256：<code>{report["data_sha256"]}</code>。图、表、CSV 均来自同一 report 对象。'
            + '完整模型/计划/实现 hash、逐点 prediction_id 和时间见 JSON；未读取或重标课程已有测量为本次实验。</p>'
            + f'<p><a href="https://github.com/amdjiahangpan/hello-mlsys/blob/{_COURSE_COMMIT}/{_COURSE_PATH}">'
            + f'课程原文 {_COURSE_COMMIT[:12]} / 第1章</a></p></html>')


def write_chapter1_report(report: dict, out: Path) -> dict[str, Path]:
    """Write a fresh directory, refusing overwrite; every artifact uses one report.

    Existing artifacts can be re-rendered from their JSON without changing the
    timestamp. The data hash protects against accidentally mixing separate runs.
    """
    _canonical(report)
    if (not isinstance(report, dict) or report.get("protocol") not in (PROTOCOL, CALIBRATED_PROTOCOL)
            or report.get("point_count") != 18):
        _fail("Expected an intact chapter 1 report")
    payload = {key: value for key, value in report.items() if key != "data_sha256"}
    if report.get("data_sha256") != _sha(payload):
        _fail("Report data hash does not match; regenerate instead of editing prediction evidence")
    assets = {"report.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
              "components.csv": _csv(report), "report.md": _markdown(report), "report.html": _html(report),
              "roofline-rx9070xt.svg": _roofline_svg(report), "roofline-w7900d-pending.svg": _pending_svg(report),
              "batching.svg": _curve_svg(report, "decode"), "prefill.svg": _curve_svg(report, "prefill")}
    target = Path(out)
    if target.exists() or target.is_symlink():
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite course report directory: {target}")
    try:
        target.mkdir(parents=True, exist_ok=False)
        for name, content in assets.items():
            with (target / name).open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
    except FileExistsError as exc:
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite course report: {target}") from exc
    return {name: target / name for name in assets}
