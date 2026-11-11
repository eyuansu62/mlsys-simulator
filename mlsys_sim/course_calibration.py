"""Import a course primitive eta/beta experiment without promoting its evidence.

The source is data, never executable code. It is not a whole-stage Profile,
native PP calibration, workspace measurement, or proof of a new hardware run.
"""

from dataclasses import replace
from datetime import datetime
import csv
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, NoReturn

from .course_roofline import _canonical, _sha, _timestamp
from .schema import Experiment, SimulationError


CALIBRATION_PROTOCOL = "hello_mlsys_ch2_primitive_eta_beta_v1"
_BASE = {"flops", "bytes", "ai", "t_comp_ms", "t_mem_ms", "t_nom_ms", "binding", "median_ms",
         "per_round_ms", "ratio_meas_over_nom", "fit"}
_CAL_SHAPES = {"gemm": ("2048^3", "4096^3", "5120^3", "8192^3"),
               "ln": ("2048x1024", "8192x4096"), "softmax": ("4096x4096", "8192x4096")}
_HOLD = {"gemm_6144": ("gemm", "6144^3"), "ln_16384x4096": ("ln", "16384x4096"),
         "softmax_16384x4096": ("softmax", "16384x4096"),
         "attn_mha16_s512": ("attn", "512"), "attn_mha16_s1024": ("attn", "1024")}
_LIMITATIONS = [
    "来源是导入的课程历史参考或显式测试夹具；未执行新的 GPU 采集，hardware_validated=false。",
    "拟合仅使用原始 fit 标记的 GEMM 与流式 LN/Softmax；留出点不参与重拟合。",
    "原始预测时间戳不证明测量前已落盘；源脚本在全部留出测量结束后统一保存，缺少独立留出测量起止时间。",
    "原始 Attention S512 按 nominal 访存侧选分支；此处另列双侧重算 max，保留原值不覆盖。",
    "eta 来自大 GEMM，beta 来自流式 LN/Softmax；迁移到全模型、向量计算或 Attention 是域外假设。",
    "事件计时不能识别 CPU T_launch 或框架开销；patch 不修改 launch_overhead_us / framework_overhead_us。",
    "本导入不替代 whole-stage Profile、native PP 通信或 Workspace，不授予 TP/PP/缓存/swap/混合批处理硬件有效性。",
]


def _fail(message: str) -> NoReturn:
    raise SimulationError("INVALID_COURSE_CALIBRATION", message)


def _object(value: Any, required: set[str], label: str, optional: set[str] | None = None) -> dict:
    if (not isinstance(value, dict) or not required <= value.keys()
            or set(value) - required - (optional or set())):
        _fail(f"{label} has missing or unsupported fields")
    return value


def _text(value: Any, label: str, limit: int = 1024) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any(ord(c) < 32 for c in value)):
        _fail(f"{label} must be bounded text without control characters")
    return value


def _number(value: Any, label: str, *, positive: bool = False, maximum: float = 1e16) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be finite numeric data")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        _fail(f"{label} must be finite numeric data")
    if not math.isfinite(number) or not (1e-12 <= number <= maximum if positive else 0 <= number <= maximum):
        _fail(f"{label} is outside its finite numeric bounds")
    return number


def _near(actual: Any, expected: float, label: str, decimals: int) -> None:
    value = _number(actual, label)
    if abs(value - expected) > 0.500001 * 10**-decimals:
        _fail(f"{label} is inconsistent with its declared counts/timing or rounding")


def _signed(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        _fail(f"{label} must be a finite signed number")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        _fail(f"{label} must be a finite signed number")
    if not math.isfinite(number) or abs(number) > 1e12:
        _fail(f"{label} is outside its finite bounds")
    return number


def _historical_time(value: Any, label: str) -> None:
    _text(value, label, 64)
    try:
        time = datetime.fromisoformat(value.replace("Z", "+00:00"))
        offset = time.utcoffset()
    except ValueError:
        _fail(f"{label} must be an explicit UTC timestamp")
    if offset is None or offset.total_seconds() != 0:
        _fail(f"{label} must be an explicit UTC timestamp")


def _counts(kind: str, shape: str) -> tuple[int, int]:
    if kind == "gemm":
        n = int(shape[:-2])
        return 2 * n**3, 6 * n**2
    if kind == "attn":
        s = int(shape)
        return 4 * s**2 * 16 * 128, 8 * s * 16 * 128
    rows, columns = (int(x) for x in shape.split("x"))
    return 4 * rows * columns, 4 * rows * columns


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON object keys are not permitted")
        result[key] = value
    return result


def _row(value: Any, kind: str, shape: str, peak: float, bandwidth: float, *, holdout: bool = False) -> dict:
    extras = {"eta_point"} if kind == "gemm" else ({"achieved_tflops", "achieved_bw_gbps"}
               if kind == "attn" else {"beta_point"})
    row = _object(value, _BASE | extras | ({"t_pred_cal_ms", "pred_err_pct"} if holdout else set()), "record")
    flops, traffic = _counts(kind, shape)
    if type(row["flops"]) is not int or type(row["bytes"]) is not int or (row["flops"], row["bytes"]) != (flops, traffic):
        _fail("Primitive FLOPs/bytes must match the declared course shape formula")
    tc, tm = flops / peak, traffic / bandwidth
    for field, expected, decimals in (("ai", flops / traffic, 1), ("t_comp_ms", tc, 5),
        ("t_mem_ms", tm, 5), ("t_nom_ms", max(tc, tm), 5)):
        _near(row[field], expected, field, decimals)
    if row["binding"] != ("compute" if tc >= tm else "memory"):
        _fail("Source nominal binding does not match its declared counts")
    if type(row["fit"]) is not bool or (holdout and row["fit"]):
        _fail("fit must be boolean and holdout points cannot participate in fitting")
    measured = _number(row["median_ms"], "median_ms", positive=True, maximum=1e6)
    rounds = row["per_round_ms"]
    if not isinstance(rounds, list) or len(rounds) != 3:
        _fail("Exactly three per-round medians are required")
    for number in rounds:
        _number(number, "per_round_ms", positive=True, maximum=1e6)
    _near(measured, statistics.median(rounds), "median of per_round_ms", 5)
    _near(row["ratio_meas_over_nom"], measured / max(tc, tm), "ratio_meas_over_nom", 3)
    if kind == "gemm":
        _near(row["eta_point"], tc / measured, "eta_point", 4)
    elif kind != "attn":
        _near(row["beta_point"], tm / measured, "beta_point", 4)
    else:
        _near(row["achieved_tflops"], flops / measured / 1e9, "achieved_tflops", 1)
        _near(row["achieved_bw_gbps"], traffic / measured / 1e6, "achieved_bw_gbps", 1)
    if holdout:
        predicted = _number(row["t_pred_cal_ms"], "t_pred_cal_ms", positive=True, maximum=1e6)
        error = _signed(row["pred_err_pct"], "pred_err_pct")
        if abs(error - (measured / predicted - 1) * 100) > 0.050001:
            _fail("Source error must equal (measured/source_prediction - 1)*100")
    return row


def _source(value: Any, digest: str, fixture: bool) -> dict:
    source = _object(value, {"kind", "label"}, "source",
                     {"repository", "commit_sha", "path", "url", "expected_sha256"})
    if source["kind"] != ("test_fixture" if fixture else "course_external_report"):
        _fail("Explicit test_fixture source kind must match the raw test_fixture=true marker")
    for key, item in source.items():
        _text(item, f"source.{key}")
    for key, length in (("expected_sha256", 64), ("commit_sha", 40)):
        if key in source and (len(source[key]) != length or any(c not in "0123456789abcdef" for c in source[key])):
            _fail(f"source.{key} must be lowercase hexadecimal")
    if "expected_sha256" in source and source["expected_sha256"] != digest:
        _fail("Raw source SHA256 does not match expected_sha256")
    return source


def import_course_calibration(raw: bytes, *, source: dict, stored_at: str | None = None) -> dict:
    """Validate, reproduce and preserve historical course calibration data.

    Published rounded coefficients are applied only after an independent check
    against fit membership and sum(nominal times)/sum(measured times).
    """
    if not isinstance(raw, bytes) or not raw or len(raw) > 1024 * 1024:
        _fail("raw must be nonempty UTF-8 JSON bytes, at most 1MiB")
    try:
        text = raw.decode("utf-8")
        data = json.loads(raw, object_pairs_hook=_unique_object)
        _canonical(data)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise SimulationError("INVALID_COURSE_CALIBRATION", "Raw course data must be finite UTF-8 JSON") from exc
    data = _object(data, {"environment", "config", "nominal_constants", "calibration", "fit",
                         "holdout_prediction", "holdout_measured", "holdout_err_pct"}, "raw", {"test_fixture"})
    fixture = data.get("test_fixture", False)
    if type(fixture) is not bool or ("test_fixture" in data and not fixture):
        _fail("Optional test_fixture marker must be explicitly true")
    digest = hashlib.sha256(raw).hexdigest()
    source = _source(source, digest, fixture)
    environment = _object(data["environment"], {"timestamp_utc", "python", "torch", "rocm", "os", "gpu",
                                              "gpu_arch", "vram_gib", "driver"}, "environment")
    for key in ("timestamp_utc", "python", "torch", "rocm", "os", "gpu", "driver"):
        _text(environment[key], f"environment.{key}")
    _number(environment["vram_gib"], "environment.vram_gib", positive=True, maximum=1e6)
    arch = environment["gpu_arch"]
    if not isinstance(arch, dict) or not 1 <= len(arch) <= 16:
        _fail("gpu_arch must be a bounded declared architecture map")
    for key, value in arch.items():
        _text(key, "gpu_arch key")
        _text(value, "gpu_arch value")
    config = _object(data["config"], {"seed", "protocol"}, "config")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**32 - 1:
        _fail("seed must be a bounded integer")
    _text(config["protocol"], "config.protocol")
    constants = _object(data["nominal_constants"], {"peak_tflops", "bw_gbps", "ridge"}, "nominal_constants")
    peak = _number(constants["peak_tflops"], "peak_tflops", positive=True, maximum=1e6) * 1e9
    bandwidth = _number(constants["bw_gbps"], "bw_gbps", positive=True, maximum=1e6) * 1e6
    _near(constants["ridge"], peak / bandwidth, "ridge", 1)
    calibration = _object(data["calibration"], set(_CAL_SHAPES), "calibration")
    records: list[dict] = []
    for kind, shapes in _CAL_SHAPES.items():
        rows = _object(calibration[kind], set(shapes), kind)
        for shape in shapes:
            row = _row(rows[shape], kind, shape, peak, bandwidth)
            records.append({"id": f"{kind}_{shape}", "family": kind, "shape": shape, "source_record": row})
    fit = _object(data["fit"], {"stored_at", "eta", "beta", "eta_formula", "beta_formula", "fit_points",
                              "eta_points", "beta_points"}, "fit")
    _historical_time(fit["stored_at"], "fit.stored_at")
    for key in ("eta_formula", "beta_formula"):
        _text(fit[key], key)
    eta = _number(fit["eta"], "eta", positive=True, maximum=1)
    beta = _number(fit["beta"], "beta", positive=True, maximum=1)
    membership = _object(fit["fit_points"], {"eta", "beta"}, "fit_points")
    expected_eta = [r["shape"] for r in records if r["family"] == "gemm" and r["source_record"]["fit"]]
    expected_beta = [r["id"] for r in records if r["family"] != "gemm" and r["source_record"]["fit"]]
    if (membership != {"eta": expected_eta, "beta": expected_beta}
            or set(expected_eta) != {"4096^3", "5120^3"}
            or set(expected_beta) != {"ln_8192x4096", "softmax_8192x4096"}):
        _fail("Explicit fit membership must match flags and the course representative regime")
    ef = [r["source_record"] for r in records if r["family"] == "gemm" and r["source_record"]["fit"]]
    bf = [r["source_record"] for r in records if r["family"] != "gemm" and r["source_record"]["fit"]]
    recomputed_eta = math.fsum(r["t_comp_ms"] for r in ef) / math.fsum(r["median_ms"] for r in ef)
    recomputed_beta = math.fsum(r["t_mem_ms"] for r in bf) / math.fsum(r["median_ms"] for r in bf)
    _near(eta, recomputed_eta, "published eta", 4)
    _near(beta, recomputed_beta, "published beta", 4)
    for label, expected in (("eta_points", {r["shape"]: r["source_record"]["eta_point"] for r in records if r["family"] == "gemm"}),
                            ("beta_points", {r["id"]: r["source_record"]["beta_point"] for r in records if r["family"] != "gemm"})):
        _object(fit[label], set(expected), label)
        for key, number in expected.items():
            _near(fit[label][key], number, label, 4)
    prediction = _object(data["holdout_prediction"], {"stored_at", "predictions_ms"}, "holdout_prediction")
    _historical_time(prediction["stored_at"], "holdout_prediction.stored_at")
    prediction_rows = _object(prediction["predictions_ms"], set(_HOLD), "predictions_ms")
    measured_rows = _object(data["holdout_measured"], set(_HOLD), "holdout_measured")
    errors = _object(data["holdout_err_pct"], set(_HOLD), "holdout_err_pct")
    holdout: list[dict] = []
    for key, (kind, shape) in _HOLD.items():
        row = _row(measured_rows[key], kind, shape, peak, bandwidth, holdout=True)
        pred = prediction_rows[key]
        if kind == "attn":
            _object(pred, {"t_pred_cal_ms", "binding_by_ai", "ai"}, "attention source prediction")
            if pred["binding_by_ai"] != row["binding"]:
                _fail("Source attention nominal binding mismatch")
            _near(pred["ai"], row["ai"], "source prediction ai", 1)
            pred = pred["t_pred_cal_ms"]
        _near(pred, row["t_pred_cal_ms"], "source holdout prediction", 5)
        source_branch_prediction = (row["flops"] / peak / eta if row["binding"] == "compute"
                                    else row["bytes"] / bandwidth / beta)
        _near(pred, source_branch_prediction, "published source nominal-side prediction", 5)
        if abs(_signed(errors[key], "holdout error summary") - row["pred_err_pct"]) > 1e-8:
            _fail("holdout error summary differs from its source row")
        tc, tm = row["flops"] / peak / eta, row["bytes"] / bandwidth / beta
        recomputed = max(tc, tm)
        holdout.append({"id": key, "family": kind, "shape": shape, "source_record": row,
            "source_prediction_ms": pred, "source_error_pct": row["pred_err_pct"],
            "recomputed_t_compute_ms": tc, "recomputed_t_memory_ms": tm, "recomputed_prediction_ms": recomputed,
            "recomputed_bound": "compute" if tc >= tm else "memory",
            "recomputed_error_pct": (row["median_ms"] / recomputed - 1) * 100,
            "error_definition": "(measured_ms / prediction_ms - 1) * 100; positive = prediction optimistic",
            "scope": "out_of_fit_domain_attention" if kind == "attn" else "source_declared_regime_new_shape",
            "unseen_fit_shape": True})
    version = f"course-eta-beta-{digest[:16]}"
    patch = {"kind": "calibration_spec_patch", "requires": {"backend": "analytical"},
             "patch": {"compute_efficiency": eta, "bandwidth_efficiency": beta, "profile_version": version},
             "apply_automatically": False, "hardware_validated": False, "source_kind": source["kind"],
             "raw_sha256": digest, "limitations": _LIMITATIONS.copy()}
    result = {"schema_version": 1, "protocol": CALIBRATION_PROTOCOL, "status": "COMPLETED",
        "calibration_id": version, "stored_at": _timestamp(stored_at), "source": dict(source),
        "source_kind": source["kind"], "raw_sha256": digest, "source_text": text,
        "source_environment": environment, "nominal_constants": constants,
        "source_config": config, "coefficients": {"eta": eta, "beta": beta},
        "fit": {"source": fit, "recomputed_from_source_rounded_times": {"eta": recomputed_eta, "beta": recomputed_beta},
                "recomputed_from_exact_counts": {"eta": math.fsum(r["flops"] / peak for r in ef) / math.fsum(r["median_ms"] for r in ef),
                    "beta": math.fsum(r["bytes"] / bandwidth for r in bf) / math.fsum(r["median_ms"] for r in bf)},
                "method": "sum_nominal_times_over_sum_measured_times; retain published four-decimal coefficients"},
        "calibration_records": records, "holdout_records": holdout,
        "source_holdout_prediction": prediction, "calibration_patch": patch,
        "fitting_domain": {"eta": "single-device BF16 square GEMM 4096^3 and 5120^3 saturated regime",
                           "beta": "single-device streaming BF16 LN/Softmax 8192x4096",
                           "timing_boundary": "isolated_operator_event_sync", "tp": 1, "pp": 1},
        "historical_prediction_before_measurement_established": False, "hardware_validated": False,
        "limitations": _LIMITATIONS.copy()}
    result["data_sha256"] = _sha(result)
    return result


def validate_course_calibration_report(report: dict) -> None:
    """Check both artifact integrity and its reproducible raw-source derivation."""
    _canonical(report)
    if not isinstance(report, dict) or report.get("protocol") != CALIBRATION_PROTOCOL:
        _fail("Expected an imported course calibration report")
    if report.get("data_sha256") != _sha({k: v for k, v in report.items() if k != "data_sha256"}):
        _fail("Course calibration report hash mismatch")
    if not isinstance(report.get("source_text"), str):
        _fail("Original source text is required")
    _historical_time(report.get("stored_at"), "import report stored_at")
    # Historical report import time is preserved; recreating artifacts is not a new fit.
    expected = import_course_calibration(report["source_text"].encode("utf-8"), source=report["source"])
    expected["stored_at"] = report["stored_at"]
    expected["data_sha256"] = _sha({k: v for k, v in expected.items() if k != "data_sha256"})
    if expected != report:
        _fail("Course calibration report does not match its raw derivation")


def apply_course_calibration(experiment: Experiment, report: dict, *, allow_test_fixture: bool = False) -> Experiment:
    """Explicitly apply only the three reviewed analytical efficiency fields."""
    from .config import validate_experiment

    validate_course_calibration_report(report)
    if experiment.calibration.backend != "analytical":
        raise SimulationError("UNSUPPORTED_MECHANISM", "Course primitive calibration applies only to an explicit analytical backend")
    validate_experiment(experiment)
    if type(allow_test_fixture) is not bool:
        _fail("allow_test_fixture must be explicitly boolean")
    if report["source_kind"] == "test_fixture" and not allow_test_fixture:
        raise SimulationError("UNSUPPORTED_MECHANISM", "Test-fixture application requires explicit allow_test_fixture=True")
    patch = report["calibration_patch"]["patch"]
    result = replace(experiment, calibration=replace(experiment.calibration, **patch))
    validate_experiment(result)
    return result


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return '<table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def _coefficient_svg(report: dict) -> str:
    rows = report["calibration_records"]
    maximum = max(1.5, max(r["source_record"].get("eta_point", r["source_record"].get("beta_point")) for r in rows) * 1.15)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="410" viewBox="0 0 1120 410" role="img">',
           '<title>各形状的效率系数；实心点参与拟合，空心点仅用于诊断</title>',
           f'<metadata>report.json data_sha256={report["data_sha256"]}; hardware_validated=false</metadata>',
           '<rect width="1120" height="410" fill="white"/><style>text{font:13px system-ui,sans-serif;fill:#172b3a}</style>',
           '<text x="30" y="30">各形状效率：实心点参与拟合，空心点仅诊断；不同 regime 不能混作全卡常数</text>']
    for index, row in enumerate(rows):
        record = row["source_record"]
        value = record.get("eta_point", record.get("beta_point"))
        x, y = 105 + index * 135, 300 - value / maximum * 220
        color = "#176b9b" if row["family"] == "gemm" else "#b0521c"
        fill = color if record["fit"] else "white"
        svg += [f'<line x1="{x}" x2="{x}" y1="300" y2="{y:.2f}" stroke="#d9e2e8"/>',
                f'<circle cx="{x}" cy="{y:.2f}" r="6" fill="{fill}" stroke="{color}" stroke-width="2"/>',
                f'<text x="{x-25}" y="{y-15:.2f}">{value:.4f}</text>',
                f'<text x="{x-44}" y="330">{escape(row["family"])}</text>',
                f'<text x="{x-44}" y="350">{escape(row["shape"])}</text>']
    svg += [f'<text x="30" y="390">数据：report.json · {report["data_sha256"][:16]}；来源 {escape(report["source_kind"])}；无新实测</text>', '</svg>']
    return '\n'.join(svg)


def _holdout_csv(report: dict) -> str:
    stream = io.StringIO(newline="")
    fields = ["id", "scope", "source_prediction_ms", "measured_ms", "source_error_pct", "nominal_bound",
              "recomputed_t_compute_ms", "recomputed_t_memory_ms", "recomputed_prediction_ms", "recomputed_bound",
              "recomputed_error_pct"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in report["holdout_records"]:
        writer.writerow({key: row["source_record"]["median_ms"] if key == "measured_ms" else
                         row["source_record"]["binding"] if key == "nominal_bound" else row[key] for key in fields})
    return stream.getvalue()


def write_course_calibration_report(report: dict, out: Path) -> dict[str, Path]:
    """Write an exclusive historical import report and an unapplied parameter patch."""
    validate_course_calibration_report(report)
    coefficient_rows = [[r["id"], str(r["source_record"]["fit"]),
        f'{r["source_record"].get("eta_point", r["source_record"].get("beta_point")):.4f}'] for r in report["calibration_records"]]
    holdout_rows = [[r["id"], f'{r["source_prediction_ms"]:.5f}', f'{r["recomputed_prediction_ms"]:.8f}',
        f'{r["source_record"]["median_ms"]:.5f}', f'{r["source_error_pct"]:+.1f}', f'{r["recomputed_error_pct"]:+.3f}',
        r["source_record"]["binding"] + " → " + r["recomputed_bound"]] for r in report["holdout_records"]]
    md = ["# 课程 η/β 导入与复算", "", f"来源 `{report['source_kind']}`；raw SHA256 `{report['raw_sha256']}`。", "",
          f"应用系数：η={report['coefficients']['eta']:.4f}，β={report['coefficients']['beta']:.4f}。", "",
          "![拟合点与非拟合诊断点](calibration.svg)", "", "| 点 | 参与拟合 | 单点效率 |", "| --- | --- | ---: |"]
    md += ["| " + " | ".join(row) + " |" for row in coefficient_rows]
    md += ["", "| 留出点 | 源预测 ms | max复算 ms | 源测量 ms | 源误差 % | max复算误差 % | 瓶颈 |",
           "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    md += ["| " + " | ".join(row) + " |" for row in holdout_rows]
    md += ["", "误差=(measured/prediction−1)×100；正值表示预测偏乐观。源测量只是导入声明，未晋升本库实测。", ""]
    md += ["- " + text for text in report["limitations"]]
    md += ["", "`calibration_patch.json` 仅供显式审阅/应用；未修改任何实验、Profile、launch、PP或Workspace。", "",
           f"导入时间 `{report['stored_at']}`；data SHA256 `{report['data_sha256']}`。", ""]
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>课程 η/β 导入报告</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:30px auto;'
        'padding:0 20px;color:#172b3a}img{width:100%}table{border-collapse:collapse;font-size:13px;width:100%;'
        'font-variant-numeric:tabular-nums}th,td{padding:9px;text-align:right;border-bottom:1px solid #d9e2e8}'
        'th:first-child,td:first-child{text-align:left}code{overflow-wrap:anywhere}</style><h1>课程 η/β 导入与复算</h1>'
        f'<p>来源 <code>{escape(report["source_kind"])}</code>；η={report["coefficients"]["eta"]:.4f}，'
        f'β={report["coefficients"]["beta"]:.4f}。历史参考或测试夹具，无新实测、无硬件PASS。</p>'
        '<p><a href="source.json">精确原始JSON</a> · <a href="report.json">复算底稿</a> · '
        '<a href="calibration_patch.json">未应用的参数patch</a> · <a href="holdout_points.csv">留出CSV</a></p>'
        '<img src="calibration.svg" alt="拟合点与非拟合诊断点的效率系数">'
        + _table(["点", "参与拟合", "单点效率"], coefficient_rows) + '<h2>留出：源结果与双侧max复算并列</h2>'
        + _table(["留出点", "源预测 ms", "max复算 ms", "源测量 ms", "源误差 %", "复算误差 %", "瓶颈"], holdout_rows)
        + '<p>误差=(measured/prediction−1)×100；正值表示预测偏乐观。未使用留出点重新拟合。</p><ul>'
        + ''.join(f'<li>{escape(text)}</li>' for text in report["limitations"]) + '</ul>'
        + f'<p>raw SHA256 <code>{report["raw_sha256"]}</code>；report SHA256 <code>{report["data_sha256"]}</code>。</p></html>')
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["id", "fit", "source_point_efficiency"])
    writer.writerows(coefficient_rows)
    assets = {"report.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
              "report.html": html, "report.md": "\n".join(md), "calibration.svg": _coefficient_svg(report),
              "calibration_points.csv": stream.getvalue(), "holdout_points.csv": _holdout_csv(report),
              "calibration_patch.json": json.dumps(report["calibration_patch"], ensure_ascii=False, indent=2) + "\n",
              "source.json": report["source_text"]}
    target = Path(out)
    if target.exists() or target.is_symlink():
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite course calibration report: {target}")
    try:
        target.mkdir(parents=True, exist_ok=False)
        for name, content in assets.items():
            with (target / name).open("x", encoding="utf-8", newline="") as file:
                file.write(content)
    except FileExistsError as exc:
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite course calibration report: {target}") from exc
    return {name: target / name for name in assets}
