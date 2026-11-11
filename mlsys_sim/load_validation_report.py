"""Render sanitized finite-cohort studies without conflating SLA and accuracy.

No hardware, network, or raw request artifacts are accessed by this module.
All tables and SVG coordinates come from one bounded study JSON snapshot.
"""

from collections import Counter
from datetime import datetime
import csv
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
from typing import Any, NoReturn

from .schema import SimulationError


_COHORTS = ("baseline", "short", "long")
_NAMES = {"baseline": "基线", "short": "短输入留出", "long": "长输入留出"}
_METRICS = {"ttft_p99_ms": ("P99 TTFT", "ttft", "p99_ttft_ms"),
            "tpot_p99_ms": ("P99 TPOT", "tpot", "p99_tpot_ms"),
            "end_to_end_p99_ms": ("P99 E2E", "end_to_end", "p99_e2e_ms"),
            "elapsed_output_tokens_per_s": ("有限队列 elapsed 吞吐", None, None)}
_VERDICTS = {"PASS", "FAIL", "UNKNOWN"}
_VALIDATION_KEYS = {"schema_version", "kind", "protocol", "status", "verdict", "study_id", "cohort",
    "predictions_sha256", "spec_sha256", "identity_sha256", "contract", "contract_sha256", "runs", "points",
    "hardware_validated", "metric_contract", "limitations"}
_NOTES = [
    "有意扫描过载：SLA FAIL 不自动表示采集失败，也不自动表示预测精度失败；三类证据分别汇总。",
    "每个点保留所有预定重复，不选择最佳 seed，不平均掩盖失败；纵向范围为可用重复的 min–max，不是置信区间。",
    "缺失、样本不足或删失的重复不画成0，也不补插值；逐重复表保留其数值、样本数与失败原因。",
    "有限 cohort 吞吐=实际发出 token / 含必要 drain 的 elapsed；不是稳态到达率或队列稳定性证明。",
    "延迟采用冻结契约的时间边界；server token-ready 指标不是HTTP客户端/网络端到端延迟。",
    "相对误差沿用验证器的 abs(predicted−measured)/measured，分母是实测值；零分母未定义时留空，不改阈值或重拟合。",
    "最大已观测安全点仅为当前网格内所有预定重复 SLA PASS 的点，不是全局容量或连续安全区间。",
    "冻结收据用于承诺预测SHA，完整原始预测可本地留档；本 renderer 只展示输入的收据/源码哈希，不验证远端收据真实性。",
    "所有输出 hardware_validated=false；measured 表示输入的采集来源声明，不等于广域硬件精度认证。",
]


def _invalid(message: str) -> NoReturn:
    raise SimulationError("INVALID_LOAD_REPORT", message)


def _canonical(value: Any) -> bytes:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, OverflowError, UnicodeError) as exc:
        raise SimulationError("INVALID_LOAD_REPORT", "Study must contain finite UTF-8 JSON data") from exc
    if len(raw) > 8 * 1024 * 1024:
        _invalid("Sanitized study exceeds 8MiB")
    return raw


def _text(value: Any, label: str, maximum: int = 1024) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(ord(char) < 32 for char in value)):
        _invalid(f"{label} must be bounded nonempty text")
    return value


def _integer(value: Any, label: str, minimum: int = 0, maximum: int = 1_000_000) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _invalid(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _value(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        _invalid("Metrics must be finite nonnegative numbers or null")
    try:
        number = float(value)
    except OverflowError:
        _invalid("Metric exceeds finite numeric range")
    if not math.isfinite(number) or not 0 <= number <= 1e15:
        _invalid("Metrics must be finite and nonnegative")
    return number


def _verdict(value: Any) -> str:
    if not isinstance(value, str) or value not in _VERDICTS:
        _invalid("Verdict must be PASS, FAIL or UNKNOWN")
    return value


def _freeze(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        _invalid("Freeze evidence must be a nonempty map containing only SHA256/UTC records")
    for key, item in value.items():
        _text(key, "freeze key", 128)
        if key.startswith(("/", "~/")) or "\\" in key:
            _invalid("Freeze keys cannot contain private absolute paths")
        if isinstance(item, dict):
            _freeze(item, key)
        elif key.endswith("_at"):
            _text(item, key, 64)
            try:
                parsed = datetime.fromisoformat(item.replace("Z", "+00:00"))
                offset = parsed.utcoffset()
            except ValueError:
                _invalid("Freeze timestamps must be explicit UTC")
            if offset is None or offset.total_seconds() != 0:
                _invalid("Freeze timestamps must be explicit UTC")
        elif (not isinstance(item, str) or len(item) != 64
              or any(char not in "0123456789abcdef" for char in item)):
            _invalid(f"freeze.{label}.{key} must be a lowercase SHA256, never a host or private path")


def _validate(study: dict) -> dict:
    raw = _canonical(study)
    required = {"schema_version", "study_id", "title", "evidence_kind", "metadata", "freeze", "selection",
                "cohorts", "attempt_summary", "limitations"}
    if not isinstance(study, dict) or set(study) != required:
        _invalid("Study has missing or unsupported top-level fields")
    if type(study["schema_version"]) is not int or study["schema_version"] != 1:
        _invalid("schema_version must be 1")
    _text(study["study_id"], "study_id", 128)
    _text(study["title"], "title", 256)
    if study["evidence_kind"] not in ("measured", "test_fixture"):
        _invalid("evidence_kind must explicitly identify measured or test_fixture")
    if not isinstance(study["metadata"], dict) or not {"model", "device", "runtime", "protocol"} <= study["metadata"].keys():
        _invalid("metadata requires sanitized model/device/runtime/protocol")
    for key in ("model", "device", "runtime", "protocol"):
        _text(study["metadata"][key], f"metadata.{key}")
    _freeze(study["freeze"], "freeze")
    if not {"frozen_at", "receipt_sha256", "raw_prediction_sha256", "canonical_prediction_sha256",
            "collection_source_sha256"} <= study["freeze"].keys():
        _invalid("Freeze must retain timestamp, receipt, raw/canonical prediction and collector hashes")
    selection = study["selection"]
    if not isinstance(selection, dict) or not {"predicted_safe_cap", "holdout_concurrencies"} <= selection.keys():
        _invalid("selection requires the frozen predicted cap and holdout grid")
    if selection["predicted_safe_cap"] is not None:
        _integer(selection["predicted_safe_cap"], "predicted_safe_cap", 1, 10_000)
    grid = selection["holdout_concurrencies"]
    if (not isinstance(grid, list) or not grid or len(grid) > 64
            or any(type(c) is not int or not 1 <= c <= 10_000 for c in grid)
            or grid != sorted(set(grid))):
        _invalid("holdout_concurrencies must be an increasing bounded grid")
    _integer(selection.get("replications", 3), "replications", 1, 100)
    cohorts = study["cohorts"]
    if not isinstance(cohorts, dict) or set(cohorts) != set(_COHORTS):
        _invalid("cohorts must contain baseline, short and long validation summaries")
    for name, report in cohorts.items():
        if (not isinstance(report, dict) or set(report) - _VALIDATION_KEYS
                or not {"kind", "protocol", "runs", "points", "contract"} <= report.keys()
                or report["kind"] != "closed_loop_validation" or report["protocol"] != "finite_cohort_closed_loop_v1"):
            _invalid(f"{name} must be a sanitized closed_loop_validation result, never raw request artifacts")
        if report.get("hardware_validated", False) is not False:
            _invalid("Renderer cannot promote a validation result to hardware_validated")
        if (not isinstance(report["contract"], dict)
                or type(report["contract"].get("min_completed_samples")) is not int
                or report["contract"]["min_completed_samples"] < 1):
            _invalid("Each cohort needs its preregistered minimum completed sample count")
        if not isinstance(report["runs"], list) or len(report["runs"]) > 256 or not isinstance(report["points"], list):
            _invalid("Validation runs/points must be bounded arrays")
        seen = set()
        for run in report["runs"]:
            if (not isinstance(run, dict) or set(run) - {"concurrency", "replication", "policy_sha256",
                    "request_stream_sha256", "verdict", "checks", "metrics", "prediction_sla_verdict",
                    "observation_sla_verdict", "prediction", "observation", "acquisition_verdict",
                    "prediction_error_verdict", "sla_agreement"}):
                _invalid("Validation cells must be objects")
            cell_key = (_integer(run.get("concurrency"), "concurrency", 1, 10_000),
                        _integer(run.get("replication"), "replication", 0, selection.get("replications", 3) - 1))
            if cell_key in seen:
                _invalid("Duplicate concurrency/replication validation cells")
            seen.add(cell_key)
            for field in ("prediction_sla_verdict", "observation_sla_verdict"):
                _verdict(run.get(field, "UNKNOWN"))
            for field in ("acquisition_verdict", "prediction_error_verdict"):
                _verdict(run.get(field, "UNKNOWN"))
            for side in ("prediction", "observation"):
                summary = run.get(side)
                if summary is None:
                    continue
                if (not isinstance(summary, dict) or "metrics" not in summary
                        or set(summary) - {"status", "metrics", "counts", "offered_count", "not_offered_count",
                                          "constraints", "source_kind", "session_sha256", "timing_boundary"}):
                    _invalid("Cell summaries must contain sanitized aggregate evidence only")
                if side == "observation" and summary.get("source_kind") != study["evidence_kind"]:
                    _invalid("Mixed measured/fixture evidence cannot be silently promoted")
                if not isinstance(summary["metrics"], dict):
                    _invalid("Cell metrics must be an object")
                counts = summary.get("counts", {})
                if not isinstance(counts, dict):
                    _invalid("Summary counts must be an object")
                for key in ("total", "completed", "failed", "timed_out", "unfinished", "not_admitted"):
                    if counts.get(key) is not None:
                        _integer(counts[key], f"counts.{key}")
                for key in ("offered_count", "not_offered_count"):
                    if summary.get(key) is not None:
                        _integer(summary[key], key)
                values = summary["metrics"]
                if values.get("completed_samples") is not None:
                    _integer(values["completed_samples"], "completed_samples")
                if not isinstance(values.get("sample_counts", {}), dict):
                    _invalid("sample_counts must be an object")
                for count in values.get("sample_counts", {}).values():
                    _integer(count, "metric sample count")
                for metric in _METRICS:
                    _value(summary["metrics"].get(metric))
            if not isinstance(run.get("checks", []), list) or any(not isinstance(c, dict) for c in run.get("checks", [])):
                _invalid("Cell checks must be an object array")
            if not isinstance(run.get("metrics", {}), dict):
                _invalid("Metric comparisons must be an object")
        for point in report["points"]:
            if not isinstance(point, dict):
                _invalid("Grid points must be objects")
            _integer(point.get("concurrency"), "point concurrency", 1, 10_000)
    attempts = study["attempt_summary"]
    if not isinstance(attempts, dict) or "expected_cells" not in attempts:
        _invalid("attempt_summary requires an explicit expected cell count")
    _integer(attempts["expected_cells"], "expected_cells", 1, 768)
    if not isinstance(study["limitations"], list) or len(study["limitations"]) > 64:
        _invalid("limitations must be a bounded text array")
    for item in study["limitations"]:
        _text(item, "limitation", 4096)
    return json.loads(raw)


def _censored(summary: dict | None) -> bool:
    if summary is None:
        return False
    counts = summary.get("counts", {})
    return (summary.get("status") != "COMPLETED"
        or any((counts.get(name) or 0) > 0 for name in ("failed", "timed_out", "unfinished", "not_admitted"))
        or (summary.get("not_offered_count") or 0) > 0)


def _seed(study: dict, replication: int) -> str:
    seed = study["metadata"].get("seed")
    return f"r{replication}（base seed {seed} 子流）" if seed is not None else f"r{replication}（base seed未提供）"


def _cells(study: dict) -> list[dict]:
    result = []
    repeats = study["selection"].get("replications", 3)
    for cohort in _COHORTS:
        validation = study["cohorts"][cohort]
        grid = sorted({p["concurrency"] for p in validation["points"]}
                      | {r["concurrency"] for r in validation["runs"]})
        if cohort != "baseline":
            grid = sorted(set(grid) | set(study["selection"]["holdout_concurrencies"]))
        indexed = {(r["concurrency"], r["replication"]): r for r in validation["runs"]}
        for concurrency in grid:
            for replication in range(repeats):
                raw = indexed.get((concurrency, replication), {})
                observation = raw.get("observation")
                predicted = raw.get("prediction")
                ps, ms = raw.get("prediction_sla_verdict", "UNKNOWN"), raw.get("observation_sla_verdict", "UNKNOWN")
                checks = raw.get("checks", [])
                flags = []
                if observation is None:
                    flags.append("缺测")
                if _censored(observation):
                    flags.append("删失/未完整完成")
                minimum = validation["contract"]["min_completed_samples"]
                if observation is not None and (observation["metrics"].get("completed_samples") or 0) < minimum:
                    flags.append("完成样本不足")
                nonpass = [c for c in checks if c.get("verdict") != "PASS"]
                result.append({"cohort": cohort, "concurrency": concurrency, "replication": replication,
                    "replication_label": _seed(study, replication), "prediction": predicted, "observation": observation,
                    "acquisition_verdict": raw.get("acquisition_verdict", "UNKNOWN"),
                    "prediction_error_verdict": raw.get("prediction_error_verdict", "UNKNOWN"),
                    "prediction_sla_verdict": ps, "observation_sla_verdict": ms,
                    "sla_agreement": ps == ms if (ps in {"PASS", "FAIL"} and ms in {"PASS", "FAIL"}
                                                  and raw.get("acquisition_verdict") == "PASS") else None,
                    "false_safe": ps == "PASS" and ms != "PASS", "censored": _censored(observation),
                    "missing": observation is None, "flags": flags, "nonpass_checks": nonpass,
                    "metric_errors": raw.get("metrics", {}), "minimum_samples": minimum})
    if len(result) != study["attempt_summary"]["expected_cells"]:
        _invalid("Declared expected_cells differs from the complete concurrency × replication grid")
    return result


def _merge(values: list[str]) -> str:
    return "FAIL" if "FAIL" in values else "UNKNOWN" if not values or "UNKNOWN" in values else "PASS"


def _summarize(study: dict, cells: list[dict], digest: str) -> dict:
    cohorts = {}
    repeats = study["selection"].get("replications", 3)
    for cohort in _COHORTS:
        selected = [row for row in cells if row["cohort"] == cohort]
        points = []
        for concurrency in sorted({row["concurrency"] for row in selected}):
            rows = [row for row in selected if row["concurrency"] == concurrency]
            observed = _merge([row["observation_sla_verdict"] for row in rows])
            points.append({"concurrency": concurrency, "observed_sla_verdict": observed,
                "observed_all_repetitions_pass": len(rows) == repeats and all(
                    r["observation_sla_verdict"] == "PASS" and r["acquisition_verdict"] == "PASS"
                    and not r["missing"] and not r["censored"] for r in rows),
                "observed_repetition_verdicts": [r["observation_sla_verdict"] for r in rows]})
        safe = [p["concurrency"] for p in points if p["observed_all_repetitions_pass"]]
        nonmonotonic = any(a["observed_sla_verdict"] == "FAIL" and b["observed_all_repetitions_pass"]
                           for index, a in enumerate(points) for b in points[index + 1:])
        cohorts[cohort] = {"points": points, "largest_observed_safe_grid_point": max(safe) if safe else None,
                          "safe_grid_nonmonotonic": nonmonotonic, "required_repetitions": repeats,
                          "interpolated": False, "global_capacity_established": False}
    known = [r for r in cells if r["sla_agreement"] is not None]
    return {"schema_version": 1, "kind": "closed_loop_study_report", "status": "COMPLETED",
        "study_id": study["study_id"], "study_sha256": digest, "evidence_kind": study["evidence_kind"],
        "hardware_validated": False, "expected_cells": len(cells),
        "observed_cells": sum(not c["missing"] for c in cells), "missing_cells": sum(c["missing"] for c in cells),
        "censored_cells": sum(c["censored"] for c in cells),
        "acquisition_verdict": _merge([r["acquisition_verdict"] for r in cells]),
        "acquisition_counts": dict(Counter(r["acquisition_verdict"] for r in cells)),
        "prediction_error_verdict": _merge([r["prediction_error_verdict"] for r in cells]),
        "prediction_error_counts": dict(Counter(r["prediction_error_verdict"] for r in cells)),
        "sla_agreement": {"agree": sum(r["sla_agreement"] is True for r in known), "known": len(known),
            "unknown": len(cells) - len(known), "false_safe_or_unconfirmed": sum(r["false_safe"] for r in cells),
            "false_safe_observed_fail": sum(r["false_safe"] and r["observation_sla_verdict"] == "FAIL" for r in cells),
            "predicted_pass_observed_unknown": sum(r["false_safe"] and r["observation_sla_verdict"] == "UNKNOWN" for r in cells)},
        "cohorts": cohorts, "no_refit": True, "no_best_repeat_selection": True,
        "range_definition": "available repetition min–max, not a confidence interval; individual repetitions retained",
        "limitations": _NOTES + study["limitations"]}


def _metric(row: dict, side: str, metric: str) -> float | None:
    summary = row[side]
    if summary is None or _censored(summary):
        return None
    values = summary["metrics"]
    count_name = _METRICS[metric][1]
    count = values.get("sample_counts", {}).get(count_name) if count_name is not None else values.get("completed_samples")
    if type(count) is not int or count < row["minimum_samples"]:
        return None
    return _value(values.get(metric))


def _svg(study: dict, cells: list[dict], digest: str, title: str, panels: list[tuple[str, str, str]]) -> str:
    width, height, panel_width = 440 * len(panels), 490, 440
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{escape(title)}</title>', f'<metadata>canonical report输入SHA256={digest}; hardware_validated=false</metadata>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<style>text{font:12px system-ui,"PingFang SC",sans-serif;fill:#172b3a}.title{font-size:19px;font-weight:600}</style>',
        f'<text x="24" y="30" class="title">{escape(title)}</text>',
        '<text x="24" y="55">蓝空心：冻结预测各重复；橙点：实测各重复；红点：实测SLA非PASS；竖线：min–max，非置信区间</text>']
    for panel, (cohort, metric, subtitle) in enumerate(panels):
        rows = [r for r in cells if r["cohort"] == cohort]
        grid = sorted({r["concurrency"] for r in rows})
        x0, x1, top, bottom = panel * panel_width + 65, (panel + 1) * panel_width - 25, 110, 375
        values = [v for r in rows for side in ("prediction", "observation") if (v := _metric(r, side, metric)) is not None]
        latency = study["cohorts"][cohort]["contract"].get("constraints", {}).get("latency", {})
        limit = latency.get(_METRICS[metric][2])
        limit = _value(limit) if limit is not None else None
        ymax = max([*values, limit or 0, 1]) * 1.12

        def xposition(concurrency: int) -> float:
            if len(grid) == 1:
                return (x0 + x1) / 2
            return x0 + (math.log2(concurrency) - math.log2(grid[0])) / (math.log2(grid[-1]) - math.log2(grid[0])) * (x1 - x0)

        def yposition(value: float) -> float:
            return bottom - value / ymax * (bottom - top)

        parts.append(f'<text x="{x0}" y="88">{escape(subtitle)}</text>')
        for step in range(5):
            value = ymax * step / 4
            y = yposition(value)
            parts += [f'<line x1="{x0}" x2="{x1}" y1="{y:.2f}" y2="{y:.2f}" stroke="#d9e2e8"/>',
                      f'<text x="{x0-48}" y="{y+4:.2f}">{value:.3g}</text>']
        if limit is not None:
            y = yposition(limit)
            parts += [f'<line x1="{x0}" x2="{x1}" y1="{y:.2f}" y2="{y:.2f}" stroke="#8d4860" stroke-dasharray="5,4"/>',
                      f'<text x="{x0+5}" y="{y-5:.2f}">SLA {limit:g} ms</text>']
        for concurrency in grid:
            x = xposition(concurrency)
            parts.append(f'<text x="{x-8:.2f}" y="398">{concurrency}</text>')
            group = [r for r in rows if r["concurrency"] == concurrency]
            for side, color, offset in (("prediction", "#176b9b", -5), ("observation", "#b0521c", 5)):
                available = [(r, v) for r in group if (v := _metric(r, side, metric)) is not None]
                if available:
                    low, high = min(v for _, v in available), max(v for _, v in available)
                    parts.append(f'<line x1="{x+offset:.2f}" x2="{x+offset:.2f}" y1="{yposition(low):.2f}" '
                                 f'y2="{yposition(high):.2f}" stroke="{color}" stroke-width="1.5"/>')
                for row, value in available:
                    marker_color = "#b42538" if side == "observation" and row["observation_sla_verdict"] != "PASS" else color
                    fill = "white" if side == "prediction" else marker_color
                    jitter = (row["replication"] - (len(group) - 1) / 2) * 2
                    label = f"{cohort} C{concurrency} {row['replication_label']} {side} {metric}={value:.12g}"
                    parts.append(f'<g><title>{escape(label)}</title><circle cx="{x+offset+jitter:.2f}" cy="{yposition(value):.2f}" '
                                 f'r="3.2" fill="{fill}" stroke="{marker_color}" stroke-width="1.4"/></g>')
                if side == "observation" and len(available) < len(group):
                    parts.append(f'<text x="{x-10:.2f}" y="420">n={len(available)}/{len(group)}</text>')
        unit = "token/s" if metric == "elapsed_output_tokens_per_s" else "ms"
        parts.append(f'<text x="{x0+50}" y="445">并发 clients（log₂）；纵轴 {unit}</text>')
        if not values:
            parts.append(f'<text x="{x0+15}" y="245">无完整可比较样本；不补0、不插值</text>')
    parts += [f'<text x="24" y="476">来源 {escape(study["evidence_kind"])} · canonical report输入SHA256 {digest[:16]} · finite cohort 含必要drain · hardware_validated=false</text>', '</svg>']
    return '\n'.join(parts)


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}" if type(value) is float else str(value)


def _cell_rows(cells: list[dict]) -> tuple[list[str], list[list[Any]]]:
    headers = ["cohort", "clients", "replication", "stream", "acquisition", "prediction_error", "predicted_sla",
               "observed_sla", "sla_agreement", "false_safe_or_unconfirmed", "completed_samples",
               "ttft_samples", "tpot_samples", "e2e_samples"]
    for metric in _METRICS:
        headers += [f"predicted_{metric}", f"observed_{metric}", f"relative_error_{metric}"]
    headers += ["flags", "metric_errors_json", "nonpass_checks"]
    rows = []
    for cell in cells:
        observed_metrics = cell["observation"]["metrics"] if cell["observation"] is not None else {}
        sample_counts = observed_metrics.get("sample_counts", {})
        values: list[Any] = [cell["cohort"], cell["concurrency"], cell["replication"], cell["replication_label"],
            cell["acquisition_verdict"], cell["prediction_error_verdict"], cell["prediction_sla_verdict"],
            cell["observation_sla_verdict"], cell["sla_agreement"], cell["false_safe"], observed_metrics.get("completed_samples"),
            sample_counts.get("ttft"), sample_counts.get("tpot"), sample_counts.get("end_to_end")]
        for metric in _METRICS:
            values += [(cell[side]["metrics"].get(metric) if cell[side] is not None else None)
                       for side in ("prediction", "observation")]
            comparison = cell["metric_errors"].get(metric, {})
            values.append(comparison.get("relative_error") if isinstance(comparison, dict) else None)
        values += ["；".join(cell["flags"]), json.dumps(cell["metric_errors"], ensure_ascii=False, separators=(",", ":")),
                   json.dumps(cell["nonpass_checks"], ensure_ascii=False, separators=(",", ":"))]
        rows.append(values)
    return headers, rows


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(_fmt(value))}</td>' for value in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return _fmt(value).replace("|", "\\|").replace("\n", " ")
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *['| ' + ' | '.join(clean(v) for v in row) + ' |' for row in rows]])


def write_closed_loop_study_report(study: dict, out: Path) -> dict[str, Path]:
    """Write a new report directory, preserving each predefined repeat and gap."""
    study = _validate(study)
    digest = hashlib.sha256(_canonical(study)).hexdigest()
    cells = _cells(study)
    summary = _summarize(study, cells, digest)
    headers, rows = _cell_rows(cells)
    csv_stream = io.StringIO(newline="")
    writer = csv.writer(csv_stream)
    writer.writerow(headers)
    writer.writerows(rows)
    known = summary["sla_agreement"]
    overview = [["协议/采集检查", summary["acquisition_verdict"], json.dumps(summary["acquisition_counts"], ensure_ascii=False)],
                ["预注册精度目标", summary["prediction_error_verdict"], json.dumps(summary["prediction_error_counts"], ensure_ascii=False)],
                ["SLA判定一致", f"{known['agree']}/{known['known']} 已知且采集PASS单元", f"UNKNOWN/协议未PASS={known['unknown']}"],
                ["预测PASS但实测未PASS", known["false_safe_or_unconfirmed"],
                 f"实测FAIL={known['false_safe_observed_fail']}；实测UNKNOWN={known['predicted_pass_observed_unknown']}"],
                ["观测完整性", f"{summary['observed_cells']}/{summary['expected_cells']}",
                 f"缺测={summary['missing_cells']}；删失={summary['censored_cells']}"]]
    safe_rows = [[_NAMES[name], summary["cohorts"][name]["largest_observed_safe_grid_point"],
                  summary["cohorts"][name]["required_repetitions"], summary["cohorts"][name]["safe_grid_nonmonotonic"]]
                 for name in _COHORTS]
    metadata_rows = [[key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value]
                     for key, value in study["metadata"].items()]
    freeze_rows = [[key, json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value]
                   for key, value in study["freeze"].items()]
    attempt_rows = [[key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value]
                    for key, value in study["attempt_summary"].items()]
    prefix = "测试夹具：没有GPU实测" if study["evidence_kind"] == "test_fixture" else "真机采集来源声明：有限冻结网格研究"
    boundary = ("server token-ready（非HTTP）" if all(v["contract"].get("timing_boundary") == "server_token_ready_v1"
                for v in study["cohorts"].values()) else "各cohort冻结时间边界")
    intro = (f"{prefix}。单元数{summary['expected_cells']}；冻结预测安全上限C={_fmt(study['selection']['predicted_safe_cap'])}；{boundary}。"
             "SLA过载点与采集/精度分开判断；不重拟合、不挑重复、不插值。")
    notes = summary["limitations"]
    md = [f"# {study['title']}", "", intro, "", _md_table(["证据层", "结论", "分母与细分"], overview), "",
        "## 基线延迟：冻结预测与实测", "", "![基线P99 TTFT/TPOT/E2E](baseline-latency.svg)", "",
        "每个点显示各重复；范围为可用重复的min–max，不是置信区间。红色实测点SLA非PASS仍完整保留。", "",
        "## 有限队列吞吐", "", "![含必要drain的elapsed输出token吞吐](cohort-throughput.svg)", "",
        "分母是有限cohort实际elapsed，含必要drain；不是稳态负载或容量证明。", "",
        "## 短/长输入独立留出", "", "![短长输入E2E留出对照](holdout-e2e.svg)", "",
        "## 已观测安全网格点", "", _md_table(["cohort", "最大已观测安全C", "要求重复数", "FAIL后再PASS（非单调）"], safe_rows), "",
        "安全点要求所有预定重复均SLA PASS且完整；UNKNOWN和删失不能算安全。非单调时尤其不能解释为连续安全区间。", "",
        "## 采集尝试记录（输入声明）", "", _md_table(["尝试记录", "值"], attempt_rows), "",
        "## 每个预定单元", "", "重复r序号是base seed派生的长度子流编号，不是不同Runtime采样seed。缺样本为—，CSV中为空。", "",
        _md_table(headers, rows), "", "## 冻结与输入追溯", "", _md_table(["元数据", "值"], metadata_rows), "",
        _md_table(["冻结记录", "UTC / SHA256"], freeze_rows), "", f"canonical report输入SHA256：`{digest}`。", "",
        "## 边界", "", *["- " + text for text in notes], ""]
    figures = ''.join(f'<figure><img src="{name}" alt="{title}"><figcaption>{title}</figcaption></figure>'
                     for name, title in (("baseline-latency.svg", "基线：P99 TTFT / TPOT / E2E；保留各重复与min–max范围"),
                     ("cohort-throughput.svg", "有限cohort实际输出token / 含必要drain的elapsed"),
                     ("holdout-e2e.svg", "短/长输入留出：冻结预测与实测E2E")))
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(study["title"])}</title><style>body{{font:16px/1.65 system-ui,"PingFang SC",sans-serif;'
        'max-width:1360px;margin:32px auto;padding:0 24px;color:#172b3a}h2{margin-top:38px}img{width:100%;display:block}'
        'figure{margin:30px 0}figcaption{font-size:14px;color:#526775}.scroll{overflow:auto;max-height:700px}'
        'table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}'
        'th,td{padding:8px 10px;border-bottom:1px solid #d9e2e8;text-align:right;white-space:nowrap}'
        'th:first-child,td:first-child{text-align:left}th{background:#f3f6f8;position:sticky;top:0}code{overflow-wrap:anywhere}'
        f'</style><h1>{escape(study["title"])}</h1><p><strong>{escape(intro)}</strong></p>'
        '<p><a href="report.json">sanitized输入</a> · <a href="summary.json">分层汇总</a> · '
        '<a href="cells.csv">全部单元CSV</a> · <a href="report.md">Markdown报告</a></p>'
        + _html_table(["证据层", "结论", "分母与细分"], overview)
        + '<details><summary>适用边界与证据限制（展开）</summary><ul>'
        + ''.join(f'<li>{escape(text)}</li>' for text in notes) + '</ul></details>'
        + '<h2>冻结预测与实测</h2><p>各重复独立显示；min–max不是置信区间。缺测/删失留空，不补0。'
        '红色点表示SLA非PASS，不能被平均隐藏。</p>' + figures
        + '<h2>最大已观测安全网格点</h2>'
        + _html_table(["cohort", "安全C（仅当前网格）", "要求全部重复数", "FAIL后再PASS（非单调）"], safe_rows)
        + '<p>要求每个预定重复SLA PASS且完整；不插值，不推断全局容量。FAIL后再PASS时不解释为连续安全区间。</p>'
        + '<h2>采集尝试记录（输入声明）</h2>' + _html_table(["尝试记录", "值"], attempt_rows)
        + '<h2>全部预定单元</h2><p>重复r序号是base seed派生长度子流，不是不同Runtime采样seed。'
        '表可横向/纵向滚动，缺值为—。部分完成时保留原始汇总，但图中不绘制其删失统计。'
        '相对误差=abs(预测−实测)/实测，以小数表示；完整p/m/error/n/verdict保留于metric_errors_json。</p>'
        + _html_table(headers, rows) + '<h2>冻结、源码与输入追溯</h2>'
        + _html_table(["元数据", "值"], metadata_rows) + _html_table(["冻结记录", "UTC / SHA256"], freeze_rows)
        + f'<p>canonical report输入SHA256：<code>{digest}</code>。所有图、表和CSV都由该sanitized输入生成。hardware_validated=false。</p></html>')
    assets = {"report.json": json.dumps(study, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "report.html": html, "report.md": '\n'.join(md), "cells.csv": csv_stream.getvalue(),
        "baseline-latency.svg": _svg(study, cells, digest, "基线：冻结预测与实测 P99 延迟", [
            ("baseline", metric, label) for metric, label in (("ttft_p99_ms", "P99 TTFT"),
                ("tpot_p99_ms", "P99 TPOT"), ("end_to_end_p99_ms", "P99 E2E"))]),
        "cohort-throughput.svg": _svg(study, cells, digest, "有限 cohort：实际输出 token / elapsed（含必要drain）", [
            (cohort, "elapsed_output_tokens_per_s", _NAMES[cohort]) for cohort in _COHORTS]),
        "holdout-e2e.svg": _svg(study, cells, digest, "独立留出：短 / 长输入 E2E 对照", [
            ("short", "end_to_end_p99_ms", "短输入留出 P99 E2E"),
            ("long", "end_to_end_p99_ms", "长输入留出 P99 E2E")])}
    target = Path(out)
    if target.exists() or target.is_symlink():
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite closed-loop study report: {target}")
    try:
        target.mkdir(parents=True, exist_ok=False)
        for name, content in assets.items():
            with (target / name).open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
    except FileExistsError as exc:
        raise SimulationError("OUTPUT_EXISTS", f"Refusing to overwrite closed-loop study report: {target}") from exc
    return {name: target / name for name in assets}
