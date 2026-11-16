"""Data-derived retrospective validation reports; no measured-CPU attribution."""

import csv
from html import escape
import json
from pathlib import Path


def _key(cell):
    return cell["domain_sha256"], cell["shape"], cell["session_id"]


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _error_svg(rows: list[dict], threshold: float) -> str:
    width, left, right, top, gap = 980, 250, 925, 65, 58
    extent = max([100] + [abs(row[name]) * 100 for row in rows for name in ("before_signed", "after_signed")])
    extent = int(extent / 20 + 1) * 20
    height = top + max(1, len(rows)) * gap + 50
    def x(value):
        return left + (value + extent) / (2 * extent) * (right - left)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="独立形状留出：校准前后有符号耗时误差百分比">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="12" fill="#263b33">',
             '<text x="20" y="25" font-size="17">事后分组留出：校准前后误差</text>',
             '<text x="20" y="45">橙：旧默认　绿：校准后；分母为实测时间，负值表示预测偏快</text>']
    a, b = x(-threshold * 100), x(threshold * 100)
    parts.append(f'<rect x="{a}" y="{top-15}" width="{b-a}" height="{height-top-10}" fill="#edf5ee"/>')
    for value in range(-extent, extent + 1, 20):
        position = x(value)
        parts.append(f'<path d="M{position},{top-15} V{height-30}" stroke="{"#718277" if value==0 else "#e0e8e2"}"/>')
        parts.append(f'<text x="{position}" y="{height-10}" text-anchor="middle">{value}%</text>')
    for index, row in enumerate(rows):
        y = top + index * gap + 8
        before, after = x(row["before_signed"] * 100), x(row["after_signed"] * 100)
        label = escape(f"{row['model_id']} {row['phase']} {row['shape']}")
        parts.append(f'<text x="20" y="{y+4}">{label}</text><path d="M{before},{y} H{after}" stroke="#b8c4ba" stroke-width="2"/>')
        for position, value, color in ((before, row["before_signed"], "#c87942"), (after, row["after_signed"], "#187965")):
            parts.append(f'<circle cx="{position}" cy="{y}" r="5" fill="{color}"><title>{label}: {value*100:+.2f}%</title></circle>')
    if not rows:
        parts.append('<text x="20" y="80">无可绘制配对：旧默认缺失或记录被拒绝；完整信息见报告JSON。</text>')
    return "".join(parts) + "</g></svg>"


def write_cost_validation_report(report: dict, destination: str | Path) -> None:
    """Write one consistent snapshot; output directory must not exist."""
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    rows = []
    for split in ("holdout", "repeats"):
        group = report[split]
        default = {_key(row): row for row in (group["legacy_default"] or {}).get("cells", [])}
        for cell in group["calibrated"]["cells"]:
            old = default.get(_key(cell))
            actual = cell["observed_ms"]
            rows.append({**cell, "split": split,
                "before_ms": old["predicted_ms"] if old else None,
                "before_signed": (old["predicted_ms"] - actual) / actual if old else None,
                "after_signed": (cell["predicted_ms"] - actual) / actual})
    rows.sort(key=lambda row: (row["split"], row["model_id"], row["phase"], row["shape"], row["session_id"]))
    fields = ("split", "model_id", "phase", "shape", "session_id", "sample_count", "observed_ms",
              "before_ms", "predicted_ms", "before_signed", "after_signed", "passed")
    with (out / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_rows = [row for row in rows if row["split"] == "holdout" and row["before_signed"] is not None]
    (out / "holdout-errors.svg").write_text(_error_svg(plot_rows, report["max_relative_error"]))
    summary = report["holdout"]
    old = summary["legacy_default"]
    body = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
        "<title>运行域成本校准：独立分组留出</title><style>body{font:15px/1.8 system-ui,sans-serif;color:#233b30;background:#f5f7f3;max-width:1120px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px;border-bottom:1px solid #dae2da;text-align:right}th:first-child,td:first-child{text-align:left}img{width:100%;background:white}section{overflow:auto}code{overflow-wrap:anywhere}h1{font-size:26px}small{color:#607565}</style>",
        "<h1>固定开销与 Attention 成本校准</h1>",
        f"<p><strong>主留出判定：{escape(report['verdict'])}</strong>；覆盖 {summary['covered_record_count']}/{summary['declared_record_count']} 步。</p>",
        "<p>这是事后按形状和独立采集进程分组的留出。全部18点汇总此前已被查看，不是新盲测。步骤嵌套于采集进程，不是同等数量的独立硬件试验。</p>",
        f"<p>主留出等权分组 MAPE：旧默认 {_fmt(old['cell_mape']*100 if old and old['cell_mape'] is not None else None,2)}% → 校准 {_fmt(summary['calibrated']['cell_mape']*100 if summary['calibrated']['cell_mape'] is not None else None,2)}%。阈值在拟合前冻结为 {report['max_relative_error']*100:g}%。</p>",
        "<img src='holdout-errors.svg' alt='独立形状留出校准前后误差'><h2>所有留出与保留形状重复组</h2><section><table><thead><tr><th>模型/阶段/形状</th><th>分组</th><th>步数</th><th>实测 ms</th><th>旧默认 ms</th><th>校准 ms</th><th>校准误差</th><th>判定</th></tr></thead><tbody>"]
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
            f"{row['model_id']} {row['phase']} {row['shape']}", row["split"], row["sample_count"],
            _fmt(row["observed_ms"]), _fmt(row["before_ms"]), _fmt(row["predicted_ms"]),
            f"{row['after_signed']*100:+.2f}%", "PASS" if row["passed"] else "FAIL")) + "</tr>")
    rejected = [{**row, "split": split} for split in ("holdout", "repeats") for row in report[split]["rejected"]]
    if rejected:
        body.append("</tbody></table></section><h2>拒绝记录</h2><section><table><thead><tr><th>记录</th><th>分组</th><th>原因</th></tr></thead><tbody>")
        for row in rejected:
            body.append("<tr>" + "".join(f"<td>{escape(str(row[key]))}</td>" for key in ("record_id", "split", "reason")) + "</tr>")
    body += ["</tbody></table></section><h2>覆盖与解释边界</h2>",
        "<p>固定项是经验上未建模工作，不是直接测得的 CPU 时间。GPU 分量与整步墙钟来自不同采集渠道；Graph 只绑定已记录的配置模式，逐步实际分派未观测。Decode profiler 聚合存在上下文对齐近似，保留全部失败、拒绝和进程状态变化。</p>",
        f"<p>保留形状重复判定：{escape(report['repeats']['verdict'])}。主留出与重复验证分开报告，不按结果更换训练或测试组。</p>",
        "<p><a href='report.json'>全部逐步误差与拒绝记录</a> · <a href='cells.csv'>分组CSV</a>；hardware_validated=false。</p>",
        f"<small>fit SHA256 <code>{escape(report['fit_sha256'])}</code><br>split SHA256 <code>{escape(report['split_sha256'])}</code></small></html>"]
    (out / "report.html").write_text("".join(body))
