"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("experiment-form");
const state = {presets: null, last: null, running: false};
const number = (id) => Number(byId(id).value);
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const fmt = (value, digits = 1) => finite(value)
  ? value.toLocaleString("zh-CN", {minimumFractionDigits: 0, maximumFractionDigits: digits}) : "—";
const gib = (value) => finite(value) ? value / (1024 ** 3) : null;
const geometryFields = ["num_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_kv_heads", "head_dim", "vocab_size"];

function setStatus(text, kind = "") {
  byId("status-badge").textContent = text;
  byId("status-badge").className = `status-badge ${kind}`;
}

function syncControls() {
  const custom = byId("model_id").value === "custom";
  byId("custom-model").hidden = !custom;
  for (const field of byId("custom-model").querySelectorAll("input")) field.disabled = !custom;
  const model = custom ? Object.fromEntries(geometryFields.map((key) => [key, number(key)]))
    : state.presets?.models.find((item) => item.id === byId("model_id").value)?.model;
  byId("model-geometry").textContent = model
    ? `${model.num_layers} 层 · Hidden ${model.hidden_size} · Query ${model.num_attention_heads * model.head_dim} · ${model.num_attention_heads}Q / ${model.num_kv_heads}KV`
    : "自定义模型结构";
  const burst = byId("arrival").value === "burst";
  byId("rate_rps").disabled = burst;
  byId("arrival-help").textContent = burst ? "所有请求在 0 ms 同时到达；请求总数不等于运行 batch 上限。"
    : "固定数量的请求按声明方式到达，结束后排空；不是固定时长稳态实验。";
  const fifo = byId("scheduler").value === "fifo";
  for (const key of ["chunked_prefill", "mixed_batching", "max_batch_size", "max_batch_tokens"]) {
    byId(key).disabled = fifo;
  }
  if (fifo) {byId("chunked_prefill").checked = false; byId("mixed_batching").checked = false;}
  byId("prefill_chunk_tokens").disabled = fifo || !byId("chunked_prefill").checked;
}

function setHardware(preset) {
  for (const key of ["memory_gib", "compute_tflops", "bandwidth_gbps"]) byId(key).value = preset[key];
}

function applyDefaults() {
  const defaults = state.presets.defaults;
  byId("model_id").value = defaults.model_id;
  setHardware(defaults.hardware);
  const matching = state.presets.hardware.find((item) => ["memory_gib", "compute_tflops", "bandwidth_gbps"].every((key) => item[key] === defaults.hardware[key]));
  byId("hardware_preset").value = matching?.id || "custom";
  for (const section of ["workload", "strategy", "memory"]) {
    for (const [key, value] of Object.entries(defaults[section])) {
      const field = byId(key);
      if (!field) continue;
      if (field.type === "checkbox") field.checked = value; else field.value = value;
    }
  }
  syncControls();
}

function payload() {
  const fifo = byId("scheduler").value === "fifo";
  const chunked = !fifo && byId("chunked_prefill").checked;
  const data = {
    hardware: Object.fromEntries(["memory_gib", "compute_tflops", "bandwidth_gbps"].map((key) => [key, number(key)])),
    workload: {
      requests: number("requests"), input_tokens: number("input_tokens"), output_tokens: number("output_tokens"),
      arrival: byId("arrival").value,
      rate_rps: byId("arrival").value === "burst" ? state.presets.defaults.workload.rate_rps : number("rate_rps"),
      seed: number("seed"),
    },
    strategy: {
      scheduler: byId("scheduler").value, max_batch_size: fifo ? 1 : number("max_batch_size"),
      max_batch_tokens: fifo ? state.presets.defaults.strategy.max_batch_tokens : number("max_batch_tokens"),
      chunked_prefill: chunked,
      prefill_chunk_tokens: chunked ? number("prefill_chunk_tokens") : state.presets.defaults.strategy.prefill_chunk_tokens,
      mixed_batching: !fifo && byId("mixed_batching").checked,
    },
    memory: {kv_pool_gib: number("kv_pool_gib")},
  };
  if (byId("model_id").value === "custom") {
    data.model = {...Object.fromEntries(geometryFields.map((key) => [key, number(key)])),
      name: byId("model_name").value, tie_word_embeddings: byId("tie_word_embeddings").checked,
      weight_bytes: 2, kv_bytes: 2, activation_bytes: 2};
  } else data.model_id = byId("model_id").value;
  return data;
}

function markDirty() {
  if (state.last && !state.running) {
    setStatus("配置已修改", "warning");
    byId("feedback").hidden = false;
    byId("feedback").textContent = "图表仍显示上次运行结果。点击「运行模拟」应用新的配置。";
  }
}

function clearResults() {
  state.last = null;
  for (const key of ["ttft", "e2e", "throughput", "memory"]) byId(`metric-${key}`).textContent = "—";
  byId("memory-capacity").textContent = "逐设备预算 · 不重复计算 KV";
  byId("result-content").hidden = true;
  byId("empty-state").hidden = false;
}

function svgNode(tag, attrs = {}, text = null) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== null) node.textContent = text;
  return node;
}

function chartBase(id, description, width = null, height = 228) {
  const host = byId(id);
  width = Math.max(280, Math.min(width || 1200, Math.round(host.clientWidth) || 460));
  host.replaceChildren();
  const svg = svgNode("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": description});
  svg.append(svgNode("title", {}, description));
  host.append(svg);
  return {svg, width, height, left: 48, right: width - 14, top: 19, bottom: height - 30};
}

function emptyChart(id, message) {
  const element = document.createElement("p");
  element.className = "empty-chart";
  element.textContent = message;
  byId(id).replaceChildren(element);
}

function axes(chart, maximum, xEnd = null, unit = "ms") {
  const {svg, left, right, top, bottom} = chart;
  maximum = maximum > 0 ? maximum : 1;
  const y = (value) => bottom - value / maximum * (bottom - top);
  for (let tick = 0; tick <= 4; tick++) {
    const value = maximum * tick / 4;
    const position = y(value);
    svg.append(svgNode("line", {x1: left, x2: right, y1: position, y2: position, class: "grid-line"}));
    svg.append(svgNode("text", {x: left - 8, y: position + 3, "text-anchor": "end"}, fmt(value, maximum < 2 ? 2 : 1)));
  }
  svg.append(svgNode("line", {x1: left, x2: right, y1: bottom, y2: bottom, class: "axis-line"}));
  if (finite(xEnd)) {
    for (let tick = 0; tick <= 4; tick++) {
      const value = xEnd * tick / 4;
      svg.append(svgNode("text", {x: left + (right - left) * tick / 4, y: bottom + 20, "text-anchor": "middle"}, fmt(value, 1)));
    }
    svg.append(svgNode("text", {x: right, y: top - 7, "text-anchor": "end"}, unit));
  }
  return y;
}

function drawLatency() {
  if (!state.last) return;
  const selection = byId("latency-metric").value;
  const stats = state.last.charts.latency[selection];
  const rows = ["p50", "p95", "p99"].map((key) => ({label: key.toUpperCase(), value: stats?.[key]}));
  if (!rows.some((row) => finite(row.value))) {
    emptyChart("latency-chart", selection === "tpot_ms" ? "无 TPOT 样本：单 Token 输出或没有完成请求。" : "没有足够的完成请求，无法计算此指标。");
    return;
  }
  const chart = chartBase("latency-chart", `${selection} 分位数：${rows.map((row) => `${row.label} ${fmt(row.value)} ms`).join("，")}`);
  const max = Math.max(...rows.filter((row) => finite(row.value)).map((row) => row.value));
  const y = axes(chart, max * 1.22);
  const colors = ["#a6c8b5", "#5d9d83", "#187965"];
  rows.forEach((row, index) => {
    const x = chart.left + (chart.right - chart.left) * (index + 0.5) / 3;
    if (finite(row.value)) {
      const rect = svgNode("rect", {x: x - 25, y: y(row.value), width: 50, height: chart.bottom - y(row.value), rx: 3, fill: colors[index]});
      rect.append(svgNode("title", {}, `${row.label}: ${fmt(row.value, 3)} ms`));
      chart.svg.append(rect);
      chart.svg.append(svgNode("text", {x, y: y(row.value) - 8, "text-anchor": "middle", class: "chart-value"}, fmt(row.value)));
    }
    chart.svg.append(svgNode("text", {x, y: chart.bottom + 21, "text-anchor": "middle"}, row.label));
  });
}

function drawThroughput() {
  const rows = state.last.charts.throughput.filter((row) => finite(row.tokens_per_s) && finite(row.end_ms));
  if (!rows.length) return emptyChart("throughput-chart", "没有可用的输出 Token 时间区间。");
  const end = Math.max(...rows.map((row) => row.end_ms));
  if (!(end > 0)) return emptyChart("throughput-chart", "模拟时长为零，输出速率不可计算。");
  const chart = chartBase("throughput-chart", "输出 Token 分箱速率，横轴为模拟时间 ms，纵轴为 tokens/s");
  const y = axes(chart, Math.max(...rows.map((row) => row.tokens_per_s)) * 1.18, end);
  const x = (value) => chart.left + value / end * (chart.right - chart.left);
  let line = "";
  rows.forEach((row) => {
    line += `${line ? " L" : "M"}${x(row.start_ms)} ${y(row.tokens_per_s)} L${x(row.end_ms)} ${y(row.tokens_per_s)}`;
  });
  chart.svg.append(svgNode("path", {d: `${line} L${x(end)} ${chart.bottom} L${x(rows[0].start_ms)} ${chart.bottom} Z`, fill: "#18796512"}));
  chart.svg.append(svgNode("path", {d: line, fill: "none", stroke: "#187965", "stroke-width": 2}));
  for (const row of rows) {
    const point = svgNode("circle", {cx: x((row.start_ms + row.end_ms) / 2), cy: y(row.tokens_per_s), r: 3, fill: "#187965", opacity: 0.7});
    point.append(svgNode("title", {}, `${fmt(row.start_ms)}–${fmt(row.end_ms)} ms：${row.tokens} tokens / ${fmt(row.end_ms - row.start_ms, 3)} ms = ${fmt(row.tokens_per_s)} tokens/s`));
    chart.svg.append(point);
  }
}

function drawMemory() {
  const rows = state.last.charts.memory.filter((row) => finite(row.time_ms) && finite(row.budget_peak_bytes));
  if (!rows.length) return emptyChart("memory-chart", "当前运行没有可用的显存预算采样。");
  const device = rows[0].device_id;
  const samples = rows.filter((row) => row.device_id === device);
  const capacity = state.last.charts.memory_capacity_bytes?.[device] ?? samples[0].device_memory_bytes;
  const end = Math.max(1, ...samples.map((row) => row.time_ms));
  const maximum = Math.max(0, ...samples.map((row) => gib(row.budget_peak_bytes)), finite(capacity) ? gib(capacity) : 0);
  const chart = chartBase("memory-chart", `${device} 累计显存预算峰值与逻辑 KV 使用量，横轴为模拟时间 ms，纵轴为 GiB`, 940, 220);
  const y = axes(chart, maximum * 1.12, end);
  const x = (value) => chart.left + value / end * (chart.right - chart.left);
  if (finite(capacity)) chart.svg.append(svgNode("line", {x1: chart.left, x2: chart.right, y1: y(gib(capacity)), y2: y(gib(capacity)), stroke: "#a6b5a8", "stroke-dasharray": "5 4", "stroke-width": 1.5}));
  for (const [field, color] of [["budget_peak_bytes", "#187965"], ["used_kv_bytes", "#c5783c"]]) {
    const points = samples.filter((row) => finite(row[field]));
    let line = "";
    points.forEach((row, index) => {line += index ? ` H${x(row.time_ms)} V${y(gib(row[field]))}` : `M${x(row.time_ms)} ${y(gib(row[field]))}`;});
    if (line) chart.svg.append(svgNode("path", {d: line, fill: "none", stroke: color, "stroke-width": 2}));
    for (const row of points) {
      const point = svgNode("circle", {cx: x(row.time_ms), cy: y(gib(row[field])), r: 2.5, fill: color});
      point.append(svgNode("title", {}, `${fmt(row.time_ms)} ms · ${field === "budget_peak_bytes" ? "累计预算峰值" : "逻辑 KV 已用"}：${fmt(gib(row[field]), 4)} GiB`));
      chart.svg.append(point);
    }
  }
}

function showResult(data) {
  state.last = data;
  const {result, charts, requests, experiment, meta} = data;
  const complete = meta.execution_outcome === "COMPLETED" && result.status === "COMPLETED";
  const completed = requests.filter((row) => row.status === "COMPLETED").length;
  const other = requests.length - completed;
  setStatus(complete && !other ? "模拟完成" : "存在未完成请求", complete && !other ? "success" : "failed");
  byId("empty-state").hidden = true;
  byId("result-content").hidden = false;
  byId("metric-ttft").textContent = fmt(charts.latency.ttft_ms?.p50);
  byId("metric-e2e").textContent = fmt(charts.latency.e2e_ms?.p99);
  byId("metric-throughput").textContent = fmt(result.throughput?.output_tokens_per_s);
  const peak = Math.max(0, ...charts.memory.filter((row) => finite(row.budget_peak_bytes)).map((row) => row.budget_peak_bytes));
  byId("metric-memory").textContent = charts.memory.length ? fmt(gib(peak), 2) : "—";
  const capacities = Object.values(charts.memory_capacity_bytes || {});
  byId("memory-capacity").textContent = capacities.length ? `声明显存 ${fmt(gib(capacities[0]), 2)} GiB · 含固定 KV 池` : "包含固定 KV 池与安全余量";
  byId("completion-summary").textContent = `完成 ${completed} / ${requests.length} 请求${other ? ` · 其他终态或未完成 ${other}` : " · 所有请求已排空"}`;
  byId("timing-summary").textContent = `模拟区间 ${fmt(meta.elapsed_ms)} ms · 本机计算 ${fmt(meta.wall_elapsed_ms)} ms`;
  byId("run-description").textContent = `${experiment.strategy.model.name} · 单 GPU ${fmt(gib(experiment.infra.devices[0].memory_bytes))} GiB · 固定请求数 + 排空`;
  byId("feedback").hidden = complete && !other;
  if (!complete || other) byId("feedback").textContent = `保留本次 ${result.status} 运行的全部结果；请在请求明细中检查未完成或失败项。完成请求的延迟不代表所有请求通过。`;
  drawLatency(); drawThroughput(); drawMemory();
  byId("memory-note").textContent = `${meta.memory_thinned ? "采样已压缩并保留峰值。" : "内存事件采样点。"}预算曲线是截至该时刻的峰值，不是瞬时物理显存。KV 已用量包含在固定池内，两条曲线不能相加。`;
  const tbody = byId("request-rows");
  tbody.replaceChildren();
  for (const row of requests) {
    const tr = document.createElement("tr");
    for (const value of [row.request_id, fmt(row.arrival_ms), fmt(row.ttft_ms), fmt(row.tpot_ms), fmt(row.end_to_end_ms), row.output_tokens_actual ?? row.token_times_ms?.length ?? "—", row.status]) {
      const td = document.createElement("td"); td.textContent = value; tr.append(td);
    }
    tbody.append(tr);
  }
  byId("request-note").textContent = `保留全部 ${requests.length} 条请求。— 表示缺少样本；单输出 Token 的 TPOT 不适用。图表${meta.memory_rule ? "使用累计预算峰值口径" : "来自模拟事件"}。`;
  byId("experiment-json").textContent = JSON.stringify(experiment, null, 2);
}

function focusResults() {
  const section = document.querySelector(".results");
  window.scrollTo({top: Math.max(0, section.getBoundingClientRect().top + window.scrollY - 20), behavior: "auto"});
  byId("results-title").focus({preventScroll: true});
}

async function runExperiment(event) {
  event.preventDefault();
  if (state.running || !form.reportValidity()) return;
  state.running = true;
  const requestData = payload();
  clearResults();
  byId("error").hidden = true;
  byId("feedback").hidden = false;
  byId("feedback").textContent = "正在运行现有 CPU 模拟引擎，结果将在完成后显示。";
  setStatus("运行中");
  document.body.classList.add("busy");
  byId("run").disabled = true; byId("reset").disabled = true;
  byId("run-label").textContent = "运行中…";
  const controls = [...form.querySelectorAll("input,select")];
  const disabledBefore = controls.map((control) => control.disabled);
  controls.forEach((control) => {control.disabled = true;});
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 45000);
  try {
    const response = await fetch("/api/simulate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(requestData), signal: controller.signal});
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(`${data.error?.code || "REQUEST_FAILED"}：${data.error?.message || "运行失败，请检查配置。"}`);
    byId("feedback").hidden = true;
    showResult(data);
  } catch (error) {
    clearResults();
    setStatus("运行失败", "failed");
    byId("feedback").hidden = true;
    byId("error").hidden = false;
    byId("error").textContent = error.name === "AbortError" ? "等待超时。请减少请求数、输入长度或输出长度后重试。" : error.message;
  } finally {
    clearTimeout(timeout);
    controls.forEach((control, index) => {control.disabled = disabledBefore[index];});
    state.running = false;
    document.body.classList.remove("busy");
    byId("run").disabled = false; byId("reset").disabled = false;
    byId("run-label").textContent = "运行模拟";
    syncControls();
    focusResults();
  }
}

function download(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2) + "\n"], {type: "application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function switchTab(tab) {
  const requestTab = tab === "requests-tab";
  byId("requests-tab").setAttribute("aria-selected", requestTab);
  byId("config-tab").setAttribute("aria-selected", !requestTab);
  byId("requests-tab").tabIndex = requestTab ? 0 : -1;
  byId("config-tab").tabIndex = requestTab ? -1 : 0;
  byId("request-panel").hidden = !requestTab;
  byId("config-panel").hidden = requestTab;
}

async function initialize() {
  try {
    const response = await fetch("/api/presets");
    if (!response.ok) throw new Error("读取预设失败");
    state.presets = await response.json();
    const modelSelect = byId("model_id"); modelSelect.replaceChildren();
    for (const model of state.presets.models) modelSelect.add(new Option(model.label, model.id));
    modelSelect.add(new Option("自定义 Dense / GQA 模型", "custom"));
    const hardwareSelect = byId("hardware_preset"); hardwareSelect.replaceChildren();
    for (const hardware of state.presets.hardware) hardwareSelect.add(new Option(hardware.label, hardware.id));
    hardwareSelect.add(new Option("自定义参数", "custom"));
    applyDefaults();
    modelSelect.disabled = false; hardwareSelect.disabled = false;
    byId("run").disabled = false; byId("reset").disabled = false;
  } catch (error) {
    byId("error").hidden = false;
    byId("error").textContent = `无法连接本地模拟服务：${error.message}。请确认服务仍在运行后刷新。`;
    setStatus("连接失败", "failed");
  }
}

form.addEventListener("submit", runExperiment);
form.addEventListener("input", () => {syncControls(); markDirty();});
form.addEventListener("change", () => {syncControls(); markDirty();});
byId("hardware_preset").addEventListener("change", () => {
  const preset = state.presets.hardware.find((item) => item.id === byId("hardware_preset").value);
  if (preset) setHardware(preset);
});
for (const key of ["memory_gib", "compute_tflops", "bandwidth_gbps"]) byId(key).addEventListener("input", () => {byId("hardware_preset").value = "custom";});
byId("reset").addEventListener("click", () => {applyDefaults(); markDirty();});
byId("latency-metric").addEventListener("change", drawLatency);
byId("export-config").addEventListener("click", () => {if (state.last) download(state.last.experiment, "mlsys-experiment.json");});
byId("export-result").addEventListener("click", () => {if (state.last) download(state.last, "mlsys-result.json");});
for (const id of ["requests-tab", "config-tab"]) {
  byId(id).addEventListener("click", () => switchTab(id));
  byId(id).addEventListener("keydown", (event) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const next = event.key === "Home" ? "requests-tab" : event.key === "End" ? "config-tab" : id === "requests-tab" ? "config-tab" : "requests-tab";
      switchTab(next); byId(next).focus();
    }
  });
}
let resizeFrame = 0;
window.addEventListener("resize", () => {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => {if (state.last) {drawLatency(); drawThroughput(); drawMemory();}});
});
initialize();
