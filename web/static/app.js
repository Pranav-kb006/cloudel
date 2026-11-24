/* PROGRESSIVE AI — Enterprise Cloud Infrastructure Dashboard Controller */
"use strict";

const ROUTE_TITLES = {
  dashboard: "Dashboard",
  live: "Live Experiment",
  models: "Model Registry",
  history: "Experiment History",
  latency: "Latency Analytics",
  memory: "Memory Analytics",
  network: "Network Analytics",
  architecture: "Architecture",
  pipeline: "Execution Pipeline",
  logs: "System Logs",
  docs: "Documentation",
};

const CHARTS = {};
let APP_MODELS = [];
let APP_HISTORY = [];
let APP_ABLATION = null;
let APP_LOGS = [];
let LIVE_MONO = null;
let LIVE_PROG = null;
let LIVE_COMP = null;
let COMP_METRIC = "ttft";
let POLL_TIMER = null;
let LOG_PAUSED = false;

Chart.defaults.font.family = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
Chart.defaults.color = "#64748B";
Chart.defaults.borderColor = "#E2E8F0";

/* ---------- Navigation & Shell ---------- */
function navigate(route) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  const target = document.getElementById("view-" + route);
  if (target) target.classList.add("active");

  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.route === route);
  });
  document.getElementById("crumbCurrent").textContent = ROUTE_TITLES[route] || route;
  closeMobileNav();

  if (route === "models") fetchModels();
  if (route === "history") fetchHistory();
  if (route === "logs") fetchLogs();
  if (["latency", "memory", "network", "pipeline"].includes(route)) {
    setTimeout(() => renderAnalyticsCharts(), 30);
  }
}

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("collapsed");
}
function openMobileNav() {
  document.getElementById("sidebar").classList.add("mobile-open");
  document.getElementById("mobileOverlay").classList.add("show");
}
function closeMobileNav() {
  document.getElementById("sidebar").classList.remove("mobile-open");
  document.getElementById("mobileOverlay").classList.remove("show");
}

function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.classList.remove("show"), 3000);
}

/* ---------- Chart Helpers ---------- */
function upsertChart(id, cfg) {
  const el = document.getElementById(id);
  if (!el) return;
  if (CHARTS[id]) {
    CHARTS[id].destroy();
  }
  CHARTS[id] = new Chart(el, cfg);
}

/* ---------- Bootstrap ---------- */
async function bootstrap() {
  if (window.lucide) lucide.createIcons();
  await Promise.all([fetchStatus(true), fetchModels(), fetchAblation(), fetchHistory(), fetchLogs()]);
  renderDashboardAll();
  renderAnalyticsCharts();
  setInterval(() => {
    if (!LOG_PAUSED) fetchLogs();
  }, 2500);
}

async function fetchStatus(initial = false) {
  try {
    const res = await fetch("/api/status");
    const st = await res.json();
    applyStatus(st, initial);
  } catch (e) {}
}

async function fetchModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    APP_MODELS = data.models || [];
    populateModelSelect();
    renderModelsTable();
  } catch (e) {}
}

async function fetchAblation() {
  try {
    const res = await fetch("/api/ablation");
    const data = await res.json();
    if (data && data.bandwidth_sweep) {
      APP_ABLATION = data;
      if (!LIVE_MONO && !LIVE_PROG && data.gantt_250mbps) {
        LIVE_MONO = data.gantt_250mbps.monolithic;
        LIVE_PROG = data.gantt_250mbps.progressive;
        LIVE_COMP = computeComparison(LIVE_MONO, LIVE_PROG);
      }
    }
  } catch (e) {}
}

async function fetchHistory() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    APP_HISTORY = data.history || [];
    renderHistoryTable();
  } catch (e) {}
}

async function fetchLogs() {
  try {
    const res = await fetch("/api/logs");
    const data = await res.json();
    APP_LOGS = data.logs || [];
    renderLogs();
  } catch (e) {}
}

function computeComparison(mono, prog) {
  if (!mono || !prog) return null;
  return {
    ttft_reduction_pct: Number(((1 - prog.ttft_full_s / mono.ttft_full_s) * 100).toFixed(1)),
    speedup_x: Number((mono.ttft_full_s / prog.ttft_full_s).toFixed(2)),
    first_layer_speedup_x: Number((mono.first_layer_s / prog.first_layer_s).toFixed(1)),
    masking_pct: Number((prog.masking_ratio * 100).toFixed(1)),
    rss_delta_reduction_pct: Number(((1 - prog.delta_rss_mb / Math.max(mono.delta_rss_mb, 1)) * 100).toFixed(1)),
    rss_delta_saved_mb: Number(Math.max(0, mono.delta_rss_mb - prog.delta_rss_mb).toFixed(1)),
  };
}

/* ---------- Apply Live Status ---------- */
function applyStatus(st, initial) {
  if (!st) return;
  const rt = st.runtime || {};
  document.getElementById("sysRuntime").textContent = "PyTorch " + (rt.torch || "");
  document.getElementById("sysDevice").textContent = (rt.device || "cpu").toUpperCase();
  document.getElementById("topDevice").textContent = (rt.device || "cpu").toUpperCase();
  document.getElementById("metaDevice").textContent = (rt.device || "cpu").toUpperCase() + " (PyTorch " + (rt.torch || "") + ")";
  document.getElementById("sysStore").textContent = rt.storage_backend || "HTTP RangeServer";
  document.getElementById("topStore").textContent = rt.minio_available ? "MinIO S3" : "HTTP Range";
  document.getElementById("metaEndpoint").textContent = rt.minio_available ? rt.minio_endpoint : "127.0.0.1 (RangeServer)";

  document.getElementById("topExpId").textContent = st.experiment_id || "—";
  document.getElementById("liveExpId").textContent = st.experiment_id || "—";
  document.getElementById("livePhase").textContent = st.phase || "ready";
  document.getElementById("liveElapsed").textContent = (st.elapsed_s || 0).toFixed(1) + "s";
  document.getElementById("liveProgressFill").style.width = Math.round((st.progress || 0) * 100) + "%";

  const isRunning = st.status === "running";
  const isError = st.status === "error";
  const dot = document.getElementById("topEngineDot");
  dot.className = "dot " + (isRunning ? "running" : isError ? "red" : "");
  document.getElementById("topEngineText").textContent = isRunning ? "Running" : isError ? "Failed" : "Ready";

  const sysBadge = document.getElementById("sysStatusBadge");
  sysBadge.className = "trend-badge " + (isRunning ? "blue" : isError ? "red" : "green");
  sysBadge.textContent = isRunning ? "Running" : isError ? "Failed" : "Ready";

  const runBadge = document.getElementById("liveRunStateBadge");
  runBadge.className = "trend-badge " + (isRunning ? "blue" : isError ? "red" : st.status === "done" ? "green" : "neutral");
  runBadge.textContent = (st.status || "idle").toUpperCase();

  document.getElementById("btnRunCompare").disabled = isRunning;
  document.getElementById("btnRunProg").disabled = isRunning;
  document.getElementById("btnStopRun").disabled = !isRunning;

  const errBox = document.getElementById("liveErrorBox");
  if (isError && st.error) {
    errBox.style.display = "block";
    errBox.textContent = "Execution failed: " + st.error;
  } else {
    errBox.style.display = "none";
  }

  // Update pipeline indicators
  const k = (st.config && st.config.lookahead) || 2;
  document.getElementById("pipeBufChip").innerHTML = `Buffer: <b>${st.buffer_occupancy || 0} / ${k} slots</b>`;
  document.getElementById("dashPrefetchState").textContent = st.prefetching_layer ? `Fetching ${st.prefetching_layer}` : `K = ${k}`;
  document.getElementById("dashLayerStatus").textContent = st.active_layer ? `Computing ${st.active_layer}` : (st.status === "done" ? "Completed" : "Ready");
  document.getElementById("dashEvictCount").textContent = `Evictions: ${st.eviction_count || 12}`;
  document.getElementById("pipeTrackPrefetch").textContent = st.prefetching_layer || "Idle";
  document.getElementById("pipeTrackCompute").textContent = st.active_layer || "Idle";

  if (st.monolithic) LIVE_MONO = st.monolithic;
  if (st.progressive) LIVE_PROG = st.progressive;
  if (st.comparison) {
    LIVE_COMP = st.comparison;
    document.getElementById("dataSourceTag").textContent = `Live measurement · ${st.experiment_id} (${st.config.bandwidth_mbps} Mbps, K=${st.config.lookahead}, seq=${st.config.seq_len})`;
  }

  renderLayerStrip(st.live_layers || (LIVE_PROG && LIVE_PROG.per_layer) || [], st.active_layer, st.prefetching_layer);
  renderTelemetryTable(st.live_layers && st.live_layers.length ? st.live_layers : ((LIVE_PROG && LIVE_PROG.per_layer) || []));

  if (!initial) {
    renderDashboardAll();
    if (!isRunning && POLL_TIMER) {
      clearInterval(POLL_TIMER);
      POLL_TIMER = null;
      fetchHistory();
      fetchLogs();
      renderAnalyticsCharts();
      showToast(st.status === "done" ? `Experiment ${st.experiment_id} completed.` : "Experiment stopped.");
    }
  }
}

/* ---------- Experiment Actions ---------- */
async function startExperiment(mode) {
  const cfg = {
    model: document.getElementById("cfgModel").value,
    bandwidth_mbps: parseFloat(document.getElementById("cfgBw").value),
    lookahead: parseInt(document.getElementById("cfgK").value, 10),
    seq_len: parseInt(document.getElementById("cfgSeq").value, 10),
    evict: document.getElementById("cfgEvict").value === "true",
    mode,
  };
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  const data = await res.json();
  if (data.started) {
    showToast(`Started ${mode} run (${data.experiment_id})`);
    if (POLL_TIMER) clearInterval(POLL_TIMER);
    POLL_TIMER = setInterval(() => fetchStatus(false), 350);
    fetchStatus(false);
  } else {
    showToast("An experiment is already running.");
  }
}

function quickRun(mode) {
  navigate("live");
  startExperiment(mode);
}

async function stopExperiment() {
  await fetch("/api/stop", { method: "POST" });
  showToast("Stop requested...");
}

/* ---------- Dashboard Renderers ---------- */
function renderDashboardAll() {
  const mono = LIVE_MONO;
  const prog = LIVE_PROG;
  const comp = LIVE_COMP || computeComparison(mono, prog);

  if (comp && mono && prog) {
    document.getElementById("kpiTtftVal").textContent = `${prog.ttft_full_s.toFixed(2)} s`;
    document.getElementById("kpiTtftBadge").textContent = `-${comp.ttft_reduction_pct}% (${comp.speedup_x}x)`;
    document.getElementById("kpiTtftSub").textContent = `Baseline ${mono.ttft_full_s.toFixed(2)}s vs Progressive ${prog.ttft_full_s.toFixed(2)}s`;

    document.getElementById("kpiFirstVal").textContent = `${prog.first_layer_s.toFixed(2)} s`;
    document.getElementById("kpiFirstBadge").textContent = `${comp.first_layer_speedup_x}x faster`;
    document.getElementById("kpiFirstSub").textContent = `Baseline ${mono.first_layer_s.toFixed(2)}s vs Layer 0 ${prog.first_layer_s.toFixed(2)}s`;

    document.getElementById("kpiMaskVal").textContent = `${comp.masking_pct}%`;
    document.getElementById("kpiMaskSub").textContent = `Compute ${prog.compute_s.toFixed(2)}s · Net Wait ${prog.io_wait_s.toFixed(2)}s`;

    document.getElementById("kpiRssVal").textContent = `${prog.delta_rss_mb.toFixed(1)} MB`;
    document.getElementById("kpiRssBadge").textContent = `-${comp.rss_delta_reduction_pct}% ΔRSS`;
    document.getElementById("kpiRssSub").textContent = `Monolithic ${mono.delta_rss_mb.toFixed(1)} MB → Progressive ${prog.delta_rss_mb.toFixed(1)} MB`;
  } else if (prog) {
    document.getElementById("kpiTtftVal").textContent = `${prog.ttft_full_s.toFixed(2)} s`;
    document.getElementById("kpiFirstVal").textContent = `${prog.first_layer_s.toFixed(2)} s`;
    document.getElementById("kpiMaskVal").textContent = `${(prog.masking_ratio * 100).toFixed(1)}%`;
    document.getElementById("kpiRssVal").textContent = `${prog.delta_rss_mb.toFixed(1)} MB`;
  }

  if (prog) {
    const rows = prog.per_layer || [];
    const avgFetch = rows.length ? (rows.reduce((a, r) => a + (r.download_s || 0), 0) / rows.length) : 0;
    document.getElementById("netBw").textContent = `${prog.bandwidth_mbps} Mbps`;
    document.getElementById("netBytes").textContent = `${((prog.bytes_downloaded || 0) / 1048576).toFixed(1)} MiB`;
    document.getElementById("netReqs").textContent = `${rows.length || 12}`;
    document.getElementById("netConns").textContent = `K = ${prog.lookahead || 2}`;
    document.getElementById("netAvgLat").textContent = `${avgFetch.toFixed(3)} s`;
    document.getElementById("netWait").textContent = `${prog.io_wait_s.toFixed(2)} s`;
  }

  renderComparisonChart();
  renderDashRssChart();
  renderDashNetChart();
}

function setCompMetric(metric, btn) {
  COMP_METRIC = metric;
  document.querySelectorAll("#compMetricTabs .tab-btn").forEach(b => b.classList.remove("active"));
  if (btn) btn.classList.add("active");
  renderComparisonChart();
}

function renderComparisonChart() {
  if (!APP_ABLATION || !APP_ABLATION.bandwidth_sweep) return;
  const rows = APP_ABLATION.bandwidth_sweep;
  const labels = rows.map(r => `${r.bandwidth_mbps} Mbps`);
  let datasets = [];
  let yTitle = "Seconds";

  if (COMP_METRIC === "ttft") {
    yTitle = "Cold-Start TTFT (s)";
    datasets = [
      { label: "Monolithic Loading", data: rows.map(r => r.mono_ttft_s), backgroundColor: "#DC2626", borderRadius: 5 },
      { label: "Progressive Streaming", data: rows.map(r => r.prog_ttft_s), backgroundColor: "#2563EB", borderRadius: 5 },
    ];
  } else if (COMP_METRIC === "first") {
    yTitle = "First-Layer Availability (s)";
    datasets = [
      { label: "Monolithic Loading", data: rows.map(r => r.mono_first_layer_s), backgroundColor: "#DC2626", borderRadius: 5 },
      { label: "Progressive Streaming", data: rows.map(r => r.prog_first_layer_s), backgroundColor: "#10B981", borderRadius: 5 },
    ];
  } else if (COMP_METRIC === "rss") {
    yTitle = "Peak Working-Set ΔRSS (MB)";
    datasets = [
      { label: "Monolithic Loading", data: rows.map(r => r.mono_delta_rss_mb), backgroundColor: "#DC2626", borderRadius: 5 },
      { label: "Progressive (Evicting)", data: rows.map(r => r.prog_delta_rss_mb), backgroundColor: "#2563EB", borderRadius: 5 },
    ];
  } else {
    yTitle = "I/O Masking Ratio (%)";
    datasets = [
      { label: "Monolithic Loading", data: rows.map(() => 0), backgroundColor: "#CBD5E1", borderRadius: 5 },
      { label: "Progressive Streaming", data: rows.map(r => Number((r.prog_masking_ratio * 100).toFixed(1))), backgroundColor: "#059669", borderRadius: 5 },
    ];
  }

  upsertChart("chartComparison", {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "top", align: "end" } },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: yTitle }, grid: { color: "#F1F5F9" } },
        x: { grid: { display: false } },
      },
    },
  });
}

function normalizeTrace(obj) {
  if (!obj || !obj.rss_trace) return [];
  const base = obj.baseline_rss_mb || 0;
  return obj.rss_trace.map(p => ({ x: p[0], y: p[1] > base * 0.9 && base > 50 ? Math.max(0, Number((p[1] - base).toFixed(2))) : p[1] }));
}

function renderDashRssChart() {
  const datasets = [];
  if (LIVE_MONO && LIVE_MONO.rss_trace) {
    datasets.push({
      label: `Monolithic (Peak ${LIVE_MONO.delta_rss_mb} MB)`,
      data: normalizeTrace(LIVE_MONO),
      borderColor: "#DC2626",
      backgroundColor: "rgba(220,38,38,0.08)",
      fill: true,
      pointRadius: 0,
      borderWidth: 2,
    });
  }
  if (LIVE_PROG && LIVE_PROG.rss_trace) {
    datasets.push({
      label: `Progressive Evicting (Peak ${LIVE_PROG.delta_rss_mb} MB)`,
      data: normalizeTrace(LIVE_PROG),
      borderColor: "#2563EB",
      backgroundColor: "rgba(37,99,235,0.10)",
      fill: true,
      pointRadius: 0,
      borderWidth: 2,
    });
  }
  const cfg = {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      plugins: { legend: { position: "top", align: "end" } },
      scales: {
        x: { type: "linear", title: { display: true, text: "Elapsed Time (s)" }, grid: { color: "#F1F5F9" } },
        y: { beginAtZero: true, title: { display: true, text: "Working-Set ΔRSS (MB)" }, grid: { color: "#F1F5F9" } },
      },
    },
  };
  upsertChart("chartDashRss", cfg);
  upsertChart("chartMemTrace", JSON.parse(JSON.stringify(cfg)));
}

function renderDashNetChart() {
  const rows = (LIVE_PROG && LIVE_PROG.per_layer) || [];
  if (!rows.length) return;
  upsertChart("chartDashNet", {
    type: "bar",
    data: {
      labels: rows.map(r => r.layer.replace("model.layers.", "L")),
      datasets: [
        { label: "Fetch (s)", data: rows.map(r => r.download_s), backgroundColor: "#3B82F6", borderRadius: 3 },
        { label: "Compute (s)", data: rows.map(r => r.compute_s), backgroundColor: "#10B981", borderRadius: 3 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "top", align: "end" } },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "Seconds" }, grid: { color: "#F1F5F9" } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderLayerStrip(rows, activeLayer, prefetchLayer) {
  const el = document.getElementById("dashLayerStrip");
  if (!el) return;
  const doneSet = new Set(rows.map(r => r.layer));
  let html = "";
  for (let i = 0; i < 12; i++) {
    const name = `model.layers.${i}`;
    let cls = "";
    if (name === activeLayer) cls = "computing";
    else if (name === prefetchLayer) cls = "prefetching";
    else if (doneSet.has(name)) cls = "done";
    html += `<div class="layer-cell ${cls}" title="${name}">L${i}</div>`;
  }
  el.innerHTML = html;
}

function renderTelemetryTable(rows) {
  const tb = document.getElementById("liveTelemetryBody");
  if (!tb) return;
  document.getElementById("liveTableMeta").textContent = `${rows.length} layers recorded`;
  tb.innerHTML = rows.map(r => {
    const saved = Math.max(0, (r.download_s || 0) - (r.blocked_s || 0));
    return `<tr>
      <td class="mono">${r.layer}</td>
      <td class="num">${((r.bytes || 0) / 1048576).toFixed(2)}</td>
      <td class="num">${(r.download_s || 0).toFixed(3)}</td>
      <td class="num">${(r.blocked_s || 0).toFixed(3)}</td>
      <td class="num">${(r.compute_s || 0).toFixed(3)}</td>
      <td class="num" style="color:var(--green);font-weight:600">${saved.toFixed(3)}</td>
      <td class="num">${(r.checksum || 0).toFixed(5)}</td>
    </tr>`;
  }).join("");
}

/* ---------- Analytics & Pipeline Charts ---------- */
function renderAnalyticsCharts() {
  if (!APP_ABLATION) return;
  const bw = APP_ABLATION.bandwidth_sweep || [];
  const la = APP_ABLATION.lookahead_ablation || [];
  const sq = APP_ABLATION.seqlen_scaling || [];

  upsertChart("chartLatBw", {
    type: "bar",
    data: {
      labels: bw.map(r => `${r.bandwidth_mbps} Mbps`),
      datasets: [
        { label: "Monolithic TTFT (s)", data: bw.map(r => r.mono_ttft_s), backgroundColor: "#DC2626", borderRadius: 5 },
        { label: "Progressive TTFT (s)", data: bw.map(r => r.prog_ttft_s), backgroundColor: "#2563EB", borderRadius: 5 },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, title: { display: true, text: "TTFT (s)" } } } },
  });

  upsertChart("chartLatSeq", {
    type: "bar",
    data: {
      labels: sq.map(r => `L = ${r.seq_len}`),
      datasets: [
        { label: "Monolithic TTFT (s)", data: sq.map(r => r.mono_ttft_s), backgroundColor: "#DC2626", borderRadius: 5 },
        { label: "Progressive TTFT (s)", data: sq.map(r => r.prog_ttft_s), backgroundColor: "#2563EB", borderRadius: 5 },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, title: { display: true, text: "TTFT (s)" } } } },
  });

  upsertChart("chartMemLookahead", {
    type: "line",
    data: {
      labels: la.map(r => `K = ${r.lookahead_k}`),
      datasets: [
        { label: "Cold-Start TTFT (s)", data: la.map(r => r.ttft_full_s), borderColor: "#2563EB", yAxisID: "y", tension: 0.2 },
        { label: "Peak ΔRSS (MB)", data: la.map(r => r.delta_rss_mb), borderColor: "#D97706", borderDash: [5, 4], yAxisID: "y1", tension: 0.2 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { position: "left", title: { display: true, text: "TTFT (s)" } },
        y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Peak ΔRSS (MB)" } },
      },
    },
  });

  upsertChart("chartNetMask", {
    type: "line",
    data: {
      labels: bw.map(r => `${r.bandwidth_mbps} Mbps`),
      datasets: [{
        label: "I/O Masking Ratio (%)",
        data: bw.map(r => Number((r.prog_masking_ratio * 100).toFixed(1))),
        borderColor: "#059669",
        backgroundColor: "rgba(5,150,105,0.12)",
        fill: true,
        tension: 0.25,
      }],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: "Masking (%)" } } } },
  });

  const rows = (LIVE_PROG && LIVE_PROG.per_layer) || [];
  if (rows.length) {
    upsertChart("chartNetLayer", {
      type: "bar",
      data: {
        labels: rows.map(r => r.layer.replace("model.layers.", "L")),
        datasets: [
          { label: "Download (s)", data: rows.map(r => r.download_s), backgroundColor: "#3B82F6" },
          { label: "Blocked Wait (s)", data: rows.map(r => r.blocked_s), backgroundColor: "#D97706" },
          { label: "Compute (s)", data: rows.map(r => r.compute_s), backgroundColor: "#10B981" },
        ],
      },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, title: { display: true, text: "Seconds" } } } },
    });

    upsertChart("chartPipelineTimeline", {
      type: "bar",
      data: {
        labels: rows.map(r => r.layer.replace("model.layers.", "Layer ")),
        datasets: [
          { label: "Async Range Fetch [start, end]", data: rows.map(r => [r.fetch_start_s, r.fetch_end_s]), backgroundColor: "#3B82F6", borderRadius: 4 },
          { label: "Transformer Compute [start, end]", data: rows.map(r => [r.compute_start_s, r.compute_end_s]), backgroundColor: "#10B981", borderRadius: 4 },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true, title: { display: true, text: "Wall-Clock Time (seconds)" } } },
      },
    });
  }
}

/* ---------- Models & History Tables ---------- */
function populateModelSelect() {
  const sel = document.getElementById("cfgModel");
  if (!sel) return;
  const runnable = APP_MODELS.filter(m => m.runnable);
  sel.innerHTML = runnable.map(m => `<option value="${m.name}">${m.name} (${m.size_mb} MiB)</option>`).join("");
  updateModelMetaFields();
}

function updateModelMetaFields() {
  const name = document.getElementById("cfgModel").value;
  const m = APP_MODELS.find(x => x.name === name);
  if (!m) return;
  document.getElementById("metaArch").textContent = m.architecture;
  document.getElementById("metaLayers").textContent = `${m.num_layers} blocks`;
  document.getElementById("metaSize").textContent = `${m.size_mb} MiB`;
  document.getElementById("metaDtype").textContent = m.dtype;
}

function renderModelsTable() {
  const q = (document.getElementById("modelSearch").value || "").toLowerCase();
  const rows = APP_MODELS.filter(m => m.name.toLowerCase().includes(q) || m.architecture.toLowerCase().includes(q));
  document.getElementById("modelsTableBody").innerHTML = rows.map(m => `<tr>
    <td class="mono" style="font-weight:600">${m.name}</td>
    <td>${m.architecture}</td>
    <td class="num">${m.num_layers}</td>
    <td class="num">${m.size_mb.toFixed(2)}</td>
    <td class="mono">${m.format} / ${m.dtype}</td>
    <td>${m.storage}</td>
    <td><span class="trend-badge ${m.runnable ? "green" : "neutral"}">${m.runnable ? "Available" : "Benchmark Only"}</span></td>
    <td>${m.runnable ? `<button class="btn btn-sm" onclick="selectModelAndRun('${m.name}')">Select</button>` : "—"}</td>
  </tr>`).join("");
}

function selectModelAndRun(name) {
  document.getElementById("cfgModel").value = name;
  updateModelMetaFields();
  navigate("live");
  showToast(`Selected ${name}`);
}

function renderHistoryTable() {
  const src = document.getElementById("historyFilter").value;
  const q = (document.getElementById("historySearch").value || "").toLowerCase();
  const rows = APP_HISTORY.filter(r => {
    if (src !== "all" && r.source !== src) return false;
    return !q || r.experiment_id.toLowerCase().includes(q) || (r.model || "").toLowerCase().includes(q);
  });
  document.getElementById("historyTableBody").innerHTML = rows.map(r => `<tr>
    <td class="mono" style="font-weight:600">${r.experiment_id}</td>
    <td><span class="trend-badge ${r.source === "live" ? "blue" : "neutral"}">${r.source === "live" ? "Live Run" : "Benchmark"}</span></td>
    <td class="mono">${r.model}</td>
    <td class="num">${r.bandwidth_mbps}</td>
    <td class="num">${r.lookahead}</td>
    <td class="mono">${r.mode}</td>
    <td class="num">${r.prog_ttft_s != null ? r.prog_ttft_s.toFixed(2) : "—"}</td>
    <td class="num">${r.prog_delta_rss_mb != null ? r.prog_delta_rss_mb.toFixed(1) : "—"}</td>
    <td class="mono" style="font-size:11.5px;color:var(--text-secondary)">${r.timestamp}</td>
    <td><span class="trend-badge green">${r.status}</span></td>
  </tr>`).join("");
}

/* ---------- System Logs ---------- */
function renderLogs() {
  const lvl = document.getElementById("logLevelFilter").value;
  const q = (document.getElementById("logSearch").value || "").toLowerCase();
  const rows = APP_LOGS.filter(l => (lvl === "ALL" || l.level === lvl) && (!q || l.message.toLowerCase().includes(q) || l.component.toLowerCase().includes(q)));
  const el = document.getElementById("logContainer");
  if (!el) return;
  el.innerHTML = rows.map(l => `<div class="log-row">
    <span style="color:var(--text-muted)">${l.timestamp}</span>
    <span class="log-lvl ${l.level}">${l.level}</span>
    <span style="color:var(--text-secondary)">${l.component}</span>
    <span style="color:var(--blue)">${l.experiment_id}</span>
    <span>${l.message}</span>
  </div>`).join("") || `<div style="padding:16px;color:var(--text-muted)">No matching log entries.</div>`;
  if (!LOG_PAUSED) el.scrollTop = el.scrollHeight;
}

function toggleLogPause() {
  LOG_PAUSED = !LOG_PAUSED;
  document.getElementById("btnPauseLogs").innerHTML = LOG_PAUSED ? "Resume" : "Pause";
}

function copyLogs() {
  const text = APP_LOGS.map(l => `${l.timestamp} [${l.level}] [${l.component}] [${l.experiment_id}] ${l.message}`).join("\n");
  navigator.clipboard.writeText(text);
  showToast("Logs copied to clipboard.");
}

function downloadLogs() {
  const text = APP_LOGS.map(l => `${l.timestamp} [${l.level}] [${l.component}] [${l.experiment_id}] ${l.message}`).join("\n");
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "progressive_ai_system_logs.log";
  a.click();
}

window.addEventListener("DOMContentLoaded", bootstrap);
