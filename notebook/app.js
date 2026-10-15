/* lab notebook UI: reads the experiment logs served by tools/notebook.py */
"use strict";

const COLORS = {           // same identity colors as src/plotstyle.py METHODS
  UCCSD: "#2a78d6", HEA: "#eb6834", CCSD: "#1baf7a", HF: "#8a867d", FCI: "#141414",
  "QNN-cond": "#eda100", MLP: "#e87ba4", GP: "#008300",
};
const FALLBACK = ["#6250d6", "#e34948", "#2a78d6", "#eb6834"];
const POLL_MS = 3000;
const MAX_CURVES = 60;
const CHEM_ACC_MHA = 1.6;
const LABELS = {
  dE_mHa: "error (mHa)", converged: "converged", iterations: "iterations",
  evaluations: "evaluations", duration_s: "time (s)", fidelity: "fidelity", N: "⟨N⟩",
  params: "parameters", cnot: "CNOT", depth: "depth", hf_weight: "HF weight",
  E_Ha: "energy (Ha)", d_A: "d (Å)", R_A: "R (Å)", method: "method", layers: "layers",
  seed: "seed",
};
const METRIC_ORDER = ["dE_mHa", "converged", "iterations", "evaluations", "duration_s",
  "fidelity", "N", "params", "cnot", "depth", "hf_weight", "E_Ha"];

const app = document.getElementById("app");
const state = { name: null, exp: null, filters: null, pinned: new Set(), sort: null,
  chart: null, timer: null, histories: {}, figuresShown: false };

/* ---------- helpers ---------- */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function fmtDate(iso) {
  if (!iso) return "";
  const [d, t] = iso.split("T");
  const [, m, day] = d.split("-");
  return `${Number(day)} ${MONTHS[Number(m) - 1]}`;
}
const fmtTime = (iso) => (iso ? iso.split("T")[1].slice(0, 5) : "");

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const pow10 = (e) => (e === 0 ? "1" : e === 1 ? "10" : "10" + String(e).split("").map((c) => SUP[c]).join(""));

function fmtNum(v) {
  if (v === null || v === undefined || v === "") return "–";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v !== "number") return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (Number.isInteger(v)) return a >= 10000 ? v.toLocaleString("en-US") : String(v);
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential(1).split("e");
    return `${m}·${pow10(Number(e))}`;
  }
  return String(Number(v.toPrecision(4)));
}

function colorFor(method) {
  if (COLORS[method]) return COLORS[method];
  const i = [...String(method)].reduce((s, c) => s + c.charCodeAt(0), 0) % FALLBACK.length;
  return FALLBACK[i];
}
function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function stopPolling() { clearTimeout(state.timer); state.timer = null; }
function poll(fn) { stopPolling(); state.timer = setTimeout(fn, POLL_MS); }

/* ---------- routing ---------- */
window.addEventListener("hashchange", route);
document.addEventListener("DOMContentLoaded", route);

function route() {
  stopPolling();
  const m = location.hash.match(/^#\/exp\/(.+)$/);
  if (m) showExperiment(decodeURIComponent(m[1]));
  else showIndex();
}

/* ---------- index ---------- */
async function showIndex(isPoll = false) {
  let list;
  try { list = await getJSON("/api/experiments"); }
  catch (e) { return renderError(e); }
  document.title = "Lab notebook";
  const entries = list.map((x) => {
    const total = x.n_runs || x.n_logged;
    const done = x.counts.done || 0;
    const running = x.status === "running";
    const status = running
      ? `<span>running</span><br><span class="muted">${done} of ${total}</span>`
      : `<span class="muted">${done} runs</span>`;
    return `<li class="entry">
      <div class="margin-note">${esc(fmtDate(x.started_at))}<br>${status}</div>
      <h2><a href="#/exp/${encodeURIComponent(x.name)}">${esc(x.title)}</a></h2>
      <p>${esc(x.description)}</p>
      ${running ? `<span class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="${total}" aria-valuenow="${done}"><span style="width:${(100 * done / total).toFixed(1)}%"></span></span>` : ""}
    </li>`;
  }).join("");
  app.innerHTML = `
    <h1 class="book-title">Lab notebook</h1>
    <p class="book-subtitle hand">Quantum neural networks for molecular simulation</p>
    ${list.length ? `<ol class="toc">${entries}</ol>`
      : `<p class="empty">Nothing logged yet. Run an experiment script from the repo root, for example
         <code>python experiments/exp01_h2_curve.py</code>, and it will appear here.</p>`}`;
  if (list.some((x) => x.status === "running")) poll(() => showIndex(true));
}

function renderError(e) {
  app.innerHTML = `<p class="empty">Could not reach the notebook server (${esc(e.message)}).
    Start it from the repo root with <code>python tools/notebook.py</code> and reload this page.</p>`;
}

/* ---------- experiment page ---------- */
async function showExperiment(name) {
  let exp;
  try { exp = await getJSON(`/api/experiments/${encodeURIComponent(name)}`); }
  catch (e) { return renderError(e); }

  const fresh = state.name !== name;
  if (fresh) {
    state.name = name; state.pinned = new Set(); state.histories = {};
    state.filters = null; state.sort = null;
    if (state.chart) { state.chart.destroy(); state.chart = null; }
  }
  state.exp = exp;
  document.title = `${exp.title} – Lab notebook`;

  if (fresh || !document.getElementById("runs")) renderExperimentShell(exp);
  syncFilters(exp);
  renderMargin(exp);
  renderTable();
  await renderChart();

  if (exp.status === "running") poll(() => showExperiment(name));
}

function renderExperimentShell(exp) {
  const setup = Object.entries(exp.config || {}).map(([k, v]) =>
    `<dt>${esc(k.replaceAll("_", " "))}</dt><dd>${esc(Array.isArray(v) ? v.join(", ") : typeof v === "object" ? JSON.stringify(v) : v)}</dd>`).join("");
  const figures = (exp.figures || []).map((f) =>
    `<figure class="pasted"><a href="/results/${encodeURIComponent(f)}" target="_blank" rel="noopener">
      <img src="/results/${encodeURIComponent(f)}" alt="Figure ${esc(f)}" loading="lazy"></a>
      <figcaption class="pencil">results/${esc(f)}</figcaption></figure>`).join("");
  app.innerHTML = `
    <a class="back" href="#/">All experiments</a>
    <section class="entry">
      <div class="margin-note" id="margin"></div>
      <h1 class="exp-title">${esc(exp.title)}</h1>
      <p>${esc(exp.description)}</p>
    </section>
    ${setup ? `<h2 class="section-title">Setup</h2><dl class="setup">${setup}</dl>` : ""}
    ${figures ? `<h2 class="section-title">Figures</h2><div class="figures">${figures}</div>` : ""}
    <h2 class="section-title">Training curves</h2>
    <div class="filters" id="filters"></div>
    <div class="pasted"><div class="chart-box"><canvas id="chart" aria-label="Training curves"></canvas></div></div>
    <p class="chart-note pencil" id="chart-note"></p>
    <h2 class="section-title">Runs</h2>
    <p class="pencil" id="runs-note"></p>
    <div class="table-wrap"><table id="runs"></table></div>`;
}

function renderMargin(exp) {
  const c = exp.counts, total = exp.n_runs || exp.runs.length;
  const lines = [fmtDate(exp.started_at)];
  if (exp.status === "running") {
    lines.push("running", `<span class="muted">${c.done || 0} of ${total}</span>`, `<span class="muted">live</span>`);
  } else {
    lines.push(`<span class="muted">${c.done || 0} runs</span>`);
    if (exp.finished_at) lines.push(`<span class="muted">done ${fmtTime(exp.finished_at)}</span>`);
  }
  if (c.failed) lines.push(`<span class="bad">${c.failed} failed</span>`);
  document.getElementById("margin").innerHTML = lines.join("<br>");
}

/* ---------- filters ---------- */
function paramKeys(runs) {
  const keys = [];
  runs.forEach((r) => Object.keys(r.params || {}).forEach((k) => { if (!keys.includes(k)) keys.push(k); }));
  return keys;
}
function paramValues(runs, key) {
  const vals = [...new Set(runs.map((r) => r.params?.[key]).filter((v) => v !== null && v !== undefined))];
  return vals.sort((a, b) => (typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b))));
}

function syncFilters(exp) {
  const keys = paramKeys(exp.runs);
  const prev = state.filters || {};
  const next = {};
  let changed = !state.filters;
  for (const key of keys) {
    const values = paramValues(exp.runs, key);
    if (prev[key]) {
      const known = prev[key].values;
      const sel = new Set(prev[key].selected);
      values.filter((v) => !known.includes(v)).forEach((v) => { sel.add(v); changed = true; });
      next[key] = { values, selected: sel };
    } else {
      const numeric = values.every((v) => typeof v === "number");
      const one = numeric && values.length > 6 && key !== "seed";   // geometry scans: start with one point
      next[key] = { values, selected: new Set(one ? [values[values.length - 1]] : values) };
      changed = true;
    }
  }
  state.filters = next;
  if (changed) renderFilters();
}

function renderFilters() {
  const el = document.getElementById("filters");
  el.innerHTML = Object.entries(state.filters).filter(([, f]) => f.values.length > 1).map(([key, f]) => `
    <div class="filter" role="group" aria-label="${esc(LABELS[key] || key)}">
      <span class="filter-name">${esc(LABELS[key] || key)}</span>
      ${f.values.map((v) => `<label class="tick"><input type="checkbox" data-key="${esc(key)}"
        data-value="${esc(JSON.stringify(v))}" ${f.selected.has(v) ? "checked" : ""}>
        ${key === "method" ? `<span class="swatch" style="background:${colorFor(v)}"></span>` : ""}${esc(fmtNum(v))}</label>`).join("")}
    </div>`).join("");
  el.querySelectorAll("input").forEach((inp) => inp.addEventListener("change", () => {
    const f = state.filters[inp.dataset.key], v = JSON.parse(inp.dataset.value);
    if (inp.checked) f.selected.add(v); else f.selected.delete(v);
    renderTable(); renderChart();
  }));
}

function passes(run) {
  return Object.entries(state.filters).every(([key, f]) => {
    const v = run.params?.[key];
    return v === null || v === undefined || f.selected.has(v);
  });
}

/* ---------- runs table ---------- */
function columns(runs) {
  const metricKeys = new Set();
  runs.forEach((r) => Object.keys(r.metrics || {}).forEach((k) => metricKeys.add(k)));
  const metrics = METRIC_ORDER.filter((k) => metricKeys.has(k));
  [...metricKeys].sort().forEach((k) => { if (!metrics.includes(k) && k !== "theta") metrics.push(k); });
  return [...paramKeys(runs).map((k) => ({ key: k, param: true })), ...metrics.map((k) => ({ key: k, param: false }))];
}
const cell = (r, c) => (c.param ? r.params?.[c.key] : r.metrics?.[c.key]);

function renderTable() {
  const exp = state.exp, table = document.getElementById("runs");
  const cols = columns(exp.runs);
  let rows = exp.runs.filter(passes);
  if (state.sort) {
    const c = cols.find((x) => x.key === state.sort.key);
    rows = rows.slice().sort((a, b) => {
      const x = cell(a, c), y = cell(b, c);
      if (x === y) return 0;
      if (x === undefined || x === null) return 1;
      if (y === undefined || y === null) return -1;
      return (x > y ? 1 : -1) * state.sort.dir;
    });
  }
  document.getElementById("runs-note").textContent =
    `${rows.length} of ${exp.runs.length} runs match the ticks above. Click a run to pin its curve.`;
  const head = cols.map((c) => {
    const sort = state.sort?.key === c.key ? (state.sort.dir > 0 ? "ascending" : "descending") : "none";
    return `<th scope="col" data-key="${esc(c.key)}" aria-sort="${sort}" tabindex="0">${esc(LABELS[c.key] || c.key)}</th>`;
  }).join("");
  const body = rows.map((r) => {
    const tds = cols.map((c) => {
      const v = cell(r, c);
      if (c.key === "method") return `<td class="method"><span class="dot" style="background:${colorFor(v)}"></span>${esc(v)}</td>`;
      if (c.key === "dE_mHa" && r.status === "running") return `<td class="status-running">running…</td>`;
      const bad = (c.key === "converged" && v === false) || (c.key === "dE_mHa" && typeof v === "number" && Math.abs(v) > CHEM_ACC_MHA);
      return `<td class="${bad ? "bad" : ""}">${esc(fmtNum(v))}</td>`;
    }).join("");
    return `<tr data-id="${esc(r.run_id)}" class="${state.pinned.has(r.run_id) ? "pinned" : ""}" title="${esc(r.run_id)}" tabindex="0">${tds}</tr>`;
  }).join("");
  table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;

  table.querySelectorAll("th").forEach((th) => {
    const sortBy = () => {
      const key = th.dataset.key;
      state.sort = state.sort?.key === key ? { key, dir: -state.sort.dir } : { key, dir: 1 };
      renderTable();
    };
    th.addEventListener("click", sortBy);
    th.addEventListener("keydown", (e) => { if (e.key === "Enter") sortBy(); });
  });
  table.querySelectorAll("tbody tr").forEach((tr) => {
    const toggle = () => {
      const id = tr.dataset.id;
      if (state.pinned.has(id)) state.pinned.delete(id); else state.pinned.add(id);
      tr.classList.toggle("pinned");
      renderChart();
    };
    tr.addEventListener("click", toggle);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
  });
}

/* ---------- training curves ---------- */
const threshold = {
  id: "threshold",
  afterDraw(chart) {
    const y = chart.scales.y;
    if (!y || CHEM_ACC_MHA < y.min || CHEM_ACC_MHA > y.max) return;
    const { ctx, chartArea: a } = chart, py = y.getPixelForValue(CHEM_ACC_MHA);
    ctx.save();
    ctx.strokeStyle = "#2b241e"; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);
    ctx.beginPath(); ctx.moveTo(a.left, py); ctx.lineTo(a.right, py); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = "#2846a6"; ctx.font = "20px Caveat, cursive";
    ctx.fillText("chemical accuracy", a.left + 8, py - 6);
    ctx.restore();
  },
};

function plottedRuns() {
  const runs = state.exp.runs.filter((r) => r.has_history && (passes(r) || state.pinned.has(r.run_id)));
  runs.sort((a, b) => Number(state.pinned.has(b.run_id)) - Number(state.pinned.has(a.run_id)));
  return { shown: runs.slice(0, MAX_CURVES), total: runs.length };
}

async function loadHistories(runs) {
  const need = runs.filter((r) => {
    const h = state.histories[r.run_id];
    return !h || h.status === "running" || h.status !== r.status;
  });
  for (let i = 0; i < need.length; i += 40) {
    const chunk = need.slice(i, i + 40);
    const ids = chunk.map((r) => encodeURIComponent(r.run_id)).join(",");
    const data = await getJSON(`/api/experiments/${encodeURIComponent(state.name)}/history?ids=${ids}`);
    chunk.forEach((r) => { state.histories[r.run_id] = { points: data[r.run_id] || [], status: r.status }; });
  }
}

function curve(run) {
  const ref = run.reference;
  return (state.histories[run.run_id]?.points || []).map((p) => ({
    x: p.step + 1,
    y: Math.max(Math.abs(ref === null || ref === undefined ? p.energy : (p.energy - ref) * 1000), 1e-12),
  }));
}

function logTicks(value) {
  const e = Math.log10(value);
  return Math.abs(e - Math.round(e)) < 1e-9 ? pow10(Math.round(e)) : "";
}

function decadeGrid(ctx) {
  const e = Math.log10(ctx.tick?.value ?? 0);
  return Math.abs(e - Math.round(e)) < 1e-9 ? "rgba(180,196,214,0.9)" : "transparent";
}

async function renderChart() {
  const { shown, total } = plottedRuns();
  try { await loadHistories(shown); } catch (e) { /* keep the last drawn curves */ }
  const datasets = shown.map((r) => {
    const pinned = state.pinned.has(r.run_id);
    const c = colorFor(r.params?.method);
    return {
      label: r.run_id, data: curve(r),
      borderColor: rgba(c, pinned ? 1 : 0.55), backgroundColor: c,
      borderWidth: pinned ? 2.6 : 1.3, pointRadius: 0, pointHoverRadius: 4, tension: 0,
    };
  });
  const note = document.getElementById("chart-note");
  note.textContent = !shown.length ? "No training history for the ticked runs (classical methods are not trained)."
    : total > shown.length ? `Showing ${shown.length} of ${total} matching runs; untick some values to see the rest.`
    : `${shown.length} runs. Error is measured against the exact (FCI) energy.`;

  if (state.chart) {
    state.chart.data.datasets = datasets;
    state.chart.update("none");
    return;
  }
  await document.fonts.ready;
  Chart.defaults.font.family = '"Cormorant Garamond", Georgia, serif';
  Chart.defaults.font.size = 15;
  Chart.defaults.color = "#77746d";
  state.chart = new Chart(document.getElementById("chart"), {
    type: "line",
    data: { datasets },
    plugins: [threshold],
    options: {
      animation: false, maintainAspectRatio: false, normalized: true,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: { type: "logarithmic", title: { display: true, text: "optimizer iteration + 1" },
             ticks: { callback: logTicks, autoSkip: false }, grid: { color: decadeGrid } },
        y: { type: "logarithmic", title: { display: true, text: "error (mHa)" },
             ticks: { callback: logTicks, autoSkip: false }, grid: { color: decadeGrid } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#fefefc", borderColor: "#2b241e", borderWidth: 1,
          titleColor: "#2b241e", bodyColor: "#2b241e", padding: 10,
          titleFont: { family: '"Cormorant Garamond", serif', size: 16, weight: "600" },
          bodyFont: { family: '"Cormorant Garamond", serif', size: 15 },
          callbacks: {
            title: (items) => items[0]?.dataset.label ?? "",
            label: (it) => `iteration ${it.parsed.x - 1}: ${fmtNum(it.parsed.y)} mHa`,
          },
        },
      },
    },
  });
}
