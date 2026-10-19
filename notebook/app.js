/* lab notebook UI: reads the experiment logs served by tools/notebook.py */
"use strict";

const COLORS = {           // same pens as src/plotstyle.py METHODS
  UCCSD: "#1f4fd1", HEA: "#e0482a", CCSD: "#0f9a7a", HF: "#8a8a8a", FCI: "#000000",
  "QNN-cond": "#ad7400", MLP: "#c4398c", GP: "#5a5fd6",
};
const FALLBACK = ["#1f4fd1", "#e0482a", "#0f9a7a", "#ad7400"];
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
  chart: null, timer: null, histories: {} };

/* ---------- helpers ---------- */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// H2, H4, LiH ... with subscript digits in titles
const chem = (s) => esc(s).replace(/\b(H|Li|Be|N|O|C)(\d)\b/g, "$1<sub>$2</sub>");

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function fmtDate(iso) {
  if (!iso) return "";
  const [y, m, d] = iso.split("T")[0].split("-");
  return `${Number(d)} ${MONTHS[Number(m) - 1]} ${y}`;
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
  const m = location.hash.match(/^#\/exp\/([^/]+)(?:\/(\w+))?$/);   // #/exp/<name>[/<section>]
  if (m) showExperiment(decodeURIComponent(m[1]), m[2]);
  else { state.name = null; showIndex(); }
}

/* ---------- landing page: every logged experiment, newest number last ---------- */
function runsLine(x) {
  const done = x.counts.done || 0, total = x.n_runs || x.n_logged;
  const runs = x.status === "running"
    ? `<span class="bad">running, ${done} of ${total} runs done</span>` : `${done} runs`;
  return `${fmtDate(x.started_at)}, ${runs}`;
}

function entryHTML(x) {
  const href = `#/exp/${encodeURIComponent(x.name)}`, fig = x.figures?.[0];
  const links = [`[<a href="${href}">open</a>]`,
    fig && `[<a href="${href}/figures">figures</a>]`,
    x.csv && `[<a href="/results/${encodeURIComponent(x.csv)}" download>CSV</a>]`].filter(Boolean).join(" ");
  return `<article class="entry" data-name="${esc(x.name)}">
      <a class="thumb" href="${href}" tabindex="-1" aria-hidden="true">
        <img src="photos/${encodeURIComponent(x.name)}.jpg" alt="" loading="lazy"
             data-fallback="${fig ? `/results/${encodeURIComponent(fig)}` : ""}"></a>
      <div class="entry-text">
        <h2><a href="${href}"><b>Exp. ${esc(x.label)}:</b> ${chem(x.title)}</a></h2>
        <p class="meta" data-live>${runsLine(x)}</p>
        <p class="desc">${chem(x.description)}</p>
        <p class="links">${links}</p>
      </div>
    </article>`;
}

let shownIndex = "";
async function showIndex(isPoll = false) {
  let list;
  try { list = await getJSON("/api/experiments"); }
  catch (e) { return renderError(e); }
  document.title = "Lab notebook";
  const signature = list.map((x) => `${x.name}:${x.status}`).join("|");
  if (isPoll && signature === shownIndex) {
    // only the run counts change while training: update those lines, keep the page still
    list.forEach((x) => {
      const el = app.querySelector(`.entry[data-name="${CSS.escape(x.name)}"] [data-live]`);
      if (el) el.innerHTML = runsLine(x);
    });
  } else {
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    if (!isPoll) window.scrollTo(0, 0);
    app.innerHTML = `
      <header class="masthead">
        <h1>Lab notebook</h1>
        <p>Quantum neural networks for molecular simulation</p>
        <p class="by">J. Sowińska</p>
      </header>
      ${list.length ? `<div class="entries">${list.map(entryHTML).join("")}</div>`
        : `<p class="error">Nothing logged yet. Run an experiment script from the repo root, for example
           <code>uv run python experiments/exp01_h2_curve.py</code>, and it will appear here.</p>`}
      <footer class="foot"><p>You just scrolled through <span id="px">0</span> pixels.</p></footer>`;
    app.querySelectorAll(".thumb img").forEach((img) => img.addEventListener("error", () => {
      const fb = img.dataset.fallback;
      if (fb && img.getAttribute("src") !== fb) { img.src = fb; img.parentElement.classList.add("is-figure"); }
      else img.parentElement.remove();
    }));
    shownIndex = signature;
    countPixels();
  }
  if (list.some((x) => x.status === "running")) poll(() => showIndex(true));
}

function renderError(e) {
  app.innerHTML = `<p class="error">The notebook server is not answering (${esc(e.message)}).
    Start it from the repo root with <code>uv run python tools/notebook.py</code> and reload this page.</p>`;
}

/* ---------- experiment page ---------- */
async function showExperiment(name, section) {
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

  if (fresh || !document.getElementById("runs")) {
    renderShell(exp);
    window.scrollTo(0, 0);
    if (section) requestAnimationFrame(() => document.getElementById(section)?.scrollIntoView());
  }
  syncFilters(exp);
  renderMeta(exp);
  renderTable();
  await renderChart();
  countPixels();

  if (exp.status === "running") poll(() => showExperiment(name, section));
}

function renderShell(exp) {
  const setup = Object.entries(exp.config || {}).map(([k, v]) =>
    `<b>${esc(k.replaceAll("_", " "))}:</b> ${esc(Array.isArray(v) ? v.join(", ") : typeof v === "object" ? JSON.stringify(v) : v)}`).join("<br>");
  const figures = (exp.figures || []).map((f) => `
    <figure>
      <a href="/results/${encodeURIComponent(f)}" target="_blank" rel="noopener"><img src="/results/${encodeURIComponent(f)}" alt="Figure from results/${esc(f)}" loading="lazy"></a>
      <figcaption>results/${esc(f)} [<a href="/results/${encodeURIComponent(f.replace(/\.png$/, ".pdf"))}" target="_blank" rel="noopener">pdf</a>]</figcaption>
    </figure>`).join("");
  const links = [
    figures && `[<a href="#figures" data-jump>figures</a>]`,
    `[<a href="#curves" data-jump>training curves</a>]`,
    `[<a href="#runs-title" data-jump>runs</a>]`,
    exp.csv && `[<a href="/results/${encodeURIComponent(exp.csv)}" download>download CSV</a>]`,
  ].filter(Boolean).join(" ");

  app.innerHTML = `
    <nav class="top">[<a href="#/">back to the notebook</a>]</nav>
    <article class="exp">
      <p class="stamp" id="stamp" hidden></p>
      <h1><b>Exp. ${esc(exp.label)}:</b> ${chem(exp.title)}</h1>
      <p class="meta" id="meta"></p>
      <p class="links">${links}</p>
      <p class="desc">${chem(exp.description)}</p>
      ${setup ? `<p class="setup">${setup}</p>` : ""}
      ${figures ? `<h2 id="figures">Figures</h2>${figures}` : ""}
      <h2 id="curves">Training curves</h2>
      <p class="hint">Error against the exact (FCI) energy at every optimizer iteration. Tick what to show.</p>
      <div class="filters" id="filters"></div>
      <div class="chart-box"><canvas id="chart" aria-label="Training curves"></canvas></div>
      <p class="hint" id="chart-note"></p>
      <h2 id="runs-title">Runs</h2>
      <p class="hint" id="runs-note"></p>
      <div class="table-wrap"><table id="runs"></table></div>
    </article>
    <footer class="foot">
      <p>You just scrolled through <span id="px">0</span> pixels.</p>
      <p>[<a href="#/">back to the notebook</a>]</p>
    </footer>`;
  // in-page links must not touch the router's hash
  app.querySelectorAll("[data-jump]").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    document.getElementById(a.getAttribute("href").slice(1))?.scrollIntoView({ behavior: "smooth" });
  }));
}

function renderMeta(exp) {
  const c = exp.counts, total = exp.n_runs || exp.runs.length;
  const parts = [fmtDate(exp.started_at)];
  if (exp.status === "running") parts.push(`running, ${c.done || 0} of ${total} runs done`);
  else parts.push(`${c.done || 0} runs${exp.finished_at ? `, finished ${fmtTime(exp.finished_at)}` : ""}`);
  if (c.failed) parts.push(`<span class="bad">${c.failed} failed</span>`);
  document.getElementById("meta").innerHTML = parts.join(", ");
  const stamp = document.getElementById("stamp");
  stamp.hidden = exp.status !== "running";
  stamp.innerHTML = `Still running<br>${c.done || 0} of ${total}`;
}

function countPixels() {
  const px = document.getElementById("px");
  if (px) px.textContent = document.documentElement.scrollHeight.toLocaleString("en-US");
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
      <b>${esc(LABELS[key] || key)}:</b>
      ${f.values.map((v) => `<label><input type="checkbox" data-key="${esc(key)}"
        data-value="${esc(JSON.stringify(v))}" ${f.selected.has(v) ? "checked" : ""}>${key === "method"
        ? `<span class="swatch" style="background:${colorFor(v)}"></span>` : ""}${esc(fmtNum(v))}</label>`).join("")}
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
    `${rows.length} of ${exp.runs.length} runs match the ticks above. Click a row to highlight its curve; click a column name to sort.`;
  const head = cols.map((c) => {
    const sort = state.sort?.key === c.key ? (state.sort.dir > 0 ? "ascending" : "descending") : "none";
    return `<th scope="col" aria-sort="${sort}"><button type="button" data-key="${esc(c.key)}">${esc(LABELS[c.key] || c.key)}</button></th>`;
  }).join("");
  const body = rows.map((r) => {
    const tds = cols.map((c) => {
      const v = cell(r, c);
      if (c.key === "method") return `<td class="method"><span class="swatch" style="background:${colorFor(v)}"></span>${esc(v)}</td>`;
      if (c.key === "dE_mHa" && r.status === "running") return `<td class="running">running…</td>`;
      const bad = (c.key === "converged" && v === false) || (c.key === "dE_mHa" && typeof v === "number" && Math.abs(v) > CHEM_ACC_MHA);
      return `<td class="${bad ? "bad" : ""}">${esc(fmtNum(v))}</td>`;
    }).join("");
    const cls = [state.pinned.has(r.run_id) && "pinned", r.status === "failed" && "failed"].filter(Boolean).join(" ");
    return `<tr data-id="${esc(r.run_id)}" class="${cls}" title="${esc(r.run_id)}" tabindex="0">${tds}</tr>`;
  }).join("");
  table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;

  table.querySelectorAll("th button").forEach((b) => b.addEventListener("click", () => {
    const key = b.dataset.key;
    state.sort = state.sort?.key === key ? { key, dir: -state.sort.dir } : { key, dir: 1 };
    renderTable();
  }));
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
    ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);
    ctx.beginPath(); ctx.moveTo(a.left, py); ctx.lineTo(a.right, py); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = "#000"; ctx.font = '12px "Helvetica Neue", Helvetica, Arimo, Arial, sans-serif';
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

const isDecade = (v) => { const e = Math.log10(v); return Math.abs(e - Math.round(e)) < 1e-9; };
const logTicks = (v) => (isDecade(v) ? pow10(Math.round(Math.log10(v))) : "");
const decadeGrid = (ctx) => (isDecade(ctx.tick?.value ?? 0) ? "#e3e3e3" : "transparent");

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
    : `${shown.length} run${shown.length > 1 ? "s" : ""} shown.`;

  if (state.chart) {
    state.chart.data.datasets = datasets;
    state.chart.update("none");
    return;
  }
  await document.fonts.ready;
  Chart.defaults.font.family = '"Helvetica Neue", Helvetica, Arimo, Arial, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = "#000";
  state.chart = new Chart(document.getElementById("chart"), {
    type: "line",
    data: { datasets },
    plugins: [threshold],
    options: {
      animation: false, maintainAspectRatio: false, normalized: true,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: { type: "logarithmic", title: { display: true, text: "optimizer iteration + 1" },
             ticks: { callback: logTicks, autoSkip: false, maxRotation: 0 }, grid: { color: decadeGrid }, border: { color: "#000" } },
        y: { type: "logarithmic", title: { display: true, text: "error (mHa)" },
             ticks: { callback: logTicks, autoSkip: false }, grid: { color: decadeGrid }, border: { color: "#000" } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#fff", borderColor: "#000", borderWidth: 1, cornerRadius: 0,
          titleColor: "#000", bodyColor: "#000", padding: 8,
          titleFont: { weight: "700" },
          callbacks: {
            title: (items) => items[0]?.dataset.label ?? "",
            label: (it) => `iteration ${it.parsed.x - 1}: ${fmtNum(it.parsed.y)} mHa`,
          },
        },
      },
    },
  });
}
