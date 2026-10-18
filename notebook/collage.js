/* landing page: one pasted-up notebook sheet, a photo per experiment (layout: collage.json) */
"use strict";

const Collage = (() => {
  const SHEET_W = 1200, SHEET_H = 1000;
  const SVGNS = "http://www.w3.org/2000/svg";
  const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

  /* ---------- pen strokes ---------- */
  // seeded random numbers: the "handwriting" wobble is the same on every load
  function rng(seed) {
    let s = (seed * 2654435761) >>> 0 || 1;
    return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  }
  const pt = (p) => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`;

  // smooth curve through the points (catmull-rom as cubic bezier)
  function smooth(pts) {
    let d = `M${pt(pts[0])}`;
    if (pts.length === 2) return `${d}L${pt(pts[1])}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
      const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      d += `C${pt(c1)} ${pt(c2)} ${pt(p2)}`;
    }
    return d;
  }

  // a straight-ish pen line: a few points with a slight sag and tremor
  function line(a, b, rand, wobble = 1.2) {
    const n = 4, pts = [];
    const nx = -(b[1] - a[1]), ny = b[0] - a[0], len = Math.hypot(nx, ny) || 1;
    const sag = (rand() - 0.5) * wobble * 2;
    for (let i = 0; i <= n; i++) {
      const t = i / n, bow = Math.sin(Math.PI * t) * sag + (i && i < n ? (rand() - 0.5) * wobble : 0);
      pts.push([a[0] + (b[0] - a[0]) * t + (nx / len) * bow, a[1] + (b[1] - a[1]) * t + (ny / len) * bow]);
    }
    return smooth(pts);
  }

  // a loop drawn in one go, overshooting its start like a quick pen circle
  function loop(cx, cy, rx, ry, rand, turns = 1.12, steps = 26) {
    const a0 = -2.4 + rand() * 0.6, pts = [];
    for (let i = 0; i <= steps; i++) {
      const a = a0 + (i / steps) * turns * 2 * Math.PI, k = 1 + (rand() - 0.5) * 0.06 + (i / steps) * 0.05;
      pts.push([cx + Math.cos(a) * rx * k, cy + Math.sin(a) * ry * k]);
    }
    return smooth(pts);
  }

  function arrowPaths(points, rand) {
    const pts = points.map((p, i) => (i && i < points.length - 1 ? [p[0] + (rand() - 0.5) * 3, p[1] + (rand() - 0.5) * 3] : p));
    const end = pts[pts.length - 1], prev = pts[pts.length - 2];
    const ang = Math.atan2(end[1] - prev[1], end[0] - prev[0]), L = 13;
    const wing = (s) => [end[0] - L * Math.cos(ang + s * 0.5), end[1] - L * Math.sin(ang + s * 0.5)];
    return [smooth(pts), `M${pt(wing(1))}L${pt(end)}L${pt(wing(-1))}`];
  }

  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  const ink = (g, d, cls = "") => el("path", { d, class: `ink ${cls}`.trim() }, g);
  function label(g, x, y, text, size = 18, anchor = "start") {
    const t = el("text", { x, y, "font-size": size, "text-anchor": anchor, class: "hand-svg" }, g);
    t.textContent = text;
    return t;
  }

  // overlays for notes: scribbled out, circled, underlined (drawn in a 100 x 100 box, stretched)
  function overlay(kind, seed) {
    const rand = rng(seed);
    const svg = el("svg", { viewBox: "0 0 100 100", preserveAspectRatio: "none", class: `mark mark-${kind}`, "aria-hidden": "true" });
    if (kind === "strike") {
      const pts = [];
      for (let x = -1, up = true; x <= 101; x += 2.6 + rand() * 1.8, up = !up) pts.push([x, up ? 22 + rand() * 12 : 70 + rand() * 12]);
      ink(svg, smooth(pts));
    } else if (kind === "circle") {
      ink(svg, loop(50, 50, 47, 44, rand));
    } else {
      ink(svg, smooth([[0, 40 + rand() * 20], [35, 55], [70, 45 + rand() * 15], [100, 30]]));
    }
    return svg;
  }

  /* ---------- sketches ---------- */
  const SKETCH = {
    // potential energy curve: E(R) = De (1 - exp(-a (R - Re)))^2 - De
    curve(g, w, h, rand) {
      const ox = 22, oy = h - 26, top = 10, right = w - 8;
      ink(g, line([ox, oy], [ox, top], rand, 1));
      ink(g, `M${ox - 5} ${top + 8}L${ox} ${top}L${ox + 5} ${top + 8}`);
      ink(g, line([ox, oy], [right, oy], rand, 1));
      ink(g, `M${right - 8} ${oy - 5}L${right} ${oy}L${right - 8} ${oy + 5}`);
      label(g, ox - 16, top + 14, "E", 22);
      label(g, right - 4, oy + 22, "R", 22, "end");
      const zero = top + 52, depth = 74, re = 0.3, a = 3.3, pts = [];
      for (let i = 0; i <= 26; i++) {
        const x = 0.05 + i / 26 * 0.95, e = (1 - Math.exp(-a * (x - re))) ** 2 - 1;
        const y = zero - e * depth;
        if (y > top + 4) pts.push([ox + x * (right - ox - 10), y + (rand() - 0.5) * 1.2]);
      }
      ink(g, smooth(pts));
      el("path", { d: `M${ox + 52} ${zero}L${right - 6} ${zero}`, class: "ink dashed" }, g);
      label(g, right - 6, zero - 7, "H + H", 17, "end");
      const minX = ox + re * (right - ox - 10);
      el("circle", { cx: minX, cy: zero + depth, r: 3.2, class: "dot" }, g);
      label(g, minX + 6, zero + depth + 19, "R eq", 16);
    },
    // H4 chain: four atoms, spacing d
    chain(g, w, h, rand) {
      const r = 13, cy = 22, gap = (w - 2 * r - 60) / 3;
      for (let i = 0; i < 4; i++) {
        const cx = 50 + r + i * gap;
        ink(g, loop(cx, cy, r, r, rand, 1.05, 18));
        label(g, cx, cy + 6, "H", 17, "middle");
      }
      const y = cy + r + 14, x1 = 50 + r, x2 = x1 + gap;
      ink(g, line([x1, y], [x2, y], rand, 0.8));
      ink(g, `M${x1} ${y - 5}L${x1} ${y + 5}M${x2} ${y - 5}L${x2} ${y + 5}`);
      label(g, (x1 + x2) / 2, y + 19, "d", 20, "middle");
      label(g, 0, cy + 7, "H₄", 22);
    },
    // one hardware-efficient layer: rotations on every qubit, then a ladder of CNOTs
    circuit(g, w, h, rand) {
      const n = 4, top = 14, step = (h - 50) / (n - 1), x0 = 8, x1 = w - 8;
      const ys = [...Array(n)].map((_, i) => top + i * step);
      ys.forEach((y) => ink(g, line([x0, y], [x1, y], rand, 0.6)));
      const box = (cx, y) => ink(g, line([cx - 9, y - 9], [cx + 9, y - 9], rand, 0.4) + line([cx + 9, y - 9], [cx + 9, y + 9], rand, 0.4).replace("M", "L") + line([cx + 9, y + 9], [cx - 9, y + 9], rand, 0.4).replace("M", "L") + "Z", "box");
      ys.forEach((y) => box(x0 + 26, y));
      for (let i = 0; i < n - 1; i++) {
        const cx = x0 + 60 + i * 22, a = ys[i], b = ys[i + 1];
        el("circle", { cx, cy: a, r: 3.2, class: "dot" }, g);
        ink(g, loop(cx, b, 7, 7, rand, 1.02, 14));
        ink(g, `M${cx} ${a}L${cx} ${b + 7}M${cx - 7} ${b}L${cx + 7} ${b}`);
      }
      ys.forEach((y) => box(x1 - 22, y));
    },
  };

  /* ---------- live text ---------- */
  function fmtDate(iso) {
    if (!iso) return "";
    const [, m, d] = iso.split("T")[0].split("-");
    return `${Number(d)} ${MONTHS[Number(m) - 1]}`;
  }
  function runsText(x) {
    if (!x) return "NOT RUN YET";
    const done = x.counts.done || 0, total = x.n_runs || x.n_logged;
    if (x.status === "running") return `<span class="live">RUNNING, ${done} OF ${total}</span>`;
    return `${done} RUNS`;
  }
  const fill = (tpl, x) => tpl.replace("{date}", x ? fmtDate(x.started_at) : "").replace("{runs}", runsText(x));

  /* ---------- building the sheet ---------- */
  function place(node, it, i) {
    const s = node.style;
    s.setProperty("--x", `${(it.x / SHEET_W) * 100}%`);
    s.setProperty("--y", `${(it.y / SHEET_H) * 100}%`);
    if (it.w) s.setProperty("--w", `${(it.w / SHEET_W) * 100}%`);
    s.setProperty("--r", `${it.r || 0}deg`);
    s.setProperty("--d", it.d ?? 0.5);
    s.setProperty("--i", i);
    if (it.size) s.setProperty("--fs", it.size);
    if (it.w) s.setProperty("--sw", it.w);
    if (it.exp) node.dataset.exp = it.exp;
  }

  function photo(it, x) {
    const a = document.createElement(x ? "a" : "div");
    a.className = "item photo";
    if (x) { a.href = `#/exp/${encodeURIComponent(it.exp)}`; a.setAttribute("aria-label", `${x.title}: open experiment`); }
    const inner = document.createElement("span");
    inner.className = "inner";
    const img = document.createElement("img");
    img.alt = "";
    img.src = `photos/${it.exp}.jpg`;
    img.addEventListener("error", () => {           // no photo yet: show the experiment's first figure
      const fig = x?.figures?.[0];
      if (fig && !img.dataset.fallback) { img.dataset.fallback = "1"; a.classList.add("is-figure"); img.src = `/results/${fig}`; }
      else img.remove();
    });
    inner.appendChild(img);
    a.appendChild(inner);
    return a;
  }

  function text(it, x, i, live) {
    const node = document.createElement(it.exp && x && /“/.test(it.html) ? "a" : "div");
    node.className = `item note${it.tone ? ` tone-${it.tone}` : ""}`;
    if (node.tagName === "A") node.href = `#/exp/${encodeURIComponent(it.exp)}`;
    const inner = document.createElement("span");
    inner.className = "inner";
    const body = document.createElement("span");
    body.className = "words";
    body.innerHTML = fill(it.html, x);
    if (/\{(date|runs)\}/.test(it.html)) live.push({ node: body, tpl: it.html, exp: it.exp });
    inner.appendChild(body);
    for (const kind of ["strike", "circle", "underline"]) if (it[kind]) inner.appendChild(overlay(kind, i + 7));
    node.appendChild(inner);
    return node;
  }

  function planList(it, live) {
    const node = document.createElement("div");
    node.className = "item note plan";
    const inner = document.createElement("span");
    inner.className = "inner";
    const h = document.createElement("span");
    h.className = "words plan-title";
    h.textContent = it.title;
    h.appendChild(overlay("underline", 3));
    const ol = document.createElement("ol");
    it.steps.forEach((s, k) => {
      const li = document.createElement("li");
      if (s.exp) li.dataset.exp = s.exp;
      li.innerHTML = s.html;
      const tick = el("svg", { viewBox: "0 0 24 20", class: "tick", "aria-hidden": "true" });
      ink(tick, smooth([[3, 11], [9, 17], [21, 2]]));
      li.appendChild(tick);
      const status = () => {
        const x = current[s.exp];
        li.classList.toggle("done", x?.status === "done");
        li.classList.toggle("running", x?.status === "running");
      };
      status();
      if (s.exp) live.push({ update: status });
      ol.appendChild(li);
    });
    inner.append(h, ol);
    node.appendChild(inner);
    return node;
  }

  function sketch(it, i) {
    const node = document.createElement("div");
    node.className = "item sketch";
    node.style.setProperty("--h", `${(it.h / SHEET_H) * 100}%`);
    const inner = document.createElement("span");
    inner.className = "inner";
    const svg = el("svg", { viewBox: `0 0 ${it.w} ${it.h}`, "aria-hidden": "true" }, inner);
    SKETCH[it.kind](el("g", {}, svg), it.w, it.h, rng(i * 31 + 5));
    if (it.label) {
      const cap = document.createElement("span");
      cap.className = "words sketch-label";
      cap.textContent = it.label;
      inner.appendChild(cap);
    }
    node.appendChild(inner);
    return node;
  }

  function frame(it) {
    const node = document.createElement("div");
    node.className = "item frame tone-pencil";
    node.style.setProperty("--h", `${(it.h / SHEET_H) * 100}%`);
    node.style.setProperty("--fs", 26);
    node.innerHTML = `<span class="inner"><span class="words">${it.html}</span></span>`;
    return node;
  }

  /* ---------- interaction ---------- */
  function interact(sheet) {
    const groups = (exp) => sheet.querySelectorAll(`[data-exp="${CSS.escape(exp)}"]`);
    let focused = null, leaving = 0;
    const light = (exp) => {
      clearTimeout(leaving);
      if (focused === exp) return;              // moving between parts of the same experiment
      if (focused) dimNow();
      focused = exp;
      sheet.classList.add("focus");
      groups(exp).forEach((n) => {
        n.classList.add("lit");
        n.querySelectorAll?.("path.ink").forEach((p) => { p.classList.remove("redraw"); void p.getBoundingClientRect(); p.classList.add("redraw"); });
      });
    };
    const dimNow = () => { focused = null; sheet.classList.remove("focus"); sheet.querySelectorAll(".lit").forEach((n) => n.classList.remove("lit")); };
    const dim = () => { clearTimeout(leaving); leaving = setTimeout(dimNow, 120); };
    sheet.querySelectorAll(".item[data-exp]").forEach((n) => {
      n.addEventListener("pointerenter", () => light(n.dataset.exp));
      n.addEventListener("pointerleave", dim);
      n.addEventListener("focusin", () => light(n.dataset.exp));
      n.addEventListener("focusout", dim);
    });

    // parallax: things lying higher on the sheet follow the pointer a little more
    if (!matchMedia("(pointer: fine)").matches || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let frameReq = 0, mx = 0, my = 0;
    sheet.addEventListener("pointermove", (e) => {
      const r = sheet.getBoundingClientRect();
      mx = ((e.clientX - r.left) / r.width - 0.5) * 2;
      my = ((e.clientY - r.top) / r.height - 0.5) * 2;
      if (!frameReq) frameReq = requestAnimationFrame(() => {
        frameReq = 0;
        sheet.style.setProperty("--mx", mx.toFixed(3));
        sheet.style.setProperty("--my", my.toFixed(3));
      });
    });
    sheet.addEventListener("pointerleave", () => { sheet.style.setProperty("--mx", 0); sheet.style.setProperty("--my", 0); });
  }

  /* ---------- public ---------- */
  let layout = null, live = [], built = false, current = {};

  async function show(app, list) {
    if (!layout) layout = await (await fetch("collage.json", { cache: "no-store" })).json();
    const byName = Object.fromEntries(list.map((x) => [x.name, x]));
    current = byName;
    live = [];
    const sheet = document.createElement("div");
    sheet.className = "sheet enter";
    const strokes = el("svg", { viewBox: `0 0 ${SHEET_W} ${SHEET_H}`, class: "strokes", "aria-hidden": "true" });
    sheet.appendChild(strokes);

    let n = 0;
    layout.items.forEach((it, i) => {
      const x = it.exp ? byName[it.exp] : null;
      if (it.type === "arrow") {
        const g = el("g", { class: "item arrow" }, strokes);
        place(g, it, n++);
        arrowPaths(it.points, rng(i + 11)).forEach((d) => ink(g, d));
        return;
      }
      const node = it.type === "photo" ? photo(it, x)
        : it.type === "text" ? text(it, x, i, live)
        : it.type === "plan" ? planList(it, live)
        : it.type === "sketch" ? sketch(it, i)
        : it.type === "frame" ? frame(it) : null;
      if (!node) return;
      place(node, it, n++);
      sheet.appendChild(node);
    });

    const unknown = list.filter((x) => !layout.items.some((it) => it.exp === x.name));
    const foot = document.createElement("p");
    foot.className = "sheet-foot";
    foot.innerHTML = "Click a photo to open the experiment." + (unknown.length
      ? ` Not on the page yet: ${unknown.map((x) => `<a href="#/exp/${encodeURIComponent(x.name)}">[${x.name}]</a>`).join(" ")} (add it to notebook/collage.json).`
      : "");

    app.replaceChildren(sheet, foot);
    // stroke lengths for the pen animation
    sheet.querySelectorAll(".arrow path.ink, .sketch path.ink").forEach((p) => {
      p.style.setProperty("--len", Math.ceil(p.getTotalLength() + 2));
    });
    interact(sheet);
    setTimeout(() => sheet.classList.remove("enter"), 2600);
    built = true;
  }

  // during polling only the dates, run counts and ticks change; nothing is redrawn
  function update(list) {
    if (!built) return;
    const byName = Object.fromEntries(list.map((x) => [x.name, x]));
    current = byName;
    live.forEach((l) => {
      if (l.update) { l.update(); return; }
      l.node.innerHTML = fill(l.tpl, byName[l.exp]);
    });
  }

  return { show, update, reset: () => { built = false; } };
})();
