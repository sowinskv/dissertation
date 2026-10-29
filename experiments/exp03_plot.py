"""
figures for experiment 03 (one conditional QNN for the H4 curve), drawn from the saved CSV.

run (from repo root):  python experiments/exp03_plot.py
input:   results/exp03_h4_conditional_results.csv, results/exp03_h4_conditional_summary.csv,
         results/exp03_h4_reference.csv, results/exp03_h4_baselines_*.csv (exp03_baselines.py)
output:  results/fig_exp03_h4_curve.pdf/.png  error along the chain: conditional QNN vs UCCSD per point
         results/fig_exp03_h4_grid.pdf/.png   worst error inside / outside the training range,
                                              by encoder degree and number of training spacings
         results/fig_exp03_h4_baselines.pdf/.png  RQ3: QNN (no labels) vs GP / MLP on FCI labels
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import plotstyle as ps  # noqa: E402
from exp02_plot import load  # noqa: E402

BASE_NOTE = "H₄ chain, STO-3G, noiseless statevector, UCCSD circuit (26 angles)."
LO, HI = 0.5, 2.5
FLOOR = 1e-3


def curves(rows, degree, m):
    """spacings and the error of every seed at each spacing: (n_d,), (n_d, n_seeds)."""
    out = defaultdict(list)
    for r in rows:
        if r["degree"] == degree and r["n_train"] == m:
            out[r["d_A"]].append(r["dE_mHa"])
    d = np.array(sorted(out))
    return d, np.array([out[x] for x in d])


def shade_extrapolation(ax, x0, x1):
    t = ps.theme()
    for a, b in ((x0, LO), (HI, x1)):
        ax.axvspan(a, b, color=t["grid"], alpha=0.55, lw=0, zorder=0)


def fig_curve(rows, ref, m=5, degrees=(1, 3)):
    t = ps.theme()
    d_ref = np.array([r["d_A"] for r in ref])
    hf = np.array([r["dE_HF_mHa"] for r in ref])
    uc = np.maximum(np.array([r["dE_UCCSD_mHa"] for r in ref]), FLOOR)
    train = np.linspace(LO, HI, m)
    colors = dict(zip(degrees, ps.shades("QNN-cond", len(degrees), lightest=0.55)))

    fig, ax = ps.figure(nrows=1, panel_height=3.2)
    ps.log_axis(ax, "y", every=1)
    ax.set_ylim(FLOOR / 2, 2e3)
    x0, x1 = d_ref[0] - 0.05, d_ref[-1] + 0.05
    shade_extrapolation(ax, x0, x1)
    ps.threshold(ax, ps.CHEM_ACC_MHA, "chemical accuracy", above=True, x=0.2)
    ps.plot(ax, d_ref, hf, "HF")
    ps.plot(ax, d_ref, uc, "UCCSD", markevery=4)
    labels = [(hf[-1], "HF"), (uc[-1], "UCCSD per point")]
    s = ps.style("QNN-cond")
    for K in degrees:
        d, err = curves(rows, K, m)
        err = np.maximum(err, FLOOR)
        med = np.median(err, axis=1)
        ax.fill_between(d, err.min(axis=1), err.max(axis=1), color=colors[K], alpha=0.18, lw=0, zorder=2)
        ax.plot(d, med, color=colors[K], lw=1.6, zorder=4)
        on_train = np.isin(np.round(d, 2), np.round(train, 2))
        ax.plot(d[on_train], med[on_train], ls="none", marker=s["marker"], markersize=6,
                color=colors[K], markeredgecolor=t["surface"], markeredgewidth=0.8, zorder=5)
        labels.append((med[-1], f"QNN, K = {K}"))
    ps.panel_title(ax, "Error vs FCI", "mHa, log scale")
    ax.set_xlabel("Atom spacing d (Å)")
    ax.set_xticks(np.arange(0.5, 3.01, 0.5))
    ps.number_axis(ax, "x", decimals=1)
    ax.set_xlim(x0, x1)
    ps.end_labels(ax, d_ref[-1], labels)

    key = ["HF", Line2D([], [], **{k: v for k, v in ps.style("UCCSD").items() if k in ("color", "ls", "marker")},
                        markersize=4.5, lw=1.6, markeredgecolor=t["surface"], label="VQE-UCCSD per point")]
    for K in degrees:
        key.append(Line2D([], [], color=colors[K], lw=1.6,
                          label=f"conditional QNN, {'linear' if K == 1 else f'degree {K}'} encoder"))
    key.append(Line2D([], [], color=colors[degrees[-1]], ls="none", marker=s["marker"], markersize=6,
                      markeredgecolor=t["surface"], label="training spacings"))
    ps.header(fig, f"Error along the H₄ chain: one conditional QNN vs UCCSD trained at every spacing",
              key=key, key_cols=3)
    ps.footer(fig, f"{BASE_NOTE} Conditional QNN: angles from a Chebyshev encoder of d, trained on the "
                   f"mean energy of {m} spacings (no FCI labels), K = encoder degree; line = median of 5 initialisations, "
                   "band = all 5. UCCSD per point: best of 3 starts. Gray: outside the training range. "
                   "Errors below 10⁻³ mHa are drawn at the floor.")
    ps.save(fig, "results/fig_exp03_h4_curve")


def fig_grid(summary, ref):
    t = ps.theme()
    degrees = sorted({int(r["degree"]) for r in summary})
    ms = sorted({int(r["n_train"]) for r in summary})
    colors = dict(zip(ms, ps.shades("QNN-cond", len(ms), lightest=0.5)))
    inside = np.array([r["dE_UCCSD_mHa"] for r in ref if LO - 1e-9 <= r["d_A"] <= HI + 1e-9])
    outside = np.array([r["dE_UCCSD_mHa"] for r in ref if r["d_A"] > HI + 1e-9])
    s = ps.style("QNN-cond")

    fig, axes = ps.figure(nrows=1, ncols=2, panel_height=2.8)
    panels = [(axes[0], "interp_max_dE_mHa", "Inside the training range", "0.5–2.5 Å, worst error, mHa",
               inside.max()),
              (axes[1], "extrap_long_max_dE_mHa", "Beyond the training range", "2.55–3.0 Å, worst error, mHa",
               outside.max())]
    x = np.arange(len(degrees))
    for ax, col, title, sub, uccsd in panels:
        ps.log_axis(ax, "y", every=1)
        ps.threshold(ax, ps.CHEM_ACC_MHA, "chemical accuracy", above=True)
        ax.axhline(uccsd, color=ps.color("UCCSD"), lw=1.2, ls=(0, (4, 2)), zorder=2)
        ends = [(uccsd, "UCCSD per point")]
        for j, m in enumerate(ms):
            vals = np.array([[r[col] for r in summary if r["degree"] == K and r["n_train"] == m]
                             for K in degrees])
            off = (j - (len(ms) - 1) / 2) * 0.12
            med = np.median(vals, axis=1)
            ax.vlines(x + off, vals.min(axis=1), vals.max(axis=1), color=colors[m], lw=1.3, alpha=0.6, zorder=3)
            ax.plot(x + off, med, color=colors[m], marker=s["marker"], markersize=5,
                    markeredgecolor=t["surface"], markeredgewidth=0.8, lw=1.3, zorder=4)
            ends.append((med[-1], f"M = {m}"))
        ax.set_xticks(x, [str(K) for K in degrees])
        ax.set_xlim(-0.4, len(degrees) - 0.6)
        ax.set_xlabel("Encoder degree K")
        ps.panel_title(ax, title, sub)
        ps.end_labels(ax, x[-1] + 0.2, ends)
    lo = min(min(a.get_ylim()[0] for a, *_ in panels), 0.5)
    hi = max(a.get_ylim()[1] for a, *_ in panels)
    for a, *_ in panels:
        a.set_ylim(lo, hi)
    fig.subplots_adjust(wspace=0.55)

    key = [Line2D([], [], color=colors[m], marker=s["marker"], markersize=5, lw=1.3,
                  markeredgecolor=t["surface"], label=f"{m} training spacings") for m in ms]
    key.append(Line2D([], [], color=ps.color("UCCSD"), lw=1.2, ls=(0, (4, 2)), label="VQE-UCCSD per point"))
    ps.header(fig, "Worst error of the conditional QNN on the H₄ chain by encoder degree and training set size",
              key=key)
    ps.footer(fig, f"{BASE_NOTE} K = 1 is the linear encoder of Meta-VQE. Markers: median of 5 initialisations, bars: all 5. Test spacings every "
                   "0.05 Å. Dashed: worst error of UCCSD trained separately at every spacing of the same range.")
    ps.save(fig, "results/fig_exp03_h4_grid")


def baseline_curves(rows, method, target, m):
    out = defaultdict(list)
    for r in rows:
        if r["method"] == method and r["target"] == target and r["n_train"] == m:
            out[r["d_A"]].append(abs(r["dE_mHa"]))
    d = np.array(sorted(out))
    return d, np.array([out[x] for x in d])


def fig_baselines(rows, base_rows, base_summary, ref, m=5, degree=3):
    """RQ3: the conditional QNN (energy only) against regression on FCI labels at the same spacings."""
    t = ps.theme()
    d_ref = np.array([r["d_A"] for r in ref])
    uc = np.maximum(np.array([r["dE_UCCSD_mHa"] for r in ref]), FLOOR)
    fig, (ax, ax2) = ps.figure(nrows=2, panel_height=2.6, hspace_in=ps.PANEL_TITLE_IN + 0.75)

    # top: error along the chain, all models trained on the same m spacings
    ps.log_axis(ax, "y", every=1)
    ax.set_ylim(FLOOR / 2, 2e3)
    x0, x1 = d_ref[0] - 0.05, d_ref[-1] + 0.05
    shade_extrapolation(ax, x0, x1)
    ps.threshold(ax, ps.CHEM_ACC_MHA, "chemical accuracy", above=True, x=0.2)
    ax.plot(d_ref, uc, color=ps.color("UCCSD"), lw=1.1, ls=(0, (4, 2)), zorder=3)
    labels = [(uc[-1], "UCCSD per point")]
    series = [("QNN-cond", curves(rows, degree, m), "QNN (no labels)")]
    for meth in ("GP", "MLP"):
        series.append((meth, baseline_curves(base_rows, meth, "E-HF", m), f"{meth} (FCI labels)"))
    for meth, (d, err), lab in series:
        err = np.maximum(np.abs(err), FLOOR)
        med = np.median(err, axis=1)
        if err.shape[1] > 1:
            ax.fill_between(d, err.min(axis=1), err.max(axis=1), color=ps.color(meth), alpha=0.18, lw=0, zorder=2)
        ps.plot(ax, d, med, meth, markevery=4, zorder=4)
        labels.append((med[-1], lab))
    ps.panel_title(ax, f"Error vs FCI, trained on {m} spacings", "mHa, log scale")
    ax.set_xlabel("Atom spacing d (Å)")
    ax.set_xticks(np.arange(0.5, 3.01, 0.5))
    ps.number_axis(ax, "x", decimals=1)
    ax.set_xlim(x0, x1)
    ps.end_labels(ax, d_ref[-1], labels)

    # bottom: worst error inside the training range vs number of training spacings
    ms = sorted({int(r["n_train"]) for r in base_summary})
    ps.log_axis(ax2, "y", every=1)
    ps.threshold(ax2, ps.CHEM_ACC_MHA, "chemical accuracy", above=True, x=0.6)
    qnn = load("results/exp03_h4_conditional_summary.csv")
    ends = []
    pts = [("QNN-cond", [[r["interp_max_dE_mHa"] for r in qnn if r["degree"] == degree and r["n_train"] == k]
                         for k in ms], "QNN")]
    for meth in ("GP", "MLP"):
        for target, ls, lab in (("E-HF", "-", f"{meth}, correlation energy"), ("E", ":", f"{meth}, total energy")):
            pts.append((meth, [[r["interp_max_abs_dE_mHa"] for r in base_summary
                                if r["method"] == meth and r["target"] == target and r["n_train"] == k]
                               for k in ms], lab, ls))
    for item in pts:
        meth, vals, lab = item[:3]
        ls = item[3] if len(item) > 3 else "-"
        med = np.array([np.median(v) for v in vals])
        ax2.vlines(ms, [min(v) for v in vals], [max(v) for v in vals], color=ps.color(meth), lw=1.2, alpha=0.5)
        ps.plot(ax2, ms, med, meth, ls=ls, zorder=4)
        ends.append((med[-1], lab))
    ax2.set_xticks(ms)
    ax2.set_xlim(ms[0] - 0.4, ms[-1] + 0.4)
    ax2.set_xlabel("Training spacings M (labels for GP and MLP)")
    ps.panel_title(ax2, "Worst error inside 0.5–2.5 Å", "mHa, log scale")
    ps.end_labels(ax2, ms[-1], ends)

    key = [Line2D([], [], color=ps.color("QNN-cond"), marker=ps.style("QNN-cond")["marker"], lw=1.6,
                  markersize=4.5, markeredgecolor=t["surface"], label=f"conditional QNN (degree {degree}), no labels")]
    for meth in ("GP", "MLP"):
        s = ps.style(meth)
        key.append(Line2D([], [], color=s["color"], marker=s["marker"], lw=1.6, markersize=4.5,
                          markeredgecolor=t["surface"], label=f"{s['label']}, FCI labels"))
    key.append(Line2D([], [], color=ps.color("UCCSD"), lw=1.1, ls=(0, (4, 2)), label="VQE-UCCSD per point"))
    ps.header(fig, "Conditional QNN without labels vs regression on exact energies, H₄ chain", key=key, key_cols=2)
    ps.footer(fig, f"{BASE_NOTE} GP and MLP fit the correlation energy (FCI minus HF) at the training "
                   "spacings, HF is computed at every test spacing; dotted, bottom: fitted to the total FCI "
                   "energy instead. Lines: median of 5 initialisations (GP: one fit), bands and bars: all 5. "
                   "Gray: outside the training range. Errors below 10⁻³ mHa are drawn at the floor.")
    ps.save(fig, "results/fig_exp03_h4_baselines")


def main():
    ps.use()
    rows = load("results/exp03_h4_conditional_results.csv")
    summary = load("results/exp03_h4_conditional_summary.csv")
    ref = load("results/exp03_h4_reference.csv")
    fig_curve(rows, ref)
    fig_grid(summary, ref)
    if Path("results/exp03_h4_baselines_results.csv").exists():
        fig_baselines(rows, load("results/exp03_h4_baselines_results.csv"),
                      load("results/exp03_h4_baselines_summary.csv"), ref)


if __name__ == "__main__":
    main()
