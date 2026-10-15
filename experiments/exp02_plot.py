"""
figures for experiment 02 (H4: UCCSD vs HEA), drawn from the saved CSV.

run (from repo root):  python experiments/exp02_plot.py
input:   results/exp02_h4_results.csv
output:  results/fig_exp02_h4.pdf/.png        error vs spacing, with correlation strength
         results/fig_exp02_h4_cost.pdf/.png   cost of accuracy: CNOT count and training cost
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import plotstyle as ps  # noqa: E402

BASE_NOTE = "H₄ chain, STO-3G, noiseless statevector."


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k == "method":
                continue
            r[k] = None if v == "" else (v == "True") if v in ("True", "False") else float(v)
    return rows


def by_spacing(rows, method, layers=None, key="dE_mHa"):
    """spacings and, per spacing, an array of values (several seeds for HEA)."""
    out = defaultdict(list)
    for r in rows:
        if r["method"] == method and (layers is None or r["layers"] == layers):
            out[r["d_A"]].append(r[key])
    d = np.array(sorted(out))
    return d, [np.array(out[x]) for x in d]


def first(rows, method, key, layers=None):
    return next(r[key] for r in rows
                if r["method"] == method and (layers is None or r["layers"] == layers))


def fig_error_vs_spacing(rows, best_l):
    d, w = by_spacing(rows, "HF", key="hf_weight")
    w = np.array([v[0] for v in w]) * 100
    hf = np.array([v[0] for v in by_spacing(rows, "HF")[1]])
    cc = np.array([v[0] for v in by_spacing(rows, "CCSD")[1]])
    uc = np.array([v[0] for v in by_spacing(rows, "UCCSD")[1]])
    hea = by_spacing(rows, "HEA", best_l)[1]
    med = np.array([np.median(v) for v in hea])
    lo, hi = np.array([v.min() for v in hea]), np.array([v.max() for v in hea])

    fig, (ax1, ax2) = ps.figure(nrows=2, height_ratios=[0.5, 1.45], sharex=True,
                                panel_height=2.7)

    # ---- panel 1: correlation strength ----
    ax1.plot(d, w, color=ps.theme()["ink"], lw=1.6, marker="o", markersize=3.5,
             markerfacecolor=ps.theme()["ink"], markeredgecolor=ps.theme()["surface"])
    ps.panel_title(ax1, "HF weight in the exact state", "%")
    ax1.set_ylim(0, 110)
    ax1.set_yticks([0, 50, 100])
    ps.number_axis(ax1, "y", decimals=0)

    # ---- panel 2: error vs FCI ----
    ps.log_axis(ax2, "y", every=1)
    ax2.set_ylim(1e-3, 2e3)
    ps.threshold(ax2, ps.CHEM_ACC_MHA, "chemical accuracy", above=True)

    ps.plot(ax2, d, hf, "HF")
    ps.plot(ax2, d, np.abs(cc), "CCSD")
    below = cc < 0
    s = ps.style("CCSD")
    ax2.plot(d[below], np.abs(cc[below]), ls="none", marker=s["marker"], markersize=6,
             markerfacecolor=ps.theme()["surface"], markeredgecolor=s["color"],
             markeredgewidth=1.2, zorder=6)
    ps.plot(ax2, d, uc, "UCCSD")
    ax2.vlines(d, lo, hi, color=ps.color("HEA"), lw=1.3, alpha=0.5, zorder=2)
    ps.plot(ax2, d, med, "HEA")

    ps.panel_title(ax2, "Error vs FCI", "mHa, log scale")
    ax2.set_xlabel("Atom spacing d (Å)")
    ax2.set_xticks(d[::2])
    ps.number_axis(ax2, "x", decimals=1)
    ax2.set_xlim(d[0] - 0.08, d[-1] + 0.08)

    ps.header(fig, "Stretching H₄ breaks every approximate method",
              key=["HF", "CCSD", "UCCSD", "HEA"])
    ps.footer(fig, f"{BASE_NOTE} HEA: {best_l} layers, median and range of 5 random starts. "
                   "Hollow diamonds: CCSD below the exact energy.")
    ps.end_labels(ax2, d[-1], [(hf[-1], "HF"), (abs(cc[-1]), "CCSD"),
                               (uc[-1], "UCCSD"), (med[-1], "HEA")])
    ps.save(fig, "results/fig_exp02_h4")


def fig_cost(rows, layers):
    """worst error over the curve against hardware cost (CNOT) and training cost (evaluations)."""
    hea = []
    for L in layers:
        _, vals = by_spacing(rows, "HEA", L)
        _, evals = by_spacing(rows, "HEA", L, key="evaluations")
        hea.append((first(rows, "HEA", "cnot", L), np.median(np.concatenate(evals)),
                    max(np.median(v) for v in vals), L))
    hea = np.array(hea)
    uc_err = max(v[0] for v in by_spacing(rows, "UCCSD")[1])
    uc_cnot = first(rows, "UCCSD", "cnot")
    uc_eval = np.median(np.concatenate(by_spacing(rows, "UCCSD", key="evaluations")[1]))

    fig, (ax1, ax2) = ps.figure(nrows=1, ncols=2, panel_height=2.9, sharey=True)
    fig.subplots_adjust(wspace=0.12, right=0.95)

    for ax, x_hea, x_uc, title, xlim in (
            (ax1, hea[:, 0], uc_cnot, "CNOT gates", (2, 4000)),
            (ax2, hea[:, 1], uc_eval, "Circuit evaluations", (300, 3e6))):
        ps.log_axis(ax, "x", every=1)
        ps.log_axis(ax, "y", every=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(1, 400)
        ps.threshold(ax, ps.CHEM_ACC_MHA, "chemical accuracy")
        ps.plot(ax, x_hea, hea[:, 2], "HEA")
        ps.plot(ax, [x_uc], [uc_err], "UCCSD", markersize=7)
        for x, y, L in zip(x_hea, hea[:, 2], hea[:, 3]):
            ax.annotate(f"{int(L)}", (x, y), xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=6.5, color=ps.theme()["ink_2"])
        ps.end_label(ax, x_uc, uc_err, "UCCSD", dx_pt=8)
        ps.panel_title(ax, title, "log scale")
    ax1.set_ylabel("Worst error on the curve (mHa)")

    ps.header(fig, "UCCSD pays in gates, HEA pays in training", key=["UCCSD", "HEA"])
    ps.footer(fig, f"{BASE_NOTE} HEA: median of 5 random starts; numbers = layers.")
    ps.save(fig, "results/fig_exp02_h4_cost")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/exp02_h4_results.csv")
    args = parser.parse_args()
    ps.use()
    rows = load(args.csv)
    layers = sorted({int(r["layers"]) for r in rows if r["method"] == "HEA"})
    fig_error_vs_spacing(rows, max(layers))
    fig_cost(rows, layers)


if __name__ == "__main__":
    main()
