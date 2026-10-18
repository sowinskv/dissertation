"""
figures for experiment 02 (H4: UCCSD vs HEA), drawn from the saved CSV.

run (from repo root):  python experiments/exp02_plot.py
input:   results/exp02_h4_results.csv, results/exp02_h4_spin_results.csv (exp02_spin_check.py)
output:  results/fig_exp02_h4.pdf/.png        error vs spacing, with correlation strength
         results/fig_exp02_h4_cost.pdf/.png   cost of accuracy: CNOT count and training cost
         results/fig_exp02_h4_spin.pdf/.png   final HEA states: energy error vs total spin
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import plotstyle as ps  # noqa: E402

BASE_NOTE = "H₄ chain, STO-3G, noiseless statevector."


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v == "":
                r[k] = None
            elif v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass   # text columns (method, spin names) stay strings
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

    ps.header(fig, "Error of approximate methods for the linear H₄ chain vs atom spacing",
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

    ps.header(fig, "VQE accuracy vs circuit size and training cost, H₄\u00a0chain", key=["UCCSD", "HEA"])
    ps.footer(fig, f"{BASE_NOTE} HEA: median of 5 random starts; numbers = layers.")
    ps.save(fig, "results/fig_exp02_h4_cost")


def fig_spin(spin_rows, layers):
    """energy error of every final HEA state against its total spin <S^2>."""
    t = ps.theme()
    floor = 1e-4
    hea = [r for r in spin_rows if r["method"] == "HEA"]
    ramp = ps.shades("HEA", len(layers))

    fig, ax = ps.figure(nrows=1, panel_height=3.1)
    ps.log_axis(ax, "y", every=1)
    ax.set_ylim(floor / 2, 600)
    ax.set_xlim(-0.3, 6.6)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.grid(axis="x", visible=False)
    for s2, name in ((0, "singlet"), (2, "triplet"), (6, "quintet")):
        ax.axvline(s2, color=t["muted"], lw=0.7, ls=(0, (1, 2)), zorder=0)
        ax.annotate(name, (s2, 1), xycoords=ax.get_xaxis_transform(), xytext=(4, -3),
                    textcoords="offset points", ha="left", va="top", fontsize=7, color=t["ink_2"])
    ps.threshold(ax, ps.CHEM_ACC_MHA, "chemical accuracy", x=0.42)
    for L, c in zip(layers, ramp):
        sel = [r for r in hea if r["layers"] == L]
        ax.plot([r["S2"] for r in sel], [max(r["dE_mHa"], floor) for r in sel], ls="none",
                marker="o", markersize=4.2, markerfacecolor=c, markeredgecolor=ps.color("HEA"),
                markeredgewidth=0.5, alpha=0.9, zorder=3 + L)
    ax.set_xlabel("Total spin ⟨S²⟩ of the final state")
    ps.panel_title(ax, "Error vs FCI", "mHa, log scale")

    key = [Line2D([], [], ls="none", marker="o", markersize=4.5, markerfacecolor=c,
                  markeredgecolor=ps.color("HEA"), markeredgewidth=0.5, label=f"{L} layer{'s' if L > 1 else ''}")
           for L, c in zip(layers, ramp)]
    ps.header(fig, "Energy error vs total spin of the final HEA states, H₄\u00a0chain", key=key)
    n_q = sum(1 for r in hea if r["is_eigenstate"] and r["spin"] == "quintet")
    ps.footer(fig, f"{BASE_NOTE} All 9 spacings, 5 random starts per depth; states rebuilt from the saved "
                   f"parameters. The {n_q} runs at S² = 6 are exact quintet eigenstates (many overlap). "
                   f"Errors below 10⁻⁴ mHa are drawn at the floor.")
    ps.save(fig, "results/fig_exp02_h4_spin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/exp02_h4_results.csv")
    args = parser.parse_args()
    ps.use()
    rows = load(args.csv)
    layers = sorted({int(r["layers"]) for r in rows if r["method"] == "HEA"})
    fig_error_vs_spacing(rows, max(layers))
    fig_cost(rows, layers)
    spin_csv = Path("results/exp02_h4_spin_results.csv")
    if spin_csv.exists():
        fig_spin(load(spin_csv), layers)


if __name__ == "__main__":
    main()
