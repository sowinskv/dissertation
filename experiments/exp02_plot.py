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

BASE_NOTE = "Łańcuch H₄, baza STO-3G, symulator wektora stanu (bez szumu), 9 odległości."


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
    n_conv_fail = sum(1 for r in rows if r["method"] == "HEA" and r["layers"] == best_l
                      and not r["converged"])

    fig, (ax1, ax2) = ps.figure(nrows=2, height_ratios=[0.5, 1.45], sharex=True,
                                panel_height=2.7)

    # ---- panel 1: correlation strength ----
    ax1.plot(d, w, color=ps.theme()["ink"], lw=1.6, marker="o", markersize=3.5,
             markerfacecolor=ps.theme()["ink"], markeredgecolor=ps.theme()["surface"])
    ps.panel_title(ax1, "Siła korelacji",
                   "udział konfiguracji Hartree–Focka w dokładnym stanie [%]")
    ax1.set_ylim(0, 110)
    ax1.set_yticks([0, 50, 100])
    ps.number_axis(ax1, "y", decimals=0)
    ps.callout(ax1, (d[0], w[0]), f"{ps._num(round(w[0]))}%: jedna konfiguracja\nwystarcza",
               xytext=(0.85, 35))
    ps.callout(ax1, (d[-1], w[-1]), f"{ps._num(round(w[-1]))}%: silna korelacja",
               xytext=(1.9, 12))

    # ---- panel 2: error vs FCI ----
    ps.log_axis(ax2, "y", every=1)
    ax2.set_ylim(1e-3, 2e3)
    ps.threshold(ax2, ps.CHEM_ACC_MHA, "dokładność chemiczna: błąd < 1,6 mHa ↓", x=0.01, above=True)

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

    ps.panel_title(ax2, "Błąd względem FCI", "|E − E_FCI| [mHa], skala logarytmiczna")
    ax2.set_xlabel("Odległość między sąsiednimi atomami d [Å]")
    ax2.set_xticks(d[::2])
    ps.number_axis(ax2, "x", decimals=1)
    ax2.set_xlim(d[0] - 0.08, d[-1] + 0.08)

    i_cc = int(np.argmax(below))   # first spacing where CCSD goes below FCI
    ps.callout(ax2, (d[i_cc], abs(cc[i_cc])),
               "puste romby: CCSD poniżej energii\ndokładnej (metoda niewariacyjna)",
               xytext=(1.55, 1.2e-2))
    i_worst = int(np.argmax(hi))
    ps.callout(ax2, (d[i_worst], hi[i_worst]),
               f"najgorszy start HEA:\n{ps._num(round(hi[i_worst], 1))} mHa",
               xytext=(1.3, 5e2), ha="right")

    ps.header(fig, "Po rozciągnięciu H₄ żadna z przybliżonych metod nie utrzymuje "
                   "dokładności chemicznej",
              "HF i CCSD mylą się o dziesiątki i setki mHa, a CCSD schodzi poniżej energii "
              f"dokładnej. Błąd UCCSD rośnie do {ps._num(round(uc.max(), 1))} mHa; wynik HEA "
              f"({best_l} warstw) zależy od losowego startu.",
              key=["HF", "CCSD", "UCCSD", "HEA"])
    ps.footer(fig, f"{BASE_NOTE} HEA: {best_l} warstw, 5 losowych startów na punkt; linia = "
                   "mediana, pionowy odcinek = najlepszy i najgorszy start. "
                   f"{n_conv_fail} z {5 * len(d)} treningów HEA osiągnęło limit 10 000 iteracji.")
    ps.end_labels(ax2, d[-1], [(hf[-1], "Hartree–Fock"), (abs(cc[-1]), "CCSD"),
                               (uc[-1], "VQE-UCCSD"), (med[-1], f"VQE-HEA ({best_l} w.)")])
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

    for ax, x_hea, x_uc, title, sub, xlim in (
            (ax1, hea[:, 0], uc_cnot, "Koszt sprzętowy",
             "bramki CNOT w obwodzie (skala log.)", (2, 4000)),
            (ax2, hea[:, 1], uc_eval, "Koszt treningu",
             "ewaluacje obwodu do zbieżności, mediana (skala log.)", (300, 3e6))):
        ps.log_axis(ax, "x", every=1)
        ps.log_axis(ax, "y", every=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(1, 400)
        ps.threshold(ax, ps.CHEM_ACC_MHA, "↓ dokładność chemiczna", above=False)
        ps.plot(ax, x_hea, hea[:, 2], "HEA")
        ps.plot(ax, [x_uc], [uc_err], "UCCSD", markersize=7)
        for x, y, L in zip(x_hea, hea[:, 2], hea[:, 3]):
            ax.annotate(f"{int(L)}", (x, y), xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=6.5, color=ps.theme()["ink_2"])
        ps.panel_title(ax, title, sub)
    ax1.set_ylabel("największy błąd na krzywej [mHa]")
    ps.callout(ax1, (uc_cnot, uc_err), f"UCCSD:\n{int(uc_cnot)} CNOT", xytext=(260, 30),
               ha="center")
    ps.callout(ax1, (hea[0, 0], hea[0, 2]), "HEA, liczby =\nliczba warstw",
               xytext=(25, 200), ha="left")
    ratio = hea[-1, 1] / uc_eval
    ps.callout(ax2, (hea[-1, 1], hea[-1, 2]),
               f"{len(layers)} warstw: ok. {ps._num(round(ratio, -1))}×\nwięcej ewaluacji niż UCCSD",
               xytext=(2.5e4, 18), ha="center")
    ps.end_label(ax2, uc_eval, uc_err, "UCCSD", dx_pt=8)

    ps.header(fig, "Ten sam błąd, inny rachunek: UCCSD płaci bramkami, HEA treningiem",
              f"Na najtrudniejszej geometrii oba ansatze mylą się o ok. 5 mHa. UCCSD potrzebuje "
              f"{int(uc_cnot)} bramek CNOT, HEA ({len(layers)} warstw) tylko "
              f"{int(hea[-1, 0])}, ale ok. {ps._num(round(ratio, -1))} razy więcej ewaluacji obwodu.",
              key=["UCCSD", "HEA"])
    ps.footer(fig, f"{BASE_NOTE} Błąd = największa wartość na krzywej; dla HEA mediana z 5 "
                   "losowych startów. CNOT po transpilacji do bramek natywnych IBM.")
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
