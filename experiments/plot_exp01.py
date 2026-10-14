"""
figure for experiment 01 (H2 potential energy curve), drawn from the saved CSV.

plots are separate from experiments: change the figure without re-running the physics.

run (from repo root):  python experiments/plot_exp01.py
input:   results/exp01_h2_wyniki.csv  (energies in Ha)
output:  results/fig_exp01_h2.pdf, results/fig_exp01_h2.png
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import plotstyle as ps  # noqa: E402

FLOOR_MHA = 1e-10   # errors below this are at machine precision; drawn on the floor


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    ps.use()
    d = load("results/exp01_h2_wyniki.csv")
    R = d["R"]

    # errors vs FCI in mHa (the CSV stores Ha)
    err = {
        "HF": np.abs(d["HF"] - d["FCI"]) * 1e3,
        "UCCSD": np.maximum(d["dE_UCCSD"] * 1e3, FLOOR_MHA),
        "HEA": np.maximum(d["dE_HEA"] * 1e3, FLOOR_MHA),
    }
    i_eq = int(np.argmin(d["FCI"]))
    hf_gap = err["HF"][-1]
    hea_margin = ps.CHEM_ACC_MHA / err["HEA"].max()

    fig, (ax1, ax2) = ps.figure(nrows=2, height_ratios=[1.25, 1], sharex=True,
                                panel_height=2.4)

    # ---- panel 1: energy curve ----
    ps.plot(ax1, R, d["HF"], "HF")
    ps.plot(ax1, R, d["FCI"], "FCI", zorder=3)
    ps.plot(ax1, R, d["UCCSD"], "UCCSD", ls="none", markersize=5.5, zorder=4)
    ps.plot(ax1, R, d["HEA"], "HEA", ls="none", markersize=3.2, zorder=5)
    ps.panel_title(ax1, "Energia całkowita", "energia stanu podstawowego [Ha] w funkcji długości wiązania")
    ps.number_axis(ax1, "y", decimals=1)
    ax1.set_ylim(-1.28, -0.55)

    ps.end_label(ax1, R[-1], d["HF"][-1], "Hartree–Fock")
    ps.end_label(ax1, R[-1], d["FCI"][-1], "FCI = VQE")
    ps.bracket(ax1, R[-1] - 0.12, d["FCI"][-1], d["HF"][-1],
               f"błąd HF\n{ps._num(round(hf_gap))} mHa", side="left")
    ps.callout(ax1, (R[i_eq], d["FCI"][i_eq]),
               f"minimum: długość wiązania\nR ≈ {ps._num(R[i_eq], 2)} Å",
               xytext=(1.0, -1.225))
    ps.callout(ax1, (2.4, d["FCI"][-6]),
               "rozerwane wiązanie:\ndwa osobne atomy H",
               xytext=(2.4, -1.09), ha="center")

    # ---- panel 2: error vs FCI, log scale ----
    ps.log_axis(ax2, "y", every=2)
    ax2.set_ylim(FLOOR_MHA * 0.4, 3e3)
    ps.threshold(ax2, ps.CHEM_ACC_MHA, "↓ dokładność chemiczna: błąd < 1,6 mHa")
    for m in ["HF", "HEA", "UCCSD"]:
        ps.plot(ax2, R, err[m], m)
    ps.panel_title(ax2, "Błąd względem FCI", "|E − E_FCI| [mHa], skala logarytmiczna")

    ps.end_label(ax2, R[-1], err["HF"][-1], "Hartree–Fock")
    ps.end_label(ax2, R[-1], err["HEA"][-1], "VQE-HEA")
    ps.end_label(ax2, R[-1], err["UCCSD"][-1], "VQE-UCCSD")
    ps.callout(ax2, (1.2, err["HEA"].max()),
               f"najgorszy punkt HEA: ok. {ps._num(round(hea_margin, -2))}×\nponiżej progu",
               xytext=(1.6, 8e-3))

    ax2.set_xlabel("Długość wiązania R [Å]")
    ps.number_axis(ax2, "x", decimals=1)
    ax2.set_xlim(R[0] - 0.05, R[-1] + 0.05)

    ps.header(fig, "Oba modele VQE odtwarzają dokładną krzywą energii H₂",
              "Hartree–Fock myli się coraz bardziej przy rozciąganiu wiązania; "
              "błąd VQE pozostaje tysiące razy poniżej progu",
              key=["FCI", "HF", "UCCSD", "HEA"])
    ps.footer(fig, "H₂, baza STO-3G, symulator wektora stanu (bez szumu), 19 geometrii, HEA: 2 warstwy, 1 ziarno. "
                   "Błędy < 10⁻¹⁰ mHa (precyzja maszynowa) narysowano na dolnej krawędzi.")

    ps.save(fig, "results/fig_exp01_h2")


if __name__ == "__main__":
    main()
