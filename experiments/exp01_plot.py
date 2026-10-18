"""
figure for experiment 01 (H2 potential energy curve), drawn from the saved CSV.

plots are separate from experiments: change the figure without re-running the physics.

run (from repo root):  python experiments/exp01_plot.py
input:   results/exp01_h2_results.csv
output:  results/fig_exp01_h2.pdf, results/fig_exp01_h2.png
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import plotstyle as ps  # noqa: E402

FLOOR_MHA = 1e-13   # errors below this are at machine precision; drawn on the floor


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    ps.use()
    raw = load("results/exp01_h2_results.csv")
    R = raw["R_A"]
    d = {m: raw[f"E_{m}_Ha"] for m in ("HF", "FCI", "UCCSD", "HEA")}

    # errors vs FCI in mHa
    err = {
        "HF": np.abs(d["HF"] - d["FCI"]) * 1e3,
        "UCCSD": np.maximum(raw["dE_UCCSD_mHa"], FLOOR_MHA),
        "HEA": np.maximum(raw["dE_HEA_mHa"], FLOOR_MHA),
    }
    fig, (ax1, ax2) = ps.figure(nrows=2, height_ratios=[1.25, 1], sharex=True,
                                panel_height=2.4)

    # ---- panel 1: energy curve ----
    ps.plot(ax1, R, d["HF"], "HF")
    ps.plot(ax1, R, d["FCI"], "FCI", zorder=3)
    ps.plot(ax1, R, d["UCCSD"], "UCCSD", ls="none", markersize=5.5, zorder=4)
    ps.plot(ax1, R, d["HEA"], "HEA", ls="none", markersize=3.2, zorder=5)
    ps.panel_title(ax1, "Energy", "Ha")
    ps.number_axis(ax1, "y", decimals=1)
    ax1.set_ylim(-1.2, -0.55)

    # ---- panel 2: error vs FCI, log scale ----
    ps.log_axis(ax2, "y", every=3)
    ax2.set_ylim(FLOOR_MHA * 0.4, 3e3)
    ps.threshold(ax2, ps.CHEM_ACC_MHA, "chemical accuracy")
    for m in ["HF", "HEA", "UCCSD"]:
        ps.plot(ax2, R, err[m], m)
    ps.panel_title(ax2, "Error vs FCI", "mHa, log scale")

    ax2.set_xlabel("Bond length R (Å)")
    ps.number_axis(ax2, "x", decimals=1)
    ax2.set_xlim(R[0] - 0.05, R[-1] + 0.05)

    ps.header(fig, "Potential energy curve of H₂: HF, FCI and VQE", key=["FCI", "HF", "UCCSD", "HEA"])
    ps.footer(fig, "H₂, STO-3G, noiseless statevector. HEA: 2 layers. "
                   "Errors below 10⁻¹³ mHa are drawn at the floor.")

    ps.end_labels(ax1, R[-1], [(d["HF"][-1], "HF"), (d["FCI"][-1], "FCI = VQE")])
    e_h, e_u = err["HEA"][-1], err["UCCSD"][-1]
    vqe = ([(np.sqrt(e_h * e_u), "VQE")] if abs(np.log10(e_h / e_u)) < 1.5
           else [(e_h, "HEA"), (e_u, "UCCSD")])
    ps.end_labels(ax2, R[-1], [(err["HF"][-1], "HF")] + vqe)

    ps.save(fig, "results/fig_exp01_h2")


if __name__ == "__main__":
    main()
